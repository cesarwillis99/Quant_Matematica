# -*- coding: utf-8 -*-
"""
================================================================================
ou_backtest.py -- Backtest da Estrategia Ornstein-Uhlenbeck -- EURUSD H1
================================================================================
Autor: Quant Developer Senior
Data: 2026

Descricao:
    Simula historicamente as operacoes geradas pelo modulo ou_strategy.py,
    calculando metricas de performance robustas e gerando equity curve.

    IMPLEMENTACAO 100% MANUAL -- sem bibliotecas de backtesting externas.
    Modulo completamente independente dos demais backtests do projeto.

Mecanica de simulacao:
    - Iteracao candle a candle sobre todos os dados H1
    - 1 posicao aberta por vez
    - Spread fixo de 0.5 pips por operacao
    - Capital inicial: 10.000 USD
    - Risco por operacao: 1% do capital atual (position sizing dinamico)
    - Entrada no Open do candle SEGUINTE ao sinal
    - Saida por: TP em pips, SL em pips, retorno a zona neutra Z_ou [-0.3, +0.3]
    - Saida adicional: fechamento forcado sexta 21h UTC

Regras de fim de semana:
    - Fechamento forcado na sexta-feira as 21h00 UTC
    - Bloqueio de novas entradas apos sexta-feira as 20h00 UTC
    - Reabertura a partir de domingo as 21h00 UTC

Arquivos de entrada:
    - data/eurusd_h1_ou.parquet

Arquivos de saida:
    - graficos/ou_equity.png
    - resultados/ou_operacoes.csv
================================================================================
"""

import logging
import warnings
import math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

warnings.filterwarnings("ignore", category=UserWarning)
matplotlib.use("Agg")

# =============================================================================
# CONFIGURACAO DE LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# CAMINHOS
# =============================================================================
DIR_MODULO     = Path(__file__).resolve().parent
DIR_DATA       = DIR_MODULO / "data"
DIR_GRAFICOS   = DIR_MODULO / "graficos"
DIR_RESULTADOS = DIR_MODULO / "resultados"

PARQUET_OU       = DIR_DATA / "eurusd_h1_ou.parquet"
GRAFICO_EQUITY   = DIR_GRAFICOS / "ou_equity.png"
CSV_OPERACOES    = DIR_RESULTADOS / "ou_operacoes.csv"

# =============================================================================
# PARAMETROS DO BACKTEST
# =============================================================================
CAPITAL_INICIAL    = 10_000.0   # USD
RISCO_POR_TRADE    = 0.01       # 1% do capital atual por operacao
SPREAD_PIPS        = 0.5        # Spread fixo por operacao (mais apertado para OU)
VALOR_PIP_POR_LOTE = 10.0       # USD por pip por lote padrao (EURUSD)
FATOR_PIPS         = 10_000     # Conversao preco -> pips

# Saida por retorno a zona neutra do Z-Score OU
# Mais agressivo que o Z-Score de preco ([-0.5, +0.5])
Z_OU_SAIDA_NEUTRO_MIN = -0.3
Z_OU_SAIDA_NEUTRO_MAX =  0.3

# =============================================================================
# PALETA DE CORES (dark mode premium)
# =============================================================================
COR_FUNDO      = "#0D1117"
COR_TEXTO      = "#E6EDF3"
COR_GRADE      = "#21262D"
COR_OU         = "#D2A8FF"   # Lilas (mesma cor do Z-Score OU no ou_strategy.py)
COR_DD_OU      = "#8B5FBF"   # Lilas escuro para drawdown
COR_POSITIVO   = "#3FB950"   # Verde
COR_NEGATIVO   = "#F85149"   # Vermelho
COR_REFERENCIA = "#484F58"   # Cinza medio

# =============================================================================
# PERIODOS HISTORICOS PARA ANALISE
# (adaptados para a serie disponivel 2016-2026)
# =============================================================================
PERIODOS = {
    "Brexit (2016-2017)":              ("2016-01-01", "2017-12-31"),
    "Estavel (2018-2019)":             ("2018-01-01", "2019-12-31"),
    "COVID (2020)":                    ("2020-01-01", "2020-12-31"),
    "Inflacao / Fed (2021-2023)":      ("2021-01-01", "2023-12-31"),
    "Recente (2024-2026)":             ("2024-01-01", "2026-12-31"),
}


# =============================================================================
# ESTRUTURA DE DADOS
# =============================================================================

@dataclass
class Operacao:
    """Representa uma operacao completa (entrada + saida)."""
    id:              int
    estrategia:      str
    direcao:         int            # +1=LONG | -1=SHORT
    entrada_dt:      pd.Timestamp
    entrada_preco:   float
    saida_dt:        Optional[pd.Timestamp] = None
    saida_preco:     Optional[float]         = None
    motivo_saida:    str                      = ""
    sl_preco:        float                    = 0.0
    tp_preco:        float                    = 0.0
    sl_pips:         float                    = 0.0
    tp_pips:         float                    = 0.0
    lot_size:        float                    = 0.0
    capital_entrada: float                    = 0.0
    pnl_pips:        float                    = 0.0
    pnl_monetario:   float                    = 0.0
    duracao_candles: int                      = 0


# =============================================================================
# POSITION SIZING E PnL
# =============================================================================

def calcular_lot_size(capital: float, sl_pips: float) -> float:
    """
    Calcula o tamanho do lote com base no risco percentual do capital.

    risco_monetario = capital * 0.01
    lot_size = risco_monetario / (sl_pips * 10)

    Limitado entre 0.01 (minimo) e 100.0 (maximo de seguranca).
    """
    if sl_pips <= 0 or not math.isfinite(sl_pips):
        return 0.01
    risco = capital * RISCO_POR_TRADE
    lot   = risco / (sl_pips * VALOR_PIP_POR_LOTE)
    return float(np.clip(lot, 0.01, 100.0))


def calcular_pnl(
    direcao: int,
    entrada: float,
    saida:   float,
    lot:     float,
) -> Tuple[float, float]:
    """
    Calcula o PnL em pips e em USD (liquido de spread).

    LONG : PnL_pips = (saida - entrada) * 10000
    SHORT: PnL_pips = (entrada - saida) * 10000
    PnL_USD = PnL_pips * lot * 10 - spread * lot * 10
    """
    pnl_pips     = direcao * (saida - entrada) * FATOR_PIPS
    pnl_bruto    = pnl_pips * lot * VALOR_PIP_POR_LOTE
    custo_spread = SPREAD_PIPS * lot * VALOR_PIP_POR_LOTE
    return float(pnl_pips), float(pnl_bruto - custo_spread)


# =============================================================================
# MECANISMO DE SAIDA POR CANDLE
# =============================================================================

def verificar_saida_candle(
    direcao:     int,
    high:        float,
    low:         float,
    close:       float,
    sl_preco:    float,
    tp_preco:    float,
    zscore_ou:   Optional[float] = None,
) -> Tuple[bool, float, str]:
    """
    Verifica se a posicao deve ser fechada no candle atual.

    Ordem de prioridade:
        1. Stop Loss (via Low/High) -- pior caso
        2. Take Profit (via High/Low)
        3. Retorno a zona neutra do Z-Score OU ([-0.3, +0.3])
    """
    if direcao == 1:   # LONG
        # SL e TP no mesmo candle: assume SL primeiro (pior caso)
        if low <= sl_preco and high >= tp_preco:
            return True, sl_preco, "SL"
        if low <= sl_preco:
            return True, sl_preco, "SL"
        if high >= tp_preco:
            return True, tp_preco, "TP_PIPS"
        # Saida por retorno a zona neutra do Z_ou
        if zscore_ou is not None and not math.isnan(zscore_ou):
            if Z_OU_SAIDA_NEUTRO_MIN <= zscore_ou <= Z_OU_SAIDA_NEUTRO_MAX:
                return True, close, "Z_OU_NEUTRO"

    elif direcao == -1:   # SHORT
        if high >= sl_preco and low <= tp_preco:
            return True, sl_preco, "SL"
        if high >= sl_preco:
            return True, sl_preco, "SL"
        if low <= tp_preco:
            return True, tp_preco, "TP_PIPS"
        if zscore_ou is not None and not math.isnan(zscore_ou):
            if Z_OU_SAIDA_NEUTRO_MIN <= zscore_ou <= Z_OU_SAIDA_NEUTRO_MAX:
                return True, close, "Z_OU_NEUTRO"

    return False, 0.0, ""


# =============================================================================
# SIMULADOR PRINCIPAL
# =============================================================================

def simular_estrategia_ou(
    df:             pd.DataFrame,
    capital_inicio: float = CAPITAL_INICIAL,
) -> Tuple[List[Operacao], float]:
    """
    Executa a simulacao historica candle a candle da estrategia OU.

    Mecanica:
        - 1 posicao aberta por vez
        - Position sizing dinamico: 1% do capital / SL em pips
        - Entrada no Open do candle SEGUINTE ao sinal
        - Saida por TP, SL, retorno a Z_ou neutro ou fechamento de fim de semana
        - Fechamento forcado na sexta-feira as 21h UTC
        - Bloqueio de entradas a partir de sexta 20h UTC

    Parametros:
        df: pd.DataFrame com colunas do ou_strategy.py
        capital_inicio: float -- capital inicial em USD

    Retorna:
        Tuple[List[Operacao], float] -- lista de operacoes e capital final
    """
    logger.info(f"Iniciando simulacao OU | Capital: ${capital_inicio:,.2f}")

    operacoes: List[Operacao] = []
    capital   = capital_inicio
    posicao   = None
    candle_entrada = 0
    id_op     = 0

    # Arrays numpy para iteracao rapida
    indices   = df.index
    highs     = df["High"].values.astype(np.float64)
    lows      = df["Low"].values.astype(np.float64)
    closes    = df["Close"].values.astype(np.float64)
    opens     = df["Open"].values.astype(np.float64)
    sinais    = df["sinal_ou"].values.astype(np.int8)
    sl_pips_v = df["sl_pips"].values.astype(np.float64)
    tp_pips_v = df["tp_pips"].values.astype(np.float64)
    zscores_v = df["ou_zscore"].values.astype(np.float64)

    n = len(indices)

    for i in range(n):
        dt_atual   = indices[i]
        dia_semana = dt_atual.weekday()   # 0=Seg ... 4=Sex, 5=Sab, 6=Dom
        hora       = dt_atual.hour
        high_i     = highs[i]
        low_i      = lows[i]
        close_i    = closes[i]
        sinal_i    = sinais[i]
        zscore_i   = float(zscores_v[i])

        # Regras de fim de semana
        eh_sexta_21h      = (dia_semana == 4) and (hora == 21)
        bloqueio_entrada  = (
            ((dia_semana == 4) and (hora >= 21)) or   # Sexta >= 21h
            (dia_semana == 5) or                       # Sabado
            ((dia_semana == 6) and (hora < 21))        # Domingo antes das 21h
        )

        # -- Gerenciar posicao aberta ----------------------------------------
        if posicao is not None:

            # Fechamento forcado sexta 21h UTC
            if eh_sexta_21h:
                pnl_p, pnl_m = calcular_pnl(
                    posicao.direcao, posicao.entrada_preco, close_i, posicao.lot_size
                )
                posicao.saida_dt        = dt_atual
                posicao.saida_preco     = close_i
                posicao.motivo_saida    = "FIM_SEMANA"
                posicao.pnl_pips        = pnl_p
                posicao.pnl_monetario   = pnl_m
                posicao.duracao_candles = i - candle_entrada
                capital += pnl_m
                operacoes.append(posicao)
                posicao = None

            else:
                deve, preco_s, motivo = verificar_saida_candle(
                    posicao.direcao, high_i, low_i, close_i,
                    posicao.sl_preco, posicao.tp_preco,
                    zscore_i,
                )
                if deve:
                    pnl_p, pnl_m = calcular_pnl(
                        posicao.direcao, posicao.entrada_preco, preco_s, posicao.lot_size
                    )
                    posicao.saida_dt        = dt_atual
                    posicao.saida_preco     = preco_s
                    posicao.motivo_saida    = motivo
                    posicao.pnl_pips        = pnl_p
                    posicao.pnl_monetario   = pnl_m
                    posicao.duracao_candles = i - candle_entrada
                    capital += pnl_m
                    operacoes.append(posicao)
                    posicao = None

        # -- Verificar entrada ------------------------------------------------
        if posicao is None and sinal_i != 0 and not bloqueio_entrada:
            if i + 1 >= n:
                continue  # sem proximo candle, ignorar sinal

            sl_p = sl_pips_v[i]
            tp_p = tp_pips_v[i]

            if not (math.isfinite(sl_p) and sl_p > 0 and
                    math.isfinite(tp_p) and tp_p > 0):
                continue

            lot      = calcular_lot_size(capital, sl_p)
            delta_sl = sl_p / FATOR_PIPS
            delta_tp = tp_p / FATOR_PIPS

            # Entrada no Open do candle SEGUINTE ao sinal
            preco_entrada = opens[i + 1]

            if sinal_i == 1:     # LONG
                sl_abs = preco_entrada - delta_sl
                tp_abs = preco_entrada + delta_tp
            else:                # SHORT
                sl_abs = preco_entrada + delta_sl
                tp_abs = preco_entrada - delta_tp

            id_op += 1
            posicao = Operacao(
                id=id_op, estrategia="OU", direcao=int(sinal_i),
                entrada_dt=dt_atual, entrada_preco=preco_entrada,
                sl_preco=sl_abs, tp_preco=tp_abs,
                sl_pips=sl_p, tp_pips=tp_p,
                lot_size=lot, capital_entrada=capital,
            )
            candle_entrada = i

    # Fechar posicao aberta ao fim dos dados
    if posicao is not None:
        pnl_p, pnl_m = calcular_pnl(
            posicao.direcao, posicao.entrada_preco, closes[-1], posicao.lot_size
        )
        posicao.saida_dt        = indices[-1]
        posicao.saida_preco     = closes[-1]
        posicao.motivo_saida    = "FIM_DADOS"
        posicao.pnl_pips        = pnl_p
        posicao.pnl_monetario   = pnl_m
        posicao.duracao_candles = n - 1 - candle_entrada
        capital += pnl_m
        operacoes.append(posicao)

    logger.info(
        f"Simulacao OU concluida: {len(operacoes)} operacoes | "
        f"Capital final: ${capital:,.2f}"
    )
    return operacoes, capital


# =============================================================================
# EQUITY CURVE
# =============================================================================

def construir_equity_curve(
    operacoes:       List[Operacao],
    df_index:        pd.DatetimeIndex,
    capital_inicial: float = CAPITAL_INICIAL,
) -> pd.Series:
    """Constroi a curva de patrimonio ao longo do tempo."""
    equity = pd.Series(capital_inicial, index=df_index, dtype=np.float64)
    ops_ordenadas = sorted(
        [op for op in operacoes if op.saida_dt is not None],
        key=lambda op: op.saida_dt,
    )
    capital_acum = capital_inicial
    for op in ops_ordenadas:
        capital_acum += op.pnl_monetario
        equity[op.saida_dt:] = capital_acum
    return equity


# =============================================================================
# METRICAS DE PERFORMANCE (13 metricas obrigatorias)
# =============================================================================

def calcular_metricas(
    operacoes:       List[Operacao],
    equity_curve:    pd.Series,
    capital_inicial: float = CAPITAL_INICIAL,
    nome:            str = "OU",
) -> dict:
    """
    Calcula as 13 metricas obrigatorias de performance.

    1.  Total de Operacoes
    2.  Win Rate
    3.  Media de Ganho
    4.  Media de Perda
    5.  Fator de Lucro
    6.  Payoff Ratio
    7.  PnL Total (USD e %)
    8.  Drawdown Maximo
    9.  Fator de Recuperacao
    10. Sharpe Ratio (anualizado, base 252)
    11. Expectancia por Operacao
    12. Periodo Testado
    13. Percentual do Tempo em Posicao
    """
    m: dict = {"nome": nome}

    if not operacoes:
        m.update({
            "total_operacoes": 0, "win_rate": 0.0,
            "media_ganho": 0.0, "media_perda": 0.0,
            "fator_lucro": 0.0, "payoff_ratio": 0.0,
            "pnl_total_usd": 0.0, "pnl_total_pct": 0.0,
            "drawdown_max_usd": 0.0, "drawdown_max_pct": 0.0,
            "fator_recuperacao": 0.0, "sharpe_ratio": 0.0,
            "expectancia": 0.0, "periodo_inicio": "N/A",
            "periodo_fim": "N/A", "pct_tempo_posicao": 0.0,
        })
        return m

    pnls   = np.array([op.pnl_monetario for op in operacoes])
    ganhos = pnls[pnls > 0]
    perdas = pnls[pnls <= 0]

    # 1. Total de operacoes
    m["total_operacoes"] = len(operacoes)

    # 2. Win Rate
    m["win_rate"] = (len(ganhos) / len(operacoes)) * 100

    # 3. Media de Ganho
    m["media_ganho"] = float(np.mean(ganhos)) if len(ganhos) > 0 else 0.0

    # 4. Media de Perda
    m["media_perda"] = float(np.mean(perdas)) if len(perdas) > 0 else 0.0

    # 5. Fator de Lucro
    soma_g = float(np.sum(ganhos)) if len(ganhos) > 0 else 0.0
    soma_p = float(np.sum(perdas)) if len(perdas) > 0 else -1e-9
    m["fator_lucro"] = soma_g / abs(soma_p) if soma_p != 0 else float("inf")

    # 6. Payoff Ratio
    m["payoff_ratio"] = (
        m["media_ganho"] / abs(m["media_perda"])
        if m["media_perda"] != 0 else float("inf")
    )

    # 7. PnL Total
    m["pnl_total_usd"] = float(np.sum(pnls))
    m["pnl_total_pct"] = (m["pnl_total_usd"] / capital_inicial) * 100

    # 8. Drawdown Maximo
    pico       = equity_curve.cummax()
    dd_abs     = equity_curve - pico
    dd_pct     = dd_abs / pico
    m["drawdown_max_usd"] = float(dd_abs.min())
    m["drawdown_max_pct"] = float(dd_pct.min()) * 100

    # 9. Fator de Recuperacao
    m["fator_recuperacao"] = (
        m["pnl_total_usd"] / abs(m["drawdown_max_usd"])
        if m["drawdown_max_usd"] != 0 else float("inf")
    )

    # 10. Sharpe Ratio (anualizado)
    equity_daily     = equity_curve.resample("1D").last().dropna()
    retornos_diarios = equity_daily.pct_change().dropna()
    if len(retornos_diarios) >= 2 and retornos_diarios.std() > 0:
        m["sharpe_ratio"] = (
            retornos_diarios.mean() / retornos_diarios.std()
        ) * np.sqrt(252)
    else:
        m["sharpe_ratio"] = 0.0

    # 11. Expectancia por operacao
    wr = m["win_rate"] / 100
    m["expectancia"] = (wr * m["media_ganho"]) + ((1 - wr) * m["media_perda"])

    # 12. Periodo testado
    m["periodo_inicio"] = str(operacoes[0].entrada_dt.date())
    m["periodo_fim"]    = str(
        operacoes[-1].saida_dt.date()
        if operacoes[-1].saida_dt else "em_aberto"
    )

    # 13. Percentual do tempo em posicao
    total_candles_pos = sum(op.duracao_candles for op in operacoes)
    m["pct_tempo_posicao"] = (
        total_candles_pos / len(equity_curve) * 100
        if len(equity_curve) > 0 else 0.0
    )

    return m


def exibir_metricas(m: dict) -> None:
    """Exibe as 13 metricas formatadas no terminal."""
    print(f"\n{'='*62}")
    print(f"  METRICAS DE PERFORMANCE -- {m['nome']}")
    print(f"{'='*62}")
    print(f"  1.  Periodo testado          : {m['periodo_inicio']} -> {m['periodo_fim']}")
    print(f"  2.  Total de Operacoes       : {m['total_operacoes']:>8,}")
    print(f"  3.  Win Rate                 : {m['win_rate']:>7.2f}%")
    print(f"  4.  Media de Ganho           : ${m['media_ganho']:>10.2f}")
    print(f"  5.  Media de Perda           : ${m['media_perda']:>10.2f}")
    print(f"  6.  Fator de Lucro           : {m['fator_lucro']:>8.3f}x")
    print(f"  7.  Payoff Ratio             : {m['payoff_ratio']:>8.3f}x")
    print(f"  8.  PnL Total                : ${m['pnl_total_usd']:>+10.2f} ({m['pnl_total_pct']:+.2f}%)")
    print(f"  9.  Drawdown Maximo          : {m['drawdown_max_pct']:>8.2f}% (${m['drawdown_max_usd']:,.2f})")
    print(f"  10. Fator de Recuperacao     : {m['fator_recuperacao']:>8.3f}x")
    print(f"  11. Sharpe Ratio (anual)     : {m['sharpe_ratio']:>8.4f}")
    print(f"  12. Expectancia/Operacao     : ${m['expectancia']:>+10.2f}")
    print(f"  13. Tempo em Posicao         : {m['pct_tempo_posicao']:>7.2f}%")
    print(f"{'='*62}")


# =============================================================================
# ANALISE POR PERIODO HISTORICO
# =============================================================================

def analise_por_periodo(
    df:        pd.DataFrame,
    operacoes: List[Operacao],
) -> pd.DataFrame:
    """
    Analisa a performance da estrategia OU em cada periodo macroeconomico.

    Periodos:
        - Brexit (2016-2017)
        - Estavel (2018-2019)
        - COVID (2020)
        - Inflacao / Fed (2021-2023)
        - Recente (2024-2026)
    """
    print(f"\n{'='*70}")
    print(f"  ANALISE POR PERIODO HISTORICO -- ROBUSTEZ MACROECONOMICA (OU)")
    print(f"{'='*70}")

    registros = []

    for nome_periodo, (inicio, fim) in PERIODOS.items():
        dt_inicio = pd.Timestamp(inicio)
        dt_fim    = pd.Timestamp(fim)

        # Verificar se ha dados disponiveis neste periodo
        if dt_fim < df.index.min() or dt_inicio > df.index.max():
            print(f"\n  Periodo: {nome_periodo}")
            print(f"  {'-'*60}")
            print(f"    Sem dados na serie (cobertura: {df.index.min().date()} -> {df.index.max().date()})")
            continue

        print(f"\n  Periodo: {nome_periodo}")
        print(f"  {'-'*60}")

        df_periodo = df[(df.index >= dt_inicio) & (df.index <= dt_fim)]
        if df_periodo.empty:
            print("    (sem candles neste intervalo)")
            continue

        ops_p = [
            op for op in operacoes
            if dt_inicio <= op.entrada_dt <= dt_fim
        ]

        if not ops_p:
            print(f"    OU: sem operacoes no periodo")
            continue

        eq_p = construir_equity_curve(ops_p, df_periodo.index)
        m    = calcular_metricas(ops_p, eq_p, CAPITAL_INICIAL, "OU")

        print(
            f"    OU: {m['total_operacoes']:3d} ops | "
            f"WR={m['win_rate']:5.1f}% | PnL={m['pnl_total_pct']:+6.2f}% | "
            f"DD={m['drawdown_max_pct']:5.2f}% | Sharpe={m['sharpe_ratio']:+.3f}"
        )

        registros.append({
            "periodo":     nome_periodo,
            "estrategia":  "OU",
            "operacoes":   m["total_operacoes"],
            "win_rate":    round(m["win_rate"], 2),
            "pnl_pct":     round(m["pnl_total_pct"], 2),
            "pnl_usd":     round(m["pnl_total_usd"], 2),
            "drawdown":    round(m["drawdown_max_pct"], 2),
            "sharpe":      round(m["sharpe_ratio"], 4),
            "fator_lucro": round(m["fator_lucro"], 3),
        })

    return pd.DataFrame(registros)


# =============================================================================
# ANALISE DE MOTIVOS DE SAIDA
# =============================================================================

def analise_motivos_saida(operacoes: List[Operacao]) -> None:
    """
    Exibe a distribuicao dos motivos de saida das operacoes.
    Util para entender o comportamento da estrategia:
    - Quanto % sai por TP, SL, Z_ou neutro ou fim de semana?
    """
    if not operacoes:
        return

    motivos = {}
    for op in operacoes:
        motivos[op.motivo_saida] = motivos.get(op.motivo_saida, 0) + 1

    total = len(operacoes)

    print(f"\n{'='*62}")
    print(f"  DISTRIBUICAO DE MOTIVOS DE SAIDA -- OU")
    print(f"{'='*62}")
    for motivo, contagem in sorted(motivos.items(), key=lambda x: -x[1]):
        pct = contagem / total * 100
        # PnL medio por motivo
        pnl_motivo = [op.pnl_monetario for op in operacoes if op.motivo_saida == motivo]
        pnl_medio = np.mean(pnl_motivo) if pnl_motivo else 0.0
        print(f"  {motivo:<16}: {contagem:>5,} ops ({pct:>5.1f}%) | PnL medio: ${pnl_medio:>+8.2f}")
    print(f"{'='*62}")


# =============================================================================
# GRAFICO: EQUITY CURVE OU
# =============================================================================

def gerar_grafico_equity_ou(
    equity_curve:  pd.Series,
    operacoes:     List[Operacao],
    metricas:      dict,
    caminho:       Path,
) -> None:
    """
    Gera grafico premium de equity curve para a estrategia Ornstein-Uhlenbeck.

    Layout: 3 paineis
        Painel 1 (topo):    Crescimento de capital com area preenchida
        Painel 2 (meio):    Drawdown ao longo do tempo
        Painel 3 (rodape):  Scorecard de metricas
    """
    logger.info("Gerando grafico de equity curve OU...")

    plt.rcParams.update({
        "figure.facecolor":  COR_FUNDO,
        "axes.facecolor":    COR_FUNDO,
        "axes.edgecolor":    COR_GRADE,
        "axes.labelcolor":   COR_TEXTO,
        "xtick.color":       COR_TEXTO,
        "ytick.color":       COR_TEXTO,
        "text.color":        COR_TEXTO,
        "grid.color":        COR_GRADE,
        "font.family":       "monospace",
    })

    fig = plt.figure(figsize=(22, 16))
    gs  = gridspec.GridSpec(
        3, 1,
        height_ratios=[3.5, 1.5, 1.2],
        hspace=0.06,
    )
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2])

    cap_final  = equity_curve.iloc[-1]
    ret_total  = (cap_final / CAPITAL_INICIAL - 1) * 100
    n_ops      = len(operacoes)

    fig.suptitle(
        "EURUSD H1 -- Equity Curve Ornstein-Uhlenbeck  |  "
        f"Capital: $10.000  |  Risco: 1%/trade  |  Spread: {SPREAD_PIPS} pips",
        color=COR_TEXTO, fontsize=13, fontweight="bold", y=0.995,
    )

    # -- Painel 1: Crescimento de capital ------------------------------------
    ax1.axhline(
        CAPITAL_INICIAL, color=COR_REFERENCIA, linestyle=":",
        linewidth=1.0, alpha=0.6, label=f"Capital Inicial (${CAPITAL_INICIAL:,.0f})",
        zorder=1,
    )

    # Area preenchida: verde se acima do capital, vermelho se abaixo
    ax1.fill_between(
        equity_curve.index, equity_curve.values, CAPITAL_INICIAL,
        where=(equity_curve.values >= CAPITAL_INICIAL),
        color=COR_OU, alpha=0.08, zorder=2,
    )
    ax1.fill_between(
        equity_curve.index, equity_curve.values, CAPITAL_INICIAL,
        where=(equity_curve.values < CAPITAL_INICIAL),
        color=COR_NEGATIVO, alpha=0.08, zorder=2,
    )

    ax1.plot(
        equity_curve.index, equity_curve.values,
        color=COR_OU, linewidth=1.8, alpha=0.95, zorder=4,
        label=f"OU   Capital Final: ${cap_final:,.2f}  ({ret_total:+.1f}%)  |  {n_ops} ops",
    )

    # Ponto final
    ax1.scatter(
        [equity_curve.index[-1]], [cap_final],
        color=COR_OU, s=80, zorder=5, edgecolors="white", linewidths=0.8,
    )

    ax1.set_ylabel("Capital (USD)", color=COR_TEXTO, fontsize=12)
    ax1.legend(loc="upper left", fontsize=10, framealpha=0.3, handlelength=2)
    ax1.grid(True, linestyle="--", alpha=0.2)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.setp(ax1.get_xticklabels(), visible=False)

    # -- Painel 2: Drawdown --------------------------------------------------
    pico   = equity_curve.cummax()
    dd_pct = ((equity_curve - pico) / pico) * 100
    ax2.fill_between(equity_curve.index, dd_pct, 0, color=COR_DD_OU, alpha=0.18, zorder=2)
    ax2.plot(
        equity_curve.index, dd_pct, color=COR_OU, linewidth=0.9, alpha=0.85, zorder=3,
        label=f"OU  DD max: {dd_pct.min():.1f}%",
    )
    ax2.set_ylabel("Drawdown (%)", color=COR_TEXTO, fontsize=10)
    ax2.axhline(0, color=COR_REFERENCIA, linewidth=0.7, alpha=0.5)
    ax2.legend(loc="lower left", fontsize=9, framealpha=0.25)
    ax2.grid(True, linestyle="--", alpha=0.25)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    plt.setp(ax2.get_xticklabels(), visible=False)

    # -- Painel 3: Scorecard de metricas (tabela visual) ----------------------
    ax3.set_axis_off()
    ax3.set_facecolor(COR_FUNDO)

    anos  = max((equity_curve.index.max() - equity_curve.index.min()).days / 365.25, 1)

    cabec = [
        "Estrategia", "Ops", "Win Rate", "PnL %", "Sharpe",
        "Max DD", "Fator Lucro", "Expect./Op", "Ops/Ano",
    ]

    m = metricas
    linhas = [[
        "OU",
        str(m["total_operacoes"]),
        f"{m['win_rate']:.2f}%",
        f"{m['pnl_total_pct']:+.2f}%",
        f"{m['sharpe_ratio']:+.4f}",
        f"{m['drawdown_max_pct']:.2f}%",
        f"{m['fator_lucro']:.3f}x",
        f"${m['expectancia']:+.2f}",
        f"{m['total_operacoes'] / anos:.1f}",
    ]]

    tabela = ax3.table(
        cellText=linhas,
        colLabels=cabec,
        cellLoc="center",
        loc="center",
        bbox=[0, 0.2, 1, 0.6],
    )
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(10)

    # Estilizacao da tabela
    for (row, col), cell in tabela.get_celld().items():
        cell.set_edgecolor(COR_GRADE)
        cell.set_facecolor(COR_FUNDO)
        cell.set_text_props(color=COR_TEXTO)
        if row == 0:
            cell.set_facecolor("#161B22")
            cell.set_text_props(color=COR_TEXTO, fontweight="bold")
        elif row > 0 and col == 0:
            cell.set_text_props(color=COR_OU, fontweight="bold")
        elif row > 0 and col == 3:
            val_txt = linhas[row - 1][3]
            cor_pnl = COR_POSITIVO if not val_txt.startswith("-") else COR_NEGATIVO
            cell.set_text_props(color=cor_pnl, fontweight="bold")

    fig.autofmt_xdate(rotation=28, ha="right")
    plt.savefig(caminho, dpi=160, bbox_inches="tight", facecolor=COR_FUNDO)
    plt.close()
    logger.info(f"Grafico equity OU salvo: {caminho.name} ({caminho.stat().st_size / 1024:.0f} KB)")


# =============================================================================
# SALVAMENTO DE RESULTADOS
# =============================================================================

def salvar_resultados_ou(
    operacoes: List[Operacao],
) -> None:
    """Salva ou_operacoes.csv com todas as operacoes da estrategia OU."""
    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)

    registros = []
    for op in operacoes:
        registros.append({
            "id":              op.id,
            "estrategia":      op.estrategia,
            "direcao":         "LONG" if op.direcao == 1 else "SHORT",
            "entrada_dt":      op.entrada_dt,
            "entrada_preco":   round(op.entrada_preco, 5),
            "saida_dt":        op.saida_dt,
            "saida_preco":     round(op.saida_preco, 5) if op.saida_preco else None,
            "motivo_saida":    op.motivo_saida,
            "sl_pips":         round(op.sl_pips, 1),
            "tp_pips":         round(op.tp_pips, 1),
            "lot_size":        round(op.lot_size, 4),
            "pnl_pips":        round(op.pnl_pips, 1),
            "pnl_monetario":   round(op.pnl_monetario, 2),
            "duracao_candles": op.duracao_candles,
            "capital_entrada": round(op.capital_entrada, 2),
        })

    pd.DataFrame(registros).to_csv(CSV_OPERACOES, index=False)
    logger.info(f"ou_operacoes.csv salvo: {len(registros)} operacoes")


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def executar_backtest_ou(
    data_inicio: str = None,
    data_fim:    str = None,
) -> None:
    """
    Pipeline completo do backtest Ornstein-Uhlenbeck.

    Sequencia:
        1. Carregar dados OU (eurusd_h1_ou.parquet)
        2. Aplicar filtros de periodo (opcional)
        3. Simular estrategia OU
        4. Construir equity curve
        5. Calcular 13 metricas
        6. Analise de motivos de saida
        7. Analise por periodo historico
        8. Gerar grafico de equity curve
        9. Salvar CSV de operacoes
        10. Resumo executivo final
    """
    DIR_GRAFICOS.mkdir(parents=True, exist_ok=True)
    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)

    # -- 1. Carregar dados OU ------------------------------------------------
    if not PARQUET_OU.exists():
        raise FileNotFoundError(
            f"\n{'='*60}\n"
            f"  ERRO: {PARQUET_OU.name} nao encontrado!\n"
            f"  Execute o ou_strategy.py primeiro:\n"
            f"    python ou_strategy.py\n"
            f"{'='*60}\n"
        )

    logger.info(f"Carregando dados OU: {PARQUET_OU.name}")
    df = pd.read_parquet(PARQUET_OU, engine="pyarrow")
    logger.info(f"Dados carregados: {len(df):,} candles H1")

    # -- 2. Filtros de periodo -----------------------------------------------
    if data_inicio:
        df = df[df.index >= data_inicio]
    if data_fim:
        df = df[df.index <= data_fim]

    logger.info(
        f"Periodo de backtest: {df.index.min().date()} -> {df.index.max().date()} "
        f"({len(df):,} candles)"
    )

    n_sinais = (df["sinal_ou"] != 0).sum()
    logger.info(f"Sinais OU disponiveis: {n_sinais}")

    # -- 3. Simular estrategia OU --------------------------------------------
    print(f"\n{'='*62}")
    print(f"  SIMULANDO ESTRATEGIA: ORNSTEIN-UHLENBECK")
    print(f"{'='*62}")
    ops_ou, capital_final = simular_estrategia_ou(df)

    # -- 4. Construir equity curve -------------------------------------------
    eq_ou = construir_equity_curve(ops_ou, df.index)

    # -- 5. Calcular metricas ------------------------------------------------
    metricas_ou = calcular_metricas(ops_ou, eq_ou, CAPITAL_INICIAL, "OU")
    exibir_metricas(metricas_ou)

    # -- 6. Analise de motivos de saida --------------------------------------
    analise_motivos_saida(ops_ou)

    # -- 7. Analise por periodo historico ------------------------------------
    df_periodos = analise_por_periodo(df, ops_ou)

    # -- 8. Gerar grafico de equity curve ------------------------------------
    gerar_grafico_equity_ou(eq_ou, ops_ou, metricas_ou, GRAFICO_EQUITY)

    # -- 9. Salvar CSV -------------------------------------------------------
    salvar_resultados_ou(ops_ou)

    # -- 10. Resumo executivo final ------------------------------------------
    anos = max((df.index.max() - df.index.min()).days / 365.25, 1)
    m = metricas_ou

    print(f"\n{'='*76}")
    print(f"  RESUMO EXECUTIVO -- ORNSTEIN-UHLENBECK | "
          f"{df.index.min().date()} -> {df.index.max().date()}")
    print(f"{'='*76}")
    print(f"  {'Metrica':<22} | {'OU':>12}")
    print(f"  {'-'*40}")
    print(f"  {'Total Operacoes':<22} | {m['total_operacoes']:>12,}")
    print(f"  {'Win Rate':<22} | {m['win_rate']:>11.2f}%")
    print(f"  {'PnL % (10 anos)':<22} | {m['pnl_total_pct']:>+11.2f}%")
    print(f"  {'Sharpe Ratio':<22} | {m['sharpe_ratio']:>+11.4f}")
    print(f"  {'Drawdown Maximo':<22} | {m['drawdown_max_pct']:>11.2f}%")
    print(f"  {'Fator de Lucro':<22} | {m['fator_lucro']:>11.3f}x")
    print(f"  {'Expectancia/op':<22} | ${m['expectancia']:>+10.2f}")
    print(f"  {'Ops/ano':<22} | {m['total_operacoes']/anos:>11.1f}")
    print(f"{'='*76}")
    print(f"\n  Arquivos gerados:")
    print(f"    -> graficos/ou_equity.png")
    print(f"    -> resultados/ou_operacoes.csv")
    print(f"{'='*76}\n")


# =============================================================================
# EXECUCAO DIRETA
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Backtest manual -- Estrategia Ornstein-Uhlenbeck -- EURUSD H1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python ou_backtest.py
  python ou_backtest.py --inicio 2018-01-01 --fim 2023-12-31
        """,
    )
    parser.add_argument("--inicio", type=str, default=None, help="Data de inicio (YYYY-MM-DD)")
    parser.add_argument("--fim",    type=str, default=None, help="Data de fim (YYYY-MM-DD)")
    args = parser.parse_args()

    print("\n" + "=" * 62)
    print("  QUANT EURUSD -- BACKTEST ORNSTEIN-UHLENBECK")
    print(f"  Capital: $10.000 | Risco: 1%/trade | Spread: {SPREAD_PIPS} pips")
    print("=" * 62)

    executar_backtest_ou(data_inicio=args.inicio, data_fim=args.fim)
