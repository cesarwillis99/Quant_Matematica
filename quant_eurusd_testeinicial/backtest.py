# -*- coding: utf-8 -*-
"""
================================================================================
backtest.py — Simulação Histórica de Estratégias Quantitativas EURUSD H1
================================================================================
Autor: Quant Developer Sênior
Data: 2026

Descrição:
    Simula historicamente as operações geradas pelos módulos zscore.py e
    momentum.py de forma separada e combinada, calculando métricas de
    performance robustas e gerando equity curves.

    IMPLEMENTAÇÃO 100% MANUAL — sem bibliotecas de backtesting externas
    (backtrader, zipline, etc.). Controle total sobre cada operação.

Mecânica de simulação:
    - Iteração candle a candle sobre todos os dados H1
    - 1 posição aberta por vez por estratégia
    - Spread fixo de 1.2 pips por operação
    - Capital inicial: 10.000 USD
    - Risco por operação: 1% do capital atual (position sizing dinâmico)
    - Saída por preço (High/Low do candle): SL e TP em pips
    - Saída adicional para ZScore: retorno à zona neutra do Z

Regras de fim de semana:
    - Fechamento forçado na sexta-feira às 21h00 UTC
    - Bloqueio de novas entradas após sexta-feira às 20h00 UTC
    - Reabertura a partir de domingo às 21h00 UTC

Arquivos de entrada:
    - data/eurusd_h1_zscore.parquet
    - data/eurusd_h1_momentum.parquet

Arquivos de saída:
    - graficos/equity_curves.png          (gráfico técnico com drawdown)
    - graficos/equity_curves_capital.png  (gráfico de crescimento de capital)
    - resultados/operacoes.csv
    - resultados/metricas.csv
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
# CONFIGURAÇÃO DE LOGGING
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

PARQUET_ZSCORE   = DIR_DATA / "eurusd_h1_zscore.parquet"
PARQUET_MOMENTUM = DIR_DATA / "eurusd_h1_momentum.parquet"

GRAFICO_EQUITY         = DIR_GRAFICOS / "equity_curves.png"
GRAFICO_CAPITAL        = DIR_GRAFICOS / "equity_curves_capital.png"
CSV_OPERACOES          = DIR_RESULTADOS / "operacoes.csv"
CSV_METRICAS           = DIR_RESULTADOS / "metricas.csv"

# =============================================================================
# PARÂMETROS DO BACKTEST
# =============================================================================
CAPITAL_INICIAL    = 10_000.0   # USD
RISCO_POR_TRADE    = 0.01       # 1% do capital atual por operação
SPREAD_PIPS        = 1.2        # Spread fixo por operação
VALOR_PIP_POR_LOTE = 10.0       # USD por pip por lote padrão (EURUSD)
FATOR_PIPS         = 10_000     # Conversão preço → pips

# Saída por retorno à zona neutra do Z-Score
Z_SAIDA_NEUTRO_MIN = -0.5
Z_SAIDA_NEUTRO_MAX =  0.5

# =============================================================================
# PALETA DE CORES (dark mode premium)
# =============================================================================
COR_FUNDO      = "#0D1117"
COR_TEXTO      = "#E6EDF3"
COR_GRADE      = "#21262D"
COR_ZSCORE     = "#A371F7"   # Roxo
COR_MOMENTUM   = "#F0A500"   # Laranja/dourado
COR_COMBINADA  = "#58A6FF"   # Azul celeste
COR_DD_Z       = "#7B4FBF"
COR_DD_M       = "#B07800"
COR_DD_C       = "#2E6FBF"
COR_POSITIVO   = "#3FB950"   # Verde
COR_NEGATIVO   = "#F85149"   # Vermelho
COR_REFERENCIA = "#484F58"   # Cinza médio

# =============================================================================
# PERÍODOS HISTÓRICOS PARA ANÁLISE
# (a série disponível cobre 2016-2026; períodos anteriores são ignorados)
# =============================================================================
PERIODOS = {
    "Crise de 2008 (2007-2009)":          ("2007-01-01", "2009-12-31"),
    "Crise do Euro (2010-2012)":           ("2010-01-01", "2012-12-31"),
    "Baixa Volatilidade (2013-2019)":      ("2013-01-01", "2019-12-31"),
    "COVID (2020)":                        ("2020-01-01", "2020-12-31"),
    "Alta Inflação / Fed (2021-2023)":     ("2021-01-01", "2023-12-31"),
    "Período Recente (2024-2026)":         ("2024-01-01", "2026-12-31"),
}


# =============================================================================
# ESTRUTURA DE DADOS
# =============================================================================

@dataclass
class Operacao:
    """Representa uma operação completa (entrada + saída)."""
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

    Limitado entre 0.01 (mínimo) e 100.0 (máximo de segurança).
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
    Calcula o PnL em pips e em USD (líquido de spread).

    LONG : PnL_pips = (saida - entrada) * 10000
    SHORT: PnL_pips = (entrada - saida) * 10000
    PnL_USD = PnL_pips * lot * 10 - spread * lot * 10
    """
    pnl_pips     = direcao * (saida - entrada) * FATOR_PIPS
    pnl_bruto    = pnl_pips * lot * VALOR_PIP_POR_LOTE
    custo_spread = SPREAD_PIPS * lot * VALOR_PIP_POR_LOTE
    return float(pnl_pips), float(pnl_bruto - custo_spread)


# =============================================================================
# MECANISMO DE SAÍDA POR CANDLE
# =============================================================================

def verificar_saida_candle(
    direcao:     int,
    high:        float,
    low:         float,
    close:       float,
    sl_preco:    float,
    tp_preco:    float,
    zscore:      Optional[float] = None,
    usar_zscore: bool = False,
) -> Tuple[bool, float, str]:
    """
    Verifica se a posição deve ser fechada no candle atual.

    Ordem de prioridade:
        1. Stop Loss (via Low/High) → pior caso
        2. Take Profit (via High/Low)
        3. Retorno à zona neutra do Z-Score (apenas ZScore strategy)
    """
    if direcao == 1:   # LONG
        if low <= sl_preco and high >= tp_preco:
            return True, sl_preco, "SL"
        if low <= sl_preco:
            return True, sl_preco, "SL"
        if high >= tp_preco:
            return True, tp_preco, "TP_PIPS"
        if usar_zscore and zscore is not None and not math.isnan(zscore):
            if Z_SAIDA_NEUTRO_MIN <= zscore <= Z_SAIDA_NEUTRO_MAX:
                return True, close, "ZSCORE_NEUTRO"

    elif direcao == -1:   # SHORT
        if high >= sl_preco and low <= tp_preco:
            return True, sl_preco, "SL"
        if high >= sl_preco:
            return True, sl_preco, "SL"
        if low <= tp_preco:
            return True, tp_preco, "TP_PIPS"
        if usar_zscore and zscore is not None and not math.isnan(zscore):
            if Z_SAIDA_NEUTRO_MIN <= zscore <= Z_SAIDA_NEUTRO_MAX:
                return True, close, "ZSCORE_NEUTRO"

    return False, 0.0, ""


# =============================================================================
# SIMULADOR PRINCIPAL
# =============================================================================

def simular_estrategia(
    df:               pd.DataFrame,
    coluna_sinal:     str,
    nome:             str,
    usar_zscore_exit: bool = False,
    capital_inicio:   float = CAPITAL_INICIAL,
    coluna_sinal_origem: str = None,
) -> Tuple[List[Operacao], float]:
    """
    Executa a simulação histórica candle a candle.

    Mecânica:
        - 1 posição aberta por vez
        - Position sizing dinâmico: 1% do capital / SL em pips
        - Fechamento forçado na sexta-feira às 21h UTC
        - Bloqueio de entradas a partir de sexta 20h UTC
    """
    logger.info(f"Iniciando simulação: {nome} | Capital: ${capital_inicio:,.2f}")

    operacoes: List[Operacao] = []
    capital   = capital_inicio
    posicao   = None
    candle_entrada = 0
    id_op     = 0

    # Arrays numpy para iteração rápida
    indices   = df.index
    highs     = df["High"].values.astype(np.float64)
    lows      = df["Low"].values.astype(np.float64)
    closes    = df["Close"].values.astype(np.float64)
    opens     = df["Open"].values.astype(np.float64)
    sinais    = df[coluna_sinal].values.astype(np.int8)
    sl_pips_v = df["sl_pips"].values.astype(np.float64)
    tp_pips_v = df["tp_pips"].values.astype(np.float64)

    sinais_z_origem = df["sinal_zscore"].values.astype(np.int8) if "sinal_zscore" in df.columns else None

    zscores_v = None
    if (usar_zscore_exit or coluna_sinal_origem is not None) and "zscore" in df.columns:
        zscores_v = df["zscore"].values.astype(np.float64)

    origem_zscore = False

    n = len(indices)

    for i in range(n):
        dt_atual   = indices[i]
        dia_semana = dt_atual.weekday()   # 0=Seg … 4=Sex, 5=Sab, 6=Dom
        hora       = dt_atual.hour
        high_i     = highs[i]
        low_i      = lows[i]
        close_i    = closes[i]
        sinal_i    = sinais[i]
        zscore_i   = float(zscores_v[i]) if zscores_v is not None else None

        # Regras de fim de semana
        eh_sexta_21h      = (dia_semana == 4) and (hora == 21)
        bloqueio_entrada  = (
            ((dia_semana == 4) and (hora >= 21)) or   # Sexta ≥ 21h
            (dia_semana == 5) or                       # Sábado
            ((dia_semana == 6) and (hora < 21))        # Domingo antes das 21h
        )

        # ── Gerenciar posição aberta ──────────────────────────────────────────
        if posicao is not None:

            # Fechamento forçado sexta 21h UTC
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
                usar_z_saida = usar_zscore_exit or (
                    coluna_sinal_origem is not None and origem_zscore
                )
                deve, preco_s, motivo = verificar_saida_candle(
                    posicao.direcao, high_i, low_i, close_i,
                    posicao.sl_preco, posicao.tp_preco,
                    zscore_i, usar_z_saida,
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

        # ── Verificar entrada ─────────────────────────────────────────────────
        if posicao is None and sinal_i != 0 and not bloqueio_entrada:
            if i + 1 >= n:
                continue  # sem próximo candle, ignorar sinal

            sl_p = sl_pips_v[i]
            tp_p = tp_pips_v[i]

            if not (math.isfinite(sl_p) and sl_p > 0 and
                    math.isfinite(tp_p) and tp_p > 0):
                continue

            lot      = calcular_lot_size(capital, sl_p)
            delta_sl = sl_p / FATOR_PIPS
            delta_tp = tp_p / FATOR_PIPS

            preco_entrada = opens[i + 1]

            if sinal_i == 1:     # LONG
                sl_abs = preco_entrada - delta_sl
                tp_abs = preco_entrada + delta_tp
            else:                # SHORT
                sl_abs = preco_entrada + delta_sl
                tp_abs = preco_entrada - delta_tp

            id_op += 1
            origem_zscore = (sinais_z_origem[i] != 0) if sinais_z_origem is not None else False
            posicao = Operacao(
                id=id_op, estrategia=nome, direcao=int(sinal_i),
                entrada_dt=dt_atual, entrada_preco=preco_entrada,
                sl_preco=sl_abs, tp_preco=tp_abs,
                sl_pips=sl_p, tp_pips=tp_p,
                lot_size=lot, capital_entrada=capital,
            )
            candle_entrada = i

    # Fechar posição aberta ao fim dos dados
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
        f"Simulação {nome} concluída: {len(operacoes)} operações | "
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
    """Constrói a curva de patrimônio ao longo do tempo."""
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
# MÉTRICAS DE PERFORMANCE (13 métricas obrigatórias)
# =============================================================================

def calcular_metricas(
    operacoes:       List[Operacao],
    equity_curve:    pd.Series,
    capital_inicial: float = CAPITAL_INICIAL,
    nome:            str = "",
) -> dict:
    """
    Calcula as 13 métricas obrigatórias de performance.

    1.  Total de Operações
    2.  Win Rate
    3.  Média de Ganho
    4.  Média de Perda
    5.  Fator de Lucro
    6.  Payoff Ratio
    7.  PnL Total (USD e %)
    8.  Drawdown Máximo
    9.  Fator de Recuperação
    10. Sharpe Ratio (anualizado, base 252)
    11. Expectância por Operação
    12. Período Testado
    13. Percentual do Tempo em Posição
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

    # 1. Total de operações
    m["total_operacoes"] = len(operacoes)

    # 2. Win Rate
    m["win_rate"] = (len(ganhos) / len(operacoes)) * 100

    # 3. Média de Ganho
    m["media_ganho"] = float(np.mean(ganhos)) if len(ganhos) > 0 else 0.0

    # 4. Média de Perda
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

    # 8. Drawdown Máximo
    pico       = equity_curve.cummax()
    dd_abs     = equity_curve - pico
    dd_pct     = dd_abs / pico
    m["drawdown_max_usd"] = float(dd_abs.min())
    m["drawdown_max_pct"] = float(dd_pct.min()) * 100

    # 9. Fator de Recuperação
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

    # 11. Expectância por operação
    wr = m["win_rate"] / 100
    m["expectancia"] = (wr * m["media_ganho"]) + ((1 - wr) * m["media_perda"])

    # 12. Período testado
    m["periodo_inicio"] = str(operacoes[0].entrada_dt.date())
    m["periodo_fim"]    = str(
        operacoes[-1].saida_dt.date()
        if operacoes[-1].saida_dt else "em_aberto"
    )

    # 13. Percentual do tempo em posição
    total_candles_pos = sum(op.duracao_candles for op in operacoes)
    m["pct_tempo_posicao"] = (
        total_candles_pos / len(equity_curve) * 100
        if len(equity_curve) > 0 else 0.0
    )

    return m


def exibir_metricas(m: dict) -> None:
    """Exibe as 13 métricas formatadas no terminal."""
    print(f"\n{'='*62}")
    print(f"  MÉTRICAS DE PERFORMANCE — {m['nome']}")
    print(f"{'='*62}")
    print(f"  1.  Período testado          : {m['periodo_inicio']} → {m['periodo_fim']}")
    print(f"  2.  Total de Operações       : {m['total_operacoes']:>8,}")
    print(f"  3.  Win Rate                 : {m['win_rate']:>7.2f}%")
    print(f"  4.  Média de Ganho           : ${m['media_ganho']:>10.2f}")
    print(f"  5.  Média de Perda           : ${m['media_perda']:>10.2f}")
    print(f"  6.  Fator de Lucro           : {m['fator_lucro']:>8.3f}x")
    print(f"  7.  Payoff Ratio             : {m['payoff_ratio']:>8.3f}x")
    print(f"  8.  PnL Total                : ${m['pnl_total_usd']:>+10.2f} ({m['pnl_total_pct']:+.2f}%)")
    print(f"  9.  Drawdown Máximo          : {m['drawdown_max_pct']:>8.2f}% (${m['drawdown_max_usd']:,.2f})")
    print(f"  10. Fator de Recuperação     : {m['fator_recuperacao']:>8.3f}x")
    print(f"  11. Sharpe Ratio (anual)     : {m['sharpe_ratio']:>8.4f}")
    print(f"  12. Expectância/Operação     : ${m['expectancia']:>+10.2f}")
    print(f"  13. Tempo em Posição         : {m['pct_tempo_posicao']:>7.2f}%")
    print(f"{'='*62}")


# =============================================================================
# ANÁLISE POR PERÍODO HISTÓRICO
# =============================================================================

def analise_por_periodo(
    df:              pd.DataFrame,
    operacoes_dict:  Dict[str, List[Operacao]],
) -> pd.DataFrame:
    """
    Analisa a performance em cada período macroeconômico definido.

    Períodos analisados:
        - Crise de 2008 (2007-2009)
        - Crise do Euro (2010-2012)
        - Baixa Volatilidade (2013-2019)
        - COVID (2020)
        - Alta Inflação / Fed (2021-2023)
        - Período Recente (2024-2026)

    Para períodos sem dados disponíveis na série (antes de 2016),
    exibe aviso e segue para o próximo.
    """
    print(f"\n{'='*70}")
    print(f"  ANÁLISE POR PERÍODO HISTÓRICO — ROBUSTEZ MACROECONÔMICA")
    print(f"{'='*70}")

    registros = []

    for nome_periodo, (inicio, fim) in PERIODOS.items():
        dt_inicio = pd.Timestamp(inicio)
        dt_fim    = pd.Timestamp(fim)

        # Verificar se há dados disponíveis neste período
        if dt_fim < df.index.min() or dt_inicio > df.index.max():
            print(f"\n  Período: {nome_periodo}")
            print(f"  {'─'*60}")
            print(f"    ⚠  Sem dados na série (cobertura: {df.index.min().date()} → {df.index.max().date()})")
            continue

        print(f"\n  Período: {nome_periodo}")
        print(f"  {'─'*60}")

        df_periodo = df[(df.index >= dt_inicio) & (df.index <= dt_fim)]
        if df_periodo.empty:
            print("    (sem candles neste intervalo)")
            continue

        for nome_estrat, operacoes in operacoes_dict.items():
            ops_p = [
                op for op in operacoes
                if dt_inicio <= op.entrada_dt <= dt_fim
            ]

            if not ops_p:
                print(f"    {nome_estrat:<12}: sem operações no período")
                continue

            eq_p = construir_equity_curve(ops_p, df_periodo.index)
            m    = calcular_metricas(ops_p, eq_p, CAPITAL_INICIAL, nome_estrat)

            print(
                f"    {nome_estrat:<12}: {m['total_operacoes']:3d} ops | "
                f"WR={m['win_rate']:5.1f}% | PnL={m['pnl_total_pct']:+6.2f}% | "
                f"DD={m['drawdown_max_pct']:5.2f}% | Sharpe={m['sharpe_ratio']:+.3f}"
            )

            registros.append({
                "periodo":     nome_periodo,
                "estrategia":  nome_estrat,
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
# GRÁFICO 1: EQUITY CURVES TÉCNICAS COM DRAWDOWN
# =============================================================================

def gerar_grafico_equity(
    equity_curves: Dict[str, pd.Series],
    operacoes_dict: Dict[str, List[Operacao]],
    caminho: Path,
) -> None:
    """
    Gráfico técnico profissional com equity curves e drawdown.

    Layout: 2 painéis (equity + drawdown compartilhando eixo X).
    Inclui anotações de capital final e retorno percentual.
    """
    logger.info("Gerando gráfico 1: equity curves técnico com drawdown...")

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

    mapa = {
        "ZSCORE":    (COR_ZSCORE,    COR_DD_Z),
        "MOMENTUM":  (COR_MOMENTUM,  COR_DD_M),
        "COMBINADA": (COR_COMBINADA, COR_DD_C),
    }

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(22, 13), sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1.3], "hspace": 0.04},
    )

    # Cabeçalho
    fig.suptitle(
        "EURUSD H1 — Equity Curves e Drawdown  |  Capital: $10.000  |  Risco: 1%/trade  |  Spread: 1.2 pips",
        color=COR_TEXTO, fontsize=13, fontweight="bold", y=0.995,
    )

    # Linha de referência do capital inicial
    ax1.axhline(
        CAPITAL_INICIAL, color=COR_REFERENCIA, linestyle=":",
        linewidth=1.0, alpha=0.6, label=f"Capital Inicial (${CAPITAL_INICIAL:,.0f})",
        zorder=1,
    )

    for nome, eq in equity_curves.items():
        cor, _ = mapa.get(nome, (COR_TEXTO, COR_TEXTO))
        cap_final  = eq.iloc[-1]
        ret_pct    = (cap_final / CAPITAL_INICIAL - 1) * 100
        n_ops      = len(operacoes_dict.get(nome, []))
        ax1.plot(
            eq.index, eq.values, color=cor, linewidth=1.4, alpha=0.92, zorder=3,
            label=f"{nome}  →  ${cap_final:,.2f}  ({ret_pct:+.1f}%)  |  {n_ops} ops",
        )

    ax1.set_ylabel("Capital (USD)", color=COR_TEXTO, fontsize=11)
    ax1.legend(loc="upper left", fontsize=9.5, framealpha=0.25, handlelength=2)
    ax1.grid(True, linestyle="--", alpha=0.25, zorder=0)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.setp(ax1.get_xticklabels(), visible=False)

    # Drawdown
    for nome, eq in equity_curves.items():
        cor, cor_dd = mapa.get(nome, (COR_TEXTO, COR_TEXTO))
        pico   = eq.cummax()
        dd_pct = ((eq - pico) / pico) * 100
        ax2.fill_between(eq.index, dd_pct, 0, color=cor_dd, alpha=0.18, zorder=2)
        ax2.plot(
            eq.index, dd_pct, color=cor, linewidth=0.9, alpha=0.85, zorder=3,
            label=f"{nome}  DD máx: {dd_pct.min():.1f}%",
        )

    ax2.set_ylabel("Drawdown (%)", color=COR_TEXTO, fontsize=10)
    ax2.set_xlabel("Data", color=COR_TEXTO, fontsize=10)
    ax2.axhline(0, color=COR_REFERENCIA, linewidth=0.7, alpha=0.5)
    ax2.legend(loc="lower left", fontsize=9, framealpha=0.25)
    ax2.grid(True, linestyle="--", alpha=0.25)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))

    fig.autofmt_xdate(rotation=28, ha="right")
    plt.savefig(caminho, dpi=160, bbox_inches="tight", facecolor=COR_FUNDO)
    plt.close()
    logger.info(f"Gráfico 1 salvo: {caminho.name} ({caminho.stat().st_size / 1024:.0f} KB)")


# =============================================================================
# GRÁFICO 2: CRESCIMENTO DE CAPITAL (VISUALIZAÇÃO PREMIUM)
# =============================================================================

def gerar_grafico_capital(
    equity_curves: Dict[str, pd.Series],
    todas_metricas: Dict[str, dict],
    caminho: Path,
) -> None:
    """
    Segundo gráfico de capital: visualização premium focada no crescimento.

    Layout de 3 painéis:
        Painel 1 (topo):    Crescimento do capital em escala linear com área preenchida
        Painel 2 (meio):    Retorno acumulado em % ao longo do tempo
        Painel 3 (rodapé):  Tabela de scorecard de métricas principais
    """
    logger.info("Gerando gráfico 2: crescimento de capital premium...")

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

    mapa_cores = {
        "ZSCORE":    COR_ZSCORE,
        "MOMENTUM":  COR_MOMENTUM,
        "COMBINADA": COR_COMBINADA,
    }

    fig = plt.figure(figsize=(22, 16))
    gs  = gridspec.GridSpec(
        3, 1,
        height_ratios=[3.5, 2.0, 1.2],
        hspace=0.06,
    )
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2])

    fig.suptitle(
        "EURUSD H1 — Crescimento de Capital  |  2016–2026  |  Capital Inicial: $10.000  |  Risco: 1%/trade",
        color=COR_TEXTO, fontsize=13, fontweight="bold", y=0.995,
    )

    # ── Painel 1: Crescimento de capital (área preenchida) ────────────────────
    ax1.axhline(
        CAPITAL_INICIAL, color=COR_REFERENCIA, linestyle="--",
        linewidth=1.0, alpha=0.5, zorder=1, label="Capital Inicial",
    )

    for nome, eq in equity_curves.items():
        cor       = mapa_cores.get(nome, COR_TEXTO)
        cap_final = eq.iloc[-1]
        ret_total = (cap_final / CAPITAL_INICIAL - 1) * 100

        # Área preenchida: verde se acima do inicial, vermelho se abaixo
        ax1.fill_between(
            eq.index, eq.values, CAPITAL_INICIAL,
            where=(eq.values >= CAPITAL_INICIAL),
            color=cor, alpha=0.08, zorder=2,
        )
        ax1.fill_between(
            eq.index, eq.values, CAPITAL_INICIAL,
            where=(eq.values < CAPITAL_INICIAL),
            color=COR_NEGATIVO, alpha=0.08, zorder=2,
        )
        ax1.plot(
            eq.index, eq.values, color=cor, linewidth=1.8, alpha=0.95, zorder=4,
            label=f"{nome}   Capital Final: ${cap_final:,.2f}  ({ret_total:+.1f}%)",
        )

        # Anotação do capital final (ponto + rótulo)
        ax1.scatter(
            [eq.index[-1]], [cap_final],
            color=cor, s=80, zorder=5, edgecolors="white", linewidths=0.8,
        )

    ax1.set_ylabel("Capital (USD)", color=COR_TEXTO, fontsize=12)
    ax1.legend(loc="upper left", fontsize=10, framealpha=0.3, handlelength=2)
    ax1.grid(True, linestyle="--", alpha=0.2)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.setp(ax1.get_xticklabels(), visible=False)

    # ── Painel 2: Retorno acumulado em % ─────────────────────────────────────
    for nome, eq in equity_curves.items():
        cor        = mapa_cores.get(nome, COR_TEXTO)
        ret_acum   = ((eq / CAPITAL_INICIAL) - 1) * 100

        ax2.fill_between(
            eq.index, ret_acum, 0,
            where=(ret_acum >= 0), color=cor, alpha=0.07, zorder=2,
        )
        ax2.fill_between(
            eq.index, ret_acum, 0,
            where=(ret_acum < 0), color=COR_NEGATIVO, alpha=0.07, zorder=2,
        )
        ax2.plot(
            eq.index, ret_acum, color=cor, linewidth=1.3, alpha=0.9, zorder=3,
            label=f"{nome}",
        )

    ax2.axhline(0, color=COR_REFERENCIA, linewidth=1.0, alpha=0.6, linestyle="--")
    ax2.set_ylabel("Retorno Acumulado (%)", color=COR_TEXTO, fontsize=11)
    ax2.legend(loc="upper left", fontsize=9, framealpha=0.3)
    ax2.grid(True, linestyle="--", alpha=0.2)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+.1f}%"))
    plt.setp(ax2.get_xticklabels(), visible=False)

    # ── Painel 3: Scorecard de métricas (tabela visual) ───────────────────────
    ax3.set_axis_off()
    ax3.set_facecolor(COR_FUNDO)

    anos  = 10.25
    cabec = ["Estratégia", "Ops", "Win Rate", "PnL %", "Sharpe", "Max DD", "Fator Lucro", "Expect./Op", "Ops/Ano"]
    linhas = []
    for nome, m in todas_metricas.items():
        linhas.append([
            nome,
            str(m["total_operacoes"]),
            f"{m['win_rate']:.2f}%",
            f"{m['pnl_total_pct']:+.2f}%",
            f"{m['sharpe_ratio']:+.4f}",
            f"{m['drawdown_max_pct']:.2f}%",
            f"{m['fator_lucro']:.3f}x",
            f"${m['expectancia']:+.2f}",
            f"{m['total_operacoes'] / anos:.1f}",
        ])

    tabela = ax3.table(
        cellText=linhas,
        colLabels=cabec,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(9.5)

    # Estilização da tabela
    cores_estrat = {"ZSCORE": COR_ZSCORE, "MOMENTUM": COR_MOMENTUM, "COMBINADA": COR_COMBINADA}
    for (row, col), cell in tabela.get_celld().items():
        cell.set_edgecolor(COR_GRADE)
        cell.set_facecolor(COR_FUNDO)
        cell.set_text_props(color=COR_TEXTO)
        if row == 0:
            # Cabeçalho
            cell.set_facecolor("#161B22")
            cell.set_text_props(color=COR_TEXTO, fontweight="bold")
        elif row > 0 and col == 0:
            # Nome da estratégia com cor dedicada
            nome_lin = linhas[row - 1][0]
            cell.set_text_props(color=cores_estrat.get(nome_lin, COR_TEXTO), fontweight="bold")
        elif row > 0 and col == 3:
            # PnL: verde se positivo, vermelho se negativo
            val_txt = linhas[row - 1][3]
            cor_pnl = COR_POSITIVO if not val_txt.startswith("-") else COR_NEGATIVO
            cell.set_text_props(color=cor_pnl, fontweight="bold")

    fig.autofmt_xdate(rotation=28, ha="right")
    plt.savefig(caminho, dpi=160, bbox_inches="tight", facecolor=COR_FUNDO)
    plt.close()
    logger.info(f"Gráfico 2 salvo: {caminho.name} ({caminho.stat().st_size / 1024:.0f} KB)")


# =============================================================================
# SALVAMENTO DE RESULTADOS
# =============================================================================

def salvar_resultados(
    todas_operacoes: Dict[str, List[Operacao]],
    todas_metricas:  Dict[str, dict],
) -> None:
    """Salva operacoes.csv e metricas.csv."""
    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)

    # operacoes.csv
    registros = []
    for nome, ops in todas_operacoes.items():
        for op in ops:
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
    logger.info(f"operacoes.csv salvo: {len(registros)} linhas")

    # metricas.csv
    pd.DataFrame(list(todas_metricas.values())).to_csv(CSV_METRICAS, index=False)
    logger.info("metricas.csv salvo")


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def executar_backtest(
    data_inicio: str = None,
    data_fim:    str = None,
) -> None:
    """
    Pipeline completo do backtest.

    Sequência:
        1. Carregar e mesclar parquets de ZScore e Momentum
        2. Aplicar filtros de período (opcional)
        3. Montar sinal combinado (ZScore tem prioridade)
        4. Simular ZScore  → ops_z
        5. Simular Momentum → ops_m
        6. Simular Combinada → ops_c
        7. Construir equity curves
        8. Calcular 13 métricas por estratégia
        9. Análise por período histórico
        10. Gráfico 1: equity curves técnico com drawdown
        11. Gráfico 2: crescimento de capital premium
        12. Salvar CSVs
        13. Resumo executivo final
    """
    DIR_GRAFICOS.mkdir(parents=True, exist_ok=True)
    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)

    # ── 1. Carregar dados ─────────────────────────────────────────────────────
    for p in [PARQUET_ZSCORE, PARQUET_MOMENTUM]:
        if not p.exists():
            raise FileNotFoundError(
                f"\n{'='*60}\n  ERRO: {p.name} não encontrado!\n"
                f"  Execute o módulo correspondente primeiro.\n{'='*60}\n"
            )

    logger.info("Carregando parquets de entrada...")
    df_z = pd.read_parquet(PARQUET_ZSCORE,   engine="pyarrow")
    df_m = pd.read_parquet(PARQUET_MOMENTUM, engine="pyarrow")

    df = df_z.copy()
    df["sinal_momentum"] = df_m["sinal_momentum"].reindex(df.index, fill_value=0)
    for col in ["velocidade", "aceleracao", "percentil_acel", "entropia_shannon"]:
        if col in df_m.columns:
            df[col] = df_m[col].reindex(df.index)

    logger.info(f"Dados mesclados: {len(df):,} candles H1")

    # ── 2. Filtros de período ─────────────────────────────────────────────────
    if data_inicio:
        df = df[df.index >= data_inicio]
    if data_fim:
        df = df[df.index <= data_fim]

    logger.info(
        f"Período de backtest: {df.index.min().date()} → {df.index.max().date()} "
        f"({len(df):,} candles)"
    )

    # ── 3. Montar sinal combinado (ZScore tem prioridade) ─────────────────────
    sinal_comb = df["sinal_zscore"].copy()
    mask_m = (sinal_comb == 0) & (df["sinal_momentum"] != 0)
    sinal_comb[mask_m] = df["sinal_momentum"][mask_m]
    df["sinal_combinado"] = sinal_comb.astype("int8")

    n_z = (df["sinal_zscore"] != 0).sum()
    n_m = (df["sinal_momentum"] != 0).sum()
    n_c = (df["sinal_combinado"] != 0).sum()
    logger.info(f"Sinais disponíveis: ZScore={n_z} | Momentum={n_m} | Combinado={n_c}")

    # ── 4–6. Simulações ───────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  SIMULANDO ESTRATEGIA: Z-SCORE")
    print(f"{'='*62}")
    # Para o Z-Score usamos o df padrão, que tem sl_pips e tp_pips do zscore.py
    ops_z, _ = simular_estrategia(df, "sinal_zscore", "ZSCORE", usar_zscore_exit=True)

    print(f"\n{'='*62}")
    print(f"  SIMULANDO ESTRATEGIA: MOMENTUM")
    print(f"{'='*62}")
    # Para o Momentum, carregamos os limites de stop e take profit do eurusd_h1_momentum.parquet
    df_m_sim = df.copy()
    df_m_sim["sl_pips"] = df_m["sl_pips"].reindex(df.index)
    df_m_sim["tp_pips"] = df_m["tp_pips"].reindex(df.index)
    ops_m, _ = simular_estrategia(df_m_sim, "sinal_momentum", "MOMENTUM", usar_zscore_exit=False)

    print(f"\n{'='*62}")
    print(f"  SIMULANDO ESTRATEGIA: COMBINADA (ZScore + Momentum)")
    print(f"{'='*62}")
    # Para a Combinada, selecionamos o sl/tp dependendo de qual sinal disparou a entrada
    df_c_sim = df.copy()
    sl_c = df_z["sl_pips"].copy()
    tp_c = df_z["tp_pips"].copy()
    
    # Se o sinal do Z-Score for nulo e o do Momentum for ativo, usamos as colunas do Momentum
    mask_c_m = (df["sinal_zscore"] == 0) & (df["sinal_momentum"] != 0)
    sl_c[mask_c_m] = df_m["sl_pips"].reindex(df.index)[mask_c_m]
    tp_c[mask_c_m] = df_m["tp_pips"].reindex(df.index)[mask_c_m]
    
    df_c_sim["sl_pips"] = sl_c
    df_c_sim["tp_pips"] = tp_c
    ops_c, _ = simular_estrategia(
        df_c_sim, 
        "sinal_combinado", 
        "COMBINADA",
        usar_zscore_exit=False,
        coluna_sinal_origem="sinal_zscore"
    )

    # ── 7. Equity curves ──────────────────────────────────────────────────────
    eq_z = construir_equity_curve(ops_z, df.index)
    eq_m = construir_equity_curve(ops_m, df.index)
    eq_c = construir_equity_curve(ops_c, df.index)

    equity_curves  = {"ZSCORE": eq_z, "MOMENTUM": eq_m, "COMBINADA": eq_c}
    operacoes_dict = {"ZSCORE": ops_z, "MOMENTUM": ops_m, "COMBINADA": ops_c}

    # ── 8. Calcular métricas ──────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  MÉTRICAS DE PERFORMANCE COMPLETAS (2016–2026)")
    print(f"{'='*62}")

    todas_metricas: Dict[str, dict] = {}
    for nome, ops, eq in [
        ("ZSCORE",    ops_z, eq_z),
        ("MOMENTUM",  ops_m, eq_m),
        ("COMBINADA", ops_c, eq_c),
    ]:
        m = calcular_metricas(ops, eq, CAPITAL_INICIAL, nome)
        todas_metricas[nome] = m
        exibir_metricas(m)

    # ── 9. Análise por período ────────────────────────────────────────────────
    analise_por_periodo(df, operacoes_dict)

    # ── 10. Gráfico 1: equity curves técnico ─────────────────────────────────
    gerar_grafico_equity(equity_curves, operacoes_dict, GRAFICO_EQUITY)

    # ── 11. Gráfico 2: crescimento de capital premium ─────────────────────────
    gerar_grafico_capital(equity_curves, todas_metricas, GRAFICO_CAPITAL)

    # ── 12. Salvar CSVs ───────────────────────────────────────────────────────
    salvar_resultados(operacoes_dict, todas_metricas)

    # ── 13. Resumo executivo final ────────────────────────────────────────────
    anos = (df.index.max() - df.index.min()).days / 365.25

    print(f"\n{'='*76}")
    print(f"  RESUMO EXECUTIVO FINAL — EURUSD H1 | {df.index.min().date()} → {df.index.max().date()}")
    print(f"{'='*76}")
    print(f"  {'Estratégia':<13} | {'Ops':>4} | {'WR':>7} | {'PnL %':>8} | {'Sharpe':>8} | {'DD Máx':>8} | {'Ops/Ano':>7}")
    print(f"  {'─'*72}")
    for nome, m in todas_metricas.items():
        print(
            f"  {nome:<13} | {m['total_operacoes']:>4} | "
            f"{m['win_rate']:>6.2f}% | {m['pnl_total_pct']:>+7.2f}% | "
            f"{m['sharpe_ratio']:>+7.4f} | {m['drawdown_max_pct']:>7.2f}% | "
            f"{m['total_operacoes']/anos:>7.1f}"
        )
    print(f"{'='*76}")
    print(f"\n  Arquivos gerados:")
    print(f"    → graficos/equity_curves.png         (equity + drawdown técnico)")
    print(f"    → graficos/equity_curves_capital.png (crescimento de capital premium)")
    print(f"    → resultados/operacoes.csv")
    print(f"    → resultados/metricas.csv")
    print(f"{'='*76}\n")


# =============================================================================
# EXECUÇÃO DIRETA
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Backtest manual — Estratégias ZScore e Momentum — EURUSD H1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python backtest.py
  python backtest.py --inicio 2018-01-01 --fim 2023-12-31
        """,
    )
    parser.add_argument("--inicio", type=str, default=None, help="Data de início (YYYY-MM-DD)")
    parser.add_argument("--fim",    type=str, default=None, help="Data de fim (YYYY-MM-DD)")
    args = parser.parse_args()

    print("\n" + "=" * 62)
    print("  QUANT EURUSD — BACKTEST MANUAL")
    print("  Capital: $10.000 | Risco: 1%/trade | Spread: 1.2 pips")
    print("=" * 62)

    executar_backtest(data_inicio=args.inicio, data_fim=args.fim)
