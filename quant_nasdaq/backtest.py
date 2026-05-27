# -*- coding: utf-8 -*-
"""
================================================================================
backtest.py — Simulação Histórica Multiestratégia NASDAQ M10 (Day Trade)
================================================================================

Objetivo:
    Simular as operações históricas geradas pelos dois módulos de estratégia:
    ZScore (Mean Reversion) e Momentum (Trend Following),
    tanto de forma isolada quanto combinada (portfólio integrado).

Mecânica do Backtest (Day Trade M10):
    - Iteração candle a candle sobre a base NASDAQ M10 completa (01h05 às 23h50).
    - Entradas acontecem APENAS nos candles da série operacional (16h30-22h30).
    - 1 posição aberta por vez por estratégia.
    - Spread fixo de 1.0 ponto (US100 CFD).
    - Capital inicial: 10.000 USD.
    - Risco por operação: 1% do capital atual (position sizing dinâmico).
    - Valor do ponto: $1.00 USD por contrato (lote 1.0).
    - Fechamento forçado de posições na sexta-feira às 22h50.

Arquivos de entrada:
    - data/nasdaq_m10_zscore.parquet
    - data/nasdaq_m10_momentum.parquet

Arquivos de saída:
    - graficos/equity_curves.png
    - resultados/operacoes.csv
    - resultados/metricas.csv

================================================================================
"""

import os
import sys
import argparse
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

# Configurar para salvamento de gráficos sem janela
matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# Configurar encoding para Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

# Configuração do Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTES E CAMINHOS
# =============================================================================
DIR_ATUAL = Path(__file__).resolve().parent

# Parquets de Entrada
PARQUET_ZSCORE   = DIR_ATUAL / "data" / "nasdaq_m10_zscore.parquet"
PARQUET_MOMENTUM = DIR_ATUAL / "data" / "nasdaq_m10_momentum.parquet"

# Parâmetros de Simulação
CAPITAL_INICIAL      = 10_000.0   # USD
RISCO_POR_TRADE      = 0.01       # 1% por trade
SPREAD_PONTOS        = 2.0        # Spread de 2.0 pontos (FTMO US100: 200 points @ 2 digits)
VALOR_PONTO_POR_LOTE = 1.0        # $1 USD por ponto (Tamanho de Contrato = 1)
FATOR_PONTOS         = 1.0        # Não há fator de pips (multiplicador 10000 não usado)

# Regras de Saída Neutra
Z_NEUTRO_MIN, Z_NEUTRO_MAX = -0.5, 0.5

# Paleta de Cores Premium (Dark Mode)
COR_FUNDO      = "#0D1117"
COR_TEXTO      = "#E6EDF3"
COR_GRADE      = "#21262D"
COR_ZSCORE     = "#A371F7"   # Roxo
COR_MOMENTUM   = "#F0A500"   # Laranja
COR_COMBINADA  = "#58A6FF"   # Azul Celeste
COR_POSITIVO   = "#3FB950"   # Verde
COR_NEGATIVO   = "#F85149"   # Vermelho
COR_REFERENCIA = "#484F58"   # Cinza

# Cores de Drawdown Correspondentes
COR_DD_Z = "#503080"
COR_DD_M = "#805000"
COR_DD_C = "#1E4F8A"

# Períodos de regimes macroeconômicos (Anos correspondentes da base real)
PERIODOS = {
    "Baixa Volatilidade (2016-2019)":      ("2016-01-01", "2019-12-31"),
    "COVID / Estímulos (2020)":            ("2020-01-01", "2020-12-31"),
    "Alta Inflação / Fed (2021-2023)":     ("2021-01-01", "2023-12-31"),
    "Período Recente AI (2024-2026)":      ("2024-01-01", "2026-12-31"),
}

# =============================================================================
# DATACLASS DA OPERAÇÃO
# =============================================================================

@dataclass
class Operacao:
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
    sl_pontos:       float                    = 0.0
    tp_pontos:       float                    = 0.0
    lot_size:        float                    = 0.0
    capital_entrada: float                    = 0.0
    pnl_pontos:      float                    = 0.0
    pnl_monetario:   float                    = 0.0
    duracao_candles: int                      = 0

# =============================================================================
# CÁLCULOS AUXILIARES
# =============================================================================

def calcular_tamanho_lote(capital: float, sl_pontos: float) -> float:
    """
    Position Sizing Dinâmico baseado em 1% de risco.
    tamanho_lote = (capital * 0.01) / (sl_pontos * VALOR_PONTO_POR_LOTE)
    """
    if sl_pontos <= 0 or not math.isfinite(sl_pontos):
        return 0.01
    risco_monetario = capital * RISCO_POR_TRADE
    lote = risco_monetario / (sl_pontos * VALOR_PONTO_POR_LOTE)
    return float(np.clip(lote, 0.01, 100.0))

def calcular_pnl(direcao: int, entrada: float, saida: float, lote: float) -> Tuple[float, float]:
    """
    PnL_pontos = direcao * (saida - entrada)
    PnL_monetario = PnL_pontos * lote * 1.0 - spread * lote * 1.0
    """
    pnl_pontos = direcao * (saida - entrada)
    pnl_bruto = pnl_pontos * lote * VALOR_PONTO_POR_LOTE
    custo_spread = SPREAD_PONTOS * lote * VALOR_PONTO_POR_LOTE
    pnl_net = pnl_bruto - custo_spread
    return float(pnl_pontos), float(pnl_net)

def verificar_saida_candle(
    direcao: int,
    high: float,
    low: float,
    close: float,
    sl_preco: float,
    tp_preco: float,
    zscore: Optional[float] = None,
    usar_zscore_exit: bool = False,
) -> Tuple[bool, float, str]:
    """
    Verifica se houve batida de SL, TP ou saída neutra (Z-Score).
    Prioridade: SL -> TP -> Neutro
    """
    if direcao == 1:  # LONG
        if low <= sl_preco:
            return True, sl_preco, "SL"
        if high >= tp_preco:
            return True, tp_preco, "TP"
        if usar_zscore_exit and zscore is not None and not math.isnan(zscore):
            if Z_NEUTRO_MIN <= zscore <= Z_NEUTRO_MAX:
                return True, close, "Z_NEUTRO"
                
    elif direcao == -1:  # SHORT
        if high >= sl_preco:
            return True, sl_preco, "SL"
        if low <= tp_preco:
            return True, tp_preco, "TP"
        if usar_zscore_exit and zscore is not None and not math.isnan(zscore):
            if Z_NEUTRO_MIN <= zscore <= Z_NEUTRO_MAX:
                return True, close, "Z_NEUTRO"
                
    return False, 0.0, ""

# =============================================================================
# SIMULADOR DE ESTRATÉGIA
# =============================================================================

def simular_estrategia(
    df: pd.DataFrame,
    coluna_sinal: str,
    nome: str,
    usar_zscore_exit: bool = False,
    is_combinada: bool = False,
) -> List[Operacao]:
    """
    Executa backtest manual e fiel candle a candle para a estratégia selecionada.
    """
    logger.info(f"Simulando estratégia: {nome}...")
    
    operacoes: List[Operacao] = []
    capital = CAPITAL_INICIAL
    posicao: Optional[Operacao] = None
    candle_entrada = 0
    id_op = 0
    
    indices = df.index
    opens = df["Open"].values.astype(np.float64)
    highs = df["High"].values.astype(np.float64)
    lows = df["Low"].values.astype(np.float64)
    closes = df["Close"].values.astype(np.float64)
    sinais = df[coluna_sinal].values.astype(np.int8)
    
    # Parâmetros de Stop de cada estratégia
    sl_z = df["sl_pips_zscore"].values.astype(np.float64)
    tp_z = df["tp_pips_zscore"].values.astype(np.float64)
    sl_m = df["sl_pips_momentum"].values.astype(np.float64)
    tp_m = df["tp_pips_momentum"].values.astype(np.float64)
    
    zscores = df["zscore"].values.astype(np.float64) if "zscore" in df.columns else None
    
    origens = df["origem_sinal"].values.astype(np.int8) if "origem_sinal" in df.columns else None
    
    n_rows = len(df)
    
    for i in range(n_rows):
        dt_atual = indices[i]
        dia_semana = dt_atual.weekday()  # 0=Seg... 4=Sex
        hora_str = dt_atual.strftime('%H:%M')
        
        # Fechamento forçado sexta 22h50
        eh_sexta_fechamento = (dia_semana == 4) and (hora_str >= "22:50")
        
        # Bloqueio de fim de semana (segurança, mas nossa base já está filtrada)
        bloqueio_entrada = (
            ((dia_semana == 4) and (hora_str >= "22:50")) or   # Sexta-feira pós-22h50
            (dia_semana == 5) or                               # Sábado
            ((dia_semana == 6) and (hora_str < "22:50"))       # Domingo
        )
        
        # 1. Gerenciar Posição Aberta
        if posicao is not None:
            if eh_sexta_fechamento:
                pnl_p, pnl_m = calcular_pnl(posicao.direcao, posicao.entrada_preco, closes[i], posicao.lot_size)
                posicao.saida_dt = dt_atual
                posicao.saida_preco = closes[i]
                posicao.motivo_saida = "FIM_SEMANA"
                posicao.pnl_pontos = pnl_p
                posicao.pnl_monetario = pnl_m
                posicao.duracao_candles = i - candle_entrada
                capital += pnl_m
                operacoes.append(posicao)
                posicao = None
            else:
                # Determinar saída neutra Z-Score
                z_val = zscores[i] if zscores is not None else None
                u_z_exit = usar_zscore_exit
                
                if is_combinada and posicao.motivo_saida == "":
                    orig = posicao.estrategia
                    u_z_exit = (orig == "ZSCORE")
                
                deve_fechar, preco_saida, motivo = verificar_saida_candle(
                    posicao.direcao, highs[i], lows[i], closes[i],
                    posicao.sl_preco, posicao.tp_preco,
                    z_val, u_z_exit
                )
                
                if deve_fechar:
                    pnl_p, pnl_m = calcular_pnl(posicao.direcao, posicao.entrada_preco, preco_saida, posicao.lot_size)
                    posicao.saida_dt = dt_atual
                    posicao.saida_preco = preco_saida
                    posicao.motivo_saida = motivo
                    posicao.pnl_pontos = pnl_p
                    posicao.pnl_monetario = pnl_m
                    posicao.duracao_candles = i - candle_entrada
                    if is_combinada:
                        posicao.estrategia = "COMBINADA"
                    capital += pnl_m
                    operacoes.append(posicao)
                    posicao = None
                    
        # 2. Abrir Nova Posição
        if posicao is None and sinais[i] != 0 and not bloqueio_entrada:
            if i + 1 >= n_rows:
                continue
                
            sl_pontos, tp_pontos = 0.0, 0.0
            orig_nome = nome
            
            if is_combinada and origens is not None:
                orig_s = origens[i]
                if orig_s == 2:  # ZSCORE
                    sl_pontos = sl_z[i]
                    tp_pontos = tp_z[i]
                    orig_nome = "ZSCORE"
                elif orig_s == 3:  # MOMENTUM
                    sl_pontos = sl_m[i]
                    tp_pontos = tp_m[i]
                    orig_nome = "MOMENTUM"
            else:
                if nome == "ZSCORE":
                    sl_pontos = sl_z[i]
                    tp_pontos = tp_z[i]
                elif nome == "MOMENTUM":
                    sl_pontos = sl_m[i]
                    tp_pontos = tp_m[i]
                    
            if not (math.isfinite(sl_pontos) and sl_pontos > 0 and math.isfinite(tp_pontos) and tp_pontos > 0):
                continue
                
            lot = calcular_tamanho_lote(capital, sl_pontos)
            
            preco_entrada = opens[i + 1]
            direc = int(sinais[i])
            
            if direc == 1:  # LONG
                sl_abs = preco_entrada - sl_pontos
                tp_abs = preco_entrada + tp_pontos
            else:  # SHORT
                sl_abs = preco_entrada + sl_pontos
                tp_abs = preco_entrada - tp_pontos
                
            id_op += 1
            posicao = Operacao(
                id=id_op,
                estrategia=orig_nome,
                direcao=direc,
                entrada_dt=dt_atual,
                entrada_preco=preco_entrada,
                sl_preco=sl_abs,
                tp_preco=tp_abs,
                sl_pontos=sl_pontos,
                tp_pontos=tp_pontos,
                lot_size=lot,
                capital_entrada=capital
            )
            candle_entrada = i
            
    # Fechar posição aberta no último candle
    if posicao is not None:
        pnl_p, pnl_m = calcular_pnl(posicao.direcao, posicao.entrada_preco, closes[-1], posicao.lot_size)
        posicao.saida_dt = indices[-1]
        posicao.saida_preco = closes[-1]
        posicao.motivo_saida = "FIM_DADOS"
        posicao.pnl_pontos = pnl_p
        posicao.pnl_monetario = pnl_m
        posicao.duracao_candles = n_rows - 1 - candle_entrada
        if is_combinada:
            posicao.estrategia = "COMBINADA"
        capital += pnl_m
        operacoes.append(posicao)
        
    logger.info(f"Fim da simulação {nome}: {len(operacoes)} operações | Capital Final: ${capital:,.2f}")
    return operacoes

# =============================================================================
# CONSTRUTOR DE EQUITY CURVE
# =============================================================================

def construir_equity_curve(operacoes: List[Operacao], df_index: pd.DatetimeIndex) -> pd.Series:
    """
    Gera a série temporal de capital acumulado.
    """
    equity = pd.Series(CAPITAL_INICIAL, index=df_index, dtype=np.float64)
    ops_ordenadas = sorted(
        [op for op in operacoes if op.saida_dt is not None],
        key=lambda op: op.saida_dt
    )
    cap = CAPITAL_INICIAL
    for op in ops_ordenadas:
        cap += op.pnl_monetario
        equity[op.saida_dt:] = cap
    return equity

# =============================================================================
# CÁLCULO DE MÉTRICAS
# =============================================================================

def calcular_metricas_performance(
    operacoes: List[Operacao],
    equity: pd.Series,
    nome: str
) -> dict:
    m = {"nome": nome}
    
    if not operacoes:
        m.update({
            "total_operacoes": 0, "win_rate": 0.0,
            "media_ganho": 0.0, "media_perda": 0.0,
            "fator_lucro": 0.0, "payoff_ratio": 0.0,
            "pnl_total_usd": 0.0, "pnl_total_pct": 0.0,
            "drawdown_max_usd": 0.0, "drawdown_max_pct": 0.0,
            "fator_recuperacao": 0.0, "sharpe_ratio": 0.0,
            "expectancia": 0.0, "periodo_inicio": "N/A",
            "periodo_fim": "N/A", "pct_tempo_posicao": 0.0
        })
        return m
        
    pnls = np.array([op.pnl_monetario for op in operacoes])
    ganhos = pnls[pnls > 0]
    perdas = pnls[pnls <= 0]
    
    m["total_operacoes"] = len(operacoes)
    m["win_rate"] = (len(ganhos) / len(operacoes)) * 100 if len(operacoes) > 0 else 0.0
    m["media_ganho"] = float(np.mean(ganhos)) if len(ganhos) > 0 else 0.0
    m["media_perda"] = float(np.mean(perdas)) if len(perdas) > 0 else 0.0
    
    soma_g = float(np.sum(ganhos))
    soma_p = float(np.sum(perdas))
    m["fator_lucro"] = soma_g / abs(soma_p) if soma_p != 0.0 else float("inf")
    m["payoff_ratio"] = m["media_ganho"] / abs(m["media_perda"]) if m["media_perda"] != 0.0 else float("inf")
    
    m["pnl_total_usd"] = float(np.sum(pnls))
    m["pnl_total_pct"] = (m["pnl_total_usd"] / CAPITAL_INICIAL) * 100
    
    pico = equity.cummax()
    dd_abs = equity - pico
    dd_pct = dd_abs / pico
    m["drawdown_max_usd"] = float(dd_abs.min())
    m["drawdown_max_pct"] = float(dd_pct.min()) * 100
    
    m["fator_recuperacao"] = m["pnl_total_usd"] / abs(m["drawdown_max_usd"]) if m["drawdown_max_usd"] != 0.0 else float("inf")
    
    eq_daily = equity.resample("1D").last().dropna()
    ret_daily = eq_daily.pct_change().dropna()
    if len(ret_daily) >= 2 and ret_daily.std() > 0:
        m["sharpe_ratio"] = (ret_daily.mean() / ret_daily.std()) * np.sqrt(252)
    else:
        m["sharpe_ratio"] = 0.0
        
    wr = m["win_rate"] / 100.0
    m["expectancia"] = (wr * m["media_ganho"]) + ((1.0 - wr) * m["media_perda"])
    
    m["periodo_inicio"] = str(operacoes[0].entrada_dt.date())
    m["periodo_fim"] = str(operacoes[-1].saida_dt.date()) if operacoes[-1].saida_dt else "em_aberto"
    
    duracao_total = sum(op.duracao_candles for op in operacoes)
    m["pct_tempo_posicao"] = (duracao_total / len(equity)) * 100
    
    return m

def exibir_metricas(m: dict):
    sep = "─" * 65
    print(f"\n{sep}")
    print(f"  MÉTRICAS DE PERFORMANCE — {m['nome']}")
    print(f"{sep}")
    print(f"  1.  Período Testado          : {m['periodo_inicio']} a {m['periodo_fim']}")
    print(f"  2.  Total de Operações       : {m['total_operacoes']:>8,}")
    print(f"  3.  Win Rate                 : {m['win_rate']:>7.2f}%")
    print(f"  4.  Média de Ganho           : ${m['media_ganho']:>10.2f}")
    print(f"  5.  Média de Perda           : ${m['media_perda']:>10.2f}")
    print(f"  6.  Fator de Lucro           : {m['fator_lucro']:>8.3f}x")
    print(f"  7.  Payoff Ratio             : {m['payoff_ratio']:>8.3f}x")
    print(f"  8.  PnL Total                : ${m['pnl_total_usd']:>+10.2f} ({m['pnl_total_pct']:+.2f}%)")
    print(f"  9.  Drawdown Máximo          : {m['drawdown_max_pct']:>8.2f}% (${m['drawdown_max_usd']:,.2f})")
    print(f"  10. Fator de Recuperação     : {m['fator_recuperacao']:>8.3f}x")
    print(f"  11. Sharpe Ratio (Anual)     : {m['sharpe_ratio']:>8.4f}")
    print(f"  12. Expectância Operacional  : ${m['expectancia']:>+10.2f}")
    print(f"  13. Tempo em Posição         : {m['pct_tempo_posicao']:>7.2f}%")
    print(f"{sep}")

# =============================================================================
# ANÁLISE MACROECONÔMICA POR PERÍODOS
# =============================================================================

def executar_analise_periodos(df: pd.DataFrame, todas_operacoes: Dict[str, List[Operacao]]):
    print(f"\n" + "═" * 75)
    print("  ANÁLISE DE ROBUSTEZ POR REGIMES MACROECONÔMICOS (NASDAQ)")
    print("═" * 75)
    
    for nome_per, (inicio, fim) in PERIODOS.items():
        dt_ini = pd.Timestamp(inicio)
        dt_fim = pd.Timestamp(fim)
        
        if dt_fim < df.index.min() or dt_ini > df.index.max():
            print(f"\n  Período: {nome_per}")
            print("  " + "─" * 65)
            print(f"    ⚠ Sem cobertura de dados na série (Disponível: {df.index.min().date()} a {df.index.max().date()})")
            continue
            
        print(f"\n  Período: {nome_per}")
        print("  " + "─" * 65)
        
        df_p = df[(df.index >= dt_ini) & (df.index <= dt_fim)]
        if len(df_p) == 0:
            print("    (sem candles neste intervalo)")
            continue
            
        for nome_est, ops in todas_operacoes.items():
            ops_p = [op for op in ops if dt_ini <= op.entrada_dt <= dt_fim]
            
            if not ops_p:
                print(f"    {nome_est:<12}: sem operações no período")
                continue
                
            eq_p = construir_equity_curve(ops_p, df_p.index)
            m = calcular_metricas_performance(ops_p, eq_p, nome_est)
            
            print(
                f"    {nome_est:<12}: {m['total_operacoes']:3d} ops | "
                f"WR={m['win_rate']:5.1f}% | PnL={m['pnl_total_pct']:+6.2f}% | "
                f"DD={m['drawdown_max_pct']:5.2f}% | Sharpe={m['sharpe_ratio']:+.3f}"
            )
            
    print("═" * 75 + "\n")

# =============================================================================
# GERAÇÃO DAS PLOTAGENS DE CURVAS DE EQUIDADE
# =============================================================================

def gerar_grafico_equity_curves(
    equity_curves: Dict[str, pd.Series],
    todas_operacoes: Dict[str, List[Operacao]],
    caminhos_saida: List[Path]
):
    logger.info("Gerando gráfico profissional de Equity Curves e Drawdowns...")
    
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
    
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(20, 12), sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1.2], "hspace": 0.05}
    )
    
    fig.suptitle(
        "NASDAQ Day Trade M10 — Equity Curves Históricas e Drawdown (ZScore, Momentum, Combinada)",
        color=COR_TEXTO, fontsize=14, fontweight="bold", y=0.995
    )
    
    ax1.axhline(CAPITAL_INICIAL, color=COR_REFERENCIA, linestyle=":", alpha=0.6, label="Capital Inicial")
    
    mapa_visual = {
        "ZSCORE":    (COR_ZSCORE,    COR_DD_Z),
        "MOMENTUM":  (COR_MOMENTUM,  COR_DD_M),
        "COMBINADA": (COR_COMBINADA, COR_DD_C)
    }
    
    for nome, eq in equity_curves.items():
        cor, cor_dd = mapa_visual[nome]
        cap_f = eq.iloc[-1]
        pct_f = (cap_f / CAPITAL_INICIAL - 1) * 100
        n_ops = len(todas_operacoes[nome])
        
        ax1.plot(
            eq.index, eq.values, color=cor, linewidth=1.5, alpha=0.92,
            label=f"{nome:<9} → ${cap_f:,.2f} ({pct_f:+.2f}%) | {n_ops} trades"
        )
        
        pico = eq.cummax()
        dd = ((eq - pico) / pico) * 100
        ax2.fill_between(eq.index, dd, 0, color=cor_dd, alpha=0.15)
        ax2.plot(eq.index, dd, color=cor, linewidth=0.8, alpha=0.8)
        
    ax1.set_ylabel("Capital (USD)", color=COR_TEXTO, fontsize=11)
    ax1.legend(loc="upper left", fontsize=10, framealpha=0.3)
    ax1.grid(True, linestyle="--", alpha=0.15)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    
    ax2.set_ylabel("Drawdown (%)", color=COR_TEXTO, fontsize=11)
    ax2.set_xlabel("Data", color=COR_TEXTO, fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.15)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    
    fig.autofmt_xdate(rotation=25, ha="right")
    
    for caminho in caminhos_saida:
        caminho.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(caminho, dpi=150, bbox_inches="tight", facecolor=COR_FUNDO)
        logger.info(f"Gráfico de Equity Curves salvo com sucesso em: {caminho.resolve()}")
        
    plt.close()

# =============================================================================
# SALVAMENTO EM ARQUIVO
# =============================================================================

def salvar_arquivos_resultados(
    todas_operacoes: Dict[str, List[Operacao]],
    todas_metricas: Dict[str, dict],
    caminho_ops: Path,
    caminho_met: Path
):
    registros = []
    for nome, ops in todas_operacoes.items():
        for op in ops:
            registros.append({
                "id": op.id,
                "estrategia": op.estrategia,
                "direcao": "LONG" if op.direcao == 1 else "SHORT",
                "entrada_dt": op.entrada_dt,
                "entrada_preco": round(op.entrada_preco, 2),
                "saida_dt": op.saida_dt,
                "saida_preco": round(op.saida_preco, 2) if op.saida_preco else None,
                "motivo_saida": op.motivo_saida,
                "sl_pontos": round(op.sl_pontos, 2),
                "tp_pontos": round(op.tp_pontos, 2),
                "lot_size": round(op.lot_size, 4),
                "pnl_pontos": round(op.pnl_pontos, 2),
                "pnl_monetario": round(op.pnl_monetario, 2),
                "duracao_candles": op.duracao_candles,
                "capital_entrada": round(op.capital_entrada, 2),
            })
            
    df_ops = pd.DataFrame(registros)
    caminho_ops.parent.mkdir(parents=True, exist_ok=True)
    df_ops.to_csv(caminho_ops, index=False)
    logger.info(f"operacoes.csv gravado em: {caminho_ops.resolve()}")
        
    df_met = pd.DataFrame(list(todas_metricas.values()))
    caminho_met.parent.mkdir(parents=True, exist_ok=True)
    df_met.to_csv(caminho_met, index=False)
    logger.info(f"metricas.csv gravado em: {caminho_met.resolve()}")

# =============================================================================
# CONTROLADOR DO PIPELINE
# =============================================================================

def processar_pipeline_backtest():
    logger.info("Verificando bases de dados do ZScore e Momentum...")
    for p in [PARQUET_ZSCORE, PARQUET_MOMENTUM]:
        if not p.exists():
            raise FileNotFoundError(f"Parquet não encontrado: {p.name}")
            
    logger.info("Carregando parquets...")
    df_z = pd.read_parquet(PARQUET_ZSCORE, engine="pyarrow")
    df_m = pd.read_parquet(PARQUET_MOMENTUM, engine="pyarrow")
    
    df = df_z.copy()
    
    df["sinal_momentum"] = df_m["sinal_momentum"].reindex(df.index, fill_value=0)
    
    # O SL e TP da v2 baseada no eurusd foi renomeado de sl_pips para sl_pips. Nós deixamos a base de dados
    # gerando colunas "sl_pips", "tp_pips". Elas precisam ser referenciadas como sl_pips_zscore, etc.
    df["sl_pips_zscore"] = df_z["sl_pips"].reindex(df.index)
    df["tp_pips_zscore"] = df_z["tp_pips"].reindex(df.index)
    
    df["sl_pips_momentum"] = df_m["sl_pips"].reindex(df.index)
    df["tp_pips_momentum"] = df_m["tp_pips"].reindex(df.index)
    
    logger.info(f"Dados unificados! {len(df):,} candles M10 no período.")
    
    sinal_comb = np.zeros(len(df), dtype=np.int8)
    origem_sinal = np.zeros(len(df), dtype=np.int8)  # 2 = ZSCORE, 3 = MOMENTUM
    
    s_z = df["sinal_zscore"].values
    s_m = df["sinal_momentum"].values
    
    for i in range(len(df)):
        if s_z[i] != 0:
            sinal_comb[i] = s_z[i]
            origem_sinal[i] = 2
        elif s_m[i] != 0:
            sinal_comb[i] = s_m[i]
            origem_sinal[i] = 3
            
    df["sinal_combinado"] = sinal_comb
    df["origem_sinal"] = origem_sinal
    
    n_z_tot = (s_z != 0).sum()
    n_m_tot = (s_m != 0).sum()
    n_comb_tot = (sinal_comb != 0).sum()
    
    logger.info(f"Sinais gerados na base: ZScore={n_z_tot} | Momentum={n_m_tot} | Combinada={n_comb_tot}")
    
    # ── Simulações Individuais ──
    ops_z = simular_estrategia(df, "sinal_zscore", "ZSCORE", usar_zscore_exit=True)
    ops_m = simular_estrategia(df, "sinal_momentum", "MOMENTUM", usar_zscore_exit=False)
    ops_c = simular_estrategia(df, "sinal_combinado", "COMBINADA", is_combinada=True)
    
    # ── Construção de Equity Curves ──
    eq_z = construir_equity_curve(ops_z, df.index)
    eq_m = construir_equity_curve(ops_m, df.index)
    eq_c = construir_equity_curve(ops_c, df.index)
    
    equity_curves = {"ZSCORE": eq_z, "MOMENTUM": eq_m, "COMBINADA": eq_c}
    todas_operacoes = {"ZSCORE": ops_z, "MOMENTUM": ops_m, "COMBINADA": ops_c}
    
    # ── Métricas Completas de Performance ──
    print(f"\n" + "█" * 70)
    print("█   MÉTRICAS DE PERFORMANCE COMPLETAS (Série Histórica Total)  █")
    print("█" * 70)
    
    todas_metricas = {}
    for nome in ["ZSCORE", "MOMENTUM", "COMBINADA"]:
        ops = todas_operacoes[nome]
        eq = equity_curves[nome]
        m = calcular_metricas_performance(ops, eq, nome)
        todas_metricas[nome] = m
        exibir_metricas(m)
        
    # ── Análise por Período de Regimes ──
    executar_analise_periodos(df, todas_operacoes)
    
    # ── Geração de Gráfico ──
    caminho_grafico = DIR_ATUAL / "graficos" / "equity_curves.png"
    gerar_grafico_equity_curves(equity_curves, todas_operacoes, [caminho_grafico])
    
    # ── Gravação de Resultados ──
    caminho_ops = DIR_ATUAL / "resultados" / "operacoes.csv"
    caminho_met = DIR_ATUAL / "resultados" / "metricas.csv"
    salvar_arquivos_resultados(todas_operacoes, todas_metricas, caminho_ops, caminho_met)
    
    # Resumo Executivo Rápido
    print(f"\n" + "█" * 75)
    print("█   TABELA COMPACTA DE PERFORMANCE (NASDAQ Day Trade M10)")
    print("█" + "─" * 73)
    print(f"  {'Estratégia':<11} | {'Ops':>4} | {'WinRate':>7} | {'PnL Total':>12} | {'Drawdown':>8} | {'Sharpe':>7}")
    print("  " + "─" * 71)
    
    for nome, m in todas_metricas.items():
        print(
            f"  {nome:<11} | {m['total_operacoes']:>4} | "
            f"{m['win_rate']:>6.2f}% | "
            f"${m['pnl_total_usd']:>+9.2f} ({m['pnl_total_pct']:>+5.2f}%) | "
            f"{m['drawdown_max_pct']:>7.2f}% | "
            f"{m['sharpe_ratio']:>+6.3f}"
        )
    print("█" * 75 + "\n")

# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backtester Manual Multiestratégia NASDAQ M10",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    args = parser.parse_args()
    
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█   INICIANDO BACKTEST MULTIESTRATÉGIA NASDAQ M10 (Day Trade)  █")
    print("█   Capital Inicial: $10.000 | Risco por operação: 1%          █")
    print("█   Ativos simulados: ZScore, Momentum, Portfólio Combinado    █")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    
    try:
        processar_pipeline_backtest()
        print("✅ Simulação Histórica (Backtest) concluída com sucesso!\n")
    except Exception as e:
        logger.exception("Erro crítico durante a execução do backtest:")
        sys.exit(1)
