# -*- coding: utf-8 -*-
"""
================================================================================
backtest.py — Simulação Histórica Multiestratégia EURUSD H1
================================================================================

Objetivo:
    Simular as operações históricas geradas pelos três módulos de estratégia:
    ZScore (Mean Reversion clássico), Momentum (Trend Following) e Ornstein-Uhlenbeck (OU),
    tanto de forma isolada quanto combinada (portfólio integrado).

Mecânica do Backtest:
    - Iteração candle a candle sobre toda a base EURUSD H1
    - 1 posição aberta por vez
    - Spread fixo de 0.5 pips por operação
    - Sem slippage adicional (conservador para H1)
    - Capital inicial: 10.000 USD
    - Risco por operação: 1% do capital atual (position sizing dinâmico)
    - Saída por preço (High/Low do candle): SL e TP em pips
    - Saídas por nível neutro (ZScore clássico retornando a [-0.5, 0.5] e OU retornando a [-0.3, 0.3])
    - Fechamento forçado de fim de semana nas sextas-feiras às 21h00

Arquivos de entrada:
    - quant_eurusd_v2/data/eurusd_h1_zscore.parquet
    - quant_eurusd_v2/data/eurusd_h1_momentum.parquet
    - quant_eurusd_v2/data/eurusd_h1_ou.parquet

Arquivos de saída:
    - quant_eurusd/graficos/equity_curves.png e quant_eurusd_v2/graficos/equity_curves.png
    - quant_eurusd/resultados/operacoes.csv e quant_eurusd_v2/resultados/operacoes.csv
    - quant_eurusd/resultados/metricas.csv e quant_eurusd_v2/resultados/metricas.csv

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
DIR_PROJETO_V2 = DIR_ATUAL
DIR_PROJETO_V1 = DIR_ATUAL.parent / "quant_eurusd"

# Parquets de Entrada (Sempre lidos de V2)
PARQUET_ZSCORE   = DIR_PROJETO_V2 / "data" / "eurusd_h1_zscore.parquet"
PARQUET_MOMENTUM = DIR_PROJETO_V2 / "data" / "eurusd_h1_momentum.parquet"
PARQUET_OU       = DIR_PROJETO_V2 / "data" / "eurusd_h1_ou.parquet"

# Parâmetros de Simulação
CAPITAL_INICIAL    = 10_000.0   # USD
RISCO_POR_TRADE    = 0.01       # 1% por trade
SPREAD_PIPS        = 0.5        # Spread de 0.5 pips
VALOR_PIP_POR_LOTE = 10.0       # 10 USD por pip por lote padrão
FATOR_PIPS         = 10_000.0   # 1 pip = 0.0001 no EURUSD

# Regras de Saída Neutra
Z_NEUTRO_MIN, Z_NEUTRO_MAX = -0.5, 0.5
OU_NEUTRO_MIN, OU_NEUTRO_MAX = -0.3, 0.3

# Paleta de Cores Premium (Dark Mode)
COR_FUNDO      = "#0D1117"
COR_TEXTO      = "#E6EDF3"
COR_GRADE      = "#21262D"
COR_ZSCORE     = "#A371F7"   # Roxo
COR_MOMENTUM   = "#F0A500"   # Laranja
COR_OU         = "#00E676"   # Verde Esmeralda
COR_COMBINADA  = "#58A6FF"   # Azul Celeste
COR_POSITIVO   = "#3FB950"   # Verde
COR_NEGATIVO   = "#F85149"   # Vermelho
COR_REFERENCIA = "#484F58"   # Cinza

# Cores de Drawdown Correspondentes
COR_DD_Z = "#503080"
COR_DD_M = "#805000"
COR_DD_O = "#006030"
COR_DD_C = "#1E4F8A"

# Períodos de regimes macroeconômicos
PERIODOS = {
    "Crise de 2008 (2007-2009)":          ("2007-01-01", "2009-12-31"),
    "Crise do Euro (2010-2012)":           ("2010-01-01", "2012-12-31"),
    "Baixa Volatilidade (2013-2019)":      ("2013-01-01", "2019-12-31"),
    "COVID (2020)":                        ("2020-01-01", "2020-12-31"),
    "Alta Inflação / Fed (2021-2023)":     ("2021-01-01", "2023-12-31"),
    "Período Recente (2024-2026)":         ("2024-01-01", "2026-12-31"),
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
    sl_pips:         float                    = 0.0
    tp_pips:         float                    = 0.0
    lot_size:        float                    = 0.0
    capital_entrada: float                    = 0.0
    pnl_pips:        float                    = 0.0
    pnl_monetario:   float                    = 0.0
    duracao_candles: int                      = 0

# =============================================================================
# CÁLCULOS AUXILIARES
# =============================================================================

def calcular_tamanho_lote(capital: float, sl_pips: float) -> float:
    """
    Position Sizing Dinâmico baseado em 1% de risco.
    tamanho_lote = (capital * 0.01) / (sl_pips * 10)
    """
    if sl_pips <= 0 or not math.isfinite(sl_pips):
        return 0.01
    risco_monetario = capital * RISCO_POR_TRADE
    lote = risco_monetario / (sl_pips * VALOR_PIP_POR_LOTE)
    return float(np.clip(lote, 0.01, 100.0))

def calcular_pnl(direcao: int, entrada: float, saida: float, lote: float) -> Tuple[float, float]:
    """
    PnL_pips = direcao * (saida - entrada) * 10000
    PnL_monetario = PnL_pips * lote * 10 - spread * lote * 10
    """
    pnl_pips = direcao * (saida - entrada) * FATOR_PIPS
    pnl_bruto = pnl_pips * lote * VALOR_PIP_POR_LOTE
    custo_spread = SPREAD_PIPS * lote * VALOR_PIP_POR_LOTE
    pnl_net = pnl_bruto - custo_spread
    return float(pnl_pips), float(pnl_net)

def verificar_saida_candle(
    direcao: int,
    high: float,
    low: float,
    close: float,
    sl_preco: float,
    tp_preco: float,
    zscore: Optional[float] = None,
    ou_zscore: Optional[float] = None,
    usar_zscore_exit: bool = False,
    usar_ou_exit: bool = False,
    wav_d_fase: Optional[float] = None,
    wav_d_fase_prev: Optional[float] = None,
    usar_wavelet_exit: bool = False,
    curv_direcao: Optional[int] = None,
    usar_curvatura_exit: bool = False,
) -> Tuple[bool, float, str]:
    """
    Verifica se houve batida de SL, TP ou saída neutra (Z-Score ou OU).
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
        if usar_ou_exit and ou_zscore is not None and not math.isnan(ou_zscore):
            if OU_NEUTRO_MIN <= ou_zscore <= OU_NEUTRO_MAX:
                return True, close, "OU_NEUTRO"
        if usar_wavelet_exit and wav_d_fase is not None and wav_d_fase_prev is not None:
            if wav_d_fase < 0 and wav_d_fase_prev >= 0:  # Inflexão para baixo anula compra
                return True, close, "WAVELET_NEUTRO"
        if usar_curvatura_exit and curv_direcao is not None:
            if curv_direcao == -1:  # Pico de topo anula compra
                return True, close, "CURVATURA_NEUTRO"
                
    elif direcao == -1:  # SHORT
        if high >= sl_preco:
            return True, sl_preco, "SL"
        if low <= tp_preco:
            return True, tp_preco, "TP"
        if usar_zscore_exit and zscore is not None and not math.isnan(zscore):
            if Z_NEUTRO_MIN <= zscore <= Z_NEUTRO_MAX:
                return True, close, "Z_NEUTRO"
        if usar_ou_exit and ou_zscore is not None and not math.isnan(ou_zscore):
            if OU_NEUTRO_MIN <= ou_zscore <= OU_NEUTRO_MAX:
                return True, close, "OU_NEUTRO"
        if usar_wavelet_exit and wav_d_fase is not None and wav_d_fase_prev is not None:
            if wav_d_fase > 0 and wav_d_fase_prev <= 0:  # Inflexão para cima anula venda
                return True, close, "WAVELET_NEUTRO"
        if usar_curvatura_exit and curv_direcao is not None:
            if curv_direcao == 1:  # Pico de fundo anula venda
                return True, close, "CURVATURA_NEUTRO"
                
    return False, 0.0, ""

# =============================================================================
# SIMULADOR DE ESTRATÉGIA
# =============================================================================

def simular_estrategia(
    df: pd.DataFrame,
    coluna_sinal: str,
    nome: str,
    usar_zscore_exit: bool = False,
    usar_ou_exit: bool = False,
    is_combinada: bool = False,
    usar_wavelet_exit: bool = False,
    usar_curvatura_exit: bool = False,
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
    
    # Parâmetros de Stop da estratégia
    sl = df[f"sl_pips_{nome.lower()}"].values.astype(np.float64)
    tp = df[f"tp_pips_{nome.lower()}"].values.astype(np.float64)
    
    zscores = df["zscore"].values.astype(np.float64) if "zscore" in df.columns else None
    ou_zscores = df["ou_zscore"].values.astype(np.float64) if "ou_zscore" in df.columns else None
    
    if "wav_d_fase" in df.columns:
        wav_d_fases = df["wav_d_fase"].values.astype(np.float64)
        wav_d_fases_prev = np.roll(wav_d_fases, 1)
        wav_d_fases_prev[0] = 0
    else:
        wav_d_fases = None
        wav_d_fases_prev = None
        
    curv_direcoes = df["curv_direcao"].values.astype(np.int8) if "curv_direcao" in df.columns else None
    
    origens = df["origem_sinal"].values.astype(np.int8) if "origem_sinal" in df.columns else None
    
    n_rows = len(df)
    
    for i in range(n_rows):
        dt_atual = indices[i]
        dia_semana = dt_atual.weekday()  # 0=Seg... 4=Sex
        hora = dt_atual.hour
        
        # Filtros e bloqueio de fim de semana
        eh_sexta_21h = (dia_semana == 4) and (hora == 21)
        bloqueio_entrada = (
            ((dia_semana == 4) and (hora >= 21)) or   # Sexta-feira pós-21h
            (dia_semana == 5) or                       # Sábado
            ((dia_semana == 6) and (hora < 21))        # Domingo pré-21h
        )
        
        # 1. Gerenciar Posição Aberta
        if posicao is not None:
            # Fechamento forçado sexta 21h
            if eh_sexta_21h:
                pnl_p, pnl_m = calcular_pnl(posicao.direcao, posicao.entrada_preco, closes[i], posicao.lot_size)
                posicao.saida_dt = dt_atual
                posicao.saida_preco = closes[i]
                posicao.motivo_saida = "FIM_SEMANA"
                posicao.pnl_pips = pnl_p
                posicao.pnl_monetario = pnl_m
                posicao.duracao_candles = i - candle_entrada
                capital += pnl_m
                operacoes.append(posicao)
                posicao = None
            else:
                # Determinar quais saídas neutras monitorar dependendo da posição
                z_val = zscores[i] if zscores is not None else None
                o_val = ou_zscores[i] if ou_zscores is not None else None
                wav_val = wav_d_fases[i] if wav_d_fases is not None else None
                wav_prev = wav_d_fases_prev[i] if wav_d_fases_prev is not None else None
                curv_val = curv_direcoes[i] if curv_direcoes is not None else None
                
                u_z_exit = usar_zscore_exit
                u_o_exit = usar_ou_exit
                u_w_exit = (nome == "WAVELET")
                u_c_exit = (nome == "CURVATURA")
                
                if is_combinada and posicao.motivo_saida == "":
                    orig = posicao.estrategia
                    u_z_exit = (orig == "ZSCORE")
                    u_o_exit = (orig == "OU" or orig == "HAWKES")
                    u_w_exit = (orig == "WAVELET")
                    u_c_exit = (orig == "CURVATURA")
                
                deve_fechar, preco_saida, motivo = verificar_saida_candle(
                    posicao.direcao, highs[i], lows[i], closes[i],
                    posicao.sl_preco, posicao.tp_preco,
                    z_val, o_val, u_z_exit, u_o_exit,
                    wav_d_fase=wav_val, wav_d_fase_prev=wav_prev, usar_wavelet_exit=u_w_exit,
                    curv_direcao=curv_val, usar_curvatura_exit=u_c_exit
                )
                
                if deve_fechar:
                    pnl_p, pnl_m = calcular_pnl(posicao.direcao, posicao.entrada_preco, preco_saida, posicao.lot_size)
                    posicao.saida_dt = dt_atual
                    posicao.saida_preco = preco_saida
                    posicao.motivo_saida = motivo
                    posicao.pnl_pips = pnl_p
                    posicao.pnl_monetario = pnl_m
                    posicao.duracao_candles = i - candle_entrada
                    # Restaurar nome correto da estratégia na combinada
                    if is_combinada:
                        posicao.estrategia = "COMBINADA"
                    capital += pnl_m
                    operacoes.append(posicao)
                    posicao = None
                    
        # 2. Abrir Nova Posição
        if posicao is None and sinais[i] != 0 and not bloqueio_entrada:
            if i + 1 >= n_rows:
                continue
                
            # Identificar parâmetros de stop
            sl_pips, tp_pips = sl[i], tp[i]
            orig_nome = nome
            
            if not (math.isfinite(sl_pips) and sl_pips > 0 and math.isfinite(tp_pips) and tp_pips > 0):
                continue
                
            lot = calcular_tamanho_lote(capital, sl_pips)
            delta_sl = sl_pips / FATOR_PIPS
            delta_tp = tp_pips / FATOR_PIPS
            
            preco_entrada = opens[i + 1]
            direc = int(sinais[i])
            
            if direc == 1:  # LONG
                sl_abs = preco_entrada - delta_sl
                tp_abs = preco_entrada + delta_tp
            else:  # SHORT
                sl_abs = preco_entrada + delta_sl
                tp_abs = preco_entrada - delta_tp
                
            id_op += 1
            posicao = Operacao(
                id=id_op,
                estrategia=orig_nome,  # Na combinada, usamos o nome original temporariamente para saber como sair
                direcao=direc,
                entrada_dt=dt_atual,
                entrada_preco=preco_entrada,
                sl_preco=sl_abs,
                tp_preco=tp_abs,
                sl_pips=sl_pips,
                tp_pips=tp_pips,
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
        posicao.pnl_pips = pnl_p
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
# CÁLCULO DE MÉTRICAS (13 MÉTRIAS OBRIGATÓRIAS)
# =============================================================================

def calcular_metricas_performance(
    operacoes: List[Operacao],
    equity: pd.Series,
    nome: str
) -> dict:
    """
    Calcula as 13 métricas solicitadas de forma extremamente precisa.
    """
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
    
    # 1. Total de Operações
    m["total_operacoes"] = len(operacoes)
    
    # 2. Win Rate
    m["win_rate"] = (len(ganhos) / len(operacoes)) * 100 if len(operacoes) > 0 else 0.0
    
    # 3. Média de Ganho
    m["media_ganho"] = float(np.mean(ganhos)) if len(ganhos) > 0 else 0.0
    
    # 4. Média de Perda
    m["media_perda"] = float(np.mean(perdas)) if len(perdas) > 0 else 0.0
    
    # 5. Fator de Lucro
    soma_g = float(np.sum(ganhos))
    soma_p = float(np.sum(perdas))
    m["fator_lucro"] = soma_g / abs(soma_p) if soma_p != 0.0 else float("inf")
    
    # 6. Payoff Ratio
    m["payoff_ratio"] = m["media_ganho"] / abs(m["media_perda"]) if m["media_perda"] != 0.0 else float("inf")
    
    # 7. PnL Total (USD e %)
    m["pnl_total_usd"] = float(np.sum(pnls))
    m["pnl_total_pct"] = (m["pnl_total_usd"] / CAPITAL_INICIAL) * 100
    
    # 8. Drawdown Máximo
    pico = equity.cummax()
    dd_abs = equity - pico
    dd_pct = dd_abs / pico
    m["drawdown_max_usd"] = float(dd_abs.min())
    m["drawdown_max_pct"] = float(dd_pct.min()) * 100
    
    # 9. Fator de Recuperação
    m["fator_recuperacao"] = m["pnl_total_usd"] / abs(m["drawdown_max_usd"]) if m["drawdown_max_usd"] != 0.0 else float("inf")
    
    # 10. Sharpe Ratio (anualizado)
    eq_daily = equity.resample("1D").last().dropna()
    ret_daily = eq_daily.pct_change().dropna()
    if len(ret_daily) >= 2 and ret_daily.std() > 0:
        m["sharpe_ratio"] = (ret_daily.mean() / ret_daily.std()) * np.sqrt(252)
    else:
        m["sharpe_ratio"] = 0.0
        
    # 11. Expectância por Operação
    wr = m["win_rate"] / 100.0
    m["expectancia"] = (wr * m["media_ganho"]) + ((1.0 - wr) * m["media_perda"])
    
    # 12. Período Testado
    m["periodo_inicio"] = str(operacoes[0].entrada_dt.date())
    m["periodo_fim"] = str(operacoes[-1].saida_dt.date()) if operacoes[-1].saida_dt else "em_aberto"
    
    # 13. Percentual do tempo em posição
    duracao_total = sum(op.duracao_candles for op in operacoes)
    m["pct_tempo_posicao"] = (duracao_total / len(equity)) * 100
    
    return m

def exibir_metricas(m: dict):
    """
    Imprime na tela as 13 métricas de forma profissional.
    """
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
    """
    Gera a análise em regimes específicos de volatilidade e regimes macroeconômicos.
    """
    print(f"\n" + "═" * 75)
    print("  ANÁLISE DE ROBUSTEZ POR REGIMES MACROECONÔMICOS")
    print("═" * 75)
    
    for nome_per, (inicio, fim) in PERIODOS.items():
        dt_ini = pd.Timestamp(inicio)
        dt_fim = pd.Timestamp(fim)
        
        # Ignorar períodos inteiramente fora do intervalo de dados (2016-2026)
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
            # Pegar operações abertas dentro do período
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
    todas_metricas: Dict[str, dict],
    caminhos_saida: List[Path]
):
    """
    Desenha 4 equity curves com drawdowns individuais acoplados abaixo em Dark Mode Premium.
    """
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
    
    fig = plt.figure(figsize=(24, 12))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[2.5, 1.2], hspace=0.05, wspace=0.05)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[:, 1])
    ax3.axis("off")
    
    titulo_estrategias = ", ".join(equity_curves.keys())
    fig.suptitle(
        f"EURUSD H1 — Equity Curve e Drawdown ({titulo_estrategias})",
        color=COR_TEXTO, fontsize=14, fontweight="bold", y=0.995
    )
    
    # Eixo de Capital Inicial
    ax1.axhline(CAPITAL_INICIAL, color=COR_REFERENCIA, linestyle=":", alpha=0.6, label="Capital Inicial")
    
    mapa_visual = {
        "ZSCORE":    (COR_ZSCORE,    COR_DD_Z),
        "MOMENTUM":  (COR_MOMENTUM,  COR_DD_M),
        "COMBINADA": (COR_COMBINADA, COR_DD_C)
    }
    
    for nome, eq in equity_curves.items():
        cor, cor_dd = mapa_visual.get(nome, ("#58A6FF", "#1E4F8A"))
        cap_f = eq.iloc[-1]
        pct_f = (cap_f / CAPITAL_INICIAL - 1) * 100
        n_ops = len(todas_operacoes[nome])
        
        # Curva de Equidade
        ax1.plot(
            eq.index, eq.values, color=cor, linewidth=1.5, alpha=0.92,
            label=f"{nome:<9} → ${cap_f:,.2f} ({pct_f:+.2f}%) | {n_ops} trades"
        )
        
        # Curva de Drawdown
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
    
    # Preencher painel lateral com métricas (ax3)
    texto_metricas = "PERFORMANCE ESTATÍSTICA\n"
    texto_metricas += "=" * 30 + "\n\n"
    
    for nome, metricas in todas_metricas.items():
        if nome not in equity_curves:
            continue
        texto_metricas += f"[{nome}]\n"
        texto_metricas += f"PnL Líquido  : {metricas.get('pnl_total_pct', 0.0):.2f}%\n"
        texto_metricas += f"Win Rate     : {metricas.get('win_rate', 0.0):.2f}%\n"
        texto_metricas += f"Sharpe Ratio : {metricas.get('sharpe_ratio', 0.0):.2f}\n"
        texto_metricas += f"Drawdown Max : {metricas.get('drawdown_max_pct', 0.0):.2f}%\n"
        texto_metricas += f"Fator Recup. : {metricas.get('fator_recuperacao', 0.0):.2f}\n"
        texto_metricas += f"Total Trades : {metricas.get('total_operacoes', 0)}\n\n"
        
    ax3.text(0.05, 0.95, texto_metricas, color="#CFD8DC", fontsize=11, 
             va='top', family='monospace', bbox=dict(facecolor='#121212', edgecolor='#333333', pad=10))
    
    # Salvar nos múltiplos destinos fornecidos
    for caminho in caminhos_saida:
        caminho.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(caminho, dpi=150, bbox_inches="tight", facecolor=COR_FUNDO)
        logger.info(f"Gráfico de Equity Curves salvo com sucesso em: {caminho.resolve()}")
        
    plt.close()

# =============================================================================
# SALVAMENTO EM ARQUIVO (CSV E METRICAS)
# =============================================================================

def salvar_arquivos_resultados(
    todas_operacoes: Dict[str, List[Operacao]],
    todas_metricas: Dict[str, dict],
    caminhos_ops: List[Path],
    caminhos_met: List[Path]
):
    """
    Grava os resultados em CSV de forma redundante sênior.
    """
    # 1. operacoes.csv
    registros = []
    for nome, ops in todas_operacoes.items():
        for op in ops:
            registros.append({
                "id": op.id,
                "estrategia": op.estrategia,
                "direcao": "LONG" if op.direcao == 1 else "SHORT",
                "entrada_dt": op.entrada_dt,
                "entrada_preco": round(op.entrada_preco, 5),
                "saida_dt": op.saida_dt,
                "saida_preco": round(op.saida_preco, 5) if op.saida_preco else None,
                "motivo_saida": op.motivo_saida,
                "sl_pips": round(op.sl_pips, 2),
                "tp_pips": round(op.tp_pips, 2),
                "lot_size": round(op.lot_size, 4),
                "pnl_pips": round(op.pnl_pips, 2),
                "pnl_monetario": round(op.pnl_monetario, 2),
                "duracao_candles": op.duracao_candles,
                "capital_entrada": round(op.capital_entrada, 2),
            })
            
    df_ops = pd.DataFrame(registros)
    for caminho in caminhos_ops:
        caminho.parent.mkdir(parents=True, exist_ok=True)
        df_ops.to_csv(caminho, index=False)
        logger.info(f"operacoes.csv gravado com sucesso em: {caminho.resolve()}")
        
    # 2. metricas.csv
    df_met = pd.DataFrame(list(todas_metricas.values()))
    for caminho in caminhos_met:
        caminho.parent.mkdir(parents=True, exist_ok=True)
        df_met.to_csv(caminho, index=False)
        logger.info(f"metricas.csv gravado com sucesso em: {caminho.resolve()}")

# =============================================================================
# CONTROLADOR DO PIPELINE DO BACKTEST
# =============================================================================

def processar_pipeline_backtest(estrategia: str):
    estrategia = estrategia.upper()
    logger.info(f"Iniciando pipeline de backtest individual para a estratégia {estrategia}...")
    
    # Identificar o parquet correto
    parquet_map = {
        "ZSCORE": PARQUET_ZSCORE,
        "MOMENTUM": PARQUET_MOMENTUM,
        "OU": PARQUET_OU
    }
    
    caminho_parquet = parquet_map.get(estrategia, DIR_PROJETO_V2 / "data" / f"eurusd_h1_{estrategia.lower()}.parquet")
    
    if not caminho_parquet.exists():
        raise FileNotFoundError(f"Arquivo parquet não encontrado: {caminho_parquet.name}")
        
    logger.info(f"Carregando dados: {caminho_parquet.name}")
    df = pd.read_parquet(caminho_parquet, engine="pyarrow")
    
    # Padronizar colunas para o simulador
    col_sinal = f"sinal_{estrategia.lower()}"
    if col_sinal not in df.columns:
        if "sinal" in df.columns:
            col_sinal = "sinal"
        else:
            raise ValueError(f"Coluna de sinal {col_sinal} não encontrada no dataframe.")
            
    # Criar colunas temporárias para compatibilidade com simular_estrategia
    df[f"sl_pips_{estrategia.lower()}"] = df["sl_pips"] if "sl_pips" in df.columns else df.get(f"sl_pips_{estrategia.lower()}", 0)
    df[f"tp_pips_{estrategia.lower()}"] = df["tp_pips"] if "tp_pips" in df.columns else df.get(f"tp_pips_{estrategia.lower()}", 0)
    
    logger.info(f"Simulando {len(df):,} candles H1 no período de backtest.")
    
    ops = simular_estrategia(
        df, 
        col_sinal, 
        estrategia, 
        usar_zscore_exit=(estrategia == "ZSCORE"), 
        usar_ou_exit=(estrategia == "OU" or estrategia == "HAWKES" or estrategia == "OU_REVERSO"),
        usar_wavelet_exit=(estrategia == "WAVELET"),
        usar_curvatura_exit=(estrategia == "CURVATURA")
    )
    
    eq = construir_equity_curve(ops, df.index)
    
    equity_curves = {estrategia: eq}
    todas_operacoes = {estrategia: ops}
    
    print(f"\n" + "█" * 70)
    print(f"█   MÉTRICAS DE PERFORMANCE ({estrategia})  █")
    print("█" * 70)
    
    m = calcular_metricas_performance(ops, eq, estrategia)
    todas_metricas = {estrategia: m}
    exibir_metricas(m)
    
    executar_analise_periodos(df, todas_operacoes)
    
    caminhos_grafico = [DIR_PROJETO_V2 / "graficos" / f"equity_curve_{estrategia.lower()}_eurusd.png"]
    
    # Prepara caminhos para métricas (CSV)
    caminhos_ops = [DIR_PROJETO_V2 / "resultados" / f"operacoes_{estrategia.lower()}_eurusd.csv"]
    caminhos_met = [DIR_PROJETO_V2 / "resultados" / f"metricas_{estrategia.lower()}_eurusd.csv"]
    gerar_grafico_equity_curves(equity_curves, todas_operacoes, todas_metricas, caminhos_grafico)
    
    salvar_arquivos_resultados(todas_operacoes, todas_metricas, caminhos_ops, caminhos_met)
    
    print(f"\n" + "█" * 75)
    print("█   RESUMO RÁPIDO")
    print("█" + "─" * 73)
    print(
        f"  {estrategia:<11} | {m['total_operacoes']:>4} ops | "
        f"{m['win_rate']:>6.2f}% WR | "
        f"${m['pnl_total_usd']:>+9.2f} ({m['pnl_total_pct']:>+5.2f}%) | "
        f"{m['drawdown_max_pct']:>7.2f}% DD | "
        f"{m['sharpe_ratio']:>+6.3f} SR"
    )
    print("█" * 75 + "\\n")

# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backtester Manual Individual EURUSD H1",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--estrategia", type=str, required=True, help="Nome da estratégia (ex: ZSCORE, MOMENTUM, OU, NOVA)")
    args = parser.parse_args()
    
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print(f"█   INICIANDO BACKTEST INDIVIDUAL: {args.estrategia.upper():<33} █")
    print("█   Capital Inicial: $10.000 | Risco por operação: 1%          █")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    
    try:
        processar_pipeline_backtest(args.estrategia)
        print("✅ Simulação Histórica Individual concluída com sucesso!\\n")
    except Exception as e:
        logger.exception("Erro crítico durante a execução do backtest:")
        sys.exit(1)
