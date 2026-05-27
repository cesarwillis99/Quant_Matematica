# -*- coding: utf-8 -*-
"""
================================================================================
momentum_optimizer.py — Otimizador de Parâmetros para a Estratégia Momentum H1
================================================================================

Objetivo:
    Testar sistematicamente combinações de parâmetros da estratégia Momentum:
    1. Thresholds de Regime do Hurst
    2. Limite da Entropia de Shannon (filtro de caos)
    3. Threshold de Percentil de Aceleração
    4. Proporção Retorno-Risco (R:R)

    Identificar a parametrização de maior PnL, menor Drawdown e maior Sharpe.

================================================================================
"""

import os
import sys
import logging
import warnings
import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple

# Suprimir warnings
warnings.filterwarnings("ignore")

# Configurar encoding para Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

# Configurar Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTES E CAMINHOS
# =============================================================================
DIR_PROJETO = Path(__file__).resolve().parent
PARQUET_ENTRADA = DIR_PROJETO / "data" / "eurusd_h1_momentum.parquet"

CAPITAL_INICIAL    = 10_000.0
RISCO_POR_TRADE    = 0.01
SPREAD_PIPS        = 0.5  # Recalculado sob spread de 0.5 pips
VALOR_PIP_POR_LOTE = 10.0
FATOR_PIPS         = 10_000.0

# =============================================================================
# OPERAÇÕES AUXILIARES
# =============================================================================

def calcular_tamanho_lote(capital: float, sl_pips: float) -> float:
    if sl_pips <= 0 or not math.isfinite(sl_pips):
        return 0.01
    risco_monetario = capital * RISCO_POR_TRADE
    lote = risco_monetario / (sl_pips * VALOR_PIP_POR_LOTE)
    return float(np.clip(lote, 0.01, 100.0))

def calcular_pnl(direcao: int, entrada: float, saida: float, lote: float) -> Tuple[float, float]:
    pnl_pips = direcao * (saida - entrada) * FATOR_PIPS
    pnl_bruto = pnl_pips * lote * VALOR_PIP_POR_LOTE
    custo_spread = SPREAD_PIPS * lote * VALOR_PIP_POR_LOTE
    return float(pnl_pips), float(pnl_bruto - custo_spread)

# =============================================================================
# FUNÇÕES DE FILTRAGEM OPERACIONAL
# =============================================================================

def verificar_janela_operacional(dt_index: pd.DatetimeIndex) -> np.ndarray:
    weekday = dt_index.weekday
    hora    = dt_index.strftime('%H:%M')
    seg_sex      = (weekday >= 0) & (weekday <= 4)
    horario_op   = (hora >= "10:00") & (hora <= "22:30")
    return (seg_sex & horario_op)

# =============================================================================
# SIMULADOR RÁPIDO EM NUMPY
# =============================================================================

def simular_rapido(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    opens: np.ndarray,
    sinais: np.ndarray,
    sl_pips_v: np.ndarray,
    tp_pips_v: np.ndarray,
    dt_index: pd.DatetimeIndex
) -> Tuple[List[float], float, float]:
    """
    Simulador extremamente rápido candle a candle para Grid Search.
    Não tem saída por Z-Score neutro para o Momentum puro.
    """
    capital = CAPITAL_INICIAL
    posicao = 0  # 0=neutro, 1=LONG, -1=SHORT
    entrada_preco = 0.0
    sl_preco = 0.0
    tp_preco = 0.0
    lot_size = 0.0
    candle_entrada = 0
    
    pnls = []
    equity = [CAPITAL_INICIAL]
    
    n = len(closes)
    
    for i in range(n):
        dt = dt_index[i]
        dia = dt.weekday()
        hora = dt.hour
        
        eh_sexta_21h = (dia == 4) and (hora == 21)
        bloqueio = (dia == 4 and hora >= 21) or (dia == 5) or (dia == 6 and hora < 21)
        
        if posicao != 0:
            # Gerenciar saída
            high_i = highs[i]
            low_i = lows[i]
            close_i = closes[i]
            
            deve_fechar = False
            preco_saida = 0.0
            
            if eh_sexta_21h:
                deve_fechar = True
                preco_saida = close_i
            elif posicao == 1:
                if low_i <= sl_preco:
                    deve_fechar = True
                    preco_saida = sl_preco
                elif high_i >= tp_preco:
                    deve_fechar = True
                    preco_saida = tp_preco
            elif posicao == -1:
                if high_i >= sl_preco:
                    deve_fechar = True
                    preco_saida = sl_preco
                elif low_i <= tp_preco:
                    deve_fechar = True
                    preco_saida = tp_preco
                    
            if deve_fechar:
                _, pnl_m = calcular_pnl(posicao, entrada_preco, preco_saida, lot_size)
                pnls.append(pnl_m)
                capital += pnl_m
                posicao = 0
                
        # Atualizar curva de equidade
        equity.append(capital)
        
        # Tentar abrir nova posição
        if posicao == 0 and sinais[i] != 0 and not bloqueio:
            if i + 1 >= n:
                continue
            sl_p = sl_pips_v[i]
            tp_p = tp_pips_v[i]
            if not (math.isfinite(sl_p) and sl_p > 0 and math.isfinite(tp_p) and tp_p > 0):
                continue
                
            lot_size = calcular_tamanho_lote(capital, sl_p)
            delta_sl = sl_p / FATOR_PIPS
            delta_tp = tp_p / FATOR_PIPS
            
            entrada_preco = opens[i + 1]
            posicao = int(sinais[i])
            candle_entrada = i
            
            if posicao == 1:
                sl_preco = entrada_preco - delta_sl
                tp_preco = entrada_preco + delta_tp
            else:
                sl_preco = entrada_preco + delta_sl
                tp_preco = entrada_preco - delta_tp
                
    # Fechar posição residual
    if posicao != 0:
        _, pnl_m = calcular_pnl(posicao, entrada_preco, closes[-1], lot_size)
        pnls.append(pnl_m)
        capital += pnl_m
        equity.append(capital)
        
    # Calcular Drawdown Máximo %
    equity_arr = np.array(equity)
    picos = np.maximum.accumulate(equity_arr)
    dds = (equity_arr - picos) / picos * 100
    dd_max = float(dds.min())
    
    return pnls, capital, dd_max

# =============================================================================
# EXECUTOR DO GRID SEARCH
# =============================================================================

def rodar_otimizacao():
    logger.info("Carregando parquets de Momentum...")
    df = pd.read_parquet(PARQUET_ENTRADA, engine="pyarrow")
    
    closes = df["Close"].to_numpy().astype(np.float64)
    highs = df["High"].to_numpy().astype(np.float64)
    lows = df["Low"].to_numpy().astype(np.float64)
    opens = df["Open"].to_numpy().astype(np.float64)
    
    hurst = df["hurst"].to_numpy().astype(np.float64)
    velocidade = df["velocidade"].to_numpy().astype(np.float64)
    percentil_acel = df["percentil_acel"].to_numpy().astype(np.float64)
    entropia_shannon = df["entropia_shannon"].to_numpy().astype(np.float64)
    
    # Calcular volatilidade realizada
    if "log_return" not in df.columns:
        df["log_return"] = np.log(df["Close"] / df["Close"].shift(1)).astype(np.float32)
    vr = df["log_return"].rolling(window=50, min_periods=50).std(ddof=1).to_numpy()
    vr_pips = vr * closes * FATOR_PIPS
    
    dt_index = df.index
    op_window = verificar_janela_operacional(dt_index)
    
    # Grades de Parâmetros
    hurst_thresholds = [0.50, 0.55, 0.60, 999.0]  # 999.0 = Sem filtro de Hurst
    entropia_thresholds = [0.50, 0.55, 0.60, 0.65]
    acel_thresholds = [0.75, 0.80, 0.85]
    rr_proporcoes = ["1:1.5", "1:2", "1:2.5"]
    
    resultados = []
    
    total_combinacoes = len(hurst_thresholds) * len(entropia_thresholds) * len(acel_thresholds) * len(rr_proporcoes)
    logger.info(f"Iniciando Grid Search do Momentum com {total_combinacoes} combinações...")
    
    count = 0
    for h_t in hurst_thresholds:
        for e_t in entropia_thresholds:
            for a_t in acel_thresholds:
                for rr in rr_proporcoes:
                    count += 1
                    
                    # 1. Proporção R:R
                    if rr == "1:1.5":
                        sl_mult, tp_mult = 2.0, 3.0
                    elif rr == "1:2":
                        sl_mult, tp_mult = 2.0, 4.0
                    elif rr == "1:2.5":
                        sl_mult, tp_mult = 2.0, 5.0
                        
                    sl_pips_v = sl_mult * vr_pips
                    tp_pips_v = tp_mult * vr_pips
                    
                    # 2. Gerar sinais
                    sinais = np.zeros(len(df), dtype=np.int8)
                    
                    # Filtros operacionais básicos
                    cond_base = op_window & (hurst > h_t) & (entropia_shannon < e_t)
                    
                    # LONG
                    cond_long = cond_base & (velocidade > 0) & (percentil_acel > a_t)
                    sinais[cond_long] = 1
                    
                    # SHORT
                    cond_short = cond_base & (velocidade < 0) & (percentil_acel < (1.0 - a_t))
                    sinais[cond_short] = -1
                    
                    # 3. Rodar simulação rápida
                    pnls, cap_final, dd_max = simular_rapido(
                        closes, highs, lows, opens, sinais,
                        sl_pips_v, tp_pips_v, dt_index
                    )
                    
                    # 4. Métricas
                    n_trades = len(pnls)
                    if n_trades > 0:
                        win_rate = (np.sum(np.array(pnls) > 0) / n_trades) * 100
                        pnl_total_pct = (cap_final / CAPITAL_INICIAL - 1) * 100
                        expectancia = np.mean(pnls)
                        pnls_arr = np.array(pnls)
                        ganhos = pnls_arr[pnls_arr > 0]
                        perdas = pnls_arr[pnls_arr <= 0]
                        fator_lucro = np.sum(ganhos) / abs(np.sum(perdas)) if len(perdas) > 0 else float("inf")
                    else:
                        win_rate = 0.0
                        pnl_total_pct = 0.0
                        expectancia = 0.0
                        fator_lucro = 0.0
                        
                    resultados.append({
                        "Hurst_Threshold": h_t if h_t < 99.0 else "Sem Filtro",
                        "Entropia_Threshold": e_t,
                        "Acel_Threshold": a_t,
                        "R:R": rr,
                        "Trades": n_trades,
                        "Win_Rate": round(win_rate, 2),
                        "PnL %": round(pnl_total_pct, 2),
                        "Max_DD %": round(dd_max, 2),
                        "Fator_Lucro": round(fator_lucro, 3),
                        "Expectancia/Op": round(expectancia, 2)
                    })
                    
    df_res = pd.DataFrame(resultados)
    
    # Ordenar por PnL % descendente
    df_res = df_res.sort_values(by="PnL %", ascending=False).reset_index(drop=True)
    
    # Exibir no Console
    print("\n" + "█" * 85)
    print("█   RESULTADO DA OTIMIZAÇÃO MOMENTUM (TOP 10 PARAMETRIZAÇÕES DE MAIOR LUCRO)")
    print("█" + "─" * 83)
    print(df_res.head(10).to_string())
    print("█" * 85 + "\n")
    
    print("\n" + "█" * 85)
    print("█   RESULTADO DA OTIMIZAÇÃO MOMENTUM (TOP 5 COM MENOR DRAWDOWN - MÍNIMO 30 TRADES)")
    print("█" + "─" * 83)
    df_filtrado = df_res[df_res["Trades"] >= 30].sort_values(by="Max_DD %", ascending=False)
    print(df_filtrado.head(5).to_string())
    print("█" * 85 + "\n")
    
    return df_res

if __name__ == "__main__":
    try:
        rodar_otimizacao()
    except Exception as e:
        logger.exception("Erro durante a otimização do Momentum:")
