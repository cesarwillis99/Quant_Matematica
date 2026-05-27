# -*- coding: utf-8 -*-
"""
================================================================================
ou_optimizer.py — Otimizador de Parâmetros para a Estratégia Ornstein-Uhlenbeck
================================================================================

Objetivo:
    Testar sistematicamente combinações de parâmetros da estratégia baseada no
    processo estocástico de Ornstein-Uhlenbeck (OU):
    1. Tipo de Gatilho: Direto (batida do threshold) vs Retorno (Hooking Back)
    2. Threshold de Entrada (Z_OU)
    3. Limite de Meia-Vida Máximo (Half-Life)
    4. Proporção Retorno-Risco (R:R)
    5. Limite de Saída Neutra (Z_OU retorno ao centro)

    Garantir conformidade matemática estrita e mapeamento completo de PnL e Drawdowns.

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
PARQUET_ENTRADA = DIR_PROJETO / "data" / "eurusd_h1_ou.parquet"

CAPITAL_INICIAL    = 10_000.0
RISCO_POR_TRADE    = 0.01
SPREAD_PIPS        = 0.6  # Calibrado sob spread de 0.6 pips da V2
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
# FILTRAGEM OPERACIONAL
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

def simular_ou_rapido(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    opens: np.ndarray,
    sinais: np.ndarray,
    ou_zscore: np.ndarray,
    ou_valido: np.ndarray,
    vr_pips: np.ndarray,
    sl_mult: float,
    tp_mult: float,
    neutro_lim: float,
    dt_index: pd.DatetimeIndex
) -> Tuple[List[float], float, float]:
    """
    Simulador candle a candle ultrarrápido em NumPy.
    Garante 100% de consistência técnica com as regras do backtest oficial.
    """
    capital = CAPITAL_INICIAL
    posicao = 0  # 0=neutro, 1=LONG, -1=SHORT
    entrada_preco = 0.0
    sl_preco = 0.0
    tp_preco = 0.0
    lot_size = 0.0
    
    pnls = []
    equity = [CAPITAL_INICIAL]
    
    n = len(closes)
    
    for i in range(n):
        dt = dt_index[i]
        dia = dt.weekday()
        hora = dt.hour
        
        eh_sexta_21h = (dia == 4) and (hora == 21)
        bloqueio = (dia == 4 and hora >= 21) or (dia == 5) or (dia == 6 and hora < 21)
        
        # 1. Gerenciar posição aberta
        if posicao != 0:
            high_i = highs[i]
            low_i = lows[i]
            close_i = closes[i]
            z_i = ou_zscore[i]
            valido_i = ou_valido[i]
            
            deve_fechar = False
            preco_saida = 0.0
            motivo = ""
            
            if eh_sexta_21h:
                deve_fechar = True
                preco_saida = close_i
                motivo = "FIM_SEMANA"
            elif posicao == 1:
                if low_i <= sl_preco:
                    deve_fechar = True
                    preco_saida = sl_preco
                    motivo = "SL"
                elif high_i >= tp_preco:
                    deve_fechar = True
                    preco_saida = tp_preco
                    motivo = "TP"
                elif valido_i and (-neutro_lim <= z_i <= neutro_lim):
                    deve_fechar = True
                    preco_saida = close_i
                    motivo = "NEUTRO"
            elif posicao == -1:
                if high_i >= sl_preco:
                    deve_fechar = True
                    preco_saida = sl_preco
                    motivo = "SL"
                elif low_i <= tp_preco:
                    deve_fechar = True
                    preco_saida = tp_preco
                    motivo = "TP"
                elif valido_i and (-neutro_lim <= z_i <= neutro_lim):
                    deve_fechar = True
                    preco_saida = close_i
                    motivo = "NEUTRO"
                    
            if deve_fechar:
                _, pnl_m = calcular_pnl(posicao, entrada_preco, preco_saida, lot_size)
                pnls.append(pnl_m)
                capital += pnl_m
                posicao = 0
                
        # Atualizar curva de equidade
        equity.append(capital)
        
        # 2. Tentar abrir nova posição
        if posicao == 0 and sinais[i] != 0 and not bloqueio:
            if i + 1 >= n:
                continue
                
            vr_i = vr_pips[i]
            if not (math.isfinite(vr_i) and vr_i > 0):
                continue
                
            sl_pips = sl_mult * vr_i
            tp_pips = tp_mult * vr_i
            
            lot_size = calcular_tamanho_lote(capital, sl_pips)
            delta_sl = sl_pips / FATOR_PIPS
            delta_tp = tp_pips / FATOR_PIPS
            
            entrada_preco = opens[i + 1]
            posicao = int(sinais[i])
            
            if posicao == 1:
                sl_preco = entrada_preco - delta_sl
                tp_preco = entrada_preco + delta_tp
            else:
                sl_preco = entrada_preco + delta_sl
                tp_preco = entrada_preco - delta_tp
                
    # Fechar posição residual no fim dos dados
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

def rodar_otimizacao_ou():
    logger.info("Carregando parquets da Ornstein-Uhlenbeck (V2)...")
    if not PARQUET_ENTRADA.exists():
        raise FileNotFoundError(f"Parquet OU não encontrado em: {PARQUET_ENTRADA}. Rode primeiro: python ou_strategy.py")
        
    df = pd.read_parquet(PARQUET_ENTRADA, engine="pyarrow")
    
    closes = df["Close"].to_numpy().astype(np.float64)
    highs = df["High"].to_numpy().astype(np.float64)
    lows = df["Low"].to_numpy().astype(np.float64)
    opens = df["Open"].to_numpy().astype(np.float64)
    
    ou_zscore = df["ou_zscore"].to_numpy().astype(np.float64)
    ou_valido = df["ou_valido"].to_numpy().astype(bool)
    ou_halflife = df["ou_halflife"].to_numpy().astype(np.float64)
    
    # Recalcular vr_pips de forma robusta e consistente
    if "log_return" not in df.columns:
        df["log_return"] = np.log(df["Close"] / df["Close"].shift(1)).astype(np.float32)
    vr = df["log_return"].rolling(window=50, min_periods=50).std(ddof=1).to_numpy()
    vr_pips = vr * closes * FATOR_PIPS
    
    dt_index = df.index
    op_window = verificar_janela_operacional(dt_index)
    
    # Grades de Parâmetros
    gatilho_tipos = ["Direto", "Retorno"]
    thresholds = [1.5, 2.0, 2.5]
    halflife_cortes = [20.0, 35.0, 50.0]
    rr_proporcoes = ["1:1", "1:1.5", "1:2"]
    saidas_neutras = [0.1, 0.3, 0.5]
    
    resultados = []
    
    total_combinacoes = len(gatilho_tipos) * len(thresholds) * len(halflife_cortes) * len(rr_proporcoes) * len(saidas_neutras)
    logger.info(f"Iniciando Grid Search do processo OU com {total_combinacoes} combinações...")
    
    count = 0
    for gatilho in gatilho_tipos:
        for th in thresholds:
            for hl_max in halflife_cortes:
                for rr in rr_proporcoes:
                    for neutro_lim in saidas_neutras:
                        count += 1
                        
                        # 1. Configurar Proporções R:R
                        if rr == "1:1":
                            sl_mult, tp_mult = 2.0, 2.0
                        elif rr == "1:1.5":
                            sl_mult, tp_mult = 2.0, 3.0
                        elif rr == "1:2":
                            sl_mult, tp_mult = 2.0, 4.0
                            
                        # 2. Gerar sinais da combinação
                        sinais = np.zeros(len(df), dtype=np.int8)
                        
                        # Pré-condição de processo válido, half-life limitado e janela operacional
                        cond_base = ou_valido & (ou_halflife >= 1.0) & (ou_halflife <= hl_max) & op_window
                        
                        if gatilho == "Direto":
                            # LONG: Z_ou <= -th
                            cond_long = cond_base & (ou_zscore <= -th)
                            sinais[cond_long] = 1
                            
                            # SHORT: Z_ou >= th
                            cond_short = cond_base & (ou_zscore >= th)
                            sinais[cond_short] = -1
                        elif gatilho == "Retorno":
                            # Gatilho de cruzamento de volta (Hooking Back)
                            # LONG: Zscore anterior <= -th E Zscore atual > -th
                            z_prev = np.roll(ou_zscore, 1)
                            z_prev[0] = 0.0
                            
                            cond_long = cond_base & (z_prev <= -th) & (ou_zscore > -th)
                            sinais[cond_long] = 1
                            
                            # SHORT: Zscore anterior >= th E Zscore atual < th
                            cond_short = cond_base & (z_prev >= th) & (ou_zscore < th)
                            sinais[cond_short] = -1
                            
                        # 3. Rodar simulação manual candle a candle
                        pnls, cap_final, dd_max = simular_ou_rapido(
                            closes, highs, lows, opens, sinais,
                            ou_zscore, ou_valido, vr_pips,
                            sl_mult, tp_mult, neutro_lim, dt_index
                        )
                        
                        # 4. Calcular métricas
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
                            "Gatilho": gatilho,
                            "Threshold": th,
                            "Max_HalfLife": hl_max,
                            "R:R": rr,
                            "Saida_Neutro": neutro_lim,
                            "Trades": n_trades,
                            "Win_Rate %": round(win_rate, 2),
                            "PnL %": round(pnl_total_pct, 2),
                            "Max_DD %": round(dd_max, 2),
                            "Fator_Lucro": round(fator_lucro, 3),
                            "Expectancia/Op": round(expectancia, 2)
                        })
                        
    df_res = pd.DataFrame(resultados)
    
    # Ordenar por PnL % de forma descendente
    df_res = df_res.sort_values(by="PnL %", ascending=False).reset_index(drop=True)
    
    # Exibir resultados no console
    print("\n" + "█" * 95)
    print("█   RESULTADO DA OTIMIZAÇÃO ORNSTEIN-UHLENBECK (TOP 10 PARAMETRIZAÇÕES DE MAIOR LUCRO)")
    print("█" + "─" * 93)
    print(df_res.head(10).to_string())
    print("█" * 95 + "\n")
    
    print("\n" + "█" * 95)
    print("█   RESULTADO DA OTIMIZAÇÃO ORNSTEIN-UHLENBECK (TOP 5 COM MENOR DRAWDOWN - MÍNIMO 30 TRADES)")
    print("█" + "─" * 93)
    df_filtrado = df_res[df_res["Trades"] >= 30].sort_values(by="Max_DD %", ascending=False)
    print(df_filtrado.head(5).to_string())
    print("█" * 95 + "\n")
    
    return df_res

if __name__ == "__main__":
    try:
        rodar_otimizacao_ou()
    except Exception as e:
        logger.exception("Erro crítico durante a otimização de Ornstein-Uhlenbeck:")
