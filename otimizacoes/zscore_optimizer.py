# -*- coding: utf-8 -*-
"""
================================================
zscore_optimizer.py — Estratégia Quantitativa
Versão Genérica — Reutilizável para qualquer ativo
================================================
Configuração:
  Definir ATIVO e TIMEFRAME no bloco de
  configuração no topo deste arquivo antes
  de executar.

Uso:
  1. Configurar ATIVO e TIMEFRAME
  2. Garantir que os parquets de entrada
     existam na pasta data/ do projeto alvo
  3. Executar: python zscore_optimizer.py
================================================
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

# ===========================================
# CONFIGURAÇÃO DO ATIVO — ALTERAR AQUI
# ===========================================
ATIVO          = "EURUSD"
TIMEFRAME      = "H1"
DIR_PROJETO    = Path(__file__).resolve().parent.parent
DIR_DATA       = DIR_PROJETO / "data"
DIR_GRAFICOS   = DIR_PROJETO / "graficos"

PARQUET_COMPLETO    = DIR_DATA / f"{ATIVO.lower()}_{TIMEFRAME.lower()}_completo.parquet"
PARQUET_OPERACIONAL = DIR_DATA / f"{ATIVO.lower()}_{TIMEFRAME.lower()}_operacional.parquet"
PARQUET_HURST       = DIR_DATA / f"{ATIVO.lower()}_{TIMEFRAME.lower()}_hurst.parquet"
PARQUET_SAIDA       = DIR_DATA / f"{ATIVO.lower()}_{TIMEFRAME.lower()}_zscore.parquet"
CAMINHO_GRAFICO     = DIR_GRAFICOS / f"{ATIVO.lower()}_{TIMEFRAME.lower()}_zscore_sinais.png"

# ================================================
# COMO USAR PARA NOVO ATIVO:
# 1. Alterar ATIVO = "NASDAQ" (ou outro)
# 2. Alterar TIMEFRAME = "M10" (ou outro)
# 3. Garantir que existam os parquets:
#    data/nasdaq_m10_completo.parquet
#    data/nasdaq_m10_operacional.parquet
#    data/nasdaq_m10_hurst.parquet (se necessário)
# 4. Executar normalmente
# ================================================

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
    dt_index: pd.DatetimeIndex,
    zscores: np.ndarray
) -> Tuple[List[float], float, float]:
    """
    Simulador extremamente rápido candle a candle para Grid Search.
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
    
    # Lógica de simulação de alta velocidade
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
                elif -0.5 <= zscores[i] <= 0.5:
                    deve_fechar = True
                    preco_saida = close_i
                    motivo = "Z_NEUTRO"
            elif posicao == -1:
                if high_i >= sl_preco:
                    deve_fechar = True
                    preco_saida = sl_preco
                    motivo = "SL"
                elif low_i <= tp_preco:
                    deve_fechar = True
                    preco_saida = tp_preco
                    motivo = "TP"
                elif -0.5 <= zscores[i] <= 0.5:
                    deve_fechar = True
                    preco_saida = close_i
                    motivo = "Z_NEUTRO"
                    
            if deve_fechar:
                _, pnl_m = calcular_pnl(posicao, entrada_preco, preco_saida, lot_size)
                pnls.append(pnl_m)
                capital += pnl_m
                posicao = 0
                
        # Atualizar curva de equidade por candle
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
    logger.info("Carregando parquets...")
    df = pd.read_parquet(PARQUET_COMPLETO, engine="pyarrow")
    
    # Extrair vetores NumPy do DataFrame principal
    closes = df["Close"].to_numpy().astype(np.float64)
    highs = df["High"].to_numpy().astype(np.float64)
    lows = df["Low"].to_numpy().astype(np.float64)
    opens = df["Open"].to_numpy().astype(np.float64)
    zscores = df["zscore"].to_numpy().astype(np.float64)
    hurst = df["hurst"].to_numpy().astype(np.float64) if "hurst" in df.columns else np.full(len(df), 0.5)
    
    # Calcular a volatilidade realizada de 50 barras (para stop dinâmico em pips)
    if "log_return" not in df.columns:
        df["log_return"] = np.log(df["Close"] / df["Close"].shift(1)).astype(np.float32)
    vr = df["log_return"].rolling(window=50, min_periods=50).std(ddof=1).to_numpy()
    vr_pips = vr * closes * FATOR_PIPS
    
    dt_index = df.index
    op_window = verificar_janela_operacional(dt_index)
    
    # Grades de Parâmetros
    hurst_thresholds = [0.40, 0.45, 0.50, 999.0]  # 999.0 = Sem filtro de Hurst
    z_entries = [2.0, 2.5, 3.0, 3.5]
    entrada_tipos = ["DIRETO", "RETORNO"]
    rr_proporcoes = ["1:1.5", "1:1", "1.5:1"]
    
    resultados = []
    
    total_combinacoes = len(hurst_thresholds) * len(z_entries) * len(entrada_tipos) * len(rr_proporcoes)
    logger.info(f"Iniciando Grid Search com {total_combinacoes} combinações...")
    
    count = 0
    for h_t in hurst_thresholds:
        for z_e in z_entries:
            for e_t in entrada_tipos:
                for rr in rr_proporcoes:
                    count += 1
                    
                    # 1. Definir proporção R:R
                    if rr == "1:1.5":
                        sl_mult, tp_mult = 2.0, 3.0
                    elif rr == "1:1":
                        sl_mult, tp_mult = 2.0, 2.0
                    elif rr == "1.5:1":
                        sl_mult, tp_mult = 3.0, 2.0
                        
                    sl_pips_v = sl_mult * vr_pips
                    tp_pips_v = tp_mult * vr_pips
                    
                    # 2. Gerar Sinais baseados no tipo de gatilho
                    sinais = np.zeros(len(df), dtype=np.int8)
                    
                    # Filtro de regime de Hurst e horário operacional
                    cond_base = op_window & (hurst < h_t)
                    
                    if e_t == "DIRETO":
                        cond_long = cond_base & (zscores <= -z_e)
                        cond_short = cond_base & (zscores >= z_e)
                        sinais[cond_long] = 1
                        sinais[cond_short] = -1
                    else:  # "RETORNO"
                        # Cruzamento de volta para dentro
                        z_shift = np.roll(zscores, 1)
                        z_shift[0] = 0.0
                        cond_long = cond_base & (z_shift <= -z_e) & (zscores > -z_e)
                        cond_short = cond_base & (z_shift >= z_e) & (zscores < z_e)
                        sinais[cond_long] = 1
                        sinais[cond_short] = -1
                        
                    # 3. Rodar simulação histórica rápida
                    pnls, cap_final, dd_max = simular_rapido(
                        closes, highs, lows, opens, sinais,
                        sl_pips_v, tp_pips_v, dt_index, zscores
                    )
                    
                    # 4. Calcular métricas resumidas
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
                        "ZScore_Entry": z_e,
                        "Gatilho": e_t,
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
    print("\n" + "█" * 80)
    print("█   RESULTADO DA OTIMIZAÇÃO (TOP 10 PARAMETRIZAÇÕES DE MAIOR LUCRO)")
    print("█" + "─" * 78)
    print(df_res.head(10).to_string())
    print("█" * 80 + "\n")
    
    print("\n" + "█" * 80)
    print("█   RESULTADO DA OTIMIZAÇÃO (TOP 5 COM MENOR DRAWDOWN - MÍNIMO 20 TRADES)")
    print("█" + "─" * 78)
    df_filtrado = df_res[df_res["Trades"] >= 20].sort_values(by="Max_DD %", ascending=False) # dd_max é negativo
    print(df_filtrado.head(5).to_string())
    print("█" * 80 + "\n")
    
    return df_res

if __name__ == "__main__":
    try:
        rodar_otimizacao()
    except Exception as e:
        logger.exception("Erro durante a otimização:")
