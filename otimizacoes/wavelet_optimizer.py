# -*- coding: utf-8 -*-
"""
================================================
wavelet_optimizer.py — Estratégia Quantitativa
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
  3. Executar: python wavelet_optimizer.py
================================================
"""

import os
import sys
import logging
import itertools
import multiprocessing
import numpy as np
import pandas as pd
import pywt
from pathlib import Path

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
PARQUET_SAIDA       = DIR_DATA / f"{ATIVO.lower()}_{TIMEFRAME.lower()}_wavelet.parquet"
CAMINHO_GRAFICO     = DIR_GRAFICOS / f"{ATIVO.lower()}_{TIMEFRAME.lower()}_wavelet_sinais.png"

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

# Ajustar saída padrão
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DIR_ATUAL = Path(__file__).resolve().parent

# =============================================================================
# SIMULADOR VETORIZADO (ALTA PERFORMANCE)
# =============================================================================
def simular_estrategia_rapida(
    sinais: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    sl_pips_array: np.ndarray,
    tp_pips_array: np.ndarray,
    d_fase: np.ndarray,
    is_operacional: np.ndarray,
    capital_inicial: float = 10000.0,
    risco_perc: float = 0.01
) -> dict:
    
    capital = capital_inicial
    pnl_total = 0.0
    pnl_max = 0.0
    dd_max = 0.0
    
    ops = 0
    wins = 0
    soma_ganhos = 0.0
    soma_perdas = 0.0
    
    posicao_direcao = 0
    preco_entrada = 0.0
    sl_preco = 0.0
    tp_preco = 0.0
    lot_size = 0.0
    
    n = len(sinais)
    d_fase_prev = np.roll(d_fase, 1)
    d_fase_prev[0] = 0
    
    # Pre-calcular mask
    pode_abrir = is_operacional
    
    for i in range(n):
        # 1. Checar Saídas
        if posicao_direcao != 0:
            h = highs[i]
            l = lows[i]
            c = closes[i]
            
            fechar = False
            p_saida = 0.0
            
            if posicao_direcao == 1:
                if l <= sl_preco:
                    fechar, p_saida = True, sl_preco
                elif h >= tp_preco:
                    fechar, p_saida = True, tp_preco
                elif d_fase[i] < 0 and d_fase_prev[i] >= 0:
                    fechar, p_saida = True, c
            else:
                if h >= sl_preco:
                    fechar, p_saida = True, sl_preco
                elif l <= tp_preco:
                    fechar, p_saida = True, tp_preco
                elif d_fase[i] > 0 and d_fase_prev[i] <= 0:
                    fechar, p_saida = True, c
                    
            if fechar:
                # Calcular PnL
                pnl_pips = posicao_direcao * (p_saida - preco_entrada) * 10000
                pnl_net = (pnl_pips * lot_size * 10) - (1.2 * lot_size * 10) # 1.2 spread
                
                capital += pnl_net
                pnl_total += pnl_net
                ops += 1
                
                if pnl_net > 0:
                    wins += 1
                    soma_ganhos += pnl_net
                else:
                    soma_perdas += pnl_net
                    
                if pnl_total > pnl_max:
                    pnl_max = pnl_total
                
                dd_atual = (capital - (capital_inicial + pnl_max)) / (capital_inicial + pnl_max) * 100
                if dd_atual < dd_max:
                    dd_max = dd_atual
                    
                posicao_direcao = 0
                
        # 2. Abrir Novas
        if posicao_direcao == 0 and sinais[i] != 0 and i < n-1 and pode_abrir[i]:
            posicao_direcao = sinais[i]
            preco_entrada = closes[i] # Open do next
            sl = sl_pips_array[i]
            tp = tp_pips_array[i]
            
            risco_mon = capital * risco_perc
            lot_size = max(0.01, risco_mon / (sl * 10))
            
            if posicao_direcao == 1:
                sl_preco = preco_entrada - (sl / 10000.0)
                tp_preco = preco_entrada + (tp / 10000.0)
            else:
                sl_preco = preco_entrada + (sl / 10000.0)
                tp_preco = preco_entrada - (tp / 10000.0)

    wr = (wins / ops) if ops > 0 else 0
    pnl_perc = (capital - capital_inicial) / capital_inicial * 100
    fat_rec = abs(pnl_perc / dd_max) if dd_max < 0 else 0
    
    return {
        "ops": ops,
        "wr": wr * 100,
        "pnl": pnl_perc,
        "dd": dd_max,
        "f_rec": fat_rec
    }

# =============================================================================
# WORKER PARALELO
# =============================================================================
def worker_simulacao(args):
    (
        c_thresh,
        e_perc,
        p_s1,
        sm_win,
        
        cross_12,
        pot0,
        pot1,
        
        pot_norm0,
        energia_total,
        
        log_ret,
        mask_op,
        d_fase,
        d_fase_prev,
        
        highs,
        lows,
        closes,
        sl_pips,
        tp_pips
    ) = args

    # 1. Reconstruir Coerencia Smooth
    n = len(cross_12)
    s12_r = pd.Series(np.real(cross_12)).rolling(sm_win, min_periods=1).mean().values
    s12_i = pd.Series(np.imag(cross_12)).rolling(sm_win, min_periods=1).mean().values
    s12_sq = s12_r**2 + s12_i**2
    s11 = pd.Series(pot0).rolling(sm_win, min_periods=1).mean().values
    s22 = pd.Series(pot1).rolling(sm_win, min_periods=1).mean().values
    
    # Evitar div/0
    denom = s11 * s22
    denom[denom == 0] = 1e-9
    coer_12 = s12_sq / denom
    
    # 2. Energia Quantile
    energia_pX = pd.Series(energia_total).rolling(100, min_periods=1).quantile(e_perc / 100.0).values
    
    # 3. Gerar Sinais Vectorizados
    sinais = np.zeros(n, dtype=np.int8)
    
    coer_ok = coer_12 >= c_thresh
    energ_ok = energia_total > energia_pX
    
    inflex_up = (d_fase > 0) & (d_fase_prev <= 0)
    inflex_dn = (d_fase < 0) & (d_fase_prev >= 0)
    
    pot_s1_ok = pot_norm0 < p_s1
    ret_neg = log_ret < 0
    ret_pos = log_ret > 0
    
    # Operacional já vem checado no backtester no "pode_abrir", mas checamos no sinal tb
    cond_compra = mask_op & coer_ok & energ_ok & inflex_up & pot_s1_ok & ret_neg
    cond_venda  = mask_op & coer_ok & energ_ok & inflex_dn & pot_s1_ok & ret_pos
    
    sinais[cond_compra] = 1
    sinais[cond_venda] = -1
    
    # 4. Backtest 
    res = simular_estrategia_rapida(
        sinais, highs, lows, closes, sl_pips, tp_pips, d_fase, mask_op
    )
    
    return {
        "coer": c_thresh,
        "ene_p": e_perc,
        "pot_s1": p_s1,
        "sm_win": sm_win,
        "ops": res["ops"],
        "wr": res["wr"],
        "pnl": res["pnl"],
        "dd": res["dd"],
        "f_rec": res["f_rec"]
    }

# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================
def main():
    logger.info("Iniciando Otimização Wavelet...")
    
    df = pd.read_parquet(PARQUET_COMPLETO)
    df_op = pd.read_parquet(PARQUET_OPERACIONAL)
    mask_op = df.index.isin(df_op.index)
    
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    log_ret = np.log(closes / np.roll(closes, 1))
    log_ret[0] = 0.0
    
    # ── PRÉ-CÁLCULO PESADO CWT (Rodar apenas 1x) ──
    logger.info("Calculando CWT Base...")
    coefs, _ = pywt.cwt(log_ret, scales=np.array([4, 12, 24, 120]), wavelet='cmor1.5-1.0')
    potencia = np.abs(coefs)**2
    fase = np.angle(coefs)
    
    cross_12 = coefs[0] * np.conj(coefs[1])
    pot0 = potencia[0]
    pot1 = potencia[1]
    
    df_pot = pd.DataFrame(potencia.T)
    pot_norm0 = (df_pot[0] / df_pot[0].rolling(100, min_periods=1).mean()).values
    energia_total = np.sum(potencia, axis=0)
    
    fase_unwrapped = np.unwrap(fase, axis=1)
    d_fase_all = np.zeros_like(fase_unwrapped)
    d_fase_all[:, 1:] = np.diff(fase_unwrapped, axis=1)
    dom_idx = np.argmax(potencia, axis=0)
    d_fase = d_fase_all[dom_idx, np.arange(len(dom_idx))]
    d_fase_prev = np.roll(d_fase, 1)
    d_fase_prev[0] = 0
    
    vr = pd.Series(log_ret).rolling(window=50).std(ddof=1).values
    vr_pips = vr * closes * 10000.0
    sl_pips = np.clip(2.0 * vr_pips, 3.0, 60.0)
    sl_pips[np.isnan(sl_pips)] = 10.0
    tp_pips = np.clip(3.5 * vr_pips, 4.5, 90.0)
    tp_pips[np.isnan(tp_pips)] = 15.0

    # ── GRID DE OTIMIZAÇÃO ──
    grid_coer = [0.2, 0.3, 0.4, 0.5, 0.6]
    grid_ene_p = [20, 30, 40, 50]
    grid_pot_s1 = [0.6, 0.8, 1.0, 1.2]
    grid_sm_win = [2, 4, 6]
    
    params = list(itertools.product(grid_coer, grid_ene_p, grid_pot_s1, grid_sm_win))
    logger.info(f"Gerado espaço de {len(params)} combinações. Rodando multi-core...")
    
    pack = []
    for p in params:
        pack.append((
            p[0], p[1], p[2], p[3],
            cross_12, pot0, pot1, pot_norm0, energia_total,
            log_ret, mask_op, d_fase, d_fase_prev,
            highs, lows, closes, sl_pips, tp_pips
        ))
        
    import time
    start = time.time()
    
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        resultados = pool.map(worker_simulacao, pack)
        
    df_res = pd.DataFrame(resultados)
    df_res = df_res[df_res["ops"] >= 20] # Pelo menos 20 trades
    
    if len(df_res) == 0:
        print("Nenhuma combinacao teve mais de 20 trades.")
        return
        
    logger.info(f"Otimização concluída em {time.time() - start:.2f}s!")
    
    # ── TOP 10 FATOR DE RECUPERAÇÃO ──
    df_res = df_res.sort_values(by="f_rec", ascending=False)
    
    print("\n" + "█" * 70)
    print("█   TOP 10 WAVELET (RANK BY FATOR RECUPERAÇÃO)")
    print("█" * 70)
    cols = ["coer", "ene_p", "pot_s1", "sm_win", "ops", "wr", "pnl", "dd", "f_rec"]
    print(df_res[cols].head(10).to_string(index=False, float_format="{:.2f}".format))
    
    # ── TOP 10 PNL TOTAL ──
    df_res = df_res.sort_values(by="pnl", ascending=False)
    
    print("\n" + "█" * 70)
    print("█   TOP 10 WAVELET (RANK BY PNL TOTAL)")
    print("█" * 70)
    print(df_res[cols].head(10).to_string(index=False, float_format="{:.2f}".format))
    print("\n")

if __name__ == "__main__":
    main()
