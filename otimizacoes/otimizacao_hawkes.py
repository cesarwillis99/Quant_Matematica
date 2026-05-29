import os
import sys
import math
import numpy as np
import pandas as pd
import itertools
from scipy.optimize import minimize
from pathlib import Path
from tqdm import tqdm
import warnings
import argparse
import json

warnings.filterwarnings("ignore")

# =============================================================================
# CONSTANTES E CAMINHOS
# =============================================================================
DIR_PROJETO = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser(description="Otimizador Hawkes Grid Search")
parser.add_argument("--ativo", type=str, default="eurusd")
parser.add_argument("--timeframe", type=str, default="h1")
args = parser.parse_args()

ativo = args.ativo.lower()
timeframe = args.timeframe.lower()

DIR_DATA    = DIR_PROJETO / f"quant_{ativo}_{timeframe}" / "data"
DIR_SAIDA   = DIR_DATA / "otimizacoes"

PARQUET_COMPLETO    = DIR_DATA / f"{ativo}_{timeframe}_completo.parquet"
PARQUET_OPERACIONAL = DIR_DATA / f"{ativo}_{timeframe}_operacional.parquet"
ARQUIVO_SAIDA       = DIR_SAIDA / "otimizacao_hawkes_resultados.parquet"

# =============================================================================
# GRID SEARCH
# =============================================================================
GRID_PARAMS = {
    "excitacao_maxima": [0.55, 0.65, 0.75],
    "lambda_norm_min_sinal": [2.0, 2.5, 3.0, 3.5],
    "multiplicador_sl": [1.5, 2.0, 2.5],
    "multiplicador_tp": [2.0, 3.0, 4.0]
}

JANELA_EVENTOS = 60
JANELA_MLE = 120
PASSO_MLE = 24
JANELA_VOL = 50

# =============================================================================
# CORE MATH: HAWKES MLE
# =============================================================================
def log_likelihood_hawkes_numba(params, N_array):
    mu, alpha, beta = params
    if mu <= 0 or alpha <= 0 or beta <= 0 or alpha / beta >= 1.0:
        return 1e9
        
    n_len = len(N_array)
    ll = 0.0
    lam_prev = mu
    
    for t in range(1, n_len):
        lam_t = mu + math.exp(-beta) * (lam_prev - mu) + alpha * N_array[t-1]
        if lam_t <= 0: lam_t = 1e-9
        ll += N_array[t] * math.log(lam_t) - lam_t
        lam_prev = lam_t
        
    return -ll

def estimar_hawkes_mle(N_array: np.ndarray):
    if np.sum(N_array) == 0:
        return 0.1, 0.001, 1.0, False
        
    x0 = np.array([0.1, 0.5, 1.0])
    bnds = ((0.001, 5.0), (0.001, 5.0), (0.001, 10.0))
    
    res = minimize(
        log_likelihood_hawkes_numba, 
        x0, 
        args=(N_array,), 
        method="L-BFGS-B", 
        bounds=bnds
    )
    
    mu_est, alpha_est, beta_est = res.x
    valido = res.success and (alpha_est / beta_est < 1.0)
    return float(mu_est), float(alpha_est), float(beta_est), bool(valido)

# =============================================================================
# PRÉ-CÁLCULO FORA DO LOOP
# =============================================================================
def pre_calcular_base() -> tuple:
    print("Carregando bases de dados H1 para Hawkes...")
    df = pd.read_parquet(PARQUET_COMPLETO, engine="pyarrow")
    df_op = pd.read_parquet(PARQUET_OPERACIONAL, engine="pyarrow")
    n_candles = len(df)
    
    print("Calculando R_t e std_60...")
    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values
    
    R_t = np.zeros(n_candles, dtype=np.float32)
    R_t[1:] = np.log(closes[1:] / closes[:-1])
    R_t_abs = np.abs(R_t)
    
    std_60 = pd.Series(R_t).rolling(window=JANELA_EVENTOS).std(ddof=1).values
    threshold_dyn = 2.0 * std_60
    
    N_t = np.zeros(n_candles, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        N_t = np.where(R_t_abs > threshold_dyn, 1.0, 0.0)
    N_t[np.isnan(N_t)] = 0.0
    
    print("Iniciando MLE do processo de Hawkes a cada 24 horas...")
    mu_hist = np.full(n_candles, np.nan, dtype=np.float32)
    alpha_hist = np.full(n_candles, np.nan, dtype=np.float32)
    beta_hist = np.full(n_candles, np.nan, dtype=np.float32)
    valido_hist = np.zeros(n_candles, dtype=np.int8)
    
    mu_atual, alpha_atual, beta_atual = 0.1, 0.001, 1.0
    valido_atual = False
    
    total_opt = (n_candles - JANELA_MLE) // PASSO_MLE
    opt_count = 0
    
    for t in tqdm(range(JANELA_MLE, n_candles), desc="MLE Rolling"):
        if t % PASSO_MLE == 0:
            window_N = N_t[t-JANELA_MLE:t]
            mu_est, alpha_est, beta_est, is_val = estimar_hawkes_mle(window_N)
            mu_atual = mu_est
            alpha_atual = alpha_est
            beta_atual = beta_est
            valido_atual = is_val
            
        mu_hist[t] = mu_atual
        alpha_hist[t] = alpha_atual
        beta_hist[t] = beta_atual
        valido_hist[t] = 1 if valido_atual else 0

    print("Calculando intensidade lambda_t recursivamente...")
    lambda_t = np.full(n_candles, np.nan, dtype=np.float32)
    lambda_norm = np.full(n_candles, np.nan, dtype=np.float32)
    excitacao = np.full(n_candles, np.nan, dtype=np.float32)
    
    lam_prev = 0.1
    for t in range(JANELA_MLE, n_candles):
        mu = mu_hist[t]
        alpha = alpha_hist[t]
        beta = beta_hist[t]
        
        lam_curr = mu + math.exp(-beta) * (lam_prev - mu) + alpha * N_t[t-1]
        if lam_curr < mu:
            lam_curr = mu
            
        lambda_t[t] = lam_curr
        lambda_norm[t] = (lam_curr - mu) / mu if mu > 0 else 0.0
        excitacao[t] = alpha / beta if beta > 0 else 0.0
        lam_prev = lam_curr
        
    print("Calculando VR...")
    VR = pd.Series(R_t).rolling(window=JANELA_VOL).std(ddof=1).values
    vr_pips = VR * closes * 10000.0
    mask_op = df.index.isin(df_op.index)
    
    return closes, highs, lows, R_t, vr_pips, lambda_norm, excitacao, valido_hist, mask_op, n_candles


def main():
    os.makedirs(DIR_SAIDA, exist_ok=True)
    
    closes, highs, lows, R_t, vr_pips, lambda_norm, excitacao, valido_hist, mask_op, n_candles = pre_calcular_base()
    
    keys = list(GRID_PARAMS.keys())
    combinacoes = list(itertools.product(*[GRID_PARAMS[k] for k in keys]))
    
    print(f"Total absoluto de cenários Hawkes (Grid): {len(combinacoes)}")
    
    valido = valido_hist == 1
    lam_norm_prev = np.roll(lambda_norm, 1)
    lam_norm_prev[0] = np.nan
    norm_falling = lambda_norm < lam_norm_prev
    
    resultados = []
    
    pbar = tqdm(total=len(combinacoes), desc="Otimizando Grid")
    
    for comb in combinacoes:
        params = dict(zip(keys, comb))
        
        excit_ok = excitacao < params["excitacao_maxima"]
        norm_high = lambda_norm > params["lambda_norm_min_sinal"]
        
        cond_base = valido & excit_ok & mask_op & norm_falling & norm_high
        
        sinal = np.zeros(n_candles, dtype=np.int8)
        
        # Apostar em reversão
        mask_buy = cond_base & (R_t < 0)
        sinal[mask_buy] = 1
        
        mask_sell = cond_base & (R_t > 0)
        sinal[mask_sell] = -1
        
        entradas_idx = np.where(sinal != 0)[0]
        if len(entradas_idx) < 60:
            pbar.update(1)
            continue
            
        sl_arr = np.clip(params["multiplicador_sl"] * vr_pips, 3.0, 60.0)
        tp_arr = np.clip(params["multiplicador_tp"] * vr_pips, 4.5, 90.0)
        
        lucro_total_pips = 0.0
        max_drawdown_pips = 0.0
        pico_capital = 0.0
        capital = 0.0
        trades_count = 0
        trade_ativo_ate_indice = -1
        
        for i in entradas_idx:
            if i <= trade_ativo_ate_indice:
                continue
                
            direcao = sinal[i]
            preco_entrada = closes[i]
            
            if direcao == 1:
                preco_entrada += 0.00005
                sl_preco = preco_entrada - (sl_arr[i] / 10000.0)
                tp_preco = preco_entrada + (tp_arr[i] / 10000.0)
            else:
                preco_entrada -= 0.00005
                sl_preco = preco_entrada + (sl_arr[i] / 10000.0)
                tp_preco = preco_entrada - (tp_arr[i] / 10000.0)
                
            end_idx = min(i + 300, n_candles)
            fut_highs = highs[i+1:end_idx]
            fut_lows = lows[i+1:end_idx]
            fut_closes = closes[i+1:end_idx]
            
            if direcao == 1:
                hit_sl = np.where(fut_lows <= sl_preco)[0]
                hit_tp = np.where(fut_highs >= tp_preco)[0]
            else:
                hit_sl = np.where(fut_highs >= sl_preco)[0]
                hit_tp = np.where(fut_lows <= tp_preco)[0]
                
            idx_sl = hit_sl[0] if len(hit_sl) > 0 else 9999
            idx_tp = hit_tp[0] if len(hit_tp) > 0 else 9999
            
            min_idx = min(idx_sl, idx_tp)
            
            if min_idx == 9999:
                saida_preco = fut_closes[-1]
                trade_ativo_ate_indice = end_idx - 1
            elif min_idx == idx_sl:
                saida_preco = sl_preco
                trade_ativo_ate_indice = i + 1 + idx_sl
            elif min_idx == idx_tp:
                saida_preco = tp_preco
                trade_ativo_ate_indice = i + 1 + idx_tp
                
            if direcao == 1:
                pnl = (saida_preco - preco_entrada) * 10000.0
            else:
                pnl = (preco_entrada - saida_preco) * 10000.0
                
            lucro_total_pips += pnl
            trades_count += 1
            
            capital += pnl
            if capital > pico_capital:
                pico_capital = capital
            dd = pico_capital - capital
            if dd > max_drawdown_pips:
                max_drawdown_pips = dd
                
        pbar.update(1)
        
        if trades_count >= 60 and max_drawdown_pips > 0:
            ret_dd = lucro_total_pips / max_drawdown_pips
            resultados.append({
                "excitacao_maxima": params["excitacao_maxima"],
                "lambda_norm_min_sinal": params["lambda_norm_min_sinal"],
                "mult_sl": params["multiplicador_sl"],
                "mult_tp": params["multiplicador_tp"],
                "Trades": trades_count,
                "Lucro_Total_Pips": round(lucro_total_pips, 1),
                "Max_DD_Pips": round(max_drawdown_pips, 1),
                "Ret_DD": round(ret_dd, 2)
            })
            
    pbar.close()
    
    if resultados:
        df_res = pd.DataFrame(resultados)
        df_res = df_res.sort_values(by="Ret_DD", ascending=False).reset_index(drop=True)
        df_res.to_parquet(ARQUIVO_SAIDA)
        
        # Salvar as Top 10 em JSON no formato exigido pela esteira
        top10_list = []
        for i, row in df_res.head(10).iterrows():
            top10_list.append({
                "id": f"HAWKES_TOP{i+1}",
                "excitacao_maxima": float(row["excitacao_maxima"]),
                "lambda_norm_min_sinal": float(row["lambda_norm_min_sinal"]),
                "mult_sl": float(row["mult_sl"]),
                "mult_tp": float(row["mult_tp"]),
                "Trades": int(row["Trades"]),
                "Lucro_Total_Pips": float(row["Lucro_Total_Pips"]),
                "Max_DD_Pips": float(row["Max_DD_Pips"]),
                "Ret_DD": float(row["Ret_DD"])
            })
            
        json_saida = DIR_SAIDA / "otimizacao_hawkes_top10.json"
        with open(json_saida, 'w') as f:
            json.dump(top10_list, f, indent=4)
            
        print("\n================================================================================")
        print("TOP 10 PARAMETRIZAÇÕES HAWKES (RANKING POR RET/DD > 60 Trades):")
        print("================================================================================")
        print(df_res.head(10).to_string())
        print(f"\nResultados salvos em: {ARQUIVO_SAIDA}")
        print(f"Top 10 JSON salvo em: {json_saida}")
    else:
        print("Nenhuma combinação atingiu os critérios mínimos (60 trades).")

if __name__ == "__main__":
    main()
