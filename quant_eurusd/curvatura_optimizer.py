import os
import sys
import numpy as np
import pandas as pd
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time

DIR_DATA = Path("c:/Users/cesar/.gemini/antigravity/scratch/Quant_Matematica.Trade/quant_eurusd/data")
PARQUET_COMPLETO = DIR_DATA / "eurusd_h1_completo.parquet"
PARQUET_OP = DIR_DATA / "eurusd_h1_operacional.parquet"

def fast_backtest(sinais, C, H, L, sl_pips, tp_pips):
    capital_inicial = 10000.0
    capital = capital_inicial
    pnl_total = 0.0
    pnl_max = 0.0
    dd_max = 0.0
    ops = 0
    wins = 0
    
    pos = 0
    p_ent = 0.0
    p_sl = 0.0
    p_tp = 0.0
    lot = 0.0
    
    n = len(sinais)
    for i in range(n):
        if pos != 0:
            fechar = False
            p_saida = 0.0
            if pos == 1:
                if L[i] <= p_sl:
                    fechar = True
                    p_saida = p_sl
                elif H[i] >= p_tp:
                    fechar = True
                    p_saida = p_tp
            else:
                if H[i] >= p_sl:
                    fechar = True
                    p_saida = p_sl
                elif L[i] <= p_tp:
                    fechar = True
                    p_saida = p_tp
            
            if fechar:
                pnl = pos * (p_saida - p_ent) * 10000.0
                pnl_net = (pnl * lot * 10.0) - (1.0 * lot * 10.0)
                capital += pnl_net
                pnl_total += pnl_net
                ops += 1
                if pnl_net > 0: wins += 1
                if pnl_total > pnl_max: pnl_max = pnl_total
                dd = (capital - (capital_inicial + pnl_max)) / (capital_inicial + pnl_max) * 100.0
                if dd < dd_max: dd_max = dd
                pos = 0
                
        if pos == 0 and sinais[i] != 0 and i < n-1:
            pos = sinais[i]
            p_ent = C[i]
            s = sl_pips[i]
            t = tp_pips[i]
            lot = max(0.01, (capital * 0.01) / (s * 10.0))
            if pos == 1:
                p_sl = p_ent - (s / 10000.0)
                p_tp = p_ent + (t / 10000.0)
            else:
                p_sl = p_ent + (s / 10000.0)
                p_tp = p_ent - (t / 10000.0)
                
    wr = (wins / ops * 100.0) if ops > 0 else 0.0
    pnl_perc = (capital - capital_inicial) / capital_inicial * 100.0
    rf = abs(pnl_perc / dd_max) if dd_max < 0 else 0.0
    return ops, wr, pnl_perc, dd_max, rf

def simulate_params(params, df_dict):
    z_win, k_thresh, sl_m, tp_m = params
    
    kappa = df_dict["kappa"]
    tau = df_dict["tau"]
    C = df_dict["C"]
    H = df_dict["H"]
    L = df_dict["L"]
    mask_op = df_dict["mask_op"]
    sma50 = df_dict["sma50"]
    A_y = df_dict["A_y"]
    vr_pips_base = df_dict["vr_pips_base"]
    
    n = len(C)
    
    k_s = pd.Series(kappa)
    t_s = pd.Series(tau)
    kappa_norm = (k_s - k_s.rolling(z_win).mean()) / k_s.rolling(z_win).std().replace(0, 1e-9)
    tau_norm = (t_s - t_s.rolling(z_win).mean()) / t_s.rolling(z_win).std().replace(0, 1e-9)
    
    kn = kappa_norm.fillna(0).values
    tn = tau_norm.fillna(0).values
    tn_prev = np.roll(tn, 1)
    tn_prev[0] = 0
    
    cross_up = (tn > 0) & (tn_prev <= 0)
    cross_dn = (tn < 0) & (tn_prev >= 0)
    
    cond_compra = mask_op & (kn > k_thresh) & cross_up & (A_y > 0) & (C < sma50)
    cond_venda = mask_op & (kn > k_thresh) & cross_dn & (A_y < 0) & (C > sma50)
    
    sinais = np.zeros(n, dtype=np.int8)
    sinais[cond_compra] = 1
    sinais[cond_venda] = -1
    
    sl_pips = np.clip(sl_m * vr_pips_base, 3.0, 60.0)
    tp_pips = np.clip(tp_m * vr_pips_base, 4.5, 90.0)
    sl_pips = pd.Series(sl_pips).fillna(10.0).values
    tp_pips = pd.Series(tp_pips).fillna(15.0).values
    
    ops, wr, pnl, dd, rf = fast_backtest(sinais, C, H, L, sl_pips, tp_pips)
    
    return {
        "z_win": z_win, "k_thresh": k_thresh, "sl_m": sl_m, "tp_m": tp_m,
        "Ops": ops, "WR": wr, "PnL": pnl, "DD": dd, "RF": rf
    }

def main():
    print("Carregando bases para Otimização da Curvatura (Rota 3)...")
    df = pd.read_parquet(PARQUET_COMPLETO)
    df_op = pd.read_parquet(PARQUET_OP)
    mask_op = df.index.isin(df_op.index)
    
    C = df["Close"].values
    H = df["High"].values
    L = df["Low"].values
    R = np.log(C / np.roll(C, 1))
    R[0] = 0.0
    n = len(C)
    
    T_x = np.ones(n, dtype=np.float32)
    T_y = np.zeros(n, dtype=np.float32)
    T_z = np.zeros(n, dtype=np.float32)
    A_x = np.zeros(n, dtype=np.float32)
    A_y = np.zeros(n, dtype=np.float32)
    A_z = np.zeros(n, dtype=np.float32)
    
    T_y[1:] = C[1:] - C[:-1]
    T_z[1:] = R[1:] - R[:-1]
    A_y[2:] = C[2:] - 2*C[1:-1] + C[:-2]
    A_z[2:] = R[2:] - 2*R[1:-1] + R[:-2]
    
    T = np.column_stack((T_x, T_y, T_z))
    A = np.column_stack((A_x, A_y, A_z))
    
    norm_T = np.linalg.norm(T, axis=1)
    T_hat = T / norm_T[:, np.newaxis]
    TxA = np.cross(T, A)
    kappa = np.linalg.norm(TxA, axis=1) / (norm_T ** 3)
    
    dT_hat = np.zeros_like(T_hat)
    dT_hat[1:] = T_hat[1:] - T_hat[:-1]
    norm_dT = np.linalg.norm(dT_hat, axis=1)
    norm_dT[norm_dT == 0] = 1e-9
    N = dT_hat / norm_dT[:, np.newaxis]
    B = np.cross(T, N)
    dB = np.zeros_like(B)
    dB[1:] = B[1:] - B[:-1]
    tau = -np.sum(N * dB, axis=1)
    
    sma50 = pd.Series(C).rolling(50).mean().values
    vr = pd.Series(R).rolling(50).std(ddof=1).values
    vr_pips_base = vr * C * 10000.0
    
    df_dict = {
        "kappa": kappa, "tau": tau, "C": C, "H": H, "L": L,
        "mask_op": mask_op, "sma50": sma50, "A_y": A_y, "vr_pips_base": vr_pips_base
    }
    
    z_wins = [50, 100, 200]
    k_threshs = [1.0, 1.25, 1.5, 1.75, 2.0]
    sl_mults = [1.0, 1.5, 2.0, 2.5]
    tp_mults = [2.0, 3.0, 3.5, 4.0, 5.0]
    
    grids = list(product(z_wins, k_threshs, sl_mults, tp_mults))
    print(f"Iniciando Grid Search: {len(grids)} combinações...")
    
    results = []
    t0 = time.time()
    
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        futures = [executor.submit(simulate_params, g, df_dict) for g in grids]
        for idx, fut in enumerate(as_completed(futures), 1):
            results.append(fut.result())
            if idx % 50 == 0:
                print(f"Progresso: {idx}/{len(grids)}")
                
    print(f"Tempo total: {time.time() - t0:.2f} segundos")
    
    df_res = pd.DataFrame(results)
    df_valid = df_res[(df_res["Ops"] >= 100) & (df_res["PnL"] > 0)]
    
    if df_valid.empty:
        print("Nenhuma combinação lucrativa encontrou >= 100 operações.")
        df_valid = df_res[df_res["Ops"] >= 50]
        
    df_rf = df_valid.sort_values("RF", ascending=False).head(10)
    df_pnl = df_valid.sort_values("PnL", ascending=False).head(10)
    
    print("\n--- TOP 10 POR FATOR DE RECUPERAÇÃO (RF) ---")
    print(df_rf.to_string(index=False))
    
    print("\n--- TOP 10 POR PNL (%) ---")
    print(df_pnl.to_string(index=False))

if __name__ == "__main__":
    main()
