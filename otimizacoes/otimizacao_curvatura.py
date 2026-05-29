import os
import sys
import numpy as np
import pandas as pd
import itertools
import time
from pathlib import Path
from tqdm import tqdm
import warnings
import argparse
import json

warnings.filterwarnings("ignore")

# =============================================================================
# =============================================================================
# CLI & CAMINHOS GENERICOS
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--ativo", type=str, required=True, help="Ex: eurusd")
parser.add_argument("--timeframe", type=str, required=True, help="Ex: h1")
args = parser.parse_args()

ativo = args.ativo.lower()
timeframe = args.timeframe.lower()
estrategia_nome = Path(__file__).stem.replace('otimizacao_', '')

DIR_PROJETO = Path(__file__).resolve().parent.parent
DIR_DATA    = DIR_PROJETO / f"quant_{ativo}_{timeframe}" / "data"
DIR_SAIDA   = DIR_DATA / "otimizacoes"

PARQUET_COMPLETO    = DIR_DATA / f"{ativo}_{timeframe}_completo.parquet"
PARQUET_OPERACIONAL = DIR_DATA / f"{ativo}_{timeframe}_operacional.parquet"
ARQUIVO_SAIDA       = DIR_SAIDA / f"otimizacao_{estrategia_nome}_resultados.parquet"
ARQUIVO_JSON        = DIR_SAIDA / f"otimizacao_{estrategia_nome}_top10.json"

# Verifica se existe a pasta de saida
os.makedirs(DIR_SAIDA, exist_ok=True)

# GRID SEARCH
# =============================================================================
GRID_PARAMS = {
    "janela_norm": [50, 80, 100, 120],
    "janela_vol": [20, 30, 50, 80],
    "multiplicador_sl": [1.5, 2.0, 2.5],
    "multiplicador_tp": [2.0, 3.0, 3.5, 4.0]
}

def main():
    os.makedirs(DIR_SAIDA, exist_ok=True)
    
    print("Carregando dados...")
    if not PARQUET_COMPLETO.exists():
        raise FileNotFoundError(f"Arquivo completo nao encontrado: {PARQUET_COMPLETO}")
    if not PARQUET_OPERACIONAL.exists():
        raise FileNotFoundError(f"Arquivo operacional nao encontrado: {PARQUET_OPERACIONAL}")
        
    df_comp = pd.read_parquet(PARQUET_COMPLETO)
    df_op = pd.read_parquet(PARQUET_OPERACIONAL)
    
    mask_op = df_comp.index.isin(df_op.index)
    if isinstance(mask_op, pd.Series) or hasattr(mask_op, 'to_numpy'):
        mask_op = mask_op.to_numpy()
        
    closes = df_comp["Close"].values.astype(np.float32)
    highs = df_comp["High"].values.astype(np.float32)
    lows = df_comp["Low"].values.astype(np.float32)
    opens = df_comp["Open"].values.astype(np.float32)
    
    # 1. Referencial 3D e calculo de kappa e tau (Brutos)
    print("Mapeando espaco 3D e extraindo derivadas centrais...")
    n = len(df_comp)
    R = np.log(closes / np.roll(closes, 1))
    R[0] = 0.0
    
    T_x = np.ones(n, dtype=np.float32)
    T_y = np.zeros(n, dtype=np.float32)
    T_z = np.zeros(n, dtype=np.float32)
    
    A_x = np.zeros(n, dtype=np.float32)
    A_y = np.zeros(n, dtype=np.float32)
    A_z = np.zeros(n, dtype=np.float32)
    
    # Diferenças para Trás Causal (Sem Look-Ahead Bias)
    T_y[1:] = closes[1:] - closes[:-1]
    T_z[1:] = R[1:] - R[:-1]
    
    A_y[2:] = closes[2:] - 2 * closes[1:-1] + closes[:-2]
    A_z[2:] = R[2:] - 2 * R[1:-1] + R[:-2]
    
    T = np.column_stack((T_x, T_y, T_z))
    A = np.column_stack((A_x, A_y, A_z))
    
    norm_T = np.linalg.norm(T, axis=1)
    T_hat = T / norm_T[:, np.newaxis]
    
    print("Calculando Curvatura kappa...")
    TxA = np.cross(T, A)
    norm_TxA = np.linalg.norm(TxA, axis=1)
    kappa = norm_TxA / (norm_T ** 3)
    
    print("Calculando Torcao tau...")
    dT_hat = np.zeros_like(T_hat)
    dT_hat[1:] = T_hat[1:] - T_hat[:-1]
    
    norm_dT_hat = np.linalg.norm(dT_hat, axis=1)
    mask_zero = norm_dT_hat == 0
    norm_dT_hat[mask_zero] = 1e-9
    
    N = dT_hat / norm_dT_hat[:, np.newaxis]
    B = np.cross(T, N)
    
    dB = np.zeros_like(B)
    dB[1:] = B[1:] - B[:-1]
    
    tau = -np.sum(N * dB, axis=1)
    
    # Pre-calcular SMA 50 do Close
    sma50 = pd.Series(closes).rolling(50, min_periods=1).mean().values
    
    # Dicionario de normalizacoes por Janela
    print("Pre-calculando matrizes de Z-Score para kappa e tau...")
    normalizacoes = {}
    for j_n in GRID_PARAMS["janela_norm"]:
        k_series = pd.Series(kappa)
        k_roll = k_series.rolling(j_n, min_periods=1)
        k_mean = k_roll.mean().values
        k_std = k_roll.std().replace(0, 1e-9).fillna(1e-9).values
        kappa_norm = (kappa - k_mean) / k_std
        
        t_series = pd.Series(tau)
        t_roll = t_series.rolling(j_n, min_periods=1)
        t_mean = t_roll.mean().values
        t_std = t_roll.std().replace(0, 1e-9).fillna(1e-9).values
        tau_norm = (tau - t_mean) / t_std
        
        # Calcular os picos geometricos para esta janela
        kn_t1 = np.concatenate(([0], kappa_norm[:-1]))
        kn_t2 = np.concatenate(([0, 0], kappa_norm[:-2]))
        # Garantir bordas limpas
        kn_t1[0] = 0
        kn_t2[0] = 0; kn_t2[1] = 0
        
        pico_mask = (kn_t1 > kn_t2) & (kn_t1 > kappa_norm) & (kn_t1 > 2.0)
        
        A_y_t1 = np.concatenate(([0], A_y[:-1]))
        A_y_t1[0] = 0
        
        direcao = np.zeros(n, dtype=np.int8)
        direcao[pico_mask & (A_y_t1 > 0)] = 1
        direcao[pico_mask & (A_y_t1 < 0)] = -1
        
        normalizacoes[j_n] = {
            "tau_norm": tau_norm,
            "pico_mask": pico_mask,
            "direcao": direcao
        }
        
    # Dicionario de Volatilidade por Janela
    print("Pre-calculando matrizes de Volatilidade...")
    volatilidades = {}
    s_lr = pd.Series(R)
    for j_v in GRID_PARAMS["janela_vol"]:
        vr = s_lr.rolling(window=j_v, min_periods=j_v).std(ddof=1).values
        vr_pips = vr * closes * 10000.0
        volatilidades[j_v] = vr_pips

    # Gerar combinacoes
    keys = ["janela_norm", "janela_vol", "multiplicador_sl", "multiplicador_tp"]
    combinacoes = list(itertools.product(*[GRID_PARAMS[k] for k in keys]))
    total_comb = len(combinacoes)
    print(f"Total de combinacoes a simular: {total_comb}")
    
    resultados = []
    t0 = time.time()
    
    print("\nIniciando Processamento Vetorizado...")
    pbar = tqdm(total=total_comb, desc="Otimizando Curvatura")
    
    for idx, (j_n, j_v, m_sl, m_tp) in enumerate(combinacoes):
        norm_data = normalizacoes[j_n]
        tau_norm = norm_data["tau_norm"]
        pico_mask = norm_data["pico_mask"]
        direcao = norm_data["direcao"]
        
        vr_pips = volatilidades[j_v]
        
        tau_norm_t1 = np.concatenate(([0], tau_norm[:-1]))
        tau_norm_t1[0] = 0
        
        cond_comum = mask_op & pico_mask
        cond_compra = cond_comum & (direcao == 1) & (tau_norm_t1 > 0) & (closes < sma50)
        cond_venda = cond_comum & (direcao == -1) & (tau_norm_t1 < 0) & (closes > sma50)
        
        sinais = np.zeros(n, dtype=np.int8)
        sinais[cond_compra] = 1
        sinais[cond_venda] = -1
        
        entradas_idx = np.where(sinais != 0)[0]
        if len(entradas_idx) < 30:
            pbar.update(1)
            continue
            
        sl_arr = np.clip(m_sl * vr_pips, 3.0, 60.0)
        tp_arr = np.clip(m_tp * vr_pips, 4.5, 90.0)
        
        lucro_total_pips = 0.0
        max_drawdown_pips = 0.0
        pico_capital = 0.0
        capital = 0.0
        trades_count = 0
        soma_ganhos = 0.0
        soma_perdas = 0.0
        
        trade_ativo_ate_indice = -1
        
        for i in entradas_idx:
            if i <= trade_ativo_ate_indice:
                continue
                
            if i + 1 >= n:
                continue
                
            dir_trade = sinais[i]
            preco_entrada = opens[i+1] # Entrada no open do proximo candle
            
            # SL e TP
            vr_atual = vr_pips[i]
            if np.isnan(vr_atual) or vr_atual <= 0:
                continue
                
            sl_pips = sl_arr[i]
            tp_pips = tp_arr[i]
            
            delta_sl = sl_pips / 10000.0
            delta_tp = tp_pips / 10000.0
            
            if dir_trade == 1:
                sl_preco = preco_entrada - delta_sl
                tp_preco = preco_entrada + delta_tp
            else:
                sl_preco = preco_entrada + delta_sl
                tp_preco = preco_entrada - delta_tp
                
            end_idx = min(i + 300, n)
            fut_highs = highs[i+1:end_idx]
            fut_lows = lows[i+1:end_idx]
            fut_closes = closes[i+1:end_idx]
            
            if dir_trade == 1:
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
                
            # Calculo do PnL
            if dir_trade == 1:
                pnl = (saida_preco - preco_entrada) * 10000.0
            else:
                pnl = (preco_entrada - saida_preco) * 10000.0
                
            pnl -= 0.5 # Spread
            lucro_total_pips += pnl
            trades_count += 1
            
            if pnl > 0:
                soma_ganhos += pnl
            else:
                soma_perdas += abs(pnl)
                
            capital += pnl
            if capital > pico_capital:
                pico_capital = capital
            dd = pico_capital - capital
            if dd > max_drawdown_pips:
                max_drawdown_pips = dd
                
        pbar.update(1)
        
        if trades_count >= 30 and max_drawdown_pips > 0:
            ret_dd = lucro_total_pips / max_drawdown_pips
            profit_factor = (soma_ganhos / soma_perdas) if soma_perdas > 0 else 99.0
            
            resultados.append({
                "janela_norm": j_n,
                "janela_vol": j_v,
                "mult_sl": m_sl,
                "mult_tp": m_tp,
                "Trades": trades_count,
                "Lucro_Total_Pips": round(lucro_total_pips, 1),
                "Max_DD_Pips": round(max_drawdown_pips, 1),
                "Ret_DD": round(ret_dd, 2),
                "Profit_Factor": round(profit_factor, 2)
            })
            
    pbar.close()
    t1 = time.time()
    print(f"\nOtimizacao concluida em {t1 - t0:.2f} segundos!")
    
    if resultados:
        df_res = pd.DataFrame(resultados)
        # Filtro de Sobrevivencia (Trades >= 60 e F.R. > 1.0)
        mask_survivor = (df_res["Trades"] >= 60) & (df_res["Profit_Factor"] > 1.0)
        df_res = df_res[mask_survivor]
        
        df_res = df_res.sort_values(by="Ret_DD", ascending=False).reset_index(drop=True)
        
        if len(df_res) > 0:
            df_res.to_parquet(ARQUIVO_SAIDA)
            
            top10 = df_res.head(10).to_dict(orient="records")
            # Adiciona um ID a cada parametro
            for i, p in enumerate(top10):
                p["id"] = f"{estrategia_nome.upper()}_TOP{i+1}"
                
            with open(ARQUIVO_JSON, 'w') as f:
                json.dump(top10, f, indent=4)
                
            print("\n================================================================================")
            print("TOP 10 PARAMETRIZACOES SOBREVIVENTES (RANKING POR RET/DD):")
            print("================================================================================")
            print(df_res.head(10).to_string())
            print(f"\nResultados salvos em: {ARQUIVO_SAIDA} e {ARQUIVO_JSON}")
        else:
            print("Nenhuma combinacao sobreviveu aos criterios rigorosos (Min 60 Trades, F.R > 1.0).")
    else:
        print("Nenhuma combinacao atingiu os criterios minimos (30 trades).")

if __name__ == "__main__":
    main()
