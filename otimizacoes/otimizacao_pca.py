import os
import sys
import numpy as np
import pandas as pd
import itertools
from pathlib import Path
from tqdm import tqdm
import warnings
from numpy.lib.stride_tricks import sliding_window_view

warnings.filterwarnings("ignore")

# =============================================================================
# CONSTANTES E CAMINHOS
# =============================================================================
DIR_PROJETO = Path(r"c:\Users\cesar\.gemini\antigravity\scratch\Quant_Matematica.Trade")
DIR_DATA    = DIR_PROJETO / "quant_eurusd" / "data"
DIR_SAIDA   = DIR_DATA / "otimizacoes"

PARQUET_COMPLETO    = DIR_DATA / "eurusd_h1_completo.parquet"
PARQUET_OPERACIONAL = DIR_DATA / "eurusd_h1_operacional.parquet"
ARQUIVO_SAIDA       = DIR_SAIDA / "otimizacao_pca_resultados.parquet"

# =============================================================================
# GRID SEARCH
# =============================================================================
GRID_PARAMS = {
    "dominancia_minima": [0.40, 0.45, 0.50, 0.55],
    "zscore_threshold": [1.8, 2.0, 2.2, 2.5],
    "multiplicador_sl": [1.5, 2.0, 2.5],
    "multiplicador_tp": [2.0, 3.0, 4.0]
}

# =============================================================================
# CORE: CÁLCULO VETORIZADO DO PCA ROLANTE
# =============================================================================
def calcular_pca_rolante(features: np.ndarray, janela: int):
    n_candles, n_features = features.shape
    pca_z_pc1 = np.full(n_candles, np.nan, dtype=np.float32)
    pca_dominancia = np.full(n_candles, np.nan, dtype=np.float32)
    
    try:
        windows = sliding_window_view(features, window_shape=(janela, n_features)).squeeze()
    except Exception as e:
        print(f"Erro ao criar sliding window: {e}. Atualize o numpy.")
        raise
        
    pc1_scores_hist = np.full(n_candles, np.nan, dtype=np.float32)
    
    for i in range(windows.shape[0]):
        idx_atual = i + janela - 1
        X = windows[i]
        
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        stds[stds == 0] = 1.0 
        
        X_std = (X - means) / stds
        C = (1.0 / (janela - 1)) * (X_std.T @ X_std)
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        
        idx_sort = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx_sort]
        eigenvectors = eigenvectors[:, idx_sort]
        
        soma_evals = np.sum(eigenvalues)
        if soma_evals == 0:
            continue
            
        var_expl = eigenvalues / soma_evals
        dominancia = var_expl[0]
        
        pc1_vector = eigenvectors[:, 0]
        pc1_scores = X_std @ pc1_vector
        
        pc1_scores_hist[idx_atual] = pc1_scores[-1]
        pca_dominancia[idx_atual] = dominancia

    for i in range(windows.shape[0]):
        idx_atual = i + janela - 1
        X = windows[i]
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        stds[stds == 0] = 1.0
        X_std = (X - means) / stds
        C = (1.0 / (janela - 1)) * (X_std.T @ X_std)
        evals, evecs = np.linalg.eigh(C)
        idx_sort = np.argsort(evals)[::-1]
        evecs = evecs[:, idx_sort]
        pc1_vector = evecs[:, 0]
        pc1_scores = X_std @ pc1_vector
        
        mean_pc1 = np.mean(pc1_scores)
        std_pc1 = np.std(pc1_scores)
        if std_pc1 > 0:
            z_pc1 = (pc1_scores[-1] - mean_pc1) / std_pc1
            pca_z_pc1[idx_atual] = z_pc1
        else:
            pca_z_pc1[idx_atual] = 0.0

    return pca_z_pc1, pca_dominancia

def main():
    os.makedirs(DIR_SAIDA, exist_ok=True)
    
    print("Carregando dados...")
    df_comp = pd.read_parquet(PARQUET_COMPLETO)
    df_op = pd.read_parquet(PARQUET_OPERACIONAL)
    
    mask_op = df_comp.index.isin(df_op.index)
    
    closes = df_comp["Close"].values
    highs = df_comp["High"].values
    lows = df_comp["Low"].values
    
    # Gerar Features do PCA
    R_t = np.log(closes / np.roll(closes, 1))
    R_t[0] = 0.0
    R_t_sq = R_t ** 2
    R_t_abs = np.abs(R_t)
    R_t_lag1 = np.roll(R_t, 1); R_t_lag1[0] = 0.0
    R_t_lag2 = np.roll(R_t, 2); R_t_lag2[:2] = 0.0
    
    features = np.column_stack([R_t, R_t_sq, R_t_abs, R_t_lag1, R_t_lag2])
    n_len = len(closes)
    
    # VR fixa em 50 para otimização
    vr_series = pd.Series(R_t).rolling(window=50, min_periods=50).std(ddof=1).values
    vr_pips = vr_series * closes * 10000.0
    
    janelas_pca = [40, 60, 80, 100]
    
    keys = list(GRID_PARAMS.keys())
    combinacoes = list(itertools.product(*[GRID_PARAMS[k] for k in keys]))
    
    print(f"Total de combinações internas por janela: {len(combinacoes)}")
    print(f"Total absoluto de cenários (Janelas x Combinações): {len(janelas_pca) * len(combinacoes)}")
    
    resultados = []
    
    for janela_pca in janelas_pca:
        print(f"\n[+] Iniciando cálculo da Janela PCA = {janela_pca} (Processamento Pesado)...")
        pca_z_pc1, pca_dominancia = calcular_pca_rolante(features, janela_pca)
        
        pbar = tqdm(total=len(combinacoes), desc=f"Otimizando Internamente (Janela {janela_pca})")
        
        for comb in combinacoes:
            params = dict(zip(keys, comb))
            
            cond_comum = (pca_dominancia >= params["dominancia_minima"]) & mask_op
            
            cond_compra = cond_comum & (pca_z_pc1 <= -params["zscore_threshold"])
            cond_venda = cond_comum & (pca_z_pc1 >= params["zscore_threshold"])
            
            sinal = np.zeros(n_len, dtype=np.int8)
            sinal[cond_compra] = 1
            sinal[cond_venda] = -1
            
            entradas_idx = np.where(sinal != 0)[0]
            if len(entradas_idx) < 30:
                pbar.update(1)
                continue
                
            sl_arr = np.clip(params["multiplicador_sl"] * vr_pips, 3.0, 60.0)
            tp_arr = np.clip(params["multiplicador_tp"] * vr_pips, 4.5, 90.0)
            
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
                    
                direcao = sinal[i]
                preco_entrada = closes[i]
                
                # Spread
                if direcao == 1:
                    preco_entrada += 0.00005
                    sl_preco = preco_entrada - (sl_arr[i] / 10000.0)
                    tp_preco = preco_entrada + (tp_arr[i] / 10000.0)
                else:
                    preco_entrada -= 0.00005
                    sl_preco = preco_entrada + (sl_arr[i] / 10000.0)
                    tp_preco = preco_entrada - (tp_arr[i] / 10000.0)
                    
                end_idx = min(i + 300, n_len)
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
                    "janela_pca": janela_pca,
                    "dominancia_minima": params["dominancia_minima"],
                    "zscore_threshold": params["zscore_threshold"],
                    "mult_sl": params["multiplicador_sl"],
                    "mult_tp": params["multiplicador_tp"],
                    "Trades": trades_count,
                    "Lucro_Total_Pips": round(lucro_total_pips, 1),
                    "Max_DD_Pips": round(max_drawdown_pips, 1),
                    "Ret_DD": round(ret_dd, 2),
                    "Profit_Factor": round(profit_factor, 2)
                })
                
        pbar.close()
    
    if resultados:
        df_res = pd.DataFrame(resultados)
        df_res = df_res.sort_values(by="Ret_DD", ascending=False).reset_index(drop=True)
        df_res.to_parquet(ARQUIVO_SAIDA)
        print("\n================================================================================")
        print("TOP 10 PARAMETRIZAÇÕES (RANKING POR RET/DD):")
        print("================================================================================")
        print(df_res.head(10).to_string())
        print(f"\nResultados salvos em: {ARQUIVO_SAIDA}")
    else:
        print("Nenhuma combinação atingiu os critérios mínimos (30 trades).")

if __name__ == "__main__":
    main()
