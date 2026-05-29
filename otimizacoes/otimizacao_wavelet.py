import os
import sys
import numpy as np
import pandas as pd
import pywt
import itertools
from pathlib import Path
from tqdm import tqdm
import warnings
import argparse
import json

warnings.filterwarnings("ignore")

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

# =============================================================================
# GRID SEARCH
# =============================================================================
GRID_PARAMS = {
    "janela_smooth": [3, 5, 8],
    "janela_energia": [50, 100, 150],
    "coerencia_cutoff": [0.40, 0.50, 0.60, 0.70],
    "energia_quantile": [0.30, 0.40, 0.50],
    "pot_s1_cutoff": [0.6, 0.8, 1.0],
    "multiplicador_sl": [1.5, 2.0, 2.5],
    "multiplicador_tp": [2.5, 3.5, 4.5]
}

ESCALAS = [4, 12, 24, 120]
WAVELET_TYPE = 'cmor1.5-1.0'

def main():
    os.makedirs(DIR_SAIDA, exist_ok=True)
    
    print("Carregando dados...")
    df_comp = pd.read_parquet(PARQUET_COMPLETO)
    df_op = pd.read_parquet(PARQUET_OPERACIONAL)
    
    mask_op = df_comp.index.isin(df_op.index)
    
    closes = df_comp["Close"].values
    highs = df_comp["High"].values
    lows = df_comp["Low"].values
    log_ret = np.log(closes / np.roll(closes, 1))
    log_ret[0] = 0.0
    
    ret_neg = log_ret < 0
    ret_pos = log_ret > 0
    
    n_len = len(closes)
    
    # ── 1. PRE-COMPUTE WAVELET CWT ──
    print("Calculando CWT, Fases e Potências Iniciais (PyWavelets)...")
    coefs, freqs = pywt.cwt(log_ret, ESCALAS, WAVELET_TYPE)
    potencia = np.abs(coefs)**2
    pot_norm = potencia / (np.sum(potencia, axis=0) + 1e-15)
    fase = np.angle(coefs)
    
    wav_energia_total = np.sum(potencia, axis=0)
    
    fase_unwrapped = np.unwrap(fase, axis=1)
    d_fase_all = np.zeros_like(fase_unwrapped)
    d_fase_all[:, 1:] = np.diff(fase_unwrapped, axis=1)
    
    wav_dom_escala = np.argmax(potencia, axis=0)
    d_fase = d_fase_all[wav_dom_escala, np.arange(n_len)]
    
    d_fase_prev = np.roll(d_fase, 1)
    d_fase_prev[0] = 0
    
    inflex_up = (d_fase > 0) & (d_fase_prev <= 0)
    inflex_dn = (d_fase < 0) & (d_fase_prev >= 0)
    
    cross_12 = coefs[0] * np.conj(coefs[1])
    cross_12_real = np.real(cross_12)
    cross_12_imag = np.imag(cross_12)
    S11_raw = potencia[0]
    S22_raw = potencia[1]
    pot_norm_s1 = pot_norm[0]
    
    # VR Constants
    vr_series = pd.Series(log_ret).rolling(window=50, min_periods=50).std(ddof=1).values
    vr_pips = vr_series * closes * 10000.0
    
    # Combinacoes
    keys = list(GRID_PARAMS.keys())
    combinacoes = list(itertools.product(*[GRID_PARAMS[k] for k in keys]))
    
    print(f"Total de combinações do Wavelet a simular: {len(combinacoes)}")
    
    resultados = []
    
    # Otimização Loop
    pbar = tqdm(total=len(combinacoes), desc="Otimizando Wavelet")
    
    # Pre-cache de smoothings para evitar recalculá-los a cada comb
    cache_smooth = {}
    for js in GRID_PARAMS["janela_smooth"]:
        S12_r = pd.Series(cross_12_real).rolling(js, min_periods=1).mean().values
        S12_i = pd.Series(cross_12_imag).rolling(js, min_periods=1).mean().values
        S11 = pd.Series(S11_raw).rolling(js, min_periods=1).mean().values
        S22 = pd.Series(S22_raw).rolling(js, min_periods=1).mean().values
        S12_sq = S12_r**2 + S12_i**2
        coerencia_12 = S12_sq / (S11 * S22 + 1e-15)
        cache_smooth[js] = coerencia_12
        
    cache_energia = {}
    for je in GRID_PARAMS["janela_energia"]:
        for eq in GRID_PARAMS["energia_quantile"]:
            thresh = pd.Series(wav_energia_total).rolling(je, min_periods=1).quantile(eq).values
            energ_ok = wav_energia_total > thresh
            cache_energia[(je, eq)] = energ_ok

    for comb in combinacoes:
        params = dict(zip(keys, comb))
        
        coerencia_12 = cache_smooth[params["janela_smooth"]]
        energ_ok = cache_energia[(params["janela_energia"], params["energia_quantile"])]
        
        coer_ok = coerencia_12 >= params["coerencia_cutoff"]
        pot_s1_ok = pot_norm_s1 < params["pot_s1_cutoff"]
        
        cond_compra = mask_op & coer_ok & energ_ok & inflex_up & pot_s1_ok & ret_neg
        cond_venda  = mask_op & coer_ok & energ_ok & inflex_dn & pot_s1_ok & ret_pos
        
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
        
        # Arrays para Saída Neutra
        inflex_up_indices = np.where(inflex_up)[0]
        inflex_dn_indices = np.where(inflex_dn)[0]
        
        for i in entradas_idx:
            if i <= trade_ativo_ate_indice:
                continue
                
            direcao = sinal[i]
            preco_entrada = closes[i]
            
            # Aplicar spread real de 0.5
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
                # Saída neutra COMPRA = inflexão para baixo
                nt_mask = (inflex_dn_indices > i) & (inflex_dn_indices < end_idx)
                hit_nt = inflex_dn_indices[nt_mask] - (i + 1)
            else:
                hit_sl = np.where(fut_highs >= sl_preco)[0]
                hit_tp = np.where(fut_lows <= tp_preco)[0]
                # Saída neutra VENDA = inflexão para cima
                nt_mask = (inflex_up_indices > i) & (inflex_up_indices < end_idx)
                hit_nt = inflex_up_indices[nt_mask] - (i + 1)
                
            idx_sl = hit_sl[0] if len(hit_sl) > 0 else 9999
            idx_tp = hit_tp[0] if len(hit_tp) > 0 else 9999
            idx_nt = hit_nt[0] if len(hit_nt) > 0 else 9999
            
            min_idx = min(idx_sl, idx_tp, idx_nt)
            
            if min_idx == 9999:
                saida_preco = fut_closes[-1]
                trade_ativo_ate_indice = end_idx - 1
            elif min_idx == idx_sl:
                saida_preco = sl_preco
                trade_ativo_ate_indice = i + 1 + idx_sl
            elif min_idx == idx_tp:
                saida_preco = tp_preco
                trade_ativo_ate_indice = i + 1 + idx_tp
            else:
                saida_preco = fut_closes[idx_nt]
                trade_ativo_ate_indice = i + 1 + idx_nt
                
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
                "janela_smooth": params["janela_smooth"],
                "janela_energia": params["janela_energia"],
                "coerencia_cutoff": params["coerencia_cutoff"],
                "energia_quantile": params["energia_quantile"],
                "pot_s1_cutoff": params["pot_s1_cutoff"],
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
        # Filtro de Sobrevivencia (Trades >= 60 e F.R. > 1.0)
        mask_survivor = (df_res["Trades"] >= 60) & (df_res["Profit_Factor"] > 1.0)
        df_res = df_res[mask_survivor]
        df_res = df_res.sort_values(by="Ret_DD", ascending=False).reset_index(drop=True)

        if len(df_res) > 0:
            df_res.to_parquet(ARQUIVO_SAIDA)

            top10 = df_res.head(10).to_dict(orient="records")
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
