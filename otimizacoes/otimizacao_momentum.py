import os
import sys
import numpy as np
import pandas as pd
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
estrategia_nome = "momentum"

DIR_PROJETO = Path(__file__).resolve().parent.parent
DIR_DATA    = DIR_PROJETO / f"quant_{ativo}_{timeframe}" / "data"
DIR_SAIDA   = DIR_DATA / "otimizacoes"

PARQUET_ENTRADA = DIR_DATA / f"{ativo}_{timeframe}_hurst.parquet"
ARQUIVO_SAIDA   = DIR_SAIDA / f"otimizacao_{estrategia_nome}_resultados.parquet"
ARQUIVO_JSON    = DIR_SAIDA / f"otimizacao_{estrategia_nome}_top10.json"

# =============================================================================
# GRID SEARCH
# =============================================================================
GRID_PARAMS = {
    "hurst_cutoff": [0.50, 0.52, 0.55],
    "entropia_cutoff": [0.55, 0.60, 0.65],
    "percentil_long_trigger": [0.75, 0.80, 0.85],
    "multiplicador_sl": [1.5, 2.0, 2.5],
    "multiplicador_tp": [3.5, 4.5, 5.5, 6.5]
}

# =============================================================================
# FUNÇÕES DE CÁLCULO BASE
# =============================================================================
def _percentrank(arr: np.ndarray) -> float:
    if len(arr) < 2:
        return np.nan
    val = arr[-1]
    n_historico = len(arr) - 1
    n_menores = np.sum(arr[:-1] < val)
    return float(n_menores) / float(n_historico)

def _calcular_entropia_janela(arr: np.ndarray) -> float:
    if len(arr) < 30:
        return np.nan
    s_std = arr.std(ddof=1)
    if s_std < 1e-15:
        return 0.0
    contagens, _ = np.histogram(arr, bins=10)
    p_i = contagens / 30.0
    p_i = p_i[p_i > 0.0]
    h_shannon = -np.sum(p_i * np.log2(p_i))
    h_max = np.log2(10.0)
    h_norm = h_shannon / h_max
    return float(np.clip(h_norm, 0.0, 1.0))

def verificar_janela_operacional(dt_index: pd.DatetimeIndex) -> pd.Series:
    weekday = dt_index.weekday
    hora    = dt_index.strftime('%H:%M')
    seg_sex      = (weekday >= 0) & (weekday <= 4)
    horario_op   = (hora >= "10:00") & (hora <= "22:30")
    return pd.Series(seg_sex & horario_op, index=dt_index)

def main():
    os.makedirs(DIR_SAIDA, exist_ok=True)
    
    print("Carregando dados...")
    df = pd.read_parquet(PARQUET_ENTRADA)
    
    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values
    opens = df["Open"].values
    n_len = len(closes)
    
    print("Calculando indicadores base (Velocidade, Aceleração, Percentil, Entropia e VR)...")
    
    # 1. Velocidade e Aceleração
    df["velocidade"] = df["Close"].diff(1).astype("float32")
    df["aceleracao"] = df["velocidade"].diff(1).astype("float32")
    
    # 2. Percentil de Aceleração
    df["percentil_acel"] = df["aceleracao"].rolling(window=100, min_periods=100).apply(_percentrank, raw=True).astype("float32")
    
    # 3. Entropia
    if "log_return" not in df.columns:
        log_return = np.log(closes / np.roll(closes, 1))
        log_return[0] = 0.0
        df["log_return"] = log_return
        
    df["entropia_shannon"] = df["log_return"].rolling(window=30, min_periods=30).apply(_calcular_entropia_janela, raw=True).astype("float32")
    
    # 4. Volatilidade Realizada (VR)
    vr_series = df["log_return"].rolling(window=50, min_periods=50).std(ddof=1).values
    vr_pips = vr_series * closes * 10000.0
    
    # 5. Janela Operacional e Hurst
    op_window = verificar_janela_operacional(df.index).values
    hurst_vals = df["hurst"].values
    entropia_vals = df["entropia_shannon"].values
    percentil_vals = df["percentil_acel"].values
    velocidade_vals = df["velocidade"].values
    
    keys = list(GRID_PARAMS.keys())
    combinacoes = list(itertools.product(*[GRID_PARAMS[k] for k in keys]))
    
    print(f"Total absoluto de cenários (Grid Completo): {len(combinacoes)}")
    
    resultados = []
    
    pbar = tqdm(total=len(combinacoes), desc="Otimizando Grid")
    
    for comb in combinacoes:
        params = dict(zip(keys, comb))
        
        # Filtros de Entrada
        cond_operacional = (hurst_vals > params["hurst_cutoff"]) & (entropia_vals < params["entropia_cutoff"]) & op_window
        
        # LONG
        cond_long = cond_operacional & (percentil_vals > params["percentil_long_trigger"]) & (velocidade_vals > 0.0)
        
        # SHORT (trigger simétrico)
        trigger_short = 1.0 - params["percentil_long_trigger"]
        cond_short = cond_operacional & (percentil_vals < trigger_short) & (velocidade_vals < 0.0)
        
        sinal = np.zeros(n_len, dtype=np.int8)
        sinal[cond_long] = 1
        sinal[cond_short] = -1
        
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
            preco_entrada = opens[i+1]
            
            if direcao == 1:
                sl_preco = preco_entrada - (sl_arr[i] / 10000.0)
                tp_preco = preco_entrada + (tp_arr[i] / 10000.0)
            else:
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
                
            pnl -= 0.5 # Spread
                
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
                "hurst_cutoff": params["hurst_cutoff"],
                "entropia_cutoff": params["entropia_cutoff"],
                "percentil_trigger": params["percentil_long_trigger"],
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
        
        # Gerar o JSON top10 com ID para a esteira
        top10 = df_res.head(10).to_dict(orient="records")
        for i, p in enumerate(top10):
            p["id_parametro"] = f"{estrategia_nome.upper()}_{ativo.upper()}_{timeframe.upper()}_TOP{i+1}"
            
        with open(ARQUIVO_JSON, 'w') as f:
            json.dump(top10, f, indent=4)
            
        print("\n================================================================================")
        print("TOP 10 PARAMETRIZAÇÕES MOMENTUM (RANKING POR RET/DD > 60 Trades):")
        print("================================================================================")
        print(df_res.head(10).to_string())
        print(f"\nResultados salvos em: {ARQUIVO_SAIDA} e {ARQUIVO_JSON}")
    else:
        print("Nenhuma combinação atingiu os critérios mínimos (60 trades).")

if __name__ == "__main__":
    main()
