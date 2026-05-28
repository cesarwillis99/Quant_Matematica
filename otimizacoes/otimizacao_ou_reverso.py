import os
import sys
import numpy as np
import pandas as pd
import itertools
from scipy import stats
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# CONSTANTES E CAMINHOS
# =============================================================================
DIR_PROJETO = Path(r"c:\Users\cesar\.gemini\antigravity\scratch\Quant_Matematica.Trade")
DIR_DATA    = DIR_PROJETO / "quant_eurusd" / "data"
DIR_SAIDA   = DIR_DATA / "otimizacoes"

PARQUET_COMPLETO    = DIR_DATA / "eurusd_h1_completo.parquet"
PARQUET_OPERACIONAL = DIR_DATA / "eurusd_h1_operacional.parquet"
ARQUIVO_SAIDA       = DIR_SAIDA / "otimizacao_ou_reverso_resultados.parquet"

# =============================================================================
# GRID SEARCH
# =============================================================================
GRID_PARAMS = {
    "halflife_max": [30.0, 40.0, 50.0, 60.0],
    "zscore_threshold": [1.5, 2.0, 2.5, 3.0],
    "multiplicador_sl": [2.0, 2.5, 3.0],
    "multiplicador_tp": [1.5, 2.0, 2.5, 3.0]
}

# =============================================================================
# REGRESSÃO OLS RÁPIDA
# =============================================================================
def _quick_ols_stats(x: np.ndarray, y: np.ndarray) -> tuple:
    try:
        n = len(x)
        mx = x.mean()
        my = y.mean()
        xm = x - mx
        ym = y - my
        
        ss_xx = np.sum(xm ** 2)
        if ss_xx <= 0:
            return np.nan, np.nan, np.nan, np.nan, None
            
        ss_xy = np.sum(xm * ym)
        slope = ss_xy / ss_xx
        intercept = my - slope * mx
        
        y_pred = intercept + slope * x
        residuos = y - y_pred
        rss = np.sum(residuos ** 2)
        
        df_resid = n - 2
        if df_resid <= 0:
            return slope, intercept, np.nan, np.nan, residuos
            
        var_resid = rss / df_resid
        sigma_resid = np.sqrt(var_resid)
        
        se_slope = np.sqrt(var_resid / ss_xx)
        if se_slope <= 0:
            return slope, intercept, 0.0, sigma_resid, residuos
            
        t_stat = slope / se_slope
        p_value = 2.0 * stats.t.sf(np.abs(t_stat), df=df_resid)
        
        return slope, intercept, p_value, sigma_resid, residuos
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, None


def calcular_ou_rolling(df: pd.DataFrame, janela: int = 100) -> tuple:
    log_preco = np.log(df["Close"].values).astype("float32")
    n_rows = len(df)
    
    ou_theta    = np.full(n_rows, np.nan, dtype=np.float32)
    ou_mu       = np.full(n_rows, np.nan, dtype=np.float32)
    ou_sigma_eq = np.full(n_rows, np.nan, dtype=np.float32)
    ou_halflife = np.full(n_rows, np.nan, dtype=np.float32)
    ou_zscore   = np.full(n_rows, np.nan, dtype=np.float32)
    ou_valido   = np.zeros(n_rows, dtype=bool)
    
    for t in tqdm(range(janela - 1, n_rows), desc=f"Processando OU (Janela {janela})"):
        w = log_preco[t - (janela - 1): t + 1]
        
        y = w[1:]
        x = w[:-1]
        
        slope, intercept, p_value, sigma_resid, residuos = _quick_ols_stats(x, y)
        
        if np.isnan(slope) or np.isnan(intercept) or np.isnan(p_value) or np.isnan(sigma_resid):
            continue
            
        beta = slope
        alpha = intercept
        
        if beta <= 0 or beta >= 1:
            continue
            
        theta = -np.log(beta)
        if theta <= 0:
            continue
            
        if p_value > 0.05:
            continue
            
        half_life = np.log(2.0) / theta
        
        if half_life < 1.0:
            continue
            
        mu = alpha / (1.0 - beta)
        sigma_eq = sigma_resid / np.sqrt(1.0 - np.exp(-2.0 * theta))
        z_ou = (w[-1] - mu) / sigma_eq
        
        ou_theta[t]    = theta
        ou_mu[t]       = mu
        ou_sigma_eq[t] = sigma_eq
        ou_halflife[t] = half_life
        ou_zscore[t]   = z_ou
        ou_valido[t]   = True
        
    return ou_zscore, ou_halflife, ou_valido

def main():
    os.makedirs(DIR_SAIDA, exist_ok=True)
    
    print("Carregando dados...")
    df_comp = pd.read_parquet(PARQUET_COMPLETO)
    df_op = pd.read_parquet(PARQUET_OPERACIONAL)
    
    mask_op = df_comp.index.isin(df_op.index)
    
    closes = df_comp["Close"].values
    highs = df_comp["High"].values
    lows = df_comp["Low"].values
    
    n_len = len(closes)
    
    log_return = np.log(closes / np.roll(closes, 1))
    log_return[0] = 0.0
    vr_series = pd.Series(log_return).rolling(window=50, min_periods=50).std(ddof=1).values
    vr_pips = vr_series * closes * 10000.0
    
    janelas_ou = [60, 80, 100, 120, 150]
    
    keys = list(GRID_PARAMS.keys())
    combinacoes = list(itertools.product(*[GRID_PARAMS[k] for k in keys]))
    
    print(f"Total de combinações internas por janela: {len(combinacoes)}")
    print(f"Total absoluto de cenários (Janelas x Combinações): {len(janelas_ou) * len(combinacoes)}")
    
    resultados = []
    
    for janela_ou in janelas_ou:
        print(f"\n[+] Iniciando cálculo da Janela OU (Reverso) = {janela_ou} (Loop Externo)...")
        ou_zscore, ou_halflife, ou_valido = calcular_ou_rolling(df_comp, janela_ou)
        
        pbar = tqdm(total=len(combinacoes), desc=f"Otimizando Internamente (Janela {janela_ou})")
        
        for comb in combinacoes:
            params = dict(zip(keys, comb))
            
            cond_operacional = ou_valido & (ou_halflife >= 1.0) & (ou_halflife <= params["halflife_max"]) & mask_op
            
            # --- LÓGICA REVERSA ---
            # Compra se Z >= threshold (Momento a favor da fuga)
            cond_compra = cond_operacional & (ou_zscore >= params["zscore_threshold"])
            # Venda se Z <= -threshold (Momento a favor da fuga)
            cond_venda  = cond_operacional & (ou_zscore <= -params["zscore_threshold"])
            
            sinal = np.zeros(n_len, dtype=np.int8)
            sinal[cond_compra] = 1
            sinal[cond_venda]  = -1
            
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
                    "janela_ou": janela_ou,
                    "halflife_max": params["halflife_max"],
                    "zscore_threshold": params["zscore_threshold"],
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
        print("\n================================================================================")
        print("TOP 10 PARAMETRIZAÇÕES (RANKING POR RET/DD > 60 Trades):")
        print("================================================================================")
        print(df_res.head(10).to_string())
        print(f"\nResultados salvos em: {ARQUIVO_SAIDA}")
    else:
        print("Nenhuma combinação atingiu os critérios mínimos (60 trades).")

if __name__ == "__main__":
    main()
