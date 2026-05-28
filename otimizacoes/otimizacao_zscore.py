import sys
import numpy as np
import pandas as pd
import time
from pathlib import Path
import itertools

# Configuração de caminhos
DIR_PROJETO = Path(__file__).resolve().parent.parent
DIR_DATA = DIR_PROJETO / "quant_eurusd" / "data"
DIR_OUT = DIR_DATA / "otimizacoes"

def rodar_grid_search():
    DIR_OUT.mkdir(parents=True, exist_ok=True)
    
    print("Carregando dados...")
    # O eurusd_h1_hurst.parquet contém hurst, preços e log_return.
    df = pd.read_parquet(DIR_DATA / "eurusd_h1_hurst.parquet")
    
    closes = df["Close"].to_numpy()
    highs = df["High"].to_numpy()
    lows = df["Low"].to_numpy()
    opens = df["Open"].to_numpy()
    log_returns = df["log_return"].to_numpy()
    hurst = df["hurst"].to_numpy()
    
    janelas_zscore = [20, 30, 50, 80, 100]
    janelas_volatilidade = [20, 30, 50, 80, 100]
    z_entries = [1.5, 2.0, 2.5, 3.0]
    mult_sls = [1.5, 2.0, 2.5]
    mult_tps = [2.0, 3.0, 4.0, 5.0]
    hurst_cutoffs = [0.30, 0.35, 0.40, 0.45]
    
    zscore_dict = {}
    print("Pré-calculando matrizes de Z-Score...")
    s_close = pd.Series(closes)
    for j_z in janelas_zscore:
        roll_mean = s_close.rolling(window=j_z).mean().to_numpy()
        roll_std = s_close.rolling(window=j_z).std(ddof=1).to_numpy()
        roll_std = np.where(roll_std == 0, 1e-9, roll_std) # Evitar divisao por zero
        z = (closes - roll_mean) / roll_std
        zscore_dict[j_z] = z

    vr_dict = {}
    print("Pré-calculando matrizes de Volatilidade...")
    s_lr = pd.Series(log_returns)
    for j_v in janelas_volatilidade:
        vr = s_lr.rolling(window=j_v).std(ddof=1).to_numpy()
        vr_pips = vr * closes * 10000.0
        vr_dict[j_v] = vr_pips
        
    print("Gerando combinações do Grid Search...")
    combinacoes = list(itertools.product(
        janelas_zscore, janelas_volatilidade, z_entries, mult_sls, mult_tps, hurst_cutoffs
    ))
    total_comb = len(combinacoes)
    print(f"Total de combinações a simular: {total_comb}")
    
    # Determinar a janela operacional de forma vetorizada
    dt_index = df.index
    weekday = dt_index.weekday
    hora = dt_index.strftime('%H:%M')
    seg_sex = (weekday >= 0) & (weekday <= 4)
    horario_op = (hora >= "10:00") & (hora <= "22:30")
    janela_operacional = (seg_sex & horario_op)
    if isinstance(janela_operacional, pd.Series) or hasattr(janela_operacional, 'to_numpy'):
        janela_operacional = janela_operacional.to_numpy()
    
    resultados = []
    t0 = time.time()
    n_len = len(closes)
    
    print("\nIniciando Processamento Vetorizado...")
    # Pseudo-compilador (loop otimizado)
    for idx, (j_z, j_v, z_e, m_sl, m_tp, h_cut) in enumerate(combinacoes):
        if idx % 600 == 0 and idx > 0:
            print(f"Progresso: {idx}/{total_comb} ({(idx/total_comb)*100:.1f}%)")
            
        z_array = zscore_dict[j_z]
        vr_array = vr_dict[j_v]
        
        # Lógica de Sinais de RETORNO Totalmente Vetorizada
        z_prev = np.concatenate(([0], z_array[:-1]))
        sinais = np.zeros(n_len, dtype=np.int8)
        cond_base = janela_operacional & (hurst < h_cut)
        
        # LONG: no candle anterior estava <= -z_e, e no atual está > -z_e
        cond_long = cond_base & (z_prev <= -z_e) & (z_array > -z_e)
        
        # SHORT: no candle anterior estava >= z_e, e no atual está < z_e
        cond_short = cond_base & (z_prev >= z_e) & (z_array < z_e)
        
        sinais[cond_long] = 1
        sinais[cond_short] = -1
        
        entradas_idx = np.where(sinais != 0)[0]
        
        if len(entradas_idx) < 30:
            continue
            
        pnls = []
        
        trade_ativo_ate_indice = -1
        
        # Simulador vetorizado nos trades
        for i in entradas_idx:
            if i <= trade_ativo_ate_indice:
                continue
                
            if i + 1 >= n_len: continue
            
            direcao = sinais[i]
            entrada_preco = opens[i+1]
            
            # SL e TP em pips
            vr_pips_atual = vr_array[i]
            if np.isnan(vr_pips_atual) or vr_pips_atual <= 0: continue
                
            sl_pips = m_sl * vr_pips_atual
            tp_pips = m_tp * vr_pips_atual
            
            delta_sl = sl_pips / 10000.0
            delta_tp = tp_pips / 10000.0
            
            if direcao == 1:
                sl_preco = entrada_preco - delta_sl
                tp_preco = entrada_preco + delta_tp
            else:
                sl_preco = entrada_preco + delta_sl
                tp_preco = entrada_preco - delta_tp
                
            # Buscar futuro de forma vetorizada (máximo de 300 candles para frente)
            end_idx = min(i + 300, n_len)
            fut_highs = highs[i+1:end_idx]
            fut_lows = lows[i+1:end_idx]
            fut_closes = closes[i+1:end_idx]
            fut_z = z_array[i+1:end_idx]
            
            # Array checking
            if direcao == 1:
                hit_sl = np.where(fut_lows <= sl_preco)[0]
                hit_tp = np.where(fut_highs >= tp_preco)[0]
                hit_nt = np.where(fut_z >= -0.5)[0]
            else:
                hit_sl = np.where(fut_highs >= sl_preco)[0]
                hit_tp = np.where(fut_lows <= tp_preco)[0]
                hit_nt = np.where(fut_z <= 0.5)[0]
                
            idx_sl = hit_sl[0] if len(hit_sl) > 0 else 9999
            idx_tp = hit_tp[0] if len(hit_tp) > 0 else 9999
            idx_nt = hit_nt[0] if len(hit_nt) > 0 else 9999
            
            min_idx = min(idx_sl, idx_tp, idx_nt)
            
            if min_idx == 9999:
                saida_preco = fut_closes[-1] # Sai pelo limite temporal de 300 barras
                trade_ativo_ate_indice = end_idx - 1
            elif min_idx == idx_sl:
                saida_preco = sl_preco
                trade_ativo_ate_indice = i + 1 + idx_sl
            elif min_idx == idx_tp:
                saida_preco = tp_preco
                trade_ativo_ate_indice = i + 1 + idx_tp
            else:
                saida_preco = fut_closes[idx_nt] # Sai neutro no fechamento do candle
                trade_ativo_ate_indice = i + 1 + idx_nt
                
            # Calculo do PnL em Pips
            if direcao == 1:
                trade_pips = (saida_preco - entrada_preco) * 10000.0
            else:
                trade_pips = (entrada_preco - saida_preco) * 10000.0
                
            trade_pips -= 0.5 # Subtrai o Spread (0.5)
            pnls.append(trade_pips)
            
        if len(pnls) < 30:
            continue
            
        pnls_arr = np.array(pnls)
        lucro_total = pnls_arr.sum()
        
        # Max Drawdown (em pips)
        cum_pips = np.cumsum(pnls_arr)
        peak_pips = np.maximum.accumulate(cum_pips)
        dd_pips = peak_pips - cum_pips
        max_dd = dd_pips.max()
        max_dd = max_dd if max_dd > 0 else 1.0
        
        ret_dd = lucro_total / max_dd
        
        ganhos = pnls_arr[pnls_arr > 0]
        perdas = np.abs(pnls_arr[pnls_arr < 0])
        
        sum_perdas = perdas.sum()
        pf = ganhos.sum() / sum_perdas if sum_perdas > 0 else 99.0
        
        resultados.append({
            "janela_zscore": j_z,
            "janela_vol": j_v,
            "z_entry": z_e,
            "mult_sl": m_sl,
            "mult_tp": m_tp,
            "hurst_cut": h_cut,
            "Trades": len(pnls),
            "Lucro_Total_Pips": round(lucro_total, 1),
            "Max_DD_Pips": round(max_dd, 1),
            "Ret_DD": round(ret_dd, 2),
            "Profit_Factor": round(pf, 2)
        })
        
    t1 = time.time()
    print(f"\nOtimização concluída em {t1 - t0:.2f} segundos!")
    
    df_res = pd.DataFrame(resultados)
    
    if len(df_res) > 0:
        df_res = df_res.sort_values(by="Ret_DD", ascending=False).reset_index(drop=True)
        print("\n" + "="*80)
        print("TOP 10 PARAMETRIZAÇÕES (RANKING POR RET/DD):")
        print("="*80)
        print(df_res.head(10).to_string())
        
        out_file = DIR_OUT / "otimizacao_zscore_resultados.parquet"
        df_res.to_parquet(out_file)
        print(f"\nResultados salvos em: {out_file}")
    else:
        print("Nenhuma combinação obteve 30+ trades.")

if __name__ == "__main__":
    rodar_grid_search()
