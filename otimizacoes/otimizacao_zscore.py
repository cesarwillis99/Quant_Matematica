import sys
import numpy as np
import pandas as pd
import time
import os
from pathlib import Path
import itertools
import argparse
import json

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
DIR_OUT     = DIR_DATA / "otimizacoes" / estrategia_nome

PARQUET_COMPLETO    = DIR_DATA / f"{ativo}_{timeframe}_completo.parquet"
PARQUET_OPERACIONAL = DIR_DATA / f"{ativo}_{timeframe}_operacional.parquet"
ARQUIVO_SAIDA       = DIR_OUT / f"otimizacao_{estrategia_nome}_resultados.parquet"
ARQUIVO_JSON        = DIR_OUT / f"otimizacao_{estrategia_nome}_top10.json"

def main():
    DIR_OUT.mkdir(parents=True, exist_ok=True)

    print("Carregando dados...")
    if not PARQUET_COMPLETO.exists():
        raise FileNotFoundError(f"Arquivo completo nao encontrado: {PARQUET_COMPLETO}")
    if not PARQUET_OPERACIONAL.exists():
        raise FileNotFoundError(f"Arquivo operacional nao encontrado: {PARQUET_OPERACIONAL}")

    df = pd.read_parquet(PARQUET_COMPLETO)
    df_op = pd.read_parquet(PARQUET_OPERACIONAL)

    closes = df["Close"].to_numpy()
    highs = df["High"].to_numpy()
    lows = df["Low"].to_numpy()
    opens = df["Open"].to_numpy()

    if "log_return" not in df.columns:
        log_returns = np.log(closes / np.roll(closes, 1))
        log_returns[0] = 0.0
    else:
        log_returns = df["log_return"].to_numpy()

    if "hurst" not in df.columns:
        # Fallback: hurst fixo abaixo do limiar (assume anti-persistente)
        hurst = np.full(len(closes), 0.35, dtype=np.float32)
    else:
        hurst = df["hurst"].to_numpy()

    # Janela operacional - via parquet operacional
    mask_op = df.index.isin(df_op.index)
    if isinstance(mask_op, pd.Series) or hasattr(mask_op, 'to_numpy'):
        mask_op = mask_op.to_numpy()

    janelas_zscore = [20, 30, 50, 80, 100]
    janelas_volatilidade = [20, 30, 50, 80, 100]
    z_entries = [1.5, 2.0, 2.5, 3.0]
    mult_sls = [1.5, 2.0, 2.5]
    mult_tps = [2.0, 3.0, 4.0, 5.0]
    hurst_cutoffs = [0.30, 0.35, 0.40, 0.45]
    usar_saida_neutra_vals = [0, 1]

    zscore_dict = {}
    print("Pre-calculando matrizes de Z-Score...")
    s_close = pd.Series(closes)
    for j_z in janelas_zscore:
        roll_mean = s_close.rolling(window=j_z).mean().to_numpy()
        roll_std = s_close.rolling(window=j_z).std(ddof=1).to_numpy()
        roll_std = np.where(roll_std == 0, 1e-9, roll_std)
        z = (closes - roll_mean) / roll_std
        zscore_dict[j_z] = z

    vr_dict = {}
    print("Pre-calculando matrizes de Volatilidade...")
    s_lr = pd.Series(log_returns)
    for j_v in janelas_volatilidade:
        vr = s_lr.rolling(window=j_v).std(ddof=1).to_numpy()
        vr_pips = vr * closes * 10000.0
        vr_dict[j_v] = vr_pips

    print("Gerando combinacoes do Grid Search...")
    combinacoes = list(itertools.product(
        janelas_zscore, janelas_volatilidade, z_entries, mult_sls, mult_tps, hurst_cutoffs, usar_saida_neutra_vals
    ))
    total_comb = len(combinacoes)
    print(f"Total de combinacoes a simular: {total_comb}")

    resultados = []
    t0 = time.time()
    n_len = len(closes)

    print("\nIniciando Processamento Vetorizado...")
    for idx, (j_z, j_v, z_e, m_sl, m_tp, h_cut, usar_sn) in enumerate(combinacoes):
        if idx % 600 == 0 and idx > 0:
            print(f"Progresso: {idx}/{total_comb} ({(idx/total_comb)*100:.1f}%)")

        z_array = zscore_dict[j_z]
        vr_array = vr_dict[j_v]

        sinais = np.zeros(n_len, dtype=np.int8)
        cond_base = mask_op & (hurst < h_cut)

        # LONG: fechou abaixo do limiar negativo — entrada imediata
        cond_long = cond_base & (z_array <= -z_e)
        # SHORT: fechou acima do limiar positivo — entrada imediata
        cond_short = cond_base & (z_array >= z_e)

        sinais[cond_long] = 1
        sinais[cond_short] = -1

        entradas_idx = np.where(sinais != 0)[0]

        if len(entradas_idx) < 30:
            continue

        pnls = []
        trade_ativo_ate_indice = -1

        for i in entradas_idx:
            if i <= trade_ativo_ate_indice:
                continue

            if i + 1 >= n_len:
                continue

            direcao = sinais[i]
            entrada_preco = opens[i+1]

            vr_pips_atual = vr_array[i]
            if np.isnan(vr_pips_atual) or vr_pips_atual <= 0:
                continue

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

            end_idx = min(i + 300, n_len)
            fut_highs = highs[i+1:end_idx]
            fut_lows = lows[i+1:end_idx]
            fut_closes = closes[i+1:end_idx]
            fut_z = z_array[i+1:end_idx]

            if direcao == 1:
                hit_sl = np.where(fut_lows <= sl_preco)[0]
                hit_tp = np.where(fut_highs >= tp_preco)[0]
                hit_nt = np.where(fut_z >= -0.5)[0] if usar_sn == 1 else np.array([])
            else:
                hit_sl = np.where(fut_highs >= sl_preco)[0]
                hit_tp = np.where(fut_lows <= tp_preco)[0]
                hit_nt = np.where(fut_z <= 0.5)[0] if usar_sn == 1 else np.array([])

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
                trade_pips = (saida_preco - entrada_preco) * 10000.0
            else:
                trade_pips = (entrada_preco - saida_preco) * 10000.0

            trade_pips -= 0.5  # Spread
            pnls.append(trade_pips)

        if len(pnls) < 30:
            continue

        pnls_arr = np.array(pnls)
        lucro_total = pnls_arr.sum()

        cum_pips = np.cumsum(pnls_arr)
        peak_pips = np.maximum.accumulate(cum_pips)
        dd_pips = peak_pips - cum_pips
        max_dd = dd_pips.max()
        max_dd = max_dd if max_dd > 0 else 1.0

        ret_dd = lucro_total / max_dd

        ganhos = pnls_arr[pnls_arr > 0]
        perdas = np.abs(pnls_arr[pnls_arr < 0])
        sum_perdas = perdas.sum()
        sum_ganhos = ganhos.sum()
        pf = sum_ganhos / sum_perdas if sum_perdas > 0 else 99.0
        
        win_count = len(ganhos)
        loss_count = len(perdas)
        trades_count = len(pnls)
        
        win_rate = (win_count / trades_count) * 100.0 if trades_count > 0 else 0.0
        avg_win = (sum_ganhos / win_count) if win_count > 0 else 0.0
        avg_loss = (sum_perdas / loss_count) if loss_count > 0 else 0.0
        payoff = (avg_win / avg_loss) if avg_loss > 0 else 99.0

        resultados.append({
            "janela_zscore": j_z,
            "janela_vol": j_v,
            "z_entry": z_e,
            "mult_sl": m_sl,
            "mult_tp": m_tp,
            "hurst_cut": h_cut,
            "usar_saida_neutra": usar_sn,
            "Trades": trades_count,
            "Win_Rate": round(win_rate, 2),
            "Payoff": round(payoff, 2),
            "Lucro_Total_Pips": round(lucro_total, 1),
            "Max_DD_Pips": round(max_dd, 1),
            "Ret_DD": round(ret_dd, 2),
            "Profit_Factor": round(pf, 2)
        })

    t1 = time.time()
    print(f"\nOtimizacao concluida em {t1 - t0:.2f} segundos!")

    if resultados:
        df_res = pd.DataFrame(resultados)
        # Filtro de Sobrevivencia (Trades >= 60 e F.R. > 1.0)
        mask_survivor = (df_res["Trades"] >= 60) & (df_res["Ret_DD"] > 1.0)
        df_res = df_res[mask_survivor]
        df_res = df_res.sort_values(by="Ret_DD", ascending=False).reset_index(drop=True)

        if len(df_res) > 0:
            df_res.to_parquet(ARQUIVO_SAIDA)

            top50 = df_res.head(50)
            
            dir_relatorios = DIR_OUT / "relatorios_top50"
            os.makedirs(dir_relatorios, exist_ok=True)
            ARQUIVO_PNG = dir_relatorios / f"relatorio_top50_{estrategia_nome.lower()}.png"
            try:
                from otimizacoes.export_utils import salvar_tabela_png
            except ImportError:
                import sys
                from pathlib import Path
                sys.path.append(str(Path(__file__).resolve().parent))
                from export_utils import salvar_tabela_png
            salvar_tabela_png(top50, ARQUIVO_PNG, titulo=f"Top 50 Parametrizações - {estrategia_nome.upper()}")
            
            top50_dicts = top50.to_dict(orient="records")
            for i, p in enumerate(top50_dicts):
                p["id_parametro"] = f"{estrategia_nome.upper()}_{ativo.upper()}_{timeframe.upper()}_TOP{i+1}"

            dir_candidatos = DIR_OUT / "candidatos_testes"
            os.makedirs(dir_candidatos, exist_ok=True)
            ARQUIVO_JSON_CANDIDATOS = dir_candidatos / f"candidatos_{estrategia_nome.lower()}.json"
            with open(ARQUIVO_JSON_CANDIDATOS, 'w') as f:
                json.dump(top50_dicts, f, indent=4)

            print("\n" + "="*80)
            print("TOP 10 PARAMETRIZACOES SOBREVIVENTES (RANKING POR RET/DD):")
            print("="*80)
            print(df_res.head(10).to_string())
            print(f"\nResultados salvos em: {ARQUIVO_SAIDA}")
            print(f"Relatório Elegante PNG salvo em: {ARQUIVO_PNG}")
            print(f"Candidatos JSON salvo em: {ARQUIVO_JSON_CANDIDATOS}")
        else:
            print("Nenhuma combinacao sobreviveu aos criterios rigorosos (Min 60 Trades, F.R > 1.0).")
    else:
        print("Nenhuma combinacao atingiu os criterios minimos (30 trades).")

if __name__ == "__main__":
    main()
