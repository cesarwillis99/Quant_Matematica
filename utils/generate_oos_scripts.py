import os
from pathlib import Path

ESTRATEGIAS = ["ZSCORE", "MOMENTUM", "OU", "HAWKES", "WAVELET", "PCA", "HURST"]

def obter_logica_estrategia(estr):
    if estr == "ZSCORE":
        return '''
    # 1. Z-Score Rolling (Janela 50)
    sma = df["Close"].rolling(50, min_periods=50).mean()
    std = df["Close"].rolling(50, min_periods=50).std(ddof=1)
    df["zscore"] = (df["Close"] - sma) / std
    
    # 2. Gestão de Risco (Volatilidade 50)
    vr = df["log_return"].rolling(50, min_periods=50).std(ddof=1)
    df["vr_pips"] = vr * df["Close"] * FATOR_PIPS
    df["sl_pips"] = 2.0 * df["vr_pips"]
    df["tp_pips"] = 3.0 * df["vr_pips"]
    
    # 3. Janela Operacional
    weekday = df.index.weekday
    hora = df.index.strftime('%H:%M')
    janela_op = (weekday >= 0) & (weekday <= 4) & (hora >= HORA_INICIO_OP) & (hora <= HORA_FIM_OP)
    
    # 4. Sinais (Gatilho de Retorno: Z-Score extremado + Filtro Hurst < 0.40)
    filtro_regime = (df["hurst"] < 0.40)
    condicao_entrada = janela_op & filtro_regime
    
    z = df["zscore"].values
    z_prev = df["zscore"].shift(1).values
    z_entry = 2.0
    
    cond_long = condicao_entrada & (z_prev <= -z_entry) & (z > -z_entry)
    cond_short = condicao_entrada & (z_prev >= z_entry) & (z < z_entry)
    
    valid_shift = (~df["zscore"].isna()) & (~df["zscore"].shift(1).isna())
    cond_long = cond_long & valid_shift
    cond_short = cond_short & valid_shift

    sinal = np.zeros(len(df), dtype=np.int8)
    sinal[cond_long] = 1
    sinal[cond_short] = -1
    df["sinal"] = sinal
    
    # Saída do Z-Score quando cruza zero (neutro)
    df["zscore_neutro"] = df["zscore"]
'''
    elif estr == "MOMENTUM":
        return '''
    # 1. Velocidade e Aceleração
    df["velocidade"] = df["Close"].diff(1)
    df["aceleracao"] = df["velocidade"].diff(1)

    def _percentrank(arr):
        val = arr[-1]
        hist = arr[:-1]
        if len(hist) == 0: return 0.5
        return float(np.sum(hist < val)) / float(len(hist))

    df["percentil_acel"] = df["aceleracao"].rolling(100, min_periods=100).apply(_percentrank, raw=True)

    # 2. Entropia Shannon
    def _entropia(arr):
        if np.std(arr) < 1e-15: return 0.0
        counts, _ = np.histogram(arr, bins=10)
        p = counts / len(arr)
        p = p[p > 0]
        h = -np.sum(p * np.log2(p))
        return float(np.clip(h / np.log2(10), 0, 1))

    df["entropia_shannon"] = df["log_return"].rolling(30, min_periods=30).apply(_entropia, raw=True)

    # 3. Gestão de Risco
    vr = df["log_return"].rolling(50, min_periods=50).std(ddof=1)
    df["vr_pips"] = vr * df["Close"] * FATOR_PIPS
    df["sl_pips"] = 1.5 * df["vr_pips"]
    df["tp_pips"] = 4.0 * df["vr_pips"]

    # 4. Sinais
    weekday = df.index.weekday
    hora = df.index.strftime('%H:%M')
    janela_op = (weekday >= 0) & (weekday <= 4) & (hora >= HORA_INICIO_OP) & (hora <= HORA_FIM_OP)

    sinal = np.zeros(len(df), dtype=np.int8)
    cond_long = janela_op & (df["percentil_acel"] > 0.75) & (df["velocidade"] > 0) & (df["entropia_shannon"] < 0.60)
    cond_short = janela_op & (df["percentil_acel"] < 0.25) & (df["velocidade"] < 0) & (df["entropia_shannon"] < 0.60)

    sinal[cond_long] = 1
    sinal[cond_short] = -1
    df["sinal"] = sinal
'''
    elif estr == "OU":
        return '''
    import statsmodels.api as sm
    
    janela_ou = 100
    residuos = np.full(len(df), np.nan)
    precos = df["Close"].values
    for i in range(janela_ou, len(df)):
        y = precos[i-janela_ou+1 : i+1]
        x = precos[i-janela_ou : i]
        beta = np.cov(x, y)[0,1] / np.var(x) if np.var(x) > 0 else 0
        alpha = np.mean(y) - beta * np.mean(x)
        res_t = precos[i] - (alpha + beta * precos[i-1])
        residuos[i] = res_t
        
    df["ou_residuos"] = residuos
    std_res = df["ou_residuos"].rolling(janela_ou).std()
    df["ou_zscore"] = df["ou_residuos"] / std_res

    vr = df["log_return"].rolling(50).std()
    df["vr_pips"] = vr * df["Close"] * FATOR_PIPS
    df["sl_pips"] = 1.5 * df["vr_pips"]
    df["tp_pips"] = 3.5 * df["vr_pips"]

    weekday = df.index.weekday
    hora = df.index.strftime('%H:%M')
    janela_op = (weekday >= 0) & (weekday <= 4) & (hora >= HORA_INICIO_OP) & (hora <= HORA_FIM_OP)

    sinal = np.zeros(len(df), dtype=np.int8)
    cond_long = janela_op & (df["ou_zscore"] < -2.0)
    cond_short = janela_op & (df["ou_zscore"] > 2.0)

    sinal[cond_long] = 1
    sinal[cond_short] = -1
    df["sinal"] = sinal
'''
    elif estr == "HAWKES":
        return '''
    retornos_abs = np.abs(df["log_return"].fillna(0).values)
    kappa = 0.1
    intensidade = np.zeros(len(df))
    for i in range(1, len(df)):
        intensidade[i] = intensidade[i-1] * np.exp(-kappa) + retornos_abs[i-1]
        
    df["hawkes_intensity"] = intensidade
    df["hawkes_zscore"] = (df["hawkes_intensity"] - df["hawkes_intensity"].rolling(100).mean()) / df["hawkes_intensity"].rolling(100).std()

    vr = df["log_return"].rolling(50).std()
    df["vr_pips"] = vr * df["Close"] * FATOR_PIPS
    df["sl_pips"] = 1.5 * df["vr_pips"]
    df["tp_pips"] = 3.0 * df["vr_pips"]

    weekday = df.index.weekday
    hora = df.index.strftime('%H:%M')
    janela_op = (weekday >= 0) & (weekday <= 4) & (hora >= HORA_INICIO_OP) & (hora <= HORA_FIM_OP)

    sinal = np.zeros(len(df), dtype=np.int8)
    ret_suave = df["log_return"].rolling(10).mean()
    
    cond_long = janela_op & (df["hawkes_zscore"] > 1.5) & (ret_suave > 0)
    cond_short = janela_op & (df["hawkes_zscore"] > 1.5) & (ret_suave < 0)

    sinal[cond_long] = 1
    sinal[cond_short] = -1
    df["sinal"] = sinal
'''
    elif estr == "WAVELET":
        return '''
    sma_fast = df["Close"].rolling(10).mean()
    sma_slow = df["Close"].rolling(40).mean()
    df["wavelet_phase"] = sma_fast - sma_slow

    vr = df["log_return"].rolling(50).std()
    df["vr_pips"] = vr * df["Close"] * FATOR_PIPS
    df["sl_pips"] = 1.5 * df["vr_pips"]
    df["tp_pips"] = 3.5 * df["vr_pips"]

    weekday = df.index.weekday
    hora = df.index.strftime('%H:%M')
    janela_op = (weekday >= 0) & (weekday <= 4) & (hora >= HORA_INICIO_OP) & (hora <= HORA_FIM_OP)

    sinal = np.zeros(len(df), dtype=np.int8)
    fase = df["wavelet_phase"].values
    fase_prev = np.roll(fase, 1)
    fase_prev[0] = 0
    
    cross_up = (fase > 0) & (fase_prev <= 0)
    cross_down = (fase < 0) & (fase_prev >= 0)
    
    cond_long = janela_op & cross_up
    cond_short = janela_op & cross_down

    sinal[cond_long] = 1
    sinal[cond_short] = -1
    df["sinal"] = sinal
'''
    elif estr == "PCA":
        return '''
    df["feat_ret"] = df["log_return"].rolling(10).sum()
    df["feat_vol"] = df["log_return"].rolling(10).std()
    df["feat_mom"] = df["Close"].diff(10)
    
    df["pca_score_1"] = df["feat_ret"] * 0.6 + df["feat_mom"] * 0.4
    
    vr = df["log_return"].rolling(50).std()
    df["vr_pips"] = vr * df["Close"] * FATOR_PIPS
    df["sl_pips"] = 1.5 * df["vr_pips"]
    df["tp_pips"] = 3.5 * df["vr_pips"]

    weekday = df.index.weekday
    hora = df.index.strftime('%H:%M')
    janela_op = (weekday >= 0) & (weekday <= 4) & (hora >= HORA_INICIO_OP) & (hora <= HORA_FIM_OP)

    sinal = np.zeros(len(df), dtype=np.int8)
    
    cond_long = janela_op & (df["pca_score_1"] > df["pca_score_1"].rolling(100).mean() + 2 * df["pca_score_1"].rolling(100).std())
    cond_short = janela_op & (df["pca_score_1"] < df["pca_score_1"].rolling(100).mean() - 2 * df["pca_score_1"].rolling(100).std())

    sinal[cond_long] = 1
    sinal[cond_short] = -1
    df["sinal"] = sinal
'''
    elif estr == "HURST":
        return '''
    df["volatilidade"] = df["log_return"].rolling(50).std()
    
    df["vr_pips"] = df["volatilidade"] * df["Close"] * FATOR_PIPS
    df["sl_pips"] = 2.0 * df["vr_pips"]
    df["tp_pips"] = 4.0 * df["vr_pips"]
    
    df["sinal"] = 0
'''
    else:
        return ""


TEMPLATE = '''# -*- coding: utf-8 -*-
"""
================================================================================
oos_backtest_{tipo_lower}_{estr_lower}.py - Teste Out-of-Sample (OOS) {tipo_upper}
================================================================================
Script 100% autocontido para validacao OOS da estrategia {estr_upper}.
Inclui motor matematico Hurst e exportacao customizada de CSV/Imagens.
"""

import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Corrige encoding de stdout para Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

ATIVO           = "EURUSD"
TIMEFRAME       = "H1"
ESTRATEGIA      = "{estr_upper}"
TIPO_OOS        = "{tipo_upper}"

DIR_PROJETO = Path(__file__).resolve().parent.parent.parent
DIR_DATA = DIR_PROJETO / f"quant_{ATIVO.lower()}" / "data"

SUFIXO_ANO = "{sufixo_ano}"
PARQUET_COMPLETO    = DIR_DATA / f"{ATIVO.lower()}_{TIMEFRAME.lower()}_completo_OOS_{TIPO_OOS.lower()}_{SUFIXO_ANO}.parquet"
PARQUET_OPERACIONAL = DIR_DATA / f"{ATIVO.lower()}_{TIMEFRAME.lower()}_operacional_OOS_{TIPO_OOS.lower()}_{SUFIXO_ANO}.parquet"

DIR_BASE_OOS = Path(__file__).resolve().parent
DIR_SAIDA = DIR_BASE_OOS / ATIVO.lower() / ESTRATEGIA.lower()
DIR_SAIDA.mkdir(parents=True, exist_ok=True)

CAPITAL_INICIAL     = 10_000.0
RISCO_POR_TRADE     = 0.01
SPREAD_PIPS         = 0.5
VALOR_PIP_POR_LOTE  = 10.0
FATOR_PIPS          = 10_000

HORA_INICIO_OP      = "10:00"
HORA_FIM_OP         = "22:30"
HORA_FECHAMENTO_FDS = 21
HORA_BLOQUEIO_FDS   = 20

COLUNA_SAIDA_ESTRATEGIA = {col_saida}
VALOR_SAIDA_MIN = {val_min}
VALOR_SAIDA_MAX = {val_max}

# =============================================================================
# MOTOR MATEMATICO HURST
# =============================================================================
def _quick_ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    x_mean = x.mean()
    y_mean = y.mean()
    num = ((x - x_mean) * (y - y_mean)).sum()
    den = ((x - x_mean) ** 2).sum()
    if den == 0: return np.nan
    return num / den

def calcular_hurst_janela(retornos: np.ndarray) -> float:
    if len(retornos) < 100: return np.nan
    log_n, log_rs = [], []
    for n in [10, 20, 40, 80]:
        num_segmentos = 100 // n
        rs_segmentos = []
        for k in range(num_segmentos):
            segmento = retornos[k * n : (k + 1) * n]
            mu = segmento.mean()
            y_t = np.cumsum(segmento - mu)
            r_range = y_t.max() - y_t.min()
            s_std = segmento.std(ddof=1)
            if s_std == 0: return np.nan
            rs_segmentos.append(r_range / s_std)
        rs_medio = np.mean(rs_segmentos)
        if rs_medio > 0:
            log_n.append(np.log(n))
            log_rs.append(np.log(rs_medio))
    if len(log_n) < 2: return np.nan
    h = _quick_ols_slope(np.array(log_n), np.array(log_rs))
    if np.isnan(h) or h < 0.0 or h > 1.5: return np.nan
    return h

# =============================================================================

def recalcular_sinais_oos(df_completo: pd.DataFrame, df_operacional: pd.DataFrame) -> pd.DataFrame:
    df = df_completo.copy()
    
    print("Calculando Regime de Hurst OOS (Janela 100)... Isso pode levar alguns segundos.")
    retornos = df["log_return"].fillna(0).to_numpy()
    n_candles = len(df)
    hurst_values = np.full(n_candles, np.nan, dtype=np.float32)
    for i in range(99, n_candles):
        janela_retornos = retornos[i - 99 : i + 1]
        hurst_values[i] = calcular_hurst_janela(janela_retornos)
    df["hurst"] = hurst_values
    
    {logica_recalculo}
    return df

def simular_backtest_candle_a_candle(df: pd.DataFrame):
    capital = CAPITAL_INICIAL
    equity_curve = []
    trades = []
    
    posicao = 0 
    preco_entrada = 0.0
    sl_preco = 0.0
    tp_preco = 0.0
    lote = 0.0
    data_entrada = None
    
    times = df.index
    opens = df["Open"].values
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    sinais = df["sinal"].values
    sls_pips = df["sl_pips"].values
    tps_pips = df["tp_pips"].values
    
    if COLUNA_SAIDA_ESTRATEGIA is not None and COLUNA_SAIDA_ESTRATEGIA in df.columns:
        valores_saida_estr = df[COLUNA_SAIDA_ESTRATEGIA].values
    else:
        valores_saida_estr = np.zeros(len(df))
    
    weekdays = df.index.weekday
    hours = df.index.hour
    
    for i in range(len(df) - 1):
        t_time = times[i]
        wd = weekdays[i]
        hr = hours[i]
        
        eh_sexta = (wd == 4)
        eh_sexta_fechamento = eh_sexta and hr == HORA_FECHAMENTO_FDS
        bloqueio_entrada = (eh_sexta and hr >= HORA_BLOQUEIO_FDS) or (wd == 5) or (wd == 6 and hr < 21)
        
        if posicao != 0:
            fechou = False
            preco_saida = 0.0
            motivo = ""
            
            saida_estr_ativa = False
            if COLUNA_SAIDA_ESTRATEGIA is not None:
                val = valores_saida_estr[i]
                if posicao == 1 and VALOR_SAIDA_MIN is not None and val <= VALOR_SAIDA_MIN:
                    saida_estr_ativa = True
                elif posicao == -1 and VALOR_SAIDA_MAX is not None and val >= VALOR_SAIDA_MAX:
                    saida_estr_ativa = True
            
            if eh_sexta_fechamento:
                preco_saida = closes[i]
                fechou = True
                motivo = "FIM_SEMANA"
            else:
                if posicao == 1:
                    if lows[i] <= sl_preco and highs[i] >= tp_preco:
                        preco_saida, fechou, motivo = sl_preco, True, "SL (Prioridade)"
                    elif lows[i] <= sl_preco:
                        preco_saida, fechou, motivo = sl_preco, True, "SL"
                    elif highs[i] >= tp_preco:
                        preco_saida, fechou, motivo = tp_preco, True, "TP"
                    elif saida_estr_ativa:
                        preco_saida, fechou, motivo = closes[i], True, "SAIDA_ESTRATEGIA"
                
                elif posicao == -1:
                    if highs[i] >= sl_preco and lows[i] <= tp_preco:
                        preco_saida, fechou, motivo = sl_preco, True, "SL (Prioridade)"
                    elif highs[i] >= sl_preco:
                        preco_saida, fechou, motivo = sl_preco, True, "SL"
                    elif lows[i] <= tp_preco:
                        preco_saida, fechou, motivo = tp_preco, True, "TP"
                    elif saida_estr_ativa:
                        preco_saida, fechou, motivo = closes[i], True, "SAIDA_ESTRATEGIA"
            
            if fechou:
                if posicao == 1: pnl_pips = (preco_saida - preco_entrada) * FATOR_PIPS
                else:            pnl_pips = (preco_entrada - preco_saida) * FATOR_PIPS
                
                pnl_usd = (pnl_pips * lote * VALOR_PIP_POR_LOTE) - (SPREAD_PIPS * lote * VALOR_PIP_POR_LOTE)
                capital += pnl_usd
                
                trades.append({
                    "data_entrada": data_entrada,
                    "data_saida": t_time,
                    "direcao": "LONG" if posicao == 1 else "SHORT",
                    "preco_entrada": preco_entrada,
                    "preco_saida": preco_saida,
                    "lote": lote,
                    "pnl_usd": pnl_usd,
                    "motivo": motivo
                })
                posicao = 0
        
        if posicao == 0 and sinais[i] != 0 and not bloqueio_entrada:
            posicao = sinais[i]
            preco_entrada = opens[i+1]
            data_entrada = times[i+1]
            
            sl_p = sls_pips[i]
            tp_p = tps_pips[i]
            
            if pd.isna(sl_p) or pd.isna(tp_p):
                posicao = 0
                continue
                
            if posicao == 1:
                sl_preco = preco_entrada - (sl_p / FATOR_PIPS)
                tp_preco = preco_entrada + (tp_p / FATOR_PIPS)
            else:
                sl_preco = preco_entrada + (sl_p / FATOR_PIPS)
                tp_preco = preco_entrada - (tp_p / FATOR_PIPS)
            
            lote = (capital * RISCO_POR_TRADE) / (sl_p * VALOR_PIP_POR_LOTE)
            lote = max(0.01, min(lote, 100.0))
            
        equity_curve.append(capital)
    
    equity_curve.append(capital)
    return pd.Series(equity_curve, index=df.index), trades

def calcular_metricas(equity_curve: pd.Series, trades: list) -> dict:
    capital_final = equity_curve.iloc[-1]
    pnl_pct = ((capital_final - CAPITAL_INICIAL) / CAPITAL_INICIAL) * 100
    total_trades = len(trades)
    trades_ganhos = sum(1 for t in trades if t["pnl_usd"] > 0)
    win_rate = (trades_ganhos / total_trades * 100) if total_trades > 0 else 0.0
    
    retornos_diarios = equity_curve.resample("1D").last().pct_change()
    retornos_diarios = retornos_diarios.fillna(0.0)
    std_diario = retornos_diarios.std()
    sharpe = (retornos_diarios.mean() / std_diario) * math.sqrt(252) if std_diario > 0 else 0.0
    
    pico = equity_curve.cummax()
    dd_serie = (equity_curve - pico) / pico * 100
    dd_pct = dd_serie.min()
    
    pnl_usd_total = capital_final - CAPITAL_INICIAL
    dd_usd_max = (equity_curve - equity_curve.cummax()).min()
    fator_recup = abs(pnl_usd_total / dd_usd_max) if dd_usd_max < 0 else float('inf')
    
    return {
        "pnl_pct": pnl_pct, "win_rate": win_rate, "sharpe": sharpe,
        "dd_pct": dd_pct, "fator_recup": fator_recup,
        "total_trades": total_trades, "capital_final": capital_final,
        "dd_serie": dd_serie
    }

def gerar_relatorio_e_grafico(metricas: dict, equity_curve: pd.Series, trades: list):
    print("=" * 60)
    print(f"  TESTE OOS {TIPO_OOS} - {ATIVO} {TIMEFRAME}")
    print(f"  Estrategia: {ESTRATEGIA}")
    print(f"  Periodo: {equity_curve.index[0].date()} -> {equity_curve.index[-1].date()}")
    print("=" * 60)
    print(f"  PnL %              : {metricas['pnl_pct']:>+10.2f}%")
    print(f"  Win Rate           : {metricas['win_rate']:>10.2f}%")
    print(f"  Sharpe Ratio       : {metricas['sharpe']:>+10.4f}")
    print(f"  Drawdown Maximo    : {metricas['dd_pct']:>10.2f}%")
    print(f"  Fator Recuperacao  : {metricas['fator_recup']:>10.3f}x")
    print(f"  Total de Trades    : {metricas['total_trades']:>10,}")
    print(f"  Capital Final      : ${metricas['capital_final']:>10,.2f}")
    print("=" * 60)
    
    # 1. Salvar CSV Operacoes
    if trades:
        df_trades = pd.DataFrame(trades)
        csv_path = DIR_SAIDA / f"operacoes_OOS_{TIPO_OOS}_{ATIVO}_{ESTRATEGIA}.csv"
        df_trades.to_csv(csv_path, index=False)
        print(f"CSV de operacoes salvo em: {csv_path}")

    # 2. Gerar Tabela Metricas PNG
    fig_tbl, ax_tbl = plt.subplots(figsize=(6, 4))
    ax_tbl.axis('tight')
    ax_tbl.axis('off')
    
    dados_tabela = [
        ["PnL (%)", f"{metricas['pnl_pct']:.2f}%"],
        ["Win Rate", f"{metricas['win_rate']:.2f}%"],
        ["Sharpe Ratio", f"{metricas['sharpe']:.4f}"],
        ["Drawdown Máx", f"{metricas['dd_pct']:.2f}%"],
        ["Fator Recup.", f"{metricas['fator_recup']:.3f}x"],
        ["Total Trades", f"{metricas['total_trades']}"],
        ["Capital Final", f"${metricas['capital_final']:.2f}"]
    ]
    
    table = ax_tbl.table(cellText=dados_tabela, colLabels=["Métrica", "Valor OOS"], loc='center', cellLoc='center')
    table.scale(1, 2.5)
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#1e1e1e')
        else:
            cell.set_facecolor('#f4f4f4' if row % 2 == 0 else '#ffffff')
            
    img_tbl_path = DIR_SAIDA / f"tabela_metricas_OOS_{TIPO_OOS}_{ATIVO}_{ESTRATEGIA}.png"
    plt.savefig(img_tbl_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Tabela de metricas (PNG) salva em: {img_tbl_path}")
    
    # 3. Gerar Grafico Equity Curve
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(f"OOS {TIPO_OOS} - {ATIVO} {TIMEFRAME} - {ESTRATEGIA}", fontsize=14)
    
    cor_equity = '#00E676' if metricas['pnl_pct'] >= 0 else '#FF1744'
    ax1.plot(equity_curve.index, equity_curve, color=cor_equity, linewidth=1.5)
    ax1.set_ylabel("Capital (USD)")
    ax1.grid(True, alpha=0.1)
    
    ax2.fill_between(metricas['dd_serie'].index, metricas['dd_serie'], 0, color='#FF1744', alpha=0.3)
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(True, alpha=0.1)
    
    img_eq_path = DIR_SAIDA / f"equity_curve_OOS_{TIPO_OOS}_{ATIVO}_{ESTRATEGIA}.png"
    plt.tight_layout()
    plt.savefig(img_eq_path, dpi=150)
    plt.close()
    print(f"Grafico de equity curve salvo em: {img_eq_path}\\n")

def main():
    if not PARQUET_COMPLETO.exists():
        print(f"ERRO: Parquet completo nao encontrado em {PARQUET_COMPLETO}")
        return
    if not PARQUET_OPERACIONAL.exists():
        print(f"ERRO: Parquet operacional nao encontrado em {PARQUET_OPERACIONAL}")
        return
        
    df_comp = pd.read_parquet(PARQUET_COMPLETO)
    df_oper = pd.read_parquet(PARQUET_OPERACIONAL)
    
    print(f"Iniciando calculo de sinais OOS para {ESTRATEGIA}...")
    df_sinais = recalcular_sinais_oos(df_comp, df_oper)
    
    print("Iniciando simulacao candle-a-candle...")
    equity_curve, trades = simular_backtest_candle_a_candle(df_sinais)
    
    metricas = calcular_metricas(equity_curve, trades)
    gerar_relatorio_e_grafico(metricas, equity_curve, trades)

if __name__ == "__main__":
    main()
'''

DIR_ROBUSTEZ = Path("c:/Users/cesar/.gemini/antigravity/scratch/Quant_Matematica.Trade/testes_robustez")
DIR_PASSADO = DIR_ROBUSTEZ / "oos_passado"
DIR_FUTURO = DIR_ROBUSTEZ / "oos_futuro"

for tipo in ["PASSADO", "FUTURO"]:
    dir_target = DIR_PASSADO if tipo == "PASSADO" else DIR_FUTURO
    dir_target.mkdir(parents=True, exist_ok=True)
    
    suffix = "2013_2016" if tipo == "PASSADO" else "2024_2026"
    
    for estr in ESTRATEGIAS:
        logica = obter_logica_estrategia(estr)
        
        col_saida = "None"
        val_min = "None"
        val_max = "None"
        
        if estr == "ZSCORE":
            col_saida = '"zscore_neutro"'
            val_min = "-0.5"
            val_max = "0.5"
            
        script_content = TEMPLATE
        script_content = script_content.replace("{tipo_lower}", tipo.lower())
        script_content = script_content.replace("{tipo_upper}", tipo)
        script_content = script_content.replace("{estr_lower}", estr.lower())
        script_content = script_content.replace("{estr_upper}", estr)
        script_content = script_content.replace("{logica_recalculo}", logica)
        script_content = script_content.replace("{col_saida}", col_saida)
        script_content = script_content.replace("{val_min}", val_min)
        script_content = script_content.replace("{val_max}", val_max)
        script_content = script_content.replace("{sufixo_ano}", suffix)
        
        filename = dir_target / f"oos_backtest_{tipo.lower()}_{estr.lower()}.py"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(script_content)

print(f"Gerador recriou os scripts genericos na raiz e exportação customizada para ativo/estrategia/.")
