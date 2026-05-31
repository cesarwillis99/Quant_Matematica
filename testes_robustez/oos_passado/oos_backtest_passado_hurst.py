# -*- coding: utf-8 -*-
"""
================================================================================
oos_backtest_passado_hurst.py - Teste Out-of-Sample (OOS) PASSADO
================================================================================
Script 100% autocontido para validacao OOS da estrategia HURST.
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
ESTRATEGIA      = "HURST"
TIPO_OOS        = "PASSADO"

DIR_PROJETO = Path(__file__).resolve().parent.parent.parent
DIR_DATA = DIR_PROJETO / f"quant_{ATIVO.lower()}" / "data"

SUFIXO_ANO = "2013_2016"
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

COLUNA_SAIDA_ESTRATEGIA = None
VALOR_SAIDA_MIN = None
VALOR_SAIDA_MAX = None

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
    
    
    df["volatilidade"] = df["log_return"].rolling(50).std()
    
    df["vr_pips"] = df["volatilidade"] * df["Close"] * FATOR_PIPS
    df["sl_pips"] = 2.0 * df["vr_pips"]
    df["tp_pips"] = 4.0 * df["vr_pips"]
    
    df["sinal"] = 0

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
    print(f"Grafico de equity curve salvo em: {img_eq_path}\n")

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
