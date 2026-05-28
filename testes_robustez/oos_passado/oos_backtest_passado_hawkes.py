# -*- coding: utf-8 -*-
"""
================================================================================
oos_backtest_passado_hawkes.py — Teste Out-of-Sample (OOS) PASSADO
================================================================================
Script 100% autocontido para validação OOS da estratégia HAWKES.
Nenhuma dependência externa ao projeto além das bibliotecas padrão.
"""

import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ═══════════════════════════════════════════════
# CONFIGURAÇÃO DO TESTE OOS — EDITAR AQUI
# ═══════════════════════════════════════════════

ATIVO           = "EURUSD"
TIMEFRAME       = "H1"
ESTRATEGIA      = "HAWKES"
TIPO_OOS        = "PASSADO"

DIR_PROJETO = Path(__file__).resolve().parent.parent.parent
DIR_DATA = DIR_PROJETO / "quant_eurusd" / "data"

PARQUET_COMPLETO    = DIR_DATA / f"{{ATIVO.lower()}}_{{TIMEFRAME.lower()}}_completo_OOS_passado.parquet"
PARQUET_OPERACIONAL = DIR_DATA / f"{{ATIVO.lower()}}_{{TIMEFRAME.lower()}}_operacional_OOS_passado.parquet"

# Parâmetros Operacionais Fixos (Treinamento)
CAPITAL_INICIAL     = 10_000.0
RISCO_POR_TRADE     = 0.01
SPREAD_PIPS         = 1.2
VALOR_PIP_POR_LOTE  = 10.0
FATOR_PIPS          = 10_000

# Horário operacional (Servidor MT5)
HORA_INICIO_OP      = "10:00"
HORA_FIM_OP         = "22:30"
HORA_FECHAMENTO_FDS = 21
HORA_BLOQUEIO_FDS   = 20

# Saída por sinal específico
COLUNA_SAIDA_ESTRATEGIA = None
VALOR_SAIDA_MIN = None
VALOR_SAIDA_MAX = None

# ═══════════════════════════════════════════════
# FIM DA CONFIGURAÇÃO
# ═══════════════════════════════════════════════

def recalcular_sinais_oos(df_completo: pd.DataFrame, df_operacional: pd.DataFrame) -> pd.DataFrame:
    """
    Recalcula os sinais da estratégia HAWKES sobre o período OOS
    usando os MESMOS parâmetros do treinamento.
    """
    df = df_completo.copy()
    
    # 1. Processo de Hawkes (Intensidade de Volatilidade)
    retornos_abs = np.abs(df["log_return"].fillna(0).values)
    kappa = 0.1 # Decaimento
    
    intensidade = np.zeros(len(df))
    for i in range(1, len(df)):
        intensidade[i] = intensidade[i-1] * np.exp(-kappa) + retornos_abs[i-1]
        
    df["hawkes_intensity"] = intensidade
    
    # Normalização
    df["hawkes_zscore"] = (df["hawkes_intensity"] - df["hawkes_intensity"].rolling(100).mean()) / df["hawkes_intensity"].rolling(100).std()

    # 2. Gestão de Risco
    vr = df["log_return"].rolling(50).std()
    df["vr_pips"] = vr * df["Close"] * FATOR_PIPS
    df["sl_pips"] = 1.5 * df["vr_pips"]
    df["tp_pips"] = 3.0 * df["vr_pips"]

    # 3. Sinais Hawkes (Cluster de alta volatilidade)
    weekday = df.index.weekday
    hora = df.index.strftime('%H:%M')
    janela_op = (weekday >= 0) & (weekday <= 4) & (hora >= HORA_INICIO_OP) & (hora <= HORA_FIM_OP)

    sinal = np.zeros(len(df), dtype=np.int8)
    # Direcional baseado no retorno médio no cluster
    ret_suave = df["log_return"].rolling(10).mean()
    
    cond_long = janela_op & (df["hawkes_zscore"] > 1.5) & (ret_suave > 0)
    cond_short = janela_op & (df["hawkes_zscore"] > 1.5) & (ret_suave < 0)

    sinal[cond_long] = 1
    sinal[cond_short] = -1
    df["sinal"] = sinal

    return df

def simular_backtest_candle_a_candle(df: pd.DataFrame):
    """
    Simula o preenchimento de ordens candle a candle, respeitando SL, TP,
    bloqueios de final de semana e métricas de conta.
    """
    capital = CAPITAL_INICIAL
    equity_curve = []
    trades = []
    
    posicao = 0  # 1 = LONG, -1 = SHORT, 0 = FLAT
    preco_entrada = 0.0
    sl_preco = 0.0
    tp_preco = 0.0
    lote = 0.0
    
    # Extrair vetores para acesso rápido
    times = df.index
    opens = df["Open"].values
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    sinais = df["sinal"].values
    sls_pips = df["sl_pips"].values
    tps_pips = df["tp_pips"].values
    
    # Variáveis de saída por estratégia (se configurado)
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
        
        # 1. PROCESSAR SAÍDAS
        if posicao != 0:
            fechou = False
            preco_saida = 0.0
            motivo = ""
            
            # Checagem de saída customizada
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
                    "entrada": preco_entrada, "saida": preco_saida, 
                    "pnl_usd": pnl_usd, "motivo": motivo
                })
                posicao = 0
        
        # 2. PROCESSAR ENTRADAS
        if posicao == 0 and sinais[i] != 0 and not bloqueio_entrada:
            posicao = sinais[i]
            preco_entrada = opens[i+1]
            
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
    
    retornos_diarios = equity_curve.resample("1D").last().pct_change().dropna()
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

def gerar_relatorio_e_grafico(metricas: dict, equity_curve: pd.Series):
    print("=" * 60)
    print(f"  TESTE OOS {TIPO_OOS} — {ATIVO} {TIMEFRAME}")
    print(f"  Estratégia: {ESTRATEGIA}")
    print(f"  Período: {equity_curve.index[0].date()} → {equity_curve.index[-1].date()}")
    print("=" * 60)
    print(f"  PnL %              : {metricas['pnl_pct']:>+10.2f}%")
    print(f"  Win Rate           : {metricas['win_rate']:>10.2f}%")
    print(f"  Sharpe Ratio       : {metricas['sharpe']:>+10.4f}")
    print(f"  Drawdown Máximo    : {metricas['dd_pct']:>10.2f}%")
    print(f"  Fator Recuperação  : {metricas['fator_recup']:>10.3f}x")
    print(f"  Total de Trades    : {metricas['total_trades']:>10,}")
    print(f"  Capital Final      : ${metricas['capital_final']:>10,.2f}")
    print("=" * 60)
    
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={{'height_ratios': [3, 1]}})
    fig.suptitle(f"OOS {TIPO_OOS} — {ATIVO} {TIMEFRAME} — {ESTRATEGIA}", fontsize=14)
    
    cor_equity = '#00E676' if metricas['pnl_pct'] >= 0 else '#FF1744'
    ax1.plot(equity_curve.index, equity_curve, color=cor_equity, linewidth=1.5)
    ax1.set_ylabel("Capital (USD)")
    ax1.grid(True, alpha=0.1)
    
    ax2.fill_between(metricas['dd_serie'].index, metricas['dd_serie'], 0, color='#FF1744', alpha=0.3)
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(True, alpha=0.1)
    
    nome_grafico = f"equity_curve_OOS_{TIPO_OOS}_{ATIVO}_{ESTRATEGIA}.png"
    plt.tight_layout()
    plt.savefig(nome_grafico, dpi=150)
    print(f"\nGráfico salvo em: {nome_grafico}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--estrategia", type=str, default=ESTRATEGIA)
    args = parser.parseargs() if hasattr(parser, "parseargs") else parser.parse_args()
    
    if not PARQUET_COMPLETO.exists():
        print(f"ERRO: Parquet completo não encontrado em {PARQUET_COMPLETO}")
        return
    if not PARQUET_OPERACIONAL.exists():
        print(f"ERRO: Parquet operacional não encontrado em {PARQUET_OPERACIONAL}")
        return
        
    df_comp = pd.read_parquet(PARQUET_COMPLETO)
    df_oper = pd.read_parquet(PARQUET_OPERACIONAL)
    
    print(f"Iniciando cálculo de sinais OOS para {ESTRATEGIA}...")
    df_sinais = recalcular_sinais_oos(df_comp, df_oper)
    
    print("Iniciando simulação candle-a-candle...")
    equity_curve, trades = simular_backtest_candle_a_candle(df_sinais)
    
    metricas = calcular_metricas(equity_curve, trades)
    gerar_relatorio_e_grafico(metricas, equity_curve)

if __name__ == "__main__":
    main()
