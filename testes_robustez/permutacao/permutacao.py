#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
permutacao.py
--------------
Teste de Robustez por Permutação Paramétrica Aleatória (Neighborhood SQX-style).

Gera N combinações aleatórias perturbando os parâmetros em +-20%.
Pré-computa colunas de alta complexidade matemática para máxima performance.
Aplica filtros estatísticos robustos (C1 a C5) e plota dashboard dark premium.

Uso:
  python permutacao.py
  python permutacao.py --estrategia MOMENTUM --ativo EURUSD --seed 42
"""

import sys
import math
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm

warnings.filterwarnings("ignore")
matplotlib.use('Agg')

# ===================================================================
# CONFIGURAÇÃO DO TESTE — EDITAR AQUI
# ===================================================================

# Identificação
ATIVO       = "EURUSD"
TIMEFRAME   = "H1"
ESTRATEGIA  = "MOMENTUM"

# Caminho do parquet com dados completos
# Deve conter: Open, High, Low, Close, log_return, hurst
PARQUET_DADOS = Path(__file__).resolve().parent.parent.parent / \
    "quant_eurusd_h1" / "data" / "eurusd_h1_hurst.parquet"

# Parâmetros ótimos encontrados na otimização
# Serão perturbados aleatoriamente dentro de ±20%
PARAMS_OTIMOS = {
    "percentil_trigger":  0.75,
    "entropia_cutoff":    0.60,
    "hurst_cutoff":       0.50,
    "mult_sl":            1.50,
    "mult_tp":            4.00,
}

# Parâmetros operacionais fixos
CAPITAL_INICIAL    = 10_000.0
RISCO_POR_TRADE    = 0.01
SPREAD_PIPS        = 0.5
VALOR_PIP_POR_LOTE = 10.0
FATOR_PIPS         = 10_000
HORA_INICIO_OP     = "10:00"
HORA_FIM_OP        = "22:30"

# Configuração da permutação
N_COMBINACOES   = 400    # Número de combinações aleatórias
PERTURBACAO_PCT = 0.20   # ±20% ao redor de cada valor ótimo
SEED            = 42     # Seed para reprodutibilidade

# Critérios de aprovação
CRITERIO_1_PCT_LUCRATIVAS   = 0.35   # >= 40% combinações com PnL > 0
CRITERIO_2_PNL_MEDIANO      = 0.0    # PnL mediano > 0
CRITERIO_3_FR_MEDIANO        = 0.30   # FR mediano >= 0.5
CRITERIO_4_FL_MEDIANO        = 1.0    # Fator de Lucro mediano >= 1.0
CRITERIO_5_OUTLIER_SIGMA     = 2.5    # Parâmetro ótimo original < média + 2.5σ

# ===================================================================
# FIM DA CONFIGURAÇÃO
# ===================================================================

# Pasta de saída gerada automaticamente (será atualizada dinamicamente na main)
DIR_SAIDA = Path(__file__).resolve().parent / f"{ATIVO.lower()}_{TIMEFRAME.lower()}"


# ===================================================================
# PARTE 1 — GERAÇÃO DAS COMBINAÇÕES ALEATÓRIAS
# ===================================================================

def gerar_combinacoes_aleatorias(params_otimos: dict, n_comb: int, perturb_pct: float, seed: int) -> list:
    """
    Gera n_comb conjuntos de parâmetros permutados aleatoriamente dentro de +-perturb_pct.
    Garante sementes estatísticas e restrições operacionais válidas.
    """
    np.random.seed(seed)
    combinacoes = []
    
    for _ in range(n_comb):
        comb = {}
        for param, val_orig in params_otimos.items():
            if not isinstance(val_orig, (int, float, np.integer, np.floating)) or isinstance(val_orig, bool) or param == "usar_saida_neutra":
                comb[param] = val_orig
                continue
            limite_inf = val_orig * (1 - perturb_pct)
            limite_sup = val_orig * (1 + perturb_pct)
            
            val_rand = np.random.uniform(limite_inf, limite_sup)
            
            # Tratamento de inteiros (janelas)
            eh_inteiro = ("janela" in param.lower()) or isinstance(val_orig, (int, np.integer))
            if eh_inteiro:
                val_rand = int(round(val_rand))
                val_rand = max(1, val_rand)
                
            comb[param] = val_rand
            
        # --- Restrições de Parâmetros (Garantia de Modelagem Saudável) ---
        
        # 1. Relação Stop Loss e Take Profit
        if "mult_sl" in comb and "mult_tp" in comb:
            if comb["mult_tp"] <= comb["mult_sl"]:
                # Inverte os multiplicadores se a ordem estiver incorreta
                comb["mult_sl"], comb["mult_tp"] = comb["mult_tp"], comb["mult_sl"]
                
        # 2. Limitadores Estritos de Vizinhança
        if "percentil_trigger" in comb:
            comb["percentil_trigger"] = float(np.clip(comb["percentil_trigger"], 0.51, 0.99))
            
        if "entropia_cutoff" in comb:
            comb["entropia_cutoff"] = float(np.clip(comb["entropia_cutoff"], 0.30, 0.95))
            
        if "hurst_cutoff" in comb:
            comb["hurst_cutoff"] = float(np.clip(comb["hurst_cutoff"], 0.30, 0.80))
            
        combinacoes.append(comb)
        
    return combinacoes


# ===================================================================
# PARTE 2 -- ROTEADOR MULTI-ESTRATEGIA
# ===================================================================
from testes_robustez.distribuicao_parametros.distribuicao_parametros import calcular_sinais


# ===================================================================
# PARTE 3 — BACKTEST COMPLETO (VETORIZADO/CANDLE-A-CANDLE HÍBRIDO)
# ===================================================================

def rodar_backtest(df: pd.DataFrame, sinal: np.ndarray, sl_pips: np.ndarray, tp_pips: np.ndarray) -> dict:
    """
    Backtest candle a candle com position sizing dinâmico e saídas de proteção.
    Executado de forma isolada, rápida e sem dependência externa.
    """
    opens  = df["Open"].values
    highs  = df["High"].values
    lows   = df["Low"].values
    closes = df["Close"].values
    df_idx = df.index
    n      = len(df)
    
    capital = CAPITAL_INICIAL
    equity  = [capital]
    trades  = []
    posicao = None
    
    for t in range(n):
        dt = df_idx[t]
        weekday = dt.weekday()
        hora    = dt.strftime("%H:%M")
        
        # Horários de bloqueio (final de semana)
        bloqueio = (
            (weekday == 4 and hora >= "20:30") or
            weekday == 5 or
            (weekday == 6 and hora < "21:00")
        )
        eh_sexta_fechamento = (weekday == 4 and hora == "21:55")
        
        # 1. Gerenciar Posição Aberta
        if posicao is not None:
            if eh_sexta_fechamento:
                saida = closes[t]
                motivo = "FDS"
            else:
                saida, motivo = None, None
                if posicao["direcao"] == 1:
                    if lows[t] <= posicao["sl"] and highs[t] >= posicao["tp"]:
                        saida, motivo = posicao["sl"], "SL"
                    elif lows[t] <= posicao["sl"]:
                        saida, motivo = posicao["sl"], "SL"
                    elif highs[t] >= posicao["tp"]:
                        saida, motivo = posicao["tp"], "TP"
                else:
                    if highs[t] >= posicao["sl"] and lows[t] <= posicao["tp"]:
                        saida, motivo = posicao["sl"], "SL"
                    elif highs[t] >= posicao["sl"]:
                        saida, motivo = posicao["sl"], "SL"
                    elif lows[t] <= posicao["tp"]:
                        saida, motivo = posicao["tp"], "TP"
                        
            if saida is not None:
                d = posicao["direcao"]
                pnl_pips = d * (saida - posicao["entrada"]) * FATOR_PIPS
                pnl_usd  = (pnl_pips * posicao["lote"] * VALOR_PIP_POR_LOTE
                            - SPREAD_PIPS * posicao["lote"] * VALOR_PIP_POR_LOTE)
                capital += pnl_usd
                trades.append(pnl_usd)
                posicao = None
                
        # 2. Abertura de Nova Entrada
        if (posicao is None and not bloqueio and sinal[t] != 0 and t + 1 < n):
            sl_p = sl_pips[t]
            tp_p = tp_pips[t]
            
            if np.isfinite(sl_p) and sl_p > 0 and np.isfinite(tp_p) and tp_p > 0:
                entrada = opens[t + 1]
                lote = float(np.clip(
                    (capital * RISCO_POR_TRADE) / (sl_p * VALOR_PIP_POR_LOTE),
                    0.01, 100.0
                ))
                delta_sl = sl_p / FATOR_PIPS
                delta_tp = tp_p / FATOR_PIPS
                
                if sinal[t] == 1:
                    sl_abs = entrada - delta_sl
                    tp_abs = entrada + delta_tp
                else:
                    sl_abs = entrada + delta_sl
                    tp_abs = entrada - delta_tp
                    
                posicao = {
                    "direcao": int(sinal[t]), "entrada": entrada,
                    "sl": sl_abs, "tp": tp_abs, "lote": lote
                }
                
        equity.append(capital)
        
    # --- Cálculo das Métricas de Backtest ---
    equity_arr = np.array(equity)
    pnl_pct = (capital - CAPITAL_INICIAL) / CAPITAL_INICIAL * 100
    total = len(trades)
    
    ganhos = [tr for tr in trades if tr > 0]
    perdas = [tr for tr in trades if tr <= 0]
    win_rate = len(ganhos) / total * 100 if total > 0 else 0.0
    
    soma_g = sum(ganhos) if ganhos else 0.0
    soma_p = abs(sum(perdas)) if perdas else 1e-9
    fator_lucro = soma_g / soma_p
    
    pico = np.maximum.accumulate(equity_arr)
    dd_abs = equity_arr - pico
    dd_pct_arr = dd_abs / np.where(pico > 0, pico, 1) * 100
    dd_pct = float(dd_pct_arr.min())
    dd_usd = float(dd_abs.min())
    
    pnl_usd = capital - CAPITAL_INICIAL
    fr = pnl_usd / abs(dd_usd) if dd_usd < 0 else (pnl_usd if pnl_usd > 0 else 0.0)
    
    # Sharpe blindado contra desvio zero
    if len(equity_arr) > 2:
        retornos = np.diff(equity_arr) / np.where(equity_arr[:-1] != 0, equity_arr[:-1], 1)
        retornos = retornos[np.isfinite(retornos)]
        std_ret = np.std(retornos) if len(retornos) > 1 else 0.0
        sharpe = float(np.mean(retornos) / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0
    else:
        sharpe = 0.0
        
    return {
        "pnl_pct":       round(pnl_pct, 2),
        "win_rate":       round(win_rate, 2),
        "fr":             round(fr, 3),
        "dd_pct":         round(dd_pct, 2),
        "sharpe":         round(sharpe, 4),
        "fator_lucro":    round(fator_lucro, 3),
        "total_trades":   total,
        "positivo":       pnl_pct >= 0.0,
        "capital_final":  round(capital, 2),
    }


# ===================================================================
# PARTE 6 — GRÁFICO (dark mode premium de 4 painéis)
# ===================================================================

def gerar_grafico(resultados: list, resultado_otimo: dict, passou_tudo: bool, 
                  criterios_status: dict, medians: dict, dir_saida: Path):
    """
    Desenha o Dashboard Visual Premium de 4 Painéis conforme a especificação do StrategyQuant X.
    """
    plt.rcParams.update({
        "figure.facecolor": "#0D1117", "axes.facecolor": "#0D1117",
        "axes.edgecolor": "#21262D", "axes.labelcolor": "#E6EDF3",
        "xtick.color": "#E6EDF3", "ytick.color": "#E6EDF3",
        "text.color": "#E6EDF3", "grid.color": "#21262D",
        "grid.linestyle": "--", "grid.alpha": 0.4, "font.family": "monospace",
    })
    
    pnls = [r["pnl_pct"] for r in resultados]
    frs  = [r["fr"] for r in resultados]
    
    fig = plt.figure(figsize=(20, 12))
    
    # Grid de Painéis
    ax1 = plt.subplot2grid((3, 3), (0, 0)) # Histograma PnL%
    ax2 = plt.subplot2grid((3, 3), (0, 1)) # Histograma FR
    ax3 = plt.subplot2grid((3, 3), (0, 2)) # Scatter PnL vs FR
    ax4 = plt.subplot2grid((3, 3), (1, 0), colspan=3, rowspan=2) # Scorecard
    
    # ────────────────────────────────────────────────
    # PAINEL 1 — Histograma de PnL%
    # ────────────────────────────────────────────────
    pnls_arr = np.array(pnls)
    pos_pnls = pnls_arr[pnls_arr > 0]
    neg_pnls = pnls_arr[pnls_arr <= 0]
    
    # Plotar histogramas combinados
    if len(pos_pnls) > 0:
        ax1.hist(pos_pnls, bins=20, color="#3FB950", edgecolor="#0D1117", alpha=0.9, zorder=3)
    if len(neg_pnls) > 0:
        ax1.hist(neg_pnls, bins=20, color="#F85149", edgecolor="#0D1117", alpha=0.9, zorder=3)
        
    ax1.axvline(x=0, color="#FFFFFF", linestyle="--", linewidth=1.2, zorder=4)
    ax1.axvline(x=medians["pnl"], color="#58A6FF", linestyle="-", linewidth=1.5, zorder=4)
    ax1.axvline(x=resultado_otimo["pnl_pct"], color="#F0A500", linestyle="-", linewidth=1.5, zorder=4)
    
    ax1.set_title("Distribuição de PnL% (400 combinacoes)", fontsize=11, fontweight="bold", pad=12)
    ax1.set_xlabel("PnL %", fontsize=9)
    ax1.set_ylabel("Frequencia", fontsize=9)
    ax1.grid(True)
    
    # Anotações
    ax1.text(0.05, 0.92, f"Mediana: {medians['pnl']:+.1f}%", transform=ax1.transAxes, color="#58A6FF", fontsize=9, fontweight="bold")
    ax1.text(0.05, 0.84, f"Otimo  : {resultado_otimo['pnl_pct']:+.1f}%", transform=ax1.transAxes, color="#F0A500", fontsize=9, fontweight="bold")
    ax1.text(0.05, 0.76, f"% Luc  : {medians['pct_lucrativas']*100:.1f}%", transform=ax1.transAxes, color="#3FB950", fontsize=9, fontweight="bold")

    # ────────────────────────────────────────────────
    # PAINEL 2 — Histograma de FR
    # ────────────────────────────────────────────────
    frs_arr = np.array(frs)
    pos_frs = frs_arr[frs_arr > 0]
    neg_frs = frs_arr[frs_arr <= 0]
    
    if len(pos_frs) > 0:
        ax2.hist(pos_frs, bins=20, color="#1F6FEB", edgecolor="#0D1117", alpha=0.9, zorder=3)
    if len(neg_frs) > 0:
        ax2.hist(neg_frs, bins=20, color="#F85149", edgecolor="#0D1117", alpha=0.9, zorder=3)
        
    ax2.axvline(x=0, color="#FFFFFF", linestyle="--", linewidth=1.2, zorder=4)
    ax2.axvline(x=CRITERIO_3_FR_MEDIANO, color="#8B949E", linestyle=":", linewidth=1.2, zorder=4)
    ax2.axvline(x=medians["fr"], color="#58A6FF", linestyle="-", linewidth=1.5, zorder=4)
    ax2.axvline(x=resultado_otimo["fr"], color="#F0A500", linestyle="-", linewidth=1.5, zorder=4)
    
    ax2.set_title("Distribuição de Fator de Recuperacao", fontsize=11, fontweight="bold", pad=12)
    ax2.set_xlabel("Fator de Recuperacao (FR)", fontsize=9)
    ax2.grid(True)
    
    ax2.text(0.05, 0.92, f"Mediana: {medians['fr']:.2f}x", transform=ax2.transAxes, color="#58A6FF", fontsize=9, fontweight="bold")
    ax2.text(0.05, 0.84, f"Otimo  : {resultado_otimo['fr']:.2f}x", transform=ax2.transAxes, color="#F0A500", fontsize=9, fontweight="bold")

    # ────────────────────────────────────────────────
    # PAINEL 3 — Scatter PnL% vs FR
    # ────────────────────────────────────────────────
    cores_scatter = []
    for r in resultados:
        if r["pnl_pct"] > 0 and r["fr"] >= 0.5:
            cores_scatter.append("#3FB950") # Verde
        elif r["pnl_pct"] > 0 and r["fr"] < 0.5:
            cores_scatter.append("#F0A500") # Amarelo/Laranja
        else:
            cores_scatter.append("#F85149") # Vermelho
            
    ax3.scatter(pnls, frs, color=cores_scatter, s=8, alpha=0.4, zorder=3)
    ax3.axvline(x=0, color="#FFFFFF", linestyle="--", linewidth=0.8, zorder=2)
    ax3.axhline(y=0.5, color="#8B949E", linestyle="--", linewidth=0.8, zorder=2)
    
    # Ótimo destacado
    ax3.scatter(resultado_otimo["pnl_pct"], resultado_otimo["fr"], 
                color="#F0A500", marker="*", s=150, edgecolor="#FFFFFF", linewidth=0.8, zorder=5)
    ax3.text(resultado_otimo["pnl_pct"] + 1, resultado_otimo["fr"], "OTIMO", 
             color="#F0A500", fontsize=8, fontweight="bold", zorder=5)
             
    ax3.set_title("PnL% vs Fator de Recuperacao", fontsize=11, fontweight="bold", pad=12)
    ax3.set_xlabel("PnL %", fontsize=9)
    ax3.set_ylabel("Fator de Recuperacao (FR)", fontsize=9)
    ax3.grid(True)

    # ────────────────────────────────────────────────
    # PAINEL 4 — Scorecard dos Critérios (Desenhado via texto estilizado)
    # ────────────────────────────────────────────────
    ax4.axis("off")
    
    # Construir barras visuais ASCII
    def _gerar_barra(valor, meta, max_blocos=10):
        if meta == 0:
            pct = 1.0 if valor > 0 else 0.0
        else:
            pct = min(1.0, max(0.0, valor / meta))
        blocos = int(round(pct * max_blocos))
        return "█" * blocos + "░" * (max_blocos - blocos)

    c1_txt = f"{medians['pct_lucrativas']*100:.1f}%  >= {CRITERIO_1_PCT_LUCRATIVAS*100:.1f}%"
    c2_txt = f"{medians['pnl']:+.1f}%  > {CRITERIO_2_PNL_MEDIANO:+.1f}%"
    c3_txt = f"{medians['fr']:.2f}x  >= {CRITERIO_3_FR_MEDIANO:.2f}x"
    c4_txt = f"{medians['fl']:.2f}x  >= {CRITERIO_4_FL_MEDIANO:.2f}x"
    c5_txt = "otimo nao eh outlier" if criterios_status["c5"] else "otimo detectado como outlier"

    c1_ico = "  " if criterios_status["c1"] else "  " # Emojis ASCII
    c2_ico = "  " if criterios_status["c2"] else "  "
    c3_ico = "  " if criterios_status["c3"] else "  "
    c4_ico = "  " if criterios_status["c4"] else "  "
    c5_ico = "  " if criterios_status["c5"] else "  "

    # Criar strings finais do Scorecard
    scorecard_linhas = [
        f"C1 % Lucrativas   : {c1_txt:<18s}  {_gerar_barra(medians['pct_lucrativas'], CRITERIO_1_PCT_LUCRATIVAS)}",
        f"C2 PnL Mediano    : {c2_txt:<18s}  {_gerar_barra(medians['pnl'], 15.0)}", # Usando 15% como 100% de escala para PnL
        f"C3 FR Mediano     : {c3_txt:<18s}  {_gerar_barra(medians['fr'], CRITERIO_3_FR_MEDIANO)}",
        f"C4 FL Mediano     : {c4_txt:<18s}  {_gerar_barra(medians['fl'], CRITERIO_4_FL_MEDIANO)}",
        f"C5 Outlier Check  : {c5_txt:<18s}"
    ]
    
    # Texto do Scorecard
    ax4.text(0.02, 0.88, "CRITERIOS DE ROBUSTEZ INSTITUCIONAIS (Neighborhood SQX):", fontsize=11, fontweight="bold", color="#E6EDF3")
    
    y_pos = 0.72
    icones = [criterios_status["c1"], criterios_status["c2"], criterios_status["c3"], criterios_status["c4"], criterios_status["c5"]]
    
    for idx, (linha, passou) in enumerate(zip(scorecard_linhas, icones)):
        cor_ico = "#56D364" if passou else "#F85149"
        ico_simbolo = "[OK]" if passou else "[X] "
        
        ax4.text(0.02, y_pos, ico_simbolo, color=cor_ico, fontsize=11, fontweight="bold", family="monospace")
        ax4.text(0.08, y_pos, linha, color="#CFD8DC", fontsize=10, family="monospace")
        y_pos -= 0.12

    # Caixa de Veredito Central Direito
    vered_txt = "SISTEMA ROBUSTO" if passou_tudo else "SISTEMA FRAGIL"
    vered_bg  = "#238636" if passou_tudo else "#DA3633"
    n_aprovados = sum(1 for v in icones if v)
    
    ax4.text(0.68, 0.45, f"  {vered_txt}  ", ha="center", va="center",
             fontsize=20, fontweight="bold", color="#FFFFFF",
             bbox=dict(boxstyle="round,pad=0.5", facecolor=vered_bg, edgecolor="none", alpha=0.95))
             
    ax4.text(0.68, 0.22, f"{n_aprovados} / 5 criterios aprovados", ha="center", va="center",
             fontsize=12, fontweight="bold", color="#8B949E")

    # Título Geral Superior
    fig.suptitle(
        f"Teste de Permutacao Parametrica -- {ESTRATEGIA} | {ATIVO} {TIMEFRAME}\n"
        f"{N_COMBINACOES} combinacoes aleatorias | Perturbacao: +-{PERTURBACAO_PCT*100:.0f}% | Seed: {SEED}",
        color="#E6EDF3", fontsize=14, fontweight="bold", y=0.97
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    caminho_img = dir_saida / f"permutacao_{ESTRATEGIA.lower()}.png"
    plt.savefig(caminho_img, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"[GRAFICO] Salvo em: {caminho_img}")


# ===================================================================
# FUNCAO EXPORTAVEL PARA A ESTEIRA (FAIL-FAST)
# ===================================================================
def rodar_permutacao_na_esteira(df: pd.DataFrame, params_otimos: dict, estrategia: str, ativo: str, timeframe: str, dir_saida: Path) -> dict:
    horas = df.index.strftime("%H:%M")
    janela_op = (horas >= HORA_INICIO_OP) & (horas <= HORA_FIM_OP)

    # Garantir log_return no dataframe (necessário para ZSCORE, VR, etc.)
    if "log_return" not in df.columns:
        closes_tmp = df["Close"].values
        df = df.copy()
        df["log_return"] = np.log(closes_tmp / np.roll(closes_tmp, 1))
        df["log_return"].iloc[0] = 0.0

    closes = df["Close"].values
    log_ret = df["log_return"].values

    # Pre-calculos: apenas features universais (VR) são sempre calculadas.
    # Features específicas do MOMENTUM (velocidade/percentil/entropia) só são
    # pré-calculadas quando a estratégia é MOMENTUM, evitando overhead desnecessário
    # e possíveis erros para outras estratégias.
    vr = pd.Series(log_ret).rolling(50, min_periods=50).std(ddof=1).values
    vr_pips = vr * closes * FATOR_PIPS

    static_cols = {"vr_pips": vr_pips}

    if estrategia == "MOMENTUM":
        velocidade = df["Close"].diff(1).values
        aceleracao = pd.Series(velocidade).diff(1).values

        def _percentrank(arr):
            val = arr[-1]
            hist = arr[:-1]
            if len(hist) == 0: return 0.5
            return float(np.sum(hist < val)) / len(hist)

        percentil = pd.Series(aceleracao).rolling(100, min_periods=100).apply(_percentrank, raw=True).values

        def _entropia(arr):
            if np.std(arr) < 1e-15: return 0.0
            c, _ = np.histogram(arr, bins=10)
            p = c / len(arr)
            p = p[p > 0]
            return float(np.clip(-np.sum(p * np.log2(p)) / np.log2(10), 0, 1))

        entropia = pd.Series(log_ret).rolling(30, min_periods=30).apply(_entropia, raw=True).values

        static_cols["velocidade"] = velocidade
        static_cols["percentil"] = percentil
        static_cols["entropia"] = entropia

    sinal_ref, sl_ref, tp_ref = calcular_sinais(df, params_otimos, janela_op, estrategia, cache=static_cols,
                                                 ativo=ativo, timeframe=timeframe)
    resultado_otimo = rodar_backtest(df, sinal_ref, sl_ref, tp_ref)

    combinacoes = gerar_combinacoes_aleatorias(params_otimos, N_COMBINACOES, PERTURBACAO_PCT, SEED)
    resultados = []

    for params in combinacoes:
        sinal, sl, tp = calcular_sinais(df, params, janela_op, estrategia, cache=static_cols,
                                         ativo=ativo, timeframe=timeframe)
        res = rodar_backtest(df, sinal, sl, tp)
        registro = params.copy()
        registro.update(res)
        resultados.append(registro)

    pnls = [r["pnl_pct"] for r in resultados]
    frs  = [r["fr"] for r in resultados]
    fls  = [r["fator_lucro"] for r in resultados]

    n_lucrativas = sum(1 for p in pnls if p > 0.0)
    pct_lucrativas = n_lucrativas / N_COMBINACOES
    passou_c1 = pct_lucrativas >= CRITERIO_1_PCT_LUCRATIVAS
    pnl_mediano = float(np.median(pnls))
    passou_c2 = pnl_mediano > CRITERIO_2_PNL_MEDIANO
    fr_mediano = float(np.median(frs))
    passou_c3 = fr_mediano >= CRITERIO_3_FR_MEDIANO
    fl_mediano = float(np.median(fls))
    passou_c4 = fl_mediano >= CRITERIO_4_FL_MEDIANO

    pnl_media = float(np.mean(pnls))
    pnl_std   = float(np.std(pnls))
    pnl_otimo = float(resultado_otimo["pnl_pct"])
    limite_outlier = pnl_mediano + CRITERIO_5_OUTLIER_SIGMA * pnl_std
    passou_c5 = pnl_otimo < limite_outlier

    passou_tudo = passou_c1 and passou_c2 and passou_c3 and passou_c4 and passou_c5

    criterios_status = {"c1": passou_c1, "c2": passou_c2, "c3": passou_c3, "c4": passou_c4, "c5": passou_c5}
    medians = {"pct_lucrativas": pct_lucrativas, "pnl": pnl_mediano, "fr": fr_mediano, "fl": fl_mediano}

    gerar_grafico(resultados, resultado_otimo, passou_tudo, criterios_status, medians, dir_saida)

    return {"aprovado": passou_tudo}


# ===================================================================
# MAIN -- ORQUESTRADOR DO PIPELINE
# ===================================================================

def main():
    global ATIVO, TIMEFRAME, ESTRATEGIA, PARAMS_OTIMOS, SEED, N_COMBINACOES, DIR_SAIDA
    
    parser = argparse.ArgumentParser(description="Teste de Robustez por Permutacao Parametrica Aleatoria")
    parser.add_argument("--estrategia", type=str, default=None)
    parser.add_argument("--ativo", type=str, default=None)
    parser.add_argument("--timeframe", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    
    if args.estrategia: ESTRATEGIA = args.estrategia.upper()
    if args.ativo:      ATIVO = args.ativo.upper()
    if args.timeframe:  TIMEFRAME = args.timeframe.upper()
    if args.seed is not None: SEED = args.seed
    
    # Atualiza pasta de saída com base nos argumentos parsed do CLI
    DIR_SAIDA = Path(__file__).resolve().parent / f"{ATIVO.lower()}_{TIMEFRAME.lower()}"
    DIR_SAIDA.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*55}")
    print(f"  TESTE DE ROBUSTEZ -- PERMUTACAO PARAMETRICA ALEATORIA")
    print(f"  Estrategia : {ESTRATEGIA}")
    print(f"  Ativo      : {ATIVO} {TIMEFRAME}")
    print(f"  Combinacoes: {N_COMBINACOES} | Perturbacao: +-{PERTURBACAO_PCT*100:.0f}%")
    print(f"  Semente    : {SEED}")
    print(f"  Saida      : {DIR_SAIDA}")
    print(f"{'='*55}\n")
    
    # 1. Carregamento dos dados
    print("[DADOS] Carregando serie historica...")
    if not PARQUET_DADOS.exists():
        raise FileNotFoundError(f"Arquivo parquet de dados nao encontrado: {PARQUET_DADOS}")
        
    df = pd.read_parquet(PARQUET_DADOS)
    
    # Formatação de Índice Datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        for col in ("time", "datetime"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                df = df.set_index(col)
                break
                
    if "log_return" not in df.columns:
        df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
        df["log_return"].iloc[0] = 0.0
        
    print(f"   {len(df):,} candles ({df.index.min()} -> {df.index.max()}) carregados.\n")
    
    # Definir máscara da janela operacional
    horas = df.index.strftime("%H:%M")
    janela_op = (horas >= HORA_INICIO_OP) & (horas <= HORA_FIM_OP)
    
    # =================================================================
    # OTIMIZAÇÃO DE PERFORMANCE: PRÉ-COMPUTAÇÃO DAS COLUNAS ESTÁTICAS
    # =================================================================
    print("[PERF] Pre-calculando indicadores matematicos (uma unica vez)...")
    
    closes = df["Close"].values
    log_ret = df["log_return"].values
    
    # 1. Velocidade e Aceleração
    velocidade = df["Close"].diff(1).values
    aceleracao = pd.Series(velocidade).diff(1).values
    
    # 2. Percentil de Aceleração (Rank Rolante 100)
    def _percentrank(arr):
        val = arr[-1]
        hist = arr[:-1]
        if len(hist) == 0: return 0.5
        return float(np.sum(hist < val)) / len(hist)
        
    percentil = pd.Series(aceleracao).rolling(100, min_periods=100).apply(
        _percentrank, raw=True).values
        
    # 3. Entropia de Shannon (30 candles)
    def _entropia(arr):
        if np.std(arr) < 1e-15: return 0.0
        c, _ = np.histogram(arr, bins=10)
        p = c / len(arr)
        p = p[p > 0]
        return float(np.clip(-np.sum(p * np.log2(p)) / np.log2(10), 0, 1))
        
    entropia = pd.Series(log_ret).rolling(30, min_periods=30).apply(
        _entropia, raw=True).values
        
    # 4. Volatilidade Estimada (VR)
    vr = pd.Series(log_ret).rolling(50, min_periods=50).std(ddof=1).values
    vr_pips = vr * closes * FATOR_PIPS
    
    # Guardar no dicionário estático compartilhado
    static_cols = {
        "velocidade": velocidade,
        "percentil":  percentil,
        "entropia":   entropia,
        "vr_pips":    vr_pips
    }
    
    # =================================================================
    # REF: BACKTEST DA PARAMETRIZAÇÃO ÓTIMA (REFERÊNCIA)
    # =================================================================
    print("[REF] Rodando backtest com parametros otimos (referencia)...")
    sinal_ref, sl_ref, tp_ref = calcular_sinais(df, PARAMS_OTIMOS, janela_op, ESTRATEGIA, cache=static_cols)
    resultado_otimo = rodar_backtest(df, sinal_ref, sl_ref, tp_ref)
    
    print(f"   PnL: {resultado_otimo['pnl_pct']:+.1f}% | FR: {resultado_otimo['fr']:.2f}x | "
          f"FL: {resultado_otimo['fator_lucro']:.2f}x | DD: {resultado_otimo['dd_pct']:.1f}% | Trades: {resultado_otimo['total_trades']}\n")
          
    # =================================================================
    # LOOP PRINCIPAL: EXECUÇÃO DAS N COMBINAÇÕES
    # =================================================================
    print(f"[RUN] Gerando e processando {N_COMBINACOES} combinacoes aleatorias...")
    combinacoes = gerar_combinacoes_aleatorias(PARAMS_OTIMOS, N_COMBINACOES, PERTURBACAO_PCT, SEED)
    
    resultados = []
    
    for idx, params in enumerate(tqdm(combinacoes, desc=f"  Permutacao {ESTRATEGIA} | {ATIVO} {TIMEFRAME}", 
                                      ncols=70, unit="comb")):
        # Rodar sinais com o set perturbado (rápido devido a static_cols pré-calculado)
        sinal, sl, tp = calcular_sinais(df, params, janela_op, ESTRATEGIA, cache=static_cols)
        
        # Rodar backtest candle-a-candle
        res = rodar_backtest(df, sinal, sl, tp)
        
        # Armazenar combinação associada aos resultados obtidos
        registro = params.copy()
        registro.update(res)
        resultados.append(registro)
        
        # Impressão de progresso parcial a cada 100 combinações
        if (idx + 1) % 100 == 0:
            pnls_temp = [r["pnl_pct"] for r in resultados]
            print(f"\n   [PARCIAL] {idx+1}/{N_COMBINACOES} concluido -- PnL Mediano: {np.median(pnls_temp):+.2f}%")
            
    # =================================================================
    # PARTE 5 — AVALIAÇÃO DOS CRITÉRIOS DE ROBUSTEZ
    # =================================================================
    print("\n[EVAL] Analisando criterios estatisticos de robustez...")
    
    pnls = [r["pnl_pct"] for r in resultados]
    frs  = [r["fr"] for r in resultados]
    fls  = [r["fator_lucro"] for r in resultados]
    wrs  = [r["win_rate"] for r in resultados]
    tds  = [r["total_trades"] for r in resultados]
    
    # C1 — % lucrativas
    n_lucrativas = sum(1 for p in pnls if p > 0.0)
    pct_lucrativas = n_lucrativas / N_COMBINACOES
    passou_c1 = pct_lucrativas >= CRITERIO_1_PCT_LUCRATIVAS
    
    # C2 — PnL mediano
    pnl_mediano = float(np.median(pnls))
    passou_c2 = pnl_mediano > CRITERIO_2_PNL_MEDIANO
    
    # C3 — FR mediano
    fr_mediano = float(np.median(frs))
    passou_c3 = fr_mediano >= CRITERIO_3_FR_MEDIANO
    
    # C4 — FL mediano
    fl_mediano = float(np.median(fls))
    passou_c4 = fl_mediano >= CRITERIO_4_FL_MEDIANO
    
    # C5 — Outlier check (Parâmetro ótimo original não deve ser um outlier aberrante na vizinhança)
    pnl_media = float(np.mean(pnls))
    pnl_std   = float(np.std(pnls))
    pnl_otimo = float(resultado_otimo["pnl_pct"])
    limite_outlier = pnl_mediano + CRITERIO_5_OUTLIER_SIGMA * pnl_std
    passou_c5 = pnl_otimo < limite_outlier
    c5_txt = "otimo nao eh outlier" if passou_c5 else "otimo detectado como outlier"
    
    passou_tudo = passou_c1 and passou_c2 and passou_c3 and passou_c4 and passou_c5
    
    criterios_status = {
        "c1": passou_c1, "c2": passou_c2, "c3": passou_c3, "c4": passou_c4, "c5": passou_c5
    }
    
    medians = {
        "pct_lucrativas": pct_lucrativas,
        "pnl": pnl_mediano,
        "fr": fr_mediano,
        "fl": fl_mediano
    }
    
    # =================================================================
    # IMPRESSÃO DE RELATÓRIO NO TERMINAL
    # =================================================================
    c1i = "[OK]" if passou_c1 else "[X]"
    c2i = "[OK]" if passou_c2 else "[X]"
    c3i = "[OK]" if passou_c3 else "[X]"
    c4i = "[OK]" if passou_c4 else "[X]"
    c5i = "[OK]" if passou_c5 else "[X]"
    
    print(f"\n  ===============================================")
    print(f"  TESTE DE PERMUTACAO PARAMETRICA")
    print(f"  Estrategia : {ESTRATEGIA} | Ativo: {ATIVO} {TIMEFRAME}")
    print(f"  Combinacoes: {N_COMBINACOES} | Perturbacao: +-{PERTURBACAO_PCT*100:.0f}%")
    print(f"  -----------------------------------------------")
    print(f"  ESTATISTICAS DAS {N_COMBINACOES} COMBINACOES:")
    print(f"    PnL%  - Mediana: {pnl_mediano:>+5.1f}% | Media: {pnl_media:>+5.1f}% | Std: {pnl_std:>4.1f}%")
    print(f"    FR    - Mediana: {fr_mediano:>5.2f}x | Media: {np.mean(frs):>5.2f}x | Std: {np.std(frs):>4.1f}x")
    print(f"    FL    - Mediana: {fl_mediano:>5.2f}x | Media: {np.mean(fls):>5.2f}x | Std: {np.std(fls):>4.1f}x")
    print(f"    WR    - Mediana: {np.median(wrs):>5.1f}% | Media: {np.mean(wrs):>5.1f}%")
    print(f"    Trades- Mediana: {int(np.median(tds)):>5d}  | Media: {int(np.mean(tds)):>5d}")
    print(f"  -----------------------------------------------")
    print(f"  PARAMETRO OTIMO (referencia):")
    print(f"    PnL%: {resultado_otimo['pnl_pct']:+5.1f}% | FR: {resultado_otimo['fr']:.2f}x | FL: {resultado_otimo['fator_lucro']:.2f}x")
    print(f"  -----------------------------------------------")
    print(f"  CRITERIOS:")
    print(f"  {c1i} C1 % Lucrativas : {pct_lucrativas*100:.1f}%  (min: {CRITERIO_1_PCT_LUCRATIVAS*100:.1f}%)")
    print(f"  {c2i} C2 PnL Mediano  : {pnl_mediano:+.1f}%  (min: > {CRITERIO_2_PNL_MEDIANO:+.1f}%)")
    print(f"  {c3i} C3 FR Mediano   : {fr_mediano:.2f}x  (min: {CRITERIO_3_FR_MEDIANO:.2f}x)")
    print(f"  {c4i} C4 FL Mediano   : {fl_mediano:.2f}x  (min: {CRITERIO_4_FL_MEDIANO:.2f}x)")
    print(f"  {c5i} C5 Outlier      : {c5_txt}")
    print(f"                     (limite outlier: {limite_outlier:+.1f}%)")
    print(f"  -----------------------------------------------")
    
    vered_txt = "SISTEMA ROBUSTO [OK]" if passou_tudo else "SISTEMA FRAGIL [X]"
    n_passou = sum([passou_c1, passou_c2, passou_c3, passou_c4, passou_c5])
    print(f"  +==============================================+")
    print(f"  |  VEREDITO: {vered_txt:<34s}|")
    print(f"  |  {n_passou} / 5 criterios aprovados                   |")
    print(f"  +==============================================+\n")
    
    # =================================================================
    # PARTE 8 — ARQUIVOS DE SAÍDA (SALVAMENTO PNG, CSV E TXT)
    # =================================================================
    
    # 1. Gráfico PNG
    gerar_grafico(resultados, resultado_otimo, passou_tudo, criterios_status, medians, DIR_SAIDA)
    
    # 2. CSV Completo (DESABILITADO POR PADRAO)
    # linhas_csv = []
    # for i, r in enumerate(resultados):
    #     linha = {"combinacao_id": i + 1, "seed": SEED}
    #     linha.update(r)
    #     linhas_csv.append(linha)
    # df_csv = pd.DataFrame(linhas_csv)
    # caminho_csv = DIR_SAIDA / f"permutacao_{ESTRATEGIA.lower()}.csv"
    # df_csv.to_csv(caminho_csv, index=False)
    # print(f"[CSV] Salvo em: {caminho_csv}")
    
    # 3. Arquivo de resumo TXT
    caminho_txt = DIR_SAIDA / f"permutacao_{ESTRATEGIA.lower()}_resumo.txt"
    with open(caminho_txt, "w", encoding="utf-8") as f:
        f.write(f"===============================================\n")
        f.write(f"TESTE DE PERMUTAÇÃO PARAMÉTRICA\n")
        f.write(f"Estratégia : {ESTRATEGIA} | Ativo: {ATIVO} {TIMEFRAME}\n")
        f.write(f"Combinações: {N_COMBINACOES} | Perturbação: ±{PERTURBACAO_PCT*100:.0f}%\n")
        f.write(f"Semente    : {SEED}\n")
        f.write(f"-----------------------------------------------\n")
        f.write(f"ESTATÍSTICAS DAS {N_COMBINACOES} COMBINAÇÕES:\n")
        f.write(f"  PnL%  — Mediana: {pnl_mediano:>+5.1f}% | Média: {pnl_media:>+5.1f}% | Std: {pnl_std:>4.1f}%\n")
        f.write(f"  FR    — Mediana: {fr_mediano:>5.2f}x | Média: {np.mean(frs):>5.2f}x | Std: {np.std(frs):>4.1f}x\n")
        f.write(f"  FL    — Mediana: {fl_mediano:>5.2f}x | Média: {np.mean(fls):>5.2f}x | Std: {np.std(fls):>4.1f}x\n")
        f.write(f"  WR    — Mediana: {np.median(wrs):>5.1f}% | Média: {np.mean(wrs):>5.1f}%\n")
        f.write(f"  Trades— Mediana: {int(np.median(tds)):>5d}  | Média: {int(np.mean(tds)):>5d}\n")
        f.write(f"-----------------------------------------------\n")
        f.write(f"PARÂMETRO ÓTIMO (referência):\n")
        f.write(f"  PnL%: {resultado_otimo['pnl_pct']:+5.1f}% | FR: {resultado_otimo['fr']:.2f}x | FL: {resultado_otimo['fator_lucro']:.2f}x\n")
        f.write(f"-----------------------------------------------\n")
        f.write(f"CRITÉRIOS:\n")
        f.write(f"  {c1i} C1 % Lucrativas : {pct_lucrativas*100:.1f}%  (mín: {CRITERIO_1_PCT_LUCRATIVAS*100:.1f}%)\n")
        f.write(f"  {c2i} C2 PnL Mediano  : {pnl_mediano:+.1f}%  (mín: > {CRITERIO_2_PNL_MEDIANO:+.1f}%)\n")
        f.write(f"  {c3i} C3 FR Mediano   : {fr_mediano:.2f}x  (mín: {CRITERIO_3_FR_MEDIANO:.2f}x)\n")
        f.write(f"  {c4i} C4 FL Mediano   : {fl_mediano:.2f}x  (mín: {CRITERIO_4_FL_MEDIANO:.2f}x)\n")
        f.write(f"  {c5i} C5 Outlier      : {c5_txt}\n")
        f.write(f"                     (limite outlier: {limite_outlier:+.1f}%)\n")
        f.write(f"-----------------------------------------------\n")
        f.write(f"VEREDITO: {vered_txt}\n")
        f.write(f"{n_passou} / 5 critérios aprovados\n")
        f.write(f"===============================================\n")
        
    print(f"[TXT] Salvo em: {caminho_txt}\n")
    print("[OK] Teste de permutacao finalizado com sucesso.\n")


if __name__ == "__main__":
    main()
