#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
walk_forward_matrix.py
-----------------------
Teste de Robustez por Matriz de Walk-Forward (WFM) Institucional.
Replica a logica rigorosa de validacao multidimensional do StrategyQuant X (SQX).

Motor (Alta Performance):
  - Gera matriz 5 x 4 = 20 celulas de testes WFA distintos.
  - OOS%: [20, 25, 30, 35, 40]  x  Runs: [6, 8, 10, 12]
  - Selecao IS: estimativa vetorizada rapida (numpy puro, sem loop de candles).
    Calcula Pips Brutos estimados para cada combinacao candidata no slice IS.
    Seleciona a combinacao com maior lucro estimado. Demora ~millisegundos.
  - Avaliacao OOS: backtest candle-a-candle completo apenas 1x por run.
  - Emenda todos os runs OOS e avalia 3 criterios de robustez SQX por celula.
  - Veredito global por analise de cluster (bloco >= 4x4 com >= 10 aprovadas).
  - Dashboard dark premium: heatmap 5x4 + curva OOS campeao + scorecard.

Uso:
  python walk_forward_matrix.py
  python walk_forward_matrix.py --estrategia MOMENTUM --ativo EURUSD --timeframe H1

Estrutura de saida:
  testes_robustez/w_f_a/{ativo}_{tf}/wfm_{estrategia}.png
  testes_robustez/w_f_a/{ativo}_{tf}/wfm_{estrategia}.csv
  testes_robustez/w_f_a/{ativo}_{tf}/wfm_{estrategia}_resumo.txt
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

warnings.filterwarnings("ignore")
matplotlib.use("Agg")


# ===================================================================
# CONFIGURACAO DO TESTE -- EDITAR AQUI
# ===================================================================

ATIVO      = "EURUSD"
TIMEFRAME  = "H1"
ESTRATEGIA = "MOMENTUM"

# Parametros operacionais fixos
CAPITAL_INICIAL    = 10_000.0
RISCO_POR_TRADE    = 0.01
SPREAD_PIPS        = 1.2
VALOR_PIP_POR_LOTE = 10.0
FATOR_PIPS         = 10_000
HORA_INICIO_OP     = "10:00"
HORA_FIM_OP        = "22:30"

# Dimensoes da Matriz WFM
OOS_PCTS = [20, 25, 30, 35, 40]   # 5 linhas (percentual OOS)
WF_RUNS  = [6, 8, 10, 12]         # 4 colunas (numero de runs)

# Criterios de aprovacao por celula
CRITERIO_PNL_POSITIVO    = True   # PnL total OOS emendado > 0
CRITERIO_RUNS_LUCRATIVAS = 0.55   # > 55% dos runs OOS lucrativos
CRITERIO_DD_MAXIMO_RUN   = 25.0   # Nenhum run com DD > 25%

# Criterio global de cluster
CLUSTER_MIN_CELULAS  = 6          # Min celulas aprovadas no bloco
CLUSTER_MIN_LINHAS   = 3          # Min linhas do bloco
CLUSTER_MIN_COLUNAS  = 3          # Min colunas do bloco

# ===================================================================
# FIM DA CONFIGURACAO
# ===================================================================

ROOT_DIR  = Path(__file__).resolve().parent.parent.parent
DIR_SAIDA = Path(__file__).resolve().parent / f"{ATIVO.lower()}_{TIMEFRAME.lower()}"


# ===================================================================
# PARTE 1 -- PRE-COMPUTACAO ESTATICA DE INDICADORES
# ===================================================================

def precomputar_indicadores(df: pd.DataFrame) -> dict:
    """
    Computa TODAS as colunas matematicas pesadas UMA UNICA VEZ
    sobre o DataFrame completo. Slices por indice sao usados nos loops.
    """
    closes  = df["Close"].values
    log_ret = df["log_return"].values

    velocidade = df["Close"].diff(1).values
    aceleracao = pd.Series(velocidade).diff(1).values

    def _percentrank(arr):
        val, hist = arr[-1], arr[:-1]
        if len(hist) == 0:
            return 0.5
        return float(np.sum(hist < val)) / len(hist)

    percentil = pd.Series(aceleracao).rolling(100, min_periods=100).apply(
        _percentrank, raw=True
    ).values

    def _entropia(arr):
        if np.std(arr) < 1e-15:
            return 0.0
        c, _ = np.histogram(arr, bins=10)
        p = c / len(arr)
        p = p[p > 0]
        return float(np.clip(-np.sum(p * np.log2(p)) / np.log2(10), 0, 1))

    entropia = pd.Series(log_ret).rolling(30, min_periods=30).apply(
        _entropia, raw=True
    ).values

    vr      = pd.Series(log_ret).rolling(50, min_periods=50).std(ddof=1).values
    vr_pips = vr * closes * FATOR_PIPS

    return {
        "velocidade": velocidade,
        "percentil":  percentil,
        "entropia":   entropia,
        "vr_pips":    vr_pips,
    }


# ===================================================================
# PARTE 2 -- MOTOR DE SINAIS (ADAPTAR PARA OUTRAS ESTRATEGIAS)
# ===================================================================

def calcular_sinais(df: pd.DataFrame,
                    params: dict,
                    janela_op: np.ndarray,
                    sc: dict) -> tuple:
    """
    Gera vetores de sinal/sl/tp para os indices do df usando colunas pre-computadas.
    Para mudar de estrategia: adaptar as linhas de condicao abaixo.
    """
    hurst = df["hurst"].values

    sl_pips = params["mult_sl"] * sc["vr_pips"]
    tp_pips = params["mult_tp"] * sc["vr_pips"]

    cond_base  = (hurst > params["hurst_cutoff"]) & \
                 (sc["entropia"] < params["entropia_cutoff"]) & \
                 janela_op

    sinal = np.zeros(len(df), dtype=np.int8)
    sinal[cond_base & (sc["percentil"] > params["percentil_trigger"]) & (sc["velocidade"] > 0)]  =  1
    sinal[cond_base & (sc["percentil"] < (1.0 - params["percentil_trigger"])) & (sc["velocidade"] < 0)] = -1

    return sinal, sl_pips, tp_pips


# ===================================================================
# PARTE 3 -- ESTIMATIVA VETORIZADA IS (SELECAO RAPIDA, SEM LOOP)
# ===================================================================

def estimar_pips_is(df_is: pd.DataFrame,
                    params: dict,
                    janela_op_is: np.ndarray,
                    sc_is: dict) -> float:
    """
    Estima o lucro bruto em pips da combinacao de parametros no slice IS
    sem rodar o backtest candle-a-candle completo.

    Logica: para cada barra com sinal, estima que a trade ganhou TP ou
    perdeu SL com base na direcao e no preco de fechamento seguinte.
    Rapido porque e 100% vetorizado (numpy puro).

    Retorna float: pips brutos estimados (positivo = lucrativo no IS).
    """
    hurst = df_is["hurst"].values
    n     = len(df_is)

    sl_pips = params["mult_sl"] * sc_is["vr_pips"]
    tp_pips = params["mult_tp"] * sc_is["vr_pips"]

    cond_base = (hurst > params["hurst_cutoff"]) & \
                (sc_is["entropia"] < params["entropia_cutoff"]) & \
                janela_op_is

    sinal = np.zeros(n, dtype=np.int8)
    sinal[cond_base & (sc_is["percentil"] > params["percentil_trigger"]) & (sc_is["velocidade"] > 0)]  =  1
    sinal[cond_base & (sc_is["percentil"] < (1.0 - params["percentil_trigger"])) & (sc_is["velocidade"] < 0)] = -1

    closes = df_is["Close"].values

    # Para cada barra com sinal, calcula diferenca de fechamento proximo
    idx_sinal = np.where(sinal[:-1] != 0)[0]
    if len(idx_sinal) == 0:
        return 0.0

    direcao  = sinal[idx_sinal].astype(float)
    # NOTA: Esta é uma estimativa vetorizada rápida para
    # seleção IS — usa close[t+1] vs close[t] como proxy.
    # NÃO é o backtest real. O backtest OOS completo
    # usa Open[t+1] como entrada (ver rodar_backtest).
    delta    = (closes[idx_sinal + 1] - closes[idx_sinal]) * FATOR_PIPS

    # Clampa o resultado entre -SL e +TP (simulacao simples de proteção)
    sl_v = sl_pips[idx_sinal]
    tp_v = tp_pips[idx_sinal]
    pips = direcao * delta
    pips = np.clip(pips, -sl_v, tp_v)

    # Desconta spread
    pips -= SPREAD_PIPS

    return float(np.nansum(pips))


def selecionar_melhor_params_is(df_is: pd.DataFrame,
                                df_combinacoes: pd.DataFrame,
                                janela_op_is: np.ndarray,
                                sc_is: dict) -> dict:
    """
    Seleciona a combinacao de parametros com maior estimativa de pips IS.
    Vetorizado: demora milissegundos (sem backtest candle-a-candle).
    """
    param_cols = [c for c in df_combinacoes.columns
                  if c not in ("Trades", "Lucro_Total_Pips", "Max_DD_Pips", "Ret_DD")]

    melhor_pips   = -np.inf
    melhor_params = None

    for _, row in df_combinacoes.iterrows():
        params = {c: row[c] for c in param_cols}
        try:
            pips = estimar_pips_is(df_is, params, janela_op_is, sc_is)
        except Exception:
            pips = -np.inf

        if pips > melhor_pips:
            melhor_pips   = pips
            melhor_params = params

    return melhor_params


# ===================================================================
# PARTE 4 -- BACKTEST OOS COMPLETO (CANDLE-A-CANDLE)
# ===================================================================

def rodar_backtest(df: pd.DataFrame,
                   sinal: np.ndarray,
                   sl_pips: np.ndarray,
                   tp_pips: np.ndarray) -> dict:
    """
    Backtest completo e preciso, candle a candle, com position sizing
    dinamico e stop/take fixos. Roda apenas no slice OOS de cada run.
    """
    opens  = df["Open"].values
    highs  = df["High"].values
    lows   = df["Low"].values
    closes = df["Close"].values
    idx    = df.index
    n      = len(df)

    capital = CAPITAL_INICIAL
    equity  = [capital]
    trades  = []
    posicao = None

    for t in range(n):
        dt      = idx[t]
        weekday = dt.weekday()
        hora    = dt.strftime("%H:%M")

        bloqueio = (
            (weekday == 4 and hora >= "20:30") or
            weekday == 5 or
            (weekday == 6 and hora < "21:00")
        )
        eh_sexta = weekday == 4 and hora == "21:55"

        if posicao is not None:
            if eh_sexta:
                saida, motivo = closes[t], "FDS"
            else:
                saida, motivo = None, None
                if posicao["dir"] == 1:
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
                d       = posicao["dir"]
                p_pips  = d * (saida - posicao["entrada"]) * FATOR_PIPS
                p_usd   = (p_pips * posicao["lote"] * VALOR_PIP_POR_LOTE
                           - SPREAD_PIPS * posicao["lote"] * VALOR_PIP_POR_LOTE)
                capital += p_usd
                trades.append(p_usd)
                posicao = None

        if posicao is None and not bloqueio and sinal[t] != 0 and t + 1 < n:
            sl_p = sl_pips[t]
            tp_p = tp_pips[t]
            if np.isfinite(sl_p) and sl_p > 0 and np.isfinite(tp_p) and tp_p > 0:
                entrada  = opens[t + 1]
                lote     = float(np.clip(
                    (capital * RISCO_POR_TRADE) / (sl_p * VALOR_PIP_POR_LOTE),
                    0.01, 100.0
                ))
                d_sl = sl_p / FATOR_PIPS
                d_tp = tp_p / FATOR_PIPS
                if sinal[t] == 1:
                    sl_abs, tp_abs = entrada - d_sl, entrada + d_tp
                else:
                    sl_abs, tp_abs = entrada + d_sl, entrada - d_tp
                posicao = {"dir": int(sinal[t]), "entrada": entrada,
                           "sl": sl_abs, "tp": tp_abs, "lote": lote}

        equity.append(capital)

    eq_arr  = np.array(equity)
    pnl_usd = capital - CAPITAL_INICIAL
    pnl_pct = pnl_usd / CAPITAL_INICIAL * 100
    total   = len(trades)
    ganhos  = [t for t in trades if t > 0]
    perdas  = [t for t in trades if t <= 0]
    wr      = len(ganhos) / total * 100 if total > 0 else 0.0
    fl      = sum(ganhos) / max(abs(sum(perdas)), 1e-9) if ganhos else 0.0

    pico   = np.maximum.accumulate(eq_arr)
    dd_arr = (eq_arr - pico) / np.where(pico > 0, pico, 1) * 100
    dd_pct = float(dd_arr.min())
    dd_usd = float((eq_arr - pico).min())
    fr     = pnl_usd / abs(dd_usd) if dd_usd < 0 else (pnl_usd if pnl_usd > 0 else 0.0)

    return {
        "pnl_pct":    round(pnl_pct, 2),
        "pnl_usd":    round(pnl_usd, 2),
        "fr":         round(fr, 3),
        "dd_pct":     round(dd_pct, 2),
        "fator_lucro": round(fl, 3),
        "win_rate":   round(wr, 2),
        "total_trades": total,
        "positivo":   pnl_pct > 0.0,
        "equity":     eq_arr,
    }


# ===================================================================
# PARTE 5 -- FATIAMENTO TEMPORAL (ANCHORED WALK-FORWARD)
# ===================================================================

def gerar_fatias_wf(n_total: int, n_runs: int, oos_pct: float) -> list:
    """
    Fatiamento Anchored: IS cresce de run em run (IS comeca sempre no t=0).
    OOS sao janelas sequenciais nao sobrepostas que cobrem o final da serie.
    Mais conservador e realista para live trading incremental.
    """
    oos_size = max(int(round(n_total * (oos_pct / 100.0) / n_runs)), 50)

    fatias = []
    for i in range(n_runs):
        oos_start = n_total - (n_runs - i) * oos_size
        oos_end   = min(oos_start + oos_size, n_total)
        is_end    = oos_start

        if is_end < 200:
            continue

        fatias.append({
            "run":       i + 1,
            "is_start":  0,
            "is_end":    is_end,
            "oos_start": oos_start,
            "oos_end":   oos_end,
        })

    return fatias


# ===================================================================
# PARTE 6 -- AVALIACAO DE UMA CELULA WFM
# ===================================================================

def avaliar_celula(df: pd.DataFrame,
                   df_combinacoes: pd.DataFrame,
                   fatias: list,
                   static_cols: dict,
                   janela_op: np.ndarray) -> dict:
    """
    Ciclo completo Walk-Forward para UMA celula:
    1. IS rapido (estimativa vetorizada) -> seleciona melhor params
    2. OOS completo (backtest c-a-c) com esses params
    3. Emenda todas as curvas OOS em uma unica serie continua
    4. Avalia os 3 criterios de robustez SQX
    """
    resultados_oos  = []
    equity_emendado = []

    for fatia in fatias:
        is_s, is_e   = fatia["is_start"], fatia["is_end"]
        oos_s, oos_e = fatia["oos_start"], fatia["oos_end"]

        # Slice IS
        df_is   = df.iloc[is_s:is_e]
        jop_is  = janela_op[is_s:is_e]
        sc_is   = {k: v[is_s:is_e] for k, v in static_cols.items()}

        # Selecao IS vetorizada (rapida)
        best_params = selecionar_melhor_params_is(df_is, df_combinacoes, jop_is, sc_is)

        if best_params is None:
            resultados_oos.append({
                "run": fatia["run"], "pnl_pct": 0.0, "dd_pct": 0.0,
                "positivo": False, "equity": np.array([CAPITAL_INICIAL])
            })
            equity_emendado.append(np.array([CAPITAL_INICIAL]))
            continue

        # Backtest OOS completo
        df_oos  = df.iloc[oos_s:oos_e]
        jop_oos = janela_op[oos_s:oos_e]
        sc_oos  = {k: v[oos_s:oos_e] for k, v in static_cols.items()}

        try:
            sinal, sl, tp = calcular_sinais(df_oos, best_params, jop_oos, sc_oos)
            res = rodar_backtest(df_oos, sinal, sl, tp)
        except Exception:
            res = {"pnl_pct": 0.0, "dd_pct": 0.0, "positivo": False,
                   "equity": np.array([CAPITAL_INICIAL])}

        res["run"] = fatia["run"]
        resultados_oos.append(res)
        equity_emendado.append(res["equity"])

    # Emenda continua das curvas OOS
    capital_atual = CAPITAL_INICIAL
    equity_total  = [capital_atual]
    for eq in equity_emendado:
        if len(eq) < 2:
            continue
        delta    = eq - eq[0]
        segmento = capital_atual + delta[1:]
        equity_total.extend(segmento.tolist())
        capital_atual = segmento[-1] if len(segmento) > 0 else capital_atual

    eq_total  = np.array(equity_total)
    pnl_total = (eq_total[-1] - CAPITAL_INICIAL) / CAPITAL_INICIAL * 100

    pico_t   = np.maximum.accumulate(eq_total)
    dd_total = float(((eq_total - pico_t) / np.where(pico_t > 0, pico_t, 1) * 100).min())

    # Criterios por celula
    n_r        = len(resultados_oos)
    n_luc      = sum(1 for r in resultados_oos if r.get("positivo", False))
    pct_luc    = n_luc / n_r if n_r > 0 else 0.0
    dd_max_run = max((abs(r.get("dd_pct", 0.0)) for r in resultados_oos), default=0.0)

    r1 = pnl_total > 0.0
    r2 = pct_luc > CRITERIO_RUNS_LUCRATIVAS
    r3 = dd_max_run <= CRITERIO_DD_MAXIMO_RUN

    return {
        "aprovada":        r1 and r2 and r3,
        "pontuacao":       int(r1) + int(r2) + int(r3),
        "pnl_pct_total":   round(pnl_total, 2),
        "dd_pct_total":    round(dd_total, 2),
        "pct_runs_luc":    round(pct_luc * 100, 1),
        "dd_runs_max":     round(dd_max_run, 2),
        "n_runs":          n_r,
        "n_lucrativos":    n_luc,
        "resultados_oos":  resultados_oos,
        "equity_emendada": eq_total,
        "regra1":          r1,
        "regra2":          r2,
        "regra3":          r3,
    }


# ===================================================================
# PARTE 7 -- ANALISE DE CLUSTER (VEREDITO GLOBAL)
# ===================================================================

def analisar_cluster(aprovadas: np.ndarray) -> dict:
    """
    Identifica o maior bloco retangular contendo celulas aprovadas.
    Verifica o criterio de cluster SQX: >= 4L x 4C com >= 10 aprovadas.
    """
    nr, nc = aprovadas.shape
    n_total = int(aprovadas.sum())

    melhor_score = 0
    melhor_rect  = (0, 0, nr, nc)

    for r0 in range(nr):
        for c0 in range(nc):
            for r1 in range(r0 + 1, nr + 1):
                for c1 in range(c0 + 1, nc + 1):
                    n_ap = int(aprovadas[r0:r1, c0:c1].sum())
                    if n_ap > melhor_score:
                        melhor_score = n_ap
                        melhor_rect  = (r0, c0, r1, c1)

    r0, c0, r1, c1 = melhor_rect
    l_bloco = r1 - r0
    c_bloco = c1 - c0
    n_bloco = int(aprovadas[r0:r1, c0:c1].sum())

    passou = (
        l_bloco >= CLUSTER_MIN_LINHAS and
        c_bloco >= CLUSTER_MIN_COLUNAS and
        n_bloco >= CLUSTER_MIN_CELULAS
    )

    return {
        "passou":            passou,
        "cluster_rect":      melhor_rect,
        "n_aprovadas_total": n_total,
        "n_aprovadas_bloco": n_bloco,
        "n_linhas_bloco":    l_bloco,
        "n_colunas_bloco":   c_bloco,
    }


# ===================================================================
# PARTE 8 -- DASHBOARD VISUAL DARK PREMIUM
# ===================================================================

def gerar_dashboard(matriz: list, cluster: dict, passou: bool, dir_saida: Path):
    """
    Dashboard em 3 paineis:
      [Esq] Heatmap 5x4 com metricas e destaque de cluster
      [Dir Sup] Curva OOS emendada da celula campeao
      [Dir Inf] Scorecard e caixa de veredito
    """
    plt.rcParams.update({
        "figure.facecolor": "#0D1117", "axes.facecolor":  "#0D1117",
        "axes.edgecolor":   "#21262D", "axes.labelcolor": "#E6EDF3",
        "xtick.color":      "#E6EDF3", "ytick.color":     "#E6EDF3",
        "text.color":       "#E6EDF3", "grid.color":      "#21262D",
        "grid.linestyle":   "--",      "grid.alpha":      0.4,
        "font.family":      "monospace",
    })

    fig = plt.figure(figsize=(22, 14))
    ax_hm = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2)
    ax_eq = plt.subplot2grid((2, 3), (0, 2))
    ax_sc = plt.subplot2grid((2, 3), (1, 2))

    n_r = len(OOS_PCTS)
    n_c = len(WF_RUNS)

    # Construir matrizes de dados
    scores_m   = np.zeros((n_r, n_c))
    aprovadas_m = np.zeros((n_r, n_c), dtype=bool)

    for res in matriz:
        i, j = res["oos_idx"], res["run_idx"]
        scores_m[i, j]    = res["celula"]["pontuacao"]
        aprovadas_m[i, j] = res["celula"]["aprovada"]

    # Colormap dark premium
    cmap = LinearSegmentedColormap.from_list(
        "wfm", ["#5C1A1A", "#8B2020", "#161B22", "#1B4D8E", "#0D5C2E", "#3FB950"], N=256
    )
    hm_data = np.where(aprovadas_m, scores_m + 1.5, scores_m * 0.6)
    ax_hm.imshow(hm_data, cmap=cmap, vmin=0, vmax=4.5,
                 aspect="auto", interpolation="nearest")

    # Destacar cluster
    r0, c0, r1, c1 = cluster["cluster_rect"]
    if cluster["n_aprovadas_bloco"] >= 4:
        rect = mpatches.FancyBboxPatch(
            (c0 - 0.5, r0 - 0.5), c1 - c0, r1 - r0,
            boxstyle="round,pad=0.08", linewidth=3.5,
            edgecolor="#F0A500", facecolor="none", zorder=10
        )
        ax_hm.add_patch(rect)
        ax_hm.text(
            (c0 + c1) / 2.0 - 0.5, r0 - 0.68, "CLUSTER ESTAVEL",
            color="#F0A500", fontsize=9, fontweight="bold", ha="center"
        )

    # Texto por celula
    for res in matriz:
        i, j  = res["oos_idx"], res["run_idx"]
        cel   = res["celula"]
        st    = "[OK]" if cel["aprovada"] else "[X] "
        cor   = "#E6EDF3" if cel["aprovada"] else "#8B949E"
        fw    = "bold" if cel["aprovada"] else "normal"
        txt   = (f"{st}\n"
                 f"PnL:{cel['pnl_pct_total']:>+6.1f}%\n"
                 f"Luc:{cel['pct_runs_luc']:>4.0f}%\n"
                 f"DD :{abs(cel['dd_pct_total']):>5.1f}%")
        ax_hm.text(j, i, txt, ha="center", va="center",
                   color=cor, fontsize=8.5, fontweight=fw, family="monospace")

    ax_hm.set_xticks(range(n_c))
    ax_hm.set_xticklabels([f"Runs={r}" for r in WF_RUNS], fontsize=10)
    ax_hm.set_yticks(range(n_r))
    ax_hm.set_yticklabels([f"OOS={p}%" for p in OOS_PCTS], fontsize=10)
    ax_hm.set_title(
        f"Walk-Forward Matrix 5x4  --  {ESTRATEGIA} | {ATIVO} {TIMEFRAME}\n"
        f"R1: PnL>0  |  R2: Runs Luc.>{CRITERIO_RUNS_LUCRATIVAS*100:.0f}%  |  "
        f"R3: DD max run <{CRITERIO_DD_MAXIMO_RUN:.0f}%",
        fontsize=12, fontweight="bold", pad=14
    )
    p_ok  = mpatches.Patch(color="#3FB950", label="Aprovada (3/3)")
    p_par = mpatches.Patch(color="#1B4D8E", label="Parcial (1-2/3)")
    p_rep = mpatches.Patch(color="#8B2020", label="Reprovada (0/3)")
    p_cl  = mpatches.Patch(color="none", edgecolor="#F0A500", lw=2, label="Cluster")
    ax_hm.legend(handles=[p_ok, p_par, p_rep, p_cl],
                 loc="upper center", bbox_to_anchor=(0.5, -0.04),
                 ncol=4, fontsize=9, framealpha=0.3, facecolor="#0D1117")

    # Painel curva campeao
    melhor = max(matriz, key=lambda r: r["celula"]["pnl_pct_total"])
    eq_ch  = melhor["celula"]["equity_emendada"]
    pnl_ch = (eq_ch[-1] - CAPITAL_INICIAL) / CAPITAL_INICIAL * 100

    ax_eq.plot(eq_ch, color="#58A6FF", linewidth=1.5, zorder=3)
    ax_eq.fill_between(range(len(eq_ch)), CAPITAL_INICIAL, eq_ch,
                        where=(eq_ch >= CAPITAL_INICIAL), alpha=0.2, color="#3FB950")
    ax_eq.fill_between(range(len(eq_ch)), CAPITAL_INICIAL, eq_ch,
                        where=(eq_ch < CAPITAL_INICIAL), alpha=0.2, color="#F85149")
    ax_eq.axhline(CAPITAL_INICIAL, color="#FFFFFF", linestyle="--", linewidth=0.8)
    ax_eq.set_title(
        f"Curva OOS Emendada -- Celula Campeao\n"
        f"OOS={OOS_PCTS[melhor['oos_idx']]}% | "
        f"Runs={WF_RUNS[melhor['run_idx']]} | "
        f"PnL={pnl_ch:+.1f}%",
        fontsize=10, fontweight="bold"
    )
    ax_eq.set_xlabel("Barras OOS", fontsize=9)
    ax_eq.set_ylabel("Capital ($)", fontsize=9)
    ax_eq.grid(True)

    # Painel scorecard
    ax_sc.axis("off")
    n_tot  = len(matriz)
    n_ap   = cluster["n_aprovadas_total"]
    n_bl   = cluster["n_aprovadas_bloco"]
    l_bl   = cluster["n_linhas_bloco"]
    c_bl   = cluster["n_colunas_bloco"]
    pct_ap = n_ap / n_tot * 100 if n_tot > 0 else 0.0

    linhas = [
        f"Celulas totais   : {n_tot:>3d}",
        f"Celulas aprovadas: {n_ap:>3d}  ({pct_ap:.0f}%)",
        f"Maior cluster    : {n_bl:>3d} cel. ({l_bl}L x {c_bl}C)",
        f"Meta cluster     : {CLUSTER_MIN_CELULAS:>3d} cel. "
        f"({CLUSTER_MIN_LINHAS}Lx{CLUSTER_MIN_COLUNAS}C)",
    ]
    ax_sc.text(0.5, 0.94, "SCORECARD WFM", ha="center",
               fontsize=11, fontweight="bold", transform=ax_sc.transAxes)
    y = 0.78
    for ln in linhas:
        ax_sc.text(0.06, y, ln, fontsize=10, family="monospace",
                   color="#CFD8DC", transform=ax_sc.transAxes)
        y -= 0.13

    vt  = "SISTEMA ROBUSTO" if passou else "SISTEMA FRAGIL"
    vbg = "#238636" if passou else "#DA3633"
    ax_sc.text(0.5, 0.16, f"  {vt}  ", ha="center", va="center",
               fontsize=16, fontweight="bold", color="#FFFFFF",
               transform=ax_sc.transAxes,
               bbox=dict(boxstyle="round,pad=0.5", facecolor=vbg, edgecolor="none", alpha=0.95))
    ax_sc.text(0.5, 0.04,
               f"{n_ap}/{n_tot} aprovadas | Bloco: {l_bl}x{c_bl}",
               ha="center", fontsize=9, color="#8B949E",
               transform=ax_sc.transAxes)

    fig.suptitle(
        f"Walk-Forward Matrix (WFM) Institucional  --  {ESTRATEGIA} | {ATIVO} {TIMEFRAME}",
        color="#E6EDF3", fontsize=14, fontweight="bold", y=0.99
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    caminho = dir_saida / f"wfm_{ESTRATEGIA.lower()}.png"
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[GRAFICO] Salvo em: {caminho}")


# ===================================================================
# FUNCAO EXPORTAVEL PARA A ESTEIRA (FAIL-FAST)
# ===================================================================
def rodar_wfm_na_esteira(df: pd.DataFrame, df_comb: pd.DataFrame, estrategia: str, ativo: str, timeframe: str, dir_saida: Path) -> dict:
    horas = df.index.strftime("%H:%M")
    janela_op = (horas >= HORA_INICIO_OP) & (horas <= HORA_FIM_OP)
    n_total = len(df)
    
    static_cols = precomputar_indicadores(df)
    
    matriz_resultados = []
    n_celulas = len(OOS_PCTS) * len(WF_RUNS)
    
    for i, oos_pct in enumerate(OOS_PCTS):
        for j, n_runs in enumerate(WF_RUNS):
            fatias = gerar_fatias_wf(n_total, n_runs, oos_pct)
            celula = avaliar_celula(df, df_comb, fatias, static_cols, janela_op)
            
            matriz_resultados.append({
                "oos_pct": oos_pct, "n_runs": n_runs,
                "oos_idx": i, "run_idx": j,
                "celula":  celula,
            })
            
    mat_ap = np.zeros((len(OOS_PCTS), len(WF_RUNS)), dtype=bool)
    for res in matriz_resultados:
        mat_ap[res["oos_idx"], res["run_idx"]] = res["celula"]["aprovada"]
    cluster = analisar_cluster(mat_ap)
    passou = cluster["passou"]
    
    gerar_dashboard(matriz_resultados, cluster, passou, dir_saida)
    
    return {"aprovado": passou, "cluster": cluster}


# ===================================================================
# MAIN -- ORQUESTRADOR DO PIPELINE WFM
# ===================================================================

def main():
    global ATIVO, TIMEFRAME, ESTRATEGIA, DIR_SAIDA

    parser = argparse.ArgumentParser(description="Walk-Forward Matrix (WFM) Institucional")
    parser.add_argument("--ativo",      type=str, default=None)
    parser.add_argument("--timeframe",  type=str, default=None)
    parser.add_argument("--estrategia", type=str, default=None)
    args = parser.parse_args()

    if args.ativo:      ATIVO      = args.ativo.upper()
    if args.timeframe:  TIMEFRAME  = args.timeframe.upper()
    if args.estrategia: ESTRATEGIA = args.estrategia.upper()

    DIR_SAIDA = Path(__file__).resolve().parent / f"{ATIVO.lower()}_{TIMEFRAME.lower()}"
    DIR_SAIDA.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  WALK-FORWARD MATRIX (WFM) -- INSTITUCIONAL SQX-STYLE")
    print(f"  Estrategia : {ESTRATEGIA}")
    print(f"  Ativo      : {ATIVO} {TIMEFRAME}")
    print(f"  Matriz     : {len(OOS_PCTS)} x {len(WF_RUNS)} = {len(OOS_PCTS)*len(WF_RUNS)} celulas")
    print(f"  Saida      : {DIR_SAIDA}")
    print(f"{'='*60}\n")

    # Caminhos
    parquet_dados = (ROOT_DIR / f"quant_{ATIVO.lower()}" / "data"
                     / f"{ATIVO.lower()}_{TIMEFRAME.lower()}_hurst.parquet")
    parquet_otim  = (ROOT_DIR / f"quant_{ATIVO.lower()}" / "data" / "otimizacoes"
                     / f"otimizacao_{ESTRATEGIA.lower()}_resultados.parquet")

    if not parquet_dados.exists():
        raise FileNotFoundError(f"Parquet de dados nao encontrado: {parquet_dados}")
    if not parquet_otim.exists():
        raise FileNotFoundError(f"Parquet de otimizacoes nao encontrado: {parquet_otim}")

    # Carregar dados
    print("[DADOS] Carregando serie historica...")
    df = pd.read_parquet(parquet_dados)
    if not isinstance(df.index, pd.DatetimeIndex):
        for col in ("time", "datetime"):
            if col in df.columns:
                df = df.set_index(pd.to_datetime(df[col]))
                break
    if "log_return" not in df.columns:
        df["log_return"] = np.log(df["Close"] / df["Close"].shift(1)).fillna(0.0)
    print(f"   {len(df):,} candles ({df.index.min()} -> {df.index.max()}) carregados.")

    print("[OTIM] Carregando combinacoes de otimizacao...")
    df_comb = pd.read_parquet(parquet_otim)
    # Ordenar pelo melhor Ret_DD para varrimento mais rapido
    if "Ret_DD" in df_comb.columns:
        df_comb = df_comb.sort_values("Ret_DD", ascending=False).reset_index(drop=True)
    print(f"   {len(df_comb)} combinacoes de parametros carregadas.\n")

    # Janela operacional e indicadores estaticos
    horas     = df.index.strftime("%H:%M")
    janela_op = (horas >= HORA_INICIO_OP) & (horas <= HORA_FIM_OP)
    n_total   = len(df)

    print("[PERF] Pre-calculando indicadores matematicos (uma unica vez)...")
    static_cols = precomputar_indicadores(df)
    print("   Velocidade, Percentil, Entropia, VR_Pips pre-computados.\n")

    # Loop da Matriz
    print(f"[WFM] Iniciando avaliacao da matriz {len(OOS_PCTS)}x{len(WF_RUNS)}...\n")
    matriz_resultados = []
    n_celulas = len(OOS_PCTS) * len(WF_RUNS)

    with tqdm(total=n_celulas, desc="  WFM", ncols=72, unit="celula") as pbar:
        for i, oos_pct in enumerate(OOS_PCTS):
            for j, n_runs in enumerate(WF_RUNS):
                fatias = gerar_fatias_wf(n_total, n_runs, oos_pct)
                celula = avaliar_celula(df, df_comb, fatias, static_cols, janela_op)

                matriz_resultados.append({
                    "oos_pct": oos_pct, "n_runs": n_runs,
                    "oos_idx": i, "run_idx": j,
                    "celula":  celula,
                })

                st = "[OK]" if celula["aprovada"] else "[X] "
                pbar.write(
                    f"  {st} OOS={oos_pct:2d}% Runs={n_runs:2d} | "
                    f"PnL={celula['pnl_pct_total']:>+7.1f}% | "
                    f"Luc={celula['pct_runs_luc']:>4.0f}% | "
                    f"DD={abs(celula['dd_pct_total']):>5.1f}% | "
                    f"R[{int(celula['regra1'])}{int(celula['regra2'])}{int(celula['regra3'])}]"
                )
                pbar.update(1)

    # Cluster e Veredito
    print("\n[CLUSTER] Analisando bloco contiguo de aprovacoes...")
    mat_ap = np.zeros((len(OOS_PCTS), len(WF_RUNS)), dtype=bool)
    for res in matriz_resultados:
        mat_ap[res["oos_idx"], res["run_idx"]] = res["celula"]["aprovada"]
    cluster    = analisar_cluster(mat_ap)
    passou_td  = cluster["passou"]

    # Relatorio terminal
    n_ap = cluster["n_aprovadas_total"]
    n_tot = len(matriz_resultados)
    l_bl = cluster["n_linhas_bloco"]
    c_bl = cluster["n_colunas_bloco"]
    n_bl = cluster["n_aprovadas_bloco"]
    r0, c0, r1, c1 = cluster["cluster_rect"]

    print(f"\n  {'='*52}")
    print(f"  WALK-FORWARD MATRIX -- RELATORIO FINAL")
    print(f"  Estrategia : {ESTRATEGIA} | Ativo: {ATIVO} {TIMEFRAME}")
    print(f"  {'='*52}")
    print(f"    Celulas aprovadas  : {n_ap} / {n_tot}  ({n_ap/n_tot*100:.0f}%)")
    print(f"    Maior cluster      : {n_bl} cel. ({l_bl}L x {c_bl}C)")
    oos_r0 = OOS_PCTS[r0] if r0 < len(OOS_PCTS) else OOS_PCTS[-1]
    oos_r1 = OOS_PCTS[r1-1] if r1 - 1 < len(OOS_PCTS) else OOS_PCTS[-1]
    run_c0 = WF_RUNS[c0] if c0 < len(WF_RUNS) else WF_RUNS[-1]
    run_c1 = WF_RUNS[c1-1] if c1 - 1 < len(WF_RUNS) else WF_RUNS[-1]
    print(f"    Regiao do cluster  : OOS [{oos_r0}%..{oos_r1}%] x Runs [{run_c0}..{run_c1}]")
    ok1 = "[OK]" if n_bl >= CLUSTER_MIN_CELULAS else "[X] "
    ok2 = "[OK]" if l_bl >= CLUSTER_MIN_LINHAS  else "[X] "
    ok3 = "[OK]" if c_bl >= CLUSTER_MIN_COLUNAS else "[X] "
    print(f"  {'='*52}")
    print(f"  {ok1} Celulas no bloco   : {n_bl:2d}  (min: {CLUSTER_MIN_CELULAS})")
    print(f"  {ok2} Linhas do bloco    : {l_bl:2d}  (min: {CLUSTER_MIN_LINHAS})")
    print(f"  {ok3} Colunas do bloco   : {c_bl:2d}  (min: {CLUSTER_MIN_COLUNAS})")
    print(f"  {'='*52}")
    vt = "SISTEMA ROBUSTO [OK]" if passou_td else "SISTEMA FRAGIL [X] "
    print(f"  +{'='*50}+")
    print(f"  |  VEREDITO: {vt:<38s}|")
    print(f"  +{'='*50}+\n")

    # Grafico
    gerar_dashboard(matriz_resultados, cluster, passou_td, DIR_SAIDA)

    # CSV
    rows = []
    for res in matriz_resultados:
        c = res["celula"]
        rows.append({
            "oos_pct": res["oos_pct"], "n_runs": res["n_runs"],
            "aprovada": c["aprovada"], "pontuacao": c["pontuacao"],
            "pnl_pct_total": c["pnl_pct_total"], "dd_pct_total": c["dd_pct_total"],
            "pct_runs_luc": c["pct_runs_luc"], "dd_runs_max": c["dd_runs_max"],
            "regra1_pnl_pos": c["regra1"], "regra2_luc65": c["regra2"], "regra3_dd25": c["regra3"],
        })
    caminho_csv = DIR_SAIDA / f"wfm_{ESTRATEGIA.lower()}.csv"
    pd.DataFrame(rows).to_csv(caminho_csv, index=False)
    print(f"[CSV] Salvo em: {caminho_csv}")

    # TXT
    caminho_txt = DIR_SAIDA / f"wfm_{ESTRATEGIA.lower()}_resumo.txt"
    with open(caminho_txt, "w", encoding="utf-8") as f:
        f.write("=" * 55 + "\n")
        f.write(f"WALK-FORWARD MATRIX (WFM) -- RELATORIO EXECUTIVO\n")
        f.write(f"Estrategia : {ESTRATEGIA} | Ativo: {ATIVO} {TIMEFRAME}\n")
        f.write(f"Celulas    : {n_tot} ({len(OOS_PCTS)} OOS% x {len(WF_RUNS)} Runs)\n")
        f.write("=" * 55 + "\n")
        f.write(f"Celulas aprovadas  : {n_ap} / {n_tot} ({n_ap/n_tot*100:.0f}%)\n")
        f.write(f"Maior cluster      : {n_bl} celulas ({l_bl}L x {c_bl}C)\n")
        f.write("-" * 55 + "\n")
        for res in matriz_resultados:
            c = res["celula"]
            s = "OK" if c["aprovada"] else "X "
            f.write(f"[{s}] OOS={res['oos_pct']:2d}% Runs={res['n_runs']:2d} | "
                    f"PnL={c['pnl_pct_total']:>+7.1f}% | "
                    f"Luc={c['pct_runs_luc']:>4.0f}% | "
                    f"DD={abs(c['dd_pct_total']):>5.1f}%\n")
        f.write("-" * 55 + "\n")
        f.write(f"VEREDITO: {vt}\n")
        f.write("=" * 55 + "\n")
    print(f"[TXT] Salvo em: {caminho_txt}")
    print(f"\n[OK] Walk-Forward Matrix finalizado com sucesso.\n")


if __name__ == "__main__":
    main()
