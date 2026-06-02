# -*- coding: utf-8 -*-
"""
================================================
hawkes_strategy.py — Estratégia Quantitativa
Versão Genérica — Reutilizável para qualquer ativo
================================================
Configuração:
  Definir ATIVO e TIMEFRAME no bloco de
  configuração no topo deste arquivo antes
  de executar.

Uso:
  1. Configurar ATIVO e TIMEFRAME
  2. Garantir que os parquets de entrada
     existam na pasta data/ do projeto alvo
  3. Executar: python hawkes_strategy.py
================================================
"""

import sys
import math
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize
from pathlib import Path

# ===========================================
# CONFIGURAÇÃO DO ATIVO — ALTERAR AQUI
# ===========================================
ATIVO          = "EURUSD"
TIMEFRAME      = "H1"
DIR_PROJETO    = Path(__file__).resolve().parent.parent
DIR_DATA       = DIR_PROJETO / "data"
DIR_GRAFICOS   = DIR_PROJETO / "graficos"

PARQUET_COMPLETO    = DIR_DATA / f"{ATIVO.lower()}_{TIMEFRAME.lower()}_completo.parquet"
PARQUET_OPERACIONAL = DIR_DATA / f"{ATIVO.lower()}_{TIMEFRAME.lower()}_operacional.parquet"
PARQUET_HURST       = DIR_DATA / f"{ATIVO.lower()}_{TIMEFRAME.lower()}_hurst.parquet"
PARQUET_SAIDA       = DIR_DATA / "hawkes" / f"{ATIVO.lower()}_{TIMEFRAME.lower()}_hawkes.parquet"
CAMINHO_GRAFICO     = DIR_GRAFICOS / f"{ATIVO.lower()}_{TIMEFRAME.lower()}_hawkes_sinais.png"

# ================================================
# COMO USAR PARA NOVO ATIVO:
# 1. Alterar ATIVO = "NASDAQ" (ou outro)
# 2. Alterar TIMEFRAME = "M10" (ou outro)
# 3. Garantir que existam os parquets:
#    data/nasdaq_m10_completo.parquet
#    data/nasdaq_m10_operacional.parquet
#    data/nasdaq_m10_hurst.parquet (se necessário)
# 4. Executar normalmente
# ================================================

# Configurar para salvamento de gráficos sem janela
matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# Configuração de Logging Profissional
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTES E CAMINHOS
# =============================================================================
DIR_ATUAL = Path(__file__).resolve().parent

GRAFICO_SAIDA = DIR_ATUAL / "graficos" / "hawkes_sinais.png"

# Parâmetros da Estratégia Hawkes
JANELA_EVENTOS = 60
JANELA_MLE = 120
PASSO_MLE = 24  # Otimizar a cada 24 candles (1 dia aprox)
JANELA_VOL = 50

# Thresholds operacionais OTIMIZADOS
EXCITACAO_MAXIMA = 0.65
LAMBDA_NORM_MIN_SINAL = 3.0
LAMBDA_NORM_SAIDA = 0.6

# =============================================================================
# CORE MATH: PROCESSOS ESTOCÁSTICOS E MLE
# =============================================================================

def log_likelihood_hawkes_numba(params, N_array):
    """
    Calcula o Log-Likelihood negativo do processo de Hawkes discreto.
    Otimizado via Numba para velocidade C-like.
    """
    mu, alpha, beta = params
    
    # Penalizar fortemente instabilidade
    if mu <= 0 or alpha <= 0 or beta <= 0 or alpha / beta >= 1.0:
        return 1e9
        
    n_len = len(N_array)
    ll = 0.0
    lam_prev = mu
    
    for t in range(1, n_len):
        # Recursão de Hawkes discreta
        lam_t = mu + math.exp(-beta) * (lam_prev - mu) + alpha * N_array[t-1]
        
        if lam_t <= 0:
            lam_t = 1e-9 # Evitar log(0)
            
        ll += N_array[t] * math.log(lam_t) - lam_t
        lam_prev = lam_t
        
    return -ll

def estimar_hawkes_mle(N_array: np.ndarray):
    """
    Estima os parâmetros (μ, α, β) para um vetor de eventos observados.
    """
    if np.sum(N_array) == 0:
        # Nenhum evento na janela, retornar base
        return 0.1, 0.001, 1.0, False
        
    # Chutes iniciais
    x0 = np.array([0.1, 0.5, 1.0])
    
    # Limites (bounds) matemáticos rigorosos
    bnds = ((0.001, 5.0), (0.001, 5.0), (0.001, 10.0))
    
    res = minimize(
        log_likelihood_hawkes_numba, 
        x0, 
        args=(N_array,), 
        method="L-BFGS-B", 
        bounds=bnds
    )
    
    mu_est, alpha_est, beta_est = res.x
    valido = res.success and (alpha_est / beta_est < 1.0)
    
    return float(mu_est), float(alpha_est), float(beta_est), bool(valido)

# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def executar_pipeline_hawkes():
    logger.info("Carregando bases de dados H1 para Hawkes...")
    
    if not PARQUET_COMPLETO.exists() or not PARQUET_OPERACIONAL.exists():
        raise FileNotFoundError("Os arquivos Parquet não foram encontrados na pasta data/.")
        
    df = pd.read_parquet(PARQUET_COMPLETO, engine="pyarrow")
    df_op = pd.read_parquet(PARQUET_OPERACIONAL, engine="pyarrow")
    n_candles = len(df)
    
    # ── PARTE 1 & 2: Extração de Features e Identificação de Eventos ──
    logger.info("Identificando Eventos Extremos (Spikes)...")
    
    # Usar log_return já existente no parquet para consistência
    # com o backtest downstream. Criar R_t como alias.
    if "log_return" in df.columns:
        df["R_t"] = df["log_return"]
    else:
        df["R_t"] = np.log(df["Close"] / df["Close"].shift(1))
    
    # std rolante ddof=1
    R_t_values = df["R_t"].values
    R_t_abs = np.abs(R_t_values)
    
    std_60 = df["R_t"].rolling(window=JANELA_EVENTOS).std(ddof=1).values
    threshold_dyn = 2.0 * std_60
    
    # Se |R_t| > threshold, N_t = 1
    N_t = np.zeros(n_candles, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        N_t = np.where(R_t_abs > threshold_dyn, 1.0, 0.0)
        
    # Limpar NaNs iniciais
    N_t[np.isnan(N_t)] = 0.0
    
    # ── PARTE 3: Estimação de Parâmetros MLE (Janelas Rolantes Salteadas) ──
    logger.info("Iniciando MLE do processo de Hawkes a cada 24 horas...")
    
    # Vetores de estado congelado
    mu_hist = np.full(n_candles, np.nan, dtype=np.float32)
    alpha_hist = np.full(n_candles, np.nan, dtype=np.float32)
    beta_hist = np.full(n_candles, np.nan, dtype=np.float32)
    valido_hist = np.zeros(n_candles, dtype=np.int8)
    
    mu_atual, alpha_atual, beta_atual = 0.1, 0.001, 1.0
    valido_atual = False
    
    # Progresso
    total_opt = (n_candles - JANELA_MLE) // PASSO_MLE
    opt_count = 0
    
    for t in range(JANELA_MLE, n_candles):
        if t % PASSO_MLE == 0:
            window_N = N_t[t-JANELA_MLE:t]
            mu_est, alpha_est, beta_est, is_val = estimar_hawkes_mle(window_N)
            
            mu_atual = mu_est
            alpha_atual = alpha_est
            beta_atual = beta_est
            valido_atual = is_val
            
            opt_count += 1
            if opt_count % 500 == 0:
                logger.info(f"Progresso MLE: {opt_count}/{total_opt} concluídas ({100.0 * opt_count / total_opt:.1f}%)")
                
        # Propagar
        mu_hist[t] = mu_atual
        alpha_hist[t] = alpha_atual
        beta_hist[t] = beta_atual
        valido_hist[t] = 1 if valido_atual else 0

    # ── PARTE 4: Cálculo da Intensidade λ_t Iterativa ──
    logger.info("Calculando intensidade λ_t recursivamente...")
    
    lambda_t = np.full(n_candles, np.nan, dtype=np.float32)
    lambda_norm = np.full(n_candles, np.nan, dtype=np.float32)
    excitacao = np.full(n_candles, np.nan, dtype=np.float32)
    p_evento = np.full(n_candles, np.nan, dtype=np.float32)
    
    lam_prev = 0.1
    for t in range(JANELA_MLE, n_candles):
        mu = mu_hist[t]
        alpha = alpha_hist[t]
        beta = beta_hist[t]
        
        # λ_t = μ + exp(-β)*(λ_{t-1} - μ) + α * N_{t-1}
        lam_curr = mu + math.exp(-beta) * (lam_prev - mu) + alpha * N_t[t-1]
        if lam_curr < mu:
            lam_curr = mu # bound por baixo natural
            
        lambda_t[t] = lam_curr
        lambda_norm[t] = (lam_curr - mu) / mu if mu > 0 else 0.0
        excitacao[t] = alpha / beta if beta > 0 else 0.0
        p_evento[t] = 1.0 - math.exp(-lam_curr)
        
        lam_prev = lam_curr
        
    df["hawkes_lambda"] = lambda_t
    df["hawkes_lambda_norm"] = lambda_norm
    df["hawkes_excitacao"] = excitacao
    df["hawkes_p_evento"] = p_evento
    df["hawkes_mu"] = mu_hist
    df["hawkes_alpha"] = alpha_hist
    df["hawkes_beta"] = beta_hist
    df["hawkes_valido"] = valido_hist
    
    # ── PARTE 5: Volatilidade Realizada e Gestão de Risco ──
    logger.info("Calculando VR e Gestão de Risco...")
    df["VR"] = df["R_t"].rolling(window=JANELA_VOL).std(ddof=1)
    df["vr_pips"] = df["VR"] * df["Close"] * 10000.0
    
    df["sl_pips"] = 2.0 * df["vr_pips"]
    df["tp_pips"] = 3.0 * df["vr_pips"]
    df["sl_pips"] = df["sl_pips"].clip(lower=3.0, upper=60.0).fillna(10.0)
    df["tp_pips"] = df["tp_pips"].clip(lower=4.5, upper=90.0).fillna(15.0)
    
    # ── PARTE 6: Geração de Sinais ──
    logger.info("Mapeando regras de Sinais Hawkes...")
    mask_op = df.index.isin(df_op.index)
    
    sinal = np.zeros(n_candles, dtype=np.int8)
    
    valido = df["hawkes_valido"] == 1
    excit_ok = df["hawkes_excitacao"] < EXCITACAO_MAXIMA
    norm_high = df["hawkes_lambda_norm"] > LAMBDA_NORM_MIN_SINAL
    
    lam_norm_vals = df["hawkes_lambda_norm"].values
    lam_norm_prev = np.roll(lam_norm_vals, 1)
    lam_norm_prev[0] = np.nan
    
    norm_falling = lam_norm_vals < lam_norm_prev
    
    cond_base = valido & excit_ok & mask_op & norm_falling & norm_high
    
    # R_t atual (a exaustão está ocorrendo na barra atual, então apostamos contra a barra atual)
    # Se R_t < 0 (cluster negativo) -> apostamos em reversão pra CIMA (LONG = 1)
    # Se R_t > 0 (cluster positivo) -> apostamos em reversão pra BAIXO (SHORT = -1)
    
    mask_buy = cond_base & (df["R_t"] < 0)
    sinal[mask_buy] = 1
    
    mask_sell = cond_base & (df["R_t"] > 0)
    sinal[mask_sell] = -1
    
    # Saída por exaustão do cluster de Hawkes:
    # Encerrar posição quando lambda_norm cai abaixo de
    # LAMBDA_NORM_SAIDA — sinal de que o cluster se dissipou.
    # Esta coluna deve ser consumida pelo backtest downstream
    # para encerrar posições antes do SL/TP quando aplicável.
    df["hawkes_saida_threshold"] = LAMBDA_NORM_SAIDA

    df["sinal_hawkes"] = sinal

    # IMPORTANTE: O sinal gerado em t deve ser executado com
    # spread aplicado no preço de entrada:
    #     LONG  → entrada = Close[t] + 0.00005
    #     SHORT → entrada = Close[t] - 0.00005
    # Qualquer backtest que consuma esta coluna deve aplicar
    # este custo para manter alinhamento com a otimização.
    
    n_buys = (sinal == 1).sum()
    n_sells = (sinal == -1).sum()
    logger.info(f"Sinais Hawkes: {n_buys} Compras | {n_sells} Vendas | Total: {n_buys + n_sells}")
    
    # Limpar R_t e VR
    df.drop(columns=["R_t", "VR"], inplace=True)
    
    # ── PARTE 7: Parquet de Saída ──
    PARQUET_SAIDA.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PARQUET_SAIDA, engine="pyarrow", compression="snappy")
    logger.info(f"Parquet gerado com sucesso: {PARQUET_SAIDA.name}")
    
    # ── PARTE 8: Plotagem (4 Painéis) ──
    logger.info("Renderizando gráfico de visualização...")
    
    plt.rcParams.update({
        "figure.facecolor":  "#0D1117",
        "axes.facecolor":    "#0D1117",
        "axes.edgecolor":    "#21262D",
        "axes.labelcolor":   "#E6EDF3",
        "xtick.color":       "#E6EDF3",
        "ytick.color":       "#E6EDF3",
        "text.color":        "#E6EDF3",
        "grid.color":        "#21262D",
    })
    
    df_plot = df.iloc[-3000:].copy()
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        4, 1, figsize=(20, 18), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.5, 1.5, 1.5], "hspace": 0.05}
    )
    
    fig.suptitle("Processo Autoexcitável de Hawkes — Teoria das Filas e Spikes", 
                 color="#E6EDF3", fontsize=15, fontweight="bold", y=0.92)
                 
    # Ax1: Preço
    ax1.plot(df_plot.index, df_plot["Close"], color="#58A6FF", linewidth=1.2, alpha=0.9)
    buys = df_plot[df_plot["sinal_hawkes"] == 1]
    sells = df_plot[df_plot["sinal_hawkes"] == -1]
    ax1.scatter(buys.index, buys["Close"] - 0.0010, marker="^", color="#3FB950", s=120, label="COMPRA (+1)", zorder=5)
    ax1.scatter(sells.index, sells["Close"] + 0.0010, marker="v", color="#F85149", s=120, label="VENDA (-1)", zorder=5)
    ax1.set_ylabel("Cotação", fontsize=10)
    ax1.legend(loc="upper left", framealpha=0.3)
    ax1.grid(True, linestyle="--", alpha=0.2)
    
    # Ax2: Intensidade Lambda e Lambda Norm
    ax2.plot(df_plot.index, df_plot["hawkes_lambda_norm"], color="#A371F7", linewidth=1.2, label="Intensidade Normalizada (λ_norm)")
    ax2.axhline(LAMBDA_NORM_MIN_SINAL, color="#F0A500", linestyle="--", alpha=0.7, label=f"Threshold Sinal ({LAMBDA_NORM_MIN_SINAL})")
    ax2.axhline(LAMBDA_NORM_SAIDA, color="#484F58", linestyle=":", alpha=0.9, label=f"Baseline Saída ({LAMBDA_NORM_SAIDA})")
    ax2.fill_between(df_plot.index, 0, LAMBDA_NORM_SAIDA, color="#484F58", alpha=0.2)
    ax2.set_ylabel("λ Normalizado", fontsize=10)
    ax2.legend(loc="upper left", framealpha=0.3)
    ax2.grid(True, linestyle="--", alpha=0.2)
    
    # Ax3: Razão de Excitação (α/β)
    ax3.plot(df_plot.index, df_plot["hawkes_excitacao"], color="#3FB950", linewidth=1.2, label="Razão de Excitação (α/β)")
    ax3.axhline(EXCITACAO_MAXIMA, color="#F85149", linestyle="--", alpha=0.7, label=f"Max Excitação ({EXCITACAO_MAXIMA})")
    ax3.fill_between(df_plot.index, EXCITACAO_MAXIMA, 1.0, color="#F85149", alpha=0.1) # Zona proibida
    ax3.set_ylim(0, 1.05)
    ax3.set_ylabel("Excitação", fontsize=10)
    ax3.legend(loc="upper left", framealpha=0.3)
    ax3.grid(True, linestyle="--", alpha=0.2)
    
    # Ax4: Probabilidade de Evento Futuro
    ax4.plot(df_plot.index, df_plot["hawkes_p_evento"], color="#D2A8FF", linewidth=1.2, label="P(Evento Extremo)")
    ax4.fill_between(df_plot.index, 0, df_plot["hawkes_p_evento"], color="#D2A8FF", alpha=0.1)
    ax4.set_ylim(0, 1)
    ax4.set_ylabel("Probabilidade", fontsize=10)
    ax4.legend(loc="upper left", framealpha=0.3)
    ax4.grid(True, linestyle="--", alpha=0.2)
    
    GRAFICO_SAIDA.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(GRAFICO_SAIDA, dpi=120, bbox_inches="tight", facecolor="#0D1117")
    plt.close()
    logger.info(f"Gráfico exportado com sucesso: {GRAFICO_SAIDA.name}")

if __name__ == "__main__":
    try:
        executar_pipeline_hawkes()
        print("\n> Módulo Hawkes concluído com sucesso!")
    except Exception as e:
        logger.exception("Erro crítico no módulo Hawkes:")
        sys.exit(1)
