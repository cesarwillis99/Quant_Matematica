# -*- coding: utf-8 -*-
"""
================================================
pca_strategy.py — Estratégia Quantitativa
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
  3. Executar: python pca_strategy.py
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
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view

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
PARQUET_SAIDA       = DIR_DATA / f"{ATIVO.lower()}_{TIMEFRAME.lower()}_pca.parquet"
CAMINHO_GRAFICO     = DIR_GRAFICOS / f"{ATIVO.lower()}_{TIMEFRAME.lower()}_pca_sinais.png"

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

GRAFICO_SAIDA = DIR_ATUAL / "graficos" / "pca_sinais.png"

# Parâmetros da Estratégia PCA
JANELA_PCA = 60
JANELA_VOL = 50
DOMINANCIA_MINIMA = 0.45
ZSCORE_THRESHOLD = 2.0
ZSCORE_NEUTRO_MIN = -0.5
ZSCORE_NEUTRO_MAX = 0.5

# =============================================================================
# CORE: CÁLCULO VETORIZADO DO PCA ROLANTE
# =============================================================================

def calcular_pca_rolante(features: np.ndarray, janela: int):
    """
    Realiza a Análise de Componentes Principais (PCA) em janelas rolantes 
    com alta velocidade.
    
    features: Array 2D (N_candles, N_features).
    """
    n_candles, n_features = features.shape
    pca_z_pc1 = np.full(n_candles, np.nan, dtype=np.float32)
    pca_dominancia = np.full(n_candles, np.nan, dtype=np.float32)
    pca_var_pc1 = np.full(n_candles, np.nan, dtype=np.float32)
    pca_var_pc2 = np.full(n_candles, np.nan, dtype=np.float32)
    
    # Criar view de janelas rolantes: shape = (N - janela + 1, janela, N_features)
    try:
        windows = sliding_window_view(features, window_shape=(janela, n_features)).squeeze()
    except Exception as e:
        logger.error(f"Erro ao criar sliding window: {e}. Atualize o numpy.")
        raise
        
    logger.info("Decomposição espectral em andamento...")
    
    pc1_scores_hist = np.full(n_candles, np.nan, dtype=np.float32)
    
    for i in range(windows.shape[0]):
        idx_atual = i + janela - 1
        X = windows[i] # Matriz X de dimensão (60, 5)
        
        # Passo 1: Padronizar X
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        stds[stds == 0] = 1.0 
        
        X_std = (X - means) / stds
        
        # Passo 2: Matriz de Covariância
        C = (1.0 / (janela - 1)) * (X_std.T @ X_std)
        
        # Passo 3: Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        
        # Ordenar decrescente
        idx_sort = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx_sort]
        eigenvectors = eigenvectors[:, idx_sort]
        
        # Passo 4: Variância explicada
        soma_evals = np.sum(eigenvalues)
        if soma_evals == 0:
            continue
            
        var_expl = eigenvalues / soma_evals
        dominancia = var_expl[0]
        
        # Passo 5: Projetar os retornos na PC1
        pc1_vector = eigenvectors[:, 0]
        pc1_scores = X_std @ pc1_vector # shape: (60,)
        
        # Gravar a projeção da barra atual (último elemento da janela)
        pc1_scores_hist[idx_atual] = pc1_scores[-1]
        
        pca_dominancia[idx_atual] = dominancia
        pca_var_pc1[idx_atual] = var_expl[0]
        pca_var_pc2[idx_atual] = var_expl[1] if len(var_expl) > 1 else 0.0

    # Passo 6: Z-Score da PC1
    for i in range(windows.shape[0]):
        idx_atual = i + janela - 1
        X = windows[i]
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        stds[stds == 0] = 1.0
        X_std = (X - means) / stds
        C = (1.0 / (janela - 1)) * (X_std.T @ X_std)
        evals, evecs = np.linalg.eigh(C)
        idx_sort = np.argsort(evals)[::-1]
        evecs = evecs[:, idx_sort]
        pc1_vector = evecs[:, 0]
        pc1_scores = X_std @ pc1_vector
        
        mean_pc1 = np.mean(pc1_scores)
        std_pc1 = np.std(pc1_scores)
        if std_pc1 > 0:
            z_pc1 = (pc1_scores[-1] - mean_pc1) / std_pc1
            pca_z_pc1[idx_atual] = z_pc1
        else:
            pca_z_pc1[idx_atual] = 0.0

    return pca_z_pc1, pca_dominancia, pca_var_pc1, pca_var_pc2

# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def executar_pipeline_pca():
    logger.info("Carregando bases de dados H1...")
    
    if not PARQUET_COMPLETO.exists() or not PARQUET_OPERACIONALERACIONAL.exists():
        raise FileNotFoundError("Os arquivos Parquet completos e operacionais não foram encontrados na pasta data/.")
        
    df = pd.read_parquet(PARQUET_COMPLETO, engine="pyarrow")
    df_op = pd.read_parquet(PARQUET_OPERACIONALERACIONAL, engine="pyarrow")
    
    # ── PARTE 1: Construção da Matriz de Features ──
    logger.info("Calculando Features para o PCA...")
    df["R_t"] = np.log(df["Close"] / df["Close"].shift(1))
    df["R_t_sq"] = df["R_t"] ** 2
    df["R_t_abs"] = df["R_t"].abs()
    df["R_t_lag1"] = df["R_t"].shift(1)
    df["R_t_lag2"] = df["R_t"].shift(2)
    
    # Substituir NAs por 0 antes do PCA
    df_features = df[["R_t", "R_t_sq", "R_t_abs", "R_t_lag1", "R_t_lag2"]].fillna(0.0)
    features_array = df_features.values
    
    # ── PARTE 2: PCA e Decomposição Espectral ──
    pca_z_pc1, pca_dominancia, pca_var_pc1, pca_var_pc2 = calcular_pca_rolante(features_array, JANELA_PCA)
    
    df["pca_z_pc1"] = pca_z_pc1
    df["pca_dominancia"] = pca_dominancia
    df["pca_var_pc1"] = pca_var_pc1
    df["pca_var_pc2"] = pca_var_pc2
    
    # ── PARTE 3: Volatilidade Realizada e Gestão de Risco ──
    logger.info("Calculando Volatilidade Realizada e Stops...")
    df["VR"] = df["R_t"].rolling(window=JANELA_VOL).std(ddof=1)
    df["vr_pips"] = df["VR"] * df["Close"] * 10000.0
    
    df["sl_pips"] = 2.0 * df["vr_pips"]
    df["tp_pips"] = 3.0 * df["vr_pips"]
    
    df["sl_pips"] = df["sl_pips"].clip(lower=3.0, upper=60.0).fillna(10.0)
    df["tp_pips"] = df["tp_pips"].clip(lower=4.5, upper=90.0).fillna(15.0)
    
    # ── PARTE 4: Geração de Sinais ──
    logger.info("Gerando Sinais na janela operacional...")
    mask_op = df.index.isin(df_op.index)
    
    sinal = np.zeros(len(df), dtype=np.int8)
    
    cond_comum = (df["pca_dominancia"] >= DOMINANCIA_MINIMA) & mask_op
    
    # Compra (LONG)
    mask_buy = cond_comum & (df["pca_z_pc1"] <= -ZSCORE_THRESHOLD)
    sinal[mask_buy] = 1
    
    # Venda (SHORT)
    mask_sell = cond_comum & (df["pca_z_pc1"] >= ZSCORE_THRESHOLD)
    sinal[mask_sell] = -1
    
    df["sinal_pca"] = sinal
    
    n_buys = (sinal == 1).sum()
    n_sells = (sinal == -1).sum()
    logger.info(f"Sinais Gerados: {n_buys} Compras | {n_sells} Vendas | Total: {n_buys + n_sells}")
    
    # Limpeza
    colunas_drop = ["R_t", "R_t_sq", "R_t_abs", "R_t_lag1", "R_t_lag2", "VR"]
    df.drop(columns=colunas_drop, inplace=True)
    
    # ── PARTE 5: Parquet de Saída ──
    df.to_parquet(PARQUET_SAIDA, engine="pyarrow", compression="snappy")
    logger.info(f"Parquet salvo com sucesso em: {PARQUET_SAIDA}")
    
    # ── PARTE 6: Plotagem ──
    logger.info("Gerando gráfico profissional de demonstração...")
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
    
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(18, 14), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.5, 1.5], "hspace": 0.05}
    )
    
    fig.suptitle("Análise de Componentes Principais (PCA) — Detecção de Extremos Estruturais", 
                 color="#E6EDF3", fontsize=14, fontweight="bold", y=0.92)
                 
    # Ax1: Preço e Sinais
    ax1.plot(df_plot.index, df_plot["Close"], color="#58A6FF", linewidth=1.2, alpha=0.9)
    buys = df_plot[df_plot["sinal_pca"] == 1]
    sells = df_plot[df_plot["sinal_pca"] == -1]
    ax1.scatter(buys.index, buys["Close"] - 0.0010, marker="^", color="#3FB950", s=100, label="COMPRA (+1)", zorder=5)
    ax1.scatter(sells.index, sells["Close"] + 0.0010, marker="v", color="#F85149", s=100, label="VENDA (-1)", zorder=5)
    ax1.set_ylabel("Cotação", fontsize=10)
    ax1.legend(loc="upper left", framealpha=0.3)
    ax1.grid(True, linestyle="--", alpha=0.2)
    
    # Ax2: Z-Score da PC1
    ax2.plot(df_plot.index, df_plot["pca_z_pc1"], color="#A371F7", linewidth=1.0)
    ax2.axhline(0, color="#E6EDF3", linestyle="-", alpha=0.3)
    ax2.axhline(ZSCORE_THRESHOLD, color="#F85149", linestyle="--", alpha=0.7, label="Threshold Venda (+2.0)")
    ax2.axhline(-ZSCORE_THRESHOLD, color="#3FB950", linestyle="--", alpha=0.7, label="Threshold Compra (-2.0)")
    ax2.fill_between(df_plot.index, ZSCORE_NEUTRO_MIN, ZSCORE_NEUTRO_MAX, color="#484F58", alpha=0.3, label="Zona Neutra")
    ax2.set_ylabel("Z-Score (PC1)", fontsize=10)
    ax2.legend(loc="upper left", framealpha=0.3)
    ax2.grid(True, linestyle="--", alpha=0.2)
    
    # Ax3: Dominância Espectral
    ax3.plot(df_plot.index, df_plot["pca_dominancia"], color="#F0A500", linewidth=1.0)
    ax3.axhline(DOMINANCIA_MINIMA, color="#E6EDF3", linestyle=":", alpha=0.8, label=f"Threshold ({DOMINANCIA_MINIMA})")
    ax3.fill_between(df_plot.index, 0, DOMINANCIA_MINIMA, color="#F85149", alpha=0.1)
    ax3.fill_between(df_plot.index, DOMINANCIA_MINIMA, 1.0, color="#3FB950", alpha=0.1)
    ax3.set_ylim(0, 1)
    ax3.set_ylabel("Dominância (PC1)", fontsize=10)
    ax3.legend(loc="upper left", framealpha=0.3)
    ax3.grid(True, linestyle="--", alpha=0.2)
    
    GRAFICO_SAIDA.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(GRAFICO_SAIDA, dpi=120, bbox_inches="tight", facecolor="#0D1117")
    plt.close()
    logger.info(f"Gráfico exportado para: {GRAFICO_SAIDA}")

if __name__ == "__main__":
    try:
        executar_pipeline_pca()
        print("\n> Módulo PCA concluído com sucesso!")
    except Exception as e:
        logger.exception("Falha na execução do pipeline PCA:")
        sys.exit(1)
