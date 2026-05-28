# -*- coding: utf-8 -*-
"""
================================================================================
pca_strategy.py — Estratégia Baseada em Análise de Componentes Principais (PCA)
================================================================================

Módulo de Álgebra Linear & Decomposição Espectral para o projeto quant_eurusd_v2.
Lógica:
    - Extração de matriz de features com 60 períodos (R_t, R_t^2, |R_t|, R_t-1, R_t-2).
    - PCA rolante sobre a matriz padronizada de covariância.
    - Z-Score da Primeira Componente Principal (PC1) para detectar distorções extremas.
    - Ratio de Dominância para confirmar estrutura clara no mercado.
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
DIR_DATA = DIR_ATUAL / "data"
DIR_GRAFICOS = DIR_ATUAL / "graficos"
PARQUET_COMPLETO = DIR_DATA / "eurusd_h1_completo.parquet"
PARQUET_OPERACIONAL = DIR_DATA / "eurusd_h1_operacional.parquet"
PARQUET_SAIDA = DIR_DATA / "eurusd_h1_pca.parquet"

GRAFICO_SAIDA = DIR_GRAFICOS / "pca_sinais.png"

# Parâmetros da Estratégia PCA
JANELA_PCA = 40
JANELA_VOL = 50
DOMINANCIA_MINIMA = 0.55
ZSCORE_THRESHOLD = 2.2
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

def gerar_tabela_parametros():
    import matplotlib.pyplot as plt
    logger.info("Gerando gráfico da tabela de parâmetros em Dark Mode...")
    DIR_GRAFICOS.mkdir(parents=True, exist_ok=True)
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    dados = [[k, str(v)] for k, v in {'Estratégia': 'PCA Arbitrage', 'Janela PCA': 40, 'Dominância PC1': '> 55%', 'Desvio Ativação': 2.2, 'Stop Loss (Risco)': '2.0x Vol', 'Take Profit (Alvo)': '4.0x Vol'}.items()]
    tabela = ax.table(cellText=dados, colLabels=["Métrica", "Valor Otimizado"], loc="center", cellLoc="left")
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(12)
    tabela.scale(1.2, 2.0)
    
    for (row, col), cell in tabela.get_celld().items():
        cell.set_edgecolor("#333333")
        if row == 0:
            cell.set_text_props(weight="bold", color="#FFFFFF", fontsize=13)
            cell.set_facecolor("#1E4F8A")
        else:
            cell.set_facecolor("#121212")
            cell.set_text_props(color="#CFD8DC")
            if col == 0:
                cell.set_text_props(weight="bold", color="#82AAFF")
                
    fig.suptitle("Configuração de Hiperparâmetros — PCA", color="#FFFFFF", fontsize=16, fontweight="bold", y=0.95)
    plt.tight_layout()
    caminho = DIR_GRAFICOS / "parametros_pca.png"
    plt.savefig(caminho, dpi=150, facecolor="#121212")
    plt.close()
    logger.info(f"Tabela de parâmetros salva em: {caminho}")

# PIPELINE PRINCIPAL
# =============================================================================

def executar_pipeline_pca():
    logger.info("Carregando bases de dados H1...")
    
    if not PARQUET_COMPLETO.exists() or not PARQUET_OPERACIONAL.exists():
        raise FileNotFoundError("Os arquivos Parquet completos e operacionais não foram encontrados na pasta data/.")
        
    df = pd.read_parquet(PARQUET_COMPLETO, engine="pyarrow")
    df_op = pd.read_parquet(PARQUET_OPERACIONAL, engine="pyarrow")
    
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
    df["tp_pips"] = 4.0 * df["vr_pips"]
    
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
    # Desativado a pedido do usuário
    pass

if __name__ == "__main__":
    try:
        gerar_tabela_parametros()
        executar_pipeline_pca()
        print("\n> Módulo PCA concluído com sucesso!")
    except Exception as e:
        logger.exception("Falha na execução do pipeline PCA:")
        sys.exit(1)
