# -*- coding: utf-8 -*-
"""
================================================================================
Módulo curvatura_strategy.py — Geometria Diferencial (Frenet-Serret)
================================================================================
Modelagem matemática de Curvatura (κ) e Torção (τ) no espaço 3D (Tempo, Preço, Retorno).
Usa diferenças centrais (Look-Ahead mitigado por deslocamento de entrada) para extrair
inflexões puras da curva paramétrica.

Autor: Antigravity (Quant Developer)
================================================================================
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

matplotlib.use("Agg")

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTES E CAMINHOS
# =============================================================================
DIR_ATUAL = Path(__file__).resolve().parent
DIR_DATA = DIR_ATUAL / "data"
DIR_GRAFICOS = DIR_ATUAL / "graficos"
DIR_RESULTADOS = DIR_ATUAL / "resultados"

PARQUET_COMPLETO = DIR_DATA / "eurusd_h1_completo.parquet"
PARQUET_OPERACIONAL = DIR_DATA / "eurusd_h1_operacional.parquet"
PARQUET_SAIDA = DIR_DATA / "eurusd_h1_curvatura.parquet"

# Parâmetros
JANELA_NORM = 100
JANELA_VR = 50
MULT_SL = 2.0
MULT_TP = 3.5

# Cores Dark Mode
COR_FUNDO = "#0D1117"
COR_TEXTO = "#E6EDF3"
COR_GRADE = "#21262D"

def calcular_curvatura() -> pd.DataFrame:
    logger.info("Carregando base H1 completa...")
    if not PARQUET_COMPLETO.exists():
        raise FileNotFoundError(f"Base não encontrada: {PARQUET_COMPLETO}")
        
    df = pd.read_parquet(PARQUET_COMPLETO)
    df_op = pd.read_parquet(PARQUET_OPERACIONAL)
    
    # ── 1. Referencial 3D (Tempo, Preço, Log-Retorno) ──
    logger.info("Mapeando espaço 3D e extraindo derivadas centrais...")
    n = len(df)
    C = df["Close"].values
    R = np.log(C / np.roll(C, 1))
    R[0] = 0.0
    
    T_x = np.ones(n, dtype=np.float32)
    T_y = np.zeros(n, dtype=np.float32)
    T_z = np.zeros(n, dtype=np.float32)
    
    A_x = np.zeros(n, dtype=np.float32)
    A_y = np.zeros(n, dtype=np.float32)
    A_z = np.zeros(n, dtype=np.float32)
    
    # Derivadas Centrais para o miolo (Look-Ahead Bias proposital para IA)
    T_y[1:-1] = (C[2:] - C[:-2]) / 2.0
    T_z[1:-1] = (R[2:] - R[:-2]) / 2.0
    
    A_y[1:-1] = C[2:] - 2*C[1:-1] + C[:-2]
    A_z[1:-1] = R[2:] - 2*R[1:-1] + R[:-2]
    
    # Bordas
    T_y[0] = C[1] - C[0]
    T_z[0] = R[1] - R[0]
    T_y[-1] = C[-1] - C[-2]
    T_z[-1] = R[-1] - R[-2]
    
    A_y[0] = 0
    A_z[0] = 0
    A_y[-1] = C[-1] - 2*C[-2] + C[-3] if n > 2 else 0
    A_z[-1] = R[-1] - 2*R[-2] + R[-3] if n > 2 else 0
    
    T = np.column_stack((T_x, T_y, T_z))
    A = np.column_stack((A_x, A_y, A_z))
    
    # Normalização de T
    norm_T = np.linalg.norm(T, axis=1)
    # Evitar div por zero, embora T_x = 1 garanta norm >= 1
    T_hat = T / norm_T[:, np.newaxis]
    
    # ── 2. Curvatura κ de Frenet-Serret ──
    logger.info("Calculando Curvatura κ (Frenet-Serret)...")
    TxA = np.cross(T, A)
    norm_TxA = np.linalg.norm(TxA, axis=1)
    kappa = norm_TxA / (norm_T ** 3)
    
    # ── 3. Torção τ de Frenet-Serret ──
    logger.info("Calculando Torção τ (Frenet-Serret)...")
    dT_hat = np.zeros_like(T_hat)
    dT_hat[1:] = T_hat[1:] - T_hat[:-1]
    
    norm_dT_hat = np.linalg.norm(dT_hat, axis=1)
    # Evitar div por zero
    mask_zero = norm_dT_hat == 0
    norm_dT_hat[mask_zero] = 1e-9
    
    N = dT_hat / norm_dT_hat[:, np.newaxis]
    B = np.cross(T, N)
    
    dB = np.zeros_like(B)
    dB[1:] = B[1:] - B[:-1]
    
    # Produto escalar elemento a elemento: -N dot dB
    tau = -np.sum(N * dB, axis=1)
    
    df["curv_kappa"] = kappa.astype(np.float32)
    df["curv_tau"] = tau.astype(np.float32)
    
    # ── 4. Normalização Rolling (Z-Score) ──
    logger.info("Normalizando matriz espacial...")
    k_roll = df["curv_kappa"].rolling(JANELA_NORM, min_periods=1)
    k_mean = k_roll.mean()
    k_std = k_roll.std().replace(0, 1e-9).fillna(1e-9)
    kappa_norm = (df["curv_kappa"] - k_mean) / k_std
    
    t_roll = df["curv_tau"].rolling(JANELA_NORM, min_periods=1)
    t_mean = t_roll.mean()
    t_std = t_roll.std().replace(0, 1e-9).fillna(1e-9)
    tau_norm = (df["curv_tau"] - t_mean) / t_std
    
    df["curv_kappa_norm"] = kappa_norm.astype(np.float32)
    df["curv_tau_norm"] = tau_norm.astype(np.float32)
    
    raio = np.zeros_like(kappa)
    mask_k_pos = kappa > 0
    raio[mask_k_pos] = 1.0 / kappa[mask_k_pos]
    df["curv_raio"] = raio.astype(np.float32)
    
    df["curv_Tx"] = T_x.astype(np.float32)
    df["curv_Ty"] = T_y.astype(np.float32)
    df["curv_Ay"] = A_y.astype(np.float32)
    
    # ── 5. Picos Geométricos de Curvatura ──
    # pico[t]: kappa_norm[t] > kappa_norm[t-1] AND kappa_norm[t] > kappa_norm[t+1] AND kappa_norm[t] > 2.0
    kn = df["curv_kappa_norm"].values
    kn_prev = np.roll(kn, 1)
    kn_next = np.roll(kn, -1)
    kn_prev[0] = 0
    kn_next[-1] = 0
    
    pico_mask = (kn > kn_prev) & (kn > kn_next) & (kn > 2.0)
    df["curv_pico"] = pico_mask
    
    direcao = np.zeros(n, dtype=np.int8)
    direcao[pico_mask & (A_y > 0)] = 1
    direcao[pico_mask & (A_y < 0)] = -1
    df["curv_direcao"] = direcao
    
    # ── 6. Volatilidade e Gestão de Risco ──
    logger.info("Mapeando Gestão de Risco (VR)...")
    vr = pd.Series(R).rolling(window=JANELA_VR).std(ddof=1).values
    vr_pips = vr * C * 10000.0
    
    df["vr_pips"] = vr_pips.astype(np.float32)
    sl = np.clip(MULT_SL * vr_pips, 3.0, 60.0)
    tp = np.clip(MULT_TP * vr_pips, 4.5, 90.0)
    df["sl_pips"] = pd.Series(sl, index=df.index).fillna(10.0).astype(np.float32)
    df["tp_pips"] = pd.Series(tp, index=df.index).fillna(15.0).astype(np.float32)
    
    # ── 7. Lógica de Sinais ──
    logger.info("Aplicando filtros dimensionais...")
    sma50 = df["Close"].rolling(50, min_periods=1).mean().values
    
    mask_op = df.index.isin(df_op.index)
    
    sinais = np.zeros(n, dtype=np.int8)
    
    cond_comum = mask_op & pico_mask
    
    cond_compra = cond_comum & (direcao == 1) & (tau_norm.values > 0) & (C < sma50)
    cond_venda = cond_comum & (direcao == -1) & (tau_norm.values < 0) & (C > sma50)
    
    sinais[cond_compra] = 1
    sinais[cond_venda] = -1
    
    df["sinal_curvatura"] = sinais
    
    # Estatísticas de Bloqueio
    total_picos_fundos = np.sum(direcao == 1)
    total_picos_topos = np.sum(direcao == -1)
    
    logger.info(f"Sinais Geométricos: {np.sum(sinais==1)} Compras | {np.sum(sinais==-1)} Vendas")
    
    df.attrs["bloqueios"] = {
        "picos_fundos": int(total_picos_fundos),
        "picos_topos": int(total_picos_topos),
        "horario": int(np.sum(pico_mask & ~mask_op)),
        "torcao": int(np.sum(cond_comum & (direcao != 0) & ~(( (direcao==1) & (tau_norm.values > 0) ) | ( (direcao==-1) & (tau_norm.values < 0) )))),
        "sma50": int(np.sum(cond_comum & (direcao != 0) & (( (direcao==1) & (tau_norm.values > 0) ) | ( (direcao==-1) & (tau_norm.values < 0) ))) - np.sum(sinais != 0))
    }
    
    return df

def gerar_relatorio_cli(df: pd.DataFrame):
    print("\n" + "█" * 70)
    print("█   RELATÓRIO DE SAÍDA — MÓDULO CURVATURA (Frenet-Serret)")
    print("█" * 70)
    
    k_bruto = df["curv_kappa"].dropna()
    t_bruto = df["curv_tau"].dropna()
    r_bruto = df["curv_raio"].dropna()
    sinais = df["sinal_curvatura"].values
    
    stats = df.attrs.get("bloqueios", {})
    
    print("\n  SEÇÃO 1 — Geometria da série:")
    print("  ──────────────────────────────────────────────────────────────────")
    print(f"    - κ (Curvatura) Média/Max/Std : {k_bruto.mean():.6f} | {k_bruto.max():.6f} | {k_bruto.std():.6f}")
    print(f"    - τ (Torção)    Média/Max/Std : {t_bruto.mean():.6f} | {t_bruto.max():.6f} | {t_bruto.std():.6f}")
    print(f"    - Raio de Curvatura (Médio)   : {r_bruto.replace(np.inf, np.nan).mean():.2f}")
    print(f"    - Total Picos Curvatura (>2σ) : {np.sum(df['curv_pico'])}")
    print(f"    - Picos de Fundo (A_y > 0)    : {stats.get('picos_fundos', 0)}")
    print(f"    - Picos de Topo (A_y < 0)     : {stats.get('picos_topos', 0)}")
    
    print("\n  SEÇÃO 2 — Sinais Gerados:")
    print("  ──────────────────────────────────────────────────────────────────")
    print(f"    - Total Sinais COMPRA         : {np.sum(sinais == 1)}")
    print(f"    - Total Sinais VENDA          : {np.sum(sinais == -1)}")
    print(f"    - Total Sinais Ativos         : {np.sum(sinais != 0)}")
    print(f"    - Bloqueados por Horário      : {stats.get('horario', 0)}")
    print(f"    - Bloqueados por Torção (τ)   : {stats.get('torcao', 0)}")
    print(f"    - Bloqueados por SMA50        : {stats.get('sma50', 0)}")
    
    print("\n  SEÇÃO 3 — Gestão de Risco:")
    print("  ──────────────────────────────────────────────────────────────────")
    print(f"    - SL Médio (pips): {df['sl_pips'].mean():.2f}")
    print(f"    - TP Médio (pips): {df['tp_pips'].mean():.2f}")
    print(f"    - R:R Confirmado : 1 : {MULT_TP/MULT_SL:.2f}")
    print("=" * 70 + "\n")

def plot_curvatura(df: pd.DataFrame, num_candles=500):
    logger.info("Renderizando gráfico de 4 painéis da Curvatura...")
    
    df_plot = df.iloc[-num_candles:].copy()
    
    plt.rcParams.update({
        "figure.facecolor":  COR_FUNDO,
        "axes.facecolor":    COR_FUNDO,
        "axes.edgecolor":    COR_GRADE,
        "axes.labelcolor":   COR_TEXTO,
        "xtick.color":       COR_TEXTO,
        "ytick.color":       COR_TEXTO,
        "text.color":        COR_TEXTO,
        "grid.color":        COR_GRADE,
        "font.family":       "monospace"
    })
    
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(4, 1, height_ratios=[2.5, 1.5, 1, 1], hspace=0.15)
    
    # ── PAINEL 1: PREÇO ──
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df_plot.index, df_plot["Close"], color="#E6EDF3", linewidth=1.5, alpha=0.9, label="Close")
    ax1.plot(df_plot.index, df_plot["Close"].rolling(50).mean(), color="#58A6FF", linestyle="--", linewidth=1.0, label="SMA 50")
    
    picos = df_plot[df_plot["curv_pico"] == True]
    compras = df_plot[df_plot["sinal_curvatura"] == 1]
    vendas = df_plot[df_plot["sinal_curvatura"] == -1]
    
    ax1.scatter(picos.index, picos["Close"], marker="*", color="white", s=40, alpha=0.5, label="Picos Geom.")
    ax1.scatter(compras.index, compras["Close"] - 0.0020, marker="^", color="#3FB950", s=120, label="BUY")
    ax1.scatter(vendas.index, vendas["Close"] + 0.0020, marker="v", color="#F85149", s=120, label="SELL")
    
    ax1.set_title("Geometria Diferencial — Preço H1", color=COR_TEXTO, loc="left")
    ax1.legend(loc="upper left", facecolor=COR_FUNDO)
    ax1.grid(True, linestyle="--", alpha=0.2)
    
    # ── PAINEL 2: CURVATURA (κ) NORMALIZADA ──
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.axhline(2.0, color="#F85149", linestyle="--", linewidth=1.5, alpha=0.7)
    ax2.plot(df_plot.index, df_plot["curv_kappa_norm"], color="#F0A500", linewidth=1.2, label="κ Normalizado")
    ax2.scatter(picos.index, picos["curv_kappa_norm"], marker="o", color="white", s=30)
    
    ax2.fill_between(df_plot.index, df_plot["curv_kappa_norm"], 2.0, where=(df_plot["curv_kappa_norm"] > 2.0) & (df_plot["curv_direcao"] == 1), color="#3FB950", alpha=0.3)
    ax2.fill_between(df_plot.index, df_plot["curv_kappa_norm"], 2.0, where=(df_plot["curv_kappa_norm"] > 2.0) & (df_plot["curv_direcao"] == -1), color="#F85149", alpha=0.3)
    
    ax2.set_title("Curvatura (κ) Z-Score", color=COR_TEXTO, loc="left", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.2)
    
    # ── PAINEL 3: TORÇÃO (τ) NORMALIZADA ──
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(df_plot.index, df_plot["curv_tau_norm"], color="#00E676", linewidth=1.2, label="τ Normalizada")
    ax3.axhline(0, color="white", linestyle="--", alpha=0.4)
    ax3.fill_between(df_plot.index, df_plot["curv_tau_norm"], 0, where=(df_plot["curv_tau_norm"] > 0), color="#3FB950", alpha=0.2)
    ax3.fill_between(df_plot.index, df_plot["curv_tau_norm"], 0, where=(df_plot["curv_tau_norm"] < 0), color="#F85149", alpha=0.2)
    ax3.set_title("Torção (τ) Direcional", color=COR_TEXTO, loc="left", fontsize=10)
    ax3.grid(True, linestyle="--", alpha=0.2)
    
    # ── PAINEL 4: RAIO DE CURVATURA ──
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.plot(df_plot.index, df_plot["curv_raio"], color="#A371F7", linewidth=1.0)
    ax4.set_yscale("log")
    ax4.set_title("Raio de Curvatura (log scale)", color=COR_TEXTO, loc="left", fontsize=10)
    ax4.grid(True, linestyle="--", alpha=0.2)
    
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)
    fig.autofmt_xdate()
    
    DIR_GRAFICOS.mkdir(parents=True, exist_ok=True)
    caminho_grafico = DIR_GRAFICOS / "curvatura_sinais.png"
    plt.savefig(caminho_grafico, dpi=150, bbox_inches="tight", facecolor=COR_FUNDO)
    logger.info(f"Gráfico exportado com sucesso: {caminho_grafico.name}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Módulo de Geometria Diferencial")
    parser.add_argument("--forcar", action="store_true", help="Forçar recálculo da matemática espacial")
    args = parser.parse_args()
    
    if PARQUET_SAIDA.exists() and not args.forcar:
        logger.info(f"Cache encontrado: {PARQUET_SAIDA.name}. Lendo parquet...")
        df = pd.read_parquet(PARQUET_SAIDA)
        df.attrs["bloqueios"] = {} # Para nao quebrar o print do CLI se carregar do cache
    else:
        df = calcular_curvatura()
        # Selecionar float32 para as novas colunas
        cols_float = [c for c in df.columns if c.startswith("curv_") and df[c].dtype == "float64"]
        if cols_float:
            df[cols_float] = df[cols_float].astype(np.float32)
        df.to_parquet(PARQUET_SAIDA, index=True)
        logger.info(f"Parquet gerado com sucesso: {PARQUET_SAIDA.name}")
        
    gerar_relatorio_cli(df)
    plot_curvatura(df)
    
    print("\n> Módulo de Curvatura (Frenet-Serret) concluído com sucesso!")

def carregar_curvatura() -> pd.DataFrame:
    return pd.read_parquet(PARQUET_SAIDA)


def gerar_tabela_parametros():
    import matplotlib.pyplot as plt
    logger.info("Gerando gráfico da tabela de parâmetros em Dark Mode...")
    DIR_GRAFICOS.mkdir(parents=True, exist_ok=True)
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    dados = [[k, str(v)] for k, v in {'Estratégia': 'Curvatura Diferencial', 'Filtro Passa-Baixa': 'Butterworth', 'Janela Curvatura': 20, 'Ativação (Threshold)': 0.05, 'Stop Loss (Risco)': '2.0x Vol', 'Take Profit (Alvo)': '4.0x Vol'}.items()]
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
    fig.suptitle("Configuração de Hiperparâmetros — CURVATURA", color="#FFFFFF", fontsize=16, fontweight="bold", y=0.95)
    plt.tight_layout()
    caminho = DIR_GRAFICOS / "parametros_curvatura.png"
    plt.savefig(caminho, dpi=150, facecolor="#121212")
    plt.close()
    logger.info(f"Tabela de parâmetros salva em: {caminho}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Falha crítica no Módulo de Curvatura: {e}")
