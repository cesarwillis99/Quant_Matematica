# -*- coding: utf-8 -*-
"""
================================================================================
Módulo wavelet_strategy.py — Transformada Wavelet Contínua (CWT) H1
================================================================================
Matemática: 
- Decomposição em 4 escalas usando Morlet Complexa (cmor1.5-1.0)
- Extração de Fases, Potências e Coerência Cruzada
- Captura de pontos de inflexão do ciclo dominante

Autor: Antigravity (Quant Developer)
================================================================================
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import pywt
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
PARQUET_SAIDA = DIR_DATA / "eurusd_h1_wavelet.parquet"

# Parâmetros
ESCALAS = np.array([4, 12, 24, 120])
WAVELET = 'cmor1.5-1.0'
SMOOTH_WINDOW = 4  # Janela curta para evitar cancelamento de fase na escala 4h
POT_NORM_WINDOW = 100
ENERGIA_WINDOW = 100

JANELA_VR = 50
MULT_SL = 2.0
MULT_TP = 3.5

# Cores Dark Mode
COR_FUNDO = "#0D1117"
COR_TEXTO = "#E6EDF3"
COR_GRADE = "#21262D"

def calcular_wavelet() -> pd.DataFrame:
    logger.info("Carregando base H1 completa...")
    if not PARQUET_COMPLETO.exists():
        raise FileNotFoundError(f"Base não encontrada: {PARQUET_COMPLETO}")
        
    df = pd.read_parquet(PARQUET_COMPLETO)
    df_op = pd.read_parquet(PARQUET_OPERACIONAL)
    mask_op = df.index.isin(df_op.index)
    
    # ── 1. CWT (Transformada Wavelet Contínua) ──
    logger.info("Aplicando CWT Morlet nas 4 escalas...")
    log_ret = np.log(df["Close"] / df["Close"].shift(1)).fillna(0).values
    
    coefs, freqs = pywt.cwt(log_ret, scales=ESCALAS, wavelet=WAVELET)
    
    potencia = np.abs(coefs) ** 2
    fase = np.angle(coefs)
    
    df["wav_pot_s1"] = potencia[0]
    df["wav_pot_s2"] = potencia[1]
    df["wav_pot_s3"] = potencia[2]
    df["wav_pot_s4"] = potencia[3]
    
    df["wav_fase_s1"] = fase[0]
    df["wav_fase_s2"] = fase[1]
    
    logger.info("Calculando coerências e gradientes...")
    df_pot = pd.DataFrame(potencia.T, index=df.index)
    pot_norm = df_pot / df_pot.rolling(window=POT_NORM_WINDOW, min_periods=1).mean()
    
    df["wav_pot_norm_s1"] = pot_norm[0].values
    df["wav_pot_norm_s2"] = pot_norm[1].values
    
    # Cross-Scale Coherence (S12 e S34) - Separando partes Real e Imag para evitar warnings/NaNs
    cross_12 = coefs[0] * np.conj(coefs[1])
    S12_real = pd.Series(np.real(cross_12)).rolling(SMOOTH_WINDOW, min_periods=1).mean().values
    S12_imag = pd.Series(np.imag(cross_12)).rolling(SMOOTH_WINDOW, min_periods=1).mean().values
    S12_sq = S12_real**2 + S12_imag**2
    
    S11 = df_pot[0].rolling(SMOOTH_WINDOW, min_periods=1).mean().values
    S22 = df_pot[1].rolling(SMOOTH_WINDOW, min_periods=1).mean().values
    df["wav_coerencia_12"] = S12_sq / (S11 * S22)
    
    cross_34 = coefs[2] * np.conj(coefs[3])
    S34_real = pd.Series(np.real(cross_34)).rolling(SMOOTH_WINDOW, min_periods=1).mean().values
    S34_imag = pd.Series(np.imag(cross_34)).rolling(SMOOTH_WINDOW, min_periods=1).mean().values
    S34_sq = S34_real**2 + S34_imag**2
    
    S33 = df_pot[2].rolling(SMOOTH_WINDOW, min_periods=1).mean().values
    S44 = df_pot[3].rolling(SMOOTH_WINDOW, min_periods=1).mean().values
    df["wav_coerencia_34"] = S34_sq / (S33 * S44)
    
    df["wav_dom_escala"] = np.argmax(potencia, axis=0).astype(np.int8)
    df["wav_energia_total"] = np.sum(potencia, axis=0)
    
    # Gradiente de Fase (com unwrap para evitar pulos de 2pi)
    fase_unwrapped = np.unwrap(fase, axis=1)
    d_fase_all = np.zeros_like(fase_unwrapped)
    d_fase_all[:, 1:] = np.diff(fase_unwrapped, axis=1)
    
    dom_idx = df["wav_dom_escala"].values
    d_fase = d_fase_all[dom_idx, np.arange(len(dom_idx))]
    df["wav_d_fase"] = d_fase
    df["wav_d_fase_s3"] = d_fase_all[2, :]  # Derivada de fase fixa da escala diária (24h)
    
    df["wav_energia_p40"] = df["wav_energia_total"].rolling(ENERGIA_WINDOW, min_periods=1).quantile(0.40)
    
    # ── 2. VR e Gestão de Risco ──
    logger.info("Calculando Volatilidade Realizada (VR)...")
    vr = pd.Series(log_ret).rolling(window=JANELA_VR).std(ddof=1).values
    df["vr_pips"] = vr * df["Close"] * 10000.0
    
    df["sl_pips"] = MULT_SL * df["vr_pips"]
    df["tp_pips"] = MULT_TP * df["vr_pips"]
    df["sl_pips"] = df["sl_pips"].clip(lower=3.0, upper=60.0).fillna(10.0)
    df["tp_pips"] = df["tp_pips"].clip(lower=4.5, upper=90.0).fillna(15.0)
    
    # ── 3. Lógica de Geração de Sinais ──
    logger.info("Aplicando as regras de estado direcional...")
    sinais = np.zeros(len(df), dtype=np.int8)
    
    coer_ok = df["wav_coerencia_12"].values >= 0.50
    energ_ok = df["wav_energia_total"].values > df["wav_energia_p40"].values
    
    df_d_fase = df["wav_d_fase"].values
    d_fase_prev = np.roll(df_d_fase, 1)
    d_fase_prev[0] = 0
    
    inflex_up = (df_d_fase > 0) & (d_fase_prev <= 0)
    inflex_dn = (df_d_fase < 0) & (d_fase_prev >= 0)
    
    pot_s1_ok = df["wav_pot_norm_s1"].values < 0.8
    ret_neg = log_ret < 0
    ret_pos = log_ret > 0
    
    cond_compra = mask_op & coer_ok & energ_ok & inflex_up & pot_s1_ok & ret_neg
    cond_venda  = mask_op & coer_ok & energ_ok & inflex_dn & pot_s1_ok & ret_pos
    
    sinais[cond_compra] = 1
    sinais[cond_venda] = -1
    
    df["sinal_wavelet"] = sinais
    
    total_c = np.sum(sinais == 1)
    total_v = np.sum(sinais == -1)
    logger.info(f"Sinais Wavelet: {total_c} Compras | {total_v} Vendas | Total: {total_c + total_v}")
    
    return df

def gerar_relatorio_cli(df: pd.DataFrame):
    print("\n" + "█" * 70)
    print("█   RELATÓRIO DE SAÍDA — MÓDULO WAVELET (CWT)")
    print("█" * 70)
    
    sinais = df["sinal_wavelet"].values
    coer = df["wav_coerencia_12"].dropna()
    ene = df["wav_energia_total"].dropna()
    dom = df["wav_dom_escala"].value_counts(normalize=True) * 100
    d_fase = df["wav_d_fase"].values
    d_fase_prev = np.roll(d_fase, 1)
    
    inflex_up = np.sum((d_fase > 0) & (d_fase_prev <= 0))
    inflex_dn = np.sum((d_fase < 0) & (d_fase_prev >= 0))
    
    print("\n  SEÇÃO 1 — Análise Espectral:")
    print("  ──────────────────────────────────────────────────────────────────")
    print(f"    - Escala Dominante 1 (4h)  : {dom.get(0, 0):.2f}% do tempo")
    print(f"    - Escala Dominante 2 (12h) : {dom.get(1, 0):.2f}% do tempo")
    print(f"    - Escala Dominante 3 (24h) : {dom.get(2, 0):.2f}% do tempo")
    print(f"    - Escala Dominante 4 (120h): {dom.get(3, 0):.2f}% do tempo")
    print(f"    - Energia Total (Med/Min/Max): {ene.mean():.6f} | {ene.min():.6f} | {ene.max():.6f}")
    print(f"    - Coerência s1-s2 (Média)    : {coer.mean():.4f}")
    print(f"    - % Tempo Coerência >= 0.50  : {(np.sum(coer >= 0.50) / len(coer) * 100):.2f}%")
    
    print("\n  SEÇÃO 2 — Pontos de Inflexão (Fase):")
    print("  ──────────────────────────────────────────────────────────────────")
    print(f"    - Total Inversões de Fase: {inflex_up + inflex_dn:,}")
    print(f"    - Inversões Ascendentes  : {inflex_up:,}")
    print(f"    - Inversões Descendentes : {inflex_dn:,}")
    
    print("\n  SEÇÃO 3 — Sinais Gerados:")
    print("  ──────────────────────────────────────────────────────────────────")
    print(f"    - Total Sinais COMPRA : {np.sum(sinais == 1)}")
    print(f"    - Total Sinais VENDA  : {np.sum(sinais == -1)}")
    print(f"    - Total Sinais Ativos : {np.sum(sinais != 0)}")
    
    # Bloqueios teóricos
    mask_op = df.index.isin(pd.read_parquet(PARQUET_OPERACIONAL).index)
    bloq_horario = np.sum(~mask_op)
    bloq_coer = np.sum(df["wav_coerencia_12"] < 0.50)
    bloq_ene = np.sum(df["wav_energia_total"] <= df["wav_energia_p40"])
    print(f"    - Bloqueados por Horário    : {bloq_horario:,} candles")
    print(f"    - Bloqueados por Coerência  : {bloq_coer:,} candles")
    print(f"    - Bloqueados por Energia    : {bloq_ene:,} candles")
    
    print("\n  SEÇÃO 4 — Gestão de Risco:")
    print("  ──────────────────────────────────────────────────────────────────")
    print(f"    - SL Médio (pips): {df['sl_pips'].mean():.2f}")
    print(f"    - TP Médio (pips): {df['tp_pips'].mean():.2f}")
    print(f"    - R:R Confirmado : 1 : {MULT_TP/MULT_SL:.2f}")
    print("=" * 70 + "\n")

def plot_wavelet(df: pd.DataFrame, num_candles=500):
    logger.info("Renderizando gráfico de 4 painéis...")
    
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
    
    # ── PAINEL 1: PREÇO E SINAIS ──
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df_plot.index, df_plot["Close"], color="#E6EDF3", linewidth=1.5, alpha=0.9)
    
    compras = df_plot[df_plot["sinal_wavelet"] == 1]
    vendas = df_plot[df_plot["sinal_wavelet"] == -1]
    
    ax1.scatter(compras.index, compras["Close"] - 0.0020, marker="^", color="#3FB950", s=120, label="BUY")
    ax1.scatter(vendas.index, vendas["Close"] + 0.0020, marker="v", color="#F85149", s=120, label="SELL")
    
    ax1.set_title("Wavelet Morlet (CWT) — Ação de Preço H1", color=COR_TEXTO, loc="left")
    ax1.legend(loc="upper left", facecolor=COR_FUNDO)
    ax1.grid(True, linestyle="--", alpha=0.2)
    
    # ── PAINEL 2: POTÊNCIA WAVELET (ESCALAS) ──
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(df_plot.index, df_plot["wav_pot_s1"], color="#58A6FF", linewidth=1.2, label="s1 (4h)")
    ax2.plot(df_plot.index, df_plot["wav_pot_s2"], color="#3FB950", linewidth=1.2, label="s2 (12h)")
    ax2.plot(df_plot.index, df_plot["wav_pot_s3"], color="#F0A500", linewidth=1.2, label="s3 (24h)")
    ax2.plot(df_plot.index, df_plot["wav_pot_s4"], color="#A371F7", linewidth=1.2, label="s4 (120h)")
    ax2.set_title("Potência Espectral (Power)", color=COR_TEXTO, loc="left", fontsize=10)
    ax2.legend(loc="upper left", ncol=4, facecolor=COR_FUNDO, fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.2)
    
    # ── PAINEL 3: CROSS-COHERENCE S1-S2 ──
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(df_plot.index, df_plot["wav_coerencia_12"], color="#00E676", linewidth=1.2, label="Coerência s1-s2")
    ax3.axhline(0.50, color="#F85149", linestyle="--", alpha=0.7)
    ax3.fill_between(df_plot.index, df_plot["wav_coerencia_12"], 0.50, where=(df_plot["wav_coerencia_12"] >= 0.50), color="#00E676", alpha=0.2)
    ax3.set_ylim(0, 1.05)
    ax3.set_title("Cross-Scale Coherence (Sincronia de Ciclos)", color=COR_TEXTO, loc="left", fontsize=10)
    ax3.grid(True, linestyle="--", alpha=0.2)
    
    # ── PAINEL 4: GRADIENTE DE FASE DA ESCALA DOMINANTE ──
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.plot(df_plot.index, df_plot["wav_d_fase"], color="#E6EDF3", linewidth=1.0)
    ax4.axhline(0, color="white", linestyle="--", alpha=0.4)
    ax4.fill_between(df_plot.index, df_plot["wav_d_fase"], 0, where=(df_plot["wav_d_fase"] > 0), color="#3FB950", alpha=0.2)
    ax4.fill_between(df_plot.index, df_plot["wav_d_fase"], 0, where=(df_plot["wav_d_fase"] < 0), color="#F85149", alpha=0.2)
    ax4.set_title("Gradiente de Fase (d_fase) - Inflexões", color=COR_TEXTO, loc="left", fontsize=10)
    ax4.grid(True, linestyle="--", alpha=0.2)
    
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)
    fig.autofmt_xdate()
    
    DIR_GRAFICOS.mkdir(parents=True, exist_ok=True)
    caminho_grafico = DIR_GRAFICOS / "wavelet_sinais.png"
    plt.savefig(caminho_grafico, dpi=150, bbox_inches="tight", facecolor=COR_FUNDO)
    logger.info(f"Gráfico exportado com sucesso: {caminho_grafico.name}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Módulo Wavelet CWT EURUSD H1")
    parser.add_argument("--forcar", action="store_true", help="Forçar recálculo da CWT")
    args = parser.parse_args()
    
    if PARQUET_SAIDA.exists() and not args.forcar:
        logger.info(f"Cache encontrado: {PARQUET_SAIDA.name}. Lendo parquet...")
        df = pd.read_parquet(PARQUET_SAIDA)
    else:
        df = calcular_wavelet()
        # Selecionar float32 para as novas colunas
        cols_float = [c for c in df.columns if c.startswith("wav_") and df[c].dtype == "float64"]
        df[cols_float] = df[cols_float].astype(np.float32)
        df["vr_pips"] = df["vr_pips"].astype(np.float32)
        df["sl_pips"] = df["sl_pips"].astype(np.float32)
        df["tp_pips"] = df["tp_pips"].astype(np.float32)
        
        df.to_parquet(PARQUET_SAIDA, index=True)
        logger.info(f"Parquet gerado com sucesso: {PARQUET_SAIDA.name}")
        
    gerar_relatorio_cli(df)
    plot_wavelet(df)
    
    print("\n> Módulo Wavelet concluído com sucesso!")

# Utilitário externo
def carregar_wavelet() -> pd.DataFrame:
    return pd.read_parquet(PARQUET_SAIDA)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Falha crítica no Módulo Wavelet: {e}")
