# -*- coding: utf-8 -*-
"""
================================================================================
momentum.py — Módulo de Cinemática do Preço + Entropia de Shannon (24h)
================================================================================
Autor: Quant Developer Sênior
Data: 2026

Descrição:
    Este módulo implementa uma estratégia de Trend Following baseada em cinemática
    do preço, Entropia de Shannon e filtro de regime Hurst.

    CORREÇÃO ESTRUTURAL:
        - Lê eurusd_h1_hurst.parquet
        - Calcula cinemática e Entropia de Shannon
        - APENAS na geração de sinais aplica o filtro operacional de trading
          (domingo 21h00 UTC até sexta 20h00 UTC)
        - Salva os dados em data/eurusd_h1_momentum.parquet

    GESTÃO DE RISCO ATUAL:
        "Stop Loss  = 1.5 × VR_pips  (R:R assimétrico intencional)"
        "Take Profit = 4.0 × VR_pips (R:R = 1:2.67)"
================================================================================
"""

import sys
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)
matplotlib.use("Agg")

# =============================================================================
# CONFIGURAÇÃO DE LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# CAMINHOS E PARÂMETROS
# =============================================================================
DIR_MODULO   = Path(__file__).resolve().parent
DIR_DATA     = DIR_MODULO / "data"
DIR_GRAFICOS = DIR_MODULO / "graficos"

PARQUET_ENTRADA = DIR_DATA / "eurusd_h1_hurst.parquet"      # Série 24h completa
PARQUET_SAIDA   = DIR_DATA / "eurusd_h1_momentum.parquet"    # Novo Parquet de saída
GRAFICO_SAIDA   = DIR_GRAFICOS / "momentum_sinais.png"

# Parâmetros de cinemática
JANELA_PERCENTIL  = 100

# Parâmetros de Entropia de Shannon
JANELA_ENTROPIA   = 30
N_BINS_ENTROPIA   = 10
H_MAX_TEORICO     = np.log2(N_BINS_ENTROPIA)

# Limiares de sinal
PCT_ACEL_COMPRA     = 0.70
PCT_ACEL_VENDA      = 0.30
ENTROPIA_MAX_OPERAR = 0.60
ENTROPIA_BLOQUEIO   = 0.80

# Parâmetros de gestão de risco
JANELA_VR          = 50
MULT_STOP_PIPS     = 1.5
MULT_TP_PIPS       = 4.0
FATOR_PIPS_EURUSD  = 10_000

# Paleta de cores do gráfico (dark mode)
COR_FUNDO       = "#0D1117"
COR_TEXTO       = "#E6EDF3"
COR_GRADE       = "#21262D"
COR_PRECO       = "#58A6FF"
COR_VELOCIDADE  = "#79C0FF"
COR_ACELERACAO  = "#FF7B72"
COR_ENTROPIA    = "#D2A8FF"
COR_COMPRA      = "#3FB950"
COR_VENDA       = "#F85149"
COR_REVERSAO    = "#F85149"
COR_TENDENCIA   = "#3FB950"
COR_INDEFINIDO  = "#6E7681"


# =============================================================================
# 1. CINEMÁTICA DO PREÇO
# =============================================================================

def calcular_velocidade(preco_close: pd.Series) -> pd.Series:
    """
    Calcula a velocidade do preço (primeira derivada discreta).
    """
    velocidade = preco_close.diff(1)
    logger.info(
        f"Velocidade calculada: media={velocidade.mean():.6f}, std={velocidade.std():.6f}"
    )
    return velocidade.astype("float32")


def calcular_aceleracao(velocidade: pd.Series) -> pd.Series:
    """
    Calcula a aceleração do preço (segunda derivada discreta).
    """
    aceleracao = velocidade.diff(1)
    logger.info(
        f"Aceleração calculada: media={aceleracao.mean():.6f}, std={aceleracao.std():.6f}"
    )
    return aceleracao.astype("float32")


def calcular_percentil_aceleracao(aceleracao: pd.Series, janela: int = JANELA_PERCENTIL) -> pd.Series:
    """
    Calcula o percentil rolling da aceleração.
    """
    logger.info(f"Calculando percentil rolling da aceleração (janela={janela})...")

    def _percentrank(arr: np.ndarray) -> float:
        val_atual = arr[-1]
        historico = arr[:-1]
        n_historico = len(historico)
        if n_historico == 0:
            return 0.5
        n_menores = np.sum(historico < val_atual)
        return float(n_menores) / float(n_historico)

    percentil = aceleracao.rolling(window=janela, min_periods=janela).apply(_percentrank, raw=True)
    return percentil.astype("float32")


# =============================================================================
# 2. ENTROPIA DE SHANNON
# =============================================================================

def calcular_entropia_shannon(
    log_retornos: pd.Series,
    janela: int = JANELA_ENTROPIA,
    n_bins: int = N_BINS_ENTROPIA,
) -> pd.Series:
    """
    Calcula a Entropia de Shannon normalizada dos log-retornos em janela móvel.
    """
    logger.info(f"Calculando Entropia de Shannon (janela={janela}, bins={n_bins})...")
    h_max = np.log2(n_bins)

    def _entropia_shannon(arr: np.ndarray) -> float:
        if np.std(arr) < 1e-15:
            return 0.0
        contagens, _ = np.histogram(arr, bins=n_bins)
        n = float(len(arr))
        probs = contagens / n
        probs_positivas = probs[probs > 0]
        h_shannon = -np.sum(probs_positivas * np.log2(probs_positivas))
        return float(np.clip(h_shannon / h_max, 0.0, 1.0))

    entropia = log_retornos.rolling(window=janela, min_periods=janela).apply(_entropia_shannon, raw=True)
    return entropia.astype("float32")


# =============================================================================
# 3. VOLATILIDADE REALIZADA
# =============================================================================

def calcular_volatilidade_realizada(
    log_retornos: pd.Series,
    preco_close: pd.Series,
    janela: int = JANELA_VR,
) -> tuple:
    """
    Calcula a Volatilidade Realizada e converte para pips (EURUSD).
    """
    vr = log_retornos.rolling(window=janela, min_periods=janela).std(ddof=1)
    vr_pips = vr * preco_close * FATOR_PIPS_EURUSD
    sl_pips = MULT_STOP_PIPS * vr_pips
    tp_pips = MULT_TP_PIPS   * vr_pips
    return (
        vr_pips.astype("float32"),
        sl_pips.astype("float32"),
        tp_pips.astype("float32"),
    )


# =============================================================================
# 4. GERAÇÃO DE SINAIS OPERACIONAIS COM FILTRO
# =============================================================================

def verificar_janela_operacional(dt_index: pd.DatetimeIndex) -> pd.Series:
    """
    Verifica se cada candle está dentro da janela operacional de trading:
    De domingo às 21h00 UTC até sexta-feira às 20h00 UTC.
    """
    weekday = dt_index.weekday
    hour = dt_index.hour
    
    domingo_operacional = (weekday == 6) & (hour >= 21)
    dias_uteis = (weekday >= 0) & (weekday <= 3) # Segunda (0) a Quinta (3)
    sexta_operacional = (weekday == 4) & (hour <= 20)
    
    esta_dentro = domingo_operacional | dias_uteis | sexta_operacional
    return pd.Series(esta_dentro, index=dt_index)


def gerar_sinais_momentum(
    velocidade:    pd.Series,
    percentil_acel: pd.Series,
    entropia:      pd.Series,
    regime:        pd.Series,
) -> pd.Series:
    """
    Gera sinais operacionais de Momentum condicionados ao regime Hurst = "TENDENCIA"
    e à janela operacional de trading.
    """
    logger.info("Gerando sinais de Momentum condicionados ao regime Hurst...")

    sinais = pd.Series(0, index=velocidade.index, dtype="int8")

    # Condições
    em_tendencia = (regime == "TENDENCIA")
    nao_ruidoso  = (entropia < ENTROPIA_MAX_OPERAR)
    nao_bloqueio = (entropia <= ENTROPIA_BLOQUEIO)
    pre_condicao = em_tendencia & nao_ruidoso & nao_bloqueio

    # Compra
    acel_alta  = (percentil_acel > PCT_ACEL_COMPRA)
    vel_positiva = (velocidade > 0)
    cond_compra = pre_condicao & acel_alta & vel_positiva

    # Venda
    acel_baixa   = (percentil_acel < PCT_ACEL_VENDA)
    vel_negativa = (velocidade < 0)
    cond_venda = pre_condicao & acel_baixa & vel_negativa

    # Filtro operacional de horário
    janela_operacional = verificar_janela_operacional(velocidade.index)

    # Aplicar apenas na janela operacional de trading
    sinais[cond_compra & janela_operacional] =  1
    sinais[cond_venda & janela_operacional]  = -1

    # Estatísticas
    sinais_brutos = (cond_compra | cond_venda).sum()
    sinais_finais = (sinais != 0).sum()
    sinais_filtrados = sinais_brutos - sinais_finais

    logger.info(
        f"Sinais Momentum brutos calculados   : {sinais_brutos} | "
        f"Sinais finais ativos (dentro janela): {sinais_finais} | "
        f"Sinais bloqueados pelo filtro hor. : {sinais_filtrados}"
    )

    return sinais


# =============================================================================
# 5. GERAÇÃO DO GRÁFICO
# =============================================================================

def gerar_grafico_momentum(df: pd.DataFrame, caminho: Path) -> None:
    """
    Gera gráfico de 4 painéis para a série de Momentum completa.
    """
    logger.info("Gerando gráfico de 4 painéis Momentum...")

    plt.rcParams.update({
        "figure.facecolor":  COR_FUNDO,
        "axes.facecolor":    COR_FUNDO,
        "axes.edgecolor":    COR_GRADE,
        "axes.labelcolor":   COR_TEXTO,
        "xtick.color":       COR_TEXTO,
        "ytick.color":       COR_TEXTO,
        "text.color":        COR_TEXTO,
        "grid.color":        COR_GRADE,
        "font.family":       "monospace",
    })

    fig = plt.figure(figsize=(22, 16))
    gs  = gridspec.GridSpec(4, 1, height_ratios=[2.5, 1.5, 1.5, 0.8], hspace=0.06)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax4 = fig.add_subplot(gs[3], sharex=ax1)

    fig.suptitle(
        "EURUSD H1 — Estratégia Momentum (Série Completa 24h)\n"
        f"Percentil Acel. Compra > {PCT_ACEL_COMPRA} | Venda < {PCT_ACEL_VENDA} | Entropia < {ENTROPIA_MAX_OPERAR}",
        color=COR_TEXTO, fontsize=13, fontweight="bold", y=0.99,
    )

    df_compra = df[df["sinal_momentum"] ==  1]
    df_venda  = df[df["sinal_momentum"] == -1]
    df_valido = df.dropna(subset=["percentil_acel", "entropia_shannon"])

    # Painel 1
    ax1.plot(df.index, df["Close"], color=COR_PRECO, linewidth=0.7, alpha=0.9, label="EURUSD Close")
    ax1.scatter(df_compra.index, df_compra["Close"], marker="^", color=COR_COMPRA, s=45, zorder=5, label="Compra")
    ax1.scatter(df_venda.index, df_venda["Close"], marker="v", color=COR_VENDA, s=45, zorder=5, label="Venda")
    ax1.set_ylabel("Preço (EURUSD)", color=COR_TEXTO, fontsize=10)
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.25)

    # Painel 2
    ax2.plot(df_valido.index, df_valido["velocidade"], color=COR_VELOCIDADE, linewidth=0.7, alpha=0.85, label="Velocidade")
    ax2.plot(df_valido.index, df_valido["aceleracao"], color=COR_ACELERACAO, linewidth=0.7, alpha=0.85, label="Aceleração")
    ax2.axhline(0, color=COR_TEXTO, linewidth=0.6, alpha=0.4, linestyle="--")
    ax2.set_ylabel("Cinemática do Preço", color=COR_TEXTO, fontsize=10)
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle="--", alpha=0.25)

    # Painel 3
    ax3.fill_between(df_valido.index, df_valido["entropia_shannon"], ENTROPIA_BLOQUEIO,
                     where=(df_valido["entropia_shannon"] > ENTROPIA_BLOQUEIO), color=COR_INDEFINIDO, alpha=0.30)
    ax3.fill_between(df_valido.index, df_valido["entropia_shannon"], 0,
                     where=(df_valido["entropia_shannon"] < ENTROPIA_MAX_OPERAR), color=COR_COMPRA, alpha=0.08)
    ax3.plot(df_valido.index, df_valido["entropia_shannon"], color=COR_ENTROPIA, linewidth=0.8, alpha=0.9, label="Entropia Shannon")
    ax3.axhline(ENTROPIA_BLOQUEIO, color=COR_VENDA, linestyle="--", linewidth=1.3)
    ax3.axhline(ENTROPIA_MAX_OPERAR, color="#E3B341", linestyle="--", linewidth=1.0)
    ax3.set_ylabel("Entropia Shannon Norm.", color=COR_TEXTO, fontsize=10)
    ax3.set_ylim(-0.02, 1.05)
    ax3.legend(loc="upper left")
    ax3.grid(True, linestyle="--", alpha=0.25)

    # Painel 4
    mapa_cor_regime = {"REVERSAO": COR_REVERSAO, "TENDENCIA": COR_TENDENCIA, "INDEFINIDO": COR_INDEFINIDO}
    for regime_nome, cor in mapa_cor_regime.items():
        mask = df["regime"] == regime_nome
        if mask.any():
            ax4.bar(df.index[mask], [1] * mask.sum(), color=cor, alpha=0.6, width=0.04, label=regime_nome)
    ax4.set_ylabel("Regime", color=COR_TEXTO, fontsize=9)
    ax4.set_xlabel("Data", color=COR_TEXTO, fontsize=10)
    ax4.set_yticks([])
    ax4.legend(loc="upper left", ncol=3)
    
    fig.autofmt_xdate(rotation=30, ha="right")
    plt.savefig(caminho, dpi=150, bbox_inches="tight", facecolor=COR_FUNDO)
    plt.close()


# =============================================================================
# 6. FUNÇÃO PRINCIPAL
# =============================================================================

def calcular_e_salvar_momentum(
    data_inicio: str = None,
    data_fim:    str = None,
    forcar_reprocessamento: bool = False,
) -> pd.DataFrame:
    """
    Pipeline principal para cálculo e salvamento da estratégia Momentum.
    """
    DIR_GRAFICOS.mkdir(parents=True, exist_ok=True)
    DIR_DATA.mkdir(parents=True, exist_ok=True)

    if PARQUET_SAIDA.exists() and not forcar_reprocessamento:
        logger.info(f"Parquet Momentum já existe: {PARQUET_SAIDA.name}. Carregando cache...")
        return pd.read_parquet(PARQUET_SAIDA)

    if not PARQUET_ENTRADA.exists():
        raise FileNotFoundError(f"Parquet de entrada não encontrado: {PARQUET_ENTRADA.name}")

    logger.info(f"Carregando dados H1 com Hurst: {PARQUET_ENTRADA.name}")
    df = pd.read_parquet(PARQUET_ENTRADA)

    if data_inicio:
        df = df[df.index >= data_inicio]
    if data_fim:
        df = df[df.index <= data_fim]

    # Cinemática
    df["velocidade"] = calcular_velocidade(df["Close"])
    df["aceleracao"] = calcular_aceleracao(df["velocidade"])
    df["percentil_acel"] = calcular_percentil_aceleracao(df["aceleracao"])

    # Entropia
    df["entropia_shannon"] = calcular_entropia_shannon(df["log_return"])

    # Volatilidade e stops
    df["vr_pips"], df["sl_pips"], df["tp_pips"] = calcular_volatilidade_realizada(df["log_return"], df["Close"])

    # Sinais
    df["sinal_momentum"] = gerar_sinais_momentum(
        velocidade=df["velocidade"],
        percentil_acel=df["percentil_acel"],
        entropia=df["entropia_shannon"],
        regime=df["regime"]
    )

    # Salvar parquet
    df.to_parquet(PARQUET_SAIDA, engine="pyarrow", compression="snappy", index=True)
    logger.info(f"Parquet de Momentum salvo com sucesso em: {PARQUET_SAIDA.name}")

    # Gerar gráfico
    gerar_grafico_momentum(df, GRAFICO_SAIDA)

    # Exibir resumo
    _exibir_resumo_momentum(df)

    return df


def _exibir_resumo_momentum(df: pd.DataFrame) -> None:
    """
    Imprime um resumo estatístico das métricas calculadas para a Etapa 4.
    """
    df_v = df.dropna(subset=["percentil_acel", "entropia_shannon"])
    n_total = len(df_v)
    n_compra = (df_v["sinal_momentum"] == 1).sum()
    n_venda  = (df_v["sinal_momentum"] == -1).sum()
    n_sinais = n_compra + n_venda

    print("\n" + "=" * 70)
    print("  ETAPA 4 CONCLUÍDA — MÓDULO MOMENTUM (SÉRIE COMPLETA)")
    print("=" * 70)
    print(f"  Candles analisados (24h)           : {n_total:>10,}")
    print(f"  Período                            : {df_v.index.min().date()} → {df_v.index.max().date()}")
    print(f"")
    print(f"  ── Estatísticas dos Sinais Operacionais (Percentis + Janela Trading) ──")
    print(f"  Sinais de COMPRA (LONG)            : {n_compra:>10,} ({n_compra/n_total*100:.3f}%)")
    print(f"  Sinais de VENDA (SHORT)            : {n_venda:>10,} ({n_venda/n_total*100:.3f}%)")
    print(f"  TOTAL de Sinais Ativos             : {n_sinais:>10,} ({n_sinais/n_total*100:.3f}%)")
    print(f"")
    print(f"  ── Entropia de Shannon (Desordem dos retornos) ──")
    print(f"  Entropia Média                     : {df_v['entropia_shannon'].mean():>10.4f}")
    print(f"  Sessões bloqueadas por ruído (>0.8): {(df_v['entropia_shannon'] > ENTROPIA_BLOQUEIO).sum():>10,} candles")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  QUANT EURUSD — ETAPA 4: MOMENTUM & ENTROPIA (24H)")
    print("  Trend Following com Cinemática e Shannon sobre Série Completa")
    print("=" * 60)

    calcular_e_salvar_momentum(forcar_reprocessamento=True)
