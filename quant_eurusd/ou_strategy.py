# -*- coding: utf-8 -*-
"""
================================================================================
ou_strategy.py — Estratégia de Reversão à Média via Processo Ornstein-Uhlenbeck
================================================================================
Autor: Quant Developer Sênior
Data: 2026

Descrição:
    Este módulo implementa uma estratégia de Mean Reversion baseada no processo
    de Ornstein-Uhlenbeck (OU), que modela a dinâmica de preços com a equação
    diferencial estocástica:

        dX_t = θ(μ - X_t)dt + σdW_t

    Onde:
        X_t  = log-preço no tempo t
        θ    = velocidade de reversão à média (mean reversion speed)
        μ    = nível de equilíbrio (média de longo prazo)
        σ    = volatilidade do processo
        dW_t = incremento do movimento Browniano padrão

    A discretização para dados H1 resulta em:

        X_t = X_{t-1} * exp(-θΔt) + μ(1 - exp(-θΔt)) + ε_t

    A estimação dos parâmetros é feita via OLS (scipy.stats.linregress) em
    janela móvel de N candles H1 sobre os log-preços.

    MÓDULO 100% INDEPENDENTE: não depende de zscore.py, momentum.py ou
    hurst.py. Carrega diretamente o eurusd_h1_clean.parquet.

Fluxo de processamento:
    1. Carregar dados H1 limpos (eurusd_h1_clean.parquet)
    2. Calcular log-preço = ln(Close)
    3. Calcular VR para dimensionamento de stops
    4. Estimar parâmetros OU em janela móvel (θ, μ, σ, σ_eq, half-life)
    5. Calcular Z-Score do processo OU
    6. Gerar sinais condicionados à validade do processo OU
    7. Salvar DataFrame em Parquet
    8. Gerar gráfico de 4 painéis (dark mode)
    9. Exibir resumo estatístico
================================================================================
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
from scipy import stats
from tqdm import tqdm

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

# Entrada: dados H1 limpos (sem dependência de Hurst, ZScore ou Momentum)
PARQUET_ENTRADA = DIR_DATA / "eurusd_h1_clean.parquet"
PARQUET_SAIDA   = DIR_DATA / "eurusd_h1_ou.parquet"
GRAFICO_SAIDA   = DIR_GRAFICOS / "ou_sinais.png"

# Parâmetros do processo OU
JANELA_OU          = 100   # Candles H1 para estimação dos parâmetros OU
DELTA_T            = 1.0   # Intervalo temporal (1 candle H1)

# Limiares de validação dos parâmetros OU
P_VALOR_MAX        = 0.05  # Significância mínima da regressão OLS
HALFLIFE_MIN       = 2     # Half-life mínimo (candles H1) — não é ruído puro
HALFLIFE_MAX       = 50    # Half-life máximo (candles H1) — reversão em até ~2 dias

# Limiares do Z-Score OU para geração de sinais
Z_OU_ENTRADA_COMPRA = -2.0  # Z_ou <= -2.0 → sinal de LONG
Z_OU_ENTRADA_VENDA  =  2.0  # Z_ou >= +2.0 → sinal de SHORT
Z_OU_SAIDA_MIN      = -0.3  # Fechar posição quando Z_ou retornar à zona neutra
Z_OU_SAIDA_MAX      =  0.3  #   (mais agressivo que o Z-Score de preço)

# Parâmetros de gestão de risco (VR)
JANELA_VR          = 50     # Candles H1 para cálculo de Volatilidade Realizada
MULT_STOP_PIPS     = 2.0    # SL = 2.0 * VR_pips
MULT_TP_PIPS       = 3.0    # TP = 3.0 * VR_pips (R:R de 1:1.5)
FATOR_PIPS_EURUSD  = 10_000 # 1 pip EURUSD = 0.0001

# Paleta de cores do gráfico (dark mode — mesmo padrão dos outros módulos)
COR_FUNDO       = "#0D1117"
COR_TEXTO       = "#E6EDF3"
COR_GRADE       = "#21262D"
COR_PRECO       = "#58A6FF"
COR_ZSCORE_OU   = "#D2A8FF"  # Lilás para Z-Score OU
COR_HALFLIFE    = "#79C0FF"  # Azul claro para Half-life
COR_THETA       = "#FFA657"  # Laranja para Theta
COR_COMPRA      = "#3FB950"  # Verde para sinais de compra
COR_VENDA       = "#F85149"  # Vermelho para sinais de venda
COR_INVALIDO    = "#6E7681"  # Cinza para regiões sem OU válido
COR_ZONA_VALIDA = "#1C3D2E"  # Fundo verde escuro para zona half-life válida


# =============================================================================
# 1. ESTIMAÇÃO DOS PARÂMETROS ORNSTEIN-UHLENBECK
# =============================================================================

def estimar_parametros_ou(
    serie: pd.Series,
    janela: int = JANELA_OU,
) -> dict:
    """
    Estima os parâmetros do processo Ornstein-Uhlenbeck via regressão OLS
    sobre uma janela de log-preços.

    A versão discreta do OU é:
        X_t = a + b * X_{t-1} + ε_t

    Onde:
        b = exp(-θ * Δt)
        a = μ * (1 - exp(-θ * Δt)) = μ * (1 - b)

    Portanto, recuperamos os parâmetros contínuos:
        θ     = -ln(b) / Δt                    (velocidade de reversão)
        μ     = a / (1 - b)                     (nível de equilíbrio)
        σ     = σ_resid * sqrt(2θ / (1-e^{-2θΔt}))  (volatilidade do processo)
        σ_eq  = σ / sqrt(2θ)                    (desvio padrão estacionário)
        t½    = ln(2) / θ                       (half-life em candles)

    Parâmetros:
        serie:  pd.Series ou np.ndarray de log-preços com tamanho >= janela
        janela: int — tamanho da janela de estimação (padrão: 100)

    Retorna:
        dict com chaves:
            theta, mu, sigma, sigma_eq, half_life, p_valor, r_squared
        Retorna dict com NaN se os parâmetros falharem na validação.
    """
    # Resultado inválido padrão
    resultado_invalido = {
        "theta":     np.nan,
        "mu":        np.nan,
        "sigma":     np.nan,
        "sigma_eq":  np.nan,
        "half_life": np.nan,
        "p_valor":   np.nan,
        "r_squared": np.nan,
    }

    try:
        # Converter para array numpy se necessário
        if isinstance(serie, pd.Series):
            dados = serie.values[-janela:]
        else:
            dados = np.asarray(serie)[-janela:]

        if len(dados) < janela:
            return resultado_invalido

        # Passo 1 — Definir variáveis de regressão
        #   Y = X_t     (log-preço atual)
        #   X = X_{t-1}  (log-preço anterior)
        Y = dados[1:]    # X_t
        X = dados[:-1]   # X_{t-1}

        # Passo 2 — Regressão linear: Y = a + b*X + ε via OLS
        resultado_ols = stats.linregress(X, Y)
        b_hat    = resultado_ols.slope
        a_hat    = resultado_ols.intercept
        p_valor  = resultado_ols.pvalue
        r_sq     = resultado_ols.rvalue ** 2
        residuos = Y - (a_hat + b_hat * X)

        # Passo 3 — Recuperar parâmetros OU
        # Validação de b_hat: deve estar estritamente entre 0 e 1
        if b_hat <= 0 or b_hat >= 1:
            return resultado_invalido

        theta = -np.log(b_hat) / DELTA_T
        mu    = a_hat / (1.0 - b_hat)

        sigma_resid = np.std(residuos, ddof=1)
        denominador = 1.0 - np.exp(-2.0 * theta * DELTA_T)

        if denominador <= 0 or theta <= 0:
            return resultado_invalido

        sigma = sigma_resid * np.sqrt(2.0 * theta / denominador)

        # Passo 4 — Desvio padrão de equilíbrio (distribuição estacionária)
        sigma_eq = sigma / np.sqrt(2.0 * theta)

        # Half-life: tempo esperado para reverter metade do desvio
        half_life = np.log(2.0) / theta

        # Passo 5 — Validação dos parâmetros
        if theta <= 0:
            return resultado_invalido
        if p_valor > P_VALOR_MAX:
            return resultado_invalido
        if half_life > janela / 2.0:
            return resultado_invalido

        return {
            "theta":     float(theta),
            "mu":        float(mu),
            "sigma":     float(sigma),
            "sigma_eq":  float(sigma_eq),
            "half_life": float(half_life),
            "p_valor":   float(p_valor),
            "r_squared": float(r_sq),
        }

    except Exception:
        return resultado_invalido


# =============================================================================
# 2. CÁLCULO DO Z-SCORE DO PROCESSO OU
# =============================================================================

def calcular_zscore_ou(
    log_preco_atual: float,
    mu: float,
    sigma_eq: float,
) -> float:
    """
    Calcula o Z-Score do processo Ornstein-Uhlenbeck.

    Fórmula:
        Z_ou = (X_t - μ) / σ_eq

    Este Z-Score é matematicamente superior ao Z-Score de preço porque:
        1. μ é estimado dinamicamente pelo próprio processo OU
        2. σ_eq representa a dispersão real da distribuição estacionária
        3. Só é calculado quando o processo é comprovadamente mean-reverting
           (θ > 0, p-valor < 0.05, half-life dentro do intervalo válido)

    Parâmetros:
        log_preco_atual: float — log-preço atual (ln(Close))
        mu:              float — nível de equilíbrio estimado pelo OU
        sigma_eq:        float — desvio padrão estacionário do OU

    Retorna:
        float — Z-Score OU (pode ser NaN se sigma_eq for inválido)
    """
    if np.isnan(mu) or np.isnan(sigma_eq) or sigma_eq <= 0:
        return np.nan

    return (log_preco_atual - mu) / sigma_eq


# =============================================================================
# 3. PIPELINE ROLLING COMPLETO
# =============================================================================

def calcular_ou_rolling(
    df: pd.DataFrame,
    janela: int = JANELA_OU,
) -> pd.DataFrame:
    """
    Calcula os parâmetros Ornstein-Uhlenbeck em janela móvel para todo o DataFrame.

    Para cada candle t (a partir do candle janela):
        1. Pegar janela de N log-preços anteriores
        2. Estimar parâmetros OU via OLS
        3. Se válidos, calcular Z_ou do candle atual
        4. Armazenar θ, μ, σ_eq, half_life, Z_ou

    Colunas adicionadas ao DataFrame:
        - log_preco    : ln(Close)
        - ou_theta     : velocidade de reversão (θ)
        - ou_mu        : nível de equilíbrio (μ)
        - ou_sigma_eq  : desvio padrão estacionário (σ_eq)
        - ou_halflife  : half-life em candles
        - ou_zscore    : Z-Score do processo OU
        - ou_valido    : bool — True se parâmetros passaram validação

    Parâmetros:
        df:     pd.DataFrame com coluna 'log_preco'
        janela: int — tamanho da janela de estimação (padrão: 100)

    Retorna:
        pd.DataFrame com colunas OU adicionadas
    """
    logger.info(
        f"Iniciando cálculo rolling do Ornstein-Uhlenbeck "
        f"(janela={janela} candles H1)..."
    )

    n = len(df)
    log_precos = df["log_preco"].values.astype(np.float64)

    # Pré-alocar arrays de saída
    theta_arr     = np.full(n, np.nan, dtype=np.float64)
    mu_arr        = np.full(n, np.nan, dtype=np.float64)
    sigma_eq_arr  = np.full(n, np.nan, dtype=np.float64)
    halflife_arr  = np.full(n, np.nan, dtype=np.float64)
    zscore_arr    = np.full(n, np.nan, dtype=np.float64)
    valido_arr    = np.zeros(n, dtype=bool)

    # Iterar sobre cada candle a partir do candle 'janela'
    for t in tqdm(range(janela, n), desc="OU Rolling", unit="candle"):
        # Janela de log-preços: [t-janela, t) — ou seja, os N candles anteriores
        janela_dados = log_precos[t - janela:t]

        # Estimar parâmetros OU
        params = estimar_parametros_ou(janela_dados, janela)

        theta = params["theta"]
        mu    = params["mu"]
        s_eq  = params["sigma_eq"]
        hl    = params["half_life"]

        # Verificar se a estimação foi válida (não NaN)
        if not np.isnan(theta):
            theta_arr[t]    = theta
            mu_arr[t]       = mu
            sigma_eq_arr[t] = s_eq
            halflife_arr[t] = hl

            # Z-Score OU do candle atual
            zscore_arr[t] = calcular_zscore_ou(log_precos[t], mu, s_eq)

            # Validação completa para sinais
            valido_arr[t] = (
                hl >= HALFLIFE_MIN and
                hl <= HALFLIFE_MAX
            )

    # Adicionar colunas ao DataFrame
    df["ou_theta"]    = theta_arr.astype("float32")
    df["ou_mu"]       = mu_arr.astype("float32")
    df["ou_sigma_eq"] = sigma_eq_arr.astype("float32")
    df["ou_halflife"]  = halflife_arr.astype("float32")
    df["ou_zscore"]   = zscore_arr.astype("float32")
    df["ou_valido"]   = valido_arr

    # Estatísticas de validação
    n_valido   = valido_arr.sum()
    n_estimado = (~np.isnan(theta_arr)).sum()
    n_total    = n - janela

    logger.info(
        f"OU Rolling concluído: "
        f"estimações válidas={n_valido:,}/{n_total:,} ({n_valido/n_total*100:.1f}%), "
        f"estimações com θ>0={n_estimado:,}"
    )

    return df


# =============================================================================
# 4. GERAÇÃO DE SINAIS
# =============================================================================

def verificar_janela_operacional(dt_index: pd.DatetimeIndex) -> pd.Series:
    """
    Verifica se cada candle está dentro da janela operacional de trading:
    De domingo às 21h00 UTC até sexta-feira às 20h00 UTC.

    Parâmetros:
        dt_index: pd.DatetimeIndex — índice temporal do DataFrame

    Retorna:
        pd.Series de bool — True se o candle está dentro da janela operacional
    """
    weekday = dt_index.weekday
    hour = dt_index.hour

    domingo_operacional = (weekday == 6) & (hour >= 21)
    dias_uteis = (weekday >= 0) & (weekday <= 3)   # Segunda (0) a Quinta (3)
    sexta_operacional = (weekday == 4) & (hour <= 20)

    esta_dentro = domingo_operacional | dias_uteis | sexta_operacional
    return pd.Series(esta_dentro, index=dt_index)


def gerar_sinais_ou(df: pd.DataFrame) -> pd.Series:
    """
    Gera sinais de entrada da estratégia Ornstein-Uhlenbeck.

    PRÉ-CONDIÇÃO OBRIGATÓRIA:
        - ou_valido == True (processo comprovadamente mean-reverting)
        - half_life entre 2 e 50 candles H1

    SINAL DE COMPRA (LONG, +1):
        Z_ou <= -2.0
        Interpretação: preço está 2 desvios abaixo do equilíbrio estimado
        pelo OU → espera-se reversão para cima em direção a μ.

    SINAL DE VENDA (SHORT, -1):
        Z_ou >= +2.0
        Interpretação: preço está 2 desvios acima do equilíbrio estimado
        pelo OU → espera-se reversão para baixo em direção a μ.

    Filtro de horário operacional:
        Mesma janela do momentum.py:
        - Domingo 21h UTC até sexta 20h UTC
        - Sem entradas após sexta 20h UTC

    Parâmetros:
        df: pd.DataFrame com colunas 'ou_zscore', 'ou_valido', 'ou_halflife'

    Retorna:
        pd.Series de int8 com valores {-1, 0, 1}
    """
    logger.info("Gerando sinais Ornstein-Uhlenbeck...")

    sinais = pd.Series(0, index=df.index, dtype="int8")

    # Pré-condições obrigatórias
    valido = df["ou_valido"]
    halflife_ok = (df["ou_halflife"] >= HALFLIFE_MIN) & (df["ou_halflife"] <= HALFLIFE_MAX)
    pre_condicao = valido & halflife_ok

    # Sinais brutos
    sinal_compra = (df["ou_zscore"] <= Z_OU_ENTRADA_COMPRA)
    sinal_venda  = (df["ou_zscore"] >= Z_OU_ENTRADA_VENDA)

    # Filtro operacional de horário
    janela_operacional = verificar_janela_operacional(df.index)

    # Aplicar condições + filtro operacional
    sinais[pre_condicao & sinal_compra & janela_operacional] =  1
    sinais[pre_condicao & sinal_venda  & janela_operacional] = -1

    # Estatísticas
    n_compra = (sinais == 1).sum()
    n_venda  = (sinais == -1).sum()
    n_total  = len(sinais)

    logger.info(
        f"Sinais OU gerados: "
        f"COMPRA={n_compra:,} ({n_compra/n_total*100:.3f}%), "
        f"VENDA={n_venda:,} ({n_venda/n_total*100:.3f}%), "
        f"NEUTRO={n_total - n_compra - n_venda:,}"
    )

    return sinais


# =============================================================================
# 5. VOLATILIDADE REALIZADA (VR) — INDEPENDENTE
# =============================================================================

def calcular_volatilidade_realizada_ou(
    log_retornos: pd.Series,
    preco_close: pd.Series,
    janela: int = JANELA_VR,
) -> tuple:
    """
    Calcula a Volatilidade Realizada (VR) dos log-retornos e converte para pips.

    Fórmulas:
        VR       = std(log_retornos_{t-N+1..t})    (desvio padrão rolling)
        VR_pips  = VR * Close_t * 10.000            (conversão para pips EURUSD)
        Stop Loss   (pips) = 2.0 * VR_pips
        Take Profit (pips) = 3.0 * VR_pips

    Parâmetros:
        log_retornos: pd.Series — log-retornos H1
        preco_close:  pd.Series — preço de fechamento H1
        janela: int — janela para cálculo da VR (padrão: 50)

    Retorna:
        tuple (vr_pips, sl_pips, tp_pips) — três pd.Series com float32
    """
    logger.info(
        f"Calculando Volatilidade Realizada OU (janela={janela} candles H1)..."
    )

    # VR = desvio padrão dos log-retornos na janela
    vr = log_retornos.rolling(window=janela, min_periods=janela).std(ddof=1)

    # Converter para pips: VR * Close * 10.000
    vr_pips = vr * preco_close * FATOR_PIPS_EURUSD

    # Stop Loss e Take Profit
    sl_pips = MULT_STOP_PIPS * vr_pips
    tp_pips = MULT_TP_PIPS   * vr_pips

    return (
        vr_pips.astype("float32"),
        sl_pips.astype("float32"),
        tp_pips.astype("float32"),
    )


# =============================================================================
# 6. GERAÇÃO DO GRÁFICO (4 PAINÉIS — DARK MODE)
# =============================================================================

def gerar_grafico_ou(df: pd.DataFrame, caminho: Path) -> None:
    """
    Gera gráfico profissional de 4 painéis para a estratégia Ornstein-Uhlenbeck.

    Painel 1 (topo): Preço Close com marcadores de entrada
        - Triângulo verde (▲) = COMPRA (preço abaixo de μ)
        - Triângulo vermelho (▼) = VENDA (preço acima de μ)

    Painel 2: Z-Score OU ao longo do tempo
        - Linha tracejada em ±2.0 (entrada)
        - Linha tracejada em ±0.3 (alvo de saída)
        - Zona cinza quando ou_valido == False

    Painel 3: Half-life em candles
        - Linha tracejada em 2 (mínimo)
        - Linha tracejada em 50 (máximo permitido)
        - Zona verde: half-life válido (2-50 candles)

    Painel 4: Theta (velocidade de reversão)
        - Quanto maior θ, mais forte a força de reversão

    Parâmetros:
        df: pd.DataFrame com todas as colunas OU calculadas
        caminho: Path — destino do arquivo PNG
    """
    logger.info("Gerando gráfico de 4 painéis Ornstein-Uhlenbeck...")

    plt.rcParams.update({
        "figure.facecolor":  COR_FUNDO,
        "axes.facecolor":    COR_FUNDO,
        "axes.edgecolor":    COR_GRADE,
        "axes.labelcolor":   COR_TEXTO,
        "xtick.color":       COR_TEXTO,
        "ytick.color":       COR_TEXTO,
        "text.color":        COR_TEXTO,
        "grid.color":        COR_GRADE,
        "grid.alpha":        0.5,
        "font.family":       "monospace",
    })

    fig = plt.figure(figsize=(22, 18))
    gs = gridspec.GridSpec(4, 1, height_ratios=[2.5, 2.0, 1.2, 1.2], hspace=0.06)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax4 = fig.add_subplot(gs[3], sharex=ax1)

    fig.suptitle(
        "EURUSD H1 — Estratégia Mean Reversion via Ornstein-Uhlenbeck\n"
        f"Janela: {JANELA_OU} candles | Entrada: |Z_ou| ≥ {abs(Z_OU_ENTRADA_VENDA):.1f} | "
        f"Half-life: {HALFLIFE_MIN}–{HALFLIFE_MAX} candles",
        color=COR_TEXTO, fontsize=13, fontweight="bold", y=0.99,
    )

    # Subconjuntos de sinais para plotagem
    df_compra = df[df["sinal_ou"] ==  1]
    df_venda  = df[df["sinal_ou"] == -1]
    df_valido = df.dropna(subset=["ou_zscore"])

    # ── Painel 1: Preço + Marcadores de Entrada ──────────────────────────────
    ax1.plot(
        df.index, df["Close"],
        color=COR_PRECO, linewidth=0.7, alpha=0.9, label="EURUSD Close",
    )
    ax1.scatter(
        df_compra.index, df_compra["Close"],
        marker="^", color=COR_COMPRA, s=45, zorder=5, alpha=0.85,
        label=f"Compra (Z_ou ≤ {Z_OU_ENTRADA_COMPRA}) [{len(df_compra):,}]",
    )
    ax1.scatter(
        df_venda.index, df_venda["Close"],
        marker="v", color=COR_VENDA, s=45, zorder=5, alpha=0.85,
        label=f"Venda (Z_ou ≥ +{Z_OU_ENTRADA_VENDA}) [{len(df_venda):,}]",
    )
    ax1.set_ylabel("Preço (EURUSD)", color=COR_TEXTO, fontsize=10)
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.3)
    ax1.grid(True, linestyle="--", alpha=0.3)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # ── Painel 2: Z-Score OU ─────────────────────────────────────────────────
    if len(df_valido) > 0:
        idx_v = df_valido.index
        zs    = df_valido["ou_zscore"].values

        # Zona cinza para regiões inválidas (OU não confirmado)
        invalido_mask = ~df["ou_valido"]
        if invalido_mask.any():
            # Pintar fundo cinza onde OU não é válido
            ax2.fill_between(
                df.index, -10, 10,
                where=invalido_mask.values,
                color=COR_INVALIDO, alpha=0.08, label="OU inválido",
            )

        # Zonas coloridas de entrada
        ax2.fill_between(
            idx_v, zs, Z_OU_ENTRADA_VENDA,
            where=(zs >= Z_OU_ENTRADA_VENDA),
            color=COR_VENDA, alpha=0.15, label=f"Sobrecomprado (Z ≥ +{Z_OU_ENTRADA_VENDA})",
        )
        ax2.fill_between(
            idx_v, zs, Z_OU_ENTRADA_COMPRA,
            where=(zs <= Z_OU_ENTRADA_COMPRA),
            color=COR_COMPRA, alpha=0.15, label=f"Sobrevendido (Z ≤ {Z_OU_ENTRADA_COMPRA})",
        )

        # Linha do Z-Score OU
        ax2.plot(
            idx_v, zs, color=COR_ZSCORE_OU, linewidth=0.7, alpha=0.9,
            label="Z-Score OU",
        )

        # Linhas de referência
        for nivel, cor, ls, lw in [
            (Z_OU_ENTRADA_VENDA,  COR_VENDA,    "--", 1.2),
            (Z_OU_ENTRADA_COMPRA, COR_COMPRA,   "--", 1.2),
            (Z_OU_SAIDA_MAX,      COR_TEXTO,    ":",  0.8),
            (Z_OU_SAIDA_MIN,      COR_TEXTO,    ":",  0.8),
            (0.0,                 COR_TEXTO,    "-",  0.5),
        ]:
            ax2.axhline(nivel, color=cor, linestyle=ls, linewidth=lw, alpha=0.7)

        ax2.set_ylim(max(zs.min() - 0.5, -6), min(zs.max() + 0.5, 6))

    ax2.set_ylabel("Z-Score OU", color=COR_TEXTO, fontsize=10)
    ax2.legend(loc="upper left", fontsize=7, framealpha=0.3, ncol=3)
    ax2.grid(True, linestyle="--", alpha=0.3)
    plt.setp(ax2.get_xticklabels(), visible=False)

    # ── Painel 3: Half-life ──────────────────────────────────────────────────
    df_hl = df.dropna(subset=["ou_halflife"])
    if len(df_hl) > 0:
        hl_vals = df_hl["ou_halflife"].values

        # Zona verde: half-life no intervalo válido [2, 50]
        ax3.fill_between(
            df_hl.index, hl_vals, HALFLIFE_MIN,
            where=(hl_vals >= HALFLIFE_MIN) & (hl_vals <= HALFLIFE_MAX),
            color=COR_COMPRA, alpha=0.10,
        )
        ax3.plot(
            df_hl.index, hl_vals,
            color=COR_HALFLIFE, linewidth=0.6, alpha=0.85, label="Half-life (candles H1)",
        )

    # Linhas de referência do half-life
    ax3.axhline(HALFLIFE_MIN, color=COR_COMPRA, linestyle="--", linewidth=1.0, alpha=0.8)
    ax3.axhline(HALFLIFE_MAX, color=COR_VENDA, linestyle="--", linewidth=1.0, alpha=0.8)
    ax3.text(
        df.index[-1], HALFLIFE_MIN, f"  Mín: {HALFLIFE_MIN}",
        color=COR_COMPRA, fontsize=7, va="bottom",
    )
    ax3.text(
        df.index[-1], HALFLIFE_MAX, f"  Máx: {HALFLIFE_MAX}",
        color=COR_VENDA, fontsize=7, va="bottom",
    )
    ax3.set_ylabel("Half-life (candles)", color=COR_TEXTO, fontsize=10)
    ax3.set_ylim(0, min(100, df["ou_halflife"].quantile(0.99) * 1.2) if df["ou_halflife"].notna().any() else 100)
    ax3.legend(loc="upper left", fontsize=8, framealpha=0.3)
    ax3.grid(True, linestyle="--", alpha=0.3)
    plt.setp(ax3.get_xticklabels(), visible=False)

    # ── Painel 4: Theta (velocidade de reversão) ─────────────────────────────
    df_th = df.dropna(subset=["ou_theta"])
    if len(df_th) > 0:
        ax4.plot(
            df_th.index, df_th["ou_theta"],
            color=COR_THETA, linewidth=0.6, alpha=0.85, label="θ (velocidade de reversão)",
        )
        ax4.fill_between(
            df_th.index, df_th["ou_theta"], 0,
            color=COR_THETA, alpha=0.08,
        )

    ax4.axhline(0, color=COR_TEXTO, linewidth=0.5, alpha=0.4)
    ax4.set_ylabel("θ (Theta)", color=COR_TEXTO, fontsize=10)
    ax4.set_xlabel("Data", color=COR_TEXTO, fontsize=10)
    ax4.legend(loc="upper left", fontsize=8, framealpha=0.3)
    ax4.grid(True, linestyle="--", alpha=0.3)

    fig.autofmt_xdate(rotation=30, ha="right")

    plt.savefig(caminho, dpi=150, bbox_inches="tight", facecolor=COR_FUNDO)
    plt.close()

    tamanho_kb = caminho.stat().st_size / 1024
    logger.info(f"Gráfico OU salvo: {caminho} ({tamanho_kb:.0f} KB)")


# =============================================================================
# 7. FUNÇÃO PRINCIPAL
# =============================================================================

def calcular_e_salvar_ou(
    data_inicio: str = None,
    data_fim: str = None,
    forcar: bool = False,
) -> pd.DataFrame:
    """
    Pipeline completo do módulo Ornstein-Uhlenbeck.

    Sequência:
        1. Carregar eurusd_h1_clean.parquet
        2. Calcular log_preco = ln(Close)
        3. Calcular VR para stops
        4. Rodar calcular_ou_rolling()
        5. Gerar sinais
        6. Salvar em: data/eurusd_h1_ou.parquet
        7. Gerar gráfico
        8. Exibir resumo

    Parâmetros:
        data_inicio: str — filtro de início "YYYY-MM-DD" (opcional)
        data_fim:    str — filtro de fim "YYYY-MM-DD" (opcional)
        forcar:      bool — recalcular mesmo que Parquet exista

    Retorna:
        pd.DataFrame com colunas OU completas

    Levanta:
        FileNotFoundError — se eurusd_h1_clean.parquet não existir
    """
    DIR_GRAFICOS.mkdir(parents=True, exist_ok=True)
    DIR_DATA.mkdir(parents=True, exist_ok=True)

    # ── Cache ─────────────────────────────────────────────────────────────────
    if PARQUET_SAIDA.exists() and not forcar:
        logger.info(
            f"Parquet OU já existe. Carregando cache... "
            f"(use --forcar para recalcular)"
        )
        df = pd.read_parquet(PARQUET_SAIDA, engine="pyarrow")
        if data_inicio:
            df = df[df.index >= data_inicio]
        if data_fim:
            df = df[df.index <= data_fim]
        _exibir_resumo_ou(df)
        return df

    # ── Passo 1: Carregar dados H1 limpos ─────────────────────────────────────
    if not PARQUET_ENTRADA.exists():
        raise FileNotFoundError(
            f"\n{'='*60}\n"
            f"  ERRO: Parquet de entrada não encontrado!\n"
            f"  Esperado em: {PARQUET_ENTRADA}\n"
            f"\n"
            f"  Solução: Execute o data_loader.py primeiro:\n"
            f"    python data_loader.py\n"
            f"{'='*60}\n"
        )

    logger.info(f"Carregando dados H1 limpos: {PARQUET_ENTRADA.name}")
    df = pd.read_parquet(PARQUET_ENTRADA, engine="pyarrow")
    logger.info(f"Dados carregados: {len(df):,} candles H1")

    if data_inicio:
        df = df[df.index >= data_inicio]
    if data_fim:
        df = df[df.index <= data_fim]

    # ── Passo 2: Calcular log-preço ───────────────────────────────────────────
    logger.info("Calculando log-preço = ln(Close)...")
    df["log_preco"] = np.log(df["Close"].astype(np.float64)).astype("float32")

    # ── Passo 3: Calcular VR para stops ───────────────────────────────────────
    df["vr_pips"], df["sl_pips"], df["tp_pips"] = calcular_volatilidade_realizada_ou(
        df["log_return"], df["Close"], janela=JANELA_VR,
    )

    # ── Passo 4: Rodar pipeline rolling OU ────────────────────────────────────
    df = calcular_ou_rolling(df, janela=JANELA_OU)

    # ── Passo 5: Gerar sinais ─────────────────────────────────────────────────
    df["sinal_ou"] = gerar_sinais_ou(df)

    # ── Passo 6: Salvar Parquet ───────────────────────────────────────────────
    df.to_parquet(PARQUET_SAIDA, engine="pyarrow", compression="snappy", index=True)
    tamanho_mb = PARQUET_SAIDA.stat().st_size / 1024 / 1024
    logger.info(f"Parquet OU salvo: {PARQUET_SAIDA.name} ({tamanho_mb:.1f} MB)")

    # ── Passo 7: Gerar gráfico ────────────────────────────────────────────────
    gerar_grafico_ou(df, GRAFICO_SAIDA)

    # ── Passo 8: Exibir resumo ────────────────────────────────────────────────
    _exibir_resumo_ou(df)

    return df


# =============================================================================
# RESUMO ESTATÍSTICO
# =============================================================================

def _exibir_resumo_ou(df: pd.DataFrame) -> None:
    """
    Exibe no terminal um resumo completo da estratégia Ornstein-Uhlenbeck.

    Inclui:
        - Total de candles com OU válido
        - % do tempo com processo mean-reverting confirmado
        - Half-life médio quando válido
        - Total de sinais gerados (compra/venda)
        - Z_ou mínimo e máximo histórico
        - SL médio e TP médio em pips

    Parâmetros:
        df: pd.DataFrame com colunas OU completas
    """
    df_v = df.dropna(subset=["ou_zscore"])
    n_total   = len(df_v)
    n_valido  = df_v["ou_valido"].sum() if "ou_valido" in df_v.columns else 0
    n_compra  = (df_v["sinal_ou"] == 1).sum() if "sinal_ou" in df_v.columns else 0
    n_venda   = (df_v["sinal_ou"] == -1).sum() if "sinal_ou" in df_v.columns else 0
    n_sinais  = n_compra + n_venda

    # Half-life médio nos candles válidos
    df_ou_valido = df_v[df_v["ou_valido"] == True] if "ou_valido" in df_v.columns else df_v
    hl_medio = df_ou_valido["ou_halflife"].mean() if len(df_ou_valido) > 0 else float("nan")

    # Stops médios
    sl_medio = df_v["sl_pips"].mean() if "sl_pips" in df_v.columns else float("nan")
    tp_medio = df_v["tp_pips"].mean() if "tp_pips" in df_v.columns else float("nan")

    print("\n" + "=" * 65)
    print("  RESUMO DA ESTRATEGIA ORNSTEIN-UHLENBECK -- EURUSD H1")
    print("=" * 65)
    print(f"  Periodo analisado: "
          f"{df_v.index.min().date()} -> {df_v.index.max().date()}")
    print(f"  Candles com Z-Score OU valido: {n_total:>8,}")
    print()
    print(f"  -- Processo Ornstein-Uhlenbeck --")
    print(f"  Candles com OU valido (theta>0, p<0.05, HL em [2,50]):")
    print(f"    Total                         : {n_valido:>8,} "
          f"({n_valido/n_total*100:.1f}% do tempo)")
    print(f"  Half-life medio (quando valido) : {hl_medio:>8.1f} candles H1")
    print()
    print(f"  -- Sinais Gerados (Z_ou + OU valido) --")
    print(f"  COMPRA  (Z_ou <= {Z_OU_ENTRADA_COMPRA})     : "
          f"{n_compra:>5,} sinais ({n_compra/n_total*100:.3f}%)")
    print(f"  VENDA   (Z_ou >= +{Z_OU_ENTRADA_VENDA})     : "
          f"{n_venda:>5,} sinais ({n_venda/n_total*100:.3f}%)")
    print(f"  TOTAL de sinais             : {n_sinais:>5,}")
    print()
    print(f"  -- Z-Score OU Estatisticas --")
    print(f"  Minimo Z_ou                 : {df_v['ou_zscore'].min():>+.3f}")
    print(f"  Maximo Z_ou                 : {df_v['ou_zscore'].max():>+.3f}")
    print(f"  Desvio Padrao Z_ou          : {df_v['ou_zscore'].std():>.4f}")
    print()
    print(f"  -- Gestao de Risco (media) --")
    print(f"  Stop Loss medio             : {sl_medio:>6.1f} pips")
    print(f"  Take Profit medio           : {tp_medio:>6.1f} pips")
    print(f"  R:R medio                   : 1:{MULT_TP_PIPS/MULT_STOP_PIPS:.1f}")
    print()
    print(f"  Grafico salvo em: graficos/ou_sinais.png")
    print(f"  Parquet salvo em: data/eurusd_h1_ou.parquet")
    print("=" * 65 + "\n")


# =============================================================================
# EXECUÇÃO DIRETA (python ou_strategy.py)
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ornstein-Uhlenbeck Mean Reversion — EURUSD H1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python ou_strategy.py
  python ou_strategy.py --inicio 2018-01-01 --fim 2026-04-10
  python ou_strategy.py --forcar
        """
    )
    parser.add_argument("--inicio", type=str, default=None,
                        help="Data de início (YYYY-MM-DD)")
    parser.add_argument("--fim", type=str, default=None,
                        help="Data de fim (YYYY-MM-DD)")
    parser.add_argument("--forcar", action="store_true",
                        help="Forçar recálculo mesmo se Parquet existir")

    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("  QUANT EURUSD -- MODULO: ORNSTEIN-UHLENBECK MEAN REVERSION")
    print("  Estrategia baseada no processo OU (dX = theta(mu-X)dt + sigma*dW)")
    print("=" * 65)

    df = calcular_e_salvar_ou(
        data_inicio=args.inicio,
        data_fim=args.fim,
        forcar=args.forcar,
    )

    print(f"\nConcluido! Proximo passo: execute ou_backtest.py\n")
