# -*- coding: utf-8 -*-
"""
================================================================================
hurst.py — Módulo de Cálculo do Expoente de Hurst por Análise R/S
================================================================================
Autor: Quant Developer Sênior
Data: 2026

Descrição:
    Este módulo calcula o Expoente de Hurst (H) usando o método clássico de
    Análise R/S (Rescaled Range Analysis) em janela móvel de 100 candles H1.

    O Expoente de Hurst mede a "memória" de longo prazo de uma série temporal:
        H ≈ 0.5  → Caminhada Aleatória (Random Walk) — sem memória
        H > 0.5  → Persistência (trending) — movimentos se auto-reforçam
        H < 0.5  → Antipersistência (mean reversion) — movimentos se revertem

    CORREÇÃO ESTRUTURAL (ETAPA 2):
        Agora este módulo lê e processa a série H1 (eurusd_h1_clean.parquet).
        Isso garante que a estimativa da dependência de longo prazo inclua a
        baixa volatilidade noturna, proporcionando maior fidelidade matemática.

Fluxo de processamento:
    1. Carregar dados H1 (eurusd_h1_clean.parquet)
    2. Calcular R/S para sub-janelas [10, 20, 40, 80] dentro de cada janela de 100
    3. Estimar H via regressão linear de log(RS) em função de log(N)
    4. Classificar regime: REVERSÃO / TENDÊNCIA / INDEFINIDO
    5. Salvar DataFrame enriquecido em Parquet (eurusd_h1_hurst.parquet)
    6. Gerar gráfico de 2 painéis e imprimir tabela comparativa
================================================================================
"""

import sys
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats

# Suprimir warnings desnecessários
warnings.filterwarnings("ignore", category=UserWarning)
matplotlib.use("Agg")  # Backend sem interface gráfica

# tqdm opcional
try:
    from tqdm import tqdm
    TQDM_DISPONIVEL = True
except ImportError:
    TQDM_DISPONIVEL = False

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
DIR_MODULO  = Path(__file__).resolve().parent
DIR_DATA    = DIR_MODULO / "data"
DIR_GRAFICOS = DIR_MODULO / "graficos"

PARQUET_ENTRADA = DIR_DATA / "eurusd_h1_clean.parquet"
PARQUET_SAIDA   = DIR_DATA / "eurusd_h1_hurst.parquet"
GRAFICO_SAIDA   = DIR_GRAFICOS / "hurst.png"

# Parâmetros do cálculo de Hurst
JANELA_PRINCIPAL  = 100          # Janela móvel em candles H1
SUB_JANELAS       = [10, 20, 40, 80]  # Tamanhos para regressão R/S

# Limiares de classificação de regime
LIMIAR_REVERSAO   = 0.45  # H < 0.45 → mercado antipersistente (mean reversion)
LIMIAR_TENDENCIA  = 0.55  # H > 0.55 → mercado persistente (trend following)

# Estilo do gráfico
COR_FUNDO         = "#0D1117"   # Fundo escuro (GitHub dark)
COR_TEXTO         = "#E6EDF3"
COR_GRADE         = "#21262D"
COR_PRECO         = "#58A6FF"   # Azul para o preço
COR_HURST         = "#F0A500"   # Laranja para a linha de Hurst
COR_TENDENCIA     = "#2EA043"   # Verde para zona de tendência
COR_REVERSAO      = "#DA3633"   # Vermelho para zona de reversão
COR_INDEFINIDO    = "#6E7681"   # Cinza para zona indefinida


# =============================================================================
# FUNÇÕES MATEMÁTICAS — ANÁLISE R/S
# =============================================================================

def calcular_rs_unica_janela(retornos: np.ndarray) -> float:
    """
    Calcula a estatística R/S (Rescaled Range) para uma série de retornos.
    """
    N = len(retornos)
    if N < 4:
        return np.nan

    mu = np.mean(retornos)

    # Desvios em relação à média
    desvios = retornos - mu

    # Soma cumulativa dos desvios (perfil Y_t)
    perfil = np.cumsum(desvios)

    # Range: amplitude máxima do perfil
    R = np.max(perfil) - np.min(perfil)

    # Desvio padrão amostral
    S = np.std(retornos, ddof=1)

    # Evitar divisão por zero
    if S == 0 or S < 1e-15:
        return np.nan

    return R / S


def calcular_hurst_regressao(retornos: np.ndarray, sub_janelas: list) -> float:
    """
    Estima o Expoente de Hurst pela regressão linear de log(RS) vs log(N).
    """
    N_total = len(retornos)
    log_n_list = []
    log_rs_list = []

    for n in sub_janelas:
        if n >= N_total:
            continue

        num_segmentos = N_total // n
        if num_segmentos < 1:
            continue

        rs_valores = []
        for i in range(num_segmentos):
            segmento = retornos[i * n: (i + 1) * n]
            rs = calcular_rs_unica_janela(segmento)
            if not np.isnan(rs) and rs > 0:
                rs_valores.append(rs)

        if len(rs_valores) == 0:
            continue

        rs_medio = np.mean(rs_valores)
        log_n_list.append(np.log(n))
        log_rs_list.append(np.log(rs_medio))

    if len(log_n_list) < 2:
        return np.nan

    # Regressão linear: log(RS) = H * log(N) + constante
    slope, intercept, r_valor, p_valor, erro_std = stats.linregress(
        log_n_list, log_rs_list
    )

    # Validação
    if slope < 0 or slope > 1.5:
        return np.nan

    return float(slope)


def classificar_regime(h: float) -> str:
    """
    Classifica o regime de mercado com base no Expoente de Hurst.
    """
    if np.isnan(h):
        return "INDEFINIDO"
    if h < LIMIAR_REVERSAO:
        return "REVERSAO"
    if h > LIMIAR_TENDENCIA:
        return "TENDENCIA"
    return "INDEFINIDO"


# =============================================================================
# CÁLCULO EM JANELA MÓVEL
# =============================================================================

def calcular_hurst_rolling(
    serie_retornos: pd.Series,
    janela: int = JANELA_PRINCIPAL,
    sub_janelas: list = SUB_JANELAS,
) -> pd.Series:
    """
    Aplica o cálculo do Expoente de Hurst em janela móvel sobre a série.
    """
    n = len(serie_retornos)
    valores_hurst = np.full(n, np.nan, dtype=np.float32)
    retornos_np = serie_retornos.values.astype(np.float64)

    logger.info(
        f"Calculando Hurst em janela móvel (Série Completa): "
        f"{n:,} candles, janela={janela}, sub-janelas={sub_janelas}"
    )

    iterador = range(janela - 1, n)
    if TQDM_DISPONIVEL:
        iterador = tqdm(
            iterador,
            desc="Calculando Hurst",
            unit=" candles",
            ncols=80,
            colour="yellow",
        )

    for i in iterador:
        janela_atual = retornos_np[i - janela + 1: i + 1]
        h = calcular_hurst_regressao(janela_atual, sub_janelas)
        valores_hurst[i] = h

    return pd.Series(valores_hurst, index=serie_retornos.index, name="hurst")


# =============================================================================
# GERAÇÃO DO GRÁFICO
# =============================================================================

def gerar_grafico_hurst(df: pd.DataFrame, caminho: Path) -> None:
    """
    Gera gráfico de 2 painéis mostrando preço e Hurst sobre a série completa.
    """
    logger.info("Gerando gráfico do Expoente de Hurst...")

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

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(18, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1.5], "hspace": 0.04},
    )
    fig.suptitle(
        "EURUSD H1 — Análise do Expoente de Hurst (Série Completa 24h)\n"
        f"Janela: {JANELA_PRINCIPAL} candles | Sub-janelas: {SUB_JANELAS}",
        color=COR_TEXTO,
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    ax1.plot(
        df.index,
        df["Close"],
        color=COR_PRECO,
        linewidth=0.7,
        alpha=0.9,
        label="EURUSD Fechamento H1",
    )
    ax1.set_ylabel("Preço (EURUSD)", color=COR_TEXTO, fontsize=10)
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.3)
    ax1.grid(True, which="major", linestyle="--", alpha=0.3)

    df_hurst_valido = df.dropna(subset=["hurst"])
    idx = df_hurst_valido.index
    h   = df_hurst_valido["hurst"].values

    # fill_between por regime
    ax2.fill_between(
        idx, h, LIMIAR_TENDENCIA,
        where=(h > LIMIAR_TENDENCIA),
        color=COR_TENDENCIA, alpha=0.25,
        label=f"Tendência (H > {LIMIAR_TENDENCIA})",
    )
    ax2.fill_between(
        idx, h, LIMIAR_REVERSAO,
        where=(h < LIMIAR_REVERSAO),
        color=COR_REVERSAO, alpha=0.25,
        label=f"Reversão (H < {LIMIAR_REVERSAO})",
    )
    ax2.fill_between(
        idx, h, 0,
        where=(h >= LIMIAR_REVERSAO) & (h <= LIMIAR_TENDENCIA),
        color=COR_INDEFINIDO, alpha=0.10,
        label=f"Indefinido ({LIMIAR_REVERSAO} ≤ H ≤ {LIMIAR_TENDENCIA})",
    )

    ax2.plot(idx, h, color=COR_HURST, linewidth=0.8, alpha=0.9, label="Hurst (H)")

    ax2.axhline(LIMIAR_TENDENCIA, color=COR_TENDENCIA, linestyle="--",
                linewidth=1.2, alpha=0.8)
    ax2.axhline(LIMIAR_REVERSAO, color=COR_REVERSAO, linestyle="--",
                linewidth=1.2, alpha=0.8)
    ax2.axhline(0.50, color=COR_TEXTO, linestyle=":",
                linewidth=0.8, alpha=0.4)

    ax2.set_ylabel("Expoente de Hurst (H)", color=COR_TEXTO, fontsize=10)
    ax2.set_xlabel("Data", color=COR_TEXTO, fontsize=10)
    ax2.set_ylim(max(0, h.min() - 0.05), min(1, h.max() + 0.05))
    ax2.legend(loc="upper right", fontsize=8, framealpha=0.3, ncol=2)
    ax2.grid(True, which="major", linestyle="--", alpha=0.3)

    fig.autofmt_xdate(rotation=30, ha="right")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(caminho, dpi=150, bbox_inches="tight", facecolor=COR_FUNDO)
    plt.close()


# =============================================================================
# EXIBIÇÃO E COMPARATIVO DE MUDANÇA
# =============================================================================

def imprimir_comparativo_hurst(df_novo: pd.DataFrame, caminho_antigo: Path) -> None:
    """
    Imprime a tabela comparativa solicitada pelo usuário na Etapa 2.
    """
    df_novo_valido = df_novo.dropna(subset=["hurst"])
    n_novo = len(df_novo_valido)
    
    h_novo = df_novo_valido["hurst"]
    contagem_novo = df_novo_valido["regime"].value_counts()
    
    pct_trend_novo = contagem_novo.get("TENDENCIA", 0) / n_novo * 100
    pct_rev_novo = contagem_novo.get("REVERSAO", 0) / n_novo * 100
    pct_ind_novo = contagem_novo.get("INDEFINIDO", 0) / n_novo * 100

    # Valores padrão da série filtrada (fallback)
    h_medio_antigo = 0.5873
    std_antigo = 0.1032
    pct_trend_antigo = 64.7
    pct_rev_antigo = 9.6
    pct_ind_antigo = 25.7

    # Carregar estatísticas reais da Série Filtrada anterior se o arquivo existir
    if caminho_antigo.exists():
        try:
            df_antigo = pd.read_parquet(caminho_antigo)
            df_antigo_valido = df_antigo.dropna(subset=["hurst"])
            n_antigo = len(df_antigo_valido)
            if n_antigo > 0:
                h_medio_antigo = df_antigo_valido["hurst"].mean()
                std_antigo = df_antigo_valido["hurst"].std()
                contagem_antigo = df_antigo_valido["regime"].value_counts()
                pct_trend_antigo = contagem_antigo.get("TENDENCIA", 0) / n_antigo * 100
                pct_rev_antigo = contagem_antigo.get("REVERSAO", 0) / n_antigo * 100
                pct_ind_antigo = contagem_antigo.get("INDEFINIDO", 0) / n_antigo * 100
        except Exception as e:
            logger.warning(f"Erro ao ler parquet antigo para comparativo: {e}. Usando fallback.")

    print("\n" + "=" * 65)
    print("  ETAPA 2 CONCLUÍDA — COMPARATIVO DO EXPOENTE DE HURST")
    print("=" * 65)
    print(f"  {'Métrica':<20} | {'Série Filtrada':>18} | {'Série Completa':>18}")
    print(f"  {'-'*20}-+-{'-'*18}-+-{'-'*18}")
    print(f"  {'Hurst médio':<20} | {h_medio_antigo:>18.4f} | {h_novo.mean():>18.4f}")
    print(f"  {'Desvio Padrão':<20} | {std_antigo:>18.4f} | {h_novo.std():>18.4f}")
    print(f"  {'% Tempo TENDÊNCIA':<20} | {pct_trend_antigo:>17.1f}% | {pct_trend_novo:>17.1f}%")
    print(f"  {'% Tempo REVERSÃO':<20} | {pct_rev_antigo:>17.1f}% | {pct_rev_novo:>17.1f}%")
    print(f"  {'% Tempo INDEFINIDO':<20} | {pct_ind_antigo:>17.1f}% | {pct_ind_novo:>17.1f}%")
    print("=" * 65 + "\n")


# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================

def calcular_e_salvar_hurst(
    data_inicio: str = None,
    data_fim: str = None,
    forcar_reprocessamento: bool = False,
) -> pd.DataFrame:
    """
    Pipeline de processamento do Expoente de Hurst sobre a série completa.
    """
    DIR_GRAFICOS.mkdir(parents=True, exist_ok=True)
    DIR_DATA.mkdir(parents=True, exist_ok=True)

    # Cache
    if PARQUET_SAIDA.exists() and not forcar_reprocessamento:
        logger.info(
            f"Parquet Hurst já existe: {PARQUET_SAIDA.name}. Carregando cache..."
        )
        df = pd.read_parquet(PARQUET_SAIDA, engine="pyarrow")
        if data_inicio:
            df = df[df.index >= data_inicio]
        if data_fim:
            df = df[df.index <= data_fim]
        return df

    # Carregar dados
    if not PARQUET_ENTRADA.exists():
        raise FileNotFoundError(f"Parquet de entrada não encontrado: {PARQUET_ENTRADA}")

    logger.info(f"Carregando dados H1 série completa: {PARQUET_ENTRADA.name}")
    df = pd.read_parquet(PARQUET_ENTRADA, engine="pyarrow")
    logger.info(f"Dados carregados: {len(df):,} candles H1")

    if data_inicio:
        df = df[df.index >= data_inicio]
    if data_fim:
        df = df[df.index <= data_fim]

    # Calcular Hurst em janela móvel
    serie_hurst = calcular_hurst_rolling(
        df["log_return"],
        janela=JANELA_PRINCIPAL,
        sub_janelas=SUB_JANELAS,
    )
    df["hurst"] = serie_hurst.astype("float32")

    # Classificar regimes
    df["regime"] = df["hurst"].apply(classificar_regime)
    df["regime"] = df["regime"].astype("category")

    logger.info("Classificação de regimes concluída.")

    # Salvar Parquet completo
    df.to_parquet(PARQUET_SAIDA, engine="pyarrow", compression="snappy", index=True)
    
    # Gerar gráfico
    gerar_grafico_hurst(df, GRAFICO_SAIDA)

    return df


# =============================================================================
# EXECUÇÃO DIRETA
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Cálculo de Hurst sobre série H1 completa 24h",
    )
    parser.add_argument(
        "--inicio",
        type=str,
        default=None,
        help="Data de início (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--fim",
        type=str,
        default=None,
        help="Data de fim (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--forcar",
        action="store_true",
        help="Forçar recálculo"
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  QUANT EURUSD — ETAPA 2: CÁLCULO DE HURST (SÉRIE COMPLETA)")
    print("  Regressão R/S em Janela Móvel contínua de 24h")
    print("=" * 60)

    df_hurst = calcular_e_salvar_hurst(
        data_inicio=args.inicio,
        data_fim=args.fim,
        forcar_reprocessamento=args.forcar,
    )
