# -*- coding: utf-8 -*-
"""
================================================================================
hurst.py — Módulo de Análise de Regime Estatístico para NASDAQ M10 (Day Trade)
================================================================================

Objetivo:
    Calcular em janela móvel (rolling window) o Expoente de Hurst e a Meia-Vida
    de Reversão à Média para o US100 (CFD NASDAQ) em escala gráfica de 10 minutos (M10).
    Classificar o mercado em regimes de:
    - REVERSÃO à Média (H < 0.45)
    - TENDÊNCIA (H > 0.55)
    - INDEFINIDO (0.45 <= H <= 0.55)

Justificativa:
    Permitir a identificação estatística contínua de inércia direcional ou
    oscilação de reversão no NASDAQ intradiário para guiar day trades.

Saída:
    - data/nasdaq_m10_hurst.parquet (Série completa com as colunas hurst, regime e half_life)
    - graficos/hurst.png (Visualização em Dark Mode do regime do mercado)

================================================================================
"""

import os
import sys
import argparse
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Suprimir warnings desnecessários
warnings.filterwarnings("ignore")

# Configurar encoding para Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

# Configuração do Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTES E PARÂMETROS
# =============================================================================
DIR_PROJETO = Path(__file__).resolve().parent
DIR_DATA    = DIR_PROJETO / "data"
DIR_GRAFICOS = DIR_PROJETO / "graficos"

PARQUET_ENTRADA = DIR_DATA / "nasdaq_m10_completo.parquet"
PARQUET_SAIDA   = DIR_DATA / "nasdaq_m10_hurst.parquet"
CAMINHO_GRAFICO = DIR_GRAFICOS / "hurst.png"

JANELA_PRINCIPAL = 100
SUB_JANELAS      = [10, 20, 40, 80]

LIMIAR_REVERSAO  = 0.45
LIMIAR_TENDENCIA = 0.55

# Nomes dos dias da semana (em Português)
DIAS_SEMANA = {
    0: "Segunda",
    1: "Terça",
    2: "Quarta",
    3: "Quinta",
    4: "Sexta",
    5: "Sábado",
    6: "Domingo"
}

# =============================================================================
# FUNÇÕES MATEMÁTICAS OTIMIZADAS
# =============================================================================

def _quick_ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Calcula a inclinação (slope) de uma regressão linear OLS simples de forma rápida."""
    x_mean = x.mean()
    y_mean = y.mean()
    num = ((x - x_mean) * (y - y_mean)).sum()
    den = ((x - x_mean) ** 2).sum()
    if den == 0:
        return np.nan
    return num / den


def calcular_hurst_janela(retornos: np.ndarray) -> float:
    """
    Calcula o Expoente de Hurst para um vetor de retornos usando o método R/S.
    """
    if len(retornos) < JANELA_PRINCIPAL:
        return np.nan

    log_n = []
    log_rs = []

    for n in SUB_JANELAS:
        # Dividir retornos em segmentos sem sobreposição
        num_segmentos = JANELA_PRINCIPAL // n
        rs_segmentos = []

        for k in range(num_segmentos):
            segmento = retornos[k * n : (k + 1) * n]
            
            # Calcular R/S para o segmento
            mu = segmento.mean()
            y_t = np.cumsum(segmento - mu)
            
            r_range = y_t.max() - y_t.min()
            s_std = segmento.std(ddof=1)
            
            # Validação: Retornos constantes (S == 0)
            if s_std == 0:
                return np.nan
                
            rs = r_range / s_std
            rs_segmentos.append(rs)

        # RS médio para este n
        rs_medio = np.mean(rs_segmentos)
        if rs_medio > 0:
            log_n.append(np.log(n))
            log_rs.append(np.log(rs_medio))

    # Regressão linear OLS log(RS_medio) ~ log(n)
    if len(log_n) < 2:
        return np.nan

    h = _quick_ols_slope(np.array(log_n), np.array(log_rs))

    # Validação do Hurst
    if np.isnan(h) or h < 0.0 or h > 1.5:
        return np.nan

    return h


def calcular_half_life_janela(retornos: np.ndarray) -> float:
    """
    Estima a Meia-Vida de Reversão à Média via regressão AR(1) dos retornos.
    """
    if len(retornos) < JANELA_PRINCIPAL:
        return np.nan

    x = retornos[:-1]
    y = retornos[1:]

    # Regressão AR(1)
    beta = _quick_ols_slope(x, y)

    # Reversão à média requer 0 < beta < 1
    if np.isnan(beta) or beta <= 0.0 or beta >= 1.0:
        return np.nan

    theta = -np.log(beta)
    half_life = np.log(2.0) / theta

    return half_life

# =============================================================================
# PIPELINE ROLLING
# =============================================================================

def calcular_hurst_rolling(df: pd.DataFrame, janela: int = 100) -> pd.DataFrame:
    """
    Executa o cálculo móvel (rolling window) do Hurst e Meia-Vida sobre a série completa.
    """
    logger.info(f"Iniciando cálculo móvel (rolling) de Hurst na janela de {janela} candles M10...")
    
    # Calcular log-retornos caso não existam
    if "log_return" not in df.columns:
        df["log_return"] = np.log(df["Close"] / df["Close"].shift(1)).astype(np.float32)
        
    # Limpar qualquer valor NaNs inicial nos retornos para não corromper NumPy
    df["log_return"] = df["log_return"].fillna(0.0)
    
    # Extrair log-retornos
    retornos = df["log_return"].to_numpy()
    n_candles = len(df)

    hurst_values = np.full(n_candles, np.nan, dtype=np.float32)
    half_life_values = np.full(n_candles, np.nan, dtype=np.float32)

    # Loop móvel
    for i in tqdm(range(janela - 1, n_candles), desc="Processando Regime Hurst M10"):
        janela_retornos = retornos[i - janela + 1 : i + 1]
        
        # Calcular Hurst
        h = calcular_hurst_janela(janela_retornos)
        hurst_values[i] = h
        
        # Calcular Meia-Vida
        hl = calcular_half_life_janela(janela_retornos)
        half_life_values[i] = hl

    df["hurst"] = hurst_values
    df["half_life"] = half_life_values

    # Classificar regimes
    logger.info("Classificando regimes de mercado...")
    df["regime"] = "INDEFINIDO"
    df.loc[df["hurst"] < LIMIAR_REVERSAO, "regime"] = "REVERSAO"
    df.loc[df["hurst"] > LIMIAR_TENDENCIA, "regime"] = "TENDENCIA"
    df.loc[df["hurst"].isna(), "regime"] = np.nan

    df["regime"] = df["regime"].astype("category")

    return df

# =============================================================================
# RELATÓRIO DE SAÍDA E ESTATÍSTICAS
# =============================================================================

def imprimir_relatorio_hurst(df: pd.DataFrame):
    """Gera e imprime na tela o relatório detalhado do regime de Hurst do NASDAQ."""
    sep = "═" * 70
    sub_sep = "─" * 70
    
    df_valido = df.dropna(subset=["hurst"])
    total_validos = len(df_valido)
    
    if total_validos == 0:
        logger.error("Não há dados válidos de Hurst para gerar o relatório!")
        return

    # SEÇÃO 1 — Parâmetros utilizados
    print(f"\n{sep}")
    print("  SEÇÃO 1 — PARÂMETROS UTILIZADOS (NASDAQ M10)")
    print(sep)
    print(f"  Janela Principal de Retornos          : {JANELA_PRINCIPAL} M10 candles")
    print(f"  Sub-janelas de Escalonamento (n)     : {SUB_JANELAS}")
    print(f"  Limiar de Reversão à Média (Antipers.): < {LIMIAR_REVERSAO:.2f}")
    print(f"  Limiar de Tendência (Persistente)    : > {LIMIAR_TENDENCIA:.2f}")
    print(f"  Regime Indefinido (Passeio Aleatório) : {LIMIAR_REVERSAO:.2f} ≤ H ≤ {LIMIAR_TENDENCIA:.2f}")

    # SEÇÃO 2 — Distribuição de regimes
    rev_count = (df_valido["regime"] == "REVERSAO").sum()
    ten_count = (df_valido["regime"] == "TENDENCIA").sum()
    ind_count = (df_valido["regime"] == "INDEFINIDO").sum()

    rev_pct = (rev_count / total_validos) * 100
    ten_pct = (ten_count / total_validos) * 100
    ind_pct = (ind_count / total_validos) * 100

    print(f"\n{sep}")
    print("  SEÇÃO 2 — DISTRIBUIÇÃO DE REGIMES")
    print(sep)
    print(f"  Total de candles com Hurst válido     : {total_validos:>12,}")
    print(f"  REVERSÃO (Antipersistente)            : {rev_count:>12,} candles ({rev_pct:>5.2f}%)")
    print(f"  TENDÊNCIA (Persistente)               : {ten_count:>12,} candles ({ten_pct:>5.2f}%)")
    print(f"  INDEFINIDO (Passeio Aleatório)        : {ind_count:>12,} candles ({ind_pct:>5.2f}%)")

    # SEÇÃO 3 — Estatísticas do Hurst
    h_mean = df_valido["hurst"].mean()
    h_std = df_valido["hurst"].std()
    h_min = df_valido["hurst"].min()
    h_max = df_valido["hurst"].max()
    h_median = df_valido["hurst"].median()

    hl_valido = df_valido["half_life"].dropna()
    hl_mean = hl_valido.mean()
    hl_median = hl_valido.median()

    print(f"\n{sep}")
    print("  SEÇÃO 3 — ESTATÍSTICAS DO HURST & MEIA-VIDA")
    print(sep)
    print(f"  Hurst Médio                          : {h_mean:>12.4f}")
    print(f"  Hurst Mediana                        : {h_median:>12.4f}")
    print(f"  Desvio Padrão (Hurst)                : {h_std:>12.4f}")
    print(f"  Hurst Mínimo                         : {h_min:>12.4f}")
    print(f"  Hurst Máximo                         : {h_max:>12.4f}")
    print(f"  Meia-Vida Média de Reversão          : {hl_mean:>12.2f} candles M10")
    print(f"  Meia-Vida Mediana de Reversão        : {hl_median:>12.2f} candles M10")

    # SEÇÃO 4 — Distribuição de regimes por hora do dia (00h até 23h)
    print(f"\n{sep}")
    print("  SEÇÃO 4 — DISTRIBUIÇÃO DE REGIMES POR HORA DO DIA (Servidor US100)")
    print(sep)
    print("   Hora  │  REVERSÃO  │  TENDÊNCIA  │ INDEFINIDO │ regime Dominante")
    print("  " + "─" * 66)
    
    for h in range(24):
        df_hora = df_valido[df_valido.index.hour == h]
        t_hora = len(df_hora)
        if t_hora > 0:
            rev_h = (df_hora["regime"] == "REVERSAO").sum() / t_hora * 100
            ten_h = (df_hora["regime"] == "TENDENCIA").sum() / t_hora * 100
            ind_h = (df_hora["regime"] == "INDEFINIDO").sum() / t_hora * 100
            
            regimes_pct = {"REVERSÃO": rev_h, "TENDÊNCIA": ten_h, "INDEFINIDO": ind_h}
            dominante = max(regimes_pct, key=regimes_pct.get)
            
            op_tag = " 💼" if 16 <= h <= 22 else " 🌙"
            
            print(f"    {h:02d}h{op_tag} │   {rev_h:>5.1f}%   │   {ten_h:>5.1f}%   │   {ind_h:>5.1f}%   │ {dominante}")

    # SEÇÃO 5 — Evolução anual do Hurst
    print(f"\n{sep}")
    print("  SEÇÃO 5 — EVOLUÇÃO ANUAL DO EXPOENTE DE HURST (NASDAQ)")
    print(sep)
    anos = sorted(df_valido.index.year.unique())
    for ano in anos:
        df_ano = df_valido[df_valido.index.year == ano]
        h_ano = df_ano["hurst"].mean()
        ten_ano = (df_ano["regime"] == "TENDENCIA").sum() / len(df_ano) * 100
        rev_ano = (df_ano["regime"] == "REVERSAO").sum() / len(df_ano) * 100
        print(f"    Ano {ano} │ H Médio: {h_ano:.4f} │ Reversão: {rev_ano:>5.1f}% │ Tendência: {ten_ano:>5.1f}%")
    
    print(f"\n{sep}\n")

# =============================================================================
# GERAÇÃO DO GRÁFICO (DARK MODE)
# =============================================================================

def gerar_grafico_regime(df: pd.DataFrame):
    """
    Gera um gráfico analítico duplo em Dark Mode contendo o preço
    NASDAQ (US100) com coloração por regime e a evolução do Hurst.
    """
    logger.info("Gerando gráfico estatístico em Dark Mode para o NASDAQ...")
    
    DIR_GRAFICOS.mkdir(parents=True, exist_ok=True)
    
    df_plot = df.dropna(subset=["hurst"]).tail(4000)  # Foco nos últimos 4000 candles M10 para clareza
    if len(df_plot) == 0:
        logger.error("Sem dados de Hurst suficientes para plotar!")
        return

    plt.style.use('dark_background')
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 11), sharex=True, gridspec_kw={'height_ratios': [3, 2]})
    fig.suptitle("NASDAQ US100 M10 — Árbitro de Regime por Expoente de Hurst", fontsize=16, fontweight='bold', color='#FFFFFF')
    
    cor_tendencia = '#00E676'
    cor_reversao = '#FF1744'
    cor_indefinido = '#555555'
    
    # -------------------------------------------------------------------------
    # PAINEL 1 — Preço Close com coloração por Regime
    # -------------------------------------------------------------------------
    ax1.plot(df_plot.index, df_plot["Close"], color='#ECEFF1', linewidth=1.2, label='Preço US100')
    ax1.set_title("Identificação de Regime sobre a Série de Preços US100 M10", fontsize=12, fontweight='semibold', color='#CFD8DC')
    ax1.set_ylabel("NASDAQ Close", fontsize=11, color='#CFD8DC')
    ax1.grid(True, linestyle='--', alpha=0.15)
    
    regimes = df_plot["regime"].to_numpy()
    times = df_plot.index
    
    # Transições de regime rápidas
    mudancas = np.where(regimes[:-1] != regimes[1:])[0]
    limites = [0] + list(mudancas + 1) + [len(df_plot) - 1]
    
    for start, end in zip(limites[:-1], limites[1:]):
        reg = regimes[start]
        if reg == "TENDENCIA":
            ax1.axvspan(times[start], times[end], color=cor_tendencia, alpha=0.08)
        elif reg == "REVERSAO":
            ax1.axvspan(times[start], times[end], color=cor_reversao, alpha=0.08)
        else:
            ax1.axvspan(times[start], times[end], color=cor_indefinido, alpha=0.03)

    # -------------------------------------------------------------------------
    # PAINEL 2 — Expoente de Hurst
    # -------------------------------------------------------------------------
    ax2.plot(df_plot.index, df_plot["hurst"], color='#64B5F6', linewidth=1.0, label='Hurst (Janela=100)')
    ax2.axhline(LIMIAR_REVERSAO, color=cor_reversao, linestyle='--', alpha=0.6, linewidth=1.0, label='Limiar Reversão (0.45)')
    ax2.axhline(LIMIAR_TENDENCIA, color=cor_tendencia, linestyle='--', alpha=0.6, linewidth=1.0, label='Limiar Tendência (0.55)')
    
    ax2.fill_between(df_plot.index, df_plot["hurst"], LIMIAR_TENDENCIA, where=(df_plot["hurst"] > LIMIAR_TENDENCIA),
                     color=cor_tendencia, alpha=0.2, interpolate=True)
    ax2.fill_between(df_plot.index, df_plot["hurst"], LIMIAR_REVERSAO, where=(df_plot["hurst"] < LIMIAR_REVERSAO),
                     color=cor_reversao, alpha=0.2, interpolate=True)
    ax2.fill_between(df_plot.index, LIMIAR_REVERSAO, LIMIAR_TENDENCIA, 
                     where=((df_plot["hurst"] >= LIMIAR_REVERSAO) & (df_plot["hurst"] <= LIMIAR_TENDENCIA)),
                     color=cor_indefinido, alpha=0.1, interpolate=True)
    
    ax2.set_title("Evolução Temporal do Expoente de Hurst", fontsize=12, fontweight='semibold', color='#CFD8DC')
    ax2.set_ylabel("Hurst H", fontsize=11, color='#CFD8DC')
    ax2.set_ylim(0.15, 0.85)
    ax2.grid(True, linestyle='--', alpha=0.15)
    
    fig.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    plt.savefig(CAMINHO_GRAFICO, dpi=150, facecolor='#121212')
    plt.close()
    
    logger.info(f"Gráfico analítico salvo com sucesso em: {CAMINHO_GRAFICO.resolve()}")

# =============================================================================
# PIPELINE DE EXECUÇÃO
# =============================================================================

def processar_pipeline_hurst(forcar: bool = False) -> pd.DataFrame:
    """
    Controla o pipeline de carregamento, cálculo e cache do módulo hurst.
    """
    if PARQUET_SAIDA.exists() and not forcar:
        logger.info("Cache de Hurst NASDAQ encontrado. Carregando dados pré-calculados...")
        df = pd.read_parquet(PARQUET_SAIDA, engine="pyarrow")
        
        if not CAMINHO_GRAFICO.exists():
            gerar_grafico_regime(df)
            
        imprimir_relatorio_hurst(df)
        return df

    if not PARQUET_ENTRADA.exists():
        raise FileNotFoundError(
            f"Arquivo de série completa não encontrado: {PARQUET_ENTRADA}\n"
            f"Por favor, execute primeiro o data_loader.py do NASDAQ."
        )

    logger.info("Reprocessando base completa do NASDAQ M10 para o cálculo de Hurst...")
    df_completo = pd.read_parquet(PARQUET_ENTRADA, engine="pyarrow")

    # Calcular Hurst Rolling
    df_resultado = calcular_hurst_rolling(df_completo, JANELA_PRINCIPAL)

    # Salvar cache
    logger.info(f"Salvando resultados no cache: {PARQUET_SAIDA.name}")
    df_resultado.to_parquet(PARQUET_SAIDA, engine="pyarrow", compression="snappy", index=True)

    # Gerar Gráfico e Relatório
    gerar_grafico_regime(df_resultado)
    imprimir_relatorio_hurst(df_resultado)

    return df_resultado

# =============================================================================
# INTERFACE CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Árbitro de Regime NASDAQ M10 — Expoente de Hurst e Meia-Vida",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--forcar",
        action="store_true",
        help="Forçar reprocessamento total da base de retornos"
    )
    args = parser.parse_args()

    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█   ÁRBITRO DE REGIME NASDAQ M10 — EXPOENTE DE HURST            █")
    print("█   Método R/S móvel (Janela 100)                                 █")
    print("█   Classificação: REVERSÃO (<0.45) │ TENDÊNCIA (>0.55)           █")
    print("█" + " " * 68 + "█")
    print("█" * 70)

    try:
        processar_pipeline_hurst(args.forcar)
        print("✅ Módulo executado com sucesso!\n")
    except Exception as e:
        logger.exception("Erro crítico no cálculo de Hurst para NASDAQ:")
        sys.exit(1)
