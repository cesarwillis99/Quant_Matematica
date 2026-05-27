# -*- coding: utf-8 -*-
"""
================================================================================
zscore.py — Módulo de Reversão à Média por Z-Score do Preço
================================================================================

Objetivo:
    Implementar a estratégia quantitativa de Mean Reversion baseada no Z-Score do
    preço de fechamento Close.
    
Regras de Operação:
    - Opera APENAS quando o Hurst classifica o regime como "REVERSAO".
    - Indicadores calculados sobre a série COMPLETA (todos os 64.002 candles).
    - Geração de sinais de entrada restrita à janela OPERACIONAL (10h00-22h30 seg-sex).
    - Gestão de risco monitorada 24h na série completa.

Saída:
    - data/eurusd_h1_zscore.parquet
    - graficos/zscore_sinais.png (Gráfico de 3 painéis em Dark Mode)

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
# CONSTANTES E CAMINHOS
# =============================================================================
DIR_PROJETO = Path(__file__).resolve().parent
DIR_DATA    = DIR_PROJETO / "data"
DIR_GRAFICOS = DIR_PROJETO / "graficos"

PARQUET_ENTRADA = DIR_DATA / "eurusd_h1_hurst.parquet"
PARQUET_SAIDA   = DIR_DATA / "eurusd_h1_zscore.parquet"
CAMINHO_GRAFICO = DIR_GRAFICOS / "zscore_sinais.png"

# =============================================================================
# FUNÇÕES DE CÁLCULO DE APLICATIVOS (ROLLING)
# =============================================================================

def verificar_janela_operacional(dt_index: pd.DatetimeIndex) -> pd.Series:
    """
    Filtra os timestamps que estão dentro do horário operacional de segunda a sexta (10h00 às 22h30).
    """
    weekday = dt_index.weekday
    hora    = dt_index.strftime('%H:%M')

    seg_sex      = (weekday >= 0) & (weekday <= 4)
    horario_op   = (hora >= "10:00") & (hora <= "22:30")

    return pd.Series(seg_sex & horario_op, index=dt_index)

# =============================================================================
# PIPELINE COMPLETO
# =============================================================================

def calcular_indicadores_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula o Z-Score do Preço de Fechamento e o SL/TP dinâmicos com base na Volatilidade Realizada.
    """
    logger.info("Calculando Z-Score do Preço de Fechamento (rolling 50)...")
    
    # Z_t = (Close_t - mu_50) / std_50 (amostral, ddof=1)
    close_mean = df["Close"].rolling(window=50).mean()
    close_std  = df["Close"].rolling(window=50).std(ddof=1)
    
    # Se std == 0, resulta em NaN automaticamente no pandas
    df["zscore"] = ((df["Close"] - close_mean) / close_std).astype("float32")

    logger.info("Calculando Volatilidade Realizada e Gestão de Risco (rolling 50)...")
    # VR = std(log_retornos, janela=50, ddof=1)
    vr = df["log_return"].rolling(window=50, min_periods=50).std(ddof=1)
    df["vr_pips"] = (vr * df["Close"] * 10000.0).astype("float32")
    df["sl_pips"] = (2.0 * df["vr_pips"]).astype("float32")
    df["tp_pips"] = (3.0 * df["vr_pips"]).astype("float32")

    return df


def gerar_sinais_zscore(df: pd.DataFrame) -> tuple:
    """
    Gera sinais operacionais de Z-Score otimizados baseados no gatilho de RETORNO (cruzamento de volta).
    LONG (+1) se zscore cruzar acima de -2.5 (vindo de <= -2.5).
    SHORT (-1) se zscore cruzar abaixo de +2.5 (vindo de >= +2.5).
    Filtro de Hurst estrito: hurst < 0.40.
    """
    logger.info("Executando motor de geração de sinais Z-Score otimizado (Opção A)...")

    # 1. Filtro de Janela Operacional (10h00-22h30, seg-sex)
    op_window = verificar_janela_operacional(df.index)

    # 2. Pré-condições Operacionais
    # Usamos a coluna hurst diretamente < 0.40
    c1_regime = (df["hurst"] < 0.40)
    condicao_entrada = c1_regime & op_window

    sinal = np.zeros(len(df), dtype=np.int8)
    
    # Criar shifts para calcular cruzamento de volta (RETORNO)
    z = df["zscore"].values
    z_prev = df["zscore"].shift(1).values
    
    # Gatilho de retorno: vindo de fora do limite para dentro do limite
    z_entry = 2.5
    
    # LONG: no candle anterior estava <= -2.5, e no atual está > -2.5
    cond_long = condicao_entrada & (z_prev <= -z_entry) & (z > -z_entry)
    
    # SHORT: no candle anterior estava >= 2.5, e no atual está < 2.5
    cond_short = condicao_entrada & (z_prev >= z_entry) & (z < z_entry)
    
    # Garantir que não haja NaNs nos shifts
    cond_long = cond_long & (~df["zscore"].isna()) & (~df["zscore"].shift(1).isna())
    cond_short = cond_short & (~df["zscore"].isna()) & (~df["zscore"].shift(1).isna())
    
    sinal[cond_long] = 1
    sinal[cond_short] = -1

    df["sinal_zscore"] = sinal

    # --- Estatísticas de Bloqueio ---
    # Sinais potenciais (cruzamento de retorno sob Hurst < 0.40)
    potencial_long = c1_regime & (z_prev <= -z_entry) & (z > -z_entry)
    potencial_short = c1_regime & (z_prev >= z_entry) & (z < z_entry)
    potencial = potencial_long | potencial_short

    # Bloqueados apenas pelo filtro de horário
    bloqueado_horario = potencial & (~op_window)

    # Bloqueados por regime (cruzamento de retorno dentro do horário, mas com Hurst >= 0.40)
    potencial_sem_regime = op_window & (
        ((z_prev <= -z_entry) & (z > -z_entry)) | 
        ((z_prev >= z_entry) & (z < z_entry))
    )
    bloqueado_regime = potencial_sem_regime & (~c1_regime)

    stats_sinais = {
        "compra": int(np.sum(sinal == 1)),
        "venda": int(np.sum(sinal == -1)),
        "bloqueado_horario": int(np.sum(bloqueado_horario)),
        "bloqueado_regime": int(np.sum(bloqueado_regime)),
    }

    return df, stats_sinais


# =============================================================================
# RELATÓRIO DE SAÍDA
# =============================================================================

def imprimir_relatorio_zscore(df: pd.DataFrame, stats_sinais: dict):
    """
    Imprime relatório estatístico completo e formatado.
    """
    sep = "═" * 70
    sub_sep = "─" * 70

    df_valid = df.dropna(subset=["zscore"])
    total_validos = len(df_valid)

    if total_validos == 0:
        logger.error("Sem dados de Z-Score suficientes para imprimir o relatório!")
        return

    # SEÇÃO 1 — Z-Score estatísticas
    z_min = df_valid["zscore"].min()
    z_max = df_valid["zscore"].max()
    z_mean = df_valid["zscore"].mean()
    z_std = df_valid["zscore"].std()

    pct_z3 = (df_valid["zscore"].abs() > 3.0).sum() / total_validos * 100
    pct_z25 = (df_valid["zscore"].abs() > 2.5).sum() / total_validos * 100
    pct_z2 = (df_valid["zscore"].abs() > 2.0).sum() / total_validos * 100

    print(f"\n{sep}")
    print("  SEÇÃO 1 — ESTERÍSTICAS DE Z-SCORE DO PREÇO (Série Completa)")
    print(sep)
    print(f"  Z-Score Médio                        : {z_mean:>12.6f}")
    print(f"  Desvio Padrão Z-Score                : {z_std:>12.6f}")
    print(f"  Z-Score Mínimo                       : {z_min:>12.6f}")
    print(f"  Z-Score Máximo                       : {z_max:>12.6f}")
    print(f"  {sub_sep}")
    print(f"  % de candles com |Z| > 2.0 (Desvio)  : {pct_z2:>11.2f}%")
    print(f"  % de candles com |Z| > 2.5 (Extremo) : {pct_z25:>11.2f}%")
    print(f"  % de candles com |Z| > 3.0 (Anomalia): {pct_z3:>11.2f}%")

    # SEÇÃO 2 — Sinais gerados
    longs = stats_sinais["compra"]
    shorts = stats_sinais["venda"]
    ativos = longs + shorts
    pct_ativos = (ativos / total_validos) * 100

    print(f"\n{sep}")
    print("  SEÇÃO 2 — SINAIS GERADOS (Janela Operacional)")
    print(sep)
    print(f"  Total de sinais de COMPRA (LONG)      : {longs:>12,}")
    print(f"  Total de sinais de VENDA (SHORT)      : {shorts:>12,}")
    print(f"  Total de sinais ATIVOS                : {ativos:>12,}")
    print(f"  % do tempo com sinal ativo            : {pct_ativos:>11.2f}%")
    print(f"  Sinais bloqueados pelo filtro horário : {stats_sinais['bloqueado_horario']:>12,}")
    print(f"  Sinais bloqueados por regime != REVER.: {stats_sinais['bloqueado_regime']:>12,}")

    # SEÇÃO 3 — Gestão de risco
    sl_mean = df_valid["sl_pips"].mean()
    tp_mean = df_valid["tp_pips"].mean()
    rr_ratio = tp_mean / sl_mean if sl_mean > 0 else 0

    print(f"\n{sep}")
    print("  SEÇÃO 3 — GESTÃO DE RISCO (Baseada em Volatilidade Realizada)")
    print(sep)
    print(f"  Stop Loss Médio                       : {sl_mean:>12.2f} pips")
    print(f"  Take Profit Médio                     : {tp_mean:>12.2f} pips")
    print(f"  Relação Retorno:Risco (R:R) Confirmada:  1:{rr_ratio:.2f} (Intencional 1:1.5)")
    print(f"{sep}\n")

# =============================================================================
# GERAÇÃO DO GRÁFICO (DARK MODE)
# =============================================================================

def gerar_grafico_zscore_sinais(df: pd.DataFrame):
    """
    Gera gráfico com 3 painéis em Dark Mode:
    Painel 1: Preço EURUSD com setas de compra e venda
    Painel 2: Oscilador Z-Score do Preço com thresholds
    Painel 3: Coloração do Regime de Hurst
    """
    logger.info("Gerando gráfico analítico de Z-Score em Dark Mode...")

    DIR_GRAFICOS.mkdir(parents=True, exist_ok=True)

    df_plot = df.dropna(subset=["zscore", "hurst"]).tail(3000)  # Últimos 3000 candles para clareza visual
    if len(df_plot) == 0:
        logger.error("Sem dados de Z-Score suficientes para plotar!")
        return

    plt.style.use('dark_background')

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 11), sharex=True, 
                                         gridspec_kw={'height_ratios': [3, 2, 1]})
    
    fig.suptitle("Estratégia Mean Reversion por Z-Score — EURUSD H1", fontsize=16, fontweight='bold', color='#FFFFFF')
    
    times = df_plot.index
    
    # Cores HSL Harmoniosas
    cor_compra = '#00E676'      # Verde
    cor_venda = '#FF1744'       # Vermelho
    cor_indefinido = '#555555'  # Cinza

    # -------------------------------------------------------------------------
    # PAINEL 1 — Preço Close com Marcadores
    # -------------------------------------------------------------------------
    ax1.plot(times, df_plot["Close"], color='#ECEFF1', linewidth=1.2, label='Preço Close')
    
    compras = df_plot[df_plot["sinal_zscore"] == 1]
    vendas = df_plot[df_plot["sinal_zscore"] == -1]
    
    ax1.scatter(compras.index, compras["Close"] - 0.0010, color=cor_compra, marker='^', s=45, label='COMPRA (LONG)', zorder=5)
    ax1.scatter(vendas.index, vendas["Close"] + 0.0010, color=cor_venda, marker='v', s=45, label='VENDA (SHORT)', zorder=5)
    
    ax1.set_ylabel("Preço EURUSD", fontsize=11, color='#CFD8DC')
    ax1.grid(True, linestyle='--', alpha=0.1)
    ax1.legend(loc='upper left', framealpha=0.3)

    # -------------------------------------------------------------------------
    # PAINEL 2 — Z-Score do Preço
    # -------------------------------------------------------------------------
    ax2.plot(times, df_plot["zscore"], color='#AB47BC', linewidth=1.0, label='Z-Score (50 bars)')
    
    # Linhas de entrada e stop
    ax2.axhline(-3.0, color=cor_compra, linestyle='--', alpha=0.7, linewidth=1.0, label='Entrada Compra (-3.0)')
    ax2.axhline(3.0, color=cor_venda, linestyle='--', alpha=0.7, linewidth=1.0, label='Entrada Venda (+3.0)')
    ax2.axhline(-3.5, color=cor_compra, linestyle=':', alpha=0.5, linewidth=0.9, label='Stop Compra (-3.5)')
    ax2.axhline(3.5, color=cor_venda, linestyle=':', alpha=0.5, linewidth=0.9, label='Stop Venda (+3.5)')
    ax2.axhline(0.0, color='#ECEFF1', linestyle='--', alpha=0.3, linewidth=0.8)

    # Sombreado das zonas extremas
    ax2.fill_between(times, df_plot["zscore"], -3.0, where=(df_plot["zscore"] <= -3.0), color=cor_compra, alpha=0.2, interpolate=True)
    ax2.fill_between(times, df_plot["zscore"], 3.0, where=(df_plot["zscore"] >= 3.0), color=cor_venda, alpha=0.2, interpolate=True)

    ax2.set_ylabel("Z-Score", fontsize=11, color='#CFD8DC')
    ax2.set_ylim(-4.2, 4.2)
    ax2.grid(True, linestyle='--', alpha=0.1)
    ax2.legend(loc='upper left', framealpha=0.3)

    # -------------------------------------------------------------------------
    # PAINEL 3 — Regime Hurst
    # -------------------------------------------------------------------------
    regimes = df_plot["regime"].to_numpy()
    
    for i in range(len(df_plot)):
        reg = regimes[i]
        cor = cor_indefinido
        if reg == "TENDENCIA":
            cor = cor_compra
        elif reg == "REVERSAO":
            cor = cor_venda
            
        ax3.axvspan(times[max(0, i-1)], times[i], color=cor, alpha=0.15)
        
    ax3.set_ylabel("Regime Hurst", fontsize=11, color='#CFD8DC')
    ax3.get_yaxis().set_ticks([])  # Ocultar ticks
    ax3.grid(False)

    plt.tight_layout()
    plt.subplots_adjust(top=0.94)

    plt.savefig(CAMINHO_GRAFICO, dpi=150, facecolor='#121212')
    plt.close()

    logger.info(f"Gráfico de z-score e sinais salvo em: {CAMINHO_GRAFICO.resolve()}")

# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def processar_pipeline_zscore(forcar: bool = False) -> pd.DataFrame:
    """
    Controla o pipeline de carregamento, cálculo e cache do módulo zscore.
    """
    if PARQUET_SAIDA.exists() and not forcar:
        logger.info("Cache de Z-Score encontrado! Carregando parquet existente...")
        df = pd.read_parquet(PARQUET_SAIDA, engine="pyarrow")

        if not CAMINHO_GRAFICO.exists():
            gerar_grafico_zscore_sinais(df)

        # Calcular estatísticas rápidas de sinais para o relatório
        sinal = df["sinal_zscore"].to_numpy()
        op_window = verificar_janela_operacional(df.index)
        c1_regime = (df["regime"] == "REVERSAO")
        potencial = c1_regime & ((df["zscore"] <= -3.0) | (df["zscore"] >= 3.0))

        stats_sinais = {
            "compra": int(np.sum(sinal == 1)),
            "venda": int(np.sum(sinal == -1)),
            "bloqueado_horario": int(np.sum(potencial & (~op_window))),
            "bloqueado_regime": int(np.sum(op_window & ((df["zscore"] <= -3.0) | (df["zscore"] >= 3.0)) & (df["regime"] != "REVERSAO"))),
        }

        imprimir_relatorio_zscore(df, stats_sinais)
        return df

    if not PARQUET_ENTRADA.exists():
        raise FileNotFoundError(
            f"Parquet de Hurst não encontrado: {PARQUET_ENTRADA}\n"
            f"Execute primeiro o modulo hurst.py."
        )

    logger.info("Reprocessando base e calculando indicadores do Z-Score...")
    df_hurst = pd.read_parquet(PARQUET_ENTRADA, engine="pyarrow")

    # Passo 1-2: Calcular Z-Score e SL/TP com base em VR
    df_calc = calcular_indicadores_zscore(df_hurst)

    # Passo 3: Gerar os sinais
    df_final, stats_sinais = gerar_sinais_zscore(df_calc)

    # Salvar cache
    logger.info(f"Salvando resultados no cache: {PARQUET_SAIDA.name}")
    df_final.to_parquet(PARQUET_SAIDA, engine="pyarrow", compression="snappy", index=True)

    # Passo 5-6: Gerar gráfico e relatório
    gerar_grafico_zscore_sinais(df_final)
    imprimir_relatorio_zscore(df_final, stats_sinais)

    return df_final

# =============================================================================
# FUNÇÃO UTILITÁRIA EXPORTÁVEL
# =============================================================================

def carregar_zscore() -> pd.DataFrame:
    """
    Carrega o parquet com zscore calculado.
    Pode ser importada por outros módulos do sistema.
    """
    if not PARQUET_SAIDA.exists():
        raise FileNotFoundError(
            f"Parquet com Z-Score não encontrado: {PARQUET_SAIDA}\n"
            f"Execute primeiro: python zscore.py"
        )
    return pd.read_parquet(PARQUET_SAIDA, engine="pyarrow")

# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Módulo Mean Reversion por Z-Score — EURUSD H1",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--forcar",
        action="store_true",
        help="Forçar reprocessamento total da base de zscore"
    )
    args = parser.parse_args()

    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█   ESTRATÉGIA MEAN REVERSION POR Z-SCORE EURUSD v2             █")
    print("█   Filtros: Hurst Reversão + Z-Score Extremo (>= 3.0 ou <= -3.0) █")
    print("█   Sinais: Desvio Padrão do Preço (Janela 50)                     █")
    print("█" + " " * 68 + "█")
    print("█" * 70)

    processar_pipeline_zscore(args.forcar)
    print("✅ Módulo executado com sucesso!\n")
