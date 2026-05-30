# -*- coding: utf-8 -*-
"""
================================================
momentum.py — Estratégia Quantitativa
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
  3. Executar: python momentum.py
================================================
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
PARQUET_SAIDA       = DIR_DATA / f"{ATIVO.lower()}_{TIMEFRAME.lower()}_momentum.parquet"
CAMINHO_GRAFICO     = DIR_GRAFICOS / f"{ATIVO.lower()}_{TIMEFRAME.lower()}_momentum_sinais.png"

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

# =============================================================================
# FUNÇÕES DE CÁLCULO DE APLICATIVOS (ROLLING)
# =============================================================================

def _percentrank(arr: np.ndarray) -> float:
    """
    Calcula o posto percentílico (percentil) do último valor em relação aos anteriores.
    Exclui o valor atual do cálculo do histórico para evitar auto-inflação.
    """
    val_atual  = arr[-1]
    historico  = arr[:-1]   # 99 valores anteriores
    n_historico = len(historico)
    if n_historico == 0:
        return 0.5
    n_menores = np.sum(historico < val_atual)
    return float(n_menores) / float(n_historico)

def _calcular_entropia_janela(arr: np.ndarray) -> float:
    """
    Calcula a Entropia de Shannon normalizada para uma janela de log-retornos.
    Normalizada por log2(10) para clipar estritamente entre 0.0 e 1.0.
    """
    if len(arr) < 30:
        return np.nan
    
    s_std = arr.std(ddof=1)
    if s_std < 1e-15:
        return 0.0
        
    # Discretização em 10 bins de igual largura
    contagens, _ = np.histogram(arr, bins=10)
    
    # Calcular probabilidades
    p_i = contagens / 30.0
    p_i = p_i[p_i > 0.0]  # Evitar log2(0)
    
    h_shannon = -np.sum(p_i * np.log2(p_i))
    h_max = np.log2(10.0)
    h_norm = h_shannon / h_max
    
    return float(np.clip(h_norm, 0.0, 1.0))

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

def calcular_indicadores_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula os indicadores cinemáticos, de entropia e gestão de risco sobre a série completa.
    """
    logger.info("Calculando Cinemática do Preço (Velocidade e Aceleração)...")
    df["velocidade"] = df["Close"].diff(1).astype("float32")
    df["aceleracao"] = df["velocidade"].diff(1).astype("float32")

    logger.info("Calculando Percentil da Aceleração (rolling 100)...")
    df["percentil_acel"] = df["aceleracao"].rolling(
        window=100, min_periods=100
    ).apply(_percentrank, raw=True).astype("float32")

    logger.info("Calculando Entropia de Shannon (rolling 30)...")
    df["entropia_shannon"] = df["log_return"].rolling(
        window=30, min_periods=30
    ).apply(_calcular_entropia_janela, raw=True).astype("float32")

    logger.info("Calculando Volatilidade Realizada e Gestão de Risco (rolling 50)...")
    # VR = std(log_retornos, janela=50, ddof=1)
    vr = df["log_return"].rolling(window=50, min_periods=50).std(ddof=1)
    df["vr_pips"] = (vr * df["Close"] * 10000.0).astype("float32")
    # Clipping alinhado com a otimização:
    # SL mínimo 3.0 pips, máximo 60.0 pips
    # TP mínimo 4.5 pips, máximo 90.0 pips
    df["sl_pips"] = np.clip(2.0 * df["vr_pips"], 3.0, 60.0).astype("float32")
    df["tp_pips"] = np.clip(5.0 * df["vr_pips"], 4.5, 90.0).astype("float32")

    return df

def gerar_sinais_momentum(df: pd.DataFrame) -> tuple:
    """
    Gera sinais operacionais LONG (+1), SHORT (-1) ou NEUTRO (0) sobre a janela operacional.
    Lógica otimizada (Opção A): Hurst > 0.50, Entropia < 0.60 e Aceleração > 75% ou < 25%.
    
    IMPORTANTE: O sinal gerado em t deve ser executado no
    Close[t] com spread aplicado:
        LONG  → entrada = Close[t] + 0.00005
        SHORT → entrada = Close[t] - 0.00005
    Qualquer backtest que consuma esta coluna deve aplicar
    este custo para manter alinhamento com a otimização.
    """
    logger.info("Executando motor de geração de sinais Momentum otimizado (Opção A)...")
    
    # 1. Filtro de Janela Operacional (10h00-22h30, seg-sex)
    op_window = verificar_janela_operacional(df.index)

    # 2. Pré-condições Operacionais
    # Usamos o Hurst > 0.50 diretamente do parquet
    c1_regime = (df["hurst"] > 0.50)
    c2_entropia_operavel = (df["entropia_shannon"] < 0.60)

    # Filtro Operacional — c2 já cobre toda a zona de entropia
    condicao_entrada = c1_regime & c2_entropia_operavel & op_window

    sinal = np.zeros(len(df), dtype=np.int8)

    # LONG (+1)
    percentil_long_trigger = 0.80
    cond_long = condicao_entrada & (df["percentil_acel"] > percentil_long_trigger) & (df["velocidade"] > 0.0)
    sinal[cond_long] = 1

    # SHORT (-1) — trigger sempre espelho do LONG (alinhado com a otimização)
    percentil_short_trigger = 1.0 - percentil_long_trigger
    cond_short = condicao_entrada & (df["percentil_acel"] < percentil_short_trigger) & (df["velocidade"] < 0.0)
    sinal[cond_short] = -1

    df["sinal_momentum"] = sinal

    # --- Estatísticas de Bloqueio ---
    # Sinais potenciais com cinemática alinhada e sob regime de tendência
    potencial_long  = c1_regime & (df["percentil_acel"] > percentil_long_trigger)  & (df["velocidade"] > 0.0)
    potencial_short = c1_regime & (df["percentil_acel"] < percentil_short_trigger) & (df["velocidade"] < 0.0)
    potencial = potencial_long | potencial_short

    # Bloqueados apenas pelo fuso/janela de horário
    bloqueado_horario = potencial & (~op_window)

    # Bloqueados por entropia caótica (>= 0.60 ou > 0.80) dentro da janela operacional
    bloqueado_entropia = potencial & op_window & (df["entropia_shannon"] >= 0.60)

    stats_sinais = {
        "compra": int(np.sum(sinal == 1)),
        "venda": int(np.sum(sinal == -1)),
        "bloqueado_horario": int(np.sum(bloqueado_horario)),
        "bloqueado_entropia": int(np.sum(bloqueado_entropia)),
    }

    return df, stats_sinais

# =============================================================================
# RELATÓRIO DE SAÍDA
# =============================================================================

def imprimir_relatorio_momentum(df: pd.DataFrame, stats_sinais: dict):
    """
    Imprime relatório completo com 4 seções após o processamento.
    """
    sep = "═" * 70
    sub_sep = "─" * 70

    df_valid = df.dropna(subset=["percentil_acel", "entropia_shannon"])
    total_validos = len(df_valid)

    if total_validos == 0:
        logger.error("Sem dados válidos suficientes para imprimir o relatório!")
        return

    # SEÇÃO 1 — Cinemática do Preço
    v_mean = df_valid["velocidade"].mean()
    v_std = df_valid["velocidade"].std()
    v_min = df_valid["velocidade"].min()
    v_max = df_valid["velocidade"].max()

    a_mean = df_valid["aceleracao"].mean()
    a_std = df_valid["aceleracao"].std()
    a_min = df_valid["aceleracao"].min()
    a_max = df_valid["aceleracao"].max()

    print(f"\n{sep}")
    print("  SEÇÃO 1 — CINEMÁTICA DO PREÇO (Série Completa)")
    print(sep)
    print(f"  Velocidade Média (Close diff)        : {v_mean:>12.6f} pips")
    print(f"  Desvio Padrão Velocidade             : {v_std:>12.6f}")
    print(f"  Velocidade Mínima                    : {v_min:>12.6f} pips")
    print(f"  Velocidade Máxima                    : {v_max:>12.6f} pips")
    print(f"  {sub_sep}")
    print(f"  Aceleração Média (Velocidade diff)   : {a_mean:>12.6f} pips")
    print(f"  Desvio Padrão Aceleração             : {a_std:>12.6f}")
    print(f"  Aceleração Mínima                    : {a_min:>12.6f} pips")
    print(f"  Aceleração Máxima                    : {a_max:>12.6f} pips")

    # SEÇÃO 2 — Entropia de Shannon
    e_mean = df_valid["entropia_shannon"].mean()
    e_median = df_valid["entropia_shannon"].median()
    e_ruido_pct = (df_valid["entropia_shannon"] > 0.80).sum() / total_validos * 100
    e_operavel_pct = (df_valid["entropia_shannon"] < 0.60).sum() / total_validos * 100

    print(f"\n{sep}")
    print("  SEÇÃO 2 — ENTROPIA DE SHANNON (Série Completa)")
    print(sep)
    print(f"  Entropia Média Normalizada (30 bars) : {e_mean:>12.4f}")
    print(f"  Entropia Mediana                     : {e_median:>12.4f}")
    print(f"  % de candles em Ruído Puro (> 0.80)  : {e_ruido_pct:>11.2f}% (Operações Bloqueadas)")
    print(f"  % de candles em Zona Operável (< 0.60): {e_operavel_pct:>11.2f}%")

    # SEÇÃO 3 — Sinais Gerados
    longs = stats_sinais["compra"]
    shorts = stats_sinais["venda"]
    ativos = longs + shorts
    pct_ativos = (ativos / total_validos) * 100

    print(f"\n{sep}")
    print("  SEÇÃO 3 — GERAÇÃO DE SINAIS (Janela Operacional)")
    print(sep)
    print(f"  Total de sinais de COMPRA (LONG)      : {longs:>12,}")
    print(f"  Total de sinais de VENDA (SHORT)      : {shorts:>12,}")
    print(f"  Total de sinais ATIVOS                : {ativos:>12,}")
    print(f"  % do tempo com sinal ativo            : {pct_ativos:>11.2f}%")
    print(f"  Sinais bloqueados pelo filtro horário : {stats_sinais['bloqueado_horario']:>12,}")
    print(f"  Sinais bloqueados por entropia/caos   : {stats_sinais['bloqueado_entropia']:>12,}")

    # SEÇÃO 4 — Gestão de Risco
    sl_mean = df_valid["sl_pips"].mean()
    tp_mean = df_valid["tp_pips"].mean()
    rr_ratio = tp_mean / sl_mean if sl_mean > 0 else 0

    print(f"\n{sep}")
    print("  SEÇÃO 4 — GESTÃO DE RISCO (Baseada em Volatilidade Realizada)")
    print(sep)
    print(f"  Stop Loss Médio                       : {sl_mean:>12.2f} pips")
    print(f"  Take Profit Médio                     : {tp_mean:>12.2f} pips")
    print(f"  Relação Retorno:Risco (R:R) Confirmada:  1:{rr_ratio:.2f} (Intencional 1:2.67)")
    print(f"{sep}\n")

# =============================================================================
# GERAÇÃO DO GRÁFICO (DARK MODE)
# =============================================================================

def gerar_grafico_momentum_sinais(df: pd.DataFrame):
    """
    Gera um gráfico analítico com 4 painéis em Dark Mode:
    Painel 1: Preço EURUSD com setas de compra e venda
    Painel 2: Velocidade e Aceleração do Preço
    Painel 3: Entropia de Shannon normalizada
    Painel 4: Regimes de Hurst (sombreamento de fundo)
    """
    logger.info("Gerando gráfico de sinais em Dark Mode...")
    
    DIR_GRAFICOS.mkdir(parents=True, exist_ok=True)
    
    df_plot = df.dropna(subset=["percentil_acel", "entropia_shannon", "hurst"]).tail(3000)  # Últimos 3000 candles para clareza visual
    if len(df_plot) == 0:
        logger.error("Sem dados válidos suficientes para plotar!")
        return

    plt.style.use('dark_background')
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(16, 12), sharex=True, 
                                             gridspec_kw={'height_ratios': [3, 2, 2, 1]})
    
    fig.suptitle(f"Estratégia Momentum & Entropia — {ATIVO} {TIMEFRAME}", fontsize=16, fontweight='bold', color='#FFFFFF')
    
    times = df_plot.index
    
    # -------------------------------------------------------------------------
    # PAINEL 1 — Preço Close com Marcadores de Entrada
    # -------------------------------------------------------------------------
    ax1.plot(times, df_plot["Close"], color='#ECEFF1', linewidth=1.2, label='Preço Close')
    
    # Adicionar setas de Compra/Venda
    compras = df_plot[df_plot["sinal_momentum"] == 1]
    vendas = df_plot[df_plot["sinal_momentum"] == -1]
    
    ax1.scatter(compras.index, compras["Close"] - 0.0010, color='#00E676', marker='^', s=45, label='COMPRA (LONG)', zorder=5)
    ax1.scatter(vendas.index, vendas["Close"] + 0.0010, color='#FF1744', marker='v', s=45, label='VENDA (SHORT)', zorder=5)
    
    ax1.set_ylabel(f"Preço {ATIVO}", fontsize=11, color='#CFD8DC')
    ax1.grid(True, linestyle='--', alpha=0.1)
    ax1.legend(loc='upper left', framealpha=0.3)
    
    # -------------------------------------------------------------------------
    # PAINEL 2 — Velocidade e Aceleração
    # -------------------------------------------------------------------------
    ax2.plot(times, df_plot["velocidade"], color='#29B6F6', linewidth=0.9, alpha=0.8, label='Velocidade (v_t)')
    ax2.plot(times, df_plot["aceleracao"], color='#FFA726', linewidth=0.8, alpha=0.8, label='Aceleração (a_t)')
    ax2.axhline(0.0, color='#90A4AE', linestyle='--', linewidth=0.8, alpha=0.5)
    
    ax2.set_ylabel("Cinemática", fontsize=11, color='#CFD8DC')
    ax2.grid(True, linestyle='--', alpha=0.1)
    ax2.legend(loc='upper left', framealpha=0.3)
    
    # -------------------------------------------------------------------------
    # PAINEL 3 — Entropia de Shannon
    # -------------------------------------------------------------------------
    ax3.plot(times, df_plot["entropia_shannon"], color='#AB47BC', linewidth=1.0, label='Entropia de Shannon')
    ax3.axhline(0.80, color='#FF1744', linestyle='--', alpha=0.6, linewidth=1.0, label='Bloqueio Ruído (0.80)')
    ax3.axhline(0.60, color='#FBC02D', linestyle='--', alpha=0.6, linewidth=1.0, label='Limite Operação (0.60)')
    
    # Sombreado das zonas de Entropia
    ax3.fill_between(times, 0.80, 1.0, color='#FF1744', alpha=0.08, label='Ruído Puro')
    ax3.fill_between(times, 0.0, 0.60, color='#00E676', alpha=0.05, label='Zona Operável')
    
    ax3.set_ylabel("Entropia Shannon", fontsize=11, color='#CFD8DC')
    ax3.set_ylim(0.0, 1.0)
    ax3.grid(True, linestyle='--', alpha=0.1)
    ax3.legend(loc='upper left', framealpha=0.3)
    
    # -------------------------------------------------------------------------
    # PAINEL 4 — Regime Hurst Colorido
    # -------------------------------------------------------------------------
    regimes = df_plot["regime"].to_numpy()
    
    # Colorir blocos contínuos no tempo para o regime Hurst
    for i in range(len(df_plot)):
        reg = regimes[i]
        cor = '#555555'  # Indefinido
        if reg == "TENDENCIA":
            cor = '#00E676'
        elif reg == "REVERSAO":
            cor = '#FF1744'
            
        ax4.axvspan(times[max(0, i-1)], times[i], color=cor, alpha=0.15)
        
    ax4.set_ylabel("Regime Hurst", fontsize=11, color='#CFD8DC')
    ax4.get_yaxis().set_ticks([])  # Ocultar ticks do Hurst
    ax4.grid(False)
    
    # Ajustes finais de layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    
    plt.savefig(CAMINHO_GRAFICO, dpi=150, facecolor='#121212')
    plt.close()
    
    logger.info(f"Gráfico de momentum e sinais salvo com sucesso em: {CAMINHO_GRAFICO.resolve()}")

# =============================================================================
# FUNÇÃO PRINCIPAL — PIPELINE
# =============================================================================

def processar_pipeline_momentum(forcar: bool = False) -> pd.DataFrame:
    """
    Controla o pipeline de carregamento, cálculo e cache do módulo momentum.
    """
    if PARQUET_SAIDA.exists() and not forcar:
        logger.info("Cache de Momentum encontrado! Carregando parquet existente...")
        df = pd.read_parquet(PARQUET_SAIDA, engine="pyarrow")
        
        # Gerar o gráfico se não existir
        if not CAMINHO_GRAFICO.exists():
            gerar_grafico_momentum_sinais(df)
            
        # Calcular estatísticas rápidas de sinais para o relatório impresso
        sinal = df["sinal_momentum"].to_numpy()
        op_window = verificar_janela_operacional(df.index)
        c1_regime = (df["regime"] == "TENDENCIA")
        potencial = (c1_regime & (df["percentil_acel"] > 0.80) & (df["velocidade"] > 0.0)) | \
                    (c1_regime & (df["percentil_acel"] < 0.20) & (df["velocidade"] < 0.0))
        
        stats_sinais = {
            "compra": int(np.sum(sinal == 1)),
            "venda": int(np.sum(sinal == -1)),
            "bloqueado_horario": int(np.sum(potencial & (~op_window))),
            "bloqueado_entropia": int(np.sum(potencial & op_window & (df["entropia_shannon"] >= 0.60))),
        }
        
        imprimir_relatorio_momentum(df, stats_sinais)
        return df

    if not PARQUET_COMPLETO.exists():
        raise FileNotFoundError(
            f"Arquivo de entrada de Hurst não encontrado: {PARQUET_COMPLETO}\n"
            f"Por favor, execute primeiro o módulo hurst.py."
        )

    logger.info("Reprocessando base e calculando indicadores de Momentum...")
    df_hurst = pd.read_parquet(PARQUET_COMPLETO, engine="pyarrow")

    # Passo 1-3: Calcular cinemática, entropia e SL/TP
    df_calc = calcular_indicadores_momentum(df_hurst)

    # Passo 4: Gerar os sinais operacionais e estatísticas
    df_final, stats_sinais = gerar_sinais_momentum(df_calc)

    # Salvar cache Parquet
    logger.info(f"Salvando resultados no cache: {PARQUET_SAIDA.name}")
    df_final.to_parquet(PARQUET_SAIDA, engine="pyarrow", compression="snappy", index=True)

    # Passo 5: Gerar gráfico de sinais e relatório
    gerar_grafico_momentum_sinais(df_final)
    imprimir_relatorio_momentum(df_final, stats_sinais)

    return df_final

# =============================================================================
# FUNÇÃO UTILITÁRIA EXPORTÁVEL
# =============================================================================

def carregar_momentum() -> pd.DataFrame:
    """
    Carrega o parquet com momentum calculado.
    Pode ser importada por outros módulos do sistema.
    """
    if not PARQUET_SAIDA.exists():
        raise FileNotFoundError(
            f"Parquet com Momentum não encontrado: {PARQUET_SAIDA}\n"
            f"Execute primeiro: python momentum.py"
        )
    return pd.read_parquet(PARQUET_SAIDA, engine="pyarrow")

# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Módulo Momentum e Entropia — {ATIVO} {TIMEFRAME}",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--forcar",
        action="store_true",
        help="Forçar reprocessamento total da base de momentum"
    )
    args = parser.parse_args()

    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print(f"█   ESTRATÉGIA DE MOMENTUM E ENTROPIA {ATIVO} v2                 █")
    print("█   Filtros: Hurst Tendência + Entropia Shannon < 0.60            █")
    print("█   Sinais: Cinemática da Aceleração (Janela 100)                  █")
    print("█" + " " * 68 + "█")
    print("█" * 70)

    processar_pipeline_momentum(args.forcar)
    print("✅ Módulo executado com sucesso!\n")
