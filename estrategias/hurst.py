# -*- coding: utf-8 -*-
"""
================================================
hurst.py — Estratégia Quantitativa
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
  3. Executar: python hurst.py
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
PARQUET_SAIDA       = DIR_DATA / f"{ATIVO.lower()}_{TIMEFRAME.lower()}_hurst.parquet"
CAMINHO_GRAFICO     = DIR_GRAFICOS / f"{ATIVO.lower()}_{TIMEFRAME.lower()}_hurst_sinais.png"

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
# CONSTANTES E PARÂMETROS
# =============================================================================

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

    Parâmetros:
        retornos: Array NumPy de tamanho JANELA_PRINCIPAL (100)

    Retorna:
        Expoente de Hurst (float) ou np.nan se houver erro ou inconsistência
    """
    if len(retornos) < JANELA_PRINCIPAL:
        return np.nan

    log_n = []
    log_rs = []

    for n in SUB_JANELAS:
        # Passo 1 — Dividir retornos em segmentos sem sobreposição
        num_segmentos = JANELA_PRINCIPAL // n
        rs_segmentos = []

        for k in range(num_segmentos):
            segmento = retornos[k * n : (k + 1) * n]
            
            # Passo 2 — Calcular R/S para o segmento
            mu = segmento.mean()
            y_t = np.cumsum(segmento - mu)
            
            r_range = y_t.max() - y_t.min()
            s_std = segmento.std(ddof=1)
            
            # Validação: Retornos constantes (S == 0) implicam em divisão por zero
            if s_std == 0:
                return np.nan
                
            rs = r_range / s_std
            rs_segmentos.append(rs)

        # Passo 3 — RS médio dos segmentos para este n
        rs_medio = np.mean(rs_segmentos)
        if rs_medio > 0:
            log_n.append(np.log(n))
            log_rs.append(np.log(rs_medio))

    # Passo 4 — Regressão linear OLS log(RS_medio) ~ log(n)
    if len(log_n) < 2:
        return np.nan

    h = _quick_ols_slope(np.array(log_n), np.array(log_rs))

    # Validação do Hurst obtido
    if np.isnan(h) or h < 0.0 or h > 1.5:
        return np.nan

    return h

def calcular_half_life_janela(retornos: np.ndarray) -> float:
    """
    Estima a Meia-Vida de Reversão à Média via regressão AR(1) dos retornos.
    Modelo: r_t = c + beta * r_{t-1} + epsilon_t
    Taxa de reversão theta = -ln(beta)
    Meia-vida = ln(2) / theta

    Parâmetros:
        retornos: Array NumPy contendo retornos consecutivos da janela

    Retorna:
        Meia-vida em número de candles (float) ou np.nan
    """
    if len(retornos) < JANELA_PRINCIPAL:
        return np.nan

    x = retornos[:-1]
    y = retornos[1:]

    # Regressão AR(1)
    beta = _quick_ols_slope(x, y)

    # Reversão à média requer que 0 < beta < 1
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

    Parâmetros:
        df: DataFrame contendo a série de preços EURUSD H1 completa
        janela: Janela principal de cálculo (100)

    Retorna:
        DataFrame com colunas 'hurst', 'regime' e 'half_life' adicionadas
    """
    logger.info(f"Iniciando cálculo móvel (rolling) de Hurst na janela de {janela} candles...")
    
    # Extrair log-retornos como array NumPy para alta performance
    retornos = df["log_return"].to_numpy()
    n_candles = len(df)

    hurst_values = np.full(n_candles, np.nan, dtype=np.float32)
    half_life_values = np.full(n_candles, np.nan, dtype=np.float32)

    # Loop móvel otimizado com barra de progresso tqdm
    # Primeiros 99 candles ficam como NaN pois necessitam de janela cheia (100)
    for i in tqdm(range(janela - 1, n_candles), desc="Processando Regime Hurst"):
        janela_retornos = retornos[i - janela + 1 : i + 1]
        
        # Calcular Hurst
        h = calcular_hurst_janela(janela_retornos)
        hurst_values[i] = h
        
        # Calcular Meia-Vida
        hl = calcular_half_life_janela(janela_retornos)
        half_life_values[i] = hl

    # Adicionar colunas ao DataFrame original
    df["hurst"] = hurst_values
    df["half_life"] = half_life_values

    # Classificar o regime de mercado com base nos thresholds
    logger.info("Classificando regimes de mercado...")
    df["regime"] = "INDEFINIDO"
    df.loc[df["hurst"] < LIMIAR_REVERSAO, "regime"] = "REVERSAO"
    df.loc[df["hurst"] > LIMIAR_TENDENCIA, "regime"] = "TENDENCIA"
    df.loc[df["hurst"].isna(), "regime"] = np.nan

    # Converter para tipo categórico otimizado
    df["regime"] = df["regime"].astype("category")

    return df

# =============================================================================
# RELATÓRIO DE SAÍDA E ESTATÍSTICAS
# =============================================================================

def imprimir_relatorio_hurst(df: pd.DataFrame):
    """Gera e imprime na tela o relatório detalhado do árbitro de regime de Hurst."""
    sep = "═" * 70
    sub_sep = "─" * 70
    
    df_valido = df.dropna(subset=["hurst"])
    total_validos = len(df_valido)
    
    if total_validos == 0:
        logger.error("Não há dados válidos de Hurst para gerar o relatório!")
        return

    # SEÇÃO 1 — Parâmetros utilizados
    print(f"\n{sep}")
    print("  SEÇÃO 1 — PARÂMETROS UTILIZADOS")
    print(sep)
    print(f"  Janela Principal de Retornos          : {JANELA_PRINCIPAL} H1 candles")
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
    print(f"  Meia-Vida Média de Reversão          : {hl_mean:>12.2f} candles H1")
    print(f"  Meia-Vida Mediana de Reversão        : {hl_median:>12.2f} candles H1")

    # SEÇÃO 4 — Distribuição de regimes por hora do dia (00h até 23h)
    print(f"\n{sep}")
    print("  SEÇÃO 4 — DISTRIBUIÇÃO DE REGIMES POR HORA DO DIA (Servidor MT5)")
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
            
            # Identificar o maior
            regimes_pct = {"REVERSÃO": rev_h, "TENDÊNCIA": ten_h, "INDEFINIDO": ind_h}
            dominante = max(regimes_pct, key=regimes_pct.get)
            
            # Detalhe visual para a madrugada
            madrugada_tag = " 🌙" if h < 10 else " ☀️"
            
            print(f"    {h:02d}h{madrugada_tag} │   {rev_h:>5.1f}%   │   {ten_h:>5.1f}%   │   {ind_h:>5.1f}%   │ {dominante}")

    # SEÇÃO 5 — Distribuição de regimes por dia da semana
    print(f"\n{sep}")
    print("  SEÇÃO 5 — DISTRIBUIÇÃO DE REGIMES POR DIA DA SEMANA")
    print(sep)
    print("   Dia da Semana │  REVERSÃO  │  TENDÊNCIA  │ INDEFINIDO │ regime Dominante")
    print("  " + "─" * 66)
    
    for wd in range(5):  # Segunda (0) a Sexta (4)
        df_dia = df_valido[df_valido.index.weekday == wd]
        t_dia = len(df_dia)
        if t_dia > 0:
            rev_d = (df_dia["regime"] == "REVERSAO").sum() / t_dia * 100
            ten_d = (df_dia["regime"] == "TENDENCIA").sum() / t_dia * 100
            ind_d = (df_dia["regime"] == "INDEFINIDO").sum() / t_dia * 100
            
            regimes_pct = {"REVERSÃO": rev_d, "TENDÊNCIA": ten_d, "INDEFINIDO": ind_d}
            dominante = max(regimes_pct, key=regimes_pct.get)
            nome_dia = DIAS_SEMANA[wd]
            
            print(f"   {nome_dia:13s} │   {rev_d:>5.1f}%   │   {ten_d:>5.1f}%   │   {ind_d:>5.1f}%   │ {dominante}")

    # SEÇÃO 6 — Evolução anual do Hurst (2016 a 2026)
    print(f"\n{sep}")
    print("  SEÇÃO 6 — EVOLUÇÃO ANUAL DO EXPOENTE DE HURST")
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
    EURUSD com coloração por regime e a evolução do Hurst.
    Salva em graficos/hurst.png.
    """
    logger.info("Gerando gráfico estatístico em Dark Mode...")
    
    # Criar pasta de gráficos se não existir
    DIR_GRAFICOS.mkdir(parents=True, exist_ok=True)
    
    df_plot = df.dropna(subset=["hurst"])
    if len(df_plot) == 0:
        logger.error("Sem dados de Hurst suficientes para plotar!")
        return

    # Ativar estilo escuro
    plt.style.use('dark_background')
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [3, 2]})
    fig.suptitle(f"{ATIVO} {TIMEFRAME} — Árbitro de Regime por Expoente de Hurst", fontsize=16, fontweight='bold', color='#FFFFFF')
    
    # Cores harmoniosas (Tailored HSL/Hex)
    cor_tendencia = '#00E676'   # Verde Neon Translúcido
    cor_reversao = '#FF1744'    # Vermelho Neon Translúcido
    cor_indefinido = '#555555'  # Cinza Médio Translúcido
    
    # -------------------------------------------------------------------------
    # PAINEL 1 — Preço Close EURUSD com coloração por Regime
    # -------------------------------------------------------------------------
    ax1.plot(df_plot.index, df_plot["Close"], color='#ECEFF1', linewidth=1.2, label=f'Preço {ATIVO}')
    ax1.set_title("Identificação de Regime sobre a Série de Preços", fontsize=12, fontweight='semibold', color='#CFD8DC')
    ax1.set_ylabel(f"{ATIVO} {TIMEFRAME} Close", fontsize=11, color='#CFD8DC')
    ax1.grid(True, linestyle='--', alpha=0.15)
    
    # Preenchimento de regime de fundo
    # Otimização por blocos contínuos de regime para evitar lentidão extrema no plot
    regimes = df_plot["regime"].to_numpy()
    times = df_plot.index
    
    # Detectar transições de regime
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
    # PAINEL 2 — Expoente de Hurst ao longo do tempo
    # -------------------------------------------------------------------------
    ax2.plot(df_plot.index, df_plot["hurst"], color='#64B5F6', linewidth=1.0, label='Hurst (Janela=100)')
    ax2.axhline(LIMIAR_REVERSAO, color=cor_reversao, linestyle='--', alpha=0.6, linewidth=1.0, label='Limiar Reversão (0.45)')
    ax2.axhline(LIMIAR_TENDENCIA, color=cor_tendencia, linestyle='--', alpha=0.6, linewidth=1.0, label='Limiar Tendência (0.55)')
    
    # Preenchimento das zonas de Hurst
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
    
    # Ajustes finos de layout
    fig.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Salvar o gráfico
    plt.savefig(CAMINHO_GRAFICO, dpi=150, facecolor='#121212')
    plt.close()
    
    logger.info(f"Gráfico analítico salvo com sucesso em: {CAMINHO_GRAFICO.resolve()}")

# =============================================================================
# PIPELINE DE EXECUÇÃO E CACHE
# =============================================================================

def processar_pipeline_hurst(forcar: bool = False) -> pd.DataFrame:
    """
    Controla o pipeline de carregamento, cálculo e cache do módulo hurst.
    """
    # Se já existir cache e não forçar, carregar direto
    if PARQUET_SAIDA.exists() and not forcar:
        logger.info("Cache encontrado. Carregando dados de Hurst pré-calculados...")
        df = pd.read_parquet(PARQUET_SAIDA, engine="pyarrow")
        
        # Gerar o gráfico também se não existir, mesmo com cache
        if not CAMINHO_GRAFICO.exists():
            gerar_grafico_regime(df)
            
        imprimir_relatorio_hurst(df)
        return df

    # Caso contrário, processar do zero
    if not PARQUET_COMPLETO.exists():
        raise FileNotFoundError(
            f"Arquivo de série completa não encontrado: {PARQUET_COMPLETO}\n"
            f"Por favor, execute primeiro o data_loader.py do projeto."
        )

    logger.info(f"Reprocessando base completa do {ATIVO} {TIMEFRAME} para o cálculo de Hurst...")
    df_completo = pd.read_parquet(PARQUET_COMPLETO, engine="pyarrow")

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
# FUNÇÕES UTILIÁRIAS EXPORTÁVEIS
# =============================================================================

def carregar_hurst() -> pd.DataFrame:
    """
    Carrega o parquet com Hurst calculado.
    Pode ser importada por outros módulos do sistema.
    """
    if not PARQUET_SAIDA.exists():
        raise FileNotFoundError(
            f"Parquet com Hurst não encontrado: {PARQUET_SAIDA}\n"
            f"Execute: python hurst.py"
        )
    return pd.read_parquet(PARQUET_SAIDA, engine="pyarrow")

def get_regime_atual(df_hurst: pd.DataFrame, datetime) -> str:
    """
    Retorna o regime de mercado no datetime naive especificado.
    """
    if datetime not in df_hurst.index:
        return "INDEFINIDO"
    val = df_hurst.loc[datetime, "regime"]
    return str(val) if pd.notna(val) else "INDEFINIDO"

def is_reversao(df_hurst: pd.DataFrame, datetime) -> bool:
    """
    Verifica se o mercado está em regime de REVERSÃO no datetime especificado.
    """
    return get_regime_atual(df_hurst, datetime) == "REVERSAO"

def is_tendencia(df_hurst: pd.DataFrame, datetime) -> bool:
    """
    Verifica se o mercado está em regime de TENDÊNCIA no datetime especificado.
    """
    return get_regime_atual(df_hurst, datetime) == "TENDENCIA"

# =============================================================================
# INTERFACE CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Árbitro de Regime {ATIVO} {TIMEFRAME} — Cálculo do Expoente de Hurst e Meia-Vida",
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
    print(f"█   ÁRBITRO DE REGIME {ATIVO} v2 — EXPOENTE DE HURST              █")
    print("█   Método R/S móvel (Janela 100)                                 █")
    print("█   Classificação: REVERSÃO (<0.45) │ TENDÊNCIA (>0.55)           █")
    print("█" + " " * 68 + "█")
    print("█" * 70)

    processar_pipeline_hurst(args.forcar)
    print("✅ Módulo executado com sucesso!\n")
