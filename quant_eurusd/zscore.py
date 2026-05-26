# -*- coding: utf-8 -*-
"""
================================================================================
zscore.py — Módulo de Z-Score e Sinais de Mean Reversion
================================================================================
Autor: Quant Developer Sênior
Data: 2026

Descrição:
    Este módulo calcula o Z-Score do preço de fechamento em janela móvel de
    50 candles H1 e gera sinais de entrada/saída para a estratégia de
    reversão à média (Mean Reversion).

    CONDIÇÃO OBRIGATÓRIA: o sinal só é gerado quando o regime Hurst = "REVERSAO".
    Em outros regimes, nenhum sinal é emitido, independentemente do Z-Score.

    Fórmula do Z-Score:
        Z_t = (Close_t - mu_N) / sigma_N

    Onde:
        mu_N    = média dos últimos N=50 fechamentos (rolling mean)
        sigma_N = desvio padrão dos últimos N=50 fechamentos (rolling std, ddof=1)

    Regras de Sinal:
        COMPRA  (LONG):  Z_t <= -3.0  E regime = "REVERSAO"
        VENDA   (SHORT): Z_t >= +3.0  E regime = "REVERSAO"
        NEUTRO:  caso contrário

    Gestão de Risco:
        Stop Loss  = 2.0 * VR_pips  (volatilidade realizada em pips)
        Take Profit = 3.0 * VR_pips  (R:R mínimo de 1:1.5)

Fluxo de processamento:
    1. Carregar dados H1 com Hurst (eurusd_h1_hurst.parquet)
    2. Calcular Z-Score em janela móvel de 50 candles
    3. Calcular Volatilidade Realizada (VR) em 50 candles
    4. Derivar Stop Loss e Take Profit em pips
    5. Gerar sinais condicionados ao regime Hurst
    6. Salvar DataFrame em Parquet
    7. Gerar gráfico de 3 painéis
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

PARQUET_ENTRADA = DIR_DATA / "eurusd_h1_hurst.parquet"
PARQUET_SAIDA   = DIR_DATA / "eurusd_h1_zscore.parquet"
GRAFICO_SAIDA   = DIR_GRAFICOS / "zscore_sinais.png"

# Parâmetros do Z-Score
JANELA_ZSCORE      = 50    # Candles H1 para rolling mean/std do preço
JANELA_VR          = 50    # Candles H1 para cálculo de Volatilidade Realizada

# Limiares do Z-Score para geração de sinais
Z_ENTRADA_COMPRA   = -3.0  # Z <= -3.0 → sinal de LONG
Z_ENTRADA_VENDA    =  3.0  # Z >= +3.0 → sinal de SHORT
Z_STOP_COMPRA      = -3.5  # Stop loss para LONG (Z abaixo do stop)
Z_STOP_VENDA       =  3.5  # Stop loss para SHORT (Z acima do stop)
Z_FECHAMENTO_MIN   = -0.5  # Fechar posição quando Z retornar à zona neutra
Z_FECHAMENTO_MAX   =  0.5

# Parâmetros de gestão de risco
MULT_STOP_PIPS     = 2.0   # SL = 2.0 * VR_pips
MULT_TP_PIPS       = 3.0   # TP = 3.0 * VR_pips (R:R de 1:1.5)

# Constante para conversão de retornos para pips no EURUSD
# 1 pip EURUSD = 0.0001; multiplicar por 10.000 → pips
FATOR_PIPS_EURUSD  = 10_000

# Paleta de cores do gráfico (dark mode)
COR_FUNDO       = "#0D1117"
COR_TEXTO       = "#E6EDF3"
COR_GRADE       = "#21262D"
COR_PRECO       = "#58A6FF"
COR_ZSCORE      = "#A371F7"  # Roxo para Z-Score
COR_COMPRA      = "#3FB950"  # Verde para sinais de compra
COR_VENDA       = "#F85149"  # Vermelho para sinais de venda
COR_REVERSAO    = "#F85149"
COR_TENDENCIA   = "#3FB950"
COR_INDEFINIDO  = "#6E7681"
COR_ZONA_VENDA  = "#3D1C1C"  # Fundo zona sobrecomprado
COR_ZONA_COMPRA = "#1C3D2E"  # Fundo zona sobrevendido


# =============================================================================
# CÁLCULOS ESTATÍSTICOS
# =============================================================================

def calcular_zscore_rolling(
    preco_close: pd.Series,
    janela: int = JANELA_ZSCORE,
) -> pd.Series:
    """
    Calcula o Z-Score do preço de fechamento em janela móvel.

    Fórmula:
        Z_t = (Close_t - mu_N) / sigma_N

    Onde:
        mu_N    = média aritmética dos últimos N fechamentos
        sigma_N = desvio padrão amostral (ddof=1) dos últimos N fechamentos

    IMPORTANTE: O Z-Score é calculado sobre o PREÇO (Close), não sobre
    os log-retornos. Isso mede o desvio do preço atual em relação à sua
    média recente, em unidades de desvio padrão. É adequado para Mean
    Reversion porque assume que o preço tende a retornar à sua média.

    Parâmetros:
        preco_close: pd.Series — série de preços de fechamento H1
        janela: int — tamanho da janela móvel (padrão: 50)

    Retorna:
        pd.Series — série do Z-Score com mesmo índice de entrada
    """
    logger.info(
        f"Calculando Z-Score do preço (janela={janela} candles H1)..."
    )

    # Média móvel dos últimos N fechamentos
    media_rolling = preco_close.rolling(window=janela, min_periods=janela).mean()

    # Desvio padrão amostral (ddof=1 é o padrão do pandas rolling.std)
    std_rolling = preco_close.rolling(window=janela, min_periods=janela).std(ddof=1)

    # Z-Score: desvio do preço atual em relação à média, em sigma
    zscore = (preco_close - media_rolling) / std_rolling

    # Onde std = 0 (preço completamente constante), Z-Score é indefinido
    zscore[std_rolling == 0] = np.nan

    logger.info(
        f"Z-Score calculado: "
        f"min={zscore.min():.2f}, max={zscore.max():.2f}, "
        f"valores válidos={zscore.notna().sum():,}"
    )

    return zscore.astype("float32")


def calcular_volatilidade_realizada(
    log_retornos: pd.Series,
    preco_close: pd.Series,
    janela: int = JANELA_VR,
) -> tuple:
    """
    Calcula a Volatilidade Realizada (VR) dos log-retornos e converte para pips.

    A VR mede a dispersão real dos retornos nas últimas N horas, servindo
    como estimativa dinâmica de risco para dimensionar Stop Loss e Take Profit.

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
        f"Calculando Volatilidade Realizada (janela={janela} candles H1)..."
    )

    # Volatilidade Realizada = desvio padrão dos log-retornos na janela
    vr = log_retornos.rolling(window=janela, min_periods=janela).std(ddof=1)

    # Converter para pips: VR * Close * 10.000
    # Close é necessário porque a VR está em retornos percentuais (adimensional)
    vr_pips = vr * preco_close * FATOR_PIPS_EURUSD

    # Stop Loss e Take Profit
    sl_pips = MULT_STOP_PIPS * vr_pips
    tp_pips = MULT_TP_PIPS   * vr_pips

    return (
        vr_pips.astype("float32"),
        sl_pips.astype("float32"),
        tp_pips.astype("float32"),
    )


def gerar_sinais_zscore(
    zscore: pd.Series,
    regime: pd.Series,
) -> pd.Series:
    """
    Gera os sinais de entrada da estratégia Mean Reversion com base no Z-Score.

    PRÉ-CONDIÇÃO OBRIGATÓRIA: regime Hurst = "REVERSAO".
    Sem essa condição, nenhum sinal é gerado independentemente do Z-Score.

    Lógica de sinal:
        +1 (COMPRA/LONG):  Z_t <= -3.0  E  regime = "REVERSAO"
        -1 (VENDA/SHORT):  Z_t >= +3.0  E  regime = "REVERSAO"
         0 (NEUTRO):       qualquer outro caso

    Interpretação econômica:
        Z <= -3.0 significa que o preço está 3.0 desvios abaixo da média.
        Estatisticamente, preços tão distantes da média tendem a reverter.
        Esperamos que o preço suba de volta para a média (Z → 0).

        Z >= +3.0 é o oposto: preço extremamente acima da média → esperamos queda.

    Parâmetros:
        zscore: pd.Series — série do Z-Score calculado
        regime: pd.Series — série de regime Hurst ("REVERSAO", "TENDENCIA", etc.)

    Retorna:
        pd.Series de int8 com valores {-1, 0, 1}
    """
    logger.info("Gerando sinais Z-Score condicionados ao regime Hurst...")

    # Condição de regime ativo
    em_reversao = regime == "REVERSAO"

    # Sinais brutos (sem condição de regime)
    sinal_compra = (zscore <= Z_ENTRADA_COMPRA)
    sinal_venda  = (zscore >= Z_ENTRADA_VENDA)

    # Sinais finais: apenas quando em regime de reversão
    sinais = pd.Series(0, index=zscore.index, dtype="int8")
    sinais[em_reversao & sinal_compra] =  1
    sinais[em_reversao & sinal_venda]  = -1

    # Estatísticas dos sinais gerados
    n_compra = (sinais == 1).sum()
    n_venda  = (sinais == -1).sum()
    n_total  = len(sinais)

    logger.info(
        f"Sinais gerados: COMPRA={n_compra:,} ({n_compra/n_total*100:.2f}%), "
        f"VENDA={n_venda:,} ({n_venda/n_total*100:.2f}%), "
        f"NEUTRO={n_total - n_compra - n_venda:,}"
    )

    return sinais


# =============================================================================
# GERAÇÃO DO GRÁFICO
# =============================================================================

def gerar_grafico_zscore(df: pd.DataFrame, caminho: Path) -> None:
    """
    Gera gráfico de 3 painéis para visualização da estratégia Z-Score.

    Painel 1 (topo): Preço Close com marcadores de entrada
        - Triângulo verde para baixo (▼) = sinal de COMPRA (preço sobrevendido)
        - Triângulo vermelho para cima  (▲) = sinal de VENDA (preço sobrecomprado)

    Painel 2 (meio): Z-Score ao longo do tempo
        - Linha roxa: Z-Score
        - Linhas tracejadas em ±2.5 e ±3.5
        - Zona verde (Z < -2.5): sobrevendido (compra)
        - Zona vermelha (Z > +2.5): sobrecomprado (venda)

    Painel 3 (baixo): Regime Hurst colorido por categoria
        - Verde: TENDENCIA
        - Vermelho: REVERSAO
        - Cinza: INDEFINIDO

    Parâmetros:
        df: DataFrame com todas as colunas calculadas
        caminho: Path — destino do arquivo PNG
    """
    logger.info("Gerando gráfico de 3 painéis Z-Score + Sinais...")

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

    fig = plt.figure(figsize=(20, 13))
    gs = gridspec.GridSpec(3, 1, height_ratios=[2.5, 2, 1], hspace=0.06)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    fig.suptitle(
        "EURUSD H1 — Estratégia Mean Reversion por Z-Score\n"
        f"Janela: {JANELA_ZSCORE} candles | Entrada: |Z| >= {abs(Z_ENTRADA_VENDA):.1f} | Regime: REVERSAO",
        color=COR_TEXTO, fontsize=13, fontweight="bold", y=0.99,
    )

    # Subconjuntos de sinais para plotagem
    df_compra = df[df["sinal_zscore"] ==  1]
    df_venda  = df[df["sinal_zscore"] == -1]
    df_valido = df.dropna(subset=["zscore"])

    # ── Painel 1: Preço + Marcadores de Entrada ───────────────────────────────
    ax1.plot(df.index, df["Close"],
             color=COR_PRECO, linewidth=0.7, alpha=0.9, label="EURUSD Close")

    # Marcadores de COMPRA: triângulo apontando para cima (▲ verde)
    ax1.scatter(
        df_compra.index, df_compra["Close"],
        marker="^", color=COR_COMPRA, s=40, zorder=5,
        label=f"Compra (Z <= {Z_ENTRADA_COMPRA}) [{len(df_compra):,}]",
        alpha=0.85,
    )
    # Marcadores de VENDA: triângulo apontando para baixo (▼ vermelho)
    ax1.scatter(
        df_venda.index, df_venda["Close"],
        marker="v", color=COR_VENDA, s=40, zorder=5,
        label=f"Venda (Z >= +{Z_ENTRADA_VENDA}) [{len(df_venda):,}]",
        alpha=0.85,
    )

    ax1.set_ylabel("Preço (EURUSD)", color=COR_TEXTO, fontsize=10)
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.3)
    ax1.grid(True, linestyle="--", alpha=0.3)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # ── Painel 2: Z-Score ─────────────────────────────────────────────────────
    idx_v = df_valido.index
    zs    = df_valido["zscore"].values

    # Zonas coloridas de fundo
    ax2.fill_between(idx_v, zs, Z_ENTRADA_VENDA,
                     where=(zs >= Z_ENTRADA_VENDA),
                     color=COR_VENDA, alpha=0.15, label=f"Sobrecomprado (Z >= +{Z_ENTRADA_VENDA})")
    ax2.fill_between(idx_v, zs, Z_ENTRADA_COMPRA,
                     where=(zs <= Z_ENTRADA_COMPRA),
                     color=COR_COMPRA, alpha=0.15, label=f"Sobrevendido (Z <= {Z_ENTRADA_COMPRA})")

    # Linha do Z-Score
    ax2.plot(idx_v, zs, color=COR_ZSCORE, linewidth=0.7, alpha=0.9, label="Z-Score")

    # Linhas de referência tracejadas
    for nivel, cor, ls, lw, label in [
        ( Z_ENTRADA_VENDA,  COR_VENDA,    "--", 1.2, f"+{Z_ENTRADA_VENDA}  (Venda)"),
        ( Z_ENTRADA_COMPRA, COR_COMPRA,   "--", 1.2, f"{Z_ENTRADA_COMPRA}  (Compra)"),
        ( Z_STOP_VENDA,     COR_VENDA,    ":",  0.9, f"+{Z_STOP_VENDA}  (Stop Short)"),
        ( Z_STOP_COMPRA,    COR_COMPRA,   ":",  0.9, f"{Z_STOP_COMPRA}  (Stop Long)"),
        ( 0.0,              COR_TEXTO,    "-",  0.6, "0.0  (Média)"),
    ]:
        ax2.axhline(nivel, color=cor, linestyle=ls, linewidth=lw, alpha=0.8)
        ax2.text(
            idx_v[-1], nivel,
            f"  {label}", color=cor, fontsize=7,
            va="center", ha="right", alpha=0.9,
        )

    ax2.set_ylabel("Z-Score", color=COR_TEXTO, fontsize=10)
    ax2.set_ylim(max(zs.min() - 0.5, -6), min(zs.max() + 0.5, 6))
    ax2.legend(loc="upper left", fontsize=8, framealpha=0.3, ncol=3)
    ax2.grid(True, linestyle="--", alpha=0.3)
    plt.setp(ax2.get_xticklabels(), visible=False)

    # ── Painel 3: Regime Hurst ────────────────────────────────────────────────
    mapa_cor_regime = {
        "REVERSAO":   COR_REVERSAO,
        "TENDENCIA":  COR_TENDENCIA,
        "INDEFINIDO": COR_INDEFINIDO,
    }
    mapa_num_regime = {"REVERSAO": -1, "TENDENCIA": 1, "INDEFINIDO": 0}

    regime_num = df["regime"].map(mapa_num_regime).fillna(0)
    cores_regime = df["regime"].map(mapa_cor_regime).fillna(COR_INDEFINIDO)

    # Colorir barras por regime
    for regime_nome, cor in mapa_cor_regime.items():
        mask = df["regime"] == regime_nome
        if mask.any():
            ax3.bar(
                df.index[mask], [1] * mask.sum(),
                color=cor, alpha=0.6, width=0.04,
                label=regime_nome,
            )

    ax3.set_ylabel("Regime", color=COR_TEXTO, fontsize=9)
    ax3.set_xlabel("Data", color=COR_TEXTO, fontsize=10)
    ax3.set_yticks([])
    ax3.legend(loc="upper left", fontsize=8, framealpha=0.3, ncol=3)
    ax3.grid(False)

    fig.autofmt_xdate(rotation=30, ha="right")

    plt.savefig(caminho, dpi=150, bbox_inches="tight", facecolor=COR_FUNDO)
    plt.close()

    tamanho_kb = caminho.stat().st_size / 1024
    logger.info(f"Gráfico salvo: {caminho} ({tamanho_kb:.0f} KB)")


# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================

def calcular_e_salvar_zscore(
    data_inicio: str = None,
    data_fim: str = None,
    forcar_reprocessamento: bool = False,
) -> pd.DataFrame:
    """
    Pipeline principal do módulo Z-Score.

    Executa em sequência:
        1. Carrega dados H1 com Hurst (eurusd_h1_hurst.parquet)
        2. Calcula Z-Score do preço em janela de 50 candles
        3. Calcula Volatilidade Realizada e Stop/Target em pips
        4. Gera sinais condicionados ao regime Hurst = "REVERSAO"
        5. Salva DataFrame em Parquet
        6. Gera gráfico de 3 painéis
        7. Exibe resumo estatístico

    Parâmetros:
        data_inicio: str — filtro de início "YYYY-MM-DD" (opcional)
        data_fim: str — filtro de fim "YYYY-MM-DD" (opcional)
        forcar_reprocessamento: bool — recalcular mesmo que Parquet exista

    Retorna:
        DataFrame H1 com colunas Z-Score e sinais adicionadas

    Levanta:
        FileNotFoundError — se eurusd_h1_hurst.parquet não existir
    """
    DIR_GRAFICOS.mkdir(parents=True, exist_ok=True)
    DIR_DATA.mkdir(parents=True, exist_ok=True)

    # ── Cache ─────────────────────────────────────────────────────────────────
    if PARQUET_SAIDA.exists() and not forcar_reprocessamento:
        logger.info(
            f"Parquet Z-Score já existe. Carregando cache... "
            f"(use --forcar para recalcular)"
        )
        df = pd.read_parquet(PARQUET_SAIDA, engine="pyarrow")
        if data_inicio:
            df = df[df.index >= data_inicio]
        if data_fim:
            df = df[df.index <= data_fim]
        _exibir_resumo_zscore(df)
        return df

    # ── Passo 1: Carregar dados com Hurst ─────────────────────────────────────
    if not PARQUET_ENTRADA.exists():
        raise FileNotFoundError(
            f"\n{'='*60}\n"
            f"  ERRO: Parquet de entrada não encontrado!\n"
            f"  Esperado em: {PARQUET_ENTRADA}\n"
            f"\n"
            f"  Solução: Execute os módulos em ordem:\n"
            f"    1. python data_loader.py\n"
            f"    2. python hurst.py\n"
            f"    3. python zscore.py  (este módulo)\n"
            f"{'='*60}\n"
        )

    logger.info(f"Carregando dados H1 com Hurst: {PARQUET_ENTRADA.name}")
    df = pd.read_parquet(PARQUET_ENTRADA, engine="pyarrow")
    logger.info(f"Dados carregados: {len(df):,} candles H1")

    if data_inicio:
        df = df[df.index >= data_inicio]
    if data_fim:
        df = df[df.index <= data_fim]

    # ── Passo 2: Calcular Z-Score do preço ────────────────────────────────────
    df["zscore"] = calcular_zscore_rolling(df["Close"], janela=JANELA_ZSCORE)

    # ── Passo 3: Calcular Volatilidade Realizada e stops em pips ─────────────
    df["vr_pips"], df["sl_pips"], df["tp_pips"] = calcular_volatilidade_realizada(
        df["log_return"], df["Close"], janela=JANELA_VR
    )

    # ── Passo 4: Gerar sinais condicionados ao regime ─────────────────────────
    df["sinal_zscore"] = gerar_sinais_zscore(df["zscore"], df["regime"])

    # ── Passo 5: Salvar Parquet ────────────────────────────────────────────────
    df.to_parquet(PARQUET_SAIDA, engine="pyarrow", compression="snappy", index=True)
    tamanho_mb = PARQUET_SAIDA.stat().st_size / 1024 / 1024
    logger.info(f"Parquet salvo: {PARQUET_SAIDA.name} ({tamanho_mb:.1f} MB)")

    # ── Passo 6: Gerar gráfico ────────────────────────────────────────────────
    gerar_grafico_zscore(df, GRAFICO_SAIDA)

    # ── Passo 7: Exibir resumo ────────────────────────────────────────────────
    _exibir_resumo_zscore(df)

    return df


def _exibir_resumo_zscore(df: pd.DataFrame) -> None:
    """
    Exibe no terminal um resumo da estratégia Z-Score Mean Reversion.

    Inclui:
    - Contagem de sinais gerados por tipo (COMPRA, VENDA)
    - Percentual do tempo com sinal ativo
    - Estatísticas do Z-Score (min, max, std)
    - Estatísticas de Stop Loss e Take Profit em pips

    Parâmetros:
        df: DataFrame com colunas 'zscore', 'sinal_zscore', 'sl_pips', 'tp_pips'
    """
    df_v = df.dropna(subset=["zscore"])
    n_total  = len(df_v)
    n_compra = (df_v["sinal_zscore"] ==  1).sum()
    n_venda  = (df_v["sinal_zscore"] == -1).sum()
    n_sinais = n_compra + n_venda

    sl_medio = df_v["sl_pips"].mean() if "sl_pips" in df_v.columns else float("nan")
    tp_medio = df_v["tp_pips"].mean() if "tp_pips" in df_v.columns else float("nan")

    print("\n" + "=" * 60)
    print("  RESUMO DA ESTRATÉGIA Z-SCORE MEAN REVERSION — EURUSD H1")
    print("=" * 60)
    print(f"  Período analisado: "
          f"{df_v.index.min().date()} -> {df_v.index.max().date()}")
    print(f"  Candles com Z-Score válido: {n_total:>8,}")
    print()
    print(f"  ── Sinais Gerados (regime REVERSAO) ──")
    print(f"  COMPRA  (Z <= {Z_ENTRADA_COMPRA})     : "
          f"{n_compra:>5,} sinais ({n_compra/n_total*100:.2f}%)")
    print(f"  VENDA   (Z >= +{Z_ENTRADA_VENDA})     : "
          f"{n_venda:>5,} sinais ({n_venda/n_total*100:.2f}%)")
    print(f"  TOTAL de sinais             : {n_sinais:>5,} ({n_sinais/n_total*100:.2f}%)")
    print()
    print(f"  ── Z-Score Estatísticas ──")
    print(f"  Mínimo Z                    : {df_v['zscore'].min():>+.3f}")
    print(f"  Máximo Z                    : {df_v['zscore'].max():>+.3f}")
    print(f"  Desvio Padrão Z             : {df_v['zscore'].std():>.4f}")
    print()
    print(f"  ── Gestão de Risco (média) ──")
    print(f"  Stop Loss médio             : {sl_medio:>6.1f} pips")
    print(f"  Take Profit médio           : {tp_medio:>6.1f} pips")
    print(f"  R:R médio                   : 1:{MULT_TP_PIPS/MULT_STOP_PIPS:.1f}")
    print()
    print(f"  Gráfico salvo em: graficos/zscore_sinais.png")
    print(f"  Parquet salvo em: data/eurusd_h1_zscore.parquet")
    print("=" * 60 + "\n")


# =============================================================================
# EXECUÇÃO DIRETA (python zscore.py)
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Z-Score Mean Reversion com filtro de regime Hurst — EURUSD H1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python zscore.py
  python zscore.py --inicio 2018-01-01 --fim 2026-04-10
  python zscore.py --forcar
        """
    )
    parser.add_argument("--inicio", type=str, default=None,
                        help="Data de início (YYYY-MM-DD)")
    parser.add_argument("--fim", type=str, default=None,
                        help="Data de fim (YYYY-MM-DD)")
    parser.add_argument("--forcar", action="store_true",
                        help="Forçar recálculo mesmo se Parquet existir")

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  QUANT EURUSD — MÓDULO: Z-SCORE MEAN REVERSION")
    print("  Z-Score em Janela Movel + Sinais Condicionados ao Regime")
    print("=" * 60)

    df = calcular_e_salvar_zscore(
        data_inicio=args.inicio,
        data_fim=args.fim,
        forcar_reprocessamento=args.forcar,
    )

    print(f"\nConcluido! Proximo passo: execute momentum.py\n")
