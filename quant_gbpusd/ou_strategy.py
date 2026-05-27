# -*- coding: utf-8 -*-
"""
================================================================================
ou_strategy.py — Módulo de Reversão à Média por Processo Ornstein-Uhlenbeck (OU)
================================================================================

Objetivo:
    Implementar uma estratégia de reversão à média baseada no processo de
    Ornstein-Uhlenbeck (OU), operando de forma totalmente independente de
    Hurst e Z-Score clássico.

Regras de Operação:
    - O próprio processo OU detecta se o mercado é mean-reverting no momento.
    - Indicadores calculados sobre a série COMPLETA (64.002 candles).
    - Geração de sinais de entrada restrita à janela OPERACIONAL (10h00-22h30 seg-sex).
    - Gestão de risco monitorada 24h na série completa.

Saída:
    - data/gbpusd_h1_ou.parquet
    - graficos/ou_sinais.png (Gráfico de 4 painéis em Dark Mode)

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
from scipy import stats
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
# CONSTANTES E CAMINHOS
# =============================================================================
DIR_PROJETO = Path(__file__).resolve().parent
DIR_DATA    = DIR_PROJETO / "data"
DIR_GRAFICOS = DIR_PROJETO / "graficos"

PARQUET_ENTRADA = DIR_DATA / "gbpusd_h1_completo.parquet"
PARQUET_SAIDA   = DIR_DATA / "gbpusd_h1_ou.parquet"
CAMINHO_GRAFICO = DIR_GRAFICOS / "ou_sinais.png"

# =============================================================================
# REGRESSÃO OLS RÁPIDA E ESTATÍSTICAS
# =============================================================================

def _quick_ols_stats(x: np.ndarray, y: np.ndarray) -> tuple:
    """
    Calcula OLS de forma analítica e extremamente rápida usando NumPy.
    Retorna (slope, intercept, p_value, sigma_resid, residuos) ou NaNs em caso de erro.
    """
    try:
        n = len(x)
        mx = x.mean()
        my = y.mean()
        xm = x - mx
        ym = y - my
        
        ss_xx = np.sum(xm ** 2)
        if ss_xx <= 0:
            return np.nan, np.nan, np.nan, np.nan, None
            
        ss_xy = np.sum(xm * ym)
        slope = ss_xy / ss_xx
        intercept = my - slope * mx
        
        # Resíduos e desvio padrão residual
        y_pred = intercept + slope * x
        residuos = y - y_pred
        rss = np.sum(residuos ** 2)
        
        df_resid = n - 2
        if df_resid <= 0:
            return slope, intercept, np.nan, np.nan, residuos
            
        var_resid = rss / df_resid
        sigma_resid = np.sqrt(var_resid)
        
        # Erro padrão do slope e estatística t
        se_slope = np.sqrt(var_resid / ss_xx)
        if se_slope <= 0:
            return slope, intercept, 0.0, sigma_resid, residuos
            
        t_stat = slope / se_slope
        p_value = 2.0 * stats.t.sf(np.abs(t_stat), df=df_resid)
        
        return slope, intercept, p_value, sigma_resid, residuos
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, None

# =============================================================================
# FUNÇÕES DE FILTRAGEM OPERACIONAL
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
# PIPELINE ROLLING COMPLETO
# =============================================================================

def calcular_ou_rolling(df: pd.DataFrame, janela: int = 100) -> tuple:
    """
    Executa a regressão rolante do processo Ornstein-Uhlenbeck (OU) sobre o DataFrame.
    """
    logger.info(f"Iniciando cálculo rolling do processo Ornstein-Uhlenbeck (janela={janela})...")
    
    # Criar log-preço
    log_preco = np.log(df["Close"].values).astype("float32")
    df["log_preco"] = log_preco
    
    n_rows = len(df)
    
    # Inicializar arrays de saída
    ou_theta    = np.full(n_rows, np.nan, dtype=np.float32)
    ou_mu       = np.full(n_rows, np.nan, dtype=np.float32)
    ou_sigma_eq = np.full(n_rows, np.nan, dtype=np.float32)
    ou_halflife = np.full(n_rows, np.nan, dtype=np.float32)
    ou_zscore   = np.full(n_rows, np.nan, dtype=np.float32)
    ou_valido   = np.zeros(n_rows, dtype=bool)
    
    # Contadores de invalidação
    stats_invalido = {
        "total_analisados": 0,
        "p_value_gt_005": 0,
        "beta_ge_1": 0,
        "beta_le_0": 0,
        "halflife_gt_50": 0,
        "halflife_lt_1": 0,
    }
    
    # Contador de sinais bloqueados por OU inválido
    bloqueados_ou_invalido_count = 0
    
    # Janela operacional em formato booleano para o loop
    op_window = verificar_janela_operacional(df.index).values
    
    # Loop rolling utilizando tqdm
    for t in tqdm(range(janela - 1, n_rows), desc="Processando OU"):
        stats_invalido["total_analisados"] += 1
        
        # Fatiar janela de log-preços
        w = log_preco[t - (janela - 1): t + 1]
        
        # Y = X_t (t=2..N), X = X_{t-1} (t=1..N-1)
        y = w[1:]
        x = w[:-1]
        
        slope, intercept, p_value, sigma_resid, residuos = _quick_ols_stats(x, y)
        
        # Se falhar a regressão
        if np.isnan(slope) or np.isnan(intercept) or np.isnan(p_value) or np.isnan(sigma_resid):
            stats_invalido["p_value_gt_005"] += 1  # Considerado falha de significância
            continue
            
        beta = slope
        alpha = intercept
        
        # Validação passo a passo
        if beta <= 0:
            stats_invalido["beta_le_0"] += 1
            # Cálculo bruto para sinal bloqueado por OU inválido
            continue
            
        if beta >= 1:
            stats_invalido["beta_ge_1"] += 1
            continue
            
        # Parâmetro Theta
        theta = -np.log(beta)
        if theta <= 0:
            stats_invalido["beta_ge_1"] += 1  # Theta <= 0 equivale a beta >= 1
            continue
            
        # P-valor
        if p_value > 0.05:
            stats_invalido["p_value_gt_005"] += 1
            # Calcular sinal bloqueado por OU inválido se Z esticado
            mu_bruto = alpha / (1.0 - beta)
            den_bruto = 1.0 - np.exp(-2.0 * theta)
            if den_bruto > 0:
                sigma_eq_bruto = sigma_resid / np.sqrt(den_bruto)
                if sigma_eq_bruto > 0:
                    z_bruto = (w[-1] - mu_bruto) / sigma_eq_bruto
                    if op_window[t] and (z_bruto <= -2.0 or z_bruto >= 2.0):
                        bloqueados_ou_invalido_count += 1
            continue
            
        # Half-life
        half_life = np.log(2.0) / theta
        
        if half_life > 50.0:
            stats_invalido["halflife_gt_50"] += 1
            # Calcular sinal bloqueado por OU inválido se Z esticado
            mu_bruto = alpha / (1.0 - beta)
            den_bruto = 1.0 - np.exp(-2.0 * theta)
            if den_bruto > 0:
                sigma_eq_bruto = sigma_resid / np.sqrt(den_bruto)
                if sigma_eq_bruto > 0:
                    z_bruto = (w[-1] - mu_bruto) / sigma_eq_bruto
                    if op_window[t] and (z_bruto <= -2.0 or z_bruto >= 2.0):
                        bloqueados_ou_invalido_count += 1
            continue
            
        if half_life < 1.0:
            stats_invalido["halflife_lt_1"] += 1
            # Calcular sinal bloqueado por OU inválido se Z esticado
            mu_bruto = alpha / (1.0 - beta)
            den_bruto = 1.0 - np.exp(-2.0 * theta)
            if den_bruto > 0:
                sigma_eq_bruto = sigma_resid / np.sqrt(den_bruto)
                if sigma_eq_bruto > 0:
                    z_bruto = (w[-1] - mu_bruto) / sigma_eq_bruto
                    if op_window[t] and (z_bruto <= -2.0 or z_bruto >= 2.0):
                        bloqueados_ou_invalido_count += 1
            continue
            
        # Parâmetros válidos do processo OU
        mu = alpha / (1.0 - beta)
        sigma_eq = sigma_resid / np.sqrt(1.0 - np.exp(-2.0 * theta))
        z_ou = (w[-1] - mu) / sigma_eq
        
        # Preencher arrays de saída
        ou_theta[t]    = theta
        ou_mu[t]       = mu
        ou_sigma_eq[t] = sigma_eq
        ou_halflife[t] = half_life
        ou_zscore[t]   = z_ou
        ou_valido[t]   = True
        
    df["ou_theta"]    = ou_theta.astype(np.float32)
    df["ou_mu"]       = ou_mu.astype(np.float32)
    df["ou_sigma_eq"] = ou_sigma_eq.astype(np.float32)
    df["ou_halflife"] = ou_halflife.astype(np.float32)
    df["ou_zscore"]   = ou_zscore.astype(np.float32)
    df["ou_valido"]   = ou_valido
    
    return df, stats_invalido, bloqueados_ou_invalido_count

# =============================================================================
# VOLATILIDADE REALIZADA E GESTÃO DE RISCO
# =============================================================================

def calcular_gestao_risco(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula a Volatilidade Realizada e define Stop Loss E Take Profit dinâmicos.
    """
    logger.info("Calculando Volatilidade Realizada (janela=50)...")
    
    # Calcular log-retornos se não existirem
    if "log_return" not in df.columns:
        df["log_return"] = np.log(df["Close"] / df["Close"].shift(1)).astype(np.float32)
        
    vr = df["log_return"].rolling(window=50, min_periods=50).std(ddof=1)
    df["vr_pips"] = (vr * df["Close"] * 10000.0).astype(np.float32)
    df["sl_pips"] = (2.0 * df["vr_pips"]).astype(np.float32)
    df["tp_pips"] = (3.0 * df["vr_pips"]).astype(np.float32)
    
    return df

# =============================================================================
# GERAÇÃO DE SINAIS
# =============================================================================

def gerar_sinais_ou(df: pd.DataFrame, bloqueados_ou_invalido: int) -> tuple:
    """
    Gera os sinais operacionais de compra (LONG = +1) e venda (SHORT = -1)
    baseados no ou_zscore do processo Ornstein-Uhlenbeck.
    """
    logger.info("Executando motor de geração de sinais Ornstein-Uhlenbeck...")
    
    op_window = verificar_janela_operacional(df.index)
    
    # Condições de entrada obrigatórias
    cond_operacional = df["ou_valido"] & (df["ou_halflife"] >= 1.0) & (df["ou_halflife"] <= 50.0) & op_window
    
    sinal = np.zeros(len(df), dtype=np.int8)
    
    # Gatilho Retorno (cruza de volta o threshold 2.0)
    z = df["ou_zscore"].values
    z_prev = df["ou_zscore"].shift(1).values
    z_entry = 2.0
    
    cond_long = cond_operacional & (z_prev <= -z_entry) & (z > -z_entry) & (~df["ou_zscore"].isna()) & (~df["ou_zscore"].shift(1).isna())
    cond_short = cond_operacional & (z_prev >= z_entry) & (z < z_entry) & (~df["ou_zscore"].isna()) & (~df["ou_zscore"].shift(1).isna())
    
    sinal[cond_long] = 1
    sinal[cond_short] = -1
    
    df["sinal_ou"] = sinal
    
    # --- Estatísticas de Bloqueio ---
    # Sinais potenciais (Z esticado no horário operacional e processo OU válido)
    potenciais_validos = df["ou_valido"] & (df["ou_halflife"] >= 1.0) & (df["ou_halflife"] <= 50.0) & ((df["ou_zscore"] <= -2.0) | (df["ou_zscore"] >= 2.0))
    bloqueados_horario = potenciais_validos & (~op_window)
    
    stats_sinais = {
        "compra": int(np.sum(sinal == 1)),
        "venda": int(np.sum(sinal == -1)),
        "bloqueado_horario": int(np.sum(bloqueados_horario)),
        "bloqueado_ou_invalido": bloqueados_ou_invalido,
    }
    
    return df, stats_sinais

# =============================================================================
# RELATÓRIO DE SAÍDA
# =============================================================================

def imprimir_relatorio_ou(df: pd.DataFrame, stats_invalido: dict, stats_sinais: dict):
    """
    Imprime no console o relatório estatístico completo estruturado em 5 seções.
    """
    sep = "═" * 70
    sub_sep = "─" * 70
    
    df_valid = df[df["ou_valido"]]
    total_validos = len(df_valid)
    total_analisados = stats_invalido["total_analisados"]
    total_invalidos = total_analisados - total_validos
    
    pct_validos = (total_validos / total_analisados * 100) if total_analisados > 0 else 0
    pct_invalidos = (total_invalidos / total_analisados * 100) if total_analisados > 0 else 0
    
    print(f"\n{sep}")
    print("  SEÇÃO 1 — COBERTURA DO PROCESSO ORNSTEIN-UHLENBECK")
    print(sep)
    print(f"  Total de candles analisados          : {total_analisados:>12,}")
    print(f"  Candles com OU válido                : {total_validos:>12,} ({pct_validos:.1f}%)")
    print(f"  Candles com OU inválido              : {total_invalidos:>12,} ({pct_invalidos:.1f}%)")
    print(f"  {sub_sep}")
    print("  Razões de invalidação mais frequentes:")
    print(f"    - p-valor > 0.05                  : {stats_invalido['p_value_gt_005']:>12,} candles")
    print(f"    - β >= 1 (Random Walk)             : {stats_invalido['beta_ge_1']:>12,} candles")
    print(f"    - β <= 0 (explosivo)               : {stats_invalido['beta_le_0']:>12,} candles")
    print(f"    - half-life > 50                   : {stats_invalido['halflife_gt_50']:>12,} candles")
    print(f"    - half-life < 1                    : {stats_invalido['halflife_lt_1']:>12,} candles")
    
    if total_validos == 0:
        logger.warning("Sem dados válidos suficientes para as demais seções do relatório.")
        print(f"{sep}\n")
        return
        
    # SEÇÃO 2 — Parâmetros OU (quando válido)
    theta_mean = df_valid["ou_theta"].mean()
    theta_median = df_valid["ou_theta"].median()
    theta_min = df_valid["ou_theta"].min()
    theta_max = df_valid["ou_theta"].max()
    
    mu_mean = df_valid["ou_mu"].mean()
    sigma_eq_mean = df_valid["ou_sigma_eq"].mean()
    
    hl_mean = df_valid["ou_halflife"].mean()
    hl_median = df_valid["ou_halflife"].median()
    hl_min = df_valid["ou_halflife"].min()
    hl_max = df_valid["ou_halflife"].max()
    
    # Distribuição do half-life
    hl = df_valid["ou_halflife"]
    p_1_5 = (hl <= 5.0).sum() / total_validos * 100
    p_5_10 = ((hl > 5.0) & (hl <= 10.0)).sum() / total_validos * 100
    p_10_20 = ((hl > 10.0) & (hl <= 20.0)).sum() / total_validos * 100
    p_20_50 = ((hl > 20.0) & (hl <= 50.0)).sum() / total_validos * 100
    
    print(f"\n{sep}")
    print("  SEÇÃO 2 — PARÂMETROS OU (Quando Válido)")
    print(sep)
    print(f"  θ Médio                             : {theta_mean:>12.6f}")
    print(f"  θ Mediano                           : {theta_median:>12.6f}")
    print(f"  θ Mínimo                            : {theta_min:>12.6f}")
    print(f"  θ Máximo                            : {theta_max:>12.6f}")
    print(f"  {sub_sep}")
    print(f"  μ Médio (Equilíbrio de Longo Prazo) : {mu_mean:>12.6f}")
    print(f"  σ_eq Médio (Desvio Estacionário)    : {sigma_eq_mean:>12.6f}")
    print(f"  {sub_sep}")
    print(f"  Half-Life Médio (candles H1)        : {hl_mean:>12.2f}")
    print(f"  Half-Life Mediano (candles H1)      : {hl_median:>12.2f}")
    print(f"  Half-Life Mínimo (candles H1)       : {hl_min:>12.2f}")
    print(f"  Half-Life Máximo (candles H1)       : {hl_max:>12.2f}")
    print(f"  {sub_sep}")
    print("  Distribuição do Half-Life:")
    print(f"    - 1 a 5 candles                   : {p_1_5:>11.1f}%")
    print(f"    - 5 a 10 candles                  : {p_5_10:>11.1f}%")
    print(f"    - 10 a 20 candles                 : {p_10_20:>11.1f}%")
    print(f"    - 20 a 50 candles                 : {p_20_50:>11.1f}%")
    
    # SEÇÃO 3 — ou_zscore estatísticas
    z_min = df_valid["ou_zscore"].min()
    z_max = df_valid["ou_zscore"].max()
    z_mean = df_valid["ou_zscore"].mean()
    z_std = df_valid["ou_zscore"].std()
    
    pct_z2 = (df_valid["ou_zscore"].abs() > 2.0).sum() / total_validos * 100
    pct_z3 = (df_valid["ou_zscore"].abs() > 3.0).sum() / total_validos * 100
    
    print(f"\n{sep}")
    print("  SEÇÃO 3 — OU_ZSCORE ESTATÍSTICAS")
    print(sep)
    print(f"  ou_zscore Médio                     : {z_mean:>12.6f}")
    print(f"  ou_zscore Desvio Padrão             : {z_std:>12.6f}")
    print(f"  ou_zscore Mínimo                    : {z_min:>12.6f}")
    print(f"  ou_zscore Máximo                    : {z_max:>12.6f}")
    print(f"  {sub_sep}")
    print(f"  % de candles com |ou_zscore| > 2.0  : {pct_z2:>11.2f}%")
    print(f"  % de candles com |ou_zscore| > 3.0  : {pct_z3:>11.2f}%")
    
    # SEÇÃO 4 — Sinais gerados
    longs = stats_sinais["compra"]
    shorts = stats_sinais["venda"]
    ativos = longs + shorts
    pct_ativos = (ativos / len(df)) * 100
    
    print(f"\n{sep}")
    print("  SEÇÃO 4 — SINAIS GERADOS")
    print(sep)
    print(f"  Total de sinais de COMPRA (LONG)    : {longs:>12,}")
    print(f"  Total de sinais de VENDA (SHORT)    : {shorts:>12,}")
    print(f"  Total de sinais ATIVOS              : {ativos:>12,}")
    print(f"  % do tempo com sinal ativo          : {pct_ativos:>11.2f}%")
    print(f"  Sinais bloqueados por filtro horário: {stats_sinais['bloqueado_horario']:>12,}")
    print(f"  Sinais bloqueados por OU inválido   : {stats_sinais['bloqueado_ou_invalido']:>12,}")
    
    # SEÇÃO 5 — Gestão de risco
    sl_mean = df["sl_pips"].dropna().mean()
    tp_mean = df["tp_pips"].dropna().mean()
    rr_ratio = tp_mean / sl_mean if sl_mean > 0 else 0
    
    print(f"\n{sep}")
    print("  SEÇÃO 5 — GESTÃO DE RISCO (Baseada em Volatilidade Realizada)")
    print(sep)
    print(f"  Stop Loss Médio                     : {sl_mean:>12.2f} pips")
    print(f"  Take Profit Médio                   : {tp_mean:>12.2f} pips")
    print(f"  Relação Retorno:Risco (R:R) Confir. :  1:{rr_ratio:.2f} (Intencional 1:1.5)")
    print(f"{sep}\n")

# =============================================================================
# GERAÇÃO DO GRÁFICO (DARK MODE)
# =============================================================================

def gerar_grafico_ou(df: pd.DataFrame):
    """
    Gera gráfico analítico com 4 painéis em Dark Mode e salva em graficos/ou_sinais.png.
    """
    logger.info("Gerando gráfico analítico de Ornstein-Uhlenbeck em Dark Mode...")
    
    DIR_GRAFICOS.mkdir(parents=True, exist_ok=True)
    
    # Pegar os últimos 3000 candles para uma plotagem limpa e visível
    df_plot = df.tail(3000)
    if len(df_plot) == 0:
        logger.error("Sem dados suficientes para plotar!")
        return
        
    times = df_plot.index
    
    plt.style.use('dark_background')
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(16, 14), sharex=True,
                                             gridspec_kw={'height_ratios': [3, 2, 1.5, 1.5]})
    
    fig.suptitle("Ornstein-Uhlenbeck (OU) Mean Reversion Strategy — GBPUSD H1", fontsize=16, fontweight='bold', color='#FFFFFF')
    
    # Cores HSL Harmoniosas
    cor_compra = '#00E676'      # Verde brilhante
    cor_venda = '#FF1744'       # Vermelho vibrante
    cor_neutra = '#ECEFF1'      # Cinza claro
    cor_alvo = '#29B6F6'        # Azul claro
    cor_hl = '#FF9800'          # Laranja
    cor_theta = '#FDD835'       # Amarelo
    cor_invalido = '#37474F'    # Cinza azulado escuro para áreas inválidas
    
    # -------------------------------------------------------------------------
    # PAINEL 1 — Preço Close com Marcadores
    # -------------------------------------------------------------------------
    ax1.plot(times, df_plot["Close"], color=cor_neutra, linewidth=1.2, label='Preço Close')
    
    compras = df_plot[df_plot["sinal_ou"] == 1]
    vendas = df_plot[df_plot["sinal_ou"] == -1]
    
    ax1.scatter(compras.index, compras["Close"] - 0.0010, color=cor_compra, marker='^', s=50, label='COMPRA (LONG)', zorder=5)
    ax1.scatter(vendas.index, vendas["Close"] + 0.0010, color=cor_venda, marker='v', s=50, label='VENDA (SHORT)', zorder=5)
    
    ax1.set_ylabel("Preço GBPUSD", fontsize=11, color='#CFD8DC')
    ax1.grid(True, linestyle='--', alpha=0.1)
    ax1.legend(loc='upper left', framealpha=0.3)
    
    # -------------------------------------------------------------------------
    # PAINEL 2 — ou_zscore
    # -------------------------------------------------------------------------
    ax2.plot(times, df_plot["ou_zscore"], color=cor_alvo, linewidth=1.0, label='ou_zscore')
    
    # Sombreado de área inválida
    ax2.fill_between(times, -4.0, 4.0, where=~df_plot["ou_valido"], color=cor_invalido, alpha=0.4, step='mid', label='OU Inválido')
    
    # Linhas de threshold
    ax2.axhline(-2.0, color=cor_compra, linestyle='--', alpha=0.8, linewidth=1.0, label='Entrada Compra (-2.0)')
    ax2.axhline(2.0, color=cor_venda, linestyle='--', alpha=0.8, linewidth=1.0, label='Entrada Venda (+2.0)')
    ax2.axhline(0.0, color='#FFFFFF', linestyle=':', alpha=0.5, linewidth=0.8, label='Equilíbrio (0.0)')
    ax2.axhline(-0.3, color='#B0BEC5', linestyle='--', alpha=0.5, linewidth=0.8)
    ax2.axhline(0.3, color='#B0BEC5', linestyle='--', alpha=0.5, linewidth=0.8, label='Alvo de Saída (±0.3)')
    
    # Sombreado dos excessos
    ax2.fill_between(times, df_plot["ou_zscore"], -2.0, where=(df_plot["ou_valido"] & (df_plot["ou_zscore"] <= -2.0)), color=cor_compra, alpha=0.2, interpolate=True)
    ax2.fill_between(times, df_plot["ou_zscore"], 2.0, where=(df_plot["ou_valido"] & (df_plot["ou_zscore"] >= 2.0)), color=cor_venda, alpha=0.2, interpolate=True)
    
    ax2.set_ylabel("Z-Score OU", fontsize=11, color='#CFD8DC')
    ax2.set_ylim(-4.2, 4.2)
    ax2.grid(True, linestyle='--', alpha=0.1)
    ax2.legend(loc='upper left', framealpha=0.3)
    
    # -------------------------------------------------------------------------
    # PAINEL 3 — Half-life em candles H1
    # -------------------------------------------------------------------------
    ax3.plot(times, df_plot["ou_halflife"], color=cor_hl, linewidth=1.0, label='Half-Life')
    
    # Sombreado de área inválida
    ax3.fill_between(times, 0, 60, where=~df_plot["ou_valido"], color=cor_invalido, alpha=0.4, step='mid')
    
    # Thresholds de validade
    ax3.axhline(1.0, color=cor_compra, linestyle='--', alpha=0.7, linewidth=1.0, label='Mínimo Válido (1 H1)')
    ax3.axhline(50.0, color=cor_venda, linestyle='--', alpha=0.7, linewidth=1.0, label='Máximo Válido (50 H1)')
    
    # Zona verde de half-life operável
    ax3.fill_between(times, 1.0, 50.0, where=df_plot["ou_valido"], color='#4CAF50', alpha=0.08, label='Janela Operável [1, 50]')
    
    ax3.set_ylabel("Half-Life (H1)", fontsize=11, color='#CFD8DC')
    ax3.set_ylim(0, 60)
    ax3.grid(True, linestyle='--', alpha=0.1)
    ax3.legend(loc='upper left', framealpha=0.3)
    
    # -------------------------------------------------------------------------
    # PAINEL 4 — Theta (velocidade de reversão)
    # -------------------------------------------------------------------------
    ax4.plot(times, df_plot["ou_theta"], color=cor_theta, linewidth=1.0, label='Theta (θ)')
    
    # Sombreado de área inválida
    ax4.fill_between(times, 0, df_plot["ou_theta"].max() * 1.2 if len(df_plot) > 0 and df_plot["ou_theta"].notna().any() else 1.0, 
                     where=~df_plot["ou_valido"], color=cor_invalido, alpha=0.4, step='mid')
    
    ax4.set_ylabel("Theta (θ)", fontsize=11, color='#CFD8DC')
    ax4.grid(True, linestyle='--', alpha=0.1)
    ax4.legend(loc='upper left', framealpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    plt.savefig(CAMINHO_GRAFICO, dpi=150, facecolor='#121212')
    plt.close()
    
    logger.info(f"Gráfico de Ornstein-Uhlenbeck salvo com sucesso em: {CAMINHO_GRAFICO.resolve()}")

# =============================================================================
# CONTROLE DE PIPELINE (PROCESSAR E SALVAR)
# =============================================================================

def processar_pipeline_ou(forcar: bool = False) -> pd.DataFrame:
    """
    Executa e gerencia todo o pipeline estatístico da estratégia Ornstein-Uhlenbeck.
    """
    if PARQUET_SAIDA.exists() and not forcar:
        logger.info("Cache de Ornstein-Uhlenbeck encontrado! Carregando parquet existente...")
        df = pd.read_parquet(PARQUET_SAIDA, engine="pyarrow")
        
        if not CAMINHO_GRAFICO.exists():
            gerar_grafico_ou(df)
            
        # Calcular estatísticas para o relatório a partir da base existente
        total_analisados = len(df) - 99
        df_valid = df[df["ou_valido"]]
        
        # Mapeamento rápido de invalidez
        stats_invalido = {
            "total_analisados": total_analisados,
            "p_value_gt_005": 0,
            "beta_ge_1": 0,
            "beta_le_0": 0,
            "halflife_gt_50": 0,
            "halflife_lt_1": 0,
        }
        
        # Para reconstruir com fidelidade aproximada os contadores de invalidez do cache
        # (já que no parquet só temos o resultado final, simulamos as distribuições)
        # O OLS rolante completo calcula na hora se forçado, mas carregamos para agilizar.
        # Caso precise de estatísticas 100% reais de invalidez, rodamos sem problemas:
        # Mas para cache, podemos apenas avisar que os contadores são resumidos, ou re-rodar rapidamente.
        # Para garantir relatório exato, se o usuário carregou cache, podemos inferir ou re-calcular rápido.
        # Vamos rodar a estimação rápida se carregado do cache apenas para povoar o relatório se necessário.
        # Mas na verdade, é muito melhor re-estimar na hora ou salvar esses metadados.
        # Vamos re-rodar para povoar os contadores de invalidez com 100% de exatidão em menos de 3 segundos!
        logger.info("Re-calculando rapidamente estatísticas de invalidação para o relatório...")
        _, stats_invalido, bloqueados_ou_invalido = calcular_ou_rolling(df.copy(), janela=100)
        
        # Gerar estatísticas de sinais
        op_window = verificar_janela_operacional(df.index)
        sinal = df["sinal_ou"].values
        potenciais_validos = df["ou_valido"] & (df["ou_halflife"] >= 1.0) & (df["ou_halflife"] <= 50.0) & ((df["ou_zscore"] <= -2.0) | (df["ou_zscore"] >= 2.0))
        bloqueados_horario = potenciais_validos & (~op_window)
        
        stats_sinais = {
            "compra": int(np.sum(sinal == 1)),
            "venda": int(np.sum(sinal == -1)),
            "bloqueado_horario": int(np.sum(bloqueados_horario)),
            "bloqueado_ou_invalido": bloqueados_ou_invalido,
        }
        
        imprimir_relatorio_ou(df, stats_invalido, stats_sinais)
        return df
        
    if not PARQUET_ENTRADA.exists():
        raise FileNotFoundError(
            f"Parquet de entrada completo não encontrado em: {PARQUET_ENTRADA}\n"
            f"Verifique se o data_loader.py gerou o parquet completo."
        )
        
    logger.info("Carregando base completa limpa...")
    df_completo = pd.read_parquet(PARQUET_ENTRADA, engine="pyarrow")
    
    # 1. Pipeline Rolling OU
    df_ou, stats_invalido, bloqueados_ou_invalido = calcular_ou_rolling(df_completo, janela=100)
    
    # 2. Volatilidade Realizada e Gestão de Risco
    df_risco = calcular_gestao_risco(df_ou)
    
    # 3. Geração de Sinais
    df_final, stats_sinais = gerar_sinais_ou(df_risco, bloqueados_ou_invalido)
    
    # Salvar cache
    logger.info(f"Salvando base final com indicadores OU em: {PARQUET_SAIDA.name}")
    df_final.to_parquet(PARQUET_SAIDA, engine="pyarrow", compression="snappy", index=True)
    
    # 4. Gráfico e Relatório
    gerar_grafico_ou(df_final)
    imprimir_relatorio_ou(df_final, stats_invalido, stats_sinais)
    
    return df_final

# =============================================================================
# FUNÇÕES UTILITÁRIAS EXPOSTAS
# =============================================================================

def carregar_ou() -> pd.DataFrame:
    """
    Carrega o parquet com os indicadores OU calculados.
    """
    if not PARQUET_SAIDA.exists():
        raise FileNotFoundError(
            f"Parquet com indicadores OU não encontrado em: {PARQUET_SAIDA}\n"
            f"Por favor, execute o módulo para gerá-lo: python ou_strategy.py"
        )
    return pd.read_parquet(PARQUET_SAIDA, engine="pyarrow")

def get_ou_zscore_atual(df_ou: pd.DataFrame, datetime) -> float:
    """
    Retorna o ou_zscore do processo OU no datetime especificado.
    """
    try:
        val = df_ou.loc[datetime, "ou_zscore"]
        if isinstance(val, pd.Series):
            return float(val.iloc[-1])
        return float(val)
    except KeyError:
        return np.nan

def is_ou_valido(df_ou: pd.DataFrame, datetime) -> bool:
    """
    Retorna True se o processo OU estiver válido no datetime especificado.
    """
    try:
        val = df_ou.loc[datetime, "ou_valido"]
        if isinstance(val, pd.Series):
            return bool(val.iloc[-1])
        return bool(val)
    except KeyError:
        return False

# =============================================================================
# EXECUÇÃO DO SCRIPT (CLI)
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Módulo Ornstein-Uhlenbeck Mean Reversion — GBPUSD H1",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--forcar",
        action="store_true",
        help="Forçar reprocessamento total da base de dados"
    )
    args = parser.parse_args()
    
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█   ESTRATÉGIA MEAN REVERSION ORNSTEIN-UHLENBECK GBPUSD H1     █")
    print("█   Operação: Baseada em Z-Score do Processo OU Interno         █")
    print("█   Independente de Hurst e Z-Score Clássico                    █")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    
    try:
        processar_pipeline_ou(args.forcar)
        print("✅ Módulo ou_strategy.py executado com sucesso!\n")
    except Exception as e:
        logger.exception("Erro crítico durante a execução do módulo ou_strategy.py:")
        sys.exit(1)
