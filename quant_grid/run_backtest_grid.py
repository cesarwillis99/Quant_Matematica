# -*- coding: utf-8 -*-
"""
run_backtest_grid.py — Motor de backtest candle-a-candle para Grid Trading.

Executa a simulação completa com níveis pré-calculados, alvo dinâmico
recalculado a cada candle, preenchimentos múltiplos por candle e
fechamento obrigatório de sexta-feira às 21h55.

Gera gráfico PNG dark mode com 4 painéis:
    1. Preço com níveis, preço médio e marcadores de saída
    2. Equity Curve
    3. Drawdown
    4. Distribuição de motivos de saída

Uso (CLI):
    python -m quant_grid.run_backtest_grid --grid 1 --parquet completo
    python -m quant_grid.run_backtest_grid --grid 2 --verbose
"""

import sys
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Imports do projeto ────────────────────────────────────────────────────────
from quant_grid.config import (
    CAPITAL_INICIAL, LOT_SIZE, FATOR_PIPS, JANELA_VR,
    PARQUET_COMPLETO, PARQUET_OPERACIONAL,
    DIR_GRAFICOS, DIR_METRICAS, DIR_GRID,
)
from quant_grid.core.grid_engine import GridState
from quant_grid.core.grid_executor import (
    verificar_preenchimentos,
    verificar_alvo,
    verificar_stop_hibrido,
    verificar_fechamento_sexta,
)
from quant_grid.core.grid_risk import (
    calcular_vr_atual,
    calcular_espacamento_pips,
    calcular_stop_drawdown_pips,
    calcular_alvo_pips,
)

# ── Configuração do ambiente ──────────────────────────────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding and sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Função auxiliar: obter_sinais
# ─────────────────────────────────────────────────────────────────────────────

def obter_sinais(df: pd.DataFrame, tipo_grid: int, params: dict) -> np.ndarray:
    """
    Pré-calcula os sinais do gatilho correspondente ao tipo_grid
    e retorna array numpy de int8.

    Parameters
    ----------
    df : pd.DataFrame
        Base de dados com colunas OHLCV e log_return.
    tipo_grid : int
        1, 2, 3 ou 4.
    params : dict
        Parâmetros do gatilho (chaves dependem do tipo_grid).

    Returns
    -------
    np.ndarray[int8]
        Array de sinais com o mesmo comprimento de df.
        +1 = COMPRA, -1 = VENDA, 0 = sem sinal.
    """
    col = f'sinal_grid{tipo_grid}'

    if tipo_grid == 1:
        from quant_grid.gatilhos.gatilho_daily_close import gerar_sinal_daily_close
        df_s = gerar_sinal_daily_close(df, pct_gatilho=params.get('pct_gatilho', 0.003))

    elif tipo_grid == 2:
        from quant_grid.gatilhos.gatilho_mm_slope import gerar_sinal_mm_slope
        df_s = gerar_sinal_mm_slope(
            df,
            janela_mm=params.get('janela_mm', 50),
            janela_slope=params.get('janela_slope', 5),
            threshold_slope=params.get('threshold_slope', 0.5),
        )

    elif tipo_grid == 3:
        from quant_grid.gatilhos.gatilho_afastamento import gerar_sinal_afastamento
        df_s = gerar_sinal_afastamento(
            df,
            janela_zscore=params.get('janela_zscore', 100),
            janela_percentil=params.get('janela_percentil', 200),
            threshold_zscore=params.get('threshold_zscore', 2.0),
            threshold_pct_alto=params.get('threshold_pct_alto', 0.85),
            threshold_pct_baixo=params.get('threshold_pct_baixo', 0.15),
        )

    elif tipo_grid == 4:
        from quant_grid.gatilhos.gatilho_twap_band import gerar_sinal_twap_band
        df_s = gerar_sinal_twap_band(
            df,
            janela=params.get('janela_twap', 24),
            mult_banda=params.get('mult_banda', 2.0),
        )

    else:
        raise ValueError(f'tipo_grid inválido: {tipo_grid}. Use 1, 2, 3 ou 4.')

    if col not in df_s.columns:
        raise KeyError(f'Coluna {col} não encontrada após gerar_sinal_grid{tipo_grid}.')

    return df_s[col].values.astype(np.int8)


# ─────────────────────────────────────────────────────────────────────────────
# Motor principal
# ─────────────────────────────────────────────────────────────────────────────

def rodar_backtest_grid(
    df: pd.DataFrame,
    tipo_grid: int,
    params: dict,
    capital_inicial: float = CAPITAL_INICIAL,
    verbose: bool = False,
) -> dict:
    """
    Motor candle-a-candle do Grid Trading com níveis pré-calculados,
    alvo dinâmico e preenchimentos múltiplos por candle.

    Ordem de operações em cada candle t:
        1. Fechamento obrigatório de sexta-feira às 21h55.
        2. Se grid ativo:
           a. Calcular VR atual e atualizar alvo.
           b. Verificar alvo (prioridade sobre novos níveis).
           c. Se grid ainda ativo: verificar preenchimentos + stop híbrido.
        3. Se sem grid: verificar novo gatilho.
        4. Atualizar equity curve com PnL flutuante.

    Parameters
    ----------
    df : pd.DataFrame
        Base com colunas: Open, High, Low, Close, log_return.
        Índice: DatetimeIndex naive (horário servidor MT5).
    tipo_grid : int
        Tipo do gatilho: 1, 2, 3 ou 4.
    params : dict
        Parâmetros do backtest. Chaves obrigatórias:
            'mult_espacamento', 'mult_alvo', 'mult_stop', 'stop_candles'
        Chaves adicionais dependem do tipo_grid (ex: 'pct_gatilho').
    capital_inicial : float
        Capital em USD no início da simulação.
    verbose : bool
        Se True, loga cada operação individualmente.

    Returns
    -------
    dict com:
        capital_final, pnl_total_usd, pnl_pct, total_grids,
        win_rate, profit_factor, sharpe, max_drawdown_pct,
        fator_recuperacao, media_ordens_por_grid, motivos_saida,
        equity_curve, trades.
    """
    # ── Pré-processamento ─────────────────────────────────────────────────────
    opens       = df['Open'].values.astype(np.float64)
    closes      = df['Close'].values.astype(np.float64)
    highs       = df['High'].values.astype(np.float64)
    lows        = df['Low'].values.astype(np.float64)
    log_returns = df['log_return'].values.astype(np.float64)
    dts         = df.index
    n           = len(df)

    logger.info(f'[Grid {tipo_grid}] Gerando sinais do gatilho...')
    sinais = obter_sinais(df, tipo_grid, params)
    n_sinais = int(np.sum(sinais != 0))
    logger.info(f'[Grid {tipo_grid}] {n_sinais} sinais encontrados. Iniciando backtest de {n} candles...')

    capital      = capital_inicial
    equity_curve: list[float] = []
    trades:       list[dict]  = []
    grid_ativo:   GridState | None = None

    # ── Loop Candle a Candle ──────────────────────────────────────────────────
    for t in range(n):
        high  = highs[t]
        low   = lows[t]
        close = closes[t]
        dt    = dts[t]

        # ── PASSO 1 — Verificar novo gatilho na abertura do candle t (baseado no sinal de t-1) ──
        if t > 0 and grid_ativo is None and sinais[t - 1] != 0:
            vr_atual = calcular_vr_atual(log_returns[:t], closes[t - 1], JANELA_VR)
            if vr_atual > 0.0:
                espacamento = calcular_espacamento_pips(
                    vr_atual, params['mult_espacamento']
                )
                grid_ativo = GridState(
                    direcao=sinais[t - 1],
                    preco_inicial=opens[t],  # Execução na abertura do candle seguinte
                    lot_size=LOT_SIZE,
                    espacamento_pips=espacamento,
                    vr_abertura=vr_atual,
                    candle_idx=t,
                )
                # Calcular alvo inicial logo após abertura
                grid_ativo.atualizar_alvo(vr_atual, params['mult_alvo'])
                if verbose:
                    dir_str = 'COMPRA' if sinais[t - 1] == 1 else 'VENDA'
                    logger.info(
                        f'[{dt}] Novo grid {dir_str} aberto na ABERTURA @ {opens[t]:.5f}. '
                        f'Espaçamento: {espacamento:.2f} pips | '
                        f'VR: {vr_atual:.2f} pips'
                    )

        # ── PASSO 2 — Fechamento obrigatório sexta 21h55 ──────────────────────
        if verificar_fechamento_sexta(dt) and grid_ativo is not None:
            trade = grid_ativo.fechar(close, 'SEXTA', t)
            capital += trade['pnl_usd']
            trades.append(trade)
            if verbose:
                logger.info(
                    f'[{dt}] Fechamento SEXTA. '
                    f'PnL={trade["pnl_usd"]:+.2f} USD | '
                    f'{trade["n_ordens"]} ordens'
                )
            grid_ativo = None

        # ── PASSO 3 — Gerenciar grid ativo ───────────────────────────────────
        if grid_ativo is not None:

            # 3a. Calcular VR atual e atualizar alvo ANTES de tudo
            vr_atual = calcular_vr_atual(log_returns[:t + 1], close, JANELA_VR)

            if vr_atual > 0.0:
                grid_ativo.atualizar_alvo(vr_atual, params['mult_alvo'])

            # 3b. Verificar alvo PRIMEIRO (fecha antes de abrir novos níveis)
            alvo_ok, preco_saida = verificar_alvo(grid_ativo, high, low)
            if alvo_ok:
                trade = grid_ativo.fechar(preco_saida, 'ALVO', t)
                capital += trade['pnl_usd']
                trades.append(trade)
                if verbose:
                    logger.info(
                        f'[{dt}] Fechamento ALVO @ {preco_saida:.5f}. '
                        f'PnL={trade["pnl_usd"]:+.2f} USD | '
                        f'{trade["n_ordens"]} ordens | '
                        f'{trade["pnl_pips"]:+.1f} pips'
                    )
                grid_ativo = None

            # 3c. Grid ainda ativo → verificar novos preenchimentos
            elif grid_ativo is not None:
                n_novos = verificar_preenchimentos(
                    grid_ativo, high, low, t, LOT_SIZE
                )
                if n_novos > 0 and verbose:
                    logger.info(
                        f'[{dt}] {n_novos} novo(s) nível(eis) preenchido(s). '
                        f'Total ordens: {grid_ativo.n_ordens} | '
                        f'Preço médio: {grid_ativo.preco_medio:.5f}'
                    )

                # 3d. Se houve novos preenchimentos, recalcular alvo
                #     com o novo preço médio antes de verificar o stop
                if n_novos > 0 and vr_atual > 0.0:
                    grid_ativo.atualizar_alvo(vr_atual, params['mult_alvo'])

                # 3e. Verificar stop híbrido com alvo e preço médio atualizados
                if vr_atual > 0.0:
                    stop_dd = calcular_stop_drawdown_pips(
                        vr_atual, params['mult_stop'], grid_ativo.n_ordens
                    )
                else:
                    stop_dd = 10.0   # mínimo seguro se VR não disponível

                stop_ok, motivo = verificar_stop_hibrido(
                    grid_ativo, close, t,
                    stop_dd, params['stop_candles']
                )
                if stop_ok:
                    trade = grid_ativo.fechar(close, motivo, t)
                    capital += trade['pnl_usd']
                    trades.append(trade)
                    if verbose:
                        logger.info(
                            f'[{dt}] Stop por {motivo}. '
                            f'PnL={trade["pnl_usd"]:+.2f} USD | '
                            f'{trade["n_ordens"]} ordens'
                        )
                    grid_ativo = None

        # ── PASSO 4 — Atualizar equity curve ─────────────────────────────────
        pnl_flutuante = 0.0
        if grid_ativo is not None:
            pnl_flutuante = grid_ativo.calcular_pnl_flutuante(close)
        equity_curve.append(capital + pnl_flutuante)

    # ── Forçar fechamento do último grid aberto ───────────────────────────────
    if grid_ativo is not None:
        trade = grid_ativo.fechar(closes[-1], 'FIM_SERIE', n - 1)
        capital += trade['pnl_usd']
        trades.append(trade)
        equity_curve[-1] = capital

    # ── Calcular métricas finais ──────────────────────────────────────────────
    total_grids = len(trades)
    pnl_total_usd = capital - capital_inicial
    pnl_pct = (pnl_total_usd / capital_inicial) * 100.0

    wins  = [tr for tr in trades if tr['pnl_usd'] > 0]
    loses = [tr for tr in trades if tr['pnl_usd'] <= 0]
    win_rate = (len(wins) / total_grids * 100.0) if total_grids > 0 else 0.0

    soma_ganhos = sum(tr['pnl_usd'] for tr in wins)
    soma_perdas = abs(sum(tr['pnl_usd'] for tr in loses))
    if soma_perdas > 0:
        profit_factor = soma_ganhos / soma_perdas
    else:
        profit_factor = soma_ganhos / 1e-8 if soma_ganhos > 0 else 1.0

    # Sharpe anualizado (H1 → sqrt(24 * 252))
    eq_series = pd.Series(equity_curve, dtype=float)
    eq_ret = eq_series.pct_change().dropna()
    if len(eq_ret) > 1 and eq_ret.std() > 0:
        sharpe = float((eq_ret.mean() / eq_ret.std()) * np.sqrt(24 * 252))
    else:
        sharpe = 0.0

    # Drawdown máximo
    peak = eq_series.cummax()
    dd = (eq_series - peak) / peak
    max_drawdown_pct = float(abs(dd.min()) * 100.0)
    max_dd_usd = float(abs((eq_series - peak).min()))
    fator_recuperacao = (pnl_total_usd / max_dd_usd) if max_dd_usd > 0 else 0.0

    # Média de ordens por grid
    media_ordens = float(np.mean([tr['n_ordens'] for tr in trades])) if total_grids > 0 else 0.0

    # Contagem de motivos de saída
    motivos: dict = {
        'ALVO': 0, 'DRAWDOWN': 0, 'TEMPO': 0,
        'SEXTA': 0, 'FIM_SERIE': 0,
    }
    for tr in trades:
        m = tr['motivo_saida']
        motivos[m] = motivos.get(m, 0) + 1

    logger.info(
        f'[Grid {tipo_grid}] Backtest concluído. '
        f'Grids={total_grids} | WR={win_rate:.1f}% | '
        f'PF={profit_factor:.2f} | Sharpe={sharpe:.2f} | '
        f'MDD={max_drawdown_pct:.1f}%'
    )

    return {
        'capital_final': float(capital),
        'pnl_total_usd': float(pnl_total_usd),
        'pnl_pct': float(pnl_pct),
        'total_grids': int(total_grids),
        'win_rate': float(win_rate),
        'profit_factor': float(profit_factor),
        'sharpe': float(sharpe),
        'max_drawdown_pct': float(max_drawdown_pct),
        'fator_recuperacao': float(fator_recuperacao),
        'media_ordens_por_grid': float(media_ordens),
        'motivos_saida': motivos,
        'equity_curve': equity_curve,
        'trades': trades,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Gerador de gráfico — 4 painéis dark mode
# ─────────────────────────────────────────────────────────────────────────────

def gerar_grafico_backtest(
    resultado: dict,
    tipo_grid: int,
    params: dict,
    caminho_saida: Path,
    df: pd.DataFrame | None = None,
):
    """
    Gera PNG dark mode com 4 painéis analíticos do backtest.

    Painel 1 — Preço (últimos 500 candles) com:
        - Níveis preenchidos: linha sólida verde (COMPRA) / vermelha (VENDA)
        - Níveis não preenchidos: linha tracejada cinza
        - Preço médio de entrada: linha laranja sólida
        - Marcadores de saída por motivo
    Painel 2 — Equity Curve com fill verde/vermelho
    Painel 3 — Drawdown percentual (fill vermelho)
    Painel 4 — Distribuição de motivos de saída (bar chart)

    Parameters
    ----------
    resultado : dict
        Saída de rodar_backtest_grid().
    tipo_grid : int
        Tipo do grid (1–4) para o título.
    params : dict
        Parâmetros usados no backtest.
    caminho_saida : Path
        Caminho do PNG de saída.
    df : pd.DataFrame | None
        DataFrame original para o eixo-x do painel 1.
        Se None, usa índice inteiro.
    """
    logger.info(f'Gerando gráfico dark mode em {caminho_saida} ...')
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)

    trades       = resultado['trades']
    equity_curve = resultado['equity_curve']
    total_grids  = resultado['total_grids']
    wr           = resultado['win_rate']
    motivos      = resultado['motivos_saida']

    plt.style.use('dark_background')
    fig, axes = plt.subplots(
        4, 1,
        figsize=(16, 18),
        gridspec_kw={'height_ratios': [4, 2, 1.5, 1.5]},
    )
    ax_preco, ax_eq, ax_dd, ax_mot = axes

    fig.suptitle(
        f'Grid {tipo_grid} — EURUSD H1 | {total_grids} grids | WR: {wr:.1f}%',
        fontsize=15, fontweight='bold', color='#FFFFFF',
    )
    fig.patch.set_facecolor('#0D1117')
    for ax in axes:
        ax.set_facecolor('#0D1117')

    # ── Painel 1 — Preço com níveis de grid ──────────────────────────────────
    n_eq = len(equity_curve)
    janela_plot = 500   # apenas os últimos 500 candles

    if df is not None and len(df) >= 1:
        df_plot = df.iloc[-janela_plot:]
        x_range = np.arange(len(df) - len(df_plot), len(df))
        close_plot = df_plot['Close'].values
        ax_preco.plot(x_range, close_plot, color='#ECEFF1', linewidth=0.8,
                      alpha=0.85, label='Close')
    else:
        x_range = np.arange(max(0, n_eq - janela_plot), n_eq)
        ax_preco.set_xlabel('Candle Index')

    # Paleta
    COR_COMPRA   = '#00E676'
    COR_VENDA    = '#FF1744'
    COR_PENDENTE = '#607D8B'
    COR_MEDIO    = '#FF9800'
    COR_ALVO     = '#FFEA00'

    # Marcadores de saída por motivo
    marcadores_motivo = {
        'ALVO':      ('*', COR_COMPRA, 120),
        'DRAWDOWN':  ('X', COR_VENDA, 80),
        'TEMPO':     ('^', COR_ALVO, 70),
        'SEXTA':     ('o', '#90A4AE', 60),
        'FIM_SERIE': ('D', '#B0BEC5', 55),
    }

    x_min_plot = x_range[0] if len(x_range) > 0 else 0
    x_max_plot = x_range[-1] if len(x_range) > 0 else n_eq

    for trade in trades:
        ci = trade['candle_inicio']
        cf = trade['candle_fim']

        # Só plotar trades visíveis na janela
        if cf < x_min_plot or ci > x_max_plot:
            continue

        ci_plot = max(ci, x_min_plot)
        cf_plot = min(cf, x_max_plot)

        dir_compra = trade['direcao'] == 1
        cor_ativo = COR_COMPRA if dir_compra else COR_VENDA

        # Plotar preço médio de entrada
        ax_preco.hlines(
            trade['preco_medio_entrada'],
            xmin=ci_plot, xmax=cf_plot,
            colors=COR_MEDIO, linewidths=1.2, alpha=0.9,
        )

        # Plotar níveis ativados (preenchidos)
        for preco_nivel in trade.get('niveis_ativados', []):
            ax_preco.hlines(
                preco_nivel,
                xmin=ci_plot, xmax=cf_plot,
                colors=cor_ativo, linewidths=0.9, alpha=0.8,
            )

        # Marcador de saída
        motivo = trade['motivo_saida']
        mk, cor_mk, sz_mk = marcadores_motivo.get(
            motivo, ('s', '#90A4AE', 60)
        )
        if x_min_plot <= cf <= x_max_plot:
            ax_preco.scatter(
                cf, trade['preco_saida'],
                marker=mk, color=cor_mk, s=sz_mk, zorder=6, alpha=0.95,
            )

    # Legenda manual do painel 1
    leg_elementos = [
        mpatches.Patch(color=COR_COMPRA,   label='Nível COMPRA preenchido'),
        mpatches.Patch(color=COR_VENDA,    label='Nível VENDA preenchido'),
        mpatches.Patch(color=COR_MEDIO,    label='Preço médio'),
        mpatches.Patch(color='#ECEFF1',    label='Preço Close'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor=COR_COMPRA,  markersize=9, label='Saída ALVO'),
        plt.Line2D([0], [0], marker='X', color='w', markerfacecolor=COR_VENDA,   markersize=9, label='Saída DRAWDOWN'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=COR_ALVO,    markersize=9, label='Saída TEMPO'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#90A4AE',   markersize=9, label='Saída SEXTA'),
    ]
    ax_preco.legend(handles=leg_elementos, loc='upper left',
                    framealpha=0.25, fontsize=7.5, ncol=2)
    ax_preco.set_ylabel('Preço EURUSD', fontsize=10, color='#CFD8DC')
    ax_preco.grid(True, linestyle='--', alpha=0.08)

    # ── Painel 2 — Equity Curve ───────────────────────────────────────────────
    eq_arr = np.array(equity_curve, dtype=float)
    t_eq   = np.arange(n_eq)

    ax_eq.plot(t_eq, eq_arr, color='#58A6FF', linewidth=1.4, label='Equity (USD)')
    ax_eq.axhline(CAPITAL_INICIAL, color='#FFFFFF', linestyle='--',
                  alpha=0.55, linewidth=0.9, label='Capital Inicial')
    ax_eq.fill_between(t_eq, eq_arr, CAPITAL_INICIAL,
                        where=(eq_arr >= CAPITAL_INICIAL),
                        color='#00E676', alpha=0.12, interpolate=True)
    ax_eq.fill_between(t_eq, eq_arr, CAPITAL_INICIAL,
                        where=(eq_arr < CAPITAL_INICIAL),
                        color='#FF1744', alpha=0.12, interpolate=True)
    ax_eq.set_ylabel('Equity (USD)', fontsize=10, color='#CFD8DC')
    ax_eq.grid(True, linestyle='--', alpha=0.08)
    ax_eq.legend(loc='upper left', framealpha=0.25, fontsize=8)

    # ── Painel 3 — Drawdown ───────────────────────────────────────────────────
    eq_s = pd.Series(equity_curve, dtype=float)
    peak = eq_s.cummax()
    dd_pct = ((eq_s - peak) / peak) * 100.0

    ax_dd.fill_between(t_eq, dd_pct.values, 0,
                        color='#FF1744', alpha=0.22, label='Drawdown %')
    ax_dd.plot(t_eq, dd_pct.values, color='#FF1744', linewidth=0.8, alpha=0.6)
    ax_dd.set_ylabel('Drawdown %', fontsize=10, color='#CFD8DC')
    ax_dd.grid(True, linestyle='--', alpha=0.08)
    ax_dd.legend(loc='lower left', framealpha=0.25, fontsize=8)

    # ── Painel 4 — Distribuição de motivos de saída ───────────────────────────
    labels = list(motivos.keys())
    counts = [motivos[k] for k in labels]
    total_m = sum(counts) if sum(counts) > 0 else 1
    pcts    = [(c / total_m) * 100.0 for c in counts]

    cores_motivo = {
        'ALVO': '#00E676', 'DRAWDOWN': '#FF1744',
        'TEMPO': '#FFEA00', 'SEXTA': '#90A4AE', 'FIM_SERIE': '#B0BEC5',
    }
    bar_cores = [cores_motivo.get(lb, '#90A4AE') for lb in labels]

    bars = ax_mot.bar(labels, counts, color=bar_cores, width=0.5,
                      edgecolor='#FFFFFF', alpha=0.82)
    ax_mot.set_ylabel('Quantidade', fontsize=10, color='#CFD8DC')
    ax_mot.grid(True, axis='y', linestyle='--', alpha=0.08)

    max_c = max(counts) if counts else 1
    for bar, pct in zip(bars, pcts):
        yv = bar.get_height()
        ax_mot.text(
            bar.get_x() + bar.get_width() / 2.0,
            yv + max_c * 0.02,
            f'{pct:.1f}%',
            ha='center', va='bottom',
            fontsize=9, color='#FFFFFF', fontweight='bold',
        )

    plt.tight_layout()
    plt.subplots_adjust(top=0.94, hspace=0.35)
    plt.savefig(caminho_saida, dpi=150, bbox_inches='tight', facecolor='#0D1117')
    plt.close()
    logger.info(f'Gráfico salvo em: {caminho_saida.resolve()}')
# ─────────────────────────────────────────────────────────────────────────────
# Funções adicionais de exportação (Novos Requisitos)
# ─────────────────────────────────────────────────────────────────────────────

def gerar_curva_capital_simples(
    resultado: dict,
    tipo_grid: int,
    params: dict,
    caminho_saida: Path,
    df: pd.DataFrame
):
    """
    Gera um PNG simples com a curva de capital e drawdown abaixo (2 painéis).
    Adaptado de gerar_equity_curve.py.
    """
    logger.info(f'Gerando PNG simples da curva de capital em {caminho_saida} ...')
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib.ticker as mticker
    import matplotlib.dates as mdates

    trades = resultado['trades']
    equity_curve = resultado['equity_curve']
    pnl_total = resultado['pnl_total_usd']
    pnl_pct = resultado['pnl_pct']
    win_rate = resultado['win_rate']
    mdd = resultado['max_drawdown_pct']
    sharpe = resultado['sharpe']
    pf = resultado['profit_factor']
    total_grids = resultado['total_grids']

    timestamps = df.index[:len(equity_curve)]
    eq_series = pd.Series(equity_curve, index=timestamps, dtype=float)
    peak = eq_series.cummax()
    dd_pct = ((eq_series - peak) / peak) * 100.0

    plt.style.use('dark_background')
    fig, (ax_eq, ax_dd) = plt.subplots(
        2, 1, figsize=(18, 9),
        gridspec_kw={"height_ratios": [0.68, 0.32]},
        sharex=True,
    )
    fig.patch.set_facecolor("#0D1117")
    ax_eq.set_facecolor("#161B22")
    ax_dd.set_facecolor("#161B22")

    # Equity Curve
    ax_eq.plot(timestamps, eq_series.values, color="#58A6FF", linewidth=1.2, zorder=3, label="Equity")
    ax_eq.axhline(CAPITAL_INICIAL, color="#FFFFFF", linestyle="--", linewidth=0.8, alpha=0.4, label=f"Capital inicial ${CAPITAL_INICIAL:,.0f}")

    # Fill verde acima / vermelho abaixo
    ax_eq.fill_between(timestamps, eq_series.values, CAPITAL_INICIAL,
                       where=(eq_series.values >= CAPITAL_INICIAL),
                       color="#00E676", alpha=0.10, interpolate=True)
    ax_eq.fill_between(timestamps, eq_series.values, CAPITAL_INICIAL,
                       where=(eq_series.values < CAPITAL_INICIAL),
                       color="#FF1744", alpha=0.10, interpolate=True)

    # Pontos de trades
    for tr in trades:
        ci = tr["candle_fim"]
        if ci >= len(timestamps):
            continue
        ts = timestamps[ci]
        eq = equity_curve[ci]
        cor = "#00E676" if tr["pnl_usd"] > 0 else "#FF1744"
        ax_eq.scatter(ts, eq, s=10, color=cor, zorder=4, alpha=0.65, linewidths=0)

    # Métricas formatadas na caixa de texto do gráfico
    metricas_txt = (
        f"PnL Total: ${pnl_total:+,.2f} ({pnl_pct:+.2f}%)   |   "
        f"Trades: {total_grids:,}   |   Win Rate: {win_rate:.1f}%   |   "
        f"PF: {pf:.2f}   |   Sharpe: {sharpe:.2f}   |   MDD: {mdd:.1f}%"
    )
    ax_eq.text(
        0.01, 0.97, metricas_txt,
        transform=ax_eq.transAxes,
        fontsize=9.5, color="#ECEFF1",
        va="top", ha="left",
        bbox=dict(facecolor="#21262D", edgecolor="#30363D", boxstyle="round,pad=0.4", alpha=0.85),
    )

    ax_eq.set_ylabel("Equity (USD)", fontsize=11, color="#8B949E")
    ax_eq.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax_eq.grid(True, linestyle="--", alpha=0.08, color="#58A6FF")
    ax_eq.legend(loc="lower right", framealpha=0.25, fontsize=9)
    ax_eq.tick_params(colors="#8B949E")
    ax_eq.spines[:].set_color("#30363D")

    # Linhas de anos
    for ano in range(df.index[0].year, df.index[-1].year + 1):
        ts_ano = pd.Timestamp(f"{ano}-01-01")
        if ts_ano > timestamps[-1]:
            break
        ax_eq.axvline(ts_ano, color="#30363D", linewidth=0.6, zorder=1)
        ax_eq.text(ts_ano, ax_eq.get_ylim()[0] if ax_eq.get_ylim()[0] else CAPITAL_INICIAL * 0.9,
                   str(ano), color="#8B949E", fontsize=8, va="bottom", ha="left")

    # Drawdown
    ax_dd.fill_between(timestamps, dd_pct.values, 0, color="#FF1744", alpha=0.25, interpolate=True)
    ax_dd.plot(timestamps, dd_pct.values, color="#FF1744", linewidth=0.8)
    ax_dd.set_ylabel("Drawdown (%)", fontsize=11, color="#8B949E")
    ax_dd.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax_dd.grid(True, linestyle="--", alpha=0.08, color="#FF1744")
    ax_dd.tick_params(colors="#8B949E")
    ax_dd.spines[:].set_color("#30363D")

    ax_dd.xaxis.set_major_locator(mdates.YearLocator())
    ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_dd.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    plt.setp(ax_dd.xaxis.get_majorticklabels(), color="#8B949E", fontsize=9)

    nomes_gatilhos = {1: 'Daily Close', 2: 'MM Slope', 3: 'Afastamento', 4: 'TWAP Band'}
    gatilho_nome = nomes_gatilhos.get(tipo_grid, f'Grid {tipo_grid}')
    
    fig.suptitle(
        f"Grid {tipo_grid} — {gatilho_nome} | Curva de Capital | 2016–2023",
        fontsize=15, fontweight="bold", color="#ECEFF1", y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.subplots_adjust(hspace=0.06)

    plt.savefig(caminho_saida, dpi=160, bbox_inches="tight", facecolor="#0D1117")
    plt.close()
    logger.info(f"Curva de capital simples salva em: {caminho_saida.resolve()}")


def salvar_metricas_txt(resultado: dict, caminho_saida: Path):
    """
    Exporta o relatório detalhado de métricas de backtest para um arquivo de texto.
    """
    logger.info(f'Salvando métricas formatadas em {caminho_saida} ...')
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)

    capital_inicial = CAPITAL_INICIAL
    capital_final = resultado['capital_final']
    pnl_total = resultado['pnl_total_usd']
    pnl_pct = resultado['pnl_pct']
    total_grids = resultado['total_grids']
    win_rate = resultado['win_rate']
    profit_factor = resultado['profit_factor']
    sharpe = resultado['sharpe']
    max_drawdown_pct = resultado['max_drawdown_pct']
    fator_recuperacao = resultado['fator_recuperacao']
    media_ordens = resultado['media_ordens_por_grid']
    motivos = resultado['motivos_saida']

    conteudo = f"""=====================================================
  MÉTRICAS DO BACKTEST DO ROBÔ (GRID TRADING)
=====================================================
Capital Inicial:               ${capital_inicial:,.2f}
Capital Final:                 ${capital_final:,.2f}
PnL Total:                     ${pnl_total:+,.2f} ({pnl_pct:+.2f}%)
Total de Trades (Grids):       {total_grids:,}
Win Rate:                      {win_rate:.2f}%
Profit Factor:                 {profit_factor:.2f}
Sharpe Ratio (anualizado H1):  {sharpe:.2f}
Max Drawdown:                  {max_drawdown_pct:.2f}%
Fator de Recuperação:          {fator_recuperacao:.2f}
Média Ordens/Grid:             {media_ordens:.2f}

DETALHAMENTO DAS SAÍDAS POR MOTIVO:
-----------------------------------------------------
Saídas por ALVO:               {motivos.get('ALVO', 0)}
Saídas por DRAWDOWN:           {motivos.get('DRAWDOWN', 0)}
Saídas por TEMPO:              {motivos.get('TEMPO', 0)}
Saídas por SEXTA:              {motivos.get('SEXTA', 0)}
Saídas por FIM_SERIE:          {motivos.get('FIM_SERIE', 0)}
=====================================================
"""
    with open(caminho_saida, 'w', encoding='utf-8') as f:
        f.write(conteudo)
    logger.info(f"Relatório de métricas salvo com sucesso em: {caminho_saida.resolve()}")


def gerar_candles_ultimo_mes_html(resultado: dict, df: pd.DataFrame, caminho_saida: Path):
    """
    Gera um gráfico interativo em HTML usando Plotly contendo APENAS o último mês
    de dados em formato CANDLES, mostrando detalhadamente as entradas, saídas,
    níveis de grade e preço médio dos grids ativos nesse período.
    """
    logger.info(f'Gerando gráfico interativo HTML de candles (último mês) em {caminho_saida} ...')
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)

    import plotly.graph_objects as go

    # Filtrar dados do último mês com base na data final da série
    data_fim = df.index[-1]
    data_inicio = data_fim - pd.Timedelta(days=30)
    df_mes = df[df.index >= data_inicio]

    if len(df_mes) == 0:
        logger.warning("Nenhum data encontrada para o último mês de candles!")
        return

    # Criar figura com o gráfico de velas
    fig = go.Figure()

    # Adicionar velas (Candlesticks)
    fig.add_trace(go.Candlestick(
        x=df_mes.index,
        open=df_mes['Open'],
        high=df_mes['High'],
        low=df_mes['Low'],
        close=df_mes['Close'],
        name='Preço (Candles)',
        increasing_line_color='#00E676',
        increasing_fillcolor='#00E676',
        decreasing_line_color='#FF1744',
        decreasing_fillcolor='#FF1744',
        line_width=1,
    ))

    # Controlar legendas duplicadas
    adicionou_legenda = {
        'preco_medio': False,
        'nivel_compra': False,
        'nivel_venda': False,
        'abertura_compra': False,
        'abertura_venda': False,
        'saida_lucro': False,
        'saida_perda': False,
    }

    trades = resultado['trades']

    # Iterar sobre todos os trades do backtest
    for idx, tr in enumerate(trades):
        # Índices de candles inicial e final
        ci = tr['candle_inicio']
        cf = tr['candle_fim']

        if ci >= len(df) or cf >= len(df):
            continue

        ts_inicio = df.index[ci]
        ts_fim = df.index[cf]

        # Verificar se o trade ocorreu (total ou parcialmente) no último mês
        if ts_fim < data_inicio or ts_inicio > data_fim:
            continue

        # Restringir o desenho visual aos limites do gráfico de 30 dias para evitar esticar demais
        ts_inicio_plot = max(ts_inicio, data_inicio)
        ts_fim_plot = min(ts_fim, data_fim)

        direcao = tr['direcao']
        is_compra = direcao == 1
        cor_grid = '#00E676' if is_compra else '#FF1744'
        motivo_saida = tr['motivo_saida']
        pnl_usd = tr['pnl_usd']
        n_ordens = tr['n_ordens']
        p_medio = tr['preco_medio_entrada']
        p_saida = tr['preco_saida']

        # 1. Preço médio
        show_leg = not adicionou_legenda['preco_medio']
        adicionou_legenda['preco_medio'] = True
        fig.add_trace(go.Scatter(
            x=[ts_inicio_plot, ts_fim_plot],
            y=[p_medio, p_medio],
            mode='lines',
            line=dict(color='#FF9800', width=1.5, dash='dash'),
            name='Preço Médio',
            legendgroup='preco_medio',
            showlegend=show_leg,
            hoverinfo='skip'
        ))

        # 2. Níveis preenchidos
        tipo_nivel = 'nivel_compra' if is_compra else 'nivel_venda'
        nome_nivel = 'Níveis de Compra' if is_compra else 'Níveis de Venda'
        show_leg_nivel = not adicionou_legenda[tipo_nivel]
        adicionou_legenda[tipo_nivel] = True

        for lvl in tr.get('niveis_ativados', []):
            fig.add_trace(go.Scatter(
                x=[ts_inicio_plot, ts_fim_plot],
                y=[lvl, lvl],
                mode='lines',
                line=dict(color=cor_grid, width=0.8),
                opacity=0.7,
                name=nome_nivel,
                legendgroup=tipo_nivel,
                showlegend=show_leg_nivel,
                hoverinfo='skip'
            ))

        # 3. Marcador de abertura (no primeiro candle do trade se estiver no gráfico)
        if ts_inicio >= data_inicio:
            tipo_abertura = 'abertura_compra' if is_compra else 'abertura_venda'
            nome_abertura = 'Abertura Compra' if is_compra else 'Abertura Venda'
            show_leg_ab = not adicionou_legenda[tipo_abertura]
            adicionou_legenda[tipo_abertura] = True
            
            p_abertura = tr['niveis_ativados'][0] if tr['niveis_ativados'] else p_medio
            fig.add_trace(go.Scatter(
                x=[ts_inicio],
                y=[p_abertura],
                mode='markers',
                marker=dict(symbol='triangle-up' if is_compra else 'triangle-down', size=10, color=cor_grid),
                name=nome_abertura,
                legendgroup=tipo_abertura,
                showlegend=show_leg_ab,
                hovertemplate=(
                    f"<b>Abertura Grid #{idx+1}</b><br>"
                    f"Direção: {'COMPRA' if is_compra else 'VENDA'}<br>"
                    f"Preço Entrada: %{{y:.5f}}<br>"
                    f"Data/Hora: {ts_inicio.strftime('%Y-%m-%d %H:%M')}<extra></extra>"
                )
            ))

        # 4. Marcador de fechamento (se estiver no gráfico)
        if ts_fim >= data_inicio:
            lucro = pnl_usd > 0
            tipo_saida = 'saida_lucro' if lucro else 'saida_perda'
            nome_saida = 'Saída com Lucro' if lucro else 'Saída com Perda'
            show_leg_sd = not adicionou_legenda[tipo_saida]
            adicionou_legenda[tipo_saida] = True

            cor_saida = '#00E676' if lucro else '#FF1744'
            simbolo_saida = 'star' if lucro else 'x'
            tamanho_saida = 12 if lucro else 9

            fig.add_trace(go.Scatter(
                x=[ts_fim],
                y=[p_saida],
                mode='markers',
                marker=dict(
                    symbol=simbolo_saida,
                    size=tamanho_saida,
                    color=cor_saida,
                    line=dict(width=1, color='white')
                ),
                name=nome_saida,
                legendgroup=tipo_saida,
                showlegend=show_leg_sd,
                hovertemplate=(
                    f"<b>Fechamento Grid #{idx+1}</b><br>"
                    f"Direção: {'COMPRA' if is_compra else 'VENDA'}<br>"
                    f"Motivo Saída: {motivo_saida}<br>"
                    f"Nº Ordens Preenchidas: {n_ordens}<br>"
                    f"PnL: ${pnl_usd:+,.2f} ({tr['pnl_pips']:+.1f} pips)<br>"
                    f"Data/Hora: {ts_fim.strftime('%Y-%m-%d %H:%M')}<extra></extra>"
                )
            ))

    # Atualizações no layout para visualização dark premium
    fig.update_layout(
        title=dict(
            text=f"<b>Visualização Candle-a-Candle (Últimos 30 dias)</b><br>Interações de entrada, saída e preço médio no gráfico de velas",
            font=dict(size=18, color='#ECEFF1'),
            x=0.5
        ),
        paper_bgcolor="#0D1117",
        plot_bgcolor="#161B22",
        font=dict(color="#ECEFF1", family="Inter, Arial, sans-serif", size=12),
        hovermode="x unified",
        height=800,
        margin=dict(l=60, r=40, t=90, b=60),
        xaxis=dict(
            gridcolor="#21262D",
            zerolinecolor="#30363D",
            tickfont=dict(color="#8B949E"),
            title_font=dict(color="#8B949E"),
            rangeslider=dict(visible=False), # Desativar rangeslider para focar no gráfico de velas
            title_text="Data / Hora (Servidor MT5)"
        ),
        yaxis=dict(
            gridcolor="#21262D",
            zerolinecolor="#30363D",
            tickfont=dict(color="#8B949E"),
            title_font=dict(color="#8B949E"),
            title_text="Preço EURUSD",
            tickformat=".5f"
        ),
        legend=dict(
            bgcolor="rgba(22,27,34,0.85)",
            bordercolor="#30363D",
            borderwidth=1,
            x=0.01,
            y=0.99,
            font=dict(size=10)
        )
    )

    # Escrever arquivo HTML autônomo (Plotly offline embutido)
    fig.write_html(
        str(caminho_saida),
        include_plotlyjs=True,
        full_html=True,
        config={
            "scrollZoom": True,
            "displayModeBar": True,
            "toImageButtonOptions": {
                "format": "png", "filename": "candles_ultimo_mes", "scale": 2
            }
        }
    )
    logger.info(f"Gráfico HTML interativo de velas salvo em: {caminho_saida.resolve()}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI standalone
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Backtest de Grid Trading — EURUSD H1',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--grid', type=int, required=True,
                        choices=[1, 2, 3, 4],
                        help='Tipo de grid a rodar (1, 2, 3 ou 4)')
    parser.add_argument('--parquet', type=str, default='completo',
                        choices=['completo', 'operacional'],
                        help='Parquet de entrada (default: completo)')
    parser.add_argument('--verbose', action='store_true',
                        help='Logar cada operação individualmente')
    args = parser.parse_args()

    parquet_path = PARQUET_COMPLETO if args.parquet == 'completo' else PARQUET_OPERACIONAL
    if not parquet_path.exists():
        logger.error(f'Parquet não encontrado: {parquet_path}')
        logger.error(
            'Verifique se eurusd_h1_completo.parquet (ou operacional) '
            'existe em quant_eurusd_h1/data/'
        )
        sys.exit(1)

    logger.info(f'Carregando {parquet_path.name} ...')
    df_raw = pd.read_parquet(parquet_path, engine='pyarrow')
    logger.info(f'Base carregada. {len(df_raw):,} candles.')

    # Parâmetros padrão para teste standalone
    params_padrao = {
        'mult_espacamento': 1.0,
        'mult_alvo': 1.5,
        'mult_stop': 3.0,
        'stop_candles': 120,
        # Grid 1
        'pct_gatilho': 0.003,
        # Grid 2
        'janela_mm': 50, 'janela_slope': 5, 'threshold_slope': 0.5,
        # Grid 3
        'janela_zscore': 100, 'janela_percentil': 200,
        'threshold_zscore': 2.0, 'threshold_pct_alto': 0.85, 'threshold_pct_baixo': 0.15,
        # Grid 4
        'janela_twap': 24, 'mult_banda': 2.0,
    }

    resultado = rodar_backtest_grid(
        df_raw, tipo_grid=args.grid,
        params=params_padrao,
        capital_inicial=CAPITAL_INICIAL,
        verbose=args.verbose,
    )

    # ── Relatório no terminal ──────────────────────────────────────────────────
    sep = '═' * 55
    print(f'\n{sep}')
    print(f'  BACKTEST GRID {args.grid} — {args.parquet.upper()}')
    print(sep)
    print(f'  Capital Final:         ${resultado["capital_final"]:>12,.2f}')
    print(f'  PnL Total:             ${resultado["pnl_total_usd"]:>+12,.2f} ({resultado["pnl_pct"]:+.2f}%)')
    print(f'  Total de Grids:        {resultado["total_grids"]:>12,}')
    print(f'  Win Rate:              {resultado["win_rate"]:>12.2f}%')
    print(f'  Profit Factor:         {resultado["profit_factor"]:>12.2f}')
    print(f'  Sharpe Ratio:          {resultado["sharpe"]:>12.2f}')
    print(f'  Drawdown Máximo:       {resultado["max_drawdown_pct"]:>11.2f}%')
    print(f'  Fator de Recuperação:  {resultado["fator_recuperacao"]:>12.2f}')
    print(f'  Média Ordens/Grid:     {resultado["media_ordens_por_grid"]:>12.2f}')
    print(f'  Motivos de Saída:      {resultado["motivos_saida"]}')
    print(f'{sep}\n')

    # ── Geração Dinâmica de Diretório de Saída ─────────────────────────────────
    # Determinar nome do ativo com base no arquivo parquet
    partes = parquet_path.stem.split('_')
    ativo_str = partes[0].upper()
    if len(partes) > 1 and partes[1].upper() in ['H1', 'M15', 'M30', 'H4', 'D1']:
        ativo_str = f"{ativo_str}_{partes[1].upper()}"

    # Determinar nome do gatilho com base no grid
    nomes_gatilhos = {
        1: 'daily_close',
        2: 'mm_slope',
        3: 'afastamento',
        4: 'twap_band'
    }
    gatilho_nome = nomes_gatilhos.get(args.grid, f'grid_{args.grid}')
    
    # Criar pasta final de destino
    diretorio_saida = DIR_GRID / "resultados" / f"{ativo_str}_{gatilho_nome}"
    diretorio_saida.mkdir(parents=True, exist_ok=True)
    logger.info(f"Diretório de destino das saídas: {diretorio_saida.resolve()}")

    # ── Exportar os 4 Arquivos Solicitados ──────────────────────────────────────
    # 1. analise_completa.png (Foto 1 - 4 painéis)
    caminho_analise = diretorio_saida / "analise_completa.png"
    gerar_grafico_backtest(resultado, args.grid, params_padrao, caminho_analise, df=df_raw)

    # 2. curva_capital.png (Foto 2 - curva simples)
    caminho_curva = diretorio_saida / "curva_capital.png"
    gerar_curva_capital_simples(resultado, args.grid, params_padrao, caminho_curva, df=df_raw)

    # 3. candles_ultimo_mes.html (Foto 3 - candles interativo)
    caminho_candles = diretorio_saida / "candles_ultimo_mes.html"
    gerar_candles_ultimo_mes_html(resultado, df_raw, caminho_candles)

    # 4. metricas.txt (Métricas detalhadas)
    caminho_metricas = diretorio_saida / "metricas.txt"
    salvar_metricas_txt(resultado, caminho_metricas)

    print(f"\n{sep}")
    print("  BACKTEST CONCLUÍDO E ARQUIVOS EXPORTADOS COM SUCESSO:")
    print(f"  Pasta de Destino: {diretorio_saida.resolve()}")
    print("-----------------------------------------------------")
    print(f"  1. Análise de 4 painéis     : {caminho_analise.name}")
    print(f"  2. Curva de capital simples : {caminho_curva.name}")
    print(f"  3. Velas interativo HTML   : {caminho_candles.name}")
    print(f"  4. Relatório de Métricas    : {caminho_metricas.name}")
    print(f"{sep}\n")
