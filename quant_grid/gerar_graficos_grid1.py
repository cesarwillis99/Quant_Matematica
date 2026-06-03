# -*- coding: utf-8 -*-
"""
gerar_graficos_grid1.py — Gera gráficos de análise do Grid 1 (Daily Close).

Saídas:
  1. PNG dark-mode (4 painéis) → quant_grid/resultados/graficos/
  2. HTML interativo Plotly (velas + níveis de grid) → quant_grid/resultados/graficos/

Uso:
    python -m quant_grid.gerar_graficos_grid1
    python -m quant_grid.gerar_graficos_grid1 --semana 2016-02-01   # semana inicial personalizada
    python -m quant_grid.gerar_graficos_grid1 --trade 5             # semana do trade nº 5
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# ── Imports do projeto ────────────────────────────────────────────────────────
from quant_grid.config import (
    CAPITAL_INICIAL, LOT_SIZE, FATOR_PIPS, JANELA_VR,
    PARQUET_COMPLETO, SPREAD_PIPS, DIR_GRAFICOS,
)
from quant_grid.run_backtest_grid import rodar_backtest_grid, gerar_grafico_backtest

# ── Parâmetros do backtest ────────────────────────────────────────────────────
PARAMS = {
    'pct_gatilho':      0.003,
    'mult_espacamento': 1.0,
    'mult_alvo':        1.5,
    'mult_stop':        3.0,
    'stop_candles':     120,
}


# ─────────────────────────────────────────────────────────────────────────────
# Gráfico interativo Plotly — velas + níveis de grid
# ─────────────────────────────────────────────────────────────────────────────

def gerar_grafico_velas_interativo(
    df: pd.DataFrame,
    trades: list[dict],
    data_inicio: pd.Timestamp,
    caminho_html: Path,
):
    """
    Gera um gráfico Plotly HTML interativo com:
      - Candles OHLC de 1 semana (5 dias × 24 barras H1)
      - Linha do daily_close de cada dia (referência do gatilho)
      - Níveis preenchidos de cada trade visível no período
      - Preço médio de entrada de cada trade
      - Linha do preço alvo de saída
      - Marcadores de abertura e fechamento dos trades
      - Tooltip com todos os detalhes ao passar o mouse
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("[ERRO] plotly não instalado. Execute: pip install plotly")
        return

    # ── Recorte de 1 semana (120 candles H1 = 5 dias) ─────────────────────────
    data_fim = data_inicio + timedelta(days=7)
    mask = (df.index >= data_inicio) & (df.index < data_fim)
    df_semana = df[mask].copy()

    if len(df_semana) == 0:
        print(f"[AVISO] Nenhum candle encontrado entre {data_inicio} e {data_fim}.")
        return

    # ── Calcular daily_close para o período ───────────────────────────────────
    from quant_grid.gatilhos.gatilho_daily_close import calcular_daily_close
    daily_close_full = calcular_daily_close(df)
    df_semana['daily_close'] = daily_close_full[mask]
    df_semana['afastamento_pct'] = (
        (df_semana['Close'] - df_semana['daily_close']) / df_semana['daily_close'] * 100
    )

    # Índice numérico para os candles (eixo x)
    idx_global = {ts: i for i, ts in enumerate(df.index)}

    # Mapear índice do df.index para posição no df_semana
    semana_timestamps = df_semana.index.tolist()
    n_sem = len(semana_timestamps)

    # Identificar trades que ocorrem (total ou parcialmente) neste período
    idx_inicio_semana = idx_global[semana_timestamps[0]]
    idx_fim_semana    = idx_global[semana_timestamps[-1]]

    trades_visiveis = [
        tr for tr in trades
        if tr['candle_fim'] >= idx_inicio_semana and tr['candle_inicio'] <= idx_fim_semana
    ]

    # ── Layout com 2 subplots (preço + volume) ────────────────────────────────
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.78, 0.22],
        subplot_titles=["EURUSD H1 — Velas com Grid 1 (Daily Close)", "Volume"],
    )

    # ── PAINEL 1: Candles ─────────────────────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=semana_timestamps,
            open=df_semana['Open'],
            high=df_semana['High'],
            low=df_semana['Low'],
            close=df_semana['Close'],
            name="EURUSD H1",
            increasing_line_color="#26A69A",
            decreasing_line_color="#EF5350",
            increasing_fillcolor="#26A69A",
            decreasing_fillcolor="#EF5350",
            hovertext=[
                f"<b>{ts.strftime('%Y-%m-%d %H:%M')}</b><br>"
                f"Open:  {row.Open:.5f}<br>"
                f"High:  {row.High:.5f}<br>"
                f"Low:   {row.Low:.5f}<br>"
                f"Close: {row.Close:.5f}<br>"
                f"Daily Close Ref: {row.daily_close:.5f}<br>"
                f"Afastamento: {row.afastamento_pct:+.3f}%"
                for ts, row in df_semana.iterrows()
            ],
            hoverinfo="text",
        ),
        row=1, col=1,
    )

    # ── Linhas de daily_close por dia ─────────────────────────────────────────
    # Agrupar por data e desenhar uma linha horizontal para cada dia
    for data, grupo in df_semana.groupby(df_semana.index.date):
        dc = grupo['daily_close'].iloc[0]
        if pd.isna(dc):
            continue
        ts_ini = grupo.index[0]
        ts_fim = grupo.index[-1]
        fig.add_trace(
            go.Scatter(
                x=[ts_ini, ts_fim],
                y=[dc, dc],
                mode="lines",
                line=dict(color="#FFA726", width=1.5, dash="dot"),
                name=f"Daily Close {data}",
                hovertemplate=f"<b>Daily Close {data}</b><br>Preço: {dc:.5f}<extra></extra>",
                legendgroup="daily_close",
                showlegend=(data == df_semana.index.date[0]),
            ),
            row=1, col=1,
        )

    # ── Trades visíveis no período ────────────────────────────────────────────
    CORES_DIR  = {1: "#00E676", -1: "#FF1744"}   # verde compra / vermelho venda
    COR_MEDIO  = "#FF9800"
    COR_ALVO   = "#FFEA00"
    COR_NIVEL  = {"ativo": None, "pendente": "#607D8B"}
    MOTIVO_SIM = {"ALVO": "⭐", "DRAWDOWN": "❌", "TEMPO": "⏱️", "SEXTA": "📅"}

    for k, tr in enumerate(trades_visiveis):
        dir_str  = "COMPRA" if tr['direcao'] == 1 else "VENDA"
        cor_dir  = CORES_DIR[tr['direcao']]
        motivo_icon = MOTIVO_SIM.get(tr['motivo_saida'], "?")

        # Timestamps de início e fim do trade (dentro da janela da semana)
        ts_ini_trade = df.index[tr['candle_inicio']] if tr['candle_inicio'] < len(df) else semana_timestamps[0]
        ts_fim_trade = df.index[tr['candle_fim']]    if tr['candle_fim'] < len(df) else semana_timestamps[-1]

        ts_ini_plot = max(ts_ini_trade, semana_timestamps[0])
        ts_fim_plot = min(ts_fim_trade, semana_timestamps[-1])

        # Linha do preço médio de entrada
        fig.add_trace(
            go.Scatter(
                x=[ts_ini_plot, ts_fim_plot],
                y=[tr['preco_medio_entrada'], tr['preco_medio_entrada']],
                mode="lines",
                line=dict(color=COR_MEDIO, width=2),
                name=f"Trade #{k+1} Preço Médio",
                hovertemplate=(
                    f"<b>{motivo_icon} Trade #{k+1} — {dir_str}</b><br>"
                    f"Preço Médio: {tr['preco_medio_entrada']:.5f}<br>"
                    f"Ordens: {tr['n_ordens']}/8<br>"
                    f"Lote: {tr['lot_total']:.2f}<br>"
                    f"PnL: {tr['pnl_pips']:+.2f} pips / ${tr['pnl_usd']:+.2f}<br>"
                    f"Saída: {tr['motivo_saida']}"
                    "<extra></extra>"
                ),
                legendgroup=f"trade_{k}",
                showlegend=True,
            ),
            row=1, col=1,
        )

        # Níveis preenchidos
        for n_idx, preco_nivel in enumerate(tr['niveis_ativados']):
            fig.add_trace(
                go.Scatter(
                    x=[ts_ini_plot, ts_fim_plot],
                    y=[preco_nivel, preco_nivel],
                    mode="lines",
                    line=dict(color=cor_dir, width=1, dash="solid"),
                    name=f"Trade #{k+1} Nível {n_idx}",
                    hovertemplate=(
                        f"<b>Nível {n_idx} ({dir_str})</b><br>"
                        f"Entrada real: {preco_nivel:.5f}<br>"
                        f"Spread descontado: {SPREAD_PIPS/2:.1f} pip"
                        "<extra></extra>"
                    ),
                    legendgroup=f"trade_{k}",
                    showlegend=False,
                ),
                row=1, col=1,
            )

        # Marcador de abertura do trade (triângulo)
        if ts_ini_trade >= semana_timestamps[0]:
            fig.add_trace(
                go.Scatter(
                    x=[ts_ini_trade],
                    y=[tr['niveis_ativados'][0]],
                    mode="markers",
                    marker=dict(
                        symbol="triangle-up" if tr['direcao'] == 1 else "triangle-down",
                        size=14,
                        color=cor_dir,
                        line=dict(width=1, color="white"),
                    ),
                    name=f"Abertura #{k+1}",
                    hovertemplate=(
                        f"<b>🟢 Abertura Trade #{k+1}</b><br>"
                        f"Direção: {dir_str}<br>"
                        f"Nível 0: {tr['niveis_ativados'][0]:.5f}<br>"
                        f"VR abertura: {tr['vr_abertura']:.2f} pips<br>"
                        f"Espaçamento: {tr['espacamento_pips']:.2f} pips"
                        "<extra></extra>"
                    ),
                    legendgroup=f"trade_{k}",
                    showlegend=False,
                ),
                row=1, col=1,
            )

        # Marcador de fechamento do trade (X ou estrela)
        if ts_fim_trade <= semana_timestamps[-1]:
            marker_sym = "star" if tr['motivo_saida'] == "ALVO" else "x"
            pnl_str = f"+{tr['pnl_usd']:.2f}" if tr['pnl_usd'] > 0 else f"{tr['pnl_usd']:.2f}"
            fig.add_trace(
                go.Scatter(
                    x=[ts_fim_trade],
                    y=[tr['preco_saida']],
                    mode="markers",
                    marker=dict(
                        symbol=marker_sym,
                        size=16,
                        color=COR_ALVO if tr['motivo_saida'] == "ALVO" else "#FF1744",
                        line=dict(width=1, color="white"),
                    ),
                    name=f"Saída #{k+1}",
                    hovertemplate=(
                        f"<b>{motivo_icon} Fechamento Trade #{k+1}</b><br>"
                        f"Motivo: {tr['motivo_saida']}<br>"
                        f"Preço saída: {tr['preco_saida']:.5f}<br>"
                        f"PnL: {tr['pnl_pips']:+.2f} pips / ${pnl_str} USD<br>"
                        f"Duração: {tr['duracao_candles']} candles"
                        "<extra></extra>"
                    ),
                    legendgroup=f"trade_{k}",
                    showlegend=False,
                ),
                row=1, col=1,
            )

    # ── PAINEL 2: Volume ──────────────────────────────────────────────────────
    cores_vol = [
        "#26A69A" if c >= o else "#EF5350"
        for c, o in zip(df_semana['Close'], df_semana['Open'])
    ]
    fig.add_trace(
        go.Bar(
            x=semana_timestamps,
            y=df_semana['Volume'],
            name="Volume",
            marker_color=cores_vol,
            opacity=0.7,
            hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>Volume: %{y:,.0f}<extra></extra>",
        ),
        row=2, col=1,
    )

    # ── Layout global dark mode ───────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=(
                f"<b>Grid 1 — Daily Close | EURUSD H1</b><br>"
                f"<sup>Semana: {data_inicio.strftime('%Y-%m-%d')} → {data_fim.strftime('%Y-%m-%d')} | "
                f"{len(trades_visiveis)} trade(s) no período</sup>"
            ),
            font=dict(size=18, color="#ECEFF1"),
            x=0.5,
        ),
        paper_bgcolor="#0D1117",
        plot_bgcolor="#161B22",
        font=dict(color="#ECEFF1", family="Inter, Arial, sans-serif", size=12),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend=dict(
            bgcolor="rgba(22, 27, 34, 0.85)",
            bordercolor="#30363D",
            borderwidth=1,
            font=dict(size=11),
            x=1.01,
            y=1,
        ),
        height=820,
        margin=dict(l=60, r=200, t=100, b=40),
    )

    # Estilo dos eixos
    axis_style = dict(
        gridcolor="#21262D",
        zerolinecolor="#30363D",
        tickfont=dict(color="#8B949E"),
        title_font=dict(color="#8B949E"),
    )
    fig.update_xaxes(**axis_style, showspikes=True, spikecolor="#58A6FF", spikethickness=1)
    fig.update_yaxes(**axis_style, showspikes=True, spikecolor="#58A6FF", spikethickness=1)
    fig.update_yaxes(title_text="Preço EURUSD", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_xaxes(title_text="Data / Hora (servidor MT5)", row=2, col=1)

    # Adicionar anotações para o pct_gatilho referencial
    pct_ref = PARAMS['pct_gatilho'] * 100
    fig.add_annotation(
        text=f"Gatilho: ±{pct_ref:.2f}% do daily close (linha pontilhada laranja)",
        xref="paper", yref="paper",
        x=0.01, y=-0.08,
        showarrow=False,
        font=dict(size=11, color="#FFA726"),
        align="left",
    )

    # ── Salvar HTML ───────────────────────────────────────────────────────────
    caminho_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(caminho_html),
        include_plotlyjs="cdn",
        full_html=True,
        config={
            "scrollZoom": True,
            "displayModeBar": True,
            "modeBarButtonsToAdd": ["drawline", "drawopenpath", "eraseshape"],
            "toImageButtonOptions": {"format": "png", "filename": "grid1_velas", "scale": 2},
        },
    )
    print(f"  ✅ HTML interativo salvo em: {caminho_html.resolve()}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

PARAMS = {
    'pct_gatilho':      0.003,
    'mult_espacamento': 1.0,
    'mult_alvo':        1.5,
    'mult_stop':        3.0,
    'stop_candles':     120,
}

def main():
    parser = argparse.ArgumentParser(description="Gera gráficos do Grid 1 (Daily Close)")
    parser.add_argument(
        "--semana", type=str, default=None,
        help="Data de início da semana para o gráfico de velas (YYYY-MM-DD). Default: primeira semana com trades."
    )
    parser.add_argument(
        "--trade", type=int, default=None,
        help="Número do trade (1-based) para centralizar a semana do gráfico de velas."
    )
    args = parser.parse_args()

    print("\n═" * 36)
    print("  Grid 1 — Daily Close | Gerando Gráficos")
    print("═" * 72)

    # ── Carregar dados ────────────────────────────────────────────────────────
    if not PARQUET_COMPLETO.exists():
        print(f"[ERRO] Parquet não encontrado: {PARQUET_COMPLETO}")
        sys.exit(1)

    print(f"\n  Carregando {PARQUET_COMPLETO.name}...")
    df = pd.read_parquet(PARQUET_COMPLETO, engine="pyarrow")
    if 'log_return' not in df.columns:
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    print(f"  {len(df):,} candles | {df.index[0]} → {df.index[-1]}")

    # ── Rodar backtest ────────────────────────────────────────────────────────
    print(f"\n  Rodando backtest Grid 1...")
    resultado = rodar_backtest_grid(
        df=df, tipo_grid=1, params=PARAMS,
        capital_inicial=CAPITAL_INICIAL, verbose=False,
    )
    trades = resultado['trades']
    print(f"  {len(trades)} trades concluídos.")

    # ── 1. PNG da equity curve (4 painéis) ───────────────────────────────────
    DIR_GRAFICOS.mkdir(parents=True, exist_ok=True)
    caminho_png = DIR_GRAFICOS / "grid1_completo_backtest.png"
    print(f"\n  Gerando PNG (equity curve + drawdown)...")
    gerar_grafico_backtest(resultado, 1, PARAMS, caminho_png, df=df)

    # ── 2. HTML interativo com velas ──────────────────────────────────────────
    # Determinar semana a exibir
    if args.semana:
        data_inicio = pd.Timestamp(args.semana)
    elif args.trade and 1 <= args.trade <= len(trades):
        tr = trades[args.trade - 1]
        ts_trade = df.index[tr['candle_inicio']]
        # Recuar até a segunda-feira da semana do trade
        data_inicio = ts_trade - timedelta(days=ts_trade.weekday())
        data_inicio = data_inicio.replace(hour=0, minute=0, second=0)
        print(f"  Trade #{args.trade} → semana começando em {data_inicio.strftime('%Y-%m-%d')}")
    else:
        # Default: semana do primeiro trade
        ts_primeiro = df.index[trades[0]['candle_inicio']]
        data_inicio = ts_primeiro - timedelta(days=ts_primeiro.weekday())
        data_inicio = data_inicio.replace(hour=0, minute=0, second=0)
        print(f"  Semana padrão (1º trade): {data_inicio.strftime('%Y-%m-%d')}")

    caminho_html = DIR_GRAFICOS / f"grid1_velas_{data_inicio.strftime('%Y-%m-%d')}.html"
    print(f"\n  Gerando HTML interativo de velas...")
    gerar_grafico_velas_interativo(df, trades, data_inicio, caminho_html)

    print(f"\n  📊 PNG:  {caminho_png.resolve()}")
    print(f"  🌐 HTML: {caminho_html.resolve()}")
    print(f"\n{'═' * 72}\n")


if __name__ == "__main__":
    main()
