# -*- coding: utf-8 -*-
"""
gerar_equity_curve.py — Gera dois arquivos:
  1. PNG dark-mode da equity curve completa (matplotlib)
  2. HTML interativo com Plotly embutido (sem dependência CDN)

Uso:
    python -m quant_grid.gerar_equity_curve
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates

from quant_grid.config import CAPITAL_INICIAL, PARQUET_COMPLETO, DIR_GRAFICOS
from quant_grid.run_backtest_grid import rodar_backtest_grid

PARAMS = {
    'pct_gatilho':      0.003,
    'mult_espacamento': 1.0,
    'mult_alvo':        1.5,
    'mult_stop':        3.0,
    'stop_candles':     120,
}

# ─────────────────────────────────────────────────────────────────────────────
# Carregar dados e rodar backtest
# ─────────────────────────────────────────────────────────────────────────────
print("Carregando dados...")
df = pd.read_parquet(PARQUET_COMPLETO, engine="pyarrow")
if "log_return" not in df.columns:
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
print(f"  {len(df):,} candles | {df.index[0].date()} → {df.index[-1].date()}")

print("Rodando backtest Grid 1 (Daily Close)...")
resultado = rodar_backtest_grid(
    df=df, tipo_grid=1, params=PARAMS,
    capital_inicial=CAPITAL_INICIAL, verbose=False,
)

trades        = resultado["trades"]
equity_curve  = resultado["equity_curve"]
pnl_total     = resultado["pnl_total_usd"]
pnl_pct       = resultado["pnl_pct"]
win_rate      = resultado["win_rate"]
mdd           = resultado["max_drawdown_pct"]
sharpe        = resultado["sharpe"]
pf            = resultado["profit_factor"]
total_grids   = resultado["total_grids"]
capital_final = resultado["capital_final"]

timestamps = df.index[: len(equity_curve)]
eq_series  = pd.Series(equity_curve, index=timestamps, dtype=float)
peak       = eq_series.cummax()
dd_pct     = ((eq_series - peak) / peak) * 100.0

DIR_GRAFICOS.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. PNG — Equity Curve dark mode
# ─────────────────────────────────────────────────────────────────────────────
print("\nGerando PNG da equity curve...")

plt.style.use("dark_background")
fig, (ax_eq, ax_dd) = plt.subplots(
    2, 1, figsize=(18, 9),
    gridspec_kw={"height_ratios": [0.68, 0.32]},
    sharex=True,
)
fig.patch.set_facecolor("#0D1117")
ax_eq.set_facecolor("#161B22")
ax_dd.set_facecolor("#161B22")

# ── Equity curve ──────────────────────────────────────────────────────────────
ax_eq.plot(timestamps, eq_series.values, color="#58A6FF", linewidth=1.2, zorder=3, label="Equity")
ax_eq.axhline(CAPITAL_INICIAL, color="#FFFFFF", linestyle="--", linewidth=0.8, alpha=0.4, label=f"Capital inicial ${CAPITAL_INICIAL:,.0f}")

# Fill verde acima / vermelho abaixo do capital inicial
ax_eq.fill_between(timestamps, eq_series.values, CAPITAL_INICIAL,
                   where=(eq_series.values >= CAPITAL_INICIAL),
                   color="#00E676", alpha=0.10, interpolate=True)
ax_eq.fill_between(timestamps, eq_series.values, CAPITAL_INICIAL,
                   where=(eq_series.values < CAPITAL_INICIAL),
                   color="#FF1744", alpha=0.10, interpolate=True)

# Pontos de fechamento de trades
for tr in trades:
    ci = tr["candle_fim"]
    if ci >= len(timestamps):
        continue
    ts = timestamps[ci]
    eq = equity_curve[ci]
    cor = "#00E676" if tr["pnl_usd"] > 0 else "#FF1744"
    ax_eq.scatter(ts, eq, s=10, color=cor, zorder=4, alpha=0.65, linewidths=0)

# Anotação das métricas no gráfico
metricas_txt = (
    f"PnL Total: ${pnl_total:+,.2f}  ({pnl_pct:+.2f}%)   |   "
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

# Anotação ano por ano
for ano in range(df.index[0].year, df.index[-1].year + 1):
    ts_ano = pd.Timestamp(f"{ano}-01-01")
    if ts_ano > timestamps[-1]:
        break
    ax_eq.axvline(ts_ano, color="#30363D", linewidth=0.6, zorder=1)
    ax_eq.text(ts_ano, ax_eq.get_ylim()[0] if ax_eq.get_ylim()[0] else CAPITAL_INICIAL * 0.9,
               str(ano), color="#8B949E", fontsize=8, va="bottom", ha="left")

# ── Drawdown ──────────────────────────────────────────────────────────────────
ax_dd.fill_between(timestamps, dd_pct.values, 0, color="#FF1744", alpha=0.25, interpolate=True)
ax_dd.plot(timestamps, dd_pct.values, color="#FF1744", linewidth=0.8)
ax_dd.set_ylabel("Drawdown (%)", fontsize=11, color="#8B949E")
ax_dd.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax_dd.grid(True, linestyle="--", alpha=0.08, color="#FF1744")
ax_dd.tick_params(colors="#8B949E")
ax_dd.spines[:].set_color("#30363D")

# Eixo X com datas
ax_dd.xaxis.set_major_locator(mdates.YearLocator())
ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax_dd.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
plt.setp(ax_dd.xaxis.get_majorticklabels(), color="#8B949E", fontsize=9)

fig.suptitle(
    "Grid 1 — Daily Close | EURUSD H1 | 2016–2023",
    fontsize=15, fontweight="bold", color="#ECEFF1", y=0.99,
)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.subplots_adjust(hspace=0.06)

caminho_png = DIR_GRAFICOS / "grid1_equity_curve.png"
plt.savefig(caminho_png, dpi=160, bbox_inches="tight", facecolor="#0D1117")
plt.close()
print(f"  ✅ PNG salvo em: {caminho_png.resolve()}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. HTML — Equity Curve interativa (Plotly EMBUTIDO — sem CDN)
# ─────────────────────────────────────────────────────────────────────────────
print("\nGerando HTML interativo (plotly embutido)...")

import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig2 = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.06,
    row_heights=[0.68, 0.32],
    subplot_titles=["Equity Curve (USD)", "Drawdown (%)"],
)

# Equity — linha
fig2.add_trace(go.Scatter(
    x=timestamps, y=eq_series.values,
    mode="lines", name="Equity",
    line=dict(color="#58A6FF", width=1.4),
    hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>Equity: $%{y:,.2f}<extra></extra>",
), row=1, col=1)

# Fill verde
fig2.add_trace(go.Scatter(
    x=timestamps,
    y=np.where(eq_series >= CAPITAL_INICIAL, eq_series, CAPITAL_INICIAL),
    fill="tonexty", fillcolor="rgba(0,230,118,0.10)",
    line=dict(width=0), showlegend=False, hoverinfo="skip",
), row=1, col=1)

# Fill vermelho
fig2.add_trace(go.Scatter(
    x=timestamps,
    y=np.where(eq_series < CAPITAL_INICIAL, eq_series, CAPITAL_INICIAL),
    fill="tonexty", fillcolor="rgba(255,23,68,0.10)",
    line=dict(width=0), showlegend=False, hoverinfo="skip",
), row=1, col=1)

# Linha capital inicial
fig2.add_hline(
    y=CAPITAL_INICIAL, line_dash="dash",
    line_color="rgba(255,255,255,0.35)", line_width=1,
    row=1, col=1,
    annotation_text=f"  Capital inicial ${CAPITAL_INICIAL:,.0f}",
    annotation_font_color="#8B949E", annotation_font_size=11,
)

# Pontos de trades (agrupados por motivo para performance)
for motivo, sym, cor in [
    ("ALVO",     "star",          "#00E676"),
    ("DRAWDOWN", "x",             "#FF1744"),
    ("TEMPO",    "triangle-up",   "#FFEA00"),
    ("SEXTA",    "circle",        "#90A4AE"),
]:
    ts_m  = [timestamps[tr["candle_fim"]] for tr in trades
              if tr["motivo_saida"] == motivo and tr["candle_fim"] < len(timestamps)]
    eq_m  = [equity_curve[tr["candle_fim"]] for tr in trades
              if tr["motivo_saida"] == motivo and tr["candle_fim"] < len(timestamps)]
    pnl_m = [tr["pnl_usd"] for tr in trades
              if tr["motivo_saida"] == motivo and tr["candle_fim"] < len(timestamps)]
    if not ts_m:
        continue
    fig2.add_trace(go.Scatter(
        x=ts_m, y=eq_m,
        mode="markers",
        name=f"Saída {motivo}",
        marker=dict(size=7, color=cor, symbol=sym,
                    opacity=0.75, line=dict(width=0.5, color="white")),
        customdata=pnl_m,
        hovertemplate=(
            "<b>%{x|%Y-%m-%d %H:%M}</b><br>"
            "Equity: $%{y:,.2f}<br>"
            f"Motivo: {motivo}<br>"
            "PnL: $%{customdata:+,.2f}"
            "<extra></extra>"
        ),
    ), row=1, col=1)

# Drawdown
fig2.add_trace(go.Scatter(
    x=timestamps, y=dd_pct.values,
    mode="lines", name="Drawdown %",
    line=dict(color="#FF1744", width=0.8),
    fill="tozeroy", fillcolor="rgba(255,23,68,0.18)",
    hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>DD: %{y:.2f}%<extra></extra>",
), row=2, col=1)

# Anotação das métricas
pnl_str = f"+${pnl_total:,.2f}" if pnl_total >= 0 else f"-${abs(pnl_total):,.2f}"
fig2.add_annotation(
    xref="paper", yref="paper", x=0.01, y=0.97,
    text=(
        f"<b>Grid 1 — Daily Close | EURUSD H1</b><br>"
        f"Capital Final: ${capital_final:,.2f}  |  PnL: {pnl_str} ({pnl_pct:+.2f}%)<br>"
        f"Trades: {total_grids:,}  |  Win Rate: {win_rate:.1f}%  |  "
        f"PF: {pf:.2f}  |  Sharpe: {sharpe:.2f}  |  MDD: {mdd:.1f}%"
    ),
    showarrow=False, align="left",
    bgcolor="rgba(22,27,34,0.90)",
    bordercolor="#30363D", borderwidth=1,
    font=dict(size=12, color="#ECEFF1"),
    xanchor="left", yanchor="top",
)

fig2.update_layout(
    title=dict(
        text="<b>Grid 1 — Daily Close | EURUSD H1 | 2016–2023</b>",
        font=dict(size=20, color="#ECEFF1"), x=0.5,
    ),
    paper_bgcolor="#0D1117",
    plot_bgcolor="#161B22",
    font=dict(color="#ECEFF1", family="Inter, Arial, sans-serif", size=12),
    hovermode="x unified",
    height=740,
    margin=dict(l=80, r=50, t=80, b=50),
    legend=dict(
        bgcolor="rgba(22,27,34,0.85)", bordercolor="#30363D", borderwidth=1,
        x=1.01, y=1, font=dict(size=11),
    ),
    xaxis2_rangeslider=dict(visible=True, thickness=0.04, bgcolor="#161B22"),
)

axis_style = dict(
    gridcolor="#21262D", zerolinecolor="#30363D",
    tickfont=dict(color="#8B949E"), title_font=dict(color="#8B949E"),
    showspikes=True, spikecolor="#58A6FF", spikethickness=1,
)
fig2.update_xaxes(**axis_style)
fig2.update_yaxes(**axis_style)
fig2.update_yaxes(title_text="Equity (USD)", tickprefix="$", row=1, col=1)
fig2.update_yaxes(title_text="Drawdown (%)", ticksuffix="%", row=2, col=1)
fig2.update_xaxes(title_text="Data / Hora (servidor MT5)", row=2, col=1)

# ⚠️  include_plotlyjs=True → embute o plotly.js inteiro no HTML
#     Isso garante que funciona offline e no preview do VS Code
caminho_html = DIR_GRAFICOS / "grid1_equity_curve.html"
fig2.write_html(
    str(caminho_html),
    include_plotlyjs=True,   # ← embutido, sem CDN
    full_html=True,
    config={
        "scrollZoom": True,
        "displayModeBar": True,
        "toImageButtonOptions": {
            "format": "png", "filename": "grid1_equity_curve", "scale": 2,
        },
    },
)
print(f"  ✅ HTML salvo em: {caminho_html.resolve()}")
print(f"\n  📊 PNG:  {caminho_png.name}")
print(f"  🌐 HTML: {caminho_html.name}\n")
