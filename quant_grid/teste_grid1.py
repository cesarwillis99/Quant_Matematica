# -*- coding: utf-8 -*-
"""
teste_grid1.py — Diagnóstico do Grid 1 (Daily Close) no EURUSD H1.

Roda o backtest completo e imprime os primeiros N trades com todos
os campos para verificar se a execução está correta.

Uso:
    python -m quant_grid.teste_grid1
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# ── Encoding seguro ───────────────────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# ── Imports do projeto ────────────────────────────────────────────────────────
from quant_grid.config import (
    CAPITAL_INICIAL, LOT_SIZE, FATOR_PIPS, JANELA_VR,
    PARQUET_COMPLETO, SPREAD_PIPS,
)
from quant_grid.run_backtest_grid import rodar_backtest_grid

# ── Parâmetros de teste ───────────────────────────────────────────────────────
PARAMS = {
    # Grid 1 — daily close
    'pct_gatilho':    0.003,       # 0.3% de afastamento do fechamento do dia anterior

    # Risco
    'mult_espacamento': 1.0,       # espaçamento = 1× VR (em pips)
    'mult_alvo':        1.5,       # alvo = 1.5× VR (em pips)
    'mult_stop':        3.0,       # stop_dd = 3× VR × n_ordens (em pips)
    'stop_candles':     120,       # 5 dias H1 antes do time-stop
}

N_TRADES_DETALHE = 10   # quantos trades imprimir com detalhe completo

SEP  = "═" * 72
SEP2 = "─" * 72

def print_trade(idx: int, tr: dict):
    """Imprime diagnóstico completo de um trade."""
    dir_str   = "COMPRA (+1)" if tr['direcao'] == 1 else "VENDA  (-1)"
    pnl_sinal = "✅ LUCRO" if tr['pnl_usd'] > 0 else "❌ PERDA"

    print(f"\n  {'TRADE':>5} #{idx+1:04d}  |  {pnl_sinal}")
    print(f"  {SEP2}")
    print(f"  Direção              : {dir_str}")
    print(f"  Candle abertura      : #{tr['candle_inicio']}")
    print(f"  Candle fechamento    : #{tr['candle_fim']}")
    print(f"  Duração              : {tr['duracao_candles']} candles")
    print(f"  Motivo saída         : {tr['motivo_saida']}")
    print(f"  Ordens preenchidas   : {tr['n_ordens']} / 8")
    print(f"  Lote total           : {tr['lot_total']:.2f}")
    print(f"  Espaçamento (pips)   : {tr['espacamento_pips']:.2f}")
    print(f"  VR na abertura (pips): {tr['vr_abertura']:.2f}")
    print()
    print(f"  Preço médio entrada  : {tr['preco_medio_entrada']:.5f}")
    print(f"  Preço saída          : {tr['preco_saida']:.5f}")
    print()
    print(f"  PnL (pips)           : {tr['pnl_pips']:+.2f}")
    print(f"  PnL (USD)            : {tr['pnl_usd']:+.2f}")
    print()

    # Níveis ativados com diagnóstico de spread
    spread_price = SPREAD_PIPS / (2.0 * FATOR_PIPS)
    print(f"  Níveis preenchidos (preco_entrada real com spread):")
    for k, preco_nivel in enumerate(tr['niveis_ativados']):
        # O preço nominal seria sem o spread
        if tr['direcao'] == 1:
            preco_nominal = preco_nivel + spread_price
        else:
            preco_nominal = preco_nivel - spread_price
        print(f"    [{k}] nominal={preco_nominal:.5f}  →  entrada_real={preco_nivel:.5f}  (Δ={abs(preco_nivel - preco_nominal)*FATOR_PIPS:.2f} pips spread)")


def main():
    print(f"\n{SEP}")
    print(f"  DIAGNÓSTICO — GRID 1 (Daily Close) | EURUSD H1")
    print(f"  Parquet: {PARQUET_COMPLETO.name}")
    print(SEP)

    # ── Carregar dados ────────────────────────────────────────────────────────
    if not PARQUET_COMPLETO.exists():
        print(f"\n  [ERRO] Arquivo não encontrado: {PARQUET_COMPLETO}")
        sys.exit(1)

    print(f"\n  Carregando dados...")
    df = pd.read_parquet(PARQUET_COMPLETO, engine="pyarrow")
    print(f"  {len(df):,} candles carregados.")
    print(f"  Período: {df.index[0]} → {df.index[-1]}")
    print(f"  Colunas: {list(df.columns)}")

    # Verificar se log_return existe, senão calcular
    if 'log_return' not in df.columns:
        print(f"\n  [INFO] Calculando log_return...")
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))

    print(f"\n  Parâmetros de teste:")
    for k, v in PARAMS.items():
        print(f"    {k:20s}: {v}")

    # ── Rodar backtest ────────────────────────────────────────────────────────
    print(f"\n  Rodando backtest Grid 1...")
    resultado = rodar_backtest_grid(
        df=df,
        tipo_grid=1,
        params=PARAMS,
        capital_inicial=CAPITAL_INICIAL,
        verbose=False,
    )

    trades = resultado['trades']

    # ── Sumário geral ─────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  RESULTADOS GERAIS")
    print(SEP)
    print(f"  Capital Inicial     : ${CAPITAL_INICIAL:>12,.2f}")
    print(f"  Capital Final       : ${resultado['capital_final']:>12,.2f}")
    print(f"  PnL Total           : ${resultado['pnl_total_usd']:>+12,.2f}  ({resultado['pnl_pct']:+.2f}%)")
    print(f"  Total de Trades     : {resultado['total_grids']:>12,}")
    print(f"  Win Rate            : {resultado['win_rate']:>12.2f}%")
    print(f"  Profit Factor       : {resultado['profit_factor']:>12.4f}")
    print(f"  Sharpe Ratio        : {resultado['sharpe']:>12.4f}")
    print(f"  Max Drawdown        : {resultado['max_drawdown_pct']:>11.2f}%")
    print(f"  Fator Recuperação   : {resultado['fator_recuperacao']:>12.4f}")
    print(f"  Média Ordens/Grid   : {resultado['media_ordens_por_grid']:>12.2f}")
    print(f"  Motivos de Saída    : {resultado['motivos_saida']}")

    # ── Distribuição de ordens ────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  DISTRIBUIÇÃO DE ORDENS PREENCHIDAS POR TRADE")
    print(SEP)
    from collections import Counter
    contagem_ordens = Counter(tr['n_ordens'] for tr in trades)
    for n_ord in sorted(contagem_ordens.keys()):
        pct = contagem_ordens[n_ord] / len(trades) * 100
        barra = "█" * int(pct / 2)
        print(f"  {n_ord} ordem(ns): {contagem_ordens[n_ord]:>5} trades  ({pct:5.1f}%)  {barra}")

    # ── Distribuição de motivos de saída ──────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  VERIFICAÇÃO DE SANIDADE — PRIMEIROS {N_TRADES_DETALHE} TRADES")
    print(SEP)

    for i, tr in enumerate(trades[:N_TRADES_DETALHE]):
        print_trade(i, tr)

    # ── Verificações de consistência ──────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  VERIFICAÇÕES DE CONSISTÊNCIA")
    print(SEP)

    erros = 0
    spread_price = SPREAD_PIPS / (2.0 * FATOR_PIPS)

    for i, tr in enumerate(trades):
        n = tr['n_ordens']
        niveis = tr['niveis_ativados']

        # 1. Número de níveis preenchidos deve bater com n_ordens
        if len(niveis) != n:
            print(f"  [ERRO Trade #{i+1}] n_ordens={n} mas niveis_ativados={len(niveis)}")
            erros += 1

        # 2. Verificar que preço médio bate com a média dos níveis
        if n > 0:
            media_calculada = sum(niveis) / len(niveis)
            diferenca = abs(tr['preco_medio_entrada'] - media_calculada)
            if diferenca > 1e-7:
                print(f"  [AVISO Trade #{i+1}] preco_medio_entrada={tr['preco_medio_entrada']:.7f} "
                      f"≠ media_niveis={media_calculada:.7f} (Δ={diferenca*FATOR_PIPS:.4f} pips)")
                erros += 1

        # 3. Verificar que lote total bate com n_ordens × LOT_SIZE
        lote_esperado = round(n * LOT_SIZE, 6)
        if abs(tr['lot_total'] - lote_esperado) > 1e-8:
            print(f"  [ERRO Trade #{i+1}] lot_total={tr['lot_total']:.4f} ≠ esperado={lote_esperado:.4f}")
            erros += 1

    if erros == 0:
        print(f"  ✅  Todas as verificações passaram! ({len(trades)} trades analisados)")
    else:
        print(f"\n  ❌  {erros} inconsistências encontradas!")

    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()
