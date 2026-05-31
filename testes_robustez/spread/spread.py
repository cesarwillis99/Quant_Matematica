#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spread.py
---------
Script UNIFICADO, GENÉRICO e AUTOCONTIDO para testes de robustez de Spread (Spread Stress Test).
Fusão da engine matemática com o orquestrador CLI de alta performance.

Uso:
  python spread.py --ativo EURUSD --timeframe H1 --estrategia MOMENTUM
  python spread.py --ativo EURUSD --timeframe H1 --estrategia ALL
"""

import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

warnings_opt = matplotlib.rcParams
matplotlib.use("Agg")

# Lista padrão de estratégias do projeto
ESTRATEGIAS_PADRAO = [
    "zscore",
    "hawkes",
    "momentum",
    "ou",
    "ou_reverso",
    "pca",
    "wavelet",
    "curvatura"
]

def rodar_spread(
    caminho_csv: str,
    nome_estrategia: str,
    nome_ativo: str,
    caminho_saida: str,
    spread_original: float = 0.5,
    spread_multiplo: float = 1.8,
    pip_value_por_lot: float = 10.0,
    max_degradacao: float = 0.20,
) -> dict:
    
    # 1. Carregar CSV
    df = pd.read_csv(caminho_csv)
    total_trades = len(df)
    
    if total_trades == 0:
        return None
        
    # 2. Calcular o spread extra por trade
    spread_extra_pips = spread_original * (spread_multiplo - 1.0)
    custo_extra_trade = spread_extra_pips * df['lot_size'] * pip_value_por_lot
    
    # 3. Calcular pnl ajustado
    df['pnl_ajustado'] = df['pnl_monetario'] - custo_extra_trade
    
    # 4. Cálculos globais
    lucro_original = df['pnl_monetario'].sum()
    lucro_ajustado = df['pnl_ajustado'].sum()
    custo_total = custo_extra_trade.sum()
    
    if lucro_original != 0:
        impacto_pct = (custo_total / abs(lucro_original)) * 100
    else:
        impacto_pct = 0.0
        
    # 5. Critério de aprovação SQX (lucro mantido >= lucro original deduzido da max_degradacao)
    aprovado = lucro_ajustado >= ((1.0 - max_degradacao) * lucro_original)
    
    res = {
        'estrategia': nome_estrategia,
        'ativo': nome_ativo,
        'total_trades': total_trades,
        'spread_original': spread_original,
        'spread_novo': spread_original * spread_multiplo,
        'spread_extra': spread_extra_pips,
        'lucro_original': lucro_original,
        'custo_total': custo_total,
        'lucro_ajustado': lucro_ajustado,
        'impacto_pct': impacto_pct,
        'aprovado': aprovado,
    }
    
    # 6. Geração do PNG Dark Premium
    gerar_png(res, caminho_saida)
    
    return res

def gerar_png(res: dict, caminho_saida: str):
    caminho_saida_path = Path(caminho_saida)
    caminho_saida_path.mkdir(parents=True, exist_ok=True)
    arquivo_saida = caminho_saida_path / f"spread_{res['estrategia']}_{res['ativo']}.png"
    
    plt.rcParams.update({
        "font.family": "monospace",
    })
    
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    fig.patch.set_facecolor('#0D1117')
    ax.set_facecolor('#0D1117')
    ax.axis('off')
    
    cor_titulo = '#E6EDF3'
    cor_label  = '#8B949E'
    cor_valor  = '#58A6FF'
    cor_pct    = '#FF7B72'
    cor_aprov  = '#3FB950'
    cor_reprov = '#F85149'
    cor_linha  = '#21262D'
    
    # Top Text
    ax.text(0.5, 0.93, f"{res['estrategia'].upper()} — {res['ativo'].upper()}",
            color=cor_titulo, fontsize=14, fontweight='bold', ha='center')
    
    spread_inc_pct = ((res['spread_novo'] / res['spread_original']) - 1.0) * 100
    ax.text(0.5, 0.86, f"SPREAD STRESS TEST: +{spread_inc_pct:.0f}% de Spread",
            color=cor_titulo, fontsize=11, ha='center')
            
    # Separator 1
    ax.add_patch(patches.Rectangle((0.05, 0.80), 0.9, 0.002, color=cor_linha, transform=ax.transAxes))
    
    # Layout params
    x_lbl = 0.10
    x_col = 0.45
    x_val = 0.48
    y_start = 0.68
    lh = 0.08
    
    # Row 1
    ax.text(x_lbl, y_start, "Spread Original", color=cor_label, fontsize=11)
    ax.text(x_col, y_start, ":", color=cor_label, fontsize=11)
    ax.text(x_val, y_start, f"{res['spread_original']:.1f} pip", color=cor_titulo, fontsize=11)
    
    # Row 2
    ax.text(x_lbl, y_start - lh, "Spread Testado", color=cor_label, fontsize=11)
    ax.text(x_col, y_start - lh, ":", color=cor_label, fontsize=11)
    ax.text(x_val, y_start - lh, f"{res['spread_novo']:.1f} pip  (+{spread_inc_pct:.0f}%)", color=cor_titulo, fontsize=11)
    
    # Row 3
    ax.text(x_lbl, y_start - 2*lh, "Total de Trades", color=cor_label, fontsize=11)
    ax.text(x_col, y_start - 2*lh, ":", color=cor_label, fontsize=11)
    ax.text(x_val, y_start - 2*lh, f"{res['total_trades']}", color=cor_titulo, fontsize=11)
    
    y_start2 = y_start - 3.5 * lh
    
    # Row 4
    ax.text(x_lbl, y_start2, "Lucro Original", color=cor_label, fontsize=11)
    ax.text(x_col, y_start2, ":", color=cor_label, fontsize=11)
    ax.text(x_val, y_start2, f"$ {res['lucro_original']:,.2f}", color=cor_valor, fontsize=11)
    
    # Row 5
    ax.text(x_lbl, y_start2 - lh, "Custo Extra Total", color=cor_label, fontsize=11)
    ax.text(x_col, y_start2 - lh, ":", color=cor_label, fontsize=11)
    ax.text(x_val, y_start2 - lh, f"$ {res['custo_total']:,.2f}  ({res['impacto_pct']:.2f}% do lucro)", color=cor_pct, fontsize=11)
    
    # Row 6
    ax.text(x_lbl, y_start2 - 2*lh, "Lucro Ajustado", color=cor_label, fontsize=11)
    ax.text(x_col, y_start2 - 2*lh, ":", color=cor_label, fontsize=11)
    ax.text(x_val, y_start2 - 2*lh, f"$ {res['lucro_ajustado']:,.2f}", color=cor_valor, fontsize=11)
    
    # Separator 2
    ax.add_patch(patches.Rectangle((0.05, 0.28), 0.9, 0.002, color=cor_linha, transform=ax.transAxes))
    
    # Result
    if res['aprovado']:
        texto_resultado = "APROVADO [OK]"
        cor_res = cor_aprov
    else:
        texto_resultado = "REPROVADO [X]"
        cor_res = cor_reprov
        
    ax.text(0.5, 0.15, texto_resultado, color=cor_res, fontsize=20, fontweight='bold', ha='center', va='center')
    
    # Separator 3
    ax.add_patch(patches.Rectangle((0.05, 0.02), 0.9, 0.002, color=cor_linha, transform=ax.transAxes))
    
    plt.tight_layout()
    plt.savefig(arquivo_saida, bbox_inches='tight', facecolor='#0D1117', dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Teste de Robustez por Estresse de Spread Genérico e Unificado")
    parser.add_argument("--ativo",      type=str, default="EURUSD", help="Ativo a ser testado (ex: EURUSD)")
    parser.add_argument("--timeframe",  type=str, default="H1", help="Timeframe (ex: H1)")
    parser.add_argument("--estrategia", type=str, default="ALL", help="Estratégia específica ou ALL para rodar todas")
    parser.add_argument("--spread_orig", type=float, default=0.5, help="Spread original em pips")
    parser.add_argument("--multiplo",    type=float, default=1.8, help="Múltiplo de estresse (ex: 1.8 para +80%)")
    args = parser.parse_args()

    ativo      = args.ativo.lower()
    timeframe  = args.timeframe.lower()
    estrategia = args.estrategia.lower()

    dir_projeto = Path(__file__).resolve().parent.parent.parent
    dir_saida = dir_projeto / "testes_robustez" / "spread" / f"{ativo}_{timeframe}"
    dir_saida.mkdir(parents=True, exist_ok=True)

    # Determinar quais estratégias serão executadas
    lista_estrategias = []
    if estrategia == "all":
        lista_estrategias = ESTRATEGIAS_PADRAO
    else:
        lista_estrategias = [estrategia]

    # Corrige encoding de stdout para Windows
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    print(f"\n{'='*95}")
    print(f"  SPREAD STRESS TEST (ESTRESSE DE SPREAD) -- MOTOR UNIFICADO")
    print(f"  Ativo-Timeframe   : {ativo.upper()} {timeframe.upper()}")
    print(f"  Estresse          : {args.spread_orig:.1f} pips x {args.multiplo:.1f}x = {args.spread_orig * args.multiplo:.1f} pips")
    print(f"  Estratégias       : {', '.join([e.upper() for e in lista_estrategias])}")
    print(f"  Saída de Imagens  : {dir_saida}")
    print(f"{'='*95}\n")

    resultados = []

    for est in lista_estrategias:
        # Caminho dinâmico do CSV de operações originais da estratégia
        caminho_csv = dir_projeto / f"quant_{ativo}" / "resultados" / f"operacoes_{est}_{ativo}.csv"
        
        if not caminho_csv.exists():
            print(f"[AVISO] Arquivo de operações não encontrado para {est.upper()}: {caminho_csv}. Pulando...")
            continue

        try:
            print(f"[RODANDO] Processando estresse de spread para {est.upper()}...")
            res = rodar_spread(
                caminho_csv=str(caminho_csv),
                nome_estrategia=est,
                nome_ativo=ativo,
                caminho_saida=str(dir_saida),
                spread_original=args.spread_orig,
                spread_multiplo=args.multiplo,
                pip_value_por_lot=10.0
            )
            if res is not None:
                resultados.append(res)
        except Exception as e:
            print(f"[ERRO] Ocorreu um erro no teste da estratégia {est.upper()}: {e}")
            continue

    if not resultados:
        print("\n[AVISO] Nenhum teste de spread foi processado com sucesso. Verifique os CSVs.")
        return

    # Imprimir tabela de resumo técnica bonita no terminal
    print("\n" + "="*110)
    print(f"RESUMO DO TESTE DE ROBUSTEZ: SPREAD STRESS TEST (+{(args.multiplo - 1.0)*100:.0f}%)")
    print("="*110)
    print(f"{'Estratégia':<15} | {'Trades':<8} | {'Sp.Orig':<8} | {'Sp.Novo':<8} | {'L. Original':<12} | {'L. Ajustado':<12} | {'Impacto %':<10} | {'Resultado'}")
    print("-" * 110)
    for r in resultados:
        res_str = "APROVADO [OK]" if r['aprovado'] else "REPROVADO [X]"
        print(f"{r['estrategia'].upper():<15} | {r['total_trades']:<8} | {r['spread_original']:<8.1f} | {r['spread_novo']:<8.1f} | ${r['lucro_original']:<11.2f} | ${r['lucro_ajustado']:<11.2f} | {r['impacto_pct']:<9.2f}% | {res_str}")
    print("="*110 + "\n")


if __name__ == "__main__":
    main()
