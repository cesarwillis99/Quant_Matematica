#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
what_if.py
----------
Script UNIFICADO, GENÉRICO e AUTOCONTIDO para testes de robustez "What If" (remoção de Top 1% trades).
Fusão da engine de cálculo com o orquestrador CLI de alta performance.

Uso:
  python what_if.py --ativo EURUSD --timeframe H1 --estrategia MOMENTUM
  python what_if.py --ativo EURUSD --timeframe H1 --estrategia ALL
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

def rodar_what_if(
    caminho_csv: str,
    nome_estrategia: str,
    nome_ativo: str,
    caminho_saida: str,
    pct_remocao: float = 0.01,
) -> dict:
    
    # 1. Carregar CSV
    df_completo = pd.read_csv(caminho_csv)
    total_trades = len(df_completo)
    
    if total_trades == 0:
        return None
        
    # Filtrar apenas lucros
    df_lucro = df_completo[df_completo['pnl_monetario'] > 0]
    
    # 2. Calcular N
    n_removidos = max(1, round(total_trades * pct_remocao))
    
    # 3. Top N trades
    df_top_n = df_lucro.sort_values(by='pnl_monetario', ascending=False).head(n_removidos)
    
    # 4. Cálculos Financeiros
    lucro_original = df_completo['pnl_monetario'].sum()
    valor_removido = df_top_n['pnl_monetario'].sum()
    lucro_after = lucro_original - valor_removido
    
    if lucro_original != 0:
        impacto_pct = (valor_removido / abs(lucro_original)) * 100
    else:
        impacto_pct = 0.0
        
    # 5. Critério de aprovação SQX (lucro depois da remoção >= 80% do original)
    aprovado = lucro_after >= (0.80 * lucro_original)
    
    res = {
        'estrategia': nome_estrategia,
        'ativo': nome_ativo,
        'total_trades': total_trades,
        'n_removidos': n_removidos,
        'lucro_original': lucro_original,
        'valor_removido': valor_removido,
        'lucro_after': lucro_after,
        'impacto_pct': impacto_pct,
        'aprovado': aprovado,
    }
    
    # 6. Geração do PNG Dark Premium
    gerar_png(res, caminho_saida)
    
    return res

def gerar_png(res: dict, caminho_saida: str):
    caminho_saida_path = Path(caminho_saida)
    caminho_saida_path.mkdir(parents=True, exist_ok=True)
    arquivo_saida = caminho_saida_path / f"what_if_{res['estrategia']}_{res['ativo']}.png"
    
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
    ax.text(0.5, 0.86, f"WHAT IF: Remoção dos Top 1% Trades",
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
    ax.text(x_lbl, y_start, "Trades totais", color=cor_label, fontsize=11)
    ax.text(x_col, y_start, ":", color=cor_label, fontsize=11)
    ax.text(x_val, y_start, f"{res['total_trades']}", color=cor_titulo, fontsize=11)
    
    # Row 2
    ax.text(x_lbl, y_start - lh, "Trades removidos", color=cor_label, fontsize=11)
    ax.text(x_col, y_start - lh, ":", color=cor_label, fontsize=11)
    ax.text(x_val, y_start - lh, f"{res['n_removidos']} (1% da amostra)", color=cor_titulo, fontsize=11)
    
    y_start2 = y_start - 2.5 * lh
    
    # Row 3
    ax.text(x_lbl, y_start2, "Lucro Original", color=cor_label, fontsize=11)
    ax.text(x_col, y_start2, ":", color=cor_label, fontsize=11)
    ax.text(x_val, y_start2, f"$ {res['lucro_original']:,.2f}", color=cor_valor, fontsize=11)
    
    # Row 4
    ax.text(x_lbl, y_start2 - lh, "Valor Removido", color=cor_label, fontsize=11)
    ax.text(x_col, y_start2 - lh, ":", color=cor_label, fontsize=11)
    ax.text(x_val, y_start2 - lh, f"$ {res['valor_removido']:,.2f}  ({res['impacto_pct']:.2f}% do total)", color=cor_pct, fontsize=11)
    
    # Row 5
    ax.text(x_lbl, y_start2 - 2*lh, "Lucro After", color=cor_label, fontsize=11)
    ax.text(x_col, y_start2 - 2*lh, ":", color=cor_label, fontsize=11)
    ax.text(x_val, y_start2 - 2*lh, f"$ {res['lucro_after']:,.2f}", color=cor_valor, fontsize=11)
    
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
    parser = argparse.ArgumentParser(description="Teste de Robustez 'What If' Genérico e Unificado (Remoção Top 1%)")
    parser.add_argument("--ativo",      type=str, default="EURUSD", help="Ativo a ser testado (ex: EURUSD)")
    parser.add_argument("--timeframe",  type=str, default="H1", help="Timeframe (ex: H1)")
    parser.add_argument("--estrategia", type=str, default="ALL", help="Estratégia específica ou ALL para rodar todas")
    parser.add_argument("--pct_remocao", type=float, default=0.01, help="Percentual de melhores trades a remover")
    args = parser.parse_args()

    ativo      = args.ativo.lower()
    timeframe  = args.timeframe.lower()
    estrategia = args.estrategia.lower()

    dir_projeto = Path(__file__).resolve().parent.parent.parent
    dir_saida = dir_projeto / "testes_robustez" / "what_if" / f"{ativo}_{timeframe}"
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
    print(f"  WHAT IF TEST (REMOÇÃO DOS TOP {args.pct_remocao*100:.0f}% TRADES) -- MOTOR UNIFICADO")
    print(f"  Ativo-Timeframe   : {ativo.upper()} {timeframe.upper()}")
    print(f"  Remoção           : Top {args.pct_remocao*100:.1f}% melhores trades")
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
            print(f"[RODANDO] Processando simulação What If para {est.upper()}...")
            res = rodar_what_if(
                caminho_csv=str(caminho_csv),
                nome_estrategia=est,
                nome_ativo=ativo,
                caminho_saida=str(dir_saida),
                pct_remocao=args.pct_remocao
            )
            if res is not None:
                resultados.append(res)
        except Exception as e:
            print(f"[ERRO] Ocorreu um erro no teste da estratégia {est.upper()}: {e}")
            continue

    if not resultados:
        print("\n[AVISO] Nenhum teste What If foi processado com sucesso. Verifique os CSVs.")
        return

    # Imprimir tabela de resumo técnica bonita no terminal
    print("\n" + "="*115)
    print(f"RESUMO DO TESTE DE ROBUSTEZ: WHAT IF (REMOÇÃO DOS TOP {args.pct_remocao*100:.0f}% TRADES)")
    print("="*115)
    print(f"{'Estratégia':<15} | {'Trades':<8} | {'Removidos':<9} | {'L. Original':<12} | {'L. Ajustado':<12} | {'Impacto %':<10} | {'Resultado'}")
    print("-" * 115)
    for r in resultados:
        res_str = "APROVADO [OK]" if r['aprovado'] else "REPROVADO [X]"
        print(f"{r['estrategia'].upper():<15} | {r['total_trades']:<8} | {r['n_removidos']:<9} | ${r['lucro_original']:<11.2f} | ${r['lucro_after']:<11.2f} | {r['impacto_pct']:<9.2f}% | {res_str}")
    print("="*115 + "\n")


if __name__ == "__main__":
    main()
