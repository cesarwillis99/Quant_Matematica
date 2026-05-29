#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
monte_carlo.py
--------------
Script UNIFICADO, GENÉRICO e AUTOCONTIDO para testes de robustez de Monte Carlo.
Fusão da engine matemática de simulação com o orquestrador CLI de alta performance.

Uso:
  python monte_carlo.py --ativo EURUSD --timeframe H1 --estrategia MOMENTUM
  python monte_carlo.py --ativo EURUSD --timeframe H1 --estrategia ALL
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

def rodar_monte_carlo(
    caminho_csv: str,
    nome_estrategia: str,
    nome_ativo: str,
    caminho_saida: str,
    n_simulacoes: int = 10000,
    prob_skip: float = 0.05
) -> dict:
    
    df = pd.read_csv(caminho_csv)
    if len(df) == 0:
        return None
        
    pnl = df['pnl_monetario'].values
    n_trades = len(pnl)
    
    # Métricas Originais
    orig_lucro = pnl.sum()
    orig_cum = np.cumsum(pnl)
    orig_peak = np.maximum.accumulate(orig_cum)
    orig_dd = orig_peak - orig_cum
    orig_max_dd = orig_dd.max() if orig_dd.max() > 0 else 1.0
    orig_ret_dd = orig_lucro / orig_max_dd
    
    ganhos_orig = pnl[pnl > 0].sum()
    perdas_orig = abs(pnl[pnl < 0].sum())
    orig_pf = ganhos_orig / perdas_orig if perdas_orig > 0 else 999.0
    
    # Matriz para 10.000 simulações vetorizada (muito mais rápido que loop for)
    indices = np.random.randint(0, n_trades, size=(n_simulacoes, n_trades))
    pnl_sims = pnl[indices]
    
    # Simulação de falhas (Random Skip)
    keep_mask = np.random.rand(n_simulacoes, n_trades) >= prob_skip
    pnl_sims = pnl_sims * keep_mask
    
    # Métricas para cada simulação
    sim_lucros = pnl_sims.sum(axis=1)
    
    sim_cums = np.cumsum(pnl_sims, axis=1)
    sim_peaks = np.maximum.accumulate(sim_cums, axis=1)
    sim_dds = sim_peaks - sim_cums
    sim_max_dds = sim_dds.max(axis=1)
    sim_max_dds[sim_max_dds == 0] = 1.0 # Evitar divisões por zero
    
    sim_ret_dds = sim_lucros / sim_max_dds
    
    ganhos_sims = np.where(pnl_sims > 0, pnl_sims, 0).sum(axis=1)
    perdas_sims = np.abs(np.where(pnl_sims < 0, pnl_sims, 0).sum(axis=1))
    
    safe_perdas = np.where(perdas_sims == 0, 1.0, perdas_sims)
    sim_pfs = ganhos_sims / safe_perdas
    sim_pfs[perdas_sims == 0] = 999.0
    
    # Níveis de Confiança (Percentis inferiores para garantir o Pior Cenário)
    ret_dd_50 = np.percentile(sim_ret_dds, 50)
    ret_dd_70 = np.percentile(sim_ret_dds, 30) # 70% das simulações foram melhores que isso
    pf_99 = np.percentile(sim_pfs, 1)          # 99% das simulações foram melhores que isso
    
    passou_ex1 = ret_dd_50 >= 1.5
    passou_ex2 = ret_dd_70 >= (0.40 * orig_ret_dd)
    passou_ex3 = pf_99 >= 1.01
    
    aprovado = bool(passou_ex1 and passou_ex2 and passou_ex3)
    
    res = {
        'estrategia': nome_estrategia,
        'ativo': nome_ativo,
        'n_simulacoes': n_simulacoes,
        'orig_ret_dd': orig_ret_dd,
        'ret_dd_50': ret_dd_50,
        'ret_dd_70': ret_dd_70,
        'orig_pf': orig_pf,
        'pf_99': pf_99,
        'passou_ex1': passou_ex1,
        'passou_ex2': passou_ex2,
        'passou_ex3': passou_ex3,
        'aprovado': aprovado
    }
    
    gerar_jpg(res, caminho_saida)
    
    return res

def gerar_jpg(res: dict, caminho_saida: str):
    plt.rcParams.update({
        "font.family": "monospace",
    })
    
    caminho = Path(caminho_saida)
    caminho.mkdir(parents=True, exist_ok=True)
    arquivo_saida = caminho / f"resultado_monte_carlo_{res['estrategia']}.jpg"
    
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    fig.patch.set_facecolor('#0D1117')
    ax.set_facecolor('#0D1117')
    ax.axis('off')
    
    cor_titulo = '#E6EDF3'
    cor_texto  = '#CFD8DC'
    cor_linha  = '#21262D'
    cor_ok     = '#3FB950'
    cor_fail   = '#F85149'
    
    # Cabeçalho
    ax.text(0.5, 0.94, f"{res['estrategia'].upper()} — {res['ativo'].upper()}",
            color=cor_titulo, fontsize=14, fontweight='bold', ha='center')
    ax.text(0.5, 0.88, f"Monte Carlo - Trade Manipulation ({res['n_simulacoes']:,} Runs)",
            color=cor_titulo, fontsize=11, ha='center')
            
    ax.add_patch(patches.Rectangle((0.05, 0.82), 0.9, 0.002, color=cor_linha, transform=ax.transAxes))
    
    y = 0.72
    lh = 0.07
    
    # Exigencia 1
    ax.text(0.08, y, "Exigência 1: Ret/DD (50% Conf.) >= 1.5", color=cor_titulo, fontsize=11)
    st1 = "[ OK ]" if res['passou_ex1'] else "[ FAIL ]"
    c1 = cor_ok if res['passou_ex1'] else cor_fail
    ax.text(0.08, y - lh, f"[Orig: {res['orig_ret_dd']:.2f}]  vs  [Sim 50%: {res['ret_dd_50']:.2f}]", color=cor_texto, fontsize=10)
    ax.text(0.85, y - lh, st1, color=c1, fontsize=11, fontweight='bold', ha='center')
    
    y -= 3*lh
    
    # Exigencia 2
    alvo_ex2 = 0.40 * res['orig_ret_dd']
    ax.text(0.08, y, "Exigência 2: Ret/DD (70% Conf.) >= 40% do Original", color=cor_titulo, fontsize=11)
    st2 = "[ OK ]" if res['passou_ex2'] else "[ FAIL ]"
    c2 = cor_ok if res['passou_ex2'] else cor_fail
    ax.text(0.08, y - lh, f"[Alvo: {alvo_ex2:.2f}]  vs  [Sim 70%: {res['ret_dd_70']:.2f}]", color=cor_texto, fontsize=10)
    ax.text(0.85, y - lh, st2, color=c2, fontsize=11, fontweight='bold', ha='center')
    
    y -= 3*lh
    
    # Exigencia 3
    ax.text(0.08, y, "Exigência 3: Profit Factor (99% Conf.) >= 1.01", color=cor_titulo, fontsize=11)
    st3 = "[ OK ]" if res['passou_ex3'] else "[ FAIL ]"
    c3 = cor_ok if res['passou_ex3'] else cor_fail
    ax.text(0.08, y - lh, f"[Orig: {res['orig_pf']:.2f}]  vs  [Sim 99%: {res['pf_99']:.2f}]", color=cor_texto, fontsize=10)
    ax.text(0.85, y - lh, st3, color=c3, fontsize=11, fontweight='bold', ha='center')
    
    ax.add_patch(patches.Rectangle((0.05, 0.22), 0.9, 0.002, color=cor_linha, transform=ax.transAxes))
    
    # Resultado Final
    if res['aprovado']:
        texto_resultado = "APROVADO [OK]"
        cor_res = cor_ok
    else:
        texto_resultado = "REPROVADO [X]"
        cor_res = cor_fail
        
    ax.text(0.5, 0.10, texto_resultado, color=cor_res, fontsize=22, fontweight='bold', ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(arquivo_saida, format='jpg', bbox_inches='tight', facecolor='#0D1117', dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Teste de Robustez por Simulações de Monte Carlo Genérico")
    parser.add_argument("--ativo",      type=str, default="EURUSD", help="Ativo a ser testado (ex: EURUSD)")
    parser.add_argument("--timeframe",  type=str, default="H1", help="Timeframe (ex: H1)")
    parser.add_argument("--estrategia", type=str, default="ALL", help="Estratégia específica ou ALL para rodar todas")
    parser.add_argument("--n_sim",      type=int, default=10000, help="Número de simulações")
    parser.add_argument("--prob_skip",  type=float, default=0.05, help="Probabilidade de skip trade")
    args = parser.parse_args()

    ativo      = args.ativo.lower()
    timeframe  = args.timeframe.lower()
    estrategia = args.estrategia.lower()

    dir_projeto = Path(__file__).resolve().parent.parent.parent
    dir_saida = dir_projeto / "testes_robustez" / "monte_carlo" / f"{ativo}_{timeframe}"
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
    print(f"  MONTE CARLO ROBUSTNESS TEST -- MOTOR UNIFICADO")
    print(f"  Ativo-Timeframe   : {ativo.upper()} {timeframe.upper()}")
    print(f"  Simulações        : {args.n_sim:,} cenários | Skip Prob: {args.prob_skip*100:.1f}%")
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
            print(f"[RODANDO] Processando Monte Carlo (matrizes) para {est.upper()}...")
            res = rodar_monte_carlo(
                caminho_csv=str(caminho_csv),
                nome_estrategia=est,
                nome_ativo=ativo,
                caminho_saida=str(dir_saida),
                n_simulacoes=args.n_sim,
                prob_skip=args.prob_skip
            )
            if res is not None:
                resultados.append(res)
                print(f"[OK] {est.upper()} - {args.n_sim:,} cenários calculados e avaliados.")
        except Exception as e:
            print(f"[ERRO] Ocorreu um erro no teste da estratégia {est.upper()}: {e}")
            continue

    if not resultados:
        print("\n[AVISO] Nenhum teste de Monte Carlo foi processado com sucesso. Verifique os CSVs.")
        return

    # Imprimir tabela de resumo técnica bonita no terminal
    print("\n" + "="*100)
    print(f"RESUMO DO TESTE DE ROBUSTEZ: MONTE CARLO ({args.n_sim:,} RUNS)")
    print("="*100)
    print(f"{'Estratégia':<15} | {'Ex.1 (>=1.5)':<15} | {'Ex.2 (>=40% Orig)':<20} | {'Ex.3 (PF>=1.01)':<15} | {'Resultado'}")
    print("-" * 100)
    for r in resultados:
        res_str = "APROVADO [OK]" if r['aprovado'] else "REPROVADO [X]"
        e1 = f"{r['ret_dd_50']:>6.2f} " + ("(OK)" if r['passou_ex1'] else "(X)")
        e2 = f"{r['ret_dd_70']:>6.2f} " + ("(OK)" if r['passou_ex2'] else "(X)")
        e3 = f"{r['pf_99']:>6.2f} " + ("(OK)" if r['passou_ex3'] else "(X)")
        
        print(f"{r['estrategia'].upper():<15} | {e1:<15} | {e2:<20} | {e3:<15} | {res_str}")
    print("="*100 + "\n")


if __name__ == "__main__":
    main()
