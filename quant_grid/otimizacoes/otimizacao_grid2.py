# -*- coding: utf-8 -*-
"""
Otimização para o GRID 2 — Gatilho baseado na inclinação da média móvel.
"""

import os
import sys
import json
import logging
import itertools
import pandas as pd
import numpy as np
from pathlib import Path

from quant_grid.config import PARQUET_COMPLETO, DIR_OTIM, CAPITAL_INICIAL
from quant_grid.gatilhos.gatilho_mm_slope import gerar_sinal_mm_slope
from quant_grid.run_backtest_grid import rodar_backtest_grid

# Configuração do Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Garantir UTF-8 no Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# Grades de parâmetros
PARAMS_COMUNS = {
    'mult_espacamento': [0.3, 0.5, 0.8, 1.0, 1.5],
    'mult_alvo':        [0.5, 1.0, 1.5, 2.0, 3.0],
    'mult_stop':        [2.0, 3.0, 4.0, 5.0],
    'stop_candles':     [48, 72, 120, 168],
}

PARAMS_ESPECIFICOS = {
    'janela_mm':        [20, 50, 100, 200],
    'janela_slope':     [3, 5, 8],
    'threshold_slope':  [0.3, 0.5, 1.0, 1.5],
}

def rodar_otimizacao_grid2():
    """
    Executa a otimização de parâmetros por Grid Search para o Grid 2.
    """
    if not PARQUET_COMPLETO.exists():
        raise FileNotFoundError(
            f"Arquivo de dados para otimização não encontrado: {PARQUET_COMPLETO}\n"
            f"Verifique se o arquivo eurusd_h1_completo.parquet está no diretório correto."
        )
        
    logger.info("Carregando base de dados para otimização do Grid 2...")
    df = pd.read_parquet(PARQUET_COMPLETO)
    
    DIR_OTIM.mkdir(parents=True, exist_ok=True)
    
    resultados = []
    
    # Gerar produtos cartesianos
    chaves_esp, vals_esp = zip(*PARAMS_ESPECIFICOS.items())
    combinacoes_especificas = [dict(zip(chaves_esp, v)) for v in itertools.product(*vals_esp)]
    
    chaves_com, vals_com = zip(*PARAMS_COMUNS.items())
    combinacoes_comuns = [dict(zip(chaves_com, v)) for v in itertools.product(*vals_com)]
    
    total_combinacoes = len(combinacoes_especificas) * len(combinacoes_comuns)
    logger.info(f"Iniciando Grid Search do Grid 2: {total_combinacoes} combinações totais.")
    
    # Loop das combinações específicas (para pré-calcular o sinal)
    for i_esp, p_esp in enumerate(combinacoes_especificas):
        if (i_esp + 1) % 10 == 0 or i_esp == 0:
            logger.info(f"Pré-calculando sinal para parâmetros específicos ({i_esp + 1}/{len(combinacoes_especificas)}): {p_esp}")
            
        # Gera o sinal correspondente a essa configuração
        df_sinalizado = gerar_sinal_mm_slope(
            df, 
            janela_mm=p_esp['janela_mm'], 
            janela_slope=p_esp['janela_slope'], 
            threshold_slope=p_esp['threshold_slope']
        )
        
        # Loop das combinações comuns
        for p_com in combinacoes_comuns:
            # Combina os dicionários de parâmetros
            params_teste = {**p_esp, **p_com}
            
            try:
                # Rodar Backtest
                res = rodar_backtest_grid(df_sinalizado, tipo_grid=2, params=params_teste, capital_inicial=CAPITAL_INICIAL, verbose=False)
                
                # Filtro mínimo exigido: total_grids >= 30 e profit_factor > 1.0
                if res['total_grids'] >= 30 and res['profit_factor'] > 1.0:
                    resultados.append({
                        'janela_mm': params_teste['janela_mm'],
                        'janela_slope': params_teste['janela_slope'],
                        'threshold_slope': params_teste['threshold_slope'],
                        'mult_espacamento': params_teste['mult_espacamento'],
                        'mult_alvo': params_teste['mult_alvo'],
                        'mult_stop': params_teste['mult_stop'],
                        'stop_candles': params_teste['stop_candles'],
                        'total_grids': res['total_grids'],
                        'win_rate': res['win_rate'],
                        'profit_factor': res['profit_factor'],
                        'fator_recuperacao': res['fator_recuperacao'],
                        'max_drawdown_pct': res['max_drawdown_pct'],
                        'sharpe': res['sharpe'],
                        'capital_final': res['capital_final'],
                        'pnl_total_usd': res['pnl_total_usd']
                    })
            except Exception as e:
                logger.error(f"Erro na simulação com parâmetros {params_teste}: {e}")
                
    if not resultados:
        logger.warning("Nenhuma combinação atendeu aos critérios mínimos (total_grids >= 30 e profit_factor > 1.0).")
        return
        
    # Converter para DataFrame
    df_res = pd.DataFrame(resultados)
    
    # Ordenar por fator_recuperacao decrescente
    df_res = df_res.sort_values(by='fator_recuperacao', ascending=False).reset_index(drop=True)
    
    # Salvar Parquet completo
    caminho_parquet = DIR_OTIM / "otimizacao_grid2_completo.parquet"
    df_res.to_parquet(caminho_parquet, engine="pyarrow", compression="snappy")
    logger.info(f"Parquet completo com {len(df_res)} combinações salvas em: {caminho_parquet}")
    
    # Top 10
    top10_df = df_res.head(10)
    top10_lista = []
    
    for idx, row in top10_df.iterrows():
        top10_lista.append({
            "id": f"GRID2_TOP{idx + 1}",
            "tipo_grid": 2,
            "janela_mm": int(row['janela_mm']),
            "janela_slope": int(row['janela_slope']),
            "threshold_slope": float(row['threshold_slope']),
            "mult_espacamento": float(row['mult_espacamento']),
            "mult_alvo": float(row['mult_alvo']),
            "mult_stop": float(row['mult_stop']),
            "stop_candles": int(row['stop_candles']),
            "total_grids": int(row['total_grids']),
            "win_rate": float(row['win_rate']),
            "profit_factor": float(row['profit_factor']),
            "fator_recuperacao": float(row['fator_recuperacao']),
            "max_drawdown_pct": float(row['max_drawdown_pct']),
            "sharpe": float(row['sharpe'])
        })
        
    # Salvar JSON Top 10
    caminho_json = DIR_OTIM / "otimizacao_grid2_top10.json"
    with open(caminho_json, 'w', encoding='utf-8') as f:
        json.dump(top10_lista, f, indent=4, ensure_ascii=False)
    logger.info(f"JSON Top 10 salvos em: {caminho_json}")
    
    # Exibir Tabela no Terminal
    print("\n" + "═"*125)
    print(" " * 48 + "TABELA TOP 10 OTIMIZAÇÃO GRID 2")
    print("═"*125)
    print(f"{'Rank':<5}{'Jan. MM':<9}{'Jan. Slope':<12}{'Thresh. Sl.':<13}{'Mult Esp.':<11}{'Mult Alvo':<11}{'Mult Stop':<11}{'Stop Cand.':<12}{'N Grids':<9}{'WinRate':<9}{'ProfitF.':<10}{'FatRecup.':<11}")
    print("─"*125)
    for idx, item in enumerate(top10_lista):
        print(f"#{idx+1:<4}{item['janela_mm']:<9}{item['janela_slope']:<12}{item['threshold_slope']:<13.2f}{item['mult_espacamento']:<11.1f}{item['mult_alvo']:<11.1f}{item['mult_stop']:<11.1f}{item['stop_candles']:<12}{item['total_grids']:<9}{item['win_rate']:<9.1f}%{item['profit_factor']:<10.2f}{item['fator_recuperacao']:<11.2f}")
    print("═"*125 + "\n")

if __name__ == "__main__":
    rodar_otimizacao_grid2()
