# -*- coding: utf-8 -*-
"""
Otimização para o GRID 1 — Gatilho baseado em % de afastamento do fechamento diário.
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
from quant_grid.gatilhos.gatilho_daily_close import gerar_sinal_daily_close
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
    'pct_gatilho': [0.001, 0.002, 0.003, 0.005, 0.008],
}

def rodar_otimizacao_grid1():
    """
    Executa a otimização de parâmetros por Grid Search para o Grid 1.
    """
    if not PARQUET_COMPLETO.exists():
        raise FileNotFoundError(
            f"Arquivo de dados para otimização não encontrado: {PARQUET_COMPLETO}\n"
            f"Verifique se o arquivo eurusd_h1_completo.parquet está no diretório correto."
        )
        
    logger.info("Carregando base de dados para otimização do Grid 1...")
    df = pd.read_parquet(PARQUET_COMPLETO)
    
    DIR_OTIM.mkdir(parents=True, exist_ok=True)
    
    resultados = []
    
    # Gerar produtos cartesianos
    chaves_esp, vals_esp = zip(*PARAMS_ESPECIFICOS.items())
    combinacoes_especificas = [dict(zip(chaves_esp, v)) for v in itertools.product(*vals_esp)]
    
    chaves_com, vals_com = zip(*PARAMS_COMUNS.items())
    combinacoes_comuns = [dict(zip(chaves_com, v)) for v in itertools.product(*vals_com)]
    
    total_combinacoes = len(combinacoes_especificas) * len(combinacoes_comuns)
    logger.info(f"Iniciando Grid Search do Grid 1: {total_combinacoes} combinações totais.")
    
    # Loop das combinações específicas (para pré-calcular o sinal)
    for i_esp, p_esp in enumerate(combinacoes_especificas):
        logger.info(f"Pré-calculando sinal para parâmetros específicos ({i_esp + 1}/{len(combinacoes_especificas)}): {p_esp}")
        
        # Gera o sinal correspondente a essa configuração
        df_sinalizado = gerar_sinal_daily_close(df, pct_gatilho=p_esp['pct_gatilho'])
        
        # Loop das combinações comuns
        for p_com in combinacoes_comuns:
            # Combina os dicionários de parâmetros
            params_teste = {**p_esp, **p_com}
            
            try:
                # Rodar Backtest
                res = rodar_backtest_grid(df_sinalizado, tipo_grid=1, params=params_teste, capital_inicial=CAPITAL_INICIAL, verbose=False)
                
                # Filtro mínimo exigido: total_grids >= 30 e profit_factor > 1.0
                if res['total_grids'] >= 30 and res['profit_factor'] > 1.0:
                    resultados.append({
                        'pct_gatilho': params_teste['pct_gatilho'],
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
    caminho_parquet = DIR_OTIM / "otimizacao_grid1_completo.parquet"
    df_res.to_parquet(caminho_parquet, engine="pyarrow", compression="snappy")
    logger.info(f"Parquet completo com {len(df_res)} combinações salvas em: {caminho_parquet}")
    
    # Top 10
    top10_df = df_res.head(10)
    top10_lista = []
    
    for idx, row in top10_df.iterrows():
        top10_lista.append({
            "id": f"GRID1_TOP{idx + 1}",
            "tipo_grid": 1,
            "pct_gatilho": float(row['pct_gatilho']),
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
    caminho_json = DIR_OTIM / "otimizacao_grid1_top10.json"
    with open(caminho_json, 'w', encoding='utf-8') as f:
        json.dump(top10_lista, f, indent=4, ensure_ascii=False)
    logger.info(f"JSON Top 10 salvo em: {caminho_json}")
    
    # Exibir Tabela no Terminal
    print("\n" + "═"*115)
    print(" " * 45 + "TABELA TOP 10 OTIMIZAÇÃO GRID 1")
    print("═"*115)
    print(f"{'Rank':<5}{'Gatilho':<10}{'Mult Esp.':<11}{'Mult Alvo':<11}{'Mult Stop':<11}{'Stop Cand.':<12}{'N Grids':<9}{'WinRate':<9}{'ProfitF.':<10}{'FatRecup.':<11}{'Sharpe':<8}")
    print("─"*115)
    for idx, item in enumerate(top10_lista):
        print(f"#{idx+1:<4}{item['pct_gatilho']:<10.3f}{item['mult_espacamento']:<11.1f}{item['mult_alvo']:<11.1f}{item['mult_stop']:<11.1f}{item['stop_candles']:<12}{item['total_grids']:<9}{item['win_rate']:<9.1f}%{item['profit_factor']:<10.2f}{item['fator_recuperacao']:<11.2f}{item['sharpe']:<8.2f}")
    print("═"*115 + "\n")

if __name__ == "__main__":
    rodar_otimizacao_grid1()
