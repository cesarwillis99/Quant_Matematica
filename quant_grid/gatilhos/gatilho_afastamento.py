# -*- coding: utf-8 -*-
"""
GRID 3 — Gatilho baseado em afastamento estatístico por volatilidade.
"""

import logging
import numpy as np
import pandas as pd
from quant_grid.config import PARQUET_COMPLETO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def calcular_afastamento_estatistico(
    df: pd.DataFrame,
    janela_zscore: int = 100,
    janela_percentil: int = 200,
) -> pd.DataFrame:
    """
    Calcula os dois indicadores de afastamento:

    1. Z-Score do preço em relação à média rolling:
       mean_N = Close.rolling(janela_zscore).mean()
       std_N  = Close.rolling(janela_zscore).std(ddof=1)
       zscore_preco = (Close - mean_N) / std_N

    2. Percentil histórico do retorno acumulado:
       retorno_N = log(Close / Close.shift(janela_zscore))
       Para cada t: pct_hist[t] = percentileofscore(
           retorno_N[t-janela_percentil:t], retorno_N[t]
       ) / 100.0   ← normalizado em [0,1]

    Adiciona colunas ao df:
        'zscore_afastamento': float
        'pct_hist_retorno': float

    Retorna df com as novas colunas.
    """
    df = df.copy()
    
    # 1. Z-Score do Preço
    mean_N = df['Close'].rolling(window=janela_zscore).mean()
    std_N = df['Close'].rolling(window=janela_zscore).std(ddof=1)
    df['zscore_afastamento'] = (df['Close'] - mean_N) / std_N
    
    # 2. Retorno Acumulado
    retorno_N = np.log(df['Close'] / df['Close'].shift(janela_zscore))
    
    # 3. Percentil Histórico via Rolling Rank no Pandas (100% vetorizado)
    # A janela tem tamanho janela_percentil + 1 para incluir o valor atual t
    df['pct_hist_retorno'] = retorno_N.rolling(window=janela_percentil + 1).rank(pct=True)
    
    return df

def gerar_sinal_afastamento(
    df: pd.DataFrame,
    janela_zscore: int = 100,
    janela_percentil: int = 200,
    threshold_zscore: float = 2.0,
    threshold_pct_alto: float = 0.85,  # acima de 85% → vender
    threshold_pct_baixo: float = 0.15, # abaixo de 15% → comprar
) -> pd.DataFrame:
    """
    Sinal combinado dos dois indicadores (ambos devem concordar):

    VENDA (grid contra alta):
        zscore_afastamento >= +threshold_zscore
        E pct_hist_retorno >= threshold_pct_alto

    COMPRA (grid contra queda):
        zscore_afastamento <= -threshold_zscore
        E pct_hist_retorno <= threshold_pct_baixo

    Adiciona coluna 'sinal_grid3': int8

    Retorna df com a nova coluna.
    """
    df = calcular_afastamento_estatistico(df, janela_zscore, janela_percentil)
    
    sinal = np.zeros(len(df), dtype=np.int8)
    
    # Máscaras de validade
    valid_mask = ~df['zscore_afastamento'].isna() & ~df['pct_hist_retorno'].isna()
    
    # Condições
    venda_cond = valid_mask & (df['zscore_afastamento'] >= threshold_zscore) & (df['pct_hist_retorno'] >= threshold_pct_alto)
    compra_cond = valid_mask & (df['zscore_afastamento'] <= -threshold_zscore) & (df['pct_hist_retorno'] <= threshold_pct_baixo)
    
    sinal[venda_cond] = -1
    sinal[compra_cond] = 1
    
    df['sinal_grid3'] = sinal
    return df

if __name__ == "__main__":
    import os
    logger.info("Executando gatilho_afastamento standalone para validação...")
    
    if not PARQUET_COMPLETO.exists():
        logger.error(f"Arquivo parquet não encontrado em: {PARQUET_COMPLETO}")
        print(f"Por favor, verifique se a pasta de dados 'quant_eurusd_h1/data' está correta e contém 'eurusd_h1_completo.parquet'.")
    else:
        try:
            df = pd.read_parquet(PARQUET_COMPLETO)
            logger.info(f"Parquet carregado com sucesso. Linhas: {len(df)}")
            
            df_sinal = gerar_sinal_afastamento(df)
            compras = (df_sinal['sinal_grid3'] == 1).sum()
            vendas = (df_sinal['sinal_grid3'] == -1).sum()
            
            logger.info(f"Sinais de COMPRA gerados: {compras}")
            logger.info(f"Sinais de VENDA gerados: {vendas}")
            logger.info("Validação concluída com sucesso.")
        except Exception as e:
            logger.exception(f"Erro ao processar gatilho: {e}")
