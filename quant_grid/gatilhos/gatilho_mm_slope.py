# -*- coding: utf-8 -*-
"""
GRID 2 — Gatilho baseado na inclinação da média móvel simples.
"""

import logging
import numpy as np
import pandas as pd
from quant_grid.config import PARQUET_COMPLETO, FATOR_PIPS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def calcular_slope_mm(
    close: pd.Series,
    janela_mm: int = 50,
    janela_slope: int = 5,
) -> pd.Series:
    """
    Calcula a inclinação (slope) normalizada da média móvel simples.

    1. Calcular SMA: mm = close.rolling(janela_mm).mean()
    2. Calcular slope via regressão OLS nos últimos janela_slope candles:
       Para cada t: slope_t = coeficiente angular de OLS(
           x=[0,1,...,janela_slope-1],
           y=mm[t-janela_slope+1:t+1]
       )
    3. Normalizar o slope por pip:
       slope_norm = slope_t / close[t] * FATOR_PIPS
       (slope em pips por candle)

    Retorna pd.Series com slope normalizado.
    """
    # 1. Calcular SMA
    mm = close.rolling(window=janela_mm).mean()
    
    # 2. Calcular slope via OLS vetorizado
    N = janela_slope
    Sx = N * (N - 1) / 2.0
    Sxx = (N - 1) * N * (2 * N - 1) / 6.0
    denominador = N * Sxx - (Sx ** 2)
    
    # Soma de y
    sum_y = mm.rolling(window=N).sum()
    
    # Soma de x * y usando shifts do pandas
    sum_xy = sum(i * mm.shift(N - 1 - i) for i in range(N))
    
    slope_t = (N * sum_xy - Sx * sum_y) / denominador
    
    # 3. Normalizar por pip
    slope_norm = (slope_t / close) * FATOR_PIPS
    return slope_norm

def gerar_sinal_mm_slope(
    df: pd.DataFrame,
    janela_mm: int = 50,
    janela_slope: int = 5,
    threshold_slope: float = 0.5,  # pips por candle
) -> pd.DataFrame:
    """
    Gera sinal de ativação do grid:
        sinal = -1 (grid VENDA) se slope_norm >= +threshold_slope
                   (MM subindo → vender contra)
        sinal = +1 (grid COMPRA) se slope_norm <= -threshold_slope
                   (MM caindo → comprar contra)
        sinal =  0 sem sinal (slope fraco = sem tendência clara)

    Adiciona colunas ao df:
        'mm_{janela_mm}': float     ← média móvel
        'slope_mm': float           ← slope normalizado
        'sinal_grid2': int8

    Retorna df com as novas colunas.
    """
    df = df.copy()
    
    # Nome dinâmico para a coluna de média móvel
    col_mm = f"mm_{janela_mm}"
    
    # Calcula SMA e slope
    df[col_mm] = df['Close'].rolling(window=janela_mm).mean()
    df['slope_mm'] = calcular_slope_mm(df['Close'], janela_mm, janela_slope)
    
    sinal = np.zeros(len(df), dtype=np.int8)
    
    # OLS precisa de dados suficientes
    valid_mask = ~df['slope_mm'].isna() & ~df['Close'].isna()
    
    # VENDA: slope >= threshold
    venda_cond = valid_mask & (df['slope_mm'] >= threshold_slope)
    # COMPRA: slope <= -threshold
    compra_cond = valid_mask & (df['slope_mm'] <= -threshold_slope)
    
    sinal[venda_cond] = -1
    sinal[compra_cond] = 1
    
    df['sinal_grid2'] = sinal
    return df

if __name__ == "__main__":
    import os
    logger.info("Executando gatilho_mm_slope standalone para validação...")
    
    if not PARQUET_COMPLETO.exists():
        logger.error(f"Arquivo parquet não encontrado em: {PARQUET_COMPLETO}")
        print(f"Por favor, verifique se a pasta de dados 'quant_eurusd_h1/data' está correta e contém 'eurusd_h1_completo.parquet'.")
    else:
        try:
            df = pd.read_parquet(PARQUET_COMPLETO)
            logger.info(f"Parquet carregado com sucesso. Linhas: {len(df)}")
            
            df_sinal = gerar_sinal_mm_slope(df, janela_mm=50, janela_slope=5, threshold_slope=0.5)
            compras = (df_sinal['sinal_grid2'] == 1).sum()
            vendas = (df_sinal['sinal_grid2'] == -1).sum()
            
            logger.info(f"Sinais de COMPRA gerados: {compras}")
            logger.info(f"Sinais de VENDA gerados: {vendas}")
            logger.info("Validação concluída com sucesso.")
        except Exception as e:
            logger.exception(f"Erro ao processar gatilho: {e}")
