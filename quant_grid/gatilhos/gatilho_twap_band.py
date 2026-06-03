# -*- coding: utf-8 -*-
"""
GRID 4 — Gatilho baseado no rompimento das bandas da TWAP.
"""

import logging
import numpy as np
import pandas as pd
from quant_grid.config import PARQUET_COMPLETO, FATOR_PIPS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def calcular_twap(
    df: pd.DataFrame,
    janela: int = 24,
) -> pd.Series:
    """
    Calcula a TWAP (Time Weighted Average Price) rolling.
    TWAP = mean(típico_price, janela)
    típico_price = (High + Low + Close) / 3

    Retorna pd.Series com TWAP alinhado ao índice do df.
    """
    tipico_price = (df['High'] + df['Low'] + df['Close']) / 3.0
    twap = tipico_price.rolling(window=janela).mean()
    return twap

def calcular_bandas_twap(
    df: pd.DataFrame,
    janela: int = 24,
    mult_banda: float = 2.0,
) -> pd.DataFrame:
    """
    Calcula as bandas da TWAP baseadas na volatilidade realizada.

    twap = calcular_twap(df, janela)
    vr   = log_return.rolling(janela).std(ddof=1) * Close * FATOR_PIPS

    banda_superior = twap + (mult_banda * vr / FATOR_PIPS)
    banda_inferior = twap - (mult_banda * vr / FATOR_PIPS)

    Adiciona colunas ao df:
        'twap': float
        'twap_banda_sup': float
        'twap_banda_inf': float
        'twap_largura': float    ← banda_sup - banda_inf em pips

    Retorna df com as novas colunas.
    """
    df = df.copy()
    
    twap = calcular_twap(df, janela)
    
    # log_return deve existir no df
    if 'log_return' not in df.columns:
        # Se não existir por algum motivo, calcula
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        
    vr = df['log_return'].rolling(window=janela).std(ddof=1) * df['Close'] * FATOR_PIPS
    
    banda_sup = twap + (mult_banda * vr / FATOR_PIPS)
    banda_inf = twap - (mult_banda * vr / FATOR_PIPS)
    largura = (banda_sup - banda_inf) * FATOR_PIPS
    
    df['twap'] = twap
    df['twap_banda_sup'] = banda_sup
    df['twap_banda_inf'] = banda_inf
    df['twap_largura'] = largura
    
    return df

def gerar_sinal_twap_band(
    df: pd.DataFrame,
    janela: int = 24,
    mult_banda: float = 2.0,
) -> pd.DataFrame:
    """
    Rompimento das bandas CONTRA a tendência:

    VENDA (grid contra alta):
        Close >= twap_banda_sup
        (preço rompeu para cima → caro → vender contra)

    COMPRA (grid contra queda):
        Close <= twap_banda_inf
        (preço rompeu para baixo → barato → comprar contra)

    Adiciona coluna 'sinal_grid4': int8

    Retorna df com a nova coluna.
    IMPORTANTE: sinal de rompimento válido apenas se
    twap_largura > 5.0 pips (filtro de liquidez mínima).
    """
    df = calcular_bandas_twap(df, janela, mult_banda)
    
    sinal = np.zeros(len(df), dtype=np.int8)
    
    # Máscaras de validade
    valid_mask = ~df['twap_banda_sup'].isna() & ~df['twap_banda_inf'].isna() & ~df['twap_largura'].isna()
    
    # Filtro de liquidez
    liquidez_filtro = df['twap_largura'] > 5.0
    
    # Condições
    venda_cond = valid_mask & liquidez_filtro & (df['Close'] >= df['twap_banda_sup'])
    compra_cond = valid_mask & liquidez_filtro & (df['Close'] <= df['twap_banda_inf'])
    
    sinal[venda_cond] = -1
    sinal[compra_cond] = 1
    
    df['sinal_grid4'] = sinal
    return df

if __name__ == "__main__":
    import os
    logger.info("Executando gatilho_twap_band standalone para validação...")
    
    if not PARQUET_COMPLETO.exists():
        logger.error(f"Arquivo parquet não encontrado em: {PARQUET_COMPLETO}")
        print(f"Por favor, verifique se a pasta de dados 'quant_eurusd_h1/data' está correta e contém 'eurusd_h1_completo.parquet'.")
    else:
        try:
            df = pd.read_parquet(PARQUET_COMPLETO)
            logger.info(f"Parquet carregado com sucesso. Linhas: {len(df)}")
            
            df_sinal = gerar_sinal_twap_band(df)
            compras = (df_sinal['sinal_grid4'] == 1).sum()
            vendas = (df_sinal['sinal_grid4'] == -1).sum()
            
            logger.info(f"Sinais de COMPRA gerados: {compras}")
            logger.info(f"Sinais de VENDA gerados: {vendas}")
            logger.info("Validação concluída com sucesso.")
        except Exception as e:
            logger.exception(f"Erro ao processar gatilho: {e}")
