# -*- coding: utf-8 -*-
"""
GRID 1 — Gatilho baseado em % de afastamento do fechamento diário.
"""

import logging
import numpy as np
import pandas as pd
from quant_grid.config import PARQUET_COMPLETO

# Configuração de logging local
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def calcular_daily_close(df: pd.DataFrame) -> pd.Series:
    """
    Para cada candle H1, identifica o fechamento do dia anterior
    no horário do servidor MT5.

    Dia no servidor MT5: considera mudança de dia à meia-noite.
    O 'fechamento do dia anterior' é o Close do último candle
    do dia anterior.

    Retorna pd.Series com o daily_close alinhado ao índice do df.
    NaN para os candles do primeiro dia (sem referência anterior).
    """
    # Agrupa por data e pega o último valor de Close de cada dia
    fechamento_diario = df['Close'].groupby(df.index.date).last()
    
    # Desloca 1 dia para que a data D tenha o fechamento de D-1
    fechamento_diario_anterior = fechamento_diario.shift(1)
    
    # Mapeia de volta para o dataframe original usando a data de cada candle
    daily_close_series = pd.Series(df.index.date, index=df.index).map(fechamento_diario_anterior)
    return daily_close_series

def gerar_sinal_daily_close(
    df: pd.DataFrame,
    pct_gatilho: float = 0.003,   # 0.3% de afastamento
) -> pd.DataFrame:
    """
    Calcula o afastamento percentual do Close atual em relação
    ao fechamento do dia anterior:
        afastamento = (Close - daily_close) / daily_close

    Gera sinal de ativação do grid:
        sinal = +1 (grid COMPRA) se afastamento <= -pct_gatilho
                   (preço caiu X% → comprar contra)
        sinal = -1 (grid VENDA)  se afastamento >= +pct_gatilho
                   (preço subiu X% → vender contra)
        sinal =  0 sem sinal

    Adiciona colunas ao df:
        'daily_close': float
        'afastamento_pct': float
        'sinal_grid1': int8

    Retorna df com as novas colunas.
    IMPORTANTE: sinal só pode ser gerado uma vez por direção
    por dia — se o grid já estiver ativo na mesma direção,
    não gerar novo sinal.
    """
    # Cria uma cópia para evitar warnings de SettingWithCopy
    df = df.copy()
    
    # 1. Calcula daily close se ainda não existir
    if 'daily_close' not in df.columns:
        df['daily_close'] = calcular_daily_close(df)
        
    # 2. Calcula o afastamento percentual
    df['afastamento_pct'] = (df['Close'] - df['daily_close']) / df['daily_close']
    
    # 3. Identifica as condições brutas de sinal
    sinal_compra_bruto = (df['afastamento_pct'] <= -pct_gatilho)
    sinal_venda_bruto = (df['afastamento_pct'] >= pct_gatilho)
    
    # 4. Garante apenas o primeiro sinal de cada direção por dia
    first_compra = (sinal_compra_bruto & (sinal_compra_bruto.groupby(df.index.date).cumsum() == 1))
    first_venda = (sinal_venda_bruto & (sinal_venda_bruto.groupby(df.index.date).cumsum() == 1))
    
    sinal = np.zeros(len(df), dtype=np.int8)
    sinal[first_compra] = 1
    sinal[first_venda] = -1
    
    df['sinal_grid1'] = sinal
    return df

if __name__ == "__main__":
    import os
    logger.info("Executando gatilho_daily_close standalone para validação...")
    
    if not PARQUET_COMPLETO.exists():
        logger.error(f"Arquivo parquet não encontrado em: {PARQUET_COMPLETO}")
        print(f"Por favor, verifique se a pasta de dados 'quant_eurusd_h1/data' está correta e contém 'eurusd_h1_completo.parquet'.")
    else:
        try:
            df = pd.read_parquet(PARQUET_COMPLETO)
            logger.info(f"Parquet carregado com sucesso. Linhas: {len(df)}")
            
            df_sinal = gerar_sinal_daily_close(df, pct_gatilho=0.003)
            compras = (df_sinal['sinal_grid1'] == 1).sum()
            vendas = (df_sinal['sinal_grid1'] == -1).sum()
            
            logger.info(f"Sinais de COMPRA gerados: {compras}")
            logger.info(f"Sinais de VENDA gerados: {vendas}")
            logger.info("Validação concluída com sucesso.")
        except Exception as e:
            logger.exception(f"Erro ao processar gatilho: {e}")
