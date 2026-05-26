# -*- coding: utf-8 -*-
"""
================================================================================
data_loader.py — Módulo de Carregamento e Pré-Processamento de Dados EURUSD
================================================================================
Objetivo:
    Carregar o arquivo CSV bruto exportado do MetaTrader 5 em timeframe M1,
    realizar filtro de horário operacional (07h às 23h UTC), agregar para H1,
    calcular log-retornos e salvar em formato Parquet para consumo pelos
    demais módulos do projeto.
================================================================================
"""

import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

# Suprimir warnings desnecessários
warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURAÇÃO DE LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# CAMINHOS DO PROJETO
# =============================================================================
DIR_PROJETO = Path(__file__).resolve().parent
DIR_DATA    = DIR_PROJETO / "data"

# Caminho para o CSV de entrada
# Procuramos no diretório pai ou aceitamos via variável de ambiente
CSV_PADRAO = DIR_PROJETO.parent / "EURUSD_10ANOS.csv"
CSV_PATH = Path(os.environ.get("EURUSD_CSV_PATH", str(CSV_PADRAO)))

# Arquivo de saída
PARQUET_SAIDA = DIR_DATA / "eurusd_h1_clean.parquet"

# =============================================================================
# ESTRUTURA DO CSV
# =============================================================================
# Formato exato do CSV do MetaTrader 5 (sem cabeçalho, separado por vírgula):
COLUNAS_CSV = [
    "Date",        # Col 0: YYYY.MM.DD
    "Time",        # Col 1: HH:MM
    "Open",        # Col 2: float
    "High",        # Col 3: float
    "Low",         # Col 4: float
    "Close",       # Col 5: float
    "Volume",      # Col 6: int
    "Spread",      # Col 7: int
    "RealVolume",  # Col 8: int
]

# Dtypes otimizados para 8.5M de linhas (economia de memória)
DTYPES_CSV = {
    "Date":       "str",
    "Time":       "str",
    "Open":       "float32",
    "High":       "float32",
    "Low":        "float32",
    "Close":      "float32",
    "Volume":     "int32",
    "Spread":     "int16",
    "RealVolume": "int32",
}

# =============================================================================
# FUNÇÕES DE PROCESSAMENTO
# =============================================================================

def carregar_e_processar(
    csv_path: Path = CSV_PATH,
    data_inicio: str = None,
    data_fim: str = None,
    forcar_reprocessamento: bool = False,
) -> pd.DataFrame:
    """
    Executa o pipeline completo: carrega CSV, filtra horários, agrega H1,
    calcula retornos e salva em Parquet.
    """
    # Cache
    if PARQUET_SAIDA.exists() and not forcar_reprocessamento:
        logger.info(f"Parquet já existe: {PARQUET_SAIDA.name}. Carregando cache...")
        df_h1 = pd.read_parquet(PARQUET_SAIDA, engine="pyarrow")
        if data_inicio:
            df_h1 = df_h1[df_h1.index >= data_inicio]
        if data_fim:
            df_h1 = df_h1[df_h1.index <= data_fim]
        return df_h1
    # Garantir que o diretório data existe
    DIR_DATA.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        # Vamos tentar outra variação de nome de arquivo caso o padrão falhe
        candidato = DIR_PROJETO.parent / "EURUSD.csv"
        if candidato.exists():
            csv_path = candidato
        else:
            raise FileNotFoundError(
                f"\n[ERRO] Arquivo CSV não encontrado: {csv_path}\n"
                f"Defina a variável de ambiente EURUSD_CSV_PATH ou coloque o arquivo no diretório raiz."
            )

    logger.info(f"Iniciando carregamento do CSV: {csv_path.name}")
    
    # 1.1 Carregamento do CSV
    df = pd.read_csv(
        csv_path,
        header=None,
        names=COLUNAS_CSV,
        dtype=DTYPES_CSV,
        engine="c",
        low_memory=False,
    )
    logger.info(f"CSV carregado: {len(df):,} registros M1")

    # Combinar Date e Time no formato 'YYYY.MM.DD HH:MM' -> 'YYYY-MM-DD HH:MM'
    logger.info("Processando índice de data e hora...")
    datas_str = df["Date"].str.replace(".", "-", regex=False) + " " + df["Time"]
    df.index = pd.to_datetime(datas_str, format="%Y-%m-%d %H:%M")
    df.index.name = "Datetime"
    
    # Remover colunas desnecessárias
    df.drop(columns=["Date", "Time"], inplace=True)

    # 1.2 Filtro de Horário Operacional
    # Manter apenas candles cujo horário UTC esteja entre 07:00 e 23:00
    # Equivalent to 04h00 - 20h00 BRT
    total_antes = len(df)
    df = df[df.index.hour.isin(range(7, 23))]
    
    # Também vamos remover fins de semana (sábado e domingo) pois não há operação no FOREX
    # (A instrução focou na hora, mas para evitar distorções no mercado forex, é bom manter apenas Seg-Sex)
    df = df[df.index.weekday < 5]
    
    total_depois = len(df)
    logger.info(f"Filtro de horário (07h às 23h, Seg-Sex): {total_depois:,} candles mantidos ({(total_antes - total_depois):,} removidos).")

    # 1.3 Agregação M1 -> H1
    logger.info("Iniciando agregação M1 -> H1...")
    
    # Contar os candles M1 por hora antes de agregar
    contagem_m1 = df.resample("1h").size()
    
    df_h1 = df.resample("1h").agg({
        "Open":   "first",
        "High":   "max",
        "Low":    "min",
        "Close":  "last",
        "Volume": "sum",
    })
    
    df_h1["candles_m1"] = contagem_m1
    
    # Remover candles H1 com menos de 30 candles M1
    h1_antes = len(df_h1)
    df_h1 = df_h1[df_h1["candles_m1"] >= 30]
    df_h1.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)
    h1_depois = len(df_h1)
    logger.info(f"Agregação H1 concluída: {h1_depois:,} candles H1 válidos ({(h1_antes - h1_depois):,} incompletos/vazios removidos).")
    
    # Garantir que os preços continuam como float32
    for col in ["Open", "High", "Low", "Close"]:
        df_h1[col] = df_h1[col].astype("float32")

    # 1.4 Cálculo de Log-Retornos
    logger.info("Calculando log-retornos...")
    df_h1["log_return"] = np.log(df_h1["Close"] / df_h1["Close"].shift(1)).astype("float32")
    df_h1.dropna(subset=["log_return"], inplace=True)

    # Filtros de data
    if data_inicio:
        df_h1 = df_h1[df_h1.index >= data_inicio]
    if data_fim:
        df_h1 = df_h1[df_h1.index <= data_fim]

    # 1.5 Output
    logger.info("Salvando arquivo Parquet...")
    df_h1.to_parquet(
        PARQUET_SAIDA,
        engine="pyarrow",
        compression="snappy",
        index=True
    )
    
    # Resumo
    print("\n" + "=" * 60)
    print("  RESUMO DO DATA LOADER — EURUSD H1")
    print("=" * 60)
    print(f"  Arquivo salvo       : {PARQUET_SAIDA.name}")
    print(f"  Período             : {df_h1.index.min()} a {df_h1.index.max()}")
    print(f"  Total de candles H1 : {len(df_h1):,}")
    print(f"  Média do retorno    : {df_h1['log_return'].mean():.6f}")
    print(f"  Desvio padrão ret.  : {df_h1['log_return'].std():.6f}")
    print("=" * 60 + "\n")

    return df_h1

# =============================================================================
# EXECUÇÃO DIRETA
# =============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Data Loader EURUSD H1")
    parser.add_argument("--csv", type=str, default=str(CSV_PATH), help="Caminho do CSV de entrada")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  INICIANDO PREPARAÇÃO DE DADOS (DATA LOADER)")
    print("=" * 60)
    
    carregar_e_processar(Path(args.csv))
