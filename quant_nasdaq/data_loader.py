# -*- coding: utf-8 -*-
"""
================================================================================
data_loader.py — Módulo de Carregamento e Pré-Processamento de Dados NASDAQ
                  (NAS100 / US100) — DAY TRADE M10
================================================================================

Objetivo:
    Carregar o arquivo CSV bruto exportado do MetaTrader 5 (M1 ou M10),
    detectar timeframe, agregar para M10 (se necessário), aplicar filtros
    intraday para a sessão americana, excluir feriados, e salvar em Parquet.

Regras de Negócio (Horário do Servidor MT5):
    - Sessão Completa (Indicadores/Stops): 16h30 às 23h00
    - Sessão Operacional (Entradas): 16h30 às 22h30
    - Dias: Segunda a Sexta (excluindo feriados americanos)
    - Log-retornos: Não cruzam dias (resetam na abertura)

Saídas:
    1. nasdaq_m10_completo.parquet     — Série intraday completa (16h30-23h00)
    2. nasdaq_m10_operacional.parquet  — Série operacional (16h30-22h30)

================================================================================
"""

import os
import sys
import argparse
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from pandas.tseries.holiday import (
    AbstractHolidayCalendar, Holiday, nearest_workday,
    USMartinLutherKingJr, USPresidentsDay, USMemorialDay,
    USLaborDay, USThanksgivingDay
)

# Suprimir warnings desnecessários
warnings.filterwarnings("ignore")

# Corrigir encoding para Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

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

# Caminho padrão para o CSV de entrada (diretório pai)
CSV_PADRAO = DIR_PROJETO.parent / "US100_10ANOS.csv"

# Arquivos de saída
PARQUET_COMPLETO     = DIR_DATA / "nasdaq_m10_completo.parquet"
PARQUET_OPERACIONAL  = DIR_DATA / "nasdaq_m10_operacional.parquet"

# =============================================================================
# ESTRUTURA DO CSV
# =============================================================================
COLUNAS_CSV = [
    "Date",        # Col 0: YYYY.MM.DD
    "Time",        # Col 1: HH:MM
    "Open",        # Col 2: float
    "High",        # Col 3: float
    "Low",         # Col 4: float
    "Close",       # Col 5: float
    "Volume",      # Col 6: int (tick volume)
    "Spread",      # Col 7: int
    "RealVolume",  # Col 8: int
]

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
# CALENDÁRIO DE FERIADOS AMERICANOS
# =============================================================================
class USMarketHolidayCalendar(AbstractHolidayCalendar):
    """Calendário de Feriados Americanos para o NASDAQ."""
    rules = [
        Holiday('Ano Novo', month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        USMemorialDay,
        Holiday('Independência EUA', month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday('Natal', month=12, day=25, observance=nearest_workday)
    ]

# =============================================================================
# FUNÇÕES DO PIPELINE
# =============================================================================

def carregar_csv_bruto(csv_path: Path) -> pd.DataFrame:
    """Passo 1 e 2 — Carrega o CSV bruto do MT5 com barra de progresso."""
    if not csv_path.exists():
        raise FileNotFoundError(f"\n[ERRO] Arquivo CSV não encontrado: {csv_path}\n")

    logger.info(f"Contando linhas do CSV: {csv_path.name}")
    with open(csv_path, 'rb') as f:
        total_lines = sum(1 for _ in f)

    logger.info("Carregando CSV...")
    chunksize = 500000
    chunks = []
    
    with tqdm(total=total_lines, desc=f"Lendo {csv_path.name}") as pbar:
        for chunk in pd.read_csv(
            csv_path, header=None, names=COLUNAS_CSV,
            dtype=DTYPES_CSV, engine="c", low_memory=False, chunksize=chunksize
        ):
            chunks.append(chunk)
            pbar.update(len(chunk))
            
    df = pd.concat(chunks, ignore_index=True)
    total_bruto = len(df)
    
    # Passo 2 — Parser datetime naive
    logger.info("Processando índice datetime naive...")
    datas_str = df["Date"].str.replace(".", "-", regex=False) + " " + df["Time"]
    df.index = pd.to_datetime(datas_str, format="%Y-%m-%d %H:%M")
    df.index.name = "Datetime"
    df.drop(columns=["Date", "Time"], inplace=True)
    
    # Passo 3 — Remover duplicatas de índice
    duplicatas = df.index.duplicated(keep="first")
    if duplicatas.sum() > 0:
        logger.info(f"Removendo {duplicatas.sum():,} duplicatas de índice")
        df = df[~duplicatas]
        
    return df, total_bruto


def processar_timeframe_e_horario(df: pd.DataFrame) -> tuple:
    """Passo 4, 5 e 6 — Detectar timeframe, agregar, filtrar horários e feriados."""
    stats = {}
    
    # Passo 4 — Detectar timeframe e agregar
    if len(df) > 1000:
        diffs = df.index[1:1000].to_series().diff().median()
    else:
        diffs = df.index.to_series().diff().median()
        
    if diffs == pd.Timedelta(minutes=1):
        stats['timeframe'] = 'M1'
        logger.info("Timeframe M1 detectado. Agregando para M10...")
        contagem_m1 = df.resample("10min", closed="left", label="left").size()
        df = df.resample("10min", closed="left", label="left").agg({
            "Open":   "first",
            "High":   "max",
            "Low":    "min",
            "Close":  "last",
            "Volume": "sum",
        })
        df["candles_m1"] = contagem_m1
        
        # Remover incompletos e vazios
        df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)
        antes = len(df)
        df = df[df["candles_m1"] >= 7]
        stats['removidos_incompletos'] = antes - len(df)
        df.drop(columns=["candles_m1"], inplace=True)
        
        for col in ["Open", "High", "Low", "Close"]:
            df[col] = df[col].astype("float32")
        df["Volume"] = df["Volume"].astype("int32")
    else:
        stats['timeframe'] = 'M10 (ou maior)'
        stats['removidos_incompletos'] = 0
        logger.info(f"Timeframe {stats['timeframe']} detectado. Usando diretamente.")

    # Passo 5 — Filtrar dias da semana (Segunda a Sexta = 0 a 4)
    # A série completa deve manter 24h para o cálculo contínuo dos indicadores
    weekday = df.index.weekday
    mascara_dias = (weekday >= 0) & (weekday <= 4)
    
    antes = len(df)
    df = df[mascara_dias]
    stats['removidos_fora_horario'] = antes - len(df)  # Basicamente removeu fins de semana
    
    # Passo 6 — Remover feriados americanos
    cal = USMarketHolidayCalendar()
    feriados = cal.holidays(start=df.index.min(), end=df.index.max())
    stats['feriados_removidos_lista'] = [f.strftime('%Y-%m-%d') for f in feriados if f.date() in df.index.date]
    
    antes = len(df)
    df = df[~df.index.normalize().isin(feriados)]
    stats['removidos_feriados'] = antes - len(df)
    
    if stats['removidos_feriados'] > 0:
        logger.info(f"Removidos {stats['removidos_feriados']} candles por conta de feriados americanos.")
        
    return df, stats


def limpar_dados_e_calcular_retornos(df: pd.DataFrame) -> tuple:
    """Passo 7, 8 e 9 — Limpeza OHLC, checagem e log-retornos intraday."""
    stats = {}
    
    # Passo 7 — Remover OHLC zerado ou nulo
    antes = len(df)
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df[(df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    stats['removidos_zerados'] = antes - len(df)
    
    # Passo 8 — Verificar integridade OHLC
    antes = len(df)
    high_ok = df["High"] >= df[["Open", "Close"]].max(axis=1)
    low_ok  = df["Low"]  <= df[["Open", "Close"]].min(axis=1)
    df = df[high_ok & low_ok]
    stats['removidos_integridade'] = antes - len(df)
    
    # Passo 9 — Calcular log-retornos intraday
    logger.info("Calculando log-retornos intraday (reset diário)...")
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1)).astype("float32")
    
    # Resetar (NaN) no primeiro candle de cada dia para não cruzar dias
    is_first_candle = df.index.to_series().dt.date != df.index.to_series().shift(1).dt.date
    df.loc[is_first_candle, "log_return"] = np.nan
    
    return df, stats


def executar_verificacoes(df_completo: pd.DataFrame, df_operacional: pd.DataFrame):
    """Executa os 10 checks de validação e imprime resultado."""
    print("\n" + "═" * 70)
    print("  VERIFICAÇÕES AUTOMÁTICAS (10 CHECKS) - NASDAQ")
    print("═" * 70)
    
    todos_ok = True
    
    # CHECK 1: Nenhum candle de sábado ou domingo
    fds = df_completo[df_completo.index.weekday >= 5]
    ok = len(fds) == 0
    print(f"  CHECK 1  │ Nenhum candle de sábado ou domingo")
    print(f"           │ {'✅ PASS' if ok else '❌ FAIL'} — {len(fds)} candles encontrados")
    if not ok: todos_ok = False
        
    # CHECK 2: Nenhum candle antes das 16h30 na série OPERACIONAL
    antes_1630 = df_operacional[df_operacional.index.strftime('%H:%M') < "16:30"]
    ok = len(antes_1630) == 0
    print(f"\n  CHECK 2  │ Nenhum candle antes das 16h30 (série operacional)")
    print(f"           │ {'✅ PASS' if ok else '❌ FAIL'} — {len(antes_1630)} candles encontrados")
    if not ok: todos_ok = False
        
    # CHECK 3: Nenhum candle após as 22h30 na série OPERACIONAL
    apos_2230 = df_operacional[df_operacional.index.strftime('%H:%M') > "22:30"]
    ok = len(apos_2230) == 0
    print(f"\n  CHECK 3  │ Nenhum candle após as 22h30 na série operacional")
    print(f"           │ {'✅ PASS' if ok else '❌ FAIL'} — {len(apos_2230)} candles encontrados")
    if not ok: todos_ok = False
        
    # CHECK 4: Nenhum feriado americano na base
    cal = USMarketHolidayCalendar()
    feriados = cal.holidays(start=df_completo.index.min(), end=df_completo.index.max())
    holidays_in_base = df_completo[df_completo.index.normalize().isin(feriados)]
    ok = len(holidays_in_base) == 0
    print(f"\n  CHECK 4  │ Nenhum feriado americano na base")
    print(f"           │ {'✅ PASS' if ok else '❌ FAIL'} — {len(holidays_in_base)} candles em feriados")
    if not ok: todos_ok = False
        
    # CHECK 5: Série operacional ⊂ série completa
    diff = set(df_operacional.index) - set(df_completo.index)
    ok = len(diff) == 0
    print(f"\n  CHECK 5  │ Série operacional ⊂ série completa")
    print(f"           │ {'✅ PASS' if ok else '❌ FAIL'} — {len(diff)} candles operacionais ausentes")
    if not ok: todos_ok = False
        
    # CHECK 6: Log-retornos sem NaN ou Inf (exceto primeiros candles de cada dia)
    is_first_candle = df_completo.index.to_series().dt.date != df_completo.index.to_series().shift(1).dt.date
    nan_count = df_completo[~is_first_candle]["log_return"].isna().sum()
    inf_count = np.isinf(df_completo[~is_first_candle]["log_return"]).sum()
    ok = nan_count == 0 and inf_count == 0
    print(f"\n  CHECK 6  │ Log-retornos sem NaN/Inf (exceto abertura de dia)")
    print(f"           │ {'✅ PASS' if ok else '❌ FAIL'} — NaN: {nan_count}, Inf: {inf_count}")
    if not ok: todos_ok = False
        
    # CHECK 7: High >= Open e High >= Close
    high_ok = (df_completo["High"] >= df_completo["Open"]).all() and (df_completo["High"] >= df_completo["Close"]).all()
    print(f"\n  CHECK 7  │ High >= Open e High >= Close")
    print(f"           │ {'✅ PASS' if high_ok else '❌ FAIL'}")
    if not high_ok: todos_ok = False
        
    # CHECK 8: Low <= Open e Low <= Close
    low_ok = (df_completo["Low"] <= df_completo["Open"]).all() and (df_completo["Low"] <= df_completo["Close"]).all()
    print(f"\n  CHECK 8  │ Low <= Open e Low <= Close")
    print(f"           │ {'✅ PASS' if low_ok else '❌ FAIL'}")
    if not low_ok: todos_ok = False
        
    # CHECK 9: Nenhum preço zerado ou negativo
    precos_invalidos = (df_completo[["Open", "High", "Low", "Close"]] <= 0).any().any()
    print(f"\n  CHECK 9  │ Nenhum preço zerado ou negativo")
    print(f"           │ {'✅ PASS' if not precos_invalidos else '❌ FAIL'}")
    if precos_invalidos: todos_ok = False
        
    # CHECK 10: Gaps intraday > 30 minutos dentro do mesmo dia
    diffs = pd.Series(df_completo.index[1:]) - pd.Series(df_completo.index[:-1])
    gaps = diffs[diffs > pd.Timedelta(minutes=30)]
    gaps_intraday = []
    for i, gap in gaps.items():
        dt_antes = df_completo.index[i]
        dt_depois = df_completo.index[i + 1]
        if dt_antes.date() == dt_depois.date():
            gaps_intraday.append(f"{dt_antes.strftime('%Y-%m-%d %H:%M')} → {dt_depois.strftime('%H:%M')} ({gap.total_seconds()/60:.0f}m)")
            
    ok = len(gaps_intraday) == 0
    print(f"\n  CHECK 10 │ Gaps intraday > 30 minutos no mesmo dia")
    print(f"           │ {'✅ PASS' if ok else '⚠️ WARN'} — {len(gaps_intraday)} gaps encontrados")
    for g in gaps_intraday[:5]:
        print(f"           │   - {g}")
    if len(gaps_intraday) > 5:
        print(f"           │   ... e mais {len(gaps_intraday)-5} gaps.")
        
    print("\n" + "─" * 70)
    if todos_ok:
        print("  ✅ TODAS AS VERIFICAÇÕES PASSARAM — Dados NASDAQ prontos")
    else:
        print("  ❌ ALGUMAS VERIFICAÇÕES FALHARAM — Revisar antes de usar")
    print("─" * 70)
    
    return todos_ok


def imprimir_relatorio(total_bruto, stats, df_completo, df_operacional):
    """Imprime relatório detalhado."""
    sep = "═" * 70
    
    print(f"\n{sep}\n  SEÇÃO 1 — DADOS BRUTOS\n{sep}")
    print(f"  Total CSV original      : {total_bruto:>12,}")
    print(f"  Timeframe detectado     : {stats['timeframe']:>12}")
    print(f"  Período coberto         : {df_completo.index.min().strftime('%Y-%m-%d')} → {df_completo.index.max().strftime('%Y-%m-%d')}")
    print(f"  Dias úteis cobertos     : {df_completo.index.normalize().nunique():>12,}")
    print(f"  Removidos incompl. (M10): {stats.get('removidos_incompletos', 0):>12,}")
    print(f"  Feriados removidos      : {len(stats.get('feriados_removidos_lista', [])):>12,} dias")
    if len(stats.get('feriados_removidos_lista', [])) > 0:
        print(f"  Exemplo feriados        : {', '.join(stats['feriados_removidos_lista'][:5])}...")

    print(f"\n{sep}\n  SEÇÃO 2 — SÉRIE COMPLETA (nasdaq_m10_completo.parquet)\n{sep}")
    print(f"  Total de candles M10    : {len(df_completo):>12,}")
    
    print(f"\n  Distribuição por hora do dia (servidor):")
    for h in range(24):
        count = (df_completo.index.hour == h).sum()
        if count > 0:
            print(f"    {h:02d}h : {count:>8,} candles")
            
    # Volatilidade por hora do dia
    df_temp = df_completo.copy()
    df_temp['hour'] = df_temp.index.hour
    vol_hora = df_temp.groupby('hour')['log_return'].std() * 10000  # em basis points
    print(f"\n  Volatilidade média (basis points) por hora:")
    for h, v in vol_hora.items():
        print(f"    {h:02d}h : {v:>8.2f} bps")

    lr = df_completo["log_return"].dropna()
    print(f"\n  Estatísticas Log-Retornos Intraday:")
    print(f"    Média                 : {lr.mean():>12.8f}")
    print(f"    Desvio Padrão         : {lr.std():>12.8f}")
    print(f"    Curtose               : {lr.kurtosis():>12.4f}")
    print(f"    Assimetria (Skewness) : {lr.skew():>12.4f}")
    print(f"    Mínimo                : {lr.min():>12.8f}")
    print(f"    Máximo                : {lr.max():>12.8f}")

    print(f"\n{sep}\n  SEÇÃO 3 — SÉRIE OPERACIONAL (nasdaq_m10_operacional.parquet)\n{sep}")
    print(f"  Total de candles M10    : {len(df_operacional):>12,}")
    print(f"  Horários cobertos       : 16h30 às 22h30 (servidor MT5)")
    pct_capturado = (len(df_operacional) / len(df_completo)) * 100
    print(f"  % do dia capturado      : {pct_capturado:>11.2f}%")
    fora_horario = df_operacional[(df_operacional.index.strftime('%H:%M') < "16:30") | 
                                  (df_operacional.index.strftime('%H:%M') > "22:30")]
    print(f"  Candles fora do horário : {len(fora_horario):>12,} {'✅' if len(fora_horario)==0 else '❌'}")

    print(f"\n{sep}\n  SEÇÃO 4 — VOLATILIDADE HISTÓRICA POR ANO\n{sep}")
    df_temp['year'] = df_temp.index.year
    candles_por_dia = df_completo.groupby(df_completo.index.date).size().median()
    print(f"  (Assumindo ~{candles_por_dia:.0f} candles M10 / dia)")
    vol_ano = df_temp.groupby('year')['log_return'].std() * np.sqrt(candles_por_dia)
    for y, v in vol_ano.items():
        print(f"    Ano {y} : Vol Realizada Média = {v*100:.2f}% por dia")

    print(f"\n{sep}\n  SEÇÃO 5 — VALIDAÇÃO CRUZADA\n{sep}")
    diff = len(df_completo) - len(df_operacional)
    print(f"  Operacional ⊂ Completa  : {'✅ SIM' if set(df_operacional.index).issubset(set(df_completo.index)) else '❌ NÃO'}")
    print(f"  Diferença em candles    : {diff:>12,}")
    print(f"  % Fechamento (22h30-23h): {(diff / len(df_completo)) * 100:>11.2f}%")


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================
def executar_pipeline(csv_path: Path, forcar: bool = False):
    if PARQUET_COMPLETO.exists() and PARQUET_OPERACIONAL.exists() and not forcar:
        logger.info("Cache encontrado. Usando parquets existentes.")
        return pd.read_parquet(PARQUET_COMPLETO), pd.read_parquet(PARQUET_OPERACIONAL)
        
    DIR_DATA.mkdir(parents=True, exist_ok=True)
    
    # 1. Carregar CSV
    df, total_bruto = carregar_csv_bruto(csv_path)
    
    # 2. Timeframe, Filtro de Horário Completo e Feriados
    df_completo, stats1 = processar_timeframe_e_horario(df)
    del df
    
    # 3. Limpeza OHLC e Log-retornos
    df_completo, stats2 = limpar_dados_e_calcular_retornos(df_completo)
    stats = {**stats1, **stats2}
    
    # 4. Criar Série Operacional (16:30 - 22:30)
    df_operacional = df_completo[
        (df_completo.index.strftime('%H:%M') >= "16:30") &
        (df_completo.index.strftime('%H:%M') <= "22:30")
    ]
    
    # 5. Salvar
    logger.info("Salvando parquets...")
    df_completo.to_parquet(PARQUET_COMPLETO, engine="pyarrow", compression="snappy")
    df_operacional.to_parquet(PARQUET_OPERACIONAL, engine="pyarrow", compression="snappy")
    
    # 6. Checks
    executar_verificacoes(df_completo, df_operacional)
    
    # 7. Relatório
    imprimir_relatorio(total_bruto, stats, df_completo, df_operacional)
    
    return df_completo, df_operacional


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Loader NASDAQ Day Trade M10")
    parser.add_argument("--csv", type=str, default=str(CSV_PADRAO), help="Caminho CSV")
    parser.add_argument("--forcar", action="store_true", help="Forçar reprocessamento")
    args = parser.parse_args()

    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█   DATA LOADER NASDAQ v1 — DAY TRADE M10                         █")
    print("█   Série Completa: 16h30 às 23h00 (Servidor MT5)                  █")
    print("█   Série Operacional: 16h30 às 22h30                              █")
    print("█" + " " * 68 + "█")
    print("█" * 70)

    executar_pipeline(Path(args.csv), args.forcar)
    print("\n✅ Concluído.\n")
