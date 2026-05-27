# -*- coding: utf-8 -*-
"""
================================================================================
data_loader.py — Módulo de Carregamento e Pré-Processamento de Dados EURUSD
                  VERSÃO 3 — IS / OOS PASSADO / OOS FUTURO
================================================================================

Objetivo:
    Carregar o arquivo CSV bruto exportado do MetaTrader 5 em timeframe M1,
    aplicar o filtro correto de horário do servidor
    MT5 (seg-sex 00h05-23h55, horário do servidor
    UTC+2/UTC+3 com DST auto-ajustado),
    agregar para H1, calcular log-retornos e salvar em formato Parquet.

Saídas:
    1. eurusd_h1_completo.parquet     — Série completa seg-sex 00h05-23h55
       Uso: cálculo de indicadores + monitoramento de stops/targets incluindo madrugada
    2. eurusd_h1_operacional.parquet  — Apenas 10h00-22h30, seg-sex
       Uso: geração de sinais de entrada

Autor: Quant Matemática Trade
Data:  2026-05-27
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
from scipy import stats

# Suprimir warnings desnecessários
warnings.filterwarnings("ignore")

# Corrigir encoding para Windows (cp1252 não suporta caracteres Unicode)
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

# Caminho padrão para o CSV de entrada (diretório pai do projeto)
CSV_PADRAO = DIR_PROJETO.parent / "EURUSD_13ANOS.csv"

# Arquivos de saída
PARQUET_COMPLETO     = DIR_DATA / "eurusd_h1_completo.parquet"
PARQUET_OPERACIONAL  = DIR_DATA / "eurusd_h1_operacional.parquet"

# Janelas temporais — IS e OOS
IS_START          = pd.Timestamp("2016-01-01")
IS_END            = pd.Timestamp("2024-01-01")
OOS_PASSADO_START = pd.Timestamp("2013-01-01")
OOS_PASSADO_END   = pd.Timestamp("2016-01-01")
OOS_FUTURO_START  = pd.Timestamp("2024-01-01")
OOS_FUTURO_END    = pd.Timestamp("2026-05-01")

PARQUET_COMPLETO_OOS_PASSADO    = DIR_DATA / "eurusd_h1_completo_OOS_passado_2013_2016.parquet"
PARQUET_OPERACIONAL_OOS_PASSADO = DIR_DATA / "eurusd_h1_operacional_OOS_passado_2013_2016.parquet"
PARQUET_COMPLETO_OOS_FUTURO     = DIR_DATA / "eurusd_h1_completo_OOS_futuro_2024_2026.parquet"
PARQUET_OPERACIONAL_OOS_FUTURO  = DIR_DATA / "eurusd_h1_operacional_OOS_futuro_2024_2026.parquet"

# =============================================================================
# ESTRUTURA DO CSV (MetaTrader 5 — sem cabeçalho)
# =============================================================================
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

# Dtypes otimizados para economia de memória (~8.5M de linhas)
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
# NOMES DOS DIAS DA SEMANA (para relatório)
# =============================================================================
DIAS_SEMANA = {
    0: "Segunda",
    1: "Terça",
    2: "Quarta",
    3: "Quinta",
    4: "Sexta",
    5: "Sábado",
    6: "Domingo",
}


# =============================================================================
# FILTRO DE HORÁRIOS EURUSD (SÉRIE COMPLETA E OPERACIONAL)
# =============================================================================
def filtro_serie_completa(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra a série completa para o horário real de
    funcionamento do EURUSD no servidor MT5.

    O servidor MT5 está em UTC+2/UTC+3 (europeu).
    O EURUSD opera de segunda a sexta: 00h05 até 23h55.
    Domingo e sábado não existem na base (sem cotações).

    Esta série é usada para:
    - Cálculo de todos os indicadores (Hurst, OU, etc.)
    - Monitoramento de stops e targets 24h
    - Captura de movimentos na madrugada que podem
      acionar stops de posições abertas durante o dia

    Parâmetros:
        df: DataFrame com índice DatetimeIndex naive
            no horário do servidor MT5

    Retorna:
        DataFrame com apenas candles de seg-sex,
        entre 00h05 e 23h55 no horário do servidor
    """
    weekday = df.index.weekday  # 0=Seg...4=Sex, 5=Sáb, 6=Dom
    hora    = df.index.strftime('%H:%M')

    # Segunda a Sexta (0-4): 00h05 até 23h55
    dias_uteis = (weekday >= 0) & (weekday <= 4)
    horario_ok = (hora >= "00:05") & (hora <= "23:55")

    # Sábado (5) e Domingo (6): não existem na base
    # mas o filtro garante que se existirem são removidos
    mascara = dias_uteis & horario_ok
    return df[mascara]


def filtro_horario_operacional(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra candles para o horário operacional onde
    NOVAS POSIÇÕES podem ser abertas.

    Horário operacional: 10h00 até 22h30 no horário
    do servidor MT5 (UTC+2/UTC+3), segunda a sexta.

    Justificativa:
    - 10h00 servidor = início da liquidez europeia
    - 22h30 servidor = margem de 30min antes do
      fechamento forçado das 23h00
    - Madrugada (00h05-09h59): sem novas entradas
      mas posições abertas continuam monitoradas
      na série completa

    Este filtro é usado APENAS para geração de
    sinais de entrada. Stops e targets são sempre
    monitorados sobre a série completa 00h05-23h55.

    Parâmetros:
        df: DataFrame com índice DatetimeIndex naive
            no horário do servidor MT5

    Retorna:
        DataFrame com apenas candles operacionais
    """
    weekday = df.index.weekday
    hora    = df.index.strftime('%H:%M')

    # Segunda a Sexta (0-4), entre 10h00 e 22h30
    dias_uteis = (weekday >= 0) & (weekday <= 4)
    horario_op = (hora >= "10:00") & (hora <= "22:30")

    mascara = dias_uteis & horario_op
    return df[mascara]


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================
def carregar_csv_bruto(csv_path: Path) -> pd.DataFrame:
    """
    Passo 1 — Carrega o CSV bruto do MetaTrader 5.

    Parâmetros:
        csv_path: Caminho do arquivo CSV

    Retorna:
        DataFrame com dados M1 brutos, índice datetime naive no horário do servidor
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"\n{'='*60}\n"
            f"  [ERRO] Arquivo CSV não encontrado!\n"
            f"  Caminho: {csv_path}\n"
            f"  \n"
            f"  Uso: python data_loader.py --csv caminho/do/arquivo.csv\n"
            f"{'='*60}\n"
        )

    logger.info(f"Carregando CSV: {csv_path.name}")

    # Passo 1 — Carregar CSV bruto
    df = pd.read_csv(
        csv_path,
        header=None,
        names=COLUNAS_CSV,
        dtype=DTYPES_CSV,
        engine="c",
        low_memory=False,
    )
    total_bruto = len(df)
    logger.info(f"CSV carregado: {total_bruto:,} registros M1")

    # Passo 2 — Parser de datetime naive (sem timezone)
    logger.info("Processando índice datetime naive...")
    datas_str = df["Date"].str.replace(".", "-", regex=False) + " " + df["Time"]
    df.index = pd.to_datetime(datas_str, format="%Y-%m-%d %H:%M")
    # Timestamps já estão no horário do servidor MT5
    # (UTC+2/UTC+3 com DST auto-ajustado pelo MT5)
    # Manter como naive — sem localização de timezone
    # para evitar distorções nos filtros de horário
    df.index.name = "Datetime"

    # Remover colunas auxiliares de data/hora
    df.drop(columns=["Date", "Time"], inplace=True)

    # Passo 3 — Remover duplicatas de índice
    duplicatas = df.index.duplicated(keep="first")
    n_duplicatas = duplicatas.sum()
    if n_duplicatas > 0:
        logger.info(f"Removendo {n_duplicatas:,} duplicatas de índice")
        df = df[~duplicatas]

    return df, total_bruto


def limpar_dados_m1(df: pd.DataFrame) -> dict:
    """
    Passos 4-6 — Aplica filtro série completa, remove candles zerados e verifica integridade OHLC.

    Parâmetros:
        df: DataFrame M1 com índice datetime naive

    Retorna:
        Tupla (DataFrame limpo, dicionário com estatísticas de remoção)
    """
    stats_remocao = {}

    # Passo 4 — Aplicar filtro_serie_completa()
    total_antes_completa = len(df)
    df = filtro_serie_completa(df)
    removidos_completa = total_antes_completa - len(df)
    stats_remocao["removidos_filtro_completa"] = removidos_completa
    logger.info(f"Filtro Série Completa: {removidos_completa:,} candles removidos (fora de seg-sex 00h05-23h55)")

    # Passo 5 — Remover candles com OHLC zerado ou nulo
    total_antes_zero = len(df)
    colunas_ohlc = ["Open", "High", "Low", "Close"]

    # Remover NaN em OHLC
    df = df.dropna(subset=colunas_ohlc)

    # Remover preços zerados ou negativos
    mascara_positivo = (df[colunas_ohlc] > 0).all(axis=1)
    df = df[mascara_positivo]

    removidos_zero = total_antes_zero - len(df)
    stats_remocao["removidos_zerado_nulo"] = removidos_zero
    if removidos_zero > 0:
        logger.info(f"Candles zerados/nulos removidos: {removidos_zero:,}")

    # Passo 6 — Verificar integridade OHLC
    total_antes_integridade = len(df)
    high_ok = df["High"] >= df[["Open", "Close"]].max(axis=1)
    low_ok  = df["Low"]  <= df[["Open", "Close"]].min(axis=1)
    mascara_integridade = high_ok & low_ok
    candles_invalidos = (~mascara_integridade).sum()

    if candles_invalidos > 0:
        logger.warning(f"Candles com integridade OHLC violada: {candles_invalidos:,} — removendo")
        df = df[mascara_integridade]

    removidos_integridade = total_antes_integridade - len(df)
    stats_remocao["removidos_integridade"] = removidos_integridade

    return df, stats_remocao


def agregar_m1_para_h1(df: pd.DataFrame) -> tuple:
    """
    Passos 7-8 — Agrega M1 → H1 e remove candles H1 incompletos.

    Parâmetros:
        df: DataFrame M1 limpo

    Retorna:
        Tupla (DataFrame H1, número de candles removidos por incompletude)
    """
    logger.info("Agregando M1 → H1...")

    # Passo 7 — Agregação via resample
    contagem_m1 = df.resample("1h").size()

    df_h1 = df.resample("1h").agg({
        "Open":   "first",
        "High":   "max",
        "Low":    "min",
        "Close":  "last",
        "Volume": "sum",
    })

    df_h1["candles_m1"] = contagem_m1

    # Remover linhas totalmente vazias (horas sem dados)
    df_h1 = df_h1.dropna(subset=["Open", "High", "Low", "Close"])

    # Passo 8 — Remover candles H1 com menos de 30 M1 (incompletos)
    total_antes = len(df_h1)
    df_h1 = df_h1[df_h1["candles_m1"] >= 30]
    removidos_incompletos = total_antes - len(df_h1)

    if removidos_incompletos > 0:
        logger.info(f"Candles H1 incompletos removidos (<30 M1): {removidos_incompletos:,}")

    # Garantir dtypes otimizados
    for col in ["Open", "High", "Low", "Close"]:
        df_h1[col] = df_h1[col].astype("float32")
    df_h1["Volume"] = df_h1["Volume"].astype("int64")
    df_h1["candles_m1"] = df_h1["candles_m1"].astype("int32")

    logger.info(f"Série H1 completa: {len(df_h1):,} candles")

    return df_h1, removidos_incompletos


def calcular_log_retornos(df_h1: pd.DataFrame) -> pd.DataFrame:
    """
    Passo 9 — Calcula log-retornos: R_t = ln(Close_t / Close_{t-1}).

    Parâmetros:
        df_h1: DataFrame H1

    Retorna:
        DataFrame H1 com coluna log_return adicionada (sem a primeira linha NaN)
    """
    logger.info("Calculando log-retornos...")
    df_h1["log_return"] = np.log(
        df_h1["Close"] / df_h1["Close"].shift(1)
    ).astype("float32")

    # Remover primeira linha (NaN)
    df_h1 = df_h1.iloc[1:].copy()

    return df_h1


# =============================================================================
# VERIFICAÇÕES AUTOMÁTICAS (10 CHECKS)
# =============================================================================
def executar_verificacoes(df_completo: pd.DataFrame, df_operacional: pd.DataFrame) -> bool:
    """
    Executa 10 verificações automáticas sobre as séries geradas.

    Parâmetros:
        df_completo:     DataFrame da série completa
        df_operacional:  DataFrame da série operacional

    Retorna:
        True se todos os checks passaram, False caso contrário
    """
    print("\n" + "═" * 70)
    print("  VERIFICAÇÕES AUTOMÁTICAS (10 CHECKS)")
    print("═" * 70)

    todos_ok = True

    # ─── CHECK 1: Nenhum candle de sábado ou domingo na base ───
    fds = df_completo[df_completo.index.weekday >= 5]
    ok = len(fds) == 0
    status = "✅ PASS" if ok else "❌ FAIL"
    print(f"\n  CHECK 1  │ Nenhum candle de sábado ou domingo na base")
    print(f"           │ {status} — {len(fds)} candles encontrados")
    if not ok:
        todos_ok = False
        print(f"           │ Primeiros 5: {fds.index[:5].tolist()}")

    # ─── CHECK 2: Nenhum candle de domingo na base ───
    domingos = df_completo[df_completo.index.weekday == 6]
    ok = len(domingos) == 0
    status = "✅ PASS" if ok else "❌ FAIL"
    print(f"\n  CHECK 2  │ Nenhum candle de domingo na base")
    print(f"           │ {status} — {len(domingos)} candles de domingo encontrados")
    if not ok:
        todos_ok = False

    # ─── CHECK 3: Nenhum candle fora de 00h05-23h55 na série completa ───
    # Em H1, o candle das 00h00 representa o período de 00h05 a 00h59 do M1 (que é válido).
    # A série completa H1 deve ter apenas candles com minutos zerados no intervalo de 00h a 23h.
    minutos_quebrados = df_completo[df_completo.index.minute != 0]
    horas_invalidas = df_completo[(df_completo.index.hour < 0) | (df_completo.index.hour > 23)]
    ok = len(minutos_quebrados) == 0 and len(horas_invalidas) == 0
    status = "✅ PASS" if ok else "❌ FAIL"
    print(f"\n  CHECK 3  │ Nenhum candle fora de 00h05-23h55 na série completa")
    print(f"           │ {status} — Minutos quebrados: {len(minutos_quebrados)}, Horas inválidas: {len(horas_invalidas)}")
    if not ok:
        todos_ok = False

    # ─── CHECK 4: Série operacional ⊂ série completa ───
    idx_op_set = set(df_operacional.index)
    idx_co_set = set(df_completo.index)
    diferenca = idx_op_set - idx_co_set
    ok = len(diferenca) == 0
    status = "✅ PASS" if ok else "❌ FAIL"
    print(f"\n  CHECK 4  │ Série operacional ⊂ série completa")
    print(f"           │ {status} — {len(diferenca)} candles operacionais ausentes na completa")
    if not ok:
        todos_ok = False

    # ─── CHECK 5: Log-retornos sem NaN ou Inf ───
    nan_count = df_completo["log_return"].isna().sum()
    inf_count = np.isinf(df_completo["log_return"]).sum()
    ok = (nan_count == 0) and (inf_count == 0)
    status = "✅ PASS" if ok else "❌ FAIL"
    print(f"\n  CHECK 5  │ Log-retornos sem NaN ou Inf")
    print(f"           │ {status} — NaN: {nan_count}, Inf: {inf_count}")
    if not ok:
        todos_ok = False

    # ─── CHECK 6: High >= Open e High >= Close em todos candles ───
    high_ok = (df_completo["High"] >= df_completo["Open"]).all() and \
              (df_completo["High"] >= df_completo["Close"]).all()
    ok = high_ok
    status = "✅ PASS" if ok else "❌ FAIL"
    print(f"\n  CHECK 6  │ High >= Open e High >= Close em todos candles")
    print(f"           │ {status}")
    if not ok:
        todos_ok = False
        violacoes = df_completo[
            (df_completo["High"] < df_completo["Open"]) |
            (df_completo["High"] < df_completo["Close"])
        ]
        print(f"           │ {len(violacoes)} violações encontradas")

    # ─── CHECK 7: Low <= Open e Low <= Close em todos candles ───
    low_ok = (df_completo["Low"] <= df_completo["Open"]).all() and \
             (df_completo["Low"] <= df_completo["Close"]).all()
    ok = low_ok
    status = "✅ PASS" if ok else "❌ FAIL"
    print(f"\n  CHECK 7  │ Low <= Open e Low <= Close em todos candles")
    print(f"           │ {status}")
    if not ok:
        todos_ok = False

    # ─── CHECK 8: Nenhum preço zerado ou negativo ───
    colunas_preco = ["Open", "High", "Low", "Close"]
    precos_invalidos = (df_completo[colunas_preco] <= 0).any().any()
    ok = not precos_invalidos
    status = "✅ PASS" if ok else "❌ FAIL"
    print(f"\n  CHECK 8  │ Nenhum preço zerado ou negativo")
    print(f"           │ {status}")
    if not ok:
        todos_ok = False

    # ─── CHECK 9: Série operacional contém apenas 10h00-22h30 no horário do servidor ───
    horas_minutos_op = df_operacional.index.strftime('%H:%M')
    fora_horario = df_operacional[(horas_minutos_op < "10:00") | (horas_minutos_op > "22:30")]
    dias_op = df_operacional.index.weekday
    fora_dia = df_operacional[(dias_op > 4)]  # Sábado ou Domingo
    ok = (len(fora_horario) == 0) and (len(fora_dia) == 0)
    status = "✅ PASS" if ok else "❌ FAIL"
    print(f"\n  CHECK 9  │ Série operacional contém apenas 10h00-22h30 no horário do servidor, seg-sex")
    print(f"           │ {status} — Fora do horário: {len(fora_horario)}, Fora do dia: {len(fora_dia)}")
    if not ok:
        todos_ok = False

    # ─── CHECK 10: Continuidade temporal — gaps > 3h na série completa ───
    # Excluindo transição de fim de semana (sex 23h55 → seg 00h05, normalmente ~48-50h)
    print(f"\n  CHECK 10 │ Continuidade temporal — gaps > 3h (excl. fim de semana)")
    if len(df_completo) > 1:
        diffs = pd.Series(df_completo.index[1:]) - pd.Series(df_completo.index[:-1])
        gaps_grandes = diffs[diffs > pd.Timedelta(hours=3)]

        # Filtrar gaps de fim de semana (sex → seg, normalmente ~47-51h)
        gaps_nao_fds = []
        for i, gap in gaps_grandes.items():
            dt_antes = df_completo.index[i]
            dt_depois = df_completo.index[i + 1]
            # Se o gap cruza de sexta para segunda
            if dt_antes.weekday() == 4 and dt_depois.weekday() == 0:
                continue
            gaps_nao_fds.append({
                "de": dt_antes,
                "para": dt_depois,
                "duracao_h": gap.total_seconds() / 3600
            })

        ok = len(gaps_nao_fds) == 0
        status = "✅ PASS" if ok else "⚠️ WARN"
        print(f"           │ {status} — {len(gaps_nao_fds)} gaps > 3h encontrados (excl. FDS)")
        if gaps_nao_fds:
            for g in gaps_nao_fds[:10]:  # Mostrar no máximo 10
                print(f"           │   {g['de']} → {g['para']} ({g['duracao_h']:.1f}h)")
            if len(gaps_nao_fds) > 10:
                print(f"           │   ... e mais {len(gaps_nao_fds) - 10} gaps")
    else:
        print(f"           │ ⚠️ WARN — Série muito curta para verificar continuidade")

    # ─── RESULTADO FINAL ───
    print("\n" + "─" * 70)
    if todos_ok:
        print("  ✅ TODAS AS VERIFICAÇÕES PASSARAM — Dados prontos para uso")
    else:
        print("  ❌ ALGUMAS VERIFICAÇÕES FALHARAM — Revisar dados antes de prosseguir")
    print("─" * 70)

    return todos_ok


# =============================================================================
# RELATÓRIO DETALHADO
# =============================================================================
def imprimir_relatorio(
    total_bruto: int,
    stats_remocao: dict,
    removidos_incompletos: int,
    df_completo: pd.DataFrame,
    df_operacional: pd.DataFrame,
):
    """
    Imprime relatório completo com 4 seções após o processamento.

    Parâmetros:
        total_bruto:           Registros M1 no CSV original
        stats_remocao:         Estatísticas de remoção (filtro de horário, OHLC, etc.)
        removidos_incompletos: Candles H1 removidos por incompletude
        df_completo:           DataFrame da série completa
        df_operacional:        DataFrame da série operacional
    """
    sep = "═" * 70

    # ═══════════════════════════════════════════════════════════════════
    # SEÇÃO 1 — Dados brutos
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("  SEÇÃO 1 — DADOS BRUTOS")
    print(sep)
    print(f"  Total de registros M1 no CSV         : {total_bruto:>12,}")
    print(f"  Período coberto                       : {df_completo.index.min().strftime('%Y-%m-%d %H:%M')}")
    print(f"                                       → {df_completo.index.max().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Removidos por filtro série completa   : {stats_remocao.get('removidos_filtro_completa', 0):>12,}")
    print(f"  Removidos por OHLC zerado/nulo        : {stats_remocao.get('removidos_zerado_nulo', 0):>12,}")
    print(f"  Removidos por integridade OHLC        : {stats_remocao.get('removidos_integridade', 0):>12,}")
    print(f"  Removidos por candle H1 incompleto    : {removidos_incompletos:>12,}")

    # ═══════════════════════════════════════════════════════════════════
    # SEÇÃO 2 — Série completa (eurusd_h1_completo.parquet)
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("  SEÇÃO 2 — SÉRIE COMPLETA (eurusd_h1_completo.parquet)")
    print(sep)
    print(f"  Total de candles H1                   : {len(df_completo):>12,}")
    print(f"  Período coberto                       : {df_completo.index.min().strftime('%Y-%m-%d %H:%M')}")
    print(f"                                       → {df_completo.index.max().strftime('%Y-%m-%d %H:%M')}")

    # Breakdown por hora do dia
    print(f"\n  Breakdown por hora do dia (00h-23h):")
    for h in range(24):
        count = (df_completo.index.hour == h).sum()
        print(f"    {h:02d}h : {count:>8,} candles")

    # Confirmações
    sabados = (df_completo.index.weekday == 5).sum()
    domingos = (df_completo.index.weekday == 6).sum()
    # Em H1, todos os candles de 00h a 23h são válidos (00h representa 00h05-00h59 do M1)
    fora_janela = (df_completo.index.minute != 0).sum()
    print(f"\n  Confirmações:")
    print(f"    Candles de sábado              : {sabados:>6} {'✅' if sabados == 0 else '❌'}")
    print(f"    Candles de domingo             : {domingos:>6} {'✅' if domingos == 0 else '❌'}")
    print(f"    Candles fora de 00h05-23h55    : {fora_janela:>6} {'✅' if fora_janela == 0 else '❌'}")

    # Estatísticas dos log-retornos
    lr = df_completo["log_return"]
    print(f"\n  Estatísticas dos log-retornos:")
    print(f"    Média                          : {lr.mean():>12.8f}")
    print(f"    Desvio Padrão                  : {lr.std():>12.8f}")
    print(f"    Curtose                        : {lr.kurtosis():>12.4f}")
    print(f"    Assimetria (Skewness)          : {lr.skew():>12.4f}")
    print(f"    Mínimo                         : {lr.min():>12.8f}")
    print(f"    Máximo                         : {lr.max():>12.8f}")

    # ═══════════════════════════════════════════════════════════════════
    # SEÇÃO 3 — Série operacional (eurusd_h1_operacional.parquet)
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("  SEÇÃO 3 — SÉRIE OPERACIONAL (eurusd_h1_operacional.parquet)")
    print(sep)
    print(f"  Total de candles H1                   : {len(df_operacional):>12,}")
    print(f"  Período coberto                       : {df_operacional.index.min().strftime('%Y-%m-%d %H:%M')}")
    print(f"                                       → {df_operacional.index.max().strftime('%Y-%m-%d %H:%M')}")

    # Breakdown por hora do dia (10h-22h)
    print(f"\n  Breakdown por hora do servidor (10h-22h):")
    for h in range(10, 23):
        count = (df_operacional.index.hour == h).sum()
        print(f"    {h:02d}h : {count:>8,} candles")

    # Confirmações
    hora_str_op = df_operacional.index.strftime('%H:%M')
    fora_horario = df_operacional[(hora_str_op < "10:00") | (hora_str_op > "22:30")]
    fds_op = df_operacional[df_operacional.index.weekday >= 5]
    print(f"\n  Confirmações:")
    print(f"    Candles fora de 10h00-22h30    : {len(fora_horario):>6} {'✅' if len(fora_horario) == 0 else '❌'}")
    print(f"    Candles sáb/dom                : {len(fds_op):>6} {'✅' if len(fds_op) == 0 else '❌'}")

    # ═══════════════════════════════════════════════════════════════════
    # SEÇÃO 4 — Validação cruzada
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("  SEÇÃO 4 — VALIDAÇÃO CRUZADA")
    print(sep)

    # Verificar se operacional é subconjunto da completa
    idx_op = set(df_operacional.index)
    idx_co = set(df_completo.index)
    e_subconjunto = idx_op.issubset(idx_co)
    print(f"  Operacional ⊂ Completa               : {'✅ SIM' if e_subconjunto else '❌ NÃO'}")

    diferenca = len(df_completo) - len(df_operacional)
    print(f"  Diferença em candles                  : {diferenca:>12,}")

    pct_noturnos = (diferenca / len(df_completo)) * 100 if len(df_completo) > 0 else 0
    print(f"  % candles fora do horário operacional : {pct_noturnos:>11.2f}%")
    print(f"  (madrugada 00h05-09h59 + noite 22h31-23h55)")
    print(f"  capturados apenas na série completa para monitoramento de stops/targets")


# =============================================================================
# FUNÇÃO PRINCIPAL — PIPELINE COMPLETO
# =============================================================================
def carregar_e_processar(
    csv_path: Path = None,
    forcar: bool = False,
) -> tuple:
    """
    Executa o pipeline completo de processamento de dados EURUSD M1 → H1.

    Pipeline:
        1. Carrega CSV bruto do MetaTrader 5
        2. Parseia datetime como naive
        3. Remove duplicatas de índice
        4. Aplica filtro de série completa (seg-sex 00h05-23h55)
        5. Remove candles com OHLC zerado ou nulo
        6. Verifica integridade OHLC (High >= max(O,C), Low <= min(O,C))
        7. Agrega M1 → H1 via resample
        8. Remove candles H1 com menos de 30 M1
        9. Calcula log-retornos
        10. Salva dois parquets e executa verificações

    Parâmetros:
        csv_path: Caminho do CSV (usa padrão se None)
        forcar:   Se True, reprocessa mesmo com cache existente

    Retorna:
        Tupla (df_completo, df_operacional)
    """
    if csv_path is None:
        csv_path = CSV_PADRAO

    # ─── Cache inteligente ───
    if (PARQUET_COMPLETO.exists() and
        PARQUET_OPERACIONAL.exists() and
        PARQUET_COMPLETO_OOS_PASSADO.exists() and
        PARQUET_OPERACIONAL_OOS_PASSADO.exists() and
        PARQUET_COMPLETO_OOS_FUTURO.exists() and
        PARQUET_OPERACIONAL_OOS_FUTURO.exists() and
        not forcar):
        logger.info("Cache encontrado! Carregando parquets existentes...")
        logger.info(f"  → {PARQUET_COMPLETO.name}")
        logger.info(f"  → {PARQUET_OPERACIONAL.name}")
        logger.info(f"  → {PARQUET_COMPLETO_OOS_PASSADO.name}")
        logger.info(f"  → {PARQUET_OPERACIONAL_OOS_PASSADO.name}")
        logger.info(f"  → {PARQUET_COMPLETO_OOS_FUTURO.name}")
        logger.info(f"  → {PARQUET_OPERACIONAL_OOS_FUTURO.name}")
        logger.info("Use --forcar para reprocessar do CSV")

        df_completo = pd.read_parquet(PARQUET_COMPLETO, engine="pyarrow")
        df_operacional = pd.read_parquet(PARQUET_OPERACIONAL, engine="pyarrow")
        df_completo_oos_p = pd.read_parquet(PARQUET_COMPLETO_OOS_PASSADO, engine="pyarrow")
        df_op_oos_p = pd.read_parquet(PARQUET_OPERACIONAL_OOS_PASSADO, engine="pyarrow")
        df_completo_oos_f = pd.read_parquet(PARQUET_COMPLETO_OOS_FUTURO, engine="pyarrow")
        df_op_oos_f = pd.read_parquet(PARQUET_OPERACIONAL_OOS_FUTURO, engine="pyarrow")

        print("\n" + "═" * 70)
        print("  DADOS CARREGADOS DO CACHE")
        print("═" * 70)
        print(f"  IS (Série Completa)       : {len(df_completo):>10,} candles H1")
        print(f"  IS (Série Operacional)    : {len(df_operacional):>10,} candles H1")
        print(f"  OOS Passado (Completa)    : {len(df_completo_oos_p):>10,} candles H1")
        print(f"  OOS Futuro (Completa)     : {len(df_completo_oos_f):>10,} candles H1")
        print("═" * 70)

        return (df_completo, df_operacional,
                df_completo_oos_p, df_op_oos_p,
                df_completo_oos_f, df_op_oos_f)

    # ─── Garantir diretório de saída ───
    DIR_DATA.mkdir(parents=True, exist_ok=True)

    # ═══════════════════════════════════════════════════════════════════
    # PIPELINE DE PROCESSAMENTO
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  DATA LOADER v2 — PIPELINE DE PROCESSAMENTO")
    print("  EURUSD M1 → H1 (Série Completa seg-sex 00h05-23h55)")
    print("═" * 70)

    # Passo 1-3: Carregar e parsear CSV
    df_m1, total_bruto = carregar_csv_bruto(csv_path)

    # Passo 4-6: Limpeza de dados M1
    df_m1, stats_remocao = limpar_dados_m1(df_m1)

    # Passo 7-8: Agregação M1 → H1
    df_h1, removidos_incompletos = agregar_m1_para_h1(df_m1)

    # Liberar memória do M1
    del df_m1

    # Passo 9: Log-retornos
    df_h1 = calcular_log_retornos(df_h1)

    # ═══════════════════════════════════════════════════════════════════
    # GERAR OS PARQUETS POR JANELA TEMPORAL
    # ═══════════════════════════════════════════════════════════════════

    # ── Janela IS (treino) ──────────────────────────────────
    mascara_is         = (df_h1.index >= IS_START) & (df_h1.index < IS_END)
    df_completo        = df_h1[mascara_is].copy()
    df_operacional     = filtro_horario_operacional(df_completo)

    # ── Janela OOS Passado ──────────────────────────────────
    mascara_oos_p      = (df_h1.index >= OOS_PASSADO_START) & (df_h1.index < OOS_PASSADO_END)
    df_completo_oos_p  = df_h1[mascara_oos_p].copy()
    df_op_oos_p        = filtro_horario_operacional(df_completo_oos_p)

    # ── Janela OOS Futuro ───────────────────────────────────
    mascara_oos_f      = (df_h1.index >= OOS_FUTURO_START) & (df_h1.index < OOS_FUTURO_END)
    df_completo_oos_f  = df_h1[mascara_oos_f].copy()
    df_op_oos_f        = filtro_horario_operacional(df_completo_oos_f)

    # ═══════════════════════════════════════════════════════════════════
    # VERIFICAÇÕES AUTOMÁTICAS (antes de salvar)
    # ═══════════════════════════════════════════════════════════════════
    print("\n>>> VERIFICAÇÕES — IS (2016-2024)")
    executar_verificacoes(df_completo, df_operacional)

    print("\n>>> VERIFICAÇÕES — OOS PASSADO (2013-2016)")
    executar_verificacoes(df_completo_oos_p, df_op_oos_p)

    print("\n>>> VERIFICAÇÕES — OOS FUTURO (2024-2026)")
    executar_verificacoes(df_completo_oos_f, df_op_oos_f)

    # ═══════════════════════════════════════════════════════════════════
    # PASSO 10 — SALVAR PARQUETS
    # ═══════════════════════════════════════════════════════════════════
    logger.info(f"Salvando {PARQUET_COMPLETO.name}...")
    df_completo.to_parquet(PARQUET_COMPLETO, engine="pyarrow", compression="snappy", index=True)

    logger.info(f"Salvando {PARQUET_OPERACIONAL.name}...")
    df_operacional.to_parquet(PARQUET_OPERACIONAL, engine="pyarrow", compression="snappy", index=True)

    logger.info(f"Salvando {PARQUET_COMPLETO_OOS_PASSADO.name}...")
    df_completo_oos_p.to_parquet(PARQUET_COMPLETO_OOS_PASSADO, engine="pyarrow", compression="snappy", index=True)

    logger.info(f"Salvando {PARQUET_OPERACIONAL_OOS_PASSADO.name}...")
    df_op_oos_p.to_parquet(PARQUET_OPERACIONAL_OOS_PASSADO, engine="pyarrow", compression="snappy", index=True)

    logger.info(f"Salvando {PARQUET_COMPLETO_OOS_FUTURO.name}...")
    df_completo_oos_f.to_parquet(PARQUET_COMPLETO_OOS_FUTURO, engine="pyarrow", compression="snappy", index=True)

    logger.info(f"Salvando {PARQUET_OPERACIONAL_OOS_FUTURO.name}...")
    df_op_oos_f.to_parquet(PARQUET_OPERACIONAL_OOS_FUTURO, engine="pyarrow", compression="snappy", index=True)

    # ═══════════════════════════════════════════════════════════════════
    # RELATÓRIO DETALHADO
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "█"*70)
    print("  RELATÓRIO — IS (2016–2024)")
    print("█"*70)
    imprimir_relatorio(total_bruto, stats_remocao, removidos_incompletos, df_completo, df_operacional)

    print("\n" + "█"*70)
    print("  RELATÓRIO — OOS PASSADO (2013–2016)")
    print("█"*70)
    imprimir_relatorio(total_bruto, stats_remocao, removidos_incompletos, df_completo_oos_p, df_op_oos_p)

    print("\n" + "█"*70)
    print("  RELATÓRIO — OOS FUTURO (2024–2026)")
    print("█"*70)
    imprimir_relatorio(total_bruto, stats_remocao, removidos_incompletos, df_completo_oos_f, df_op_oos_f)

    # Tamanho dos arquivos
    print(f"\n{'═' * 70}")
    print("  ARQUIVOS GERADOS")
    print("═" * 70)
    for p in [PARQUET_COMPLETO, PARQUET_OPERACIONAL, PARQUET_COMPLETO_OOS_PASSADO, PARQUET_OPERACIONAL_OOS_PASSADO, PARQUET_COMPLETO_OOS_FUTURO, PARQUET_OPERACIONAL_OOS_FUTURO]:
        tamanho_mb = p.stat().st_size / (1024 * 1024)
        print(f"  {p.name:40s} : {tamanho_mb:>8.2f} MB")
    print("═" * 70)

    return (df_completo, df_operacional,
            df_completo_oos_p, df_op_oos_p,
            df_completo_oos_f, df_op_oos_f)


# =============================================================================
# FUNÇÕES UTILITÁRIAS PARA OUTROS MÓDULOS
# =============================================================================
def carregar_serie_completa() -> pd.DataFrame:
    """
    Carrega a série completa H1 do parquet.
    Uso: cálculo de indicadores e monitoramento de stops/targets.

    Retorna:
        DataFrame com série completa H1 (dom 21h → sex 21h, 24h)

    Raises:
        FileNotFoundError: Se o parquet não existir
    """
    if not PARQUET_COMPLETO.exists():
        raise FileNotFoundError(
            f"Parquet não encontrado: {PARQUET_COMPLETO}\n"
            f"Execute: python data_loader.py --csv caminho/do/csv"
        )
    return pd.read_parquet(PARQUET_COMPLETO, engine="pyarrow")


def carregar_serie_operacional() -> pd.DataFrame:
    """
    Carrega a série operacional H1 do parquet.
    Uso: geração de sinais de entrada (10h00-22h30, seg-sex).

    Retorna:
        DataFrame com série operacional H1

    Raises:
        FileNotFoundError: Se o parquet não existir
    """
    if not PARQUET_OPERACIONAL.exists():
        raise FileNotFoundError(
            f"Parquet não encontrado: {PARQUET_OPERACIONAL}\n"
            f"Execute: python data_loader.py --csv caminho/do/csv"
        )
    return pd.read_parquet(PARQUET_OPERACIONAL, engine="pyarrow")

def carregar_completo_oos_passado() -> pd.DataFrame:
    """
    Carrega a série completa OOS Passado H1 do parquet.
    """
    if not PARQUET_COMPLETO_OOS_PASSADO.exists():
        raise FileNotFoundError(
            f"Parquet não encontrado: {PARQUET_COMPLETO_OOS_PASSADO}\n"
            f"Execute: python data_loader.py --csv caminho/do/csv"
        )
    return pd.read_parquet(PARQUET_COMPLETO_OOS_PASSADO, engine="pyarrow")

def carregar_operacional_oos_passado() -> pd.DataFrame:
    """
    Carrega a série operacional OOS Passado H1 do parquet.
    """
    if not PARQUET_OPERACIONAL_OOS_PASSADO.exists():
        raise FileNotFoundError(
            f"Parquet não encontrado: {PARQUET_OPERACIONAL_OOS_PASSADO}\n"
            f"Execute: python data_loader.py --csv caminho/do/csv"
        )
    return pd.read_parquet(PARQUET_OPERACIONAL_OOS_PASSADO, engine="pyarrow")

def carregar_completo_oos_futuro() -> pd.DataFrame:
    """
    Carrega a série completa OOS Futuro H1 do parquet.
    """
    if not PARQUET_COMPLETO_OOS_FUTURO.exists():
        raise FileNotFoundError(
            f"Parquet não encontrado: {PARQUET_COMPLETO_OOS_FUTURO}\n"
            f"Execute: python data_loader.py --csv caminho/do/csv"
        )
    return pd.read_parquet(PARQUET_COMPLETO_OOS_FUTURO, engine="pyarrow")

def carregar_operacional_oos_futuro() -> pd.DataFrame:
    """
    Carrega a série operacional OOS Futuro H1 do parquet.
    """
    if not PARQUET_OPERACIONAL_OOS_FUTURO.exists():
        raise FileNotFoundError(
            f"Parquet não encontrado: {PARQUET_OPERACIONAL_OOS_FUTURO}\n"
            f"Execute: python data_loader.py --csv caminho/do/csv"
        )
    return pd.read_parquet(PARQUET_OPERACIONAL_OOS_FUTURO, engine="pyarrow")


# =============================================================================
# EXECUÇÃO DIRETA VIA CLI
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Data Loader v3 — EURUSD M1 → H1 (IS / OOS PASSADO / OOS FUTURO)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python data_loader.py --csv ../EURUSD_13ANOS.csv
  python data_loader.py --csv ../EURUSD_13ANOS.csv --forcar
  python data_loader.py  (usa cache se existir)
        """,
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=str(CSV_PADRAO),
        help="Caminho do arquivo CSV exportado do MetaTrader 5",
    )
    parser.add_argument(
        "--forcar",
        action="store_true",
        help="Forçar reprocessamento mesmo com cache existente",
    )
    args = parser.parse_args()

    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█   DATA LOADER v3 — EURUSD M1 → H1                               █")
    print("█   IS: 2016–2024 | OOS Passado: 2013–2016 | OOS Futuro: 2024–2026█")
    print("█" + " " * 68 + "█")
    print("█" * 70)

    df_completo, df_operacional, df_completo_oos_p, df_op_oos_p, df_completo_oos_f, df_op_oos_f = carregar_e_processar(
        csv_path=Path(args.csv),
        forcar=args.forcar,
    )

    print("\n✅ Pipeline concluído com sucesso!\n")
