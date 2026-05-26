# -*- coding: utf-8 -*-
"""
================================================================================
data_loader.py — Módulo de Carregamento e Pré-Processamento de Dados EURUSD
                  VERSÃO 2 — DEFINITIVA
================================================================================

Objetivo:
    Carregar o arquivo CSV bruto exportado do MetaTrader 5 em timeframe M1,
    aplicar o filtro correto de horário do mercado FOREX (dom 21h → sex 21h UTC),
    agregar para H1, calcular log-retornos e salvar em formato Parquet.

Correção crítica em relação à v1:
    A versão anterior removia barras noturnas ANTES de salvar o parquet,
    fazendo com que stops e targets atingidos durante a madrugada não
    fossem capturados no backtest — gerando resultados irreais e otimistas.

Saídas:
    1. eurusd_h1_completo.parquet     — Série 24h completa (dom 21h → sex 21h)
       Uso: cálculo de indicadores + monitoramento de stops/targets
    2. eurusd_h1_operacional.parquet  — Apenas 07h-20h UTC, seg-sex
       Uso: geração de sinais de entrada (evita baixa liquidez)

Autor: Quant Matemática Trade
Data:  2026-05-26
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
CSV_PADRAO = DIR_PROJETO.parent / "EURUSD_10ANOS.csv"

# Arquivos de saída
PARQUET_COMPLETO     = DIR_DATA / "eurusd_h1_completo.parquet"
PARQUET_OPERACIONAL  = DIR_DATA / "eurusd_h1_operacional.parquet"

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
# FILTRO DE MERCADO FOREX — JANELA CORRETA DOM 21H → SEX 21H UTC
# =============================================================================
def filtro_mercado_forex(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica o filtro de horário do mercado FOREX spot.

    Estrutura real do mercado FOREX:
        ABERTURA:    domingo às 21h00 UTC (Sydney)
        FECHAMENTO:  sexta-feira às 21h59 UTC (New York)

    Mantém:
        - Domingo:   apenas 21h00-23h59 UTC (abertura Sydney)
        - Seg-Qui:   00h00-23h59 UTC (24h completas)
        - Sexta:     00h00-21h59 UTC (até fechamento NY)

    Remove:
        - Sábado:    TODOS os candles (mercado fechado)
        - Domingo:   00h00-20h59 UTC (antes da abertura)
        - Sexta:     22h00-23h59 UTC (após fechamento)

    Parâmetros:
        df: DataFrame com índice DatetimeIndex em UTC

    Retorna:
        DataFrame filtrado contendo apenas candles dentro da janela FOREX
    """
    weekday = df.index.weekday  # 0=Seg, 1=Ter, ..., 4=Sex, 5=Sáb, 6=Dom
    hour    = df.index.hour

    # Domingo: apenas 21h-23h (abertura Sydney)
    domingo_valido = (weekday == 6) & (hour >= 21)

    # Segunda a Quinta: 24h completas
    seg_qui_valido = (weekday >= 0) & (weekday <= 3)

    # Sexta: apenas 00h-21h (até fechamento NY)
    sexta_valida = (weekday == 4) & (hour <= 21)

    # Sábado: nada (weekday == 5 não está em nenhuma máscara)
    # Domingo antes das 21h: nada (não atende domingo_valido)

    mascara = domingo_valido | seg_qui_valido | sexta_valida
    return df[mascara]


def filtro_horario_operacional(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra candles para o horário operacional onde NOVAS POSIÇÕES podem ser abertas.

    Horário operacional: 07h00 UTC até 20h59 UTC, segunda a sexta.
    Exclui: madrugada (00h-06h), domingo, sexta noite (21h+).

    Este filtro é mais restritivo que o filtro_mercado_forex e é usado
    APENAS para gerar sinais de entrada — nunca para cálculo de indicadores
    ou monitoramento de stops/targets.

    Parâmetros:
        df: DataFrame com índice DatetimeIndex em UTC

    Retorna:
        DataFrame contendo apenas candles no horário operacional
    """
    weekday = df.index.weekday
    hour    = df.index.hour

    # Segunda a Sexta (0-4), entre 07h e 20h
    mascara = (weekday >= 0) & (weekday <= 4) & (hour >= 7) & (hour <= 20)
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
        DataFrame com dados M1 brutos, índice datetime UTC
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

    # Passo 2 — Parser de datetime com timezone UTC
    logger.info("Processando índice datetime UTC...")
    datas_str = df["Date"].str.replace(".", "-", regex=False) + " " + df["Time"]
    df.index = pd.to_datetime(datas_str, format="%Y-%m-%d %H:%M")
    df.index = df.index.tz_localize("UTC")
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
    Passos 4-6 — Aplica filtro FOREX, remove candles zerados e verifica integridade OHLC.

    Parâmetros:
        df: DataFrame M1 com índice datetime UTC

    Retorna:
        Tupla (DataFrame limpo, dicionário com estatísticas de remoção)
    """
    stats_remocao = {}

    # Passo 4 — Aplicar filtro_mercado_forex()
    total_antes_forex = len(df)
    df = filtro_mercado_forex(df)
    removidos_forex = total_antes_forex - len(df)
    stats_remocao["removidos_filtro_forex"] = removidos_forex
    logger.info(f"Filtro FOREX: {removidos_forex:,} candles removidos (fora da janela dom 21h → sex 21h)")

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
        df_completo:     DataFrame da série completa (dom 21h → sex 21h)
        df_operacional:  DataFrame da série operacional (07h-20h seg-sex)

    Retorna:
        True se todos os checks passaram, False caso contrário
    """
    print("\n" + "═" * 70)
    print("  VERIFICAÇÕES AUTOMÁTICAS (10 CHECKS)")
    print("═" * 70)

    todos_ok = True

    # ─── CHECK 1: Nenhum candle de sábado na série completa ───
    sabados = df_completo[df_completo.index.weekday == 5]
    ok = len(sabados) == 0
    status = "✅ PASS" if ok else "❌ FAIL"
    print(f"\n  CHECK 1  │ Nenhum candle de sábado na série completa")
    print(f"           │ {status} — {len(sabados)} candles de sábado encontrados")
    if not ok:
        todos_ok = False
        print(f"           │ Primeiros 5: {sabados.index[:5].tolist()}")

    # ─── CHECK 2: Nenhum candle domingo antes das 21h UTC ───
    domingos_cedo = df_completo[
        (df_completo.index.weekday == 6) & (df_completo.index.hour < 21)
    ]
    ok = len(domingos_cedo) == 0
    status = "✅ PASS" if ok else "❌ FAIL"
    print(f"\n  CHECK 2  │ Nenhum candle domingo antes das 21h UTC")
    print(f"           │ {status} — {len(domingos_cedo)} candles encontrados")
    if not ok:
        todos_ok = False

    # ─── CHECK 3: Nenhum candle sexta após 21h UTC ───
    sexta_tarde = df_completo[
        (df_completo.index.weekday == 4) & (df_completo.index.hour > 21)
    ]
    ok = len(sexta_tarde) == 0
    status = "✅ PASS" if ok else "❌ FAIL"
    print(f"\n  CHECK 3  │ Nenhum candle sexta após 21h UTC")
    print(f"           │ {status} — {len(sexta_tarde)} candles encontrados")
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

    # ─── CHECK 9: Série operacional contém apenas 07h-20h UTC ───
    horas_op = df_operacional.index.hour
    fora_horario = df_operacional[(horas_op < 7) | (horas_op > 20)]
    dias_op = df_operacional.index.weekday
    fora_dia = df_operacional[(dias_op > 4)]  # Sábado ou Domingo
    ok = (len(fora_horario) == 0) and (len(fora_dia) == 0)
    status = "✅ PASS" if ok else "❌ FAIL"
    print(f"\n  CHECK 9  │ Série operacional contém apenas 07h-20h UTC, seg-sex")
    print(f"           │ {status} — Fora do horário: {len(fora_horario)}, Fora do dia: {len(fora_dia)}")
    if not ok:
        todos_ok = False

    # ─── CHECK 10: Continuidade temporal — gaps > 3h na série completa ───
    # Excluindo transição de fim de semana (sex 21h → dom 21h)
    print(f"\n  CHECK 10 │ Continuidade temporal — gaps > 3h (excl. fim de semana)")
    if len(df_completo) > 1:
        diffs = pd.Series(df_completo.index[1:]) - pd.Series(df_completo.index[:-1])
        gaps_grandes = diffs[diffs > pd.Timedelta(hours=3)]

        # Filtrar gaps de fim de semana (sex → dom ou sex → seg, normalmente ~47-51h)
        gaps_nao_fds = []
        for i, gap in gaps_grandes.items():
            dt_antes = df_completo.index[i]
            dt_depois = df_completo.index[i + 1]
            # Se o gap cruza de sexta para domingo, é fim de semana normal
            if dt_antes.weekday() == 4 and dt_depois.weekday() == 6:
                continue
            # Se o gap cruza de sexta para segunda (broker sem dados de domingo)
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
        stats_remocao:         Estatísticas de remoção (filtro FOREX, OHLC, etc.)
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
    print(f"  Período coberto                       : {df_completo.index.min().strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"                                       → {df_completo.index.max().strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"  Removidos por filtro FOREX            : {stats_remocao.get('removidos_filtro_forex', 0):>12,}")
    print(f"  Removidos por OHLC zerado/nulo        : {stats_remocao.get('removidos_zerado_nulo', 0):>12,}")
    print(f"  Removidos por integridade OHLC        : {stats_remocao.get('removidos_integridade', 0):>12,}")
    print(f"  Removidos por candle H1 incompleto    : {removidos_incompletos:>12,}")

    # ═══════════════════════════════════════════════════════════════════
    # SEÇÃO 2 — Série completa (eurusd_h1_completo)
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("  SEÇÃO 2 — SÉRIE COMPLETA (eurusd_h1_completo.parquet)")
    print(sep)
    print(f"  Total de candles H1                   : {len(df_completo):>12,}")
    print(f"  Período coberto                       : {df_completo.index.min().strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"                                       → {df_completo.index.max().strftime('%Y-%m-%d %H:%M')} UTC")

    # Breakdown por dia da semana
    print(f"\n  Breakdown por dia da semana:")
    for wd in [6, 0, 1, 2, 3, 4, 5]:  # Dom, Seg, Ter, Qua, Qui, Sex, Sáb
        count = (df_completo.index.weekday == wd).sum()
        nome = DIAS_SEMANA[wd]
        extra = " (21h+)" if wd == 6 else " (até 21h)" if wd == 4 else ""
        print(f"    {nome:10s}{extra:12s}: {count:>8,} candles")

    # Confirmações
    sabados = (df_completo.index.weekday == 5).sum()
    dom_cedo = ((df_completo.index.weekday == 6) & (df_completo.index.hour < 21)).sum()
    sex_tarde = ((df_completo.index.weekday == 4) & (df_completo.index.hour > 21)).sum()
    print(f"\n  Confirmações:")
    print(f"    Candles de sábado              : {sabados:>6} {'✅' if sabados == 0 else '❌'}")
    print(f"    Candles domingo antes das 21h  : {dom_cedo:>6} {'✅' if dom_cedo == 0 else '❌'}")
    print(f"    Candles sexta após 21h         : {sex_tarde:>6} {'✅' if sex_tarde == 0 else '❌'}")

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
    # SEÇÃO 3 — Série operacional (eurusd_h1_operacional)
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("  SEÇÃO 3 — SÉRIE OPERACIONAL (eurusd_h1_operacional.parquet)")
    print(sep)
    print(f"  Total de candles H1                   : {len(df_operacional):>12,}")
    print(f"  Período coberto                       : {df_operacional.index.min().strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"                                       → {df_operacional.index.max().strftime('%Y-%m-%d %H:%M')} UTC")

    # Breakdown por hora do dia
    print(f"\n  Breakdown por hora UTC (07h-20h):")
    for h in range(7, 21):
        count = (df_operacional.index.hour == h).sum()
        print(f"    {h:02d}h UTC : {count:>8,} candles")

    # Confirmações
    fora_horario = df_operacional[
        (df_operacional.index.hour < 7) | (df_operacional.index.hour > 20)
    ]
    fds_op = df_operacional[df_operacional.index.weekday >= 5]
    sex_21_op = df_operacional[
        (df_operacional.index.weekday == 4) & (df_operacional.index.hour > 20)
    ]
    print(f"\n  Confirmações:")
    print(f"    Candles fora de 07h-20h UTC    : {len(fora_horario):>6} {'✅' if len(fora_horario) == 0 else '❌'}")
    print(f"    Candles sáb/dom                : {len(fds_op):>6} {'✅' if len(fds_op) == 0 else '❌'}")
    print(f"    Candles sexta após 20h         : {len(sex_21_op):>6} {'✅' if len(sex_21_op) == 0 else '❌'}")

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
    print(f"  % candles noturnos (só na completa)   : {pct_noturnos:>11.2f}%")
    print(f"  (candles capturados para stops/targets que a v1 ignorava)")


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
        2. Parseia datetime com timezone UTC
        3. Remove duplicatas de índice
        4. Aplica filtro de mercado FOREX (dom 21h → sex 21h)
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
    if PARQUET_COMPLETO.exists() and PARQUET_OPERACIONAL.exists() and not forcar:
        logger.info("Cache encontrado! Carregando parquets existentes...")
        logger.info(f"  → {PARQUET_COMPLETO.name}")
        logger.info(f"  → {PARQUET_OPERACIONAL.name}")
        logger.info("Use --forcar para reprocessar do CSV")

        df_completo = pd.read_parquet(PARQUET_COMPLETO, engine="pyarrow")
        df_operacional = pd.read_parquet(PARQUET_OPERACIONAL, engine="pyarrow")

        print("\n" + "═" * 70)
        print("  DADOS CARREGADOS DO CACHE")
        print("═" * 70)
        print(f"  Série completa     : {len(df_completo):>10,} candles H1")
        print(f"  Série operacional  : {len(df_operacional):>10,} candles H1")
        print(f"  Período            : {df_completo.index.min()} → {df_completo.index.max()}")
        print("═" * 70)

        return df_completo, df_operacional

    # ─── Garantir diretório de saída ───
    DIR_DATA.mkdir(parents=True, exist_ok=True)

    # ═══════════════════════════════════════════════════════════════════
    # PIPELINE DE PROCESSAMENTO
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  DATA LOADER v2 — PIPELINE DE PROCESSAMENTO")
    print("  EURUSD M1 → H1 (Série Completa 24h FOREX)")
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
    # GERAR OS DOIS PARQUETS
    # ═══════════════════════════════════════════════════════════════════

    # ARQUIVO A — Série completa (dom 21h → sex 21h, 24h)
    df_completo = df_h1.copy()

    # ARQUIVO B — Série operacional (07h-20h, seg-sex)
    df_operacional = filtro_horario_operacional(df_completo)

    # ═══════════════════════════════════════════════════════════════════
    # VERIFICAÇÕES AUTOMÁTICAS (antes de salvar)
    # ═══════════════════════════════════════════════════════════════════
    checks_ok = executar_verificacoes(df_completo, df_operacional)

    # ═══════════════════════════════════════════════════════════════════
    # PASSO 10 — SALVAR PARQUETS
    # ═══════════════════════════════════════════════════════════════════
    logger.info(f"Salvando {PARQUET_COMPLETO.name}...")
    df_completo.to_parquet(
        PARQUET_COMPLETO,
        engine="pyarrow",
        compression="snappy",
        index=True,
    )

    logger.info(f"Salvando {PARQUET_OPERACIONAL.name}...")
    df_operacional.to_parquet(
        PARQUET_OPERACIONAL,
        engine="pyarrow",
        compression="snappy",
        index=True,
    )

    # ═══════════════════════════════════════════════════════════════════
    # RELATÓRIO DETALHADO
    # ═══════════════════════════════════════════════════════════════════
    imprimir_relatorio(
        total_bruto=total_bruto,
        stats_remocao=stats_remocao,
        removidos_incompletos=removidos_incompletos,
        df_completo=df_completo,
        df_operacional=df_operacional,
    )

    # Tamanho dos arquivos
    print(f"\n{'═' * 70}")
    print("  ARQUIVOS GERADOS")
    print("═" * 70)
    for p in [PARQUET_COMPLETO, PARQUET_OPERACIONAL]:
        tamanho_mb = p.stat().st_size / (1024 * 1024)
        print(f"  {p.name:40s} : {tamanho_mb:>8.2f} MB")
    print("═" * 70)

    return df_completo, df_operacional


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
    Uso: geração de sinais de entrada (07h-20h UTC, seg-sex).

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


# =============================================================================
# EXECUÇÃO DIRETA VIA CLI
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Data Loader v2 — EURUSD M1 → H1 (Série Completa 24h FOREX)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python data_loader.py --csv ../EURUSD_10ANOS.csv
  python data_loader.py --csv ../EURUSD_10ANOS.csv --forcar
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
    print("█   DATA LOADER v2 — EURUSD M1 → H1                               █")
    print("█   Série Completa 24h FOREX (Dom 21h UTC → Sex 21h UTC)           █")
    print("█   Correção: Mantém barras noturnas para stops/targets reais      █")
    print("█" + " " * 68 + "█")
    print("█" * 70)

    df_completo, df_operacional = carregar_e_processar(
        csv_path=Path(args.csv),
        forcar=args.forcar,
    )

    print("\n✅ Pipeline concluído com sucesso!\n")
