# -*- coding: utf-8 -*-
"""
================================================================================
build_base_gbpusd.py — Construção da Base de Dados GBPUSD H1
================================================================================
Lê o CSV bruto GBPUSD_13ANOS.csv (minutal), agrega para H1,
calcula indicadores base e gera os parquets:
  - gbpusd_h1_completo.parquet              (IS: 2017-2023)
  - gbpusd_h1_completo_OOS_futuro_2024_2026.parquet
  - gbpusd_h1_completo_OOS_passado_2013_2016.parquet
  - gbpusd_h1_operacional.parquet           (IS: filtro horário)
  - gbpusd_h1_operacional_OOS_futuro_2024_2026.parquet
  - gbpusd_h1_operacional_OOS_passado_2013_2016.parquet
  - gbpusd_h1_hurst.parquet                 (Hurst pré-computado - IS completo)
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================
DIR_PROJETO = Path(__file__).resolve().parent.parent
CSV_BRUTO   = DIR_PROJETO / "db_CSV" / "GBPUSD_13ANOS.csv"
DIR_SAIDA   = DIR_PROJETO / "quant_gbpusd_h1" / "data"
DIR_SAIDA.mkdir(parents=True, exist_ok=True)

ATIVO     = "gbpusd"
TIMEFRAME = "h1"

# Períodos
IS_START  = "2017-01-01"
IS_END    = "2023-12-31 23:59:59"
OOS_FUT_START = "2024-01-01"
OOS_FUT_END   = "2026-12-31 23:59:59"
OOS_PAS_START = "2013-01-01"
OOS_PAS_END   = "2016-12-31 23:59:59"

# Janela operacional (dias úteis, 07:00-22:30)
HORA_INICIO_OP = "07:00"
HORA_FIM_OP    = "22:30"

# Hurst
HURST_JANELA = 100

# =============================================================================
# FUNÇÕES
# =============================================================================
def _quick_ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    x_mean, y_mean = x.mean(), y.mean()
    num = ((x - x_mean) * (y - y_mean)).sum()
    den = ((x - x_mean) ** 2).sum()
    return num / den if den != 0 else np.nan

def calcular_hurst_janela(retornos: np.ndarray) -> float:
    if len(retornos) < HURST_JANELA:
        return np.nan
    log_n, log_rs = [], []
    for n in [10, 20, 40, 80]:
        num_seg = HURST_JANELA // n
        rs_segs = []
        for k in range(num_seg):
            seg = retornos[k * n : (k + 1) * n]
            mu  = seg.mean()
            y_t = np.cumsum(seg - mu)
            r   = y_t.max() - y_t.min()
            s   = seg.std(ddof=1)
            if s > 0:
                rs_segs.append(r / s)
        if rs_segs:
            rs_med = np.mean(rs_segs)
            if rs_med > 0:
                log_n.append(np.log(n))
                log_rs.append(np.log(rs_med))
    if len(log_n) < 2:
        return np.nan
    h = _quick_ols_slope(np.array(log_n), np.array(log_rs))
    if np.isnan(h) or h < 0.0 or h > 1.5:
        return np.nan
    return float(h)

def calcular_hurst_serie(df: pd.DataFrame) -> np.ndarray:
    retornos = df["log_return"].fillna(0).to_numpy()
    n = len(retornos)
    hurst_vals = np.full(n, np.nan, dtype=np.float32)
    for i in tqdm(range(HURST_JANELA - 1, n), desc="Calculando Hurst"):
        hurst_vals[i] = calcular_hurst_janela(retornos[i - HURST_JANELA + 1 : i + 1])
    return hurst_vals

def filtro_operacional(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra apenas candles de dias úteis dentro da janela operacional."""
    weekday = df.index.weekday  # 0=Seg, 4=Sex
    hora    = df.index.strftime("%H:%M")
    mask    = (weekday >= 0) & (weekday <= 4) & (hora >= HORA_INICIO_OP) & (hora <= HORA_FIM_OP)
    return df[mask].copy()

def adicionar_indicadores_base(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona log_return, SMA50, STD50 ao dataframe."""
    closes = df["Close"].values
    log_ret = np.log(closes / np.roll(closes, 1))
    log_ret[0] = 0.0
    df["log_return"] = log_ret
    df["sma_50"]     = df["Close"].rolling(50, min_periods=50).mean()
    df["std_50"]     = df["Close"].rolling(50, min_periods=50).std(ddof=1)
    return df

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("BUILD BASE GBPUSD H1")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. Ler CSV bruto e agregar para H1
    # -------------------------------------------------------------------------
    print("\n[1/7] Lendo CSV bruto e agregando para H1...")
    df_raw = pd.read_csv(
        CSV_BRUTO,
        header=None,
        names=["Date", "Time", "Open", "High", "Low", "Close", "VolTick", "VolReal", "Spread"],
        dtype={"Date": str, "Time": str}
    )

    df_raw["datetime"] = pd.to_datetime(df_raw["Date"] + " " + df_raw["Time"], format="%Y.%m.%d %H:%M")
    df_raw = df_raw.set_index("datetime").sort_index()
    df_raw = df_raw[["Open", "High", "Low", "Close", "VolTick"]]

    # Agregar para H1 (OHLCV)
    df_h1 = df_raw.resample("1h").agg({
        "Open":    "first",
        "High":    "max",
        "Low":     "min",
        "Close":   "last",
        "VolTick": "sum"
    }).dropna(subset=["Open", "Close"])

    print(f"    Candles H1 totais: {len(df_h1):,} | Período: {df_h1.index[0]} → {df_h1.index[-1]}")

    # Adicionar indicadores base ao full (todos os 13 anos)
    df_full = adicionar_indicadores_base(df_h1.copy())

    # -------------------------------------------------------------------------
    # 2. Calcular Hurst no período IS completo (2017-2023) e salvar
    # -------------------------------------------------------------------------
    print("\n[2/7] Calculando Hurst no IS (2017-2023)...")
    df_is_full = df_full.loc[IS_START:IS_END].copy()
    hurst_vals = calcular_hurst_serie(df_is_full)
    df_is_full["hurst"] = hurst_vals

    hurst_path = DIR_SAIDA / f"{ATIVO}_{TIMEFRAME}_hurst.parquet"
    df_is_full[["hurst"]].to_parquet(hurst_path)
    print(f"    Hurst salvo em: {hurst_path}")

    # -------------------------------------------------------------------------
    # 3. Parquet IS Completo (2017-2023)
    # -------------------------------------------------------------------------
    print("\n[3/7] Gerando parquet IS Completo (2017-2023)...")
    out_is = DIR_SAIDA / f"{ATIVO}_{TIMEFRAME}_completo.parquet"
    df_is_full.to_parquet(out_is)
    print(f"    Salvo: {out_is} | {len(df_is_full):,} candles")

    # -------------------------------------------------------------------------
    # 4. Parquet IS Operacional (2017-2023 filtrado)
    # -------------------------------------------------------------------------
    print("\n[4/7] Gerando parquet IS Operacional...")
    df_is_op = filtro_operacional(df_is_full)
    out_op = DIR_SAIDA / f"{ATIVO}_{TIMEFRAME}_operacional.parquet"
    df_is_op.to_parquet(out_op)
    print(f"    Salvo: {out_op} | {len(df_is_op):,} candles")

    # -------------------------------------------------------------------------
    # 5. OOS Futuro (2024-2026)
    # -------------------------------------------------------------------------
    print("\n[5/7] Gerando parquets OOS Futuro (2024-2026)...")
    df_oos_fut = df_full.loc[OOS_FUT_START:OOS_FUT_END].copy()
    out_oos_fut = DIR_SAIDA / f"{ATIVO}_{TIMEFRAME}_completo_OOS_futuro_2024_2026.parquet"
    df_oos_fut.to_parquet(out_oos_fut)
    print(f"    Salvo: {out_oos_fut} | {len(df_oos_fut):,} candles")

    df_oos_fut_op = filtro_operacional(df_oos_fut)
    out_oos_fut_op = DIR_SAIDA / f"{ATIVO}_{TIMEFRAME}_operacional_OOS_futuro_2024_2026.parquet"
    df_oos_fut_op.to_parquet(out_oos_fut_op)
    print(f"    Salvo: {out_oos_fut_op} | {len(df_oos_fut_op):,} candles")

    # -------------------------------------------------------------------------
    # 6. OOS Passado (2013-2016)
    # -------------------------------------------------------------------------
    print("\n[6/7] Gerando parquets OOS Passado (2013-2016)...")
    df_oos_pas = df_full.loc[OOS_PAS_START:OOS_PAS_END].copy()
    out_oos_pas = DIR_SAIDA / f"{ATIVO}_{TIMEFRAME}_completo_OOS_passado_2013_2016.parquet"
    df_oos_pas.to_parquet(out_oos_pas)
    print(f"    Salvo: {out_oos_pas} | {len(df_oos_pas):,} candles")

    df_oos_pas_op = filtro_operacional(df_oos_pas)
    out_oos_pas_op = DIR_SAIDA / f"{ATIVO}_{TIMEFRAME}_operacional_OOS_passado_2013_2016.parquet"
    df_oos_pas_op.to_parquet(out_oos_pas_op)
    print(f"    Salvo: {out_oos_pas_op} | {len(df_oos_pas_op):,} candles")

    # -------------------------------------------------------------------------
    # 7. Resumo Final
    # -------------------------------------------------------------------------
    print("\n[7/7] CONCLUÍDO! Resumo dos arquivos gerados:")
    print("=" * 70)
    for f in sorted(DIR_SAIDA.glob("*.parquet")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:<65} {size_kb:>8.1f} KB")
    print("=" * 70)

if __name__ == "__main__":
    main()
