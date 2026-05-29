#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
distribuicao_parametros.py
--------------------------
Teste de Robustez por Distribuicao de Parametros (SQX-style).

Para cada parametro otimo, perturba +-20% em N steps e re-roda o backtest.
Avalia estabilidade via 3 criterios hibridos (% positivo, FR medio, DD max).

Estrategias suportadas:
  MOMENTUM, ZSCORE, HAWKES, OU, OU_REVERSO, PCA, WAVELET

Uso:
  python distribuicao_parametros.py
  python distribuicao_parametros.py --estrategia ZSCORE
  python distribuicao_parametros.py --estrategia HAWKES --ativo EURUSD --timeframe H1
"""

# ===================================================================
#  IMPORTS
# ===================================================================
import sys
import argparse
import warnings
from math import ceil
from pathlib import Path

# Adiciona o diretorio raiz do projeto ao sys.path para garantir que 'otimizacoes' possa ser importado
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ===================================================================
#  CONFIGURACAO DO TESTE -- EDITAR AQUI
# ===================================================================

# Identificacao
ATIVO       = "EURUSD"
TIMEFRAME   = "H1"
ESTRATEGIA  = "MOMENTUM"

# Caminho do parquet com dados completos
# Deve conter: Open, High, Low, Close, log_return, hurst
PARQUET_DADOS = Path(__file__).resolve().parent.parent.parent / \
    "quant_eurusd" / "data" / "eurusd_h1_hurst.parquet"

# Parametros otimos encontrados na otimizacao
# Se vazio, serao lidos automaticamente do parquet de otimizacao (Top1)
PARAMS_OTIMOS = {}

# Parametros operacionais fixos
CAPITAL_INICIAL    = 10_000.0
RISCO_POR_TRADE    = 0.01
SPREAD_PIPS        = 1.2
VALOR_PIP_POR_LOTE = 10.0
FATOR_PIPS         = 10_000
HORA_INICIO_OP     = "10:00"
HORA_FIM_OP        = "22:30"

# Configuracao do grid
PERTURBACAO_PCT = 0.20   # +-20%
N_STEPS         = 11     # 11 pontos, step de 4%

# Criterios de aprovacao (abordagem hibrida)
CRITERIO_1_PCT_POSITIVO = 0.80   # >= 80% dos steps com PnL > 0
CRITERIO_2_FR_MEDIO_MIN = 0.50   # FR medio > 0.5
CRITERIO_3_DD_MAX_MULT  = 2.00   # DD maximo <= 2x DD do otimo

# ===================================================================
#  FIM DA CONFIGURACAO
# ===================================================================


# ===================================================================
#  PARTE 0 -- CARREGAMENTO AUTOMATICO DOS PARAMETROS DO TOP1
# ===================================================================

def carregar_params_top1(estrategia: str) -> dict:
    """
    Le o parquet de resultados da otimizacao e retorna
    o dicionario de parametros da linha Top1 (Ret_DD mais alto).
    """
    dir_otim = Path(__file__).resolve().parent.parent.parent / \
        "quant_eurusd" / "data" / "otimizacoes"

    nome_arquivo = f"otimizacao_{estrategia.lower()}_resultados.parquet"
    caminho = dir_otim / nome_arquivo

    if not caminho.exists():
        raise FileNotFoundError(
            f"Parquet de otimizacao nao encontrado: {caminho}\n"
            f"Rode a otimizacao primeiro ou defina PARAMS_OTIMOS manualmente."
        )

    df = pd.read_parquet(caminho)
    metricas = {"Trades", "Lucro_Total_Pips", "Max_DD_Pips", "Ret_DD", "Profit_Factor"}
    param_cols = [c for c in df.columns if c not in metricas]
    top1 = df.iloc[0][param_cols].to_dict()

    # Converter numpy types para float/int nativos
    top1_clean = {}
    for k, v in top1.items():
        if isinstance(v, (np.integer,)):
            top1_clean[k] = int(v)
        elif isinstance(v, (np.floating,)):
            top1_clean[k] = float(v)
        else:
            top1_clean[k] = v

    return top1_clean


# ===================================================================
#  PARTE 1 -- GERACAO DO GRID
# ===================================================================

def gerar_grid(param_nome: str, valor_original) -> np.ndarray:
    """
    Gera um grid de valores perturbados +-PERTURBACAO_PCT
    ao redor do valor original, com N_STEPS pontos.
    Parametros inteiros (janelas) sao arredondados e deduplicados.
    """
    valor = float(valor_original)
    limite_inf = valor * (1 - PERTURBACAO_PCT)
    limite_sup = valor * (1 + PERTURBACAO_PCT)
    grid = np.linspace(limite_inf, limite_sup, N_STEPS)

    eh_inteiro = ("janela" in param_nome.lower()) or \
                 isinstance(valor_original, (int, np.integer))

    if eh_inteiro:
        grid = np.unique(np.round(grid).astype(int))

    return grid


# ===================================================================
#  PARTE 2 -- RECALCULO DE SINAIS (ROTEADOR MULTI-ESTRATEGIA)
# ===================================================================

def calcular_sinais(df: pd.DataFrame, params: dict, janela_op: np.ndarray,
                    estrategia: str, cache: dict):
    """
    Roteador que seleciona a logica de sinais correta para cada estrategia.
    As chaves usadas correspondem EXATAMENTE as chaves do PARAMS_OTIMOS
    (que vem do parquet de otimizacao).

    Args:
        df: DataFrame com OHLC indexado por datetime.
        params: Dicionario com parametros (perturbados ou originais).
        janela_op: Mascara booleana da janela operacional.
        estrategia: Nome da estrategia (ex: MOMENTUM, ZSCORE, etc.)
        cache: Dicionario mutavel para armazenar pre-calculos pesados.

    Returns:
        Tupla (sinal, sl_pips, tp_pips) como arrays numpy.
    """
    closes = df["Close"].values
    n = len(closes)

    # VR para SL/TP dinamico (comum a todas)
    if "log_return" not in df.columns:
        df["log_return"] = np.log(closes / np.roll(closes, 1))
        df["log_return"].iloc[0] = 0.0

    log_ret = df["log_return"].values

    if "vr_pips" not in cache:
        vr = pd.Series(log_ret).rolling(50, min_periods=50).std(ddof=1).values
        cache["vr_pips"] = vr * closes * FATOR_PIPS
    vr_pips = cache["vr_pips"]

    sl_pips = params["mult_sl"] * vr_pips
    tp_pips = params["mult_tp"] * vr_pips

    sinal = np.zeros(n, dtype=np.int8)

    # =================================================================
    # MOMENTUM
    # =================================================================
    if estrategia == "MOMENTUM":
        if "velocidade" not in cache:
            cache["velocidade"] = df["Close"].diff(1).values
            acel = pd.Series(cache["velocidade"]).diff(1)

            def _percentrank(arr):
                val = arr[-1]
                hist = arr[:-1]
                if len(hist) == 0: return 0.5
                return float(np.sum(hist < val)) / len(hist)

            cache["percentil"] = acel.rolling(100, min_periods=100).apply(
                _percentrank, raw=True).values

            def _entropia(arr):
                if np.std(arr) < 1e-15: return 0.0
                c, _ = np.histogram(arr, bins=10)
                p = c / len(arr)
                p = p[p > 0]
                return float(np.clip(-np.sum(p * np.log2(p)) / np.log2(10), 0, 1))

            cache["entropia"] = df["log_return"].rolling(30, min_periods=30).apply(
                _entropia, raw=True).values

        vel = cache["velocidade"]
        pct = cache["percentil"]
        ent = cache["entropia"]
        hurst = df["hurst"].values

        cond_hurst = hurst > params["hurst_cutoff"]
        cond_ent   = ent < params["entropia_cutoff"]
        cond_base  = cond_hurst & cond_ent & janela_op

        cond_long  = cond_base & (pct > params["percentil_trigger"]) & (vel > 0)
        cond_short = cond_base & (pct < (1 - params["percentil_trigger"])) & (vel < 0)
        sinal[cond_long]  =  1
        sinal[cond_short] = -1

    # =================================================================
    # ZSCORE (mean reversion)
    # =================================================================
    elif estrategia == "ZSCORE":
        j_z = int(params["janela_zscore"])
        j_v = int(params["janela_vol"])
        z_e = params["z_entry"]
        h_cut = params["hurst_cut"]

        cache_key_z = f"zscore_{j_z}"
        if cache_key_z not in cache:
            s_close = pd.Series(closes)
            roll_mean = s_close.rolling(j_z).mean().values
            roll_std = s_close.rolling(j_z).std(ddof=1).values
            roll_std = np.where(roll_std == 0, 1e-9, roll_std)
            cache[cache_key_z] = (closes - roll_mean) / roll_std

        z_array = cache[cache_key_z]
        hurst = df["hurst"].values

        z_prev = np.concatenate(([0], z_array[:-1]))
        cond_base = janela_op & (hurst < h_cut)
        cond_long  = cond_base & (z_prev <= -z_e) & (z_array > -z_e)
        cond_short = cond_base & (z_prev >= z_e) & (z_array < z_e)
        sinal[cond_long]  =  1
        sinal[cond_short] = -1

        # Recalcular VR com janela especifica
        cache_key_v = f"vr_{j_v}"
        if cache_key_v not in cache:
            vr_v = pd.Series(log_ret).rolling(j_v, min_periods=j_v).std(ddof=1).values
            cache[cache_key_v] = vr_v * closes * FATOR_PIPS
        vr_pips_v = cache[cache_key_v]
        sl_pips = params["mult_sl"] * vr_pips_v
        tp_pips = params["mult_tp"] * vr_pips_v

    # =================================================================
    # HAWKES (reversao pos-excitacao)
    # =================================================================
    elif estrategia == "HAWKES":
        if "hawkes_lambda_norm" not in cache:
            # Carregar pre-calculos pesados do Hawkes
            import math as _math
            from scipy.optimize import minimize as _minimize

            dir_data = Path(__file__).resolve().parent.parent.parent / "quant_eurusd" / "data"
            df_comp = pd.read_parquet(dir_data / "eurusd_h1_completo.parquet")
            df_op = pd.read_parquet(dir_data / "eurusd_h1_operacional.parquet")

            closes_h = df_comp["Close"].values
            n_h = len(closes_h)
            R_t = np.zeros(n_h, dtype=np.float32)
            R_t[1:] = np.log(closes_h[1:] / closes_h[:-1])
            R_t_abs = np.abs(R_t)
            std_60 = pd.Series(R_t).rolling(60).std(ddof=1).values
            threshold_dyn = 2.0 * std_60
            N_t = np.where(R_t_abs > threshold_dyn, 1.0, 0.0)
            N_t[np.isnan(N_t)] = 0.0

            def _ll_hawkes(p, N):
                mu, alpha, beta = p
                if mu <= 0 or alpha <= 0 or beta <= 0 or alpha/beta >= 1: return 1e9
                ll = 0.0; lp = mu
                for t in range(1, len(N)):
                    lt = mu + _math.exp(-beta)*(lp - mu) + alpha*N[t-1]
                    if lt <= 0: lt = 1e-9
                    ll += N[t]*_math.log(lt) - lt
                    lp = lt
                return -ll

            mu_h = np.full(n_h, np.nan, dtype=np.float32)
            al_h = np.full(n_h, np.nan, dtype=np.float32)
            be_h = np.full(n_h, np.nan, dtype=np.float32)
            val_h = np.zeros(n_h, dtype=np.int8)
            mu_c, al_c, be_c, val_c = 0.1, 0.001, 1.0, False

            for t in tqdm(range(120, n_h), desc="  Hawkes MLE Rolling"):
                if t % 24 == 0:
                    w = N_t[t-120:t]
                    res = _minimize(_ll_hawkes, [0.1, 0.5, 1.0], args=(w,),
                                    method="L-BFGS-B",
                                    bounds=((0.001,5),(0.001,5),(0.001,10)))
                    mu_c, al_c, be_c = res.x
                    val_c = res.success and (al_c/be_c < 1.0)
                mu_h[t]=mu_c; al_h[t]=al_c; be_h[t]=be_c
                val_h[t] = 1 if val_c else 0

            lambda_t = np.full(n_h, np.nan, dtype=np.float32)
            lambda_norm = np.full(n_h, np.nan, dtype=np.float32)
            excitacao = np.full(n_h, np.nan, dtype=np.float32)
            lp = 0.1
            for t in range(120, n_h):
                lc = mu_h[t] + _math.exp(-be_h[t])*(lp - mu_h[t]) + al_h[t]*N_t[t-1]
                if lc < mu_h[t]: lc = mu_h[t]
                lambda_t[t] = lc
                lambda_norm[t] = (lc - mu_h[t])/mu_h[t] if mu_h[t] > 0 else 0
                excitacao[t] = al_h[t]/be_h[t] if be_h[t] > 0 else 0
                lp = lc

            cache["hawkes_lambda_norm"] = lambda_norm
            cache["hawkes_excitacao"]   = excitacao
            cache["hawkes_valido"]      = val_h == 1
            cache["hawkes_R_t"]         = R_t
            cache["hawkes_mask_op"]     = df_comp.index.isin(df_op.index)
            cache["hawkes_closes"]      = closes_h
            cache["hawkes_highs"]       = df_comp["High"].values
            cache["hawkes_lows"]        = df_comp["Low"].values
            cache["hawkes_n"]           = n_h

        lam_norm = cache["hawkes_lambda_norm"]
        exc      = cache["hawkes_excitacao"]
        valido   = cache["hawkes_valido"]
        R_t      = cache["hawkes_R_t"]
        mask_op  = cache["hawkes_mask_op"]
        n_h      = cache["hawkes_n"]

        lam_prev = np.roll(lam_norm, 1); lam_prev[0] = np.nan
        norm_falling = lam_norm < lam_prev

        excit_ok  = exc < params["excitacao_maxima"]
        norm_high = lam_norm > params["lambda_norm_min_sinal"]
        cond_base = valido & excit_ok & mask_op & norm_falling & norm_high

        sinal_h = np.zeros(n_h, dtype=np.int8)
        sinal_h[cond_base & (R_t < 0)] =  1
        sinal_h[cond_base & (R_t > 0)] = -1

        # Retornar com dados do Hawkes (usa df_comp, nao df)
        vr_h = pd.Series(R_t).rolling(50, min_periods=50).std(ddof=1).values
        vr_pips_h = vr_h * cache["hawkes_closes"] * FATOR_PIPS
        return sinal_h, params["mult_sl"] * vr_pips_h, params["mult_tp"] * vr_pips_h

    # =================================================================
    # OU (mean reversion - Ornstein-Uhlenbeck)
    # =================================================================
    elif estrategia == "OU":
        j_ou = int(params["janela_ou"])
        cache_key = f"ou_{j_ou}"
        if cache_key not in cache:
            from otimizacoes.otimizacao_ou import calcular_ou_rolling
            dir_data = Path(__file__).resolve().parent.parent.parent / "quant_eurusd" / "data"
            df_comp = pd.read_parquet(dir_data / "eurusd_h1_completo.parquet")
            z, hl, val = calcular_ou_rolling(df_comp, j_ou)
            cache[cache_key] = (z, hl, val)
            if "ou_mask_op" not in cache:
                df_op = pd.read_parquet(dir_data / "eurusd_h1_operacional.parquet")
                cache["ou_mask_op"] = df_comp.index.isin(df_op.index)
                cache["ou_closes"] = df_comp["Close"].values
                cache["ou_highs"]  = df_comp["High"].values
                cache["ou_lows"]   = df_comp["Low"].values
                cache["ou_n"]      = len(df_comp)
                lr = np.log(cache["ou_closes"] / np.roll(cache["ou_closes"], 1))
                lr[0] = 0.0
                vr_ou = pd.Series(lr).rolling(50, min_periods=50).std(ddof=1).values
                cache["ou_vr_pips"] = vr_ou * cache["ou_closes"] * FATOR_PIPS

        z, hl, val = cache[cache_key]
        mask_op = cache["ou_mask_op"]
        n_ou = cache["ou_n"]

        cond_op = val & (hl >= 1.0) & (hl <= params["halflife_max"]) & mask_op
        cond_buy  = cond_op & (z <= -params["zscore_threshold"])
        cond_sell = cond_op & (z >= params["zscore_threshold"])

        sinal_ou = np.zeros(n_ou, dtype=np.int8)
        sinal_ou[cond_buy]  =  1
        sinal_ou[cond_sell] = -1
        vr_ou = cache["ou_vr_pips"]
        return sinal_ou, params["mult_sl"] * vr_ou, params["mult_tp"] * vr_ou

    # =================================================================
    # OU_REVERSO (breakout - Ornstein-Uhlenbeck invertido)
    # =================================================================
    elif estrategia == "OU_REVERSO":
        j_ou = int(params["janela_ou"])
        cache_key = f"our_{j_ou}"
        if cache_key not in cache:
            from otimizacoes.otimizacao_ou_reverso import calcular_ou_rolling
            dir_data = Path(__file__).resolve().parent.parent.parent / "quant_eurusd" / "data"
            df_comp = pd.read_parquet(dir_data / "eurusd_h1_completo.parquet")
            z, hl, val = calcular_ou_rolling(df_comp, j_ou)
            cache[cache_key] = (z, hl, val)
            if "our_mask_op" not in cache:
                df_op = pd.read_parquet(dir_data / "eurusd_h1_operacional.parquet")
                cache["our_mask_op"] = df_comp.index.isin(df_op.index)
                cache["our_closes"] = df_comp["Close"].values
                cache["our_highs"]  = df_comp["High"].values
                cache["our_lows"]   = df_comp["Low"].values
                cache["our_n"]      = len(df_comp)
                lr = np.log(cache["our_closes"] / np.roll(cache["our_closes"], 1))
                lr[0] = 0.0
                vr_r = pd.Series(lr).rolling(50, min_periods=50).std(ddof=1).values
                cache["our_vr_pips"] = vr_r * cache["our_closes"] * FATOR_PIPS

        z, hl, val = cache[cache_key]
        mask_op = cache["our_mask_op"]
        n_our = cache["our_n"]

        cond_op = val & (hl >= 1.0) & (hl <= params["halflife_max"]) & mask_op
        # REVERSO: compra se Z >= threshold, vende se Z <= -threshold
        cond_buy  = cond_op & (z >= params["zscore_threshold"])
        cond_sell = cond_op & (z <= -params["zscore_threshold"])

        sinal_our = np.zeros(n_our, dtype=np.int8)
        sinal_our[cond_buy]  =  1
        sinal_our[cond_sell] = -1
        vr_r = cache["our_vr_pips"]
        return sinal_our, params["mult_sl"] * vr_r, params["mult_tp"] * vr_r

    # =================================================================
    # PCA (mean reversion via componente principal)
    # =================================================================
    elif estrategia == "PCA":
        j_pca = int(params["janela_pca"])
        cache_key = f"pca_{j_pca}"
        if cache_key not in cache:
            from otimizacoes.otimizacao_pca import calcular_pca_rolante
            dir_data = Path(__file__).resolve().parent.parent.parent / "quant_eurusd" / "data"
            df_comp = pd.read_parquet(dir_data / "eurusd_h1_completo.parquet")
            cl = df_comp["Close"].values
            lr_p = np.log(cl / np.roll(cl, 1)); lr_p[0] = 0.0
            lr_sq = lr_p ** 2; lr_abs = np.abs(lr_p)
            lr_lag1 = np.roll(lr_p, 1); lr_lag1[0] = 0.0
            lr_lag2 = np.roll(lr_p, 2); lr_lag2[:2] = 0.0
            features = np.column_stack([lr_p, lr_sq, lr_abs, lr_lag1, lr_lag2])
            pca_z, pca_dom = calcular_pca_rolante(features, j_pca)
            cache[cache_key] = (pca_z, pca_dom)
            if "pca_mask_op" not in cache:
                df_op = pd.read_parquet(dir_data / "eurusd_h1_operacional.parquet")
                cache["pca_mask_op"] = df_comp.index.isin(df_op.index)
                cache["pca_closes"] = cl
                cache["pca_n"] = len(df_comp)
                vr_p = pd.Series(lr_p).rolling(50, min_periods=50).std(ddof=1).values
                cache["pca_vr_pips"] = vr_p * cl * FATOR_PIPS

        pca_z, pca_dom = cache[cache_key]
        mask_op = cache["pca_mask_op"]
        n_pca = cache["pca_n"]

        cond_base = (pca_dom >= params["dominancia_minima"]) & mask_op
        cond_buy  = cond_base & (pca_z <= -params["zscore_threshold"])
        cond_sell = cond_base & (pca_z >= params["zscore_threshold"])

        sinal_pca = np.zeros(n_pca, dtype=np.int8)
        sinal_pca[cond_buy]  =  1
        sinal_pca[cond_sell] = -1
        vr_p = cache["pca_vr_pips"]
        return sinal_pca, params["mult_sl"] * vr_p, params["mult_tp"] * vr_p

    # =================================================================
    # WAVELET (CWT inflexion reversal)
    # =================================================================
    elif estrategia == "WAVELET":
        if "wav_cache_ready" not in cache:
            import pywt
            dir_data = Path(__file__).resolve().parent.parent.parent / "quant_eurusd" / "data"
            df_comp = pd.read_parquet(dir_data / "eurusd_h1_completo.parquet")
            df_op = pd.read_parquet(dir_data / "eurusd_h1_operacional.parquet")

            cl = df_comp["Close"].values
            lr_w = np.log(cl / np.roll(cl, 1)); lr_w[0] = 0.0
            n_w = len(cl)

            ESCALAS = [4, 12, 24, 120]
            coefs, _ = pywt.cwt(lr_w, ESCALAS, 'cmor1.5-1.0')
            potencia = np.abs(coefs)**2
            pot_norm = potencia / (np.sum(potencia, axis=0) + 1e-15)
            fase = np.angle(coefs)

            wav_energia_total = np.sum(potencia, axis=0)
            fase_unwrapped = np.unwrap(fase, axis=1)
            d_fase_all = np.zeros_like(fase_unwrapped)
            d_fase_all[:, 1:] = np.diff(fase_unwrapped, axis=1)

            wav_dom_escala = np.argmax(potencia, axis=0)
            d_fase = d_fase_all[wav_dom_escala, np.arange(n_w)]
            d_fase_prev = np.roll(d_fase, 1); d_fase_prev[0] = 0

            cache["wav_inflex_up"] = (d_fase > 0) & (d_fase_prev <= 0)
            cache["wav_inflex_dn"] = (d_fase < 0) & (d_fase_prev >= 0)
            cache["wav_ret_neg"] = lr_w < 0
            cache["wav_ret_pos"] = lr_w > 0
            cache["wav_pot_norm_s1"] = pot_norm[0]
            cache["wav_energia_total"] = wav_energia_total

            cross12 = coefs[0] * np.conj(coefs[1])
            cache["wav_cross12_real"] = np.real(cross12)
            cache["wav_cross12_imag"] = np.imag(cross12)
            cache["wav_S11_raw"] = potencia[0]
            cache["wav_S22_raw"] = potencia[1]

            cache["wav_mask_op"] = df_comp.index.isin(df_op.index)
            cache["wav_closes"]  = cl
            cache["wav_n"]       = n_w
            vr_w = pd.Series(lr_w).rolling(50, min_periods=50).std(ddof=1).values
            cache["wav_vr_pips"] = vr_w * cl * FATOR_PIPS

            cache["wav_cache_ready"] = True

        js = int(params["janela_smooth"])
        je = int(params["janela_energia"])
        eq = params["energia_quantile"]
        cc = params["coerencia_cutoff"]
        ps1 = params["pot_s1_cutoff"]

        # Coerencia smoothed
        cache_key_s = f"wav_coer_{js}"
        if cache_key_s not in cache:
            S12_r = pd.Series(cache["wav_cross12_real"]).rolling(js, min_periods=1).mean().values
            S12_i = pd.Series(cache["wav_cross12_imag"]).rolling(js, min_periods=1).mean().values
            S11 = pd.Series(cache["wav_S11_raw"]).rolling(js, min_periods=1).mean().values
            S22 = pd.Series(cache["wav_S22_raw"]).rolling(js, min_periods=1).mean().values
            cache[cache_key_s] = (S12_r**2 + S12_i**2) / (S11 * S22 + 1e-15)
        coerencia_12 = cache[cache_key_s]

        cache_key_e = f"wav_enrg_{je}_{eq}"
        if cache_key_e not in cache:
            thresh = pd.Series(cache["wav_energia_total"]).rolling(je, min_periods=1).quantile(eq).values
            cache[cache_key_e] = cache["wav_energia_total"] > thresh
        energ_ok = cache[cache_key_e]

        mask_op = cache["wav_mask_op"]
        n_w = cache["wav_n"]

        coer_ok = coerencia_12 >= cc
        pot_s1_ok = cache["wav_pot_norm_s1"] < ps1

        cond_buy  = mask_op & coer_ok & energ_ok & cache["wav_inflex_up"] & pot_s1_ok & cache["wav_ret_neg"]
        cond_sell = mask_op & coer_ok & energ_ok & cache["wav_inflex_dn"] & pot_s1_ok & cache["wav_ret_pos"]

        sinal_w = np.zeros(n_w, dtype=np.int8)
        sinal_w[cond_buy]  =  1
        sinal_w[cond_sell] = -1
        vr_w = cache["wav_vr_pips"]
        return sinal_w, params["mult_sl"] * vr_w, params["mult_tp"] * vr_w

    else:
        raise ValueError(f"Estrategia '{estrategia}' nao suportada.")

    return sinal, sl_pips, tp_pips


# ===================================================================
#  PARTE 3 -- BACKTEST COMPLETO
# ===================================================================

def rodar_backtest(df: pd.DataFrame, sinal: np.ndarray,
                   sl_pips: np.ndarray, tp_pips: np.ndarray,
                   estrategia: str = None, cache: dict = None) -> dict:
    """
    Backtest candle a candle com position sizing dinamico.
    Para estrategias que usam df_comp (Hawkes, OU, PCA, Wavelet),
    os arrays de preco sao buscados do cache.
    """
    # Selecionar arrays de preco corretos
    usa_df_comp = estrategia in ("HAWKES", "OU", "OU_REVERSO", "PCA", "WAVELET")

    if usa_df_comp and cache:
        prefix_map = {
            "HAWKES": "hawkes", "OU": "ou", "OU_REVERSO": "our",
            "PCA": "pca", "WAVELET": "wav"
        }
        pfx = prefix_map[estrategia]
        opens  = cache.get(f"{pfx}_closes", df["Close"].values)  # Usa close como proxy
        highs  = cache.get(f"{pfx}_highs", df["High"].values)
        lows   = cache.get(f"{pfx}_lows", df["Low"].values)
        closes = cache.get(f"{pfx}_closes", df["Close"].values)
        n = cache.get(f"{pfx}_n", len(df))
        # Usar index do df_comp para weekday
        dir_data = Path(__file__).resolve().parent.parent.parent / "quant_eurusd" / "data"
        df_idx = pd.read_parquet(dir_data / "eurusd_h1_completo.parquet", columns=["Close"]).index
    else:
        opens  = df["Open"].values
        highs  = df["High"].values
        lows   = df["Low"].values
        closes = df["Close"].values
        n = len(df)
        df_idx = df.index

    capital = CAPITAL_INICIAL
    equity  = [capital]
    trades  = []
    posicao = None

    for t in range(n):
        dt = df_idx[t]
        weekday = dt.weekday()
        hora    = dt.strftime("%H:%M")
        bloqueio = (
            (weekday == 4 and hora >= "20:30") or
            weekday == 5 or
            (weekday == 6 and hora < "21:00")
        )
        eh_sexta_fechamento = (weekday == 4 and hora == "21:55")

        if posicao is not None:
            if eh_sexta_fechamento:
                saida = closes[t]; motivo = "FDS"
            else:
                saida, motivo = None, None
                if posicao["direcao"] == 1:
                    if lows[t] <= posicao["sl"] and highs[t] >= posicao["tp"]:
                        saida, motivo = posicao["sl"], "SL"
                    elif lows[t] <= posicao["sl"]:
                        saida, motivo = posicao["sl"], "SL"
                    elif highs[t] >= posicao["tp"]:
                        saida, motivo = posicao["tp"], "TP"
                else:
                    if highs[t] >= posicao["sl"] and lows[t] <= posicao["tp"]:
                        saida, motivo = posicao["sl"], "SL"
                    elif highs[t] >= posicao["sl"]:
                        saida, motivo = posicao["sl"], "SL"
                    elif lows[t] <= posicao["tp"]:
                        saida, motivo = posicao["tp"], "TP"

            if saida is not None:
                d = posicao["direcao"]
                pnl_pips = d * (saida - posicao["entrada"]) * FATOR_PIPS
                pnl_usd  = (pnl_pips * posicao["lote"] * VALOR_PIP_POR_LOTE
                            - SPREAD_PIPS * posicao["lote"] * VALOR_PIP_POR_LOTE)
                capital += pnl_usd
                trades.append(pnl_usd)
                posicao = None

        if (posicao is None and not bloqueio and
                sinal[t] != 0 and t + 1 < n):
            sl_p = sl_pips[t]
            tp_p = tp_pips[t]
            if (not np.isfinite(sl_p) or sl_p <= 0 or
                    not np.isfinite(tp_p) or tp_p <= 0):
                equity.append(capital)
                continue

            entrada = closes[t]  # Entrada no close do sinal
            lote = float(np.clip(
                (capital * RISCO_POR_TRADE) / (sl_p * VALOR_PIP_POR_LOTE),
                0.01, 100.0))
            delta_sl = sl_p / FATOR_PIPS
            delta_tp = tp_p / FATOR_PIPS

            if sinal[t] == 1:
                sl_abs = entrada - delta_sl
                tp_abs = entrada + delta_tp
            else:
                sl_abs = entrada + delta_sl
                tp_abs = entrada - delta_tp

            posicao = {"direcao": int(sinal[t]), "entrada": entrada,
                       "sl": sl_abs, "tp": tp_abs, "lote": lote}

        equity.append(capital)

    # -- Metricas --
    equity_arr = np.array(equity)
    pnl_pct = (capital - CAPITAL_INICIAL) / CAPITAL_INICIAL * 100
    total   = len(trades)
    ganhos  = [t for t in trades if t > 0]
    perdas  = [t for t in trades if t <= 0]
    win_rate = len(ganhos) / total * 100 if total > 0 else 0.0

    soma_g = sum(ganhos) if ganhos else 0.0
    soma_p = abs(sum(perdas)) if perdas else 1e-9
    fator_lucro = soma_g / soma_p if soma_p > 0 else 0.0

    pico   = np.maximum.accumulate(equity_arr)
    dd_abs = equity_arr - pico
    dd_pct_arr = dd_abs / np.where(pico > 0, pico, 1) * 100
    dd_pct = float(dd_pct_arr.min())
    dd_usd = float(dd_abs.min())

    pnl_usd = capital - CAPITAL_INICIAL
    fr = pnl_usd / abs(dd_usd) if dd_usd < 0 else (pnl_usd if pnl_usd > 0 else 0.0)

    # Sharpe blindado
    if len(equity_arr) > 2:
        retornos = np.diff(equity_arr) / np.where(equity_arr[:-1] != 0, equity_arr[:-1], 1)
        retornos = retornos[np.isfinite(retornos)]
        std_ret = np.std(retornos) if len(retornos) > 1 else 0.0
        sharpe = float(np.mean(retornos) / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0
    else:
        sharpe = 0.0

    return {
        "pnl_pct":       round(pnl_pct, 2),
        "win_rate":       round(win_rate, 2),
        "fr":             round(fr, 3),
        "dd_pct":         round(dd_pct, 2),
        "sharpe":         round(sharpe, 4),
        "fator_lucro":    round(fator_lucro, 3),
        "total_trades":   total,
        "positivo":       pnl_pct > 0,
        "capital_final":  round(capital, 2),
    }


# ===================================================================
#  PARTE 4 -- CRITERIOS HIBRIDOS
# ===================================================================

def avaliar_criterios(resultados_steps: list, dd_referencia: float) -> dict:
    """Avalia C1 (% positivo), C2 (FR medio), C3 (DD max)."""
    n_r = len(resultados_steps)
    if n_r == 0:
        return {"passou": False, "c1_valor": 0, "c1_passou": False,
                "c2_valor": 0, "c2_passou": False,
                "c3_valor": 0, "c3_limite": 100, "c3_passou": False,
                "n_positivos": 0, "n_total": 0}

    n_pos = sum(1 for r in resultados_steps if r["positivo"])
    pct_pos = n_pos / n_r
    passou_c1 = pct_pos >= CRITERIO_1_PCT_POSITIVO

    fr_medio = float(np.mean([r["fr"] for r in resultados_steps]))
    passou_c2 = fr_medio >= CRITERIO_2_FR_MEDIO_MIN

    dd_max = max(abs(r["dd_pct"]) for r in resultados_steps)
    limite_dd = abs(dd_referencia) * CRITERIO_3_DD_MAX_MULT if dd_referencia != 0 else 100.0
    passou_c3 = dd_max <= limite_dd

    return {
        "passou": passou_c1 and passou_c2 and passou_c3,
        "c1_valor": pct_pos, "c1_passou": passou_c1,
        "c2_valor": fr_medio, "c2_passou": passou_c2,
        "c3_valor": dd_max, "c3_limite": limite_dd, "c3_passou": passou_c3,
        "n_positivos": n_pos, "n_total": n_r,
    }


# ===================================================================
#  PARTE 5 -- GRAFICO (dark mode premium)
# ===================================================================

def gerar_grafico(resultados_por_param: dict, passou_tudo: bool,
                  estrategia: str, ativo: str, timeframe: str,
                  dir_saida: Path):
    """Dashboard visual: barras PnL% + linha FR por parametro."""
    plt.rcParams.update({
        "figure.facecolor": "#0D1117", "axes.facecolor": "#0D1117",
        "axes.edgecolor": "#21262D", "axes.labelcolor": "#E6EDF3",
        "xtick.color": "#E6EDF3", "ytick.color": "#E6EDF3",
        "text.color": "#E6EDF3", "grid.color": "#21262D",
        "grid.linestyle": "--", "grid.alpha": 0.4, "font.family": "monospace",
    })

    n_params = len(resultados_por_param)
    cols = min(3, n_params)
    rows = ceil(n_params / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows))

    if n_params == 1:   axes = np.array([[axes]])
    elif rows == 1:     axes = axes.reshape(1, -1)
    elif cols == 1:     axes = axes.reshape(-1, 1)

    step_size = PERTURBACAO_PCT * 2 / (N_STEPS - 1) * 100 if N_STEPS > 1 else 0
    fig.suptitle(
        f"Distribuicao de Parametros -- {estrategia} | {ativo} {timeframe}\n"
        f"Perturbacao: +-{PERTURBACAO_PCT*100:.0f}% | "
        f"{N_STEPS} steps | Step size: {step_size:.1f}%",
        fontsize=14, fontweight="bold", color="#E6EDF3", y=0.98)

    veredito_txt = "SISTEMA ROBUSTO" if passou_tudo else "SISTEMA FRAGIL"
    veredito_bg  = "#238636" if passou_tudo else "#DA3633"
    fig.text(0.98, 0.98, f"  {veredito_txt}  ", ha="right", va="top",
             fontsize=12, fontweight="bold", color="white",
             bbox=dict(boxstyle="round,pad=0.4", facecolor=veredito_bg,
                       edgecolor="none", alpha=0.95))

    for idx, (pname, dados) in enumerate(resultados_por_param.items()):
        r, c = idx // cols, idx % cols
        ax = axes[r, c]

        grid = dados["grid"]; res_list = dados["resultados"]
        crit = dados["criterios"]; idx_ot = dados["idx_otimo"]
        passou = crit["passou"]; val_orig = dados["valor_original"]

        x = np.arange(len(grid))
        pnl_vals = [r["pnl_pct"] for r in res_list]
        fr_vals  = [r["fr"] for r in res_list]

        eh_int = ("janela" in pname.lower()) or isinstance(val_orig, (int, np.integer))
        cores = []
        for i, res in enumerate(res_list):
            if not res["positivo"]:     cores.append("#F85149")
            elif passou:                cores.append("#3FB950")
            else:                       cores.append("#1F6FEB")

        ax.bar(x, pnl_vals, color=cores, edgecolor="#0D1117", width=0.8, zorder=3)
        if 0 <= idx_ot < len(pnl_vals):
            ax.plot(idx_ot, pnl_vals[idx_ot], marker="*", color="white",
                    markersize=14, zorder=5)
        ax.axhline(y=0, color="#484F58", linestyle="--", linewidth=0.8, zorder=2)
        ax.set_ylabel("PnL %", fontsize=9); ax.grid(axis="y", alpha=0.3)

        ax2 = ax.twinx()
        ax2.plot(x, fr_vals, color="#F0A500", linestyle=":", linewidth=1.5,
                 marker="o", markersize=4, zorder=4)
        ax2.set_ylabel("FR", fontsize=9, color="#F0A500")
        ax2.tick_params(axis="y", labelcolor="#F0A500")

        labels = [f"{int(v)}" if eh_int else f"{v:.3f}" for v in grid]
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)

        status_txt = "STABLE [OK]" if passou else "FAILED [X]"
        status_cor = "#3FB950" if passou else "#F85149"
        ax.set_title(pname, fontsize=11, fontweight="bold", color="#E6EDF3", pad=12)
        ax.text(0.5, 1.02, status_txt, transform=ax.transAxes,
                ha="center", fontsize=9, fontweight="bold", color=status_cor)

        c1i = "[OK]" if crit["c1_passou"] else "[X]"
        c2i = "[OK]" if crit["c2_passou"] else "[X]"
        c3i = "[OK]" if crit["c3_passou"] else "[X]"
        info = (f"C1: {crit['c1_valor']*100:.0f}% pos {c1i}\n"
                f"C2: FR={crit['c2_valor']:.2f} {c2i}\n"
                f"C3: DD={crit['c3_valor']:.1f}% {c3i}")
        ax.text(0.02, 0.02, info, transform=ax.transAxes, fontsize=8,
                color="#8B949E", va="bottom", family="monospace")

    for idx in range(n_params, rows * cols):
        axes[idx // cols, idx % cols].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    caminho_img = dir_saida / f"resultado_distribuicao_{estrategia.lower()}.png"
    plt.savefig(caminho_img, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[GRAFICO] Salvo em: {caminho_img}")


# ===================================================================
#  PARTE 6 -- RELATORIO NO TERMINAL
# ===================================================================

def imprimir_relatorio(resultados_por_param: dict, passou_tudo: bool,
                       estrategia: str, ativo: str, timeframe: str):
    for pname, dados in resultados_por_param.items():
        grid = dados["grid"]; res_list = dados["resultados"]
        crit = dados["criterios"]; idx_ot = dados["idx_otimo"]
        val_orig = dados["valor_original"]
        eh_int = ("janela" in pname.lower()) or isinstance(val_orig, (int, np.integer))

        print(f"\n  ===============================================")
        val_str = f"{val_orig:.0f}" if eh_int else f"{val_orig:.3f}"
        print(f"  Parametro: {pname} | Otimo: {val_str}")
        print(f"  -----------------------------------------------")
        print(f"  {'Step':>6}  {'Valor':>10}  {'PnL%':>8}  {'FR':>7}  {'DD%':>8}  Status")

        ns = len(grid)
        for i, (val, res) in enumerate(zip(grid, res_list)):
            s = f"{i+1:02d}/{ns}"
            v = f"{int(val):>10}" if eh_int else f"{val:>10.3f}"
            p = f"{res['pnl_pct']:>+7.1f}%"
            f_ = f"{res['fr']:>7.2f}"
            d = f"{res['dd_pct']:>7.1f}%"
            st = "* OTIMO" if i == idx_ot else ("[OK]" if res["positivo"] else "[X]")
            print(f"  {s}  {v}  {p}  {f_}  {d}   {st}")

        print(f"  -----------------------------------------------")
        c = crit
        c1i = "[OK]" if c["c1_passou"] else "[X]"
        c2i = "[OK]" if c["c2_passou"] else "[X]"
        c3i = "[OK]" if c["c3_passou"] else "[X]"
        print(f"  C1 Steps positivos : {c['n_positivos']}/{c['n_total']} ({c['c1_valor']*100:.1f}%)  {c1i} (min: {CRITERIO_1_PCT_POSITIVO*100:.0f}%)")
        print(f"  C2 FR medio        : {c['c2_valor']:.2f}x         {c2i} (min: {CRITERIO_2_FR_MEDIO_MIN:.2f})")
        print(f"  C3 DD maximo grid  : -{c['c3_valor']:.1f}%        {c3i} (lim: -{c['c3_limite']:.1f}%)")
        rt = "PASSOU [OK]" if c["passou"] else "REPROVADO [X]"
        print(f"  RESULTADO: {rt}")
        print(f"  ===============================================")

    total_p = len(resultados_por_param)
    aprov = sum(1 for d in resultados_por_param.values() if d["criterios"]["passou"])
    reprov = total_p - aprov
    vered = "SISTEMA ROBUSTO [OK]" if passou_tudo else "SISTEMA FRAGIL [X]"

    print(f"\n  +=============================================+")
    print(f"  |  DISTRIBUICAO DE PARAMETROS                 |")
    print(f"  |  Estrategia : {estrategia:<31s}|")
    print(f"  |  Ativo      : {ativo} {timeframe:<26s}|")
    print(f"  +---------------------------------------------+")
    print(f"  |  Parametros testados   : {total_p:<20d}|")
    print(f"  |  Aprovados             : {aprov:<20d}|")
    print(f"  |  Reprovados            : {reprov:<20d}|")
    print(f"  +---------------------------------------------+")
    print(f"  |  VEREDITO: {vered:<34s}|")
    print(f"  +=============================================+")


# ===================================================================
#  PARTE 7 -- CSV DE SAIDA
# ===================================================================

def salvar_csv(resultados_por_param: dict, estrategia: str, dir_saida: Path):
    linhas = []
    for pname, dados in resultados_por_param.items():
        for i, (val, res) in enumerate(zip(dados["grid"], dados["resultados"])):
            linhas.append({
                "parametro": pname, "step": i+1,
                "valor_grid": round(float(val), 6),
                "pnl_pct": res["pnl_pct"], "win_rate": res["win_rate"],
                "fr": res["fr"], "dd_pct": res["dd_pct"],
                "sharpe": res["sharpe"], "fator_lucro": res["fator_lucro"],
                "total_trades": res["total_trades"], "positivo": res["positivo"],
                "passou_c1": dados["criterios"]["c1_passou"],
                "passou_c2": dados["criterios"]["c2_passou"],
                "passou_c3": dados["criterios"]["c3_passou"],
                "passou_parametro": dados["criterios"]["passou"],
                "valor_otimo_original": dados["valor_original"],
            })
    df_csv = pd.DataFrame(linhas)
    caminho = dir_saida / f"resultado_distribuicao_{estrategia.lower()}.csv"
    df_csv.to_csv(caminho, index=False)
    print(f"[CSV] Salvo em: {caminho}")


# ===================================================================
#  PARTE 8 -- TABELA GERAL RESUMO DE ROBUSTEZ (FUSAO FÁCIL)
# ===================================================================

def gerar_tabela_robustez(dir_res: Path, ativo: str, timeframe: str):
    import os
    import glob
    csv_files = glob.glob(os.path.join(dir_res, "resultado_distribuicao_*.csv"))
    
    registros = []
    
    for caminho in csv_files:
        nome_arquivo = os.path.basename(caminho)
        estrategia = nome_arquivo.replace("resultado_distribuicao_", "").replace(".csv", "").upper()
        
        df = pd.read_csv(caminho)
        
        df_params = df.groupby("parametro")["passou_parametro"].first().reset_index()
        total_params = len(df_params)
        aprovados = int(df_params["passou_parametro"].sum())
        reprovados = total_params - aprovados
        
        pct_aprovados = aprovados / total_params if total_params > 0 else 0.0
        todos_positivos = bool((df["pnl_pct"] >= 0.0).all())
        
        passou_tudo = (pct_aprovados >= 0.80) and todos_positivos
        veredito = "ROBUSTO" if passou_tudo else "FRÁGIL"
        
        params_ok = df_params[df_params["passou_parametro"] == True]["parametro"].tolist()
        params_nok = df_params[df_params["passou_parametro"] == False]["parametro"].tolist()
        
        str_ok = ", ".join([p.replace("janela_", "").replace("mult_", "") for p in params_ok])
        str_nok = ", ".join([p.replace("janela_", "").replace("mult_", "") for p in params_nok])
        
        if not str_ok: str_ok = "-"
        if not str_nok: str_nok = "-"
        
        registros.append({
            "Estratégia": estrategia,
            "Total Params": total_params,
            "Aprovados": aprovados,
            "Reprovados": reprovados,
            "Parâmetros OK": str_ok,
            "Parâmetros Falhos": str_nok,
            "Veredito": veredito,
            "_sort_aprov": aprovados / total_params if total_params > 0 else 0
        })
        
    if not registros:
        return
        
    df_resumo = pd.DataFrame(registros)
    df_resumo = df_resumo.sort_values(by="_sort_aprov", ascending=False).drop(columns=["_sort_aprov"])
    
    # Plotar tabela dark premium
    plt.rcParams.update({
        "font.family": "monospace",
    })
    
    fig, ax = plt.subplots(figsize=(15, len(df_resumo)/1.5 + 2.5), dpi=150)
    fig.patch.set_facecolor('#0D1117')
    ax.set_facecolor('#0D1117')
    ax.axis("off")
    
    cores_celulas = []
    for i in range(len(df_resumo)):
        linha = []
        for j in range(len(df_resumo.columns)):
            linha.append("#0D1117")
        cores_celulas.append(linha)
        
    tabela = ax.table(
        cellText=df_resumo.values,
        colLabels=df_resumo.columns,
        loc="center",
        cellLoc="center",
        cellColours=cores_celulas
    )
    
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(10)
    tabela.scale(1.2, 2.2)
    
    col_widths = {0: 0.12, 1: 0.08, 2: 0.08, 3: 0.08, 4: 0.28, 5: 0.28, 6: 0.08}
    for col_idx, width in col_widths.items():
        for row_idx in range(len(df_resumo) + 1):
            if (row_idx, col_idx) in tabela.get_celld():
                tabela.get_celld()[(row_idx, col_idx)].set_width(width)
    
    for (row, col), cell in tabela.get_celld().items():
        cell.set_edgecolor("#21262D")
        if row == 0:
            cell.set_text_props(weight="bold", color="#FFFFFF", fontsize=11)
            cell.set_facecolor("#1F4F8A")
        else:
            val_celula = cell.get_text().get_text()
            cell.set_text_props(color="#CFD8DC")
            cell.set_facecolor("#0D1117" if row % 2 == 0 else "#161B22")
            if col == 0:
                cell.set_text_props(weight="bold", color="#58A6FF")
            elif col == 6:
                if val_celula == "ROBUSTO":
                    cell.set_text_props(weight="bold", color="#56D364")
                    cell.set_facecolor("#14321A")
                else:
                    cell.set_text_props(weight="bold", color="#F85149")
                    cell.set_facecolor("#2E1414")
                    
    fig.suptitle(f"Relatório Oficial de Robustez Paramétrica (Neighborhood SQX)\n{ativo.upper()} {timeframe.upper()} | Filtro: Estável em 7 passos com DD < 30%", 
                 color="#E6EDF3", fontsize=13, fontweight="bold", y=0.98)
    
    plt.tight_layout()
    caminho_img = os.path.join(dir_res, "tabela_robustez_parametros.png")
    plt.savefig(caminho_img, dpi=180, facecolor="#0D1117", bbox_inches="tight")
    plt.close()
    
    print(f"\n[SUCESSO] Tabela resumo parametrizada gerada e salva em: {caminho_img}\n")


# ===================================================================
#  MAIN -- ORQUESTRADOR
# ===================================================================

def main():
    global ATIVO, TIMEFRAME, ESTRATEGIA, PARAMS_OTIMOS

    parser = argparse.ArgumentParser(description="Teste de Robustez -- Distribuicao de Parametros")
    parser.add_argument("--estrategia", type=str, default=None)
    parser.add_argument("--ativo", type=str, default=None)
    parser.add_argument("--timeframe", type=str, default=None)
    args = parser.parse_args()

    if args.estrategia: ESTRATEGIA = args.estrategia.upper()
    if args.ativo:      ATIVO = args.ativo.upper()
    if args.timeframe:  TIMEFRAME = args.timeframe.upper()

    # DIR_SAIDA recalculado APOS parsing do CLI
    dir_saida = Path(__file__).resolve().parent / f"{ATIVO.lower()}_{TIMEFRAME.lower()}"
    dir_saida.mkdir(parents=True, exist_ok=True)

    # Carregar parametros automaticamente se nao definidos
    if not PARAMS_OTIMOS:
        print(f"[AUTO] Carregando Top1 do parquet de otimizacao para {ESTRATEGIA}...")
        PARAMS_OTIMOS.update(carregar_params_top1(ESTRATEGIA))

    print(f"\n{'='*55}")
    print(f"  TESTE DE ROBUSTEZ -- DISTRIBUICAO DE PARAMETROS")
    print(f"  Estrategia : {ESTRATEGIA}")
    print(f"  Ativo      : {ATIVO} {TIMEFRAME}")
    print(f"  Saida      : {dir_saida}")
    print(f"  Params     : {PARAMS_OTIMOS}")
    print(f"{'='*55}\n")

    # Carregar dados base
    print("[DADOS] Carregando serie historica...")
    df = pd.read_parquet(PARQUET_DADOS)
    if not isinstance(df.index, pd.DatetimeIndex):
        for col in ("time", "datetime"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                df = df.set_index(col)
                break
    if "log_return" not in df.columns:
        df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    print(f"   {len(df):,} candles ({df.index.min()} -> {df.index.max()})\n")

    horas = df.index.strftime("%H:%M")
    janela_op = (horas >= HORA_INICIO_OP) & (horas <= HORA_FIM_OP)

    # Cache compartilhado para pre-calculos pesados
    cache = {}

    # Referencia (ponto otimo)
    print("[REF] Rodando backtest com parametros otimos (referencia)...")
    sinal_ref, sl_ref, tp_ref = calcular_sinais(df, PARAMS_OTIMOS, janela_op, ESTRATEGIA, cache)
    resultado_ref = rodar_backtest(df, sinal_ref, sl_ref, tp_ref, ESTRATEGIA, cache)
    dd_referencia = resultado_ref["dd_pct"]
    print(f"   PnL: {resultado_ref['pnl_pct']:+.1f}% | FR: {resultado_ref['fr']:.2f} | "
          f"DD: {resultado_ref['dd_pct']:.1f}% | Trades: {resultado_ref['total_trades']}\n")

    # Loop principal
    resultados_por_param = {}
    for param_name, val_orig in PARAMS_OTIMOS.items():
        grid = gerar_grid(param_name, val_orig)
        idx_otimo = int(np.argmin(np.abs(grid.astype(float) - float(val_orig))))

        resultados_steps = []
        for val in tqdm(grid, desc=f"  {param_name}", ncols=70):
            params_teste = PARAMS_OTIMOS.copy()
            params_teste[param_name] = float(val)
            sinal, sl, tp = calcular_sinais(df, params_teste, janela_op, ESTRATEGIA, cache)
            resultado = rodar_backtest(df, sinal, sl, tp, ESTRATEGIA, cache)
            resultados_steps.append(resultado)

        criterios = avaliar_criterios(resultados_steps, dd_referencia)
        resultados_por_param[param_name] = {
            "grid": grid, "resultados": resultados_steps,
            "criterios": criterios, "idx_otimo": idx_otimo,
            "valor_original": val_orig,
        }

    # Nova Regra de Robustez SQX customizada:
    total_p = len(resultados_por_param)
    aprovados = sum(1 for d in resultados_por_param.values() if d["criterios"]["passou"])
    pct_aprovados = aprovados / total_p if total_p > 0 else 0.0
    
    todos_positivos = all(
        res["pnl_pct"] >= 0.0
        for d in resultados_por_param.values() 
        for res in d["resultados"]
    )
    
    passou_tudo = (pct_aprovados >= 0.80) and todos_positivos

    imprimir_relatorio(resultados_por_param, passou_tudo, ESTRATEGIA, ATIVO, TIMEFRAME)
    gerar_grafico(resultados_por_param, passou_tudo, ESTRATEGIA, ATIVO, TIMEFRAME, dir_saida)
    salvar_csv(resultados_por_param, ESTRATEGIA, dir_saida)
    gerar_tabela_robustez(dir_saida, ATIVO, TIMEFRAME)

    print("\n[OK] Teste de robustez finalizado com sucesso.\n")


if __name__ == "__main__":
    main()
