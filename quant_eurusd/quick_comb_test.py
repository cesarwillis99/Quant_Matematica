import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple
import math

DIR_ATUAL = Path(r"c:\Users\cesar\.gemini\antigravity\scratch\Quant_Matematica.Trade\quant_eurusd")
PARQUET_ZSCORE = DIR_ATUAL / "data" / "eurusd_h1_zscore.parquet"
PARQUET_MOMENTUM = DIR_ATUAL / "data" / "eurusd_h1_momentum.parquet"
PARQUET_HAWKES = DIR_ATUAL / "data" / "eurusd_h1_hawkes.parquet"

CAPITAL_INICIAL = 10_000.0
RISCO_POR_TRADE = 0.01
SPREAD_PIPS = 1.2
VALOR_PIP_POR_LOTE = 10.0
FATOR_PIPS = 10_000.0
Z_NEUTRO_MIN, Z_NEUTRO_MAX = -0.5, 0.5

@dataclass
class Operacao:
    direcao: int
    entrada_preco: float
    saida_dt: Optional[pd.Timestamp] = None
    sl_preco: float = 0.0
    tp_preco: float = 0.0
    lot_size: float = 0.0
    pnl_monetario: float = 0.0
    estrategia: str = ""
    norm_saida: float = 0.0

def calcular_tamanho_lote(capital: float, sl_pips: float) -> float:
    if sl_pips <= 0 or not math.isfinite(sl_pips): return 0.01
    risco_monetario = capital * RISCO_POR_TRADE
    lote = risco_monetario / (sl_pips * VALOR_PIP_POR_LOTE)
    return float(np.clip(lote, 0.01, 100.0))

def calcular_pnl(direcao: int, entrada: float, saida: float, lote: float) -> float:
    pnl_pips = direcao * (saida - entrada) * FATOR_PIPS
    return float(pnl_pips * lote * VALOR_PIP_POR_LOTE - SPREAD_PIPS * lote * VALOR_PIP_POR_LOTE)

def run_combined(nome_teste: str, config_hawkes: dict):
    df_z = pd.read_parquet(PARQUET_ZSCORE)
    df_m = pd.read_parquet(PARQUET_MOMENTUM)
    df_h = pd.read_parquet(PARQUET_HAWKES)
    
    df = df_z.copy()
    
    # ── Mapear Sinais Hawkes Dinamicamente ──
    # config_hawkes = {"exc_max": 0.65, "norm_min": 3.0, "norm_saida": 0.6, "sl_mult": 2.0, "tp_mult": 3.0}
    exc_max = config_hawkes["exc_max"]
    norm_min = config_hawkes["norm_min"]
    norm_saida = config_hawkes["norm_saida"]
    sl_mult = config_hawkes["sl_mult"]
    tp_mult = config_hawkes["tp_mult"]
    
    rt = np.log(df_h["Close"] / df_h["Close"].shift(1))
    vr_pips = df_h["Close"] * 10000.0 * rt.rolling(50).std(ddof=1)
    
    lam_norm = df_h["hawkes_lambda_norm"].values
    lam_norm_prev = np.roll(lam_norm, 1)
    lam_norm_prev[0] = np.nan
    
    valido = df_h["hawkes_valido"] == 1
    excit_ok = df_h["hawkes_excitacao"] < exc_max
    norm_high = lam_norm > norm_min
    norm_falling = lam_norm < lam_norm_prev
    
    mask_op = df_h.index.isin(pd.read_parquet(DIR_ATUAL / "data" / "eurusd_h1_operacional.parquet").index)
    cond_base = valido & excit_ok & mask_op & norm_falling & norm_high
    
    sinal_h = np.zeros(len(df_h), dtype=np.int8)
    sinal_h[cond_base & (rt < 0)] = 1
    sinal_h[cond_base & (rt > 0)] = -1
    
    sl_h = sl_mult * vr_pips
    tp_h = tp_mult * vr_pips
    sl_h = sl_h.clip(lower=3.0, upper=60.0).fillna(10.0)
    tp_h = tp_h.clip(lower=4.5, upper=90.0).fillna(15.0)
    
    # ── Combinar Sinais (Prioridade: ZSCORE -> MOMENTUM -> HAWKES) ──
    s_z = df["sinal_zscore"].values
    s_m = df_m["sinal_momentum"].values
    
    sinal_comb = np.zeros(len(df), dtype=np.int8)
    origem_sinal = np.zeros(len(df), dtype=np.int8) # 1=Z, 2=M, 3=H
    
    for i in range(len(df)):
        if s_z[i] != 0:
            sinal_comb[i] = s_z[i]
            origem_sinal[i] = 1
        elif s_m[i] != 0:
            sinal_comb[i] = s_m[i]
            origem_sinal[i] = 2
        elif sinal_h[i] != 0:
            sinal_comb[i] = sinal_h[i]
            origem_sinal[i] = 3
            
    # ── Simular ──
    opens = df["Open"].values
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    zscores = df["zscore"].values
    
    sl_z = df["sl_pips"].values
    tp_z = df["tp_pips"].values
    sl_m_arr = df_m["sl_pips"].values
    tp_m_arr = df_m["tp_pips"].values
    sl_h_arr = sl_h.values
    tp_h_arr = tp_h.values
    
    capital = CAPITAL_INICIAL
    posicao = None
    pnls = []
    equity = [CAPITAL_INICIAL]
    
    indices = df.index
    
    for i in range(len(df)):
        dt = indices[i]
        dia = dt.weekday()
        hora = dt.hour
        eh_sexta_21h = (dia == 4) and (hora == 21)
        bloqueio = (dia == 4 and hora >= 21) or (dia == 5) or (dia == 6 and hora < 21)
        
        if posicao is not None:
            if eh_sexta_21h:
                pnl_m = calcular_pnl(posicao.direcao, posicao.entrada_preco, closes[i], posicao.lot_size)
                capital += pnl_m
                pnls.append(pnl_m)
                posicao = None
            else:
                fechar = False
                preco_saida = 0.0
                
                if posicao.direcao == 1:
                    if lows[i] <= posicao.sl_preco:
                        fechar, preco_saida = True, posicao.sl_preco
                    elif highs[i] >= posicao.tp_preco:
                        fechar, preco_saida = True, posicao.tp_preco
                    elif posicao.estrategia == "ZSCORE" and Z_NEUTRO_MIN <= zscores[i] <= Z_NEUTRO_MAX:
                        fechar, preco_saida = True, closes[i]
                    elif posicao.estrategia == "HAWKES" and lam_norm[i] < posicao.norm_saida:
                        fechar, preco_saida = True, closes[i]
                else:
                    if highs[i] >= posicao.sl_preco:
                        fechar, preco_saida = True, posicao.sl_preco
                    elif lows[i] <= posicao.tp_preco:
                        fechar, preco_saida = True, posicao.tp_preco
                    elif posicao.estrategia == "ZSCORE" and Z_NEUTRO_MIN <= zscores[i] <= Z_NEUTRO_MAX:
                        fechar, preco_saida = True, closes[i]
                    elif posicao.estrategia == "HAWKES" and lam_norm[i] < posicao.norm_saida:
                        fechar, preco_saida = True, closes[i]
                        
                if fechar:
                    pnl_m = calcular_pnl(posicao.direcao, posicao.entrada_preco, preco_saida, posicao.lot_size)
                    capital += pnl_m
                    pnls.append(pnl_m)
                    posicao = None
                    
        equity.append(capital)
        
        if posicao is None and sinal_comb[i] != 0 and not bloqueio:
            if i + 1 >= len(df): continue
            
            orig = origem_sinal[i]
            if orig == 1:
                sl_pips, tp_pips, estr = sl_z[i], tp_z[i], "ZSCORE"
            elif orig == 2:
                sl_pips, tp_pips, estr = sl_m_arr[i], tp_m_arr[i], "MOMENTUM"
            else:
                sl_pips, tp_pips, estr = sl_h_arr[i], tp_h_arr[i], "HAWKES"
                
            if not (math.isfinite(sl_pips) and sl_pips > 0): continue
            
            lot = calcular_tamanho_lote(capital, sl_pips)
            ent = opens[i + 1]
            dir_op = sinal_comb[i]
            
            sl_pr = ent - (sl_pips/10000) if dir_op == 1 else ent + (sl_pips/10000)
            tp_pr = ent + (tp_pips/10000) if dir_op == 1 else ent - (tp_pips/10000)
            
            posicao = Operacao(
                direcao=dir_op, entrada_preco=ent, sl_preco=sl_pr, tp_preco=tp_pr,
                lot_size=lot, estrategia=estr, norm_saida=norm_saida
            )
            
    # metrics
    eq_arr = np.array(equity)
    picos = np.maximum.accumulate(eq_arr)
    dd_max = ((eq_arr - picos) / picos * 100).min()
    
    pnl_pct = ((capital / CAPITAL_INICIAL) - 1) * 100
    fr = pnl_pct / abs(dd_max) if dd_max < 0 else 0
    
    print(f"[{nome_teste}] -> PnL: {pnl_pct:+.2f}% | Drawdown: {dd_max:.2f}% | Fator Recup: {fr:.3f} | Ops: {len(pnls)}")

print("Avaliando Combinações (ZScore + Momentum + Hawkes)...")

# Otimizada (29.79% F.R. 4.51)
config_opt = {"exc_max": 0.65, "norm_min": 3.0, "norm_saida": 0.6, "sl_mult": 2.0, "tp_mult": 3.0}
run_combined("PORTFÓLIO 1: COM HAWKES OTIMIZADO (Curva de 29%)", config_opt)

# Original (44.10% F.R. 3.23)
config_orig = {"exc_max": 0.85, "norm_min": 1.5, "norm_saida": 0.5, "sl_mult": 2.0, "tp_mult": 3.0}
run_combined("PORTFÓLIO 2: COM HAWKES ORIGINAL (Curva de 44%)", config_orig)
