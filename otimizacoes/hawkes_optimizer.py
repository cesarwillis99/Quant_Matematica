# -*- coding: utf-8 -*-
"""
================================================
hawkes_optimizer.py — Estratégia Quantitativa
Versão Genérica — Reutilizável para qualquer ativo
================================================
Configuração:
  Definir ATIVO e TIMEFRAME no bloco de
  configuração no topo deste arquivo antes
  de executar.

Uso:
  1. Configurar ATIVO e TIMEFRAME
  2. Garantir que os parquets de entrada
     existam na pasta data/ do projeto alvo
  3. Executar: python hawkes_optimizer.py
================================================
"""

import sys
import math
import logging
import warnings
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict
import concurrent.futures

# ===========================================
# CONFIGURAÇÃO DO ATIVO — ALTERAR AQUI
# ===========================================
ATIVO          = "EURUSD"
TIMEFRAME      = "H1"
DIR_PROJETO    = Path(__file__).resolve().parent.parent
DIR_DATA       = DIR_PROJETO / "data"
DIR_GRAFICOS   = DIR_PROJETO / "graficos"

PARQUET_COMPLETO    = DIR_DATA / f"{ATIVO.lower()}_{TIMEFRAME.lower()}_completo.parquet"
PARQUET_OPERACIONAL = DIR_DATA / f"{ATIVO.lower()}_{TIMEFRAME.lower()}_operacional.parquet"
PARQUET_HURST       = DIR_DATA / f"{ATIVO.lower()}_{TIMEFRAME.lower()}_hurst.parquet"
PARQUET_SAIDA       = DIR_DATA / f"{ATIVO.lower()}_{TIMEFRAME.lower()}_hawkes.parquet"
CAMINHO_GRAFICO     = DIR_GRAFICOS / f"{ATIVO.lower()}_{TIMEFRAME.lower()}_hawkes_sinais.png"

# ================================================
# COMO USAR PARA NOVO ATIVO:
# 1. Alterar ATIVO = "NASDAQ" (ou outro)
# 2. Alterar TIMEFRAME = "M10" (ou outro)
# 3. Garantir que existam os parquets:
#    data/nasdaq_m10_completo.parquet
#    data/nasdaq_m10_operacional.parquet
#    data/nasdaq_m10_hurst.parquet (se necessário)
# 4. Executar normalmente
# ================================================

warnings.filterwarnings("ignore")

# Configuração de Logging Profissional
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTES E CAMINHOS
# =============================================================================
DIR_ATUAL = Path(__file__).resolve().parent

CSV_SAIDA = DIR_ATUAL / "resultados" / "top_10_hawkes.csv"

# Parâmetros Base
CAPITAL_INICIAL = 10_000.0
RISCO_POR_TRADE = 0.01
VALOR_PIP_POR_LOTE = 10.0
SPREAD_PIPS = 0.5
FATOR_PIPS = 10_000.0

# =============================================================================
# GRADE DE OTIMIZAÇÃO (720 Combinações)
# =============================================================================
GRID_PARAMS = {
    "excitacao_max": [0.65, 0.75, 0.85, 0.95],
    "norm_min_sinal": [1.0, 1.5, 2.0, 2.5, 3.0],
    "norm_saida": [0.2, 0.4, 0.6, 0.8],
    "sl_mult": [1.5, 2.0, 2.5],
    "tp_mult": [2.5, 3.0, 4.0]
}

# =============================================================================
# FUNÇÕES DE SIMULAÇÃO (ULTRARRÁPIDAS)
# =============================================================================

def calcular_tamanho_lote(capital: float, sl_pips: float) -> float:
    if sl_pips <= 0 or not math.isfinite(sl_pips):
        return 0.01
    risco_monetario = capital * RISCO_POR_TRADE
    lote = risco_monetario / (sl_pips * VALOR_PIP_POR_LOTE)
    return float(np.clip(lote, 0.01, 100.0))

def calcular_pnl(direcao: int, entrada: float, saida: float, lote: float) -> Tuple[float, float]:
    pnl_pips = direcao * (saida - entrada) * FATOR_PIPS
    pnl_bruto = pnl_pips * lote * VALOR_PIP_POR_LOTE
    custo_spread = SPREAD_PIPS * lote * VALOR_PIP_POR_LOTE
    pnl_net = pnl_bruto - custo_spread
    return float(pnl_pips), float(pnl_net)

def simular_hawkes_rapido(
    closes: np.ndarray, highs: np.ndarray, lows: np.ndarray, opens: np.ndarray,
    rt: np.ndarray, hawkes_valido: np.ndarray, excitacao: np.ndarray, 
    lambda_norm: np.ndarray, lambda_norm_prev: np.ndarray, mask_op: np.ndarray,
    vr_pips: np.ndarray, dt_index_day: np.ndarray, dt_index_hour: np.ndarray,
    exc_max: float, norm_min: float, norm_saida: float, sl_mult: float, tp_mult: float
) -> Tuple[List[float], float, float]:
    
    capital = CAPITAL_INICIAL
    posicao = 0
    entrada_preco = 0.0
    sl_preco = 0.0
    tp_preco = 0.0
    lot_size = 0.0
    
    pnls = []
    equity = [CAPITAL_INICIAL]
    
    n = len(closes)
    
    for i in range(n):
        dia = dt_index_day[i]
        hora = dt_index_hour[i]
        
        eh_sexta_21h = (dia == 4) and (hora == 21)
        bloqueio = (dia == 4 and hora >= 21) or (dia == 5) or (dia == 6 and hora < 21)
        
        # 1. Gerenciar posição aberta
        if posicao != 0:
            high_i = highs[i]
            low_i = lows[i]
            close_i = closes[i]
            norm_i = lambda_norm[i]
            
            deve_fechar = False
            preco_saida = 0.0
            
            if eh_sexta_21h:
                deve_fechar = True
                preco_saida = close_i
            elif posicao == 1:
                if low_i <= sl_preco:
                    deve_fechar = True
                    preco_saida = sl_preco
                elif high_i >= tp_preco:
                    deve_fechar = True
                    preco_saida = tp_preco
                elif norm_i < norm_saida:
                    deve_fechar = True
                    preco_saida = close_i
            elif posicao == -1:
                if high_i >= sl_preco:
                    deve_fechar = True
                    preco_saida = sl_preco
                elif low_i <= tp_preco:
                    deve_fechar = True
                    preco_saida = tp_preco
                elif norm_i < norm_saida:
                    deve_fechar = True
                    preco_saida = close_i
                    
            if deve_fechar:
                _, pnl_m = calcular_pnl(posicao, entrada_preco, preco_saida, lot_size)
                pnls.append(pnl_m)
                capital += pnl_m
                posicao = 0
                
        equity.append(capital)
        
        # 2. Tentar abrir nova posição
        if posicao == 0 and not bloqueio and mask_op[i]:
            if i + 1 >= n:
                continue
                
            valido_i = hawkes_valido[i]
            if not valido_i:
                continue
                
            exc_i = excitacao[i]
            norm_i = lambda_norm[i]
            norm_prev_i = lambda_norm_prev[i]
            
            # Condições base
            if exc_i < exc_max and norm_i > norm_min and norm_i < norm_prev_i:
                # Gatilhos
                rt_i = rt[i]
                sinal = 0
                if rt_i < 0:
                    sinal = 1
                elif rt_i > 0:
                    sinal = -1
                    
                if sinal != 0:
                    vr_i = vr_pips[i]
                    if not (math.isfinite(vr_i) and vr_i > 0):
                        continue
                        
                    sl_pips = sl_mult * vr_i
                    tp_pips = tp_mult * vr_i
                    
                    lot_size = calcular_tamanho_lote(capital, sl_pips)
                    delta_sl = sl_pips / FATOR_PIPS
                    delta_tp = tp_pips / FATOR_PIPS
                    
                    entrada_preco = opens[i + 1]
                    posicao = sinal
                    
                    if posicao == 1:
                        sl_preco = entrada_preco - delta_sl
                        tp_preco = entrada_preco + delta_tp
                    else:
                        sl_preco = entrada_preco + delta_sl
                        tp_preco = entrada_preco - delta_tp
                        
    # Fechar resíduos
    if posicao != 0:
        _, pnl_m = calcular_pnl(posicao, entrada_preco, closes[-1], lot_size)
        pnls.append(pnl_m)
        capital += pnl_m
        equity.append(capital)
        
    equity_arr = np.array(equity)
    picos = np.maximum.accumulate(equity_arr)
    dds = (equity_arr - picos) / picos * 100
    dd_max = float(dds.min())
    
    return pnls, capital, dd_max

def avaliar_parametros_hawkes(params, arrays_dict):
    exc_max, norm_min, norm_saida, sl_mult, tp_mult = params
    
    pnls, capital_final, dd_max = simular_hawkes_rapido(
        arrays_dict["closes"], arrays_dict["highs"], arrays_dict["lows"], arrays_dict["opens"],
        arrays_dict["rt"], arrays_dict["valido"], arrays_dict["excitacao"],
        arrays_dict["lambda_norm"], arrays_dict["lambda_norm_prev"], arrays_dict["mask_op"],
        arrays_dict["vr_pips"], arrays_dict["dt_dia"], arrays_dict["dt_hora"],
        exc_max, norm_min, norm_saida, sl_mult, tp_mult
    )
    
    n_ops = len(pnls)
    if n_ops < 50:
        return None
        
    pnls_arr = np.array(pnls)
    ganhos = pnls_arr[pnls_arr > 0]
    perdas = pnls_arr[pnls_arr <= 0]
    
    win_rate = len(ganhos) / n_ops if n_ops > 0 else 0
    soma_g = np.sum(ganhos)
    soma_p = np.abs(np.sum(perdas))
    fator_lucro = soma_g / soma_p if soma_p > 0 else 0
    
    media_g = np.mean(ganhos) if len(ganhos) > 0 else 0
    media_p = np.abs(np.mean(perdas)) if len(perdas) > 0 else 0
    payoff = media_g / media_p if media_p > 0 else 0
    
    pnl_pct = ((capital_final / CAPITAL_INICIAL) - 1) * 100
    
    # Sharpe Simplificado (apenas PnL por trade para filtro)
    std_pnls = np.std(pnls_arr)
    sharpe = (np.mean(pnls_arr) / std_pnls * math.sqrt(n_ops)) if std_pnls > 0 else 0
    
    # Fator de Recuperação
    fator_recuperacao = (pnl_pct / abs(dd_max)) if dd_max < 0 else 0
    
    # Filtros de Qualidade Básica
    if dd_max < -25.0: return None
    if fator_lucro < 1.1: return None
    
    return {
        "Excitacao_Max": exc_max,
        "Norm_Min_Gatilho": norm_min,
        "Norm_Saida_Neutra": norm_saida,
        "SL_Mult": sl_mult,
        "TP_Mult": tp_mult,
        "Total_Ops": n_ops,
        "WinRate_%": round(win_rate * 100, 2),
        "Fator_Lucro": round(fator_lucro, 3),
        "Payoff": round(payoff, 3),
        "Drawdown_%": round(dd_max, 2),
        "PnL_%": round(pnl_pct, 2),
        "Sharpe_Index": round(sharpe, 3),
        "Fator_Recuperacao": round(fator_recuperacao, 3)
    }

# =============================================================================
# WORKER PARA MULTIPROCESSAMENTO
# =============================================================================
# Usa dicionário global para não copiar arrays via pickle
global_arrays = {}

def init_worker(arrays):
    global global_arrays
    global_arrays = arrays

def worker_tarefa(params):
    return avaliar_parametros_hawkes(params, global_arrays)

# =============================================================================
# EXECUTOR PRINCIPAL
# =============================================================================

def rodar_otimizacao_hawkes():
    logger.info("Carregando bases do Hawkes...")
    df = pd.read_parquet(PARQUET_COMPLETO)
    df_op = pd.read_parquet(PARQUET_OPERACIONALERACIONAL)
    
    # Preparar Vetores para Numpy C-Speed
    logger.info("Vetorizando dados históricos...")
    df["R_t"] = np.log(df["Close"] / df["Close"].shift(1))
    df["VR"] = df["R_t"].rolling(window=50).std(ddof=1)
    vr_pips = (df["VR"] * df["Close"] * 10000.0).values
    
    rt = df["R_t"].values
    valido = df["hawkes_valido"].values == 1
    excitacao = df["hawkes_excitacao"].values
    lambda_norm = df["hawkes_lambda_norm"].values
    
    lam_norm_prev = np.roll(lambda_norm, 1)
    lam_norm_prev[0] = np.nan
    
    mask_op = df.index.isin(df_op.index).astype(bool)
    
    arrays_dict = {
        "closes": df["Close"].values.astype(np.float64),
        "highs": df["High"].values.astype(np.float64),
        "lows": df["Low"].values.astype(np.float64),
        "opens": df["Open"].values.astype(np.float64),
        "rt": rt,
        "valido": valido,
        "excitacao": excitacao,
        "lambda_norm": lambda_norm,
        "lambda_norm_prev": lam_norm_prev,
        "mask_op": mask_op,
        "vr_pips": vr_pips,
        "dt_dia": df.index.weekday.values.astype(np.int8),
        "dt_hora": df.index.hour.values.astype(np.int8)
    }
    
    # Gerar Grade
    chaves = list(GRID_PARAMS.keys())
    valores = list(GRID_PARAMS.values())
    combinacoes = list(itertools.product(*valores))
    
    total_combs = len(combinacoes)
    logger.info(f"Grade de Otimização gerada: {total_combs} combinações.")
    logger.info("Iniciando Multiprocessamento...")
    
    resultados_validos = []
    
    with concurrent.futures.ProcessPoolExecutor(initializer=init_worker, initargs=(arrays_dict,)) as executor:
        resultados = list(executor.map(worker_tarefa, combinacoes))
        
        for res in resultados:
            if res is not None:
                resultados_validos.append(res)
                
    if not resultados_validos:
        logger.warning("Nenhuma combinação superou os filtros mínimos de PnL e Drawdown!")
        return
        
    df_res = pd.DataFrame(resultados_validos)
    df_res.sort_values(by="Fator_Recuperacao", ascending=False, inplace=True)
    
    top_10 = df_res.head(10)
    
    CSV_SAIDA.parent.mkdir(parents=True, exist_ok=True)
    top_10.to_csv(CSV_SAIDA, index=False)
    
    logger.info(f"Otimização concluída com sucesso! Top 10 salvo em {CSV_SAIDA.name}")
    
    print("\n" + "=" * 80)
    print("=   TOP 5 RESULTADOS — OTIMIZAÇÃO HAWKES PROCESS")
    print("=" + "=" * 78)
    
    for i, (_, row) in enumerate(top_10.head(5).iterrows()):
        print(f"  #{i+1} | Recup: {row['Fator_Recuperacao']:.3f} | PnL: {row['PnL_%']:+.2f}% | DD: {row['Drawdown_%']:.2f}% | "
              f"WinRate: {row['WinRate_%']:.2f}% | Fator Lucro: {row['Fator_Lucro']:.2f}x")
        print(f"       -> [ExcMax: {row['Excitacao_Max']} | NormMin: {row['Norm_Min_Gatilho']} | NormSaida: {row['Norm_Saida_Neutra']} | SL: {row['SL_Mult']}x | TP: {row['TP_Mult']}x]\n")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    try:
        rodar_otimizacao_hawkes()
    except Exception as e:
        logger.exception("Falha crítica no Otimizador Hawkes:")
        sys.exit(1)
