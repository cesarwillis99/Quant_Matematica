#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_oos.py
----------
Script UNIFICADO e 100% GENÉRICO para testes de robustez Out-of-Sample (OOS).
Centralizado sob a pasta testes_robustez/oos/

Uso:
  python run_oos.py --ativo EURUSD --timeframe H1 --estrategia MOMENTUM --tipo PASSADO
  python run_oos.py --ativo EURUSD --timeframe H1 --estrategia HAWKES --tipo FUTURO
"""

import sys
import math
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

# Corrige encoding de stdout para Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# Configurações globais de negociação
CAPITAL_INICIAL    = 10_000.0
RISCO_POR_TRADE    = 0.01

SPREAD_POR_ATIVO = {
    "eurusd": 0.5,
    "gbpusd": 1.0,
    "usdjpy": 0.7,
    "usdcad": 0.7,
    "audusd": 0.7,
    "nzdusd": 0.7,
    "usdchf": 0.7,
}
SPREAD_PIPS        = 0.5  # Atualizado via global

VALOR_PIP_POR_LOTE = 10.0
FATOR_PIPS         = 10_000
HORA_INICIO_OP     = "10:00"
HORA_FIM_OP        = "22:30"
HORA_BLOQUEIO_FDS  = 20
HORA_FECHAMENTO_FDS = 21


# ===================================================================
# MOTOR MATEMATICO HURST (GENERICO E OTIMIZADO)
# ===================================================================
def _quick_ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    x_mean = x.mean()
    y_mean = y.mean()
    num = ((x - x_mean) * (y - y_mean)).sum()
    den = ((x - x_mean) ** 2).sum()
    if den == 0: return np.nan
    return num / den

def calcular_hurst_janela(retornos: np.ndarray) -> float:
    if len(retornos) < 100: return np.nan
    log_n, log_rs = [], []
    for n in [10, 20, 40, 80]:
        num_segmentos = 100 // n
        rs_segmentos = []
        for k in range(num_segmentos):
            segmento = retornos[k * n : (k + 1) * n]
            mu = segmento.mean()
            y_t = np.cumsum(segmento - mu)
            r_range = y_t.max() - y_t.min()
            s_std = segmento.std(ddof=1)
            if s_std == 0: return np.nan
            rs_segmentos.append(r_range / s_std)
        rs_medio = np.mean(rs_segmentos)
        if rs_medio > 0:
            log_n.append(np.log(n))
            log_rs.append(np.log(rs_medio))
    if len(log_n) < 2: return np.nan
    h = _quick_ols_slope(np.array(log_n), np.array(log_rs))
    if np.isnan(h) or h < 0.0 or h > 1.5: return np.nan
    return float(np.clip(h, 0.0, 1.0))
# ===================================================================
#  PARTE 2 -- ROTEADOR MULTI-ESTRATEGIA (BASEADO NO DISTRIBUICAO)
# ===================================================================
from testes_robustez.distribuicao_parametros.distribuicao_parametros import calcular_sinais

# =================================================================================
# PARTE 3 -- MOTOR DE BACKTEST CANDLE-A-CANDLE INSTITUCIONAL
# ===================================================================
def simular_backtest_candle_a_candle(df: pd.DataFrame,
                                     sinal: np.ndarray,
                                     sl_pips: np.ndarray,
                                     tp_pips: np.ndarray) -> tuple:
    capital = CAPITAL_INICIAL
    equity_curve = []
    trades = []
    
    posicao = 0 
    preco_entrada = 0.0
    sl_preco = 0.0
    tp_preco = 0.0
    lote = 0.0
    data_entrada = None
    
    times = df.index
    opens = df["Open"].values
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    
    weekdays = df.index.weekday
    hours = df.index.hour
    n = len(df)
    
    for i in range(n - 1):
        t_time = times[i]
        wd = weekdays[i]
        hr = hours[i]
        
        eh_sexta = (wd == 4)
        eh_sexta_fechamento = eh_sexta and hr == HORA_FECHAMENTO_FDS
        bloqueio_entrada = (eh_sexta and hr >= HORA_BLOQUEIO_FDS) or (wd == 5) or (wd == 6 and hr < 21)
        
        # 1. Gerenciar Posição Aberta
        if posicao != 0:
            fechou = False
            preco_saida = 0.0
            motivo = ""
            
            if eh_sexta_fechamento:
                preco_saida = closes[i]
                fechou = True
                motivo = "FDS"
            else:
                if posicao == 1:
                    if lows[i] <= sl_preco and highs[i] >= tp_preco:
                        preco_saida, fechou, motivo = sl_preco, True, "SL (Prioridade)"
                    elif lows[i] <= sl_preco:
                        preco_saida, fechou, motivo = sl_preco, True, "SL"
                    elif highs[i] >= tp_preco:
                        preco_saida, fechou, motivo = tp_preco, True, "TP"
                
                elif posicao == -1:
                    if highs[i] >= sl_preco and lows[i] <= tp_preco:
                        preco_saida, fechou, motivo = sl_preco, True, "SL (Prioridade)"
                    elif highs[i] >= sl_preco:
                        preco_saida, fechou, motivo = sl_preco, True, "SL"
                    elif lows[i] <= tp_preco:
                        preco_saida, fechou, motivo = tp_preco, True, "TP"
            
            if fechou:
                pnl_pips = (preco_saida - preco_entrada) * FATOR_PIPS if posicao == 1 else (preco_entrada - preco_saida) * FATOR_PIPS
                pnl_usd = (pnl_pips * lote * VALOR_PIP_POR_LOTE) - (SPREAD_PIPS * lote * VALOR_PIP_POR_LOTE)
                capital += pnl_usd
                
                trades.append({
                    "data_entrada": data_entrada,
                    "data_saida": t_time,
                    "direcao": "LONG" if posicao == 1 else "SHORT",
                    "preco_entrada": preco_entrada,
                    "preco_saida": preco_saida,
                    "lot_size": lote, 
                    "pnl_monetario": pnl_usd, 
                    "motivo": motivo
                })
                posicao = 0
        
        # 2. Entrar em nova posição
        if posicao == 0 and sinal[i] != 0 and not bloqueio_entrada:
            sl_p = sl_pips[i]
            tp_p = tp_pips[i]
            
            if np.isfinite(sl_p) and sl_p > 0 and np.isfinite(tp_p) and tp_p > 0:
                posicao = sinal[i]
                preco_entrada = opens[i+1]
                data_entrada = times[i+1]
                
                d_sl = sl_p / FATOR_PIPS
                d_tp = tp_p / FATOR_PIPS
                
                if posicao == 1:
                    sl_preco = preco_entrada - d_sl
                    tp_preco = preco_entrada + d_tp
                else:
                    sl_preco = preco_entrada + d_sl
                    tp_preco = preco_entrada - d_tp
                
                lote = (capital * RISCO_POR_TRADE) / (sl_p * VALOR_PIP_POR_LOTE)
                lote = float(np.clip(lote, 0.01, 100.0))
            
        equity_curve.append(capital)
    
    equity_curve.append(capital)
    return pd.Series(equity_curve, index=df.index), trades


# ===================================================================
# PARTE 4 -- CALCULADORA DE METRICAS E SCORECARD
# ===================================================================
def calcular_metricas(equity_curve: pd.Series, trades: list) -> dict:
    capital_final = equity_curve.iloc[-1]
    pnl_usd_total = capital_final - CAPITAL_INICIAL
    pnl_pct = (pnl_usd_total / CAPITAL_INICIAL) * 100
    
    total_trades = len(trades)
    ganhos = [t for t in trades if t["pnl_monetario"] > 0]
    perdas = [t for t in trades if t["pnl_monetario"] <= 0]
    win_rate = (len(ganhos) / total_trades * 100) if total_trades > 0 else 0.0
    fator_lucro = sum(t["pnl_monetario"] for t in ganhos) / max(abs(sum(t["pnl_monetario"] for t in perdas)), 1e-9) if ganhos else 0.0
    
    # Sharpe blindado contra desvio zero
    if not isinstance(equity_curve.index, pd.DatetimeIndex):
        sharpe = 0.0
    else:
        retornos_diarios = equity_curve.resample("1D").last().pct_change().fillna(0.0)
        std_diario = retornos_diarios.std()
        sharpe = (retornos_diarios.mean() / std_diario) * math.sqrt(252) if std_diario > 0.0 else 0.0
    
    pico = equity_curve.cummax()
    dd_serie = (equity_curve - pico) / pico * 100
    dd_pct = float(dd_serie.min())
    dd_usd = float((equity_curve - pico).min())
    
    fator_recup = pnl_usd_total / abs(dd_usd) if dd_usd < 0 else (pnl_usd_total if pnl_usd_total > 0 else 0.0)
    
    return {
        "pnl_pct":     round(pnl_pct, 2),
        "win_rate":    round(win_rate, 2),
        "sharpe":      round(sharpe, 4),
        "dd_pct":      round(dd_pct, 2),
        "fator_recup": round(fator_recup, 3),
        "fator_lucro": round(fator_lucro, 3),
        "total_trades": total_trades,
        "capital_final": round(capital_final, 2),
        "dd_serie":    dd_serie
    }


# ===================================================================
# PARTE 5 -- RENDERIZADOR DE IMAGENS E DADOS (DARK MODE PREMIUM)
# ===================================================================
def gerar_relatorio_e_graficos(metricas: dict, equity_curve: pd.Series, trades: list,
                               estrategia: str, ativo: str, timeframe: str, tipo_oos: str, sufixo_ano: str, dir_saida: Path, param_id: str = "") -> None:
    
    # 1. Salvar CSV Operações (Será usado em Spread e What If)
    # csv_path = dir_saida / f"operacoes_OOS_{tipo_oos.upper()}_{ativo.upper()}_{estrategia.upper()}_{param_id}.csv"
    # if trades:
    #     pd.DataFrame(trades).to_csv(csv_path, index=False)
    # else:
    #     pd.DataFrame(columns=["data_entrada","data_saida","direcao","preco_entrada","preco_saida","lot_size","pnl_monetario","motivo"]).to_csv(csv_path, index=False)
    # print(f"[CSV] Operações OOS salvas em: {csv_path}")

    # Configuração visual Matplotlib Dark Premium
    plt.rcParams.update({
        "figure.facecolor": "#0D1117", "axes.facecolor":  "#0D1117",
        "axes.edgecolor":   "#21262D", "axes.labelcolor": "#E6EDF3",
        "xtick.color":      "#E6EDF3", "ytick.color":     "#E6EDF3",
        "text.color":       "#E6EDF3", "grid.color":      "#21262D",
        "grid.linestyle":   "--",      "grid.alpha":      0.4,
        "font.family":      "monospace",
    })

    # Criar figura com GridSpec: lado esquerdo (grafico) e lado direito (tabela)
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(20, 8.5))
    fig.patch.set_facecolor('#0D1117')
    
    gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1.2], height_ratios=[3, 1])
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax_tbl = fig.add_subplot(gs[:, 1])
    
    # --- Painel 1: Equity Curve ---
    cor_pnl = "#58A6FF" if metricas["pnl_pct"] >= 0 else "#FF7B72"
    ax1.plot(equity_curve.index, equity_curve.values, color=cor_pnl, linewidth=1.8, label="Curva OOS")
    ax1.fill_between(equity_curve.index, CAPITAL_INICIAL, equity_curve.values,
                     where=(equity_curve.values >= CAPITAL_INICIAL), alpha=0.1, color="#3FB950")
    ax1.fill_between(equity_curve.index, CAPITAL_INICIAL, equity_curve.values,
                     where=(equity_curve.values < CAPITAL_INICIAL), alpha=0.1, color="#F85149")
    ax1.axhline(CAPITAL_INICIAL, color="#FFFFFF", linestyle="--", linewidth=0.8, alpha=0.5)
    ax1.set_ylabel("Capital (USD)", fontsize=10)
    ax1.set_title(f"BACKTEST OUT-OF-SAMPLE ({tipo_oos.upper()}) — {estrategia.upper()} | {ativo.upper()} {timeframe.upper()} ({sufixo_ano})",
                  fontsize=12, fontweight="bold", pad=12)
    ax1.grid(True)
    ax1.legend(loc="upper left")
    
    # --- Painel 2: Drawdown ---
    ax2.fill_between(metricas["dd_serie"].index, metricas["dd_serie"].values, 0, color="#FF7B72", alpha=0.3)
    ax2.plot(metricas["dd_serie"].index, metricas["dd_serie"].values, color="#FF7B72", linewidth=1.0)
    ax2.set_ylabel("Drawdown (%)", fontsize=10)
    ax2.set_xlabel("Data", fontsize=10)
    ax2.grid(True)
    
    # --- Painel 3: Tabela de Métricas (Lado Direito) ---
    ax_tbl.axis('tight')
    ax_tbl.axis('off')
    
    dados_tabela = [
        ["PnL (%)", f"{metricas['pnl_pct']:+.2f}%"],
        ["Fator Recuperação", f"{metricas['fator_recup']:.3f}x"],
        ["Fator Lucro", f"{metricas['fator_lucro']:.3f}x"],
        ["Win Rate", f"{metricas['win_rate']:.2f}%"],
        ["Sharpe Ratio", f"{metricas['sharpe']:.4f}"],
        ["Drawdown Máximo", f"{metricas['dd_pct']:.2f}%"],
        ["Total de Trades", f"{metricas['total_trades']}"],
        ["Capital Final", f"${metricas['capital_final']:,.2f}"]
    ]
    
    table = ax_tbl.table(cellText=dados_tabela, colLabels=["Métrica", f"Valor OOS {tipo_oos.upper()}"], loc='center', cellLoc='center')
    table.scale(1, 2.2)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#21262D')
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#161B22')
        else:
            cell.set_text_props(color='#E6EDF3')
            cell.set_facecolor('#0D1117' if row % 2 == 0 else '#161B22')
            
    # Salvar a imagem única unificada
    img_unica_path = dir_saida / f"relatorio_OOS_{tipo_oos.upper()}_{ativo.upper()}_{estrategia.upper()}_{param_id}.png"
    plt.tight_layout()
    plt.savefig(img_unica_path, dpi=150, bbox_inches="tight", facecolor='#0D1117')
    plt.close()
    print(f"[RELATORIO UNIFICADO] Relatório salvo em: {img_unica_path}")


# ===================================================================
# FUNCAO EXPORTAVEL PARA A ESTEIRA (FAIL-FAST)
# ===================================================================
def rodar_oos_na_esteira(df_oos: pd.DataFrame, params_otimos: dict, estrategia: str, ativo: str, timeframe: str, tipo_oos: str, sufixo_ano: str, dir_saida: Path, param_id: str = "") -> dict:
    global SPREAD_PIPS
    SPREAD_PIPS = SPREAD_POR_ATIVO.get(ativo.lower(), 0.5)

    horas = df_oos.index.strftime("%H:%M")
    janela_op = (
        (df_oos.index.weekday >= 0) &
        (df_oos.index.weekday <= 4) &
        (horas >= HORA_INICIO_OP) &
        (horas <= HORA_FIM_OP)
    )
    # NOTA: janela_op filtra apenas ENTRADAS de sinais.
    # Os indicadores matemáticos devem ser calculados sobre
    # a série completa (df_oos sem filtro), não sobre janela_op.
    # Garantir que calcular_sinais aplica janela_op apenas na
    # geração do sinal final, nunca nos cálculos de indicadores.
    
    cache = {}
    sinal, sl_pips, tp_pips = calcular_sinais(df_oos, params_otimos, janela_op, estrategia, cache,
                                               ativo=ativo, timeframe=timeframe)
    equity_curve, trades = simular_backtest_candle_a_candle(df_oos, sinal, sl_pips, tp_pips)
    metricas = calcular_metricas(equity_curve, trades)
    
    gerar_relatorio_e_graficos(metricas, equity_curve, trades, estrategia, ativo, timeframe, tipo_oos, sufixo_ano, dir_saida, param_id)
    
    aprovado = (
        metricas["fator_recup"] >= 0.5 and
        metricas["pnl_pct"] > 0.0 and
        metricas["fator_lucro"] >= 1.0
    )
    return {"aprovado": aprovado, "metricas": metricas}


# ===================================================================
# MAIN ORQUESTRADOR CLI
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="Validação de Robustez Out-of-Sample (OOS) Unificada")
    parser.add_argument("--ativo",      type=str, default="EURUSD")
    parser.add_argument("--timeframe",  type=str, default="H1")
    parser.add_argument("--estrategia", type=str, required=True, help="MOMENTUM, HAWKES, ZSCORE, OU, OU_REVERSO, PCA, WAVELET")
    parser.add_argument("--tipo",       type=str, required=True, help="PASSADO ou FUTURO")
    parser.add_argument("--sufixo_ano", type=str, default=None, help="Ex: 2013_2016 ou 2024_2026")
    args = parser.parse_args()

    ativo      = args.ativo.upper()
    timeframe  = args.timeframe.upper()
    estrategia = args.estrategia.upper()
    tipo_oos   = args.tipo.upper()

    if tipo_oos not in ("PASSADO", "FUTURO"):
        raise ValueError("--tipo deve ser PASSADO ou FUTURO")

    # dir_projeto é o bisavô (três níveis acima: run_oos.py está em testes_robustez/oos/)
    dir_projeto = Path(__file__).resolve().parent.parent.parent
    dir_data = dir_projeto / f"quant_{ativo.lower()}" / "data"

    # Resolver sufixo do ano automaticamente
    sufixo_ano = args.sufixo_ano
    if not sufixo_ano:
        try:
            pattern = f"{ativo.lower()}_{timeframe.lower()}_completo_OOS_{tipo_oos.lower()}_*.parquet"
            files = list(dir_data.glob(pattern))
            if not files:
                raise FileNotFoundError()
            # Extrair sufixo
            filename = files[0].stem
            sufixo_ano = filename.split(f"_OOS_{tipo_oos.lower()}_")[1]
            print(f"[AUTO] Sufixo de ano detectado: {sufixo_ano}")
        except Exception:
            sufixo_ano = "2013_2016" if tipo_oos == "PASSADO" else "2024_2026"
            print(f"[AVISO] Falha ao autodetectar sufixo. Usando fallback default: {sufixo_ano}")

    # Salva na pasta testes_robustez/oos/{passado|futuro}/{ativo}_{timeframe}/{estrategia}
    dir_saida = Path(__file__).resolve().parent / tipo_oos.lower() / f"{ativo.lower()}_{timeframe.lower()}" / estrategia.lower()
    dir_saida.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  TESTE DE ROBUSTEZ OUT-OF-SAMPLE ({tipo_oos}) -- MOTOR CENTRALIZADO OOS/")
    print(f"  Estratégia : {estrategia}")
    print(f"  Ativo      : {ativo} {timeframe}")
    print(f"  Período    : {sufixo_ano}")
    print(f"  Saída      : {dir_saida}")
    print(f"{'='*70}\n")

    # Parquets
    parquet_completo    = dir_data / f"{ativo.lower()}_{timeframe.lower()}_completo_OOS_{tipo_oos.lower()}_{sufixo_ano}.parquet"
    parquet_operacional = dir_data / f"{ativo.lower()}_{timeframe.lower()}_operacional_OOS_{tipo_oos.lower()}_{sufixo_ano}.parquet"

    if not parquet_completo.exists() or not parquet_operacional.exists():
        raise FileNotFoundError(f"Parquet OOS {tipo_oos} correspondente não encontrado em {dir_data}")

    # Otimizações
    parquet_otim = dir_data / "otimizacoes" / estrategia.lower() / f"otimizacao_{estrategia.lower()}_resultados.parquet"
    if not parquet_otim.exists():
        raise FileNotFoundError(f"Combinações de otimização não encontradas em {parquet_otim}. Rode a otimização antes.")

    df_comb = pd.read_parquet(parquet_otim)
    col_sort = "Ret_DD" if "Ret_DD" in df_comb.columns else ("Fator_Recuperacao" if "Fator_Recuperacao" in df_comb.columns else "Lucro_Total_Pips")
    df_comb = df_comb.sort_values(col_sort, ascending=False).reset_index(drop=True)
    melhor_row = df_comb.iloc[0]

    # Mapear parâmetros da combinação ótima campeã
    param_cols = [c for c in df_comb.columns if c not in ("Trades", "Lucro_Total_Pips", "Max_DD_Pips", "Ret_DD", "Ret_DD_Pips", "lucro", "drawdown")]
    params_otimos = {c: melhor_row[c] for c in param_cols}

    print("[COMB] Parâmetros Ótimos Campeões Selecionados da Otimização:")
    for k, v in params_otimos.items():
        print(f"  - {k}: {v}")
    print("")

    # Carregar séries
    print("[DADOS] Carregando série temporal OOS...")
    df_comp = pd.read_parquet(parquet_completo)
    if "hurst" not in df_comp.columns:
        parquet_hurst = dir_data / f"{ativo.lower()}_{timeframe.lower()}_hurst.parquet"
        if parquet_hurst.exists():
            print(f"[DADOS] Carregando coluna 'hurst' ausente a partir de {parquet_hurst.name}...")
            df_hurst = pd.read_parquet(parquet_hurst, columns=["hurst"])
            df_comp = df_comp.join(df_hurst, how="left")
    
    if not isinstance(df_comp.index, pd.DatetimeIndex):
        for col in ("time", "datetime"):
            if col in df_comp.columns:
                df_comp = df_comp.set_index(pd.to_datetime(df_comp[col]))
                break

    print(f"   {len(df_comp):,} candles carregados.")

    # Janela operacional
    horas = df_comp.index.strftime("%H:%M")
    janela_op = (df_comp.index.weekday >= 0) & (df_comp.index.weekday <= 4) & (horas >= HORA_INICIO_OP) & (horas <= HORA_FIM_OP)

    # Recalcular Sinais OOS
    print(f"[SINAIS] Calculando sinais vetorizados para {estrategia}...")
    cache = {}
    sinal, sl_pips, tp_pips = calcular_sinais(df_comp, params_otimos, janela_op, estrategia, cache,
                                               ativo=ativo, timeframe=timeframe)

    # Simular backtest candle-a-candle
    print("[BACKTEST] Rodando simulação candle-a-candle institucional no OOS...")
    equity_curve, trades = simular_backtest_candle_a_candle(df_comp, sinal, sl_pips, tp_pips)

    # Calcular Métricas
    print("[MÉTRICAS] Calculando scorecard e drawdowns...")
    metricas = calcular_metricas(equity_curve, trades)

    # Salvar resultados e renderizar
    print("[OUTPUT] Gerando relatórios executivos...")
    gerar_relatorio_e_graficos(metricas, equity_curve, trades, estrategia, ativo, timeframe, tipo_oos, sufixo_ano, dir_saida)

    print(f"\n[OK] Teste Out-of-Sample {tipo_oos} finalizado para {estrategia} com sucesso!\n")


if __name__ == "__main__":
    main()
