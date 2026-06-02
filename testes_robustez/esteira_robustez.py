import argparse
import os
import json
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("EsteiraRobustez")

# Adicionar path para importar módulos do projeto
DIR_PROJETO = Path(__file__).resolve().parent.parent
if str(DIR_PROJETO) not in sys.path:
    sys.path.insert(0, str(DIR_PROJETO))

SPREAD_POR_ATIVO = {
    "eurusd": 0.5,
    "gbpusd": 1.0,
    "usdjpy": 0.7,
    "usdcad": 0.7,
    "audusd": 0.7,
    "nzdusd": 0.7,
    "usdchf": 0.7,
}

from testes_robustez.distribuicao_parametros.distribuicao_parametros import calcular_sinais
from backtests.backtest_individual import simular_estrategia, Operacao

def calcular_hurst(retornos: np.ndarray, janela: int = 100) -> np.ndarray:
    n = len(retornos)
    hurst_values = np.full(n, 0.5, dtype=np.float32)
    if n < janela:
        return hurst_values
    lags = [10, 20, 40, 80]
    for i in range(janela - 1, n):
        window = retornos[i - janela + 1 : i + 1]
        log_n = []
        log_rs = []
        for lag in lags:
            num_segmentos = janela // lag
            rs_segmentos = []
            for k in range(num_segmentos):
                segmento = window[k * lag : (k + 1) * lag]
                mu = segmento.mean()
                if len(segmento) == 0: continue
                y_t = np.cumsum(segmento - mu)
                r_range = y_t.max() - y_t.min()
                s_std = segmento.std(ddof=1)
                if s_std > 0:
                    rs_segmentos.append(r_range / s_std)
            if rs_segmentos:
                rs_medio = np.mean(rs_segmentos)
                if rs_medio > 0:
                    log_n.append(np.log(lag))
                    log_rs.append(np.log(rs_medio))
        if len(log_n) >= 2:
            x = np.array(log_n)
            y = np.array(log_rs)
            x_mean = x.mean()
            y_mean = y.mean()
            num = ((x - x_mean) * (y - y_mean)).sum()
            den = ((x - x_mean) ** 2).sum()
            h = num / den if den != 0 else np.nan
            if np.isfinite(h) and 0.0 <= h <= 1.5:
                hurst_values[i] = h
    return hurst_values

# --- WRAPPERS DOS TESTES ---
# Os testes agora serão importados e executados como funções python, passando os parâmetros
def run_oos_futuro(params, ativo, timeframe, estrategia, param_id, df_ops, dir_saida, df_comp=None) -> bool:
    from testes_robustez.oos.run_oos import rodar_oos_na_esteira
    especific_dir = os.path.join(dir_saida, "oos_futuro", estrategia.lower())
    os.makedirs(especific_dir, exist_ok=True)
    
    dir_data = DIR_PROJETO / f"quant_{ativo.lower()}_{timeframe.lower()}" / "data"
    padrao = f"{ativo.lower()}_{timeframe.lower()}_completo_OOS_futuro_*.parquet"
    arquivos = list(dir_data.glob(padrao))
    if not arquivos:
        logger.warning(f"OOS FUTURO ignorado (arquivo não encontrado: {padrao}).")
        return True
        
    sufixo_ano = arquivos[0].stem.split("_OOS_futuro_")[1]
    df_oos = pd.read_parquet(arquivos[0])
    if "hurst" not in df_oos.columns:
        parquet_hurst = dir_data / f"{ativo.lower()}_{timeframe.lower()}_hurst.parquet"
        if parquet_hurst.exists():
            df_hurst = pd.read_parquet(parquet_hurst, columns=["hurst"])
            df_oos = df_oos.join(df_hurst, how="left")
            
    # Se hurst continuar nulo (caso de dados OOS novos sem indicadores históricos pré-calculados)
    if "hurst" not in df_oos.columns or df_oos["hurst"].isna().any():
        logger.info(f"[{param_id}] Calculando expoente de Hurst dinamicamente para OOS Futuro...")
        if "log_return" not in df_oos.columns:
            closes = df_oos["Close"].values
            df_oos["log_return"] = np.log(closes / np.roll(closes, 1))
            df_oos["log_return"].iloc[0] = 0.0
        df_oos["hurst"] = calcular_hurst(df_oos["log_return"].values, janela=100)
            
    res = rodar_oos_na_esteira(df_oos, params, estrategia, ativo, timeframe, "FUTURO", sufixo_ano, Path(especific_dir), param_id)
    if res and res.get('aprovado', False):
        return True
    return False

def run_oos_passado(params, ativo, timeframe, estrategia, param_id, df_ops, dir_saida, df_comp=None) -> bool:
    from testes_robustez.oos.run_oos import rodar_oos_na_esteira
    especific_dir = os.path.join(dir_saida, "oos_passado", estrategia.lower())
    os.makedirs(especific_dir, exist_ok=True)
    
    dir_data = DIR_PROJETO / f"quant_{ativo.lower()}_{timeframe.lower()}" / "data"
    padrao = f"{ativo.lower()}_{timeframe.lower()}_completo_OOS_passado_*.parquet"
    arquivos = list(dir_data.glob(padrao))
    if not arquivos:
        logger.warning(f"OOS PASSADO ignorado (arquivo não encontrado: {padrao}).")
        return True
        
    sufixo_ano = arquivos[0].stem.split("_OOS_passado_")[1]
    df_oos = pd.read_parquet(arquivos[0])
    if "hurst" not in df_oos.columns:
        parquet_hurst = dir_data / f"{ativo.lower()}_{timeframe.lower()}_hurst.parquet"
        if parquet_hurst.exists():
            df_hurst = pd.read_parquet(parquet_hurst, columns=["hurst"])
            df_oos = df_oos.join(df_hurst, how="left")
            
    # Se hurst continuar nulo (caso de dados OOS novos sem indicadores históricos pré-calculados)
    if "hurst" not in df_oos.columns or df_oos["hurst"].isna().any():
        logger.info(f"[{param_id}] Calculando expoente de Hurst dinamicamente para OOS Passado...")
        if "log_return" not in df_oos.columns:
            closes = df_oos["Close"].values
            df_oos["log_return"] = np.log(closes / np.roll(closes, 1))
            df_oos["log_return"].iloc[0] = 0.0
        df_oos["hurst"] = calcular_hurst(df_oos["log_return"].values, janela=100)
            
    res = rodar_oos_na_esteira(df_oos, params, estrategia, ativo, timeframe, "PASSADO", sufixo_ano, Path(especific_dir), param_id)
    if res and res.get('aprovado', False):
        return True
    return False

def run_monte_carlo(params, ativo, timeframe, estrategia, param_id, df_ops, dir_saida, df_comp=None) -> bool:
    from testes_robustez.monte_carlo.monte_carlo import rodar_monte_carlo
    # Cria pasta especifica: resultados_robustez/monte_carlo/estrategia/
    especific_dir = os.path.join(dir_saida, "monte_carlo", estrategia.lower())
    os.makedirs(especific_dir, exist_ok=True)
    
    # Salva csv temp
    csv_temp = os.path.join(especific_dir, f"temp_ops_{estrategia}.csv")
    df_ops.to_csv(csv_temp, index=False)
    
    res = rodar_monte_carlo(csv_temp, f"{estrategia}_{param_id}", ativo, especific_dir)
    if res and res.get('aprovado', False):
        return True
    return False

def run_spread(params, ativo, timeframe, estrategia, param_id, df_ops, dir_saida, df_comp=None) -> bool:
    from testes_robustez.spread.spread import rodar_spread
    especific_dir = os.path.join(dir_saida, "spread", estrategia.lower())
    os.makedirs(especific_dir, exist_ok=True)
    
    csv_temp = os.path.join(especific_dir, f"temp_ops_{estrategia}.csv")
    df_ops.to_csv(csv_temp, index=False)
    
    spread_orig = SPREAD_POR_ATIVO.get(ativo.lower(), 0.5)
    res = rodar_spread(csv_temp, f"{estrategia}_{param_id}", ativo, especific_dir, spread_original=spread_orig, spread_multiplo=1.8, pip_value_por_lot=10.0, max_degradacao=0.25)
    if res and res.get('aprovado', False):
        return True
    return False

def run_what_if(params, ativo, timeframe, estrategia, param_id, df_ops, dir_saida, df_comp=None) -> bool:
    from testes_robustez.what_if.what_if import rodar_what_if
    especific_dir = os.path.join(dir_saida, "what_if", estrategia.lower())
    os.makedirs(especific_dir, exist_ok=True)
    
    csv_temp = os.path.join(especific_dir, f"temp_ops_{estrategia}.csv")
    df_ops.to_csv(csv_temp, index=False)
    
    res = rodar_what_if(csv_temp, f"{estrategia}_{param_id}", ativo, especific_dir, pct_remocao=0.01, max_degradacao=0.25) # Remove top 1% trades
    if res and res.get('aprovado', False):
        return True
    return False

def run_distribuicao_parametros(params, ativo, timeframe, estrategia, param_id, df_ops, dir_saida, df_comp=None) -> bool:
    from testes_robustez.distribuicao_parametros.distribuicao_parametros import rodar_distribuicao_na_esteira
    especific_dir = os.path.join(dir_saida, "distribuicao_parametros", estrategia.lower())
    os.makedirs(especific_dir, exist_ok=True)
    
    if df_comp is None:
        logger.warning("DF completo nao fornecido para Distribuicao de Parametros.")
        return True
        
    logger.info(f"Executando Distribuicao de Parametros para params ID {param_id}")
    res = rodar_distribuicao_na_esteira(df_comp, params, estrategia, ativo, timeframe, Path(especific_dir), param_id)
    if res and res.get('aprovado', False):
        return True
    return False

def run_wfa(params, ativo, timeframe, estrategia, param_id, df_ops, dir_saida, df_comp=None) -> bool:
    from testes_robustez.w_f_a.walk_forward_matrix import rodar_wfm_na_esteira
    especific_dir = os.path.join(dir_saida, "w_f_a", estrategia.lower())
    os.makedirs(especific_dir, exist_ok=True)
    
    dir_data = DIR_PROJETO / f"quant_{ativo.lower()}_{timeframe.lower()}" / "data"
    parquet_otim = dir_data / "otimizacoes" / estrategia.lower() / f"otimizacao_{estrategia.lower()}_resultados.parquet"
    
    if df_comp is None or not parquet_otim.exists():
        logger.warning(f"WFA ignorado (dados incompletos).")
        return True
        
    df_comb = pd.read_parquet(parquet_otim)
    res = rodar_wfm_na_esteira(df_comp, df_comb, estrategia, ativo, timeframe, Path(especific_dir))
    if res and res.get('aprovado', False):
        return True
    return False

def run_permutacao(params, ativo, timeframe, estrategia, param_id, df_ops, dir_saida, df_comp=None) -> bool:
    from testes_robustez.permutacao.permutacao import rodar_permutacao_na_esteira
    especific_dir = os.path.join(dir_saida, "permutacao", estrategia.lower())
    os.makedirs(especific_dir, exist_ok=True)
    
    if df_comp is None:
        logger.warning("DF completo não fornecido para Permutacao.")
        return True
        
    meta_cols = {
        "id", "id_parametro", "Trades", "Lucro_Total_Pips", "Max_DD_Pips", 
        "Ret_DD", "Profit_Factor", "lucro", "drawdown", "Win_Rate", "Payoff"
    }
    params_filtrados = {k: v for k, v in params.items() if k not in meta_cols}
        
    res = rodar_permutacao_na_esteira(df_comp, params_filtrados, estrategia, ativo, timeframe, Path(especific_dir))
    if res and res.get('aprovado', False):
        return True
    return False

# Lista ordenada de testes (Degraus da Esteira - Fail Fast)
# OOS Passado e WFA removidos temporariamente a pedido do usuário
ESTEIRA_DEGRAUS = [
    ("OOS Futuro",              run_oos_futuro),
    ("What If",                 run_what_if),
    ("Spread",                  run_spread),
    ("Monte Carlo",             run_monte_carlo),
    ("Dist. Parametros",        run_distribuicao_parametros),
    ("Permutacao",              run_permutacao)
]


def gerar_relatorio_eliminacao(historico: dict, estrategia: str, resultados_dir: Path):
    """Gera um PNG dark-mode mostrando o resultado de cada variação em cada degrau da esteira."""
    os.makedirs(resultados_dir, exist_ok=True)
    degraus = [nome for nome, _ in ESTEIRA_DEGRAUS]
    variacoes = list(historico.keys())

    if not variacoes:
        logger.warning("Historico vazio, nao foi possivel gerar o relatorio PNG.")
        return

    n_var = len(variacoes)
    n_deg = len(degraus)

    # Paleta de cores
    COR_PASSOU  = "#2ecc71"   # verde
    COR_REPROV  = "#e74c3c"   # vermelho
    COR_PULADO  = "#4a4a5a"   # cinza escuro
    COR_BG      = "#0d0d1a"   # fundo muito escuro
    COR_TITULO  = "#a78bfa"   # roxo suave
    COR_TEXTO   = "#e2e8f0"   # branco acinzentado
    COR_GRID    = "#1e1e30"   # divisores sutis

    fig_w = max(12, n_deg * 1.8 + 3)
    fig_h = max(5, n_var * 0.65 + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(COR_BG)
    ax.set_facecolor(COR_BG)

    # Preencher células
    for r, var_id in enumerate(variacoes):
        hist_var = historico[var_id]
        pulou = False
        for c, degrau in enumerate(degraus):
            if pulou:
                cor = COR_PULADO
                texto = "—"
            else:
                resultado = hist_var.get(degrau, "PULADO")
                if resultado == "PASSOU":
                    cor = COR_PASSOU
                    texto = "✓"
                elif resultado == "REPROVOU":
                    cor = COR_REPROV
                    texto = "✗"
                    pulou = True  # Fail-Fast: próximos serão PULADO
                else:
                    cor = COR_PULADO
                    texto = "—"
                    pulou = True

            rect = mpatches.FancyBboxPatch(
                (c + 0.05, n_var - r - 0.95),
                0.90, 0.82,
                boxstyle="round,pad=0.02",
                linewidth=0,
                facecolor=cor,
                alpha=0.88
            )
            ax.add_patch(rect)
            ax.text(
                c + 0.5, n_var - r - 0.53, texto,
                ha='center', va='center',
                fontsize=13, fontweight='bold',
                color='white',
                fontfamily='DejaVu Sans'
            )

    # Rótulos das colunas (degraus)
    for c, degrau in enumerate(degraus):
        ax.text(
            c + 0.5, n_var + 0.3, degrau,
            ha='center', va='center',
            fontsize=9, fontweight='bold',
            color=COR_TITULO, rotation=20
        )

    # Rótulos das linhas (variações)
    for r, var_id in enumerate(variacoes):
        ax.text(
            -0.15, n_var - r - 0.53, var_id,
            ha='right', va='center',
            fontsize=8.5, color=COR_TEXTO
        )

    # Linhas de grade horizontais
    for r in range(n_var + 1):
        ax.axhline(n_var - r, color=COR_GRID, linewidth=0.6)
    for c in range(n_deg + 1):
        ax.axvline(c, color=COR_GRID, linewidth=0.6)

    # Legenda
    legend_handles = [
        mpatches.Patch(color=COR_PASSOU, label='Passou ✓'),
        mpatches.Patch(color=COR_REPROV, label='Reprovou ✗'),
        mpatches.Patch(color=COR_PULADO, label='Pulado —'),
    ]
    ax.legend(
        handles=legend_handles,
        loc='lower right',
        fontsize=8,
        framealpha=0.15,
        labelcolor=COR_TEXTO,
        facecolor=COR_BG,
        edgecolor=COR_GRID
    )

    ax.set_xlim(-2.0, n_deg)
    ax.set_ylim(-0.3, n_var + 0.9)
    ax.axis('off')

    # Título
    fig.suptitle(
        f"Esteira de Robustez — {estrategia.upper()}",
        fontsize=14, fontweight='bold',
        color=COR_TITULO, y=0.98
    )

    png_path = resultados_dir / f"{estrategia.lower()}_robustez.png"
    plt.tight_layout(rect=[0.12, 0, 1, 0.95])
    plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor=COR_BG)
    plt.close(fig)
    logger.info(f"Relatorio PNG de eliminacao salvo em: {png_path}")

def main():
    parser = argparse.ArgumentParser(description="Orquestrador de Esteira de Robustez - Fail Fast")
    parser.add_argument("--ativo", type=str, required=True, help="Nome do ativo (ex: EURUSD)")
    parser.add_argument("--timeframe", type=str, required=True, help="Timeframe (ex: H1)")
    parser.add_argument("--estrategia", type=str, required=True, help="Nome da estratégia (ex: CURVATURA)")
    parser.add_argument("--force_all", action="store_true", help="Força a execução de todos os testes apenas para a TOP1 para validação.")
    
    args = parser.parse_args()
    ativo = args.ativo.lower()
    timeframe = args.timeframe.lower()
    estrategia = args.estrategia.upper()
    
    base_dir = DIR_PROJETO / f"quant_{ativo}_{timeframe}"
    data_dir = base_dir / "data"
    
    parquet_otim = data_dir / "otimizacoes" / estrategia.lower() / f"otimizacao_{estrategia.lower()}_resultados.parquet"
    
    if not parquet_otim.exists():
        logger.error(f"Arquivo de parâmetros não encontrado: {parquet_otim}")
        sys.exit(1)
        
    df_otim = pd.read_parquet(parquet_otim)
    # Filtro básico (ignorar curvas com poucas operações)
    df_otim = df_otim[df_otim["Trades"] >= 60]
    if "Profit_Factor" in df_otim.columns and estrategia != "CURVATURA":
        df_otim = df_otim[df_otim["Profit_Factor"] >= 1.0]
    json_path = data_dir / "otimizacoes" / estrategia.lower() / f"otimizacao_{estrategia.lower()}_top10.json"
    if json_path.exists():
        with open(json_path, 'r') as f:
            top_params_list = json.load(f)
        logger.info(f"Carregado {len(top_params_list)} parametros do JSON customizado.")
    else:
        df_otim = df_otim.sort_values("Ret_DD", ascending=False)
        top_params_list = df_otim.head(10).to_dict(orient='records')
        
    if args.force_all:
        logger.info("⚠️ Modo DEBUG ativo: Rodando apenas a variação TOP1 e forçando execução completa de todos os testes!")
        top_params_list = top_params_list[:1]
        
    logger.info(f"[{ativo.upper()}_{timeframe.upper()} | {estrategia}] Iniciando Esteira. Procurando 10 variações aprovadas em até {len(top_params_list)} variações disponíveis.")
    
    # Pre-carregar DataFrames base para geração de sinais
    parquet_completo = data_dir / f"{ativo}_{timeframe}_completo.parquet"
    parquet_op = data_dir / f"{ativo}_{timeframe}_operacional.parquet"
    
    if not parquet_completo.exists() or not parquet_op.exists():
        # Fallback para nomes antigos
        parquet_completo = data_dir / f"eurusd_h1_{estrategia.lower()}.parquet" # Exemplo fallback
        
    logger.info(f"Carregando {parquet_completo}")
    df_comp = pd.read_parquet(parquet_completo)
    
    # Suporte dinâmico para colunas calculadas de indicadores que possam estar ausentes no completo
    if "hurst" not in df_comp.columns:
        parquet_hurst = data_dir / f"{ativo}_{timeframe}_hurst.parquet"
        if parquet_hurst.exists():
            logger.info(f"Carregando coluna 'hurst' ausente a partir de {parquet_hurst}")
            df_hurst = pd.read_parquet(parquet_hurst, columns=["hurst"])
            df_comp = df_comp.join(df_hurst, how="left")
            
    if "hurst" not in df_comp.columns or df_comp["hurst"].isna().any():
        logger.info("Calculando expoente de Hurst dinamicamente para dados completos...")
        if "log_return" not in df_comp.columns:
            closes = df_comp["Close"].values
            df_comp["log_return"] = np.log(closes / np.roll(closes, 1))
            df_comp["log_return"].iloc[0] = 0.0
        df_comp["hurst"] = calcular_hurst(df_comp["log_return"].values, janela=100)
    
    try:
        df_op = pd.read_parquet(parquet_op)
        mask_op = df_comp.index.isin(df_op.index)
    except:
        mask_op = np.ones(len(df_comp), dtype=bool) # Fallback assume tudo valido
    
    resultados_finais = []
    
    # Cache otimizado para não recalcular variaveis basicas
    cache_motor = {}
    
    resultados_dir = base_dir / "resultados_robustez"
    os.makedirs(resultados_dir, exist_ok=True)

    # Rastreamento para o relatório PNG de eliminação
    historico_testes = {}  # {param_id: {nome_degrau: "PASSOU" | "REPROVOU" | "PULADO"}}
    
    aprovadas_count = 0
    sobreviventes = []

    for i, param_set in enumerate(top_params_list):
        if aprovadas_count >= 10:
            logger.info("🎯 10 variações aprovadas encontradas! Encerrando esteira.")
            break
            
        param_id = param_set.get('id', param_set.get('id_parametro', f"{estrategia}_VAR{i+1}"))
        logger.info(f"\n--- Processando Variação ID: {param_id} ({i+1}/{len(top_params_list)}) | Aprovadas: {aprovadas_count}/10 ---")
        
        # 1. Gerar Sinais com o Motor da Distribuição de Parâmetros
        sinal, sl_pips, tp_pips = calcular_sinais(
            df=df_comp, 
            params=param_set, 
            janela_op=mask_op, 
            estrategia=estrategia, 
            cache=cache_motor,
            ativo=ativo,
            timeframe=timeframe
        )
        
        # Preparar DF para simulação
        df_sim = df_comp.copy()
        df_sim["sinal"] = sinal
        df_sim[f"sl_pips_{estrategia.lower()}"] = sl_pips
        df_sim[f"tp_pips_{estrategia.lower()}"] = tp_pips
        
        # Inicializar rastreamento desta variação para aparecer no PNG
        historico_testes[param_id] = {}
        
        # 2. Backtest Dinâmico Isolado
        ops = simular_estrategia(
            df_sim, 
            "sinal", 
            estrategia,
            usar_zscore_exit=(estrategia == "ZSCORE"), 
            usar_ou_exit=(estrategia in ["OU", "OU_REVERSO", "HAWKES"]),
            usar_wavelet_exit=(estrategia == "WAVELET"),
            usar_curvatura_exit=(estrategia == "CURVATURA")
        )
        
        if len(ops) < 30:
            logger.warning(f"❌ Variação ID {param_id} REPROVADA (Menos de 30 trades gerados na amostra inteira).")
            # Marca como falha no primeiro degrau para constar no PNG
            primeiro_degrau = ESTEIRA_DEGRAUS[0][0] if ESTEIRA_DEGRAUS else "In-Sample"
            historico_testes[param_id][primeiro_degrau] = "REPROVOU"
            continue
            
        # Converter para DataFrame p/ export temp
        registros = []
        for op in ops:
            registros.append({
                "id": op.id,
                "estrategia": op.estrategia,
                "direcao": "LONG" if op.direcao == 1 else "SHORT",
                "entrada_dt": op.entrada_dt,
                "entrada_preco": round(op.entrada_preco, 5),
                "saida_dt": op.saida_dt,
                "saida_preco": round(op.saida_preco, 5) if op.saida_preco else None,
                "motivo_saida": op.motivo_saida,
                "sl_pips": round(op.sl_pips, 2),
                "tp_pips": round(op.tp_pips, 2),
                "lot_size": round(op.lot_size, 4),
                "pnl_pips": round(op.pnl_pips, 2),
                "pnl_monetario": round(op.pnl_monetario, 2),
                "duracao_candles": op.duracao_candles,
                "capital_entrada": round(op.capital_entrada, 2),
            })
        df_ops = pd.DataFrame(registros)
        
        aprovado = True
        
        # Criar pasta temporária exclusiva para esta variação de parâmetro
        dir_saida_temp = resultados_dir / f"temp_{param_id}"
        os.makedirs(dir_saida_temp, exist_ok=True)

        
        # 3. Esteira Fail-Fast (Desabilitável via --force_all)
        for nome_teste, func_teste in ESTEIRA_DEGRAUS:
            passou = func_teste(param_set, ativo, timeframe, estrategia, param_id, df_ops, str(dir_saida_temp), df_comp)
            
            if not passou:
                historico_testes[param_id][nome_teste] = "REPROVOU"
                if args.force_all:
                    logger.warning(f"❌ [DEBUG] Variação ID {param_id} seria REPROVADA no teste: {nome_teste}. Continuando devido a --force_all.")
                else:
                    logger.warning(f"❌ Variação ID {param_id} REPROVADA no teste: {nome_teste}. Interrompendo esteira (Fail-Fast).")
                    aprovado = False
                    break
            else:
                historico_testes[param_id][nome_teste] = "PASSOU"
                
        # Função interna de mover arquivos de forma flexível
        import shutil
        def mover_conteudo(src, dst, mover_apenas_imagens=False):
            if not os.path.exists(src):
                return
            for item in os.listdir(src):
                s = os.path.join(src, item)
                d = os.path.join(dst, item)
                if os.path.isdir(s):
                    os.makedirs(d, exist_ok=True)
                    mover_conteudo(s, d, mover_apenas_imagens)
                    try:
                        if not os.listdir(s):
                            os.rmdir(s)
                    except:
                        pass
                else:
                    ext = os.path.splitext(item)[1].lower()
                    eh_imagem = ext in ['.png', '.jpg', '.jpeg']
                    if not mover_apenas_imagens or eh_imagem:
                        if os.path.exists(d):
                            try:
                                os.remove(d)
                            except:
                                pass
                        shutil.move(s, d)

        if aprovado or args.force_all:
            if args.force_all:
                logger.info(f"✅ [DEBUG] Finalizando validação. Forçando mover completo para {param_id}.")
            else:
                logger.info(f"✅ Variação ID {param_id} APROVADA em todos os {len(ESTEIRA_DEGRAUS)} testes de robustez!")
                aprovadas_count += 1
                
            # Mover imagens e CSVs de resultados que passaram para resumo_aprovadas
            dir_aprovadas = resultados_dir / "resumo_aprovadas" / estrategia.lower() / param_id
            os.makedirs(dir_aprovadas, exist_ok=True)
            mover_conteudo(str(dir_saida_temp), str(dir_aprovadas))
            
            # Remover pasta temporária
            import shutil
            shutil.rmtree(dir_saida_temp, ignore_errors=True)
            
            # Adicionar aos sobreviventes
            sobreviventes.append(param_set)
        else:
            logger.info(f"❌ Variação ID {param_id} REPROVADA. Movendo imagens para resumo_eliminadas.")
            dir_eliminadas = resultados_dir / "resumo_eliminadas" / estrategia.lower() / param_id
            os.makedirs(dir_eliminadas, exist_ok=True)
            # Move apenas os PNGs para não lotar de CSVs pesados
            mover_conteudo(str(dir_saida_temp), str(dir_eliminadas), mover_apenas_imagens=True)
            import shutil
            shutil.rmtree(dir_saida_temp, ignore_errors=True)
            
    logger.info(f"\nResumo: {len(sobreviventes)}/10 variações aprovadas preenchidas na esteira.")
    
    if sobreviventes:
        # Exporta as top variações sobreviventes (até 10)
        output_json = resultados_dir / f"{estrategia.lower()}_sobreviventes.json"
        with open(output_json, "w") as f:
            json.dump(sobreviventes, f, indent=4)
        logger.info(f"Sobreviventes salvos em: {output_json}")

    # Gerar relatório PNG de eliminação
    gerar_relatorio_eliminacao(historico_testes, estrategia, resultados_dir / estrategia.lower())

if __name__ == "__main__":
    main()
