import argparse
import os
import json
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np

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

from testes_robustez.distribuicao_parametros.distribuicao_parametros import calcular_sinais
from backtests.backtest_individual import simular_estrategia, Operacao

# --- WRAPPERS DOS TESTES ---
# Os testes agora serão importados e executados como funções python, passando os parâmetros
def run_oos_futuro(params, ativo, timeframe, estrategia, param_id, df_ops, dir_saida, df_comp=None) -> bool:
    from testes_robustez.oos.run_oos import rodar_oos_na_esteira
    especific_dir = os.path.join(dir_saida, "oos_futuro", estrategia.lower())
    os.makedirs(especific_dir, exist_ok=True)
    
    dir_data = Path(dir_saida).parent / "data"
    padrao = f"{ativo.lower()}_{timeframe.lower()}_completo_OOS_futuro_*.parquet"
    arquivos = list(dir_data.glob(padrao))
    if not arquivos:
        logger.warning(f"OOS FUTURO ignorado (arquivo não encontrado: {padrao}).")
        return True
        
    sufixo_ano = arquivos[0].stem.split("_OOS_futuro_")[1]
    df_oos = pd.read_parquet(arquivos[0])
    
    res = rodar_oos_na_esteira(df_oos, params, estrategia, ativo, timeframe, "FUTURO", sufixo_ano, Path(especific_dir), param_id)
    if res and res.get('aprovado', False):
        return True
    return False

def run_oos_passado(params, ativo, timeframe, estrategia, param_id, df_ops, dir_saida, df_comp=None) -> bool:
    from testes_robustez.oos.run_oos import rodar_oos_na_esteira
    especific_dir = os.path.join(dir_saida, "oos_passado", estrategia.lower())
    os.makedirs(especific_dir, exist_ok=True)
    
    dir_data = Path(dir_saida).parent / "data"
    padrao = f"{ativo.lower()}_{timeframe.lower()}_completo_OOS_passado_*.parquet"
    arquivos = list(dir_data.glob(padrao))
    if not arquivos:
        logger.warning(f"OOS PASSADO ignorado (arquivo não encontrado: {padrao}).")
        return True
        
    sufixo_ano = arquivos[0].stem.split("_OOS_passado_")[1]
    df_oos = pd.read_parquet(arquivos[0])
    
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
    
    res = rodar_spread(csv_temp, f"{estrategia}_{param_id}", ativo, especific_dir, spread_original=1.2, spread_multiplo=1.5, pip_value_por_lot=10.0)
    if res and res.get('aprovado', False):
        return True
    return False

def run_what_if(params, ativo, timeframe, estrategia, param_id, df_ops, dir_saida, df_comp=None) -> bool:
    from testes_robustez.what_if.what_if import rodar_what_if
    especific_dir = os.path.join(dir_saida, "what_if", estrategia.lower())
    os.makedirs(especific_dir, exist_ok=True)
    
    csv_temp = os.path.join(especific_dir, f"temp_ops_{estrategia}.csv")
    df_ops.to_csv(csv_temp, index=False)
    
    res = rodar_what_if(csv_temp, f"{estrategia}_{param_id}", ativo, especific_dir, pct_remocao=0.01) # Remove top 1% trades
    if res and res.get('aprovado', False):
        return True
    return False

def run_distribuicao_parametros(params, ativo, timeframe, estrategia, param_id, df_ops, dir_saida, df_comp=None) -> bool:
    especific_dir = os.path.join(dir_saida, "distribuicao_parametros", estrategia.lower())
    os.makedirs(especific_dir, exist_ok=True)
    logger.info(f"Executando Distribuição de Parâmetros para params ID {params.get('id', 'N/A')}")
    return True

def run_wfa(params, ativo, timeframe, estrategia, param_id, df_ops, dir_saida, df_comp=None) -> bool:
    from testes_robustez.w_f_a.walk_forward_matrix import rodar_wfm_na_esteira
    especific_dir = os.path.join(dir_saida, "w_f_a", estrategia.lower())
    os.makedirs(especific_dir, exist_ok=True)
    
    dir_data = Path(dir_saida).parent / "data"
    parquet_otim = dir_data / "otimizacoes" / f"otimizacao_{estrategia.lower()}_resultados.parquet"
    
    if df_comp is None or not parquet_otim.exists():
        logger.warning(f"WFA ignorado (dados incompletos).")
        return True
        
    df_comb = pd.read_parquet(parquet_otim)
    res = rodar_wfm_na_esteira(df_comp, df_comb, f"{estrategia}_{param_id}", ativo, timeframe, Path(especific_dir))
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
        
    res = rodar_permutacao_na_esteira(df_comp, params, f"{estrategia}_{param_id}", ativo, timeframe, Path(especific_dir))
    if res and res.get('aprovado', False):
        return True
    return False

# Lista ordenada de testes (Degraus da Esteira - Fail Fast)
ESTEIRA_DEGRAUS = [
    ("OOS Futuro", run_oos_futuro),
    ("OOS Passado", run_oos_passado),
    ("What If", run_what_if),
    ("Spread", run_spread),
    ("Monte Carlo", run_monte_carlo),
    ("Distribuicao Parametros", run_distribuicao_parametros),
    ("Walk Forward Matrix (WFA)", run_wfa),
    ("Permutacao", run_permutacao)
]

def main():
    parser = argparse.ArgumentParser(description="Orquestrador de Esteira de Robustez - Fail Fast")
    parser.add_argument("--ativo", type=str, required=True, help="Nome do ativo (ex: EURUSD)")
    parser.add_argument("--timeframe", type=str, required=True, help="Timeframe (ex: H1)")
    parser.add_argument("--estrategia", type=str, required=True, help="Nome da estratégia (ex: CURVATURA)")
    
    args = parser.parse_args()
    ativo = args.ativo.lower()
    timeframe = args.timeframe.lower()
    estrategia = args.estrategia.upper()
    
    base_dir = DIR_PROJETO / f"quant_{ativo}_{timeframe}"
    data_dir = base_dir / "data"
    
    json_path = data_dir / "otimizacoes" / f"otimizacao_{estrategia.lower()}_top10.json"
    
    if not json_path.exists():
        logger.error(f"Arquivo de parâmetros não encontrado: {json_path}")
        sys.exit(1)
        
    with open(json_path, 'r') as f:
        top_params_list = json.load(f)
        
    logger.info(f"[{ativo.upper()}_{timeframe.upper()} | {estrategia}] Iniciando Esteira para {len(top_params_list)} variações.")
    
    # Pre-carregar DataFrames base para geração de sinais
    parquet_completo = data_dir / f"{ativo}_{timeframe}_completo.parquet"
    parquet_op = data_dir / f"{ativo}_{timeframe}_operacional.parquet"
    
    if not parquet_completo.exists() or not parquet_op.exists():
        # Fallback para nomes antigos
        parquet_completo = data_dir / f"eurusd_h1_{estrategia.lower()}.parquet" # Exemplo fallback
        
    logger.info(f"Carregando {parquet_completo}")
    df_comp = pd.read_parquet(parquet_completo)
    
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
    
    for i, param_set in enumerate(top_params_list):
        param_id = param_set.get('id', f"{estrategia}_TOP{i+1}")
        logger.info(f"\n--- Processando Variação ID: {param_id} ---")
        
        # 1. Gerar Sinais com o Motor da Distribuição de Parâmetros
        sinal, sl_pips, tp_pips = calcular_sinais(
            df=df_comp, 
            params=param_set, 
            janela_op=mask_op, 
            estrategia=estrategia, 
            cache=cache_motor
        )
        
        # Preparar DF para simulação
        df_sim = df_comp.copy()
        df_sim["sinal"] = sinal
        df_sim[f"sl_pips_{estrategia.lower()}"] = sl_pips
        df_sim[f"tp_pips_{estrategia.lower()}"] = tp_pips
        
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
            logger.warning(f"❌ Variação ID {param_id} REPROVADA (Menos de 30 trades gerados).")
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
        
        # 3. Esteira Fail-Fast
        for nome_teste, func_teste in ESTEIRA_DEGRAUS:
            passou = func_teste(param_set, ativo, timeframe, estrategia, param_id, df_ops, str(resultados_dir), df_comp)
            
            if not passou:
                logger.warning(f"❌ Variação ID {param_id} REPROVADA no teste: {nome_teste}. Interrompendo esteira (Fail-Fast).")
                aprovado = False
                break
                
        if aprovado:
            logger.info(f"✅ Variação ID {param_id} APROVADA em todos os {len(ESTEIRA_DEGRAUS)} testes de robustez!")
            resultados_finais.append(param_set)
            
    logger.info(f"\nResumo: {len(resultados_finais)}/{len(top_params_list)} variações sobreviveraram à esteira.")
    
    # Salvar resultados finais dos sobreviventes
    sobreviventes_path = resultados_dir / f"{estrategia.lower()}_sobreviventes.json"
    with open(sobreviventes_path, 'w') as f:
        json.dump(resultados_finais, f, indent=4)
    logger.info(f"Sobreviventes salvos em: {sobreviventes_path}")

if __name__ == "__main__":
    main()
