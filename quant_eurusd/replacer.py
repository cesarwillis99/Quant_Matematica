import sys
import re

with open('backtest_individual.py', 'r', encoding='utf-8') as f:
    content = f.read()

new_process = '''def processar_pipeline_backtest(estrategia: str):
    estrategia = estrategia.upper()
    logger.info(f"Iniciando pipeline de backtest individual para a estratégia {estrategia}...")
    
    # Identificar o parquet correto
    parquet_map = {
        "ZSCORE": PARQUET_ZSCORE,
        "MOMENTUM": PARQUET_MOMENTUM,
        "OU": PARQUET_OU
    }
    
    caminho_parquet = parquet_map.get(estrategia, DIR_PROJETO_V2 / "data" / f"eurusd_h1_{estrategia.lower()}.parquet")
    
    if not caminho_parquet.exists():
        raise FileNotFoundError(f"Arquivo parquet não encontrado: {caminho_parquet.name}")
        
    logger.info(f"Carregando dados: {caminho_parquet.name}")
    df = pd.read_parquet(caminho_parquet, engine="pyarrow")
    
    # Padronizar colunas para o simulador
    col_sinal = f"sinal_{estrategia.lower()}"
    if col_sinal not in df.columns:
        if "sinal" in df.columns:
            col_sinal = "sinal"
        else:
            raise ValueError(f"Coluna de sinal {col_sinal} não encontrada no dataframe.")
            
    # Criar colunas temporárias para compatibilidade com simular_estrategia
    df[f"sl_pips_{estrategia.lower()}"] = df["sl_pips"] if "sl_pips" in df.columns else df.get(f"sl_pips_{estrategia.lower()}", 0)
    df[f"tp_pips_{estrategia.lower()}"] = df["tp_pips"] if "tp_pips" in df.columns else df.get(f"tp_pips_{estrategia.lower()}", 0)
    
    logger.info(f"Simulando {len(df):,} candles H1 no período de backtest.")
    
    ops = simular_estrategia(
        df, 
        col_sinal, 
        estrategia, 
        usar_zscore_exit=(estrategia == "ZSCORE"), 
        usar_ou_exit=(estrategia == "OU")
    )
    
    eq = construir_equity_curve(ops, df.index)
    
    equity_curves = {estrategia: eq}
    todas_operacoes = {estrategia: ops}
    
    print(f"\\n" + "█" * 70)
    print(f"█   MÉTRICAS DE PERFORMANCE ({estrategia})  █")
    print("█" * 70)
    
    m = calcular_metricas_performance(ops, eq, estrategia)
    todas_metricas = {estrategia: m}
    exibir_metricas(m)
    
    executar_analise_periodos(df, todas_operacoes)
    
    caminhos_grafico = [DIR_PROJETO_V2 / "graficos" / f"equity_curve_{estrategia.lower()}.png"]
    gerar_grafico_equity_curves(equity_curves, todas_operacoes, caminhos_grafico)
    
    caminhos_ops = [DIR_PROJETO_V2 / "resultados" / f"operacoes_{estrategia.lower()}.csv"]
    caminhos_met = [DIR_PROJETO_V2 / "resultados" / f"metricas_{estrategia.lower()}.csv"]
    salvar_arquivos_resultados(todas_operacoes, todas_metricas, caminhos_ops, caminhos_met)
    
    print(f"\\n" + "█" * 75)
    print("█   RESUMO RÁPIDO")
    print("█" + "─" * 73)
    print(
        f"  {estrategia:<11} | {m['total_operacoes']:>4} ops | "
        f"{m['win_rate']:>6.2f}% WR | "
        f"${m['pnl_total_usd']:>+9.2f} ({m['pnl_total_pct']:>+5.2f}%) | "
        f"{m['drawdown_max_pct']:>7.2f}% DD | "
        f"{m['sharpe_ratio']:>+6.3f} SR"
    )
    print("█" * 75 + "\\n")

# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backtester Manual Individual EURUSD H1",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--estrategia", type=str, required=True, help="Nome da estratégia (ex: ZSCORE, MOMENTUM, OU, NOVA)")
    args = parser.parse_args()
    
    print("\\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print(f"█   INICIANDO BACKTEST INDIVIDUAL: {args.estrategia.upper():<33} █")
    print("█   Capital Inicial: $10.000 | Risco por operação: 1%          █")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    
    try:
        processar_pipeline_backtest(args.estrategia)
        print("✅ Simulação Histórica Individual concluída com sucesso!\\n")
    except Exception as e:
        logger.exception("Erro crítico durante a execução do backtest:")
        sys.exit(1)
'''

content = re.sub(r'def processar_pipeline_backtest\(\):.*', new_process, content, flags=re.DOTALL)

with open('backtest_individual.py', 'w', encoding='utf-8') as f:
    f.write(content)
