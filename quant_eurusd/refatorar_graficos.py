import os
import re
from pathlib import Path

params_map = {
    "zscore_eurusd.py": {"Estratégia": "Z-Score Mean Reversion", "Regime Operacional": "Reversão", "Hurst Filtro": "< 0.40", "Janela Z-Score": 50, "Janela Volatilidade": 50, "Stop Loss (Risco)": "2.0x Vol", "Take Profit (Alvo)": "3.0x Vol", "Gatilho Retorno": "2.0 e -2.0", "Horário": "10h as 22h30"},
    "momentum_eurusd.py": {"Estratégia": "Momentum & Entropia", "Regime Operacional": "Tendência", "Hurst Filtro": "> 0.50", "Entropia Janela": 30, "Entropia Operável": "< 0.60", "Bloqueio Caos": "> 0.80", "Aceleração Alvo": "> 0.80 ou < 0.20", "Stop Loss (Risco)": "2.0x Vol", "Take Profit (Alvo)": "5.0x Vol"},
    "hawkes_strategy_eurusd.py": {"Estratégia": "Processos de Hawkes", "Regime Operacional": "Tendência", "Hurst Filtro": "> 0.55", "Kappa (Decaimento)": 0.1, "Janela Intensidade": 50, "Limiar de Excitação": 1.5, "Stop Loss (Risco)": "1.5x Vol", "Take Profit (Alvo)": "3.0x Vol"},
    "ou_strategy_eurusd.py": {"Estratégia": "Ornstein-Uhlenbeck", "Regime Operacional": "Reversão", "Hurst Filtro": "< 0.45", "Meia-Vida": "Dinâmica", "Limiar Z-Score OU": 2.5, "Stop Loss (Risco)": "2.0x Vol", "Take Profit (Alvo)": "3.0x Vol"},
    "pca_strategy_eurusd.py": {"Estratégia": "PCA Arbitrage", "Componentes Principais": 3, "Janela PCA": 100, "Desvio Ativação": 2.5, "Stop Loss (Risco)": "2.0x Vol", "Take Profit (Alvo)": "4.0x Vol"},
    "wavelet_strategy_eurusd.py": {"Estratégia": "Wavelet Transform", "Tipo Wavelet": "db4", "Nível Decomposição": 4, "Sinal Filtro": "Detalhe nível 4", "Stop Loss (Risco)": "1.5x Vol", "Take Profit (Alvo)": "3.0x Vol"},
    "curvatura_strategy_eurusd.py": {"Estratégia": "Curvatura Diferencial", "Filtro Passa-Baixa": "Butterworth", "Janela Curvatura": 20, "Ativação (Threshold)": 0.05, "Stop Loss (Risco)": "2.0x Vol", "Take Profit (Alvo)": "4.0x Vol"}
}

base_dir = Path("c:/Users/cesar/.gemini/antigravity/scratch/Quant_Matematica.Trade/quant_eurusd")

for file in os.listdir(base_dir):
    if file.endswith("_eurusd.py") and file != "hurst_eurusd.py":
        filepath = base_dir / file
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        # 1. Comentar a chamada original gerar_grafico_*_sinais(df_final)
        content = re.sub(r'([ \t]+)(gerar_grafico_[a-z0-9_]+_sinais\(.*?\))', r'\1# \2', content)
        
        # 2. Injetar função de gerar tabela se já não tiver sido injetada
        if "def gerar_tabela_parametros" not in content:
            estr_name = file.split("_")[0].upper()
            params = params_map.get(file, {"Estratégia": estr_name})
            
            func_code = f'''
def gerar_tabela_parametros():
    import matplotlib.pyplot as plt
    logger.info("Gerando gráfico da tabela de parâmetros em Dark Mode...")
    DIR_GRAFICOS.mkdir(parents=True, exist_ok=True)
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    dados = [[k, str(v)] for k, v in {params}.items()]
    tabela = ax.table(cellText=dados, colLabels=["Métrica", "Valor Otimizado"], loc="center", cellLoc="left")
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(12)
    tabela.scale(1.2, 2.0)
    
    for (row, col), cell in tabela.get_celld().items():
        cell.set_edgecolor("#333333")
        if row == 0:
            cell.set_text_props(weight="bold", color="#FFFFFF", fontsize=13)
            cell.set_facecolor("#1E4F8A")
        else:
            cell.set_facecolor("#121212")
            cell.set_text_props(color="#CFD8DC")
            if col == 0:
                cell.set_text_props(weight="bold", color="#82AAFF")
                
    fig.suptitle("Configuração de Hiperparâmetros — {estr_name}", color="#FFFFFF", fontsize=16, fontweight="bold", y=0.95)
    plt.tight_layout()
    caminho = DIR_GRAFICOS / "parametros_{estr_name.lower()}.png"
    plt.savefig(caminho, dpi=150, facecolor="#121212")
    plt.close()
    logger.info(f"Tabela de parâmetros salva em: {{caminho}}")
'''
            # Injetar a definição da função logo antes do "# PIPELINE PRINCIPAL"
            if "# PIPELINE PRINCIPAL" in content:
                content = content.replace("# PIPELINE PRINCIPAL", func_code + "\n# PIPELINE PRINCIPAL")
            
            # 3. Adicionar chamada à nova função ao invés do gerar_grafico
            # Localizar 'imprimir_relatorio_X' e anexar 'gerar_tabela_parametros()'
            content = re.sub(r'([ \t]+)(imprimir_relatorio_[a-z0-9_]+\(.*?\))', r'\1\2\n\1gerar_tabela_parametros()', content)
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Modificado {file}")
