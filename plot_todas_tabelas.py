import matplotlib.pyplot as plt
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DIR_GRAFICOS = Path("c:/Users/cesar/.gemini/antigravity/scratch/Quant_Matematica.Trade/quant_eurusd/graficos")
DIR_GRAFICOS.mkdir(parents=True, exist_ok=True)

params_map = {
    'zscore': {'Estratégia': 'Z-Score Mean Reversion', 'Regime Operacional': 'Reversão', 'Hurst Filtro': '< 0.40', 'Janela Z-Score': 50, 'Janela Volatilidade': 50, 'Stop Loss (Risco)': '2.0x Vol', 'Take Profit (Alvo)': '3.0x Vol', 'Gatilho Retorno': '2.0 e -2.0', 'Horário': '10h as 22h30'},
    'momentum': {'Estratégia': 'Momentum & Entropia', 'Regime Operacional': 'Tendência', 'Hurst Filtro': '> 0.50', 'Entropia Janela': 30, 'Entropia Operável': '< 0.60', 'Bloqueio Caos': '> 0.80', 'Aceleração Alvo': '> 0.80 ou < 0.20', 'Stop Loss (Risco)': '2.0x Vol', 'Take Profit (Alvo)': '5.0x Vol'},
    'hawkes': {'Estratégia': 'Processos de Hawkes', 'Regime Operacional': 'Tendência', 'Hurst Filtro': '> 0.55', 'Kappa (Decaimento)': 0.1, 'Janela Intensidade': 50, 'Limiar de Excitação': 1.5, 'Stop Loss (Risco)': '1.5x Vol', 'Take Profit (Alvo)': '3.0x Vol'},
    'ou': {'Estratégia': 'Ornstein-Uhlenbeck', 'Regime Operacional': 'Reversão', 'Hurst Filtro': '< 0.45', 'Meia-Vida': 'Dinâmica', 'Limiar Z-Score OU': 2.5, 'Stop Loss (Risco)': '2.0x Vol', 'Take Profit (Alvo)': '3.0x Vol'},
    'pca': {'Estratégia': 'PCA Arbitrage', 'Componentes Principais': 3, 'Janela PCA': 100, 'Desvio Ativação': 2.5, 'Stop Loss (Risco)': '2.0x Vol', 'Take Profit (Alvo)': '4.0x Vol'},
    'wavelet': {'Estratégia': 'Wavelet Transform', 'Tipo Wavelet': 'db4', 'Nível Decomposição': 4, 'Sinal Filtro': 'Detalhe nível 4', 'Stop Loss (Risco)': '1.5x Vol', 'Take Profit (Alvo)': '3.0x Vol'},
    'curvatura': {'Estratégia': 'Curvatura Diferencial', 'Filtro Passa-Baixa': 'Butterworth', 'Janela Curvatura': 20, 'Ativação (Threshold)': 0.05, 'Stop Loss (Risco)': '2.0x Vol', 'Take Profit (Alvo)': '4.0x Vol'}
}

def gerar_tabela(nome, params):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    dados = [[k, str(v)] for k, v in params.items()]
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
                
    fig.suptitle(f"Configuração de Hiperparâmetros — {nome.upper()}", color="#FFFFFF", fontsize=16, fontweight="bold", y=0.95)
    plt.tight_layout()
    caminho = DIR_GRAFICOS / f"parametros_{nome}_eurusd.png"
    plt.savefig(caminho, dpi=150, facecolor="#121212")
    plt.close()
    logger.info(f"Tabela gerada: {caminho}")

for nome, p in params_map.items():
    gerar_tabela(nome, p)
