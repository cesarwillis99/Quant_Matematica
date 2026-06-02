import pandas as pd
import os

csv_file = r'c:\Users\cesar\.gemini\antigravity\scratch\Quant_Matematica.Trade\quant_eurusd_h1\data\otimizacoes\ou_reverso\relatorios_top50\relatorio_top50_ou_reverso.csv'
saida_file = r'C:\Users\cesar\.gemini\antigravity-ide\brain\a675ab6b-b9e9-478f-8b51-99a569a7a022\top50_ou_reverso.md'

df = pd.read_csv(csv_file).head(50)

markdown = '# Top 50 - Estratégia OU Reverso (EURUSD H1)\n\n'
markdown += 'Aqui estão as 50 melhores parametrizações da versão Reversa do Ornstein-Uhlenbeck (operando a favor do rompimento). O Fator de Recuperação atingiu valores espetaculares (até 7.14x)!\n\n'

headers = ['Rank', 'Trades', 'Win_Rate', 'Payoff', 'Profit_Factor', 'PnL Pips', 'Max_DD', 'Ret_DD', 'Janela_OU', 'HalfLife_Max', 'Z-Score', 'Mult. SL', 'Mult. TP', 'Saída Neutra']
markdown += '| ' + ' | '.join(headers) + ' |\n'
markdown += '|' + '|'.join([':---' for _ in headers]) + '|\n'

for i, row in df.iterrows():
    usar_sn = 'SIM' if row.get('usar_saida_neutra', 0) == 1 else 'NÃO'
    line = [
        f'**{i+1}**',
        str(int(row['Trades'])),
        f"{row['Win_Rate']} %",
        str(row['Payoff']),
        str(row['Profit_Factor']),
        str(row['Lucro_Total_Pips']),
        str(row['Max_DD_Pips']),
        f"**{row['Ret_DD']}**",
        str(int(row['janela_ou'])),
        str(row['halflife_max']),
        str(row['zscore_threshold']),
        str(row['mult_sl']),
        str(row['mult_tp']),
        usar_sn
    ]
    markdown += '| ' + ' | '.join(line) + ' |\n'

os.makedirs(os.path.dirname(saida_file), exist_ok=True)
with open(saida_file, 'w', encoding='utf-8') as f:
    f.write(markdown)

print('Artifact top50_ou_reverso.md gerado com sucesso.')
