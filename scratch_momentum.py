import json
import os

json_file = r'c:\Users\cesar\.gemini\antigravity\scratch\Quant_Matematica.Trade\quant_eurusd_h1\data\otimizacoes\candidatos_testes\candidatos_momentum.json'
saida_file = r'C:\Users\cesar\.gemini\antigravity-ide\brain\a675ab6b-b9e9-478f-8b51-99a569a7a022\top50_momentum.md'

with open(json_file, 'r') as f:
    data = json.load(f)

markdown = '# Top 50 - Estratégia Momentum (EURUSD H1)\n\n'
markdown += 'Aqui estão as 50 melhores parametrizações da estratégia de Momentum, focada em capturar tendências fortes após a quebra do limiar direcional de Hurst e Entropia.\n\n'

headers = ['Rank', 'Trades', 'Win_Rate', 'Payoff', 'Profit_Factor', 'PnL Pips', 'Max_DD', 'Ret_DD', 'Hurst Cutoff', 'Entropia Cutoff', 'Percentil Trigger', 'Mult. SL', 'Mult. TP']
markdown += '| ' + ' | '.join(headers) + ' |\n'
markdown += '|' + '|'.join([':---' for _ in headers]) + '|\n'

for i, row in enumerate(data[:50]):
    line = [
        f'**{i+1}**',
        str(int(row['Trades'])),
        f"{row['Win_Rate']} %",
        str(row['Payoff']),
        str(row['Profit_Factor']),
        str(row['Lucro_Total_Pips']),
        str(row['Max_DD_Pips']),
        f"**{row['Ret_DD']}**",
        str(row['hurst_cutoff']),
        str(row['entropia_cutoff']),
        str(row['percentil_trigger']),
        str(row['mult_sl']),
        str(row['mult_tp'])
    ]
    markdown += '| ' + ' | '.join(line) + ' |\n'

os.makedirs(os.path.dirname(saida_file), exist_ok=True)
with open(saida_file, 'w', encoding='utf-8') as f:
    f.write(markdown)

print('Artifact top50_momentum.md gerado com sucesso.')
