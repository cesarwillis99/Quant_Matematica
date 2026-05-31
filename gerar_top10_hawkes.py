import json
with open('quant_eurusd_h1/data/otimizacoes/candidatos_testes/candidatos_hawkes.json', 'r') as f:
    cand = json.load(f)
indices = [1, 3, 8, 20, 9, 13, 22, 23, 28, 37]
sel = [cand[i-1] for i in indices]
with open('quant_eurusd_h1/data/otimizacoes/hawkes/otimizacao_hawkes_top10.json', 'w') as f:
    json.dump(sel, f, indent=4)
