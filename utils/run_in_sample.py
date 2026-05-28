import subprocess

scripts = [
    "quant_eurusd/zscore_eurusd.py",
    "quant_eurusd/momentum_eurusd.py",
    "quant_eurusd/hawkes_strategy_eurusd.py",
    "quant_eurusd/ou_strategy_eurusd.py",
    "quant_eurusd/pca_strategy_eurusd.py",
    "quant_eurusd/wavelet_strategy_eurusd.py",
    "quant_eurusd/curvatura_strategy_eurusd.py",
    "quant_eurusd/backtest_individual.py",
    "quant_eurusd/backtest.py"
]

for script in scripts:
    print(f"\n[{'='*50}]")
    print(f"Rodando {script}...")
    print(f"[{'='*50}]\n")
    try:
        subprocess.run(["python", script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Erro ao executar {script}: {e}")
        break
print("\n\n[SUCESSO] Tudo finalizado com sucesso!")
