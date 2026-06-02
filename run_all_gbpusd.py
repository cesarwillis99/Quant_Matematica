import subprocess
import sys
import os
from pathlib import Path

scripts = [
    "otimizacoes/otimizacao_zscore.py",
    "otimizacoes/otimizacao_hawkes.py",
    "otimizacoes/otimizacao_wavelet.py",
    "otimizacoes/otimizacao_pca.py",
    "otimizacoes/otimizacao_ou.py",
    "otimizacoes/otimizacao_ou_reverso.py",
    "otimizacoes/otimizacao_momentum.py"
]

ativo = "gbpusd"
timeframe = "h1"

print(f"[{'='*50}]")
print(f"Iniciando Lote de Otimizacao {ativo.upper()} {timeframe.upper()}")
print(f"[{'='*50}]\n")

for script in scripts:
    print(f"\n[{'='*50}]")
    print(f"Rodando {script} para {ativo.upper()} {timeframe.upper()}...")
    print(f"[{'='*50}]\n")
    try:
        subprocess.run([sys.executable, script, "--ativo", ativo, "--timeframe", timeframe], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Erro ao executar {script}: {e}")
        print("Interrompendo lote.")
        sys.exit(1)

print("\n\n[SUCESSO] Todas as 7 otimizacoes finalizadas com sucesso!")
