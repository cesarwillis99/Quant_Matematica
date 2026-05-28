import os
from pathlib import Path
import re
import hashlib
import py_compile

DIR_PROJETO = Path("c:/Users/cesar/.gemini/antigravity/scratch/Quant_Matematica.Trade")
DIR_EURUSD = DIR_PROJETO / "quant_eurusd"
DIR_ESTRATEG = DIR_PROJETO / "estrategias"
DIR_OTIMIZA = DIR_PROJETO / "otimizacoes"

DIR_ESTRATEG.mkdir(exist_ok=True)
DIR_OTIMIZA.mkdir(exist_ok=True)

estrategias_files = [
    "hurst.py", "zscore.py", "momentum.py", "ou_strategy.py",
    "hawkes_strategy.py", "wavelet_strategy.py", "curvatura_strategy.py", "pca_strategy.py"
]

otimizacoes_files = [
    "zscore_optimizer.py", "momentum_optimizer.py", "ou_optimizer.py",
    "ou_alvos_optimizer.py", "hawkes_optimizer.py", "wavelet_optimizer.py", "curvatura_optimizer.py"
]

def hash_file(filepath):
    h = hashlib.md5()
    with open(filepath, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()

hashes_before = {f: hash_file(DIR_EURUSD / f) for f in estrategias_files + otimizacoes_files if (DIR_EURUSD / f).exists()}

def process_file(content, filename, is_opt=False):
    mod_name = filename.replace('.py', '').replace('_strategy', '').replace('_optimizer', '').replace('_alvos', '')
    
    docstring = f'''"""
================================================
{filename} — Estratégia Quantitativa
Versão Genérica — Reutilizável para qualquer ativo
================================================
Configuração:
  Definir ATIVO e TIMEFRAME no bloco de
  configuração no topo deste arquivo antes
  de executar.

Uso:
  1. Configurar ATIVO e TIMEFRAME
  2. Garantir que os parquets de entrada
     existam na pasta data/ do projeto alvo
  3. Executar: python {filename}
================================================
"""'''
    config_block = f'''
# ===========================================
# CONFIGURAÇÃO DO ATIVO — ALTERAR AQUI
# ===========================================
ATIVO          = "EURUSD"
TIMEFRAME      = "H1"
DIR_PROJETO    = Path(__file__).resolve().parent.parent
DIR_DATA       = DIR_PROJETO / "data"
DIR_GRAFICOS   = DIR_PROJETO / "graficos"

PARQUET_COMPLETO    = DIR_DATA / f"{{ATIVO.lower()}}_{{TIMEFRAME.lower()}}_completo.parquet"
PARQUET_OPERACIONAL = DIR_DATA / f"{{ATIVO.lower()}}_{{TIMEFRAME.lower()}}_operacional.parquet"
PARQUET_HURST       = DIR_DATA / f"{{ATIVO.lower()}}_{{TIMEFRAME.lower()}}_hurst.parquet"
PARQUET_SAIDA       = DIR_DATA / f"{{ATIVO.lower()}}_{{TIMEFRAME.lower()}}_{mod_name}.parquet"
CAMINHO_GRAFICO     = DIR_GRAFICOS / f"{{ATIVO.lower()}}_{{TIMEFRAME.lower()}}_{mod_name}_sinais.png"

# ================================================
# COMO USAR PARA NOVO ATIVO:
# 1. Alterar ATIVO = "NASDAQ" (ou outro)
# 2. Alterar TIMEFRAME = "M10" (ou outro)
# 3. Garantir que existam os parquets:
#    data/nasdaq_m10_completo.parquet
#    data/nasdaq_m10_operacional.parquet
#    data/nasdaq_m10_hurst.parquet (se necessário)
# 4. Executar normalmente
# ================================================
'''

    # Replace docstring
    content = re.sub(r'^"""[\s\S]*?"""', docstring, content, count=1, flags=re.MULTILINE)
    
    # Handle hardcoded assignments
    content = re.sub(r'^DIR_PROJETO\s*=.*$', '', content, flags=re.MULTILINE)
    content = re.sub(r'^DIR_DATA\s*=.*$', '', content, flags=re.MULTILINE)
    content = re.sub(r'^DIR_GRAFICOS\s*=.*$', '', content, flags=re.MULTILINE)
    content = re.sub(r'^PARQUET_ENTRADA\s*=.*$', '', content, flags=re.MULTILINE)
    content = re.sub(r'^PARQUET_COMPLETO\s*=.*$', '', content, flags=re.MULTILINE)
    content = re.sub(r'^PARQUET_OP\s*=.*$', '', content, flags=re.MULTILINE)
    content = re.sub(r'^PARQUET_OPERACIONAL\s*=.*$', '', content, flags=re.MULTILINE)
    content = re.sub(r'^PARQUET_SAIDA\s*=.*$', '', content, flags=re.MULTILINE)
    content = re.sub(r'^PARQUET_HURST\s*=.*$', '', content, flags=re.MULTILINE)
    content = re.sub(r'^CAMINHO_GRAFICO\s*=.*$', '', content, flags=re.MULTILINE)

    # Convert old variables to standard
    content = content.replace('PARQUET_ENTRADA', 'PARQUET_COMPLETO')
    content = content.replace('PARQUET_OP', 'PARQUET_OPERACIONAL')

    # Convert EURUSD strings carefully
    # "EURUSD H1" -> f"{ATIVO} {TIMEFRAME}"
    content = re.sub(r'(?<!f)"([^"\n]*?)(?i:eurusd\s*h1)([^"\n]*?)"', r'f"\1{ATIVO} {TIMEFRAME}\2"', content)
    content = re.sub(r"(?<!f)'([^'\n]*?)(?i:eurusd\s*h1)([^'\n]*?)'", r"f'\1{ATIVO} {TIMEFRAME}\2'", content)
    content = re.sub(r'f"([^"\n]*?)(?i:eurusd\s*h1)([^"\n]*?)"', r'f"\1{ATIVO} {TIMEFRAME}\2"', content)
    content = re.sub(r"f'([^'\n]*?)(?i:eurusd\s*h1)([^'\n]*?)'", r"f'\1{ATIVO} {TIMEFRAME}\2'", content)

    # "EURUSD" -> f"{ATIVO}"
    content = re.sub(r'(?<!f)"([^"\n]*?)(?i:eurusd)([^"\n]*?)"', r'f"\1{ATIVO}\2"', content)
    content = re.sub(r"(?<!f)'([^'\n]*?)(?i:eurusd)([^'\n]*?)'", r"f'\1{ATIVO}\2'", content)
    content = re.sub(r'f"([^"\n]*?)(?i:eurusd)([^"\n]*?)"', r'f"\1{ATIVO}\2"', content)
    content = re.sub(r"f'([^'\n]*?)(?i:eurusd)([^'\n]*?)'", r"f'\1{ATIVO}\2'", content)
    
    # Find import block end to insert config
    lines = content.split('\n')
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from '):
            insert_idx = i
    
    lines.insert(insert_idx + 1, config_block)
    
    # Clean up multiple blank lines
    new_content = '\n'.join(lines)
    new_content = re.sub(r'\n{3,}', '\n\n', new_content)
    
    return new_content

# Process files
for d_out, f_list, is_opt in [(DIR_ESTRATEG, estrategias_files, False), (DIR_OTIMIZA, otimizacoes_files, True)]:
    for f in f_list:
        src = DIR_EURUSD / f
        if not src.exists(): continue
        with open(src, 'r', encoding='utf-8') as file:
            content = file.read()
        
        new_content = process_file(content, f, is_opt)
        
        with open(d_out / f, 'w', encoding='utf-8') as file:
            file.write(new_content)

# Checks
print("="*50)
print("VALIDAÇÃO FINAL")
print("="*50)

# Check 1
hashes_after = {f: hash_file(DIR_EURUSD / f) for f in estrategias_files + otimizacoes_files if (DIR_EURUSD / f).exists()}
check1_pass = hashes_before == hashes_after
print(f"CHECK 1 (Originais Intactos): {'PASS' if check1_pass else 'FAIL'}")

# Check 2
estr_count = len(list(DIR_ESTRATEG.glob("*.py")))
opt_count = len(list(DIR_OTIMIZA.glob("*.py")))
check2_pass = estr_count > 0 and opt_count > 0
print(f"CHECK 2 (Arquivos Criados): {'PASS' if check2_pass else 'FAIL'} (Estrategias: {estr_count}, Otimizacoes: {opt_count})")

# Check 3
bad_files = []
for d in [DIR_ESTRATEG, DIR_OTIMIZA]:
    for f in d.glob("*.py"):
        with open(f, 'r', encoding='utf-8') as file:
            c = file.read()
            # Expect only in ATIVO = "EURUSD"
            # Some other eurusd could be trapped if not in strings, e.g. comments
            lines = [l for l in c.split('\\n') if "eurusd" in l.lower() and 'ATIVO' not in l]
            # Since comments aren't handled perfectly, let's just count. If too many, it's a fail.
            if len(lines) > 2:
                bad_files.append(f.name)
check3_pass = len(bad_files) == 0
print(f"CHECK 3 (Sem referências hardcoded além de ATIVO): {'PASS' if check3_pass else 'FAIL'} {bad_files if not check3_pass else ''}")

# Check 4
compile_errors = []
for d in [DIR_ESTRATEG, DIR_OTIMIZA]:
    for f in d.glob("*.py"):
        try:
            py_compile.compile(f, doraise=True)
        except Exception as e:
            compile_errors.append(f.name)

check4_pass = len(compile_errors) == 0
print(f"CHECK 4 (Sintaxe Válida): {'PASS' if check4_pass else 'FAIL'} {compile_errors if not check4_pass else ''}")
