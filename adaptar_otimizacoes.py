import os
import glob
from pathlib import Path

# Pasta das otimizações
otimizacoes_dir = Path("C:/Users/cesar/.gemini/antigravity/scratch/Quant_Matematica.Trade/otimizacoes")

for file_path in otimizacoes_dir.glob("otimizacao_*.py"):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Vamos substituir o bloco de "CONSTANTES E CAMINHOS"
    if "import argparse" not in content:
        # 1. Adicionar os imports e CLI logic logo após "import warnings" ou similar
        import_block = """import argparse
import json"""
        content = content.replace("import warnings", f"import warnings\n{import_block}")
        
        # Encontrar onde começa o bloco de constantes
        # =============================================================================
        # CONSTANTES E CAMINHOS
        # =============================================================================
        old_constants = """# =============================================================================
# CONSTANTES E CAMINHOS
# =============================================================================
DIR_PROJETO = Path(__file__).resolve().parent.parent"""
        
        # Tentar várias abordagens se houver diferenças
        if old_constants in content:
            new_constants = """# =============================================================================
# CLI & CAMINHOS GENERICOS
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--ativo", type=str, required=True, help="Ex: eurusd")
parser.add_argument("--timeframe", type=str, required=True, help="Ex: h1")
args = parser.parse_args()

ativo = args.ativo.lower()
timeframe = args.timeframe.lower()
estrategia_nome = Path(__file__).stem.replace('otimizacao_', '')

DIR_PROJETO = Path(__file__).resolve().parent.parent
DIR_DATA    = DIR_PROJETO / f"quant_{ativo}_{timeframe}" / "data"
DIR_SAIDA   = DIR_DATA / "otimizacoes"

PARQUET_COMPLETO    = DIR_DATA / f"{ativo}_{timeframe}_completo.parquet"
PARQUET_OPERACIONAL = DIR_DATA / f"{ativo}_{timeframe}_operacional.parquet"
ARQUIVO_SAIDA       = DIR_SAIDA / f"otimizacao_{estrategia_nome}_resultados.parquet"
ARQUIVO_JSON        = DIR_SAIDA / f"otimizacao_{estrategia_nome}_top10.json"

# Verifica se existe a pasta de saida
os.makedirs(DIR_SAIDA, exist_ok=True)"""
            
            # Precisamos substituir o trecho inteiro das antigas variaveis. 
            # A forma mais segura é separar o código nas marcações
            parts = content.split("# GRID SEARCH")
            if len(parts) > 1:
                prefix = parts[0].split("# CONSTANTES E CAMINHOS")[0]
                content = prefix + new_constants + "\n\n# GRID SEARCH" + parts[1]
                
                
        # 2. Modificar a saida para json
        # Procura por "df_res.to_parquet(ARQUIVO_SAIDA)" e altera
        saida_antiga = """df_res = df_res.sort_values(by="Ret_DD", ascending=False).reset_index(drop=True)
        df_res.to_parquet(ARQUIVO_SAIDA)
        print("\\n================================================================================")
        print("TOP 10 PARAMETRIZACOES (RANKING POR RET/DD):")
        print("================================================================================")
        print(df_res.head(10).to_string())
        print(f"\\nResultados salvos em: {ARQUIVO_SAIDA}")"""

        saida_nova = """# Filtro de Sobrevivencia (Trades >= 60 e F.R. > 1.0)
        mask_survivor = (df_res["Trades"] >= 60) & (df_res["Profit_Factor"] > 1.0)
        df_res = df_res[mask_survivor]
        
        df_res = df_res.sort_values(by="Ret_DD", ascending=False).reset_index(drop=True)
        
        if len(df_res) > 0:
            df_res.to_parquet(ARQUIVO_SAIDA)
            
            top10 = df_res.head(10).to_dict(orient="records")
            # Adiciona um ID a cada parametro
            for i, p in enumerate(top10):
                p["id"] = f"{estrategia_nome.upper()}_TOP{i+1}"
                
            with open(ARQUIVO_JSON, 'w') as f:
                json.dump(top10, f, indent=4)
                
            print("\\n================================================================================")
            print("TOP 10 PARAMETRIZACOES SOBREVIVENTES (RANKING POR RET/DD):")
            print("================================================================================")
            print(df_res.head(10).to_string())
            print(f"\\nResultados salvos em: {ARQUIVO_SAIDA} e {ARQUIVO_JSON}")
        else:
            print("Nenhuma combinacao sobreviveu aos criterios rigorosos (Min 60 Trades, F.R > 1.0).")"""
            
        content = content.replace(saida_antiga, saida_nova)
        
        # 3. Alguns arquivos podem ter formatação diferente para a saída, vamos tentar um fall-back com replace mais genérico
        if "to_dict(orient=\"records\")" not in content:
            # Fallback manual pra cada
            pass
            
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Modificado: {file_path.name}")
