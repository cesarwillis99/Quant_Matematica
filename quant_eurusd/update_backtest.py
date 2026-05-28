import os
from pathlib import Path

file_path = Path(r"c:\Users\cesar\.gemini\antigravity\scratch\Quant_Matematica.Trade\quant_eurusd\backtest.py")

with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# 1. Update Imports/Constants
text = text.replace('eurusd_h1_ou.parquet', 'eurusd_h1_hawkes.parquet')
text = text.replace('PARQUET_OU', 'PARQUET_HAWKES')
text = text.replace('COR_OU', 'COR_HAWKES')
text = text.replace('COR_DD_O', 'COR_DD_H')
text = text.replace('OU_NEUTRO_MIN, OU_NEUTRO_MAX = -0.3, 0.3', 'LAMBDA_NORM_SAIDA = 0.6')

# 2. Update params names
text = text.replace('ou_zscore', 'hawkes_lambda_norm')
text = text.replace('usar_ou_exit', 'usar_hawkes_exit')
text = text.replace('sl_pips_ou', 'sl_pips_hawkes')
text = text.replace('tp_pips_ou', 'tp_pips_hawkes')
text = text.replace('sinal_ou', 'sinal_hawkes')

# 3. Fix strategy names
text = text.replace('"OU"', '"HAWKES"')
text = text.replace('df_o', 'df_h')

# 4. Fix Neutral exit logic in verificar_saida_candle
old_exit = """        if usar_hawkes_exit and hawkes_lambda_norm is not None and not math.isnan(hawkes_lambda_norm):
            if HAWKES_NEUTRO_MIN <= hawkes_lambda_norm <= HAWKES_NEUTRO_MAX:
                return True, close, "HAWKES_NEUTRO" """
new_exit = """        if usar_hawkes_exit and hawkes_lambda_norm is not None and not math.isnan(hawkes_lambda_norm):
            if hawkes_lambda_norm < LAMBDA_NORM_SAIDA:
                return True, close, "HAWKES_NEUTRO" """

text = text.replace(old_exit, new_exit)
text = text.replace("OU_NEUTRO", "HAWKES_NEUTRO")

# Write back
with open(file_path, "w", encoding="utf-8") as f:
    f.write(text)

print("Updated backtest.py successfully.")
