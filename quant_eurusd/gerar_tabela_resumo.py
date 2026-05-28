import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

DIR_RES = r"c:\Users\cesar\.gemini\antigravity\scratch\Quant_Matematica.Trade\quant_eurusd\resultados"
DIR_OUT = r"c:\Users\cesar\.gemini\antigravity\scratch\Quant_Matematica.Trade\quant_eurusd\metricas"

os.makedirs(DIR_OUT, exist_ok=True)

df = pd.read_csv(os.path.join(DIR_RES, "metricas.csv"))
registros = []

estrategias_permitidas = ["MOMENTUM", "PCA", "HAWKES", "OU_REVERSO", "ZSCORE", "WAVELET", "COMBINADA"]

for idx, row in df.iterrows():
    nome = row["nome"].upper()
    if nome not in estrategias_permitidas: continue
    
    trades = int(row["total_operacoes"])
    pnl = f"+{row['pnl_total_pct']:.2f}%" if row['pnl_total_pct'] > 0 else f"{row['pnl_total_pct']:.2f}%"
    wr = f"{row['win_rate']:.1f}%"
    sharpe = f"{row['sharpe_ratio']:.2f}"
    dd = f"{row['drawdown_max_pct']:.2f}%"
    fr = f"{row['fator_recuperacao']:.2f}x"
    
    registros.append({
        "Estratégia": nome,
        "PnL Líquido": pnl,
        "Win Rate": wr,
        "Sharpe": sharpe,
        "Max DD": dd,
        "F.R.": fr,
        "Total Trades": trades,
        "_sort_pnl": row['pnl_total_pct']
    })

df_resumo = pd.DataFrame(registros)
df_resumo = df_resumo.sort_values(by="_sort_pnl", ascending=False).drop(columns=["_sort_pnl"])

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(14, len(df_resumo)/2 + 2))
ax.axis("off")

cores_celulas = []
for i in range(len(df_resumo)):
    linha = []
    for j in range(len(df_resumo.columns)):
        linha.append("#121212")
    cores_celulas.append(linha)
    
tabela = ax.table(
    cellText=df_resumo.values,
    colLabels=df_resumo.columns,
    loc="center",
    cellLoc="center",
    cellColours=cores_celulas
)

tabela.auto_set_font_size(False)
tabela.set_fontsize(12)
tabela.scale(1.2, 2.0)

for (row, col), cell in tabela.get_celld().items():
    cell.set_edgecolor("#333333")
    if row == 0:
        cell.set_text_props(weight="bold", color="#FFFFFF", fontsize=13)
        cell.set_facecolor("#1E4F8A")
    else:
        cell.set_text_props(color="#CFD8DC")
        if col == 0:
            cell.set_text_props(weight="bold", color="#82AAFF")

fig.suptitle("Performance Oficial - Estratégias Quantitativas EURUSD (2016-2023)", color="#E6EDF3", fontsize=18, fontweight="bold", y=0.95)
plt.tight_layout()

caminho = os.path.join(DIR_OUT, "tabela_performance_estrategias.png")
plt.savefig(caminho, dpi=150, facecolor="#0D1117", bbox_inches="tight")
plt.close()

print(f"Salvo em {caminho}")
