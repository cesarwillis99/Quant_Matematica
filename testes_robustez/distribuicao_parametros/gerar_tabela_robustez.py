#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gerar_tabela_robustez.py
-------------------------
Le os CSVs de distribuicao de parametros e gera uma tabela resumo 
premium em imagem PNG (dark mode) na pasta dos resultados.
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

DIR_RES = r"c:\Users\cesar\.gemini\antigravity\scratch\Quant_Matematica.Trade\testes_robustez\distribuicao_parametros\eurusd_h1"

def main():
    csv_files = glob.glob(os.path.join(DIR_RES, "resultado_distribuicao_*.csv"))
    
    registros = []
    
    for caminho in csv_files:
        nome_arquivo = os.path.basename(caminho)
        # Extrair estrategia do nome do arquivo
        estrategia = nome_arquivo.replace("resultado_distribuicao_", "").replace(".csv", "").upper()
        
        df = pd.read_csv(caminho)
        
        # Agrupar por parametro e ver se passou
        df_params = df.groupby("parametro")["passou_parametro"].first().reset_index()
        
        total_params = len(df_params)
        aprovados = int(df_params["passou_parametro"].sum())
        reprovados = total_params - aprovados
        
        # O veredito da estrategia passa se pelo menos 80% dos parametros forem aceitaveis
        # E todas as perturbacoes da distribuicao forem nao-negativas (>= 0.0%)
        pct_aprovados = aprovados / total_params if total_params > 0 else 0.0
        todos_positivos = bool((df["pnl_pct"] >= 0.0).all())
        
        passou_tudo = (pct_aprovados >= 0.80) and todos_positivos
        veredito = "ROBUSTO" if passou_tudo else "FRÁGIL"
        
        # Criar strings de parametros aprovados e reprovados
        params_ok = df_params[df_params["passou_parametro"] == True]["parametro"].tolist()
        params_nok = df_params[df_params["passou_parametro"] == False]["parametro"].tolist()
        
        str_ok = ", ".join([p.replace("janela_", "").replace("mult_", "") for p in params_ok])
        str_nok = ", ".join([p.replace("janela_", "").replace("mult_", "") for p in params_nok])
        
        if not str_ok: str_ok = "-"
        if not str_nok: str_nok = "-"
        
        registros.append({
            "Estratégia": estrategia,
            "Total Params": total_params,
            "Aprovados": aprovados,
            "Reprovados": reprovados,
            "Parâmetros OK": str_ok,
            "Parâmetros Falhos": str_nok,
            "Veredito": veredito,
            "_sort_aprov": aprovados / total_params if total_params > 0 else 0
        })
        
    df_resumo = pd.DataFrame(registros)
    if not df_resumo.empty:
        # Ordenar por maior taxa de aprovacao
        df_resumo = df_resumo.sort_values(by="_sort_aprov", ascending=False).drop(columns=["_sort_aprov"])
    
    # Plotar tabela dark premium
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(15, len(df_resumo)/1.5 + 2.5))
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
    tabela.set_fontsize(10)
    tabela.scale(1.2, 2.2)
    
    # Ajustar larguras das colunas
    col_widths = {0: 0.12, 1: 0.08, 2: 0.08, 3: 0.08, 4: 0.28, 5: 0.28, 6: 0.08}
    for col_idx, width in col_widths.items():
        for row_idx in range(len(df_resumo) + 1):
            if (row_idx, col_idx) in tabela.get_celld():
                tabela.get_celld()[(row_idx, col_idx)].set_width(width)
    
    for (row, col), cell in tabela.get_celld().items():
        cell.set_edgecolor("#21262D")
        if row == 0:
            cell.set_text_props(weight="bold", color="#FFFFFF", fontsize=11)
            cell.set_facecolor("#1E4F8A")
        else:
            val_celula = cell.get_text().get_text()
            cell.set_text_props(color="#CFD8DC")
            if col == 0:
                cell.set_text_props(weight="bold", color="#58A6FF")
            elif col == 6: # Veredito
                if val_celula == "ROBUSTO":
                    cell.set_text_props(weight="bold", color="#56D364")
                    cell.set_facecolor("#14321A")
                else:
                    cell.set_text_props(weight="bold", color="#F85149")
                    cell.set_facecolor("#2E1414")
                    
    fig.suptitle("Relatório Oficial de Robustez Paramétrica (Neighborhood SQX)\nEURUSD H1 (2016-2023) | Filtro: Estável em 7 passos com DD < 30%", 
                 color="#E6EDF3", fontsize=14, fontweight="bold", y=0.98)
    
    plt.tight_layout()
    caminho_img = os.path.join(DIR_RES, "tabela_robustez_parametros.png")
    plt.savefig(caminho_img, dpi=180, facecolor="#0D1117", bbox_inches="tight")
    plt.close()
    
    print(f"\n[SUCESSO] Tabela resumo gerada e salva em: {caminho_img}\n")

if __name__ == "__main__":
    main()
