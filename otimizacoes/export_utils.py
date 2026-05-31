import matplotlib.pyplot as plt
import pandas as pd

def salvar_tabela_png(df, caminho_saida, titulo="Top 50 Parametrizações"):
    bg_color = '#0F0F13'
    text_color = '#E2E2E2'
    header_color = '#1A1A24'
    row_even_color = '#14141A'
    row_odd_color = '#0F0F13'
    edge_color = '#2A2A35'
    
    df_plot = df.copy()
    
    # Adicionar Rank caso não tenha
    if 'Rank' not in df_plot.columns:
        df_plot.insert(0, 'Rank', range(1, len(df_plot) + 1))
    
    for col in df_plot.columns:
        if df_plot[col].dtype in ['float64', 'float32']:
            df_plot[col] = df_plot[col].apply(lambda x: f"{x:.2f}")
            
    num_cols = len(df_plot.columns)
    num_rows = len(df_plot)
    
    # 50 linhas precisam de altura suficiente (0.22 por linha é uma boa estimativa apertada)
    fig_width = max(14, num_cols * 1.5)
    fig_height = max(5, num_rows * 0.28 + 2.0)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=150)
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    ax.axis('off')
    
    col_labels = df_plot.columns
    cell_text = df_plot.values.tolist()
    
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(edge_color)
        cell.set_text_props(color=text_color)
        
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(weight='bold', color='#4DA8DA')
        else:
            if row % 2 == 0:
                cell.set_facecolor(row_even_color)
            else:
                cell.set_facecolor(row_odd_color)
                
            # Destaques para colunas chaves
            if col > 0:
                col_name = col_labels[col]
                try:
                    val = float(cell_text[row-1][col])
                    if col_name == 'Win_Rate' and val >= 40:
                        cell.set_text_props(color='#4CAF50', weight='bold') # Verde
                    elif col_name == 'Win_Rate' and val < 30:
                        cell.set_text_props(color='#F44336') # Vermelho
                    elif col_name == 'Payoff' and val >= 2.0:
                        cell.set_text_props(color='#4CAF50', weight='bold')
                    elif col_name == 'Ret_DD' and val >= 3.0:
                        cell.set_text_props(color='#FFC107', weight='bold') # Amarelo/Ouro
                except ValueError:
                    pass

    plt.suptitle(titulo, color='#FFFFFF', fontsize=18, weight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(caminho_saida, facecolor=bg_color, edgecolor='none', bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)
