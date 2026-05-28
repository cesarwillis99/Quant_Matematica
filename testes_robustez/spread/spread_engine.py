import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def rodar_spread(
    caminho_csv: str,
    nome_estrategia: str,
    nome_ativo: str,
    caminho_saida: str,
    spread_original: float = 1.0,
    spread_multiplo: float = 1.5,
    pip_value_por_lot: float = 10.0,
) -> dict:
    
    # 1. Carregar CSV
    df = pd.read_csv(caminho_csv)
    total_trades = len(df)
    
    if total_trades == 0:
        return None
        
    # 2. Calcular o spread extra por trade
    spread_extra_pips = spread_original * (spread_multiplo - 1.0)
    custo_extra_trade = spread_extra_pips * df['lot_size'] * pip_value_por_lot
    
    # 3. Calcular pnl ajustado
    df['pnl_ajustado'] = df['pnl_monetario'] - custo_extra_trade
    
    # 4. Cálculos globais
    lucro_original = df['pnl_monetario'].sum()
    lucro_ajustado = df['pnl_ajustado'].sum()
    custo_total = custo_extra_trade.sum()
    
    if lucro_original != 0:
        impacto_pct = (custo_total / abs(lucro_original)) * 100
    else:
        impacto_pct = 0.0
        
    # 5. Critério de aprovação
    aprovado = lucro_ajustado >= (0.80 * lucro_original)
    
    res = {
        'estrategia': nome_estrategia,
        'ativo': nome_ativo,
        'total_trades': total_trades,
        'spread_original': spread_original,
        'spread_novo': spread_original * spread_multiplo,
        'spread_extra': spread_extra_pips,
        'lucro_original': lucro_original,
        'custo_total': custo_total,
        'lucro_ajustado': lucro_ajustado,
        'impacto_pct': impacto_pct,
        'aprovado': aprovado,
    }
    
    # 6. Geração do PNG
    gerar_png(res, caminho_saida)
    
    return res

def gerar_png(res: dict, caminho_saida: str):
    caminho_saida_path = Path(caminho_saida)
    caminho_saida_path.mkdir(parents=True, exist_ok=True)
    arquivo_saida = caminho_saida_path / f"spread_{res['estrategia']}_{res['ativo']}.png"
    
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    fig.patch.set_facecolor('#000000')
    ax.set_facecolor('#000000')
    ax.axis('off')
    
    cor_titulo = '#FFFFFF'
    cor_label = '#AAAAAA'
    cor_valor = '#FFD700'
    cor_pct = '#FFA500'
    cor_aprov = '#00FF99'
    cor_reprov = '#FF4444'
    cor_linha = '#333333'
    
    # Top Text
    ax.text(0.5, 0.93, f"{res['estrategia'].upper()} — {res['ativo'].upper()}",
            color=cor_titulo, fontsize=14, fontweight='bold', ha='center', family='monospace')
    
    spread_inc_pct = ((res['spread_novo'] / res['spread_original']) - 1.0) * 100
    ax.text(0.5, 0.86, f"SPREAD STRESS TEST: +{spread_inc_pct:.0f}% de Spread",
            color=cor_titulo, fontsize=11, ha='center', family='monospace')
            
    # Separator 1
    ax.add_patch(patches.Rectangle((0.05, 0.80), 0.9, 0.002, color=cor_linha, transform=ax.transAxes))
    
    # Layout params
    x_lbl = 0.10
    x_col = 0.45
    x_val = 0.48
    y_start = 0.68
    lh = 0.08
    
    # Row 1
    ax.text(x_lbl, y_start, "Spread Original", color=cor_label, fontsize=11, family='monospace')
    ax.text(x_col, y_start, ":", color=cor_label, fontsize=11, family='monospace')
    ax.text(x_val, y_start, f"{res['spread_original']:.1f} pip", color=cor_titulo, fontsize=11, family='monospace')
    
    # Row 2
    ax.text(x_lbl, y_start - lh, "Spread Testado", color=cor_label, fontsize=11, family='monospace')
    ax.text(x_col, y_start - lh, ":", color=cor_label, fontsize=11, family='monospace')
    ax.text(x_val, y_start - lh, f"{res['spread_novo']:.1f} pip  (+{spread_inc_pct:.0f}%)", color=cor_titulo, fontsize=11, family='monospace')
    
    # Row 3
    ax.text(x_lbl, y_start - 2*lh, "Total de Trades", color=cor_label, fontsize=11, family='monospace')
    ax.text(x_col, y_start - 2*lh, ":", color=cor_label, fontsize=11, family='monospace')
    ax.text(x_val, y_start - 2*lh, f"{res['total_trades']}", color=cor_titulo, fontsize=11, family='monospace')
    
    # Blank line
    y_start2 = y_start - 3.5 * lh
    
    # Row 4
    ax.text(x_lbl, y_start2, "Lucro Original", color=cor_label, fontsize=11, family='monospace')
    ax.text(x_col, y_start2, ":", color=cor_label, fontsize=11, family='monospace')
    ax.text(x_val, y_start2, f"$ {res['lucro_original']:,.2f}", color=cor_valor, fontsize=11, family='monospace')
    
    # Row 5
    ax.text(x_lbl, y_start2 - lh, "Custo Extra Total", color=cor_label, fontsize=11, family='monospace')
    ax.text(x_col, y_start2 - lh, ":", color=cor_label, fontsize=11, family='monospace')
    ax.text(x_val, y_start2 - lh, f"$ {res['custo_total']:,.2f}  ({res['impacto_pct']:.2f}% do lucro)", color=cor_pct, fontsize=11, family='monospace')
    
    # Row 6
    ax.text(x_lbl, y_start2 - 2*lh, "Lucro Ajustado", color=cor_label, fontsize=11, family='monospace')
    ax.text(x_col, y_start2 - 2*lh, ":", color=cor_label, fontsize=11, family='monospace')
    ax.text(x_val, y_start2 - 2*lh, f"$ {res['lucro_ajustado']:,.2f}", color=cor_valor, fontsize=11, family='monospace')
    
    # Separator 2
    ax.add_patch(patches.Rectangle((0.05, 0.28), 0.9, 0.002, color=cor_linha, transform=ax.transAxes))
    
    # Result
    if res['aprovado']:
        texto_resultado = "✅ APROVADO"
        cor_res = cor_aprov
    else:
        texto_resultado = "❌ REPROVADO"
        cor_res = cor_reprov
        
    ax.text(0.5, 0.15, texto_resultado, color=cor_res, fontsize=22, fontweight='bold', ha='center', va='center', family='monospace')
    
    # Separator 3
    ax.add_patch(patches.Rectangle((0.05, 0.02), 0.9, 0.002, color=cor_linha, transform=ax.transAxes))
    
    plt.tight_layout()
    plt.savefig(arquivo_saida, bbox_inches='tight', facecolor='#000000', dpi=150)
    plt.close()
