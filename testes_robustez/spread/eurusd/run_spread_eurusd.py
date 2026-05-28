import sys
from pathlib import Path

# Adiciona o diretorio pai (spread) ao path para importar a engine
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from spread_engine import rodar_spread

ESTRATEGIAS = [
    ('zscore',      'quant_eurusd/resultados/operacoes_zscore_eurusd.csv'),
    ('hawkes',      'quant_eurusd/resultados/operacoes_hawkes_eurusd.csv'),
    ('momentum',    'quant_eurusd/resultados/operacoes_momentum_eurusd.csv'),
    ('ou',          'quant_eurusd/resultados/operacoes_ou_eurusd.csv'),
    ('ou_reverso',  'quant_eurusd/resultados/operacoes_ou_reverso_eurusd.csv'),
    ('pca',         'quant_eurusd/resultados/operacoes_pca_eurusd.csv'),
    ('wavelet',     'quant_eurusd/resultados/operacoes_wavelet_eurusd.csv'),
    ('curvatura',   'quant_eurusd/resultados/operacoes_curvatura_eurusd.csv'),
]

def main():
    resultados = []
    
    # O script assume que estamos rodando da raiz. O output vai para a pasta em que este script está salvo.
    caminho_saida = str(Path(__file__).resolve().parent)
    
    for nome_est, cam_csv in ESTRATEGIAS:
        try:
            res = rodar_spread(
                caminho_csv=cam_csv,
                nome_estrategia=nome_est,
                nome_ativo='eurusd',
                caminho_saida=caminho_saida,
                spread_original=0.5,
                spread_multiplo=1.8,
                pip_value_por_lot=10.0
            )
            if res is not None:
                resultados.append(res)
        except FileNotFoundError:
            print(f"[AVISO] Arquivo não encontrado: {cam_csv}. Pulando {nome_est}...")
            continue
        except Exception as e:
            print(f"[ERRO] Ocorreu um erro ao processar {nome_est}: {e}")
            continue

    print("\n" + "="*95)
    print("RESUMO DO TESTE DE ROBUSTEZ: SPREAD STRESS TEST (+80%)")
    print("="*95)
    print(f"{'Estratégia':<15} | {'Trades':<8} | {'Sp.Orig':<8} | {'Sp.Novo':<8} | {'L. Original':<12} | {'L. Ajustado':<12} | {'Impacto %':<10} | {'Resultado'}")
    print("-" * 95)
    for r in resultados:
        res_str = "APROVADO" if r['aprovado'] else "REPROVADO"
        print(f"{r['estrategia']:<15} | {r['total_trades']:<8} | {r['spread_original']:<8.1f} | {r['spread_novo']:<8.1f} | ${r['lucro_original']:<11.2f} | ${r['lucro_ajustado']:<11.2f} | {r['impacto_pct']:<9.2f}% | {res_str}")
    print("="*95)

if __name__ == "__main__":
    main()
