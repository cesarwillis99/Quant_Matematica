import sys
from pathlib import Path

# Adiciona o diretorio pai (what_if) ao path para importar a engine
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from what_if_engine import rodar_what_if

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
            res = rodar_what_if(
                caminho_csv=cam_csv,
                nome_estrategia=nome_est,
                nome_ativo='eurusd',
                caminho_saida=caminho_saida,
                pct_remocao=0.01
            )
            if res is not None:
                resultados.append(res)
        except FileNotFoundError:
            print(f"[AVISO] Arquivo não encontrado: {cam_csv}. Pulando {nome_est}...")
            continue
        except Exception as e:
            print(f"[ERRO] Ocorreu um erro ao processar {nome_est}: {e}")
            continue

    print("\n" + "="*85)
    print("RESUMO DO TESTE DE ROBUSTEZ: WHAT IF (Remoção dos Top 1% Trades)")
    print("="*85)
    print(f"{'Estratégia':<15} | {'Trades':<8} | {'Removidos':<10} | {'L. Original':<12} | {'L. After':<12} | {'Impacto %':<10} | {'Resultado'}")
    print("-" * 85)
    for r in resultados:
        res_str = "APROVADO" if r['aprovado'] else "REPROVADO"
        print(f"{r['estrategia']:<15} | {r['total_trades']:<8} | {r['n_removidos']:<10} | ${r['lucro_original']:<11.2f} | ${r['lucro_after']:<11.2f} | {r['impacto_pct']:<9.2f}% | {res_str}")
    print("="*85)

if __name__ == "__main__":
    main()
