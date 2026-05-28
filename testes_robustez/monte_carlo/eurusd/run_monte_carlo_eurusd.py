import sys
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from monte_carlo_engine import rodar_monte_carlo

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
    caminho_saida = str(Path(__file__).resolve().parent)
    
    print("Iniciando bateria de 10.000 simulações de Monte Carlo por estratégia...")
    print("Processamento de Matrizes ativado. Aguarde...\n")
    
    for nome_est, cam_csv in ESTRATEGIAS:
        try:
            res = rodar_monte_carlo(
                caminho_csv=cam_csv,
                nome_estrategia=nome_est,
                nome_ativo='eurusd',
                caminho_saida=caminho_saida,
                n_simulacoes=10000,
                prob_skip=0.05
            )
            if res is not None:
                resultados.append(res)
                print(f"[OK] {nome_est.upper()} - 10.000 cenários calculados e avaliados.")
        except FileNotFoundError:
            print(f"[AVISO] Arquivo não encontrado: {cam_csv}. Pulando {nome_est}...")
            continue
        except Exception as e:
            print(f"[ERRO] Ocorreu um erro ao processar {nome_est}: {e}")
            continue

    print("\n" + "="*95)
    print("RESUMO DO TESTE DE ROBUSTEZ: MONTE CARLO (10.000 RUNS)")
    print("="*95)
    print(f"{'Estratégia':<15} | {'Ex.1 (>=1.5)':<15} | {'Ex.2 (>=40% Orig)':<20} | {'Ex.3 (PF>=1.01)':<15} | {'Resultado'}")
    print("-" * 95)
    for r in resultados:
        res_str = "APROVADO" if r['aprovado'] else "REPROVADO"
        e1 = f"{r['ret_dd_50']:>6.2f} " + ("(OK)" if r['passou_ex1'] else "(X)")
        e2 = f"{r['ret_dd_70']:>6.2f} " + ("(OK)" if r['passou_ex2'] else "(X)")
        e3 = f"{r['pf_99']:>6.2f} " + ("(OK)" if r['passou_ex3'] else "(X)")
        
        print(f"{r['estrategia']:<15} | {e1:<15} | {e2:<20} | {e3:<15} | {res_str}")
    print("="*95)

if __name__ == "__main__":
    main()
