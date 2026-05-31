import os
import sys
import json
from pathlib import Path

def main():
    print("="*60)
    print(" GERADOR DE JSON PARA A ESTEIRA DE ROBUSTEZ (TOP 10)")
    print("="*60)
    
    estrategias = [
        "curvatura", "hawkes", "momentum", "ou", 
        "ou_reverso", "pca", "wavelet", "zscore"
    ]
    
    print("\nEstratégias disponíveis:")
    for i, est in enumerate(estrategias):
        print(f"[{i+1}] {est.upper()}")
        
    try:
        est_idx = int(input("\nEscolha o número da estratégia: ")) - 1
        if est_idx < 0 or est_idx >= len(estrategias):
            print("Número inválido.")
            return
    except ValueError:
        print("Entrada inválida.")
        return
        
    estrategia = estrategias[est_idx]
    
    # Resolvendo dinamicamente o caminho da pasta otimizacoes baseado na estrutura de data
    # Normalmente: Quant_Matematica.Trade/quant_eurusd_h1/data/otimizacoes
    ativo = input("\nQual o ativo? (padrao: eurusd): ").strip().lower() or "eurusd"
    timeframe = input("Qual o timeframe? (padrao: h1): ").strip().lower() or "h1"
    
    dir_projeto = Path(__file__).resolve().parent.parent
    dir_otimizacoes = dir_projeto / f"quant_{ativo}_{timeframe}" / "data" / "otimizacoes"
    
    dir_candidatos = dir_otimizacoes / "candidatos_testes"
    arquivo_candidatos = dir_candidatos / f"candidatos_{estrategia}.json"
    
    if not arquivo_candidatos.exists():
        print(f"\n[ERRO] Arquivo de candidatos não encontrado: {arquivo_candidatos}")
        print("Você precisa rodar a otimização dessa estratégia primeiro para gerar os Top 50.")
        return
        
    with open(arquivo_candidatos, 'r') as f:
        candidatos = json.load(f)
        
    if not candidatos:
        print("\nO arquivo de candidatos está vazio.")
        return
        
    print(f"\nCarregado {len(candidatos)} candidatos para {estrategia.upper()}.")
    print("Por favor, digite os índices (1 a 50) da planilha CSV que você selecionou.")
    print("Você pode usar vírgulas (ex: 1,3,5,10) para selecionar as 10 (ou menos) parametrizações para a esteira.")
    
    ids_input = input("\nÍndices selecionados: ")
    
    try:
        # Pega a string, separa por virgula, remove espaços e converte para int
        indices = [int(x.strip()) for x in ids_input.split(',') if x.strip()]
    except ValueError:
        print("Erro ao ler os índices. Digite apenas números separados por vírgula.")
        return
        
    if not indices:
        print("Nenhum índice fornecido.")
        return
        
    selecionados = []
    for idx in indices:
        if idx < 1 or idx > len(candidatos):
            print(f"[AVISO] Índice {idx} ignorado (fora dos limites).")
            continue
            
        selecionados.append(candidatos[idx - 1])
        
    if not selecionados:
        print("\nNenhum candidato válido foi selecionado.")
        return
        
    # O arquivo que a esteira de testes lê. 
    # Mantendo o formato original 'otimizacao_{estrategia}_top10.json' 
    # para nao quebrar a compatibilidade da esteira
    arquivo_saida = dir_otimizacoes / estrategia.lower() / f"otimizacao_{estrategia}_top10.json"
    arquivo_saida.parent.mkdir(parents=True, exist_ok=True)
    
    with open(arquivo_saida, 'w') as f:
        json.dump(selecionados, f, indent=4)
        
    print(f"\n[SUCESSO] Arquivo gerado: {arquivo_saida.name} com {len(selecionados)} parametrizações.")
    print("A esteira de testes de robustez agora pode ser executada para esta estratégia!")

if __name__ == "__main__":
    main()
