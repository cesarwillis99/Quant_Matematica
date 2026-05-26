# -*- coding: utf-8 -*-
"""
================================================================================
main.py — Orquestrador Principal do Projeto Quant EURUSD
================================================================================
Autor: Quant Developer Sênior
Data: 2026

Descrição:
    Ponto de entrada único para execução completa do pipeline quantitativo.
    Importa e executa cada módulo em sequência, propagando os parâmetros
    de linha de comando para todos os sub-módulos.

Sequência de execução:
    1. data_loader.py  → Carregamento M1 → H1, log-retornos, Parquet
    2. hurst.py        → Expoente de Hurst R/S, classificação de regimes
    3. zscore.py       → Z-Score do preço, sinais de Mean Reversion
    4. momentum.py     → Cinemática + Entropia de Shannon, sinais de Tendência
    5. backtest.py     → Simulação histórica, métricas, equity curves
    6. Resumo executivo final no terminal

Parâmetros de linha de comando:
    --csv          Caminho para o arquivo CSV bruto (padrão: auto-detectado)
    --inicio       Data de início do backtest (YYYY-MM-DD)
    --fim          Data de fim do backtest (YYYY-MM-DD)
    --ativo        Identificador do ativo (padrão: EURUSD — para expansão futura)
    --timeframe    Timeframe alvo (padrão: H1 — para expansão futura)
    --forcar       Reprocessar todos os módulos mesmo com cache existente
    --so-backtest  Pular processamento e ir direto ao backtest (usa cache)
    --so-graficos  Regenerar apenas os gráficos (sem recalcular indicadores)

Exemplos de uso:
    python main.py
    python main.py --inicio 2018-01-01 --fim 2026-04-10
    python main.py --csv /caminho/EURUSD.csv --forcar
    python main.py --so-backtest --inicio 2020-01-01
================================================================================
"""

import sys
import time
import logging
import argparse
import traceback
from pathlib import Path
from datetime import datetime

# =============================================================================
# CONFIGURAÇÃO DE LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# CAMINHOS DO PROJETO
# =============================================================================
DIR_PROJETO  = Path(__file__).resolve().parent
DIR_DATA     = DIR_PROJETO / "data"
DIR_GRAFICOS = DIR_PROJETO / "graficos"
DIR_RESULTADOS = DIR_PROJETO / "resultados"


# =============================================================================
# BANNER E UTILITÁRIOS VISUAIS
# =============================================================================

def exibir_banner() -> None:
    """
    Exibe o banner de boas-vindas do projeto no terminal.
    """
    print("\n" + "=" * 65)
    print("  ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗")
    print("  ██╔═══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝")
    print("  ██║   ██║██║   ██║███████║██╔██╗ ██║   ██║   ")
    print("  ██║▄▄ ██║██║   ██║██╔══██║██║╚██╗██║   ██║   ")
    print("  ╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║   ██║   ")
    print("   ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝   ")
    print()
    print("  Plataforma de Análise Quantitativa — EURUSD H1")
    print("  Estratégias: Mean Reversion (Z-Score) + Trend Following")
    print("  Indicadores: Hurst R/S | Z-Score | Entropia de Shannon")
    print("=" * 65)
    print(f"  Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65 + "\n")


def exibir_etapa(numero: int, total: int, nome: str) -> None:
    """
    Exibe o cabeçalho de uma etapa do pipeline.

    Parâmetros:
        numero: int — número da etapa atual
        total:  int — total de etapas
        nome:   str — descrição da etapa
    """
    print(f"\n{'─'*65}")
    print(f"  ETAPA {numero}/{total}: {nome}")
    print(f"{'─'*65}")


def exibir_ok(nome: str, duracao: float) -> None:
    """
    Exibe mensagem de conclusão de uma etapa com tempo decorrido.

    Parâmetros:
        nome:     str   — nome da etapa
        duracao:  float — tempo em segundos
    """
    print(f"  [OK] {nome} concluído em {duracao:.1f}s")


def exibir_erro(nome: str, erro: Exception) -> None:
    """
    Exibe mensagem de erro formatada para uma etapa com falha.

    Parâmetros:
        nome: str       — nome da etapa que falhou
        erro: Exception — exceção capturada
    """
    print(f"\n{'!'*65}")
    print(f"  [ERRO] Falha na etapa: {nome}")
    print(f"  Mensagem: {erro}")
    print(f"{'!'*65}\n")


# =============================================================================
# EXECUÇÃO DO PIPELINE
# =============================================================================

def executar_pipeline(args: argparse.Namespace) -> bool:
    """
    Executa o pipeline completo de análise quantitativa em sequência.

    O pipeline é tolerante a falhas: se uma etapa falhar, o erro é exibido
    e a execução para. Módulos com cache (Parquet já existente) são pulados
    a menos que --forcar seja especificado.

    Sequência:
        Etapa 1: data_loader  → CSV M1 → Parquet H1 limpo
        Etapa 2: hurst        → Expoente de Hurst + regimes
        Etapa 3: zscore       → Z-Score + sinais de reversão
        Etapa 4: momentum     → Cinemática + Entropia + sinais de tendência
        Etapa 5: backtest     → Simulação + métricas + gráficos

    Parâmetros:
        args: argparse.Namespace — argumentos de linha de comando parseados

    Retorna:
        bool — True se pipeline completo com sucesso, False se houve erro
    """
    tempo_total_inicio = time.time()
    tempos: dict = {}
    TOTAL_ETAPAS = 5

    # ── Etapa 1: Carregamento de Dados ────────────────────────────────────────
    if not args.so_backtest and not args.so_graficos:
        exibir_etapa(1, TOTAL_ETAPAS, "DATA LOADER — M1 → H1 + Log-Retornos")
        t0 = time.time()
        try:
            from data_loader import carregar_e_processar
            csv_path = Path(args.csv) if args.csv else None

            # Detectar CSV automaticamente se não especificado
            if csv_path is None:
                candidatos = [
                    DIR_PROJETO.parent / "EURUSD_10ANOS.csv",
                    DIR_PROJETO.parent / "EURUSD.csv",
                    DIR_PROJETO / "EURUSD_10ANOS.csv",
                ]
                for c in candidatos:
                    if c.exists():
                        csv_path = c
                        break
                if csv_path is None:
                    raise FileNotFoundError(
                        "Arquivo CSV não encontrado. Use --csv /caminho/arquivo.csv"
                    )

            df_h1 = carregar_e_processar(
                csv_path=csv_path,
                data_inicio=args.inicio,
                data_fim=args.fim,
                forcar_reprocessamento=args.forcar,
            )
            duracao = time.time() - t0
            tempos["data_loader"] = duracao
            exibir_ok("Data Loader", duracao)

        except Exception as e:
            exibir_erro("Data Loader", e)
            logger.debug(traceback.format_exc())
            return False
    else:
        print("  [PULADO] Data Loader (modo so-backtest/so-graficos ativo)")

    # ── Etapa 2: Expoente de Hurst ────────────────────────────────────────────
    if not args.so_backtest and not args.so_graficos:
        exibir_etapa(2, TOTAL_ETAPAS, "HURST — Análise R/S + Classificação de Regimes")
        t0 = time.time()
        try:
            from hurst import calcular_e_salvar_hurst
            df_hurst = calcular_e_salvar_hurst(
                data_inicio=args.inicio,
                data_fim=args.fim,
                forcar_reprocessamento=args.forcar,
            )
            duracao = time.time() - t0
            tempos["hurst"] = duracao
            exibir_ok("Hurst", duracao)

        except Exception as e:
            exibir_erro("Hurst", e)
            logger.debug(traceback.format_exc())
            return False
    else:
        print("  [PULADO] Hurst (modo so-backtest/so-graficos ativo)")

    # ── Etapa 3: Z-Score ──────────────────────────────────────────────────────
    if not args.so_backtest and not args.so_graficos:
        exibir_etapa(3, TOTAL_ETAPAS, "ZSCORE — Mean Reversion + Sinais Condicionados")
        t0 = time.time()
        try:
            from zscore import calcular_e_salvar_zscore
            df_zscore = calcular_e_salvar_zscore(
                data_inicio=args.inicio,
                data_fim=args.fim,
                forcar_reprocessamento=args.forcar,
            )
            duracao = time.time() - t0
            tempos["zscore"] = duracao
            exibir_ok("ZScore", duracao)

        except Exception as e:
            exibir_erro("ZScore", e)
            logger.debug(traceback.format_exc())
            return False
    else:
        print("  [PULADO] ZScore (modo so-backtest/so-graficos ativo)")

    # ── Etapa 4: Momentum ─────────────────────────────────────────────────────
    if not args.so_backtest and not args.so_graficos:
        exibir_etapa(4, TOTAL_ETAPAS, "MOMENTUM — Cinemática + Entropia de Shannon")
        t0 = time.time()
        try:
            from momentum import calcular_e_salvar_momentum
            df_momentum = calcular_e_salvar_momentum(
                data_inicio=args.inicio,
                data_fim=args.fim,
                forcar_reprocessamento=args.forcar,
            )
            duracao = time.time() - t0
            tempos["momentum"] = duracao
            exibir_ok("Momentum", duracao)

        except Exception as e:
            exibir_erro("Momentum", e)
            logger.debug(traceback.format_exc())
            return False
    else:
        print("  [PULADO] Momentum (modo so-backtest/so-graficos ativo)")

    # ── Etapa 5: Backtest ─────────────────────────────────────────────────────
    if not args.so_graficos:
        exibir_etapa(5, TOTAL_ETAPAS, "BACKTEST — Simulação + Métricas + Equity Curves")
        t0 = time.time()
        try:
            from backtest import executar_backtest
            executar_backtest(
                data_inicio=args.inicio,
                data_fim=args.fim,
            )
            duracao = time.time() - t0
            tempos["backtest"] = duracao
            exibir_ok("Backtest", duracao)

        except Exception as e:
            exibir_erro("Backtest", e)
            logger.debug(traceback.format_exc())
            return False
    else:
        print("  [PULADO] Backtest (modo so-graficos ativo)")

    # ── Resumo de tempo de execução ───────────────────────────────────────────
    tempo_total = time.time() - tempo_total_inicio
    _exibir_resumo_final(tempos, tempo_total, args)

    return True


def _exibir_resumo_final(
    tempos:      dict,
    tempo_total: float,
    args:        argparse.Namespace,
) -> None:
    """
    Exibe o resumo executivo final do pipeline com tempos e arquivos gerados.

    Parâmetros:
        tempos:      dict — tempos de execução por etapa {nome: segundos}
        tempo_total: float — tempo total do pipeline em segundos
        args:        Namespace — argumentos de linha de comando
    """
    print(f"\n{'='*65}")
    print(f"  PIPELINE CONCLUÍDO COM SUCESSO")
    print(f"{'='*65}")
    print(f"  Ativo         : {args.ativo}")
    print(f"  Timeframe     : {args.timeframe}")
    if args.inicio or args.fim:
        print(f"  Período       : {args.inicio or 'início'} → {args.fim or 'fim'}")
    print()

    # Tempos por etapa
    if tempos:
        print(f"  ── Tempo de Execução por Etapa ──")
        mapa_nomes = {
            "data_loader": "Data Loader (M1→H1)",
            "hurst":       "Hurst (R/S rolling)",
            "zscore":      "Z-Score + Sinais",
            "momentum":    "Momentum + Entropia",
            "backtest":    "Backtest + Métricas",
        }
        for chave, nome in mapa_nomes.items():
            if chave in tempos:
                print(f"  {nome:<28}: {tempos[chave]:>6.1f}s")
        print(f"  {'─'*40}")
        print(f"  {'TOTAL':<28}: {tempo_total:>6.1f}s")

    # Arquivos gerados
    print(f"\n  ── Arquivos Gerados ──")
    arquivos = [
        (DIR_DATA / "eurusd_h1_clean.parquet",    "H1 limpo (base)"),
        (DIR_DATA / "eurusd_h1_hurst.parquet",    "H1 + Hurst + Regime"),
        (DIR_DATA / "eurusd_h1_zscore.parquet",   "H1 + Z-Score + Sinais"),
        (DIR_DATA / "eurusd_h1_momentum.parquet", "H1 + Momentum + Entropia"),
        (DIR_GRAFICOS / "hurst.png",              "Gráfico Hurst"),
        (DIR_GRAFICOS / "zscore_sinais.png",      "Gráfico Z-Score"),
        (DIR_GRAFICOS / "momentum_sinais.png",    "Gráfico Momentum"),
        (DIR_GRAFICOS / "equity_curves.png",      "Equity Curves"),
        (DIR_RESULTADOS / "operacoes.csv",        "Tabela de Operações"),
        (DIR_RESULTADOS / "metricas.csv",         "Métricas de Performance"),
    ]

    for caminho, descricao in arquivos:
        if caminho.exists():
            tamanho = caminho.stat().st_size
            if tamanho >= 1024 * 1024:
                tam_str = f"{tamanho/1024/1024:.1f} MB"
            else:
                tam_str = f"{tamanho/1024:.0f} KB"
            print(f"  [OK] {descricao:<30}: {tam_str}")
        else:
            print(f"  [--] {descricao:<30}: (não gerado)")

    print(f"\n{'='*65}")
    print(f"  Projeto Quant EURUSD finalizado em {tempo_total:.1f}s")
    print(f"  Fim: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*65}\n")


# =============================================================================
# PARSER DE ARGUMENTOS
# =============================================================================

def criar_parser() -> argparse.ArgumentParser:
    """
    Cria e configura o parser de argumentos de linha de comando.

    Argumentos disponíveis:
        --csv         Caminho para o CSV de entrada
        --inicio      Data de início (YYYY-MM-DD)
        --fim         Data de fim (YYYY-MM-DD)
        --ativo       Identificador do ativo (default: EURUSD)
        --timeframe   Timeframe (default: H1)
        --forcar      Reprocessar mesmo com cache
        --so-backtest Pular processamento, ir direto ao backtest
        --so-graficos Regenerar apenas gráficos

    Retorna:
        argparse.ArgumentParser configurado
    """
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Plataforma de Análise Quantitativa EURUSD\n"
            "Estratégias: Z-Score Mean Reversion + Momentum Trend Following\n"
            "Indicadores: Expoente de Hurst | Z-Score | Entropia de Shannon"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  # Execução completa (detecta CSV automaticamente):
  python main.py

  # Com CSV específico e período definido:
  python main.py --csv C:/dados/EURUSD.csv --inicio 2018-01-01 --fim 2026-04-10

  # Forçar reprocessamento completo (ignora cache):
  python main.py --forcar

  # Apenas backtest (usa Parquets existentes):
  python main.py --so-backtest --inicio 2020-01-01

  # Apenas gráficos (usa dados já calculados):
  python main.py --so-graficos

Estrutura de saída gerada:
  quant_eurusd/
  ├── data/
  │   ├── eurusd_h1_clean.parquet
  │   ├── eurusd_h1_hurst.parquet
  │   ├── eurusd_h1_zscore.parquet
  │   └── eurusd_h1_momentum.parquet
  ├── graficos/
  │   ├── hurst.png
  │   ├── zscore_sinais.png
  │   ├── momentum_sinais.png
  │   └── equity_curves.png
  └── resultados/
      ├── operacoes.csv
      └── metricas.csv
        """
    )

    # Parâmetros de entrada
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        metavar="CAMINHO",
        help=(
            "Caminho completo para o arquivo CSV de entrada "
            "(padrão: detectado automaticamente no diretório pai)"
        )
    )
    parser.add_argument(
        "--inicio",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Data de início do período de análise (padrão: início dos dados)"
    )
    parser.add_argument(
        "--fim",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Data de fim do período de análise (padrão: fim dos dados)"
    )

    # Parâmetros de ativo e timeframe (para expansão futura)
    parser.add_argument(
        "--ativo",
        type=str,
        default="EURUSD",
        metavar="SIMBOLO",
        help="Símbolo do ativo (padrão: EURUSD — suporte a outros pares no futuro)"
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="H1",
        metavar="TF",
        help="Timeframe de análise (padrão: H1 — suporte a outros TFs no futuro)"
    )

    # Flags de controle de execução
    parser.add_argument(
        "--forcar",
        action="store_true",
        help=(
            "Forçar reprocessamento completo de todos os módulos, "
            "ignorando Parquets em cache"
        )
    )
    parser.add_argument(
        "--so-backtest",
        action="store_true",
        dest="so_backtest",
        help=(
            "Pular etapas 1-4 e executar apenas o backtest "
            "(requer Parquets existentes em data/)"
        )
    )
    parser.add_argument(
        "--so-graficos",
        action="store_true",
        dest="so_graficos",
        help=(
            "Regenerar apenas os gráficos "
            "(requer todos os Parquets existentes em data/)"
        )
    )

    return parser


# =============================================================================
# PONTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    # Garantir que o diretório do módulo está no sys.path
    # (permite execução de qualquer diretório)
    sys.path.insert(0, str(DIR_PROJETO))

    # Exibir banner
    exibir_banner()

    # Parsear argumentos
    parser = criar_parser()
    args = parser.parse_args()

    # Validar datas se fornecidas
    for nome_arg, valor_arg in [("--inicio", args.inicio), ("--fim", args.fim)]:
        if valor_arg is not None:
            try:
                from datetime import datetime as dt
                dt.strptime(valor_arg, "%Y-%m-%d")
            except ValueError:
                print(f"  [ERRO] Formato de data inválido para {nome_arg}: '{valor_arg}'")
                print(f"         Use o formato YYYY-MM-DD (ex: 2020-01-15)")
                sys.exit(1)

    # Verificar conflito de flags
    if args.so_backtest and args.so_graficos:
        print("  [ERRO] --so-backtest e --so-graficos são mutuamente exclusivos.")
        sys.exit(1)

    # Exibir configuração da execução
    print(f"  Configuração:")
    print(f"    Ativo     : {args.ativo}")
    print(f"    Timeframe : {args.timeframe}")
    print(f"    Início    : {args.inicio or '(início dos dados)'}")
    print(f"    Fim       : {args.fim    or '(fim dos dados)'}")
    print(f"    Forçar    : {'Sim' if args.forcar else 'Não'}")
    if args.so_backtest:
        print(f"    Modo      : Apenas Backtest")
    elif args.so_graficos:
        print(f"    Modo      : Apenas Gráficos")
    else:
        print(f"    Modo      : Pipeline Completo")
    print()

    # Executar pipeline
    sucesso = executar_pipeline(args)

    # Código de saída
    sys.exit(0 if sucesso else 1)
