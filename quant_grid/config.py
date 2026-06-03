# -*- coding: utf-8 -*-
"""
Configurações gerais do Grid Trading para EURUSD H1.
Define caminhos de arquivos, parâmetros padrões, e constantes.
"""

from pathlib import Path

# Diretórios principais do módulo
DIR_GRID        = Path(__file__).resolve().parent
DIR_PROJETO     = DIR_GRID.parent
DIR_QUANT       = DIR_PROJETO / "quant_eurusd_h1"

# Arquivos de dados (parquet) do projeto quant_eurusd_h1
PARQUET_COMPLETO    = DIR_QUANT / "data" / "eurusd_h1_completo.parquet"
PARQUET_OPERACIONAL = DIR_QUANT / "data" / "eurusd_h1_operacional.parquet"

# Diretórios de resultados
DIR_GRAFICOS    = DIR_GRID / "resultados" / "graficos"
DIR_METRICAS    = DIR_GRID / "resultados" / "metricas"
DIR_OTIM        = DIR_GRID / "otimizacoes"

# ── Configurações do Grid ──────────────────────────────────────
MAX_ORDENS          = 8          # máximo absoluto de ordens simultâneas
SPREAD_PIPS         = 1.0        # spread em pips por entrada
PIP_VALUE_POR_LOT   = 10.0       # valor do pip por lote padrão (EURUSD)
FATOR_PIPS          = 10000      # multiplicador para converter preço em pips
LOT_SIZE            = 0.1        # lote padrão por ordem
CAPITAL_INICIAL     = 10000.0    # capital inicial em USD

# ── Horário do servidor MT5 (UTC+2/UTC+3 DST europeu) ────────
# Fechamento obrigatório sexta-feira
HORA_FECHAMENTO_SEXTA = 21       # hora (servidor MT5)
MIN_FECHAMENTO_SEXTA  = 55       # minuto

# ── Parâmetros de stop híbrido (defaults — serão otimizados) ─
STOP_DRAWDOWN_MULT  = 3.0        # múltiplo de VR para drawdown máximo do grid
STOP_CANDLES_MAX    = 120        # candles máximos antes do time stop (5 dias H1)

# ── VR (Volatilidade Realizada) ───────────────────────────────
JANELA_VR           = 50         # janela rolling para cálculo da VR
