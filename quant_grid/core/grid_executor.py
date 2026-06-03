# -*- coding: utf-8 -*-
"""
core/grid_executor.py — Decisões de execução do Grid Trading.

Funções de verificação de preenchimentos múltiplos, alvo dinâmico,
stop híbrido e fechamento obrigatório de sexta-feira.
"""

import logging
from quant_grid.core.grid_engine import GridState

logger = logging.getLogger(__name__)


def verificar_preenchimentos(
    grid: GridState,
    high: float,
    low: float,
    candle_idx: int,
    lot_size: float,
) -> int:
    """
    Verifica TODOS os níveis pendentes do grid e preenche
    os que foram atingidos pelo candle atual.

    Em um único candle de grande movimento, múltiplos níveis
    podem ser preenchidos simultaneamente. Todos os pendentes
    são verificados, não apenas o mais próximo.

    Regras de preenchimento:
        COMPRA (direcao=+1): preenche nível i se low <= niveis[i]
        VENDA  (direcao=-1): preenche nível i se high >= niveis[i]

    O preco_entrada real (com spread) é aplicado em
    grid.preencher_nivel().

    Parameters
    ----------
    grid : GridState
        Grid ativo com níveis pré-calculados.
    high : float
        High do candle atual.
    low : float
        Low do candle atual.
    candle_idx : int
        Índice inteiro do candle atual.
    lot_size : float
        Lote por ordem.

    Returns
    -------
    int
        Número de novos preenchimentos neste candle.
        Zero se nenhum nível foi tocado.
    """
    pendentes = grid.niveis_pendentes()
    novos = 0

    for i in pendentes:
        if grid.direcao == 1:    # COMPRA — nível ativado se low toca ou cruza
            if low <= grid.niveis[i]:
                if grid.preencher_nivel(i, candle_idx, lot_size):
                    novos += 1
        else:                    # VENDA — nível ativado se high toca ou cruza
            if high >= grid.niveis[i]:
                if grid.preencher_nivel(i, candle_idx, lot_size):
                    novos += 1

    return novos


def verificar_alvo(
    grid: GridState,
    high: float,
    low: float,
) -> tuple[bool, float]:
    """
    Verifica se o alvo atual (grid.preco_alvo) foi atingido
    pelo candle corrente.

    O preco_alvo já foi atualizado externamente via
    grid.atualizar_alvo() antes desta chamada.

    Deve ser chamado ANTES de verificar_preenchimentos no loop
    principal — se o preço voltou ao alvo, o grid fecha sem
    abrir mais níveis.

    Regras:
        COMPRA: high >= preco_alvo  → fechar no preco_alvo
        VENDA:  low  <= preco_alvo  → fechar no preco_alvo

    Parameters
    ----------
    grid : GridState
        Grid ativo.
    high : float
        High do candle atual.
    low : float
        Low do candle atual.

    Returns
    -------
    tuple[bool, float]
        (True, preco_alvo) se alvo atingido,
        (False, 0.0) caso contrário.
    """
    # preco_alvo = 0.0 significa que ainda não foi calculado
    if grid.preco_alvo == 0.0:
        return False, 0.0

    if grid.direcao == 1:
        if high >= grid.preco_alvo:
            return True, grid.preco_alvo
    else:
        if low <= grid.preco_alvo:
            return True, grid.preco_alvo

    return False, 0.0


def verificar_stop_hibrido(
    grid: GridState,
    close: float,
    candle_idx: int,
    stop_drawdown_pips: float,
    stop_candles_max: int,
) -> tuple[bool, str]:
    """
    Verifica as duas condições de stop híbrido — ativa a
    primeira que for atingida.

    1. DRAWDOWN:
       pnl_pips = grid.calcular_pnl_pips(close)
       Se pnl_pips < -stop_drawdown_pips → return (True, 'DRAWDOWN')

    2. TEMPO:
       duracao = candle_idx - grid.candle_inicio
       Se duracao >= stop_candles_max → return (True, 'TEMPO')

    Parameters
    ----------
    grid : GridState
        Grid ativo.
    close : float
        Close do candle atual.
    candle_idx : int
        Índice inteiro do candle atual.
    stop_drawdown_pips : float
        Drawdown máximo permitido em pips (positivo).
    stop_candles_max : int
        Duração máxima do grid em número de candles.

    Returns
    -------
    tuple[bool, str]
        (True, motivo) se stop atingido,
        (False, '') caso contrário.
    """
    # 1. Stop por drawdown
    pnl_pips = grid.calcular_pnl_pips(close)
    if pnl_pips < -stop_drawdown_pips:
        return True, 'DRAWDOWN'

    # 2. Stop por tempo
    duracao = candle_idx - grid.candle_inicio
    if duracao >= stop_candles_max:
        return True, 'TEMPO'

    return False, ''


def verificar_fechamento_sexta(candle_dt) -> bool:
    """
    Retorna True se o candle é sexta-feira às 21h55 no
    horário do servidor MT5 (horário naive do índice).

    Parameters
    ----------
    candle_dt : datetime-like
        Elemento do DatetimeIndex do DataFrame.

    Returns
    -------
    bool
        True se weekday == 4, hour == 21, minute == 55.
    """
    return (
        candle_dt.weekday() == 4
        and candle_dt.hour == 21
        and candle_dt.minute == 55
    )
