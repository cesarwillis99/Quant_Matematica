# -*- coding: utf-8 -*-
"""
core/grid_risk.py — Cálculos dinâmicos de parâmetros de risco.

Todas as funções retornam valores em PIPS ou recebem VR em PIPS,
mantendo unidade consistente com o resto do engine.
"""

import numpy as np
import logging
from quant_grid.config import FATOR_PIPS

logger = logging.getLogger(__name__)


def calcular_vr_atual(
    log_returns: np.ndarray,
    close_atual: float,
    janela: int = 50,
) -> float:
    """
    Calcula a Volatilidade Realizada atual em PIPS.

    vr_frac = std(log_returns[-janela:], ddof=1)
    vr_pips = vr_frac * close_atual * FATOR_PIPS

    Parameters
    ----------
    log_returns : np.ndarray
        Array de log-retornos disponíveis até o candle atual.
    close_atual : float
        Preço de fechamento do candle atual (para converter em pips).
    janela : int
        Número de barras para o rolling std (default 50).

    Returns
    -------
    float
        VR em pips. Retorna 0.0 se dados insuficientes (< janela).
    """
    if len(log_returns) < janela:
        return 0.0

    janela_slice = log_returns[-janela:]

    # Ignorar NaNs
    validos = janela_slice[~np.isnan(janela_slice)]
    if len(validos) < 2:
        return 0.0

    vr_frac = float(np.std(validos, ddof=1))
    vr_pips = vr_frac * close_atual * FATOR_PIPS
    return max(vr_pips, 0.0)


def calcular_espacamento_pips(
    vr_pips: float,
    mult_espacamento: float,
) -> float:
    """
    Calcula o espaçamento entre níveis do grid em pips.

    espacamento = mult_espacamento * vr_pips
    clip: [3.0, 50.0] pips

    Parameters
    ----------
    vr_pips : float
        Volatilidade Realizada em pips.
    mult_espacamento : float
        Multiplicador de espaçamento (parâmetro otimizável).

    Returns
    -------
    float
        Espaçamento em pips, clampado em [3.0, 50.0].
    """
    return float(np.clip(mult_espacamento * vr_pips, 3.0, 50.0))


def calcular_stop_drawdown_pips(
    vr_pips: float,
    mult_stop: float,
    n_ordens_atual: int,
) -> float:
    """
    Calcula o stop de drawdown dinâmico em pips.

    stop = mult_stop * vr_pips * n_ordens_atual
    Quanto mais ordens abertas, maior o stop permitido.
    clip: [10.0, 200.0] pips

    Parameters
    ----------
    vr_pips : float
        Volatilidade Realizada em pips.
    mult_stop : float
        Multiplicador do stop (parâmetro otimizável).
    n_ordens_atual : int
        Número atual de ordens preenchidas no grid.

    Returns
    -------
    float
        Stop em pips, clampado em [10.0, 200.0].
    """
    n = max(n_ordens_atual, 1)
    return float(np.clip(mult_stop * vr_pips * n, 10.0, 200.0))


def calcular_alvo_pips(
    vr_pips: float,
    mult_alvo: float,
) -> float:
    """
    Calcula o alvo de saída em pips a partir do preço médio.

    alvo = mult_alvo * vr_pips
    clip: [5.0, 100.0] pips

    Parameters
    ----------
    vr_pips : float
        Volatilidade Realizada em pips.
    mult_alvo : float
        Multiplicador do alvo (parâmetro otimizável).

    Returns
    -------
    float
        Alvo em pips, clampado em [5.0, 100.0].
    """
    return float(np.clip(mult_alvo * vr_pips, 5.0, 100.0))
