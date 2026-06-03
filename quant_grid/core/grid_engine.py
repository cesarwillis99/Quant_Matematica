# -*- coding: utf-8 -*-
"""
core/grid_engine.py — Motor genérico do Grid Trading.

Todos os 8 níveis de entrada são pré-calculados na abertura
com a VR congelada. O alvo de saída é recalculado a cada
candle externamente via atualizar_alvo().
"""

import logging
import numpy as np
from quant_grid.config import MAX_ORDENS, FATOR_PIPS, SPREAD_PIPS

logger = logging.getLogger(__name__)


class GridState:
    """
    Estado completo de um grid ativo.

    Todos os 8 níveis de entrada são calculados na abertura e
    permanecem fixos durante toda a vida do grid (VR congelada).
    O alvo de saída é recalculado a cada candle externamente pelo
    backtest via método atualizar_alvo().

    Atributos
    ---------
    direcao : int
        +1 = grid de COMPRA, -1 = grid de VENDA.
    niveis : list[float]
        Lista com exatamente MAX_ORDENS (8) preços pré-calculados.
        niveis[0] = primeiro nível (já preenchido na abertura).
        niveis[1..7] = níveis pendentes.
    niveis_preenchidos : list[bool]
        Lista de 8 booleans. True = ordem já preenchida.
        Na abertura: [True, False, False, ..., False].
    ordens_preenchidas : list[dict]
        Ordens efetivamente preenchidas. Cada dict:
        {
            'nivel_idx': int,        — índice em niveis[]
            'preco_entrada': float,  — preço real com spread
            'lot_size': float,
            'candle_idx': int,
        }
    n_ordens : int
        len(ordens_preenchidas). Atualizado a cada preenchimento.
    preco_medio : float
        Preço médio ponderado por lote das ordens preenchidas.
        Recalculado a cada novo preenchimento.
    lot_total : float
        Soma dos lotes das ordens preenchidas.
    preco_alvo : float
        Último alvo calculado. Atualizado externamente via
        atualizar_alvo(). Inicia em 0.0.
    espacamento_pips : float
        Espaçamento congelado na abertura (em pips).
    vr_abertura : float
        VR em pips no candle de abertura. Congelada.
    candle_inicio : int
        Índice do candle em que o grid foi aberto.
    ativo : bool
        True enquanto o grid estiver aberto.
    """

    def __init__(
        self,
        direcao: int,
        preco_inicial: float,
        lot_size: float,
        espacamento_pips: float,
        vr_abertura: float,
        candle_idx: int,
    ):
        """
        Inicializa o grid e pré-calcula TODOS os MAX_ORDENS (8) níveis.

        Para COMPRA (direcao=+1):
            niveis[0] = preco_inicial - SPREAD/2   (já preenchido)
            niveis[i] = niveis[i-1] - espacamento_pips/FATOR_PIPS

        Para VENDA (direcao=-1):
            niveis[0] = preco_inicial + SPREAD/2   (já preenchido)
            niveis[i] = niveis[i-1] + espacamento_pips/FATOR_PIPS

        Parameters
        ----------
        direcao : int
            +1 = COMPRA, -1 = VENDA.
        preco_inicial : float
            Close do candle de abertura (sem ajuste de spread;
            o spread é descontado internamente no nível 0).
        lot_size : float
            Tamanho do lote por ordem.
        espacamento_pips : float
            Espaçamento entre níveis em pips (já clampado em [3,50]).
        vr_abertura : float
            VR em pips no momento da abertura (congelada).
        candle_idx : int
            Índice do candle de abertura.
        """
        self.direcao = direcao
        self.espacamento_pips = espacamento_pips
        self.vr_abertura = vr_abertura
        self.candle_inicio = candle_idx
        self.ativo = True
        self.preco_alvo = 0.0

        spread_price = SPREAD_PIPS / (2.0 * FATOR_PIPS)
        passo = espacamento_pips / FATOR_PIPS

        # Calcular nível 0 com spread descontado
        if direcao == 1:      # COMPRA — spread subtrai
            nivel_zero = preco_inicial - spread_price
        else:                 # VENDA — spread adiciona
            nivel_zero = preco_inicial + spread_price

        # Pré-calcular TODOS os MAX_ORDENS níveis (base de preço bruto)
        # O desconto de spread é aplicado em preencher_nivel()
        # niveis[] guarda o preço NOMINAL do nível (sem spread)
        # Para nível 0, usamos nivel_zero diretamente
        self.niveis: list[float] = []
        referencia = preco_inicial   # base sem spread para cálculo dos demais
        for i in range(MAX_ORDENS):
            if i == 0:
                self.niveis.append(nivel_zero)   # já tem spread descontado
            else:
                if direcao == 1:
                    self.niveis.append(self.niveis[i - 1] - passo)
                else:
                    self.niveis.append(self.niveis[i - 1] + passo)

        # Estado de preenchimento — apenas o nível 0 está preenchido
        self.niveis_preenchidos: list[bool] = [False] * MAX_ORDENS
        self.niveis_preenchidos[0] = True

        # Registrar primeira ordem
        self.ordens_preenchidas: list[dict] = []
        self.ordens_preenchidas.append({
            'nivel_idx': 0,
            'preco_entrada': nivel_zero,
            'lot_size': lot_size,
            'candle_idx': candle_idx,
        })
        self.n_ordens = 1
        self.lot_total = lot_size
        self.preco_medio = nivel_zero

    # ──────────────────────────────────────────────────────────────
    # Cálculos internos
    # ──────────────────────────────────────────────────────────────

    def calcular_preco_medio(self) -> float:
        """
        Recalcula e retorna o preço médio ponderado por lote
        de todas as ordens preenchidas.

        preco_medio = Σ(preco_i * lot_i) / Σ(lot_i)

        Atualiza self.preco_medio e self.lot_total.
        """
        if not self.ordens_preenchidas:
            return 0.0
        soma_ponderada = sum(
            o['preco_entrada'] * o['lot_size']
            for o in self.ordens_preenchidas
        )
        self.lot_total = sum(o['lot_size'] for o in self.ordens_preenchidas)
        self.preco_medio = soma_ponderada / self.lot_total
        return self.preco_medio

    def preencher_nivel(
        self,
        nivel_idx: int,
        candle_idx: int,
        lot_size: float,
    ) -> bool:
        """
        Preenche o nível nivel_idx se ainda estiver pendente.

        O preco_entrada real com spread é:
            COMPRA: niveis[nivel_idx] - SPREAD_PIPS/(2*FATOR_PIPS)
            VENDA:  niveis[nivel_idx] + SPREAD_PIPS/(2*FATOR_PIPS)

        Para o nível 0 o spread já foi aplicado no __init__;
        para os demais é aplicado aqui.

        Atualiza niveis_preenchidos[nivel_idx] = True,
        adiciona dict a ordens_preenchidas, recalcula preco_medio
        e incrementa n_ordens.

        Returns
        -------
        bool
            True se o nível foi preenchido agora,
            False se já estava preenchido.
        """
        if self.niveis_preenchidos[nivel_idx]:
            return False

        spread_price = SPREAD_PIPS / (2.0 * FATOR_PIPS)
        nivel_base = self.niveis[nivel_idx]

        if self.direcao == 1:    # COMPRA
            preco_real = nivel_base - spread_price
        else:                    # VENDA
            preco_real = nivel_base + spread_price

        self.niveis_preenchidos[nivel_idx] = True
        self.ordens_preenchidas.append({
            'nivel_idx': nivel_idx,
            'preco_entrada': preco_real,
            'lot_size': lot_size,
            'candle_idx': candle_idx,
        })
        self.n_ordens = len(self.ordens_preenchidas)
        self.calcular_preco_medio()
        return True

    def atualizar_alvo(
        self,
        vr_pips_atual: float,
        mult_alvo: float,
    ) -> float:
        """
        Recalcula e atualiza self.preco_alvo com base no
        preço médio atual e na VR atual.

        alvo_pips = clip(mult_alvo * vr_pips_atual, 5.0, 100.0)

        Para COMPRA: preco_alvo = preco_medio + alvo_pips/FATOR_PIPS
        Para VENDA:  preco_alvo = preco_medio - alvo_pips/FATOR_PIPS

        Returns
        -------
        float
            Novo preco_alvo calculado.
        """
        alvo_pips = float(np.clip(mult_alvo * vr_pips_atual, 5.0, 100.0))
        if self.direcao == 1:
            self.preco_alvo = self.preco_medio + (alvo_pips / FATOR_PIPS)
        else:
            self.preco_alvo = self.preco_medio - (alvo_pips / FATOR_PIPS)
        return self.preco_alvo

    def calcular_pnl_flutuante(self, preco_atual: float) -> float:
        """
        PnL flutuante em USD das ordens preenchidas.

        Para COMPRA: (preco_atual - preco_medio) * lot_total * 100000
        Para VENDA:  (preco_medio - preco_atual) * lot_total * 100000
        """
        if self.direcao == 1:
            return (preco_atual - self.preco_medio) * self.lot_total * 100_000.0
        else:
            return (self.preco_medio - preco_atual) * self.lot_total * 100_000.0

    def calcular_pnl_pips(self, preco_saida: float) -> float:
        """
        PnL em pips na saída.

        Para COMPRA: (preco_saida - preco_medio) * FATOR_PIPS
        Para VENDA:  (preco_medio - preco_saida) * FATOR_PIPS
        """
        if self.direcao == 1:
            return (preco_saida - self.preco_medio) * FATOR_PIPS
        else:
            return (self.preco_medio - preco_saida) * FATOR_PIPS

    def niveis_pendentes(self) -> list[int]:
        """
        Retorna lista de índices dos níveis ainda não preenchidos.

        Returns
        -------
        list[int]
            [i for i in range(MAX_ORDENS) if not niveis_preenchidos[i]]
        """
        return [i for i in range(MAX_ORDENS) if not self.niveis_preenchidos[i]]

    def fechar(
        self,
        preco_saida: float,
        motivo: str,
        candle_idx: int,
    ) -> dict:
        """
        Fecha o grid e retorna o dicionário completo do trade.

        Define self.ativo = False.

        Returns
        -------
        dict com as chaves:
            direcao, n_ordens, niveis_ativados, preco_medio_entrada,
            preco_saida, lot_total, pnl_pips, pnl_usd,
            duracao_candles, espacamento_pips, vr_abertura,
            motivo_saida, candle_inicio, candle_fim.
        """
        self.ativo = False

        pnl_pips = self.calcular_pnl_pips(preco_saida)
        pnl_usd = pnl_pips * self.lot_total * 100_000.0 / FATOR_PIPS

        niveis_ativados = [
            o['preco_entrada'] for o in self.ordens_preenchidas
        ]

        return {
            'direcao': self.direcao,
            'n_ordens': self.n_ordens,
            'niveis_ativados': niveis_ativados,
            'preco_medio_entrada': self.preco_medio,
            'preco_saida': preco_saida,
            'lot_total': self.lot_total,
            'pnl_pips': pnl_pips,
            'pnl_usd': pnl_usd,
            'duracao_candles': candle_idx - self.candle_inicio,
            'espacamento_pips': self.espacamento_pips,
            'vr_abertura': self.vr_abertura,
            'motivo_saida': motivo,
            'candle_inicio': self.candle_inicio,
            'candle_fim': candle_idx,
        }
