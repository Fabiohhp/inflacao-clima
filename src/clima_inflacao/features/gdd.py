"""Graus-dia de crescimento (GDD) e graus-dia de calor acima de um limiar."""

from __future__ import annotations

import numpy as np


def graus_dia(tmax, tmin, t_base: float, t_teto: float):
    """GDD diário com corte nos dois limites: Tmax e Tmin são limitadas a [base, teto].

    Por construção, nunca é negativo nem passa de teto − base.
    """
    tmax_corte = np.clip(tmax, t_base, t_teto)
    tmin_corte = np.clip(tmin, t_base, t_teto)
    return (tmax_corte + tmin_corte) / 2 - t_base


def graus_dia_calor(tmax, limiar: float):
    """Graus-dia acima de `limiar` (estresse térmico): max(Tmax − limiar, 0)."""
    return np.maximum(tmax - limiar, 0.0)
