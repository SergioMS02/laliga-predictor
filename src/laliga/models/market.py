"""Probabilidades implícitas en cuotas, eliminando el margen de la casa (método de potencias)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def devig_power(odds: np.ndarray) -> np.ndarray:
    """odds: array (n, 3) de cuotas decimales. Devuelve probabilidades que suman 1.

    Busca k tal que sum((1/o)^k) = 1. Reparte el margen de forma no proporcional
    (corrige el sesgo favorito-sorpresa mejor que la normalización simple)."""
    odds = np.asarray(odds, dtype=float)
    out = np.full(odds.shape, np.nan)
    for r, o in enumerate(odds):
        if np.isnan(o).any() or (o <= 1).any():
            continue
        q = 1.0 / o
        if abs(q.sum() - 1) < 1e-9:
            out[r] = q
            continue
        try:
            k = brentq(lambda k: (q ** k).sum() - 1, 0.5, 3.0)
            out[r] = q ** k
        except ValueError:
            out[r] = q / q.sum()
    return out


def devig_proportional(odds: np.ndarray) -> np.ndarray:
    q = 1.0 / np.asarray(odds, dtype=float)
    return q / q.sum(axis=1, keepdims=True)


def overround(odds: np.ndarray) -> np.ndarray:
    return (1.0 / np.asarray(odds, dtype=float)).sum(axis=1) - 1
