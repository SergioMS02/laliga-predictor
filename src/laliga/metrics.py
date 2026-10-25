"""Métricas para probabilidades 1X2. y: 0=local, 1=empate, 2=visitante."""
from __future__ import annotations

import numpy as np
import pandas as pd

RESULT_IDX = {"H": 0, "D": 1, "A": 2}


def _clip(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1)
    return p / p.sum(axis=1, keepdims=True)


def log_loss(p, y) -> float:
    p = _clip(p)
    return float(-np.log(p[np.arange(len(y)), y]).mean())


def brier(p, y) -> float:
    p = _clip(p)
    o = np.eye(3)[y]
    return float(((p - o) ** 2).sum(axis=1).mean())


def rps(p, y) -> float:
    """Ranked Probability Score (respeta el orden local > empate > visitante)."""
    p = _clip(p)
    o = np.eye(3)[y]
    cp, co = np.cumsum(p, axis=1)[:, :2], np.cumsum(o, axis=1)[:, :2]
    return float(((cp - co) ** 2).sum(axis=1).mean() / 2)


def accuracy(p, y) -> float:
    return float((np.asarray(p).argmax(axis=1) == y).mean())


def evaluate(p, y) -> dict:
    return {"n": int(len(y)), "log_loss": log_loss(p, y), "brier": brier(p, y), "rps": rps(p, y),
            "accuracy": accuracy(p, y)}


def calibration_table(p, y, bins: int = 10) -> pd.DataFrame:
    """Probabilidad predicha vs frecuencia observada, agregando las 3 salidas."""
    p = _clip(p)
    o = np.eye(3)[y]
    pf, of = p.ravel(), o.ravel()
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(pf, edges) - 1, 0, bins - 1)
    df = pd.DataFrame({"bin": b, "p": pf, "o": of})
    t = df.groupby("bin").agg(pred=("p", "mean"), obs=("o", "mean"), n=("o", "size")).reset_index()
    t["range"] = [f"{edges[i]:.1f}-{edges[i + 1]:.1f}" for i in t.bin]
    return t[["range", "pred", "obs", "n"]]
