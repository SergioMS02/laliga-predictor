"""Mercado recalibrado: en vez de intentar ganar a las casas, corregir sus sesgos.

Las probabilidades implícitas de Bet365 (sin margen) se pasan por una regresión logística
multinomial sobre sus logaritmos. Con tres coeficientes por resultado puede aprender el sesgo
favorito-sorpresa (los favoritos claros ganan algo más de lo que pagan sus cuotas y las
sorpresas, algo menos) sin ver nada más que las propias cuotas.

Todo es walk-forward: la temporada S se predice con un modelo entrenado solo con las
`window` temporadas anteriores. Así se compara de forma honesta con:
  - mercado: las mismas cuotas de Bet365 sin margen (lo que se recalibra),
  - cierre: las cuotas de cierre (lo mejor que sabe el mercado justo antes del partido).
Si el recalibrado no mejora al mercado de partida fuera de muestra, no hay sesgo aprovechable.

Salidas: data/state/market_recal.json (coeficientes con todo el histórico, para el directo),
predictions/market_recal.json (evaluación por temporada y calibración por tramo de cuota) y
data/processed/market_recal_oos.csv (probabilidades fuera de muestra, para el backtest de apuestas).
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from .config import Settings

log = logging.getLogger(__name__)

ODDS = ("b365_h", "b365_d", "b365_a")
CLOSE = ("b365c_h", "b365c_d", "b365c_a")
EPS = 1e-6


def devig(odds) -> np.ndarray:
    """Cuotas (n, 3) -> probabilidades proporcionales sin margen; NaN si falta alguna cuota."""
    o = np.asarray(odds, float).reshape(-1, 3)
    bad = ~np.isfinite(o).all(1) | (o <= 1).any(1)
    inv = 1 / np.where(bad[:, None], np.nan, o)
    return inv / inv.sum(1, keepdims=True)


def features(p: np.ndarray) -> np.ndarray:
    return np.log(np.clip(p, EPS, 1))


def fit(p: np.ndarray, y: np.ndarray, C: float = 1.0) -> dict:
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=C, max_iter=1000).fit(features(p), y)
    return {"coef": np.round(m.coef_, 6).tolist(), "intercept": np.round(m.intercept_, 6).tolist(),
            "classes": [int(c) for c in m.classes_], "n": int(len(y))}


def apply(model: dict, p: np.ndarray) -> np.ndarray:
    """Probabilidades recalibradas (n, 3) en orden local/empate/visitante."""
    p = np.asarray(p, float).reshape(-1, 3)
    z = features(p) @ np.asarray(model["coef"]).T + np.asarray(model["intercept"])
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    out = np.zeros_like(p)
    out[:, model["classes"]] = e / e.sum(1, keepdims=True)
    return out


def recal_from_odds(model: dict | None, b365) -> list[float] | None:
    if not model or b365 is None:
        return None
    p = devig([b365])
    if not np.isfinite(p).all():
        return None
    return [round(float(x), 4) for x in apply(model, p)[0]]


def load_model(settings: Settings) -> dict | None:
    p = settings.state_dir / "market_recal.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except Exception:
        return None


def _y(m: pd.DataFrame) -> np.ndarray:
    return np.where(m.hg > m.ag, 0, np.where(m.hg == m.ag, 1, 2))


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], EPS, 1))))


def _boot_diff(a: np.ndarray, b: np.ndarray, y: np.ndarray, n: int = 2000, seed: int = 0) -> list[float]:
    """IC 95 % de la diferencia de log-loss (a - b) remuestreando partidos."""
    la = -np.log(np.clip(a[np.arange(len(y)), y], EPS, 1))
    lb = -np.log(np.clip(b[np.arange(len(y)), y], EPS, 1))
    d = la - lb
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def run(settings: Settings, matches: pd.DataFrame) -> dict | None:
    """Evaluación walk-forward + modelo final. Devuelve el resumen (también se escribe a disco)."""
    if not set(ODDS) <= set(matches.columns):
        return None
    m = matches[matches.hg.notna()].copy()
    p = devig(m[list(ODDS)].to_numpy(float))
    ok = np.isfinite(p).all(1)
    m, p = m[ok], p[ok]
    y = _y(m)
    seasons = m.season.to_numpy().astype(int)
    window = int(settings.get("market_recal", "window", 10))
    first = int(settings.get("backtest", "first_test_season", 2021))
    oos = np.full_like(p, np.nan)
    per_season = []
    for s in sorted(set(seasons)):
        if s < first:
            continue
        tr = (seasons < s) & (seasons >= s - window)
        te = seasons == s
        if tr.sum() < 1000 or not te.any():
            continue
        oos[te] = apply(fit(p[tr], y[tr]), p[te])
        per_season.append({"season": int(s), "n": int(te.sum()),
                           "mercado": round(log_loss(p[te], y[te]), 4),
                           "recalibrado": round(log_loss(oos[te], y[te]), 4)})
    if not per_season:
        return None
    t = np.isfinite(oos).all(1)
    res = {"window": window, "n": int(t.sum()),
           "log_loss": {"mercado": round(log_loss(p[t], y[t]), 4), "recalibrado": round(log_loss(oos[t], y[t]), 4)},
           "diff_vs_mercado": {"mean": round(log_loss(oos[t], y[t]) - log_loss(p[t], y[t]), 4),
                               "ci95": _boot_diff(oos[t], p[t], y[t])},
           "by_season": per_season}
    if set(CLOSE) <= set(m.columns):
        pc = devig(m[list(CLOSE)].to_numpy(float))
        tc = t & np.isfinite(pc).all(1)
        if tc.sum() > 300:
            res["con_cierre"] = {"n": int(tc.sum()), "mercado": round(log_loss(p[tc], y[tc]), 4),
                                 "recalibrado": round(log_loss(oos[tc], y[tc]), 4),
                                 "cierre": round(log_loss(pc[tc], y[tc]), 4),
                                 "diff_vs_cierre_ci95": _boot_diff(oos[tc], pc[tc], y[tc])}
    # calibración del favorito por tramo de probabilidad implícita: ¿ganan más de lo que dice la cuota?
    fav = p[t].argmax(1)
    pf = p[t][np.arange(t.sum()), fav]
    rf = oos[t][np.arange(t.sum()), fav]
    hit = (y[t] == fav).astype(float)
    bins = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    res["favoritos"] = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        k = (pf >= lo) & (pf < hi)
        if k.sum() >= 30:
            res["favoritos"].append({"tramo": f"{int(100 * lo)}-{int(100 * hi)} %", "n": int(k.sum()),
                                     "cuota_dice": round(float(pf[k].mean()), 3),
                                     "recalibrado_dice": round(float(rf[k].mean()), 3),
                                     "ganaron": round(float(hit[k].mean()), 3)})
    # modelo para el directo: las últimas `window` temporadas completas con resultado
    last = seasons.max()
    full = (seasons >= last - window) & (seasons <= last)
    final = fit(p[full], y[full])
    (settings.state_dir / "market_recal.json").write_text(json.dumps(final), encoding="utf-8")
    pd.DataFrame(oos, index=m.index, columns=["recal_H", "recal_D", "recal_A"]).dropna().to_csv(
        settings.processed_dir / "market_recal_oos.csv")
    (settings.predictions_dir / "market_recal.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    return res
