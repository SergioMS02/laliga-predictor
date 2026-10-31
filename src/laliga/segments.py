"""¿Dónde discrepa el modelo del mercado, y quién acierta cuando discrepan?

Con las predicciones fuera de muestra del backtest (temporadas de prueba) se compara, segmento a
segmento, el log-loss del modelo propio (sin cuotas) con el de las cuotas de Bet365 antes del partido.
Si el modelo tiene ventaja en algún sitio, estará en un rincón concreto (primeras jornadas, recién
ascendidos, tras jugar en Europa...), no en la media. Cada diferencia lleva su intervalo de confianza
del 95 % (bootstrap por partidos): si cruza el 0, el segmento no dice nada.

Salida: predictions/segments.json.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from .config import Settings

log = logging.getLogger(__name__)
EPS = 1e-6


def _ll(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(p[np.arange(len(y)), y], EPS, 1))


def _ci(d: np.ndarray, n: int = 2000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def compare(model: np.ndarray, market: np.ndarray, y: np.ndarray, mask: np.ndarray, min_n: int = 60) -> dict | None:
    k = mask & np.isfinite(model).all(1) & np.isfinite(market).all(1)
    if k.sum() < min_n:
        return None
    lm, lk = _ll(model[k], y[k]), _ll(market[k], y[k])
    d = lm - lk
    ci = _ci(d)
    return {"n": int(k.sum()), "modelo": round(float(lm.mean()), 4), "mercado": round(float(lk.mean()), 4),
            "diff": round(float(d.mean()), 4), "ci95": ci,
            # partidos en los que el modelo dio más probabilidad que el mercado a lo que pasó
            "modelo_mas_cerca": round(float((lm < lk).mean()), 3),
            "veredicto": "modelo mejor" if ci[1] < 0 else "mercado mejor" if ci[0] > 0 else "sin diferencia clara"}


def run(settings: Settings, matches: pd.DataFrame) -> dict | None:
    bp = settings.processed_dir / "backtest_predictions.csv"
    tp = settings.processed_dir / "oos_table.csv"
    if not bp.exists() or not tp.exists():
        return None
    bt = pd.read_csv(bp, index_col=0)
    if "mercado_H" not in bt:
        return None
    keys = matches.loc[matches.index.intersection(bt.index), ["season", "home", "away"]]
    bt = bt.loc[keys.index].join(keys.drop(columns="season"))
    cols = ["season", "home", "away", "early_season", "new_home", "new_away", "euro_prev_home", "euro_prev_away",
            "midweek", "post_break", "derby", "new_coach_home", "new_coach_away"]
    t = pd.read_csv(tp, usecols=lambda c: c in cols, low_memory=False).drop_duplicates(["season", "home", "away"])
    df = bt.reset_index().merge(t, on=["season", "home", "away"], how="left").set_index(bt.index.name or "index")
    y = df.y.to_numpy().astype(int)
    model = df[["stacker_H", "stacker_D", "stacker_A"]].to_numpy(float)
    market = df[["mercado_H", "mercado_D", "mercado_A"]].to_numpy(float)
    f = lambda c: (df[c].fillna(0).to_numpy(float) > 0) if c in df else np.zeros(len(df), bool)  # noqa: E731
    fav = np.nanmax(market, 1)
    gap = np.nanmax(np.abs(model - market), 1)
    segs = [
        ("Todos los partidos", np.ones(len(df), bool)),
        ("Primeras jornadas (algún equipo con menos de 3 partidos)", f("early_season")),
        ("Juega un recién ascendido", f("new_home") | f("new_away")),
        ("Algún equipo viene de jugar en Europa", f("euro_prev_home") | f("euro_prev_away")),
        ("Entre semana", f("midweek")),
        ("Vuelta de parón internacional", f("post_break")),
        ("Derbi", f("derby")),
        ("Entrenador nuevo", f("new_coach_home") | f("new_coach_away")),
        ("Favorito claro (≥ 60 % según el mercado)", fav >= 0.6),
        ("Partido igualado (favorito < 45 %)", fav < 0.45),
        ("Modelo y mercado discrepan ≥ 8 puntos", gap >= 0.08),
        ("Modelo y mercado discrepan ≥ 12 puntos", gap >= 0.12),
    ]
    out = []
    for label, mask in segs:
        r = compare(model, market, y, mask)
        if r:
            out.append({"segmento": label, **r})
    res = {"seasons": sorted(int(s) for s in df.season.unique()), "modelo": "stacker (sin cuotas)",
           "mercado": "Bet365 antes del partido, sin margen", "segmentos": out,
           "nota": "diff = log-loss del modelo − log-loss del mercado: negativo significa que el modelo acierta más."}
    (settings.predictions_dir / "segments.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    return res
