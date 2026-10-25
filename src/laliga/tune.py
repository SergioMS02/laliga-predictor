"""Ajuste de hiperparámetros SOLO con temporadas de validación (anteriores a las de test).

1. Rejilla de Dixon-Coles (decaimiento, regularización, ventana).
2. Selección hacia delante de grupos de variables del meta-modelo + C.
Resultado en data/state/tuned.json, que `load_settings` aplica automáticamente.
"""
from __future__ import annotations

import itertools
import json
import logging

import numpy as np
import pandas as pd

from . import metrics, pipeline
from .config import Settings, load_settings
from .models.stacker import GROUPS, Stacker

log = logging.getLogger(__name__)

DC_GRID = {"xi": [0.0008, 0.0012, 0.0019, 0.0028], "ridge": [3.0, 6.0, 12.0], "window_days": [1100, 1500]}
C_GRID = [0.003, 0.01, 0.03, 0.1]
MIN_GAIN = 0.0005        # mejora media mínima de log-loss para aceptar un grupo
MIN_WIN_SHARE = 0.6      # ...y debe mejorar en al menos el 60 % de las temporadas de validación
MIN_COVERAGE = 0.5       # el grupo debe tener datos en al menos la mitad de las filas de validación


def _ll(df: pd.DataFrame, cols=("dc_ph", "dc_pd", "dc_pa")) -> float:
    ok = df.result.notna() & df[list(cols)].notna().all(axis=1)
    return metrics.log_loss(df.loc[ok, list(cols)].to_numpy(float), df.loc[ok, "result"].map(metrics.RESULT_IDX).to_numpy())


def stacker_cv_detail(tab: pd.DataFrame, seasons, groups, C, use_market=False) -> tuple[float, dict]:
    ps, ys, per = [], [], {}
    for s in seasons:
        tr, te = tab[tab.season < s], tab[tab.season == s]
        if use_market:
            from .models.stacker import has_market
            te = te[has_market(te)]
        if len(tr) < 500 or te.empty:
            continue
        st = Stacker(use_market=use_market, C=C, groups=list(groups)).fit(tr)
        p = st.predict_proba(te)
        y = te.result.map(metrics.RESULT_IDX).to_numpy()
        per[s] = metrics.log_loss(p, y)
        ps.append(p)
        ys.append(y)
    return metrics.log_loss(np.vstack(ps), np.concatenate(ys)), per


def stacker_cv(tab: pd.DataFrame, seasons, groups, C, use_market=False) -> float:
    return stacker_cv_detail(tab, seasons, groups, C, use_market)[0]


def group_coverage(tab: pd.DataFrame, seasons, group: str) -> float:
    from .models.stacker import engineer
    X = engineer(tab[tab.season.isin(list(seasons))])
    cols = GROUPS[group]
    return float(X[cols].notna().mean().mean()) if len(X) else 0.0


def forward_select(tab: pd.DataFrame, seasons, use_market=False) -> tuple[list[str], float, float]:
    """Selección hacia delante conservadora: un grupo entra solo si mejora la media,
    mejora en la mayoría de temporadas y tiene datos suficientes en validación."""
    eligible = [g for g in GROUPS if group_coverage(tab, seasons, g) >= MIN_COVERAGE]
    log.info("Grupos con datos suficientes en validación: %s", eligible)
    best = (None, float("inf"), None)
    for C in C_GRID:
        sel: list[str] = []
        cur, cur_per = stacker_cv_detail(tab, seasons, sel, C, use_market)
        while True:
            cands = []
            for g in eligible:
                if g in sel:
                    continue
                ll, per = stacker_cv_detail(tab, seasons, sel + [g], C, use_market)
                wins = np.mean([per[s] < cur_per[s] for s in per if s in cur_per])
                if ll < cur - MIN_GAIN and wins >= MIN_WIN_SHARE:
                    cands.append((ll, g, per))
            if not cands:
                break
            ll, g, per = min(cands, key=lambda c: c[0])
            sel.append(g)
            cur, cur_per = ll, per
        log.info("C=%s grupos=%s ll=%.4f", C, sel, cur)
        if cur < best[1]:
            best = (sel, cur, C)
    return best[0], best[1], best[2]


def tune(settings: Settings, matches: pd.DataFrame, with_weather: bool = False) -> dict:
    first_oos = int(settings.get("backtest", "first_oos_season"))
    first_val = int(settings.get("backtest", "first_val_season"))
    first_test = int(settings.get("backtest", "first_test_season"))
    hist = matches[matches.season < first_test]
    val_seasons = list(range(first_val, first_test))

    results = []
    for xi, ridge, win in itertools.product(DC_GRID["xi"], DC_GRID["ridge"], DC_GRID["window_days"]):
        s2 = load_settings(root=settings.root, overrides={"dixon_coles": {"xi": xi, "ridge": ridge, "window_days": win}},
                           use_tuned=False)
        d = pipeline.walk_forward_dc(s2, hist, first_val)
        ll = _ll(hist.loc[d.index].join(d))
        results.append((ll, xi, ridge, win))
        log.info("DC xi=%s ridge=%s ventana=%s -> %.4f", xi, ridge, win, ll)
    ll_dc, xi, ridge, win = min(results)
    dc_best = {"xi": xi, "ridge": ridge, "window_days": win}

    s_best = load_settings(root=settings.root, overrides={"dixon_coles": dc_best}, use_tuned=False)
    tab = pipeline.build_table(s_best, hist, first_oos, with_weather)
    tab = tab[tab.result.notna()]
    groups, ll_st, C = forward_select(tab, val_seasons)
    out = {
        "dixon_coles": dc_best,
        "stacker": {"C": C, "groups": groups},
        "validation": {"seasons": val_seasons, "dc_log_loss": ll_dc, "stacker_log_loss": ll_st},
    }
    if tab[["mkt_ph", "mkt_pd", "mkt_pa"]].notna().all(axis=1).sum() > 1000:
        g_m, ll_m, C_m = forward_select(tab, val_seasons, use_market=True)
        out["stacker_market"] = {"C": C_m, "groups": g_m}
        out["validation"]["stacker_market_log_loss"] = ll_m
    (settings.state_dir / "tuned.json").write_text(json.dumps(out, indent=2))
    return out
