"""Tiros de cada partido de LaLiga desde Understat: minuto, posición, xG, resultado del tiro, jugador,
asistente y situación (jugada, córner, falta, penalti).

Sirve para:
  - calibrar la probabilidad en directo con los tiros a puerta que se llevan (lo que da ESPN en vivo),
  - los mapas de tiros de la web,
  - saber quién tira los penaltis y quién da las asistencias.

Una petición por partido (/getMatchData/{id}); se guarda por temporada y solo se piden los que faltan.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import Settings
from ..teams import canonical
from . import http
from .understat import BASE, HEADERS, fetch_season

log = logging.getLogger(__name__)

COLS = ["season", "match_id", "date", "home", "away", "side", "team", "minute", "player", "assist", "situation",
        "shot_type", "result", "xg", "x", "y"]
ON_TARGET = {"Goal", "SavedShot"}


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def parse_match(js: dict, meta: dict) -> list[dict]:
    rows = []
    shots = (js or {}).get("shots") or {}
    for side in ("h", "a"):
        for s in shots.get(side) or []:
            rows.append({**meta, "side": side, "team": meta["home"] if side == "h" else meta["away"],
                         "minute": int(_f(s.get("minute")) if np.isfinite(_f(s.get("minute"))) else 0),
                         "player": s.get("player", ""), "assist": s.get("player_assisted") or "",
                         "situation": s.get("situation", ""), "shot_type": s.get("shotType", ""),
                         "result": s.get("result", ""), "xg": _f(s.get("xG")), "x": _f(s.get("X")), "y": _f(s.get("Y"))})
    return rows


def matches_of(js: dict, season: int) -> list[dict]:
    out = []
    for m in (js or {}).get("dates", []) or []:
        if str(m.get("isResult")).lower() not in ("true", "1"):
            continue
        try:
            out.append({"season": season, "match_id": str(m["id"]), "date": str(m.get("datetime", ""))[:10],
                        "home": canonical(m["h"]["title"]), "away": canonical(m["a"]["title"])})
        except (KeyError, TypeError):
            continue
    return out


def fetch_shots(settings: Settings, season: int, online: bool = True, max_new: int = 400) -> pd.DataFrame:
    cache = settings.raw_dir / "understat" / f"shots_{season}.csv"
    df = pd.read_csv(cache, dtype={"match_id": str}) if cache.exists() else pd.DataFrame(columns=COLS)
    if not online:
        return df
    js = fetch_season(settings, season, online)
    todo = [m for m in matches_of(js, season) if m["match_id"] not in set(df.match_id.astype(str))][:max_new]
    rows, done = [], 0
    for m in todo:
        try:
            r = http.get(f"{BASE}/getMatchData/{m['match_id']}", headers=HEADERS, timeout=30, retries=1).json()
            new = parse_match(r, m)
            if not new:  # partido sin tiros (raro): fila vacía para no volver a pedirlo
                new = [{**m, "side": "", "team": "", "minute": -1}]
            rows += new
            done += 1
        except Exception as e:
            log.warning("Tiros Understat %s: %s", m["match_id"], e)
            if isinstance(e, http.HostUnavailable):
                break
    if rows:
        df = pd.concat([df, pd.DataFrame(rows, columns=COLS)], ignore_index=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)
    log.info("Tiros Understat %s: %d partidos nuevos", season, done)
    return df


def load_shots(settings: Settings, seasons: list[int], online: bool = True) -> pd.DataFrame:
    frames = [fetch_shots(settings, s, online) for s in seasons]
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLS)


# ---------------------------------------------------------------------------
# Probabilidad en directo con tiros a puerta
# ---------------------------------------------------------------------------
def live_dataset(shots: pd.DataFrame, oos: pd.DataFrame, minutes=(15, 30, 45, 60, 75)) -> pd.DataFrame:
    """Para cada partido y minuto t: goles esperados previos (Dixon-Coles), tiros a puerta y xG hasta t,
    y goles que se marcaron después de t (lo que hay que predecir)."""
    s = shots[shots.minute >= 0].copy()
    s["sot"] = s.result.isin(ON_TARGET).astype(int)
    s["goal"] = (s.result == "Goal").astype(int)
    pri = {(int(a), b, c): (float(l), float(m)) for a, b, c, l, m in
           zip(oos.season, oos.home, oos.away, oos.dc_lam, oos.dc_mu)}
    rows = []
    for (season, mid, home, away), g in s.groupby(["season", "match_id", "home", "away"]):
        key = (int(season), home, away)
        if key not in pri:
            continue
        lam, mu = pri[key]
        for t in minutes:
            b, a_ = g[g.minute < t], g[g.minute >= t]
            rows.append({"season": season, "match_id": mid, "t": t, "lam": lam, "mu": mu,
                         "sot_h": int(b[b.side == "h"].sot.sum()), "sot_a": int(b[b.side == "a"].sot.sum()),
                         "xg_h": float(b[b.side == "h"].xg.sum()), "xg_a": float(b[b.side == "a"].xg.sum()),
                         "g_h_after": int(a_[a_.side == "h"].goal.sum()), "g_a_after": int(a_[a_.side == "a"].goal.sum())})
    return pd.DataFrame(rows)


def remaining_rate(lam, t, obs, gamma: float, per_goal: float):
    """Goles esperados que quedan para un equipo: previa × (90−t)/90 × exp(γ·z), con z la desviación
    estandarizada de sus tiros a puerta (u xG) respecto a lo esperado hasta el minuto t."""
    lam, t, obs = np.asarray(lam, float), np.asarray(t, float), np.asarray(obs, float)
    exp = lam * per_goal * t / 90
    z = (obs - exp) / np.sqrt(exp + 1e-9) * np.sqrt(t / 90)
    return lam * np.clip(90 - t, 0, None) / 90 * np.exp(gamma * z)


def calibrate_live(ds: pd.DataFrame) -> dict:
    """Ajusta γ por máxima verosimilitud Poisson de los goles posteriores, con tiros a puerta (lo que
    ESPN da en directo) y, como referencia, con xG."""
    from scipy.optimize import minimize_scalar
    if ds.empty:
        return {}
    base_t = (ds.lam * ds.t / 90).sum() + (ds.mu * ds.t / 90).sum()
    out = {"n": int(len(ds)), "matches": int(ds.match_id.nunique())}
    for col, key in (("sot", "tiros_a_puerta"), ("xg", "xg")):
        per = float((ds[f"{col}_h"].sum() + ds[f"{col}_a"].sum()) / max(base_t, 1e-9))

        def nll(g):
            r_h = remaining_rate(ds.lam, ds.t, ds[f"{col}_h"], g, per)
            r_a = remaining_rate(ds.mu, ds.t, ds[f"{col}_a"], g, per)
            return float((np.sum(r_h - ds.g_h_after * np.log(r_h + 1e-12)) +
                          np.sum(r_a - ds.g_a_after * np.log(r_a + 1e-12))) / len(ds) / 2)
        r = minimize_scalar(nll, bounds=(-0.5, 1.0), method="bounded")
        out[key] = {"gamma": round(float(r.x), 4), "per_goal": round(per, 3),
                    "nll_solo_marcador": round(nll(0.0), 5), "nll_con_dato": round(float(r.fun), 5)}
    out["gamma"], out["sot_per_goal"] = out["tiros_a_puerta"]["gamma"], out["tiros_a_puerta"]["per_goal"]
    return out


def run_live_calibration(settings: Settings, matches_oos: pd.DataFrame | None = None, online: bool = True) -> dict:
    import json
    from datetime import datetime, timezone
    from .results import current_season
    cur = current_season()
    shots = load_shots(settings, list(range(cur - 5, cur + 1)), online=online)
    p = settings.processed_dir / "oos_table.csv"
    oos = matches_oos if matches_oos is not None else (pd.read_csv(p) if p.exists() else None)
    if oos is None or shots.empty:
        return {}
    out = calibrate_live(live_dataset(shots, oos))
    if out:
        out["updated"] = datetime.now(timezone.utc).date().isoformat()
        (settings.state_dir / "live_calib.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out
