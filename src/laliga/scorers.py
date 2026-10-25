"""Probables goleadores: probabilidad de que cada titular marque al menos un gol.

Para cada jugador del once (oficial o probable) se estima qué parte de los goles de su
equipo suele producir, con su xG por 90 minutos de Understat (temporada actual y anterior,
ponderadas por minutos) respecto al xG por partido del equipo. Los goles esperados del
equipo en este partido (Dixon-Coles, ya ajustados por bajas) se reparten con esa proporción
y P(marca) = 1 − exp(−goles esperados del jugador).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

MIN_MINUTES = 270          # por debajo, el xG por 90 es demasiado ruidoso
STARTER_MINUTES = {"F": 75, "M": 78, "D": 85, "G": 90}
SHARE_CAP = 0.95           # nunca se reparte más del 95 % de los goles (suplentes y goles en propia)


def player_rates(players: pd.DataFrame | None, team: str, season: int) -> pd.DataFrame:
    """xG por 90 de cada jugador del equipo y xG por partido del equipo (temporada actual + anterior)."""
    cols = ["player", "xg90", "goals", "minutes"]
    if players is None or players.empty:
        return pd.DataFrame(columns=cols)
    p = players[(players.team == team) & players.season.isin([season, season - 1])].copy()
    if p.empty:
        return pd.DataFrame(columns=cols)
    # la temporada actual pesa el doble que la anterior
    p["w"] = np.where(p.season == season, 2.0, 1.0)
    agg = p.assign(wxg=p.xg * p.w, wmin=p.minutes * p.w, wgoals=p.goals * p.w).groupby("player").agg(
        wxg=("wxg", "sum"), wmin=("wmin", "sum"), goals=("goals", "sum"), minutes=("minutes", "sum"))
    agg = agg[agg.minutes >= MIN_MINUTES]
    agg["xg90"] = agg.wxg / agg.wmin * 90
    return agg.reset_index()[cols]


def team_xg_per_match(players: pd.DataFrame | None, team: str, season: int) -> float | None:
    if players is None or players.empty:
        return None
    vals = []
    for s, w in ((season, 2.0), (season - 1, 1.0)):
        p = players[(players.team == team) & (players.season == s)]
        games = p.games.max() if len(p) and "games" in p else 0
        if games and games >= 3:
            vals.append((p.xg.sum() / games, w * games))
    if not vals:
        return None
    return float(sum(v * w for v, w in vals) / sum(w for _, w in vals))


def scorer_probs(xi: list[dict], team_goals: float, players: pd.DataFrame | None, team: str, season: int,
                 top: int = 5) -> list[dict]:
    """Lista ordenada [{name, p, xg90}] para los titulares con datos suficientes."""
    if not xi or not np.isfinite(team_goals) or team_goals <= 0:
        return []
    rates = player_rates(players, team, season)
    txg = team_xg_per_match(players, team, season)
    if rates.empty or not txg:
        return []
    names = list(rates.player)
    rows = []
    for pl in xi:
        if pl.get("absent") == "out":
            continue
        best = process.extractOne(str(pl.get("name", "")), names, scorer=fuzz.token_set_ratio)
        if not best or best[1] < 85:
            continue
        r = rates.iloc[best[2]]
        mins = STARTER_MINUTES.get(pl.get("role") or "M", 78)
        share = r.xg90 * mins / 90 / txg
        rows.append({"name": pl["name"], "share": float(share), "xg90": round(float(r.xg90), 2)})
    if not rows:
        return []
    total = sum(r["share"] for r in rows)
    k = SHARE_CAP / total if total > SHARE_CAP else 1.0
    out = []
    for r in rows:
        lam = team_goals * r["share"] * k
        out.append({"name": r["name"], "p": round(1 - math.exp(-lam), 3), "xg90": r["xg90"]})
    out.sort(key=lambda d: -d["p"])
    return out[:top]
