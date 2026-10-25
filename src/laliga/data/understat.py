"""xG por partido y por jugador desde Understat (gratis, LaLiga desde 2014/15).

Understat sirve los datos por una API interna: primero se visita la portada para
obtener cookies y después /getLeagueData/La_liga/{temporada} con la cabecera
X-Requested-With (mismo método que la librería soccerdata).
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from ..config import Settings
from ..teams import canonical
from . import http
from .results import current_season

log = logging.getLogger(__name__)

BASE = "https://understat.com"
FIRST_SEASON = 2014
HEADERS = {"X-Requested-With": "XMLHttpRequest", "Referer": BASE + "/league/La_liga"}
MATCH_COLS = ["season", "home", "away", "h_xg", "a_xg"]
PLAYER_COLS = ["season", "team", "player", "position", "games", "minutes", "xg", "xa", "goals", "assists", "shots"]


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def parse_league(js: dict, season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    matches = []
    for m in js.get("dates", []) or []:
        if str(m.get("isResult")).lower() not in ("true", "1"):
            continue
        try:
            h = canonical(m["h"]["title"])
            a = canonical(m["a"]["title"])
        except (KeyError, TypeError) as e:
            log.warning("Understat: %s", e)
            continue
        xg = m.get("xG") or {}
        matches.append({"season": season, "home": h, "away": a, "h_xg": _f(xg.get("h")), "a_xg": _f(xg.get("a"))})
    players = []
    for p in js.get("players", []) or []:
        team = str(p.get("team_title", "")).split(",")[-1].strip()  # si cambió de equipo, el último
        try:
            team = canonical(team)
        except KeyError:
            continue
        players.append({"season": season, "team": team, "player": p.get("player_name", ""),
                        "position": p.get("position", ""), "games": _f(p.get("games")),
                        "minutes": _f(p.get("time")), "xg": _f(p.get("xG")), "xa": _f(p.get("xA")),
                        "goals": _f(p.get("goals")), "assists": _f(p.get("assists")), "shots": _f(p.get("shots"))})
    return (pd.DataFrame(matches, columns=MATCH_COLS), pd.DataFrame(players, columns=PLAYER_COLS))


def fetch_season(settings: Settings, season: int, online: bool = True) -> dict | None:
    cache = settings.raw_dir / "understat" / f"La_liga_{season}.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    stale = season >= current_season()
    if cache.exists() and (not stale or not online):
        return json.loads(cache.read_text(encoding="utf-8"))
    if not online:
        return None
    try:
        http.get(BASE, timeout=20, retries=1)  # cookies de sesión
        js = http.get(f"{BASE}/getLeagueData/La_liga/{season}", headers=HEADERS, timeout=30, retries=1).json()
        cache.write_text(json.dumps(js), encoding="utf-8")
        return js
    except Exception as e:
        log.warning("Understat %s no disponible: %s", season, e)
        return json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else None


def load_understat(settings: Settings, online: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    ms, ps = [], []
    for y in range(FIRST_SEASON, current_season() + 1):
        js = fetch_season(settings, y, online)
        if js:
            m, p = parse_league(js, y)
            ms.append(m)
            ps.append(p)
    m = pd.concat(ms, ignore_index=True) if ms else pd.DataFrame(columns=MATCH_COLS)
    p = pd.concat(ps, ignore_index=True) if ps else pd.DataFrame(columns=PLAYER_COLS)
    return m.drop_duplicates(["season", "home", "away"]), p


def attach_xg(matches: pd.DataFrame, xg: pd.DataFrame) -> pd.DataFrame:
    out = matches.drop(columns=[c for c in ("h_xg", "a_xg") if c in matches.columns])
    if xg is None or xg.empty:
        return out.assign(h_xg=np.nan, a_xg=np.nan)
    return out.merge(xg, on=["season", "home", "away"], how="left")
