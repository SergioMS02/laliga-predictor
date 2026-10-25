"""Cliente de la API pública (no documentada) de ESPN: calendario, resultados,
alineaciones confirmadas y árbitro. Gratis y sin clave.

La API no es oficial: el parser es defensivo y cualquier fallo degrada a "sin datos".
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from ..config import Settings
from ..teams import canonical
from . import http
from .results import COLUMNS, season_of

log = logging.getLogger(__name__)

BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer/{league}"


@dataclass
class Player:
    player_id: str
    name: str
    position: str
    starter: bool
    jersey: str = ""
    formation_place: int | None = None


@dataclass
class MatchDetail:
    event_id: str
    referee: str | None = None
    venue: str | None = None
    lineups: dict[str, list[Player]] = field(default_factory=dict)  # 'home'/'away'
    formations: dict[str, str] = field(default_factory=dict)          # p. ej. {'home': '4-3-3'}

    @property
    def lineup_confirmed(self) -> bool:
        return all(sum(p.starter for p in self.lineups.get(side, [])) >= 11 for side in ("home", "away"))


def _g(d, *path, default=None):
    for p in path:
        if isinstance(d, dict):
            d = d.get(p)
        elif isinstance(d, list) and isinstance(p, int) and -len(d) <= p < len(d):
            d = d[p]
        else:
            return default
        if d is None:
            return default
    return d


def parse_scoreboard(js: dict, only_known: bool = False) -> pd.DataFrame:
    """only_known=True: competiciones mixtas (Europa, Copa); los equipos ajenos a LaLiga
    se conservan con su nombre original en lugar de descartar el partido."""
    rows = []
    for ev in js.get("events", []) or []:
        comp = _g(ev, "competitions", 0, default={})
        comps = comp.get("competitors", []) or []
        teams = {c.get("homeAway"): c for c in comps}
        if not {"home", "away"} <= set(teams) and len(comps) == 2:  # sin homeAway: 0 = local
            teams = {"home": comps[0], "away": comps[1]}
        if "home" not in teams or "away" not in teams:
            continue
        hn = _g(teams["home"], "team", "displayName", default="") or _g(teams["home"], "team", "name", default="")
        an = _g(teams["away"], "team", "displayName", default="") or _g(teams["away"], "team", "name", default="")
        try:
            if only_known:
                home, away = canonical(hn, strict=False, fuzzy=False), canonical(an, strict=False, fuzzy=False)
            else:
                home, away = canonical(hn), canonical(an)
        except KeyError as e:
            log.warning("ESPN: %s", e)
            continue
        state = _g(comp, "status", "type", "state") or _g(ev, "status", "type", "state", default="pre")
        completed = bool(_g(comp, "status", "type", "completed") or _g(ev, "status", "type", "completed", default=False))

        def score(c):
            s = c.get("score")
            if isinstance(s, dict):
                s = s.get("value", s.get("displayValue"))
            try:
                return float(s)
            except (TypeError, ValueError):
                return np.nan
        rows.append({
            "event_id": str(ev.get("id")),
            "kickoff": pd.to_datetime(ev.get("date") or comp.get("date"), utc=True, errors="coerce"),
            "home": home, "away": away,
            "hg": score(teams["home"]) if completed else np.nan,
            "ag": score(teams["away"]) if completed else np.nan,
            "state": state, "completed": completed,
            "venue": _g(comp, "venue", "fullName"),
        })
    return pd.DataFrame(rows)


def parse_summary(js: dict, event_id: str) -> MatchDetail:
    det = MatchDetail(event_id=str(event_id))
    officials = _g(js, "gameInfo", "officials", default=[]) or []
    for o in officials:
        pos = (_g(o, "position", "name", default="") or _g(o, "position", "displayName", default="") or "").lower()
        if "referee" in pos and "assistant" not in pos and "video" not in pos and "var" not in pos.split():
            det.referee = o.get("displayName") or o.get("fullName")
            break
    if det.referee is None and officials:
        o = sorted(officials, key=lambda x: x.get("order", 99))[0]
        det.referee = o.get("displayName") or o.get("fullName")
    det.venue = _g(js, "gameInfo", "venue", "fullName")
    rosters = _g(js, "rosters", default=[]) or []
    for i, r in enumerate(rosters):
        side = r.get("homeAway")
        if side not in ("home", "away"):
            if len(rosters) != 2:
                continue
            side = "home" if i == 0 else "away"  # mismo convenio que el resto de la API
        players = []
        for p in r.get("roster", []) or []:
            ath = p.get("athlete", {}) or {}
            try:
                fp = int(p.get("formationPlace")) if p.get("formationPlace") not in (None, "", "0") else None
            except (TypeError, ValueError):
                fp = None
            players.append(Player(
                player_id=str(ath.get("id", "")),
                name=ath.get("displayName") or ath.get("fullName") or "",
                position=_g(p, "position", "abbreviation", default="") or _g(p, "position", "name", default="") or "",
                starter=bool(p.get("starter", False)),
                jersey=str(p.get("jersey") or ""),
                formation_place=fp,
            ))
        det.lineups[side] = players
        form = r.get("formation")
        if isinstance(form, dict):
            form = form.get("name") or form.get("summary")
        if isinstance(form, str) and re.fullmatch(r"\d(-\d){2,4}", form.strip()):
            det.formations[side] = form.strip()
    return det


def scoreboard(settings: Settings, d_from: date, d_to: date, only_known: bool = False) -> pd.DataFrame:
    league = settings.get("espn", "league", "esp.1")
    url = BASE.format(league=league) + "/scoreboard"
    frames = []
    cur = d_from
    errors = []

    def fetch(params):
        try:
            return parse_scoreboard(http.get(url, params=params).json(), only_known)
        except Exception as e:  # un día fallido no tumba todo el calendario
            errors.append(f"{params.get('dates')}: {e}")
            return pd.DataFrame()

    while cur <= d_to:  # trozos de 30 días
        end = min(cur + timedelta(days=30), d_to)
        df = fetch({"dates": f"{cur:%Y%m%d}-{end:%Y%m%d}", "limit": 500})
        if df.empty and (end - cur).days > 0:
            # si el rango no está soportado, día a día (formato documentado ?dates=YYYYMMDD)
            day = cur
            while day <= end:
                frames.append(fetch({"dates": f"{day:%Y%m%d}"}))
                day += timedelta(days=1)
        else:
            frames.append(df)
        cur = end + timedelta(days=1)
    frames = [f for f in frames if len(f)]
    if not frames and errors:
        raise RuntimeError(f"ESPN sin datos ({len(errors)} errores; primero: {errors[0]})")
    if errors:
        log.warning("ESPN: %d peticiones fallidas (p. ej. %s)", len(errors), errors[0])
    return pd.concat(frames, ignore_index=True).drop_duplicates("event_id") if frames else pd.DataFrame()


def summary(settings: Settings, event_id: str) -> MatchDetail:
    league = settings.get("espn", "league", "esp.1")
    js = http.get(BASE.format(league=league) + "/summary", params={"event": event_id}).json()
    return parse_summary(js, event_id)


def to_results(sb: pd.DataFrame) -> pd.DataFrame:
    if sb is None or sb.empty:
        return pd.DataFrame(columns=COLUMNS)
    df = sb.copy()
    df["date"] = df["kickoff"].dt.tz_convert("Europe/Madrid").dt.tz_localize(None).dt.normalize()
    df["season"] = df["date"].map(season_of)
    df["result"] = np.select([df.hg > df.ag, df.hg == df.ag, df.hg < df.ag], ["H", "D", "A"], default=None)
    df.loc[df.hg.isna(), "result"] = None
    df["source"] = "espn"
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    return df[COLUMNS + ["event_id"]]


# ---------------------------------------------------------------------------
# Almacén de alineaciones (para calcular la importancia de cada jugador)
# ---------------------------------------------------------------------------
LINEUP_COLS = ["event_id", "date", "team", "player_id", "name", "position", "starter"]


def lineup_rows(detail: MatchDetail, match_date, home: str, away: str) -> pd.DataFrame:
    rows = []
    for side, team in (("home", home), ("away", away)):
        for p in detail.lineups.get(side, []):
            rows.append({"event_id": detail.event_id, "date": pd.Timestamp(match_date).normalize(),
                         "team": team, "player_id": p.player_id, "name": p.name,
                         "position": p.position, "starter": p.starter})
    return pd.DataFrame(rows, columns=LINEUP_COLS)


def load_lineups(settings: Settings) -> pd.DataFrame:
    path = settings.processed_dir / "lineups.csv"
    if not path.exists():
        return pd.DataFrame(columns=LINEUP_COLS)
    return pd.read_csv(path, parse_dates=["date"], dtype={"event_id": str, "player_id": str})


def save_lineups(settings: Settings, new: pd.DataFrame) -> pd.DataFrame:
    old = load_lineups(settings)
    allv = pd.concat([old, new], ignore_index=True) if len(old) else new
    allv = allv.drop_duplicates(["event_id", "team", "player_id"], keep="last")
    allv.to_csv(settings.processed_dir / "lineups.csv", index=False)
    return allv


REF_COLS = ["season", "home", "away", "referee"]


def load_referees(settings: Settings) -> pd.DataFrame:
    """Árbitros conocidos: ESPN acumulado + overrides manuales (data/referee_overrides.csv)."""
    frames = []
    for name in ("referees.csv", "referee_overrides.csv"):
        p = (settings.processed_dir if name == "referees.csv" else settings.data_dir) / name
        if p.exists():
            f = pd.read_csv(p)
            if "season" not in f.columns and "date" in f.columns:
                f["season"] = pd.to_datetime(f["date"]).map(season_of)
            f["home"] = f["home"].map(canonical)
            f["away"] = f["away"].map(canonical)
            frames.append(f[REF_COLS])
    if not frames:
        return pd.DataFrame(columns=REF_COLS)
    # los overrides van al final -> prevalecen
    return pd.concat(frames, ignore_index=True).drop_duplicates(["season", "home", "away"], keep="last")


def save_referees(settings: Settings, new: pd.DataFrame) -> None:
    p = settings.processed_dir / "referees.csv"
    old = pd.read_csv(p) if p.exists() else pd.DataFrame(columns=REF_COLS)
    allv = pd.concat([old, new[REF_COLS]], ignore_index=True).dropna(subset=["referee"])
    allv.drop_duplicates(["season", "home", "away"], keep="last").to_csv(p, index=False)
