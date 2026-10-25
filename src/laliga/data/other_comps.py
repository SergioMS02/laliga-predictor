"""Partidos de los equipos de LaLiga en otras competiciones (Champions, Europa League,
Conference, Copa del Rey). Solo hacen falta fechas: sirven para medir congestión,
cansancio y rotaciones previsibles.

Histórico: openfootball (GitHub, gratis). Temporada en curso: ESPN (gratis).
"""
from __future__ import annotations

import logging
import re
from datetime import date, timedelta

import pandas as pd

from ..config import Settings
from ..teams import canonical, is_known
from . import espn, http
from .openfootball import _DATE_RE, _MONTHS, _TIME_RE, season_dir
from .results import current_season

log = logging.getLogger(__name__)

OF_EURO = "https://raw.githubusercontent.com/openfootball/champions-league/master/{sd}/{comp}.txt"
OF_CUP = "https://raw.githubusercontent.com/openfootball/espana/master/{sd}/cup.txt"
EURO_COMPS = ("cl", "el", "conf")
ESPN_LEAGUES = {"uefa.champions": "cl", "uefa.europa": "el", "uefa.europa.conf": "conf", "esp.copa_del_rey": "cup"}
COLS = ["date", "team", "comp"]


def _teams_in_line(line: str) -> list[str]:
    """Equipos de LaLiga presentes en una línea de partido (coincidencia exacta)."""
    line = re.sub(r"@.*$", "", line)
    line = re.sub(r"\(([A-Z]{3})\)", " ", line)  # (ESP), (ITA)...
    parts = re.split(r"\s{2,}|\s+v\s+|\d+\s*-\s*\d+|\b(?:pen\.|a\.e\.t\.)|\([^)]*\)", line)
    out = []
    for p in parts:
        p = re.sub(r"^v\s+|\s+v$", "", p.strip()).strip(" ,.-")
        if len(p) < 3:
            continue
        c = canonical(p, strict=False, fuzzy=False)
        if is_known(c):
            out.append(c)
    return out


def parse_comp_txt(text: str, start_year: int, comp: str) -> pd.DataFrame:
    rows = []
    cur = None
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith(("#", "=", "▪", "»", "(")):
            continue
        if "|" in line and " v " not in line:  # tabla de grupos
            continue
        dm = _DATE_RE.match(line)
        if dm:
            mon, day, year, tail = dm.groups()
            month = _MONTHS[mon]
            y = int(year) if year else (start_year if month >= 7 else start_year + 1)
            cur = pd.Timestamp(year=y, month=month, day=int(day))
            line = tail
            if not line.strip():
                continue
        if cur is None:
            continue
        tm = _TIME_RE.match(line)
        if tm:
            line = tm.group(3)
        for t in _teams_in_line(line):
            rows.append({"date": cur, "team": t, "comp": comp})
    return pd.DataFrame(rows, columns=COLS).drop_duplicates()


def _get_text(settings: Settings, url: str, cache_name: str, refresh: bool, online: bool) -> str | None:
    cache = settings.raw_dir / "other" / cache_name
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists() and not refresh:
        return cache.read_text(encoding="utf-8")
    if not online:
        return cache.read_text(encoding="utf-8") if cache.exists() else None
    try:
        r = http.get(url, timeout=20, retries=1)
        cache.write_text(r.text, encoding="utf-8")
        return r.text
    except Exception:
        if cache.exists():
            return cache.read_text(encoding="utf-8")
        cache.write_text("", encoding="utf-8")  # 404 conocido: no reintentar
        return None


def load_other_matches(settings: Settings, first_season: int, online: bool = True) -> pd.DataFrame:
    frames = []
    cur = current_season()
    for y in range(first_season, cur + 1):
        sd = season_dir(y)
        refresh = y == cur
        for comp in EURO_COMPS:
            t = _get_text(settings, OF_EURO.format(sd=sd, comp=comp), f"{comp}-{sd}.txt", refresh, online)
            if t:
                frames.append(parse_comp_txt(t, y, comp))
        t = _get_text(settings, OF_CUP.format(sd=sd), f"cup-{sd}.txt", refresh, online)
        if t:
            frames.append(parse_comp_txt(t, y, "cup"))
    if online and settings.get("espn", "enabled", True):
        save_espn_other(settings, espn_other_matches(
            settings, date(cur, 7, 1), date.today() + timedelta(days=int(settings.get("predict", "days_ahead", 8)) + 7)))
    p = settings.processed_dir / "other_matches_espn.csv"
    if p.exists():
        frames.append(pd.read_csv(p, parse_dates=["date"])[COLS])
    p = settings.data_dir / "other_matches_manual.csv"
    if p.exists():
        m = pd.read_csv(p, parse_dates=["date"])
        m["team"] = m["team"].map(canonical)
        frames.append(m[COLS])
    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        return pd.DataFrame(columns=COLS)
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    return out.drop_duplicates(["date", "team"]).sort_values("date").reset_index(drop=True)


def espn_other_matches(settings: Settings, d_from: date, d_to: date) -> pd.DataFrame:
    rows = []
    for league, comp in ESPN_LEAGUES.items():
        s2 = type(settings)(raw={**settings.raw, "espn": {**settings.raw.get("espn", {}), "league": league}},
                            root=settings.root)
        try:
            sb = espn.scoreboard(s2, d_from, d_to, only_known=True)
        except Exception as e:
            log.warning("ESPN %s no disponible: %s", league, e)
            continue
        for r in sb.itertuples():
            d = pd.Timestamp(r.kickoff).tz_convert("Europe/Madrid").tz_localize(None).normalize()
            for t in (r.home, r.away):
                if is_known(t):
                    rows.append({"date": d, "team": t, "comp": comp})
    return pd.DataFrame(rows, columns=COLS)


def save_espn_other(settings: Settings, new: pd.DataFrame) -> None:
    if new is None or new.empty:
        return
    p = settings.processed_dir / "other_matches_espn.csv"
    old = pd.read_csv(p, parse_dates=["date"]) if p.exists() else pd.DataFrame(columns=COLS)
    pd.concat([old, new], ignore_index=True).drop_duplicates(["date", "team"]).to_csv(p, index=False)
