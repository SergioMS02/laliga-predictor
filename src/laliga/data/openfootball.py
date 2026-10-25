"""Parser del formato Football.TXT de openfootball (calendario + resultados)."""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

from ..config import Settings
from ..teams import canonical
from . import http
from .results import COLUMNS, current_season

log = logging.getLogger(__name__)

_MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}
_DATE_RE = re.compile(r"^\s*(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\w*\s+([A-Z][a-z]{2})\w*\s+(\d{1,2})(?:\s+(\d{4}))?\s*(.*)$")
_TIME_RE = re.compile(r"^\s*(\d{1,2})[:.](\d{2})\s+(.*)$")
_SCORE_RE = re.compile(r"(\d+)\s*-\s*(\d+)(?:\s*\(\s*\d+\s*-\s*\d+\s*\))?")
_ROUND_RE = re.compile(r"Matchday\s+(\d+)|Jornada\s+(\d+)", re.I)


def season_dir(start_year: int) -> str:
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def _parse_match(body: str):
    body = re.sub(r"\[[^\]]*\]", "", body).strip()
    body = re.sub(r"@.*$", "", body).strip()  # sede opcional
    if not body:
        return None
    # Formato A: "Local v Visitante   2-1 (1-0)"
    m = re.match(r"^(.*?)\s+v(?:s)?\.?\s+(.*)$", body)
    if m:
        home = m.group(1).strip()
        rest = re.split(r"\s{2,}", m.group(2).strip(), maxsplit=1)
        away = rest[0].strip()
        sm = _SCORE_RE.search(rest[1]) if len(rest) > 1 else None
        if sm is None:
            sm2 = re.match(r"^(.*?)\s+(\d+\s*-\s*\d+.*)$", away)
            if sm2:
                away, sm = sm2.group(1).strip(), _SCORE_RE.search(sm2.group(2))
        return home, away, (int(sm.group(1)), int(sm.group(2))) if sm else None
    # Formato B: "Local   2-1 (1-0)   Visitante"
    sm = _SCORE_RE.search(body)
    if sm:
        home = body[: sm.start()].strip()
        away = body[sm.end():].strip()
        if home and away:
            return home, away, (int(sm.group(1)), int(sm.group(2)))
    parts = re.split(r"\s{2,}", body)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip(), None
    return None


def parse_football_txt(text: str, start_year: int) -> pd.DataFrame:
    rows = []
    cur_date = None
    rnd = None
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith(("#", "=")):
            continue
        rm = _ROUND_RE.search(line)
        if rm and ("▪" in line or line.lstrip().startswith(("»", "Matchday", "Jornada"))):
            rnd = int(rm.group(1) or rm.group(2))
            continue
        dm = _DATE_RE.match(line)
        if dm:
            mon, day, year, tail = dm.groups()
            month = _MONTHS[mon]
            y = int(year) if year else (start_year if month >= 7 else start_year + 1)
            cur_date = pd.Timestamp(year=y, month=month, day=int(day))
            line = tail
            if not line.strip():
                continue
        if cur_date is None:
            continue
        hh = mm = None
        tm = _TIME_RE.match(line)
        if tm:
            hh, mm, line = int(tm.group(1)), int(tm.group(2)), tm.group(3)
        parsed = _parse_match(line)
        if not parsed:
            continue
        home, away, score = parsed
        try:
            h, a = canonical(home), canonical(away)
        except KeyError as e:
            log.warning("openfootball: %s", e)
            continue
        ko = pd.NaT
        if hh is not None:
            ko = (cur_date + pd.Timedelta(hours=hh, minutes=mm)).tz_localize(
                "Europe/Madrid", ambiguous="NaT", nonexistent="NaT").tz_convert("UTC")
        rows.append({"date": cur_date, "kickoff": ko, "home": h, "away": a, "round": rnd,
                     "hg": score[0] if score else np.nan, "ag": score[1] if score else np.nan})
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    df["season"] = start_year
    df["result"] = np.select([df.hg > df.ag, df.hg == df.ag, df.hg < df.ag], ["H", "D", "A"], default=None)
    df.loc[df.hg.isna(), "result"] = None
    df["source"] = "openfootball"
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    df["kickoff"] = pd.to_datetime(df["kickoff"], utc=True)
    return df[COLUMNS]


def fetch_openfootball(settings: Settings, start_year: int | None = None) -> pd.DataFrame:
    start_year = start_year or current_season()
    url = settings.get("data", "openfootball_url").format(season_dir=season_dir(start_year))
    cache = settings.raw_dir / f"openfootball_{start_year}.txt"
    try:
        text = http.get(url, timeout=20).text
        cache.write_text(text, encoding="utf-8")
    except Exception as e:
        log.warning("openfootball no disponible: %s", e)
        if not cache.exists():
            return pd.DataFrame(columns=COLUMNS)
        text = cache.read_text(encoding="utf-8")
    return parse_football_txt(text, start_year)
