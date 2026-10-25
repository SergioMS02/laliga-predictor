"""Elo europeo (y de la Liga F) para los partidos de la quiniela que no son de LaLiga ni de Segunda.

Fuentes gratuitas:
  · football-data.co.uk: ligas principales y segundas divisiones de Europa (mmz4281/{temp}/{div}.csv)
    y ligas "extra" (new/{PAIS}.csv: Austria, Suiza, Dinamarca, Noruega, Suecia, Polonia...).
  · ESPN: Champions, Europa League y Conference (los cruces entre ligas que conectan las escalas)
    y la Liga F (fútbol femenino).

Se juega un Elo clásico partido a partido (ventaja de campo, factor por diferencia de goles,
regresión a la media de su liga en verano) con valores iniciales según el nivel de cada liga.
La escala se desplaza para que la media de LaLiga sea 1500, la misma que usa la quiniela.
Resultado: data/processed/euro_elo.csv (team, rating, league, n).
"""
from __future__ import annotations

import io
import json
import logging
import math
import re
import unicodedata
from datetime import date, timedelta

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from .config import Settings
from .data import http
from .data.results import current_season, season_code

log = logging.getLogger(__name__)

FD_URL = "https://www.football-data.co.uk/mmz4281/{code}/{div}.csv"
FD_NEW = "https://www.football-data.co.uk/new/{c}.csv"
ESPN_SB = "https://site.api.espn.com/apis/site/v2/sports/soccer/{lg}/scoreboard"
# nivel inicial (escala Elo) de cada liga
DIVS = {"SP1": 1500, "E0": 1530, "D1": 1490, "I1": 1500, "F1": 1460, "P1": 1420, "N1": 1420, "B1": 1390,
        "T1": 1380, "SC0": 1330, "G1": 1340, "SP2": 1340, "E1": 1360, "D2": 1330, "I2": 1310, "F2": 1290}
NEW = {"AUT": 1340, "SWZ": 1340, "DNK": 1340, "NOR": 1300, "SWE": 1290, "POL": 1300, "ROU": 1270,
       "CZE": 1340, "FIN": 1200, "IRL": 1180}
CUPS = {"uefa.champions": "UCL", "uefa.europa": "UEL", "uefa.europa.conf": "UECL"}
WOMEN = "esp.w.1"
K, HOME = 20.0, 65.0
COLS = ["date", "home", "away", "hg", "ag", "league"]

# nombres en español (boleto) o de ESPN -> nombre de football-data
ALIASES = {
    "napoles": "Napoli", "inter milan": "Inter", "inter de milan": "Inter", "bayern m": "Bayern Munich", "bayern munich": "Bayern Munich", "bayern": "Bayern Munich",
    "inter de milan": "Inter", "inter": "Inter", "internazionale": "Inter", "milan": "Milan", "ac milan": "Milan",
    "manch united": "Man United", "manchester united": "Man United", "manchester city": "Man City",
    "manch city": "Man City", "man city": "Man City", "brujas": "Club Brugge", "club brugge": "Club Brugge",
    "oporto": "Porto", "porto": "Porto", "fc porto": "Porto", "psv": "PSV Eindhoven", "psv eindhoven": "PSV Eindhoven",
    "aek atenas": "AEK", "aek athens": "AEK", "olympiacos": "Olympiakos", "olympiakos": "Olympiakos",
    "paok": "PAOK", "psg": "Paris SG", "paris saint germain": "Paris SG", "paris sg": "Paris SG",
    "juventus": "Juventus", "atalanta": "Atalanta", "roma": "Roma", "lazio": "Lazio", "fiorentina": "Fiorentina",
    "borussia dortmund": "Dortmund", "dortmund": "Dortmund", "b dortmund": "Dortmund", "leverkusen": "Leverkusen",
    "bayer leverkusen": "Leverkusen", "rb leipzig": "RB Leipzig", "leipzig": "RB Leipzig",
    "eintracht frankfurt": "Ein Frankfurt", "e frankfurt": "Ein Frankfurt", "stuttgart": "Stuttgart",
    "vfb stuttgart": "Stuttgart", "friburgo": "Freiburg", "sc freiburg": "Freiburg", "wolfsburgo": "Wolfsburg",
    "benfica": "Benfica", "sporting cp": "Sp Lisbon", "sporting de lisboa": "Sp Lisbon", "sporting lisboa": "Sp Lisbon",
    "braga": "Sp Braga", "sc braga": "Sp Braga", "ajax": "Ajax", "feyenoord": "Feyenoord", "az alkmaar": "AZ Alkmaar",
    "celtic": "Celtic", "rangers": "Rangers", "galatasaray": "Galatasaray", "fenerbahce": "Fenerbahce",
    "besiktas": "Besiktas", "arsenal": "Arsenal", "liverpool": "Liverpool", "chelsea": "Chelsea",
    "tottenham": "Tottenham", "newcastle": "Newcastle", "aston villa": "Aston Villa", "lens": "Lens", "lille": "Lille",
    "monaco": "Monaco", "marsella": "Marseille", "marseille": "Marseille", "lyon": "Lyon", "olympique lyon": "Lyon",
    "niza": "Nice", "nice": "Nice", "como": "Como", "bolonia": "Bologna", "bologna": "Bologna",
    "bodoglimt": "Bodo/Glimt", "bodo glimt": "Bodo/Glimt", "bodo/glimt": "Bodo/Glimt", "lillestroem": "Lillestrom",
    "lillestrom": "Lillestrom", "salzburgo": "Salzburg", "rb salzburg": "Salzburg", "red bull salzburg": "Salzburg",
    "basilea": "Basel", "fc basel": "Basel", "young boys": "Young Boys", "lugano": "Lugano", "copenhague": "FC Copenhagen",
    "fc copenhagen": "FC Copenhagen", "midtjylland": "Midtjylland", "shakhtar": "Shakhtar Donetsk",
    "shakhtar donetsk": "Shakhtar Donetsk", "dinamo zagreb": "Dinamo Zagreb", "estrella roja": "Crvena Zvezda",
    "slavia praga": "Slavia Prague", "sparta praga": "Sparta Prague", "malmo": "Malmo FF", "malmo ff": "Malmo FF",
    "anderlecht": "Anderlecht", "gent": "Gent", "union saint gilloise": "St. Gilloise", "royale union sg": "St. Gilloise",
    "stade rennais": "Rennes", "rennes": "Rennes", "f c kobenhavn": "FC Copenhagen", "kobenhavn": "FC Copenhagen",
    "strasbourg": "Strasbourg", "estrasburgo": "Strasbourg", "sevilla": "Sevilla", "athletic club": "Ath Bilbao",
    "atletico madrid": "Ath Madrid", "real sociedad": "Sociedad", "celta vigo": "Celta", "real betis": "Betis",
    "rayo vallecano": "Vallecano", "west ham": "West Ham", "brighton": "Brighton", "nottingham forest": "Nott'm Forest", "nottingham": "Nott'm Forest",
}

STOP = {"fc", "cf", "sc", "ac", "afc", "fk", "sk", "if", "bk", "cd", "ud", "club", "de", "the", "1", "sv"}


def _plain(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9/ ]+", " ", s)
    return " ".join(t for t in s.split() if t not in STOP)


def _cache(settings: Settings):
    d = settings.raw_dir / "euro"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Descarga
# ---------------------------------------------------------------------------
def _fd_frame(text: str, league: str) -> pd.DataFrame:
    raw = pd.read_csv(io.StringIO(text), on_bad_lines="skip")
    if "HomeTeam" in raw:
        h, a, hg, ag = "HomeTeam", "AwayTeam", "FTHG", "FTAG"
    elif "Home" in raw:
        h, a, hg, ag = "Home", "Away", "HG", "AG"
    else:
        return pd.DataFrame(columns=COLS)
    d = raw["Date"].astype(str)
    dt = pd.to_datetime(d, format="%d/%m/%Y", errors="coerce").fillna(pd.to_datetime(d, format="%d/%m/%y", errors="coerce"))
    out = pd.DataFrame({"date": dt, "home": raw[h], "away": raw[a], "hg": pd.to_numeric(raw[hg], errors="coerce"),
                        "ag": pd.to_numeric(raw[ag], errors="coerce"), "league": league})
    return out.dropna(subset=["date", "hg", "ag"])


def _get(url: str, cache_file, online: bool, max_age_days: float | None) -> str | None:
    if cache_file.exists():
        age = (pd.Timestamp.now() - pd.Timestamp(cache_file.stat().st_mtime, unit="s")).days
        if not online or max_age_days is None or age <= max_age_days:
            return cache_file.read_text(encoding="utf-8", errors="replace")
    if not online:
        return None
    try:
        text = http.get(url, timeout=30, retries=1).content.decode("utf-8-sig", errors="replace")
        cache_file.write_text(text, encoding="utf-8")
        return text
    except Exception as e:
        log.warning("Elo europeo: no se pudo descargar %s: %s", url, e)
        return cache_file.read_text(encoding="utf-8", errors="replace") if cache_file.exists() else None


def _espn_matches(lg: str, start: date, end: date, cache_file, online: bool, weekdays: tuple[int, ...]) -> pd.DataFrame:
    """Partidos terminados de una competición de ESPN, día a día (ESPN ya no acepta rangos de fechas).

    Solo se consultan los días de la semana en que se juega y cada día se pide una sola vez
    (se guarda la lista de días consultados); los tres últimos días se vuelven a pedir."""
    seen_file = cache_file.with_suffix(".days.json")
    df = pd.read_csv(cache_file, parse_dates=["date"]) if cache_file.exists() else pd.DataFrame(columns=COLS)
    seen = set(json.loads(seen_file.read_text())) if seen_file.exists() else set()
    if not online:
        return df
    today = date.today()
    rows, day, n_req = [], start, 0
    while day <= min(end, today):
        key = day.isoformat()
        recent = (today - day).days <= 3
        if day.weekday() in weekdays and (key not in seen or recent):
            try:
                js = http.get(ESPN_SB.format(lg=lg), params={"dates": f"{day:%Y%m%d}"}, timeout=20, retries=1).json()
                n_req += 1
                for ev in js.get("events", []):
                    comp = ev["competitions"][0]
                    if not comp.get("status", {}).get("type", {}).get("completed"):
                        continue
                    side = {c["homeAway"]: c for c in comp["competitors"]}
                    rows.append({"date": pd.Timestamp(day), "home": side["home"]["team"]["displayName"],
                                 "away": side["away"]["team"]["displayName"],
                                 "hg": float(side["home"].get("score") or 0), "ag": float(side["away"].get("score") or 0),
                                 "league": CUPS.get(lg, "W" if lg == WOMEN else lg)})
                if not recent:
                    seen.add(key)
            except Exception as e:
                log.warning("Elo europeo: ESPN %s %s: %s", lg, day, e)
                if isinstance(e, http.HostUnavailable):
                    break
        day += timedelta(days=1)
    if rows:
        new = pd.DataFrame(rows, columns=COLS)
        df = pd.concat([df, new], ignore_index=True).drop_duplicates(["date", "home", "away"], keep="last")
    df.to_csv(cache_file, index=False)
    seen_file.write_text(json.dumps(sorted(seen)))
    if n_req:
        log.info("Elo europeo: %s, %d consultas a ESPN", lg, n_req)
    return df


def load_matches(settings: Settings, online: bool = True, seasons: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(partidos de clubes masculinos, partidos de la Liga F)."""
    cdir = _cache(settings)
    cur = current_season()
    frames = []
    for y in range(cur - seasons + 1, cur + 1):
        code = season_code(y)
        for div in DIVS:
            if div in ("SP1", "SP2"):  # ya están en las cachés del proyecto
                p = settings.raw_dir / f"{div}_{code}.csv"
                if p.exists():
                    d = pd.read_csv(p, parse_dates=["date"])[["date", "home", "away", "hg", "ag"]].dropna()
                    frames.append(d.assign(league=div))
                continue
            text = _get(FD_URL.format(code=code, div=div), cdir / f"{div}_{code}.csv", online,
                        max_age_days=1 if y == cur else None)
            if text:
                frames.append(_fd_frame(text, div))
    for c in NEW:
        text = _get(FD_NEW.format(c=c), cdir / f"new_{c}.csv", online, max_age_days=1)
        if text:
            f = _fd_frame(text, c)
            frames.append(f[f.date >= pd.Timestamp(f"{cur - seasons + 1}-07-01")])
    for lg in CUPS:  # martes, miércoles y jueves; dos temporadas bastan para enlazar las ligas
        for y in range(cur - 1, cur + 1):
            frames.append(_espn_matches(lg, date(y, 7, 1), date(y + 1, 6, 30), cdir / f"{CUPS[lg]}_{y}.csv",
                                        online, weekdays=(1, 2, 3)))
    men = pd.concat([f for f in frames if len(f)], ignore_index=True) if frames else pd.DataFrame(columns=COLS)
    women = pd.concat([_espn_matches(WOMEN, date(y, 8, 15), date(y + 1, 6, 15), cdir / f"W_{y}.csv",
                                     online, weekdays=(4, 5, 6, 0))
                       for y in range(cur - 1, cur + 1)], ignore_index=True)
    return men, women


# ---------------------------------------------------------------------------
# Unificación de nombres (las copas vienen de ESPN con nombres en inglés)
# ---------------------------------------------------------------------------
def unify_cup_names(men: pd.DataFrame) -> pd.DataFrame:
    league_teams = sorted(set(men.loc[~men.league.isin(CUPS.values()), "home"]) |
                          set(men.loc[~men.league.isin(CUPS.values()), "away"]))
    plain = {t: _plain(t) for t in league_teams}
    from .teams import canonical
    cache: dict[str, str] = {}

    def fix(name: str) -> str:
        if name in cache:
            return cache[name]
        p = _plain(name)
        out = ALIASES.get(p)
        if out is None:
            c = canonical(name, strict=False, fuzzy=False)
            if c in plain:
                out = c
        if out is None:
            best = process.extractOne(p, plain, scorer=fuzz.token_set_ratio)
            out = best[2] if best and best[1] >= 90 else name
        cache[name] = out
        return out
    m = men.copy()
    cups = m.league.isin(CUPS.values())
    m.loc[cups, "home"] = m.loc[cups, "home"].map(fix)
    m.loc[cups, "away"] = m.loc[cups, "away"].map(fix)
    return m


# ---------------------------------------------------------------------------
# Elo
# ---------------------------------------------------------------------------
def _gd_mult(gd: float) -> float:
    gd = abs(gd)
    return 1.0 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8


def run_elo(m: pd.DataFrame, init: dict[str, float], k: float = K, home: float = HOME,
            regress: float = 0.25) -> pd.DataFrame:
    """Elo partido a partido. init: rating inicial por liga. Devuelve team, rating, league, n."""
    m = m.sort_values("date")
    rating: dict[str, float] = {}
    league_of: dict[str, str] = {}
    n: dict[str, int] = {}
    last_season = None
    for r in m.itertuples(index=False):
        season = r.date.year if r.date.month >= 7 else r.date.year - 1
        if last_season is not None and season != last_season:  # verano: regresión a la media de su liga
            by = {}
            for t, v in rating.items():
                by.setdefault(league_of.get(t), []).append(v)
            means = {lg: float(np.mean(v)) for lg, v in by.items()}
            for t in rating:
                rating[t] = rating[t] * (1 - regress) + means[league_of.get(t)] * regress
        last_season = season
        is_cup = r.league in CUPS.values()
        for t in (r.home, r.away):
            if t not in rating:
                rating[t] = init.get(r.league, 1350.0) if not is_cup else 1350.0
            if not is_cup:
                league_of[t] = r.league
            n[t] = n.get(t, 0) + 1
        dr = rating[r.home] + home - rating[r.away]
        e = 1 / (1 + 10 ** (-dr / 400))
        s = 1.0 if r.hg > r.ag else 0.5 if r.hg == r.ag else 0.0
        delta = k * (1.5 if is_cup else 1.0) * _gd_mult(r.hg - r.ag) * (s - e)
        rating[r.home] += delta
        rating[r.away] -= delta
    out = pd.DataFrame({"team": list(rating), "rating": list(rating.values()),
                        "league": [league_of.get(t, "copa") for t in rating], "n": [n[t] for t in rating]})
    return out


def build(settings: Settings, online: bool = True) -> pd.DataFrame:
    men, women = load_matches(settings, online=online)
    out = []
    if len(men):
        men = unify_cup_names(men)
        r = run_elo(men, {**DIVS, **NEW})
        sp1 = r[r.league == "SP1"].rating
        shift = 1500 - float(sp1.mean()) if len(sp1) else 0.0
        r["rating"] = (r.rating + shift).round(1)
        out.append(r)
    if len(women):
        w = run_elo(women.assign(league="W"), {"W": 1400.0})
        w["rating"] = (w.rating - w.rating.mean() + 1400).round(1)
        w["team"] = w.team.map(lambda t: f"{t} (F)")
        out.append(w)
    res = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["team", "rating", "league", "n"])
    res.to_csv(settings.processed_dir / "euro_elo.csv", index=False)
    log.info("Elo europeo: %d equipos", len(res))
    return res


def load(settings: Settings) -> pd.DataFrame:
    p = settings.processed_dir / "euro_elo.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame(columns=["team", "rating", "league", "n"])


class Lookup:
    """Busca el rating de un nombre del boleto (en español) entre los equipos con Elo."""

    def __init__(self, table: pd.DataFrame):
        self.men = table[~table.team.str.endswith(" (F)")]
        self.women = table[table.team.str.endswith(" (F)")]
        self.men_plain = {t: _plain(t) for t in self.men.team}
        self.women_plain = {t: _plain(t[:-4]) for t in self.women.team}
        self.rating = dict(zip(table.team, table.rating))
        self.league = dict(zip(table.team, table.league))

    def find(self, name: str, women: bool = False) -> tuple[str, float, str] | None:
        p = _plain(re.sub(r"\b(femenino|fem|women|cf femenino)\b", "", name, flags=re.I))
        if women:
            pool = self.women_plain
            best = process.extractOne(p, pool, scorer=fuzz.token_set_ratio) if pool else None
            if best and best[1] >= 85:
                return best[2], self.rating[best[2]], "W"
            return None
        alias = ALIASES.get(p)
        if alias and alias in self.rating:
            return alias, self.rating[alias], self.league[alias]
        best = process.extractOne(p, self.men_plain, scorer=fuzz.token_sort_ratio) if self.men_plain else None
        if best and best[1] >= 88:
            return best[2], self.rating[best[2]], self.league[best[2]]
        return None
