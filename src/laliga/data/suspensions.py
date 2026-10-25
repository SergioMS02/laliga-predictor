"""Sancionados para el próximo partido de liga.

Dos fuentes que se complementan:
  1. Tarjetas de la temporada (ESPN, día a día con caché): roja → se pierde el siguiente
     partido de liga; 5, 10, 15... amarillas → ciclo cumplido, se pierde el siguiente.
     Si el equipo ya jugó un partido de liga después, la sanción se da por cumplida.
  2. Noticias de sanciones de los últimos 30 días ("cuatro partidos de sanción"...): se
     restan los partidos de liga que el equipo ha jugado desde la noticia. Así se detectan
     sanciones largas que las noticias de los últimos días ya no mencionan.
"""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from datetime import date, timedelta
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

import pandas as pd

from ..config import Settings
from ..teams import canonical
from . import http
from .news import SEARCH_NAMES, parse_feed

log = logging.getLogger(__name__)

ESPN_SB = "https://site.api.espn.com/apis/site/v2/sports/soccer/{lg}/scoreboard"
CARD_COLS = ["date", "team", "player", "card"]
YELLOW_CYCLE = 5
NUMBERS = {"un": 1, "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7,
           "ocho": 8, "nueve": 9, "diez": 10}


def plain(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s)).strip()


# ---------------------------------------------------------------------------
# 1) Tarjetas
# ---------------------------------------------------------------------------
def fetch_cards(settings: Settings, season_start: date, online: bool = True, league: str = "esp.1") -> pd.DataFrame:
    cache = settings.raw_dir / f"events_{league}_{season_start.year}.csv"  # tarjetas y goles
    seen_f = cache.with_suffix(".days.json")
    df = pd.read_csv(cache, parse_dates=["date"]) if cache.exists() else pd.DataFrame(columns=CARD_COLS)
    seen = set(json.loads(seen_f.read_text())) if seen_f.exists() else set()
    if not online:
        return df
    rows, today, day = [], date.today(), season_start
    while day <= today:
        key = day.isoformat()
        if key not in seen or (today - day).days <= 2:
            try:
                js = http.get(ESPN_SB.format(lg=league), params={"dates": f"{day:%Y%m%d}"}, timeout=20, retries=1).json()
                done = True
                for ev in js.get("events", []):
                    comp = ev["competitions"][0]
                    if not comp.get("status", {}).get("type", {}).get("completed"):
                        done = False
                        continue
                    names = {c["team"]["id"]: c["team"]["displayName"] for c in comp["competitors"]}
                    for d in comp.get("details", []):
                        is_goal = bool(d.get("scoringPlay")) and not d.get("shootout")
                        if not (d.get("yellowCard") or d.get("redCard") or is_goal):
                            continue
                        ath = (d.get("athletesInvolved") or [{}])[0]
                        team = canonical(names.get((d.get("team") or {}).get("id"), ""), strict=False)
                        kind = "red" if d.get("redCard") else "yellow" if d.get("yellowCard") else \
                            ("own_goal" if d.get("ownGoal") else "goal")
                        rows.append({"date": pd.Timestamp(day), "team": team, "player": ath.get("displayName", ""),
                                     "card": kind})
                if done and (today - day).days > 2:
                    seen.add(key)
            except Exception as e:
                log.warning("Tarjetas ESPN %s: %s", day, e)
                if isinstance(e, http.HostUnavailable):
                    break
        day += timedelta(days=1)
    if rows:
        new = pd.DataFrame(rows, columns=CARD_COLS)
        recent = pd.Timestamp(today - timedelta(days=2))
        df = pd.concat([df[df.date < recent], new], ignore_index=True).drop_duplicates()
    df.to_csv(cache, index=False)
    seen_f.write_text(json.dumps(sorted(seen)))
    return df


def suspended_by_cards(cards: pd.DataFrame, matches: pd.DataFrame, team: str, before: pd.Timestamp) -> list[dict]:
    """Jugadores del equipo con una sanción por tarjetas sin cumplir antes de `before`."""
    if cards is None or cards.empty:
        return []
    c = cards[(cards.team == team) & (cards.date < before)].sort_values("date")
    # partidos de liga del equipo antes de este (jugados o ya programados): en ellos se cumple la sanción
    mine = matches[((matches.home == team) | (matches.away == team)) &
                   (pd.to_datetime(matches.date).dt.normalize() < pd.Timestamp(before).normalize())]
    played_dates = sorted(pd.to_datetime(mine.date).dt.normalize())
    out = []
    for player, g in c.groupby("player"):
        triggers = []
        reds = g[g.card == "red"]
        triggers += [(d, "roja") for d in reds.date]
        y = g[g.card == "yellow"].reset_index(drop=True)
        for k in range(YELLOW_CYCLE - 1, len(y), YELLOW_CYCLE):
            triggers.append((y.date.iloc[k], f"{k + 1}ª amarilla"))
        for d, why in triggers:
            served = any(pd.Timestamp(d).normalize() < pdx for pdx in played_dates)
            if not served:
                out.append({"team": team, "player": player, "status": "out", "reason": "suspension",
                            "source": f"sanción por {why} ({pd.Timestamp(d):%d/%m})"})
    return out


# ---------------------------------------------------------------------------
# 2) Noticias de sanciones
# ---------------------------------------------------------------------------
_NUM = r"(\d+|un|uno|una|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez)"
_SANCTION = [
    re.compile(_NUM + r"\s+partidos?\s+de\s+sanci[oó]n", re.I),
    re.compile(r"sancionad[oa]\s+con\s+" + _NUM + r"\s+partidos?", re.I),
    re.compile(_NUM + r"\s+partidos?\s+de\s+suspensi[oó]n", re.I),
    re.compile(r"suspendid[oa]\s+(?:por|durante)\s+" + _NUM + r"\s+partidos?", re.I),
]
_ANY_SANCTION = re.compile(r"sancion|sanción|suspend|se\s+pierde|no\s+podr[aá]\s+jugar|expulsi", re.I)


def news_url(team: str, days: int = 30) -> str:
    q = f'"{SEARCH_NAMES.get(team, team)}" (sanción OR sancionado OR "partidos de sanción" OR suspendido) when:{days}d'
    return f"https://news.google.com/rss/search?q={quote_plus(q)}&hl=es&gl=ES&ceid=ES:es"


_LOWER_OK = {"el", "la", "los", "las", "de", "del", "y", "a", "e", "o", "que", "con", "para", "por", "sin", "tras"}


def _other_person(text: str, player: str) -> bool:
    """True si el apellido aparece precedido de otro nombre propio (p. ej. "Facundo Bernal" no es
    "Marc Bernal"): evita atribuir a un jugador la sanción de un rival con el mismo apellido."""
    toks = player.split()
    if len(toks) < 2:
        return False
    first, last = plain(toks[0]), plain(toks[-1])
    hits = list(re.finditer(r"([A-ZÁÉÍÓÚÑÜ][\wáéíóúñü]+)\s+" + re.escape(toks[-1]), text))
    if not hits:
        return False
    return all(plain(h.group(1)) not in (first, *[plain(x) for x in toks[:-1]]) and plain(h.group(1)) not in _LOWER_OK
               for h in hits)


def parse_sanctions(items, squad: list[str]) -> list[dict]:
    """[{player, games, date}] de los titulares que hablan de sanciones de jugadores de la plantilla."""
    sq = {p: plain(p) for p in squad}
    # apellidos/apodos distintivos de cada jugador (≥ 4 letras)
    keys = {p: [t for t in v.split() if len(t) >= 4] for p, v in sq.items()}
    out = []
    for it in items:
        text = f"{it.title}. {it.summary}"
        if not _ANY_SANCTION.search(text):
            continue
        t = plain(text)
        who = [p for p, ks in keys.items() if ks and any(re.search(rf"\b{k}\b", t) for k in ks[-1:] + ks[:1])
               and not _other_person(text, p)]
        if len(who) != 1:  # ambiguo o ninguno: mejor no adivinar
            continue
        games = None
        for pat in _SANCTION:
            m = pat.search(text)
            if m:
                v = m.group(1).lower()
                games = int(v) if v.isdigit() else NUMBERS.get(v)
                break
        try:
            pub = parsedate_to_datetime(it.published).replace(tzinfo=None) if it.published else None
        except (TypeError, ValueError):
            pub = None
        out.append({"player": who[0], "games": games, "date": pub, "title": it.title})
    return out


def _had_red(cards: pd.DataFrame | None, team: str, player: str, around: pd.Timestamp) -> bool:
    """¿Vio el jugador una roja en las semanas anteriores a la noticia? Las sanciones de varios
    partidos casi siempre vienen de una expulsión: así se descartan los titulares que nombran al
    agredido, a un rival con el mismo apellido o a quien "no fue sancionado"."""
    if cards is None or cards.empty:
        return False
    from rapidfuzz import fuzz
    c = cards[(cards.team == team) & (cards.card == "red") & (cards.date <= around + pd.Timedelta(days=2)) &
              (cards.date >= around - pd.Timedelta(days=45))]
    return any(fuzz.token_set_ratio(plain(player), plain(x)) >= 85 for x in c.player)


def suspended_by_news(settings: Settings, team: str, squad: list[str], matches: pd.DataFrame,
                      before: pd.Timestamp, online: bool = True, cards: pd.DataFrame | None = None) -> list[dict]:
    if not online or not squad:
        return []
    try:
        items = parse_feed(http.get(news_url(team), timeout=15).content, team, 30)
    except Exception as e:
        log.warning("Noticias de sanciones no disponibles para %s: %s", team, e)
        return []
    found = parse_sanctions(items, squad)
    mine = matches[(matches.home == team) | (matches.away == team)]
    mine_dates = pd.to_datetime(mine.date).dt.normalize()
    best: dict[str, dict] = {}
    for f in found:
        if f["date"] is None or not _had_red(cards, team, f["player"], pd.Timestamp(f["date"])):
            continue
        # partidos entre la noticia y este (jugados o programados antes): en ellos se va cumpliendo
        n_played = int(((mine_dates > pd.Timestamp(f["date"]).normalize()) &
                        (mine_dates < pd.Timestamp(before).normalize())).sum())
        if f["games"]:
            left = f["games"] - n_played
        else:  # "sancionado" sin número: un partido, solo si es reciente
            left = 1 - n_played if (before - pd.Timestamp(f["date"])).days <= 7 else 0
        if left > 0:
            prev = best.get(f["player"])
            if prev is None or left > prev["left"]:
                best[f["player"]] = {"left": left, "title": f["title"]}
    return [{"team": team, "player": p, "status": "out", "reason": "suspension",
             "source": f"sancionado (le quedan {v['left']} partido{'s' if v['left'] > 1 else ''}): {v['title'][:90]}"}
            for p, v in best.items()]


def for_match(settings: Settings, team: str, squad: list[str], matches: pd.DataFrame, cards: pd.DataFrame | None,
              kickoff: pd.Timestamp, online: bool = True) -> pd.DataFrame:
    """Sancionados de un equipo para un partido (nombres ya ajustados a la plantilla)."""
    from rapidfuzz import fuzz, process
    rows = suspended_by_news(settings, team, squad, matches, kickoff, online, cards) + \
        suspended_by_cards(cards, matches, team, kickoff)
    out, seen = [], set()
    plain_sq = {p: plain(p) for p in squad}
    for r in rows:
        if r["player"] in squad:
            name = r["player"]
        else:
            best = process.extractOne(plain(r["player"]), plain_sq, scorer=fuzz.token_set_ratio) if plain_sq else None
            if not best or best[1] < 85:
                continue
            name = best[2]
        if name in seen:
            continue
        seen.add(name)
        out.append({**r, "player": name})
    return pd.DataFrame(out, columns=["team", "player", "status", "reason", "source"])
