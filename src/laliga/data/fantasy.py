"""Lesionados y sancionados de LaLiga desde futbolfantasy.com (listas estructuradas y actualizadas a
diario: tipo de lesión, desde cuándo, hasta cuándo, duda o baja confirmada, y sanciones con su motivo
y cuántos partidos lleva cumplidos). Mucho más fiable que leer titulares de noticias.

Cada fila se convierte en "jornadas" del equipo: la jornada de un partido es su orden por fecha
dentro de la temporada del equipo, así que "Duda para la jornada 11" se aplica al partido nº 11
del equipo y una "Baja hasta mediados de noviembre" a todos los partidos anteriores a esa fecha.
"""
from __future__ import annotations

import html as _html
import logging
import re
from datetime import date

import pandas as pd

from ..config import Settings
from ..teams import canonical
from . import http

log = logging.getLogger(__name__)

BASE = "https://www.futbolfantasy.com/laliga/"
_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
       "Accept-Language": "es-ES,es;q=0.9"}
MONTHS = {m: i + 1 for i, m in enumerate(["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
                                          "septiembre", "octubre", "noviembre", "diciembre"])}
TEAM_FIX = {"rayo": "Vallecano", "racing": "Santander", "atletico": "Ath Madrid", "atlético": "Ath Madrid",
            "athletic": "Ath Bilbao", "deportivo": "La Coruna", "real sociedad": "Sociedad", "alaves": "Alaves",
            "alavés": "Alaves", "espanyol": "Espanol", "betis": "Betis", "celta": "Celta", "oviedo": "Oviedo",
            "real oviedo": "Oviedo", "leganes": "Leganes", "leganés": "Leganes", "mallorca": "Mallorca"}
COLS = ["team", "player", "status", "reason", "detail", "until", "jornadas", "start_prob", "source"]


def _txt(s: str) -> str:
    return re.sub(r"\s+", " ", _html.unescape(re.sub(r"<[^>]+>", " ", s or ""))).strip()


def _team(name: str) -> str:
    n = _txt(name)
    return TEAM_FIX.get(n.lower()) or canonical(n, strict=False)


def until_date(text: str, today: date) -> date | None:
    """'Baja hasta mediados de noviembre' -> fecha aproximada (final del periodo, para ir sobre seguro)."""
    t = text.lower().replace("mediadios", "mediados").replace("principio de", "principios de")
    m = re.search(r"hasta\s+(?:(principios|mediados|finales)\s+de\s+)?([a-záéíóú]+)(?:-([a-záéíóú]+))?(?:\s+(\d{4}))?", t)
    if not m:
        return None
    part, mon, mon2, year = m.groups()
    mon = (mon2 or mon)
    if mon not in MONTHS:
        return None
    month = MONTHS[mon]
    y = int(year) if year else (today.year if month >= today.month - 1 else today.year + 1)
    day = {"principios": 10, "mediados": 20, "finales": 31}.get(part, 25 if not part else 15)
    import calendar
    day = min(day, calendar.monthrange(y, month)[1])
    return date(y, month, day)


def parse_injuries(page: str, today: date | None = None) -> pd.DataFrame:
    today = today or date.today()
    rows = []
    for block in re.split(r'<header class="title[^"]*v3-widget-header', page)[1:]:
        tm = re.search(r'v3-widget-header__label"><a[^>]*>(.*?)</a>', block, flags=re.S)
        if not tm:
            continue
        team = _team(tm.group(1))
        for el in re.findall(r'<div class="elemento lesionado.*?(?=<div class="elemento lesionado|</section>)', block, flags=re.S):
            pl = re.search(r'class="jugador">(.*?)</a>', el, flags=re.S)
            if not pl:
                continue
            les = re.search(r'<span class="lesion">(.*?)</span>', el, flags=re.S)
            st = re.search(r'rounded-card__status[^"]*">(.*?)</span>', el, flags=re.S)
            prob = re.search(r'probabilidad-widget">\s*<span[^>]*>(\d+)%', el, flags=re.S)
            status_txt = _txt(st.group(1)) if st else ""
            s_low = status_txt.lower()
            if s_low.startswith("disponible"):
                continue
            jm = re.search(r"jornada\s+(\d+)", s_low)
            status = "doubtful" if s_low.startswith("duda") else "out"
            rows.append({"team": team, "player": _txt(pl.group(1)), "status": status, "reason": "injury",
                         "detail": _txt(les.group(1)) if les else "", "until": until_date(status_txt, today),
                         "jornadas": [int(jm.group(1))] if jm else None,
                         "start_prob": int(prob.group(1)) if prob else None,
                         "source": f"futbolfantasy: {_txt(les.group(1)) if les else 'lesión'} · {status_txt}"})
    return pd.DataFrame(rows, columns=COLS)


def parse_sanctions(page: str) -> pd.DataFrame:
    jor = re.search(r"Sancionados\s*-\s*Jornada\s*(\d+)", _txt(page))
    j = int(jor.group(1)) if jor else None
    rows = []
    for block in re.split(r'<header class="title[^"]*v3-widget-header', page)[1:]:
        tm = re.search(r'v3-widget-header__label"><a[^>]*>(.*?)</a>', block, flags=re.S)
        if not tm:
            continue
        team = _team(tm.group(1))
        for el in re.findall(r'<div class="elemento sancionado.*?(?=<div class="elemento sancionado|</section>)', block, flags=re.S):
            pl = re.search(r'class="jugador">(.*?)</a>', el, flags=re.S)
            sc = re.search(r'<span class="sancion">(.*?)</span>', el, flags=re.S)
            if not pl or j is None:
                continue
            why = _txt(re.sub(r"<div.*", "", sc.group(1), flags=re.S)) if sc else "Sanción"
            js = [j]
            m = re.search(r"\((\d+)\s*/\s*(\d+)\)", why)
            if m:  # (3/4): cumple el 3º de 4 en esta jornada -> también las anteriores y las siguientes
                served, total = int(m.group(1)), int(m.group(2))
                js = list(range(j - served + 1, j + total - served + 1))
            rows.append({"team": team, "player": _txt(pl.group(1)), "status": "out", "reason": "suspension",
                         "detail": why.replace("amarijlla", "amarilla"), "until": None, "jornadas": js,
                         "start_prob": 0, "source": f"futbolfantasy: {why.replace('amarijlla', 'amarilla')} (jornada {j})"})
    return pd.DataFrame(rows, columns=COLS)


def fetch(settings: Settings, online: bool = True) -> pd.DataFrame | None:
    """Lesionados + sancionados. None si la fuente no responde (para usar el respaldo)."""
    cache_dir = settings.raw_dir / "fantasy"
    cache_dir.mkdir(parents=True, exist_ok=True)
    pages = {}
    for key in ("lesionados", "sancionados"):
        f = cache_dir / f"{key}.html"
        text = None
        if online:
            try:
                text = http.get(BASE + key, timeout=25, retries=1, headers=_UA).text
                if "elemento" in text:
                    f.write_text(text, encoding="utf-8")
                else:
                    text = None
            except Exception as e:
                log.warning("futbolfantasy %s no disponible: %s", key, e)
        if text is None and f.exists() and (pd.Timestamp.now() - pd.Timestamp(f.stat().st_mtime, unit="s")).days < 2:
            text = f.read_text(encoding="utf-8")
        pages[key] = text
    if not pages["lesionados"] and not pages["sancionados"]:
        return None
    frames = []
    if pages["lesionados"]:
        frames.append(parse_injuries(pages["lesionados"]))
    if pages["sancionados"]:
        frames.append(parse_sanctions(pages["sancionados"]))
    return pd.concat(frames, ignore_index=True) if frames else None


def team_jornada(matches: pd.DataFrame, team: str, season: int, when: pd.Timestamp) -> int | None:
    """Nº de jornada del partido del equipo en la fecha `when` (orden por fecha en su temporada)."""
    s = matches[(matches.season == season) & ((matches.home == team) | (matches.away == team))].sort_values("date")
    d = pd.Timestamp(when).tz_localize(None) if pd.Timestamp(when).tzinfo else pd.Timestamp(when)
    d = d.normalize()
    dates = pd.to_datetime(s.date).dt.normalize().tolist()
    best = None
    for k, x in enumerate(dates, start=1):  # la fecha más cercana (las fuentes difieren a veces en un día)
        gap = abs((x - d).days)
        if gap <= 3 and (best is None or gap < best[0]):
            best = (gap, k)
    return best[1] if best else None


def page_jornada(av: pd.DataFrame | None) -> int | None:
    """Jornada a la que se refiere la página de sancionados (la próxima)."""
    if av is None or av.empty:
        return None
    js = [min(x) for x in av[av.reason == "suspension"].jornadas if x]
    return max(js) if js else None


def for_match(av: pd.DataFrame | None, matches: pd.DataFrame, team: str, season: int, when: pd.Timestamp) -> pd.DataFrame:
    """Bajas y dudas de un equipo para un partido concreto."""
    if av is None or av.empty:
        return pd.DataFrame(columns=["team", "player", "status", "reason", "source", "start_prob"])
    jn = team_jornada(matches, team, season, when)
    d = pd.Timestamp(when).date()
    out = []
    for r in av[av.team == team].itertuples():
        if r.jornadas:
            if jn is None or jn not in r.jornadas:
                continue
        elif r.until is not None and not pd.isna(r.until) and d > r.until:
            continue  # ya habrá vuelto
        out.append({"team": team, "player": r.player, "status": r.status, "reason": r.reason,
                    "source": r.source, "start_prob": r.start_prob})
    return pd.DataFrame(out, columns=["team", "player", "status", "reason", "source", "start_prob"])
