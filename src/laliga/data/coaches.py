"""Cambios de entrenador.

Fuente automática: Wikidata (gratis, sin clave), propiedad P286 «entrenador» con
fecha de inicio (P580). La cobertura de Wikidata es irregular, así que se combina
con data/coaches_manual.csv (team,date,coach), que siempre tiene prioridad.
"""
from __future__ import annotations

import logging

import pandas as pd

from ..config import Settings
from ..teams import canonical, is_known
from . import http

log = logging.getLogger(__name__)

SPARQL = "https://query.wikidata.org/sparql"
LALIGA_QID = "Q324867"
QUERY = """
SELECT ?clubLabel ?coachLabel ?start WHERE {
  ?club p:P118/ps:P118 wd:%s .
  ?club p:P286 ?st . ?st ps:P286 ?coach . ?st pq:P580 ?start .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "es,en". }
}""" % LALIGA_QID
COLS = ["team", "date", "coach"]


def parse_wikidata(js: dict) -> pd.DataFrame:
    rows = []
    for b in (js.get("results", {}) or {}).get("bindings", []) or []:
        try:
            team = canonical(b["clubLabel"]["value"], strict=False, fuzzy=False)
            start = pd.to_datetime(b["start"]["value"][:10], errors="coerce")
            coach = b["coachLabel"]["value"]
        except (KeyError, TypeError):
            continue
        if is_known(team) and pd.notna(start):
            rows.append({"team": team, "date": start, "coach": coach})
    return pd.DataFrame(rows, columns=COLS)


def load_coaches(settings: Settings, online: bool = True) -> pd.DataFrame:
    cache = settings.raw_dir / "coaches_wikidata.csv"
    frames = []
    wd = None
    if online:
        try:
            js = http.get(SPARQL, params={"query": QUERY, "format": "json"}, timeout=60, retries=1,
                          headers={"Accept": "application/sparql-results+json"}).json()
            wd = parse_wikidata(js)
            if len(wd):
                wd.to_csv(cache, index=False)
        except Exception as e:
            log.warning("Wikidata no disponible: %s", e)
    if wd is None and cache.exists():
        wd = pd.read_csv(cache, parse_dates=["date"])
    if wd is not None and len(wd):
        frames.append(wd.assign(_src=0))
    man = settings.data_dir / "coaches_manual.csv"
    if man.exists():
        m = pd.read_csv(man, parse_dates=["date"])
        m["team"] = m["team"].map(canonical)
        frames.append(m[COLS].assign(_src=1))
    if not frames:
        return pd.DataFrame(columns=COLS)
    out = pd.concat(frames, ignore_index=True).sort_values(["team", "date", "_src"])
    # mismo equipo y fecha: gana el manual
    return out.drop_duplicates(["team", "date"], keep="last")[COLS].reset_index(drop=True)
