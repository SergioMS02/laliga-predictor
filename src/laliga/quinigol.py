"""Quinigol oficial: los 6 partidos de la jornada (y lo que juega la gente) desde el marcador en
directo que publica dataradar.es para eduardolosilla.es, rellenado con el modelo.

En el Quinigol se acierta un partido si se aciertan los goles de los dos equipos (0, 1, 2 o M = 3 o
más). El boleto del modelo marca, en cada partido, la combinación local-visitante más probable de la
matriz de marcadores. Los premios empiezan en 2 aciertos.

Fuente: https://static.dataradar.es/marcador/json/partidos_quinigol.json (público, sin clave).
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone

import numpy as np

from .config import Settings
from .data import http
from .models.dixon_coles import score_matrix

log = logging.getLogger(__name__)

URL = "https://static.dataradar.es/marcador/json/partidos_quinigol.json"
CATS = ["0", "1", "2", "M"]
ALIASES = {"at madrid": "Atlético de Madrid", "atl madrid": "Atlético de Madrid", "ath club": "Athletic Club",
           "r sociedad": "Real Sociedad", "r madrid": "Real Madrid", "r betis": "Betis", "r oviedo": "Oviedo",
           "deportivo": "Deportivo La Coruña", "rayo": "Rayo Vallecano", "r racing": "Racing de Santander",
           "racing": "Racing de Santander", "espanyol": "Espanyol", "celta": "Celta", "r zaragoza": "Zaragoza",
           "r valladolid": "Valladolid", "r gijon": "Sporting de Gijón", "sporting": "Sporting de Gijón",
           "las palmas": "Las Palmas", "u d las palmas": "Las Palmas", "alaves": "Alavés", "leganes": "Leganés"}


def _plain(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]+", " ", s)).strip()


def nice(name: str) -> str:
    """'AT.MADRID' -> 'Atlético de Madrid'; el resto, con mayúscula inicial."""
    p = _plain(name)
    return ALIASES.get(p) or " ".join(w.capitalize() for w in p.split())


def parse(js) -> dict | None:
    if not isinstance(js, list) or not js:
        return None
    rows = []
    for m in sorted(js, key=lambda x: x.get("orden", 0)):
        g = lambda k: m.get(k)  # noqa: E731
        sg = str(g("signo_goles") or "")
        def pub(side):
            v = [g(f"porc15{side}_{c}") for c in CATS]
            known = [x for x in v if isinstance(x, (int, float))]
            if len(known) == 3:  # a veces falta la «M»: es el resto hasta 100
                v = [x if isinstance(x, (int, float)) else max(0, 100 - sum(known)) for x in v]
            return v
        pub_h, pub_a = pub("L"), pub("V")
        rows.append({"n": int(g("orden") or len(rows) + 1), "home_raw": g("local"), "away_raw": g("visitante"),
                     "home": nice(g("local") or ""), "away": nice(g("visitante") or ""),
                     "when": f"{g('dia') or ''} {g('hora') or ''}".strip(),
                     "hg": g("local_goles") if g("local_goles") not in (None, "") else None,
                     "ag": g("visitante_goles") if g("visitante_goles") not in (None, "") else None,
                     "result": sg if len(sg) == 2 else None, "live": bool(g("live")),
                     "public": [[x / 100 for x in pub_h], [x / 100 for x in pub_a]]
                     if all(isinstance(x, (int, float)) for x in pub_h + pub_a) else None})
    return {"jornada": js[0].get("jornada"), "temporada": js[0].get("temporada"), "rows": rows}


def probs_from_goals(lam: float, mu: float) -> dict:
    m = score_matrix(lam, mu, -0.05, 10)
    m = m / m.sum()
    q = np.zeros((4, 4))
    for a in range(m.shape[0]):
        for b in range(m.shape[1]):
            q[min(a, 3), min(b, 3)] += m[a, b]
    a, b = np.unravel_index(int(q.argmax()), q.shape)
    return {"home": [round(float(x), 4) for x in q.sum(1)], "away": [round(float(x), 4) for x in q.sum(0)],
            "joint": [[round(float(x), 4) for x in r] for r in q], "pick": CATS[a] + CATS[b], "p_pick": round(float(q[a, b]), 4)}


def hits_dist(ps: list[float]) -> list[float]:
    """Probabilidad de 0..n aciertos (partidos independientes)."""
    d = np.zeros(len(ps) + 1)
    d[0] = 1.0
    for p in ps:
        d[1:] = d[1:] * (1 - p) + d[:-1] * p
        d[0] *= 1 - p
    return [round(float(x), 5) for x in d]


def update(settings: Settings, preds: list[dict] | None = None, online: bool = True) -> dict:
    path = settings.predictions_dir / "quinigol.json"
    state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"jornadas": {}}
    if not online:
        return state
    try:
        js = http.get(URL, timeout=20, retries=1).json()
    except Exception as e:
        log.warning("Quinigol no disponible: %s", e)
        return state
    pg = parse(js)
    if not pg or not pg["rows"]:
        return state
    key = f"{pg['temporada']}-{pg['jornada']}"
    j = state["jornadas"].get(key, {"jornada": pg["jornada"], "temporada": pg["temporada"], "frozen": False})
    started = any(r["live"] or r["result"] for r in pg["rows"])
    if not j.get("frozen"):
        from . import quiniela
        if preds is None:
            lp = settings.predictions_dir / "latest.json"
            preds = json.loads(lp.read_text(encoding="utf-8")) if lp.exists() else []
        by_pair = {(p["home"], p["away"]): p for p in preds}
        laliga = {t for p in preds for t in (p["home"], p["away"])}
        from .data import results
        import pandas as pd
        sp1 = settings.raw_dir / f"SP1_{results.season_code(results.current_season())}.csv"
        if sp1.exists():
            cur = pd.read_csv(sp1)
            laliga |= set(cur.home) | set(cur.away)
        try:
            from . import eurelo
            tab = eurelo.load(settings)
            euro = eurelo.Lookup(tab) if len(tab) else None
        except Exception:
            euro = None
        elo = quiniela._laliga_elo(settings)
        rows = []
        for r in pg["rows"]:
            mp = quiniela.match_probs(r["home"], r["away"], by_pair, None, laliga, elo, False, euro=euro)
            pr = by_pair.get(tuple(quiniela.resolve(x, laliga, set())[0] for x in (r["home"], r["away"])))
            q = (pr or {}).get("markets", {}).get("quinigol") if pr else None
            if q:
                jm = np.array(q["joint"])
                a, b = np.unravel_index(int(jm.argmax()), jm.shape)
                qq = {**q, "pick": CATS[a] + CATS[b], "p_pick": round(float(jm[a, b]), 4)}
            else:
                qq = probs_from_goals(float(mp["lam"]), float(mp["mu"]))
            rows.append({**r, "source": mp["source"], "p1x2": [round(float(x), 4) for x in mp["p"]], **{"q": qq}})
        j["rows"] = rows
        j["ticket"] = [r["q"]["pick"] for r in rows]
        j["hits_dist"] = hits_dist([r["q"]["p_pick"] for r in rows])
        j["expected_hits"] = round(float(sum(r["q"]["p_pick"] for r in rows)), 2)
        j["late"] = started and "updated" not in j
        j["updated"] = datetime.now(timezone.utc).isoformat(timespec="minutes")
        if started:
            j["frozen"] = True
    # resultados
    for r, live in zip(j.get("rows", []), pg["rows"]):
        r.update(hg=live["hg"], ag=live["ag"], result=live["result"], live=live["live"])
    if j.get("rows") and all(r.get("result") for r in j["rows"]) and not any(r.get("live") for r in j["rows"]):
        j["hits"] = int(sum(r["result"] == t for r, t in zip(j["rows"], j["ticket"])))
        j["team_hits"] = int(sum((r["result"][0] == t[0]) + (r["result"][1] == t[1]) for r, t in zip(j["rows"], j["ticket"])))
        j["settled"] = True
    state["jornadas"][key] = j
    done = [x for x in state["jornadas"].values() if x.get("settled") and not x.get("late")]
    state["summary"] = {"jornadas": len(done), "hits": sum(x["hits"] for x in done),
                        "with_prize": sum(x["hits"] >= 2 for x in done),
                        "expected": round(sum(x.get("expected_hits", 0) for x in done), 2)}
    path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    return state
