"""¿Gana dinero la quiniela «contra la mayoría»? Medición con jornadas reales y premios reales.

Para cada jornada ya terminada de la temporada se reúne:
  - los partidos, resultados, premios oficiales por categoría y recaudación (combinacionganadora.com,
    yendo hacia atrás jornada a jornada con el enlace «anterior»);
  - lo que jugó la gente (porcentajes de Loterías) y las probabilidades del mercado (Betfair) que
    eduardolosilla.es publica en la página de escrutinio de cada jornada.

Con las probabilidades del mercado se rellenan los tres boletos del proyecto (sencilla, reducida y
«contra la mayoría») exactamente como se hace en directo, y se cobran con la tabla oficial de premios
de esa jornada. Así se ve el rendimiento real de cada boleto, sin simulación.

Limitaciones: solo hay porcentajes de la temporada en curso (las páginas públicas no dan las
anteriores), así que la muestra es pequeña y la varianza enorme (casi todo el premio esperado viene
de plenos muy improbables). Los porcentajes de la gente son los finales de Loterías.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import numpy as np

from . import quiniela as q
from .config import Settings
from .data import http

log = logging.getLogger(__name__)

EL_JORNADA = "https://www.eduardolosilla.es/quiniela/ayudas/escrutinio/jornada_{n}"


def _cache(settings: Settings):
    d = settings.raw_dir / "quiniela_hist"
    d.mkdir(parents=True, exist_ok=True)
    return d


def collect(settings: Settings, online: bool = True, max_back: int = 45) -> list[dict]:
    """Jornadas terminadas de la temporada en curso (las páginas terminadas se guardan y no se repiten)."""
    d = _cache(settings)
    out = []
    page = q.fetch_page() if online else None
    seen = set()
    date = page.get("prev") if page else None
    if page and q._finished(page):
        date = page["date"]
    first_jornada = None
    for _ in range(max_back):
        if not date or date in seen:
            break
        seen.add(date)
        f = d / f"cg_{date}.json"
        if f.exists():
            pg = json.loads(f.read_text(encoding="utf-8"))
        elif online:
            pg = q.fetch_page(date)
            if not pg:
                break
            if q._finished(pg) and pg.get("prizes"):
                f.write_text(json.dumps(pg, ensure_ascii=False), encoding="utf-8")
        else:
            break
        jn = pg.get("jornada")
        if first_jornada is not None and jn and jn >= first_jornada:
            break  # se pasó a la temporada anterior
        if jn:
            first_jornada = jn
        if q._finished(pg) and pg.get("prizes"):
            out.append(pg)
        if jn == 1:
            break
        date = pg.get("prev")
    return out


def public_for(settings: Settings, jornada: int, season_tag: int | None, online: bool = True) -> dict | None:
    d = _cache(settings)
    f = d / f"el_{season_tag or 'cur'}_{jornada}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    if not online:
        return None
    try:
        pub = q.parse_public(http.get(EL_JORNADA.format(n=jornada), timeout=25, retries=1, headers=q._UA).text)
    except Exception as e:
        log.warning("Porcentajes de la jornada %s no disponibles: %s", jornada, e)
        return None
    v = pub.get(jornada)
    if v and v.get("lae") and v.get("real") and all(v["lae"][:14]) and all(v["real"][:14]):
        f.write_text(json.dumps(v), encoding="utf-8")
        return v
    return None


def run(settings: Settings, online: bool = True) -> dict:
    c = q.cfg(settings)
    pages = collect(settings, online)
    diag = {"paginas": len(pages), "sin_porcentajes": [], "errores": []}
    rows_out = []
    for pg in sorted(pages, key=lambda p: p["date"]):
        pub = public_for(settings, pg["jornada"], None, online)
        if not pub or len(pg["matches"]) < 14:
            diag["sin_porcentajes"].append(pg["jornada"])
            continue
        rows = []
        for k, m in enumerate(pg["matches"][:15]):
            p = pub["real"][k] if k < len(pub["real"]) and pub["real"][k] else None
            if k < 14 and p is None:
                break
            p = p or [0.45, 0.28, 0.27]
            lam, mu = q.goals_from_probs(p)
            rows.append({"n": m["n"], "home": m["home"], "away": m["away"], "p": p, "lam": lam, "mu": mu})
        if len(rows) < 14:
            continue
        rec = pg.get("recaudacion") or q.DEFAULT_RECAUDACION
        try:
            tickets = q.build_tickets(rows, c, pub["lae"][:14], rec)
            j = {"tickets": tickets, "matches": pg["matches"][:15], "prizes": pg["prizes"],
                 "pleno": q.pleno_probs(rows[14]["lam"], rows[14]["mu"]) if len(rows) > 14 else None}
            res = q.settle(j)
        except Exception as e:
            diag["errores"].append(f"{pg['jornada']}: {e}")
            continue
        rows_out.append({"date": pg["date"], "jornada": pg["jornada"], "recaudacion": rec,
                         **{k: {"hits": res[k]["hits"], "prize": res[k]["prize"], "cost": tickets[k]["cost"],
                                "ev": tickets[k].get("ev"), "p10": tickets[k].get("p10plus")}
                            for k in tickets}})
    summary = {}
    for k in ("sencilla", "reducida", "valor"):
        r = [x[k] for x in rows_out if k in x]
        if not r:
            continue
        cost = sum(x["cost"] for x in r)
        prize = sum(x["prize"] for x in r)
        evs = [x["ev"] for x in r if x.get("ev") is not None]
        summary[k] = {"jornadas": len(r), "coste": round(cost, 2), "premios": round(prize, 2),
                      "roi": round(prize / cost - 1, 4) if cost else None,
                      "con_premio": int(sum(x["prize"] > 0 for x in r)),
                      "aciertos_medios": round(float(np.mean([x["hits"] for x in r])), 2),
                      "ev_estimado": round(float(np.sum(evs)), 2) if evs else None}
    out = {"updated": datetime.now(timezone.utc).isoformat(timespec="minutes"), "jornadas": rows_out, "summary": summary,
           "diagnostico": diag,
           "nota": "Probabilidades del mercado (Betfair) y porcentajes de Loterías publicados por eduardolosilla.es; "
                   "premios y resultados oficiales de combinacionganadora.com."}
    (settings.predictions_dir / "quiniela_backtest.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out
