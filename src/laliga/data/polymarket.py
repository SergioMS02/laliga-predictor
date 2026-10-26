"""Precios de Polymarket (solo lectura) como fuente de mercado.

Polymarket es un mercado de predicción: el precio del "Sí" de cada contrato es la probabilidad que
el dinero asigna a ese resultado, sin margen de casa. Se usa para tres cosas:

  1. Partidos de LaLiga (1X2): precio fresco y casi sin margen. Se usa en la predicción final solo
     si el mercado tiene volumen y liquidez suficientes (los mercados finos son ruido).
  2. Mercados de temporada (campeón, Champions, descenso, Pichichi): para compararlos con la
     proyección Monte Carlo del modelo.
  3. En directo, la web lee el precio del partido desde el navegador (la API permite CORS).

Comprobado con su histórico (partidos cerrados desde agosto de 2025, precio justo antes del inicio):
su log-loss queda al nivel del cierre de Pinnacle/media de casas, ligeramente mejor. `evaluate`
repite la medición cada semana y decide cuánto peso conserva el modelo frente a Polymarket.

API pública sin clave: gamma-api.polymarket.com (eventos y mercados) y clob.polymarket.com
(histórico de precios). Solo se leen datos; el proyecto no opera en Polymarket.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..config import Settings
from ..teams import canonical
from . import http

log = logging.getLogger(__name__)

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
MIN_MODEL_WEIGHT = 0.1
SERIES = "10193"  # LaLiga en /sports
MATCH_SLUG = re.compile(r"^lal-([a-z0-9]+)-([a-z0-9]+)-(\d{4}-\d{2}-\d{2})$")
ALIASES = {"real racing club": "Santander", "racing santander": "Santander", "rc deportivo a coruna": "La Coruna",
           "deportivo a coruna": "La Coruna", "real sociedad de futbol": "Sociedad"}
SEASON_KINDS = [  # (clave, patrón del slug)
    ("champion", re.compile(r"laliga-\d{4}-champion|laliga-.*winner")),
    ("top2", re.compile(r"laliga-top-2")),
    ("ucl", re.compile(r"laliga-team-to-qualify-for-the-.*champions-league")),
    ("releg", re.compile(r"laliga-teams-relegated")),
    ("pichichi", re.compile(r"laliga-top-goalscorer")),
]


def _plain(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s)).strip()


def team(name: str) -> str | None:
    n = re.sub(r"^Draw \(.*\)$", "", name or "").strip()
    if not n:
        return None
    return ALIASES.get(_plain(n)) or canonical(n, strict=False)


def _yes(m: dict) -> float | None:
    try:
        p = float(json.loads(m.get("outcomePrices") or "[]")[0])
    except (ValueError, IndexError, TypeError):
        return None
    return p if 0 <= p <= 1 else None


def _f(x) -> float:
    try:
        return float(x or 0)
    except (TypeError, ValueError):
        return 0.0


def cfg(settings: Settings) -> dict:
    g = lambda k, d: settings.get("polymarket", k, d)  # noqa: E731
    return {"enabled": bool(g("enabled", True)), "min_volume": float(g("min_volume", 50000)),
            "min_liquidity": float(g("min_liquidity", 10000)), "model_weight": g("model_weight", None),
            "max_overround": float(g("max_overround", 0.06))}


# ---------------------------------------------------------------------------
# 1) Partidos
# ---------------------------------------------------------------------------
def parse_match_event(e: dict) -> dict | None:
    """Evento principal de un partido -> probabilidades 1X2 (normalizadas) y liquidez."""
    slug = e.get("slug") or ""
    if not MATCH_SLUG.match(slug) or " vs. " not in (e.get("title") or ""):
        return None
    hn, an = e["title"].split(" vs. ", 1)
    home, away = team(hn), team(an)
    ph = pd_ = pa = None
    for m in e.get("markets") or []:
        title = m.get("groupItemTitle") or ""
        p = _yes(m)
        if title.startswith("Draw"):
            pd_ = p
        elif team(title) == home:
            ph = p
        elif team(title) == away:
            pa = p
    if None in (ph, pd_, pa) or not home or not away:
        return None
    raw = np.array([ph, pd_, pa], float)
    if raw.sum() <= 0:
        return None
    return {"slug": slug, "home": home, "away": away, "kickoff": e.get("startTime") or e.get("endDate"),
            "p": [round(float(x), 4) for x in raw / raw.sum()], "overround": round(float(raw.sum() - 1), 4),
            "volume": round(_f(e.get("volume"))), "liquidity": round(_f(e.get("liquidity"))),
            "live": bool(e.get("live")), "closed": bool(e.get("closed")),
            "url": f"https://polymarket.com/event/{slug}"}


def fetch_matches(settings: Settings, online: bool = True) -> list[dict]:
    """Partidos de LaLiga abiertos en Polymarket (caché de respaldo de < 1 día)."""
    cache = settings.raw_dir / "polymarket" / "matches.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if online and cfg(settings)["enabled"]:
        out, seen = [], set()
        try:
            for off in range(0, 600, 100):
                js = http.get(f"{GAMMA}/events", params={"series_id": SERIES, "closed": "false", "limit": 100,
                                                         "offset": off}, timeout=40, retries=1).json()
                if not isinstance(js, list) or not js:
                    break
                for e in js:
                    m = parse_match_event(e)
                    if m and m["slug"] not in seen:
                        seen.add(m["slug"])
                        out.append(m)
                if len(js) < 100:
                    break
            if out:
                cache.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
                return out
        except Exception as e:
            log.warning("Polymarket (partidos) no disponible: %s", e)
    if cache.exists() and (pd.Timestamp.now() - pd.Timestamp(cache.stat().st_mtime, unit="s")).days < 1:
        return json.loads(cache.read_text(encoding="utf-8"))
    return []


# ---------------------------------------------------------------------------
# 1b) Mercados secundarios de cada partido (goles, ambos marcan, descanso, córners)
# ---------------------------------------------------------------------------
MIN_SECONDARY_VOLUME = 2000  # $ en el contrato; por debajo, el precio suele ser 0,50 sin operaciones


def parse_secondary(more: dict | None, ht: dict | None, corners: dict | None, home: str, away: str) -> dict:
    """Probabilidades de los mercados secundarios con volumen suficiente."""
    out: dict = {}

    def ok(m):
        return _f(m.get("volumeNum") or m.get("volume")) >= MIN_SECONDARY_VOLUME and not m.get("closed")
    for m in (more or {}).get("markets") or []:
        if not ok(m):
            continue
        t, line, p = m.get("sportsMarketType"), m.get("line"), _yes(m)
        if p is None:
            continue
        vol = round(_f(m.get("volumeNum")))
        if t == "totals" and line is not None:
            out.setdefault("totals", {})[str(float(line))] = {"p": round(p, 4), "volume": vol}
        elif t == "both_teams_to_score":
            out["btts"] = {"p": round(p, 4), "volume": vol}
        elif t == "first_half_totals" and line is not None:
            out.setdefault("ht_totals", {})[str(float(line))] = {"p": round(p, 4), "volume": vol}
    if ht:
        hp = {}
        vol = 0.0
        for m in ht.get("markets") or []:
            p = _yes(m)
            title = m.get("groupItemTitle") or ""
            k = "X" if title.startswith("Draw") else "1" if team(title) == home else "2" if team(title) == away else None
            if k and p is not None:
                hp[k] = p
                vol += _f(m.get("volumeNum"))
        if len(hp) == 3 and vol >= 3 * MIN_SECONDARY_VOLUME:
            tot = sum(hp.values())
            out["half_time"] = {"p": [round(hp[k] / tot, 4) for k in "1X2"], "volume": round(vol)}
    for m in (corners or {}).get("markets") or []:
        if ok(m) and m.get("line") is not None and _yes(m) is not None and float(m["line"]) >= 6.5 \
                and (m.get("groupItemTitle") or "").startswith("Total"):  # córners del partido, no de un equipo
            out.setdefault("corners", {})[str(float(m["line"]))] = {"p": round(_yes(m), 4), "volume": round(_f(m.get("volumeNum")))}
    return out


def fetch_secondary(settings: Settings, m: dict, online: bool = True) -> dict:
    if not online or not m:
        return {}
    ev = {}
    for suf in ("-more-markets", "-halftime-result", "-total-corners"):
        try:
            js = http.get(f"{GAMMA}/events", params={"slug": m["slug"] + suf}, timeout=25, retries=1).json()
            ev[suf] = js[0] if isinstance(js, list) and js else None
        except Exception as e:
            log.warning("Polymarket %s%s: %s", m["slug"], suf, e)
            ev[suf] = None
    return parse_secondary(ev["-more-markets"], ev["-halftime-result"], ev["-total-corners"], m["home"], m["away"])


def _naive(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_convert(None) if t.tzinfo else t


def for_match(markets: list[dict], home: str, away: str, kickoff=None) -> dict | None:
    for m in markets:
        if m["home"] != home or m["away"] != away:
            continue
        if kickoff is not None and m.get("kickoff"):
            try:
                if abs(_naive(m["kickoff"]) - _naive(kickoff)) > pd.Timedelta(days=3):
                    continue
            except (ValueError, TypeError):
                pass
        return m
    return None


def usable(m: dict | None, c: dict, now: pd.Timestamp | None = None) -> tuple[bool, str]:
    """¿Sirve el precio para la predicción? (antes del inicio, con volumen y liquidez)."""
    if not m:
        return False, "sin mercado"
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    if m.get("live") or m.get("closed"):
        return False, "en juego"
    try:
        if m.get("kickoff") and pd.Timestamp(m["kickoff"]) <= now:
            return False, "en juego"
    except (ValueError, TypeError):
        pass
    if m["volume"] < c["min_volume"]:
        return False, f"poco volumen ({m['volume']:,.0f} $)".replace(",", ".")
    if m["liquidity"] < c["min_liquidity"]:
        return False, f"poca liquidez ({m['liquidity']:,.0f} $)".replace(",", ".")
    if abs(m.get("overround", 0)) > c["max_overround"]:
        return False, "precios incoherentes"
    return True, "ok"


def model_weight(settings: Settings) -> float:
    """Peso que conserva el modelo frente a Polymarket (medido cada semana; 0,25 si aún no hay medida)."""
    c = cfg(settings)
    if c["model_weight"] is not None:
        return float(c["model_weight"])
    p = settings.state_dir / "polymarket_eval.json"
    try:
        ev = json.loads(p.read_text(encoding="utf-8"))
        if ev.get("n", 0) >= 100 and ev.get("best_model_weight") is not None:
            # nunca menos de 0,1: la medida dice que casi no cambia la log-loss y así la predicción
            # conserva algo de lo que el modelo sabe (bajas, alineaciones) y es más estable
            return max(MIN_MODEL_WEIGHT, float(ev["best_model_weight"]))
    except (OSError, ValueError):
        pass
    return 0.25


def blend(model: np.ndarray, poly: list[float], w_model: float) -> np.ndarray:
    p = (1 - w_model) * np.asarray(poly, float) + w_model * np.asarray(model, float)
    return p / p.sum()


# ---------------------------------------------------------------------------
# 2) Mercados de temporada
# ---------------------------------------------------------------------------
def parse_season_event(e: dict, kind: str) -> dict:
    outs = []
    for m in e.get("markets") or []:
        name = (m.get("groupItemTitle") or "").strip()
        vol = _f(m.get("volumeNum") or m.get("volume"))
        p = _yes(m)
        if not name or p is None or m.get("closed") or not m.get("active", True):
            continue
        if re.fullmatch(r"(Other|Team [A-Z]|Player [A-Z])", name):
            continue
        row = {"name": name, "p": round(p, 4), "volume": round(vol)}
        if kind != "pichichi":
            row["team"] = team(name)
        outs.append(row)
    outs.sort(key=lambda r: -r["p"])
    return {"kind": kind, "title": e.get("title"), "slug": e.get("slug"), "volume": round(_f(e.get("volume"))),
            "liquidity": round(_f(e.get("liquidity"))), "url": f"https://polymarket.com/event/{e.get('slug')}",
            "outcomes": outs}


def _season_tokens(season: int) -> tuple[str, ...]:
    return (f"{season}-{str(season + 1)[2:]}", f"{season + 1}")


def fetch_season(settings: Settings, season: int, online: bool = True) -> dict:
    """{kind: mercado} para la temporada `season` (año de inicio)."""
    cache = settings.raw_dir / "polymarket" / f"season_{season}.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if online and cfg(settings)["enabled"]:
        try:
            out = {}
            for off in range(0, 400, 100):
                js = http.get(f"{GAMMA}/events", params={"tag_slug": "la-liga", "closed": "false", "limit": 100,
                                                         "offset": off}, timeout=40, retries=1).json()
                if not isinstance(js, list) or not js:
                    break
                for e in js:
                    slug = e.get("slug") or ""
                    if not any(t in slug for t in _season_tokens(season)):
                        continue
                    for kind, pat in SEASON_KINDS:
                        if pat.search(slug) and (kind not in out or _f(e.get("volume")) > out[kind]["volume"]):
                            out[kind] = parse_season_event(e, kind)
                if len(js) < 100:
                    break
            if out:
                out["fetched_at"] = datetime.now(timezone.utc).isoformat(timespec="minutes")
                cache.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
                return out
        except Exception as e:
            log.warning("Polymarket (temporada) no disponible: %s", e)
    if cache.exists() and (pd.Timestamp.now() - pd.Timestamp(cache.stat().st_mtime, unit="s")).days < 3:
        return json.loads(cache.read_text(encoding="utf-8"))
    return {}


def compare_season(markets: dict, projection: dict | None, scorers: dict | None, min_volume: float = 1000) -> dict | None:
    """Modelo frente a Polymarket por equipo/jugador. Los contratos con menos de `min_volume` $
    negociados se marcan como poco fiables (el precio puede ser de una sola orden)."""
    if not markets:
        return None
    out = {"fetched_at": markets.get("fetched_at"), "min_volume": min_volume, "markets": {}}
    proj = {t["team"]: t for t in (projection or {}).get("teams", [])}
    key = {"champion": "p_title", "ucl": "p_ucl", "releg": "p_releg", "top2": None}
    for kind in ("champion", "top2", "ucl", "releg"):
        mk = markets.get(kind)
        if not mk:
            continue
        rows = []
        for o in mk["outcomes"]:
            t = proj.get(o.get("team"))
            model = None
            if t is not None:
                if kind == "top2":
                    pp = t.get("pos_probs") or []
                    model = round(sum(pp[:2]), 4) if pp else None
                else:
                    model = t.get(key[kind])
            rows.append({"team": o.get("team") or o["name"], "name": o["name"], "poly": o["p"], "model": model,
                         "volume": o["volume"], "thin": o["volume"] < min_volume})
        out["markets"][kind] = {"title": mk["title"], "url": mk["url"], "volume": mk["volume"], "rows": rows}
    mk = markets.get("pichichi")
    if mk:
        sc = {_plain(p["name"]): p for p in (scorers or {}).get("players", [])}
        from rapidfuzz import fuzz, process
        rows = []
        for o in mk["outcomes"][:15]:
            best = process.extractOne(_plain(o["name"]), list(sc), scorer=fuzz.token_set_ratio) if sc else None
            m = sc[best[0]] if best and best[1] >= 85 else None
            rows.append({"name": o["name"], "poly": o["p"], "model": m["p"] if m else None,
                         "team": m["team"] if m else None, "volume": o["volume"], "thin": o["volume"] < min_volume})
        out["markets"]["pichichi"] = {"title": mk["title"], "url": mk["url"], "volume": mk["volume"], "rows": rows}
    return out


# ---------------------------------------------------------------------------
# Histórico: precio justo antes del inicio de cada partido cerrado (para medir)
# ---------------------------------------------------------------------------
HIST_COLS = ["slug", "date", "kickoff", "home", "away", "volume", "p_h", "p_d", "p_a"]


def _price_before(token: str, t0: int) -> float | None:
    js = http.get(f"{CLOB}/prices-history", params={"market": token, "startTs": t0 - 12 * 3600, "endTs": t0,
                                                     "fidelity": 5}, timeout=30, retries=1).json()
    pre = [x for x in js.get("history", []) if x["t"] <= t0]
    return float(pre[-1]["p"]) if pre else None


def update_history(settings: Settings, online: bool = True, max_pages: int = 21) -> pd.DataFrame:
    """Añade al histórico los partidos cerrados que falten (de los más recientes hacia atrás)."""
    path = settings.processed_dir / "polymarket_history.csv"
    df = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=HIST_COLS)
    if not online or not cfg(settings)["enabled"]:
        return df
    known, rows = set(df.slug), []
    try:
        for page in range(max_pages):
            js = http.get(f"{GAMMA}/events", params={"series_id": SERIES, "closed": "true", "limit": 100,
                                                     "offset": page * 100, "order": "endDate", "ascending": "false"},
                          timeout=40, retries=1).json()
            if not isinstance(js, list) or not js:
                break
            new_here = 0
            for e in js:
                m = parse_match_event(e)
                if not m or m["slug"] in known:
                    continue
                new_here += 1
                try:
                    t0 = int(pd.Timestamp(m["kickoff"]).timestamp())
                except (ValueError, TypeError):
                    continue
                prices = {}
                for mk in e.get("markets") or []:
                    title = mk.get("groupItemTitle") or ""
                    k = "d" if title.startswith("Draw") else "h" if team(title) == m["home"] else \
                        "a" if team(title) == m["away"] else None
                    try:
                        tok = json.loads(mk["clobTokenIds"])[0]
                        prices[k] = _price_before(tok, t0) if k else None
                    except Exception:
                        prices[k] = None
                if all(prices.get(k) is not None for k in "hda"):
                    s = prices["h"] + prices["d"] + prices["a"]
                    rows.append({"slug": m["slug"], "date": m["slug"][-10:], "kickoff": m["kickoff"], "home": m["home"],
                                 "away": m["away"], "volume": m["volume"], "p_h": round(prices["h"] / s, 4),
                                 "p_d": round(prices["d"] / s, 4), "p_a": round(prices["a"] / s, 4)})
                known.add(m["slug"])
            if new_here == 0 and page > 1 and len(df):
                break  # ya está todo lo reciente
    except Exception as e:
        log.warning("Histórico de Polymarket incompleto: %s", e)
    if rows:
        df = pd.concat([df, pd.DataFrame(rows, columns=HIST_COLS)], ignore_index=True).drop_duplicates("slug")
        df.sort_values("kickoff").to_csv(path, index=False)
    return df


OU_COLS = ["slug", "date", "home", "away", "p_over25", "p_btts", "vol_over25", "vol_btts"]


def update_history_secondary(settings: Settings, online: bool = True, max_new: int = 600) -> pd.DataFrame:
    """Precio antes del inicio de +2,5 goles y de ambos marcan en los partidos cerrados."""
    path = settings.processed_dir / "polymarket_history_goals.csv"
    df = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=OU_COLS)
    main = settings.processed_dir / "polymarket_history.csv"
    if not online or not main.exists():
        return df
    h = pd.read_csv(main)
    todo = h[~h.slug.isin(set(df.slug))].sort_values("kickoff", ascending=False).head(max_new)
    rows = []
    for r in todo.itertuples():
        try:
            js = http.get(f"{GAMMA}/events", params={"slug": r.slug + "-more-markets"}, timeout=25, retries=1).json()
            ev = js[0] if isinstance(js, list) and js else None
            t0 = int(pd.Timestamp(r.kickoff).timestamp())
            p_o = p_b = None
            v_o = v_b = 0.0
            for m in (ev or {}).get("markets") or []:
                t = m.get("sportsMarketType")
                if t == "totals" and m.get("line") is not None and abs(float(m["line"]) - 2.5) < 1e-6:
                    p_o, v_o = _price_before(json.loads(m["clobTokenIds"])[0], t0), _f(m.get("volumeNum"))
                elif t == "both_teams_to_score":
                    p_b, v_b = _price_before(json.loads(m["clobTokenIds"])[0], t0), _f(m.get("volumeNum"))
            rows.append({"slug": r.slug, "date": r.date, "home": r.home, "away": r.away, "p_over25": p_o, "p_btts": p_b,
                         "vol_over25": round(v_o), "vol_btts": round(v_b)})
        except Exception as e:
            log.warning("Polymarket goles %s: %s", r.slug, e)
            if isinstance(e, http.HostUnavailable):
                break
    if rows:
        df = pd.concat([df, pd.DataFrame(rows, columns=OU_COLS)], ignore_index=True).drop_duplicates("slug", keep="last")
        df.to_csv(path, index=False)
    return df


def evaluate_goals(matches: pd.DataFrame, hist: pd.DataFrame, oos: pd.DataFrame | None, min_volume: float = 5000) -> dict:
    """Más de 2,5 goles y ambos marcan: Polymarket frente al cierre de las casas y al modelo."""
    from ..betting import devig2
    from ..models.dixon_coles import score_matrix
    if hist is None or hist.empty:
        return {}
    res = matches[matches.hg.notna()].copy()
    res["d"] = pd.to_datetime(res.date).dt.normalize()
    h = hist.copy()
    h["d"] = pd.to_datetime(h.date)
    x = h.merge(res, on=["home", "away"], suffixes=("", "_m"))
    x = x[(x.d - x.d_m).abs() <= pd.Timedelta(days=3)]
    if oos is not None and len(oos):
        x = x.merge(oos[["season", "home", "away", "dc_lam", "dc_mu"]], on=["season", "home", "away"], how="left")
    out = {}
    for key, ycol, vcol in (("over25", None, "vol_over25"), ("btts", None, "vol_btts")):
        d = x[x[f"p_{key}"].notna() & (x[vcol] >= min_volume)].copy()
        if len(d) < 30:
            out[key] = {"n": int(len(d))}
            continue
        y = ((d.hg + d.ag) > 2.5).astype(float).to_numpy() if key == "over25" else ((d.hg > 0) & (d.ag > 0)).astype(float).to_numpy()
        P = d[f"p_{key}"].to_numpy(float)

        def ll(p, m):
            p = np.clip(p[m], 1e-6, 1 - 1e-6)
            return round(float(-np.mean(y[m] * np.log(p) + (1 - y[m]) * np.log(1 - p))), 4)
        m_all = np.ones(len(d), bool)
        r = {"n": int(len(d)), "polymarket": ll(P, m_all)}
        if "dc_lam" in d:
            mod = []
            for lam, mu in zip(d.dc_lam, d.dc_mu):
                if not np.isfinite([lam, mu]).all():
                    mod.append(np.nan)
                    continue
                sm = score_matrix(lam, mu, -0.05, 10)
                sm = sm / sm.sum()
                i, j = np.indices(sm.shape)
                mod.append(float(sm[(i + j) > 2].sum()) if key == "over25" else float(sm[(i > 0) & (j > 0)].sum()))
            mod = np.array(mod)
            ok = np.isfinite(mod)
            if ok.sum() >= 30:
                r["modelo"] = ll(np.where(ok, mod, 0.5), ok)
                r["polymarket_mismos"] = ll(P, ok)
        if key == "over25" and "ouc_o25" in d:
            bk = np.array([(devig2(a, b) or np.nan) if np.isfinite([a, b]).all() else np.nan for a, b in zip(d.ouc_o25, d.ouc_u25)])
            ok = np.isfinite(bk)
            if ok.sum() >= 30:
                r["cierre_casas"] = ll(np.where(ok, bk, 0.5), ok)
                r["polymarket_mismos_casas"] = ll(P, ok)
        out[key] = r
    return out


def evaluate(settings: Settings, matches: pd.DataFrame, hist: pd.DataFrame, bt: pd.DataFrame | None = None) -> dict:
    """Log-loss de Polymarket (precio antes del inicio) frente al cierre de las casas y al modelo, y
    el peso del modelo que minimiza la log-loss de la mezcla."""
    from ..models.market import devig_power
    if hist is None or hist.empty:
        return {}
    c = cfg(settings)
    res = matches[matches.hg.notna()].copy()
    res["d"] = pd.to_datetime(res.date).dt.normalize()
    h = hist.copy()
    h["d"] = pd.to_datetime(h.date)
    x = h.merge(res.reset_index(), on=["home", "away"], suffixes=("", "_m"))
    x = x[(x.d - x.d_m).abs() <= pd.Timedelta(days=3)]
    if x.empty:
        return {}
    y = np.where(x.hg > x.ag, 0, np.where(x.hg == x.ag, 1, 2))
    P = x[["p_h", "p_d", "p_a"]].to_numpy(float)
    close = devig_power(x[["odds_close_h", "odds_close_d", "odds_close_a"]].to_numpy(float))
    ok = np.isfinite(close).all(1)
    # fuera los precios que delatan un horario mal puesto (precio ya en juego)
    ok &= ~(np.abs(P - np.where(np.isfinite(close), close, P)).max(1) > 0.15)
    model = None
    if bt is not None and len(bt):
        cols = ["stacker_mercado_H", "stacker_mercado_D", "stacker_mercado_A"]
        b = bt.reindex(x["index"])
        model = b[cols].to_numpy(float)
        alt = b[["stacker_H", "stacker_D", "stacker_A"]].to_numpy(float)
        model = np.where(np.isfinite(model), model, alt)
        ok &= np.isfinite(model).all(1)
    liquid = ok & (x.volume.to_numpy(float) >= c["min_volume"])

    def ll(p, m):
        return round(float(-np.mean(np.log(np.clip(p[m][np.arange(m.sum()), y[m]], 1e-6, 1)))), 4)
    out = {"n": int(liquid.sum()), "n_all": int(ok.sum()), "min_volume": c["min_volume"],
           "since": str(x.d[ok].min().date()) if ok.any() else None}
    if liquid.sum() < 30:
        return out
    out["logloss"] = {"polymarket": ll(P, liquid), "cierre_casas": ll(close, liquid)}
    if model is not None:
        out["logloss"]["modelo"] = ll(model, liquid)
        grid = np.round(np.arange(0, 1.01, 0.05), 2)
        scores = [ll((1 - w) * P + w * model, liquid) for w in grid]
        k = int(np.argmin(scores))
        out["blend"] = [{"w_model": float(w), "logloss": s} for w, s in zip(grid, scores)]
        out["best_model_weight"] = float(grid[k])
        w_used = max(MIN_MODEL_WEIGHT, float(grid[k]))
        out["model_weight_used"] = w_used
        out["logloss"]["mezcla"] = ll((1 - w_used) * P + w_used * model, liquid)
    # diferencia Polymarket - cierre con intervalo bootstrap
    d = -np.log(np.clip(P[liquid][np.arange(liquid.sum()), y[liquid]], 1e-6, 1)) + \
        np.log(np.clip(close[liquid][np.arange(liquid.sum()), y[liquid]], 1e-6, 1))
    rng = np.random.default_rng(3)
    bs = [rng.choice(d, len(d)).mean() for _ in range(1000)]
    out["diff_vs_close"] = {"mean": round(float(d.mean()), 4), "ci95": [round(float(np.percentile(bs, 2.5)), 4),
                                                                      round(float(np.percentile(bs, 97.5)), 4)]}
    return out


def run_evaluation(settings: Settings, matches: pd.DataFrame, online: bool = True) -> dict:
    hist = update_history(settings, online)
    p = settings.processed_dir / "backtest_predictions.csv"
    bt = pd.read_csv(p, index_col=0) if p.exists() else None
    ev = evaluate(settings, matches.reset_index(drop=True), hist, bt)
    try:
        goals = update_history_secondary(settings, online)
        op = settings.processed_dir / "oos_table.csv"
        oos = pd.read_csv(op) if op.exists() else None
        ev["goals"] = evaluate_goals(matches.reset_index(drop=True), goals, oos)
    except Exception as e:
        log.warning("Polymarket (goles) sin evaluar: %s", e)
    if ev:
        ev["updated"] = datetime.now(timezone.utc).date().isoformat()
        (settings.state_dir / "polymarket_eval.json").write_text(json.dumps(ev, ensure_ascii=False, indent=1),
                                                                 encoding="utf-8")
    return ev
