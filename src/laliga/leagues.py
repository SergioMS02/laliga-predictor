"""Otras competiciones: Segunda, Premier League, Serie A, Bundesliga y Ligue 1 (Dixon-Coles + mercado),
y partidos de Copa del Rey, Champions, Europa League, Conference y Liga F (Elo europeo).

Para cada liga: clasificación, proyección de la temporada (los partidos que faltan son los cruces de
ida y vuelta aún no jugados), predicción de los próximos partidos y su seguimiento. La mezcla entre
el modelo y el mercado se mide con la última temporada completa de cada liga (`evaluate`).

Escribe predictions/leagues.json.
"""
from __future__ import annotations

import io
import json
import logging
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .config import Settings
from .data import http, results
from .data.results import current_season, season_code
from .models.dixon_coles import DixonColes, score_matrix
from .models.market import devig_power

log = logging.getLogger(__name__)

LEAGUES = {
    "SP2": {"name": "Segunda División", "espn": "esp.2", "teams": 22,
            "zones": {"ascenso": (1, 2), "playoff": (3, 6), "descenso": (19, 22)}},
    "E0": {"name": "Premier League", "espn": "eng.1", "teams": 20,
           "zones": {"campeón": (1, 1), "champions": (1, 4), "descenso": (18, 20)}},
    "I1": {"name": "Serie A", "espn": "ita.1", "teams": 20,
           "zones": {"campeón": (1, 1), "champions": (1, 4), "descenso": (18, 20)}},
    "D1": {"name": "Bundesliga", "espn": "ger.1", "teams": 18,
           "zones": {"campeón": (1, 1), "champions": (1, 4), "descenso": (17, 18)}},
    "F1": {"name": "Ligue 1", "espn": "fra.1", "teams": 18,
           "zones": {"campeón": (1, 1), "champions": (1, 3), "descenso": (17, 18)}},
}
CUP_LEAGUES = {"esp.copa_del_rey": "Copa del Rey", "uefa.champions": "Champions League",
               "uefa.europa": "Europa League", "uefa.europa.conf": "Conference League", "esp.w.1": "Liga F"}
ESPN_SB = "https://site.api.espn.com/apis/site/v2/sports/soccer/{lg}/scoreboard"
SEASONS_BACK = 3


def _r(x, n=4):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(x) else round(x, n)


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
def load_league(settings: Settings, div: str, online: bool = True) -> pd.DataFrame:
    """Temporadas recientes de una liga de football-data (con cuotas), nombres tal cual."""
    frames = []
    cur = current_season()
    for y in range(cur - SEASONS_BACK, cur + 1):
        if div == "SP2":
            df = results.fetch_season(settings, y, online=online, div="SP2")
        else:
            df = _fetch_foreign(settings, div, y, online)
        if df is not None and len(df):
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=results.COLUMNS)
    m = pd.concat(frames, ignore_index=True)
    fx = results.fetch_fixtures(settings, online=online, div=div)
    if len(fx):  # partidos próximos (con cuotas) que aún no estén
        key = set(zip(m.season, m.home, m.away))
        fx = fx[[(s, h, a) not in key for s, h, a in zip(fx.season, fx.home, fx.away)]]
        m = pd.concat([m, fx], ignore_index=True)
    return m.sort_values("date").reset_index(drop=True)


def _fetch_foreign(settings: Settings, div: str, y: int, online: bool) -> pd.DataFrame | None:
    code = season_code(y)
    cache = settings.raw_dir / "leagues" / f"{div}_{code}.csv"
    cache.parent.mkdir(parents=True, exist_ok=True)
    is_cur = y >= current_season()
    if cache.exists() and (not online or not is_cur):
        return pd.read_csv(cache, parse_dates=["date"])
    if not online:  # la descarga del Elo europeo guarda el mismo CSV original
        raw_euro = settings.raw_dir / "euro" / f"{div}_{code}.csv"
        if raw_euro.exists():
            raw = pd.read_csv(raw_euro, on_bad_lines="skip", encoding_errors="replace")
            return results.standardize_football_data(raw, "football-data", div) if "HomeTeam" in raw else None
        return None
    try:
        text = http.get(f"https://www.football-data.co.uk/mmz4281/{code}/{div}.csv", timeout=30, retries=1) \
            .content.decode("utf-8-sig", errors="replace")
        raw = pd.read_csv(io.StringIO(text), on_bad_lines="skip")
        if "HomeTeam" not in raw:
            return None
        df = results.standardize_football_data(raw, "football-data", div)
        df.to_csv(cache, index=False)
        return df
    except Exception as e:
        log.warning("Liga %s %s no disponible: %s", div, code, e)
        return pd.read_csv(cache, parse_dates=["date"]) if cache.exists() else None


def standings(m: pd.DataFrame, season: int) -> pd.DataFrame:
    from .site_export import standings as st
    return st(m[m.season == season], season)


def remaining_pairs(m: pd.DataFrame, season: int) -> pd.DataFrame:
    """Cruces de ida y vuelta que faltan (liga a doble vuelta)."""
    s = m[m.season == season]
    teams = sorted(set(s.home) | set(s.away))
    played = set(zip(s[s.hg.notna()].home, s[s.hg.notna()].away))
    sched = {(h, a): d for h, a, d in zip(s.home, s.away, s.date)}
    rows = [{"home": h, "away": a, "date": sched.get((h, a))} for h in teams for a in teams
            if h != a and (h, a) not in played]
    return pd.DataFrame(rows, columns=["home", "away", "date"])


# ---------------------------------------------------------------------------
# Modelo y medición
# ---------------------------------------------------------------------------
def _dc(settings: Settings, m: pd.DataFrame, ref, season: int) -> DixonColes:
    s = m[m.season == season]
    teams = sorted(set(s.home) | set(s.away))
    prev = m[m.season == season - 1]
    new = (set(teams) - (set(prev.home) | set(prev.away))) if len(prev) else set()
    return DixonColes.from_settings(settings).fit(m, ref, teams, new)


def evaluate(settings: Settings, m: pd.DataFrame, season: int) -> dict:
    """Log-loss del modelo, del mercado y de su mezcla en la temporada `season` (reajuste mensual)."""
    s = m[(m.season == season) & m.hg.notna()].copy()
    if len(s) < 200:
        return {}
    s["month"] = s.date.dt.to_period("M")
    P, Q, Y = [], [], []
    for _, blk in s.groupby("month"):
        dc = _dc(settings, m, blk.date.min(), season)
        for r in blk.itertuples():
            p = dc.predict(r.home, r.away)
            P.append([p["p_home"], p["p_draw"], p["p_away"]])
            Q.append(devig_power(np.array([[r.odds_h, r.odds_d, r.odds_a]], float))[0])
            Y.append(0 if r.hg > r.ag else 1 if r.hg == r.ag else 2)
    P, Q, Y = np.array(P), np.array(Q), np.array(Y)
    ok = np.isfinite(Q).all(1)
    P, Q, Y = P[ok], Q[ok], Y[ok]

    def ll(x):
        return round(float(-np.mean(np.log(np.clip(x[np.arange(len(Y)), Y], 1e-9, 1)))), 4)
    grid = np.round(np.arange(0, 1.01, 0.1), 1)
    scores = [ll((1 - w) * Q + w * P) for w in grid]
    k = int(np.argmin(scores))
    return {"season": season, "n": int(len(Y)), "modelo": ll(P), "mercado": ll(Q), "mezcla": scores[k],
            "w_model": float(grid[k])}


def project(settings: Settings, m: pd.DataFrame, season: int, zones: dict, n: int = 5000) -> dict:
    from .site_export import season_sigma, simulate_rest
    table = standings(m, season)
    rem = remaining_pairs(m, season)
    lam = mu = np.array([])
    if len(rem):
        dc = _dc(settings, m, pd.Timestamp.now().normalize() + pd.Timedelta(days=1), season)
        eg = [dc.expected_goals(h, a) for h, a in zip(rem.home, rem.away)]
        lam, mu = np.array([e[0] for e in eg]), np.array([e[1] for e in eg])
    sim = simulate_rest(table, rem, lam, mu, n if len(rem) else 1, season_sigma(settings), seed=3)
    ranks, pts = sim["ranks"], sim["pts"]
    out = []
    for i, t in enumerate(sim["teams"]):
        r = ranks[:, i]
        out.append({"team": t, **{f"p_{z}": _r(((r >= lo) & (r <= hi)).mean()) for z, (lo, hi) in zones.items()},
                    "exp_pts": _r(pts[:, i].mean(), 1), "exp_pos": _r(r.mean(), 1)})
    out.sort(key=lambda d: d["exp_pos"])
    return {"table": table.drop(columns=["home_pts", "home_pj", "away_pts", "away_pj"], errors="ignore").to_dict(orient="records"),
            "projection": out, "remaining": int(len(rem)), "zones": {k: list(v) for k, v in zones.items()}}


def upcoming(settings: Settings, m: pd.DataFrame, season: int, w_model: float, days: int = 8) -> list[dict]:
    now = pd.Timestamp.now(tz="UTC")
    s = m[(m.season == season) & m.hg.isna()]
    s = s[(s.date >= now.tz_localize(None).normalize() - pd.Timedelta(days=1)) &
          (s.date <= now.tz_localize(None).normalize() + pd.Timedelta(days=days))]
    if s.empty:
        return []
    dc = _dc(settings, m, now.tz_localize(None).normalize() + pd.Timedelta(days=1), season)
    out = []
    for r in s.itertuples():
        started = bool(isinstance(r.kickoff, pd.Timestamp) and not pd.isna(r.kickoff) and r.kickoff <= now)
        if started and r.kickoff <= now - pd.Timedelta(hours=3):
            continue  # terminado: se queda solo en el seguimiento
        p = dc.predict(r.home, r.away)
        pm = np.array([p["p_home"], p["p_draw"], p["p_away"]])
        q = devig_power(np.array([[r.odds_h, r.odds_d, r.odds_a]], float))[0]
        has = np.isfinite(q).all()
        fin = (1 - w_model) * q + w_model * pm if has else pm
        ko = r.kickoff if isinstance(r.kickoff, pd.Timestamp) and not pd.isna(r.kickoff) else None
        out.append({"date": str(r.date.date()), "kickoff_utc": ko.isoformat() if ko is not None else None,
                    "home": r.home, "away": r.away, "started": started, "p": [_r(x) for x in fin], "p_model": [_r(x) for x in pm],
                    "market": [_r(x) for x in q] if has else None, "xg": [_r(p["xg_home"], 2), _r(p["xg_away"], 2)],
                    "b365": [_r(x, 2) for x in (r.b365_h, r.b365_d, r.b365_a)] if not pd.isna(r.b365_h) else None})
    out.sort(key=lambda d: (d["kickoff_utc"] or d["date"], d["home"]))
    return out


# ---------------------------------------------------------------------------
# Seguimiento
# ---------------------------------------------------------------------------
def update_log(settings: Settings, code: str, preds: list[dict], m: pd.DataFrame) -> dict:
    """Guarda la última predicción antes de cada partido y mide los ya jugados."""
    p = settings.predictions_dir / "leagues_log.csv"
    cols = ["league", "date", "home", "away", "ph", "pd", "pa", "hg", "ag"]
    log_df = pd.read_csv(p) if p.exists() else pd.DataFrame(columns=cols)
    now = pd.Timestamp.now(tz="UTC")
    for x in preds:
        ko = pd.Timestamp(x["kickoff_utc"]) if x.get("kickoff_utc") else None
        if x.get("started") or (ko is not None and ko <= now):
            continue
        mask = (log_df.league == code) & (log_df.home == x["home"]) & (log_df.away == x["away"]) & (log_df.date == x["date"])
        log_df = log_df[~mask]
        log_df = pd.concat([log_df, pd.DataFrame([{"league": code, "date": x["date"], "home": x["home"], "away": x["away"],
                                                   "ph": x["p"][0], "pd": x["p"][1], "pa": x["p"][2]}])], ignore_index=True)
    res = m[m.hg.notna()].assign(d=lambda d: d.date.dt.strftime("%Y-%m-%d"))
    key = {(h, a, d): (hg, ag) for h, a, d, hg, ag in zip(res.home, res.away, res.d, res.hg, res.ag)}
    for i, r in log_df[(log_df.league == code) & log_df.hg.isna()].iterrows():
        k = (r.home, r.away, r.date)
        if k in key:
            log_df.loc[i, "hg"], log_df.loc[i, "ag"] = key[k]
    log_df.to_csv(p, index=False)
    done = log_df[(log_df.league == code) & log_df.hg.notna()]
    if done.empty:
        return {"n": 0}
    y = np.where(done.hg > done.ag, 0, np.where(done.hg == done.ag, 1, 2))
    P = done[["ph", "pd", "pa"]].to_numpy(float)
    return {"n": int(len(done)), "accuracy": _r((P.argmax(1) == y).mean(), 3),
            "log_loss": _r(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-9, 1))), 4)}


# ---------------------------------------------------------------------------
# Copas, Europa y Liga F (Elo)
# ---------------------------------------------------------------------------
def espn_upcoming(lg: str, days: int, online: bool) -> list[dict]:
    if not online:
        return []
    out = []
    for k in range(-1, days + 1):
        d = date.today() + timedelta(days=k)
        try:
            js = http.get(ESPN_SB.format(lg=lg), params={"dates": f"{d:%Y%m%d}"}, timeout=20, retries=1).json()
        except Exception as e:
            log.warning("ESPN %s %s: %s", lg, d, e)
            if isinstance(e, http.HostUnavailable):
                break
            continue
        for ev in js.get("events", []):
            comp = ev["competitions"][0]
            st = comp.get("status", {}).get("type", {})
            side = {c["homeAway"]: c for c in comp["competitors"]}
            out.append({"id": str(ev.get("id")), "date": str(d), "kickoff_utc": ev.get("date"),
                        "home": side["home"]["team"]["displayName"], "away": side["away"]["team"]["displayName"],
                        "state": st.get("state"),
                        "score": [side["home"].get("score"), side["away"].get("score")] if st.get("state") != "pre" else None})
    return out


SPANISH = {"real madrid", "barcelona", "atletico madrid", "atletico de madrid", "athletic club", "real sociedad",
           "villarreal", "real betis", "betis", "sevilla", "valencia", "celta vigo", "rayo vallecano", "girona",
           "osasuna", "getafe", "mallorca", "las palmas", "espanyol", "alaves", "leganes", "real valladolid", "levante",
           "elche", "real oviedo", "deportivo la coruna", "racing santander", "malaga"}


def cups(settings: Settings, online: bool = True, days: int = 8) -> dict:
    from .eurelo import Lookup, load as load_elo
    from .quiniela import elo_probs, goals_from_probs
    try:
        lk = Lookup(load_elo(settings))
    except Exception as e:
        log.warning("Elo europeo no disponible: %s", e)
        return {}
    import unicodedata

    def plain(s):
        return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower().strip()
    out = {}
    for lg, name in CUP_LEAGUES.items():
        women = lg == "esp.w.1"
        rows = []
        for ev in espn_upcoming(lg, days, online):
            if lg.startswith("uefa") and not ({plain(ev["home"]), plain(ev["away"])} & SPANISH):
                continue
            fh, fa = lk.find(ev["home"], women=women), lk.find(ev["away"], women=women)
            p = elo_probs(fh[1], fa[1]) if fh and fa else None
            xg = goals_from_probs(p) if p else None
            rows.append({**ev, "p": [_r(x) for x in p] if p else None, "xg": [_r(x, 2) for x in xg] if xg else None,
                         "elo": [_r(fh[1], 0) if fh else None, _r(fa[1], 0) if fa else None]})
        out[lg] = {"name": name, "matches": rows}
    return out


def liga_f_table(settings: Settings) -> dict | None:
    """Clasificación y proyección de la Liga F con resultados de ESPN y el Elo femenino."""
    from .eurelo import Lookup, load as load_elo
    from .quiniela import elo_probs, goals_from_probs
    from .site_export import simulate_rest
    cur = current_season()
    p = settings.raw_dir / "euro" / f"W_{cur}.csv"
    if not p.exists():
        return None
    m = pd.read_csv(p, parse_dates=["date"]).assign(season=cur)
    if m.empty:
        return None
    table = standings(m, cur)
    lk = Lookup(load_elo(settings))
    rem = remaining_pairs(m, cur)
    lam, mu = [], []
    for h, a in zip(rem.home, rem.away):
        fh, fa = lk.find(h, women=True), lk.find(a, women=True)
        pr = elo_probs(fh[1], fa[1]) if fh and fa else [0.42, 0.26, 0.32]
        x = goals_from_probs(pr)
        lam.append(x[0])
        mu.append(x[1])
    sim = simulate_rest(table, rem, np.array(lam), np.array(mu), 4000 if len(rem) else 1, 0.0, seed=9)
    k = len(sim["teams"])
    zones = {"campeón": (1, 1), "champions": (1, 3), "descenso": (k - 1, k)}
    proj = []
    for i, t in enumerate(sim["teams"]):
        r = sim["ranks"][:, i]
        proj.append({"team": t, **{f"p_{z}": _r(((r >= lo) & (r <= hi)).mean()) for z, (lo, hi) in zones.items()},
                     "exp_pts": _r(sim["pts"][:, i].mean(), 1), "exp_pos": _r(r.mean(), 1)})
    proj.sort(key=lambda d: d["exp_pos"])
    return {"name": "Liga F", "table": table.drop(columns=["home_pts", "home_pj", "away_pts", "away_pj"], errors="ignore").to_dict(orient="records"),
            "projection": proj, "remaining": int(len(rem)), "zones": {k_: list(v) for k_, v in zones.items()}}


# ---------------------------------------------------------------------------
def build(settings: Settings, online: bool = True, with_eval: bool = False) -> dict:
    cur = current_season()
    path = settings.predictions_dir / "leagues.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    ev_path = settings.state_dir / "leagues_eval.json"
    evals = json.loads(ev_path.read_text(encoding="utf-8")) if ev_path.exists() else {}
    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="minutes"), "leagues": {}}
    for code, meta in LEAGUES.items():
        try:
            m = load_league(settings, code, online)
            if m.empty:
                continue
            if with_eval or code not in evals:
                try:
                    evals[code] = evaluate(settings, m, cur - 1)
                except Exception as e:
                    log.warning("Evaluación %s: %s", code, e)
            w = max(0.1, (evals.get(code) or {}).get("w_model", 0.2))  # como en LaLiga, mínimo 10 % de modelo
            preds = upcoming(settings, m, cur, w)
            track = update_log(settings, code, preds, m)
            out["leagues"][code] = {"name": meta["name"], "espn": meta["espn"], **project(settings, m, cur, meta["zones"]),
                                    "upcoming": preds, "eval": evals.get(code), "track": track}
            log.info("%s: %d próximos partidos", meta["name"], len(preds))
        except Exception as e:
            log.warning("Liga %s no disponible: %s", code, e)
            if code in (old.get("leagues") or {}):
                out["leagues"][code] = old["leagues"][code]
    ev_path.write_text(json.dumps(evals, indent=1), encoding="utf-8")
    try:
        lf = liga_f_table(settings)
        if lf:
            out["leagues"]["LF"] = {**lf, "espn": "esp.w.1"}
    except Exception as e:
        log.warning("Liga F: %s", e)
    try:
        out["cups"] = cups(settings, online)
    except Exception as e:
        log.warning("Copas: %s", e)
        out["cups"] = old.get("cups", {})
    path.write_text(json.dumps(out, ensure_ascii=False, default=str), encoding="utf-8")
    return out
