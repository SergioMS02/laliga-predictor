"""Datos extra para la web: clasificación y proyección de la temporada, estadísticas de
equipos, historial directo, jugadores clave, ranking Elo y aciertos del modelo.

Escribe predictions/site_extra.json (y añade una línea diaria a projection_history.jsonl).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import metrics, pipeline
from .config import Settings
from .data import results
from .features.sequential import build_sequential_features
from .models.dixon_coles import DixonColes

log = logging.getLogger(__name__)

N_SIMS = 10000
ZONES = {"title": (1, 1), "ucl": (1, 4), "europe": (1, 7), "releg": (18, 20)}


def _r(x, n=2):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(x) else round(x, n)


# ---------------------------------------------------------------------------
# Clasificación
# ---------------------------------------------------------------------------
def standings(matches: pd.DataFrame, season: int) -> pd.DataFrame:
    s = matches[matches.season == season]
    teams = sorted(set(s.home) | set(s.away))
    played = s[s.hg.notna()].sort_values("date")
    rows = {t: {"team": t, "pj": 0, "g": 0, "e": 0, "p": 0, "gf": 0, "gc": 0, "form": [],
                "home_pts": 0, "home_pj": 0, "away_pts": 0, "away_pj": 0} for t in teams}
    for r in played.itertuples():
        for t, gf, ga, home in ((r.home, r.hg, r.ag, True), (r.away, r.ag, r.hg, False)):
            d = rows[t]
            d["pj"] += 1
            d["gf"] += int(gf)
            d["gc"] += int(ga)
            res = "V" if gf > ga else "E" if gf == ga else "D"
            d[{"V": "g", "E": "e", "D": "p"}[res]] += 1
            pts = {"V": 3, "E": 1, "D": 0}[res]
            d["home_pts" if home else "away_pts"] += pts
            d["home_pj" if home else "away_pj"] += 1
            d["form"].append(res)
    df = pd.DataFrame(rows.values())
    df["pts"] = 3 * df.g + df.e
    df["dg"] = df.gf - df.gc
    df["form"] = df.form.map(lambda f: "".join(f[-5:]))
    df = df.sort_values(["pts", "dg", "gf", "team"], ascending=[False, False, False, True]).reset_index(drop=True)
    df["pos"] = np.arange(1, len(df) + 1)
    return df


# ---------------------------------------------------------------------------
# Proyección de la temporada (Monte Carlo con Dixon-Coles)
# ---------------------------------------------------------------------------
def project_season(settings: Settings, matches: pd.DataFrame, season: int, table: pd.DataFrame,
                   n: int = N_SIMS, seed: int = 7) -> dict:
    s = matches[matches.season == season]
    rem = s[s.hg.isna()]
    teams = list(table.team)
    idx = {t: i for i, t in enumerate(teams)}
    k = len(teams)
    base_pts = table.set_index("team").pts.reindex(teams).to_numpy(float)
    base_gd = table.set_index("team").dg.reindex(teams).to_numpy(float)
    base_gf = table.set_index("team").gf.reindex(teams).to_numpy(float)
    if rem.empty:
        sim_pts = np.tile(base_pts, (1, 1))
        ranks = np.argsort(np.argsort(-(base_pts * 1e6 + base_gd * 1e3 + base_gf)))[None, :] + 1
    else:
        ref = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
        dc = DixonColes.from_settings(settings).fit(matches, ref, teams, pipeline._new_teams(matches, season))
        lam = np.array([dc.expected_goals(r.home, r.away)[0] for r in rem.itertuples()])
        mu = np.array([dc.expected_goals(r.home, r.away)[1] for r in rem.itertuples()])
        hi = rem.home.map(idx).to_numpy()
        ai = rem.away.map(idx).to_numpy()
        rng = np.random.default_rng(seed)
        hg = rng.poisson(lam, size=(n, len(rem)))
        ag = rng.poisson(mu, size=(n, len(rem)))
        ph = np.where(hg > ag, 3, np.where(hg == ag, 1, 0))
        pa = np.where(ag > hg, 3, np.where(hg == ag, 1, 0))
        sim_pts = np.tile(base_pts, (n, 1))
        sim_gd = np.tile(base_gd, (n, 1))
        sim_gf = np.tile(base_gf, (n, 1))
        for j in range(len(rem)):  # acumulación por partido (columnas), vectorizada en simulaciones
            sim_pts[:, hi[j]] += ph[:, j]
            sim_pts[:, ai[j]] += pa[:, j]
            sim_gd[:, hi[j]] += hg[:, j] - ag[:, j]
            sim_gd[:, ai[j]] += ag[:, j] - hg[:, j]
            sim_gf[:, hi[j]] += hg[:, j]
            sim_gf[:, ai[j]] += ag[:, j]
        # desempate aproximado: puntos, diferencia de goles, goles a favor (LaLiga usa el golaveraje particular)
        key = sim_pts * 1e6 + (sim_gd + 500) * 1e3 + sim_gf + rng.random(sim_pts.shape) * 1e-3
        ranks = np.argsort(np.argsort(-key, axis=1), axis=1) + 1
    nsim = ranks.shape[0]
    out = []
    for t in teams:
        i = idx[t]
        r = ranks[:, i]
        pts = sim_pts[:, i]
        out.append({
            "team": t,
            **{f"p_{z}": _r(((r >= lo) & (r <= hi_)).mean(), 4) for z, (lo, hi_) in ZONES.items()},
            "exp_pts": _r(pts.mean(), 1), "pts_p10": _r(np.percentile(pts, 10), 0), "pts_p90": _r(np.percentile(pts, 90), 0),
            "exp_pos": _r(r.mean(), 1),
            "pos_probs": [_r((r == p).mean(), 4) for p in range(1, k + 1)],
        })
    out.sort(key=lambda d: d["exp_pos"])
    return {"n_sims": int(nsim), "remaining": int(len(rem)), "teams": out}


def record_projection_history(settings: Settings, proj: dict) -> list[dict]:
    p = settings.predictions_dir / "projection_history.jsonl"
    rows = []
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    today = datetime.now(timezone.utc).date().isoformat()
    snap = {"date": today, "teams": {t["team"]: {k: t[k] for k in ("p_title", "p_ucl", "p_europe", "p_releg", "exp_pts")}
                                     for t in proj["teams"]}}
    rows = [r for r in rows if r.get("date") != today] + [snap]
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    return rows


# ---------------------------------------------------------------------------
# Equipos
# ---------------------------------------------------------------------------
def team_stats(matches: pd.DataFrame, season: int, table: pd.DataFrame, seq: pd.DataFrame,
               players: pd.DataFrame | None) -> dict:
    s = matches[matches.season == season]
    played = s[s.hg.notna()].sort_values("date")
    out = {}
    tj = table.set_index("team")
    for t in table.team:
        tm = played[(played.home == t) | (played.away == t)]
        is_home = tm.home == t
        gf = np.where(is_home, tm.hg, tm.ag)
        ga = np.where(is_home, tm.ag, tm.hg)

        def per_game(hcol, acol):
            if hcol not in tm or tm[hcol].isna().all():
                return None, None
            f = np.where(is_home, tm[hcol], tm[acol]).astype(float)
            a = np.where(is_home, tm[acol], tm[hcol]).astype(float)
            ok = ~np.isnan(f) & ~np.isnan(a)
            return (_r(f[ok].mean()) if ok.any() else None, _r(a[ok].mean()) if ok.any() else None)
        xgf, xga = per_game("h_xg", "a_xg")
        shf, sha = per_game("h_shots", "a_shots")
        sotf, sota = per_game("h_sot", "a_sot")
        last = []
        for r in tm.tail(5).itertuples():
            h = r.home == t
            g1, g2 = (int(r.hg), int(r.ag)) if h else (int(r.ag), int(r.hg))
            last.append({"date": str(r.date.date()), "opp": r.away if h else r.home, "venue": "C" if h else "F",
                         "gf": g1, "ga": g2, "res": "V" if g1 > g2 else "E" if g1 == g2 else "D"})
        row = tj.loc[t]
        n = max(int(row.pj), 1)
        out[t] = {
            "pos": int(row.pos), "pts": int(row.pts), "pj": int(row.pj), "g": int(row.g), "e": int(row.e), "p": int(row.p),
            "gf_pg": _r(gf.mean()) if len(gf) else None, "ga_pg": _r(ga.mean()) if len(ga) else None,
            "xg_pg": xgf, "xga_pg": xga, "shots_pg": shf, "shots_against_pg": sha, "sot_pg": sotf, "sot_against_pg": sota,
            "clean_sheets": int((ga == 0).sum()) if len(ga) else 0,
            "home_ppg": _r(row.home_pts / row.home_pj) if row.home_pj else None,
            "away_ppg": _r(row.away_pts / row.away_pj) if row.away_pj else None,
            "ppg": _r(row.pts / n),
            "last5": last,
        }
    # Elo actual e historial (pre-partido) de las dos últimas temporadas
    m2 = matches.join(seq[["elo_home", "elo_away"]])
    recent = m2[m2.season >= season - 1].sort_values("date")
    for t in out:
        hist = []
        cur = None
        for r in recent[(recent.home == t) | (recent.away == t)].itertuples():
            e = r.elo_home if r.home == t else r.elo_away
            if pd.isna(r.hg):
                cur = e if cur is None else cur
                if cur is not None:
                    break
                continue
            hist.append([str(r.date.date()), round(float(e), 1)])
        if cur is None and hist:
            cur = hist[-1][1]
        if cur is not None:
            hist.append(["ahora", round(float(cur), 1)])
        out[t]["elo"] = None if cur is None else round(float(cur), 1)
        out[t]["elo_hist"] = hist[-60:]
        prev5 = [h[1] for h in hist[-6:-1]]
        out[t]["elo_change5"] = round(float(cur) - prev5[0], 1) if prev5 and cur is not None else None
    # jugadores clave (Understat): temporada actual, o la anterior si aún no hay datos
    if players is not None and len(players):
        for t in out:
            pl = players[(players.team == t) & (players.season == season)]
            src = season
            if pl.empty or pl.minutes.sum() < 900:
                pl = players[(players.team == t) & (players.season == season - 1)]
                src = season - 1
            if pl.empty:
                continue
            pl = pl.assign(contrib=pl.xg.fillna(0) + pl.xa.fillna(0)).sort_values("contrib", ascending=False).head(6)
            out[t]["key_players"] = {"season": int(src), "players": [
                {"name": r.player, "pos": r.position, "games": _r(r.games, 0), "minutes": _r(r.minutes, 0),
                 "goals": _r(r.goals, 0) if "goals" in pl else None, "assists": _r(r.assists, 0) if "assists" in pl else None,
                 "xg": _r(r.xg, 1), "xa": _r(r.xa, 1),
                 "xgxa90": _r((r.xg + r.xa) / r.minutes * 90, 2) if r.minutes else None}
                for r in pl.itertuples()]}
    return out


def head_to_head(matches: pd.DataFrame, home: str, away: str, n: int = 6) -> list[dict]:
    m = matches[matches.hg.notna() & (((matches.home == home) & (matches.away == away)) |
                                     ((matches.home == away) & (matches.away == home)))].sort_values("date")
    return [{"date": str(r.date.date()), "home": r.home, "away": r.away, "hg": int(r.hg), "ag": int(r.ag)}
            for r in m.tail(n).itertuples()][::-1]


# ---------------------------------------------------------------------------
# Aciertos del modelo
# ---------------------------------------------------------------------------
def track_record(settings: Settings, matches: pd.DataFrame) -> dict:
    p = settings.predictions_dir / "log.csv"
    if not p.exists():
        return {"n": 0, "matches": [], "weeks": []}
    lg = pd.read_csv(p)
    if lg.empty:
        return {"n": 0, "matches": [], "weeks": []}
    lg["date"] = pd.to_datetime(lg["date"])
    lg["gen"] = pd.to_datetime(lg["generated_at"], utc=True)
    res = matches[matches.hg.notna()][["season", "home", "away", "date", "hg", "ag", "result", "kickoff"]]
    j = lg.merge(res, on=["season", "home", "away"], suffixes=("", "_m"))
    if j.empty:
        return {"n": 0, "matches": [], "weeks": []}
    # última predicción emitida antes del inicio (o antes del día del partido si no hay hora)
    ko = pd.to_datetime(j.kickoff, utc=True)
    limit = ko.fillna(j.date_m.dt.tz_localize("Europe/Madrid").dt.tz_convert("UTC") + pd.Timedelta(hours=12))
    j = j[j.gen <= limit].sort_values("gen").drop_duplicates(["season", "home", "away"], keep="last")
    if j.empty:
        return {"n": 0, "matches": [], "weeks": []}
    P = j[["p_home", "p_draw", "p_away"]].to_numpy(float)
    y = j.result.map(metrics.RESULT_IDX).to_numpy()
    j = j.assign(pick=P.argmax(axis=1), y=y, ll=-np.log(np.clip(P[np.arange(len(y)), y], 1e-9, 1)),
                 p_real=P[np.arange(len(y)), y])
    j["correct"] = j.pick == j.y
    j["week"] = (j.date_m - pd.to_timedelta(j.date_m.dt.weekday, unit="D")).dt.date.astype(str)
    weeks = (j.groupby("week").agg(n=("y", "size"), correct=("correct", "sum"), ll=("ll", "mean"))
             .reset_index().sort_values("week"))
    weeks["cum_acc"] = (weeks.correct.cumsum() / weeks.n.cumsum()).round(4)
    lab = ["1", "X", "2"]
    rows = [{"date": str(r.date_m.date()), "home": r.home, "away": r.away, "score": f"{int(r.hg)}-{int(r.ag)}",
             "p": [_r(r.p_home, 3), _r(r.p_draw, 3), _r(r.p_away, 3)], "pick": lab[r.pick], "result": lab[r.y],
             "correct": bool(r.correct), "p_real": _r(r.p_real, 3), "lineups": bool(str(r.lineup_confirmed).lower() == "true")}
            for r in j.sort_values("date_m", ascending=False).head(120).itertuples()]
    return {"n": int(len(j)), "accuracy": _r(j.correct.mean(), 4), "log_loss": _r(j.ll.mean(), 4),
            "matches": rows, "weeks": [{"week": w.week, "n": int(w.n), "correct": int(w.correct), "ll": _r(w.ll, 4),
                                        "cum_acc": _r(w.cum_acc, 4)} for w in weeks.itertuples()]}


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    return str(o)


# ---------------------------------------------------------------------------
def export(settings: Settings, matches: pd.DataFrame, preds: list[dict] | None = None,
           with_projection: bool = True) -> dict:
    season = results.current_season()
    matches = matches.reset_index(drop=True)
    table = standings(matches, season)
    seq = build_sequential_features(matches, settings["elo"])
    players = pipeline.load_extras(settings, online=False).get("players")
    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="minutes"), "season": season,
           "table": table.drop(columns=["home_pts", "home_pj", "away_pts", "away_pj"]).to_dict(orient="records"),
           "teams": team_stats(matches, season, table, seq, players)}
    path = settings.predictions_dir / "site_extra.json"
    old = {}
    if path.exists():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            old = {}
    if with_projection:
        out["projection"] = project_season(settings, matches, season, table)
        hist = record_projection_history(settings, out["projection"])
        out["projection_history"] = hist[-60:]
    else:
        out["projection"] = old.get("projection")
        out["projection_history"] = old.get("projection_history", [])
    if preds is None:
        lp = settings.predictions_dir / "latest.json"
        preds = json.loads(lp.read_text(encoding="utf-8")) if lp.exists() else []
    out["h2h"] = {f"{p['home']}|{p['away']}": head_to_head(matches, p["home"], p["away"]) for p in preds}
    out["track"] = track_record(settings, matches)
    path.write_text(json.dumps(out, ensure_ascii=False, default=_json_default), encoding="utf-8")
    return out
