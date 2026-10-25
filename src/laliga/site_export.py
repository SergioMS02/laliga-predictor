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


def project_scorers(settings: Settings, matches: pd.DataFrame, season: int, players: pd.DataFrame | None,
                    n: int = N_SIMS, seed: int = 11, top: int = 12) -> dict | None:
    """Pichichi: goles finales de cada jugador = goles actuales + goles en lo que queda.

    Los goles que quedan siguen una Poisson con media = goles esperados de su equipo en sus
    partidos restantes (Dixon-Coles) × la parte del xG del equipo que produce el jugador
    (temporada actual, completada con la anterior si aún hay pocos partidos)."""
    if players is None or players.empty:
        return None
    s = matches[matches.season == season]
    rem = s[s.hg.isna()]
    cur = players[players.season == season]
    if cur.empty:
        return None
    teams = sorted(set(s.home) | set(s.away))
    ref = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
    dc = DixonColes.from_settings(settings).fit(matches, ref, teams, pipeline._new_teams(matches, season))
    team_left = {t: 0.0 for t in teams}
    for r in rem.itertuples():
        lam, mu = dc.expected_goals(r.home, r.away)
        team_left[r.home] = team_left.get(r.home, 0) + lam
        team_left[r.away] = team_left.get(r.away, 0) + mu
    prev = players[players.season == season - 1]
    rows = []
    for t, g in cur.groupby("team"):
        txg = g.xg.sum()
        games = max(int(g.games.max() or 0), 1)
        w_cur = min(1.0, games / 15)  # con pocas jornadas pesa también la temporada pasada
        pv = prev[prev.team == t]
        pv_share = (pv.set_index("player").xg / pv.xg.sum()) if len(pv) and pv.xg.sum() > 0 else pd.Series(dtype=float)
        for r in g.itertuples():
            share = (r.xg / txg) if txg > 0 else 0.0
            if r.player in pv_share.index:
                share = w_cur * share + (1 - w_cur) * float(pv_share[r.player])
            rows.append({"name": r.player, "team": t, "goals": int(r.goals or 0), "share": share,
                         "left": team_left.get(t, 0.0) * share})
    df = pd.DataFrame(rows)
    if df.empty:
        return None
    df["exp"] = df.goals + df.left
    cand = df.sort_values("exp", ascending=False).head(40).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    sims = cand.goals.to_numpy()[None, :] + rng.poisson(cand.left.to_numpy()[None, :], size=(n, len(cand)))
    mx = sims.max(1, keepdims=True)
    win = (sims == mx)
    p_top = (win / win.sum(1, keepdims=True)).mean(0)  # empates repartidos
    cand["p"] = p_top
    cand["p10"] = np.percentile(sims, 10, axis=0)
    cand["p90"] = np.percentile(sims, 90, axis=0)
    out = cand.sort_values(["p", "exp"], ascending=False).head(top)
    return {"n_sims": int(n), "players": [
        {"name": r.name, "team": r.team, "goals": int(r.goals), "exp": _r(r.exp, 1), "p": _r(r.p, 4),
         "range": [int(r.p10), int(r.p90)]} for r in out.itertuples()]}


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
def _bins(p: np.ndarray, y: np.ndarray, edges=(0, .1, .2, .3, .4, .5, .6, .7, .8, 1.01)) -> list[dict]:
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() >= 5:
            out.append({"pred": _r(p[m].mean(), 3), "obs": _r(y[m].mean(), 3), "n": int(m.sum())})
    return out


def track_props(settings: Settings, matches: pd.DataFrame) -> dict:
    """¿Aciertan los mercados secundarios? Goles (+2,5, ambos marcan), goleadores, tarjetas y córners.

    Se usa la última predicción antes de cada partido ya jugado."""
    p = settings.predictions_dir / "props_log.csv"
    if not p.exists():
        return {}
    lg = pd.read_csv(p)
    lg = lg.sort_values("generated_at").drop_duplicates(["season", "home", "away"], keep="last")
    res = matches[matches.hg.notna()][["season", "home", "away", "date", "hg", "ag", "h_yellow", "a_yellow",
                                       "h_corners", "a_corners"]]
    j = lg.merge(res, on=["season", "home", "away"], suffixes=("", "_m"))
    if j.empty:
        return {"n": 0}
    out = {"n": int(len(j))}
    tot = (j.hg + j.ag).to_numpy(float)
    for key, pcol, yv in (("over25", "p_over25", (tot > 2.5).astype(float)),
                          ("btts", "p_btts", ((j.hg > 0) & (j.ag > 0)).astype(float).to_numpy())):
        ok = j[pcol].notna().to_numpy()
        if ok.sum():
            pp = np.clip(j[pcol].to_numpy(float)[ok], 1e-6, 1 - 1e-6)
            yy = np.asarray(yv)[ok]
            base = yy.mean()
            out[key] = {"n": int(ok.sum()), "pred_mean": _r(pp.mean(), 3), "obs": _r(base, 3),
                        "brier": _r(float(np.mean((pp - yy) ** 2)), 4),
                        "acierto": _r(float(np.mean((pp > 0.5) == (yy > 0.5))), 3), "bins": _bins(pp, yy)}
    out["goals"] = {"pred": _r(float((j.xg_home + j.xg_away).mean()), 2), "obs": _r(float(tot.mean()), 2),
                    "mae": _r(float(np.abs(j.xg_home + j.xg_away - tot).mean()), 2)}
    for key, col, a, b in (("cards", "yellow_total", "h_yellow", "a_yellow"), ("corners", "corners_total", "h_corners", "a_corners")):
        ok = j[col].notna() & j[a].notna() & j[b].notna()
        if ok.sum():
            real = (j.loc[ok, a] + j.loc[ok, b]).to_numpy(float)
            pr = j.loc[ok, col].to_numpy(float)
            out[key] = {"n": int(ok.sum()), "pred": _r(pr.mean(), 2), "obs": _r(real.mean(), 2),
                        "mae": _r(float(np.abs(pr - real).mean()), 2)}
    # goleadores: ¿marcó cada jugador al que se le dio una probabilidad?
    try:
        from .data.suspensions import plain
        cur = results.current_season()
        evf = settings.raw_dir / f"events_esp.1_{cur}.csv"
        ev = pd.read_csv(evf, parse_dates=["date"]) if evf.exists() else pd.DataFrame()
        pts, ys, tops = [], [], []
        if len(ev):
            goals = ev[ev.card == "goal"]
            for r in j.itertuples():
                d = pd.Timestamp(r.date_m if hasattr(r, "date_m") else r.date)
                g = goals[((goals.date - d).abs() <= pd.Timedelta(days=1)) & goals.team.isin([r.home, r.away])]
                if (r.hg + r.ag) > 0 and g.empty:
                    continue  # aún sin datos de goleadores de este partido
                scorers = {plain(x) for x in g.player}
                for side, name, pr in json.loads(r.scorers or "[]"):
                    hit = any(set(plain(name).split()) & set(s.split()) and
                              (plain(name).split()[-1] in s or s.split()[-1] in plain(name)) for s in scorers)
                    pts.append(pr)
                    ys.append(float(hit))
        if pts:
            pp, yy = np.array(pts), np.array(ys)
            out["scorers"] = {"n": int(len(pp)), "pred_mean": _r(pp.mean(), 3), "obs": _r(yy.mean(), 3),
                              "expected_goals": _r(pp.sum(), 1), "actual": int(yy.sum()), "bins": _bins(pp, yy)}
    except Exception as e:
        log.warning("Seguimiento de goleadores no disponible: %s", e)
    return out


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
        try:
            out["scorers"] = project_scorers(settings, matches, season, players)
        except Exception as e:  # nunca debe tumbar la web
            log.warning("Pichichi no disponible: %s", e)
            out["scorers"] = old.get("scorers")
        hist = record_projection_history(settings, out["projection"])
        out["projection_history"] = hist[-60:]
    else:
        out["projection"] = old.get("projection")
        out["scorers"] = old.get("scorers")
        out["projection_history"] = old.get("projection_history", [])
    if preds is None:
        lp = settings.predictions_dir / "latest.json"
        preds = json.loads(lp.read_text(encoding="utf-8")) if lp.exists() else []
    out["h2h"] = {f"{p['home']}|{p['away']}": head_to_head(matches, p["home"], p["away"]) for p in preds}
    out["track"] = track_record(settings, matches)
    try:
        out["track_props"] = track_props(settings, matches)
    except Exception as e:
        log.warning("Seguimiento de mercados secundarios no disponible: %s", e)
    if out.get("scorers"):  # evolución del Pichichi día a día
        hist = old.get("scorers_history", [])
        today = datetime.now(timezone.utc).date().isoformat()
        snap = {"date": today, "p": {x["name"]: x["p"] for x in out["scorers"]["players"][:6]}}
        hist = [h for h in hist if h["date"] != today] + [snap]
        out["scorers_history"] = hist[-120:]
    else:
        out["scorers_history"] = old.get("scorers_history", [])
    path.write_text(json.dumps(out, ensure_ascii=False, default=_json_default), encoding="utf-8")
    return out
