"""Tareas sobre histórico: rellenar alineaciones/árbitros desde ESPN, calibrar el
efecto de las bajas y evaluar las predicciones reales que ya se hicieron."""
from __future__ import annotations

import json
import logging
import time
from datetime import date

import numpy as np
import pandas as pd

from . import metrics
from .config import Settings
from .data import espn, other_comps
from .pipeline import load_extras
from .features import availability as av

log = logging.getLogger(__name__)


def backfill_espn(settings: Settings, seasons: list[int], pause: float = 0.25, other: bool = True) -> dict:
    done = set(espn.load_lineups(settings).event_id.astype(str))
    stats = {"events": 0, "lineups": 0, "referees": 0}
    for s in seasons:
        sb = espn.scoreboard(settings, date(s, 8, 1), min(date(s + 1, 6, 30), date.today()))
        if sb.empty:
            continue
        sb = sb[sb.completed]
        refs, lus = [], []
        for r in sb.itertuples():
            if r.event_id in done:
                continue
            try:
                det = espn.summary(settings, r.event_id)
            except Exception as e:
                log.warning("summary %s: %s", r.event_id, e)
                continue
            stats["events"] += 1
            if det.referee:
                refs.append({"season": s, "home": r.home, "away": r.away, "referee": det.referee})
            if det.lineup_confirmed:
                lus.append(espn.lineup_rows(det, r.kickoff.tz_convert("Europe/Madrid").tz_localize(None), r.home, r.away))
            time.sleep(pause)
        if refs:
            espn.save_referees(settings, pd.DataFrame(refs))
            stats["referees"] += len(refs)
        if lus:
            espn.save_lineups(settings, pd.concat(lus, ignore_index=True))
            stats["lineups"] += len(lus)
        # calendario de Champions/Europa/Conference/Copa de esa temporada
        if not other:
            log.info("Temporada %s: %s", s, stats)
            continue
        try:
            oth = other_comps.espn_other_matches(settings, date(s, 7, 1), min(date(s + 1, 6, 30), date.today()))
            other_comps.save_espn_other(settings, oth)
            stats["other_comp_dates"] = stats.get("other_comp_dates", 0) + len(oth)
        except Exception as e:
            log.warning("Otras competiciones %s: %s", s, e)
        log.info("Temporada %s: %s", s, stats)
    return stats


def availability_dataset(settings: Settings, matches: pd.DataFrame, oos: pd.DataFrame,
                         lineups: pd.DataFrame | None = None, min_history: int = 5, keep_keys: bool = False) -> pd.DataFrame:
    """Une λ/μ del modelo base (fuera de muestra) con el déficit real del once."""
    lineups = espn.load_lineups(settings) if lineups is None else lineups
    if lineups.empty:
        return pd.DataFrame()
    lookback = int(settings.get("availability", "lookback_matches", 10))
    players = load_extras(settings, online=False).get("players")
    ev = lineups.drop_duplicates(["event_id", "team"])[["event_id", "date", "team"]]
    rows = []
    o = oos[oos.hg.notna()].set_index(["season", "home", "away"])
    for r in matches[matches.hg.notna()].itertuples():
        key = (r.season, r.home, r.away)
        if key not in o.index:
            continue
        evs = ev[(ev.date == r.date) & ev.team.isin([r.home, r.away])]
        if evs.team.nunique() < 2:
            continue
        rec = {"lam": o.loc[key, "dc_lam"], "mu": o.loc[key, "dc_mu"], "hg": r.hg, "ag": r.ag}
        if keep_keys:
            rec.update(season=r.season, home=r.home, away=r.away)
        ok = True
        for side, team in (("home", r.home), ("away", r.away)):
            imp = av.player_importance(lineups, team, r.date, lookback)
            n_hist = lineups[(lineups.team == team) & (lineups.date < r.date)].event_id.nunique()
            if n_hist < min_history:
                ok = False
                break
            pl = None
            if players is not None and len(players):
                # producción de la temporada anterior (conocida antes del partido)
                pl = players[(players.team == team) & (players.season == r.season - 1)]
            imp = av.attach_production(imp, pl)
            e_id = evs[evs.team == team].event_id.iloc[0]
            starters = lineups[(lineups.event_id == e_id) & (lineups.team == team) & lineups.starter.astype(bool)]
            datt, ddef = av.deficits_from_lineup(imp, list(starters.player_id.astype(str)))
            rec[f"datt_{side}"], rec[f"ddef_{side}"] = datt, ddef
            rec[f"d_{side}"] = (datt + ddef) / 2
        if ok:
            rows.append(rec)
    return pd.DataFrame(rows)


def calibrate_availability(settings: Settings, matches: pd.DataFrame) -> dict:
    oos = pd.read_csv(settings.processed_dir / "oos_table.csv", parse_dates=["date"])
    data = availability_dataset(settings, matches, oos)
    if len(data) < 150:
        raise RuntimeError(f"Solo hay {len(data)} partidos con alineaciones; ejecuta antes `laliga backfill-espn`.")
    prior = (float(settings.get("availability", "attack_coef")), float(settings.get("availability", "defence_coef")))
    a, b = av.calibrate(data, prior=prior)
    p = settings.state_dir / "tuned.json"
    t = json.loads(p.read_text()) if p.exists() else {}
    t.setdefault("availability", {}).update({"attack_coef": round(a, 4), "defence_coef": round(b, 4), "n_matches": len(data)})
    p.write_text(json.dumps(t, indent=2))
    return t["availability"]


def lineup_market_weight(settings: Settings, matches: pd.DataFrame, data: pd.DataFrame | None = None) -> dict:
    """¿Cuánto hay que mover la probabilidad de las casas cuando se conoce el once?

    Para cada partido con alineaciones se calcula el efecto del once en el modelo de goles
    (1X2 con los goles esperados ajustados por el déficit real frente a los mismos sin ajustar) y
    se estima w en  p ∝ p_casas · (p_ajustada / p_base)^w  por máxima verosimilitud, con las cuotas
    previas (publicadas antes de conocerse el once) y con las de cierre (ya lo conocen).
    w ≈ 1: el mercado previo no sabe nada del once; w ≈ 0: ya lo descuenta."""
    from scipy.optimize import minimize_scalar
    from .models.dixon_coles import score_matrix
    oos = pd.read_csv(settings.processed_dir / "oos_table.csv", parse_dates=["date"])
    if data is None:
        data = availability_dataset(settings, matches, oos, keep_keys=True)
    if data is None or len(data) < 150:
        raise RuntimeError(f"Solo hay {0 if data is None else len(data)} partidos con alineaciones")
    a = float(settings.get("availability", "attack_coef", 0.6))
    b = float(settings.get("availability", "defence_coef", 0.35))
    o = oos.set_index(["season", "home", "away"])
    rows = []
    for r in data.itertuples():
        k = (r.season, r.home, r.away)
        if k not in o.index:
            continue
        rr = o.loc[k]
        if isinstance(rr, pd.DataFrame):
            rr = rr.iloc[0]
        lam2, mu2 = av.adjust_goals(r.lam, r.mu, (r.datt_home, r.ddef_home), (r.datt_away, r.ddef_away), a, b)
        def p1x2(l, m):
            M = score_matrix(l, m, -0.05, 10)
            M = M / M.sum()
            i, j = np.indices(M.shape)
            return np.array([M[i > j].sum(), M[i == j].sum(), M[i < j].sum()])
        ratio = p1x2(lam2, mu2) / p1x2(r.lam, r.mu)
        y = 0 if r.hg > r.ag else 1 if r.hg == r.ag else 2
        rows.append((ratio, np.array([rr.get("mkt_ph"), rr.get("mkt_pd"), rr.get("mkt_pa")], float),
                     np.array([rr.get("mktc_ph"), rr.get("mktc_pd"), rr.get("mktc_pa")], float), y))
    out = {"n": len(rows)}
    for name, idx in (("previas", 1), ("cierre", 2)):
        R = [(x[0], x[idx], x[3]) for x in rows if np.isfinite(x[idx]).all()]
        if len(R) < 100:
            continue
        ratio = np.array([x[0] for x in R]); P = np.array([x[1] for x in R]); y = np.array([x[2] for x in R])

        def nll(w):
            q = P * ratio ** w
            q = q / q.sum(1, keepdims=True)
            return -np.mean(np.log(np.clip(q[np.arange(len(y)), y], 1e-12, 1)))
        res = minimize_scalar(nll, bounds=(0, 2), method="bounded")
        out[name] = {"n": len(R), "w": round(float(res.x), 3), "log_loss_casas": round(nll(0.0), 5),
                     "log_loss_con_once": round(float(res.fun), 5), "log_loss_w1": round(nll(1.0), 5)}
    if "previas" in out:
        p = settings.state_dir / "tuned.json"
        t = json.loads(p.read_text()) if p.exists() else {}
        t.setdefault("availability", {})["market_weight_lineup"] = out["previas"]["w"]
        t["availability"]["market_weight_eval"] = out
        p.write_text(json.dumps(t, indent=2))
    return out


def evaluate_live(settings: Settings, matches: pd.DataFrame) -> dict:
    """Evalúa la última predicción emitida antes de cada partido ya jugado."""
    p = settings.predictions_dir / "log.csv"
    if not p.exists():
        return {"n": 0}
    log_df = pd.read_csv(p)
    log_df["date"] = pd.to_datetime(log_df["date"])
    last = log_df.sort_values("generated_at").drop_duplicates(["season", "home", "away"], keep="last")
    res = matches[matches.result.notna()][["season", "home", "away", "result"]]
    j = last.merge(res, on=["season", "home", "away"])
    if j.empty:
        return {"n": 0}
    y = j.result.map(metrics.RESULT_IDX).to_numpy()
    out = metrics.evaluate(j[["p_home", "p_draw", "p_away"]].to_numpy(float), y)
    out["with_lineups"] = int(j.lineup_confirmed.astype(str).str.lower().eq("true").sum())
    return out
