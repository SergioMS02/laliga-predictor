"""Variables previas al partido calculadas en orden cronológico (sin fuga de información).

Para cada partido se registra el estado ANTES de jugarse y solo después se actualiza
con su resultado. Partidos futuros reciben el estado actual sin modificarlo.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd

from ..teams import haversine_km, is_island

FORM_STATS = {  # nombre: (columna local, columna visitante)
    "gf": ("hg", "ag"), "sh": ("h_shots", "a_shots"), "sot": ("h_sot", "a_sot"),
    "cor": ("h_corners", "a_corners"), "xg": ("h_xg", "a_xg"),
}
ALPHA = 0.15  # EWMA por partido (vida media ~4 partidos)
PROMO_N = 10  # partidos iniciales de un ascendido que alimentan el prior
REF_SHRINK = 20.0


def _goal_mult(gd: int) -> float:
    gd = abs(gd)
    return 1.0 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8


class _EWM:
    def __init__(self):
        self.v: dict[str, float] = {}

    def get(self, k):
        return self.v.get(k, np.nan)

    def upd(self, k, x):
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return
        old = self.v.get(k)
        self.v[k] = x if old is None else (1 - ALPHA) * old + ALPHA * x


def build_sequential_features(matches: pd.DataFrame, elo_cfg: dict | None = None) -> pd.DataFrame:
    """matches: esquema común, ordenado o no. Devuelve un DataFrame alineado por índice."""
    cfg = {"k": 20, "home_adv": 60, "season_regress": 0.25, "init_new_team": 1420, **(elo_cfg or {})}
    m = matches.sort_values(["date", "home"], kind="stable")
    elo: dict[str, float] = {}
    season_teams: dict[int, set] = defaultdict(set)
    for s, g in m.groupby("season"):
        season_teams[s] = set(g.home) | set(g.away)
    ew_for, ew_against = _EWM(), _EWM()
    ew_pts = _EWM()
    ew_home_pts, ew_away_pts = _EWM(), _EWM()
    last_date: dict[str, pd.Timestamp] = {}
    ref_stats: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])  # n, pts_local, tarjetas
    league_pts = [0.0, 0.0]  # n, suma puntos local
    league_cards = [0.0, 0.0]
    # Prior para ascendidos: media de lo que hicieron los ascendidos anteriores
    # en sus primeros PROMO_N partidos (solo pasado -> sin fuga). Sustituye a la
    # forma "rancia" de equipos que vuelven tras años fuera.
    promo = defaultdict(lambda: [0.0, 0.0])      # clave -> [suma, n]
    league = defaultdict(lambda: [0.0, 0.0])
    n_season: dict[str, int] = defaultdict(int)
    new_this: set[str] = set()
    cur_season = None
    rows = {}

    def prior(key):
        sm, n = promo[key]
        if n >= 20:
            return sm / n
        lsm, ln = league[key]
        return lsm / ln if ln else np.nan
    for idx, r in m.iterrows():
        s = r.season
        if s != cur_season:  # cambio de temporada: regresión a la media y ascendidos
            prev = season_teams.get(s - 1, set())
            if elo:
                mean = np.mean(list(elo.values()))
                for t in list(elo):
                    elo[t] = mean + (1 - cfg["season_regress"]) * (elo[t] - mean)
            new_this = {t for t in season_teams[s] if prev and t not in prev}
            for t in season_teams[s]:
                if t not in prev and (prev or t not in elo):
                    elo[t] = min(elo.get(t, cfg["init_new_team"]), cfg["init_new_team"])
            for t in new_this:
                for k in FORM_STATS:
                    ew_for.v[(t, k)] = prior(("for", k))
                    ew_against.v[(t, k)] = prior(("against", k))
                ew_pts.v[t] = prior(("pts",))
                ew_home_pts.v[t] = prior(("hpts",))
                ew_away_pts.v[t] = prior(("apts",))
                for d in (ew_for.v, ew_against.v, ew_pts.v, ew_home_pts.v, ew_away_pts.v):
                    for kk in [kk for kk, vv in d.items() if isinstance(vv, float) and math.isnan(vv)]:
                        del d[kk]
            n_season = defaultdict(int)
            cur_season = s
        h, a = r.home, r.away
        eh, ea = elo.get(h, cfg["init_new_team"]), elo.get(a, cfg["init_new_team"])
        prev_teams = season_teams.get(s - 1, set())
        feat = {
            "elo_home": eh, "elo_away": ea, "elo_diff": eh - ea + cfg["home_adv"],
            "new_home": float(bool(prev_teams) and h not in prev_teams),
            "new_away": float(bool(prev_teams) and a not in prev_teams),
            "rest_home": min((r.date - last_date[h]).days, 21) if h in last_date else 21,
            "rest_away": min((r.date - last_date[a]).days, 21) if a in last_date else 21,
            "travel_km": haversine_km(a, h),
            "island": float(is_island(h) or is_island(a)),
            "pts_home": ew_pts.get(h), "pts_away": ew_pts.get(a),
            "venue_edge": ew_home_pts.get(h) - ew_away_pts.get(h),
        }
        for k in FORM_STATS:
            feat[f"{k}_for_home"] = ew_for.get((h, k))
            feat[f"{k}_against_home"] = ew_against.get((h, k))
            feat[f"{k}_for_away"] = ew_for.get((a, k))
            feat[f"{k}_against_away"] = ew_against.get((a, k))
        ref = r.get("referee")
        has_ref = isinstance(ref, str) and ref.strip() != ""
        lp = league_pts[1] / league_pts[0] if league_pts[0] else 1.6
        lc = league_cards[1] / league_cards[0] if league_cards[0] else 5.0
        if has_ref:
            n, sp, sc = ref_stats[ref]
            feat["ref_home_ppg_dev"] = (sp + REF_SHRINK * lp) / (n + REF_SHRINK) - lp
            feat["ref_cards_dev"] = (sc + REF_SHRINK * lc) / (n + REF_SHRINK) - lc
            feat["ref_n"] = n
        else:
            feat["ref_home_ppg_dev"], feat["ref_cards_dev"], feat["ref_n"] = np.nan, np.nan, 0.0
        rows[idx] = feat

        if pd.isna(r.hg):
            continue
        # --- actualización tras el partido ---
        hg, ag = int(r.hg), int(r.ag)
        res = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        exp_h = 1 / (1 + 10 ** ((ea - eh - cfg["home_adv"]) / 400))
        delta = cfg["k"] * _goal_mult(hg - ag) * (res - exp_h)
        elo[h], elo[a] = eh + delta, ea - delta
        for k, (ch, ca) in FORM_STATS.items():
            vh, va = r.get(ch), r.get(ca)
            vh = float(vh) if vh is not None and not pd.isna(vh) else np.nan
            va = float(va) if va is not None and not pd.isna(va) else np.nan
            ew_for.upd((h, k), vh)
            ew_against.upd((h, k), va)
            ew_for.upd((a, k), va)
            ew_against.upd((a, k), vh)
        ph = 3.0 if hg > ag else 1.0 if hg == ag else 0.0
        pa = 3.0 if ag > hg else 1.0 if hg == ag else 0.0
        for team, side_pts, venue, (cf, ca) in ((h, ph, "hpts", (0, 1)), (a, pa, "apts", (1, 0))):
            n_season[team] += 1
            targets = [league] + ([promo] if team in new_this and n_season[team] <= PROMO_N else [])
            for acc in targets:
                for k, cols in FORM_STATS.items():
                    vf, va = r.get(cols[cf]), r.get(cols[ca])
                    if vf is not None and not pd.isna(vf):
                        acc[("for", k)][0] += float(vf)
                        acc[("for", k)][1] += 1
                    if va is not None and not pd.isna(va):
                        acc[("against", k)][0] += float(va)
                        acc[("against", k)][1] += 1
                acc[("pts",)][0] += side_pts
                acc[("pts",)][1] += 1
                acc[(venue,)][0] += side_pts
                acc[(venue,)][1] += 1
        ew_pts.upd(h, ph)
        ew_pts.upd(a, pa)
        ew_home_pts.upd(h, ph)
        ew_away_pts.upd(a, pa)
        last_date[h] = last_date[a] = r.date
        league_pts[0] += 1
        league_pts[1] += ph
        cards = r.get("h_yellow", np.nan) + r.get("a_yellow", np.nan)
        if has_ref:
            ref_stats[ref][0] += 1
            ref_stats[ref][1] += ph
        if not pd.isna(cards):
            league_cards[0] += 1
            league_cards[1] += cards
            if has_ref:
                ref_stats[ref][2] += cards
    out = pd.DataFrame.from_dict(rows, orient="index")
    return out.reindex(matches.index)
