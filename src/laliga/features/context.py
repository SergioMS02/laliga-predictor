"""Variables de contexto previas al partido (sin fuga: solo usan información anterior,
salvo el calendario futuro de otras competiciones, que se conoce de antemano).

Grupos:
  stakes      lo que se juega cada equipo según la clasificación (título, Europa, descenso)
  h2h         historial de enfrentamientos directos
  discipline  roja en el partido anterior (sanción probable), tendencia a tarjetas
  timing      entre semana, hora de inicio, vuelta de parón internacional, inicio de temporada
  derby       derbi de ciudad o regional
  momentum    forma reciente vs forma de fondo
  venue_form  rendimiento del local en casa y del visitante fuera
  congestion  partidos europeos/Copa antes y después, descanso real, carga de 14 días
  coach       entrenador recién llegado
  no_fans     partidos a puerta cerrada (2020-21)
"""
from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections import defaultdict

import numpy as np
import pandas as pd

from ..teams import is_derby

H2H_YEARS = 6
H2H_N = 6
H2H_SHRINK = 4.0
EURO_WINDOW = 4
NEW_COACH_DAYS = 60
NO_FANS = (pd.Timestamp("2020-06-01"), pd.Timestamp("2021-05-31"))


def _race(p: float, boundary: float, m_left: int) -> float:
    gap = abs(p - boundary)
    if gap > 3 * m_left + 0.5:
        return 0.0
    return math.exp(-gap / (m_left + 1))


def _stakes(table: dict, team: str, rounds: int) -> tuple[float, int]:
    pts = sorted(((v[0], v[1]) for v in table.values()), reverse=True)
    p, _, played = table[team]
    m_left = max(rounds - played, 0)
    vals = [x[0] for x in pts]
    n = len(vals)
    races = []
    if n >= 2:
        races.append(_race(p, (vals[0] + vals[1]) / 2, m_left))
    if n >= 8:
        races.append(_race(p, (vals[6] + vals[7]) / 2, m_left))
    if n >= 18:
        races.append(_race(p, (vals[16] + vals[17]) / 2, m_left))
    return (max(races) if races else 1.0), m_left


def coverage_seasons(other: pd.DataFrame) -> set[int]:
    """Temporadas con calendario completo de otras competiciones (Champions, Europa League y Copa)."""
    if other is None or other.empty:
        return set()
    o = other.assign(season=[d.year if d.month >= 7 else d.year - 1 for d in other.date])
    comps = o.groupby("season").comp.agg(set)
    return {int(s) for s, c in comps.items() if {"cl", "el", "cup"} <= c}


def build_context_features(matches: pd.DataFrame, other: pd.DataFrame | None = None,
                           coaches: pd.DataFrame | None = None) -> pd.DataFrame:
    m = matches.sort_values(["date", "home"], kind="stable")
    season_teams = {s: set(g.home) | set(g.away) for s, g in m.groupby("season")}

    # calendario por equipo: liga (todas las fechas, jugadas o no) + otras competiciones
    league_dates: dict[str, list] = defaultdict(list)
    for r in m.itertuples():
        league_dates[r.home].append(r.date)
        league_dates[r.away].append(r.date)
    other_dates: dict[str, list] = defaultdict(list)
    covered = coverage_seasons(other)
    if other is not None and len(other):
        for r in other.itertuples():
            other_dates[r.team].append(pd.Timestamp(r.date))
    for d in (league_dates, other_dates):
        for k in d:
            d[k] = sorted(d[k])
    coach_dates: dict[str, list] = defaultdict(list)
    if coaches is not None and len(coaches):
        for r in coaches.itertuples():
            coach_dates[r.team].append(pd.Timestamp(r.date))
        for k in coach_dates:
            coach_dates[k] = sorted(coach_dates[k])

    table: dict[int, dict] = {}
    h2h: dict[frozenset, list] = defaultdict(list)
    red_prev: dict[str, float] = {}
    yel = {}
    mom_s, mom_l = {}, {}
    home_gd, away_gd = {}, {}
    league_draw = [0.0, 0.0]
    rows = {}

    def ewm(d, k, x, a):
        d[k] = x if k not in d else (1 - a) * d[k] + a * x

    for idx, r in m.iterrows():
        s, h, a, d = r.season, r.home, r.away, r.date
        if s not in table:
            table[s] = {t: [0.0, 0.0, 0] for t in season_teams[s]}
        tab = table[s]
        rounds = 2 * (len(season_teams[s]) - 1)
        st_h, left_h = _stakes(tab, h, rounds)
        st_a, left_a = _stakes(tab, a, rounds)
        late = 1 - (left_h + left_a) / (2 * rounds) if rounds else 0.0
        f = {
            "late": late,
            "stake_diff_late": (st_h - st_a) * late,
            "dead_home": float(left_h <= 10 and st_h < 0.05),
            "dead_away": float(left_a <= 10 and st_a < 0.05),
        }
        # h2h desde la perspectiva del local actual
        past = [x for x in h2h[frozenset((h, a))] if (d - x[0]).days <= 365 * H2H_YEARS][-H2H_N:]
        gd = sum((x[3] - x[4]) if x[1] == h else (x[4] - x[3]) for x in past)
        dr = sum(x[3] == x[4] for x in past)
        base_draw = league_draw[1] / league_draw[0] if league_draw[0] else 0.25
        f["h2h_gd"] = gd / (len(past) + H2H_SHRINK)
        f["h2h_draw"] = (dr + H2H_SHRINK * base_draw) / (len(past) + H2H_SHRINK) - base_draw
        # disciplina
        f["red_prev_home"] = red_prev.get(h, 0.0)
        f["red_prev_away"] = red_prev.get(a, 0.0)
        f["yellow_diff"] = yel.get(h, np.nan) - yel.get(a, np.nan)
        # horario
        ko = r.get("kickoff")
        hour = np.nan
        if ko is not None and not pd.isna(ko):
            hour = pd.Timestamp(ko).tz_convert("Europe/Madrid").hour + pd.Timestamp(ko).tz_convert("Europe/Madrid").minute / 60
        f["midweek"] = float(d.weekday() in (1, 2, 3))
        f["ko_early"] = np.nan if np.isnan(hour) else float(hour < 16)
        f["ko_late"] = np.nan if np.isnan(hour) else float(hour >= 21)
        lh = league_dates[h]
        la = league_dates[a]
        ih, ia = bisect_left(lh, d), bisect_left(la, d)
        rest_h = (d - lh[ih - 1]).days if ih > 0 else 99
        rest_a = (d - la[ia - 1]).days if ia > 0 else 99
        f["post_break"] = float(12 <= rest_h <= 20 and 12 <= rest_a <= 20 and d.month not in (7, 8))
        f["early_season"] = float(tab[h][2] < 3 or tab[a][2] < 3)
        # partidos a puerta cerrada (COVID): sin público la ventaja de local cae
        f["no_fans"] = float(NO_FANS[0] <= d <= NO_FANS[1])
        f["derby"] = float(is_derby(h, a))
        f["mom_diff"] = (mom_s.get(h, np.nan) - mom_l.get(h, np.nan)) - (mom_s.get(a, np.nan) - mom_l.get(a, np.nan))
        f["venue_form_diff"] = home_gd.get(h, np.nan) - away_gd.get(a, np.nan)
        # congestión (solo temporadas con calendario completo)
        if s in covered:
            for side, t, rest_lg in (("home", h, rest_h), ("away", a, rest_a)):
                od = other_dates.get(t, [])
                lo = bisect_left(od, d - pd.Timedelta(days=EURO_WINDOW))
                hi = bisect_left(od, d)
                nx = bisect_right(od, d + pd.Timedelta(days=EURO_WINDOW))
                prev_o = od[hi - 1] if hi > 0 else None
                rest_all = min(rest_lg, (d - prev_o).days if prev_o is not None else 99)
                load = (bisect_left(od, d) - bisect_left(od, d - pd.Timedelta(days=14))) + \
                       (bisect_left(league_dates[t], d) - bisect_left(league_dates[t], d - pd.Timedelta(days=14)))
                f[f"euro_prev_{side}"] = float(hi - lo > 0)
                f[f"euro_next_{side}"] = float(nx - bisect_right(od, d) > 0)
                f[f"rest_all_{side}"] = min(rest_all, 14)
                f[f"load14_{side}"] = load
            f["rest_all_diff"] = f.pop("rest_all_home") - f.pop("rest_all_away")
            f["load14_diff"] = f.pop("load14_home") - f.pop("load14_away")
        else:
            for c in ("euro_prev_home", "euro_prev_away", "euro_next_home", "euro_next_away", "rest_all_diff", "load14_diff"):
                f[c] = np.nan
        # entrenador
        for side, t in (("home", h), ("away", a)):
            cd = coach_dates.get(t, [])
            i = bisect_right(cd, d)
            f[f"new_coach_{side}"] = float(i > 0 and (d - cd[i - 1]).days <= NEW_COACH_DAYS)
        rows[idx] = f

        if pd.isna(r.hg):
            continue
        hg, ag = int(r.hg), int(r.ag)
        ph = 3 if hg > ag else 1 if hg == ag else 0
        pa = 3 if ag > hg else 1 if hg == ag else 0
        for t, p, g in ((h, ph, hg - ag), (a, pa, ag - hg)):
            tab[t][0] += p
            tab[t][1] += g
            tab[t][2] += 1
            ewm(mom_s, t, p, 0.5)
            ewm(mom_l, t, p, 0.1)
        ewm(home_gd, h, hg - ag, 0.15)
        ewm(away_gd, a, ag - hg, 0.15)
        h2h[frozenset((h, a))].append((d, h, a, hg, ag))
        league_draw[0] += 1
        league_draw[1] += float(hg == ag)
        hr, ar = r.get("h_red"), r.get("a_red")
        red_prev[h] = float(hr > 0) if hr is not None and not pd.isna(hr) else 0.0
        red_prev[a] = float(ar > 0) if ar is not None and not pd.isna(ar) else 0.0
        for t, c in ((h, r.get("h_yellow")), (a, r.get("a_yellow"))):
            if c is not None and not pd.isna(c):
                ewm(yel, t, float(c), 0.15)
    return pd.DataFrame.from_dict(rows, orient="index").reindex(matches.index)
