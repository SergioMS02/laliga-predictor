"""Bajas y rotaciones -> 'déficit' del once -> ajuste de goles esperados.

importancia(jugador) = titularidades / partidos en los últimos N partidos del equipo.
Se separa en dos ejes:
  ataque  = importancia × producción ofensiva (xG + xA por 90 de Understat; si no hay,
            un valor típico según su posición)
  defensa = importancia × peso defensivo de su posición (portero > defensa > medio > delantero)
déficit = 1 - (lo que aporta el once que juega) / (lo que aporta el once tipo), en [0, 1].
Ajuste: λ_propio *= exp(-a·déficit_ataque_propio + b·déficit_defensa_rival).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from scipy.optimize import minimize

UNKNOWN_STARTER_IMPORTANCE = 0.5
MAX_NEWS_DEFICIT = 0.6
# xG+xA por 90 típicos y peso defensivo por posición
ROLE_ATTACK = {"G": 0.01, "D": 0.10, "M": 0.25, "F": 0.50}
ROLE_DEFENCE = {"G": 1.5, "D": 1.0, "M": 0.5, "F": 0.15}
MIN_MINUTES_XG = 270


def role(position: str | None) -> str:
    p = (position or "").upper().replace("-", "").replace(" ", "")
    if not p:
        return "M"
    if p in ("G", "GK", "GOALKEEPER", "POR", "PORTERO") or p.startswith("GOAL"):
        return "G"
    if p.startswith(("DM", "CDM")):
        return "M"
    if p in ("F", "FW", "ST", "CF", "LW", "RW", "SS", "FORWARD", "STRIKER") or p.startswith(("CF", "ST", "FORW", "STRIK", "WING")) or p.endswith("W"):
        return "F"
    if p in ("D", "DF", "CB", "LB", "RB", "LWB", "RWB", "SW", "DEFENDER") or p.startswith(("CD", "CB", "DEF")) or p.endswith("B"):
        return "D"
    return "M"


def player_importance(lineups: pd.DataFrame, team: str, before, lookback: int = 10) -> pd.DataFrame:
    """DataFrame [player_id, name, position, importance] con las alineaciones previas a `before`."""
    cols = ["player_id", "name", "position", "importance"]
    t = lineups[(lineups.team == team) & (lineups.date < pd.Timestamp(before))]
    if t.empty:
        return pd.DataFrame(columns=cols)
    events = t.drop_duplicates("event_id").sort_values("date").tail(lookback)["event_id"]
    t = t[t.event_id.isin(events)].assign(start=lambda d: d.starter.astype(bool))
    n = len(events)
    # incluye suplentes (importancia 0): solo los jugadores nunca vistos reciben el valor neutro
    pos = t["position"] if "position" in t else pd.Series("", index=t.index)
    t = t.assign(position=pos.fillna("").astype(str))
    imp = (t.groupby("player_id").agg(name=("name", "last"), position=("position", "last"),
                                      starts=("event_id", lambda s: s[t.loc[s.index, "start"]].nunique()))
           .assign(importance=lambda d: d.starts / n).reset_index())
    return imp[cols].sort_values("importance", ascending=False)


def _xi_strength(imp: pd.DataFrame) -> float:
    return float(imp.importance.head(11).sum()) if len(imp) else 0.0


# ---------------------------------------------------------------------------
# Versión global (un único déficit)
# ---------------------------------------------------------------------------
def deficit_from_lineup(imp: pd.DataFrame, starter_ids: list[str]) -> float:
    base = _xi_strength(imp)
    if base <= 0 or not starter_ids:
        return 0.0
    m = dict(zip(imp.player_id.astype(str), imp.importance))
    s = sum(m.get(str(p), UNKNOWN_STARTER_IMPORTANCE) for p in starter_ids[:11])
    return float(np.clip(1 - s / base, 0.0, 1.0))


def deficit_from_absences(imp: pd.DataFrame, absences: pd.DataFrame, doubtful_weight: float = 0.5) -> float:
    base = _xi_strength(imp)
    if base <= 0 or absences is None or absences.empty:
        return 0.0
    m = dict(zip(imp.name, imp.importance))
    lost = 0.0
    for _, a in absences.iterrows():
        w = 1.0 if a.status == "out" else doubtful_weight
        lost += w * m.get(a.player, 0.0)
    return float(np.clip(lost / base, 0.0, MAX_NEWS_DEFICIT))


# ---------------------------------------------------------------------------
# Versión ataque / defensa
# ---------------------------------------------------------------------------
def attach_production(imp: pd.DataFrame, players: pd.DataFrame | None) -> pd.DataFrame:
    """Añade att_w (xG+xA por 90) y def_w por jugador. players: filas de Understat del equipo."""
    imp = imp.copy()
    roles = imp["position"].map(role) if "position" in imp else pd.Series("M", index=imp.index)
    imp["att_w"] = roles.map(ROLE_ATTACK).astype(float)
    imp["def_w"] = roles.map(ROLE_DEFENCE).astype(float)
    if players is not None and len(players) and len(imp):
        p = players[players.minutes >= MIN_MINUTES_XG]
        if len(p):
            agg = p.groupby("player").agg(xg=("xg", "sum"), xa=("xa", "sum"), minutes=("minutes", "sum"))
            per90 = ((agg.xg + agg.xa) / agg.minutes * 90).clip(0.01, 1.5)
            names = list(per90.index)
            for i, r in imp.iterrows():
                best = process.extractOne(str(r["name"]), names, scorer=fuzz.token_set_ratio)
                if best and best[1] >= 88:
                    imp.at[i, "att_w"] = float(per90[best[0]])
    imp["att_imp"] = imp.importance * imp.att_w
    imp["def_imp"] = imp.importance * imp.def_w
    return imp


def _bases(imp: pd.DataFrame) -> tuple[float, float, pd.DataFrame]:
    xi = imp.sort_values("importance", ascending=False).head(11)
    return float(xi.att_imp.sum()), float(xi.def_imp.sum()), xi


def deficits_from_lineup(imp: pd.DataFrame, starter_ids: list[str]) -> tuple[float, float]:
    """(déficit_ataque, déficit_defensa). imp debe venir de attach_production."""
    if imp.empty or not starter_ids:
        return 0.0, 0.0
    ba, bd, xi = _bases(imp)
    if ba <= 0 or bd <= 0:
        return 0.0, 0.0
    att = dict(zip(imp.player_id.astype(str), imp.att_imp))
    dfn = dict(zip(imp.player_id.astype(str), imp.def_imp))
    unk_a, unk_d = UNKNOWN_STARTER_IMPORTANCE * ba / 11, UNKNOWN_STARTER_IMPORTANCE * bd / 11
    sa = sum(att.get(str(p), unk_a) for p in starter_ids[:11])
    sd = sum(dfn.get(str(p), unk_d) for p in starter_ids[:11])
    return float(np.clip(1 - sa / ba, 0, 1)), float(np.clip(1 - sd / bd, 0, 1))


def deficits_from_absences(imp: pd.DataFrame, absences: pd.DataFrame,
                           doubtful_weight: float = 0.5) -> tuple[float, float]:
    if imp.empty or absences is None or absences.empty:
        return 0.0, 0.0
    ba, bd, _ = _bases(imp)
    if ba <= 0 or bd <= 0:
        return 0.0, 0.0
    by_name = imp.drop_duplicates("name").set_index("name")
    la = ld = 0.0
    for _, a in absences.iterrows():
        if a.player not in by_name.index:
            continue
        w = 1.0 if a.status == "out" else doubtful_weight
        la += w * float(by_name.at[a.player, "att_imp"])
        ld += w * float(by_name.at[a.player, "def_imp"])
    return float(np.clip(la / ba, 0, MAX_NEWS_DEFICIT)), float(np.clip(ld / bd, 0, MAX_NEWS_DEFICIT))


def _split(d) -> tuple[float, float]:
    if isinstance(d, (tuple, list)):
        return float(d[0]), float(d[1])
    return float(d), float(d)


def adjust_goals(lam: float, mu: float, d_home, d_away, a: float, b: float) -> tuple[float, float]:
    """d_home/d_away: déficit único o (ataque, defensa)."""
    ah, dh = _split(d_home)
    aa, da = _split(d_away)
    return (lam * float(np.exp(-a * ah + b * da)), mu * float(np.exp(-a * aa + b * dh)))


def calibrate(df: pd.DataFrame, prior=(0.6, 0.35), prior_strength: float = 5.0) -> tuple[float, float]:
    """Estima (a, b) por máxima verosimilitud Poisson con un prior gaussiano suave.

    df: lam, mu (modelo base), hg, ag y o bien d_home/d_away o bien
    datt_home, ddef_home, datt_away, ddef_away."""
    lam, mu = df.lam.to_numpy(float), df.mu.to_numpy(float)
    if {"datt_home", "ddef_home", "datt_away", "ddef_away"} <= set(df.columns):
        ah, dh = df.datt_home.to_numpy(float), df.ddef_home.to_numpy(float)
        aa, da = df.datt_away.to_numpy(float), df.ddef_away.to_numpy(float)
    else:
        ah = dh = df.d_home.to_numpy(float)
        aa = da = df.d_away.to_numpy(float)
    x, y = df.hg.to_numpy(float), df.ag.to_numpy(float)

    def nll(p):
        a, b = p
        ll_ = np.log(lam) - a * ah + b * da
        lm_ = np.log(mu) - a * aa + b * dh
        v = -(x * ll_ - np.exp(ll_) + y * lm_ - np.exp(lm_)).sum()
        return v + 0.5 * prior_strength * ((a - prior[0]) ** 2 + (b - prior[1]) ** 2)
    res = minimize(nll, np.array(prior), method="L-BFGS-B", bounds=[(0, 3), (0, 3)])
    return float(res.x[0]), float(res.x[1])
