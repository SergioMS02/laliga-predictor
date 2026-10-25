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
                plain_names = [_plain(n) for n in names]
                best = process.extractOne(_plain(r["name"]), plain_names, scorer=fuzz.token_set_ratio)
                if best and best[1] >= 88:
                    imp.at[i, "att_w"] = float(per90.iloc[best[2]])
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
    plain_idx = {n: _plain(n) for n in by_name.index}
    la = ld = 0.0
    for _, a in absences.iterrows():
        name = a.player
        if name not in by_name.index:  # mismo jugador escrito distinto (acentos, apellidos compuestos)
            best = process.extractOne(_plain(name), plain_idx, scorer=fuzz.token_set_ratio) if plain_idx else None
            if not best or best[1] < 85:
                continue
            name = best[2]
        a = a.copy()
        a["player"] = name
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


# ---------------------------------------------------------------------------
# Vista de plantilla para la web (once esperado u oficial)
# ---------------------------------------------------------------------------
def expected_xi(imp: pd.DataFrame) -> pd.DataFrame:
    """Once tipo: el portero más habitual y los 10 jugadores de campo con más titularidades."""
    if imp is None or imp.empty:
        return imp if imp is not None else pd.DataFrame()
    d = imp[imp.importance > 0].copy()
    if d.empty:
        return d
    d["role"] = d["position"].map(role) if "position" in d else "M"
    gk = d[d.role == "G"].sort_values("importance", ascending=False).head(1)
    field = d[d.role != "G"].sort_values("importance", ascending=False).head(11 - len(gk))
    return pd.concat([gk, field])


def _pinfo(row, absent=None) -> dict:
    return {"name": row.get("name", ""), "pos": row.get("position", "") or "", "role": role(row.get("position")),
            "imp": round(float(row.get("importance", 0) or 0), 2),
            "att": round(float(row.get("att_w", 0) or 0), 2), "absent": absent}


def squad_view(imp: pd.DataFrame, starters: list | None = None, formation: str | None = None,
               absences: pd.DataFrame | list | None = None) -> dict:
    """Once para dibujar en la web.

    starters: lista de espn.Player (alineación oficial) o None (se usa el once tipo)."""
    imp = imp if imp is not None else pd.DataFrame(columns=["player_id", "name", "position", "importance"])
    by_id = {str(r.player_id): r._asdict() for r in imp.itertuples(index=False)} if len(imp) else {}
    exp = expected_xi(imp)
    if starters:
        xi = []
        ids = set()
        for p in starters:
            r = by_id.get(str(p.player_id), {"name": p.name, "position": p.position, "importance": 0, "att_w": 0})
            info = _pinfo({**r, "name": p.name, "position": p.position or r.get("position", "")})
            info["jersey"] = getattr(p, "jersey", "")
            info["usual"] = info["imp"] >= 0.5
            xi.append(info)
            ids.add(str(p.player_id))
        out = [{"name": r["name"], "status": "no titular", "imp": round(float(r["importance"]), 2)}
               for _, r in exp.iterrows() if str(r["player_id"]) not in ids]
        return {"confirmed": True, "formation": formation, "xi": xi, "out": out}
    ab, meta = {}, {}
    if absences is not None:
        rows = absences.to_dict(orient="records") if isinstance(absences, pd.DataFrame) else absences
        for a in rows:
            ab[a["player"]] = a.get("status", "out")
            meta[a["player"]] = a
    is_out = lambda n: _status_of(n, ab) == "out"  # noqa: E731
    # once probable: los habituales que no están de baja; cada baja la ocupa el siguiente más
    # habitual de su misma demarcación (si no hay, el siguiente jugador de campo)
    xi, used = [], set()
    pool = imp[imp.importance > 0].copy() if len(imp) else imp
    if len(pool):
        pool["role"] = pool["position"].map(role) if "position" in pool else "M"
        pool = pool.sort_values("importance", ascending=False)
    for _, r in exp.iterrows():
        used.add(r["name"])
    for _, r in exp.iterrows():
        if not is_out(r["name"]):
            info = _pinfo(r, _status_of(r["name"], ab))
            info["usual"] = True
            xi.append(info)
            continue
        rl = role(r.get("position"))
        cand = pool[(~pool.name.isin(used)) & (pool.role == rl)] if len(pool) else pool
        if cand is None or cand.empty:
            cand = pool[(~pool.name.isin(used)) & (pool.role != "G")] if len(pool) else pool
        cand = cand[[not is_out(n) for n in cand.name]] if cand is not None and len(cand) else cand
        if cand is not None and len(cand):
            rep = cand.iloc[0]
            used.add(rep["name"])
            info = _pinfo(rep, _status_of(rep["name"], ab))
            info["usual"] = False
            info["replaces"] = r["name"]
            xi.append(info)
    out = [{"name": n, "status": s, "reason": meta[n].get("reason"), "source": meta[n].get("source"),
            "imp": round(float(imp[imp.name == n].importance.max()), 2) if len(imp) and (imp.name == n).any() else 0}
           for n, s in ab.items()]
    return {"confirmed": False, "formation": None, "xi": xi, "out": out}


def _status_of(name: str, ab: dict) -> str | None:
    if name in ab:
        return ab[name]
    if not ab:
        return None
    best = process.extractOne(_plain(name), {k: _plain(k) for k in ab}, scorer=fuzz.token_set_ratio)
    return ab[best[2]] if best and best[1] >= 88 else None


def roster(imp: pd.DataFrame | None, players: pd.DataFrame | None, absences: list | None = None,
           starters: list[str] | None = None) -> list[dict]:
    """Plantilla con su disponibilidad: titularidades recientes, xG+xA por 90, goles y estado."""
    rows: dict[str, dict] = {}
    if imp is not None and len(imp):
        for _, r in imp.iterrows():
            rows[r["name"]] = {"name": r["name"], "role": role(r.get("position")), "imp": round(float(r.get("importance", 0) or 0), 2),
                               "att": round(float(r.get("att_w", 0) or 0), 2), "goals": None, "minutes": None}
    if players is not None and len(players):
        plain_rows = {k: _plain(k) for k in rows}
        for r in players.itertuples():
            best = process.extractOne(_plain(r.player), plain_rows, scorer=fuzz.token_set_ratio) if plain_rows else None
            key = best[2] if best and best[1] >= 88 else None
            pos = str(r.position or "").split(" ")[0]
            rl = {"GK": "G", "D": "D", "M": "M", "F": "F"}.get(pos, "M")
            if key is None:
                if (r.minutes or 0) < 90:
                    continue
                key = r.player
                rows[key] = {"name": r.player, "role": rl, "imp": 0.0, "att": None, "goals": None, "minutes": None}
            rows[key]["goals"] = int(r.goals or 0)
            rows[key]["minutes"] = int(r.minutes or 0)
            if r.minutes and r.minutes >= MIN_MINUTES_XG:
                rows[key]["att"] = round(float((r.xg + r.xa) / r.minutes * 90), 2)
    ab = {a["player"]: a for a in (absences or [])}
    st = set(starters or [])
    out = []
    for k, v in rows.items():
        a = ab.get(k)
        if a is None and ab:
            best = process.extractOne(_plain(k), {n: _plain(n) for n in ab}, scorer=fuzz.token_set_ratio)
            a = ab[best[2]] if best and best[1] >= 88 else None
        if a is not None:
            status = "duda" if a.get("status") == "doubtful" else ("sancionado" if a.get("reason") == "suspension" else "baja")
            v.update(status=status, source=a.get("source"))
        elif starters is not None:
            v["status"] = "titular" if k in st else "suplente"
        else:
            v["status"] = "disponible"
        out.append(v)
    order = {"G": 0, "D": 1, "M": 2, "F": 3}
    out.sort(key=lambda x: (order.get(x["role"], 2), -(x["imp"] or 0), -(x["minutes"] or 0)))
    return out


def squad_from_players(players: pd.DataFrame | None, absences: list | None = None) -> dict:
    """Once probable por minutos jugados (Understat) cuando aún no hay historial de alineaciones."""
    if players is None or players.empty:
        return {"confirmed": False, "formation": None, "xi": [], "out": [], "source": None}
    p = players.copy()
    p["role"] = p["position"].fillna("").map(lambda s: {"GK": "G", "D": "D", "M": "M", "F": "F"}.get(str(s).split(" ")[0], "M"))
    p = p.sort_values("minutes", ascending=False)
    total = p.minutes.max() or 1
    pick = pd.concat([p[p.role == "G"].head(1), p[p.role == "D"].head(4), p[p.role == "M"].head(3), p[p.role == "F"].head(3)])
    if len(pick) < 11:
        pick = pd.concat([pick, p[~p.index.isin(pick.index) & (p.role != "G")].head(11 - len(pick))])
    ab = {a["player"]: a.get("status", "out") for a in (absences or [])}
    names = list(ab)
    xi = []
    for r in pick.head(11).itertuples():
        hit = None
        if names:
            from rapidfuzz import fuzz as _f, process as _p
            best = _p.extractOne(r.player, names, scorer=_f.token_set_ratio)
            hit = ab[best[0]] if best and best[1] >= 88 else None
        per90 = (r.xg + r.xa) / r.minutes * 90 if r.minutes else 0
        xi.append({"name": r.player, "pos": r.role, "role": r.role, "imp": round(float(r.minutes / total), 2),
                   "att": round(float(per90), 2), "absent": hit, "usual": True})
    return {"confirmed": False, "formation": None, "xi": xi,
            "out": [{"name": n, "status": s, "imp": None} for n, s in ab.items()], "source": "minutos"}


def _plain(s) -> str:
    import unicodedata as _u
    s = _u.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return " ".join(s.replace("-", " ").replace(".", " ").split())
