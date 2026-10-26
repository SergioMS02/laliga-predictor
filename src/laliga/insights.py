"""Análisis extra para la web: tabla de la suerte (puntos frente a puntos esperados por xG), estilo de
juego de cada equipo, Elo histórico, árbitros, fichas de jugador, premios de temporada (Zamora,
máximo asistente, equipo más goleador) y previas automáticas de cada partido.

Fuentes: Understat (xG, xPts, presión PPDA, pases profundos, jugadores), football-data (árbitros,
tarjetas), ESPN (tarjetas y goles de la temporada) y las simulaciones de la proyección.
"""
from __future__ import annotations

import json
import logging
import math
import unicodedata

import numpy as np
import pandas as pd

from .config import Settings
from .teams import canonical

log = logging.getLogger(__name__)


def _r(x, n=2):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if not math.isfinite(x) else round(x, n)


def _plain(s) -> str:
    return " ".join(unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower().split())


def _understat(settings: Settings, season: int) -> dict | None:
    p = settings.raw_dir / "understat" / f"La_liga_{season}.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Suerte y estilo
# ---------------------------------------------------------------------------
def team_history(settings: Settings, season: int) -> pd.DataFrame:
    js = _understat(settings, season) or {}
    rows = []
    for t in (js.get("teams") or {}).values():
        try:
            team = canonical(t["title"])
        except KeyError:
            continue
        for h in t.get("history") or []:
            pp, pa = h.get("ppda") or {}, h.get("ppda_allowed") or {}
            rows.append({"team": team, "date": h.get("date", "")[:10], "venue": h.get("h_a"),
                         "xg": float(h.get("xG") or 0), "xga": float(h.get("xGA") or 0),
                         "npxg": float(h.get("npxG") or 0), "npxga": float(h.get("npxGA") or 0),
                         "gf": int(h.get("scored") or 0), "ga": int(h.get("missed") or 0),
                         "pts": int(h.get("pts") or 0), "xpts": float(h.get("xpts") or 0),
                         "ppda_att": float(pp.get("att") or 0), "ppda_def": float(pp.get("def") or 0),
                         "oppda_att": float(pa.get("att") or 0), "oppda_def": float(pa.get("def") or 0),
                         "deep": int(h.get("deep") or 0), "deep_allowed": int(h.get("deep_allowed") or 0)})
    return pd.DataFrame(rows)


def luck_table(hist: pd.DataFrame) -> list[dict]:
    """Puntos reales frente a puntos esperados (xPts de Understat: la probabilidad de cada resultado
    según las ocasiones que tuvo cada equipo en cada partido)."""
    if hist.empty:
        return []
    g = hist.groupby("team").agg(pj=("pts", "size"), pts=("pts", "sum"), xpts=("xpts", "sum"), gf=("gf", "sum"),
                                 ga=("ga", "sum"), xg=("xg", "sum"), xga=("xga", "sum"))
    g["diff"] = g.pts - g.xpts
    out = [{"team": t, "pj": int(r.pj), "pts": int(r.pts), "xpts": _r(r.xpts, 1), "diff": _r(r["diff"], 1),
            "gf": int(r.gf), "xg": _r(r.xg, 1), "ga": int(r.ga), "xga": _r(r.xga, 1)} for t, r in g.iterrows()]
    out.sort(key=lambda d: -d["diff"])
    return out


STYLE_FEATURES = {
    "ppda": "Presión (pases rivales por acción defensiva; menos = presiona más)",
    "oppda": "Presión que recibe (más = el rival le deja tener el balón)",
    "deep": "Pases al área cercana por partido",
    "deep_allowed": "Pases al área cercana que concede",
    "npxg": "xG sin penaltis por partido",
    "npxga": "xG sin penaltis en contra por partido",
}


def team_styles(hist: pd.DataFrame) -> dict:
    """Rasgos de juego de cada equipo (valores por partido y percentil dentro de la liga) y un
    agrupamiento simple (k-medias sobre los rasgos estandarizados) en estilos."""
    if hist.empty:
        return {}
    g = hist.groupby("team").agg(ppda_att=("ppda_att", "sum"), ppda_def=("ppda_def", "sum"),
                                 oppda_att=("oppda_att", "sum"), oppda_def=("oppda_def", "sum"),
                                 deep=("deep", "mean"), deep_allowed=("deep_allowed", "mean"),
                                 npxg=("npxg", "mean"), npxga=("npxga", "mean"), n=("pts", "size"))
    g["ppda"] = g.ppda_att / g.ppda_def.clip(lower=1)
    g["oppda"] = g.oppda_att / g.oppda_def.clip(lower=1)
    feats = list(STYLE_FEATURES)
    X = g[feats].to_numpy(float)
    Z = (X - X.mean(0)) / X.std(0).clip(min=1e-9)
    labels = _kmeans(Z, 4)
    names = {}
    for c in set(labels):
        mean = Z[labels == c].mean(0)
        f = dict(zip(feats, mean))
        if f["ppda"] < -0.5 and f["deep"] > 0.3:
            names[c] = "Presión alta y dominio"
        elif f["oppda"] > 0.5 and f["deep"] > 0.3:
            names[c] = "Posesión y control"
        elif f["ppda"] > 0.4 and f["npxga"] > 0.2:
            names[c] = "Bloque bajo que sufre"
        elif f["ppda"] > 0.4:
            names[c] = "Bloque bajo y contra"
        elif f["npxg"] < -0.3:
            names[c] = "Pocas ocasiones"
        else:
            names[c] = "Equilibrado"
    pct = (pd.DataFrame(X, columns=feats, index=g.index).rank(pct=True))
    teams = {}
    for k, (t, r) in enumerate(g.iterrows()):
        teams[t] = {"style": names[labels[k]], "values": {f: _r(r[f], 2) for f in feats},
                    "pct": {f: _r(pct.loc[t, f], 3) for f in feats}}
    return {"features": STYLE_FEATURES, "teams": teams}


def _kmeans(Z: np.ndarray, k: int, iters: int = 50, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = min(k, len(Z))
    best, best_in = None, np.inf
    for _ in range(10):
        c = Z[rng.choice(len(Z), k, replace=False)]
        for _ in range(iters):
            lab = ((Z[:, None, :] - c[None]) ** 2).sum(2).argmin(1)
            new = np.array([Z[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
            if np.allclose(new, c):
                break
            c = new
        inertia = ((Z - c[lab]) ** 2).sum()
        if inertia < best_in:
            best, best_in = lab, inertia
    return best


def style_matchups(matches: pd.DataFrame, settings: Settings, seasons: list[int]) -> list[dict]:
    """Puntos por partido de cada estilo contra cada estilo en las últimas temporadas."""
    rows = []
    for s in seasons:
        hist = team_history(settings, s)
        st = team_styles(hist).get("teams", {})
        if not st:
            continue
        sm = matches[(matches.season == s) & matches.hg.notna()]
        for r in sm.itertuples():
            a, b = st.get(r.home, {}).get("style"), st.get(r.away, {}).get("style")
            if a and b:
                rows.append((a, b, 3 if r.hg > r.ag else 1 if r.hg == r.ag else 0))
                rows.append((b, a, 3 if r.ag > r.hg else 1 if r.hg == r.ag else 0))
    if not rows:
        return []
    d = pd.DataFrame(rows, columns=["style", "vs", "pts"]).groupby(["style", "vs"]).pts.agg(["mean", "size"])
    return [{"style": a, "vs": b, "ppg": _r(r["mean"], 2), "n": int(r["size"])} for (a, b), r in d.iterrows()]


# ---------------------------------------------------------------------------
# Elo histórico
# ---------------------------------------------------------------------------
def elo_history(matches: pd.DataFrame, seq: pd.DataFrame) -> dict:
    """Elo al final de cada temporada de cada equipo (desde 2005) y su máximo histórico."""
    m = matches.join(seq[["elo_home", "elo_away"]]).dropna(subset=["hg"]).sort_values("date")
    # Elo después del último partido de la temporada ≈ el de antes + el cambio; se usa el de antes del
    # último partido (suficiente para ver la evolución)
    rows = {}
    for r in m.itertuples():
        rows[(r.season, r.home)] = r.elo_home
        rows[(r.season, r.away)] = r.elo_away
    out: dict[str, list] = {}
    for (season, team), e in sorted(rows.items()):
        out.setdefault(team, []).append([int(season), round(float(e), 1)])
    best = {t: max(v, key=lambda x: x[1]) for t, v in out.items()}
    return {"seasons": out, "best": best}


# ---------------------------------------------------------------------------
# Árbitros
# ---------------------------------------------------------------------------
def referee_stats(matches: pd.DataFrame, season: int, min_matches: int = 3) -> list[dict]:
    from .props import ref_key
    d = matches[(matches.season >= season - 1) & matches.hg.notna() & matches.referee.notna()].copy()
    if d.empty:
        return []
    d["key"] = d.referee.map(ref_key)
    lg = {"yellow": float((d.h_yellow + d.a_yellow).mean()), "red": float((d.h_red + d.a_red).mean()),
          "fouls": float((d.h_fouls + d.a_fouls).mean()), "goals": float((d.hg + d.ag).mean())}
    out = []
    for key, g in d.groupby("key"):
        if len(g) < min_matches:
            continue
        cur = g[g.season == season]
        out.append({"referee": g.referee.mode().iloc[0], "key": key, "n": int(len(g)), "n_season": int(len(cur)),
                    "yellow": _r((g.h_yellow + g.a_yellow).mean(), 2), "red": _r((g.h_red + g.a_red).mean(), 3),
                    "fouls": _r((g.h_fouls + g.a_fouls).mean(), 1), "goals": _r((g.hg + g.ag).mean(), 2),
                    "home_win": _r((g.hg > g.ag).mean(), 3),
                    "teams": [{"team": t, "n": int(n)} for t, n in
                              pd.concat([cur.home, cur.away]).value_counts().head(5).items()]})
    out.sort(key=lambda r: -(r["yellow"] or 0))
    return [{"league": {k: _r(v, 2) for k, v in lg.items()}}] + out


# ---------------------------------------------------------------------------
# Jugadores
# ---------------------------------------------------------------------------
def player_cards(settings: Settings, season: int, events: pd.DataFrame | None, shots: pd.DataFrame | None,
                 top_per_team: int = 18) -> dict:
    """Fichas de jugador: goles, xG, asistencias, xA, minutos, tiros, pases clave, tarjetas, penaltis
    y sus tiros de la temporada (para el mapa)."""
    js = _understat(settings, season) or {}
    out: dict[str, list] = {}
    for p in js.get("players") or []:
        team = str(p.get("team_title", "")).split(",")[-1].strip()
        try:
            team = canonical(team)
        except KeyError:
            continue
        f = lambda k: float(p.get(k) or 0)  # noqa: E731
        mins = f("time")
        card = {"name": p.get("player_name", ""), "id": p.get("id"), "pos": p.get("position", ""),
                "games": int(f("games")), "minutes": int(mins), "goals": int(f("goals")), "npg": int(f("npg")),
                "pens": int(f("goals") - f("npg")), "xg": _r(f("xG"), 2), "npxg": _r(f("npxG"), 2),
                "assists": int(f("assists")), "xa": _r(f("xA"), 2), "shots": int(f("shots")),
                "key_passes": int(f("key_passes")), "yellow": int(f("yellow_cards")), "red": int(f("red_cards")),
                "xg90": _r(f("xG") / mins * 90, 2) if mins >= 90 else None,
                "xa90": _r(f("xA") / mins * 90, 2) if mins >= 90 else None,
                "chain90": _r(f("xGChain") / mins * 90, 2) if mins >= 90 else None}
        out.setdefault(team, []).append(card)
    pens = penalty_takers(shots)
    shot_map = {}
    if shots is not None and len(shots):
        s = shots[(shots.season == season) & (shots.minute >= 0)]
        for (team, player), g in s.groupby(["team", "player"]):
            shot_map[(team, _plain(player))] = [[_r(x, 3), _r(y, 3), _r(xg, 3), res[:1] if res != "Goal" else "G"]
                                                for x, y, xg, res in zip(g.x, g.y, g.xg, g.result)]
    for team, lst in out.items():
        lst.sort(key=lambda c: -(c["minutes"]))
        lst[:] = lst[:top_per_team]
        for c in lst:
            c["shots_map"] = shot_map.get((team, _plain(c["name"])), [])
            pk = pens.get(team, {}).get(_plain(c["name"]))
            if pk:
                c["pen_taken"], c["pen_scored"] = pk
    return out


def penalty_takers(shots: pd.DataFrame | None) -> dict:
    """Penaltis lanzados y marcados por jugador (temporada actual y anterior): {team: {jugador: (n, goles)}}."""
    if shots is None or shots.empty:
        return {}
    s = shots[shots.situation == "Penalty"]
    out: dict = {}
    for (team, player), g in s.groupby(["team", "player"]):
        out.setdefault(team, {})[_plain(player)] = (int(len(g)), int((g.result == "Goal").sum()))
    return out


def main_penalty_taker(shots: pd.DataFrame | None, team: str, season: int) -> str | None:
    if shots is None or shots.empty:
        return None
    s = shots[(shots.team == team) & (shots.situation == "Penalty") & (shots.season >= season - 1)]
    if s.empty:
        return None
    w = np.where(s.season == season, 2.0, 1.0)
    return str(pd.Series(w, index=s.player.values).groupby(level=0).sum().idxmax())


# ---------------------------------------------------------------------------
# Premios de temporada (con las simulaciones de la proyección)
# ---------------------------------------------------------------------------
def season_awards(settings: Settings, table: pd.DataFrame, sim: dict, players: pd.DataFrame | None, season: int,
                  dc_left: dict, top: int = 8, seed: int = 5) -> dict:
    """Equipo más goleador y menos goleado (de las simulaciones), Zamora (portero titular con menos goles
    encajados por partido, mínimo 28 partidos) y máximo asistente (Monte Carlo con su parte del xA)."""
    out = {}
    if sim and "hg" in sim and sim.get("teams"):
        teams = sim["teams"]
        gf = table.set_index("team").gf.reindex(teams).to_numpy(float)
        ga = table.set_index("team").gc.reindex(teams).to_numpy(float)
        n = sim["hg"].shape[0]
        GF, GA = np.tile(gf, (n, 1)), np.tile(ga, (n, 1))
        idx = {t: i for i, t in enumerate(teams)}
        for j, (h, a) in enumerate(sim["pairs"]):
            GF[:, idx[h]] += sim["hg"][:, j]
            GF[:, idx[a]] += sim["ag"][:, j]
            GA[:, idx[h]] += sim["ag"][:, j]
            GA[:, idx[a]] += sim["hg"][:, j]

        def share(M, best):
            w = (M == best[:, None]).astype(float)
            return (w / w.sum(1, keepdims=True)).mean(0)
        p_att = share(GF, GF.max(1))
        p_def = share(GA, GA.min(1))
        out["most_goals"] = sorted([{"team": t, "p": _r(p_att[i], 4), "exp": _r(GF[:, i].mean(), 1)} for i, t in enumerate(teams)],
                                   key=lambda d: -d["p"])[:5]
        out["least_conceded"] = sorted([{"team": t, "p": _r(p_def[i], 4), "exp": _r(GA[:, i].mean(), 1)} for i, t in enumerate(teams)],
                                       key=lambda d: -d["p"])[:5]
        # Zamora: portero con más minutos de cada equipo; encaja los goles del equipo en sus minutos
        if players is not None and len(players):
            cur = players[(players.season == season) & players.position.astype(str).str.contains("GK")]
            gks = cur.sort_values("minutes", ascending=False).drop_duplicates("team")
            games_total = 38
            rows = []
            for r in gks.itertuples():
                if r.team not in idx:
                    continue
                pj = float(table.set_index("team").pj.get(r.team, 0)) or 1
                frac = min(1.0, (r.minutes or 0) / (90 * pj))
                if frac < 0.6:
                    continue
                i = idx[r.team]
                ratio = GA[:, i] / games_total  # ritmo de equipo (aprox. con el portero jugándolo todo)
                rows.append((r.player, r.team, ratio, frac))
            if rows:
                R = np.vstack([x[2] for x in rows]).T
                best = R.min(1)
                p = share(R, best)
                out["zamora"] = sorted([{"name": x[0], "team": x[1], "p": _r(p[k], 4), "ratio": _r(x[2].mean(), 2)}
                                        for k, x in enumerate(rows)], key=lambda d: -d["p"])[:top]
    # máximo asistente
    if players is not None and len(players):
        cur = players[players.season == season]
        prev = players[players.season == season - 1]
        rows = []
        for t, g in cur.groupby("team"):
            txa = g.xa.sum()
            games = max(int(g.games.max() or 0), 1)
            w = min(1.0, games / 15)
            pv = prev[prev.team == t]
            pshare = (pv.set_index("player").xa / pv.xa.sum()) if len(pv) and pv.xa.sum() > 0 else pd.Series(dtype=float)
            # asistencias reales ≈ 0,8 × xA en LaLiga; se usa el ritmo de goles esperados del equipo
            for r in g.itertuples():
                sh = (r.xa / txa) if txa > 0 else 0.0
                if r.player in pshare.index:
                    sh = w * sh + (1 - w) * float(pshare[r.player])
                rows.append((r.player, t, int(r.assists or 0), sh * dc_left.get(t, 0) * 0.7))
        df = pd.DataFrame(rows, columns=["name", "team", "a", "left"])
        if len(df):
            df["exp"] = df.a + df.left
            cand = df.sort_values("exp", ascending=False).head(30).reset_index(drop=True)
            rng = np.random.default_rng(seed)
            S = cand.a.to_numpy()[None, :] + rng.poisson(cand.left.to_numpy()[None, :], size=(10000, len(cand)))
            w = (S == S.max(1, keepdims=True)).astype(float)
            cand["p"] = (w / w.sum(1, keepdims=True)).mean(0)
            out["assists"] = [{"name": r.name, "team": r.team, "assists": int(r.a), "exp": _r(r.exp, 1), "p": _r(r.p, 4)}
                              for r in cand.sort_values(["p", "exp"], ascending=False).head(top).itertuples()]
    return out


# ---------------------------------------------------------------------------
# Previas automáticas (texto a partir de los datos; sin inventar nada)
# ---------------------------------------------------------------------------
def _pct(x) -> str:
    return f"{100 * x:.0f} %"


def preview(p: dict, team_info: dict, names: dict | None = None) -> str:
    """Texto corto de previa de un partido con los datos de la predicción."""
    nm = (names or {}).get
    h, a = p["home"], p["away"]
    H, A = nm(h, h), nm(a, a)
    th, ta = team_info.get(h, {}), team_info.get(a, {})
    ph, pdw, pa = p["p_home"], p["p_draw"], p["p_away"]
    fav = H if ph > pa else A
    fp = max(ph, pa)
    parts = []
    if fp >= 0.6:
        parts.append(f"{fav} es claro favorito ({_pct(fp)}).")
    elif fp >= 0.45:
        parts.append(f"{fav} parte con ventaja ({_pct(fp)}), con el empate en {_pct(pdw)}.")
    else:
        parts.append(f"Partido muy abierto: {_pct(ph)} / {_pct(pdw)} / {_pct(pa)}.")
    if th.get("pos") and ta.get("pos"):
        parts.append(f"{H} llega {th['pos']}.º con {th['pts']} puntos y {A} {ta['pos']}.º con {ta['pts']}.")
    form = lambda t: "".join(g["res"] for g in (t.get("last5") or []))  # noqa: E731
    if form(th) and form(ta):
        parts.append(f"Últimos cinco: {H} {form(th)}, {A} {form(ta)}.")
    xg = p.get("dc") or {}
    if xg.get("xg_home") is not None:
        parts.append(f"Goles esperados {xg['xg_home']:.1f}–{xg['xg_away']:.1f}; más de 2,5 goles {_pct((p.get('ou') or {}).get('p_over', xg.get('p_over25', 0)) or 0)}.")
    abs_ = p.get("absences") or {}
    for t, T in ((h, H), (a, A)):
        out = [x["player"] for x in abs_.get(t, []) if x.get("status") == "out"][:3]
        if out:
            parts.append(f"Bajas de {T}: {', '.join(out)}.")
    sc = p.get("scorers") or {}
    top = []
    for side in ("home", "away"):
        lst = sc.get(side) or []
        if lst:
            top.append(f"{lst[0]['name']} ({_pct(lst[0]['p'])})")
    if top:
        parts.append(f"Goleadores más probables: {' y '.join(top)}.")
    pm = p.get("polymarket") or {}
    if pm.get("p"):
        parts.append(f"Polymarket: {_pct(pm['p'][0])} / {_pct(pm['p'][1])} / {_pct(pm['p'][2])}.")
    if p.get("referee"):
        parts.append(f"Arbitra {p['referee']}.")
    return " ".join(parts)
