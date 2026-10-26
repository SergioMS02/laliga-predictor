"""Mercados derivados de la matriz de marcadores del modelo.

A partir de la probabilidad de cada marcador (Dixon-Coles con los goles esperados del partido) se
calculan: doble oportunidad, hándicap asiático (líneas enteras, medias y de cuarto), más/menos de
0,5 a 4,5 goles, goles de cada equipo, ambos marcan, margen de victoria, marcador exacto, resultado
al descanso y descanso/final, y probabilidades del Quinigol (0, 1, 2 o M goles por equipo).

Descanso: los goles de cada parte siguen Poisson independientes con la fracción de goles de la
primera parte medida en LaLiga (`HT_SHARE`, recalculada con `ht_share` si hay datos).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import poisson

HT_SHARE = 0.445  # fracción de los goles que llegan antes del descanso (LaLiga, 2015-2025)
AH_LINES = (-2.5, -2.25, -2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)


def ht_share(matches) -> float:
    """Fracción de goles de la primera parte en los partidos con datos de descanso."""
    try:
        d = matches.dropna(subset=["h_ht", "a_ht", "hg", "ag"])
        if len(d) < 500:
            return HT_SHARE
        return float((d.h_ht.sum() + d.a_ht.sum()) / (d.hg.sum() + d.ag.sum()))
    except (AttributeError, KeyError):
        return HT_SHARE


def _ah_side(diff: np.ndarray, probs: np.ndarray, line: float) -> tuple[float, float, float]:
    """Resultado de apostar al local con hándicap `line`: (gana, devuelve, pierde) en valor esperado
    de unidades «enteras». Las líneas de cuarto se reparten a medias entre las dos líneas vecinas."""
    if abs(line * 4) % 2 == 1:  # cuarto: mitad en cada línea
        a = _ah_side(diff, probs, line - 0.25)
        b = _ah_side(diff, probs, line + 0.25)
        return tuple((x + y) / 2 for x, y in zip(a, b))
    adj = diff + line
    return float(probs[adj > 0].sum()), float(probs[adj == 0].sum()), float(probs[adj < 0].sum())


def ah_fair_odds(diff: np.ndarray, probs: np.ndarray, line: float) -> tuple[float | None, float | None]:
    """Cuota justa del local y del visitante con hándicap `line` (para el local)."""
    w, p, l_ = _ah_side(diff, probs, line)
    # cuota justa o: w·(o−1) − l = 0 (la devolución no cuenta)
    oh = 1 + l_ / w if w > 0 else None
    oa = 1 + w / l_ if l_ > 0 else None
    return (round(oh, 3) if oh else None, round(oa, 3) if oa else None)


def from_matrix(m: np.ndarray, lam: float, mu: float, rho: float = 0.0, share: float = HT_SHARE,
                top_scores: int = 8) -> dict:
    m = np.asarray(m, float)
    m = m / m.sum()
    i, j = np.indices(m.shape)
    tot = i + j
    diff = i - j
    p1, px, p2 = float(m[i > j].sum()), float(m[i == j].sum()), float(m[i < j].sum())
    out = {
        "double_chance": {"1X": round(p1 + px, 4), "X2": round(px + p2, 4), "12": round(p1 + p2, 4)},
        "draw_no_bet": {"1": round(p1 / (p1 + p2), 4), "2": round(p2 / (p1 + p2), 4)},
        "totals": {str(x): round(float(m[tot > x].sum()), 4) for x in (0.5, 1.5, 2.5, 3.5, 4.5)},
        "team_totals": {"home": {str(x): round(float(m[i > x].sum()), 4) for x in (0.5, 1.5, 2.5)},
                        "away": {str(x): round(float(m[j > x].sum()), 4) for x in (0.5, 1.5, 2.5)}},
        "btts": round(float(m[(i > 0) & (j > 0)].sum()), 4),
        "clean_sheet": {"home": round(float(m[j == 0].sum()), 4), "away": round(float(m[i == 0].sum()), 4)},
        "margin": {k: round(float(v), 4) for k, v in (
            ("home3+", m[diff >= 3].sum()), ("home2", m[diff == 2].sum()), ("home1", m[diff == 1].sum()),
            ("draw", px), ("away1", m[diff == -1].sum()), ("away2", m[diff == -2].sum()), ("away3+", m[diff <= -3].sum()))},
    }
    flat = sorted(((float(m[a, b]), f"{a}-{b}") for a in range(min(7, m.shape[0])) for b in range(min(7, m.shape[1]))),
                  reverse=True)[:top_scores]
    out["exact"] = [{"score": s, "p": round(p, 4)} for p, s in flat]
    ah = {}
    for line in AH_LINES:
        w, push, lose = _ah_side(diff, m, line)
        oh, oa = ah_fair_odds(diff, m, line)
        ah[str(line)] = {"win": round(w, 4), "push": round(push, 4), "lose": round(lose, 4), "fair_home": oh, "fair_away": oa}
    out["asian"] = ah
    # descanso y descanso/final (partes independientes)
    g = np.arange(11)
    h1, a1 = poisson.pmf(g, lam * share), poisson.pmf(g, mu * share)
    h2, a2 = poisson.pmf(g, lam * (1 - share)), poisson.pmf(g, mu * (1 - share))
    m1 = np.outer(h1, a1)
    m2 = np.outer(h2, a2)
    res1 = np.sign(i - j)  # 1 local, 0 empate, −1 visitante (al descanso)
    htft = {}
    lab = {1: "1", 0: "X", -1: "2"}
    # resultado final = descanso + segunda parte
    diffs1 = (i - j)
    for k1 in (1, 0, -1):
        mask = res1 == k1
        for k2 in (1, 0, -1):
            p = 0.0
            for dd, pp in zip(diffs1[mask], m1[mask]):
                fin = dd + (i - j)
                p += pp * m2[np.sign(fin) == k2].sum()
            htft[f"{lab[k1]}/{lab[k2]}"] = round(float(p), 4)
    out["half_time"] = {"1": round(float(m1[i > j].sum()), 4), "X": round(float(m1[i == j].sum()), 4),
                        "2": round(float(m1[i < j].sum()), 4), "over_0.5": round(float(1 - m1[0, 0]), 4),
                        "over_1.5": round(float(m1[tot > 1].sum()), 4)}
    out["ht_ft"] = htft
    # Quinigol: 0, 1, 2 o M (3 o más) goles de cada equipo
    cat = np.minimum(i, 3), np.minimum(j, 3)
    q = np.zeros((4, 4))
    for a in range(4):
        for b in range(4):
            q[a, b] = m[(cat[0] == a) & (cat[1] == b)].sum()
    out["quinigol"] = {"home": [round(float(q[a].sum()), 4) for a in range(4)],
                       "away": [round(float(q[:, b].sum()), 4) for b in range(4)],
                       "joint": [[round(float(q[a, b]), 4) for b in range(4)] for a in range(4)]}
    return out


def ah_value(markets: dict, line: float | None, odds_h: float | None, odds_a: float | None) -> dict | None:
    """Compara la línea asiática del mercado (football-data) con el modelo."""
    if line is None or odds_h is None or odds_a is None or not np.isfinite([line, odds_h, odds_a]).all():
        return None
    key = str(float(line))
    a = markets.get("asian", {}).get(key)
    if not a:
        return None
    ev_h = a["win"] * (odds_h - 1) - a["lose"]
    ev_a = a["lose"] * (odds_a - 1) - a["win"]
    return {"line": float(line), "odds": [round(float(odds_h), 3), round(float(odds_a), 3)],
            "fair": [a["fair_home"], a["fair_away"]], "ev": [round(ev_h, 4), round(ev_a, 4)]}
