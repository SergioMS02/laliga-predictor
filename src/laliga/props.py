"""Tarjetas y córners esperados de cada partido (y quién tiene más papeletas de ver amarilla).

Modelo multiplicativo con medias recientes encogidas hacia la media de la liga:
  amarillas del local = media de liga en casa × tendencia del equipo a ver tarjetas ×
                        lo que provoca el rival × lo que saca el árbitro × derbi
(lo mismo para el visitante y para los córners, sin árbitro). Ventana de ~1,5 temporadas con
decaimiento temporal; los equipos y árbitros con pocos partidos se acercan a la media.
Fuente: football-data.co.uk (tarjetas, córners y árbitro de cada partido desde 2005).
"""
from __future__ import annotations

import math
import re
import unicodedata

import numpy as np
import pandas as pd

WINDOW_DAYS = 550
HALF_LIFE = 240.0
K_TEAM, K_REF = 8.0, 12.0
DERBY_CARDS = 1.12


def _plain(s) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z ]+", " ", s).strip()


def ref_key(name) -> str | None:
    """Clave del árbitro: apellido principal (football-data usa "J. Munuera"; ESPN "Juan Martínez Munuera")."""
    if not isinstance(name, str) or not name.strip():
        return None
    toks = [t for t in _plain(name).split() if len(t) > 2 and t not in ("del", "las", "los")]
    return toks[-1] if toks else None


def _wmean(v: np.ndarray, w: np.ndarray) -> float:
    return float((v * w).sum() / w.sum()) if w.sum() > 0 else float("nan")


class PropsModel:
    def __init__(self, matches: pd.DataFrame, ref_date: pd.Timestamp):
        d = matches[(matches.date < ref_date) & (matches.date >= ref_date - pd.Timedelta(days=WINDOW_DAYS))]
        d = d.dropna(subset=["h_yellow", "a_yellow", "h_corners", "a_corners"])
        self.ok = len(d) >= 100
        if not self.ok:
            return
        age = (ref_date - d.date).dt.days.to_numpy(float)
        w = 0.5 ** (age / HALF_LIFE)
        self.d, self.w = d, w
        self.avg = {
            "hy": _wmean(d.h_yellow.to_numpy(float), w), "ay": _wmean(d.a_yellow.to_numpy(float), w),
            "hc": _wmean(d.h_corners.to_numpy(float), w), "ac": _wmean(d.a_corners.to_numpy(float), w),
            "hr": _wmean(d.h_red.fillna(0).to_numpy(float), w), "ar": _wmean(d.a_red.fillna(0).to_numpy(float), w),
        }
        self.avg["y"] = (self.avg["hy"] + self.avg["ay"]) / 2
        self.avg["c"] = (self.avg["hc"] + self.avg["ac"]) / 2
        self.refkeys = d.referee.map(ref_key) if "referee" in d else pd.Series(None, index=d.index)

    def _team_factor(self, team: str, own: str, opp: str, avg: float, k: float = K_TEAM) -> float:
        """own/opp: columnas ('h_yellow','a_yellow')... factor del equipo encogido hacia 1."""
        d, w = self.d, self.w
        hm, am = (d.home == team).to_numpy(), (d.away == team).to_numpy()
        vals = np.concatenate([d[own[0]].to_numpy(float)[hm] / self.avg[own[2]], d[own[1]].to_numpy(float)[am] / self.avg[own[3]]])
        ww = np.concatenate([w[hm], w[am]])
        if ww.sum() == 0:
            return 1.0
        n_eff = ww.sum()
        return float((vals * ww).sum() + k) / (n_eff + k)

    def _ref_factor(self, referee: str | None) -> tuple[float, int]:
        key = ref_key(referee)
        if key is None:
            return 1.0, 0
        m = (self.refkeys == key).to_numpy()
        if m.sum() == 0:
            return 1.0, 0
        tot = (self.d.h_yellow.to_numpy(float) + self.d.a_yellow.to_numpy(float))[m] / (2 * self.avg["y"])
        ww = self.w[m]
        return float(((tot * ww).sum() + K_REF) / (ww.sum() + K_REF)), int(m.sum())

    def predict(self, home: str, away: str, referee: str | None = None, derby: bool = False) -> dict | None:
        if not self.ok:
            return None
        # tarjetas que ve cada equipo y que provoca en el rival
        th = self._team_factor(home, ("h_yellow", "a_yellow", "hy", "ay"), None, self.avg["y"])
        ta = self._team_factor(away, ("h_yellow", "a_yellow", "hy", "ay"), None, self.avg["y"])
        ph = self._team_factor(home, ("a_yellow", "h_yellow", "ay", "hy"), None, self.avg["y"])  # provoca (rival)
        pa = self._team_factor(away, ("a_yellow", "h_yellow", "ay", "hy"), None, self.avg["y"])
        rf, n_ref = self._ref_factor(referee)
        dz = DERBY_CARDS if derby else 1.0
        yh = self.avg["hy"] * th * pa * rf * dz
        ya = self.avg["ay"] * ta * ph * rf * dz
        # córners a favor y en contra
        ch = self._team_factor(home, ("h_corners", "a_corners", "hc", "ac"), None, self.avg["c"])
        ca = self._team_factor(away, ("h_corners", "a_corners", "hc", "ac"), None, self.avg["c"])
        dh = self._team_factor(home, ("a_corners", "h_corners", "ac", "hc"), None, self.avg["c"])  # concede
        da = self._team_factor(away, ("a_corners", "h_corners", "ac", "hc"), None, self.avg["c"])
        kh = self.avg["hc"] * ch * da
        ka = self.avg["ac"] * ca * dh
        red = (self.avg["hr"] + self.avg["ar"]) * rf
        ty, tc = yh + ya, kh + ka
        return {
            "yellow": [round(yh, 2), round(ya, 2)], "yellow_total": round(ty, 2),
            "p_cards_over": {str(x): round(1 - _pois_cdf(math.floor(x), ty), 3) for x in (3.5, 4.5, 5.5)},
            "corners": [round(kh, 2), round(ka, 2)], "corners_total": round(tc, 2),
            "p_corners_over": {str(x): round(1 - _pois_cdf(math.floor(x), tc), 3) for x in (8.5, 9.5, 10.5)},
            "p_red": round(1 - math.exp(-red), 3),
            "referee_factor": round(rf, 2), "referee_matches": n_ref,
        }


def _pois_cdf(k: int, lam: float) -> float:
    s, p = 0.0, math.exp(-lam)
    for i in range(k + 1):
        s += p
        p *= lam / (i + 1)
    return min(1.0, s)


def player_card_probs(cards: pd.DataFrame | None, lineups: pd.DataFrame | None, team: str, season_start: pd.Timestamp,
                      xi: list[str], team_yellow_exp: float, team_yellow_avg: float, top: int = 4) -> list[dict]:
    """Probabilidad de que cada titular vea amarilla: sus amarillas por partido como titular esta
    temporada (encogidas hacia la media del equipo), escaladas por las tarjetas esperadas de su equipo."""
    if cards is None or cards.empty or not xi:
        return []
    c = cards[(cards.team == team) & (cards.date >= season_start) & (cards.card == "yellow")]
    apps = {}
    if lineups is not None and len(lineups):
        lu = lineups[(lineups.team == team) & (pd.to_datetime(lineups.date) >= season_start) & lineups.starter.astype(bool)]
        apps = lu.groupby("name").event_id.nunique().to_dict()
    n_games = max(1, int(lineups[(lineups.team == team) & (pd.to_datetime(lineups.date) >= season_start)].event_id.nunique())
                  if lineups is not None and len(lineups) else 1)
    base = team_yellow_avg / 11.0  # amarillas por jugador y partido en un equipo medio
    from rapidfuzz import fuzz, process
    names = list(set(c.player) | set(apps))
    out = []
    for name in xi:
        best = process.extractOne(_plain(name), {n: _plain(n) for n in names}, scorer=fuzz.token_set_ratio) if names else None
        key = best[2] if best and best[1] >= 85 else None
        y = int((c.player == key).sum()) if key else 0
        a = apps.get(key, 0) if key else 0
        a = max(a, min(y, n_games))
        rate = (y + 4 * base) / (a + 4)  # encogido con 4 partidos de la media
        out.append({"name": name, "rate": rate, "yellows": y})
    # las amarillas del once suman las esperadas del equipo (el 90 %: el resto, suplentes y banquillo)
    tot = sum(o["rate"] for o in out) or 1.0
    k = 0.9 * team_yellow_exp / tot
    for o in out:
        o["p"] = round(1 - math.exp(-o.pop("rate") * k), 3)
    out.sort(key=lambda d: -d["p"])
    return out[:top]


def evaluate(matches: pd.DataFrame, seasons: list[int]) -> dict:
    """Log-verosimilitud Poisson del modelo frente a usar la media de la liga (temporadas de prueba)."""
    rows = []
    for s in seasons:
        sm = matches[(matches.season == s)].dropna(subset=["h_yellow", "a_yellow", "h_corners", "a_corners"])
        for wk, blk in sm.groupby(sm.date.dt.to_period("W")):
            ref = blk.date.min()
            pm = PropsModel(matches, ref)
            if not pm.ok:
                continue
            for r in blk.itertuples():
                p = pm.predict(r.home, r.away, r.referee if isinstance(r.referee, str) else None)
                rows.append((p["yellow_total"], (pm.avg["hy"] + pm.avg["ay"]), r.h_yellow + r.a_yellow,
                             p["corners_total"], pm.avg["hc"] + pm.avg["ac"], r.h_corners + r.a_corners))
    if not rows:
        return {}
    a = np.array(rows, float)

    def ll(lam, k):
        return float(np.mean(k * np.log(lam) - lam - np.array([math.lgamma(x + 1) for x in k])))
    return {"n": len(a),
            "cards": {"modelo": round(ll(a[:, 0], a[:, 2]), 4), "media_liga": round(ll(a[:, 1], a[:, 2]), 4),
                      "mae_modelo": round(float(np.abs(a[:, 0] - a[:, 2]).mean()), 3),
                      "mae_media": round(float(np.abs(a[:, 1] - a[:, 2]).mean()), 3)},
            "corners": {"modelo": round(ll(a[:, 3], a[:, 5]), 4), "media_liga": round(ll(a[:, 4], a[:, 5]), 4),
                        "mae_modelo": round(float(np.abs(a[:, 3] - a[:, 5]).mean()), 3),
                        "mae_media": round(float(np.abs(a[:, 4] - a[:, 5]).mean()), 3)}}
