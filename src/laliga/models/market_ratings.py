"""Fuerza de cada equipo deducida de las cuotas de cierre del mercado.

Para cada partido ya jugado, las cuotas de cierre (1X2 sin margen y, si las hay, más/menos de 2,5)
dicen qué goles esperaba el mercado de cada equipo (λ local, μ visitante): se buscan en una tabla
precalculada de marcadores (Poisson con la corrección de Dixon-Coles). Con esos goles esperados se
ajusta un modelo log-lineal con ataque y defensa por equipo y ventaja de campo:

    log λ = c + campo + ataque(local) − defensa(visitante)
    log μ = c + ataque(visitante) − defensa(local)

por mínimos cuadrados ponderados (los partidos recientes pesan más) con una pequeña regularización
hacia cero (o hacia la previa de recién ascendido). El resultado se usa igual que Dixon-Coles
(`expected_goals`, `predict`), pero resume lo que sabía el mercado: fichajes, lesiones largas,
cambios de entrenador… y no solo los goles que han ocurrido.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.stats import poisson

from .dixon_coles import score_matrix, summarize_matrix
from .market import devig_power

RHO = -0.05


@lru_cache(maxsize=1)
def _grid():
    """Tabla (λ, μ) → (P1, PX, P2, P más de 2,5) y su árbol de búsqueda."""
    from scipy.spatial import cKDTree
    vals = np.arange(0.15, 4.01, 0.025)
    L, M = np.meshgrid(vals, vals, indexing="ij")
    L, M = L.ravel(), M.ravel()
    g = np.arange(11)
    ph = poisson.pmf(g[None, :], L[:, None])
    pa = poisson.pmf(g[None, :], M[:, None])
    m = ph[:, :, None] * pa[:, None, :]
    m[:, 0, 0] *= np.clip(1 - L * M * RHO, 1e-10, None)
    m[:, 0, 1] *= 1 + L * RHO
    m[:, 1, 0] *= 1 + M * RHO
    m[:, 1, 1] *= 1 - RHO
    m /= m.sum((1, 2), keepdims=True)
    i, j = np.indices((11, 11))
    p1 = m[:, i > j].sum(1)
    p2 = m[:, i < j].sum(1)
    po = m[:, (i + j) > 2].sum(1)
    feats = np.column_stack([_logit(p1), _logit(p2), _logit(po)])
    return L, M, cKDTree(feats), cKDTree(feats[:, :2]), feats


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def implied_goals(p1x2: np.ndarray, p_over: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Goles esperados (λ, μ) coherentes con las probabilidades del mercado. Sin más/menos se toma el
    par con ese 1X2 más cercano a un total típico (2,6 goles)."""
    L, M, tree3, tree2, feats = _grid()
    p1x2 = np.asarray(p1x2, float)
    q = np.column_stack([_logit(p1x2[:, 0]), _logit(p1x2[:, 2])])
    lam, mu = np.full(len(q), np.nan), np.full(len(q), np.nan)
    ok = np.isfinite(q).all(1)
    has_o = ok & (np.isfinite(p_over) if p_over is not None else np.zeros(len(q), bool))
    if has_o.any():
        _, idx = tree3.query(np.column_stack([q[has_o], _logit(np.asarray(p_over, float)[has_o])]))
        lam[has_o], mu[has_o] = L[idx], M[idx]
    rest = ok & ~has_o
    if rest.any():
        _, idx = tree2.query(q[rest], k=40)  # los 40 pares más parecidos en 1X2: el de total más típico
        tot = L[idx] + M[idx]
        best = idx[np.arange(len(idx)), np.argmin(np.abs(tot - 2.6), axis=1)]
        lam[rest], mu[rest] = L[best], M[best]
    return lam, mu


def market_probs(matches: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """1X2 sin margen (cierre; si no, apertura) y P(más de 2,5) del cierre (si no, de antes)."""
    def cols(prefix):
        return matches[[f"{prefix}_h", f"{prefix}_d", f"{prefix}_a"]].to_numpy(float)
    close = devig_power(cols("odds_close")) if "odds_close_h" in matches else np.full((len(matches), 3), np.nan)
    pre = devig_power(cols("odds")) if "odds_h" in matches else np.full((len(matches), 3), np.nan)
    p = np.where(np.isfinite(close).all(1, keepdims=True), close, pre)
    po = np.full(len(matches), np.nan)
    for o, u in (("ouc_o25", "ouc_u25"), ("ou_o25", "ou_u25")):
        if o in matches:
            a, b = matches[o].to_numpy(float), matches[u].to_numpy(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                v = np.where((a > 1) & (b > 1), (1 / a) / (1 / a + 1 / b), np.nan)
            po = np.where(np.isfinite(po), po, v)
    return p, po


@dataclass
class _Params:
    teams: list
    attack: np.ndarray
    defence: np.ndarray
    home_adv: float
    intercept: float
    rho: float = RHO
    n: dict = field(default_factory=dict)


class MarketRatings:
    """Misma interfaz que DixonColes para lo que usan la proyección y el Pichichi."""

    def __init__(self, xi: float = 0.008, ridge: float = 2.0, window_days: int = 700,
                 new_team_attack: float = -0.2, new_team_defence: float = -0.15, max_goals: int = 10):
        self.xi, self.ridge, self.window_days = xi, ridge, window_days
        self.new_att, self.new_def = new_team_attack, new_team_defence
        self.max_goals = max_goals
        self.params: _Params | None = None

    @classmethod
    def from_settings(cls, settings) -> "MarketRatings":
        c = settings.get("market_ratings", "xi", None)
        return cls(xi=float(c) if c is not None else 0.008,
                   ridge=float(settings.get("market_ratings", "ridge", 2.0)))

    def fit(self, matches: pd.DataFrame, ref_date, current_teams=None, new_teams=None) -> "MarketRatings":
        ref = pd.Timestamp(ref_date)
        d = matches[(matches.date < ref) & (matches.date >= ref - pd.Timedelta(days=self.window_days))].copy()
        p, po = market_probs(d)
        lam, mu = implied_goals(p, po)
        ok = np.isfinite(lam) & np.isfinite(mu)
        d, lam, mu = d[ok], lam[ok], mu[ok]
        teams = sorted(set(d.home) | set(d.away) | set(current_teams or []))
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        new_teams = set(new_teams or set())
        w = np.exp(-self.xi * (ref - d.date).dt.days.to_numpy(float))
        hi, ai = d.home.map(idx).to_numpy(), d.away.map(idx).to_numpy()
        # incógnitas: [ataque (n), defensa (n), campo, constante]
        k = 2 * n + 2
        rows = len(d)
        X = np.zeros((2 * rows, k))
        r = np.arange(rows)
        X[r, hi] = 1
        X[r, n + ai] = -1
        X[r, 2 * n] = 1
        X[r, 2 * n + 1] = 1
        X[rows + r, ai] = 1
        X[rows + r, n + hi] = -1
        X[rows + r, 2 * n + 1] = 1
        y = np.concatenate([np.log(lam), np.log(mu)])
        W = np.concatenate([w, w])
        # previa: 0 para todos; recién ascendidos, la previa de ascendido (pesa como unos pocos partidos)
        prior = np.zeros(k)
        for t in new_teams & set(teams):
            prior[idx[t]], prior[n + idx[t]] = self.new_att, self.new_def
        R = np.zeros(k)
        R[:2 * n] = self.ridge
        A = X.T @ (X * W[:, None]) + np.diag(R)
        b = X.T @ (W * y) + R * prior
        # identificabilidad: media del ataque = 0 (restricción blanda)
        cons = np.zeros(k)
        cons[:n] = 1.0
        A += 10.0 * np.outer(cons, cons)
        theta = np.linalg.solve(A, b)
        counts = pd.concat([d.home, d.away]).value_counts().to_dict()
        self.params = _Params(teams, theta[:n], theta[n:2 * n], float(theta[2 * n]), float(theta[2 * n + 1]),
                              n={t: int(counts.get(t, 0)) for t in teams})
        return self

    def _ad(self, t: str) -> tuple[float, float]:
        p = self.params
        if t in p.teams:
            i = p.teams.index(t)
            return float(p.attack[i]), float(p.defence[i])
        return self.new_att, self.new_def

    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        p = self.params
        ah, dh = self._ad(home)
        aa, da = self._ad(away)
        return (float(np.exp(p.intercept + p.home_adv + ah - da)), float(np.exp(p.intercept + aa - dh)))

    def predict(self, home: str, away: str) -> dict:
        lam, mu = self.expected_goals(home, away)
        return summarize_matrix(score_matrix(lam, mu, RHO, self.max_goals), lam, mu)

    def strength(self, team: str) -> float:
        a, d = self._ad(team)
        return a + d


# ---------------------------------------------------------------------------
# Medición frente a Dixon-Coles
# ---------------------------------------------------------------------------
def evaluate_matches(settings, matches: pd.DataFrame, seasons: list[int], horizons=(7, 28)) -> dict:
    """Log-loss al predecir los partidos de los próximos días SIN usar sus cuotas (como pasa con la
    jornada siguiente o con la proyección): ratings del mercado frente a Dixon-Coles y su mezcla."""
    from .dixon_coles import DixonColes
    from .. import pipeline
    out = {}
    for hz in horizons:
        P_dc, P_mr, Y = [], [], []
        for s in seasons:
            sm = matches[(matches.season == s) & matches.hg.notna()].sort_values("date")
            if len(sm) < 300:
                continue
            refs = pd.date_range(sm.date.min() + pd.Timedelta(days=21), sm.date.max(), freq="14D")
            teams = sorted(set(sm.home) | set(sm.away))
            new = pipeline._new_teams(matches, s)
            for ref in refs:
                blk = sm[(sm.date >= ref + pd.Timedelta(days=hz - 7)) & (sm.date < ref + pd.Timedelta(days=hz))]
                if blk.empty:
                    continue
                dc = DixonColes.from_settings(settings).fit(matches, ref, teams, new)
                mr = MarketRatings.from_settings(settings).fit(matches, ref, teams, new)
                for r in blk.itertuples():
                    a = dc.predict(r.home, r.away)
                    b = mr.predict(r.home, r.away)
                    P_dc.append([a["p_home"], a["p_draw"], a["p_away"]])
                    P_mr.append([b["p_home"], b["p_draw"], b["p_away"]])
                    Y.append(0 if r.hg > r.ag else 1 if r.hg == r.ag else 2)
        if not Y:
            continue
        A, B, y = np.array(P_dc), np.array(P_mr), np.array(Y)

        def ll(p):
            return round(float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1)))), 4)
        grid = np.round(np.arange(0, 1.01, 0.1), 1)
        sc = [ll(w * B + (1 - w) * A) for w in grid]
        k = int(np.argmin(sc))
        out[f"{hz - 7}-{hz}d"] = {"n": int(len(y)), "dixon_coles": ll(A), "mercado": ll(B), "mezcla": sc[k],
                                  "w_mercado": float(grid[k])}
    return out


class BlendRates:
    """Goles esperados combinados: (ratings del mercado)^w · (Dixon-Coles)^(1−w)."""

    def __init__(self, dc, mr, w: float):
        self.dc, self.mr, self.w = dc, mr, float(w)

    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        a = self.dc.expected_goals(home, away)
        b = self.mr.expected_goals(home, away)
        return (float(b[0] ** self.w * a[0] ** (1 - self.w)), float(b[1] ** self.w * a[1] ** (1 - self.w)))

    def predict(self, home: str, away: str) -> dict:
        lam, mu = self.expected_goals(home, away)
        return summarize_matrix(score_matrix(lam, mu, RHO, 10), lam, mu)


def market_weight(settings) -> float:
    """Peso de los ratings del mercado medido (`evaluate_matches`, guardado en market_ratings.json)."""
    import json
    p = settings.state_dir / "market_ratings.json"
    try:
        ev = json.loads(p.read_text(encoding="utf-8"))
        return float(ev["w_mercado"])
    except (OSError, ValueError, KeyError):
        return float(settings.get("market_ratings", "weight", 0.9))


def rates_model(settings, matches: pd.DataFrame, ref, teams, new_teams, w: float | None = None):
    """El modelo de goles para lo que no tiene cuotas propias (proyección, Pichichi, simulador)."""
    from .dixon_coles import DixonColes
    w = market_weight(settings) if w is None else w
    dc = DixonColes.from_settings(settings).fit(matches, ref, teams, new_teams)
    if w <= 0:
        return dc
    try:
        mr = MarketRatings.from_settings(settings).fit(matches, ref, teams, new_teams)
    except Exception:
        return dc
    if not mr.params or len(mr.params.teams) < 10:
        return dc
    return mr if w >= 1 else BlendRates(dc, mr, w)


def evaluate_no_odds(matches: pd.DataFrame, bt: pd.DataFrame) -> dict:
    """Partidos sin cuotas todavía: el meta-modelo sin mercado (backtest) frente a los ratings del
    mercado ajustados dos días antes de cada jornada, y la mejor mezcla."""
    from .. import pipeline
    x = matches.loc[matches.index.intersection(bt.index)].copy()
    if x.empty:
        return {}
    x["wk"] = x.date.dt.to_period("W")
    P, S, Y = [], [], []
    for _, g in x.groupby("wk"):
        ref = g.date.min() - pd.Timedelta(days=2)
        season = int(g.season.iloc[0])
        teams = sorted(set(matches[matches.season == season].home))
        mr = MarketRatings().fit(matches, ref, teams, pipeline._new_teams(matches, season))
        for i, r in g.iterrows():
            q = mr.predict(r.home, r.away)
            P.append([q["p_home"], q["p_draw"], q["p_away"]])
            S.append(bt.loc[i, ["stacker_H", "stacker_D", "stacker_A"]].to_numpy(float))
            Y.append(int(bt.loc[i, "y"]))
    P, S, Y = np.array(P), np.array(S), np.array(Y)

    def ll(p):
        return round(float(-np.mean(np.log(np.clip(p[np.arange(len(Y)), Y], 1e-9, 1)))), 4)
    grid = np.round(np.arange(0, 1.01, 0.05), 2)
    sc = [ll(w * P + (1 - w) * S) for w in grid]
    k = int(np.argmin(sc))
    return {"n": int(len(Y)), "modelo_sin_cuotas": ll(S), "ratings_mercado": ll(P), "mezcla": sc[k], "w": float(grid[k])}


def no_odds_weight(settings) -> float:
    import json
    try:
        ev = json.loads((settings.state_dir / "market_ratings.json").read_text(encoding="utf-8"))
        return float(ev["sin_cuotas"]["w"])
    except (OSError, ValueError, KeyError, TypeError):
        return 0.75


def run_evaluation(settings, matches: pd.DataFrame, seasons: list[int]) -> dict:
    import json
    from datetime import datetime, timezone
    ev = evaluate_matches(settings, matches, seasons)
    if not ev:
        return {}
    main = ev.get("0-7d") or next(iter(ev.values()))
    out = {"by_horizon": ev, "w_mercado": main["w_mercado"], "seasons": seasons,
           "updated": datetime.now(timezone.utc).date().isoformat()}
    bp = settings.processed_dir / "backtest_predictions.csv"
    if bp.exists():
        try:
            out["sin_cuotas"] = evaluate_no_odds(matches, pd.read_csv(bp, index_col=0))
        except Exception:
            pass
    (settings.state_dir / "market_ratings.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out
