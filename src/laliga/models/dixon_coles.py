"""Modelo Dixon-Coles (1997) con ponderación temporal exponencial y regularización.

λ (goles local)     = exp(c + γ + ataque_local + defensa_visitante)
μ (goles visitante) = exp(c + ataque_visitante + defensa_local)
c = nivel medio de la liga (sin regularizar), de modo que la regularización y los
priors de ascendidos son relativos a la media y el modelo sirve también para tiros.
Corrección τ(ρ) para marcadores bajos. Gradiente analítico -> ajuste en milisegundos.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson


@dataclass
class DCParams:
    teams: list[str]
    attack: np.ndarray
    defence: np.ndarray
    home_adv: float
    rho: float
    intercept: float = 0.0
    prior_attack: dict[str, float] = field(default_factory=dict)
    prior_defence: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({"teams": self.teams, "attack": self.attack.tolist(), "defence": self.defence.tolist(),
                           "home_adv": self.home_adv, "rho": self.rho, "intercept": self.intercept,
                           "prior_attack": self.prior_attack, "prior_defence": self.prior_defence})

    @classmethod
    def from_json(cls, s: str) -> "DCParams":
        d = json.loads(s)
        return cls(d["teams"], np.array(d["attack"]), np.array(d["defence"]), d["home_adv"], d["rho"],
                   d.get("intercept", 0.0), d.get("prior_attack", {}), d.get("prior_defence", {}))


def _tau_terms(x, y, lam, mu, rho):
    """Devuelve log τ y sus derivadas respecto a log λ, log μ y ρ."""
    n = len(x)
    logtau = np.zeros(n)
    d_ll = np.zeros(n)
    d_lm = np.zeros(n)
    d_r = np.zeros(n)
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    t = np.maximum(1 - lam[m00] * mu[m00] * rho, 1e-10)
    logtau[m00] = np.log(t)
    d_ll[m00] = -lam[m00] * mu[m00] * rho / t
    d_lm[m00] = d_ll[m00]
    d_r[m00] = -lam[m00] * mu[m00] / t
    t = np.maximum(1 + lam[m01] * rho, 1e-10)
    logtau[m01] = np.log(t)
    d_ll[m01] = lam[m01] * rho / t
    d_r[m01] = lam[m01] / t
    t = np.maximum(1 + mu[m10] * rho, 1e-10)
    logtau[m10] = np.log(t)
    d_lm[m10] = mu[m10] * rho / t
    d_r[m10] = mu[m10] / t
    t = max(1 - rho, 1e-10)
    logtau[m11] = np.log(t)
    d_r[m11] = -1 / t
    return logtau, d_ll, d_lm, d_r


def neg_loglik_and_grad(theta, hi, ai, x, y, w, n, ridge, pa, pd_):
    att, de = theta[:n], theta[n:2 * n]
    gamma, rho, c = theta[2 * n], theta[2 * n + 1], theta[2 * n + 2]
    loglam = c + gamma + att[hi] + de[ai]
    logmu = c + att[ai] + de[hi]
    lam, mu = np.exp(loglam), np.exp(logmu)
    logtau, d_ll, d_lm, d_r = _tau_terms(x, y, lam, mu, rho)
    ll = w * (logtau + x * loglam - lam + y * logmu - mu)
    nll = -ll.sum() + 0.5 * ridge * (((att - pa) ** 2).sum() + ((de - pd_) ** 2).sum())
    g_lam = w * (x - lam + d_ll)
    g_mu = w * (y - mu + d_lm)
    g_att = np.bincount(hi, g_lam, n) + np.bincount(ai, g_mu, n)
    g_def = np.bincount(ai, g_lam, n) + np.bincount(hi, g_mu, n)
    grad = -np.concatenate([g_att, g_def, [g_lam.sum(), (w * d_r).sum(), g_lam.sum() + g_mu.sum()]])
    grad[:n] += ridge * (att - pa)
    grad[n:2 * n] += ridge * (de - pd_)
    return nll, grad


class DixonColes:
    def __init__(self, xi: float = 0.0019, ridge: float = 1.0, window_days: int = 1100,
                 new_team_attack: float = -0.2, new_team_defence: float = 0.15, max_goals: int = 10):
        self.xi, self.ridge, self.window_days = xi, ridge, window_days
        self.new_att, self.new_def = new_team_attack, new_team_defence
        self.max_goals = max_goals
        self.params: DCParams | None = None
        self._warm: dict[str, tuple[float, float]] = {}
        self._warm_global: tuple[float, float] | None = None

    @classmethod
    def from_settings(cls, settings) -> "DixonColes":
        c = settings["dixon_coles"]
        return cls(xi=c["xi"], ridge=c["ridge"], window_days=c["window_days"],
                   new_team_attack=c["new_team_attack"], new_team_defence=c["new_team_defence"],
                   max_goals=c["max_goals"])

    def fit(self, matches: pd.DataFrame, ref_date, current_teams: list[str] | None = None,
            new_teams: set[str] | None = None, cols: tuple[str, str] = ("hg", "ag")) -> "DixonColes":
        """Ajusta con partidos jugados antes de ref_date (estrictamente).

        cols permite ajustar el mismo modelo sobre otro conteo (p. ej. tiros a puerta)."""
        ref = pd.Timestamp(ref_date)
        hc, ac = cols
        d = matches[(matches["date"] < ref) & matches[hc].notna() & matches[ac].notna() &
                    (matches["date"] >= ref - pd.Timedelta(days=self.window_days))]
        teams = sorted(set(d.home) | set(d.away) | set(current_teams or []))
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        new_teams = new_teams or set()
        pa = np.array([self.new_att if t in new_teams else 0.0 for t in teams])
        pd_ = np.array([self.new_def if t in new_teams else 0.0 for t in teams])
        if len(d) == 0:
            self.params = DCParams(teams, pa.copy(), pd_.copy(), 0.25, -0.05, 0.1,
                                   {t: self.new_att for t in new_teams}, {t: self.new_def for t in new_teams})
            return self
        hi = d["home"].map(idx).to_numpy()
        ai = d["away"].map(idx).to_numpy()
        x = d[hc].to_numpy(dtype=float)
        y = d[ac].to_numpy(dtype=float)
        age = (ref - d["date"]).dt.days.to_numpy(dtype=float)
        w = np.exp(-self.xi * age)
        level = np.log(max((w * (x + y)).sum() / (2 * w.sum()), 0.05))
        theta0 = np.concatenate([
            [self._warm.get(t, (pa[i], 0))[0] for i, t in enumerate(teams)],
            [self._warm.get(t, (0, pd_[i]))[1] for i, t in enumerate(teams)],
            list(self._warm_global or (0.25, -0.05)), [level],
        ])
        bounds = [(-3, 3)] * (2 * n) + [(-1, 1), (-0.25, 0.25), (-5, 5)]
        res = minimize(neg_loglik_and_grad, theta0, jac=True, method="L-BFGS-B", bounds=bounds,
                       args=(hi, ai, x, y, w, n, self.ridge, pa, pd_), options={"maxiter": 500})
        th = res.x
        self.params = DCParams(teams, th[:n], th[n:2 * n], float(th[2 * n]), float(th[2 * n + 1]),
                               float(th[2 * n + 2]), {t: self.new_att for t in new_teams}, {t: self.new_def for t in new_teams})
        self._warm = {t: (th[i], th[n + i]) for i, t in enumerate(teams)}
        self._warm_global = (float(th[2 * n]), float(th[2 * n + 1]))
        return self

    # ------------------------------------------------------------------
    def team_strength(self, team: str) -> tuple[float, float]:
        p = self.params
        assert p is not None, "modelo sin ajustar"
        if team in p.teams:
            i = p.teams.index(team)
            return float(p.attack[i]), float(p.defence[i])
        return p.prior_attack.get(team, self.new_att), p.prior_defence.get(team, self.new_def)

    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        ah, dh = self.team_strength(home)
        aa, da = self.team_strength(away)
        c = self.params.intercept
        return float(np.exp(c + self.params.home_adv + ah + da)), float(np.exp(c + aa + dh))

    def score_matrix(self, lam: float, mu: float) -> np.ndarray:
        return score_matrix(lam, mu, self.params.rho, self.max_goals)

    def predict(self, home: str, away: str, lam: float | None = None, mu: float | None = None) -> dict:
        if lam is None or mu is None:
            lam, mu = self.expected_goals(home, away)
        return summarize_matrix(self.score_matrix(lam, mu), lam, mu)


def score_matrix(lam: float, mu: float, rho: float, max_goals: int = 10) -> np.ndarray:
    g = np.arange(max_goals + 1)
    m = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
    m[0, 0] *= max(1 - lam * mu * rho, 1e-10)
    m[0, 1] *= 1 + lam * rho
    m[1, 0] *= 1 + mu * rho
    m[1, 1] *= 1 - rho
    return m / m.sum()


def summarize_matrix(m: np.ndarray, lam: float, mu: float) -> dict:
    i, j = np.indices(m.shape)
    total = i + j
    flat = np.argsort(m.ravel())[::-1][:3]
    top = [(f"{a}-{b}", float(m[a, b])) for a, b in zip(*np.unravel_index(flat, m.shape))]
    return {
        "p_home": float(m[i > j].sum()), "p_draw": float(m[i == j].sum()), "p_away": float(m[i < j].sum()),
        "xg_home": lam, "xg_away": mu,
        "p_over25": float(m[total > 2.5].sum()), "p_btts": float(m[(i > 0) & (j > 0)].sum()),
        "top_scores": top,
    }
