import numpy as np
import pandas as pd
import pytest
from scipy.optimize import approx_fprime

from laliga import metrics
from laliga.features import availability as av
from laliga.models.dixon_coles import DixonColes, neg_loglik_and_grad
from laliga.models.market import devig_power, devig_proportional, overround


def test_dc_gradient_matches_numeric():
    rng = np.random.default_rng(1)
    n, m = 8, 300
    hi = rng.integers(0, n, m)
    ai = (hi + rng.integers(1, n, m)) % n
    x = rng.poisson(1.4, m).astype(float)
    y = rng.poisson(1.1, m).astype(float)
    w = rng.random(m)
    pa, pd_ = rng.normal(0, .1, n), rng.normal(0, .1, n)
    th = np.concatenate([rng.normal(0, .2, 2 * n), [0.3, -0.08, 0.1]])
    f = lambda t: neg_loglik_and_grad(t, hi, ai, x, y, w, n, 2.0, pa, pd_)[0]
    g = neg_loglik_and_grad(th, hi, ai, x, y, w, n, 2.0, pa, pd_)[1]
    num = approx_fprime(th, f, 1e-7)
    assert np.allclose(g, num, atol=1e-3, rtol=1e-4)


def test_dc_recovers_strengths(league):
    df, strength = league
    dc = DixonColes(xi=0.0005, ridge=0.5, window_days=4000).fit(df, df.date.max() + pd.Timedelta(days=1))
    p = dc.params
    true_att = np.array([strength[t][0] for t in p.teams])
    true_def = np.array([strength[t][1] for t in p.teams])
    # identificable salvo una constante: comparamos centrados
    assert np.corrcoef(p.attack, true_att)[0, 1] > 0.9
    assert np.corrcoef(p.defence, true_def)[0, 1] > 0.85
    assert 0.15 < p.home_adv < 0.35


def test_dc_predictions_valid(league):
    df, _ = league
    dc = DixonColes().fit(df, "2018-01-01")
    r = dc.predict("Barcelona", "Sevilla")
    assert abs(r["p_home"] + r["p_draw"] + r["p_away"] - 1) < 1e-9
    assert 0 < r["p_over25"] < 1 and 0 < r["p_btts"] < 1
    assert len(r["top_scores"]) == 3
    m = dc.score_matrix(1.5, 1.1)
    assert abs(m.sum() - 1) < 1e-12 and (m >= 0).all()


def test_dc_uses_only_past(league):
    df, _ = league
    ref = pd.Timestamp("2016-01-01")
    a = DixonColes().fit(df, ref).predict("Barcelona", "Sevilla")
    future = df.date >= ref
    df2 = df.copy()
    df2.loc[future, ["hg", "ag"]] = df2.loc[future, ["ag", "hg"]].to_numpy() + 5
    b = DixonColes().fit(df2, ref).predict("Barcelona", "Sevilla")
    assert a["p_home"] == pytest.approx(b["p_home"], abs=1e-9)


def test_dc_unknown_team_gets_promoted_prior(league):
    df, _ = league
    dc = DixonColes(new_team_attack=-0.2, new_team_defence=0.15).fit(df, "2016-01-01", new_teams={"Oviedo"})
    assert dc.team_strength("Tenerife") == (-0.2, 0.15)
    lam, mu = dc.expected_goals("Real Madrid", "Tenerife")
    lam2, mu2 = dc.expected_goals("Real Madrid", "Barcelona")
    assert lam > lam2


def test_devig():
    odds = np.array([[1.80, 3.60, 4.50], [2.5, 3.2, 2.9], [np.nan, 3, 3]])
    p = devig_power(odds)
    assert np.allclose(p[:2].sum(axis=1), 1)
    assert np.isnan(p[2]).all()
    pp = devig_proportional(odds[:2])
    # el método de potencias quita relativamente más margen a la sorpresa
    assert p[0, 2] < pp[0, 2] and p[0, 0] > pp[0, 0]
    assert (overround(odds[:2]) > 0).all()


def test_metrics_basics():
    y = np.array([0, 1, 2, 0])
    perfect = np.eye(3)[y]
    unif = np.full((4, 3), 1 / 3)
    assert metrics.log_loss(perfect, y) < 1e-6
    assert metrics.log_loss(unif, y) == pytest.approx(np.log(3))
    assert metrics.brier(perfect, y) == pytest.approx(0)
    assert metrics.rps(perfect, y) == pytest.approx(0)
    # RPS castiga más fallar por dos "escalones" (local vs visitante) que por uno
    assert metrics.rps(np.array([[0, 0, 1.0]]), np.array([0])) > metrics.rps(np.array([[0, 1.0, 0]]), np.array([0]))
    assert metrics.accuracy(perfect, y) == 1
    t = metrics.calibration_table(unif, y)
    assert t.n.sum() == 12


# --------------------------------------------------------------------------
# Disponibilidad (bajas)
# --------------------------------------------------------------------------
def _lineups(n_matches=10):
    rows = []
    for e in range(n_matches):
        for i in range(14):
            rows.append({"event_id": f"e{e}", "date": pd.Timestamp("2026-08-01") + pd.Timedelta(days=7 * e),
                         "team": "Barcelona", "player_id": f"p{i}", "name": f"Jugador {i}", "position": "M",
                         # p0..p9 siempre titulares; p10 la mitad; p11 la otra mitad; p12-13 suplentes
                         "starter": i < 10 or (i == 10 and e % 2 == 0) or (i == 11 and e % 2 == 1)})
    return pd.DataFrame(rows)


def test_player_importance_and_deficits():
    lu = _lineups()
    imp = av.player_importance(lu, "Barcelona", "2027-01-01", lookback=10)
    d = dict(zip(imp.player_id, imp.importance))
    assert d["p0"] == 1.0 and d["p10"] == 0.5 and d["p12"] == 0.0  # suplente conocido = 0
    full = [f"p{i}" for i in range(11)]
    assert av.deficit_from_lineup(imp, full) == pytest.approx(0, abs=1e-9)
    rotated = [f"p{i}" for i in range(5)] + ["p12", "p13", "x1", "x2", "x3", "x4"]
    assert av.deficit_from_lineup(imp, rotated) > 0.3
    ab = pd.DataFrame([{"player": "Jugador 0", "status": "out"}, {"player": "Jugador 1", "status": "doubtful"}])
    dd = av.deficit_from_absences(imp, ab, doubtful_weight=0.5)
    assert dd == pytest.approx(1.5 / imp.importance.head(11).sum())
    # sin historial no se inventa nada
    assert av.deficit_from_lineup(av.player_importance(lu, "Getafe", "2027-01-01"), full) == 0


def test_importance_ignores_future_lineups():
    lu = _lineups()
    imp = av.player_importance(lu, "Barcelona", "2026-08-02")  # solo el primer partido
    assert imp.importance.max() == 1.0 and (imp.importance > 0).sum() == 11


def test_adjust_goals_direction():
    lam, mu = av.adjust_goals(1.5, 1.0, d_home=0.3, d_away=0.0, a=0.6, b=0.35)
    assert lam < 1.5 and mu > 1.0
    assert av.adjust_goals(1.5, 1.0, 0, 0, 0.6, 0.35) == (1.5, 1.0)


def test_calibrate_recovers_effect():
    rng = np.random.default_rng(3)
    n = 4000
    lam, mu = rng.uniform(0.8, 2.2, n), rng.uniform(0.6, 1.8, n)
    dh, da = rng.uniform(0, 0.4, n), rng.uniform(0, 0.4, n)
    a, b = 0.9, 0.2
    hg = rng.poisson(lam * np.exp(-a * dh + b * da))
    ag = rng.poisson(mu * np.exp(-a * da + b * dh))
    est = av.calibrate(pd.DataFrame({"lam": lam, "mu": mu, "d_home": dh, "d_away": da, "hg": hg, "ag": ag}),
                       prior=(0.6, 0.35), prior_strength=0.1)
    assert est[0] == pytest.approx(a, abs=0.2) and est[1] == pytest.approx(b, abs=0.2)
