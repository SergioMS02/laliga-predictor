import json

import numpy as np
import pandas as pd
import pytest

from laliga import betting

C = {"stake": 10, "min_edge": 0.05, "max_odds": 8.0, "min_prob": 0.0, "ou_min_edge": 0.03,
     "kelly_bankroll": 100.0, "kelly_fraction": 0.25, "kelly_max_frac": 0.05, "kelly_min_stake": 0.5}


def test_decide():
    p = [0.5, 0.3, 0.2]
    assert betting.decide(p, [1.8, 3.5, 4.0], "favorito", C)[0] == 0
    # valor: 0.3 * 3.8 = 1.14 -> empate; 0.5*1.8=0.9 no
    i, o, ev = betting.decide(p, [1.8, 3.8, 4.0], "valor", C)
    assert (i, o) == (1, 3.8) and ev == pytest.approx(0.14)
    assert betting.decide(p, [1.8, 3.2, 4.5], "valor", C) is None  # ninguna supera el 5 %
    assert betting.decide(p, [1.8, 3.2, 12.0], "valor", C) is None  # 0.2*12 pero cuota demasiado alta
    assert betting.decide(p, [None, 3, 3], "favorito", C) is None
    # goles: 0.6 * 1.8 = 1.08 >= 1.03
    assert betting.decide([0.6, 0.4], [1.8, 2.0], "goles", C)[0] == 0
    assert betting.decide([0.5, 0.5], [1.9, 1.9], "goles", C) is None


def test_kelly_stake():
    # p=0.3, o=3.8: f = 0.25 * 0.14 / 2.8 = 0.0125 -> 1.25 € sobre 100 -> 1.0 € (redondeo a 0.5)
    assert betting.kelly_stake(0.3, 3.8, 100, C) == pytest.approx(1.0)
    # tope del 5 %
    assert betting.kelly_stake(0.9, 3.0, 100, C) == pytest.approx(5.0)
    assert betting.kelly_stake(0.3, 3.8, 0, C) == 0


def test_blend_ou():
    assert betting.blend_ou(0.6, None) == pytest.approx(0.6)
    w = {"a_model": 0.0, "a_market": 1.0, "b": 0.0}
    assert betting.blend_ou(0.9, 0.4, w) == pytest.approx(0.4)
    assert betting.devig2(1.9, 1.9) == pytest.approx(0.5)
    assert betting.devig2(None, 2.0) is None


def _pred(home="Barcelona", away="Real Madrid", p=(0.5, 0.3, 0.2), b365=(1.8, 3.8, 4.0),
          ko="2026-10-25T15:15:00+00:00", p_over=0.6, ou=(1.8, 2.0)):
    return {"season": 2026, "date": "2026-10-25", "kickoff_utc": ko, "home": home, "away": away,
            "p_home": p[0], "p_draw": p[1], "p_away": p[2], "b365": list(b365) if b365 else None,
            "ou": {"p_over": p_over}, "b365_ou": list(ou) if ou else None}


def _matches(hg=np.nan, ag=np.nan, close=None):
    r = {"season": 2026, "date": pd.Timestamp("2026-10-25"), "home": "Barcelona", "away": "Real Madrid",
         "hg": hg, "ag": ag}
    if close:
        r.update(close)
    return pd.DataFrame([r])


def test_ledger_lifecycle(settings):
    t0 = pd.Timestamp("2026-10-24T10:00", tz="UTC")
    out = betting.update(settings, _matches(), [_pred()], now=t0)
    led = out["ledger"]
    assert set(led.strategy) == {"favorito", "valor", "goles", "kelly"}
    assert (led.status == "pendiente").all() and len(out["placed"]) == 4
    assert led.set_index("strategy").loc["valor", "pick"] == "X"
    assert led.set_index("strategy").loc["goles", "pick"] == "+2.5"
    assert led.set_index("strategy").loc["kelly", "stake"] == pytest.approx(1.0)
    # la cuota cambia antes del partido: se actualiza, no se duplica; "valor" deja de tener valor y se retira;
    # la primera cuota vista se conserva para medir el CLV
    out = betting.update(settings, _matches(), [_pred(b365=(1.75, 3.2, 4.5), ou=(1.75, 2.1))],
                         now=t0 + pd.Timedelta(hours=5))
    led = out["ledger"].set_index("strategy")
    assert set(led.index) == {"favorito", "goles"} and led.loc["favorito", "odds"] == 1.75
    assert led.loc["favorito", "odds_first"] == 1.8 and not out["placed"]
    # una vez empezado el partido no se toca, aunque cambie la predicción
    led = betting.update(settings, _matches(), [_pred(p=(0.2, 0.3, 0.5))],
                         now=pd.Timestamp("2026-10-25T16:00", tz="UTC"))["ledger"]
    assert led.set_index("strategy").loc["favorito", "pick"] == "1"
    # liquidación con cuota de cierre de Bet365 disponible
    out = betting.update(settings, _matches(2, 1, {"b365c_h": 1.6, "b365c_o25": 1.65}), [],
                         now=pd.Timestamp("2026-10-25T20:00", tz="UTC"))
    led = out["ledger"].set_index("strategy")
    assert led.loc["favorito", "status"] == "ganada" and led.loc["favorito", "score"] == "2-1"
    assert led.loc["favorito", "profit"] == pytest.approx(7.5)
    assert led.loc["favorito", "clv"] == pytest.approx(1.8 / 1.6 - 1, abs=1e-4)
    assert led.loc["goles", "status"] == "ganada"  # 3 goles: más de 2,5
    assert len(out["settled"]) == 2
    s = json.loads((settings.predictions_dir / "bets_summary.json").read_text())
    fav = s["strategies"]["favorito"]
    assert fav["bets"] == 1 and fav["profit"] == pytest.approx(7.5) and fav["roi"] == pytest.approx(0.75)
    assert fav["curve"][-1][1] == pytest.approx(7.5) and fav["clv"] == pytest.approx(0.125)
    assert "ganada" in betting.format_settled(out["settled"], s["strategies"]) or "✅" in betting.format_settled(out["settled"])
    # ya liquidada: nuevas predicciones del mismo partido no crean otra apuesta
    led = betting.update(settings, _matches(2, 1), [_pred(ko="2026-12-01T15:00:00+00:00")],
                         now=pd.Timestamp("2026-10-26", tz="UTC"))["ledger"]
    assert (led.strategy == "favorito").sum() == 1


def test_lost_bet_and_no_odds(settings):
    t0 = pd.Timestamp("2026-10-24T10:00", tz="UTC")
    betting.update(settings, _matches(), [_pred(b365=None, ou=None)], now=t0)
    assert betting.load_ledger(settings).empty  # sin cuotas de Bet365 no se apuesta
    betting.update(settings, _matches(), [_pred()], now=t0)
    led = betting.update(settings, _matches(0, 2), [], now=t0 + pd.Timedelta(days=2))["ledger"]
    assert set(led.status) == {"perdida"}
    assert (led.profit == -led.stake).all()
    s = json.loads((settings.predictions_dir / "bets_summary.json").read_text())
    assert s["strategies"]["kelly"]["bankroll"] == pytest.approx(99.0)
    # sin cuota de cierre de Bet365 se usa la última vista (CLV 0)
    assert s["strategies"]["favorito"]["clv"] == pytest.approx(0.0)


def test_backtest_strategies(settings):
    rng = np.random.default_rng(0)
    n = 6000
    p = rng.dirichlet([5, 3, 3], n)
    y = np.array([rng.choice(3, p=pp) for pp in p])
    fair = 1 / p
    odds = fair * 0.95  # casa con margen y probabilidades "verdaderas": se debe perder dinero
    m = pd.DataFrame({"season": np.repeat([2023, 2024], n // 2), "date": pd.date_range("2023-08-01", periods=n),
                      "home": "A", "away": "B", "hg": 1.0, "ag": 0.0,
                      "b365_h": odds[:, 0], "b365_d": odds[:, 1], "b365_a": odds[:, 2],
                      "b365c_h": odds[:, 0], "b365c_d": odds[:, 1], "b365c_a": odds[:, 2]})
    bt = pd.DataFrame({"season": m.season, "y": y, "stacker_H": p[:, 0], "stacker_D": p[:, 1], "stacker_A": p[:, 2]},
                      index=m.index)
    bt.to_csv(settings.processed_dir / "backtest_predictions.csv")
    res = betting.backtest_strategies(settings, m)
    fav = res["strategies"]["favorito"]
    assert fav["bets"] == n and fav["roi"] < 0 and fav["clv"] == pytest.approx(0.0)
    assert "valor" not in res["strategies"]  # con margen ninguna apuesta tiene valor
    assert len(fav["by_season"]) == 2
    assert (settings.predictions_dir / "betting_backtest.json").exists()


def test_ou_backtest(settings):
    rng = np.random.default_rng(1)
    n = 3000
    lam, mu = rng.uniform(0.8, 2.0, n), rng.uniform(0.6, 1.6, n)
    hg, ag = rng.poisson(lam), rng.poisson(mu)
    seasons = np.repeat([2019, 2020, 2021], n // 3)
    p_true = betting.p_over_from_goals(lam, mu, rho=0.0)
    oo, ou_ = 0.95 / p_true, 0.95 / (1 - p_true)
    m = pd.DataFrame({"season": seasons, "date": pd.date_range("2019-08-01", periods=n), "home": [f"H{i}" for i in range(n)],
                      "away": "B", "hg": hg.astype(float), "ag": ag.astype(float),
                      "b365_o25": oo, "b365_u25": ou_, "ou_o25": oo, "ou_u25": ou_,
                      "b365c_o25": oo, "b365c_u25": ou_})
    m[["season", "home", "away", "date", "hg", "ag"]].assign(dc_lam=lam, dc_mu=mu).to_csv(
        settings.processed_dir / "oos_table.csv", index=False)
    c = betting.cfg(settings)
    r = betting._ou_backtest(settings, m, {**c, "ou_min_edge": 0.0})
    assert r is None or r["roi"] < 0.05
    w = json.loads((settings.state_dir / "ou_blend.json").read_text())
    assert w["n"] > 1000


def test_odds_compare_in_backtest(settings):
    rng = np.random.default_rng(2)
    n = 800
    p = rng.dirichlet([5, 3, 3], n)
    y = np.array([rng.choice(3, p=pp) for pp in p])
    b = 0.93 / p
    m = pd.DataFrame({"season": 2024, "date": pd.date_range("2024-08-01", periods=n), "home": "A", "away": "B",
                      "hg": 1.0, "ag": 0.0, "b365_h": b[:, 0], "b365_d": b[:, 1], "b365_a": b[:, 2],
                      "max_h": b[:, 0] * 1.06, "max_d": b[:, 1] * 1.06, "max_a": b[:, 2] * 1.06,
                      "avg_h": b[:, 0] * 1.01, "avg_d": b[:, 1] * 1.01, "avg_a": b[:, 2] * 1.01})
    pd.DataFrame({"season": m.season, "y": y, "stacker_H": p[:, 0], "stacker_D": p[:, 1], "stacker_A": p[:, 2]},
                 index=m.index).to_csv(settings.processed_dir / "backtest_predictions.csv")
    res = betting.backtest_strategies(settings, m)
    oc = res["odds_compare"]["favorito"]
    assert oc["mejor"]["roi"] > oc["bet365"]["roi"]
    assert res["odds_quality"]["b365_is_best"] < 0.5
