import json

import numpy as np
import pandas as pd
import pytest

from laliga import betting


def test_decide():
    c = {"stake": 10, "min_edge": 0.05, "max_odds": 8.0, "min_prob": 0.0}
    p = [0.5, 0.3, 0.2]
    assert betting.decide(p, [1.8, 3.5, 4.0], "favorito", c)[0] == 0
    # valor: 0.3 * 3.8 = 1.14 -> empate; 0.5*1.8=0.9 no
    i, o, ev = betting.decide(p, [1.8, 3.8, 4.0], "valor", c)
    assert (i, o) == (1, 3.8) and ev == pytest.approx(0.14)
    assert betting.decide(p, [1.8, 3.2, 4.5], "valor", c) is None  # ninguna supera el 5 %
    assert betting.decide(p, [1.8, 3.2, 12.0], "valor", {**c, "max_odds": 8.0}) is None  # 0.2*12 pero cuota demasiado alta
    assert betting.decide(p, [None, 3, 3], "favorito", c) is None


def _pred(home="Barcelona", away="Real Madrid", p=(0.5, 0.3, 0.2), b365=(1.8, 3.8, 4.0), ko="2026-10-25T15:15:00+00:00"):
    return {"season": 2026, "date": "2026-10-25", "kickoff_utc": ko, "home": home, "away": away,
            "p_home": p[0], "p_draw": p[1], "p_away": p[2], "b365": list(b365) if b365 else None}


def _matches(hg=np.nan, ag=np.nan):
    return pd.DataFrame([{"season": 2026, "date": pd.Timestamp("2026-10-25"), "home": "Barcelona", "away": "Real Madrid",
                          "hg": hg, "ag": ag}])


def test_ledger_lifecycle(settings):
    t0 = pd.Timestamp("2026-10-24T10:00", tz="UTC")
    led = betting.update(settings, _matches(), [_pred()], now=t0)
    assert set(led.strategy) == {"favorito", "valor"}
    assert (led.status == "pendiente").all()
    assert led.set_index("strategy").loc["valor", "pick"] == "X"
    # la cuota cambia antes del partido: se actualiza, no se duplica; "valor" deja de tener valor y se retira
    led = betting.update(settings, _matches(), [_pred(b365=(1.75, 3.2, 4.5))], now=t0 + pd.Timedelta(hours=5))
    assert list(led.strategy) == ["favorito"] and led.odds.iloc[0] == 1.75
    # una vez empezado el partido no se toca, aunque cambie la predicción
    led = betting.update(settings, _matches(), [_pred(p=(0.2, 0.3, 0.5))], now=pd.Timestamp("2026-10-25T16:00", tz="UTC"))
    assert led.pick.iloc[0] == "1" and led.odds.iloc[0] == 1.75
    # liquidación
    led = betting.update(settings, _matches(2, 1), [], now=pd.Timestamp("2026-10-25T20:00", tz="UTC"))
    r = led.iloc[0]
    assert r.status == "ganada" and r.score == "2-1" and r.profit == pytest.approx(7.5)
    s = json.loads((settings.predictions_dir / "bets_summary.json").read_text())
    fav = s["strategies"]["favorito"]
    assert fav["bets"] == 1 and fav["profit"] == pytest.approx(7.5) and fav["roi"] == pytest.approx(0.75)
    assert fav["curve"][-1][1] == pytest.approx(7.5)
    # ya liquidada: nuevas predicciones del mismo partido no crean otra apuesta
    led = betting.update(settings, _matches(2, 1), [_pred(ko="2026-12-01T15:00:00+00:00")], now=pd.Timestamp("2026-10-26", tz="UTC"))
    assert (led.strategy == "favorito").sum() == 1


def test_lost_bet_and_no_odds(settings):
    t0 = pd.Timestamp("2026-10-24T10:00", tz="UTC")
    betting.update(settings, _matches(), [_pred(b365=None)], now=t0)
    assert betting.load_ledger(settings).empty  # sin cuotas de Bet365 no se apuesta
    betting.update(settings, _matches(), [_pred()], now=t0)
    led = betting.update(settings, _matches(0, 3), [], now=t0 + pd.Timedelta(days=2))
    assert set(led.status) == {"perdida"} and (led.profit == -10).all()


def test_backtest_strategies(settings):
    rng = np.random.default_rng(0)
    n = 6000
    p = rng.dirichlet([5, 3, 3], n)
    y = np.array([rng.choice(3, p=pp) for pp in p])
    fair = 1 / p
    odds = fair * 0.95  # casa con margen y probabilidades "verdaderas": se debe perder dinero
    m = pd.DataFrame({"season": np.repeat([2023, 2024], n // 2), "date": pd.date_range("2023-08-01", periods=n),
                      "home": "A", "away": "B", "hg": 1.0, "ag": 0.0,
                      "b365_h": odds[:, 0], "b365_d": odds[:, 1], "b365_a": odds[:, 2]})
    bt = pd.DataFrame({"season": m.season, "y": y, "stacker_H": p[:, 0], "stacker_D": p[:, 1], "stacker_A": p[:, 2]},
                      index=m.index)
    bt.to_csv(settings.processed_dir / "backtest_predictions.csv")
    res = betting.backtest_strategies(settings, m)
    fav = res["strategies"]["favorito"]
    assert fav["bets"] == n and fav["roi"] < 0
    assert "valor" not in res["strategies"]  # con margen ninguna apuesta tiene valor
    assert len(fav["by_season"]) == 2
    assert (settings.predictions_dir / "betting_backtest.json").exists()
