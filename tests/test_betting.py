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
    assert set(led.strategy) == {"favorito", "valor", "goles"}
    assert (led.status == "pendiente").all() and len(out["placed"]) == 3
    assert led.set_index("strategy").loc["valor", "pick"] == "X"
    assert led.set_index("strategy").loc["goles", "pick"] == "+2.5"
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
    assert "kelly" not in s["strategies"]
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


def test_slot_of():
    # viernes-lunes = mismo fin de semana; martes-jueves = jornada entre semana
    assert betting.slot_of("2026-10-30") == betting.slot_of("2026-11-02") == "2026-10-30"
    assert betting.slot_of("2026-11-03") == betting.slot_of("2026-11-05") == "2026-11-03"
    assert betting.slot_of("2026-11-06") == "2026-11-06"


def test_choose_legs():
    c = betting.cfg(type("S", (), {"raw": {}})())
    cands = [{"home": "A", "away": "B", "p": [0.70, 0.2, 0.1], "odds": [1.4, 4.5, 8.0]},
             {"home": "C", "away": "D", "p": [0.15, 0.2, 0.65], "odds": [6.5, 4.5, 1.5]},
             {"home": "E", "away": "F", "p": [0.45, 0.3, 0.25], "odds": [2.1, 3.3, 3.6]},  # favorito < 60 %
             {"home": "G", "away": "H", "p": [0.80, 0.15, 0.05], "odds": [1.2, 6.0, 13.0]},
             {"home": "I", "away": "J", "p": [0.62, 0.2, 0.18], "odds": [1.6, 4.0, 5.0]}]
    legs = betting.choose_legs(cands, "comb_favoritos", c)
    assert [x["home"] for x in legs] == ["G", "A", "C"]  # los 3 más probables, cada uno ≥ 60 %
    assert betting.choose_legs(cands[:1], "comb_favoritos", c) is None  # una sola selección no es combinada


def _wk(home, away, p, b365, ko):
    return _pred(home=home, away=away, p=p, b365=b365, ko=ko) | {"date": ko[:10]}


def test_parlay_lifecycle(settings):
    preds = [_wk("Barcelona", "Getafe", (0.75, 0.15, 0.10), (1.35, 5.0, 9.0), "2026-10-31T15:00:00+00:00"),
             _wk("Sevilla", "Real Madrid", (0.15, 0.20, 0.65), (6.0, 4.5, 1.55), "2026-11-01T20:00:00+00:00"),
             _wk("Betis", "Elche", (0.62, 0.22, 0.16), (1.70, 3.8, 5.0), "2026-11-02T20:00:00+00:00")]
    t0 = pd.Timestamp("2026-10-30T10:00", tz="UTC")
    mt = pd.DataFrame([{"season": 2026, "home": p["home"], "away": p["away"], "hg": np.nan, "ag": np.nan}
                       for p in preds])
    out = betting.update(settings, mt, preds, now=t0)
    par = betting.load_parlays(settings).set_index("strategy")
    fav = par.loc["comb_favoritos"]
    assert fav.n_legs == 3 and fav.slot == "2026-10-30" and fav.status == "pendiente"
    assert fav.odds == pytest.approx(1.35 * 1.55 * 1.70, abs=1e-3)
    assert fav.prob == pytest.approx(0.75 * 0.65 * 0.62, abs=1e-3)
    assert len(out["parlays_placed"]) >= 1
    # empieza el primer partido: la combinada queda congelada aunque cambien las cuotas
    preds2 = [dict(p, b365=[x * 1.1 for x in p["b365"]]) for p in preds]
    betting.update(settings, mt, preds2, now=pd.Timestamp("2026-10-31T16:00", tz="UTC"))
    assert betting.load_parlays(settings).set_index("strategy").loc["comb_favoritos", "odds"] == pytest.approx(fav.odds)
    # el Barça gana: sigue pendiente; el Madrid empata: perdida sin esperar al lunes
    mt.loc[0, ["hg", "ag"]] = [2, 0]
    betting.update(settings, mt, [], now=pd.Timestamp("2026-10-31T18:00", tz="UTC"))
    assert betting.load_parlays(settings).set_index("strategy").loc["comb_favoritos", "status"] == "pendiente"
    mt.loc[1, ["hg", "ag"]] = [1, 1]
    out = betting.update(settings, mt, [], now=pd.Timestamp("2026-11-01T22:00", tz="UTC"))
    r = betting.load_parlays(settings).set_index("strategy").loc["comb_favoritos"]
    assert r.status == "perdida" and r.profit == -10
    assert any(x["strategy"] == "comb_favoritos" for x in out["parlays_settled"])
    s = json.loads((settings.predictions_dir / "bets_summary.json").read_text())
    assert s["strategies"]["comb_favoritos"]["bets"] == 1 and s["strategies"]["comb_favoritos"]["profit"] == -10
    assert s["parlays"][0]["legs"][0]["home"]
    assert "Combinada" in betting.format_parlays(out["parlays_placed"], out["parlays_settled"])


def test_parlay_won(settings):
    preds = [_wk("Barcelona", "Getafe", (0.75, 0.15, 0.10), (1.35, 5.0, 9.0), "2026-10-31T15:00:00+00:00"),
             _wk("Betis", "Elche", (0.62, 0.22, 0.16), (1.70, 3.8, 5.0), "2026-11-01T20:00:00+00:00")]
    mt = pd.DataFrame([{"season": 2026, "home": p["home"], "away": p["away"], "hg": np.nan, "ag": np.nan,
                        "b365c_h": 1.30} for p in preds])
    betting.update(settings, mt, preds, now=pd.Timestamp("2026-10-30T10:00", tz="UTC"))
    mt[["hg", "ag"]] = [[3, 0], [2, 1]]
    betting.update(settings, mt, [], now=pd.Timestamp("2026-11-02T10:00", tz="UTC"))
    r = betting.load_parlays(settings).set_index("strategy").loc["comb_favoritos"]
    assert r.status == "ganada" and r.profit == pytest.approx(10 * (1.35 * 1.70 - 1), abs=0.02)
    assert r.clv == pytest.approx(1.35 * 1.70 / (1.30 * 1.30) - 1, abs=1e-3)


def test_parlay_backtest_loses_with_margin(settings):
    rng = np.random.default_rng(3)
    n = 4000
    p = rng.dirichlet([6, 3, 3], n)
    y = np.array([rng.choice(3, p=pp) for pp in p])
    odds = 0.95 / p
    m = pd.DataFrame({"season": 2024, "date": pd.date_range("2022-08-05", periods=n, freq="8h"),
                      "home": [f"H{i}" for i in range(n)], "away": "B", "hg": 1.0, "ag": 0.0,
                      "b365_h": odds[:, 0], "b365_d": odds[:, 1], "b365_a": odds[:, 2]})
    pd.DataFrame({"season": m.season, "y": y, "stacker_H": p[:, 0], "stacker_D": p[:, 1], "stacker_A": p[:, 2]},
                 index=m.index).to_csv(settings.processed_dir / "backtest_predictions.csv")
    res = betting.backtest_strategies(settings, m)
    cf = res["strategies"]["comb_favoritos"]
    single = res["strategies"]["favorito"]
    assert cf["bets"] > 50 and cf["roi"] < 0
    assert cf["hit_rate"] == pytest.approx(cf["avg_prob"], abs=0.08)  # probabilidades bien calibradas
    assert cf["roi"] < single["roi"]  # el margen se multiplica: la combinada pierde más


def test_market_recal_learns_favourite_bias(settings):
    from laliga import market_recal
    rng = np.random.default_rng(4)
    n = 12000
    true = rng.dirichlet([1.5, 1, 1], n)
    y = np.array([rng.choice(3, p=pp) for pp in true])
    # casa con sesgo favorito-sorpresa: aplana las probabilidades (p^0.8) y cobra un 5 %
    flat = true ** 0.8
    flat /= flat.sum(1, keepdims=True)
    odds = 1 / (flat * 1.05)
    m = pd.DataFrame({"season": np.repeat(np.arange(2010, 2022), n // 12), "date": "2020-01-01",
                      "home": "A", "away": "B",
                      "hg": np.where(y == 0, 1.0, 0.0), "ag": np.where(y == 2, 1.0, 0.0),
                      "b365_h": odds[:, 0], "b365_d": odds[:, 1], "b365_a": odds[:, 2]})
    settings.raw.setdefault("backtest", {})["first_test_season"] = 2020
    r = market_recal.run(settings, m)
    assert r["log_loss"]["recalibrado"] < r["log_loss"]["mercado"]
    assert r["diff_vs_mercado"]["ci95"][1] < 0
    model = market_recal.load_model(settings)
    p = market_recal.recal_from_odds(model, [1.4, 4.5, 8.0])
    implied = market_recal.devig([[1.4, 4.5, 8.0]])[0]
    assert sum(p) == pytest.approx(1, abs=1e-3) and p[0] > implied[0]  # sube al favorito
    assert market_recal.recal_from_odds(model, None) is None


def test_roi_ci_and_preregistration(settings):
    assert betting.roi_ci([10] * 3, [5, -10, 5]) is None  # muy pocas apuestas
    lo, hi = betting.roi_ci([10] * 200, [9.0] * 120 + [-10.0] * 80)
    assert lo < 0.14 < hi  # ROI real = (1080 - 800) / 2000 = 0.14
    rows = [{"strategy": "comb_favoritos", "season": 2026, "slot": f"2026-11-{d:02d}", "legs": "[]", "n_legs": 3,
             "prob": 0.4, "odds": 2.5, "odds_first": 2.5, "odds_close": 2.4, "clv": 0.04, "ev": 0.0, "stake": 10.0,
             "kickoff_first": "x", "placed_at": "x", "status": "ganada" if d % 2 else "perdida", "score": "",
             "profit": 15.0 if d % 2 else -10.0, "settled_at": "x"} for d in range(1, 29)]
    par = betting._typed_parlays(pd.DataFrame(rows))
    pr = betting.preregistration(settings, par)
    assert pr["estado"] == "en curso" and pr["n"] == 28 and pr["clv"] == pytest.approx(0.04)
    settings.raw["preregistro"]["min_apuestas"] = 20
    assert betting.preregistration(settings, par)["estado"] in ("superado", "no superado")
    settings.raw["betting"]["parlay_min_leg_prob"] = 0.55  # alguien toca el umbral a mitad del test
    pr = betting.preregistration(settings, par)
    assert pr["estado"] == "invalidado" and pr["reglas_cambiadas"] == ["parlay_min_leg_prob"]
