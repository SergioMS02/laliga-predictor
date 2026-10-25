import json

import numpy as np
import pandas as pd

from laliga.data import polymarket as pm


def _mkt(title, yes, vol=1000.0, active=True, closed=False):
    return {"groupItemTitle": title, "outcomePrices": json.dumps([str(yes), str(1 - yes)]), "volumeNum": vol,
            "active": active, "closed": closed, "clobTokenIds": json.dumps(["1", "2"])}


def _event(slug="lal-bar-rea-2026-10-25", title="FC Barcelona vs. Real Madrid CF", prices=(0.555, 0.205, 0.235),
           volume=6.5e6, liquidity=7e5, start="2026-10-25T20:00:00Z", live=False):
    return {"slug": slug, "title": title, "startTime": start, "volume": volume, "liquidity": liquidity, "live": live,
            "closed": False, "markets": [_mkt("FC Barcelona", prices[0]), _mkt(f"Draw ({title})", prices[1]),
                                         _mkt("Real Madrid CF", prices[2])]}


class S:  # ajustes mínimos
    def __init__(self, tmp=None, poly=None):
        self.poly = poly or {}
        self.state_dir = tmp

    def get(self, sec, key, default=None):
        return self.poly.get(key, default) if sec == "polymarket" else default


def test_parse_match_event_normaliza_y_mapea_equipos():
    m = pm.parse_match_event(_event())
    assert (m["home"], m["away"]) == ("Barcelona", "Real Madrid")
    assert abs(sum(m["p"]) - 1) < 1e-3 and m["p"][0] > m["p"][2] > 0
    assert abs(m["overround"] + 0.005) < 1e-6
    # submercados (córners, marcador exacto...) no son el 1X2
    assert pm.parse_match_event(_event(slug="lal-bar-rea-2026-10-25-total-corners")) is None


def test_nombres_raros_de_polymarket():
    assert pm.team("Real Racing Club") == "Santander"
    assert pm.team("RC Deportivo A Coruña") == "La Coruna"
    assert pm.team("Club Atlético de Madrid") == "Ath Madrid"


def test_usable_exige_volumen_liquidez_y_que_no_haya_empezado():
    c = pm.cfg(S())
    now = pd.Timestamp("2026-10-25T10:00Z")
    m = pm.parse_match_event(_event())
    assert pm.usable(m, c, now)[0]
    assert not pm.usable({**m, "volume": 10000}, c, now)[0]
    assert not pm.usable({**m, "liquidity": 500}, c, now)[0]
    assert pm.usable(m, c, pd.Timestamp("2026-10-25T20:30Z"))[1] == "en juego"
    assert not pm.usable({**m, "live": True}, c, now)[0]
    assert not pm.usable({**m, "overround": 0.2}, c, now)[0]
    assert not pm.usable(None, c, now)[0]


def test_for_match_respeta_fecha():
    m = pm.parse_match_event(_event())
    assert pm.for_match([m], "Barcelona", "Real Madrid", pd.Timestamp("2026-10-25T20:00Z")) is m
    assert pm.for_match([m], "Barcelona", "Real Madrid", pd.Timestamp("2027-03-01")) is None
    assert pm.for_match([m], "Real Madrid", "Barcelona") is None


def test_blend_y_peso_medido(tmp_path):
    p = pm.blend(np.array([0.6, 0.2, 0.2]), [0.5, 0.3, 0.2], 0.1)
    assert np.allclose(p, [0.51, 0.29, 0.2])
    s = S(tmp_path)
    assert pm.model_weight(s) == 0.25  # sin medida
    (tmp_path / "polymarket_eval.json").write_text(json.dumps({"n": 400, "best_model_weight": 0.0}))
    assert pm.model_weight(s) == pm.MIN_MODEL_WEIGHT  # nunca por debajo del mínimo
    (tmp_path / "polymarket_eval.json").write_text(json.dumps({"n": 400, "best_model_weight": 0.35}))
    assert pm.model_weight(s) == 0.35
    assert pm.model_weight(S(tmp_path, {"model_weight": 0.5})) == 0.5


def test_season_event_quita_marcadores_de_posicion_y_compara():
    ev = {"slug": "laliga-2027-champion-x", "title": "LALIGA: 2027 Champion", "volume": 9e6, "liquidity": 1,
          "markets": [_mkt("Barcelona", 0.755, 5e5), _mkt("Real Madrid", 0.22, 7e5), _mkt("Team A", 0.5, 0, active=False),
                      _mkt("Other", 0.5, 0)]}
    mk = pm.parse_season_event(ev, "champion")
    assert [o["team"] for o in mk["outcomes"]] == ["Barcelona", "Real Madrid"]
    sc = {"kind": "pichichi", "title": "t", "url": "u", "volume": 1, "outcomes": [
        {"name": "Kylian Mbappé", "p": 0.47, "volume": 8e4}, {"name": "Nadie Conocido", "p": 0.01, "volume": 10}]}
    proj = {"teams": [{"team": "Barcelona", "p_title": 0.9, "pos_probs": [0.9, 0.08]},
                      {"team": "Real Madrid", "p_title": 0.06, "pos_probs": [0.06, 0.8]}]}
    scorers = {"players": [{"name": "Kylian Mbappe-Lottin", "team": "Real Madrid", "p": 0.78}]}
    out = pm.compare_season({"champion": mk, "pichichi": sc}, proj, scorers)
    rows = out["markets"]["champion"]["rows"]
    assert rows[0]["model"] == 0.9 and rows[0]["poly"] == 0.755 and not rows[0]["thin"]
    pr = out["markets"]["pichichi"]["rows"]
    assert pr[0]["model"] == 0.78 and pr[1]["model"] is None and pr[1]["thin"]


def test_slugs_de_temporada():
    kinds = {k for k, pat in pm.SEASON_KINDS for s in ["laliga-2027-champion-20260701200737375",
                                                       "laliga-top-goalscorer-2026-27", "laliga-teams-relegated-2026-27",
                                                       "laliga-team-to-qualify-for-the-2027-28-uefa-champions-league",
                                                       "laliga-top-2-finishers-2026-27"] if pat.search(s)}
    assert kinds == {"champion", "pichichi", "releg", "ucl", "top2"}


def test_evaluate_mide_frente_al_cierre_y_elige_peso():
    rng = np.random.default_rng(0)
    n = 300
    true = rng.dirichlet([4, 2.5, 3], n)
    y = np.array([rng.choice(3, p=p) for p in true])
    hg = np.where(y == 0, 2, np.where(y == 1, 1, 0))
    ag = np.where(y == 2, 2, np.where(y == 1, 1, 0))
    dates = pd.date_range("2025-08-15", periods=n, freq="D")
    teams = [(f"H{i}", f"A{i}") for i in range(n)]
    noisy = np.clip(true + rng.normal(0, 0.05, true.shape), 0.02, None)
    noisy /= noisy.sum(1, keepdims=True)
    matches = pd.DataFrame({"date": dates, "home": [t[0] for t in teams], "away": [t[1] for t in teams], "hg": hg, "ag": ag,
                            "odds_close_h": 1 / noisy[:, 0] / 1.03, "odds_close_d": 1 / noisy[:, 1] / 1.03,
                            "odds_close_a": 1 / noisy[:, 2] / 1.03})
    hist = pd.DataFrame({"slug": [f"s{i}" for i in range(n)], "date": dates.strftime("%Y-%m-%d"), "kickoff": dates,
                         "home": matches.home, "away": matches.away, "volume": 1e5,
                         "p_h": true[:, 0], "p_d": true[:, 1], "p_a": true[:, 2]})
    bt = pd.DataFrame({c: np.full(n, 1 / 3) for c in ["stacker_mercado_H", "stacker_mercado_D", "stacker_mercado_A",
                                                       "stacker_H", "stacker_D", "stacker_A"]}, index=matches.index)
    ev = pm.evaluate(S(), matches, hist, bt)
    assert ev["n"] == n
    assert ev["logloss"]["polymarket"] < ev["logloss"]["modelo"]
    assert ev["best_model_weight"] <= 0.3  # un modelo inútil apenas debe pesar
    assert ev["model_weight_used"] == max(pm.MIN_MODEL_WEIGHT, ev["best_model_weight"])


def test_movimiento_atribuido_a_polymarket():
    from laliga import movements
    snap = {"lam0": 1.5, "mu0": 1.0, "d_home": [0, 0], "d_away": [0, 0], "rho": 0.0, "max_goals": 10, "raw": {}}
    base = movements.probs_from_snapshot(snap, "dixon_coles", {}, 0.0, 0.0)
    with_pm = movements.probs_from_snapshot({**snap, "poly": {"p": [0.2, 0.3, 0.5], "w": 0.1}}, "dixon_coles", {}, 0.0, 0.0)
    assert np.allclose(with_pm, 0.9 * np.array([0.2, 0.3, 0.5]) + 0.1 * base)
    assert "empieza a usarse" in movements._detail_poly(None, {"p": [0.2, 0.3, 0.5], "w": 0.1})
