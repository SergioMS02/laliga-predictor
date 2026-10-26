import numpy as np
import pandas as pd

from laliga import markets, quinigol
from laliga.data import polymarket as pm
from laliga.data import shots
from laliga.models.dixon_coles import score_matrix


def test_mercados_derivados_coherentes():
    m = score_matrix(1.6, 1.0, -0.05, 10)
    o = markets.from_matrix(m, 1.6, 1.0)
    i, j = np.indices(m.shape)
    mm = m / m.sum()
    p1, px = mm[i > j].sum(), mm[i == j].sum()
    assert abs(o["double_chance"]["1X"] - (p1 + px)) < 1e-3
    assert abs(sum(o["ht_ft"].values()) - 1) < 2e-3
    assert abs(sum(o["quinigol"]["home"]) - 1) < 1e-3
    assert o["totals"]["0.5"] > o["totals"]["2.5"] > o["totals"]["4.5"]
    # hándicap 0: el empate devuelve; −0,5 equivale a ganar
    assert o["asian"]["0.0"]["push"] > 0.2 and o["asian"]["-0.5"]["push"] == 0
    assert abs(o["asian"]["-0.5"]["win"] - p1) < 1e-3
    # cuota justa de −0,25 entre la de 0 y la de −0,5
    assert o["asian"]["0.0"]["fair_home"] < o["asian"]["-0.25"]["fair_home"] < o["asian"]["-0.5"]["fair_home"]


def test_valor_en_hándicap():
    o = markets.from_matrix(score_matrix(1.6, 1.0, -0.05, 10), 1.6, 1.0)
    fair = o["asian"]["-0.5"]["fair_home"]
    v = markets.ah_value(o, -0.5, fair * 1.1, 1.5)
    assert v["ev"][0] > 0.05
    assert markets.ah_value(o, None, 2.0, 2.0) is None


def test_partidos_que_faltan_en_liga_a_doble_vuelta():
    from laliga.leagues import remaining_pairs
    teams = ["A", "B", "C"]
    rows = [{"season": 2026, "date": pd.Timestamp("2026-09-01"), "home": "A", "away": "B", "hg": 1, "ag": 0},
            {"season": 2026, "date": pd.Timestamp("2026-09-08"), "home": "B", "away": "C", "hg": 0, "ag": 0},
            {"season": 2026, "date": pd.Timestamp("2026-09-15"), "home": "C", "away": "A", "hg": np.nan, "ag": np.nan}]
    rem = remaining_pairs(pd.DataFrame(rows), 2026)
    assert len(rem) == 6 - 2 and set(zip(rem.home, rem.away)) == {("A", "C"), ("B", "A"), ("C", "A"), ("C", "B")}
    assert rem.set_index(["home", "away"]).loc[("C", "A"), "date"] == pd.Timestamp("2026-09-15")
    assert teams


def test_quinigol_lectura_y_aciertos():
    js = [{"orden": 2, "local": "AT.MADRID", "visitante": "DEPORTIVO", "local_goles": "1", "visitante_goles": "0",
           "signo_goles": "10", "live": 0, "dia": "SAB", "hora": "21:00", "jornada": 17, "temporada": 2027,
           "porc15L_0": 2, "porc15L_1": 9, "porc15L_2": 29, "porc15L_M": 60, "porc15V_0": 55, "porc15V_1": 39, "porc15V_2": 5},
          {"orden": 1, "local": "RAYO", "visitante": "ELCHE", "local_goles": "", "visitante_goles": "", "signo_goles": "",
           "live": 0, "jornada": 17, "temporada": 2027}]
    pg = quinigol.parse(js)
    assert [r["n"] for r in pg["rows"]] == [1, 2]
    r = pg["rows"][1]
    assert r["home"] == "Atlético de Madrid" and r["result"] == "10"
    assert r["public"][1][3] == 0.01  # la «M» que falta es el resto hasta 100
    assert pg["rows"][0]["result"] is None
    q = quinigol.probs_from_goals(1.8, 0.7)
    assert abs(sum(q["home"]) - 1) < 1e-3 and len(q["pick"]) == 2
    d = quinigol.hits_dist([0.5, 0.5])
    assert np.allclose(d, [0.25, 0.5, 0.25])


def test_directo_con_tiros_a_puerta():
    # γ > 0: un equipo que tira más a puerta de lo esperado tiene más goles por delante
    hi = shots.remaining_rate(1.5, 45, 6, 0.1, 2.8)
    lo = shots.remaining_rate(1.5, 45, 0, 0.1, 2.8)
    base = shots.remaining_rate(1.5, 45, 1.5 * 2.8 / 2, 0.1, 2.8)
    assert lo < base < hi and abs(base - 0.75) < 1e-6
    rng = np.random.default_rng(1)
    rows = []
    for k in range(400):  # datos sintéticos sin efecto de los tiros: γ ≈ 0
        lam, mu = 1.5, 1.1
        rows.append({"match_id": k, "t": 45, "lam": lam, "mu": mu, "sot_h": rng.poisson(lam * 2.8 / 2),
                     "sot_a": rng.poisson(mu * 2.8 / 2), "xg_h": lam / 2, "xg_a": mu / 2,
                     "g_h_after": rng.poisson(lam / 2), "g_a_after": rng.poisson(mu / 2)})
    out = shots.calibrate_live(pd.DataFrame(rows))
    assert abs(out["gamma"]) < 0.15


def test_polymarket_secundarios_ignora_contratos_finos():
    more = {"markets": [{"sportsMarketType": "totals", "line": 2.5, "outcomePrices": '["0.55", "0.45"]', "volumeNum": 9000},
                        {"sportsMarketType": "totals", "line": 6.5, "outcomePrices": '["0.5", "0.5"]', "volumeNum": 10},
                        {"sportsMarketType": "both_teams_to_score", "outcomePrices": '["0.6", "0.4"]', "volumeNum": 3000}]}
    out = pm.parse_secondary(more, None, None, "Barcelona", "Real Madrid")
    assert out["totals"] == {"2.5": {"p": 0.55, "volume": 9000}} and out["btts"]["p"] == 0.6


def test_previa_sin_inventar():
    from laliga import insights
    p = {"home": "Barcelona", "away": "Alaves", "p_home": 0.8, "p_draw": 0.12, "p_away": 0.08,
         "dc": {"xg_home": 2.4, "xg_away": 0.7, "p_over25": 0.6}, "ou": {"p_over": 0.58},
         "absences": {"Alaves": [{"player": "X Y", "status": "out"}]}, "scorers": {"home": [{"name": "Raphinha", "p": 0.45}]},
         "referee": None}
    t = insights.preview(p, {"Barcelona": {"pos": 1, "pts": 27, "last5": [{"res": "V"}] * 5}, "Alaves": {"pos": 15, "pts": 9}},
                         {"Alaves": "Alavés"})
    assert "Barcelona es claro favorito (80 %)" in t and "Bajas de Alavés: X Y" in t and "Raphinha (45 %)" in t
