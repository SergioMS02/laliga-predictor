"""Fuentes y variables añadidas: Europa/Copa, Understat, entrenadores, contexto, bajas ataque/defensa."""
import json

import numpy as np
import pandas as pd
import pytest

from conftest import FIX
from laliga.data import coaches, other_comps, understat
from laliga.features import availability as av
from laliga.features.context import _stakes, build_context_features, coverage_seasons
from laliga.models.stacker import GROUPS, Stacker, engineer
from laliga.teams import is_derby


# --------------------------------------------------------------------------- datos
def test_parse_champions():
    df = other_comps.parse_comp_txt((FIX / "cl_sample.txt").read_text(), 2023, "cl")
    got = set(zip(df.date.dt.strftime("%Y-%m-%d"), df.team))
    assert got == {("2023-09-19", "Ath Madrid"), ("2023-09-19", "Barcelona"),
                   ("2023-09-20", "Real Madrid"), ("2024-04-17", "Real Madrid")}


def test_parse_cup_ignores_lower_divisions():
    df = other_comps.parse_comp_txt((FIX / "cup_sample.txt").read_text(), 2023, "cup")
    # "Atlético de Lugones" o "Real Madrid Castilla" NO son el Atlético ni el Real Madrid
    assert set(df.team) == {"Espanol", "Vallecano", "Ath Bilbao"}
    assert (df.date.dt.month == [10, 11, 11]).all()


def test_understat_parse_and_attach():
    js = json.loads((FIX / "understat_league.json").read_text())
    m, p = understat.parse_league(js, 2025)
    assert len(m) == 2  # el partido sin jugar se ignora
    r = m[m.home == "Barcelona"].iloc[0]
    assert (r.away, r.h_xg, r.a_xg) == ("Ath Madrid", 2.31, 0.84)
    assert set(p.team) == {"Barcelona"}  # traspasado -> último equipo; desconocido fuera
    assert p[p.player == "Robert Lewandowski"].minutes.iloc[0] == 2400
    base = pd.DataFrame({"season": [2025, 2025], "home": ["Barcelona", "Sevilla"], "away": ["Ath Madrid", "Betis"]})
    out = understat.attach_xg(base, m)
    assert out.h_xg.iloc[0] == 2.31 and np.isnan(out.h_xg.iloc[1])


def test_understat_offline_without_cache(settings):
    assert understat.fetch_season(settings, 2020, online=False) is None
    m, p = understat.load_understat(settings, online=False)
    assert m.empty and p.empty


def test_wikidata_coaches_and_manual_priority(settings, monkeypatch):
    js = json.loads((FIX / "wikidata_coaches.json").read_text())
    df = coaches.parse_wikidata(js)
    assert set(df.team) == {"Real Madrid", "Sevilla"}
    (settings.data_dir / "coaches_manual.csv").write_text("team,date,coach\nSevilla FC,2025-10-01,Manual\n", encoding="utf-8")

    class R:
        def json(self):
            return js
    monkeypatch.setattr(coaches.http, "get", lambda *a, **k: R())
    allc = coaches.load_coaches(settings, online=True)
    sev = allc[allc.team == "Sevilla"]
    assert len(sev) == 1 and sev.coach.iloc[0] == "Manual"


# --------------------------------------------------------------------------- contexto
def test_context_no_leakage(league):
    df, _ = league
    cut = pd.Timestamp("2016-02-01")
    f1 = build_context_features(df)
    df2 = df.copy()
    fut = df2.date >= cut
    df2.loc[fut, ["hg", "ag", "h_red", "h_yellow"]] = [9.0, 0.0, 1.0, 8.0]
    f2 = build_context_features(df2)
    past = df.date < cut
    pd.testing.assert_frame_equal(f1[past], f2[past])


def test_stakes_logic():
    # 20 equipos, quedan 2 jornadas: el líder con 10 puntos de ventaja ya no se juega nada
    table = {f"T{i}": [60 - 2 * i, 0, 36] for i in range(20)}
    table["T0"][0] = 80
    st_leader, left = _stakes(table, "T0", 38)
    assert left == 2 and st_leader == 0
    # dos equipos empatados en la frontera del descenso sí se juegan mucho
    table["T16"][0] = table["T17"][0] = 26
    st_rel, _ = _stakes(table, "T17", 38)
    assert st_rel > 0.9
    # al principio de la temporada todo está abierto
    early = {f"T{i}": [0, 0, 0] for i in range(20)}
    assert _stakes(early, "T5", 38)[0] == pytest.approx(1)


def test_h2h_red_and_derby():
    rows = [
        ("2015-09-01", "Sevilla", "Betis", 3, 0, 0, 1),
        ("2016-02-01", "Betis", "Sevilla", 0, 2, 0, 0),
        ("2016-02-08", "Betis", "Celta", 1, 1, 0, 0),
        ("2016-09-01", "Sevilla", "Betis", np.nan, np.nan, np.nan, np.nan),
    ]
    df = pd.DataFrame(rows, columns=["date", "home", "away", "hg", "ag", "h_red", "a_red"])
    df["date"] = pd.to_datetime(df.date)
    df["season"] = [2015, 2015, 2015, 2016]
    f = build_context_features(df)
    last = f.iloc[3]
    assert last.h2h_gd == pytest.approx((3 + 2) / (2 + 4))  # Sevilla ganó las dos
    assert last.derby == 1 and f.iloc[2].derby == 0
    assert f.iloc[1].red_prev_home == 1.0  # Betis vio una roja en su partido anterior
    assert f.iloc[2].red_prev_home == 0.0  # y en el siguiente ya no
    assert is_derby("Real Madrid", "Getafe") and not is_derby("Barcelona", "Sevilla")


def test_congestion_and_coach(league):
    df, _ = league
    s = df.season.max()
    target = df[df.season == s].iloc[50]
    other = pd.DataFrame({
        "date": [target.date - pd.Timedelta(days=3), target.date + pd.Timedelta(days=3)],
        "team": [target.home, target.away], "comp": ["cl", "el"]})
    # cobertura completa de esa temporada (cl+el+cup)
    other = pd.concat([other, pd.DataFrame({"date": [pd.Timestamp(f"{s}-12-01")], "team": ["Nadie"], "comp": ["cup"]})])
    assert coverage_seasons(other) == {s}
    co = pd.DataFrame({"team": [target.away], "date": [target.date - pd.Timedelta(days=10)], "coach": ["Nuevo"]})
    f = build_context_features(df, other, co)
    r = f.loc[target.name]
    assert r.euro_prev_home == 1 and r.euro_prev_away == 0
    assert r.euro_next_away == 1 and r.euro_next_home == 0
    assert r.new_coach_away == 1 and r.new_coach_home == 0
    assert r.rest_all_diff < 0  # el local descansó menos
    other_season = f[df.season < s]
    assert other_season.euro_prev_home.isna().all()  # sin cobertura -> desconocido, no 0


# --------------------------------------------------------------------------- bajas
def test_roles():
    assert [av.role(p) for p in ("G", "GK", "Goalkeeper", "CB", "LB", "D", "CD-L", "DM", "CM", "AM", "M",
                                 "LW", "ST", "F", "CF-R", "Forward", "")] == \
        ["G", "G", "G", "D", "D", "D", "D", "M", "M", "M", "M", "F", "F", "F", "F", "F", "M"]


def _imp():
    rows = []
    pos = ["G", "D", "D", "D", "D", "M", "M", "M", "F", "F", "F", "G", "F"]
    for e in range(10):
        for i, p in enumerate(pos):
            rows.append({"event_id": f"e{e}", "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=7 * e),
                         "team": "Barcelona", "player_id": f"p{i}", "name": f"Jugador {i}", "position": p,
                         "starter": i < 11})
    return av.player_importance(pd.DataFrame(rows), "Barcelona", "2027-01-01")


def test_split_deficits_goalkeeper_vs_striker():
    imp = av.attach_production(_imp(), None)
    xi = [f"p{i}" for i in range(11)]
    assert av.deficits_from_lineup(imp, xi) == pytest.approx((0.0, 0.0), abs=1e-12)
    no_gk = ["p11"] + xi[1:]  # portero suplente (conocido, 0 titularidades)
    da, dd = av.deficits_from_lineup(imp, no_gk)
    assert dd > 0.1 and da < 0.01
    no_st = xi[:8] + ["p12", "p9", "p10"]
    da2, dd2 = av.deficits_from_lineup(imp, no_st)
    assert da2 > 0.1 and dd2 < 0.05
    ab = pd.DataFrame([{"player": "Jugador 8", "status": "out"}])
    a3, d3 = av.deficits_from_absences(imp, ab)
    assert a3 == pytest.approx(da2, rel=1e-6)


def test_production_from_understat():
    imp = _imp()
    players = pd.DataFrame([{"season": 2025, "team": "Barcelona", "player": "Jugador 8", "position": "F",
                             "games": 30, "minutes": 2700, "xg": 25.0, "xa": 5.0},
                            {"season": 2025, "team": "Barcelona", "player": "Jugador 9", "position": "F",
                             "games": 30, "minutes": 2700, "xg": 1.0, "xa": 0.5}])
    p = av.attach_production(imp, players)
    w = dict(zip(p.name, p.att_w))
    assert w["Jugador 8"] == pytest.approx(1.0) and w["Jugador 9"] == pytest.approx(0.05)
    xi = [f"p{i}" for i in range(11)]
    lose_star = av.deficits_from_lineup(p, [x for x in xi if x != "p8"] + ["p12"])[0]
    lose_other = av.deficits_from_lineup(p, [x for x in xi if x != "p9"] + ["p12"])[0]
    assert lose_star > 3 * lose_other  # perder al goleador pesa mucho más


def test_adjust_goals_split():
    lam, mu = av.adjust_goals(1.5, 1.0, (0.3, 0.0), (0.0, 0.3), 0.6, 0.35)
    assert lam == pytest.approx(1.5 * np.exp(-0.18 + 0.105))
    assert mu == pytest.approx(1.0)


def test_calibrate_split_columns():
    rng = np.random.default_rng(4)
    n = 4000
    d = pd.DataFrame({"lam": rng.uniform(1, 2, n), "mu": rng.uniform(0.7, 1.5, n),
                      "datt_home": rng.uniform(0, .4, n), "ddef_home": rng.uniform(0, .4, n),
                      "datt_away": rng.uniform(0, .4, n), "ddef_away": rng.uniform(0, .4, n)})
    d["hg"] = rng.poisson(d.lam * np.exp(-0.8 * d.datt_home + 0.5 * d.ddef_away))
    d["ag"] = rng.poisson(d.mu * np.exp(-0.8 * d.datt_away + 0.5 * d.ddef_home))
    a, b = av.calibrate(d, prior_strength=0.1)
    assert a == pytest.approx(0.8, abs=0.2) and b == pytest.approx(0.5, abs=0.2)


# --------------------------------------------------------------------------- meta-modelo
def test_every_group_is_engineered(league):
    df, _ = league
    tab = df.head(50).assign(dc_ph=.45, dc_pd=.27, dc_pa=.28, dc_lam=1.4, dc_mu=1.1, elo_diff=60.0, rest_home=7,
                             rest_away=7, travel_km=300.0, island=0.0, new_home=0.0, new_away=0.0)
    X = engineer(tab)
    for g, cols in GROUPS.items():
        assert set(cols) <= set(X.columns), g


def test_stacker_drops_empty_groups(league):
    df, _ = league
    seq_cols = {"dc_ph": .45, "dc_pd": .27, "dc_pa": .28, "dc_lam": 1.4, "dc_mu": 1.1}
    t = df.assign(**seq_cols)
    from laliga.features.sequential import build_sequential_features
    t = t.join(build_sequential_features(df)).join(build_context_features(df))
    st = Stacker(groups=list(GROUPS)).fit(t[t.result.notna()])
    assert "xg_model_diff" not in st.cols  # sin datos de xG el grupo se descarta solo
    assert "h2h_gd" in st.cols and "stake_diff_late" in st.cols
    p = st.predict_proba(t.tail(5))
    assert np.allclose(p.sum(axis=1), 1)


def test_odds_movement(settings):
    from laliga import predict
    o1 = pd.DataFrame([{"season": 2026, "home": "Barcelona", "away": "Real Madrid", "odds_h": 2.0, "odds_d": 3.6, "odds_a": 3.5}])
    o2 = o1.assign(odds_h=1.8, odds_a=4.2)
    predict.record_odds_snapshot(settings, o1, pd.Timestamp("2026-10-20", tz="UTC"))
    predict.record_odds_snapshot(settings, o2, pd.Timestamp("2026-10-25", tz="UTC"))
    mv = predict.odds_movement(settings).iloc[0]
    assert (mv.odds_h, mv.odds_close_h, mv.odds_close_a) == (2.0, 1.8, 4.2)


def test_espn_other_matches_keeps_only_laliga(settings, monkeypatch):
    from datetime import date
    from laliga.data import espn
    js = json.loads((FIX / "espn_scoreboard.json").read_text())
    js["events"][2]["competitions"][0]["competitors"][0]["team"]["displayName"] = "Bayer Leverkusen"
    leagues = []

    def fake_scoreboard(s, d0, d1, only_known=False):
        leagues.append(s.get("espn", "league"))
        assert only_known
        return espn.parse_scoreboard(js, only_known=True)
    monkeypatch.setattr(espn, "scoreboard", fake_scoreboard)
    df = other_comps.espn_other_matches(settings, date(2026, 10, 1), date(2026, 10, 30))
    assert set(leagues) == set(other_comps.ESPN_LEAGUES)
    assert "Bayer Leverkusen" not in set(df.team) and {"Barcelona", "Real Madrid", "Celta"} <= set(df.team)
    other_comps.save_espn_other(settings, df)
    loaded = other_comps.load_other_matches(settings, 2026, online=False)
    assert len(loaded) == len(df.drop_duplicates(["date", "team"]))
