"""Pruebas de extremo a extremo sobre una liga sintética (sin red)."""
import json

import numpy as np
import pandas as pd
import pytest

from conftest import FIX, make_league
from laliga import cli, history, pipeline, predict
from laliga.data import espn, news


@pytest.fixture
def bt_settings(settings):
    settings.raw["backtest"].update({"first_oos_season": 2014, "first_val_season": 2015, "first_test_season": 2016})
    settings.raw["stacker"]["min_train_rows"] = 300
    settings.raw["weather"]["enabled"] = False
    return settings


def test_walk_forward_is_out_of_sample(bt_settings, league):
    df, _ = league
    d1 = pipeline.walk_forward_dc(bt_settings, df, 2016)
    df2 = df.copy()
    late = df2.date >= pd.Timestamp("2017-01-01")
    df2.loc[late, "hg"] = 7.0
    d2 = pipeline.walk_forward_dc(bt_settings, df2, 2016)
    early = df.loc[d1.index].date < pd.Timestamp("2017-01-01")
    pd.testing.assert_frame_equal(d1[early], d2[early])
    assert d1[["dc_ph", "dc_pd", "dc_pa"]].sum(axis=1).sub(1).abs().max() < 1e-9
    assert d1.sot_lam.notna().mean() > 0.9


def test_backtest_beats_naive_baselines(bt_settings, league):
    df, _ = league
    r = pipeline.backtest(bt_settings, df)
    ov = r["overall"]
    assert ov["dixon_coles"]["log_loss"] < ov["frecuencias"]["log_loss"] < ov["uniforme"]["log_loss"]
    assert ov["stacker"]["log_loss"] < ov["frecuencias"]["log_loss"]
    assert "mercado" not in ov  # sin cuotas no se inventa un benchmark
    assert (bt_settings.processed_dir / "backtest.json").exists()
    info = pipeline.train_final(bt_settings)
    assert info["stacker_market"] is False and (bt_settings.state_dir / "stacker.pkl").exists()


def test_backtest_with_market(bt_settings):
    df, strength = make_league(seed=5)
    # cuotas "informadas": derivadas de las fuerzas reales con margen del 5 %
    from laliga.models.dixon_coles import DixonColes
    dc = DixonColes()
    dc.fit(df.head(0), "2012-01-01", current_teams=list(strength))
    probs = []
    for r in df.itertuples():
        ah, dh = strength[r.home]
        aa, da = strength[r.away]
        p = dc.predict(r.home, r.away, np.exp(0.25 + ah + da), np.exp(aa + dh))
        probs.append([p["p_home"], p["p_draw"], p["p_away"]])
    odds = 1 / (np.array(probs) * 1.05)
    df[["odds_h", "odds_d", "odds_a"]] = odds
    r = pipeline.backtest(bt_settings, df)
    ms = r["market_subset"]
    assert ms["mercado"]["log_loss"] < ms["dixon_coles"]["log_loss"]  # el mercado conoce la verdad
    assert ms["stacker_mercado"]["log_loss"] < ms["stacker"]["log_loss"]  # y el meta-modelo lo aprovecha


def _with_upcoming(df):
    df = df.copy()
    last_season = df.season.max()
    up_idx = df[df.season == last_season].index[-10:]
    df.loc[up_idx, ["hg", "ag", "result", "h_shots", "a_shots", "h_sot", "a_sot"]] = np.nan
    df.loc[up_idx, "result"] = None
    return df, up_idx


def test_predict_end_to_end_offline(bt_settings, league):
    df, _ = league
    pipeline.backtest(bt_settings, df)
    pipeline.train_final(bt_settings)
    df, up_idx = _with_upcoming(df)
    now = pd.Timestamp(df.loc[up_idx].date.min(), tz="Europe/Madrid").tz_convert("UTC")
    preds = predict.predict(bt_settings, df, None, now=now, days=30, online=False)
    assert len(preds) == 10
    for p in preds:
        assert abs(p["p_home"] + p["p_draw"] + p["p_away"] - 1) < 1e-3
        assert p["model"] == "stacker" and not p["lineup_confirmed"]
    predict.save_predictions(bt_settings, preds)
    saved = json.loads((bt_settings.predictions_dir / "latest.json").read_text())
    assert len(saved) == 10
    md = (bt_settings.predictions_dir / "latest.md").read_text()
    assert "| Fecha | Partido |" in md and md.count("\n|") >= 12
    assert len(pd.read_csv(bt_settings.predictions_dir / "log.csv")) == 10


def test_lineup_with_missing_stars_lowers_team(bt_settings, league, monkeypatch):
    """Con alineación confirmada sin los habituales, baja la probabilidad del equipo."""
    df, _ = league
    df, up_idx = _with_upcoming(df)
    target = df.loc[up_idx[0]]
    home, away = target.home, target.away
    # historial de alineaciones: p0..p10 titulares fijos en ambos equipos
    rows = []
    for e in range(10):
        for team in (home, away):
            for i in range(16):
                rows.append({"event_id": f"h{e}", "date": target.date - pd.Timedelta(days=7 * (e + 1)), "team": team,
                             "player_id": f"{team}{i}", "name": f"{team} {i}", "position": "M", "starter": i < 11})
    espn.save_lineups(bt_settings, pd.DataFrame(rows))
    summary = json.loads((FIX / "espn_summary.json").read_text())

    def fake_summary(settings, event_id, rotate_home):
        det = espn.parse_summary(summary, event_id)
        for side, team in (("home", home), ("away", away)):
            for i, p in enumerate(det.lineups[side]):
                p.player_id = f"{team}{i}"
                p.starter = (i >= 5 and i < 16) if (side == "home" and rotate_home) else i < 11
        return det

    sb = pd.DataFrame([{"event_id": "E1", "home": home, "away": away}])
    now = pd.Timestamp(target.date, tz="Europe/Madrid").tz_convert("UTC")
    bt_settings.raw["news"]["enabled"] = False

    def run(rotate):
        monkeypatch.setattr(espn, "summary", lambda s, ev: fake_summary(s, ev, rotate))
        preds = predict.predict(bt_settings, df, sb, now=now, days=30, online=True)
        return next(p for p in preds if p["home"] == home and p["away"] == away)

    full, rotated = run(False), run(True)
    assert full["lineup_confirmed"] and rotated["lineup_confirmed"]
    assert full["deficit"][0] == pytest.approx(0, abs=1e-9)
    assert rotated["deficit"][0] > 0.3
    assert rotated["p_home"] < full["p_home"] - 0.03
    assert rotated["dc"]["xg_home"] < full["dc"]["xg_home"]
    assert full["referee"] == "José Luis Munuera Montero"


def test_news_absences_flow(bt_settings, league, monkeypatch):
    df, _ = league
    df, up_idx = _with_upcoming(df)
    target = df.loc[up_idx[0]]
    home = target.home
    rows = [{"event_id": f"h{e}", "date": target.date - pd.Timedelta(days=7 * (e + 1)), "team": home,
             "player_id": f"id{i}", "name": n, "position": "M", "starter": i < 11}
            for e in range(5) for i, n in enumerate(["Pedri", "Raphinha"] + [f"Otro {k}" for k in range(12)])]
    espn.save_lineups(bt_settings, pd.DataFrame(rows))
    items = news.parse_feed((FIX / "news_rss.xml").read_bytes(), home, 10)
    monkeypatch.setattr(news, "fetch_team_news", lambda s, t: items if t == home else [])
    now = pd.Timestamp(target.date, tz="Europe/Madrid").tz_convert("UTC")
    preds = predict.predict(bt_settings, df, None, now=now, days=30, online=True)
    p = next(p for p in preds if p["home"] == home)
    players = {a["player"]: a["status"] for a in p["absences"][home]}
    assert players == {"Pedri": "out", "Raphinha": "doubtful"}
    assert p["deficit"][0] == pytest.approx((1 + 0.5) / 11, abs=1e-3)
    assert p["availability_source"] == "noticias"


def test_availability_dataset_and_live_eval(bt_settings, league):
    df, _ = league
    pipeline.backtest(bt_settings, df)
    oos = pd.read_csv(bt_settings.processed_dir / "oos_table.csv", parse_dates=["date"])
    rows = []
    for r in df[df.season >= 2016].itertuples():
        for team in (r.home, r.away):
            for i in range(12):
                rows.append({"event_id": f"{r.Index}", "date": r.date, "team": team, "player_id": f"{team}{i}",
                             "name": f"{team} {i}", "position": "M", "starter": i < 11})
    data = history.availability_dataset(bt_settings, df, oos, pd.DataFrame(rows))
    assert len(data) > 200 and data[["d_home", "d_away"]].max().max() < 1e-9
    # evaluación en vivo
    log = df[df.season == 2018].head(20)[["season", "date", "home", "away"]].assign(
        generated_at="2018-01-01T00:00:00", p_home=0.45, p_draw=0.27, p_away=0.28, model="stacker", lineup_confirmed=False)
    log.to_csv(bt_settings.predictions_dir / "log.csv", index=False)
    ev = history.evaluate_live(bt_settings, df)
    assert ev["n"] == 20 and 0.8 < ev["log_loss"] < 1.4


def test_cli_smoke(bt_settings, league, monkeypatch, capsys):
    df, _ = league
    df, _ = _with_upcoming(df)
    monkeypatch.setattr(cli, "load_settings", lambda **k: bt_settings)
    monkeypatch.setattr(pipeline, "load_matches", lambda s, online=True, days_ahead=None: (df, pd.DataFrame()))
    import laliga.predict as P
    real = P.predict
    monkeypatch.setattr(P, "predict", lambda s, m, sb, **k: real(
        s, m, sb, now=pd.Timestamp(df[df.hg.isna()].date.min(), tz="UTC"), days=30, online=False))
    assert cli.main(["all", "--offline"]) == 0
    out = capsys.readouterr().out
    assert "Backtest walk-forward" in out and "| Fecha | Partido |" in out
    assert cli.main(["evaluate-live", "--offline"]) == 0


def test_tune_end_to_end(bt_settings, league, monkeypatch):
    from laliga import tune
    df, _ = league
    bt_settings.raw["backtest"].update({"first_oos_season": 2014, "first_val_season": 2015, "first_test_season": 2018})
    monkeypatch.setattr(tune, "DC_GRID", {"xi": [0.0019], "ridge": [6.0], "window_days": [1100, 1500]})
    monkeypatch.setattr(tune, "C_GRID", [0.03])
    out = tune.tune(bt_settings, df)
    assert set(out["dixon_coles"]) == {"xi", "ridge", "window_days"}
    assert set(out["stacker"]["groups"]) <= set(tune.GROUPS)
    assert "xg_model" not in out["stacker"]["groups"]  # sin datos -> no elegible
    assert (bt_settings.state_dir / "tuned.json").exists()
    assert out["validation"]["seasons"] == [2015, 2016, 2017]
