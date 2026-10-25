"""Historial de probabilidades y explicación de cambios importantes."""
import json

import numpy as np
import pandas as pd
import pytest

from conftest import FIX
from laliga import movements, pipeline, predict
from laliga.data import espn


@pytest.fixture
def trained(settings, league):
    settings.raw["backtest"].update({"first_oos_season": 2014, "first_val_season": 2015, "first_test_season": 2016})
    settings.raw["stacker"]["min_train_rows"] = 300
    settings.raw["weather"]["enabled"] = False
    settings.raw["news"]["enabled"] = False
    df, _ = league
    pipeline.backtest(settings, df)
    pipeline.train_final(settings)
    df = df.copy()
    up = df[df.season == df.season.max()].index[-10:]
    df.loc[up, ["hg", "ag", "h_shots", "a_shots", "h_sot", "a_sot"]] = np.nan
    df.loc[up, "result"] = None
    return settings, df, up


def _history_lineups(settings, target, home, away):
    rows = []
    for e in range(10):
        for team in (home, away):
            for i in range(16):
                rows.append({"event_id": f"h{e}", "date": target.date - pd.Timedelta(days=7 * (e + 1)), "team": team,
                             "player_id": f"{team}{i}", "name": f"{team} {i}", "position": "G" if i == 0 else "M",
                             "starter": i < 11})
    espn.save_lineups(settings, pd.DataFrame(rows))


def _fake_summary(home, away, rotate_home):
    summary = json.loads((FIX / "espn_summary.json").read_text())

    def f(settings, event_id):
        det = espn.parse_summary(summary, event_id)
        for side, team in (("home", home), ("away", away)):
            for i, p in enumerate(det.lineups[side]):
                p.player_id, p.name = f"{team}{i}", f"{team} {i}"
                p.starter = (5 <= i < 16) if (side == "home" and rotate_home) else i < 11
        return det
    return f


def test_snapshot_reproduces_prediction(trained):
    settings, df, up = trained
    now = pd.Timestamp(df.loc[up].date.min(), tz="Europe/Madrid").tz_convert("UTC")
    preds = predict.predict(settings, df, None, now=now, days=30, online=False)
    st = movements.load_stackers(settings)
    for p in preds:
        snap = p["_snapshot"]
        rep = movements.probs_from_snapshot(snap, p["model"], st, snap["a"], snap["b"])
        assert rep == pytest.approx([p["p_home"], p["p_draw"], p["p_away"]], abs=2e-4)


def test_history_and_explained_drop(trained, monkeypatch):
    settings, df, up = trained
    target = df.loc[up[0]]
    home, away = target.home, target.away
    _history_lineups(settings, target, home, away)
    sb = pd.DataFrame([{"event_id": "E1", "home": home, "away": away}])
    t1 = pd.Timestamp(target.date, tz="Europe/Madrid").tz_convert("UTC")

    # 1) sin alineación todavía
    monkeypatch.setattr(espn, "summary", lambda s, ev: espn.parse_summary({"rosters": []}, ev))
    p1 = predict.predict(settings, df, sb, now=t1, days=30, online=True)
    assert predict.save_predictions(settings, p1, matches=df) == []
    # repetir sin cambios no duplica el historial
    p1b = predict.predict(settings, df, sb, now=t1, days=30, online=True)
    predict.save_predictions(settings, p1b, matches=df)
    hist = movements.load_history(settings)
    assert len(hist) == len(p1)

    # 2) alineación oficial con 5 titulares habituales fuera (incluido el portero)
    monkeypatch.setattr(espn, "summary", _fake_summary(home, away, rotate_home=True))
    p2 = predict.predict(settings, df, sb, now=t1 + pd.Timedelta(minutes=30), days=30, online=True)
    big = predict.save_predictions(settings, p2, matches=df)
    mv = next(m for m in big if m["home"] == home and m["away"] == away)
    assert mv["important"] and mv["focus"] == "1"
    top = mv["reasons"][0]
    assert top["key"] == "bajas" and top["effect_pts"][0] < -2
    assert "alineaciones oficiales publicadas" in top["detail"]
    assert f"{home} 0" in top["detail"] and "no salen de inicio" in top["detail"]
    # la suma de efectos explica el cambio total
    total = sum(r["effect_pts"][0] for r in mv["reasons"])
    assert total == pytest.approx(mv["delta_pts"][0], abs=0.6)

    # ficheros e informes
    assert (settings.predictions_dir / "movements.jsonl").exists()
    md = (settings.predictions_dir / "latest.md").read_text()
    assert "## Cambios importantes" in md and "Bajas y alineaciones" in md and "→" in md
    evo = (settings.predictions_dir / "evolucion.md").read_text()
    assert f"## {home} – {away}" in evo and "✅" in evo
    assert len(movements.load_history(settings)) == len(p1) + 1


def _snap(**kw):
    s = {"raw": {"elo_diff": 80.0, "odds_h": 2.0, "odds_d": 3.4, "odds_a": 3.8, "precip_mm": 0.0},
         "lam0": 1.5, "mu0": 1.1, "sot_lam0": None, "sot_mu0": None, "xg_lam0": None, "xg_mu0": None,
         "d_home": [0.0, 0.0], "d_away": [0.0, 0.0], "rho": -0.05, "a": 0.6, "b": 0.35, "max_goals": 10,
         "absences": {}, "lineups": None, "expected_xi": {}, "lineup_confirmed": False, "referee": None}
    for k, v in kw.items():
        if k == "raw":
            s["raw"].update(v)
        else:
            s[k] = v
    return s


def _rec(snap, ts, p=None):
    p = p if p is not None else list(movements.probs_from_snapshot(snap, "dixon_coles", {}, 0.6, 0.35))
    return {"ts": ts, "season": 2026, "home": "Barcelona", "away": "Real Madrid", "date": "2026-10-25",
            "p": p, "model": "dixon_coles", "snapshot": snap}


def test_explain_results_and_news():
    old = _snap()
    new = _snap(lam0=1.2, mu0=1.3, raw={"elo_diff": 40.0},
                absences={"Barcelona": [{"player": "Pedri", "status": "out", "reason": "injury"}]},
                d_home=[0.15, 0.05])
    played = pd.DataFrame({"date": [pd.Timestamp("2026-10-22")], "home": ["Barcelona"], "away": ["Sevilla"],
                           "hg": [0.0], "ag": [3.0]})
    ex = movements.explain(_rec(old, "2026-10-20T08:00:00+00:00"), _rec(new, "2026-10-24T08:00:00+00:00"), {}, played)
    assert ex["focus"] == "1"
    keys = [r["key"] for r in ex["reasons"]]
    assert keys[:2] in (["resultados", "bajas"], ["bajas", "resultados"])
    det = {r["key"]: r["detail"] for r in ex["reasons"]}
    assert "Barcelona 0-3 Sevilla (22/10)" in det["resultados"]
    assert "baja: Pedri" in det["bajas"]


def test_explain_market_and_weather_without_effect_in_dc():
    # con el modelo Dixon-Coles puro, cuotas y clima no mueven nada -> no aparecen como causa
    old = _snap()
    new = _snap(raw={"odds_h": 2.6, "precip_mm": 8.0}, lam0=1.3)
    ex = movements.explain(_rec(old, "2026-10-20T08:00:00+00:00"), _rec(new, "2026-10-21T08:00:00+00:00"), {})
    assert [r["key"] for r in ex["reasons"]] == ["resultados"]


def test_sparkline_and_trend():
    assert movements.sparkline([0.0, 0.5, 0.99]) == "▁▅█"
    h = pd.DataFrame({"ts": ["a", "b"], "season": [2026, 2026], "home": ["X", "X"], "away": ["Y", "Y"],
                      "p1": [0.5, 0.45], "pX": [.25, .27], "p2": [.25, .28]})
    t = movements.trend_text(h, {"home": "X", "away": "Y", "season": 2026, "p_home": 0.41, "p_draw": .3, "p_away": .29})
    assert t.endswith("50→41% ▼")
    assert movements.trend_text(h.iloc[0:0], {"home": "X", "away": "Y", "season": 2026, "p_home": .4,
                                              "p_draw": .3, "p_away": .3}) == "—"
