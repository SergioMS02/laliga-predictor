import json

import numpy as np
import pandas as pd

from laliga import props, site_export


def _hist(n=400, seed=0):
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(10)]
    rows = []
    for k in range(n):
        h, a = rng.choice(teams, 2, replace=False)
        rows.append({"season": 2025, "date": pd.Timestamp("2025-08-01") + pd.Timedelta(days=k), "home": h, "away": a,
                     "hg": rng.poisson(1.5), "ag": rng.poisson(1.1),
                     "h_yellow": rng.poisson(3.5 if h == "T0" else 2.0), "a_yellow": rng.poisson(2.3),
                     "h_red": 0, "a_red": 0, "h_corners": rng.poisson(5.5), "a_corners": rng.poisson(4.5),
                     "referee": "J. Munuera" if k % 3 == 0 else "R. Pérez"})
    return pd.DataFrame(rows)


def test_props_model():
    m = _hist()
    pm = props.PropsModel(m, pd.Timestamp("2026-09-10"))
    assert pm.ok
    p0 = pm.predict("T0", "T1", "Juan Martínez Munuera")
    p1 = pm.predict("T2", "T1", None)
    assert p0["yellow"][0] > p1["yellow"][0]  # T0 ve más tarjetas
    assert 0 < p0["p_cards_over"]["4.5"] < 1 and p0["corners_total"] > 6
    assert props.ref_key("J. Munuera") == props.ref_key("Juan Martínez Munuera") == "munuera"


def test_player_card_probs():
    cards = pd.DataFrame({"date": pd.to_datetime(["2026-09-01", "2026-09-08", "2026-09-15"]), "team": "T0",
                          "player": ["Duro Pérez", "Duro Pérez", "Otro"], "card": "yellow"})
    lu = pd.DataFrame({"event_id": [1, 1, 2, 2, 3, 3], "date": pd.to_datetime(["2026-09-01"] * 2 + ["2026-09-08"] * 2 + ["2026-09-15"] * 2),
                       "team": "T0", "name": ["Duro Pérez", "Otro"] * 3, "starter": True})
    out = props.player_card_probs(cards, lu, "T0", pd.Timestamp("2026-08-01"), ["Duro Pérez", "Otro", "Nuevo"], 2.5, 2.2)
    assert out[0]["name"] == "Duro Pérez" and out[0]["p"] > out[-1]["p"]


def test_track_props(settings):
    log = pd.DataFrame([{"generated_at": "2026-10-01T10:00", "season": 2026, "date": "2026-10-02", "home": "A", "away": "B",
                         "kickoff_utc": None, "lineup_confirmed": False, "xg_home": 1.5, "xg_away": 1.0, "p_over25": 0.6,
                         "p_btts": 0.5, "yellow_total": 4.0, "corners_total": 9.0, "p_cards_over45": 0.4,
                         "p_corners_over95": 0.45, "scorers": json.dumps([["home", "Delantero Uno", 0.4]])}] * 1)
    log.to_csv(settings.predictions_dir / "props_log.csv", index=False)
    m = pd.DataFrame({"season": [2026], "home": ["A"], "away": ["B"], "date": pd.to_datetime(["2026-10-02"]), "hg": [2.0],
                      "ag": [1.0], "h_yellow": [3.0], "a_yellow": [2.0], "h_corners": [6.0], "a_corners": [4.0]})
    out = site_export.track_props(settings, m)
    assert out["n"] == 1 and out["over25"]["obs"] == 1.0 and out["cards"]["obs"] == 5.0
