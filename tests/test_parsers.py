import json

import numpy as np
import pandas as pd

from conftest import FIX
from laliga.data import espn, news, openfootball, results, weather


def test_football_data_standardize_with_odds():
    raw = pd.read_csv(FIX / "fd_sample.csv")
    df = results.standardize_football_data(raw, "football-data")
    assert list(df.columns) == results.COLUMNS
    assert df.loc[0, "date"] == pd.Timestamp("2025-08-15")
    assert df.loc[0, "season"] == 2025
    assert df.loc[0, "result"] == "A" and df.loc[1, "result"] == "D"
    # Pinnacle tiene prioridad; si falta se usa la media
    assert df.loc[0, "odds_h"] == 2.15
    assert df.loc[1, "odds_h"] == 1.72
    assert df.loc[2, "odds_h"] == 2.00  # solo Bet365
    assert df.loc[0, "odds_close_a"] == 3.50 and np.isnan(df.loc[1, "odds_close_h"])
    # hora UK 18:00 en agosto (BST) = 17:00 UTC
    assert df.loc[0, "kickoff"] == pd.Timestamp("2025-08-15 17:00", tz="UTC")
    assert pd.isna(df.loc[2, "hg"]) and pd.isna(df.loc[2, "result"])
    assert df.loc[1, "h_sot"] == 7 and df.loc[1, "a_red"] == 1


def test_mirror_format_iso_dates():
    raw = pd.DataFrame({"Date": ["2024-08-15"], "HomeTeam": ["Ath Bilbao"], "AwayTeam": ["Getafe"],
                        "FTHG": [1], "FTAG": [1], "Referee": [None]})
    df = results.standardize_football_data(raw, "mirror")
    assert df.loc[0, "date"] == pd.Timestamp("2024-08-15") and df.loc[0, "result"] == "D"
    assert df[["odds_h", "odds_d", "odds_a"]].isna().all().all()


def test_two_digit_years():
    raw = pd.DataFrame({"Date": ["27/08/05"], "HomeTeam": ["Alaves"], "AwayTeam": ["Barcelona"], "FTHG": [0], "FTAG": [0]})
    assert results.standardize_football_data(raw, "x").loc[0, "date"] == pd.Timestamp("2005-08-27")


def test_season_helpers():
    assert results.season_of(pd.Timestamp("2026-10-25")) == 2026
    assert results.season_of(pd.Timestamp("2027-03-01")) == 2026
    assert results.season_code(2005) == "0506" and results.season_code(2099) == "9900"


def test_openfootball_format_a():
    df = openfootball.parse_football_txt((FIX / "of_formatA.txt").read_text(), 2026)
    assert len(df) == 5
    first = df.iloc[0]
    assert (first.home, first.away, first.hg, first.ag) == ("Alaves", "Getafe", 3, 0)
    assert first.kickoff == pd.Timestamp("2026-08-15 17:30", tz="UTC")  # 19:30 CEST
    celta = df[df.home == "Celta"].iloc[0]
    assert celta.hg == 0 and celta.ag == 0 and pd.isna(celta.kickoff)
    fut = df[df.home == "Malaga"].iloc[0]
    assert fut.date == pd.Timestamp("2027-01-10") and pd.isna(fut.hg)  # el año se infiere


def test_openfootball_format_b():
    df = openfootball.parse_football_txt((FIX / "of_formatB.txt").read_text(), 2023)
    assert len(df) == 3
    assert df.iloc[0][["home", "away", "hg", "ag"]].tolist() == ["Almeria", "Vallecano", 0, 2]
    assert df.iloc[2][["home", "away", "hg", "ag"]].tolist() == ["Getafe", "Barcelona", 0, 0]


def test_espn_scoreboard():
    sb = espn.parse_scoreboard(json.loads((FIX / "espn_scoreboard.json").read_text()))
    assert len(sb) == 2  # el equipo desconocido se descarta sin romper
    clasico = sb[sb.event_id == "9001"].iloc[0]
    assert (clasico.home, clasico.away, clasico.state) == ("Barcelona", "Real Madrid", "pre")
    assert np.isnan(clasico.hg)  # partido no terminado: sin marcador aunque ESPN mande "0"
    done = sb[sb.event_id == "9000"].iloc[0]
    assert done.hg == 3 and done.ag == 1
    res = espn.to_results(sb)
    assert res[res.event_id == "9000"].iloc[0].result == "H"
    assert res[res.event_id == "9001"].iloc[0].date == pd.Timestamp("2026-10-25")


def test_espn_summary_lineups_and_referee():
    det = espn.parse_summary(json.loads((FIX / "espn_summary.json").read_text()), "9001")
    assert det.referee == "José Luis Munuera Montero"  # no el asistente
    assert det.lineup_confirmed
    assert sum(p.starter for p in det.lineups["home"]) == 11
    rows = espn.lineup_rows(det, "2026-10-25", "Barcelona", "Real Madrid")
    assert len(rows) == 36 and set(rows.team) == {"Barcelona", "Real Madrid"}


def test_espn_summary_before_lineups():
    det = espn.parse_summary(json.loads((FIX / "espn_summary_pre.json").read_text()), "1")
    assert not det.lineup_confirmed and det.referee is None


def test_lineup_store_roundtrip(settings):
    det = espn.parse_summary(json.loads((FIX / "espn_summary.json").read_text()), "9001")
    rows = espn.lineup_rows(det, "2026-10-25", "Barcelona", "Real Madrid")
    espn.save_lineups(settings, rows)
    espn.save_lineups(settings, rows)  # idempotente
    back = espn.load_lineups(settings)
    assert len(back) == 36 and back.starter.sum() == 22


def test_referee_overrides_win(settings):
    espn.save_referees(settings, pd.DataFrame([{"season": 2026, "home": "Barcelona", "away": "Real Madrid",
                                                "referee": "ESPN Ref"}]))
    (settings.data_dir / "referee_overrides.csv").write_text(
        "date,home,away,referee\n2026-10-25,FC Barcelona,Real Madrid CF,Manual Ref\n", encoding="utf-8")
    refs = espn.load_referees(settings)
    assert len(refs) == 1 and refs.iloc[0].referee == "Manual Ref"


def test_weather_at_kickoff():
    hourly = weather.parse_hourly(json.loads((FIX / "open_meteo.json").read_text()))
    w = weather.at_kickoff(hourly, pd.Timestamp("2026-10-25 15:15", tz="UTC"))
    assert w == {"temp_c": 20.5, "wind_kmh": 22.0, "precip_mm": 1.6}
    assert np.isnan(weather.at_kickoff(hourly, pd.Timestamp("2027-01-01", tz="UTC"))["temp_c"])


def test_rss_parse():
    items = news.parse_feed((FIX / "news_rss.xml").read_bytes(), "Barcelona", 10)
    assert len(items) == 4 and items[0].title.startswith("Baja de Pedri")
    assert "<" not in items[0].summary


def test_merge_sources_priority():
    base = dict(season=2026, date=pd.Timestamp("2026-10-18"), home="Real Madrid", away="Sevilla")
    fd = pd.DataFrame([{**base, "hg": np.nan, "ag": np.nan, "source": "football-data", "odds_h": 1.5}])
    es = pd.DataFrame([{**base, "hg": 3.0, "ag": 1.0, "result": "H", "source": "espn",
                        "kickoff": pd.Timestamp("2026-10-18 19:00", tz="UTC")}])
    of = pd.DataFrame([{**base, "hg": 2.0, "ag": 2.0, "result": "D", "source": "openfootball"}])
    m = results.merge_sources(fd, es, of)
    assert len(m) == 1
    r = m.iloc[0]
    assert (r.hg, r.ag, r.result, r.source) == (3, 1, "H", "espn")  # la primera fuente CON resultado
    assert r.odds_h == 1.5  # huecos completados desde otras fuentes
    assert r.kickoff == pd.Timestamp("2026-10-18 19:00", tz="UTC")


def test_espn_summary_without_homeaway_uses_order():
    js = json.loads((FIX / "espn_summary.json").read_text())
    for r in js["rosters"]:
        r.pop("homeAway")
        for p in r["roster"]:
            p["position"] = {"name": "Midfielder"}
            p["subbedIn"] = {"didSub": False}  # variante vista en la API
    det = espn.parse_summary(js, "9001")
    assert det.lineup_confirmed
    assert det.lineups["home"][0].name.startswith("Barcelona") and det.lineups["away"][0].name.startswith("Real Madrid")
    assert det.lineups["home"][0].position == "Midfielder"


def test_espn_scoreboard_falls_back_to_single_days(settings, monkeypatch):
    from datetime import date

    full = json.loads((FIX / "espn_scoreboard.json").read_text())
    calls = []

    class R:
        def __init__(self, js):
            self._js = js

        def json(self):
            return self._js

    def fake_get(url, params=None, **k):
        calls.append(params["dates"])
        if "-" in params["dates"]:
            return R({"events": []})  # rango no soportado
        return R(full if params["dates"] == "20261025" else {"events": []})

    monkeypatch.setattr(espn.http, "get", fake_get)
    sb = espn.scoreboard(settings, date(2026, 10, 24), date(2026, 10, 26))
    assert set(sb.event_id) == {"9000", "9001"}
    assert calls[0] == "20261024-20261026" and len(calls) == 4
