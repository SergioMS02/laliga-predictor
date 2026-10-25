import pandas as pd

from laliga.data import suspensions as sp
from laliga.data.news import NewsItem


def _it(title, pub="Wed, 14 Oct 2026 10:00:00 GMT", team="Real Madrid"):
    return NewsItem(team=team, title=title, summary="", published=pub, link="")


SQUAD = ["Vinícius Júnior", "Kylian Mbappé", "Jude Bellingham", "Federico Valverde"]


def test_parse_sanctions():
    items = [_it("Vinícius Jr. se pierde el Clásico ante el FC Barcelona tras recibir cuatro partidos de sanción"),
             _it("El Real Madrid recurrirá la sanción"),
             _it("Bellingham y Valverde, sancionados")]  # ambiguo: no se adivina
    f = sp.parse_sanctions(items, SQUAD)
    assert len(f) == 1 and f[0]["player"] == "Vinícius Júnior" and f[0]["games"] == 4


def test_news_games_left(monkeypatch, settings):
    m = pd.DataFrame({"date": pd.to_datetime(["2026-10-18", "2026-10-21"]), "home": ["Real Madrid", "X"],
                      "away": ["Sevilla", "Real Madrid"], "hg": [4, 1], "ag": [1, 2]})
    monkeypatch.setattr(sp, "parse_feed", lambda *a, **k: [
        _it("Vinícius Jr. se pierde el Clásico tras recibir cuatro partidos de sanción")])
    monkeypatch.setattr(sp.http, "get", lambda *a, **k: type("R", (), {"content": b""})())
    cards = pd.DataFrame({"date": pd.to_datetime(["2026-10-10"]), "team": "Real Madrid",
                          "player": ["Vinícius Júnior"], "card": ["red"]})
    out = sp.suspended_by_news(settings, "Real Madrid", SQUAD, m, pd.Timestamp("2026-10-25"), cards=cards)
    assert out and out[0]["player"] == "Vinícius Júnior" and "le quedan 2" in out[0]["source"]
    # sin roja previa no se cree al titular (agredido, rival con el mismo apellido...)
    assert sp.suspended_by_news(settings, "Real Madrid", SQUAD, m, pd.Timestamp("2026-10-25"), cards=cards.iloc[0:0]) == []


def test_cards():
    cards = pd.DataFrame({"date": pd.to_datetime(["2026-09-01", "2026-09-08", "2026-09-15", "2026-09-22", "2026-10-18",
                                                  "2026-10-18"]),
                          "team": "Real Madrid", "player": ["A", "A", "A", "A", "A", "B"],
                          "card": ["yellow", "yellow", "yellow", "yellow", "yellow", "red"]})
    m = pd.DataFrame({"date": pd.to_datetime(["2026-09-01", "2026-09-08", "2026-09-15", "2026-09-22", "2026-10-18"]),
                      "home": "Real Madrid", "away": "X", "hg": 1, "ag": 0})
    out = sp.suspended_by_cards(cards, m, "Real Madrid", pd.Timestamp("2026-10-25"))
    assert {o["player"] for o in out} == {"A", "B"}
    # ya cumplida si el equipo jugó después
    m2 = pd.concat([m, pd.DataFrame({"date": [pd.Timestamp("2026-10-22")], "home": "Real Madrid", "away": "Y", "hg": 0, "ag": 0})])
    assert sp.suspended_by_cards(cards, m2, "Real Madrid", pd.Timestamp("2026-10-25")) == []


def test_served_in_scheduled_match():
    cards = pd.DataFrame({"date": pd.to_datetime(["2026-10-18"]), "team": "Barcelona", "player": ["B"], "card": ["red"]})
    m = pd.DataFrame({"date": pd.to_datetime(["2026-10-18", "2026-10-25", "2026-11-01"]), "home": "Barcelona",
                      "away": ["X", "Real Madrid", "Alaves"], "hg": [1, None, None], "ag": [0, None, None]})
    assert sp.suspended_by_cards(cards, m, "Barcelona", pd.Timestamp("2026-10-25 20:00"))
    assert sp.suspended_by_cards(cards, m, "Barcelona", pd.Timestamp("2026-11-01 16:00")) == []


def test_same_surname_other_team():
    items = [_it("Facundo Bernal, sancionado con dos partidos tras su roja ante el Barça", team="Barcelona")]
    assert sp.parse_sanctions(items, ["Marc Bernal", "Pedri"]) == []
    items = [_it("Marc Bernal, sancionado con un partido", team="Barcelona")]
    assert sp.parse_sanctions(items, ["Marc Bernal", "Pedri"])[0]["player"] == "Marc Bernal"
