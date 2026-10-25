import pandas as pd

from conftest import FIX
from laliga.data import news

SQUAD = ["Pedri", "Raphinha", "Lamine Yamal", "Robert Lewandowski", "Frenkie de Jong"]


def _items():
    return news.parse_feed((FIX / "news_rss.xml").read_bytes(), "Barcelona", 10)


def test_rule_based_extraction():
    ab = news.rule_based_absences(_items(), SQUAD)
    got = dict(zip(ab.player, ab.status))
    assert got.get("Pedri") == "out"
    assert got.get("Raphinha") == "doubtful"
    assert "Lamine Yamal" not in got  # el texto "inyectado" no produce una baja
    assert "Flick" not in got


def test_rule_based_needs_known_squad():
    assert news.rule_based_absences(_items(), []).empty


def test_llm_json_validation():
    txt = 'Claro: {"absences": [{"player": "Pedri", "status": "out", "reason": "injury"},' \
          ' {"player": "X", "status": "maybe"}, {"player": 5, "status": "out"},' \
          ' {"player": "Raphinha", "status": "doubtful", "reason": "hack"}]} fin'
    df = news.parse_llm_json(txt, "Barcelona")
    assert list(df.player) == ["Pedri", "Raphinha"]
    assert list(df.reason) == ["injury", "other"]
    assert news.parse_llm_json("no json", "Barcelona").empty
    assert news.parse_llm_json('{"absences": "todos"}', "Barcelona").empty
    assert news.parse_llm_json("{roto", "Barcelona").empty


def test_extract_with_mocked_llm(settings, monkeypatch):
    settings.raw["news"]["llm_backend"] = "gemini"
    monkeypatch.setattr(news, "_call_llm", lambda s, p: '{"absences": [{"player": "Pedri González", "status": "out"},'
                                                         ' {"player": "Jugador Inventado", "status": "out"}]}')
    ab = news.extract_absences(settings, "Barcelona", _items(), SQUAD)
    assert list(ab.player) == ["Pedri"]  # se normaliza al nombre de la plantilla y se descarta el inventado


def test_extract_falls_back_when_llm_fails(settings, monkeypatch):
    settings.raw["news"]["llm_backend"] = "openai_compatible"

    def fail(*a, **k):
        raise RuntimeError("sin cuota")
    monkeypatch.setattr(news, "_call_llm", fail)
    ab = news.extract_absences(settings, "Barcelona", _items(), SQUAD)
    assert "Pedri" in set(ab.player)


def test_fetch_news_survives_network_errors(settings):
    assert news.fetch_team_news(settings, "Barcelona") == []


def test_google_news_url():
    u = news.google_news_url("Ath Madrid", 3)
    assert "Atl%C3%A9tico+de+Madrid" in u and "when%3A3d" in u and "hl=es" in u
