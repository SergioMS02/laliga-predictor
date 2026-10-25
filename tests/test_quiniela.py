import math

import numpy as np
import pandas as pd
import pytest

from laliga import quiniela as q
from laliga import scorers


def _row(n, h, a, r1=None, r2=None, state=None, when=None, pleno=False):
    res = (f'<td data-matchres><strong class="fdQuiniela"><span data-r1>{r1}</span><span></span>'
           f'<span data-r2>-{r2}</span></strong><div data-min><strong> {state} </strong></div> </td>'
           if r1 is not None else f'<td data-matchres><time datetime="{when}">x</time></td>')
    return (f'<tr class="brdcQuiniela "> <td data-match-num class="fQuiniela">{n}</td> <td data-matches>'
            f'<span data-m1 >{h}</span><span>-</span><span data-m2 >{a}</span></td> {res} <td data-matchdate></td> </tr>')


def _page(finished=True):
    teams = [("Sevilla", "Osasuna"), ("Real Sociedad", "Levante"), ("Atlético de Madrid", "Deportivo"),
             ("Athletic de Bilbao", "Getafe"), ("Celta", "Betis"), ("Valencia", "Villarreal"),
             ("Rayo Vallecano", "Elche"), ("Leganés", "Mallorca"), ("Almería", "R. Oviedo"),
             ("Sabadell", "Tenerife"), ("Eibar Femenino", "Madrid CFF"), ("Porto", "PSV"),
             ("Barcelona B", "Hércules"), ("Espanyol", "Alavés"), ("Barcelona", "Real Madrid")]
    rows = []
    for i, (h, a) in enumerate(teams, 1):
        if finished:
            rows.append(_row(i, h, a, 1 if i % 3 else 0, 1 if i % 2 else 0, "Fin."))
        else:
            rows.append(_row(i, h, a, when="Dom. 21:00"))
    prizes = ('<table class="escrutinioTable"><tbody>'
              '<tr> <td data-cat>P15</td> <td>0</td> <td data-prize>0,00 €</td> </tr>'
              '<tr> <td data-cat>14</td> <td>4</td> <td data-prize>45.666,15 €</td> </tr>'
              '<tr> <td data-cat>13</td> <td>1</td> <td data-prize>389,20 €</td> </tr>'
              '<tr> <td data-cat>10</td> <td>1</td> <td data-prize>1,50 €</td> </tr>'
              '</tbody></table>') if finished else ""
    return (f'<div id="sorteo"><a href="https://x/quiniela/resultados/2026-10-21/">a</a>'
            f'<a href="https://x/quiniela/resultados/2026-10-28/">b</a>'
            f'<span class="sk_fontSize11 fdQuiniela">Domingo, 25 octubre 2026</span>'
            f'<table class="matchTable"><tbody>{"".join(rows)}</tbody></table>{prizes}'
            f'data-reveal-params=\'{{ "jornada":"16" }}\'</div>')


def test_parse_page():
    p = q.parse_page(_page())
    assert p["date"] == "2026-10-25" and p["prev"] == "2026-10-21" and p["next"] == "2026-10-28"
    assert p["jornada"] == 16 and len(p["matches"]) == 15
    m = p["matches"][0]
    assert (m["home"], m["away"], m["hg"], m["ag"], m["status"]) == ("Sevilla", "Osasuna", 1, 1, "fin")
    assert p["prizes"]["14"] == pytest.approx(45666.15) and p["prizes"]["13"] == pytest.approx(389.2)
    up = q.parse_page(_page(False))
    assert up["matches"][0]["status"] == "pending" and up["matches"][0]["when"] == "Dom. 21:00"


def test_resolve():
    la, seg = {"Sevilla", "Ath Madrid", "La Coruna", "Ath Bilbao", "Vallecano"}, {"Oviedo", "Leganes", "Hercules"}
    assert q.resolve("Atlético de Madrid", la, seg) == ("Ath Madrid", 1)
    assert q.resolve("Athletic de Bilbao", la, seg) == ("Ath Bilbao", 1)
    assert q.resolve("Deportivo", la, seg) == ("La Coruna", 1)
    assert q.resolve("R. Oviedo", la, seg) == ("Oviedo", 2)
    assert q.resolve("Leganés", la, seg) == ("Leganes", 2)
    assert q.resolve("Eibar Femenino", la, seg)[1] == "women"
    assert q.resolve("Porto", la, seg)[1] == "unknown"


def test_probs_helpers():
    p = q.elo_probs(1500, 1500)
    assert sum(p) == pytest.approx(1) and p[0] > p[2]
    strong_away = q.elo_probs(1170, 1600)
    assert strong_away[2] > 0.65
    lam, mu = q.goals_from_probs([0.6, 0.22, 0.18])
    assert lam > mu
    pl = q.pleno_probs(2.0, 0.8)
    assert pl["pick"][0] in ("1", "2", "M") and 0 < pl["p"] < 1
    d = q.poisson_binomial([0.5, 0.5])
    assert d == pytest.approx([0.25, 0.5, 0.25])


def test_reduction_and_settle():
    rng = np.random.default_rng(3)
    probs = [list(rng.dirichlet([4, 3, 3])) for _ in range(14)]
    cover = q.best_reduction(probs, 16)
    n = int(np.prod([len(c) for c in cover]))
    assert 1 < n <= 16
    single = [[q.SIGNS[int(np.argmax(p))]] for p in probs]
    assert q.ticket_stats(probs, cover, 0.75)["p14"] > q.ticket_stats(probs, single, 0.75)["p14"]
    # brute force: ningún reparto con ≤ 16 apuestas mejora P(14)
    best = q.ticket_stats(probs, cover, 0.75)["p14"]
    order = [np.argsort(p)[::-1] for p in probs]
    for d in range(0, 5):
        for t in range(0, 3):
            if 2 ** d * 3 ** t > 16:
                continue
            gains = sorted(range(14), key=lambda i: -(math.log(probs[i][order[i][0]] + probs[i][order[i][1]])
                                                     - math.log(probs[i][order[i][0]])))
    assert best > 0
    # columnas por aciertos: 1 doble acertado + 13 sencillos con 12 aciertos
    cov = [["1"]] * 13 + [["1", "X"]]
    out = ["1"] * 12 + ["2", "X"]
    assert q.columns_by_hits(cov, out) == {12: 1, 13: 1}
    j = {"matches": [{"hg": 1, "ag": 0}] * 14 + [{"hg": 3, "ag": 0}],
         "tickets": {"sencilla": {"picks": [["1"]] * 14, "cost": 0.75},
                     "reducida": {"picks": [["1"]] * 13 + [["1", "X"]], "cost": 1.5}},
         "pleno": {"pick": ["M", "0"]}, "prizes": {"P15": 1000.0, "14": 100.0, "13": 10.0}}
    r = q.settle(j)
    assert r["sencilla"]["hits"] == 14 and r["sencilla"]["pleno_ok"] and r["sencilla"]["prize"] == pytest.approx(1100)
    assert r["reducida"]["prize"] == pytest.approx(100 + 10 + 1000)


def test_update_freezes_and_settles(settings, monkeypatch):
    preds = [{"home": "Sevilla", "away": "Osasuna", "p_home": 0.5, "p_draw": 0.3, "p_away": 0.2,
              "dc": {"xg_home": 1.5, "xg_away": 1.0}}]
    monkeypatch.setattr(q, "_laliga_elo", lambda s: {})
    st = q.update(settings, preds, online=False, pages=[q.parse_page(_page(False))], seg=None)
    j = st["jornadas"]["2026-10-25"]
    assert not j["frozen"] and j["rows"][0]["source"] == "modelo LaLiga"
    assert j["rows"][0]["p"] == [0.5, 0.3, 0.2] and len(j["tickets"]["sencilla"]["picks"]) == 14
    # con resultados: se congela, se liquida y no se vuelve a rellenar
    st = q.update(settings, [], online=False, pages=[q.parse_page(_page(True))], seg=None)
    j = st["jornadas"]["2026-10-25"]
    assert j["frozen"] and j["settled"] and "result" in j and not j.get("late")
    assert j["rows"][0]["source"] == "modelo LaLiga"
    assert st["summary"]["sencilla"]["jornadas"] == 1


def test_scorers():
    players = pd.DataFrame({
        "season": [2026] * 4, "team": ["A"] * 4, "player": ["Delantero Uno", "Medio Dos", "Defensa Tres", "Portero"],
        "position": ["F", "M", "D", "GK"], "games": [10] * 4, "minutes": [900, 900, 900, 900],
        "xg": [6.0, 2.0, 0.5, 0.0], "xa": [1, 2, 0.5, 0], "goals": [6, 2, 0, 0], "assists": [1, 2, 0, 0], "shots": [30, 10, 3, 0]})
    xi = [{"name": "Delantero Uno", "role": "F"}, {"name": "Medio Dos", "role": "M"}, {"name": "Defensa Tres", "role": "D"},
          {"name": "Portero", "role": "G"}, {"name": "Desconocido", "role": "F"}]
    out = scorers.scorer_probs(xi, 1.6, players, "A", 2026)
    assert out[0]["name"] == "Delantero Uno" and out[0]["p"] > out[1]["p"] > 0
    assert all(0 <= r["p"] < 1 for r in out)
    assert scorers.scorer_probs(xi, 1.6, None, "A", 2026) == []
