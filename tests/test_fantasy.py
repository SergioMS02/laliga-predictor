from datetime import date

import pandas as pd

from laliga.data import fantasy as f

INJ = """
<header class="title col-12 v3-widget-header"><img><span class="v3-widget-header__label"><a href="x">Real Madrid</a></span></header>
<div class="elemento lesionado col-12 rounded-card">
 <span class="probabilidad-widget"> <span class="prob-0">0%</span></span>
 <a href="x" class="jugador">Thibaut Courtois</a>
 <div class="comentario"><span class="lesion">Gripe</span>
 <span class="rounded-card__status gravedad-0">Baja confirmada para la jornada 10</span></div></div>
<div class="elemento lesionado col-12 rounded-card">
 <span class="probabilidad-widget"> <span class="prob-50">50%</span></span>
 <a href="x" class="jugador">Arda Güler</a>
 <div class="comentario"><span class="lesion">Molestias</span>
 <span class="rounded-card__status gravedad-1">Duda para la jornada 11</span></div></div>
<div class="elemento lesionado col-12 rounded-card">
 <a href="x" class="jugador">Rodrygo Goes</a>
 <div class="comentario"><span class="lesion">Rotura de ligamento</span>
 <span class="rounded-card__status gravedad-0">Baja hasta mediados de diciembre</span></div></div>
</section>
<header class="title col-12 v3-widget-header"><img><span class="v3-widget-header__label"><a href="x">Rayo</a></span></header>
<div class="elemento lesionado col-12 rounded-card">
 <a href="x" class="jugador">Randy Nteka</a>
 <div class="comentario"><span class="lesion">Esguince</span>
 <span class="rounded-card__status">Disponible para la jornada 11</span></div></div>
</section>
"""
SAN = """<h1>Sancionados - Jornada 11</h1>
<header class="title col-12 v3-widget-header"><img><span class="v3-widget-header__label"><a href="x">Real Madrid</a></span></header>
<div class="elemento sancionado col-12"><a href="x" class="jugador">Vinicius Junior</a>
<span class="sancion"> Sanción disciplinaria (3/4) </span></div>
</section>
<header class="title col-12 v3-widget-header"><img><span class="v3-widget-header__label"><a href="x">Getafe</a></span></header>
<div class="elemento sancionado col-12"><a href="x" class="jugador">Johan Mojica</a>
<span class="sancion"> Acumulación de tarjetas <div class="tarjetascontainer">1r ciclo</div></span></div>
</section>
"""


def test_parse():
    inj = f.parse_injuries(INJ, date(2026, 10, 25))
    assert list(inj.player) == ["Thibaut Courtois", "Arda Güler", "Rodrygo Goes"]  # Nteka ya disponible
    assert inj.iloc[0].jornadas == [10] and inj.iloc[1].status == "doubtful" and inj.iloc[1].start_prob == 50
    assert inj.iloc[2].until == date(2026, 12, 20)
    san = f.parse_sanctions(SAN)
    assert san.iloc[0].jornadas == [9, 10, 11, 12] and san.iloc[1].jornadas == [11]
    assert set(san.team) == {"Real Madrid", "Getafe"}


def test_for_match():
    av = pd.concat([f.parse_injuries(INJ, date(2026, 10, 25)), f.parse_sanctions(SAN)], ignore_index=True)
    dates = pd.date_range("2026-08-16", periods=12, freq="7D")
    m = pd.DataFrame({"season": 2026, "date": dates, "home": "Real Madrid", "away": [f"T{i}" for i in range(12)]})
    d10, d11, d13 = dates[9], dates[10], dates[12 - 1]
    p10 = set(f.for_match(av, m, "Real Madrid", 2026, d10).player)
    p11 = set(f.for_match(av, m, "Real Madrid", 2026, d11).player)
    assert {"Thibaut Courtois", "Vinicius Junior", "Rodrygo Goes"} <= p10 and "Arda Güler" not in p10
    assert {"Arda Güler", "Vinicius Junior", "Rodrygo Goes"} <= p11 and "Thibaut Courtois" not in p11
    assert f.page_jornada(av) == 9 or f.page_jornada(av) == 11
