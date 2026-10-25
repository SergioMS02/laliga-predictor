import pytest

from laliga.teams import STADIUMS, canonical, haversine_km, is_island

# Nombres reales vistos en football-data, openfootball y ESPN
VARIANTS = {
    "Ath Bilbao": ["Athletic Club", "Athletic Bilbao", "Ath Bilbao"],
    "Ath Madrid": ["Club Atlético de Madrid", "Atlético Madrid", "Atletico de Madrid", "Ath Madrid"],
    "Barcelona": ["FC Barcelona", "Barcelona"],
    "Betis": ["Real Betis Balompié", "Real Betis", "Betis"],
    "Celta": ["RC Celta de Vigo", "Celta Vigo", "Celta"],
    "La Coruna": ["RC Deportivo La Coruña", "Deportivo La Coruña", "La Coruna"],
    "Espanol": ["RCD Espanyol de Barcelona", "Espanyol", "Espanol"],
    "Vallecano": ["Rayo Vallecano de Madrid", "Rayo Vallecano", "Vallecano"],
    "Sociedad": ["Real Sociedad de Fútbol", "Real Sociedad", "Sociedad"],
    "Santander": ["Real Racing Club de Santander", "Racing Santander", "Santander"],
    "Alaves": ["Deportivo Alavés", "Alavés", "Alaves"],
    "Real Madrid": ["Real Madrid CF", "Real Madrid"],
    "Las Palmas": ["UD Las Palmas", "Las Palmas"],
    "Leganes": ["CD Leganés", "Leganés"],
    "Valladolid": ["Real Valladolid CF", "Real Valladolid"],
    "Mallorca": ["RCD Mallorca", "Mallorca"],
    "Girona": ["Girona FC", "Girona"],
    "Oviedo": ["Real Oviedo"],
    "Malaga": ["Málaga CF", "Malaga"],
    "Elche": ["Elche CF"],
    "Levante": ["Levante UD"],
    "Osasuna": ["CA Osasuna"],
    "Villarreal": ["Villarreal CF"],
    "Sevilla": ["Sevilla FC"],
    "Getafe": ["Getafe CF"],
    "Valencia": ["Valencia CF"],
    "Sp Gijon": ["Real Sporting de Gijón", "Sporting Gijon"],
}


@pytest.mark.parametrize("canon,names", VARIANTS.items())
def test_canonical(canon, names):
    for n in names:
        assert canonical(n) == canon, n


def test_unknown_team_raises():
    with pytest.raises(KeyError):
        canonical("Equipo Inventado XYZ")
    assert canonical("Equipo Inventado XYZ", strict=False) == "Equipo Inventado XYZ"


def test_madrid_teams_not_confused():
    assert canonical("Real Madrid") != canonical("Atlético Madrid")
    assert canonical("Rayo Vallecano") != canonical("Real Madrid")
    assert canonical("Espanyol") != canonical("FC Barcelona")


def test_geography():
    assert haversine_km("Real Madrid", "Ath Madrid") < 15
    assert 480 < haversine_km("Barcelona", "Real Madrid") < 520
    assert haversine_km("Las Palmas", "Real Madrid") > 1500
    assert is_island("Las Palmas") and is_island("Mallorca") and not is_island("Sevilla")
    assert all(-90 <= v[0] <= 90 and -180 <= v[1] <= 180 for v in STADIUMS.values())
