"""Normalización de nombres de equipo entre fuentes y datos de estadios.

Nombre canónico = el de football-data.co.uk (p. ej. "Ath Madrid", "Sociedad").
"""
from __future__ import annotations

import math
import re
import unicodedata
from functools import lru_cache

from rapidfuzz import fuzz, process

# canónico -> (lat, lon, estadio, isla)
STADIUMS: dict[str, tuple[float, float, str, bool]] = {
    "Alaves": (42.8372, -2.6880, "Mendizorroza", False),
    "Albacete": (38.9893, -1.8552, "Carlos Belmonte", False),
    "Almeria": (36.8400, -2.4350, "Estadio de los Juegos Mediterráneos", False),
    "Ath Bilbao": (43.2641, -2.9494, "San Mamés", False),
    "Ath Madrid": (40.4362, -3.5995, "Metropolitano", False),
    "Barcelona": (41.3809, 2.1228, "Spotify Camp Nou", False),
    "Betis": (37.3565, -5.9817, "Benito Villamarín", False),
    "Cadiz": (36.5027, -6.2727, "Nuevo Mirandilla", False),
    "Celta": (42.2118, -8.7397, "Balaídos", False),
    "Cordoba": (37.8726, -4.7613, "Nuevo Arcángel", False),
    "Eibar": (43.1817, -2.4757, "Ipurua", False),
    "Elche": (38.2671, -0.6634, "Martínez Valero", False),
    "Espanol": (41.3480, 2.0756, "RCDE Stadium", False),
    "Getafe": (40.3257, -3.7146, "Coliseum", False),
    "Gimnastic": (41.1270, 1.2390, "Nou Estadi de Tarragona", False),
    "Girona": (41.9613, 2.8287, "Montilivi", False),
    "Granada": (37.1530, -3.5957, "Nuevo Los Cármenes", False),
    "Hercules": (38.3581, -0.4908, "José Rico Pérez", False),
    "Huesca": (42.1313, -0.4250, "El Alcoraz", False),
    "La Coruna": (43.3687, -8.4174, "Riazor", False),
    "Las Palmas": (28.1003, -15.4567, "Estadio de Gran Canaria", True),
    "Leganes": (40.3403, -3.7604, "Butarque", False),
    "Levante": (39.4948, -0.3641, "Ciutat de València", False),
    "Malaga": (36.7340, -4.4265, "La Rosaleda", False),
    "Mallorca": (39.5899, 2.6302, "Son Moix", True),
    "Murcia": (38.0420, -1.1430, "Enrique Roca", False),
    "Numancia": (41.7680, -2.4740, "Los Pajaritos", False),
    "Osasuna": (42.7966, -1.6370, "El Sadar", False),
    "Oviedo": (43.3608, -5.8700, "Carlos Tartiere", False),
    "Real Madrid": (40.4531, -3.6883, "Santiago Bernabéu", False),
    "Recreativo": (37.2470, -6.9540, "Nuevo Colombino", False),
    "Santander": (43.4762, -3.7935, "El Sardinero", False),
    "Sevilla": (37.3840, -5.9706, "Ramón Sánchez-Pizjuán", False),
    "Sociedad": (43.3014, -1.9737, "Anoeta", False),
    "Sp Gijon": (43.5362, -5.6372, "El Molinón", False),
    "Tenerife": (28.4632, -16.2612, "Heliodoro Rodríguez López", True),
    "Valencia": (39.4746, -0.3583, "Mestalla", False),
    "Valladolid": (41.6444, -4.7612, "José Zorrilla", False),
    "Vallecano": (40.3919, -3.6587, "Vallecas", False),
    "Villarreal": (39.9441, -0.1036, "La Cerámica", False),
    "Xerez": (36.7030, -6.1200, "Chapín", False),
    "Zaragoza": (41.6366, -0.9018, "La Romareda", False),
    "Burgos": (42.3450, -3.6890, "El Plantío", False),
    "Mirandes": (42.6800, -2.9420, "Anduva", False),
    "Castellon": (39.9930, -0.0430, "Castalia", False),
}

CANONICAL = sorted(STADIUMS)

# Alias explícitos (en minúsculas y sin acentos) -> canónico
_ALIASES = {
    "athletic club": "Ath Bilbao", "athletic bilbao": "Ath Bilbao", "athletic": "Ath Bilbao",
    "ath bilbao": "Ath Bilbao", "bilbao": "Ath Bilbao",
    "atletico madrid": "Ath Madrid", "atletico de madrid": "Ath Madrid",
    "club atletico de madrid": "Ath Madrid", "ath madrid": "Ath Madrid", "atletico": "Ath Madrid",
    "atleti": "Ath Madrid", "atl madrid": "Ath Madrid",
    "fc barcelona": "Barcelona", "barcelona": "Barcelona", "barca": "Barcelona",
    "real betis": "Betis", "real betis balompie": "Betis", "betis": "Betis",
    "celta vigo": "Celta", "celta de vigo": "Celta", "rc celta de vigo": "Celta", "celta": "Celta",
    "deportivo la coruna": "La Coruna", "rc deportivo la coruna": "La Coruna",
    "deportivo": "La Coruna", "la coruna": "La Coruna", "depor": "La Coruna",
    "rcd espanyol de barcelona": "Espanol", "espanyol": "Espanol", "rcd espanyol": "Espanol",
    "espanol": "Espanol",
    "rayo vallecano": "Vallecano", "rayo vallecano de madrid": "Vallecano", "vallecano": "Vallecano",
    "rayo": "Vallecano",
    "real sociedad": "Sociedad", "real sociedad de futbol": "Sociedad", "sociedad": "Sociedad",
    "alaves": "Alaves", "deportivo alaves": "Alaves",
    "racing santander": "Santander", "real racing club de santander": "Santander",
    "racing de santander": "Santander", "racing": "Santander", "santander": "Santander",
    "sporting gijon": "Sp Gijon", "real sporting de gijon": "Sp Gijon", "sporting de gijon": "Sp Gijon",
    "sp gijon": "Sp Gijon",
    "real valladolid": "Valladolid", "real valladolid cf": "Valladolid",
    "real oviedo": "Oviedo", "real zaragoza": "Zaragoza", "real mallorca": "Mallorca",
    "rcd mallorca": "Mallorca", "real madrid": "Real Madrid", "real madrid cf": "Real Madrid",
    "ud las palmas": "Las Palmas", "las palmas": "Las Palmas",
    "cd leganes": "Leganes", "leganes": "Leganes", "ud almeria": "Almeria",
    "cadiz cf": "Cadiz", "granada cf": "Granada", "sd eibar": "Eibar", "sd huesca": "Huesca",
    "cd tenerife": "Tenerife", "recreativo de huelva": "Recreativo", "recreativo huelva": "Recreativo",
    "real murcia": "Murcia", "hercules cf": "Hercules", "gimnastic de tarragona": "Gimnastic",
    "nastic": "Gimnastic", "xerez cd": "Xerez", "cd numancia": "Numancia", "cordoba cf": "Cordoba",
    "albacete balompie": "Albacete", "burgos cf": "Burgos", "cd mirandes": "Mirandes",
    "cd castellon": "Castellon",
    "espanyol barcelona": "Espanol", "athletic club bilbao": "Ath Bilbao", "real club celta de vigo": "Celta",
    "racing club de santander": "Santander", "sporting gijon": "Sp Gijon",
    "sd huesca": "Huesca", "deportivo alaves": "Alaves",
}

_STRIP_TOKENS = {"cf", "fc", "ud", "cd", "rcd", "rc", "sd", "ca", "sad", "club", "de", "balompie", "futbol"}


def _plain(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-zA-Z0-9 ]+", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


@lru_cache(maxsize=8192)
def canonical(name: str, strict: bool = True, fuzzy: bool = True) -> str:
    """Devuelve el nombre canónico de un equipo a partir de cualquier variante.

    fuzzy=False solo acepta coincidencias exactas (útil con competiciones donde
    aparecen equipos de otras categorías, p. ej. "Atlético de Lugones" en Copa)."""
    if name in STADIUMS:
        return name
    p = _plain(re.sub(r"\((?:[A-Z]{3})\)", "", name))
    if p in _ALIASES:
        return _ALIASES[p]
    stripped = " ".join(t for t in p.split() if t not in _STRIP_TOKENS)
    if stripped in _ALIASES:
        return _ALIASES[stripped]
    for c in CANONICAL:
        if _plain(c) in (p, stripped):
            return c
    if fuzzy:
        choices = {**_ALIASES, **{_plain(c): c for c in CANONICAL}}
        # token_sort (no token_set): evita que "atletico lugones" case con "atletico"
        best = process.extractOne(stripped or p, list(choices), scorer=fuzz.token_sort_ratio)
        if best and best[1] >= 90:
            return choices[best[0]]
    if strict:
        raise KeyError(f"Equipo desconocido: {name!r}. Añádelo a teams._ALIASES/STADIUMS.")
    return name


def is_known(name: str) -> bool:
    return canonical(name, strict=False, fuzzy=False) in STADIUMS


# Derbis: misma ciudad (<35 km) o rivalidades regionales clásicas
_REGIONAL_DERBIES = [
    {"Ath Bilbao", "Sociedad", "Alaves", "Eibar", "Osasuna"},
    {"Celta", "La Coruna"},
    {"Oviedo", "Sp Gijon"},
    {"Las Palmas", "Tenerife"},
    {"Sevilla", "Betis"},
    {"Valencia", "Levante", "Villarreal", "Elche", "Hercules", "Castellon"},
    {"Malaga", "Granada", "Almeria", "Cadiz", "Cordoba", "Xerez", "Recreativo"},
    {"Zaragoza", "Huesca"},
]


def is_derby(a: str, b: str) -> bool:
    d = haversine_km(a, b)
    if d == d and d < 35:
        return True
    return any(a in g and b in g for g in _REGIONAL_DERBIES)


def haversine_km(a: str, b: str) -> float:
    if a not in STADIUMS or b not in STADIUMS:
        return float("nan")
    lat1, lon1 = map(math.radians, STADIUMS[a][:2])
    lat2, lon2 = map(math.radians, STADIUMS[b][:2])
    d = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(d))


def is_island(team: str) -> bool:
    return STADIUMS.get(team, (0, 0, "", False))[3]


def coords(team: str) -> tuple[float, float] | None:
    s = STADIUMS.get(team)
    return (s[0], s[1]) if s else None


# Nombre para mostrar (el mismo que usa la web)
DISPLAY = {
    "Ath Madrid": "Atlético", "Ath Bilbao": "Athletic", "Sociedad": "Real Sociedad", "Vallecano": "Rayo Vallecano",
    "Espanol": "Espanyol", "La Coruna": "Deportivo", "Santander": "Racing", "Betis": "Real Betis", "Alaves": "Alavés",
    "Malaga": "Málaga", "Leganes": "Leganés", "Cadiz": "Cádiz", "Almeria": "Almería", "Sp Gijon": "Sporting",
    "Mirandes": "Mirandés", "Castellon": "Castellón", "Cordoba": "Córdoba",
}
