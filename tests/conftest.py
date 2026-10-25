from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from laliga.config import load_settings
from laliga.data import http
from laliga.data.results import COLUMNS

REPO = Path(__file__).resolve().parents[1]
FIX = Path(__file__).parent / "fixtures"

TEAMS = ["Barcelona", "Real Madrid", "Ath Madrid", "Sevilla", "Valencia", "Villarreal", "Betis", "Sociedad",
         "Ath Bilbao", "Celta", "Getafe", "Osasuna", "Espanol", "Alaves", "Vallecano", "Mallorca", "Girona",
         "Las Palmas", "Levante", "Elche", "Malaga", "La Coruna", "Santander", "Oviedo", "Leganes", "Valladolid"]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Ningún test toca la red: cualquier petición real falla."""
    def boom(*a, **k):
        raise http.requests.ConnectionError("red desactivada en tests")
    monkeypatch.setattr(http.session(), "get", boom)
    monkeypatch.setattr(http.session(), "post", boom)
    http.reset_breaker()
    yield


@pytest.fixture
def settings(tmp_path):
    shutil.copy(REPO / "config.yaml", tmp_path / "config.yaml")
    return load_settings(root=tmp_path, use_tuned=False)


def make_league(seasons=range(2012, 2019), seed=0, n_teams=20, with_stats=True) -> tuple[pd.DataFrame, dict]:
    """Liga sintética con fuerzas conocidas, ascensos/descensos y estadísticas de tiro."""
    rng = np.random.default_rng(seed)
    strength = {t: (rng.normal(0, 0.3), rng.normal(0, 0.25)) for t in TEAMS}
    current = TEAMS[:n_teams]
    pool = TEAMS[n_teams:]
    rows = []
    for s in seasons:
        teams = list(current)
        fixtures = [(h, a) for h in teams for a in teams if h != a]
        rng.shuffle(fixtures)
        start = pd.Timestamp(year=s, month=8, day=15)
        per_round = len(teams) // 2
        for i, (h, a) in enumerate(fixtures):
            d = start + pd.Timedelta(days=7 * (i // per_round) + (i % 3))
            ah, dh = strength[h]
            aa, da = strength[a]
            lam, mu = np.exp(0.25 + ah + da), np.exp(aa + dh)
            hg, ag = rng.poisson(lam), rng.poisson(mu)
            r = {"season": s, "date": d, "kickoff": pd.NaT, "home": h, "away": a, "hg": float(hg), "ag": float(ag),
                 "result": "H" if hg > ag else "D" if hg == ag else "A", "referee": f"Árbitro {rng.integers(0, 15)}",
                 "source": "synthetic"}
            if with_stats:
                r.update({"h_shots": hg * 3 + rng.poisson(9 * lam), "a_shots": ag * 3 + rng.poisson(9 * mu),
                          "h_sot": hg + rng.poisson(3 * lam), "a_sot": ag + rng.poisson(3 * mu),
                          "h_corners": rng.poisson(5), "a_corners": rng.poisson(4),
                          "h_yellow": rng.poisson(2), "a_yellow": rng.poisson(2.3)})
            rows.append(r)
        # 3 descensos aleatorios y 3 ascensos
        down = list(rng.choice(teams, 3, replace=False))
        up = pool[:3]
        current = [t for t in teams if t not in down] + up
        pool = pool[3:] + down
    df = pd.DataFrame(rows)
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[COLUMNS].sort_values(["date", "home"]).reset_index(drop=True)
    return df, strength


@pytest.fixture(scope="session")
def league():
    return make_league()
