"""Clima en el estadio a la hora del partido vía Open-Meteo (gratis, sin clave).

Previsión para partidos futuros y archivo histórico (ERA5) para entrenar.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import Settings
from ..teams import coords
from . import http

log = logging.getLogger(__name__)

FORECAST = "https://api.open-meteo.com/v1/forecast"
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
HOURLY = "temperature_2m,precipitation,wind_speed_10m"
WEATHER_COLS = ["temp_c", "precip_mm", "wind_kmh"]
DEFAULT_LOCAL_HOUR = 19


def parse_hourly(js: dict) -> pd.DataFrame:
    h = js.get("hourly") or {}
    if not h.get("time"):
        return pd.DataFrame(columns=["time", *WEATHER_COLS])
    df = pd.DataFrame({
        "time": pd.to_datetime(h["time"], utc=True),
        "temp_c": pd.to_numeric(pd.Series(h.get("temperature_2m")), errors="coerce"),
        "precip_mm": pd.to_numeric(pd.Series(h.get("precipitation")), errors="coerce"),
        "wind_kmh": pd.to_numeric(pd.Series(h.get("wind_speed_10m")), errors="coerce"),
    })
    return df


def at_kickoff(hourly: pd.DataFrame, kickoff: pd.Timestamp) -> dict:
    """Temperatura y viento a la hora de inicio; lluvia acumulada en las 2 h de partido."""
    if hourly.empty or pd.isna(kickoff):
        return {c: np.nan for c in WEATHER_COLS}
    k = pd.Timestamp(kickoff).floor("h")
    win = hourly[(hourly.time >= k) & (hourly.time < k + pd.Timedelta(hours=2))]
    if win.empty:
        return {c: np.nan for c in WEATHER_COLS}
    return {"temp_c": float(win.temp_c.iloc[0]), "wind_kmh": float(win.wind_kmh.max()),
            "precip_mm": float(win.precip_mm.sum())}


def _kickoff_or_default(row) -> pd.Timestamp:
    ko = row.get("kickoff")
    if ko is not None and not pd.isna(ko):
        return pd.Timestamp(ko)
    d = pd.Timestamp(row["date"]).normalize() + pd.Timedelta(hours=DEFAULT_LOCAL_HOUR)
    return d.tz_localize("Europe/Madrid", ambiguous="NaT", nonexistent="NaT").tz_convert("UTC")


def fetch_hourly(lat: float, lon: float, start: pd.Timestamp, end: pd.Timestamp, archive: bool) -> pd.DataFrame:
    params = {"latitude": round(lat, 3), "longitude": round(lon, 3), "hourly": HOURLY,
              "timezone": "UTC", "start_date": f"{start:%Y-%m-%d}", "end_date": f"{end:%Y-%m-%d}"}
    js = http.get(ARCHIVE if archive else FORECAST, params=params, timeout=30).json()
    return parse_hourly(js)


def weather_for_matches(settings: Settings, matches: pd.DataFrame, archive: bool) -> pd.DataFrame:
    """Devuelve season/home/away + columnas de clima. Usa y actualiza caché."""
    cache_p = settings.processed_dir / "weather.csv"
    cache = pd.read_csv(cache_p) if cache_p.exists() else pd.DataFrame(columns=["season", "home", "away", *WEATHER_COLS])
    key = ["season", "home", "away"]
    todo = matches.merge(cache[key], on=key, how="left", indicator=True)
    todo = todo[todo["_merge"] == "left_only"].drop(columns="_merge") if archive else matches
    new_rows = []
    if len(todo):
        todo = todo.assign(ko=todo.apply(_kickoff_or_default, axis=1))
        for (team, _season), grp in todo.groupby(["home", "season"]):
            c = coords(team)
            if not c:
                continue
            try:
                hourly = fetch_hourly(c[0], c[1], grp.ko.min() - pd.Timedelta(days=1),
                                      grp.ko.max() + pd.Timedelta(days=1), archive)
            except Exception as e:
                log.warning("Clima no disponible para %s: %s", team, e)
                continue
            for _, r in grp.iterrows():
                new_rows.append({"season": r.season, "home": r.home, "away": r.away, **at_kickoff(hourly, r.ko)})
    new = pd.DataFrame(new_rows, columns=["season", "home", "away", *WEATHER_COLS])
    if archive and len(new):
        cache = pd.concat([cache, new], ignore_index=True).drop_duplicates(key, keep="last")
        cache.to_csv(cache_p, index=False)
        return cache
    if not archive:
        return new
    return cache
