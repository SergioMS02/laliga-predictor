"""Resultados históricos y de la temporada actual con un esquema único.

Clave natural de un partido de liga: (season, home, away) — cada cruce ocurre una vez por temporada.
"""
from __future__ import annotations

import io
import logging
from datetime import date, datetime

import numpy as np
import pandas as pd

from ..config import Settings
from ..teams import canonical
from . import http

log = logging.getLogger(__name__)

STAT_MAP = {
    "HS": "h_shots", "AS": "a_shots", "HST": "h_sot", "AST": "a_sot",
    "HC": "h_corners", "AC": "a_corners", "HY": "h_yellow", "AY": "a_yellow",
    "HR": "h_red", "AR": "a_red", "HF": "h_fouls", "AF": "a_fouls",
}
# Prioridad de cuotas previas (publicadas antes de la jornada) y de cierre
PRE_ODDS = [("PSH", "PSD", "PSA"), ("AvgH", "AvgD", "AvgA"), ("BbAvH", "BbAvD", "BbAvA"),
            ("B365H", "B365D", "B365A"), ("WHH", "WHD", "WHA")]
CLOSE_ODDS = [("PSCH", "PSCD", "PSCA"), ("AvgCH", "AvgCD", "AvgCA"), ("B365CH", "B365CD", "B365CA")]

B365_COLS = ["b365_h", "b365_d", "b365_a", "b365c_h", "b365c_d", "b365c_a"]
# Más/menos de 2,5 goles: Bet365 (apertura y cierre) y mercado de referencia (Pinnacle o media)
OU_COLS = ["b365_o25", "b365_u25", "b365c_o25", "b365c_u25", "ou_o25", "ou_u25", "ouc_o25", "ouc_u25"]
PRE_OU = [("P>2.5", "P<2.5"), ("Avg>2.5", "Avg<2.5"), ("BbAv>2.5", "BbAv<2.5")]
CLOSE_OU = [("PC>2.5", "PC<2.5"), ("AvgC>2.5", "AvgC<2.5")]
COLUMNS = ["season", "date", "kickoff", "home", "away", "hg", "ag", "result", "referee",
           *STAT_MAP.values(), "odds_h", "odds_d", "odds_a", "odds_close_h", "odds_close_d",
           "odds_close_a", *B365_COLS, *OU_COLS, "source"]
DIVISIONS = {"SP1": "LaLiga", "SP2": "Segunda"}


def season_of(d: date | datetime | pd.Timestamp) -> int:
    d = pd.Timestamp(d)
    return d.year if d.month >= 7 else d.year - 1


def season_code(start_year: int) -> str:
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def current_season(today: date | None = None) -> int:
    return season_of(today or date.today())


def _parse_dates(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    iso = s.str.match(r"^\d{4}-\d{2}-\d{2}")
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    if iso.any():
        out[iso] = pd.to_datetime(s[iso].str[:10], format="%Y-%m-%d", errors="coerce")
    rest = ~iso
    if rest.any():
        long_y = s[rest].str.match(r"^\d{1,2}/\d{1,2}/\d{4}$")
        idx_l = s[rest][long_y].index
        idx_s = s[rest][~long_y].index
        out[idx_l] = pd.to_datetime(s[idx_l], format="%d/%m/%Y", errors="coerce")
        out[idx_s] = pd.to_datetime(s[idx_s], format="%d/%m/%y", errors="coerce")
    return out


def _pick_odds(df: pd.DataFrame, options) -> np.ndarray:
    out = np.full((len(df), len(options[0])), np.nan)
    for cols in options:
        if all(c in df.columns for c in cols):
            vals = df[list(cols)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            fill = np.isnan(out).any(axis=1) & ~np.isnan(vals).any(axis=1)
            out[fill] = vals[fill]
    return out


def _team_name(name: str, div: str) -> str:
    if div == "SP1":
        return canonical(name)
    # Segunda: football-data usa los mismos nombres que en Primera para los equipos que bajan;
    # los demás se dejan tal cual (sin fuzzy, para no confundir filiales con primeros equipos)
    return canonical(str(name).strip(), strict=False, fuzzy=False)


def standardize_football_data(raw: pd.DataFrame, source: str, div: str = "SP1") -> pd.DataFrame:
    """Convierte un CSV de football-data.co.uk (o su espejo) al esquema común."""
    raw = raw.dropna(how="all")
    raw = raw[raw["HomeTeam"].notna() & raw["AwayTeam"].notna()].copy()
    df = pd.DataFrame(index=raw.index)
    df["date"] = _parse_dates(raw["Date"])
    if "Time" in raw.columns:
        t = raw["Time"].astype(str).str.extract(r"(\d{1,2}):(\d{2})")
        ko = df["date"] + pd.to_timedelta(t[0].astype(float), unit="h") + pd.to_timedelta(t[1].astype(float), unit="m")
        # football-data publica la hora en horario del Reino Unido
        df["kickoff"] = ko.dt.tz_localize("Europe/London", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    else:
        df["kickoff"] = pd.NaT
    df["home"] = raw["HomeTeam"].map(lambda n: _team_name(n, div))
    df["away"] = raw["AwayTeam"].map(lambda n: _team_name(n, div))
    df["hg"] = pd.to_numeric(raw.get("FTHG"), errors="coerce")
    df["ag"] = pd.to_numeric(raw.get("FTAG"), errors="coerce")
    df["result"] = np.select([df.hg > df.ag, df.hg == df.ag, df.hg < df.ag], ["H", "D", "A"], default=None)
    df.loc[df.hg.isna(), "result"] = None
    ref = raw["Referee"] if "Referee" in raw.columns else pd.Series(np.nan, index=raw.index)
    df["referee"] = ref.where(ref.astype(str).str.strip().ne(""), np.nan)
    for k, v in STAT_MAP.items():
        df[v] = pd.to_numeric(raw[k], errors="coerce") if k in raw.columns else np.nan
    pre = _pick_odds(raw, PRE_ODDS)
    close = _pick_odds(raw, CLOSE_ODDS)
    df["odds_h"], df["odds_d"], df["odds_a"] = pre.T
    df["odds_close_h"], df["odds_close_d"], df["odds_close_a"] = close.T
    # Bet365 por separado (para la cartera de apuestas simulada)
    b = _pick_odds(raw, [("B365H", "B365D", "B365A")])
    bc = _pick_odds(raw, [("B365CH", "B365CD", "B365CA")])
    df["b365_h"], df["b365_d"], df["b365_a"] = b.T
    df["b365c_h"], df["b365c_d"], df["b365c_a"] = bc.T
    for cols, opts in ((("b365_o25", "b365_u25"), [("B365>2.5", "B365<2.5")]),
                       (("b365c_o25", "b365c_u25"), [("B365C>2.5", "B365C<2.5")]),
                       (("ou_o25", "ou_u25"), PRE_OU), (("ouc_o25", "ouc_u25"), CLOSE_OU)):
        v = _pick_odds(raw, opts)
        df[cols[0]], df[cols[1]] = v.T
    df["season"] = df["date"].map(season_of)
    df["source"] = source
    return df[COLUMNS].reset_index(drop=True)


def _read_cache(cache) -> pd.DataFrame:
    return pd.read_csv(cache, parse_dates=["date"]).assign(
        kickoff=lambda d: pd.to_datetime(d["kickoff"], utc=True, errors="coerce"))


def fetch_season(settings: Settings, start_year: int, refresh: bool = False,
                 online: bool = True, div: str = "SP1") -> pd.DataFrame | None:
    """Descarga una temporada; usa caché local salvo para la temporada en curso o si refresh."""
    code = season_code(start_year)
    cache = settings.raw_dir / f"{div}_{code}.csv"
    is_current = start_year >= current_season()
    if not online:
        return _read_cache(cache) if cache.exists() else None
    if cache.exists() and not (refresh or is_current):
        cached = _read_cache(cache)
        # una copia del espejo no trae cuotas, y las cachés antiguas no traen Bet365 ni más/menos:
        # en ambos casos se reintenta la fuente principal
        if not (cached["source"] == "mirror").all() and "b365_o25" in cached.columns:
            return cached
    sources = (("football_data_url", "football-data"), ("mirror_url", "mirror")) if div == "SP1" else \
        (("football_data_url", "football-data"),)
    for url_key, source in sources:
        url = settings.get("data", url_key).format(code=code).replace("/SP1.csv", f"/{div}.csv")
        try:
            r = http.get(url, timeout=25, retries=1)
            text = r.content.decode("utf-8-sig", errors="replace")
            raw = pd.read_csv(io.StringIO(text))
            if raw.empty or "HomeTeam" not in raw.columns:
                continue
            df = standardize_football_data(raw, source, div)
            df.to_csv(cache, index=False)
            log.info("Temporada %s: %d partidos desde %s", code, len(df), source)
            return df
        except Exception as e:  # red caída, 404, etc.
            log.warning("No se pudo descargar %s (%s): %s", code, source, e)
    if cache.exists():
        return _read_cache(cache)
    return None


def load_history(settings: Settings, refresh_current: bool = True, online: bool = True,
                 div: str = "SP1") -> pd.DataFrame:
    first = int(settings.get("data", "first_season", 2005))
    frames = []
    for y in range(first, current_season() + 1):
        df = fetch_season(settings, y, refresh=(refresh_current and y == current_season()), online=online, div=div)
        if df is not None and len(df):
            frames.append(df)
    if not frames:
        raise RuntimeError("No hay datos históricos disponibles (sin red y sin caché).")
    return pd.concat(frames, ignore_index=True)


def merge_sources(*frames: pd.DataFrame) -> pd.DataFrame:
    """Une fuentes por (season, home, away). La primera fuente con resultado gana;
    los campos vacíos se completan con las siguientes (p. ej. hora o árbitro de ESPN)."""
    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    allf = []
    for prio, f in enumerate(frames):
        f = f.copy()
        for c in COLUMNS:
            if c not in f.columns:
                f[c] = np.nan
        f["_prio"] = prio
        f["_played"] = f["hg"].notna()
        allf.append(f[COLUMNS + ["_prio", "_played"]])
    cat = pd.concat(allf, ignore_index=True)
    cat["kickoff"] = pd.to_datetime(cat["kickoff"], utc=True, errors="coerce")
    cat = cat.sort_values(["season", "home", "away", "_played", "_prio"],
                          ascending=[True, True, True, False, True])
    key = ["season", "home", "away"]
    first = cat.groupby(key, sort=False).first()  # first() ignora NaN por columna -> completa huecos
    # Para goles y resultado no mezclamos fuentes: tomamos la fila prioritaria completa
    lead = cat.drop_duplicates(key, keep="first").set_index(key)
    for c in ("hg", "ag", "result", "date", "source"):
        first[c] = lead[c]
    out = first.reset_index()[COLUMNS]
    return out.sort_values(["date", "home"]).reset_index(drop=True)


FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"


_FIXTURES_RAW: dict[str, str] = {}


def fetch_fixtures(settings: Settings, online: bool = True, div: str = "SP1") -> pd.DataFrame:
    """Próximos partidos con fecha, hora y cuotas (football-data.co.uk/fixtures.csv)."""
    cache = settings.raw_dir / f"fixtures_{div}.csv"
    if online:
        try:
            if "text" not in _FIXTURES_RAW:  # una sola descarga por ejecución para las dos divisiones
                _FIXTURES_RAW["text"] = http.get(FIXTURES_URL, timeout=20, retries=1).content.decode("utf-8-sig")
            raw = pd.read_csv(io.StringIO(_FIXTURES_RAW["text"]))
            raw = raw[raw.get("Div") == div] if "Div" in raw.columns else raw.iloc[0:0]
            df = standardize_football_data(raw, "football-data-fixtures", div) if len(raw) else pd.DataFrame(columns=COLUMNS)
            df.to_csv(cache, index=False)
            return df
        except Exception as e:
            log.warning("fixtures.csv no disponible: %s", e)
    if cache.exists():
        return _read_cache(cache)
    return pd.DataFrame(columns=COLUMNS)
