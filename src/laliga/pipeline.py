"""Orquestación: datos -> variables -> Dixon-Coles walk-forward -> meta-modelo -> predicciones."""
from __future__ import annotations

import json
import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .config import Settings
from .data import coaches, espn, openfootball, other_comps, results, understat, weather
from .features.context import build_context_features
from .features.sequential import build_sequential_features
from .models.dixon_coles import DixonColes
from .models.market import devig_power
from .models.stacker import GROUPS, Stacker
from . import metrics

log = logging.getLogger(__name__)

# Modelos de "fuerza" sobre otros conteos además de goles: tiros a puerta y xG
COUNT_MODELS = {"sot": ("h_sot", "a_sot"), "xg": ("h_xg", "a_xg")}
DC_COLS = ["dc_ph", "dc_pd", "dc_pa", "dc_lam", "dc_mu"] + [f"{k}_{s}" for k in COUNT_MODELS for s in ("lam", "mu")]
DC_EXTRA = ["dc_po25", "dc_btts"]  # más de 2,5 goles y ambos marcan (para el mercado de goles)
MIN_SOT_MATCHES = 200  # partidos mínimos con el dato en la ventana para ajustar un modelo de conteo


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
def load_matches(settings: Settings, online: bool = True, days_ahead: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Histórico + temporada en curso combinando fuentes. Devuelve (partidos, scoreboard_espn).

    Con online=True también refresca las cachés de fuentes extra (xG, otras competiciones,
    entrenadores) que después se leen sin red en build_table/predict."""
    hist = results.load_history(settings, refresh_current=online, online=online)
    cur = results.current_season()
    sb = pd.DataFrame()
    espn_res = None
    if online and settings.get("espn", "enabled", True):
        try:
            ahead = days_ahead or int(settings.get("predict", "days_ahead", 8))
            sb = espn.scoreboard(settings, date(cur, 7, 15), date.today() + timedelta(days=ahead))
            espn_res = espn.to_results(sb)
        except Exception as e:
            log.warning("ESPN no disponible: %s", e)
            _ESPN_ERR["e"] = str(e)[:200]
    of = openfootball.fetch_openfootball(settings, cur) if online else _cached_openfootball(settings, cur)
    fx = results.fetch_fixtures(settings, online)
    hist_cur = hist[hist.season == cur]
    hist_old = hist[hist.season < cur]
    # prioridad: resultados de football-data > ESPN > calendario+hora de fixtures.csv > openfootball
    cur_merged = results.merge_sources(hist_cur, espn_res, fx, of)
    if online:
        _status("football-data (histórico)", len(hist_old) > 0, int((hist.source == "football-data").sum()))
        _status("football-data (temporada actual)", len(hist_cur) > 0, len(hist_cur))
        _status("football-data (próximos partidos y cuotas)", len(fx) > 0, len(fx))
        _status("ESPN (calendario, horarios, alineaciones)", espn_res is not None and len(espn_res) > 0,
                0 if espn_res is None else len(espn_res), None if espn_res is not None else _ESPN_ERR.get("e"))
        _status("openfootball (calendario de respaldo)", len(of) > 0, len(of))
    matches = pd.concat([hist_old, cur_merged], ignore_index=True)
    matches = apply_referees(settings, matches)
    xg = load_extras(settings, online)["xg"]
    matches = understat.attach_xg(matches, xg)
    matches = matches.sort_values(["date", "home"], kind="stable").reset_index(drop=True)
    return matches, sb


_EXTRAS: dict = {}
_ESPN_ERR: dict = {}
SOURCE_STATUS: dict = {}


def _status(name: str, ok: bool, rows: int | None = None, error: str | None = None, note: str | None = None) -> None:
    """Registra qué fuentes respondieron en esta ejecución (se guarda en predictions/sources.json).
    `note`: fuente de reserva que no hizo falta consultar."""
    SOURCE_STATUS[name] = {"ok": bool(ok), "rows": None if rows is None else int(rows), "error": error, "note": note}


def save_status(settings: Settings) -> None:
    if SOURCE_STATUS:
        out = {"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="minutes"), "sources": SOURCE_STATUS}
        (settings.predictions_dir / "sources.json").write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                                              encoding="utf-8")


def load_extras(settings: Settings, online: bool = False) -> dict:
    """xG (Understat), otras competiciones y entrenadores. Se memoriza por ejecución."""
    key = (str(settings.root), online)
    if key in _EXTRAS:
        return _EXTRAS[key]
    ex = {}
    first = int(settings.get("data", "first_season", 2005))
    ex_cfg = settings.raw.get("extras", {})
    try:
        ex["xg"], ex["players"] = understat.load_understat(settings, online) if ex_cfg.get("understat", True) \
            else (pd.DataFrame(columns=understat.MATCH_COLS), pd.DataFrame(columns=understat.PLAYER_COLS))
    except Exception as e:
        log.warning("Understat: %s", e)
        ex["xg"], ex["players"] = pd.DataFrame(columns=understat.MATCH_COLS), pd.DataFrame(columns=understat.PLAYER_COLS)
    try:
        ex["other"] = other_comps.load_other_matches(settings, first, online) if ex_cfg.get("other_comps", True) \
            else pd.DataFrame(columns=other_comps.COLS)
    except Exception as e:
        log.warning("Otras competiciones: %s", e)
        ex["other"] = pd.DataFrame(columns=other_comps.COLS)
    try:
        ex["coaches"] = coaches.load_coaches(settings, online) if ex_cfg.get("coaches", True) \
            else pd.DataFrame(columns=coaches.COLS)
    except Exception as e:
        log.warning("Entrenadores: %s", e)
        ex["coaches"] = pd.DataFrame(columns=coaches.COLS)
    if online:
        _status("Understat (xG)", len(ex["xg"]) > 0, len(ex["xg"]))
        _status("Europa y Copa (calendario)", len(ex["other"]) > 0, len(ex["other"]))
        _status("Wikidata (entrenadores)", len(ex["coaches"]) > 0, len(ex["coaches"]))
    _EXTRAS[key] = ex
    if online:
        _EXTRAS[(str(settings.root), False)] = ex
    return ex


def _cached_openfootball(settings: Settings, season: int) -> pd.DataFrame:
    p = settings.raw_dir / f"openfootball_{season}.txt"
    if p.exists():
        return openfootball.parse_football_txt(p.read_text(encoding="utf-8"), season)
    return pd.DataFrame(columns=results.COLUMNS)


def apply_referees(settings: Settings, matches: pd.DataFrame) -> pd.DataFrame:
    refs = espn.load_referees(settings)
    if refs.empty:
        return matches
    m = matches.merge(refs.rename(columns={"referee": "_ref"}), on=["season", "home", "away"], how="left")
    m["referee"] = m["_ref"].combine_first(m["referee"])
    return m.drop(columns="_ref")


def add_market(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    p = devig_power(df[["odds_h", "odds_d", "odds_a"]].to_numpy(dtype=float))
    df["mkt_ph"], df["mkt_pd"], df["mkt_pa"] = p.T
    pc = devig_power(df[["odds_close_h", "odds_close_d", "odds_close_a"]].to_numpy(dtype=float))
    df["mktc_ph"], df["mktc_pd"], df["mktc_pa"] = pc.T
    return df


def attach_weather(settings: Settings, df: pd.DataFrame, archive: bool) -> pd.DataFrame:
    for c in weather.WEATHER_COLS:
        if c in df.columns:
            df = df.drop(columns=c)
    if not settings.get("weather", "enabled", True):
        return df.assign(**{c: np.nan for c in weather.WEATHER_COLS})
    try:
        w = weather.weather_for_matches(settings, df, archive=archive)
    except Exception as e:
        log.warning("Clima no disponible: %s", e)
        w = pd.DataFrame(columns=["season", "home", "away", *weather.WEATHER_COLS])
    if w.empty:
        return df.assign(**{c: np.nan for c in weather.WEATHER_COLS})
    return df.merge(w, on=["season", "home", "away"], how="left")


# ---------------------------------------------------------------------------
# Dixon-Coles fuera de muestra
# ---------------------------------------------------------------------------
def _new_teams(matches: pd.DataFrame, season: int) -> set[str]:
    prev = matches[matches.season == season - 1]
    cur = matches[matches.season == season]
    if prev.empty:
        return set()
    return (set(cur.home) | set(cur.away)) - (set(prev.home) | set(prev.away))


def walk_forward_dc(settings: Settings, matches: pd.DataFrame, start_season: int,
                    until: pd.Timestamp | None = None) -> pd.DataFrame:
    """Predicción Dixon-Coles de cada partido usando solo partidos anteriores al bloque."""
    block = int(settings.get("backtest", "block_days", 7))
    dc = DixonColes.from_settings(settings)
    counts = {k: (DixonColes.from_settings(settings), cols, matches[list(cols)].notna().all(axis=1) if set(cols) <= set(matches.columns) else pd.Series(False, index=matches.index))
              for k, cols in COUNT_MODELS.items()}
    target = matches[matches.season >= start_season]
    if until is not None:
        target = target[target.date < until]
    out = pd.DataFrame(index=target.index, columns=DC_COLS + DC_EXTRA, dtype=float)
    if target.empty:
        return out
    start = target.date.min().normalize()
    end = target.date.max()
    b = start
    while b <= end:
        blk = target[(target.date >= b) & (target.date < b + pd.Timedelta(days=block))]
        if len(blk):
            seasons = blk.season.unique()
            new = set().union(*[_new_teams(matches, s) for s in seasons])
            teams = sorted(set(blk.home) | set(blk.away))
            dc.fit(matches, b, current_teams=teams, new_teams=new)
            fitted = {}
            for k, (model, cols, has) in counts.items():
                win = has & (matches.date < b) & (matches.date >= b - pd.Timedelta(days=model.window_days))
                if win.sum() >= MIN_SOT_MATCHES:
                    model.fit(matches, b, current_teams=teams, new_teams=new, cols=cols)
                    fitted[k] = model
            for idx, r in blk.iterrows():
                lam, mu = dc.expected_goals(r.home, r.away)
                p = dc.predict(r.home, r.away, lam, mu)
                row = [p["p_home"], p["p_draw"], p["p_away"], lam, mu]
                for k in COUNT_MODELS:
                    row += list(fitted[k].expected_goals(r.home, r.away)) if k in fitted else [np.nan, np.nan]
                row += [p.get("p_over25", np.nan), p.get("p_btts", np.nan)]
                out.loc[idx] = row
            b = b + pd.Timedelta(days=block)
        else:
            nxt = target[target.date >= b].date
            b = nxt.min().normalize() if len(nxt) else end + pd.Timedelta(days=1)
    return out


def context_features(settings: Settings, matches: pd.DataFrame) -> pd.DataFrame:
    ex = load_extras(settings, online=False)
    return build_context_features(matches, ex.get("other"), ex.get("coaches"))


def build_table(settings: Settings, matches: pd.DataFrame, start_season: int, with_weather: bool,
                until: pd.Timestamp | None = None) -> pd.DataFrame:
    seq = build_sequential_features(matches, settings["elo"])
    ctx = context_features(settings, matches)
    dcp = walk_forward_dc(settings, matches, start_season, until)
    tab = matches.join(seq).join(ctx).loc[dcp.index].join(dcp)
    tab = add_market(tab)
    if with_weather:
        tab = attach_weather(settings, tab, archive=True)
    return tab


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
def backtest(settings: Settings, matches: pd.DataFrame, with_weather: bool = False) -> dict:
    first_oos = int(settings.get("backtest", "first_oos_season", 2012))
    first_test = int(settings.get("backtest", "first_test_season", 2017))
    today = pd.Timestamp(date.today())
    tab = build_table(settings, matches, first_oos, with_weather, until=today)
    tab = tab[tab.result.notna()].copy()
    tab.to_csv(settings.processed_dir / "oos_table.csv", index=False)
    y_all = tab.result.map(metrics.RESULT_IDX).to_numpy()
    tab["y"] = y_all
    preds = {k: [] for k in ("uniforme", "frecuencias", "elo", "dixon_coles", "stacker", "stacker_mercado",
                             "mercado", "mercado_cierre")}
    rows_idx = []
    for season in sorted(tab.season.unique()):
        if season < first_test:
            continue
        train = tab[tab.season < season]
        test = tab[tab.season == season]
        if len(train) < int(settings.get("stacker", "min_train_rows", 1000)) or test.empty:
            continue
        rows_idx.append(test.index)
        n = len(test)
        preds["uniforme"].append(np.full((n, 3), 1 / 3))
        freq = np.bincount(train.y, minlength=3) / len(train)
        preds["frecuencias"].append(np.tile(freq, (n, 1)))
        preds["elo"].append(_elo_only(train, test))
        preds["dixon_coles"].append(test[["dc_ph", "dc_pd", "dc_pa"]].to_numpy(dtype=float))
        st = make_stacker(settings, market=False).fit(train)
        preds["stacker"].append(st.predict_proba(test))
        has_mkt = test[["mkt_ph", "mkt_pd", "mkt_pa"]].notna().all(axis=1).to_numpy()
        tr_mkt = train[["mkt_ph", "mkt_pd", "mkt_pa"]].notna().all(axis=1).sum()
        pm = np.full((n, 3), np.nan)
        if tr_mkt >= 500 and has_mkt.any():
            stm = make_stacker(settings, market=True).fit(train)
            pm[has_mkt] = stm.predict_proba(test[has_mkt])
        preds["stacker_mercado"].append(pm)
        preds["mercado"].append(test[["mkt_ph", "mkt_pd", "mkt_pa"]].to_numpy(dtype=float))
        preds["mercado_cierre"].append(test[["mktc_ph", "mktc_pd", "mktc_pa"]].to_numpy(dtype=float))
    if not rows_idx:
        raise RuntimeError("Datos insuficientes para el backtest")
    idx = np.concatenate(rows_idx)
    y = tab.loc[idx, "y"].to_numpy()
    seasons = tab.loc[idx, "season"].to_numpy()
    report = {"overall": {}, "by_season": {}, "market_subset": {}}
    P = {k: np.vstack(v) for k, v in preds.items()}
    for k, p in P.items():
        ok = ~np.isnan(p).any(axis=1)
        if ok.sum() == len(y):
            report["overall"][k] = metrics.evaluate(p, y)
            report["by_season"][k] = {int(s): metrics.evaluate(p[seasons == s], y[seasons == s])["log_loss"]
                                      for s in np.unique(seasons)}
    mk = ~np.isnan(P["mercado"]).any(axis=1) & ~np.isnan(P["stacker_mercado"]).any(axis=1)
    if mk.sum() > 100:
        for k, p in P.items():
            if not np.isnan(p[mk]).any():
                report["market_subset"][k] = metrics.evaluate(p[mk], y[mk])
    best = "stacker" if report["overall"]["stacker"]["log_loss"] <= report["overall"]["dixon_coles"]["log_loss"] else "dixon_coles"
    report["calibration"] = metrics.calibration_table(P[best], y).to_dict(orient="records")
    report["calibration_model"] = best
    report["test_seasons"] = sorted(int(s) for s in np.unique(seasons))
    (settings.processed_dir / "backtest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    pd.DataFrame({"season": seasons, "y": y, **{f"{k}_{c}": P[k][:, i] for k in P for i, c in enumerate("HDA")}},
                 index=idx).to_csv(settings.processed_dir / "backtest_predictions.csv")
    try:
        from . import betting
        betting.backtest_strategies(settings, matches)
    except Exception as e:  # la simulación de apuestas nunca debe romper el backtest
        log.warning("Simulación de apuestas no disponible: %s", e)
    return report


def make_stacker(settings: Settings, market: bool) -> Stacker:
    sec = "stacker_market" if market else "stacker"
    groups = settings.get(sec, "groups", None)
    if groups is None:
        groups = settings.get("stacker", "groups", list(GROUPS))
    return Stacker(use_market=market, C=float(settings.get(sec, "C", 0.005)), groups=list(groups))


def _elo_only(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(max_iter=1000).fit(train[["elo_diff"]].to_numpy() / 100, train.y)
    p = np.zeros((len(test), 3))
    p[:, m.classes_] = m.predict_proba(test[["elo_diff"]].to_numpy() / 100)
    return p


def train_final(settings: Settings, table: pd.DataFrame | None = None) -> dict:
    if table is None:
        table = pd.read_csv(settings.processed_dir / "oos_table.csv", parse_dates=["date"])
    table = table[table.result.notna()]
    info = {}
    st = make_stacker(settings, market=False).fit(table)
    st.save(settings.state_dir / "stacker.pkl")
    info["stacker"] = len(table)
    if table[["mkt_ph", "mkt_pd", "mkt_pa"]].notna().all(axis=1).sum() >= 500:
        stm = make_stacker(settings, market=True).fit(table)
        stm.save(settings.state_dir / "stacker_market.pkl")
        info["stacker_market"] = True
    else:
        p = settings.state_dir / "stacker_market.pkl"
        if p.exists():
            p.unlink()
        info["stacker_market"] = False
    st.coefficients().to_csv(settings.processed_dir / "stacker_coefficients.csv")
    return info
