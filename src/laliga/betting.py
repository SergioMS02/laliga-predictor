"""Cartera de apuestas SIMULADA con las cuotas de Bet365 (sin dinero real).

Estrategias:
  favorito  10 € al resultado más probable según el modelo (rellenar la quiniela)
  valor     10 € solo si probabilidad del modelo × cuota ≥ 1 + margen mínimo
  goles     10 € a más/menos de 2,5 goles cuando esa cuota tiene valor
  kelly     las mismas apuestas que "valor", pero el importe depende de la ventaja
            (fracción de Kelly) sobre un fondo que empieza en 100 €

Cada apuesta queda "colocada" con la última cuota de Bet365 vista antes del partido y se
liquida cuando hay resultado. También se guarda la primera cuota a la que el modelo la vio
(odds_first) y la cuota de cierre (odds_close) para medir si el modelo "gana al cierre"
(CLV): si apostar en cuanto lo ve da mejor cuota que la final, el modelo detecta algo antes
que el mercado, y esa señal es mucho menos ruidosa que el beneficio.

Todo se guarda en predictions/bets.csv (historial completo), el resumen en
predictions/bets_summary.json y la simulación sobre temporadas pasadas en
predictions/betting_backtest.json.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import Settings

log = logging.getLogger(__name__)

OUT = ("1", "X", "2")
OU_OUT = ("+2.5", "-2.5")
STRATEGIES = {
    "favorito": {"label": "Al resultado más probable de cada partido", "market": "1x2"},
    "valor": {"label": "Solo cuando la cuota paga más de lo que dice el modelo", "market": "1x2"},
    "goles": {"label": "Más o menos de 2,5 goles cuando esa cuota tiene valor", "market": "ou25"},
    "kelly": {"label": "Las apuestas de valor, con importe según la ventaja (Kelly) sobre un fondo de 100 €",
              "market": "1x2"},
}
LEDGER_COLS = ["strategy", "market", "season", "date", "kickoff_utc", "home", "away", "pick", "prob", "odds",
               "odds_first", "odds_close", "close_src", "clv", "ev", "stake", "placed_at", "status", "score",
               "profit", "settled_at"]
OBJ_COLS = ("strategy", "market", "status", "score", "settled_at", "pick", "kickoff_utc", "placed_at", "close_src", "date")
FLOAT_COLS = ("prob", "odds", "odds_first", "odds_close", "clv", "ev", "stake", "profit")
OU_BLEND_DEFAULT = {"a_model": 0.2, "a_market": 0.8, "b": 0.0, "n": 0}


def cfg(settings: Settings) -> dict:
    c = {"stake": 10.0, "min_edge": 0.05, "max_odds": 8.0, "min_prob": 0.0, "ou_min_edge": 0.02,
         "kelly_bankroll": 100.0, "kelly_fraction": 0.25, "kelly_max_frac": 0.05, "kelly_min_stake": 0.5}
    c.update(getattr(settings, "raw", {}).get("betting", {}) or {})
    return c


def _clean(odds) -> np.ndarray:
    return np.asarray([np.nan if v is None else v for v in odds], float)


def decide(p, odds, strategy: str, c: dict):
    """Devuelve (índice del resultado, cuota, valor esperado) o None si no se apuesta."""
    p = np.asarray(p, float)
    o = _clean(odds)
    if len(o) != len(p) or np.isnan(o).any() or (o <= 1).any() or np.isnan(p).any():
        return None
    ev = p * o - 1
    if strategy == "favorito":
        i = int(np.argmax(p))
        return i, float(o[i]), float(ev[i])
    if strategy in ("valor", "kelly", "goles"):
        edge = c["ou_min_edge"] if strategy == "goles" else c["min_edge"]
        ok = (ev >= edge) & (o <= c["max_odds"]) & (p >= c["min_prob"])
        if not ok.any():
            return None
        i = int(np.argmax(np.where(ok, ev, -np.inf)))
        return i, float(o[i]), float(ev[i])
    raise ValueError(strategy)


def kelly_stake(p: float, o: float, bankroll: float, c: dict) -> float:
    """Fracción de Kelly (por defecto 1/4) con tope por apuesta; redondeado a 0,5 €."""
    if bankroll <= 0 or o <= 1:
        return 0.0
    f = c["kelly_fraction"] * (p * o - 1) / (o - 1)
    f = min(max(f, 0.0), c["kelly_max_frac"])
    st = round(bankroll * f * 2) / 2
    return st if st >= c["kelly_min_stake"] else 0.0


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def blend_ou(p_model: float, p_market: float | None, w: dict | None = None) -> float:
    """Probabilidad de más de 2,5 goles: modelo (Dixon-Coles) combinado con el mercado.

    El mercado acierta más que el modelo en goles, así que pesa más; los pesos los
    estima el backtest (data/state/ou_blend.json). Sin cuotas se usa solo el modelo."""
    if p_market is None or not np.isfinite(p_market):
        return float(p_model)
    w = w or OU_BLEND_DEFAULT
    z = w["a_model"] * logit(p_model) + w["a_market"] * logit(p_market) + w["b"]
    return float(1 / (1 + np.exp(-z)))


def load_ou_blend(settings: Settings) -> dict:
    p = settings.state_dir / "ou_blend.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else dict(OU_BLEND_DEFAULT)
    except Exception:
        return dict(OU_BLEND_DEFAULT)


def devig2(o1, o2):
    try:
        o1, o2 = float(o1), float(o2)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(o1) or not np.isfinite(o2) or o1 <= 1 or o2 <= 1:
        return None
    a, b = 1 / o1, 1 / o2
    return a / (a + b)


# ---------------------------------------------------------------------------
# Libro de apuestas
# ---------------------------------------------------------------------------
def _path(settings: Settings):
    return settings.predictions_dir / "bets.csv"


def _typed(led: pd.DataFrame) -> pd.DataFrame:
    led = led.copy()
    for c in LEDGER_COLS:
        if c not in led:
            led[c] = np.nan
    led["market"] = led["market"].fillna("1x2")
    for c in OBJ_COLS:
        led[c] = led[c].astype(object)
    for c in FLOAT_COLS:
        led[c] = pd.to_numeric(led[c], errors="coerce").astype(float)
    led["odds_first"] = led["odds_first"].fillna(led["odds"])
    return led[LEDGER_COLS]


def load_ledger(settings: Settings) -> pd.DataFrame:
    p = _path(settings)
    if not p.exists():
        return _typed(pd.DataFrame(columns=LEDGER_COLS))
    return _typed(pd.read_csv(p, dtype={"pick": str, "score": str}))


def _market_inputs(p: dict, market: str):
    if market == "1x2":
        return [p["p_home"], p["p_draw"], p["p_away"]], p.get("b365")
    ou = p.get("ou") or {}
    po = ou.get("p_over")
    if po is None and (p.get("dc") or {}).get("p_over25") is not None:
        po = p["dc"]["p_over25"]
    if po is None:
        return None, None
    return [po, 1 - po], p.get("b365_ou")


def bankroll(led: pd.DataFrame, c: dict) -> float:
    k = led[(led.strategy == "kelly") & led.status.isin(["ganada", "perdida"])]
    return float(c["kelly_bankroll"] + k.profit.sum())


def _closing(m: pd.Series, market: str, pick: str) -> float:
    if market == "1x2":
        col = {"1": "b365c_h", "X": "b365c_d", "2": "b365c_a"}.get(pick)
    else:
        col = {"+2.5": "b365c_o25", "-2.5": "b365c_u25"}.get(pick)
    if col is None or col not in m.index:
        return np.nan
    try:
        v = float(m[col])
    except (TypeError, ValueError):
        return np.nan
    return v if np.isfinite(v) and v > 1 else np.nan


def update(settings: Settings, matches: pd.DataFrame, preds: list[dict], now: pd.Timestamp | None = None) -> dict:
    """Coloca/actualiza apuestas de los próximos partidos y liquida las terminadas.

    Devuelve {"ledger": DataFrame, "settled": filas liquidadas en esta llamada, "placed": nuevas}."""
    c = cfg(settings)
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    stamp = now.isoformat(timespec="minutes")
    led = load_ledger(settings)
    bank = bankroll(led, c)
    rows, placed = [], []
    # 1) colocar o actualizar (hasta el inicio del partido)
    for p in preds:
        ko = pd.Timestamp(p["kickoff_utc"]) if p.get("kickoff_utc") else None
        if ko is not None and ko <= now:
            continue  # ya empezó: la apuesta queda congelada
        for strat, meta in STRATEGIES.items():
            mk = meta["market"]
            mask = (led.strategy == strat) & (led.season == p["season"]) & (led.home == p["home"]) & (led.away == p["away"])
            if (mask & (led.status != "pendiente")).any():
                continue
            prev = led[mask & (led.status == "pendiente")]
            probs, odds = _market_inputs(p, mk)
            d = decide(probs, odds, strat, c) if probs is not None and odds else None
            led = led[~(mask & (led.status == "pendiente"))]  # se sustituye (o se retira si ya no hay apuesta)
            if d is None:
                continue
            i, o, ev = d
            pick = (OUT if mk == "1x2" else OU_OUT)[i]
            stake = kelly_stake(probs[i], o, bank, c) if strat == "kelly" else float(c["stake"])
            if stake <= 0:
                continue
            same = len(prev) > 0 and str(prev.pick.iloc[0]) == pick
            first = float(prev.odds_first.iloc[0]) if same and np.isfinite(prev.odds_first.iloc[0]) else o
            row = {"strategy": strat, "market": mk, "season": p["season"], "date": p["date"],
                   "kickoff_utc": p.get("kickoff_utc"), "home": p["home"], "away": p["away"], "pick": pick,
                   "prob": round(float(probs[i]), 4), "odds": o, "odds_first": first, "odds_close": np.nan,
                   "close_src": None, "clv": np.nan, "ev": round(ev, 4), "stake": stake,
                   "placed_at": prev.placed_at.iloc[0] if same else stamp,
                   "status": "pendiente", "score": None, "profit": 0.0, "settled_at": None}
            rows.append(row)
            if not same:
                placed.append(row)
    if rows:
        led = _typed(pd.concat([led, pd.DataFrame(rows, columns=LEDGER_COLS)], ignore_index=True))
    # 2) liquidar y completar la cuota de cierre
    res = matches[matches.hg.notna()].set_index(["season", "home", "away"]).sort_index()
    settled = []
    for i, r in led.iterrows():
        if r.status != "pendiente" and r.close_src == "bet365":
            continue
        k = (int(r.season), r.home, r.away)
        if k not in res.index:
            continue
        m = res.loc[k]
        if isinstance(m, pd.DataFrame):
            m = m.iloc[0]
        close = _closing(m, r.market, str(r.pick))
        if np.isfinite(close):
            led.loc[i, "odds_close"], led.loc[i, "close_src"] = close, "bet365"
        elif r.status == "pendiente":
            led.loc[i, "odds_close"], led.loc[i, "close_src"] = float(r.odds), "última vista"
        oc = led.loc[i, "odds_close"]
        if np.isfinite(oc) and oc > 1:
            led.loc[i, "clv"] = round(float(r.odds_first) / oc - 1, 4)
        if r.status != "pendiente":
            continue
        if r.market == "1x2":
            outcome = "1" if m.hg > m.ag else "X" if m.hg == m.ag else "2"
        else:
            outcome = "+2.5" if m.hg + m.ag > 2.5 else "-2.5"
        won = str(r.pick) == outcome
        led.loc[i, "status"] = "ganada" if won else "perdida"
        led.loc[i, "score"] = f"{int(m.hg)}-{int(m.ag)}"
        led.loc[i, "profit"] = round(float(r.stake) * (float(r.odds) - 1), 2) if won else -float(r.stake)
        led.loc[i, "settled_at"] = stamp
        settled.append(led.loc[i].to_dict())
    led = led.sort_values(["date", "home", "strategy"]).reset_index(drop=True)
    led.to_csv(_path(settings), index=False)
    write_summary(settings, led)
    return {"ledger": led, "settled": settled, "placed": placed}


def _curve(done: pd.DataFrame) -> list:
    curve, acc = [], 0.0
    for r in done.itertuples():
        acc += float(r.profit)
        curve.append([r.date, round(acc, 2)])
    return curve


def summarize(led: pd.DataFrame, c: dict) -> dict:
    out = {}
    for strat, meta in STRATEGIES.items():
        d = led[led.strategy == strat]
        done = d[d.status.isin(["ganada", "perdida"])].sort_values(["settled_at", "date", "home"])
        staked = float(done.stake.sum())
        profit = float(done.profit.sum())
        clv = done.clv.dropna()
        s = {
            "label": meta["label"], "market": meta["market"], "bets": int(len(done)),
            "won": int((done.status == "ganada").sum()), "staked": round(staked, 2), "profit": round(profit, 2),
            "roi": round(profit / staked, 4) if staked else None,
            "avg_odds": round(float(done.odds.mean()), 2) if len(done) else None,
            "pending": int((d.status == "pendiente").sum()), "curve": _curve(done),
            "clv": round(float(clv.mean()), 4) if len(clv) else None,
            "beat_close": round(float((clv > 0).mean()), 3) if len(clv) else None,
        }
        if strat == "kelly":
            s["bankroll_start"] = float(c["kelly_bankroll"])
            s["bankroll"] = round(float(c["kelly_bankroll"]) + profit, 2)
        out[strat] = s
    return out


def write_summary(settings: Settings, led: pd.DataFrame) -> dict:
    c = cfg(settings)
    s = summarize(led, c)
    recent = led.sort_values(["date", "home"], ascending=False).head(300)
    out = {"updated": datetime.now(timezone.utc).isoformat(timespec="minutes"), "config": c,
           "strategies": s, "bets": json.loads(recent.to_json(orient="records", force_ascii=False))}
    (settings.predictions_dir / "bets_summary.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def format_settled(settled: list[dict], summary: dict | None = None) -> str:
    """Texto para Telegram con las apuestas liquidadas y el balance."""
    if not settled:
        return ""
    lines = ["Apuestas liquidadas (dinero ficticio)"]
    for r in settled:
        lines.append(f"{'✅' if r['status'] == 'ganada' else '❌'} {r['home']}–{r['away']} {r['score']}: "
                     f"{r['strategy']} {r['pick']} a {r['odds']:.2f} → {r['profit']:+.2f} €")
    if summary:
        lines.append("")
        lines.append("Balance total: " + " · ".join(f"{k} {v['profit']:+.0f} €" for k, v in summary.items() if v["bets"]))
    return "\n".join(lines)


def format_placed(placed: list[dict]) -> str:
    if not placed:
        return ""
    lines = ["Apuestas nuevas (dinero ficticio)"]
    for r in placed:
        lines.append(f"• {r['home']}–{r['away']}: {r['strategy']} {r['pick']} a {r['odds']:.2f} "
                     f"({r['stake']:.1f} €, prob. {100 * r['prob']:.0f} %)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Simulación sobre temporadas pasadas (predicciones fuera de muestra del backtest)
# ---------------------------------------------------------------------------
def _fit_ou_blend(df: pd.DataFrame) -> dict:
    from sklearn.linear_model import LogisticRegression
    ok = df.p_model.notna() & df.p_mkt.notna()
    if ok.sum() < 300:
        return dict(OU_BLEND_DEFAULT)
    X = np.column_stack([logit(df.p_model[ok]), logit(df.p_mkt[ok])])
    m = LogisticRegression(C=1.0).fit(X, df.y_over[ok])
    return {"a_model": round(float(m.coef_[0, 0]), 4), "a_market": round(float(m.coef_[0, 1]), 4),
            "b": round(float(m.intercept_[0]), 4), "n": int(ok.sum())}


def p_over_from_goals(lam, mu, rho: float = -0.05) -> np.ndarray:
    from .models.dixon_coles import score_matrix
    out = np.full(len(lam), np.nan)
    for k, (l, m) in enumerate(zip(lam, mu)):
        if np.isfinite(l) and np.isfinite(m):
            M = score_matrix(float(l), float(m), rho, 10)
            M = M / M.sum()
            i, j = np.indices(M.shape)
            out[k] = M[(i + j) > 2].sum()
    return out


def _strategy_result(rows: list, label: str, extra: dict | None = None) -> dict:
    df = pd.DataFrame(rows, columns=["season", "date", "stake", "profit", "won", "odds", "clv"]).sort_values("date")
    cum = df.profit.cumsum().to_numpy()
    step = max(1, len(df) // 150)
    by = df.groupby("season").agg(bets=("profit", "size"), won=("won", "sum"), profit=("profit", "sum"),
                                  staked=("stake", "sum")).reset_index()
    staked = float(df.stake.sum())
    out = {
        "label": label, "bets": int(len(df)), "won": int(df.won.sum()), "staked": round(staked, 2),
        "profit": round(float(df.profit.sum()), 2), "roi": round(float(df.profit.sum()) / staked, 4) if staked else None,
        "avg_odds": round(float(df.odds.mean()), 2),
        "clv": round(float(df.clv.mean()), 4) if df.clv.notna().any() else None,
        "beat_close": round(float((df.clv.dropna() > 0).mean()), 3) if df.clv.notna().any() else None,
        "by_season": [{"season": int(r.season), "bets": int(r.bets), "won": int(r.won), "profit": round(float(r.profit), 2),
                       "roi": round(float(r.profit) / r.staked, 4) if r.staked else None} for r in by.itertuples()],
        "curve": [[df.date.iloc[i], round(float(cum[i]), 2)] for i in range(0, len(df), step)]
                 + [[df.date.iloc[-1], round(float(cum[-1]), 2)]],
    }
    out.update(extra or {})
    return out


def backtest_strategies(settings: Settings, matches: pd.DataFrame) -> dict | None:
    p = settings.processed_dir / "backtest_predictions.csv"
    if not p.exists() or "b365_h" not in matches.columns:
        return None
    bt = pd.read_csv(p, index_col=0)
    m = matches.loc[matches.index.intersection(bt.index)]
    bt = bt.loc[m.index]
    if "stacker_mercado_H" in bt:
        has = bt["stacker_mercado_H"].notna().to_numpy()[:, None]
        probs = np.where(has, bt[["stacker_mercado_H", "stacker_mercado_D", "stacker_mercado_A"]].to_numpy(),
                         bt[["stacker_H", "stacker_D", "stacker_A"]].to_numpy())
    else:
        probs = bt[["stacker_H", "stacker_D", "stacker_A"]].to_numpy()
    odds = m[["b365_h", "b365_d", "b365_a"]].to_numpy(float)
    close = m[["b365c_h", "b365c_d", "b365c_a"]].to_numpy(float) if "b365c_h" in m else np.full_like(odds, np.nan)
    y = bt.y.to_numpy().astype(int)
    c = cfg(settings)
    res = {"config": c, "strategies": {}}
    seasons = m.season.to_numpy().astype(int)
    dates = m.date.astype(str).str[:10].to_numpy()
    order = np.argsort(dates, kind="stable")
    for strat in ("favorito", "valor", "kelly"):
        rows, bank = [], float(c["kelly_bankroll"])
        for k in order:
            d = decide(probs[k], odds[k], strat, c)
            if d is None:
                continue
            i, o, _ = d
            stake = kelly_stake(probs[k][i], o, bank, c) if strat == "kelly" else float(c["stake"])
            if stake <= 0:
                continue
            won = i == y[k]
            prof = stake * (o - 1) if won else -stake
            bank += prof
            cl = o / close[k][i] - 1 if np.isfinite(close[k][i]) and close[k][i] > 1 else np.nan
            rows.append((seasons[k], dates[k], stake, prof, won, o, cl))
        if rows:
            extra = {"bankroll_start": c["kelly_bankroll"], "bankroll": round(bank, 2)} if strat == "kelly" else None
            res["strategies"][strat] = _strategy_result(rows, STRATEGIES[strat]["label"], extra=extra)
    try:
        ou = _ou_backtest(settings, matches, c)
        if ou:
            res["strategies"]["goles"] = ou
    except Exception as e:
        log.warning("Backtest de más/menos no disponible: %s", e)
    if not res["strategies"]:
        return None
    (settings.predictions_dir / "betting_backtest.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    return res


def _ou_backtest(settings: Settings, matches: pd.DataFrame, c: dict) -> dict | None:
    """Más/menos 2,5: Dixon-Coles fuera de muestra combinado con el mercado; los pesos de la
    combinación se estiman solo con temporadas anteriores a cada temporada de prueba."""
    tp = settings.processed_dir / "oos_table.csv"
    need = {"b365_o25", "b365_u25", "ou_o25", "ou_u25"}
    if not tp.exists() or not need <= set(matches.columns):
        return None
    t = pd.read_csv(tp, parse_dates=["date"], usecols=lambda col: col in {
        "season", "home", "away", "date", "hg", "ag", "dc_lam", "dc_mu", "dc_po25"})
    odds_cols = [x for x in ("b365_o25", "b365_u25", "b365c_o25", "b365c_u25", "ou_o25", "ou_u25") if x in matches]
    t = t.merge(matches[["season", "home", "away", *odds_cols]], on=["season", "home", "away"], how="left")
    t = t[t.hg.notna()].copy()
    if "dc_po25" in t and t.dc_po25.notna().mean() > 0.9:
        t["p_model"] = t.dc_po25.astype(float)
    else:
        t["p_model"] = p_over_from_goals(t.dc_lam.to_numpy(float), t.dc_mu.to_numpy(float))
    t["p_mkt"] = pd.array([devig2(a, b) for a, b in zip(t.ou_o25, t.ou_u25)], dtype="Float64").astype(float)
    t["y_over"] = ((t.hg + t.ag) > 2.5).astype(int)
    first_test = int(settings.get("backtest", "first_test_season", 2021))
    rows = []
    t["p_wf"] = np.nan
    for season in sorted(t.season.unique()):
        if season < first_test:
            continue
        w = _fit_ou_blend(t[t.season < season])
        cur = t.season == season
        t.loc[cur, "p_wf"] = [blend_ou(a, b, w) for a, b in zip(t.p_model[cur], t.p_mkt[cur])]
        te = t[cur & t.b365_o25.notna()]
        for r in te.sort_values("date").itertuples():
            po = blend_ou(r.p_model, r.p_mkt, w)
            d = decide([po, 1 - po], [r.b365_o25, r.b365_u25], "goles", c)
            if d is None:
                continue
            i, o, _ = d
            won = (i == 0) == bool(r.y_over)
            cl_o = [getattr(r, "b365c_o25", np.nan), getattr(r, "b365c_u25", np.nan)][i]
            cl = o / cl_o - 1 if cl_o == cl_o and cl_o > 1 else np.nan
            rows.append((int(season), str(r.date)[:10], float(c["stake"]),
                         c["stake"] * (o - 1) if won else -c["stake"], won, o, cl))
    wf = _fit_ou_blend(t)  # pesos con todo el histórico, para las predicciones en vivo
    (settings.state_dir / "ou_blend.json").write_text(json.dumps(wf), encoding="utf-8")
    if not rows:
        return None
    ll = {}
    ok = t.p_mkt.notna() & (t.season >= first_test)
    if ok.any():
        yy = t.y_over[ok].to_numpy()
        for name, pp in (("modelo", t.p_model[ok]), ("mercado", t.p_mkt[ok]),
                         ("combinado", t.p_wf[ok])):
            pp = np.clip(np.asarray(pp, float), 1e-6, 1 - 1e-6)
            ll[name] = round(float(-np.mean(yy * np.log(pp) + (1 - yy) * np.log(1 - pp))), 4)
    return _strategy_result(rows, STRATEGIES["goles"]["label"], extra={"log_loss": ll, "blend": wf})
