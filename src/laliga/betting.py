"""Cartera de apuestas SIMULADA con las cuotas de Bet365 (sin dinero real).

Dos estrategias, cada una con una apuesta fija por partido:
  favorito  apuesta al resultado más probable según el modelo (rellenar la quiniela)
  valor     apuesta solo si probabilidad del modelo × cuota ≥ 1 + margen mínimo

Cada apuesta queda "colocada" con la última cuota de Bet365 vista antes del partido y se
liquida cuando hay resultado. Todo se guarda en predictions/bets.csv (historial completo),
el resumen en predictions/bets_summary.json y la simulación sobre temporadas pasadas en
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
STRATEGIES = {
    "favorito": "Al resultado más probable de cada partido",
    "valor": "Solo cuando la cuota paga más de lo que dice el modelo",
}
LEDGER_COLS = ["strategy", "season", "date", "kickoff_utc", "home", "away", "pick", "prob", "odds", "ev", "stake",
               "placed_at", "status", "score", "profit", "settled_at"]


def cfg(settings: Settings) -> dict:
    c = {"stake": 10.0, "min_edge": 0.05, "max_odds": 8.0, "min_prob": 0.0}
    c.update(settings.raw.get("betting", {}) or {})
    return c


def decide(p, odds, strategy: str, c: dict):
    """Devuelve (índice del resultado, cuota, valor esperado) o None si no se apuesta."""
    p = np.asarray(p, float)
    o = np.asarray([np.nan if v is None else v for v in odds], float)
    if np.isnan(o).any() or (o <= 1).any() or np.isnan(p).any():
        return None
    ev = p * o - 1
    if strategy == "favorito":
        i = int(np.argmax(p))
        return i, float(o[i]), float(ev[i])
    if strategy == "valor":
        ok = (ev >= c["min_edge"]) & (o <= c["max_odds"]) & (p >= c["min_prob"])
        if not ok.any():
            return None
        i = int(np.argmax(np.where(ok, ev, -np.inf)))
        return i, float(o[i]), float(ev[i])
    raise ValueError(strategy)


# ---------------------------------------------------------------------------
# Libro de apuestas
# ---------------------------------------------------------------------------
def _path(settings: Settings):
    return settings.predictions_dir / "bets.csv"


def load_ledger(settings: Settings) -> pd.DataFrame:
    p = _path(settings)
    if not p.exists():
        return pd.DataFrame(columns=LEDGER_COLS)
    return pd.read_csv(p, dtype={"pick": str, "score": str})


def update(settings: Settings, matches: pd.DataFrame, preds: list[dict], now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Coloca/actualiza apuestas pendientes de los próximos partidos y liquida las terminadas."""
    c = cfg(settings)
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    led = load_ledger(settings)
    key = ["strategy", "season", "home", "away"]
    pending = led.status == "pendiente"
    # 1) colocar o actualizar (hasta el inicio del partido)
    rows = []
    for p in preds:
        ko = pd.Timestamp(p["kickoff_utc"]) if p.get("kickoff_utc") else None
        if ko is not None and ko <= now:
            continue  # ya empezó: la apuesta queda congelada
        odds = p.get("b365")
        for strat in STRATEGIES:
            mask = (led.strategy == strat) & (led.season == p["season"]) & (led.home == p["home"]) & (led.away == p["away"])
            settled = mask & ~pending
            if settled.any():
                continue
            d = decide([p["p_home"], p["p_draw"], p["p_away"]], odds or [None] * 3, strat, c) if odds else None
            led = led[~(mask & pending)]  # se sustituye la pendiente (o se retira si ya no hay apuesta)
            pending = led.status == "pendiente"
            if d is None:
                continue
            i, o, ev = d
            rows.append({"strategy": strat, "season": p["season"], "date": p["date"], "kickoff_utc": p.get("kickoff_utc"),
                         "home": p["home"], "away": p["away"], "pick": OUT[i],
                         "prob": round([p["p_home"], p["p_draw"], p["p_away"]][i], 4), "odds": o, "ev": round(ev, 4),
                         "stake": c["stake"], "placed_at": now.isoformat(timespec="minutes"), "status": "pendiente",
                         "score": None, "profit": 0.0, "settled_at": None})
    if rows:
        led = pd.concat([led, pd.DataFrame(rows, columns=LEDGER_COLS)], ignore_index=True)
    # 2) liquidar
    for col in ("status", "score", "settled_at", "pick", "kickoff_utc", "placed_at"):
        led[col] = led[col].astype(object)
    led["profit"] = led["profit"].astype(float)
    res = matches[matches.hg.notna()].set_index(["season", "home", "away"]).sort_index()
    for i, r in led[led.status == "pendiente"].iterrows():
        k = (int(r.season), r.home, r.away)
        if k not in res.index:
            continue
        m = res.loc[k]
        if isinstance(m, pd.DataFrame):
            m = m.iloc[0]
        outcome = "1" if m.hg > m.ag else "X" if m.hg == m.ag else "2"
        won = str(r.pick) == outcome
        led.loc[i, "status"] = "ganada" if won else "perdida"
        led.loc[i, "score"] = f"{int(m.hg)}-{int(m.ag)}"
        led.loc[i, "profit"] = round(float(r.stake) * (float(r.odds) - 1), 2) if won else -float(r.stake)
        led.loc[i, "settled_at"] = now.isoformat(timespec="minutes")
    led = led.sort_values(["date", "home", "strategy"]).reset_index(drop=True)
    led.to_csv(_path(settings), index=False)
    write_summary(settings, led)
    return led


def summarize(led: pd.DataFrame) -> dict:
    out = {}
    for strat, label in STRATEGIES.items():
        d = led[led.strategy == strat]
        done = d[d.status.isin(["ganada", "perdida"])].sort_values(["settled_at", "date", "home"])
        staked = float(done.stake.sum())
        profit = float(done.profit.sum())
        curve, acc = [], 0.0
        for r in done.itertuples():
            acc += float(r.profit)
            curve.append([r.date, round(acc, 2)])
        out[strat] = {
            "label": label, "bets": int(len(done)), "won": int((done.status == "ganada").sum()),
            "staked": round(staked, 2), "profit": round(profit, 2), "roi": round(profit / staked, 4) if staked else None,
            "avg_odds": round(float(done.odds.mean()), 2) if len(done) else None,
            "pending": int((d.status == "pendiente").sum()), "curve": curve,
        }
    return out


def write_summary(settings: Settings, led: pd.DataFrame) -> dict:
    s = summarize(led)
    recent = led.sort_values(["date", "home"], ascending=False).head(200)
    out = {"updated": datetime.now(timezone.utc).isoformat(timespec="minutes"), "config": cfg(settings),
           "strategies": s,
           "bets": json.loads(recent.to_json(orient="records", force_ascii=False))}
    (settings.predictions_dir / "bets_summary.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Simulación sobre temporadas pasadas (predicciones fuera de muestra del backtest)
# ---------------------------------------------------------------------------
def backtest_strategies(settings: Settings, matches: pd.DataFrame) -> dict | None:
    p = settings.processed_dir / "backtest_predictions.csv"
    if not p.exists() or "b365_h" not in matches.columns:
        return None
    bt = pd.read_csv(p, index_col=0)
    m = matches.loc[matches.index.intersection(bt.index)]
    bt = bt.loc[m.index]
    probs = np.where(bt[["stacker_mercado_H"]].notna().to_numpy(),
                     bt[["stacker_mercado_H", "stacker_mercado_D", "stacker_mercado_A"]].to_numpy(),
                     bt[["stacker_H", "stacker_D", "stacker_A"]].to_numpy()) \
        if "stacker_mercado_H" in bt else bt[["stacker_H", "stacker_D", "stacker_A"]].to_numpy()
    odds = m[["b365_h", "b365_d", "b365_a"]].to_numpy(float)
    y = bt.y.to_numpy().astype(int)
    c = cfg(settings)
    res = {"config": c, "strategies": {}}
    for strat, label in STRATEGIES.items():
        rows = []
        for k in range(len(m)):
            d = decide(probs[k], odds[k], strat, c)
            if d is None:
                continue
            i, o, _ = d
            won = i == y[k]
            rows.append((int(m.season.iloc[k]), str(m.date.iloc[k])[:10], c["stake"] * (o - 1) if won else -c["stake"], won, o))
        if not rows:
            continue
        df = pd.DataFrame(rows, columns=["season", "date", "profit", "won", "odds"]).sort_values("date")
        cum = df.profit.cumsum().to_numpy()
        step = max(1, len(df) // 120)
        by = df.groupby("season").agg(bets=("profit", "size"), won=("won", "sum"), profit=("profit", "sum")).reset_index()
        res["strategies"][strat] = {
            "label": label, "bets": int(len(df)), "won": int(df.won.sum()),
            "staked": round(len(df) * c["stake"], 2), "profit": round(float(df.profit.sum()), 2),
            "roi": round(float(df.profit.sum()) / (len(df) * c["stake"]), 4), "avg_odds": round(float(df.odds.mean()), 2),
            "by_season": [{"season": int(r.season), "bets": int(r.bets), "won": int(r.won), "profit": round(float(r.profit), 2),
                           "roi": round(float(r.profit) / (r.bets * c["stake"]), 4)} for r in by.itertuples()],
            "curve": [[df.date.iloc[i], round(float(cum[i]), 2)] for i in range(0, len(df), step)] + [[df.date.iloc[-1], round(float(cum[-1]), 2)]],
        }
    if not res["strategies"]:
        return None
    (settings.predictions_dir / "betting_backtest.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    return res
