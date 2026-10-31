"""Cartera de apuestas SIMULADA con las cuotas de Bet365 (sin dinero real).

Estrategias:
  favorito  10 € al resultado más probable según el modelo (rellenar la quiniela)
  valor     10 € solo si probabilidad del modelo × cuota ≥ 1 + margen mínimo
  goles     10 € a más/menos de 2,5 goles cuando esa cuota tiene valor
  comb_favoritos  una combinada por jornada con los favoritos más claros (ver más abajo)
  recalibrado     10 € cuando la probabilidad del mercado recalibrado (market_recal.py) × cuota
                  supera 1 + margen mínimo: busca el sesgo favorito-sorpresa de las casas

kelly_stake() se conserva para cuando una estrategia demuestre ventaja: con ventaja negativa,
apostar según Kelly solo sirve para perder más despacio.

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
    "recalibrado": {"label": "Solo cuando la cuota paga más de lo que dice el mercado recalibrado", "market": "1x2"},
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
    c.update({"recal_min_edge": 0.02, "parlay_legs": 3, "parlay_min_legs": 2, "parlay_min_leg_prob": 0.6})
    c.update(getattr(settings, "raw", {}).get("betting", {}) or {})
    return c


def roi_ci(stakes, profits, n_boot: int = 4000, seed: int = 0) -> list[float] | None:
    """IC 95 % del ROI remuestreando apuestas (bootstrap). Con pocas apuestas es muy ancho, y es la idea."""
    st, pr = np.asarray(stakes, float), np.asarray(profits, float)
    if len(st) < 5 or st.sum() <= 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(st), (n_boot, len(st)))
    bs = pr[idx].sum(1) / st[idx].sum(1)
    return [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


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
    if strategy in ("valor", "recalibrado", "goles"):
        edge = {"goles": c["ou_min_edge"], "recalibrado": c.get("recal_min_edge", 0.02)}.get(strategy, c["min_edge"])
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


def _market_inputs(p: dict, market: str, strategy: str = ""):
    if strategy == "recalibrado":
        r = p.get("market_recal")
        return (list(r), p.get("b365")) if r else (None, None)
    if market == "1x2":
        return [p["p_home"], p["p_draw"], p["p_away"]], p.get("b365")
    ou = p.get("ou") or {}
    po = ou.get("p_over")
    if po is None and (p.get("dc") or {}).get("p_over25") is not None:
        po = p["dc"]["p_over25"]
    if po is None:
        return None, None
    return [po, 1 - po], p.get("b365_ou")


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
    from . import market_recal
    rm = market_recal.load_model(settings)
    for p in preds:
        if p.get("market_recal") is None:
            p["market_recal"] = market_recal.recal_from_odds(rm, p.get("b365"))
    led = led[led.strategy.isin(list(STRATEGIES)) | (led.status != "pendiente")]  # estrategias retiradas
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
            probs, odds = _market_inputs(p, mk, strat)
            d = decide(probs, odds, strat, c) if probs is not None and odds else None
            led = led[~(mask & (led.status == "pendiente"))]  # se sustituye (o se retira si ya no hay apuesta)
            if d is None:
                continue
            i, o, ev = d
            pick = (OUT if mk == "1x2" else OU_OUT)[i]
            stake = float(c["stake"])
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
    par = update_parlays(settings, matches, preds, now)
    write_summary(settings, led, par["parlays"])
    return {"ledger": led, "settled": settled, "placed": placed,
            "parlays_placed": par["placed"], "parlays_settled": par["settled"]}


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
            "roi_ci95": roi_ci(done.stake, done.profit),
            "avg_odds": round(float(done.odds.mean()), 2) if len(done) else None,
            "pending": int((d.status == "pendiente").sum()), "curve": _curve(done),
            "clv": round(float(clv.mean()), 4) if len(clv) else None,
            "beat_close": round(float((clv > 0).mean()), 3) if len(clv) else None,
        }
        out[strat] = s
    return out


def write_summary(settings: Settings, led: pd.DataFrame, parlays: pd.DataFrame | None = None) -> dict:
    c = cfg(settings)
    s = summarize(led, c)
    parlays = load_parlays(settings) if parlays is None else parlays
    s.update(summarize_parlays(parlays))
    recent = led[led.strategy.isin(list(STRATEGIES))].sort_values(["date", "home"], ascending=False).head(300)
    rp = parlays[parlays.strategy.isin(list(PARLAYS))].sort_values("slot", ascending=False).head(100).copy()
    rp["legs"] = [json.loads(x) for x in rp.legs]
    out = {"updated": datetime.now(timezone.utc).isoformat(timespec="minutes"), "config": c,
           "strategies": s, "bets": json.loads(recent.to_json(orient="records", force_ascii=False)),
           "parlays": json.loads(rp.to_json(orient="records", force_ascii=False)),
           "preregistro": preregistration(settings, parlays, led)}
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
# Combinadas (varias selecciones en un solo boleto; se gana solo si aciertan todas)
# ---------------------------------------------------------------------------
# Probabilidad y cuota se multiplican. El margen de la casa también: con un 5 % por
# selección, una combinada de 3 lleva en torno a un 14 %. Se simulan para comprobarlo.
PARLAYS = {
    "comb_favoritos": {"label": "Combinada de los favoritos más claros de la jornada (cada uno con al menos "
                                "un 60 % según el modelo)"},
}
PARLAY_COLS = ["strategy", "season", "slot", "legs", "n_legs", "prob", "odds", "odds_first", "odds_close", "clv",
               "ev", "stake", "kickoff_first", "placed_at", "status", "score", "profit", "settled_at"]
PARLAY_FLOAT = ("prob", "odds", "odds_first", "odds_close", "clv", "ev", "stake", "profit")


def slot_of(day) -> str:
    """Jornada aproximada de un partido: viernes-lunes cuentan como fin de semana (clave: el viernes)
    y martes-jueves como jornada entre semana (clave: el martes)."""
    d = pd.Timestamp(day).normalize()
    wd = d.weekday()  # lunes = 0
    back = {0: 3, 1: 0, 2: 1, 3: 2, 4: 0, 5: 1, 6: 2}[wd]
    return (d - pd.Timedelta(days=back)).strftime("%Y-%m-%d")


def _local_day(p: dict) -> str:
    if p.get("kickoff_utc"):
        try:
            return pd.Timestamp(p["kickoff_utc"]).tz_convert("Europe/Madrid").strftime("%Y-%m-%d")
        except (TypeError, ValueError):
            pass
    return str(p["date"])[:10]


def choose_legs(cands: list[dict], strategy: str, c: dict) -> list[dict] | None:
    """cands: {"home","away","p": [3 probs], "odds": [3 cuotas], ...}. Devuelve las selecciones o None."""
    legs = []
    for m in cands:
        if strategy == "comb_favoritos":
            d = decide(m["p"], m["odds"], "favorito", c)
            if d is None or m["p"][d[0]] < c["parlay_min_leg_prob"]:
                continue
        else:
            raise ValueError(strategy)
        i, o, ev = d
        legs.append({**{k: v for k, v in m.items() if k not in ("p", "odds")},
                     "pick": OUT[i], "prob": round(float(m["p"][i]), 4), "odds": float(o), "ev": round(ev, 4)})
    legs = sorted(legs, key=lambda x: -x["prob"])[: int(c["parlay_legs"])]
    return legs if len(legs) >= int(c["parlay_min_legs"]) else None


def _parlay_path(settings: Settings):
    return settings.predictions_dir / "parlays.csv"


def _typed_parlays(led: pd.DataFrame) -> pd.DataFrame:
    led = led.copy()
    for col in PARLAY_COLS:
        if col not in led:
            led[col] = np.nan
    for col in PARLAY_COLS:
        led[col] = pd.to_numeric(led[col], errors="coerce").astype(float) if col in PARLAY_FLOAT else led[col].astype(object)
    return led[PARLAY_COLS]


def load_parlays(settings: Settings) -> pd.DataFrame:
    p = _parlay_path(settings)
    if not p.exists():
        return _typed_parlays(pd.DataFrame(columns=PARLAY_COLS))
    return _typed_parlays(pd.read_csv(p, dtype={"slot": str, "score": str}))


def _leg_key(legs: list[dict]) -> tuple:
    return tuple(sorted((x["home"], x["away"], x["pick"]) for x in legs))


def update_parlays(settings: Settings, matches: pd.DataFrame, preds: list[dict], now: pd.Timestamp) -> dict:
    """Coloca/ajusta las combinadas de cada jornada hasta que empieza su primer partido y las liquida:
    perdida en cuanto falla una selección; ganada cuando aciertan todas."""
    c = cfg(settings)
    stamp = now.isoformat(timespec="minutes")
    led = load_parlays(settings)
    placed, settled = [], []
    by_slot: dict[tuple, list] = {}
    for p in preds:
        if not p.get("b365") or not p.get("kickoff_utc"):
            continue
        if pd.Timestamp(p["kickoff_utc"]) <= now:
            continue
        cand = {"home": p["home"], "away": p["away"], "date": p["date"], "kickoff_utc": p["kickoff_utc"],
                "p": [p["p_home"], p["p_draw"], p["p_away"]], "odds": list(p["b365"])}
        by_slot.setdefault((int(p["season"]), slot_of(_local_day(p))), []).append(cand)
    rows = []
    for (season, slot), cands in by_slot.items():
        for strat in PARLAYS:
            mask = (led.strategy == strat) & (led.season.astype(float) == season) & (led.slot == slot)
            prev = led[mask]
            if len(prev) and (prev.status.iloc[0] != "pendiente"
                              or pd.Timestamp(prev.kickoff_first.iloc[0]) <= now):
                continue  # liquidada o congelada (ya empezó alguna selección)
            legs = choose_legs(cands, strat, c)
            led = led[~mask]
            if legs is None:
                continue
            prob, odds = float(np.prod([x["prob"] for x in legs])), float(np.prod([x["odds"] for x in legs]))
            same = len(prev) > 0 and _leg_key(json.loads(prev.legs.iloc[0])) == _leg_key(legs)
            row = {"strategy": strat, "season": season, "slot": slot, "legs": json.dumps(legs, ensure_ascii=False),
                   "n_legs": len(legs), "prob": round(prob, 4), "odds": round(odds, 3),
                   "odds_first": float(prev.odds_first.iloc[0]) if same else round(odds, 3),
                   "odds_close": np.nan, "clv": np.nan, "ev": round(prob * odds - 1, 4), "stake": float(c["stake"]),
                   "kickoff_first": min(x["kickoff_utc"] for x in legs),
                   "placed_at": prev.placed_at.iloc[0] if same else stamp, "status": "pendiente",
                   "score": None, "profit": 0.0, "settled_at": None}
            rows.append(row)
            if not same:
                placed.append(row)
    if rows:
        led = _typed_parlays(pd.concat([led, pd.DataFrame(rows, columns=PARLAY_COLS)], ignore_index=True))
    res = matches[matches.hg.notna()].set_index(["season", "home", "away"]).sort_index()
    for i, r in led.iterrows():
        if r.status != "pendiente":
            continue
        legs = json.loads(r.legs)
        hits, done, closes = 0, 0, []
        for leg in legs:
            k = (int(r.season), leg["home"], leg["away"])
            if k not in res.index:
                closes.append(np.nan)
                continue
            m = res.loc[k]
            m = m.iloc[0] if isinstance(m, pd.DataFrame) else m
            outcome = "1" if m.hg > m.ag else "X" if m.hg == m.ag else "2"
            leg["score"] = f"{int(m.hg)}-{int(m.ag)}"
            leg["won"] = outcome == leg["pick"]
            done += 1
            hits += leg["won"]
            closes.append(_closing(m, "1x2", leg["pick"]))
        if done == 0:
            continue
        led.loc[i, "legs"] = json.dumps(legs, ensure_ascii=False)
        lost = hits < done
        if not lost and done < len(legs):
            continue  # de momento acierta todas, faltan partidos
        if all(np.isfinite(closes)):
            oc = float(np.prod(closes))
            led.loc[i, "odds_close"], led.loc[i, "clv"] = round(oc, 3), round(float(r.odds_first) / oc - 1, 4)
        led.loc[i, "status"] = "perdida" if lost else "ganada"
        led.loc[i, "score"] = f"{hits}/{len(legs)}" if not lost else f"falla {done - hits} de {done} jugadas"
        led.loc[i, "profit"] = -float(r.stake) if lost else round(float(r.stake) * (float(r.odds) - 1), 2)
        led.loc[i, "settled_at"] = stamp
        settled.append(led.loc[i].to_dict())
    led = led.sort_values(["slot", "strategy"]).reset_index(drop=True)
    led.to_csv(_parlay_path(settings), index=False)
    return {"parlays": led, "placed": placed, "settled": settled}


def preregistration(settings: Settings, parlays: pd.DataFrame, singles: pd.DataFrame | None = None) -> dict | None:
    """Test preregistrado: reglas fijadas en config.yaml → preregistro ANTES de ver los datos en directo.

    Solo cuentan los boletos/apuestas colocados desde la fecha de inicio. Si alguien cambia el umbral
    después, el test queda invalidado (en vez de reajustarlo hasta que salga bonito)."""
    pr = getattr(settings, "raw", {}).get("preregistro") or {}
    if not pr:
        return None
    c = cfg(settings)
    strat, since = pr.get("estrategia", "comb_favoritos"), str(pr.get("desde", "2026-11-01"))
    if strat in PARLAYS:
        d = parlays[(parlays.strategy == strat) & parlays.status.isin(["ganada", "perdida"])]
        d = d[d.slot.astype(str) >= since]
    else:
        d = singles if singles is not None else pd.DataFrame(columns=LEDGER_COLS)
        d = d[(d.strategy == strat) & d.status.isin(["ganada", "perdida"]) & (d.date.astype(str) >= since)]
    n, need = int(len(d)), int(pr.get("min_apuestas", 60))
    staked = float(d.stake.sum())
    ci = roi_ci(d.stake, d.profit)
    clv = d.clv.dropna()
    changed = [k for k, v in (pr.get("reglas") or {}).items() if k in c and float(c[k]) != float(v)]
    out = {"estrategia": strat, "desde": since, "min_apuestas": need, "reglas": pr.get("reglas") or {},
           "criterio": pr.get("criterio", ""), "n": n,
           "roi": round(float(d.profit.sum()) / staked, 4) if staked else None, "roi_ci95": ci,
           "clv": round(float(clv.mean()), 4) if len(clv) else None, "reglas_cambiadas": changed}
    if changed:
        out["estado"] = "invalidado"
    elif n < need:
        out["estado"] = "en curso"
    else:
        ok = ci is not None and ci[0] > 0 and out["clv"] is not None and out["clv"] > 0
        out["estado"] = "superado" if ok else "no superado"
    return out


def summarize_parlays(led: pd.DataFrame) -> dict:
    out = {}
    for strat, meta in PARLAYS.items():
        d = led[led.strategy == strat]
        done = d[d.status.isin(["ganada", "perdida"])].sort_values(["settled_at", "slot"])
        staked, profit = float(done.stake.sum()), float(done.profit.sum())
        clv = done.clv.dropna()
        out[strat] = {
            "label": meta["label"], "market": "combinada", "bets": int(len(done)),
            "won": int((done.status == "ganada").sum()), "staked": round(staked, 2), "profit": round(profit, 2),
            "roi": round(profit / staked, 4) if staked else None,
            "roi_ci95": roi_ci(done.stake, done.profit),
            "avg_odds": round(float(done.odds.mean()), 2) if len(done) else None,
            "avg_prob": round(float(done.prob.mean()), 4) if len(done) else None,
            "pending": int((d.status == "pendiente").sum()),
            "curve": [[r.slot, v] for r, v in zip(done.itertuples(), np.round(done.profit.cumsum(), 2))],
            "clv": round(float(clv.mean()), 4) if len(clv) else None,
            "beat_close": round(float((clv > 0).mean()), 3) if len(clv) else None,
        }
    return out


def format_parlays(placed: list[dict], settled: list[dict]) -> str:
    lines = []
    for r in placed:
        legs = " + ".join(f"{x['home']}–{x['away']} {x['pick']}" for x in json.loads(r["legs"]))
        lines.append(f"🧩 Combinada nueva ({r['strategy']}): {legs} · cuota {r['odds']:.2f}, "
                     f"prob. {100 * r['prob']:.0f} %")
    for r in settled:
        lines.append(f"{'✅' if r['status'] == 'ganada' else '❌'} Combinada {r['strategy']} ({r['slot']}): "
                     f"{r['score']} → {r['profit']:+.2f} €")
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
        "roi_ci95": roi_ci(df.stake, df.profit),
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
    rp = settings.processed_dir / "market_recal_oos.csv"
    recal = np.full_like(probs, np.nan)
    if rp.exists():
        r = pd.read_csv(rp, index_col=0).reindex(m.index)
        recal = r[["recal_H", "recal_D", "recal_A"]].to_numpy(float)
    for strat in ("favorito", "valor", "recalibrado"):
        pr = recal if strat == "recalibrado" else probs
        rows = []
        for k in order:
            d = decide(pr[k], odds[k], strat, c)
            if d is None:
                continue
            i, o, _ = d
            stake = float(c["stake"])
            won = i == y[k]
            prof = stake * (o - 1) if won else -stake
            cl = o / close[k][i] - 1 if np.isfinite(close[k][i]) and close[k][i] > 1 else np.nan
            rows.append((seasons[k], dates[k], stake, prof, won, o, cl))
        if rows:
            res["strategies"][strat] = _strategy_result(rows, STRATEGIES[strat]["label"])
    res["strategies"].update(_parlay_backtest(m, probs, odds, close, y, seasons, dates, c))
    # ¿Y si se apostara siempre a la mejor cuota del mercado, o a la media?
    res["odds_compare"] = {}
    for label, cols in (("bet365", ("b365_h", "b365_d", "b365_a")), ("mejor", ("max_h", "max_d", "max_a")),
                        ("media", ("avg_h", "avg_d", "avg_a"))):
        if not set(cols) <= set(m.columns):
            continue
        od = m[list(cols)].to_numpy(float)
        for strat in ("favorito", "valor"):
            prof, n = 0.0, 0
            for k in range(len(m)):
                d = decide(probs[k], od[k], strat, c)
                if d is None:
                    continue
                i, o, _ = d
                prof += c["stake"] * (o - 1) if i == y[k] else -c["stake"]
                n += 1
            res["odds_compare"].setdefault(strat, {})[label] = {
                "bets": n, "profit": round(prof, 2), "roi": round(prof / (n * c["stake"]), 4) if n else None}
    if {"b365_h", "max_h", "avg_h"} <= set(m.columns):
        b, mx, av = (m[[f"{x}_h", f"{x}_d", f"{x}_a"]].to_numpy(float) for x in ("b365", "max", "avg"))
        ok = ~np.isnan(b).any(1) & ~np.isnan(mx).any(1) & ~np.isnan(av).any(1)
        res["odds_quality"] = {"n": int(ok.sum()),
                               "b365_vs_avg": round(float(np.nanmean(b[ok] / av[ok] - 1)), 4),
                               "max_vs_b365": round(float(np.nanmean(mx[ok] / b[ok] - 1)), 4),
                               "b365_is_best": round(float((b[ok] >= mx[ok] - 1e-9).mean()), 4),
                               "margin_b365": round(float(np.nanmean((1 / b[ok]).sum(1) - 1)), 4),
                               "margin_max": round(float(np.nanmean((1 / mx[ok]).sum(1) - 1)), 4)}
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


def _parlay_backtest(m: pd.DataFrame, probs, odds, close, y, seasons, dates, c: dict) -> dict:
    """Combinadas en temporadas pasadas: una por estrategia y jornada, con las cuotas previas al partido."""
    slots = np.array([slot_of(d) for d in dates])
    home, away = m.home.to_numpy(), m.away.to_numpy()
    out = {}
    for strat, meta in PARLAYS.items():
        rows, probs_used = [], []
        for (season, slot) in sorted(set(zip(seasons, slots)), key=lambda t: t[1]):
            idx = np.flatnonzero((seasons == season) & (slots == slot))
            cands = [{"k": int(k), "home": home[k], "away": away[k], "p": probs[k], "odds": odds[k]} for k in idx]
            legs = choose_legs(cands, strat, c)
            if legs is None:
                continue
            o = float(np.prod([x["odds"] for x in legs]))
            won = all(OUT[y[x["k"]]] == x["pick"] for x in legs)
            cl = [close[x["k"]][OUT.index(x["pick"])] for x in legs]
            clv = o / float(np.prod(cl)) - 1 if all(np.isfinite(cl)) and min(cl) > 1 else np.nan
            prof = c["stake"] * (o - 1) if won else -c["stake"]
            rows.append((int(season), slot, float(c["stake"]), prof, won, o, clv))
            probs_used.append(float(np.prod([x["prob"] for x in legs])))
        if rows:
            n = len(rows)
            out[strat] = _strategy_result(rows, meta["label"], extra={
                "avg_prob": round(float(np.mean(probs_used)), 4),
                "hit_rate": round(sum(r[4] for r in rows) / n, 4)})
    return out


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
