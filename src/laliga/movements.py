"""Historial de probabilidades por partido y explicación de los cambios importantes.

Cada predicción guarda una "foto" de sus entradas (goles esperados base, déficits por
bajas, forma, calendario, cuotas, árbitro, clima...). Cuando una probabilidad cambia
más de `movements.threshold_pts` puntos respecto a la foto anterior, se explica:

1. se recalcula la foto antigua con el modelo actual (lo que cambie aquí es efecto de
   reentrenar/recalibrar el modelo);
2. se sustituye, grupo a grupo, la parte antigua por la nueva y se mide cuánto mueve
   la probabilidad cada grupo (bajas, resultados, calendario, cuotas, árbitro, clima);
3. cada grupo se acompaña de un detalle legible (quién se cae, qué resultado hubo...).

Ficheros (en predictions/):
  history.jsonl    una línea por partido y cambio de predicción (con su foto)
  movements.jsonl  cambios importantes con sus causas
  evolucion.md     evolución legible de cada partido próximo
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import Settings
from .features import availability as av
from .models.dixon_coles import score_matrix
from .models.stacker import GROUPS, Stacker

log = logging.getLogger(__name__)

OUTCOMES = ("1", "X", "2")
_FORM = [f"{k}_{s}_{side}" for k in ("gf", "sh", "sot", "cor", "xg") for s in ("for", "against") for side in ("home", "away")]
CALENDAR_COLS = ["rest_home", "rest_away", "midweek", "ko_early", "ko_late", "post_break", "early_season",
                 "euro_prev_home", "euro_prev_away", "euro_next_home", "euro_next_away", "rest_all_diff", "load14_diff"]
MARKET_COLS = ["odds_h", "odds_d", "odds_a", "odds_close_h", "odds_close_d", "odds_close_a",
               "mkt_ph", "mkt_pd", "mkt_pa", "mktc_ph", "mktc_pd", "mktc_pa"]
REF_COLS = ["ref_home_ppg_dev", "ref_cards_dev"]
WEATHER_COLS = ["temp_c", "precip_mm", "wind_kmh"]
RESULT_RAW = (["elo_home", "elo_away", "elo_diff", "pts_home", "pts_away", "venue_edge", "travel_km", "island",
               "new_home", "new_away"] + _FORM +
              [c for g in ("stakes", "h2h", "discipline", "derby", "momentum", "venue_form", "coach", "no_fans")
               for c in GROUPS[g]])
RAW_COLS = list(dict.fromkeys(RESULT_RAW + CALENDAR_COLS + MARKET_COLS + REF_COLS + WEATHER_COLS))
BASE_KEYS = ["lam0", "mu0", "sot_lam0", "sot_mu0", "xg_lam0", "xg_mu0", "rho"]

EXPL_GROUPS = [
    ("bajas", "Bajas y alineaciones", ["d_home", "d_away"], []),
    ("resultados", "Resultados y forma recientes", BASE_KEYS, RESULT_RAW),
    ("calendario", "Calendario y descanso", [], CALENDAR_COLS),
    ("mercado", "Cuotas de mercado", [], MARKET_COLS),
    ("arbitro", "Árbitro designado", [], REF_COLS),
    ("clima", "Clima previsto", [], WEATHER_COLS),
]


def _num(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


# ---------------------------------------------------------------------------
# Foto de las entradas de una predicción
# ---------------------------------------------------------------------------
def make_snapshot(r: pd.Series, o: dict, rho: float, a: float, b: float, max_goals: int) -> dict:
    eb = o.get("extra_base", {})
    return {
        "raw": {c: _num(r.get(c)) for c in RAW_COLS if c in r.index},
        "lam0": float(o["lam0"]), "mu0": float(o["mu0"]),
        "sot_lam0": _num(eb.get("sot", (None, None))[0]), "sot_mu0": _num(eb.get("sot", (None, None))[1]),
        "xg_lam0": _num(eb.get("xg", (None, None))[0]), "xg_mu0": _num(eb.get("xg", (None, None))[1]),
        "d_home": [float(x) for x in o["d_home"]], "d_away": [float(x) for x in o["d_away"]],
        "rho": float(rho), "a": float(a), "b": float(b), "max_goals": int(max_goals),
        "absences": o.get("absences") or {}, "lineups": o.get("lineups"),
        "expected_xi": o.get("expected_xi") or {}, "lineup_confirmed": bool(o.get("lineup_confirmed")),
        "referee": r.get("referee") if isinstance(r.get("referee"), str) else None,
    }


def probs_from_snapshot(snap: dict, model: str, stackers: dict, a: float, b: float) -> np.ndarray:
    """Reproduce la probabilidad final a partir de una foto (con el modelo indicado)."""
    lam, mu = av.adjust_goals(snap["lam0"], snap["mu0"], snap["d_home"], snap["d_away"], a, b)
    m = score_matrix(lam, mu, snap["rho"], snap.get("max_goals", 10))
    i, j = np.indices(m.shape)
    dc = np.array([m[i > j].sum(), m[i == j].sum(), m[i < j].sum()])
    st = stackers.get(model)
    if st is None:
        return dc
    row = {k: (np.nan if v is None else v) for k, v in snap["raw"].items()}
    row.update({"dc_ph": dc[0], "dc_pd": dc[1], "dc_pa": dc[2], "dc_lam": lam, "dc_mu": mu})
    for k in ("sot", "xg"):
        l0, m0 = snap.get(f"{k}_lam0"), snap.get(f"{k}_mu0")
        if l0 is None or m0 is None:
            row[f"{k}_lam"], row[f"{k}_mu"] = np.nan, np.nan
        else:
            row[f"{k}_lam"], row[f"{k}_mu"] = av.adjust_goals(l0, m0, snap["d_home"], snap["d_away"], a, b)
    return st.predict_proba(pd.DataFrame([row]))[0]


def load_stackers(settings: Settings) -> dict:
    out = {}
    for name, f in (("stacker", "stacker.pkl"), ("stacker_mercado", "stacker_market.pkl")):
        p = settings.state_dir / f
        if p.exists():
            out[name] = Stacker.load(p)
    return out


# ---------------------------------------------------------------------------
# Explicación de un cambio
# ---------------------------------------------------------------------------
def _swap(old: dict, new: dict, keys: list[str], raw_keys: list[str]) -> dict:
    s = json.loads(json.dumps(old))
    for k in keys:
        if k in new:
            s[k] = new[k]
    for k in raw_keys:
        if k in new["raw"]:
            s["raw"][k] = new["raw"][k]
    return s


def _fmt_pct(x) -> str:
    return "?" if x is None else f"{100 * x:.0f}%"


def _detail_bajas(old: dict, new: dict, home: str, away: str) -> str:
    parts = []
    for side, team in (("home", home), ("away", away)):
        oa = {a["player"]: a["status"] for a in (old.get("absences") or {}).get(team, [])}
        na = {a["player"]: a["status"] for a in (new.get("absences") or {}).get(team, [])}
        fuera = [p for p, s in na.items() if s == "out" and oa.get(p) != "out"]
        duda = [p for p, s in na.items() if s == "doubtful" and oa.get(p) != "doubtful"]
        vuelve = [p for p in oa if p not in na]
        bits = []
        if new.get("lineup_confirmed") and new.get("lineups"):
            starters = set((new["lineups"] or {}).get(side, []))
            xi = (new.get("expected_xi") or {}).get(team, [])
            fuera_xi = [p for p in xi if p not in starters]
            if fuera_xi:
                bits.append("no salen de inicio: " + ", ".join(fuera_xi[:5]))
        if fuera:
            bits.append("baja: " + ", ".join(fuera))
        if duda:
            bits.append("duda: " + ", ".join(duda))
        if vuelve:
            bits.append("recupera: " + ", ".join(vuelve))
        do, dn = old[f"d_{side}"], new[f"d_{side}"]
        if abs(dn[0] - do[0]) >= 0.02 or abs(dn[1] - do[1]) >= 0.02:
            bits.append(f"déficit ataque {_fmt_pct(do[0])}→{_fmt_pct(dn[0])}, defensa {_fmt_pct(do[1])}→{_fmt_pct(dn[1])}")
        if bits:
            parts.append(f"{team}: " + "; ".join(bits))
    if new.get("lineup_confirmed") and not old.get("lineup_confirmed"):
        parts.insert(0, "alineaciones oficiales publicadas")
    return " · ".join(parts)


def _detail_resultados(old: dict, new: dict, home: str, away: str, matches: pd.DataFrame | None,
                       t_old: str, t_new: str) -> str:
    parts = []
    if matches is not None and len(matches):
        a = pd.Timestamp(t_old).tz_convert("Europe/Madrid").tz_localize(None).normalize()
        b = pd.Timestamp(t_new).tz_convert("Europe/Madrid").tz_localize(None).normalize()
        played = matches[matches.hg.notna() & (matches.date >= a) & (matches.date <= b) &
                         (matches.home.isin([home, away]) | matches.away.isin([home, away]))]
        for r in played.sort_values("date").itertuples():
            parts.append(f"{r.home} {int(r.hg)}-{int(r.ag)} {r.away} ({r.date:%d/%m})")
    if abs(new["lam0"] - old["lam0"]) >= 0.05 or abs(new["mu0"] - old["mu0"]) >= 0.05:
        parts.append(f"goles esperados base {old['lam0']:.2f}–{old['mu0']:.2f} → {new['lam0']:.2f}–{new['mu0']:.2f}")
    eo, en = old["raw"].get("elo_diff"), new["raw"].get("elo_diff")
    if eo is not None and en is not None and abs(en - eo) >= 10:
        parts.append(f"diferencia Elo {eo:+.0f} → {en:+.0f}")
    return " · ".join(parts) or "actualización de forma y fuerzas"


_CAL_LABELS = {"euro_prev_home": "el local jugó en Europa/Copa días antes", "euro_prev_away": "el visitante jugó en Europa/Copa días antes",
               "euro_next_home": "el local juega en Europa/Copa días después", "euro_next_away": "el visitante juega en Europa/Copa días después",
               "midweek": "partido entre semana", "post_break": "vuelta de parón", "ko_late": "horario nocturno", "ko_early": "horario temprano"}


def _detail_calendario(old: dict, new: dict) -> str:
    parts = []
    for k, label in _CAL_LABELS.items():
        o, n = old["raw"].get(k), new["raw"].get(k)
        if (o or 0) != (n or 0) and n is not None:
            parts.append(("ahora " if n else "ya no: ") + label)
    for k, label in (("rest_home", "descanso local"), ("rest_away", "descanso visitante")):
        o, n = old["raw"].get(k), new["raw"].get(k)
        if o is not None and n is not None and o != n:
            parts.append(f"{label} {o:.0f}→{n:.0f} días")
    return " · ".join(parts) or "cambio de calendario"


def _detail_mercado(old: dict, new: dict) -> str:
    def odds(s, close):
        k = "odds_close" if close and s["raw"].get("odds_close_h") is not None else "odds"
        return [s["raw"].get(f"{k}_{x}") for x in "hda"]
    o, n = odds(old, True), odds(new, True)
    if all(v is not None for v in o + n):
        return f"cuotas 1/X/2 {o[0]:.2f}/{o[1]:.2f}/{o[2]:.2f} → {n[0]:.2f}/{n[1]:.2f}/{n[2]:.2f}"
    if all(v is not None for v in n):
        return f"primeras cuotas disponibles {n[0]:.2f}/{n[1]:.2f}/{n[2]:.2f}"
    return "cambio en las cuotas"


def _detail_arbitro(old: dict, new: dict) -> str:
    if new.get("referee") and new.get("referee") != old.get("referee"):
        dev = new["raw"].get("ref_home_ppg_dev")
        extra = " (sin sesgo local apreciable)" if dev is None or abs(dev) < 0.01 else f" (sesgo local histórico {dev:+.2f} pts/partido)"
        return f"designado {new['referee']}{extra}"
    return "cambio en datos del árbitro"


def _detail_clima(old: dict, new: dict) -> str:
    parts = []
    for k, label, unit in (("precip_mm", "lluvia", "mm"), ("wind_kmh", "viento", "km/h"), ("temp_c", "temperatura", "°C")):
        o, n = old["raw"].get(k), new["raw"].get(k)
        if n is not None and (o is None or abs(n - o) >= 1):
            parts.append(f"{label} {'?' if o is None else f'{o:.0f}'}→{n:.0f} {unit}")
    return " · ".join(parts) or "cambio en la previsión"


def explain(prev: dict, new: dict, stackers: dict, matches: pd.DataFrame | None = None,
            min_effect: float = 0.005) -> dict:
    """Atribuye el cambio de probabilidades entre dos registros del historial."""
    so, sn = prev["snapshot"], new["snapshot"]
    a, b = sn["a"], sn["b"]
    model = new["model"]
    p_old = np.array(prev["p"])
    p_new = np.array(new["p"])
    p_old_now = probs_from_snapshot(so, model, stackers, a, b)
    contrib = []
    d_model = p_old_now - p_old
    if np.abs(d_model).max() >= min_effect:
        why = "modelo reentrenado o recalibrado" if prev["model"] == model else f"cambio de modelo ({prev['model']} → {model})"
        contrib.append(("modelo", "Modelo", d_model, why))
    details = {
        "bajas": lambda: _detail_bajas(so, sn, new["home"], new["away"]),
        "resultados": lambda: _detail_resultados(so, sn, new["home"], new["away"], matches, prev["ts"], new["ts"]),
        "calendario": lambda: _detail_calendario(so, sn),
        "mercado": lambda: _detail_mercado(so, sn),
        "arbitro": lambda: _detail_arbitro(so, sn),
        "clima": lambda: _detail_clima(so, sn),
    }
    explained = d_model.copy()
    for key, label, top, raw in EXPL_GROUPS:
        swapped = _swap(so, sn, top, raw)
        if swapped == so:
            continue
        d = probs_from_snapshot(swapped, model, stackers, a, b) - p_old_now
        explained += d
        if np.abs(d).max() >= min_effect:
            contrib.append((key, label, d, details[key]()))
    resid = (p_new - p_old) - explained
    if np.abs(resid).max() >= min_effect:
        contrib.append(("combinado", "Efecto combinado de varios factores", resid, ""))
    delta = p_new - p_old
    focus = int(np.argmin(delta))  # la salida que más baja
    contrib.sort(key=lambda c: c[2][focus])
    return {
        "focus": OUTCOMES[focus],
        "reasons": [{"factor": label, "key": key, "effect_pts": [round(100 * x, 1) for x in d], "detail": det}
                    for key, label, d, det in contrib],
    }


# ---------------------------------------------------------------------------
# Historial
# ---------------------------------------------------------------------------
def _hist_path(settings: Settings):
    return settings.predictions_dir / "history.jsonl"


def _read_jsonl(path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def load_history(settings: Settings) -> pd.DataFrame:
    rows = _read_jsonl(_hist_path(settings))
    if not rows:
        return pd.DataFrame(columns=["ts", "season", "home", "away", "p1", "pX", "p2", "lineup_confirmed"])
    return pd.DataFrame([{"ts": r["ts"], "season": r["season"], "home": r["home"], "away": r["away"],
                          "p1": r["p"][0], "pX": r["p"][1], "p2": r["p"][2],
                          "lineup_confirmed": r["snapshot"].get("lineup_confirmed", False) if r.get("snapshot") else False}
                         for r in rows])


def record(settings: Settings, preds: list[dict], snaps: list[dict | None],
           matches: pd.DataFrame | None = None) -> dict:
    """Añade al historial y devuelve {(local, visitante): movimiento} para esta ejecución."""
    thr = float(settings.get("movements", "threshold_pts", 4)) / 100
    min_store = float(settings.get("movements", "min_store_pts", 0.5)) / 100
    hp = _hist_path(settings)
    last: dict = {}
    for r in _read_jsonl(hp):
        last[(r["season"], r["home"], r["away"])] = r
    stackers = None
    moves: dict = {}
    new_lines, move_lines = [], []
    for p, snap in zip(preds, snaps):
        key = (p["season"], p["home"], p["away"])
        rec = {"ts": p["generated_at"], "season": p["season"], "date": p["date"], "home": p["home"], "away": p["away"],
               "p": [p["p_home"], p["p_draw"], p["p_away"]], "model": p["model"], "snapshot": snap}
        prev = last.get(key)
        if prev is not None:
            delta = np.array(rec["p"]) - np.array(prev["p"])
            same_info = (snap is None or prev.get("snapshot") is None or
                         (snap.get("lineup_confirmed") == prev["snapshot"].get("lineup_confirmed")
                          and snap.get("absences") == prev["snapshot"].get("absences")))
            if np.abs(delta).max() < min_store and same_info:
                continue  # sin cambios relevantes: no se duplica la línea
            mv = {"home": p["home"], "away": p["away"], "date": p["date"], "from_ts": prev["ts"], "to_ts": rec["ts"],
                  "p_from": prev["p"], "p_to": rec["p"], "delta_pts": [round(100 * x, 1) for x in delta],
                  "important": bool(np.abs(delta).max() >= thr)}
            if mv["important"] and snap is not None and prev.get("snapshot") is not None:
                if stackers is None:
                    stackers = load_stackers(settings)
                try:
                    mv.update(explain(prev, rec, stackers, matches))
                except Exception as e:  # una explicación fallida no debe parar la predicción
                    log.warning("No se pudo explicar el cambio de %s-%s: %s", p["home"], p["away"], e)
                move_lines.append(mv)
            moves[(p["home"], p["away"])] = mv
        new_lines.append(rec)
    if new_lines:
        with hp.open("a", encoding="utf-8") as f:
            for r in new_lines:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if move_lines:
        with (settings.predictions_dir / "movements.jsonl").open("a", encoding="utf-8") as f:
            for m in move_lines:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
    return moves


# ---------------------------------------------------------------------------
# Presentación
# ---------------------------------------------------------------------------
_SPARK = "▁▂▃▄▅▆▇█"


def sparkline(values: list[float]) -> str:
    if not values:
        return ""
    lo, hi = 0.0, 1.0
    return "".join(_SPARK[min(int((v - lo) / (hi - lo) * len(_SPARK)), len(_SPARK) - 1)] for v in values)


def trend_text(history: pd.DataFrame, p: dict, col: str = "p1") -> str:
    h = history[(history.home == p["home"]) & (history.away == p["away"]) & (history.season == p["season"])]
    vals = list(h.sort_values("ts")[col])
    cur = {"p1": p["p_home"], "pX": p["p_draw"], "p2": p["p_away"]}[col]
    if not vals or abs(vals[-1] - cur) > 1e-9:
        vals.append(cur)
    if len(vals) < 2:
        return "—"
    arrow = "▲" if vals[-1] > vals[0] + 0.005 else "▼" if vals[-1] < vals[0] - 0.005 else "="
    return f"{sparkline(vals[-12:])} {100 * vals[0]:.0f}→{100 * vals[-1]:.0f}% {arrow}"


def format_move(m: dict) -> str:
    """Texto corto de un cambio importante (para informe y Telegram)."""
    i = OUTCOMES.index(m.get("focus", "1"))
    name = {"1": f"gana {m['home']}", "X": "empate", "2": f"gana {m['away']}"}[OUTCOMES[i]]
    s = f"{m['home']} – {m['away']}: {name} {100 * m['p_from'][i]:.0f}% → {100 * m['p_to'][i]:.0f}%"
    for r in m.get("reasons", [])[:4]:
        eff = r["effect_pts"][i]
        if abs(eff) < 0.5:
            continue
        s += f"\n  {eff:+.1f} pts · {r['factor']}" + (f": {r['detail']}" if r.get("detail") else "")
    return s


def important_moves(settings: Settings, preds: list[dict]) -> list[dict]:
    keys = {(p["home"], p["away"], p["date"]) for p in preds}
    return [m for m in _read_jsonl(settings.predictions_dir / "movements.jsonl")
            if (m["home"], m["away"], m["date"]) in keys]


def render_evolution(settings: Settings, preds: list[dict]) -> str:
    hist = load_history(settings)
    moves = important_moves(settings, preds)
    lines = ["# Evolución de las probabilidades", "",
             f"Actualizado: {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC. Umbral de cambio importante: "
             f"{settings.get('movements', 'threshold_pts', 4)} puntos.", ""]
    for p in preds:
        h = hist[(hist.home == p["home"]) & (hist.away == p["away"]) & (hist.season == p["season"])].sort_values("ts")
        lines += [f"## {p['home']} – {p['away']} ({p['date']})", ""]
        if h.empty:
            lines += ["Sin historial todavía.", ""]
            continue
        lines += [f"1 {sparkline(list(h.p1))} · X {sparkline(list(h.pX))} · 2 {sparkline(list(h.p2))}", "",
                  "| Momento (Madrid) | 1 | X | 2 | Alineación |", "|---|---|---|---|---|"]
        for r in h.itertuples():
            t = pd.Timestamp(r.ts).tz_convert("Europe/Madrid").strftime("%d/%m %H:%M")
            lines.append(f"| {t} | {100 * r.p1:.0f}% | {100 * r.pX:.0f}% | {100 * r.p2:.0f}% | {'✅' if r.lineup_confirmed else ''} |")
        mv = [m for m in moves if m["home"] == p["home"] and m["away"] == p["away"]]
        if mv:
            lines += ["", "**Cambios importantes**", ""]
            for m in mv:
                t = pd.Timestamp(m["to_ts"]).tz_convert("Europe/Madrid").strftime("%d/%m %H:%M")
                lines += [f"- {t} — " + format_move(m).replace("\n  ", "\n  - ")]
        lines.append("")
    return "\n".join(lines) + "\n"
