"""Genera site/data.json a partir de predictions/ y data/processed/ (solo biblioteca estándar).

Uso: python site/build.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "predictions"
OUT = ROOT / "site" / "data.json"


def read_json(p: Path, default):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def threshold() -> float:
    try:
        for line in (ROOT / "config.yaml").read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("threshold_pts:"):
                return float(line.split(":")[1].split("#")[0])
    except OSError:
        pass
    return 4.0


def main() -> None:
    preds = read_json(PRED / "latest.json", [])
    history = read_jsonl(PRED / "history.jsonl")
    moves = read_jsonl(PRED / "movements.jsonl")
    hist_by = {}
    for h in history:
        hist_by.setdefault((h["season"], h["home"], h["away"]), []).append(
            {"ts": h["ts"], "p": h["p"], "lineup": bool((h.get("snapshot") or {}).get("lineup_confirmed"))})
    matches = []
    for p in preds:
        key = (p["season"], p["home"], p["away"])
        hs = sorted(hist_by.get(key, []), key=lambda x: x["ts"])
        if not hs or abs(hs[-1]["p"][0] - p["p_home"]) > 1e-9:
            hs.append({"ts": p["generated_at"], "p": [p["p_home"], p["p_draw"], p["p_away"]],
                       "lineup": bool(p.get("lineup_confirmed"))})
        mv = [m for m in moves if m["home"] == p["home"] and m["away"] == p["away"] and m["date"] == p["date"]]
        q = {k: v for k, v in p.items() if not k.startswith("_")}
        q["history"] = hs
        q["moves"] = mv
        matches.append(q)
    keys = {(m["home"], m["away"], m["date"]) for m in matches}
    recent = sorted([m for m in moves if (m["home"], m["away"], m["date"]) in keys],
                    key=lambda m: m["to_ts"], reverse=True)[:20]
    bt = read_json(ROOT / "data" / "processed" / "backtest.json", {})
    data = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="minutes"),
        "generated_at": preds[0]["generated_at"] if preds else None,
        "threshold": threshold(),
        "matches": matches,
        "recent_moves": recent,
        "backtest": {k: bt.get(k) for k in ("overall", "market_subset", "calibration", "test_seasons", "by_season")},
        "live": read_json(PRED / "live_eval.json", {}),
        "extra": read_json(PRED / "site_extra.json", {}),
        "sources": read_json(PRED / "sources.json", {}),
        "bets": read_json(PRED / "bets_summary.json", {}),
        "bets_backtest": read_json(PRED / "betting_backtest.json", {}),
    }
    OUT.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"{OUT}: {len(matches)} partidos, {len(recent)} cambios importantes")


if __name__ == "__main__":
    main()
