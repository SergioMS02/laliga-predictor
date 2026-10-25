"""Avisos por Telegram (gratis): crea un bot con @BotFather y pon TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID."""
from __future__ import annotations

import logging

import pandas as pd

from .config import Settings
from .data import http

log = logging.getLogger(__name__)


def upcoming_within(preds: list[dict], hours: int) -> list[dict]:
    now = pd.Timestamp.now(tz="UTC")
    out = []
    for p in preds:
        t = pd.Timestamp(p["kickoff_utc"]) if p.get("kickoff_utc") else pd.Timestamp(p["date"], tz="Europe/Madrid")
        if t <= now + pd.Timedelta(hours=hours):
            out.append(p)
    return out


def format_prediction(p: dict) -> str:
    s = (f"{p['home']} – {p['away']}\n"
         f"1 {100 * p['p_home']:.0f}% · X {100 * p['p_draw']:.0f}% · 2 {100 * p['p_away']:.0f}%\n"
         f"xG {p['dc']['xg_home']:.2f}–{p['dc']['xg_away']:.2f} · {p['dc']['top_scores'][0][0]}")
    if p.get("lineup_confirmed"):
        s += "\n✅ con alineaciones oficiales"
    return s


def send(settings: Settings, text: str) -> bool:
    """Envía si notify.telegram es true, o si es "auto" y existen las dos variables de entorno."""
    mode = settings.get("notify", "telegram", "auto")
    token, chat = settings.env("TELEGRAM_BOT_TOKEN"), settings.env("TELEGRAM_CHAT_ID")
    if mode is False or (mode == "auto" and not (token and chat)):
        return False
    if not token or not chat:
        log.warning("Telegram activado pero faltan TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID")
        return False
    try:
        http.post_json(f"https://api.telegram.org/bot{token}/sendMessage",
                       {"chat_id": chat, "text": text[:4000], "disable_web_page_preview": True}, timeout=20)
        return True
    except Exception as e:
        log.warning("Telegram falló: %s", e)
        return False
