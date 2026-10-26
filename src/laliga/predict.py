"""Predicción de los próximos partidos con toda la información disponible en este momento."""
from __future__ import annotations

import io
import json
import logging
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from . import movements, pipeline
from .config import Settings
from .data import espn, fantasy, http, news, results, suspensions
from .features import availability as av
from .features.sequential import build_sequential_features
from .models.dixon_coles import DixonColes, summarize_matrix
from .models.market import devig_power
from .models.stacker import Stacker, has_market

log = logging.getLogger(__name__)
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"


def upcoming_odds(settings: Settings) -> pd.DataFrame:
    """Cuotas de los próximos partidos (football-data.co.uk/fixtures.csv, gratis; caché de load_matches)."""
    df = results.fetch_fixtures(settings, online=False)
    if df.empty:
        df = results.fetch_fixtures(settings, online=True)
    if df.empty:
        return pd.DataFrame()
    cols = ["season", "home", "away", "odds_h", "odds_d", "odds_a"] + [
        c for c in ("b365_h", "b365_d", "b365_a", "b365_o25", "b365_u25", "ou_o25", "ou_u25",
                    "max_h", "max_d", "max_a", "avg_h", "avg_d", "avg_a", "ah_line", "ah_h", "ah_a") if c in df]
    return df[cols].dropna(subset=["odds_h"])


def record_odds_snapshot(settings: Settings, odds: pd.DataFrame, now: pd.Timestamp) -> None:
    """Guarda cada lectura de cuotas para medir cómo se mueven hasta el partido."""
    p = settings.processed_dir / "odds_snapshots.csv"
    snap = odds.assign(ts=now.isoformat())
    snap.to_csv(p, mode="a", header=not p.exists(), index=False)


def odds_movement(settings: Settings) -> pd.DataFrame:
    """Primera cuota vista (como 'apertura') y última (como 'cierre provisional')."""
    p = settings.processed_dir / "odds_snapshots.csv"
    if not p.exists():
        return pd.DataFrame()
    s = pd.read_csv(p).dropna(subset=["odds_h", "odds_d", "odds_a"]).sort_values("ts")
    if s.empty:
        return pd.DataFrame()
    key = ["season", "home", "away"]
    first = s.groupby(key)[["odds_h", "odds_d", "odds_a"]].first()
    last = s.groupby(key)[["odds_h", "odds_d", "odds_a"]].last()
    last.columns = ["odds_close_h", "odds_close_d", "odds_close_a"]
    return first.join(last).reset_index()


def select_upcoming(matches: pd.DataFrame, now: pd.Timestamp, days: int) -> pd.DataFrame:
    today = now.tz_convert("Europe/Madrid").tz_localize(None).normalize()
    up = matches[matches.hg.isna() & (matches.date >= today) & (matches.date < today + pd.Timedelta(days=days))]
    if "kickoff" in up:
        ko = pd.to_datetime(up.kickoff, utc=True)
        up = up[ko.isna() | (ko > now - pd.Timedelta(minutes=10))]  # descarta los ya empezados
    return up


def _event_map(sb: pd.DataFrame) -> dict[tuple[str, str], str]:
    if sb is None or sb.empty:
        return {}
    return {(r.home, r.away): r.event_id for r in sb.itertuples()}


def _load_stackers(settings: Settings) -> tuple[Stacker | None, Stacker | None]:
    p, pm = settings.state_dir / "stacker.pkl", settings.state_dir / "stacker_market.pkl"
    return (Stacker.load(p) if p.exists() else None, Stacker.load(pm) if pm.exists() else None)


def predict(settings: Settings, matches: pd.DataFrame, sb: pd.DataFrame | None = None,
            now: pd.Timestamp | None = None, days: int | None = None, online: bool = True,
            only_within_minutes: int | None = None) -> list[dict]:
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    days = days or int(settings.get("predict", "days_ahead", 8))
    matches = matches.reset_index(drop=True)
    up = select_upcoming(matches, now, days)
    if only_within_minutes is not None:
        ko = pd.to_datetime(up.kickoff, utc=True)
        up = up[ko.notna() & (ko <= now + pd.Timedelta(minutes=only_within_minutes))]
    if up.empty:
        return []
    events = _event_map(sb)

    # 1) Detalle ESPN: árbitro y alineaciones (si ya se publicaron)
    details: dict[int, espn.MatchDetail] = {}
    if online and settings.get("espn", "enabled", True):
        for idx, r in up.iterrows():
            ev = events.get((r.home, r.away))
            if not ev:
                continue
            try:
                details[idx] = espn.summary(settings, ev)
            except Exception as e:
                log.warning("ESPN summary %s falló: %s", ev, e)
        if details:
            refs = pd.DataFrame([{"season": up.loc[i, "season"], "home": up.loc[i, "home"], "away": up.loc[i, "away"],
                                  "referee": d.referee} for i, d in details.items() if d.referee])
            if len(refs):
                espn.save_referees(settings, refs)
                matches = pipeline.apply_referees(settings, matches)

    # 2) Variables secuenciales y de contexto (Elo, forma, descanso, viaje, árbitro,
    #    clasificación, h2h, Europa/Copa, entrenador...)
    seq = build_sequential_features(matches, settings["elo"])
    ctx = pipeline.context_features(settings, matches)
    tab = up.drop(columns=["referee"]).join(matches[["referee"]]).join(seq).join(ctx)

    # 3) Dixon-Coles (goles) + modelos de fuerza sobre tiros a puerta y xG
    today = now.tz_convert("Europe/Madrid").tz_localize(None).normalize()
    ref = today + pd.Timedelta(days=1)
    season_now = results.current_season(today.date())
    new = pipeline._new_teams(matches, season_now)
    teams = sorted(set(up.home) | set(up.away))
    dc = DixonColes.from_settings(settings).fit(matches, ref, teams, new)
    count_models = {}
    for k, cols in pipeline.COUNT_MODELS.items():
        if not set(cols) <= set(matches.columns):
            continue
        cm = DixonColes.from_settings(settings)
        has = matches[list(cols)].notna().all(axis=1) & (matches.date >= today - pd.Timedelta(days=cm.window_days))
        if has.sum() >= pipeline.MIN_SOT_MATCHES:
            count_models[k] = cm.fit(matches, ref, teams, new, cols=cols)

    # 4) Bajas: alineación confirmada > noticias. Déficit separado en ataque y defensa.
    lineups_hist = espn.load_lineups(settings)
    players = pipeline.load_extras(settings, online=False).get("players")
    a_coef = float(settings.get("availability", "attack_coef", 0.6))
    b_coef = float(settings.get("availability", "defence_coef", 0.35))
    lookback = int(settings.get("availability", "lookback_matches", 10))
    dw = float(settings.get("availability", "doubtful_weight", 0.5))
    use_news = online and settings.get("news", "enabled", True)
    cards = None
    fantasy_av = None
    if use_news:
        try:
            fantasy_av = fantasy.fetch(settings, online=online)
            pipeline._status("futbolfantasy (lesionados y sancionados)", fantasy_av is not None,
                             0 if fantasy_av is None else len(fantasy_av))
        except Exception as e:
            log.warning("futbolfantasy no disponible: %s", e)
        try:
            cards = suspensions.fetch_cards(settings, date(season_now, 8, 1), online=online)
        except Exception as e:
            log.warning("Tarjetas no disponibles: %s", e)
    out_rows = []
    new_lineups = []
    n_news = 0
    for idx, r in tab.iterrows():
        info = {"lineup_confirmed": False, "absences": {}, "d_home": (0.0, 0.0), "d_away": (0.0, 0.0),
                "availability_source": "ninguna"}
        det = details.get(idx)
        imps = {}
        for side, team in (("home", r.home), ("away", r.away)):
            imp = av.player_importance(lineups_hist, team, r.date, lookback)
            pl = None
            if players is not None and len(players):
                pl = players[(players.team == team) & players.season.isin([season_now, season_now - 1])]
            imps[side] = av.attach_production(imp, pl)
        squads = {}
        if det is not None and det.lineup_confirmed:
            info["lineup_confirmed"] = True
            info["availability_source"] = "alineación oficial"
            for side in ("home", "away"):
                starters = [p.player_id for p in det.lineups[side] if p.starter]
                info[f"d_{side}"] = av.deficits_from_lineup(imps[side], starters)
                info.setdefault("lineups", {})[side] = [p.name for p in det.lineups[side] if p.starter]
                squads[side] = av.squad_view(imps[side], [p for p in det.lineups[side] if p.starter],
                                             det.formations.get(side))
            new_lineups.append(espn.lineup_rows(det, r.date, r.home, r.away))
        elif use_news:
            for side, team in (("home", r.home), ("away", r.away)):
                squad = list(imps[side].name)
                if players is not None and len(players):  # también los que aún no tienen historial de titularidades
                    extra = players[(players.team == team) & (players.season == season_now)].player
                    squad += [x for x in extra if x not in squad]
                ko = pd.Timestamp(r.kickoff) if not pd.isna(r.get("kickoff")) else pd.Timestamp(r.date)
                ko = ko.tz_convert(None) if ko.tzinfo else ko
                items = []
                if fantasy_av is not None:
                    # fuente principal: lesionados y sancionados de futbolfantasy.com
                    ab = _match_fantasy(fantasy.for_match(fantasy_av, matches, team, season_now, ko), squad)
                    jn = fantasy.team_jornada(matches, team, season_now, ko)
                    pj = fantasy.page_jornada(fantasy_av)
                    need_cards = jn is not None and pj is not None and jn < pj  # jornada anterior a la de la página
                else:
                    items = news.fetch_team_news(settings, team)
                    ab = news.extract_absences(settings, team, items, squad)
                    need_cards = True
                try:  # respaldo: sanciones por tarjetas (y noticias de sanciones si no hay futbolfantasy)
                    if need_cards:
                        sus = suspensions.for_match(settings, team, squad, matches, cards, ko,
                                                    online and fantasy_av is None)
                        if len(sus):
                            ab = pd.concat([ab[~ab.player.isin(sus.player)], sus], ignore_index=True)
                except Exception as e:
                    log.warning("Sanciones no disponibles para %s: %s", team, e)
                cols = [c for c in ("player", "status", "reason", "source") if c in ab.columns]
                info["absences"][team] = ab[cols].astype(object).where(ab[cols].notna(), None).to_dict(orient="records")
                info[f"d_{side}"] = av.deficits_from_absences(imps[side], ab, dw)
                info.setdefault("news_count", {})[team] = len(items)
                n_news += len(items)
            if any(info["absences"].values()):
                info["availability_source"] = "lesionados y sancionados" if fantasy_av is not None else "noticias"
        hints = {}
        for side, team in (("home", r.home), ("away", r.away)):
            hints[side] = {}
            if players is not None and len(players):
                for x in players[players.team == team].sort_values("season").itertuples():
                    hints[side][x.player] = {"GK": "G", "D": "D", "M": "M", "F": "F"}.get(str(x.position or "").split(" ")[0], "M")
        for side, team in (("home", r.home), ("away", r.away)):
            if side not in squads:
                sq = av.squad_view(imps[side], None, None, info["absences"].get(team), hints[side])
                if not sq["xi"] and players is not None and len(players):
                    pl = players[(players.team == team) & (players.season == season_now)]
                    if pl.minutes.sum() < 900:
                        pl = players[(players.team == team) & (players.season == season_now - 1)]
                    sq = av.squad_from_players(pl, info["absences"].get(team))
                squads[side] = sq
        rost = {}
        for side, team in (("home", r.home), ("away", r.away)):
            pl_t = players[(players.team == team) & (players.season == season_now)] if players is not None and len(players) else None
            starters = info.get("lineups", {}).get(side) if info["lineup_confirmed"] else None
            try:
                rost[side] = av.roster(imps[side], pl_t, info["absences"].get(team), starters, hints[side])
            except Exception as e:
                log.warning("Plantilla no disponible para %s: %s", team, e)
                rost[side] = []
        info["roster"] = rost
        for side in ("home", "away"):  # portero suplente: el disponible con más minutos según la plantilla
            for pl in (squads.get(side) or {}).get("xi", []):
                if pl.get("name") == "Portero suplente":
                    gks = [x for x in rost.get(side, []) if x["role"] == "G" and x["status"] in ("disponible", "suplente")
                           and x["name"] != pl.get("replaces")]
                    if gks:
                        pl["name"] = max(gks, key=lambda x: ((x.get("minutes") or 0), x.get("imp") or 0))["name"]
        info["squads"] = squads
        lam0, mu0 = dc.expected_goals(r.home, r.away)
        lam, mu = av.adjust_goals(lam0, mu0, info["d_home"], info["d_away"], a_coef, b_coef)
        dcp = summarize_matrix(dc.score_matrix(lam, mu), lam, mu)
        extra, extra_base = {}, {}
        for k in pipeline.COUNT_MODELS:
            if k in count_models:
                l_, m_ = count_models[k].expected_goals(r.home, r.away)
                extra_base[k] = (l_, m_)
                extra[k] = av.adjust_goals(l_, m_, info["d_home"], info["d_away"], a_coef, b_coef)
            else:
                extra[k] = extra_base[k] = (np.nan, np.nan)
        expected_xi = {team: list(imps[side].sort_values("importance", ascending=False).head(11).name)
                       for side, team in (("home", r.home), ("away", r.away))}
        out_rows.append({"idx": idx, "dc": dcp, "lam0": lam0, "mu0": mu0, "extra": extra,
                         "extra_base": extra_base, "expected_xi": expected_xi, **info})
    if new_lineups:
        espn.save_lineups(settings, pd.concat(new_lineups, ignore_index=True))
    if online:
        pipeline._status("ESPN (alineaciones y árbitro del partido)", len(details) > 0, len(details))
        if use_news:
            pipeline._status("Google News (bajas)", n_news > 0, n_news)

    for o in out_rows:
        i = o["idx"]
        vals = [o["dc"]["p_home"], o["dc"]["p_draw"], o["dc"]["p_away"], o["dc"]["xg_home"], o["dc"]["xg_away"]]
        for k in pipeline.COUNT_MODELS:
            vals += list(o["extra"][k])
        tab.loc[i, pipeline.DC_COLS] = vals

    # 5) Mercado (y su movimiento desde la primera cuota vista) y clima
    odds = upcoming_odds(settings) if online else pd.DataFrame()
    b365, b365_ou, mkt_ou, best_odds, ah_odds = {}, {}, {}, {}, {}
    from . import markets as markets_mod
    ht_sh = markets_mod.ht_share(matches)
    if len(odds):
        if "ah_line" in odds:
            ah_odds = {(r.season, r.home, r.away): (r.ah_line, r.ah_h, r.ah_a) for r in odds.itertuples()
                       if not pd.isna(r.ah_line) and not pd.isna(r.ah_h)}
        if "b365_h" in odds:
            b365 = {(r.season, r.home, r.away): [r.b365_h, r.b365_d, r.b365_a] for r in odds.itertuples()
                    if not pd.isna(r.b365_h)}
        if "b365_o25" in odds:
            b365_ou = {(r.season, r.home, r.away): [r.b365_o25, r.b365_u25] for r in odds.itertuples()
                       if not pd.isna(r.b365_o25) and not pd.isna(r.b365_u25)}
        if "max_h" in odds:
            best_odds = {(r.season, r.home, r.away): {"max": [r.max_h, r.max_d, r.max_a], "avg": [r.avg_h, r.avg_d, r.avg_a]}
                         for r in odds.itertuples() if not pd.isna(r.max_h)}
        if "ou_o25" in odds:
            from .betting import devig2
            mkt_ou = {(r.season, r.home, r.away): devig2(r.ou_o25, r.ou_u25) for r in odds.itertuples()}
        record_odds_snapshot(settings, odds[["season", "home", "away", "odds_h", "odds_d", "odds_a"]], now)
    from .data import polymarket as pmk
    poly_cfg = pmk.cfg(settings)
    poly_mkts = pmk.fetch_matches(settings, online)
    poly_w = pmk.model_weight(settings)
    if online and poly_cfg["enabled"]:
        pipeline._status("Polymarket (precios de partidos)", len(poly_mkts) > 0, len(poly_mkts))
    snaps = odds_movement(settings)
    if len(snaps):
        tab = tab.drop(columns=["odds_h", "odds_d", "odds_a", "odds_close_h", "odds_close_d", "odds_close_a"],
                       errors="ignore").reset_index().merge(snaps, on=["season", "home", "away"], how="left").set_index("index")
    for c in ("odds_h", "odds_d", "odds_a", "odds_close_h", "odds_close_d", "odds_close_a"):
        if c not in tab:
            tab[c] = np.nan
    tab = pipeline.add_market(tab)
    if online:
        tab = pipeline.attach_weather(settings, tab.reset_index(), archive=False).set_index("index")
    else:
        for c in ("temp_c", "precip_mm", "wind_kmh"):
            tab[c] = np.nan

    # 6) Meta-modelo
    st, stm = _load_stackers(settings)
    final = tab[["dc_ph", "dc_pd", "dc_pa"]].to_numpy(dtype=float).copy()
    model_used = np.array(["dixon_coles"] * len(tab), dtype=object)
    if st is not None:
        final = st.predict_proba(tab)
        model_used[:] = "stacker"
    if stm is not None:
        hm = has_market(tab).to_numpy()
        if hm.any():
            final[hm] = stm.predict_proba(tab[hm])
            model_used[hm] = "stacker_mercado"

    # 6b) Tarjetas y córners
    from . import props as props_mod
    try:
        props_model = props_mod.PropsModel(matches, ref)
    except Exception as e:
        log.warning("Modelo de tarjetas y córners no disponible: %s", e)
        props_model = None

    # 7) Salida
    from .betting import blend_ou, load_ou_blend
    ou_w = load_ou_blend(settings)
    gen = now.isoformat()
    preds = []
    by_idx = {o["idx"]: o for o in out_rows}
    for k, (idx, r) in enumerate(tab.iterrows()):
        o = by_idx[idx]
        ko = r.get("kickoff")
        impact = _impact(dc, o)
        if model_used[k] == "stacker_mercado" and impact:
            # Con cuotas, el meta-modelo se apoya sobre todo en el mercado, cuyas cuotas pueden ser de
            # hace días (antes de la alineación o de la última noticia). Se aplica el efecto de bajas y
            # alineaciones del modelo de goles: completo con alineación oficial, a la mitad con noticias
            # (lo más probable es que el mercado ya las conozca en parte).
            w_l = float(settings.get("availability", "market_weight_lineup", 1.0))  # medido con alineaciones históricas
            w = w_l if o["lineup_confirmed"] else 0.5 * w_l
            ratio = (np.asarray(impact["p_adj"]) / np.clip(np.asarray(impact["p_full"]), 1e-6, None)) ** w
            f = final[k] * ratio
            final[k] = f / f.sum()
            impact["market_weight"] = round(w, 3)
        p_model = final[k].copy()
        pm_m = pmk.for_match(poly_mkts, r.home, r.away, ko if not pd.isna(ko) else r.date)
        pm_ok, pm_why = pmk.usable(pm_m, poly_cfg, now)
        poly = None
        if pm_m:
            poly = {k2: pm_m[k2] for k2 in ("p", "volume", "liquidity", "url", "slug", "overround")}
            poly.update(used=pm_ok, reason=pm_why, model_weight=round(poly_w, 2) if pm_ok else None)
            if online and not pm_m.get("live") and not pm_m.get("closed"):
                try:
                    poly["more"] = pmk.fetch_secondary(settings, pm_m, online)
                except Exception as e:
                    log.warning("Polymarket secundarios: %s", e)
            if pm_ok:
                final[k] = pmk.blend(p_model, pm_m["p"], poly_w)
        preds.append({
            "generated_at": gen,
            "season": int(r.season), "date": str(pd.Timestamp(r.date).date()),
            "kickoff_utc": None if pd.isna(ko) else pd.Timestamp(ko).isoformat(),
            "home": r.home, "away": r.away,
            "p_home": round(float(final[k, 0]), 4), "p_draw": round(float(final[k, 1]), 4),
            "p_away": round(float(final[k, 2]), 4), "model": model_used[k],
            "p_model": [round(float(x), 4) for x in p_model],
            "polymarket": poly,
            "dc": {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in o["dc"].items()},
            "xg_base": [round(o["lam0"], 3), round(o["mu0"], 3)],
            "elo_diff": round(float(r.elo_diff), 1),
            "referee": r.referee if isinstance(r.referee, str) else None,
            "weather": {c: (None if pd.isna(r.get(c)) else round(float(r.get(c)), 1))
                        for c in ("temp_c", "precip_mm", "wind_kmh")},
            "market": None if pd.isna(r.mkt_ph) else [round(float(r.mkt_ph), 4), round(float(r.mkt_pd), 4),
                                                       round(float(r.mkt_pa), 4)],
            "lineup_confirmed": o["lineup_confirmed"],
            "availability_source": o["availability_source"],
            "deficit": [round(o["d_home"][0], 3), round(o["d_away"][0], 3)],
            "deficit_defence": [round(o["d_home"][1], 3), round(o["d_away"][1], 3)],
            "context": {k: (None if pd.isna(r.get(k)) else round(float(r.get(k)), 3)) for k in (
                "stake_diff_late", "h2h_gd", "derby", "euro_prev_home", "euro_prev_away", "euro_next_home",
                "euro_next_away", "new_coach_home", "new_coach_away", "red_prev_home", "red_prev_away", "post_break")
                if k in r.index},
            "absences": o["absences"],
            "lineups": o.get("lineups"),
            "squads": o.get("squads"),
            "b365": [float(x) for x in b365[(r.season, r.home, r.away)]] if (r.season, r.home, r.away) in b365 else None,
            "b365_ou": [float(x) for x in b365_ou[(r.season, r.home, r.away)]] if (r.season, r.home, r.away) in b365_ou else None,
            "ou": _ou_block(o["dc"], _poly_ou(poly) or mkt_ou.get((r.season, r.home, r.away)), ou_w, blend_ou,
                            "Polymarket" if _poly_ou(poly) else ("casas" if mkt_ou.get((r.season, r.home, r.away)) else None)),
            "odds_market": _clean_odds(best_odds.get((r.season, r.home, r.away))),
            "markets": _markets(final[k], o["dc"], dc.params.rho, ah_odds.get((r.season, r.home, r.away)), ht_sh),
            "espn_id": str(events[(r.home, r.away)]) if (r.home, r.away) in events else None,
            "scorers": _scorers(o, players, r, season_now),
            "roster": o.get("roster"),
            "props": _props(props_model, r, o, cards, lineups_hist, season_now),
            "absence_impact": impact,
            "_snapshot": movements.make_snapshot(
                r, o, rho=dc.params.rho, a=a_coef, b=b_coef, max_goals=dc.max_goals),
        })
        preds[-1]["_snapshot"]["poly"] = {"p": pm_m["p"], "w": round(poly_w, 3)} if pm_ok else None
    preds.sort(key=lambda d: (d["kickoff_utc"] or d["date"], d["home"]))
    return preds


def _markets(p_final, dcp: dict, rho: float, ah: tuple | None, share: float) -> dict | None:
    """Mercados derivados, con goles esperados coherentes con la probabilidad final del 1X2."""
    try:
        from . import markets as markets_mod
        from .models.dixon_coles import score_matrix
        tot = float(dcp["xg_home"]) + float(dcp["xg_away"])
        lam, mu = xg_for_probs([float(x) for x in p_final], tot)
        out = markets_mod.from_matrix(score_matrix(lam, mu, rho, 10), lam, mu, rho, share)
        out["goals"] = [lam, mu]
        if ah:
            out["ah_market"] = markets_mod.ah_value(out, *ah)
        return out
    except Exception as e:
        log.warning("Mercados derivados: %s", e)
        return None


def _props(pm, r, o: dict, cards, lineups, season: int) -> dict | None:
    if pm is None or not getattr(pm, "ok", False):
        return None
    try:
        from . import props as props_mod
        from .teams import is_derby
        ref = r.referee if isinstance(r.referee, str) else None
        out = pm.predict(r.home, r.away, ref, derby=is_derby(r.home, r.away))
        if out is None:
            return None
        start = pd.Timestamp(f"{season}-08-01")
        out["players"] = {}
        for k, (side, team) in enumerate((("home", r.home), ("away", r.away))):
            xi = [p["name"] for p in ((o.get("squads") or {}).get(side) or {}).get("xi", []) if p.get("absent") != "out"]
            avg = pm.avg["hy"] if side == "home" else pm.avg["ay"]
            out["players"][side] = props_mod.player_card_probs(cards, lineups, team, start, xi, out["yellow"][k], avg)
        return out
    except Exception as e:
        log.warning("Tarjetas y córners no disponibles para %s-%s: %s", r.home, r.away, e)
        return None


def _match_fantasy(ab: pd.DataFrame, squad: list[str]) -> pd.DataFrame:
    """Ajusta los nombres de futbolfantasy a los de la plantilla (sin acentos, apellidos compuestos)."""
    from rapidfuzz import fuzz, process
    if ab.empty or not squad:
        return ab
    plain_sq = [news._plain(x) for x in squad]
    rows = []
    for r in ab.to_dict(orient="records"):
        best = process.extractOne(news._plain(r["player"]), plain_sq, scorer=fuzz.token_set_ratio)
        if best and best[1] >= 80:
            r["player"] = squad[best[2]]
        rows.append(r)  # aunque no esté en la plantilla conocida, se muestra igualmente
    return pd.DataFrame(rows, columns=ab.columns)


def _impact(dc, o: dict) -> dict | None:
    """Cómo cambian goles esperados y 1X2 del modelo de goles por las bajas y alineaciones."""
    try:
        base = summarize_matrix(dc.score_matrix(o["lam0"], o["mu0"]), o["lam0"], o["mu0"])
        adj = o["dc"]
        return {"xg_full": [round(o["lam0"], 2), round(o["mu0"], 2)],
                "xg_adj": [round(adj["xg_home"], 2), round(adj["xg_away"], 2)],
                "p_full": [round(base["p_home"], 4), round(base["p_draw"], 4), round(base["p_away"], 4)],
                "p_adj": [round(adj["p_home"], 4), round(adj["p_draw"], 4), round(adj["p_away"], 4)]}
    except Exception:
        return None


def _clean_odds(d: dict | None) -> dict | None:
    if not d:
        return None
    out = {k: [None if pd.isna(x) else round(float(x), 2) for x in v] for k, v in d.items()}
    return out if any(x is not None for v in out.values() for x in v) else None


def _scorers(o: dict, players, r, season: int) -> dict:
    from .scorers import scorer_probs
    out = {}
    try:
        for side, team, goals in (("home", r.home, o["dc"]["xg_home"]), ("away", r.away, o["dc"]["xg_away"])):
            xi = ((o.get("squads") or {}).get(side) or {}).get("xi") or []
            out[side] = scorer_probs(xi, float(goals), players, team, season)
    except Exception as e:  # nunca debe romper la predicción
        log.warning("Goleadores no disponibles: %s", e)
    return out


def _poly_ou(poly: dict | None, min_volume: float = 5000) -> float | None:
    """Más de 2,5 en Polymarket si el contrato tiene volumen: medido mejor que el cierre de las casas
    (log-loss 0,6605 frente a 0,6619 en 332 partidos) y que el modelo (0,6774)."""
    t = ((poly or {}).get("more") or {}).get("totals", {}).get("2.5")
    return float(t["p"]) if t and t.get("volume", 0) >= min_volume else None


def _ou_block(dcp: dict, p_mkt: float | None, w: dict, blend, src: str | None = None) -> dict:
    """Goles: más de 2,5 (modelo combinado con el mercado) y ambos marcan (modelo)."""
    pm = float(dcp["p_over25"])
    return {"p_over": round(blend(pm, p_mkt, w), 4), "p_over_model": round(pm, 4),
            "p_over_market": None if p_mkt is None else round(float(p_mkt), 4), "market_src": src,
            "p_btts": round(float(dcp["p_btts"]), 4)}


# ---------------------------------------------------------------------------
# Guardado e informe
# ---------------------------------------------------------------------------
def save_predictions(settings: Settings, preds: list[dict], merge: bool = False,
                     matches: pd.DataFrame | None = None) -> list[dict]:
    """Guarda las predicciones, el historial de probabilidades y los cambios importantes
    (con su explicación). Devuelve la lista de cambios importantes de esta ejecución."""
    d = settings.predictions_dir
    snaps = [p.pop("_snapshot", None) for p in preds]
    moves = movements.record(settings, preds, snaps, matches)
    for p in preds:
        p["movement"] = moves.get((p["home"], p["away"]))
    latest_p = d / "latest.json"
    if merge and latest_p.exists():
        old = {(p["home"], p["away"]): p for p in json.loads(latest_p.read_text(encoding="utf-8"))}
        for p in preds:
            old[(p["home"], p["away"])] = p
        preds = sorted(old.values(), key=lambda x: (x["kickoff_utc"] or x["date"], x["home"]))
    latest_p.write_text(json.dumps(preds, indent=1, ensure_ascii=False), encoding="utf-8")
    (d / "latest.md").write_text(render_markdown(preds, movements.load_history(settings),
                                                 movements.important_moves(settings, preds)), encoding="utf-8")
    (d / "evolucion.md").write_text(movements.render_evolution(settings, preds), encoding="utf-8")
    log_p = d / "log.csv"
    rows = pd.DataFrame([{k: p[k] for k in ("generated_at", "season", "date", "home", "away", "p_home", "p_draw",
                                              "p_away", "model", "lineup_confirmed")} for p in preds])
    rows.to_csv(log_p, mode="a", header=not log_p.exists(), index=False)
    save_prematch(settings, preds)
    save_props_log(settings, preds)
    return [m for m in moves.values() if m and m.get("important")]


def save_props_log(settings: Settings, preds: list[dict]) -> None:
    """Registro de goles, goleadores, tarjetas y córners predichos (para medir después si aciertan)."""
    rows = []
    for x in preds:
        ou, pr, sc = x.get("ou") or {}, x.get("props") or {}, x.get("scorers") or {}
        rows.append({
            "generated_at": x["generated_at"], "season": x["season"], "date": x["date"], "home": x["home"],
            "away": x["away"], "kickoff_utc": x.get("kickoff_utc"), "lineup_confirmed": x.get("lineup_confirmed"),
            "xg_home": x["dc"]["xg_home"], "xg_away": x["dc"]["xg_away"],
            "p_over25": ou.get("p_over", x["dc"].get("p_over25")), "p_btts": ou.get("p_btts", x["dc"].get("p_btts")),
            "yellow_total": pr.get("yellow_total"), "corners_total": pr.get("corners_total"),
            "p_cards_over45": (pr.get("p_cards_over") or {}).get("4.5"),
            "p_corners_over95": (pr.get("p_corners_over") or {}).get("9.5"),
            "scorers": json.dumps([[side, s["name"], s["p"]] for side in ("home", "away") for s in sc.get(side, [])],
                                  ensure_ascii=False),
        })
    if rows:
        p = settings.predictions_dir / "props_log.csv"
        pd.DataFrame(rows).to_csv(p, mode="a", header=not p.exists(), index=False)


def xg_for_probs(p: list[float], total: float) -> list[float]:
    """Goles esperados (Poisson independientes) que reproducen un 1X2, cerca del total dado."""
    from scipy.stats import poisson
    k = np.arange(11)
    best, err = [total / 2, total / 2], 9.0
    for t in total * np.linspace(0.92, 1.08, 5):  # el total de goles casi fijo (coherente con más/menos)
        for s in np.linspace(0.15, 0.85, 71):
            ph, pa = poisson.pmf(k, t * s), poisson.pmf(k, t * (1 - s))
            m = np.outer(ph, pa)
            q = [np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()]
            e = sum((a - b) ** 2 for a, b in zip(q, p)) + 0.0005 * (t / total - 1) ** 2
            if e < err:
                best, err = [round(float(t * s), 3), round(float(t * (1 - s)), 3)], e
    return best


def save_prematch(settings: Settings, preds: list[dict], keep_days: int = 3) -> None:
    """Última predicción antes de empezar de cada partido (para la probabilidad en directo de la web)."""
    p = settings.predictions_dir / "prematch.json"
    try:
        old = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except json.JSONDecodeError:
        old = {}
    now = pd.Timestamp.now(tz="UTC")
    for x in preds:
        xg = [x["dc"]["xg_home"], x["dc"]["xg_away"]]
        pm = x.get("polymarket") or {}
        if pm.get("used"):  # goles esperados coherentes con la probabilidad final (para el directo)
            xg = xg_for_probs([x["p_home"], x["p_draw"], x["p_away"]], xg[0] + xg[1])
        old[f"{x['home']}|{x['away']}"] = {
            "home": x["home"], "away": x["away"], "date": x["date"], "kickoff_utc": x.get("kickoff_utc"),
            "espn_id": x.get("espn_id"), "p": [x["p_home"], x["p_draw"], x["p_away"]],
            "xg": xg, "poly_slug": pm.get("slug")}
    keep = {}
    for k, v in old.items():
        t = pd.Timestamp(v["kickoff_utc"]) if v.get("kickoff_utc") else pd.Timestamp(v["date"], tz="Europe/Madrid")
        if t >= now - pd.Timedelta(days=keep_days):
            keep[k] = v
    p.write_text(json.dumps(keep, ensure_ascii=False), encoding="utf-8")


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def render_markdown(preds: list[dict], history: pd.DataFrame | None = None, moves: list[dict] | None = None) -> str:
    if not preds:
        return "# Predicciones LaLiga\n\nNo hay partidos en el horizonte configurado.\n"
    lines = [f"# Predicciones LaLiga", "", f"Generado: {preds[0]['generated_at'][:16].replace('T', ' ')} UTC", "",
             "| Fecha | Partido | 1 | X | 2 | Evolución 1 | xG | Marcador más probable | +2.5 | Ambos marcan | Info |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in preds:
        when = p["date"]
        if p["kickoff_utc"]:
            when = pd.Timestamp(p["kickoff_utc"]).tz_convert("Europe/Madrid").strftime("%d/%m %H:%M")
        info = []
        if p["lineup_confirmed"]:
            info.append("✅ alineaciones")
        if p["referee"]:
            info.append(f"🧑‍⚖️ {p['referee']}")
        absn = [a["player"] + ("?" if a["status"] == "doubtful" else "") for v in p["absences"].values() for a in v]
        if absn:
            info.append("🚑 " + ", ".join(absn[:6]))
        top = p["dc"]["top_scores"][0]
        evo = movements.trend_text(history, p) if history is not None else ""
        lines.append(f"| {when} | {p['home']} – {p['away']} | {_pct(p['p_home'])} | {_pct(p['p_draw'])} | "
                     f"{_pct(p['p_away'])} | {evo} | {p['dc']['xg_home']:.2f}–{p['dc']['xg_away']:.2f} | {top[0]} ({_pct(top[1])}) | "
                     f"{_pct(p['dc']['p_over25'])} | {_pct(p['dc']['p_btts'])} | {' · '.join(info)} |")
    if moves:
        lines += ["", "## Cambios importantes", ""]
        for m in moves[-15:]:
            t = pd.Timestamp(m["to_ts"]).tz_convert("Europe/Madrid").strftime("%d/%m %H:%M")
            lines += [f"- {t} — " + movements.format_move(m).replace("\n  ", "\n  - ")]
    lines += ["", "1/X/2 = modelo final (meta-modelo). xG, marcadores, +2.5 y ambos marcan = Dixon-Coles "
              "ajustado por bajas. `?` = duda. Evolución 1 = probabilidad de victoria local en cada actualización "
              "(detalle en `evolucion.md`)."]
    return "\n".join(lines) + "\n"
