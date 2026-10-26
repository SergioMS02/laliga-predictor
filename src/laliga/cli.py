"""Línea de comandos.

  laliga update                 descarga/actualiza datos
  laliga tune                   ajusta hiperparámetros con temporadas de validación
  laliga backtest               evaluación walk-forward en temporadas de test
  laliga train                  entrena el meta-modelo final
  laliga predict                predice los próximos partidos (noticias, árbitro, clima, cuotas...)
  laliga watch-lineups          re-predice partidos que empiezan pronto si ya hay alineación
  laliga backfill-espn          histórico de alineaciones y árbitros (ESPN)
  laliga calibrate-availability estima el efecto real de las bajas
  laliga evaluate-live          rendimiento de las predicciones ya emitidas
  laliga export-site            datos de la web (clasificación, proyección, equipos, aciertos)
  laliga settle                 liquida apuestas y quiniela en cuanto terminan los partidos
  laliga quiniela               actualiza la quiniela oficial (boleto, probabilidades, resultados)
  laliga all                    update + backtest + train + predict
"""
from __future__ import annotations

import argparse
import json
import logging
import sys


from . import history, notify, pipeline
from . import predict as pred
from .config import load_settings
from .data import results

log = logging.getLogger("laliga")


def _print_backtest(r: dict) -> None:
    print(f"\nBacktest walk-forward, temporadas de test {r['test_seasons'][0]}-{r['test_seasons'][-1]}")
    print(f"{'modelo':18s} {'log-loss':>9s} {'Brier':>7s} {'RPS':>7s} {'acierto':>8s}")
    for k, v in sorted(r["overall"].items(), key=lambda kv: kv[1]["log_loss"]):
        print(f"{k:18s} {v['log_loss']:9.4f} {v['brier']:7.4f} {v['rps']:7.4f} {100 * v['accuracy']:7.1f}%")
    if r.get("market_subset"):
        print("\nSolo partidos con cuotas:")
        for k, v in sorted(r["market_subset"].items(), key=lambda kv: kv[1]["log_loss"]):
            print(f"{k:18s} {v['log_loss']:9.4f} {v['brier']:7.4f} {v['rps']:7.4f} {100 * v['accuracy']:7.1f}%")


def _bets(s, matches, preds, send: bool = False) -> dict | None:
    from . import betting
    try:
        r = betting.update(s, matches, preds)
    except Exception as e:
        log.warning("No se pudo actualizar la cartera simulada: %s", e)
        return None
    txt = "\n\n".join(t for t in (betting.format_placed(r["placed"]),
                                    betting.format_settled(r["settled"], betting.summarize(r["ledger"], betting.cfg(s))))
                       if t)
    if txt:
        print(txt)
        if send:
            notify.send(s, txt)
    return r


def _quiniela(s, preds=None, online: bool = True, send: bool = False) -> None:
    from . import quiniela
    try:
        before = quiniela.load(s)["jornadas"]
        st = quiniela.update(s, preds, online=online)
    except Exception as e:
        log.warning("No se pudo actualizar la quiniela: %s", e)
        return
    msgs = []
    for d, j in st["jornadas"].items():
        old = before.get(d, {})
        if j.get("frozen") and not old.get("frozen") and not j.get("late"):
            msgs.append(quiniela.format_ticket(j))
        if j.get("result") and not old.get("result"):
            r = j["result"]
            msgs.append(f"Quiniela {d}: sencilla {r['sencilla']['hits']} aciertos ({r['sencilla']['prize']:.2f} €), "
                        f"reducida {r['reducida']['hits']} aciertos ({r['reducida']['prize']:.2f} € por {r['reducida']['cost']:.2f} €)")
    for m in msgs:
        print(m)
        if send:
            notify.send(s, m)


def _leagues(s, online: bool, with_eval: bool = False) -> None:
    from . import leagues
    try:
        leagues.build(s, online=online, with_eval=with_eval)
    except Exception as e:  # nunca debe tumbar la predicción de LaLiga
        log.warning("Otras competiciones no disponibles: %s", e)


def _export_site(s, matches, projection: bool, online: bool = True) -> None:
    from . import site_export
    try:
        site_export.export(s, matches, with_projection=projection, online=online)
    except Exception as e:  # la web nunca debe tumbar la predicción
        log.warning("No se pudieron generar los datos de la web: %s", e)
    pipeline.save_status(s)


def _notify_moves(s, moves: list[dict], send: bool) -> None:
    from .movements import format_move
    if not moves:
        return
    txt = "Cambios importantes\n\n" + "\n\n".join(format_move(m) for m in moves)
    print(txt)
    if send:
        notify.send(s, txt)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="laliga", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["update", "tune", "backtest", "train", "predict", "watch-lineups",
                                        "backfill-espn", "calibrate-availability", "leagues", "evaluate-live", "export-site",
                                        "settle", "quiniela", "all"])
    ap.add_argument("--offline", action="store_true", help="no usar red (solo caché)")
    ap.add_argument("--days", type=int, default=None, help="horizonte de predicción en días")
    ap.add_argument("--weather", action="store_true", help="incluir clima histórico en tune/backtest (≈400 peticiones)")
    ap.add_argument("--seasons", type=str, default=None, help="backfill-espn: p. ej. 2022-2026")
    ap.add_argument("--notify", action="store_true", help="enviar resumen por Telegram")
    ap.add_argument("--no-other", action="store_true", help="backfill-espn: sin el calendario de otras competiciones")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    s = load_settings()
    online = not a.offline
    cmd = a.command

    if cmd == "backfill-espn":
        if a.seasons:
            lo, _, hi = a.seasons.partition("-")
            seasons = list(range(int(lo), int(hi or lo) + 1))
        else:  # desde la primera temporada de validación: así tune puede usar alineaciones/árbitros/Europa
            cur = results.current_season()
            seasons = list(range(int(s.get("backtest", "first_val_season", cur - 3)), cur + 1))
        print(json.dumps(history.backfill_espn(s, seasons, other=not a.no_other)))
        return 0

    matches, sb = pipeline.load_matches(s, online=online, days_ahead=a.days)
    played = matches.hg.notna().sum()
    print(f"Partidos cargados: {len(matches)} ({played} jugados, {len(matches) - played} pendientes)")
    if cmd == "update":
        return 0
    if cmd == "tune":
        from .tune import tune
        r = tune(load_settings(use_tuned=False), matches, with_weather=a.weather)
        print(json.dumps(r, indent=2))
        return 0
    if cmd in ("backtest", "all"):
        _print_backtest(pipeline.backtest(s, matches, with_weather=a.weather))
        try:  # tarjetas y córners en las tres últimas temporadas completas
            from . import props
            cur = results.current_season()
            pe = props.evaluate(matches, [cur - 3, cur - 2, cur - 1])
            (s.predictions_dir / "props_eval.json").write_text(json.dumps(pe), encoding="utf-8")
            print("Tarjetas y córners:", pe)
        except Exception as e:
            log.warning("Evaluación de tarjetas y córners no disponible: %s", e)
        try:  # incertidumbre de la proyección de temporada, medida con temporadas pasadas
            from . import site_export
            cur = results.current_season()
            cp = site_export.calibrate_projection(s, matches.reset_index(drop=True), list(range(cur - 11, cur)))
            print("Proyección: sigma", cp.get("sigma"))
        except Exception as e:
            log.warning("Calibración de la proyección no disponible: %s", e)
        _leagues(s, online, with_eval=True)
        try:  # probabilidad en directo con tiros a puerta (Understat, minuto a minuto)
            from .data import shots as shots_mod
            print("Directo:", shots_mod.run_live_calibration(s, online=online))
        except Exception as e:
            log.warning("Calibración del directo no disponible: %s", e)
        try:  # Polymarket frente al cierre de las casas y al modelo (y peso del modelo en la mezcla)
            from .data import polymarket
            print("Polymarket:", polymarket.run_evaluation(s, matches, online=online))
        except Exception as e:
            log.warning("Evaluación de Polymarket no disponible: %s", e)
        bp = s.predictions_dir / "betting_backtest.json"
        bb = json.loads(bp.read_text(encoding="utf-8")) if bp.exists() else None
        if bb:
            print("\nApuestas simuladas en temporadas de prueba (Bet365, %s € por apuesta):" % bb["config"]["stake"])
            for k, v in bb["strategies"].items():
                print(f"  {k:9s} {v['bets']:5d} apuestas, aciertos {v['won']}, beneficio {v['profit']:+.0f} €, ROI {100 * v['roi']:+.1f} %")
    if cmd in ("train", "all"):
        if not (s.processed_dir / "oos_table.csv").exists():
            pipeline.backtest(s, matches, with_weather=a.weather)
        print("Meta-modelo entrenado:", pipeline.train_final(s))
    if cmd == "calibrate-availability":
        try:
            print(history.calibrate_availability(s, matches))
        except (RuntimeError, FileNotFoundError) as e:
            print(f"No se pudo calibrar: {e}")
            return 1
        try:
            print(json.dumps(history.lineup_market_weight(load_settings(), matches), indent=2))
        except (RuntimeError, FileNotFoundError) as e:
            print(f"Peso del once frente al mercado no disponible: {e}")
        return 0
    if cmd == "leagues":
        _leagues(s, online)
        return 0
    if cmd == "export-site":
        _export_site(s, matches, projection=True, online=online)
        return 0
    if cmd == "quiniela":
        _quiniela(s, online=online, send=a.notify)
        return 0
    if cmd == "settle":
        _bets(s, matches, [], send=a.notify)
        _quiniela(s, online=online, send=a.notify)
        _export_site(s, matches, projection=True, online=online)
        return 0
    if cmd == "evaluate-live":
        ev = json.dumps(history.evaluate_live(s, matches), indent=2)
        (s.predictions_dir / "live_eval.json").write_text(ev, encoding="utf-8")
        print(ev)
        return 0
    if cmd in ("predict", "all"):
        preds = pred.predict(s, matches, sb, days=a.days, online=online)
        big = pred.save_predictions(s, preds, matches=matches)
        print((s.predictions_dir / "latest.md").read_text(encoding="utf-8"))
        _notify_moves(s, big, a.notify)
        _bets(s, matches, preds, send=a.notify)
        _quiniela(s, preds, online=online, send=a.notify)
        _leagues(s, online)
        _export_site(s, matches, projection=True, online=online)
        if a.notify and preds:
            soon = notify.upcoming_within(preds, int(s.get("notify", "summary_hours", 48)))
            if soon:
                notify.send(s, "Predicciones LaLiga\n\n" + "\n\n".join(notify.format_prediction(p) for p in soon))
    if cmd == "watch-lineups":
        window = int(s.get("predict", "lineup_window_minutes", 100))
        prev = {}
        lp = s.predictions_dir / "latest.json"
        if lp.exists():
            prev = {(p["home"], p["away"]): p for p in json.loads(lp.read_text(encoding="utf-8"))}
        preds = pred.predict(s, matches, sb, online=online, only_within_minutes=window)
        fresh = [p for p in preds if p["lineup_confirmed"] and not prev.get((p["home"], p["away"]), {}).get("lineup_confirmed")]
        big = pred.save_predictions(s, preds, merge=True, matches=matches) if preds else []
        _notify_moves(s, big, a.notify)
        _bets(s, matches, preds, send=a.notify)  # también liquida lo ya terminado
        if preds:
            _quiniela(s, online=online, send=a.notify)
        _export_site(s, matches, projection=False, online=online)
        print(f"Partidos en ventana: {len(preds)}; con alineación nueva: {len(fresh)}")
        explained = {(m["home"], m["away"]) for m in big}
        for p in fresh:
            print(notify.format_prediction(p))
            if a.notify and (p["home"], p["away"]) not in explained:  # si hubo cambio importante ya se avisó con el porqué
                old = prev.get((p["home"], p["away"]))
                txt = notify.format_prediction(p)
                if old:
                    txt += f"\n(antes: 1 {100 * old['p_home']:.0f}% · X {100 * old['p_draw']:.0f}% · 2 {100 * old['p_away']:.0f}%)"
                notify.send(s, "Alineaciones confirmadas\n\n" + txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
