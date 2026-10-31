import json

import numpy as np
import pandas as pd

from laliga import segments


def test_segments_find_where_model_is_better(settings):
    rng = np.random.default_rng(5)
    n = 3000
    true = rng.dirichlet([3, 2, 2], n)
    y = np.array([rng.choice(3, p=pp) for pp in true])
    early = rng.random(n) < 0.2
    noise = rng.dirichlet([3, 3, 3], n)
    market = 0.85 * true + 0.15 * noise
    model = 0.6 * true + 0.4 * noise
    model[early] = true[early]  # el modelo sabe algo que el mercado no en las primeras jornadas
    m = pd.DataFrame({"season": 2023, "home": [f"H{i}" for i in range(n)], "away": "B"})
    pd.DataFrame({"season": 2023, "y": y, **{f"stacker_{c}": model[:, i] for i, c in enumerate("HDA")},
                  **{f"mercado_{c}": market[:, i] for i, c in enumerate("HDA")}},
                 index=m.index).to_csv(settings.processed_dir / "backtest_predictions.csv")
    m.assign(early_season=early.astype(float)).to_csv(settings.processed_dir / "oos_table.csv", index=False)
    r = segments.run(settings, m)
    by = {x["segmento"]: x for x in r["segmentos"]}
    assert by["Todos los partidos"]["veredicto"] == "mercado mejor"
    early_row = next(v for k, v in by.items() if k.startswith("Primeras jornadas"))
    assert early_row["veredicto"] == "modelo mejor" and early_row["ci95"][1] < 0
    assert json.loads((settings.predictions_dir / "segments.json").read_text())["segmentos"]
