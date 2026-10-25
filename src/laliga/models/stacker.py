"""Meta-modelo: regresión logística multinomial sobre Dixon-Coles + Elo + contexto (+ mercado).

Las variables se organizan en grupos; `laliga tune` elige qué grupos entran y la
regularización usando solo temporadas de validación (nunca las de test).
Se entrenan dos variantes: con mercado (si hay cuotas) y sin mercado.
Las variables ausentes se imputan con la mediana de entrenamiento + indicador.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..metrics import RESULT_IDX

EPS = 1e-6

CORE = ["dc_hd", "dc_draw", "dc_xg_sum", "elo_diff"]
GROUPS: dict[str, list[str]] = {
    "sot_model": ["sot_model_diff", "sot_model_sum"],
    "form_shots": ["sh_net_diff", "sot_net_diff", "cor_net_diff"],
    "form_goals": ["gf_net_diff", "pts_diff"],
    "venue": ["venue_edge"],
    "rest": ["rest_diff", "short_rest_home", "short_rest_away"],
    "travel": ["log_travel", "island"],
    "promoted": ["new_home", "new_away"],
    "referee": ["ref_home_ppg_dev", "ref_cards_dev"],
    "weather": ["rain", "wind", "temp_dev"],
    "xg_model": ["xg_model_diff", "xg_model_sum"],
    "form_xg": ["xg_net_diff"],
    "stakes": ["late", "stake_diff_late", "dead_home", "dead_away"],
    "h2h": ["h2h_gd", "h2h_draw"],
    "discipline": ["red_prev_home", "red_prev_away", "yellow_diff"],
    "timing": ["midweek", "ko_early", "ko_late", "post_break", "early_season"],
    "derby": ["derby"],
    "momentum": ["mom_diff"],
    "venue_form": ["venue_form_diff"],
    "congestion": ["euro_prev_home", "euro_prev_away", "euro_next_home", "euro_next_away",
                   "rest_all_diff", "load14_diff"],
    "coach": ["new_coach_home", "new_coach_away"],
    "no_fans": ["no_fans"],
    "market_move": ["mkt_move_hd", "mkt_move_draw"],
}
MARKET_COLS = ["mkt_hd", "mkt_draw"]
# columnas que pueden faltar en parte de las filas -> llevan indicador de ausencia
OPTIONAL_COLS = [c for g in ("sot_model", "form_shots", "form_goals", "venue", "referee", "weather",
                             "xg_model", "form_xg", "momentum", "venue_form", "market_move")
                 for c in GROUPS[g]] + ["ko_early", "ko_late", "yellow_diff", "euro_prev_home", "rest_all_diff"]
# columnas que se copian tal cual desde la tabla
PASSTHROUGH = [c for g in ("stakes", "h2h", "discipline", "timing", "derby", "momentum", "venue_form",
                           "congestion", "coach", "no_fans") for c in GROUPS[g]]


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    return df[name] if name in df else pd.Series(np.nan, index=df.index)


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Transforma columnas crudas en variables del meta-modelo."""
    X = pd.DataFrame(index=df.index)
    X["dc_hd"] = np.log(np.clip(df.dc_ph, EPS, 1) / np.clip(df.dc_pa, EPS, 1))
    X["dc_draw"] = _logit(df.dc_pd)
    X["dc_xg_sum"] = df.dc_lam + df.dc_mu
    X["elo_diff"] = df.elo_diff / 100
    sl, sm = _col(df, "sot_lam"), _col(df, "sot_mu")
    X["sot_model_diff"] = np.log(sl / sm)
    X["sot_model_sum"] = np.log(sl + sm)
    for k in ("gf", "sh", "sot", "cor", "xg"):
        X[f"{k}_net_diff"] = ((_col(df, f"{k}_for_home") - _col(df, f"{k}_against_home"))
                              - (_col(df, f"{k}_for_away") - _col(df, f"{k}_against_away")))
    X["pts_diff"] = _col(df, "pts_home") - _col(df, "pts_away")
    X["venue_edge"] = _col(df, "venue_edge")
    X["rest_diff"] = (df.rest_home - df.rest_away).clip(-10, 10)
    X["short_rest_home"] = (df.rest_home <= 3).astype(float)
    X["short_rest_away"] = (df.rest_away <= 3).astype(float)
    X["log_travel"] = np.log1p(df.travel_km)
    X["island"] = df.island
    X["new_home"], X["new_away"] = df.new_home, df.new_away
    X["ref_home_ppg_dev"] = _col(df, "ref_home_ppg_dev")
    X["ref_cards_dev"] = _col(df, "ref_cards_dev")
    X["rain"] = np.log1p(_col(df, "precip_mm"))
    X["wind"] = _col(df, "wind_kmh")
    X["temp_dev"] = (_col(df, "temp_c") - 18).abs()
    X["mkt_hd"] = np.log(np.clip(_col(df, "mkt_ph"), EPS, 1) / np.clip(_col(df, "mkt_pa"), EPS, 1))
    X["mkt_draw"] = _logit(_col(df, "mkt_pd"))
    # movimiento de cuotas: cierre (o última instantánea) menos apertura
    X["mkt_move_hd"] = np.log(np.clip(_col(df, "mktc_ph"), EPS, 1) / np.clip(_col(df, "mktc_pa"), EPS, 1)) - X["mkt_hd"]
    X["mkt_move_draw"] = _logit(_col(df, "mktc_pd")) - X["mkt_draw"]
    xl, xm = _col(df, "xg_lam"), _col(df, "xg_mu")
    X["xg_model_diff"] = np.log(xl / xm)
    X["xg_model_sum"] = np.log(xl + xm)
    for c in PASSTHROUGH:
        X[c] = _col(df, c)
    return X


def has_market(df: pd.DataFrame) -> pd.Series:
    return pd.concat([_col(df, c) for c in ("mkt_ph", "mkt_pd", "mkt_pa")], axis=1).notna().all(axis=1)


@dataclass
class Stacker:
    use_market: bool = False
    C: float = 0.005
    groups: list[str] = field(default_factory=lambda: list(GROUPS))
    cols: list | None = None
    ind_cols: list | None = None
    medians: pd.Series | None = None
    scaler: StandardScaler | None = None
    model: LogisticRegression | None = None

    def feature_list(self) -> list[str]:
        f = list(CORE) + [c for g in self.groups for c in GROUPS[g]]
        return f + (MARKET_COLS if self.use_market else [])

    def _prep(self, X: pd.DataFrame, fit: bool) -> np.ndarray:
        if fit:
            wanted = self.feature_list()
            self.cols = [c for c in wanted if X[c].notna().any()]  # sin datos -> fuera
            self.medians = X[self.cols].median()
            self.ind_cols = [c for c in self.cols if c in OPTIONAL_COLS and X[c].isna().any()]
        X = X.reindex(columns=self.cols)
        ind = X[self.ind_cols].isna().astype(float).add_suffix("_na")
        arr = pd.concat([X.fillna(self.medians), ind], axis=1).to_numpy(dtype=float)
        if fit:
            self.scaler = StandardScaler().fit(arr)
        return self.scaler.transform(arr)

    def fit(self, df: pd.DataFrame) -> "Stacker":
        if self.use_market:
            df = df[has_market(df)]
        X = engineer(df)
        y = df.result.map(RESULT_IDX).to_numpy()
        self.model = LogisticRegression(C=self.C, max_iter=3000).fit(self._prep(X, fit=True), y)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        p = self.model.predict_proba(self._prep(engineer(df), fit=False))
        out = np.zeros((len(df), 3))
        out[:, self.model.classes_] = p
        return out

    def coefficients(self) -> pd.DataFrame:
        names = list(self.cols) + [f"{c}_na" for c in self.ind_cols]
        return pd.DataFrame(self.model.coef_.T, index=names, columns=["H", "D", "A"]).round(3)

    def save(self, path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path) -> "Stacker":
        with open(path, "rb") as f:
            return pickle.load(f)
