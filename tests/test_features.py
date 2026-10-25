import numpy as np
import pandas as pd
import pytest

from laliga.features.sequential import build_sequential_features


def test_no_leakage(league):
    df, _ = league
    cut = pd.Timestamp("2016-02-01")
    f1 = build_sequential_features(df)
    df2 = df.copy()
    fut = df2.date >= cut
    df2.loc[fut, "hg"] = 9.0
    df2.loc[fut, "h_shots"] = 99.0
    df2.loc[fut, "referee"] = "Otro"
    f2 = build_sequential_features(df2)
    past = df.date < cut
    pd.testing.assert_frame_equal(f1[past], f2[past])
    # y el primer partido tras el corte también es idéntico (estado previo)
    first = df[fut].index[0]
    assert f1.loc[first, "elo_diff"] == f2.loc[first, "elo_diff"]


def test_elo_is_zero_sum_and_moves_right(league):
    df, _ = league
    f = build_sequential_features(df)
    t = df.join(f)
    a = t.iloc[0]
    nxt_home = t[(t.index > 0) & ((t.home == a.home) | (t.away == a.home))].iloc[0]
    elo_after = nxt_home.elo_home if nxt_home.home == a.home else nxt_home.elo_away
    if a.result == "H":
        assert elo_after > a.elo_home
    elif a.result == "A":
        assert elo_after < a.elo_home
    # suma de ratings conservada dentro de una temporada
    s = df[df.season == df.season.min()]
    ratings = {}
    for _, r in s.join(f).iterrows():
        ratings[r.home], ratings[r.away] = r.elo_home, r.elo_away
    assert np.mean(list(ratings.values())) == pytest.approx(1420, abs=1)


def test_rest_and_travel(league):
    df, _ = league
    f = build_sequential_features(df)
    assert f.rest_home.between(0, 21).all()
    assert (f.travel_km >= 0).all()
    lp = df[(df.home == "Las Palmas") | (df.away == "Las Palmas")].index
    assert (f.loc[lp, "island"] == 1).all()


def test_promoted_teams_flagged_and_get_prior(league):
    df, _ = league
    f = build_sequential_features(df)
    t = df.join(f)
    s1 = df.season.min() + 1
    prev = set(df[df.season == s1 - 1].home)
    newcomers = set(df[df.season == s1].home) - prev
    assert newcomers
    first_games = t[(t.season == s1) & t.home.isin(newcomers)].groupby("home").head(1)
    assert (first_games.new_home == 1).all()
    assert first_games.sh_for_home.notna().all()  # recibe un prior, no NaN ni forma antigua


def test_future_rows_get_current_state(league):
    df, _ = league
    df = df.copy()
    last = df.index[-1]
    df.loc[last, ["hg", "ag", "result"]] = [np.nan, np.nan, None]
    f = build_sequential_features(df)
    assert f.loc[last].notna()[["elo_home", "elo_away", "pts_home"]].all()


def test_referee_features(league):
    df, _ = league
    f = build_sequential_features(df)
    late = f.iloc[-100:]
    assert late.ref_n.min() > 10
    assert late.ref_home_ppg_dev.abs().max() < 1
