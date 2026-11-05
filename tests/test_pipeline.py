import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.data import load  # noqa: E402
from src.evaluate import evaluate_rows, make_eval_rows  # noqa: E402
from src.events import EVENTS, ramp_for  # noqa: E402
from src.recommender import Catalog, EventLifts, Interactions, SeasonalRecommender  # noqa: E402
from synthetic import make_fake_jewelry_csv  # noqa: E402


def test_event_dates():
    ev = {e.name: e for e in EVENTS}
    assert ev["black_friday"].date(2019) == date(2019, 11, 29)
    assert ev["black_friday"].date(2020) == date(2020, 11, 27)
    assert ev["black_friday"].date(2021) == date(2021, 11, 26)
    assert ev["mothers_day"].date(2021) == date(2021, 5, 9)


def test_ramp():
    e, days, ramp = ramp_for("2021-02-14")
    assert e.name == "valentines_day" and days == 0 and ramp == 1.0
    e, days, ramp = ramp_for("2021-02-07")
    assert e.name == "valentines_day" and 0 < ramp < 1
    assert ramp_for("2021-01-05") == (None, None, 0.0)


def _fake(tmp_path):
    p = tmp_path / "fake.csv"
    make_fake_jewelry_csv(p, start="2019-01-01", end="2020-12-31", base_orders=20, n_products=800, n_users=3000)
    return load(str(p))


def test_load_and_pipeline(tmp_path):
    df = _fake(tmp_path)
    assert {"user_id", "product_id", "event_time", "price"} <= set(df.columns)
    assert df.user_id.str.len().min() >= 19          # ids were not rounded

    rec = SeasonalRecommender(df)
    user = df.user_id.iloc[-1]
    out = rec.recommend(user, "2020-12-20", k=5)
    assert len(out) == 5 and out["score"].is_monotonic_decreasing
    assert out["event_fit"].notna().all()            # Dec 20 is inside the Christmas window
    # the season must change the ranking inside an event window ...
    a = rec.recommend(user, "2020-12-24", k=20)["product_id"].tolist()
    b = rec.recommend(user, "2020-12-24", k=20, seasonal=False)["product_id"].tolist()
    assert a != b
    # ... and must not change it on a normal day
    a = rec.recommend(user, "2020-10-01", k=20)["product_id"].tolist()
    b = rec.recommend(user, "2020-10-01", k=20, seasonal=False)["product_id"].tolist()
    assert a == b

    catalog = Catalog(df)
    inter = Interactions(df, catalog)
    lifts = EventLifts(inter, "2020-01-01")
    rows = make_eval_rows(df, catalog, "2020-01-01", "2020-12-31", max_per_group=60)
    res = evaluate_rows(inter, lifts, rows, [
        {"label": "pop", "model": "popularity", "alpha_max": 0, "gamma": 0.5},
        {"label": "ps", "model": "personalized_seasonal", "alpha_max": 0.5, "gamma": 0.5}])
    for c in ["hit@10", "ndcg@10", "seg@10"]:
        v = res[c].dropna()
        assert ((v >= 0) & (v <= 1)).all()


def test_no_future_leakage(tmp_path):
    df = _fake(tmp_path)
    inter = Interactions(df, Catalog(df))
    user = df.user_id.iloc[0]
    first = df[df.user_id == user].event_time.min()
    assert inter.user_profile(user, first.value) is None          # nothing known before 1st purchase
    assert inter.user_profile(user, first.value + 1) is not None

    # event lifts must ignore event occurrences that end after the cutoff
    lifts = EventLifts(inter, "2019-02-01")
    assert lifts.n_occurrences["valentines_day"] == 0
    assert np.allclose(lifts.lift["valentines_day"], 0)
