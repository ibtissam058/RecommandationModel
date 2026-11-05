"""Fake header-less CSV in the exact REES46 jewelry format (13 columns, 19-digit ids).

Only for testing the pipeline when the real file is not around. It has built-in seasonality
(e.g. rubies/pendants before Valentine's), so a working pipeline should show a gain.
Never report results computed on this data.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.events import EVENTS, ramp_for  # noqa: E402

CATS = ["jewelry.earring", "jewelry.ring", "jewelry.pendant", "jewelry.necklace", "jewelry.bracelet", "jewelry.brooch"]
METALS = ["gold", "silver"]
GEMS = ["diamond", "sapphire", "garnet", "amethyst", "fianit", "pearl", "topaz", "ruby", "quartz", ""]
COLORS = ["red", "white", "yellow"]

# event -> list of (attribute, value, multiplier)
BOOSTS = {
    "valentines_day": [("gem", "ruby", 3), ("gem", "diamond", 2.5), ("cat", "jewelry.pendant", 2), ("gender", "f", 1.5)],
    "womens_day": [("cat", "jewelry.earring", 2.5), ("gem", "pearl", 3)],
    "mothers_day": [("cat", "jewelry.brooch", 3), ("gem", "pearl", 2)],
    "halloween": [("gem", "garnet", 3), ("gem", "amethyst", 3), ("metal", "silver", 2)],
    "black_friday": [("cheap", None, 2.5)],
    "christmas": [("cat", "jewelry.ring", 2.5), ("metal", "gold", 1.5)],
}


def make_fake_jewelry_csv(path, start="2018-12-01", end="2021-12-01", seed=0, base_orders=45, n_products=2500, n_users=12000):
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, end, freq="D")
    P = n_products
    prod_id = rng.integers(10**18, 2 * 10**18, size=P)
    cat = rng.integers(0, len(CATS), P)
    metal = rng.integers(0, len(METALS), P)
    gem = rng.integers(0, len(GEMS), P)
    color = rng.integers(0, len(COLORS), P)
    brand = rng.integers(0, 3, P)
    gender = rng.random(P) < 0.5
    price = np.round(np.exp(rng.normal(4.8, 0.9, P)), 2)
    intro = rng.integers(-150, len(days), P)
    life = rng.integers(90, 300, P)
    base_w = np.exp(rng.normal(0, 1.0, P))

    # per-event multiplier vector over products
    ev_mult = {}
    for name, rules in BOOSTS.items():
        m = np.ones(P)
        for kind, val, mult in rules:
            if kind == "gem":
                m[np.array([GEMS[g] for g in gem]) == val] *= mult
            elif kind == "cat":
                m[np.array([CATS[c] for c in cat]) == val] *= mult
            elif kind == "metal":
                m[np.array([METALS[x] for x in metal]) == val] *= mult
            elif kind == "gender":
                m[gender] *= mult
            elif kind == "cheap":
                m[price < np.median(price)] *= mult
        ev_mult[name] = m

    user_id = rng.integers(10**18, 2 * 10**18, size=n_users)
    activity = rng.gamma(0.6, 1.0, n_users)
    activity /= activity.sum()
    pref_cat = rng.integers(0, len(CATS), n_users)
    pref_metal = rng.integers(0, len(METALS), n_users)
    pref_gem = rng.integers(0, len(GEMS), n_users)

    rows = []
    for di, day in enumerate(days):
        act = np.flatnonzero((intro <= di) & (di < intro + life))
        if len(act) < 20:
            continue
        ev, _, ramp = ramp_for(day)
        n = rng.poisson(base_orders * (1 + 2.0 * ramp))
        if n == 0:
            continue
        u = rng.choice(n_users, size=n, p=activity)
        w = np.tile(base_w[act], (n, 1))
        if ev is not None:
            w = w * (1 + ramp * (ev_mult[ev.name][act] - 1))[None, :]
        w = w * (1 + 2.0 * (cat[act][None, :] == pref_cat[u][:, None])
                 + 1.0 * (metal[act][None, :] == pref_metal[u][:, None])
                 + 1.0 * (gem[act][None, :] == pref_gem[u][:, None]))
        cdf = np.cumsum(w, axis=1)
        pick = (cdf < rng.random((n, 1)) * cdf[:, -1:]).sum(axis=1)
        for j in range(n):
            p = act[min(pick[j], len(act) - 1)]
            t = day + pd.Timedelta(seconds=int(rng.integers(0, 86400)))
            rows.append([t.strftime("%Y-%m-%d %H:%M:%S UTC"), int(rng.integers(10**18, 2 * 10**18)), int(prod_id[p]), 1,
                         1806829201890738522, CATS[cat[p]], int(brand[p]), float(price[p]), int(user_id[u[j]]),
                         "f" if gender[p] else "", COLORS[color[p]], METALS[metal[p]], GEMS[gem[p]]])
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    return len(rows)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "fake_jewelry.csv"
    print(make_fake_jewelry_csv(out), "rows written to", out)
