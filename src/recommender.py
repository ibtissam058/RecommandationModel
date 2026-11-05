"""Core of the project: catalog, purchase history, event lifts and the scoring blend.

final_score(user, item, date) = (1 - alpha) * personal(user, item) + alpha * event(item, event)

  personal = (1 - gamma) * recent_popularity + gamma * user_affinity
  event    = how over-represented the item's ATTRIBUTES (category, metal, gem, colour,
             gender, price bucket) were in the same event window in past years.
  alpha    = alpha_max * ramp(date); ramp is 0 on normal days and grows to 1 on the event day.

Why attributes and not product ids? Jewelry products come and go, so a product that sold
before Valentine's 2019 usually no longer exists in 2021. "Rubies and pendants sell before
Valentine's" does transfer across years.

Every look-up is restricted to data strictly BEFORE the time being predicted (no leakage).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .events import EVENTS, ramp_for

DAY_NS = 86_400 * 10**9
ATTRS = ["category_code", "brand", "product_gender", "color", "metal", "gem", "price_bucket"]
MODELS = ["popularity", "seasonal_popularity", "personalized", "personalized_seasonal"]


def _minmax(x: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return x.astype(float)
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi > lo else np.zeros(len(x))


class Catalog:
    """One row per product + an integer 'token' matrix describing its attributes."""

    def __init__(self, df: pd.DataFrame, price_cuts=None):
        d = df.sort_values("event_time")
        agg = {c: "last" for c in ATTRS[:-1]}
        agg["price"] = "median"
        items = d.groupby("product_id").agg(agg)

        if price_cuts is None:
            price_cuts = np.nanquantile(d["price"], [0.2, 0.4, 0.6, 0.8])
        self.price_cuts = np.asarray(price_cuts)
        bucket = np.digitize(items["price"].fillna(0).to_numpy(), self.price_cuts)
        items["price_bucket"] = pd.Series([f"P{b}" for b in bucket], index=items.index).where(items["price"].notna())

        columns, offset, names = [], 0, []
        for a in ATTRS:
            codes, uniques = pd.factorize(items[a].astype(object))
            columns.append(np.where(codes >= 0, codes + offset, -1))
            names += [f"{a}={u}" for u in uniques]
            offset += len(uniques)
        self.V = offset                       # number of distinct attribute values
        self.token_names = names
        T = np.stack(columns, axis=1)
        T[T < 0] = self.V                     # missing value -> dummy token V (always weight 0)
        self.T = T.astype(np.int32)

        seg = (items["category_code"].fillna("?").astype(str) + "|"
               + items["metal"].fillna("?").astype(str) + "|"
               + items["gem"].fillna("?").astype(str))
        self.segment = pd.factorize(seg)[0]   # coarse "type of jewelry" id
        self.items = items
        self.index = pd.Index(items.index)
        self.N = len(items)


class Interactions:
    """Time-sorted purchases with fast 'what happened before time t' look-ups."""

    def __init__(self, df: pd.DataFrame, catalog: Catalog):
        d = df.sort_values("event_time", kind="stable").reset_index(drop=True)
        self.catalog = catalog
        self.t = d["event_time"].to_numpy().astype("datetime64[ns]").astype("int64")
        self.item = catalog.index.get_indexer(d["product_id"])
        if (self.item < 0).any():
            raise ValueError("Some purchased products are missing from the catalog")
        self.user_rows = d.groupby("user_id").indices
        self._pool_cache: dict = {}

    def pool(self, t_ns: int, days: int):
        """Items sold in the `days` before t_ns ('what is currently on sale') + normalised popularity."""
        key = (t_ns, days)
        if key not in self._pool_cache:
            if len(self._pool_cache) > 3000:
                self._pool_cache.clear()
            lo = np.searchsorted(self.t, t_ns - days * DAY_NS, "left")
            hi = np.searchsorted(self.t, t_ns, "left")
            counts = np.bincount(self.item[lo:hi], minlength=self.catalog.N)
            pool = np.flatnonzero(counts)
            self._pool_cache[key] = (pool, _minmax(np.log1p(counts[pool])))
        return self._pool_cache[key]

    def user_profile(self, user, t_ns: int) -> Optional[np.ndarray]:
        """Share of the user's past purchases (strictly before t_ns) having each attribute value."""
        rows = self.user_rows.get(user)
        if rows is None:
            return None
        k = np.searchsorted(self.t[rows], t_ns, "left")
        if k == 0:
            return None
        toks = self.catalog.T[self.item[rows[:k]]].ravel()
        u = np.bincount(toks, minlength=self.catalog.V + 1).astype(float) / k
        u[self.catalog.V] = 0.0
        return u


class EventLifts:
    """Per event: log-lift of every attribute value inside the event window vs. all other days.

    Fitted ONLY on past occurrences of the event that ended before `cutoff`.
    """

    def __init__(self, inter: Interactions, cutoff, smooth: float = 20.0):
        # smooth = pseudo-counts added to both sides of the ratio: rare attribute values
        # (a gem seen 3 times in the window) are pulled strongly towards "no effect",
        # frequent ones are barely changed.
        cat = inter.catalog
        V, T = cat.V, cat.T
        cutoff_ns = pd.Timestamp(cutoff).value
        hi = int(np.searchsorted(inter.t, cutoff_ns, "left"))
        self.lift = {ev.name: np.zeros(V + 1) for ev in EVENTS}
        self.n_occurrences = {ev.name: 0 for ev in EVENTS}
        if hi == 0:
            return
        s_all = np.bincount(T[inter.item[:hi]].ravel(), minlength=V + 1) / hi
        years = range(pd.Timestamp(int(inter.t[0])).year, pd.Timestamp(int(inter.t[hi - 1])).year + 1)
        ok = s_all > 0
        for ev in EVENTS:
            counts, n = np.zeros(V + 1), 0
            for y in years:
                d = pd.Timestamp(ev.date(y))
                start = (d - pd.Timedelta(days=ev.window)).value
                end = (d + pd.Timedelta(days=1)).value
                if end > cutoff_ns:
                    continue                      # occurrence not finished before cutoff -> unseen
                a, b = np.searchsorted(inter.t, [start, end], "left")
                if b <= a:
                    continue
                counts += np.bincount(T[inter.item[a:b]].ravel(), minlength=V + 1)
                n += int(b - a)
                self.n_occurrences[ev.name] += 1
            if n > 0:
                lift = np.zeros(V + 1)
                lift[ok] = np.log((counts[ok] + smooth) / (n * s_all[ok] + smooth))
                lift[V] = 0.0
                self.lift[ev.name] = lift


@dataclass
class Context:
    pool: np.ndarray                    # candidate item indices (sorted)
    pop_n: np.ndarray                   # popularity score in [0,1]
    aff_n: Optional[np.ndarray]         # user-affinity score in [0,1] (None = no history)
    lift_n: Optional[np.ndarray]        # event score in [0,1] (None = no event now)
    ramp: float
    event: Optional[str]


def build_context(inter: Interactions, lifts: EventLifts, user, when,
                  pool_at=None, profile_at=None, pool_days: int = 60) -> Context:
    """Everything needed to score the candidates for one user at one moment.

    `when` decides the season. By default the catalog and the user history are taken as of
    `when`; the demo app overrides pool_at/profile_at to 'today's catalog' while simulating dates.
    """
    when = pd.Timestamp(when)
    pool_t = (pd.Timestamp(pool_at) if pool_at is not None else when.normalize()).value
    prof_t = (pd.Timestamp(profile_at) if profile_at is not None else when).value
    cat = inter.catalog
    pool, pop_n = inter.pool(pool_t, pool_days)

    aff_n = None
    if len(pool):
        u = inter.user_profile(user, prof_t)
        if u is not None:
            aff_n = _minmax(u[cat.T[pool]].mean(axis=1))

    ev, _, ramp = ramp_for(when)
    lift_n = None
    if ev is not None and len(pool):
        lift_n = _minmax(lifts.lift[ev.name][cat.T[pool]].mean(axis=1))
    return Context(pool, pop_n, aff_n, lift_n, ramp, ev.name if ev else None)


def score(ctx: Context, model: str, alpha_max: float = 0.5, gamma: float = 0.5) -> np.ndarray:
    alpha = alpha_max * ctx.ramp if ctx.lift_n is not None else 0.0
    personal = ctx.pop_n if ctx.aff_n is None else (1 - gamma) * ctx.pop_n + gamma * ctx.aff_n
    if model == "popularity":
        return ctx.pop_n
    if model == "personalized":
        return personal
    if model == "seasonal_popularity":
        base = ctx.pop_n
    elif model == "personalized_seasonal":
        base = personal
    else:
        raise ValueError(model)
    return (1 - alpha) * base + alpha * ctx.lift_n if alpha > 0 else base


class SeasonalRecommender:
    """Convenience wrapper used by the demo app: fit on all data, then recommend(user, date)."""

    def __init__(self, df: pd.DataFrame, alpha_max: float = 0.5, gamma: float = 0.5,
                 pool_days: int = 60, smooth: float = 20.0):
        self.catalog = Catalog(df)
        self.inter = Interactions(df, self.catalog)
        self.data_end = df["event_time"].max() + pd.Timedelta(seconds=1)
        self.lifts = EventLifts(self.inter, self.data_end, smooth=smooth)
        self.alpha_max, self.gamma, self.pool_days = alpha_max, gamma, pool_days

    def recommend(self, user_id, when, k: int = 10, seasonal: bool = True,
                  pool_at=None, profile_at=None) -> pd.DataFrame:
        ctx = build_context(self.inter, self.lifts, user_id, when, pool_at, profile_at, self.pool_days)
        if len(ctx.pool) == 0:
            return pd.DataFrame()
        model = "personalized_seasonal" if seasonal else "personalized"
        s = score(ctx, model, self.alpha_max, self.gamma)
        top = np.argsort(-s)[:k]
        out = self.catalog.items.iloc[ctx.pool[top]].copy()
        out["score"] = s[top]
        out["popularity"] = ctx.pop_n[top]
        out["user_affinity"] = ctx.aff_n[top] if ctx.aff_n is not None else np.nan
        out["event_fit"] = ctx.lift_n[top] if ctx.lift_n is not None else np.nan
        return out.reset_index()