"""Evaluation protocol.

For every purchase in the evaluation period we pretend we are recommending right before it:
  * candidates = products that sold in the previous `pool_days` days (the "live catalog"),
  * user history = that user's purchases strictly before the purchase,
  * target = the product (and the "segment" category|metal|gem) actually bought.

Metrics (one purchase per row, so HitRate@K == Recall@K):
  hit@K, ndcg@K : the exact product is in the top K. Only rows whose product was already in
                  the live catalog are counted (brand-new products cannot be recommended;
                  the share of such rows is reported as `pool_coverage`).
  seg@K         : some item in the top K has the same category|metal|gem as the purchase.
                  Counted for all rows - more forgiving, matches how jewelry shoppers browse.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .events import ramp_for
from .recommender import build_context, score

KS = (5, 10, 20)


def make_eval_rows(df: pd.DataFrame, catalog, start, end, max_per_group: int = 10_000, seed: int = 0):
    d = df[(df.event_time >= pd.Timestamp(start)) & (df.event_time < pd.Timestamp(end))].copy()
    d["item"] = catalog.index.get_indexer(d["product_id"])
    d["seg"] = catalog.segment[d["item"].to_numpy()]
    days = d["event_time"].dt.normalize()
    info = {x: ramp_for(x)[0] for x in days.drop_duplicates()}
    d["event"] = [info[x].name if info[x] else "normal" for x in days]
    d["in_window"] = d["event"] != "normal"
    parts = []
    for _, g in d.groupby("in_window"):
        parts.append(g.sample(max_per_group, random_state=seed) if len(g) > max_per_group else g)
    return pd.concat(parts).sort_values("event_time").reset_index(drop=True)


def evaluate_rows(inter, lifts, rows: pd.DataFrame, configs: list[dict],
                  pool_days: int = 60, ks=KS, seed: int = 0) -> pd.DataFrame:
    """configs: [{'label':..., 'model':..., 'alpha_max':..., 'gamma':...}, ...]
    Returns one record per (row, config)."""
    rng = np.random.default_rng(seed)
    seg_of = inter.catalog.segment
    kmax = max(ks)
    records = []
    for row_id, r in enumerate(rows.itertuples(index=False)):
        ctx = build_context(inter, lifts, r.user_id, r.event_time, pool_days=pool_days)
        if len(ctx.pool) == 0:
            continue
        pos = int(np.searchsorted(ctx.pool, r.item))
        in_pool = pos < len(ctx.pool) and ctx.pool[pos] == r.item
        noise = rng.random(len(ctx.pool)) * 1e-9          # random tie-breaking, same for all models
        for cfg in configs:
            s = score(ctx, cfg["model"], cfg["alpha_max"], cfg["gamma"]) + noise
            order = np.argsort(-s)[:kmax]
            top_seg = seg_of[ctx.pool[order]]
            rec = {"row_id": row_id, "label": cfg["label"], "event": r.event,
                   "in_window": bool(r.in_window), "warm": ctx.aff_n is not None, "in_pool": bool(in_pool)}
            rank = int((s > s[pos]).sum()) + 1 if in_pool else None
            for k in ks:
                rec[f"hit@{k}"] = float(rank <= k) if in_pool else np.nan
                rec[f"ndcg@{k}"] = (1.0 / np.log2(rank + 1) if rank <= k else 0.0) if in_pool else np.nan
                rec[f"seg@{k}"] = float(r.seg in top_seg[:k])
            records.append(rec)
    return pd.DataFrame(records)


def summarize(res: pd.DataFrame, by) -> pd.DataFrame:
    metric_cols = [c for c in res.columns if c.split("@")[0] in ("hit", "ndcg", "seg") and "@" in c]
    g = res.groupby(by, sort=False)
    out = g[metric_cols].mean()
    out["n_rows"] = g.size()
    out["pool_coverage"] = g["in_pool"].mean()
    return out


def paired_bootstrap(res: pd.DataFrame, a: str, b: str, metric: str, mask=None,
                     n_boot: int = 2000, seed: int = 0):
    """Mean of (metric[a] - metric[b]) over the same rows, with a 95% bootstrap interval."""
    d = res if mask is None else res[mask]
    pv = d[d.label.isin([a, b])].pivot(index="row_id", columns="label", values=metric).dropna()
    if len(pv) < 10:
        return np.nan, np.nan, np.nan, len(pv)
    diff = (pv[a] - pv[b]).to_numpy()
    rng = np.random.default_rng(seed)
    boots = rng.choice(diff, size=(n_boot, len(diff)), replace=True).mean(axis=1)
    return float(diff.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), len(diff)
