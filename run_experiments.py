"""Run the whole study: EDA -> tune on validation -> evaluate on test -> tables and figures.

    python run_experiments.py --data data/raw/jewelry.csv

Outputs go to results/ (CSV tables) and reports/figures/ (PNG charts).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data import load, summary  # noqa: E402
from src.evaluate import evaluate_rows, make_eval_rows, paired_bootstrap, summarize  # noqa: E402
from src.events import EVENTS, event_calendar  # noqa: E402
from src.recommender import Catalog, EventLifts, Interactions, MODELS  # noqa: E402

NICE = {"popularity": "Popularity", "seasonal_popularity": "Popularity + season",
        "personalized": "Personalized", "personalized_seasonal": "Personalized + season (ours)"}


def cfg(label, model, alpha_max=0.0, gamma=0.5):
    return {"label": label, "model": model, "alpha_max": alpha_max, "gamma": gamma}


def eda_plot(df, figdir):
    weekly = df.set_index("event_time").resample("W").size()
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(weekly.index, weekly.values, lw=1.4, color="#1f4e79")
    for d, name in event_calendar(df.event_time.min().year, df.event_time.max().year):
        d = pd.Timestamp(d)
        if weekly.index.min() <= d <= weekly.index.max():
            ax.axvline(d, color="#c0392b", alpha=0.35, lw=1)
            ax.text(d, ax.get_ylim()[1] * 0.98, name.replace("_", " "), rotation=90, va="top", ha="right", fontsize=7, color="#c0392b")
    ax.set_title("Weekly purchases with special days (red lines)")
    ax.set_ylabel("purchases / week")
    fig.tight_layout()
    fig.savefig(figdir / "01_weekly_orders_events.png", dpi=150)
    plt.close(fig)


def event_uplift(df, out):
    """Volume effect of each event: mean daily purchases in the window vs. the 45 days before it.
    Approximate (the baseline can include growth or another event) but easy to explain."""
    daily = df.set_index("event_time").resample("D").size()
    # days belonging to ANY event window (+3 days after) are excluded from the baseline
    in_event = pd.Series(False, index=daily.index)
    for ev in EVENTS:
        for y in range(daily.index.min().year, daily.index.max().year + 1):
            d = pd.Timestamp(ev.date(y))
            in_event.loc[d - pd.Timedelta(days=ev.window): d + pd.Timedelta(days=3)] = True
    rows = []
    for ev in EVENTS:
        for y in range(daily.index.min().year, daily.index.max().year + 1):
            d = pd.Timestamp(ev.date(y))
            w0 = d - pd.Timedelta(days=ev.window)
            if w0 - pd.Timedelta(days=45) < daily.index.min() or d > daily.index.max():
                continue
            win = daily[w0:d].mean()
            around = daily[w0 - pd.Timedelta(days=45): d + pd.Timedelta(days=45)]
            base = around[~in_event.reindex(around.index)].median()
            rows.append({"event": ev.name, "year": y, "window_daily_mean": round(win, 1),
                         "baseline_daily_median": round(base, 1),
                         "uplift_x": round(win / base, 2) if base > 0 else np.nan})
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "event_volume_uplift.csv", index=False)
    return tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/jewelry.csv")
    ap.add_argument("--val-start", default="2020-07-01")
    ap.add_argument("--test-start", default="2021-01-01")
    ap.add_argument("--pool-days", type=int, default=60)
    ap.add_argument("--max-rows", type=int, default=8000, help="max evaluated rows per group (event / normal)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--figs", default="reports/figures")
    args = ap.parse_args()
    out, figs = Path(args.out), Path(args.figs)
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    print("== 1. Load data")
    df = load(args.data)
    print(summary(df))
    eda_plot(df, figs)
    print("\nVolume uplift per event (window daily mean / median of normal days within +-45 days):")
    print(event_uplift(df, out).to_string(index=False))

    print("\n== 2. Build catalog / history")
    train_df = df[df.event_time < pd.Timestamp(args.val_start)]
    catalog = Catalog(df, price_cuts=np.nanquantile(train_df["price"], [0.2, 0.4, 0.6, 0.8]))
    inter = Interactions(df, catalog)
    print(f"{catalog.N:,} products, {catalog.V} attribute values")

    print("\n== 3. Tune gamma and alpha_max on the validation period")
    lifts_val = EventLifts(inter, args.val_start)
    print("event occurrences seen before validation:", lifts_val.n_occurrences)
    val_all = make_eval_rows(df, catalog, args.val_start, args.test_start, max_per_group=1000)
    gammas = [0.0, 0.25, 0.5, 0.75, 1.0]
    res_g = evaluate_rows(inter, lifts_val, val_all, [cfg(f"g={g}", "personalized", 0, g) for g in gammas], args.pool_days)
    gtab = summarize(res_g, "label")[["ndcg@10", "hit@10", "seg@10"]]
    print(gtab.round(4))
    best_gamma = gammas[int(np.argmax([gtab.loc[f"g={g}", "ndcg@10"] for g in gammas]))]

    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
    val_ev = val_all[val_all.in_window]
    if len(val_ev) < 30:
        print("WARNING: very few validation rows inside event windows -> alpha tuning is unreliable")
    smooths = [5.0, 20.0, 50.0, 200.0]
    best = (-1.0, smooths[0], 0.0)
    sweep_rows = []
    for sm in smooths:
        lv = EventLifts(inter, args.val_start, smooth=sm)
        ra = evaluate_rows(inter, lv, val_ev,
                           [cfg(f"a={a}", "personalized_seasonal", a, best_gamma) for a in alphas], args.pool_days)
        t = summarize(ra, "label")[["ndcg@10", "hit@10", "seg@10"]]
        for a in alphas:
            sweep_rows.append({"smooth": sm, "alpha_max": a, **t.loc[f"a={a}"].to_dict()})
            if t.loc[f"a={a}", "ndcg@10"] > best[0] + 1e-9:
                best = (float(t.loc[f"a={a}", "ndcg@10"]), sm, a)
    sweep = pd.DataFrame(sweep_rows)
    print(sweep.round(4).to_string(index=False))
    best_smooth, best_alpha = best[1], best[2]
    print(f"-> best gamma = {best_gamma}, best smooth = {best_smooth}, best alpha_max = {best_alpha}"
          + ("   (alpha 0 = the season did not help on validation)" if best_alpha == 0 else ""))
    atab = sweep[sweep.smooth == best_smooth].copy()
    atab.index = ["a=" + str(a) for a in atab.alpha_max]
    gtab.to_csv(out / "tuning_gamma.csv")
    sweep.to_csv(out / "tuning_alpha_smooth.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(alphas, [atab.loc[f"a={a}", "ndcg@10"] for a in alphas], marker="o", label="NDCG@10 (exact product)")
    ax.plot(alphas, [atab.loc[f"a={a}", "seg@10"] for a in alphas], marker="s", label="SegHit@10 (type of jewelry)")
    ax.set_xlabel("alpha_max (weight of the seasonal score)")
    ax.set_title("Validation, event windows only")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs / "02_alpha_sweep.png", dpi=150)
    plt.close(fig)

    print("\n== 4. Final evaluation on the test period")
    lifts_test = EventLifts(inter, args.test_start, smooth=best_smooth)
    print("event occurrences seen before test:", lifts_test.n_occurrences)
    test_rows = make_eval_rows(df, catalog, args.test_start, df.event_time.max() + pd.Timedelta(days=1),
                               max_per_group=args.max_rows)
    configs = [cfg("popularity", "popularity"),
               cfg("seasonal_popularity", "seasonal_popularity", best_alpha, best_gamma),
               cfg("personalized", "personalized", 0, best_gamma),
               cfg("personalized_seasonal", "personalized_seasonal", best_alpha, best_gamma)]
    res = evaluate_rows(inter, lifts_test, test_rows, configs, args.pool_days)
    res["group"] = np.where(res.in_window, "event windows", "normal days")
    res.drop(columns=["row_id"]).to_csv(out / "test_row_level.csv", index=False)

    main_tab = summarize(res, ["group", "label"])
    main_tab.to_csv(out / "test_metrics.csv")
    print(main_tab.round(4).to_string())

    ev_tab = summarize(res[res.in_window], ["event", "label"])
    ev_tab.to_csv(out / "test_metrics_by_event.csv")
    warm_tab = summarize(res, ["warm", "group", "label"])
    warm_tab.to_csv(out / "test_metrics_by_user_history.csv")

    print("\n== 5. Is the gain real? (paired bootstrap, personalized_seasonal - personalized)")
    boot = []
    for grp in ["event windows", "normal days"]:
        for metric in ["ndcg@10", "hit@10", "seg@10"]:
            m, lo, hi, n = paired_bootstrap(res, "personalized_seasonal", "personalized", metric, res.group == grp)
            boot.append({"group": grp, "metric": metric, "mean_diff": m, "ci_low": lo, "ci_high": hi, "n_rows": n})
    boot = pd.DataFrame(boot)
    boot.to_csv(out / "test_significance.csv", index=False)
    print(boot.round(4).to_string(index=False))

    # what does each event favour? (fitted on all data, for the report)
    lifts_all = EventLifts(inter, df.event_time.max() + pd.Timedelta(days=1), smooth=best_smooth)
    rows = []
    for ev in EVENTS:
        lv = lifts_all.lift[ev.name][: catalog.V]
        for i in np.argsort(-lv)[:8]:
            rows.append({"event": ev.name, "attribute": catalog.token_names[i], "log_lift": round(float(lv[i]), 3)})
    pd.DataFrame(rows).to_csv(out / "event_top_attributes.csv", index=False)

    # ---- figures
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    for ax, metric, title in zip(axes, ["ndcg@10", "seg@10"], ["NDCG@10 (exact product)", "SegHit@10 (type of jewelry)"]):
        piv = main_tab[metric].unstack("group")[["event windows", "normal days"]].loc[MODELS]
        piv.index = [NICE[m] for m in piv.index]
        piv.plot.bar(ax=ax, rot=20, color=["#c0392b", "#1f4e79"])
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylim(0, float(piv.to_numpy().max()) * 1.3)
        ax.legend(loc="upper left", ncol=2, fontsize=8, title=None)
    fig.tight_layout()
    fig.savefig(figs / "03_models_event_vs_normal.png", dpi=150)
    plt.close(fig)

    gain = []
    for ev_name, g in res[res.in_window].groupby("event"):
        m, lo, hi, n = paired_bootstrap(res, "personalized_seasonal", "personalized", "ndcg@10", (res.event == ev_name))
        gain.append((ev_name, m, n))
    if gain:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar([g[0].replace("_", " ") for g in gain], [g[1] for g in gain], color="#2e7d32")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title("NDCG@10 gain from season features, per event (test)")
        plt.setp(ax.get_xticklabels(), rotation=25)
        fig.tight_layout()
        fig.savefig(figs / "04_gain_per_event.png", dpi=150)
        plt.close(fig)

    print(f"\nDone. Tables in {out}/, figures in {figs}/")


if __name__ == "__main__":
    main()