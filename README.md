# Seasonal-aware jewelry recommender

Recommendations are **personalized every day** (from each user's past purchases) and, when a
special day is near (Valentine's Day, Black Friday, Christmas, Halloween, Women's Day, Mother's
Day), they are **blended with what typically sells for that event**.

```
final_score(user, item, date) = (1 - alpha) * personal(user, item) + alpha * event(item, event)
personal = (1 - gamma) * recent_popularity + gamma * user_affinity
alpha    = alpha_max * ramp(date)     # 0 on normal days -> 1 on the event day
```

* **user_affinity**: share of the user's past purchases with the same category / metal / gem / colour / gender / price bucket / brand code.
* **event**: log-lift of the item's attributes inside the same event window in *past years* (attributes, not product ids, because jewelry products expire).
* **candidates**: products sold in the last 60 days (the "live catalog").

## Setup

```bash
pip install -r requirements.txt
# put the ORIGINAL jewelry CSV (never re-saved from Excel) at data/raw/jewelry.csv
python -m pytest tests -q                       # 4 tests, ~2 s
python run_experiments.py --data data/raw/jewelry.csv
streamlit run app/app.py
```

`run_experiments.py` prints a data summary, tunes `gamma` / `alpha_max` on the validation period,
evaluates on the test period and writes:

| output | what it is |
|---|---|
| `results/test_metrics.csv` | main table: 4 models x (event windows / normal days) |
| `results/test_metrics_by_event.csv` | same, per event |
| `results/test_metrics_by_user_history.csv` | users with vs. without history |
| `results/test_significance.csv` | paired-bootstrap 95% CI of (ours - personalized) |
| `results/event_top_attributes.csv` | which attributes each event favours (great for the report) |
| `reports/figures/*.png` | weekly orders + events, alpha sweep, model comparison, gain per event |

Defaults: validation = 2020-07-01 .. 2020-12-31, test = 2021-01-01 .. end. Change with `--val-start`, `--test-start`.

## Models compared

| label | idea |
|---|---|
| `popularity` | most-sold items of the last 60 days |
| `seasonal_popularity` | popularity + event score (no personalization) |
| `personalized` | popularity + user affinity |
| `personalized_seasonal` | **ours**: personalized + event score near special days |

## Evaluation (all time-based, no leakage)

For each test purchase we rank the live catalog using only data from before that purchase.
`hit@K` / `ndcg@K` = exact product in the top K (rows whose product was already in the catalog; `pool_coverage` = share of such rows).
`seg@K` = an item with the same category|metal|gem is in the top K (counted for every row).
Results are reported separately for **event windows** and **normal days**: the claim to demonstrate is "season features help in event windows and do not hurt on normal days".

## Report outline (suggested)

1. Problem and research question: do event features improve recommendations?
2. Data: REES46 jewelry, columns, 3 years, one-time-buyer share, weekly plot with event lines (`01_*.png`).
3. Method: formula above, event calendar, attribute lift, why attributes and not product ids.
4. Experimental setup: time split, candidates, metrics, baselines, tuning on validation.
5. Results: `test_metrics.csv`, per-event chart, bootstrap CI, alpha sweep.
6. Demo screenshots (same user, normal day vs. Valentine's / Black Friday).
7. Limitations: purchases only (no views/clicks), many one-time buyers, new products can't be recommended by id, windows and alpha chosen by hand/validation, only ~3 cycles per event.
8. Future work: LightGBM ranker with `days_to_event` features, sequence models, real product names/images.

## Files

```
src/data.py          load (no header), time split, summary
src/events.py        event calendar + ramp(date)           <- edit EVENTS to add/remove days
src/recommender.py   catalog, history, event lifts, scoring, SeasonalRecommender
src/evaluate.py      protocol, metrics, bootstrap
run_experiments.py   full study
app/app.py           Streamlit demo (simulated date picker)
tests/               unit tests + synthetic data generator (for pipeline testing only)
```
