# Seasonal-Aware Jewelry Recommender

A recommender that suggests jewelry to each shopper from their own purchase history, and adapts
when a special day is near (Valentine's Day, Women's Day, Mother's Day, Halloween, Black Friday,
Christmas). It is built and tested on the real **REES46 Jewelry** purchase data and shown in a
demo store called **Éclat**.

> Course project. The store, its prices and its event discounts are a demo; no real orders are taken.

## What it found

| Question | Answer |
|---|---|
| Does a shopper's own history help? | **Yes.** About 2x better than a popularity baseline (NDCG@10 0.058 vs 0.026 on normal days, test year 2021). |
| Does knowing a special day is near help the ranking? | **Not measurably.** Gain in NDCG@10 is +0.0002 (95% interval -0.0004 to +0.0009). |
| Does the season change anything? | **Purchase volume**, up to about 2.5x around Black Friday 2021 and 2.25x at Christmas 2020, but not which items rank higher. |

The full tables are in `results/` and the figures in `reports/figures/`.

## How the model works

Each product on sale (sold in the last 60 days) gets a score, and the top items are shown:

```
score    = (1 - alpha) * personal + alpha * event
personal = (1 - gamma) * popularity + gamma * affinity
alpha    = alpha_max * ramp(date)      # 0 on normal days, rises to 1 on the special day
```

- **popularity**: how often the product sold recently. It is the only signal for a new visitor.
- **affinity**: how much the product looks like the shopper's past purchases (category, brand, gender, colour, metal, gem, price bucket).
- **event**: how over-represented the product's attributes were in the same special-day window in earlier years. Attributes are used instead of product ids because jewelry products come and go.

No future data is used: a shopper's history and the event scores only use data from before the moment being predicted.

## Project structure

```
seasonal-recsys/
├── app/
│   ├── app.py            # the Streamlit demo (shop, cart, model check)
│   ├── ui.py             # HTML/CSS pieces of the store
│   └── assets/           # banners (banner_*.jpg) and product photos (products/*.jpg)
├── src/
│   ├── data.py           # loads the CSV, checks it, splits by date
│   ├── events.py         # special days and their windows
│   ├── recommender.py    # catalog, event scores, the scoring blend
│   └── evaluate.py       # evaluation rows, metrics, paired bootstrap
├── tests/                # automated tests (use a small fake dataset, not the real one)
├── run_experiments.py    # the full study: tuning, test evaluation, tables, figures
├── results/              # CSV tables written by run_experiments.py
├── reports/figures/      # PNG charts written by run_experiments.py
├── data/raw/             # put jewelry.csv here (not committed)
├── .streamlit/config.toml  # forces the light theme for the demo
└── requirements.txt
```

## Setup

Python 3.10 or newer.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

**Data.** Download the REES46 Jewelry file from the
[Kaggle dataset page](https://www.kaggle.com/datasets/mkechinov/ecommerce-purchase-history-from-jewelry-store)
and save it as `data/raw/jewelry.csv`. Use the original file: re-saving it in Excel destroys the
19-digit ids (they become `1.5E+18`), and the loader stops with an error if it sees that.
The file has no header row; the loader names the columns.

## Run

```bash
python -m pytest tests -q                          # 4 quick tests
python run_experiments.py --data data/raw/jewelry.csv
streamlit run app/app.py                           # run from this folder
```

`run_experiments.py` takes several minutes. It tunes the settings on the validation period, then
evaluates once on the test period and writes:

| File | Content |
|---|---|
| `results/test_metrics.csv` | Main table: 4 models, event windows vs. normal days |
| `results/test_metrics_by_event.csv` | The same, for each special day |
| `results/test_metrics_by_user_history.csv` | Shoppers with and without history |
| `results/test_significance.csv` | Paired-bootstrap 95% intervals for the season's gain |
| `results/event_volume_uplift.csv` | Purchase volume in each event window vs. normal days |
| `results/tuning_*.csv` | Validation sweeps for gamma, alpha and smoothing |
| `results/event_top_attributes.csv` | Which attributes each special day favours |
| `reports/figures/*.png` | Weekly orders with events, alpha sweep, model comparison, gain per event |

## Evaluation

- **Time split:** train before 2020-07-01, validation 2020-07 to 2020-12, test 2021 (data ends 2021-12-01).
- **Task:** for each real purchase, rank the products on sale that day for that shopper, using only data from before the purchase, and see where the real product lands.
- **Metrics:** `hit@10` (the bought product is in the top 10), `ndcg@10` (the same, but a higher rank counts more), `seg@10` (the top 10 holds the same category, metal and gem).
- **Groups:** reported separately for event windows and normal days.
- **Models compared:** `popularity`, `seasonal_popularity`, `personalized`, `personalized_seasonal`.

## The demo store

Run `streamlit run app/app.py` and open the address it prints.

- **Shop:** set the shopper and a **simulated date** in the *Demo controls* panel (try 5 Mar, 25 Nov, 20 Dec). The banner, sale prices and recommendations follow the date. Items have a details window and an add-to-cart button.
- **Cart:** quantities, promo code `ECLAT10`, shipping and a checkout form. No payment data is collected.
- **Model check:**
  - *Try the model*: the same shopper on two dates, with the items that entered or left.
  - *Hidden-purchase test*: hides a real purchase and shows where each model ranks it.
  - *Quick evaluation*: the test on a small sample from a date you choose, with a verdict on whether the season helps.

**Not learned from data:** event discounts (for example 20% on Black Friday) are a store rule in
`app/ui.py` (`DEALS`); the dataset has no discount information. The *Season strength* slider is set
higher than the tuned value so the effect is visible. Product photos are illustrative stock images
picked by jewelry type, because the dataset has no pictures; they belong to their owners and are
used only for this non-commercial project.

## Limitations

- 62.6% of shoppers bought only once, so most have little history.
- Only purchases are available (no views or carts).
- Each event has two or three past occurrences, and the data ends before Christmas 2021, so Christmas is not in the test period.
- The evaluation is offline: it predicts past purchases and does not measure real shoppers.
- Event-day prices in the data probably include promotions, so part of the volume increase may come from discounts.

## Credits

Data: REES46 e-commerce jewelry purchase history (Kaggle). Built with pandas, NumPy, matplotlib and Streamlit.
