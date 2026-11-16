"""Store-style demo:  streamlit run app/app.py   (run it from the seasonal-recsys folder)

Pages (buttons at the top):
  Shop        - banner and recommendations that follow the simulated date; category filter;
                "View details" window with "Add to cart".
  Cart        - quantities, remove, promo code (ECLAT10), shipping, checkout form, order confirmation.
  Model check - test the recommender on real past purchases (one shopper, or a quick evaluation).
Pictures: put files in app/assets/ (see the docstring of app/ui.py for the names).
"""
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

APP = Path(__file__).resolve().parent
ROOT = APP.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(APP))

import importlib  # noqa: E402

import ui  # noqa: E402

ui = importlib.reload(ui)  # always use the ui.py on disk (Streamlit can keep an old copy in memory)
if not hasattr(ui, "announce_html"):
    st.error(f"The ui.py that was loaded is out of date: {ui.__file__}. Replace it with the newest ui.py and restart.")
    st.stop()
from src.data import load  # noqa: E402
from src.evaluate import evaluate_rows, make_eval_rows, paired_bootstrap, summarize  # noqa: E402
from src.events import ramp_for  # noqa: E402
from src.recommender import MODELS, EventLifts, SeasonalRecommender, build_context, score  # noqa: E402

# settings used by the "Model check" page (the values tuned on the validation period)
CHECK_ALPHA, CHECK_GAMMA, CHECK_SMOOTH = 0.1, 0.75, 200.0
NICE = {"popularity": "Popularity", "seasonal_popularity": "Popularity + season",
        "personalized": "Personalized", "personalized_seasonal": "Personalized + season"}

st.set_page_config(page_title="Éclat | Jewelry", page_icon="💎", layout="wide", initial_sidebar_state="expanded")
st.markdown(ui.css(), unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading data and fitting the model...")
def get_model(path: str):
    df = load(path)
    return SeasonalRecommender(df, alpha_max=0.5, gamma=0.75), df


for key, default in [("cart", []), ("orders", []), ("last_order", None), ("page", "shop")]:
    if key not in st.session_state:
        st.session_state[key] = default


def goto(page: str):
    st.session_state["page"] = page
    st.rerun()


def cart_count() -> int:
    return sum(line["qty"] for line in st.session_state["cart"])


def add_to_cart(row: pd.Series):
    pid = str(row["product_id"])
    for line in st.session_state["cart"]:
        if line["pid"] == pid:
            line["qty"] = min(line["qty"] + 1, 10)
            return
    price = row.get("price")
    st.session_state["cart"].append({"pid": pid, "row": row.to_dict(), "qty": 1,
                                     "price": float(price) if pd.notna(price) else 0.0})


# ---------------------------------------------------------------- item details (pop-up window)
def _details(row: dict, ev_name, season_on):
    row = pd.Series(row)
    left, right = st.columns([1, 1.15])
    left.markdown(ui.detail_image_html(row), unsafe_allow_html=True)
    right.markdown(ui.detail_html(row, ev_name, season_on), unsafe_allow_html=True)
    if right.button("Add to cart", type="primary", key=f"add_{row['product_id']}"):
        add_to_cart(row)
        st.session_state["toast"] = f"Added to cart: {ui.item_title(row)}"
        st.rerun()


if hasattr(st, "dialog"):
    try:
        open_details = st.dialog("Item details", width="large")(_details)
    except TypeError:  # older Streamlit without the width option
        open_details = st.dialog("Item details")(_details)
else:  # very old Streamlit: show the details inline instead
    open_details = None

# ---------------------------------------------------------------- sidebar
st.sidebar.header("Settings")
data_path = st.sidebar.text_input("Data file", str(ROOT / "data" / "raw" / "jewelry.csv"))
_n = ui.photo_count()
st.sidebar.caption(f"Product photos found: {_n} of {len(ui.PHOTOS)}" + ("" if _n else f"  (expected in {ui.ASSETS / 'products'})"))
if not Path(data_path).exists():
    st.error(f"File not found: {data_path}")
    st.stop()
model, df = get_model(data_path)

counts = df.user_id.value_counts()
repeat_users = counts[(counts >= 3) & (counts < 200)].head(300).index.tolist()  # skip bulk/outlier accounts


def shopper_label(u):
    return u if u.startswith("New") else f"Customer …{u[-5:]}  ({counts[u]} purchases)"


# The demo controls live on the page itself (not in the sidebar), so they are always visible.
top = st.container()  # the store's top bar is drawn into this container further down
with st.expander("Demo controls: shopper, simulated date, season strength", expanded=True):
    c1, c2, c3 = st.columns([2, 1.3, 1])
    with c1:
        user_choice = st.selectbox("Shopper", ["New visitor (no history)"] + repeat_users,
                                   format_func=shopper_label, key="ctl_user")
    with c2:
        when = st.date_input("Simulated date", date.today(), key="ctl_date",
                             help="Try Mar 5 (Women's Day), Nov 25 (Black Friday) or Dec 20 (Christmas).")
    with c3:
        k = st.slider("Items to show", 4, 16, 8, step=4, key="ctl_items")
    c4, c5 = st.columns([2, 1.3])
    with c4:
        model.alpha_max = st.slider(
            "Season strength (demo)", 0.0, 1.0, 0.5, step=0.05, key="ctl_alpha",
            help="The evaluation found only a small ranking effect of the season, so the demo exaggerates it "
                 "to make the change visible. Use 0.1 to see the value tuned on validation data.")
    with c5:
        compare = st.checkbox("Compare with / without season", value=False, key="ctl_compare")
user = None if user_choice.startswith("New") else user_choice

msg = st.session_state.get("toast")
if msg:
    st.toast(msg)
    st.session_state["toast"] = ""

now = model.data_end  # catalog and history "as of today"; only the season is simulated
ts = pd.Timestamp(when)
ev, days_to, ramp = ramp_for(when)
ev_name = ev.name if ev else None
label = ui.EVENT_LABEL.get(ev_name, "") if ev_name else ""


# ================================================================ page: shop
def get_recs(nav: str, seasonal: bool) -> pd.DataFrame:
    # rank a big slice first, then filter by category, so a category filter never comes back empty by accident
    recs = model.recommend(user, ts, k=600, seasonal=seasonal, pool_at=now, profile_at=now)
    if recs.empty:
        return recs
    recs = ui.apply_deal(recs, ev_name if ramp > 0 else None)  # event sale: same prices with or without the model's season
    return ui.filter_recs(recs, nav, k)


def show_grid(recs: pd.DataFrame, seasonal: bool, columns: int, prefix: str):
    if recs.empty:
        st.info("No items of this type in the current catalog.")
        return
    rows = [r for _, r in recs.iterrows()]
    for i in range(0, len(rows), columns):
        for col, r in zip(st.columns(columns), rows[i:i + columns]):
            with col:
                st.markdown(ui.card_html(r, ui.badge_for(r, label, seasonal)), unsafe_allow_html=True)
                if st.button("View details", key=f"{prefix}_{r['product_id']}"):
                    if open_details is not None:
                        open_details(r.to_dict(), ev_name, seasonal)
                    else:
                        st.session_state["detail"] = (r.to_dict(), ev_name, seasonal)
                        st.rerun()


def page_shop():
    if hasattr(st, "pills"):
        nav = st.pills("Browse", ui.NAV, default="All", label_visibility="collapsed", key="nav_pills") or "All"
    else:
        nav = st.radio("Browse", ui.NAV, horizontal=True, label_visibility="collapsed")
    st.markdown(ui.banner_html(ev_name, days_to), unsafe_allow_html=True)
    st.markdown(ui.season_strip(ev_name, days_to, model.alpha_max * ramp), unsafe_allow_html=True)

    if open_details is None and st.session_state.get("detail"):
        with st.container(border=True):
            _details(*st.session_state["detail"])
            if st.button("Close"):
                st.session_state["detail"] = None
                st.rerun()

    if compare:
        left, right = st.columns(2)
        for col, seasonal, title in [(left, False, "Personal picks"), (right, True, "Personal picks + season")]:
            with col:
                st.markdown(ui.section_title(title), unsafe_allow_html=True)
                show_grid(get_recs(nav, seasonal), seasonal, 2, "cmp_season" if seasonal else "cmp_plain")
    else:
        title = "Recommended for you" if user else "Popular right now"
        sub = ("Based on your past purchases" + (f" and {ui.EVENT_LABEL[ev_name]}" if ev_name else "")) if user \
            else "New here? These are the pieces shoppers love at the moment."
        st.markdown(ui.section_title(title, sub), unsafe_allow_html=True)
        show_grid(get_recs(nav, True), True, 4, "main")
    st.markdown(ui.trust_html(), unsafe_allow_html=True)


# ================================================================ page: cart + checkout
def page_cart():
    st.markdown(ui.section_title("Your cart", "Review your items, then check out."), unsafe_allow_html=True)
    cart = st.session_state["cart"]
    order = st.session_state.get("last_order")
    if not cart:
        if order:
            st.markdown(ui.order_html(order), unsafe_allow_html=True)
        else:
            st.info("Your cart is empty. Open the shop and add something you like.")
        if st.button("Continue shopping", key="cart_continue"):
            st.session_state["last_order"] = None
            goto("shop")
        return

    promo = st.session_state.get("promo", "")
    subtotal = sum(line["price"] * line["qty"] for line in cart)
    discount = round(0.10 * subtotal, 2) if promo.strip().upper() == "ECLAT10" else 0.0
    shipping = 0.0 if subtotal - discount >= 100 else 7.90
    total = subtotal - discount + shipping

    left, right = st.columns([1.8, 1])
    with left:
        for line in list(cart):
            row = pd.Series(line["row"])
            c1, c2, c3, c4 = st.columns([4, 1.3, 1.3, 1.1])
            c1.markdown(ui.cart_item_html(row, line["price"]), unsafe_allow_html=True)
            qty = c2.number_input("Quantity", min_value=1, max_value=10, value=int(line["qty"]),
                                  key=f"qty_{line['pid']}", label_visibility="collapsed")
            line["qty"] = int(qty)
            c3.markdown(f'<div class="ec-ltotal">{ui.money(line["price"] * line["qty"])}</div>', unsafe_allow_html=True)
            if c4.button("Remove", key=f"rm_{line['pid']}"):
                st.session_state["cart"] = [x for x in cart if x["pid"] != line["pid"]]
                st.rerun()

        st.markdown(ui.section_title("Delivery and payment"), unsafe_allow_html=True)
        with st.form("checkout"):
            name = st.text_input("Full name")
            email = st.text_input("E-mail")
            addr = st.text_input("Address")
            city = st.text_input("City")
            pay = st.radio("Payment method", ["Credit card", "Cash on delivery", "Bank transfer"], horizontal=True)
            st.caption("Demo store: no payment is processed and no card details are collected.")
            submitted = st.form_submit_button("Place order", type="primary")
        if submitted:
            errors = []
            if not name.strip():
                errors.append("Please enter your name.")
            if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email.strip()):
                errors.append("Please enter a valid e-mail address.")
            if not addr.strip() or not city.strip():
                errors.append("Please enter your address and city.")
            for e in errors:
                st.error(e)
            if not errors:
                new_order = {
                    "id": datetime.now().strftime("EC-%y%m%d-%H%M%S"), "name": name.strip(), "email": email.strip(),
                    "address": addr.strip(), "city": city.strip(), "payment": pay, "total": total,
                    "items": [{"title": ui.item_title(pd.Series(x["row"])), "qty": x["qty"], "price": x["price"]} for x in cart],
                }
                st.session_state["orders"].append(new_order)
                st.session_state["last_order"] = new_order
                st.session_state["cart"] = []
                st.rerun()
    with right:
        st.text_input("Promo code", key="promo", placeholder="Try ECLAT10")
        if promo and not discount:
            st.caption("Unknown code. In this demo only ECLAT10 works (10% off).")
        st.markdown(ui.summary_html(subtotal, discount, shipping, total), unsafe_allow_html=True)


# ================================================================ page: model check
def _view(rows: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"date": rows["event_time"].dt.strftime("%Y-%m-%d"),
                        "item": [ui.item_title(r) for _, r in rows.iterrows()],
                        "price": rows["price"].round(2).to_numpy()})
    return out.reset_index(drop=True)


def rec_table(recs: pd.DataFrame) -> pd.DataFrame:
    """Recommendations with the score behind each pick (all scores 0-100)."""
    if recs.empty:
        return pd.DataFrame()

    def pct(s):
        return [None if pd.isna(v) else int(round(100 * v)) for v in s]

    return pd.DataFrame({
        "#": range(1, len(recs) + 1),
        "item": [ui.item_title(r) for _, r in recs.iterrows()],
        "price": recs["price"].round(0).to_numpy(),
        "taste match %": pct(recs["user_affinity"]),
        "popularity %": pct(recs["popularity"]),
        "season fit %": pct(recs["event_fit"]),
    })


def next_event_day(start: date) -> date:
    d = start
    for _ in range(400):
        if ramp_for(d)[2] >= 0.8:
            return d
        d += timedelta(days=1)
    return start


def backtest(user_id, last: pd.Series) -> pd.DataFrame:
    """Hide `last`, recommend as of its timestamp with only earlier data, see where it ranks."""
    t = last["event_time"]
    lifts = EventLifts(model.inter, t, smooth=CHECK_SMOOTH)
    ctx = build_context(model.inter, lifts, user_id, t, pool_days=model.pool_days)
    idx = model.catalog.index.get_indexer([last["product_id"]])[0]
    seg = model.catalog.segment
    out = []
    for m in MODELS:
        if len(ctx.pool) == 0:
            break
        s = score(ctx, m, CHECK_ALPHA, CHECK_GAMMA)
        order = np.argsort(-s)
        top = ctx.pool[order[:10]]
        pos = int(np.searchsorted(ctx.pool, idx))
        in_pool = pos < len(ctx.pool) and ctx.pool[pos] == idx
        out.append({
            "model": NICE[m],
            "rank of the exact item": int((s > s[pos]).sum()) + 1 if in_pool else "not in the 60-day catalog",
            "same type in top 10": "yes" if seg[idx] in seg[top] else "no",
            "top 3 picks": " | ".join(ui.item_title(model.catalog.items.iloc[i]) for i in top[:3]),
        })
    return pd.DataFrame(out)


def page_check():
    st.markdown(ui.section_title("Model check", "Does the recommender find what people really bought?"),
                unsafe_allow_html=True)
    tab0, tab1, tab2 = st.tabs(["Try the model", "Hidden-purchase test", "Quick evaluation"])
    with tab0:
        st.write("See how the recommendations change with the shopper and the date. **Any date works, including "
                 "future ones**: the model only needs the calendar of special days. The season strength is set "
                 "in the sidebar.")
        who = st.selectbox("Shopper", ["New visitor (no history)"] + repeat_users, format_func=shopper_label,
                           key="try_user")
        uid = None if who.startswith("New") else who
        if uid:
            st.markdown("**This shopper's latest purchases**")
            st.dataframe(_view(df[df.user_id == uid].sort_values("event_time").tail(10)), hide_index=True)
        else:
            st.caption("New visitor: no history, so the list comes from what is popular right now (plus the season).")
        n_items = st.slider("Items per list", 5, 15, 8, key="try_n")
        ca, cb = st.columns(2)
        today = date.today()
        da = ca.date_input("Date A", today, key="try_a")
        db = cb.date_input("Date B", next_event_day(today), key="try_b")
        lists = {}
        for col, dd, tag in [(ca, da, "A"), (cb, db, "B")]:
            ev_, days_, ramp_ = ramp_for(dd)
            col.markdown(f"**List {tag}: {dd:%d %b %Y}** - " + (
                f"{ui.EVENT_LABEL[ev_.name]} in {days_} day(s), season weight {model.alpha_max * ramp_:.2f}"
                if ev_ else "a normal day, no season effect"))
            recs = model.recommend(uid, pd.Timestamp(dd), k=n_items, seasonal=True, pool_at=now, profile_at=now)
            lists[tag] = recs
            col.dataframe(rec_table(recs), hide_index=True)
        if not lists["A"].empty and not lists["B"].empty:
            a_ids, b_ids = set(lists["A"]["product_id"]), set(lists["B"]["product_id"])
            only_b = lists["B"][~lists["B"]["product_id"].isin(a_ids)]
            only_a = lists["A"][~lists["A"]["product_id"].isin(b_ids)]
            st.markdown(f"**{len(only_b)} of {len(lists['B'])} items in list B are not in list A.**")
            if len(only_b):
                st.write("Entered on date B: " + "; ".join(ui.item_title(r) for _, r in only_b.head(5).iterrows()))
                st.write("Left from date A: " + "; ".join(ui.item_title(r) for _, r in only_a.head(5).iterrows()))
            else:
                st.caption("Same list on both dates: no special day is near, or the season has little effect on "
                           "this shopper. Try a date close to Women's Day, Black Friday or Christmas, and raise the "
                           "season strength in the sidebar.")
    with tab1:
        st.write("We hide this shopper's **last purchase**, recommend using only what happened **before** it, "
                 "and see where the hidden item lands. Rank 1 means the model's top pick was exactly that item.")
        pick = st.selectbox("Shopper", repeat_users, format_func=shopper_label, key="chk_user")
        rows = df[df.user_id == pick].sort_values("event_time")
        st.markdown("**Bought before**")
        st.dataframe(_view(rows.iloc[:-1]), hide_index=True)
        st.markdown("**Hidden purchase**")
        st.dataframe(_view(rows.iloc[[-1]]), hide_index=True)
        st.markdown("**What each model would have recommended**")
        st.dataframe(backtest(pick, rows.iloc[-1]), hide_index=True)
    with tab2:
        st.write("Re-runs the evaluation from the report on a sample of real purchases: for each purchase we recommend "
                 "using only earlier data and check whether the purchased item (or its type) is in the top 10.")
        n = st.select_slider("Sample size (per group)", options=[200, 500, 1000], value=500)
        first_day, last_day = df.event_time.min().date(), df.event_time.max().date()
        st.caption(f"Real purchases exist from {first_day} to {last_day}: a test needs known purchases to compare "
                   "against, so dates outside this range (such as the future) can't be tested.")
        start = st.date_input("Evaluate purchases from", date(2021, 1, 1), min_value=first_day,
                              max_value=last_day, key="chk_start")
        if st.button("Run quick test", key="run_eval", type="primary"):
            with st.spinner("Evaluating..."):
                start_ts = pd.Timestamp(start)
                rows = make_eval_rows(df, model.catalog, start_ts, df.event_time.max() + pd.Timedelta(days=1),
                                      max_per_group=n)
                lifts = EventLifts(model.inter, start_ts, smooth=CHECK_SMOOTH)
                cfgs = [{"label": m, "model": m, "alpha_max": CHECK_ALPHA, "gamma": CHECK_GAMMA} for m in MODELS]
                res = evaluate_rows(model.inter, lifts, rows, cfgs, model.pool_days)
            res["group"] = np.where(res.in_window, "event windows", "normal days")
            tab = summarize(res, ["group", "label"])[["hit@10", "ndcg@10", "seg@10", "n_rows", "pool_coverage"]]
            shown = tab.copy()
            shown.index = pd.MultiIndex.from_tuples([(g, NICE[m]) for g, m in tab.index])
            st.dataframe(shown.round(4))
            try:
                per, pop = tab.loc[("normal days", "personalized"), "ndcg@10"], tab.loc[("normal days", "popularity"), "ndcg@10"]
                if per > pop:
                    st.success(f"Personalization works: NDCG@10 {per:.3f} vs {pop:.3f} for plain popularity "
                               f"({per / pop:.1f}x better on normal days).")
                else:
                    st.warning(f"Personalization does NOT beat popularity here: NDCG@10 {per:.3f} vs {pop:.3f}. "
                               "Check the data file and the evaluation period.")
            except KeyError:
                pass
            m_, lo, hi, n_ = paired_bootstrap(res, "personalized_seasonal", "personalized", "ndcg@10",
                                              res.group == "event windows")
            if not np.isnan(m_):
                verdict = "a reliable gain" if lo > 0 else "not distinguishable from zero"
                st.info(f"Season effect inside event windows: {m_:+.4f} NDCG@10 (95% interval {lo:+.4f} to {hi:+.4f}): "
                        f"{verdict}.")


# ================================================================ layout
page = st.session_state["page"]
with top:
    st.markdown(ui.announce_html(ev_name if ramp > 0 else None, when), unsafe_allow_html=True)
    try:
        bar = st.columns([4.4, 0.9, 1.1, 1.4], vertical_alignment="center")
    except TypeError:  # older Streamlit without vertical_alignment
        bar = st.columns([4.4, 0.9, 1.1, 1.4])
    bar[0].markdown(ui.header_html(when, date=False), unsafe_allow_html=True)
    for col, key, text in [(bar[1], "shop", "Shop"), (bar[2], "cart", f"Cart ({cart_count()})"),
                           (bar[3], "check", "Model check")]:
        if col.button(text, key=f"nav_{key}", use_container_width=True, type="primary" if page == key else "secondary"):
            goto(key)
    st.markdown(ui.rule_html(), unsafe_allow_html=True)

if page == "cart":
    page_cart()
elif page == "check":
    page_check()
else:
    page_shop()
st.markdown(ui.footer_html(), unsafe_allow_html=True)