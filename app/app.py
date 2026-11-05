"""Streamlit demo:  streamlit run app/app.py

Pick a user and a simulated date. The left column ignores the season, the right column uses it.
Move the date to Feb 12 or Nov 25 and watch the right column change.
"""
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import load  # noqa: E402
from src.events import describe_season, ramp_for  # noqa: E402
from src.recommender import SeasonalRecommender  # noqa: E402

st.set_page_config(page_title="Seasonal jewelry recommender", page_icon="💎", layout="wide")

EMOJI = {"ring": "💍", "earring": "✨", "pendant": "📿", "necklace": "📿", "bracelet": "⛓️",
         "brooch": "🌸", "watch": "⌚", "cufflinks": "👔"}
BG = {"gold": "#f7e8b0", "silver": "#e4e8ee"}


@st.cache_resource(show_spinner="Loading data and fitting the model...")
def get_model(path: str):
    df = load(path)
    return SeasonalRecommender(df), df


def card(row) -> str:
    kind = str(row.get("category_code") or "jewelry").split(".")[-1]
    metal, gem = row.get("metal"), row.get("gem")
    title = " ".join(str(x) for x in [metal, gem, kind] if isinstance(x, str)).title()
    price = f"${row['price']:,.0f}" if pd.notna(row.get("price")) else ""
    bg = BG.get(str(metal), "#efe9e1")
    return (f"<div style='background:{bg};border-radius:12px;padding:14px;margin-bottom:10px;color:#222'>"
            f"<div style='font-size:34px'>{EMOJI.get(kind, '💎')}</div>"
            f"<div style='font-weight:600'>{title}</div>"
            f"<div style='opacity:.7;font-size:13px'>{price}</div></div>")


st.title("💎 Seasonal jewelry recommender")
data_path = st.sidebar.text_input("Data file", str(ROOT / "data" / "raw" / "jewelry.csv"))
if not Path(data_path).exists():
    st.error(f"File not found: {data_path}")
    st.stop()
model, df = get_model(data_path)

counts = df.user_id.value_counts()
top_users = counts.head(300).index.tolist()
user_choice = st.sidebar.selectbox(
    "User", ["(new user, no history)"] + top_users,
    format_func=lambda u: u if u.startswith("(") else f"{u[:6]}…{u[-4:]}  ({counts[u]} purchases)")
user = None if user_choice.startswith("(") else user_choice
when = st.sidebar.date_input("Simulated date", date.today())
k = st.sidebar.slider("Items to show", 4, 20, 8)
model.alpha_max = st.sidebar.slider("Season strength (alpha_max)", 0.0, 1.0, float(model.alpha_max), 0.05)

ev, days_to, ramp = ramp_for(when)
(st.info if ev else st.success)(f"**{when:%A %d %B}** - {describe_season(when)}"
                                + (f"  (season weight = {model.alpha_max * ramp:.2f})" if ev else ""))

# the catalog and the user's history are "as of today" (end of the data); only the season is simulated
now = model.data_end
kwargs = dict(k=k, pool_at=now, profile_at=now)
left, right = st.columns(2)
for col, seasonal, title in [(left, False, "Personalized only"), (right, True, "Personalized + season")]:
    with col:
        st.subheader(title)
        recs = model.recommend(user, pd.Timestamp(when), seasonal=seasonal, **kwargs)
        if recs.empty:
            st.warning("No candidate items.")
            continue
        for _, r in recs.iterrows():
            st.markdown(card(r), unsafe_allow_html=True)

st.caption("Products show attributes only (the dataset has no names or images). "
           "The catalog is the set of products sold in the last 60 days of the data.")
