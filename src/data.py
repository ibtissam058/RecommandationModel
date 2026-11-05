"""Loading and splitting the REES46 jewelry purchase file (it has NO header row)."""
from __future__ import annotations

import pandas as pd

COLS = ["event_time", "order_id", "product_id", "quantity", "category_id",
        "category_code", "brand", "price", "user_id",
        "product_gender", "color", "metal", "gem"]
ID_COLS = ["order_id", "product_id", "category_id", "user_id"]
TEXT_COLS = ["category_code", "brand", "product_gender", "color", "metal", "gem"]


def load(path: str = "data/raw/jewelry.csv") -> pd.DataFrame:
    """Read the raw CSV. IDs are read as text so 19-digit numbers are never rounded."""
    ncols = pd.read_csv(path, header=None, nrows=5).shape[1]
    if ncols not in (len(COLS), len(COLS) + 1):
        raise ValueError(
            f"Expected {len(COLS)} columns (or {len(COLS) + 1} with an empty trailing one) "
            f"but the file has {ncols}. Check the file (or edit COLS in src/data.py).")
    names = COLS + ["_empty"] if ncols == len(COLS) + 1 else COLS
    df = pd.read_csv(path, header=None, names=names,
                     dtype={c: "string" for c in ID_COLS}, low_memory=False)
    df = df.drop(columns=[c for c in ["_empty"] if c in df.columns])
    if df["user_id"].str.contains("E+", regex=False).any():
        raise ValueError("IDs are in scientific notation: the file was saved by Excel. "
                         "Re-download the original file.")
    # Timestamps -> naive UTC. Bad rows (e.g. a stray header line) become NaT and are dropped.
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True, errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["event_time", "product_id", "user_id"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    for c in TEXT_COLS:
        s = df[c].astype("string").str.strip()
        df[c] = s.mask(s == "")
    df = df.drop(columns=["quantity"])  # always 1 -> carries no information
    return df.sort_values("event_time").reset_index(drop=True)


def split_by_date(df: pd.DataFrame, val_start: str, test_start: str):
    """Time-based split (never random: a random split leaks the future)."""
    vs, ts = pd.Timestamp(val_start), pd.Timestamp(test_start)
    train = df[df.event_time < vs]
    val = df[(df.event_time >= vs) & (df.event_time < ts)]
    test = df[df.event_time >= ts]
    return train, val, test


def summary(df: pd.DataFrame) -> str:
    per_user = df.user_id.value_counts()
    lines = [
        f"rows           : {len(df):,}",
        f"date range     : {df.event_time.min()}  ->  {df.event_time.max()}",
        f"users          : {df.user_id.nunique():,}",
        f"products       : {df.product_id.nunique():,}",
        f"one-time buyers: {(per_user == 1).mean():.1%} of users",
        f"purchases/user : mean {per_user.mean():.2f}, max {per_user.max()}",
        "missing share  : " + ", ".join(f"{c}={df[c].isna().mean():.0%}" for c in TEXT_COLS + ["price"]),
    ]
    return "\n".join(lines)
