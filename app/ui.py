"""HTML/CSS building blocks for the store-style demo. Pure functions, no Streamlit imports.

Images are looked up in app/assets/ by name; anything missing falls back to a drawn placeholder:
  banners : banner_default, banner_valentines, banner_womens_day, banner_mothers_day,
            banner_halloween, banner_christmas   (.jpg/.png/.webp)  - Black Friday is drawn with CSS
  products: ring_gold, ring_silver, earring_gold, ... brooch_silver, other   (.jpg/.png/.webp)
"""
from __future__ import annotations

import base64
import html
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

ASSETS = Path(__file__).resolve().parent / "assets"
BRAND = "ÉCLAT"
CREDITS = "Product photos belong to their respective owners and are used only for this non-commercial university project."

EVENT_LABEL = {
    "valentines_day": "Valentine's Day", "womens_day": "Women's Day", "mothers_day": "Mother's Day",
    "halloween": "Halloween", "black_friday": "Black Friday", "christmas": "Christmas",
}

# mode: light = dark text on a cream gradient, dark = white text on a dark gradient,
#       none = the picture already contains its own text, gold = drawn with CSS only
BANNERS = {
    None: dict(file="banner_default", title="Jewelry, chosen for you",
               sub="Personal picks that follow the season.", mode="light"),
    "valentines_day": dict(file="banner_valentines", title="For the one you love",
                           sub="Rubies and diamonds for Valentine's Day.", mode="dark"),
    "womens_day": dict(file="banner_womens_day", mode="none", pos="center 8%"),
    "mothers_day": dict(file="banner_mothers_day", title="For Mom, with love",
                        sub="Gifts that sparkle as much as she does.", mode="light"),
    "halloween": dict(file="banner_halloween", title="A little sparkle after dark",
                      sub="Statement pieces for the spookiest night.", mode="dark"),
    "christmas": dict(file="banner_christmas", mode="none"),
    "black_friday": dict(file=None, title="Black Friday",
                         sub="The most-wanted pieces, picked for you.", mode="gold"),
}

KINDS = ["ring", "earring", "pendant", "necklace", "bracelet", "brooch"]
KIND_LABEL = {"ring": "Ring", "earring": "Earrings", "pendant": "Pendant", "necklace": "Necklace",
              "bracelet": "Bracelet", "brooch": "Brooch", "other": "Piece"}

METAL = {  # line colour, light fill, tile gradient
    "gold": ("#c4973b", "#f6e7bd", ("#fbf4e6", "#efe0c0")),
    "silver": ("#8f9aab", "#eef1f6", ("#f6f8fb", "#dde3ec")),
}
GEM_COLORS = {
    "diamond": "#eaf6ff", "fianit": "#e6f3ff", "ruby": "#d0213a", "sapphire": "#2a5bd7",
    "emerald": "#1f9d62", "garnet": "#8f1d2c", "amethyst": "#8e5bd6", "topaz": "#f2a93b",
    "pearl": "#f7f1e6", "quartz": "#e8d9ee", "turquoise": "#3cc6c0", "agate": "#b98c6a",
    "coral": "#f08a7c", "amber": "#e0a030", "onyx": "#2d2d33", "citrine": "#f0c43c",
}


def _s(v) -> str:
    return "" if v is None or (not isinstance(v, str) and pd.isna(v)) else str(v)


def _min(s: str) -> str:
    """Collapse whitespace so Streamlit's markdown never treats indented HTML as a code block."""
    return re.sub(r">\s+<", "><", re.sub(r"\n\s*", "", s))


# --------------------------------------------------------------------------- images
@lru_cache(maxsize=64)
def _find(stem: str):
    """Look for assets/<stem>.<ext>, also inside sub-folders and ignoring upper/lower case."""
    if not ASSETS.exists():
        return None
    for p in sorted(ASSETS.rglob("*")):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} and p.stem.lower() == stem.lower():
            return p
    return None


@lru_cache(maxsize=64)
def data_uri(stem: str):
    p = _find(stem)
    if p is None:
        return None
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}[p.suffix[1:].lower()]
    return f"data:image/{mime};base64," + base64.b64encode(p.read_bytes()).decode()


# --------------------------------------------------------------------------- item helpers
def kind_of(row) -> str:
    k = _s(row.get("category_code")).split(".")[-1].lower()
    return k if k in KINDS else "other"


def metal_of(row) -> str:
    return "silver" if _s(row.get("metal")).lower() == "silver" else "gold"


def item_title(row) -> str:
    parts = [_s(row.get("metal")), _s(row.get("gem")), KIND_LABEL[kind_of(row)]]
    return " ".join(p.replace("_", " ") for p in parts if p).title()


def _gem_color(gem: str) -> str:
    g = gem.lower()
    for k, c in GEM_COLORS.items():
        if k in g:
            return c
    return "#e9eef5"


def placeholder_svg(kind: str, metal: str, gem: str) -> str:
    m, light, _ = METAL[metal]
    g = _gem_color(gem)
    st = f'stroke="{m}" stroke-linecap="round" stroke-linejoin="round"'
    if kind == "ring":
        body = (f'<circle cx="100" cy="124" r="44" fill="none" {st} stroke-width="9"/>'
                f'<polygon points="100,44 124,68 100,94 76,68" fill="{g}" {st} stroke-width="4"/>'
                f'<line x1="76" y1="68" x2="124" y2="68" {st} stroke-width="2" opacity=".6"/>')
    elif kind == "earring":
        body = "".join(
            f'<circle cx="{x}" cy="48" r="5" fill="{m}"/><line x1="{x}" y1="53" x2="{x}" y2="82" {st} stroke-width="3"/>'
            f'<path d="M{x} 82 C{x + 21} 106 {x + 17} 140 {x} 148 C{x - 17} 140 {x - 21} 106 {x} 82Z" fill="{g}" {st} stroke-width="4"/>'
            for x in (66, 134))
    elif kind == "pendant":
        body = (f'<path d="M36 36 Q100 134 164 36" fill="none" {st} stroke-width="3"/>'
                f'<circle cx="100" cy="86" r="4" fill="{m}"/>'
                f'<path d="M100 92 C121 112 119 138 100 150 C81 138 79 112 100 92Z" fill="{g}" {st} stroke-width="4"/>')
    elif kind == "necklace":
        body = (f'<path d="M28 46 Q100 196 172 46" fill="none" {st} stroke-width="4"/>'
                f'<circle cx="65" cy="102" r="6" fill="{light}" {st} stroke-width="3"/>'
                f'<circle cx="135" cy="102" r="6" fill="{light}" {st} stroke-width="3"/>'
                f'<circle cx="100" cy="120" r="13" fill="{g}" {st} stroke-width="4"/>')
    elif kind == "bracelet":
        body = (f'<ellipse cx="100" cy="106" rx="70" ry="44" fill="none" {st} stroke-width="9"/>'
                f'<circle cx="100" cy="62" r="13" fill="{g}" {st} stroke-width="4"/>')
    elif kind == "brooch":
        petals = "".join(f'<ellipse cx="0" cy="-34" rx="14" ry="26" fill="{light}" {st} stroke-width="3" transform="rotate({a})"/>'
                         for a in range(0, 360, 60))
        body = f'<g transform="translate(100 100)">{petals}<circle r="15" fill="{g}" {st} stroke-width="4"/></g>'
    else:
        body = (f'<polygon points="100,46 150,86 100,154 50,86" fill="{g}" {st} stroke-width="5"/>'
                f'<polyline points="50,86 150,86" fill="none" {st} stroke-width="3" opacity=".6"/>'
                f'<polyline points="75,86 100,46 125,86 100,154 75,86" fill="none" {st} stroke-width="2" opacity=".5"/>')
    return f'<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">{body}</svg>'


# Product photos in app/assets/products/<kind>_<metal>_<n>.jpg. The dataset has no pictures, so each product
# gets the photo whose type / metal / tone / stone fits best (ties are broken by product id: stable per product).
# stem: (metals it can stand for, gold tone, stones, suits men)
CLEAR = {"diamond", "fianit", "cubic_zirconia", ""}
PHOTOS = {
    "pendant_gold_1": ({"gold"}, "yellow", CLEAR, False),
    "pendant_gold_2": ({"gold"}, "red", {"topaz", "sapphire", "quartz", "turquoise", "amethyst"}, False),
    "earring_gold_1": ({"gold"}, "yellow", CLEAR, False),
    "earring_gold_2": ({"gold"}, "yellow", CLEAR, False),
    "earring_gold_3": ({"gold"}, "red", CLEAR, False),
    "earring_gold_4": ({"gold"}, "red", CLEAR, False),
    "brooch_gold_1": ({"gold"}, "yellow", {"ruby", "garnet"}, False),
    "brooch_gold_2": ({"gold"}, "yellow", {"pearl"}, False),
    "brooch_gold_3": ({"gold"}, "red", {"ruby", "garnet"}, False),
    "necklace_silver_1": ({"silver"}, None, {"spinel", "onyx", "agate", ""}, None),
    "brooch_silver_1": ({"silver"}, None, {""}, False),
    "necklace_gold_1": ({"gold"}, "yellow", CLEAR, False),
    "necklace_gold_2": ({"gold"}, "yellow", CLEAR, False),
    "bracelet_gold_1": ({"gold"}, "yellow", {""}, None),
    "bracelet_gold_2": ({"gold"}, "yellow", {""}, None),
    "bracelet_gold_3": ({"gold"}, "yellow", {""}, None),
    "bracelet_gold_4": ({"gold"}, "yellow", {""}, None),
    "ring_gold_1": ({"gold"}, "red", CLEAR, False),
    "ring_gold_2": ({"gold"}, "yellow", {""}, None),
    "ring_gold_3": ({"gold"}, "yellow", CLEAR, False),
    "ring_gold_4": ({"gold"}, "yellow", {""}, None),
    "ring_silver_1": ({"silver", "gold"}, "white", CLEAR, None),
    "ring_silver_2": ({"silver", "gold"}, "white", {""}, None),
}


def photo_for(row):
    """Stem of the best photo for this product, or None (then a drawn placeholder is used)."""
    import zlib
    kind, metal = kind_of(row), _s(row.get("metal")).lower() or "gold"
    metal = "silver" if metal == "silver" else "gold"
    tone, gem = _s(row.get("color")).lower(), _s(row.get("gem")).lower()
    men = _s(row.get("product_gender")) == "m"
    scored = []
    for stem, (metals, ptone, gems, for_men) in PHOTOS.items():
        if not stem.startswith(kind + "_") or metal not in metals or not _find(stem):
            continue
        s = 10                                              # same kind and metal is a must
        if gem not in CLEAR and gem in gems:
            s += 6                                          # a coloured stone that really matches (ruby, pearl, ...)
        elif gem in CLEAR and gems & CLEAR:
            s += 3
        elif gem not in CLEAR and gems <= CLEAR:
            s += 1                                          # coloured stone but only clear-stone photos exist
        s += 3 if (metal == "gold" and tone and tone == ptone) else 0
        s += 2 if men == for_men else (1 if for_men is None else 0)
        scored.append((s, stem, (for_men is None or men == for_men) and not (metal == "gold" and ptone == "white" and tone != "white")))
    if not scored:
        return None
    top = max(s for s, _, _ in scored)
    best = sorted(st for s, st, _ in scored if s == top)
    if len(best) < 3:                                       # too few exact matches: also use close ones, for variety
        best = sorted(st for s, st, ok in scored if s >= top - 4 and ok) or best
    return best[zlib.crc32(_s(row.get("product_id")).encode()) % len(best)]


def _thumb(row) -> str:
    kind, metal, gem = kind_of(row), metal_of(row), _s(row.get("gem"))
    stem = photo_for(row)
    uri = data_uri(stem) if stem else None
    if uri:
        return f'<img class="ec-photo" src="{uri}" alt="{html.escape(item_title(row))}">'
    a, b = METAL[metal][2]
    return (f'<div class="ec-tile" style="background:linear-gradient(145deg,{a},{b})">'
            f'{placeholder_svg(kind, metal, gem)}</div>')


# --------------------------------------------------------------------------- blocks
def badge_for(row, event_label: str, season_on: bool):
    ef, ua = row.get("event_fit"), row.get("user_affinity")
    if season_on and event_label and pd.notna(ef) and ef >= 0.75:
        return "season", f"Trending for {event_label}"
    if pd.notna(ua) and ua >= 0.75:
        return "match", "Matches your taste"
    return "pop", "Popular now"


# Store pricing rule (NOT learned from data, the dataset has no discount information):
# a store-wide sale during an event window. The recommender only ranks; it never sets prices.
DEALS = {"black_friday": 0.20, "christmas": 0.15, "valentines_day": 0.10,
         "womens_day": 0.10, "mothers_day": 0.10, "halloween": 0.05}


def apply_deal(recs: pd.DataFrame, event_name) -> pd.DataFrame:
    """Add `list_price` (original) and `deal_pct`; lower `price` when an event sale is on."""
    recs = recs.copy()
    pct = DEALS.get(event_name, 0.0) if event_name else 0.0
    recs["list_price"] = recs["price"]
    recs["deal_pct"] = pct
    if pct > 0:
        recs["price"] = (recs["price"] * (1 - pct)).round(2)
    return recs


def price_html(row) -> str:
    price, old, pct = row.get("price"), row.get("list_price"), row.get("deal_pct")
    if pd.isna(price):
        return ""
    if pd.notna(pct) and pct > 0 and pd.notna(old):
        return (f'${price:,.0f}<s style="color:#9a8f84;font-weight:400;font-size:12px;margin:0 7px">${old:,.0f}</s>'
                f'<span style="color:#a33;font-size:11px;font-weight:600">-{int(round(pct * 100))}%</span>')
    return f"${price:,.0f}"


def card_html(row, badge) -> str:
    price_s = price_html(row)
    tone = {"yellow": "Yellow gold", "red": "Rose gold", "white": "White gold"}.get(_s(row.get("color")).lower())
    sub = []
    if tone and metal_of(row) == "gold":
        sub.append(tone)
    elif metal_of(row) == "silver":
        sub.append("Silver")
    gem = _s(row.get("gem")).replace("_", " ").title()
    if gem:
        sub.append(gem)
    who = {"f": "For her", "m": "For him"}.get(_s(row.get("product_gender")), "")
    pct = row.get("deal_pct")
    sale = f'<span class="ec-sale">-{int(round(pct * 100))}%</span>' if pd.notna(pct) and pct > 0 else ""
    cls, text = badge
    return _min(f"""
    <div class="ec-card">
      <div class="ec-imgwrap">{_thumb(row)}<span class="ec-badge {cls}">{html.escape(text)}</span>{sale}</div>
      <div class="ec-info">
        <div class="ec-kind">{html.escape(KIND_LABEL[kind_of(row)])}{' · ' + who if who else ''}</div>
        <div class="ec-name">{html.escape(item_title(row))}</div>
        <div class="ec-sub">{html.escape(' · '.join(sub))}</div>
        <div class="ec-row"><span class="ec-price">{price_s}</span><span class="ec-ship">Free shipping $100+</span></div>
      </div>
    </div>""")


def grid_html(recs: pd.DataFrame, event_name, season_on: bool, columns: int = 4) -> str:
    label = EVENT_LABEL.get(event_name, "") if event_name else ""
    cards = "".join(card_html(r, badge_for(r, label, season_on)) for _, r in recs.iterrows())
    return f'<div class="ec-grid c{columns}">{cards}</div>'


def header_html(when, cart_n=None, date: bool = True) -> str:
    cart = f'<span class="ec-cart">Cart · {cart_n}</span>' if cart_n is not None else ""
    right = f'<div class="ec-right"><span class="ec-date">{pd.Timestamp(when):%a, %d %b %Y}</span>{cart}</div>' if date else ""
    return _min(f"""
    <div class="ec-header"><div><div class="ec-brand">{BRAND}</div><div class="ec-tag">Fine jewelry</div></div>{right}</div>""")


def date_pill_html(when) -> str:
    return f'<div class="ec-datebox"><span class="ec-date">{pd.Timestamp(when):%a, %d %b %Y}</span></div>'


def photo_count() -> int:
    return sum(1 for s in PHOTOS if _find(s))


def rule_html() -> str:
    return '<div class="ec-rule"></div>'


NAV = ["All", "Rings", "Earrings", "Necklaces", "Pendants", "Bracelets", "Brooches"]
NAV_KIND = {"Rings": "ring", "Earrings": "earring", "Necklaces": "necklace", "Pendants": "pendant",
            "Bracelets": "bracelet", "Brooches": "brooch"}


def filter_recs(recs: pd.DataFrame, nav: str, k: int) -> pd.DataFrame:
    """Keep the ranking order; apply the category filter; hide items with no category when possible."""
    if recs.empty:
        return recs
    kinds = recs.apply(kind_of, axis=1)
    if nav in NAV_KIND:
        sub = recs[kinds == NAV_KIND[nav]]
    else:
        sub = recs[kinds != "other"]
        if len(sub) < k:
            sub = pd.concat([sub, recs[kinds == "other"]])
    return sub.head(k)


def detail_image_html(row) -> str:
    return _min(f'<div class="ec-imgwrap big">{_thumb(row)}</div>')


def _meter(label: str, v) -> str:
    pct = int(round(100 * float(v)))
    return f'<div class="ec-meter"><span>{html.escape(label)}</span><div><i style="width:{pct}%"></i></div><b>{pct}%</b></div>'


def detail_html(row, event_name, season_on: bool) -> str:
    price_s = price_html(row)
    tone = {"yellow": "Yellow gold", "red": "Rose gold", "white": "White gold"}.get(_s(row.get("color")).lower(), "")
    rows = [("Type", KIND_LABEL[kind_of(row)]), ("Metal", _s(row.get("metal")).title()),
            ("Gemstone", _s(row.get("gem")).replace("_", " ").title()),
            ("Gold tone", tone if metal_of(row) == "gold" else ""),
            ("Made for", {"f": "Women", "m": "Men"}.get(_s(row.get("product_gender")), ""))]
    attrs = "".join(f"<tr><td>{k}</td><td>{html.escape(v)}</td></tr>" for k, v in rows if v)
    ua, pop, ef = row.get("user_affinity"), row.get("popularity"), row.get("event_fit")
    why = ""
    if pd.notna(ua):
        why += _meter("Matches your past purchases", ua)
    if pd.notna(pop):
        why += _meter("Popular with recent shoppers", pop)
    if season_on and event_name and pd.notna(ef):
        why += _meter(f"Fits {EVENT_LABEL[event_name]}", ef)
    return _min(f"""<div class="ec-detail"><div class="ec-dname">{html.escape(item_title(row))}</div>
        <div class="ec-dprice">{price_s}</div><table>{attrs}</table>
        <div class="ec-why"><h4>Why we recommend it</h4>{why or "<p>Popular right now.</p>"}</div>
        <div class="ec-dnote">Free shipping over $100 · 30-day returns · Gift packaging included</div></div>""")


def banner_html(event_name, days_to=None) -> str:
    cfg = BANNERS.get(event_name) or BANNERS[None]
    mode = cfg["mode"]
    label = EVENT_LABEL.get(event_name, "")
    pill = ""
    if event_name:
        when = "today" if days_to == 0 else f"{days_to} day{'s' if days_to != 1 else ''} to go"
        pill = f'<span class="ec-pill">{html.escape(label)} · {when}</span>'
    if mode == "gold":
        return _min(f"""<div class="ec-hero gold"><div class="ec-hero-txt gold">{pill}
            <h1>{cfg['title']}</h1><p>{cfg['sub']}</p></div></div>""")
    uri = data_uri(cfg["file"]) or data_uri(BANNERS[None]["file"])
    bg = (f"background-image:url({uri});background-position:{cfg.get('pos', 'center')}" if uri
          else "background:linear-gradient(120deg,#f3e7d3,#d9c3a0)")
    if mode == "none":
        return f'<div class="ec-hero" style="{bg}">{pill and f"<div class=ec-pillbox>{pill}</div>"}</div>'
    return _min(f"""<div class="ec-hero" style="{bg}"><div class="ec-ov {mode}"></div>
        <div class="ec-hero-txt {mode}">{pill}<h1>{cfg['title']}</h1><p>{cfg['sub']}</p></div></div>""")


def season_strip(event_name, days_to, weight: float) -> str:
    if not event_name:
        return '<div class="ec-strip"><b>Everyday picks</b><span>No special day nearby: recommendations are purely personal.</span></div>'
    return _min(f"""<div class="ec-strip on"><b>{html.escape(EVENT_LABEL[event_name])}</b>
        <span>{days_to} day(s) away · season weight {weight:.2f}: the ranking leans towards what sells for this occasion.</span></div>""")


def money(x: float) -> str:
    return f"${x:,.2f}"


def cart_item_html(row, unit: float) -> str:
    return _min(f"""<div class="ec-line"><div class="ec-lthumb">{_thumb(row)}</div>
        <div><div class="ec-name">{html.escape(item_title(row))}</div>
        <div class="ec-lsub">{money(unit)} each</div></div></div>""")


def summary_html(subtotal: float, discount: float, shipping: float, total: float) -> str:
    rows = f'<div class="ec-srow"><span>Subtotal</span><span>{money(subtotal)}</span></div>'
    if discount:
        rows += f'<div class="ec-srow"><span>Promo code</span><span>-{money(discount)}</span></div>'
    rows += f'<div class="ec-srow"><span>Shipping</span><span>{"Free" if shipping == 0 else money(shipping)}</span></div>'
    rows += f'<div class="ec-srow total"><span>Total</span><span>{money(total)}</span></div>'
    return _min(f"""<div class="ec-summary"><h4>Order summary</h4>{rows}
        <p>Free shipping on orders over $100. Demo store: no real payment is processed.</p></div>""")


def order_html(o: dict) -> str:
    items = "".join(f'<tr><td>{html.escape(i["title"])} × {i["qty"]}</td><td>{money(i["price"] * i["qty"])}</td></tr>'
                    for i in o["items"])
    return _min(f"""<div class="ec-order"><div class="ec-ok">✓</div>
        <h3>Thank you, {html.escape(o['name'])}!</h3>
        <p>Order <b>{html.escape(o['id'])}</b> has been placed (demo). A confirmation would be sent to {html.escape(o['email'])}.</p>
        <table>{items}<tr class="t"><td>Total</td><td>{money(o['total'])}</td></tr></table>
        <p>Delivery to: {html.escape(o['address'])}, {html.escape(o['city'])}<br>Payment: {html.escape(o['payment'])}</p></div>""")


def section_title(text: str, sub: str = "") -> str:
    return f'<div class="ec-sec"><h2>{html.escape(text)}</h2><p>{html.escape(sub)}</p></div>'


def announce_html(event_name=None, when=None) -> str:
    pct = DEALS.get(event_name, 0.0) if event_name else 0.0
    deal = f'<b>{html.escape(EVENT_LABEL[event_name])}: {int(round(pct * 100))}% off everything</b><i>·</i>' if pct else ""
    day = f'<span class="ec-adate">{pd.Timestamp(when):%a, %d %b %Y}</span>' if when is not None else ""
    return _min(f"""<div class="ec-announce">{deal}Free shipping over $100<i>·</i>30-day returns<i>·</i>Gift packaging included{day}</div>""")


_ICON = {
    "ship": '<path d="M3 7h11v9H3zM14 10h4l3 3v3h-7z"/><circle cx="7" cy="17.5" r="1.8"/><circle cx="17" cy="17.5" r="1.8"/>',
    "return": '<path d="M4 8h11a5 5 0 0 1 0 10H8"/><path d="M8 4 4 8l4 4"/>',
    "gem": '<path d="M6 4h12l4 6-10 11L2 10z"/><path d="M2 10h20M9 4l3 6 3-6M12 10v11"/>',
    "lock": '<rect x="5" y="11" width="14" height="10" rx="2"/><path d="M8 11V8a4 4 0 0 1 8 0v3"/>',
}


def trust_html() -> str:
    items = [("ship", "Free shipping", "On every order over $100"), ("return", "30-day returns", "Changed your mind? No problem"),
             ("gem", "Certified stones", "Every gem is checked by hand"), ("lock", "Secure checkout", "Your details stay private")]
    cells = "".join(
        f'<div class="ec-trust-i"><svg viewBox="0 0 24 24" fill="none" stroke="#b08d57" stroke-width="1.6" '
        f'stroke-linecap="round" stroke-linejoin="round">{_ICON[k]}</svg><div><b>{a}</b><span>{b}</span></div></div>'
        for k, a, b in items)
    return f'<div class="ec-trust">{cells}</div>'


def footer_html() -> str:
    cols = [("Shop", ["Rings", "Earrings", "Necklaces", "Pendants", "Bracelets", "Brooches"]),
            ("Help", ["Shipping & delivery", "Returns & exchanges", "Ring size guide", "Jewelry care"]),
            ("Éclat", ["Our story", "Craftsmanship", "Sustainability", "Contact us"])]
    body = "".join(f'<div><h5>{h}</h5>' + "".join(f"<p>{x}</p>" for x in xs) + "</div>" for h, xs in cols)
    return _min(f"""<div class="ec-footer"><div class="ec-fcols"><div><div class="ec-brand sm">{BRAND}</div>
        <p class="ec-fabout">Timeless pieces, chosen for you. This is a university demo store: no real orders are taken.</p></div>{body}</div>
        <div class="ec-fnote">© 2026 Éclat demo store · Product photos are illustrative and chosen by jewelry type (the dataset has no photos).<br>{html.escape(CREDITS)}</div></div>""")


def css() -> str:
    return """<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');
.stApp{background:#faf6f0}
header[data-testid="stHeader"]{background:transparent}
[data-testid="stToolbar"],#MainMenu,footer{visibility:hidden}
.block-container{max-width:1280px;padding-top:3.2rem;padding-bottom:2rem}
.ec-header{display:flex;align-items:center;justify-content:space-between;padding:4px;min-height:46px;font-family:Inter,system-ui,sans-serif}
.ec-rule{border-bottom:1px solid #e6dccb;margin:2px 0 18px}
.ec-brand{font-family:'Cormorant Garamond',Georgia,serif;font-size:30px;letter-spacing:.28em;color:#2b2623;font-weight:600}
.ec-nav{display:flex;gap:26px;font-size:13px;letter-spacing:.08em;text-transform:uppercase;color:#6b5f55}
.ec-date{font-size:12px;color:#8a7a68;background:#f1e8d8;padding:6px 12px;border-radius:999px}
.ec-hero{position:relative;aspect-ratio:16/5;min-height:300px;border-radius:14px;overflow:hidden;background-size:cover;background-position:center;margin-bottom:14px}
.ec-hero.gold{background:radial-gradient(120% 140% at 85% 20%,#6b5126 0%,#1c1712 45%,#0b0a09 100%)}
.ec-ov{position:absolute;inset:0}
.ec-ov.light{background:linear-gradient(90deg,rgba(250,246,240,.95) 0%,rgba(250,246,240,.78) 34%,rgba(250,246,240,0) 62%)}
.ec-ov.dark{background:linear-gradient(90deg,rgba(18,6,8,.82) 0%,rgba(18,6,8,.55) 36%,rgba(18,6,8,0) 66%)}
.ec-hero-txt{position:absolute;left:46px;top:50%;transform:translateY(-50%);max-width:46%;font-family:Inter,system-ui,sans-serif}
.ec-hero-txt h1{font-family:'Cormorant Garamond',Georgia,serif;font-size:50px;line-height:1.05;margin:12px 0 10px;padding:0;font-weight:600}
.ec-hero-txt p{font-size:16px;margin:0;line-height:1.5}
.ec-hero-txt.light h1{color:#2b2623}.ec-hero-txt.light p{color:#5c5047}
.ec-hero-txt.dark h1,.ec-hero-txt.gold h1{color:#fff}.ec-hero-txt.dark p{color:#f1e3e3}
.ec-hero-txt.gold h1{color:#e8c777}.ec-hero-txt.gold p{color:#d9cdb6}
.ec-pill{display:inline-block;font-size:11px;letter-spacing:.14em;text-transform:uppercase;background:#b08d57;color:#fff;padding:6px 12px;border-radius:999px}
.ec-pillbox{position:absolute;left:18px;top:16px}
.ec-strip{display:flex;gap:14px;align-items:baseline;flex-wrap:wrap;font-family:Inter,system-ui,sans-serif;font-size:13.5px;color:#6b5f55;background:#f3ece0;border-radius:10px;padding:11px 16px;margin-bottom:6px}
.ec-strip b{color:#2b2623}.ec-strip.on{background:#f6ead2}
.ec-sec{margin:26px 0 14px;font-family:Inter,system-ui,sans-serif}
.ec-sec h2{font-family:'Cormorant Garamond',Georgia,serif;font-size:32px;font-weight:600;color:#2b2623;margin:0;padding:0}
.ec-sec p{margin:2px 0 0;color:#8a7a68;font-size:14px}
.ec-grid{display:grid;gap:20px;font-family:Inter,system-ui,sans-serif}
.ec-grid.c4{grid-template-columns:repeat(4,minmax(0,1fr))}.ec-grid.c2{grid-template-columns:repeat(2,minmax(0,1fr))}
@media(max-width:1000px){.ec-grid.c4{grid-template-columns:repeat(2,minmax(0,1fr))}.ec-nav{display:none}.ec-hero-txt{max-width:80%}.ec-hero-txt h1{font-size:34px}}
.ec-card{background:#fff;border-radius:14px;overflow:hidden;box-shadow:0 1px 3px rgba(60,40,10,.08);transition:transform .18s,box-shadow .18s}
.ec-card:hover{transform:translateY(-4px);box-shadow:0 10px 24px rgba(60,40,10,.14)}
.ec-imgwrap{position:relative;aspect-ratio:1/1;background:#f4eee4}
.ec-photo,.ec-tile{width:100%;height:100%;object-fit:cover;display:block}
.ec-tile{display:flex;align-items:center;justify-content:center}.ec-tile svg{width:68%;height:68%}
.ec-badge{position:absolute;left:10px;top:10px;font-size:11px;font-weight:600;letter-spacing:.03em;padding:5px 10px;border-radius:999px;background:#fff;color:#4a3f35;box-shadow:0 1px 4px rgba(0,0,0,.12)}
.ec-badge.season{background:#b08d57;color:#fff}.ec-badge.match{background:#2b2623;color:#fff}
.ec-info{padding:12px 14px 15px}
.ec-name{font-size:14.5px;font-weight:500;color:#2b2623;margin-bottom:7px}
.ec-row{display:flex;justify-content:space-between;align-items:center;gap:8px}
.ec-price{font-size:15px;font-weight:600;color:#7b5f2c}
.ec-chip{font-size:11px;color:#7d6f62;background:#f3ece0;border-radius:999px;padding:3px 8px;margin-left:4px}
.ec-right{display:flex;gap:10px;align-items:center}
.ec-cart{font-size:12px;color:#fff;background:#2b2623;padding:6px 14px;border-radius:999px}
.ec-card{margin-bottom:10px}
.stButton{margin:-2px 0 18px}
div.stButton>button{width:100%;border-radius:999px;border:1px solid #b08d57;background:#fff;color:#7b5f2c;font-weight:500}
div.stButton>button:hover{background:#b08d57;color:#fff;border-color:#b08d57}
div.stButton>button[kind="primary"],div.stButton>button[data-testid="stBaseButton-primary"]{background:#2b2623;border-color:#2b2623;color:#fff}
.ec-imgwrap.big{border-radius:12px;overflow:hidden}
.ec-detail{font-family:Inter,system-ui,sans-serif;color:#2b2623}
.ec-dname{font-family:'Cormorant Garamond',Georgia,serif;font-size:30px;font-weight:600;line-height:1.1}
.ec-dprice{font-size:22px;font-weight:600;color:#7b5f2c;margin:6px 0 12px}
.ec-detail table{border-collapse:collapse;width:100%;font-size:14px;margin-bottom:12px}
.ec-detail td{padding:5px 0;border-bottom:1px solid #eee4d5}.ec-detail td:first-child{color:#8a7a68;width:42%}
.ec-why h4{margin:6px 0 8px;font-size:13px;letter-spacing:.08em;text-transform:uppercase;color:#8a7a68;font-weight:600}
.ec-why p{font-size:14px;margin:0}
.ec-dnote{margin-top:12px;padding-top:10px;border-top:1px solid #eee4d5;font-size:12px;color:#8a7a68}
.ec-meter{display:grid;grid-template-columns:1.5fr 1.2fr auto;gap:10px;align-items:center;font-size:13px;margin-bottom:7px}
.ec-meter div{height:6px;background:#efe6d6;border-radius:99px;overflow:hidden}.ec-meter i{display:block;height:100%;background:#b08d57}
.ec-meter b{font-weight:600;color:#7b5f2c;font-size:12px}
div[role="radiogroup"]{gap:8px;flex-wrap:wrap}
div[role="radiogroup"] label{background:#fff;border:1px solid #e0d4bf;border-radius:999px;padding:5px 16px;cursor:pointer}
div[role="radiogroup"] label>div:first-child{display:none}
div[role="radiogroup"] label p{color:#2b2623;font-size:13px;letter-spacing:.05em}
div[role="radiogroup"] label:has(input:checked){background:#2b2623;border-color:#2b2623}
div[role="radiogroup"] label:has(input:checked) p{color:#fff}
.ec-line{display:flex;gap:14px;align-items:center;font-family:Inter,system-ui,sans-serif}
.ec-lthumb{width:78px;height:78px;border-radius:10px;overflow:hidden;flex:none;background:#f4eee4}
.ec-lsub{font-size:13px;color:#8a7a68}
.ec-ltotal{font-family:Inter,system-ui,sans-serif;font-weight:600;color:#7b5f2c;padding-top:9px}
.ec-summary{background:#fff;border-radius:14px;padding:18px 20px;box-shadow:0 1px 3px rgba(60,40,10,.08);font-family:Inter,system-ui,sans-serif;color:#2b2623}
.ec-summary h4{margin:0 0 10px;font-size:13px;letter-spacing:.08em;text-transform:uppercase;color:#8a7a68;font-weight:600}
.ec-srow{display:flex;justify-content:space-between;padding:6px 0;font-size:14.5px}
.ec-srow.total{border-top:1px solid #eee4d5;margin-top:6px;padding-top:12px;font-weight:600;font-size:18px}
.ec-summary p{font-size:12px;color:#9a8b78;margin:10px 0 0;line-height:1.5}
.ec-order{background:#fff;border-radius:14px;padding:30px 34px;max-width:660px;margin:14px auto;text-align:center;font-family:Inter,system-ui,sans-serif;color:#2b2623;box-shadow:0 1px 3px rgba(60,40,10,.08)}
.ec-order h3{font-family:'Cormorant Garamond',Georgia,serif;font-size:30px;margin:6px 0}
.ec-ok{width:54px;height:54px;border-radius:50%;background:#b08d57;color:#fff;font-size:28px;line-height:54px;margin:0 auto 8px}
.ec-order table{width:100%;text-align:left;border-collapse:collapse;margin:14px 0;font-size:14.5px}
.ec-order td{padding:7px 0;border-bottom:1px solid #eee4d5}.ec-order td:last-child{text-align:right}
.ec-order tr.t td{font-weight:600;border-bottom:none}
.ec-order p{font-size:14px;color:#6b5f55;line-height:1.6}


.ec-announce{background:#2b2623;color:#e9dcc3;text-align:center;font:500 12px/1 Inter,system-ui,sans-serif;letter-spacing:.06em;padding:10px 12px;border-radius:8px;margin-bottom:10px}
.ec-announce{position:relative}.ec-adate{position:absolute;right:12px;top:50%;transform:translateY(-50%);background:rgba(255,255,255,.14);color:#f1e6cf;padding:5px 12px;border-radius:999px;font-size:11px;letter-spacing:.04em;white-space:nowrap}
@media(max-width:900px){.ec-adate{display:none}}
.ec-announce b{color:#e8c777;font-weight:600}.ec-announce i{font-style:normal;opacity:.45;margin:0 12px}
.ec-tag{font:500 10px Inter,system-ui,sans-serif;letter-spacing:.42em;text-transform:uppercase;color:#a58a5a;margin:-2px 0 0 3px}
.ec-brand.sm{font-size:22px}
.ec-card{border:1px solid #f0e7d8;box-shadow:none}
.ec-card:hover{box-shadow:0 14px 30px rgba(60,40,10,.12);border-color:#e6d6b8}
.ec-photo{transition:transform .5s}.ec-card:hover .ec-photo{transform:scale(1.045)}
.ec-imgwrap{overflow:hidden}
.ec-sale{position:absolute;right:10px;top:10px;background:#a33a3a;color:#fff;font:600 11px Inter,system-ui,sans-serif;padding:5px 9px;border-radius:999px}
.ec-kind{font-size:10.5px;letter-spacing:.14em;text-transform:uppercase;color:#a58a5a;margin-bottom:3px}
.ec-name{font-family:'Cormorant Garamond',Georgia,serif;font-size:19px;font-weight:600;line-height:1.15;margin-bottom:2px}
.ec-sub{font-size:12.5px;color:#8a7a68;margin-bottom:9px;min-height:16px}
.ec-ship{font-size:10.5px;color:#9a8b78}
.ec-price{font-size:16px}
.ec-sec h2:after{content:"";display:block;width:42px;height:2px;background:#b08d57;margin-top:8px}
.ec-trust{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:34px 0 6px;padding:22px 8px;border-top:1px solid #e6dccb;border-bottom:1px solid #e6dccb;font-family:Inter,system-ui,sans-serif}
.ec-trust-i{display:flex;gap:12px;align-items:center;justify-content:center}.ec-trust-i svg{width:30px;height:30px;flex:none}
.ec-trust-i b{display:block;font-size:13.5px;color:#2b2623}.ec-trust-i span{font-size:12px;color:#8a7a68}
@media(max-width:1000px){.ec-trust{grid-template-columns:repeat(2,minmax(0,1fr))}}
.ec-footer{margin-top:26px;font-family:Inter,system-ui,sans-serif;color:#6b5f55}
.ec-fcols{display:grid;grid-template-columns:1.6fr 1fr 1fr 1fr;gap:30px;padding:10px 4px 22px}
.ec-fcols h5{margin:0 0 10px;font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:#2b2623;font-weight:600}
.ec-fcols p{margin:0 0 6px;font-size:13px}.ec-fabout{max-width:300px;line-height:1.6;color:#8a7a68}
.ec-fnote{border-top:1px solid #e6dccb;padding:14px 4px 4px;text-align:center;font-size:11.5px;color:#9a8b78;line-height:1.7}
@media(max-width:1000px){.ec-fcols{grid-template-columns:1fr 1fr}}
div[data-testid="stExpander"]{border:1px dashed #cdb98f;border-radius:12px;background:#fffdf8}
div[data-testid="stExpander"] summary p{font-size:13px;color:#7b5f2c;font-weight:600;letter-spacing:.03em}
.ec-hero{box-shadow:0 8px 28px rgba(60,40,10,.12)}
div.stButton>button{font-size:13px;letter-spacing:.04em}
</style>"""