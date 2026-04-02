"""
Large Shipment — State-wise Performance Dashboard
===================================================
Two pages:
  1. Last Mile Performance  — state-wise breakdown (ZRTO, Breach, Conv, FAC)
  2. Pickup Performance     — state & seller breakdown (D0, D1, D2, D2+)

Run:
    streamlit run large_dashboard.py

Requirements:
    pip install streamlit pandas numpy
"""

import pathlib
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent

# ═════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Large Shipment Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═════════════════════════════════════════════════════════════════════════════
#  CUSTOM CSS
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif}
.block-container{padding:1.5rem 2rem 2rem 2rem}

.kpi-card{
    background:#fff;border-radius:12px;padding:0.8rem 1rem;
    border:1px solid #E2E8F0;border-left:4px solid #3B82F6;
    box-shadow:0 1px 3px rgba(0,0,0,0.04);
    transition:transform .15s,box-shadow .15s;
}
.kpi-card:hover{transform:translateY(-2px);box-shadow:0 4px 12px rgba(0,0,0,0.08)}
.kpi-card.green{border-left-color:#10B981}
.kpi-card.red{border-left-color:#EF4444}
.kpi-card.orange{border-left-color:#F59E0B}
.kpi-card.purple{border-left-color:#8B5CF6}
.kpi-card.cyan{border-left-color:#06B6D4}
.kpi-label{font-size:.66rem;font-weight:600;color:#94A3B8;
            letter-spacing:.06em;text-transform:uppercase;margin-bottom:3px}
.kpi-value{font-size:1.35rem;font-weight:700;color:#0F172A;
            font-family:'JetBrains Mono',monospace;line-height:1.1}
.kpi-sub{font-size:.66rem;color:#94A3B8;margin-top:3px}

.section-hdr{
    background:linear-gradient(135deg,#1E293B 0%,#334155 100%);
    border-radius:12px;padding:.85rem 1.2rem;margin:.6rem 0;
    color:#fff;display:flex;align-items:center;gap:.75rem;
    box-shadow:0 2px 8px rgba(30,41,59,.18);
}
.section-hdr .ico{font-size:1.2rem;background:rgba(255,255,255,.12);
    border-radius:8px;padding:.35rem .45rem}
.section-hdr .ttl{font-size:.95rem;font-weight:700}
.section-hdr .sub{font-size:.72rem;color:rgba(255,255,255,.7)}

.drill-hdr{
    background:linear-gradient(135deg,#1D4ED8 0%,#3B82F6 100%);
    border-radius:12px;padding:1rem 1.4rem;margin:1rem 0 .5rem 0;
    color:#fff;box-shadow:0 4px 16px rgba(29,78,216,.22);
}
.drill-hdr .ttl{font-size:1.05rem;font-weight:700}
.drill-hdr .sub{font-size:.75rem;color:rgba(255,255,255,.7);margin-top:2px}
</style>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
#  CONSTANTS — Last Mile
# ═════════════════════════════════════════════════════════════════════════════
LM_NUMERIC_COLS = [
    "PHin", "conv_num", "zero_attempt_num", "zero_attempt_denom",
    "DHin", "D0_OFD", "First_attempt_delivered", "fac_deno",
    "total_delivered_attempts", "total_attempts", "rfr_num", "rfr_deno",
    "Breach_Num", "Breach_Den", "breach_plus1_num",
]

LM_AGG_COLS = [
    "PHin", "conv_num", "zero_attempt_num",
    "First_attempt_delivered", "fac_deno",
    "Breach_Num", "Breach_Den",
]

PU_NUMERIC_COLS = [
    "total_shipments", "d0_shipments", "d1_shipments", "d2_shipments", "d2_plus_shipments",
]

PU_AGG_COLS = PU_NUMERIC_COLS

# ═════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════
LM_PATH = str(_SCRIPT_DIR / "116292636eb7b0d9c72abada9282d640.csv")
PU_PATH = str(_SCRIPT_DIR / "e77ab7ceb72919e89f9c4ba93e78120e.csv")


@st.cache_data(ttl=600, max_entries=1, show_spinner="Loading Last-Mile data …")
def load_lm_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    for col in LM_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("float32")
    df["reporting_date"] = df["reporting_date"].astype(str)
    df["destination_state"] = df["destination_state"].astype("category")
    df["seller_type"] = df["seller_type"].astype("category")
    pt = df["payment_type"].str.strip().str.lower()
    df["payment_norm"] = pt.map({"cod": "COD", "pp": "Prepaid"}).fillna("Other").astype("category")
    return df


@st.cache_data(ttl=600, max_entries=1, show_spinner="Loading Pickup data …")
def load_pu_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    for col in PU_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("float32")
    df["reporting_date"] = df["reporting_date"].astype(str)
    df["source_state"] = df["source_state"].astype("category")
    df["seller_type"] = df["seller_type"].astype("category")
    return df


# ═════════════════════════════════════════════════════════════════════════════
#  VECTORISED HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def _safe_pct(num, den):
    den_safe = np.where(den > 0, den, 1)
    return np.where(den > 0, num / den_safe * 100, 0.0)


# ═════════════════════════════════════════════════════════════════════════════
#  LAST-MILE AGGREGATION
# ═════════════════════════════════════════════════════════════════════════════
def _add_lm_pct_cols(df):
    p = df["PHin"].values.astype("float64")
    df["Conv %"] = np.round(_safe_pct(df["conv_num"].values, p), 2)
    df["ZRTO %"] = np.round(_safe_pct(df["zero_attempt_num"].values, p), 2)
    fd = df["fac_deno"].values.astype("float64")
    df["FAC %"] = np.round(_safe_pct(df["First_attempt_delivered"].values, fd), 2)
    bd = df["Breach_Den"].values.astype("float64")
    df["Breach %"] = np.round(_safe_pct(df["Breach_Num"].values, bd), 2)
    if "cod_vol" in df.columns:
        cv = df["cod_vol"].values.astype("float64")
        pv = df["pp_vol"].values.astype("float64")
        df["COD Share %"] = np.round(_safe_pct(cv, p), 2)
        df["Prepaid Share %"] = np.round(_safe_pct(pv, p), 2)
        df["COD Conv %"] = np.round(_safe_pct(df["cod_conv"].values, cv), 2)
        df["Prepaid Conv %"] = np.round(_safe_pct(df["pp_conv"].values, pv), 2)
    return df


def lm_aggregate_by(df: pd.DataFrame, group_cols, with_payment_split: bool = True):
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    present = [c for c in LM_AGG_COLS if c in df.columns]
    base = df.groupby(group_cols, observed=True)[present].sum().reset_index()
    if with_payment_split:
        cod = (
            df[df["payment_norm"] == "COD"]
            .groupby(group_cols, observed=True)
            .agg(cod_vol=("PHin", "sum"), cod_conv=("conv_num", "sum"))
            .reset_index()
        )
        pp = (
            df[df["payment_norm"] == "Prepaid"]
            .groupby(group_cols, observed=True)
            .agg(pp_vol=("PHin", "sum"), pp_conv=("conv_num", "sum"))
            .reset_index()
        )
        base = base.merge(cod, on=group_cols, how="left").merge(pp, on=group_cols, how="left")
    num_cols = base.select_dtypes(include="number").columns
    base[num_cols] = base[num_cols].fillna(0)
    return _add_lm_pct_cols(base).sort_values("PHin", ascending=False)


def lm_overall_kpis(df: pd.DataFrame) -> dict:
    if df.empty:
        return {k: 0 for k in ["Volume", "Conv %", "ZRTO %", "FAC %", "Breach %"]}
    tv = float(df["PHin"].sum())
    td = float(df["conv_num"].sum())
    zn = float(df["zero_attempt_num"].sum())
    fn = float(df["First_attempt_delivered"].sum())
    fd = float(df["fac_deno"].sum())
    bn = float(df["Breach_Num"].sum())
    bd = float(df["Breach_Den"].sum())
    pct = lambda n, d: round(n / d * 100, 2) if d > 0 else 0.0
    return {
        "Volume": int(tv),
        "Conv %": pct(td, tv),
        "ZRTO %": pct(zn, tv),
        "FAC %": pct(fn, fd),
        "Breach %": pct(bn, bd),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  PICKUP AGGREGATION
# ═════════════════════════════════════════════════════════════════════════════
def _add_pu_pct_cols(df):
    ts = df["total_shipments"].values.astype("float64")
    df["D0 %"] = np.round(_safe_pct(df["d0_shipments"].values, ts), 2)
    df["D1 %"] = np.round(_safe_pct(df["d1_shipments"].values, ts), 2)
    df["D2 %"] = np.round(_safe_pct(df["d2_shipments"].values, ts), 2)
    df["D2+ %"] = np.round(_safe_pct(df["d2_plus_shipments"].values, ts), 2)
    return df


def pu_aggregate_by(df: pd.DataFrame, group_cols):
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    present = [c for c in PU_AGG_COLS if c in df.columns]
    base = df.groupby(group_cols, observed=True)[present].sum().reset_index()
    num_cols = base.select_dtypes(include="number").columns
    base[num_cols] = base[num_cols].fillna(0)
    return _add_pu_pct_cols(base).sort_values("total_shipments", ascending=False)


def pu_overall_kpis(df: pd.DataFrame) -> dict:
    if df.empty:
        return {k: 0 for k in ["Total Shipments", "D0 %", "D1 %", "D2 %", "D2+ %"]}
    ts = float(df["total_shipments"].sum())
    d0 = float(df["d0_shipments"].sum())
    d1 = float(df["d1_shipments"].sum())
    d2 = float(df["d2_shipments"].sum())
    d2p = float(df["d2_plus_shipments"].sum())
    pct = lambda n, d: round(n / d * 100, 2) if d > 0 else 0.0
    return {
        "Total Shipments": int(ts),
        "D0 %": pct(d0, ts),
        "D1 %": pct(d1, ts),
        "D2 %": pct(d2, ts),
        "D2+ %": pct(d2p, ts),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  COLOUR / STYLE HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def _clr_breach(v):
    if pd.isna(v) or v == 0:
        return "background-color:#F8FAFC;color:#64748B;"
    if v <= 5:
        return "background-color:#DCFCE7;color:#166534;font-weight:600;"
    if v <= 10:
        return "background-color:#FEF9C3;color:#854D0E;font-weight:600;"
    return "background-color:#FEE2E2;color:#991B1B;font-weight:700;"


def _clr_zrto(v):
    if pd.isna(v) or v == 0:
        return "background-color:#F8FAFC;color:#64748B;"
    if v <= 1.5:
        return "background-color:#DCFCE7;color:#166534;font-weight:600;"
    if v <= 3:
        return "background-color:#FEF9C3;color:#854D0E;font-weight:600;"
    return "background-color:#FEE2E2;color:#991B1B;font-weight:700;"


def _clr_high_good(v):
    if pd.isna(v) or v == 0:
        return "background-color:#F8FAFC;color:#64748B;"
    if v >= 70:
        return "background-color:#DCFCE7;color:#166534;font-weight:600;"
    if v >= 50:
        return "background-color:#FEF9C3;color:#854D0E;font-weight:600;"
    return "background-color:#FEE2E2;color:#991B1B;font-weight:600;"


_clr_low_good = _clr_breach


def _clr_vol(v):
    if pd.isna(v) or v == 0:
        return "background-color:#F8FAFC;color:#64748B;"
    return "background-color:#EFF6FF;color:#1E40AF;font-weight:500;"


LM_PCT_FMT = {
    "Breach %": "{:.1f}%", "FAC %": "{:.1f}%", "ZRTO %": "{:.2f}%",
    "Conv %": "{:.1f}%", "COD Conv %": "{:.1f}%", "Prepaid Conv %": "{:.1f}%",
    "COD Share %": "{:.1f}%", "Prepaid Share %": "{:.1f}%",
}

PU_PCT_FMT = {
    "D0 %": "{:.1f}%", "D1 %": "{:.1f}%", "D2 %": "{:.1f}%", "D2+ %": "{:.1f}%",
}


def style_lm_overview(df, extra_fmt=None):
    fmt = {"Volume": "{:,.0f}", **LM_PCT_FMT}
    if extra_fmt:
        fmt.update(extra_fmt)
    cols = set(df.columns)
    styler = df.style
    if "Breach %" in cols:
        styler = styler.map(_clr_breach, subset=["Breach %"])
    if "ZRTO %" in cols:
        styler = styler.map(_clr_zrto, subset=["ZRTO %"])
    high_cols = [c for c in ("FAC %", "Conv %", "COD Conv %", "Prepaid Conv %") if c in cols]
    if high_cols:
        styler = styler.map(_clr_high_good, subset=high_cols)
    if "Volume" in cols:
        styler = styler.map(_clr_vol, subset=["Volume"])
    active_fmt = {k: v for k, v in fmt.items() if k in cols}
    styler = styler.format(active_fmt)
    return styler


def style_pu_overview(df, extra_fmt=None):
    fmt = {"Total Shipments": "{:,.0f}", **PU_PCT_FMT}
    if extra_fmt:
        fmt.update(extra_fmt)
    cols = set(df.columns)
    styler = df.style
    if "D0 %" in cols:
        styler = styler.map(_clr_high_good, subset=["D0 %"])
    if "D1 %" in cols:
        styler = styler.map(_clr_high_good, subset=["D1 %"])
    if "D2 %" in cols:
        styler = styler.map(_clr_low_good, subset=["D2 %"])
    if "D2+ %" in cols:
        styler = styler.map(_clr_low_good, subset=["D2+ %"])
    if "Total Shipments" in cols:
        styler = styler.map(_clr_vol, subset=["Total Shipments"])
    active_fmt = {k: v for k, v in fmt.items() if k in cols}
    styler = styler.format(active_fmt)
    return styler


# ═════════════════════════════════════════════════════════════════════════════
#  DATE HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def _parse_date_range(date_strs):
    try:
        mn = datetime.strptime(min(date_strs), "%Y%m%d").date()
        mx = datetime.strptime(max(date_strs), "%Y%m%d").date()
    except (ValueError, TypeError):
        mn = mx = datetime.now().date()
    return mn, mx


def _fmt_d(s):
    try:
        return datetime.strptime(str(s), "%Y%m%d").strftime("%d %b")
    except ValueError:
        return str(s)


# ═════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📦 Large Shipment Dashboard")
    st.divider()
    page = st.radio(
        "Navigation",
        ["Last Mile Performance", "Pickup Performance"],
        index=0,
        label_visibility="collapsed",
    )

# ═════════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ═════════════════════════════════════════════════════════════════════════════
try:
    lm_df = load_lm_data(LM_PATH)
except FileNotFoundError:
    st.error(f"Last-Mile file not found: `{LM_PATH}`")
    st.stop()

try:
    pu_df = load_pu_data(PU_PATH)
except FileNotFoundError:
    st.error(f"Pickup file not found: `{PU_PATH}`")
    st.stop()


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 1 — LAST MILE PERFORMANCE                                          ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
if page == "Last Mile Performance":
    filtered_df = lm_df

    kpis = lm_overall_kpis(filtered_df)
    state_table = lm_aggregate_by(filtered_df, "destination_state")
    seller_table = lm_aggregate_by(filtered_df, "seller_type")

    date_strs = sorted(filtered_df["reporting_date"].unique())
    min_d, max_d = _parse_date_range(date_strs)

    # ── KPI cards ─────────────────────────────────────────────────────────
    st.markdown(
        f"<div style='font-size:.8rem;color:#64748B;margin-bottom:8px'>"
        f"<b>{len(seller_table)}</b> sellers · <b>{len(state_table)}</b> states · "
        f"{min_d} to {max_d}</div>",
        unsafe_allow_html=True,
    )

    kc = st.columns(5)
    cards = [
        ("Total Volume", f"{kpis['Volume']:,}",       "Total shipments",  ""),
        ("Conv %",       f"{kpis['Conv %']:.1f}%",     "Conversion rate",  "green"),
        ("Breach %",     f"{kpis['Breach %']:.1f}%",   "SLA breach rate",  "red"),
        ("ZRTO %",       f"{kpis['ZRTO %']:.2f}%",    "Zero-attempt RTO", "orange"),
        ("FAC %",        f"{kpis['FAC %']:.1f}%",      "1st attempt conv", "purple"),
    ]
    for col, (lbl, val, sub, cls) in zip(kc, cards):
        with col:
            st.markdown(
                f'<div class="kpi-card {cls}">'
                f'<div class="kpi-label">{lbl}</div>'
                f'<div class="kpi-value">{val}</div>'
                f'<div class="kpi-sub">{sub}</div></div>',
                unsafe_allow_html=True,
            )
    st.markdown("")

    # ── Overview tables — State & Seller tabs ─────────────────────────────
    LM_DISPLAY_COLS = [
        "Volume", "Breach %", "FAC %", "ZRTO %", "Conv %",
        "COD Conv %", "Prepaid Conv %", "COD Share %", "Prepaid Share %",
    ]

    def _lm_display_table(agg_df, label_col, label_name, tab_key):
        search = st.text_input(
            f"Search {label_name.lower()}",
            placeholder=f"🔍  Filter by {label_name.lower()} name …",
            label_visibility="collapsed",
            key=f"lm_search_{tab_key}",
        )
        disp = agg_df.rename(columns={label_col: label_name, "PHin": "Volume"})
        show = [label_name] + [c for c in LM_DISPLAY_COLS if c in disp.columns]
        disp = disp[show]
        if search:
            disp = disp[disp[label_name].astype(str).str.upper().str.contains(search.strip().upper())]
        st.dataframe(
            style_lm_overview(disp),
            width="stretch",
            height=min(460, 38 + 35 * len(disp)),
            hide_index=True,
        )
        st.download_button(
            f"⬇ Download {label_name} table",
            disp.to_csv(index=False).encode(),
            file_name=f"lm_{label_name.lower()}_performance.csv",
            mime="text/csv",
            key=f"lm_dl_{tab_key}",
        )

    tab_state, tab_seller = st.tabs(["🗺  State Overview", "🏪  Seller Overview"])

    with tab_state:
        st.markdown(
            '<div class="section-hdr">'
            '<span class="ico">🗺</span>'
            '<div><div class="ttl">State-wise Performance</div>'
            '<div class="sub">Aggregated last-mile metrics per destination state</div></div></div>',
            unsafe_allow_html=True,
        )
        _lm_display_table(state_table, "destination_state", "State", "state")

    with tab_seller:
        st.markdown(
            '<div class="section-hdr">'
            '<span class="ico">🏪</span>'
            '<div><div class="ttl">Seller-wise Performance</div>'
            '<div class="sub">Aggregated last-mile metrics per seller type</div></div></div>',
            unsafe_allow_html=True,
        )
        _lm_display_table(seller_table, "seller_type", "Seller", "seller")

    # ── Seller drill-down ─────────────────────────────────────────────────
    st.divider()
    st.markdown(
        '<div class="drill-hdr">'
        '<div class="ttl">🔎  Seller Drill-Down</div>'
        '<div class="sub">Select a seller to view detailed state-wise and date-wise performance. '
        'This section re-renders independently for a lag-free experience.</div></div>',
        unsafe_allow_html=True,
    )

    @st.fragment
    def lm_seller_drilldown():
        st.markdown(
            "<div style='font-size:.8rem;color:#64748B;margin-bottom:4px'>"
            "<b>Step 1</b> — Choose a date range for the drill-down</div>",
            unsafe_allow_html=True,
        )
        avail_dates = sorted(filtered_df["reporting_date"].unique())
        drill_min, drill_max = _parse_date_range(avail_dates)

        dc1, dc2, dc3 = st.columns([1, 1, 2])
        with dc1:
            drill_start = st.date_input(
                "From", value=drill_min, min_value=drill_min, max_value=drill_max,
                key="lm_drill_from",
            )
        with dc2:
            drill_end = st.date_input(
                "To", value=drill_max, min_value=drill_min, max_value=drill_max,
                key="lm_drill_to",
            )
        if drill_start > drill_end:
            drill_start, drill_end = drill_end, drill_start

        ds = drill_start.strftime("%Y%m%d")
        de = drill_end.strftime("%Y%m%d")
        date_scoped = filtered_df[
            (filtered_df["reporting_date"] >= ds)
            & (filtered_df["reporting_date"] <= de)
        ]

        sellers_in_range = sorted(date_scoped["seller_type"].unique())
        with dc3:
            st.markdown(
                "<div style='font-size:.8rem;color:#64748B;margin-bottom:4px'>"
                "<b>Step 2</b> — Pick a seller</div>",
                unsafe_allow_html=True,
            )
            chosen = st.selectbox(
                "Choose seller",
                options=["\u2014 Select a seller \u2014"] + sellers_in_range,
                key="lm_drill_sel",
                label_visibility="collapsed",
            )

        if chosen == "\u2014 Select a seller \u2014":
            return

        sdf = date_scoped[date_scoped["seller_type"] == chosen]
        if sdf.empty:
            st.warning("No data for the selected seller in this date range.")
            return

        sk = lm_overall_kpis(sdf)
        sc = st.columns(5)
        s_cards = [
            ("Volume",   f"{sk['Volume']:,}",       ""),
            ("Conv %",   f"{sk['Conv %']:.1f}%",     "green"),
            ("Breach %", f"{sk['Breach %']:.1f}%",   "red"),
            ("ZRTO %",   f"{sk['ZRTO %']:.2f}%",    "orange"),
            ("FAC %",    f"{sk['FAC %']:.1f}%",      "purple"),
        ]
        for col, (lbl, val, cls) in zip(sc, s_cards):
            with col:
                st.markdown(
                    f'<div class="kpi-card {cls}">'
                    f'<div class="kpi-label">{lbl}</div>'
                    f'<div class="kpi-value">{val}</div></div>',
                    unsafe_allow_html=True,
                )

        state_detail = lm_aggregate_by(sdf, "destination_state")
        state_detail = state_detail[state_detail["PHin"] > 0]
        disp_state = state_detail.rename(columns={"destination_state": "State", "PHin": "Volume"})
        show_cols = ["State"] + [c for c in LM_DISPLAY_COLS if c in disp_state.columns]
        disp_state = disp_state[show_cols]

        state_search = st.text_input(
            "Search state", placeholder="🔍  Filter states …",
            label_visibility="collapsed", key="lm_drill_state_search",
        )
        if state_search:
            disp_state = disp_state[
                disp_state["State"].astype(str).str.upper().str.contains(state_search.strip().upper())
            ]

        st.dataframe(
            style_lm_overview(disp_state),
            width="stretch",
            height=min(420, 38 + 35 * len(disp_state)),
            hide_index=True,
        )
        st.download_button(
            "⬇ Download state breakdown",
            disp_state.to_csv(index=False).encode(),
            file_name=f"{chosen}_state_breakdown.csv",
            mime="text/csv",
            key="lm_dl_drill_state",
        )

        st.divider()
        states_available = (
            sorted(state_detail["destination_state"].unique())
            if "destination_state" in state_detail.columns
            else sorted(disp_state["State"].unique())
        )
        st.markdown(
            "<div style='font-size:.8rem;color:#64748B;margin-bottom:4px'>"
            "<b>Step 3</b> — Select a state to view day-wise trend for this seller</div>",
            unsafe_allow_html=True,
        )
        chosen_state = st.selectbox(
            "Choose state",
            options=["\u2014 Select a state \u2014"] + states_available,
            key="lm_drill_state_sel",
            label_visibility="collapsed",
        )

        if chosen_state != "\u2014 Select a state \u2014":
            state_df = sdf[sdf["destination_state"] == chosen_state]
            if state_df.empty:
                st.warning(f"No data for **{chosen}** in **{chosen_state}**.")
            else:
                day_agg = lm_aggregate_by(state_df, "reporting_date", with_payment_split=False)
                day_agg = day_agg.sort_values("reporting_date")
                day_agg["Date"] = day_agg["reporting_date"].apply(_fmt_d)
                disp_day = day_agg.rename(columns={"PHin": "Volume"})
                day_cols = ["Date", "Volume", "Breach %", "FAC %", "ZRTO %", "Conv %"]
                disp_day = disp_day[[c for c in day_cols if c in disp_day.columns]]

                st.markdown(
                    f"<div style='font-size:.82rem;color:#64748B;margin-bottom:6px'>"
                    f"Day-wise trend for <b>{chosen}</b> in <b>{chosen_state}</b> "
                    f"({drill_start} \u2192 {drill_end})</div>",
                    unsafe_allow_html=True,
                )
                st.dataframe(
                    style_lm_overview(disp_day),
                    width="stretch",
                    height=min(400, 38 + 35 * len(disp_day)),
                    hide_index=True,
                )
                st.download_button(
                    "⬇ Download day-wise state trend",
                    disp_day.to_csv(index=False).encode(),
                    file_name=f"{chosen}_{chosen_state}_daily.csv",
                    mime="text/csv",
                    key="lm_dl_drill_state_day",
                )

    lm_seller_drilldown()


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 2 — PICKUP PERFORMANCE                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
elif page == "Pickup Performance":
    filtered_pu = pu_df

    kpis = pu_overall_kpis(filtered_pu)
    state_table = pu_aggregate_by(filtered_pu, "source_state")
    seller_table = pu_aggregate_by(filtered_pu, "seller_type")

    date_strs = sorted(filtered_pu["reporting_date"].unique())
    min_d, max_d = _parse_date_range(date_strs)

    # ── KPI cards ─────────────────────────────────────────────────────────
    st.markdown(
        f"<div style='font-size:.8rem;color:#64748B;margin-bottom:8px'>"
        f"<b>{len(seller_table)}</b> sellers · <b>{len(state_table)}</b> states · "
        f"{min_d} to {max_d}</div>",
        unsafe_allow_html=True,
    )

    kc = st.columns(5)
    cards = [
        ("Total Shipments", f"{kpis['Total Shipments']:,}", "Pickup volume",     ""),
        ("D0 %",            f"{kpis['D0 %']:.1f}%",         "Same-day pickup",   "green"),
        ("D1 %",            f"{kpis['D1 %']:.1f}%",         "Next-day pickup",   "cyan"),
        ("D2 %",            f"{kpis['D2 %']:.1f}%",         "Day-2 pickup",      "orange"),
        ("D2+ %",           f"{kpis['D2+ %']:.1f}%",        "Delayed pickup",    "red"),
    ]
    for col, (lbl, val, sub, cls) in zip(kc, cards):
        with col:
            st.markdown(
                f'<div class="kpi-card {cls}">'
                f'<div class="kpi-label">{lbl}</div>'
                f'<div class="kpi-value">{val}</div>'
                f'<div class="kpi-sub">{sub}</div></div>',
                unsafe_allow_html=True,
            )
    st.markdown("")

    # ── Overview tables — State & Seller tabs ─────────────────────────────
    PU_DISPLAY_COLS = ["Total Shipments", "D0 %", "D1 %", "D2 %", "D2+ %"]

    def _pu_display_table(agg_df, label_col, label_name, tab_key):
        search = st.text_input(
            f"Search {label_name.lower()}",
            placeholder=f"🔍  Filter by {label_name.lower()} name …",
            label_visibility="collapsed",
            key=f"pu_search_{tab_key}",
        )
        disp = agg_df.rename(columns={label_col: label_name, "total_shipments": "Total Shipments"})
        show = [label_name] + [c for c in PU_DISPLAY_COLS if c in disp.columns]
        disp = disp[show]
        if search:
            disp = disp[disp[label_name].astype(str).str.upper().str.contains(search.strip().upper())]
        st.dataframe(
            style_pu_overview(disp),
            width="stretch",
            height=min(460, 38 + 35 * len(disp)),
            hide_index=True,
        )
        st.download_button(
            f"⬇ Download {label_name} table",
            disp.to_csv(index=False).encode(),
            file_name=f"pu_{label_name.lower()}_performance.csv",
            mime="text/csv",
            key=f"pu_dl_{tab_key}",
        )

    tab_state, tab_seller = st.tabs(["🗺  State Overview", "🏪  Seller Overview"])

    with tab_state:
        st.markdown(
            '<div class="section-hdr">'
            '<span class="ico">🗺</span>'
            '<div><div class="ttl">State-wise Pickup Performance</div>'
            '<div class="sub">Aggregated pickup metrics per source state</div></div></div>',
            unsafe_allow_html=True,
        )
        _pu_display_table(state_table, "source_state", "State", "state")

    with tab_seller:
        st.markdown(
            '<div class="section-hdr">'
            '<span class="ico">🏪</span>'
            '<div><div class="ttl">Seller-wise Pickup Performance</div>'
            '<div class="sub">Aggregated pickup metrics per seller type</div></div></div>',
            unsafe_allow_html=True,
        )
        _pu_display_table(seller_table, "seller_type", "Seller", "seller")

    # ── Seller drill-down ─────────────────────────────────────────────────
    st.divider()
    st.markdown(
        '<div class="drill-hdr">'
        '<div class="ttl">🔎  Seller Drill-Down</div>'
        '<div class="sub">Select a seller to view detailed state-wise and date-wise pickup performance. '
        'This section re-renders independently for a lag-free experience.</div></div>',
        unsafe_allow_html=True,
    )

    @st.fragment
    def pu_seller_drilldown():
        st.markdown(
            "<div style='font-size:.8rem;color:#64748B;margin-bottom:4px'>"
            "<b>Step 1</b> — Choose a date range for the drill-down</div>",
            unsafe_allow_html=True,
        )
        avail_dates = sorted(filtered_pu["reporting_date"].unique())
        drill_min, drill_max = _parse_date_range(avail_dates)

        dc1, dc2, dc3 = st.columns([1, 1, 2])
        with dc1:
            drill_start = st.date_input(
                "From", value=drill_min, min_value=drill_min, max_value=drill_max,
                key="pu_drill_from",
            )
        with dc2:
            drill_end = st.date_input(
                "To", value=drill_max, min_value=drill_min, max_value=drill_max,
                key="pu_drill_to",
            )
        if drill_start > drill_end:
            drill_start, drill_end = drill_end, drill_start

        ds = drill_start.strftime("%Y%m%d")
        de = drill_end.strftime("%Y%m%d")
        date_scoped = filtered_pu[
            (filtered_pu["reporting_date"] >= ds)
            & (filtered_pu["reporting_date"] <= de)
        ]

        sellers_in_range = sorted(date_scoped["seller_type"].unique())
        with dc3:
            st.markdown(
                "<div style='font-size:.8rem;color:#64748B;margin-bottom:4px'>"
                "<b>Step 2</b> — Pick a seller</div>",
                unsafe_allow_html=True,
            )
            chosen = st.selectbox(
                "Choose seller",
                options=["\u2014 Select a seller \u2014"] + sellers_in_range,
                key="pu_drill_sel",
                label_visibility="collapsed",
            )

        if chosen == "\u2014 Select a seller \u2014":
            return

        sdf = date_scoped[date_scoped["seller_type"] == chosen]
        if sdf.empty:
            st.warning("No data for the selected seller in this date range.")
            return

        sk = pu_overall_kpis(sdf)
        sc = st.columns(5)
        s_cards = [
            ("Total Shipments", f"{sk['Total Shipments']:,}", ""),
            ("D0 %",            f"{sk['D0 %']:.1f}%",         "green"),
            ("D1 %",            f"{sk['D1 %']:.1f}%",         "cyan"),
            ("D2 %",            f"{sk['D2 %']:.1f}%",         "orange"),
            ("D2+ %",           f"{sk['D2+ %']:.1f}%",        "red"),
        ]
        for col, (lbl, val, cls) in zip(sc, s_cards):
            with col:
                st.markdown(
                    f'<div class="kpi-card {cls}">'
                    f'<div class="kpi-label">{lbl}</div>'
                    f'<div class="kpi-value">{val}</div></div>',
                    unsafe_allow_html=True,
                )

        state_detail = pu_aggregate_by(sdf, "source_state")
        state_detail = state_detail[state_detail["total_shipments"] > 0]
        disp_state = state_detail.rename(
            columns={"source_state": "State", "total_shipments": "Total Shipments"},
        )
        show_cols = ["State"] + [c for c in PU_DISPLAY_COLS if c in disp_state.columns]
        disp_state = disp_state[show_cols]

        state_search = st.text_input(
            "Search state", placeholder="🔍  Filter states …",
            label_visibility="collapsed", key="pu_drill_state_search",
        )
        if state_search:
            disp_state = disp_state[
                disp_state["State"].astype(str).str.upper().str.contains(state_search.strip().upper())
            ]

        st.dataframe(
            style_pu_overview(disp_state),
            width="stretch",
            height=min(420, 38 + 35 * len(disp_state)),
            hide_index=True,
        )
        st.download_button(
            "⬇ Download state breakdown",
            disp_state.to_csv(index=False).encode(),
            file_name=f"{chosen}_pickup_state_breakdown.csv",
            mime="text/csv",
            key="pu_dl_drill_state",
        )

        st.divider()
        states_available = (
            sorted(state_detail["source_state"].unique())
            if "source_state" in state_detail.columns
            else sorted(disp_state["State"].unique())
        )
        st.markdown(
            "<div style='font-size:.8rem;color:#64748B;margin-bottom:4px'>"
            "<b>Step 3</b> — Select a state to view day-wise pickup trend for this seller</div>",
            unsafe_allow_html=True,
        )
        chosen_state = st.selectbox(
            "Choose state",
            options=["\u2014 Select a state \u2014"] + states_available,
            key="pu_drill_state_sel",
            label_visibility="collapsed",
        )

        if chosen_state != "\u2014 Select a state \u2014":
            state_df = sdf[sdf["source_state"] == chosen_state]
            if state_df.empty:
                st.warning(f"No data for **{chosen}** in **{chosen_state}**.")
            else:
                day_agg = pu_aggregate_by(state_df, "reporting_date")
                day_agg = day_agg.sort_values("reporting_date")
                day_agg["Date"] = day_agg["reporting_date"].apply(_fmt_d)
                disp_day = day_agg.rename(columns={"total_shipments": "Total Shipments"})
                day_cols = ["Date", "Total Shipments", "D0 %", "D1 %", "D2 %", "D2+ %"]
                disp_day = disp_day[[c for c in day_cols if c in disp_day.columns]]

                st.markdown(
                    f"<div style='font-size:.82rem;color:#64748B;margin-bottom:6px'>"
                    f"Day-wise pickup trend for <b>{chosen}</b> in <b>{chosen_state}</b> "
                    f"({drill_start} \u2192 {drill_end})</div>",
                    unsafe_allow_html=True,
                )
                st.dataframe(
                    style_pu_overview(disp_day),
                    width="stretch",
                    height=min(400, 38 + 35 * len(disp_day)),
                    hide_index=True,
                )
                st.download_button(
                    "⬇ Download day-wise state trend",
                    disp_day.to_csv(index=False).encode(),
                    file_name=f"{chosen}_{chosen_state}_pickup_daily.csv",
                    mime="text/csv",
                    key="pu_dl_drill_state_day",
                )

    pu_seller_drilldown()


# ═════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ═════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown(
    "<div style='text-align:center;color:#94A3B8;font-size:.7rem'>"
    "Large Shipment Dashboard · Data refreshed on load</div>",
    unsafe_allow_html=True,
)
