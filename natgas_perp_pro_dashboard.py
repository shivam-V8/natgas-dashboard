"""
NatGas Perp Pro Dashboard
=========================
A professional, real-time analytics dashboard for the Bitget NATGASUSDT perpetual
swap, with full Henry Hub (NG=F) momentum comparison, EIA storage fundamentals,
and funding/OI analytics. Read-only — no trade execution.

Bitget is used as the perp data source because Binance returns HTTP 451 from
US-hosted servers (e.g. Streamlit Cloud). Bitget lists the same NATGAS/USDT:USDT
instrument and has no US geo-block on public endpoints.

Run with:
    streamlit run natgas_perp_pro_dashboard.py

Tech: Streamlit + Plotly + ccxt + yfinance + EIA v2 API.

DISCLAIMER: NOT financial advice. Crypto futures are high-risk.
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import io
import time
import math
import json
import datetime as dt
from datetime import timezone, timedelta
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Optional imports - wrapped in try/except so the app degrades gracefully
try:
    import ccxt
    CCXT_OK = True
except Exception:
    CCXT_OK = False

try:
    import yfinance as yf
    YF_OK = True
except Exception:
    YF_OK = False


# =============================================================================
# PAGE CONFIG & CUSTOM DARK CSS
# =============================================================================
st.set_page_config(
    page_title="NatGas Perp Pro Dashboard",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
/* --- Global dark trading theme --- */
.stApp {
    background: linear-gradient(180deg, #0b0f1a 0%, #0e1320 100%);
    color: #e6e9ef;
}
/* Hide sidebar entirely for the public read-only dashboard */
section[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
h1, h2, h3, h4 { color: #e6e9ef; letter-spacing: 0.2px; }

/* Big price tile */
.big-price {
    font-size: 44px; font-weight: 800; line-height: 1.0;
    font-variant-numeric: tabular-nums;
}
.sub-price { font-size: 16px; color: #9aa3b2; font-weight: 500; }
.up   { color: #16c784 !important; }
.down { color: #ea3943 !important; }
.flat { color: #9aa3b2 !important; }

/* Card */
.card {
    background: #111726;
    border: 1px solid #1e2533;
    border-radius: 12px;
    padding: 14px 16px;
    box-shadow: 0 1px 0 rgba(255,255,255,0.02) inset;
}
.card h4 { margin: 0 0 6px 0; font-size: 13px; color: #9aa3b2; text-transform: uppercase; letter-spacing: 0.6px; }
.card .val { font-size: 22px; font-weight: 700; font-variant-numeric: tabular-nums; }
.card .delta { font-size: 13px; font-weight: 600; }

/* Banner */
.banner {
    padding: 10px 14px; border-radius: 10px; font-weight: 700; text-align: center;
    border: 1px solid transparent;
}
.banner-green { background: rgba(22,199,132,0.10); color: #16c784; border-color: rgba(22,199,132,0.35); }
.banner-red   { background: rgba(234,57,67,0.10);  color: #ea3943; border-color: rgba(234,57,67,0.35); }
.banner-amber { background: rgba(255,176,32,0.10); color: #ffb020; border-color: rgba(255,176,32,0.35); }
.banner-info  { background: rgba(99,138,255,0.10); color: #8aa6ff; border-color: rgba(99,138,255,0.35); }

/* Footer */
.footer {
    text-align: center; color: #6b7280; font-size: 12px; padding: 10px 0 4px 0;
    border-top: 1px solid #1e2533; margin-top: 18px;
}

/* Tables */
.dataframe tbody tr th, .dataframe tbody tr td { color: #d6dbe6 !important; }
hr { border-color: #1e2533; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =============================================================================
# CONSTANTS
# =============================================================================
PERP_EXCHANGE_ID = "bitget"                        # ccxt exchange id for the perp source
PERP_SYMBOL_CCXT = "NATGAS/USDT:USDT"              # ccxt unified perpetual symbol
PERP_VENUE_LABEL = "Bitget"                        # human label used in UI
HENRY_HUB_TICKER = "NG=F"                          # Yahoo Finance front-month NG futures
EIA_SERIES_LOWER48 = "NG.NW2_EPG0_SWO_R48_BCF.W"   # Lower-48 weekly working gas

# Fixed runtime config (previously sidebar-driven)
REFRESH_SEC        = 60      # auto-refresh interval
ALERT_FUNDING_BPS  = 0.05    # banner if |funding| > 0.05% per 8h
ALERT_OI_PCT       = 5.0     # banner if |OI 24h delta| > 5%


def _get_eia_key() -> str:
    """Resolve the EIA API key from Streamlit secrets, then env var. Never from source."""
    try:
        v = st.secrets.get("EIA_API_KEY")  # type: ignore[attr-defined]
        if v:
            return str(v)
    except Exception:
        pass
    return os.environ.get("EIA_API_KEY", "")

EIA_API_KEY = _get_eia_key()


# =============================================================================
# UTILITY HELPERS
# =============================================================================
def _fmt_num(x: Optional[float], dp: int = 4, dash: str = "—") -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return dash
    try:
        return f"{x:,.{dp}f}"
    except Exception:
        return dash

def _pct_class(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "flat"
    return "up" if x > 0 else ("down" if x < 0 else "flat")

def _arrow(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return "▲" if x > 0 else ("▼" if x < 0 else "▬")


# =============================================================================
# DATA FETCHERS (all cached, all wrapped in try/except)
# =============================================================================
@st.cache_data(ttl=60, show_spinner=False)
def fetch_henry_hub() -> Dict[str, Any]:
    """Yahoo Finance NG=F front-month natural gas futures - last 5d daily."""
    out = {"ok": False, "price": None, "change_pct": None, "df": None, "error": None}
    if not YF_OK:
        out["error"] = "yfinance not installed"
        return out
    try:
        t = yf.Ticker(HENRY_HUB_TICKER)
        df = t.history(period="5d", interval="1d", auto_adjust=False)
        if df is None or df.empty:
            out["error"] = "No Henry Hub data returned"
            return out
        df = df.reset_index()
        last = float(df["Close"].iloc[-1])
        prev = float(df["Close"].iloc[-2]) if len(df) >= 2 else last
        out.update(ok=True, price=last,
                   change_pct=((last - prev) / prev * 100.0) if prev else 0.0,
                   df=df)
    except Exception as e:
        out["error"] = str(e)
    return out


@st.cache_data(ttl=300, show_spinner=False)
def fetch_henry_hub_intraday() -> Optional[pd.DataFrame]:
    """15m candles for the chart in the Momentum tab."""
    if not YF_OK:
        return None
    try:
        df = yf.download(HENRY_HUB_TICKER, period="5d", interval="15m",
                         progress=False, auto_adjust=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df.reset_index().rename(columns={
            "Datetime": "time", "Date": "time",
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        })
        return df.tail(100).reset_index(drop=True)
    except Exception:
        return None


def _ccxt_client():
    """Build a public ccxt client for the perp source (Bitget swap)."""
    if not CCXT_OK:
        return None
    klass = getattr(ccxt, PERP_EXCHANGE_ID, None)
    if klass is None:
        return None
    return klass({"enableRateLimit": True, "options": {"defaultType": "swap"}})


@st.cache_data(ttl=20, show_spinner=False)
def fetch_perp_ticker() -> Dict[str, Any]:
    """24h ticker for the perp via ccxt unified API."""
    out = {"ok": False, "error": None}
    ex = _ccxt_client()
    if ex is None:
        out["error"] = "ccxt unavailable"
        return out
    try:
        t = ex.fetch_ticker(PERP_SYMBOL_CCXT)
        out.update(
            ok=True,
            last=t.get("last"),
            bid=t.get("bid"),
            ask=t.get("ask"),
            high=t.get("high"),
            low=t.get("low"),
            change_pct=t.get("percentage"),
            quote_volume=t.get("quoteVolume"),
            base_volume=t.get("baseVolume"),
        )
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


@st.cache_data(ttl=30, show_spinner=False)
def fetch_funding_now() -> Dict[str, Any]:
    """Current funding rate + next funding time via ccxt unified API."""
    out = {"ok": False, "error": None}
    ex = _ccxt_client()
    if ex is None:
        out["error"] = "ccxt unavailable"
        return out
    try:
        fr = ex.fetch_funding_rate(PERP_SYMBOL_CCXT)
        out.update(
            ok=True,
            funding_rate=fr.get("fundingRate"),
            next_funding_ts=fr.get("nextFundingTimestamp") or fr.get("fundingTimestamp"),
            mark_price=fr.get("markPrice"),
            index_price=fr.get("indexPrice"),
        )
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


@st.cache_data(ttl=120, show_spinner=False)
def fetch_funding_history(limit: int = 50) -> Optional[pd.DataFrame]:
    """Last N funding settlements via ccxt unified API."""
    ex = _ccxt_client()
    if ex is None:
        return None
    try:
        rows = ex.fetch_funding_rate_history(PERP_SYMBOL_CCXT, limit=limit)
        if not rows:
            return None
        df = pd.DataFrame([{
            "fundingTime": r.get("timestamp"),
            "fundingRate": r.get("fundingRate"),
        } for r in rows])
        df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
        df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
        return df.dropna().sort_values("fundingTime").reset_index(drop=True)
    except Exception:
        return None


@st.cache_data(ttl=30, show_spinner=False)
def fetch_open_interest() -> Dict[str, Any]:
    """Latest OI snapshot via ccxt unified API."""
    out = {"ok": False, "error": None}
    ex = _ccxt_client()
    if ex is None:
        out["error"] = "ccxt unavailable"
        return out
    try:
        oi = ex.fetch_open_interest(PERP_SYMBOL_CCXT)
        # ccxt returns slightly different shapes per exchange; prefer base contracts amount
        val = (oi.get("openInterestAmount") or oi.get("openInterestValue")
               or oi.get("openInterest") or 0.0)
        out.update(ok=True, open_interest=float(val), ts=oi.get("timestamp"))
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


@st.cache_data(ttl=120, show_spinner=False)
def fetch_oi_history(period: str = "5m", limit: int = 288) -> Optional[pd.DataFrame]:
    """OI history — not supported on Bitget via ccxt. Returns None so the UI degrades."""
    return None


@st.cache_data(ttl=120, show_spinner=False)
def fetch_long_short(endpoint: str, period: str = "5m", limit: int = 60) -> Optional[pd.DataFrame]:
    """L/S ratio endpoints — Binance-specific, not available on Bitget. Returns None."""
    return None


@st.cache_data(ttl=60, show_spinner=False)
def fetch_liquidations(limit: int = 20) -> Optional[pd.DataFrame]:
    """Public liquidations — not exposed by Bitget. Returns None so UI shows the fallback."""
    return None


@st.cache_data(ttl=60, show_spinner=False)
def fetch_ohlcv(timeframe: str = "15m", limit: int = 100) -> Optional[pd.DataFrame]:
    """OHLCV via ccxt for the perp."""
    ex = _ccxt_client()
    if ex is None:
        return None
    try:
        raw = ex.fetch_ohlcv(PERP_SYMBOL_CCXT, timeframe=timeframe, limit=limit)
        if not raw:
            return None
        df = pd.DataFrame(raw, columns=["time", "open", "high", "low", "close", "volume"])
        df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
        return df
    except Exception:
        return None


@st.cache_data(ttl=900, show_spinner=False)
def fetch_eia_storage(api_key: str) -> Dict[str, Any]:
    """EIA v2 weekly working gas in storage - Lower 48."""
    out = {"ok": False, "error": None}
    if not api_key:
        out["error"] = "EIA_API_KEY not configured (set in Streamlit secrets or env)."
        return out
    try:
        url = "https://api.eia.gov/v2/natural-gas/stor/wkly/data/"
        params = {
            "api_key": api_key,
            "frequency": "weekly",
            "data[0]": "value",
            "facets[series][]": EIA_SERIES_LOWER48,
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
            "offset": 0,
            "length": 260,  # ~5 years
        }
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        j = r.json()
        rows = j.get("response", {}).get("data", [])
        if not rows:
            out["error"] = "EIA returned no rows"
            return out
        df = pd.DataFrame(rows)
        df["period"] = pd.to_datetime(df["period"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.sort_values("period").reset_index(drop=True)

        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) >= 2 else None
        wow = (latest["value"] - prev["value"]) if prev is not None else None

        # 5-year average for the same week-of-year (excluding current year)
        df["woy"] = df["period"].dt.isocalendar().week
        same_woy = df[df["woy"] == latest["period"].isocalendar().week]
        cur_year = latest["period"].year
        five_y = same_woy[(same_woy["period"].dt.year >= cur_year - 5) &
                          (same_woy["period"].dt.year < cur_year)]
        five_y_avg = float(five_y["value"].mean()) if not five_y.empty else None

        out.update(
            ok=True,
            df=df,
            latest_value=float(latest["value"]),
            latest_period=latest["period"].date(),
            wow_change=float(wow) if wow is not None else None,
            five_y_avg=five_y_avg,
            vs_5yr=(float(latest["value"]) - five_y_avg) if five_y_avg else None,
        )
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


# =============================================================================
# TECHNICAL INDICATORS (pure pandas/numpy - no extra deps)
# =============================================================================
def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def macd(close: pd.Series, fast=12, slow=26, signal=9):
    line = _ema(close, fast) - _ema(close, slow)
    sig = _ema(line, signal)
    hist = line - sig
    return line, sig, hist

def bollinger(close: pd.Series, n=20, k=2):
    m = close.rolling(n).mean()
    s = close.rolling(n).std()
    return m, m + k * s, m - k * s

def stochastic(high, low, close, k=14, d=3):
    ll = low.rolling(k).min()
    hh = high.rolling(k).max()
    kf = 100 * (close - ll) / (hh - ll).replace(0, np.nan)
    return kf, kf.rolling(d).mean()

def adx(high, low, close, n=14) -> pd.Series:
    up_move = high.diff()
    dn_move = -low.diff()
    plus_dm = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/n, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(alpha=1/n, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(alpha=1/n, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1/n, adjust=False).mean()

def momentum(close: pd.Series, n: int = 10) -> pd.Series:
    return close.diff(n)

def roc(close: pd.Series, n: int = 10) -> pd.Series:
    return (close / close.shift(n) - 1.0) * 100.0


def composite_momentum_score(df: pd.DataFrame) -> Optional[float]:
    """
    Combine RSI, MACD-hist sign, Stoch-K, ROC into a 0-100 score.
    >55 bullish, <45 bearish.
    """
    if df is None or df.empty or "close" not in df.columns:
        return None
    try:
        c = df["close"].astype(float)
        h = df["high"].astype(float) if "high" in df else c
        l = df["low"].astype(float) if "low" in df else c
        r = rsi(c).iloc[-1]
        ml, ms, mh = macd(c)
        macd_h = mh.iloc[-1]
        macd_norm = 50 + np.tanh(macd_h / max(c.iloc[-1] * 0.001, 1e-9)) * 25
        k, d = stochastic(h, l, c)
        st_k = k.iloc[-1]
        ro = roc(c).iloc[-1]
        roc_norm = 50 + np.clip(ro * 5, -50, 50)
        components = [v for v in (r, macd_norm, st_k, roc_norm) if pd.notna(v)]
        if not components:
            return None
        return float(np.clip(np.mean(components), 0, 100))
    except Exception:
        return None


def momentum_edge_score(perp_score, hh_score, funding_rate, oi_delta_pct, basis_pct) -> Optional[float]:
    """
    Custom 'Momentum Edge Score' (-100 .. +100):

      40% : perp momentum bias vs neutral 50
      20% : divergence vs Henry Hub (perp leading HH = positive edge)
      15% : -funding (high positive funding = crowded longs = bearish edge)
      15% : OI delta (rising OI in direction of trend = confirmation)
      10% : basis (perp premium vs HH = froth = bearish edge)
    """
    if perp_score is None:
        return None
    try:
        bias = (perp_score - 50) * 2.0                                      # -100..100
        div = ((perp_score - hh_score) * 2.0) if hh_score is not None else 0
        fund_pen = -np.clip((funding_rate or 0) * 100 / 0.05, -100, 100) * 1.0
        oi_conf = np.clip((oi_delta_pct or 0) * (1 if bias >= 0 else -1) * 5, -100, 100)
        basis_pen = -np.clip((basis_pct or 0) * 10, -100, 100)
        edge = (0.40 * bias + 0.20 * div + 0.15 * fund_pen +
                0.15 * oi_conf + 0.10 * basis_pen)
        return float(np.clip(edge, -100, 100))
    except Exception:
        return None


# =============================================================================
# MASTER BULL/BEAR SIGNAL
# =============================================================================
# Research-grounded weights. Inputs and rationale:
#   1) Perp momentum  (RSI/MACD/Stoch/ROC composite)         25%  trend read
#   2) Henry Hub momentum (cash market confirmation)         10%  cash anchors perp
#   3) EIA storage vs 5-yr same-week avg                     15%  #1 NG fundamental
#   4) Seasonality (calendar bias)                           10%  heating/cooling cycle
#   5) Funding rate  (CONTRARIAN — crowded longs = bearish)  15%  perp-specific froth
#   6) OI 24h delta in direction of trend                    10%  trend confirmation
#   7) Long/Short ratio  (CONTRARIAN at extremes)            10%  positioning
#   8) Basis (perp − HH premium)  (premium = bearish froth)   5%  speculative excess
# Each component is normalized to [-100, +100]; weighted sum is clipped to that range.
# Buckets: ≥60 STRONG BULLISH · 25..60 BULLISH · ±25 NEUTRAL · -60..-25 BEARISH · ≤-60 STRONG BEARISH

def _seasonality_bias(month: int) -> Tuple[float, str]:
    """Henry Hub calendar bias (score in -100..+100)."""
    table = {
        1:  ( 55, "Peak winter heating demand"),
        2:  ( 40, "Late winter heating"),
        3:  (-15, "Withdrawal-season end (shoulder)"),
        4:  (-40, "Injection start, demand trough"),
        5:  (-30, "Injection season"),
        6:  ( 15, "Early cooling demand"),
        7:  ( 45, "Peak power-burn cooling"),
        8:  ( 40, "Cooling + hurricane risk"),
        9:  ( 20, "Hurricane peak / shoulder"),
        10: ( 10, "Pre-winter inventory build"),
        11: ( 35, "Heating demand ramps"),
        12: ( 55, "Peak winter heating demand"),
    }
    return table.get(month, (0.0, "—"))


def compute_master_signal(
    perp_momentum: Optional[float],         # 0..100
    hh_momentum: Optional[float],           # 0..100
    funding_rate: Optional[float],          # decimal (0.0005 = 0.05%/8h)
    oi_24h_pct: Optional[float],            # %
    ls_ratio: Optional[float],              # global long/short ratio
    basis_pct: Optional[float],             # perp premium vs HH, %
    storage_vs_5yr: Optional[float],        # Bcf delta vs 5-yr same-week avg
    storage_five_y_avg: Optional[float],    # baseline Bcf for normalization
    month: int,
) -> Dict[str, Any]:
    """Synthesize a unified -100..+100 bullish/bearish signal."""
    comps: List[Dict[str, Any]] = []

    def add(name, weight, score, detail):
        comps.append({"name": name, "weight": weight,
                      "score": float(np.clip(score, -100, 100)), "detail": detail})

    if perp_momentum is not None:
        add("Perp Momentum", 0.25, (perp_momentum - 50) * 2,
            f"Composite {perp_momentum:.0f}/100")
    if hh_momentum is not None:
        add("Henry Hub Momentum", 0.10, (hh_momentum - 50) * 2,
            f"Composite {hh_momentum:.0f}/100")
    if storage_vs_5yr is not None and storage_five_y_avg:
        pct = storage_vs_5yr / storage_five_y_avg * 100
        add("EIA Storage vs 5-yr", 0.15, -pct * 10,
            f"{storage_vs_5yr:+.0f} Bcf ({pct:+.1f}%)")
    s_score, s_label = _seasonality_bias(month)
    add("Seasonality", 0.10, s_score, s_label)
    if funding_rate is not None:
        fr_pct = funding_rate * 100
        add("Funding (contrarian)", 0.15, -fr_pct / 0.05 * 50,
            f"{fr_pct:+.4f}%/8h")
    if oi_24h_pct is not None and perp_momentum is not None:
        bias_sign = 1 if perp_momentum >= 50 else -1
        add("OI 24h confirmation", 0.10, oi_24h_pct * bias_sign * 8,
            f"{oi_24h_pct:+.2f}% (trend-aligned)" if bias_sign > 0
            else f"{oi_24h_pct:+.2f}% (counter-trend)")
    if ls_ratio is not None:
        score = -(ls_ratio - 1) * 60 if ls_ratio >= 1 else (1 - ls_ratio) * 100
        add("L/S Ratio (contrarian)", 0.10, score, f"{ls_ratio:.2f}")
    if basis_pct is not None:
        add("Basis (perp − HH)", 0.05, -basis_pct * 20, f"{basis_pct:+.2f}%")

    if not comps:
        return {"overall": None, "label": "NO DATA", "color": "info", "components": []}

    total_w = sum(c["weight"] for c in comps)
    overall = float(np.clip(
        sum(c["score"] * c["weight"] for c in comps) / total_w, -100, 100
    ))

    if   overall >=  60: label, color = "STRONG BULLISH", "green"
    elif overall >=  25: label, color = "BULLISH",        "green"
    elif overall <= -60: label, color = "STRONG BEARISH", "red"
    elif overall <= -25: label, color = "BEARISH",        "red"
    else:                label, color = "NEUTRAL",        "info"

    return {"overall": overall, "label": label, "color": color, "components": comps}


def render_master_signal(sig: Dict[str, Any]) -> None:
    """Hero panel: gauge on the left, weighted component breakdown on the right."""
    if sig["overall"] is None:
        st.markdown(
            "<div class='banner banner-info'>🧭 Master Signal: insufficient data</div>",
            unsafe_allow_html=True,
        )
        return

    overall, label, color = sig["overall"], sig["label"], sig["color"]
    bar_color = "#16c784" if color == "green" else ("#ea3943" if color == "red" else "#8aa6ff")

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=int(round(overall)),
        number={"font": {"size": 34, "color": "#e6e9ef"}, "valueformat": "d"},
        domain={"x": [0, 1], "y": [0.18, 1]},
        gauge={
            "axis": {"range": [-100, 100], "tickcolor": "#9aa3b2",
                     "tickvals": [-100, -60, -25, 0, 25, 60, 100],
                     "tickfont": {"size": 10}},
            "bar": {"color": bar_color, "thickness": 0.28},
            "bgcolor": "#0e1320",
            "borderwidth": 1, "bordercolor": "#1e2533",
            "steps": [
                {"range": [-100, -60], "color": "rgba(234,57,67,0.55)"},
                {"range": [-60,  -25], "color": "rgba(234,57,67,0.22)"},
                {"range": [-25,   25], "color": "rgba(154,163,178,0.15)"},
                {"range": [ 25,   60], "color": "rgba(22,199,132,0.22)"},
                {"range": [ 60,  100], "color": "rgba(22,199,132,0.55)"},
            ],
        },
    ))
    gauge.update_layout(
        height=280, margin=dict(l=20, r=20, t=20, b=10),
        paper_bgcolor="#0e1320", font={"color": "#e6e9ef"},
    )

    c_left, c_right = st.columns([1, 1.4])
    with c_left:
        st.markdown(
            f"<div class='banner banner-{color}'>🧭 Master Signal: {label} ({overall:+.0f})</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(gauge, use_container_width=True)
    with c_right:
        st.markdown("**Signal Breakdown** — weighted contribution to the overall score")
        rows = []
        for c in sig["components"]:
            arrow = "▲" if c["score"] > 5 else ("▼" if c["score"] < -5 else "▬")
            rows.append({
                "Factor": c["name"],
                "Weight": f"{c['weight']*100:.0f}%",
                "Score": f"{arrow} {c['score']:+.0f}",
                "Contribution": f"{c['score'] * c['weight']:+.1f}",
                "Detail": c["detail"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# =============================================================================
# HEADER & TOP STRIP
# =============================================================================
st.markdown("# 🔥 NatGas Perp Pro Dashboard")
st.caption(
    f"Live {PERP_VENUE_LABEL} NATGASUSDT perp · Henry Hub (NG=F) basis · Funding · OI · "
    "EIA storage"
)

# Pull "always-needed" data once at the top so the header can render
ticker = fetch_perp_ticker()
hh = fetch_henry_hub()
funding_now = fetch_funding_now()
oi_now = fetch_open_interest()
oi_hist = fetch_oi_history(period="5m", limit=288)
liqs = fetch_liquidations(limit=20)

perp_price = ticker.get("last") if ticker.get("ok") else None
perp_chg = ticker.get("change_pct") if ticker.get("ok") else None
hh_price = hh.get("price") if hh.get("ok") else None
hh_chg = hh.get("change_pct") if hh.get("ok") else None
basis = (perp_price - hh_price) if (perp_price is not None and hh_price is not None) else None
basis_pct = (basis / hh_price * 100.0) if (basis is not None and hh_price) else None

c1, c2, c3, c4 = st.columns([1.4, 1.0, 1.0, 1.2])
with c1:
    st.markdown(
        f"""
        <div class="card">
          <h4>NATGASUSDT Perp · {PERP_VENUE_LABEL}</h4>
          <div class="big-price {_pct_class(perp_chg)}">{_fmt_num(perp_price, 4)}</div>
          <div class="sub-price {_pct_class(perp_chg)}">
            {_arrow(perp_chg)} {_fmt_num(perp_chg, 2)}% (24h)
          </div>
        </div>
        """, unsafe_allow_html=True
    )
with c2:
    st.markdown(
        f"""
        <div class="card">
          <h4>Henry Hub (NG=F)</h4>
          <div class="val">{_fmt_num(hh_price, 3)}</div>
          <div class="delta {_pct_class(hh_chg)}">{_arrow(hh_chg)} {_fmt_num(hh_chg, 2)}%</div>
        </div>
        """, unsafe_allow_html=True
    )
with c3:
    st.markdown(
        f"""
        <div class="card">
          <h4>Basis (Perp − HH)</h4>
          <div class="val">{_fmt_num(basis, 4)}</div>
          <div class="delta {_pct_class(basis_pct)}">{_fmt_num(basis_pct, 2)}%</div>
        </div>
        """, unsafe_allow_html=True
    )
with c4:
    fr = funding_now.get("funding_rate") if funding_now.get("ok") else None
    fr_pct = (fr * 100) if fr is not None else None
    next_ts = funding_now.get("next_funding_ts") if funding_now.get("ok") else None
    if next_ts:
        delta = dt.datetime.fromtimestamp(next_ts/1000, tz=timezone.utc) - dt.datetime.now(timezone.utc)
        secs = max(int(delta.total_seconds()), 0)
        cd = f"{secs//3600:02d}:{(secs%3600)//60:02d}:{secs%60:02d}"
    else:
        cd = "—"
    st.markdown(
        f"""
        <div class="card">
          <h4>Funding (next in {cd})</h4>
          <div class="val {_pct_class(fr_pct)}">{_fmt_num(fr_pct, 4)}%</div>
          <div class="delta">per 8h settlement</div>
        </div>
        """, unsafe_allow_html=True
    )

# Banners for any missing critical data
warn_msgs = []
if not ticker.get("ok"):
    warn_msgs.append(f"Perp ticker: {ticker.get('error')}")
if not hh.get("ok"):
    warn_msgs.append(f"Henry Hub: {hh.get('error')}")
if warn_msgs:
    st.markdown(
        f"<div class='banner banner-amber'>⚠ Data Unavailable — {' · '.join(warn_msgs)}</div>",
        unsafe_allow_html=True,
    )


# =============================================================================
# MASTER SIGNAL — synthesize all data into a single Bull/Bear read
# =============================================================================
# These fetches/derivations are also consumed by individual tabs below.
perp_df_15m = fetch_ohlcv("15m", 100)
hh_df_15m = fetch_henry_hub_intraday()
gls_df = fetch_long_short("globalLongShortAccountRatio", "5m", 60)
tls_acc_df = fetch_long_short("topLongShortAccountRatio", "5m", 60)
tls_pos_df = fetch_long_short("topLongShortPositionRatio", "5m", 60)
eia_data = fetch_eia_storage(EIA_API_KEY)

perp_score_now = composite_momentum_score(perp_df_15m)
hh_score_now = composite_momentum_score(hh_df_15m)
fr_now = funding_now.get("funding_rate") if funding_now.get("ok") else None

# OI 24h % change from 5m history (288 bars ≈ 24h)
_oi_now_val = oi_now.get("open_interest") if oi_now.get("ok") else None
oi_24h_pct_now: Optional[float] = None
if oi_hist is not None and not oi_hist.empty and _oi_now_val:
    try:
        _past = oi_hist.iloc[max(0, len(oi_hist) - 288)]["sumOpenInterest"]
        oi_24h_pct_now = (_oi_now_val - _past) / _past * 100.0 if _past else None
    except Exception:
        pass

# Latest Global L/S ratio (most-recent bar)
gls_ratio_now: Optional[float] = None
if gls_df is not None and not gls_df.empty and "longShortRatio" in gls_df.columns:
    gls_ratio_now = float(gls_df["longShortRatio"].iloc[-1])

st.markdown("---")
st.markdown("### 🧭 Master Bull/Bear Signal")
st.caption(
    "Weighted composite of momentum (perp + Henry Hub), EIA storage vs 5-yr, "
    "seasonality, funding (contrarian), OI confirmation, L/S positioning (contrarian), "
    "and perp basis. Range: −100 (max bearish) to +100 (max bullish)."
)
master_sig = compute_master_signal(
    perp_momentum=perp_score_now,
    hh_momentum=hh_score_now,
    funding_rate=fr_now,
    oi_24h_pct=oi_24h_pct_now,
    ls_ratio=gls_ratio_now,
    basis_pct=basis_pct,
    storage_vs_5yr=eia_data.get("vs_5yr") if eia_data.get("ok") else None,
    storage_five_y_avg=eia_data.get("five_y_avg") if eia_data.get("ok") else None,
    month=dt.datetime.now(timezone.utc).month,
)
render_master_signal(master_sig)
st.markdown("---")


# =============================================================================
# TABS
# =============================================================================
tab_overview, tab_momentum, tab_funda = st.tabs([
    "📊 Live Overview",
    "🚀 Momentum Analyzer",
    "🌎 Fundamentals",
])


# ---------------------------------------------------------------------------
# TAB 1 - LIVE OVERVIEW
# ---------------------------------------------------------------------------
with tab_overview:
    st.subheader("Live Metrics")

    # Long/Short ratios (already fetched above for the Master Signal)
    gls, tls_acc, tls_pos = gls_df, tls_acc_df, tls_pos_df

    # OI deltas (already computed above for the Master Signal)
    oi_now_val = _oi_now_val
    oi_24h_pct = oi_24h_pct_now

    g1, g2, g3, g4, g5, g6 = st.columns(6)

    def _ratio_card(title, df, col):
        if df is None or df.empty or col not in df.columns:
            val, delta = None, None
        else:
            val = float(df[col].iloc[-1])
            delta = (val - float(df[col].iloc[0])) if len(df) > 1 else 0.0
        return title, val, delta

    cards = [
        ("Funding (8h)", (fr * 100) if fr is not None else None, None, "%"),
        ("OI (contracts)", oi_now_val, oi_24h_pct, ""),
        ("Global L/S", *_ratio_card("Global L/S", gls, "longShortRatio")[1:], ""),
        ("Top Acc L/S", *_ratio_card("Top Acc L/S", tls_acc, "longShortRatio")[1:], ""),
        ("Top Pos L/S", *_ratio_card("Top Pos L/S", tls_pos, "longShortRatio")[1:], ""),
        ("24h Volume (USDT)", ticker.get("quote_volume"), None, ""),
    ]
    cols = [g1, g2, g3, g4, g5, g6]
    for (title, val, delta, suffix), col in zip(cards, cols):
        with col:
            v_class = _pct_class(delta) if delta is not None else "flat"
            delta_str = f"{_arrow(delta)} {_fmt_num(delta, 2)}{'%' if title=='OI (contracts)' else ''}" if delta is not None else "—"
            st.markdown(
                f"""
                <div class="card">
                  <h4>{title}</h4>
                  <div class="val {v_class if title=='Funding (8h)' else ''}">{_fmt_num(val, 4)}{suffix}</div>
                  <div class="delta {v_class}">{delta_str}</div>
                </div>
                """, unsafe_allow_html=True
            )

    # Alert banners
    if fr is not None and abs(fr * 100) > ALERT_FUNDING_BPS:
        st.markdown(
            f"<div class='banner banner-amber'>🔔 Funding {fr*100:.4f}% exceeds "
            f"alert threshold {ALERT_FUNDING_BPS:.3f}% — crowded book.</div>",
            unsafe_allow_html=True,
        )
    if oi_24h_pct is not None and abs(oi_24h_pct) > ALERT_OI_PCT:
        st.markdown(
            f"<div class='banner banner-info'>🔔 24h OI change {oi_24h_pct:+.2f}% — "
            f"sharp positioning shift.</div>",
            unsafe_allow_html=True,
        )

    st.markdown("### Recent Liquidations")
    if liqs is not None and not liqs.empty:
        cols_show = [c for c in ["time", "side", "price", "origQty", "executedQty",
                                 "averagePrice", "status"] if c in liqs.columns]
        df_show = liqs[cols_show].copy()

        def _row_color(r):
            color = "#16c784" if str(r.get("side", "")).upper() == "BUY" else "#ea3943"
            return [f"color: {color}"] * len(r)

        st.dataframe(df_show.style.apply(_row_color, axis=1),
                     use_container_width=True, hide_index=True)
    else:
        st.info(f"Public liquidation feed isn't exposed by {PERP_VENUE_LABEL}.")

    st.markdown("### NATGASUSDT 15m chart (last 100 candles)")
    ohlcv = perp_df_15m
    if ohlcv is not None and not ohlcv.empty:
        fig = go.Figure(data=[go.Candlestick(
            x=ohlcv["time"], open=ohlcv["open"], high=ohlcv["high"],
            low=ohlcv["low"], close=ohlcv["close"],
            increasing_line_color="#16c784", decreasing_line_color="#ea3943",
        )])
        fig.update_layout(
            template="plotly_dark", height=440, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_rangeslider_visible=False,
            paper_bgcolor="#0e1320", plot_bgcolor="#0e1320",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("OHLCV unavailable.")


# ---------------------------------------------------------------------------
# TAB 2 - MOMENTUM ANALYZER (the heart of the app)
# ---------------------------------------------------------------------------
with tab_momentum:
    st.subheader("Momentum Analyzer — Perp vs Henry Hub")

    perp_df = perp_df_15m
    hh_df = hh_df_15m

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.markdown("**NATGASUSDT (15m)**")
        if perp_df is not None and not perp_df.empty:
            ma, ub, lb = bollinger(perp_df["close"])
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=perp_df["time"], open=perp_df["open"], high=perp_df["high"],
                low=perp_df["low"], close=perp_df["close"],
                increasing_line_color="#16c784", decreasing_line_color="#ea3943",
                name="Perp",
            ))
            fig.add_trace(go.Scatter(x=perp_df["time"], y=ub, line=dict(color="#8aa6ff", width=1), name="BB Up"))
            fig.add_trace(go.Scatter(x=perp_df["time"], y=ma, line=dict(color="#ffb020", width=1), name="BB Mid"))
            fig.add_trace(go.Scatter(x=perp_df["time"], y=lb, line=dict(color="#8aa6ff", width=1), name="BB Lo"))
            fig.update_layout(template="plotly_dark", height=420,
                              margin=dict(l=10, r=10, t=10, b=10),
                              xaxis_rangeslider_visible=False,
                              paper_bgcolor="#0e1320", plot_bgcolor="#0e1320")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Perp OHLCV unavailable.")

    with chart_col2:
        st.markdown("**Henry Hub NG=F (15m)**")
        if hh_df is not None and not hh_df.empty:
            ma, ub, lb = bollinger(hh_df["close"])
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=hh_df["time"], open=hh_df["open"], high=hh_df["high"],
                low=hh_df["low"], close=hh_df["close"],
                increasing_line_color="#16c784", decreasing_line_color="#ea3943",
                name="HH",
            ))
            fig.add_trace(go.Scatter(x=hh_df["time"], y=ub, line=dict(color="#8aa6ff", width=1), name="BB Up"))
            fig.add_trace(go.Scatter(x=hh_df["time"], y=ma, line=dict(color="#ffb020", width=1), name="BB Mid"))
            fig.add_trace(go.Scatter(x=hh_df["time"], y=lb, line=dict(color="#8aa6ff", width=1), name="BB Lo"))
            fig.update_layout(template="plotly_dark", height=420,
                              margin=dict(l=10, r=10, t=10, b=10),
                              xaxis_rangeslider_visible=False,
                              paper_bgcolor="#0e1320", plot_bgcolor="#0e1320")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Henry Hub intraday data unavailable.")

    # Composite Momentum Score
    perp_score = perp_score_now
    hh_score = hh_score_now

    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(
            f"<div class='card'><h4>Perp Momentum (0–100)</h4>"
            f"<div class='val'>{_fmt_num(perp_score, 1)}</div></div>",
            unsafe_allow_html=True
        )
    with s2:
        st.markdown(
            f"<div class='card'><h4>Henry Hub Momentum (0–100)</h4>"
            f"<div class='val'>{_fmt_num(hh_score, 1)}</div></div>",
            unsafe_allow_html=True
        )
    with s3:
        edge = momentum_edge_score(
            perp_score, hh_score, fr,
            oi_24h_pct if 'oi_24h_pct' in dir() else None,
            basis_pct,
        )
        eclass = "up" if (edge or 0) > 5 else ("down" if (edge or 0) < -5 else "flat")
        st.markdown(
            f"<div class='card'><h4>Momentum Edge Score (-100..+100)</h4>"
            f"<div class='val {eclass}'>{_fmt_num(edge, 1)}</div></div>",
            unsafe_allow_html=True
        )

    # Divergence banner
    if perp_score is not None and hh_score is not None:
        diff = perp_score - hh_score
        if diff > 12:
            st.markdown(
                f"<div class='banner banner-green'>📈 Perp leading Henry Hub (+{diff:.1f}) — "
                f"crypto momentum stronger.</div>", unsafe_allow_html=True)
        elif diff < -12:
            st.markdown(
                f"<div class='banner banner-red'>📉 Henry Hub leading Perp ({diff:.1f}) — "
                f"crypto lagging cash market.</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                f"<div class='banner banner-info'>🟰 Momentum aligned (Δ {diff:+.1f}).</div>",
                unsafe_allow_html=True)

    # Oscillator comparison table
    st.markdown("### Oscillator Comparison")

    def _indicator_row(df: Optional[pd.DataFrame]) -> Dict[str, Any]:
        if df is None or df.empty:
            return {k: None for k in
                    ["RSI(14)", "MACD hist", "Stoch %K", "Stoch %D",
                     "ADX(14)", "Momentum(10)", "ROC(10)", "BB %B"]}
        c = df["close"].astype(float)
        h = df["high"].astype(float) if "high" in df else c
        l = df["low"].astype(float) if "low" in df else c
        ml, ms, mh = macd(c)
        k, d = stochastic(h, l, c)
        m_ma, m_ub, m_lb = bollinger(c)
        bb_pct_b = (c.iloc[-1] - m_lb.iloc[-1]) / (m_ub.iloc[-1] - m_lb.iloc[-1]) \
            if (m_ub.iloc[-1] - m_lb.iloc[-1]) else np.nan
        return {
            "RSI(14)":      float(rsi(c).iloc[-1]),
            "MACD hist":    float(mh.iloc[-1]),
            "Stoch %K":     float(k.iloc[-1]),
            "Stoch %D":     float(d.iloc[-1]),
            "ADX(14)":      float(adx(h, l, c).iloc[-1]),
            "Momentum(10)": float(momentum(c).iloc[-1]),
            "ROC(10)":      float(roc(c).iloc[-1]),
            "BB %B":        float(bb_pct_b),
        }

    perp_ind = _indicator_row(perp_df)
    hh_ind = _indicator_row(hh_df)
    cmp_df = pd.DataFrame({"Perp": perp_ind, "Henry Hub": hh_ind}).round(3)
    cmp_df["Δ (Perp − HH)"] = (cmp_df["Perp"] - cmp_df["Henry Hub"]).round(3)
    st.dataframe(cmp_df, use_container_width=True)

    # Funding history vs price
    st.markdown("### Funding Rate History (last 50) vs Perp Price")
    fhist = fetch_funding_history(50)
    if fhist is not None and perp_df is not None and not perp_df.empty:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=fhist["fundingTime"], y=fhist["fundingRate"] * 100,
            name="Funding %", marker_color=np.where(fhist["fundingRate"] >= 0, "#16c784", "#ea3943"),
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=perp_df["time"], y=perp_df["close"],
            name="Perp Close", line=dict(color="#8aa6ff", width=1.5),
        ), secondary_y=True)
        fig.update_layout(template="plotly_dark", height=350,
                          margin=dict(l=10, r=10, t=10, b=10),
                          paper_bgcolor="#0e1320", plot_bgcolor="#0e1320")
        fig.update_yaxes(title_text="Funding %", secondary_y=False)
        fig.update_yaxes(title_text="Perp Close", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Funding history or perp price unavailable.")

    # OI momentum
    st.markdown("### Open Interest Momentum (last 24h, 5m bars)")
    if oi_hist is not None and not oi_hist.empty:
        df = oi_hist.copy()
        df["oi_delta_8h"] = df["sumOpenInterest"].diff(96)
        df["oi_delta_24h"] = df["sumOpenInterest"].diff(288)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["sumOpenInterest"],
                                 name="OI", line=dict(color="#ffb020", width=1.5)),
                      secondary_y=False)
        fig.add_trace(go.Bar(x=df["timestamp"], y=df["oi_delta_8h"],
                             name="Δ8h",
                             marker_color=np.where(df["oi_delta_8h"].fillna(0) >= 0,
                                                   "#16c784", "#ea3943"),
                             opacity=0.5),
                      secondary_y=True)
        fig.update_layout(template="plotly_dark", height=350,
                          margin=dict(l=10, r=10, t=10, b=10),
                          paper_bgcolor="#0e1320", plot_bgcolor="#0e1320")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("OI history unavailable.")

    # Technical sub-charts (RSI, MACD)
    if perp_df is not None and not perp_df.empty:
        st.markdown("### Perp Technical Indicators")
        c = perp_df["close"].astype(float)
        ml, ms, mh = macd(c)
        r = rsi(c)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.5, 0.5], vertical_spacing=0.06,
                            subplot_titles=("RSI(14)", "MACD(12,26,9)"))
        fig.add_trace(go.Scatter(x=perp_df["time"], y=r, line=dict(color="#8aa6ff"), name="RSI"),
                      row=1, col=1)
        fig.add_hline(y=70, line=dict(color="#ea3943", dash="dot"), row=1, col=1)
        fig.add_hline(y=30, line=dict(color="#16c784", dash="dot"), row=1, col=1)
        fig.add_trace(go.Scatter(x=perp_df["time"], y=ml, line=dict(color="#16c784"), name="MACD"),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=perp_df["time"], y=ms, line=dict(color="#ffb020"), name="Signal"),
                      row=2, col=1)
        fig.add_trace(go.Bar(x=perp_df["time"], y=mh, name="Hist",
                             marker_color=np.where(mh >= 0, "#16c784", "#ea3943")),
                      row=2, col=1)
        fig.update_layout(template="plotly_dark", height=480,
                          margin=dict(l=10, r=10, t=40, b=10),
                          paper_bgcolor="#0e1320", plot_bgcolor="#0e1320",
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# TAB 3 - FUNDAMENTALS
# ---------------------------------------------------------------------------
with tab_funda:
    st.subheader("Fundamentals")
    today = dt.datetime.now(timezone.utc)
    is_thu = today.weekday() == 3  # Thu = 3

    fcol1, fcol2 = st.columns([1.2, 1])
    with fcol1:
        st.markdown("### EIA Weekly Storage (Lower 48)")
        eia = eia_data
        if eia.get("ok"):
            wow = eia.get("wow_change")
            vs5 = eia.get("vs_5yr")
            tag = "🗓️ EIA Report Day (Thu 10:30 ET)" if is_thu else ""
            st.markdown(
                f"""
                <div class="card">
                  <h4>Working Gas (Bcf) — period {eia['latest_period']} {tag}</h4>
                  <div class="val">{_fmt_num(eia['latest_value'], 0)} Bcf</div>
                  <div class="delta {_pct_class(wow)}">{_arrow(wow)} {_fmt_num(wow, 0)} Bcf WoW</div>
                  <div class="sub-price">5-yr same-week avg: {_fmt_num(eia.get('five_y_avg'), 0)} Bcf
                    · vs avg: <span class="{_pct_class(vs5)}">{_fmt_num(vs5, 0)} Bcf</span>
                  </div>
                </div>
                """, unsafe_allow_html=True
            )
            df_plot = eia["df"].tail(104)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_plot["period"], y=df_plot["value"],
                                     line=dict(color="#8aa6ff", width=2), name="Storage"))
            fig.update_layout(template="plotly_dark", height=320,
                              margin=dict(l=10, r=10, t=10, b=10),
                              paper_bgcolor="#0e1320", plot_bgcolor="#0e1320")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"EIA Data Unavailable — {eia.get('error')}")

    with fcol2:
        st.markdown("### Baker Hughes Rig Count")
        st.markdown(
            """
            <div class="card">
              <h4>North America Natural Gas Rigs</h4>
              <div class="val">Manual lookup</div>
              <div class="sub-price">No free real-time API. Updated weekly Fri ~1pm ET.</div>
              <div class="sub-price">
                🔗 <a href="https://rigcount.bakerhughes.com/" target="_blank">rigcount.bakerhughes.com</a>
              </div>
            </div>
            """, unsafe_allow_html=True
        )

        st.markdown("### Weather (HDD/CDD drivers)")
        st.markdown(
            """
            <div class="card">
              <h4>NOAA / Outlook Links</h4>
              <ul style="margin:6px 0 0 14px; color:#cdd3df;">
                <li><a href="https://www.cpc.ncep.noaa.gov/products/predictions/610day/" target="_blank">NOAA 6–10 day outlook</a></li>
                <li><a href="https://www.cpc.ncep.noaa.gov/products/predictions/814day/" target="_blank">NOAA 8–14 day outlook</a></li>
                <li><a href="https://www.natgasweather.com/" target="_blank">NatGasWeather.com</a> (paid HDD/CDD model — recommended)</li>
              </ul>
            </div>
            """, unsafe_allow_html=True
        )

    st.markdown("### Seasonality Calendar")
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    season = ["High demand (winter heat)", "High demand (winter heat)",
              "Shoulder (withdrawal end)", "Shoulder (injection start)",
              "Injection season", "Cooling demand build",
              "Peak cooling demand", "Peak cooling + hurricane risk",
              "Hurricane peak / shoulder", "Injection end / pre-winter",
              "Winter demand begins", "Peak winter heat"]
    season_df = pd.DataFrame({"Month": months, "Driver": season,
                              "Current": ["⬅️" if i == today.month - 1 else "" for i in range(12)]})
    st.dataframe(season_df, use_container_width=True, hide_index=True)


# =============================================================================
# FOOTER + AUTO-REFRESH
# =============================================================================
_last_update_ts = dt.datetime.now(timezone.utc)
st.markdown(
    f"""
    <div class="footer">
      Last update: {_last_update_ts.strftime('%Y-%m-%d %H:%M:%S UTC')} ·
      Auto-refresh every {REFRESH_SEC}s ·
      <b>NOT financial advice — for analytics only.</b>
    </div>
    """, unsafe_allow_html=True,
)

# Auto-refresh: sleep then rerun. Streamlit reruns the whole script.
time.sleep(REFRESH_SEC)
st.rerun()
