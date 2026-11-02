"""
RORO RISK SCORE — standalone calculation (extracted from investing_weekly.py v2.4)
=========================================================================
12 macro/market signals -> weighted composite 0->1 -> EMA smoothing -> regime label.

Requires a FRED API key in the FRED_API_KEY environment variable.

Usage:
    python roro_risk_score.py
    from roro_risk_score import compute_risk_score
"""

import os
import socket
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
SIGNAL_WEIGHTS = {
    "vix":                      0.048,
    "credit_spread":            0.105,
    "hy_spread":                0.058,
    "skew":                     0.039,
    "nasdaq_primary_momentum":  0.129,
    "copper_momentum":          0.068,
    "dxy_momentum":             0.051,
    "yield_curve":              0.054,
    "unemployment":             0.136,
    "industrial_prod":          0.050,
    "nfci":                     0.081,
    "earnings_rev":             0.181,
}
assert abs(sum(SIGNAL_WEIGHTS.values()) - 1.0) < 0.001

ZSCORE_LOOKBACK    = 504
EMA_HALFLIFE       = 8
MOM_LOOKBACK_SHORT = 21
MOM_LOOKBACK_LONG  = 63
LEVEL_WEIGHT       = 0.70
MOMENTUM_WEIGHT    = 0.30
MACRO_STALE_DAYS   = 45

DOWNLOAD_MAX_RETRIES = 5
DOWNLOAD_RETRY_DELAY = 8
socket.setdefaulttimeout(90)


# ─────────────────────────────────────────────────────────────
# DATA HELPERS
# ─────────────────────────────────────────────────────────────
def safe_download(ticker, period="5y"):
    for attempt in range(1, DOWNLOAD_MAX_RETRIES + 1):
        try:
            df = yf.download(ticker, period=period, progress=False,
                             auto_adjust=True, threads=False)
            if df is None or df.empty:
                raise ValueError("empty DataFrame returned")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if "Close" not in df.columns:
                raise ValueError("no Close column")
            return df
        except Exception as exc:
            if attempt < DOWNLOAD_MAX_RETRIES:
                print(f"  retry {ticker} {attempt}/{DOWNLOAD_MAX_RETRIES}: {exc}")
                time.sleep(DOWNLOAD_RETRY_DELAY)
            else:
                print(f"  FAILED {ticker}: {exc}")
                return None


def load_close(ticker, label, scale=1.0):
    df = safe_download(ticker)
    if df is None:
        print(f"  x  {label:<22}: unavailable")
        return pd.Series(dtype=float)
    s = df["Close"].squeeze().dropna()
    if not isinstance(s, pd.Series) or len(s) == 0:
        return pd.Series(dtype=float)
    print(f"  {label:<22}: {float(s.iloc[-1]) / scale:.3f}  ({len(s)} days)")
    return (s / scale).sort_index()


def to_daily_index(s):
    s = s.copy()
    s.index = pd.to_datetime(s.index)
    return s.sort_index().dropna()


def splice_backbone_recent(backbone, recent, overlap_months=24):
    if len(backbone) == 0 and len(recent) == 0:
        return pd.Series(dtype=float)
    if len(recent) == 0:
        return backbone
    if len(backbone) == 0:
        return recent
    join_date = recent.index[0]
    overlap = backbone.index.intersection(recent.index)
    if len(overlap) >= 3:
        window = overlap[:overlap_months]
        offset = recent.loc[window].mean() - backbone.loc[window].mean()
    else:
        offset = 0.0
    pre = (backbone + offset)[backbone.index < join_date]
    combined = pd.concat([pre, recent]).sort_index()
    return combined[~combined.index.duplicated(keep="last")]


# ─────────────────────────────────────────────────────────────
# SCORING HELPERS
# ─────────────────────────────────────────────────────────────
def sigmoid(x, k=1.0):
    return float(1.0 / (1.0 + np.exp(-k * x)))


def series_is_usable(s, min_len=60):
    return isinstance(s, pd.Series) and len(s.dropna()) >= min_len


def zscore_score(series, lookback, invert=False):
    if not series_is_usable(series, 60):
        return 0.5, 0.0
    hist = series.dropna().iloc[-lookback:]
    mu, sigma = float(hist.mean()), float(hist.std())
    latest = float(series.dropna().iloc[-1])
    if sigma < 1e-8:
        return 0.5, 0.0
    z = (latest - mu) / sigma
    if invert:
        z = -z
    return float(np.clip(sigmoid(z, k=1.2), 0.0, 1.0)), round(z, 3)


def compute_zscore_raw(series, lookback):
    if not series_is_usable(series, 60):
        return None
    hist = series.dropna().iloc[-lookback:]
    mu, sigma = float(hist.mean()), float(hist.std())
    if sigma < 1e-8:
        return None
    return (float(series.dropna().iloc[-1]) - mu) / sigma


def momentum_score(series, short_lb=MOM_LOOKBACK_SHORT,
                   long_lb=MOM_LOOKBACK_LONG, invert=False):
    s = series.dropna()
    if len(s) < long_lb + 5:
        return 0.5, 0.0
    latest = float(s.iloc[-1])
    short_ago = float(s.iloc[-short_lb]) if len(s) > short_lb else latest
    long_ago = float(s.iloc[-long_lb]) if len(s) > long_lb else latest
    if abs(short_ago) < 1e-8 or abs(long_ago) < 1e-8:
        return 0.5, 0.0
    chg_short = (latest - short_ago) / abs(short_ago) * 100
    chg_long = (latest - long_ago) / abs(long_ago) * 100
    chg_blend = 0.6 * chg_short + 0.4 * chg_long
    if invert:
        chg_blend = -chg_blend
    return float(np.clip(sigmoid(chg_blend / 10.0), 0.0, 1.0)), round(chg_blend, 2)


def blend_lm(lv, mv):
    return float(np.clip(LEVEL_WEIGHT * lv + MOMENTUM_WEIGHT * mv, 0.0, 1.0))


def macro_staleness_weight(series):
    if not series_is_usable(series, 1):
        return 0.0, "no data"
    last_date = series.dropna().index[-1]
    if hasattr(last_date, "date"):
        last_date = last_date.date()
    age = (datetime.today().date() - last_date).days
    if age <= MACRO_STALE_DAYS:
        return 1.0, f"fresh ({age}d old)"
    if age >= 2 * MACRO_STALE_DAYS:
        return 0.0, f"STALE ({age}d old) - fully neutral"
    w = 1.0 - (age - MACRO_STALE_DAYS) / MACRO_STALE_DAYS
    return round(w, 3), f"aging ({age}d old, weight={w:.2f})"


def apply_staleness(raw_score, series):
    w, note = macro_staleness_weight(series)
    return float(np.clip(w * raw_score + (1.0 - w) * 0.5, 0.0, 1.0)), note


# ─────────────────────────────────────────────────────────────
# INDIVIDUAL SIGNALS  (each returns score, raw value, detail)
# ─────────────────────────────────────────────────────────────
def _level_momentum(s, invert_level, invert_mom, fmt="{:+.1f}%"):
    lv, lz = zscore_score(s, ZSCORE_LOOKBACK, invert=invert_level)
    mv, mc = momentum_score(s, invert=invert_mom)
    return blend_lm(lv, mv), float(s.dropna().iloc[-1]), f"z={lz:+.2f} d={fmt.format(mc)}"


def score_inverted(s):
    """VIX / credit spread / HY spread / DXY / NFCI: higher = more risk-off."""
    if not series_is_usable(s, 60):
        return 0.5, "N/A", "unavailable"
    return _level_momentum(s, True, True)


def score_skew(vix_s, skew_s):
    if series_is_usable(skew_s, 60):
        score, raw, detail = _level_momentum(skew_s, True, True)
        return score, raw, detail + " [SKEW live]"
    if series_is_usable(vix_s, 60):
        lv, lz = zscore_score(vix_s, ZSCORE_LOOKBACK, invert=True)
        return lv, float(vix_s.dropna().iloc[-1]), f"z={lz:+.2f} [VIX PROXY - ^SKEW unavailable]"
    return 0.5, "N/A", "unavailable"


def score_nasdaq_primary_momentum(s):
    if not series_is_usable(s, 201):
        return 0.5, "N/A", "insufficient data"
    sma200 = s.rolling(200).mean().iloc[-1]
    gap_pct = (float(s.iloc[-1]) - sma200) / sma200 * 100
    lv = float(np.clip(0.5 + gap_pct / 10.0, 0.0, 1.0))
    mv, mc = momentum_score(s)
    return blend_lm(lv, mv), round(gap_pct, 2), f"IXIC SMA gap={gap_pct:+.1f}% d={mc:+.1f}%"


def score_earnings_rev(nasdaq_s, nfci_s):
    nfci_z = compute_zscore_raw(nfci_s, ZSCORE_LOOKBACK) if series_is_usable(nfci_s, 60) else None
    if nfci_z is not None:
        score = float(np.clip(sigmoid(-nfci_z / 1.5), 0.0, 1.0))
        raw = float(nfci_s.dropna().iloc[-1])
        return score, round(raw, 3), f"NFCI z={nfci_z:+.2f} -> score={score:.3f}"
    if not series_is_usable(nasdaq_s, MOM_LOOKBACK_LONG + 5):
        return 0.5, "N/A", "insufficient data (no NFCI, no NASDAQ)"
    nq = nasdaq_s.dropna()
    lb6 = 126
    if len(nq) < lb6 + 5:
        mv, mc = momentum_score(nasdaq_s)
        return mv, round(mc, 2), f"6M IXIC fallback d={mc:+.1f}%"
    ret = (float(nq.iloc[-1]) - float(nq.iloc[-lb6])) / abs(float(nq.iloc[-lb6])) * 100
    return float(np.clip(sigmoid(ret / 12.0), 0.0, 1.0)), round(ret, 2), f"6M IXIC d={ret:+.1f}% (NFCI unavailable)"


def score_copper(s):
    if not series_is_usable(s, MOM_LOOKBACK_LONG + 5):
        return 0.5, "N/A", "unavailable"
    return _level_momentum(s, False, False)


def score_dxy(s):
    if not series_is_usable(s, MOM_LOOKBACK_LONG + 5):
        return 0.5, "N/A", "unavailable"
    return _level_momentum(s, True, True)


def score_yield_curve(s):
    if not series_is_usable(s, 60):
        return 0.5, "N/A", "unavailable"
    return _level_momentum(s, False, False, fmt="{:+.2f}pp")


def score_unemployment(s):
    if not series_is_usable(s, 60):
        return 0.5, "N/A", "unavailable"
    lv, lz = zscore_score(s, ZSCORE_LOOKBACK, invert=True)
    mv, mc = momentum_score(s, invert=True)
    blended, note = apply_staleness(blend_lm(lv, mv), s)
    return blended, float(s.dropna().iloc[-1]), f"z={lz:+.2f} d={mc:+.1f}% | {note}"


def score_industrial_prod(s):
    if not series_is_usable(s, 13):
        return 0.5, "N/A", "insufficient data"
    yoy = s.pct_change(12) * 100
    lv, lz = zscore_score(yoy, ZSCORE_LOOKBACK, invert=False)
    mv, mc = momentum_score(s)
    blended, note = apply_staleness(blend_lm(lv, mv), s)
    return blended, round(float(yoy.dropna().iloc[-1]), 2), f"z={lz:+.2f} d={mc:+.1f}% | {note}"


def apply_ema_smoothing(raw, history, halflife=EMA_HALFLIFE):
    history.append(raw)
    if len(history) == 1:
        return raw
    alpha = 1 - np.exp(-np.log(2) / halflife)
    ema = history[0]
    for v in history[1:]:
        ema = alpha * v + (1 - alpha) * ema
    return float(np.clip(ema, 0.0, 1.0))


def risk_label(s):
    if s >= 0.75: return "RISK-ON  (strong)"
    if s >= 0.60: return "RISK-ON  (moderate)"
    if s >= 0.50: return "NEUTRAL"
    if s >= 0.35: return "RISK-OFF (moderate)"
    return "RISK-OFF (strong)"


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────
def load_fred_data(fred):
    data = {}

    print("\n  Yields (yfinance):")
    for ticker, key, label in [("^TNX", "yield_10y", "10Y Treasury"),
                               ("^IRX", "tbill_3m", "3M T-bill")]:
        s = load_close(ticker, label, scale=10.0)
        if len(s) >= 60:
            data[key] = s

    print("\n  Credit spreads (Moody's backbone + ICE BofA):")
    try:
        baa = to_daily_index(fred.get_series("BAA10Y", observation_start="1986-01-01").dropna())
        ice = to_daily_index(fred.get_series("BAMLC0A0CM").dropna())
        spliced = splice_backbone_recent(baa, ice)
        if len(spliced) < 60:
            raise ValueError("too short")
        data["credit_spread"] = spliced
    except Exception as e:
        print(f"  !! BBB OAS splice failed ({e}) - fallback BAMLC0A0CM")
        try:
            fb = to_daily_index(fred.get_series("BAMLC0A0CM").dropna())
            if len(fb) >= 60:
                data["credit_spread"] = fb
        except Exception:
            print("  !! credit_spread unavailable - neutral 0.5")

    try:
        dbaa = fred.get_series("DBAA", observation_start="1986-01-01")
        gs10 = fred.get_series("GS10", observation_start="1986-01-01")
        dbaa.index = pd.to_datetime(dbaa.index)
        gs10.index = pd.to_datetime(gs10.index)
        back = to_daily_index((dbaa - gs10).dropna())
        ice = to_daily_index(fred.get_series("BAMLH0A0HYM2").dropna())
        spliced = splice_backbone_recent(back, ice)
        if len(spliced) < 60:
            raise ValueError("too short")
        data["hy_spread"] = spliced
    except Exception as e:
        print(f"  !! HY OAS splice failed ({e}) - fallback BAMLH0A0HYM2")
        try:
            fb = to_daily_index(fred.get_series("BAMLH0A0HYM2").dropna())
            if len(fb) >= 60:
                data["hy_spread"] = fb
        except Exception:
            print("  !! hy_spread unavailable - neutral 0.5")

    print("\n  Monthly macro (FRED):")
    for code, key, label in [("UNRATE", "unemployment", "Unemployment rate"),
                             ("INDPRO", "indpro", "Industrial production"),
                             ("NFCI", "nfci", "NFCI")]:
        try:
            s = fred.get_series(code)
            s.index = pd.to_datetime(s.index)
            s = s.dropna()
            if len(s) == 0:
                raise ValueError("empty")
            data[key] = s
            age = (datetime.today().date() - s.index[-1].date()).days
            print(f"  ok  {label:<24} latest={float(s.iloc[-1]):.3f} ({age}d old)")
        except Exception as e:
            print(f"  !!  {label:<24} FAILED ({e}) - neutral 0.5")

    return data


# ─────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────
def compute_risk_score(fred_api_key=None, verbose=True):
    """Fetch live data and return (risk_score, raw_score, signals, signal_log)."""
    key = fred_api_key or os.environ.get("FRED_API_KEY")
    if not key:
        raise RuntimeError("Set the FRED_API_KEY environment variable (or pass fred_api_key).")

    print("\n── Loading market data ──────────────────────────────────────────────────")
    nasdaq = load_close("^IXIC", "NASDAQ (primary)")
    vix = load_close("^VIX", "VIX")
    copper = load_close("HG=F", "Copper (HG=F)")
    dxy = load_close("DX-Y.NYB", "DXY")
    skew = load_close("^SKEW", "SKEW Index")

    print("\n── Loading FRED macro signals ───────────────────────────────────────────")
    fred_data = load_fred_data(Fred(api_key=key))
    empty = pd.Series(dtype=float)

    yc = None
    if "yield_10y" in fred_data and "tbill_3m" in fred_data:
        raw_yc = (fred_data["yield_10y"] - fred_data["tbill_3m"]).dropna()
        if len(raw_yc) >= 60:
            yc = raw_yc

    scored = {
        "vix":                     score_inverted(vix),
        "credit_spread":           score_inverted(fred_data.get("credit_spread", empty)),
        "hy_spread":               score_inverted(fred_data.get("hy_spread", empty)),
        "skew":                    score_skew(vix, skew),
        "nasdaq_primary_momentum": score_nasdaq_primary_momentum(nasdaq),
        "copper_momentum":         score_copper(copper),
        "dxy_momentum":            score_dxy(dxy),
        "yield_curve":             score_yield_curve(yc if yc is not None else empty),
        "unemployment":            score_unemployment(fred_data.get("unemployment", empty)),
        "industrial_prod":         score_industrial_prod(fred_data.get("indpro", empty)),
        "nfci":                    score_inverted(fred_data.get("nfci", empty)),
        "earnings_rev":            score_earnings_rev(nasdaq, fred_data.get("nfci", empty)),
    }
    signals = {k: float(v[0]) for k, v in scored.items()}

    raw_score = sum(signals[k] * SIGNAL_WEIGHTS[k] for k in SIGNAL_WEIGHTS)
    risk_score = apply_ema_smoothing(raw_score, [], EMA_HALFLIFE)

    if verbose:
        print("\n── Risk signal scores ───────────────────────────────────────────────────")
        print(f"  {'Signal':<24} {'Raw Value':<14} {'Score':>6} {'Weight':>7} {'Contrib':>8}  Detail")
        for k in sorted(SIGNAL_WEIGHTS, key=SIGNAL_WEIGHTS.get, reverse=True):
            score, raw, detail = scored[k]
            raw_str = raw if isinstance(raw, str) else f"{float(raw):.2f}"
            w = SIGNAL_WEIGHTS[k]
            print(f"  {k:<24} {raw_str:<14} {score:>6.3f} {w:>7.3f} {score * w:>8.4f}  {detail}")
        print(f"\n  Raw composite score : {raw_score:.3f}")
        print(f"  EMA-smoothed score  : {risk_score:.3f}")
        print(f"  Regime              : {risk_label(risk_score)}")

    return risk_score, raw_score, signals, scored


if __name__ == "__main__":
    compute_risk_score()
