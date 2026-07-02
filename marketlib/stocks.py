"""Per-stock TWSE data fetchers and technical indicators.

Pure parse functions (_parse_*) accept inline fixture payloads for testing.
Network fetch_* wrappers call http_get_json then delegate to parsers,
returning None / [] on any failure — never raise to caller.
"""
from __future__ import annotations

from datetime import date
from statistics import mean
from typing import Any, Optional

from marketlib.core import MarketBar, clean_num, pct_change  # noqa: F401
from marketlib.features import volume_ratio_excluding_t
from marketlib.fetchers import (
    TWSE_BASE,
    http_get,  # noqa: F401
    http_get_json,
    iso_to_yyyymmdd,
    month_starts,
    roc_date_to_iso,
    twse_asof,  # noqa: F401
)

MONTHLY_REVENUE_URL = "https://openapi.twse.com.tw/v1/opendata/t187ap05_L"


# ---------------------------------------------------------------------------
# Pure parse helpers (tested directly without network)
# ---------------------------------------------------------------------------


def _parse_stock_day(payload: dict[str, Any]) -> list[MarketBar]:
    """Parse STOCK_DAY JSON payload into a list of MarketBar."""
    if payload.get("stat") != "OK":
        return []
    fields = payload.get("fields", [])
    data = payload.get("data", [])
    try:
        idx_date = _find_field_index(fields, "日期")
        idx_vol = _find_field_index(fields, "成交股數")
        idx_turnover = _find_field_index(fields, "成交金額")
        idx_open = _find_field_index(fields, "開盤價")
        idx_high = _find_field_index(fields, "最高價")
        idx_low = _find_field_index(fields, "最低價")
        idx_close = _find_field_index(fields, "收盤價")
        idx_change = _find_field_index(fields, "漲跌價差")
        idx_trades = _find_field_index(fields, "成交筆數")
    except (ValueError, IndexError):
        return []

    bars: list[MarketBar] = []
    for row in data:
        try:
            iso = roc_date_to_iso(row[idx_date])
            close = clean_num(row[idx_close])
            if close is None:
                continue
            bars.append(
                MarketBar(
                    date=iso,
                    close=close,
                    volume=clean_num(row[idx_vol]),
                    turnover=clean_num(row[idx_turnover]),
                    open=clean_num(row[idx_open]),
                    high=clean_num(row[idx_high]),
                    low=clean_num(row[idx_low]),
                    change_points=clean_num(row[idx_change]),
                    trades=clean_num(row[idx_trades]),
                )
            )
        except (IndexError, ValueError):
            continue
    return bars


def _parse_t86(payload: dict[str, Any], stock_no: str) -> Optional[dict[str, Any]]:
    """Parse T86 institutional net buy/sell payload for a given stock_no.

    Maps columns by header substring — layout can change over time.
    Converts shares (股數) to lots (張) by dividing by 1000.
    """
    if payload.get("stat") != "OK":
        return None
    fields = payload.get("fields", [])
    data = payload.get("data", [])

    try:
        code_idx = _find_field_index(fields, "證券代號")
        total_idx = _find_field_index_substring(fields, "三大法人買賣超股數")
        trust_idx = _find_field_index_substring(fields, "投信買賣超股數")
        # Foreign: prefer full name, fallback to shorter name
        try:
            foreign_idx = _find_field_index_substring(fields, "外陸資買賣超股數(不含外資自營商)")
        except ValueError:
            foreign_idx = _find_field_index_substring(fields, "外資買賣超股數")
        # Dealer: sum all columns containing "自營商買賣超股數"
        dealer_indices = _find_all_field_indices_substring(fields, "自營商買賣超股數")
    except ValueError:
        return None

    for row in data:
        try:
            if row[code_idx].strip() != stock_no:
                continue
            foreign_shares = clean_num(row[foreign_idx]) or 0.0
            trust_shares = clean_num(row[trust_idx]) or 0.0
            total_shares = clean_num(row[total_idx]) or 0.0
            dealer_shares = sum(
                clean_num(row[i]) or 0.0 for i in dealer_indices
            )
            return {
                "foreign_net_lots": foreign_shares / 1000,
                "trust_net_lots": trust_shares / 1000,
                "dealer_net_lots": dealer_shares / 1000,
                "inst_total_net_lots": total_shares / 1000,
            }
        except (IndexError, ValueError):
            continue
    return None


def _parse_bwibbu(payload: dict[str, Any], stock_no: str) -> Optional[dict[str, Any]]:
    """Parse BWIBBU_d valuation payload for a given stock_no.

    Maps columns by header substring.
    """
    if payload.get("stat") != "OK":
        return None
    fields = payload.get("fields", [])
    data = payload.get("data", [])

    try:
        code_idx = _find_field_index(fields, "證券代號")
        pe_idx = _find_field_index_substring(fields, "本益比")
        pb_idx = _find_field_index_substring(fields, "股價淨值比")
        div_idx = _find_field_index_substring(fields, "殖利率(%)")
    except ValueError:
        return None

    for row in data:
        try:
            if row[code_idx].strip() != stock_no:
                continue
            return {
                "pe": clean_num(row[pe_idx]),
                "pb": clean_num(row[pb_idx]),
                "dividend_yield": clean_num(row[div_idx]),
            }
        except (IndexError, ValueError):
            continue
    return None


def _parse_monthly_revenue(items: list[dict[str, Any]], stock_no: str) -> Optional[dict[str, Any]]:
    """Parse monthly revenue OpenAPI list for a given stock_no.

    ROC 資料年月 format "11506" → year=115+1911=2026, month=06 → "2026-06".
    """
    for item in items:
        if item.get("公司代號") != stock_no:
            continue
        try:
            roc_ym = str(item["資料年月"])
            year = int(roc_ym[:-2]) + 1911
            month = int(roc_ym[-2:])
            month_str = f"{year:04d}-{month:02d}"
            return {
                "month": month_str,
                "revenue_thousand": clean_num(item.get("營業收入-當月營收")),
                "yoy_pct": clean_num(item.get("去年同月增減(%)")),
                "mom_pct": clean_num(item.get("上月比較增減(%)")),
                "cum_yoy_pct": clean_num(item.get("前期比較增減(%)")),
            }
        except (KeyError, ValueError, TypeError):
            continue
    return None


# ---------------------------------------------------------------------------
# Field index helpers
# ---------------------------------------------------------------------------


def _find_field_index(fields: list[str], exact: str) -> int:
    """Return index of exact field name match."""
    return fields.index(exact)


def _find_field_index_substring(fields: list[str], substring: str) -> int:
    """Return index of first field containing substring."""
    for i, f in enumerate(fields):
        if substring in f:
            return i
    raise ValueError(f"No field containing {substring!r} found in {fields}")


def _find_all_field_indices_substring(fields: list[str], substring: str) -> list[int]:
    """Return all indices of fields containing substring."""
    indices = [i for i, f in enumerate(fields) if substring in f]
    if not indices:
        raise ValueError(f"No field containing {substring!r} found in {fields}")
    return indices


# ---------------------------------------------------------------------------
# Network fetch wrappers
# ---------------------------------------------------------------------------


def fetch_stock_day(stock_no: str, month_start: date) -> list[MarketBar]:
    """Fetch one month of daily OHLCV bars for stock_no.

    Returns [] on any failure.
    """
    try:
        url = (
            f"{TWSE_BASE}/afterTrading/STOCK_DAY?"
            f"date={iso_to_yyyymmdd(month_start)}&stockNo={stock_no}&response=json"
        )
        payload = http_get_json(url)
        return _parse_stock_day(payload)
    except Exception:  # noqa: BLE001
        return []


def fetch_stock_history(
    stock_no: str, end: date, lookback_months: int
) -> list[MarketBar]:
    """Fetch multiple months of daily bars, deduplicated and sorted by date."""
    bars: list[MarketBar] = []
    for month_start in month_starts(end, lookback_months):
        bars.extend(fetch_stock_day(stock_no, month_start))
    dedup = {bar.date: bar for bar in bars if bar.close}
    return [dedup[key] for key in sorted(dedup)]


def fetch_t86_for_stock(stock_no: str, date_iso: str) -> Optional[dict[str, Any]]:
    """Fetch T86 institutional net buy/sell for stock_no on date_iso.

    Returns None on any failure.
    """
    try:
        url = (
            f"{TWSE_BASE}/fund/T86?"
            f"date={iso_to_yyyymmdd(date_iso)}&selectType=ALL&response=json"
        )
        payload = http_get_json(url)
        return _parse_t86(payload, stock_no)
    except Exception:  # noqa: BLE001
        return None


def fetch_valuation_for_stock(stock_no: str, date_iso: str) -> Optional[dict[str, Any]]:
    """Fetch BWIBBU_d valuation (PE/PB/dividend yield) for stock_no on date_iso.

    Returns None on any failure.
    """
    try:
        url = (
            f"{TWSE_BASE}/afterTrading/BWIBBU_d?"
            f"date={iso_to_yyyymmdd(date_iso)}&selectType=ALL&response=json"
        )
        payload = http_get_json(url)
        return _parse_bwibbu(payload, stock_no)
    except Exception:  # noqa: BLE001
        return None


def fetch_monthly_revenue(stock_no: str) -> Optional[dict[str, Any]]:
    """Fetch latest monthly revenue for stock_no from TWSE OpenAPI.

    Returns None on any failure.
    """
    try:
        items = http_get_json(MONTHLY_REVENUE_URL)
        if not isinstance(items, list):
            return None
        return _parse_monthly_revenue(items, stock_no)
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Technical indicator helpers
# ---------------------------------------------------------------------------


def ema(values: list[float], span: int) -> list[float]:
    """Compute EMA of values with given span.

    Seed = first value; alpha = 2/(span+1); iterate forward.
    Returns list of same length as values.
    """
    if not values:
        return []
    alpha = 2.0 / (span + 1)
    out = [values[0]]
    for v in values[1:]:
        out.append(alpha * v + (1 - alpha) * out[-1])
    return out


def macd(
    closes: list[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Optional[dict]:
    """Compute MACD for closes list.

    Returns dict with keys: dif, dea, hist, cross.
    cross: "golden" if last (dif-dea) flipped negative->positive,
           "dead" if positive->negative, else None.
    Returns None when len(closes) < slow.
    """
    if len(closes) < slow:
        return None

    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)

    # Align on the tail where both series exist — both have same length as closes
    dif_series = [f - s for f, s in zip(ema_fast, ema_slow)]

    dea_series = ema(dif_series, signal)

    last_dif = dif_series[-1]
    last_dea = dea_series[-1]
    hist = last_dif - last_dea

    # Cross: compare sign of (dif - dea) at last two points
    cross = None
    if len(dif_series) >= 2:
        prev_diff = dif_series[-2] - dea_series[-2]
        curr_diff = dif_series[-1] - dea_series[-1]
        prev_sign = prev_diff >= 0
        curr_sign = curr_diff >= 0
        if not prev_sign and curr_sign:
            cross = "golden"
        elif prev_sign and not curr_sign:
            cross = "dead"

    return {"dif": last_dif, "dea": last_dea, "hist": hist, "cross": cross}


def rsi(closes: list[float], period: int = 14) -> Optional[float]:
    """Compute RSI using Wilder's smoothing.

    Returns None when len(closes) < period + 1.
    """
    if len(closes) < period + 1:
        return None

    changes = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(c, 0.0) for c in changes]
    losses = [max(-c, 0.0) for c in changes]

    # Seed: simple average over first period
    avg_gain = mean(gains[:period])
    avg_loss = mean(losses[:period])

    # Wilder smoothing over remaining values
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def kd(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    n: int = 9,
) -> Optional[dict]:
    """Compute KD stochastic oscillator.

    RSV = (close - minLow_n) / (maxHigh_n - minLow_n) * 100.
    K and D via 3-period EMA-style smoothing seeded at 50.
    Returns None when insufficient data (len < n).
    """
    if len(closes) < n or len(highs) < n or len(lows) < n:
        return None

    # Build RSV series
    rsv_series: list[float] = []
    for i in range(n - 1, len(closes)):
        hi = max(highs[i - n + 1: i + 1])
        lo = min(lows[i - n + 1: i + 1])
        if hi == lo:
            rsv_series.append(50.0)
        else:
            rsv_series.append((closes[i] - lo) / (hi - lo) * 100.0)

    # K/D via EMA-3 smoothing seeded at 50
    k = 50.0
    d = 50.0
    alpha = 1.0 / 3.0
    for rsv in rsv_series:
        k = alpha * rsv + (1 - alpha) * k
        d = alpha * k + (1 - alpha) * d

    return {"k": k, "d": d}


def resample_weekly(bars: list[MarketBar]) -> list[MarketBar]:
    """Group daily MarketBar list into weekly bars using ISO calendar (year, week).

    Per week: close = last daily close, high = max, low = min,
    volume = sum, date = last date in that week.
    Returns ordered list of weekly MarketBar.
    """
    from datetime import date as date_cls
    from collections import OrderedDict

    groups: dict[tuple, list[MarketBar]] = OrderedDict()
    for bar in bars:
        d = date_cls.fromisoformat(bar.date)
        iso_yw = d.isocalendar()[:2]  # (year, week)
        if iso_yw not in groups:
            groups[iso_yw] = []
        groups[iso_yw].append(bar)

    weekly: list[MarketBar] = []
    for week_bars in groups.values():
        last = week_bars[-1]
        highs = [b.high for b in week_bars if b.high is not None]
        lows = [b.low for b in week_bars if b.low is not None]
        vols = [b.volume for b in week_bars if b.volume is not None]
        weekly.append(
            MarketBar(
                date=last.date,
                close=last.close,
                high=max(highs) if highs else None,
                low=min(lows) if lows else None,
                volume=sum(vols) if vols else None,
            )
        )
    return weekly


def technical_metrics(daily_bars: list[MarketBar]) -> dict:
    """Assemble technical metrics for a stock from its daily bar history.

    Returns a flat dict with keys:
    close, ma5, ma20, ma60, ret5, ret20, ret60, drawdown,
    rsi, kd_k, kd_d,
    macd_daily (dict or None), macd_weekly (dict or None),
    volume_ratio_20d.
    Values are None where insufficient data.
    """
    closes = [b.close for b in daily_bars]
    highs = [b.high for b in daily_bars]
    lows = [b.low for b in daily_bars]
    volumes = [b.volume for b in daily_bars]

    n = len(closes)

    ma5 = mean(closes[-5:]) if n >= 5 else None
    ma20 = mean(closes[-20:]) if n >= 20 else None
    ma60 = mean(closes[-60:]) if n >= 60 else None

    ret5 = (closes[-1] / closes[-6] - 1) if n >= 6 and closes[-6] else None
    ret20 = (closes[-1] / closes[-21] - 1) if n >= 21 and closes[-21] else None
    ret60 = (closes[-1] / closes[-61] - 1) if n >= 61 and closes[-61] else None

    hi20 = max(closes[-20:]) if n >= 20 else None
    drawdown = (closes[-1] / hi20 - 1) if hi20 else None

    rsi_val = rsi(closes)

    # KD: require high/low data
    valid_highs = [h for h in highs if h is not None]
    valid_lows = [l for l in lows if l is not None]
    kd_result = None
    if len(valid_highs) >= 9 and len(valid_lows) >= 9 and n >= 9:
        kd_result = kd(
            [h if h is not None else closes[i] for i, h in enumerate(highs)],
            [l if l is not None else closes[i] for i, l in enumerate(lows)],
            closes,
        )

    macd_daily = macd(closes)

    weekly_bars = resample_weekly(daily_bars)
    weekly_closes = [b.close for b in weekly_bars]
    macd_weekly = macd(weekly_closes)

    vol_ratio = volume_ratio_excluding_t(volumes, 20)

    return {
        "close": closes[-1] if closes else None,
        "ma5": ma5,
        "ma20": ma20,
        "ma60": ma60,
        "ret5": ret5,
        "ret20": ret20,
        "ret60": ret60,
        "drawdown": drawdown,
        "rsi": rsi_val,
        "kd_k": kd_result["k"] if kd_result else None,
        "kd_d": kd_result["d"] if kd_result else None,
        "macd_daily": macd_daily,
        "macd_weekly": macd_weekly,
        "volume_ratio_20d": vol_ratio,
    }


# ---------------------------------------------------------------------------
# Fundamental metrics helpers
# ---------------------------------------------------------------------------


def percentile_rank(history: list[float], value: float) -> Optional[float]:
    """Compute percentile rank of value within history.

    Returns fraction of history <= value (0..1); None if history empty.
    """
    if not history:
        return None
    return sum(1 for h in history if h <= value) / len(history)


def valuation_metrics(
    current: Optional[dict[str, Any]],
    pe_history: list[float],
    pb_history: list[float],
) -> dict[str, Any]:
    """Compute valuation metrics with percentile ranks.

    Args:
        current: dict with keys pe, pb, dividend_yield (or None).
        pe_history: list of PE ratios for percentile calculation.
        pb_history: list of PB ratios for percentile calculation.

    Returns:
        dict with keys: pe, pb, dividend_yield, pe_pct, pb_pct.
        If current is None, all values are None.
        pe_pct/pb_pct are percentile ranks (lower = cheaper).
    """
    if current is None:
        return {
            "pe": None,
            "pb": None,
            "dividend_yield": None,
            "pe_pct": None,
            "pb_pct": None,
        }

    pe = current.get("pe")
    pb = current.get("pb")
    dividend_yield = current.get("dividend_yield")

    pe_pct = percentile_rank(pe_history, pe) if pe is not None else None
    pb_pct = percentile_rank(pb_history, pb) if pb is not None else None

    return {
        "pe": pe,
        "pb": pb,
        "dividend_yield": dividend_yield,
        "pe_pct": pe_pct,
        "pb_pct": pb_pct,
    }


def revenue_metrics(rev: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Pass through monthly revenue metrics, marking missing data.

    Args:
        rev: dict with keys yoy_pct, mom_pct, cum_yoy_pct (or None).

    Returns:
        dict with keys: yoy_pct, mom_pct, cum_yoy_pct.
        When rev is None, all values are None.
    """
    if rev is None:
        return {
            "yoy_pct": None,
            "mom_pct": None,
            "cum_yoy_pct": None,
        }

    return {
        "yoy_pct": rev.get("yoy_pct"),
        "mom_pct": rev.get("mom_pct"),
        "cum_yoy_pct": rev.get("cum_yoy_pct"),
    }


# ---------------------------------------------------------------------------
# Scorecard — per-facet scoring, lean combination, 2330 scorecard builder
# ---------------------------------------------------------------------------

# ------------------------------------------------------------------
# Module-level constants (tunable)
# ------------------------------------------------------------------

# Technical facet thresholds
TECH_RSI_OVERBOUGHT = 80.0   # RSI above this → deduct
TECH_RSI_OVERSOLD = 20.0     # RSI below this → add (reversal)

# Revenue facet thresholds (YoY %)
REV_BULL_THRESHOLD = 20.0    # ≥ +20 % → strong positive signal
REV_BEAR_THRESHOLD = -10.0   # ≤ −10 % → negative signal

# Lean bias thresholds
LEAN_BULL_THRESHOLD = 0.25   # score ≥ +0.25 → 偏多
LEAN_BEAR_THRESHOLD = -0.25  # score ≤ −0.25 → 偏空
LEAN_HIGH_CONF = 0.5         # |score| > 0.5 → 高
LEAN_MED_CONF = 0.25         # |score| > 0.25 → 中

# Weights for short-term lean (日線+量能+籌碼為主)
SHORT_WEIGHTS: dict[str, float] = {
    "technical": 0.35,
    "volume": 0.20,
    "chip": 0.25,
    "valuation": 0.05,
    "revenue": 0.05,
    "peer": 0.05,
    "market": 0.05,
}

# Weights for medium-term lean (週線+基本面+同業/大盤)
MEDIUM_WEIGHTS: dict[str, float] = {
    "technical": 0.20,
    "volume": 0.10,
    "chip": 0.10,
    "valuation": 0.20,
    "revenue": 0.20,
    "peer": 0.10,
    "market": 0.10,
}

_MISSING_FACET = {"score": None, "reason": "資料不足", "metrics": {}}


def _clamp(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    """Clamp v to [lo, hi]."""
    return max(lo, min(hi, v))


# ------------------------------------------------------------------
# Facet scoring functions
# ------------------------------------------------------------------


def technical_facet(metrics: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Score technical posture from technical_metrics() output.

    Score direction (v1):
    - +: 站上 ma20, 多頭排列 (ma5>ma20>ma60), 日 MACD 黃金交叉, 週線多頭
    - −: 跌破 ma20, 死亡交叉
    - RSI>80 減分; RSI<20 加分(反轉)
    Returns {"metrics":dict, "score":Optional[float], "reason":str}.
    """
    if not metrics:
        return dict(_MISSING_FACET)

    close = metrics.get("close")
    ma5 = metrics.get("ma5")
    ma20 = metrics.get("ma20")
    ma60 = metrics.get("ma60")
    rsi_val = metrics.get("rsi")
    macd_daily = metrics.get("macd_daily")
    macd_weekly = metrics.get("macd_weekly")

    if close is None or ma20 is None:
        return dict(_MISSING_FACET)

    score = 0.0
    reasons: list[str] = []

    # MA20 position (+0.3 / −0.3)
    if close > ma20:
        score += 0.3
        reasons.append("站上ma20")
    else:
        score -= 0.3
        reasons.append("跌破ma20")

    # 多頭排列 ma5>ma20>ma60 (+0.2)
    if ma5 is not None and ma60 is not None:
        if ma5 > ma20 > ma60:
            score += 0.2
            reasons.append("多頭排列")
        elif ma5 < ma20 < ma60:
            score -= 0.2
            reasons.append("空頭排列")

    # Daily MACD cross (+0.25 / −0.25)
    if macd_daily is not None:
        cross = macd_daily.get("cross")
        if cross == "golden":
            score += 0.25
            reasons.append("日MACD黃金交叉")
        elif cross == "dead":
            score -= 0.25
            reasons.append("日MACD死亡交叉")
        # No cross — check histogram sign for mild signal
        elif macd_daily.get("hist") is not None:
            if macd_daily["hist"] > 0:
                score += 0.1
            else:
                score -= 0.1

    # Weekly MACD (+0.15 / −0.15)
    if macd_weekly is not None:
        cross_w = macd_weekly.get("cross")
        if cross_w == "golden":
            score += 0.15
            reasons.append("週MACD黃金交叉")
        elif cross_w == "dead":
            score -= 0.15
            reasons.append("週MACD死亡交叉")
        elif macd_weekly.get("hist") is not None:
            if macd_weekly["hist"] > 0:
                score += 0.05
            else:
                score -= 0.05

    # RSI extremes
    if rsi_val is not None:
        if rsi_val > TECH_RSI_OVERBOUGHT:
            score -= 0.15
            reasons.append(f"RSI超買({rsi_val:.0f})")
        elif rsi_val < TECH_RSI_OVERSOLD:
            score += 0.15
            reasons.append(f"RSI超賣({rsi_val:.0f})")

    score = _clamp(score)
    return {
        "metrics": metrics,
        "score": score,
        "reason": "，".join(reasons) if reasons else "無明顯信號",
    }


def volume_facet(metrics: Optional[dict[str, Any]], ret5: Optional[float] = None) -> dict[str, Any]:
    """Score volume behaviour.

    放量漲+ / 放量跌− / 量縮追價不足 slight −.
    metrics: must contain volume_ratio_20d; ret5 is 5-day return.
    """
    if not metrics:
        return dict(_MISSING_FACET)

    vol_ratio = metrics.get("volume_ratio_20d")
    if vol_ratio is None:
        return dict(_MISSING_FACET)

    ret = ret5 if ret5 is not None else metrics.get("ret5")

    score = 0.0
    reasons: list[str] = []

    high_vol = vol_ratio > 1.5   # 放量
    low_vol = vol_ratio < 0.7    # 縮量

    if ret is not None:
        if high_vol and ret > 0:
            score += 0.6
            reasons.append(f"放量上漲(量比{vol_ratio:.1f}x)")
        elif high_vol and ret < 0:
            score -= 0.6
            reasons.append(f"放量下跌(量比{vol_ratio:.1f}x)")
        elif low_vol and ret > 0:
            score -= 0.2
            reasons.append("縮量追價不足")
        else:
            score = 0.0
    else:
        # No return info — just use volume ratio directionally
        if high_vol:
            score = 0.0  # unknown direction
        elif low_vol:
            score = -0.1
            reasons.append("縮量")

    score = _clamp(score)
    return {
        "metrics": {"volume_ratio_20d": vol_ratio, "ret5": ret},
        "score": score,
        "reason": "，".join(reasons) if reasons else "量能中性",
    }


def valuation_facet(metrics: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Score valuation cheapness.

    pe_pct: percentile rank of current PE vs history (0=cheapest, 1=most expensive).
    Score = (0.5 - pe_pct) * 2, clipped to [-1, 1].
    Cheaper (low pe_pct) → more bullish (positive score).
    """
    if not metrics:
        return dict(_MISSING_FACET)

    pe_pct = metrics.get("pe_pct")
    pb_pct = metrics.get("pb_pct")

    if pe_pct is None and pb_pct is None:
        return dict(_MISSING_FACET)

    scores = []
    reasons: list[str] = []

    if pe_pct is not None:
        pe_score = (0.5 - pe_pct) * 2
        scores.append(pe_score)
        if pe_pct < 0.3:
            reasons.append(f"估值便宜(PE百分位{pe_pct:.0%})")
        elif pe_pct > 0.7:
            reasons.append(f"估值偏貴(PE百分位{pe_pct:.0%})")

    if pb_pct is not None:
        pb_score = (0.5 - pb_pct) * 2
        scores.append(pb_score)

    score = _clamp(mean(scores))
    return {
        "metrics": metrics,
        "score": score,
        "reason": "，".join(reasons) if reasons else "估值中性",
    }


def revenue_facet(metrics: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Score revenue momentum.

    yoy_pct >= REV_BULL_THRESHOLD (+20%) → +1.0 (linear scaling).
    yoy_pct <= REV_BEAR_THRESHOLD (-10%) → -1.0.
    Linear interpolation between those two anchors.
    """
    if not metrics:
        return dict(_MISSING_FACET)

    yoy = metrics.get("yoy_pct")
    if yoy is None:
        return dict(_MISSING_FACET)

    # Linear map: [-10, +20] → [-1, +1]
    span = REV_BULL_THRESHOLD - REV_BEAR_THRESHOLD  # 30
    score = _clamp((yoy - REV_BEAR_THRESHOLD) / span * 2 - 1)

    if yoy >= REV_BULL_THRESHOLD:
        reason = f"營收YoY強勁+{yoy:.1f}%"
    elif yoy <= REV_BEAR_THRESHOLD:
        reason = f"營收YoY衰退{yoy:.1f}%"
    else:
        reason = f"營收YoY {yoy:.1f}%"

    return {
        "metrics": metrics,
        "score": score,
        "reason": reason,
    }


def chip_facet(t86_recent: Optional[list[dict[str, Any]]]) -> dict[str, Any]:
    """Score institutional chip posture.

    t86_recent: list of up to 5 daily T86 dicts (each has inst_total_net_lots).
    法人近5日淨買超(張)為正且占比高 → +; 連賣 → −.
    """
    if not t86_recent:
        return dict(_MISSING_FACET)

    lots = []
    for day in t86_recent:
        if day and day.get("inst_total_net_lots") is not None:
            lots.append(day["inst_total_net_lots"])

    if not lots:
        return dict(_MISSING_FACET)

    total = sum(lots)
    days_positive = sum(1 for l in lots if l > 0)
    days_n = len(lots)

    # Score: net lots sign × fraction of positive days, normalised
    # Rough scale: 5000 lots over 5 days = max signal
    norm = min(abs(total) / 5000.0, 1.0)
    direction = 1.0 if total > 0 else (-1.0 if total < 0 else 0.0)
    pos_ratio = days_positive / days_n
    # Blend magnitude and consistency
    score = _clamp(direction * (0.5 * norm + 0.5 * abs(2 * pos_ratio - 1)) * direction)

    if total > 0:
        reason = f"法人近{days_n}日淨買超{total:.0f}張"
    elif total < 0:
        reason = f"法人近{days_n}日淨賣超{abs(total):.0f}張"
    else:
        reason = "法人中性"

    return {
        "metrics": {"inst_net_lots_total": total, "days": days_n, "days_positive": days_positive},
        "score": score,
        "reason": reason,
    }


def peer_facet(peer: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Score peer/sector environment.

    peer: dict with keys smh_ret20 (SMH 20-day return), adr_ret (ADR daily return).
    SMH 20日多頭(ret>0) + ADR 收紅(ret>0) → +.
    """
    if not peer:
        return dict(_MISSING_FACET)

    smh_ret20 = peer.get("smh_ret20")
    adr_ret = peer.get("adr_ret")

    if smh_ret20 is None and adr_ret is None:
        return dict(_MISSING_FACET)

    scores = []
    reasons: list[str] = []

    if smh_ret20 is not None:
        # SMH 20d return: normalise ±20% = ±1
        s = _clamp(smh_ret20 / 0.20)
        scores.append(s)
        if smh_ret20 > 0:
            reasons.append(f"SMH多頭({smh_ret20:.1%})")
        else:
            reasons.append(f"SMH空頭({smh_ret20:.1%})")

    if adr_ret is not None:
        s = _clamp(adr_ret / 0.05)  # ±5% = ±1
        scores.append(s)
        if adr_ret > 0:
            reasons.append(f"ADR收紅({adr_ret:.1%})")
        else:
            reasons.append(f"ADR收黑({adr_ret:.1%})")

    score = _clamp(mean(scores))
    return {
        "metrics": peer,
        "score": score,
        "reason": "，".join(reasons) if reasons else "同業中性",
    }


def market_facet(market: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Score overall market environment.

    market: dict with keys taiex_ret20 (20-day return), vix (current VIX level).
    加權多頭(ret>0) + VIX 低(<20) → +.
    """
    if not market:
        return dict(_MISSING_FACET)

    taiex_ret20 = market.get("taiex_ret20")
    vix = market.get("vix")

    if taiex_ret20 is None and vix is None:
        return dict(_MISSING_FACET)

    scores = []
    reasons: list[str] = []

    if taiex_ret20 is not None:
        s = _clamp(taiex_ret20 / 0.15)  # ±15% = ±1
        scores.append(s)
        if taiex_ret20 > 0:
            reasons.append(f"加權多頭({taiex_ret20:.1%})")
        else:
            reasons.append(f"加權空頭({taiex_ret20:.1%})")

    if vix is not None:
        # VIX: low (<15) = +0.5, medium (15-25) = 0, high (>30) = -0.5
        if vix < 15:
            vix_score = 0.5
            reasons.append(f"VIX低({vix:.0f})")
        elif vix > 30:
            vix_score = -0.5
            reasons.append(f"VIX高({vix:.0f})")
        else:
            vix_score = _clamp((20 - vix) / 20)
        scores.append(vix_score)

    score = _clamp(mean(scores)) if scores else 0.0
    return {
        "metrics": market,
        "score": score,
        "reason": "，".join(reasons) if reasons else "大盤中性",
    }


# ------------------------------------------------------------------
# Lean combination
# ------------------------------------------------------------------


def combine_lean(
    facet_scores: dict[str, Optional[float]],
    weights: dict[str, float],
) -> dict[str, Any]:
    """Combine per-facet scores into a directional lean.

    - Ignores facets with score=None.
    - Re-normalises weights over the non-None facets.
    - Weighted average → score in [-1, 1].
    - Bias: score >= +0.25 → 偏多; <= -0.25 → 偏空; else 中性.
    - Confidence by |score|: >0.5 高 / >0.25 中 / else 低.
    - If all facets are None: score=0, 中性, 低.
    Returns {"bias", "confidence", "score", "reasons"}.
    """
    active = {
        k: v for k, v in facet_scores.items()
        if v is not None and k in weights
    }

    if not active:
        return {"bias": "中性", "confidence": "低", "score": 0.0, "reasons": []}

    total_weight = sum(weights[k] for k in active)
    if total_weight == 0:
        return {"bias": "中性", "confidence": "低", "score": 0.0, "reasons": []}

    score = sum(active[k] * weights[k] for k in active) / total_weight

    if score >= LEAN_BULL_THRESHOLD:
        bias = "偏多"
    elif score <= LEAN_BEAR_THRESHOLD:
        bias = "偏空"
    else:
        bias = "中性"

    abs_score = abs(score)
    if abs_score > LEAN_HIGH_CONF:
        confidence = "高"
    elif abs_score > LEAN_MED_CONF:
        confidence = "中"
    else:
        confidence = "低"

    reasons = list(active.keys())

    return {
        "bias": bias,
        "confidence": confidence,
        "score": score,
        "reasons": reasons,
    }


# ------------------------------------------------------------------
# 2330 scorecard builder
# ------------------------------------------------------------------


def build_2330_scorecard(
    daily_bars: list[MarketBar],
    weekly_context: Optional[dict[str, Any]],
    valuation: Optional[dict[str, Any]],
    revenue: Optional[dict[str, Any]],
    t86_recent: Optional[list[dict[str, Any]]],
    peer: Optional[dict[str, Any]],
    market: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Build a multi-facet scorecard for 2330 (TSMC) with short and medium leans.

    Args:
        daily_bars: list of recent daily MarketBar (at least 60 recommended).
        weekly_context: optional dict passed through to peer facet.
        valuation: output of valuation_metrics() — dict with pe_pct, pb_pct, etc.
        revenue: output of revenue_metrics() — dict with yoy_pct, etc.
        t86_recent: list of up to 5 daily T86 dicts for chip facet.
        peer: dict with smh_ret20, adr_ret for peer facet.
        market: dict with taiex_ret20, vix for market facet.

    Returns:
        {
            "facets": {name: facet_result, ...},
            "short_lean": combine_lean result,
            "medium_lean": combine_lean result,
            "missing": [list of facet names with score=None],
        }
    """
    tech_m = technical_metrics(daily_bars) if daily_bars else None

    facets: dict[str, dict[str, Any]] = {
        "technical": technical_facet(tech_m),
        "volume": volume_facet(tech_m),
        "valuation": valuation_facet(valuation),
        "revenue": revenue_facet(revenue),
        "chip": chip_facet(t86_recent),
        "peer": peer_facet(peer),
        "market": market_facet(market),
    }

    scores: dict[str, Optional[float]] = {name: f["score"] for name, f in facets.items()}

    short_lean = combine_lean(scores, SHORT_WEIGHTS)
    medium_lean = combine_lean(scores, MEDIUM_WEIGHTS)

    missing = [name for name, f in facets.items() if f["score"] is None]

    return {
        "facets": facets,
        "short_lean": short_lean,
        "medium_lean": medium_lean,
        "missing": missing,
    }
