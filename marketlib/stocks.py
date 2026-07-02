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
