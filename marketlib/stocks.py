"""Per-stock TWSE data fetchers and technical indicators.

Pure parse functions (_parse_*) accept inline fixture payloads for testing.
Network fetch_* wrappers call http_get_json then delegate to parsers,
returning None / [] on any failure — never raise to caller.
"""
from __future__ import annotations

from datetime import date, timedelta
from statistics import mean, median
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

# Minimum number of PE/PB history points required for a reliable percentile rank.
# Fewer points → percentile is unreliable (off by a large fraction), so we suppress
# the bogus cheap/expensive score rather than misleading the user.
MIN_VALUATION_HISTORY = 60


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
                "yoy_pct": clean_num(item.get("營業收入-去年同月增減(%)")),
                "mom_pct": clean_num(item.get("營業收入-上月比較增減(%)")),
                "cum_yoy_pct": clean_num(item.get("累計營業收入-前期比較增減(%)")),
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
# Batch all-stocks parse helpers
# ---------------------------------------------------------------------------


def _parse_bwibbu_all(payload: dict[str, Any], codes: set) -> dict[str, dict]:
    """Parse BWIBBU_d valuation payload for all codes in the given set.

    Returns {code: {pe, pb, dividend_yield, name}} for each code found in the
    payload.  The "name" field is the live-fetched 證券名稱 and is the primary
    source of stock names; TW50_NAMES in universe.py is only a fallback.
    Codes not present in the payload are silently skipped.
    Reuses the same header-index helpers and substring targets as _parse_bwibbu.
    """
    if payload.get("stat") != "OK":
        return {}
    fields = payload.get("fields", [])
    data = payload.get("data", [])

    try:
        code_idx = _find_field_index(fields, "證券代號")
        pe_idx = _find_field_index_substring(fields, "本益比")
        pb_idx = _find_field_index_substring(fields, "股價淨值比")
        div_idx = _find_field_index_substring(fields, "殖利率(%)")
    except ValueError:
        return {}

    # 證券名稱 column is optional — gracefully absent in some fixtures.
    try:
        name_idx: int | None = _find_field_index(fields, "證券名稱")
    except ValueError:
        name_idx = None

    result: dict[str, dict] = {}
    for row in data:
        try:
            code = row[code_idx].strip()
            if code not in codes:
                continue
            name = row[name_idx].strip() if name_idx is not None and name_idx < len(row) else ""
            result[code] = {
                "pe": clean_num(row[pe_idx]),
                "pb": clean_num(row[pb_idx]),
                "dividend_yield": clean_num(row[div_idx]),
                "name": name,
            }
        except (IndexError, ValueError):
            continue
    return result


def _parse_t86_all(payload: dict[str, Any], codes: set) -> dict[str, dict]:
    """Parse T86 institutional net buy/sell payload for all codes in the given set.

    Returns {code: {foreign_net_lots, trust_net_lots, dealer_net_lots,
    inst_total_net_lots, name}} for each code found in the payload.
    The "name" field is the live-fetched 證券名稱 and is the primary source of
    stock names; TW50_NAMES in universe.py is only a fallback.
    Shares → lots by /1000.
    Reuses the same header-index helpers and substring targets as _parse_t86.
    """
    if payload.get("stat") != "OK":
        return {}
    fields = payload.get("fields", [])
    data = payload.get("data", [])

    try:
        code_idx = _find_field_index(fields, "證券代號")
        total_idx = _find_field_index_substring(fields, "三大法人買賣超股數")
        trust_idx = _find_field_index_substring(fields, "投信買賣超股數")
        try:
            foreign_idx = _find_field_index_substring(fields, "外陸資買賣超股數(不含外資自營商)")
        except ValueError:
            foreign_idx = _find_field_index_substring(fields, "外資買賣超股數")
        dealer_indices = _find_all_field_indices_substring(fields, "自營商買賣超股數")
    except ValueError:
        return {}

    # 證券名稱 column is optional — gracefully absent in some fixtures.
    try:
        name_idx: int | None = _find_field_index(fields, "證券名稱")
    except ValueError:
        name_idx = None

    result: dict[str, dict] = {}
    for row in data:
        try:
            code = row[code_idx].strip()
            if code not in codes:
                continue
            foreign_shares = clean_num(row[foreign_idx]) or 0.0
            trust_shares = clean_num(row[trust_idx]) or 0.0
            total_shares = clean_num(row[total_idx]) or 0.0
            dealer_shares = sum(clean_num(row[i]) or 0.0 for i in dealer_indices)
            name = row[name_idx].strip() if name_idx is not None and name_idx < len(row) else ""
            result[code] = {
                "foreign_net_lots": foreign_shares / 1000,
                "trust_net_lots": trust_shares / 1000,
                "dealer_net_lots": dealer_shares / 1000,
                "inst_total_net_lots": total_shares / 1000,
                "name": name,
            }
        except (IndexError, ValueError):
            continue
    return result


def _parse_monthly_revenue_all(items: list[dict[str, Any]], codes: set) -> dict[str, dict]:
    """Parse monthly revenue OpenAPI list for all codes in the given set.

    Returns {code: {month, revenue_thousand, yoy_pct, mom_pct, cum_yoy_pct}}
    for each code found in items. Uses the same REAL prefixed keys as _parse_monthly_revenue.
    """
    result: dict[str, dict] = {}
    for item in items:
        code = item.get("公司代號")
        if code not in codes:
            continue
        try:
            roc_ym = str(item["資料年月"])
            year = int(roc_ym[:-2]) + 1911
            month = int(roc_ym[-2:])
            month_str = f"{year:04d}-{month:02d}"
            result[code] = {
                "month": month_str,
                "revenue_thousand": clean_num(item.get("營業收入-當月營收")),
                "yoy_pct": clean_num(item.get("營業收入-去年同月增減(%)")),
                "mom_pct": clean_num(item.get("營業收入-上月比較增減(%)")),
                "cum_yoy_pct": clean_num(item.get("累計營業收入-前期比較增減(%)")),
            }
        except (KeyError, ValueError, TypeError):
            continue
    return result


# ---------------------------------------------------------------------------
# Name resolution helper
# ---------------------------------------------------------------------------


def resolve_name(code: str, *sources: dict[str, dict]) -> str:
    """Return the best available display name for a stock code.

    Resolution order:
    1. First non-empty ``sources[i][code]["name"]`` (live-fetched 證券名稱,
       from _parse_bwibbu_all / _parse_t86_all — primary source).
    2. ``TW50_NAMES.get(code)`` — static fallback map; best-effort only.
    3. The ``code`` string itself — always returns something displayable.

    Args:
        code: Stock code string (e.g. "2330").
        *sources: Zero or more dicts keyed by code, each value being a dict
                  that may contain a "name" key (output of batch parse helpers).

    Returns:
        A non-empty display string.
    """
    from marketlib.universe import TW50_NAMES  # local import avoids circular dependency

    for source in sources:
        entry = source.get(code)
        if entry:
            name = entry.get("name")
            if name:
                return name
    fallback = TW50_NAMES.get(code)
    if fallback:
        return fallback
    return code


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


def fetch_valuation_all(date_iso: str, codes) -> dict[str, dict]:
    """Fetch BWIBBU_d valuation for all stocks in one request, filter to codes.

    Returns {code: {pe, pb, dividend_yield}} on success; {} on any failure.
    """
    try:
        url = (
            f"{TWSE_BASE}/afterTrading/BWIBBU_d?"
            f"date={iso_to_yyyymmdd(date_iso)}&selectType=ALL&response=json"
        )
        payload = http_get_json(url)
        return _parse_bwibbu_all(payload, set(codes))
    except Exception:  # noqa: BLE001
        return {}


def fetch_t86_all(date_iso: str, codes) -> dict[str, dict]:
    """Fetch T86 institutional net buy/sell for all stocks in one request, filter to codes.

    Returns {code: {foreign_net_lots, trust_net_lots, dealer_net_lots, inst_total_net_lots}}
    on success; {} on any failure.
    """
    try:
        url = (
            f"{TWSE_BASE}/fund/T86?"
            f"date={iso_to_yyyymmdd(date_iso)}&selectType=ALL&response=json"
        )
        payload = http_get_json(url)
        return _parse_t86_all(payload, set(codes))
    except Exception:  # noqa: BLE001
        return {}


def fetch_monthly_revenue_all(codes) -> dict[str, dict]:
    """Fetch latest monthly revenue for all stocks in one request, filter to codes.

    Returns {code: {month, revenue_thousand, yoy_pct, mom_pct, cum_yoy_pct}}
    on success; {} on any failure.
    """
    try:
        items = http_get_json(MONTHLY_REVENUE_URL)
        if not isinstance(items, list):
            return {}
        return _parse_monthly_revenue_all(items, set(codes))
    except Exception:  # noqa: BLE001
        return {}


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


def bollinger(closes: list[float], n: int = 20, k: float = 2.0) -> Optional[dict]:
    """Compute Bollinger Bands for the last n closes.

    Returns {"mid", "upper", "lower", "pctb"}.
    mid = SMA(n); upper = mid + k*pstdev; lower = mid - k*pstdev.
    pctb = (close - lower) / (upper - lower); None when band width is 0.
    Returns None when len(closes) < n.
    """
    if len(closes) < n:
        return None
    from statistics import pstdev
    window = closes[-n:]
    mid = mean(window)
    std = pstdev(window)
    upper = mid + k * std
    lower = mid - k * std
    band_width = upper - lower
    if band_width == 0:
        pctb = None
    else:
        pctb = (closes[-1] - lower) / band_width
    return {"mid": mid, "upper": upper, "lower": lower, "pctb": pctb}


def vwap(bars: list[MarketBar], n: int) -> Optional[float]:
    """Compute volume-weighted average price (VWAP proxy) over last n bars.

    VWAP = Σ(close * volume) / Σ(volume) over last n bars.
    This is a 大戶成本 (institutional cost) ESTIMATE — not real broker-cost data.
    Returns None when len(bars) < n or Σvolume == 0.
    """
    if len(bars) < n:
        return None
    window = bars[-n:]
    total_vol = sum(b.volume for b in window if b.volume is not None)
    if total_vol == 0:
        return None
    total_pv = sum(
        b.close * b.volume
        for b in window
        if b.volume is not None and b.close is not None
    )
    return total_pv / total_vol


def resample_monthly(bars: list[MarketBar]) -> list[MarketBar]:
    """Group daily MarketBar list into monthly bars by (year, month).

    Per month: close = last daily close, high = max, low = min,
    volume = sum, date = last date in that month.
    Returns ordered list of monthly MarketBar.
    """
    from datetime import date as date_cls
    from collections import OrderedDict

    groups: dict[tuple, list[MarketBar]] = OrderedDict()
    for bar in bars:
        d = date_cls.fromisoformat(bar.date)
        ym = (d.year, d.month)
        if ym not in groups:
            groups[ym] = []
        groups[ym].append(bar)

    monthly: list[MarketBar] = []
    for month_bars in groups.values():
        last = month_bars[-1]
        highs = [b.high for b in month_bars if b.high is not None]
        lows = [b.low for b in month_bars if b.low is not None]
        vols = [b.volume for b in month_bars if b.volume is not None]
        monthly.append(
            MarketBar(
                date=last.date,
                close=last.close,
                high=max(highs) if highs else None,
                low=min(lows) if lows else None,
                volume=sum(vols) if vols else None,
            )
        )
    return monthly


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
    close, ma5, ma20, ma60, ma120, ma240,
    ret5, ret20, ret60, drawdown,
    rsi, kd_k, kd_d, kd (dict),
    macd_daily (dict or None), macd_weekly (dict or None),
    macd_monthly (dict or None),
    boll (dict or None),
    cost_vwap60, cost_vwap120, price_vs_cost60,
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
    ma120 = mean(closes[-120:]) if n >= 120 else None
    ma240 = mean(closes[-240:]) if n >= 240 else None

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

    monthly_bars = resample_monthly(daily_bars)
    monthly_closes = [b.close for b in monthly_bars]
    macd_monthly = macd(monthly_closes)

    boll = bollinger(closes)

    cost_vwap60 = vwap(daily_bars, 60)
    cost_vwap120 = vwap(daily_bars, 120)

    price_vs_cost60: Optional[float] = None
    if cost_vwap60 is not None and cost_vwap60 != 0 and closes:
        price_vs_cost60 = closes[-1] / cost_vwap60 - 1

    vol_ratio = volume_ratio_excluding_t(volumes, 20)

    return {
        "close": closes[-1] if closes else None,
        "ma5": ma5,
        "ma20": ma20,
        "ma60": ma60,
        "ma120": ma120,
        "ma240": ma240,
        "ret5": ret5,
        "ret20": ret20,
        "ret60": ret60,
        "drawdown": drawdown,
        "rsi": rsi_val,
        "kd_k": kd_result["k"] if kd_result else None,
        "kd_d": kd_result["d"] if kd_result else None,
        "kd": kd_result,
        "macd_daily": macd_daily,
        "macd_weekly": macd_weekly,
        "macd_monthly": macd_monthly,
        "boll": boll,
        "cost_vwap60": cost_vwap60,
        "cost_vwap120": cost_vwap120,
        "price_vs_cost60": price_vs_cost60,
        "volume_ratio_20d": vol_ratio,
    }


def tech_raw_score(daily_bars: list[MarketBar]) -> Optional[float]:
    """Compute a raw 0..1 technical bullishness score from daily bar history.

    Scores the fraction of these seven conditions that hold:
    1. close >= ma20 (月線)
    2. ma20 >= ma60 (季線)
    3. daily MACD bullish: cross=="golden" OR hist>0
    4. weekly MACD hist>0
    5. boll pctb > 0.5
    6. close >= cost_vwap60 (站上大戶成本)
    7. KD: k >= d

    Returns the fraction of *computable* conditions that hold (0.0 to 1.0).
    Returns None if fewer than 2 of the conditions are computable
    (insufficient bars to evaluate them).
    """
    m = technical_metrics(daily_bars)

    close = m.get("close")
    ma20 = m.get("ma20")
    ma60 = m.get("ma60")
    macd_daily = m.get("macd_daily")
    macd_weekly = m.get("macd_weekly")
    boll = m.get("boll")
    cost_vwap60 = m.get("cost_vwap60")
    kd_result = m.get("kd")

    results: list[bool] = []

    # Condition 1: close >= ma20
    if close is not None and ma20 is not None:
        results.append(close >= ma20)

    # Condition 2: ma20 >= ma60
    if ma20 is not None and ma60 is not None:
        results.append(ma20 >= ma60)

    # Condition 3: daily MACD bullish (golden cross OR hist > 0)
    if macd_daily is not None:
        cross = macd_daily.get("cross")
        hist = macd_daily.get("hist")
        bullish = (cross == "golden") or (hist is not None and hist > 0)
        results.append(bullish)

    # Condition 4: weekly MACD hist > 0
    if macd_weekly is not None:
        hist_w = macd_weekly.get("hist")
        if hist_w is not None:
            results.append(hist_w > 0)

    # Condition 5: boll pctb > 0.5
    if boll is not None and boll.get("pctb") is not None:
        results.append(boll["pctb"] > 0.5)

    # Condition 6: close >= cost_vwap60 (站上大戶成本代理)
    if close is not None and cost_vwap60 is not None:
        results.append(close >= cost_vwap60)

    # Condition 7: KD k >= d
    if kd_result is not None:
        results.append(kd_result["k"] >= kd_result["d"])

    if len(results) < 2:
        return None

    return sum(results) / len(results)


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

    # Apply minimum-history guard: percentile rank is unreliable when the history
    # is too thin (< MIN_VALUATION_HISTORY points).  Set to None so valuation_facet
    # can detect the "insufficient history" case and suppress a misleading score.
    pe_pct: Optional[float] = None
    if pe is not None and len(pe_history) >= MIN_VALUATION_HISTORY:
        pe_pct = percentile_rank(pe_history, pe)

    pb_pct: Optional[float] = None
    if pb is not None and len(pb_history) >= MIN_VALUATION_HISTORY:
        pb_pct = percentile_rank(pb_history, pb)

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

    When pe_pct is None (insufficient history via valuation_metrics guard),
    returns score=None with raw PE/PB still in metrics so the UI can show them.
    """
    if not metrics:
        return dict(_MISSING_FACET)

    pe_pct = metrics.get("pe_pct")
    pb_pct = metrics.get("pb_pct")

    # Insufficient history: pe_pct is None despite having a current PE value.
    # Show raw metrics (pe/pb/dividend_yield) but suppress the bogus score.
    if pe_pct is None and pb_pct is None:
        # Distinguish "no current data at all" from "thin history"
        has_raw = metrics.get("pe") is not None or metrics.get("pb") is not None
        if has_raw:
            return {
                "score": None,
                "reason": "估值歷史不足(需累積更多PE)",
                "metrics": {
                    "pe": metrics.get("pe"),
                    "pb": metrics.get("pb"),
                    "dividend_yield": metrics.get("dividend_yield"),
                },
            }
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
    score = _clamp(direction * (0.5 * norm + 0.5 * abs(2 * pos_ratio - 1)))

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
    加權多頭(ret>0) + VIX 低(<20) → +; VIX 高(>30) → −.
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
        # VIX: low (<20) = +0.5, medium (20-30) = 0, high (>30) = -0.5
        if vix < 20:
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


# ---------------------------------------------------------------------------
# Valuation band — "河流圖上現在算便宜還是貴"
#
# A PE river chart answers one question visually: where does today's valuation
# sit inside this stock's own historical range? That is a percentile, so we
# compute the percentile directly rather than drawing bands and eyeballing them.
# ---------------------------------------------------------------------------

# Trading days in the lookback. Eleven years of stored history spans regimes
# that no longer price the same way; five years is the window Taiwanese river
# charts conventionally use.
VALUATION_LOOKBACK_DAYS = 1250
MIN_VALUATION_BAND_HISTORY = 250

VALUATION_BANDS: list[tuple[float, str]] = [
    (0.20, "極便宜"),
    (0.40, "便宜"),
    (0.60, "合理"),
    (0.80, "偏貴"),
    (1.01, "昂貴"),
]

# Taiwanese financial holdings: 28xx, plus 5880 合庫金 and 5876 上海商銀.
# Their PE swings on one-off investment gains, so the market reads them on PB.
_FINANCIAL_CODES = {"5880", "5876"}

# Petrochemical, steel, shipping, cement — deep cyclicals whose trailing PE is
# LOWEST at the earnings peak, which is usually the worst moment to buy. A bare
# "cheap" reading on these is the classic value trap, so it ships with a caveat.
_CYCLICAL_CODES = {"1301", "1303", "1326", "1101", "1102", "2002",
                   "2603", "2609", "2615"}


def _is_financial(code: str) -> bool:
    return code.startswith("28") or code in _FINANCIAL_CODES


def valuation_band(
    value: Optional[float],
    history: list[float],
    lookback: int = VALUATION_LOOKBACK_DAYS,
    min_history: int = MIN_VALUATION_BAND_HISTORY,
) -> Optional[dict[str, Any]]:
    """Where `value` sits inside its own recent history.

    Returns ``{"percentile", "band", "sample_size", "low", "high"}``, or None
    when there is no value or too little history — a percentile over a handful
    of points is noise wearing a label, and would read as a real verdict.

    Lower percentile = cheaper, for both PE and PB.
    """
    clean = [h for h in history if h is not None and h > 0]
    if value is None or value <= 0:
        return None
    window = clean[-lookback:] if lookback else clean
    if len(window) < min_history:
        return None
    pct = percentile_rank(window, value)
    band = next(label for edge, label in VALUATION_BANDS if pct < edge)
    ordered = sorted(window)
    return {
        "percentile": pct,
        "band": band,
        "sample_size": len(window),
        "low": ordered[0],
        "high": ordered[-1],
    }


def river_valuation(
    code: str,
    pe: Optional[float],
    pb: Optional[float],
    pe_history: list[float],
    pb_history: list[float],
    lookback: int = VALUATION_LOOKBACK_DAYS,
    min_history: int = MIN_VALUATION_BAND_HISTORY,
    splits: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """The cheap-or-expensive verdict, on whichever multiple actually applies.

    PE leads for an ordinary operating company. PB leads for a financial
    holding, whose PE moves with one-off investment gains rather than with the
    business, and PB also stands in whenever PE is unusable (a loss-making
    quarter leaves 1101 台泥 with no PE at all).

    Both bands are always returned, so the reader can see the two disagree when
    they do, and `caveats` carries the reasons this reading may mislead.
    """
    # An unadjusted split makes the percentile meaningless rather than merely
    # noisy: the whole stored history sits on the old share base, so today's
    # restated multiple lands at the bottom of it by construction. 6669 read
    # "cheapest in five years" on a PE that was a third of its true value.
    split = (splits or [None])[-1] if splits else None
    if split:
        factor = split.get("factor") or 1.0
        return {
            "primary_metric": None, "band": None, "percentile": None,
            "sample_size": None, "pe": pe, "pb": pb,
            "pe_band": None, "pb_band": None,
            "split_date": split.get("date"), "split_factor": factor,
            "pe_split_adjusted": pe * factor if pe else None,
            "pb_split_adjusted": pb * factor if pb else None,
            "caveats": [
                f"{split.get('date')} 發生約 {factor:.2g}:1 股票分割，"
                f"交易所已調整股價但尚未調整每股盈餘/淨值，"
                f"目前 PE、PB 被低估約 {factor:.2g} 倍；"
                f"歷史分位全部建立在分割前基準，無法直接比較，故不給位階"
            ],
        }

    pe_band = valuation_band(pe, pe_history, lookback, min_history)
    pb_band = valuation_band(pb, pb_history, lookback, min_history)

    financial = _is_financial(code)
    prefer_pb = financial or pe_band is None
    primary = pb_band if prefer_pb else pe_band
    metric = None if primary is None else ("PB" if prefer_pb else "PE")

    caveats: list[str] = []
    if financial:
        caveats.append("金融股以 PB 判讀，PE 易受一次性投資損益扭曲")
    elif pe_band is None and pb_band is not None:
        caveats.append("近四季無可用 PE（可能為虧損或資料缺漏），改以 PB 判讀")
    if code in _CYCLICAL_CODES:
        caveats.append(
            "循環股：獲利高峰時 trailing PE 最低，河流圖顯示「便宜」常出現在景氣頂部"
        )
    if primary is not None and primary["percentile"] < 0.20:
        caveats.append("低分位也可能是市場已在反映獲利衰退，非必然便宜")

    return {
        "primary_metric": metric,
        "split_date": None,
        "split_factor": None,
        "pe_split_adjusted": None,
        "pb_split_adjusted": None,
        "band": None if primary is None else primary["band"],
        "percentile": None if primary is None else primary["percentile"],
        "sample_size": None if primary is None else primary["sample_size"],
        "pe": pe,
        "pb": pb,
        "pe_band": pe_band,
        "pb_band": pb_band,
        "caveats": caveats,
    }


# ---------------------------------------------------------------------------
# Unadjusted corporate actions
# ---------------------------------------------------------------------------

# TWSE caps a listed stock at ±10% per session, so a close-to-close move beyond
# that cannot come from trading. It is a split, a capital reduction, a large
# ex-dividend, or a data error — in every case the series is unadjusted and any
# indicator spanning the gap is computed across two different share bases.
# 11% leaves room for rounding on a limit move.
PRICE_LIMIT_PCT = 0.11


def detect_price_discontinuities(
    bars: list[MarketBar],
    limit: float = PRICE_LIMIT_PCT,
    lookback: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Close-to-close moves that the daily price limit makes impossible.

    Returns one entry per break: ``{"date", "prev_close", "close", "change_pct"}``.
    ``lookback`` restricts the scan to the last N bars, which is how a caller
    asks the question that actually matters — "does a break fall inside the
    window my indicators use?". A split three years back is irrelevant to a
    60-day average; one last week invalidates everything on screen.

    This uses the market's own rule rather than a guessed threshold, so it needs
    no corporate-action feed to know the series is broken.
    """
    window = bars[-(lookback + 1):] if lookback else bars
    out: list[dict[str, Any]] = []
    for prev, cur in zip(window, window[1:]):
        if not prev.close or cur.close is None:
            continue
        change = cur.close / prev.close - 1
        if abs(change) > limit:
            out.append({"date": cur.date, "prev_close": prev.close,
                        "close": cur.close, "change_pct": change})
    return out


# Longest trailing window any published indicator looks back over (MA240). A
# corporate action inside this many bars still distorts something on screen.
INDICATOR_SPAN_BARS = 240

# Daily bars needed before the weekly MACD's 26-period EMA has converged:
# MIN_WEEKLY_BARS (78) weeks at five sessions each.
MIN_BARS_FOR_WEEKLY_TREND = 390


# TWSE posts closing prices around 14:30 Taipei (see fetchers.twse_asof); 15:00
# is a conservative point after which today's session can be expected.
TWSE_PRICE_PUBLISH_HOUR = 15


def last_completed_trading_day(today: date, now=None) -> date:
    """The most recent session whose closing data TWSE has actually published.

    Today only counts once its close is out — a report generated at noon must
    ask for the previous session, or every stock fails a freshness check against
    data that does not exist yet.

    Weekends are skipped; holidays deliberately are not, since this carries no
    holiday calendar. On a holiday it names a day with no data, the freshness
    check asks for a refetch, and that refetch finds nothing new. One wasted
    request is the right failure: the alternative it replaced — a 20-calendar-day
    tolerance — let bars nine sessions old pass as current.
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo

    if now is None:
        now = datetime.now(ZoneInfo("Asia/Taipei"))
    d = today
    if d == now.date() and now.hour < TWSE_PRICE_PUBLISH_HOUR:
        d -= timedelta(days=1)
    while d.weekday() >= 5:            # Sat/Sun
        d -= timedelta(days=1)
    return d


# A split restates the price but, until the next quarterly filing, not trailing
# EPS or book value — so TWSE's PE, PB and yield all move by the same factor on
# the same day. That agreement is the signature: a genuine re-rating moves the
# multiples independently. 20% is well clear of daily drift and well below the
# smallest split worth worrying about.
_SPLIT_MIN_FACTOR_DEVIATION = 0.20
_SPLIT_RATIO_AGREEMENT = 0.05
# Sessions the restated level must hold for before it counts as a split rather
# than a bad row, and how far it may drift over them.
_SPLIT_PERSISTENCE_ROWS = 3
_SPLIT_PERSISTENCE_TOL = 0.15


def detect_valuation_splits(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Days where PE and PB jump by the same factor — an unadjusted split.

    Returns ``{"date", "factor", "pe_ratio", "pb_ratio"}`` per break, where
    ``factor`` is how far the multiples were divided (3.0 for a 3-for-1).

    This matters beyond the price column: the multiples are what a river chart
    reads, so an undetected split drops today's PE to a third of its true value
    and the percentile lands at the very bottom of a history recorded entirely
    on the old share base. The stock reads "cheapest in five years" precisely
    because the number is wrong.
    """
    out: list[dict[str, Any]] = []
    for i in range(1, len(rows)):
        prev, cur = rows[i - 1], rows[i]
        pe0, pe1 = prev.get("pe"), cur.get("pe")
        pb0, pb1 = prev.get("pb"), cur.get("pb")
        if not all(isinstance(v, (int, float)) and v > 0 for v in (pe0, pe1, pb0, pb1)):
            continue
        pe_ratio, pb_ratio = pe0 / pe1, pb0 / pb1
        # A split raises the share count, so PE and PB can only fall: factor > 1.
        # Anything below that is the data moving the other way.
        if pe_ratio < 1 + _SPLIT_MIN_FACTOR_DEVIATION:
            continue
        # Both multiples must move together by the same factor — a re-rating
        # moves them independently.
        if abs(pe_ratio - pb_ratio) / max(pe_ratio, pb_ratio) > _SPLIT_RATIO_AGREEMENT:
            continue
        # The level must be stable on BOTH sides, compared on medians so a
        # single bad row cannot pass for a level. TWSE's feed prints occasional
        # one-day outliers: 5880 showed PE 13.52 between two days at ~20.5, and
        # 2890 showed 16.91 between days at ~12. Checking only that the NEW
        # level sticks mistakes the spike for the old level and the return to
        # normal for a split — which is how an impossible 0.66:1 "split" got
        # reported. A real split moves both medians and neither comes back.
        k = _SPLIT_PERSISTENCE_ROWS
        before = [r["pe"] for r in rows[max(0, i - k): i]
                  if isinstance(r.get("pe"), (int, float)) and r["pe"] > 0]
        after = [r["pe"] for r in rows[i: i + k]
                 if isinstance(r.get("pe"), (int, float)) and r["pe"] > 0]
        if len(before) < k or len(after) < k:
            continue
        med_ratio = median(before) / median(after)
        if med_ratio < 1 + _SPLIT_MIN_FACTOR_DEVIATION:
            continue
        if abs(med_ratio - pe_ratio) / pe_ratio > _SPLIT_PERSISTENCE_TOL:
            continue
        # Medians decide WHETHER this is a split (robust to a bad row); the
        # consecutive-day ratio reports HOW BIG it was, since the restatement
        # happened between exactly those two sessions. A drifting pre-split
        # level pulls the median away from the true factor — 2.81 vs 2.99 for
        # 6669's 3:1.
        out.append({"date": cur["date"], "factor": (pe_ratio + pb_ratio) / 2,
                    "median_ratio": med_ratio,
                    "pe_ratio": pe_ratio, "pb_ratio": pb_ratio})
    return out
