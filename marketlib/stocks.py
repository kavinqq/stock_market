"""Per-stock TWSE data fetchers.

Pure parse functions (_parse_*) accept inline fixture payloads for testing.
Network fetch_* wrappers call http_get_json then delegate to parsers,
returning None / [] on any failure — never raise to caller.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Optional

from marketlib.core import MarketBar, clean_num, pct_change  # noqa: F401
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
