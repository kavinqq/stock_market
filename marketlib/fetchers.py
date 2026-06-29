from __future__ import annotations
import csv, html, json, re, shutil, subprocess, time, urllib.parse, urllib.request
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Any, Optional, Union
from marketlib.core import MarketBar, clean_num, pct_change

TAIPEI_TZ = ZoneInfo("Asia/Taipei")

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0 Safari/537.36"
)

TWSE_BASE = "https://www.twse.com.tw/rwd/zh"
NASDAQ_BASE = "https://api.nasdaq.com/api/quote"
CBOE_VIX_CSV = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
FINRA_MARGIN_URL = "https://www.finra.org/rules-guidance/key-topics/margin-accounts/margin-statistics"

NASDAQ_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://www.nasdaq.com",
    "Referer": "https://www.nasdaq.com/",
}


def curl_get(url: str, headers: dict[str, str]) -> str:
    curl_path = shutil.which("curl")
    if not curl_path:
        raise RuntimeError("curl is not available")

    args = [
        curl_path,
        "--http1.1",
        "-fsSL",
        "--retry",
        "2",
        "--connect-timeout",
        "20",
        "--max-time",
        "90",
        "-A",
        headers.get("User-Agent", USER_AGENT),
    ]
    for key, value in headers.items():
        if key.lower() == "user-agent":
            continue
        args.extend(["-H", f"{key}: {value}"])
    args.append(url)

    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout[:200]
        raise RuntimeError(f"curl failed for {url}: {detail}")
    return result.stdout


def curl_get_bare(url: str) -> str:
    curl_path = shutil.which("curl")
    if not curl_path:
        raise RuntimeError("curl is not available")

    args = [
        curl_path,
        "-sSL",
        "--retry",
        "2",
        "--connect-timeout",
        "20",
        "--max-time",
        "90",
        url,
    ]
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout[:200]
        raise RuntimeError(f"bare curl failed for {url}: {detail}")
    return result.stdout


def http_get(url: str, headers: Optional[dict[str, str]] = None, retries: int = 3) -> str:
    request_headers = {"User-Agent": USER_AGENT}
    if headers:
        request_headers.update(headers)

    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=request_headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                charset = resp.headers.get_content_charset() or "utf-8"
                return resp.read().decode(charset, errors="replace")
        except Exception as exc:  # noqa: BLE001 - keep cron failure readable.
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * attempt)
    try:
        return curl_get(url, request_headers)
    except Exception as curl_error:  # noqa: BLE001 - include both failure paths.
        raise RuntimeError(
            f"failed to fetch {url}: urllib={last_error}; curl={curl_error}"
        ) from curl_error


def http_get_json(url: str, headers: Optional[dict[str, str]] = None) -> dict[str, Any]:
    text = http_get(url, headers=headers)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        snippet = text[:240].replace("\n", " ")
        raise RuntimeError(f"expected JSON from {url}, got: {snippet}") from exc


def month_starts(end: date, count: int) -> list[date]:
    months: list[date] = []
    year, month = end.year, end.month
    for _ in range(count):
        months.append(date(year, month, 1))
        month -= 1
        if month == 0:
            month = 12
            year -= 1
    return list(reversed(months))


def roc_date_to_iso(value: str) -> str:
    year_s, month_s, day_s = value.split("/")
    return f"{int(year_s) + 1911:04d}-{int(month_s):02d}-{int(day_s):02d}"


def iso_to_yyyymmdd(value: Union[str, date]) -> str:
    if isinstance(value, date):
        return value.strftime("%Y%m%d")
    return value.replace("-", "")


def parse_us_date(value: str) -> str:
    dt = datetime.strptime(value, "%m/%d/%Y")
    return dt.strftime("%Y-%m-%d")


def fetch_twse_market_month(month_start: date) -> list[MarketBar]:
    url = (
        f"{TWSE_BASE}/afterTrading/FMTQIK?"
        f"date={month_start.strftime('%Y%m%d')}&response=json"
    )
    payload = http_get_json(url)
    if payload.get("stat") != "OK":
        return []
    bars: list[MarketBar] = []
    for row in payload.get("data", []):
        iso = roc_date_to_iso(row[0])
        bars.append(
            MarketBar(
                date=iso,
                volume=clean_num(row[1]),
                turnover=clean_num(row[2]),
                trades=clean_num(row[3]),
                close=clean_num(row[4]) or 0.0,
                change_points=clean_num(row[5]),
            )
        )
    return bars


def fetch_twse_market_history(end: date, lookback_months: int) -> list[MarketBar]:
    bars: list[MarketBar] = []
    for month_start in month_starts(end, lookback_months):
        bars.extend(fetch_twse_market_month(month_start))
    dedup = {bar.date: bar for bar in bars if bar.close}
    return [dedup[key] for key in sorted(dedup)]


def find_twse_table(payload: dict[str, Any], title_part: str) -> Optional[dict[str, Any]]:
    for table in payload.get("tables", []):
        if title_part in table.get("title", ""):
            return table
    return None


def fetch_twse_margin(latest_date: str) -> Optional[dict[str, Any]]:
    url = (
        f"{TWSE_BASE}/marginTrading/MI_MARGN?"
        f"date={iso_to_yyyymmdd(latest_date)}&selectType=MS&response=json"
    )
    payload = http_get_json(url)
    if payload.get("stat") != "OK":
        return None
    table = find_twse_table(payload, "信用交易統計")
    if not table:
        return None

    parsed: dict[str, Any] = {"source_date": latest_date}
    for row in table.get("data", []):
        label = row[0]
        values = [clean_num(v) for v in row[1:]]
        if label == "融資金額(仟元)":
            buy_k, sell_k, repay_k, prev_k, today_k = values
            parsed["margin_financing"] = {
                "buy_ntd": buy_k * 1000 if buy_k is not None else None,
                "sell_ntd": sell_k * 1000 if sell_k is not None else None,
                "cash_repay_ntd": repay_k * 1000 if repay_k is not None else None,
                "previous_balance_ntd": prev_k * 1000 if prev_k is not None else None,
                "today_balance_ntd": today_k * 1000 if today_k is not None else None,
                "balance_change_ntd": (
                    (today_k - prev_k) * 1000
                    if today_k is not None and prev_k is not None
                    else None
                ),
            }
        elif label == "融資(交易單位)":
            buy, sell, repay, prev, today = values
            parsed["margin_lots"] = {
                "buy": buy,
                "sell": sell,
                "repay": repay,
                "previous": prev,
                "today": today,
                "change": today - prev if today is not None and prev is not None else None,
            }
        elif label == "融券(交易單位)":
            buy, sell, repay, prev, today = values
            parsed["short_lots"] = {
                "buy": buy,
                "sell": sell,
                "repay": repay,
                "previous": prev,
                "today": today,
                "change": today - prev if today is not None and prev is not None else None,
            }
    return parsed


def fetch_twse_institutional(latest_date: str) -> Optional[dict[str, Any]]:
    ymd = iso_to_yyyymmdd(latest_date)
    query = urllib.parse.urlencode(
        {
            "dayDate": ymd,
            "weekDate": ymd,
            "monthDate": ymd,
            "type": "day",
            "response": "json",
        }
    )
    payload = http_get_json(f"{TWSE_BASE}/fund/BFI82U?{query}")
    if payload.get("stat") != "OK":
        return None
    rows: dict[str, dict[str, Optional[float]]] = {}
    for row in payload.get("data", []):
        rows[row[0]] = {
            "buy_ntd": clean_num(row[1]),
            "sell_ntd": clean_num(row[2]),
            "net_ntd": clean_num(row[3]),
        }
    return {"source_date": latest_date, "rows": rows}


def fetch_twse_summary(latest_date: str) -> Optional[dict[str, Any]]:
    url = (
        f"{TWSE_BASE}/afterTrading/MI_INDEX?"
        f"date={iso_to_yyyymmdd(latest_date)}&type=MS&response=json"
    )
    payload = http_get_json(url)
    if payload.get("stat") != "OK":
        return None

    summary: dict[str, Any] = {"source_date": latest_date}
    market_table = find_twse_table(payload, "大盤統計資訊")
    if market_table:
        for row in market_table.get("data", []):
            if row[0].startswith("證券合計"):
                summary["securities_total"] = {
                    "turnover_ntd": clean_num(row[1]),
                    "shares": clean_num(row[2]),
                    "trades": clean_num(row[3]),
                }
            elif row[0] == "總計(1~15)":
                summary["market_total"] = {
                    "turnover_ntd": clean_num(row[1]),
                    "shares": clean_num(row[2]),
                    "trades": clean_num(row[3]),
                }

    breadth_table = find_twse_table(payload, "漲跌證券數合計")
    if breadth_table:
        breadth: dict[str, Any] = {}
        for row in breadth_table.get("data", []):
            label = re.sub(r"\(.*?\)", "", row[0])
            whole = clean_num(str(row[1]).split("(")[0])
            stocks = clean_num(str(row[2]).split("(")[0])
            breadth[label] = {"whole_market": whole, "stocks": stocks}
        up = breadth.get("上漲", {}).get("stocks")
        down = breadth.get("下跌", {}).get("stocks")
        breadth["advance_decline_ratio_stocks"] = (
            up / down if up is not None and down not in {None, 0} else None
        )
        summary["breadth"] = breadth
    return summary


def fetch_twse_sector_indices(latest_date: str) -> Optional[dict[str, Any]]:
    url = (
        f"{TWSE_BASE}/afterTrading/MI_INDEX?"
        f"date={iso_to_yyyymmdd(latest_date)}&type=IND&response=json"
    )
    payload = http_get_json(url)
    if payload.get("stat") != "OK":
        return None
    wanted = {
        "發行量加權股價指數",
        "臺灣50指數",
        "臺灣中型100指數",
        "電子工業類指數",
        "半導體類指數",
        "電腦及週邊設備類指數",
        "金融保險類指數",
        "航運類指數",
    }
    sectors: dict[str, Any] = {}
    table = find_twse_table(payload, "價格指數(臺灣證券交易所)")
    if table:
        for row in table.get("data", []):
            if row[0] in wanted:
                sectors[row[0]] = {
                    "close": clean_num(row[1]),
                    "change_points": clean_num(row[3]),
                    "change_pct": (clean_num(row[4]) or 0.0) / 100,
                }
    return {"source_date": latest_date, "rows": sectors}


def fetch_nasdaq_history(
    symbol: str,
    assetclass: str,
    start_date: date,
    end_date: date,
) -> list[MarketBar]:
    query = urllib.parse.urlencode(
        {
            "assetclass": assetclass,
            "fromdate": start_date.strftime("%Y-%m-%d"),
            "todate": end_date.strftime("%Y-%m-%d"),
            "limit": 9999,
        }
    )
    url = f"{NASDAQ_BASE}/{symbol}/historical?{query}"
    payload = http_get_json(url, headers=NASDAQ_HEADERS)
    if payload.get("status", {}).get("rCode") != 200 or not payload.get("data"):
        raise RuntimeError(f"Nasdaq returned no data for {symbol}: {payload.get('status')}")

    rows = payload["data"]["tradesTable"].get("rows", [])
    bars: list[MarketBar] = []
    for row in rows:
        close = clean_num(row.get("close"))
        if close is None:
            continue
        bars.append(
            MarketBar(
                date=parse_us_date(row["date"]),
                close=close,
                volume=clean_num(row.get("volume")),
                open=clean_num(row.get("open")),
                high=clean_num(row.get("high")),
                low=clean_num(row.get("low")),
            )
        )
    return sorted(bars, key=lambda bar: bar.date)


def fetch_vix_history(start_date: date, end_date: date) -> list[MarketBar]:
    text = http_get(CBOE_VIX_CSV)
    rows = csv.DictReader(text.splitlines())
    bars: list[MarketBar] = []
    for row in rows:
        iso = parse_us_date(row["DATE"])
        current_date = datetime.strptime(iso, "%Y-%m-%d").date()
        if start_date <= current_date <= end_date:
            close = clean_num(row.get("CLOSE"))
            if close is not None:
                bars.append(
                    MarketBar(
                        date=iso,
                        close=close,
                        open=clean_num(row.get("OPEN")),
                        high=clean_num(row.get("HIGH")),
                        low=clean_num(row.get("LOW")),
                    )
                )
    return sorted(bars, key=lambda bar: bar.date)


def fetch_finra_margin_statistics() -> Optional[dict[str, Any]]:
    try:
        text = curl_get_bare(FINRA_MARGIN_URL)
    except Exception:
        return None
    if "Just a moment" in text and "challenges.cloudflare.com" in text:
        return None
    stripped = html.unescape(re.sub(r"<[^>]+>", " ", text))
    stripped = re.sub(r"\s+", " ", stripped)
    pattern = re.compile(
        r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{2})\s+"
        r"([\d,]+)\s+([\d,]+)\s+([\d,]+)"
    )
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for match in pattern.finditer(stripped):
        month = match.group(1)
        if month in seen:
            continue
        seen.add(month)
        debit = clean_num(match.group(2))
        free_cash = clean_num(match.group(3))
        free_margin = clean_num(match.group(4))
        rows.append(
            {
                "month": month,
                "debit_balances_millions": debit,
                "free_credit_cash_millions": free_cash,
                "free_credit_margin_millions": free_margin,
                "free_credit_total_millions": (
                    free_cash + free_margin
                    if free_cash is not None and free_margin is not None
                    else None
                ),
            }
        )
    if not rows:
        return None

    latest = rows[0]
    previous = rows[1] if len(rows) > 1 else None
    year_back = rows[12] if len(rows) > 12 else None
    latest_debit = latest.get("debit_balances_millions")
    latest_credit = latest.get("free_credit_total_millions")
    latest["debit_to_free_credit_ratio"] = (
        latest_debit / latest_credit
        if latest_debit is not None and latest_credit not in {None, 0}
        else None
    )
    latest["mom_debit_pct"] = (
        pct_change(latest_debit, previous.get("debit_balances_millions"))
        if previous and latest_debit is not None
        else None
    )
    latest["yoy_debit_pct"] = (
        pct_change(latest_debit, year_back.get("debit_balances_millions"))
        if year_back and latest_debit is not None
        else None
    )
    return {"latest": latest, "rows": rows[:24], "source_url": FINRA_MARGIN_URL}


def _iso(dt: datetime) -> str:
    return dt.astimezone(TAIPEI_TZ).isoformat(timespec="seconds")

def twse_asof(date_iso: str, group: str) -> str:
    d = datetime.fromisoformat(date_iso).replace(tzinfo=TAIPEI_TZ)
    # 保守近似:價量收盤後 14:30;法人/廣度 ~16:00;融資 ~16:30(皆 < 18:30 cron)
    hours = {"price": (14, 30), "breadth": (16, 0), "inst": (16, 0), "margin": (16, 30)}
    h, m = hours.get(group, (16, 30))
    return _iso(d.replace(hour=h, minute=m))

def us_price_asof(date_iso: str) -> str:
    # 美股當日收盤 16:00 ET ≈ 次日 Taipei 凌晨;保守記為次日 05:00 Taipei
    d = datetime.fromisoformat(date_iso).replace(tzinfo=TAIPEI_TZ) + timedelta(days=1)
    return _iso(d.replace(hour=5, minute=0))

def vix_asof(date_iso: str) -> str:
    return us_price_asof(date_iso)

def finra_asof(month: str) -> str:
    # month "YYYY-MM";保守:次月第 20 日 23:59 Taipei 起可用
    y, m = (int(x) for x in month.split("-"))
    ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
    return _iso(datetime(ny, nm, 20, 23, 59, tzinfo=TAIPEI_TZ))
