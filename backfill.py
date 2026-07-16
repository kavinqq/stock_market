#!/usr/bin/env python3
"""Offline multi-year history backfill into SQLite (point-in-time aware)."""
from __future__ import annotations
import argparse, random, sys, time
from datetime import date, datetime, timedelta
from marketlib import fetchers as F
from marketlib.store import HistoryStore


class BudgetExhausted(Exception):
    """Raised when the per-run request budget is consumed."""


class Budget:
    """Counts outgoing network requests and enforces a per-run cap.

    Call spend() around each fetch; it raises BudgetExhausted once the running
    total exceeds max_requests so the caller can stop cleanly and resume next
    run via ingest_log.
    """

    def __init__(self, max_requests: int) -> None:
        self._max = max_requests
        self._count = 0

    @property
    def remaining(self) -> int:
        return max(0, self._max - self._count)

    def spend(self, n: int = 1) -> None:
        self._count += n
        if self._count > self._max:
            raise BudgetExhausted(
                f"request budget exhausted ({self._count}/{self._max})"
            )


def _spend(budget, n: int = 1) -> None:
    """Spend n requests from budget if provided; no-op when budget is None."""
    if budget is not None:
        budget.spend(n)


class _MergedNamespace:
    """Attribute proxy resolving fetchers from stocks first, then fetchers.

    incremental_run needs fetch_valuation_all / fetch_stock_day (in
    marketlib.stocks) AND fetch_twse_market_month / _institutional / _margin /
    _summary (in marketlib.fetchers). This proxy exposes both as one ns object.
    """

    def __init__(self, *modules):
        self._modules = modules

    def __getattr__(self, name):
        for mod in self._modules:
            if hasattr(mod, name):
                return getattr(mod, name)
        raise AttributeError(name)


def _max_asof(*vals):
    present = [v for v in vals if v]
    return max(present) if present else None

def _is_block_error(exc: BaseException) -> bool:
    """Return True when exc text indicates a TWSE anti-scraping block."""
    msg = str(exc)
    return any(kw in msg for kw in ("無法呈現", "CAN NOT BE ACCESSED", "SECURITY REASONS"))

def backfill_tw(store, ns, months, sleep: float = 1.5,
                max_retries: int = 3, backoff: float = 30.0,
                budget=None) -> None:
    for month_start in months:
        key = month_start.strftime("%Y-%m")
        month_had_block = False
        _spend(budget)
        bars = ns.fetch_twse_market_month(month_start)
        for bar in bars:
            d = bar.date

            # --- fetch_twse_institutional with block-retry ---
            inst = {}
            for attempt in range(max_retries + 1):
                _spend(budget)
                try:
                    inst = ns.fetch_twse_institutional(d) or {}
                    break
                except Exception as exc:
                    if _is_block_error(exc):
                        if attempt < max_retries:
                            wait = backoff * (2 ** attempt)
                            wait = min(wait, backoff * 8)
                            time.sleep(wait)
                        else:
                            month_had_block = True
                            inst = {}
                    else:
                        print(f"[backfill_tw] institutional fetch error on {d}: {exc}",
                              file=sys.stderr)
                        inst = {}
                        break

            # --- fetch_twse_margin with block-retry ---
            margin = {}
            for attempt in range(max_retries + 1):
                _spend(budget)
                try:
                    margin = ns.fetch_twse_margin(d) or {}
                    break
                except Exception as exc:
                    if _is_block_error(exc):
                        if attempt < max_retries:
                            wait = backoff * (2 ** attempt)
                            wait = min(wait, backoff * 8)
                            time.sleep(wait)
                        else:
                            month_had_block = True
                            margin = {}
                    else:
                        print(f"[backfill_tw] margin fetch error on {d}: {exc}",
                              file=sys.stderr)
                        margin = {}
                        break

            # --- fetch_twse_summary with block-retry ---
            summary = {}
            for attempt in range(max_retries + 1):
                _spend(budget)
                try:
                    summary = ns.fetch_twse_summary(d) or {}
                    break
                except Exception as exc:
                    if _is_block_error(exc):
                        if attempt < max_retries:
                            wait = backoff * (2 ** attempt)
                            wait = min(wait, backoff * 8)
                            time.sleep(wait)
                        else:
                            month_had_block = True
                            summary = {}
                    else:
                        print(f"[backfill_tw] summary fetch error on {d}: {exc}",
                              file=sys.stderr)
                        summary = {}
                        break

            net = (inst.get("rows", {}).get("合計") or {}).get("net_ntd")
            fin = margin.get("margin_financing") or {}
            breadth = summary.get("breadth") or {}
            up = (breadth.get("上漲") or {}).get("stocks")
            down = (breadth.get("下跌") or {}).get("stocks")
            price_asof = F.twse_asof(d, "price")
            inst_asof = F.twse_asof(d, "inst") if net is not None else None
            margin_asof = F.twse_asof(d, "margin") if fin else None
            breadth_asof = F.twse_asof(d, "breadth") if up is not None else None
            missing = [k for k, v in (("inst", net), ("margin", fin or None),
                                      ("breadth", up)) if v is None]
            store.upsert_tw_daily({
                "date": d, "close": bar.close, "turnover": bar.turnover,
                "volume": bar.volume, "trades": bar.trades,
                "change_points": bar.change_points,
                "inst_total_net_ntd": net,
                "margin_balance_ntd": fin.get("today_balance_ntd"),
                "margin_balance_change_ntd": fin.get("balance_change_ntd"),
                "adv_count": up, "dec_count": down,
                "price_asof_taipei": price_asof, "inst_asof_taipei": inst_asof,
                "margin_asof_taipei": margin_asof, "breadth_asof_taipei": breadth_asof,
                "available_at_taipei": _max_asof(price_asof, inst_asof, margin_asof, breadth_asof),
                "missing_fields": ",".join(missing) if missing else None,
            })
            if sleep:
                time.sleep(sleep + random.uniform(0, 0.5))
        if not month_had_block:
            store.mark_ingested("tw_month", key)

def _month_end(month_start: date) -> date:
    nxt = date(month_start.year + 1, 1, 1) if month_start.month == 12 \
        else date(month_start.year, month_start.month + 1, 1)
    return nxt - timedelta(days=1)

def backfill_taifex(store, ns, months, sleep: float = 1.5, budget=None,
                    today: date = None) -> None:
    """Backfill TAIFEX TX net OI into EXISTING tw_daily rows (partial upsert).

    One date-range POST per month (resume key ingest_log("taifex_month",
    "YYYY-MM")). Only the tx_* / taifex_asof columns are written, so price/chip
    columns already in tw_daily are preserved. Rows for dates absent from
    tw_daily are still upserted (date + tx cols) so a later price backfill can
    fill the rest.

    The current month's query end is capped at `today`: TAIFEX rejects ranges
    containing future dates with an HTML error page. A fetch that raises (block
    / out-of-range / network) does NOT mark the month ingested, so a later run
    retries it instead of silently losing data.
    """
    if today is None:
        today = datetime.now(F.TAIPEI_TZ).date()
    for month_start in months:
        key = month_start.strftime("%Y-%m")
        if store.is_ingested("taifex_month", key):
            continue
        end = min(_month_end(month_start), today)
        if end < month_start:
            continue  # 整個月都在未來,尚無資料可抓
        # 當月最後幾日(今日尚未公布 / 週末假日)會讓 TAIFEX 整段回 HTML,
        # 逐日往前退到有公布資料的交易日為止(過去的完整月份第一次即命中)。
        taifex, last_exc = None, None
        attempt_end = end
        floor = max(month_start, end - timedelta(days=9))
        while attempt_end >= floor:
            _spend(budget)
            try:
                taifex = ns.fetch_taifex_inst_futures(
                    month_start.isoformat(), attempt_end.isoformat())
                break
            except BudgetExhausted:
                raise
            except Exception as exc:
                last_exc = exc
                attempt_end -= timedelta(days=1)
        if taifex is None:
            print(f"[backfill_taifex] fetch error {key}: {last_exc}", file=sys.stderr)
            continue  # 不標記 ingested → 下次重試
        for d, vals in taifex.items():
            store.upsert_tw_daily({
                "date": d,
                "tx_foreign_oi_net_lots": vals.get("tx_foreign_oi_net_lots"),
                "tx_inst_oi_net_lots": vals.get("tx_inst_oi_net_lots"),
                "taifex_asof_taipei": F.taifex_asof(d),
            })
        store.mark_ingested("taifex_month", key)
        if sleep:
            time.sleep(sleep + random.uniform(0, 0.5))

def backfill_us(store, start: date, end: date, sleep: float = 0.5,
                max_retries: int = 3) -> None:
    try:
        spy = {b.date: b for b in F.fetch_nasdaq_history("SPY", "etf", start, end)}
    except Exception as exc:
        print(f"[backfill_us] SPY fetch error: {exc}", file=sys.stderr)
        spy = {}
    try:
        qqq = {b.date: b for b in F.fetch_nasdaq_history("QQQ", "etf", start, end)}
    except Exception as exc:
        print(f"[backfill_us] QQQ fetch error: {exc}", file=sys.stderr)
        qqq = {}
    try:
        iwm = {b.date: b for b in F.fetch_nasdaq_history("IWM", "etf", start, end)}
    except Exception as exc:
        print(f"[backfill_us] IWM fetch error: {exc}", file=sys.stderr)
        iwm = {}
    try:
        comp = {b.date: b for b in F.fetch_nasdaq_history("COMP", "index", start, end)}
    except Exception as exc:
        print(f"[backfill_us] COMP fetch error: {exc}", file=sys.stderr)
        comp = {}
    try:
        vix = {b.date: b for b in F.fetch_vix_history(start, end)}
    except Exception as exc:
        print(f"[backfill_us] VIX fetch error: {exc}", file=sys.stderr)
        vix = {}
    for d in sorted(set(spy) | set(vix)):
        store.upsert_us_daily({
            "date": d,
            "spy_close": spy.get(d).close if d in spy else None,
            "spy_volume": spy.get(d).volume if d in spy else None,
            "qqq_close": qqq.get(d).close if d in qqq else None,
            "qqq_volume": qqq.get(d).volume if d in qqq else None,
            "iwm_close": iwm.get(d).close if d in iwm else None,
            "iwm_volume": iwm.get(d).volume if d in iwm else None,
            "comp_close": comp.get(d).close if d in comp else None,
            "vix_close": vix.get(d).close if d in vix else None,
            "price_asof_taipei": F.us_price_asof(d) if d in spy else None,
            "vix_asof_taipei": F.vix_asof(d) if d in vix else None,
            "available_at_taipei": _max_asof(
                F.us_price_asof(d) if d in spy else None,
                F.vix_asof(d) if d in vix else None,
            ),
        })

def backfill_finra(store) -> None:
    try:
        data = F.fetch_finra_margin_statistics()
    except Exception as exc:
        print(f"[backfill_finra] fetch error: {exc}", file=sys.stderr)
        return
    if not data:
        return
    for row in data.get("rows", []):
        # row["month"] 形如 "Jun-25";轉成 "YYYY-MM"
        try:
            dt = datetime.strptime(row["month"], "%b-%y")
        except (ValueError, KeyError):
            continue
        month = dt.strftime("%Y-%m")
        store.upsert_finra_monthly({
            "month": month,
            "debit_millions": row.get("debit_balances_millions"),
            "free_credit_cash_millions": row.get("free_credit_cash_millions"),
            "free_credit_margin_millions": row.get("free_credit_margin_millions"),
            "available_at_taipei": F.finra_asof(month),
            "quality_flags": "approx_asof",
        })

def backfill_valuation(store, ns, codes, dates, budget=None,
                       sleep: float = 1.5) -> None:
    """Backfill stock_valuation for all codes on each trading date in dates.

    One BWIBBU_d ALL-market request per date (ns.fetch_valuation_all) returns
    PE/PB/yield for every code at once — 1 request per trading day fills the
    per-stock PE history. Dates already marked ingested("valuation", date) are
    skipped. budget.spend() is called once per fetch; BudgetExhausted propagates
    to the caller for a clean stop (resume next run via ingest_log).
    """
    for date_iso in dates:
        if store.is_ingested("valuation", date_iso):
            continue
        _spend(budget)  # raises BudgetExhausted if cap exceeded
        result = ns.fetch_valuation_all(date_iso, codes)
        asof = F.twse_asof(date_iso, "price")
        for code in codes:
            vals = result.get(code)
            if vals is None:
                continue
            store.upsert_stock_valuation({
                "stock_no": code,
                "date": date_iso,
                "pe": vals.get("pe"),
                "pb": vals.get("pb"),
                "dividend_yield": vals.get("dividend_yield"),
                "asof_taipei": asof,
            })
        store.mark_ingested("valuation", date_iso)
        if sleep:
            time.sleep(sleep + random.uniform(0, 0.5))

def backfill_stock_day_month(store, ns, stock_no: str, month_start: date,
                             budget=None, sleep: float = 1.5) -> None:
    """Fetch one month of STOCK_DAY for stock_no and upsert into stock_daily.

    Resume key: ingest_log("stock_day", "stock_day:<stock_no>:<YYYY-MM>").
    budget.spend() is called once per fetch; BudgetExhausted propagates.
    """
    key = f"stock_day:{stock_no}:{month_start.strftime('%Y-%m')}"
    if store.is_ingested("stock_day", key):
        return
    _spend(budget)
    bars = ns.fetch_stock_day(stock_no, month_start)
    for bar in bars:
        price_asof = F.twse_asof(bar.date, "price")
        store.upsert_stock_daily({
            "stock_no": stock_no,
            "date": bar.date,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
            "price_asof_taipei": price_asof,
            "available_at_taipei": price_asof,
        })
    store.mark_ingested("stock_day", key)
    if sleep:
        time.sleep(sleep + random.uniform(0, 0.5))

def incremental_run(store, ns, start: date, budget: Budget,
                    sleep: float = 1.5, max_retries: int = 3,
                    backoff: float = 30.0) -> None:
    """Bounded incremental backfill driven by a priority queue.

    P1: Valuation history for TW50_CONSTITUENTS — most-recent missing trading
        day backward to start (trading days = existing tw_daily.date values).
    P2: Extend TW index chip history backward — tw_months between start and
        today not yet ingested (via backfill_tw, resume-aware, budget-counted).
    P3: Pre-warm per-stock STOCK_DAY for constituents (recent 6 months) into
        stock_daily.

    Stops cleanly at BudgetExhausted; remaining work resumes next run via
    ingest_log.
    """
    from marketlib.universe import TW50_CONSTITUENTS
    from marketlib.fetchers import month_starts as _month_starts

    today_d = datetime.now(F.TAIPEI_TZ).date()
    today_iso = today_d.isoformat()

    # Known trading dates from tw_daily, most-recent first.
    rows = store.conn.execute(
        "SELECT date FROM tw_daily WHERE date >= ? AND date <= ? ORDER BY date DESC",
        (start.isoformat(), today_iso),
    ).fetchall()
    trading_dates = [r[0] for r in rows]

    try:
        # --- P1: valuation history ---
        backfill_valuation(store, ns, TW50_CONSTITUENTS, trading_dates,
                           budget, sleep=sleep)

        # --- P2: extend TW index chip history backward ---
        ingested_months = {
            row[0] for row in store.conn.execute(
                "SELECT key FROM ingest_log WHERE source='tw_month' AND status='ok'"
            ).fetchall()
        }
        cur = date(start.year, start.month, 1)
        end_month = date(today_d.year, today_d.month, 1)
        missing_months = []
        while cur <= end_month:
            if cur.strftime("%Y-%m") not in ingested_months:
                missing_months.append(cur)
            cur = date(cur.year + 1, 1, 1) if cur.month == 12 \
                else date(cur.year, cur.month + 1, 1)
        backfill_tw(store, ns, missing_months, sleep=sleep,
                    max_retries=max_retries, backoff=backoff, budget=budget)

        # --- P2.5: TAIFEX TX futures OI history (~1 request/month) ---
        taifex_ingested = {
            row[0] for row in store.conn.execute(
                "SELECT key FROM ingest_log WHERE source='taifex_month' AND status='ok'"
            ).fetchall()
        }
        # TAIFEX futContractsDate 只服務近約 3 年的滾動視窗,再早只會回 HTML,
        # 故起點不早於 today−3 年,避免每次都白抓一批 pre-window 月份。
        taifex_start = max(date(start.year, start.month, 1),
                           date(today_d.year - 3, today_d.month, 1))
        cur = taifex_start
        taifex_months = []
        while cur <= end_month:
            if cur.strftime("%Y-%m") not in taifex_ingested:
                taifex_months.append(cur)
            cur = date(cur.year + 1, 1, 1) if cur.month == 12 \
                else date(cur.year, cur.month + 1, 1)
        backfill_taifex(store, ns, taifex_months, sleep=sleep, budget=budget)

        # --- P3: pre-warm per-stock STOCK_DAY (recent 6 months) ---
        recent_months = _month_starts(today_d, 6)
        for code in TW50_CONSTITUENTS:
            for month_start in recent_months:
                backfill_stock_day_month(store, ns, code, month_start,
                                         budget, sleep=sleep)
    except BudgetExhausted:
        return

def coverage(store, start: str) -> dict:
    """Return row counts and date ranges for all history tables — no network.

    start: ISO date string (e.g. "2014-01-01"); used to estimate pct_filled for
    tw_daily (approx trading days = calendar days × 252/365).
    """
    from datetime import date as date_cls

    def _range(table: str, col: str) -> dict:
        row = store.conn.execute(
            f"SELECT COUNT(*), MIN({col}), MAX({col}) FROM {table}"
        ).fetchone()
        return {"rows": row[0] or 0, f"min_{col}": row[1], f"max_{col}": row[2]}

    tw = _range("tw_daily", "date")
    try:
        start_d = date_cls.fromisoformat(start)
        total_calendar = max((date_cls.today() - start_d).days, 1)
        approx_trading_days = int(total_calendar * 252 / 365)
        tw["pct_filled"] = (
            round(tw["rows"] / approx_trading_days, 3)
            if approx_trading_days > 0 else None
        )
    except Exception:
        tw["pct_filled"] = None

    us = _range("us_daily", "date")
    finra = store.conn.execute(
        "SELECT COUNT(*), MIN(month), MAX(month) FROM finra_monthly"
    ).fetchone()
    sv = _range("stock_valuation", "date")
    sd = _range("stock_daily", "date")

    return {
        "tw_daily": tw,
        "us_daily": us,
        "finra_monthly": {
            "rows": finra[0] or 0,
            "min_month": finra[1],
            "max_month": finra[2],
        },
        "stock_valuation": sv,
        "stock_daily": sd,
    }

def _print_coverage(info: dict) -> None:
    """Print a human-readable coverage summary to stdout."""
    print("=== History Coverage Summary ===")
    for name, d in info.items():
        rows = d.get("rows", 0)
        min_k = [k for k in d if k.startswith("min_")]
        max_k = [k for k in d if k.startswith("max_")]
        min_v = d.get(min_k[0]) if min_k else None
        max_v = d.get(max_k[0]) if max_k else None
        pct = d.get("pct_filled")
        pct_str = f"  {pct:.1%} filled" if pct is not None else ""
        print(f"  {name:16s}: {rows:7d} rows  [{min_v or '—'} … {max_v or '—'}]{pct_str}")

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="data/history.db")
    p.add_argument("--start", default="2014-01-01")
    p.add_argument("--end", default=datetime.now(F.TAIPEI_TZ).date().isoformat())
    p.add_argument("--sources", default="tw,us,finra")
    p.add_argument("--sleep", type=float, default=1.5)
    p.add_argument("--max-retries", type=int, default=3, dest="max_retries")
    p.add_argument("--backoff", type=float, default=30.0)
    p.add_argument("--max-requests", type=int, default=500, dest="max_requests",
                   help="Per-run request budget (default 500); stop cleanly when exhausted.")
    p.add_argument("--coverage", action="store_true",
                   help="Print coverage summary without network access, then exit.")
    p.add_argument("--incremental", action="store_true",
                   help="Priority-queue incremental mode: valuation → TW months → stock_daily.")
    a = p.parse_args()
    store = HistoryStore(a.db); store.init_schema()
    start = datetime.strptime(a.start, "%Y-%m-%d").date()
    end = datetime.strptime(a.end, "%Y-%m-%d").date()

    if a.coverage:
        _print_coverage(coverage(store, a.start))
        store.close()
        return 0

    if a.incremental:
        import marketlib.stocks as stocks_ns
        ns = _MergedNamespace(stocks_ns, F)
        budget = Budget(max_requests=a.max_requests)
        try:
            incremental_run(store, ns, start, budget, sleep=a.sleep,
                            max_retries=a.max_retries, backoff=a.backoff)
        except BudgetExhausted:
            pass
        store.close()
        return 0

    sources = set(a.sources.split(","))
    if "tw" in sources:
        months = []
        cur = date(start.year, start.month, 1)
        while cur <= end:
            if not store.is_ingested("tw_month", cur.strftime("%Y-%m")):
                months.append(cur)
            cur = date(cur.year + 1, 1, 1) if cur.month == 12 else date(cur.year, cur.month + 1, 1)
        backfill_tw(store, F, months, sleep=a.sleep,
                    max_retries=a.max_retries, backoff=a.backoff)
    if "taifex" in sources:
        months = []
        cur = date(start.year, start.month, 1)
        while cur <= end:
            months.append(cur)
            cur = date(cur.year + 1, 1, 1) if cur.month == 12 else date(cur.year, cur.month + 1, 1)
        budget = Budget(max_requests=a.max_requests)
        try:
            backfill_taifex(store, F, months, sleep=a.sleep, budget=budget)
        except BudgetExhausted:
            pass
    if "us" in sources:
        backfill_us(store, start, end, sleep=a.sleep, max_retries=a.max_retries)
    if "finra" in sources:
        backfill_finra(store)
    store.close()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
