#!/usr/bin/env python3
"""Offline multi-year history backfill into SQLite (point-in-time aware)."""
from __future__ import annotations
import argparse, random, sys, time
from datetime import date, datetime
from marketlib import fetchers as F
from marketlib.store import HistoryStore

def _max_asof(*vals):
    present = [v for v in vals if v]
    return max(present) if present else None

def _is_block_error(exc: BaseException) -> bool:
    """Return True when exc text indicates a TWSE anti-scraping block."""
    msg = str(exc)
    return any(kw in msg for kw in ("無法呈現", "CAN NOT BE ACCESSED", "SECURITY REASONS"))

def backfill_tw(store, ns, months, sleep: float = 1.5,
                max_retries: int = 3, backoff: float = 30.0) -> None:
    for month_start in months:
        key = month_start.strftime("%Y-%m")
        month_had_block = False
        bars = ns.fetch_twse_market_month(month_start)
        for bar in bars:
            d = bar.date

            # --- fetch_twse_institutional with block-retry ---
            inst = {}
            for attempt in range(max_retries + 1):
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

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="data/history.db")
    p.add_argument("--start", default="2014-01-01")
    p.add_argument("--end", default=datetime.now(F.TAIPEI_TZ).date().isoformat())
    p.add_argument("--sources", default="tw,us,finra")
    p.add_argument("--sleep", type=float, default=1.5)
    p.add_argument("--max-retries", type=int, default=3, dest="max_retries")
    p.add_argument("--backoff", type=float, default=30.0)
    a = p.parse_args()
    store = HistoryStore(a.db); store.init_schema()
    start = datetime.strptime(a.start, "%Y-%m-%d").date()
    end = datetime.strptime(a.end, "%Y-%m-%d").date()
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
    if "us" in sources:
        backfill_us(store, start, end, sleep=a.sleep, max_retries=a.max_retries)
    if "finra" in sources:
        backfill_finra(store)
    store.close()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
