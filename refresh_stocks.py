#!/usr/bin/env python3
"""Rate-limit-respecting refresh of TW50 daily bars.

Split out of the daily report because the two have different failure modes. The
report must finish; this must be gentle. TWSE starts refusing after a few
hundred rapid STOCK_DAY calls, and the screener's inline fetch had no budget, no
backoff and no resume — so a blocked run left bars silently stale and the
freshness check (a 20-calendar-day tolerance) waved them through.

Resume is via the store's ingest_log, so re-running only fetches what is still
missing. Run it repeatedly rather than raising --max-requests.

    python refresh_stocks.py --months 24 --max-requests 200
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from datetime import date, datetime

import backfill as B
from marketlib import fetchers as F, stocks as S
from marketlib.store import HistoryStore
from marketlib.universe import TW50_CONSTITUENTS


class _Namespace:
    """Attribute lookup across the fetch modules, as backfill expects."""

    def __getattr__(self, name):
        for mod in (S, F):
            if hasattr(mod, name):
                return getattr(mod, name)
        raise AttributeError(name)


def refresh(db: str, months: int, max_requests: int, sleep: float,
            codes: list[str], refresh_recent: int = 2) -> int:
    store = HistoryStore(db)
    ns = _Namespace()
    end = date.today()
    month_list = list(S.month_starts(end, months))

    # The most recent months are re-fetched every run: they are the ones that
    # change, and a resume marker would otherwise freeze them forever.
    for m in month_list[-refresh_recent:]:
        for c in codes:
            store.conn.execute(
                "DELETE FROM ingest_log WHERE source=? AND key=?",
                ("stock_day", f"stock_day:{c}:{m.strftime('%Y-%m')}"))
    store.conn.commit()

    # Month-major, newest first — NOT stock-major. Walking one stock through all
    # 24 months before touching the next means an interrupted run leaves a few
    # perfect stocks and the rest unusable. Sweeping the newest month across
    # every stock first makes all of them current within ~100 requests, and each
    # further pass adds depth uniformly, so stopping anywhere leaves a coherent
    # dataset rather than a ragged one.
    spent = blocked = consecutive_fail = 0
    t0 = time.time()
    for m in reversed(month_list):
        for code in codes:
            if spent >= max_requests:
                print(f"\n預算用盡（{max_requests} 次請求）。再執行一次即可從此處續傳。")
                store.close()
                return _summary(db, codes, spent, blocked, t0)
            key = f"stock_day:{code}:{m.strftime('%Y-%m')}"
            if store.is_ingested("stock_day", key):
                continue
            try:
                bars = ns.fetch_stock_day(code, m)
                spent += 1
                if not bars:
                    # An empty month is NEVER recorded as done. The obvious
                    # shortcut — treat "older than this stock's earliest bar" as
                    # pre-listing — is circular: those bars are missing precisely
                    # because we never fetched them, so it froze ten TW50
                    # large-caps at 136 bars by concluding 富邦金 was not listed
                    # before March 2026. Every constituent here has years of
                    # history, so an empty month is a transient TWSE response and
                    # is retried on the next run; the request budget and the
                    # consecutive-failure guard are what bound the retrying.
                    blocked += 1
                    consecutive_fail += 1
                    print(f"  ?? {code} {m:%Y-%m}: 回傳空值，不標記完成（下輪重試）",
                          flush=True)
                    continue
                consecutive_fail = 0
                for bar in bars:
                    asof = F.twse_asof(bar.date, "price")
                    store.upsert_stock_daily({
                        "stock_no": code, "date": bar.date, "open": bar.open,
                        "high": bar.high, "low": bar.low, "close": bar.close,
                        "volume": bar.volume, "price_asof_taipei": asof,
                        "available_at_taipei": asof,
                    })
                store.mark_ingested("stock_day", key)
            except Exception as exc:  # noqa: BLE001
                spent += 1
                blocked += 1
                consecutive_fail += 1
                print(f"  !! {code} {m:%Y-%m}: {type(exc).__name__}", flush=True)
                # CONSECUTIVE, not cumulative: scattered empty months across a
                # 1200-request sweep are normal (a stock not yet listed, a quiet
                # month), and counting them cumulatively stopped the run a few
                # minutes in. A run of failures back-to-back is what actually
                # signals a block.
                if consecutive_fail >= 8:
                    print("\n連續失敗過多，TWSE 可能已限流。停止本輪，稍後再執行續傳。")
                    store.close()
                    return _summary(db, codes, spent, blocked, t0)
            time.sleep(sleep + random.uniform(0, 0.4))
    store.close()
    return _summary(db, codes, spent, blocked, t0)


def _summary(db: str, codes: list[str], spent: int, blocked: int, t0: float) -> int:
    store = HistoryStore(db)
    expected = S.last_completed_trading_day(date.today()).isoformat()
    current = sum(1 for c in codes
                  if (r := store.get_stock_daily(c)) and r[-1]["date"] >= expected)
    deep = sum(1 for c in codes
               if len(store.get_stock_daily(c)) >= S.MIN_BARS_FOR_WEEKLY_TREND)
    store.close()
    print(f"\n請求 {spent} 次（失敗 {blocked}），耗時 {time.time() - t0:.0f}s")
    print(f"{current}/{len(codes)} 檔已更新到 {expected}；"
          f"{deep}/{len(codes)} 檔資料深度足以判讀週線趨勢")
    return 0 if current == len(codes) and deep == len(codes) else 2


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="data/history.db")
    p.add_argument("--months", type=int, default=24,
                   help="回補月數。週 MACD 的 26 期 EMA 要收斂，需遠多於 5 個月。")
    p.add_argument("--max-requests", type=int, default=200, dest="max_requests",
                   help="單輪請求上限。TWSE 會限流，寧可多跑幾輪。")
    p.add_argument("--sleep", type=float, default=1.2)
    p.add_argument("--codes", default="", help="逗號分隔；預設為 0050 成分股全部")
    a = p.parse_args()
    codes = [c.strip() for c in a.codes.split(",") if c.strip()] or TW50_CONSTITUENTS
    return refresh(a.db, a.months, a.max_requests, a.sleep, codes)


if __name__ == "__main__":
    raise SystemExit(main())
