from __future__ import annotations
from datetime import datetime
from statistics import mean, pstdev
from zoneinfo import ZoneInfo
from typing import Any, Optional

TAIPEI_TZ = ZoneInfo("Asia/Taipei")
DECISION_HOUR, DECISION_MINUTE = 18, 30

def decision_time_for(date_iso: str) -> datetime:
    d = datetime.fromisoformat(date_iso)
    return d.replace(hour=DECISION_HOUR, minute=DECISION_MINUTE, second=0,
                     microsecond=0, tzinfo=TAIPEI_TZ)

def is_available(asof_iso: Optional[str], decision_time: datetime) -> bool:
    if not asof_iso:
        return False
    return datetime.fromisoformat(asof_iso) <= decision_time

def usable_value(row: dict, value_col: str, asof_col: str,
                 decision_time: datetime) -> Optional[float]:
    if not is_available(row.get(asof_col), decision_time):
        return None
    return row.get(value_col)

def zscore_excluding_t(series: list[float], n: int) -> Optional[float]:
    vals = [v for v in series if v is not None]
    if len(vals) < n + 1:
        return None
    baseline = vals[-n - 1:-1]
    sd = pstdev(baseline)
    if sd == 0:
        return None
    return (vals[-1] - mean(baseline)) / sd

def realized_vol(returns: list[float], n: int) -> Optional[float]:
    vals = [r for r in returns if r is not None]
    if len(vals) < n:
        return None
    return pstdev(vals[-n:])

def volume_ratio_excluding_t(volumes: list[float], n: int = 20) -> Optional[float]:
    vals = [v for v in volumes if v is not None]
    if len(vals) < n + 1:
        return None
    base = mean(vals[-n - 1:-1])
    if base == 0:
        return None
    return vals[-1] / base

FEATURE_NAMES_COMMON = [
    "ret_1d", "ret_5d", "ret_20d", "ret_60d",
    "close_vs_ma5", "close_vs_ma20", "close_vs_ma60", "ma20_vs_ma60",
    "vol_ratio_20d", "vol_ratio_5d", "drawdown_20d", "drawdown_60d",
    "realized_vol_20d", "realized_vol_60d",
]
FEATURE_NAMES_TW = FEATURE_NAMES_COMMON + [
    "margin_change_pct", "inst_net_z60", "breadth_ratio_z60",
    "tx_foreign_oi_z60", "tx_inst_oi_chg_z60",
    "us_spy_ret_1d", "us_qqq_ret_1d", "us_vix_level",
]
FEATURE_NAMES_US = FEATURE_NAMES_COMMON + [
    "vix_level", "vix_chg_1d", "vix_chg_5d", "qqq_vs_spy_20d", "iwm_vs_spy_20d",
]

def _ret(closes, n):
    if len(closes) <= n or closes[-1 - n] in (None, 0):
        return None
    return closes[-1] / closes[-1 - n] - 1

def _ma_dist(closes, n):
    if len(closes) < n:
        return None
    m = mean(closes[-n:])
    return closes[-1] / m - 1 if m else None

def _drawdown(closes, n):
    if len(closes) < n:
        return None
    hi = max(closes[-n:])
    return closes[-1] / hi - 1 if hi else None

def latest_usable_us(us_rows: list[dict], decision_time: datetime) -> Optional[dict]:
    usable = [r for r in us_rows if is_available(r.get("price_asof_taipei"), decision_time)]
    return usable[-1] if usable else None

def _close_col(market: str) -> str:
    return "close" if market == "tw" else "spy_close"

def _vol_col(market: str) -> str:
    return "volume" if market == "tw" else "spy_volume"

def build_feature_row(market, rows, idx, us_rows, decision_time):
    window = rows[: idx + 1]
    cc, vc = _close_col(market), _vol_col(market)
    closes = [r.get(cc) for r in window if r.get(cc) is not None]
    vols = [r.get(vc) for r in window if r.get(vc) is not None]
    rets = []
    for a, b in zip(closes, closes[1:]):
        rets.append(b / a - 1 if a else None)
    f: dict[str, Optional[float]] = {
        "ret_1d": _ret(closes, 1), "ret_5d": _ret(closes, 5),
        "ret_20d": _ret(closes, 20), "ret_60d": _ret(closes, 60),
        "close_vs_ma5": _ma_dist(closes, 5), "close_vs_ma20": _ma_dist(closes, 20),
        "close_vs_ma60": _ma_dist(closes, 60),
        "ma20_vs_ma60": (mean(closes[-20:]) / mean(closes[-60:]) - 1)
                        if len(closes) >= 60 else None,
        "vol_ratio_20d": volume_ratio_excluding_t(vols, 20),
        "vol_ratio_5d": volume_ratio_excluding_t(vols, 5),
        "drawdown_20d": _drawdown(closes, 20), "drawdown_60d": _drawdown(closes, 60),
        "realized_vol_20d": realized_vol(rets, 20),
        "realized_vol_60d": realized_vol(rets, 60),
    }
    if market == "tw":
        cur = rows[idx]
        bal = usable_value(cur, "margin_balance_ntd", "margin_asof_taipei", decision_time)
        chg = usable_value(cur, "margin_balance_change_ntd", "margin_asof_taipei", decision_time)
        f["margin_change_pct"] = (chg / (bal - chg)) if (bal and chg is not None and bal != chg) else None
        inst_series = [usable_value(r, "inst_total_net_ntd", "inst_asof_taipei",
                                    decision_time_for(r["date"])) for r in window]
        f["inst_net_z60"] = zscore_excluding_t(inst_series, 60)
        ratios = []
        for r in window:
            up = usable_value(r, "adv_count", "breadth_asof_taipei", decision_time_for(r["date"]))
            dn = usable_value(r, "dec_count", "breadth_asof_taipei", decision_time_for(r["date"]))
            ratios.append(up / dn if (up is not None and dn not in (None, 0)) else None)
        f["breadth_ratio_z60"] = zscore_excluding_t(ratios, 60)
        # TAIFEX 期貨籌碼:外資期貨未平倉淨口數(布局水位)與三大法人淨口數日變化
        # (當日籌碼動能)。每列都用自己的決策時點 + taifex asof 閘門防洩漏。
        foreign_oi = [usable_value(r, "tx_foreign_oi_net_lots", "taifex_asof_taipei",
                                   decision_time_for(r["date"])) for r in window]
        f["tx_foreign_oi_z60"] = zscore_excluding_t(foreign_oi, 60)
        inst_oi = [usable_value(r, "tx_inst_oi_net_lots", "taifex_asof_taipei",
                                decision_time_for(r["date"])) for r in window]
        inst_oi_chg = [b - a if (a is not None and b is not None) else None
                       for a, b in zip(inst_oi, inst_oi[1:])]
        f["tx_inst_oi_chg_z60"] = zscore_excluding_t(inst_oi_chg, 60)
        us = latest_usable_us(us_rows, decision_time)
        prev_us = latest_usable_us(
            [r for r in us_rows if us and r["date"] < us["date"]], decision_time) if us else None
        f["us_spy_ret_1d"] = (us["spy_close"] / prev_us["spy_close"] - 1) if (us and prev_us and prev_us.get("spy_close")) else None
        f["us_qqq_ret_1d"] = (us["qqq_close"] / prev_us["qqq_close"] - 1) if (us and prev_us and prev_us.get("qqq_close")) else None
        f["us_vix_level"] = us.get("vix_close") if us else None
    else:
        qqq = [r.get("qqq_close") for r in window if r.get("qqq_close") is not None]
        iwm = [r.get("iwm_close") for r in window if r.get("iwm_close") is not None]
        vix = [r.get("vix_close") for r in window if r.get("vix_close") is not None]
        f["vix_level"] = vix[-1] if vix else None
        f["vix_chg_1d"] = _ret(vix, 1)
        f["vix_chg_5d"] = _ret(vix, 5)
        spy20 = _ret(closes, 20)
        f["qqq_vs_spy_20d"] = (_ret(qqq, 20) - spy20) if (_ret(qqq, 20) is not None and spy20 is not None) else None
        f["iwm_vs_spy_20d"] = (_ret(iwm, 20) - spy20) if (_ret(iwm, 20) is not None and spy20 is not None) else None
    return f

def build_dataset(market, rows, us_rows):
    from marketlib.evaluation import neutral_threshold_from_bars, direction_from_return
    from marketlib.core import MarketBar
    cc = _close_col(market)
    out = []
    for idx in range(len(rows)):
        dt = decision_time_for(rows[idx]["date"])
        feats = build_feature_row(market, rows, idx, us_rows, dt)
        bars = [MarketBar(date=r["date"], close=r.get(cc) or 0.0) for r in rows[: idx + 1]]
        thr = neutral_threshold_from_bars(bars)
        if idx + 1 < len(rows) and rows[idx].get(cc) and rows[idx + 1].get(cc):
            nxt = rows[idx + 1][cc] / rows[idx][cc] - 1
            label = 1 if nxt > 0 else 0
            actual = direction_from_return(nxt, thr)
        else:
            nxt, label, actual = None, None, None
        out.append({"date": rows[idx]["date"], "features": feats, "label": label,
                    "next_return": nxt, "actual_direction": actual,
                    "neutral_threshold_pct": thr})
    return out
