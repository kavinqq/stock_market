from __future__ import annotations

import html
import math
import re
from dataclasses import dataclass
from statistics import mean
from typing import Any, Optional


@dataclass
class MarketBar:
    date: str
    close: float
    change_points: Optional[float] = None
    turnover: Optional[float] = None
    volume: Optional[float] = None
    trades: Optional[float] = None
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None


@dataclass
class IndicatorSet:
    close: float
    day_change_pct: Optional[float]
    five_day_pct: Optional[float]
    twenty_day_pct: Optional[float]
    sixty_day_pct: Optional[float]
    ma5: Optional[float]
    ma20: Optional[float]
    ma60: Optional[float]
    volume_ratio_20d: Optional[float]
    drawdown_from_20d_high: Optional[float]
    drawdown_from_60d_high: Optional[float]
    trend_label: str


def clean_num(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace(",", "").replace("$", "").replace("%", "").strip()
    if not text or text in {"--", "-"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def simple_moving_average(values: list[float], window: int) -> Optional[float]:
    if len(values) < window:
        return None
    return mean(values[-window:])


def pct_change(current: float, previous: Optional[float]) -> Optional[float]:
    if previous is None or previous == 0:
        return None
    return current / previous - 1


def compute_indicators(bars: list[MarketBar], volume_attr: str = "volume") -> IndicatorSet:
    if not bars:
        raise ValueError("no bars")
    closes = [bar.close for bar in bars]
    latest = bars[-1]
    close = latest.close

    def close_n_back(n: int) -> Optional[float]:
        if len(closes) <= n:
            return None
        return closes[-1 - n]

    ma5 = simple_moving_average(closes, 5)
    ma20 = simple_moving_average(closes, 20)
    ma60 = simple_moving_average(closes, 60)

    prior_vols = [getattr(bar, volume_attr) for bar in bars[:-1]]
    prior_vols = [float(v) for v in prior_vols if v is not None]
    latest_vol = getattr(latest, volume_attr)
    avg20_vol = mean(prior_vols[-20:]) if len(prior_vols) >= 20 else None
    volume_ratio = (
        float(latest_vol) / avg20_vol
        if latest_vol is not None and avg20_vol and avg20_vol > 0
        else None
    )

    high20 = max(closes[-20:]) if len(closes) >= 20 else None
    high60 = max(closes[-60:]) if len(closes) >= 60 else None

    if ma20 and ma60:
        if close >= ma20 and ma20 >= ma60:
            trend = "多頭延續"
        elif close >= ma20 and ma20 < ma60:
            trend = "短線反彈"
        elif close < ma20 and close >= ma60:
            trend = "多頭修正"
        else:
            trend = "偏空修正"
    elif ma20:
        trend = "站上20日線" if close >= ma20 else "跌破20日線"
    else:
        trend = "資料不足"

    return IndicatorSet(
        close=close,
        day_change_pct=pct_change(close, close_n_back(1)),
        five_day_pct=pct_change(close, close_n_back(5)),
        twenty_day_pct=pct_change(close, close_n_back(20)),
        sixty_day_pct=pct_change(close, close_n_back(60)),
        ma5=ma5,
        ma20=ma20,
        ma60=ma60,
        volume_ratio_20d=volume_ratio,
        drawdown_from_20d_high=pct_change(close, high20),
        drawdown_from_60d_high=pct_change(close, high60),
        trend_label=trend,
    )
