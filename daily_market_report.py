#!/usr/bin/env python3
"""
Daily US/Taiwan broad-market report.

This script is intentionally deterministic and dependency-free. It fetches
public market data, computes rule-based indicators, and writes Markdown plus
JSON outputs. It does not call any LLM or paid API.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import shutil
import sys
import time
from dataclasses import asdict
from marketlib.core import (
    MarketBar, IndicatorSet, clean_num, pct_change,
    simple_moving_average, compute_indicators,
)
from marketlib.evaluation import (
    DIRECTION_LABELS, DEFAULT_NEUTRAL_THRESHOLD,
    direction_from_return, neutral_threshold_from_bars,
    summarize_classification, summarize_baselines,
)
from marketlib.fetchers import (
    TAIPEI_TZ,
    fetch_twse_market_history,
    fetch_twse_margin,
    fetch_twse_institutional,
    fetch_twse_summary,
    fetch_twse_sector_indices,
    fetch_nasdaq_history,
    fetch_vix_history,
    fetch_finra_margin_statistics,
    twse_asof,
    us_price_asof,
    vix_asof,
)
from marketlib import model as ml_model
from marketlib.features import build_feature_row, decision_time_for, FEATURE_NAMES_TW, FEATURE_NAMES_US
from marketlib.store import HistoryStore
from marketlib import stocks as _stocks
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Any, Optional, Union


BASE_DIR = Path(__file__).resolve().parent



def fmt_num(value: Optional[Union[float, int]], decimals: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    return f"{value:,.{decimals}f}"


def fmt_pct(value: Optional[float], decimals: int = 2, signed: bool = True) -> str:
    if value is None:
        return "N/A"
    sign = "+" if signed and value > 0 else ""
    return f"{sign}{value * 100:.{decimals}f}%"


def fmt_big_twd(value_ntd: Optional[float]) -> str:
    if value_ntd is None:
        return "N/A"
    if abs(value_ntd) >= 1_000_000_000_000:
        return f"{value_ntd / 1_000_000_000_000:,.2f} 兆"
    return f"{value_ntd / 100_000_000:,.1f} 億"


def fmt_big_usd_millions(value_millions: Optional[float]) -> str:
    if value_millions is None:
        return "N/A"
    if abs(value_millions) >= 1_000_000:
        return f"{value_millions / 1_000_000:,.2f} 兆美元"
    return f"{value_millions / 1_000:,.1f} 十億美元"


def classify_volume_price(day_pct: Optional[float], volume_ratio: Optional[float]) -> str:
    if volume_ratio is None:
        return "量能資料不足"
    direction = "上漲" if day_pct is not None and day_pct > 0 else "下跌"
    if volume_ratio >= 1.35:
        return f"放量{direction}"
    if volume_ratio <= 0.75:
        return f"量縮{direction}"
    return f"量能正常{direction}"


def leverage_taiwan_reading(margin: Optional[dict[str, Any]], day_pct: Optional[float]) -> str:
    if not margin or "margin_financing" not in margin:
        return "融資資料不足"
    change = margin["margin_financing"].get("balance_change_ntd")
    if change is None:
        return "融資變化資料不足"
    if abs(change) < 3_000_000_000:
        return "融資餘額變化不大，籌碼槓桿沒有明顯方向"
    if change > 0 and day_pct is not None and day_pct < 0:
        return "下跌但融資增加，散戶槓桿偏逆勢承接，短線壓力未完全釋放"
    if change < 0 and day_pct is not None and day_pct < 0:
        return "下跌且融資減少，代表去槓桿正在發生，急跌後風險有部分釋放"
    if change > 0:
        return "融資增加，風險偏好升溫；若價格未續強，要注意追價槓桿"
    return "融資減少，籌碼轉保守"


def leverage_us_reading(finra: Optional[dict[str, Any]]) -> str:
    if not finra:
        return "FINRA margin debt 資料不足"
    latest = finra["latest"]
    mom = latest.get("mom_debit_pct")
    yoy = latest.get("yoy_debit_pct")
    ratio = latest.get("debit_to_free_credit_ratio")
    if mom is not None and mom >= 0.06:
        return (
            "美股融資餘額月增明顯，槓桿偏熱；"
            f"debit/free credit 約 {fmt_num(ratio, 2)} 倍，回檔時波動容易放大"
        )
    if yoy is not None and yoy >= 0.20:
        return "美股融資餘額年增偏高，風險偏好已明顯升溫"
    if mom is not None and mom < -0.03:
        return "美股融資餘額月減，槓桿風險降溫"
    return "美股融資餘額沒有極端變化，槓桿面中性偏高"


def vix_reading(vix_indicators: Optional[IndicatorSet]) -> str:
    if not vix_indicators:
        return "VIX 資料不足"
    close = vix_indicators.close
    if close >= 25:
        level = "避險情緒偏高"
    elif close >= 20:
        level = "風險意識升溫"
    elif close <= 15:
        level = "波動預期偏低"
    else:
        level = "波動預期中性"
    return f"VIX {close:.2f}，{level}，日變動 {fmt_pct(vix_indicators.day_change_pct)}"


def build_taiwan_report(end: date, lookback_months: int) -> dict[str, Any]:
    bars = fetch_twse_market_history(end, lookback_months)
    if not bars:
        raise RuntimeError("no TWSE market data")
    latest_date = bars[-1].date
    indicators = compute_indicators(bars, volume_attr="turnover")
    margin = fetch_twse_margin(latest_date)
    institutional = fetch_twse_institutional(latest_date)
    summary = fetch_twse_summary(latest_date)
    sectors = fetch_twse_sector_indices(latest_date)
    return {
        "latest_date": latest_date,
        "bars": [asdict(bar) for bar in bars],
        "indicators": asdict(indicators),
        "margin": margin,
        "institutional": institutional,
        "summary": summary,
        "sectors": sectors,
        "volume_price_label": classify_volume_price(
            indicators.day_change_pct,
            indicators.volume_ratio_20d,
        ),
        "leverage_label": leverage_taiwan_reading(margin, indicators.day_change_pct),
    }


def build_us_report(end: date, lookback_days: int) -> dict[str, Any]:
    start = end - timedelta(days=lookback_days)
    assets = {
        "SPY": ("SPY", "etf"),
        "QQQ": ("QQQ", "etf"),
        "IWM": ("IWM", "etf"),
        "COMP": ("COMP", "index"),
    }
    series: dict[str, Any] = {}
    for label, (symbol, assetclass) in assets.items():
        bars = fetch_nasdaq_history(symbol, assetclass, start, end)
        series[label] = {
            "bars": [asdict(bar) for bar in bars],
            "indicators": asdict(compute_indicators(bars, volume_attr="volume")),
        }

    vix_bars = fetch_vix_history(start, end)
    vix_indicators = compute_indicators(vix_bars) if vix_bars else None
    finra = fetch_finra_margin_statistics()
    spy_indicators = IndicatorSet(**series["SPY"]["indicators"])

    return {
        "latest_date": series["SPY"]["bars"][-1]["date"],
        "series": series,
        "vix": {
            "bars": [asdict(bar) for bar in vix_bars],
            "indicators": asdict(vix_indicators) if vix_indicators else None,
            "label": vix_reading(vix_indicators),
        },
        "finra_margin": finra,
        "volume_price_label": classify_volume_price(
            spy_indicators.day_change_pct,
            spy_indicators.volume_ratio_20d,
        ),
        "leverage_label": leverage_us_reading(finra),
    }


def twse_net_by_name(institutional: Optional[dict[str, Any]], name: str) -> Optional[float]:
    if not institutional:
        return None
    row = institutional.get("rows", {}).get(name)
    return row.get("net_ntd") if row else None


def summarize_twse(tw: dict[str, Any]) -> list[str]:
    ind = IndicatorSet(**tw["indicators"])
    margin = tw.get("margin") or {}
    financing = (margin.get("margin_financing") or {})
    institutional = tw.get("institutional")
    summary = tw.get("summary") or {}
    breadth = summary.get("breadth") or {}
    adv_decl = breadth.get("advance_decline_ratio_stocks")
    foreign_net = twse_net_by_name(institutional, "外資及陸資(不含外資自營商)")
    trust_net = twse_net_by_name(institutional, "投信")
    total_net = twse_net_by_name(institutional, "合計")

    lines = [
        (
            f"加權指數 {fmt_num(ind.close, 2)}，日變動 {fmt_pct(ind.day_change_pct)}，"
            f"5日 {fmt_pct(ind.five_day_pct)}，20日 {fmt_pct(ind.twenty_day_pct)}；"
            f"技術狀態：{ind.trend_label}。"
        ),
        (
            f"量能：{tw['volume_price_label']}，成交金額 / 前20日均量比 "
            f"{fmt_num(ind.volume_ratio_20d, 2)}；20日高點回落 "
            f"{fmt_pct(ind.drawdown_from_20d_high, signed=True)}。"
        ),
    ]
    if adv_decl is not None:
        lines.append(f"廣度：上市股票漲跌家數比約 {fmt_num(adv_decl, 2)}，低於 1 代表下跌家數較多。")
    if financing:
        lines.append(
            "融資：餘額 "
            f"{fmt_big_twd(financing.get('today_balance_ntd'))}，"
            f"單日變化 {fmt_big_twd(financing.get('balance_change_ntd'))}；"
            f"{tw['leverage_label']}。"
        )
    if institutional:
        lines.append(
            "法人：合計 "
            f"{fmt_big_twd(total_net)}，外資 {fmt_big_twd(foreign_net)}，"
            f"投信 {fmt_big_twd(trust_net)}。"
        )
    return lines


def summarize_us(us: dict[str, Any]) -> list[str]:
    spy = IndicatorSet(**us["series"]["SPY"]["indicators"])
    qqq = IndicatorSet(**us["series"]["QQQ"]["indicators"])
    iwm = IndicatorSet(**us["series"]["IWM"]["indicators"])
    finra = us.get("finra_margin")
    lines = [
        (
            f"SPY {fmt_num(spy.close, 2)}，日變動 {fmt_pct(spy.day_change_pct)}，"
            f"5日 {fmt_pct(spy.five_day_pct)}，20日 {fmt_pct(spy.twenty_day_pct)}；"
            f"技術狀態：{spy.trend_label}。"
        ),
        (
            f"QQQ 日變動 {fmt_pct(qqq.day_change_pct)} / 20日 {fmt_pct(qqq.twenty_day_pct)}，"
            f"IWM 日變動 {fmt_pct(iwm.day_change_pct)} / 20日 {fmt_pct(iwm.twenty_day_pct)}；"
            "用來觀察大型科技與小型股是否同向。"
        ),
        (
            f"量能：SPY {us['volume_price_label']}，成交量 / 前20日均量比 "
            f"{fmt_num(spy.volume_ratio_20d, 2)}；VIX：{us['vix']['label']}。"
        ),
        f"槓桿：{us['leverage_label']}。",
    ]
    if finra:
        latest = finra["latest"]
        lines.append(
            "FINRA margin debt："
            f"{latest['month']} debit balance {fmt_big_usd_millions(latest.get('debit_balances_millions'))}，"
            f"月變動 {fmt_pct(latest.get('mom_debit_pct'))}，"
            f"年變動 {fmt_pct(latest.get('yoy_debit_pct'))}。"
        )
    return lines


def infer_cross_market(tw: dict[str, Any], us: dict[str, Any]) -> list[str]:
    tw_ind = IndicatorSet(**tw["indicators"])
    spy = IndicatorSet(**us["series"]["SPY"]["indicators"])
    qqq = IndicatorSet(**us["series"]["QQQ"]["indicators"])
    vix_data = us.get("vix", {}).get("indicators")
    vix_close = vix_data.get("close") if vix_data else None

    lines: list[str] = []
    if qqq.twenty_day_pct is not None and tw_ind.twenty_day_pct is not None:
        if qqq.twenty_day_pct < 0 and tw_ind.twenty_day_pct < 0:
            lines.append("台股與 QQQ 同步轉弱，半導體/AI 權值股的共振風險較高。")
        elif qqq.twenty_day_pct > 0 and tw_ind.twenty_day_pct < 0:
            lines.append("QQQ 仍強但台股轉弱，台股自身籌碼或匯率/法人賣壓可能是主因。")
        elif qqq.twenty_day_pct < 0 and tw_ind.twenty_day_pct > 0:
            lines.append("美國科技股偏弱但台股仍強，台股有相對強勢，但隔夜美股風險要盯。")
        else:
            lines.append("台股與美國科技股 20 日趨勢同步偏強，順勢格局仍在。")

    if spy.volume_ratio_20d and tw_ind.volume_ratio_20d:
        if spy.volume_ratio_20d > 1.25 and tw_ind.volume_ratio_20d > 1.25:
            lines.append("兩邊都放量，代表不是單一市場噪音，短線要優先看支撐是否守住。")
        elif tw_ind.volume_ratio_20d > 1.35 and spy.volume_ratio_20d <= 1.1:
            lines.append("台股放量程度高於美股，短線較像台股本地籌碼壓力。")

    if vix_close is not None and vix_close >= 20:
        lines.append("VIX 高於 20 時，隔夜消息會放大台股開盤跳空風險。")
    return lines or ["跨市場訊號分歧不大，先用各自 20 日線與成交量作風控。"]


def support_resistance_from_bars(bars: list[dict[str, Any]]) -> dict[str, Optional[float]]:
    closes = [bar["close"] for bar in bars if bar.get("close") is not None]
    if not closes:
        return {"support_20d": None, "resistance_20d": None}
    recent = closes[-20:] if len(closes) >= 20 else closes
    return {"support_20d": min(recent), "resistance_20d": max(recent)}


def next_weekday(iso_date: str) -> str:
    current = datetime.strptime(iso_date, "%Y-%m-%d").date()
    nxt = current + timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    return nxt.isoformat()


def score_to_prediction(score: float) -> str:
    if score >= 1.0:
        return "偏多"
    if score <= -1.0:
        return "偏空"
    return "震盪"


def score_to_confidence(score: float, missing_count: int = 0) -> str:
    adjusted = abs(score) - 0.25 * missing_count
    if adjusted >= 2.4:
        return "高"
    if adjusted >= 1.2:
        return "中"
    return "低"


def trend_score(ind: IndicatorSet) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    if ind.ma20 is not None and ind.close < ind.ma20:
        score -= 0.7
        reasons.append("收盤跌破20日線，短線動能偏弱")
    elif ind.ma20 is not None and ind.close >= ind.ma20:
        score += 0.5
        reasons.append("收盤站上20日線，短線動能偏強")
    if ind.ma20 is not None and ind.ma60 is not None:
        if ind.ma20 >= ind.ma60:
            score += 0.5
            reasons.append("20日線高於60日線，中期趨勢仍有支撐")
        else:
            score -= 0.5
            reasons.append("20日線低於60日線，中期趨勢偏弱")
    if ind.five_day_pct is not None:
        if ind.five_day_pct > 0.015:
            score += 0.35
            reasons.append("5日報酬為正，短線慣性偏多")
        elif ind.five_day_pct < -0.015:
            score -= 0.35
            reasons.append("5日報酬為負，短線慣性偏空")
    if ind.drawdown_from_20d_high is not None and ind.drawdown_from_20d_high < -0.05:
        score -= 0.35
        reasons.append("自20日高點回落超過5%，回檔壓力仍在")
    return score, reasons


def volume_score(ind: IndicatorSet) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    if ind.volume_ratio_20d is None or ind.day_change_pct is None:
        return score, ["量能或日漲跌資料不足"]
    if ind.volume_ratio_20d >= 1.25 and ind.day_change_pct < 0:
        score -= 0.8
        reasons.append("下跌且量能高於前20日均量，供給壓力偏大")
    elif ind.volume_ratio_20d >= 1.25 and ind.day_change_pct > 0:
        score += 0.7
        reasons.append("上漲且量能高於前20日均量，追價動能偏強")
    elif ind.volume_ratio_20d <= 0.75 and ind.day_change_pct < 0:
        score += 0.2
        reasons.append("下跌量縮，賣壓尚未擴大")
    elif ind.volume_ratio_20d <= 0.75 and ind.day_change_pct > 0:
        score -= 0.2
        reasons.append("上漲量縮，追價力道不足")
    return score, reasons


def prediction_range(score: float) -> dict[str, float]:
    width = min(0.028, 0.010 + abs(score) * 0.004)
    center = max(-0.012, min(0.012, score * 0.003))
    return {
        "low_pct": round(center - width, 4),
        "high_pct": round(center + width, 4),
    }


def build_taiwan_prediction(tw: dict[str, Any], us: dict[str, Any]) -> dict[str, Any]:
    ind = IndicatorSet(**tw["indicators"])
    neutral_threshold = neutral_threshold_from_bars(tw["bars"])
    score, reasons = trend_score(ind)
    vol_score, vol_reasons = volume_score(ind)
    score += vol_score
    reasons.extend(vol_reasons)
    missing: list[str] = []

    margin = tw.get("margin") or {}
    financing = margin.get("margin_financing") or {}
    margin_change = financing.get("balance_change_ntd")
    if margin_change is None:
        missing.append("台股融資餘額變化")
    elif margin_change > 5_000_000_000 and (ind.day_change_pct or 0) < 0:
        score -= 0.55
        reasons.append("下跌但融資增加，槓桿承接容易形成隔日賣壓")
    elif margin_change < -5_000_000_000 and (ind.day_change_pct or 0) < 0:
        score += 0.25
        reasons.append("下跌且融資減少，已有部分去槓桿")

    institutional = tw.get("institutional")
    total_net = twse_net_by_name(institutional, "合計")
    latest_turnover = (tw["bars"][-1] or {}).get("turnover")
    if total_net is None or latest_turnover in {None, 0}:
        missing.append("三大法人買賣超占成交金額")
    else:
        inst_ratio = total_net / latest_turnover
        if inst_ratio <= -0.05:
            score -= 0.7
            reasons.append("三大法人賣超占成交金額超過5%，籌碼偏空")
        elif inst_ratio >= 0.03:
            score += 0.45
            reasons.append("三大法人買超占成交金額超過3%，籌碼偏多")

    breadth = ((tw.get("summary") or {}).get("breadth") or {})
    adv_decl = breadth.get("advance_decline_ratio_stocks")
    if adv_decl is None:
        missing.append("上市漲跌家數比")
    elif adv_decl < 0.5:
        score -= 0.45
        reasons.append("漲跌家數比低於0.5，市場廣度偏弱")
    elif adv_decl > 1.5:
        score += 0.35
        reasons.append("漲跌家數比高於1.5，市場廣度偏強")

    qqq = IndicatorSet(**us["series"]["QQQ"]["indicators"])
    vix_data = us.get("vix", {}).get("indicators")
    vix_close = vix_data.get("close") if vix_data else None
    if qqq.day_change_pct is not None and qqq.day_change_pct < -0.01:
        score -= 0.35
        reasons.append("QQQ 單日跌幅超過1%，台股科技權值股隔日承壓")
    if vix_close is not None and vix_close >= 20:
        score -= 0.35
        reasons.append("VIX 高於20，隔夜風險升溫")

    bias = score_to_prediction(score)
    return {
        "market": "台股",
        "target_date_estimate": next_weekday(tw["latest_date"]),
        "target": "下一個交易日收盤方向",
        "bias": bias,
        "score": round(score, 2),
        "confidence": score_to_confidence(score, len(missing)),
        "expected_range_pct": prediction_range(score),
        "neutral_threshold_pct": neutral_threshold,
        "decision_score_threshold": 1.0,
        "formula_version": "rule-v1.1",
        "reasons": reasons[:8],
        "missing_data": missing + ["未接交易所假日曆，target_date 只用週末推估"],
        "training_note": "預測只用規則式訊號，尚未納入消息面、匯率、期貨夜盤與估值資料。",
    }


def build_us_prediction(us: dict[str, Any]) -> dict[str, Any]:
    spy = IndicatorSet(**us["series"]["SPY"]["indicators"])
    qqq = IndicatorSet(**us["series"]["QQQ"]["indicators"])
    iwm = IndicatorSet(**us["series"]["IWM"]["indicators"])
    neutral_threshold = neutral_threshold_from_bars(us["series"]["SPY"]["bars"])
    score, reasons = trend_score(spy)
    vol_score, vol_reasons = volume_score(spy)
    score += vol_score
    reasons.extend(vol_reasons)
    missing: list[str] = []

    if qqq.twenty_day_pct is not None:
        if qqq.twenty_day_pct < -0.03:
            score -= 0.45
            reasons.append("QQQ 20日跌幅超過3%，大型科技股動能偏弱")
        elif qqq.twenty_day_pct > 0.03:
            score += 0.35
            reasons.append("QQQ 20日漲幅超過3%，大型科技股動能偏強")
    if iwm.twenty_day_pct is not None and spy.twenty_day_pct is not None:
        if iwm.twenty_day_pct > spy.twenty_day_pct:
            score += 0.2
            reasons.append("IWM 相對 SPY 強，小型股風險偏好尚可")
        else:
            score -= 0.15
            reasons.append("IWM 相對 SPY 弱，市場風險偏好分歧")

    vix_data = us.get("vix", {}).get("indicators")
    if not vix_data:
        missing.append("VIX")
    else:
        vix_close = vix_data.get("close")
        vix_day = vix_data.get("day_change_pct")
        if vix_close is not None and vix_close >= 20:
            score -= 0.55
            reasons.append("VIX 高於20，短線波動風險偏高")
        elif vix_close is not None and vix_close <= 15:
            score += 0.25
            reasons.append("VIX 低於15，波動預期偏低")
        if vix_day is not None and vix_day > 0.08:
            score -= 0.35
            reasons.append("VIX 單日上升超過8%，避險需求升溫")

    finra = us.get("finra_margin")
    if not finra:
        missing.append("FINRA margin debt")
    else:
        mom = finra["latest"].get("mom_debit_pct")
        if mom is not None and mom >= 0.06 and (spy.day_change_pct or 0) < 0:
            score -= 0.25
            reasons.append("FINRA 融資餘額月增偏快，回檔時槓桿波動可能放大")

    bias = score_to_prediction(score)
    return {
        "market": "美股",
        "target_date_estimate": next_weekday(us["latest_date"]),
        "target": "下一個交易日收盤方向",
        "bias": bias,
        "score": round(score, 2),
        "confidence": score_to_confidence(score, len(missing)),
        "expected_range_pct": prediction_range(score),
        "neutral_threshold_pct": neutral_threshold,
        "decision_score_threshold": 1.0,
        "formula_version": "rule-v1.1",
        "reasons": reasons[:8],
        "missing_data": missing + ["未接 NYSE/Nasdaq 假日曆，target_date 只用週末推估"],
        "training_note": "預測只用規則式訊號，尚未納入盤前期貨、利率、美元與重大新聞。",
    }


def build_predictions(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "taiwan": build_taiwan_prediction(report["taiwan"], report["us"]),
        "us": build_us_prediction(report["us"]),
    }


def model_prediction(market: str, model_path: str, rows: list, us_rows: list) -> Optional[dict]:
    import os
    if not os.path.exists(model_path) or not rows:
        return None
    try:
        model = ml_model.load_model(model_path)
    except Exception:
        return None
    names = FEATURE_NAMES_TW if market == "tw" else FEATURE_NAMES_US
    dt = decision_time_for(rows[-1]["date"])
    feats = build_feature_row(market, rows, len(rows) - 1, us_rows, dt)
    # Check sparsity against the full canonical feature list for the market.
    # If the model uses a subset of features, we still fall back when the data
    # pipeline itself is too sparse (guards against a broken store feed).
    # Use the larger of the model's own feature list and the canonical list for
    # the threshold, so minimal test models do not trigger a spurious fallback.
    model_names = model.get("feature_names", names)
    check_names = model_names if len(model_names) < len(names) else names
    present = sum(1 for n in check_names if feats.get(n) is not None)
    if present < len(check_names) // 2:   # 特徵大量缺 -> 退回規則式
        return None
    p = ml_model.predict_proba(model, feats)
    return {"bias": ml_model.proba_to_direction(model, p),
            "probability": round(p, 4),
            "confidence": ml_model.proba_to_confidence(model, p),
            "model_version": model.get("trained_at", "unknown")}


def _max_asof(*vals: Optional[str]) -> Optional[str]:
    present = [v for v in vals if v]
    return max(present) if present else None


def upsert_report_into_store(report: dict[str, Any], store: HistoryStore) -> None:
    """Map the report's latest-day data into store rows and upsert."""
    # --- Taiwan tw_daily row ---
    tw = report.get("taiwan", {})
    tw_date = tw.get("latest_date")
    if tw_date:
        bars = tw.get("bars", [])
        latest_bar = bars[-1] if bars else {}

        margin = tw.get("margin") or {}
        financing = margin.get("margin_financing") or {}
        institutional = tw.get("institutional") or {}
        summary = tw.get("summary") or {}
        breadth = summary.get("breadth") or {}

        net = (institutional.get("rows", {}).get("合計") or {}).get("net_ntd")
        up = (breadth.get("上漲") or {}).get("stocks")
        down = (breadth.get("下跌") or {}).get("stocks")

        price_asof = twse_asof(tw_date, "price")
        inst_asof = twse_asof(tw_date, "inst") if net is not None else None
        margin_asof = twse_asof(tw_date, "margin") if financing else None
        breadth_asof = twse_asof(tw_date, "breadth") if up is not None else None

        missing = [k for k, v in (("inst", net), ("margin", financing or None),
                                  ("breadth", up)) if v is None]

        store.upsert_tw_daily({
            "date": tw_date,
            "close": latest_bar.get("close"),
            "turnover": latest_bar.get("turnover"),
            "volume": latest_bar.get("volume"),
            "inst_total_net_ntd": net,
            "margin_balance_ntd": financing.get("today_balance_ntd"),
            "margin_balance_change_ntd": financing.get("balance_change_ntd"),
            "adv_count": up,
            "dec_count": down,
            "price_asof_taipei": price_asof,
            "inst_asof_taipei": inst_asof,
            "margin_asof_taipei": margin_asof,
            "breadth_asof_taipei": breadth_asof,
            "available_at_taipei": _max_asof(price_asof, inst_asof, margin_asof, breadth_asof),
            "missing_fields": ",".join(missing) if missing else None,
        })

    # --- US us_daily rows ---
    us = report.get("us", {})
    us_series = us.get("series", {})
    vix_bars = (us.get("vix") or {}).get("bars") or []
    vix_by_date = {b["date"]: b for b in vix_bars}

    # Collect all dates from all US series
    us_dates: set[str] = set()
    for asset_data in us_series.values():
        for bar in asset_data.get("bars", []):
            if bar.get("date"):
                us_dates.add(bar["date"])

    # Index bars by date per asset
    spy_by_date = {b["date"]: b for b in (us_series.get("SPY") or {}).get("bars", [])}
    qqq_by_date = {b["date"]: b for b in (us_series.get("QQQ") or {}).get("bars", [])}
    iwm_by_date = {b["date"]: b for b in (us_series.get("IWM") or {}).get("bars", [])}
    comp_by_date = {b["date"]: b for b in (us_series.get("COMP") or {}).get("bars", [])}

    for d in sorted(us_dates):
        spy_bar = spy_by_date.get(d)
        qqq_bar = qqq_by_date.get(d)
        iwm_bar = iwm_by_date.get(d)
        comp_bar = comp_by_date.get(d)
        vix_bar = vix_by_date.get(d)

        price_asof = us_price_asof(d) if spy_bar else None
        vix_asof_val = vix_asof(d) if vix_bar else None

        store.upsert_us_daily({
            "date": d,
            "spy_close": spy_bar.get("close") if spy_bar else None,
            "spy_volume": spy_bar.get("volume") if spy_bar else None,
            "qqq_close": qqq_bar.get("close") if qqq_bar else None,
            "qqq_volume": qqq_bar.get("volume") if qqq_bar else None,
            "iwm_close": iwm_bar.get("close") if iwm_bar else None,
            "iwm_volume": iwm_bar.get("volume") if iwm_bar else None,
            "comp_close": comp_bar.get("close") if comp_bar else None,
            "vix_close": vix_bar.get("close") if vix_bar else None,
            "price_asof_taipei": price_asof,
            "vix_asof_taipei": vix_asof_val,
            "available_at_taipei": _max_asof(price_asof, vix_asof_val),
        })


def backtest_signal_from_window(bars: list[dict[str, Any]], idx: int, volume_attr: str) -> dict[str, Any]:
    source_window = bars[: idx + 1]
    window = [
        MarketBar(
            date=bar["date"],
            close=bar["close"],
            turnover=bar.get("turnover"),
            volume=bar.get("volume"),
        )
        for bar in source_window
    ]
    ind = compute_indicators(window, volume_attr=volume_attr)
    neutral_threshold = neutral_threshold_from_bars(source_window)
    score, reasons = trend_score(ind)
    vol_score, vol_reasons = volume_score(ind)
    score += vol_score
    reasons.extend(vol_reasons)
    prediction = score_to_prediction(score)
    actual_ret = pct_change(bars[idx + 1]["close"], bars[idx]["close"])
    actual_direction = direction_from_return(actual_ret, neutral_threshold)
    momentum_ret = pct_change(bars[idx]["close"], bars[idx - 1]["close"]) if idx > 0 else None
    momentum_prediction = direction_from_return(momentum_ret, neutral_threshold)
    baselines = {
        "always_bull": {
            "label": "永遠偏多",
            "prediction": "偏多",
            "hit": actual_direction == "偏多",
        },
        "momentum": {
            "label": "今日方向延續",
            "prediction": momentum_prediction,
            "hit": momentum_prediction == actual_direction,
        },
    }
    hit = prediction == actual_direction
    return {
        "source_date": bars[idx]["date"],
        "target_date": bars[idx + 1]["date"],
        "prediction": prediction,
        "actual_direction": actual_direction,
        "actual_return_pct": actual_ret,
        "neutral_threshold_pct": neutral_threshold,
        "hit": hit,
        "baselines": baselines,
        "score": round(score, 2),
        "miss_reason": None if hit else classify_miss_reason(prediction, actual_direction, reasons),
        "reasons": reasons[:5],
    }


def classify_miss_reason(prediction: str, actual_direction: str, reasons: list[str]) -> str:
    if prediction == "震盪" and actual_direction in {"偏多", "偏空"}:
        return "公式門檻過保守：預測震盪但實際出現單邊走勢"
    if prediction in {"偏多", "偏空"} and actual_direction == "震盪":
        return "公式權重過強：訊號分數過度放大但實際盤整"
    if any("量能" in reason or "供給" in reason for reason in reasons):
        return "公式錯誤：量價權重未捕捉隔日反轉"
    return "缺少資料源或公式不足：未納入消息面、期貨、利率/匯率或籌碼歷史"


def summarize_backtest(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    hits = sum(1 for row in records if row.get("hit"))
    misses = [row for row in records if not row.get("hit")]
    return {
        "sample_size": total,
        "hit_count": hits,
        "miss_count": len(misses),
        "hit_rate": hits / total if total else None,
        "classification_metrics": summarize_classification(records),
        "baselines": summarize_baselines(records),
        "recent_misses": misses[-8:],
    }


def rolling_backtest(bars: list[dict[str, Any]], volume_attr: str, sample_size: int = 40) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    if len(bars) < 65:
        return {
            "sample_size": 0,
            "hit_count": 0,
            "miss_count": 0,
            "hit_rate": None,
            "recent_misses": [],
            "limitations": ["歷史資料不足，至少需要約65個交易日"],
        }
    start = max(60, len(bars) - sample_size - 1)
    for idx in range(start, len(bars) - 1):
        records.append(backtest_signal_from_window(bars, idx, volume_attr))
    summary = summarize_backtest(records)
    summary["records"] = records
    summary["limitations"] = [
        "價量診斷回測不是正式模型完整回測；未回填歷史融資、法人、VIX 或 FINRA 月資料",
        "報告會同時列出永遠偏多、今日方向延續、依實際先驗隨機等 baseline",
        "正確率定義為下一交易日收盤方向：偏多 / 偏空 / 震盪",
        "震盪門檻為近60日收盤日報酬波動的一半，並限制在 0.3% 到 0.9%",
    ]
    return summary


def build_accuracy_report(report: dict[str, Any], data_dir: Path) -> dict[str, Any]:
    taiwan_bt = rolling_backtest(report["taiwan"]["bars"], "turnover")
    us_bt = rolling_backtest(report["us"]["series"]["SPY"]["bars"], "volume")
    archive_eval = evaluate_archived_predictions(report, data_dir)
    return {
        "formula_version": "rule-v1.1",
        "taiwan": taiwan_bt,
        "us": us_bt,
        "archived_predictions": archive_eval,
        "improvement_log": build_improvement_log(taiwan_bt, us_bt, archive_eval),
    }


def market_bars_by_date(report: dict[str, Any], market_key: str) -> dict[str, dict[str, Any]]:
    if market_key == "taiwan":
        bars = report["taiwan"]["bars"]
    else:
        bars = report["us"]["series"]["SPY"]["bars"]
    return {bar["date"]: bar for bar in bars}


def evaluate_archived_predictions(report: dict[str, Any], data_dir: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    if not data_dir.exists():
        return {"sample_size": 0, "hit_rate": None, "records": [], "limitations": ["尚無歷史 JSON 報告"]}

    current_maps = {
        "taiwan": market_bars_by_date(report, "taiwan"),
        "us": market_bars_by_date(report, "us"),
    }
    current_dates = set(current_maps["taiwan"]) | set(current_maps["us"])
    for path in sorted(data_dir.glob("*.json")):
        if path.name == "latest.json":
            continue
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        predictions = old.get("predictions")
        if not predictions:
            continue
        for key in ("taiwan", "us"):
            pred = predictions.get(key)
            if not pred:
                continue
            source_date = old.get(key, {}).get("latest_date") if key in old else None
            target_date = pred.get("target_date_estimate")
            if not source_date or not target_date:
                continue
            bars_map = current_maps[key]
            if source_date not in bars_map:
                continue
            adjusted_target = False
            if target_date not in bars_map:
                future_dates = sorted(day for day in bars_map if day > source_date)
                if not future_dates:
                    continue
                target_date = future_dates[0]
                adjusted_target = True
            actual_ret = pct_change(bars_map[target_date]["close"], bars_map[source_date]["close"])
            neutral_threshold = pred.get("neutral_threshold_pct")
            threshold_source = "prediction_snapshot"
            if neutral_threshold is None:
                neutral_threshold = DEFAULT_NEUTRAL_THRESHOLD
                threshold_source = "default_fallback"
            actual_direction = direction_from_return(actual_ret, neutral_threshold)
            hit = pred.get("bias") == actual_direction
            records.append(
                {
                    "report_date": old.get("report_date"),
                    "market": pred.get("market"),
                    "source_date": source_date,
                    "target_date": target_date,
                    "prediction": pred.get("bias"),
                    "actual_direction": actual_direction,
                    "actual_return_pct": actual_ret,
                    "neutral_threshold_pct": neutral_threshold,
                    "threshold_source": threshold_source,
                    "hit": hit,
                    "miss_reason": None if hit else classify_miss_reason(pred.get("bias"), actual_direction, pred.get("reasons", [])),
                    "formula_version": pred.get("formula_version"),
                    "prediction_score": pred.get("score"),
                    "confidence": pred.get("confidence"),
                    "target_date_adjusted": adjusted_target,
                    "missing_data": pred.get("missing_data", []),
                    "reasons": pred.get("reasons", []),
                }
            )
    summary = summarize_backtest(records)
    summary["records"] = records[-30:]
    summary["limitations"] = [
        "只有從 rule-v1.1 開始產出的 JSON 才含可驗證預測",
        "若 target_date 遇到假日，需等實際資料日出現後才會被納入驗證",
    ]
    return summary


def build_improvement_log(
    taiwan_bt: dict[str, Any],
    us_bt: dict[str, Any],
    archive_eval: dict[str, Any],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for market_name, bt in (("台股", taiwan_bt), ("美股", us_bt)):
        for miss in bt.get("recent_misses", [])[-5:]:
            rows.append(
                {
                    "market": market_name,
                    "date": miss["target_date"],
                    "reason": miss["miss_reason"],
                    "action": "保留供後續調整公式權重或補資料源",
                }
            )
    for miss in [row for row in archive_eval.get("records", []) if not row.get("hit")][-5:]:
        rows.append(
            {
                "market": miss.get("market", "歷史預測"),
                "date": miss.get("target_date", "N/A"),
                "reason": miss.get("miss_reason", "N/A"),
                "action": "保留供後續 AI 分析與訓練",
            }
        )
    return rows[-12:]


def evaluation_record_key(row: dict[str, Any]) -> str:
    return "|".join(
        str(row.get(part, ""))
        for part in ("report_date", "market", "source_date", "target_date", "formula_version")
    )


def update_evaluation_history(report: dict[str, Any], data_dir: Path) -> Path:
    path = data_dir / "evaluation_history.json"
    current_records = report.get("accuracy", {}).get("archived_predictions", {}).get("records", [])
    existing_records: list[dict[str, Any]] = []
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            existing_records = existing.get("records", [])
        except Exception:
            existing_records = []

    merged: dict[str, dict[str, Any]] = {
        evaluation_record_key(row): row for row in existing_records
    }
    for row in current_records:
        merged[evaluation_record_key(row)] = row

    records = sorted(
        merged.values(),
        key=lambda row: (
            str(row.get("target_date", "")),
            str(row.get("market", "")),
            str(row.get("report_date", "")),
        ),
    )
    hits = sum(1 for row in records if row.get("hit"))
    summary = summarize_backtest(records)
    payload = {
        "updated_at": report.get("generated_at"),
        "formula_version": report.get("accuracy", {}).get("formula_version"),
        "sample_size": len(records),
        "hit_count": hits,
        "miss_count": len(records) - hits,
        "hit_rate": hits / len(records) if records else None,
        "classification_metrics": summary.get("classification_metrics"),
        "baselines": summary.get("baselines"),
        "records": records,
        "notes": [
            "這裡保存每日正式預測和後續實際收盤方向的對照",
            "miss_reason 用來區分公式權重問題、門檻問題、缺少資料源或假日 target 調整",
            "actual_direction 使用當時預測 JSON 保存的 neutral_threshold_pct；舊資料缺值時才用預設 0.5%",
            "價量診斷回測留在 daily JSON；正式每日預測驗證累積在此檔案",
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def accuracy_baseline_rate(acc: dict[str, Any], key: str, field: str = "hit_rate") -> Optional[float]:
    return ((acc.get("baselines") or {}).get(key) or {}).get(field)


def accuracy_baseline_note(acc: dict[str, Any]) -> str:
    return (
        f"永遠偏多 {fmt_pct(accuracy_baseline_rate(acc, 'always_bull'), signed=False)} / "
        f"動能 {fmt_pct(accuracy_baseline_rate(acc, 'momentum'), signed=False)} / "
        f"隨機先驗 {fmt_pct(accuracy_baseline_rate(acc, 'random_prior', 'expected_hit_rate'), signed=False)}"
    )


def accuracy_macro_f1(acc: dict[str, Any]) -> Optional[float]:
    return (acc.get("classification_metrics") or {}).get("macro_f1")


def fmt_neutral_threshold(value: Optional[float]) -> str:
    return fmt_pct(value, decimals=2, signed=False)


def _facet_tone(score: Optional[float]) -> str:
    if score is None:
        return "muted"
    if score >= 0.25:
        return "good"
    if score <= -0.25:
        return "bad"
    return "warn"


def _lean_tone(bias: str) -> str:
    if bias == "偏多":
        return "good"
    if bias == "偏空":
        return "bad"
    return "warn"


def html_stock_2330_section(stock: Optional[dict[str, Any]]) -> str:
    """Render the 台積電 2330 HTML section."""
    if not stock or "error" in stock:
        err_msg = (stock or {}).get("error", "資料未取得")
        return (
            '<section id="stock2330" class="page">'
            '<div class="hero"><h1>台積電 2330</h1></div>'
            f'<div class="panel"><h2>資料不足</h2><p class="subtle">{esc(err_msg)}</p></div>'
            '</section>'
        )

    facets = stock.get("facets") or {}
    short_lean = stock.get("short_lean") or {}
    medium_lean = stock.get("medium_lean") or {}
    missing = stock.get("missing") or []
    levels = stock.get("observe_levels") or {}

    # Facet cards
    facet_names = {
        "technical": "技術",
        "volume": "量能",
        "chip": "籌碼",
        "valuation": "估值",
        "revenue": "營收",
        "peer": "同業",
        "market": "大盤",
    }
    facet_cards_html = ""
    for key, label in facet_names.items():
        f = facets.get(key, {})
        score = f.get("score")
        reason = f.get("reason", "資料不足")
        score_str = f"{score:.2f}" if score is not None else "N/A"
        tone = _facet_tone(score)
        facet_cards_html += metric_card(label, score_str, reason, tone)

    # Lean cards
    lean_cards_html = (
        metric_card("短線方向", short_lean.get("bias", "N/A"),
                    f"信心 {short_lean.get('confidence', 'N/A')} / 分數 {fmt_num(short_lean.get('score'), 2)}",
                    _lean_tone(short_lean.get("bias", "")))
        + metric_card("中線方向", medium_lean.get("bias", "N/A"),
                      f"信心 {medium_lean.get('confidence', 'N/A')} / 分數 {fmt_num(medium_lean.get('score'), 2)}",
                      _lean_tone(medium_lean.get("bias", "")))
    )

    missing_note = f"缺少資料：{', '.join(missing)}" if missing else "各構面資料完整"

    return (
        '<section id="stock2330" class="page">'
        '<div class="hero">'
        '<h1>台積電 2330</h1>'
        '<div class="subtle">多構面評分：技術/量能/籌碼/估值/營收/同業/大盤，短線+中線方向。</div>'
        '</div>'
        '<h3>構面評分</h3>'
        f'<div class="grid">{facet_cards_html}</div>'
        '<h3>方向研判</h3>'
        f'<div class="grid">{lean_cards_html}</div>'
        '<div class="panel">'
        '<h2>觀察價位</h2>'
        + bullet_list([
            f"現價 {fmt_num(levels.get('close'), 2)}",
            f"20日區間支撐 {fmt_num(levels.get('support_20d'), 2)}，壓力 {fmt_num(levels.get('resistance_20d'), 2)}",
            f"MA5 {fmt_num(levels.get('ma5'), 2)}，MA20 {fmt_num(levels.get('ma20'), 2)}，MA60 {fmt_num(levels.get('ma60'), 2)}",
        ]) +
        '</div>'
        f'<p class="subtle">{esc(missing_note)}</p>'
        '</section>'
    )


def render_markdown(report: dict[str, Any]) -> str:
    generated = report["generated_at"]
    tw = report["taiwan"]
    us = report["us"]
    tw_ind = IndicatorSet(**tw["indicators"])
    spy = IndicatorSet(**us["series"]["SPY"]["indicators"])
    qqq = IndicatorSet(**us["series"]["QQQ"]["indicators"])
    predictions = report.get("predictions", {})
    tw_pred = predictions.get("taiwan", {})
    us_pred = predictions.get("us", {})
    accuracy = report.get("accuracy", {})
    tw_acc = accuracy.get("taiwan", {})
    us_acc = accuracy.get("us", {})
    archived_acc = accuracy.get("archived_predictions", {})

    tw_levels = support_resistance_from_bars(tw["bars"])
    spy_levels = support_resistance_from_bars(us["series"]["SPY"]["bars"])
    qqq_levels = support_resistance_from_bars(us["series"]["QQQ"]["bars"])

    lines = [
        f"# 台美股大盤日報 - {report['report_date']}",
        "",
        f"產生時間：{generated}",
        "",
        "## 資料日期",
        "",
        f"- 台股：{tw['latest_date']}（TWSE）",
        f"- 美股：{us['latest_date']}（Nasdaq API / Cboe VIX）",
        "- 美股融資：FINRA 月資料，通常次月第三週更新",
        "",
        "## 總結",
        "",
        f"- 台股：{tw_ind.trend_label}，{tw['volume_price_label']}，{tw['leverage_label']}。",
        f"- 美股：SPY {spy.trend_label}，QQQ {qqq.trend_label}，{us['volume_price_label']}，{us['vix']['label']}。",
        f"- 回推：{infer_cross_market(tw, us)[0]}",
        "",
        "## 台股大盤",
        "",
    ]
    lines.extend(f"- {line}" for line in summarize_twse(tw))

    sectors = (tw.get("sectors") or {}).get("rows") or {}
    if sectors:
        lines.extend(["", "### 台股族群"])
        for name, row in sectors.items():
            lines.append(
                f"- {name}：{fmt_num(row.get('close'), 2)}，"
                f"{fmt_pct(row.get('change_pct'))}"
            )

    lines.extend(
        [
            "",
            "## 美股大盤",
            "",
        ]
    )
    lines.extend(f"- {line}" for line in summarize_us(us))

    lines.extend(
        [
            "",
            "## 下一交易日走勢預測",
            "",
            (
                f"- 台股：{tw_pred.get('target_date_estimate', 'N/A')} "
                f"{tw_pred.get('bias', 'N/A')}，信心 {tw_pred.get('confidence', 'N/A')}，"
                f"分數 {tw_pred.get('score', 'N/A')}，預估區間 "
                f"{fmt_pct((tw_pred.get('expected_range_pct') or {}).get('low_pct'))} ~ "
                f"{fmt_pct((tw_pred.get('expected_range_pct') or {}).get('high_pct'))}；"
                f"實際震盪帶 {fmt_neutral_threshold(tw_pred.get('neutral_threshold_pct'))}。"
            ),
            (
                f"- 美股：{us_pred.get('target_date_estimate', 'N/A')} "
                f"{us_pred.get('bias', 'N/A')}，信心 {us_pred.get('confidence', 'N/A')}，"
                f"分數 {us_pred.get('score', 'N/A')}，預估區間 "
                f"{fmt_pct((us_pred.get('expected_range_pct') or {}).get('low_pct'))} ~ "
                f"{fmt_pct((us_pred.get('expected_range_pct') or {}).get('high_pct'))}；"
                f"實際震盪帶 {fmt_neutral_threshold(us_pred.get('neutral_threshold_pct'))}。"
            ),
            "- 台股理由：" + "；".join(tw_pred.get("reasons", [])[:4]),
            "- 美股理由：" + "；".join(us_pred.get("reasons", [])[:4]),
            "- 預測限制：" + "；".join((tw_pred.get("missing_data", []) + us_pred.get("missing_data", []))[:5]),
            "",
            "## 準確率追蹤",
            "",
            (
                f"- 台股價量診斷回測：樣本 {tw_acc.get('sample_size', 0)}，"
                f"模型 {fmt_pct(tw_acc.get('hit_rate'), signed=False)}，"
                f"{accuracy_baseline_note(tw_acc)}，"
                f"macro-F1 {fmt_pct(accuracy_macro_f1(tw_acc), signed=False)}。"
            ),
            (
                f"- 美股價量診斷回測：樣本 {us_acc.get('sample_size', 0)}，"
                f"模型 {fmt_pct(us_acc.get('hit_rate'), signed=False)}，"
                f"{accuracy_baseline_note(us_acc)}，"
                f"macro-F1 {fmt_pct(accuracy_macro_f1(us_acc), signed=False)}。"
            ),
            (
                f"- 正式每日預測驗證：樣本 {archived_acc.get('sample_size', 0)}，"
                f"命中率 {fmt_pct(archived_acc.get('hit_rate'), signed=False)}。"
            ),
        ]
    )
    misses = accuracy.get("improvement_log", [])
    if misses:
        lines.extend(["", "### 失準紀錄 / 待調校"])
        for row in misses:
            lines.append(
                f"- {row.get('market')} {row.get('date')}：{row.get('reason')}；{row.get('action')}"
            )

    lines.extend(["", "## 回推邏輯", ""])
    lines.extend(f"- {line}" for line in infer_cross_market(tw, us))
    lines.extend(
        [
            "- 價格跌破 20 日線且放量，優先視為供給壓力；若隔日縮量止跌，才改看整理。",
            "- 融資增加搭配下跌，通常代表槓桿尚未洗乾淨；融資減少搭配下跌，則是去槓桿正在發生。",
            "- 美股 margin debt 是月資料，不適合判斷單日轉折，適合判斷槓桿背景溫度。",
            "",
            "## 觀察價位",
            "",
            (
                f"- 台股：20日區間支撐 {fmt_num(tw_levels['support_20d'], 2)}，"
                f"20日區間壓力 {fmt_num(tw_levels['resistance_20d'], 2)}，"
                f"20日線 {fmt_num(tw_ind.ma20, 2)}，60日線 {fmt_num(tw_ind.ma60, 2)}。"
            ),
            (
                f"- SPY：20日區間支撐 {fmt_num(spy_levels['support_20d'], 2)}，"
                f"20日區間壓力 {fmt_num(spy_levels['resistance_20d'], 2)}，"
                f"20日線 {fmt_num(spy.ma20, 2)}，60日線 {fmt_num(spy.ma60, 2)}。"
            ),
            (
                f"- QQQ：20日區間支撐 {fmt_num(qqq_levels['support_20d'], 2)}，"
                f"20日區間壓力 {fmt_num(qqq_levels['resistance_20d'], 2)}，"
                f"20日線 {fmt_num(qqq.ma20, 2)}，60日線 {fmt_num(qqq.ma60, 2)}。"
            ),
            "",
            "## 資料源",
            "",
            "- TWSE FMTQIK：台股每日成交資訊",
            "- TWSE MI_MARGN：上市信用交易統計",
            "- TWSE BFI82U：三大法人買賣金額統計",
            "- Nasdaq historical API：SPY / QQQ / IWM / COMP 價量資料",
            "- Cboe VIX History：VIX 日資料",
            "- FINRA Margin Statistics：美股 margin debt 月資料",
            "",
            "註：這是規則式市場分析，不是投資建議。",
        ]
    )

    # --- 台積電 2330 section ---
    stock = report.get("stock_2330")
    lines.extend(["", "## 台積電 2330", ""])
    if not stock or "error" in stock:
        err_msg = (stock or {}).get("error", "資料未取得")
        lines.append(f"- 資料不足：{err_msg}")
    else:
        facets = stock.get("facets") or {}
        short_lean = stock.get("short_lean") or {}
        medium_lean = stock.get("medium_lean") or {}
        missing = stock.get("missing") or []
        levels = stock.get("observe_levels") or {}

        lines.append(
            f"- 短線：{short_lean.get('bias', 'N/A')}，信心 {short_lean.get('confidence', 'N/A')}，"
            f"分數 {fmt_num(short_lean.get('score'), 2)}"
        )
        lines.append(
            f"- 中線：{medium_lean.get('bias', 'N/A')}，信心 {medium_lean.get('confidence', 'N/A')}，"
            f"分數 {fmt_num(medium_lean.get('score'), 2)}"
        )
        facet_names = {"technical": "技術", "volume": "量能", "chip": "籌碼",
                       "valuation": "估值", "revenue": "營收", "peer": "同業", "market": "大盤"}
        for key, label in facet_names.items():
            f = facets.get(key, {})
            score = f.get("score")
            reason = f.get("reason", "資料不足")
            score_str = f"{score:.2f}" if score is not None else "N/A"
            lines.append(f"- {label}：{score_str}，{reason}")
        lines.extend([
            "",
            "### 觀察價位",
            "",
            f"- 現價 {fmt_num(levels.get('close'), 2)}",
            f"- 20日區間支撐 {fmt_num(levels.get('support_20d'), 2)}，壓力 {fmt_num(levels.get('resistance_20d'), 2)}",
            f"- MA5 {fmt_num(levels.get('ma5'), 2)}，MA20 {fmt_num(levels.get('ma20'), 2)}，MA60 {fmt_num(levels.get('ma60'), 2)}",
        ])
        if missing:
            lines.append(f"- 缺少資料：{', '.join(missing)}")

    return "\n".join(lines) + "\n"


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def metric_card(label: str, value: str, note: str, tone: str = "neutral") -> str:
    return (
        f'<article class="metric metric-{esc(tone)}">'
        f'<div class="metric-label">{esc(label)}</div>'
        f'<div class="metric-value">{esc(value)}</div>'
        f'<div class="metric-note">{esc(note)}</div>'
        "</article>"
    )


def bullet_list(items: list[str]) -> str:
    return "<ul>" + "".join(f"<li>{esc(item)}</li>" for item in items) + "</ul>"


def unique_strings(items: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


def prediction_tone(prediction: str) -> str:
    if prediction == "偏多":
        return "good"
    if prediction == "偏空":
        return "bad"
    return "warn"


def fmt_prediction_range(pred: dict[str, Any]) -> str:
    expected = pred.get("expected_range_pct") or {}
    return f"{fmt_pct(expected.get('low_pct'))} ~ {fmt_pct(expected.get('high_pct'))}"


def html_prediction_block(pred: dict[str, Any]) -> str:
    return (
        f'<div class="panel pred-{esc(prediction_tone(pred.get("bias", "")))}">'
        f'<h2>{esc(pred.get("market", "N/A"))}</h2>'
        '<div class="grid">'
        + metric_card("目標交易日", str(pred.get("target_date_estimate", "N/A")), pred.get("target", ""), "neutral")
        + metric_card("方向", str(pred.get("bias", "N/A")), f"信心 {pred.get('confidence', 'N/A')}", prediction_tone(pred.get("bias", "")))
        + metric_card("分數", str(pred.get("score", "N/A")), f"公式 {pred.get('formula_version', 'N/A')}", "warn")
        + metric_card("預估區間", fmt_prediction_range(pred), "下一交易日收盤報酬", "neutral")
        + metric_card("震盪判定", fmt_neutral_threshold(pred.get("neutral_threshold_pct")), "實際驗證使用的中性帶", "muted")
        + "</div>"
        "<h3>理由</h3>"
        + bullet_list(pred.get("reasons", []) or ["理由資料不足"])
        + "<h3>缺資料 / 限制</h3>"
        + bullet_list(pred.get("missing_data", []) or ["未記錄缺資料"])
        + f'<p class="subtle">{esc(pred.get("training_note", ""))}</p>'
        + "</div>"
    )


def html_miss_rows(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<tr><td colspan="5">目前沒有可顯示的失準紀錄</td></tr>'
    rendered = []
    for row in rows:
        rendered.append(
            "<tr>"
            f"<td>{esc(row.get('market', 'N/A'))}</td>"
            f"<td>{esc(row.get('date') or row.get('target_date') or 'N/A')}</td>"
            f"<td>{esc(row.get('prediction', 'N/A'))}</td>"
            f"<td>{esc(row.get('actual_direction', 'N/A'))}</td>"
            f"<td>{esc(row.get('reason') or row.get('miss_reason') or 'N/A')}</td>"
            "</tr>"
        )
    return "".join(rendered)


def render_html(report: dict[str, Any]) -> str:
    tw = report["taiwan"]
    us = report["us"]
    tw_ind = IndicatorSet(**tw["indicators"])
    spy = IndicatorSet(**us["series"]["SPY"]["indicators"])
    qqq = IndicatorSet(**us["series"]["QQQ"]["indicators"])
    iwm = IndicatorSet(**us["series"]["IWM"]["indicators"])
    tw_levels = support_resistance_from_bars(tw["bars"])
    spy_levels = support_resistance_from_bars(us["series"]["SPY"]["bars"])
    qqq_levels = support_resistance_from_bars(us["series"]["QQQ"]["bars"])
    financing = ((tw.get("margin") or {}).get("margin_financing") or {})
    institutional = tw.get("institutional")
    foreign_net = twse_net_by_name(institutional, "外資及陸資(不含外資自營商)")
    trust_net = twse_net_by_name(institutional, "投信")
    total_net = twse_net_by_name(institutional, "合計")
    finra_latest = ((us.get("finra_margin") or {}).get("latest") or {})
    cross_lines = infer_cross_market(tw, us)
    predictions = report.get("predictions", {})
    tw_pred = predictions.get("taiwan", {})
    us_pred = predictions.get("us", {})
    accuracy = report.get("accuracy", {})
    tw_acc = accuracy.get("taiwan", {})
    us_acc = accuracy.get("us", {})
    archived_acc = accuracy.get("archived_predictions", {})
    improvement_log = accuracy.get("improvement_log", [])
    stock_2330_section = html_stock_2330_section(report.get("stock_2330"))

    sectors = (tw.get("sectors") or {}).get("rows") or {}
    sector_rows = "".join(
        "<tr>"
        f"<td>{esc(name)}</td>"
        f"<td>{esc(fmt_num(row.get('close'), 2))}</td>"
        f"<td class=\"{esc('pos' if (row.get('change_pct') or 0) > 0 else 'neg')}\">"
        f"{esc(fmt_pct(row.get('change_pct')))}</td>"
        "</tr>"
        for name, row in sectors.items()
    )

    layer_cards = "".join(
        [
            metric_card(
                "技術面",
                "有",
                "1/5/20/60日漲跌、均線、20日支撐壓力、高點回落",
                "good",
            ),
            metric_card(
                "籌碼面",
                "有",
                "台股融資/融券、三大法人；美股 FINRA margin debt",
                "good",
            ),
            metric_card(
                "量價/風險",
                "有",
                "成交量或成交金額對前20日均量、VIX、跨市場回推",
                "warn",
            ),
            metric_card(
                "基本面",
                "未納入",
                "尚未抓 PE、PB、EPS、殖利率、利率與總經資料",
                "muted",
            ),
        ]
    )

    overview_cards = "".join(
        [
            metric_card(
                "台股收盤",
                fmt_num(tw_ind.close, 2),
                f"{fmt_pct(tw_ind.day_change_pct)} / {tw_ind.trend_label}",
                "bad" if (tw_ind.day_change_pct or 0) < 0 else "good",
            ),
            metric_card(
                "台股量能",
                fmt_num(tw_ind.volume_ratio_20d, 2),
                f"前20日均量比，{tw['volume_price_label']}",
                "warn",
            ),
            metric_card(
                "SPY",
                fmt_num(spy.close, 2),
                f"{fmt_pct(spy.day_change_pct)} / {spy.trend_label}",
                "bad" if (spy.day_change_pct or 0) < 0 else "good",
            ),
            metric_card(
                "VIX",
                fmt_num((us.get("vix", {}).get("indicators") or {}).get("close"), 2),
                us["vix"]["label"],
                "warn",
            ),
        ]
    )

    taiwan_cards = "".join(
        [
            metric_card("加權指數", fmt_num(tw_ind.close, 2), f"20日 {fmt_pct(tw_ind.twenty_day_pct)}", "neutral"),
            metric_card("成交金額比", fmt_num(tw_ind.volume_ratio_20d, 2), "相對前20日平均成交金額", "warn"),
            metric_card("融資餘額", fmt_big_twd(financing.get("today_balance_ntd")), f"單日 {fmt_big_twd(financing.get('balance_change_ntd'))}", "bad" if (financing.get("balance_change_ntd") or 0) < 0 else "good"),
            metric_card("法人合計", fmt_big_twd(total_net), f"外資 {fmt_big_twd(foreign_net)} / 投信 {fmt_big_twd(trust_net)}", "bad" if (total_net or 0) < 0 else "good"),
        ]
    )

    us_cards = "".join(
        [
            metric_card("SPY", fmt_num(spy.close, 2), f"20日 {fmt_pct(spy.twenty_day_pct)}", "neutral"),
            metric_card("QQQ", fmt_num(qqq.close, 2), f"20日 {fmt_pct(qqq.twenty_day_pct)}", "neutral"),
            metric_card("IWM", fmt_num(iwm.close, 2), f"20日 {fmt_pct(iwm.twenty_day_pct)}", "neutral"),
            metric_card("FINRA 融資", fmt_big_usd_millions(finra_latest.get("debit_balances_millions")), f"{finra_latest.get('month', 'N/A')} / 月變動 {fmt_pct(finra_latest.get('mom_debit_pct'))}", "warn"),
        ]
    )
    prediction_cards = "".join(
        [
            metric_card(
                "台股下次開盤",
                str(tw_pred.get("bias", "N/A")),
                f"{tw_pred.get('target_date_estimate', 'N/A')} / 信心 {tw_pred.get('confidence', 'N/A')}",
                prediction_tone(tw_pred.get("bias", "")),
            ),
            metric_card(
                "台股分數",
                str(tw_pred.get("score", "N/A")),
                fmt_prediction_range(tw_pred),
                "warn",
            ),
            metric_card(
                "美股下次開盤",
                str(us_pred.get("bias", "N/A")),
                f"{us_pred.get('target_date_estimate', 'N/A')} / 信心 {us_pred.get('confidence', 'N/A')}",
                prediction_tone(us_pred.get("bias", "")),
            ),
            metric_card(
                "美股分數",
                str(us_pred.get("score", "N/A")),
                fmt_prediction_range(us_pred),
                "warn",
            ),
        ]
    )
    accuracy_cards = "".join(
        [
            metric_card(
                "台股價量診斷",
                fmt_pct(tw_acc.get("hit_rate"), signed=False),
                f"樣本 {tw_acc.get('sample_size', 0)} / {accuracy_baseline_note(tw_acc)}",
                "neutral",
            ),
            metric_card(
                "美股價量診斷",
                fmt_pct(us_acc.get("hit_rate"), signed=False),
                f"樣本 {us_acc.get('sample_size', 0)} / {accuracy_baseline_note(us_acc)}",
                "neutral",
            ),
            metric_card(
                "正式預測",
                fmt_pct(archived_acc.get("hit_rate"), signed=False),
                f"樣本 {archived_acc.get('sample_size', 0)} / 每日報告對照",
                "warn",
            ),
            metric_card(
                "Macro-F1",
                fmt_pct(accuracy_macro_f1(archived_acc), signed=False),
                "正式預測三分類品質；樣本少時僅供觀察",
                "warn",
            ),
            metric_card(
                "失準待調校",
                str(len(improvement_log)),
                "原因會留在 JSON 與本頁表格",
                "bad" if improvement_log else "good",
            ),
        ]
    )

    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>台美股大盤日報 - {esc(report['report_date'])}</title>
  <style>
    :root {{
      --ink: #18212f;
      --muted: #657085;
      --line: #d7deea;
      --paper: #f6f8fb;
      --panel: #ffffff;
      --tw: #0f766e;
      --us: #1d4ed8;
      --warn: #b45309;
      --bad: #b42318;
      --good: #047857;
      --shadow: 0 10px 30px rgba(24, 33, 47, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: var(--paper);
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans TC", Arial, sans-serif;
      line-height: 1.55;
    }}
    .layout {{ display: grid; grid-template-columns: 240px minmax(0, 1fr); min-height: 100vh; }}
    .sidebar {{
      position: sticky;
      top: 0;
      height: 100vh;
      padding: 22px 16px;
      background: #111827;
      color: #f8fafc;
      border-right: 1px solid #0b1220;
    }}
    .brand {{ margin-bottom: 24px; }}
    .brand small {{ display: block; color: #9ca3af; font-size: 12px; }}
    .brand strong {{ display: block; font-size: 22px; letter-spacing: 0; }}
    .nav {{ display: grid; gap: 8px; }}
    .nav a {{
      color: #cbd5e1;
      text-decoration: none;
      padding: 10px 12px;
      border-radius: 8px;
      font-weight: 650;
      outline: none;
    }}
    .nav a:hover, .nav a:focus, .nav a.active {{ color: white; background: #243044; }}
    .meta {{ position: absolute; left: 16px; right: 16px; bottom: 20px; color: #9ca3af; font-size: 12px; }}
    main {{ padding: 28px; max-width: 1240px; width: 100%; }}
    .page {{ display: none; }}
    .page.active {{ display: block; }}
    .hero {{
      display: grid;
      gap: 10px;
      margin-bottom: 20px;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--line);
    }}
    h1, h2, h3 {{ margin: 0; line-height: 1.18; letter-spacing: 0; }}
    h1 {{ font-size: clamp(28px, 4vw, 48px); }}
    h2 {{ font-size: 26px; margin-bottom: 14px; }}
    h3 {{ font-size: 18px; margin: 22px 0 10px; }}
    .subtle {{ color: var(--muted); }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 16px 0 22px; }}
    .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-top: 4px solid #94a3b8;
      border-radius: 8px;
      padding: 14px;
      min-height: 128px;
      box-shadow: var(--shadow);
    }}
    .metric-good {{ border-top-color: var(--good); }}
    .metric-bad {{ border-top-color: var(--bad); }}
    .metric-warn {{ border-top-color: var(--warn); }}
    .metric-muted {{ border-top-color: #64748b; }}
    .metric-label {{ color: var(--muted); font-size: 12px; font-weight: 750; text-transform: uppercase; }}
    .metric-value {{ font-size: 26px; font-weight: 800; margin-top: 8px; word-break: break-word; }}
    .metric-note {{ color: var(--muted); font-size: 13px; margin-top: 8px; }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      margin: 14px 0;
      box-shadow: var(--shadow);
    }}
    .panel.tw {{ border-left: 5px solid var(--tw); }}
    .panel.us {{ border-left: 5px solid var(--us); }}
    .pred-good {{ border-left: 5px solid var(--good); }}
    .pred-bad {{ border-left: 5px solid var(--bad); }}
    .pred-warn {{ border-left: 5px solid var(--warn); }}
    ul {{ margin: 8px 0 0 20px; padding: 0; }}
    li {{ margin: 7px 0; }}
    table {{ width: 100%; border-collapse: collapse; background: var(--panel); border-radius: 8px; overflow: hidden; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: left; }}
    th {{ color: var(--muted); font-size: 12px; text-transform: uppercase; }}
    .pos {{ color: var(--good); font-weight: 750; }}
    .neg {{ color: var(--bad); font-weight: 750; }}
    .split {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
    .stamp {{ display: inline-flex; gap: 8px; flex-wrap: wrap; }}
    .stamp span {{ background: #e8eef8; color: #263449; padding: 5px 8px; border-radius: 8px; font-size: 12px; font-weight: 700; }}
    @media (max-width: 900px) {{
      .layout {{ grid-template-columns: 1fr; }}
      .sidebar {{ position: relative; height: auto; }}
      .meta {{ position: static; margin-top: 20px; }}
      .grid, .split {{ grid-template-columns: 1fr; }}
      main {{ padding: 18px; }}
    }}
  </style>
</head>
<body>
  <div class="layout">
    <aside class="sidebar">
      <div class="brand">
        <small>Daily Market Desk</small>
        <strong>台美股大盤日報</strong>
      </div>
      <nav class="nav" aria-label="report pages">
        <a href="#overview" class="active">總覽</a>
        <a href="#prediction">預測</a>
        <a href="#accuracy">準確率</a>
        <a href="#taiwan">台股</a>
        <a href="#us">美股</a>
        <a href="#cross">回推</a>
        <a href="#stock2330">台積電 2330</a>
        <a href="#sources">資料源</a>
      </nav>
      <div class="meta">
        <div>{esc(report['report_date'])}</div>
        <div>{esc(report['generated_at'])}</div>
      </div>
    </aside>
    <main>
      <section id="overview" class="page active">
        <div class="hero">
          <h1>台美股大盤日報</h1>
          <div class="subtle">台股資料日 {esc(tw['latest_date'])}；美股資料日 {esc(us['latest_date'])}</div>
          <div class="stamp"><span>技術</span><span>量價</span><span>籌碼</span><span>風險</span><span>基本面未納入</span></div>
        </div>
        <div class="grid">{overview_cards}</div>
        <div class="panel">
          <h2>分析層面</h2>
          <div class="grid">{layer_cards}</div>
        </div>
        <div class="panel">
          <h2>當日結論</h2>
          {bullet_list([
              f"台股：{tw_ind.trend_label}，{tw['volume_price_label']}，{tw['leverage_label']}。",
              f"美股：SPY {spy.trend_label}，QQQ {qqq.trend_label}，{us['volume_price_label']}，{us['vix']['label']}。",
              f"回推：{cross_lines[0]}",
          ])}
        </div>
      </section>

      <section id="prediction" class="page">
        <div class="hero">
          <h1>下一交易日走勢預測</h1>
          <div class="subtle">用規則式訊號產生方向、信心、分數與失準調校所需的理由紀錄。</div>
        </div>
        <div class="grid">{prediction_cards}</div>
        {html_prediction_block(tw_pred)}
        {html_prediction_block(us_pred)}
      </section>

      <section id="accuracy" class="page">
        <div class="hero">
          <h1>準確率測試</h1>
          <div class="subtle">每天自動讀取過去 JSON 預測，對照目前已取得的實際收盤資料。</div>
        </div>
        <div class="grid">{accuracy_cards}</div>
        <div class="panel">
          <h2>正式每日預測驗證</h2>
          <p class="subtle">這裡才是每天報告產出的預測與後續真實盤勢對照；價量診斷回測只檢查歷史價格與量能子模型。</p>
          <table>
            <thead><tr><th>市場</th><th>目標日</th><th>預測</th><th>實際</th><th>失準原因</th></tr></thead>
            <tbody>{html_miss_rows(archived_acc.get("records", []))}</tbody>
          </table>
        </div>
        <div class="panel">
          <h2>Baseline 對照</h2>
          {bullet_list([
              f"台股價量診斷：模型 {fmt_pct(tw_acc.get('hit_rate'), signed=False)}；{accuracy_baseline_note(tw_acc)}；macro-F1 {fmt_pct(accuracy_macro_f1(tw_acc), signed=False)}。",
              f"美股價量診斷：模型 {fmt_pct(us_acc.get('hit_rate'), signed=False)}；{accuracy_baseline_note(us_acc)}；macro-F1 {fmt_pct(accuracy_macro_f1(us_acc), signed=False)}。",
              "若模型長期贏不了永遠偏多或今日方向延續，代表規則沒有可用 edge，應優先調整資料源或公式。",
          ])}
        </div>
        <div class="panel">
          <h2>近期失準 / 待調校</h2>
          <table>
            <thead><tr><th>市場</th><th>日期</th><th>預測</th><th>實際</th><th>原因</th></tr></thead>
            <tbody>{html_miss_rows(improvement_log)}</tbody>
          </table>
        </div>
        <div class="panel">
          <h2>限制</h2>
          {bullet_list(unique_strings(tw_acc.get("limitations", []) + us_acc.get("limitations", []) + archived_acc.get("limitations", []))[:8])}
        </div>
      </section>

      <section id="taiwan" class="page">
        <div class="hero">
          <h1>台股</h1>
          <div class="subtle">TWSE 加權指數、成交金額、融資融券、三大法人與族群表現。</div>
        </div>
        <div class="grid">{taiwan_cards}</div>
        <div class="panel tw">
          <h2>台股大盤</h2>
          {bullet_list(summarize_twse(tw))}
        </div>
        <div class="panel">
          <h2>觀察價位</h2>
          {bullet_list([
              f"20日區間支撐 {fmt_num(tw_levels['support_20d'], 2)}",
              f"20日區間壓力 {fmt_num(tw_levels['resistance_20d'], 2)}",
              f"20日線 {fmt_num(tw_ind.ma20, 2)}，60日線 {fmt_num(tw_ind.ma60, 2)}",
          ])}
        </div>
        <div class="panel">
          <h2>族群</h2>
          <table>
            <thead><tr><th>指數</th><th>收盤</th><th>日變動</th></tr></thead>
            <tbody>{sector_rows}</tbody>
          </table>
        </div>
      </section>

      <section id="us" class="page">
        <div class="hero">
          <h1>美股</h1>
          <div class="subtle">SPY/QQQ/IWM 作為大盤、科技與小型股代理，搭配 VIX 與 FINRA margin debt。</div>
        </div>
        <div class="grid">{us_cards}</div>
        <div class="panel us">
          <h2>美股大盤</h2>
          {bullet_list(summarize_us(us))}
        </div>
        <div class="panel">
          <h2>觀察價位</h2>
          <div class="split">
            <div>
              <h3>SPY</h3>
              {bullet_list([
                  f"20日區間支撐 {fmt_num(spy_levels['support_20d'], 2)}",
                  f"20日區間壓力 {fmt_num(spy_levels['resistance_20d'], 2)}",
                  f"20日線 {fmt_num(spy.ma20, 2)}，60日線 {fmt_num(spy.ma60, 2)}",
              ])}
            </div>
            <div>
              <h3>QQQ</h3>
              {bullet_list([
                  f"20日區間支撐 {fmt_num(qqq_levels['support_20d'], 2)}",
                  f"20日區間壓力 {fmt_num(qqq_levels['resistance_20d'], 2)}",
                  f"20日線 {fmt_num(qqq.ma20, 2)}，60日線 {fmt_num(qqq.ma60, 2)}",
              ])}
            </div>
          </div>
        </div>
      </section>

      <section id="cross" class="page">
        <div class="hero">
          <h1>回推</h1>
          <div class="subtle">用台股、美國科技股、量能、VIX 與槓桿背景判斷隔日風險。</div>
        </div>
        <div class="panel">
          <h2>跨市場訊號</h2>
          {bullet_list(cross_lines)}
        </div>
        <div class="panel">
          <h2>規則</h2>
          {bullet_list([
              "價格跌破 20 日線且放量，優先視為供給壓力；若隔日縮量止跌，才改看整理。",
              "融資增加搭配下跌，通常代表槓桿尚未洗乾淨；融資減少搭配下跌，則是去槓桿正在發生。",
              "美股 margin debt 是月資料，不適合判斷單日轉折，適合判斷槓桿背景溫度。",
          ])}
        </div>
      </section>

      <section id="sources" class="page">
        <div class="hero">
          <h1>資料源</h1>
          <div class="subtle">全部由腳本自動抓取，沒有呼叫 LLM。</div>
        </div>
        <div class="panel">
          {bullet_list([
              "TWSE FMTQIK：台股每日成交資訊",
              "TWSE MI_MARGN：上市信用交易統計",
              "TWSE BFI82U：三大法人買賣金額統計",
              "Nasdaq historical API：SPY / QQQ / IWM / COMP 價量資料",
              "Cboe VIX History：VIX 日資料",
              "FINRA Margin Statistics：美股 margin debt 月資料",
          ])}
          <p class="subtle">註：這是規則式市場分析，不是投資建議。</p>
        </div>
      </section>
      {stock_2330_section}
    </main>
  </div>
  <script>
    const pages = Array.from(document.querySelectorAll(".page"));
    const links = Array.from(document.querySelectorAll(".nav a"));
    function showPage(hash) {{
      const id = (hash || "#overview").replace("#", "");
      pages.forEach((page) => page.classList.toggle("active", page.id === id));
      links.forEach((link) => link.classList.toggle("active", link.getAttribute("href") === "#" + id));
      if (!document.getElementById(id)) showPage("#overview");
    }}
    links.forEach((link) => link.addEventListener("click", () => showPage(link.getAttribute("href"))));
    window.addEventListener("hashchange", () => showPage(location.hash));
    showPage(location.hash);
  </script>
</body>
</html>
"""


def write_outputs(report: dict[str, Any], report_dir: Path, data_dir: Path) -> tuple[Path, Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    report_date = report["report_date"]
    markdown_path = report_dir / f"{report_date}.md"
    html_path = report_dir / f"{report_date}.html"
    json_path = data_dir / f"{report_date}.json"

    markdown = render_markdown(report)
    markdown_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(render_html(report), encoding="utf-8")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    latest_md = report_dir / "latest.md"
    latest_html = report_dir / "latest.html"
    latest_json = data_dir / "latest.json"
    shutil.copyfile(markdown_path, latest_md)
    shutil.copyfile(html_path, latest_html)
    shutil.copyfile(json_path, latest_json)
    update_evaluation_history(report, data_dir)

    return markdown_path, html_path, json_path


# ---------------------------------------------------------------------------
# 台積電 2330 multi-facet scorecard integration
# ---------------------------------------------------------------------------

# Map trend_label strings to a synthetic 20-day return proxy used by market_facet.
_TREND_TO_RET20: dict[str, float] = {
    "多頭": 0.05,
    "偏多": 0.03,
    "中性": 0.0,
    "偏空": -0.03,
    "空頭": -0.05,
}


def assemble_2330(
    daily_bars: list,
    valuation: Optional[dict[str, Any]],
    revenue: Optional[dict[str, Any]],
    t86: Optional[dict[str, Any]],
    peer: Optional[dict[str, Any]],
    market: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Pure assembly: compute metrics → facets → scorecard for 2330.

    Args:
        daily_bars: list of MarketBar (at least 60 recommended).
        valuation: dict with pe, pb, dividend_yield, pe_pct, pb_pct (or None).
        revenue: dict with yoy_pct, mom_pct, cum_yoy_pct (or None → revenue facet score None).
        t86: single T86 dict with inst_total_net_lots (or None).
        peer: dict with smh_20d (SMH 20d return), adr_day (ADR day change).
        market: dict with index_trend (str label) and vix (float).

    Returns:
        scorecard dict with keys: facets, short_lean, medium_lean, missing,
        plus observe_levels with support/resistance/ma20/ma60.
    """
    # Map incoming peer keys to what peer_facet expects
    peer_mapped: Optional[dict[str, Any]] = None
    if peer is not None:
        peer_mapped = {
            "smh_ret20": peer.get("smh_20d"),
            "adr_ret": peer.get("adr_day"),
        }

    # Map market: translate trend label to synthetic taiex_ret20
    market_mapped: Optional[dict[str, Any]] = None
    if market is not None:
        trend_label = market.get("index_trend", "中性")
        taiex_ret20 = _TREND_TO_RET20.get(trend_label, 0.0)
        market_mapped = {
            "taiex_ret20": taiex_ret20,
            "vix": market.get("vix"),
        }

    # Wrap single T86 dict into a list for chip_facet
    t86_list: Optional[list[dict[str, Any]]] = [t86] if t86 is not None else None

    # revenue_metrics returns {"yoy_pct": None, ...} when None → revenue_facet → score None
    rev_metrics = _stocks.revenue_metrics(revenue)

    scorecard = _stocks.build_2330_scorecard(
        daily_bars=daily_bars,
        weekly_context=None,
        valuation=valuation,
        revenue=rev_metrics,
        t86_recent=t86_list,
        peer=peer_mapped,
        market=market_mapped,
    )

    # Attach observe levels from the daily bars
    tech_m = _stocks.technical_metrics(daily_bars) if daily_bars else {}
    closes = [b.close for b in daily_bars] if daily_bars else []
    recent20 = closes[-20:] if len(closes) >= 20 else closes
    scorecard["observe_levels"] = {
        "support_20d": min(recent20) if recent20 else None,
        "resistance_20d": max(recent20) if recent20 else None,
        "ma20": tech_m.get("ma20"),
        "ma60": tech_m.get("ma60"),
        "ma5": tech_m.get("ma5"),
        "close": tech_m.get("close"),
    }

    return scorecard


def build_stock_2330(
    report_us: dict[str, Any],
    report_tw: dict[str, Any],
    end: date,
    store: "HistoryStore",
) -> dict[str, Any]:
    """Fetch 2330 data, upsert to store, delegate to assemble_2330.

    Each data source is independently wrapped; failure → that facet input None.
    Returns scorecard dict or {"error": ...} if core price fetch fails.
    """
    from marketlib.fetchers import fetch_nasdaq_history

    end_iso = end.isoformat()

    # --- Core: 2330 daily price history (required) ---
    try:
        daily_bars = _stocks.fetch_stock_history("2330", end, lookback_months=5)
        if not daily_bars:
            raise ValueError("empty 2330 price history")
    except Exception as exc:  # noqa: BLE001
        return {"error": f"2330 price fetch failed: {exc}"}

    # --- T86 institutional chip (best-effort) ---
    t86: Optional[dict[str, Any]] = None
    try:
        t86 = _stocks.fetch_t86_for_stock("2330", end_iso)
    except Exception:  # noqa: BLE001
        pass

    # --- Valuation PE/PB (best-effort) ---
    valuation_raw: Optional[dict[str, Any]] = None
    try:
        valuation_raw = _stocks.fetch_valuation_for_stock("2330", end_iso)
    except Exception:  # noqa: BLE001
        pass

    # --- Monthly revenue (best-effort) ---
    revenue_raw: Optional[dict[str, Any]] = None
    try:
        revenue_raw = _stocks.fetch_monthly_revenue("2330")
    except Exception:  # noqa: BLE001
        pass

    # --- Upsert fetched data into store ---
    try:
        # Upsert daily bars
        for bar in daily_bars:
            row: dict[str, Any] = {
                "stock_no": "2330",
                "date": bar.date,
                "close": bar.close,
                "high": bar.high,
                "low": bar.low,
                "open": bar.open,
                "volume": bar.volume,
            }
            if t86 is not None and bar.date == daily_bars[-1].date:
                row.update({
                    "foreign_net_lots": t86.get("foreign_net_lots"),
                    "trust_net_lots": t86.get("trust_net_lots"),
                    "dealer_net_lots": t86.get("dealer_net_lots"),
                    "inst_total_net_lots": t86.get("inst_total_net_lots"),
                })
            store.upsert_stock_daily(row)

        # Upsert valuation
        if valuation_raw is not None:
            store.upsert_stock_valuation({
                "stock_no": "2330",
                "date": end_iso,
                "pe": valuation_raw.get("pe"),
                "pb": valuation_raw.get("pb"),
                "dividend_yield": valuation_raw.get("dividend_yield"),
            })

        # Upsert revenue
        if revenue_raw is not None:
            store.upsert_stock_monthly_revenue({
                "stock_no": "2330",
                "month": revenue_raw.get("month", end_iso[:7]),
                "revenue_thousand": revenue_raw.get("revenue_thousand"),
                "yoy_pct": revenue_raw.get("yoy_pct"),
                "mom_pct": revenue_raw.get("mom_pct"),
                "cum_yoy_pct": revenue_raw.get("cum_yoy_pct"),
            })
    except Exception:  # noqa: BLE001
        pass  # store upsert failure is non-fatal

    # --- Build PE/PB history from store for percentile calculation ---
    pe_history: list[float] = []
    pb_history: list[float] = []
    try:
        val_rows = store.get_stock_valuation("2330")
        pe_history = [r["pe"] for r in val_rows if r.get("pe") is not None]
        pb_history = [r["pb"] for r in val_rows if r.get("pb") is not None]
    except Exception:  # noqa: BLE001
        pass

    valuation = _stocks.valuation_metrics(valuation_raw, pe_history, pb_history)

    # --- Peer: fetch SMH and TSM ADR (best-effort) ---
    peer: Optional[dict[str, Any]] = None
    try:
        start_peer = end - timedelta(days=40)
        smh_bars = fetch_nasdaq_history("SMH", "etf", start_peer, end)
        smh_closes = [b.close for b in smh_bars if b.close]
        smh_ret20: Optional[float] = None
        if len(smh_closes) >= 21:
            smh_ret20 = smh_closes[-1] / smh_closes[-21] - 1
        elif len(smh_closes) >= 2:
            smh_ret20 = smh_closes[-1] / smh_closes[0] - 1

        tsm_bars = fetch_nasdaq_history("TSM", "stocks", start_peer, end)
        adr_day: Optional[float] = None
        if len(tsm_bars) >= 2:
            adr_day = tsm_bars[-1].close / tsm_bars[-2].close - 1

        if smh_ret20 is not None or adr_day is not None:
            peer = {"smh_20d": smh_ret20, "adr_day": adr_day}
    except Exception:  # noqa: BLE001
        pass  # peer facet will be missing

    # --- Market: reuse existing report data ---
    tw_ind_dict = report_tw.get("indicators") or {}
    vix_data = (report_us.get("vix") or {}).get("indicators") or {}
    market: Optional[dict[str, Any]] = None
    index_trend = tw_ind_dict.get("trend_label")
    vix_close = vix_data.get("close")
    if index_trend is not None or vix_close is not None:
        market = {
            "index_trend": index_trend or "中性",
            "vix": vix_close,
        }

    return assemble_2330(daily_bars, valuation, revenue_raw, t86, peer, market)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate daily US/Taiwan market report.")
    parser.add_argument(
        "--date",
        default=datetime.now(TAIPEI_TZ).date().isoformat(),
        help="Report date in YYYY-MM-DD. Defaults to today in Asia/Taipei.",
    )
    parser.add_argument(
        "--lookback-months",
        type=int,
        default=5,
        help="TWSE monthly history window. 5 months usually gives 80-100 trading days.",
    )
    parser.add_argument(
        "--us-lookback-days",
        type=int,
        default=140,
        help="US calendar-day lookback window.",
    )
    parser.add_argument(
        "--report-dir",
        default=str(BASE_DIR / "reports"),
        help="Markdown output directory.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(BASE_DIR / "data"),
        help="JSON output directory.",
    )
    parser.add_argument(
        "--model-dir",
        default=str(BASE_DIR / "models"),
        help="Directory containing model_tw.json and model_us.json.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    generated_at = datetime.now(TAIPEI_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")

    taiwan = build_taiwan_report(report_date, args.lookback_months)
    us = build_us_report(report_date, args.us_lookback_days)
    report = {
        "report_date": report_date.isoformat(),
        "generated_at": generated_at,
        "taiwan": taiwan,
        "us": us,
    }
    # Rule-based predictions (primary until model overrides)
    report["predictions"] = build_predictions(report)

    # Model inference: upsert current report data to store, then read store rows
    # for inference (avoids train/serve skew — model was trained on store-shaped rows).
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    db_path = data_dir / "history.db"
    data_dir.mkdir(parents=True, exist_ok=True)
    store = HistoryStore(str(db_path))
    store.init_schema()
    try:
        upsert_report_into_store(report, store)
        tw_rows = store.get_tw_rows()
        us_rows = store.get_us_rows()

        # 2330 per-stock scorecard (best-effort; never aborts index report)
        try:
            report["stock_2330"] = build_stock_2330(us, taiwan, report_date, store)
        except Exception as exc:  # noqa: BLE001
            report["stock_2330"] = {"error": str(exc)}
    finally:
        store.close()

    # Overlay model predictions on top of rule-based predictions where available.
    for market_key, market_label, model_file, market_rows in (
        ("taiwan", "tw", "model_tw.json", tw_rows),
        ("us", "us", "model_us.json", us_rows),
    ):
        rule_pred = report["predictions"][market_key]
        mp = model_prediction(
            market_label,
            str(model_dir / model_file),
            market_rows,
            us_rows,
        )
        if mp is not None:
            # Overlay: keep all rule dict keys intact, override bias/confidence,
            # add probability/model_version/prediction_engine/secondary_rule_opinion.
            # Do NOT clobber neutral_threshold_pct (rule dict's real value is needed
            # by evaluate_archived_predictions and renderers).
            report["predictions"][market_key] = dict(rule_pred)
            report["predictions"][market_key]["bias"] = mp["bias"]
            report["predictions"][market_key]["confidence"] = mp["confidence"]
            report["predictions"][market_key]["probability"] = mp["probability"]
            report["predictions"][market_key]["model_version"] = mp["model_version"]
            report["predictions"][market_key]["prediction_engine"] = "model"
            report["predictions"][market_key]["secondary_rule_opinion"] = {
                "bias": rule_pred["bias"],
                "score": rule_pred["score"],
            }
        else:
            report["predictions"][market_key]["prediction_engine"] = "rule_fallback"

    report["accuracy"] = build_accuracy_report(report, data_dir)

    markdown_path, html_path, json_path = write_outputs(
        report,
        Path(args.report_dir),
        data_dir,
    )
    print(f"Wrote {markdown_path}")
    print(f"Wrote {html_path}")
    print(f"Wrote {json_path}")
    print(f"Latest Markdown report: {Path(args.report_dir) / 'latest.md'}")
    print(f"Latest HTML report: {Path(args.report_dir) / 'latest.html'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - cron should record readable failures.
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
