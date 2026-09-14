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
    fetch_taifex_inst_futures,
    twse_asof,
    taifex_asof,
    us_price_asof,
    vix_asof,
)
from marketlib import model as ml_model
from marketlib.features import build_feature_row, decision_time_for, FEATURE_NAMES_TW, FEATURE_NAMES_US
from marketlib.store import HistoryStore
from marketlib import stocks as _stocks
from marketlib import screener as _screener
from marketlib.universe import TW50_CONSTITUENTS
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
    try:
        taifex = (fetch_taifex_inst_futures(latest_date) or {}).get(latest_date)
    except Exception as exc:  # 期貨為輔助訊號,抓取失敗不應中斷整份報告
        print(f"[taiwan] TAIFEX fetch error on {latest_date}: {exc}", file=sys.stderr)
        taifex = None
    return {
        "latest_date": latest_date,
        "bars": [asdict(bar) for bar in bars],
        "indicators": asdict(indicators),
        "margin": margin,
        "institutional": institutional,
        "summary": summary,
        "taifex": taifex,
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
    # US output was dropped (focus on TW). US data (report["us"]) is still fetched
    # and passed into the TW prediction as an input signal — see build_taiwan_prediction
    # and the us_* model features. build_us_prediction() is retained but no longer called.
    return {
        "taiwan": build_taiwan_prediction(report["taiwan"], report["us"]),
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
    if ml_model.is_two_stage(model):
        probs = ml_model.predict_three_class(model, feats)
        bias = ml_model.direction_from_probs(probs)
        return {
            "bias": bias,
            "probability": round(probs[bias], 4),
            "class_probabilities": {k: round(v, 4) for k, v in probs.items()},
            "confidence": ml_model.confidence_from_probs(model, probs),
            "confidence_calibrated": True,
            "tier_evidence": ml_model.tier_evidence(model, probs),
            "oos_reference": model.get("oos_reference"),
            "edge_demonstrated": (model.get("oos_reference") or {}).get(
                "edge_demonstrated"),
            "model_version": model.get("trained_at", "unknown"),
        }
    p = ml_model.predict_proba(model, feats)
    return {"bias": ml_model.proba_to_direction(model, p),
            "probability": round(p, 4),
            "confidence": ml_model.proba_to_confidence(model, p),
            "confidence_calibrated": (model.get("calibration") or {}).get("method") == "platt",
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
        taifex = tw.get("taifex") or {}
        tx_foreign = taifex.get("tx_foreign_oi_net_lots")
        tx_inst = taifex.get("tx_inst_oi_net_lots")

        price_asof = twse_asof(tw_date, "price")
        inst_asof = twse_asof(tw_date, "inst") if net is not None else None
        margin_asof = twse_asof(tw_date, "margin") if financing else None
        breadth_asof = twse_asof(tw_date, "breadth") if up is not None else None
        taifex_asof_val = taifex_asof(tw_date) if tx_foreign is not None else None

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
            "tx_foreign_oi_net_lots": tx_foreign,
            "tx_inst_oi_net_lots": tx_inst,
            "price_asof_taipei": price_asof,
            "inst_asof_taipei": inst_asof,
            "margin_asof_taipei": margin_asof,
            "breadth_asof_taipei": breadth_asof,
            "taifex_asof_taipei": taifex_asof_val,
            "available_at_taipei": _max_asof(price_asof, inst_asof, margin_asof,
                                             breadth_asof, taifex_asof_val),
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
    # US dropped from output. TW-only price/volume diagnostic backtest + archived
    # prediction eval. build_improvement_log still takes a US backtest slot; we
    # pass an empty one so no US misses appear in the improvement log.
    taiwan_bt = rolling_backtest(report["taiwan"]["bars"], "turnover")
    archive_eval = evaluate_archived_predictions(report, data_dir)
    return {
        "formula_version": "rule-v1.1",
        "taiwan": taiwan_bt,
        "archived_predictions": archive_eval,
        "improvement_log": build_improvement_log(taiwan_bt, {}, archive_eval),
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

    # US dropped from output: only re-evaluate TW archived predictions. Historical
    # US rows already in evaluation_history.json are preserved by the idempotent
    # merge in update_evaluation_history (they are never in current_records to
    # overwrite, and existing rows are kept).
    current_maps = {
        "taiwan": market_bars_by_date(report, "taiwan"),
    }
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
        for key in ("taiwan",):
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
                    "prediction": miss.get("prediction"),
                    "actual_direction": miss.get("actual_direction"),
                    "reason": miss["miss_reason"],
                    "action": "保留供後續調整公式權重或補資料源",
                }
            )
    for miss in [row for row in archive_eval.get("records", []) if not row.get("hit")][-5:]:
        rows.append(
            {
                "market": miss.get("market", "歷史預測"),
                "date": miss.get("target_date", "N/A"),
                "prediction": miss.get("prediction"),
                "actual_direction": miss.get("actual_direction"),
                "reason": miss.get("miss_reason", "N/A"),
                "action": "保留供後續 AI 分析與訓練",
            }
        )
    # De-duplicate on (market, date, prediction, actual_direction, reason)
    seen: set[tuple] = set()
    deduped: list[dict[str, str]] = []
    for row in rows:
        key = (
            row.get("market"),
            row.get("date"),
            row.get("prediction"),
            row.get("actual_direction"),
            row.get("reason"),
        )
        if key not in seen:
            seen.add(key)
            deduped.append(row)
    return deduped[-12:]


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


def _fmt_macd_row(label: str, m: Optional[dict[str, Any]]) -> str:
    """Render one MACD row (日/週/月) as a <li>."""
    if m is None:
        return f"{label} MACD：N/A"
    dif = fmt_num(m.get("dif"), 2)
    dea = fmt_num(m.get("dea"), 2)
    hist = fmt_num(m.get("hist"), 2)
    cross = m.get("cross")
    cross_str = ""
    if cross == "golden":
        cross_str = "【金叉】"
    elif cross == "dead":
        cross_str = "【死叉】"
    return f"{label} MACD：DIF {dif} / DEA {dea} / 柱 {hist}{cross_str}"


def _html_tech_indicators_panel(tm: dict[str, Any]) -> str:
    """Build the 技術指標 panel HTML from a technical_metrics dict."""
    # Moving averages
    ma_items = [
        f"週線 MA5：{fmt_num(tm.get('ma5'), 2)}",
        f"月線 MA20：{fmt_num(tm.get('ma20'), 2)}",
        f"季線 MA60：{fmt_num(tm.get('ma60'), 2)}",
        f"半年線 MA120：{fmt_num(tm.get('ma120'), 2)}",
        f"年線 MA240：{fmt_num(tm.get('ma240'), 2)}",
    ]

    # Bollinger
    boll = tm.get("boll")
    if boll is not None:
        boll_items = [
            f"上軌：{fmt_num(boll.get('upper'), 2)}",
            f"中軌：{fmt_num(boll.get('mid'), 2)}",
            f"下軌：{fmt_num(boll.get('lower'), 2)}",
            f"%B：{fmt_num(boll.get('pctb'), 3) if boll.get('pctb') is not None else 'N/A'}",
        ]
    else:
        boll_items = ["布林通道資料不足"]

    # MACD
    macd_items = [
        _fmt_macd_row("日", tm.get("macd_daily")),
        _fmt_macd_row("週", tm.get("macd_weekly")),
        _fmt_macd_row("月", tm.get("macd_monthly")),
    ]

    # KD
    kd_k = fmt_num(tm.get("kd_k"), 2)
    kd_d = fmt_num(tm.get("kd_d"), 2)
    kd_items = [f"K：{kd_k}", f"D：{kd_d}"]

    # 大戶成本 (VWAP estimate)
    cost_items = [
        f"VWAP60：{fmt_num(tm.get('cost_vwap60'), 2)}",
        f"VWAP120：{fmt_num(tm.get('cost_vwap120'), 2)}",
        f"現價相對VWAP60：{fmt_pct(tm.get('price_vs_cost60'))}",
        "（估計,非真實分點成本）",
    ]

    def _sub_panel(title: str, items: list[str]) -> str:
        return (
            f'<div class="panel" style="margin-bottom:10px">'
            f'<strong>{esc(title)}</strong>'
            + bullet_list(items)
            + '</div>'
        )

    return (
        '<div class="panel">'
        '<h2>技術指標</h2>'
        '<div class="split">'
        '<div>'
        + _sub_panel("均線", ma_items)
        + _sub_panel("布林通道", boll_items)
        + '</div>'
        '<div>'
        + _sub_panel("MACD", macd_items)
        + _sub_panel("KD", kd_items)
        + _sub_panel("大戶成本（估計）", cost_items)
        + '</div>'
        '</div>'
        '</div>'
    )


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
    tm = stock.get("technical_metrics") or {}

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

    # Technical indicators panel (rendered from full technical_metrics)
    tech_indicators_html = _html_tech_indicators_panel(tm)

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
        + tech_indicators_html
        + f'<p class="subtle">{esc(missing_note)}</p>'
        '</section>'
    )


def markdown_intersection_block(screener: list[dict]) -> list[str]:
    """The technical-intersection shortlist, with the reasoning spelled out.

    Kept separate from the composite ranking below it because the two answer
    different questions. The ranking always produces a top 15, even in a market
    where nothing qualifies; the intersection can legitimately come back empty,
    and that emptiness is itself the finding.
    """
    lines: list[str] = ["### 技術面交集（全部條件同時成立）", ""]
    if not screener:
        return lines + ["- 資料不足，無法計算技術面交集。", ""]

    scored = [r for r in screener if r.get("intersection")]
    if not scored:
        return lines + ["- 個股技術指標資料不足，無法計算交集。", ""]

    hits = _screener.rank_candidates(scored)

    core_total = scored[0]["intersection"]["core_total"]
    lines.append(
        f"必要條件 {core_total} 項全數成立才列入；RSI ≥ {_screener.RSI_OVERHEATED:.0f} "
        f"視為過熱直接排除。共 {len(scored)} 檔納入檢查，**{len(hits)} 檔通過**。"
    )
    lines.append("")
    lines.append("必要條件（通過的個股一律全數成立，故僅列一次）：")
    for text in _screener.core_condition_texts():
        lines.append(f"1. {text}")
    lines.append("")

    if not hits:
        # Name the stocks that came closest — a near-miss list is more useful
        # than silence, as long as it is not dressed up as a pass.
        near = sorted(scored, key=lambda r: len(r["intersection"]["core_failed"]))[:5]
        lines.append("**本日無個股通過全部條件。** 最接近的幾檔與其卡關處：")
        lines.append("")
        for r in near:
            ins = r["intersection"]
            blockers = "；".join(ins["core_failed"] + ins["vetoes"]) or "—"
            lines.append(
                f"- **{r.get('code','')} {r.get('name','')}** "
                f"(必要條件 {ins['core_passed']}/{ins['core_total']}) — 未過：{blockers}"
            )
        lines.append("")
        return lines

    lines.append(
        "通過名單，依「延伸過度警訊少 → 加分項多 → 綜合排名前」排序。"
        "排名反映**今日技術面型態的乾淨程度**，不是報酬預測，也未做過勝率回測："
    )
    lines.append("")
    lines.append(
        "| 排名 | 代號 | 名稱 | 股價 | 河流圖位階 | 分位 | 本益比 | 股價淨值比 "
        "| 警訊 | 加分 | 綜排 | 距月線 | KD (K/D) | RSI | MACD柱 | 布林%b | 量比 |"
    )
    lines.append(
        "|------|------|------|------|------------|------|--------|------------"
        "|------|------|------|--------|----------|-----|--------|--------|------|"
    )
    for r in hits:
        ins = r["intersection"]
        m = ins["metrics_snapshot"]
        kd_str = (f"{m['kd_k']:.0f}/{m['kd_d']:.0f}"
                  if m.get("kd_k") is not None and m.get("kd_d") is not None else "N/A")
        riv = r.get("river") or {}
        band = riv.get("band")
        if band:
            band_str = f"{band}（{riv.get('primary_metric')}）"
        elif riv.get("split_factor"):
            band_str = f"分割未還原（{riv['split_factor']:.2g}:1）"
        else:
            band_str = "資料不足"
        pct_str = (f"{riv['percentile']:.0%}" if riv.get("percentile") is not None else "—")
        lines.append(
            f"| **{ins.get('shortlist_rank','—')}** | {r.get('code','')} | {r.get('name','')} "
            f"| {fmt_num(m.get('close'), 2)} | {band_str} | {pct_str} "
            f"| {fmt_num(riv.get('pe'), 2)} | {fmt_num(riv.get('pb'), 2)} "
            f"| {'⚠️ ' + str(len(ins['warnings'])) if ins['warnings'] else '—'} "
            f"| {ins['bonus_passed']}/{ins['bonus_total']} | {r.get('rank','—')} "
            f"| {fmt_pct(m.get('close_vs_ma20'))} | {kd_str} "
            f"| {fmt_num(m.get('rsi'), 0)} | {fmt_num(m.get('macd_hist'), 2)} "
            f"| {fmt_num(m.get('pctb'), 2)} | {fmt_num(m.get('volume_ratio_20d'), 2)} |"
        )
    lines.append("")
    banded = [r for r in hits if (r.get("river") or {}).get("percentile") is not None]
    rich = [r for r in banded if r["river"]["percentile"] >= 0.80]
    cheap = [r for r in banded if r["river"]["percentile"] < 0.20]
    lines.append(
        "河流圖位階 = 今日估值在該股**自身近 5 年**分布中的百分位（分位越低越便宜）。"
        "金融股與無可用 PE 者改以 PB 判讀，括號內為實際採用的指標。"
    )
    if banded:
        # The distribution is the finding: a technical screen run in a strong
        # tape naturally surfaces stocks that are already richly valued, and the
        # count says how much of that is happening today.
        lines.append("")
        lines.append(
            f"**{len(banded)} 檔可判讀估值中，{len(rich)} 檔位於自身 5 年分位 80% 以上"
            f"（昂貴區），{len(cheap)} 檔在 20% 以下（便宜區）。**"
            + ("技術面通過交集與估值便宜是兩件事，本清單同時符合者為："
               + "、".join(f"{r.get('code','')} {r.get('name','')}" for r in cheap)
               + "。" if cheap else "")
        )
    stale = [r for r in hits if (r.get("river") or {}).get("is_stale")]
    if stale:
        asof = sorted({r["river"].get("asof") for r in stale if r["river"].get("asof")})
        lines.append("")
        lines.append(
            f"> 估值資料為 {('、'.join(asof) if asof else '前一交易日')} 收盤（今日尚未公布），"
            f"價量指標則為最新。"
        )
    lines.append("")
    val_caveats = [(r, (r.get("river") or {}).get("caveats") or []) for r in hits]
    val_caveats = [(r, c) for r, c in val_caveats if c]
    if val_caveats:
        lines.append("估值判讀注意事項：")
        for r, cav in val_caveats:
            lines.append(f"- **{r.get('code','')} {r.get('name','')}** — " + "；".join(cav))
        lines.append("")

    flagged = [r for r in hits if r["intersection"]["warnings"]]
    if flagged:
        lines.append("⚠️ 延伸過度提醒（條件仍成立，但屬追高風險，是否進場請自行判斷）：")
        for r in flagged:
            lines.append(f"- **#{r['intersection'].get('shortlist_rank','—')} "
                         f"{r.get('code','')} {r.get('name','')}**")
            for w in r["intersection"]["warnings"]:
                lines.append(f"  - {w}")
        lines.append("")
    partial = [r for r in hits if r["intersection"]["bonus_failed"]]
    if partial:
        lines.append("加分項未滿的個股與缺項：")
        for r in partial:
            lines.append(
                f"- {r.get('code','')} {r.get('name','')} — "
                + "；".join(r["intersection"]["bonus_failed"])
            )
        lines.append("")
    lines.append(
        "> 交集只描述技術面目前的狀態，不含未來報酬的統計驗證；"
        "本清單未做過勝率回測，請當作篩選起點而非買賣訊號。"
    )
    lines.append("")
    return lines


def markdown_model_credibility(pred: dict[str, Any]) -> list[str]:
    """What today's call is actually worth, printed next to the call itself.

    A direction and a confidence word are cheap to produce and easy to over-read.
    These lines carry the measured track record for this confidence tier and,
    when the model's edge over a constant guess does not survive its own error
    bars, say so — in the report, not only in a commit message.
    """
    probs = pred.get("class_probabilities")
    if not probs:
        return []
    lines = ["", "### 這個預測值多少（可信度）", ""]
    lines.append(
        "- 三類機率："
        + "、".join(f"{k} {v:.1%}" if isinstance(v, (int, float)) else f"{k} N/A"
                    for k, v in probs.items())
        + "（已校準，總和為 1）"
    )

    tier = pred.get("tier_evidence") or {}
    if tier.get("oos_hit_rate") is not None:
        base = tier.get("oos_baseline_hit_rate")
        line = (
            f"- 本檔信心「{tier.get('label')}」的實測命中率："
            f"{tier['oos_hit_rate']:.1%}（滾動樣本 {tier.get('oos_sample_size')} 天）"
        )
        if base is not None:
            delta = tier["oos_hit_rate"] - base
            line += (f"，同期間「永遠猜震盪」為 {base:.1%}，"
                     f"差 {delta:+.1%}")
            if delta <= 0:
                line += "。**此信心層的方向判斷不優於常數猜測，建議忽略方向、只參考機率分布**"
        lines.append(line)

    ref = pred.get("oos_reference") or {}
    roll = ref.get("rolling") or {}
    if roll.get("hit_rate") is not None:
        best = max([v for v in (roll.get("baselines") or {}).values()
                    if v is not None], default=None)
        line = (f"- 整體滾動驗證：{roll['hit_rate']:.1%}"
                f"（{roll.get('pooled_sample_size')} 天，"
                f"贏 {roll.get('windows_won')}/{roll.get('windows_total')} 個窗口）")
        if best is not None:
            line += f"，最強常數基準 {best:.1%}"
        lines.append(line)

    edge = pred.get("edge_demonstrated")
    if edge is None:
        lines.append(
            "- 本次未取得滾動驗證的信賴區間，**無法判定**模型相對常數基準是否有優勢；"
            "請僅將機率視為參考。"
        )
    elif edge is False:
        lines.append(
            "- ⚠️ **整體而言，本模型相對「永遠猜震盪」的優勢未通過信賴區間檢定**"
            "（差異的 95% 區間跨越 0）。它的價值在於已校準的機率與高信心日的篩選，"
            "不在於每日方向本身。"
        )
    elif edge is True:
        lines.append("- 模型相對最強常數基準的優勢通過信賴區間檢定。")
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    generated = report["generated_at"]
    tw = report["taiwan"]
    us = report["us"]
    tw_ind = IndicatorSet(**tw["indicators"])
    spy = IndicatorSet(**us["series"]["SPY"]["indicators"])
    qqq = IndicatorSet(**us["series"]["QQQ"]["indicators"])
    predictions = report.get("predictions", {})
    tw_pred = predictions.get("taiwan", {})
    accuracy = report.get("accuracy", {})
    tw_acc = accuracy.get("taiwan", {})
    archived_acc = accuracy.get("archived_predictions", {})

    tw_levels = support_resistance_from_bars(tw["bars"])

    lines = [
        f"# 台股大盤日報 - {report['report_date']}",
        "",
        f"產生時間：{generated}",
        "",
        "## 資料日期",
        "",
        f"- 台股：{tw['latest_date']}（TWSE）",
        f"- 美股（僅作台股輸入訊號）：{us['latest_date']}（Nasdaq API / Cboe VIX）",
        "",
        "## 總結",
        "",
        f"- 台股：{tw_ind.trend_label}，{tw['volume_price_label']}，{tw['leverage_label']}。",
        f"- 隔夜美股脈絡：SPY {spy.trend_label}，QQQ {qqq.trend_label}，{us['vix']['label']}。",
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
            "- 台股理由：" + "；".join(tw_pred.get("reasons", [])[:4]),
            "- 預測限制：" + "；".join(tw_pred.get("missing_data", [])[:5]),
        ]
    )
    lines.extend(markdown_model_credibility(tw_pred))
    lines.extend(
        [
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
            "",
            "## 觀察價位",
            "",
            (
                f"- 台股：20日區間支撐 {fmt_num(tw_levels['support_20d'], 2)}，"
                f"20日區間壓力 {fmt_num(tw_levels['resistance_20d'], 2)}，"
                f"20日線 {fmt_num(tw_ind.ma20, 2)}，60日線 {fmt_num(tw_ind.ma60, 2)}。"
            ),
            "",
            "## 資料源",
            "",
            "- TWSE FMTQIK：台股每日成交資訊",
            "- TWSE MI_MARGN：上市信用交易統計",
            "- TWSE BFI82U：三大法人買賣金額統計",
            "- Nasdaq historical API + Cboe VIX：SPY / QQQ / VIX，僅作台股模型輸入特徵",
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

    # --- 0050 潛力股 screener section ---
    screener = report.get("screener_0050") or []
    lines.extend(["", "## 0050 潛力股", ""])
    lines.extend(markdown_intersection_block(screener))
    if not screener:
        lines.append("- 資料不足，無法顯示 0050 潛力股排名。")
    else:
        lines.append("| 名次 | 代號 | 名稱 | 綜合分 | 估值 | 營收 | 技術 | 動能 | 籌碼 | PE | 營收YoY% | 20日報酬 |")
        lines.append("|------|------|------|--------|------|------|------|------|------|-----|-----------|---------|")
        for item in screener[:15]:
            rank = item.get("rank", "")
            code = item.get("code", "")
            name = item.get("name") or code
            composite = item.get("composite")
            scores = item.get("scores") or {}
            raw = item.get("raw") or {}
            composite_str = f"{composite:.3f}" if composite is not None else "N/A"
            val_str = f"{scores.get('valuation'):.2f}" if scores.get('valuation') is not None else "N/A"
            rev_str = f"{scores.get('revenue'):.2f}" if scores.get('revenue') is not None else "N/A"
            tech_str = f"{scores.get('technical'):.2f}" if scores.get('technical') is not None else "N/A"
            mom_str = f"{scores.get('momentum'):.2f}" if scores.get('momentum') is not None else "N/A"
            chip_str = f"{scores.get('chip'):.2f}" if scores.get('chip') is not None else "N/A"
            pe_str = fmt_num(raw.get("pe"), 1)
            rev_yoy = raw.get("rev_yoy")
            rev_yoy_str = fmt_pct(rev_yoy / 100 if rev_yoy is not None else None)
            ret20_str = fmt_pct(raw.get("ret20"))
            lines.append(
                f"| {rank} | {code} | {name} | {composite_str} | {val_str} | {rev_str} | "
                f"{tech_str} | {mom_str} | {chip_str} | {pe_str} | {rev_yoy_str} | {ret20_str} |"
            )
        lines.append("")
        lines.append(
            "> 本表為橫截面基本面+技術面綜合排名啟發式評分，非經驗證的交易訊號，亦不構成任何投資建議。"
        )

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
    # US output removed (TW focus). US data (spy/qqq/vix) is kept only as overnight
    # context in the overview + as TW model input features — no US page/prediction.
    tw = report["taiwan"]
    us = report["us"]
    tw_ind = IndicatorSet(**tw["indicators"])
    spy = IndicatorSet(**us["series"]["SPY"]["indicators"])
    qqq = IndicatorSet(**us["series"]["QQQ"]["indicators"])
    tw_levels = support_resistance_from_bars(tw["bars"])
    financing = ((tw.get("margin") or {}).get("margin_financing") or {})
    institutional = tw.get("institutional")
    foreign_net = twse_net_by_name(institutional, "外資及陸資(不含外資自營商)")
    trust_net = twse_net_by_name(institutional, "投信")
    total_net = twse_net_by_name(institutional, "合計")
    cross_lines = infer_cross_market(tw, us)
    predictions = report.get("predictions", {})
    tw_pred = predictions.get("taiwan", {})
    accuracy = report.get("accuracy", {})
    tw_acc = accuracy.get("taiwan", {})
    archived_acc = accuracy.get("archived_predictions", {})
    improvement_log = accuracy.get("improvement_log", [])
    stock_2330_section = html_stock_2330_section(report.get("stock_2330"))
    screener_0050_section = html_screener_0050_section(report.get("screener_0050"))

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
                "台股融資/融券、三大法人",
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
  <title>台股大盤日報 - {esc(report['report_date'])}</title>
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
        <strong>台股大盤日報</strong>
      </div>
      <nav class="nav" aria-label="report pages">
        <a href="#overview" class="active">總覽</a>
        <a href="#prediction">預測</a>
        <a href="#accuracy">準確率</a>
        <a href="#taiwan">台股</a>
        <a href="#cross">回推</a>
        <a href="#stock2330">台積電 2330</a>
        <a href="#screener0050">0050 潛力股</a>
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
          <h1>台股大盤日報</h1>
          <div class="subtle">台股資料日 {esc(tw['latest_date'])}（隔夜美股 {esc(us['latest_date'])} 僅作台股輸入訊號）</div>
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
              f"隔夜美股脈絡：SPY {spy.trend_label}，QQQ {qqq.trend_label}，{us['vix']['label']}。",
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
          {bullet_list(unique_strings(tw_acc.get("limitations", []) + archived_acc.get("limitations", []))[:8])}
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
              "隔夜美股（SPY/QQQ/VIX）作為台股開盤的領先訊號，已納入台股模型輸入。",
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
              "Nasdaq historical API + Cboe VIX：SPY / QQQ / VIX，僅作台股模型輸入特徵",
          ])}
          <p class="subtle">註：這是規則式市場分析，不是投資建議。</p>
        </div>
      </section>
      {stock_2330_section}
      {screener_0050_section}
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


def _pick_query_date(
    daily_bars: list,
    fetch_fn,
    stock_no: str,
    max_lookback: int = 3,
) -> Optional[tuple[str, Any]]:
    """Walk back through the tail of daily_bars (up to max_lookback bars) and
    return the first (date_iso, result) for which fetch_fn returns non-None.

    This handles same-day publication lag: TWSE T86/BWIBBU data for the latest
    trading day is often not posted until after ~16:00, so we fall back to the
    previous bar if the latest returns None.

    Returns (date_iso, result) or (date_iso, None) for the latest bar if all
    attempts return None.
    """
    tail = daily_bars[-(max_lookback):] if len(daily_bars) >= max_lookback else daily_bars
    for bar in reversed(tail):
        result = fetch_fn(stock_no, bar.date)
        if result is not None:
            return bar.date, result
    # All returned None; return latest date so callers know which date was tried
    return daily_bars[-1].date, None

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

    # Attach full technical_metrics for display in HTML (all indicators)
    scorecard["technical_metrics"] = tech_m

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
        daily_bars = _stocks.fetch_stock_history("2330", end, lookback_months=14)
        if not daily_bars:
            raise ValueError("empty 2330 price history")
    except Exception as exc:  # noqa: BLE001
        return {"error": f"2330 price fetch failed: {exc}"}

    # Use the latest ACTUAL trading bar date for per-stock TWSE queries.
    # Today's data (end_iso) is often not posted until after ~16:00, so we
    # walk back up to 3 bars to find the most recent date with real data.

    # --- T86 institutional chip (best-effort) ---
    t86: Optional[dict[str, Any]] = None
    t86_date: str = end_iso
    try:
        t86_date, t86 = _pick_query_date(daily_bars, _stocks.fetch_t86_for_stock, "2330")
    except Exception:  # noqa: BLE001
        pass

    # --- Valuation PE/PB (best-effort) ---
    valuation_raw: Optional[dict[str, Any]] = None
    val_date: str = end_iso
    try:
        val_date, valuation_raw = _pick_query_date(
            daily_bars, _stocks.fetch_valuation_for_stock, "2330"
        )
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
            # Attach chip data to the bar whose date matches the resolved chip date
            if t86 is not None and bar.date == t86_date:
                row.update({
                    "foreign_net_lots": t86.get("foreign_net_lots"),
                    "trust_net_lots": t86.get("trust_net_lots"),
                    "dealer_net_lots": t86.get("dealer_net_lots"),
                    "inst_total_net_lots": t86.get("inst_total_net_lots"),
                })
            store.upsert_stock_daily(row)

        # Upsert valuation using the date _pick_query_date actually resolved
        if valuation_raw is not None:
            store.upsert_stock_valuation({
                "stock_no": "2330",
                "date": val_date,
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


# ---------------------------------------------------------------------------
# 0050 潛力股 screener integration
# ---------------------------------------------------------------------------


def assemble_screener(
    codes: list[str],
    valuation_by_code: dict[str, dict],
    revenue_by_code: dict[str, dict],
    chip_by_code: dict[str, dict],
    tech_by_code: dict[str, dict],
) -> list[dict]:
    """Pure assembly: map per-code dicts → build_screener records → return ranked list.

    Args:
        codes: list of stock codes to include.
        valuation_by_code: {code: {pe, pb, dividend_yield}} (may be empty or missing keys).
        revenue_by_code:   {code: {yoy_pct, ...}} (may be empty or missing keys).
        chip_by_code:      {code: {inst_total_net_lots, ...}} (may be empty).
        tech_by_code:      {code: {tech_raw, ret20, ret60, ...}} (may be empty).

    Returns:
        Ranked list from build_screener (composite DESC, rank 1 = strongest).
    """
    records: list[dict] = []
    for c in codes:
        val = valuation_by_code.get(c) or {}
        rev = revenue_by_code.get(c) or {}
        chip = chip_by_code.get(c) or {}
        tech = tech_by_code.get(c) or {}
        name = _stocks.resolve_name(c, valuation_by_code, chip_by_code)
        records.append({
            "code": c,
            "name": name,
            "pe": val.get("pe"),
            "pb": val.get("pb"),
            "dividend_yield": val.get("dividend_yield"),
            "rev_yoy": rev.get("yoy_pct"),
            "chip_net_lots": chip.get("inst_total_net_lots"),
            "tech_raw": tech.get("tech_raw"),
            "ret20": tech.get("ret20"),
            "ret60": tech.get("ret60"),
            "intersection": tech.get("intersection"),
            "river": tech.get("river"),
            "bars_asof": tech.get("bars_asof"),
            "bars_count": tech.get("bars_count"),
        })
    ranked = _screener.build_screener(records)
    # Re-attach name and the technical intersection to ranked rows (build_screener
    # only carries the fields it ranks on).
    by_code = {r["code"]: r for r in records}
    for row in ranked:
        src = by_code.get(row.get("code", ""), {})
        if "name" not in row or not row["name"]:
            row["name"] = src.get("name", row.get("code", ""))
        row["intersection"] = src.get("intersection")
        row["river"] = src.get("river")
        row["bars_asof"] = src.get("bars_asof")
        row["bars_count"] = src.get("bars_count")
    return ranked


def build_screener_0050(end: date, store: "HistoryStore") -> list[dict]:
    """Fetch all inputs for TW50 stocks, compute tech metrics, delegate to assemble_screener.

    Batch-fetches valuation, T86 chip, and revenue data ONCE (all-market).
    Per-code: reads daily history from store cache; fetches from network only
    when cache is stale or missing.  Adds a short sleep between network STOCK_DAY
    fetches to be gentle on the TWSE API (~50 stocks).

    Returns ranked screener list or [] on total failure.
    """
    codes = TW50_CONSTITUENTS
    end_iso = end.isoformat()
    # Stored valuation history, read once: it backs both the river band and the
    # stale-row fallback below.
    valuation_history: dict[str, list[dict]] = {}
    for c in codes:
        try:
            valuation_history[c] = store.get_stock_valuation(c) or []
        except Exception:  # noqa: BLE001
            valuation_history[c] = []

    # --- Batch fetch: valuation, T86 chip, revenue (ONE request each) ---
    # For valuation and T86: try latest date, fall back to prior day if empty
    valuation_by_code: dict[str, dict] = {}
    chip_by_code: dict[str, dict] = {}
    try:
        # Walk back up to 3 days to find a date with data
        for delta in range(3):
            try_date = (end - timedelta(days=delta)).isoformat()
            valuation_by_code = _stocks.fetch_valuation_all(try_date, codes)
            if valuation_by_code:
                break
    except Exception:  # noqa: BLE001
        valuation_by_code = {}

    # A Monday (or post-holiday) fetch finds nothing in a three-day window, which
    # used to drop the valuation facet entirely. Carry the last stored row
    # forward instead, stamped with its real date.
    valuation_by_code = _screener.merge_valuation_with_fallback(
        codes, valuation_by_code, valuation_history, asof=end_iso)

    try:
        for delta in range(3):
            try_date = (end - timedelta(days=delta)).isoformat()
            chip_by_code = _stocks.fetch_t86_all(try_date, codes)
            if chip_by_code:
                break
    except Exception:  # noqa: BLE001
        chip_by_code = {}

    revenue_by_code: dict[str, dict] = {}
    try:
        revenue_by_code = _stocks.fetch_monthly_revenue_all(codes)
    except Exception:  # noqa: BLE001
        revenue_by_code = {}

    # --- Per-code: history from store cache; fetch+upsert if stale/missing ---
    tech_by_code: dict[str, dict] = {}
    # Freshness is decided against the last COMPLETED trading day, not a
    # 20-calendar-day tolerance. The old threshold (end - 20 days) accepted bars
    # from 2026-09-01 on 2026-09-14 as current, so the screener published prices
    # nine trading days stale and never refetched. Anything short of the latest
    # session now triggers a fetch.
    expected_asof = _stocks.last_completed_trading_day(end).isoformat()

    for code in codes:
        try:
            cached_rows = store.get_stock_daily(code)
            need_fetch = (
                not cached_rows
                or cached_rows[-1]["date"] < expected_asof
                or len(cached_rows) < _stocks.MIN_BARS_FOR_WEEKLY_TREND
            )

            daily_bars: list = []
            if need_fetch:
                time.sleep(0.6)
                # 5 months left ~126 bars: MA240 impossible and only 27 weekly bars,
                # far short of what the weekly MACD needs to converge.
                fetched_bars = _stocks.fetch_stock_history(code, end, lookback_months=24)
                if fetched_bars:
                    for bar in fetched_bars:
                        store.upsert_stock_daily({
                            "stock_no": code,
                            "date": bar.date,
                            "close": bar.close,
                            "high": bar.high,
                            "low": bar.low,
                            "open": bar.open,
                            "volume": bar.volume,
                        })
                    daily_bars = fetched_bars
                else:
                    # Fall back to whatever is in the store
                    from marketlib.core import MarketBar
                    daily_bars = [
                        MarketBar(
                            date=r["date"],
                            close=r["close"],
                            high=r.get("high"),
                            low=r.get("low"),
                            open=r.get("open"),
                            volume=r.get("volume"),
                        )
                        for r in cached_rows if r.get("close") is not None
                    ]
            else:
                from marketlib.core import MarketBar
                daily_bars = [
                    MarketBar(
                        date=r["date"],
                        close=r["close"],
                        high=r.get("high"),
                        low=r.get("low"),
                        open=r.get("open"),
                        volume=r.get("volume"),
                    )
                    for r in cached_rows if r.get("close") is not None
                ]

            if daily_bars:
                tech_raw = _stocks.tech_raw_score(daily_bars)
                tech_m = _stocks.technical_metrics(daily_bars)
                tech_by_code[code] = {
                    "tech_raw": tech_raw,
                    "ret20": tech_m.get("ret20"),
                    "ret60": tech_m.get("ret60"),
                    "intersection": _screener.technical_intersection(
                        tech_m,
                        weekly_bars=len(_stocks.resample_weekly(daily_bars)),
                        discontinuities=_stocks.detect_price_discontinuities(
                            daily_bars, lookback=_stocks.INDICATOR_SPAN_BARS),
                        bars_asof=daily_bars[-1].date if daily_bars else None,
                        expected_asof=expected_asof,
                    ),
                    "bars_asof": daily_bars[-1].date if daily_bars else None,
                    "bars_count": len(daily_bars),
                }
                # Where today's multiple sits inside this stock's own recent
                # range — the question a PE river chart answers by eye.
                try:
                    hist = valuation_history.get(code) or []
                    val_now = valuation_by_code.get(code) or {}
                    tech_by_code[code]["river"] = _stocks.river_valuation(
                        code,
                        pe=val_now.get("pe"),
                        pb=val_now.get("pb"),
                        pe_history=[r.get("pe") for r in hist],
                        pb_history=[r.get("pb") for r in hist],
                        splits=_stocks.detect_valuation_splits(
                            hist[-_stocks.VALUATION_LOOKBACK_DAYS:]),
                    )
                    tech_by_code[code]["river"]["asof"] = val_now.get("valuation_asof")
                    tech_by_code[code]["river"]["is_stale"] = val_now.get(
                        "valuation_is_stale")
                except Exception:  # noqa: BLE001 — valuation never blocks the screen
                    tech_by_code[code]["river"] = None
        except Exception:  # noqa: BLE001
            # One failure never aborts the whole screener
            pass

    return assemble_screener(codes, valuation_by_code, revenue_by_code, chip_by_code, tech_by_code)


def html_intersection_panel(screener_data: Optional[list[dict]]) -> str:
    """HTML twin of markdown_intersection_block — the shortlist plus its reasons."""
    if not screener_data:
        return ""
    scored = [r for r in screener_data if r.get("intersection")]
    if not scored:
        return ('<div class="panel"><h2>技術面交集</h2>'
                '<p class="subtle">個股技術指標資料不足，無法計算交集。</p></div>')

    hits = _screener.rank_candidates(scored)
    core_total = scored[0]["intersection"]["core_total"]
    core_list = "".join(f'<li>{esc(t)}</li>' for t in _screener.core_condition_texts())
    head = (
        f'<p class="subtle">必要條件 {core_total} 項全數成立才列入；'
        f'RSI ≥ {_screener.RSI_OVERHEATED:.0f} 視為過熱直接排除。'
        f'共 {len(scored)} 檔納入檢查，<strong>{len(hits)}</strong> 檔通過。</p>'
        f'<p class="subtle">必要條件（通過者一律全數成立，故僅列一次）：</p>'
        f'<ol class="subtle">{core_list}</ol>'
    )

    if not hits:
        near = sorted(scored, key=lambda r: len(r["intersection"]["core_failed"]))[:5]
        items = "".join(
            f'<li><strong>{esc(r.get("code",""))} {esc(r.get("name",""))}</strong>'
            f'（必要條件 {r["intersection"]["core_passed"]}/{r["intersection"]["core_total"]}）'
            f'<br><span class="subtle">未過：'
            f'{esc("；".join(r["intersection"]["core_failed"] + r["intersection"]["vetoes"]) or "—")}'
            f'</span></li>'
            for r in near
        )
        return ('<div class="panel"><h2>技術面交集</h2>' + head +
                '<p><strong>本日無個股通過全部條件。</strong> 最接近的幾檔與其卡關處：</p>'
                f'<ul>{items}</ul></div>')

    body_rows = ""
    for r in hits:
        ins = r["intersection"]
        m = ins["metrics_snapshot"]
        kd_str = (f'{m["kd_k"]:.0f}/{m["kd_d"]:.0f}'
                  if m.get("kd_k") is not None and m.get("kd_d") is not None else "N/A")
        missing = "；".join(ins["bonus_failed"])
        riv = r.get("river") or {}
        band = riv.get("band")
        if band:
            band_str = f'{band}（{riv.get("primary_metric")}）'
        elif riv.get("split_factor"):
            band_str = f'分割未還原（{riv["split_factor"]:.2g}:1）'
        else:
            band_str = "資料不足"
        pct_str = (f'{riv["percentile"]:.0%}' if riv.get("percentile") is not None else "—")
        body_rows += (
            f'<tr><td><strong>{ins.get("shortlist_rank","—")}</strong></td>'
            f'<td>{esc(r.get("code",""))}</td><td>{esc(r.get("name",""))}</td>'
            f'<td>{esc(fmt_num(m.get("close"), 2))}</td>'
            f'<td>{esc(band_str)}</td><td>{esc(pct_str)}</td>'
            f'<td>{esc(fmt_num(riv.get("pe"), 2))}</td>'
            f'<td>{esc(fmt_num(riv.get("pb"), 2))}</td>'
            f'<td>{ins["bonus_passed"]}/{ins["bonus_total"]}</td>'
            f'<td>{r.get("rank","—")}</td>'
            f'<td>{esc(fmt_pct(m.get("close_vs_ma20")))}</td>'
            f'<td>{esc(kd_str)}</td>'
            f'<td>{esc(fmt_num(m.get("rsi"), 0))}</td>'
            f'<td>{esc(fmt_num(m.get("macd_hist"), 2))}</td>'
            f'<td>{esc(fmt_num(m.get("pctb"), 2))}</td>'
            f'<td>{esc(fmt_num(m.get("volume_ratio_20d"), 2))}</td>'
            f'<td class="subtle">{esc(missing or "—")}</td>'
            f'<td class="subtle">{esc("；".join(ins["warnings"]) or "—")}</td></tr>'
        )
    blocks = (
        '<div style="overflow-x:auto"><table><thead><tr>'
        '<th>排名</th><th>代號</th><th>名稱</th><th>股價</th>'
        '<th>河流圖位階</th><th>分位</th><th>本益比</th><th>股價淨值比</th>'
        '<th>加分</th><th>綜排</th><th>距月線</th>'
        '<th>KD (K/D)</th><th>RSI</th><th>MACD柱</th><th>布林%b</th><th>量比</th>'
        '<th>加分缺項</th><th>延伸過度提醒</th>'
        f'</tr></thead><tbody>{body_rows}</tbody></table></div>'
    )
    return ('<div class="panel"><h2>技術面交集</h2>' + head + blocks +
            '<p class="subtle">交集只描述技術面目前的狀態，不含未來報酬的統計驗證；'
            '本清單未做過勝率回測，請當作篩選起點而非買賣訊號。'
            '排名依「延伸過度警訊少 → 加分項多 → 綜合排名前」，'
            '反映今日技術面型態的乾淨程度，不是報酬預測。'
            '河流圖位階為該股自身近 5 年估值分布的百分位（越低越便宜）；'
            '金融股與無可用 PE 者改以 PB 判讀。</p></div>')


def html_screener_0050_section(screener_data: Optional[list[dict]]) -> str:
    """Render the 0050 潛力股 screener HTML section."""
    disclaimer = (
        "本頁為橫截面基本面+技術面綜合排名啟發式評分，非經驗證的交易訊號，亦不構成任何投資建議。"
        " 排名僅反映當期各股相對強弱，不代表絕對報酬預測。"
    )

    if not screener_data:
        return (
            '<section id="screener0050" class="page">'
            '<div class="hero"><h1>0050 潛力股</h1></div>'
            '<div class="panel"><h2>資料不足</h2>'
            f'<p class="subtle">{esc(disclaimer)}</p></div>'
            '</section>'
        )

    rows_html = ""
    for item in screener_data:
        rank = item.get("rank", "")
        code = item.get("code", "")
        name = item.get("name") or code
        composite = item.get("composite")
        scores = item.get("scores") or {}
        raw = item.get("raw") or {}

        composite_str = f"{composite:.3f}" if composite is not None else "N/A"
        val_str = f"{scores.get('valuation'):.2f}" if scores.get('valuation') is not None else "N/A"
        rev_str = f"{scores.get('revenue'):.2f}" if scores.get('revenue') is not None else "N/A"
        tech_str = f"{scores.get('technical'):.2f}" if scores.get('technical') is not None else "N/A"
        mom_str = f"{scores.get('momentum'):.2f}" if scores.get('momentum') is not None else "N/A"
        chip_str = f"{scores.get('chip'):.2f}" if scores.get('chip') is not None else "N/A"
        pe_str = fmt_num(raw.get("pe"), 1)
        rev_yoy_str = fmt_pct(raw.get("rev_yoy") / 100 if raw.get("rev_yoy") is not None else None)
        ret20_str = fmt_pct(raw.get("ret20"))

        rows_html += (
            "<tr>"
            f"<td>{esc(str(rank))}</td>"
            f"<td>{esc(code)}</td>"
            f"<td>{esc(name)}</td>"
            f"<td><strong>{esc(composite_str)}</strong></td>"
            f"<td>{esc(val_str)}</td>"
            f"<td>{esc(rev_str)}</td>"
            f"<td>{esc(tech_str)}</td>"
            f"<td>{esc(mom_str)}</td>"
            f"<td>{esc(chip_str)}</td>"
            f"<td>{esc(pe_str)}</td>"
            f"<td>{esc(rev_yoy_str)}</td>"
            f"<td>{esc(ret20_str)}</td>"
            "</tr>"
        )

    return (
        '<section id="screener0050" class="page">'
        '<div class="hero">'
        '<h1>0050 潛力股</h1>'
        '<div class="subtle">台灣50成分股橫截面多因子排名：估值 / 營收 / 技術 / 動能 / 籌碼。</div>'
        '</div>'
        + html_intersection_panel(screener_data) +
        '<div class="panel">'
        '<h2>綜合排名</h2>'
        '<div style="overflow-x:auto">'
        '<table>'
        '<thead><tr>'
        '<th>名次</th><th>代號</th><th>名稱</th><th>綜合分</th>'
        '<th>估值</th><th>營收</th><th>技術</th><th>動能</th><th>籌碼</th>'
        '<th>PE</th><th>營收YoY%</th><th>20日報酬</th>'
        '</tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        '</table>'
        '</div>'
        f'<p class="subtle">{esc(disclaimer)}</p>'
        '</div>'
        '</section>'
    )


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

        # 0050 screener (best-effort; never aborts index report)
        try:
            report["screener_0050"] = build_screener_0050(report_date, store)
        except Exception as exc:  # noqa: BLE001
            report["screener_0050"] = []
    finally:
        store.close()

    # Overlay model predictions on top of rule-based predictions where available.
    # US removed from output; only TW is predicted (US rows still feed TW features).
    for market_key, market_label, model_file, market_rows in (
        ("taiwan", "tw", "model_tw.json", tw_rows),
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
            # Carry the two-stage extras through so the reader sees the full
            # distribution and the measured track record behind today's tier,
            # not just a single word.
            for key in ("class_probabilities", "tier_evidence", "oos_reference",
                        "edge_demonstrated"):
                if mp.get(key) is not None:
                    report["predictions"][market_key][key] = mp[key]
            if not mp.get("confidence_calibrated", True):
                md = list(report["predictions"][market_key].get("missing_data") or [])
                md.append("信心為未校準機率（原始 sigmoid，僅供方向參考）")
                report["predictions"][market_key]["missing_data"] = md
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
