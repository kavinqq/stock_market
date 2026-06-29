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
