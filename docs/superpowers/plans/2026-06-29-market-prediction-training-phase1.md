# 台美股預測訓練系統 Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把手填規則式預測,升級為「用多年歷史資料、point-in-time 正確、walk-forward 訓練、輸出上漲機率」的 logistic regression 模型,並整合進每日報告(每日 runtime 維持純標準庫)。

**Architecture:** 離線(可用 sklearn)回填 SQLite 歷史庫 → 共用 `features.py` 算 causal 特徵 → walk-forward 訓練/選模/調門檻 → 匯出權重 JSON;每日 cron 純標準庫讀 JSON 做推論。特徵工程**單一來源**(離線/線上共用)以杜絕 train/serve skew。

**Tech Stack:** Python 3.9+(stdlib:`sqlite3`、`json`、`statistics`、`datetime`、`zoneinfo`、`unittest`);離線訓練額外用 `numpy` + `scikit-learn`(僅在 venv,不進 cron runtime)。

## Global Constraints

- **每日 runtime(`daily_market_report.py` + `marketlib/` 的 runtime 模組:`core`、`evaluation`、`stats`、`store`、`features`、`model`)只能 import 標準庫。** 嚴禁在這些模組頂層 import numpy / sklearn / pandas。
- **離線模組(`train.py`)可用 numpy / scikit-learn**,須在 venv 安裝:`python3 -m venv .venv && .venv/bin/pip install numpy scikit-learn`。離線指令一律用 `.venv/bin/python`。
- **測試框架:`unittest`(標準庫)**,執行:`python3 -m unittest -v tests.<module>`;需 sklearn 的測試用 `.venv/bin/python -m unittest -v tests.test_train`。
- **時區一律 Asia/Taipei**;所有 `*_asof_taipei` / `available_at_taipei` 以 ISO8601 含時區字串存(例:`2026-06-29T18:30:00+08:00`),比較時 parse 成 `datetime`。
- **decision_time = 執行日 D 的 18:30 Taipei**;特徵只能用 `asof <= decision_time` 的值(細到欄位群)。
- **單一特徵來源**:任何特徵計算只能在 `marketlib/features.py`,離線與線上都呼叫它。
- **DIRECTION_LABELS = ["偏多", "震盪", "偏空"]**(沿用現有字串,勿改)。
- 每個 Task 結束都要 commit。本資料夾若非 git repo,先 `git init`(Task 0)。

---

## File Structure

- `marketlib/__init__.py` — 空 package marker
- `marketlib/core.py` — `MarketBar`、`IndicatorSet`、`clean_num`、`pct_change`、`simple_moving_average`、`compute_indicators`(自 `daily_market_report.py` 抽出)
- `marketlib/evaluation.py` — `DIRECTION_LABELS`、`DEFAULT_NEUTRAL_THRESHOLD` 等常數、`direction_from_return`、`neutral_threshold_from_bars`、`summarize_classification`、`summarize_baselines`(抽出)
- `marketlib/stats.py` —(新)`mcnemar_test`、`block_bootstrap_diff`(成對顯著性)
- `marketlib/store.py` —(新)`HistoryStore`:SQLite schema / upsert / query / ingest 游標
- `marketlib/fetchers.py` — 抓資料函式(自 `daily_market_report.py` 抽出)+ asof 標註 helper
- `marketlib/features.py` —(新)decision_time gating + 視窗政策 + `build_feature_row` / `build_dataset`(單一來源)
- `marketlib/model.py` —(新)純標準庫 LR 推論
- `backfill.py` —(新,離線)多來源歷史回填 → SQLite,含 asof 中繼欄位、續傳
- `train.py` —(新,離線,sklearn)walk-forward 訓練/選模/調門檻 → `models/model_*.json` + `models/metrics_*.json`
- `daily_market_report.py` —(改)upsert 進 store、用 `features.py` 算特徵、`model.py` 推論、規則式保留為次要意見 + fallback
- `tests/` — `test_core.py`、`test_evaluation.py`、`test_stats.py`、`test_store.py`、`test_fetchers.py`、`test_backfill.py`、`test_features.py`、`test_model.py`、`test_train.py`、`test_daily.py`、`test_parity.py`

---

## Task 0: Repo init 與 package scaffold

**Files:**
- Create: `marketlib/__init__.py`, `tests/__init__.py`, `.gitignore`

- [ ] **Step 1: 初始化 git 並建立目錄**

```bash
cd /Users/kuanlin/stock_market
git rev-parse --is-inside-work-tree 2>/dev/null || git init
mkdir -p marketlib tests models docs/superpowers/plans
printf '%s\n' '.venv/' 'data/history.db' 'data/history.db-*' '__pycache__/' '*.pyc' > .gitignore
touch marketlib/__init__.py tests/__init__.py
```

- [ ] **Step 2: 建立離線 venv 並安裝訓練相依**

```bash
python3 -m venv .venv && .venv/bin/pip install -q numpy scikit-learn
.venv/bin/python -c "import numpy, sklearn; print('ok', sklearn.__version__)"
```
Expected: 印出 `ok <version>`

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "chore: scaffold marketlib package and offline training venv"
```

---

## Task 1: 抽出 `marketlib/core.py`(技術指標基元)

**Files:**
- Create: `marketlib/core.py`
- Modify: `daily_market_report.py`(改為 `from marketlib.core import ...`,刪除原定義)
- Test: `tests/test_core.py`

**Interfaces:**
- Produces: `MarketBar`(dataclass,欄位同現行)、`IndicatorSet`(dataclass,欄位同現行)、`clean_num(value)->Optional[float]`、`pct_change(current, previous)->Optional[float]`、`simple_moving_average(values:list[float], window:int)->Optional[float]`、`compute_indicators(bars:list[MarketBar], volume_attr:str="volume")->IndicatorSet`

- [ ] **Step 1: 寫失敗測試**

```python
# tests/test_core.py
import unittest
from marketlib.core import (
    MarketBar, clean_num, pct_change, simple_moving_average, compute_indicators,
)

class TestCore(unittest.TestCase):
    def test_clean_num_strips_symbols(self):
        self.assertEqual(clean_num("1,234.5"), 1234.5)
        self.assertIsNone(clean_num("--"))

    def test_pct_change(self):
        self.assertAlmostEqual(pct_change(110, 100), 0.1)
        self.assertIsNone(pct_change(110, 0))

    def test_sma(self):
        self.assertEqual(simple_moving_average([1, 2, 3, 4], 2), 3.5)
        self.assertIsNone(simple_moving_average([1], 2))

    def test_volume_ratio_excludes_today(self):
        # 21 根:前 20 根量=100,今日量=200 -> ratio=2.0(分母不含今日)
        bars = [MarketBar(date=f"2026-01-{i+1:02d}", close=10.0, volume=100.0) for i in range(20)]
        bars.append(MarketBar(date="2026-02-01", close=10.0, volume=200.0))
        ind = compute_indicators(bars, volume_attr="volume")
        self.assertAlmostEqual(ind.volume_ratio_20d, 2.0)

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `python3 -m unittest -v tests.test_core`
Expected: FAIL(`ModuleNotFoundError: marketlib.core`)

- [ ] **Step 3: 建立 `marketlib/core.py`**

把 `daily_market_report.py` 內這些定義**原樣剪下**移入 `marketlib/core.py`:`MarketBar`、`IndicatorSet`、`clean_num`、`fmt_*` 中與計算無關的留在原檔(只移 `clean_num`、`pct_change`、`simple_moving_average`、`compute_indicators` 與兩個 dataclass)。`core.py` 開頭需要的 import:

```python
from __future__ import annotations
import html, math, re
from dataclasses import dataclass
from statistics import mean
from typing import Any, Optional
```

`compute_indicators` 維持現行邏輯(含已修正的「前 20 日均量、不含今日」:`prior_vols = [...for bar in bars[:-1]]`、`avg20_vol = mean(prior_vols[-20:])`)。

- [ ] **Step 4: 改 `daily_market_report.py` 改用 import**

刪除原檔內已移出的定義,頂部加:
```python
from marketlib.core import (
    MarketBar, IndicatorSet, clean_num, pct_change,
    simple_moving_average, compute_indicators,
)
```

- [ ] **Step 5: 跑測試 + 冒煙測試原腳本可 import**

Run:
```bash
python3 -m unittest -v tests.test_core
python3 -c "import daily_market_report"
```
Expected: 測試 PASS;import 無錯

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "refactor: extract indicator primitives into marketlib.core"
```

---

## Task 2: 抽出 `marketlib/evaluation.py`(分類/baseline 指標)

**Files:**
- Create: `marketlib/evaluation.py`
- Modify: `daily_market_report.py`(改 import)
- Test: `tests/test_evaluation.py`

**Interfaces:**
- Produces: `DIRECTION_LABELS:list[str]`、`DEFAULT_NEUTRAL_THRESHOLD:float`、`MIN_NEUTRAL_THRESHOLD`、`MAX_NEUTRAL_THRESHOLD`、`NEUTRAL_VOL_MULTIPLIER`、`direction_from_return(ret, threshold=None)->str`、`neutral_threshold_from_bars(bars, lookback=60)->float`、`summarize_classification(records)->dict`、`summarize_baselines(records)->dict`

- [ ] **Step 1: 寫失敗測試**

```python
# tests/test_evaluation.py
import unittest
from marketlib.evaluation import (
    DIRECTION_LABELS, direction_from_return, summarize_classification,
)

class TestEvaluation(unittest.TestCase):
    def test_direction_threshold(self):
        self.assertEqual(direction_from_return(0.01, 0.003), "偏多")
        self.assertEqual(direction_from_return(-0.01, 0.003), "偏空")
        self.assertEqual(direction_from_return(0.001, 0.003), "震盪")

    def test_macro_f1_present(self):
        records = [
            {"actual_direction": "偏多", "prediction": "偏多"},
            {"actual_direction": "偏空", "prediction": "偏多"},
        ]
        out = summarize_classification(records)
        self.assertIn("macro_f1", out)
        self.assertEqual(out["confusion_matrix"]["偏空"]["偏多"], 1)

    def test_labels(self):
        self.assertEqual(DIRECTION_LABELS, ["偏多", "震盪", "偏空"])

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `python3 -m unittest -v tests.test_evaluation`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 建立 `marketlib/evaluation.py`**

把 `daily_market_report.py` 內 `DIRECTION_LABELS`、`DEFAULT_NEUTRAL_THRESHOLD`、`MIN/MAX_NEUTRAL_THRESHOLD`、`NEUTRAL_VOL_MULTIPLIER`、`direction_from_return`、`neutral_threshold_from_bars`、`close_return_series`、`bar_close`、`summarize_classification`、`summarize_baselines` 原樣移入。頂部 import:
```python
from statistics import mean, pstdev
from typing import Any, Optional
```

- [ ] **Step 4: 改 `daily_market_report.py` import 這些名稱**(刪原定義)

```python
from marketlib.evaluation import (
    DIRECTION_LABELS, DEFAULT_NEUTRAL_THRESHOLD,
    direction_from_return, neutral_threshold_from_bars,
    summarize_classification, summarize_baselines,
)
```

- [ ] **Step 5: 跑測試 + 冒煙**

Run:
```bash
python3 -m unittest -v tests.test_evaluation
python3 -c "import daily_market_report"
```
Expected: PASS;import 無錯

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "refactor: extract classification/baseline metrics into marketlib.evaluation"
```

---

## Task 3: `marketlib/stats.py` — McNemar + block bootstrap(成對顯著性)

**Files:**
- Create: `marketlib/stats.py`
- Test: `tests/test_stats.py`

**Interfaces:**
- Produces:
  - `mcnemar_test(model_hits:list[bool], baseline_hits:list[bool])->dict` 回 `{"b":int,"c":int,"statistic":float,"p_value":float}`(b=模型對/baseline 錯,c=模型錯/baseline 對;用連續性校正卡方,df=1)
  - `block_bootstrap_diff(model_hits, baseline_hits, block:int=5, n_resamples:int=2000, seed:int=0)->dict` 回 `{"mean_diff":float,"ci_low":float,"ci_high":float}`(差=模型命中率−baseline 命中率,95% CI)

- [ ] **Step 1: 寫失敗測試**

```python
# tests/test_stats.py
import unittest
from marketlib.stats import mcnemar_test, block_bootstrap_diff

class TestStats(unittest.TestCase):
    def test_mcnemar_counts(self):
        model =    [True,  True,  False, True]
        baseline = [False, True,  False, False]
        out = mcnemar_test(model, baseline)
        self.assertEqual(out["b"], 2)   # model right, baseline wrong (idx 0,3)
        self.assertEqual(out["c"], 0)
        self.assertIn("p_value", out)

    def test_bootstrap_positive_when_model_better(self):
        model = [True] * 80 + [False] * 20
        baseline = [True] * 50 + [False] * 50
        out = block_bootstrap_diff(model, baseline, block=5, n_resamples=500, seed=1)
        self.assertGreater(out["mean_diff"], 0)
        self.assertGreater(out["ci_low"], 0)  # 顯著為正

    def test_bootstrap_deterministic(self):
        m = [True, False] * 50
        b = [False, True] * 50
        a = block_bootstrap_diff(m, b, seed=7, n_resamples=300)
        c = block_bootstrap_diff(m, b, seed=7, n_resamples=300)
        self.assertEqual(a, c)  # 同 seed 結果一致

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `python3 -m unittest -v tests.test_stats`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 實作 `marketlib/stats.py`**

```python
from __future__ import annotations
import math, random
from typing import Any

def mcnemar_test(model_hits: list[bool], baseline_hits: list[bool]) -> dict[str, Any]:
    if len(model_hits) != len(baseline_hits):
        raise ValueError("model_hits and baseline_hits must align by sample")
    b = sum(1 for m, x in zip(model_hits, baseline_hits) if m and not x)
    c = sum(1 for m, x in zip(model_hits, baseline_hits) if x and not m)
    if b + c == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0}
    stat = (abs(b - c) - 1) ** 2 / (b + c)  # continuity-corrected chi-square, df=1
    # df=1 chi-square survival = erfc(sqrt(stat/2))
    p = math.erfc(math.sqrt(stat / 2.0))
    return {"b": b, "c": c, "statistic": stat, "p_value": p}

def block_bootstrap_diff(
    model_hits: list[bool], baseline_hits: list[bool],
    block: int = 5, n_resamples: int = 2000, seed: int = 0,
) -> dict[str, float]:
    if len(model_hits) != len(baseline_hits):
        raise ValueError("inputs must align by sample")
    n = len(model_hits)
    diffs = [(1.0 if m else 0.0) - (1.0 if x else 0.0)
             for m, x in zip(model_hits, baseline_hits)]
    if n == 0:
        return {"mean_diff": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    point = sum(diffs) / n
    rng = random.Random(seed)
    n_blocks = math.ceil(n / block)
    means = []
    for _ in range(n_resamples):
        sample = []
        for _ in range(n_blocks):
            start = rng.randint(0, n - 1)
            for k in range(block):
                sample.append(diffs[(start + k) % n])
        sample = sample[:n]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = means[int(0.025 * len(means))]
    hi = means[min(len(means) - 1, int(0.975 * len(means)))]
    return {"mean_diff": round(point, 6), "ci_low": round(lo, 6), "ci_high": round(hi, 6)}
```

- [ ] **Step 4: 跑測試確認通過**

Run: `python3 -m unittest -v tests.test_stats`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: add paired significance tools (McNemar, block bootstrap)"
```

---

## Task 4: `marketlib/store.py` — SQLite 歷史庫(含 asof 中繼欄位)

**Files:**
- Create: `marketlib/store.py`
- Test: `tests/test_store.py`

**Interfaces:**
- Produces: class `HistoryStore`:
  - `HistoryStore(path:str)`、`init_schema()`
  - `upsert_tw_daily(row:dict)`、`upsert_us_daily(row:dict)`、`upsert_finra_monthly(row:dict)`(row key = 欄位名;缺的欄位存 NULL)
  - `get_tw_rows(start=None, end=None)->list[dict]`、`get_us_rows(...)`、`get_finra_rows()->list[dict]`(皆依日期升冪)
  - `mark_ingested(source:str, key:str)`、`is_ingested(source:str, key:str)->bool`
  - `close()`

- [ ] **Step 1: 寫失敗測試**

```python
# tests/test_store.py
import os, tempfile, unittest
from marketlib.store import HistoryStore

class TestStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "h.db")
        self.s = HistoryStore(self.db)
        self.s.init_schema()

    def tearDown(self):
        self.s.close()

    def test_upsert_and_query_tw(self):
        self.s.upsert_tw_daily({
            "date": "2026-06-26", "close": 100.0, "turnover": 5e11,
            "inst_total_net_ntd": 1e9, "inst_asof_taipei": "2026-06-26T16:00:00+08:00",
            "price_asof_taipei": "2026-06-26T14:30:00+08:00",
            "available_at_taipei": "2026-06-26T16:00:00+08:00",
        })
        # upsert 同日覆蓋
        self.s.upsert_tw_daily({"date": "2026-06-26", "close": 101.0})
        rows = self.s.get_tw_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["close"], 101.0)
        self.assertEqual(rows[0]["inst_asof_taipei"], "2026-06-26T16:00:00+08:00")

    def test_ingest_cursor(self):
        self.assertFalse(self.s.is_ingested("tw_margin", "2026-06-26"))
        self.s.mark_ingested("tw_margin", "2026-06-26")
        self.assertTrue(self.s.is_ingested("tw_margin", "2026-06-26"))

    def test_date_range_query(self):
        for d in ("2026-06-01", "2026-06-10", "2026-06-20"):
            self.s.upsert_us_daily({"date": d, "spy_close": 1.0})
        rows = self.s.get_us_rows(start="2026-06-05", end="2026-06-15")
        self.assertEqual([r["date"] for r in rows], ["2026-06-10"])

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `python3 -m unittest -v tests.test_store`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 實作 `marketlib/store.py`**

```python
from __future__ import annotations
import sqlite3
from typing import Any, Optional

TW_COLUMNS = [
    "date", "close", "change_points", "turnover", "volume", "trades",
    "margin_balance_ntd", "margin_balance_change_ntd", "short_lots",
    "foreign_net_ntd", "trust_net_ntd", "dealer_net_ntd", "inst_total_net_ntd",
    "adv_count", "dec_count",
    "price_asof_taipei", "margin_asof_taipei", "inst_asof_taipei", "breadth_asof_taipei",
    "source_updated_at", "available_at_taipei", "quality_flags", "missing_fields",
]
US_COLUMNS = [
    "date", "spy_close", "spy_volume", "qqq_close", "qqq_volume",
    "iwm_close", "iwm_volume", "comp_close",
    "vix_close", "vix_open", "vix_high", "vix_low",
    "price_asof_taipei", "vix_asof_taipei",
    "source_updated_at", "available_at_taipei", "quality_flags", "missing_fields",
]
FINRA_COLUMNS = [
    "month", "debit_millions", "free_credit_cash_millions", "free_credit_margin_millions",
    "publish_date", "available_at_taipei", "source_updated_at", "quality_flags",
]

def _col_defs(cols: list[str], pk: str) -> str:
    parts = []
    for c in cols:
        if c == pk:
            parts.append(f"{c} TEXT PRIMARY KEY")
        elif c in {"date", "month"} or c.endswith("_taipei") or c in {
            "source_updated_at", "quality_flags", "missing_fields", "publish_date",
        }:
            parts.append(f"{c} TEXT")
        else:
            parts.append(f"{c} REAL")
    return ", ".join(parts)

class HistoryStore:
    def __init__(self, path: str):
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row

    def init_schema(self) -> None:
        c = self.conn
        c.execute(f"CREATE TABLE IF NOT EXISTS tw_daily ({_col_defs(TW_COLUMNS, 'date')})")
        c.execute(f"CREATE TABLE IF NOT EXISTS us_daily ({_col_defs(US_COLUMNS, 'date')})")
        c.execute(f"CREATE TABLE IF NOT EXISTS finra_monthly ({_col_defs(FINRA_COLUMNS, 'month')})")
        c.execute(
            "CREATE TABLE IF NOT EXISTS ingest_log "
            "(source TEXT, key TEXT, status TEXT, fetched_at TEXT, PRIMARY KEY(source, key))"
        )
        c.commit()

    def _upsert(self, table: str, cols: list[str], pk: str, row: dict) -> None:
        names = [c for c in cols if c in row]
        if pk not in names:
            raise ValueError(f"{table} upsert requires '{pk}'")
        placeholders = ", ".join("?" for _ in names)
        updates = ", ".join(f"{c}=excluded.{c}" for c in names if c != pk)
        sql = (
            f"INSERT INTO {table} ({', '.join(names)}) VALUES ({placeholders}) "
            f"ON CONFLICT({pk}) DO UPDATE SET {updates}"
        ) if updates else (
            f"INSERT OR IGNORE INTO {table} ({', '.join(names)}) VALUES ({placeholders})"
        )
        self.conn.execute(sql, [row[c] for c in names])
        self.conn.commit()

    def upsert_tw_daily(self, row: dict) -> None:
        self._upsert("tw_daily", TW_COLUMNS, "date", row)

    def upsert_us_daily(self, row: dict) -> None:
        self._upsert("us_daily", US_COLUMNS, "date", row)

    def upsert_finra_monthly(self, row: dict) -> None:
        self._upsert("finra_monthly", FINRA_COLUMNS, "month", row)

    def _query(self, table: str, order_col: str, start, end) -> list[dict]:
        sql = f"SELECT * FROM {table}"
        clauses, params = [], []
        if start is not None:
            clauses.append(f"{order_col} >= ?"); params.append(start)
        if end is not None:
            clauses.append(f"{order_col} <= ?"); params.append(end)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += f" ORDER BY {order_col} ASC"
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def get_tw_rows(self, start=None, end=None) -> list[dict]:
        return self._query("tw_daily", "date", start, end)

    def get_us_rows(self, start=None, end=None) -> list[dict]:
        return self._query("us_daily", "date", start, end)

    def get_finra_rows(self) -> list[dict]:
        return self._query("finra_monthly", "month", None, None)

    def mark_ingested(self, source: str, key: str, status: str = "ok") -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO ingest_log (source, key, status, fetched_at) "
            "VALUES (?, ?, ?, datetime('now'))", (source, key, status),
        )
        self.conn.commit()

    def is_ingested(self, source: str, key: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM ingest_log WHERE source=? AND key=? AND status='ok'",
            (source, key),
        ).fetchone()
        return row is not None

    def close(self) -> None:
        self.conn.close()
```

- [ ] **Step 4: 跑測試確認通過**

Run: `python3 -m unittest -v tests.test_store`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: add SQLite history store with point-in-time metadata columns"
```

---

## Task 5: 抽出 `marketlib/fetchers.py`(資料抓取 + asof 標註)

**Files:**
- Create: `marketlib/fetchers.py`
- Modify: `daily_market_report.py`(改 import 抓取函式)
- Test: `tests/test_fetchers.py`

**Interfaces:**
- Produces:(自現檔原樣移入)`http_get`、`http_get_json`、`curl_get`、`curl_get_bare`、所有 `fetch_twse_*`、`fetch_nasdaq_history`、`fetch_vix_history`、`fetch_finra_margin_statistics`、`roc_date_to_iso`、`iso_to_yyyymmdd`、`parse_us_date`、`month_starts`、常數 `TWSE_BASE`/`NASDAQ_*`/`CBOE_VIX_CSV`/`FINRA_MARGIN_URL`/`USER_AGENT`
- 新增 helper:`twse_asof(date_iso:str, group:str)->str`、`us_price_asof(date_iso:str)->str`、`vix_asof(date_iso:str)->str`、`finra_asof(month:str)->str`(回 ISO8601+08:00 字串)

- [ ] **Step 1: 寫失敗測試**

```python
# tests/test_fetchers.py
import unittest
from marketlib.fetchers import twse_asof, us_price_asof, finra_asof
from datetime import datetime
from zoneinfo import ZoneInfo

def _dt(s): return datetime.fromisoformat(s)

class TestAsof(unittest.TestCase):
    def test_twse_price_before_1830(self):
        self.assertLess(_dt(twse_asof("2026-06-26", "price")),
                        _dt("2026-06-26T18:30:00+08:00"))

    def test_us_price_next_morning_taipei(self):
        # 美股 2026-06-25 收盤 -> Taipei 2026-06-26 凌晨,故 > 2026-06-25 18:30
        self.assertGreater(_dt(us_price_asof("2026-06-25")),
                           _dt("2026-06-25T18:30:00+08:00"))

    def test_finra_lag_approx(self):
        # 2026-05 月資料 -> 約 2026-06-20 後才可用
        self.assertGreaterEqual(_dt(finra_asof("2026-05")),
                                _dt("2026-06-20T00:00:00+08:00"))

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `python3 -m unittest -v tests.test_fetchers`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 建立 `marketlib/fetchers.py`**

把 `daily_market_report.py` 內所有抓取/解析函式與相關常數**原樣移入**(見上方 Produces 清單)。頂部 import:
```python
from __future__ import annotations
import csv, html, json, re, shutil, subprocess, time, urllib.parse, urllib.request
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Any, Optional, Union
from marketlib.core import MarketBar, clean_num
TAIPEI_TZ = ZoneInfo("Asia/Taipei")
```
於檔末新增 asof helper:
```python
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
```

- [ ] **Step 4: 改 `daily_market_report.py` import**(刪原定義,改 `from marketlib.fetchers import ...`)

- [ ] **Step 5: 跑測試 + 冒煙**

Run:
```bash
python3 -m unittest -v tests.test_fetchers
python3 -c "import daily_market_report"
```
Expected: PASS;import 無錯

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "refactor: extract fetchers into marketlib.fetchers; add asof helpers"
```

---

## Task 6: `backfill.py` — 多年歷史回填(含 asof、續傳)

**Files:**
- Create: `backfill.py`
- Test: `tests/test_backfill.py`

**Interfaces:**
- Consumes: `marketlib.fetchers.*`、`marketlib.store.HistoryStore`、asof helpers
- Produces: `backfill_tw(store, fetchers_ns, months:list[date])`、`backfill_us(store, start, end)`、`backfill_finra(store)`、`main()`(CLI:`--db --start --end --sources --sleep`)。每筆寫入都填 `*_asof_taipei`、`available_at_taipei`、`missing_fields`;以 `store.is_ingested/mark_ingested` 續傳。

- [ ] **Step 1: 寫失敗測試(用假 fetcher,不打網路)**

```python
# tests/test_backfill.py
import os, tempfile, unittest
from marketlib.store import HistoryStore
import backfill

class FakeFetchers:
    def fetch_twse_market_month(self, month_start):
        from marketlib.core import MarketBar
        return [MarketBar(date="2026-06-26", close=100.0, turnover=5e11, volume=1e9)]
    def fetch_twse_institutional(self, date_iso):
        return {"rows": {"合計": {"net_ntd": 1.2e9}}}
    def fetch_twse_margin(self, date_iso):
        return {"margin_financing": {"today_balance_ntd": 2e11, "balance_change_ntd": 3e9}}
    def fetch_twse_summary(self, date_iso):
        return {"breadth": {"上漲": {"stocks": 600}, "下跌": {"stocks": 300}}}

class TestBackfill(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = HistoryStore(os.path.join(self.tmp, "h.db"))
        self.store.init_schema()

    def test_tw_backfill_populates_asof(self):
        from datetime import date
        backfill.backfill_tw(self.store, FakeFetchers(), [date(2026, 6, 1)], sleep=0)
        rows = self.store.get_tw_rows()
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertIsNotNone(r["price_asof_taipei"])
        self.assertIsNotNone(r["inst_asof_taipei"])
        self.assertIsNotNone(r["available_at_taipei"])
        self.assertAlmostEqual(r["inst_total_net_ntd"], 1.2e9)

    def test_resume_skips_done(self):
        from datetime import date
        backfill.backfill_tw(self.store, FakeFetchers(), [date(2026, 6, 1)], sleep=0)
        # 第二次:已 ingested 的月份不再覆蓋(用 is_ingested 判斷)
        self.assertTrue(self.store.is_ingested("tw_month", "2026-06"))

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `python3 -m unittest -v tests.test_backfill`
Expected: FAIL（`ModuleNotFoundError: backfill`）

- [ ] **Step 3: 實作 `backfill.py`**

```python
#!/usr/bin/env python3
"""Offline multi-year history backfill into SQLite (point-in-time aware)."""
from __future__ import annotations
import argparse, time
from datetime import date, datetime, timedelta
from marketlib import fetchers as F
from marketlib.store import HistoryStore

def _max_asof(*vals):
    present = [v for v in vals if v]
    return max(present) if present else None

def backfill_tw(store, ns, months, sleep: float = 0.5) -> None:
    for month_start in months:
        key = month_start.strftime("%Y-%m")
        bars = ns.fetch_twse_market_month(month_start)
        for bar in bars:
            d = bar.date
            inst = ns.fetch_twse_institutional(d) or {}
            margin = ns.fetch_twse_margin(d) or {}
            summary = ns.fetch_twse_summary(d) or {}
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
                time.sleep(sleep)
        store.mark_ingested("tw_month", key)

def backfill_us(store, start: date, end: date, sleep: float = 0.5) -> None:
    spy = {b.date: b for b in F.fetch_nasdaq_history("SPY", "etf", start, end)}
    qqq = {b.date: b for b in F.fetch_nasdaq_history("QQQ", "etf", start, end)}
    iwm = {b.date: b for b in F.fetch_nasdaq_history("IWM", "etf", start, end)}
    comp = {b.date: b for b in F.fetch_nasdaq_history("COMP", "index", start, end)}
    vix = {b.date: b for b in F.fetch_vix_history(start, end)}
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
            "available_at_taipei": F.us_price_asof(d),
        })

def backfill_finra(store) -> None:
    data = F.fetch_finra_margin_statistics()
    if not data:
        return
    for row in data.get("rows", []):
        # row["month"] 形如 "Jun-25";轉成 "YYYY-MM"
        try:
            dt = datetime.strptime(row["month"], "%b-%y")
        except ValueError:
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
    p.add_argument("--sleep", type=float, default=0.5)
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
        backfill_tw(store, F, months, sleep=a.sleep)
    if "us" in sources:
        backfill_us(store, start, end, sleep=a.sleep)
    if "finra" in sources:
        backfill_finra(store)
    store.close()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

> 註:`backfill_tw` 第二參數 `ns` 是「fetcher namespace」,正式跑時傳 `marketlib.fetchers` 模組(`F`),測試時傳 `FakeFetchers()`,介面相同。

- [ ] **Step 4: 跑測試確認通過**

Run: `python3 -m unittest -v tests.test_backfill`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: add offline history backfill with point-in-time asof + resume"
```

---

## Task 7: `features.py`(A)— decision_time 與**欄位群 gating**(Guardrail #1)

**Files:**
- Create: `marketlib/features.py`
- Test: `tests/test_features.py`(本任務先加 gating 測試)

**Interfaces:**
- Produces:
  - `decision_time_for(date_iso:str)->datetime`(該日 18:30 Taipei)
  - `is_available(asof_iso:Optional[str], decision_time:datetime)->bool`(`asof` 為 None 視為不可用;`asof<=decision_time` 為可用)
  - `usable_value(row:dict, value_col:str, asof_col:str, decision_time)->Optional[float]`(欄位群 gating:該群 asof 不可用則回 None,**不影響其他欄位群**)

- [ ] **Step 1: 寫失敗測試(欄位群 gating)**

```python
# tests/test_features.py
import unittest
from datetime import datetime
from zoneinfo import ZoneInfo
from marketlib.features import decision_time_for, is_available, usable_value

TZ = ZoneInfo("Asia/Taipei")

class TestGating(unittest.TestCase):
    def test_decision_time_is_1830(self):
        dt = decision_time_for("2026-06-26")
        self.assertEqual((dt.hour, dt.minute), (18, 30))

    def test_is_available(self):
        dt = decision_time_for("2026-06-26")
        self.assertTrue(is_available("2026-06-26T16:00:00+08:00", dt))
        self.assertFalse(is_available("2026-06-26T19:00:00+08:00", dt))  # 晚於 cron
        self.assertFalse(is_available(None, dt))

    def test_late_inst_does_not_drop_price(self):
        # Guardrail #1:法人晚發布(19:00),但價量(14:30)仍應可用
        dt = decision_time_for("2026-06-26")
        row = {
            "close": 100.0, "price_asof_taipei": "2026-06-26T14:30:00+08:00",
            "inst_total_net_ntd": 9e8, "inst_asof_taipei": "2026-06-26T19:00:00+08:00",
        }
        self.assertEqual(usable_value(row, "close", "price_asof_taipei", dt), 100.0)
        self.assertIsNone(usable_value(row, "inst_total_net_ntd", "inst_asof_taipei", dt))

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `python3 -m unittest -v tests.test_features`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 實作 `marketlib/features.py`(A 部分)**

```python
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
```

- [ ] **Step 4: 跑測試確認通過**

Run: `python3 -m unittest -v tests.test_features`
Expected: PASS（含 `test_late_inst_does_not_drop_price`）

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(features): decision_time and per-field-group asof gating"
```

---

## Task 8: `features.py`(B)— 視窗政策(z-score 不含 T、regime 含 T)

**Files:**
- Modify: `marketlib/features.py`
- Test: `tests/test_features.py`(加視窗政策測試)

**Interfaces:**
- Produces:
  - `zscore_excluding_t(series:list[float], n:int)->Optional[float]`(用 `series[-n-1:-1]` 當基準,比 `series[-1]`)
  - `realized_vol(returns:list[float], n:int)->Optional[float]`(含 T,`returns[-n:]` 標準差)
  - `volume_ratio_excluding_t(volumes:list[float], n:int=20)->Optional[float]`

- [ ] **Step 1: 寫失敗測試**

```python
# 追加到 tests/test_features.py
from marketlib.features import zscore_excluding_t, realized_vol, volume_ratio_excluding_t

class TestWindowPolicy(unittest.TestCase):
    def test_zscore_excludes_t(self):
        # 基準前 4 個都=10(std=0 -> 回 None 以免除以零);改成有變異
        series = [10, 12, 8, 10, 20]  # 基準=[10,12,8,10] mean=10 std≈1.414
        z = zscore_excluding_t(series, 4)
        self.assertAlmostEqual(z, (20 - 10) / pstdev_helper([10, 12, 8, 10]), places=4)

    def test_realized_vol_includes_t(self):
        rets = [0.01, -0.01, 0.02, -0.02]
        self.assertAlmostEqual(realized_vol(rets, 4), pstdev_helper(rets), places=6)

    def test_volume_ratio_excludes_t(self):
        vols = [100, 100, 100, 200]  # 前 3 均=100,今日=200
        self.assertAlmostEqual(volume_ratio_excluding_t(vols, 3), 2.0)

def pstdev_helper(xs):
    from statistics import pstdev
    return pstdev(xs)
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `python3 -m unittest -v tests.test_features`
Expected: FAIL（`ImportError: cannot import name 'zscore_excluding_t'`）

- [ ] **Step 3: 在 `features.py` 追加實作**

```python
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
```

- [ ] **Step 4: 跑測試確認通過**

Run: `python3 -m unittest -v tests.test_features`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(features): window policy (z-score excludes T, regime includes T)"
```

---

## Task 9: `features.py`(C)— `build_dataset`(整合特徵 + 標籤 + 跨市場 US-lag)

**Files:**
- Modify: `marketlib/features.py`
- Test: `tests/test_features.py`(加 dataset / causality / US-lag 測試)

**Interfaces:**
- Produces:
  - `FEATURE_NAMES_TW:list[str]`、`FEATURE_NAMES_US:list[str]`(固定順序)
  - `latest_usable_us(us_rows:list[dict], decision_time)->Optional[dict]`(回 `price_asof <= decision_time` 的最後一列 → 自動實現「US 落後台股一場」)
  - `build_feature_row(market:str, rows:list[dict], idx:int, us_rows:list[dict], decision_time:datetime)->dict[str,Optional[float]]`
  - `build_dataset(market:str, rows:list[dict], us_rows:list[dict])->list[dict]`(每筆:`{"date","features":dict,"label":int|None,"next_return":float|None,"actual_direction":str|None,"neutral_threshold_pct":float}`;最後一列 label=None)

- [ ] **Step 1: 寫失敗測試**

```python
# 追加到 tests/test_features.py
from marketlib.features import build_dataset, latest_usable_us, build_feature_row

def _tw_row(d, close, vol=1e9, turnover=5e11):
    return {"date": d, "close": close, "volume": vol, "turnover": turnover,
            "price_asof_taipei": f"{d}T14:30:00+08:00",
            "available_at_taipei": f"{d}T14:30:00+08:00"}

def _us_row(d, spy):
    return {"date": d, "spy_close": spy, "qqq_close": spy, "spy_volume": 1e8,
            "vix_close": 18.0,
            "price_asof_taipei": __import__("marketlib.fetchers", fromlist=["us_price_asof"]).us_price_asof(d)}

class TestDataset(unittest.TestCase):
    def test_us_lag_one_session(self):
        # 台股 2026-06-26 預測,US 2026-06-26 收盤次日凌晨才可用 -> 只能用 06-25
        dt = decision_time_for("2026-06-26")
        us = [_us_row("2026-06-25", 100.0), _us_row("2026-06-26", 200.0)]
        latest = latest_usable_us(us, dt)
        self.assertEqual(latest["date"], "2026-06-25")

    def test_label_is_next_day_up(self):
        rows = [_tw_row(f"2026-06-{d:02d}", close) for d, close in
                [(1, 100), (2, 101), (3, 100)]]
        ds = build_dataset("tw", rows, [])
        # 第 0 筆:隔日 101>100 -> label 1;最後一筆 label None
        self.assertEqual(ds[0]["label"], 1)
        self.assertIsNone(ds[-1]["label"])

    def test_feature_row_has_named_keys(self):
        rows = [_tw_row(f"2026-06-{d:02d}", 100 + d) for d in range(1, 30)]
        dt = decision_time_for(rows[-1]["date"])
        feats = build_feature_row("tw", rows, len(rows) - 1, [], dt)
        self.assertIn("ret_1d", feats)
        self.assertIn("close_vs_ma20", feats)
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `python3 -m unittest -v tests.test_features`
Expected: FAIL（`ImportError: build_dataset`）

- [ ] **Step 3: 在 `features.py` 追加實作**

```python
FEATURE_NAMES_COMMON = [
    "ret_1d", "ret_5d", "ret_20d", "ret_60d",
    "close_vs_ma5", "close_vs_ma20", "close_vs_ma60", "ma20_vs_ma60",
    "vol_ratio_20d", "vol_ratio_5d", "drawdown_20d", "drawdown_60d",
    "realized_vol_20d", "realized_vol_60d",
]
FEATURE_NAMES_TW = FEATURE_NAMES_COMMON + [
    "margin_change_pct", "inst_net_z60", "breadth_ratio_z60",
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
```

- [ ] **Step 4: 跑測試確認通過**

Run: `python3 -m unittest -v tests.test_features`
Expected: PASS（含 `test_us_lag_one_session`）

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(features): build_dataset with labels and point-in-time cross-market US lag"
```

---

## Task 10: `marketlib/model.py` — 純標準庫 LR 推論

**Files:**
- Create: `marketlib/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: 模型 JSON(§8.7 schema)
- Produces: `load_model(path:str)->dict`、`predict_proba(model:dict, features:dict)->float`、`proba_to_direction(model:dict, p:float)->str`、`proba_to_confidence(model:dict, p:float)->str`

- [ ] **Step 1: 寫失敗測試**

```python
# tests/test_model.py
import json, os, tempfile, unittest
from marketlib.model import load_model, predict_proba, proba_to_direction, proba_to_confidence

MODEL = {
    "market": "tw", "model_type": "logistic_regression",
    "feature_names": ["a", "b"],
    "standardization": {"mean": {"a": 0.0, "b": 0.0}, "std": {"a": 1.0, "b": 1.0}},
    "imputation_median": {"a": 0.0, "b": 0.0},
    "coefficients": {"a": 2.0, "b": 0.0}, "intercept": 0.0,
    "thresholds": {"bull": 0.6, "bear": 0.4},
    "confidence_bands": {"mid": 0.05, "high": 0.2},
}

class TestModel(unittest.TestCase):
    def test_predict_proba_logistic(self):
        p = predict_proba(MODEL, {"a": 1.0, "b": 0.0})  # z = 2 -> sigmoid(2)
        self.assertAlmostEqual(p, 1 / (1 + pow(2.718281828, -2.0)), places=4)

    def test_missing_feature_uses_median(self):
        p = predict_proba(MODEL, {"b": 0.0})  # a 缺 -> median 0 -> z=0 -> 0.5
        self.assertAlmostEqual(p, 0.5, places=6)

    def test_direction_and_confidence(self):
        self.assertEqual(proba_to_direction(MODEL, 0.7), "偏多")
        self.assertEqual(proba_to_direction(MODEL, 0.3), "偏空")
        self.assertEqual(proba_to_direction(MODEL, 0.5), "震盪")
        self.assertEqual(proba_to_confidence(MODEL, 0.95), "高")
        self.assertEqual(proba_to_confidence(MODEL, 0.5), "低")

    def test_load_model_roundtrip(self):
        d = tempfile.mkdtemp(); path = os.path.join(d, "m.json")
        json.dump(MODEL, open(path, "w"))
        self.assertEqual(load_model(path)["market"], "tw")

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `python3 -m unittest -v tests.test_model`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 實作 `marketlib/model.py`**

```python
from __future__ import annotations
import json, math
from typing import Any, Optional

def load_model(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)

def predict_proba(model: dict, features: dict) -> float:
    mean = model["standardization"]["mean"]
    std = model["standardization"]["std"]
    median = model.get("imputation_median", {})
    coef = model["coefficients"]
    z = float(model.get("intercept", 0.0))
    for name in model["feature_names"]:
        x = features.get(name)
        if x is None:
            x = median.get(name, 0.0)
        s = std.get(name, 1.0) or 1.0
        x_std = (x - mean.get(name, 0.0)) / s
        z += coef.get(name, 0.0) * x_std
    return 1.0 / (1.0 + math.exp(-z))

def proba_to_direction(model: dict, p: float) -> str:
    t = model["thresholds"]
    if p >= t["bull"]:
        return "偏多"
    if p <= t["bear"]:
        return "偏空"
    return "震盪"

def proba_to_confidence(model: dict, p: float) -> str:
    bands = model.get("confidence_bands", {"mid": 0.05, "high": 0.15})
    dist = abs(p - 0.5)
    if dist >= bands["high"]:
        return "高"
    if dist >= bands["mid"]:
        return "中"
    return "低"
```

- [ ] **Step 4: 跑測試確認通過**

Run: `python3 -m unittest -v tests.test_model`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: stdlib logistic-regression inference (marketlib.model)"
```

---

## Task 11: `train.py` — walk-forward 訓練 + 雙 artifact(Guardrail #2)

**Files:**
- Create: `train.py`
- Test: `tests/test_train.py`(用 `.venv` 跑;小型合成資料)

**Interfaces:**
- Consumes: `marketlib.store.HistoryStore`、`marketlib.features.build_dataset`、`marketlib.evaluation.*`、`marketlib.stats.*`、`numpy`、`sklearn`
- Produces:
  - `walk_forward_select(samples, feature_names, seed=0)->dict`(回最佳超參數 + 跨 fold OOS 指標 + 門檻)
  - `fit_logistic(X, y, params)->dict`(回 coefficients/intercept/standardization/median)
  - `evaluate_on_test(model, test_samples, feature_names)->dict`(含 baseline、McNemar、bootstrap)
  - `train_market(db, market, out_dir, test_months=18, seed=0)->tuple[str,str]`
  - **產出兩個檔,metadata 明確區隔**:
    - `models/metrics_{market}.json` — `{"artifact":"final_test_evaluation","trained_on":["start","val_end"],"tested_on":["test_start","test_end"], "metrics":..., "baselines":..., "significance":...}`
    - `models/model_{market}.json` — `{"artifact":"production_refit","train_period":["start","all_end"],"note":"refit on ALL data incl. test window; final-test metrics in metrics_*.json were from a model NOT trained on the test window", ...weights...}`

- [ ] **Step 1: 寫失敗測試(合成資料、驗證雙 artifact 分離)**

```python
# tests/test_train.py  (run with: .venv/bin/python -m unittest -v tests.test_train)
import os, json, tempfile, unittest
import train

def _synth_samples(n=400):
    # 造一個「ret_1d 正相關於隔日上漲」的可學訊號
    import random
    rng = random.Random(0)
    samples = []
    for i in range(n):
        sig = rng.gauss(0, 1)
        feats = {name: rng.gauss(0, 1) for name in train.MINIMAL_FEATURES}
        feats["ret_1d"] = sig
        label = 1 if (sig + rng.gauss(0, 0.5)) > 0 else 0
        samples.append({"date": f"2020-{(i%12)+1:02d}-{(i%28)+1:02d}",
                        "features": feats, "label": label,
                        "next_return": 0.01 if label else -0.01,
                        "actual_direction": "偏多" if label else "偏空",
                        "neutral_threshold_pct": 0.003})
    return samples

class TestTrain(unittest.TestCase):
    def test_artifacts_have_distinct_periods_and_metadata(self):
        samples = _synth_samples()
        out = tempfile.mkdtemp()
        model_path, metrics_path = train.train_from_samples(
            samples, train.MINIMAL_FEATURES, "tw", out, test_frac=0.25, seed=0)
        model = json.load(open(model_path)); metrics = json.load(open(metrics_path))
        self.assertEqual(model["artifact"], "production_refit")
        self.assertEqual(metrics["artifact"], "final_test_evaluation")
        # production 訓練期間涵蓋到最後;final-test 訓練期間不含 test 窗
        self.assertNotEqual(model["train_period"], metrics["trained_on"])
        self.assertIn("baselines", metrics)
        self.assertIn("significance", metrics)

    def test_learns_signal_beats_random(self):
        samples = _synth_samples()
        out = tempfile.mkdtemp()
        _, metrics_path = train.train_from_samples(
            samples, train.MINIMAL_FEATURES, "tw", out, test_frac=0.25, seed=0)
        metrics = json.load(open(metrics_path))
        self.assertGreater(metrics["metrics"]["hit_rate"], 0.55)

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `.venv/bin/python -m unittest -v tests.test_train`
Expected: FAIL（`ModuleNotFoundError: train`）

- [ ] **Step 3: 實作 `train.py`**

```python
#!/usr/bin/env python3
"""Offline walk-forward training. Produces final-test metrics + production refit model."""
from __future__ import annotations
import argparse, json, os
from datetime import datetime
import numpy as np
from sklearn.linear_model import LogisticRegression
from marketlib.evaluation import summarize_classification, summarize_baselines
from marketlib.stats import mcnemar_test, block_bootstrap_diff

MINIMAL_FEATURES = [
    "ret_1d", "ret_5d", "ret_20d", "close_vs_ma20", "ma20_vs_ma60",
    "vol_ratio_20d", "realized_vol_20d",
]
PARAM_GRID = [
    {"C": c, "class_weight": cw}
    for c in (0.1, 0.3, 1.0, 3.0) for cw in (None, "balanced")
]

def _matrix(samples, feature_names):
    med = {n: float(np.nanmedian([_get(s, n) for s in samples])) for n in feature_names}
    X, y = [], []
    for s in samples:
        X.append([_imp(_get(s, n), med[n]) for n in feature_names])
        y.append(s["label"])
    return np.array(X, dtype=float), np.array(y, dtype=int), med

def _get(s, n):
    v = s["features"].get(n)
    return float("nan") if v is None else float(v)

def _imp(v, m):
    return m if (v != v) else v  # NaN -> median

def _standardize(X):
    mean = X.mean(axis=0); std = X.std(axis=0); std[std == 0] = 1.0
    return (X - mean) / std, mean, std

def _fit(X, y, params):
    Xs, mean, std = _standardize(X)
    clf = LogisticRegression(C=params["C"], class_weight=params["class_weight"],
                             max_iter=1000)
    clf.fit(Xs, y)
    return clf, mean, std

def _proba(clf, mean, std, X):
    Xs = (X - mean) / np.where(std == 0, 1.0, std)
    return clf.predict_proba(Xs)[:, 1]

def _select_thresholds(p, y):
    best, best_f1 = (0.55, 0.45), -1.0
    for bull in (0.52, 0.55, 0.58, 0.6):
        for bear in (0.4, 0.42, 0.45, 0.48):
            recs = [{"prediction": "偏多" if pi >= bull else "偏空" if pi <= bear else "震盪",
                     "actual_direction": "偏多" if yi else "偏空"} for pi, yi in zip(p, y)]
            f1 = summarize_classification(recs)["macro_f1"] or 0.0
            if f1 > best_f1:
                best_f1, best = f1, (bull, bear)
    return best

def walk_forward_select(samples, feature_names, n_folds=4, seed=0):
    n = len(samples); fold = n // (n_folds + 1)
    scored = []
    for params in PARAM_GRID:
        f1s, briers = [], []
        for k in range(1, n_folds + 1):
            tr = samples[: fold * k]; va = samples[fold * k: fold * (k + 1)]
            if len(va) < 10 or len(set(s["label"] for s in tr)) < 2:
                continue
            Xtr, ytr, _ = _matrix(tr, feature_names)
            Xva, yva, _ = _matrix(va, feature_names)
            clf, mean, std = _fit(Xtr, ytr, params)
            p = _proba(clf, mean, std, Xva)
            recs = [{"prediction": "偏多" if pi >= 0.5 else "偏空",
                     "actual_direction": "偏多" if yi else "偏空"} for pi, yi in zip(p, yva)]
            f1s.append(summarize_classification(recs)["macro_f1"] or 0.0)
            briers.append(float(np.mean((p - yva) ** 2)))
        if f1s:
            scored.append({"params": params, "macro_f1": float(np.mean(f1s)),
                           "brier": float(np.mean(briers))})
    # 主指標 macro_f1;Brier 為硬條件:剔除 brier > 最小 brier + 0.02 的設定
    if not scored:
        return {"params": PARAM_GRID[0], "macro_f1": None, "brier": None}
    min_brier = min(s["brier"] for s in scored)
    feasible = [s for s in scored if s["brier"] <= min_brier + 0.02]
    return max(feasible, key=lambda s: s["macro_f1"])

def evaluate_on_test(clf, mean, std, med, test_samples, feature_names, thresholds):
    X = np.array([[_imp(_get(s, n), med[n]) for n in feature_names] for s in test_samples])
    y = np.array([s["label"] for s in test_samples])
    p = _proba(clf, mean, std, X)
    bull, bear = thresholds
    recs = []
    model_hits, bull_hits, mom_hits = [], [], []
    prev = None
    for i, (pi, s) in enumerate(zip(p, test_samples)):
        pred = "偏多" if pi >= bull else "偏空" if pi <= bear else "震盪"
        actual = s["actual_direction"]
        recs.append({"prediction": pred, "actual_direction": actual,
                     "baselines": {"always_bull": {"label": "永遠偏多", "hit": actual == "偏多"},
                                   "momentum": {"label": "今日方向延續",
                                                "hit": (prev == actual) if prev else False}}})
        model_hits.append(pred == actual)
        bull_hits.append(actual == "偏多")
        prev = actual
    cls = summarize_classification(recs)
    base = summarize_baselines(recs)
    sig = {
        "vs_always_bull_mcnemar": mcnemar_test(model_hits, bull_hits),
        "vs_always_bull_bootstrap": block_bootstrap_diff(model_hits, bull_hits, seed=0),
    }
    hit = sum(1 for r in recs if r["prediction"] == r["actual_direction"]) / len(recs)
    brier = float(np.mean((p - y) ** 2))
    return {"metrics": {"hit_rate": hit, "macro_f1": cls["macro_f1"], "brier": brier,
                        "classification": cls},
            "baselines": base, "significance": sig}

def train_from_samples(samples, feature_names, market, out_dir, test_frac=None,
                       test_months=18, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    labeled = [s for s in samples if s["label"] is not None]
    if test_frac is not None:
        cut = int(len(labeled) * (1 - test_frac))
    else:
        cut = len(labeled) - min(len(labeled) // 4, test_months * 21)
    trainval, test = labeled[:cut], labeled[cut:]
    best = walk_forward_select(trainval, feature_names, seed=seed)
    # final-test 模型:只用 trainval 擬合
    Xtv, ytv, med_tv = _matrix(trainval, feature_names)
    clf, mean, std = _fit(Xtv, ytv, best["params"])
    p_tv = _proba(clf, mean, std, Xtv)
    thresholds = _select_thresholds(p_tv, ytv)
    test_eval = evaluate_on_test(clf, mean, std, med_tv, test, feature_names, thresholds)
    metrics = {"artifact": "final_test_evaluation", "market": market,
               "trained_on": [trainval[0]["date"], trainval[-1]["date"]],
               "tested_on": [test[0]["date"], test[-1]["date"]] if test else None,
               "chosen_params": best["params"], "thresholds": {"bull": thresholds[0], "bear": thresholds[1]},
               **test_eval}
    metrics_path = os.path.join(out_dir, f"metrics_{market}.json")
    json.dump(metrics, open(metrics_path, "w"), ensure_ascii=False, indent=2)
    # production 模型:用「全部」資料重擬(含 test 窗)
    Xall, yall, med_all = _matrix(labeled, feature_names)
    clf2, mean2, std2 = _fit(Xall, yall, best["params"])
    model = {
        "artifact": "production_refit", "market": market,
        "model_type": "logistic_regression",
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "train_period": [labeled[0]["date"], labeled[-1]["date"]],
        "note": ("refit on ALL data incl. test window; the hit_rate/macro_f1 in "
                 "metrics_*.json came from a model NOT trained on the test window and "
                 "are the honest OOS estimate, NOT a direct test of this refit model."),
        "feature_names": feature_names,
        "standardization": {"mean": dict(zip(feature_names, mean2.tolist())),
                            "std": dict(zip(feature_names, std2.tolist()))},
        "imputation_median": med_all,
        "coefficients": dict(zip(feature_names, clf2.coef_[0].tolist())),
        "intercept": float(clf2.intercept_[0]),
        "class_weight": best["params"]["class_weight"],
        "calibration": {"method": "none", "params": {}},
        "thresholds": {"bull": thresholds[0], "bear": thresholds[1]},
        "confidence_bands": {"mid": 0.04, "high": 0.10},
    }
    model_path = os.path.join(out_dir, f"model_{market}.json")
    json.dump(model, open(model_path, "w"), ensure_ascii=False, indent=2)
    return model_path, metrics_path

def train_market(db, market, out_dir, test_months=18, seed=0):
    from marketlib.store import HistoryStore
    from marketlib.features import build_dataset, FEATURE_NAMES_TW, FEATURE_NAMES_US
    store = HistoryStore(db)
    if market == "tw":
        rows, us = store.get_tw_rows(), store.get_us_rows(); names = FEATURE_NAMES_TW
    else:
        rows, us = store.get_us_rows(), []; names = FEATURE_NAMES_US
    samples = build_dataset(market, rows, us)
    store.close()
    return train_from_samples(samples, names, market, out_dir, test_months=test_months, seed=seed)

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="data/history.db")
    p.add_argument("--market", choices=["tw", "us"], required=True)
    p.add_argument("--out", default="models")
    p.add_argument("--test-months", type=int, default=18)
    a = p.parse_args()
    mp, xp = train_market(a.db, a.market, a.out, test_months=a.test_months)
    print(f"Wrote {mp}\nWrote {xp}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 跑測試確認通過**

Run: `.venv/bin/python -m unittest -v tests.test_train`
Expected: PASS（`test_artifacts_have_distinct_periods_and_metadata`、`test_learns_signal_beats_random`)

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: walk-forward training with separated final-test metrics vs production refit artifacts"
```

---

## Task 12: 每日報告整合(模型推論 + 規則式 fallback)

**Files:**
- Modify: `daily_market_report.py`(`build_taiwan_prediction` / `build_us_prediction` / `main` / `write_outputs`)
- Test: `tests/test_daily.py`

**Interfaces:**
- Consumes: `marketlib.model`、`marketlib.features.build_feature_row`、`marketlib.store.HistoryStore`
- Produces: `model_prediction(market, model_path, rows, us_rows)->Optional[dict]`(回 `{"bias","probability","confidence","model_version","neutral_threshold_pct"}`;模型缺檔或特徵大量缺回 None)、`upsert_report_into_store(report, store)`

- [ ] **Step 1: 寫失敗測試**

```python
# tests/test_daily.py
import json, os, tempfile, unittest
from daily_market_report import model_prediction

MODEL = {
    "market": "tw", "feature_names": ["ret_1d"],
    "standardization": {"mean": {"ret_1d": 0.0}, "std": {"ret_1d": 1.0}},
    "imputation_median": {"ret_1d": 0.0}, "coefficients": {"ret_1d": 5.0},
    "intercept": 0.0, "thresholds": {"bull": 0.6, "bear": 0.4},
    "confidence_bands": {"mid": 0.05, "high": 0.2},
}

def _rows():
    return [{"date": f"2026-06-{d:02d}", "close": 100 + d, "volume": 1e9,
             "turnover": 5e11, "price_asof_taipei": f"2026-06-{d:02d}T14:30:00+08:00"}
            for d in range(1, 28)]

class TestDaily(unittest.TestCase):
    def test_model_prediction_returns_proba(self):
        d = tempfile.mkdtemp(); path = os.path.join(d, "model_tw.json")
        json.dump(MODEL, open(path, "w"))
        out = model_prediction("tw", path, _rows(), [])
        self.assertIn(out["bias"], ("偏多", "震盪", "偏空"))
        self.assertTrue(0.0 <= out["probability"] <= 1.0)

    def test_missing_model_returns_none(self):
        self.assertIsNone(model_prediction("tw", "/no/such.json", _rows(), []))

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `python3 -m unittest -v tests.test_daily`
Expected: FAIL（`ImportError: model_prediction`）

- [ ] **Step 3: 在 `daily_market_report.py` 實作 `model_prediction` 並接進預測**

```python
# daily_market_report.py (新增)
from marketlib import model as ml_model
from marketlib.features import build_feature_row, decision_time_for, FEATURE_NAMES_TW, FEATURE_NAMES_US

def model_prediction(market, model_path, rows, us_rows):
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
    present = sum(1 for n in names if feats.get(n) is not None)
    if present < len(names) // 2:   # 特徵大量缺 -> 退回規則式
        return None
    p = ml_model.predict_proba(model, feats)
    return {"bias": ml_model.proba_to_direction(model, p),
            "probability": round(p, 4),
            "confidence": ml_model.proba_to_confidence(model, p),
            "model_version": model.get("trained_at", "unknown"),
            "neutral_threshold_pct": None}
```

在 `build_taiwan_prediction` / `build_us_prediction` 內:先試 `model_prediction(...)`,成功則用模型的 `bias/probability/confidence`、把規則式 `score` 改放進 `secondary_rule_opinion` 欄位並列入報告;失敗(None)則沿用現行規則式並標 `"prediction_engine": "rule_fallback"`。在 `main()` 把 `model_*.json` 路徑由 `--model-dir`(預設 `models/`)帶入。

- [ ] **Step 4: 跑測試 + 冒煙**

Run:
```bash
python3 -m unittest -v tests.test_daily
python3 -c "import daily_market_report"
```
Expected: PASS;import 無錯

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: integrate model inference into daily report with rule-based fallback"
```

---

## Task 13: Train/serve parity 測試(防 skew)

**Files:**
- Test: `tests/test_parity.py`

**Interfaces:**
- Consumes: `marketlib.features.build_feature_row`

- [ ] **Step 1: 寫測試(同一日,離線 dataset 路徑 vs 線上單日路徑,特徵必須完全一致)**

```python
# tests/test_parity.py
import unittest
from marketlib.features import build_dataset, build_feature_row, decision_time_for

def _rows():
    return [{"date": f"2026-06-{d:02d}", "close": 100 + (d % 7), "volume": 1e9 + d,
             "turnover": 5e11, "price_asof_taipei": f"2026-06-{d:02d}T14:30:00+08:00",
             "available_at_taipei": f"2026-06-{d:02d}T14:30:00+08:00"}
            for d in range(1, 28)]

class TestParity(unittest.TestCase):
    def test_offline_equals_online_for_last_day(self):
        rows = _rows()
        offline = build_dataset("tw", rows, [])[-1]["features"]
        dt = decision_time_for(rows[-1]["date"])
        online = build_feature_row("tw", rows, len(rows) - 1, [], dt)
        self.assertEqual(offline, online)  # 同一份程式 -> 完全一致

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 跑測試確認通過**(同一函式來源,應直接 PASS)

Run: `python3 -m unittest -v tests.test_parity`
Expected: PASS

- [ ] **Step 3: 跑全部測試**

Run: `python3 -m unittest -v 2>&1 | tail -5 && .venv/bin/python -m unittest -v tests.test_train`
Expected: 全 PASS

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "test: train/serve feature parity guard"
```

---

## End-to-end 驗收(人工跑一次)

```bash
# 1) 回填歷史(會花時間;可分批 / 中斷續傳)
python3 backfill.py --db data/history.db --start 2016-01-01 --sleep 0.4
# 2) 訓練(離線 venv)
.venv/bin/python train.py --market tw --db data/history.db --out models
.venv/bin/python train.py --market us --db data/history.db --out models
# 3) 看 OOS 成績是否贏 baseline(誠實門檻在 spec §2)
python3 -c "import json;m=json.load(open('models/metrics_tw.json'));print('hit',m['metrics']['hit_rate']);print('baseline',m['baselines']);print('sig',m['significance'])"
# 4) 跑每日報告(會自動載 models/ 下模型,失敗則 rule fallback)
python3 daily_market_report.py
```

**判讀**:若 `metrics_*.json` 的 `hit_rate` / `macro_f1` 沒有在 McNemar + bootstrap 下穩定贏過 `always_bull` 與 `random_prior`,依 spec §2 視為「尚未找到 edge」——**不可為了好看回頭調 final test**,而是回到特徵 / 資料層改善後重訓。

---

## Phase 2(後續增量,本計畫不展開為任務)

依 spec §10 順序,**一次只加一個特徵**,保留與否只用 train+val walk-forward 的成對顯著性決定,final test 全程不碰、只最後跑一次:

1. 台指期夜盤 → 2. 美股期貨 ES/NQ → 3. 台積電 ADR 折溢價 → 4. USD/TWD → 5. SOX/SMH(共線,最後驗)。

每個領先訊號需要:`fetchers.py` 新增抓取 + asof、`store.py` 新增欄位群(含 `*_asof_taipei`)、`backfill.py` 回填、`features.py` 新增特徵(套 §5.3 gating),再以 `train.py` 比較加入前後。

---

## Self-Review(對照 spec)

- **§2 驗收/顯著性** → Task 3(stats)、Task 11(evaluate_on_test 帶 McNemar+bootstrap+baseline)、End-to-end 判讀。✓
- **§4 模組抽取** → Task 1/2/5(core/evaluation/fetchers),Task 3/4/7-9/10(新模組)。✓
- **§5.2 schema 中繼欄位** → Task 4。✓
- **§5.3 available_at<=decision_time(含 US-lag、FINRA、欄位群)** → Task 7(gating)、Task 9(latest_usable_us)、Task 5(asof helpers)。✓
- **§5.4 回填填 asof** → Task 6。✓
- **§6 視窗政策(含/不含 T)** → Task 8。✓
- **§7 標籤** → Task 9(build_dataset label/next_return/actual_direction)。✓
- **§8 walk-forward / class_weight 超參 / Brier 硬條件 / 門檻 / 雙 artifact** → Task 11。✓
- **§8.7 model JSON(含 class_weight/calibration/confidence_bands)** → Task 10/11。✓
- **§9 每日整合 + fallback** → Task 12。✓
- **Guardrail #1 欄位群 gating(不整列丟)** → Task 7 `test_late_inst_does_not_drop_price`。✓
- **Guardrail #2 artifact 分離 metadata** → Task 11 `test_artifacts_have_distinct_periods_and_metadata`。✓
- **§13 測試(gating / 視窗 / parity / 顯著性)** → Task 7/8/13/3。✓
- **train/serve skew** → Task 13 parity。✓
