# 台美股「下一交易日方向」預測訓練系統 — 設計文件

- 日期:2026-06-29
- 狀態:設計待使用者最終確認
- 相關程式:`daily_market_report.py`(現行每日報告 / 規則式預測)

---

## 1. 背景與目標

現行 `daily_market_report.py` 每日抓台/美股大盤資料,用**手填權重的規則式分數**產生「下一交易日方向」預測,並有 rolling 價量診斷與 baseline 對照。

加入 baseline 後已證實:**目前規則模型沒有 edge**——台股命中率(31%)輸給「永遠偏多」(44%)與隨機(36%);美股僅約與隨機持平。根因是 score 被永遠開著的趨勢項主導,結構上幾乎喊不出「偏空」(32 天 0 次)。

**核心目標:把「下一交易日方向」的預測準確率,做到在從未參與調參的測試集上、跨多個市場 regime,穩定且顯著贏過 baseline。** 方法是把手填規則升級為「用多年歷史資料訓練、輸出上漲機率」的模型,並建立誠實的 walk-forward 回測。

### 設計決策(已與使用者確認)

1. **混合引擎**:離線訓練可用 scikit-learn;訓練後匯出**權重 JSON**;每日 cron **維持純標準庫**,只讀 JSON 做推論。
2. **預測目標 = 上漲機率**:模型輸出 `P(next_return > 0)`,再用驗證集調出的機率門檻轉成 偏多 / 震盪 / 偏空 顯示。
3. **兩階段資料**:Phase 1 先用現有免費資料做完整歷史回填與訓練、建立可信基準;Phase 2 再逐一加入隔日領先訊號,每個都要通過 OOS 驗收才保留。

---

## 2. 成功標準 / 驗收門檻

誠實前提:隔日大盤方向接近銅板。做對的模型,二元方向命中率現實落在約 **52~56%**;宣稱 65%+ 幾乎都是洩漏或過擬合。

**驗收門檻(在「從不碰的最終測試集」上,且力求跨多個 regime 穩定):**

- 二元方向命中率(`p>0.5` vs `return>0`)> always-bull 與 random baseline。**顯著性用成對(paired)檢定,不可只用獨立樣本 SE**:模型與 baseline 預測同一批日子,屬成對比較,故用 **McNemar test(對 baseline 的逐日命中/失誤配對)** 搭配 **block bootstrap(區塊長度需涵蓋日報酬序列相關)** 估計差異信賴區間;差異需在 block bootstrap 下穩定為正。
- 三類 macro-F1 > 規則式模型、> random_prior。
- 機率校準合理(Brier score 不差於常數機率基準;校準曲線無嚴重偏離)。
- 模型**會發出三類**(偏空預測數量不可長期為 0)。

**「找不到 edge」是合法且必須誠實回報的結果**——不可為了好看而在測試集上反覆調參。

---

## 3. 非目標(YAGNI)

- 不做盤中 / 多日 / 報酬大小回歸,只做「下一交易日方向」。
- 不做個股,只做大盤(台股加權、美股 SPY/QQQ/IWM/COMP)。
- Phase 1 不引入新聞 / 財報 / 總經估值(PE、Fed 等;對隔日邊際貢獻低)。
- 不在每日 cron 內訓練模型;不在每日 runtime 引入 numpy/sklearn。
- 不做完整框架化重寫;只做為達成目標所需的最小共用程式抽取。

---

## 4. 架構總覽

離線(偶爾手動 / 每月)與線上(每日 cron)分離:

```
[offline] backfill.py            抓多年歷史 → 寫入 SQLite 特徵庫
[shared]  marketlib/store.py     SQLite 讀寫、ingest 續傳游標
[shared]  marketlib/features.py  ★單一來源★ 由每日資料列算出特徵向量(causal)
[offline] train.py (sklearn)     walk-forward 訓練/選模/調門檻 → 匯出
            └─> models/model_tw.json, models/model_us.json   (係數/標準化/門檻/後設)
            └─> models/metrics_tw.json, models/metrics_us.json (OOS 報告)
[shared]  marketlib/model.py     純標準庫:載入 JSON → predict_proba → 方向
[online]  daily_market_report.py 每日:append 今日資料 → 算特徵 → 載模型 → 輸出機率/三類
```

### 模組(最小抽取,非重寫)

把現有檔內、離線與線上都需要的共用基元抽到 `marketlib/`,新功能建在其上;`daily_market_report.py` 改成 import 這些模組,其餘維持原樣。

- `marketlib/core.py` — `MarketBar`、`clean_num`、`pct_change`、`simple_moving_average`、`compute_indicators` 等(自現檔抽出)。
- `marketlib/evaluation.py` — `summarize_classification`、`summarize_baselines`、`direction_from_return`、`neutral_threshold_from_bars`(自現檔抽出,訓練與每日共用)。
- `marketlib/store.py` —(新)SQLite schema、upsert、查詢、ingest 游標。
- `marketlib/features.py` —(新)**單一來源特徵工程**(見 §6)。
- `marketlib/model.py` —(新)純標準庫 LR 推論:載入權重 JSON、標準化、logistic、套門檻。
- `backfill.py` —(新,離線)多來源歷史回填,寫入 SQLite。
- `train.py` —(新,離線,sklearn)walk-forward 訓練、選模、調門檻、匯出 JSON 與 metrics。
- `daily_market_report.py` —(改)用 `marketlib` + 載入模型輸出機率;規則式分數保留為次要意見與 baseline。

> **train/serve skew 防線(最重要):** 特徵工程只有 `features.py` 一份,離線訓練與每日推論都呼叫它;嚴禁在兩邊各寫一份。

---

## 5. 資料層:歷史回填 + SQLite + point-in-time

### 5.1 來源與深度

| 來源 | 內容 | 回填方式 | 風險 |
|---|---|---|---|
| TWSE FMTQIK | 加權指數價量 | 逐月 | 順 |
| TWSE MI_MARGN | 融資/融券 | 逐日 | 慢、需限速/續傳 |
| TWSE BFI82U | 三大法人 | 逐日 | 慢、需限速/續傳 |
| TWSE MI_INDEX | 漲跌家數/族群 | 逐日 | 慢、需限速/續傳 |
| Nasdaq API | SPY/QQQ/IWM/COMP 日線 | 區間 | 偶被擋,需重試 |
| Cboe VIX | VIX 日線 | 單檔 CSV | 順 |
| FINRA | margin debt 月資料 | 歷史檔 | 僅近月於網頁;歷史需另尋官方檔;**有發布延遲** |

深度目標:**抓到來源能給的最早,至少 5 年、力求 10 年**;TWSE 逐日端點實際可回填深度以實測為準,並在 metrics 報告誠實標註各特徵的有效起始日。

### 5.2 SQLite schema(`data/history.db`)

每一列除了資料欄位,都要帶 **point-in-time 與品質中繼欄位**(不能只有 `date`):

- `tw_daily(date PK, close, change_points, turnover, volume, trades, margin_balance_ntd, margin_balance_change_ntd, short_lots, foreign_net_ntd, trust_net_ntd, dealer_net_ntd, inst_total_net_ntd, adv_count, dec_count, ...,`
  `price_asof_taipei, margin_asof_taipei, inst_asof_taipei, breadth_asof_taipei,`
  `source_updated_at, available_at_taipei, quality_flags, missing_fields)`
- `us_daily(date PK, spy_close, spy_volume, qqq_close, qqq_volume, iwm_close, iwm_volume, comp_close, vix_close, vix_open, vix_high, vix_low,`
  `price_asof_taipei, vix_asof_taipei, source_updated_at, available_at_taipei, quality_flags, missing_fields)`
- `finra_monthly(month PK, debit_millions, free_credit_cash_millions, free_credit_margin_millions, publish_date, available_at_taipei, source_updated_at, quality_flags)`
- `ingest_log(source, key, status, fetched_at, PRIMARY KEY(source, key))` — 回填續傳用,記錄已成功抓取的日期/月份,可中斷續跑。

中繼欄位語意:
- `*_asof_taipei`:該**欄位群**(價量 / 融資 / 法人 / 廣度 / VIX / FINRA)實際可被取得的 Taipei 時間。同一交易日不同來源發布時間不同,故分群記錄(例:法人、融資常晚於指數價量)。
- `source_updated_at`:來源端標示的最後更新時間(若有)。
- `available_at_taipei`:整列「保守可用時間」= 該列所有非空欄位群 `*_asof_taipei` 的最大值;特徵層做總體 gating 用它,需細到欄位群時再看各自 `*_asof`。
- `quality_flags`:品質旗標(如 `revised` / `partial` / `stale` / `approx_asof`)。
- `missing_fields`:該列缺哪些欄位,供缺值處理與「缺資料源」診斷。

原始每日資料 + 中繼欄位入庫;特徵一律由 `features.py` 從這些資料列即時計算(不另存特徵表),確保單一來源。

### 5.3 Point-in-time 主規則:`available_at <= decision_time`

所有點時間正確性收斂到**單一原則**:每個資料值都帶 `available_at_taipei`(§5.2);特徵層在為某次預測建特徵時,**只能使用 `available_at_taipei <= decision_time` 的值**(細到欄位群時比對各自 `*_asof_taipei`)。

- **decision_time**:模擬每日 cron 執行時刻 =「執行日 D 的 18:30 Taipei」。回測對每個歷史 D 都用同一時刻模擬,各市場各自取「截至該時刻最後一場已收的交易日」當最新樣本;線上 `decision_time` = 當次實際執行時刻。
- 此主規則**自動涵蓋**下列情形,不需各寫特例:
  1. **US → 台股落後一場**:美股 D 日的 Taipei 收盤約 D+1 凌晨,其 `price_asof_taipei > D 18:30`,對台股 D 日預測**自動排除**,只會用到 ≤ D−1 的美股。
  2. **TWSE 法人 / 融資 / 廣度發布延遲**:若某日該欄位群 `*_asof_taipei` 晚於 18:30,當日值**不會被採用**,自動退回前一日——杜絕「cron 當下其實還沒發布卻被回測偷用」。
  3. **FINRA 發布延遲**:月份 M 資料用其 `available_at_taipei`(優先真實 publish 時間;不可得時近似為「`M+1` 月第 20 日收盤後」,標 `quality_flags=approx_asof` 並寫入庫),僅 `<= decision_time` 可用。
  4. **Phase 2 intraday / 期貨**:帶分鐘級 `asof`,同一規則直接適用(只用 18:30 前已成形的夜盤 / 盤後值)。
- 回測必須完整複製上述 gating;`features.py` 對離線與線上套**同一** `decision_time` 邏輯。

### 5.4 回填器行為(`backfill.py`)

- 限速(每請求間隔可設定)、失敗重試、`ingest_log` 續傳;可多次執行直到補滿。
- 可指定起訖日 / 來源子集。
- **每筆寫入都必須填 §5.2 中繼欄位**:各來源 `*_asof_taipei`(用來源已知發布時間;不可得時用保守近似並標 `quality_flags=approx_asof`)、`source_updated_at`、`available_at_taipei`、`missing_fields`。
- 不做任何特徵或標籤計算(那是 `features.py` 的事),只負責把乾淨原始資料 + 中繼欄位入庫。

---

## 6. 特徵層(Phase 1,全部 causal、僅用 `available_at <= decision_time` 的已發布資訊;見 §5.3)

`features.py` 提供單一函式:輸入「某市場依日期排序的每日資料列 + 對應的另一市場資料(供跨市場特徵,套 §5.3 對齊)」,輸出每一個可預測日 T 的特徵字典。離線(對所有歷史日呼叫)與每日(對最新日呼叫)共用。

### 共同(台/美皆有)
- 報酬:1/5/20/60 日
- 均線距離:close/ma5−1、close/ma20−1、close/ma60−1、ma20/ma60−1
- 量能:`volume_ratio_20d`(前 20 日均量,不含當日;已修正)、5 日量能比
- 高點回落:自 20 日 / 60 日高點
- 已實現波動:20 日 / 60 日報酬標準差
- 星期(one-hot,捕捉週一/週五季節性,可選)

### 台股籌碼
- 融資餘額變化(占前日餘額 %)
- 三大法人合計淨額(近 60 日 z-score)
- 外資、投信淨額(分開,z-score)
- 漲跌家數比(z-score)

### 美股風險/結構
- VIX 水位、日變動、5 日變動
- QQQ 對 SPY 相對(20 日報酬差)
- IWM 對 SPY 相對(20 日報酬差)
- FINRA debit mom / yoy(point-in-time)

### 跨市場(台股模型用)
- 落後一場的美股報酬(SPY、QQQ 各 1 日)
- 落後一場的 VIX 水位

> 原本所有手填規則 → 全部變成特徵,權重交給資料學習(一次解決手填權重 Q1~Q7 的合理性問題)。

### 視窗基準政策(rolling window 是否含 T)
明確區分兩類特徵,避免把「今天」算進它自己的基準而稀釋訊號:
- **異常型(z-score / 比值)**:基準統計量用**前 N 日、不含 T**計算,再拿 T 當日值去比——這才是在量「今天的訊號相對近期是否異常」。適用:三大法人淨額 z-score、外資 / 投信 z-score、漲跌家數比 z-score、`volume_ratio_20d`(= T 量 / 前 20 日均量,不含 T,已修正)。預設 N=60(法人 / 廣度)、N=20(量能)。
- **狀態型(regime 描述)**:描述「當前狀態」,自然**含 T**(T 當日值收盤時已知,causal)。適用:均線距離、已實現波動(近 20 / 60 日報酬標準差,含 T 的報酬)、報酬類特徵。
- 每個特徵在 `feature_names` 與本文件標註其視窗與含不含 T,並由 `features.py` 集中實作,離線 / 線上一致。

### 缺值處理
- 推論時若某特徵缺(如 FINRA 當下抓不到):用**訓練集中位數**補(中位數存於模型 JSON);可對最不穩來源加一個 `*_was_missing` 旗標特徵。
- 若同時缺太多關鍵特徵 → 該市場當日**退回規則式分數**並在報告標註(graceful degradation)。

---

## 7. 標籤定義

- 對日期 T 的樣本:`label = 1 if next_trading_day_return > 0 else 0`(二元;模型學 `P(up)`)。
- 同時保存 `next_return`(實數)與「以 T 日 point-in-time 動態中性門檻」算出的三類實際方向,供三類 macro-F1 評估。
- 「下一交易日」以實際資料庫中 T 之後第一個有資料的交易日為準(自然處理假日,不靠假日曆)。

---

## 8. 模型與驗證協定

### 8.1 模型
- **主/可上線:logistic regression(L2 正則)** → 輸出機率,係數可解釋,推論可純標準庫實作。`class_weight` **不寫死**,改為超參數(`None` vs `balanced`)在 walk-forward 比較——`balanced` 可能改善分類但會**扭曲機率校準**,故選擇時主指標需同時看 macro-F1 **與 Brier / 校準**(見 §8.2、§8.5)。若選用會扭曲機率的設定,於驗證集做**機率重校準(Platt / isotonic)**,校準參數一併存入模型 JSON。
- **挑戰者:gradient boosting** → 僅在 walk-forward OOS **顯著**勝過 LR 才考慮採用;若採用,需另做「樹 → JSON + 標準庫樹評估器」匯出以維持每日純標準庫。預設 ship LR。

### 8.2 切分與 walk-forward
- 依日期排序所有樣本。
- **最終測試集 = 最近約 18 個月(可設定 12~24)**,全程不參與任何訓練/選模/調門檻,僅最後驗收跑一次。
- 其餘為 train+val 區:**擴張視窗 walk-forward**——初始訓練約前 3 年,之後每次往前推一個約 3 個月的 fold,在下一個 fold 上驗證;跨 fold 聚合 OOS 指標來(a)選超參數(L2 強度、`class_weight` None/balanced、LR/GBM、是否重校準)、(b)選機率門檻。選模主指標為 macro-F1,**但 Brier / 校準為並列硬條件**:會明顯惡化機率校準的設定不予採用。
- **嚴禁隨機 k-fold**(會把未來洩漏到過去)。
- **embargo**:train/val(與 train/test)邊界各留約 5 個交易日空檔,降低滾動視窗重疊洩漏。
- 確定設定後:於 train+val 全量重擬 → 在測試集評估一次並寫入 metrics → 最後用全部資料重擬產出**上線權重**。

### 8.3 標準化與洩漏防護
- 每個特徵的 mean/std **只用訓練資料**計算,存入模型 JSON,推論/測試套用同一組;**絕不**用測試或線上資料 fit scaler。
- 缺值中位數同樣只由訓練資料取得。

### 8.4 門檻(機率 → 三類)
- `p >= bull_threshold → 偏多`;`p <= bear_threshold → 偏空`;其間 → 震盪。
- 門檻在驗證集上選,以最大化驗收主指標(macro-F1),並確保三類都會被預測。

### 8.5 指標(沿用 `marketlib/evaluation.py`)
- 二元方向命中率、三類 macro-F1、每類 precision/recall/F1、confusion matrix。
- Brier score + 校準曲線(**同時作為 §8.2 選模硬條件,非僅事後報告**)。
- baseline 對照:always-bull、momentum、random_prior(與模型同樣本),並附 **McNemar / block bootstrap 成對顯著性**(見 §2),非僅命中率數字。
- 全部寫入 `models/metrics_*.json`,並在每日 HTML 準確率頁呈現「模型 vs baseline」。

### 8.6 決定性
- 固定隨機種子(資料切分、GBM)。記錄訓練期間與資料起訖於模型 JSON。

### 8.7 模型 JSON schema(LR)
```json
{
  "market": "tw",
  "model_type": "logistic_regression",
  "trained_at": "YYYY-MM-DD HH:MM:SS",
  "train_period": ["YYYY-MM-DD", "YYYY-MM-DD"],
  "feature_names": ["ret_1d", "..."],
  "standardization": {"mean": {"ret_1d": 0.0}, "std": {"ret_1d": 1.0}},
  "imputation_median": {"finra_mom": 0.0},
  "coefficients": {"ret_1d": 0.0},
  "intercept": 0.0,
  "class_weight": null,
  "calibration": {"method": "none", "params": {}},
  "thresholds": {"bull": 0.56, "bear": 0.46},
  "confidence_bands": {"mid": 0.04, "high": 0.10},
  "oos_metrics": {"hit_rate": 0.0, "macro_f1": 0.0, "brier": 0.0},
  "baseline_metrics": {"always_bull": 0.0, "random_prior": 0.0}
}
```
推論:`x' = (x − mean)/std`(缺值先補中位數)→ `z = intercept + Σ coef·x'` → `p = 1/(1+e^−z)` → 套門檻。

---

## 9. 每日推論整合(`daily_market_report.py`)

1. 照舊抓當日資料 → upsert 進 SQLite。
2. 用 `features.py` 算今日特徵(與訓練同一份程式,套 point-in-time 對齊)。
3. `model.py` 載入 `models/model_*.json` → 輸出 `P(up)` → 三類 + 信心。信心由 `|p − 0.5|` 決定,分界用驗證集的三分位(tertile)切成 高/中/低,分界值存入模型 JSON(`confidence_bands`),推論時直接套用。
4. 報告主預測改為模型輸出;**規則式分數保留**為「次要意見」並列入 baseline。
5. 模型缺檔或特徵大量缺 → 退回規則式並標註。
6. 正式每日驗證沿用既有機制(讀過去 JSON 預測對照實際),並改為記錄模型機率與 point-in-time 門檻。

---

## 10. Phase 2:逐一加入領先訊號

**驗證協定(關鍵:final test 不可用來決定加不加特徵)**:
- 候選特徵「保留 / 丟棄」**只在 train+val 的 walk-forward 上決定**(一次只加一個;改善需 §2 成對顯著才保留,否則丟棄,防 p-hacking)。
- **最終測試集全程不碰**,只在所有 Phase 2 決策完成後**跑一次**做最終驗收。
- 若多個候選都在 train+val 通過,組合時依**預先註冊規則**(例:依 train+val OOS 增益排序、貪婪逐一加入且每步仍需成對顯著、共線特徵保留增益較大者),**不得回頭用 final test 反覆試組合**。
- 資料取得 / 歷史回填難度高者,先評估可行性與可回填深度再納入。

**優先順序(高 → 低,含共線性考量):**
1. **台指期夜盤** — 對台股下一交易日最直接;先確認歷史可得與精確 timestamp(套 §5.3 gating)。
2. **美股期貨 ES/NQ** — 對美股下一交易日、台股隔日風險都比「落後一場的 QQQ」更有用。
3. **台積電 ADR** — 對台股科技權值影響大;但只用昨晚 ADR 收盤會與 QQQ/SOX 高度重疊,**改用 ADR 溢價(TSM 台股 vs ADR 折溢價)** 較有增量資訊。
4. **USD/TWD** — 對外資 / 台股有用,但隔日方向反應較慢,當輔助特徵。
5. **SOX/SMH** — 有用,但多與 QQQ / TSM ADR 共線,排最後驗,僅在仍有 OOS 增量時保留。

---

## 11. 重訓與維運

- 重訓:離線**每月一次或手動**(`train.py`);每日 cron 不重訓,只讀權重。
- 回填:`backfill.py` 可續傳,首次補滿歷史後,日常由每日 cron 的 upsert 維持最新。
- 版本:模型 JSON 帶 `trained_at` 與訓練期間;每日 JSON 記錄使用的模型版本,維持可追溯。

---

## 12. 風險與緩解

| 風險 | 緩解 |
|---|---|
| TWSE 多年逐日回填被限速/擋 IP | 限速、重試、`ingest_log` 續傳、分多次跑 |
| FINRA 歷史不可得 | 美股籌碼特徵深度受限 → 誠實標註、缺值退化 |
| 小樣本、低訊號 | 設定務實期待;嚴格 OOS;接受「無 edge」為合法結論 |
| regime 轉變 | walk-forward + 定期重訓 |
| train/serve skew | 特徵工程單一來源(強制) |
| Phase 2 特徵獵取過擬合 | 預先註冊指標、一次一特徵、成對顯著才留 |
| **final test 被 Phase 2 污染** | 特徵保留只用 train+val;final test 僅最後跑一次;組合用預先註冊規則 |
| look-ahead / 用到未發布資料 | `available_at<=decision_time` gating、embargo、scaler 僅 fit 訓練集 |
| **機率校準被 class_weight 扭曲** | class_weight 設為超參數、Brier/校準列為選模硬條件、必要時 Platt/isotonic 重校準 |
| **成對顯著性被高估** | 用 McNemar / block bootstrap,不用獨立樣本 SE |
| 非決定性 | 固定種子、記錄訓練後設 |

---

## 13. 交付物清單

- `marketlib/`:`core.py`、`evaluation.py`、`store.py`、`features.py`、`model.py`
- `backfill.py`、`train.py`
- 改寫後的 `daily_market_report.py`(import marketlib + 載模型)
- `data/history.db`(產出)、`models/model_{tw,us}.json`、`models/metrics_{tw,us}.json`(產出)
- 對應測試:`features.py` causality 測試(同一日離線/線上特徵一致)、**`available_at <= decision_time` gating 測試**(刻意放一筆 asof 晚於 18:30 的法人/FINRA 值,確認當日特徵不採用、退回前一日)、**視窗含不含 T 政策測試**、`model.py` 推論與 sklearn 對拍測試、walk-forward 無洩漏測試、**成對顯著性工具(McNemar / block bootstrap)單元測試**。

---

## 14. 開放問題

無(設計決策皆已確認;判斷性預設已列於各節並可由使用者否決)。
