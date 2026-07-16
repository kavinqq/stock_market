# 台股加入期貨（TAIFEX）特徵 — 設計文件

- 日期：2026-07-16
- 分支：feature/prediction-training
- 狀態：使用者已口頭核可「直接做」，分階段實作、慢慢修正

## 背景與目的

每日報告的失準分類器（`daily_market_report.py:classify_miss_reason`）有一句 catch-all：
「缺少資料源或公式不足：未納入消息面、期貨、利率/匯率或籌碼歷史」。

使用者目標：**盡量提高隔日大盤方向預測命中率**（承認 100% 不可能）。

現有 `FEATURE_NAMES_TW`（14 個共同 + margin/inst/breadth + 美股 3 個）幾乎都是「同一條收盤價序列的技術變形」。記憶 `tw-platt-calibration-collapse` 顯示：模型塌成「永遠偏多」是**訊號太弱**、Platt 校準誠實壓回基準率，不是 bug。要贏過隨機需要**與現貨價格線性獨立的新資訊維度**。

四個候選（期貨 / 利率匯率 / 籌碼歷史回填 / 消息面）中，**期貨**訊號最強、歷史深度足夠、無洩漏風險可控，最可能把 OOS 從「輸給隨機」推到「穩定小贏隨機」、進而通過部署 gate。

**誠實期望**：隔日方向接近隨機漫步，此變更目標是務實的小幅、穩定的贏過 baseline，不是高命中率保證。

## 範圍

**本輪做**：TAIFEX 台指期日資料 → store 欄位 → 點位（point-in-time）特徵 → 重訓 + gate → 期現價差接進規則公式。

**本輪不做（YAGNI，留待後續各自開 spec）**：散戶多空比、選擇權 Put/Call ratio、大額交易人部位、利率/匯率、消息面。

## 資料源

TAIFEX 免費日資料（歷史足夠涵蓋訓練窗 2015→今）：

1. 三大法人期貨買賣超與未平倉（台指期 TX）— 口數與未平倉餘額。
2. 台指期近月收盤 → 與加權指數算**期現價差**。
3. （選配、次要）台指期夜盤已成交段報酬。

公布時間：約當日 15:00 收盤後（早於 18:30 決策時點），故無 look-ahead 洩漏。

## Store schema（`market_daily` 新增欄位）

每個資料欄位都**帶自己的 `*_asof_taipei` 時戳**，沿用現有 `usable_value()` 閘門：

- `tx_foreign_oi_net_lots`（外資期貨未平倉淨口數）
- `tx_inst_oi_net_lots`（三大法人期貨未平倉淨口數）
- `tx_front_close`（台指期近月收盤）
- `taifex_asof_taipei`（上述三者共用時戳）
- 夜盤欄位（選配）：`tx_night_ret`、`tx_night_asof_taipei`

新增欄位一律可為 NULL；歷史缺值（如夜盤 2017-05 前）由特徵層回傳 None、模型沿用既有缺值處理。

## 新特徵（加進 `marketlib/features.py` 的 `FEATURE_NAMES_TW`）

| 特徵 | 定義 | 意義 |
|---|---|---|
| `tx_foreign_oi_z60` | 外資期貨未平倉淨口數 z-score(60d)，排除當日 | 外資布局偏多/偏空水位 |
| `tx_inst_oi_chg_z60` | 三大法人期貨未平倉淨口數**日變化** z-score(60d) | 當日籌碼動能 |
| `tx_basis_pct` | (台指期近月收盤 − 加權指數) / 加權指數 | 正價差(偏多)/逆價差(避險) |

計算全部套 `decision_time` + `asof` 檢查（與 `features.py:usable_value` 一致），排除當日做 z-score（沿用 `zscore_excluding_t`）。

## 防洩漏

- 特徵讀值一律經 `usable_value(row, value_col, "taifex_asof_taipei", decision_time)`。
- 夜盤（若納入）只用「18:30 前已成交段」報酬，spec 明確標示此時序限制；夜盤完整收盤（05:00）在 18:30 尚未發生，不得使用。

## 重訓 + 部署 gate

沿用 `train.py`：walk-forward + embargo(5) + Platt 校準 + 門檻搜尋 + `passes_deploy_gate`。

- 誠實比較「加期貨特徵前/後」的 OOS macro-F1 與各 baseline，數字自己說話。
- 通過 gate → 自動 promote 到 `models/model_tw.json`，線上下次由 rule_fallback 切成 model。
- 未通過 → 留 candidate、線上續用 rule_fallback、報告標記原因（現況即如此）。

## 規則公式即時受惠

ML 模型目前被 gate 擋著、線上跑 rule_fallback。因此**同時**把 `tx_basis_pct` 以小權重接進 `daily_market_report.py` 的 `trend_score`（或其量價/籌碼分數的加總），讓每日報告不必等 ML 通過 gate 就能受惠。低風險、可回退。

## 測試

- TAIFEX fetcher 解析：以 fixture payload 驗證欄位解析（比照現有 `fetch_twse_*` 測試風格）。
- asof 閘門：決策時點早於公布時，特徵須為 None（擋未來資料）。
- 特徵計算：z-score / 期現價差數值正確；缺值回傳 None。
- gate 行為：既有測試維持綠燈。
- 規則公式：加入 basis 項後既有 case 不回歸、方向符合預期。

## 分階段交付（慢慢修正）

1. **基礎建設**：fetcher + store schema + 點位特徵 + 單元測試（不需網路）。
2. **回填 + 重訓**：跑歷史回填腳本、重訓、輸出 OOS 前後對照、跑 gate。
3. **規則公式**：`tx_basis_pct` 接進 `trend_score`，實跑一次每日報告驗證。

## 驗收標準

- 新特徵有點位（point-in-time）防洩漏保護且有測試覆蓋。
- 重訓輸出誠實 OOS 前後對照；通過 gate 才進 `models/`。
- 規則公式納入期現價差、每日報告實跑成功。
- 既有測試全綠。
