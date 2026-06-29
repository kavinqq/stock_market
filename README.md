# Stock Market Daily Report

每天自動抓台股與美股大盤資料，產出規則式分析報告，不會呼叫 LLM，也不會消耗 token。

## 直接跑一次

```bash
cd /Users/kuanlin/stock_market
/usr/bin/python3 daily_market_report.py
```

輸出：

- `reports/latest.html`
- `reports/YYYY-MM-DD.html`
- `reports/latest.md`
- `reports/YYYY-MM-DD.md`
- `data/latest.json`
- `data/YYYY-MM-DD.json`
- `data/evaluation_history.json`

HTML 報告是單檔靜態頁，左側 sidebar 分成：

- `總覽`
- `預測`
- `準確率`
- `台股`
- `美股`
- `回推`
- `資料源`

## 每日預測與驗證

每天腳本會自動做：

1. 產生台股 / 美股下一交易日方向預測。
2. 讀取過去 `data/*.json` 內保存的預測。
3. 對照目前抓到的實際收盤資料。
4. 記錄命中、失準、實際漲跌、公式版本、預測理由、缺少資料源與失準原因。
5. 更新 `data/evaluation_history.json`，供後續用 AI 分析、調權重或訓練。

### 驗證方法

- 正式成績單：只看每天實際產出的預測，等後續收盤資料出現後再對照，累積在 `data/evaluation_history.json`。
- 價量診斷回測：只用歷史價格與量能重跑子模型，目的是檢查價量規則，不代表正式模型完整準確率。
- Baseline：報告同時列出永遠偏多、今日方向延續、依實際類別先驗隨機三個對照，避免單看命中率。
- 三分類指標：除了 accuracy，也保存 macro-F1、每類 precision/recall/F1、confusion matrix。
- 震盪判定：實際方向的中性帶用近 60 日收盤日報酬波動的一半，並限制在 0.3% 到 0.9%；每筆預測會保存當時門檻。
- 量能比：成交量或成交金額除以前 20 個交易日均量，不把當日放進分母。

失準原因會分類成例如：

- 公式門檻過保守
- 公式權重過強
- 量價權重未捕捉隔日反轉
- 缺少消息面、期貨、利率、匯率、籌碼歷史等資料源
- 交易所假日造成 target date 調整

## 安裝 crontab

```bash
cd /Users/kuanlin/stock_market
./install_cron.sh
```

預設排程：

```cron
30 18 * * 1-5 cd "/Users/kuanlin/stock_market" && /usr/bin/python3 "/Users/kuanlin/stock_market/daily_market_report.py" >> "/Users/kuanlin/stock_market/logs/market_report.log" 2>&1 # stock_market_daily_report
```

這個時間點適合台北時間：台股已收盤、TWSE 日資料通常已更新；美股則使用最近一個已收盤交易日。

## 資料源

- TWSE `FMTQIK`：台股每日成交資訊
- TWSE `MI_MARGN`：上市信用交易統計
- TWSE `BFI82U`：三大法人買賣金額統計
- Nasdaq historical API：SPY / QQQ / IWM / COMP 價量
- Cboe VIX History：VIX
- FINRA Margin Statistics：美股 margin debt 月資料

## 分析邏輯

腳本會計算：

- 1日、5日、20日、60日漲跌
- 5日、20日、60日均線
- 成交量或成交金額相對前 20 日均量
- 20日與60日高點回落幅度
- 台股融資餘額變化
- 台股三大法人買賣超
- 美股 FINRA margin debt 月變化與年變化
- VIX 風險狀態

目前沒有納入 PE、PB、EPS、殖利率等基本面，因為本腳本預測的是下一交易日方向，估值資料更適合週到月級別判斷。後續比較值得補的是台指期夜盤、美股期貨、美元/台幣、SOX、TSM ADR、利率與重大新聞事件。

報告是規則式市場分析，不是投資建議。
