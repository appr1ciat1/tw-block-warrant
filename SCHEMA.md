# signals.json 格式

每日 `update.py` 輸出 `signals.json`，供程式串接。頂層也內嵌一份 `schema` 欄自述。

## 頂層

| key | 說明 |
|---|---|
| `schema` | 本檔欄位自述（與本文件一致） |
| `date` | 資料日 `YYYY-MM-DD` |
| `primary_window` | 主窗名稱；頂層 `buy_signals`/`excluded_hedge`/`block_directions` 皆為此窗 |
| `params` | 本次計算參數（門檻、偵查窗、主窗等） |
| `buy_signals` | 主窗 🟢 同向買入清單（record 陣列） |
| `excluded_hedge` | 主窗 🚫 避險換倉剔除清單 |
| `block_directions` | 主窗**全部**鉅額標的判定（完整表，依 verdict 排序） |
| `windows` | `{窗名: {buy_signals, excluded_hedge, block_directions}}`——**每窗皆含完整判定表** |

## record 欄位（buy_signals / excluded_hedge / block_directions 陣列元素）

| 欄位 | 說明 |
|---|---|
| `code` / `name` | 標的代號 / 名稱 |
| `verdict` | `same_dir_buy`/`lean_buy`/`hedge_suspect`/`sell_avoid`/`unclear` |
| `score` | 排序分數（連買 × min(量能,5) × (1+鉅額方向) × 權證品質係數） |
| `window` | 該筆所屬偵查窗 |
| `blk_dir` | 鉅額方向 `buy`/`sell`/`neutral` |
| `blk_wdir` | 鉅額方向分數 [-1,1]（金額加權） |
| `blk_prem` | 金額加權溢折價（成交價/收盤−1） |
| `blk_inst_ratio` | 窗內三大法人買賣超金額 ÷ 窗內鉅額金額 |
| `streak` | 認購權證連買日數 |
| `vol_mult` | 當日認購成交額 ÷ 前 20 交易日中位數 |
| `buy_days` / `sell_days` | 窗內認購買方/賣方主導日數 |
| `call_net_win` | 窗內認購淨買壓（元） |
| `call_val_win` | 窗內認購成交額（元） |
| `put_call_win` | 窗內認售/認購成交額比 |
| `blk_n` / `blk_value` | 窗內鉅額筆數 / 總金額（元） |
| `blk_last` | 最近一筆鉅額日期 |
| `wq` | **權證品質分 [0,1]**（NaN=品質未知，不降權） |
| `call_spread` | **認購相對買賣價差** (賣−買)/賣，成交額加權（僅計雙邊都有報價者） |
| `call_quote_ratio` | **雙邊報價成交額佔認購比例**（造市雙邊穩定度） |

## 權證品質（wq）如何降權

`wq = 0.6·q_spread + 0.4·q_quote`：相對買賣價差（主，≥6% 記 0 分）、雙邊報價比（次）。
`wq` 只乘進 `score`（排序降權，最差保留 35%），**不改變 verdict 判定矩陣**——
爛流動性權證的假訊號會沉到 🟢 清單底部、並標 ⚠️。

> 註：權證「剩餘天數/到期日、IV、履約價、價內外」官方**無乾淨免費 API**
> （TWT84U 的 LastTradingDay 是全市場當日參考價基準日、非到期日），故品質只用
> 價差＋雙邊報價比。要 IV/到期需付費資料源。
