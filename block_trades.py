"""
block_warrant.block_trades — 鉅額交易日成交資訊（TWSE BFIAUU）

【獨立策略模組】不依賴 twstk，資料存放於本資料夾 data/ 下。

來源：https://www.twse.com.tw/zh/trading/block/bfiauu.html
API ：https://www.twse.com.tw/rwd/zh/block/BFIAUU?date=YYYYMMDD&selectType=S&response=json

用途：偵測現貨大額轉手，作為權證訊號的「避險換倉」剔除條件——
有些大戶「賣現貨、買權證」只是避險，該權證買盤非方向性看多；
凡權證強勢但同期現貨有鉅額成交者，update.py 會將其剔除不買。

提供：
- fetch_block_trades(date_str)               單日鉅額交易 DataFrame
- update_block_history(path, backfill_days)  增量累積歷史到 data/block_trades.csv
- load_block_history(path)                   讀取累積歷史

歷史 CSV 欄位：date, code, name, trade_type(配對/逐筆), price, shares, value
"""

import os
import ssl
import json
import time
import urllib.request
from datetime import date, timedelta

import pandas as pd

_CTX = ssl.create_default_context()
_CTX.verify_flags &= ~ssl.VERIFY_X509_STRICT  # OpenSSL3 嚴格檢查會擋 TWSE 憑證(缺 SKI 擴展)

API_URL = "https://www.twse.com.tw/rwd/zh/block/BFIAUU?date={date}&selectType=S&response=json"
_HERE = os.path.dirname(os.path.abspath(__file__))
HISTORY_CSV = os.path.join(_HERE, "data", "block_trades.csv")
_REQUEST_PAUSE = 3.0   # TWSE 對高頻抓取會封鎖，務必保守
TIMEOUT = 30

COLUMNS = ["date", "code", "name", "trade_type", "price", "shares", "value"]


def _fetch_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "twstk/1.0"})
    with urllib.request.urlopen(req, context=_CTX, timeout=TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _num(s):
    """'1,234,567' -> float；空值回 NaN。"""
    try:
        return float(str(s).replace(",", ""))
    except (ValueError, TypeError):
        return float("nan")


def fetch_block_trades(date_str, verbose=False):
    """
    抓單日鉅額交易（單一證券）。

    Parameters
    ----------
    date_str : str  'YYYYMMDD' 或 'YYYY-MM-DD'

    Returns
    -------
    pd.DataFrame(COLUMNS)；非交易日/無資料回空 DataFrame；網路錯誤拋例外。
    """
    ymd = str(date_str).replace("-", "")
    try:
        d = _fetch_json(API_URL.format(date=ymd))
    except Exception as e:  # noqa: BLE001 — 呼叫端決定重試
        if verbose:
            print(f"   ⚠️ 鉅額交易抓取失敗 {ymd}: {e}")
        raise
    if d.get("stat") != "OK" or not d.get("data"):
        return pd.DataFrame(columns=COLUMNS)

    iso = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}"
    rows = [
        {
            "date": iso,
            "code": r[0].strip(),
            "name": r[1].strip(),
            "trade_type": r[2].strip(),
            "price": _num(r[3]),
            "shares": _num(r[4]),
            "value": _num(r[5]),
        }
        for r in d["data"]
    ]
    return pd.DataFrame(rows, columns=COLUMNS)


def load_block_history(path=HISTORY_CSV):
    """讀取累積歷史；不存在回空 DataFrame。"""
    if not os.path.exists(path):
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path, dtype={"code": str})
    df["date"] = df["date"].astype(str)
    return df


def update_block_history(path=HISTORY_CSV, backfill_days=0, end_date=None, verbose=True):
    """
    增量更新鉅額交易歷史。

    - 首次執行（無歷史檔）：回補 backfill_days 個「日曆日」（非交易日自動略過）。
    - 之後每日執行：從歷史最後日期的次日補到 end_date（預設今天）。
    - 同一日重跑會整日覆蓋（先刪舊列再寫入），避免重複列。

    Returns
    -------
    pd.DataFrame 更新後完整歷史。
    """
    end = pd.Timestamp(end_date) if end_date else pd.Timestamp(date.today())
    hist = load_block_history(path)

    if len(hist):
        start = pd.Timestamp(hist["date"].max()) + timedelta(days=1)
    else:
        start = end - timedelta(days=int(backfill_days) if backfill_days else 0)

    days = pd.date_range(start, end, freq="D")
    # 週末必非交易日，直接跳過省 API 次數；國定假日由 stat!=OK 過濾
    days = [d for d in days if d.weekday() < 5]
    if not days:
        if verbose:
            print(f"📦 [鉅額] 歷史已是最新（至 {hist['date'].max() if len(hist) else '—'}）")
        return hist

    if verbose:
        print(f"📥 [鉅額] 抓取 {days[0].date()} → {days[-1].date()}（{len(days)} 個平日）...")

    new_frames = []
    fetched = 0
    for i, d in enumerate(days):
        ymd = d.strftime("%Y%m%d")
        try:
            df = fetch_block_trades(ymd)
        except Exception as e:  # noqa: BLE001 — 單日失敗不中斷整批
            print(f"   ⚠️ {ymd} 失敗，略過: {e}")
            time.sleep(_REQUEST_PAUSE)
            continue
        if len(df):
            new_frames.append(df)
            fetched += 1
        if verbose and (i + 1) % 10 == 0:
            print(f"   {i + 1}/{len(days)}（有資料 {fetched} 日）")
        time.sleep(_REQUEST_PAUSE)

    if not new_frames:
        if verbose:
            print("   （無新增資料）")
        return hist

    new_df = pd.concat(new_frames, ignore_index=True)
    new_dates = set(new_df["date"])
    if len(hist):
        hist = hist[~hist["date"].isin(new_dates)]  # 同日覆蓋
    merged = pd.concat([hist, new_df], ignore_index=True)
    merged = merged.sort_values(["date", "code"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    merged.to_csv(path, index=False)
    if verbose:
        print(f"   ✅ [鉅額] 新增 {len(new_df)} 筆（{len(new_dates)} 日），"
              f"累積 {len(merged)} 筆 → {path}")
    return merged
