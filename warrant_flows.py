"""
block_warrant.warrant_flows — 權證資金流（按標的聚合的每日認購/認售成交）

【獨立策略模組】不依賴 twstk，資料存放於本資料夾 data/ 下。

來源：TWSE 每日收盤行情 MI_INDEX
    type=0999  認購權證（不含牛證）
    type=0999P 認售權證（不含熊證）
行情表每列自帶「標的代號」，故可直接把全市場權證成交聚合到標的股。

用途：偵測「權證連續買進、量多」——真正同向被大量買的標的。
權證的主要流動性提供者是發行商，散戶/大戶買權證時發行商賣出並以現貨
避險，因此權證成交量放大 + 價漲（up_value 佔比高）可視為該標的的槓桿
多頭資金流入。但若同期現貨出現鉅額轉手（block_trades），該權證買盤
可能只是「賣現貨、買權證」的避險，update.py 會剔除。

歷史 CSV（data/warrant_flows.csv）欄位：
    date, underlying, underlying_name,
    call_value, call_volume, call_trades, call_up_value, call_down_value, call_n,
    put_value,  put_volume,  put_trades,  put_up_value,  put_down_value,  put_n
value=成交金額(元)、volume=成交股數、n=有成交的權證檔數、
up/down_value=權證收盤上漲/下跌者的成交金額（買賣方向 proxy）。
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

API_URL = "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX?date={date}&type={type}&response=json"
SIDES = (("call", "0999"), ("put", "0999P"))   # (側別, MI_INDEX type)
_HERE = os.path.dirname(os.path.abspath(__file__))
HISTORY_CSV = os.path.join(_HERE, "data", "warrant_flows.csv")
_REQUEST_PAUSE = 3.0   # TWSE 對高頻抓取會封鎖，務必保守
TIMEOUT = 60

_METRICS = ("value", "volume", "trades", "up_value", "down_value", "n")
COLUMNS = ["date", "underlying", "underlying_name"] + [
    f"{side}_{m}" for side, _ in SIDES for m in _METRICS
]


def _fetch_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "twstk/1.0"})
    with urllib.request.urlopen(req, context=_CTX, timeout=TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _num(s):
    try:
        return float(str(s).replace(",", ""))
    except (ValueError, TypeError):
        return 0.0


def _sign(html):
    """漲跌欄是 HTML（<p style=...>+</p> / <p> </p> / <p>X</p>）→ +1/-1/0。"""
    s = str(html)
    if ">+<" in s:
        return 1
    if ">-<" in s:
        return -1
    return 0


def _pick_table(payload):
    """MI_INDEX 回多張表，取含『標的代號』欄的權證行情表。"""
    for t in payload.get("tables") or []:
        fields = t.get("fields") or []
        if "標的代號" in fields and t.get("data"):
            return fields, t["data"]
    return None, None


def fetch_warrant_flows(date_str, verbose=False):
    """
    抓單日全市場認購/認售權證行情，聚合成 (標的 × 指標) DataFrame。

    Returns
    -------
    pd.DataFrame(COLUMNS)；非交易日/無資料回空 DataFrame；網路錯誤拋例外。
    """
    ymd = str(date_str).replace("-", "")
    iso = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}"
    agg = {}   # underlying -> dict

    for side, mi_type in SIDES:
        d = _fetch_json(API_URL.format(date=ymd, type=mi_type))
        if d.get("stat") != "OK":
            return pd.DataFrame(columns=COLUMNS)   # 非交易日
        fields, data = _pick_table(d)
        if not data:
            continue
        i_vol = fields.index("成交股數")
        i_trd = fields.index("成交筆數")
        i_val = fields.index("成交金額")
        i_chg = fields.index("漲跌(+/-)")
        i_und = fields.index("標的代號")
        i_unm = fields.index("標的名稱")

        for r in data:
            vol = _num(r[i_vol])
            if vol <= 0:
                continue
            und = str(r[i_und]).strip()
            rec = agg.setdefault(und, {"underlying_name": str(r[i_unm]).strip()})
            val = _num(r[i_val])
            sgn = _sign(r[i_chg])
            rec[f"{side}_value"] = rec.get(f"{side}_value", 0.0) + val
            rec[f"{side}_volume"] = rec.get(f"{side}_volume", 0.0) + vol
            rec[f"{side}_trades"] = rec.get(f"{side}_trades", 0.0) + _num(r[i_trd])
            rec[f"{side}_n"] = rec.get(f"{side}_n", 0) + 1
            if sgn > 0:
                rec[f"{side}_up_value"] = rec.get(f"{side}_up_value", 0.0) + val
            elif sgn < 0:
                rec[f"{side}_down_value"] = rec.get(f"{side}_down_value", 0.0) + val
        time.sleep(1.0)   # 同日兩個 type 間的小間隔

    if not agg:
        return pd.DataFrame(columns=COLUMNS)

    rows = []
    for und, rec in agg.items():
        row = {"date": iso, "underlying": und,
               "underlying_name": rec["underlying_name"]}
        for side, _ in SIDES:
            for m in _METRICS:
                row[f"{side}_{m}"] = rec.get(f"{side}_{m}", 0.0)
        rows.append(row)
    df = pd.DataFrame(rows, columns=COLUMNS).sort_values("underlying")
    if verbose:
        print(f"   {iso}: {len(df)} 個標的有權證成交")
    return df.reset_index(drop=True)


def load_warrant_history(path=HISTORY_CSV):
    """讀取累積歷史；不存在回空 DataFrame。"""
    if not os.path.exists(path):
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path, dtype={"underlying": str})
    df["date"] = df["date"].astype(str)
    return df


def update_warrant_history(path=HISTORY_CSV, backfill_days=0, end_date=None, verbose=True):
    """
    增量更新權證資金流歷史（邏輯同 block_trades.update_block_history）。

    - 首次執行：回補 backfill_days 個日曆日。
    - 之後：從歷史最後日期的次日補到 end_date（預設今天）。
    - 同一日重跑整日覆蓋，避免重複列。
    """
    end = pd.Timestamp(end_date) if end_date else pd.Timestamp(date.today())
    hist = load_warrant_history(path)

    if len(hist):
        start = pd.Timestamp(hist["date"].max()) + timedelta(days=1)
    else:
        start = end - timedelta(days=int(backfill_days) if backfill_days else 0)

    days = [d for d in pd.date_range(start, end, freq="D") if d.weekday() < 5]
    if not days:
        if verbose:
            print(f"📦 [權證] 歷史已是最新（至 {hist['date'].max() if len(hist) else '—'}）")
        return hist

    if verbose:
        print(f"📥 [權證] 抓取 {days[0].date()} → {days[-1].date()}（{len(days)} 個平日）...")

    new_frames = []
    fetched = 0
    for i, d in enumerate(days):
        ymd = d.strftime("%Y%m%d")
        try:
            df = fetch_warrant_flows(ymd)
        except Exception as e:  # noqa: BLE001 — 單日失敗不中斷整批
            print(f"   ⚠️ {ymd} 失敗，略過: {e}")
            time.sleep(_REQUEST_PAUSE)
            continue
        if len(df):
            new_frames.append(df)
            fetched += 1
        if verbose and (i + 1) % 5 == 0:
            print(f"   {i + 1}/{len(days)}（有資料 {fetched} 日）")
        time.sleep(_REQUEST_PAUSE)

    if not new_frames:
        if verbose:
            print("   （無新增資料）")
        return hist

    new_df = pd.concat(new_frames, ignore_index=True)
    new_dates = set(new_df["date"])
    if len(hist):
        hist = hist[~hist["date"].isin(new_dates)]
    merged = pd.concat([hist, new_df], ignore_index=True)
    merged = merged.sort_values(["date", "underlying"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    merged.to_csv(path, index=False)
    if verbose:
        print(f"   ✅ [權證] 新增 {len(new_dates)} 日，累積 {len(merged)} 筆 → {path}")
    return merged
