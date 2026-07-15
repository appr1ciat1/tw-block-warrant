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

歷史 CSV（data/warrant_flows/YYYY.csv）欄位：
    date, underlying, underlying_name,
    call_value, call_volume, call_trades, call_up_value, call_down_value, call_n,
    put_value,  put_volume,  put_trades,  put_up_value,  put_down_value,  put_n,
    call_spread, call_quote_ratio, call_bidqty, call_askqty        ← 權證品質欄
value=成交金額(元)、volume=成交股數、n=有成交的權證檔數、
up/down_value=權證收盤上漲/下跌者的成交金額（買賣方向 proxy）。

權證品質欄（2026-07 新增，供 update.py 對爛流動性權證的假訊號降權；僅認購側）：
- call_spread      : 成交額加權「相對買賣價差」(ask-bid)/ask（僅計雙邊都有揭示報價者）
- call_quote_ratio : 雙邊報價權證佔認購成交額比例（造市雙邊穩定度）
- call_bidqty/askqty: 雙邊報價權證的最後揭示買/賣量總和
資料來源：MI_INDEX 0999 表已含最後揭示買/賣價量欄。舊年檔無這些欄 →
storage 讀取時補 NaN（品質不參與＝不降權）。
註：權證「剩餘天數/到期日」官方無乾淨 API（TWT84U 的 LastTradingDay 實為當日
參考價基準的最近交易日、非到期日，全市場同值），故不納入品質；同 IV/履約價缺口。
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
HISTORY_BASE = os.path.join(_HERE, "data", "warrant_flows")   # 年度分檔目錄（storage.py）
TIMEOUT = 60

_METRICS = ("value", "volume", "trades", "up_value", "down_value", "n")
# 認購側品質欄（僅 call；put 不需，訊號在 call）
#   call_spread      : 成交額加權相對價差（僅計「雙邊都有揭示報價」的權證）
#   call_quote_ratio : 雙邊報價權證佔認購成交額比例（造市雙邊穩定度）
#   call_bidqty/askqty: 雙邊報價權證的最後揭示買/賣量總和
QUAL_COLS = ["call_spread", "call_quote_ratio", "call_bidqty", "call_askqty"]
COLUMNS = ["date", "underlying", "underlying_name"] + [
    f"{side}_{m}" for side, _ in SIDES for m in _METRICS
] + QUAL_COLS

from twse_http import fetch_json as _http_fetch, REQUEST_PAUSE as _REQUEST_PAUSE  # noqa: E402


def _fetch_json(url):
    return _http_fetch(url, timeout=TIMEOUT)   # 含 retry-backoff（反爬蟲退避）


def _num(s):
    try:
        return float(str(s).replace(",", ""))
    except (ValueError, TypeError):
        return 0.0


def _check_stat(d):
    """TWSE 限流時會回假錯誤而非 429——當成非交易日跳過會留洞。
    真的查無資料才回空，其餘一律 raise（由呼叫端記錄、heal 補洞）。"""
    msg = str(d.get("stat"))
    if "沒有符合條件" in msg or "查無" in msg:
        return
    raise RuntimeError(f"TWSE 異常回應（疑似限流）: {msg}")


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
            _check_stat(d)   # 限流假錯誤 → raise；真非交易日 → 該側跳過
            continue          # 不 return：保留另一側已聚合的資料（put 查無≠整日無資料）
        fields, data = _pick_table(d)
        if not data:
            continue
        i_vol = fields.index("成交股數")
        i_trd = fields.index("成交筆數")
        i_val = fields.index("成交金額")
        i_chg = fields.index("漲跌(+/-)")
        i_und = fields.index("標的代號")
        i_unm = fields.index("標的名稱")
        is_call = side == "call"
        if is_call:   # 品質欄只取認購側
            i_bid = fields.index("最後揭示買價")
            i_bidq = fields.index("最後揭示買量")
            i_ask = fields.index("最後揭示賣價")
            i_askq = fields.index("最後揭示賣量")

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

            if is_call:
                bid, ask = _num(r[i_bid]), _num(r[i_ask])
                # 只用「雙邊都有揭示報價」的權證算價差（收盤單邊/缺賣價很常見，非爛）
                if ask > 0 and bid > 0 and ask >= bid:
                    rec["call_spread_vsum"] = rec.get("call_spread_vsum", 0.0) + val * (ask - bid) / ask
                    rec["call_qval"] = rec.get("call_qval", 0.0) + val   # 雙邊報價成交額
                    rec["call_bidqty"] = rec.get("call_bidqty", 0.0) + _num(r[i_bidq])
                    rec["call_askqty"] = rec.get("call_askqty", 0.0) + _num(r[i_askq])
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
        cv = rec.get("call_value", 0.0)
        qv = rec.get("call_qval", 0.0)   # 雙邊報價成交額
        row["call_spread"] = (rec.get("call_spread_vsum", 0.0) / qv) if qv > 0 else float("nan")
        row["call_quote_ratio"] = (qv / cv) if cv > 0 else float("nan")
        row["call_bidqty"] = rec.get("call_bidqty", 0.0)
        row["call_askqty"] = rec.get("call_askqty", 0.0)
        rows.append(row)
    df = pd.DataFrame(rows, columns=COLUMNS).sort_values("underlying")
    if verbose:
        print(f"   {iso}: {len(df)} 個標的有權證成交")
    return df.reset_index(drop=True)


from storage import load_history as _load_hist, save_history as _save_hist  # noqa: E402


def load_warrant_history(base=HISTORY_BASE, years=None):
    """讀取累積歷史（年度分檔＋舊制單檔自動相容）；不存在回空 DataFrame。"""
    return _load_hist(base, COLUMNS, years=years)


def update_warrant_history(base=HISTORY_BASE, backfill_days=0, end_date=None,
                           deepen_to=None, keep_years=None, verbose=True):
    """
    增量更新權證資金流歷史（邏輯同 block_trades.update_block_history）。

    - 首次執行：回補 backfill_days 個日曆日。
    - 之後：從歷史最後日期的次日補到 end_date（預設今天）。
    - deepen_to：往「過去」回補到指定日。
    - 同一日重跑整日覆蓋，避免重複列。
    """
    end = pd.Timestamp(end_date) if end_date else pd.Timestamp(date.today())
    hist = load_warrant_history(base)

    if len(hist):
        start = pd.Timestamp(hist["date"].max()) + timedelta(days=1)
    else:
        start = end - timedelta(days=int(backfill_days) if backfill_days else 0)

    days = [d for d in pd.date_range(start, end, freq="D") if d.weekday() < 5]

    if deepen_to is not None and len(hist):
        dmin = pd.Timestamp(hist["date"].min())
        dt = pd.Timestamp(deepen_to)
        if dt < dmin:
            older = [d for d in pd.date_range(dt, dmin - timedelta(days=1), freq="D")
                     if d.weekday() < 5]
            days = older + days   # 先補舊、再補新
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
    merged = _save_hist(base, merged, keep_years=keep_years,
                        sort_cols=("date", "underlying"))
    if verbose:
        print(f"   ✅ [權證] 新增 {len(new_dates)} 日，累積 {len(merged)} 筆 → {base}/")
    return merged
