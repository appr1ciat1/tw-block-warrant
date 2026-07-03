"""
block_warrant.market_refs — 鉅額方向判定的獨立參考資料（收盤價 + 三大法人）

【獨立策略模組】不依賴 twstk，資料存放於本資料夾 data/ 下。

鉅額交易（BFIAUU）不揭露買賣方，判定方向需要「獨立於權證」的證據
（權證流本身是訊號腿，拿它判方向再拿它當訊號會循環論證）：

1. 收盤價（MI_INDEX type=ALLBUT0999 的每日收盤行情表）
   → 鉅額成交價 vs 當日收盤價的溢折價：買方急著要貨付溢價、
     賣方急著出貨給折價；貼平盤（尤其配對交易）多為關係人移轉/
     節稅/ETF 實物申贖，方向中性。市場微結構文獻的標準判別法。
2. 三大法人買賣超（fund/T86）
   → TWSE 法人統計「含」鉅額交易：當日某股法人買超金額與鉅額
     金額相當 → 法人大概率是鉅額買方（吸籌）；反之為賣方（出貨）。

歷史 CSV：
    data/stock_closes.csv  date, code, close                （全市場，供溢折價+日後訊號驗證）
    data/inst_flows.csv    date, code, foreign_net, trust_net, dealer_net, total_net（股數，買賣超）
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

CLOSE_URL = "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX?date={date}&type=ALLBUT0999&response=json"
T86_URL = "https://www.twse.com.tw/rwd/zh/fund/T86?date={date}&selectType=ALLBUT0999&response=json"
_HERE = os.path.dirname(os.path.abspath(__file__))
CLOSES_BASE = os.path.join(_HERE, "data", "stock_closes")   # 年度分檔目錄（storage.py）
INST_BASE = os.path.join(_HERE, "data", "inst_flows")
_REQUEST_PAUSE = 3.0   # TWSE 對高頻抓取會封鎖，務必保守
TIMEOUT = 60

CLOSE_COLUMNS = ["date", "code", "close"]
INST_COLUMNS = ["date", "code", "foreign_net", "trust_net", "dealer_net", "total_net"]


def _fetch_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "twstk/1.0"})
    with urllib.request.urlopen(req, context=_CTX, timeout=TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _num(s):
    try:
        return float(str(s).replace(",", ""))
    except (ValueError, TypeError):
        return float("nan")


def _check_stat(d):
    """TWSE 限流時會回假錯誤而非 429——當成非交易日跳過會留洞。
    真的查無資料才回空，其餘一律 raise（由呼叫端記錄、heal 補洞）。"""
    msg = str(d.get("stat"))
    if "沒有符合條件" in msg or "查無" in msg:
        return
    raise RuntimeError(f"TWSE 異常回應（疑似限流）: {msg}")


def fetch_stock_closes(date_str):
    """單日全市場個股收盤價（不含權證/牛熊證）；非交易日回空 DataFrame。"""
    ymd = str(date_str).replace("-", "")
    d = _fetch_json(CLOSE_URL.format(date=ymd))
    if d.get("stat") != "OK":
        _check_stat(d)
        return pd.DataFrame(columns=CLOSE_COLUMNS)
    fields, data = None, None
    for t in d.get("tables") or []:
        f = t.get("fields") or []
        if "收盤價" in f and "證券代號" in f and t.get("data"):
            fields, data = f, t["data"]
            break
    if not data:
        return pd.DataFrame(columns=CLOSE_COLUMNS)

    i_code, i_close = fields.index("證券代號"), fields.index("收盤價")
    iso = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}"
    rows = [{"date": iso, "code": str(r[i_code]).strip(), "close": _num(r[i_close])}
            for r in data]
    df = pd.DataFrame(rows, columns=CLOSE_COLUMNS)
    return df[df["close"].notna()].reset_index(drop=True)


def fetch_inst_flows(date_str):
    """單日全市場三大法人買賣超（股數）；非交易日回空 DataFrame。"""
    ymd = str(date_str).replace("-", "")
    d = _fetch_json(T86_URL.format(date=ymd))
    if d.get("stat") != "OK":
        _check_stat(d)
        return pd.DataFrame(columns=INST_COLUMNS)
    if not d.get("data"):
        return pd.DataFrame(columns=INST_COLUMNS)
    f = d["fields"]

    def _idx(*names):
        for n in names:
            if n in f:
                return f.index(n)
        return None

    # 欄位歷代變革：2012(12欄)/2015(16欄)用「外資買賣超股數」；2017-12 起(19欄)
    # 拆成「外陸資(不含外資自營商)」+「外資自營商」兩欄
    i_code = f.index("證券代號")
    i_for = _idx("外陸資買賣超股數(不含外資自營商)", "外資買賣超股數")
    i_fdl = _idx("外資自營商買賣超股數")            # 舊制無此欄 → None
    i_tru = _idx("投信買賣超股數")
    i_dlr = _idx("自營商買賣超股數")
    i_tot = _idx("三大法人買賣超股數")
    if None in (i_for, i_tru, i_dlr, i_tot):
        raise ValueError(f"T86 欄位無法辨識: {f}")
    need = max(x for x in (i_code, i_for, i_fdl, i_tru, i_dlr) if x is not None)
    iso = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}"
    rows = []
    for r in d["data"]:
        # 少數日子有短列（缺尾端欄位，例：2026-06-04 有 16 欄列）
        if len(r) <= need:
            continue
        for_n, tru_n, dlr_n = _num(r[i_for]), _num(r[i_tru]), _num(r[i_dlr])
        if len(r) > i_tot:
            tot = _num(r[i_tot])
        else:   # 缺合計欄 → 官方定義 fallback：外陸資+外資自營商+投信+自營商
            parts = [for_n, tru_n, dlr_n]
            if i_fdl is not None and len(r) > i_fdl:
                parts.append(_num(r[i_fdl]))
            tot = float(pd.Series(parts).sum(skipna=True))
        rows.append({"date": iso, "code": str(r[i_code]).strip(),
                     "foreign_net": for_n, "trust_net": tru_n,
                     "dealer_net": dlr_n, "total_net": tot})
    return pd.DataFrame(rows, columns=INST_COLUMNS)


from storage import load_history as _load_hist, save_history as _save_hist  # noqa: E402


def load_close_history(base=CLOSES_BASE, years=None):
    return _load_hist(base, CLOSE_COLUMNS, years=years)


def load_inst_history(base=INST_BASE, years=None):
    return _load_hist(base, INST_COLUMNS, years=years)


def _update_history(base, columns, fetch_fn, label,
                    first_start=None, end_date=None, deepen_to=None,
                    keep_years=None, verbose=True):
    """
    通用增量更新（邏輯同 block_trades.update_block_history）：
    - 首次執行（無歷史檔）：從 first_start 開始回補（預設今天）。
    - 之後：從歷史最後日期的次日補到 end_date（預設今天）。
    - deepen_to：往「過去」回補——補齊 deepen_to → 歷史最早日前一日的缺口。
    - 同一日重跑整日覆蓋，避免重複列。
    """
    end = pd.Timestamp(end_date) if end_date else pd.Timestamp(date.today())
    hist = _load_hist(base, columns)

    if len(hist):
        start = pd.Timestamp(hist["date"].max()) + timedelta(days=1)
    else:
        start = pd.Timestamp(first_start) if first_start else end

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
            print(f"📦 [{label}] 歷史已是最新（至 {hist['date'].max() if len(hist) else '—'}）")
        return hist

    if verbose:
        print(f"📥 [{label}] 抓取 {days[0].date()} → {days[-1].date()}（{len(days)} 個平日）...")

    new_frames = []
    fetched = 0
    for i, d in enumerate(days):
        ymd = d.strftime("%Y%m%d")
        try:
            df = fetch_fn(ymd)
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
        hist = hist[~hist["date"].isin(new_dates)]
    merged = pd.concat([hist, new_df], ignore_index=True)
    merged = _save_hist(base, merged, keep_years=keep_years, sort_cols=("date", "code"))
    if verbose:
        print(f"   ✅ [{label}] 新增 {len(new_dates)} 日，累積 {len(merged)} 筆 → {base}/")
    return merged


def update_close_history(base=CLOSES_BASE, first_start=None, end_date=None,
                         deepen_to=None, keep_years=None, verbose=True):
    """增量更新全市場收盤價。首次執行從 first_start（通常=鉅額歷史最早日）回補。"""
    return _update_history(base, CLOSE_COLUMNS, fetch_stock_closes, "收盤價",
                           first_start=first_start, end_date=end_date,
                           deepen_to=deepen_to, keep_years=keep_years, verbose=verbose)


def update_inst_history(base=INST_BASE, first_start=None, end_date=None,
                        deepen_to=None, keep_years=None, verbose=True):
    """增量更新三大法人買賣超。首次執行從 first_start（通常=鉅額歷史最早日）回補。"""
    return _update_history(base, INST_COLUMNS, fetch_inst_flows, "法人",
                           first_start=first_start, end_date=end_date,
                           deepen_to=deepen_to, keep_years=keep_years, verbose=verbose)
