#!/usr/bin/env python3
"""
block_warrant/update.py — 鉅額成交方向判定 × 權證同向訊號：每日更新

【獨立策略】與 twstk / 既有四策略完全無關，所有資料與輸出都在 block_warrant/ 內。

═══ 鉅額買賣方向怎麼判定（核心問題）═══════════════════════════════
鉅額交易（BFIAUU）不揭露買賣雙方。方向必須用「獨立於權證」的證據推定
（若用權證流判方向、再用權證流當訊號，是循環論證）。兩條獨立證據：

1. 溢折價：鉅額成交價 vs 當日收盤價（data/stock_closes.csv）
   買方急著要貨 → 付溢價成交；賣方急著出貨 → 給折價成交。
   貼平盤（尤其配對交易）多為關係人移轉/節稅/ETF 實物申贖 → 方向中性。
2. 法人比對：同日三大法人買賣超（data/inst_flows.csv，T86「含」鉅額交易）
   當日法人買超金額與鉅額金額相當 → 法人大概率是鉅額買方（吸籌）；
   反之賣超相當 → 法人是賣方（出貨）。

每筆鉅額：dir = (溢折價證據 ±1/0 + 法人證據 ±1/0) / 2 ∈ [-1, +1]
每檔標的：視窗內以成交金額加權平均 → blk_dir ∈ {buy / sell / neutral}

═══ 訊號（同向操作才買）═══════════════════════════════════════════
權證腿（連買+量多）判斷這檔的槓桿多頭資金流；與鉅額方向交叉：

- 🟢 同向買入：鉅額=買 且 認購連買 ≥ streak-min 日、量能 ≥ vol-mult×
  （大戶鉅額吸籌 + 權證同向被連續大量買 → 跟進買入）
- 🟡 偏買：鉅額=買+權證偏買（未達強門檻），或鉅額中性(移轉類)+權證強
- 🚫 避險換倉（剔除）：鉅額=賣 但權證有買盤 → 「賣現貨、買權證」嫌疑，
  權證買盤非新多頭資金 → 不買（這正是要淘汰的形態）
- 🔴 出貨避開：鉅額=賣 且權證無買盤
- ⚪ 不明：權證成交太少（< min-call-value）或證據矛盾

═══ 時間段輪動偵查窗 ═══════════════════════════════════════════
每日對五個偵查窗各判定一次（--windows 可調）：
  近 5 / 10 / 15 / 20 個「交易日」＋ W3（上週三→本輪，週選結算週期錨定）。
主窗（--primary-window，預設 10）出主表；多窗共振矩陣顯示跨窗一致性
（🟢 共振 ≥2 窗更可信）。每窗判定都累積進 signal_history.csv（window 欄），
供日後統計「哪個窗長最預測」。

輸出：report.html / index.html + signals.json
     + data/signal_history.csv（每日 窗×標的 判定累積，供日後驗證各級後續報酬）

用法（任何工作目錄皆可）：
    python update.py                    # 增量更新 + 出報表
    python update.py --backfill 90      # 首次回補 90 個日曆日
    python update.py --no-update        # 只用既有 CSV 出報表
    python update.py --deepen-to 2025-07-01   # 歷史往過去加深（供 Actions 跑）
"""

import argparse
import html
import json
import os
import re
import sys
import time

import pandas as pd


def _enable_utf8_console():
    """Windows 主控台預設 cp950，print emoji 會 UnicodeEncodeError 中斷腳本。
    入口把 stdout/stderr 轉 UTF-8（errors='replace' 保底不炸）。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def _esc(x):
    """HTML 轉義動態欄位（TWSE 名稱/交易別等，防破版/注入）。"""
    return html.escape(str(x))


_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)   # 讓 update.py 從任何 CWD 都能 import 同資料夾模組

import block_trades as _bt                                                # noqa: E402
import warrant_flows as _wf                                               # noqa: E402
import market_refs as _mr                                                 # noqa: E402
from block_trades import update_block_history, load_block_history        # noqa: E402
from warrant_flows import update_warrant_history, load_warrant_history   # noqa: E402
from market_refs import (update_close_history, load_close_history,       # noqa: E402
                         update_inst_history, load_inst_history)
from storage import load_history as _load_hist, save_history as _save_hist  # noqa: E402

HTML_FILE = os.path.join(_HERE, "report.html")
INDEX_FILE = os.path.join(_HERE, "index.html")   # GitHub Pages 首頁 = 報表
SIGNALS_JSON = os.path.join(_HERE, "signals.json")
SIGNAL_HISTORY_BASE = os.path.join(_HERE, "data", "signal_history")   # 年度分檔
STOCK_CODE = re.compile(r"^\d{4,6}$")   # 個股/ETF；排除 IX0001 等指數標的
SLICE_DAYS = 200   # 訊號計算只需近期資料；深歷史留給 validate.py

# signals.json 正式 schema（內嵌於輸出 "schema" 欄，供串接者對照；詳見 SCHEMA.md）
SIGNALS_SCHEMA = {
    "date": "資料日 (YYYY-MM-DD)",
    "primary_window": "主窗名稱（頂層 buy_signals/excluded_hedge/block_directions 皆此窗）",
    "params": "本次計算參數（門檻/偵查窗等）",
    "buy_signals": "主窗 🟢 同向買入清單（record 陣列，欄位見 record_fields）",
    "excluded_hedge": "主窗 🚫 避險換倉剔除清單",
    "block_directions": "主窗全部鉅額標的判定（完整表）",
    "windows": "各偵查窗 → {buy_signals, excluded_hedge, block_directions}（每窗完整表）",
    "record_fields": {
        "code": "標的代號", "name": "標的名稱",
        "verdict": "same_dir_buy/lean_buy/hedge_suspect/sell_avoid/unclear",
        "score": "排序分數（含權證品質降權）", "window": "偵查窗",
        "blk_dir": "鉅額方向 buy/sell/neutral", "blk_wdir": "方向分數 [-1,1]",
        "blk_prem": "金額加權溢折價", "blk_inst_ratio": "窗內法人金額/鉅額金額",
        "streak": "認購連買日", "vol_mult": "當日認購額/前20日中位數倍數",
        "buy_days": "窗內認購買方日", "sell_days": "窗內認購賣方日",
        "call_net_win": "窗內認購淨買壓(元)", "call_val_win": "窗內認購成交額(元)",
        "put_call_win": "窗內認售/認購額比", "blk_n": "窗內鉅額筆數",
        "blk_value": "窗內鉅額金額(元)", "blk_last": "最近鉅額日",
        "wq": "權證品質分 [0,1]（NaN=未知）",
        "call_spread": "認購相對買賣價差（雙邊報價權證，成交額加權）",
        "call_quote_ratio": "雙邊報價成交額佔認購比例",
    },
}


# ── 鉅額方向：獨立證據（溢折價 + 法人）───────────────────────────
def classify_block_direction(block, closes, inst, cutoff,
                             prem_th=0.005, inst_ratio_th=0.5):
    """
    對視窗內每筆鉅額成交推定買賣方向，再按標的以金額加權聚合。

    Parameters
    ----------
    prem_th       : 溢折價門檻（|成交價/收盤價-1| ≥ 此值才算方向證據）
    inst_ratio_th : 法人比門檻（|當日法人買賣超金額/鉅額金額| ≥ 此值才算證據）

    Returns
    -------
    trades : pd.DataFrame 逐筆（含 prem / inst_ratio / dir_score）
    stocks : pd.DataFrame 每檔一列（blk_dir / blk_wdir / blk_prem / blk_inst_ratio）
    """
    b = block[block["date"] >= cutoff].copy()
    if b.empty:
        return b, pd.DataFrame()

    b = b.merge(closes, on=["date", "code"], how="left")
    b = b.merge(inst[["date", "code", "total_net"]], on=["date", "code"], how="left")

    b["prem"] = b["price"] / b["close"] - 1.0
    # 法人比 = 當日三大法人買賣超金額 ÷ 當日該股鉅額「總」金額。
    # total_net 是「整檔整日」一個值（T86 每 date,code 一列）；一檔當日常有多筆
    # 鉅額，分母必須用檔級當日鉅額總額，不能用單筆 value（否則 inst 證據被筆數
    # 放大、過度觸發、污染方向判定，並使聚合 blk_inst_ratio 被筆數 n 重複放大）。
    day_val = b.groupby(["date", "code"])["value"].transform("sum")
    b["inst_value"] = b["total_net"] * b["close"]          # 法人買賣超金額(近似)
    b["inst_ratio"] = b["inst_value"] / day_val.where(day_val > 0)

    price_ev = pd.Series(0.0, index=b.index)
    price_ev[b["prem"] >= prem_th] = 1.0
    price_ev[b["prem"] <= -prem_th] = -1.0
    inst_ev = pd.Series(0.0, index=b.index)
    inst_ev[b["inst_ratio"] >= inst_ratio_th] = 1.0
    inst_ev[b["inst_ratio"] <= -inst_ratio_th] = -1.0
    b["price_ev"] = price_ev
    b["inst_ev"] = inst_ev
    b["dir_score"] = (price_ev + inst_ev) / 2.0            # ∈ {-1,-0.5,0,0.5,1}

    w = b["value"].clip(lower=0)
    tmp = pd.DataFrame({
        "code": b["code"], "w": w, "wd": w * b["dir_score"],
        "wp": w * b["prem"], "wp_w": w.where(b["prem"].notna()),
        "wi": w * b["inst_ratio"], "wi_w": w.where(b["inst_ratio"].notna()),
    })
    g = tmp.groupby("code").sum(min_count=1)
    tot = g["w"].replace(0, 1.0)
    stocks = pd.DataFrame({
        "code": g.index,
        "blk_wdir": (g["wd"] / tot).fillna(0.0),
        "blk_prem": g["wp"] / g["wp_w"],           # 金額加權(僅計有收盤價的筆)
        "blk_inst_ratio": g["wi"] / g["wi_w"],
    }).reset_index(drop=True)
    stocks["blk_dir"] = "neutral"
    stocks.loc[stocks["blk_wdir"] >= 0.25, "blk_dir"] = "buy"
    stocks.loc[stocks["blk_wdir"] <= -0.25, "blk_dir"] = "sell"
    return b, stocks


# ── 時間段輪動偵查窗 ────────────────────────────────────────────
def window_cutoff(win, dates):
    """
    偵查窗 → cutoff（ISO 日期字串，視窗 = date >= cutoff）。

    win : int/str  數字 N = 最近 N 個「交易日」；'W3' = 上個週三起
          （週三→下週三錨定窗；latest 為週三時涵蓋完整一輪週期）
    dates : 已排序的交易日 ISO 字串 list（最新在最後）
    """
    s = str(win).strip().upper()
    if s == "W3":
        latest = pd.Timestamp(dates[-1])
        off = (latest.weekday() - 2) % 7 or 7   # Mon=0…Wed=2；當天是週三→回推整週
        return (latest - pd.Timedelta(days=off)).strftime("%Y-%m-%d")
    n = int(s)
    return dates[-n] if len(dates) >= n else dates[0]


# ── 權證方向證據 + 鉅額標的判定 ─────────────────────────────────
def build_evidence(warr, block, blk_dirs, cutoff, vol_lookback=20):
    """
    以「視窗內（date >= cutoff）有鉅額成交的標的」為母體，逐檔計算權證方向
    證據，並併入鉅額方向（blk_dirs，由 classify_block_direction 產生）。

    Returns
    -------
    ev : pd.DataFrame  每個鉅額標的一列：
         blk_n / blk_value / blk_last / blk_dir / blk_wdir / blk_prem / blk_inst_ratio
         streak / vol_mult / buy_days / sell_days / call_net_win /
         call_val_win / put_call_win / buy_ratio（權證證據）/ win_len（窗內交易日數）
    latest : str  最新權證資料日
    """
    warr = warr[warr["underlying"].astype(str).str.match(STOCK_CODE)].copy()
    if warr.empty or not len(block):
        return pd.DataFrame(), None
    dates = sorted(warr["date"].unique())
    latest = dates[-1]
    win_dates = [d for d in dates if d >= cutoff]   # 判定視窗（交易日）

    # 母體：視窗內有鉅額成交的標的
    recent_blk = block[block["date"] >= cutoff]
    blk = recent_blk.groupby("code").agg(
        blk_n=("value", "size"), blk_value=("value", "sum"),
        blk_last=("date", "max"), name=("name", "last"))
    if blk.empty:
        return pd.DataFrame(), latest

    num_cols = [c for c in warr.columns if c not in ("date", "underlying", "underlying_name")]
    warr_g = {u: g for u, g in warr.groupby("underlying")}

    _QNAN = {"call_spread": float("nan"), "call_quote_ratio": float("nan"),
             "call_bidqty": float("nan"), "call_askqty": float("nan")}

    def _quality(g0):
        """取『最新資料日』該標的認購權證品質（缺欄/未交易 → NaN，不參與降權）。"""
        if g0 is None:
            return dict(_QNAN)
        qr = g0[g0["date"] == latest]
        if not len(qr):
            return dict(_QNAN)
        q = qr.iloc[-1]
        return {k: (float(q[k]) if (k in q and pd.notna(q[k])) else float("nan"))
                for k in _QNAN}

    rows = []
    for code, b in blk.iterrows():
        g0 = warr_g.get(code)
        qual = _quality(g0)
        if g0 is None:
            rows.append({"code": code, "name": b["name"], "blk_n": int(b.blk_n),
                         "blk_value": float(b.blk_value), "blk_last": b.blk_last,
                         "streak": 0, "vol_mult": float("nan"), "buy_days": 0,
                         "sell_days": 0, "call_net_win": 0.0, "call_val_win": 0.0,
                         "put_call_win": float("nan"), "buy_ratio": 0.0, **qual})
            continue

        g = g0.set_index("date")[num_cols].reindex(dates).fillna(0.0)
        cv = g["call_value"]
        net = g["call_up_value"] - g["call_down_value"]

        # 現在進行中的連買天數（全歷史往回數）
        buy = ((cv > 0) & (net > 0)).tolist()
        streak = 0
        for x in reversed(buy):
            if not x:
                break
            streak += 1

        # 量能：當日 vs 前 vol_lookback 日中位數
        last_cv = float(cv.iloc[-1])
        base = cv.iloc[-(vol_lookback + 1):-1]
        base = base[base > 0]
        vol_mult = (last_cv / float(base.median())) if len(base) >= 5 else float("nan")

        # 判定視窗內的方向證據
        w = g.loc[g.index.isin(win_dates)]
        w_net = w["call_up_value"] - w["call_down_value"]
        call_val_win = float(w["call_value"].sum())
        put_val_win = float(w["put_value"].sum())

        rows.append({
            "code": code, "name": b["name"], "blk_n": int(b.blk_n),
            "blk_value": float(b.blk_value), "blk_last": b.blk_last,
            "streak": streak, "vol_mult": vol_mult,
            "buy_days": int(((w["call_value"] > 0) & (w_net > 0)).sum()),
            "sell_days": int(((w["call_value"] > 0) & (w_net < 0)).sum()),
            "call_net_win": float(w_net.sum()),
            "call_val_win": call_val_win,
            "put_call_win": (put_val_win / call_val_win) if call_val_win > 0 else float("nan"),
            "buy_ratio": (float(g["call_up_value"].iloc[-1]) / last_cv) if last_cv > 0 else 0.0,
            **qual,
        })
    ev = pd.DataFrame(rows)
    if len(blk_dirs):
        ev = ev.merge(blk_dirs, on="code", how="left")
    else:
        for c in ("blk_wdir", "blk_prem", "blk_inst_ratio"):
            ev[c] = float("nan")
        ev["blk_dir"] = "neutral"
    ev["blk_dir"] = ev["blk_dir"].fillna("neutral")
    ev["blk_wdir"] = ev["blk_wdir"].fillna(0.0)
    ev["win_len"] = len(win_dates)
    return ev, latest


def warrant_quality(ev):
    """
    認購權證品質分 ∈ [0,1]（NaN = 品質未知，不降權）。爛流動性=假訊號，用來降權：
    - 相對買賣價差 q_spread（主）：雙邊報價權證的成交額加權價差，≥6% 記 0 分。
    - 雙邊報價比 q_quote（次）：雙邊都掛報價的成交額佔比（造市穩定度）。
    價差 NaN（無任何雙邊報價權證）→ 回 NaN 視為未知不降權（保守）。
    （剩餘天數/到期日官方無乾淨 API，不納入；見 warrant_flows 說明。）
    """
    q_spread = (1.0 - ev["call_spread"] / 0.06).clip(0, 1)
    q_quote = ev["call_quote_ratio"].clip(0, 1).fillna(0.5)
    return 0.6 * q_spread + 0.4 * q_quote


def classify(ev, args):
    """
    鉅額方向 × 權證證據 → verdict：
        same_dir_buy(🟢 同向買入) / lean_buy(🟡 偏買) /
        hedge_suspect(🚫 賣現貨買權證嫌疑→剔除) /
        sell_avoid(🔴 出貨避開) / unclear(⚪ 不明)
    """
    ev = ev.copy()
    # 流動性底線隨窗長比例調整（<10 交易日的短窗按比例放寬，≥10 維持原門檻）
    thin_th = args.min_call_value * (ev["win_len"].clip(upper=10) / 10.0)
    thin = ev["call_val_win"] < thin_th
    strong = ((ev["streak"] >= args.streak_min)
              & (ev["vol_mult"].fillna(0) >= args.vol_mult))
    bearish = (ev["call_net_win"] < 0) | (ev["put_call_win"].fillna(0) >= 1.0)
    lean_buy = (ev["call_net_win"] > 0) & (ev["buy_days"] > ev["sell_days"])
    blk_buy = ev["blk_dir"] == "buy"
    blk_sell = ev["blk_dir"] == "sell"
    blk_neutral = ev["blk_dir"] == "neutral"

    ev["verdict"] = "unclear"
    ev.loc[~thin & blk_sell, "verdict"] = "sell_avoid"
    ev.loc[~thin & blk_sell & (strong | lean_buy), "verdict"] = "hedge_suspect"
    ev.loc[~thin & blk_buy & lean_buy & ~bearish, "verdict"] = "lean_buy"
    ev.loc[~thin & blk_neutral & strong & ~bearish, "verdict"] = "lean_buy"
    ev.loc[~thin & blk_buy & strong & ~bearish, "verdict"] = "same_dir_buy"

    # 權證品質分：爛流動性權證的假訊號降權（品質未知→factor=1 不降權；最差保留 35%）
    ev["wq"] = warrant_quality(ev)
    qfac = 0.35 + 0.65 * ev["wq"].fillna(1.0)

    # 分數 = 權證強度 ×（1 + 鉅額方向加權）× 品質係數，僅供排序（判定矩陣不受影響）
    ev["score"] = (ev["streak"] * ev["vol_mult"].fillna(0).clip(upper=5.0)
                   * (1.0 + ev["blk_wdir"].fillna(0)) * qfac)
    return ev


def append_signal_history(ev, latest, base=SIGNAL_HISTORY_BASE):
    """
    把當日全部（窗×標的）判定 append 到歷史（同日重跑整日覆蓋），供日後統計後續報酬。
    回傳 (今日筆數, 覆蓋掉的舊同日筆數)——讓呼叫端不會把「同日覆蓋」誤報成「淨新增」。
    """
    cols = ["date", "window", "code", "name", "verdict", "score",
            "blk_dir", "blk_wdir", "blk_prem", "blk_inst_ratio",
            "streak", "vol_mult", "buy_days", "sell_days", "call_net_win",
            "call_val_win", "put_call_win", "buy_ratio", "blk_n", "blk_value",
            "blk_last", "win_len", "wq", "call_spread", "call_quote_ratio"]
    today = ev.copy()
    today.insert(0, "date", latest)
    today = today.reindex(columns=cols)

    hist = _load_hist(base, cols, years=1)   # 去重只需當年檔（舊制單檔自動併入遷移）
    old_same = int((hist["date"] == latest).sum())   # 檔內既有的同日列數
    hist = hist[hist["date"] != latest]
    merged = pd.concat([hist, today], ignore_index=True)
    _save_hist(base, merged, sort_cols=("date", "window", "code"))
    return len(today), old_same


# ── 補洞：TWSE 限流時 stat!=OK 會被當「非交易日」靜默跳過而留洞 ──
def heal_recent(block, warr, closes, inst, days=45, keep_years=None, verbose=True):
    """
    以四資料源近 days 天的日期「聯集」為交易日參考，補抓各源缺少的日期。
    （例：2026-06-04 T86 曾因短列失敗、MI_INDEX 限流時回假錯誤 → 都會留洞，
    每日執行時自動回填；抓不到就留待下次。）
    """
    if not days:
        return block, warr, closes, inst
    from collections import Counter
    cnt = Counter()
    for df in (block, warr, closes, inst):
        if len(df):
            cnt.update(set(df["date"]))
    if not cnt:
        return block, warr, closes, inst
    cutoff = (pd.Timestamp(max(cnt)) - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    # 多數決：≥2 源皆有的日期才當可信交易日參考，避免單源幽靈日（盤中幽靈/某源
    # 部分表）被當交易日而每天對其他源做無效重試（含退避）浪費時間。
    ref = {d for d, c in cnt.items() if c >= 2 and d >= cutoff}

    def _heal(name, df, fetch_fn, base, sort_cols):
        own = set(df["date"])
        missing = sorted(d for d in ref if d not in own)
        if not missing:
            return df
        if verbose:
            print(f"🩹 [{name}] 補洞 {len(missing)} 日：{'、'.join(missing)}")
        frames = []
        for d in missing:
            try:
                got = fetch_fn(d.replace("-", ""))
                if len(got):
                    frames.append(got)
            except Exception as e:  # noqa: BLE001 — 補不到留待下次
                print(f"   ⚠️ {d} 補洞失敗: {e}")
            time.sleep(3.0)
        if frames:
            df = pd.concat([df] + frames, ignore_index=True)
            df = _save_hist(base, df, keep_years=keep_years, sort_cols=sort_cols)
        return df

    block = _heal("鉅額", block, _bt.fetch_block_trades, _bt.HISTORY_BASE, ("date", "code"))
    warr = _heal("權證", warr, _wf.fetch_warrant_flows, _wf.HISTORY_BASE, ("date", "underlying"))
    closes = _heal("收盤價", closes, _mr.fetch_stock_closes, _mr.CLOSES_BASE, ("date", "code"))
    inst = _heal("法人", inst, _mr.fetch_inst_flows, _mr.INST_BASE, ("date", "code"))
    return block, warr, closes, inst


# ── 報表 ────────────────────────────────────────────────────────
_TAG = {"same_dir_buy": "🟢 同向買入", "lean_buy": "🟡 偏買",
        "hedge_suspect": "🚫 避險換倉", "sell_avoid": "🔴 出貨避開",
        "unclear": "⚪ 不明"}
_CLS = {"same_dir_buy": ' class="hl"', "hedge_suspect": ' class="hl3"',
        "sell_avoid": ' class="hl2"', "lean_buy": "", "unclear": ""}
_DIR = {"buy": "▲買", "sell": "▽賣", "neutral": "–中性"}


def _fmt_m(v):
    """元 → 百萬元字串。"""
    return f"{v / 1e6:,.1f}"


def _fmt_ratio(v, cap=10.0):
    """法人比顯示：|值|≥cap 顯示 ±cap↑（法人流量遠大於鉅額，超過此意義不大）。"""
    if pd.isna(v):
        return "—"
    if v >= cap:
        return f"+{cap:.0f}↑"
    if v <= -cap:
        return f"−{cap:.0f}↑"
    return f"{v:+.2f}"


def _stk_cell(code, name):
    """標的欄：連 Yahoo 股價頁（上市 .TW），並 html 轉義。"""
    c, n = _esc(code), _esc(name)
    return (f'<td class="stk"><a href="https://tw.stock.yahoo.com/quote/{c}.TW" '
            f'target="_blank" rel="noopener"><b>{c}</b> {n}</a></td>')


def _quality_cell(r):
    """權證品質欄：優/中/⚠️差 + 價差% · 雙邊報價比（品質未知→—）。"""
    wq = getattr(r, "wq", float("nan"))
    if pd.isna(wq):
        return "<td>—</td>"
    tier = "優" if wq >= 0.7 else ("中" if wq >= 0.4 else "⚠️差")
    bits = []
    if pd.notna(getattr(r, "call_spread", float("nan"))):
        bits.append(f"價差{r.call_spread * 100:.1f}%")
    if pd.notna(getattr(r, "call_quote_ratio", float("nan"))):
        bits.append(f"雙邊{r.call_quote_ratio * 100:.0f}%")
    sub = f"<span class='qsub'> {'·'.join(bits)}</span>" if bits else ""
    cls = ' class="qbad"' if wq < 0.4 else ""
    return f"<td{cls}>{tier}{sub}</td>"


def _row_html(r):
    vm = f"{r.vol_mult:.1f}×" if pd.notna(r.vol_mult) else "—"
    pc = f"{r.put_call_win:.2f}" if pd.notna(r.put_call_win) else "—"
    prem = f"{r.blk_prem * 100:+.2f}%" if pd.notna(r.blk_prem) else "—"
    ir = _fmt_ratio(r.blk_inst_ratio)
    return (f"<tr{_CLS[r.verdict]}><td>{_TAG[r.verdict]}</td>"
            f"{_stk_cell(r.code, r.name)}"
            f"<td>{_DIR.get(r.blk_dir, '–')} ({r.blk_wdir:+.2f})</td>"
            f"<td>{prem}</td><td>{ir}</td>"
            f"<td>{r.blk_n} 筆 / {_fmt_m(r.blk_value)}M</td><td>{_esc(r.blk_last[5:])}</td>"
            f"<td>{r.streak}</td><td>{vm}</td><td>{r.buy_days}/{r.sell_days}</td>"
            f"<td>{_fmt_m(r.call_net_win)}</td><td>{pc}</td>"
            f"{_quality_cell(r)}</tr>")


_THEAD = ("<tr><th>判定</th><th>標的</th><th>鉅額方向</th><th>溢折價</th><th>法人比</th>"
          "<th>鉅額(視窗)</th><th>最近鉅額</th>"
          "<th>連買(日)</th><th>量能</th><th>買/賣日</th><th>認購淨買壓(百萬)</th>"
          "<th>Put/Call</th><th>品質</th></tr>")
_NCOL = 13


_WIN_LABEL = {"W3": "週三→下週三"}


def _win_label(w):
    return _WIN_LABEL.get(str(w).upper(), f"近{w}交易日")


def _resonance_card(evs, windows, primary):
    """多窗共振矩陣：每檔 × 每窗的判定，統計 🟢/🚫 共振數。"""
    frames = {w: evs[w].set_index("code") for w in windows if w in evs}
    if not frames:
        return ""
    codes = {}
    for w, df in frames.items():
        for code, r in df.iterrows():
            if r["verdict"] == "unclear":
                continue
            codes.setdefault(code, {"name": r["name"], "blk_value": 0.0})
            codes[code]["blk_value"] = max(codes[code]["blk_value"], float(r["blk_value"]))
    if not codes:
        return ""

    rows = []
    for code, meta in codes.items():
        cells, n_buy, n_hedge = [], 0, 0
        for w in windows:
            df = frames.get(w)
            if df is None or code not in df.index:
                cells.append("·")
                continue
            v = df.loc[code, "verdict"]
            cells.append(_TAG[v].split()[0])
            n_buy += v == "same_dir_buy"
            n_hedge += v == "hedge_suspect"
        rows.append((n_buy, n_hedge, meta["blk_value"], code, meta["name"], cells))
    rows.sort(key=lambda x: (-x[0], -x[1], -x[2]))

    head = "".join(f"<th>{_win_label(w)}</th>" for w in windows)
    body = []
    for n_buy, n_hedge, _, code, name, cells in rows[:80]:
        cls = ' class="hl"' if n_buy >= 2 else (' class="hl3"' if n_hedge >= 2 else "")
        body.append(f"<tr{cls}>{_stk_cell(code, name)}"
                    + "".join(f"<td>{c}</td>" for c in cells)
                    + f"<td>{n_buy}</td><td>{n_hedge}</td></tr>")
    return f"""
<div class="card">
 <h2>⏱ 時間段輪動偵查（多窗共振）</h2>
 <p class="gsub">同一標的在 {len(windows)} 個偵查窗（{' / '.join(_win_label(w) for w in windows)}）
 的判定。🟢 共振 ≥ 2 窗 = 同向買入訊號跨窗成立（更可信）；🚫 共振 ≥ 2 窗 = 避險換倉形態跨窗成立。
 「·」= 該窗內無鉅額成交（不在母體）。主窗 = {_win_label(primary)}。</p>
 <div class="tbl"><table>
  <tr><th>標的</th>{head}<th>🟢共振</th><th>🚫共振</th></tr>
  {''.join(body)}
 </table></div>
</div>"""


def write_report(ev, trades, latest, args, evs=None, windows=(), primary=""):
    buy_tbl = ev[ev["verdict"] == "same_dir_buy"].sort_values("score", ascending=False)
    hedge_tbl = ev[ev["verdict"] == "hedge_suspect"].sort_values("blk_value", ascending=False)
    all_tbl = ev.copy()
    all_tbl["verdict"] = pd.Categorical(
        all_tbl["verdict"],
        categories=["same_dir_buy", "lean_buy", "hedge_suspect", "sell_avoid", "unclear"])
    all_tbl = all_tbl.sort_values(["verdict", "blk_value"], ascending=[True, False])

    def rows_of(df):
        out = [_row_html(r) for r in df.itertuples()]
        return "\n".join(out) or (f'<tr><td colspan="{_NCOL}" style="color:#64748b">'
                                  '（無符合條件標的）</td></tr>')

    # 近 N 日鉅額交易逐筆明細（金額前 30，含每筆方向證據）
    blk_rows = ""
    if len(trades):
        recent = trades.sort_values("value", ascending=False).head(30)
        ev_glyph = {1.0: "▲", 0.5: "▲", 0.0: "–", -0.5: "▽", -1.0: "▽"}
        rws = []
        for r in recent.itertuples():
            prem = f"{r.prem * 100:+.2f}%" if pd.notna(r.prem) else "—"
            ir = _fmt_ratio(r.inst_ratio)
            rws.append(
                f"<tr><td>{_esc(r.date)}</td>{_stk_cell(r.code, r.name)}<td>{_esc(r.trade_type)}</td>"
                f"<td>{r.price:,.2f}</td><td>{prem}</td><td>{ir}</td>"
                f"<td>{ev_glyph.get(r.dir_score, '–')} {r.dir_score:+.1f}</td>"
                f"<td>{r.shares / 1000:,.0f}</td><td>{_fmt_m(r.value)}</td></tr>")
        blk_rows = "\n".join(rws)

    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>鉅額成交方向判定 × 權證同向訊號</title>
<style>
 body{{font-family:system-ui,"Noto Sans TC",sans-serif;background:#0f172a;color:#e2e8f0;margin:0;padding:28px 16px}}
 .wrap{{max-width:1180px;margin:0 auto}}
 h1{{font-size:1.5rem;margin:0 0 4px}}
 .sub{{color:#94a3b8;font-size:.9rem;margin:0 0 18px}}
 .card{{background:#1e293b;border:1px solid #334155;border-radius:14px;padding:16px 18px;margin-bottom:18px}}
 .card h2{{font-size:1.1rem;margin:0 0 4px}}
 .card .gsub{{color:#94a3b8;font-size:.84rem;margin:0 0 10px;line-height:1.6}}
 table{{width:100%;border-collapse:collapse;font-size:.86rem}}
 th,td{{text-align:left;padding:7px 8px;border-bottom:1px solid #283449;white-space:nowrap}}
 th{{color:#94a3b8;font-weight:600}}
 tr.hl td{{background:#14321f}}
 tr.hl2 td{{background:#3a1d24}}
 tr.hl3 td{{background:#3a2d14}}
 .tbl{{overflow-x:auto}}
 a{{color:#7dd3fc;text-decoration:none}} a:hover{{text-decoration:underline}}
 .qbad{{color:#f87171}} .qsub{{color:#64748b;font-size:.92em}}
 /* 凍結前兩欄（判定 / 標的），水平捲動時固定 */
 .frz th:nth-child(1),.frz td:nth-child(1){{position:sticky;left:0;z-index:2;width:104px;min-width:104px}}
 .frz th:nth-child(2),.frz td:nth-child(2){{position:sticky;left:104px;z-index:2;min-width:148px}}
 .frz th:nth-child(-n+2){{z-index:3}}
 .frz th:nth-child(1),.frz td:nth-child(1),
 .frz th:nth-child(2),.frz td:nth-child(2){{background:#1e293b}}
 .frz tr.hl td:nth-child(-n+2){{background:#14321f}}
 .frz tr.hl2 td:nth-child(-n+2){{background:#3a1d24}}
 .frz tr.hl3 td:nth-child(-n+2){{background:#3a2d14}}
 footer{{color:#64748b;font-size:.78rem;margin-top:16px;line-height:1.6}}
</style></head><body><div class="wrap">
<h1>🧲 鉅額成交方向判定 × 權證同向訊號</h1>
<p class="sub">資料日：<b>{latest}</b> · 獨立策略 · 每交易日 19:00 (TW) 自動更新 ·
偵查窗輪動：{' / '.join(_win_label(w) for w in windows)}（主窗 = {_win_label(primary)}，以下三表皆為主窗）·
鉅額方向 = 溢折價（vs 收盤價，門檻 ±{args.prem_th * 100:.1f}%）+ 同日法人買賣超比對（門檻 ±{args.inst_ratio:.1f}×）兩條獨立證據，金額加權</p>

<div class="card">
 <h2>🟢 同向買入（鉅額=買進 × 權證連買+量多）</h2>
 <p class="gsub">鉅額成交判定為<b>買進/吸籌</b>（溢價成交或同日法人大額買超），且認購權證
 「價漲金額 &gt; 價跌金額」連買 ≥ {args.streak_min} 日、當日認購額 ≥ {args.vol_mult:.1f}× 前 {args.vol_lookback} 日中位數
 ——現貨與槓桿資金<b>同向</b>操作，跟進買入。分數 = 連買 × min(量能,5) × (1+鉅額方向)。</p>
 <div class="tbl frz"><table>
  {_THEAD}
  {rows_of(buy_tbl)}
 </table></div>
</div>

<div class="card">
 <h2>🚫 避險換倉剔除（鉅額=賣出 × 權證有買盤）</h2>
 <p class="gsub">現貨被<b>折價鉅額出脫</b>（或同日法人大額賣超），權證卻有買盤——
 典型「<b>賣現貨、買權證</b>」：大戶出脫現貨後以權證留倉/避險，權證買盤非新多頭資金。
 此形態一律剔除不買（本表留存供人工覆核與日後驗證）。</p>
 <div class="tbl frz"><table>
  {_THEAD}
  {rows_of(hedge_tbl)}
 </table></div>
</div>

<div class="card">
 <h2>📊 全部鉅額標的判定</h2>
 <p class="gsub">🟢 同向買入 = 鉅額買 × 權證強 · 🟡 偏買 = 鉅額買×權證偏買，或鉅額中性(移轉類)×權證強 ·
 🚫 避險換倉 = 鉅額賣×權證買盤（剔除）· 🔴 出貨避開 = 鉅額賣×權證無買盤 ·
 ⚪ 不明 = 視窗內認購成交 &lt; {args.min_call_value / 1e6:.0f}M 或證據矛盾。
 「鉅額方向」括號內為金額加權方向分數（-1~+1）；「法人比」= 同日三大法人買賣超金額 ÷ 鉅額金額。</p>
 <div class="tbl frz"><table>
  {_THEAD}
  {rows_of(all_tbl)}
 </table></div>
</div>

{_resonance_card(evs, list(windows), primary) if evs else ""}

<div class="card">
 <h2>📋 主窗（{_win_label(primary)}）鉅額交易逐筆明細（金額前 30）</h2>
 <p class="gsub">溢折價 = 成交價/當日收盤 − 1（+溢價偏買、−折價偏賣、貼平盤中性）·
 法人比 = 同日三大法人買賣超金額 ÷ 該筆鉅額金額 · 方向 = 兩證據平均。</p>
 <div class="tbl"><table>
  <tr><th>日期</th><th>證券</th><th>交易別</th><th>成交價</th><th>溢折價</th><th>法人比</th>
  <th>方向</th><th>張數</th><th>金額(百萬)</th></tr>
  {blk_rows or '<tr><td colspan="9" style="color:#64748b">（近期無鉅額交易資料）</td></tr>'}
 </table></div>
</div>

<footer>來源：TWSE 鉅額交易日成交資訊（BFIAUU）、每日收盤行情（MI_INDEX，權證表 0999/0999P + 個股收盤價）、
三大法人買賣超（T86，統計含鉅額交易）。僅上市；上櫃不含。
鉅額交易不揭露買賣方，方向為溢折價與法人流推估，非事實揭露，僅供研究參考。</footer>
</div></body></html>"""
    for p in (HTML_FILE, INDEX_FILE):
        with open(p, "w", encoding="utf-8") as f:
            f.write(html)
    print(f"📝 報表 → {HTML_FILE}（index.html 同步）")

    def _sorted_all(e):
        et = e.copy()
        et["verdict"] = pd.Categorical(
            et["verdict"],
            categories=["same_dir_buy", "lean_buy", "hedge_suspect", "sell_avoid", "unclear"])
        return et.sort_values(["verdict", "blk_value"], ascending=[True, False])

    payload = {
        "schema": SIGNALS_SCHEMA,
        "date": latest,
        "primary_window": primary,
        "params": {"streak_min": args.streak_min, "vol_mult": args.vol_mult,
                   "vol_lookback": args.vol_lookback,
                   "windows": list(windows), "primary_window": primary,
                   "min_call_value": args.min_call_value,
                   "prem_th": args.prem_th, "inst_ratio": args.inst_ratio},
        "buy_signals": buy_tbl.to_dict("records"),
        "excluded_hedge": hedge_tbl.to_dict("records"),
        "block_directions": all_tbl.to_dict("records"),
        "windows": {w: {
            "buy_signals": e[e["verdict"] == "same_dir_buy"]
            .sort_values("score", ascending=False).to_dict("records"),
            "excluded_hedge": e[e["verdict"] == "hedge_suspect"].to_dict("records"),
            "block_directions": _sorted_all(e).to_dict("records"),   # 每窗完整判定表
        } for w, e in (evs or {}).items()},
    }
    with open(SIGNALS_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1, default=str)
    print(f"📝 訊號 → {SIGNALS_JSON}")


def main():
    _enable_utf8_console()   # Windows cp950 主控台 print emoji 不炸
    ap = argparse.ArgumentParser(description="鉅額成交方向判定 × 權證同向訊號：每日更新")
    ap.add_argument("--backfill", type=int, default=0, help="首次回補的日曆日數")
    ap.add_argument("--no-update", action="store_true", help="跳過抓取，只用既有 CSV 出報表")
    ap.add_argument("--streak-min", type=int, default=3, help="連買天數門檻(強訊號)")
    ap.add_argument("--vol-mult", type=float, default=2.0, help="量能倍數門檻(vs 中位數)")
    ap.add_argument("--vol-lookback", type=int, default=20, help="量能基期(交易日)")
    ap.add_argument("--min-call-value", type=float, default=10_000_000,
                    help="視窗內最低認購成交金額(元)，低於此判定=不明；短窗按比例放寬")
    ap.add_argument("--windows", default="5,10,15,20,W3",
                    help="偵查窗輪動：逗號分隔，數字=最近N交易日，W3=週三→下週三")
    ap.add_argument("--primary-window", default="10", help="主窗（報表主表用）")
    ap.add_argument("--prem-th", type=float, default=0.005,
                    help="溢折價方向門檻(0.005=±0.5%%)")
    ap.add_argument("--inst-ratio", type=float, default=0.5,
                    help="法人買賣超金額/鉅額金額 的方向門檻")
    ap.add_argument("--deepen-to", default=None,
                    help="往過去回補歷史到指定日(YYYY-MM-DD)；API 下限：鉅額 2005-04-04、T86 2012-05-02")
    ap.add_argument("--keep-years", type=int, default=10,
                    help="滾動留存年數(0=不刪)；年度分檔，過舊整年檔自動移除")
    ap.add_argument("--heal-days", type=int, default=45,
                    help="自動補洞回看天數(0=關)；補限流/短列造成的缺日")
    args = ap.parse_args()
    keep = args.keep_years or None

    if args.no_update:
        block, warr = load_block_history(), load_warrant_history()
        closes, inst = load_close_history(), load_inst_history()
    else:
        block = update_block_history(backfill_days=args.backfill,
                                     deepen_to=args.deepen_to, keep_years=keep)
        warr = update_warrant_history(backfill_days=args.backfill,
                                      deepen_to=args.deepen_to, keep_years=keep)
        # 方向參考資料：首次從鉅額歷史最早日回補，之後增量
        first = block["date"].min() if len(block) else None
        closes = update_close_history(first_start=first,
                                      deepen_to=args.deepen_to, keep_years=keep)
        inst = update_inst_history(first_start=first,
                                   deepen_to=args.deepen_to, keep_years=keep)
        block, warr, closes, inst = heal_recent(block, warr, closes, inst,
                                                days=args.heal_days, keep_years=keep)

    if not len(warr) or not len(block):
        print("⚠️ 尚無足夠歷史資料，先跑 --backfill 回補")
        return

    warr_dates = sorted(warr[warr["underlying"].astype(str).str.match(STOCK_CODE)]["date"].unique())
    if not warr_dates:
        print("⚠️ 權證歷史為空")
        return

    # ── 近期切片：訊號只需近 SLICE_DAYS 天，避免深歷史拖慢每日計算 ──
    latest = warr_dates[-1]
    s_from = (pd.Timestamp(latest) - pd.Timedelta(days=SLICE_DAYS)).strftime("%Y-%m-%d")
    warr_s = warr[warr["date"] >= s_from]
    block_s = block[block["date"] >= s_from]
    closes_s = closes[closes["date"] >= s_from]
    inst_s = inst[inst["date"] >= s_from]
    warr_dates = [d for d in warr_dates if d >= s_from]

    # ── 時間段輪動：每個偵查窗各判定一次 ──
    windows = [w.strip().upper() if w.strip().upper() == "W3" else w.strip()
               for w in args.windows.split(",") if w.strip()]
    evs, trades_by_win = {}, {}
    for w in windows:
        cutoff = window_cutoff(w, warr_dates)
        trades_w, blk_dirs = classify_block_direction(
            block_s, closes_s, inst_s, cutoff,
            prem_th=args.prem_th, inst_ratio_th=args.inst_ratio)
        ev_w, _ = build_evidence(warr_s, block_s, blk_dirs, cutoff,
                                 vol_lookback=args.vol_lookback)
        if ev_w.empty:
            continue
        ev_w = classify(ev_w, args)
        ev_w["window"] = str(w)
        evs[str(w)] = ev_w
        trades_by_win[str(w)] = trades_w

    if not evs:
        print("⚠️ 各偵查窗內皆無鉅額成交標的")
        return
    primary = args.primary_window if args.primary_window in evs else next(iter(evs))
    ev, trades = evs[primary], trades_by_win[primary]

    counts = ev["verdict"].value_counts()
    print(f"🧲 {latest} 主窗[{_win_label(primary)}]: 鉅額標的 {len(ev)} 檔 → "
          f"🟢同向買入 {counts.get('same_dir_buy', 0)} · 🟡偏買 {counts.get('lean_buy', 0)} · "
          f"🚫避險換倉 {counts.get('hedge_suspect', 0)} · "
          f"🔴出貨 {counts.get('sell_avoid', 0)} · ⚪不明 {counts.get('unclear', 0)}")
    for r in ev[ev["verdict"] == "same_dir_buy"].sort_values("score", ascending=False).itertuples():
        print(f"   🟢 {r.code} {r.name}  分數{r.score:.1f} 鉅額{_DIR.get(r.blk_dir)}({r.blk_wdir:+.2f}) "
              f"連買{r.streak}日 量能{r.vol_mult:.1f}× 鉅額{r.blk_n}筆/{r.blk_value / 1e6:,.0f}M")
    for r in ev[ev["verdict"] == "hedge_suspect"].itertuples():
        print(f"   🚫 {r.code} {r.name}  鉅額▽賣({r.blk_wdir:+.2f}) 但權證有買盤 → 賣現貨買權證嫌疑，剔除")

    # 多窗共振摘要
    all_ev = pd.concat(evs.values(), ignore_index=True)
    reso = (all_ev[all_ev["verdict"] == "same_dir_buy"].groupby(["code", "name"])["window"]
            .agg(list))
    for (code, name), ws in reso.items():
        if len(ws) >= 2:
            print(f"   ⏱ {code} {name} 同向買入共振 {len(ws)} 窗：{'、'.join(_win_label(w) for w in ws)}")

    n_today, n_over = append_signal_history(all_ev, latest)
    if n_over:
        print(f"📚 判定歷史：本日 {n_today} 筆（{len(evs)} 窗，同日覆蓋既有 {n_over} 筆，淨增 0）"
              f"→ {SIGNAL_HISTORY_BASE}/")
    else:
        print(f"📚 判定歷史：本日新增 {n_today} 筆（{len(evs)} 窗）→ {SIGNAL_HISTORY_BASE}/")
    write_report(ev, trades, latest, args, evs=evs, windows=windows, primary=primary)


if __name__ == "__main__":
    main()
