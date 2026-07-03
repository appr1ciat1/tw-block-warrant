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

輸出：block_warrant/report.html + block_warrant/signals.json
     + data/signal_history.csv（每日全部判定累積，供日後驗證各級後續報酬）

用法（任何工作目錄皆可）：
    python block_warrant/update.py                  # 增量更新 + 出報表
    python block_warrant/update.py --backfill 90    # 首次回補 90 個日曆日
    python block_warrant/update.py --no-update      # 只用既有 CSV 出報表
"""

import argparse
import json
import os
import re
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)   # 讓 update.py 從任何 CWD 都能 import 同資料夾模組

from block_trades import update_block_history, load_block_history        # noqa: E402
from warrant_flows import update_warrant_history, load_warrant_history   # noqa: E402
from market_refs import (update_close_history, load_close_history,       # noqa: E402
                         update_inst_history, load_inst_history)

HTML_FILE = os.path.join(_HERE, "report.html")
INDEX_FILE = os.path.join(_HERE, "index.html")   # GitHub Pages 首頁 = 報表
SIGNALS_JSON = os.path.join(_HERE, "signals.json")
SIGNAL_HISTORY_CSV = os.path.join(_HERE, "data", "signal_history.csv")
STOCK_CODE = re.compile(r"^\d{4,6}$")   # 個股/ETF；排除 IX0001 等指數標的


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
    b["inst_value"] = b["total_net"] * b["close"]          # 法人買賣超金額(近似)
    b["inst_ratio"] = b["inst_value"] / b["value"].where(b["value"] > 0)

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


# ── 權證方向證據 + 鉅額標的判定 ─────────────────────────────────
def build_evidence(warr, block, blk_dirs, vol_lookback=20, block_window=10):
    """
    以「近 block_window 日有鉅額成交的標的」為母體，逐檔計算權證方向證據，
    並併入鉅額方向（blk_dirs，由 classify_block_direction 產生）。

    Returns
    -------
    ev : pd.DataFrame  每個鉅額標的一列：
         blk_n / blk_value / blk_last / blk_dir / blk_wdir / blk_prem / blk_inst_ratio
         streak / vol_mult / buy_days / sell_days / call_net_win /
         call_val_win / put_call_win / buy_ratio（權證證據）
    latest : str  最新權證資料日
    """
    warr = warr[warr["underlying"].astype(str).str.match(STOCK_CODE)].copy()
    if warr.empty or not len(block):
        return pd.DataFrame(), None
    dates = sorted(warr["date"].unique())
    latest = dates[-1]
    cutoff = (pd.Timestamp(latest) - pd.Timedelta(days=block_window)).strftime("%Y-%m-%d")
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

    rows = []
    for code, b in blk.iterrows():
        g = warr_g.get(code)
        if g is None:
            rows.append({"code": code, "name": b["name"], "blk_n": int(b.blk_n),
                         "blk_value": float(b.blk_value), "blk_last": b.blk_last,
                         "streak": 0, "vol_mult": float("nan"), "buy_days": 0,
                         "sell_days": 0, "call_net_win": 0.0, "call_val_win": 0.0,
                         "put_call_win": float("nan"), "buy_ratio": 0.0})
            continue

        g = g.set_index("date")[num_cols].reindex(dates).fillna(0.0)
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
    return ev, latest


def classify(ev, args):
    """
    鉅額方向 × 權證證據 → verdict：
        same_dir_buy(🟢 同向買入) / lean_buy(🟡 偏買) /
        hedge_suspect(🚫 賣現貨買權證嫌疑→剔除) /
        sell_avoid(🔴 出貨避開) / unclear(⚪ 不明)
    """
    ev = ev.copy()
    thin = ev["call_val_win"] < args.min_call_value
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

    # 分數 = 權證強度 ×（1 + 鉅額方向加權），僅供排序
    ev["score"] = (ev["streak"] * ev["vol_mult"].fillna(0).clip(upper=5.0)
                   * (1.0 + ev["blk_wdir"].fillna(0)))
    return ev


def append_signal_history(ev, latest, path=SIGNAL_HISTORY_CSV):
    """把當日全部鉅額標的判定 append 到歷史（同日重跑覆蓋），供日後統計各級後續報酬。"""
    cols = ["date", "code", "name", "verdict", "score",
            "blk_dir", "blk_wdir", "blk_prem", "blk_inst_ratio",
            "streak", "vol_mult", "buy_days", "sell_days", "call_net_win",
            "call_val_win", "put_call_win", "buy_ratio", "blk_n", "blk_value", "blk_last"]
    today = ev.copy()
    today.insert(0, "date", latest)
    today = today[cols]

    if os.path.exists(path):
        hist = pd.read_csv(path, dtype={"code": str})
        hist = hist[hist["date"] != latest]
        merged = pd.concat([hist, today], ignore_index=True)
    else:
        merged = today
    os.makedirs(os.path.dirname(path), exist_ok=True)
    merged.to_csv(path, index=False)
    return len(today)


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


def _row_html(r):
    vm = f"{r.vol_mult:.1f}×" if pd.notna(r.vol_mult) else "—"
    pc = f"{r.put_call_win:.2f}" if pd.notna(r.put_call_win) else "—"
    prem = f"{r.blk_prem * 100:+.2f}%" if pd.notna(r.blk_prem) else "—"
    ir = f"{r.blk_inst_ratio:+.2f}" if pd.notna(r.blk_inst_ratio) else "—"
    return (f"<tr{_CLS[r.verdict]}><td>{_TAG[r.verdict]}</td>"
            f"<td><b>{r.code}</b> {r.name}</td>"
            f"<td>{_DIR.get(r.blk_dir, '–')} ({r.blk_wdir:+.2f})</td>"
            f"<td>{prem}</td><td>{ir}</td>"
            f"<td>{r.blk_n} 筆 / {_fmt_m(r.blk_value)}M</td><td>{r.blk_last[5:]}</td>"
            f"<td>{r.streak}</td><td>{vm}</td><td>{r.buy_days}/{r.sell_days}</td>"
            f"<td>{_fmt_m(r.call_net_win)}</td><td>{pc}</td></tr>")


_THEAD = ("<tr><th>判定</th><th>標的</th><th>鉅額方向</th><th>溢折價</th><th>法人比</th>"
          "<th>鉅額(視窗)</th><th>最近鉅額</th>"
          "<th>連買(日)</th><th>量能</th><th>買/賣日</th><th>認購淨買壓(百萬)</th><th>Put/Call</th></tr>")
_NCOL = 12


def write_report(ev, trades, latest, args):
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
            ir = f"{r.inst_ratio:+.2f}" if pd.notna(r.inst_ratio) else "—"
            rws.append(
                f"<tr><td>{r.date}</td><td><b>{r.code}</b> {r.name}</td><td>{r.trade_type}</td>"
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
 footer{{color:#64748b;font-size:.78rem;margin-top:16px;line-height:1.6}}
</style></head><body><div class="wrap">
<h1>🧲 鉅額成交方向判定 × 權證同向訊號</h1>
<p class="sub">資料日：<b>{latest}</b> · 獨立策略 · 每日收盤後自動更新 ·
母體 = 近 {args.block_window} 日有鉅額成交的標的 ·
鉅額方向 = 溢折價（vs 收盤價，門檻 ±{args.prem_th * 100:.1f}%）+ 同日法人買賣超比對（門檻 ±{args.inst_ratio:.1f}×）兩條獨立證據，金額加權</p>

<div class="card">
 <h2>🟢 同向買入（鉅額=買進 × 權證連買+量多）</h2>
 <p class="gsub">鉅額成交判定為<b>買進/吸籌</b>（溢價成交或同日法人大額買超），且認購權證
 「價漲金額 &gt; 價跌金額」連買 ≥ {args.streak_min} 日、當日認購額 ≥ {args.vol_mult:.1f}× 前 {args.vol_lookback} 日中位數
 ——現貨與槓桿資金<b>同向</b>操作，跟進買入。分數 = 連買 × min(量能,5) × (1+鉅額方向)。</p>
 <div class="tbl"><table>
  {_THEAD}
  {rows_of(buy_tbl)}
 </table></div>
</div>

<div class="card">
 <h2>🚫 避險換倉剔除（鉅額=賣出 × 權證有買盤）</h2>
 <p class="gsub">現貨被<b>折價鉅額出脫</b>（或同日法人大額賣超），權證卻有買盤——
 典型「<b>賣現貨、買權證</b>」：大戶出脫現貨後以權證留倉/避險，權證買盤非新多頭資金。
 此形態一律剔除不買（本表留存供人工覆核與日後驗證）。</p>
 <div class="tbl"><table>
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
 <div class="tbl"><table>
  {_THEAD}
  {rows_of(all_tbl)}
 </table></div>
</div>

<div class="card">
 <h2>📋 近 {args.block_window} 日鉅額交易逐筆明細（金額前 30）</h2>
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

    payload = {
        "date": latest,
        "params": {"streak_min": args.streak_min, "vol_mult": args.vol_mult,
                   "vol_lookback": args.vol_lookback, "block_window": args.block_window,
                   "min_call_value": args.min_call_value,
                   "prem_th": args.prem_th, "inst_ratio": args.inst_ratio},
        "buy_signals": buy_tbl.to_dict("records"),
        "excluded_hedge": hedge_tbl.to_dict("records"),
        "block_directions": all_tbl.to_dict("records"),
    }
    with open(SIGNALS_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1, default=str)
    print(f"📝 訊號 → {SIGNALS_JSON}")


def main():
    ap = argparse.ArgumentParser(description="鉅額成交方向判定 × 權證同向訊號：每日更新")
    ap.add_argument("--backfill", type=int, default=0, help="首次回補的日曆日數")
    ap.add_argument("--no-update", action="store_true", help="跳過抓取，只用既有 CSV 出報表")
    ap.add_argument("--streak-min", type=int, default=3, help="連買天數門檻(強訊號)")
    ap.add_argument("--vol-mult", type=float, default=2.0, help="量能倍數門檻(vs 中位數)")
    ap.add_argument("--vol-lookback", type=int, default=20, help="量能基期(交易日)")
    ap.add_argument("--min-call-value", type=float, default=10_000_000,
                    help="視窗內最低認購成交金額(元)，低於此判定=不明")
    ap.add_argument("--block-window", type=int, default=10, help="鉅額成交回看日曆日數(母體視窗)")
    ap.add_argument("--prem-th", type=float, default=0.005,
                    help="溢折價方向門檻(0.005=±0.5%%)")
    ap.add_argument("--inst-ratio", type=float, default=0.5,
                    help="法人買賣超金額/鉅額金額 的方向門檻")
    args = ap.parse_args()

    if args.no_update:
        block, warr = load_block_history(), load_warrant_history()
        closes, inst = load_close_history(), load_inst_history()
    else:
        block = update_block_history(backfill_days=args.backfill)
        warr = update_warrant_history(backfill_days=args.backfill)
        # 方向參考資料：首次從鉅額歷史最早日回補，之後增量
        first = block["date"].min() if len(block) else None
        closes = update_close_history(first_start=first)
        inst = update_inst_history(first_start=first)

    if not len(warr) or not len(block):
        print("⚠️ 尚無足夠歷史資料，先跑 --backfill 回補")
        return

    # 先以最新權證資料日決定視窗，再判定視窗內鉅額方向
    warr_dates = sorted(warr[warr["underlying"].astype(str).str.match(STOCK_CODE)]["date"].unique())
    if not warr_dates:
        print("⚠️ 權證歷史為空")
        return
    latest = warr_dates[-1]
    cutoff = (pd.Timestamp(latest) - pd.Timedelta(days=args.block_window)).strftime("%Y-%m-%d")

    trades, blk_dirs = classify_block_direction(
        block, closes, inst, cutoff,
        prem_th=args.prem_th, inst_ratio_th=args.inst_ratio)

    ev, latest = build_evidence(warr, block, blk_dirs,
                                vol_lookback=args.vol_lookback,
                                block_window=args.block_window)
    if ev.empty:
        print(f"⚠️ 近 {args.block_window} 日無鉅額成交標的")
        return
    ev = classify(ev, args)

    counts = ev["verdict"].value_counts()
    print(f"🧲 {latest}: 鉅額標的 {len(ev)} 檔 → "
          f"🟢同向買入 {counts.get('same_dir_buy', 0)} · 🟡偏買 {counts.get('lean_buy', 0)} · "
          f"🚫避險換倉 {counts.get('hedge_suspect', 0)} · "
          f"🔴出貨 {counts.get('sell_avoid', 0)} · ⚪不明 {counts.get('unclear', 0)}")
    for r in ev[ev["verdict"] == "same_dir_buy"].sort_values("score", ascending=False).itertuples():
        print(f"   🟢 {r.code} {r.name}  分數{r.score:.1f} 鉅額{_DIR.get(r.blk_dir)}({r.blk_wdir:+.2f}) "
              f"連買{r.streak}日 量能{r.vol_mult:.1f}× 鉅額{r.blk_n}筆/{r.blk_value / 1e6:,.0f}M")
    for r in ev[ev["verdict"] == "hedge_suspect"].itertuples():
        print(f"   🚫 {r.code} {r.name}  鉅額▽賣({r.blk_wdir:+.2f}) 但權證有買盤 → 賣現貨買權證嫌疑，剔除")

    n_hist = append_signal_history(ev, latest)
    print(f"📚 判定歷史 +{n_hist} 筆 → {SIGNAL_HISTORY_CSV}")
    write_report(ev, trades, latest, args)


if __name__ == "__main__":
    main()
