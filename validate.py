#!/usr/bin/env python3
"""
block_warrant/validate.py — 鉅額方向判定的事後驗證

回答「哪種資料判定鉅額買賣方向最正確」：把累積歷史中的每筆鉅額
用（溢折價 / 同日法人買賣超）分類方向，再統計各組後續 5/10/20 個
交易日「相對 0050 的超額報酬」。若判定有效，應看到：

    判定=買 的組別事後超額報酬 > 中性 > 判定=賣

同時分開統計「只用溢折價」「只用法人」「兩證據同向」的組別，
比較單一證據與雙證據的準確度。資料越累積越可信（樣本數 n 會顯示）。

用法：
    python block_warrant/validate.py
    python block_warrant/validate.py --prem-th 0.01 --inst-ratio 0.3   # 靈敏度測試
"""

import argparse
import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from block_trades import load_block_history                       # noqa: E402
from market_refs import load_close_history, load_inst_history     # noqa: E402
from update import classify_block_direction, _enable_utf8_console  # noqa: E402

MARKET = "0050"          # 市場基準（超額報酬用）
HORIZONS = (5, 10, 20)   # 交易日


def forward_returns(trades, closes):
    """為每筆鉅額算後續 k 交易日報酬與相對 0050 超額報酬。"""
    px = closes.pivot_table(index="date", columns="code", values="close")
    px = px.sort_index()
    dates = list(px.index)
    pos = {d: i for i, d in enumerate(dates)}
    mkt = px[MARKET] if MARKET in px.columns else None

    out = trades.copy()
    for k in HORIZONS:
        rets, excs = [], []
        for r in trades.itertuples():
            i = pos.get(r.date)
            ret = exc = float("nan")
            if i is not None and i + k < len(dates) and r.code in px.columns:
                p0, p1 = px[r.code].iloc[i], px[r.code].iloc[i + k]
                if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                    ret = p1 / p0 - 1
                    if mkt is not None:
                        m0, m1 = mkt.iloc[i], mkt.iloc[i + k]
                        if pd.notna(m0) and pd.notna(m1) and m0 > 0:
                            exc = ret - (m1 / m0 - 1)
            rets.append(ret)
            excs.append(exc)
        out[f"ret_{k}d"] = rets
        out[f"exc_{k}d"] = excs
    return out


def summarize(df, by, title):
    """各組 n / 平均超額 / 勝率（超額>0 比例）。"""
    print(f"\n── {title} " + "─" * max(1, 46 - len(title)))
    rows = []
    for key, g in df.groupby(by, observed=True):
        row = {by: key, "n": len(g)}
        for k in HORIZONS:
            e = g[f"exc_{k}d"].dropna()
            row[f"exc{k}d"] = f"{e.mean() * 100:+.2f}%" if len(e) else "—"
            row[f"med{k}d"] = f"{e.median() * 100:+.2f}%" if len(e) else "—"
            row[f"win{k}d"] = f"{(e > 0).mean() * 100:.0f}%" if len(e) else "—"
        rows.append(row)
    if not rows:
        print("（無樣本）")
        return
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    _enable_utf8_console()
    ap = argparse.ArgumentParser(description="鉅額方向判定的事後驗證")
    ap.add_argument("--prem-th", type=float, default=0.005)
    ap.add_argument("--inst-ratio", type=float, default=0.5)
    args = ap.parse_args()

    block = load_block_history()
    closes = load_close_history()
    inst = load_inst_history()
    if not len(block) or not len(closes):
        print("⚠️ 資料不足：先跑 python block_warrant/update.py 累積歷史")
        return

    # 全歷史逐筆分類（cutoff 設最早日 → 不截斷）
    trades, _ = classify_block_direction(
        block, closes, inst, cutoff=str(block["date"].min()),
        prem_th=args.prem_th, inst_ratio_th=args.inst_ratio)
    trades = trades[trades["close"].notna()].copy()   # 需有收盤價才能算報酬
    trades = forward_returns(trades, closes)

    n_dates = trades["date"].nunique()
    print(f"🧪 鉅額方向判定驗證：{trades['date'].min()} → {trades['date'].max()}"
          f"（{n_dates} 個交易日，{len(trades)} 筆鉅額）")
    print(f"   門檻：溢折價 ±{args.prem_th * 100:.1f}% · 法人比 ±{args.inst_ratio:.1f}×"
          f" · 超額報酬基準 = {MARKET}")

    # 1) 綜合方向分數（兩證據平均）
    lab = pd.cut(trades["dir_score"], [-1.01, -0.75, -0.25, 0.25, 0.75, 1.01],
                 labels=["雙證據賣", "偏賣", "中性", "偏買", "雙證據買"])
    trades["dir_class"] = lab
    summarize(trades, "dir_class", "綜合判定（溢折價+法人 平均）")

    # 2) 單一證據：只看溢折價
    trades["price_class"] = pd.cut(trades["price_ev"], [-1.5, -0.5, 0.5, 1.5],
                                   labels=["折價(賣)", "貼平盤(中性)", "溢價(買)"])
    summarize(trades, "price_class", "只用溢折價")

    # 3) 單一證據：只看同日法人
    trades["inst_class"] = pd.cut(trades["inst_ev"], [-1.5, -0.5, 0.5, 1.5],
                                  labels=["法人賣超(賣)", "無法人證據", "法人買超(買)"])
    summarize(trades, "inst_class", "只用同日法人買賣超")

    print("\n判讀：exc = 平均超額報酬、win = 超額>0 比例。"
          "買組 > 中性 > 賣組（且雙證據組差距最大）→ 方向判定有效；"
          "樣本少時（n < 30）僅供參考，資料每日累積後再看。")


if __name__ == "__main__":
    main()
