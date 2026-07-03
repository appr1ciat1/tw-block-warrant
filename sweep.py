#!/usr/bin/env python3
"""
sweep.py — 一次訊號回放，掃多組回測設定，並列比較風報比

訊號回放是最慢的一步（每日×多窗）。本檔只回放一次，快取後對多組
進場/持有/加權/上限設定各跑一次 run_backtest，並列出 Sharpe/MDD/Calmar，
用來挑「風報比堆疊」的最佳組態，避免逐一重跑 backtest.py。

用法：
    python sweep.py
    python sweep.py --hold 5,10,20        # 額外掃持有期
"""
import argparse
import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from block_trades import load_block_history
from warrant_flows import load_warrant_history
from market_refs import load_close_history, load_inst_history
from backtest import compute_signals, run_backtest, SArgs


def main():
    ap = argparse.ArgumentParser(description="回測設定掃描")
    ap.add_argument("--windows", default="5,10,15,20,W3")
    ap.add_argument("--primary", default="10")
    ap.add_argument("--cost-bps", type=float, default=58.5)
    args = ap.parse_args()

    print("載入 + 回放訊號（一次）...")
    block, warr = load_block_history(), load_warrant_history()
    closes, inst = load_close_history(), load_inst_history()
    windows = [w.strip().upper() if w.strip().upper() == "W3" else w.strip()
               for w in args.windows.split(",") if w.strip()]
    signals = compute_signals(block, warr, closes, inst, windows, args.primary, SArgs())

    # 風報比堆疊：逐步加條件，看每步對 Sharpe/MDD/Calmar 的影響
    configs = [
        ("基準 🟢·持有10·等權", dict(verdicts=("same_dir_buy",), hold=10, weight="equal")),
        ("🟢·持有5", dict(verdicts=("same_dir_buy",), hold=5, weight="equal")),
        ("🟢·持有20", dict(verdicts=("same_dir_buy",), hold=20, weight="equal")),
        ("🟢·共振≥3", dict(verdicts=("same_dir_buy",), hold=10, min_reso=3, weight="equal")),
        ("🟢·共振≥3·conviction", dict(verdicts=("same_dir_buy",), hold=10, min_reso=3, weight="conviction")),
        ("🟢·共振≥3·持有20", dict(verdicts=("same_dir_buy",), hold=20, min_reso=3, weight="equal")),
        ("🟢·共振≥3·上限8檔", dict(verdicts=("same_dir_buy",), hold=10, min_reso=3, max_pos=8, weight="conviction")),
        ("🟢+🟡·持有10", dict(verdicts=("same_dir_buy", "lean_buy"), hold=10, weight="equal")),
    ]

    rows = []
    mkt = None
    for name, cfg in configs:
        eq, tdf, m = run_backtest(signals, closes, cost_bps=args.cost_bps, **cfg)
        s = m["strat"]
        mkt = m["mkt"]
        rows.append({
            "設定": name, "總報酬": f"{s['total']*100:+.0f}%", "年化": f"{s['cagr']*100:+.0f}%",
            "波動": f"{s['vol']*100:.0f}%", "Sharpe": f"{s['sharpe']:.2f}",
            "MDD": f"{s['mdd']*100:.0f}%", "Calmar": f"{s['calmar']:.2f}",
            "勝率": f"{m.get('hit',float('nan'))*100:.0f}%", "交易": m.get("n_trades", "—"),
            "持股": f"{m['avg_pos']:.1f}",
        })
    print("\n═══ 風報比堆疊比較（期間同上）═══")
    print(pd.DataFrame(rows).to_string(index=False))
    if mkt:
        print(f"\n0050 基準：總報酬 {mkt['total']*100:+.0f}% · 年化 {mkt['cagr']*100:+.0f}% · "
              f"Sharpe {mkt['sharpe']:.2f} · MDD {mkt['mdd']*100:.0f}% · Calmar {mkt['calmar']:.2f}")
    print("\n判讀：風報比看 Sharpe 與 Calmar（越高越好）、MDD（越淺越好）。"
          "樣本僅一年單 regime，深挖後需重驗。")


if __name__ == "__main__":
    main()
