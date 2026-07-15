#!/usr/bin/env python3
"""
alpha.py — 訊號有沒有「純 alpha」（剝掉市場 beta 後的超額報酬）

2 年回測顯示 naive 長多贏不了 0050，因為它承載市場 beta。本檔隔離 alpha：

1. CAPM 迴歸：strat_r = α + β·mkt_r + ε → 年化 α、β、α 的 t 值（是否顯著）。
2. 市場中性（對沖）：多 🟢 籃子、空 0050（β 對沖）→ 可交易的 alpha 權益曲線。
3. 純橫斷面價差：多 🟢、空 ⚪（先前證據 ⚪ 組後續 −5%）→ 訊號的選股價值。

三者若一致為正且顯著，訊號有真 alpha，市場中性/多空是對的方向；
若對沖後 ≈0 或負，訊號只是騎 beta，長多沒意義。

用法：python research/alpha.py [--hold 10 --min-reso 3]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from block_trades import load_block_history
from warrant_flows import load_warrant_history
from market_refs import load_close_history, load_inst_history
from backtest import compute_signals, run_backtest, SArgs, _enable_utf8_console

TRADING_DAYS = 252


def curve_stats(r):
    """日報酬序列 → 年化/波動/Sharpe/MDD。"""
    r = pd.Series(r).fillna(0.0)
    eq = (1 + r).cumprod()
    n = len(r)
    cagr = eq.iloc[-1] ** (TRADING_DAYS / n) - 1 if n else np.nan
    vol = r.std() * np.sqrt(TRADING_DAYS)
    sharpe = r.mean() / r.std() * np.sqrt(TRADING_DAYS) if r.std() > 0 else np.nan
    mdd = (eq / eq.cummax() - 1).min()
    return dict(total=eq.iloc[-1] - 1, cagr=cagr, vol=vol, sharpe=sharpe, mdd=mdd)


def capm(strat_r, mkt_r):
    """OLS：strat = α + β·mkt。回傳年化 α、β、α 的 t 值、R²。"""
    x = np.asarray(mkt_r, float)
    y = np.asarray(strat_r, float)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = beta_hat
    resid = y - X @ beta_hat
    dof = n - 2
    s2 = (resid @ resid) / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    se_a = np.sqrt(cov[0, 0])
    t_a = a / se_a if se_a > 0 else np.nan
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid @ resid) / ss_tot if ss_tot > 0 else np.nan
    return dict(alpha_ann=a * TRADING_DAYS, beta=b, t_alpha=t_a, r2=r2, n=n)


def main():
    _enable_utf8_console()
    ap = argparse.ArgumentParser(description="訊號 alpha 分析（市場中性/多空）")
    ap.add_argument("--hold", type=int, default=10)
    ap.add_argument("--min-reso", type=int, default=3)
    ap.add_argument("--cost-bps", type=float, default=58.5)
    ap.add_argument("--windows", default="5,10,15,20,W3")
    ap.add_argument("--primary", default="10")
    args = ap.parse_args()

    print("載入 + 回放訊號（一次）...")
    block, warr = load_block_history(), load_warrant_history()
    closes, inst = load_close_history(), load_inst_history()
    windows = [w.strip().upper() if w.strip().upper() == "W3" else w.strip()
               for w in args.windows.split(",") if w.strip()]
    signals = compute_signals(block, warr, closes, inst, windows, args.primary, SArgs())

    # 多腿：🟢·共振≥K
    eqL, _, mL = run_backtest(signals, closes, hold=args.hold,
                              verdicts=("same_dir_buy",), min_reso=args.min_reso,
                              weight="equal", cost_bps=args.cost_bps)
    # 空腿：⚪ unclear 籃子（先前證據事後 −5%）
    eqS, _, mS = run_backtest(signals, closes, hold=args.hold,
                              verdicts=("unclear",), min_reso=0,
                              weight="equal", cost_bps=args.cost_bps)

    L = eqL["strat_r"].values
    S = eqS["strat_r"].values
    M = eqL["mkt_r"].values
    inv = (eqL["n_pos"].values > 0).astype(float)   # 多腿有部位的日子

    period = f"{eqL['date'].iloc[0]} → {eqL['date'].iloc[-1]}（{len(eqL)} 交易日）"
    print(f"\n═══ Alpha 分析：🟢·共振≥{args.min_reso}·持有{args.hold}日 · {period} ═══\n")

    # 1) CAPM 迴歸
    c = capm(L, M)
    print("① CAPM 迴歸（strat = α + β·0050）")
    print(f"   年化 α = {c['alpha_ann']*100:+.1f}%   β = {c['beta']:.2f}   "
          f"α 的 t 值 = {c['t_alpha']:.2f}（|t|>2 才顯著）  R² = {c['r2']:.2f}")
    verdict_a = "顯著正 alpha ✅" if c["t_alpha"] > 2 else (
        "顯著負 ❌" if c["t_alpha"] < -2 else "不顯著（≈騎 beta）⚠️")
    print(f"   → {verdict_a}\n")

    # 2) 市場中性：多籃子、空 0050（β 對沖，僅在有部位日對沖）
    hedged = L - c["beta"] * M * inv
    hm = curve_stats(hedged)
    print("② 市場中性（多🟢籃子、β 對沖空 0050）")
    print(f"   年化 {hm['cagr']*100:+.1f}%  Sharpe {hm['sharpe']:.2f}  MDD {hm['mdd']*100:.1f}%  "
          f"總報酬 {hm['total']*100:+.1f}%\n")

    # 3) 純橫斷面價差：多 🟢、空 ⚪（dollar-neutral，僅在多腿有部位日）
    spread = (L - S) * inv
    sm = curve_stats(spread)
    print("③ 多🟢空⚪ 純橫斷面價差（dollar-neutral）")
    print(f"   年化 {sm['cagr']*100:+.1f}%  Sharpe {sm['sharpe']:.2f}  MDD {sm['mdd']*100:.1f}%  "
          f"總報酬 {sm['total']*100:+.1f}%")
    print(f"   （⚪ 空腿本身：年化 {mS['strat']['cagr']*100:+.1f}%  Sharpe {mS['strat']['sharpe']:.2f}）\n")

    # 對照：長多與 0050
    print("對照：")
    print(f"   長多🟢    年化 {mL['strat']['cagr']*100:+.1f}%  Sharpe {mL['strat']['sharpe']:.2f}  MDD {mL['strat']['mdd']*100:.1f}%")
    print(f"   0050     年化 {mL['mkt']['cagr']*100:+.1f}%  Sharpe {mL['mkt']['sharpe']:.2f}  MDD {mL['mkt']['mdd']*100:.1f}%")
    print("\n判讀：① t>2 且 ②③ Sharpe 明顯為正 → 訊號有真 alpha，市場中性/多空可行；"
          "若 ① 不顯著、對沖後 Sharpe≈0 → 只是騎 beta。樣本仍含單一 crash（2024/8），深挖 2022 後再確認。")


if __name__ == "__main__":
    main()
