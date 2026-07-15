#!/usr/bin/env python3
"""
backtest.py — 鉅額×權證訊號的事件驅動回測引擎（地基）

把每日訊號變成真實組合的權益曲線與風報比指標，是所有優化的量化基礎。

模型（標準重疊組合法 overlapping portfolios）：
- 每個交易日 D 收盤後可得當日訊號（鉅額/權證/法人資料皆盤後發布）。
- 進場：訊號標的於 D+1 收盤買入（無前視：訊號用 D 資料、成交在 D+1）。
- 持有：holding_days 個交易日後於收盤賣出。
- 同一檔在持有期內重複觸發 → 維持單一部位並刷新出場時鐘（union）。
- 加權：equal（等權）或 conviction（依共振數/分數）。
- 成本：台股來回約 0.585%（手續費 0.1425%×2 + 賣出證交稅 0.3%），依週轉套用。
- 基準：0050 買進持有。

用法：
    python backtest.py                         # 預設：verdict=same_dir_buy，持有10日
    python backtest.py --min-reso 3            # 進場門檻改共振≥3窗
    python backtest.py --hold 20 --weight conviction
    python backtest.py --verdicts same_dir_buy,lean_buy --cost-bps 40
"""
import argparse
import bisect
import hashlib
import os
import pickle
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))   # research/
_ROOT = os.path.dirname(_HERE)                        # repo 根（原始每日訊號版）
sys.path.insert(0, _ROOT)   # 從根目錄 import 原始資料/訊號模組（唯讀，不改動原始版）

from block_trades import load_block_history
from warrant_flows import load_warrant_history
from market_refs import load_close_history, load_inst_history
from update import (classify_block_direction, build_evidence, classify,
                    window_cutoff, STOCK_CODE, _enable_utf8_console)

REPORT = os.path.join(_HERE, "backtest.html")   # 延伸版輸出留在 research/
CACHE_DIR = os.path.join(_HERE, "artifacts")    # 訊號回放快取（research/artifacts/）
MKT = "0050"
TRADING_DAYS = 252


class SArgs:
    """訊號計算參數（與 update.py 預設一致）。"""
    streak_min = 3
    vol_mult = 2.0
    vol_lookback = 20
    min_call_value = 10_000_000
    prem_th = 0.005
    inst_ratio = 0.5


def _signals_key(block, warr, closes, inst, windows, primary, sargs, warmup):
    """訊號快取指紋：資料範圍(len+min+max)+**內容雜湊**+窗+參數。
    內容雜湊讓「同筆數同日期範圍但值被在地更正(補洞/重抓覆寫)」也能失效重算。"""
    parts = []
    for df in (block, warr, closes, inst):
        lo = df["date"].min() if len(df) else "-"
        hi = df["date"].max() if len(df) else "-"
        h = int(pd.util.hash_pandas_object(df, index=False).sum()) if len(df) else 0
        parts.append(f"{len(df)}:{lo}:{hi}:{h}")
    parts += [",".join(map(str, windows)), str(primary), str(warmup),
              f"{sargs.streak_min},{sargs.vol_mult},{sargs.vol_lookback},"
              f"{sargs.min_call_value},{sargs.prem_th},{sargs.inst_ratio}"]
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:16]


def compute_signals(block, warr, closes, inst, windows, primary, sargs, warmup=25, cache=True):
    """
    日級回放，回傳每個 as-of 日的訊號：
        {date: {code: {"verdict":..., "reso":共振窗數, "score":...}}}
    reso = 該檔被幾個窗判為 same_dir_buy。cache=True 時以資料指紋快取到 research/artifacts/。
    """
    if cache:
        cpath = os.path.join(
            CACHE_DIR,
            f"signals_{_signals_key(block, warr, closes, inst, windows, primary, sargs, warmup)}.pkl")
        if os.path.exists(cpath):
            print(f"  ✓ 訊號快取命中 {os.path.basename(cpath)}")
            with open(cpath, "rb") as f:
                return pickle.load(f)

    wd = sorted(warr[warr["underlying"].astype(str).str.match(STOCK_CODE)]["date"].unique())
    asof_list = wd[warmup:]
    out = {}
    for k, asof in enumerate(asof_list):
        b_s = block[block["date"] <= asof]
        w_s = warr[warr["date"] <= asof]
        c_s = closes[closes["date"] <= asof]
        i_s = inst[inst["date"] <= asof]
        wd_s = wd[:bisect.bisect_right(wd, asof)]   # wd 是純字串 list，bisect 安全

        per_win, reso = {}, {}
        for win in windows:
            cut = window_cutoff(win, wd_s)
            _, bd = classify_block_direction(b_s, c_s, i_s, cut,
                                             prem_th=sargs.prem_th, inst_ratio_th=sargs.inst_ratio)
            ev, _ = build_evidence(w_s, b_s, bd, cut, vol_lookback=sargs.vol_lookback)
            if ev.empty:
                continue
            ev = classify(ev, sargs)
            per_win[win] = ev
            for r in ev[ev["verdict"] == "same_dir_buy"].itertuples():
                reso[r.code] = reso.get(r.code, 0) + 1
        if primary not in per_win:
            continue
        day = {}
        for r in per_win[primary].itertuples():
            day[r.code] = {"verdict": r.verdict, "reso": reso.get(r.code, 0),
                           "score": float(r.score)}
        out[asof] = day
        if (k + 1) % 50 == 0:
            print(f"  訊號回放 {k + 1}/{len(asof_list)}", flush=True)
    if cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(cpath, "wb") as f:
            pickle.dump(out, f)
    return out


def select(day_sigs, verdicts, min_reso):
    """從當日訊號挑出符合進場條件的 {code: conviction}。"""
    picks = {}
    for code, s in day_sigs.items():
        if s["verdict"] in verdicts and s["reso"] >= min_reso:
            picks[code] = max(s["reso"], 1)   # conviction 權重用共振數
    return picks


def run_backtest(signals, closes, hold=10, verdicts=("same_dir_buy",), min_reso=0,
                 weight="equal", cost_bps=58.5, max_pos=0):
    """
    重疊組合法回測。回傳 (equity_df, trades_df, metrics)。
    equity_df: date, strat_equity, mkt_equity, n_pos
    """
    px = closes.pivot_table(index="date", columns="code", values="close").sort_index()
    dates = list(px.index)
    # fill_method=None：不 forward-fill，停牌缺口 → NaN（否則會把跨缺口的多日
    # 報酬灌進一天，產生虛假巨幅單日報酬）。
    ret = px.pct_change(fill_method=None)
    # 除權息/分割/停牌復牌：台股單日漲跌幅上限 ±10%，超過者必為還原前的
    # 除權息跌價或資料異常。無還原股價下，中性化為 0（假設配息≈跌價、對持有
    # 者淨中性），優於夾到 ±10.5%（那會把除息大跌當成 -10.5% 虛假虧損）。
    LIMIT = 0.105
    n_neut = int((ret.abs() > LIMIT).sum().sum())
    if n_neut:
        print(f"   ⚠️ {n_neut} 個單日報酬 >±10.5%（除權息/資料異常）→ 中性化為 0"
              f"（占 {n_neut / ret.notna().sum().sum() * 100:.2f}%）")
    ret = ret.where(ret.abs() <= LIMIT, 0.0)

    # 每個進場日 → {code: conviction}
    entries = {d: select(s, set(verdicts), min_reso) for d, s in signals.items()}

    # 逐日建構持有組合。active: code -> (first_earn_idx, last_earn_idx, conviction)。
    # 訊號用第 t-1 日資料 → 第 t 日收盤買入 → 次日 t+1 起計酬、持有到 t+hold 收盤賣出。
    # 進場當日 t 不吃 ret.iloc[t]（那是「買入前」close(t-1)→close(t) 的報酬）——消除
    # 一日前視，使權益曲線與逐筆 close(t)→close(t+hold) 口徑一致。
    date_idx = {d: k for k, d in enumerate(dates)}
    active = {}
    strat_r, mkt_r, npos_series, prev_w = [], [], [], {}
    trades = []
    cost = cost_bps / 1e4
    ret_np = {c: ret[c].to_numpy() for c in px.columns}   # 逐格 iloc 太慢，先轉 numpy
    mkt_np = ret[MKT].to_numpy() if MKT in px.columns else np.zeros(len(dates))

    for t, d in enumerate(dates):
        active = {c: v for c, v in active.items() if v[1] >= t}   # last_earn < t → 出場
        if t > 0:
            for code, cv in entries.get(dates[t - 1], {}).items():
                if code in px.columns:
                    # 重複觸發：保留原 first_earn（避免計酬缺口），只延長出場到 t+hold
                    fe = active[code][0] if code in active else t + 1
                    active[code] = (fe, t + hold, cv)
                    trades.append({"entry": d, "code": code, "conviction": cv,
                                   "exit_idx": t + hold})
        # 當日計酬集合：first_earn <= t <= last_earn（進場當日尚未計酬）
        earn = [c for c, v in active.items() if v[0] <= t <= v[1]]
        if max_pos and len(earn) > max_pos:
            earn = sorted(earn, key=lambda c: -active[c][2])[:max_pos]
        if earn:
            if weight == "conviction":
                cv = np.array([active[c][2] for c in earn], float)
                w = cv / cv.sum()
            else:
                w = np.full(len(earn), 1.0 / len(earn))
            wmap = dict(zip(earn, w))
        else:
            wmap = {}

        day_r = 0.0
        for c, wt in wmap.items():
            rc = ret_np[c][t]
            if not np.isnan(rc):
                day_r += wt * rc
        turn = sum(abs(wmap.get(c, 0) - prev_w.get(c, 0)) for c in set(wmap) | set(prev_w))
        day_r -= turn * cost
        prev_w = wmap

        strat_r.append(day_r)
        m = mkt_np[t]
        mkt_r.append(m if not np.isnan(m) else 0.0)
        npos_series.append(len(wmap))

    eq = pd.DataFrame({"date": dates, "strat_r": strat_r, "mkt_r": mkt_r, "n_pos": npos_series})
    eq["strat_equity"] = (1 + eq["strat_r"]).cumprod()
    eq["mkt_equity"] = (1 + eq["mkt_r"]).cumprod()

    # 逐筆交易報酬（供勝率）
    tdf = pd.DataFrame(trades)
    if len(tdf):
        rr = []
        for r in tdf.itertuples():
            ei = date_idx[r.entry]
            xi = min(r.exit_idx, len(dates) - 1)
            if r.code in px.columns:
                p0, p1 = px[r.code].iloc[ei], px[r.code].iloc[xi]
                rr.append(p1 / p0 - 1 if p0 > 0 and pd.notna(p1) else np.nan)
            else:
                rr.append(np.nan)
        tdf["trade_ret"] = rr

    metrics = _metrics(eq, tdf)
    return eq, tdf, metrics


def _metrics(eq, tdf):
    def stats(r_col, eq_col):
        r = eq[r_col]
        total = eq[eq_col].iloc[-1] - 1
        n = len(r)
        cagr = eq[eq_col].iloc[-1] ** (TRADING_DAYS / n) - 1 if n > 0 else np.nan
        vol = r.std() * np.sqrt(TRADING_DAYS)
        sharpe = r.mean() / r.std() * np.sqrt(TRADING_DAYS) if r.std() > 0 else np.nan
        cummax = eq[eq_col].cummax()
        mdd = (eq[eq_col] / cummax - 1).min()
        calmar = cagr / abs(mdd) if mdd < 0 else np.nan
        return dict(total=total, cagr=cagr, vol=vol, sharpe=sharpe, mdd=mdd, calmar=calmar)

    m = {"strat": stats("strat_r", "strat_equity"), "mkt": stats("mkt_r", "mkt_equity")}
    if len(tdf) and "trade_ret" in tdf:
        tr = tdf["trade_ret"].dropna()
        m["n_trades"] = len(tr)
        m["hit"] = (tr > 0).mean() if len(tr) else np.nan
        m["avg_trade"] = tr.mean() if len(tr) else np.nan
    m["avg_pos"] = eq["n_pos"].mean()
    m["days"] = len(eq)
    return m


def _spark(eq):
    """權益曲線 inline SVG（策略 vs 0050），自足無外部相依。"""
    W, H, pad = 900, 300, 34
    s, mkt = eq["strat_equity"].values, eq["mkt_equity"].values
    lo, hi = min(s.min(), mkt.min()), max(s.max(), mkt.max())
    rng = hi - lo or 1
    n = len(s)

    def pts(a):
        return " ".join(f"{pad + i / (n - 1) * (W - 2 * pad):.1f},"
                        f"{H - pad - (a[i] - lo) / rng * (H - 2 * pad):.1f}" for i in range(n))
    y1 = H - pad - (1 - lo) / rng * (H - 2 * pad)   # 淨值=1 基準線
    return f"""<svg viewBox="0 0 {W} {H}" width="100%" preserveAspectRatio="xMidYMid meet">
 <line x1="{pad}" y1="{y1:.1f}" x2="{W - pad}" y2="{y1:.1f}" stroke="#334155" stroke-dasharray="4 4"/>
 <polyline fill="none" stroke="#64748b" stroke-width="1.5" points="{pts(mkt)}"/>
 <polyline fill="none" stroke="#22c55e" stroke-width="2" points="{pts(s)}"/>
 <text x="{pad}" y="18" fill="#22c55e" font-size="13">策略</text>
 <text x="{pad + 44}" y="18" fill="#64748b" font-size="13">0050</text>
</svg>"""


def write_report(eq, metrics, cfg):
    m = metrics
    def pct(x): return f"{x * 100:+.1f}%" if pd.notna(x) else "—"
    def num(x): return f"{x:.2f}" if pd.notna(x) else "—"
    s, k = m["strat"], m["mkt"]
    rows = [
        ("總報酬", pct(s["total"]), pct(k["total"])),
        ("年化報酬", pct(s["cagr"]), pct(k["cagr"])),
        ("年化波動", pct(s["vol"]), pct(k["vol"])),
        ("Sharpe", num(s["sharpe"]), num(k["sharpe"])),
        ("最大回撤", pct(s["mdd"]), pct(k["mdd"])),
        ("Calmar", num(s["calmar"]), num(k["calmar"])),
    ]
    mrows = "\n".join(f"<tr><td>{n}</td><td><b>{a}</b></td><td>{b}</td></tr>" for n, a, b in rows)
    html = f"""<!DOCTYPE html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>鉅額×權證策略回測</title>
<style>
 body{{font-family:system-ui,"Noto Sans TC",sans-serif;background:#0f172a;color:#e2e8f0;margin:0;padding:28px 16px}}
 .wrap{{max-width:960px;margin:0 auto}} h1{{font-size:1.4rem;margin:0 0 4px}}
 .sub{{color:#94a3b8;font-size:.9rem;margin:0 0 18px}}
 .card{{background:#1e293b;border:1px solid #334155;border-radius:14px;padding:16px 18px;margin-bottom:18px}}
 table{{width:100%;border-collapse:collapse;font-size:.9rem}}
 th,td{{text-align:left;padding:8px;border-bottom:1px solid #283449}} th{{color:#94a3b8}}
 td:nth-child(2){{color:#22c55e}}
</style></head><body><div class="wrap">
<h1>🧪 鉅額×權證策略回測</h1>
<p class="sub">{cfg} · 期間 {eq['date'].iloc[0]} → {eq['date'].iloc[-1]}（{m['days']} 交易日）·
交易 {m.get('n_trades','—')} 筆 · 勝率 {pct(m.get('hit'))} · 平均持股 {m['avg_pos']:.1f} 檔</p>
<div class="card">{_spark(eq)}</div>
<div class="card"><table>
 <tr><th>指標</th><th>策略</th><th>0050</th></tr>
 {mrows}
</table></div>
<p class="sub">⚠️ 樣本期間與 regime 有限；深挖多年後需重驗。前瞻無前視（D 訊號 → D+1 收盤成交）。</p>
</div></body></html>"""
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"📝 回測報表 → {REPORT}")


def main():
    _enable_utf8_console()
    ap = argparse.ArgumentParser(description="鉅額×權證訊號事件驅動回測")
    ap.add_argument("--verdicts", default="same_dir_buy", help="進場判定(逗號分隔)")
    ap.add_argument("--min-reso", type=int, default=0, help="最低共振窗數")
    ap.add_argument("--hold", type=int, default=10, help="持有交易日數")
    ap.add_argument("--weight", choices=["equal", "conviction"], default="equal")
    ap.add_argument("--cost-bps", type=float, default=58.5, help="來回成本(bps)，預設台股 0.585%%")
    ap.add_argument("--max-pos", type=int, default=0, help="最大同時持股數(0=不限)")
    ap.add_argument("--windows", default="5,10,15,20,W3")
    ap.add_argument("--primary", default="10")
    args = ap.parse_args()

    print("載入資料...")
    block, warr = load_block_history(), load_warrant_history()
    closes, inst = load_close_history(), load_inst_history()
    windows = [w.strip().upper() if w.strip().upper() == "W3" else w.strip()
               for w in args.windows.split(",") if w.strip()]

    print("日級訊號回放（首次較久）...")
    signals = compute_signals(block, warr, closes, inst, windows, args.primary, SArgs())

    verdicts = tuple(v.strip() for v in args.verdicts.split(","))
    eq, tdf, m = run_backtest(signals, closes, hold=args.hold, verdicts=verdicts,
                              min_reso=args.min_reso, weight=args.weight,
                              cost_bps=args.cost_bps, max_pos=args.max_pos)

    cfg = (f"進場={'+'.join(verdicts)}"
           + (f"·共振≥{args.min_reso}" if args.min_reso else "")
           + f"·持有{args.hold}日·{args.weight}"
           + (f"·上限{args.max_pos}檔" if args.max_pos else "")
           + f"·成本{args.cost_bps:.0f}bps")
    s, k = m["strat"], m["mkt"]
    print(f"\n═══ {cfg} ═══")
    print(f"           {'策略':>12} {'0050':>12}")
    print(f"總報酬     {s['total']*100:>11.1f}% {k['total']*100:>11.1f}%")
    print(f"年化報酬   {s['cagr']*100:>11.1f}% {k['cagr']*100:>11.1f}%")
    print(f"年化波動   {s['vol']*100:>11.1f}% {k['vol']*100:>11.1f}%")
    print(f"Sharpe     {s['sharpe']:>12.2f} {k['sharpe']:>12.2f}")
    print(f"最大回撤   {s['mdd']*100:>11.1f}% {k['mdd']*100:>11.1f}%")
    print(f"Calmar     {s['calmar']:>12.2f} {k['calmar']:>12.2f}")
    print(f"交易筆數 {m.get('n_trades','—')} · 勝率 {m.get('hit',float('nan'))*100:.0f}% · "
          f"平均每筆 {m.get('avg_trade',float('nan'))*100:+.2f}% · 平均持股 {m['avg_pos']:.1f} 檔")
    write_report(eq, m, cfg)


if __name__ == "__main__":
    main()
