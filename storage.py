"""
storage — 年度分檔儲存 + 滾動留存

為什麼分檔：單一 CSV 累積 10 年會超過 GitHub 單檔 100MB 硬限制
（權證流 ~11MB/年），且每日 commit 重寫整個大檔讓 repo 膨脹。
按年度分檔後：舊年份檔永不變動（git delta 幾乎為零）、
單檔最大約 11MB、留存政策 = 刪除過舊的整年檔，乾淨俐落。

佈局：data/<name>/YYYY.csv；舊制單檔 data/<name>.csv 首次儲存時自動遷移移除。
"""

import glob
import os

import pandas as pd

_STR_COLS = ("code", "underlying", "window")   # 這些欄一律以字串讀入


def _read(path, columns):
    df = pd.read_csv(path, dtype={c: str for c in _STR_COLS})
    df["date"] = df["date"].astype(str)
    for c in columns:          # 舊檔缺新欄 → 補 NaN，維持 schema 一致
        if c not in df.columns:
            df[c] = float("nan")
    return df[columns]


def load_history(base, columns, years=None):
    """
    讀歷史。base 不含副檔名（如 data/stock_closes）：
    讀 base/YYYY.csv 全部（years=N 時只讀最近 N 個年檔）＋舊制單檔 base.csv。
    兩者日期重疊時以年檔為準。
    """
    frames = []
    part_dates = set()
    if os.path.isdir(base):
        files = sorted(glob.glob(os.path.join(base, "*.csv")))
        if years:
            files = files[-int(years):]
        for f in files:
            df = _read(f, columns)
            frames.append(df)
            part_dates.update(df["date"])
    legacy = base + ".csv"
    if os.path.exists(legacy):
        df = _read(legacy, columns)
        if part_dates:
            df = df[~df["date"].isin(part_dates)]
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=columns)
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values("date", kind="stable").reset_index(drop=True)


def save_history(base, df, keep_years=None, sort_cols=("date",)):
    """
    寫歷史：df 依年度拆檔寫入 base/YYYY.csv（只寫 df 中出現的年度，
    未載入的舊年檔不動）。keep_years=N 時刪除超過 N 年的整年檔並過濾 df。
    寫入成功後移除舊制單檔 base.csv（完成遷移）。
    """
    df = df.sort_values(list(sort_cols), kind="stable").reset_index(drop=True)
    os.makedirs(base, exist_ok=True)

    if keep_years:
        cutoff_year = pd.Timestamp.today().year - int(keep_years) + 1
        df = df[df["date"].str[:4].astype(int) >= cutoff_year]
        for f in glob.glob(os.path.join(base, "*.csv")):
            try:
                if int(os.path.splitext(os.path.basename(f))[0]) < cutoff_year:
                    os.remove(f)
            except ValueError:
                continue

    for year, g in df.groupby(df["date"].str[:4]):
        g.to_csv(os.path.join(base, f"{year}.csv"), index=False)

    legacy = base + ".csv"
    if os.path.exists(legacy):
        os.remove(legacy)
    return df
