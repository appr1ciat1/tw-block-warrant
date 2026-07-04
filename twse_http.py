"""
twse_http — TWSE 抓取共用工具：SSL 寬鬆 + retry-backoff 反爬蟲退避

TWSE 對高頻抓取的反爬蟲：不回 429，而是 `HTTP 307 Temporary Redirect`
（重導到擋頁）整段封鎖來源 IP。深挖狂抓（每次 ~1000+ 請求）易觸發。
本模組集中處理：
- SSL 寬鬆（TWSE 憑證缺 SKI 擴展，OpenSSL3 嚴格檢查會擋）。
- 遇 307/429/403/5xx 退避重試（給 TWSE 時間解除軟封鎖）。
- 節奏 REQUEST_PAUSE 可用環境變數 TWSTK_PAUSE 覆寫（深挖放慢避免觸發封鎖）。
"""
import os
import ssl
import json
import time
import urllib.request
import urllib.error

_CTX = ssl.create_default_context()
_CTX.verify_flags &= ~ssl.VERIFY_X509_STRICT

# 每請求間隔（秒）。深挖時設 TWSTK_PAUSE=6 之類放慢，避免踩反爬蟲。
REQUEST_PAUSE = float(os.environ.get("TWSTK_PAUSE", "3.0"))

# 退避秒數（遞增）。刻意精簡：持續封鎖時每請求最多浪費 ~33s，避免整段深挖
# 被塞爆到 timeout（大段深挖前應先小段探測封鎖是否解除）。
_RETRY_WAITS = (8, 25)
_RETRY_CODES = {307, 429, 403, 500, 502, 503, 504}


def fetch_json(url, timeout=60):
    """抓 JSON；遇反爬蟲/暫時性錯誤退避重試，全部失敗才拋最後一個例外。"""
    last = None
    for wait in (0,) + _RETRY_WAITS:
        if wait:
            time.sleep(wait)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "twstk/1.0"})
            with urllib.request.urlopen(req, context=_CTX, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last = e
            if e.code in _RETRY_CODES:
                continue           # 反爬蟲/暫時性 → 退避重試
            raise
        except urllib.error.URLError as e:
            last = e
            continue               # 網路暫時性 → 重試
    raise last
