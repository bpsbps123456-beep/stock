"""바이낸스 현물 봉 이력 받기 (백테스트용, 공개 API, 페이지 넘김)."""
import json
import pathlib
import sys
import time
import urllib.request

OUT = pathlib.Path(__file__).resolve().parent.parent / "cache" / "hist"
MS = {"5m": 300_000, "1h": 3600_000, "2h": 7200_000, "4h": 14400_000, "12h": 43200_000, "1d": 86400_000}
START = {"5m": "2024-12-25", "1h": "2024-07-01", "2h": "2024-07-01", "4h": "2024-07-01", "12h": "2023-06-01", "1d": "2023-01-01"}


def ms_of(day: str) -> int:
    t = time.strptime(day, "%Y-%m-%d")
    return int(time.mktime(t) - time.timezone) * 1000   # UTC 자정


def fetch(sym: str, iv: str) -> list:
    rows, start = [], ms_of(START[iv])
    while True:
        url = f"https://api.binance.com/api/v3/klines?symbol={sym}&interval={iv}&startTime={start}&limit=1000"
        part = json.load(urllib.request.urlopen(url, timeout=30))
        if not part:
            break
        rows += [[r[0], float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])] for r in part]
        if len(part) < 1000:
            break
        start = part[-1][0] + MS[iv]
        time.sleep(0.15)
    now_ms = int(time.time() * 1000)
    return [r for r in rows if r[0] + MS[iv] <= now_ms]      # 아직 안 끝난 봉은 뺀다


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    ivs = sys.argv[1].split(",") if len(sys.argv) > 1 else ["1h", "2h", "4h", "12h", "1d"]
    for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        for iv in ivs:
            rows = fetch(sym, iv)
            gaps = sum(1 for a, b in zip(rows, rows[1:]) if b[0] - a[0] != MS[iv])
            (OUT / f"{sym}_{iv}.json").write_text(json.dumps(rows))
            print(sym, iv, len(rows), "gaps", gaps, time.strftime("%Y-%m-%d %H:%M", time.gmtime(rows[0][0] / 1000)), "~", time.strftime("%Y-%m-%d %H:%M", time.gmtime(rows[-1][0] / 1000)))
