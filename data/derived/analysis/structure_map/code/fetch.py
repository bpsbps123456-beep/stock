"""바이낸스 BTCUSDT 봉 받기 (검증용, 공개 API)."""
import json
import pathlib
import urllib.request

OUT = pathlib.Path(__file__).resolve().parent.parent / "cache"
SYM = "BTCUSDT"


def klines(interval: str, limit: int = 1000) -> list:
    url = f"https://api.binance.com/api/v3/klines?symbol={SYM}&interval={interval}&limit={limit}"
    rows = json.load(urllib.request.urlopen(url, timeout=20))
    return [[r[0], float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]), r[6]] for r in rows]


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for iv in ("1h", "2h", "4h", "12h", "1d"):
        rows = klines(iv)
        (OUT / f"{SYM}_{iv}.json").write_text(json.dumps(rows))
        print(iv, len(rows), rows[0][0], rows[-1][0])
