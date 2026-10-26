"""이격매수 종목 궁합 연구용 최근 1분봉(2026-10-09). dip_reason의 fetch.download로 새 폴더에 받고(1분봉 29일·5분봉 60일·일봉 1년),
예전 폴더(dip/raw, 9/9~10/6)에만 있는 앞쪽 1분봉을 이어 붙여 9/9~ 지금까지를 만든다.
대상: dip_reason 유니버스(AI 115 + 바이오 13) ∪ 사용자 78종목 ∪ 비교 ETF 7개.
사용: py fetch_recent.py [새 폴더 이름 예: raw_1010] [앞쪽을 이어 붙일 폴더 이름 예: raw_1009]  (기본 raw_1009 ← raw)"""
import os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "../../dip_reason/code"))
from fetch import download, universe, user_universe, ETF

SP = r"C:/Users/geunt/AppData/Local/Temp/claude/C--Users-geunt-OneDrive-Desktop---/4e686c73-74be-4337-8ccc-2f6136beaf7d/scratchpad/dip"
if __name__ == "__main__":
    NEW = os.path.join(SP, sys.argv[1] if len(sys.argv) > 1 else "raw_1009")
    OLD = os.path.join(SP, sys.argv[2] if len(sys.argv) > 2 else "raw")
    os.makedirs(NEW, exist_ok=True)
    syms = sorted(set(universe()) | set(user_universe()["합집합"]) | set(ETF))
    download(syms, NEW)
    n = 0
    for s in syms:
        p_new, p_old = os.path.join(NEW, f"1m_{s}.parquet"), os.path.join(OLD, f"1m_{s}.parquet")
        if not (os.path.exists(p_new) and os.path.exists(p_old)):
            continue
        a, b = pd.read_parquet(p_old), pd.read_parquet(p_new)
        m = pd.concat([a[a.index < b.index.min()], b]).sort_index()
        m[~m.index.duplicated()].to_parquet(p_new)
        n += 1
    print(f"{len(syms)}종목 요청, 1분봉 이어 붙임 {n}종목 → {NEW}")
