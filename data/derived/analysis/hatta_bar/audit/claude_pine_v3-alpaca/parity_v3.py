"""현재 Pine 직역(pine_v3.py)을 Alpaca 1분봉 전체(04:00~20:00 ET, 6/24~10/28)에 돌려 엔진 덤프(tv=True)와 신호 집합을 비교한다.
성과 수치는 계산하지 않는다(눈가림 연구). 비교 키 = (cell, sym, ts_et 신호 봉 시작 뉴욕 시각).
dedup30 = False(엔진 덤프는 겹침 빼기 전). 체결 건수 거르기 없음(Pine 위생 = 거래대금·봉 길이·한 가격 봉).
사용: py parity_v3.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pine_v3 as P  # noqa: E402

ROOT = HERE.parents[1]
CACHE = ROOT / "cache"

# 칸 설정 = code/dump_signals.py CELLS 를 Pine 입력으로 옮긴 것 (orig = Pine 기본값)
FIRST = dict(brkMd=P.B_FIRST)
CELLS = {
    "orig_A": dict(),
    "orig_B": dict(),
    "first_A": dict(FIRST, pSw=3, rngMin=1.0),
    "first_B": dict(FIRST, pSw=3),
    "alt_A1": dict(FIRST, nBars=60, tolPct=0.0, pSw=2, kModeA=P.M_TOUCH, kMinA=3, trendMd=P.T_NOW, dropPct=0.0,
                   volMin=2.0, recMd=P.R_CPOS, recMin=0.65, rngMin=0.0, needAb=True),
    "alt_A2": dict(FIRST, nBars=390, tolPct=0.1, pSw=5, kModeA=P.M_LEVEL, kMinA=3, trendMd=P.T_OFF, dropPct=3.0,
                   volMin=0.0, recMd=P.R_WICK, recMin=0.65, rngMin=2.0, needAb=False),
    "alt_B1": dict(FIRST, nBars=240, tolPct=0.1, pSw=2, kModeB=P.M_LEVEL, kMinB=2, trendMd=P.T_NOW, dropPct=1.0,
                   volMin=2.0, needBl=False, engMd=P.E_FULL),
    "alt_B2": dict(FIRST, nBars=60, tolPct=0.0, pSw=5, kModeB=P.M_TOUCH, kMinB=2, trendMd=P.T_OFF, dropPct=0.0,
                   volMin=0.0, needBl=False, engMd=P.E_RECL),
}
PAT = {k: k.split("_")[1][0] for k in CELLS}


def load(sym):
    d = pd.read_parquet(CACHE / "alp1m" / f"{sym}.parquet")
    idx = d.index
    t = idx.tz_convert("UTC").as_unit("ms").asi8.astype(np.int64)  # Pine time = 봉 시작 epoch ms
    ny = idx.tz_convert("America/New_York")
    kst = idx.tz_convert("Asia/Seoul")
    day = (ny.year * 10000 + ny.month * 100 + ny.day).to_numpy(np.int64)
    return (d["Open"].to_numpy(float), d["High"].to_numpy(float), d["Low"].to_numpy(float),
            d["Close"].to_numpy(float), d["Volume"].to_numpy(float), t, day,
            ny.hour.to_numpy(np.int64), kst.hour.to_numpy(np.int64), ny.strftime("%Y-%m-%d %H:%M").to_numpy())


def universe():
    return sorted(p.stem for p in (CACHE / "alp1m").glob("*.parquet") if p.stem not in ("SPY", "QQQ"))


def run_all():
    rows = []
    for sym in universe():
        o, h, l, c, v, t, day, hr, hk, ts = load(sym)
        for cell, ov in CELLS.items():
            prm = dict(P.DEFAULTS); prm.update(ov); prm["dedup30"] = False
            kA, kB, S, nT, nL, brk = P.run(o, h, l, c, v, t, day, hr, hk, *[prm[k] for k in P.ORDER])
            sig = kA if PAT[cell] == "A" else kB
            for i in np.flatnonzero(sig):
                k = i if PAT[cell] == "A" else i - 1
                rows.append((cell, sym, ts[i], PAT[cell], int(i), int(k), S[k], nT[k], nL[k]))
    return pd.DataFrame(rows, columns=["cell", "sym", "ts_et", "pattern", "i", "k", "S", "touch", "level"])


def main():
    mine = run_all()
    mine.to_csv(HERE / "pine_signals.csv", index=False)
    eng = pd.read_csv(CACHE / "engine_signals_alp1m_tv.csv")
    m = mine.merge(eng, on=["cell", "sym", "ts_et"], how="outer", suffixes=("_pine", "_eng"), indicator=True)
    m["pre"] = m.ts_et.str[11:] < "09:30"
    m.to_csv(HERE / "diff.csv", index=False)
    out = []
    for cell in CELLS:
        mm = m[m.cell == cell]
        both = mm._merge == "both"
        out.append(dict(cell=cell, pine=int((mm._merge != "right_only").sum()), engine=int((mm._merge != "left_only").sum()),
                        matched=int(both.sum()), pine_only=int((mm._merge == "left_only").sum()),
                        engine_only=int((mm._merge == "right_only").sum()),
                        pre_pine=int((mm.pre & (mm._merge != "right_only")).sum()),
                        pre_matched=int((mm.pre & both).sum()),
                        k_S_touch_level_mismatch=int((both & ((mm.k_pine != mm.k_eng) | ~np.isclose(mm.S_pine, mm.S_eng, rtol=0, atol=1e-9)
                                                              | (mm.touch_pine != mm.touch_eng) | (mm.level_pine != mm.level_eng))).sum())))
    r = pd.DataFrame(out)
    r.to_csv(HERE / "summary.csv", index=False)
    print(r.to_string(index=False))
    print("total", r[["pine", "engine", "matched", "pine_only", "engine_only", "pre_pine", "pre_matched"]].sum().to_dict())


if __name__ == "__main__":
    main()
