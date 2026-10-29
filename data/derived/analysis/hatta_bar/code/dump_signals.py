"""검증용: 엔진이 내는 신호 시각(결과 창 거르기·겹침 빼기·시간대 거르기 전)을 칸 설정별로 CSV로 쓴다.
사용: py dump_signals.py yf1m out.csv [tv]       (CELLS의 칸 전부, 115종목. tv = 체결 건수 대신 한 점 봉만 거름)
열: cell, sym, ts_et(봉 시작, 뉴욕), pattern(A/B), k(이탈 봉 번호), S, touch, level"""
import sys
import numpy as np
import pandas as pd
import engine as E
from build import SYMS

# (이름, 패턴, N, tol, K방식, p, K, ema, D, v, 모양) — 원안과 몇 가지 다른 칸
CELLS = [
    ("orig_A", "A", 120, 0.0005, "level", 3, 2, "persist", 0.02, 0.0, ("wick", 0.5, 1.0, False)),
    ("orig_B", "B", 120, 0.0005, None, 3, 1, "persist", 0.02, 0.0, (True, "basic")),
    ("alt_A1", "A", 60, 0.0, "touch", 2, 3, "now", 0.0, 2.0, ("cpos", 0.65, 0.0, True)),
    ("alt_A2", "A", 390, 0.001, "level", 5, 3, "none", 0.03, 0.0, ("wick", 0.65, 2.0, False)),
    ("alt_B1", "B", 240, 0.001, "level", 2, 2, "now", 0.01, 2.0, (False, "full")),
    ("alt_B2", "B", 60, 0.0, "touch", 5, 2, "none", 0.0, 0.0, (False, "reclaim")),
]


def select(X, pat, kmode, p, K, ema, D, v, shape):
    m = X["sane"].to_numpy().copy()
    if kmode is not None:
        m &= X[f"{kmode}{p}"].to_numpy() >= K
    if ema == "now":
        m &= X["stack"].to_numpy()
    elif ema == "persist":
        m &= X["persist"].to_numpy()
    m &= X["dd"].to_numpy() >= D
    if v > 0:
        m &= X["volr"].to_numpy() >= v
    if pat == "A":
        rm, r, mm, ab = shape
        m &= X["openok"].to_numpy() & (X[rm].to_numpy() >= r) & (X["rngatr"].to_numpy() >= mm)
        if ab:
            m &= X["above"].to_numpy()
    else:
        bl, eng = shape
        m &= X[f"eng_{eng}"].to_numpy()
        if bl:
            m &= X["below"].to_numpy()
    return X[m]


def main(src, out, tv=False):
    rows = []
    for sym in SYMS:
        d = E.load(sym, src)
        if d is None or len(d) < 500:
            continue
        a = E.prep(d, tv=tv)
        o = E.bar_outcomes(a)
        cache = {}
        for name, pat, N, tol, kmode, p, K, ema, D, v, shape in CELLS:
            if (N, tol) not in cache:
                cache[(N, tol)] = E.candidates(sym, a, o, N, tol)
            A, B = cache[(N, tol)]
            X = A if pat == "A" else B
            if X is None:
                continue
            Y = select(X, pat, kmode, p, K, ema, D, v, shape)
            for _, r in Y.iterrows():
                ts = pd.Timestamp(int(r.ts) * 60, unit="s", tz="UTC").tz_convert(E.ET)
                rows.append((name, sym, ts.strftime("%Y-%m-%d %H:%M"), pat, int(r.k), r.S,
                             int(r[f"touch{p}"]), int(r[f"level{p}"])))
    pd.DataFrame(rows, columns=["cell", "sym", "ts_et", "pattern", "k", "S", "touch", "level"]).to_csv(out, index=False)
    print(len(rows), "signals")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], len(sys.argv) > 3 and sys.argv[3] == 'tv')
