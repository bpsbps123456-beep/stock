import numpy as np, pandas as pd
from core import *
D = load()
halfs = set()
for s in SYMS: halfs |= half_days(D[(s, "1h")])
print("half days:", sorted(halfs)[-4:])
s = "SNDK"; base = D[(s, "5m")]
BA = base_arrays(base, halfs)
for eth in (False, True):
    tb = tf_bars(base, 40, eth, halfs)
    K = kalman(tb.C.values, 0.01, 3.0)
    a5 = atr(tb.H.values, tb.L.values, tb.C.values, 5)
    line, tr = ratchet(K, a5, 0.5)
    t_end = pd.to_datetime(tb.end_ns.values, utc=True).tz_convert("America/New_York")
    print("ETH" if eth else "RTH", "bars", len(tb), "first ends", [x.strftime("%H:%M") for x in t_end[:3]],
          "last", t_end[-1], "line", round(line[-1], 2), "trend", tr[-1])
    tf_idx = map_completed(BA["start_ns"], tb.end_ns.values)
    # 원본 2색 규칙: 초록=매수(1) 빨강=매도(-1), 지정가 없음
    z = tr.copy(); z[z == 0] = -9
    j0 = int(np.searchsorted(BA["dates"], pd.Timestamp("2026-07-16").date()))
    eq, n, log, ex = simulate(BA["o"], BA["h"], BA["l"], BA["c"], BA["is_reg"], BA["is_awake"], BA["is_end"],
                              tf_idx, z, np.full(len(tb), np.nan), np.full(len(tb), np.nan), j0, True, False, False, 0.0005, True)
    bh = buy_hold(BA["o"], BA["c"], BA["is_reg"], BA["is_end"], j0)
    print("   2색 규칙 수익 %.1f%%  체결 %d  평균노출 %.0f%%  | 보유 %.1f%%" % ((eq[-1]-1)*100, n, ex.mean()*100, (bh[-1]-1)*100))
    kst = pd.to_datetime(BA["start_ns"][log[:, 0].astype(int)], utc=True).tz_convert("Asia/Seoul")
    print("   체결 시각(KST) 예:", [x.strftime("%m-%d %H:%M") for x in kst[:6]])
