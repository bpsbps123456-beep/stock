"""z3 comp 갈래 1단계: 144종목 40분 연장 봉을 '거래일 × 시각 칸' 격자로 모은다(기존 추세 5단계와의 보완성용).

- 칸 = (공통 거래일 69개, 04:00 ET부터 40분 칸 24개). 봉이 없는 칸은 비운다.
- 칸마다: 원안 상태 s, 추세 판정 flag 28개, 기존 5단계 z5(마지막 완성 정규장 일봉), 시가 O.
- 뒤 수익(h = 1·5거래일): 그 봉 마감 뒤 다음 봉 시가에 사서, h거래일 뒤 같은 시각 칸(그 칸이 비면 그날 그 뒤 첫 봉) 시가에 판다.
  그날 그 시각 뒤 봉이 없으면 비운다. 깨어 있음 표시는 산 봉(다음 봉) 시가 기준(서머타임 시간표·11/1 표준시 시간표 두 가지).
- 결과: results/z3comp_grid.npz
"""
from __future__ import annotations

import pathlib
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from engine import CACHE, HERE  # noqa: E402
from zones3 import CLASSIFIERS, awake_std, cname, prepare  # noqa: E402

SLOTS = 24  # 04:00~20:00 ET, 40분
FLAGS = [cname(k, p) for k, p in CLASSIFIERS]
HS = (1, 5)


def one(sym):
    P = prepare(sym, "5m", "dst")
    tb = P["tb"]
    st = pd.DatetimeIndex(tb.start)
    slot = ((st.hour * 60 + st.minute).values - 240) // 40
    return dict(sym=sym, day=np.array(tb.day.values), slot=slot.astype(np.int64), O=P["O"], s=P["s"].astype(np.int8),
                z5=P["z5"], flags=np.stack([P["flags"][f] for f in FLAGS]).astype(np.uint8),
                aw_dst=np.asarray(P["awake"], bool), aw_std=awake_std(tb.start))


def main():
    syms = sorted(p.name[:-11] for p in CACHE.glob("*_5m.parquet"))
    t0 = time.time()
    with Pool(3) as pool:
        res = pool.map(one, syms, chunksize=4)
    print(f"prepare {len(res)}종목 {time.time() - t0:.0f}s", flush=True)
    days = np.array(sorted(set().union(*[set(r["day"]) for r in res])))
    D, S, F = len(days), len(res), len(FLAGS)
    dix = {d: i for i, d in enumerate(days)}
    O = np.full((S, D, SLOTS), np.nan)
    s = np.full((S, D, SLOTS), -1, np.int8)
    z5 = np.full((S, D, SLOTS), np.nan)
    fl = np.zeros((S, F, D, SLOTS), np.uint8)
    awd = np.zeros((S, D, SLOTS), bool)
    aws = np.zeros((S, D, SLOTS), bool)
    for i, r in enumerate(res):
        di = np.array([dix[d] for d in r["day"]])
        k = r["slot"]
        assert k.min() >= 0 and k.max() < SLOTS
        assert len(set(zip(di, k))) == len(k)  # 칸 중복 없음
        O[i, di, k] = r["O"]
        s[i, di, k] = r["s"]
        z5[i, di, k] = r["z5"]
        fl[i, :, di, k] = r["flags"].T
        awd[i, di, k] = r["aw_dst"]
        aws[i, di, k] = r["aw_std"]
    present = ~np.isnan(O)
    # 뒤 수익
    flat_o = O.reshape(S, D * SLOTS)
    flat_p = present.reshape(S, D * SLOTS)
    M = D * SLOTS
    nxt = np.full((S, M + 1), M, np.int64)  # i 이상에서 첫 봉 위치(없으면 M)
    for i in range(M - 1, -1, -1):
        nxt[:, i] = np.where(flat_p[:, i], i, nxt[:, i + 1])
    idx = np.arange(M)
    entry = nxt[:, np.minimum(idx + 1, M)]  # 다음 봉
    entry = np.where(idx[None, :] + 1 <= M - 1, entry, M)
    ok_e = entry < M
    e_c = np.where(ok_e, entry, 0)
    e_day, e_slot = e_c // SLOTS, e_c % SLOTS
    ent_o = np.where(ok_e, np.take_along_axis(flat_o, e_c, 1), np.nan)
    aw_d = np.where(ok_e, np.take_along_axis(awd.reshape(S, M), e_c, 1), False)
    aw_s = np.where(ok_e, np.take_along_axis(aws.reshape(S, M), e_c, 1), False)
    fwd = {}
    gap = {}
    for h in HS:
        tday = e_day + h
        ok = ok_e & flat_p & (tday < D)
        tgt = np.where(ok, tday * SLOTS + e_slot, 0)
        x = np.take_along_axis(nxt, tgt, 1)
        ok &= (x < M) & (x // SLOTS == tday)
        xc = np.where(ok, x, 0)
        ex_o = np.take_along_axis(flat_o, xc, 1)
        fwd[h] = np.where(ok, ex_o / ent_o - 1, np.nan).reshape(S, D, SLOTS)
        gap[h] = np.where(ok, xc % SLOTS - e_slot, -1).reshape(S, D, SLOTS)  # 같은 칸이 비어 뒤로 밀린 칸 수
    out = HERE / "results" / "z3comp_grid.npz"
    np.savez_compressed(out, syms=np.array(syms), days=np.array([str(d) for d in days]), flags=np.array(FLAGS),
                        O=O, s=s, z5=z5, fl=fl, present=present,
                        aw_dst=aw_d.reshape(S, D, SLOTS), aw_std=aw_s.reshape(S, D, SLOTS),
                        fwd1=fwd[1], fwd5=fwd[5], gap1=gap[1], gap5=gap[5],
                        entry_gapday=(np.where(ok_e, e_day, -1).reshape(S, D, SLOTS) - np.arange(D)[None, :, None]))
    print("저장", out, f"S={S} D={D} 봉 {int(present.sum())}", flush=True)
    for h in HS:
        v = ~np.isnan(fwd[h])
        print(f"h{h}: 뒤 수익 있는 봉 {int(v.sum())} ({v.sum() / present.sum():.1%}), 같은 칸이 비어 뒤로 민 봉 {int((gap[h] > 0).sum())}")
    print("z5 없음(5단계 계산 전)", int((present & np.isnan(z5)).sum()), "봉")


if __name__ == "__main__":
    main()
