"""z3 diag 갈래: BRIEF2 채택 기준 1(진단)과 사후 표지 대조의 원자료를 만든다.

봉 진단(후보 28개, 1시간봉 창은 HTF 160·320 빼고 26개)
- 원안(A3)의 봉별 보유 pos_j(zones3.sim_frac의 pos_out, 실제 판은 체결 지연 반영)와 봉 수익 r_j(시가→다음 시가, 단순).
- 봉 j에 걸린 구간 = 직전 봉(j−1) 마감의 flag. 첫 봉(j=0)은 뺀다.
- 집합 Z(추세 = flag 참, 애매 = 거짓)의 노출 맞춘 초과 = Σ(pos·r − c·|Δpos|) − x_Z·Σr, x_Z = Z의 평균 pos. 비용은 바꾼 봉에 붙인다.
- 판 5개: ideal(모든 봉, 비용 0), real0·real003·real01(깨어 있음 서머타임 시간표, 편도 0·0.03%·0.1%), std01(표준시 시간표, 0.1%).
- 귀무: flag만 공통 거래일 달력에서 k일 원형 이동(같은 시각 칸, 신호 s·pos는 그대로). 출처 칸이 없으면 직전 봉의 이동 flag를 유지한다.
  15 ≤ k ≤ D−15. 점검용으로 np.roll(flag, k×하루 봉 수 중앙값)도 real01에서 같이 잰다.

사후 표지(5분봉 창, 판 real01·ideal)
- 날 d = 직전 거래일 16:00 ET(정규장 마감)부터 d의 16:00 ET 전까지 시작한 봉. 날 수익 = Π(1+r_j) − 1(16:00 봉 시가 → 다음 16:00 봉 시가).
- 장 시작 전 표지 = 그날 첫 봉에 걸린 구간(직전 정규장 마감 봉의 flag·s). 첫 날(7/17, 직전 봉 없음)과 10/22 16:00 뒤 봉은 뺀다.

py code/z3diag_run.py [프로세스 수=3] → results/z3diag_raw_5m.npz, z3diag_raw_1h.npz, z3diag_days_5m.npz, z3diag_run_meta.json
"""
from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from engine import CACHE, HERE, SESS, WIN
from zones3 import CLASSIFIERS, awake_std, cname, prepare, run_symbol, sim_frac

OUT = HERE / "results"
SYMS = sorted(p.stem[:-3] for p in CACHE.glob("*_5m.parquet"))
GUARD = 15
BOARDS = [("ideal", "ideal", 0.0), ("real0", "dst", 0.0), ("real003", "dst", 0.0003), ("real01", "dst", 0.001), ("std01", "std", 0.001)]
CANDS_ALL = [cname(k, p) for k, p in CLASSIFIERS]
TFMIN = {"5m": 40, "1h": 60}


def cands_for(base):
    return [c for c in CANDS_ALL if not (base == "1h" and c in ("HTF_tf160", "HTF_tf320"))]


def work(args):
    sym, base = args
    P = prepare(sym, base, "dst")
    tb = P["tb"]
    n = len(tb)
    O, C, seg, s = P["O"], P["C"], P["seg"], P["s"]
    aw = {"ideal": np.ones(n, np.bool_), "dst": P["awake"], "std": awake_std(tb.start)}
    tgt = s.astype(np.float64)
    pos = {}
    chk = {}
    for nm, a in aw.items():
        po = np.zeros(n)
        out = sim_frac(O, C, tgt, seg, a, 0.001 if nm != "ideal" else 0.0, po)
        pos[nm] = po
        chk[nm] = (float(np.expm1(out[0] + out[1])), float(np.expm1(out[8])))
    st = pd.DatetimeIndex(tb.start)
    slot = (np.asarray(st.hour * 60 + st.minute) - SESS["eth"][0]) // TFMIN[base]
    cands = cands_for(base)
    F = np.stack([P["flags"][c] for c in cands]).astype(np.bool_)
    return dict(sym=sym, day=np.asarray(tb.day.values), slot=slot.astype(np.int64), seg=seg, s=s, O=O, C=C,
                start_ns=st.asi8, F=F, pos=pos, chk=chk)


def shifted_src(day_i, slot, D, k, grid):
    """달력 k일 이동의 출처 봉 번호(없으면 직전 봉의 출처를 유지, 처음부터 없으면 −1)와 빈 칸 수."""
    src = grid[(day_i - k) % D, slot]
    miss = src < 0
    if miss.any():
        pos = np.where(~miss, np.arange(len(src)), -1)
        last = np.maximum.accumulate(pos)
        src = np.where(last >= 0, src[np.maximum(last, 0)], -1)
    return src, int(miss.sum())


def bar_sets(Fz, valid, win_masks, comps):
    """Fz: (m, n) 걸린 구간 flag(참=추세). comps: 판별 (a, p, r). 돌려줌 (판, 창, m, 집합 2[추세, 애매], 값 2[초과, 봉 수])."""
    m, n = Fz.shape
    Ff = Fz.astype(np.float64)
    res = np.zeros((len(comps), len(win_masks), m, 2, 2))
    for bi, (a, p, r) in enumerate(comps):
        for wi, wm in enumerate(win_masks):
            w = (wm & valid).astype(np.float64)
            M = np.stack([a * w, p * w, r * w, w], axis=1)  # (n, 4)
            T = Ff @ M  # 추세 집합 합
            tot = M.sum(axis=0)
            A = tot[None, :] - T  # 애매 집합 합
            for si, S in enumerate((T, A)):
                nz = S[:, 3]
                with np.errstate(invalid="ignore", divide="ignore"):
                    x = np.where(nz > 0, S[:, 1] / nz, np.nan)
                res[bi, wi, :, si, 0] = S[:, 0] - x * S[:, 2]
                res[bi, wi, :, si, 1] = nz
    return res


def day_frame(R, base):
    """사후 표지용 날 묶음(16:00 ET → 16:00 ET). 돌려줌: 봉별 날 번호(−1 제외), 날 목록."""
    day = R["day"]
    mins = R["slot"] * TFMIN[base] + SESS["eth"][0]
    udays = np.array(sorted(set(day)))
    di = np.searchsorted(udays, day)
    rdi = np.where(mins >= 960, di + 1, di)  # 16:00 ET 이후 시작한 봉은 다음 거래일 몫
    rdi = np.where(rdi >= len(udays), -1, rdi)
    rdi[rdi == 0] = -1  # 첫 날은 직전 마감 봉이 없다
    return rdi, udays


def process(base, data, cal):
    D = len(cal)
    dix = {d: i for i, d in enumerate(cal)}
    ks = [0] + list(range(GUARD, D - GUARD + 1))
    cands = cands_for(base)
    nb, nk, nc = len(BOARDS), len(ks), len(cands)
    syms = [R["sym"] for R in data]
    RES = np.full((len(syms), nb, 2, nk, nc, 2, 2), np.nan)
    ROLL = np.full((len(syms), 2, nk, nc, 2, 2), np.nan)
    miss_cells = np.zeros(nk, np.int64)
    days_rows = []
    zpre_all = []
    info = {}
    for si, R in enumerate(data):
        n = len(R["O"])
        O, C = R["O"], R["C"]
        nxt = np.r_[O[1:], C[-1]]
        r = nxt / O - 1.0
        day_i = np.array([dix[d] for d in R["day"]])
        nsl = int(R["slot"].max()) + 1
        grid = np.full((D, nsl), -1, np.int64)
        grid[day_i, R["slot"]] = np.arange(n)
        assert (grid >= 0).sum() == n, (R["sym"], "칸 중복")
        bpd = int(pd.Series(R["day"]).value_counts().median())
        valid = np.r_[False, np.ones(n - 1, np.bool_)]
        wins = (R["seg"] == 0, R["seg"] == 1)
        comps = []
        for nm, awk, c in BOARDS:
            p = R["pos"][awk]
            dp = np.abs(np.diff(np.r_[0.0, p]))
            comps.append((p * r - c * dp, p, r))
        Fz_k = np.zeros((nk, nc, n), np.bool_)
        Froll = np.zeros((nk, nc, n), np.bool_)
        for ki, k in enumerate(ks):
            if k == 0:
                Fk = R["F"]
            else:
                src, mc = shifted_src(day_i, R["slot"], D, k, grid)
                miss_cells[ki] += mc
                Fk = np.where(src[None, :] >= 0, R["F"][:, np.maximum(src, 0)], False)
            Fz_k[ki, :, 1:] = Fk[:, :-1]
            Fr = np.roll(R["F"], k * bpd, axis=1) if k else R["F"]
            Froll[ki, :, 1:] = Fr[:, :-1]
        res = bar_sets(Fz_k.reshape(nk * nc, n), valid, wins, comps)
        RES[si] = res.reshape(nb, 2, nk, nc, 2, 2)
        rr = bar_sets(Froll.reshape(nk * nc, n), valid, wins, [comps[3]])
        ROLL[si] = rr.reshape(2, nk, nc, 2, 2)
        info[R["sym"]] = dict(n=n, days=int(len(np.unique(day_i))), bpd=bpd, missing_vs_bpd=int(bpd * len(np.unique(day_i)) - n),
                              chk={k: [round(v[0] * 100, 2), round(v[1] * 100, 2)] for k, v in R["chk"].items()})
        if base == "5m":
            rdi, udays = day_frame(R, base)
            s_prev = np.r_[0, R["s"][:-1]]
            for d in np.unique(rdi[rdi >= 0]):
                j = np.flatnonzero(rdi == d)
                assert j[-1] - j[0] + 1 == len(j), (R["sym"], d, "날 묶음 불연속")
                dd = udays[d]
                sg = 0 if WIN[base]["sel"][0] <= dd <= WIN[base]["sel"][1] else (1 if WIN[base]["conf"][0] <= dd <= WIN[base]["conf"][1] else -1)
                row = dict(sym=R["sym"], date=str(dd), seg=sg, j0=int(j[0]), nbar=len(j),
                           ret=float(np.prod(1 + r[j]) - 1), s0=int(s_prev[j[0]]),
                           first_min=int(R["slot"][j[0]] * TFMIN[base] + SESS["eth"][0]))
                for nm, awk, c in BOARDS:
                    if nm not in ("real01", "ideal"):
                        continue
                    a, p, _ = comps[[b[0] for b in BOARDS].index(nm)]
                    x = p[j].mean()
                    row[f"pos_{nm}"] = float(x)
                    row[f"ex_{nm}"] = float(a[j].sum() - x * r[j].sum())
                days_rows.append(row)
                # 장 시작 전 표지: 첫 봉에 걸린 구간(flag[j0−1]) × s[j0−1]
                fz = Fz_k[:, :, j[0]]
                zpre_all.append(np.where(fz, np.where(s_prev[j[0]] == 1, 1, -1), 0).astype(np.int8))
        if (si + 1) % 24 == 0:
            print(f"{base} 이동 계산 {si + 1}/{len(data)}", flush=True)
    np.savez_compressed(OUT / f"z3diag_raw_{base}.npz", RES=RES, ROLL=ROLL, syms=np.array(syms), ks=np.array(ks),
                        cands=np.array(cands), boards=np.array([b[0] for b in BOARDS]), miss_cells=miss_cells)
    if base == "5m":
        days = pd.DataFrame(days_rows)
        Z = np.stack(zpre_all, axis=-1)  # (k, cand, 날)
        np.savez_compressed(OUT / f"z3diag_days_{base}.npz", Z=Z, ks=np.array(ks), cands=np.array(cands))
        days.to_parquet(OUT / f"z3diag_days_{base}.parquet")
    return dict(calendar=[str(cal[0]), str(cal[-1])], D=D, shifts=[ks[1], ks[-1]], n_shifts=len(ks) - 1, n_syms=len(syms),
                miss_cells_total=int(miss_cells.sum()), miss_cells_max_k=int(miss_cells.max()),
                syms_missing_bars=sorted(s for s, v in info.items() if v["missing_vs_bpd"] > 0), per_sym=info)


def run(base, nproc):
    t0 = time.time()
    with Pool(nproc) as pool:
        data = list(pool.imap_unordered(work, [(s, base) for s in SYMS], chunksize=4))
    data.sort(key=lambda R: R["sym"])
    excluded = []
    if base == "1h":  # 6/1 전 1시간봉이 없는 종목은 창이 달라 뺀다(loss_run과 같음)
        excluded = [R["sym"] for R in data if R["day"].min() > WIN[base]["sel"][0]]
        data = [R for R in data if R["sym"] not in excluded]
    cal = sorted(set().union(*[set(R["day"].tolist()) for R in data]))
    print(base, "준비", f"{time.time() - t0:.0f}s", "종목", len(data), "달력", cal[0], "~", cal[-1], len(cal), flush=True)
    meta = process(base, data, cal)
    meta.update(excluded=excluded, sec=round(time.time() - t0, 1))
    return meta


if __name__ == "__main__":
    nproc = min(int(sys.argv[1]) if len(sys.argv) > 1 else 3, 3)
    # 공통 계산 확인: MU 원안이 zones3.run_symbol·engine과 같은지
    P = prepare("MU", "5m", "dst")
    rs = run_symbol(P, 0.001, False).iloc[0]
    ri = run_symbol(P, 0.0, True).iloc[0]
    mu = dict(real=[round(rs.ret_all * 100, 1), round(rs.mdd_all * 100, 1)], ideal=[round(ri.ret_all * 100, 1), round(ri.mdd_all * 100, 1)])
    print("MU 원안 실제·이상적(수익, 낙폭 %)", mu, flush=True)
    meta = {"MU_base_check": mu}
    for b in ("5m", "1h"):
        meta[b] = run(b, nproc)
    (OUT / "z3diag_run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, default=str, indent=1), encoding="utf-8")
    for b in ("5m", "1h"):
        print(b, {k: v for k, v in meta[b].items() if k != "per_sym"})
