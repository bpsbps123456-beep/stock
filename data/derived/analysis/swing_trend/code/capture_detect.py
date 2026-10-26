"""capture 2·3·4: 판정 28개를 사후 진실 구간(지그재그 15%, k4 × ATR20)에 대본다.

- 날 j 수익 r_j = ln(O[j+1]/O[j])(구간 마지막 날은 ln(C/O)), 보유 pos[j] = st[j−1] — swing.sim_period와 같은 체결.
- 진실 상승 구간(저점 L, 고점 H)의 예지자는 L+1일 시가에 사고 H+1일 시가에 판다(날 L+1..H). 하락 구간은 H+1..L'.
- 상승 포착 = Σ 판정 수익(상승 날) / Σ 예지자 수익, 하락 회피 = 1 − Σ 판정 손실(하락 날) / Σ 그냥 보유 손실.
  기술 = 포착 + 회피 − 1(무작위로 x만큼 들고 있으면 기대 0). 귀무: 구간 안 상태 원형 이동 k(15..D−15, 모든 종목 같은 k).
- 진입 지연 = 매수 체결일 − (L+1), 청산 지연 = 매도 체결일 − (H+1), 거래일. 하락 구간 내내 안 팔았으면 '그대로 보유'로 따로 센다.
- 헛신호 = 보유 5거래일 이하로 끝난 매수.
- 늦음의 값: 예지자를 d일 늦게 사고(또는 팔고) 잃는 몫, 판정별 구간 포착을 지연에 회귀, 갑작스러운 반전 여부별 포착.
출력: results/capture_detect*.csv, capture_delay_curve.csv, capture_delay_reg.csv, capture_abrupt.csv, capture_cases.csv
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

import swing
from capture_confirm import PERS, eligible
from capture_lib import runs, truth_segments

OUT = swing.HERE / "results"
TRUTHS = {"T15": ("pct", 0.15), "TK4": ("atr", 4.0)}
DNAMES = [swing.dname(k, p) for k, p in swing.DETECTORS]
CASES = ("MU", "BE", "NBIS")
CASE_DET = ("ZZA_k3.0", "ZZP_th0.15", "CR_n10_b0.5_s0.5", "SLP_n20")
MAXD = 10


def abrupt_flag(O, C, a20, p, t):
    """반전 첫날(p+1) 새 방향 수익 ≥ 2 × ATR 또는 새 방향 시가 갭 ≥ 1 × ATR."""
    if p + 1 >= len(C) or np.isnan(a20[p]):
        return False
    a = a20[p]
    if t == 1:
        return (C[p + 1] - C[p] >= 2 * a) or (O[p + 1] - C[p] >= a)
    return (C[p] - C[p + 1] >= 2 * a) or (C[p] - O[p + 1] >= a)


def entry_delay(pos, L, H, prev_start):
    """상승 구간(L, H)의 진입 지연. 반환 (지연 or nan, 상태 'ok'/'missed'/'held')."""
    seg = pos[L + 1:H + 1]
    hit = np.flatnonzero(seg == 1)
    if len(hit) == 0:
        return np.nan, "missed"
    j = L + 1 + hit[0]
    if j > L + 1:
        return float(j - (L + 1)), "ok"
    s = j
    while s - 1 >= 0 and pos[s - 1] == 1:
        s -= 1
    if prev_start is not None and s <= prev_start:
        return np.nan, "held"
    return float(s - (L + 1)), "ok"


def exit_delay(pos, L, H, next_end, n):
    """상승 구간(L, H)의 청산 지연. next_end = 다음 하락 구간 끝 날(L')."""
    seg = pos[L + 1:H + 1]
    hit = np.flatnonzero(seg == 1)
    if len(hit) == 0:
        return np.nan, "missed"
    j = L + 1 + hit[-1]
    if j < H:
        return float(j - H), "ok"
    e = j
    while e + 1 < n and pos[e + 1] == 1:
        e += 1
    if next_end is not None and e >= next_end:
        return np.nan, "held"
    if e + 1 >= n:
        return np.nan, "open"
    return float(e - H), "ok"


def main():
    t0 = time.time()
    D = swing.load()
    el = eligible(D)
    agg = []       # 종목 × 구간 × 진실 × 판정 합계
    segrows = []   # 구간(진실 상승) × 판정: 지연과 포착
    whip = []      # 매수 단위 헛신호
    curves = []    # 예지자 지연 곡선(구간 단위)
    nullacc = {}   # (per, truth, det) → [EU_k, ED_k], 그리고 UP, DN 합
    cases = []
    for s, g in D.items():
        pers = [p for p in PERS if s in el[p]]
        if not pers and s not in CASES:
            continue
        O, H, L, C = (g[k].values.astype(np.float64) for k in ("O", "H", "L", "C"))
        n = len(C)
        a20 = swing.atr(H, L, C, 20)
        S = swing.states(g)
        POS = {d: np.r_[0, S[d][:-1]].astype(np.int64) for d in DNAMES}
        for tn, (kind, v) in TRUTHS.items():
            zz = swing.zigzag(C, v, np.zeros(0)) if kind == "pct" else swing.zigzag(C, 0.0, v * a20)
            segs = truth_segments(C, zz[1], zz[2], zz[3])
            lab = np.zeros(n, np.int64); sid = np.full(n, -1, np.int64)
            ab = []
            for i, (p, e, t, tail) in enumerate(segs):
                lab[p + 1:e + 1] = t; sid[p + 1:e + 1] = i
                ab.append(abrupt_flag(O, C, a20, p, t))
            ab = np.array(ab, bool)
            dab = np.zeros(n, bool); dab[sid >= 0] = ab[sid[sid >= 0]]
            for per in pers:
                idx = swing.period_index(g, per)
                j0, j1 = int(idx[0]), int(idx[-1])
                Dn = j1 - j0 + 1
                r = np.empty(Dn)
                r[:-1] = np.log(O[j0 + 1:j1 + 1] / O[j0:j1])
                r[-1] = np.log(C[j1] / O[j1])
                lb = lab[j0:j1 + 1]; abd = dab[j0:j1 + 1]
                up = (lb == 1).astype(float); dn = (lb == -1).astype(float)
                ru, rd = r * up, r * dn
                UP, DN = ru.sum(), rd.sum()
                UPa, UPn = (ru * abd).sum(), (ru * ~abd).sum()
                DNa, DNn = (rd * abd).sum(), (rd * ~abd).sum()
                K = np.arange(15, Dn - 15 + 1)
                M = (np.arange(Dn)[None, :] - K[:, None]) % (Dn + 1)
                for d in DNAMES:
                    pos = POS[d][j0:j1 + 1].astype(float)
                    agg.append(dict(sym=s, per=per, truth=tn, det=d, UP=UP, DN=DN, EU=pos @ ru, ED=pos @ rd,
                                    UPa=UPa, UPn=UPn, DNa=DNa, DNn=DNn,
                                    EUa=pos @ (ru * abd), EUn=pos @ (ru * ~abd), EDa=pos @ (rd * abd), EDn=pos @ (rd * ~abd),
                                    expo=float(pos[lb != 0].mean()) if (lb != 0).any() else np.nan,
                                    ret=float(np.exp(pos @ r) - 1), bh=float(np.exp(r.sum()) - 1)))
                    seg_st = S[d][j0 - 1:j1 + 1].astype(float)
                    PK = seg_st[M]
                    key = (per, tn, d)
                    if key not in nullacc:
                        nullacc[key] = [np.zeros(len(K)), np.zeros(len(K)), 0.0, 0.0]
                    acc = nullacc[key]
                    acc[0] += PK @ ru; acc[1] += PK @ rd; acc[2] += UP; acc[3] += DN
                # 구간 단위: 진입·청산 지연, 구간 포착, 예지자 지연 곡선
                for i, (p, e, t, tail) in enumerate(segs):
                    if t != 1:
                        continue
                    Lp, Hp = p, e
                    in_entry = j0 <= Lp + 1 <= j1
                    in_exit = (not tail) and j0 <= Hp + 1 <= j1
                    full = in_entry and in_exit
                    if not (in_entry or in_exit):
                        continue
                    prev_start = segs[i - 1][0] + 1 if i > 0 else None
                    next_end = segs[i + 1][1] if i + 1 < len(segs) else None
                    abs_ = bool(ab[i]); abe = bool(ab[i + 1]) if i + 1 < len(segs) else False
                    orc = np.log(O[Hp + 1] / O[Lp + 1]) if (full and Hp + 1 < n) else np.nan
                    if full and Hp + 1 < n:
                        ecur = [np.log(O[Hp + 1] / O[min(Lp + 1 + dd, Hp + 1)]) for dd in range(MAXD + 1)]
                        lim = min(next_end + 1 if next_end is not None else n - 1, n - 1)
                        xcur = [np.log(O[min(Hp + 1 + dd, lim)] / O[Hp + 1]) for dd in range(MAXD + 1)]
                        curves.append(dict(sym=s, per=per, truth=tn, L=Lp, H=Hp, days=Hp - Lp, orc=orc, ab_start=abs_, ab_end=abe,
                                           **{f"e{dd}": ecur[dd] for dd in range(MAXD + 1)},
                                           **{f"x{dd}": xcur[dd] for dd in range(MAXD + 1)}))
                    for d in DNAMES:
                        pos = POS[d]
                        ed, es = entry_delay(pos, Lp, Hp, prev_start) if in_entry else (np.nan, "na")
                        xd, xs = exit_delay(pos, Lp, Hp, next_end, n) if in_exit else (np.nan, "na")
                        earn = float(pos[Lp + 1:Hp + 1] @ np.log(O[Lp + 2:Hp + 2] / O[Lp + 1:Hp + 1])) if (full and Hp + 1 < n) else np.nan
                        # 한 바퀴(상승 구간 + 이어진 하락 구간 끝까지): 늦은 청산의 되돌림까지 포함
                        ce = min(next_end, n - 2) if next_end is not None else None
                        earn_cyc = float(pos[Lp + 1:ce + 1] @ np.log(O[Lp + 2:ce + 2] / O[Lp + 1:ce + 1])) if (full and ce is not None and ce > Hp) else np.nan
                        segrows.append(dict(sym=s, per=per, truth=tn, det=d, L=Lp, H=Hp, ab_start=abs_, ab_end=abe,
                                            ed=ed, e_st=es, xd=xd, x_st=xs, orc=orc, earn=earn, earn_cyc=earn_cyc))
            if tn == "T15":
                for per in pers:
                    idx = swing.period_index(g, per)
                    j0, j1 = int(idx[0]), int(idx[-1])
                    for d in DNAMES:
                        st_, en_ = runs(POS[d])
                        for a, b in zip(st_, en_):
                            if j0 <= a <= j1:
                                ln_ = b - a + 1
                                cens = b >= n - 1
                                whip.append(dict(sym=s, per=per, det=d, len=ln_, cens=bool(cens)))
            # 사례 표(2026)
            if s in CASES and tn == "T15":
                cases += case_rows(s, g, O, C, a20, segs, ab, zz, S, POS)
        print(s, round(time.time() - t0, 1), flush=True) if s in ("AAPL", "MU", "SPY") else None

    A = pd.DataFrame(agg)
    A.to_csv(OUT / "capture_detect_stock.csv", index=False)
    SG = pd.DataFrame(segrows); SG.to_csv(OUT / "capture_detect_segments.csv", index=False)
    W = pd.DataFrame(whip)
    CV = pd.DataFrame(curves); CV.to_csv(OUT / "capture_delay_segments.csv", index=False)
    pd.DataFrame(cases).to_csv(OUT / "capture_cases.csv", index=False)

    # 판정 요약
    rows = []
    for (per, tn, d), a in A.groupby(["per", "truth", "det"]):
        UP, DN, EU, ED = a.UP.sum(), a.DN.sum(), a.EU.sum(), a.ED.sum()
        cap, avo = EU / UP, 1 - ED / DN
        acc = nullacc[(per, tn, d)]
        sk_null = acc[0] / acc[2] + 1 - acc[1] / acc[3] - 1
        sk = cap + avo - 1
        sg = SG[(SG.per == per) & (SG.truth == tn) & (SG.det == d)]
        e_ok = sg[sg.e_st == "ok"].ed; x_ok = sg[sg.x_st == "ok"].xd
        ne = (sg.e_st != "na").sum(); nx = (sg.x_st != "na").sum()
        w = W[(W.per == per) & (W.det == d) & ~(W.cens & (W.len <= 5))] if tn == "T15" else None
        pst = a[(a.UP > 0.05) & (a.DN < -0.05)]
        rows.append(dict(per=per, truth=tn, det=d, cap_up=cap, avoid=avo, skill=sk,
                         skill_null_med=float(np.median(sk_null)), skill_null95=float(np.quantile(sk_null, 0.95)),
                         p_skill=float((sk_null >= sk).mean()),
                         cap_up_ab=a.EUa.sum() / a.UPa.sum(), cap_up_nab=a.EUn.sum() / a.UPn.sum(),
                         avoid_ab=1 - a.EDa.sum() / a.DNa.sum(), avoid_nab=1 - a.EDn.sum() / a.DNn.sum(),
                         expo=a.expo.mean(),
                         cap_up_stock_med=float((pst.EU / pst.UP).median()), avoid_stock_med=float((1 - pst.ED / pst.DN).median()),
                         entry_med=float(e_ok.median()) if len(e_ok) else np.nan, entry_mean=float(e_ok.mean()) if len(e_ok) else np.nan,
                         entry_missed=float((sg.e_st == "missed").sum() / ne) if ne else np.nan,
                         entry_held=float((sg.e_st == "held").sum() / ne) if ne else np.nan,
                         exit_med=float(x_ok.median()) if len(x_ok) else np.nan, exit_mean=float(x_ok.mean()) if len(x_ok) else np.nan,
                         exit_held=float((sg.x_st == "held").sum() / nx) if nx else np.nan,
                         n_up_seg=int(ne),
                         whip=float((w.len <= 5).mean()) if w is not None and len(w) else np.nan,
                         entries_per_month=float(len(w) / a.sym.nunique() / (len(swing.period_index(D["SPY"], per)) / 21)) if w is not None else np.nan,
                         ret_med=float(a.ret.median()), bh_med=float(a.bh.median())))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "capture_detect.csv", index=False)

    # 늦음의 값 1: 예지자 지연 곡선(풀링, 갑작스러운 반전 여부별)
    cr = []
    for (per, tn), c in CV.groupby(["per", "truth"]):
        for sub, cc in (("all", c), ("abrupt", c[c.ab_start]), ("gradual", c[~c.ab_start])):
            if not len(cc):
                continue
            row = dict(per=per, truth=tn, group=sub, kind="entry", n=len(cc), orc_sum=cc.orc.sum())
            for dd in range(MAXD + 1):
                row[f"d{dd}"] = cc[f"e{dd}"].sum() / cc.orc.sum()
            cr.append(row)
        for sub, cc in (("all", c), ("abrupt", c[c.ab_end]), ("gradual", c[~c.ab_end])):
            if not len(cc):
                continue
            row = dict(per=per, truth=tn, group=sub, kind="exit", n=len(cc), orc_sum=cc.orc.sum())
            for dd in range(MAXD + 1):
                row[f"d{dd}"] = cc[f"x{dd}"].sum() / cc.orc.sum()
            cr.append(row)
    pd.DataFrame(cr).to_csv(OUT / "capture_delay_curve.csv", index=False)

    # 늦음의 값 2: 판정·구간 단위 포착 ~ 진입 지연 + 청산 지연(예지자 수익 가중)
    # 한 바퀴 먹은 몫(예지자 상승 수익 대비) ~ 늦은/이른 진입·청산 일수, 예지자 수익 가중. 판정 28개 × 구간 풀링.
    rg = []
    ok = SG[(SG.e_st == "ok") & (SG.x_st == "ok") & SG.orc.notna() & (SG.orc > 0.02) & SG.earn_cyc.notna()]
    for (per, tn), c in ok.groupby(["per", "truth"]):
        y = (c.earn_cyc / c.orc).clip(-3, 3).values; wgt = c.orc.values
        el_, ee_ = c.ed.clip(lower=0).values, (-c.ed).clip(lower=0).values
        xl_, xe_ = c.xd.clip(lower=0).values, (-c.xd).clip(lower=0).values
        X = np.c_[np.ones(len(c)), el_, ee_, xl_, xe_]
        Wm = np.sqrt(wgt)[:, None]
        beta, *_ = np.linalg.lstsq(X * Wm, y * Wm[:, 0], rcond=None)
        row = dict(per=per, truth=tn, n=len(c), b0=beta[0], b_entry_late=beta[1], b_entry_early=beta[2],
                   b_exit_late=beta[3], b_exit_early=beta[4], ed_med=float(np.median(c.ed)), xd_med=float(np.median(c.xd)))
        # 구간: 진입 지연 묶음별 풀링 한 바퀴 몫
        for lo_, hi_ in ((-99, 0), (1, 2), (3, 5), (6, 10), (11, 999)):
            m = (c.ed >= lo_) & (c.ed <= hi_)
            row[f"cyc_entry_{lo_}_{hi_}"] = float(c.earn_cyc[m].sum() / c.orc[m].sum()) if m.any() else np.nan
            row[f"n_entry_{lo_}_{hi_}"] = int(m.sum())
        for lo_, hi_ in ((-99, -1), (0, 2), (3, 5), (6, 10), (11, 999)):
            m = (c.xd >= lo_) & (c.xd <= hi_)
            row[f"cyc_exit_{lo_}_{hi_}"] = float(c.earn_cyc[m].sum() / c.orc[m].sum()) if m.any() else np.nan
            row[f"n_exit_{lo_}_{hi_}"] = int(m.sum())
        rg.append(row)
    pd.DataFrame(rg).to_csv(OUT / "capture_delay_reg.csv", index=False)
    print("끝", round(time.time() - t0, 1), "s")


def case_rows(s, g, O, C, a20, segs, ab, zz, S, POS):
    """MU·BE·NBIS: 2026에 걸친 진실 15% 구간마다 반전점·확정일·판정이 먹은 몫(구간 날만, 단순 %)."""
    dates = g.date.values
    n = len(C)
    a, b = swing.PERIODS["P1"][0], swing.PERIODS["P2"][1]
    piv_conf = dict(zip(zz[1].tolist(), zz[2].tolist()))
    out = []
    for i, (p, e, t, tail) in enumerate(segs):
        if dates[e] < a or dates[min(p + 1, n - 1)] > b:
            continue
        lo, hi = max(p + 1, int(np.searchsorted(dates, a))), min(e, n - 1)
        dr = np.empty(hi - lo + 1)
        for k, j in enumerate(range(lo, hi + 1)):
            dr[k] = (O[j + 1] / O[j] - 1) if j + 1 < n else (C[j] / O[j] - 1)
        row = dict(sym=s, dir="상승" if t == 1 else "하락", start=str(dates[p]), end=str(dates[e]) + (" (미확정)" if tail else ""),
                   days=int(e - p), move_close=C[e] / C[p] - 1, oracle=float(np.prod(1 + dr) - 1), abrupt=bool(ab[i]))
        cf = piv_conf.get(p, -1)
        row["conf_date"] = str(dates[cf]) if cf >= 0 else ""
        row["conf_lag"] = int(cf - p) if cf >= 0 else np.nan
        row["gone_at_conf"] = (C[cf] / C[p] - 1) / (C[e] / C[p] - 1) if cf >= 0 and cf <= e else np.nan
        for d in CASE_DET:
            pos = POS[d][lo:hi + 1]
            row[d] = float(np.prod(1 + pos * dr) - 1)
            # 판정의 방향 전환 결정일(구간 시작 극점 이후 첫 번째)
            st = S[d]
            want = 1 if t == 1 else 0
            # 구간 안 첫 '새' 전환 결정일(st[k] = want, st[k−1] ≠ want, k ∈ [p, e−1]). 없으면 극점 때 이미 그 상태였는지로 표시.
            ks = [k for k in range(max(p, 1), e) if st[k] == want and st[k - 1] != want]
            if ks:
                row[d + "_sig"] = str(dates[ks[0]])
            else:
                row[d + "_sig"] = "극점 전부터" if st[p] == want else "없음"
        out.append(row)
    return out


if __name__ == "__main__":
    main()
