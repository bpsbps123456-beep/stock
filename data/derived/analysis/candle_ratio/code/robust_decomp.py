"""robust 갈래 ②③: 원안(40분 연장 N10 count follow 0.5/0.5 밤 보유)이 보유보다 덜 잃은 몫이 어디서 왔는지.

5분봉 걸음(시가→다음 시가)으로 쪼갠다. 40분 묶음봉 j의 보유가 그 안 5분 걸음 전부에 걸린다(합이 engine 봉 수익과 같다).
- 시간대 4갈래: 정규장 09:30~16:00, 프리 04:00~09:30, 애프터 16:00~20:00, 닫힌 밤 20:00→04:00(하루 마지막 종가→다음 날 첫 시가).
  '밤 갭(장 마감→다음 날 시가)' = 애프터 + 닫힌 밤 + 프리.
- 덜 잃은 몫 L = S − B(전략 − 보유, 로그) = 현금 몫 (x−1)·B + 쏠림 몫 (x_c−x)·B_c + 시간대 안 타이밍 S_c − x_c·B_c.
  E = S − x·B 는 '노출 맞춘 초과'(같은 비중 그냥 보유의 로그 근사).
- 하루(ET 20:00→20:00, 닫힌 밤은 다음 날 몫)별 E_d·L_d에서 가장 큰 3일을 뺀 나머지.
- 실적: results/robust_earnings_dates.csv의 발표일 E와 다음 거래일을 뺀다. 기간 안 발표가 없는 개별주(AEHR)는
  |밤 갭|이 가장 큰 날과 다음 날로 대신한다(ETF 8개는 빼지 않음).
- 귀무: 같은 k(15~54)로 비율을 민 판에서도 같은 분해를 낸다(시간대 쏠림이 신호 타이밍인지 하루 안 시각 구조인지 가르기).
- 보유 구간 길이·자는 동안(00~08 KST = 11:00~19:00 ET) 보유 비율, 실제 판 손해 출처(미뤄진 매수·매도).

py code/robust_decomp.py [프로세스=3] → results/robust_decomp.npy, robust_decomp_meta.json, robust_holds.csv,
                                         robust_realloss.csv, robust_topdays.csv
"""
import json
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from robust_common import COST_REAL, ORIG, RES, SYMS, bars_for, null_shifts, pos_ret, ratio, window_masks

FEAT = ["S", "B", "x"] + [f"S_{c}" for c in ("reg", "pre", "post", "closed")] + \
       [f"B_{c}" for c in ("reg", "pre", "post", "closed")] + [f"x_{c}" for c in ("reg", "pre", "post", "closed")] + \
       ["E", "L", "E_rm3", "L_rm3", "E_exearn", "L_exearn", "n_earn_days", "cost", "S_rm3s"]
ETFS = {"ARKG", "IGV", "IWM", "QQQ", "SMH", "SOXX", "SPY", "XBI"}
SLEEP = (660, 1140)  # 11:00~19:00 ET = 00~08 KST (서머타임, 창 전체가 EDT)
NAMED = ("MU", "BE", "NBIS")


def earn_days(sym, days, night_gap_by_day):
    """발표일과 다음 거래일. 없으면(개별주) |밤 갭| 최대 날과 다음 날."""
    e = pd.read_csv(RES / "robust_earnings_dates.csv")
    e = e[(e.sym == sym) & e.date.notna()]
    ud = sorted(set(days))
    out, src = set(), "earnings"
    dates = [pd.Timestamp(x).date() for x in e.date]
    dates = [x for x in dates if ud[0] <= x <= ud[-1]]
    if not dates and sym not in ETFS:
        src = "gap_fallback"
        dates = [max(night_gap_by_day, key=lambda k: abs(night_gap_by_day[k]))]
    for x in dates:
        after = [d for d in ud if d >= x]
        out.update(after[:2])  # 발표일(또는 그 뒤 첫 거래일)과 다음 거래일
    return out, src, dates


def steps(B):
    rw = B["rows"]
    O5, C5, d5, m5, gid = rw.O.values, rw.C.values, rw.d.values, rw.m.values, rw.gid.values
    n = len(O5)
    lastday = np.r_[d5[1:] != d5[:-1], True]
    final = np.zeros(n, bool); final[-1] = True
    nxt = np.r_[O5[1:], np.nan]
    intra = np.where(lastday, np.log(C5 / O5), np.log(nxt / O5))
    gap = np.where(lastday & ~final, np.log(nxt / C5), 0.0)
    gap[final] = 0.0
    comp = np.where(m5 >= 960, 2, np.where(m5 >= 570, 0, 1))
    gday = np.r_[d5[1:], d5[-1]]
    return dict(intra=intra, gap=gap, comp=comp, d=d5, gday=gday, gid=gid, m=m5, lastday=lastday & ~final,
                first=np.r_[True, gid[1:] != gid[:-1]])


def decomp(B, st, pos, r, rb, cost, earn, wmask):
    """한 창의 분해. wmask: 묶음봉 기준 창."""
    gid = st["gid"]
    w5 = wmask[gid]
    ps = pos[gid].astype(float)
    lc = np.log(1.0 - cost)
    chg = np.r_[pos[0] != 0, pos[1:] != pos[:-1]]
    c5 = np.where(st["first"], chg[gid] * lc, 0.0)  # 비용은 체결 봉 첫 걸음에
    S_c = np.zeros(4); B_c = np.zeros(4); x_c = np.zeros(4)
    for c in range(3):
        k = w5 & (st["comp"] == c)
        S_c[c] = (ps[k] * st["intra"][k]).sum() + c5[k].sum()
        B_c[c] = st["intra"][k].sum()
        x_c[c] = ps[k].mean() if k.any() else np.nan
    k = w5 & st["lastday"]
    S_c[3] = (ps[k] * st["gap"][k]).sum(); B_c[3] = st["gap"][k].sum(); x_c[3] = ps[k].mean() if k.any() else np.nan
    S, Bt = S_c.sum(), B_c.sum()
    x = pos[wmask].mean()
    # 하루별
    days = np.r_[st["d"][w5], st["gday"][k]]
    sv = np.r_[ps[w5] * st["intra"][w5] + c5[w5], ps[k] * st["gap"][k]]
    bv = np.r_[st["intra"][w5], st["gap"][k]]
    df = pd.DataFrame({"d": days, "s": sv, "b": bv}).groupby("d").sum()
    Ed = df.s - x * df.b
    Ld = df.s - df.b
    rm = df.index.isin(list(earn))
    return dict(S=S, B=Bt, x=x, S_c=S_c, B_c=B_c, x_c=x_c, E=S - x * Bt, L=S - Bt,
                E_rm3=Ed.sum() - Ed.nlargest(3).sum(), L_rm3=Ld.sum() - Ld.nlargest(3).sum(),
                E_exearn=Ed[~rm].sum(), L_exearn=Ld[~rm].sum(), n_earn=int(rm.sum()),
                cost=float(c5[w5].sum()), S_rm3s=df.s.sum() - df.s.nlargest(3).sum(),
                top3E=list(Ed.nlargest(3).items()), top3L=list(Ld.nlargest(3).items()))


def holds(B, st, pos, sym, mode):
    """보유 구간: 진입 봉 시가→청산 봉 시가. 창 끝까지 들고 있으면 censored."""
    tb = B["tb"]
    start = tb.start.values
    chg = np.diff(np.r_[0, pos, 0])
    ent = np.where(chg == 1)[0]; ex = np.where(chg == -1)[0]
    rows = []
    for a, z in zip(ent, ex):
        cens = z >= len(pos)
        zz = min(z, len(pos) - 1)
        hrs = (start[zz] - start[a]) / np.timedelta64(1, "h") + (40 / 60 if cens else 0)
        rows.append(dict(sym=sym, mode=mode, entry=str(tb.start.iloc[a]), exit=None if cens else str(tb.start.iloc[z]),
                         bars=int(z - a), hours=float(hrs), nights=int((tb.day.iloc[zz] - tb.day.iloc[a]).days if not cens else -1),
                         censored=bool(cens), seg=int(B["seg"][a])))
    gid = st["gid"]
    ps = pos[gid]
    sl = (st["m"] >= SLEEP[0]) & (st["m"] < SLEEP[1])
    inwin = B["seg"][gid] >= 0
    held = ps == 1
    sleep = dict(sym=sym, mode=mode,
                 held_share_in_sleep=float((held & sl & inwin).sum() / max((held & inwin).sum(), 1)),
                 sleep_share_of_session=float((sl & inwin).sum() / max(inwin.sum(), 1)),
                 held_while_asleep=float((held & sl & inwin).sum() / max((sl & inwin).sum(), 1)),
                 held_while_awake=float((held & ~sl & inwin).sum() / max((~sl & inwin).sum(), 1)))
    return rows, sleep


def realloss(pos_i, pos_a, rb, cost, wmask):
    """실제 − 이상적 = Σ(pos_a − pos_i)·rb + 비용. 어긋난 구간을 끝나는 방식으로 4갈래로 나눈다."""
    lc = np.log(1.0 - cost)
    chg = np.r_[pos_a[0] != 0, pos_a[1:] != pos_a[:-1]]
    out = dict(late_sell=0.0, kept_through=0.0, late_buy=0.0, missed_trip=0.0, cost=float((chg * lc)[wmask].sum()),
               n_late_sell=0, n_kept_through=0, n_late_buy=0, n_missed_trip=0)
    n = len(rb); j = 0
    while j < n:
        if pos_a[j] == pos_i[j]:
            j += 1; continue
        a = j; kind = pos_a[j] - pos_i[j]  # +1: 실제만 보유(매도 미뤄짐), −1: 이상적만 보유(매수 미뤄짐)
        while j < n and pos_a[j] - pos_i[j] == kind:
            j += 1
        val = float((kind * rb[a:j])[wmask[a:j]].sum())
        if j < n:
            real_moved = pos_a[j] != pos_a[j - 1]
        else:
            real_moved = True
        if kind == 1:
            key = "late_sell" if real_moved else "kept_through"  # 실제가 늦게 팔았나 / 이상적이 자는 새 팔았다 되샀나
        else:
            key = "late_buy" if real_moved else "missed_trip"  # 실제가 늦게 샀나 / 이상적이 자는 새 샀다 팔았나
        if wmask[a:j].any():
            out[key] += val; out["n_" + key] += 1
    return out


def work(args):
    sym, ks = args
    t0 = time.time()
    B = bars_for(sym, "5m", ORIG["tf"], "eth", 0)
    st = steps(B)
    # 검산: 걸음 합 = 봉 수익
    rb_chk = np.bincount(st["gid"], weights=st["intra"] + st["gap"], minlength=len(B["O"]))
    rb = np.r_[np.log(B["O"][1:] / B["O"][:-1]), np.log(B["C"][-1] / B["O"][-1])]
    assert np.allclose(rb_chk, rb, atol=1e-9), sym
    rw = B["rows"]
    reg = rw[(rw.m >= 570) & (rw.m < 960)]
    ro, rc = reg.groupby("d").O.first(), reg.groupby("d").C.last()
    ngap = np.log(ro.iloc[1:].values / rc.iloc[:-1].values)  # 전날 16:00 종가 → 그날 09:30 시가
    ngap = dict(zip(ro.index[1:], ngap))
    earn, src, edates = earn_days(sym, B["day"], ngap)
    p0 = ratio(B["O"], B["C"], B["V"], ORIG["N"], ORIG["kind"])
    masks = window_masks(B["seg"])
    arr = np.full((len(ks), 2, 3, len(FEAT)), np.nan)
    extra = dict(sym=sym, earn_src=src, earn_dates=[str(x) for x in edates], earn_days=sorted(str(x) for x in earn))
    hold_rows, sleep_rows, loss_rows, top_rows = [], [], [], []
    for ki, k in enumerate(ks):
        p = np.roll(p0, k * B["bpd"]) if k else p0
        P = {}
        for mi, (mode, aw, cost) in enumerate((("ideal", B["ones"], 0.0), ("real", B["awake"], COST_REAL["eth"]))):
            pos, r, rbb = pos_ret(B["O"], B["C"], p, aw, ORIG["b"], ORIG["s"], True, cost)
            P[mode] = pos
            for wi, wm in enumerate(masks):
                z = decomp(B, st, pos, r, rbb, cost, earn, wm)
                if wi == 0:
                    assert abs(z["S"] - r[wm].sum()) < 1e-9, (sym, z["S"], r[wm].sum())
                arr[ki, mi, wi] = [z["S"], z["B"], z["x"], *z["S_c"], *z["B_c"], *z["x_c"], z["E"], z["L"],
                                   z["E_rm3"], z["L_rm3"], z["E_exearn"], z["L_exearn"], z["n_earn"], z["cost"], z["S_rm3s"]]
                if k == 0 and sym in NAMED and wi == 0:
                    for lab, lst in (("E", z["top3E"]), ("L", z["top3L"])):
                        for dd, v in lst:
                            top_rows.append(dict(sym=sym, mode=mode, kind=lab, day=str(dd), val=float(v),
                                                 is_earn=dd in earn))
            if k == 0:
                hr, sr = holds(B, st, pos, sym, mode)
                hold_rows += hr; sleep_rows.append(sr)
        if k == 0:
            for wi, wname in ((0, "full"), (2, "conf")):
                lo = realloss(P["ideal"], P["real"], rb, COST_REAL["eth"], masks[wi])
                lo.update(sym=sym, window=wname)
                loss_rows.append(lo)
    return sym, arr, extra, hold_rows, sleep_rows, loss_rows, top_rows, time.time() - t0


if __name__ == "__main__":
    nproc = min(int(sys.argv[1]) if len(sys.argv) > 1 else 3, 3)
    D, shifts = null_shifts("5m")
    ks = [0] + shifts
    out = np.full((len(SYMS), len(ks), 2, 3, len(FEAT)), np.nan)
    extras, H, SL, LO, TOP = [], [], [], [], []
    t0 = time.time()
    with Pool(nproc) as pool:
        for i, (sym, arr, ex, hr, sr, lo, tp, dt) in enumerate(pool.imap_unordered(work, [(s, ks) for s in SYMS]), 1):
            out[SYMS.index(sym)] = arr
            extras.append(ex); H += hr; SL += sr; LO += lo; TOP += tp
            if i % 12 == 0 or i == len(SYMS):
                print(f"{i}/{len(SYMS)} {sym} {dt:.1f}s 누적 {time.time() - t0:.0f}s", flush=True)
    np.save(RES / "robust_decomp.npy", out)
    (RES / "robust_decomp_meta.json").write_text(json.dumps(dict(syms=SYMS, ks=ks, feat=FEAT, modes=["ideal", "real"],
                                                                 windows=["full", "sel", "conf"], sleep_et_min=SLEEP,
                                                                 earn=sorted(extras, key=lambda e: e["sym"])),
                                                            ensure_ascii=False, indent=1), encoding="utf-8")
    hd = pd.DataFrame(H)
    hd.to_csv(RES / "robust_holds.csv", index=False)
    pd.DataFrame(SL).to_csv(RES / "robust_sleep.csv", index=False)
    pd.DataFrame(LO).to_csv(RES / "robust_realloss.csv", index=False)
    pd.DataFrame(TOP).to_csv(RES / "robust_topdays.csv", index=False)
    print("끝", round(time.time() - t0), "초", flush=True)
