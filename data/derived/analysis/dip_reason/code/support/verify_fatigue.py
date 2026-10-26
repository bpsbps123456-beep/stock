"""반박 검증(2026-10-09): support/fatigue.py의 best_rule.definition만 보고 독립 재구현한다(코드 복사 없음).

다시 만드는 것
  1) 최선안 '버팀240분·생존·매도 1일'(40분봉 지지 구역, 정규장만) 두 기간 재현 + 첫닿음·무작위·같은 에피소드 첫닿음 비교
  2) 이웃: 버팀 120~480분 × 띠(생존/1%안) × 장벽(깨짐 1.5·2·2.5%, 돌아섬 2·3·4%)에서 '무작위 대비 초과' 부호
  3) 쏠림: 초과 수익 합의 종목·날짜 상위 비중, 상위 1개 빼고 부호
  4) 단조성: 버틴 시간 칸별 위비중을 진짜 구역 vs 가짜 구역(진짜 구역을 ±0.6% 옮긴 것)으로
  5) 피로(k번째 두드림) 표: 원래 결판(첫 닿음부터 3거래일 시한) vs k번째 닿음부터 새 3거래일 시한,
     진짜 구역 vs 가짜 구역. 시한이 짧아지면 가까운 아래 장벽이 먼저 걸리는 '잘림 편향'인지 본다.
  6) k4+ 위비중: 진짜 구역 vs 가짜 구역 6개(±0.3·0.6·0.9%) 평균, 종목 군집 부트스트랩
40분봉: 5분봉(프리·애프터 포함)을 그날 04:00 ET부터 40분 칸으로 groupby(리샘플 안 씀).
사용: py support/verify_fatigue.py (code 폴더) → ../results/support/verify_fatigue.txt
"""
import io, os, sys, time
from bisect import bisect_left
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, CODE)
from stop_lab import load, RAW, PERIODS, USER78, ALL

OUT = os.path.normpath(os.path.join(CODE, "../results/support/verify_fatigue.txt"))
MIN = 60 * 10**9
HOR = 1170
TMS = (30, 60, 120, 180, 240, 300, 360, 480, 960)
TBINS = (0, 15, 30, 60, 120, 240, 390, 780)


# ------------------------------------------------------------ 구역
def bars40(sym):
    m = pd.read_parquet(os.path.join(RAW, f"5m_{sym}.parquet")).dropna(subset=["High", "Low", "Close"])
    t = m.index
    a4 = t.normalize() + pd.Timedelta(hours=4)
    st = a4 + ((t - a4) // pd.Timedelta(minutes=40)) * pd.Timedelta(minutes=40)
    g = pd.DataFrame({"s": st.as_unit("ns").asi8, "h": m.High.values.astype(float), "l": m.Low.values.astype(float)})
    g = g.groupby("s", sort=True).agg(h=("h", "max"), l=("l", "min"))
    g = g[g.h > 0]
    return g.index.values.astype(np.int64), g.h.values, g.l.values


def build_zones(sw):
    order = sorted(range(len(sw)), key=lambda k: sw[k][0])
    cl = []
    for k in order:
        p, b = sw[k]
        if cl and (p - cl[-1][0]) / cl[-1][0] <= 0.005:
            cl[-1][1] = p; cl[-1][2] += 1; cl[-1][3] = max(cl[-1][3], b)
        else:
            cl.append([p, p, 1, b])
    cl.sort(key=lambda z: (z[2], z[3]), reverse=True)
    kept = []
    for z in cl:
        mz = (z[0] + z[1]) / 2
        if all(abs(mz - (y[0] + y[1]) / 2) / ((y[0] + y[1]) / 2) >= 0.012 for y in kept):
            kept.append(z)
    kept.sort(key=lambda z: z[1])
    return np.array([z[0] for z in kept]), np.array([z[1] for z in kept])


def zone_timeline(sym):
    ts, h, l = bars40(sym)
    n = len(h)
    piv = []                                     # (확정 시각, 가격, 봉)
    for i in range(5, n - 5):
        conf = ts[i + 5] + 40 * MIN
        if l[i] <= l[i - 5:i + 6].min() and (l[i - 5:i] > l[i]).all():
            piv.append((conf, l[i], i))
        if h[i] >= h[i - 5:i + 6].max() and (h[i - 5:i] < h[i]).all():
            piv.append((conf, h[i], i))
    times, sets = [], []
    k = 0
    while k < len(piv):
        t0 = piv[k][0]
        while k < len(piv) and piv[k][0] == t0:
            k += 1
        sw = [(p, b) for (_, p, b) in piv[max(0, k - 60):k]]
        times.append(t0); sets.append(build_zones(sw))
    return np.array(times, dtype=np.int64), sets


# ------------------------------------------------------------ 에피소드(정규장 봉만)
def run_sym(A, zt, zs, period, dn, up, shift=0.0, fresh=False):
    """반환: trades[(rule, e, fill, eid)], eps[(eid, p0, out, kmax)], snaps[(eid, bin)], ktab[(eid, k, out_orig, out_fresh)]"""
    R = np.flatnonzero(A["reg"])
    o, h, l, c = A["o"][R], A["h"][R], A["l"][R], A["c"][R]
    tk, mod, day = A["tk"][R], A["mod"][R], A["day"][R]
    ud = {d: i for i, d in enumerate(sorted(set(day)))}
    clk = np.array([ud[d] for d in day], dtype=np.int64) * 390 + (mod - 570)
    mins = A["mins"]
    lo_d, hi_d = PERIODS[period][3], PERIODS[period][4]
    inp = np.array([lo_d <= d <= hi_d for d in day])
    zi = np.searchsorted(zt, tk, side="right") - 1
    n = len(R)

    def fresh_out(p, zlo, zhi):
        """p 봉(닿는 봉)부터 새 1170분 시한으로 결판. 닿는 봉은 깨짐만."""
        if l[p] <= zlo * (1 - dn):
            return -1
        q = p + 1
        while q < n and clk[q] - clk[p] < HOR:
            b, u = l[q] <= zlo * (1 - dn), h[q] >= zhi * (1 + up)
            if b:
                return -1
            if u:
                return 1
            q += 1
        return 0

    trades, eps, snaps, ktab = [], [], [], []
    act = False
    for p in range(1, n):
        if not act:
            if not inp[p] or zi[p] < 0:
                continue
            los, his = zs[zi[p]]
            los, his = los * (1 + shift), his * (1 + shift)
            q = bisect_left(his, c[p - 1]) - 1
            if q < 0 or l[p] > his[q]:
                continue
            zlo, zhi = los[q], his[q]
            eid = len(eps); p0 = p; c0 = clk[p]; k = 1; bnc = False; fired = set(); nb = 0
            ktouch = [(1, p)]
            trades.append(("첫닿음", R[p], min(o[p], zhi), eid))
            if l[p] <= zlo * (1 - dn):
                eps.append((eid, p0, -1, 1)); _k(ktab, eid, ktouch, -1, fresh, fresh_out, zlo, zhi)
                continue
            bnc = c[p] > zhi * 1.01
            act = True
            el = clk[p] + mins - c0
            while nb < len(TBINS) and TBINS[nb] <= el:
                snaps.append((eid, nb)); nb += 1
            continue
        if clk[p] - c0 >= HOR:
            eps.append((eid, p0, 0, k)); _k(ktab, eid, ktouch, 0, fresh, fresh_out, zlo, zhi); act = False
            continue
        if bnc and l[p] <= zhi:
            k += 1; bnc = False; ktouch.append((k, p))
            trades.append((f"k{k}", R[p], min(o[p], zhi), eid))
        b, u = l[p] <= zlo * (1 - dn), h[p] >= zhi * (1 + up)
        if b or u:
            out = -1 if b else 1
            eps.append((eid, p0, out, k)); _k(ktab, eid, ktouch, out, fresh, fresh_out, zlo, zhi); act = False
            continue
        if c[p] > zhi * 1.01:
            bnc = True
        el = clk[p] + mins - c0
        while nb < len(TBINS) and TBINS[nb] <= el:
            snaps.append((eid, nb)); nb += 1
        rel = c[p] / zhi - 1
        for tm in TMS:
            for band in ("생존", "1%안"):
                key = (tm, band)
                if key in fired or el < tm or (band == "1%안" and rel > 0.01):
                    continue
                fired.add(key)
                if p + 1 < n:
                    trades.append((f"버팀{tm}·{band}", R[p + 1], o[p + 1], eid))
    if act:
        eps.append((eid, p0, 2, k)); _k(ktab, eid, ktouch, 2, fresh, fresh_out, zlo, zhi)
    return trades, eps, snaps, ktab


def _k(ktab, eid, ktouch, out, fresh, fresh_out, zlo, zhi):
    for (kk, p) in ktouch:
        ktab.append((eid, kk, out, fresh_out(p, zlo, zhi) if fresh else np.nan))


# ------------------------------------------------------------ 매도·무작위
def exits(A):
    n = len(A["o"])
    ud = {d: i for i, d in enumerate(sorted(set(A["day"])))}
    di = np.array([ud[d] for d in A["day"]], dtype=np.int64)
    key = di * 1440 + A["mod"]
    J1 = np.searchsorted(key, (di + 1) * 1440 + A["mod"], side="left")
    J3 = np.searchsorted(key, (di + 3) * 1440 + A["mod"], side="left")
    hot = np.flatnonzero(A["r"] > 70)
    e = np.arange(n)
    qq = np.searchsorted(hot, e, side="left")
    jr = np.where(qq < len(hot), hot[np.minimum(qq, len(hot) - 1)] + 1, n)
    jd = np.searchsorted(A["tk"], A["dead"], side="left")
    return {"1일": J1, "3일": J3, "RSI/8시": np.minimum(jr, jd)}


def run_period(period, S, Z, dn=0.02, up=0.03, shift=0.0, fresh=False, syms=None):
    lo_d, hi_d = PERIODS[period][3], PERIODS[period][4]
    TR, EP, SN, KT = [], [], [], []
    for s, A in S.items():
        if syms is not None and s not in syms:
            continue
        if s not in Z:
            continue
        zt, zs = Z[s]
        if not len(zt):
            continue
        trades, eps, snaps, ktab = run_sym(A, zt, zs, period, dn, up, shift, fresh)
        n = len(A["o"]); o = A["o"]
        J = exits(A)
        hour = A["mod"] // 60
        inday = np.array([lo_d <= d <= hi_d for d in A["day"]])
        pool = A["reg"] & inday
        rnd = {}
        for x, Jx in J.items():
            ok = pool & (Jx < n)
            rr = o[np.minimum(Jx, n - 1)] / o - 1
            rnd[x] = pd.Series(rr[ok]).groupby(hour[ok]).mean().to_dict()
        for (rule, e, fill, eid) in trades:
            if not inday[e]:
                continue
            for x, Jx in J.items():
                if Jx[e] < n:
                    TR.append((period, s, rule, x, A["day"][e], eid, o[Jx[e]] / fill - 1, rnd[x].get(hour[e], np.nan)))
        for (eid, p0, out, km) in eps:
            EP.append((period, s, eid, out, km))
        for (eid, nb) in snaps:
            SN.append((period, s, eid, nb))
        for (eid, kk, out, fo) in ktab:
            KT.append((period, s, eid, kk, out, fo))
    TR = pd.DataFrame(TR, columns=["period", "sym", "rule", "exit", "day", "eid", "ret", "rnd"])
    EP = pd.DataFrame(EP, columns=["period", "sym", "eid", "out", "kmax"])
    SN = pd.DataFrame(SN, columns=["period", "sym", "eid", "bin"]).merge(EP[["sym", "eid", "out"]], on=["sym", "eid"])
    KT = pd.DataFrame(KT, columns=["period", "sym", "eid", "k", "out", "fout"])
    return TR, EP, SN, KT


def tday(v, d):
    s = pd.Series(np.asarray(v)).groupby(np.asarray(d)).mean()
    if len(s) < 3 or s.std(ddof=1) == 0:
        return np.nan
    return s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))


def cellf(t):
    if not len(t):
        return "n=0"
    ex = t.ret - t.rnd
    return (f"n={len(t):4d} 평균 {t.ret.mean():+.2%} 승률 {(t.ret > 0).mean():.0%} 최악 {t.ret.min():+.1%} 합 {t.ret.sum():+.2f} "
            f"t {tday(t.ret, t.day):+.2f} | 무작위 {t.rnd.mean():+.2%} 초과 {ex.mean():+.2%}(t {tday(ex, t.day):+.2f})")


def ups(out):
    r = out[np.isin(out, [1, -1])]
    return ((r == 1).mean() if len(r) else np.nan), len(r)


def main():
    t0 = time.time()
    buf = io.StringIO(); W = buf.write
    W(__doc__.split("사용:")[0] + "\n")
    Z = {}
    for s in ALL:
        if os.path.exists(os.path.join(RAW, f"5m_{s}.parquet")):
            Z[s] = zone_timeline(s)
    print("구역", round(time.time() - t0), "초", flush=True)
    SS = {p: load(p) for p in ("early", "late")}
    U78 = USER78

    # ---------- 1) 재현
    base = {p: run_period(p, SS[p], Z, fresh=True) for p in ("early", "late")}
    print("기본", round(time.time() - t0), "초", flush=True)
    W("=== 1) 최선안 재현: 버팀240·생존 · 매도 1일 · 정규장만 ===\n")
    for p in ("early", "late"):
        TR = base[p][0]
        for un, us in (("사용자78", U78), ("전체", set(ALL))):
            T = TR[(TR.exit == "1일") & TR.sym.isin(us)]
            r, f = T[T.rule == "버팀240·생존"], T[T.rule == "첫닿음"]
            same = f[f.set_index(["sym", "eid"]).index.isin(r.set_index(["sym", "eid"]).index)]
            W(f"  {p} {un}: 규칙 {cellf(r)}\n      첫닿음 {cellf(f)}\n      같은 에피소드 첫닿음 평균 {same.ret.mean():+.2%} (n {len(same)})\n")
    W("\n  -- 버팀 시간별 진입 평균(정규장·사용자78) [평균/건수, 무작위 대비 초과]\n")
    for x in ("RSI/8시", "1일", "3일"):
        for p in ("early", "late"):
            T = base[p][0]; T = T[(T.exit == x) & T.sym.isin(U78)]
            f = T[T.rule == "첫닿음"]
            row = [f"첫닿음 {f.ret.mean():+.2%}/{len(f)}[{(f.ret - f.rnd).mean():+.2%}]"]
            for tm in TMS:
                r = T[T.rule == f"버팀{tm}·생존"]
                row.append(f"{tm} {r.ret.mean():+.2%}/{len(r)}[{(r.ret - r.rnd).mean():+.2%}]")
            for kk in (2, 3):
                r = T[T.rule == f"k{kk}"]
                row.append(f"k{kk} {r.ret.mean():+.2%}/{len(r)}")
            W(f"  {x:<6} {p:<5} " + " | ".join(row) + "\n")

    # ---------- 2) 이웃
    W("\n=== 2) 이웃 매개변수(정규장·사용자78·매도 1일): 평균 / 무작위 대비 초과 / 첫닿음 대비 ===\n")
    nb_rows = []
    for (dn, up) in ((0.02, 0.03), (0.015, 0.03), (0.025, 0.03), (0.02, 0.02), (0.02, 0.04)):
        for p in ("early", "late"):
            TR = base[p][0] if (dn, up) == (0.02, 0.03) else run_period(p, SS[p], Z, dn, up, syms=U78)[0]
            T = TR[(TR.exit == "1일") & TR.sym.isin(U78)]
            fa = T[T.rule == "첫닿음"].ret.mean()
            for tm in (120, 180, 240, 300, 360, 480):
                for band in ("생존", "1%안"):
                    r = T[T.rule == f"버팀{tm}·{band}"]
                    nb_rows.append(dict(dn=dn, up=up, period=p, tm=tm, band=band, n=len(r), avg=r.ret.mean(),
                                        ex=(r.ret - r.rnd).mean(), vs_first=r.ret.mean() - fa))
    NB = pd.DataFrame(nb_rows)
    pv = NB.pivot_table(index=["dn", "up", "tm", "band"], columns="period", values=["n", "ex", "vs_first"])
    W(pv.to_string(float_format=lambda v: f"{v:+.4f}" if abs(v) < 5 else f"{v:.0f}") + "\n")
    both_ex = ((pv[("ex", "early")] > 0) & (pv[("ex", "late")] > 0)).sum()
    both_f = ((pv[("vs_first", "early")] > 0) & (pv[("vs_first", "late")] > 0)).sum()
    W(f"  이웃 {len(pv)}개 중 두 기간 모두 무작위 초과 > 0: {both_ex}, 두 기간 모두 첫닿음보다 나음: {both_f}\n")

    # ---------- 3) 쏠림
    W("\n=== 3) 쏠림(최선안·사용자78·매도 1일, 무작위 대비 초과 합) ===\n")
    for p in ("early", "late"):
        T = base[p][0]; T = T[(T.exit == "1일") & T.sym.isin(U78) & (T.rule == "버팀240·생존")].copy()
        T["ex"] = T.ret - T.rnd
        tot = T.ex.sum()
        bs, bd = T.groupby("sym").ex.sum().sort_values(), T.groupby("day").ex.sum().sort_values()
        W(f"  {p}: 초과 합 {tot:+.2f} (n {len(T)}, 종목 {T.sym.nunique()}, 날짜 {T.day.nunique()})\n"
          f"     상위 종목 {', '.join(f'{k} {v:+.2f}' for k, v in bs[::-1][:3].items())} / 하위 {', '.join(f'{k} {v:+.2f}' for k, v in bs[:3].items())}\n"
          f"     상위 날짜 {', '.join(f'{k} {v:+.2f}' for k, v in bd[::-1][:3].items())} / 하위 {', '.join(f'{k} {v:+.2f}' for k, v in bd[:3].items())}\n"
          f"     상위 종목 1개 빼면 {tot - bs.iloc[-1]:+.2f}, 상위 날짜 1개 빼면 {tot - bd.iloc[-1]:+.2f}, 상위 날짜 3개 빼면 {tot - bd.iloc[-3:].sum():+.2f}\n"
          f"     날짜별 초과 평균 > 0 인 날 비율 {(T.groupby('day').ex.mean() > 0).mean():.0%}\n")

    # ---------- 4) 단조성: 진짜 vs 가짜 구역
    W("\n=== 4) 버틴 시간 칸별 위비중(살아남은 에피소드 최종 위/(위+아래)) 진짜 / 가짜(-0.6%) / 가짜(+0.6%) · 정규장·전체 ===\n")
    fake = {(p, sh): run_period(p, SS[p], Z, shift=sh, fresh=True) for p in ("early", "late") for sh in (-0.006, 0.006)}
    print("가짜", round(time.time() - t0), "초", flush=True)
    for p in ("early", "late"):
        row = []
        for nm, (TR, EP, SN, KT) in (("진짜", base[p]), ("가짜-", fake[(p, -0.006)]), ("가짜+", fake[(p, 0.006)])):
            u0, n0 = ups(EP.out.values)
            cells = [f"닿음 {u0:.0%}({n0})"]
            for b, T in enumerate(TBINS):
                u, nn = ups(SN[SN.bin == b].out.values)
                cells.append(f"{T}분 {u:.0%}({nn})")
            row.append(f"   {nm:<4} " + " → ".join(cells))
        W(f"  {p}\n" + "\n".join(row) + "\n")
    # ---------- 5) 피로 표
    W("\n=== 5) k번째 두드림 위비중: 원래 결판(첫 닿음부터 3거래일) / 새 시한(k번째 닿음부터 3거래일) · 미결 비율 · 정규장·전체 ===\n")
    for p in ("early", "late"):
        for nm, KT in (("진짜", base[p][3]), ("가짜-", fake[(p, -0.006)][3]), ("가짜+", fake[(p, 0.006)][3])):
            cells = []
            for kk in (1, 2, 3, 4, 5):
                x = KT[KT.k == kk] if kk < 5 else KT[KT.k >= 5]
                if not len(x):
                    continue
                u, nu = ups(x.out.values); uf, nf = ups(x.fout.values)
                cells.append(f"k{kk}{'+' if kk == 5 else ''} {u:.0%}/{uf:.0%} 미결 {np.isin(x.out, [0, 2]).mean():.0%}/{(x.fout == 0).mean():.0%} (n {len(x)})")
            W(f"  {p} {nm:<4} " + " | ".join(cells) + "\n")
    # ---------- 6) 가짜 구역 6개(±0.3·0.6·0.9%)와 k4+ 차이, 종목 군집 부트스트랩
    W("\n=== 6) k4+ 위비중 진짜 vs 가짜 구역 6개 평균 (원래 결판, 정규장·전체), 종목 군집 부트스트랩 95% ===\n")
    rng = np.random.default_rng(0)
    for p in ("early", "late"):
        K = {0.0: base[p][3]}
        for sh in (-0.009, -0.006, -0.003, 0.003, 0.006, 0.009):
            K[sh] = fake[(p, sh)][3] if (p, sh) in fake else run_period(p, SS[p], Z, shift=sh)[3]
        line = []
        for sh, KT in K.items():
            r = KT[KT.out.isin([1, -1])]
            line.append(f"{sh:+.3f}: " + " ".join(f"k{a}{'+' if b > 9 else ''} {(r[(r.k >= a) & (r.k <= b)].out == 1).mean():.0%}"
                                                 for a, b in ((1, 1), (2, 2), (3, 3), (4, 99))))
        W(f"  {p}\n    " + "\n    ".join(line) + "\n")
        r0 = K[0.0]; r0 = r0[r0.out.isin([1, -1]) & (r0.k >= 4)]
        pl = pd.concat([v for s_, v in K.items() if s_ != 0.0]); pl = pl[pl.out.isin([1, -1]) & (pl.k >= 4)]
        syms = sorted(set(r0.sym) | set(pl.sym))
        a0 = r0.groupby("sym").out.apply(lambda v: (v == 1).sum()).reindex(syms, fill_value=0).values
        n0 = r0.groupby("sym").size().reindex(syms, fill_value=0).values
        a1 = pl.groupby("sym").out.apply(lambda v: (v == 1).sum()).reindex(syms, fill_value=0).values
        n1 = pl.groupby("sym").size().reindex(syms, fill_value=0).values
        d = []
        for _ in range(2000):
            ix = rng.integers(0, len(syms), len(syms))
            d.append(a0[ix].sum() / max(n0[ix].sum(), 1) - a1[ix].sum() / max(n1[ix].sum(), 1))
        W(f"    k4+ 진짜 {(r0.out == 1).mean():.1%}(n {len(r0)}) vs 가짜 평균 {(pl.out == 1).mean():.1%} → 차이 95% "
          f"[{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]\n")
    W(f"\n계산 {time.time() - t0:.0f}초\n")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w", encoding="utf-8").write(buf.getvalue())
    print(buf.getvalue())


if __name__ == "__main__":
    main()
