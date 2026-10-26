"""hybrid 감사(2026-10-10): 빌더 코드와 독립으로 다시 계산해 숫자를 대조한다.

1) 원자료(cache/1d.parquet, 1h.parquet)로 종가·시가·rs20·160분봉 RSI·70 돌파를 직접 다시 만들어 H 표와 비교
2) 빌더의 entry_exit_picker를 쓰지 않는 독립 루프로 10개 변형을 다시 돌려 보유 목록·구간 수익 대조
3) 고른 변형(r160 N10 M25 손절15%)의 날짜 3개 표본 점검(진입 조건·청산 시점)
4) 강건성: 뒤 구간 기여 쏠림, 상위·하위 날 제거, 하루 더 늦게 체결(k-2 정보) 민감도, 손절 후 같은 시가 재매수 횟수
다시 돌리기: PYTHONIOENCODING=utf-8 py variants/verify_hybrid.py   (code 폴더에서)
"""
import os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, CODE)
sys.path.insert(0, HERE)
from harness import Harness, PERIODS          # noqa: E402
from rsi_breakout import CACHE, START          # noqa: E402
import hybrid                                   # noqa: E402  (빌더 결과 재현용으로만 import)

SL = 7
STOP = 0.15


# ------------------------------------------------------------------ 독립 지표
def wilder_rsi(c, n=14):
    c = list(map(float, c))
    out = [np.nan] * len(c)
    if len(c) <= n:
        return np.array(out)
    gains = [max(c[i] - c[i - 1], 0.0) for i in range(1, len(c))]
    losses = [max(c[i - 1] - c[i], 0.0) for i in range(1, len(c))]
    ag, al = sum(gains[:n]) / n, sum(losses[:n]) / n
    def val(ag, al):
        if al == 0:
            return 100.0
        if ag == 0:
            return 0.0
        return 100 - 100 / (1 + ag / al)
    out[n] = val(ag, al)
    for i in range(n, len(gains)):
        ag = (ag * (n - 1) + gains[i]) / n
        al = (al * (n - 1) + losses[i]) / n
        out[i + 1] = val(ag, al)
    return np.array(out)


def raw_tables(days):
    d = pd.read_parquet(os.path.join(CACHE, "1d.parquet"))
    d["day"] = pd.to_datetime(d.ts).dt.strftime("%Y-%m-%d")
    opn = d.pivot_table(index="day", columns="sym", values="Open", aggfunc="first").reindex(days)
    cls = d.pivot_table(index="day", columns="sym", values="Close", aggfunc="first").reindex(days)
    return opn, cls


def raw_160(days):
    h = pd.read_parquet(os.path.join(CACHE, "1h.parquet"))
    ts = pd.to_datetime(h.ts)
    if ts.dt.tz is None:
        raise SystemExit("1h ts has no tz")
    ts = ts.dt.tz_convert("America/New_York")
    h = h.assign(m=ts.dt.hour * 60 + ts.dt.minute, day=ts.dt.strftime("%Y-%m-%d"))
    h = h[(h.m >= 570) & (h.m < 960)]
    h = h.assign(g=np.select([h.m < 750, h.m < 870], [0, 1], 2))
    r_last, sig = {}, {}
    for s, g in h.groupby("sym"):
        g = g.sort_values("ts")
        b = g.groupby(["day", "g"], sort=True).Close.last()
        r = wilder_rsi(b.to_numpy())
        bd = b.index.get_level_values(0)
        rs = pd.Series(r, index=bd)
        r_last[s] = rs.groupby(level=0).last().reindex(days)
        prev = np.concatenate([[np.nan], r[:-1]])
        hit = (prev <= 70) & (r > 70)
        hit_days = set(np.asarray(bd)[hit])
        sig[s] = pd.Series([dd in hit_days for dd in days], index=days)
    return pd.DataFrame(r_last), pd.DataFrame(sig)


# ------------------------------------------------------------------ 독립 루프
def run_indep(H, opn, cls, R, sig, N, M, stop, lag=1, log=None):
    """lag=1: k일 시가 결정에 k-1행만 쓴다. lag=2: 하루 더 늦은 정보(k-2행)로 같은 규칙."""
    days = H.days
    syms = sorted(set(H.syms))
    k0 = next(i for i, d in enumerate(days) if d >= START)
    Rv = R.reindex(columns=syms).to_numpy()
    Sv = sig.reindex(columns=syms).fillna(False).to_numpy().astype(bool)
    Ov = opn.reindex(columns=syms).to_numpy()
    Cv = cls.reindex(columns=syms).to_numpy()
    ci = {s: i for i, s in enumerate(syms)}
    rcols = set(R.columns)
    held = {}           # s -> dict(k0, peak)
    rows = []
    for k in range(k0, len(days) - 1):
        j = k - lag
        elig = [s for s in H.eligible(days[k]) if s in ci]
        ev = np.array([Rv[j, ci[s]] for s in elig if s in rcols], float)
        ev = ev[np.isfinite(ev)]

        def rank(s):
            if s not in rcols:
                return None
            x = Rv[j, ci[s]]
            return None if not np.isfinite(x) else int((ev > x).sum()) + 1
        keep, sold = [], []
        for s, p in held.items():
            c = Cv[k - 1, ci[s]]                          # 손절은 항상 어제 종가(lag와 무관하게 k-1)
            if lag == 2:
                c = Cv[j, ci[s]]
            if j >= p["k0"] and np.isfinite(c):
                p["peak"] = max(p["peak"], c)
            r = rank(s)
            why = None
            if r is not None and r > M:
                why = "rank"
            elif stop and np.isfinite(c) and c <= (1 - STOP) * p["peak"]:
                why = "stop"
            if why:
                sold.append((s, why, r))
            else:
                keep.append(s)
        held = {s: held[s] for s in keep}
        cand = []
        for s in syms:
            if s in held or s not in elig or not np.isfinite(Ov[k, ci[s]]):
                continue
            if not Sv[j, ci[s]]:
                continue
            r = rank(s)
            if r is not None and r <= N:
                cand.append(s)
        cand.sort(key=lambda s: -(Rv[j, ci[s]] if np.isfinite(Rv[j, ci[s]]) else -1e9))
        add = cand[:max(0, SL - len(held))]
        rebuy = [s for s, why, _ in sold if s in add]
        for s in add:
            held[s] = {"k0": k, "peak": Ov[k, ci[s]]}
        hs = sorted(held)
        w = 1 / max(SL, len(hs))
        rets = {}
        for s in hs:
            a, b = Ov[k, ci[s]], Ov[k + 1, ci[s]]
            rets[s] = b / a - 1 if np.isfinite(a) and np.isfinite(b) and a > 0 else 0.0
        rows.append(dict(day=days[k], ret=sum(w * v for v in rets.values()), n=len(hs), held=",".join(hs),
                         sold=";".join(f"{s}:{why}:{r}" for s, why, r in sold), added=",".join(add),
                         rebuy=",".join(rebuy), rets=rets, w=w))
    return pd.DataFrame(rows)


def period_tot(df, a, b):
    x = df[(df.day >= a) & (df.day <= b)]
    return float((1 + x.ret).prod() - 1)


def contrib_share(df, a, b):
    x = df[(df.day >= a) & (df.day <= b)]
    val, c = 1.0, {}
    for _, r in x.iterrows():
        for s, v in r.rets.items():
            c[s] = c.get(s, 0.0) + val * r.w * v
        val *= 1 + r.ret
    gain = sum(c.values())
    top = sorted(c.items(), key=lambda kv: -kv[1])[:5]
    bot = sorted(c.items(), key=lambda kv: kv[1])[:3]
    share = sum(v for _, v in top[:3]) / gain if gain > 0 else None
    return share, [(s, round(v, 4)) for s, v in top], [(s, round(v, 4)) for s, v in bot], gain


def main():
    H = Harness()
    days = H.days
    print("== 1) 원자료 대조")
    opn, cls = raw_tables(days)
    for nm, a, b in (("opn", opn, H.opn), ("cls", cls, H.cls)):
        a = a.reindex(columns=b.columns)
        diff = (a - b).abs().to_numpy()
        both = np.isfinite(a.to_numpy()) & np.isfinite(b.to_numpy())
        mism = (np.isfinite(a.to_numpy()) != np.isfinite(b.to_numpy())).sum()
        print(f"  {nm}: max|diff| {np.nanmax(diff[both]):.2e}, NaN 위치 불일치 {mism}")
    q = cls["QQQ"]
    rs20 = (cls / cls.shift(20)).div(q / q.shift(20), axis=0) - 1
    rs20 = rs20.reindex(columns=H.syms)
    d = (rs20 - H.rs(20)).abs().to_numpy()
    print(f"  rs20: max|diff| {np.nanmax(d):.2e}, NaN 불일치 {(np.isfinite(rs20.to_numpy()) != np.isfinite(H.rs(20).to_numpy())).sum()}")
    r160, sig = raw_160(days)
    r160 = r160.reindex(columns=H.r160.columns)
    sig = sig.reindex(columns=H.sig70.columns)
    d = (r160 - H.r160).abs().to_numpy()
    print(f"  r160: max|diff| {np.nanmax(d):.2e}, NaN 불일치 {(np.isfinite(r160.to_numpy()) != np.isfinite(H.r160.to_numpy())).sum()}")
    sh = H.sig70.astype(bool).to_numpy()
    sm = sig.fillna(False).astype(bool).to_numpy()
    print(f"  sig70: 독립 {sm.sum()} 개, H {sh.sum()} 개, 불일치 {(sm != sh).sum()}")

    print("\n== 2) 독립 루프로 10개 변형 재현")
    grid = [(rn, N, M, True) for rn in ("rs20", "r160") for N in (10, 15) for M in (15, 25)]
    grid += [("rs20", 10, 15, False), ("rs20", 10, 25, False)]
    Rind = {"rs20": rs20, "r160": r160}
    Rh = {"rs20": H.rs(20), "r160": H.r160}
    out = {}
    for rn, N, M, st in grid:
        name = f"hybrid R={rn} N{N} M{M}" + (" 손절15%" if st else " 손절없음")
        mine = run_indep(H, opn, cls, Rind[rn], sig, N, M, st)
        res_b, df_b = hybrid.run_variant(H, Rh[rn], rn, N, M, st)
        bh = dict(zip(df_b.day, df_b.held))
        mism = sum(1 for dd, hh in zip(mine.day, mine.held) if bh.get(dd) != hh)
        cells = []
        for lab, a, b in PERIODS:
            cells.append(f"{lab[:2]} {period_tot(mine, a, b):+.4f}/{res_b[lab]['수익']:+.4f}")
        nrank = sum(x.count(":rank:") for x in mine.sold)
        nstop = sum(x.count(":stop:") for x in mine.sold)
        nreb = sum(1 for x in mine.rebuy if x)
        print(f"  {name:28s} 보유목록 불일치 {mism}일 | " + " | ".join(cells) +
              f" | 순위매도 {nrank} 손절 {nstop} 손절후재매수 {nreb}")
        out[name] = (mine, df_b, rn, N, M, st)

    best = "hybrid R=r160 N10 M25 손절15%"
    mine = out[best][0]
    print(f"\n== 3) 표본 점검: {best}")
    ci_days = {d: i for i, d in enumerate(days)}
    R = r160
    for dd in ("2026-06-16", "2026-08-12", "2026-10-01"):
        k = ci_days[dd]
        row = mine[mine.day == dd].iloc[0]
        elig = [s for s in H.eligible(dd) if s in R.columns]
        rk = R.iloc[k - 1].reindex(elig).dropna().rank(ascending=False, method="min")
        print(f"  [{dd}] 보유 {row.held} | 오늘 판 것 {row.sold or '-'} | 오늘 산 것 {row.added or '-'}")
        for s in row.held.split(","):
            if not s:
                continue
            # 이 보유가 시작된 날 찾기
            i = mine.index[mine.day == dd][0]
            while i > 0 and s in mine.loc[i - 1, "held"].split(","):
                i -= 1
            e = mine.loc[i, "day"]
            ke = ci_days[e]
            elig_e = [x for x in H.eligible(e) if x in R.columns]
            rke = R.iloc[ke - 1].reindex(elig_e).dropna().rank(ascending=False, method="min")
            print(f"    {s}: 산 날 {e} 시가 {opn[s].iloc[ke]:.2f} · 전날({days[ke-1]}) sig70 {bool(sig[s].iloc[ke-1])} · "
                  f"r160 {R[s].iloc[ke-1]:.1f} 순위 {int(rke.get(s, -1))}/{len(rke)} (N=10) | 오늘 전날 순위 "
                  f"{int(rk.get(s, -1)) if s in rk else 'NA'} (M=25)")
        # 다음 날 청산된 것
        nxt = mine[mine.day > dd].iloc[0]
        if nxt.sold:
            kn = ci_days[nxt.day]
            print(f"    다음 거래일 {nxt.day} 시가 매도: {nxt.sold} (근거 행 {days[kn-1]})")
            for item in nxt.sold.split(";"):
                s, why, r = item.split(":")
                c_prev = cls[s].iloc[kn - 1]
                print(f"      {s} 사유 {why} 순위 {r} · {days[kn-1]} 종가 {c_prev:.2f} · 다음 시가 {opn[s].iloc[kn]:.2f}")

    print("\n== 4) 강건성")
    for name in (best, "hybrid R=r160 N15 M25 손절15%", "hybrid R=r160 N10 M15 손절15%",
                 "hybrid R=rs20 N15 M25 손절15%", "hybrid R=rs20 N10 M25 손절15%"):
        m = out[name][0]
        a, b = "2026-06-01", "2026-10-09"
        share, top, bot, gain = contrib_share(m, a, b)
        x = m[(m.day >= a) & (m.day <= b)].ret.sort_values()
        tot = (1 + x).prod() - 1
        no_best2 = (1 + x.iloc[:-2]).prod() - 1
        no_worst2 = (1 + x.iloc[2:]).prod() - 1
        print(f"  {name}: 뒤 {tot:+.4f} · 기여합 {gain:+.4f} · 상위3몫 {share if share is None else round(share, 2)} · "
              f"상위5 {top} · 하위3 {bot} · 최고 2일 빼면 {no_best2:+.4f} · 최악 2일 빼면 {no_worst2:+.4f}")
    print("  -- 하루 더 늦은 정보(k-2행)로 같은 규칙")
    for rn, N, M, st in grid:
        name = f"hybrid R={rn} N{N} M{M}" + (" 손절15%" if st else " 손절없음")
        m2 = run_indep(H, opn, cls, Rind[rn], sig, N, M, st, lag=2)
        print(f"  {name:28s} " + " | ".join(f"{lab[:2]} {period_tot(m2, a, b):+.4f}" for lab, a, b in PERIODS))


def extra():
    """뒤 구간 최고·최악 날, 대상 이탈 보유일, 10월 보유 — 빌더 서술 대조."""
    H = Harness()
    days = H.days
    opn, cls = raw_tables(days)
    q = cls["QQQ"]
    rs20 = ((cls / cls.shift(20)).div(q / q.shift(20), axis=0) - 1).reindex(columns=H.syms)
    r160, sig = raw_160(days)
    cases = [("r160", 10, 25, True), ("r160", 15, 25, True), ("rs20", 15, 25, True), ("rs20", 10, 25, True)]
    for rn, N, M, st in cases:
        m = run_indep(H, opn, cls, {"rs20": rs20, "r160": r160}[rn], sig, N, M, st)
        out_elig = sum(1 for _, r in m.iterrows() for s in r.held.split(",") if s and s not in H.eligible(r.day))
        held_days = sum(r.n for _, r in m.iterrows())
        x = m[(m.day >= "2026-06-01") & (m.day <= "2026-10-09")]
        top = x.sort_values("ret").tail(3)
        bot = x.sort_values("ret").head(3)
        print(f"\n[{rn} N{N} M{M} 손절{st}] 대상 이탈 보유일 {out_elig} / 보유일 {held_days}")
        for lab, t in (("뒤 최고 3일", top), ("뒤 최악 3일", bot)):
            for _, r in t.iterrows():
                best = sorted(r.rets.items(), key=lambda kv: -abs(kv[1]))[:3]
                print(f"  {lab} {r.day} {r.ret:+.4f} QQQ {opn['QQQ'].shift(-1)[r.day] / opn['QQQ'][r.day] - 1:+.4f} "
                      f"보유 {r.held} 큰 움직임 {[(s, round(v, 3)) for s, v in best]}")
        octo = m[m.day >= "2026-10-01"]
        print("  10월 보유:", sorted({s for h in octo.held for s in h.split(",") if s}))
        print("  10월 일별:", [(r.day, round(r.ret, 4), r.held) for _, r in octo.iterrows()])


if __name__ == "__main__":
    extra() if sys.argv[1:] == ["extra"] else main()
