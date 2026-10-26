"""반박 검증 — pat_peers.py 후보(동종 동반 이격 ptouch, 혼자 빠짐 sdev_q, rel_dev, 사건 E rel_hi·phi·pdev) (2026-10-10).

기본값은 '살아남지 못함'. 확인 순서:
  0) 분석가 숫자 재현(pat_peers_trades.csv / pat_peers_events.csv, 같은 문턱).
  1) 미래 참조: ptouch를 1분봉에서 시간 창으로 독립 재계산(동종 3분봉 시작이 [t-30분, t) 안, 봉 i 미포함).
     분석가 정의는 '동종 자기 봉 10개'(봉 개수 창)라 프리장처럼 봉이 빈 동종은 30분보다 먼 과거까지 본다.
     대조로 봉 i(같은 시각 동종 봉)를 넣은 '미래 참조' 판도 만들어 효과가 그쪽에서만 큰지 본다.
  2) 시간대 교란: 날짜x시각칸(개장 뒤 3분 칸 10개 + 10시 뒤 1시간 칸) 고정효과, 칸 안 순열 2000회.
  3) 쏠림: Y2=1 거래 목록, 상위 1·2일/종목 빼기, 거래 단위 정확검정(초기하).
  4) 선택편향: 분석가 절차(특성 12 x 분할 2 x 결과 2 → 앞 |t| 상위 3 → 뒤)를
     (a) 무작위 정규 특성 300회, (b) 실제 특성 행을 날짜x시각4칸 안에서 통째로 섞기 300회로 돌려 '고른 3개 중 최대 뒤 t'와 비교.
  5) 사건 E: 독립 재구현으로 사건 목록 대조, rel_hi·phi·pdev 앞 문턱 → 뒤.
  6) 7/14~9/8 5분봉(stop_lab early) 시간 창 ptouch로 방향 대조.
  7) 10/9 사례: 지표 상태 기계를 10/9 10:18까지 돌려 실제 정규장 매수와 ptouch 값.
출력: ../results/verify_pat_peers.txt
사용: py verify_pat_peers.py"""
import os, sys
from datetime import date
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "../../dip_reason/code"))
sys.path.insert(0, HERE)
from events import bars, prep
import pat_peers as PP

RAW, RAW_OLD, RES = PP.RAW, PP.RAW_OLD, PP.RES
MID = date(2026, 9, 24)
LEV = PP.LEV
RNG = np.random.default_rng(1010)
L = []
M30 = 30 * 60 * 10 ** 9
M3 = 3 * 60 * 10 ** 9


def say(*a):
    s = " ".join(str(x) for x in a)
    L.append(s)
    print(s)


def pct(x):
    return "nan" if x is None or not np.isfinite(x) else f"{100 * x:+.2f}%"


# ---------- 통계(독립 구현) ----------
def dayt(day, y, a):
    """날짜마다 mean(A)-mean(B), 두 쪽 다 있는 날만. (평균, t, 날짜 수)"""
    ok = np.isfinite(y)
    day, y, a = day[ok], y[ok], a[ok].astype(bool)
    ds = []
    for v in np.unique(day):
        k = day == v
        ya, yb = y[k & a], y[k & ~a]
        if len(ya) and len(yb):
            ds.append(ya.mean() - yb.mean())
    ds = np.array(ds)
    if len(ds) < 3 or ds.std(ddof=1) == 0:
        return (ds.mean() if len(ds) else np.nan), np.nan, len(ds)
    return ds.mean(), ds.mean() / (ds.std(ddof=1) / np.sqrt(len(ds))), len(ds)


def dayt_fast(dcode, nd, y, a):
    sA = np.bincount(dcode[a], y[a], nd); cA = np.bincount(dcode[a], None, nd)
    sB = np.bincount(dcode[~a], y[~a], nd); cB = np.bincount(dcode[~a], None, nd)
    k = (cA > 0) & (cB > 0)
    d = sA[k] / cA[k] - sB[k] / cB[k]
    if len(d) < 3:
        return np.nan, np.nan
    sd = d.std(ddof=1)
    return d.mean(), (d.mean() / (sd / np.sqrt(len(d))) if sd > 0 else np.nan)


def fe_ols(y, X, grp, clu):
    """grp 고정효과(내포된 칸) 빼고 OLS, clu 군집 표준오차. X 첫 열 계수와 t."""
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    y, X, grp, clu = y[ok], X[ok], grp[ok], clu[ok]
    df = pd.DataFrame(np.column_stack([y, X]))
    dm = df - df.groupby(grp).transform("mean")
    yy, XX = dm.values[:, 0], dm.values[:, 1:]
    keep = np.abs(XX).sum(0) > 1e-12
    XX = XX[:, keep]
    inv = np.linalg.pinv(XX.T @ XX)
    b = inv @ XX.T @ yy
    e = yy - XX @ b
    meat = np.zeros((XX.shape[1],) * 2)
    for v in np.unique(clu):
        k = clu == v
        u = XX[k].T @ e[k]
        meat += np.outer(u, u)
    G = len(np.unique(clu))
    V = inv @ meat @ inv * G / max(G - 1, 1)
    return b[0], b[0] / np.sqrt(V[0, 0])


def perm_within(cell, a, rng):
    idx_sorted = np.argsort(cell, kind="stable")
    order = np.lexsort((rng.random(len(cell)), cell))
    out = np.empty_like(a)
    out[idx_sorted] = a[order]
    return out


def tslot(mod):
    """개장 뒤 3분 칸 0~9(09:30~09:57), 10시 뒤 1시간 칸 10~15."""
    return np.where(mod < 600, (mod - 570) // 3, 10 + (mod - 600) // 60)


def t4(mod):
    return np.select([mod < 579, mod < 600, mod < 720], [0, 1, 2], 3)


# ---------- 동종 봉 ----------
def touch_table(raw, syms, mins, el, rl):
    TT = {}
    for s in syms:
        try:
            d = prep(bars(raw, s, mins), el, rl)
        except FileNotFoundError:
            continue
        gap = 0.03 if s in LEV else 0.02
        tch = (d.l <= d.ema.shift(1) * (1 - gap)).values
        st = d.index.as_unit("ns").asi8
        TT[s] = (st, np.concatenate([[0], np.cumsum(tch)]), d)
    return TT


def ptouch_time(TT, peers, tns, win=M30, incl_now=False, drop_empty=False):
    vals = []
    for p in peers:
        if p not in TT:
            continue
        st, cs, _ = TT[p]
        lo = np.searchsorted(st, tns - win, "left")
        hi = np.searchsorted(st, tns, "right" if incl_now else "left")
        if hi - lo == 0:
            if drop_empty:
                continue
            vals.append(0.0)
            continue
        vals.append(float(cs[hi] - cs[lo] > 0))
    return np.mean(vals) if vals else np.nan


def run_tab(df, feats, outs, nd_code="dcode"):
    """앞 문턱(1/3, 2/3) → 앞·뒤 날짜 t. 반환 행 목록."""
    f1 = df[df.half == "앞"]
    rows = []
    for f in feats:
        q33, q67 = f1[f].quantile([1 / 3, 2 / 3]).values
        for sp in ("하위", "상위"):
            for yv in outs:
                rec = dict(feat=f, split=sp, y=yv, q33=q33, q67=q67)
                for h in ("앞", "뒤"):
                    x = df[df.half == h]
                    v = x[f].values
                    fin = np.isfinite(v)
                    a = (v <= q33) if sp == "하위" else (v > q67)
                    m, t, n = dayt(x.day.values[fin], x[yv].values.astype(float)[fin], a[fin])
                    rec.update({f"{h}_dd": m, f"{h}_t": t, f"{h}_nd": n})
                rows.append(rec)
    return pd.DataFrame(rows)


def selection_null(F1, F2, d1, d2, n1, n2, Y1, Y2, iters, shuffler=None, cells=None):
    """F1,F2: (n, nf) 특성 행렬. 앞 |t| 상위 3 특성(특성마다 최강 조합 1개) → 뒤 같은 방향 t. 반환 (iters, 3)."""
    res = []
    for it in range(iters):
        if shuffler == "normal":
            A1 = RNG.standard_normal(F1.shape)
            A2 = RNG.standard_normal(F2.shape)
        else:
            p1 = perm_within(cells[0], np.arange(len(F1)), RNG)
            p2 = perm_within(cells[1], np.arange(len(F2)), RNG)
            A1, A2 = F1[p1], F2[p2]
        cand = []
        for j in range(A1.shape[1]):
            v1, v2 = A1[:, j], A2[:, j]
            f1, f2 = np.isfinite(v1), np.isfinite(v2)
            q33, q67 = np.nanquantile(v1, [1 / 3, 2 / 3])
            for sp in (0, 1):
                a1 = (v1 <= q33) if sp == 0 else (v1 > q67)
                a2 = (v2 <= q33) if sp == 0 else (v2 > q67)
                for yk in Y1:
                    m, t = dayt_fast(d1[f1], n1, Y1[yk][f1], a1[f1])
                    if np.isfinite(t):
                        cand.append((abs(t), j, np.sign(m), a2, f2, yk))
        cand.sort(key=lambda c: -c[0])
        seen, pk = set(), []
        for c in cand:
            if c[1] in seen:
                continue
            seen.add(c[1])
            m, t = dayt_fast(d2[c[4]], n2, Y2[c[5]][c[4]], c[3][c[4]])
            pk.append(c[2] * t if np.isfinite(t) else np.nan)
            if len(pk) == 3:
                break
        res.append(pk + [np.nan] * (3 - len(pk)))
    return np.array(res)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    X = pd.read_csv(os.path.join(RES, "pat_peers_trades.csv"))
    X["ts"] = pd.to_datetime(X.t, utc=True).dt.tz_convert("America/New_York")
    X["tns"] = X.ts.map(lambda z: z.value).astype("int64")
    X["day"] = pd.to_datetime(X.day).dt.date
    X["fast"] = ((X.why == "RSI") & (X.bars <= 20)).astype(float)
    R = X[X.ses == "정규"].copy().reset_index(drop=True)
    say("# 반박 검증: pat_peers 후보 (verify_pat_peers, 2026-10-10)")
    say("정의(분석가와 같음, 바꾸지 않음): Y1=ret, Y2=fast=(why=RSI & bars<=20). 사건 E·EY1·EY2는 pat_peers.py 정의 그대로.")
    say(f"정규장 {len(R)}건(앞 {sum(R.half == '앞')}·뒤 {sum(R.half == '뒤')}), Y2=1 앞 {int(R[R.half == '앞'].fast.sum())}·뒤 {int(R[R.half == '뒤'].fast.sum())}")

    # ---------- 0) 재현 ----------
    say("\n## 0) 분석가 숫자 재현(같은 csv 특성, 앞 문턱)")
    tab = run_tab(R, ["ptouch", "sdev_q", "rel_dev"], ["ret", "fast"])
    for _, q in tab.iterrows():
        say(f"  {q.feat} {q.split} {q.y}: 앞 {pct(q['앞_dd'])} t={q['앞_t']:+.2f}({q['앞_nd']}일) | 뒤 {pct(q['뒤_dd'])} t={q['뒤_t']:+.2f}({q['뒤_nd']}일) | q33 {q.q33:+.4f} q67 {q.q67:+.4f}")

    # ---------- 1) 미래 참조: 시간 창 ptouch 독립 재계산 ----------
    say("\n## 1) 미래 참조 점검: ptouch를 1분봉→3분봉에서 시간 창으로 다시 계산")
    have = {f[3:-8] for f in os.listdir(RAW) if f.startswith("1m_")}
    P, lvl, _ = PP.peer_map(set(X.sym) | (have - {"QQQ", "SMH", "IGV", "ARKG", "XBI", "IWM", "SPY"}))
    need = sorted(set(X.sym) | {p for s in set(X.sym) for p in P.get(s, [])})
    TT = touch_table(RAW, need, 3, 60, 14)
    for nm, kw in [("pt_time", {}), ("pt_time_drop", {"drop_empty": True}), ("pt_now", {"incl_now": True}),
                   ("pt_15", {"win": M30 // 2}), ("pt_60", {"win": M30 * 2})]:
        R[nm] = [ptouch_time(TT, P.get(s, []), t, **kw) for s, t in zip(R.sym, R.tns)]
    ok = R.ptouch.notna() & R.pt_time.notna()
    same = (np.abs(R.ptouch[ok] - R.pt_time[ok]) < 1e-9).mean()
    say(f"  분석가 ptouch vs 시간 창 재계산: 같은 값 {same:.1%}, 순위상관 {R.ptouch.corr(R.pt_time, method='spearman'):+.3f}")
    o30 = R["mod"] < 600
    say(f"  다른 값 비율: 개장 30분 {(np.abs(R.ptouch - R.pt_time) > 1e-9)[o30 & ok].mean():.1%} / 그 뒤 {(np.abs(R.ptouch - R.pt_time) > 1e-9)[~o30 & ok].mean():.1%}"
        " (개장 직후 창에 프리장 봉이 비면 분석가 정의는 더 먼 과거 봉을 봄 — 미래 참조는 아님)")
    # 분석가 값이 봉 i를 썼다면 pt_now와 더 가까워야 한다
    say(f"  분석가 ptouch와 같은 값 비율: 봉 i 제외판 {same:.1%} vs 봉 i 포함판(미래 참조) {(np.abs(R.ptouch - R.pt_now) < 1e-9)[ok].mean():.1%}")
    th = 0.2576
    for nm in ["ptouch", "pt_time", "pt_time_drop", "pt_15", "pt_60", "pt_now"]:
        f1 = R[R.half == "앞"][nm]
        q67 = f1.quantile(2 / 3)
        parts = []
        for h in ("앞", "뒤"):
            x = R[R.half == h]
            v = x[nm].values
            fin = np.isfinite(v)
            a = v > q67
            nA, nB = int((a & fin).sum()), int((~a & fin).sum())
            kA, kB = int(x.fast.values[a & fin].sum()), int(x.fast.values[~a & fin].sum())
            m, t, n = dayt(x.day.values[fin], x.fast.values[fin], a[fin])
            m1, t1, _ = dayt(x.day.values[fin], x.ret.values[fin], a[fin])
            parts.append(f"{h} Y2 {kA}/{nA} vs {kB}/{nB} t={t:+.2f} · Y1 t={t1:+.2f}")
        say(f"  {nm:13s} 상위1/3(앞 q67={q67:.3f}): " + " | ".join(parts))

    # ---------- 2) 시간대 교란 ----------
    say("\n## 2) 시간대 교란: Y2=1 거래가 언제 나오나, 날짜x시각칸 통제")
    R["slot"] = tslot(R["mod"].values)
    R["t4"] = t4(R["mod"].values)
    R["A"] = (R.ptouch > th).astype(float)
    for h in ("앞", "뒤"):
        x = R[R.half == h]
        tbl = x.groupby("slot").agg(n=("fast", "size"), y2=("fast", "sum"), A비중=("A", "mean"))
        say(f"  {h} 시각칸별 건수·Y2·A비중: " + "; ".join(f"{('%02d:%02d' % divmod(570 + 3 * s, 60)) if s < 10 else ('%02d시' % (s))}"
                                                 f" n{int(r.n)} Y2 {int(r.y2)} A {r.A비중:.0%}" for s, r in tbl.iterrows()))
    dcodes = {d_: i for i, d_ in enumerate(sorted(R.day.unique()))}
    R["dcode"] = R.day.map(dcodes)
    R["cell"] = R.dcode * 100 + R.slot
    R["cell4"] = R.dcode * 10 + R.t4
    R["latr"] = np.log(R.atr_d)
    R["livol"] = np.log(R.ivol.clip(lower=1e-5))
    for h in ("앞", "뒤"):
        x = R[R.half == h]
        fin = x.ptouch.notna().values
        x = x[fin]
        for yv in ("fast", "ret"):
            y = x[yv].values.astype(float)
            Xm = np.column_stack([x.A.values, x.latr.values, x.livol.values])
            b1, t1 = fe_ols(y, Xm, x.cell4.values, x.dcode.values)
            b2, t2 = fe_ols(y, Xm, x.cell.values, x.dcode.values)
            Xm3 = np.column_stack([x.A.values, x.latr.values, x.livol.values, x.qqq_dev.values, x.smh_dev.values, x.pdev.values])
            b3, t3 = fe_ols(y, Xm3, x.cell.values, x.dcode.values)
            # 칸 안 순열
            a = x.A.values.astype(bool)
            obs = dayt_fast(x.dcode.values, x.dcode.max() + 1, y, a)[1]
            res = {}
            for nm, cc in [("날짜", x.dcode.values), ("날짜x시각4", x.cell4.values), ("날짜x시각칸", x.cell.values)]:
                z = np.array([dayt_fast(x.dcode.values, x.dcode.max() + 1, y, perm_within(cc, a, RNG))[1] for _ in range(2000)])
                res[nm] = np.nanmean(z <= obs)
            say(f"  {h} {yv}: 날짜x시각4 FE {pct(b1)} t={t1:+.2f} | 날짜x시각칸 FE {pct(b2)} t={t2:+.2f} | +QQQ·SMH·동종위치 {pct(b3)} t={t3:+.2f}"
                f" | 날짜 t={obs:+.2f}, 순열 p(이 이하): 날짜 안 {res['날짜']:.3f}, 날짜x시각4 안 {res['날짜x시각4']:.3f}, 날짜x시각칸 안 {res['날짜x시각칸']:.3f}")
    # 첫 3봉·개장 30분 밖
    for lab, msk in [("09:39 이후", R["mod"] >= 579), ("10:00 이후", R["mod"] >= 600)]:
        parts = []
        for h in ("앞", "뒤"):
            x = R[(R.half == h) & msk & R.ptouch.notna()]
            a = x.A.values.astype(bool)
            m, t, n = dayt(x.day.values, x.fast.values, a)
            parts.append(f"{h} Y2 {int(x.fast[a].sum())}/{a.sum()} vs {int(x.fast[~a].sum())}/{(~a).sum()} t={t:+.2f}({n}일)")
        say(f"  {lab}: " + " | ".join(parts))

    # 꼬리 체결(처음 닿은 1분봉의 직전 1분 종가가 매수가보다 1% 넘게 위) 뺀 정규장
    parts = []
    for h in ("앞", "뒤"):
        x = R[(R.half == h) & (R.clean == 1) & R.ptouch.notna()]
        a = x.A.values.astype(bool)
        m, t, n = dayt(x.day.values, x.fast.values, a)
        m1, t1, _ = dayt(x.day.values, x.ret.values, a)
        parts.append(f"{h} {len(x)}건 Y2 {int(x.fast[a].sum())}/{a.sum()} vs {int(x.fast[~a].sum())}/{(~a).sum()} t={t:+.2f} · Y1 t={t1:+.2f}")
    say("  정규장 깨끗한 체결만: " + " | ".join(parts))

    # ---------- 3) 쏠림·정확검정 ----------
    say("\n## 3) 쏠림: 정규장 Y2=1 거래 전부")
    from math import comb
    for h in ("앞", "뒤"):
        x = R[(R.half == h) & (R.fast == 1)]
        say(f"  {h}: " + "; ".join(f"{r.sym} {r.day.strftime('%m/%d')} {r.ts.strftime('%H:%M')} pt={r.ptouch:.2f}" for r in x.itertuples()))
        xx = R[(R.half == h) & R.ptouch.notna()]
        nA, N, K = int(xx.A.sum()), len(xx), int(xx.fast.sum())
        kA = int(xx.fast[xx.A == 1].sum())
        p = sum(comb(K, j) * comb(N - K, nA - j) for j in range(0, kA + 1)) / comb(N, nA)
        say(f"    거래 단위 초기하 한쪽 p(A의 Y2 <= {kA}) = {p:.3f} (A {nA}/{N}, Y2 {K}건; 거래가 독립이라는 가정이라 실제보다 작게 나옴)")
        say(f"    Y2=1이 나온 날 {x.day.nunique()}일, 종목 {x.sym.nunique()}개 | 날짜별 {dict(x.day.astype(str).value_counts())}")
    for h in ("뒤",):
        x = R[(R.half == h) & R.ptouch.notna()]
        top_days = x[x.fast == 1].day.value_counts().index[:2]
        top_syms = x[x.fast == 1].sym.value_counts().index[:2]
        for lab, k in [("Y2 최다 1일 빼기", ~x.day.isin(top_days[:1])), ("Y2 최다 2일 빼기", ~x.day.isin(top_days)),
                       ("Y2 최다 1종목 빼기", ~x.sym.isin(top_syms[:1])), ("Y2 최다 2종목 빼기", ~x.sym.isin(top_syms))]:
            xx = x[k]
            a = xx.A.values.astype(bool)
            m, t, n = dayt(xx.day.values, xx.fast.values, a)
            say(f"  {h} {lab}: Y2 {int(xx.fast[a].sum())}/{a.sum()} vs {int(xx.fast[~a].sum())}/{(~a).sum()} 날짜차 {pct(m)} t={t:+.2f}({n}일)")

    # ---------- 4) 선택편향 ----------
    say("\n## 4) 선택편향: 분석가 절차(12특성x2분할x2결과→앞 |t| 상위 3특성→뒤)를 300회")
    TF = PP.TRADE_F
    F = R[R.pdev.notna()].copy()
    F1, F2 = F[F.half == "앞"], F[F.half == "뒤"]
    d1 = F1.day.map({d_: i for i, d_ in enumerate(sorted(F1.day.unique()))}).values
    d2 = F2.day.map({d_: i for i, d_ in enumerate(sorted(F2.day.unique()))}).values
    Y1 = {y: F1[y].values.astype(float) for y in ("ret", "fast")}
    Y2 = {y: F2[y].values.astype(float) for y in ("ret", "fast")}
    obs_max = 2.54
    for lab, kw in [("무작위 정규 특성", dict(shuffler="normal")),
                    ("실제 특성 행을 날짜x시각4 안에서 섞기", dict(shuffler="cell", cells=(F1.cell4.values, F2.cell4.values))),
                    ("실제 특성 행을 날짜 안에서 섞기", dict(shuffler="cell", cells=(F1.dcode.values, F2.dcode.values)))]:
        z = selection_null(F1[TF].values.astype(float), F2[TF].values.astype(float), d1, d2, d1.max() + 1, d2.max() + 1, Y1, Y2, 300, **kw)
        mx = np.nanmax(z, axis=1)
        say(f"  {lab}: 고른 3개 중 최대 뒤 t 중앙 {np.nanmedian(mx):+.2f}, 95% {np.nanpercentile(mx, 95):+.2f} | 최대 >= 2.0 {np.mean(mx >= 2):.1%}, >= {obs_max} {np.mean(mx >= obs_max):.1%}")
    say("  (분석가 0.7%는 '2위 후보 칸'만 본 값이다. 고른 3개 중 하나라도 이만큼 나올 확률이 가족 단위 값이다.)")

    # ---------- 5) 사건 E ----------
    say("\n## 5) 사건 E 독립 재구현·후보 재확인")
    EV = pd.read_csv(os.path.join(RES, "pat_peers_events.csv"))
    EV["day"] = pd.to_datetime(EV.day).dt.date
    mine = []
    for s in sorted(set(X.sym)):
        if s not in TT:
            continue
        d = TT[s][2]
        dd = pd.read_parquet(os.path.join(RAW, f"1d_{s}.parquet"))
        dd.index = pd.to_datetime(dd.index).date
        prevc = dd.Close.shift(1)
        c, o, mod = d.c.values, d.o.values, d["mod"].values
        dts = d.date.values
        starts = np.r_[0, np.where(dts[1:] != dts[:-1])[0] + 1]
        ends = np.r_[starts[1:], len(d)]
        for lo, hi in zip(starts, ends):
            dy = dts[lo]
            pc = prevc.get(dy, np.nan)
            if lo < 61 or not np.isfinite(pc):
                continue
            H = np.maximum.accumulate(c[lo:hi])
            reg = np.where((mod[lo:hi] >= 570) & (mod[lo:hi] < 960))[0]
            if not len(reg):
                continue
            last = lo + reg[-1]
            for j in reg:
                if H[j] >= 1.05 * pc and c[lo + j] <= pc + 0.5 * (H[j] - pc):
                    if lo + j + 1 <= last:
                        mine.append((s, dy, d.index[lo + j + 1]))
                    break
    mine = pd.DataFrame(mine, columns=["sym", "day", "t"])
    a_set = set(zip(EV.sym, EV.day))
    m_set = set(zip(mine.sym, mine.day))
    say(f"  사건 수: 분석가 {len(EV)} vs 독립 {len(mine)} | 겹침 {len(a_set & m_set)}, 분석가만 {len(a_set - m_set)}, 독립만 {len(m_set - a_set)}")
    E = EV[EV.half.isin(["앞", "뒤"]) & EV.full].copy().rename(columns={"ey1": "ret", "ey2": "fast"})
    tabe = run_tab(E, ["rel_hi", "phi", "pdev", "rel_give", "lead5", "lead20"], ["ret", "fast"])
    for _, q in tabe.iterrows():
        say(f"  {q.feat} {q.split} {q.y}: 앞 {pct(q['앞_dd'])} t={q['앞_t']:+.2f}({q['앞_nd']}일) | 뒤 {pct(q['뒤_dd'])} t={q['뒤_t']:+.2f}({q['뒤_nd']}일)")
    E["dcode"] = E.day.map({d_: i for i, d_ in enumerate(sorted(E.day.unique()))})
    E["mod"] = pd.to_datetime(E.t, utc=True).dt.tz_convert("America/New_York").map(lambda z: z.hour * 60 + z.minute)
    E["t4"] = t4(E["mod"].values)
    E["latr"] = np.log(E.atr_d)
    E["livol"] = np.log(E.ivol.clip(lower=1e-5))
    for f, sp, th_ in [("rel_hi", "하위", tabe[tabe.feat == "rel_hi"].q33.iloc[0]), ("phi", "상위", tabe[tabe.feat == "phi"].q67.iloc[0]),
                       ("pdev", "상위", tabe[tabe.feat == "pdev"].q67.iloc[0])]:
        for h in ("앞", "뒤"):
            x = E[(E.half == h) & E[f].notna()]
            a = (x[f] <= th_) if sp == "하위" else (x[f] > th_)
            Xm = np.column_stack([a.values.astype(float), x.latr.values, x.livol.values, x.t4.values == 0, x.t4.values == 1])
            b, t_ = fe_ols(x.ret.values, Xm, x.dcode.values, x.dcode.values)
            say(f"  {f} {sp}1/3 {h} EY1: 날짜FE+ATR·변동성·시각 통제 {pct(b)} t={t_:+.2f}, A {int(a.sum())}·B {int((~a).sum())}건, 두 쪽 다 있는 날 {dayt(x.day.values, x.ret.values, a.values)[2]}일")
    e9 = EV[EV.day == date(2026, 10, 9)].set_index("sym")
    if {"LITE", "COHR"} <= set(e9.index):
        say("  10/9 LITE vs COHR 사건 특성: " + ", ".join(f"{k} {e9.loc['LITE', k]:+.3f}/{e9.loc['COHR', k]:+.3f}" for k in ["phi", "rel_hi", "pdev", "rel_give", "lead5", "lead20"]))
        for s in ("LITE", "COHR"):
            d = TT[s][2]
            t0 = pd.Timestamp(e9.loc[s, "t"]).tz_convert("America/New_York")
            aft = d[(d.index >= t0) & (d.date == date(2026, 10, 9))]
            say(f"    {s} 진입 {t0.strftime('%H:%M')} {aft.o.iloc[0]:.2f} → 10:18까지 종가 최고 {pct(aft.c.max() / aft.o.iloc[0] - 1)}, 마지막 {pct(aft.c.iloc[-1] / aft.o.iloc[0] - 1)} (그날 종가 데이터 없음 → EY1·EY2 계산 불가)")

    # ---------- 6) 7/14~9/8 5분봉 ----------
    say("\n## 6) 7/14~9/8 5분봉(stop_lab early, EMA36·RSI9, Y2=12봉=60분 안 RSI 매도), 시간 창 ptouch(30분)")
    try:
        import stop_lab
        E5 = stop_lab.run(stop_lab.load("early"), None, "early")
        E5 = E5[E5.ses == "정규"].copy()
        E5["fast"] = ((E5.why == "RSI") & (E5.bars <= 12)).astype(float)
        E5["ts"] = pd.to_datetime(E5.t)
        if E5.ts.dt.tz is None:
            E5["ts"] = E5.ts.dt.tz_localize("America/New_York")
        E5["tns"] = E5.ts.map(lambda z: pd.Timestamp(z).value).astype("int64")
        need5 = sorted(set(E5.sym) | {p for s in set(E5.sym) for p in P.get(s, [])})
        TT5 = touch_table(RAW_OLD, need5, 5, 36, 9)
        E5["pt"] = [ptouch_time(TT5, P.get(s, []), t) for s, t in zip(E5.sym, E5.tns)]
        E5["mod"] = E5.ts.dt.hour * 60 + E5.ts.dt.minute
        E5 = E5[E5.pt.notna()]
        dc = E5.day.map({d_: i for i, d_ in enumerate(sorted(E5.day.unique()))}).values
        for lab, thr in [("3분봉 앞 문턱 0.2576", th), ("5분봉 자체 상위1/3", E5.pt.quantile(2 / 3))]:
            a = (E5.pt > thr).values
            for yv in ("fast", "ret"):
                m, t, n = dayt(E5.day.values, E5[yv].values.astype(float), a)
                c4 = dc * 10 + t4(E5["mod"].values)
                Xm = np.column_stack([a.astype(float)])
                b, tb_ = fe_ols(E5[yv].values.astype(float), Xm, c4, dc)
                say(f"  [{lab}] {yv}: A {a.sum()}건 {pct(E5[yv][a].mean())} vs B {(~a).sum()}건 {pct(E5[yv][~a].mean())} | 날짜차 {pct(m)} t={t:+.2f}({n}일) | 날짜x시각4 FE t={tb_:+.2f}")
    except Exception as ex:
        say(f"  5분봉 보조 실패: {ex!r}")

    # ---------- 7) 10/9 사례 ----------
    say("\n## 7) 10/9 사례(1분봉 10:18 ET까지, 지표 상태 기계로 실제 매수 확인)")
    import build
    for s in ["CRCL", "MRVL", "LITE", "COHR"]:
        d, dead_of, hot = build.arrays(s)
        o, l, r, ema, tk = d.o.values, d.l.values, d.r.values, d.ema.values, d.index.as_unit("ns").asi8
        hold, ent, ei, dead, buys = False, 0.0, 0, 0, []
        for i in range(61, len(d)):
            if hold:
                why = "RSI" if (i - 1 >= ei and r[i - 1] > 70) else ("8시" if tk[i] >= dead else None)
                if why:
                    if d.date.iloc[ei] == date(2026, 10, 9):
                        buys[-1] += (d.index[i].strftime("%H:%M"), why, o[i] / ent - 1)
                    hold = False
            elif not hot[i]:
                lvl = ema[i - 1] * (1 - (0.03 if s in LEV else 0.02))
                if l[i] <= lvl:
                    ent, ei, hold, dead = min(o[i], lvl), i, True, dead_of[i]
                    if d.date.iloc[i] == date(2026, 10, 9):
                        buys.append((d.index[i].strftime("%H:%M"), d.ses.iloc[i], ent))
        reg = [b for b in buys if b[1] == "정규"]
        txt = []
        for b in buys:
            tns = pd.Timestamp(f"2026-10-09 {b[0]}", tz="America/New_York").tz_convert("UTC").value
            pt = ptouch_time(TT, P.get(s, []), tns)
            pa = (d.c[(d.date == date(2026, 10, 9)) & (d.index >= pd.Timestamp(f"2026-10-09 {b[0]}", tz="America/New_York"))])
            tail = f"매도 {b[3]} {b[4]} {pct(b[5])}" if len(b) > 3 else f"10:18 미청산 {pct(pa.iloc[-1] / b[2] - 1)}"
            txt.append(f"{b[0]} {b[1]} {b[2]:.2f} ptouch(시간창) {pt:.3f} → {'A(경고)' if pt > th else 'B'} | {tail}")
        say(f"  {s} 동종 {P.get(s)}: " + (" ; ".join(txt) if txt else "10/9 매수 없음") + ("" if reg else " (정규장 매수 없음)"))

    with open(os.path.join(RES, "verify_pat_peers.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
