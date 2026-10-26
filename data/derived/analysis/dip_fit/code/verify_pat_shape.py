"""반박 검증: pat_shape.py 후보와 관찰을 처음부터 다시 계산해 깬다(2026-10-10). 기본값은 '살아남지 못함'.

후보: [정규] rsi1<=30.15 Y1 나쁨, [정규] ndown>3 Y1 나쁨, [E1] d_prev<=1.707% Y1 좋음.
관찰: [정규] kday<=15.67 Y2 나쁨(고점에서 막 떨어진 매수는 빠른 반등이 적다), 뒤에만 보인 gap>1.77%·spike>3.6% Y1 나쁨,
      E1 사례 특성(retr·vol5·o_line — LITE vs COHR).
확인 항목:
  [1] 재현: 거래표와 특성 캐시가 같은 행인지, y2·atr 칸 재계산, 내 코드로 특성 다시 계산(분석가 feat_at 안 씀).
  [2] 미래 참조: 매수 봉 i의 고가·저가·종가를 시가로 덮고 거래량 0, i 뒤 봉을 지운 3분봉으로 분석가 feat_at을 다시 돌려 같은지.
  [3] 후보 숫자 재계산(내 날짜 군집 t·통제 t), 1종목·1일 빼기(앞은 규칙에 가장 불리한 것, 뒤는 가장 유리한 것), 시간대 분해,
      정규장 꼬리 체결 제외, ret/atr14.
  [4] 우연 대조: (a) 같은 날 안에서 특성 행을 섞는 순열 500회, (b) 무작위 정규 특성 500회로 '앞에서 고르기 → 뒤 확인' 절차 전체를 다시 돈다.
  [5] kday Y2 관찰: 네 t(앞 원·통제, 뒤 원·통제)가 같은 방향인 최소 |t|를 순열 최댓값과 비교, 5분봉 앞 기간(7/14~9/8) 확인.
  [6] E1 사례 특성·뒤에만 보인 gap·spike를 5분봉 앞 기간에서 같은 방향으로 잰다(가격 기준 특성은 같은 문턱을 그대로 씀).
  [7] 10/9 사례 값 재계산(내 코드), 매수 체결이 1분 꼬리인지, 규칙 적용 결과.
입력(다시 받지 않음): ../results/trades.csv, pat_shape_feats.csv, pat_shape_1009.csv, features.csv, raw_1009 1분봉, stop_lab early 캐시
출력: ../results/verify_pat_shape.txt
사용: py verify_pat_shape.py"""
import os, sys, time, warnings
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "../results")
sys.path.append(os.path.join(HERE, "../../dip_reason/code"))
sys.path.append(HERE)
from events import bars, prep, tod_rvol
import build
import pat_shape as P

RAW = build.RAW
warnings.filterwarnings("ignore", category=RuntimeWarning)
CUT = P.CUT
FEATS = P.FEATS
E1F = P.E1_FEATS
NPERM = 500
RNG = np.random.default_rng(20261010)
OUT = []
NT = {"n": 0}


def say(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s, flush=True)


def pct(x):
    return "nan" if x is None or not np.isfinite(x) else f"{100 * x:+.2f}%"


# ------------------------------------------------------------------ 통계(내 구현, numpy)
def tst(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return (x.mean() if n else np.nan), np.nan, n
    sd = x.std(ddof=1)
    return x.mean(), (x.mean() / (sd / np.sqrt(n)) if sd > 0 else np.nan), n


class Pool:
    """정규장(또는 부분집단) 거래 묶음. 날짜 코드·통제 칸 코드를 미리 만든다."""

    def __init__(self, df, ctrl):
        self.df = df.reset_index(drop=True)
        d = self.df
        self.days = np.array(sorted(d.day.unique()))
        self.dc = np.searchsorted(self.days, d.day.values)
        self.D = len(self.days)
        key = d.day.astype(str)
        for k in ctrl:
            key = key + "|" + d[k].astype(str)
        cu, self.cc = np.unique(key.values, return_inverse=True)
        self.C = len(cu)
        self.cday = np.zeros(self.C, int)
        self.cday[self.cc] = self.dc
        self.y = {"1": d.ret.values.astype(float), "2": d.y2.values.astype(float), "ra": d.ra.values.astype(float)}
        if "r60" in d:
            self.y["r60"] = d.r60.values.astype(float)
        self.half = d.half.values

    def diff(self, a, sub, y):
        """sub(불리언) 안에서 a vs ~a. 반환 dd,t,cdd,ct,nd."""
        yy = self.y[y]
        A, B = sub & a, sub & ~a
        dc, D = self.dc, self.D
        nA, nB = np.bincount(dc[A], minlength=D), np.bincount(dc[B], minlength=D)
        sA, sB = np.bincount(dc[A], weights=yy[A], minlength=D), np.bincount(dc[B], weights=yy[B], minlength=D)
        ok = (nA > 0) & (nB > 0)
        dd, t, nd = tst(sA[ok] / nA[ok] - sB[ok] / nB[ok])
        cc, C = self.cc, self.C
        nA, nB = np.bincount(cc[A], minlength=C), np.bincount(cc[B], minlength=C)
        sA, sB = np.bincount(cc[A], weights=yy[A], minlength=C), np.bincount(cc[B], weights=yy[B], minlength=C)
        ok = (nA > 0) & (nB > 0)
        x = sA[ok] / nA[ok] - sB[ok] / nB[ok]
        num = np.bincount(self.cday[ok], weights=x, minlength=D)
        cnt = np.bincount(self.cday[ok], minlength=D)
        cdd, ct, _ = tst(num[cnt > 0] / cnt[cnt > 0])
        return dd, t, cdd, ct, nd


def splits_of(x1, kinds):
    x1 = x1[np.isfinite(x1)]
    if len(x1) < 20:
        return []
    q33, q50, q67 = np.quantile(x1, [1 / 3, 0.5, 2 / 3])
    out = []
    if "하위1/3" in kinds:
        out.append(("하위1/3", q33, "le"))
    if "상위1/3" in kinds:
        out.append(("상위1/3", q67, "gt"))
    if "하위1/2" in kinds:
        out.append(("하위1/2", q50, "le"))
    return out


def run_tests(pool, X, names, kinds, need2=5):
    """분석가와 같은 고르기 절차를 내 코드로: 문턱·방향은 앞, 뒤에 그대로. 반환 DataFrame."""
    rows = []
    f_half = pool.half == "앞"
    b_half = pool.half == "뒤"
    for j, nm in enumerate(names):
        x = X[:, j]
        fin = np.isfinite(x)
        for sp, q, op in splits_of(x[f_half & fin], kinds):
            a = (x <= q) if op == "le" else (x > q)
            a = a & fin
            s1, s2 = f_half & fin, b_half & fin
            for y in ("1", "2"):
                dd1, t1, cdd1, ct1, nd1 = pool.diff(a, s1, y)
                dd2, t2, cdd2, ct2, nd2 = pool.diff(a, s2, y)
                sgn = 1 if (dd1 if np.isfinite(dd1) else 0) >= 0 else -1
                ok = np.isfinite(t1) and np.isfinite(ct1) and np.sign(t1) == np.sign(ct1)
                y2nA = pool.y["2"][s1 & a].sum()
                if y == "2" and y2nA < need2:
                    ok = False
                score = min(abs(t1), abs(ct1)) if ok else 0.0
                rows.append(dict(feat=nm, split=sp, thr=q, op=op, Y=f"Y{y}", sgn=sgn, score=score, t1=t1, ct1=ct1, dd1=dd1,
                                 t2=t2, ct2=ct2, dd2=dd2, y2nA1=y2nA, nA1=int((s1 & a).sum()), nA2=int((s2 & a).sum())))
    return pd.DataFrame(rows)


def consist(r):
    """네 t가 모두 같은 방향이면 최소 |t|, 아니면 0."""
    v = np.array([r.t1, r.ct1, r.t2, r.ct2], float)
    if not np.all(np.isfinite(v)) or len(set(np.sign(v))) != 1:
        return 0.0
    return float(np.min(np.abs(v)))


def null_runs(pool, X, names, kinds, kind, iters, pick_k):
    """kind='perm': 같은 날 안 특성 행 섞기, 'gauss': 무작위 정규 특성(같은 수). 고르기→뒤 확인 전체를 다시 돈다."""
    res = []
    groups = [np.where(pool.dc == k)[0] for k in range(pool.D)]
    for it in range(iters):
        if kind == "perm":
            perm = np.arange(len(pool.df))
            for g in groups:
                perm[g] = RNG.permutation(g)
            Xn = X[perm]
        else:
            Xn = RNG.standard_normal(X.shape)
        tb = run_tests(pool, Xn, names, kinds)
        best = tb.sort_values("score", ascending=False)
        picks = best[best.score > 0].drop_duplicates("feat").head(pick_k)
        same2 = (tb.sgn * tb.t2).values
        c2 = tb[tb.Y == "Y2"].apply(consist, axis=1).max()
        call = tb.apply(consist, axis=1).max()
        res.append(dict(max_score=best.score.max(), top_same2=(picks.sgn.iloc[0] * picks.t2.iloc[0]) if len(picks) else np.nan,
                        top_same2_ct=(picks.sgn.iloc[0] * picks.ct2.iloc[0]) if len(picks) else np.nan,
                        mean_same2=np.nanmean(same2), pos_same2=np.mean(same2[np.isfinite(same2)] > 0),
                        max_cons_y2=c2, max_cons_all=call))
    return pd.DataFrame(res)


# ------------------------------------------------------------------ 내 특성 계산(분석가 feat_at 안 씀)
def reg_close_open(s):
    m = pd.read_parquet(os.path.join(RAW, f"1m_{s}.parquet"))
    mm = m.index.hour * 60 + m.index.minute
    rg = (mm >= 570) & (mm < 960)
    dts = np.asarray(m.index[rg].date)
    rc = m.Close[rg].groupby(dts).last()
    ro = m.Open[rg].groupby(dts).first()
    return m, rc, ro


def my_feats(s, items, cut=True):
    """items: [(ts, ent)] → 특성 dict 목록. 1분봉에서 정규장 종가·시가를 따로 구해 쓴다."""
    b = bars(RAW, s, 3)
    d = prep(b, 60, 14)
    if cut:
        d = d[d.index < CUT]
    o, h, l, c, v, r, ema = (d[k].values.astype(float) for k in ("o", "h", "l", "c", "v", "r", "ema"))
    mod, ses = d["mod"].values, d.ses.values
    dates = np.asarray(d.date)
    udates = sorted(set(dates))
    first = {}
    for k, dt in enumerate(dates):
        first.setdefault(dt, k)
    pos = {(dt, md): k for k, (dt, md) in enumerate(zip(dates, mod))}
    m1, rc, ro = reg_close_open(s)
    rdates = list(rc.index)
    gp = 0.03 if s in build.LEV else 0.02
    out = []
    for ts, ent in items:
        i = d.index.get_loc(ts)
        dt = dates[i]
        line = ema[i - 1] * (1 - gp)
        f = {"ent_mine": min(o[i], line)}
        j0 = first[dt]
        if ses[i] == "애프터":
            pc = rc.get(dt, np.nan)
        else:
            pr = [x for x in rdates if x < dt]
            pc = rc[pr[-1]] if pr else np.nan
        if j0 < i:
            k = j0 + int(np.argmax(h[j0:i]))
            dh = h[k]
            f["kday"], f["dd_day"] = i - k, ent / dh - 1
        else:
            dh = np.nan
            f["kday"] = f["dd_day"] = np.nan
        f["rsi1"] = r[i - 1]
        n = 0
        while n < 30 and c[i - 1 - n] < o[i - 1 - n]:
            n += 1
        f["ndown"] = n
        f["o_line"] = o[i] / line - 1
        f["spike"] = dh / pc - 1
        f["retr"] = (dh - ent) / (dh - pc) if (dh - pc) > 0.02 * pc else np.nan
        f["d_prev"] = ent / pc - 1
        if ses[i] == "정규":
            f["tmin"] = mod[i] - 570
            f["gap"] = ro.get(dt, np.nan) / pc - 1
            s5 = v[i - 5:i].sum()
            p = udates.index(dt)
            base = []
            for pd_ in udates[max(0, p - 10):p]:
                jj = pos.get((pd_, mod[i]))
                if jj is not None and jj >= 5:
                    base.append(v[jj - 5:jj].sum())
            f["vol5"] = s5 / np.mean(base) if len(base) >= 5 else np.nan
        else:
            f["tmin"] = f["gap"] = f["vol5"] = np.nan
        # 1분 꼬리 체결: 닿은 1분봉의 직전 1분 종가 / 매수가 - 1
        w = m1[(m1.index >= ts) & (m1.index < ts + pd.Timedelta(minutes=3))]
        hit = w.index[w.Low <= ent * (1 + 1e-6)]
        f["prv1m_mine"] = np.nan
        if len(hit):
            p1 = m1.index.get_loc(hit[0])
            f["prv1m_mine"] = m1.Close.iloc[p1 - 1] / ent - 1 if p1 > 0 else np.nan
        out.append(f)
    return out


def lookahead_check(s, g):
    """봉 i의 고가·저가·종가=시가, 거래량 0, i 뒤 봉 삭제 → 분석가 feat_at 그대로 → 원 특성과 최대 차이."""
    b = bars(RAW, s, 3)
    b = b[b.index < CUT]
    full = prep(b, 60, 14)
    gp = 0.03 if s in build.LEV else 0.02
    diffs = []
    for _, tr in g.iterrows():
        i = full.index.get_loc(tr.ts)
        b2 = b.iloc[:i + 1].copy()
        oi = b2.Open.iloc[i]
        b2.iloc[i, b2.columns.get_loc("High")] = oi
        b2.iloc[i, b2.columns.get_loc("Low")] = oi
        b2.iloc[i, b2.columns.get_loc("Close")] = oi
        b2.iloc[i, b2.columns.get_loc("Volume")] = 0
        d2 = prep(b2, 60, 14)
        o, h, l, c, r, ema = (d2[k].values.astype(float) for k in ("o", "h", "l", "c", "r", "ema"))
        day, ses, mod = np.asarray(d2.date), d2.ses.values, d2["mod"].values
        rv5 = tod_rvol(d2, 5, 10)
        prev, tc = P.prev_close_map(day, ses, c)
        f = P.feat_at(o, h, l, c, r, ema, day, ses, mod, rv5, i, tr.ent, gp, prev, tc)
        row = {}
        for k in FEATS:
            a, bb = f[k], tr[k]
            if pd.isna(a) and pd.isna(bb):
                row[k] = 0.0
            elif pd.isna(a) or pd.isna(bb):
                row[k] = np.inf
            else:
                row[k] = abs(a - bb) / max(1.0, abs(bb))
        diffs.append(row)
    return pd.DataFrame(diffs)


# ------------------------------------------------------------------ 본문
def main():
    t0 = time.time()
    tr = pd.read_csv(os.path.join(RES, "trades.csv"))
    T = pd.read_csv(os.path.join(RES, "pat_shape_feats.csv"))
    say("# pat_shape 반박 검증 (verify_pat_shape, 2026-10-10)")
    say("결과 변수(분석가가 시험 전 고정, 여기서도 그대로): Y1 = 거래 수익 ret | Y2 = why=RSI 이고 bars<=20(매수 뒤 60분 안 RSI 매도)")
    say("사건 연구: E1 = 정규장 매수 중 매수 직전까지 그날(04시~) 고점 >= 전날 정규장 종가 +5% | E2 = 10/9 재현(10:15 봉까지, 설명용)")
    same = (len(tr) == len(T)) and (tr.sym.values == T.sym.values).all() and (tr.t.values == T.t.values).all()
    say(f"\n== [1] 재현 ==\n거래표 {len(tr)}건과 특성 캐시 {len(T)}건 같은 행 순서: {same}")
    T["ts"] = pd.to_datetime(T.t, utc=True).dt.tz_convert("America/New_York")
    T["day"] = pd.to_datetime(T.day).dt.date
    y2m = ((T.why == "RSI") & (T.bars <= 20)).astype(float)
    say(f"y2 재계산 불일치 {int((y2m != T.y2).sum())}건")
    f9 = pd.read_csv(os.path.join(RES, "features.csv")).set_index("sym")
    q1, q2 = f9.atr14.quantile([1 / 3, 2 / 3])
    at = T.sym.map(f9.atr14)
    atrt = np.select([at <= q1, at <= q2, at > q2], [0, 1, 2], -1)
    say(f"atr14 칸 재계산 불일치 {int((atrt != T.atrt).sum())}건, atr 없는 종목 거래 {int(at.isna().sum())}건({', '.join(T.sym[at.isna()].unique())})")

    # 내 코드로 특성 다시 계산
    chk = ["kday", "dd_day", "rsi1", "ndown", "o_line", "spike", "retr", "d_prev", "tmin", "gap", "vol5"]
    mine = []
    for s, g in T.groupby("sym"):
        ff = my_feats(s, list(zip(g.ts, g.ent)))
        for ix, f in zip(g.index, ff):
            f["ix"] = ix
            mine.append(f)
    M = pd.DataFrame(mine).set_index("ix").sort_index()
    say(f"매수가 재계산(min(시가, EMA60[1]x(1-2%/3%))) 최대 상대오차 {np.nanmax(np.abs(M.ent_mine / T.ent - 1)):.1e}")
    parts = []
    for k in chk:
        a, b = M[k], T[k]
        nanmis = int((a.isna() != b.isna()).sum())
        both = a.notna() & b.notna()
        mx = float(np.max(np.abs(a[both] - b[both]) / np.maximum(1, np.abs(b[both])))) if both.any() else np.nan
        parts.append(f"{k} 최대차 {mx:.1e}·NaN 불일치 {nanmis}")
    say("내 특성 vs 분석가: " + " | ".join(parts))
    say(f"1분 꼬리 지표 prv1m 최대차 {np.nanmax(np.abs(M.prv1m_mine - T.prv1m)):.1e}")

    # [2] 미래 참조
    say("\n== [2] 미래 참조 시험: 봉 i 고가·저가·종가=시가, 거래량 0, 봉 i 뒤 삭제 후 분석가 feat_at 재실행 ==")
    LA = []
    for s, g in T.groupby("sym"):
        LA.append(lookahead_check(s, g))
    LA = pd.concat(LA)
    worst = LA.max()
    say("19개 특성 최대 상대차: " + ", ".join(f"{k} {worst[k]:.1e}" for k in FEATS))
    say(f"차이 1e-9 넘는 거래·특성 칸 {int((LA.values > 1e-9).sum())}개 / {LA.size}개 → 봉 i의 고·저·종·거래량과 미래 봉을 쓰지 않음" if (LA.values > 1e-9).sum() == 0
        else f"차이 1e-9 넘는 칸 {int((LA.values > 1e-9).sum())}개 — 미래 참조 의심: " + ", ".join(k for k in FEATS if worst[k] > 1e-9))
    say(f"(경과 {time.time() - t0:.0f}초)")

    # ------------------------------------------------------------------ 정규장 묶음
    T["open30"] = (T.ses == "정규") & (T.tmin < 30)
    T["tailf"] = T.prv1m > 0.01
    T["atrt"] = atrt
    T["ra"] = T.ret / at
    reg = T[T.ses == "정규"].copy()
    say(f"\n정규장 {len(reg)}건(앞 {int((reg.half == '앞').sum())} / 뒤 {int((reg.half == '뒤').sum())}), Y2 {int(reg.y2.sum())}건 "
        f"(앞 {int(reg[reg.half == '앞'].y2.sum())} / 뒤 {int(reg[reg.half == '뒤'].y2.sum())}), 개장 30분 안 {reg.open30.mean() * 100:.0f}%, 정규장 1분 꼬리 체결 {reg.tailf.mean() * 100:.0f}%")
    PR = Pool(reg, ["open30", "atrt"])
    XR = PR.df[FEATS].values.astype(float)
    tab = run_tests(PR, XR, FEATS, ("하위1/3", "상위1/3"))
    NT["n"] += len(tab)
    e1 = reg[reg.spike >= P.SPIKE].copy()
    PE = Pool(e1, ["open30"])
    XE = PE.df[E1F].values.astype(float)
    tabe = run_tests(PE, XE, E1F, ("하위1/2",))
    NT["n"] += len(tabe)
    say(f"E1 {len(e1)}건(앞 {int((e1.half == '앞').sum())} / 뒤 {int((e1.half == '뒤').sum())}), Y2 {int(e1.y2.sum())}건(뒤 {int(e1[e1.half == '뒤'].y2.sum())}건)")
    say(f"정규장 76개 시험 재현: 앞 최고 점수 {tab.score.max():.2f}({tab.sort_values('score').iloc[-1].feat}), "
        f"뒤 같은 방향 t 평균 {np.nanmean(tab.sgn * tab.t2):+.2f}, 양수 비율 {np.mean((tab.sgn * tab.t2).dropna() > 0) * 100:.0f}%")

    # ------------------------------------------------------------------ [3] 후보
    say("\n== [3] 후보 재계산(내 날짜 군집 t, 통제 = 같은 날·개장30분·atr14 칸) ==")

    def detail(pool, X, names, feat, op, q, ylab, sgn, label):
        j = names.index(feat)
        x = X[:, j]
        fin = np.isfinite(x)
        a = ((x <= q) if op == "le" else (x > q)) & fin
        d = pool.df
        res = {}
        for h in ("앞", "뒤"):
            sub = (pool.half == h) & fin
            dd, t, cdd, ct, nd = pool.diff(a, sub, ylab)
            _, tra, _, _, _ = pool.diff(a, sub, "ra")
            yA, yB = pool.y[ylab][sub & a].mean(), pool.y[ylab][sub & ~a].mean()
            # 1종목·1일 빼기: 앞은 규칙에 가장 불리한 것, 뒤는 가장 유리한 것
            worst_s = None
            for col in ("sym", "day"):
                best = None
                for v in d.loc[sub, col].unique():
                    k = sub & (d[col].values != v)
                    dd_, t_, _, _, _ = pool.diff(a, k, ylab)
                    if not np.isfinite(dd_):
                        continue
                    key = sgn * dd_ if h == "앞" else -sgn * dd_
                    if best is None or key < best[0]:
                        best = (key, v, dd_, t_)
                res[(h, col)] = best
            tl = []
            for nm, kk in (("개장30분 안", d.open30.values), ("나머지", ~d.open30.values)):
                _, tt, _, _, _ = pool.diff(a, sub & kk, ylab)
                tl.append(f"{nm} t={tt:+.2f}")
            notail = sub & ~d["tailf"].values
            _, tnt, _, ctnt, _ = pool.diff(a, notail, ylab)
            y2c = int(pool.y["2"][sub & a].sum()), int(pool.y["2"][sub & ~a].sum())
            fy = (lambda z: pct(z)) if ylab == "1" else (lambda z: f"{100 * z:.1f}%")
            say(f"  [{label}·{h}] A {int((sub & a).sum())}건 vs B {int((sub & ~a).sum())}건 | {('Y1' if ylab == '1' else 'Y2')} {fy(yA)} vs {fy(yB)}"
                f" 날짜차 {fy(dd)} t={t:+.2f} 통제 t={ct:+.2f} ({nd}일) | ret/atr t={tra:+.2f} | Y2 사건 A {y2c[0]} / B {y2c[1]}")
            xs, xd = res[(h, "sym")], res[(h, "day")]
            say(f"      {'불리한' if h == '앞' else '유리한'} 1개 빼기: -{xs[1]} t={xs[3]:+.2f}, -{xd[1]} t={xd[3]:+.2f} | " + ", ".join(tl)
                + f" | 1분 꼬리 체결 뺌 t={tnt:+.2f} 통제 t={ctnt:+.2f}")
        return a

    C1 = tab[(tab.feat == "rsi1") & (tab.split == "하위1/3") & (tab.Y == "Y1")].iloc[0]
    C2 = tab[(tab.feat == "ndown") & (tab.split == "상위1/3") & (tab.Y == "Y1")].iloc[0]
    C3 = tabe[(tabe.feat == "d_prev") & (tabe.Y == "Y1")].iloc[0]
    O1 = tab[(tab.feat == "kday") & (tab.split == "하위1/3") & (tab.Y == "Y2")].iloc[0]
    for C, pool, X, names, lab in ((C1, PR, XR, FEATS, "정규"), (C2, PR, XR, FEATS, "정규"), (C3, PE, XE, E1F, "E1")):
        say(f"\n후보 {C.feat} {'<=' if C.op == 'le' else '>'} {C.thr:.4g} Y1, 앞 방향 {'A 좋음' if C.sgn > 0 else 'A 나쁨'}, 앞 점수 {C.score:.2f}"
            f" (분석가 숫자와 대조: 앞 t {C.t1:+.2f}/통제 {C.ct1:+.2f}, 뒤 t {C.t2:+.2f}/통제 {C.ct2:+.2f})")
        detail(pool, X, names, C.feat, C.op, C.thr, "1", int(C.sgn), lab)

    # ------------------------------------------------------------------ [4] 우연 대조
    say(f"\n== [4] 우연 대조: 고르기(앞 점수 최고)→뒤 확인 절차 전체를 {NPERM}회씩 ==")
    rank = tab.sort_values("score", ascending=False)
    picks = rank[rank.score > 0].drop_duplicates("feat").head(2)
    real = dict(max_score=tab.score.max(), top_same2=picks.sgn.iloc[0] * picks.t2.iloc[0], top_same2_ct=picks.sgn.iloc[0] * picks.ct2.iloc[0],
                mean_same2=np.nanmean(tab.sgn * tab.t2), pos_same2=np.mean((tab.sgn * tab.t2).dropna() > 0),
                max_cons_y2=tab[tab.Y == "Y2"].apply(consist, axis=1).max(), max_cons_all=tab.apply(consist, axis=1).max())
    tab["cons"] = tab.apply(consist, axis=1)
    topc = tab.sort_values("cons", ascending=False).head(5)
    say("실제: 앞 최고 점수 {:.2f}, 고른 1위의 뒤 같은 방향 t {:+.2f}(통제 {:+.2f}), 76개 뒤 같은 방향 t 평균 {:+.2f}, 양수 {:.0f}%".format(
        real["max_score"], real["top_same2"], real["top_same2_ct"], real["mean_same2"], real["pos_same2"] * 100))
    say("실제 네 t 같은 방향 최소|t| 상위 5: " + " | ".join(f"{r.feat} {r.split} {r.Y} {r.cons:.2f}" for _, r in topc.iterrows()))
    for kind, nm in (("perm", "같은 날 안 특성 행 섞기"), ("gauss", "무작위 정규 특성 19개")):
        nl = null_runs(PR, XR, FEATS, ("하위1/3", "상위1/3"), kind, NPERM, 2)
        NT["n"] += NPERM * len(tab)
        say(f"[{nm}, 정규장 76시험 x {NPERM}회] 앞 최고 점수 중앙값 {nl.max_score.median():.2f}, 95% {nl.max_score.quantile(.95):.2f} → 실제 {real['max_score']:.2f} p={np.mean(nl.max_score >= real['max_score']):.3f}")
        say(f"    고른 1위의 뒤 같은 방향 t: 중앙값 {nl.top_same2.median():+.2f}, 90% {nl.top_same2.quantile(.9):+.2f} → 실제 {real['top_same2']:+.2f}"
            f" (순열이 실제보다 높거나 같은 비율 {np.mean(nl.top_same2 >= real['top_same2']):.2f})")
        say(f"    76개 뒤 같은 방향 t 평균: 순열 중앙값 {nl.mean_same2.median():+.2f}, 5~95% {nl.mean_same2.quantile(.05):+.2f}~{nl.mean_same2.quantile(.95):+.2f} → 실제 {real['mean_same2']:+.2f}")
        say(f"    네 t 같은 방향 최소|t| 최댓값(Y2 38시험): 중앙값 {nl.max_cons_y2.median():.2f}, 95% {nl.max_cons_y2.quantile(.95):.2f} → kday Y2 {consist(O1):.2f} p={np.mean(nl.max_cons_y2 >= consist(O1)):.3f}"
            f" | (76시험) 중앙값 {nl.max_cons_all.median():.2f}, 95% {nl.max_cons_all.quantile(.95):.2f} → 실제 최댓값 {real['max_cons_all']:.2f} p={np.mean(nl.max_cons_all >= real['max_cons_all']):.3f}")
    rank_e = tabe.sort_values("score", ascending=False)
    nle = null_runs(PE, XE, E1F, ("하위1/2",), "perm", NPERM, 1)
    NT["n"] += NPERM * len(tabe)
    say(f"[E1 같은 날 안 섞기, 16시험 x {NPERM}회] 앞 최고 점수 중앙값 {nle.max_score.median():.2f}, 95% {nle.max_score.quantile(.95):.2f} → 실제 {rank_e.score.max():.2f}"
        f" p={np.mean(nle.max_score >= rank_e.score.max()):.3f}")

    # ------------------------------------------------------------------ [5] kday Y2 관찰
    say(f"\n== [5] 관찰: kday <= {O1.thr:.4g}(그날 고점 뒤 15봉=45분 이내) 이면 Y2(60분 안 RSI 매도)가 적다 ==")
    say(f"분석가 점수 {O1.score:.2f}(앞 A 사건 {int(O1.y2nA1)}건 < 5라 분석가 규칙상 고를 수 없음). 사례(CRCL vs MRVL)를 본 뒤 찾은 것.")
    jk = FEATS.index("kday")
    detail(PR, XR, FEATS, "kday", "le", O1.thr, "2", -1, "정규 kday Y2")
    a = (XR[:, jk] <= O1.thr) & np.isfinite(XR[:, jk])
    nA, nB = int(a.sum()), int((~a).sum())
    kA, kB = int(PR.y["2"][a].sum()), int(PR.y["2"][~a].sum())
    from math import comb
    # 한쪽 피셔 정확 검정(A가 적을 확률)
    N, K = nA + nB, kA + kB
    pf = sum(comb(K, x) * comb(N - K, nA - x) for x in range(0, kA + 1)) / comb(N, nA)
    say(f"두 절반 합: A {kA}/{nA} = {kA / nA * 100:.1f}% vs B {kB}/{nB} = {kB / nB * 100:.1f}%, 한쪽 피셔 p={pf:.3f}(거래 독립 가정, 날짜 군집 무시)")
    for h in ("앞", "뒤"):
        for nm, kk in (("개장30분 안", PR.df.open30.values), ("나머지", ~PR.df.open30.values)):
            m = (PR.half == h) & kk
            say(f"  [{h}·{nm}] A {int(PR.y['2'][m & a].sum())}/{int((m & a).sum())} vs B {int(PR.y['2'][m & ~a].sum())}/{int((m & ~a).sum())}")
    # Y1 쪽(같은 특성)
    o1y1 = tab[(tab.feat == "kday") & (tab.split == "하위1/3") & (tab.Y == "Y1")].iloc[0]
    say(f"같은 분할의 Y1: 앞 t {o1y1.t1:+.2f}/통제 {o1y1.ct1:+.2f}, 뒤 t {o1y1.t2:+.2f}/통제 {o1y1.ct2:+.2f}")

    # ------------------------------------------------------------------ [5b][6] 5분봉 앞 기간
    say("\n== [5b·6] 5분봉 앞 기간(7/14~9/8, EMA36·RSI9, 창 12·6·3봉) — 단위가 달라 보조 ==")
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        E = P.early_check([])
    E = E.reset_index(drop=True)
    E["half"] = "앞"
    E["tailf"] = False
    say(f"정규장 {len(E)}건 {E.day.nunique()}일, Y2(60분=12봉 안 RSI 매도) {int(E.y2.sum())}건 {E.y2.mean() * 100:.1f}%")
    PEa = Pool(E, ["open30", "atrt"])

    def early_test(feat, op, q, ylab, sgn, sub=None, ctrl_pool=None, label=""):
        pool = ctrl_pool or PEa
        x = pool.df[feat].values.astype(float)
        fin = np.isfinite(x)
        if sub is None:
            sub = np.ones(len(x), bool)
        if q is None:
            qq = np.quantile(x[fin & sub], 1 / 3 if op == "le" else 2 / 3)
        elif isinstance(q, str):
            qq = np.quantile(x[fin & sub], 0.5)
        else:
            qq = q
        a = ((x <= qq) if op == "le" else (x > qq)) & fin
        dd, t, cdd, ct, nd = pool.diff(a, sub & fin, ylab)
        NT["n"] += 1
        same = sgn * t
        say(f"  {label}{feat} {'<=' if op == 'le' else '>'} {qq:.4g} Y{ylab}: A {int((a & sub).sum())} vs B {int((~a & sub & fin).sum())} | "
            f"날짜차 {pct(dd) if ylab == '1' else f'{100 * dd:+.1f}%p'} t={t:+.2f} 통제 t={ct:+.2f} ({nd}일) → 3분봉 앞 방향 기준 {'같은 방향' if same > 0 else '반대 방향'}")
        return t, ct

    tk = early_test("kday", "le", None, "2", -1, label="kday Y2 관찰 ")
    early_test("rsi1", "le", None, "1", -1, label="후보1 ")
    early_test("ndown", "gt", None, "1", -1, label="후보2 ")
    E1e = E[E.spike >= P.SPIKE].reset_index(drop=True)
    PEe = Pool(E1e, ["open30"])
    say(f"E1(그날 고점 >= 전날 종가 +5%) {len(E1e)}건 {E1e.day.nunique()}일")
    early_test("d_prev", "le", "med", "1", 1, ctrl_pool=PEe, label="후보3 E1 ")
    # E1 사례 특성(LITE vs COHR) — 3분봉 앞 방향
    for feat in ("retr", "vol5", "o_line"):
        r = None
        if feat in E1F:
            r = tabe[(tabe.feat == feat) & (tabe.Y == "Y1")].iloc[0]
        else:
            # o_line은 분석가 E1 시험에 없음 → 여기서 새로 2개(Y1·Y2) 시험, 문턱·방향은 3분봉 앞 절반 중앙값
            tb = run_tests(PE, PE.df[["o_line"]].values.astype(float), ["o_line"], ("하위1/2",))
            NT["n"] += len(tb)
            r = tb[tb.Y == "Y1"].iloc[0]
            r2 = tb[tb.Y == "Y2"].iloc[0]
            say(f"  [3분봉 E1 새 시험] o_line <= {r.thr:.4g} Y1: 앞 t {r.t1:+.2f}/통제 {r.ct1:+.2f}, 뒤 t {r.t2:+.2f}/통제 {r.ct2:+.2f} | "
                f"Y2: 앞 t {r2.t1:+.2f}, 뒤 t {r2.t2:+.2f} (앞 방향 {'A 좋음' if r.sgn > 0 else 'A 나쁨'})")
        say(f"  [3분봉 E1] {feat} <= {r.thr:.4g} Y1 앞 방향 {'A 좋음' if r.sgn > 0 else 'A 나쁨'}: 앞 t {r.t1:+.2f}/통제 {r.ct1:+.2f}, 뒤 t {r.t2:+.2f}/통제 {r.ct2:+.2f}")
        early_test(feat, "le", "med", "1", int(r.sgn), ctrl_pool=PEe, label="    5분봉 E1 ")
    # 뒤에만 보인 gap·spike(가격 기준이라 같은 문턱)
    g1 = tab[(tab.feat == "gap") & (tab.split == "상위1/3") & (tab.Y == "Y1")].iloc[0]
    s1 = tab[(tab.feat == "spike") & (tab.split == "상위1/3") & (tab.Y == "Y1")].iloc[0]
    say(f"뒤에만 보인 것: gap > {g1.thr:.4g} 앞 t {g1.t1:+.2f} 뒤 t {g1.t2:+.2f}/통제 {g1.ct2:+.2f} | spike > {s1.thr:.4g} 앞 t {s1.t1:+.2f} 뒤 t {s1.t2:+.2f}/통제 {s1.ct2:+.2f}")
    detail(PR, XR, FEATS, "gap", "gt", g1.thr, "1", -1, "정규 gap 뒤만(나쁨 방향)")
    early_test("gap", "gt", g1.thr, "1", -1, label="  5분봉 같은 문턱 ")
    early_test("spike", "gt", s1.thr, "1", -1, label="  5분봉 같은 문턱 ")

    # ------------------------------------------------------------------ [5c] kday Y2를 더 깨 본다
    say("\n== [5c] kday Y2 추가 반박: 분 단위 문턱, 더 촘촘한 시간대 통제, 회귀(날짜 고정효과·날짜 군집 표준오차), 순열, 쏠림 ==")
    say("3분봉에서 네 t 일관성 상위 2개(conc 하위1/3 Y2, kday 하위1/3 Y2)를 똑같이 5분봉 앞 기간에 가져가 본다(고르기 기준을 사례가 아닌 일관성으로 바꾼 대조).")
    early_test("conc", "le", None, "2", -1, label="  일관성 1위 ")
    t_min, ct_min = early_test("kday", "le", 9, "2", -1, label="  분 단위 문턱(47분÷5분≈9봉) ")
    early_test("kday", "le", None, "1", -1, label="  같은 분할 Y1 ")

    def tbucket(tm):
        return np.select([tm < 6, tm < 15, tm < 30, tm < 90], [0, 1, 2, 3], 4)

    def ols_cluster(y, A, ctrls, day):
        """y ~ A + 통제 + 날짜 고정효과, 날짜 군집 표준오차. A 계수와 t."""
        days, dcode = np.unique(day.astype(str), return_inverse=True)
        Dm = np.eye(len(days))[dcode][:, 1:]
        X = np.column_stack([np.ones(len(y)), A.astype(float)] + ctrls + [Dm])
        XtX = np.linalg.pinv(X.T @ X)
        beta = XtX @ X.T @ y
        u = y - X @ beta
        G = len(days)
        meat = np.zeros((X.shape[1], X.shape[1]))
        for g in range(G):
            s = X[dcode == g].T @ u[dcode == g]
            meat += np.outer(s, s)
        n, k = X.shape
        V = XtX @ meat @ XtX * G / (G - 1) * (n - 1) / max(n - k, 1)
        return beta[1], beta[1] / np.sqrt(V[1, 1]), G

    def kday_reg(df, thr, label):
        x = df.copy()
        x = x[x.kday.notna() & x.dd_day.notna() & x.rsi1.notna() & x.spike.notna() & x.tmin.notna()]
        tb = tbucket(x.tmin.values)
        ctr = [(tb == k).astype(float) for k in (1, 2, 3, 4)] + [(x.atrt.values == k).astype(float) for k in (1, 2)]
        ctr += [x.dd_day.values, x.rsi1.values / 100, x.spike.values]
        A = (x.kday.values <= thr)
        b, tt, G = ols_cluster(x.y2.values.astype(float), A, ctr, x.day.values)
        NT["n"] += 1
        say(f"  [{label}] 회귀 Y2 ~ kday<={thr:.4g} + 개장 뒤 시간 5칸 + atr 3칸 + dd_day + rsi1 + spike + 날짜 고정효과: 계수 {100 * b:+.1f}%p t={tt:+.2f} ({len(x)}건 {G}일)")
        # 같은 날·시간 5칸·atr 칸 통제 t
        x["tb"] = tb
        pool = Pool(x.assign(half="앞"), ["tb", "atrt"])
        a2 = A & np.isfinite(x.kday.values)
        _, t0, _, ct, nd = pool.diff(a2, np.ones(len(x), bool), "2")
        say(f"      같은 날·시간 5칸·atr 칸 안 비교: 원 t={t0:+.2f} 통제 t={ct:+.2f} ({nd}일) | Y2 A {int(x.y2.values[A].sum())}/{int(A.sum())}={x.y2.values[A].mean() * 100:.1f}% vs B {int(x.y2.values[~A].sum())}/{int((~A).sum())}={x.y2.values[~A].mean() * 100:.1f}%")
        return pool, a2

    qk = np.quantile(E.kday.dropna(), 1 / 3)
    poolE, aE = kday_reg(E, qk, f"5분봉 앞 기간, 문턱 하위1/3 {qk:.0f}봉")
    kday_reg(E, 9, "5분봉 앞 기간, 분 단위 문턱 9봉")
    kday_reg(reg, O1.thr, "3분봉 9/10~10/8 두 절반 합")
    # 순열(같은 날 안에서 kday 섞기) — 5분봉 앞 기간 통제 t
    xk = poolE.df.kday.values.astype(float)
    groups = [np.where(poolE.dc == k)[0] for k in range(poolE.D)]
    _, t_obs, _, ct_obs, _ = poolE.diff(aE, np.ones(len(xk), bool), "2")
    nul = []
    for _ in range(2000):
        perm = np.arange(len(xk))
        for g in groups:
            perm[g] = RNG.permutation(g)
        ap = (xk[perm] <= qk)
        _, tp, _, ctp, _ = poolE.diff(ap, np.ones(len(xk), bool), "2")
        nul.append((tp, ctp))
    nul = np.array(nul)
    NT["n"] += 2000
    say(f"  같은 날 안 kday 섞기 2000회(5분봉, 시간 5칸 통제): 원 t <= {t_obs:+.2f} 비율 {np.mean(nul[:, 0] <= t_obs):.4f}, 통제 t <= {ct_obs:+.2f} 비율 {np.mean(nul[:, 1] <= ct_obs):.4f}")
    # 쏠림: 5분봉에서 규칙에 가장 불리한 1종목·1일·5일 빼기
    dfE = poolE.df
    for col in ("sym", "day"):
        best = None
        for v in dfE[col].unique():
            k = dfE[col].values != v
            dd_, t_, _, ct_, _ = poolE.diff(aE, k, "2")
            if np.isfinite(dd_) and (best is None or -dd_ < best[0]):
                best = (-dd_, v, t_, ct_)
        say(f"  5분봉 불리한 1{'종목' if col == 'sym' else '일'} 빼기: -{best[1]} t={best[2]:+.2f} 통제 t={best[3]:+.2f}")
    # 날짜별 차이에서 가장 큰 5일(규칙 방향 기여 큰 순) 빼기
    dd_by = []
    for k in range(poolE.D):
        m = poolE.dc == k
        if (m & aE).any() and (m & ~aE).any():
            dd_by.append((poolE.y["2"][m & aE].mean() - poolE.y["2"][m & ~aE].mean(), k))
    dd_by.sort()
    drop5 = {k for _, k in dd_by[:5]}
    keep = ~np.isin(poolE.dc, list(drop5))
    _, t5, _, ct5, nd5 = poolE.diff(aE, keep, "2")
    say(f"  5분봉 규칙 방향 기여가 큰 5일 빼기: t={t5:+.2f} 통제 t={ct5:+.2f} ({nd5}일)")
    # 시간대별
    for nm, kk in (("개장 6분 안", dfE.tmin.values < 6), ("6~30분", (dfE.tmin.values >= 6) & (dfE.tmin.values < 30)), ("30분 뒤", dfE.tmin.values >= 30)):
        _, tt, _, _, _ = poolE.diff(aE, kk, "2")
        say(f"  5분봉 {nm}: A {int(poolE.y['2'][kk & aE].sum())}/{int((kk & aE).sum())} vs B {int(poolE.y['2'][kk & ~aE].sum())}/{int((kk & ~aE).sum())} t={tt:+.2f}")
    # RSI 기억 때문인가, 실제 가격 반등 차이인가: 60분 뒤 종가 수익 r60(RSI와 무관한 가격 결과)으로 같은 비교
    say("  RSI 기억(고점 직후 큰 음봉이 RSI 평균 손실에 남음) 때문인지 보려고 60분 뒤 종가 수익 r60으로 같은 분할을 잰다(설명용, NaN 제외):")
    for nm, pool, a_ in (("5분봉 앞 기간", poolE, aE),):
        ok = np.isfinite(pool.y["r60"])
        for tn, kk in (("전체", np.ones(len(a_), bool)), ("개장 6분 안", pool.df.tmin.values < 6), ("6분 뒤", pool.df.tmin.values >= 6)):
            m = ok & kk
            dd, tt, _, ct, nd = pool.diff(a_, m, "r60")
            say(f"    [{nm}·{tn}] r60 A {pct(np.nanmean(pool.y['r60'][m & a_]))} vs B {pct(np.nanmean(pool.y['r60'][m & ~a_]))} 날짜차 {pct(dd)} t={tt:+.2f} 통제 t={ct:+.2f} ({nd}일)")
    jk2 = FEATS.index("kday")
    a3 = (XR[:, jk2] <= O1.thr) & np.isfinite(XR[:, jk2])
    ok3 = np.isfinite(PR.y["r60"])
    for h in ("앞", "뒤"):
        for tn, kk in (("전체", np.ones(len(a3), bool)), ("개장 6분 안", PR.df.tmin.values < 6), ("6분 뒤", PR.df.tmin.values >= 6)):
            m = ok3 & kk & (PR.half == h)
            dd, tt, _, ct, nd = PR.diff(a3, m, "r60")
            y2a, y2b = PR.y["2"][m & a3], PR.y["2"][m & ~a3]
            say(f"    [3분봉 {h}·{tn}] r60 날짜차 {pct(dd)} t={tt:+.2f} 통제 t={ct:+.2f} ({nd}일) | Y2 A {int(y2a.sum())}/{len(y2a)} vs B {int(y2b.sum())}/{len(y2b)}")

    # ------------------------------------------------------------------ [7] 10/9 사례
    say("\n== [7] 10/9 사례(내 코드로 다시 계산, 10:15 봉까지) ==")
    C = pd.read_csv(os.path.join(RES, "pat_shape_1009.csv"))
    C["ts"] = pd.to_datetime(C.t, utc=True).dt.tz_convert("America/New_York")
    pick = C[C.sym.isin(["CRCL", "MRVL", "LITE", "COHR"]) & (C.ses == "정규")]
    for _, x in pick.iterrows():
        f = my_feats(x.sym, [(x.ts, x.ent)])[0]
        bad = [k for k in ("kday", "rsi1", "ndown", "o_line", "spike", "retr", "d_prev", "vol5", "gap", "tmin")
               if not (pd.isna(f[k]) and pd.isna(x[k])) and not (abs(f[k] - x[k]) <= 1e-9 * max(1, abs(x[k])))]
        say(f"{x.sym} {x.ts.strftime('%H:%M')} 매수 {x.ent:.2f}(재계산 {f['ent_mine']:.2f}) → {x.why}({x.bars}봉), 10:15 평가 {pct(x.mtm)} | "
            f"rsi1 {f['rsi1']:.2f} ndown {f['ndown']} kday {f['kday']} spike {pct(f['spike'])} d_prev {pct(f['d_prev'])} retr {f['retr']:.3f} "
            f"vol5 {f['vol5']:.2f} o_line {pct(f['o_line'])} gap {pct(f['gap'])} | 직전 1분 종가/매수가 {pct(f['prv1m_mine'])}"
            f"{'(1분 꼬리 체결)' if f['prv1m_mine'] > 0.01 else ''} | 분석가 값과 다른 특성: {', '.join(bad) if bad else '없음'}")
    regc = C[C.ses == "정규"]
    rho = {k: regc[k].corr(regc.mtm, method="spearman") for k in FEATS if regc[k].notna().sum() >= 8}
    say(f"10/9 정규장 {len(regc)}건 특성 vs 10:15 평가 스피어만 |ρ| 최대: " + ", ".join(f"{k} {v:+.2f}" for k, v in sorted(rho.items(), key=lambda z: -abs(z[1]))[:3]))
    say(f"10/9 정규장 매수 중 gap > {g1.thr:.4g}: {int((regc.gap > g1.thr).sum())}건 10:15 평가 평균 {pct(regc[regc.gap > g1.thr].mtm.mean())} vs 나머지 {pct(regc[~(regc.gap > g1.thr)].mtm.mean())}")
    pA = kA / nA
    pB = kB / nB
    say(f"kday 규칙을 10/9에 넣으면: CRCL kday 30 > {O1.thr:.4g} → B(9/10~10/8 Y2 기준율 {pB * 100:.1f}%), MRVL 2·LITE 4·COHR 4 → A(기준율 {pA * 100:.1f}%)")

    say(f"\n시험 수(이 검증): 재현 정규 {len(tab)} + E1 {len(tabe)} + 순열·무작위 {3 * NPERM}회 x 76/76/16시험 + 5분봉 앞 기간·E1 새 시험 = 합계 {NT['n']}회 계산")
    say(f"경과 {time.time() - t0:.0f}초")
    with open(os.path.join(RES, "verify_pat_shape.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
