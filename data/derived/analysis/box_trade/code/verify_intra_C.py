"""intra_C 반박 검증 — best_rule.definition만 보고 독립 재구현(2026-10-09).

규칙(definition 그대로): 봉 = 3분(확인) / 5분(고르기), 프리·애프터 포함 연속 봉.
  박스 = 직전 nb봉(60분) 고가 최고·저가 최저(신호 봉 제외). 신호: close[k] > boxHi[k] & close[k-1] <= boxHi[k-1].
  거래량: 신호 봉 정규장, v[k] >= 2 x 같은 5분 칸 지난 10거래일 평균(오늘 제외, 5일 이상) x (봉분/5).
  매수: k+1 시가, 매수 봉 09:30~11:00 ET. 한 종목 한 포지션, 매도 봉 다음 봉부터 재매수.
  매도: 목표 지정가 = 매수가 + (boxHi-boxLo) (매수 다음 봉부터, max(시가, 지정가)),
        손절 = 종가 < (boxHi+boxLo)/2 → 다음 깨어 있는 봉(ET 04:00~11:00, 19:00~20:00) 시가,
        정리 = 다음 한국 08:00 이후 첫 봉 시가. 같은 봉은 시가 손절 > 지정가 순.
구현 방식: 종목별 봉 단위 상태 기계(numba). 연구자 코드의 벡터 계산·clean() 단계는 쓰지 않는다(정의에 없음).
기준: RA = 같은 시각 다른 주식(전체137) 전부, RB = 같은 종목 기간 안 다른 날 같은 시각 전부, RD = 같은 종목 같은 날
      09:30~11:00 모든 봉(과제 지정 기준, 경로 편향 있음), RDa = 같은 날 매수 봉 이후 봉만, H8 = 같은 매수 봉에서 8시까지 보유.
      기준 거래는 신호의 목표%·손절%를 그대로 쓴다(H8 제외).
사용: py verify_intra_C.py → results/verify_intra_C.txt
"""
import os, sys, time
import numpy as np, pandas as pd
from numba import njit

CODE = r"C:\Users\geunt\OneDrive\Desktop\주식\data\derived\analysis\dip_reason\code"
sys.path.insert(0, CODE)
from fetch import universe, user_universe  # noqa: E402

SCR = r"C:/Users/geunt/AppData/Local/Temp/claude/C--Users-geunt-OneDrive-Desktop---/4e686c73-74be-4337-8ccc-2f6136beaf7d/scratchpad"
RAW = SCR + "/dip/raw"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "results", "verify_intra_C.txt")

U78 = set(user_universe()["합집합"])
A137 = U78 | set(universe())
SYMS = sorted(s for s in A137 if os.path.exists(f"{RAW}/5m_{s}.parquet") and os.path.exists(f"{RAW}/1m_{s}.parquet"))
PER = {"early5": (5, 20260714, 20260908), "late3": (3, 20260910, 20261006), "late5": (5, 20260910, 20261006)}


def load(s, mins):
    if mins == 5:
        b = pd.read_parquet(f"{RAW}/5m_{s}.parquet")
    else:
        b = pd.read_parquet(f"{RAW}/1m_{s}.parquet")
        b = b.resample("3min").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
    b = b.dropna(subset=["Open", "High", "Low", "Close"])
    return b[b.High > 0]


def vol_ref(s):
    """{(날짜, 5분 칸): 지난 10거래일(오늘 제외) 평균, 5일 이상}"""
    f = pd.read_parquet(f"{RAW}/5m_{s}.parquet")
    m = f.index.hour * 60 + f.index.minute
    f = f[(m >= 570) & (m < 960)]
    d = f.index.year * 10000 + f.index.month * 100 + f.index.day
    tab = pd.DataFrame({"d": d, "m": f.index.hour * 60 + f.index.minute, "v": f.Volume.values}).pivot_table(
        index="d", columns="m", values="v", aggfunc="sum").sort_index()
    days = list(tab.index)
    out = {}
    arr = tab.values
    for i, dd in enumerate(days):
        w = arr[max(0, i - 10):i]
        cnt = np.sum(np.isfinite(w), axis=0)
        mean = np.where(cnt >= 5, np.nansum(w, axis=0) / np.maximum(cnt, 1), np.nan)
        out[dd] = dict(zip(tab.columns, mean))
    return out


def build(cfg):
    mins = PER[cfg][0]
    P = []
    for s in SYMS:
        b = load(s, mins)
        idx = b.index
        mod = np.asarray(idx.hour * 60 + idx.minute)
        dk = np.asarray(idx.year * 10000 + idx.month * 100 + idx.day)
        kst = idx.tz_convert("Asia/Seoul")
        t8 = kst.normalize() + pd.Timedelta(hours=8)
        dead = t8.where(kst < t8, t8 + pd.Timedelta(days=1))
        J = np.searchsorted(idx.as_unit("ns").asi8, dead.as_unit("ns").asi8, "left")
        J = np.where(J < len(idx), J, -1)
        vr = vol_ref(s)
        ref = np.array([vr.get(d, {}).get(m // 5 * 5, np.nan) for d, m in zip(dk, mod)]) * (mins / 5)
        P.append(dict(sym=s, o=b.Open.values.astype(float), h=b.High.values.astype(float), l=b.Low.values.astype(float),
                      c=b.Close.values.astype(float), v=b.Volume.values.astype(float), mod=mod, dk=dk, ts=idx.as_unit("s").asi8,
                      J=J, ref=ref))
    return P


@njit(cache=True)
def box(h, l, nb):
    n = len(h)
    H = np.full(n, np.nan)
    L = np.full(n, np.nan)
    for k in range(nb, n):
        H[k] = h[k - nb:k].max()
        L[k] = l[k - nb:k].min()
    return H, L


@njit(cache=True)
def exit_one(o, h, c, awake, J, e, T, S):
    """매수 봉 e(시가). 반환 (매도 봉, 매도가)."""
    je = J[e]
    pend = c[e] < S
    i = e + 1
    while True:
        if i >= je:
            return je, o[je]
        if pend and awake[i]:
            return i, o[i]
        if h[i] >= T:
            return i, max(o[i], T)
        if c[i] < S:
            pend = True
        i += 1


@njit(cache=True)
def run_rule(o, h, c, awake, J, entry_ok, T, S):
    n = len(o)
    E = []
    X = []
    R = []
    i = 0
    while i < n:
        if entry_ok[i]:
            x, px = exit_one(o, h, c, awake, J, i, T[i], S[i])
            E.append(i)
            X.append(x)
            R.append(px / o[i] - 1)
            i = x + 1
        else:
            i += 1
    return np.array(E, np.int64), np.array(X, np.int64), np.array(R)


@njit(cache=True)
def batch(o, h, c, awake, J, ent, tp, sp):
    r = np.empty(len(ent))
    for k in range(len(ent)):
        e = ent[k]
        x, px = exit_one(o, h, c, awake, J, e, o[e] * (1 + tp[k]), o[e] * (1 - sp[k]))
        r[k] = px / o[e] - 1
    return r


class Data:
    def __init__(self, cfg):
        self.cfg = cfg
        self.mins, lo, hi = PER[cfg]
        P = build(cfg)
        off = np.cumsum([0] + [len(p["o"]) for p in P])
        cat = lambda k: np.concatenate([p[k] for p in P])
        self.o, self.h, self.l, self.c, self.v = (cat(k) for k in "ohlcv")
        self.mod, self.dk, self.ts, self.ref = cat("mod"), cat("dk"), cat("ts"), cat("ref")
        self.sid = np.concatenate([np.full(len(p["o"]), i) for i, p in enumerate(P)])
        self.J = np.concatenate([np.where(p["J"] >= 0, p["J"] + off[i], -1) for i, p in enumerate(P)])
        self.off = off
        self.syms = np.array(SYMS)
        n = len(self.o)
        self.awake = ((self.mod >= 240) & (self.mod < 660)) | ((self.mod >= 1140) & (self.mod < 1200))
        reg = (self.mod >= 570) & (self.mod < 960)
        same = np.r_[False, (self.sid[1:] == self.sid[:-1]) & (self.dk[1:] == self.dk[:-1])]
        self.reg = reg
        # 매수 가능 봉: 09:30~11:00, 앞 봉이 같은 날 정규장, 기간 안, 8시 정리 봉 있음
        self.elig = (self.mod >= 570) & (self.mod < 660) & same & np.r_[False, reg[:-1]] & \
                    (self.dk >= lo) & (self.dk <= hi) & (self.J > np.arange(n))
        self.boxes = {}

    def box(self, nb):
        if nb not in self.boxes:
            H = np.full(len(self.o), np.nan)
            L = np.full(len(self.o), np.nan)
            for i in range(len(SYMS)):
                a, b = self.off[i], self.off[i + 1]
                H[a:b], L[a:b] = box(self.h[a:b], self.l[a:b], nb)
            self.boxes[nb] = (H, L)
        return self.boxes[nb]

    def trades(self, nmin=60, volk=2.0, tmult=1.0, sfrac=0.5, win=(570, 660), vol_on=True):
        nb = nmin // self.mins
        H, L = self.box(nb)
        c = self.c
        up = c > H
        firstbar = np.r_[True, self.sid[1:] != self.sid[:-1]]
        cross = up & np.r_[False, ~up[:-1]] & ~firstbar & np.isfinite(H)
        volok = self.reg & np.isfinite(self.ref) & (self.ref > 0) & (self.v >= volk * self.ref) if vol_on else self.reg
        sig = cross & volok
        ok = self.elig & np.r_[False, sig[:-1]] & (self.mod >= win[0]) & (self.mod < win[1])
        Hp, Lp = np.r_[np.nan, H[:-1]], np.r_[np.nan, L[:-1]]
        T = self.o + tmult * (Hp - Lp)
        S = Lp + sfrac * (Hp - Lp)
        T = np.where(ok, T, np.inf)
        S = np.where(ok, S, -np.inf)
        E, X, R = run_rule(self.o, self.h, self.c, self.awake, self.J, ok, T, S)
        df = pd.DataFrame(dict(sym=self.syms[self.sid[E]], day=self.dk[E], mod=self.mod[E], ts=self.ts[E], e=E, x=X, ret=R,
                               tp=tmult * (Hp[E] - Lp[E]) / self.o[E], sp=(self.o[E] - S[E]) / self.o[E],
                               boxw=(Hp[E] - Lp[E]) / Lp[E]))
        return df

    def baselines(self, df, which=("RA", "RB", "RD", "RDa", "H8")):
        """각 거래 행에 기준 평균 붙임. 주식만(3배 ETF 없음)."""
        el = np.flatnonzero(self.elig)
        by_ts, by_sm, by_sd = {}, {}, {}
        for i in el:
            by_ts.setdefault(self.ts[i], []).append(i)
            by_sm.setdefault((self.sid[i], self.mod[i]), []).append(i)
            by_sd.setdefault((self.sid[i], self.dk[i]), []).append(i)
        out = {k: np.full(len(df), np.nan) for k in which}
        owners = {k: [] for k in which}
        ents = {k: [] for k in which}
        for k, (e, tp, sp) in enumerate(zip(df.e.values, df.tp.values, df.sp.values)):
            s, d = self.sid[e], self.dk[e]
            pools = {"RA": [i for i in by_ts.get(self.ts[e], []) if self.sid[i] != s],
                     "RB": [i for i in by_sm.get((s, self.mod[e]), []) if self.dk[i] != d],
                     "RD": [i for i in by_sd.get((s, d), []) if i != e],
                     "RDa": [i for i in by_sd.get((s, d), []) if i > e]}
            for w in which:
                if w == "H8":
                    continue
                p = pools[w]
                owners[w].extend([k] * len(p))
                ents[w].extend(p)
        tpa, spa = df.tp.values, df.sp.values
        for w in which:
            if w == "H8":
                e = df.e.values
                xr = self.o[self.J[e]] / self.o[e] - 1
                out[w] = xr
                continue
            ow = np.asarray(owners[w], np.int64)
            en = np.asarray(ents[w], np.int64)
            if not len(en):
                continue
            r = batch(self.o, self.h, self.c, self.awake, self.J, en, tpa[ow], spa[ow])
            ct = np.bincount(ow, minlength=len(df))
            out[w] = np.where(ct > 0, np.bincount(ow, r, minlength=len(df)) / np.maximum(ct, 1), np.nan)
        for w in which:
            df[w] = out[w]
        return df


def tday(x, day):
    s = pd.Series(x).groupby(np.asarray(day)).sum()
    if len(s) < 3 or s.std(ddof=1) == 0:
        return np.nan
    return s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))


def summ(df, base=("RA", "RB", "RD", "RDa", "H8")):
    if not len(df):
        return "n=0"
    s = (f"n={len(df):>4} 날={df.day.nunique():>2} 평균={df.ret.mean():+.2%} 승률={(df.ret > 0).mean():.0%} "
         f"최악={df.ret.min():+.1%} 합={df.ret.sum():+.2f} t={tday(df.ret, df.day):+.2f}")
    for b in base:
        if b in df:
            m = df[b].notna()
            ex = df.ret[m] - df[b][m]
            s += f" | {b}={df[b][m].mean():+.2%} 초과={ex.mean():+.2%}(t{tday(ex, df.day[m]):+.2f})"
    return s


def ex_mean(df, b):
    m = df[b].notna()
    return (df.ret[m] - df[b][m]).mean(), tday(df.ret[m] - df[b][m], df.day[m])


def main():
    t0 = time.time()
    L = []
    w = lambda x="": (L.append(x), print(x, flush=True))
    w("intra_C 반박 검증 (verify_intra_C.py, 2026-10-09) — best_rule.definition만 보고 독립 재구현")
    w(__doc__.split("사용:")[0].strip())
    w(f"종목 파일 있는 전체 {len(SYMS)}개, 사용자78 중 {len(U78 & set(SYMS))}개")
    w("")
    D = {cfg: Data(cfg) for cfg in PER}
    w(f"[데이터 적재 {time.time() - t0:.0f}s]")
    UN = {"사용자78": U78, "전체137": A137}

    # 1) 재현
    w("== 1) 최선안 재현 (C_N60_거래량 + X6, 정규 09:30~11:00 매수)")
    BEST = {}
    for cfg, d in D.items():
        df = d.baselines(d.trades())
        BEST[cfg] = df
        for un, us in UN.items():
            w(f"  {cfg:<6} {un:<5} {summ(df[df.sym.isin(us)])}")
    w("  연구자 보고: early5 사용자78 n=258 +0.90% 승률69% 최악-5.7% 합2.33 t3.48 | late3 사용자78 n=270 +0.25% 54% -5.0% 0.69 t1.48")
    w("               early5 전체137 n=484 +0.73% | late3 전체137 n=522 +0.25% | late5 사용자78 n=163 +0.37%")
    w("")

    # 1b) 연구자 거래와 하나씩 대조
    w("== 1b) 연구자 코드(intra_C.prepare, 캐시 npz)의 거래 목록과 대조")
    try:
        sys.path.insert(0, HERE)
        import intra_C as IC
        for cfg in ("early5", "late3"):
            G = IC.prepare(cfg)
            XI, R, TP, SP = G["X6"][(60, "i")]
            tr = IC.chain(G["SIG"]["C_N60_거래량"][1] & G["elig"]["정규"], XI)
            theirs = pd.DataFrame(dict(sym=np.asarray(IC.SYMS)[G["sid"][tr]], ts=G["tk"][tr], ret_t=R[tr]))
            theirs = theirs[theirs.sym.isin(A137)]
            mine = BEST[cfg][["sym", "ts", "ret"]]
            m = theirs.merge(mine, on=["sym", "ts"], how="outer", indicator=True)
            both = m[m._merge == "both"]
            diff = (both.ret - both.ret_t).abs()
            w(f"  {cfg}: 연구자 {len(theirs)}건 / 내 {len(mine)}건 / 같은 매수 봉 {len(both)}건 / 연구자만 {int((m._merge == 'left_only').sum())} "
              f"/ 나만 {int((m._merge == 'right_only').sum())} | 같은 봉 수익 차 > 0.01%p {int((diff > 1e-4).sum())}건, 최대 {diff.max():.4%}")
            fmt = lambda t: pd.Timestamp(int(t), unit="s", tz="UTC").tz_convert("America/New_York").strftime("%m-%d %H:%M")
            xt = pd.DataFrame(dict(sym=np.asarray(IC.SYMS)[G["sid"][tr]], ts=G["tk"][tr], xt=G["tk"][XI[tr]]))
            bd = both.merge(xt, on=["sym", "ts"]).merge(BEST[cfg][["sym", "ts", "x"]], on=["sym", "ts"])
            for _, r in bd[(bd.ret - bd.ret_t).abs() > 1e-4].iterrows():
                w(f"     수익 다름 {r.sym} {fmt(r.ts)} 내 {r.ret:+.2%}(매도 {fmt(D[cfg].ts[r.x])}) / 연구자 {r.ret_t:+.2%}(매도 {fmt(r.xt)})")
            w(f"     같은 매수 봉 공통 {len(both)}건 평균 내 {both.ret.mean():+.3%} / 연구자 {both.ret_t.mean():+.3%}")
            for side in ("left_only", "right_only"):
                x = m[m._merge == side].head(4)
                for _, r in x.iterrows():
                    w(f"     {'연구자만' if side == 'left_only' else '나만'} {r.sym} {pd.Timestamp(int(r.ts), unit='s', tz='UTC').tz_convert('America/New_York'):%m-%d %H:%M} "
                      f"{(r.ret_t if side == 'left_only' else r.ret):+.2%}")
            del G
    except Exception as ex:
        w(f"  대조 실패: {ex!r}")
    w("")

    # 2) 미래 참조·데이터 점검
    w("== 2) 데이터·미래 참조 점검")
    for cfg, d in D.items():
        df = BEST[cfg]
        jump = []
        for i in range(len(SYMS)):
            a, b = d.off[i], d.off[i + 1]
            c = d.c[a:b]
            r = c[1:] / c[:-1] - 1
            if np.any((r < -0.3) | (r > 0.5)):
                jump.append(SYMS[i])
        w(f"  {cfg}: 봉 사이 -30%/+50% 넘는 튐 종목 {jump[:10]} | 박스 폭 > 20%인 거래 {int((df.boxw > 0.2).sum())}건 "
          f"| 목표% 중앙 {df.tp.median():.2%}, 손절% 중앙 {df.sp.median():.2%}, 손절%<0(매수가가 박스 중간 아래) {int((df.sp < 0).sum())}건")
        rs = df.x.values - df.e.values
        w(f"           매도까지 봉 수 중앙 {np.median(rs):.0f}, 매수 봉 다음 봉에 판 거래 {np.mean(rs == 1):.0%}")
    # 같은 봉 종가 매수(미래 참조) 넣으면 얼마나 좋아지나 — 미래 참조가 있으면 연구자 숫자가 이쪽에 가까워야 함
    w("  참고(일부러 미래 참조): 신호 봉 종가에 산다고 치면")
    for cfg in ("early5", "late3"):
        d = D[cfg]
        df = BEST[cfg]
        e = df.e.values
        alt = (df.ret.values + 1) * d.o[e] / d.c[e - 1] - 1
        w(f"    {cfg} 사용자78 평균 {alt[df.sym.isin(U78).values].mean():+.2%} (실제 규칙 {df[df.sym.isin(U78)].ret.mean():+.2%})")
    w("")

    # 3) 이웃 매개변수
    w("== 3) 이웃 매개변수 (사용자78 / 전체137). 칸 = n 평균 초과RA(t) 초과RB(t)")
    VAR = [("최선 N60 거래량2 목표1 손절중간", {}),
           ("N30", dict(nmin=30)), ("N45", dict(nmin=45)), ("N90", dict(nmin=90)), ("N120", dict(nmin=120)),
           ("거래량1.5", dict(volk=1.5)), ("거래량3", dict(volk=3.0)), ("거래량 없음", dict(vol_on=False)),
           ("목표0.75", dict(tmult=0.75)), ("목표1.5", dict(tmult=1.5)),
           ("손절 하단+25%", dict(sfrac=0.25)), ("손절 하단+75%", dict(sfrac=0.75)),
           ("매수 09:30~10:30", dict(win=(570, 630)))]
    NB = []
    for nm, kw in VAR:
        row = [f"  {nm:<26}"]
        rec = dict(name=nm)
        for cfg in ("early5", "late3"):
            d = D[cfg]
            df = d.baselines(d.trades(**kw), which=("RA", "RB"))
            for un, us in UN.items():
                x = df[df.sym.isin(us)]
                ea, ta = ex_mean(x, "RA")
                eb, tb = ex_mean(x, "RB")
                row.append(f"{cfg}·{un[:2]} n{len(x)} {x.ret.mean():+.2%} A{ea:+.2%}({ta:+.1f}) B{eb:+.2%}({tb:+.1f})")
                rec[(cfg, un)] = (len(x), x.ret.mean(), ea, eb)
        NB.append(rec)
        w(" | ".join(row))
    for un in UN:
        ok_e = sum(r[("early5", un)][2] > 0 and r[("early5", un)][3] > 0 for r in NB)
        ok_l = sum(r[("late3", un)][2] > 0 and r[("late3", un)][3] > 0 for r in NB)
        w(f"  {un}: 이웃 {len(NB)}개 중 고르기 초과A·B 모두 + {ok_e}개, 확인(3분) 초과A·B 모두 + {ok_l}개")
    w("")

    # 4) 쏠림
    w("== 4) 쏠림 (최선안, 사용자78)")
    for cfg in ("early5", "late3"):
        df = BEST[cfg]
        x = df[df.sym.isin(U78)].copy()
        x["exA"] = x.ret - x.RA
        x["exB"] = x.ret - x.RB
        bs = x.groupby("sym").ret.sum().sort_values(ascending=False)
        bd = x.groupby("day").ret.sum().sort_values(ascending=False)
        bda = x.groupby("day").exA.sum().sort_values(ascending=False)
        tot = x.ret.sum()
        w(f"  [{cfg}] 수익 합 {tot:+.2f}, 초과A 합 {x.exA.sum():+.2f}, 종목 {x.sym.nunique()}개, 날 {x.day.nunique()}일")
        w(f"    수익 합 상위 종목: " + ", ".join(f"{k} {v:+.2f}" for k, v in bs.head(5).items()) +
          f" | 상위 1종목 몫 {bs.iloc[0] / tot:.0%}, 상위 5종목 {bs.head(5).sum() / tot:.0%}")
        w(f"    수익 합 상위 날: " + ", ".join(f"{k} {v:+.2f}" for k, v in bd.head(5).items()) +
          f" | 상위 1일 몫 {bd.iloc[0] / tot:.0%}, 상위 3일 {bd.head(3).sum() / tot:.0%}")
        w(f"    초과A 상위 날: " + ", ".join(f"{k} {v:+.2f}" for k, v in bda.head(3).items()) +
          f" / 하위 날: " + ", ".join(f"{k} {v:+.2f}" for k, v in bda.tail(3).items()))
        for drop, lab in ((bd.index[:1], "상위 1일 뺌"), (bd.index[:3], "상위 3일 뺌")):
            y = x[~x.day.isin(drop)]
            w(f"    {lab}: {summ(y, ('RA', 'RB'))}")
        y = x[~x.sym.isin(bs.index[:3])]
        w(f"    상위 3종목 뺌: {summ(y, ('RA', 'RB'))}")
        # 거래 수 상위 종목
        w(f"    거래 많은 종목: " + ", ".join(f"{k} {v}" for k, v in x.sym.value_counts().head(5).items()))
        # 주별
        wk = pd.to_datetime(x.day.astype(str)).dt.to_period("W-FRI").astype(str).str[-5:]
        for k, g in x.groupby(wk.values):
            w(f"    주 ~{k}: n={len(g):>3} 평균={g.ret.mean():+.2%} 초과A={g.exA.mean():+.2%} 초과B={g.exB.mean():+.2%}")
    w("")
    w("== 5) 이격매수(같은 기간·정규장만, intra_C 보고값) 대비: early 사용자78 +0.11%(1295건), late +0.60%(383건)")
    for cfg in ("early5", "late3"):
        x = BEST[cfg][BEST[cfg].sym.isin(U78)]
        w(f"  {cfg} 박스 돌파 사용자78 평균 {x.ret.mean():+.2%} ({len(x)}건)")
    w("")
    w(f"완료 {time.time() - t0:.0f}s")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
