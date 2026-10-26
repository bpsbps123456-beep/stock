"""daily_B 최선안 독립 재구현·반박 검증 (box_trade 키 verify_daily_B, 2026-10-09).
정의(연구자 best_rule.definition만 보고 다시 짬): 일봉, 박스 저점 = 신호일 직전 N봉 저가 최소(오늘 제외),
지정가 = 박스 저점 x (1-y), 그날 저가 < 지정가면 체결, 체결가 min(시가, 지정가), k거래일 뒤 시가 매도, 종목당 1포지션.
기준: 같은 날 무작위(묶음 전체 시가→k일 뒤 시가), SPY, 같은 종목 보유(기간 안 아무 날 k일), 같은 날 같은 깊이 눌림(전날 종가 대비 같은 % 지정가).
점검: 재현(N60 y6% X1), 이웃 N·y·k, 쏠림(월·날짜·종목), 체결 '닿기' 판정, 분할 필터 유무, 5분봉으로 일봉 저가 체결 확인.
사용: py verify_daily_B.py"""
import glob, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "../results")
SP = r"C:/Users/geunt/AppData/Local/Temp/claude/C--Users-geunt-OneDrive-Desktop---/4e686c73-74be-4337-8ccc-2f6136beaf7d/scratchpad"
DD = os.path.join(SP, "box/daily")
RAW = os.path.join(SP, "dip/raw")
sys.path.insert(0, os.path.join(HERE, "../../dip_reason/code"))
from fetch import user_universe  # noqa: E402

PER = {"참고": ("2024-10-09", "2025-12-31"), "고르기": ("2026-01-02", "2026-06-30"), "확인": ("2026-07-01", "2026-10-08")}

# ------------------------------------------------------------------ 데이터
sp = set(pd.read_csv(os.path.join(DD, "sp500_list.csv")).iloc[:, 0])
u78 = set(user_universe()["합집합"])
avail = {os.path.basename(f)[3:-8] for f in glob.glob(os.path.join(DD, "1d_*.parquet"))}
names = sorted((sp | u78 | {"SPY", "SMH"}) & avail)   # SMH는 시장 맥락용(묶음 밖)
cal = pd.read_parquet(os.path.join(DD, "1d_SPY.parquet")).index
cal = pd.DatetimeIndex(cal).normalize()
cal = cal[~cal.duplicated()].sort_values()
T, S = len(cal), len(names)
O = np.full((T, S), np.nan); H = O.copy(); L = O.copy(); C = O.copy()
for j, s in enumerate(names):
    d = pd.read_parquet(os.path.join(DD, f"1d_{s}.parquet"))
    d.index = pd.DatetimeIndex(d.index).normalize()
    d = d[~d.index.duplicated(keep="last")].reindex(cal)
    O[:, j], H[:, j], L[:, j], C[:, j] = (d[c].to_numpy(float) for c in ("Open", "High", "Low", "Close"))
for A in (O, H, L, C):
    A[~(A > 0)] = np.nan
day = np.array(cal.strftime("%Y-%m-%d"))
isSP = np.array([s in sp for s in names]); isU = np.array([s in u78 for s in names])
UNIV = {"S&P500": isSP, "사용자78": isU}
jSPY = names.index("SPY")
pmask = {p: (day >= a) & (day <= b) for p, (a, b) in PER.items()}
Cprev = np.vstack([np.full((1, S), np.nan), C[:-1]])
with np.errstate(invalid="ignore", divide="ignore"):
    rr = C / Cprev
bad = (rr <= 0.6) | (rr >= 1.8)          # 분할 의심(실제로는 대부분 진짜 급등락, 분할은 yfinance가 이미 조정)
cbad = np.vstack([np.zeros((1, S), int), np.cumsum(bad, 0)])

_box = {}


def box_low(N):
    if N not in _box:
        out = np.full((T, S), np.nan)
        w = np.lib.stride_tricks.sliding_window_view(L, N, axis=0)   # (T-N+1, S, N), 창 i = i..i+N-1
        out[N:] = w.min(axis=2)[:-1]                                  # t의 박스 = t-N..t-1 (NaN 있으면 NaN)
        _box[N] = out
    return _box[N]


_fwd = {}


def fwd_ret(k):
    """R[t] = O[t+k]/O[t]-1."""
    if k not in _fwd:
        R = np.full((T, S), np.nan)
        R[:-k] = O[k:] / O[:-k] - 1
        _fwd[k] = R
    return _fwd[k]


def trades(N, y, k, touch=False, splitfilt=False):
    box = box_low(N)
    lim = box * (1 - y)
    with np.errstate(invalid="ignore"):
        fill = (L <= lim) if touch else (L < lim)
    fill &= np.isfinite(O) & np.isfinite(lim)
    ent = np.fmin(O, lim)
    rows = []
    for s in range(S):
        if s == jSPY:
            continue
        free = 0
        for t in np.nonzero(fill[:, s])[0]:
            if t < free:
                continue
            x = t + k
            if splitfilt and cbad[min(x + 1, T), s] - cbad[max(t - N, 0), s] > 0:
                continue
            if x >= T or not np.isfinite(O[x, s]):
                break                                   # 미청산: 이후 신호 없음
            rows.append((t, s, x, ent[t, s], O[x, s] / ent[t, s] - 1, lim[t, s], box[t, s]))
            free = x                                    # 시가 매도 뒤 그날 지정가 다시 가능
    df = pd.DataFrame(rows, columns=["t", "s", "x", "p", "r", "lim", "box"])
    return df


def clt(v, g):
    v = np.asarray(v, float)
    n = len(v)
    if n < 3:
        return np.nan
    u = pd.Series(v - v.mean()).groupby(np.asarray(g)).sum().to_numpy()
    G = len(u)
    if G < 2:
        return np.nan
    se = np.sqrt((u ** 2).sum() * G / (G - 1)) / n
    return v.mean() / se if se > 0 else np.nan


def add_bases(df, k, un):
    U = UNIV[un]
    R = fwd_ret(k)
    with np.errstate(all="ignore"):
        rnd = np.nanmean(np.where(U[None, :], R, np.nan), 1)
    t, s = df.t.to_numpy(), df.s.to_numpy()
    df = df.copy()
    df["rnd"] = rnd[t]
    df["spy"] = R[t, jSPY]
    # 같은 종목 보유: 신호일이 속한 기간 안 아무 날 시가에 사서 k일 뒤 시가
    bh = np.full(len(df), np.nan)
    for p, pm in pmask.items():
        with np.errstate(all="ignore"):
            m = np.nanmean(R[pm], 0)
        sel = pm[t]
        bh[sel] = m[s[sel]]
    df["bh"] = bh
    # 같은 날 같은 깊이 눌림(박스 무관): 전날 종가 x (1-d) 지정가, 그 종목 자신 제외
    dip = np.full(len(df), np.nan)
    ndip = np.zeros(len(df), int)
    for i, (ti, si, pi) in enumerate(zip(t, s, df.p.to_numpy())):
        if ti + k >= T or not np.isfinite(Cprev[ti, si]):
            continue
        d = 1 - pi / Cprev[ti, si]
        lj = Cprev[ti] * (1 - d)
        with np.errstate(invalid="ignore"):
            f = (L[ti] < lj) & U & np.isfinite(O[ti + k])
        f[si] = False
        if f.any():
            dip[i] = np.mean(O[ti + k, f] / np.fmin(O[ti, f], lj[f]) - 1)
            ndip[i] = f.sum()
    df["dip"] = dip
    df["ndip"] = ndip
    return df


def stats(df):
    if len(df) == 0:
        return dict(n=0)
    g = day[df.t.to_numpy()]
    r = df.r.to_numpy()
    out = dict(n=len(df), dates=len(set(g)), avg=r.mean(), win=(r > 0).mean(), worst=r.min(), sum=r.sum(), t=clt(r, g))
    for b in ("rnd", "spy", "bh", "dip"):
        e = r - df[b].to_numpy()
        ok = np.isfinite(e)
        out["x_" + b] = e[ok].mean() if ok.any() else np.nan
        out["t_" + b] = clt(e[ok], g[ok]) if ok.sum() > 2 else np.nan
        out["n_" + b] = int(ok.sum())
    return out


def fmt(st):
    if st.get("n", 0) == 0:
        return "건수 0"
    return (f"{st['n']:4d}건({st['dates']:3d}일) 평균 {st['avg']*100:+6.2f}% 승률 {st['win']*100:3.0f}% 최악 {st['worst']*100:+6.1f}% "
            f"합 {st['sum']*100:+7.1f}% t {st['t']:+5.2f} | 무작위 대비 {st['x_rnd']*100:+5.2f}%(t {st['t_rnd']:+5.2f}) "
            f"SPY 대비 {st['x_spy']*100:+5.2f}% 종목보유 대비 {st['x_bh']*100:+5.2f}% "
            f"같은깊이 대비 {st['x_dip']*100:+5.2f}%(t {st['t_dip']:+5.2f}, {st['n_dip']}건)")


def cell(df, un, p):
    m = UNIV[un][df.s.to_numpy()] & pmask[p][df.t.to_numpy()]
    return df[m]


def main():
    W = []
    w = W.append
    w("verify_daily_B: daily_B 최선안(L6_N60 X1) 독립 재구현·반박 검증 (2026-10-09)")
    w(f"데이터 {day[0]}~{day[-1]} {T}거래일, 종목 {S - 1}(S&P500 {isSP.sum()}, 사용자78 {isU.sum()}). 분할은 yfinance가 이미 조정(auto_adjust=False도 OHLC 분할 조정).")
    bd = [(names[j], day[i], round(float(rr[i, j]), 2)) for i, j in zip(*np.nonzero(bad))]
    w(f"'분할 의심'(종가 -40% 이하·+80% 이상) 날 {len(bd)}개: {bd} → 대부분 실제 급등락이라 걸러도 되는 날이 아님")
    w("")

    # ---------------- 1) 재현
    BASE = {}
    w("[1] 재현: N60 y6% X1 (엄격 미만 체결, 분할 필터 없음 / 연구자식 필터 [t-N, 청산] / 닿기 체결)")
    for tag, kw in (("기본", {}), ("연구자식 필터", dict(splitfilt=True)), ("닿기 체결", dict(touch=True))):
        df = trades(60, 0.06, 1, **kw)
        for un in UNIV:
            d2 = add_bases(df[UNIV[un][df.s.to_numpy()]], 1, un)
            BASE[(tag, un)] = d2
            for p in ("고르기", "확인", "참고"):
                w(f"  {tag:8s} {un:6s} {p}: {fmt(stats(cell(d2, un, p)))}")
    w("")

    # ---------------- 2) 나이키
    j = names.index("NKE")
    t0 = int(np.nonzero(day == "2026-10-02")[0][0])
    bx = box_low(60)[t0, j]
    w(f"[2] NKE 10/2: 박스 저점 {bx:.2f}, 지정가 {bx*0.94:.2f}, 시가 {O[t0, j]:.2f}, 저가 {L[t0, j]:.2f} → 체결 {min(O[t0, j], bx*0.94):.2f}, "
      f"10/5 시가 {O[t0+1, j]:.2f} → {O[t0+1, j]/min(O[t0, j], bx*0.94)*100-100:+.1f}%")
    w("")

    # ---------------- 3) 이웃
    w("[3] 이웃 매개변수: 무작위 대비 초과 %(날짜 군집 t) — 고르기 / 확인, S&P500 · 사용자78")
    grid = []
    for N in (40, 50, 60, 70, 80):
        for y in (0.04, 0.05, 0.06, 0.07, 0.08):
            for k in (1, 2, 3):
                df = trades(N, y, k)
                row = dict(N=N, y=y, k=k)
                for un in UNIV:
                    d2 = add_bases(df[UNIV[un][df.s.to_numpy()]], k, un)
                    for p in ("고르기", "확인"):
                        st = stats(cell(d2, un, p))
                        row[f"{un}_{p}_n"] = st.get("n", 0)
                        row[f"{un}_{p}_x"] = st.get("x_rnd", np.nan)
                        row[f"{un}_{p}_t"] = st.get("t_rnd", np.nan)
                        row[f"{un}_{p}_xd"] = st.get("x_dip", np.nan)
                        row[f"{un}_{p}_avg"] = st.get("avg", np.nan)
                grid.append(row)
    G = pd.DataFrame(grid)
    for _, r in G.iterrows():
        tag = " <- 최선안" if (r.N == 60 and abs(r.y - 0.06) < 1e-9 and r.k == 1) else ""
        parts = []
        for un in UNIV:
            for p in ("고르기", "확인"):
                parts.append(f"{un[:3]}{p[:1]} n{int(r[f'{un}_{p}_n']):4d} {r[f'{un}_{p}_x']*100:+5.2f}({r[f'{un}_{p}_t']:+4.1f}) 깊이{r[f'{un}_{p}_xd']*100:+5.2f}")
        w(f"  N{int(r.N):3d} y{int(round(r.y*100))}% X{int(r.k)}: " + " | ".join(parts) + tag)
    for un in UNIV:
        both = ((G[f"{un}_고르기_x"] > 0) & (G[f"{un}_확인_x"] > 0)).sum()
        bothd = ((G[f"{un}_고르기_xd"] > 0) & (G[f"{un}_확인_xd"] > 0)).sum()
        w(f"  {un}: 75개 중 두 기간 무작위 초과>0 {both}개, 확인 초과>0 {(G[f'{un}_확인_x'] > 0).sum()}개, 고르기 초과>0 {(G[f'{un}_고르기_x'] > 0).sum()}개"
          f" | 같은깊이 대비 두 기간>0 {bothd}개, 확인>0 {(G[f'{un}_확인_xd'] > 0).sum()}개")
    nb = G[((G.N.isin([50, 60, 70])) & (np.isclose(G.y, 0.06)) & (G.k == 1)) | ((G.N == 60) & (np.isclose(G.y, 0.05) | np.isclose(G.y, 0.07)) & (G.k == 1)) | ((G.N == 60) & np.isclose(G.y, 0.06) & (G.k == 2))]
    w("  한 단계 이웃(N50·70 / y5·7% / X2):")
    for _, r in nb.iterrows():
        w(f"    N{int(r.N)} y{int(round(r.y*100))}% X{int(r.k)}: S&P 고르기 {r['S&P500_고르기_x']*100:+.2f}% 확인 {r['S&P500_확인_x']*100:+.2f}% (t {r['S&P500_확인_t']:+.1f}, n {int(r['S&P500_확인_n'])}) | "
          f"사용자78 고르기 {r['사용자78_고르기_x']*100:+.2f}% 확인 {r['사용자78_확인_x']*100:+.2f}% (n {int(r['사용자78_확인_n'])})")
    w("")

    # ---------------- 4) 쏠림
    w("[4] 쏠림: 최선안(기본) 확인 구간")
    for un in UNIV:
        d2 = cell(BASE[("기본", un)], un, "확인").copy()
        d2["ex"] = d2.r - d2.rnd
        d2["exd"] = d2.r - d2.dip
        d2["m"] = [day[t][:7] for t in d2.t]
        d2["d"] = day[d2.t.to_numpy()]
        d2["tk"] = [names[s] for s in d2.s]
        w(f"  {un}: 월별 건수/평균/무작위 대비/같은깊이 대비")
        for m, g in d2.groupby("m"):
            w(f"     {m}: {len(g):3d}건 평균 {g.r.mean()*100:+6.2f}% 무작위 대비 {g.ex.mean()*100:+6.2f}% 같은깊이 대비 {g.exd.mean()*100:+6.2f}%")
        bydate = d2.groupby("d").agg(n=("ex", "size"), ex=("ex", "sum"), r=("r", "sum")).sort_values("ex", ascending=False)
        tot = d2.ex.sum()
        w(f"     날짜별 초과 합 상위 5 (전체 초과 합 {tot*100:+.1f}%p, {len(bydate)}일): " +
          ", ".join(f"{i} {v.n}건 {v.ex*100:+.1f}" for i, v in bydate.head(5).iterrows()))
        w(f"     날짜별 하위 3: " + ", ".join(f"{i} {v.n}건 {v.ex*100:+.1f}" for i, v in bydate.tail(3).iterrows()))
        bytk = d2.groupby("tk").agg(n=("ex", "size"), ex=("ex", "sum")).sort_values("ex", ascending=False)
        w(f"     종목별 상위 5: " + ", ".join(f"{i} {v.n}건 {v.ex*100:+.1f}" for i, v in bytk.head(5).iterrows()))
        for drop in (1, 3):
            top = set(bydate.index[:drop])
            rest = d2[~d2.d.isin(top)]
            w(f"     상위 {drop}일 빼면: {len(rest)}건 평균 {rest.r.mean()*100:+.2f}% 무작위 대비 {rest.ex.mean()*100:+.2f}% (t {clt(rest.ex.to_numpy(), rest.d.to_numpy()):+.2f})")
        rest = d2[d2.m != "2026-07"]
        if len(rest):
            w(f"     7월 빼면: {len(rest)}건 평균 {rest.r.mean()*100:+.2f}% 무작위 대비 {rest.ex.mean()*100:+.2f}% (t {clt(rest.ex.to_numpy(), rest.d.to_numpy()):+.2f}) 같은깊이 대비 {np.nanmean(rest.exd)*100:+.2f}%")
        top1 = bytk.index[0]
        rest = d2[d2.tk != top1]
        w(f"     최상위 종목 {top1} 빼면: {len(rest)}건 무작위 대비 {rest.ex.mean()*100:+.2f}%")
        loo = [d2[d2.d != dd].ex.mean() for dd in bydate.index]
        w(f"     날짜 하나씩 빼기(무작위 대비 평균) 최소 {min(loo)*100:+.2f}% 최대 {max(loo)*100:+.2f}%")
    w("  상위 날짜 빼기(고르기·확인 모두): 무작위 대비 초과 평균(t)")
    for un in UNIV:
        for p in ("고르기", "확인"):
            c = cell(BASE[("기본", un)], un, p).copy()
            c["ex"] = c.r - c.rnd
            c["d"] = day[c.t.to_numpy()]
            bd_ = c.groupby("d").ex.sum().sort_values(ascending=False)
            parts = [f"상위 3일 {list(bd_.index[:3])}이 초과 합의 {bd_.iloc[:3].sum()/bd_.sum()*100:.0f}%"]
            for k_ in (1, 3, 5):
                r_ = c[~c.d.isin(bd_.index[:k_])]
                parts.append(f"{k_}일 빼면 {len(r_)}건 {r_.ex.mean()*100:+.2f}%(t {clt(r_.ex.to_numpy(), r_.d.to_numpy()):+.2f}) 평균 {r_.r.mean()*100:+.2f}%")
            w(f"     {un} {p}: " + " | ".join(parts))
    w("  시장 맥락(SMH 그날 종가 등락 / 다음 날 시가 갭): " + ", ".join(
        f"{d_} {(C[i_, names.index('SMH')]/C[i_-1, names.index('SMH')]-1)*100:+.1f}%/{(O[i_+1, names.index('SMH')]/C[i_, names.index('SMH')]-1)*100:+.1f}%"
        for d_ in ("2026-07-16", "2026-07-17", "2026-07-28", "2026-07-29") for i_ in [int(np.nonzero(day == d_)[0][0])]) if "SMH" in names else "  SMH 없음")
    w("")

    # ---------------- 5) 5분봉으로 일봉 저가 체결 확인(확인 구간, 5분봉 있는 종목)
    w("[5] 5분봉 정규장(9:30~16:00) 저가로 지정가 체결 다시 확인 (7/14~10/6, 5분봉 있는 종목)")
    d2 = BASE[("기본", "S&P500")]
    d3 = BASE[("기본", "사용자78")]
    allt = pd.concat([d2, d3]).drop_duplicates(subset=["t", "s"])
    chk = []
    for _, r in allt.iterrows():
        tk = names[int(r.s)]
        dd = day[int(r.t)]
        f = os.path.join(RAW, f"5m_{tk}.parquet")
        if dd < "2026-07-14" or dd > "2026-10-06" or not os.path.exists(f):
            continue
        m5 = pd.read_parquet(f)
        idx = m5.index.tz_convert("America/New_York") if m5.index.tz is not None else m5.index
        sel = (idx.strftime("%Y-%m-%d") == dd) & (idx.strftime("%H:%M") >= "09:30") & (idx.strftime("%H:%M") < "16:00")
        g = m5[sel]
        if len(g) == 0:
            continue
        lo5 = g.Low.min()
        op5 = g.Open.iloc[0]
        p5 = min(op5, r.lim) if lo5 < r.lim else np.nan
        chk.append((tk, dd, r.lim, L[int(r.t), int(r.s)], lo5, O[int(r.t), int(r.s)], op5, r.p, p5, r.r,
                    O[int(r.x), int(r.s)] / p5 - 1 if np.isfinite(p5) else np.nan))
    if chk:
        K = pd.DataFrame(chk, columns=["tk", "d", "lim", "low_d", "low_5m", "open_d", "open_5m", "p_d", "p_5m", "r_d", "r_5m"])
        w(f"  확인 대상 {len(K)}건, 5분봉에서도 체결 {K.p_5m.notna().sum()}건, 미체결 {K.p_5m.isna().sum()}건")
        w(f"  평균 수익 일봉 {K.r_d.mean()*100:+.2f}% vs 5분봉 체결로 다시 계산 {np.nanmean(K.r_5m)*100:+.2f}% (미체결은 빼고)")
        w(K.round(3).to_string(index=False))
    w("")
    G.to_csv(os.path.join(RES, "verify_daily_B_grid.csv"), index=False, encoding="utf-8-sig")
    txt = "\n".join(W)
    with open(os.path.join(RES, "verify_daily_B.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
