"""칼손절 연구 - [폭포 감지] 계열(2026-10-07).
매수 뒤 짧은 시간 안에 매수가 대비 X% 넘게 빠지면 봉 i 시가에 판다. 모든 판단은 봉 i-1까지의 값만 쓴다.

계열(모두 stop_lab.compare로 같은 잣대 비교, 매개변수는 early 결과로만 고른다):
  G1 창 안 종가 고정%   : (i-ei)*분 <= T 이고 c[i-1] <= 매수가*(1-X)
  G2 창 안 저가 고정%   : (i-ei)*분 <= T 이고 l[i-1] <= 매수가*(1-X)
  G3 보유 내내 종가 고정%: c[i-1] <= 매수가*(1-X) (창 없음, 참고용)
  G4 창 안 종가, 장중 변동성 비례: X = k * ATR15. ATR15 = 매수 봉 직전 60분(봉 ei-n..ei-1, n=60/분) 참범위/전봉종가 평균 * sqrt(15/분)
  G5 창 안 종가, 일봉 변동성 비례: X = k * 일봉ATR%. 일봉ATR% = 매수일 '전날까지' 14거래일 참범위/전일종가 평균
  G6 하락 속도(보유 내내): c[i-1] <= 기준*(1-y) 이고 c[i-1] <= 매수가*(1-z).
     기준 = c[i-1-m] (m = M/분 봉) 단 i-1-m < ei 이면 매수가(매수 뒤 움직임만 본다)
고르는 법(미리 정함): early 사용자78 24시간 차이 최대, 단 early 사용자78 정규장만 차이 >= 0. late는 전방 확인으로만 본다.
사용: py stops/waterfall.py   (결과: ../results/stops/waterfall.txt)
"""
import os, sys, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from stop_lab import load, compare, stop_audit, base_trades, run, RAW, UNIV, PNAME

OUT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "results", "stops", "waterfall.txt")
CACHE = os.path.join(os.path.dirname(RAW), "stop_lab_waterfall")


# ---------- 규칙 만들기 ----------
def win_fixed(X, T, ref="c"):
    """T분 창(None이면 보유 내내) 안에서 i-1 종가(ref='c') 또는 저가(ref='l')가 매수가*(1-X) 이하."""
    def make(A):
        p = A[ref]
        k = None if T is None else T // A["mins"]
        def rule(i, ei, ent):
            return (k is None or i - ei <= k) and p[i - 1] <= ent * (1 - X)
        return rule
    return make


def atr15(A, look_min=60):
    """봉 j에서 '봉 j 직전 look_min분' 참범위% 평균을 15분 기준으로 환산(봉 j 자신은 안 씀)."""
    h, l, c, mins = A["h"], A["l"], A["c"], A["mins"]
    cp = np.r_[np.nan, c[:-1]]
    trp = (np.maximum(h, cp) - np.minimum(l, cp)) / cp
    n = look_min // mins
    return pd.Series(trp).rolling(n).mean().shift(1).to_numpy() * np.sqrt(15 / mins)


def win_atr(kk, T):
    def make(A):
        c, a = A["c"], atr15(A)
        k = T // A["mins"]
        def rule(i, ei, ent):
            th = a[ei]
            return i - ei <= k and th == th and c[i - 1] <= ent * (1 - kk * th)
        return rule
    return make


_DATR = {}


def daily_atr(sym):
    """{날짜: 그 날짜 '전날까지' 14거래일 참범위% 평균}."""
    if sym not in _DATR:
        d = pd.read_parquet(f"{RAW}/1d_{sym}.parquet")
        d.index = pd.to_datetime(d.index).date
        pc = d.Close.shift(1)
        tr = (np.maximum(d.High, pc) - np.minimum(d.Low, pc)) / pc
        _DATR[sym] = tr.rolling(14).mean().shift(1).to_dict()
    return _DATR[sym]


def win_datr(kk, T):
    def make(A):
        c, m = A["c"], daily_atr(A["sym"])
        da = np.array([m.get(x, np.nan) for x in A["day"]], dtype=float)
        k = T // A["mins"]
        def rule(i, ei, ent):
            th = da[ei]
            return i - ei <= k and th == th and c[i - 1] <= ent * (1 - kk * th)
        return rule
    return make


def speed(M, y, z):
    def make(A):
        c = A["c"]
        m = max(1, M // A["mins"])
        def rule(i, ei, ent):
            j = i - 1
            ref = c[j - m] if j - m >= ei else ent
            return c[j] <= ref * (1 - y) and c[j] <= ent * (1 - z)
        return rule
    return make


def grid():
    G = []
    for X in (0.015, 0.02, 0.025, 0.03, 0.04, 0.05):
        for T in (15, 30, 45, 60):
            G.append(("G1", f"종가 -{X:.1%} {T}분 안", win_fixed(X, T, "c")))
    for X in (0.02, 0.03, 0.04, 0.05):
        for T in (15, 30, 60):
            G.append(("G2", f"저가 -{X:.1%} {T}분 안", win_fixed(X, T, "l")))
    for X in (0.02, 0.03, 0.04):
        G.append(("G3", f"종가 -{X:.1%} 보유 내내", win_fixed(X, None, "c")))
    for kk in (1.5, 2.0, 3.0, 4.0):
        for T in (30, 60):
            G.append(("G4", f"종가 -{kk}xATR15 {T}분 안", win_atr(kk, T)))
    for kk in (0.3, 0.5, 0.75):
        for T in (30, 60):
            G.append(("G5", f"종가 -{kk}x일봉ATR {T}분 안", win_datr(kk, T)))
    for M in (15, 30):
        for y in (0.015, 0.02, 0.03):
            for z in (0.0, 0.02):
                G.append(("G6", f"{M}분 -{y:.1%} & 매수가 -{z:.0%} 아래", speed(M, y, z)))
    return G


# ---------- 미래 참조 점검 ----------
def lookahead_check(make, n_pts=400, seed=7):
    """봉 i 이후 값(o h l c v r ema)을 무작위로 바꿔도 rule(i, ei, ent) 판단이 같아야 한다.
    점: 기준 거래 보유 구간의 (종목, ei, i)에서 무작위 + 이 규칙의 손절 지점 전부(최대 n_pts)."""
    rng = np.random.default_rng(seed)
    bad, tot = 0, 0
    for period in ("early", "late"):
        S = load(period)
        b = base_trades(period)
        t = run(S, make, period)
        pts = [(x.sym, x.ei, x.xi, x.ent) for x in t[t.why == "손절"].itertuples()][: n_pts // 2]
        for x in b.sample(min(len(b), n_pts // 2), random_state=seed).itertuples():
            if x.xi - 1 > x.ei:
                pts.append((x.sym, x.ei, int(rng.integers(x.ei + 1, x.xi)), x.ent))
        for s, ei, i, ent in pts:
            A = S[s]
            r0 = make(A)(i, ei, ent)
            B = dict(A)
            for key in ("o", "h", "l", "c", "v", "r", "ema"):
                arr = A[key].copy()
                arr[i:] = arr[i:] * rng.uniform(0.3, 1.7, len(arr) - i)
                B[key] = arr
            r1 = make(B)(i, ei, ent)
            tot += 1
            bad += int(bool(r0) != bool(r1))
    return bad, tot


def datr_prior_check(sym="NVDA", n=20):
    """일봉 ATR이 그 날짜 전날까지만 쓰는지: 날짜 D 이후 일봉을 지운 파일로 다시 계산해도 같아야 한다."""
    d = pd.read_parquet(f"{RAW}/1d_{sym}.parquet")
    d.index = pd.to_datetime(d.index).date
    full = daily_atr(sym)
    days = list(d.index)[-n:]
    bad = 0
    for D in days:
        e = d[d.index <= D].copy()
        e.loc[D, ["High", "Low", "Close"]] = e.loc[D, ["High", "Low", "Close"]] * 1.5   # D 당일 값을 망가뜨림
        pc = e.Close.shift(1)
        tr = (np.maximum(e.High, pc) - np.minimum(e.Low, pc)) / pc
        v = tr.rolling(14).mean().shift(1)[D]
        bad += int(not np.isclose(v, full[D]))
    return bad, len(days)


# ---------- 진단: 손절이 어떤 갈래(stop_paths.py 정의)를 잘랐나 ----------
def classify(period, b):
    """기준 거래에 갈래 붙이기: 반등 / 폭포(-5% 넘게, 바닥 절반 이상을 60분 안) / 흘러내림(-3% 넘게 잃음) / 작은 손실."""
    S = load(period)
    out = []
    for x in b.itertuples():
        A = S[x.sym]
        dd = A["l"][x.ei:x.xi + 1] / x.ent - 1
        k60 = max(1, 60 // A["mins"])
        mae = dd.min()
        fast = mae < 0 and dd[:k60 + 1].min() <= mae * 0.5
        out.append("반등" if x.ret > 0 else ("폭포" if mae <= -0.05 and fast else ("흘러내림" if x.ret <= -0.03 else "작은손실")))
    return pd.Series(out, index=b.index)


def diag(make, period, df_cmp, syms=None):
    """사용자78 24시간: 손절 거래를 같은 매수 봉의 기준 거래와 짝지어 갈래별로 (손절 수익 - 기준 수익) 합을 본다.
    직접 효과 = 짝지은 합, 나머지 = 차이 - 직접(손절 뒤 재매수 금지·뒤 거래 변화)."""
    syms = syms if syms is not None else UNIV["사용자78"]
    b = base_trades(period)
    b = b[b.sym.isin(syms)].copy()
    b["cls"] = classify(period, b)
    t = run(load(period), make, period, syms=syms)
    s = t[t.why == "손절"][["sym", "ei", "ret"]].rename(columns={"ret": "손절"})
    m = s.merge(b[["sym", "ei", "ret", "cls"]], on=["sym", "ei"], how="inner")
    m["d"] = m.손절 - m.ret
    g = m.groupby("cls").agg(건수=("d", "size"), 아낀합=("d", "sum"))
    tot = b.groupby("cls").size()
    diff = cell(df_cmp, period, "24시간").차이
    direct = m.d.sum()
    parts = ", ".join(f"{k} {int(g.loc[k, '건수'])}/{int(tot.get(k, 0))}건 {g.loc[k, '아낀합']:+.2f}" for k in
                      ("반등", "작은손실", "폭포", "흘러내림") if k in g.index)
    return f"{period}: 손절 {len(s)}건 | 갈래별 잘린 건/전체 건 아낀합: {parts} | 직접 {direct:+.2f}, 나머지 {diff - direct:+.2f}, 차이 {diff:+.2f}"


# ---------- 실행 ----------
def cell(df, period, scope, un="사용자78"):
    r = df[(df.기간 == period) & (df.범위 == scope) & (df.종목 == un)].iloc[0]
    return r


def main():
    lines = []
    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    out("칼손절 연구 [폭포 감지] 계열 - waterfall.py (2026-10-07)")
    out(__doc__.split("\n", 1)[1].rsplit("사용:", 1)[0].rstrip())
    out("")
    for p in ("early", "late"):
        for reg in (False, True):
            b = base_trades(p, reg)
            out(f"기준 {PNAME[p]} {'정규장만' if reg else '24시간'}: " + ", ".join(
                f"{un} {len(b[b.sym.isin(us)])}건 합 {b[b.sym.isin(us)].ret.sum():+.2f}" for un, us in UNIV.items()))

    G = grid()
    out(f"\n시험 규칙 수: {len(G)}")
    rows, full = [], {}
    t0 = time.time()
    os.makedirs(CACHE, exist_ok=True)
    for n, (g, name, make) in enumerate(G, 1):
        cp = os.path.join(CACHE, f"{n:02d}.csv")     # 중간 저장(다시 돌릴 때 이미 끝난 규칙은 건너뜀)
        df = pd.read_csv(cp) if os.path.exists(cp) else None
        if df is None or df.규칙.iloc[0] != name:
            df = compare(make, name, show=False)
            df.to_csv(cp, index=False)
        full[name] = df
        rec = dict(번호=n, 계열=g, 규칙=name)
        for p in ("early", "late"):
            for sc, tag in (("24시간", "24"), ("정규장만", "정규")):
                r = cell(df, p, sc)
                rec[f"{p}_{tag}"] = r.차이
                rec[f"{p}_{tag}_t"] = r.t
                rec[f"{p}_{tag}_손절"] = r.손절
            rec[f"{p}_24_137"] = cell(df, p, "24시간", "전체137").차이
            rec[f"{p}_정규_137"] = cell(df, p, "정규장만", "전체137").차이
        rows.append(rec)
        print(f"  [{n}/{len(G)}] {name}: early24 {rec['early_24']:+.2f} early정규 {rec['early_정규']:+.2f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    R = pd.DataFrame(rows)

    show = R[["번호", "계열", "규칙", "early_24", "early_24_t", "early_24_손절", "early_정규", "early_정규_t",
              "late_24", "late_24_t", "late_정규", "late_정규_t", "early_24_137", "late_24_137"]].copy()
    out("\n=== 전체 결과(사용자78 차이 = 규칙 합 - 기준 합; 137은 전체137 24시간 차이) ===")
    out("  (late 열은 전방 확인용 - 고르는 데 쓰지 않음)")
    out(show.to_string(index=False, float_format=lambda v: f"{v:+.2f}"))

    four = R[(R.early_24 > 0) & (R.early_정규 > 0) & (R.late_24 > 0) & (R.late_정규 > 0)]
    e_pos = R[R.early_24 > 0]
    out(f"\nearly 사용자78 24시간 플러스: {len(e_pos)}/{len(R)}, early 두 칸 플러스: "
        f"{int(((R.early_24 > 0) & (R.early_정규 > 0)).sum())}, 사용자78 네 칸 모두 플러스: {len(four)} "
        f"{list(four.규칙)}")
    out("계열별 early 사용자78 24시간 차이 최대/중앙: " + ", ".join(
        f"{g} {x.early_24.max():+.2f}/{x.early_24.median():+.2f}" for g, x in R.groupby("계열")))

    cand = R[R.early_정규 >= 0].sort_values("early_24", ascending=False)
    if not len(cand):
        cand = R.sort_values("early_24", ascending=False)
    best = cand.iloc[0]
    out(f"\n=== early로 고른 최선안: [{best.번호}] {best.규칙} ===")
    out(full[best.규칙].assign(**{k: full[best.규칙][k].map(fmt.format) for k, fmt in (
        ("평균", "{:+.2%}"), ("승률", "{:.0%}"), ("최악", "{:+.1%}"), ("하위5", "{:+.1%}"), ("합", "{:+.2f}"),
        ("차이", "{:+.2f}"), ("t", "{:+.2f}"), ("차이_10월6일", "{:+.2f}"))}).drop(columns=["손절평균"]).to_string(index=False))
    make = dict((nm, mk) for _, nm, mk in G)[best.규칙]
    for p in ("early", "late"):
        for reg in (False, True):
            a = stop_audit(make, p, reg)
            out(f"stop_audit {p} {'정규장만' if reg else '24시간'} 사용자78: " + ", ".join(
                f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in a.items()))

    out("\n=== 차순위(early 사용자78 24시간 차이 순, 정규장만 >= 0 조건) ===")
    out(cand.head(8)[["번호", "규칙", "early_24", "early_24_t", "early_정규", "late_24", "late_24_t", "late_정규"]]
        .to_string(index=False, float_format=lambda v: f"{v:+.2f}"))
    out("\n=== 참고: 조건 없이 early 24시간 상위 5 ===")
    out(R.sort_values("early_24", ascending=False).head(5)[["번호", "규칙", "early_24", "early_정규", "late_24", "late_정규"]]
        .to_string(index=False, float_format=lambda v: f"{v:+.2f}"))

    out("\n=== 진단(사용자78 24시간): 손절이 어느 갈래를 잘랐나, 아낀합 = 손절 수익 - 기준 수익 ===")
    MK = dict((x, y) for _, x, y in G)
    for nm in dict.fromkeys([best.규칙, "종가 -2.0% 30분 안", "종가 -3.0% 15분 안", "저가 -3.0% 30분 안",
                             "종가 -3.0xATR15 30분 안", "15분 -2.0% & 매수가 -0% 아래"]):
        out(f" [{nm}]")
        for p in ("early", "late"):
            out("   " + diag(MK[nm], p, full[nm]))

    out("\n=== 미래 참조 점검 ===")
    for g in ("G1", "G2", "G4", "G5", "G6"):
        nm = R[R.계열 == g].sort_values("early_24", ascending=False).iloc[0].규칙
        mk = dict((x, y) for _, x, y in G)[nm]
        bad, tot = lookahead_check(mk, n_pts=300)
        out(f"  {g} {nm}: 봉 i 이후 값을 바꿔도 판단이 바뀐 점 {bad}/{tot}")
    bad, tot = datr_prior_check()
    out(f"  일봉ATR 당일 값 망가뜨려도 바뀐 날 {bad}/{tot}(NVDA)")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("저장:", OUT)


if __name__ == "__main__":
    main()
