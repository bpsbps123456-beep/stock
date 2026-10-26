"""칼손절 연구 — RSI·추세 경로 계열(2026-10-07). 지금 지표 매수 뒤 기준봉 RSI 경로·EMA 기울기·저점 계단·이격 확대로
'폭포·흘러내림'만 일찍 자르는 규칙을 찾는다. 시험대는 ../stop_lab.py(고치지 않음).

모든 규칙은 봉 i 시가에 팔지를 i-1 봉까지의 값(종가·저가·고가·RSI·EMA)으로 정한다. 경과 = i - ei 봉, k분 = max(1, k // 봉 길이) 봉.
매개변수 선택(사전 고정): early 사용자78 24시간 '차이'가 가장 큰 규칙, 단 early 사용자78 정규장만 '차이' >= 0.
late는 전방 확인으로만 본다. 사용: py stops/rsipath.py  → ../results/stops/rsipath.txt"""
import os, sys, json
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from stop_lab import load, compare, stop_audit, base_trades, USER78, UNIV

OUT = os.path.normpath(os.path.join(HERE, "../../results/stops/rsipath.txt"))
CACHE = os.path.join(r"C:/Users/geunt/AppData/Local/Temp/claude/C--Users-geunt-OneDrive-Desktop---/4e686c73-74be-4337-8ccc-2f6136beaf7d/scratchpad/rsipath", "cmp_cache.json")


def kbars(A, m):
    return max(1, m // A["mins"])


def runlen(cond):
    """cond[j]가 j에서 끝나는 연속 True 길이."""
    out = np.zeros(len(cond), dtype=np.int64)
    n = 0
    for j, v in enumerate(cond):
        n = n + 1 if v else 0
        out[j] = n
    return out


# ---------------- 규칙 공장 ----------------
def rsi_fail(T, m):
    """매수 봉부터 k분 동안(봉 ei..i-1, i-ei == k) RSI 최고가 T 미만이면 판다(한 번만 확인)."""
    def make(A):
        r, kb = A["r"], kbars(A, m)
        def rule(i, ei, ent):
            return i - ei == kb and r[ei:i].max() < T
        return rule
    return make


def rsi_newlow(d, m):
    """경과 >= k분이고, RSI[i-1]이 매수 뒤 새 저점(r[ei..i-2] 최저보다 낮음)이면서 매수 봉 RSI보다 d 넘게 낮으면 판다."""
    def make(A):
        r, kb = A["r"], kbars(A, m) if m > 0 else 1
        st = {"ei": -1, "j": -1, "mn": 0.0}
        def rule(i, ei, ent):
            if st["ei"] != ei:
                st.update(ei=ei, j=ei, mn=r[ei])
            new = False
            while st["j"] < i - 1:            # 봉 j = i-1까지 갱신
                st["j"] += 1
                x = r[st["j"]]
                new = x < st["mn"]
                st["mn"] = min(st["mn"], x)
            return new and i - ei >= kb and r[i - 1] < r[ei] - d
        return rule
    return make


def rsi_floor(L):
    """경과 1봉 이상, RSI[i-1] < L이면 판다."""
    def make(A):
        r = A["r"]
        def rule(i, ei, ent):
            return r[i - 1] < L
        return rule
    return make


def ema_drop(x, m=None):
    """기준 EMA[i-1] / EMA[ei-1] - 1 <= -x 이면 판다. m이 있으면 매수 뒤 m분 안(i-ei <= k)에서만."""
    def make(A):
        ema = A["ema"]
        kb = kbars(A, m) if m else 10 ** 9
        def rule(i, ei, ent):
            return i - ei <= kb and ema[i - 1] <= ema[ei - 1] * (1 - x)
        return rule
    return make


def lower_lows(N, with_lh=False, below=None):
    """i-1에서 끝나는 연속 N번 '저가 < 직전 저가'(with_lh면 고가도 직전보다 낮음), 계단 시작이 매수 봉 이후(i-1-N >= ei).
    below가 있으면 종가[i-1] <= 매수가 x (1-below)도 필요."""
    def make(A):
        l, h, c = A["l"], A["h"], A["c"]
        cond = np.r_[False, l[1:] < l[:-1]]
        if with_lh:
            cond &= np.r_[False, h[1:] < h[:-1]]
        run = runlen(cond)
        def rule(i, ei, ent):
            if run[i - 1] < N or i - 1 - N < ei:
                return False
            return below is None or c[i - 1] <= ent * (1 - below)
        return rule
    return make


def newlow_count(N):
    """매수 뒤 '그때까지 최저 저가 경신' 봉 수(봉 ei+1..i-1)가 N 이상이면 판다."""
    def make(A):
        l = A["l"]
        st = {"ei": -1, "j": -1, "mn": 0.0, "n": 0}
        def rule(i, ei, ent):
            if st["ei"] != ei:
                st.update(ei=ei, j=ei, mn=l[ei], n=0)
            while st["j"] < i - 1:
                st["j"] += 1
                x = l[st["j"]]
                if x < st["mn"]:
                    st["n"] += 1
                    st["mn"] = x
            return st["n"] >= N
        return rule
    return make


def gap(Y, use_low=False, m=None):
    """(종가 또는 저가)[i-1] / EMA[i-1] - 1 <= -Y 이면 판다. m이 있으면 매수 뒤 m분 안에서만."""
    def make(A):
        g = (A["l"] if use_low else A["c"]) / A["ema"] - 1
        kb = kbars(A, m) if m else 10 ** 9
        def rule(i, ei, ent):
            return i - ei <= kb and g[i - 1] <= -Y
        return rule
    return make


def only(sessions, mk):
    """매수 봉 세션이 sessions에 들 때만 규칙 적용."""
    def make(A):
        ok = np.isin(A["ses"], list(sessions))
        rule = mk(A)
        def r(i, ei, ent):
            return bool(ok[ei]) and rule(i, ei, ent)
        return r
    return make


def ema_drop_rsi(x, T=40):
    """EMA[i-1]/EMA[ei-1]-1 <= -x 이고, 매수 뒤 RSI 최고(봉 ei..i-1)가 T 미만이면 판다."""
    def make(A):
        ema, r = A["ema"], A["r"]
        st = {"ei": -1, "j": -1, "mx": 0.0}
        def rule(i, ei, ent):
            if st["ei"] != ei:
                st.update(ei=ei, j=ei, mx=r[ei])
            while st["j"] < i - 1:
                st["j"] += 1
                st["mx"] = max(st["mx"], r[st["j"]])
            return st["mx"] < T and ema[i - 1] <= ema[ei - 1] * (1 - x)
        return rule
    return make


def gap_rsinew(Y):
    """종가[i-1]/EMA[i-1]-1 <= -Y 이고 RSI[i-1]이 매수 뒤 새 저점(<= r[ei..i-2] 최저)이면 판다."""
    def make(A):
        g = A["c"] / A["ema"] - 1
        r = A["r"]
        st = {"ei": -1, "j": -1, "mn": 0.0}
        def rule(i, ei, ent):
            if st["ei"] != ei:
                st.update(ei=ei, j=ei, mn=r[ei])
            new = False
            while st["j"] < i - 1:
                st["j"] += 1
                x = r[st["j"]]
                new = x <= st["mn"]
                st["mn"] = min(st["mn"], x)
            return new and i - 1 > ei and g[i - 1] <= -Y
        return rule
    return make


RULES = []      # (계열, 이름, 공장)
for T in (25, 30, 35, 40):
    for m in (15, 30, 60):
        RULES.append(("A RSI회복실패", f"A_rsi<{T}_{m}분", rsi_fail(T, m)))
for d in (0, 5, 10):
    for m in (0, 15, 30):
        RULES.append(("B RSI새저점", f"B_rsinew_d{d}_{m}분후", rsi_newlow(d, m)))
for L in (15, 20):
    RULES.append(("B RSI새저점", f"B_rsi<{L}", rsi_floor(L)))
for x in (0.01, 0.015, 0.02, 0.03):
    RULES.append(("C EMA하락", f"C_ema-{x:.1%}", ema_drop(x)))
for m, xs in ((30, (0.005, 0.0075, 0.01)), (60, (0.01, 0.015, 0.02))):
    for x in xs:
        RULES.append(("C EMA하락", f"C_ema-{x:.2%}_{m}분안", ema_drop(x, m)))
for N in (3, 4, 5, 6):
    RULES.append(("D 저점계단", f"D_LL{N}&종가<=매수가", lower_lows(N, below=0.0)))
for N in (3, 4, 5):
    RULES.append(("D 저점계단", f"D_LLLH{N}&종가<=매수가", lower_lows(N, with_lh=True, below=0.0)))
for N in (3, 5, 8):
    RULES.append(("D 저점계단", f"D_새저가{N}회", newlow_count(N)))
for Y in (0.03, 0.04, 0.05, 0.06, 0.08):
    RULES.append(("E 이격확대", f"E_종가이격-{Y:.0%}", gap(Y)))
for Y in (0.04, 0.06, 0.08):
    RULES.append(("E 이격확대", f"E_저가이격-{Y:.0%}", gap(Y, use_low=True)))
for Y in (0.03, 0.04, 0.05):
    RULES.append(("E 이격확대", f"E_종가이격-{Y:.0%}_60분안", gap(Y, m=60)))
for Y in (0.04, 0.05, 0.06):
    RULES.append(("F 조합", f"F_정규만_종가이격-{Y:.0%}", only(("정규",), gap(Y))))
for T in (30, 35):
    RULES.append(("F 조합", f"F_프리만_rsi<{T}_60분", only(("프리",), rsi_fail(T, 60))))
for x in (0.01, 0.015, 0.02):
    RULES.append(("F 조합", f"F_ema-{x:.1%}&rsi최고<40", ema_drop_rsi(x, 40)))
for Y in (0.03, 0.04, 0.05):
    RULES.append(("F 조합", f"F_종가이격-{Y:.0%}&RSI새저점", gap_rsinew(Y)))
for N in (3, 4):
    RULES.append(("F 조합", f"F_LL{N}&종가<매수-2%", lower_lows(N, below=0.02)))


# ---------------- 미래 참조 점검 ----------------
def lookahead_check(mk, n=120, seed=7):
    """기준 거래 표본마다 i0(보유 중 임의 봉) 이후 배열을 망가뜨린 A'로 규칙을 만들고, i <= i0 결정이 원래와 같은지 본다."""
    rng = np.random.default_rng(seed)
    S = load("early")
    b = base_trades("early")
    b = b[b.bars >= 3].sample(n, random_state=seed)
    bad = 0
    for x in b.itertuples():
        A = S[x.sym]
        i0 = int(rng.integers(x.ei + 1, x.xi + 1))
        B = dict(A)
        for k in ("o", "h", "l", "c", "v", "ema"):
            B[k] = A[k].copy()
            B[k][i0:] *= rng.uniform(0.5, 1.5, len(B[k]) - i0)
        B["r"] = A["r"].copy()
        B["r"][i0:] = rng.uniform(0, 100, len(B["r"]) - i0)
        ra, rb = mk(A), mk(B)
        for i in range(x.ei + 1, i0 + 1):
            if ra(i, x.ei, x.ent) != rb(i, x.ei, x.ent):
                bad += 1
                break
    return bad


def cell(df, period, scope, un):
    q = df[(df.기간 == period) & (df.범위 == scope) & (df.종목 == un)].iloc[0]
    return q


def main():
    cache = {}
    if os.path.exists(CACHE):
        with open(CACHE, encoding="utf-8") as f:
            cache = json.load(f)
    lines = []

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    rows = []
    for fam, name, mk in RULES:
        if name not in cache:
            df = compare(mk, name, show=False)
            cache[name] = df.to_dict("records")
            with open(CACHE, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, default=str)
        df = pd.DataFrame(cache[name])
        rec = dict(계열=fam, 규칙=name)
        for p in ("early", "late"):
            for sc, tag in (("24시간", "24"), ("정규장만", "정규")):
                for un, ut in (("사용자78", "u"), ("전체137", "a")):
                    q = cell(df, p, sc, un)
                    rec[f"{p[0]}{tag}{ut}"] = q.차이
                    rec[f"{p[0]}{tag}{ut}_t"] = q.t
                    rec[f"{p[0]}{tag}{ut}_손절"] = q.손절
        rows.append(rec)
        print(f"{name}: early24 {rec['e24u']:+.2f} early정규 {rec['e정규u']:+.2f}", flush=True)
    R = pd.DataFrame(rows)

    out("RSI·추세 경로 계열 칼손절 시험 (stops/rsipath.py, 2026-10-07)")
    out("기준(사용자78): 24시간 early +27.12 · late +17.04 / 정규장만 early +1.47 · late +2.28. 전체137 24시간 early +41.35 · late +24.79")
    out("차이 = 규칙 수익 합 - 기준 수익 합(거래별 수익률 합, 1.00 = 100%p). 손절 뒤 다음 아침 8시까지 그 종목 재매수 없음.")
    out(f"시험 규칙 수: {len(R)}")
    out("")
    out("[사용자78] e=early(7/14~9/8, 5분봉 EMA36·RSI9) l=late(9/10~10/6, 3분봉 EMA60·RSI14), 24=24시간, 정규=정규장만")
    hdr = f"{'규칙':34s} {'e24':>7s} {'t':>6s} {'손절':>5s} {'e정규':>7s} {'t':>6s} | {'l24':>7s} {'t':>6s} {'손절':>5s} {'l정규':>7s} {'t':>6s} | {'전체e24':>7s} {'전체l24':>7s}"
    out(hdr)
    for fam in R.계열.unique():
        out(f"-- {fam}")
        for r in R[R.계열 == fam].itertuples():
            d = r._asdict()
            out(f"{r.규칙:34s} {d['e24u']:+7.2f} {d['e24u_t']:+6.2f} {int(d['e24u_손절']):5d} {d['e정규u']:+7.2f} {d['e정규u_t']:+6.2f} | "
                f"{d['l24u']:+7.2f} {d['l24u_t']:+6.2f} {int(d['l24u_손절']):5d} {d['l정규u']:+7.2f} {d['l정규u_t']:+6.2f} | {d['e24a']:+7.2f} {d['l24a']:+7.2f}")
    out("")
    pos_e = R[(R.e24u > 0)]
    four = R[(R.e24u > 0) & (R.e정규u > 0) & (R.l24u > 0) & (R.l정규u > 0)]
    out(f"early 사용자78 24시간 플러스: {len(pos_e)}개 / early 두 칸(24시간·정규장만) 플러스: {int(((R.e24u > 0) & (R.e정규u > 0)).sum())}개")
    out(f"사용자78 네 칸 모두 플러스(late는 확인용 계산): {len(four)}개 {list(four.규칙)}")

    # 선택: early 사용자78 24시간 차이 최대, early 정규장만 차이 >= 0
    cand = R[R.e정규u >= 0]
    pick = (cand if len(cand) else R).sort_values("e24u", ascending=False)
    best = pick.iloc[0]
    out("")
    out(f"선택 규칙(early 24시간 최대, early 정규장만 >= 0 조건): {best.규칙}")
    out("early 순위 상위 8(같은 조건):")
    for r in pick.head(8).itertuples():
        d = r._asdict()
        out(f"  {r.규칙:34s} early24 {d['e24u']:+.2f} early정규 {d['e정규u']:+.2f} | late24 {d['l24u']:+.2f} late정규 {d['l정규u']:+.2f}")
    out("조건 없이 early 24시간 상위 5:")
    for r in R.sort_values("e24u", ascending=False).head(5).itertuples():
        d = r._asdict()
        out(f"  {r.규칙:34s} early24 {d['e24u']:+.2f} early정규 {d['e정규u']:+.2f} | late24 {d['l24u']:+.2f} late정규 {d['l정규u']:+.2f}")

    mk = dict((n, m) for _, n, m in RULES)[best.규칙]
    out("")
    out(f"=== 선택 규칙 전체 표: {best.규칙}")
    df = pd.DataFrame(cache[best.규칙])
    f = df.copy()
    for k, fmt in (("평균", "{:+.2%}"), ("승률", "{:.0%}"), ("최악", "{:+.1%}"), ("하위5", "{:+.1%}"), ("합", "{:+.2f}"),
                   ("차이", "{:+.2f}"), ("t", "{:+.2f}"), ("손절평균", "{:+.2%}"), ("차이_10월6일", "{:+.2f}")):
        f[k] = f[k].map(lambda v: "" if v is None or pd.isna(v) else fmt.format(v))
    for ln in f.drop(columns=["규칙"]).to_string(index=False).splitlines():
        out(ln)
    out("")
    out("stop_audit(사용자78): 헛손절비율 = 손절했지만 기준이면 수익, 아낀손실합 = 손절 수익 - 기준 수익의 합")
    aud = {}
    for p in ("early", "late"):
        for reg in (False, True):
            a = stop_audit(mk, p, reg_only=reg)
            aud[(p, reg)] = a
            out(f"  {p} {'정규장만' if reg else '24시간'}: " + ", ".join(
                f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in a.items()))
    out("")
    out("미래 참조 점검(early 기준 거래 120건 표본, 보유 중 임의 봉 i0 이후 o h l c v ema r를 망가뜨려도 i <= i0 결정이 같은지):")
    bad_total = 0
    for fam, name, m in RULES:
        bad = lookahead_check(m)
        bad_total += bad
        if bad:
            out(f"  불일치 {name}: {bad}건")
    out(f"  전체 {len(RULES)}개 규칙 불일치 합: {bad_total}")
    with open(OUT, "w", encoding="utf-8") as fo:
        fo.write("\n".join(lines) + "\n")
    R.to_json(os.path.join(os.path.dirname(CACHE), "summary.json"), force_ascii=False, orient="records")
    with open(os.path.join(os.path.dirname(CACHE), "audit.json"), "w", encoding="utf-8") as fo:
        json.dump({f"{k[0]}_{k[1]}": v for k, v in aud.items()}, fo, ensure_ascii=False, default=float)
    print("saved", OUT)


if __name__ == "__main__":
    main()
