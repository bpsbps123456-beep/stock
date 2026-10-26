"""칼손절 연구 — 기준봉 RSI + 40분봉 RSI 결합 계열(g_rsiboth, 2026-10-07).

사용자 가설(10/7): "40분봉 RSI가 30 이하(이미 과매도)면 손절하면 안 좋다."
→ 손절 신호가 나와도 40분봉 RSI가 낮으면 보류, 40분봉 RSI가 아직 높은데(더 빠질 여지) 빠질 때만 자른다.
기준봉 RSI(early 5분 RSI9 / late 3분 RSI14)의 흘러내림 표시와 40분봉 RSI(dev, RSI14, 04:00 ET 기준)를 같이 본다.

시험대: ../stop_lab.py(고치지 않음). 모든 규칙은 봉 i 시가에 팔지를 정한다.
  기준봉 값은 i-1까지(r[i-1], c[i-1], r[ei..i-1]), 40분봉 값은 htf_rsi의 인덱스 i까지(dev[i] = c[i-1]을 넣은 진행 중 40분 RSI).
  매수 시점 40분 RSI = dev[ei](매수 봉 시가 때 값, c[ei-1] 기준). k분 = max(1, k // 봉 길이) 봉.
  40분 값이 NaN이면 게이트·반대 게이트 모두 '팔지 않음'.
게이트 모드: gate = 40분 dev[i] > g 일 때만 자름(사용자 가설), none = 40분 조건 없음(같은 손절), rev = dev[i] <= g 일 때만 자름(반대).
매개변수 선택(사전 고정): early 사용자78 24시간 '차이' 최대, 단 early 사용자78 정규장만 '차이' >= 0. late는 전방 확인으로만 본다.
사용: py stops/g_rsiboth.py  → ../results/stops/g_rsiboth.txt"""
import os, sys, json
from datetime import date
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from stop_lab import load, compare, stop_audit, base_trades, run, htf_rsi, UNIV, RAW  # noqa: E402

OUT = os.path.normpath(os.path.join(HERE, "../../results/stops/g_rsiboth.txt"))
SCR = r"C:/Users/geunt/AppData/Local/Temp/claude/C--Users-geunt-OneDrive-Desktop---/4e686c73-74be-4337-8ccc-2f6136beaf7d/scratchpad/g_rsiboth"
CACHE = os.path.join(SCR, "cmp_cache.json")
U78 = UNIV["사용자78"]


def kbars(A, m):
    return max(1, m // A["mins"])


def H(A, src="dev"):
    """40분봉 RSI 배열. 미래 참조 점검 때는 A['_dev'], A['_conf'](망가뜨린 사본)를 쓴다."""
    if "_dev" in A:
        return A["_dev"] if src == "dev" else A["_conf"]
    dev, conf = htf_rsi(A, 40, 14)
    return dev if src == "dev" else conf


def gate_ok(v, g, mode):
    if mode == "none":
        return True
    if v != v:              # NaN
        return False
    return v > g if mode == "gate" else v <= g


# ---------------- 규칙 공장 ----------------
def rsifail(T, m, g, mode="gate", src="dev", price=None):
    """A: 매수 뒤 m분(봉 ei..i-1, i-ei == k) 동안 기준 RSI 최고 < T(회복 실패)이고 40분 게이트면 봉 i 시가에 판다(한 번만 확인).
    price가 있으면 종가[i-1] <= 매수가 x (1-price)도 필요."""
    def make(A):
        r, c, kb, h = A["r"], A["c"], kbars(A, m), H(A, src)

        def rule(i, ei, ent):
            if i - ei != kb or r[ei:i].max() >= T:
                return False
            if price is not None and c[i - 1] > ent * (1 - price):
                return False
            return gate_ok(h[i], g, mode)
        return rule
    return make


def dropfrom(x, g, mode="gate", L=None, price=None):
    """B: 40분 dev[i] <= dev[ei] - x (매수 시점보다 x 넘게 하락)이고 40분 게이트면 판다(보유 중 매 봉 확인).
    L이 있으면 기준 RSI[i-1] < L, price가 있으면 종가[i-1] <= 매수가 x (1-price)도 필요."""
    def make(A):
        r, c, h = A["r"], A["c"], H(A)

        def rule(i, ei, ent):
            d0, d = h[ei], h[i]
            if d0 != d0 or d != d or d > d0 - x:
                return False
            if L is not None and r[i - 1] >= L:
                return False
            if price is not None and c[i - 1] > ent * (1 - price):
                return False
            return gate_ok(d, g, mode)
        return rule
    return make


def slope(x, g, mode="gate", L=None):
    """C: 최근 40분 동안 40분 dev 하락: dev[i] - dev[i-b] <= -x (b = 40 // 봉 길이, i-b >= ei), 기준 RSI[i-1] < L(있으면), 40분 게이트."""
    def make(A):
        r, h, b = A["r"], H(A), max(1, 40 // A["mins"])

        def rule(i, ei, ent):
            if i - b < ei:
                return False
            d, d0 = h[i], h[i - b]
            if d != d or d0 != d0 or d - d0 > -x:
                return False
            if L is not None and r[i - 1] >= L:
                return False
            return gate_ok(d, g, mode)
        return rule
    return make


def conf_fall2(g, mode="gate"):
    """C2: 매수 뒤 새로 완성된 40분봉 RSI(conf)가 연속 2번 직전보다 낮아짐(conf[ei..i]에서 값이 바뀐 지점 기준)이고 40분 dev 게이트."""
    def make(A):
        cf, h = H(A, "conf"), H(A)
        st = {"ei": -1, "j": -1, "run": 0}

        def rule(i, ei, ent):
            if st["ei"] != ei:
                st.update(ei=ei, j=ei, run=0)
            while st["j"] < i:
                st["j"] += 1
                j = st["j"]
                a, b = cf[j - 1], cf[j]
                if a == a and b == b and b != a:
                    st["run"] = st["run"] + 1 if b < a else 0
            return st["run"] >= 2 and gate_ok(h[i], g, mode)
        return rule
    return make


def price_stop(x):
    def make(A):
        c = A["c"]
        return lambda i, ei, ent: c[i - 1] <= ent * (1 - x)
    return make


def rsinew(d):
    """rsipath B_rsinew_d{d}_0분후와 같음: RSI[i-1]이 매수 뒤 새 저점(r[ei..i-2] 최저보다 낮음)이고 매수 봉 RSI보다 d 넘게 낮음."""
    def make(A):
        r = A["r"]
        st = {"ei": -1, "j": -1, "mn": 0.0}

        def rule(i, ei, ent):
            if st["ei"] != ei:
                st.update(ei=ei, j=ei, mn=r[ei])
            new = False
            while st["j"] < i - 1:
                st["j"] += 1
                x = r[st["j"]]
                new = x < st["mn"]
                st["mn"] = min(st["mn"], x)
            return new and r[i - 1] < r[ei] - d
        return rule
    return make


def gapstop(Y):
    def make(A):
        gp = A["c"] / A["ema"] - 1
        return lambda i, ei, ent: gp[i - 1] <= -Y
    return make


BASES = {"종가-2%": price_stop(0.02), "종가-3%": price_stop(0.03), "종가-5%": price_stop(0.05),
         "rsi<30_30분": rsifail(30, 30, 0, "none"), "rsinew_d10": rsinew(10), "종가이격-5%": gapstop(0.05)}


def release(base, R, mode="gate"):
    """D: 기본 손절 base가 켜져도, 매수 뒤 40분 dev 최저(dev[ei..i])가 R 이하로 한 번이라도 닿았으면 그 거래는 손절 영구 해제.
    gate = 한 번도 R 이하에 안 닿았을 때만 자름, rev = 닿은 적 있을 때만 자름, none = base 그대로."""
    def make(A):
        h, rule0 = H(A), BASES[base](A)
        st = {"ei": -1, "j": -1, "mn": np.inf}

        def rule(i, ei, ent):
            if st["ei"] != ei:
                st.update(ei=ei, j=ei - 1, mn=np.inf)
            while st["j"] < i:
                st["j"] += 1
                v = h[st["j"]]
                if v == v:
                    st["mn"] = min(st["mn"], v)
            if not rule0(i, ei, ent):
                return False
            if mode == "none":
                return True
            if st["mn"] == np.inf:
                return False
            return st["mn"] > R if mode == "gate" else st["mn"] <= R
        return rule
    return make


def pxrsi(x, g, mode="gate", B=None):
    """E: 종가[i-1] <= 매수가 x (1-x) 이고, (B가 있으면) 기준 RSI[i-1] > B(기준봉도 과매도 아님 = 흘러내림), 40분 게이트."""
    def make(A):
        c, r, h = A["c"], A["r"], H(A)

        def rule(i, ei, ent):
            if c[i - 1] > ent * (1 - x):
                return False
            if B is not None and r[i - 1] <= B:
                return False
            return gate_ok(h[i], g, mode)
        return rule
    return make


# 규칙 목록: (계열, 이름, 공장, (ungated 이름·공장), (reverse 이름·공장))
SPEC = []


def add(fam, name, f, *args, **kw):
    SPEC.append((fam, name, f, args, kw))


for T in (30, 35):
    for m in (15, 30, 60):
        for g in (30, 35, 40):
            add("A 기준RSI회복실패+40분", f"A_rsi<{T}_{m}분&d40>{g}", rsifail, T, m, g)
for m in (15, 30):
    for g in (35, 40):
        add("A 기준RSI회복실패+40분", f"A_rsi<35_{m}분&종가-2%&d40>{g}", rsifail, 35, m, g, price=0.02)
for x in (5, 10, 15):
    for g in (30, 35, 40):
        add("B 40분 매수대비하락", f"B_d40하락{x}&d40>{g}", dropfrom, x, g)
for x in (5, 10):
    for g in (30, 35, 40):
        add("B 40분 매수대비하락", f"B_d40하락{x}&rsi<30&d40>{g}", dropfrom, x, g, L=30)
for x in (5, 10):
    for g in (30, 35):
        add("B 40분 매수대비하락", f"B_d40하락{x}&종가-2%&d40>{g}", dropfrom, x, g, price=0.02)
for x in (5, 10):
    for g in (30, 35, 40):
        add("C 40분 기울기", f"C_40분새d40-{x}&rsi<35&d40>{g}", slope, x, g, L=35)
for x in (5, 10):
    add("C 40분 기울기", f"C_40분새d40-{x}&d40>35", slope, x, 35)
for g in (30, 35, 40):
    add("C 40분 기울기", f"C2_conf2연속하락&d40>{g}", conf_fall2, g)
for base in BASES:
    for R in (30, 35):
        add("D 40분<=R 닿으면 영구해제", f"D_{base}_해제{R}", release, base, R)
for x in (0.02, 0.03, 0.05):
    for B in (None, 30):
        for g in (30, 35, 40):
            nm = f"E_종가-{x:.0%}" + ("" if B is None else f"&rsi>{B}") + f"&d40>{g}"
            add("E 가격+기준RSI+40분", nm, pxrsi, x, g, B=B)


def mk_of(spec, mode="gate"):
    """규칙 공장 호출. mode: gate(가설) / none(40분 조건 제거) / rev(조건 뒤집기)."""
    fam, name, f, args, kw = spec
    return f(*args, mode=mode, **kw)


# ---------------- 미래 참조 점검 ----------------
def lookahead_check(mk, n=150, seed=7, period="early"):
    """기준 거래 표본마다 보유 중 임의 봉 i0를 고르고, o h l c v ema r 의 [i0:]와 40분 dev·conf의 [i0+1:]를 망가뜨린 A'로
    규칙을 만들어 i <= i0 결정이 원래와 같은지 본다(40분 값은 인덱스 i까지 써도 됨)."""
    rng = np.random.default_rng(seed)
    S = load(period)
    b = base_trades(period)
    b = b[b.bars >= 3].sample(n, random_state=seed)
    bad = 0
    for x in b.itertuples():
        A = dict(S[x.sym])
        dev, conf = htf_rsi(A, 40, 14)
        A["_dev"], A["_conf"] = dev, conf
        i0 = int(rng.integers(x.ei + 1, x.xi + 1))
        B = dict(A)
        for k in ("o", "h", "l", "c", "v", "ema"):
            B[k] = A[k].copy()
            B[k][i0:] *= rng.uniform(0.5, 1.5, len(B[k]) - i0)
        for k, lo in (("r", i0), ("_dev", i0 + 1), ("_conf", i0 + 1)):
            B[k] = A[k].copy()
            B[k][lo:] = rng.uniform(0, 100, len(B[k]) - lo)
        ra, rb = mk(A), mk(B)
        for i in range(x.ei + 1, i0 + 1):
            if ra(i, x.ei, x.ent) != rb(i, x.ei, x.ent):
                bad += 1
                break
    return bad


def htf_truncation_check(period, n=150, seed=11):
    """htf_rsi dev[i]를 '봉 i-1이 속한 40분봉 시작 전에 끝난 5분봉만'으로 다시 만든 값과 비교(5분봉 원자료 미래 참조 점검)."""
    from htf import intraday_bars, rma_state, _rsi_from
    rng = np.random.default_rng(seed)
    S = load(period)
    b = base_trades(period)
    b = b[b.sym.isin(U78)].sample(n, random_state=seed)
    diffs = []
    for x in b.itertuples():
        A = S[x.sym]
        dev, _ = htf_rsi(A, 40, 14)
        i = int(rng.integers(x.ei, x.xi + 1))
        m5 = pd.read_parquet(os.path.join(RAW, f"5m_{x.sym}.parquet"))
        t_prev = A["idx"][i - 1]
        anchor = t_prev.normalize() + pd.Timedelta(hours=4)
        cur = anchor + ((t_prev - anchor) // pd.Timedelta(minutes=40)) * pd.Timedelta(minutes=40)
        cut = m5[m5.index < cur]                      # 봉 i-1이 속한 40분봉 시작 전 5분봉만
        hs, hc = intraday_bars(cut, 40)
        g, l, r = rma_state(hc.astype(float), 14)
        v = _rsi_from(np.array([g[-1]]), np.array([l[-1]]), np.array([hc[-1]]), np.array([A["c"][i - 1]]), 14)[0]
        if np.isnan(v) and np.isnan(dev[i]):
            continue
        diffs.append(abs(v - dev[i]))
    diffs = np.array(diffs)
    return len(diffs), float(np.nanmax(diffs)) if len(diffs) else 0.0, int((diffs > 1e-6).sum())


def cell(df, p, sc, un):
    return df[(df.기간 == p) & (df.범위 == sc) & (df.종목 == un)].iloc[0]


def day_dep(mk, period, reg_only=False):
    t = run(load(period), mk, period, reg_only, syms=U78)
    b = base_trades(period, reg_only)
    b = b[b.sym.isin(U78)]
    days = sorted(set(t.day) | set(b.day))
    d = t.groupby("day").ret.sum().reindex(days, fill_value=0) - b.groupby("day").ret.sum().reindex(days, fill_value=0)
    return d


def main():
    os.makedirs(SCR, exist_ok=True)
    cache = {}
    if os.path.exists(CACHE):
        with open(CACHE, encoding="utf-8") as f:
            cache = json.load(f)
    lines = []

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    def get(name, mk):
        if name not in cache:
            df = compare(mk, name, show=False)
            cache[name] = df.to_dict("records")
            with open(CACHE, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, default=str)
        return pd.DataFrame(cache[name])

    def rec_of(fam, name, df):
        rec = dict(계열=fam, 규칙=name)
        for p in ("early", "late"):
            for sc, tag in (("24시간", "24"), ("정규장만", "정규")):
                for un, ut in (("사용자78", "u"), ("전체137", "a")):
                    q = cell(df, p, sc, un)
                    rec[f"{p[0]}{tag}{ut}"] = q.차이
                    rec[f"{p[0]}{tag}{ut}_t"] = q.t
                    rec[f"{p[0]}{tag}{ut}_손절"] = q.손절
        return rec

    rows = []
    for spec in SPEC:
        fam, name = spec[0], spec[1]
        df = get(name, mk_of(spec, "gate"))
        rec = rec_of(fam, name, df)
        rows.append(rec)
        print(f"{name}: early24 {rec['e24u']:+.2f} early정규 {rec['e정규u']:+.2f}", flush=True)
    R = pd.DataFrame(rows)

    cand = R[R.e정규u >= 0]
    pick = (cand if len(cand) else R).sort_values("e24u", ascending=False)
    best_name = pick.iloc[0].규칙
    spec_of = {s[1]: s for s in SPEC}

    # 비교: 최선안과 차순위 2개의 '조건 없는 같은 손절'과 '반대 조건'
    extra = {}
    for nm in list(pick.head(3).규칙):
        sp = spec_of[nm]
        for mode, tag in (("none", "[무조건]"), ("rev", "[반대]")):
            en = f"{nm}{tag}"
            extra[en] = rec_of(sp[0], en, get(en, mk_of(sp, mode)))
    # 가설 직접 비교(사전에 정한 추가분): 사용자 문장 그대로인 '가격 손절 + d40 > 30'의 무조건·반대, '종가-3% & d40 > 35'의 반대
    for nm, modes in (("E_종가-2%&d40>30", ("none", "rev")), ("E_종가-3%&d40>30", ("none", "rev")), ("E_종가-3%&d40>35", ("rev",))):
        sp = spec_of[nm]
        for mode in modes:
            en = f"{nm}{'[무조건]' if mode == 'none' else '[반대]'}"
            if en not in extra:
                extra[en] = rec_of(sp[0], en, get(en, mk_of(sp, mode)))
    X = pd.DataFrame(list(extra.values()))
    n_tested = len(R) + len(X)

    out("기준봉 RSI + 40분봉 RSI 결합 계열 칼손절 시험 (stops/g_rsiboth.py, 2026-10-07)")
    out("사용자 가설: 40분봉 RSI가 30 이하(이미 과매도)면 손절하지 않고, 40분봉 RSI가 아직 높은데 빠질 때만 자른다.")
    out("기준(사용자78): 24시간 early +27.12 · late +17.04 / 정규장만 early +1.47 · late +2.28. 전체137 24시간 early +41.35 · late +24.79")
    out("차이 = 규칙 수익 합 - 기준 수익 합(거래별 수익률 합, 1.00 = 100%p). 손절 뒤 다음 아침 8시까지 그 종목 재매수 없음.")
    out("d40 = 40분봉 RSI14 진행 중 값 dev[i](봉 i-1 종가를 넣은 값, 04:00 ET 기준 40분봉, 5분봉으로 만듦). 매수 시점 d40 = dev[ei].")
    out("rsi = 기준봉 RSI(early 5분 RSI9, late 3분 RSI14), 종가 = 봉 i-1 종가. 모든 규칙은 봉 i 시가에 판다.")
    out(f"시험 규칙 수: 격자 {len(R)}개 + 비교(무조건·반대) {len(X)}개 = {n_tested}개")
    out("")
    out("[사용자78] e=early(7/14~9/8, 5분봉 EMA36·RSI9) l=late(9/10~10/6, 3분봉 EMA60·RSI14), 24=24시간, 정규=정규장만")
    hdr = f"{'규칙':34s} {'e24':>7s} {'t':>6s} {'손절':>5s} {'e정규':>7s} {'t':>6s} | {'l24':>7s} {'t':>6s} {'손절':>5s} {'l정규':>7s} {'t':>6s} | {'전체e24':>7s} {'전체l24':>7s}"
    out(hdr)

    def line(r):
        d = r if isinstance(r, dict) else r._asdict()
        return (f"{d['규칙']:34s} {d['e24u']:+7.2f} {d['e24u_t']:+6.2f} {int(d['e24u_손절']):5d} {d['e정규u']:+7.2f} {d['e정규u_t']:+6.2f} | "
                f"{d['l24u']:+7.2f} {d['l24u_t']:+6.2f} {int(d['l24u_손절']):5d} {d['l정규u']:+7.2f} {d['l정규u_t']:+6.2f} | {d['e24a']:+7.2f} {d['l24a']:+7.2f}")
    for fam in R.계열.unique():
        out(f"-- {fam}")
        for r in R[R.계열 == fam].itertuples():
            out(line(r))
    out("-- 비교(최선안·차순위 2개 + 가설 직접형 E_종가-2%/-3%: 40분 조건 제거 = [무조건], 조건 뒤집기 d40 <= g = [반대])")
    for r in X.itertuples():
        out(line(r))
    out("")
    out("게이트 효과 분리(사용자78 차이; 게이트-무조건 = 40분 조건이 더한 몫, 손절 1건당 차이 = 차이/손절 수, %p):")
    allrec = {r["규칙"]: r for r in rows}
    allrec.update(extra)
    for nm in sorted({k.split("[")[0] for k in extra}):
        g0, n0, v0 = allrec.get(nm), allrec.get(nm + "[무조건]"), allrec.get(nm + "[반대]")
        for cel, tag in (("e24u", "early 24시간"), ("e정규u", "early 정규장만"), ("l24u", "late 24시간"), ("l정규u", "late 정규장만")):
            parts = []
            for lab, q in (("게이트", g0), ("무조건", n0), ("반대", v0)):
                if q is not None:
                    ns = q[cel + "_손절"]
                    parts.append(f"{lab} {q[cel]:+.2f}({int(ns)}건, 건당 {100 * q[cel] / ns if ns else 0:+.2f})")
            eff = f" | 게이트-무조건 {g0[cel] - n0[cel]:+.2f}" if n0 is not None else ""
            out(f"  {nm:26s} {tag:12s} " + " / ".join(parts) + eff)
    out("")
    four = R[(R.e24u > 0) & (R.e정규u > 0) & (R.l24u > 0) & (R.l정규u > 0)]
    fourX = X[(X.e24u > 0) & (X.e정규u > 0) & (X.l24u > 0) & (X.l정규u > 0)]
    out(f"early 사용자78 24시간 플러스: {int((R.e24u > 0).sum())}개 / early 두 칸 플러스: {int(((R.e24u > 0) & (R.e정규u > 0)).sum())}개 (격자 {len(R)}개 중)")
    out(f"사용자78 네 칸 모두 플러스(late는 확인용 계산): 격자 {len(four)}개 {list(four.규칙)} / 비교 {len(fourX)}개 {list(fourX.규칙)}")

    # 게이트 효과 요약: 같은 손절을 무조건·게이트·반대로 나눴을 때(가산성 점검용)
    out("")
    out(f"선택 규칙(early 사용자78 24시간 최대, early 정규장만 >= 0 조건): {best_name}")
    out("early 순위 상위 8(같은 조건):")
    for r in pick.head(8).itertuples():
        d = r._asdict()
        out(f"  {r.규칙:34s} early24 {d['e24u']:+.2f} early정규 {d['e정규u']:+.2f} | late24 {d['l24u']:+.2f} late정규 {d['l정규u']:+.2f} | 전체 e24 {d['e24a']:+.2f} l24 {d['l24a']:+.2f}")
    out("계열별 early 최선(같은 조건):")
    for fam in R.계열.unique():
        q = pick[pick.계열 == fam]
        if len(q):
            d = q.iloc[0]
            out(f"  {fam}: {d.규칙} early24 {d.e24u:+.2f} early정규 {d.e정규u:+.2f} | late24 {d.l24u:+.2f} late정규 {d.l정규u:+.2f}")

    sp = spec_of[best_name]
    mk = mk_of(sp, "gate")
    out("")
    out(f"=== 선택 규칙 전체 표: {best_name}")
    df = pd.DataFrame(cache[best_name])
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
            for mode, tag in (("gate", "게이트"), ("none", "무조건"), ("rev", "반대")):
                a = stop_audit(mk_of(sp, mode), p, reg_only=reg)
                aud[f"{p}_{'reg' if reg else '24'}_{mode}"] = a
                out(f"  {p} {'정규장만' if reg else '24시간'} {tag}: " + ", ".join(
                    f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in a.items()))
    out("")
    out("날짜 집중(사용자78 24시간, 선택 규칙): 차이 상위 날 빼면")
    dd = {}
    for p in ("early", "late"):
        d = day_dep(mk, p)
        dd[p] = d
        top = d.sort_values(ascending=False)
        out(f"  {p}: 합 {d.sum():+.2f}, 나은 날 {int((d > 1e-12).sum())} 나쁜 날 {int((d < -1e-12).sum())}, "
            f"상위 3일 {', '.join(f'{k} {v:+.2f}' for k, v in top.head(3).items())}, 최상위 1일 빼면 {d.sum() - top.iloc[0]:+.2f}, 2일 빼면 {d.sum() - top.iloc[:2].sum():+.2f}")
    nm2 = pick.iloc[1].규칙
    mk2 = mk_of(spec_of[nm2], "gate")
    out(f"차순위 {nm2} stop_audit(사용자78, 게이트):")
    for p in ("early", "late"):
        for reg in (False, True):
            a = stop_audit(mk2, p, reg_only=reg)
            aud[f"second_{p}_{'reg' if reg else '24'}"] = a
            out(f"  {p} {'정규장만' if reg else '24시간'}: " + ", ".join(
                f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in a.items()))
    for p in ("early", "late"):
        d = day_dep(mk2, p)
        top = d.sort_values(ascending=False)
        out(f"  날짜 집중 {p}: 합 {d.sum():+.2f}, 나은 날 {int((d > 1e-12).sum())} 나쁜 날 {int((d < -1e-12).sum())}, "
            f"상위 3일 {', '.join(f'{k} {v:+.2f}' for k, v in top.head(3).items())}, 최상위 1일 빼면 {d.sum() - top.iloc[0]:+.2f}, 2일 빼면 {d.sum() - top.iloc[:2].sum():+.2f}")
    out("")
    out("미래 참조 점검 1(early·late 기준 거래 각 150건, 보유 중 임의 봉 i0 이후 o h l c v ema r [i0:]·40분 dev conf [i0+1:]를 망가뜨려도 i <= i0 결정이 같은지):")
    bad_total = 0
    for spec in SPEC:
        for p in ("early", "late"):
            bad = lookahead_check(mk_of(spec, "gate"), period=p)
            bad_total += bad
            if bad:
                out(f"  불일치 {spec[1]} {p}: {bad}건")
    out(f"  전체 {len(SPEC)}개 규칙 x 2기간 불일치 합: {bad_total}")
    out("미래 참조 점검 2(htf_rsi dev[i]를 '봉 i-1이 속한 40분봉 시작 전 5분봉만'으로 다시 만든 값과 비교, 사용자78 기준 거래 150건씩):")
    chk = {}
    for p in ("early", "late"):
        nn, mx, nbad = htf_truncation_check(p)
        chk[p] = (nn, mx, nbad)
        out(f"  {p}: 비교 {nn}건, 최대 차이 {mx:.2e}, 1e-6 넘는 건 {nbad}")
    with open(OUT, "w", encoding="utf-8") as fo:
        fo.write("\n".join(lines) + "\n")
    R.to_json(os.path.join(SCR, "summary.json"), force_ascii=False, orient="records")
    X.to_json(os.path.join(SCR, "extra.json"), force_ascii=False, orient="records")
    with open(os.path.join(SCR, "audit.json"), "w", encoding="utf-8") as fo:
        json.dump(dict(best=best_name, n_tested=n_tested, four=len(four), fourX=len(fourX), audit=aud,
                       la_bad=bad_total, htf_chk=chk), fo, ensure_ascii=False, default=float)
    print("saved", OUT)


if __name__ == "__main__":
    main()
