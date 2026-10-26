"""칼손절 연구 — 거래량(투매) 계열(2026-10-07). 지금 지표 매수 뒤, 정규장 '거래량 실린 하락'을 보고 봉 i 시가에 판다.

정규장 봉만 신호가 된다(yfinance 프리·애프터 거래량은 0). 신호 봉 j = i-1(종가·거래량을 다 아는 마지막 봉), j >= 매수 봉 ei.
거래량 기준선(둘 다 j 이전 값만):
  TOD  같은 시각 지난 10거래일 평균(오늘 제외, 최소 3일). 분 단위 표(날짜 x 9:30~15:59)로 만든다:
       5m 파일(7/14~)은 5분봉 거래량/5를 그 5분에 고르게 나누고, 1m 파일이 있는 날(9/9~)은 1분봉 거래량 그대로.
       봉의 기준선 = 그 봉이 덮는 분(3분봉 3개, 5분봉 5개)의 지난 날 평균 합.
  R20  직전 정규장 20봉 거래량 평균(날을 넘어 이어 붙임, 봉 j 제외).
규칙 묶음:
  V1 투매봉        j가 음봉(c<o) & 거래량 >= k x 기준선 & c[j] <= 매수가 x (1-g)
  V2 장대 음봉     몸통 c/o-1 <= -x(3분봉 기준, 5분봉은 x*sqrt(5/3)) & 거래량 >= k x TOD & c[j] < 매수가
  V3 거래량 증가 하락  최근 n봉 모두 정규장·매수 뒤·음봉, 거래량이 봉마다 증가, (선택) j 거래량 >= f x TOD, c[j] <= 매수가 x (1-g)
  V4 매수 뒤 무거운 하락  최근 W분 봉이 모두 정규장·매수 뒤, 그 거래량 합 >= k x TOD 합, c[j] <= 매수가 x (1-y)
  V5 매도 거래량 우위   매수 뒤 정규장 봉 >= 3개, 그중 음봉 거래량 / 전체 >= p, c[j] <= 매수가 x (1-y)
  V6 투매 뒤 반등 없음  j-1이 투매봉(음봉 & 거래량 >= k x TOD, j-1 >= ei), j(정규장) 종가 < j-1 종가(또는 저가), c[j] < 매수가
  V7 조용한 흘러내림    최근 30분 봉이 모두 정규장·매수 뒤, 거래량 합 <= k x TOD 합인데 c[j] <= 매수가 x (1-y)
2단계: 1단계 early 상위 3개에 변형 3가지(skip1 매수 봉 신호 제외 / 정규매수 정규장 매수에만 / 60분안 매수 뒤 60분 안만).
매개변수는 early(사용자78 정규장만 차이 >= 0 중 24시간 차이 최대)로만 고른다. late는 확인용.
사용: py stops/volume.py  → ../results/stops/volume.txt"""
import os, sys, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from stop_lab import load, compare, stop_audit, base_trades, RAW, UNIV

OUT = os.path.join(HERE, "..", "..", "results", "stops", "volume.txt")
_F = {}


def _minute_table(sym):
    """날짜 x 정규장 분(390칸) 거래량. 1m 있는 날은 1m, 그 밖은 5m/5."""
    m5 = pd.read_parquet(f"{RAW}/5m_{sym}.parquet")
    mod = m5.index.hour * 60 + m5.index.minute
    m5 = m5[(mod >= 570) & (mod < 960)]
    mod = m5.index.hour * 60 + m5.index.minute
    days = np.asarray(m5.index.date)
    ud = sorted(set(days))
    ri = pd.Index(ud).get_indexer(days)
    vv = m5.Volume.values.astype(float) / 5
    arr = np.zeros((len(ud), 390))
    for k in range(5):
        col = mod.values + k - 570
        ok = col < 390
        arr[ri[ok], col[ok]] = vv[ok]
    p1 = f"{RAW}/1m_{sym}.parquet"
    if os.path.exists(p1):
        m1 = pd.read_parquet(p1)
        md = m1.index.hour * 60 + m1.index.minute
        m1 = m1[(md >= 570) & (md < 960)]
        md = (m1.index.hour * 60 + m1.index.minute).values
        d1 = np.asarray(m1.index.date)
        for d in sorted(set(d1)):
            if d not in ud:
                continue
            r = ud.index(d)
            arr[r, :] = 0.0
            sel = d1 == d
            arr[r, md[sel] - 570] = m1.Volume.values[sel].astype(float)
    return pd.DataFrame(arr, index=ud, columns=range(570, 960))


def feats(A, cut=None, cache=True):
    """종목 배열 묶음에서 거래량 특성. cut(ns)이 있으면 그 시각 이전 원자료만 쓴다(미래 참조 검사용)."""
    key = (A["sym"], A["mins"], len(A["c"]), A.get("_tag"))
    cache = cache and cut is None
    if cache and key in _F:
        return _F[key]
    sym, mins = A["sym"], A["mins"]
    o, c, v, reg, mod, day = A["o"], A["c"], A["v"], A["reg"], A["mod"], A["day"]
    n = len(c)
    # TOD 기준선
    T = _minute_table(sym)
    if cut is not None:   # 검사용: cut 이후 날짜 행 제거(오늘 행은 shift로 어차피 안 씀)
        cd = pd.Timestamp(cut, tz="UTC").tz_convert("America/New_York").date()
        T = T[T.index <= cd]
    B = T.shift(1).rolling(10, min_periods=3).mean()
    cs = np.concatenate([np.zeros((len(B), 1)), np.nancumsum(B.values, axis=1)], axis=1)
    nanrow = B.isna().any(axis=1).values
    pos = pd.Index(B.index).get_indexer(day)
    tod = np.full(n, np.nan)
    ok = reg & (pos >= 0)
    a = mod[ok] - 570
    b = np.minimum(a + mins, 390)
    val = cs[pos[ok], b] - cs[pos[ok], a]
    val[nanrow[pos[ok]]] = np.nan
    tod[ok] = val
    tod[tod <= 0] = np.nan
    # R20 기준선
    ri = np.flatnonzero(reg)
    vr = pd.Series(v[ri]).rolling(20, min_periods=20).mean().shift(1).values
    r20 = np.full(n, np.nan)
    r20[ri] = vr
    r20[r20 <= 0] = np.nan
    red = (c < o) & reg
    F = dict(rv_tod=v / tod, rv_r20=v / r20, red=red, body=c / o - 1, tod=tod,
             cv=np.cumsum(np.where(reg, v, 0.0)), cdv=np.cumsum(np.where(red, v, 0.0)),
             creg=np.cumsum(reg.astype(np.int64)), ctod=np.cumsum(np.where(reg & ~np.isnan(tod), tod, 0.0)),
             ctodn=np.cumsum((reg & ~np.isnan(tod)).astype(np.int64)))
    if cache:
        _F[key] = F
    return F


# ---------------- 규칙 묶음(봉 i 시가에 팔지; i-1까지만) ----------------
def V1(k, g, base="tod", skip=0):
    def make(A):
        F, c = feats(A), A["c"]
        sig = F["red"] & (np.nan_to_num(F["rv_" + base]) >= k)
        def rule(i, ei, ent):
            j = i - 1
            return j >= ei + skip and sig[j] and c[j] <= ent * (1 - g)
        return rule
    return make


def V2(x, k, g=0.0):
    def make(A):
        F, c = feats(A), A["c"]
        xx = x * np.sqrt(A["mins"] / 3)
        sig = A["reg"] & (F["body"] <= -xx) & (np.nan_to_num(F["rv_tod"]) >= k)
        def rule(i, ei, ent):
            j = i - 1
            return j >= ei and sig[j] and c[j] < ent * (1 - g)
        return rule
    return make


def V3(n, f, g):
    def make(A):
        F, c, v = feats(A), A["c"], A["v"]
        N, red = len(c), F["red"]
        up = np.r_[False, v[1:] > v[:-1]]
        sig = red.copy()
        for q in range(1, n):            # j-q 봉도 정규장 음봉
            sig[q:] &= red[:-q]
            sig[:q] = False
        for q in range(n - 1):           # v[j-q] > v[j-q-1]
            sig[q:] &= up[:N - q]
        if f > 0:
            sig &= np.nan_to_num(F["rv_tod"]) >= f
        def rule(i, ei, ent):
            j = i - 1
            return j - n + 1 >= ei and sig[j] and c[j] <= ent * (1 - g)
        return rule
    return make


def V4(W, y, k):
    def make(A):
        F, c = feats(A), A["c"]
        w = max(1, W // A["mins"])
        ctod, ctodn, cv, creg = F["ctod"], F["ctodn"], F["cv"], F["creg"]
        def rule(i, ei, ent):
            j = i - 1
            s = j - w + 1
            if s < ei or c[j] > ent * (1 - y):
                return False
            pv = s - 1
            if creg[j] - creg[pv] != w or ctodn[j] - ctodn[pv] != w:   # 창 전체가 정규장 + 기준선 있음
                return False
            return (cv[j] - cv[pv]) >= k * (ctod[j] - ctod[pv])
        return rule
    return make


def V5(p, y):
    def make(A):
        F, c = feats(A), A["c"]
        cv, cdv, creg = F["cv"], F["cdv"], F["creg"]
        def rule(i, ei, ent):
            j = i - 1
            if j < ei or c[j] > ent * (1 - y):
                return False
            pv = ei - 1
            if creg[j] - creg[pv] < 3:
                return False
            tv = cv[j] - cv[pv]
            return tv > 0 and (cdv[j] - cdv[pv]) >= p * tv
        return rule
    return make


def V6(k, cont):
    """투매봉(j-1) 다음 봉 j가 반등 못 함: j 종가 < 투매봉 종가(cont='c') 또는 저가(cont='l')."""
    def make(A):
        F, c, l = feats(A), A["c"], A["l"]
        heavy = F["red"] & (np.nan_to_num(F["rv_tod"]) >= k)
        ref = c if cont == "c" else l
        sig = np.zeros(len(c), bool)
        sig[1:] = heavy[:-1] & A["reg"][1:] & (c[1:] < ref[:-1])
        def rule(i, ei, ent):
            j = i - 1
            return j - 1 >= ei and sig[j] and c[j] < ent
        return rule
    return make


def V7(W, y, k):
    """조용한 흘러내림: 최근 W분(모두 정규장·매수 뒤) 거래량 합 <= k x TOD 합인데 c[j] <= 매수가 x (1-y)."""
    def make(A):
        F, c = feats(A), A["c"]
        w = max(1, W // A["mins"])
        ctod, ctodn, cv, creg = F["ctod"], F["ctodn"], F["cv"], F["creg"]
        def rule(i, ei, ent):
            j = i - 1
            s = j - w + 1
            if s < ei or c[j] > ent * (1 - y):
                return False
            pv = s - 1
            if creg[j] - creg[pv] != w or ctodn[j] - ctodn[pv] != w:
                return False
            return (cv[j] - cv[pv]) <= k * (ctod[j] - ctod[pv])
        return rule
    return make


def grid():
    G = []
    for k in (2, 3, 4, 5):
        for g in (0.0, 0.01, 0.02):
            G.append((f"V1 투매봉 TOD k{k} 손실{g:.0%}", V1(k, g, "tod")))
        for g in (0.0, 0.02):
            G.append((f"V1 투매봉 R20 k{k} 손실{g:.0%}", V1(k, g, "r20")))
    for x in (0.005, 0.01, 0.015):
        for k in (1.5, 2, 3):
            G.append((f"V2 장대음봉 몸통{x:.1%} TOD k{k}", V2(x, k)))
    for n in (2, 3):
        for f in (0, 1.5):
            for g in (0.0, 0.015):
                G.append((f"V3 증가하락 {n}봉 f{f} 손실{g:.1%}", V3(n, f, g)))
    for W in (15, 30):
        for y in (0.015, 0.03):
            for k in (1.5, 2.5):
                G.append((f"V4 무거운하락 {W}분 손실{y:.1%} k{k}", V4(W, y, k)))
    for p in (0.65, 0.75):
        for y in (0.02, 0.03):
            G.append((f"V5 매도우위 p{p} 손실{y:.0%}", V5(p, y)))
    for k in (2, 3):
        for cont in ("c", "l"):
            G.append((f"V6 투매뒤반등없음 k{k} {'종가' if cont == 'c' else '저가'}아래", V6(k, cont)))
    for y in (0.015, 0.025):
        for k in (0.8, 1.2):
            G.append((f"V7 조용한하락 30분 손실{y:.1%} k{k}", V7(30, y, k)))
    return G


def lookahead_check():
    """원자료를 cut 시각 이전으로 자른 특성과 전체 특성이 cut 이전 봉에서 같은지(TOD·R20)."""
    S = load("late")
    bad, tested = 0, 0
    rng = np.random.default_rng(7)
    for sym in ("NVDA", "AAOI", "TEM"):
        if sym not in S:
            continue
        A = S[sym]
        full = feats(A)
        for n in rng.integers(2000, len(A["c"]) - 10, 3):
            sub = {k: (A[k][:n] if isinstance(A[k], np.ndarray) and len(A[k]) == len(A["c"]) else A[k]) for k in A}
            sub["day"] = A["day"][:n]
            fc = feats(sub, cut=A["tk"][n])
            for key in ("rv_tod", "rv_r20"):
                x, y = full[key][:n], fc[key]
                m = ~(np.isnan(x) & np.isnan(y))
                tested += int(m.sum())
                bad += int((~np.isclose(x[m], y[m], equal_nan=False)).sum())
    # 규칙 판단 검사: 봉 n 이후(n 포함) o·h·l·c·v를 마구 바꿔도 rule(n, ei, ent) 판단이 같아야 한다
    rb, rt, rp = 0, 0, 0
    reps = [V1(2, 0.0, "tod"), V1(2, 0.0, "r20"), V2(0.005, 1.5), V3(2, 0, 0.0), V4(15, 0.0, 1.0), V5(0.5, 0.0)]
    for sym in ("NVDA", "AAOI", "TEM"):
        A = S[sym]
        for n in rng.integers(2000, len(A["c"]) - 10, 40):
            A2 = dict(A)
            A2["_tag"] = f"perturb{n}"
            for k in ("o", "h", "l", "c", "v"):
                x = A[k].copy()
                x[n:] = x[n:] * rng.uniform(0.5, 1.5, len(x) - n)
                A2[k] = x
            for mk in reps:
                r1, r2 = mk(A), mk(A2)
                for ei in range(max(61, n - 40), n):
                    ent = A["c"][ei] * 1.02
                    a1 = r1(n, ei, ent)
                    rt += 1
                    rp += int(a1)
                    rb += int(a1 != r2(n, ei, ent))
            _F.pop((sym, A["mins"], len(A["c"]), A2["_tag"]), None)
    return (f"잘라낸 원자료 특성 vs 전체 특성: {tested}개 값 비교, 불일치 {bad}개 / "
            f"봉 i 이후를 바꾼 규칙 판단: {rt}건 비교(판다 {rp}건), 달라진 판단 {rb}건")


def wrap(mk, kind):
    """2단계 변형. skip1: 매수 봉 자체는 신호로 안 씀(j >= ei+1) / 정규매수: 정규장 매수에만 적용 /
    60분안: 매수 뒤 60분 안(i - ei <= 60/봉 길이)에만 적용."""
    def make(A):
        rule, reg, lim = mk(A), A["reg"], max(1, 60 // A["mins"])
        if kind == "skip1":
            return lambda i, ei, ent: i - 1 >= ei + 1 and rule(i, ei, ent)
        if kind == "정규매수":
            return lambda i, ei, ent: reg[ei] and rule(i, ei, ent)
        return lambda i, ei, ent: i - ei <= lim and rule(i, ei, ent)
    return make


def cells(mk, name):
    df = compare(mk, name, show=False)
    u = df[df.종목 == "사용자78"].set_index(["기간", "범위"])
    a = df[df.종목 == "전체137"].set_index(["기간", "범위"])
    r = dict(규칙=name)
    for p in ("early", "late"):
        for s, sk in (("24시간", "24"), ("정규장만", "reg")):
            for col, key in (("차이", ""), ("t", "_t"), ("손절", "_손절"), ("최악", "_최악"), ("차이_10월6일", "_10월6일")):
                r[f"{p}_{sk}{key}"] = u.loc[(p, s), col]
            r[f"A_{p}_{sk}"] = a.loc[(p, s), "차이"]
            r[f"A_{p}_{sk}_t"] = a.loc[(p, s), "t"]
    return r, df


def _ck_path():
    """중간 저장(프로세스가 끊겨도 이어서): 이 파일·stop_lab.py 내용이 같을 때만 다시 쓴다."""
    import hashlib
    h = hashlib.md5()
    for f in (os.path.abspath(__file__), os.path.join(os.path.dirname(HERE), "stop_lab.py")):
        h.update(open(f, "rb").read())
    return os.path.join(os.path.dirname(RAW), f"volume_ck_{h.hexdigest()[:10]}.csv")


def ck_cells(mk, name, done):
    if name in done:
        return done[name]
    r, _ = cells(mk, name)
    path = _ck_path()
    pd.DataFrame([r]).to_csv(path, mode="a", header=not os.path.exists(path), index=False, encoding="utf-8")
    done[name] = r
    return r


def pick_early(R):
    """early만으로 고른다: 사용자78 정규장만 차이 >= 0 인 규칙 중 24시간 차이 최대(없으면 전체 중 24시간 차이 최대)."""
    cand = R[R.early_reg >= 0]
    return (cand if len(cand) else R).sort_values("early_24", ascending=False)


def main():
    t0 = time.time()
    lines = []
    def out(s=""):
        print(s, flush=True)
        lines.append(s)
    def line(r):
        out(f"{r['규칙']:<36} early24 {r['early_24']:+.2f}(t{r['early_24_t']:+.1f},{r['early_24_손절']:.0f}) "
            f"earlyReg {r['early_reg']:+.2f}(t{r['early_reg_t']:+.1f},{r['early_reg_손절']:.0f}) | late24 {r['late_24']:+.2f} "
            f"lateReg {r['late_reg']:+.2f} | 137 e24 {r['A_early_24']:+.2f} eReg {r['A_early_reg']:+.2f} "
            f"l24 {r['A_late_24']:+.2f} lReg {r['A_late_reg']:+.2f}  [{time.time() - t0:.0f}s]")
    out("거래량(투매) 칼손절 계열 — 기준 대비 수익 합 차이(사용자78 / 137=전체137). 괄호: t, 손절 수")
    out("미래 참조 검사: " + lookahead_check())
    rows, mks = [], {}
    cp = _ck_path()
    done = {r["규칙"]: r for r in pd.read_csv(cp, encoding="utf-8").to_dict("records")} if os.path.exists(cp) else {}
    out("\n[1단계 격자]")
    for name, mk in grid():
        r = ck_cells(mk, name, done)
        rows.append(r); mks[name] = mk; line(r)
    R1 = pd.DataFrame(rows)
    top = pick_early(R1).head(3).규칙.tolist()
    out(f"\n[2단계] early 상위 3개 {top} x 변형(skip1·정규매수·60분안)")
    for name in top:
        for kind in ("skip1", "정규매수", "60분안"):
            nm = f"{name} +{kind}"
            mk = wrap(mks[name], kind)
            r = ck_cells(mk, nm, done)
            rows.append(r); mks[nm] = mk; line(r)
    R = pd.DataFrame(rows)
    R.to_csv(OUT.replace(".txt", "_grid.csv"), index=False, encoding="utf-8-sig")
    four = R[(R.early_24 > 0) & (R.early_reg > 0) & (R.late_24 > 0) & (R.late_reg > 0)]
    out(f"\n시험한 규칙 수 {len(R)}, 사용자78 네 칸 모두 플러스 {len(four)}개 {four.규칙.tolist()}")
    out(f"early 사용자78 24시간 플러스 {int((R.early_24 > 0).sum())}개, early 정규장만 플러스 {int((R.early_reg > 0).sum())}개, "
        f"late 24시간 플러스 {int((R.late_24 > 0).sum())}개, late 정규장만 플러스 {int((R.late_reg > 0).sum())}개")
    pick = pick_early(R)
    out("\nearly 선택 순위(" + ("정규장만 차이 >= 0 인 규칙 중, " if (R.early_reg >= 0).any() else "정규장만 >= 0 없음, 전체 중 ") +
        "사용자78 24시간 차이 순):")
    for _, x in pick.head(8).iterrows():
        out(f"  {x.규칙:<40} early24 {x.early_24:+.2f} earlyReg {x.early_reg:+.2f} | late24 {x.late_24:+.2f} lateReg {x.late_reg:+.2f}")
    best = pick.iloc[0].규칙
    out(f"\n최선안(early로 고름): {best}")
    _, df = cells(mks[best], best)
    f = df.copy()
    for k, fm in (("평균", "{:+.2%}"), ("승률", "{:.0%}"), ("최악", "{:+.1%}"), ("하위5", "{:+.1%}"), ("합", "{:+.2f}"),
                  ("차이", "{:+.2f}"), ("t", "{:+.2f}"), ("손절평균", "{:+.2%}"), ("차이_10월6일", "{:+.2f}")):
        f[k] = f[k].map(lambda v: "" if pd.isna(v) else fm.format(v))
    out(f.drop(columns=["규칙"]).to_string(index=False))
    for p in ("early", "late"):
        for reg in (False, True):
            au = stop_audit(mks[best], p, reg)
            out(f"stop_audit {p} {'정규장만' if reg else '24시간'} 사용자78: " +
                ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in au.items()))
    out(f"\n총 {time.time() - t0:.0f}초")
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
