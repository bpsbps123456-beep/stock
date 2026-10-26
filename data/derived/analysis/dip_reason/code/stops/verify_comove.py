"""반박 검증 — comove 계열 최선안 '묶음중앙 30분 <=-3.00% [정규장만 적용]'을 정의 문장만 보고 다시 만든다(2026-10-07).

정의(연구자 보고): 봉 i 시가에 판다(조건은 i-1 봉까지). (1) i-1 봉이 정규장, (2) 묶음 30분 수익률 <= -3%.
묶음 30분 수익률 = 같은 대분류(fetch.universe) 다른 종목(자기 제외, stop_lab 137종목 중 데이터 있는 것)마다
[시작 <= tk[i-1]인 마지막 봉 종가] / [시작 <= tk[i-1]-30분인 마지막 봉 종가] - 1 의 NaN 뺀 중앙값.
사전에 없거나 같은 그룹 다른 종목 3개 미만이면 QQQ 하나의 같은 30분 수익률.

구현은 연구자 코드와 다르게 '모든 종목 봉 시작 시각의 합집합 격자 + 앞값 채우기(ffill)'로 as-of 값을 만든다.
QQQ 봉은 RAW 원자료에서 직접 만든다(early 5m 그대로, late 1m을 시계 기준 3분으로 합침).
점검: (a) 잘라내기 시험 — 컷오프 뒤에 끝나는 봉을 모두 지우고 다시 계산해도 컷오프 전 결정값이 같은지,
      (b) compare·stop_audit 재현, (c) 이웃(X 2.5/3.5%, 창 15/45분, 최소 동료 2/4, 24시간 적용), (d) 하루 의존.

사용: py stops/verify_comove.py   (code 폴더에서)   출력: ../results/stops/verify_comove.txt"""
import os, sys, time, warnings
import numpy as np, pandas as pd

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)
from stop_lab import load, compare, stop_audit, run, base_trades, UNIV, RAW
from fetch import universe

warnings.filterwarnings("ignore", category=RuntimeWarning)
OUT = os.path.normpath(os.path.join(CODE, "..", "results", "stops", "verify_comove.txt"))
U78 = UNIV["사용자78"]
GRP = universe()
MIN_NS = 60 * 10**9
PER = {5: "early", 3: "late"}


def qqq_bars(mins):
    """(봉 시작 ns, 종가) — early 5m 원자료, late 1m을 3분(왼쪽 닫힘·왼쪽 이름)으로."""
    if mins == 5:
        b = pd.read_parquet(f"{RAW}/5m_QQQ.parquet")
    else:
        m = pd.read_parquet(f"{RAW}/1m_QQQ.parquet")
        b = m.resample("3min", label="left", closed="left").agg({"High": "max", "Close": "last"}).dropna(subset=["Close"])
    b = b[b.High > 0]
    return b.index.as_unit("ns").asi8, b.Close.to_numpy(float)


_S, _Q, _SIG = {}, {}, {}


def data(mins):
    if mins not in _S:
        _S[mins] = load(PER[mins])
        _Q[mins] = qqq_bars(mins)
    return _S[mins], _Q[mins]


def peers_of(s, names, minpeer):
    g = GRP.get(s)
    if g is None:
        return None
    p = [x for x in names if x != s and GRP.get(x) == g]
    return p if len(p) >= minpeer else None


def build(mins, win, minpeer, cutoff=None):
    """{종목: 신호 배열(원소 j = 봉 j 종료 시점의 묶음 30분 수익률)}. cutoff(ns)가 있으면 끝나는 시각 > cutoff 인 봉을 모두 지운다."""
    S, (qt, qc) = data(mins)
    ser = {}
    for s, A in S.items():
        t, c = A["tk"], A["c"]
        if cutoff is not None:
            k = t + mins * MIN_NS <= cutoff
            t, c = t[k], c[k]
        ser[s] = pd.Series(c, index=t)
    if cutoff is not None:
        k = qt + mins * MIN_NS <= cutoff
        qt, qc = qt[k], qc[k]
    ser["__QQQ"] = pd.Series(qc, index=qt)
    for k, v in ser.items():
        assert v.index.is_unique and v.index.is_monotonic_increasing, k
    W = pd.DataFrame(ser).sort_index().ffill()                 # 격자 시각 g: 시작 <= g 인 마지막 봉 종가
    G = W.index.to_numpy(np.int64)
    V = W.to_numpy(float)
    back = np.searchsorted(G, G - win * MIN_NS, side="right") - 1  # 격자에서 g - win 이하 마지막 격자점
    P = np.full_like(V, np.nan)
    ok = back >= 0
    P[ok] = V[back[ok]]
    R = V / P - 1                                              # 격자 x 종목 30분 수익률
    col = {c: n for n, c in enumerate(W.columns)}
    names = [c for c in W.columns if c != "__QQQ"]
    out = {}
    for s, A in S.items():
        pr = peers_of(s, names, minpeer)
        sig = np.nanmedian(R[:, [col[p] for p in pr]], axis=1) if pr else R[:, col["__QQQ"]]
        pos = np.searchsorted(G, A["tk"])
        pos = np.clip(pos, 0, len(G) - 1)
        exact = G[pos] == A["tk"]
        v = np.where(exact, sig[pos], np.nan)                  # 잘린 뒤의 자기 봉은 NaN
        out[s] = v
    return out


def sig_for(mins, win, minpeer):
    k = (mins, win, minpeer)
    if k not in _SIG:
        _SIG[k] = build(mins, win, minpeer)
    return _SIG[k]


def rule_maker(X=0.03, win=30, minpeer=3, regonly=True):
    def make_rule(A):
        sig = sig_for(int(A["mins"]), win, minpeer)[A["sym"]]
        cut = sig <= -X
        if regonly:
            cut = cut & A["reg"].astype(bool)

        def rule(i, ei, ent):
            return bool(cut[i - 1])
        return rule
    return make_rule


def say(f, s=""):
    print(s, flush=True)
    f.write(s + "\n")


def lookahead_test(f):
    """컷오프 뒤에 끝나는 모든 봉(자기·동료·QQQ)을 지우고 다시 계산 → 결정 봉 i 시가 <= 컷오프인 i-1 값이 같아야 한다."""
    rng = np.random.default_rng(7)
    for mins in (5, 3):
        full = sig_for(mins, 30, 3)
        S, _ = data(mins)
        allt = np.unique(np.concatenate([A["tk"] for A in S.values()]))
        tot = bad = 0
        for cut in rng.choice(allt[len(allt) // 10: -len(allt) // 10], 4, replace=False):
            tr = build(mins, 30, 3, cutoff=int(cut))
            for s, A in S.items():
                tk = A["tk"]
                j = np.nonzero(tk[1:] <= cut)[0]                # 봉 i = j+1 의 시가 <= 컷오프
                j = j[tk[j] + mins * MIN_NS <= cut]               # 자기 봉 j도 컷오프 전에 끝남(보통 같은 조건)
                a, b = full[s][j], tr[s][j]
                tot += len(j)
                bad += int((~np.isclose(a, b, equal_nan=True)).sum())
        say(f, f"  잘라내기 시험 {PER[mins]}: 컷오프 4개, 비교한 결정값 {tot:,}개 중 다른 값 {bad}개")


def cellrow(df, p, sc, un):
    return df[(df.기간 == p) & (df.범위 == sc) & (df.종목 == un)].iloc[0]


def summary(df):
    out = {}
    for p in ("early", "late"):
        for sc in ("24시간", "정규장만"):
            for un in ("사용자78", "전체137"):
                r = cellrow(df, p, sc, un)
                out[p, sc, un] = r
    return out


def day_dep(f, mk, label):
    for p in ("early", "late"):
        S = load(p)
        for ro in (False, True):
            t = run(S, mk, p, ro, syms=U78)
            b = base_trades(p, ro)
            b = b[b.sym.isin(U78)]
            days = sorted(set(t.day) | set(b.day))
            d = t.groupby("day").ret.sum().reindex(days, fill_value=0) - b.groupby("day").ret.sum().reindex(days, fill_value=0)
            d = d[d.abs() > 1e-12].sort_values(ascending=False)
            tot = d.sum()
            top = d.head(3)
            say(f, f"  [{label}] {p} {'정규장만' if ro else '24시간'}: 차이 {tot:+.3f}, 다른 날 {len(d)}(+{int((d > 0).sum())}/-{int((d < 0).sum())}), "
                   f"최대 이득일 {top.index[0] if len(top) else '-'} {top.iloc[0] if len(top) else 0:+.3f} → 빼면 {tot - (top.iloc[0] if len(top) else 0):+.3f}, "
                   f"상위 2일 빼면 {tot - top.head(2).sum():+.3f}; 상위 3일 " + ", ".join(f"{k} {v:+.3f}" for k, v in top.items())
                   + "; 최대 손해일 " + (f"{d.index[-1]} {d.iloc[-1]:+.3f}" if len(d) else "-"))


def main():
    t0 = time.time()
    f = open(OUT, "w", encoding="utf-8")
    say(f, "반박 검증 — comove 최선안 '묶음중앙 30분 <=-3.00% [정규장만 적용]' 독립 재구현 (2026-10-07)")
    say(f, "연구자 보고 사용자78: early 24h +0.46 / 정규 +0.26, late 24h +0.02 / 정규 +0.02; 손절 237/222/18/18")
    say(f, "\n[1] 미래 참조 — 잘라내기 시험")
    lookahead_test(f)

    # 재현 대조: 연구자 구현(comove.feats)과 신호값 비교(코드는 복사하지 않고 결과만 대조)
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import comove
        for mins in (5, 3):
            mine = sig_for(mins, 30, 3)
            S, _ = data(mins)
            n = dif = 0
            for s, A in S.items():
                g = comove.feats(A)["gwin"]
                n += len(g)
                dif += int((~np.isclose(g, mine[s], equal_nan=True, atol=1e-12)).sum())
            say(f, f"  신호값 대조 {PER[mins]}: 연구자 gwin과 다른 봉 {dif:,} / {n:,}")
    except Exception as e:  # noqa: BLE001
        say(f, f"  신호값 대조 실패: {e!r}")

    say(f, "\n[2] 재현 — compare")
    main_mk = rule_maker(0.03, 30, 3, True)
    df = compare(main_mk, "검증: 묶음중앙 30분 <=-3% 정규장만", show=False)
    cols = ["기간", "범위", "종목", "건수", "합", "차이", "t", "손절", "최악", "차이_10월6일", "나은날", "나쁜날"]
    say(f, df[cols].to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    say(f, "\n  stop_audit(사용자78)")
    for p in ("early", "late"):
        for ro in (False, True):
            a = stop_audit(main_mk, p, reg_only=ro)
            say(f, f"  {p} {'정규장만' if ro else '24시간'}: " + ", ".join(
                f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in a.items()))

    say(f, "\n[3] 이웃 매개변수(사용자78 차이; 전체137 24시간 차이)")
    nb = [("X 2.5%", dict(X=0.025)), ("X 3.5%", dict(X=0.035)), ("X 2.75%", dict(X=0.0275)), ("X 3.25%", dict(X=0.0325)),
          ("창 15분", dict(win=15)), ("창 45분", dict(win=45)), ("창 20분", dict(win=20)), ("창 40분", dict(win=40)),
          ("최소 동료 2", dict(minpeer=2)), ("최소 동료 4", dict(minpeer=4)), ("24시간 적용", dict(regonly=False))]
    say(f, f"  {'이웃':12s} {'e24':>7s} {'eR':>7s} {'l24':>7s} {'lR':>7s} | {'A_e24':>7s} {'A_l24':>7s} | 손절 e/l")
    rows = [("기준안(3%,30분,3,정규)", df)]
    for nm, kw in nb:
        p = dict(X=0.03, win=30, minpeer=3, regonly=True)
        p.update(kw)
        rows.append((nm, compare(rule_maker(**p), nm, show=False)))
    for nm, d in rows:
        c = summary(d)
        say(f, f"  {nm:12s} {c['early','24시간','사용자78'].차이:+7.2f} {c['early','정규장만','사용자78'].차이:+7.2f} "
               f"{c['late','24시간','사용자78'].차이:+7.2f} {c['late','정규장만','사용자78'].차이:+7.2f} | "
               f"{c['early','24시간','전체137'].차이:+7.2f} {c['late','24시간','전체137'].차이:+7.2f} | "
               f"{c['early','24시간','사용자78'].손절}/{c['late','24시간','사용자78'].손절}")

    say(f, "\n[4] 하루 의존(사용자78, 날짜별 합 차이)")
    day_dep(f, main_mk, "기준안")
    say(f, f"\n걸린 시간 {time.time() - t0:.0f}초")
    f.close()
    print("저장:", OUT)


if __name__ == "__main__":
    main()
