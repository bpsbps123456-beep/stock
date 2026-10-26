"""반박 검증 — 구조 붕괴 최선안 LCR_below|c|0.0%|reg 를 정의 문장만 보고 다시 구현(2026-10-07).
정의: 매수 봉 ei에서 바닥 L 고정. L = 직전에 끝난 정규장(ET 9:30~16:00) 저가.
  애프터 매수(봉 시작 ET 16:00 이후)면 같은 ET 날짜 정규장 최저 저가(그날 정규장이 없으면 그 전 정규장),
  프리·정규 매수면 매수 날짜보다 앞선, 정규장이 있던 마지막 ET 날짜의 정규장 최저 저가.
  L < 매수가일 때만 켜짐. 보유 봉 i(i-1 >= ei)에서 봉 i-1·i 모두 정규장이고 c[i-1] < L*(1-buf) 이면 봉 i 시가에 판다.
이웃: buf -0.5%/+0.5%/+1%, 확인 저가(l), 범위 all, 바닥 PDL(항상 전날).  하루 의존: 날짜별 차이에서 가장 이득 큰 날 제외.
사용: py stops/verify_structure.py  (code 폴더에서) → ../results/stops/verify_structure.txt
"""
import os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from stop_lab import load, run, compare, stop_audit, base_trades, UNIV

OUT = os.path.normpath(os.path.join(HERE, "..", "..", "results", "stops", "verify_structure.txt"))
FH = open(OUT, "w", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    FH.write(s + "\n"); FH.flush()


def floors(A, base):
    """봉마다 '그 봉에서 샀다면' 쓸 바닥 L (numpy 배열)."""
    ses = np.asarray(A["ses"]).astype(str)
    reg = ses == "정규"
    od = np.array([d.toordinal() for d in A["day"]], dtype=np.int64)
    rl = pd.Series(A["l"][reg]).groupby(od[reg]).min()          # ET 날짜별 정규장 저가
    ro, rv = rl.index.values.astype(np.int64), rl.values.astype(float)
    k_prev = np.searchsorted(ro, od, side="left") - 1             # 그 날짜보다 앞선 마지막 정규장 날짜
    k_same = np.searchsorted(ro, od, side="right") - 1            # 그 날짜 이하 마지막 정규장 날짜
    prevL = np.where(k_prev >= 0, rv[np.clip(k_prev, 0, None)], np.nan)
    sameL = np.where(k_same >= 0, rv[np.clip(k_same, 0, None)], np.nan)
    if base == "PDL":
        return prevL
    return np.where(ses == "애프터", sameL, prevL)


def make(base="LCR", conf="c", buf=0.0, scope="reg"):
    def make_rule(A):
        Lb = floors(A, base)
        src = A["c"] if conf == "c" else A["l"]
        reg = np.asarray(A["ses"]).astype(str) == "정규"

        def rule(i, ei, ent):
            L = Lb[ei]
            if not (L < ent):            # NaN 포함: 바닥이 매수가 아래가 아니면 손절 없음
                return False
            if scope == "reg" and not (reg[i - 1] and reg[i]):
                return False
            return src[i - 1] < L * (1 - buf)
        return rule
    return make_rule


def cells(df):
    return {(r.기간, r.범위, r.종목): r for r in df.itertuples()}


def daily(mk, period, reg_only):
    S = load(period)
    us = UNIV["사용자78"]
    t = run(S, mk, period, reg_only, syms=us)
    b = base_trades(period, reg_only)
    b = b[b.sym.isin(us)]
    days = sorted(set(t.day) | set(b.day))
    d = t.groupby("day").ret.sum().reindex(days, fill_value=0) - b.groupby("day").ret.sum().reindex(days, fill_value=0)
    return d


if __name__ == "__main__":
    say("반박 검증: LCR_below|c|0.0%|reg 독립 재구현. 차이 = 기준 대비 수익 합 차이")
    say("연구자 보고 사용자78: early 24h +0.81 / 정규장만 +1.04, late 24h -0.42 / 정규장만 -0.20\n")

    RULES = [("best  LCR c 0.0% reg", make("LCR", "c", 0.0, "reg")),
             ("nb    LCR c -0.5% reg", make("LCR", "c", -0.005, "reg")),
             ("nb    LCR c +0.5% reg", make("LCR", "c", 0.005, "reg")),
             ("nb    LCR c +1.0% reg", make("LCR", "c", 0.01, "reg")),
             ("nb    LCR l 0.0% reg", make("LCR", "l", 0.0, "reg")),
             ("nb    LCR c 0.0% all", make("LCR", "c", 0.0, "all")),
             ("nb    PDL c 0.0% reg", make("PDL", "c", 0.0, "reg"))]
    say(f"시험 규칙 수: {len(RULES)} (최선안 재현 1 + 이웃 6)\n")
    say(f"{'규칙':<24} | 사용자78: E24 ERg L24 LRg (t, 손절) | 전체137: E24 ERg L24 LRg")
    TAB = {}
    for nm, mk in RULES:
        df = compare(mk, nm, show=False)
        TAB[nm] = df
        C = cells(df)
        u = [C[(p, s, "사용자78")] for p in ("early", "late") for s in ("24시간", "정규장만")]
        a = [C[(p, s, "전체137")] for p in ("early", "late") for s in ("24시간", "정규장만")]
        say(f"{nm:<24} | " + " ".join(f"{x.차이:+.2f}(t{x.t:+.2f},{x.손절})" for x in u) + " | " +
            " ".join(f"{x.차이:+.2f}" for x in a))

    say("\n== 최선안 전체 표")
    f = TAB[RULES[0][0]].drop(columns=["규칙"]).copy()
    for k, fmt in (("평균", "{:+.2%}"), ("승률", "{:.0%}"), ("최악", "{:+.1%}"), ("하위5", "{:+.1%}"), ("합", "{:+.2f}"),
                   ("차이", "{:+.2f}"), ("t", "{:+.2f}"), ("손절평균", "{:+.2%}"), ("차이_10월6일", "{:+.2f}")):
        f[k] = f[k].map(lambda v: "" if pd.isna(v) else fmt.format(v))
    say(f.to_string(index=False))

    mk = RULES[0][1]
    say("\n== stop_audit (사용자78)")
    for p in ("early", "late"):
        for ro in (False, True):
            a = stop_audit(mk, p, reg_only=ro)
            say(f"  {p} {'정규장만' if ro else '24시간'}: " +
                ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in a.items()))

    say("\n== 하루 의존 (사용자78, 날짜별 합 차이)")
    for p in ("early", "late"):
        for ro in (False, True):
            d = daily(mk, p, ro)
            top = d.idxmax()
            lo = d.idxmin()
            pos = d[d > 1e-12].sum()
            say(f"  {p} {'정규장만' if ro else '24시간'}: 합 {d.sum():+.3f}, 날수 {len(d)}, 나은날 {(d > 1e-12).sum()} 나쁜날 {(d < -1e-12).sum()},"
                f" 최대이득일 {top} {d[top]:+.3f} (플러스 합 {pos:+.3f}의 {d[top] / pos if pos > 0 else np.nan:.0%}),"
                f" 그날 빼면 {d.sum() - d[top]:+.3f}; 최대손해일 {lo} {d[lo]:+.3f}")
            say("    상위 이득 3일: " + ", ".join(f"{k} {v:+.3f}" for k, v in d.sort_values(ascending=False).head(3).items()))
    FH.close()
