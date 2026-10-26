"""추세 등급 계열 'ma_stack' — 이평선 위치·정배열·기울기 개수(이산 점수) (2026-10-12).

사전 등록 격자 26개 + extra_ 6개(격자 실행 전에 정함). 공통 채점기 harness.py를 그대로 쓴다.

창 처리
    SMA: rolling(n, min_periods=min(n,120)) / EMA: ewm(span=n, adjust=False, min_periods=min(n,120)).
    252일 최저 종가·최고 고가도 rolling(252, min_periods=120). 정의되지 않은 비교(NaN)는 '조건 불충족'(0).
    점수는 그날 종가가 없으면 NaN. 단면 백분위는 H.pct_rank(x, H.universe_mask(P) & x.notna()).
    보유 종목군은 봉 130개 이상이라 MA200(최소 120개)과 MA200.shift(10)은 항상 정의된다.
    미너비니 (3) SMA200.shift(21)은 봉 141개 미만, (8) 252일 수익은 봉 253개 미만에서 정의되지 않아
    불충족으로 센다(신규 상장 감점). extra_minervini_nan252가 이 처리(→ NaN 제외)의 민감도를 잰다.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

import harness as H

FAMILY = "ma_stack"
SETS = {"S1": (5, 10, 20, 50, 100, 200), "S2": (20, 50, 200), "S3": (10, 20, 50, 100, 150, 200)}


def sma(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n, min_periods=min(n, 120)).mean()


def ema(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.ewm(span=n, adjust=False, min_periods=min(n, 120)).mean()


def xpct(P: dict, x: pd.DataFrame) -> pd.DataFrame:
    """t일 단면 백분위(보유 + 봉 130개 이상, 값 있는 종목)."""
    return H.pct_rank(x, H.universe_mask(P) & x.notna())


def b(x: pd.DataFrame) -> pd.DataFrame:
    return x.astype(float)


def stack_count(P: dict, kind: str, set_name: str, level: str, slope_lag: int = 10) -> pd.DataFrame:
    C = P["C"]
    f = sma if kind == "SMA" else ema
    ns = SETS[set_name]
    ma = {n: f(C, n) for n in ns}
    s = sum(b(C > ma[n]) for n in ns)  # a) 위치
    if level in ("b", "c"):
        s = s + sum(b(ma[ns[i]] > ma[ns[i + 1]]) for i in range(len(ns) - 1))  # 정배열(인접 쌍)
    if level == "c":
        s = s + sum(b(ma[n] > ma[n].shift(slope_lag)) for n in ns)  # 기울기
    return s.where(C.notna())


def ret(P: dict, n: int) -> pd.DataFrame:
    C = P["C"]
    return C / C.shift(n) - 1


def tiebreak(P: dict, base: pd.DataFrame, x: pd.DataFrame, w: float) -> pd.DataFrame:
    return (base + w * xpct(P, x).fillna(0.0)).where(base.notna())


def minervini_count(P: dict, nan252: bool = False) -> pd.DataFrame:
    C, Hi = P["C"], P["H"]
    s50, s150, s200 = sma(C, 50), sma(C, 150), sma(C, 200)
    lo252 = C.rolling(252, min_periods=120).min()
    hi252 = Hi.rolling(252, min_periods=120).max()
    r252 = ret(P, 252)
    pr252 = xpct(P, r252)
    conds = [
        (C > s150) & (C > s200),
        s150 > s200,
        s200 > s200.shift(21),
        (s50 > s150) & (s50 > s200),
        C > s50,
        C >= 1.30 * lo252,
        C >= 0.75 * hi252,
        pr252 >= 0.70,
    ]
    s = sum(b(c) for c in conds).where(C.notna())
    if nan252:
        s = s.where(r252.notna())
    return s


def minervini_tb(P: dict, x_days: int = 252, nan252: bool = False) -> pd.DataFrame:
    return tiebreak(P, minervini_count(P, nan252), ret(P, x_days), 0.01)


def dist50(P: dict) -> pd.DataFrame:
    return P["C"] / sma(P["C"], 50) - 1


def build_variants() -> list[tuple[str, str, object]]:
    v: list[tuple[str, str, object]] = []
    lv_desc = {"a": "위치 Σ[C>MA_n]",
               "b": "위치 + 정배열 Σ[MA_짧은>MA_긴](인접 쌍)",
               "c": "위치 + 정배열 + 기울기 Σ[MA_n>MA_n.shift(10)]"}
    # 1) MA 종류 × 집합 × 수준 = 18
    for kind in ("SMA", "EMA"):
        for sn, ns in SETS.items():
            for lv in ("a", "b", "c"):
                name = f"{kind.lower()}_{sn}_{lv}"
                spec = f"{kind} {sn}={list(ns)} {lv_desc[lv]}; 이산 개수, 동점 처리 없음"
                v.append((name, spec, (lambda P, k=kind, s=sn, l=lv: stack_count(P, k, s, l))))
    # 2) 미너비니 추세 템플릿 2
    mspec = ("미너비니 8조건 개수(SMA50/150/200, 252일 창 min_periods=120, (8) 252일 수익 단면 백분위≥0.70; "
             "정의 안 된 조건=불충족)")
    v.append(("minervini_count", mspec, lambda P: minervini_count(P)))
    v.append(("minervini_count_tb252", mspec + " + 0.01×252일 수익 백분위(NaN→0)", lambda P: minervini_tb(P, 252)))
    # 3) 동점 처리 6: SMA·S1 a·b·c × {C/SMA50-1, 63일 수익} 백분위 × 0.001
    for lv in ("a", "b", "c"):
        v.append((f"sma_S1_{lv}_tb_dist50", f"SMA S1 {lv_desc[lv]} + 0.001×(C/SMA50−1 단면 백분위)",
                  (lambda P, l=lv: tiebreak(P, stack_count(P, "SMA", "S1", l), dist50(P), 0.001))))
        v.append((f"sma_S1_{lv}_tb_ret63", f"SMA S1 {lv_desc[lv]} + 0.001×(63일 수익 단면 백분위)",
                  (lambda P, l=lv: tiebreak(P, stack_count(P, "SMA", "S1", l), ret(P, 63), 0.001))))
    # extra_ (격자 실행 전에 정한 6개, 따로 센다)
    v.append(("extra_minervini_nan252", mspec + "; 봉 253개 미만(252일 수익 없음)은 NaN으로 제외(신규 상장 처리 민감도)",
              lambda P: minervini_count(P, nan252=True)))
    v.append(("extra_minervini_tb_ret63", mspec + " + 0.01×63일 수익 백분위(신규 상장도 정의)",
              lambda P: minervini_tb(P, 63)))
    v.append(("extra_sma_S1_c_tb_ret20", "SMA S1 c + 0.001×(20일 수익 단면 백분위)",
              lambda P: tiebreak(P, stack_count(P, "SMA", "S1", "c"), ret(P, 20), 0.001)))
    v.append(("extra_ema_S1_c_tb_ret63", "EMA S1 c + 0.001×(63일 수익 단면 백분위)",
              lambda P: tiebreak(P, stack_count(P, "EMA", "S1", "c"), ret(P, 63), 0.001)))
    v.append(("extra_sma_S1_c_slope5", "SMA S1 c, 기울기를 shift(5)로(짧은 기울기)",
              lambda P: stack_count(P, "SMA", "S1", "c", slope_lag=5)))
    v.append(("extra_sma_S3_c_tb_ret63", "SMA S3 c + 0.001×(63일 수익 단면 백분위)",
              lambda P: tiebreak(P, stack_count(P, "SMA", "S3", "c"), ret(P, 63), 0.001)))
    return v


def diagnostics(score: pd.DataFrame) -> dict:
    """동점 진단: 하루 평균 서로 다른 값 수, 최고점 동점 비율, 등급 비중, S·상위20% 빈 날 비율."""
    P = H.load_panel()
    score = score.reindex(index=P["C"].index, columns=P["C"].columns)
    mask = H.eligible() & score.notna()
    pr = H.pct_rank(score, mask)
    g = H.grade_of(pr, H.PYRAMID)
    out = {}
    for w in ("front", "back", "y2025", "y2024"):
        a, e = H.WINDOWS[w]
        idx = (P["C"].index >= pd.Timestamp(a)) & (P["C"].index <= pd.Timestamp(e))
        m = mask.loc[idx]
        n = m.sum(axis=1)
        keep = n >= 10
        s = score.where(mask).loc[idx][keep]
        n = n[keep]
        gg = g.loc[idx][keep]
        p = pr.loc[idx][keep]
        top_tie = s.eq(s.max(axis=1), axis=0).sum(axis=1) / n
        out[w] = {
            "levels": round(float(s.nunique(axis=1).mean()), 1),
            "top_tie_pct": round(float(top_tie.mean()) * 100, 1),
            "share": {k: round(float(((gg == k).sum(axis=1) / n).mean()) * 100, 1) for k in H.GRADES},
            "S_empty_days_pct": round(float(((gg == "S").sum(axis=1) == 0).mean()) * 100, 1),
            "top20_empty_days_pct": round(float(((p > 0.8).sum(axis=1) == 0).mean()) * 100, 1),
        }
    return out


def g(r, w, h, k):
    return r.get(w, {}).get(h, {}).get(k)


def main():
    P = H.load_panel()
    rows = []
    t0 = time.time()
    for name, spec, fn in build_variants():
        score = fn(P)
        res = H.evaluate(score)
        nl = H.no_lookahead(fn)
        dg = diagnostics(score)
        rows.append({"name": name, "spec": spec, "result": res, "no_lookahead": bool(nl), "diag": dg})
        print(H.summary_line(name + ("" if nl else " [LOOKAHEAD]"), res), flush=True)
        print(f"    diag front lv {dg['front']['levels']} toptie {dg['front']['top_tie_pct']}% "
              f"S {dg['front']['share']['S']}% Sempty {dg['front']['S_empty_days_pct']}% | "
              f"back toptie {dg['back']['top_tie_pct']}% S {dg['back']['share']['S']}% "
              f"Sempty {dg['back']['S_empty_days_pct']}% | y24 IC20 {g(res,'y2024','h20','ic')} "
              f"late IC10 {g(res,'late','h10','ic')}", flush=True)
    path = H.save_results(FAMILY, rows)
    print("saved", path, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
