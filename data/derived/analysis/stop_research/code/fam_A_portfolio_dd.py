"""갈래 A: 포트폴리오(계좌) 단위 손절 + 재진입 (2026-10-10).

그림자 경로 = H.base(bn).close_s: 손절해도 7종목 규칙이 계속 돌았다면의 t일 종가 평가값(t일 종가에 안다).
신호·재진입 판정은 모두 t일 종가 기준 → H.at_open으로 t 다음 거래일 시가에 노출을 바꾼다.

상태 기계(t일 종가마다)
  - 들어가 있을 때(노출 1): 기준 고점 = 마지막 재진입(없으면 경로 시작) 뒤 그림자 최고 종가. 빠지기 신호가 나면 노출 OFF.
    '깊으면 안 판다'(DEEP D): 신호 날 그림자 낙폭(경로 시작 뒤 최고 종가 대비)이 이미 −D 이하면 팔지 않는다.
  - 빠져 있을 때: 저점 = 이탈일 뒤 그림자 최저 종가. 재진입 조건이 되면 노출 1, 기준 고점을 그날 그림자 값으로 다시 잡는다.
    재진입 판정은 이탈 다음 날 종가부터, 재진입한 날에는 빠지기 신호를 보지 않는다.
SMA·n일 수익의 앞부분(2026년 1~2월)은 보유 기간 경로(2025-12-31까지)를 2026-01-02 종가(현금일 1.0)에 비율로 이어 붙여 계산한다.
낙폭 기준 고점은 이어 붙이지 않는다(2026 경로 시작부터).
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

import harness as H

FAMILY = "A_portfolio_dd"

# ----------------------------------------------------------------------------- 격자(사전 고정)
DD_X = [0.06, 0.08, 0.10, 0.12, 0.15, 0.20]                    # (a) 기준 고점 대비 낙폭 ≤ −X
VEL = [(5, -0.08), (5, -0.10), (5, -0.12), (10, -0.12), (10, -0.15)]  # (b) n일 수익 ≤ 문턱
OFF = [0.0, 0.5]                                                # 빠질 때 노출
REENTRY = [("r1", 5), ("r1", 10), ("r1", 20),                  # 고정 n거래일 뒤
           ("r2", 0.05), ("r2", 0.10),                          # 이탈 뒤 저점 대비 +p
           ("r3", 10), ("r3", 20),                              # 그림자 > SMA n
           ("r4", -0.05),                                       # 이탈 때 기준 고점 대비 −5% 이내 회복
           ("r5", None)]                                        # 단계: 저점 +5%에 절반, +10%에 전부
SMA_N = [10, 20, 50]                                            # (c) 그림자 < SMA N 이면 OFF, > SMA N 이면 복귀
DEEP_D = [0.20, 0.25]
DEEP_SIG = [("dd", 0.10), ("dd", 0.15), ("vel", (5, -0.10)), ("vel", (10, -0.15))]
DEEP_RE = [("r2", 0.05), ("r3", 20)]
DEEP_SMA = [20, 50]


def grid():
    out = []
    for x in DD_X:
        for off in OFF:
            for re in REENTRY:
                out.append({"sig": ("dd", x), "off": off, "re": re, "deep": None})
    for v in VEL:
        for off in OFF:
            for re in REENTRY:
                out.append({"sig": ("vel", v), "off": off, "re": re, "deep": None})
    for n in SMA_N:
        for off in OFF:
            out.append({"sig": ("sma", n), "off": off, "re": ("sma", n), "deep": None})
    for s in DEEP_SIG:
        for re in DEEP_RE:
            for d in DEEP_D:
                out.append({"sig": s, "off": 0.0, "re": re, "deep": d})
    for n in DEEP_SMA:
        for d in DEEP_D:
            out.append({"sig": ("sma", n), "off": 0.0, "re": ("sma", n), "deep": d})
    return out


def label(p):
    s, re = p["sig"], p["re"]
    if s[0] == "dd":
        a = f"DD-{s[1]:.0%}"
    elif s[0] == "vel":
        a = f"{s[1][0]}일{s[1][1]:+.0%}"
    else:
        a = f"SMA{s[1]}이탈"
    if re[0] == "r1":
        b = f"{re[1]}일뒤"
    elif re[0] == "r2":
        b = f"저점+{re[1]:.0%}"
    elif re[0] in ("r3", "sma"):
        b = f">SMA{re[1]}"
    elif re[0] == "r4":
        b = "DD-5%회복"
    else:
        b = "단계+5/+10"
    out = f"{a}→{b} 노출{p['off']:.1f}"
    if p["deep"]:
        out += f" 깊음{p['deep']:.0%}안팜"
    return out


# ----------------------------------------------------------------------------- 그림자 경로
def shadow_full(bn: str) -> pd.Series:
    """신호 계산용 그림자 종가. 2026은 보유 기간 경로를 앞에 이어 붙인다(SMA·n일 수익 준비 기간)."""
    s = H.base(bn).close_s.copy()
    if bn == "2026":
        h = H.base("hold").close_s
        pre = h[h.index < s.index[0]] * (s.iloc[0] / h.iloc[-1])
        s = pd.concat([pre, s])
    return s


_FEAT: dict = {}


def feats(bn: str) -> dict:
    if bn not in _FEAT:
        full = shadow_full(bn)
        days = H.base(bn).days
        f = {"sh": full.reindex(days).to_numpy(float)}
        for n in sorted(set(SMA_N) | {10, 20}):
            f[f"sma{n}"] = full.rolling(n, min_periods=n).mean().reindex(days).to_numpy(float)
        for n in (5, 10):
            f[f"ret{n}"] = (full / full.shift(n) - 1).reindex(days).to_numpy(float)
        sh = f["sh"]
        f["ath_dd"] = sh / np.maximum.accumulate(sh) - 1          # 경로 시작 뒤 최고 대비(깊음 판단용)
        _FEAT[bn] = f
    return _FEAT[bn]


def simulate(bn: str, p: dict) -> pd.Series:
    """t일 종가 뒤 노출 상태(index = 기본 경로 거래일). at_open 전 값."""
    f = feats(bn)
    sh = f["sh"]
    days = H.base(bn).days
    kind, prm = p["sig"]
    rk, rp = p["re"]
    off, deep = p["off"], p["deep"]
    level = 1.0
    peak = sh[0]
    exit_i, low, peak_exit = -1, np.nan, np.nan
    out = np.empty(len(days))
    for i in range(len(days)):
        x = sh[i]
        if level >= 0.999:
            peak = max(peak, x)
            if kind == "dd":
                sig = x / peak - 1 <= -prm
            elif kind == "vel":
                r = f[f"ret{prm[0]}"][i]
                sig = np.isfinite(r) and r <= prm[1]
            else:
                m = f[f"sma{prm}"][i]
                sig = np.isfinite(m) and x < m
            if sig and deep is not None and f["ath_dd"][i] <= -deep:
                sig = False
            if sig:
                level, exit_i, low, peak_exit = off, i, x, peak
        else:
            low = min(low, x)
            back = False
            if rk == "r1":
                back = i - exit_i >= rp
            elif rk == "r2":
                back = x >= low * (1 + rp)
            elif rk in ("r3", "sma"):
                m = f[f"sma{rp}"][i]
                back = np.isfinite(m) and x > m
            elif rk == "r4":
                back = x / peak_exit - 1 >= rp
            elif rk == "r5":
                if x >= low * 1.10:
                    back = True
                elif x >= low * 1.05:
                    level = max(level, off + (1 - off) * 0.5)
            if back:
                level, peak = 1.0, x
        out[i] = level
    return pd.Series(out, index=days)


def exposure(bn: str, p: dict) -> pd.Series:
    b = H.base(bn)
    return H.at_open(simulate(bn, p), b, fill=1.0)


# ----------------------------------------------------------------------------- 행동 분석(빠진 구간별 아낀 손실·놓친 반등)
def spells(bn: str, e: pd.Series, lo: str, hi: str) -> list[dict]:
    """노출 < 1 인 연속 구간(시가 기준). saved = (1+e·r)누적 / (1+r)누적 − 1 (양수 = 아낀 손실, 음수 = 놓친 반등)."""
    b = H.base(bn)
    r = b.ret
    ee = e.reindex(r.index)
    out, cur = [], None
    for d in r.index:
        if ee[d] < 0.999:
            if cur is None:
                cur = {"out": d, "days": []}
            cur["days"].append(d)
        elif cur is not None:
            cur["back"] = d
            out.append(cur)
            cur = None
    if cur is not None:
        cur["back"] = None
        out.append(cur)
    res = []
    for s in out:
        if s["out"] > hi or s["days"][-1] < lo:
            continue
        rr = r[s["days"]]
        e_s = ee[s["days"]]
        base_chg = float((1 + rr).prod() - 1)
        mine = float((1 + e_s * rr).prod() - 1)
        res.append({"out_open": s["out"], "back_open": s["back"], "n": len(s["days"]), "min_e": float(e_s.min()),
                    "base_chg": round(base_chg, 4), "mine_chg": round(mine, 4),
                    "saved": round((1 + mine) / (1 + base_chg) - 1, 4)})
    return res


def spell_sum(sp: list[dict]) -> dict:
    return {"n": len(sp), "saved_sum": round(float(np.prod([1 + s["saved"] for s in sp]) - 1), 4) if sp else 0.0,
            "losing_spells": sum(s["saved"] < 0 for s in sp),
            "missed": round(float(sum(s["saved"] for s in sp if s["saved"] < 0)), 4),
            "avoided": round(float(sum(s["saved"] for s in sp if s["saved"] > 0)), 4)}


# ----------------------------------------------------------------------------- 실행
def main():
    G = grid()
    print("격자 개수", len(G))
    rows, exps = [], {}
    for p in G:
        name = label(p)
        e26, eh = exposure("2026", p), exposure("hold", p)
        exps[name] = (e26, eh)
        params = {"sig": p["sig"], "off": p["off"], "re": p["re"], "deep": p["deep"]}
        rows.append(H.evaluate(name, {"2026": e26, "hold": eh}, None, params))

    grid_desc = (f"(a) 기준고점 대비 낙폭 X∈{{6,8,10,12,15,20}}% ×(b) 5일≤{{-8,-10,-12}}%, 10일≤{{-12,-15}}% = 11 신호 × 노출{{0,0.5}} × "
                 f"재진입 9(r1 5·10·20일, r2 저점+5·+10%, r3 >SMA10·20, r4 고점대비 −5% 회복, r5 +5%절반·+10%전부) = 198; "
                 f"(c) SMA{{10,20,50}} 이탈·복귀 × 노출{{0,0.5}} = 6; 깊으면 안 판다 D∈{{20,25}}% × [DD10·DD15·5일-10·10일-15 × r2+5%·r3 SMA20 (노출0)] = 16 "
                 f"+ [SMA20·50 (노출0)] = 4. 합 {len(G)}")

    # 앞 고르기 결과를 미리 구해 행동 분석에 쓴다(summarize_family와 같은 함수)
    br = H.base_result()
    picks = {k: H.select_front(rows, br, kp) for k, kp in (("keep90", 0.9), ("keep80", 0.8), ("keep70", 0.7))}
    par = sorted(H.pareto_2026(rows, br, 0.9), key=lambda r: -r["2026"]["전체"]["수익"])

    def behav(r):
        e26, eh = exps[r["name"]]
        return {"trace_jun_sep": H.trace(e26),
                "spells_front": spells("2026", e26, "2026-01-02", "2026-05-29"),
                "spells_back": spells("2026", e26, "2026-06-01", "2026-10-09"),
                "front_sum": spell_sum(spells("2026", e26, "2026-01-02", "2026-05-29")),
                "back_sum": spell_sum(spells("2026", e26, "2026-06-01", "2026-10-09")),
                "exits_front": int(((e26.shift(1).fillna(1.0) >= 0.999) & (e26 < 0.999))[e26.index <= "2026-05-29"].sum()),
                "exits_2026": int(((e26.shift(1).fillna(1.0) >= 0.999) & (e26 < 0.999)).sum()),
                "hold_2025_spells": spell_sum(spells("hold", eh, "2025-01-01", "2025-12-31")),
                "hold_2024_spells": spell_sum(spells("hold", eh, "2024-01-01", "2024-12-31"))}

    extra = {"picks_behavior": {}, "pareto_behavior": {}}
    for k, v in picks.items():
        if v is not None:
            extra["picks_behavior"][k] = {"name": v["name"], **behav(v)}
    for r in par[:5]:
        extra["pareto_behavior"][r["name"]] = behav(r)

    # 변형별 앞 구간 빠짐·휩소 요약(전 변형)
    ws = []
    for r in rows:
        e26 = exps[r["name"]][0]
        sf = spell_sum(spells("2026", e26, "2026-01-02", "2026-05-29"))
        ws.append({"name": r["name"], "front_exits": sf["n"], "front_saved": sf["saved_sum"],
                   "front_losing": sf["losing_spells"]})
    extra["front_whipsaw_all"] = ws

    payload = H.summarize_family(FAMILY, rows, grid_desc, extra)

    # 짧은 표: 앞 구간 수익 유지 90% 통과 수, 신호 종류별 최선
    bw = 1 + br["2026"]["앞"]["수익"]
    n90 = sum(1 + r["2026"]["앞"]["수익"] >= 0.9 * bw for r in rows)
    n80 = sum(1 + r["2026"]["앞"]["수익"] >= 0.8 * bw for r in rows)
    print(f"앞 구간 자산 ≥ 기본 90%: {n90}/{len(rows)}, ≥80%: {n80}/{len(rows)}")
    for k, v in extra["picks_behavior"].items():
        print(k, v["name"], "앞 빠짐", v["exits_front"], "2026 빠짐", v["exits_2026"])
        print("  trace", v["trace_jun_sep"])
        print("  front", v["front_sum"], "back", v["back_sum"])
        for s in v["spells_front"]:
            print("    F", s)
        for s in v["spells_back"]:
            print("    B", s)
    return payload


if __name__ == "__main__":
    main()
