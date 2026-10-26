"""갈래 E_macro(2026-10-10): 매크로·옵션 시장 지표가 켜지면 7종목 노출을 줄이고, 꺼지면 돌아온다.

7종목 규칙(기본 경로)은 그대로 두고 노출 비율만 바꾼다(남는 몫은 현금, 이자 0).
시점: 지표는 t일 종가(FRED는 harness.fred_known의 공개 지연)까지 아는 값 → at_open으로 t 다음 거래일 시가에 적용.
문턱은 모두 고정 숫자 또는 t까지의 이동 창(SMA50, 60일 평균+1σ)이다. 전체 기간 통계는 쓰지 않는다.
격자는 아래 상수로 미리 선언했다(신호 26 × 빠질 때 노출 2 × 재진입 3 = 156).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import harness as H

# ----------------------------------------------------------------------------- 격자(사전 선언)
VIX_LVL = (20, 25, 30)              # VIX 종가 >
VIX_5D = (0.30, 0.50)               # VIX 5거래일 변화율 >
VIX_TERM = (1.00, 1.05)             # VIX/VIX3M > (역전)
VVIX_LVL = (110, 120)               # VVIX >
MOVE_LVL = (110, 130)               # MOVE >
MOVE_20D = 0.20                     # MOVE 20거래일 변화율 >
HY_20D = (0.30, 0.50, 0.75)         # HY OAS(%) 20관측 변화 > (%p)
HY_Z_WIN = 60                       # HY OAS > 60관측 평균 + 1표준편차
HYG_IEF_SMA = 50                    # HYG/IEF(Adj Close) < SMA50
JPY_10D = (-0.02, -0.03, -0.05)     # JPY=X(달러당 엔) 10관측 변화율 < (엔 강세)
TNX_20D = (0.30, 0.50)              # ^TNX(%) 20거래일 변화 > (%p)
CL_20D = 0.20                       # CL=F 20거래일 변화율 >
DXY_20D = 0.03                      # DX-Y.NYB 20거래일 변화율 >
# 종합: 서로 다른 자산군 대표 8개(변동성 2·채권변동성 1·신용 2·엔 1·금리 1·유가 1)
COMPOSITE = ("VIX>25", "VIX/VIX3M>1.00", "MOVE>110", "HY20d>+0.30", "HYG/IEF<SMA50", "JPY10d<-3%",
             "TNX20d>+0.30", "CL20d>+20%")
COMP_K = (2, 3)
LOW = (0.0, 0.5)                    # 신호가 켜진 동안 노출
REENTRY = ("해제즉시", "최소5일", "해제5일확인")
MIN_DAYS = 5
# 해제즉시: 신호가 꺼진 다음 시가에 복귀. 최소5일: 빠진 뒤 5거래일은 무조건 유지, 그 뒤 신호가 꺼지면 복귀.
# 해제5일확인: 신호가 5거래일 연속 꺼져 있어야 복귀.


# ----------------------------------------------------------------------------- 지표(index = 그 값을 아는 날)
def _flag(cond: pd.Series, valid: pd.Series) -> pd.Series:
    """bool 신호를 float(1/0)으로, 지표가 정의되지 않은 날은 NaN(at_open이 건너뛴다)."""
    return cond.astype(float).where(valid)


def raw_indicators() -> dict:
    vix, v3, vv, mv = H.px("^VIX"), H.px("^VIX3M"), H.px("^VVIX"), H.px("^MOVE")
    term = (vix / v3).dropna()
    hy = H.fred_known()["BAMLH0A0HYM2"].dropna()
    ratio = (H.px("HYG", "Adj Close") / H.px("IEF", "Adj Close")).dropna()
    jp, tn, cl, dx = H.px("JPY=X"), H.px("^TNX"), H.px("CL=F"), H.px("DX-Y.NYB")
    return {"vix": vix, "vix5": vix.pct_change(5), "term": term, "vvix": vv, "move": mv, "move20": mv.pct_change(20),
            "hy": hy, "hy20": hy.diff(20), "hy_mu": hy.rolling(HY_Z_WIN).mean(), "hy_sd": hy.rolling(HY_Z_WIN).std(),
            "ratio": ratio, "ratio_sma": ratio.rolling(HYG_IEF_SMA).mean(), "jp10": jp.pct_change(10),
            "tn20": tn.diff(20), "cl20": cl.pct_change(20), "dx20": dx.pct_change(20)}


def signals(R: dict) -> dict:
    s = {}
    for t in VIX_LVL:
        s[f"VIX>{t}"] = _flag(R["vix"] > t, R["vix"].notna())
    for t in VIX_5D:
        s[f"VIX5d>+{t:.0%}"] = _flag(R["vix5"] > t, R["vix5"].notna())
    for t in VIX_TERM:
        s[f"VIX/VIX3M>{t:.2f}"] = _flag(R["term"] > t, R["term"].notna())
    for t in VVIX_LVL:
        s[f"VVIX>{t}"] = _flag(R["vvix"] > t, R["vvix"].notna())
    for t in MOVE_LVL:
        s[f"MOVE>{t}"] = _flag(R["move"] > t, R["move"].notna())
    s[f"MOVE20d>+{MOVE_20D:.0%}"] = _flag(R["move20"] > MOVE_20D, R["move20"].notna())
    for t in HY_20D:
        s[f"HY20d>+{t:.2f}"] = _flag(R["hy20"] > t, R["hy20"].notna())
    s["HY>60평균+1σ"] = _flag(R["hy"] > R["hy_mu"] + R["hy_sd"], R["hy_sd"].notna())
    s["HYG/IEF<SMA50"] = _flag(R["ratio"] < R["ratio_sma"], R["ratio_sma"].notna())
    for t in JPY_10D:
        s[f"JPY10d<{t:.0%}"] = _flag(R["jp10"] < t, R["jp10"].notna())
    for t in TNX_20D:
        s[f"TNX20d>+{t:.2f}"] = _flag(R["tn20"] > t, R["tn20"].notna())
    s[f"CL20d>+{CL_20D:.0%}"] = _flag(R["cl20"] > CL_20D, R["cl20"].notna())
    s[f"DXY20d>+{DXY_20D:.0%}"] = _flag(R["dx20"] > DXY_20D, R["dx20"].notna())
    return s


def on_at_open(sig: dict, b: H.Base) -> dict:
    """각 신호를 기본 경로 거래일 k 시가에 아는 값(k보다 앞선 날의 최신 값)으로. 자료 시작 전은 꺼짐(0)."""
    out = {k: (H.at_open(v, b, fill=0.0) > 0.5).to_numpy() for k, v in sig.items()}
    for kk in COMP_K:
        cnt = sum(out[c].astype(int) for c in COMPOSITE)
        out[f"종합8중≥{kk}"] = cnt >= kk
    return out


def exposure(on: np.ndarray, low: float, mode: str, days) -> pd.Series:
    n = len(on)
    e = np.ones(n)
    reduced, held, off_run = False, 0, 0
    for k in range(n):
        if not reduced:
            if on[k]:
                reduced, held, off_run = True, 0, 0
        else:
            held += 1                                   # k 이전에 줄인 채로 보낸 거래일 수
            off_run = 0 if on[k] else off_run + 1
            if mode == "해제즉시" and not on[k]:
                reduced = False
            elif mode == "최소5일" and held >= MIN_DAYS and not on[k]:
                reduced = False
            elif mode == "해제5일확인" and off_run >= MIN_DAYS:
                reduced = False
        e[k] = low if reduced else 1.0
    return pd.Series(e, index=days)


# ----------------------------------------------------------------------------- 행동 분석
def spells(e: pd.Series, b: H.Base, lo: str, hi: str) -> list:
    """노출<1 구간마다: 시작(줄인 시가), 복귀(돌아온 시가), 그동안 기본 경로 시가→시가 수익, 변형 수익, 차이."""
    r = b.ret
    x = e.reindex(r.index)
    out, start = [], None
    idx = list(r.index)
    for i, d in enumerate(idx):
        red = x.iloc[i] < 0.999
        if red and start is None:
            start = i
        if start is not None and (not red or i == len(idx) - 1):
            end = i if not red else i + 1
            s_d, e_d = idx[start], (idx[end] if end < len(idx) else b.days[-1])
            if hi >= s_d >= lo or lo <= e_d <= hi:
                seg = r.iloc[start:end]
                ex = x.iloc[start:end]
                br_ = float(np.prod(1 + seg.to_numpy()) - 1)
                vr = float(np.prod(1 + (ex * seg).to_numpy()) - 1)
                out.append({"줄임": s_d, "복귀": e_d, "거래일": int(end - start), "노출": float(ex.min()),
                            "기본경로": round(br_, 4), "변형": round(vr, 4), "차이": round((1 + vr) / (1 + br_) - 1, 4)})
            start = None if not red else start
    return out


def on_dates(sig: pd.Series, lo: str, hi: str) -> list:
    x = sig[(sig.index >= lo) & (sig.index <= hi)]
    return [d for d, v in x.items() if v == 1.0]


def main():
    R = raw_indicators()
    S = signals(R)
    bases = {bn: H.base(bn) for bn in ("2026", "hold")}
    ON = {bn: on_at_open(S, b) for bn, b in bases.items()}
    names = list(ON["2026"].keys())
    assert len(names) == 26, len(names)
    rows, expo = [], {}
    for sn in names:
        for low in LOW:
            for mode in REENTRY:
                e = {bn: exposure(ON[bn][sn], low, mode, b.days) for bn, b in bases.items()}
                name = f"{sn} | 노출{low:g} | {mode}"
                rows.append(H.evaluate(name, e, None, {"signal": sn, "low": low, "reentry": mode}))
                expo[name] = e
    print("격자", len(rows))

    # 7월 AI 쇼크 전후·2~3월 신호가 켜진 날(종가 기준 날짜 = 그 값을 아는 날)
    sig_dates = {}
    for sn, sig in S.items():
        sig_dates[sn] = {"6/1~8/14": on_dates(sig, "2026-06-01", "2026-08-14"),
                         "2/1~4/10_일수": len(on_dates(sig, "2026-02-01", "2026-04-10")),
                         "2026_일수": len(on_dates(sig, "2026-01-02", "2026-10-09"))}
    comp = {f"종합8중≥{kk}": [d for d, v in zip(bases["2026"].days, ON["2026"][f"종합8중≥{kk}"])
                            if v and "2026-06-01" <= d <= "2026-08-14"] for kk in COMP_K}
    print("\n[6/1~8/14 신호 켜진 날(종가일)]")
    for sn, v in sig_dates.items():
        print(f"  {sn:<16} {v['6/1~8/14']}  | 2~3월 {v['2/1~4/10_일수']}일 2026 {v['2026_일수']}일")
    for k, v in comp.items():
        print(f"  {k:<16} (시가 적용일) {v}")

    br = H.base_result()
    picks = {"keep90": H.select_front(rows, br, 0.90), "keep80": H.select_front(rows, br, 0.80),
             "keep70": H.select_front(rows, br, 0.70)}
    par = sorted(H.pareto_2026(rows, br, 0.90), key=lambda r: -r["2026"]["전체"]["수익"])
    focus = picks["keep90"] or picks["keep80"] or picks["keep70"] or (par[0] if par else None)
    behav = {}
    for lab, r in list(picks.items()) + [("pareto1", par[0] if par else None)]:
        if r is None:
            continue
        e26 = expo[r["name"]]["2026"]
        behav[lab] = {"name": r["name"], "trace_6_15_9_30": H.trace(e26),
                      "spells_6_15_9_30": spells(e26, bases["2026"], "2026-06-15", "2026-09-30"),
                      "spells_2_1_4_10": spells(e26, bases["2026"], "2026-02-01", "2026-04-10")}
    # 상위 몇 개를 같은 방식으로(7월에 켜진 신호가 있는지 보기 위해 7월 쇼크 수익 순 상위 5)
    top_ai = sorted(rows, key=lambda r: -r["2026"]["AI쇼크"]["수익"])[:5]
    for r in top_ai:
        e26 = expo[r["name"]]["2026"]
        behav.setdefault("AI쇼크상위", []).append({"name": r["name"], "AI쇼크": r["2026"]["AI쇼크"]["수익"],
                                                  "앞": r["2026"]["앞"], "전체": r["2026"]["전체"],
                                                  "spells_6_15_9_30": spells(e26, bases["2026"], "2026-06-15", "2026-09-30")})
    print("\n[행동]")
    print(json.dumps(behav, ensure_ascii=False, indent=1, default=float)[:6000])

    hy_start = R["hy"].index[0]
    extra = {"signal_on_dates": sig_dates, "composite_on_open_6_1_8_14": comp, "behavior": behav,
             "coverage": {"HY_OAS_first_known": hy_start, "HY20d_first": R["hy20"].dropna().index[0],
                          "HY_z60_first": R["hy_sd"].dropna().index[0],
                          "note": "보유 구간 2023-05-22~HY 첫 값 전에는 HY 신호 꺼짐(노출 1)으로 둔다"}}
    grid_desc = ("신호 26개[VIX>{20,25,30}, VIX5d>{+30,+50}%, VIX/VIX3M>{1.00,1.05}, VVIX>{110,120}, MOVE>{110,130}, "
                 "MOVE20d>+20%, HY OAS 20d>{+0.30,+0.50,+0.75}%p, HY>60평균+1σ, HYG/IEF<SMA50, JPY10d<{-2,-3,-5}%, "
                 "TNX20d>{+0.30,+0.50}%p, CL20d>+20%, DXY20d>+3%, 종합8중≥{2,3}] × 빠질 때 노출{0,0.5} × "
                 "재진입{해제즉시, 최소5일 유지, 5일 연속 해제 확인} = 156")
    H.summarize_family("E_macro", rows, grid_desc, extra)
    # 표 출력용: 전 변형 앞·전체·AI쇼크
    print("\n[전 변형 요약: 앞 수익/낙폭, 전체 수익/낙폭, AI쇼크, 2025, 빠짐]")
    for r in sorted(rows, key=lambda r: r["2026"]["앞"]["낙폭"], reverse=True)[:40]:
        a, h = r["2026"], r["hold"]
        print(f"  {r['name']:<40} 앞 {a['앞']['수익']:+.2f}/{a['앞']['낙폭']:.3f} 전체 {a['전체']['수익']:+.2f}/"
              f"{a['전체']['낙폭']:.3f} AI {a['AI쇼크']['수익']:+.3f} 2025 {h['2025']['수익']:+.2f}/{h['2025']['낙폭']:.3f}"
              f" 노출 {a['평균노출']:.2f} 빠짐 {a['빠짐횟수']}")


if __name__ == "__main__":
    main()
