"""갈래 C: 변동성 목표·위험 예산(2026-10-10).

노출 e = min(1, 목표 변동성 / 예측 변동성)을 7종목 기본 경로 위에 얹는다(나머지는 현금, 레버리지 없음).
Barroso & Santa-Clara(2015)·Moreira & Muir(2017)의 '위험 조절 모멘텀'을 이 7종목에서 시험한다.

시점
  - 예측 변동성은 그림자 경로(base.close_s, t일 종가 평가) 일수익 등 t일 종가까지의 값으로만 잰다.
  - at_open → t 다음 거래일 시가에 노출을 맞춘다.
  - 2026 기본 경로는 1/2에 시작해 창이 비므로, 같은 규칙을 2023-05~2025-12 주간 PDF로 돌린 보유 경로(base_hold)의
    그림자 수익을 앞에 이어 붙여 예열한다(당시 공개된 PDF로 만든 경로라 시점 위반 없음, 사전 결정).
  - 보유 경로 자체는 앞에 이을 자료가 없어 창이 찰 때까지 노출 1.
  - 상대형 중앙값은 지난 252거래일(최소 126) 이동 창, 전체 기간 통계는 쓰지 않는다.

격자(아래 상수, 사전 선언 — 결과를 보고 고치지 않는다)
  예측 변동성 11종 × (목표 6개 + 상대형 1) × 운용 방식 4 = 308개.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

import harness as H

# ----------------------------------------------------------------------------- 격자(사전 선언)
TARGETS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]   # 연율 목표 변동성
RV_WINDOWS = [5, 10, 20, 40, 60]                 # 실현 변동성 창(거래일)
EWMA_HL = [5, 10, 20]                            # EWMA 반감기(거래일), 평균 0 가정(RiskMetrics)
SEMI_WIN = 20                                    # 하방 반편차 창, ×√2로 총변동성 단위에 맞춤
SOXX_WIN = 20                                    # SOXX 실현 변동성 창
BETA_WIN = 60                                    # 그림자의 SOXX·SPY 베타 이동 창
REL_LOOKBACK, REL_MINP = 252, 126                # 상대형: 지난 252일 예측 변동성 중앙값 / 현재
MODES = ["매일", "주1회", "밴드10", "비대칭10"]
BAND = 0.10                                      # 밴드: 차이 > 10%p(또는 원하는 값이 상한 1.0이면) 바꾼다
STEP_UP = 0.10                                   # 비대칭: 줄일 땐 바로, 늘릴 땐 하루 +10%p까지
ANN = np.sqrt(252)

# 2~3월 하락 구간(지시문 날짜), 반등은 각 구간 끝 다음 10거래일(사전 결정)
FEBMAR = [("2026-02-03", "2026-02-05"), ("2026-02-25", "2026-03-06"), ("2026-03-19", "2026-03-30")]
REBOUND_N = 10
CONST_EXPO = [0.5, 0.6, 0.7, 0.8, 0.9]          # 비교 기준(격자 밖): 늘 같은 비율만 들고 있기


# ----------------------------------------------------------------------------- 그림자 수익
def shadow_ret(bn: str) -> pd.Series:
    """t일 종가 평가값의 일수익(t일 종가에 안다). 2026은 보유 경로를 앞에 이어 예열."""
    bh = H.base("hold")
    rh = bh.close_s.pct_change().iloc[2:]          # 0일=현금, 1일=시가→종가 부분 수익이라 제외
    if bn == "hold":
        return rh
    b = H.base("2026")
    r26 = b.close_s.pct_change().iloc[2:]
    return pd.concat([rh, r26]).sort_index()


def idx_ret(sym: str) -> pd.Series:
    return H.px(sym, "Adj Close").pct_change()


def beta(rs: pd.Series, ri: pd.Series, n: int) -> pd.Series:
    d = pd.concat([rs.rename("s"), ri.rename("i")], axis=1).dropna()
    cov = d["s"].rolling(n, min_periods=n).cov(d["i"])
    var = d["i"].rolling(n, min_periods=n).var()
    return (cov / var).clip(lower=0.0).reindex(rs.index)


def estimators(bn: str) -> dict:
    """이름 → 연율 예측 변동성(index = 알게 된 날짜)."""
    r = shadow_ret(bn)
    out = {}
    for n in RV_WINDOWS:
        out[f"실현{n}"] = r.rolling(n, min_periods=n).std() * ANN
    for hl in EWMA_HL:
        out[f"EWMA{hl}"] = np.sqrt(r.pow(2).ewm(halflife=hl, min_periods=max(5, hl)).mean()) * ANN
    neg = r.clip(upper=0.0)
    out[f"하방{SEMI_WIN}"] = np.sqrt(neg.pow(2).rolling(SEMI_WIN, min_periods=SEMI_WIN).mean()) * ANN * np.sqrt(2)
    soxx = idx_ret("SOXX")
    soxx_vol = (soxx.rolling(SOXX_WIN, min_periods=SOXX_WIN).std() * ANN).reindex(r.index)
    out["SOXX×베타"] = soxx_vol * beta(r, soxx, BETA_WIN)
    vix = (H.px("^VIX", "Close") / 100.0).reindex(r.index)
    out["VIX×베타"] = vix * beta(r, idx_ret("SPY"), BETA_WIN)
    return {k: v.replace([np.inf, -np.inf], np.nan) for k, v in out.items()}


# ----------------------------------------------------------------------------- 노출
def apply_mode(desired: pd.Series, b: H.Base, mode: str) -> pd.Series:
    d = desired.reindex(b.days).to_numpy(float)
    reb = set(b.rebalance_days())
    out = np.empty(len(d))
    cur = 1.0
    for i, day in enumerate(b.days):
        x = d[i]
        if mode == "매일":
            cur = x
        elif mode == "주1회":
            if day in reb:
                cur = x
        elif mode == "밴드10":
            if abs(x - cur) > BAND + 1e-12 or (x >= 0.999 and cur < 0.999):
                cur = x
        elif mode == "비대칭10":
            cur = x if x < cur else min(x, cur + STEP_UP)
        out[i] = cur
    return pd.Series(out, index=b.days)


def exposure(est: pd.Series, b: H.Base, tgt, mode: str) -> pd.Series:
    if tgt == "상대":
        med = est.rolling(REL_LOOKBACK, min_periods=REL_MINP).median()
        sig = (med / est).clip(upper=1.0)
    else:
        sig = (tgt / est).clip(upper=1.0)
    desired = H.at_open(sig, b, fill=1.0)
    return apply_mode(desired, b, mode)


# ----------------------------------------------------------------------------- 보고용
def window_ret(r: pd.Series, lo: str, hi: str) -> float:
    return H.stats(r, lo, hi)["수익"]


def behavior(e26: pd.Series) -> dict:
    """2~3월 하락·7월 쇼크에서 노출과 손익(기본 대비)."""
    b = H.base("2026")
    rb = b.ret
    rv = H.overlay(b, e26)
    e = e26.reindex(rb.index)
    days = list(rb.index)
    out = {"2~3월": [], "AI쇼크": {}}
    for lo, hi in FEBMAR:
        j = max(i for i, d in enumerate(days) if d <= hi)
        r_lo, r_hi = days[min(j + 1, len(days) - 1)], days[min(j + REBOUND_N, len(days) - 1)]
        ew = e[(e.index >= lo) & (e.index <= hi)]
        cuts = int(((e.diff() < -1e-9) & (e.index >= "2026-01-26") & (e.index <= hi)).sum())
        out["2~3월"].append({
            "하락": f"{lo}~{hi}", "기본_하락": round(window_ret(rb, lo, hi), 4), "변형_하락": round(window_ret(rv, lo, hi), 4),
            "반등": f"{r_lo}~{r_hi}", "기본_반등": round(window_ret(rb, r_lo, r_hi), 4),
            "변형_반등": round(window_ret(rv, r_lo, r_hi), 4),
            "하락중_평균노출": round(float(ew.mean()), 3), "하락중_최저노출": round(float(ew.min()), 3),
            "반등중_평균노출": round(float(e[(e.index >= r_lo) & (e.index <= r_hi)].mean()), 3)})
    fm = (e.index >= "2026-02-01") & (e.index <= "2026-04-30")
    out["2~4월_줄인날수"] = int(((e.diff() < -1e-9) & fm).sum())
    out["2~4월_평균노출"] = round(float(e[fm].mean()), 3)
    out["2~4월_기본"] = round(window_ret(rb, "2026-02-01", "2026-04-30"), 4)
    out["2~4월_변형"] = round(window_ret(rv, "2026-02-01", "2026-04-30"), 4)
    for nm, (lo, hi) in {"쇼크(6/22~7/29)": ("2026-06-22", "2026-07-29"), "반등(7/30~9/30)": ("2026-07-30", "2026-09-30"),
                         "9/1~10/9": ("2026-09-01", "2026-10-09")}.items():
        m = (e.index >= lo) & (e.index <= hi)
        out["AI쇼크"][nm] = {"기본": round(window_ret(rb, lo, hi), 4), "변형": round(window_ret(rv, lo, hi), 4),
                            "평균노출": round(float(e[m].mean()), 3), "최저노출": round(float(e[m].min()), 3)}
    # 일별 기여 분해(6/15~9/30): 노출을 줄인 몫 × 기본 수익 — 하락일엔 아낀 손실, 상승일엔 놓친 반등(단순 합, 복리 아님)
    m = (rb.index >= "2026-06-15") & (rb.index <= "2026-09-30")
    gap = (1 - e[m]) * rb[m]
    out["6/15~9/30_아낀손실(단순합)"] = round(float(-gap[gap < 0].sum()), 4)
    out["6/15~9/30_놓친상승(단순합)"] = round(float(gap[gap > 0].sum()), 4)
    m = (rb.index >= "2026-01-26") & (rb.index <= "2026-04-30")
    gap = (1 - e[m]) * rb[m]
    out["1/26~4/30_아낀손실(단순합)"] = round(float(-gap[gap < 0].sum()), 4)
    out["1/26~4/30_놓친상승(단순합)"] = round(float(gap[gap > 0].sum()), 4)
    return out


def main():
    b26, bh = H.base("2026"), H.base("hold")
    est26, esth = estimators("2026"), estimators("hold")
    rows, expo = [], {}
    for en in est26:
        for tgt in TARGETS + ["상대"]:
            for mode in MODES:
                name = f"{en}|{tgt}|{mode}"
                e26 = exposure(est26[en], b26, tgt, mode)
                eh = exposure(esth[en], bh, tgt, mode)
                params = {"예측": en, "목표": tgt, "방식": mode}
                rows.append(H.evaluate(name, {"2026": e26, "hold": eh}, None, params))
                expo[name] = (e26, eh)
    grid_desc = (f"예측 변동성 11종(실현 {RV_WINDOWS}일, EWMA 반감기 {EWMA_HL}일, 하방 반편차 {SEMI_WIN}일×√2, "
                 f"SOXX {SOXX_WIN}일 변동성×그림자 SOXX 베타 {BETA_WIN}일, ^VIX/100×그림자 SPY 베타 {BETA_WIN}일) × "
                 f"목표 {TARGETS}(연율, 상한 1.0) + 상대형(252일 중앙값/현재, 최소 126일) 7 × 방식 4({MODES}: "
                 f"밴드는 차이>{BAND:.0%}p 또는 상한 복귀, 비대칭은 하루 +{STEP_UP:.0%}p) = {len(rows)}개")

    # 전 변형 노출 비교: 2~3월 vs 7월 쇼크
    cmp = []
    for name, (e26, _) in expo.items():
        e = e26.reindex(b26.ret.index)
        fm = float(e[(e.index >= "2026-02-01") & (e.index <= "2026-03-31")].mean())
        js = float(e[(e.index >= "2026-06-22") & (e.index <= "2026-07-29")].mean())
        cmp.append({"name": name, "2~3월노출": round(fm, 3), "7월쇼크노출": round(js, 3)})
    cdf = pd.DataFrame(cmp)
    exp_cmp = {"변형수": len(cdf), "2~3월노출<7월쇼크노출_변형수": int((cdf["2~3월노출"] < cdf["7월쇼크노출"] - 1e-9).sum()),
               "2~3월노출_중앙값": round(float(cdf["2~3월노출"].median()), 3),
               "7월쇼크노출_중앙값": round(float(cdf["7월쇼크노출"].median()), 3)}

    # 예측 변동성 월평균(설명용)
    vol_month = {}
    for en in ("실현20", "EWMA10", "SOXX×베타", "VIX×베타"):
        s = est26[en]
        s = s[(s.index >= "2026-01-01")]
        vol_month[en] = {k: round(float(v), 2) for k, v in s.groupby(s.index.str[:7]).mean().items()}

    br = H.base_result()
    picks = {"keep90": H.select_front(rows, br, 0.90), "keep80": H.select_front(rows, br, 0.80),
             "keep70": H.select_front(rows, br, 0.70)}
    par = sorted(H.pareto_2026(rows, br, 0.90), key=lambda r: -r["2026"]["전체"]["수익"])
    chosen = picks["keep90"] or picks["keep80"] or (par[0] if par else None)
    chosen_label = "keep90" if picks["keep90"] else ("keep80" if picks["keep80"] else ("사후 파레토 최선" if par else None))
    if chosen is None:   # 아무것도 없으면 2026 전체 수익 최대 변형(설명용)
        chosen = max(rows, key=lambda r: r["2026"]["전체"]["수익"])
        chosen_label = "2026 전체 수익 최대(설명용)"
    detail = {}
    todo = [("chosen", chosen)] + [(k, v) for k, v in picks.items() if v] + ([("pareto1", par[0])] if par else [])
    for lab, r in todo:
        e26 = expo[r["name"]][0]
        detail[lab] = {"name": r["name"], "trace_6_15_9_30": H.trace(e26),
                       "trace_1_26_4_30": H.trace(e26, "2026-01-26", "2026-04-30"), "behavior": behavior(e26)}
    # 대표 변형(설명용, 사전 지정): 실현20 목표 0.6 매일, 상대형 실현20 매일
    for nm in ("실현20|0.6|매일", "실현20|상대|매일", "EWMA10|0.8|비대칭10", "VIX×베타|0.6|매일"):
        detail[f"대표:{nm}"] = {"name": nm, "trace_6_15_9_30": H.trace(expo[nm][0]),
                               "behavior": behavior(expo[nm][0])}

    # 비교 기준(격자 아님, 고르기에 안 씀): 고정 노출 c, 그리고 변형마다 같은 평균 노출의 고정 노출과 비교
    # → 변동성 목표가 '타이밍'으로 번 것인지, 그냥 덜 들고 있어서 덜 잃은 것인지 가른다(Moreira & Muir의 비교 방식).
    const = {}
    for c in CONST_EXPO:
        r26, rh = c * b26.ret, c * bh.ret
        const[f"고정{c}"] = {"2026": {k: H.stats(r26, *v) for k, v in b26.periods.items()},
                           "hold": {k: H.stats(rh, *v) for k, v in bh.periods.items()}}
    match = []
    for row in rows:
        e26, eh = expo[row["name"]]
        c26 = row["2026"]["평균노출"]
        eh2 = eh.reindex(bh.ret.index)
        c25 = float(eh2[(eh2.index >= "2025-01-01") & (eh2.index <= "2025-12-31")].mean())
        k26 = {k: H.stats(c26 * b26.ret, *b26.periods[k]) for k in ("전체", "앞", "뒤", "AI쇼크")}
        k25 = H.stats(c25 * bh.ret, *bh.periods["2025"])
        match.append({"name": row["name"], "c26": round(c26, 3), "c25": round(c25, 3),
                      "전체수익차": row["2026"]["전체"]["수익"] - k26["전체"]["수익"],
                      "전체낙폭차": row["2026"]["전체"]["낙폭"] - k26["전체"]["낙폭"],
                      "앞수익차": row["2026"]["앞"]["수익"] - k26["앞"]["수익"],
                      "앞낙폭차": row["2026"]["앞"]["낙폭"] - k26["앞"]["낙폭"],
                      "AI쇼크차": row["2026"]["AI쇼크"]["수익"] - k26["AI쇼크"]["수익"],
                      "2025수익차": row["hold"]["2025"]["수익"] - k25["수익"],
                      "2025낙폭차": row["hold"]["2025"]["낙폭"] - k25["낙폭"]})
    mdf = pd.DataFrame(match)
    mdf = mdf[mdf["c26"] < 0.98]          # 사실상 노출 1인 변형은 비교 의미 없음
    match_sum = {"비교변형수(평균노출<0.98)": len(mdf)}
    for col in ("전체수익차", "전체낙폭차", "앞수익차", "앞낙폭차", "AI쇼크차", "2025수익차", "2025낙폭차"):
        match_sum[col] = {"고정보다_나은_변형수": int((mdf[col] > 0).sum()), "중앙값": round(float(mdf[col].median()), 4)}
    match_pick = {lab: next(({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}
                             for m in match if m["name"] == r["name"]), None)
                  for lab, r in todo}

    extra = {"chosen_label": chosen_label, "노출비교_2~3월_vs_7월": exp_cmp, "예측변동성_월평균_2026": vol_month,
             "고정노출_기준": {k: {"2026전체": v["2026"]["전체"], "2026앞": v["2026"]["앞"], "AI쇼크": v["2026"]["AI쇼크"],
                                "2025": v["hold"]["2025"], "관세": v["hold"]["관세"]} for k, v in const.items()},
             "같은평균노출_고정과비교_요약": match_sum, "같은평균노출_고정과비교_고른변형": match_pick,
             "detail": detail, "노출비교_전변형": cmp}
    payload = H.summarize_family("C_vol_target", rows, grid_desc, extra)
    print("고른 변형:", chosen_label, chosen["name"])
    print(json.dumps({k: v for k, v in extra.items() if k not in ("노출비교_전변형",)}, ensure_ascii=False, indent=1,
                     default=float)[:12000])
    return payload


if __name__ == "__main__":
    main()
