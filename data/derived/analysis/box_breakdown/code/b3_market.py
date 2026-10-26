"""갈래 3 — 시장 맥락(BRIEF 갈래 3). 하네스 사건표와 evaluate_cells를 그대로 쓴다.

실행: py b3_market.py
산출: ../results/b3_market_cells.csv(칸 전체), ../results/b3_market_summary.json(요약·귀무 분포),
      ../results/b3_market.md(보고서)

특징(9개, 모두 이탈일 t 종가 뒤 알 수 있음, 진입은 t+1 시가)
- qqq_above50 / qqq_above200 / soxx_above50: t 종가 > 50·200일 단순이동평균(t 포함)
- cobreak_share: 같은 날 같은 유형(종목/ETF)에서 새 사건이 난 종목 수 ÷ 그날 L_N이 있는 같은 유형 종목 수
- breadth50: AI 유니버스 미국 종목 중 t 종가 < 50일선 비율
- vix, vix_ts: ^VIX 종가, ^VIX / ^VIX3M 종가(둘 다 16:15 ET 확정, t+1 시가 전)
- semi: 반도체 대분류(칩 설계·파운드리·후공정·메모리·스토리지·장비·소재·테스트) + SOXX·SMH
- rs20_b3: QQQ 대비 20일 상대강도 = (C_t / C_{t−20}) / (QQQ_t / QQQ_{t−20}) − 1. 이 파일에서 패널로 직접
  계산하고 하네스 rs20_qqq와 같은지 대조한다.

칸: tercile_cells(사건표 N20·40·60 이어 붙임, by=("N",), 종목+ETF 함께) — 하네스 108칸과 같은 방식.
문턱은 앞 구간(2026-01-02~05-29) 사건 분포의 1/3·2/3 분위(이진 특징은 =1·=0).
판정은 evaluate_cells 기본값(top 10, 켜짐 ≥ 15건·날짜 ≥ 8일, 순열 200번, 귀무 200번, 층 N·kind).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
import harness as H  # noqa: E402

RES = H.RESULTS
FEATS = ["qqq_above50", "qqq_above200", "soxx_above50", "cobreak_share", "breadth50",
         "vix", "vix_ts", "semi", "rs20_b3"]
KNOWN = {
    "qqq_above50": "t 종가(QQQ 종가·50일선, t 포함) → t+1 시가 진입 전",
    "qqq_above200": "t 종가(QQQ 종가·200일선) → t+1 시가 진입 전",
    "soxx_above50": "t 종가(SOXX 종가·50일선) → t+1 시가 진입 전",
    "cobreak_share": "t 종가(그날 같은 유형 종목들의 종가 이탈 집계) → t+1 시가 진입 전",
    "breadth50": "t 종가(AI 유니버스 종가·50일선) → t+1 시가 진입 전",
    "vix": "t ^VIX 종가(16:15 ET) → t+1 시가 진입 전",
    "vix_ts": "t ^VIX·^VIX3M 종가(16:15 ET) → t+1 시가 진입 전",
    "semi": "사전 고정 분류(watchlist.AI_UNIVERSE) — 언제나",
    "rs20_b3": "t 종가(종목·QQQ의 t, t−20 종가) → t+1 시가 진입 전",
}
KO = {
    "qqq_above50": "QQQ 50일선 위", "qqq_above200": "QQQ 200일선 위", "soxx_above50": "SOXX 50일선 위",
    "cobreak_share": "같은 날 동반 이탈 비율", "breadth50": "AI 유니버스 50일선 아래 비율",
    "vix": "VIX", "vix_ts": "VIX/VIX3M", "semi": "반도체", "rs20_b3": "QQQ 대비 20일 상대강도",
}
PERIODS = (("앞", H.FRONT), ("뒤", H.BACK), ("2025", H.AUX))


def add_rs20(ev, P):
    """QQQ 대비 20일 상대강도를 패널에서 직접 계산(t, t−20 종가만 사용)."""
    q = P.col("C", "QQQ")
    out = np.full(len(ev), np.nan)
    for k, (s, i) in enumerate(zip(ev["sym"], ev["i"])):
        i = int(i)
        if i < 20:
            continue
        c = P.col("C", s)
        out[k] = (c[i] / c[i - 20]) / (q[i] / q[i - 20]) - 1
    ev = ev.copy()
    ev["rs20_b3"] = out
    return ev


def parse_cell(name):
    """칸 이름 → (N, 특징, 방향, 문턱 문자열)."""
    head, body = name.split("|", 1)
    N = int(head.replace("N", ""))
    for f in sorted(FEATS, key=len, reverse=True):
        if body.startswith(f):
            rest = body[len(f):]
            if rest.startswith("="):
                return N, f, ("=1" if rest.startswith("=1") else "=0"), rest
            return N, f, ("상위3분위" if "상위" in rest else "하위3분위"), rest
    raise ValueError(name)


def exact_thresholds(ev):
    """앞 구간 분위수(전체 정밀도) — 정의 재현용."""
    fr = ev[H.in_period(ev, H.FRONT)]
    th = {}
    for N, g in fr.groupby("N"):
        for f in FEATS:
            x = g[f].dropna().astype(float)
            if len(np.unique(x)) <= 2:
                continue
            th[(N, f)] = tuple(np.quantile(x, [1 / 3, 2 / 3]))
    return th


def definition(N, f, d, th):
    base = f"N={N} 박스(L=직전 {N}거래일 최저 저가) 하단 종가 이탈 사건(종목+ETF), "
    if d in ("=1", "=0"):
        return base + f"{f}{d}({KO[f]} {'예' if d == '=1' else '아니오'})"
    q1, q2 = th[(N, f)]
    if d == "상위3분위":
        return base + f"{f} ≥ {q2:.6g}(앞 구간 2/3 분위)"
    return base + f"{f} ≤ {q1:.6g}(앞 구간 1/3 분위)"


def describe(ev, mask_fn, labels, Ns=H.NS, kind="stock"):
    """기술 통계(판정 아님): 구간·N·집단별 사건 수, 날짜 수, h10 평균, 날짜 묶음 t, 진짜·가짜 비율, 되돌림 청산."""
    rows = []
    for N in Ns:
        e = ev[(ev["N"] == N) & (ev["kind"] == kind) & ev["short_ret_h10"].notna()]
        for pn, p in PERIODS:
            ep = e[H.in_period(e, p)]
            grp = mask_fn(ep, N)
            for lab in labels:
                g = ep[grp == lab]
                with np.errstate(divide="ignore", invalid="ignore"):
                    m, t, n, G = H.cluster_mean(g["short_ret_h10"], g["t"])
                if G < 2:
                    t = np.nan          # 날짜 1일이면 날짜 묶음 t를 못 구함
                c = g["cls"].dropna()
                rows.append({"N": N, "구간": pn, "집단": lab, "사건": n, "날짜": G, "h10": m, "t": t,
                             "h20": g["short_ret_h20"].mean() if len(g) else np.nan,
                             "진짜": (c == "진짜").mean() if len(c) else np.nan,
                             "가짜": (c == "가짜").mean() if len(c) else np.nan,
                             "되돌림": (g["exit_reason"] == "reclaim").mean() if len(g) else np.nan})
    return pd.DataFrame(rows)


def pct(x, nd=2, sign=True):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "못 구함"
    return f"{x * 100:+.{nd}f}%" if sign else f"{x * 100:.{nd}f}%"


def num(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "못 구함"
    return f"{x:.{nd}f}"


def main():
    P = H.load_panel()
    ev = H.load_events_all()
    ev = add_rs20(ev, P)
    chk = (ev["rs20_b3"] - ev["rs20_qqq"]).abs()
    rs_check = {"max_abs_diff_vs_harness": float(np.nanmax(chk.to_numpy())),
                "nan_mismatch": int((ev["rs20_b3"].isna() != ev["rs20_qqq"].isna()).sum())}

    cells = H.tercile_cells(ev, FEATS, by=("N",))
    tab, summ = H.evaluate_cells(ev, cells)
    th = exact_thresholds(ev)

    meta = [parse_cell(n) for n in tab["cell"]]
    tab.insert(1, "N", [m[0] for m in meta])
    tab.insert(2, "feature", [m[1] for m in meta])
    tab.insert(3, "direction", [m[2] for m in meta])
    tab.insert(4, "definition", [definition(m[0], m[1], m[2], th) for m in meta])
    tab.insert(5, "known_when", [KNOWN[m[1]] for m in meta])
    tab.to_csv(RES / "b3_market_cells.csv", index=False, encoding="utf-8-sig")

    # 기술 통계(판정 아님): 핵심 질문 두 개
    def co_bucket(ep, N):
        q1, q2 = th[(N, "cobreak_share")]
        x = ep["cobreak_share"]
        return np.where(x <= q1, "혼자 쪽(하위3분위)", np.where(x >= q2, "함께(상위3분위)", "중간"))

    def alone(ep, N):
        full = ev[(ev["N"] == N) & (ev["kind"] == "stock")]
        cnt = full.groupby("t").size()
        k = ep["t"].map(cnt)
        return np.where(k == 1, "그날 단독(1종목)", np.where(k >= 10, "10종목 이상", "2~9종목"))

    def q50(ep, N):
        return np.where(ep["qqq_above50"] == 1, "QQQ 50일선 위", "QQQ 50일선 아래")

    def s50_semi(ep, N):
        return np.where(ep["semi"] != 1, "비반도체",
                        np.where(ep["soxx_above50"] == 1, "반도체·SOXX 50일선 위", "반도체·SOXX 50일선 아래"))

    d_co = describe(ev, co_bucket, ["혼자 쪽(하위3분위)", "중간", "함께(상위3분위)"])
    d_al = describe(ev, alone, ["그날 단독(1종목)", "2~9종목", "10종목 이상"])
    d_q = describe(ev, q50, ["QQQ 50일선 아래", "QQQ 50일선 위"])
    d_s = describe(ev, s50_semi, ["반도체·SOXX 50일선 아래", "반도체·SOXX 50일선 위", "비반도체"])

    out = {"n_cells": int(summ["n_cells"]), "n_eligible": summ["n_eligible"], "n_selected": summ["n_selected"],
           "selected": summ["selected"], "n_pass": summ["n_pass"], "passed": summ["passed"],
           "passed_one_regime": summ["passed_one_regime"], "n_pass_shift": summ["n_pass_shift"],
           "events": summ["events"], "dates": summ["dates"], "params": summ["params"],
           "null": {k: v for k, v in summ.get("null", {}).items()},
           "null_shift": {k: v for k, v in summ.get("null_shift", {}).items()},
           "rs20_check": rs_check,
           "thresholds": {f"N{N}|{f}": list(v) for (N, f), v in th.items()},
           "per_N_events": {p: ev[H.in_period(ev, per) & ev["short_ret_h10"].notna()].groupby("N").size().to_dict()
                            for p, per in (("f", H.FRONT), ("b", H.BACK), ("a", H.AUX))},
           "per_N_dates": {p: ev[H.in_period(ev, per) & ev["short_ret_h10"].notna()].groupby("N")["t"].nunique().to_dict()
                           for p, per in (("f", H.FRONT), ("b", H.BACK), ("a", H.AUX))}}
    (RES / "b3_market_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str),
                                                encoding="utf-8")
    for nm, d in (("co", d_co), ("alone", d_al), ("q50", d_q), ("soxx_semi", d_s)):
        d.to_csv(RES / f"b3_market_desc_{nm}.csv", index=False, encoding="utf-8-sig")
    write_md(tab, summ, out, d_co, d_al, d_q, d_s)
    return tab, summ, out


def _g(d, N, per, grp, col):
    r = d[(d["N"] == N) & (d["구간"] == per) & (d["집단"] == grp)]
    return float(r[col].iloc[0]) if len(r) else float("nan")


def conclusion(tab, summ, out, d_co, d_q, d_s):
    """결론 문장(숫자는 모두 계산 결과에서 가져온다)."""
    sel = tab[tab["selected"]]
    pos_back = int((sel["b_mean_on"] > 0).sum())
    co = tab[tab["cell"].str.startswith("N20|cobreak_share") & tab["direction"].eq("하위3분위")].iloc[0]
    q = tab[tab["cell"].eq("N20|qqq_above50=0(lo)")].iloc[0]
    hi, lo, mid = "함께(상위3분위)", "혼자 쪽(하위3분위)", "중간"
    a, b = "QQQ 50일선 아래", "QQQ 50일선 위"
    L = [
        f"- **통과 {summ['n_pass']}/{out['n_cells']}칸.** 선택편향 귀무 평균 {out['null']['mean']:.3f}칸(p = {out['null']['p_value']:.2f}), "
        f"원형 이동 귀무 평균 {out['null_shift']['mean']:.3f}칸. 앞 구간 상위 10칸 가운데 뒤 구간에서 켜진 사건의 공매도 평균이 0보다 큰 칸은 "
        f"{pos_back}개다. 시장 맥락 하나로 '공매도해서 버는 진짜 하락'을 고르는 조건은 2026년에 없었다.",
        f"- **혼자 깨는 이탈 vs 함께 깨는 이탈: 함께 깨는 쪽이 더 가짜다.** 종목 N20 공매도 h10은 뒤 구간에서 동반 이탈 상위 3분위 "
        f"{pct(_g(d_co, 20, '뒤', hi, 'h10'))}(가짜 {pct(_g(d_co, 20, '뒤', hi, '가짜'), 0, False)}, 되돌림 청산 "
        f"{pct(_g(d_co, 20, '뒤', hi, '되돌림'), 0, False)}), 하위 3분위 {pct(_g(d_co, 20, '뒤', lo, 'h10'))}"
        f"(가짜 {pct(_g(d_co, 20, '뒤', lo, '가짜'), 0, False)})였다. 앞 구간도 {pct(_g(d_co, 20, '앞', hi, 'h10'))} vs "
        f"{pct(_g(d_co, 20, '앞', lo, 'h10'))}, 2025도 {pct(_g(d_co, 20, '2025', hi, 'h10'))} vs {pct(_g(d_co, 20, '2025', lo, 'h10'))}. "
        f"'혼자' 칸(N20 동반 비율 하위 3분위)은 앞 1위로 뽑혔고 뒤 차이 {pct(co['b_diff'])}p(t {co['b_t']:.2f}, p {co['b_p']:.3f}, "
        f"원형 p {co['b_p_shift']:.3f})였지만 켜진 사건 자체가 {pct(co['b_mean_on'])}라 '덜 잃는' 조건일 뿐 공매도 이익 조건이 아니다.",
        f"- **시장(QQQ)이 50일선 아래일 때 이탈은 진짜가 되지 않는다. 오히려 더 크게 되돌린다.** 종목 N20 뒤 구간 공매도 h10: "
        f"50일선 아래 {pct(_g(d_q, 20, '뒤', a, 'h10'))}(진짜 {pct(_g(d_q, 20, '뒤', a, '진짜'), 0, False)}, 가짜 "
        f"{pct(_g(d_q, 20, '뒤', a, '가짜'), 0, False)}) vs 위 {pct(_g(d_q, 20, '뒤', b, 'h10'))}. 앞 구간 "
        f"{pct(_g(d_q, 20, '앞', a, 'h10'))} vs {pct(_g(d_q, 20, '앞', b, 'h10'))}, 2025 {pct(_g(d_q, 20, '2025', a, 'h10'))} vs "
        f"{pct(_g(d_q, 20, '2025', b, 'h10'))}. N20 '50일선 아래' 칸의 차이는 앞 {pct(q['f_diff'])}p, 뒤 {pct(q['b_diff'])}p로 공매도에 불리한 쪽이다. "
        f"반도체 종목도 SOXX 50일선 아래에서 뒤 {pct(_g(d_s, 20, '뒤', '반도체·SOXX 50일선 아래', 'h10'))}였다.",
        f"- **2025는 장세가 달랐다(전환 위험).** 함께 깨는 이탈의 진짜 비율이 2025 N20 {pct(_g(d_co, 20, '2025', hi, '진짜'), 0, False)}, "
        f"N60 {pct(_g(d_co, 60, '2025', hi, '진짜'), 0, False)}로 2026 앞·뒤({pct(_g(d_co, 20, '앞', hi, '진짜'), 0, False)}·"
        f"{pct(_g(d_co, 20, '뒤', hi, '진짜'), 0, False)})보다 높았다. 그래도 h10 공매도는 {pct(_g(d_co, 20, '2025', hi, 'h10'))}였다. "
        f"진짜로 L−2ATR까지 빠져도 V자 반등이 되돌림 손절을 친 것이다.",
        "- 시장 맥락은 같은 날 모든 사건에 같은 값이라 '초과 공매도 손익'으로는 잴 수 없다. 이 갈래의 차이는 사실상 '어느 국면에 공매도했나'다. "
        "뒤 구간 사건은 대부분 두 국면(episode 34·35)에 몰려 있다.",
    ]
    return L


def desc_table(d, title):
    L = [f"#### {title}", "",
         "| N | 구간 | 집단 | 사건 | 날짜 | h10 공매도 | 날짜 묶음 t | h20 | 진짜 | 가짜 | 되돌림 청산 |",
         "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in d.itertuples():
        L.append(f"| {r.N} | {r.구간} | {r.집단} | {r.사건} | {r.날짜} | {pct(r.h10)} | {num(r.t)} | {pct(r.h20)} | "
                 f"{pct(r.진짜, 0, False)} | {pct(r.가짜, 0, False)} | {pct(r.되돌림, 0, False)} |")
    L.append("")
    return L


def write_md(tab, summ, out, d_co, d_al, d_q, d_s):
    sel = tab[tab["selected"]].sort_values("rank").rename(columns={"pass": "passed"})
    nl, ns = out["null"], out["null_shift"]
    L = ["# 갈래 3 — 시장 맥락: 혼자 깨는 이탈 vs 시장과 함께 깨는 이탈 (2026-10-14)", ""]
    L += ["## 결론", ""] + conclusion(tab, summ, out, d_co, d_q, d_s) + [""]
    L += ["## 시험 범위", "",
          f"- 특징 9개: {', '.join(FEATS)} (`rs20_b3`는 이 갈래에서 직접 계산, 하네스 `rs20_qqq`와 최대 차이 "
          f"{out['rs20_check']['max_abs_diff_vs_harness']:.2e}, NaN 불일치 {out['rs20_check']['nan_mismatch']}).",
          f"- 칸 = 특징 × 방향(연속형은 앞 구간 3분위 위/아래, 이진형은 =1/=0) × N(20·40·60), 종목+ETF 함께. "
          f"앞 구간에서 값이 한 가지뿐인 이진 칸은 만들지 않음 → **{out['n_cells']}칸**.",
          f"- 고르기 자격(앞 차이 > 0, 켜짐 ≥ 15건, 켜짐 날짜 ≥ 8일) {out['n_eligible']}칸 → 상위 {out['n_selected']}칸 확인.",
          "- 판정·순열·귀무는 `harness.evaluate_cells` 기본값 그대로(순열 200번, 귀무 200번, 층 N·kind). "
          "보조로 원형 이동 p(`b_p_shift`)와 원형 이동 귀무(`null_shift`)를 같이 적는다(국면형 날짜 특징 거짓 양성 보정, qa_harness.md).",
          f"- 사건(N 세 표 합친 행): 앞 {out['events']['f']}건/{out['dates']['f']}일, 뒤 {out['events']['b']}건/"
          f"{out['dates']['b']}일, 2025 {out['events']['a']}건/{out['dates']['a']}일. N별 앞 {out['per_N_events']['f']}, "
          f"뒤 {out['per_N_events']['b']}(서로 다른 날짜 앞 {out['per_N_dates']['f']}, 뒤 {out['per_N_dates']['b']}).",
          "- 앞선 박스·손절·레버리지·옵션 연구를 본 뒤의 탐색이라 선택편향이 쌓여 있다.", ""]
    L += ["## 판정", "",
          f"- 통과(BRIEF): **{summ['n_pass']}칸** {summ['passed'] or ''}",
          f"- 선택편향 귀무(날짜 i.i.d. 순열 200번): 통과 칸 수 평균 {nl.get('mean', float('nan')):.3f}, "
          f"95% 분위 {nl.get('q95', float('nan')):.2f}, 최대 {nl.get('max', 'NA')}, 관측 이상 p = {nl.get('p_value', float('nan')):.3f}",
          f"- 보조 원형 이동 귀무: 평균 {ns.get('mean', float('nan')):.3f}, 95% 분위 {ns.get('q95', float('nan')):.0f}, "
          f"p = {ns.get('p_value', float('nan')):.3f}. 원형 이동 p까지 통과(pass_shift): {summ['n_pass_shift']}칸.",
          f"- 한 국면 의존 통과: {summ['passed_one_regime'] or '없음'}", ""]
    L += ["## 앞 구간 상위 10칸 (통과 여부 무관)", "",
          "공매도 손익 = t+1 시가 공매도, 되돌림 손절 또는 t+10 종가, 비용 0.10% 뺀 값(%/거래). 차이 = 켜짐 − 꺼짐(%p). 날짜고정 = 같은 날 사건의 표지가 모두 같음.",
          "",
          "| 순위 | 칸 | 날짜고정 | 앞 차이 | 앞 켜짐 평균 | 앞 켜짐(건/일) | 뒤 차이 | 뒤 켜짐 평균 | 뒤 켜짐(건/일) | 뒤 t | p | p(원형) | 2025 차이 | 2025 같은 방향 | 최대 국면 비중 | 통과 |",
          "|---:|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:-:|---:|:-:|"]
    for r in sel.itertuples():
        L.append(f"| {int(r.rank)} | `{r.cell}` | {'예' if r.date_const else '아니오'} | {pct(r.f_diff)} | {pct(r.f_mean_on)} | "
                 f"{int(r.f_n_on)}/{int(r.f_dates_on)} | {pct(r.b_diff)} | {pct(r.b_mean_on)} | {int(r.b_n_on)}/{int(r.b_dates_on)} | "
                 f"{num(r.b_t)} | {num(r.b_p, 3)} | {num(r.b_p_shift, 3)} | {pct(r.a_diff)} | {'예' if r.a_same_dir else '아니오'} | "
                 f"{pct(r.regime_top_share, 0, False) if np.isfinite(r.regime_top_share) else '해당 없음(켜짐 손익 합 ≤ 0)'} | "
                 f"{'통과' if r.passed else '탈락'} |")
    L.append("")
    # 핵심 질문 칸(전체 칸 표에서 해당 특징만)
    L += ["## 핵심 질문 칸 (판정 표에서 발췌, 고르기 여부 무관)", "",
          "| 칸 | 앞 차이 | 앞 켜짐 평균 | 뒤 차이 | 뒤 켜짐 평균 | 뒤 t | 2025 차이 | 고름 |",
          "|---|---:|---:|---:|---:|---:|---:|:-:|"]
    for r in tab[tab["feature"].isin(["cobreak_share", "qqq_above50", "soxx_above50", "breadth50"])].itertuples():
        L.append(f"| `{r.cell}` | {pct(r.f_diff)} | {pct(r.f_mean_on)} | {pct(r.b_diff)} | {pct(r.b_mean_on)} | {num(r.b_t)} | "
                 f"{pct(r.a_diff)} | {'예' if r.selected else ''} |")
    L.append("")
    L += ["## 기술 통계 (판정 아님, 종목 사건만)", "",
          "가짜·진짜 비율은 BRIEF 분류(설명용, t 뒤 가격 사용). 3분위 문턱은 앞 구간 값.", ""]
    L += desc_table(d_co, "같은 날 동반 이탈 비율 3분위")
    L += desc_table(d_al, "그날 같은 N에서 새로 이탈한 종목 수")
    L += desc_table(d_q, "QQQ 50일선 위/아래")
    L += desc_table(d_s, "반도체 종목 × SOXX 50일선")
    L += ["## 한계", "",
          "- 시장 맥락 특징은 같은 날 모든 사건이 같은 값을 갖는다(날짜 고정). 그래서 같은 날 평균을 빼는 '초과 공매도 손익'으로는 "
          "효과가 구조적으로 0에 가깝고, 판정은 절대 공매도 손익으로만 한다. 날짜 고정 칸은 사실상 '그 국면에 공매도했나'를 재는 것이다.",
          "- 뒤 구간 사건의 97~100%가 episode 34(5/29~7/31)·35(8/11~9/16) 두 국면에 몰려 있어 '한 국면 의존' 점검 해상도가 낮다.",
          "- 국면형 날짜 특징(VIX·시장폭·50/200일선)은 BRIEF 순열만으로 거짓 양성이 명목보다 많다(가짜 AR(1) 5~8%). 원형 이동 p의 해상도는 1/D(뒤 52일이면 약 0.02).",
          "- 대차 비용·배당 지급 미반영, 개별 종목 공매도는 대리 수익(실행 가능성은 갈래 7).",
          "- 다른 해 실패는 장세 전환 위험으로만 본다(2025는 같은 방향 여부만).", ""]
    L += ["## 파일", "", "- 코드 `code/b3_market.py` (`py b3_market.py`)",
          "- 칸 전체 `results/b3_market_cells.csv`, 요약·귀무 분포 `results/b3_market_summary.json`",
          "- 기술 통계 `results/b3_market_desc_{co,alone,q50,soxx_semi}.csv`", ""]
    (RES / "b3_market.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    tab, summ, out = main()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_colwidth", 60)
    cols = ["rank", "cell", "date_const", "f_diff", "f_mean_on", "f_n_on", "f_dates_on", "b_diff", "b_mean_on", "b_n_on",
            "b_t", "b_p", "b_p_shift", "a_diff", "regime_top_share", "pass", "pass_shift"]
    print(tab[tab["selected"]].sort_values("rank")[cols].to_string())
    print({k: out[k] for k in ("n_cells", "n_eligible", "n_selected", "n_pass", "n_pass_shift", "events", "dates")})
    print("null", {k: v for k, v in out["null"].items() if k != "dist"})
    print("null_shift", {k: v for k, v in out["null_shift"].items() if k != "dist"})
    print("rs20", out["rs20_check"])
