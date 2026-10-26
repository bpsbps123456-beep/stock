"""H01 관문 미리보기 문서 생성: results/h01_preview.json → results/h01_preview.md. 숫자는 모두 json에서 읽는다."""

from __future__ import annotations

from pathlib import Path

BN = {"kr_price": "한국 가격", "us_night": "미국 밤", "voltrade": "변동·거래", "cal": "달력·품질", "hynix": "하닉 가격(보조)"}
SN = {"main": "본판(판정 자격 창 전체)", "excl_crash_union": "급락 4일 빼기: 그날과 겹치는 창 모두(주 읽기)",
      "excl_crash_P": "급락 4일 빼기: P(가격 세션)가 그날인 창", "excl_crash_D": "급락 4일 빼기: D(창 날짜)가 그날인 창",
      "excl_opened8": "열어 본 8일과 겹치는 창 빼기(강건 ③, 보조)", "excl_top2": "기여가 가장 큰 2창 빼기(강건 ②, 보조)"}
COLS = {"cal_weekend": "주말 창", "cal_holiday": "연휴 창", "cal_xholiday": "휴장 엇갈림", "cal_vacation": "방학 달(7~8월)",
        "cal_kr_opex": "KOSPI 옵션 만기", "cal_us_opex": "미국 월물 만기", "cal_fomc": "FOMC", "cal_earn": "실적(5사)",
        "missing_rate": "결번율", "kospi_rpos": "max(r(D−1),0)", "kospi_rneg": "min(r(D−1),0)", "kospi_r5": "r5",
        "kospi_r20": "r20", "kospi_r60": "r60", "kospi_DD20": "DD20", "sox_r_u": "SOX 직전 세션", "ewy_r_u": "EWY 직전 세션",
        "kospi_lnRV20": "ln RV20", "kospi_ln_vol_anom": "ln 거래량 이상치(P)"}


def f(x, d: int = 3) -> str:
    if x is None:
        return "—"
    try:
        if x != x:
            return "결측"
    except TypeError:
        return str(x)
    return f"{x:.{d}f}"


def p(x) -> str:
    if x is None:
        return "—"
    return "<0.0001" if x < 0.0001 else f"{x:.4f}"


def ok(b: bool) -> str:
    return "충족" if b else "미달"


def write_md(res: dict, path: Path) -> None:
    c, v = res["cells"], res["verdict_preview"]
    smp = res["samples"]
    n = smp["main"]["n"]
    nx = smp["excl_crash_union"]["n"]
    L = []
    a = L.append
    a("# H01 관문 미리보기 — 지난 가격만")
    a("")
    a(f"> 만든 때: {res['created']}. 코드 `interim/code/h01_preview.py`, 해석 기록 `interim/notes_h01_preview.md`, 숫자 원본 `results/h01_preview.json`.")
    a("> **미리보기다. 관문 판정이 아니다.** 지금 자료(국내 본판 창 2026-07-15~10-15)만 썼다. "
      "관문의 최종 판정은 수집이 끝난 뒤 2026 전체 창(2026-01-02~10-23, 약 195창)으로 한다(HYPOTHESES.md §1 순서 5, §2.7). "
      "이 결과를 보고 정의·문턱·단어를 바꾸지 않는다.")
    a("> S는 측정 실패 묶음으로 만든 값이다(사전 v0.2에서 E 정밀도 38%, F 35%로 60% 미만). "
      "관문 숫자와 별개로 H02(TMP 대체) 문제는 그대로 남는다.")
    a("")
    k0, h0 = c["kospi"], c["hynix"]
    rob = [nm for nm in ("main", "excl_crash_union", "excl_crash_P", "excl_crash_D", "excl_opened8")]
    sp_min = min(min(k0[nm]["spearman"]["rho"], h0[nm]["spearman"]["rho"]) for nm in rob + [])
    sp_min = min(sp_min, k0["excl_top2_kospi"]["spearman"]["rho"], h0["excl_top2_hynix"]["spearman"]["rho"])
    sp_pmax = max(max(k0[nm]["spearman"]["p_two"], h0[nm]["spearman"]["p_two"]) for nm in rob)
    sp_pmax = max(sp_pmax, k0["excl_top2_kospi"]["spearman"]["p_two"], h0["excl_top2_hynix"]["spearman"]["p_two"])
    t0 = res["m0"]["main"]
    a("## 요약")
    a("")
    a(f"- 관문 기준 대비(미리보기): KOSPI 칸 {sum(v['kospi']['conditions'].values())}/6, 하닉 칸 {sum(v['hynix']['conditions'].values())}/6 → **{v['gate_position']}**. "
      f"KOSPI Pearson {f(k0['main']['pearson']['rho'], 2)}(HAC p {p(k0['main']['pearson']['p_two'])}), Spearman {f(k0['main']['spearman']['rho'], 2)}.")
    in_crash = all(w in smp["excl_crash_union"]["dropped"] for w in res["top2"]["kospi"])
    a(f"- 약한 곳: Pearson의 유의성은 기여가 가장 큰 2창({', '.join(res['top2']['kospi'])}"
      + (", 둘 다 급락 4일과 겹치는 창" if in_crash else "") + f")에 기댄다. 이 2창을 빼면 KOSPI Pearson {f(k0['excl_top2_kospi']['pearson']['rho'], 2)}, "
      f"HAC p {p(k0['excl_top2_kospi']['pearson']['p_two'])}. 급락 4일을 'D가 그날인 창'으로만 읽어 빼면 Pearson {f(k0['excl_crash_D']['pearson']['rho'], 2)}, "
      f"p {p(k0['excl_crash_D']['pearson']['p_two'])}(0.15는 넘음). Spearman은 모든 강건·보조 표본에서 {f(sp_min, 2)} 이상, p {p(sp_pmax)} 이하였다.")
    a(f"- M0: R² {f(t0['r2'], 2)}(수정 {f(t0['adj_r2'], 2)}). Shapley 몫은 한국 가격 {f(t0['blocks']['kr_price']['shapley'], 2)}, 미국 밤 {f(t0['blocks']['us_night']['shapley'], 2)}, "
      f"변동·거래 {f(t0['blocks']['voltrade']['shapley'], 2)}, 달력·품질 {f(t0['blocks']['cal']['shapley'], 2)}. S를 원형 이동한 귀무에서도 R² 중앙값 {f(t0['null']['r2']['null_median'], 2)}로, "
      "크기는 부풀어 있다. 이동한 귀무 "
      f"{t0['null']['K']}개를 모두 넘은 블록: " + (", ".join(BN[b] for b in t0["blocks"] if t0["null"][b]["p"] <= t0["null"]["min_p"] + 1e-12) or "없음")
      + " (" + ", ".join(f"{BN[b]} p {p(t0['null'][b]['p'])}" for b in t0["blocks"]) + ").")
    a("")
    a("## 1. 관문 기준 대비 위치")
    a("")
    a(f"기준(§2.7): 2026 corr(S, KOSPI r(D−1)) 또는 같은 정의의 하닉 칸이 0.15 이상이고 HAC(5) p < 0.05. "
      f"Pearson·Spearman 모두, 급락 4일(7/29~31, 10/13)을 뺀 판도 0.15 이상. 표본 {n}창(급락 제외판 {nx}창). p는 양측(보수).")
    a("")
    a("| 조건 | KOSPI | 하닉 | 기준 |")
    a("|---|---|---|---|")
    k, h = c["kospi"], c["hynix"]
    a(f"| Pearson | {f(k['main']['pearson']['rho'])} | {f(h['main']['pearson']['rho'])} | ≥ 0.15 |")
    a(f"| Spearman | {f(k['main']['spearman']['rho'])} | {f(h['main']['spearman']['rho'])} | ≥ 0.15 |")
    a(f"| Pearson HAC(5) p | {p(k['main']['pearson']['p_two'])} | {p(h['main']['pearson']['p_two'])} | < 0.05 |")
    a(f"| Spearman HAC(5) p | {p(k['main']['spearman']['p_two'])} | {p(h['main']['spearman']['p_two'])} | < 0.05 |")
    a(f"| 급락 4일 뺀 Pearson | {f(k['excl_crash_union']['pearson']['rho'])} | {f(h['excl_crash_union']['pearson']['rho'])} | ≥ 0.15 |")
    a(f"| 급락 4일 뺀 Spearman | {f(k['excl_crash_union']['spearman']['rho'])} | {f(h['excl_crash_union']['spearman']['rho'])} | ≥ 0.15 |")
    nk = sum(v["kospi"]["conditions"].values())
    nh = sum(v["hynix"]["conditions"].values())
    a(f"| 충족한 조건 | {nk}/6 | {nh}/6 | 한 칸이 6/6이면 관문 기준 충족 |")
    a("")
    pos = v["gate_position"]
    a(f"- **미리보기 위치: 기준 {pos}.** "
      + ("두 칸 모두 여섯 조건을 다 맞췄다." if v["kospi"]["all_met"] and v["hynix"]["all_met"] else
         "충족한 칸: " + (", ".join(x for x in ("kospi", "hynix") if v[x]["all_met"]) or "없음") + "."))
    a(f"- 단측 p로 바꿔도 결론이 같다: KOSPI {'바뀜' if v['kospi']['p_one_side_would_change'] else '같음'}, "
      f"하닉 {'바뀜' if v['hynix']['p_one_side_would_change'] else '같음'}.")
    rm = res["r_min_iid_p05"]
    a(f"- 이 표본에서 p < 0.05가 되려면 상관이 대략 {f(rm['main'], 2)} 이상이어야 한다(iid 가정). 0.15 문턱보다 p 조건이 더 빡빡하다. "
      f"2026 전체(약 195창)에서는 이 값이 약 {f(rm['full_2026_approx_195'], 2)}로 내려간다.")
    a(f"- S의 이웃 창 자기상관 {f(res['ac1']['S'], 2)}, KOSPI r(D−1) {f(res['ac1']['kospi_r'], 2)}, 하닉 {f(res['ac1']['hynix_r'], 2)}.")
    a("")
    a("## 2. 강건·보조 표 (판정에 쓰지 않는다)")
    a("")
    a("| 표본 | 창 | KOSPI Pearson (p) | KOSPI Spearman (p) | 하닉 Pearson (p) | 하닉 Spearman (p) |")
    a("|---|---|---|---|---|---|")
    for nm in ("main", "excl_crash_union", "excl_crash_P", "excl_crash_D", "excl_opened8"):
        a(f"| {SN[nm]} | {smp[nm]['n']} | {f(k[nm]['pearson']['rho'])} ({p(k[nm]['pearson']['p_two'])}) | "
          f"{f(k[nm]['spearman']['rho'])} ({p(k[nm]['spearman']['p_two'])}) | {f(h[nm]['pearson']['rho'])} ({p(h[nm]['pearson']['p_two'])}) | "
          f"{f(h[nm]['spearman']['rho'])} ({p(h[nm]['spearman']['p_two'])}) |")
    kt, ht = k["excl_top2_kospi"], h["excl_top2_hynix"]
    a(f"| {SN['excl_top2']} | {smp['excl_top2_kospi']['n']} | {f(kt['pearson']['rho'])} ({p(kt['pearson']['p_two'])}) | "
      f"{f(kt['spearman']['rho'])} ({p(kt['spearman']['p_two'])}) | {f(ht['pearson']['rho'])} ({p(ht['pearson']['p_two'])}) | "
      f"{f(ht['spearman']['rho'])} ({p(ht['spearman']['p_two'])}) |")
    pm = res["partial_missing_rate"]
    a(f"| 결번율 통제(부분 상관) | {n} | {f(pm['kospi']['pearson']['rho'])} ({p(pm['kospi']['pearson']['p_two'])}) | "
      f"{f(pm['kospi']['spearman']['rho'])} ({p(pm['kospi']['spearman']['p_two'])}) | {f(pm['hynix']['pearson']['rho'])} ({p(pm['hynix']['pearson']['p_two'])}) | "
      f"{f(pm['hynix']['spearman']['rho'])} ({p(pm['hynix']['spearman']['p_two'])}) |")
    pmx = pm["kospi"]["excl_crash_union"], pm["hynix"]["excl_crash_union"]
    a(f"| 결번율 통제 + 급락 4일 빼기 | {nx} | {f(pmx[0]['pearson']['rho'])} ({p(pmx[0]['pearson']['p_two'])}) | "
      f"{f(pmx[0]['spearman']['rho'])} ({p(pmx[0]['spearman']['p_two'])}) | {f(pmx[1]['pearson']['rho'])} ({p(pmx[1]['pearson']['p_two'])}) | "
      f"{f(pmx[1]['spearman']['rho'])} ({p(pmx[1]['spearman']['p_two'])}) |")
    a("")
    a("- 괄호는 HAC(5) 양측 p. 뺀 창 목록은 json `samples`에 있다.")
    a(f"- 기여가 가장 큰 2창: KOSPI {', '.join(res['top2']['kospi'])} / 하닉 {', '.join(res['top2']['hynix'])}.")
    cb = res["contrib"]["kospi"]
    a("- KOSPI Pearson 창별 기여(z_S·z_r ÷ (n−1), 합 = 상관): 가장 큰 쪽 "
      + ", ".join(f"{w} {f(x, 3)}" for w, x in cb["most_positive"]) + " / 가장 작은 쪽 "
      + ", ".join(f"{w} {f(x, 3)}" for w, x in cb["most_negative"]) + ". "
      "'D가 그날인 창'만 빼면 큰 쪽 창은 빠지고 가장 작은 쪽 창(급락 뒤 반등 세션과 짝지은 창)은 남아 Pearson이 낮아진다.")
    ck = k["main"]["pearson"]["circ"]
    a(f"- §2.7 p 계산 ①(21~N−21 원형 이동): 본판 이동 {ck['K']}개. KOSPI Pearson {p(ck['p_two'])}, Spearman {p(k['main']['spearman']['circ']['p_two'])}, "
      f"하닉 Pearson {p(h['main']['pearson']['circ']['p_two'])}, Spearman {p(h['main']['spearman']['circ']['p_two'])}. "
      f"이동이 {ck['K']}개라 가장 작은 p가 {f(ck['min_p'], 3)}이고, 0.05 아래로 내려갈 수 없다. 관문 문장은 HAC(5) p라서 판정 대신 참고로 둔다.")
    a("")
    a("## 3. M0 블록별 부분 R² (S를 한국 M0에 회귀, 표본 안 OLS)")
    a("")
    for nm in ("main", "excl_crash_union"):
        t = res["m0"][nm]
        nu = t["null"]
        a(f"### {SN[nm]} — {t['n']}창, 열 {t['k']}개")
        a("")
        a(f"- R² {f(t['r2'])}, 수정 R² {f(t['adj_r2'])}. 서로 무관한 iid 자료에서 기대되는 R²는 약 {f(t['iid_null_mean_r2'])}(열 수 ÷ (n−1)).")
        a(f"- S를 원형 이동한 귀무(이동 {nu['K']}개): R² 중앙값 {f(nu['r2']['null_median'])}, 상위 10% {f(nu['r2']['null_p90'])}, p {p(nu['r2']['p'])}(가장 작은 값 {f(nu['min_p'])}). "
          "S와 r20·DD20처럼 천천히 움직이는 열끼리는 우연만으로도 R²가 크게 나온다.")
        bl = "; ".join(f"{BN[b]}: " + ", ".join(COLS.get(x, x) for x in t['blocks'][b]['cols']) for b in t["blocks"])
        a(f"- 열: {bl}.")
        if t["dropped"]:
            a(f"- 뺀 열: {', '.join(t['dropped'])}.")
        a("")
        a("| 블록 | 열 수 | 단독 R² | 뺄 때 줄어드는 R² | 부분 R² | Shapley 몫 | 귀무 부분 R² 중앙값 | 원형 이동 p |")
        a("|---|---|---|---|---|---|---|---|")
        for b, r in t["blocks"].items():
            a(f"| {BN[b]} | {r['k']} | {f(r['r2_alone'])} | {f(r['delta_r2'])} | {f(r['partial_r2'])} | {f(r['shapley'])} | "
              f"{f(nu[b]['null_median'])} | {p(nu[b]['p'])} |")
        he = t["hynix_extra"]
        a(f"| {BN['hynix']} | {he['k']} | {f(he['r2_alone'])} | {f(he['delta_r2'])} | {f(he['partial_r2'])} | — | — | — |")
        a("")
        a(f"- Shapley 몫은 네 블록의 합이 R²({f(t['r2'])})가 되도록 나눈 값이다. 부분 R² = (그 블록을 뺀 모형의 잔차 제곱합 − 전체 모형의 것) ÷ 뺀 모형의 것.")
        a(f"- 하닉 가격(보조)은 M0 네 블록 위에 하닉 max(r,0)·min(r,0)·r5·r20을 더한 값이다(R² {f(he['r2_with'])}, 수정 R² {f(he['adj_r2_with'])}). M0 정의에는 넣지 않았다.")
        a("")
    t = res["m0"]["main"]
    a("**H01 기록 규칙 대비(미리보기)**")
    a("")
    a(f"- M0 R² < 0.1이면 '이 자료에서는 결론 2 불성립': 해당 없음(R² {f(t['r2'])}, 수정 R² {f(t['adj_r2'])}).")
    a(f"- §2.3 'M0 R² 0.5 이상이면 메아리 우세': S는 **메아리 우세**로 표시된다. 다만 원형 이동 귀무 중앙값도 {f(t['null']['r2']['null_median'])}라서, "
      "R² 크기의 상당 부분은 느리게 움직이는 열끼리의 우연이다. "
      + ("관측 R²는 이동한 귀무 모두보다 컸다." if t["null"]["r2"]["p"] <= t["null"]["min_p"] + 1e-12 else f"관측 R²의 원형 이동 p는 {p(t['null']['r2']['p'])}다."))
    us = res["aux"]
    a(f"- '미국 밤하고만 맞으면 창 정의 문제': 해당 없음. S는 KOSPI r(D−1)과도({f(k['main']['pearson']['rho'], 2)}) 미국 밤과도"
      f"(SOX {f(us['sox_r_u']['pearson']['rho'], 2)}, EWY {f(us['ewy_r_u']['pearson']['rho'], 2)}) 비슷하게 맞는다.")
    a("")
    a("## 4. 보조 상관 (설명용)")
    a("")
    a("| S와 짝 지은 값(모두 창 마감 전) | Pearson (p) | Spearman (p) |")
    a("|---|---|---|")
    for x, lab in (("sox_r_u", "SOX 직전 미국 세션"), ("ewy_r_u", "EWY 직전 미국 세션"), ("kospi_r5", "KOSPI r5"),
                   ("kospi_r20", "KOSPI r20"), ("kospi_r60", "KOSPI r60"), ("kospi_DD20", "KOSPI DD20"),
                   ("kospi_lnRV20", "KOSPI ln RV20"), ("kospi_ln_vol_anom", "KOSPI ln 거래량 이상치"),
                   ("hynix_r5", "하닉 r5"), ("hynix_r20", "하닉 r20"), ("missing_rate", "결번율")):
        r = us[x]
        a(f"| {lab} | {f(r['pearson']['rho'])} ({p(r['pearson']['p_two'])}) | {f(r['spearman']['rho'])} ({p(r['spearman']['p_two'])}) |")
    a("")
    a("| 성분 × r(D−1) | KOSPI Pearson (p) | KOSPI Spearman (p) | 하닉 Pearson (p) | 하닉 Spearman (p) |")
    a("|---|---|---|---|---|")
    for y, lab in (("E_A_P", "E 작성자 비율(글)"), ("F_A_P", "F 작성자 비율(글)"), ("E_A_C", "E(댓글)"), ("F_A_C", "F(댓글)"),
                   ("S_P", "S 글 쪽"), ("S_C", "S 댓글 쪽")):
        r = res["components"][y]
        a(f"| {lab} | {f(r['kospi']['pearson']['rho'])} ({p(r['kospi']['pearson']['p_two'])}) | {f(r['kospi']['spearman']['rho'])} ({p(r['kospi']['spearman']['p_two'])}) | "
          f"{f(r['hynix']['pearson']['rho'])} ({p(r['hynix']['pearson']['p_two'])}) | {f(r['hynix']['spearman']['rho'])} ({p(r['hynix']['spearman']['p_two'])}) |")
    a("")
    a("지연 1~20일(설명용): corr(S(D), r(P−k)), P−k = P보다 k세션 앞. 괄호는 Pearson HAC(5) p.")
    a("")
    a("| k | KOSPI Pearson (p) | KOSPI Spearman | 하닉 Pearson (p) | 하닉 Spearman |")
    a("|---|---|---|---|---|")
    for kk in range(1, 21):
        rk, rh = res["lags"]["kospi"][str(kk)], res["lags"]["hynix"][str(kk)]
        a(f"| {kk} | {f(rk['pearson']['rho'], 2)} ({p(rk['pearson']['p_two'])}) | {f(rk['spearman']['rho'], 2)} | "
          f"{f(rh['pearson']['rho'], 2)} ({p(rh['pearson']['p_two'])}) | {f(rh['spearman']['rho'], 2)} |")
    a("")
    a("## 5. 자료와 방화벽")
    a("")
    a("| 항목 | 값 |")
    a("|---|---|")
    for kk, pr in res["prices"].items():
        a(f"| {pr['ticker']} | {pr['first']} ~ {pr['last']}, {pr['n']}봉. 종가만 남겼다" + (" (거래량도)" if kk == "kospi" else "") + " |")
    a("| 창 ↔ 가격 | 창 D에는 P일(창 안에서 끝난 마지막 KRX 세션, 주말 창은 금요일) 종가까지, 미국은 D−1일까지의 세션까지만 붙였다 |")
    a("| 방화벽 | 시가·고가·저가는 받자마자 버렸다. 받은 가격의 마지막 날짜가 10/14를 넘으면 멈춘다. 진입 이후 값(R_h·MDD_h·D일 HL·시가→종가·갭)은 만들지 않았다 |")
    ci = res["calendar_info"]
    a(f"| 일정 더미 | KOSPI 옵션 만기 {', '.join(ci['kr_opex'])} / 미국 월물 만기 {', '.join(ci['us_opex'])} / FOMC {', '.join(ci['fomc_in_range'])} / "
      f"실적 " + "; ".join(f"{t_}: {', '.join(d_) if isinstance(d_, list) else d_}" for t_, d_ in ci['earn_in_range'].items()) + " |")
    cc = res["calendar_check"]
    a(f"| 거래일 대조 | 로더 달력과 가격 봉 날짜 차이: KRX {len(cc['kr_sessions_vs_kospi_bars_diff'])}일, NYSE {len(cc['nyse_vs_us_bars_diff'])}일 |")
    a(f"| 판정 자격에서 뺀 창 | {', '.join(res['excluded_from_main'])}(96시간 초과, §2.1) |")
    a(f"| 장애 여부 모름 | 본판 {res['outage_unknown_in_main']}창(로더 기준선이 짧은 앞 창). 값은 그대로 썼다 |")
    fz = res["frozen_sha256"]
    a(f"| 동결 문서 SHA-256 | HYPOTHESES.md {fz['HYPOTHESES.md'][:8]}…, 부록 1 {fz['HYPOTHESES_addendum1.md'][:8]}…, lexicon_draft {fz['lexicon_draft.md'][:8]}… (§12.5 지문과 {'같음' if res['frozen_match'] else '다름 — 확인 필요'}) |")
    a(f"| 입력 | feat_text.parquet SHA-256 {res['inputs']['feat_text_sha256'][:16]}… |")
    a("")
    a("## 6. 못 한 것과 이유")
    a("")
    a("| 항목 | 이유 | 처리 |")
    a("|---|---|---|")
    a("| VKOSPI 1일 변화(M0 변동·거래) | yfinance ^VKOSPI 0행. 저장소 수동 CSV는 날마다 이어진 값이 아니라 기사에서 옮긴 몇몇 날의 값이다 | M0에서 뺐다 |")
    zv = ", ".join(res["kospi_zero_volume_days"]) or "없음"
    a(f"| KOSPI 거래대금 | yfinance ^KS11에는 원화 거래대금이 없다 | 거래량(주식 수) 이상치로 대신했다. 거래량 0인 날({zv})은 결측으로 두었다 |")
    a("| CPI·고용 일정 더미 | BLS 일정 페이지 403(가격 전 점검), 이번 단계 웹 검색 금지 | 뺐다 |")
    a("| MSCI 리밸런싱 더미 | 날짜 원천 없음 | 뺐다 |")
    sd = ci["earn_in_range"].get("005930.KS")
    a(f"| 삼전 잠정실적 일 | yfinance에 잠정 발표일이 없다 | 확정 실적일({', '.join(sd) if isinstance(sd, list) else sd})로 대신했다 |")
    a("| 확장 OLS(최소 120창, 월 1회 재적합) | 창이 60개뿐 | 표본 안 OLS로 R²만 냈다. X⊥ 잔차는 만들지 않았다 |")
    a("| 2026 앞 구간(1~5월) | 포럼 자료가 7/13부터 | 지금 창 전체 하나로만 냈다 |")
    a("| 원형 이동 p < 0.05 | 이동 개수가 19개(급락 제외판 13개)라 최소 p가 0.05(0.071) | HAC(5) p로 위치를 보고, 원형 이동 p는 참고로만 둔다 |")
    a("")
    a("## 7. 읽는 법")
    a("")
    a("- 이 값은 '감정 사전이 이미 본 가격을 따라 움직이는가'를 보는 관문이다. 매매 칸이 아니고, 앞날 수익과는 잇지 않았다.")
    a("- 미리보기에서 기준을 넘어도 관문 통과로 적지 않는다. 2026 전체 창으로 같은 코드를 다시 돌린 값만 판정에 쓴다.")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
