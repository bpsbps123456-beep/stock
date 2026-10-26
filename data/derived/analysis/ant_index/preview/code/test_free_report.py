"""test_free.json → test_free.md (숫자는 모두 json에서 가져온다). 실행: py test_free_report.py"""
from __future__ import annotations

import copy
import json
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / "results"


def f(v, d=3):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "예" if v else "아니오"
    if isinstance(v, int):
        return str(v)
    return f"{v:+.{d}f}" if isinstance(v, float) and d >= 2 and abs(v) < 10 else f"{v:.{d}f}"


def pf(v):
    return "—" if v is None else f"{v:.3f}"


def pct(v):
    return "휴장" if v is None else f"{v * 100:+.2f}%"


def audit_section(a, cells: dict) -> None:
    """시점 감사(audit.md) 반영 절. audit_timing.json이 없으면 쓰지 않는다. 숫자는 모두 그 json에서 가져온다."""
    p = RES / "audit_timing.json"
    if not p.exists():
        return
    au = json.loads(p.read_text("utf-8"))
    X = au["X_fixed_m0"]
    m = X["t2"]["monthly"]
    cut = au["P_price_perturb"]["cuts"]["K2026-09-15 마감"]["m0_residual"]
    t1 = X["t1_placebo"]["monthly"]
    a("## 시점 감사 반영 (audit.md)")
    a("- 특징 열(A·Fight·NW·ΔTilt·LL), M0 열, 진입 시가·결과 기간은 시점 감사를 통과했다(포럼 교란·결과 교란·가격 교란·참조 재계산). "
      "자세한 숫자는 `preview/results/audit.md`.")
    a(f"- **문제 1 — M0 잔차가 앞날 가격을 쓴다.** H46·H31의 주 특징 X⊥와 모든 칸의 위약 X̂는 표본 안 OLS 한 번으로 만들어, 계수가 뒤 창의 M0 행"
      f"(진입 뒤 가격이 든 r5·r20 등)으로 정해진다. 2026-09-15 08:00 뒤에 열린 봉만 바꾸자 그 전 창의 X⊥가 H46 "
      f"{cut['H46']['insample_xperp_changed_pre']}/{cut['H46']['pre_T_windows_in_sample']}창, H31 "
      f"{cut['H31']['insample_xperp_changed_pre']}/{cut['H31']['pre_T_windows_in_sample']}창 바뀌었다(앞 자료만 쓴 판은 0창). "
      "동결 §2.2(회귀 계수는 그날까지의 자료로만)·§2.3(확장 OLS, 최소 120창)대로면 이 자료에서 X⊥는 결측이라 두 칸은 **시험 불가**다. "
      "아래 '칸별 결과'의 H46·H31 IC·위약·모양은 '표본 안 M0 설명판'으로만 읽는다.")
    a(f"- 앞 자료만 쓴 M0(D−1까지, 월 1회 재적합, 최소 {X['t2']['monthly']['H46']['min_fit']}창 — 비등록 감사판)로 다시 센 값:")
    a("")
    a("| 칸 | n | 기간 | IC(X⊥) | t | 원형 이동 p (이동 수) | 판정 p | 위약 X̂ IC (표지) | 모양(감사판) | 원래(표본 안 M0) |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    for k in ("H46", "H31"):
        c, o = m[k], X["orig"][k]
        if "p_final_one" not in c:
            a(f"| {k} | {c['n_obs']} | — | — | — | — | — | — | 계산 안 함 | IC {o['ic']:+.3f}, {o['verdict']} |")
            continue
        a(f"| {k} | {c['n_obs']} | {c['first_D']}~{c['last_D']} | {c['ic_main_xperp_partial']:+.3f} | {c['t']:+.2f} | "
          f"{pf(c['p_shift_one'])} ({c['shift_K']}) | {c['p_final_one']:.3f} | {c['placebo']['ic']:+.3f} "
          f"({'예' if c['placebo']['flag_70pct_same_sign'] else '아니오'}) | {X['shapes_monthly'][k]} | "
          f"IC {o['ic']:+.3f}, {o['verdict']} |")
    a("")
    a(f"- 감사판은 9월부터만 값이 있어(앞 30창을 적합에 씀) n이 {m['H46']['n_obs']}·{m['H31']['n_obs']}로 작고, 검출 한계 IC가 "
      f"{m['H46']['mde_ic']:.2f}·{m['H31']['mde_ic']:.2f}다. 어느 판으로도 통과 모양 칸은 0개다(결론 그대로).")
    h08 = t1["H08"]
    a(f"- T1(H03·H08)의 주 추정(HAR-K 계수)은 X⊥를 쓰지 않아 그대로다. 위약 X̂만 표본 안이었다. 앞 자료판 위약: H03은 X̂가 있는 창이 "
      f"{t1['H03']['n_with_past_xhat']}개라 계산하지 않았고, H08은 부분 r {h08['placebo_partial_r']:+.3f}(X {h08['partial_r_x_same_sample']:+.3f}, "
      f"표지 {'예' if h08['flag_70pct_same_sign'] else '아니오'}). 두 칸 모두 부호가 반대라 모양(기각 모양)은 바뀌지 않는다.")
    a("- 새 가설 후보 3(H31 X̂ 가격 낙폭 모형)은 표본 안 X̂라 실시간 신호가 아니다. 등록하려면 앞 자료만 쓴 적합으로 다시 정의해야 한다.")
    a("- **문제 2 — 설명용 시차표의 겹침.** 종가 대 종가 수익은 앞 세션 종가에서 시작한다. KOSPI +1일(그날 15:30 시작)과 SOXX 0일(그날 05:00 KST 시작)은 "
      "그날 글과 수익 구간이 겹쳐 '글이 먼저'가 아니었다. 아래 설명용 1·2를 겹치지 않는 판으로 고쳤다.")
    a("")


def main() -> int:
    d = json.loads((RES / "test_free.json").read_text("utf-8"))
    C = d["cells"]
    h03, h08, h46, h31, h35 = C["H03"], C["H08"], C["H46"], C["H31"], C["H35"]
    cr = d["corrections"]
    de = d["descriptive"]
    # 시점 감사: KOSPI +1일은 겹치지 않는 판(다음 세션 시가→종가)으로 바꾼 시차표(llc)를 평균·문장에 쓴다
    llc = copy.deepcopy(de["lead_lag"])
    clean = de.get("lead_lag_clean")
    if clean:
        for kind in ("absr", "r"):
            llc[f"kospi_{kind}"][11] = {**llc[f"kospi_{kind}"][11], "spearman": clean["kospi_lag1"][kind]["spearman"],
                                        "n": clean["kospi_lag1"][kind]["n"]}
    L = []
    a = L.append
    a("# 개미 지수 미리보기: 사전 없이 되는 칸 시험 (2026-10-18)")
    a("")
    a("> 미리보기다(부록 9). 동결 판정(앞 1~5월 같은 부호 + 뒤 단측)을 대신하지 않는다. 앞 구간 포럼 자료가 없어 '통과'는 나올 수 없고, "
      "아래 '모양'은 뒤 구간 조건만 본 것이다. 이 결과로 정의·문턱·대상을 바꾸지 않는다.")
    a("")
    audit_section(a, C)
    a("## 결론")
    a(f"- 다섯 칸 중 예상 방향으로 통과 모양을 보인 칸은 0개다. H03·H08·H46은 점추정 부호가 예상과 반대(기각 모양), "
      f"H31은 부호는 맞지만 약하고(IC {h31['ic_main_xperp_partial']:+.3f}, p {h31['p_final_one']:.3f}) "
      f"가격만으로 만든 가짜 특징이 더 크게 맞혀 메아리 위약 규칙에 걸렸다. H35는 등록 특징 값이 {h35['n_obs']}창이라 시험하지 못했다. "
      "시점 감사 뒤 읽기: H46·H31은 등록 M0(확장 OLS, 최소 120창)를 못 채워 동결 규칙상 시험 불가이고, 위 두 칸의 숫자는 앞날 가격이 계수에 "
      "섞인 표본 안 M0 설명판이다. 앞 자료만 쓴 감사판으로도 통과 모양은 없다.")
    a(f"- 가장 뚜렷한 숫자는 예상과 반대인 H08이다: 댓글 싸움 몫이 클수록 다음 장 KOSPI 변동이 작았다"
      f"(부분 상관 {h08['partial_r']:+.3f}, HAC t {h08['t_hac5']:+.2f}, n {h08['n_obs']}, 7/28~10/16). 같은 통제로 6/01부터 쓴 판(n {h08['aux']['A 뺀 판(A 결측 창 포함, 보조)']['n']}, A 없음)에서는 "
      f"{h08['aux']['A 뺀 판(A 결측 창 포함, 보조)']['partial_r']:+.3f}로 거의 0이라, 기간에 기대는 결과다. 결과를 본 뒤의 관찰이라 새 가설 후보로만 적는다.")
    a("- 하루 글 수는 '그날' KOSPI 변동과 가장 크게 같이 움직인다(Spearman "
      f"{de['lead_lag']['kospi_absr'][10]['spearman']:+.2f}). 앞뒤 시차의 상관이 비슷해(가격 먼저 −5~−1일 평균 "
      f"{sum(de['lead_lag']['kospi_absr'][k]['spearman'] for k in range(5, 10)) / 5:+.2f}, 글 먼저 +1~+5일 평균 "
      f"{sum(llc['kospi_absr'][k]['spearman'] for k in range(11, 16)) / 5:+.2f}, +1일은 겹치지 않는 시가→종가 판) 어느 쪽이 먼저인지는 가를 수 없다. "
      "방향은 떨어진 날과 그 뒤에 글이 많은 쪽이다(수익과 음의 상관).")
    a("")
    a("## 칸별 결과")
    a("| 칸 | 특징 → 대상 | n | 점추정 | t | 원형 이동 p (이동 수) | 해석 p | 메아리 위약 | 예상 부호 | 부호 | 모양 |")
    a("|---|---|---|---|---|---|---|---|---|---|---|")
    a(f"| H03 | A(rz8 글량) → KOSPI D일 ln HL, HAR-K c | {h03['n_obs']} | c {h03['estimate_c']:+.4f} (부분 r {h03['partial_r']:+.3f}) | "
      f"{h03['t_hac5']:+.2f} (HAC5) | {pf(h03['p_shift_one'])} ({h03['shift_K']}) | {pf(h03['p_analytic_one'])} | "
      f"부분 r {h03['placebo']['partial_r']:+.3f} | + | {'맞음' if h03['sign_ok'] else '반대'} | {h03['verdict_shape']} |")
    a(f"| H08 | Fight → KOSPI D일 ln HL, HAR-K + A + ln 댓글 수 + 잘림 비율 | {h08['n_obs']} | c {h08['estimate_c']:+.3f} (부분 r {h08['partial_r']:+.3f}) | "
      f"{h08['t_hac5']:+.2f} (HAC5) | {pf(h08['p_shift_one'])} ({h08['shift_K']}) | {pf(h08['p_analytic_one'])} | "
      f"부분 r {h08['placebo']['partial_r']:+.3f} | + | {'맞음' if h08['sign_ok'] else '반대'} | {h08['verdict_shape']} |")
    a(f"| H35 | NW(신참 비율, 90일 이력 rz8) → KOSPI R10 | {h35['n_obs']} | — | — | — | — | — | − | — | 시험 불가(남은 창 0) |")
    a(f"| H46 | ΔTilt → SOXX R10 IC | {h46['n_obs']} | IC {h46['ic_main_xperp_partial']:+.3f} | {h46['t']:+.2f} (Hodrick) | "
      f"{pf(h46['p_shift_one'])} ({h46['shift_K']}) | {pf(h46['p_analytic_one'])} | IC {h46['placebo']['ic']:+.3f} | − | "
      f"{'맞음' if h46['sign_ok'] else '반대'} | {h46['verdict_shape']} |")
    a(f"| H31 | LL 미국 작성자 비율 20창 변화 잔차 → SOXX MDD10 IC | {h31['n_obs']} | IC {h31['ic_main_xperp_partial']:+.3f} | "
      f"{h31['t']:+.2f} (NW10) | {pf(h31['p_shift_one'])} ({h31['shift_K']}) | {pf(h31['p_analytic_one'])} | IC {h31['placebo']['ic']:+.3f} | − | "
      f"{'맞음' if h31['sign_ok'] else '반대'} | {h31['verdict_shape']} |")
    a("")
    a("- p는 예상 방향 단측이다. 판정 p는 원형 이동 p와 해석 p 중 큰 값이다(§2.7). 원형 이동은 특징 유효 창 안에서 21~(N−21)칸 전부다.")
    a(f"- 미리보기 보정: T1 Holm(H03·H08) 조정 p = H03 {cr['T1_holm_adj']['H03']:.3f}, H08 {cr['T1_holm_adj']['H08']:.3f}. "
      f"T2 BH(H46·H31) q = H46 {cr['T2_bh_q_H46_H31']['H46']:.3f}, H31 {cr['T2_bh_q_H46_H31']['H31']:.3f}. "
      "최종 판정에서는 예측 층·감정 칸을 더해 실제 시험 칸 수로 다시 센다.")
    a("")
    a("## 칸별 근거")
    a(f"**H03 글량 → 다음 장 변동** (기간 {h03['first_D']}~{h03['last_D']}, 같은 요일 8창 예열 때문에 7/28부터)")
    a(f"- 계수 c {h03['estimate_c']:+.4f}, 부분 R² 증가 {h03['delta_R2']:.4f}(0.01 미만 = 실용 효과 작음 기준). 저장 글 수판도 c "
      f"{h03['aux']['저장 글 수판(부호 확인)']['c']:+.4f}로 같은 음의 부호다.")
    rb = h03["robust"]
    a(f"- 강건: 급락 4일 뺀 판 c {rb['crash4']['c']:+.4f}(부호가 뒤집힘), 열어 본 8일 뺀 판 {rb['opened8']['c']:+.4f}, "
      f"기여 큰 2창({', '.join(rb['top2']['dropped_win'])}) 뺀 판 {rb['top2']['c']:+.4f}. 어느 방향으로도 안정적이지 않다.")
    a(f"- 검출 한계(부분 상관) {h03['mde_partial_r']:.2f}. M0 R² {h03['placebo']['M0_R2']:.2f}(같은 이동의 우연 기대 {h03['placebo']['M0_R2_chance_mean']:.2f})라 "
      "메아리 우세 표지가 붙지만, 그중 우연 기대값만큼은 회귀 열 수(19)가 표본(50)에 비해 많아서 생기는 몫이다.")
    a(f"**H08 댓글 싸움 → 다음 장 변동** (같은 50창, A가 7/28부터라 표본이 같다)")
    ab = h08["aux"]
    a(f"- 계수 c {h08['estimate_c']:+.3f}, t {h08['t_hac5']:+.2f}, 부분 R² 증가 {h08['delta_R2']:.3f}. 예상(+)과 반대이고, 반대 방향 원형 이동 p는 "
      f"{h08['p_shift_opposite_info']:.3f}(정보용, 판정에 안 씀)이다.")
    a(f"- 강건 3개 모두 같은 음의 부호다(급락 4일 뺀 판 {h08['robust']['crash4']['c']:+.3f}, 열어 본 8일 뺀 판 {h08['robust']['opened8']['c']:+.3f}, "
      f"기여 큰 2창 뺀 판 {h08['robust']['top2']['c']:+.3f}).")
    a(f"- 보조: A를 빼고 같은 50창만 쓰면 부분 r {ab['A 뺀 판, A 있는 창만(보조)']['partial_r']:+.3f}(t {ab['A 뺀 판, A 있는 창만(보조)']['t']:+.2f}), "
      f"A를 빼고 6/01부터 쓰면(n {ab['A 뺀 판(A 결측 창 포함, 보조)']['n']}) {ab['A 뺀 판(A 결측 창 포함, 보조)']['partial_r']:+.3f}, "
      f"마감 전 정보만 쓴 잘림판 fight_pc는 {ab['마감 전 잘림판 fight_pc(보조)']['partial_r']:+.3f}(t {ab['마감 전 잘림판 fight_pc(보조)']['t']:+.2f}). "
      "음의 관계는 A 통제가 아니라 7/28 뒤 기간에서 나온다.")
    a(f"**H35 신참 비율** — 등록 특징 NW90_rz8 값이 {h35['n_obs']}창이다. 90일 이력이 8/28부터 생기고, rz8이 같은 요일 직전 8창을 더 요구한다. "
      f"비등록 30일판(NW30_rz8, 설명용) 원 IC는 {h35['explanatory_NW30_rz8']['ic_raw']:+.3f}(n {h35['explanatory_NW30_rz8']['n']}, 원형 이동 0개)로 아무 결론도 못 낸다.")
    a(f"**H46 관심의 미장 이동 → SOXX R10** (기간 {h46['first_D']}~{h46['last_D']}, 한쪽 휴장 창 5개는 이어 쓴 값이라 뺐다)")
    a(f"- 원 IC {h46['ic_raw']:+.3f}로 예상과 반대(+)로 크다. 대상 가격 통제를 넣으면 {h46['ic_x_ctrl_partial']:+.3f}, M0 잔차(X⊥)로는 "
      f"{h46['ic_main_xperp_partial']:+.3f}로 사라진다. 원 상관은 대상 가격 흐름(직전 수익·변동, NQ)으로 설명되는 몫이다.")
    a(f"- M0 R² {h46['placebo']['M0_R2']:.2f}(우연 기대 {h46['placebo']['M0_R2_chance_mean']:.2f}). X⊥ 1차 자기상관 {h46['rho1_xperp']:.2f}, "
      f"n_eff {h46['n_eff']:.1f}, 검출 한계 IC {h46['mde_ic']:.2f}라 검정력이 거의 없다. 점추정 부호가 반대라 규칙상 기각 모양이지만 0에 가깝다.")
    a(f"**H31 레버리지 롱 언급 → SOXX 낙폭** (기간 {h31['first_D']}~{h31['last_D']}, 8월 규제 시행일 ±10거래일 제외는 날짜 미확보로 못 함 → 동결 규칙상 잠정 설명용)")
    a(f"- IC {h31['ic_main_xperp_partial']:+.3f}(원 IC {h31['ic_raw']:+.3f})로 부호는 맞다. 원형 이동 p {h31['p_shift_one']:.3f}({h31['shift_K']}개 이동), "
      f"NVDL 뺀 판 {h31['aux']['NVDL 뺀 판(강건)']['ic_main']:+.3f}.")
    a(f"- 기여 큰 2창({', '.join(h31['robust']['top2']['dropped_win'])})을 빼면 {h31['robust']['top2']['ic']:+.3f}로 거의 사라진다(취약).")
    a(f"- 가격만으로 만든 X̂의 IC가 {h31['placebo']['ic']:+.3f}로 X(통제 포함 {h31['ic_x_ctrl_partial']:+.3f})보다 커서 메아리 위약 규칙에 걸린다. "
      f"다만 M0 R²가 {h31['placebo']['M0_R2']:.2f}로 우연 기대({h31['placebo']['M0_R2_chance_mean']:.2f})보다 낮아, X̂는 대부분 적합 잡음이다. "
      "가격 변수 조합이 낙폭을 맞힌 것에 가깝고, 포럼이 더한 정보라는 근거는 없다.")
    a("")
    a("## 설명용 1: 하루 글 수와 수익의 시차 상관 (판정 아님)")
    a(f"- 글 수 = KST 달력일 번호 증가({de['first']}~{de['last']}, {de['days']}일, 지운 글 포함). 그 시장 거래일만 쓴다. "
      "시차 k: 그날 글 수와 k거래일 뒤 수익의 Spearman. **k > 0이면 글이 먼저, k < 0이면 가격이 먼저**다.")
    a("- 겹침(시점 감사): 종가 대 종가 수익은 앞 세션 종가에서 시작한다. KOSPI 0일은 같은 날 장(09:00~15:30)과 장 뒤 글이 섞이고, "
      "KOSPI +1일은 그날 15:30 뒤 글이 수익 구간(밤사이 미국장·다음 날 시가 갭) 안에 있다. SOXX 0일은 그날 05:00 KST 뒤 글이 수익 구간"
      "(선물·프리장·22:30 정규장) 안에 있다. 그래서 이 세 칸은 '글이 먼저'가 아니다. 표의 KOSPI +1일 칸(†)은 겹치지 않는 판(다음 세션 "
      "시가 09:00 → 종가)이고, 평균도 그 값으로 낸다. KOSPI +2일부터와 SOXX +1일부터는 종가 대 종가 그대로 겹치지 않는다. "
      "SOXX −1일은 '간밤 미국장 → 오늘 글'이다.")
    nz = de["noise_lag0_circular"]
    a(f"- 우연 범위(0일 원형 이동 95% 분위 |ρ|): KOSPI |수익| {nz['kospi_absr']['p95_abs_spearman_lag0']:.2f}, KOSPI 수익 {nz['kospi_r']['p95_abs_spearman_lag0']:.2f}, "
      f"SOXX |수익| {nz['soxx_absr']['p95_abs_spearman_lag0']:.2f}, SOXX 수익 {nz['soxx_r']['p95_abs_spearman_lag0']:.2f}. 두 계열 모두 자기상관이 커서 |수익| 쪽 우연 범위가 넓다. 21개 시차를 함께 보므로 한두 칸이 넘는 것은 우연으로도 생긴다.")
    a("")
    a("| 시차 k | KOSPI \\|수익\\| | KOSPI 수익 | SOXX \\|수익\\| | SOXX 수익 |")
    a("|---|---|---|---|---|")
    ll0 = de["lead_lag"]
    ll = llc
    for i in range(21):
        k = ll["kospi_absr"][i]["lag"]
        mk = "†" if (clean and k == 1) else ""
        ms = " (겹침)" if k == 0 else ""
        a(f"| {k:+d}{mk} | {ll['kospi_absr'][i]['spearman']:+.2f}{ms if k == 0 else ''} | {ll['kospi_r'][i]['spearman']:+.2f}{ms if k == 0 else ''} | "
          f"{ll['soxx_absr'][i]['spearman']:+.2f}{ms} | {ll['soxx_r'][i]['spearman']:+.2f}{ms} |")
    if clean:
        a(f"- † KOSPI +1일은 다음 세션 시가→종가(n {clean['kospi_lag1']['absr']['n']}). 종가 대 종가 원값(겹침)은 |수익| "
          f"{ll0['kospi_absr'][11]['spearman']:+.2f}, 수익 {ll0['kospi_r'][11]['spearman']:+.2f}였다. SOXX +1일 칸은 원래도 겹치지 않는다.")
    a("")
    def m(col, ks):
        return sum(ll[col][k + 10]["spearman"] for k in ks) / len(ks)

    def over(col, key):
        th = nz[key]["p95_abs_spearman_lag0"]
        return [r["lag"] for r in ll[col] if abs(r["spearman"]) > th]
    a(f"- KOSPI |수익|: 0일 {ll['kospi_absr'][10]['spearman']:+.2f}가 가장 크다. 가격이 먼저인 −5~−1일 평균 {m('kospi_absr', range(-5, 0)):+.2f}, "
      f"글이 먼저인 +1~+5일 평균 {m('kospi_absr', range(1, 6)):+.2f}로 앞뒤가 비슷하다. 어느 쪽이 먼저라고 말할 수 없고, '변동이 큰 시기에 글이 많다'는 국면 공유가 주다. "
      f"우연 범위를 넘는 시차: {', '.join(f'{k:+d}' for k in over('kospi_absr', 'kospi_absr'))}.")
    a(f"- KOSPI 수익: 0일 {ll['kospi_r'][10]['spearman']:+.2f}, 가격이 먼저인 −5~−1일 평균 {m('kospi_r', range(-5, 0)):+.2f}, 글이 먼저인 +1~+5일 평균 "
      f"{m('kospi_r', range(1, 6)):+.2f}. 떨어진 날과 그 뒤에 글이 많다. 우연 범위를 넘는 시차: {', '.join(f'{k:+d}' for k in over('kospi_r', 'kospi_r'))} "
      f"(+5일 {ll['kospi_r'][15]['spearman']:+.2f}, +9일 {ll['kospi_r'][19]['spearman']:+.2f}는 21개 시차 중 몇 칸이라 우연일 수 있다).")
    a(f"- SOXX |수익|: 모든 시차가 +{min(r['spearman'] for r in ll['soxx_absr']):.2f}~+{max(r['spearman'] for r in ll['soxx_absr']):.2f}로 비슷하다(국면 공유). "
      f"SOXX 수익: −5~−1일 평균 {m('soxx_r', range(-5, 0)):+.2f}, +1~+5일 평균 {m('soxx_r', range(1, 6)):+.2f}. 미국장이 내린 뒤 글이 늘고, 글이 앞날 SOXX를 맞히는 모양은 없다.")
    a("")
    a("## 설명용 2: 글 수 상위 10일과 KOSPI 수익 (판정 아님)")
    a(f"- 하루 글 수 중앙값 {de['median_d_no']:.0f}. 수익은 KOSPI 종가 대 종가다. 다음 거래일 종가 대 종가는 그날 15:30 뒤 글과 구간이 겹쳐"
      "(시점 감사), 겹치지 않는 다음 거래일 시가→종가를 함께 적는다.")
    a("")
    a("| 날짜 | 요일 | 글 수(번호 증가) | 그날 KOSPI | 다음 거래일 | 다음 거래일 KOSPI(종가 대 종가, 겹침) | 다음 거래일 시가→종가 |")
    a("|---|---|---|---|---|---|---|")
    for r in de["top10"]:
        a(f"| {r['date']} | {r['weekday']} | {r['d_no']:,} | {pct(r['kospi_r_that_day'])} | {r['next_session']} | "
          f"{pct(r['kospi_r_next_session'])} | {pct(r.get('kospi_oc_next_session'))} |")
    up_next = sum(1 for r in de["top10"] if (r["kospi_r_next_session"] or 0) > 0)
    up_oc = sum(1 for r in de["top10"] if (r.get("kospi_oc_next_session") or 0) > 0)
    dn_day = sum(1 for r in de["top10"] if (r["kospi_r_that_day"] or 0) < 0)
    a("")
    a(f"- 10일 중 {dn_day}일은 그날 KOSPI가 내린 날이다. 다음 거래일은 겹치지 않는 시가→종가로 {up_oc}일 올랐고 {10 - up_oc}일 내렸다"
      f"(종가 대 종가로는 {up_next}일·{10 - up_next}일). 7/28~7/31(급락·바닥·반등)이 4칸을 차지하고, "
      "이 구간은 동결 문서가 '이미 알려진 사례'로 적은 날이다.")
    a("")
    a("## 결과 보기 전에 정한 해석")
    a("| 항목 | 정한 해석 | 이유 |")
    a("|---|---|---|")
    a("| 표본 | 판정 자격 창(국내 96시간 초과 1창, 미국 장애 2창 제외) 중 특징·결과·통제가 모두 있는 창. T1은 위약과 같은 표본을 쓰려고 M0 결측 창도 뺀다 | 덜 넣는 쪽 |")
    a(r"| HAR-K | ln\|r(D−1)\|, ln RV5, ln RV22, min(r(D−1),0), ln\|SOX 직전\|, ln\|EWY 직전\|, 거래량 이상치(거래대금 대신), \|g(D)\|, 요일 4, 일정 더미(FOMC·실적·국내 옵션 만기·미국 월물 만기), 결번율. 전일 VKOSPI는 원천 없음. 1(또는 0)이 2개 미만인 더미와 랭크를 늘리지 않는 열은 뺀다 | §3 F1, h01 규칙 |")
    a("| M0 적합 | 동결 확장 OLS(최소 120창)는 창이 모자라 못 한다. 표본 안 OLS 한 번으로 X⊥(잔차)·X̂(적합값)을 만든다. 실시간으로 쓸 수 있는 신호가 아니다. M0 R²와 같은 원형 이동의 우연 R²를 함께 낸다 | 미리보기 대체 |")
    a("| IC | X⊥와 목표의 순위 부분 상관(대상 r·r5·r20·RV20·g 등 §3 통제, 순위로 잔차화) | §2.3 '주 특징은 X⊥, 판정 회귀 통제' |")
    a("| t | R 목표: Hodrick(1992) 1B(1기간 = 그 세션 종가 대 종가, 분산 계산에만). MDD 목표: 합으로 나뉘지 않아 NW HAC(10). 모의 실험(n 60, h 10, AR 0.8 특징)에서 한쪽 5% 기각률이 Hodrick 4.8%, NW(10) 15.5%라 MDD 칸은 원형 이동 p가 판정 p를 정한다 | §2.0 |")
    a("| 원형 이동 | 특징 유효 창(시간순) 안에서 특징열만 21~(N−21)칸 전부 돌리고 M0 잔차화·통제를 매번 다시 한다. p = (1 + 예상 방향으로 관측 이상인 이동 수) ÷ (1 + 이동 수) | §2.7 ① |")
    a("| 위약 | X̂로 같은 시험. X(대상 통제만 넣은 판)와 같은 부호로 70% 이상이고 X가 보정 기준을 못 넘으면 '기각 모양(메아리)' | §2.3 |")
    a("| 강건 | 급락 4일·열어 본 8일과 시간이 겹치는 창을 뺀 판, 예상 방향 기여 큰 2창을 뺀 판 | §2.7, h01 주 읽기 |")
    a(r"| 모양 | 기각 모양 = 점추정 부호 반대 또는 90% 상한 < 기대 효과의 절반(IC 0.05, 부분 r 0.075). IC의 SE는 1/√n_eff, 부분 r의 SE는 \|r\|/\|t\| | §2.7 |")
    a("| H46 | 한쪽 휴장 창(이어 쓴 값, h46_valid_win 거짓)은 표본에서 뺀다 | §2.4 '기준에서 뺀다', 덜 넣는 쪽 |")
    a("| H31 | M0(미국)에 SOX r5·r20·RV20·DD20이 들어 있어 X⊥가 동결 'SOX 잔차화'를 포함한다 | 같은 잔차 |")
    a("")
    a("## 새 가설 후보 (판정 안 함, 결과를 본 뒤의 관찰)")
    a("- 동결 정의·문턱·대상은 바꾸지 않는다. 아래는 새 id로 등록한 뒤 2026-10-19 이후 실시간 자료로만 시험할 후보다.")
    a(f"- 후보 1(H08의 반대): 댓글 싸움 몫이 클수록 다음 장 KOSPI 변동이 작다. 7/28~10/16 부분 r {h08['partial_r']:+.3f}, 6/01부터는 "
      f"{h08['aux']['A 뺀 판(A 결측 창 포함, 보조)']['partial_r']:+.3f}. 메커니즘 후보: 급변일에는 한쪽으로 쏠려 다툴 일이 적고, 다툼은 방향이 정해지지 않은 장에 몰린다.")
    a(f"- 후보 2(글 폭주 뒤 되돌림): 하루 글 수와 5·9거래일 뒤 KOSPI 수익이 양의 상관이다(+5일 {de['lead_lag']['kospi_r'][15]['spearman']:+.2f}, "
      f"+9일 {de['lead_lag']['kospi_r'][19]['spearman']:+.2f}). 동결 H29(설명용, ±3% 날 20개 쌓이면 한 번 판정)와 겹치므로 H29 판정 때 함께 본다.")
    a(f"- 후보 3(포럼 밖): H31 위약에서 미국 M0 가격 조합(X̂)이 SOXX MDD10과 IC {h31['placebo']['ic']:+.3f}였다. 포럼 가설이 아니라 가격만의 낙폭 모형 후보다.")
    a("")
    a("## 한계")
    a("- 앞 구간(1~5월)이 없어 동결 '통과'는 원리상 나올 수 없다. 최종 판정은 수집 완료 뒤 한 번이고, 그 뒤 구간은 이 미리보기와 겹친다('뒤 구간 미리 봄').")
    a(f"- 표본이 작다: T1 50창(검출 한계 부분 r {h03['mde_partial_r']:.2f}), H46 n_eff {h46['n_eff']:.0f}(검출 한계 IC {h46['mde_ic']:.2f}), H31 n_eff {h31['n_eff']:.0f}(검출 한계 {h31['mde_ic']:.2f}). "
      "동결 문서의 현실적 기대 IC 0.05~0.15보다 검출 한계가 훨씬 크다.")
    a(f"- 6/01~10/16 KOSPI 하루 수익 절대값 평균이 {de['kospi_mean_abs_r_0601_1016'] * 100:.2f}%({de['kospi_n_sessions_0601_1016']}거래일)인 매우 큰 변동 국면이다. "
      "급락 4일을 빼면 부호가 뒤집히는 칸(H03·H46)이 있다.")
    a("- 미국 창 결과는 10/02(H46)·10/05(H31) 진입분까지만 있다(10일 경로가 10/16에서 끝남).")
    a("")
    a("## 파일")
    a("- 결과: `preview/results/test_free.json`(칸별 모든 수치·강건·위약·보조판, 설명용 표 전부), 이 문서")
    a("- 코드: `preview/code/test_free.py`(계산), `preview/code/test_free_report.py`(이 문서)")
    (RES / "test_free.md").write_text("\n".join(L) + "\n", "utf-8")
    print("written", len(L), "lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
