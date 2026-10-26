"""test_pred.json → test_pred.md. 숫자는 모두 json·parquet에서 읽어 쓴다(손으로 옮기지 않는다).
실행: py test_pred_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"


def f(v, d=3, sign=True):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:+.{d}f}" if sign else f"{v:.{d}f}"


def pct(v, d=2):
    return "—" if v is None else f"{100 * v:+.{d}f}%"


def audit_section(a) -> None:
    """시점 감사(audit.md) 반영 절. audit_timing.json이 없으면 쓰지 않는다. 숫자는 모두 그 json에서 가져온다."""
    p = RES / "audit_timing.json"
    if not p.exists():
        return
    au = json.loads(p.read_text("utf-8"))
    X = au["X_fixed_m0"]
    m = X["t2"]["monthly"]
    t1 = X["t1_placebo"]["monthly"]
    cut = au["P_price_perturb"]["cuts"]["K2026-09-15 마감"]["m0_residual"]
    lp = au["L_label"]
    a("## 시점 감사 반영 (audit.md)")
    a(f"- 예측 라벨 표본·특징은 시점 감사를 통과했다: 층은 창 × 글/댓글뿐이고(창마다 글 {lp['per_window']['post_counts'][0]}·댓글 "
      f"{lp['per_window']['comment_counts'][0]}, 같은 시드로 다시 뽑으면 {lp['reproduce']['saved']}단위가 그대로), 뽑기에 가격·결과·판정 자격 열을 "
      f"쓰지 않는다. 마감 뒤 창의 라벨을 바꾸고 30%를 지워도 마감 전 창의 P̄·P_share·P_strong·P_sd·rz8과 주 창 값이 그대로다. "
      "자세한 숫자는 `preview/results/audit.md`.")
    a(f"- **문제 — M0 잔차가 앞날 가격을 쓴다.** H66·H68의 주 특징 X⊥와 모든 칸의 위약 X̂는 표본 안 OLS 한 번으로 만들어, 계수가 뒤 창의 "
      f"M0 행(진입 뒤 가격이 든 r5·r20 등)으로 정해진다. 2026-09-15 08:00 뒤에 열린 봉만 바꾸자 그 전 창의 X⊥가 H66 "
      f"{cut['H66']['insample_xperp_changed_pre']}/{cut['H66']['pre_T_windows_in_sample']}창 바뀌었다(앞 자료만 쓴 판은 0창). "
      "동결 §2.2·§2.3(확장 OLS D−1까지, 최소 120창)대로면 X⊥는 결측이라 H66은 **시험 불가**다. 아래 H66·H68 IC·위약·모양과 "
      "H70 위약 표지는 '표본 안 M0 설명판'으로만 읽는다.")
    a(f"- 앞 자료만 쓴 M0(월 1회 재적합, 최소 {m['H66']['min_fit']}창 — 비등록 감사판)로 다시 센 값:")
    a("")
    a("| 칸 | n | 기간 | IC(X⊥) | t | 원형 이동 p (이동 수) | 판정 p | 위약 X̂ IC (표지) | 모양(감사판) | 원래(표본 안 M0) |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    for k in ("H66", "H68"):
        c, o = m[k], X["orig"][k]
        a(f"| {k} | {c['n_obs']} | {c['first_D']}~{c['last_D']} | {f(c['ic_main_xperp_partial'])} | {f(c['t'], 2)} | "
          f"{f(c['p_shift_one'], 3, False)} ({c['shift_K']}) | {f(c['p_final_one'], 3, False)} | {f(c['placebo']['ic'])} "
          f"({'예' if c['placebo']['flag_70pct_same_sign'] else '아니오'}) | {X['shapes_monthly'][k]}{' (설명용)' if k == 'H68' else ''} | "
          f"IC {f(o['ic'])}, {o['verdict']} |")
    a("")
    a(f"- 감사판은 9월부터만 값이 있어 n이 {m['H66']['n_obs']}·{m['H68']['n_obs']}이고, 검출 한계 IC가 {f(m['H66']['mde_ic'], 2, False)}·"
      f"{f(m['H68']['mde_ic'], 2, False)}다. H66 점추정은 예상과 반대(+)로 바뀌었다. 어느 판으로도 통과 모양은 없다.")
    a(f"- H70: 주 추정(HAR-K 계수)은 X⊥를 쓰지 않아 그대로다. 앞 자료만 쓴 위약 X̂는 값이 있는 창이 {t1['H70']['n_with_past_xhat']}개라 "
      f"계산할 수 없다 → 위약 표지 없이 읽으면 모양은 '{X['shapes_monthly']['H70']}'이다. H67 설명판 위약도 앞 자료 X̂ 창이 "
      f"{t1['H67_lite']['n_with_past_xhat']}개뿐이다.")
    h66 = json.loads((RES / "test_pred.json").read_text("utf-8"))["cells"]["H66"]
    a(f"- 결론의 'H66 반대 신호의 대부분은 지난 가격 흐름으로 설명된다'는 표본 안 부분 상관 분해(원 {f(h66['ic_raw'], 2)} → 통제 "
      f"{f(h66['ic_x_ctrl_partial'], 2)} → M0 잔차 {f(h66['ic_main_xperp_partial'], 2)})로는 "
      "그대로 성립하는 서술이지만, 판정 모양의 근거로는 쓰지 않는다. 새 가설 후보 1(X̂ 가격 반전)은 표본 안 X̂라 실시간 신호가 아니다 "
      f"(앞 자료판 X̂의 IC는 {f(m['H66']['placebo']['ic'])}, n {m['H66']['n_obs']}).")
    a("")


def main() -> int:
    J = json.loads((RES / "test_pred.json").read_text("utf-8"))
    C, D, K = J["cells"], J["descriptive"], J["corrections"]
    FW = pd.read_parquet(RES / "pred_windows.parquet")
    meta = J["features"]
    pc = {int(k): v for k, v in meta["inputs"]["pred_counts_units"].items()}
    n_used = meta["inputs"]["units_used"]
    n_predu = sum(v for k, v in pc.items() if k > 0)
    ok = FW["n_pred"] >= 2
    se_med = float(np.median(FW.loc[ok, "P_sd_raw"] / np.sqrt(FW.loc[ok, "n_pred"])))
    h66, h70, h67, h68 = C["H66"], C["H70"], C["H67"], C["H68"]
    lite = h67["explanatory_lite"]
    fol = {(r["feature"], r["col"]): r for r in D["follow"]["rows"]}
    raw = {(r["feature"], r["target"]): r for r in D["raw_ic"]["rows"]}
    st = D["stages"]
    srow = {r["stage"]: r for r in st["rows"]}
    wk = D["weekly"]
    L = []
    a = L.append
    a("# 개미 지수 미리보기: 예측 층(군중 예측) 시험 (2026-10-18)")
    a("")
    a("> 미리보기다(부록 9). 동결 판정(앞 1~5월 같은 부호 + 뒤 단측)을 대신하지 않는다. 예측 라벨은 2026-07-01 이후 창만 있어 "
      "앞 구간이 없고, 아래 '모양'은 뒤 구간 조건만 본 것이다. 이 결과로 정의·문턱·대상을 바꾸지 않는다.")
    a("")
    audit_section(a)
    a("## 결론")
    r1, r5, r20 = fol[("P_bar", "m0_kospi_r")], fol[("P_bar", "m0_kospi_r5")], fol[("P_bar", "m0_kospi_r20")]
    a(f"- **군중 예측은 가격을 따라간다.** 창 마감 전 KOSPI 지난 1일 수익과 P̄의 Spearman {f(r1['spearman'], 2)}"
      f"(우연 범위 {f(r1['p95_abs_spearman_shift'], 2, False)}), 지난 5일 {f(r5['spearman'], 2)}"
      f"(우연 {f(r5['p95_abs_spearman_shift'], 2, False)}), 지난 20일 {f(r20['spearman'], 2)}"
      f"(우연 {f(r20['p95_abs_spearman_shift'], 2, False)}, 우연 안). 오른 다음 날엔 상승 예측 쪽으로, 내린 다음 날엔 하락 예측 쪽으로 기운다.")
    rr5 = raw[("P_bar", "fut_kospi_R5")]
    a(f"- **H66 군중 예측의 반대: 기각 모양(메아리).** 통제 없이 보면 P̄와 앞 5일 KOSPI 수익의 Spearman이 {f(rr5['spearman'], 2)}"
      f"(n {rr5['n']}, 원형 이동 양측 p {f(rr5['p_shift_two_sided'], 3, False)})로 '다들 오른다고 하면 빠진다' 모양이다. "
      f"그러나 가격 메아리(M0)를 빼면 IC {f(h66['ic_main_xperp_partial'], 3)}(t {f(h66['t'], 2)}, 원형 이동 p "
      f"{f(h66['p_shift_one'], 3, False)})로 줄고, 가격만으로 만든 가짜 예측(X̂)의 IC가 {f(h66['placebo']['ic'], 3)}로 같은 통제를 넣은 "
      f"실제 P̄({f(h66['ic_x_ctrl_partial'], 3)})의 {100 * h66['placebo']['ratio_vs_x_ctrl']:.0f}%다. 반대 신호의 대부분은 지난 가격 흐름으로 설명된다. "
      "시점 감사 뒤 읽기: 등록 M0(확장 OLS, 최소 120창)를 못 채워 동결 규칙상 시험 불가이고, M0 잔차·X̂ 숫자는 앞날 가격이 계수에 섞인 "
      "표본 안 M0 설명판이다. 앞 자료만 쓴 감사판으로도 통과 모양은 없다.")
    aux70 = list(h70["aux"].values())[0]
    rhl = raw[("P_share", "fut_kospi_lnHL")]
    a(f"- **H70 예측 몫 → 다음 장 변동: 규칙상 기각 모양(메아리 위약), 위약 X̂가 거의 적합 잡음이라 실제로는 판정 불가에 가깝다.** 원 상관은 {f(rhl['spearman'], 2)}"
      f"(n {rhl['n']}, p {f(rhl['p_shift_two_sided'], 3, False)})로 크지만, 지난 변동을 통제한 HAR-K에서 등록판(rz8, n {h70['n_obs']}) "
      f"부분 상관 {f(h70['partial_r'], 3)}(t {f(h70['t_hac5'], 2)}), 원값 보조판(n {aux70['n']}) {f(aux70['partial_r'], 3)}"
      f"(원형 이동 p {f(aux70['p_shift_one'], 3, False)})로 줄어든다. 변동이 큰 시기에 예측 글의 몫이 클 뿐, 지난 변동이 알려 주는 것 이상을 더하지는 않았다.")
    a(f"- **H67 예측 분산 → 변동: 시험 불가.** 등록 특징(P_sd rz8) 값이 0창이다. 비등록 설명판은 부분 상관 {f(lite['partial_r'], 3)}"
      f"(원형 이동 p {f(lite['p_shift_one'], 3, False)}, n {lite['n_obs']})로 아무 결론도 못 낸다. "
      f"**H68 강한 하락 예측(설명용)**: IC {f(h68['ic_main_xperp_partial'], 3)}(p {f(h68['p_shift_one'], 3, False)}), 위약 표지가 붙는다.")
    s1, s5 = srow[1], srow[st["k"]]
    a(f"- 단계표(설명용): 가장 비관적인 1단계 뒤 5일 KOSPI 평균 {pct(s1['R5_mean'])}({s1['R5_n']}창), 가장 낙관적인 "
      f"{st['k']}단계 뒤 {pct(s5['R5_mean'])}({s5['R5_n']}창). 가격 통제가 없는 표라 H66의 원 상관과 같은 모양이고, 메아리를 빼면 약해진다(H66). 매매 규칙 근거가 아니다.")
    a("")
    a("## 칸별 결과")
    a("| 칸 | 특징 → 대상 | n | 점추정 | t | 원형 이동 p (이동 수) | 해석 p | 메아리 위약 | 예상 부호 | 부호 | 모양 |")
    a("|---|---|---|---|---|---|---|---|---|---|---|")
    a(f"| H66 | P̄의 M0 잔차(미리보기 대체) → KOSPI R5 IC | {h66['n_obs']} | IC {f(h66['ic_main_xperp_partial'])} | "
      f"{f(h66['t'], 2)} (Hodrick) | {f(h66['p_shift_one'], 3, False)} ({h66['shift_K']}) | {f(h66['p_analytic_one'], 3, False)} | "
      f"IC {f(h66['placebo']['ic'])} | − | {'맞음' if h66['sign_ok'] else '반대'} | {h66['verdict_shape']} |")
    a(f"| H67 | P_sd rz8 → KOSPI D일 ln HL, HAR-K | 0 | — | — | — | — | — | + | — | 시험 불가(값 0창) |")
    a(f"| H67 설명판 | P_sd_lite 원값(비등록) → 같은 HAR-K | {lite['n_obs']} | c {f(lite['estimate_c'])} (부분 r "
      f"{f(lite['partial_r'])}) | {f(lite['t_hac5'], 2)} (HAC5) | {f(lite['p_shift_one'], 3, False)} ({lite['shift_K']}) | "
      f"{f(lite['p_analytic_one'], 3, False)} | 부분 r {f(lite['placebo']['partial_r'])} | + | "
      f"{'맞음' if lite['sign_ok'] else '반대'} | 설명용({lite['shape_info_only']}) |")
    a(f"| H70 | P_share rz8 → KOSPI D일 ln HL, HAR-K | {h70['n_obs']} | c {f(h70['estimate_c'])} (부분 r "
      f"{f(h70['partial_r'])}) | {f(h70['t_hac5'], 2)} (HAC5) | 없음 (0) | {f(h70['p_analytic_one'], 3, False)} | 부분 r "
      f"{f(h70['placebo']['partial_r'])} | + | {'맞음' if h70['sign_ok'] else '반대'} | {h70['verdict_shape']} |")
    a(f"| H68 | P_strong1의 M0 잔차 → KOSPI R10 IC | {h68['n_obs']} | IC {f(h68['ic_main_xperp_partial'])} | "
      f"{f(h68['t'], 2)} (Hodrick) | {f(h68['p_shift_one'], 3, False)} ({h68['shift_K']}) | {f(h68['p_analytic_one'], 3, False)} | "
      f"IC {f(h68['placebo']['ic'])} | + | {'맞음' if h68['sign_ok'] else '반대'} | 설명용({h68['shape_info_only']}) |")
    a("")
    a("- p는 예상 방향 단측이다. 판정 p는 원형 이동 p와 해석 p 중 큰 값이다(§2.7). 원형 이동은 특징 유효 창 안에서 21~(N−21)칸 전부다.")
    sa = h70["shift_all_info"]
    a(f"- H70은 특징 창이 {h70['n_feature_windows']}개라 이 범위가 비어 원형 이동 p가 없다(판정 p = 해석 p). 정보용으로 1~(N−1)칸 "
      f"전부({sa['K']}개)를 돌리면 p {f(sa['p_one'], 3, False)}다.")
    t1, t2 = K["T1_holm_with_test_free"], K["T2_bh_with_test_free"]
    a(f"- 미리보기 보정(사전 없는 칸과 합침): T1 Holm(H03·H08·H70) 조정 p = H70 {f(t1['H70'], 3, False)}. "
      f"T2 BH(H46·H31·H66) q = H66 {f(t2['H66'], 3, False)}(H31 빼면 {f(K['T2_bh_with_test_free_noH31']['H66'], 3, False)}). "
      f"H67은 시험 불가라 넣지 않았다. 최종 판정에서는 실제 시험 칸 수로 다시 센다.")
    a("")
    a("## 칸별 근거")
    pl = h66["placebo"]
    a(f"**H66 군중 예측의 반대** (기간 {h66['first_D']}~{h66['last_D']} 진입, 판정 자격 창 중 M0·결과가 다 있는 {h66['n_obs']}창)")
    a(f"- 원 IC {f(h66['ic_raw'])} → 대상 가격 통제만 넣으면 {f(h66['ic_x_ctrl_partial'])} → M0 잔차(X⊥) {f(h66['ic_main_xperp_partial'])}. "
      f"단계마다 줄어든다.")
    a(f"- P̄의 M0 R² {f(pl['M0_R2'], 2, False)}(같은 원형 이동의 우연 기대 {f(pl['M0_R2_chance_mean'], 2, False)}): 우연보다 뚜렷하게 "
      f"지난 가격으로 설명된다(메아리 우세 표지, R² ≥ 0.5).")
    rb = h66["robust"]
    a(f"- 강건: 급락 4일 뺀 판 {f(rb['crash4']['ic'])}, 열어 본 8일 뺀 판 {f(rb['opened8']['ic'])}, 기여 큰 2창"
      f"({', '.join(rb['top2']['dropped_win'])}) 뺀 판 {f(rb['top2']['ic'])}. 부호는 안정적이지만 크기가 검출 한계(IC "
      f"{f(h66['mde_ic'], 2, False)})보다 훨씬 작다.")
    ax = h66["aux"]
    for nm, v in ax.items():
        a(f"- 보조 {nm}: IC {f(v['ic_main'])}(원 IC {f(v['ic_raw'])}, 원형 이동 p {f(v['p_shift_one'], 3, False)}, n {v['n']}).")
    p7 = h70["placebo"]
    a(f"**H70 예측 몫 → 다음 장 변동** (기간 {h70['first_D']}~{h70['last_D']}, 같은 요일 8창 예열 때문에 8/26부터)")
    a(f"- 계수 c {f(h70['estimate_c'])}, 부분 R² 증가 {f(h70['delta_R2'], 4, False)}, 자유도 {h70['df']}(HAR-K 열 "
      f"{len(h70['controls_used'])}개 + 특징 + 상수 대비 창 {h70['n_obs']}개), 검출 한계(부분 상관) {f(h70['mde_partial_r'], 2, False)}.")
    a(f"- 위약 표지는 X̂ 부분 r {f(p7['partial_r'])}가 X의 {f(p7['ratio_partial_r'], 1, False)}배라 붙었다. 다만 M0 R² "
      f"{f(p7['M0_R2'], 2, False)}가 1~(N−1)칸 이동의 우연 기대 {f(sa['M0_R2_chance_mean'], 2, False)}와 같고 조정 R²가 "
      f"{f(p7['M0_adjR2'], 2)}라, X̂는 거의 적합 잡음이다.")
    rb = h70["robust"]
    a(f"- 강건: 급락 4일 뺀 판 c {f(rb['crash4']['c'])}, 열어 본 8일 뺀 판 {f(rb['opened8']['c'])}(n {rb['opened8']['n']}), 기여 큰 2창"
      f"({', '.join(rb['top2']['dropped_win'])}) 뺀 판 {f(rb['top2']['c'])}(부호 반대). 두 창에 기댄다.")
    a(f"- 보조(원값, rz8 없음, 7/01부터 n {aux70['n']}): c {f(aux70['c'])}, t {f(aux70['t'], 2)}, 부분 r {f(aux70['partial_r'])}, "
      f"원형 이동 p {f(aux70['p_shift_one'], 3, False)}({aux70['shift_K']}개).")
    a(f"**H67 예측 분산** — {h67['note']}. 비등록 설명판(풀 직전 창 10개 이상, 예측 쌍 5개 이상, rz8 없음): {lite['first_D']}~"
      f"{lite['last_D']}, n {lite['n_obs']}, 부분 r {f(lite['partial_r'])}, 급락 4일 뺀 판 c 비 {f(lite['robust']['crash4']['ratio'], 2, False)}, "
      f"기여 큰 2창 뺀 판 비 {f(lite['robust']['top2']['ratio'], 2, False)}(취약).")
    a(f"**H68 강한 하락 예측(설명용)** — 강한 하락(1) 예측이 하나라도 있는 창은 {h68['n_windows_P_strong1_pos']}창뿐이다. "
      f"IC {f(h68['ic_main_xperp_partial'])}(원 {f(h68['ic_raw'])}), 기여 큰 2창 뺀 판 {f(h68['robust']['top2']['ic'])}. "
      f"X̂ IC {f(h68['placebo']['ic'])}로 위약 표지가 붙는다.")
    a("")
    a("## 설명용 1: 군중 예측은 가격을 따라가는가 (판정 아님)")
    a(f"- 판정 자격 창({D['follow']['first_D']}~{D['follow']['last_D']}). 지난 수익은 창 마감(D 07:59) 전에 끝난 세션까지다. "
      f"우연 범위 = 특징열을 21~(N−21)칸 원형 이동한 |Spearman| 95% 분위.")
    a("")
    a("| 특징 | 지난 수익 | n | Spearman | Pearson | 우연 범위 |")
    a("|---|---|---|---|---|---|")
    for r in D["follow"]["rows"]:
        nm = "P̄(군중 예측)" if r["feature"] == "P_bar" else "P_share(예측 몫)"
        a(f"| {nm} | {r['past']} | {r['n']} | {f(r['spearman'], 2)} | {f(r['pearson'], 2)} | "
          f"{f(r['p95_abs_spearman_shift'], 2, False)} |")
    a("")
    a("## 설명용 2: 통제 없는 원 상관 (판정 아님)")
    a("| 특징 | 대상 | n | Spearman | 원형 이동 양측 p (이동 수) |")
    a("|---|---|---|---|---|")
    lab = {"fut_kospi_R1": "다음 장 KOSPI 수익(시가→종가)", "fut_kospi_R5": "앞 5일 KOSPI 수익", "fut_kospi_R10": "앞 10일 KOSPI 수익",
           "fut_kospi_lnHL": "다음 장 KOSPI ln(고가÷저가)"}
    for r in D["raw_ic"]["rows"]:
        nm = "P̄" if r["feature"] == "P_bar" else "P_share"
        a(f"| {nm} | {lab[r['target']]} | {r['n']} | {f(r['spearman'], 2)} | {f(r['p_shift_two_sided'], 3, False)} ({r['shift_K']}) |")
    a("- 위 칸별 결과와 견주면: 원 상관이 큰 두 줄(P̄ → 5일, P_share → 다음 장 변동)이 가격 통제 뒤 모두 작아진다.")
    a("")
    a("## 설명용 3: 군중 예측 단계표 미리보기 (판정 아님)")
    a(f"- {st['rule']}.")
    a(f"- 단계 수: {st['reason']}. 단계가 매겨진 창은 {st['first_staged_D']}~{st['last_staged_D']}다.")
    a("")
    a("| 단계 | 창 | P̄ 범위 | 1일 평균 | 1일 중앙값 | 1일 n | 5일 평균 | 5일 중앙값 | 5일 n | 10일 평균 | 10일 중앙값 | 10일 n |")
    a("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in st["rows"]:
        a(f"| {r['stage']} | {r['n_windows']} | {f(r['P_bar_min'], 2)}~{f(r['P_bar_max'], 2)} | {pct(r['R1_mean'])} | "
          f"{pct(r['R1_median'])} | {r['R1_n']} | {pct(r['R5_mean'])} | {pct(r['R5_median'])} | {r['R5_n']} | "
          f"{pct(r['R10_mean'])} | {pct(r['R10_median'])} | {r['R10_n']} |")
    b = st["all_staged"]
    a(f"| 전체 | {st['n_staged']} | | {pct(b['R1']['mean'])} | {pct(b['R1']['median'])} | {b['R1']['n']} | {pct(b['R5']['mean'])} | "
      f"{pct(b['R5']['median'])} | {b['R5']['n']} | {pct(b['R10']['mean'])} | {pct(b['R10']['median'])} | {b['R10']['n']} |")
    sp = st["stage_spearman"]
    ns = [r[f"R{h}_n"] for r in st["rows"] for h in (1, 5, 10)]
    a(f"- 단계와 수익의 Spearman: 1일 {f(sp['R1']['spearman_stage'], 2)}, 5일 {f(sp['R5']['spearman_stage'], 2)}, 10일 "
      f"{f(sp['R10']['spearman_stage'], 2)}. 칸마다 {min(ns)}~{max(ns)}창이고 5·10일은 겹치는 목표라 몇 창이 전체를 움직인다.")
    a("")
    a("## 설명용 4: 주 단위 군중 예측과 다음 주·2주 수익 (판정 아님)")
    a(f"- {wk['rule']}.")
    a("")
    a("| 주 마감(월 08:00) | 표본 단위 | 예측 쌍 | P̄ | P_share | KOSPI 다음 주 | KOSPI 2주 | SOXX 다음 주 | SOXX 2주 | 비고 |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    for r in wk["rows"]:
        a(f"| {r['week_close'][:10]} | {r['n_units']} | {r['n_pred']} | {f(r['P_bar'], 2)} | {f(r['P_share'], 3, False)} | "
          f"{pct(r['kospi_R1w'])} | {pct(r['kospi_R2w'])} | {pct(r['soxx_R1w'])} | {pct(r['soxx_R2w'])} | {r['note']} |")
    cw = wk["corr_complete_weeks"]
    a("")
    a("| 완결 주의 P̄ vs | 주 수 | Spearman | 원형 이동 양측 p(1~N−1칸) | P̄ 중앙값 위 주 평균 | 이하 주 평균 |")
    a("|---|---|---|---|---|---|")
    for k, nm in (("kospi_R1w", "KOSPI 다음 주"), ("kospi_R2w", "KOSPI 2주"), ("soxx_R1w", "SOXX 다음 주"), ("soxx_R2w", "SOXX 2주")):
        v = cw[k]
        a(f"| {nm} | {v['n_weeks']} | {f(v['spearman'], 2)} | {f(v['p_shift_two_sided_all_shifts'], 3, False)} | "
          f"{pct(v['mean_ret_Pbar_above_median'])} ({v['n_above']}주) | {pct(v['mean_ret_Pbar_le_median'])} ({v['n_le']}주) |")
    nwk = sorted({v["n_weeks"] for v in cw.values()})
    a(f"- {nwk[0]}~{nwk[-1]}주라 우연과 가를 수 없다. 방향만 보면 KOSPI는 P̄가 높은 주 뒤가 다음 주·2주 모두 더 나빴고, "
      f"SOXX는 다음 주와 2주의 방향이 엇갈린다.")
    a("")
    a("## 특징 만들기 (가격 미사용)")
    a(f"- 표본: AI 예측 라벨 {meta['inputs']['labels']}단위(15파일, 대응표와 1:1, 빠진 라벨 {meta['inputs']['unlabeled']}), "
      f"W_KR 창 {meta['windows']['windows']}개({meta['windows']['first']}~{meta['windows']['last']}), 창마다 글 30·댓글 30 층화 무작위. "
      f"지식 마감 달(06-30) 단위 {meta['inputs']['dropped_knowledge_period_units']}개(K2026-07-01 창)를 빼 {n_used}단위를 썼다. "
      f"K2026-07-01 창은 10단위만 남는다.")
    a(f"- 예측 값 분포(단위): 0 {pc[0]:,} · 1 {pc[1]} · 2 {pc[2]} · 3 {pc[3]} · 4 {pc[4]} · 5 {pc[5]}. 예측 있는 단위 "
      f"{n_predu}개({100 * n_predu / n_used:.1f}%).")
    nw = meta["windows"]["n_pred_pairs"]
    a(f"- 작성자당 1회: (창, 작성자 해시, pred) 쌍을 한 번만 센다(감정 층 창 비율과 같은 방식). {meta['windows']['author_dedup_removed_units']}"
      f"단위가 겹쳐 빠졌다. 창당 예측 쌍은 최소 {nw['min']}, 중앙값 {f(nw['median'], 0, False)}, 최대 {nw['max']}개다. 그래서 창별 P̄의 "
      f"표본 오차(창 안 SD ÷ √n의 중앙값)가 {f(se_med, 2, False)}로, P̄ 값 자체(대부분 −1~+1)에 비해 크다.")
    fin = meta["windows"]["finite"]
    a(f"- 값이 있는 창: P̄·P_share·P_strong1 {fin['P_bar']}, P_share rz8 {fin['P_share_rz8']}, P_sd(등록) {fin['P_sd']}, "
      f"P_sd rz8(등록) {fin['P_sd_rz8']}, P_sd_lite(비등록) {fin['P_sd_lite']}.")
    a("")
    a("## 결과 보기 전에 정한 해석")
    a("| 항목 | 정한 해석 | 이유 |")
    a("|---|---|---|")
    a("| 작성자당 1회 | (창, 작성자, pred) 쌍 1회. P̄·P_sd·P_strong은 예측 쌍(1~5), P_share는 전체 쌍 분모 | 감정 층 window_table 구현(audit ⑤ 확인)과 같은 방식 |")
    a("| P_strong 분모 | 예측 쌍(1~5). 전체 쌍 분모판은 보조 | P_share가 이미 '예측의 양'을 잰다. 전체 분모면 P_strong이 P_share와 섞인다 |")
    a("| P_sd 등록판 | 동결 D_SD 구현 상수 그대로(풀 직전 60창 중 40창 이상, 표 20개 이상) → 값 0창, 시험 불가 | 부록 9 '최소 창 수 미달은 결측' |")
    a("| P_sd 설명판 | 풀 10창 이상, n 5 이상, rz8 없이 원값. 판정·보정에 넣지 않음 | test_free H35 비등록 30일판과 같은 처리 |")
    a("| 지식 마감 | 단위 시각 < 2026-07-01 제외 | §2.2 LLM 라벨, 덜 넣는 쪽 |")
    a("| M0 | 동결 확장 OLS(최소 120창)를 못 채워 같은 M0 변수의 표본 안 OLS 잔차('미리보기 대체'). 실시간 신호가 아니다 | 계산 지시, test_free와 같음 |")
    a("| H66 판정 통제 | KOSPI r(D−1)·r5·r20·ln RV20·g(D) + 전야 SOX | §2.3 |")
    a("| HAR-K·위약·강건·원형 이동 | test_free.py 함수 그대로 | 같은 미리보기 안 일관성 |")
    a("| 단계표 | 확장 백분위 최소 30창, 단계당 기대 창 5개 미만이면 5단계 | 부록 9 단계표 대체 |")
    a("| 주 표 | 단위 시각으로 주 창 배정, 주 안에서 다시 작성자당 1회 | §2.0 주 창 |")
    a("")
    a("## 새 가설 후보 (판정 안 함, 결과를 본 뒤의 관찰)")
    a("- 동결 정의는 바꾸지 않는다. 아래는 새 id로 등록한 뒤 2026-10-19 이후 실시간 자료로만 시험할 후보다.")
    a(f"- 후보 1(포럼 밖, 가격 반전): H66 위약에서 가격만으로 만든 X̂가 KOSPI R5와 IC {f(h66['placebo']['ic'])}였다. "
      f"M0 가격 변수 조합만으로 KOSPI 5일 반전을 맞히는 가격 모형 후보다(포럼 가설 아님).")
    a(f"- 후보 2(예측 몫 원값): rz8 예열이 8주를 먹어 H70 등록판이 {h70['n_obs']}창뿐이다. 원값판(n {aux70['n']}) 부분 r "
      f"{f(aux70['partial_r'])}를 실시간으로 다시 볼 후보다(정의가 달라 새 id).")
    a(f"- 후보 3(군중 예측의 가격 추종 강도): P̄와 지난 1일 수익의 상관 {f(r1['spearman'], 2)}은 결론 2(심리는 가격을 따라감)의 "
      f"창 단위 확인이다. 추종에서 벗어난 날(잔차가 큰 날)만 모은 사건형은 H66과 겹치므로 H66 최종 판정 때 함께 본다.")
    a("")
    a("## 한계")
    a("- 앞 구간(1~5월)이 없어 동결 '통과'는 원리상 나올 수 없다. 최종 판정의 뒤 구간은 이 미리보기와 겹친다('뒤 구간 미리 봄').")
    a(f"- 창당 예측 쌍이 중앙값 {f(nw['median'], 0, False)}개라 창별 P̄가 시끄럽다. 표본을 늘리면(창당 단위 수) 측정 오차가 줄어든다.")
    a(f"- 표본이 작다: H66 검출 한계 IC {f(h66['mde_ic'], 2, False)}, H70 검출 한계 부분 r {f(h70['mde_partial_r'], 2, False)}. "
      f"동결 문서의 현실적 기대 IC 0.05~0.15보다 훨씬 크다.")
    a("- 7월 말 급락·반등이 들어 있는 큰 변동 국면이다(사전 없는 칸 미리보기와 같은 기간).")
    a("- 예측 라벨은 AI 한 번 판정이다. 사용자 대 AI 검수 시험에서 예측 방향은 기준을 넘었지만, 창 단위 오차는 따로 재지 않았다.")
    a("")
    a("## 파일")
    a("- 특징: `preview/results/pred_windows.parquet`(창별 P̄·P_sd·P_share·P_strong·표본 수), `pred_weeks.parquet`(주 창), "
      "`pred_windows_meta.json`")
    a("- 결과: `preview/results/test_pred.json`(칸별 수치·강건·위약·보조, 설명용 표 전부), 이 문서, `chart_data.json`(그림용)")
    a("- 코드: `preview/code/pred_windows.py`(특징, 가격 미사용), `test_pred.py`(시험·설명용), `test_pred_report.py`(이 문서)")
    (RES / "test_pred.md").write_text("\n".join(L) + "\n", "utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
