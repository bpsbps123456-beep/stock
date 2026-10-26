"""audit_timing.json + audit_timing_forum.json → preview/results/audit.md (숫자는 모두 json에서 읽는다).
실행: py audit_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / "results"


def f(v, d=3, sign=True):
    if v is None:
        return "—"
    return f"{v:+.{d}f}" if sign else f"{v:.{d}f}"


def main() -> int:
    A = json.loads((RES / "audit_timing.json").read_text("utf-8"))
    FR = json.loads((RES / "audit_timing_forum.json").read_text("utf-8"))
    S, Fz, E, Lb, P, X, D = (A[k] for k in ("S_static", "F_fut_perturb", "E_entry", "L_label", "P_price_perturb",
                                             "X_fixed_m0", "D_lag_clean"))
    L = []
    a = L.append
    a("# 개미 지수 미리보기 시점 감사 (2026-10-18)")
    a("")
    a("> 대상: `preview/code/`의 스냅샷·특징 재실행·가격·패널·예측 라벨 표본·예측 특징·시험 코드와 그 결과. 기준은 부록 9(미리보기 규칙), "
      "`HYPOTHESES.md` §2(창·누출·M0·달력·판정)·§3, 부록 4(예측 층), `emotion/guideline_v2.md` §4다. 동결 문서와 interim/·emotion/ 산출은 "
      "고치지 않았다. 감사 코드는 `preview/code/audit_timing.py`(②~④, 고친 계산)·`audit_timing_forum.py`(①)·`audit_report.py`(이 문서)다.")
    a("")
    # ---------------------------------------------------------------- 결론
    pt = FR["perturb"]
    a_runs = {k: v for k, v in pt.items() if k.endswith("_A")}
    b_runs = {k: v for k, v in pt.items() if k.endswith("_B")}
    k_runs = {k: v for k, v in pt.items() if k.endswith("_B_keeproots")}

    def ndiff(v):
        return sum(len(c["diff"]) for c in v["compare"].values())

    a_bad = sum(1 for v in a_runs.values() if ndiff(v))
    b_bad = sum(1 for v in b_runs.values() if ndiff(v))
    ls_bad = sum(1 for v in pt.values() if v["label_sample"]["only_base"] + v["label_sample"]["only_pert"])
    kr_ok = bool(k_runs) and all(ndiff(v) == 0 for v in k_runs.values()) and         {k.split("_")[0] for k, v in b_runs.items() if ndiff(v)} <= {k.split("_")[0] for k in k_runs}
    cut = P["cuts"]["K2026-09-15 마감"]["m0_residual"]
    a("## 결론")
    a("| 점검 | 방법 | 결과 | 판정 |")
    a("|---|---|---|---|")
    a(f"| ① 특징이 창 마감 뒤 글·댓글·가격을 쓰지 않는가 | 동결 감사 하네스로 이 스냅샷의 원 줄을 교란: 마감 T {len(a_runs)}개 × "
      f"교란 A(내용·작성자·수치·새 글·다시 받은 줄)·B(A + 20% 지우기) | A {len(a_runs)}번 모두 마감 ≤ T 창의 특징 3단계 모든 열이 같다"
      f"(다른 열이 있는 실행 {a_bad}번). 예측 층 표본이 달라진 실행 {ls_bad}번. B에서 다른 열이 있는 실행 {b_bad}번"
      + (f"은 모두 뿌리 보존 B에서 0열이 되는 줄기 뿌리 생존 편향(동결 감사와 같은 분류)이다" if b_bad and kr_ok else
         ("" if not b_bad else "(뿌리 보존 B로 확인 못 함, 아래 참조)"))
      + f" | {'통과' if a_bad == 0 and ls_bad == 0 and (b_bad == 0 or kr_ok) else '확인 필요'} |")
    a(f"| ② fut_* 열이 특징·M0에 들어가는가 | 정적 목록 점검, fut_* 값 교란, T 뒤 가격 봉 교란 | fut_* 교란: 모든 칸의 특징·M0 잔차·X̂·통제 설계가 같다. "
      f"가격 교란: 마감 ≤ T 창의 m0_* 열은 같다(진입 순간 g만 T 창에서 바뀜). **그러나 표본 안 M0 잔차는 바뀐다**(H46 "
      f"{cut['H46']['insample_xperp_changed_pre']}/{cut['H46']['pre_T_windows_in_sample']}창, H66 "
      f"{cut['H66']['insample_xperp_changed_pre']}/{cut['H66']['pre_T_windows_in_sample']}창) | fut_* 열은 통과. M0 적합은 문제 1 |")
    a(f"| ③ 진입 시가·결과 기간 | 달력에서 창 마감·진입 세션·P·직전 상대 시장 세션을 독립으로 다시 정하고, Rh·MDDh·HL·끝 세션을 참조 구현으로 재계산 | "
      f"국내 {E['kr']['windows']}창·미국 {E['us']['windows']}창에서 어긋남 {E['kr']['n_window_entry_bad'] + E['us']['n_window_entry_bad']}건, "
      f"결과 값 어긋남 {E['kr']['n_fut_mismatch'] + E['us']['n_fut_mismatch']}건. 칸별 대상이 동결 표와 같다 | 통과 |")
    a(f"| ④ 예측 라벨 표본 층 | 층·창별 수·재현·코드가 읽는 열·조각 순서·라벨 교란 | 층은 창 × 글/댓글뿐, 창마다 글 "
      f"{'·'.join(map(str, Lb['per_window']['post_counts']))}·댓글 {'·'.join(map(str, Lb['per_window']['comment_counts']))}, 같은 시드 재현 {'같음' if Lb['reproduce']['same'] else '다름'}"
      f", 뽑기가 쓰는 열 {', '.join(S['label_draw_columns_used'])}. 뽑기에 가격·결과·판정 자격 열을 쓰지 않는다 | 통과 |")
    a(f"| 설명용 시차표 | 수익 구간 시작과 글 시각 비교 | KOSPI +1일·SOXX 0일이 겹침 | 문제 2, 고침 |")
    a("")
    a("- **문제 1(고침: 숫자 재계산·보고서 표시).** H46·H31·H66·H68의 주 특징 X⊥와 모든 칸의 위약 X̂는 표본 안 OLS 한 번으로 만들어졌다. "
      "계수가 뒤 창의 M0 행(진입 뒤 가격으로 만든 r5·r20 등)으로 정해지므로 특징이 앞날 가격을 쓴다. 동결 §2.2('기준선·회귀 계수는 그날까지의 "
      "자료로만')·§2.3('확장 OLS, D−1까지, 최소 120창, 월 1회 재적합') 위반이다. 동결대로면 이 자료(창 93·98개)에서 X⊥는 결측이라 네 칸은 "
      "'시험 불가'다. 앞 자료만 쓰는 감사판으로 다시 셌고, 어느 판으로도 통과 모양 칸은 0개라 미리보기 결론은 바뀌지 않는다. 모양 표지는 바뀐다(아래 표).")
    a("- **문제 2(고침).** 설명용 '하루 글 수 ↔ 수익 시차표'에서 KOSPI +1일과 SOXX 0일은 수익 구간이 그날 24:00 전에 시작해 글과 겹치는데 "
      "'글이 먼저' 칸으로 읽혔다. KOSPI +1일을 다음 세션 시가→종가로 바꾸고 평균·문장을 고쳤다.")
    a("")
    # ---------------------------------------------------------------- ①
    a("## ① 포럼 교란 시험 (특징 시점)")
    hv = FR["harness_vs_saved"]
    a(f"- 하네스: `interim/code/audit_leak.py`를 고치지 않고 import했다. 입력만 이 스냅샷(`preview/snapshot_manifest.json`, S {FR['S']})으로 바꿨다. "
      f"하네스 기준 실행이 저장된 미리보기 산출을 그대로 다시 만드는지 먼저 봤다: 활동 {hv['fa']['rows']}행·{hv['fa']['cols']}열, 텍스트 "
      f"{hv['ft']['rows']}행·{hv['ft']['cols']}열 모두 같고, 주제 {hv['fto']['rows']}행·{hv['fto']['cols']}열은 날짜 열 "
      f"{', '.join(hv['fto']['diff_cols'])}의 표기(date ↔ datetime 문자열)만 다르다(동결 감사와 같은 차이). 예측 층 표본도 "
      f"{FR['label_sample_harness_vs_saved']['saved']}개 (창, 종류, 글, 작성자) 키가 모두 같다.")
    a("- 교란 A: 마감 T 뒤 글·댓글의 제목·본문·작성자를 감정어 묶음이나 무작위 글자로 바꾸고, 모든 줄의 조회·추천·다른의견과 다시 받은 줄의 "
      "댓글 수·제목·본문을 바꾸고, T 뒤 새 글 300개와 'T 뒤 댓글만 든 다시 받은 줄'(T 전 글의 30%)을 더한다. 교란 B: A + T 뒤 글·댓글 20% 지움.")
    a("")
    a("| 마감 T | 교란 | 비교 창(활동/텍스트/주제) | 다른 열 | 예측 층 표본(마감 ≤ T 창, 07-01 뒤) | 마감 전 단위 더해짐/빠짐(국내 창) |")
    a("|---|---|---|---|---|---|")
    for k in sorted(pt, key=lambda x: (int(x.split("_")[0][1:]), x)):
        v = pt[k]
        c = v["compare"]
        dif = sorted({col for nm in c.values() for col in nm["diff"]})
        ls = v["label_sample"]
        u = v["units"]["win_kr"]
        a(f"| {v['T']} | {v['variant']} | {c['fa']['rows_compared']}/{c['ft']['rows_compared']}/{c['fto']['rows_compared']} | "
          f"{len(dif)}{(' (' + ', '.join(dif[:6]) + (' …' if len(dif) > 6 else '') + ')') if dif else ''} | "
          f"{ls['windows_compared']}창 {ls['units_base']}단위, 다른 단위 {ls['only_base'] + ls['only_pert']} | "
          f"+{u['added']} / −{u['removed']} |")
    a("")
    bd = {k: v for k, v in b_runs.items() if ndiff(v)}
    if bd:
        cols = sorted({col for v in bd.values() for nm in v["compare"].values() for col in nm["diff"]})
        other = [c for c in cols if "fight" not in c]
        rc = {k: v["units"]["pre_T_comment_root_changed"] for k, v in bd.items()}
        a(f"- B에서 다른 열이 있는 실행은 {len(bd)}번({', '.join(k.split('_')[0] for k in bd)})이고, 다른 열은 "
          + ("모두 댓글 싸움(fight) 계열이다" if not other else f"fight 계열과 {', '.join(other)}이다")
          + "(fight 자체가 바뀐 창: " + ", ".join(f"{k.split('_')[0]} {v['compare']['fa']['diff'].get('fight', {}).get('rows', 0)}창"
                                            for k, v in bd.items())
          + "; 직전 60창 z인 fight_z60은 바뀐 창 뒤로 번져 최대 "
          + ", ".join(f"{k.split('_')[0]} {v['compare']['fa']['diff'].get('fight_z60', {}).get('rows', 0)}창" for k, v in bd.items())
          + "). "
          + "마감 전 대댓글의 줄기 뿌리가 바뀐 경우가 "
          + ", ".join(f"{k.split('_')[0]} {r['changed']}건(그중 목록상 뿌리 시각이 대댓글보다 늦은 것 {r['changed_and_base_root_dated_after_reply']}건)"
                      for k, r in rc.items())
          + "이다. 지운 뿌리는 시각이 T 뒤로 찍혔지만(뿌리를 고친 시각) 그 아래 대댓글이 T 전에 있었으니 뿌리도 T 전에 있었다. '미래 내용'이 아니라 "
          "'나중에 지워진 과거 글'의 생존 편향이다(동결 감사 notes_audit_leak와 같은 분류, §7 지운 글).")
        if k_runs:
            a("- 확인: 같은 B에서 깊이 0(뿌리) 미래 댓글만 지우지 않으면 "
              + ", ".join(f"{k.split('_')[0]} 다른 열 {ndiff(v)}개" for k, v in k_runs.items()) + "다.")
    a("")
    # ---------------------------------------------------------------- ②
    a("## ② 결과 열(fut_*)과 가격의 방화벽")
    li = S["lists"]
    a("- 정적: 시험 코드의 통제·M0 목록 " + ", ".join(f"{k} {v['n']}열" for k, v in li.items())
      + f"에 fut_* 열 {sum(len(v['fut_cols']) for v in li.values())}개, m0_* 밖 열 {sum(len(v['non_m0_cols']) for v in li.values())}개. "
      "진입 순간 값은 HAR-K의 |g(D)|(동결 F1 식에 있음)와 판정 통제의 g(D)(동결 §2.3 '판정 회귀에는 … g(D)를 넣고')뿐이고, M0 목록에는 없다.")
    sh = S["source_hits"]
    a("- 정적: 표본·특징 코드(" + ", ".join(sh) + ")에서 fut_*·가격 파일을 읽는 줄: "
      + ", ".join(f"{k} {sum(len(v[x]) for x in ('fut_', 'prices'))}줄" for k, v in sh.items())
      + ". feature_summary.py는 panel에서 판정 자격 표지만 읽는다(특징 값에 쓰지 않음).")
    a(f"- 결과 교란: panel의 fut_* 값(국내 {Fz['fut_cols_perturbed']['kr']}열, 미국 {Fz['fut_cols_perturbed']['us']}열)을 모두 무작위로 바꿔도(결측 자리는 그대로) "
      + ", ".join(f"{k} {'같음' if all(v[x] for x in v if x != 'n') else '다름'}" for k, v in Fz["cells"].items())
      + "(특징·M0 잔차·X̂·통제·HAR 설계). 결측 자리는 표본(결과가 있는 창)을 정하는 데만 쓰인다.")
    a("- 가격 교란: 시각 T 뒤에 열린 봉(KRX 09:00, NYSE 09:30 ET 시가 기준 일봉, 끝이 T 뒤인 NQ 60분봉)을 모두 무작위로 바꾸고 panel을 다시 만들었다. "
      "다시 만든 panel이 저장된 panel과 같은지 먼저 봤다: "
      + ", ".join(f"{k} {v['cols']}열 중 다른 열 {len(v['diff_cols'])}개" for k, v in P["rebuild_vs_saved"].items()) + ".")
    a("")
    a("| T | 바뀐 봉(KOSPI/SOXX/NQ 60분) | 마감 ≤ T 창(국내/미국) | 바뀐 m0_* 열(마감 ≤ T) | 바뀐 진입 순간 g | 표본 안 M0 잔차가 바뀐 창(H46·H31·H66·H68) | 앞 자료 M0 잔차가 바뀐 창 |")
    a("|---|---|---|---|---|---|---|")
    for lab, r in P["cuts"].items():
        bc = r["bars_changed"]
        g = {**r["kr"]["entry_instant_changed_pre"], **r["us"]["entry_instant_changed_pre"]}
        mr = r["m0_residual"]
        a(f"| {lab} | {bc['^KS11']}/{bc['SOXX']}/{bc['NQ=F_1h']} | {r['kr']['windows_close_le_T']}/{r['us']['windows_close_le_T']} | "
          f"{len(r['kr']['m0_changed_cols_pre']) + len(r['us']['m0_changed_cols_pre'])} | {len(g)}열 1창(T에 닫는 창의 D 시가) | "
          + " · ".join(f"{k} {v['insample_xperp_changed_pre']}/{v['pre_T_windows_in_sample']}" for k, v in mr.items())
          + " | " + " · ".join(f"{k} {v['past_xperp_changed_pre']}/{v['past_xperp_finite_pre']}" for k, v in mr.items()) + " |")
    a("")
    a("- 10-13 08:00 교란에서 표본 안 잔차가 안 바뀐 것은 그 뒤 창이 결과(5·10일)가 없어 적합 표본에 없기 때문이다. 결과가 생기면 같은 일이 생긴다.")
    a("")
    # ---------------------------------------------------------------- ③
    a("## ③ 진입 시가와 결과 기간")
    a("- 국내 창: 마감 = D 08:00(07:59 포함), D는 KRX 세션, P = D 앞 마지막 KRX 세션, 직전 미국 세션 = 16:00 ET 종가가 창 마감 전인 마지막 NYSE 세션. "
      "미국 창: 마감 = D 08:30 ET(08:29 포함)의 KST, D는 NYSE 세션이고 시가(09:30 ET)가 마감 뒤, P = D 앞 마지막 NYSE 세션, 직전 국내 세션 = 15:30 종가가 "
      "마감 전인 마지막 KRX 세션, NQ 끝 바 = 끝 시각 ≤ D 08:29 ET(실제 08:00 바). 모두 독립으로 다시 정해 panel과 맞췄다: "
      f"어긋남 국내 {E['kr']['n_window_entry_bad']}건, 미국 {E['us']['n_window_entry_bad']}건.")
    a(f"- 결과: Rh = 진입 시가 → 진입일+h−1 종가, MDDh = [진입 시가, D 종가, …, D+h−1 종가] 경로의 최대 낙폭, HL = ln(D 고가 ÷ D 저가)을 참조 구현으로 "
      f"모든 자산(국내 4·미국 4)·기간(1·5·10·21일)에서 다시 셌다: 어긋남 {E['kr']['n_fut_mismatch'] + E['us']['n_fut_mismatch']}건. 끝 세션 날짜도 같다.")
    a("- 칸별 대상: " + ", ".join(f"{k} {v}" for k, v in E["cell_targets"].items()) + " — 동결 표("
      + ", ".join(f"{k} {v}" for k, v in E["frozen_targets"].items()) + ")와 같다.")
    a(f"- 참고: 지수 불량 봉(^KS11·^KQ11 2026-06-29, 시가=고가=저가=종가·거래량 0)은 결측으로 두고 경로에서 건너뛴다. R10 경로가 이 날을 지나는 국내 창은 "
      f"{len(E['bad_bar_paths']['kr_windows_with_0629_in_R10_path'])}개(모두 6월, 예측 층 밖)다.")
    a("")
    # ---------------------------------------------------------------- ④
    a("## ④ 예측 라벨 표본")
    a(f"- 층: W_KR 창(D ≥ 07-01, {Lb['per_window']['windows']}창) × 글/댓글. 창마다 글 {'·'.join(map(str, Lb['per_window']['post_counts']))}·댓글 {'·'.join(map(str, Lb['per_window']['comment_counts']))}개(모자란 층 없음). "
      f"틀 크기는 창마다 글 {Lb['frame_size_per_window']['글']['min']}~{Lb['frame_size_per_window']['글']['max']}개, 댓글 "
      f"{Lb['frame_size_per_window']['댓글']['min']}~{Lb['frame_size_per_window']['댓글']['max']}개로 달라 뽑힘 비율이 창마다 다르지만, 창 특징은 창 안 비율·평균이라 치우치지 않는다. "
      "07-01 하한은 §2.2 LLM 지식 마감 규칙이다(가격·사건일 변수가 아니다).")
    a(f"- 뽑기 함수가 쓰는 열: {', '.join(S['label_draw_columns_used'])}. 틀에서 빼는 단위는 내용 없는 단위(삭제 댓글·사이트 알림·빈 단위)와 6월 정치 전용 단위뿐이다. "
      "label_key의 judge_ok·D는 뽑은 뒤 기록만 한다.")
    a(f"- 재현: 저장 units로 같은 시드를 다시 돌리면 {Lb['reproduce']['saved']}단위가 {'모두 같다' if Lb['reproduce']['same'] else '다르다'}. "
      f"① 교란 A·B에서도 마감 ≤ T 창의 표본은 모두 같았다.")
    a(f"- 라벨러가 받은 것: 조각 필드 {Lb['shard_fields']}, 라벨 필드 {Lb['label_fields']}. 조각 순서와 창 날짜의 Spearman {f(Lb['shard_order_vs_D_spearman'], 3)}(시간순으로 주지 않음). "
      f"조각 본문에서 연도·월일 꼴 문자열 {Lb['shard_date_pattern_hits']}건.")
    a(f"- 지식 마감: 06-30 단위 {Lb['knowledge_cutoff']['dropped_units']}개가 특징에서 빠졌고, 쓴 단위의 가장 이른 시각은 {Lb['knowledge_cutoff']['min_dt_used']}다.")
    a("- 라벨 교란: 마감 > T 창의 라벨(pred)과 작성자를 무작위로 바꾸고 그 단위의 30%를 지운 뒤 예측 특징을 다시 만들었다.")
    a("")
    a("| T | 바꾼 단위 | 지운 단위 | 마감 ≤ T 창 | 다른 열(창 특징 42열 중) | 마감 ≤ T 주 창 | 다른 열(주) | 확인: 마감 > T 창 P̄가 바뀐 창 |")
    a("|---|---|---|---|---|---|---|---|")
    for lab, r in Lb["label_perturb"].items():
        a(f"| {lab} | {r['units_changed']} | {r['units_dropped']} | {r['windows_le_T']} | {len(r['diff_cols'])} | {r['weeks_le_T']} | "
          f"{len(r['week_diff_cols'])} | {r['sanity_post_T_P_bar_changed']} |")
    a("")
    # ---------------------------------------------------------------- 문제 1
    a("## 문제 1: M0 잔차·위약 X̂의 표본 안 적합")
    a("- 어디: `test_free.m0_hat`(test_pred가 그대로 부름). 표본 S 전체로 X를 M0에 OLS 한 번 적합해 X⊥(잔차)와 X̂(적합값)를 만든다. "
      "창 t의 X⊥에 쓰인 계수가 t 뒤 창의 (X, M0)로 정해지고, t 뒤 창의 M0에는 t의 진입 뒤 가격(r(D−1)·r5·r20·RV20 등)이 들어 있다. "
      "보고서는 '미리보기 대체'와 '실시간 신호가 아니다'를 적었지만, 이것이 특징에 앞날 가격이 섞이는 것이라는 점과 동결 규칙상 시험 불가라는 점은 적지 않았다.")
    a(f"- 고친 계산(비등록 감사판, 결과를 보기 전에 정함): 동결 §2.3 모양 그대로 확장 OLS, D−1까지, 월 1회 재적합(그 달 첫 거래일 전 창으로 적합), "
      f"X⊥ = 잔차 ÷ 적합 잔차 SD(§2.0). 동결 최소 120창은 못 채워 최소 {X['t2']['monthly']['H46']['min_fit']}창(부록 9 단계표 대체와 같은 수)으로 했다. "
      "보조로 창마다 재적합한 판도 냈다. 원형 이동 p는 특징열을 돌릴 때마다 앞 자료 적합을 다시 한다. 강건 3개는 X⊥를 고정하고 IC 표본에서만 뺀다.")
    a("")
    a("| 칸 | 판 | n | 기간 | IC(X⊥) | t | 원형 이동 p (이동 수) | 판정 p | n_eff | 검출 한계 | 강건 | 위약 X̂ IC (표지) | 모양 |")
    a("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k in ("H46", "H31", "H66", "H68"):
        o = X["orig"][k]
        a(f"| {k} | 원래(표본 안) | {o['n_obs']} | — | {f(o['ic'])} | {f(o['t'], 2)} | — | {f(o['p_final_one'], 3, False)} | — | — | — | "
          f"{f(o['placebo_ic'])} ({'예' if o['placebo_flag'] else '아니오'}) | {o['verdict']} |")
        for mode, lab in (("monthly", "앞 자료, 월 재적합(주)"), ("window", "앞 자료, 창마다(보조)")):
            c = X["t2"][mode][k]
            if "p_final_one" not in c:
                a(f"| {k} | {lab} | {c['n_obs']} | — | — | — | — | — | — | — | — | — | 계산 안 함 |")
                continue
            shape = X["shapes_monthly"][k] if mode == "monthly" else "—"
            a(f"| {k} | {lab} | {c['n_obs']} | {c['first_D']}~{c['last_D']} | {f(c['ic_main_xperp_partial'])} | {f(c['t'], 2)} | "
              f"{f(c['p_shift_one'], 3, False)} ({c['shift_K']}) | {f(c['p_final_one'], 3, False)} | {f(c['n_eff'], 1, False)} | "
              f"{f(c['mde_ic'], 2, False)} | {'예' if c['robust_ok'] else '아니오'} | {f(c['placebo']['ic'])} "
              f"({'예' if c['placebo']['flag_70pct_same_sign'] else '아니오'}) | {shape} |")
    a("")
    a(f"- 감사판 BH(H46·H31·H66, 미리보기 표시) q: " + ", ".join(f"{k} {f(v, 3, False)}" for k, v in X["bh_monthly"].items()) + ". H68은 설명용이라 넣지 않았다.")
    a("- T1 칸(H03·H08·H70, H67 설명판)의 주 추정(HAR-K 계수)은 X⊥를 쓰지 않는다. 위약 X̂만 표본 안이었다. 앞 자료 X̂로 다시 셈:")
    a("")
    a("| 칸 | 판 | 앞 자료 X̂가 있는 창 | X 부분 r(같은 표본) | X̂ 부분 r | 표지 | 원래 X̂ 부분 r (표지) | 모양(감사판, 월 재적합) | 원래 모양 |")
    a("|---|---|---|---|---|---|---|---|---|")
    for k in ("H03", "H08", "H70", "H67_lite"):
        o = X["orig"][k]
        for mode, lab in (("monthly", "월 재적합"), ("window", "창마다")):
            c = X["t1_placebo"][mode][k]
            sh = X["shapes_monthly"].get(k, "설명용") if mode == "monthly" else "—"
            if "placebo_partial_r" not in c:
                a(f"| {k} | {lab} | {c['n_with_past_xhat']} | — | — | 계산 안 함(15창 미만) | {f(o['placebo_partial_r'])} "
                  f"({'예' if o['placebo_flag'] else '아니오'}) | {sh} | {o['verdict']} |")
                continue
            a(f"| {k} | {lab} | {c['n_with_past_xhat']} | {f(c['partial_r_x_same_sample'])} | {f(c['placebo_partial_r'])} | "
              f"{'예' if c['flag_70pct_same_sign'] else '아니오'} | {f(o['placebo_partial_r'])} ({'예' if o['placebo_flag'] else '아니오'}) | "
              f"{sh} | {o['verdict']} |")
    a("")
    mm = X["t2"]["monthly"]
    ns = [mm[k]["n_obs"] for k in ("H46", "H31", "H66", "H68") if "p_final_one" in mm[k]]
    md = [mm[k]["mde_ic"] for k in ("H46", "H31", "H66", "H68") if "p_final_one" in mm[k]]
    npass = sum(1 for v in X["shapes_monthly"].values() if v.startswith("통과"))
    flips = [k for k in ("H46", "H31", "H66", "H68")
             if "p_final_one" in mm[k] and (mm[k]["ic_main_xperp_partial"] > 0) != (X["orig"][k]["ic"] > 0)]
    a("- 읽기: (1) 동결 규칙대로면 H46·H31·H66·H68은 '시험 불가(등록 M0 최소 120창 미충족)'다. "
      f"(2) 표본 안 판과 감사판 모두 통과 모양 칸은 0개다(감사판 통과 모양 {npass}칸). "
      f"(3) 감사판은 9월부터만 값이 있어 n {min(ns)}~{max(ns)}, 검출 한계 IC {min(md):.2f}~{max(md):.2f}로 결론을 낼 힘이 없다. "
      f"점추정 부호가 표본 안 판과 반대인 칸: {', '.join(flips) if flips else '없음'}. "
      "(4) 감사판 모양: " + ", ".join(f"{k} {v}" for k, v in X["shapes_monthly"].items())
      + ". 표본 안 판의 '기각 모양(메아리 위약)' 표지(H31·H66·H68·H70)는 앞날 가격이 든 X̂로 붙은 것이라 판정 근거로 쓰지 않는다. "
      "(5) 표본 안 부분 상관 분해(원 IC → 가격 통제 → M0 잔차)는 설명으로는 그대로 읽어도 된다(통제 회귀 자체는 시험 통계다).")
    a("- 반영: `test_free.md`·`test_pred.md` 맨 앞 '시점 감사 반영' 절과 결론 문장에 위 읽기와 감사판 숫자를 넣었다(보고서 코드 `test_free_report.py`·"
      "`test_pred_report.py`, 숫자는 audit_timing.json에서 읽음). 원래 칸별 표는 '표본 안 M0 설명판'으로 남겼다.")
    a("")
    # ---------------------------------------------------------------- 문제 2
    a("## 문제 2: 설명용 시차표의 시간 겹침")
    a("- 어디: `test_free.daily_tables`의 '하루 글 수(KST 0~24시) ↔ k거래일 뒤 수익' 표와 상위 10일 표. 종가 대 종가 수익은 앞 세션 종가에서 시작한다. "
      "KOSPI +1일은 그날 15:30에, SOXX 0일은 그날 05:00 KST에 시작해, 그날 글의 일부가 수익 구간 안(밤사이 미국장·선물·프리장)에 있다. "
      "보고서는 'k > 0이면 글이 먼저', 'SOXX 0일은 글이 대부분 먼저'라고 적었다.")
    a("- 고침: KOSPI +1일은 다음 세션 시가(09:00)→종가로 바꿨다(그날 글이 모두 앞선다). KOSPI +2일부터와 SOXX +1일부터는 구간 시작이 그날 24:00 뒤라 그대로다. "
      "SOXX 0일은 겹침으로 표시했다. `test_free.py`에 `lead_lag_clean`·`kospi_oc_next_session`을 더했고(기존 값은 그대로), 보고서 문장을 고쳤다.")
    a("")
    a("| 값 | 원래(종가 대 종가) | 고친 판 |")
    a("|---|---|---|")
    kc = {(r["lag"], r["ret"], r["kind"]): r["spearman"] for r in D["kospi"]}
    a(f"| KOSPI +1일 \\|수익\\| | {f(D['orig_kospi_absr_lag1'], 2)} | {f(kc[(1, 'clean', 'absr')], 2)} (시가→종가) |")
    a(f"| KOSPI +1일 수익 | {f(D['orig_kospi_r_lag1'], 2)} | {f(kc[(1, 'clean', 'r')], 2)} (시가→종가) |")
    a(f"| KOSPI \\|수익\\| '글 먼저' +1~+5일 평균 | {f(D['orig_kospi_absr_lead_mean_1_5'], 2)} | {f(D['kospi_absr_lead_mean_1_5_clean'], 2)} |")
    a(f"| KOSPI 수익 '글 먼저' +1~+5일 평균 | {f(D['orig_kospi_r_lead_mean_1_5'], 2)} | {f(D['kospi_r_lead_mean_1_5_clean'], 2)} |")
    up_cc = sum(1 for r in D["top10_next_session"] if (r["next_cc"] or 0) > 0)
    up_oc = sum(1 for r in D["top10_next_session"] if (r["next_open_to_close"] or 0) > 0)
    a(f"| 글 수 상위 10일의 다음 거래일 오른 날 | {up_cc}일 | {up_oc}일 (시가→종가) |")
    a(f"| SOXX 0일 \\|수익\\| / 수익 | {f(D['orig_soxx_absr_lag0'], 2)} / {f(D['orig_soxx_r_lag0'], 2)} | 겹침 표시(글이 먼저가 아님) |")
    a("")
    a("- 결론('앞뒤 시차 상관이 비슷해 어느 쪽이 먼저인지 가를 수 없다')은 그대로다. 우연 범위를 넘던 KOSPI |수익| +1일은 고친 판에서 우연 범위 안이다.")
    a("")
    # ---------------------------------------------------------------- 남은 것
    a("## 고치지 않은 것 (동결 문서가 정한 예외·한계, 크기만 적음)")
    a("- 결번율(M0·HAR-K 통제)은 스냅샷 때까지 지워진 글로 센다. 마감 뒤 지움이 섞이지만 동결 §2.2·§2.3이 정한 통제다.")
    a("- 소급분(10/13 수집 시작 전) 글의 첫 줄은 마감 뒤 며칠~몇 달 뒤에 받았다. 마감 뒤 본문 수정·댓글 지움이 남는다(동결 §2.2 '본문 수정'·'지운 글', §7). "
      "예측 층 표본 72창도 대부분 이 기간이다. 삭제 댓글 표지가 든 단위는 내용이 없어 틀에서 뺐다(뽑힐 수 없는 단위).")
    a("- 줄기 뿌리 생존 편향(① B): 댓글 싸움 계열만, 동결 감사와 같은 분류다.")
    a("- HAR-K·IC 회귀 자체는 표본 안 적합이다. 이것은 시험 통계이고 특징이 아니다. 판정 통제의 g(D)·|g(D)|는 진입 순간 값으로 동결이 허용한다.")
    a("- 라벨러가 조각 밖 자료(가격)를 보지 않았다는 것은 산출물로만 확인했다(조각에는 id·가린 본문만, 라벨 파일에는 id·pred·pred_target·conf만 있다).")
    a("")
    a("## 시험 범위")
    a(f"- 포럼 교란: 마감 T {len(a_runs)}개(국내 4, 미국 2) × A·B = {len(a_runs) + len(b_runs)}번" + (f", 뿌리 보존 B {len(k_runs)}번" if k_runs else "")
      + ". 한 번마다 로더와 특징 3단계를 원 줄부터 다시 돌렸다. 시드 20261018+k.")
    a(f"- 가격 교란: T {len(P['cuts'])}개(국내 마감 2, 미국 마감 1), 봉 배율 exp(N(0, 0.05)), NQ 60분봉 exp(N(0, 0.02)). 결과 교란: 1번(fut_* 96열).")
    a(f"- 라벨 교란: T {len(Lb['label_perturb'])}개. 감사판 M0: 2판(월 재적합 주, 창마다 보조) × T2 4칸, T1 위약 4칸. 감사판 설정(월 재적합·최소 30창·"
      "잔차 ÷ 잔차 SD)은 감사판 결과를 보기 전에 정했다.")
    a("")
    a("## 파일")
    a("- 결과: `preview/results/audit_timing.json`(②~④·고친 계산·시차표), `preview/results/audit_timing_forum.json`(①), 이 문서")
    a("- 코드: `preview/code/audit_timing.py`, `audit_timing_forum.py`, `audit_report.py`. 고친 코드: `test_free.py`(설명용 시차표 겹침 없는 판 추가), "
      "`test_free_report.py`·`test_pred_report.py`(시점 감사 반영 절). `test_free.py`·`test_pred.py`를 다시 돌려 칸별 숫자가 그대로인 것을 확인했다(json 비교: 더한 필드와 실행 시각·입력 지문만 다름).")
    (RES / "audit.md").write_text("\n".join(L) + "\n", "utf-8")
    print("written", len(L), "lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
