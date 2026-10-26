"""emotion_batchA_results.json → emotion_batchA.md (표만, 그림 없음). 가격 자료는 읽지 않는다.

실행: py emotion_batchA_report.py   (emotion_batchA.py 다음)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
EMO = HERE.parent
CATS = ["HYPE", "JOY", "MOCK", "DESP", "ANGER", "FEAR", "NEUT"]
EMO6 = [c for c in CATS if c != "NEUT"]
KO = {"HYPE": "들뜸·자랑", "JOY": "기쁨·안도", "MOCK": "조롱·비꼼", "DESP": "우울·절망", "ANGER": "분노·욕",
      "FEAR": "불안·공포", "NEUT": "중립·정보"}
TGTS = ["other", "bull", "bear"]
TKO = {"other": "기타", "bull": "강세론 조롱", "bear": "약세론 조롱"}


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def f(x, nd=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{nd}f}"


def pct(x, nd=1):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{100 * x:.{nd}f}%"


def cm_table(cm: dict, labels: list[str], row_name: str, col_name: str) -> list[str]:
    # cm: {col: {row: n}} (pandas to_dict 꼴)
    out = [f"| {row_name} ＼ {col_name} | " + " | ".join(labels) + " | 합 |", "|---|" + "---:|" * (len(labels) + 1)]
    for r in labels:
        vals = [int(cm[c][r]) for c in labels]
        out.append(f"| {r} | " + " | ".join(str(v) for v in vals) + f" | {sum(vals)} |")
    return out


def main() -> int:
    o = json.loads((EMO / "emotion_batchA_results.json").read_text("utf-8"))
    ag, cl, tc, gate, ls = o["agreement"], o["classifier"], o["target_classifier"], o["gate"], o["label_share"]
    ho = cl["heldout"]
    L: list[str] = []
    a = L.append

    # 정밀도 신뢰구간
    prec_ci = {}
    for c in CATS:
        pc = ho["per_cat"][c]
        n = pc["n_pred"]
        k = round(pc["precision"] * n) if pc["precision"] is not None else 0
        prec_ci[c] = (k, n, *wilson(k, n))
    passed = [c for c in CATS if gate[c]["pass"]]
    failed = [c for c in CATS if not gate[c]["pass"]]
    neut_base = ho["per_cat"]["NEUT"]["support"] / ho["n"]

    a("# 감정 범주 묶음 A: 일치도·분류기·적용·검수지 (2026-10-17)")
    a("")
    a("> `HYPOTHESES_addendum3.md`(동결) 측정 절차 3·5·6단계와 4단계 준비. 측정만 했고 가격·지수·수익 자료는 읽지도 계산하지도 않았다.  ")
    a("> 코드 `emotion/code/emotion_batchA.py`·`emotion_batchA_report.py`, 전체 숫자 `emotion/emotion_batchA_results.json`.")
    a("")
    a("## 결론")
    a("")
    s = ls["all"]["weighted"]
    se = ls["all"]["weighted_emo"]
    a(f"- **라벨로 본 비율(답)**: 묶음 A 틀(7/13~10/15 글·마감 전 댓글 252,246단위)로 층 가중하면 중립 {pct(s['NEUT'])}, "
      f"조롱 {pct(s['MOCK'])}, 분노·욕 {pct(s['ANGER'])}, 우울·절망 {pct(s['DESP'])}, 불안·공포 {pct(s['FEAR'])}, "
      f"들뜸·자랑 {pct(s['HYPE'])}, 기쁨·안도 {pct(s['JOY'])}다. 감정 단위만 보면 조롱 {pct(se['MOCK'])}·분노 {pct(se['ANGER'])}·"
      f"우울 {pct(se['DESP'])}·불안 {pct(se['FEAR'])}·들뜸 {pct(se['HYPE'])}·기쁨 {pct(se['JOY'])}다. 이 게시판의 감정은 비웃음과 욕이 2/3다.")
    a(f"- **라벨러 일치도는 높다**: 600단위 전체 κ {f(ag['kappa_all'])}(95% {f(ag['kappa_all_ci95'][0])}~{f(ag['kappa_all_ci95'][1])}), "
      f"7범주 모두 κ ≥ {f(min(v['kappa'] for v in ag['per_cat'].values()), 2)}. 다만 두 라벨러가 같은 AI라 '같은 기준으로 읽는다'는 뜻이지 맞다는 뜻은 아니다(사용자 검수가 그 확인이다).")
    a(f"- **분류기는 약하다**: 떼어 둔 {ho['n']}단위 정확도 {f(ho['accuracy'])}(모두 NEUT로 찍으면 {f(neut_base)}), macro-F1 {f(ho['macro_f1'])}, "
      f"라벨 대비 κ {f(ho['kappa_vs_label'])}. 감정 범주 재현율은 "
      + "·".join(f"{c} {pct(ho['per_cat'][c]['recall'], 0)}" for c in EMO6)
      + "이고, JOY·FEAR는 떼어 둔 표본에서 한 번도 예측하지 않았다. 전체 적용에서 중립이 "
      f"{pct(o['apply']['pred_share_all_units']['NEUT'])}로 라벨 비율({pct(s['NEUT'])})보다 크게 나와 감정을 크게 덜 센다.")
    a(f"- **품질 기준(떼어 둔 정밀도 ≥ 60% 그리고 κ ≥ 0.6)**: 문자 그대로 통과 {'·'.join(passed)}, 미달 {'·'.join(failed)}. "
      f"통과한 감정 범주도 정밀도가 HYPE {prec_ci['HYPE'][0]}/{prec_ci['HYPE'][1]}, MOCK {prec_ci['MOCK'][0]}/{prec_ci['MOCK'][1]}, "
      f"ANGER {prec_ci['ANGER'][0]}/{prec_ci['ANGER'][1]}건에서 나온 값이라 Wilson 95% 하한이 모두 0.6 아래다(불안정 통과).")
    tb = tc["heldout"]["per_cat"]
    a(f"- **조롱 대상**: 라벨러 κ {f(ag['mock_target']['kappa'])}(둘 다 MOCK인 {ag['mock_target']['n_both_mock']}단위)로 일치하지만 분류기는 "
      f"강세론 조롱 정밀도 {f(tb['bull']['precision'], 2)}({tb['bull']['n_pred']}건), 약세론 조롱 예측 {tb['bear']['n_pred']}건이라 미달이다. H63은 설명용이다.")
    a("")

    # 1. 일치도
    a("## 1. 일치도 (조각 01·05·09 1차 라벨 × second_1~3, id로 맞춤)")
    a("")
    a(f"- 600단위(글 {ag['by_kind']['글']['n']}·댓글 {ag['by_kind']['댓글']['n']}). 전체 일치 {pct(ag['agree_all'])}, 전체 κ {f(ag['kappa_all'])} "
      f"(부트스트랩 2,000회 95% {f(ag['kappa_all_ci95'][0])}~{f(ag['kappa_all_ci95'][1])}). 글 κ {f(ag['by_kind']['글']['kappa'])}, 댓글 κ {f(ag['by_kind']['댓글']['kappa'])}.")
    a("- 범주별 κ는 일대다(그 범주인가 아닌가)의 Cohen κ다.")
    a("")
    a("| 범주 | κ | 95% 구간 | 1차 라벨 수 | 2차 라벨 수 | 둘 다 | 양성 일치 | κ ≥ 0.6 |")
    a("|---|---:|---|---:|---:|---:|---:|---|")
    for c in CATS:
        v = ag["per_cat"][c]
        a(f"| {c} {KO[c]} | {f(v['kappa'])} | {f(v['ci95'][0])}~{f(v['ci95'][1])} | {v['n_primary']} | {v['n_second']} | {v['n_both']} | "
          f"{f(v['pos_agree'])} | {'예' if v['kappa'] >= 0.6 else '아니오'} |")
    a("")
    a("혼동표(행 = 1차, 열 = 2차):")
    a("")
    L.extend(cm_table(ag["confusion"], CATS, "1차", "2차"))
    a("")
    conf = ag["agree_by_primary_conf"]
    a("- 1차 라벨러 확신도별 일치: " + ", ".join(f"conf {k} → {pct(v['agree'])}({v['n']}단위)" for k, v in conf.items())
      + ". 불일치는 대부분 확신 1점 경계 사례(감정 ↔ NEUT)에서 났다.")
    mt = ag["mock_target"]
    a(f"- 조롱 대상(둘 다 MOCK {mt['n_both_mock']}단위): 일치 {pct(mt['agree'])}, κ {f(mt['kappa'])}. "
      f"보조 표지: 시장 견해 κ {f(ag['aux']['stance_kappa'])}(일치 {pct(ag['aux']['stance_agree'])}), 가격 서술 κ {f(ag['aux']['px_kappa'])}(일치 {pct(ag['aux']['px_agree'])}).")
    a("")

    # 라벨 비율
    a("## 2. 라벨로 본 범주 비율 (분류기 없이)")
    a("")
    a("- '그대로' = 2,000단위 단순 비율(글·댓글 1:1로 뽑아 글이 과대). '층 가중' = 층(월 × 시간대 × 글/댓글 × 길이)마다 틀 크기 ÷ 표본 크기로 가중한 틀 전체 추정. 급락일을 빼지 않은 7/13~10/15 전체 평균이다.")
    a("")
    a("| 범주 | 그대로 | 층 가중(전체) | 층 가중(글) | 층 가중(댓글) | 감정 단위 안(전체) | 감정 단위 안(글) | 감정 단위 안(댓글) |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|")
    for c in CATS:
        e = lambda k: pct(ls[k]["weighted_emo"][c]) if c != "NEUT" else "—"  # noqa: E731
        a(f"| {c} {KO[c]} | {pct(ls['all']['raw'][c])} | {pct(ls['all']['weighted'][c])} | {pct(ls['글']['weighted'][c])} | "
          f"{pct(ls['댓글']['weighted'][c])} | {e('all')} | {e('글')} | {e('댓글')} |")
    a("")
    a("조롱 대상(라벨, 층 가중, MOCK 단위 안): " + ", ".join(
        f"{TKO[t]} 전체 {pct(ls['all']['weighted_mock_target'][t])}·글 {pct(ls['글']['weighted_mock_target'][t])}·댓글 {pct(ls['댓글']['weighted_mock_target'][t])}"
        for t in TGTS) + ".")
    a("")

    # 학습 자료
    tr = o["train"]
    a("## 3. 학습 자료")
    a("")
    a(f"- 2,000단위 1차 라벨. 이중 라벨 {tr['n_double']} 가운데 두 범주가 다른 {tr['n_disagree_excluded']}단위는 '불일치'로 학습·평가에서 뺐다 → {tr['n_usable']}단위.")
    a("- 원문은 `sample_key.parquet`의 `unit_idx`로 `units.parquet`에서 가져와 `sample_a.unit_text`(가림)를 다시 적용했고, 2,000단위 모두 라벨러가 본 글의 SHA-256과 같았다. "
      "분류기 입력은 이 가린 글이고, 자리표시 [숫자]·[사건]·[시각]·[링크]는 사용 영역 글자 하나씩(U+E001~E004)으로 바꿨다. 적용 단계도 같은 함수를 쓴다.")
    a(f"- 범주 층화 80/20(시드 {o['seed']}): 학습 {tr['split']['train']}, 떼어 둠 {tr['split']['heldout']}.")
    a("")
    a("| 범주 | 쓸 수 있는 단위 | 떼어 둠 |")
    a("|---|---:|---:|")
    for c in CATS:
        a(f"| {c} | {tr['usable_by_cat'][c]} | {tr['split']['heldout_by_cat'][c]} |")
    a("")

    # 분류기
    a("## 4. 분류기")
    a("")
    a("- 설정(결과를 보기 전에 고정): 문자 2~4그램 TF-IDF(analyzer='char', sublinear_tf, min_df=2) + 다항 로지스틱 회귀(lbfgs, L2, class_weight 없음, max_iter 5000), "
      f"scikit-learn {o['sklearn']}. C는 학습 80% 안 5겹 층화 교차검증 macro-F1로만 골랐다(1e-2~1e2 로그 격자 9개). 특징 수 {cl['n_features']:,}.")
    a("")
    a("| C | 5겹 macro-F1 평균 | SD |")
    a("|---:|---:|---:|")
    for r in cl["cv"]:
        a(f"| {r['C']:.4g} | {f(r['macro_f1_mean'])} | {f(r['macro_f1_sd'])} |")
    a("")
    a(f"- 고른 C = {cl['C_best']:.4g}(격자 상한). C ≤ 0.316의 macro-F1 0.110(SD 0)은 모두 NEUT로 찍을 때의 값과 같다. 격자는 지시대로 넓히지 않았다.")
    a("- 배포 모형은 80%로 학습한 모형이다(떼어 둔 20% 정밀도가 그대로 그 모형의 값이 되게).")
    a("")
    a(f"떼어 둔 {ho['n']}단위(정확도 {f(ho['accuracy'])}, macro-F1 {f(ho['macro_f1'])}, 라벨 대비 κ {f(ho['kappa_vs_label'])}):")
    a("")
    a("| 범주 | 정밀도 | 맞음/예측 | Wilson 95% | 재현율 | F1 | 정답 수 | 라벨러 κ | 정밀도 ≥ 0.6 | κ ≥ 0.6 | 기준 통과 |")
    a("|---|---:|---:|---|---:|---:|---:|---:|---|---|---|")
    for c in CATS:
        pc = ho["per_cat"][c]
        k, n, lo, hi = prec_ci[c]
        g = gate[c]
        a(f"| {c} {KO[c]} | {f(pc['precision'])} | {k}/{n} | {f(lo, 2)}~{f(hi, 2)} | {f(pc['recall'])} | {f(pc['f1'])} | {pc['support']} | "
          f"{f(g['kappa'])} | {'예' if g['precision_ok'] else '아니오'} | {'예' if g['kappa_ok'] else '아니오'} | "
          f"{'통과' + ('(불안정)' if (not math.isnan(lo) and lo < 0.6) else '') if g['pass'] else '미달'} |")
    a("")
    a("- 예측 0건인 범주(JOY·FEAR)는 정밀도를 정할 수 없어 미달로 셌다.")
    a("")
    a("혼동표(행 = 라벨, 열 = 분류기):")
    a("")
    L.extend(cm_table(cl["heldout_confusion"], CATS, "라벨", "분류기"))
    a("")
    nv = cl["heldout_no_overlap"]
    a(f"- 겹침(이중 라벨) 단위를 뺀 떼어 둔 {nv['n']}단위: 정확도 {f(nv['accuracy'])}, macro-F1 {f(nv['macro_f1'])}. 정밀도 "
      + ", ".join(f"{c} {f(nv['per_cat'][c]['precision'], 2)}({nv['per_cat'][c]['n_pred']}건)" for c in CATS) + ". 겹침 단위는 두 라벨이 같은 것만 남아 쉬울 수 있어 따로 쟀다.")
    sh = cl["heldout_share"]
    a("")
    a("떼어 둔 표본의 범주 몫(라벨 / 분류기 주 범주 / 분류기 확률 평균):")
    a("")
    a("| 범주 | 라벨 | 주 범주 | 확률 평균 |")
    a("|---|---:|---:|---:|")
    for c in CATS:
        a(f"| {c} | {pct(sh['label'][c])} | {pct(sh['pred_argmax'][c])} | {pct(sh['pred_proba_mean'][c])} |")
    a("")
    a("### 조롱 대상 분류기 (MOCK 단위만, 같은 설정)")
    a("")
    a(f"- 학습 {tc['n_train']}(기타 {tc['train_by_target']['other']}·강세론 조롱 {tc['train_by_target']['bull']}·약세론 조롱 {tc['train_by_target']['bear']}), "
      f"두 라벨러 대상이 달라 뺀 {tc['n_target_disagree_excluded']}단위, 떼어 둠 {tc['n_heldout']}(정답 MOCK 단위). C = {tc['C_best']:.4g}(격자 상한). "
      f"정확도 {f(tc['heldout']['accuracy'])}, macro-F1 {f(tc['heldout']['macro_f1'])}.")
    a("")
    a("| 대상 | 정밀도 | 맞음/예측 | Wilson 95% | 재현율 | 정답 수 | 기준 |")
    a("|---|---:|---:|---|---:|---:|---|")
    for t in TGTS:
        pc = tc["heldout"]["per_cat"][t]
        n = pc["n_pred"]
        k = round(pc["precision"] * n) if pc["precision"] is not None else 0
        lo, hi = wilson(k, n)
        ok = pc["precision"] is not None and pc["precision"] >= 0.6 and mt["kappa"] >= 0.6
        a(f"| {TKO[t]} | {f(pc['precision'])} | {k}/{n} | {f(lo, 2)}~{f(hi, 2)} | {f(pc['recall'])} | {pc['support']} | "
          f"{'통과' + ('(불안정)' if lo < 0.6 else '') if ok else '미달'} |")
    a("")

    # 모델 지문
    a("## 5. 모델 지문 (가격 연결 전 동결용)")
    a("")
    a("| 파일 | SHA-256 |")
    a("|---|---|")
    for k, v in o["model_sha256"].items():
        a(f"| `emotion/model/{k}` | `{v}` |")
    a(f"| `emotion/model/model_card.json` | `{o['model_card_sha256']}` |")
    a("")

    # 적용
    ap = o["apply"]
    a("## 6. 전체 적용")
    a("")
    a(f"- `units.parquet` {ap['n_units']:,}단위 전체에 같은 가림 → 같은 특징으로 적용했다(떼어 둔 단위에서 적용 결과가 평가 결과와 같음을 확인: 정확도 {f(ap['heldout_reproduced_accuracy'])}). "
      "조롱 대상은 주 범주가 MOCK인 단위에만 붙였다.")
    a("")
    a("| 범주 | 전체 단위 | 글 | 댓글 | 라벨 층 가중(참고) |")
    a("|---|---:|---:|---:|---:|")
    for c in CATS:
        a(f"| {c} | {pct(ap['pred_share_all_units'][c])} | {pct(ap['pred_share_by_utype']['post'][c])} | "
          f"{pct(ap['pred_share_by_utype']['comment'][c])} | {pct(ls['all']['weighted'][c])} |")
    a("")
    a("- 분류기는 감정을 라벨보다 크게 덜 센다(감정 전체 "
      f"{pct(1 - ap['pred_share_all_units']['NEUT'])} 대 라벨 {pct(1 - ls['all']['weighted']['NEUT'])}). 수준값은 쓰지 말고 창 사이 상대 변화만 볼 수 있다.")
    kr = o["windows"]["kr"]["stats"]
    a("- 감정판 구성도 라벨과 다르다(국내 창 평균 대 라벨 층 가중): " + ", ".join(
        f"{c} {pct(kr[f'r_emo_{c}']['mean'])} 대 {pct(ls['all']['weighted_emo'][c])}" for c in EMO6)
      + ". 창 셈은 (작성자, 범주) 쌍이라 단위 셈과 분모가 다르지만, 분류기가 MOCK·ANGER는 덜 놓치고 JOY·FEAR는 거의 못 잡는 쏠림이 크다.")
    a("")
    a("### 창별 범주 비율 요약 (그림 없음, 가격과 나란히 놓지 않음)")
    a("")
    a("- 창 = 로더 본판(W_KR·W_U) 가운데 완전한 창. 창 안 단위를 (작성자, 범주) 쌍으로 한 번씩 센다. 전체판 분모 = 7범주 쌍 합, 감정판 분모 = NEUT를 뺀 6범주 쌍 합. "
      "들뜸 지수 = (HYPE + JOY) − (DESP + FEAR) 감정판. 엔트로피 = 감정판 6범주 −Σp ln p(정규화 = ÷ ln 6). 조롱 대상 몫 = 주 범주 MOCK 단위의 (작성자, 대상) 쌍 기준.")
    a("- 장애 창(미국 9/24·9/25)은 비율을 결측으로 두었다. 표는 유효 창만의 평균·SD·최소·최대다(창 사이 분포).")
    a("")
    W = o["windows"]
    for mkt, nm in (("kr", "국내 W_KR"), ("us", "미국 W_U")):
        w = W[mkt]
        st = w["stats"]
        a(f"**{nm}**: 창 {w['n_windows']}개(유효 {w['n_valid']}, 96시간 초과 {w['n_over96']}), D {w['first_D']} ~ {w['last_D']}. "
          f"창당 (작성자, 범주) 쌍 평균 {st['n_pair_all']['mean']:.0f}(감정 {st['n_pair_emo']['mean']:.0f}, 최소 {st['n_pair_emo']['min']:.0f}).")
        a("")
        a("| 값 | 평균 | SD | 최소 | 최대 |")
        a("|---|---:|---:|---:|---:|")
        rows = ([(f"전체판 {c}", f"r_all_{c}", True) for c in CATS] + [(f"감정판 {c}", f"r_emo_{c}", True) for c in EMO6]
                + [("들뜸 지수", "hype_index", False), ("감정 엔트로피(정규화)", "entropy_emo_norm", False)]
                + [(f"조롱 대상 몫 {TKO[t]}", f"mock_share_{t}", True) for t in TGTS]
                + [("약세론 조롱 − 강세론 조롱", "mock_bear_minus_bull", False),
                   ("창당 약세론 조롱 쌍 수", "n_mock_bear", None), ("창당 강세론 조롱 쌍 수", "n_mock_bull", None)])
        for lab, col, is_pct in rows:
            v = st[col]
            if is_pct is None:
                a(f"| {lab} | {v['mean']:.1f} | {v['sd']:.1f} | {v['min']:.0f} | {v['max']:.0f} |")
            elif is_pct:
                a(f"| {lab} | {pct(v['mean'])} | {pct(v['sd'])} | {pct(v['min'])} | {pct(v['max'])} |")
            else:
                a(f"| {lab} | {f(v['mean'])} | {f(v['sd'])} | {f(v['min'])} | {f(v['max'])} |")
        a("")
    a("- 창당 약세론 조롱 쌍은 중앙값 1개다. H63(약세론 − 강세론 조롱 몫)은 분류기 미달과 함께 표본도 모자란다.")
    a("- 보조 열 `ash_<범주>`(그 범주 단위를 한 번이라도 쓴 고유 작성자 ÷ 창 안 고유 작성자, §2.0 X_A 꼴)도 `window_emotion.parquet`에 있다.")
    a("")

    # 검수지
    rv = o["review"]
    a("## 7. 사용자 검수지")
    a("")
    a(f"- `emotion/review_200.json`: 급락일({', '.join(rv['excluded_days'])})과 묶음 A 표본 단위를 뺀 같은 틀에서 글 {rv['by_kind']['글']}·댓글 {rv['by_kind']['댓글']} 단순 무작위. "
      "항목 = `id`, `text`(가린 글), `model_cat`, `model_cat_ko`, 빈 칸 `user_ok`·`user_cat`. 대응표는 `review_key.parquet`(사용자에게 주지 않아도 됨).")
    a("- 분류기 주 범주 분포: " + ", ".join(f"{c} {rv['model_cat'][c]}" for c in CATS) + ". 대부분 NEUT라 이 검수는 'NEUT라고 한 것이 정말 중립인가'를 주로 재게 된다.")
    a("- 표시 방법: 맞으면 `user_ok: true`, 틀리면 `user_ok: false`와 `user_cat`에 맞는 코드(HYPE·JOY·MOCK·DESP·ANGER·FEAR·NEUT). 일치 75% 이상이어야 쓴다(부록 3 4단계).")
    a("- 감사 보완(`emotion/audit.md`): 위 검수지는 분류기 범주만 보여 주므로 3단계 AI 라벨(§2 비율)을 확인하지 못한다. "
      "AI 라벨 검수지 `emotion/review_label_200.json`(묶음 A에서 급락일을 뺀 글 100·댓글 100, AI 라벨 표시, 대응표 `review_label_key.parquet`)을 추가했다. "
      "부록 3 순서(3 라벨 → 4 검수 → 5 분류기)상 4단계 검수는 이쪽이다.")
    a("")

    # 판정 칸 영향
    a("## 8. 부록 3 판정 칸에 주는 영향 (품질 기준을 그대로 적용한 결과, 결정은 아님)")
    a("")
    a("| id | 쓰는 값 | 범주 기준 | 상태 |")
    a("|---|---|---|---|")
    st_ = lambda c: ("통과(불안정)" if gate[c]["pass"] and prec_ci[c][2] < 0.6 else "통과" if gate[c]["pass"] else "미달")  # noqa: E731
    a(f"| H60 | HYPE 비율 | HYPE {st_('HYPE')} | 판정 가능(정밀도 {prec_ci['HYPE'][0]}/{prec_ci['HYPE'][1]}, 재현율 {pct(ho['per_cat']['HYPE']['recall'], 0)}) |")
    a(f"| H61 | DESP 비율 | DESP {st_('DESP')} | 설명용 |")
    a(f"| H62 | ANGER 비율 | ANGER {st_('ANGER')} | 판정 가능(정밀도 {prec_ci['ANGER'][0]}/{prec_ci['ANGER'][1]}, 재현율 {pct(ho['per_cat']['ANGER']['recall'], 0)}) |")
    a(f"| H63 | 약세론 − 강세론 조롱 몫 | MOCK {st_('MOCK')}, 대상 분류기 미달 | 설명용 |")
    a(f"| H64 | FEAR 비율 | FEAR {st_('FEAR')}(예측 0건) | 설명용 |")
    a("| H65 | 감정 6범주 엔트로피 | 6범주 가운데 JOY·DESP·FEAR 미달 | 설명용으로 읽음(미달 범주가 값에 들어감) |")
    a("| 단계표 기본 점수 | 들뜸 지수 (HYPE + JOY) − (DESP + FEAR) | JOY·DESP·FEAR 미달 | 설명용으로 읽음 |")
    a("")
    a("- 위 상태는 사용자 검수(일치 75%) 전이다. 검수가 75%에 못 미치면 모든 칸이 설명용이다.")
    a("- H65·들뜸 지수를 '설명용'으로 읽은 것은 해석이다. 부록 3은 범주 단위로만 기준을 정했다.")
    a("")

    # 한계
    a("## 9. 한계와 다음 단계 후보")
    a("")
    a("- 학습 1,563단위에 class_weight 없이 다항 회귀를 쓰자 다수 범주(NEUT)로 쏠렸다. C도 격자 상한에서 골랐다. 설정을 바꾸려면(범주 가중, 격자 확장, 확률 합 비율 등) "
      "새 설정으로 기록하고, 이번 떼어 둔 20%는 이미 봤으므로 묶음 B의 새 떼어 둔 표본으로 다시 재야 한다.")
    a("- 라벨 κ는 같은 AI 라벨러 두 번의 일치라 높게 나오기 쉽다. 타당성은 사용자 검수로만 확인된다.")
    a("- 단위별 범주 확률(`p_*`)은 `pred_units.parquet`에 있다. 이번 창 비율은 지시대로 주 범주만 썼다.")
    a("- 시장 견해·가격 서술 표지는 분류기로 만들지 않았다(라벨에만 있음).")
    a("")

    a("## 파일")
    a("")
    a("| 파일 | 내용 |")
    a("|---|---|")
    a("| `emotion/code/emotion_batchA.py` | 일치도·학습·분류기·적용·창·검수지 |")
    a("| `emotion/code/emotion_batchA_report.py` | 이 보고서 |")
    a("| `emotion/emotion_batchA_results.json` | 모든 숫자(CV 겹별 점수, 혼동표, 창 요약) |")
    a("| `emotion/train_set.parquet` | 2,000단위 학습 표(라벨 2개, 불일치 표시, 나눔, 가린 글) |")
    a("| `emotion/model/` | 벡터라이저·회귀(범주, 조롱 대상) joblib 4개 + model_card.json |")
    a("| `emotion/pred_units.parquet` | 271,595단위 범주 확률·주 범주·조롱 대상 |")
    a("| `emotion/window_emotion.parquet` | 창별(국내 61·미국 67) 범주 비율 두 판, 들뜸 지수, 엔트로피, 조롱 대상 몫 |")
    a("| `emotion/review_200.json` / `review_key.parquet` | 사용자 검수지 / 대응표 |")
    a("")
    a("파일 SHA-256: " + ", ".join(f"`{k}` `{v[:16]}…`" for k, v in o["files"].items()) + ".")
    (EMO / "emotion_batchA.md").write_text("\n".join(L) + "\n", "utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
