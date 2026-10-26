"""nine_batchA.json → nine_batchA.md (숫자는 모두 JSON에서 읽는다)."""
import json
from pathlib import Path

EMO = Path(r"C:/Users/geunt/OneDrive/Desktop/주식/data/derived/analysis/ant_index/emotion")
d = json.loads((EMO / "nine_batchA.json").read_text("utf-8"))
E9 = ["HYPE", "JOY", "MOCK", "DESP", "ANGER", "FEAR", "HOPE", "REGRET", "ENVY"]
KO = {"HYPE": "들뜸·자랑", "JOY": "기쁨·안도", "MOCK": "조롱·비꼼", "DESP": "우울·절망", "ANGER": "분노·욕",
      "FEAR": "불안·공포", "HOPE": "바람·기원", "REGRET": "후회·자책", "ENVY": "부러움·소외", "NEUT": "중립"}
PRIM = ["MOCK", "ANGER", "DESP", "FEAR", "REGRET", "HOPE", "ENVY", "HYPE", "JOY", "NEUT"]
DKO = {"none": "없음 0", "down": "하락 1·2", "amb": "모호 3", "up": "상승 4·5"}


def pc(x, k=1):
    return "—" if x is None else f"{x * 100:.{k}f}%"


def f3(x):
    return "—" if x is None else f"{x:.3f}"


def rng(v):
    return "—" if not v else f"{v[0]:.2f}~{v[1]:.2f}"


ag = d["agreement"]
A = d["distribution"]["all"]
G = d["distribution"]["글"]
C = d["distribution"]["댓글"]
RAW = d["distribution"]["raw"]
V = d["vs_first_labels"]
R = d["review200_ai"]
pe = ag["per_emotion"]
L = []
w = L.append

kp = [pe[e]["kappa_presence"] for e in E9]
kw = [pe[e]["kappa_intensity_qw"] for e in E9]
hp = A["hope_pred"]

w("# 감정 9가지 × 예측 묶음 A: 일치도·분포 (2026-10-17, 가격 연결 전)")
w("")
w("> 정의는 `HYPOTHESES_addendum6.md`(감정 9가지)·`addendum5.md`(다감정 세기 0~3)·`addendum4.md`(동결, 2층 예측 0~5)를 따른다. 측정만 했고 가격·지수·수익 자료는 읽지도 계산하지도 않았다.  ")
w("> 입력 `emotion/labels4/` primary_01~10(2,000단위)·second_1~3(600단위)·review200_ai(200단위), 1차 라벨 `emotion/labels/primary_*.json`. 전체 숫자 `emotion/nine_batchA.json`.  ")
w("> 층 가중은 묶음 A 틀(7/13~10/15 글·마감 전 댓글 252,246단위)의 층(월 × 시간대 × 글/댓글 × 길이)마다 틀 크기 ÷ 표본 크기다. κ의 95% 구간은 단위 부트스트랩 2,000회(시드 20261017)다.")
w("")
w("## 결론")
w("")
w(f"- **바람(HOPE)은 따로, 일관되게 잡힌다**: 층 가중 {pc(A['emotion']['HOPE']['share_ge1'])}(글 {pc(G['emotion']['HOPE']['share_ge1'])}·댓글 {pc(C['emotion']['HOPE']['share_ge1'])})에 HOPE가 1 이상이다. 두 라벨러 κ {f3(pe['HOPE']['kappa_presence'])}(세기 가중 {f3(pe['HOPE']['kappa_intensity_qw'])}). HOPE 단위의 {pc(hp['none'])}는 예측 0이라 바람과 예측(2층)은 다른 축이다. 옛 6감정이 모두 0이고 새 세 감정만 있는 단위가 {pc(A['only_new3'])}({A['n_only_new3']}단위)로, 6감정 체계였다면 중립으로 떨어졌을 몫이다. 1차 라벨(7범주)에서 NEUT였던 단위의 {pc(V['emotion_ge1_within_first_neut']['HOPE']['weighted'])}(가중)에 이번에 HOPE가 붙었다.")
w(f"- **라벨러 일치도는 높다**: 9감정 모두 있음/없음 κ {min(kp):.2f}~{max(kp):.2f}, 세기 가중 κ {min(kw):.2f}~{max(kw):.2f}로 0.6을 넘는다. 주된 감정 일치 {pc(ag['primary']['agree'])}·κ {f3(ag['primary']['kappa'])}. 예측 이차 가중 κ {f3(ag['pred']['qw_kappa_with0'])}(0 포함)·{f3(ag['pred']['qw_kappa_1to5'])}(1~5, {ag['pred']['n_both_nonzero']}단위), 방향 일치 {pc(ag['pred']['dir_agree'])}. 다만 ENVY는 1차 {pe['ENVY']['n_primary']}·2차 {pe['ENVY']['n_second']}단위뿐이라 κ 95% 하한이 {pe['ENVY']['kappa_presence_ci95'][0]:.2f}다. 두 라벨러가 같은 AI라 '같은 기준으로 읽는다'는 뜻이고, 정식 기준은 사용자 검수 200건과의 비교다(부록 5).")
top = A["top_mixed_combos"][0]
w(f"- **분포(층 가중, 감정 ≥ 1)**: 조롱 {pc(A['emotion']['MOCK']['share_ge1'])}, 분노 {pc(A['emotion']['ANGER']['share_ge1'])}, 우울 {pc(A['emotion']['DESP']['share_ge1'])}, 불안 {pc(A['emotion']['FEAR']['share_ge1'])}, 바람 {pc(A['emotion']['HOPE']['share_ge1'])}, 들뜸 {pc(A['emotion']['HYPE']['share_ge1'])}, 기쁨 {pc(A['emotion']['JOY']['share_ge1'])}, 후회 {pc(A['emotion']['REGRET']['share_ge1'])}, 부러움 {pc(A['emotion']['ENVY']['share_ge1'])}. 중립(9감정 모두 0) {pc(A['neutral'])}. 섞인 감정 {pc(A['mixed'])}(감정 단위 안 {pc(A['mixed_among_emotional'])}), 가장 흔한 조합은 조롱+분노({pc(top['share_all'], 2)}). 예측 몫 {pc(A['pred_share'])}, 군중 예측 평균 {A['crowd_mean_P']:+.2f}(하락 쪽).")
w(f"- **조롱이 늘어난 이유**: 조롱 ≥ 1 {pc(A['emotion']['MOCK']['share_ge1'])} 가운데 세기 1(말투에 묻은 가벼운 비꼼)이 {pc(A['emotion']['MOCK']['intensity_share']['1'])}p다. 세기 2 이상은 {pc(A['emotion']['MOCK']['intensity_share']['2'] + A['emotion']['MOCK']['intensity_share']['3'])}로, 1차 7범주의 조롱 비율(13.7%)보다 낮다.")
FU = d["frame_units"]
w(f"- **드문 칸**: 불안한 기쁨 {pc(A['anx_joy'], 2)}({A['n_anx_joy']}단위), 분노·절망 {pc(A['anger_desp'], 2)}({A['n_anger_desp']}단위), 후회×하락 {A['emo_x_pred']['REGRET×down']['n']}단위·후회×상승 {A['emo_x_pred']['REGRET×up']['n']}단위, 부러움 {A['emotion']['ENVY']['n_ge1']}단위. 틀 전체(252,246단위)로 환산하면 불안한 기쁨 약 {A['anx_joy'] * FU:,.0f}, 후회×하락 약 {A['emo_x_pred']['REGRET×down']['share_all'] * FU:,.0f}, 후회×상승 약 {A['emo_x_pred']['REGRET×up']['share_all'] * FU:,.0f}, 부러움 약 {A['emotion']['ENVY']['share_ge1'] * FU:,.0f}단위다. 창 값을 날마다 읽은 작은 표본으로 내면 이런 칸은 창마다 0건이 많을 수 있다. H71·H74·H75는 가격 전 점검의 결측률(30%) 기준을 먼저 확인해야 한다(그 점검은 아직 하지 않았다).")
w(f"- **1차 라벨과 비교**: 1차 NEUT {V['n_first_neut']:,}단위 중 {pc(V['tagged_emo_or_pred']['raw'])}(가중 {pc(V['tagged_emo_or_pred']['weighted'])})에 이번에 감정 ≥ 1 또는 예측이 붙었다(감정만 {pc(V['emo_only']['raw'])}, 예측만 {pc(V['pred_only']['raw'])}, 둘 다 {pc(V['emo_and_pred']['raw'])}). 가장 많이 붙은 것은 조롱({V['emotion_ge1_within_first_neut']['MOCK']['n']}단위)이다. 거꾸로 1차 감정 단위가 이번에 중립이 된 것은 {V['first_emotional_now_neutral']['n']}단위뿐이다.")
w("")

# ---------------- 1
w("## 1. 일치도 (조각 01·05·09 1차 × second_1~3, id로 맞춤)")
w("")
w(f"- {ag['n']}단위(글 {ag['by_kind_n']['글']}·댓글 {ag['by_kind_n']['댓글']}). 있음/없음은 세기 ≥ 1 여부의 Cohen κ, 세기는 0~3 이차 가중 κ다. 0이 많아 단순 일치율은 높게 나오므로 κ로 읽는다.")
w("")
w("| 감정 | 1차 ≥1 | 2차 ≥1 | 둘 다 | 있음/없음 일치 | κ(있음) | 95% | 세기 가중 κ | 95% | 세기 완전 일치 | κ ≥ 0.6 |")
w("|---|---:|---:|---:|---:|---:|---|---:|---|---:|---|")
for e in E9:
    v = pe[e]
    ok = "예" if v["presence_ok_0.6"] and v["intensity_ok_0.6"] else "아니오"
    lo = v["kappa_presence_ci95"][0] if v["kappa_presence_ci95"] else None
    if ok == "예" and lo is not None and lo < 0.6:
        ok = "예(하한 < 0.6)"
    w(f"| {e} {KO[e]} | {v['n_primary']} | {v['n_second']} | {v['n_both']} | {pc(v['agree_presence'])} | {f3(v['kappa_presence'])} | {rng(v['kappa_presence_ci95'])} | {f3(v['kappa_intensity_qw'])} | {rng(v['kappa_intensity_qw_ci95'])} | {pc(v['agree_intensity_exact'])} | {ok} |")
w("")
w("글·댓글별 κ(있음 / 세기 가중):")
w("")
w("| 감정 | 글 | 댓글 |")
w("|---|---|---|")
for e in E9:
    b = pe[e]["by_kind"]
    w(f"| {e} | {f3(b['글']['kappa_presence'])} / {f3(b['글']['kappa_intensity_qw'])} | {f3(b['댓글']['kappa_presence'])} / {f3(b['댓글']['kappa_intensity_qw'])} |")
w("")
p = ag["primary"]
w(f"- **주된 감정**(가장 센 것, 같으면 MOCK > ANGER > DESP > FEAR > REGRET > HOPE > ENVY > HYPE > JOY, 모두 0이면 중립): 일치 {pc(p['agree'])}, κ {f3(p['kappa'])} (95% {rng(p['kappa_ci95'])}). 글 {pc(p['by_kind']['글']['agree'])}·κ {f3(p['by_kind']['글']['kappa'])}, 댓글 {pc(p['by_kind']['댓글']['agree'])}·κ {f3(p['by_kind']['댓글']['kappa'])}. 일치 75% 기준(부록 5)을 넘는다.")
w(f"- 중립 여부: 일치 {pc(ag['neutral']['agree'])}, κ {f3(ag['neutral']['kappa'])} (1차 {ag['neutral']['n_primary']}·2차 {ag['neutral']['n_second']}). 섞인 감정 여부: 일치 {pc(ag['mixed']['agree'])}, κ {f3(ag['mixed']['kappa'])} (1차 {ag['mixed']['n_primary']}·2차 {ag['mixed']['n_second']}).")
w(f"- 조롱 대상(둘 다 MOCK ≥ 1인 {ag['mock_target']['n_both_mock']}단위): 일치 {pc(ag['mock_target']['agree'])}, κ {f3(ag['mock_target']['kappa'])}. 예측 대상(둘 다 예측이 있는 {ag['pred_target']['n_both_pred']}단위): 일치 {pc(ag['pred_target']['agree'])}, κ {f3(ag['pred_target']['kappa'])}.")
w("")
w("주된 감정 혼동표(행 = 1차, 열 = 2차):")
w("")
w("| 1차 ＼ 2차 | " + " | ".join(PRIM) + " | 합 |")
w("|---|" + "---:|" * (len(PRIM) + 1))
for r in PRIM:
    row = p["confusion"][r]
    w(f"| {r} | " + " | ".join(str(row[c]) for c in PRIM) + f" | {sum(row[c] for c in PRIM)} |")
w("")
w("- 가장 흔한 어긋남은 조롱↔분노(15단위)이고, 불안 → 중립 8, 바람 → 들뜸 4, 들뜸 → 기쁨 4, 후회 → 우울 3이다.")
w("")
pr = ag["pred"]
w("**2층 예측**")
w("")
w("| 지표 | 값 | 95% | 기준 |")
w("|---|---:|---|---|")
w(f"| 이차 가중 κ (0 포함, 600단위) | {f3(pr['qw_kappa_with0'])} | {rng(pr['qw_kappa_with0_ci95'])} | ≥ 0.6 {'통과' if pr['qw_ok_0.6_with0'] else '미달'} |")
w(f"| 이차 가중 κ (둘 다 1~5, {pr['n_both_nonzero']}단위) | {f3(pr['qw_kappa_1to5'])} | {rng(pr['qw_kappa_1to5_ci95'])} | ≥ 0.6 {'통과' if pr['qw_ok_0.6_1to5'] else '미달'} |")
w(f"| 방향 일치 (하락 1·2 / 3 / 상승 4·5 / 없음 0) | {pc(pr['dir_agree'])} | — | ≥ 75% {'통과' if pr['dir_agree_ok_0.75'] else '미달'} |")
w(f"| 방향 κ (4값) | {f3(pr['dir_kappa'])} | — | — |")
w(f"| 6값 완전 일치 | {pc(pr['exact_agree_with0'])} | — | — |")
w(f"| 예측 있음/없음 κ | {f3(pr['has_pred_kappa'])} | — | 1차 {pr['n_pred_primary']}·2차 {pr['n_pred_secondary'] if 'n_pred_secondary' in pr else pr['n_pred_second']}단위 |")
w("")
w("방향 혼동표(행 = 1차, 열 = 2차):")
w("")
DIRS = ["none", "down", "amb", "up"]
w("| 1차 ＼ 2차 | " + " | ".join(DKO[x] for x in DIRS) + " |")
w("|---|" + "---:|" * 4)
for r in DIRS:
    w(f"| {DKO[r]} | " + " | ".join(str(pr["dir_confusion"][r][c]) for c in DIRS) + " |")
w("")
w("- 어긋남은 대부분 '예측 있음 ↔ 없음'(없음 → 하락 10, 하락 → 없음 7, 없음 → 상승 5, 상승 → 없음 5)이다. 둘 다 예측이 있으면 방향 일치가 " + pc(pr["dir_agree_both_nonzero"]) + "다.")
bc = ag["by_primary_conf"]
w(f"- 1차 라벨러 확신도별: conf 1 → 주된 감정 {pc(bc['1']['primary_agree'])}·방향 {pc(bc['1']['dir_agree'])}({bc['1']['n']}단위), conf 2 → {pc(bc['2']['primary_agree'])}·{pc(bc['2']['dir_agree'])}({bc['2']['n']}), conf 3 → {pc(bc['3']['primary_agree'])}·{pc(bc['3']['dir_agree'])}({bc['3']['n']}).")
w("")

# ---------------- 2
w("## 2. 분포 (1차 라벨 2,000단위, 층 가중)")
w("")
w("- 급락일을 빼지 않은 7/13~10/15 전체 평균이다. '그대로'는 2,000단위 단순 비율(글·댓글 1:1로 뽑아 글이 과대)이다.")
w("")
w("| 감정 | ≥1 몫 | 세기 ≥2 몫 | 평균 세기 | 글 ≥1 | 댓글 ≥1 | 그대로 ≥1 | 단위 수 | 주된 감정 몫 |")
w("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for e in E9:
    v = A["emotion"][e]
    ge2 = v["intensity_share"]["2"] + v["intensity_share"]["3"]
    w(f"| {e} {KO[e]} | {pc(v['share_ge1'])} | {pc(ge2)} | {v['mean_intensity']:.3f} | {pc(G['emotion'][e]['share_ge1'])} | {pc(C['emotion'][e]['share_ge1'])} | {pc(RAW['emotion'][e]['share_ge1'])} | {v['n_ge1']} | {pc(A['primary_emotion'][e])} |")
w(f"| 중립(모두 0) | {pc(A['neutral'])} | — | — | {pc(G['neutral'])} | {pc(C['neutral'])} | {pc(RAW['neutral'])} | {A['n_neutral']} | {pc(A['primary_emotion']['NEUT'])} |")
w("")
nd = A["n_emo_dist"]
w(f"- 한 단위의 감정 개수(가중): 0개 {pc(nd['0'])}, 1개 {pc(nd['1'])}, 2개 {pc(nd['2'])}, 3개 {pc(nd['3'])}, 4개 이상 {pc(sum(v for k, v in nd.items() if int(k) >= 4))}. 감정 단위의 평균 감정 수 {A['mean_n_emo_among_emotional']:.2f}개.")
w(f"- 섞인 감정 몫(두 감정 이상 ≥ 1): 전체 {pc(A['mixed'])}(글 {pc(G['mixed'])}·댓글 {pc(C['mixed'])}), 감정 단위 안 {pc(A['mixed_among_emotional'])}. 서로 다른 섞인 조합 {A['n_distinct_mixed_combos']}가지.")
w(f"- 불안한 기쁨(JOY 또는 HYPE ≥ 1이면서 FEAR ≥ 1): {pc(A['anx_joy'], 2)}({A['n_anx_joy']}단위; 글 {pc(G['anx_joy'], 2)}·댓글 {pc(C['anx_joy'], 2)}).")
w(f"- 분노·절망(ANGER와 DESP ≥ 1): {pc(A['anger_desp'], 2)}({A['n_anger_desp']}단위; 글 {pc(G['anger_desp'], 2)}·댓글 {pc(C['anger_desp'], 2)}).")
w(f"- 새 세 감정(HOPE·REGRET·ENVY) 중 하나라도 ≥ 1: {pc(A['new3_any'])}. 옛 6감정은 모두 0이고 새 감정만 있는 단위: {pc(A['only_new3'])}({A['n_only_new3']}단위), 그중 HOPE 하나만 있는 단위 {pc(A['only_hope'])}({A['n_only_hope']}단위).")
w(f"- 조롱 대상(MOCK ≥ 1 안, 가중): 기타 {pc(A['mock_target']['other'])}, 강세론 조롱 {pc(A['mock_target']['bull'])}, 약세론 조롱 {pc(A['mock_target']['bear'])}.")
w(f"- 들뜸 지수(다감정판, 단위 평균 (HYPE + JOY) − (DESP + FEAR)): {A['elation_mean']:+.3f}.")
w("")
w("흔한 섞인 조합 상위 10(층 가중 순):")
w("")
w("| 순위 | 조합 | 단위 수 | 전체 몫 | 섞인 단위 안 몫 |")
w("|---:|---|---:|---:|---:|")
for i, c in enumerate(A["top_mixed_combos"], 1):
    w(f"| {i} | {c['combo'].replace('+', ' + ')} | {c['n']} | {pc(c['share_all'], 2)} | {pc(c['share_mixed'])} |")
w("")
w("**예측(2층)**")
w("")
w("| pred | 가중 몫 | 글 | 댓글 | 단위 수 |")
w("|---|---:|---:|---:|---:|")
PN = {"0": "0 예측 없음", "1": "1 강한 하락", "2": "2 약한 하락", "3": "3 모호", "4": "4 약한 상승", "5": "5 강한 상승"}
for k in [str(i) for i in range(6)]:
    w(f"| {PN[k]} | {pc(A['pred'][k])} | {pc(G['pred'][k])} | {pc(C['pred'][k])} | {A['pred_n'][k]} |")
w("")
w(f"- 예측 몫(pred ≠ 0): {pc(A['pred_share'])}(글 {pc(G['pred_share'])}·댓글 {pc(C['pred_share'])}). 방향: 하락 {pc(A['pred_dir']['down'])}, 모호 {pc(A['pred_dir']['amb'])}, 상승 {pc(A['pred_dir']['up'])}.")
w(f"- 군중 예측 평균 P̄(1~5 평균 − 3): {A['crowd_mean_P']:+.3f}(글 {G['crowd_mean_P']:+.3f}·댓글 {C['crowd_mean_P']:+.3f}). 강한 하락(1) {pc(A['strong_down'], 2)}, 강한 상승(5) {pc(A['strong_up'], 2)}.")
w(f"- 예측 대상(예측 단위 안): 개별 종목·섹터 {pc(A['pred_target']['stock'])}, 시장·지수 {pc(A['pred_target']['market'])}, 알 수 없음 {pc(A['pred_target']['unknown'])}.")
w("")
w("**감정 × 예측**(전체 단위 대비 몫, 그 감정 단위 안 몫)")
w("")
w("| 칸 | 단위 수 | 전체 몫 | 그 감정 안 몫 | 글 전체 몫 | 댓글 전체 몫 |")
w("|---|---:|---:|---:|---:|---:|")
CK = {"MOCK×down": "조롱 × 하락(1·2)", "DESP×down": "우울 × 하락(1·2)", "HYPE×up": "들뜸 × 상승(4·5)",
      "REGRET×down": "후회 × 하락(1·2)", "REGRET×up": "후회 × 상승(4·5)", "HOPE×down": "바람 × 하락(1·2)",
      "HOPE×up": "바람 × 상승(4·5)", "ENVY×up": "부러움 × 상승(4·5)"}
for k, nm in CK.items():
    v = A["emo_x_pred"][k]
    w(f"| {nm} | {v['n']} | {pc(v['share_all'], 2)} | {pc(v['share_within_emotion'])} | {pc(G['emo_x_pred'][k]['share_all'], 2)} | {pc(C['emo_x_pred'][k]['share_all'], 2)} |")
w("")
w(f"- H69 특징의 표본 평균((조롱 × 하락) − (우울 × 하락)): {A['H69_diff_MOCKdown_minus_DESPdown'] * 100:+.2f}%p. H75 특징의 표본 평균((후회 × 하락) − (후회 × 상승)): {A['H75_diff_REGRETdown_minus_REGRETup'] * 100:+.2f}%p(후회 단위 {A['emotion']['REGRET']['n_ge1']}개 중 하락 {A['emo_x_pred']['REGRET×down']['n']}·상승 {A['emo_x_pred']['REGRET×up']['n']}). 둘 다 설명용 수준 숫자이며 가격과 나란히 놓지 않았다.")
w(f"- 바람(HOPE ≥ 1) 단위의 예측: 없음 {pc(hp['none'])}, 하락 {pc(hp['down'])}, 모호 {pc(hp['amb'])}, 상승 {pc(hp['up'])}. HOPE 단위의 {pc(A['hope_with_fear'])}는 FEAR도 1 이상이다.")
w("")

# ---------------- 3
w("## 3. 1차 라벨(7범주)과 비교")
w("")
w(f"- 1차 NEUT {V['n_first_neut']:,}단위(1차 가중 몫 {pc(V['first_neut_weighted_share'])}) 가운데 이번에 붙은 것:")
w("")
w("| 이번 라벨 | 단위 수 | 그대로 | 층 가중 |")
w("|---|---:|---:|---:|")
for k, nm in (("tagged_emo_or_pred", "감정 ≥ 1 또는 예측 ≠ 0"), ("emo_only", "감정만"), ("pred_only", "예측만"),
              ("emo_and_pred", "감정과 예측 둘 다"), ("still_empty", "여전히 감정 0·예측 0"),
              ("only_new3_within_first_neut", "새 세 감정만(옛 6감정 0)")):
    v = V[k]
    w(f"| {nm} | {v['n']} | {pc(v['raw'])} | {pc(v['weighted'])} |")
w("")
w("1차 NEUT 안에서 감정별로 1 이상이 붙은 단위:")
w("")
w("| 감정 | " + " | ".join(E9) + " |")
w("|---|" + "---:|" * 9)
ev = V["emotion_ge1_within_first_neut"]
w("| 단위 수 | " + " | ".join(str(ev[e]["n"]) for e in E9) + " |")
w("| 가중 몫 | " + " | ".join(pc(ev[e]["weighted"]) for e in E9) + " |")
w("")
w(f"- 거꾸로 1차에서 감정이었던 {2000 - V['n_first_neut']}단위 중 이번에 9감정이 모두 0이 된 것은 {V['first_emotional_now_neutral']['n']}단위다.")
w("")
w("1차 범주 → 이번 주된 감정(단위 수, 행 = 1차):")
w("")
ct = V["crosstab_first_cat_x_new_primary"]
w("| 1차 ＼ 이번 | " + " | ".join(PRIM) + " | 합 |")
w("|---|" + "---:|" * (len(PRIM) + 1))
for r in ["HYPE", "JOY", "MOCK", "DESP", "ANGER", "FEAR", "NEUT"]:
    row = ct[r]
    w(f"| {r} | " + " | ".join(str(row.get(c, 0)) for c in PRIM) + f" | {sum(row.values())} |")
w("")
fk = V["first_cat_kept"]
w("- 1차 범주가 이번에도 1 이상인 몫 / 이번 주된 감정과 같은 몫: " + ", ".join(
    f"{c} {pc(fk[c]['still_ge1'], 0)} / {pc(fk[c]['still_primary'], 0)}" for c in ["HYPE", "JOY", "MOCK", "DESP", "ANGER", "FEAR"]) + ".")
w(f"- 1차 들뜸 67단위 중 {ct['HYPE']['HOPE']}단위, 1차 기쁨 51단위 중 {ct['JOY']['HOPE']}단위의 주된 감정이 바람으로 옮겼다. 1차 우울 114단위 중 {ct['DESP']['REGRET']}단위는 후회로 옮겼다.")
w("")

# ---------------- 4
w("## 4. 검수지 AI 판정 분포 (review200_ai, 가중 없음)")
w("")
w("- 사용자 판정과의 비교는 하지 않았다(사용자 검수 뒤에 한다). 검수지는 묶음 A 층 표본이 아니므로 그대로 센 값이다.")
w("")
RA, RG, RC = R["all"], R["글"], R["댓글"]
w("| 감정 | 전체 ≥1 (200) | 글 (100) | 댓글 (100) | 평균 세기 | 주된 감정 |")
w("|---|---:|---:|---:|---:|---:|")
for e in E9:
    w(f"| {e} {KO[e]} | {RA['emotion'][e]['n_ge1']} | {RG['emotion'][e]['n_ge1']} | {RC['emotion'][e]['n_ge1']} | {RA['emotion'][e]['mean_intensity']:.3f} | {RA['primary_emotion_n'][e]} |")
w(f"| 중립 | {RA['n_neutral']} | {RG['n_neutral']} | {RC['n_neutral']} | — | {RA['primary_emotion_n']['NEUT']} |")
w("")
w(f"- 섞인 감정 {RA['n_mixed']}단위, 불안한 기쁨 {RA['n_anx_joy']}, 분노·절망 {RA['n_anger_desp']}, 새 감정만 {RA['n_only_new3']}(HOPE만 {RA['n_only_hope']}).")
w(f"- 예측: " + " · ".join(f"{PN[k]} {RA['pred_n'][k]}" for k in [str(i) for i in range(6)]) + f". 예측 몫 {pc(RA['pred_share'])}, 군중 예측 평균 {RA['crowd_mean_P']:+.3f}.")
w(f"- 조롱 대상(MOCK 안): 기타 {pc(RA['mock_target']['other'])}, 강세론 {pc(RA['mock_target']['bull'])}, 약세론 {pc(RA['mock_target']['bear'])}. 확신도: 1 {pc(RA['conf']['1'], 0)}, 2 {pc(RA['conf']['2'], 0)}, 3 {pc(RA['conf']['3'], 0)}.")
w("- 감정 × 예측 단위 수: " + ", ".join(f"{CK[k]} {RA['emo_x_pred'][k]['n']}" for k in CK) + ".")
w("- 흔한 섞인 조합: " + ", ".join(f"{c['combo'].replace('+', ' + ')} {c['n']}" for c in RA["top_mixed_combos"][:5]) + ".")
w("")

# ---------------- 한계
w("## 점검과 한계")
w("")
w("- 형식 점검(모든 파일 통과): id 중복 없음, 조각·겹침 파일의 id 순서가 `sample_key.parquet`의 shard_pos·overlap_pos와 같음, 세기 0~3 정수, MOCK ≥ 1 ↔ mock_target, pred = 0 ↔ pred_target null, conf 1~3, 검수지 id 순서가 `review_label_key.parquet`와 같음.")
w("- κ는 같은 계열 AI 두 번의 일치다. 같은 지침으로 같은 방식의 실수를 할 수 있어 타당성 확인은 사용자 검수 200건과의 비교가 맡는다.")
w(f"- ENVY({A['emotion']['ENVY']['n_ge1']}단위)·REGRET({A['emotion']['REGRET']['n_ge1']}단위)·불안한 기쁨({A['n_anx_joy']}단위)은 표본이 작다. 이중 라벨 600단위 안 ENVY는 1차 {pe['ENVY']['n_primary']}·2차 {pe['ENVY']['n_second']}단위라 κ 구간이 넓다({rng(pe['ENVY']['kappa_presence_ci95'])}).")
w("- 조롱은 세기 1(가벼운 비꼼)이 대부분이라, 조롱 지표를 쓸 때 세기 ≥ 2 판을 함께 보는 편이 1차 7범주와 맞대기 쉽다.")
w("- 가격·지수·수익 자료는 읽지 않았고 창(하루)별 특징도 만들지 않았다.")
w("- 집계 코드는 세션 임시 폴더(`scratchpad/nine_batchA.py`, `nine_batchA_report.py`)에 있고 저장소에는 넣지 않았다.")
(EMO / "nine_batchA.md").write_text("\n".join(L) + "\n", "utf-8")
print("ok", len(L))
