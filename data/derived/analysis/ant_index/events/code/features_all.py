"""부록 13 사건 연구 입력: 창별 통합 특징 표(W_KR 283창 + W_U 294창). 가격을 읽지 않는다.

포럼–가격 연결 금지: 이 파일은 포럼 특징만 만든다. 가격·사건 날짜·사건 표지를 읽지 않는다.
feat_text의 지난 가격 표지 열(px_·ev_)은 열 이름을 골라 읽어서 아예 불러오지 않는다.

입력(읽기만)
  (부록 14 갱신: 아래 final/results·final/label_key·final/labels는 snapshot_1023/(snap_paths.SNAP)의 같은 파일이다.
   스냅샷만 2026-10-23 07:59 KST까지 늘렸고 계산은 그대로다. feat_views도 snapshot_1023/results에서 읽는다)
  final/results/feat_activity.parquet   창 표지·활동(글 수 번호 증가, 저장 글, 마감 전 댓글, 새 작성자 30일(국내))
  final/results/feat_text.parquet       통과 사전 묶음 작성자 몫(MC·LL_US·LS·ANG·SEMI)
  final/results/feat_topic.parquet      H46 Tilt(W_U) — 아래 재계산 점검용
  final/results/feat_topic_unit_hits.parquet  단위별 MEM·SEMI·TILT_US·TILT_KR 적중(units 행 순서와 같음)
  final/results/feat_text_units.parquet 단위별 v0.2 적중(MC 재계산 점검용)
  final/results/units.parquet, windows_kr.parquet, windows_us.parquet  로더 산출
  final/results/pred_windows_full.parquet  예측 층(W_KR, 검수 통과)
  final/label_key.parquet + final/labels/, preview/label_key.parquet + preview/labels/  예측 층 라벨(W_U 재배정용)
  events/label_key.parquet + events/labels/ + emotion/labels4/primary_*.json + bottom_case/labels/  감정 층 라벨
  events/results/feat_views.parquet     조회·추천 (a)(b)(c)
  events/results/label_strata.csv       감정 표본 층 틀 크기(점검용)

새로 계산하는 것(같은 동결 코드 함수를 불러 쓴다)
  - KX(국장 이탈) 작성자 몫: interim lexicon_v02_final.load_v02() 매처로 단위마다 적중을 세고, feat_text와 같은
    작성자 몫(창 안 유효 단위 = 뉴스형·삭제·정치 잡담 제외, 그 묶음을 한 번이라도 쓴 고유 작성자 ÷ 유효 단위 고유 작성자).
    같은 함수로 MC를 다시 세어 feat_text MC_A_all과 같은지 점검한다.
  - MEM·SEMI 글 작성자 몫, TILT = ln(미장 단어 글 ÷ 국장 단어 글)(feat_topic H46과 같은 글 기준)을 두 창 모두.
    W_U Tilt는 feat_topic h46_tilt와 같은지 점검한다.
  - 새 작성자 몫(30일): feat_activity.newcomer_features를 W_U 창에 돌린다(국내는 feat_activity NW30과 같은지 점검).
  - 시간대별 몫: 창 안 단위(글 + 날짜 확정 댓글) 중 낮 장중(09:00~15:30 KST)·새벽(01:00~05:00 KST) 몫.
  - 예측 층 W_U: W_KR 표본 단위를 자기 시각의 W_U 창으로 옮기고, 칸(W_U 창 × W_KR 창 × 글/댓글)마다
    사후 층 가중(틀 수 ÷ 표본 수)을 준다. 글·댓글 몫을 반씩(W_KR 설계와 같게) 맞춘다. 작성자당 1회 = (창, 작성자, pred).
    같은 함수를 W_KR에 돌리면 pred_windows_full과 같아야 한다(점검).
  - 감정 층(설명용): 9감정 몫(세기 ≥ 1 작성자 몫)·평균 세기(작성자별 최대 세기, 0~3), 섞인 감정 몫, 중립 몫,
    들뜸 지수 = (HYPE + JOY) − (DESP + FEAR) 평균 세기. 작성자당 1회(창 안 작성자마다 감정별 최대 세기).
    W_KR은 표본기의 하루 사후 가중(w_day)과 같은 칸(W_KR 창 × 글/댓글 × 하루), W_U는 칸(W_U 창 × W_KR 창 × 글/댓글 × 하루).
    7/1 이후 창에 든 6/30 이전 단위는 예측 층처럼 뺀다.

변환: 시간당 양(글 수·저장 글·마감 전 댓글)과 조회 ÷ 댓글은 ln. 나머지는 원값.
누출 표시: 없음 / 낮음(순위·분포만) / 의심(가격 연결 뒤 누출 시험 필요) / 있음(수집 나이와 같이 움직임).
검수 지위: 통과(셈·통과 사전·검수 통과 라벨) / 설명용(검수 실패 감정 층, 누출 있음, 실시간 전용, 창당 표본 부족).

산출: events/results/features_all.parquet (+ 같은 메타를 parquet 스키마 메타 'features_meta'에),
      events/results/features_all_meta.json
실행: py features_all.py
"""
from __future__ import annotations

import glob
import hashlib
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import snap_paths as SP

HERE = Path(__file__).resolve().parent
EV = HERE.parent
ANT = EV.parent
FINAL = ANT / "final"            # 동결 코드 원천(읽기만)
RES = SP.SNAP_RES                # 부록 14: 사건 연구 스냅샷 산출
PREVIEW = ANT / "preview"
EMO = ANT / "emotion"
BC = ANT / "bottom_case"
INTERIM = ANT / "interim"
OUT = EV / "results"
sys.path.insert(0, str(INTERIM / "code"))
import feat_activity as FA  # noqa: E402  (동결 코드: newcomer_features·window_frame·politics_only)
import lexicon_v02 as L  # noqa: E402
from lexicon_v02_final import load_v02  # noqa: E402

_sp = importlib.util.spec_from_file_location("final_label_sample", FINAL / "code" / "label_sample.py")
FLS = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(FLS)       # 감정·예측 표본 틀 규칙(그 모듈 main은 돌리지 않는다)
SP.patch_final(FLS)
_sp2 = importlib.util.spec_from_file_location("final_pred_windows_full", FINAL / "code" / "pred_windows_full.py")
PWF = importlib.util.module_from_spec(_sp2)
_sp2.loader.exec_module(PWF)      # 예측 층 라벨 읽기(read_labels)
SP.patch_final(PWF)               # 라벨 폴더 = snapshot_1023/labels

KNOW_END = pd.Timestamp("2026-07-01")
OPEN_H = 8
E9 = ["HYPE", "JOY", "MOCK", "DESP", "ANGER", "FEAR", "HOPE", "REGRET", "ENVY"]
E9_KO = {"HYPE": "들뜸·자랑", "JOY": "기쁨·안도", "MOCK": "조롱·비꼼", "DESP": "우울·절망", "ANGER": "분노·욕",
         "FEAR": "불안·공포", "HOPE": "바람·기원", "REGRET": "후회·자책", "ENVY": "부러움·소외"}
TEXT_COLS = ["win_id", "market", "MC_A_all", "LL_US_A_all", "LS_A_all", "ANG_A_all", "SEMI_A_P", "n_auth_all",
             "n_auth_P"]                                   # px_·ev_ 열은 읽지 않는다
FORBIDDEN = ("px_", "ev_", "kospi", "hynix", "soxx", "price", "ret_", "close_px")
RETRO_MAX_RT_SHARE = 0.05         # 소급 창 = 실시간 수집 글 5% 미만(feat_views·views_leak 3-4와 같다)
MIN_PRED_UNITS = 30               # 예측 층 창 최소 단위(설계 60의 절반)
MIN_EMO_UNITS = 20                # 감정 층 창 최소 단위(설계 40의 절반)
MIN_COVERAGE = 0.5                # W_U 재배정 덮임 최소

# name, bundle, 한글, 원천, 검수 지위, 누출, 실시간 창 비교 가능, 사건 연구 격자, 메모
SPEC = [
    ("act_ln_posts_per_h", "활동", "글 수(번호 증가) ÷ 시간, ln", "feat_activity.d_no_per_h", "통과", "없음", True, True,
     "지운 글 포함 번호 증가"),
    ("act_ln_saved_per_h", "활동", "저장 글(작성자당 5건 상한) ÷ 시간, ln", "feat_activity.saved_cap5_per_h", "통과", "없음",
     True, True, "소급 창은 지운 글 몫(결번율 8%)만큼 작다"),
    ("act_ln_comments_pre_per_h", "활동", "마감 전 댓글 ÷ 시간, ln", "feat_activity.n_comments_onposts", "통과", "없음",
     True, True, "실시간 창은 글당 댓글이 덜 붙은 채 받힘(prechecks §5) → 실시간 경계 주의"),
    ("act_comments_per_post", "활동", "글당 마감 전 댓글", "feat_activity.n_comments_onposts ÷ n_posts", "통과", "없음",
     True, True, "실시간 경계 주의(평일 4.76 vs 소급 5.77)"),
    ("act_new30_share", "활동", "새 작성자 몫(직전 30일 이력 없음)", "feat_activity.NW30 / W_U는 같은 함수", "통과", "없음",
     True, True, "이력 30일이 자료 시작 앞이면 결측(첫 값 2025-09 말)"),
    ("act_share_day", "활동", "시간대 몫: 국내 장중 09:00~15:30 KST", "units(글 + 날짜 확정 댓글)", "통과", "없음",
     True, True, ""),
    ("act_share_night", "활동", "시간대 몫: 새벽 01:00~05:00 KST", "units(글 + 날짜 확정 댓글)", "통과", "없음",
     True, True, "W_U는 feat_activity NS_share(번호 보간판)와 뜻이 비슷하나 같지 않다"),
    ("vw_top10_share", "조회·추천", "(a) 조회 상위 10글 몫", "feat_views.a_top10_view_share", "통과", "낮음", False, True,
     "창 글 수와 ρ −0.91(사실상 글 수 지표)"),
    ("vw_top10pct_share", "조회·추천", "(a) 조회 상위 10% 글 몫", "feat_views.a_top10pct_view_share", "통과", "낮음",
     False, True, "글 수와 무관(ρ +0.01)"),
    ("vw_view_gini", "조회·추천", "(a) 조회 지니", "feat_views.a_view_gini", "통과", "낮음", False, True,
     "소급 창끼리만 비교"),
    ("vw_up_gini", "조회·추천", "(a) 추천 지니", "feat_views.a_up_gini", "통과", "낮음", True, True,
     "나이·실시간 경계에 가장 덜 흔들림"),
    ("vw_ln_b_median", "조회·추천", "(b) 조회 ÷ (마감 전 댓글 + 1) 창 중앙값, ln", "feat_views.b_vpc_median", "통과",
     "의심", False, True, "부록 13 누출 시험(사건일 vs 가짜 사건일) 통과 전에는 후보 아님. 소급 창 나이와 ρ +0.42"),
    ("vw_ln_b_adj_short_median", "조회·추천", "(b) 나이 보정 짧은판, ln", "feat_views.b_vpc_adj_short_median", "통과",
     "의심", False, False, "소급 창에서는 원값과 거의 같다(격자에 넣지 않음)"),
    ("vw_down_ratio", "조회·추천", "다른의견 비율(창 합)", "feat_views.down_ratio_first", "설명용", "있음", False, True,
     "수집 나이와 같이 내려감(2026 소급 ρ −0.61)"),
    ("vw_c_views_median", "조회·추천", "(c) 실시간 첫 줄 조회 중앙값", "feat_views.c_views_median_first", "설명용", "낮음",
     True, False, "실시간 창(W_KR 6·W_U 7)만 값 → 직전 20창 z 불가"),
    ("vw_P_bar_top10", "조회·추천", "(a) 조회 상위 10% 글의 예측 P̄", "feat_views.P_bar_top10", "설명용", "낮음", False,
     False, "창당 예측 0.92개(W_KR) — 창 단위로 못 씀"),
    ("tp_MC_A", "주제 사전", "반대매매·마진콜 작성자 몫", "feat_text.MC_A_all", "통과", "없음", True, True, ""),
    ("tp_LL_US_A", "주제 사전", "미국 레버리지 롱 작성자 몫", "feat_text.LL_US_A_all", "통과", "없음", True, True,
     "LL_KR은 측정 실패라 넣지 않음"),
    ("tp_LS_A", "주제 사전", "인버스·숏 작성자 몫", "feat_text.LS_A_all", "통과", "없음", True, True, ""),
    ("tp_KX_A", "주제 사전", "국장 이탈 작성자 몫", "v0.2 매처 재계산(feat_text 규칙)", "통과", "없음", True, True, ""),
    ("tp_ANG_A", "주제 사전", "분노 단어 작성자 몫", "feat_text.ANG_A_all", "통과", "없음", True, True, ""),
    ("tp_SEMI_A", "주제 사전", "반도체 글 작성자 몫(글 작성자 중)", "feat_text.SEMI_A_P", "통과", "없음", True, True,
     "SEMI는 글 단위 묶음"),
    ("tp_MEM_A", "주제 사전", "메모리 글 작성자 몫(글 작성자 중)", "feat_topic_unit_hits.mem", "통과", "없음", True, True,
     "MEM은 글 단위 묶음"),
    ("tp_TILT", "주제 사전", "미장/국장 Tilt = ln(미장 글 ÷ 국장 글)", "feat_topic_unit_hits.tus·tkr", "통과", "없음",
     True, True, "W_U는 feat_topic h46_tilt와 같음"),
    ("pr_P_bar", "예측 층", "군중 예측 P̄(예측 쌍 평균 − 3)", "pred_windows_full / W_U 재배정", "통과", "없음", True, True,
     "D < 2026-07-01 창은 라벨 모델 지식 기간(설명·예열용)"),
    ("pr_P_share", "예측 층", "예측 몫(예측 쌍 ÷ 전체 쌍)", "pred_windows_full / W_U 재배정", "통과", "없음", True, True,
     "지식 기간 표시 같음"),
    ("pr_P_strong1", "예측 층", "강한 하락(1) 몫(÷ 예측 쌍)", "pred_windows_full / W_U 재배정", "통과", "없음", True, True,
     "창당 강한 하락 쌍이 적다"),
    ("pr_P_strong5", "예측 층", "강한 상승(5) 몫(÷ 예측 쌍)", "pred_windows_full / W_U 재배정", "통과", "없음", True, True,
     "창당 강한 상승 쌍이 적다"),
]
for e in E9:
    SPEC.append((f"em_share_{e}", "감정 층", f"{E9_KO[e]} 몫(세기 ≥ 1 작성자)", "events 감정 라벨", "설명용", "없음", True,
                 True, "부록 8 검수 실패(설명용)"))
for e in E9:
    SPEC.append((f"em_mean_{e}", "감정 층", f"{E9_KO[e]} 평균 세기(0~3)", "events 감정 라벨", "설명용", "없음", True, True,
                 "부록 8 검수 실패(설명용)"))
SPEC += [
    ("em_mixed_share", "감정 층", "섞인 감정 몫(두 감정 이상 ≥ 1인 단위가 있는 작성자)", "events 감정 라벨", "설명용", "없음",
     True, True, "부록 8 검수 실패(설명용)"),
    ("em_neutral_share", "감정 층", "중립 몫(모든 단위가 감정 0인 작성자)", "events 감정 라벨", "설명용", "없음", True, True,
     "부록 8 검수 실패(설명용)"),
    ("em_hype_index", "감정 층", "들뜸 지수 = (HYPE + JOY) − (DESP + FEAR) 평균 세기", "events 감정 라벨", "설명용", "없음",
     True, True, "부록 5·6 정의, 부록 8 검수 실패(설명용)"),
]
FEATURES = [s[0] for s in SPEC]

# 표지(특징 아님): 창 정보·표본 수·수집 방식
MARKERS = ["win_id", "market", "D", "P", "weekday", "kind", "main_kind", "kind_class", "open_kst", "close_kst",
           "length_h", "over96", "outage", "judge_ok", "realtime", "realtime_window", "retro_window",
           "share_realtime_posts", "age_first_median_h", "missing_rate", "trunc_share", "knowledge_period",
           "n_posts", "n_comments_onposts", "tp_n_auth_valid", "tp_n_auth_post", "pr_n_units", "pr_n_pairs", "pr_n_pred",
           "pr_us_coverage", "em_n_units", "em_n_authors", "em_n_units_v1", "em_n_units_new", "em_share_knowledge_units",
           "em_coverage"]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ln_pos(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(x > 0, np.log(x), np.nan)


def author_share(df: pd.DataFrame, key: str, flag: str, index: pd.Index) -> tuple[pd.Series, pd.Series]:
    """feat_text.author_ratio와 같은 꼴: 창 안 그 묶음을 한 번이라도 쓴 고유 작성자 ÷ 창 안 고유 작성자."""
    if df.empty:
        return pd.Series(np.nan, index=index), pd.Series(0, index=index)
    g = df.groupby([key, "a"], sort=False)[flag].max()
    return g.groupby(level=0).mean().reindex(index), g.groupby(level=0).size().reindex(index).fillna(0).astype(int)


# ---------------------------------------------------------------- 창 달력과 표지

def window_table() -> pd.DataFrame:
    A = pd.read_parquet(RES / "feat_activity.parquet")
    keep = ["win_id", "market", "D", "P", "weekday", "kind", "open_kst", "close_kst", "length_h", "over96", "outage",
            "judge_ok", "realtime", "missing_rate", "trunc_share", "n_posts", "n_comments_onposts", "d_no_per_h",
            "saved_cap5_per_h", "NW30"]
    W = A[keep].copy()
    mk = []
    for m, fn in (("kr", "windows_kr.parquet"), ("us", "windows_us.parquet")):
        w = pd.read_parquet(RES / fn)
        w = w[(w["variant"] == "main") & w["complete"]][["win_id", "main_kind"]]
        mk.append(w)
    W = W.merge(pd.concat(mk), on="win_id", how="left", validate="1:1")
    W["kind_class"] = np.where(W["kind"] == "weekday", "weekday", "nonweekday")
    W["D"] = pd.to_datetime(W["D"])
    W["knowledge_period"] = W["D"] < KNOW_END
    W["outage"] = W["outage"].fillna(False).astype(bool)
    return W


# ---------------------------------------------------------------- 단위 수준 재계산(KX·MEM·SEMI·TILT·시간대·새 작성자)

def unit_pass(u: pd.DataFrame) -> pd.DataFrame:
    """창(국내·미국)에 든 단위마다 v0.2 매처: 제외 사유, KX·MC 적중, 정치 전용 여부."""
    an = load_v02()
    sel = u.index[(u["win_kr"].notna() | u["win_u"].notna()).to_numpy()]
    ut_a, ti_a, tx_a = u["utype"].to_numpy(), u["title"].to_numpy(), u["text"].to_numpy()
    excl, kx, mc, pol = [], [], [], []
    for i in sel:
        ut = ut_a[i]
        ti = ti_a[i] if isinstance(ti_a[i], str) else ""
        tx = tx_a[i] if isinstance(tx_a[i], str) else ""
        r = an(ti if ut == "post" else None, tx, ut)
        st = {h.stratum for h in r.hits if L.STRATUM_SCOPE[h.stratum] in ("all", ut)}
        excl.append(r.excluded)
        kx.append("KX" in st)
        mc.append("MC" in st)
        pol.append(FA.politics_only(ti_a[i] if isinstance(ti_a[i], str) else None, tx_a[i] if isinstance(tx_a[i], str)
                                    else None, ut))
    return pd.DataFrame({"excl": excl, "h_KX": kx, "h_MC": mc, "politics_only": pol}, index=sel)


def unit_features(W: pd.DataFrame, u: pd.DataFrame, checks: dict) -> pd.DataFrame:
    """KX·MEM·SEMI·TILT·시간대 몫·새 작성자(W_U)를 창마다. 반환: win_id 색인 표."""
    hits = pd.read_parquet(RES / "feat_topic_unit_hits.parquet")
    assert len(hits) == len(u), "feat_topic_unit_hits 행 수 ≠ units"
    assert (hits["no"].to_numpy() == u["no"].to_numpy()).all(), "feat_topic_unit_hits 행 순서 ≠ units"
    lx = unit_pass(u)
    tu = pd.read_parquet(RES / "feat_text_units.parquet", columns=["uidx", "excl", "h_MC"]).set_index("uidx")
    common = lx.index.intersection(tu.index)
    checks["kx_pass_units"] = int(len(lx))
    checks["kx_pass_excl_equals_feat_text_units"] = bool((lx.loc[common, "excl"] == tu.loc[common, "excl"]).all())
    checks["kx_pass_MC_equals_feat_text_units"] = bool((lx.loc[common, "h_MC"] == tu.loc[common, "h_MC"]).all())
    checks["kx_units_hit"] = int(lx["h_KX"].sum())

    out = []
    for market, key in (("kr", "win_kr"), ("us", "win_u")):
        idx = pd.Index(W.loc[W["market"] == market, "win_id"], name="win_id")
        x = u.loc[lx.index][[key, "a", "utype", "dt_kst", "dt_conf"]].join(lx)
        x = x[x[key].isin(idx)]
        valid = x[x["excl"] == ""]
        F = pd.DataFrame(index=idx)
        F["tp_KX_A"], F["tp_n_auth_valid"] = author_share(valid, key, "h_KX", idx)
        F["chk_MC_A"], _ = author_share(valid, key, "h_MC", idx)
        # 글 단위 묶음: MEM·SEMI 글 작성자 몫, Tilt(H46 꼴: 두 쪽 단어 글 수)
        h = hits.loc[x.index]
        vp = x.assign(mem=(h["mem"] > 0).to_numpy(), semi=(h["semi"] > 0).to_numpy(), tus=(h["tus"] > 0).to_numpy(),
                      tkr=(h["tkr"] > 0).to_numpy(), excl_h=h["excl"].to_numpy())
        vp = vp[(vp["utype"] == "post") & (vp["excl_h"] == "")]
        F["tp_MEM_A"], F["tp_n_auth_post"] = author_share(vp, key, "mem", idx)
        F["chk_SEMI_A_P"], _ = author_share(vp, key, "semi", idx)
        n_us = vp.groupby(key)["tus"].sum().reindex(idx).fillna(0)
        n_kr = vp.groupby(key)["tkr"].sum().reindex(idx).fillna(0)
        F["tp_TILT"] = np.where((n_us > 0) & (n_kr > 0), np.log(n_us.clip(lower=1) / n_kr.clip(lower=1)), np.nan)
        # 시간대 몫: 창 안 단위(글 + 날짜 확정 댓글), 제외 사유 무관
        tt = u.loc[u[key].isin(idx), [key, "utype", "dt_kst", "dt_conf"]]
        tt = tt[~((tt["utype"] == "comment") & (tt["dt_conf"] == "latest"))]
        mins = tt["dt_kst"].dt.hour * 60 + tt["dt_kst"].dt.minute
        tt = tt.assign(day=((mins >= 9 * 60) & (mins < 15 * 60 + 30)), night=((mins >= 60) & (mins < 5 * 60)))
        g = tt.groupby(key)
        F["act_share_day"] = g["day"].mean().reindex(idx)
        F["act_share_night"] = g["night"].mean().reindex(idx)
        F["market"] = market
        out.append(F)
    U = pd.concat(out)

    # 새 작성자 몫(30일): feat_activity.newcomer_features 그대로(W_U 창에도)
    c0 = pd.Timestamp(u.loc[u["utype"] == "post", "dt_kst"].min())
    lex = pd.DataFrame({"politics_only": False}, index=u.index)
    lex.loc[lx.index, "politics_only"] = lx["politics_only"].to_numpy()
    nw = {}
    for market, fn, key in (("kr", "windows_kr.parquet", "win_kr"), ("us", "windows_us.parquet", "win_u")):
        f = FA.window_frame(pd.read_parquet(RES / fn))
        f = FA.newcomer_features(f, u, lex, key, c0)
        nw.update(dict(zip(f["win_id"], f["NW30"])))
    U["act_new30_share_calc"] = U.index.map(nw)
    return U


# ---------------------------------------------------------------- 라벨 층(가중)

def frame_cells(u: pd.DataFrame) -> pd.DataFrame:
    """감정·예측 표본 틀(final label_sample.frame 그대로) 단위마다 W_KR 창·W_U 창·글/댓글·하루."""
    w = FLS.load_windows()
    assert len(w) == SP.EXPECT_WINDOWS["kr"], len(w)
    uf = u[["no", "cid", "a", "utype", "dt_kst", "post_dt_kst", "dt_conf", "src_line", "title", "text", "win_kr",
            "pwin_kr", "win_u"]].copy()
    uf["unit_idx"] = np.arange(len(uf))
    f, _ = FLS.frame(uf, w)
    f = f[["unit_idx", "win_kr", "win_u", "utype", "dt_kst"]].copy()
    f["day"] = (f["dt_kst"] - pd.Timedelta(hours=OPEN_H)).dt.normalize()
    return f


def assign_us(t: pd.Series, wu: pd.DataFrame) -> np.ndarray:
    """시각 → W_U 본판 완전 창 id([open, close))."""
    o = wu["open_kst"].to_numpy(dtype="datetime64[ns]")
    c = wu["close_kst"].to_numpy(dtype="datetime64[ns]")
    ids = wu["win_id"].to_numpy()
    tt = pd.to_datetime(t).to_numpy(dtype="datetime64[ns]")
    j = np.searchsorted(c, tt, side="right")
    res = np.full(len(tt), None, dtype=object)
    ok = j < len(c)
    jj = j[ok]
    inside = tt[ok] >= o[jj]
    sel = np.flatnonzero(ok)[inside]
    res[sel] = ids[jj[inside]]
    return res


def design_weights(s: pd.DataFrame, base_w: pd.Series, fr: pd.DataFrame, win_col: str) -> tuple[pd.Series, pd.Series]:
    """표본 설계 가중(포함 확률의 역수 base_w)을 창(win_col) × 글/댓글마다 합 0.5로 맞춘다(글·댓글 반씩, W_KR 설계와 같게).
    W_U 창은 W_KR 표본의 '부분 영역' 추정(하젝)이다. 덮임 = 그 창 틀 단위 중 표본이 1개 이상인
    (창, W_KR 창, 글/댓글) 칸에 든 몫(표본이 없는 칸의 틀 단위는 추정에 대표되지 않는다)."""
    w = pd.Series(np.asarray(base_w, dtype=float), index=s.index)
    if w.isna().any() or (w <= 0).any():
        raise RuntimeError("설계 가중이 비었거나 0 이하")
    w = 0.5 * w / w.groupby([s[win_col], s["utype"]]).transform("sum")
    cells = [win_col, "win_kr", "utype"] if win_col != "win_kr" else ["win_kr", "utype"]
    N = fr.groupby(cells).size().rename("N").reset_index()
    covered = pd.MultiIndex.from_frame(N[cells]).isin(pd.MultiIndex.from_frame(s[cells].drop_duplicates()))
    N["Ncov"] = np.where(covered, N["N"], 0)
    g = N.groupby(win_col)[["Ncov", "N"]].sum()
    return w, g["Ncov"] / g["N"]


def stratum_size(fr: pd.DataFrame) -> pd.Series:
    """표본 층(W_KR 창 × 글/댓글)의 틀 크기."""
    return fr.groupby(["win_kr", "utype"]).size()


def lookup(sz: pd.Series, s: pd.DataFrame) -> np.ndarray:
    return sz.reindex(pd.MultiIndex.from_frame(s[["win_kr", "utype"]])).to_numpy(dtype=float)


def pred_weighted(m: pd.DataFrame, win_col: str, wt: pd.Series, index: pd.Index) -> pd.DataFrame:
    """작성자당 1회 = (창, 작성자, pred) 쌍, 쌍 가중 = 그 단위들 가중의 평균. 부록 4 정의의 가중판."""
    d = m.assign(w=wt.to_numpy())
    pairs = d.groupby([win_col, "a", "pred"], sort=False)["w"].mean().reset_index()
    g = pairs.groupby(win_col)
    tot = g["w"].sum().reindex(index)
    pp = pairs[pairs["pred"] > 0]
    gp = pp.groupby(win_col)
    wpred = gp["w"].sum().reindex(index)
    F = pd.DataFrame(index=index)
    with np.errstate(invalid="ignore", divide="ignore"):
        F["pr_P_bar"] = (pp.assign(x=pp["pred"] * pp["w"]).groupby(win_col)["x"].sum().reindex(index) / wpred) - 3.0
        F["pr_P_share"] = wpred.fillna(0) / tot
        for v in (1, 5):
            c = pp[pp["pred"] == v].groupby(win_col)["w"].sum().reindex(index).fillna(0)
            F[f"pr_P_strong{v}"] = np.where(wpred > 0, c / wpred, np.nan)
    F["pr_n_units"] = d.groupby(win_col).size().reindex(index).fillna(0).astype(int)
    F["pr_n_pairs"] = g.size().reindex(index).fillna(0).astype(int)
    F["pr_n_pred"] = gp.size().reindex(index).fillna(0).astype(int)
    return F


def pred_layer(W: pd.DataFrame, fr: pd.DataFrame, checks: dict) -> pd.DataFrame:
    cols = ["id", "win_id", "a", "dt_kst", "utype", "shard", "D", "judge_ok", "in_knowledge_period"]
    kf = pd.read_parquet(SP.SNAP / "label_key.parquet", columns=cols).assign(src="final")
    kp = pd.read_parquet(PREVIEW / "label_key.parquet", columns=cols).assign(src="preview")
    key = pd.concat([kf, kp], ignore_index=True)
    key["dt_kst"] = pd.to_datetime(key["dt_kst"]).astype("datetime64[ns]")
    m_all, info = PWF.read_labels(key)
    checks["pred_labels"] = {"usable": info["usable_labels"], "missing": info["missing_labels"],
                             "format_errors_excluded": info["format_errors_excluded_total"]}
    m = m_all[m_all["pred"].notna()].copy()
    m["pred"] = m["pred"].astype(int)
    s0 = m.rename(columns={"win_id": "win_kr"}).copy()
    # 설계 가중: 층(W_KR 창 × 글/댓글) 단순 무작위 → 틀 수 ÷ 표본 수(빼기 전 표본 수)
    sz = stratum_size(fr)
    n_s = s0.groupby(["win_kr", "utype"]).size()
    s0["base_w"] = lookup(sz, s0) / n_s.reindex(pd.MultiIndex.from_frame(s0[["win_kr", "utype"]])).to_numpy(float)
    wk = W[W["market"] == "kr"]
    wu = W[W["market"] == "us"].sort_values("close_kst")

    # W_KR: 7/1 이후 창의 지식 기간(6/30 이전) 단위는 뺀다(pred_windows_full과 같음)
    s = s0[~((pd.to_datetime(s0["D"]) >= KNOW_END) & (s0["dt_kst"] < KNOW_END))].copy()
    w_kr, _ = design_weights(s, s["base_w"], fr, "win_kr")
    Fk = pred_weighted(s, "win_kr", w_kr, pd.Index(wk["win_id"]))
    pw = pd.read_parquet(RES / "pred_windows_full.parquet").set_index("win_id")
    same = {}
    for c in ("P_bar", "P_share", "P_strong1", "P_strong5"):
        a = Fk[f"pr_{c}"].astype(float).to_numpy()
        b = pw.loc[Fk.index, c].astype(float).to_numpy()
        same[c] = int(np.isclose(a, b, equal_nan=True, atol=1e-9).sum())
    checks["pred_kr_weighted_equals_pred_windows_full"] = {
        "windows": int(len(Fk)), "equal": same,
        "note": "글·댓글 표본 수가 다른 창(지식 기간 단위를 뺀 K2026-07-01 등)만 가중판과 달라질 수 있다"}
    # 주판 W_KR 값은 등록된 pred_windows_full 그대로 쓴다
    for c in ("P_bar", "P_share", "P_strong1", "P_strong5"):
        Fk[f"pr_{c}"] = pw.loc[Fk.index, c].astype(float)
    Fk["pr_n_units"] = pw.loc[Fk.index, "n_units"].astype(int)
    Fk["pr_n_pairs"] = pw.loc[Fk.index, "n_pairs"].astype(int)
    Fk["pr_n_pred"] = pw.loc[Fk.index, "n_pred"].astype(int)
    Fk["pr_us_coverage"] = np.nan

    # W_U: 단위 시각의 W_U 창. 7/1 이후 W_U 창에 든 6/30 이전 단위는 뺀다(W_U 창 D 기준)
    s = s0.copy()
    s["win_u"] = assign_us(s["dt_kst"], wu)
    s = s[s["win_u"].notna()].copy()
    us_D = dict(zip(wu["win_id"], pd.to_datetime(wu["D"])))
    s = s[~((s["win_u"].map(us_D) >= KNOW_END) & (s["dt_kst"] < KNOW_END))].copy()
    fu = fr[fr["win_u"].isin(set(wu["win_id"]))]
    w_us, cov = design_weights(s, s["base_w"], fu, "win_u")
    Fu = pred_weighted(s, "win_u", w_us, pd.Index(wu["win_id"]))
    Fu["pr_us_coverage"] = Fu.index.map(cov).astype(float)
    checks["pred_us"] = {"units": int(len(s)), "windows_with_units": int((Fu["pr_n_units"] > 0).sum()),
                         "coverage_min": float(np.nanmin(Fu["pr_us_coverage"])),
                         "coverage_lt_0_95": Fu.index[Fu["pr_us_coverage"] < 0.95].tolist()}
    return pd.concat([Fk, Fu])


def read_emotion_labels(key: pd.DataFrame, checks: dict) -> pd.DataFrame:
    """대응표(events/label_key) id → 9감정 라벨. 새 단위 = events/labels, 재사용 = 원래 파일의 src_id."""
    lab = {}
    for f in sorted(glob.glob(str(EV / "labels" / "shard_*.json"))):
        for r in json.loads(Path(f).read_text("utf-8")):
            lab[("new", r["id"])] = r
    for f in sorted(glob.glob(str(EMO / "labels4" / "primary_*.json"))):
        for r in json.loads(Path(f).read_text("utf-8")):
            lab[("emotion_labels4", r["id"])] = r
    for f in sorted(glob.glob(str(BC / "labels" / "shard_*.json"))):
        for r in json.loads(Path(f).read_text("utf-8")):
            lab[("bottom_case", r["id"])] = r
    recs, miss, bad = [], 0, 0
    for i, src, sid in zip(key["id"], key["src"], key["src_id"]):
        rid = i if src == "new" else sid
        r = lab.get((src, rid))
        if r is None:
            miss += 1
            continue
        emo = r.get("emo", {})
        if set(emo) != set(E9) or not all(isinstance(emo[e], int) and 0 <= emo[e] <= 3 for e in E9):
            bad += 1
            continue
        recs.append({"id": i, **{e: int(emo[e]) for e in E9}})
    checks["emotion_labels"] = {"key_units": int(len(key)), "labeled": len(recs), "missing": miss, "bad_format": bad}
    return key.merge(pd.DataFrame(recs), on="id", how="inner", validate="1:1")


def emotion_weighted(m: pd.DataFrame, win_col: str, wt: pd.Series, index: pd.Index) -> pd.DataFrame:
    """작성자당 1회: (창, 작성자)마다 감정별 최대 세기, 섞임(섞인 단위 있음), 중립(모든 단위 0). 작성자 가중 = 단위 가중 평균."""
    d = m.assign(w=wt.to_numpy())
    d["n_emo"] = (d[E9] >= 1).sum(axis=1)
    d["mixed"] = d["n_emo"] >= 2
    d["neut"] = d["n_emo"] == 0
    agg = {e: "max" for e in E9}
    agg.update({"mixed": "max", "neut": "min", "w": "mean"})
    A = d.groupby([win_col, "a"], sort=False).agg(agg).reset_index()
    F = pd.DataFrame(index=index)
    ws = A.groupby(win_col)["w"].sum().reindex(index)

    def wmean(col_vals: pd.Series) -> pd.Series:
        return (col_vals * A["w"]).groupby(A[win_col]).sum().reindex(index) / ws

    for e in E9:
        F[f"em_share_{e}"] = wmean((A[e] >= 1).astype(float))
        F[f"em_mean_{e}"] = wmean(A[e].astype(float))
    F["em_mixed_share"] = wmean(A["mixed"].astype(float))
    F["em_neutral_share"] = wmean(A["neut"].astype(float))
    F["em_hype_index"] = (F["em_mean_HYPE"] + F["em_mean_JOY"]) - (F["em_mean_DESP"] + F["em_mean_FEAR"])
    F["em_n_units"] = d.groupby(win_col).size().reindex(index).fillna(0).astype(int)
    F["em_n_authors"] = A.groupby(win_col).size().reindex(index).fillna(0).astype(int)
    F["em_n_units_v1"] = d[d["guideline"] == "v1"].groupby(win_col).size().reindex(index).fillna(0).astype(int)
    F["em_n_units_new"] = d[d["src"] == "new"].groupby(win_col).size().reindex(index).fillna(0).astype(int)
    F["em_share_knowledge_units"] = d.groupby(win_col)["in_knowledge_period"].mean().reindex(index)
    return F


def emotion_layer(W: pd.DataFrame, fr: pd.DataFrame, u: pd.DataFrame, checks: dict) -> pd.DataFrame:
    key = pd.read_parquet(EV / "label_key.parquet")
    key["dt_kst"] = pd.to_datetime(key["dt_kst"]).astype("datetime64[ns]")
    m = read_emotion_labels(key, checks)
    m["day"] = (m["dt_kst"] - pd.Timedelta(hours=OPEN_H)).dt.normalize()
    wk = W[W["market"] == "kr"]
    wu = W[W["market"] == "us"].sort_values("close_kst")
    s0 = m.rename(columns={"win_id": "win_kr"}).copy()
    # 설계 가중 = w_day(층 안 합 1) × 층 틀 크기 = 하루 칸의 틀 수 ÷ 표본 수(표본기 day_weights와 같음)
    sz = stratum_size(fr)
    s0["base_w"] = s0["w_day"].to_numpy(float) * lookup(sz, s0)
    N_day = fr.groupby(["win_kr", "utype", "day"]).size()
    n_day = s0.groupby(["win_kr", "utype", "day"]).size()
    k = pd.MultiIndex.from_frame(s0[["win_kr", "utype", "day"]])
    raw = N_day.reindex(k).to_numpy(float) / n_day.reindex(k).to_numpy(float)
    ratio = pd.Series(raw / s0["base_w"].to_numpy(float), index=s0.index)
    # 표본이 없는 하루가 있는 층은 Σ(N/n)이 층 틀 크기보다 작아 비율이 1보다 조금 클 수 있다(층 안에서는 같은 값)
    rr = ratio.groupby([s0["win_kr"], s0["utype"]]).agg(lambda r: float(r.max() / r.min()))
    checks["emotion_kr_weights_proportional_to_day_cells"] = bool((rr < 1 + 1e-9).all())
    # W_KR: 7/1 이후 창의 지식 기간 단위 빼기(예측 층과 같은 규칙)
    drop_k = (pd.to_datetime(s0["D"]) >= KNOW_END) & (s0["dt_kst"] < KNOW_END)
    checks["emotion_dropped_knowledge_units_kr"] = int(drop_k.sum())
    s = s0[~drop_k].copy()
    w_kr, _ = design_weights(s, s["base_w"], fr, "win_kr")
    Fk = emotion_weighted(s, "win_kr", w_kr, pd.Index(wk["win_id"]))
    Fk["em_coverage"] = np.nan
    # W_U: units 행 → W_U 창(로더 win_u). 같은 설계 가중의 부분 영역 추정
    s = s0.copy()
    s["win_u"] = u["win_u"].to_numpy()[s["unit_idx"].to_numpy()]
    s["win_u_t"] = assign_us(s["dt_kst"], wu)
    checks["emotion_us_assign_matches_loader"] = bool(
        (s["win_u"].fillna("-").astype(str).to_numpy() == pd.Series(s["win_u_t"]).fillna("-").astype(str).to_numpy()).all())
    s = s[s["win_u"].isin(set(wu["win_id"]))].copy()
    us_D = dict(zip(wu["win_id"], pd.to_datetime(wu["D"])))
    s = s[~((s["win_u"].map(us_D) >= KNOW_END) & (s["dt_kst"] < KNOW_END))].copy()
    fu = fr[fr["win_u"].isin(set(wu["win_id"]))]
    w_us, cov = design_weights(s, s["base_w"], fu, "win_u")
    Fu = emotion_weighted(s, "win_u", w_us, pd.Index(wu["win_id"]))
    Fu["em_coverage"] = Fu.index.map(cov).astype(float)
    checks["emotion_us"] = {"units": int(len(s)), "coverage_min": float(np.nanmin(Fu["em_coverage"])),
                            "coverage_lt_0_95": Fu.index[Fu["em_coverage"] < 0.95].tolist()}
    return pd.concat([Fk, Fu])


# ---------------------------------------------------------------- 조립

def build() -> tuple[pd.DataFrame, dict, dict]:
    checks: dict = {}
    W = window_table()
    ucols = ["no", "cid", "a", "utype", "dt_kst", "post_dt_kst", "dt_conf", "src_line", "title", "text", "win_kr",
             "pwin_kr", "win_u"]
    u = pd.read_parquet(RES / "units.parquet", columns=ucols)
    u["dt_kst"] = u["dt_kst"].astype("datetime64[ns]")
    u["post_dt_kst"] = u["post_dt_kst"].astype("datetime64[ns]")
    assert u.index.equals(pd.RangeIndex(len(u)))

    F = W.set_index("win_id").copy()
    # 활동
    F["act_ln_posts_per_h"] = ln_pos(F["d_no_per_h"])
    F["act_ln_saved_per_h"] = ln_pos(F["saved_cap5_per_h"])
    F["act_ln_comments_pre_per_h"] = ln_pos(F["n_comments_onposts"] / F["length_h"])
    with np.errstate(divide="ignore", invalid="ignore"):
        F["act_comments_per_post"] = np.where(F["n_posts"] > 0, F["n_comments_onposts"] / F["n_posts"], np.nan)
    U = unit_features(W, u, checks)
    for c in ("tp_KX_A", "tp_MEM_A", "tp_TILT", "act_share_day", "act_share_night", "tp_n_auth_valid",
              "tp_n_auth_post"):
        F[c] = U[c]
    nw_kr = F.loc[F["market"] == "kr", "NW30"]
    nw_calc = U.loc[nw_kr.index, "act_new30_share_calc"]
    checks["new30_kr_calc_equals_feat_activity"] = bool(np.allclose(nw_kr.astype(float), nw_calc.astype(float),
                                                                    equal_nan=True, atol=1e-12))
    F["act_new30_share"] = np.where(F["market"] == "kr", F["NW30"], U["act_new30_share_calc"].reindex(F.index))
    # 통과 사전 묶음(feat_text, 가격 표지 열은 읽지 않음)
    T = pd.read_parquet(RES / "feat_text.parquet", columns=TEXT_COLS).set_index("win_id")
    for src, dst in (("MC_A_all", "tp_MC_A"), ("LL_US_A_all", "tp_LL_US_A"), ("LS_A_all", "tp_LS_A"),
                     ("ANG_A_all", "tp_ANG_A"), ("SEMI_A_P", "tp_SEMI_A")):
        F[dst] = T.loc[F.index, src].astype(float)
    no_out = F.index[~F["outage"].to_numpy()]          # feat_text는 장애 창(W_U 9/24·9/25)을 결측으로 둔다
    checks["MC_recalc_equals_feat_text"] = bool(np.allclose(U.loc[no_out, "chk_MC_A"].astype(float),
                                                            F.loc[no_out, "tp_MC_A"], equal_nan=True, atol=1e-12))
    checks["SEMI_post_recalc_equals_feat_text"] = bool(np.allclose(U.loc[no_out, "chk_SEMI_A_P"].astype(float),
                                                                   F.loc[no_out, "tp_SEMI_A"], equal_nan=True,
                                                                   atol=1e-12))
    checks["auth_valid_equals_feat_text_n_auth_all"] = bool(
        (U.loc[no_out, "tp_n_auth_valid"].astype(int).to_numpy() == T.loc[no_out, "n_auth_all"].astype(int)
         .to_numpy()).all())
    tpc = pd.read_parquet(RES / "feat_topic.parquet", columns=["table", "win_id", "h46_tilt"])
    tpc = tpc[tpc["table"] == "US_D"].set_index("win_id")["h46_tilt"]
    usi = F.index[F["market"] == "us"]
    ok = F.loc[usi, "judge_ok"].to_numpy()
    checks["TILT_us_equals_feat_topic_h46_tilt"] = bool(np.allclose(F.loc[usi, "tp_TILT"].to_numpy()[ok],
                                                                    tpc.reindex(usi).to_numpy(float)[ok],
                                                                    equal_nan=True, atol=1e-12))
    # 조회·추천
    V = pd.read_parquet(SP.SNAP_RES / "feat_views.parquet").set_index("win_id")
    F["share_realtime_posts"] = V.loc[F.index, "share_realtime_posts"].astype(float)
    F["realtime_window"] = V.loc[F.index, "realtime_window"].astype(bool)
    F["retro_window"] = F["share_realtime_posts"] < RETRO_MAX_RT_SHARE
    F["age_first_median_h"] = V.loc[F.index, "age_first_median_h"].astype(float)
    F["vw_top10_share"] = V.loc[F.index, "a_top10_view_share"].astype(float)
    F["vw_top10pct_share"] = V.loc[F.index, "a_top10pct_view_share"].astype(float)
    F["vw_view_gini"] = V.loc[F.index, "a_view_gini"].astype(float)
    F["vw_up_gini"] = V.loc[F.index, "a_up_gini"].astype(float)
    F["vw_ln_b_median"] = ln_pos(V.loc[F.index, "b_vpc_median"])
    F["vw_ln_b_adj_short_median"] = ln_pos(V.loc[F.index, "b_vpc_adj_short_median"])
    F["vw_down_ratio"] = V.loc[F.index, "down_ratio_first"].astype(float)
    F["vw_c_views_median"] = V.loc[F.index, "c_views_median_first"].astype(float)
    F["vw_P_bar_top10"] = V.loc[F.index, "P_bar_top10"].astype(float)
    # 라벨 층
    fr = frame_cells(u)
    st = pd.read_csv(EV / "results" / "label_strata.csv")
    nf = fr.groupby(["win_kr", "utype"]).size()
    chk = st.assign(utype=st["kind"].map({"글": "post", "댓글": "comment"}))
    checks["frame_sizes_equal_label_strata"] = bool(
        (nf.reindex(pd.MultiIndex.from_frame(chk[["win_id", "utype"]])).to_numpy() == chk["N_frame"].to_numpy()).all())
    P = pred_layer(W, fr, checks)
    E = emotion_layer(W, fr, u, checks)
    for c in P.columns:
        F[c] = P[c].reindex(F.index)
    for c in E.columns:
        F[c] = E[c].reindex(F.index)
    # 장애 창(§2.4: 모든 특징에서 결측)은 특징을 비운다. 96시간 초과 창은 값을 두고 judge_ok = False로만 표시한다
    # (사건 연구가 judge_ok로 거른다)
    F.loc[F["outage"].to_numpy(), FEATURES] = np.nan
    # 라벨 층 최소 표본(설계 수의 절반): 예측 층 단위 30·감정 층 단위 20 미만이거나 W_U 덮임 0.5 미만이면 결측
    pr_cols = [c for c in FEATURES if c.startswith("pr_")]
    em_cols = [c for c in FEATURES if c.startswith("em_")]
    pr_bad = (F["pr_n_units"] < MIN_PRED_UNITS) | (F["pr_us_coverage"] < MIN_COVERAGE)
    em_bad = (F["em_n_units"] < MIN_EMO_UNITS) | (F["em_coverage"] < MIN_COVERAGE)
    F.loc[pr_bad.to_numpy(), pr_cols] = np.nan
    F.loc[em_bad.to_numpy(), em_cols] = np.nan
    checks["label_min_sample_masked"] = {
        "pred": {m: F.index[(F["market"] == m).to_numpy() & pr_bad.to_numpy()].tolist() for m in ("kr", "us")},
        "emotion": {m: F.index[(F["market"] == m).to_numpy() & em_bad.to_numpy()].tolist() for m in ("kr", "us")}}
    F = F.reset_index()
    F = F[MARKERS + FEATURES]
    F = F.sort_values(["market", "close_kst"]).reset_index(drop=True)
    bad = [c for c in F.columns if any(c.lower().startswith(p) or p in c.lower() for p in FORBIDDEN)]
    assert not bad, f"가격 냄새 열: {bad}"

    meta = {"features": {}}
    for (name, bundle, ko, source, status, leak, rt_ok, grid, note) in SPEC:
        cov = {m: int(F.loc[F["market"] == m, name].notna().sum()) for m in ("kr", "us")}
        meta["features"][name] = {"bundle": bundle, "ko": ko, "source": source, "status": status, "leak": leak,
                                  "realtime_comparable": bool(rt_ok), "event_study": bool(grid), "note": note,
                                  "label_layer": bundle in ("예측 층", "감정 층"),
                                  "knowledge_cutoff": str(KNOW_END.date()) if bundle in ("예측 층", "감정 층") else None,
                                  "non_null": cov}
    return F, meta, checks


def main() -> int:
    t0 = datetime.now().replace(microsecond=0)
    F, meta, checks = build()
    meta.update({
        "made": t0.isoformat(),
        "addendum": "HYPOTHESES_addendum13.md",
        "rows": {m: int((F["market"] == m).sum()) for m in ("kr", "us")},
        "markers": MARKERS,
        "price": "가격·사건 날짜·사건 표지를 읽지 않았다. feat_text는 px_·ev_ 열을 빼고 읽었다.",
        "status_rule": "통과 = 셈(활동·조회 원값), 측정 통과 사전 묶음, 검수 통과 라벨(예측 층). 설명용 = 감정 층(부록 8 검수 "
                       "실패), 누출 있음(다른의견 비율), 실시간 전용(c), 창당 표본 부족(조회 상위 10% 글 P̄)",
        "leak_rule": "없음 / 낮음(같은 창 안 순위·분포) / 의심(사건 연구의 누출 시험을 통과해야 후보) / 있음(수집 나이와 같이 움직임)",
        "realtime_rule": "realtime_comparable = False인 특징은 소급 창(retro_window: 실시간 글 5% 미만)끼리만 비교한다",
        "label_min_sample_rule": f"예측 층 단위 {MIN_PRED_UNITS}·감정 층 단위 {MIN_EMO_UNITS} 미만이거나 W_U 덮임 {MIN_COVERAGE} "
                                 "미만인 창은 그 층 특징을 결측으로 둔다(표본 수 표지는 남김). W_U 연휴 창이 주로 걸린다",
        "outage_rule": "장애 창(W_U 2026-09-24·25)은 모든 특징 결측(§2.4)",
        "label_layers": {"pred": "W_KR = pred_windows_full 그대로(창마다 글 30·댓글 30). W_U = 같은 단위를 자기 시각의 W_U 창으로 "
                                 "옮기고 칸(W_U 창 × W_KR 창 × 글/댓글) 사후 가중, 글·댓글 반씩",
                         "emotion": "W_KR = events/label_key(창마다 글 20·댓글 20, 재사용 포함) 하루 칸 가중. W_U = 칸(W_U 창 × "
                                    "W_KR 창 × 글/댓글 × 하루) 사후 가중, 글·댓글 반씩. 지침 v1 단위 수는 em_n_units_v1"},
        "checks": checks,
        "inputs_sha256": {p.name: sha(p) for p in [RES / "feat_activity.parquet", RES / "feat_text.parquet",
                                                    RES / "feat_topic.parquet", RES / "feat_topic_unit_hits.parquet",
                                                    RES / "units.parquet", RES / "pred_windows_full.parquet",
                                                    SP.SNAP_RES / "feat_views.parquet", EV / "label_key.parquet",
                                                    SP.SNAP / "label_key.parquet", SP.SNAP / "snapshot_manifest.json"]},
    })
    OUT.mkdir(parents=True, exist_ok=True)
    tbl = pa.Table.from_pandas(F, preserve_index=False)
    md = dict(tbl.schema.metadata or {})
    md[b"features_meta"] = json.dumps(meta["features"], ensure_ascii=False).encode("utf-8")
    tbl = tbl.replace_schema_metadata(md)
    pq.write_table(tbl, OUT / "features_all.parquet")
    meta["features_all_sha256"] = sha(OUT / "features_all.parquet")
    (OUT / "features_all_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str), "utf-8")
    print(json.dumps(checks, ensure_ascii=False, indent=1, default=str))
    print("rows", meta["rows"], "features", len(FEATURES), "seconds", (datetime.now() - t0).total_seconds())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
