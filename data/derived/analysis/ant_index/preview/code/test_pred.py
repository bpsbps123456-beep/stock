"""미리보기(부록 9) 예측 층 2단계: H66·H67·H70(+H68 설명용) 시험과 설명용 표 3개, 그림 자료.

지위: 미리보기다. 동결 판정을 대신하지 않는다. 예측 라벨은 2026-07-01 이후 창만 있어 앞 구간(1~5월)이 없다.
정의: 부록 4 판정 칸 그대로. 계산 틀은 test_free.py(같은 미리보기의 사전 없는 칸)의 함수를 그대로 불러 쓴다
      (HAR-K·M0·Hodrick·원형 이동·메아리 위약·강건 3개). 결과를 보기 전에 정한 해석은 아래 상수와 md '해석' 절에 적는다.

가격 보기 전에 정한 것:
- H66 X = P̄(작성자당 1회). M0 확장 OLS(최소 120창)는 창이 72개뿐이라 못 한다 → 같은 M0 변수의 표본 안 OLS 잔차
  ('미리보기 대체', test_free.m0_hat). 판정 통제 = KOSPI r(D−1)·r5·r20·ln RV20·g(D) + 전야 SOX(§2.3).
  보조: 단위 그대로판(작성자 1회 없음), 예측 쌍 5개 이상 창만.
- H67 등록 특징 P_sd rz8은 값이 0창(pred_windows_meta) → 시험 불가. 비등록 설명판 = P_sd_lite 원값(rz8 없음)을
  같은 HAR-K에 넣는다(판정·보정에 넣지 않음).
- H70 X = P_share rz8(30창). 보조: P_share 원값(rz8 없음).
- H68 설명용: X = P_strong1(분모 = 예측 쌍)의 M0 잔차 → KOSPI R10 IC. 보조: 분모 전체 쌍판.
- 단계표: 확장 창 백분위(자기 제외, 직전 모든 유효 창, 최소 30창). 단계 수: 단계당 기대 창 수(n/10)가 5 미만이면 5단계.
- 주 표: 주 창(월 08:00 마감) 안 표본 단위를 모아 작성자당 1회로 다시 센 P̄. 진입 = 마감 뒤 첫 세션 시가
  (KOSPI 월 09:00, SOXX 월 밤 미국 시가). 다음 주 = 다음 월 전 마지막 세션 종가, 2주 = 그다음 월 전 마지막 세션 종가.

입력(읽기만): preview/results/pred_windows.parquet·pred_weeks.parquet, test_free.load()가 읽는 파일들
산출: preview/results/test_pred.json, preview/results/chart_data.json
실행: py test_pred.py
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_free as TF  # noqa: E402

PREVIEW = HERE.parent
ANT = PREVIEW.parent
RES = PREVIEW / "results"

H66_CTRL = ["m0_kospi_r", "m0_kospi_r5", "m0_kospi_r20", "m0_kospi_lnRV20", "m0_kospi_g", "m0_sox_r_u"]
STAGE_MIN = 30
STAGE_K_WANT = 10
STAGE_PER_MIN = 5
PRED_COLS = ["P_bar", "P_bar_unit", "P_share", "P_share_unit", "P_share_rz8", "P_sd_raw", "P_sd", "P_sd_rz8",
             "P_sd_lite", "P_strong1", "P_strong5", "P_strong1_all", "n_pred", "n_pairs", "n_units"]


def sha(p: Path) -> str:
    return TF.sha(p)


def load():
    Dt = TF.load()
    kr, P = Dt["kr"], Dt["px"]
    F = pd.read_parquet(RES / "pred_windows.parquet")
    kr = kr.merge(F[["win_id"] + PRED_COLS], on="win_id", how="left", validate="1:1")
    kr["P_bar_n5"] = kr["P_bar"].where(kr["n_pred"] >= 5)
    return kr, P, F


# ---------------------------------------------------------------- 설명용
def follow_table(kr: pd.DataFrame) -> dict:
    """군중 예측이 가격을 따라가는가: P̄(그리고 P_share)와 지난 1·5·20일 KOSPI 수익(창 마감 전에 끝난 세션)."""
    j = kr["judge_ok"].to_numpy(bool) & kr["P_bar"].notna().to_numpy()
    d = kr[j]
    rows = []
    for feat in ("P_bar", "P_share"):
        for pc, lab in (("m0_kospi_r", "지난 1일"), ("m0_kospi_r5", "지난 5일"), ("m0_kospi_r20", "지난 20일"),
                        ("m0_sox_r_u", "간밤 SOX")):
            ok = d[[feat, pc]].notna().all(axis=1)
            x, y = d.loc[ok, feat].to_numpy(float), d.loc[ok, pc].to_numpy(float)
            ks = TF.shifts(len(x))
            noise = [abs(stats.spearmanr(np.roll(x, k), y)[0]) for k in ks]
            rows.append({"feature": feat, "past": lab, "col": pc, "n": int(ok.sum()),
                         "spearman": float(stats.spearmanr(x, y)[0]), "pearson": float(np.corrcoef(x, y)[0, 1]),
                         "shift_K": len(ks), "p95_abs_spearman_shift": float(np.percentile(noise, 95)) if ks else None})
    return {"rows": rows, "first_D": str(d["Dd"].min()), "last_D": str(d["Dd"].max()),
            "rule": "판정 자격 창. 지난 수익 = 창 안에서 끝난 마지막 KRX 세션까지(m0_* 열). 우연 범위 = P̄열을 21~(N−21)칸 원형 "
                    "이동한 |Spearman| 95% 분위"}


def raw_ic_table(kr: pd.DataFrame) -> dict:
    """통제 없는 원 상관: P̄ → 앞날 KOSPI R1·R5·R10, P_share → 다음 장 ln HL. 원형 이동 p(양측 |ρ|)."""
    j = kr["judge_ok"].to_numpy(bool)
    rows = []
    for feat, ycol in (("P_bar", "fut_kospi_R1"), ("P_bar", "fut_kospi_R5"), ("P_bar", "fut_kospi_R10"),
                       ("P_share", "fut_kospi_lnHL"), ("P_bar", "fut_kospi_lnHL")):
        ok = j & kr[[feat, ycol]].notna().all(axis=1).to_numpy()
        x, y = kr.loc[ok, feat].to_numpy(float), kr.loc[ok, ycol].to_numpy(float)
        rho = float(stats.spearmanr(x, y)[0])
        ks = TF.shifts(len(x))
        nl = np.array([stats.spearmanr(np.roll(x, k), y)[0] for k in ks])
        p2 = float((1 + np.sum(np.abs(nl) >= abs(rho))) / (1 + len(nl))) if ks else None
        rows.append({"feature": feat, "target": ycol, "n": int(ok.sum()), "spearman": rho, "shift_K": len(ks),
                     "p_shift_two_sided": p2, "first_D": str(kr.loc[ok, "Dd"].min()),
                     "last_D": str(kr.loc[ok, "Dd"].max())})
    return {"rows": rows, "note": "설명용. M0·통제 없음. R5·R10은 겹치는 목표라 해석 p 대신 원형 이동 p만 낸다"}


def stage_table(kr: pd.DataFrame) -> dict:
    """확장 창 백분위(자기 제외, 직전 모든 유효 창, 최소 STAGE_MIN) → 단계. 결과는 판정 자격 창만."""
    lab = kr["P_bar"].notna().to_numpy()
    d = kr[lab].reset_index(drop=True)
    x = d["P_bar"].to_numpy(float)
    pct = np.full(len(x), np.nan)
    for i in range(len(x)):
        h = x[:i]
        h = h[np.isfinite(h)]
        if len(h) >= STAGE_MIN:
            pct[i] = (np.sum(h < x[i]) + 0.5 * np.sum(h == x[i])) / len(h)
    d["pct_exp"] = pct
    jj = d["judge_ok"].to_numpy(bool) & np.isfinite(pct)
    n_staged = int(jj.sum())
    k = STAGE_K_WANT if n_staged / STAGE_K_WANT >= STAGE_PER_MIN else 5
    reason = (f"단계가 매겨진 판정 자격 창 {n_staged}개 ÷ 10단계 = 단계당 {n_staged / 10:.1f}창 < {STAGE_PER_MIN}창이라 "
              f"5단계로 냈다" if k == 5 else f"단계당 {n_staged / 10:.1f}창 ≥ {STAGE_PER_MIN}")
    d["stage"] = np.where(np.isfinite(pct), np.clip(np.ceil(k * pct), 1, k), np.nan)
    rows = []
    for s in range(1, k + 1):
        m = jj & (d["stage"].to_numpy() == s)
        r = {"stage": s, "n_windows": int(m.sum()),
             "P_bar_min": float(d.loc[m, "P_bar"].min()) if m.any() else None,
             "P_bar_max": float(d.loc[m, "P_bar"].max()) if m.any() else None}
        for h in (1, 5, 10):
            v = d.loc[m, f"fut_kospi_R{h}"].dropna()
            r[f"R{h}_n"] = int(len(v))
            r[f"R{h}_mean"] = float(v.mean()) if len(v) else None
            r[f"R{h}_median"] = float(v.median()) if len(v) else None
            r[f"R{h}_up_share"] = float((v > 0).mean()) if len(v) else None
        rows.append(r)
    base = {}
    for h in (1, 5, 10):
        v = d.loc[jj, f"fut_kospi_R{h}"].dropna()
        base[f"R{h}"] = {"n": int(len(v)), "mean": float(v.mean()), "median": float(v.median()),
                         "up_share": float((v > 0).mean())}
    sp = {}
    for h in (1, 5, 10):
        ok = jj & d[f"fut_kospi_R{h}"].notna().to_numpy()
        sp[f"R{h}"] = {"n": int(ok.sum()), "spearman_stage": float(stats.spearmanr(d.loc[ok, "stage"],
                                                                                  d.loc[ok, f"fut_kospi_R{h}"])[0])}
    first = d.loc[jj, "Dd"].min()
    return {"k": k, "reason": reason, "n_staged": n_staged, "first_staged_D": str(first),
            "last_staged_D": str(d.loc[jj, "Dd"].max()), "rows": rows, "all_staged": base, "stage_spearman": sp,
            "rule": f"백분위 = (직전 창 중 작은 수 + ½ 같은 수) ÷ 직전 창 수, 직전 유효 창 {STAGE_MIN}개 이상(부록 9 대체). "
                    f"단계 = ⌈k × 백분위⌉(1~k). 수익 = 진입 시가 → 진입일+h−1 종가. 5·10일은 겹친다",
            "per_window": [{"D": str(a), "stage": (None if not np.isfinite(b) else int(b)), "pct": c}
                           for a, b, c in zip(d["Dd"], d["stage"], d["pct_exp"])]}


def week_returns(P: dict, M: date, ticker: str, k: int, last_px: date):
    px = P[ticker]
    idx = sorted(i for i in px.index if np.isfinite(px.loc[i, "open"]) or np.isfinite(px.loc[i, "close"]))
    entry = next((i for i in idx if i >= M and np.isfinite(px.loc[i, "open"])), None)
    end_lim = M + timedelta(days=7 * k)
    if entry is None or last_px < end_lim - timedelta(days=3):
        return None, None, None
    ends = [i for i in idx if entry <= i < end_lim and np.isfinite(px.loc[i, "close"])]
    if not ends:
        return None, None, None
    e = ends[-1]
    return float(px.loc[e, "close"] / px.loc[entry, "open"] - 1), str(entry), str(e)


def week_table(P: dict) -> dict:
    W = pd.read_parquet(RES / "pred_weeks.parquet")
    last_px = min(max(P["^KS11"].index), max(P["SOXX"].index))
    rows = []
    for _, w in W.iterrows():
        M = pd.Timestamp(w["week_close"]).date()
        r = {"week_close": str(pd.Timestamp(w["week_close"])), "complete": bool(w["complete"]), "note": w["note"],
             "n_units": int(w["n_units"]), "n_pred": int(w["n_pred"]), "n_pairs": int(w["n_pairs"]),
             "P_bar": float(w["P_bar"]), "P_share": float(w["P_share"])}
        for tk, lab in (("^KS11", "kospi"), ("SOXX", "soxx")):
            for k in (1, 2):
                v, a, b = week_returns(P, M, tk, k, last_px)
                r[f"{lab}_R{k}w"] = v
                r[f"{lab}_R{k}w_span"] = f"{a}→{b}" if v is not None else None
        rows.append(r)
    df = pd.DataFrame(rows)
    c = df[df["complete"]]
    corr = {}
    for col in ("kospi_R1w", "kospi_R2w", "soxx_R1w", "soxx_R2w"):
        ok = c[col].notna()
        x, y = c.loc[ok, "P_bar"].to_numpy(float), c.loc[ok, col].to_numpy(float)
        nl = [abs(stats.spearmanr(np.roll(x, s), y)[0]) for s in range(1, len(x))]
        med = np.median(x)
        hi, lo = y[x > med], y[x <= med]
        corr[col] = {"n_weeks": int(ok.sum()), "spearman": float(stats.spearmanr(x, y)[0]),
                     "pearson": float(np.corrcoef(x, y)[0, 1]),
                     "max_abs_spearman_all_shifts": float(max(nl)) if nl else None,
                     "p_shift_two_sided_all_shifts": float((1 + sum(v >= abs(stats.spearmanr(x, y)[0]) for v in nl))
                                                           / (1 + len(nl))) if nl else None,
                     "mean_ret_Pbar_above_median": float(hi.mean()) if len(hi) else None,
                     "mean_ret_Pbar_le_median": float(lo.mean()) if len(lo) else None,
                     "n_above": int(len(hi)), "n_le": int(len(lo))}
    return {"rows": rows, "corr_complete_weeks": corr,
            "rule": "주 창 = (월 08:00 − 7일, 월 08:00), 단위 시각으로 배정, (주, 작성자, pred) 1회. 진입 = 마감 뒤 첫 세션 시가. "
                    "다음 주 = 다음 월 전 마지막 세션 종가, 2주 = 그다음 월 전. 원형 이동은 주 수가 적어 1~(N−1)칸 전부(설명용)"}


def chart_data(kr: pd.DataFrame, P: dict, F: pd.DataFrame) -> dict:
    W = pd.read_parquet(RES / "pred_weeks.parquet")
    md = pd.read_csv(RES / "loader_missing_by_day.csv", encoding="utf-8-sig")
    wins = []
    for _, r in kr[kr["P_bar"].notna()].iterrows():
        wins.append({"D": str(r["Dd"]), "win_id": r["win_id"], "P_bar": float(r["P_bar"]),
                     "P_share": float(r["P_share"]), "n_pred": int(r["n_pred"]), "n_pairs": int(r["n_pairs"]),
                     "n_units_sampled": int(r["n_units"]), "judge_ok": bool(r["judge_ok"]),
                     "n_posts_window_saved": int(r["n_posts"]), "d_no_window": float(r["d_no"])})
    px = {}
    for tk, lab in (("^KS11", "KOSPI"), ("SOXX", "SOXX")):
        s = P[tk]["close"].astype(float)
        s = s[[d for d in s.index if date(2026, 6, 1) <= d <= date(2026, 10, 16)]].dropna()
        px[lab] = [{"date": str(d), "close": float(v)} for d, v in s.items()]
    posts = [{"date": r["date"], "d_no": int(r["d_no"]), "saved": int(r["saved"])} for _, r in md.iterrows()]
    weeks = [{"week_close": str(pd.Timestamp(r["week_close"])), "P_bar": float(r["P_bar"]),
              "P_share": float(r["P_share"]), "n_pred": int(r["n_pred"]), "complete": bool(r["complete"])}
             for _, r in W.iterrows()]
    return {"made": datetime.now().replace(microsecond=0).isoformat(),
            "fields": {"windows": "W_KR 창(D = 진입 거래일). P_bar = 예측 쌍 평균 − 3(+ 상승 쪽), P_share = 예측 쌍 ÷ 전체 쌍, "
                                  "n_units_sampled = 그 창 AI 라벨 표본 수(층화 60, 07-01 창은 10), "
                                  "n_posts_window_saved = 창 안 저장 글 수, d_no_window = 창 안 글 번호 증가",
                       "prices": "yfinance 일봉 종가(2026-06-01~10-16)",
                       "posts_by_day": "KST 달력일 글 번호 증가(d_no, 지운 글 포함)와 저장 글 수(saved)",
                       "weeks": "주 창(월 08:00 마감) P_bar·P_share"},
            "windows": wins, "prices": px, "posts_by_day": posts, "weeks": weeks}


def har_allshift_info(kr: pd.DataFrame, xcol: str, extra: list[str]) -> dict:
    """특징 창이 42개 이하라 §2.7 ①(21~(N−21)칸)이 비는 칸의 정보용 p: 1~(N−1)칸 전부. 판정 p로 쓰지 않는다."""
    base = kr["judge_ok"].to_numpy(bool)
    F = base & kr[xcol].notna().to_numpy()
    need = ["fut_kospi_lnHL", xcol] + TF.HAR_CONT + extra + TF.M0_KR_CONT + TF.M0_KR_DUM
    S = base & kr[need].notna().all(axis=1).to_numpy()
    x = kr[xcol].to_numpy(float)
    t_obs = TF.har_fit(kr, S, x, extra)["t"]["t"]
    nf = int(F.sum())
    ks = list(range(1, nf))
    null = np.array([TF.har_fit(kr, S, TF.rolled(x, F, k), extra)["t"]["t"] for k in ks])
    r2n = [TF.m0_hat(kr, S, TF.rolled(x, F, k), TF.M0_KR_CONT, TF.M0_KR_DUM)["R2"] for k in ks]
    return {"K": len(ks), "p_one": TF.shift_p(t_obs, null, +1), "M0_R2_chance_mean": float(np.mean(r2n)),
            "note": "정보용: 1~(N−1)칸 전부 이동. 이웃 이동은 원래 열과 겹쳐 귀무가 좁다. 판정 p는 해석 p"}


# ---------------------------------------------------------------- 실행
def main() -> int:
    t0 = datetime.now().replace(microsecond=0)
    kr, P, F = load()
    eps_k = TF.cc_by_window(kr, P["^KS11"])
    cells = {}
    cells["H66"] = TF.ic_cell(kr, "H66", "P_bar", "fut_kospi_R5", "R", 5, TF.M0_KR_CONT, TF.M0_KR_DUM, H66_CTRL, eps_k,
                              -1, None, "KOSPI R5 IC",
                              aux_cols={"단위 그대로판(작성자 1회 없음, 보조)": "P_bar_unit",
                                        "예측 쌍 5개 이상 창만(보조)": "P_bar_n5"})
    cells["H66"]["m0_mode"] = "미리보기 대체: 동결 확장 OLS(최소 120창)를 못 채워 같은 M0 변수의 표본 안 단순 OLS 잔차"
    cells["H70"] = TF.har_cell(kr, "H70", "P_share_rz8", [], aux={"P_share 원값(rz8 없음, 7/01부터, 보조)": ("P_share", [])})
    cells["H70"]["shift_note"] = (f"특징 창 {cells['H70']['n_feature_windows']}개라 §2.7 ① 이동 범위(21~N−21)가 비어 원형 이동 p가 "
                                  f"없다. 판정 p = 해석 p")
    cells["H70"]["shift_all_info"] = har_allshift_info(kr, "P_share_rz8", [])
    j = kr["judge_ok"].to_numpy(bool)
    cells["H67"] = {"id": "H67", "family": "T1", "target": "KOSPI D일 ln HL (HAR-K 계수 c)", "feature": "P_sd_rz8",
                    "n_obs": int((j & kr["P_sd_rz8"].notna()).sum()), "testable": False, "expected_sign": "+",
                    "note": json.loads((RES / "pred_windows_meta.json").read_text("utf-8"))["windows"]["P_sd_reg_reason"]}
    lite = TF.har_cell(kr, "H67_lite", "P_sd_lite", [], aux={})
    lite["label"] = ("비등록 설명판: P_sd_lite(풀 직전 창 10개 이상, 예측 쌍 5개 이상) 원값, rz8 없음. 판정·보정에 넣지 않는다")
    cells["H67"]["explanatory_lite"] = lite
    cells["H68"] = TF.ic_cell(kr, "H68", "P_strong1", "fut_kospi_R10", "R", 10, TF.M0_KR_CONT, TF.M0_KR_DUM, H66_CTRL,
                              eps_k, +1, None, "KOSPI R10 IC (설명용)",
                              aux_cols={"분모 전체 쌍판(보조)": "P_strong1_all"})
    cells["H68"]["status"] = "설명용(부록 9: 표본이 적다). 판정·보정에 넣지 않는다"
    cells["H68"]["m0_mode"] = cells["H66"]["m0_mode"]
    cells["H68"]["n_windows_P_strong1_pos"] = int((j & (kr["P_strong1"] > 0)).sum())
    # 보정(미리보기): 이번 칸만, 그리고 test_free 칸과 합친 판
    tf = json.loads((RES / "test_free.json").read_text("utf-8"))["cells"]
    t1_here = {"H70": cells["H70"]["p_final_one"]}
    t2_here = {"H66": cells["H66"]["p_final_one"]}
    t1_all = {"H03": tf["H03"]["p_final_one"], "H08": tf["H08"]["p_final_one"], **t1_here}
    t2_all = {"H46": tf["H46"]["p_final_one"], "H31": tf["H31"]["p_final_one"], **t2_here}
    corr = {"T1_holm_this": TF.holm(t1_here), "T2_bh_this": TF.bh(t2_here),
            "T1_holm_with_test_free": TF.holm(t1_all), "T2_bh_with_test_free": TF.bh(t2_all),
            "T2_bh_with_test_free_noH31": TF.bh({k: v for k, v in t2_all.items() if k != "H31"}),
            "note": "미리보기 표시다. H67은 시험 불가라 T1에 넣지 않았다. 최종 판정은 실제 시험 칸 수로 다시 센다"}
    c = cells["H70"]
    c["verdict_shape"] = TF.verdict_shape(c["partial_r"], c["se_partial_r"], corr["T1_holm_with_test_free"]["H70"],
                                          0.05, c["robust_ok"], c["placebo"]["flag_70pct_same_sign"],
                                          TF.EXPECTED["T1"] / 2)
    c = cells["H66"]
    c["verdict_shape"] = TF.verdict_shape(-1 * c["ic_main_xperp_partial"], 1 / math.sqrt(c["n_eff"]),
                                          corr["T2_bh_with_test_free"]["H66"], 0.10, c["robust_ok"],
                                          c["placebo"]["flag_70pct_same_sign"], TF.EXPECTED["IC"] / 2)
    c = cells["H67"]["explanatory_lite"]
    c["shape_info_only"] = TF.verdict_shape(c["partial_r"], c["se_partial_r"], c["p_final_one"], 0.05, c["robust_ok"],
                                            c["placebo"]["flag_70pct_same_sign"], TF.EXPECTED["T1"] / 2)
    c = cells["H68"]
    c["shape_info_only"] = TF.verdict_shape(c["ic_main_xperp_partial"], 1 / math.sqrt(c["n_eff"]), c["p_final_one"],
                                            0.10, c["robust_ok"], c["placebo"]["flag_70pct_same_sign"],
                                            TF.EXPECTED["IC"] / 2)
    desc = {"weekly": week_table(P), "stages": stage_table(kr), "follow": follow_table(kr), "raw_ic": raw_ic_table(kr)}
    meta = json.loads((RES / "pred_windows_meta.json").read_text("utf-8"))
    out = {"made": t0.isoformat(), "status": "미리보기(부록 9). 동결 판정 아님. 예측 라벨은 2026-07-01 이후만, 앞 구간 없음",
           "features": meta, "cells": cells, "corrections": corr, "descriptive": desc,
           "inputs_sha256": {f: sha(RES / f) for f in ("pred_windows.parquet", "pred_weeks.parquet", "panel_kr.parquet",
                                                       "prices.parquet", "test_free.json")},
           "frozen_sha256": {f: sha(ANT / f) for f in ("HYPOTHESES.md", "HYPOTHESES_addendum4.md",
                                                       "HYPOTHESES_addendum9.md")},
           "code_sha256": {p: sha(HERE / p) for p in ("test_pred.py", "pred_windows.py", "test_free.py")}}
    (RES / "test_pred.json").write_text(json.dumps(TF.clean(out), ensure_ascii=False, indent=1, default=str), "utf-8")
    cd = chart_data(kr, P, F)
    (RES / "chart_data.json").write_text(json.dumps(TF.clean(cd), ensure_ascii=False, indent=1, default=str), "utf-8")
    for k, c in cells.items():
        print(k, {kk: c.get(kk) for kk in ("n_obs", "estimate_c", "t_hac5", "partial_r", "ic_main_xperp_partial",
                                            "ic_raw", "t", "p_analytic_one", "p_shift_one", "shift_K", "verdict_shape",
                                            "shape_info_only")})
    print(json.dumps(TF.clean(corr), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
