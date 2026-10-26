"""판정 시점 감사 2: 가격 패널·M0·결과 기간·판정 구간 끝·라벨 단위 시각(부록 15 실행분, 2026-10-26).

final/code/judge.py는 한 줄도 고치지 않는다. SHA-256이 등록값과 다르면 멈춘다. 불러 쓴 뒤 경로 상수와 칸 등록만 메모리에서
judge_run.py와 같게 바꾼다(파일 복사 없음). 계산 함수(build_panel·expanding_m0·run_all)는 그대로 부른다.
시험
  P0 재현: 같은 입력으로 run_all(200회 이동)을 다시 돌려 judge_real.json의 판정 4칸 숫자와 같은지.
  P1 결과 기간: prices.parquet와 로더 달력만으로 따로 셈한 ln HL(D)·R5(D 시가 → D+4 종가)·R1·MDD가 패널 fut_* 열과 같은지,
     끝 세션이 2026-10-23 이하인지, D = 창 마감 날짜·KRX/NYSE 세션인지, P = 직전 세션인지.
  P2 가격 교란: 마감 T 창의 '쓸 수 있는 가격'(KRX D−1 종가까지·D 시가, 미국 D−1 세션까지, NQ는 창 마감까지) 밖의 값을
     모두 바꾼 패널에서 m0_*·wd_* 열이 그대로인지(국내 창·미국 창 각각 여러 T).
  P3 M0 확장 OLS 교란: 판정 4칸의 특징·M0 열을 T 이후 행에서 모두 바꿔도 T 앞 행의 X̂·X⊥가 그대로인지, 달마다 적합 표본의
     마지막 D가 그 달 첫날 전인지.
  P4 rz8 교란(예측 몫 P_share_rz8): T 뒤 값을 바꿔도 T 앞 rz8이 그대로인지, 저장값을 재현하는지.
  P5 라벨 단위 시각: 판정에 들어간 예측 층 단위가 모두 창 [open, close) 안인지, 7/1 이후 창에 6월 단위가 없는지.
  P6 NQ 60분봉: 봉 시작 분 분포, 쓰인 봉의 끝 시각이 창 마감 이하인지.
산출: audit_timing/results/audit_price.json
실행: py -B audit_price.py
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import copy  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from datetime import date, datetime, timedelta  # noqa: E402
from pathlib import Path  # noqa: E402
from zoneinfo import ZoneInfo  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
AUD = HERE.parent
RUN = AUD.parent
FINAL = RUN.parent
ANT = FINAL.parent
RES = RUN / "results"
OUT = AUD / "results"
JUDGE = FINAL / "code" / "judge.py"
JUDGE_SHA = "ddb1dec40a91364e148c4eb411bd184fd6b7da7a67efcb3affa8519a99575c8c"
KST = ZoneInfo("Asia/Seoul")
ET = ZoneInfo("America/New_York")
A12_DEMOTE = {"H35": "R10", "H46": "R10", "H31": "MDD10", "H68": "R10"}
LATE = {"H70_late": ("H70", "P_share_rz8_late"), "H66_late": ("H66", "P_bar_late")}
CELLS4 = ["H03", "H08", "H70", "H66"]
KR_T = ["^KS11", "000660.KS", "005930.KS", "^KQ11"]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_judge():
    j = sha(JUDGE)
    if j != JUDGE_SHA:
        raise SystemExit(f"멈춤: judge.py SHA-256 {j} ≠ 등록값 — 판정 관련 계산을 하지 않는다")
    spec = importlib.util.spec_from_file_location("judge_frozen_audit", JUDGE)
    J = importlib.util.module_from_spec(spec)
    sys.modules["judge_frozen_audit"] = J
    spec.loader.exec_module(J)
    # judge_run.patch와 같은 경로·칸 등록(메모리만, 파일 복사 없음)
    J.RES = RES
    J.FINAL = RUN / "pred_all"
    for f in (J.load_windows, J.load_features, J.load_pred_features):
        f.__defaults__ = (RES,)
    assert sha(RES / "pred_windows.parquet") == sha(RES / "pred_windows_full.parquet")
    for cid, tgt in A12_DEMOTE.items():
        c = J.CELL[cid]
        assert c["role"] == "main", cid
        c["role"] = "expl"
        c["why"] = f"부록 12: 목표 {tgt}(10일 이상) → 설명용(판정 칸 수에 넣지 않음)"
    for lid, (base, x) in LATE.items():
        s = copy.deepcopy(J.CELL[base])
        s.update({"id": lid, "role": "expl", "x": x, "family": lid, "label": s["label"] + " — _late(7/1 이후 라벨만)",
                  "why": "부록 11 #1 _late 강건판"})
        J.CELLS.append(s)
        J.CELL[lid] = s
    return J, j


def add_late_columns(inp: dict) -> None:
    pw = pd.read_parquet(RES / "pred_windows_full.parquet")
    late = pd.DataFrame({"win_id": pw["win_id"], "P_share_rz8_late": pw["P_share_rz8_late"],
                         "P_bar_late": pw["P_bar"].where(pd.to_datetime(pw["D"]) >= pd.Timestamp("2026-07-01"))})
    kr = inp["feat"]["kr"].merge(late, on="win_id", how="left", validate="1:1")
    assert (kr["win_id"].to_numpy() == inp["feat"]["kr"]["win_id"].to_numpy()).all()
    inp["feat"]["kr"] = kr


def cell_nums(res: dict) -> dict:
    out = {}
    for cid in CELLS4:
        c = res["cells"][cid]
        v = c.get("values") or {}
        out[cid] = {"verdict": c.get("verdict"), "why": c.get("why"), "n_back": (c.get("info") or {}).get("n_back"),
                    "est": v.get("est"), "t": v.get("t"), "p_an": v.get("p_an"), "p_shift": v.get("p_shift"),
                    "p_final": v.get("p_final"), "p_adj": c.get("p_adj"), "est_front": v.get("est_front"),
                    "est_2025": v.get("est_2025"), "robust": v.get("robust"), "robust_ok": v.get("robust_ok"),
                    "placebo_est": v.get("placebo_est"), "placebo_x_est": v.get("placebo_x_est"),
                    "placebo_flag": v.get("placebo_flag"), "xp_est": v.get("xp_est"), "xp_p_final": v.get("xp_p_final")}
    g = res["global_rotation"]
    out["_global"] = {k: g.get(k) for k in ("mean_main_passes", "p_ge1", "max", "hist")}
    return out


def same(a, b, tol=1e-9) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return all(same(a.get(k), b.get(k), tol) for k in set(a) | set(b))
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if not (math.isfinite(a) and math.isfinite(b)):
            return (not math.isfinite(a)) and (not math.isfinite(b))
        return abs(a - b) <= tol * max(1.0, abs(a))
    return a == b


# ------------------------------------------------------------------ P1 결과 기간(따로 셈)
def outcome_check(J, frames: dict, px: dict) -> dict:
    cal = pd.read_parquet(RES / "loader_calendar.parquet")
    krx = [d.date() for d in cal.loc[cal["krx"], "date"]]
    nyse = [d.date() for d in cal.loc[cal["nyse"], "date"]]
    out = {}
    for mk, tick, sess, pre in (("kr", "^KS11", krx, "kospi"), ("us", "SOXX", nyse, "soxx")):
        df = frames[mk]
        p = px[tick]
        # 판정 가격 패널 날짜 = 달력 세션인가(빈 막대 없이)
        sset = set(sess)
        pidx = [d for d in p.index if d >= sess[0]]
        miss_bar = sorted(str(d) for d in sset - set(pidx) if d <= date(2026, 10, 23) and d >= p.index.min())
        extra_bar = sorted(str(d) for d in set(pidx) - sset)
        pos = {d: i for i, d in enumerate(sess)}
        o, h, l, c = (p.reindex(sess)[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        rows = []
        for j, (D, P, wid, ck) in enumerate(zip(df["Dd"], df["Pd"], df["win_id"], df["close_kst"])):
            i = pos.get(D)
            r = {"win_id": wid, "D_is_session": i is not None, "P_is_prev_session": bool(i is not None and i > 0 and sess[i - 1] == P)}
            if mk == "kr":
                r["close_matches_D"] = pd.Timestamp(ck) == pd.Timestamp(datetime(D.year, D.month, D.day, 8, 0))
            else:
                ce = pd.Timestamp(datetime(D.year, D.month, D.day, 8, 30, tzinfo=ET).astimezone(KST).replace(tzinfo=None))
                r["close_matches_D"] = pd.Timestamp(ck) == ce
            ind = {}
            if i is not None and np.isfinite(o[i]) and o[i] > 0:
                for hh in J.HORIZONS:
                    e = i + hh - 1
                    ind[f"R{hh}"] = (c[e] / o[i] - 1) if e < len(sess) and sess[e] <= J.JUDGE_END and np.isfinite(c[e]) else np.nan
                if np.isfinite(h[i]) and np.isfinite(l[i]) and l[i] > 0 and sess[i] <= J.JUDGE_END:
                    hl = math.log(h[i] / l[i])
                    ind["lnHL"] = math.log(hl) if hl > 0 else np.nan
                else:
                    ind["lnHL"] = np.nan
            else:
                ind = {f"R{hh}": np.nan for hh in J.HORIZONS}
                ind["lnHL"] = np.nan
            for k, v in ind.items():
                pv = df[f"fut_{pre}_{k}"].iloc[j]
                r[f"{k}_ok"] = bool((np.isnan(v) and np.isnan(pv)) or (np.isfinite(v) and np.isfinite(pv) and abs(v - pv) <= 1e-12 * max(1, abs(v))))
            # 끝 세션(패널에 값이 있는 목표만): 달력 기준 D+h−1
            ends = []
            if i is not None:
                for hh in J.HORIZONS:
                    if np.isfinite(df[f"fut_{pre}_R{hh}"].iloc[j]):
                        e = i + hh - 1
                        ends.append(str(sess[e]) if e < len(sess) else "달력 밖")
            r["max_end_session"] = max(ends) if ends else None
            rows.append(r)
        R = pd.DataFrame(rows)
        ok_cols = [c for c in R.columns if c.endswith("_ok")]
        out[mk] = {"windows": int(len(R)), "D_is_session_false": R.loc[~R["D_is_session"], "win_id"].tolist(),
                   "P_not_prev_session": R.loc[~R["P_is_prev_session"], "win_id"].tolist(),
                   "close_not_D": R.loc[~R["close_matches_D"], "win_id"].tolist(),
                   "mismatch_by_target": {c[:-3]: R.loc[~R[c], "win_id"].tolist() for c in ok_cols},
                   "max_end_session_any": R["max_end_session"].dropna().max() if "max_end_session" in R else None,
                   "price_bars_missing_on_calendar_days": miss_bar, "price_bars_off_calendar": extra_bar}
    # 판정 칸 뒤 구간 마지막 D와 목표 끝 세션
    return out


# ------------------------------------------------------------------ P2 가격 교란
def _jitter(df: pd.DataFrame, rows: np.ndarray, cols: list[str], rng) -> None:
    if rows.any():
        for c in cols:
            v = df.loc[rows, c].to_numpy(float)
            df.loc[rows, c] = v * np.exp(0.2 * rng.standard_normal(len(v))) + (0 if c != "volume" else 1.0)


def perturb_prices(px: dict, nq: pd.DataFrame, market: str, T: date, seed: int) -> tuple[dict, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    p2 = {}
    for t, df in px.items():
        d = df.copy()
        idx = np.array(list(d.index))
        if market == "kr":       # 국내 창 D = T: KRX D−1 종가까지 + D 시가, 미국 D−1(달력) 세션까지
            if t in KR_T:
                _jitter(d, idx > T, ["open", "high", "low", "close", "volume"], rng)
                _jitter(d, idx == T, ["high", "low", "close", "volume"], rng)
            else:
                _jitter(d, idx >= T, ["open", "high", "low", "close", "volume"], rng)
        else:                    # 미국 창 D = T: 미국 D−1 세션까지 + D 시가, KRX D 종가까지(15:30 KST < 창 마감)
            if t in KR_T:
                _jitter(d, idx > T, ["open", "high", "low", "close", "volume"], rng)
            else:
                _jitter(d, idx > T, ["open", "high", "low", "close", "volume"], rng)
                _jitter(d, idx == T, ["high", "low", "close", "volume"], rng)
        p2[t] = d
    q = nq.copy()
    if market == "kr":
        lim = pd.Timestamp(datetime(T.year, T.month, T.day, 7, 59, tzinfo=KST).astimezone(ET).replace(tzinfo=None))
    else:                        # 08:29 ET 마감(분 포함) → 바 끝 08:30 이하는 쓸 수 있다(judge 질의 시각과 같음)
        lim = pd.Timestamp(datetime(T.year, T.month, T.day, 8, 30))
    m = pd.to_datetime(q["bar_end_et"]) > lim
    q.loc[m, "close"] = q.loc[m, "close"].to_numpy(float) * np.exp(0.2 * rng.standard_normal(int(m.sum())))
    return p2, q


def price_perturb_test(J, px, nq, cal, windows) -> dict:
    base = J.build_panel(px, nq, windows, cal)
    out = {}
    for mk in ("kr", "us"):
        b = base[mk]
        cols = [c for c in b.columns if c.startswith(("m0_", "wd_"))]
        Ds = b["Dd"].to_numpy()
        cand = sorted(set(Ds))
        picks = [cand[k] for k in np.linspace(25, len(cand) - 1, 24).astype(int)]
        recs = []
        for k, T in enumerate(picks):
            p2, q2 = perturb_prices(px, nq, mk, T, 1000 + k)
            pn = J.build_panel(p2, q2, windows, cal)[mk]
            assert (pn["win_id"].to_numpy() == b["win_id"].to_numpy()).all()
            past = Ds <= T
            fut = Ds > T
            bad = {}
            changed_future = 0
            for c in cols:
                x, y = b[c].to_numpy(float), pn[c].to_numpy(float)
                eq = (np.isnan(x) & np.isnan(y)) | (np.abs(x - y) <= 1e-12 * np.maximum(1, np.abs(x)))
                if (~eq & past).any():
                    bad[c] = [str(w) for w in b.loc[~eq & past, "win_id"].head(5)]
                changed_future += int((~eq & fut).any())
            fut_cols = [c for c in b.columns if c.startswith("fut_")]
            fut_changed_at_T = sorted(c for c in fut_cols
                                      if not np.allclose(b.loc[Ds == T, c].to_numpy(float), pn.loc[Ds == T, c].to_numpy(float),
                                                         equal_nan=True))
            recs.append({"T": str(T), "m0_cols": len(cols), "m0_cols_changed_D_le_T": bad,
                         "m0_cols_changed_D_gt_T(검정력)": changed_future,
                         "fut_cols_changed_at_D_eq_T": len(fut_changed_at_T)})
        out[mk] = {"cutoffs": len(picks), "any_leak": any(r["m0_cols_changed_D_le_T"] for r in recs), "runs": recs}
    return out


# ------------------------------------------------------------------ P3 M0 확장 OLS 교란
def m0_perturb_test(J, frames: dict) -> dict:
    out = {}
    for cid in CELLS4:
        s = J.CELL[cid]
        df = frames[s["market"]].copy().reset_index(drop=True)
        cont, dums = (J.M0_KR_CONT, J.M0_KR_DUM) if s["market"] == "kr" else (J.M0_US_CONT, J.M0_US_DUM)
        extras = s.get("m0_extra", [])
        x0 = df[s["x"]].to_numpy(float)
        fit_ok = df["judge_ok"].to_numpy(bool) & np.isfinite(x0)
        X = x0[:, None]
        r0 = J.expanding_m0(df, X, cont, dums, extras, fit_ok)
        Dt = pd.to_datetime(df["D"])
        # 달마다 적합 표본 마지막 D < 그 달 첫날
        allc = cont + dums + [e for e, _ in extras]
        zfin = np.isfinite(df[allc].to_numpy(float)).all(1)
        mon_rec = []
        for m in r0["months"]:
            start = pd.Timestamp(m["month"] + "-01")
            fit = fit_ok & zfin & (Dt < start).to_numpy()
            mon_rec.append({"month": m["month"], "n_fit": m["n_fit"], "fitted": "R2_obs" in m,
                            "last_fit_D": str(Dt[fit].max().date()) if fit.any() else None,
                            "last_fit_D_lt_month_start": bool((Dt[fit] < start).all())})
        tests = []
        cuts = [pd.Timestamp(x) for x in ("2025-11-14", "2026-02-11", "2026-04-15", "2026-06-17", "2026-07-15",
                                           "2026-08-03", "2026-09-10", "2026-10-07")]
        rng = np.random.default_rng(7)
        for T in cuts:
            d2 = df.copy()
            fut = (Dt >= T).to_numpy()
            for c in allc:
                v = d2[c].to_numpy(float).copy()
                v[fut] = v[fut] + rng.standard_normal(int(fut.sum())) * (np.nanstd(v) + 1.0)
                d2[c] = v
            x2 = x0.copy()
            x2[fut] = x2[fut] + rng.standard_normal(int(fut.sum())) * (np.nanstd(x0) + 1.0)
            fit2 = df["judge_ok"].to_numpy(bool) & np.isfinite(x2)
            r2 = J.expanding_m0(d2, x2[:, None], cont, dums, extras, fit2)
            past = ~fut
            dx = {}
            for k in ("xhat", "xperp"):
                a, b = r0[k][past, 0], r2[k][past, 0]
                eq = (np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= 1e-9 * np.maximum(1, np.abs(a)))
                dx[k] = int((~eq).sum())
            fa_, fb_ = r0["xperp"][fut, 0], r2["xperp"][fut, 0]
            power = int((~((np.isnan(fa_) & np.isnan(fb_)) | (np.abs(fa_ - fb_) <= 1e-9))).sum())
            tests.append({"T": str(T.date()), "rows_D_lt_T_changed": dx, "rows_D_ge_T_changed(검정력)": power})
        out[cid] = {"x": s["x"], "market": s["market"], "months": mon_rec,
                    "months_bad": [m for m in mon_rec if m["fitted"] and not m["last_fit_D_lt_month_start"]],
                    "perturb": tests, "any_leak": any(sum(t["rows_D_lt_T_changed"].values()) for t in tests)}
    return out


# ------------------------------------------------------------------ P4 P_share rz8
def rz8_test() -> dict:
    sys.path.insert(0, str(ANT / "interim" / "code"))
    import feat_text as FX  # noqa: E402
    pw = pd.read_parquet(RES / "pred_windows_full.parquet")
    w = pd.read_parquet(RES / "windows_kr.parquet")
    w = w[(w["variant"] == "main") & w["complete"]].sort_values("close_kst").reset_index(drop=True)
    m = w[["win_id", "outage", "main_kind", "weekday"]].merge(pw[["win_id", "P_share", "P_share_rz8"]], on="win_id",
                                                              how="left", validate="1:1")
    valid = ~m["outage"].fillna(False).astype(bool).to_numpy()
    hol = (m["main_kind"] == "holiday").to_numpy()
    wd = m["weekday"].to_numpy()
    x = m["P_share"].to_numpy(float)
    r = FX.rz8(x, valid, wd, hol)
    s = m["P_share_rz8"].to_numpy(float)
    repro = int((~((np.isnan(r) & np.isnan(s)) | (np.abs(r - s) <= 1e-12))).sum())
    rng = np.random.default_rng(3)
    leaks = []
    for k in np.linspace(60, len(x) - 2, 12).astype(int):
        x2 = x.copy()
        x2[k:] = rng.random(len(x) - k)
        r2 = FX.rz8(x2, valid, wd, hol)
        leaks.append(int((~((np.isnan(r[:k]) & np.isnan(r2[:k])) | (np.abs(r[:k] - r2[:k]) <= 1e-12))).sum()))
    return {"windows": int(len(x)), "reproduce_saved_diff": repro, "cut_positions": 12, "pre_cut_changed": leaks}


# ------------------------------------------------------------------ P5 라벨 단위 시각
def label_timing() -> dict:
    w = pd.read_parquet(RES / "windows_kr.parquet")
    w = w[w["variant"] == "main"][["win_id", "open_kst", "close_kst", "D"]]
    out = {}
    for name, p in (("pred_all", RUN / "pred_all" / "label_key.parquet"), ("preview", ANT / "preview" / "label_key.parquet")):
        k = pd.read_parquet(p)
        m = k.merge(w, on="win_id", how="left", suffixes=("", "_w"))
        dt = pd.to_datetime(m["dt_kst"])
        inside = (dt >= pd.to_datetime(m["open_kst"])) & (dt < pd.to_datetime(m["close_kst"]))
        late = pd.to_datetime(m["D_w"]) >= pd.Timestamp("2026-07-01")
        june = late & (dt < pd.Timestamp("2026-07-01"))
        pdt = pd.to_datetime(m["post_dt_kst"]) if "post_dt_kst" in m else None
        out[name] = {"units": int(len(m)), "no_window": int(m["open_kst"].isna().sum()),
                     "outside_window": int((~inside & m["open_kst"].notna()).sum()),
                     "outside_examples": m.loc[~inside & m["open_kst"].notna(), ["id", "win_id", "dt_kst"]].head(5).astype(str).to_dict("records"),
                     "units_in_judge_windows(D>=7/1)": int(late.sum()), "june_units_in_judge_windows": int(june.sum()),
                     "dt_conf_counts": m["dt_conf"].value_counts().to_dict() if "dt_conf" in m else None,
                     "comment_before_post": int(((m["utype"] == "comment") & (dt < pdt)).sum()) if pdt is not None else None}
    pw = pd.read_parquet(RES / "pred_windows_full.parquet")
    jw = pw[pd.to_datetime(pw["D"]) >= pd.Timestamp("2026-07-01")]
    out["pred_windows_full"] = {"judge_windows": int(len(jw)),
                                "n_units_dropped_knowledge_sum": int(jw["n_units_dropped_knowledge"].sum()),
                                "windows_with_dropped": jw.loc[jw["n_units_dropped_knowledge"] > 0, "win_id"].tolist()}
    return out


# ------------------------------------------------------------------ P6 NQ
def nq_check(frames: dict) -> dict:
    nq = pd.read_parquet(RES / "prices_nq_1h.parquet")
    st = pd.to_datetime(nq["bar_start_et"])
    en = pd.to_datetime(nq["bar_end_et"])
    mins = st.dt.minute.value_counts().to_dict()
    dur = ((en - st) / pd.Timedelta(minutes=1)).value_counts().to_dict()
    # 미국 창: b = 바 끝 ≤ D 08:30 ET 마지막 바 → 그 바 끝이 08:29 ET 마감 + 1분 이하인지(마감 분 포함)
    us = frames["us"]
    t = en.to_numpy("datetime64[m]")
    q = np.array([np.datetime64(datetime(d.year, d.month, d.day, 8, 30), "m") for d in us["Dd"]])
    k = np.searchsorted(t, q, side="right") - 1
    used_end = pd.to_datetime(t[np.clip(k, 0, None)])
    lim = pd.to_datetime(q)
    kr = frames["kr"]
    qk = np.array([np.datetime64(datetime(d.year, d.month, d.day, 7, 59, tzinfo=KST).astimezone(ET).replace(tzinfo=None), "m")
                   for d in kr["Dd"]])
    kk = np.searchsorted(t, qk, side="right") - 1
    used_k = pd.to_datetime(t[np.clip(kk, 0, None)])
    return {"bars": int(len(nq)), "start_minute_counts": {int(a): int(b) for a, b in mins.items()},
            "duration_min_counts": {float(a): int(b) for a, b in dur.items()},
            "us_used_bar_end_after_limit": int((used_end > lim).sum()),
            "us_used_bar_end_minutes_before_0830_median": float(np.median(np.asarray((lim - used_end) / pd.Timedelta(minutes=1), float))),
            "kr_used_bar_end_after_0759kst": int((used_k > pd.to_datetime(qk)).sum())}


def main() -> int:
    t0 = time.time()
    J, jsha = load_judge()
    rep = {"created": datetime.now(KST).replace(microsecond=0).isoformat(), "judge_sha256": jsha,
           "judge_sha256_registered": jsha == JUDGE_SHA}
    opts = {"h31_reg_date": None}
    inp = J.prepare_inputs(opts)
    add_late_columns(inp)
    px, nq, cal, src = J._read_real_daily()
    J._assert_real_data(px)
    panel = J.build_panel(px, nq, inp["windows"], cal)
    frames = J.build_frames(inp["feat"], {"kr": panel["kr"], "us": panel["us"]})
    rep["prices_max_date"] = {t: str(max(d.index)) for t, d in px.items()}
    # P0 재현
    res = J.run_all(frames, opts, inp["chk"], n_global=J.N_GLOBAL)
    now = json.loads(json.dumps(J.clean(cell_nums(res))))       # 저장 파일과 같은 JSON 꼴(히스토그램 키 문자열)
    saved = cell_nums(json.loads((RES / "judge_real.json").read_text("utf-8")))
    rep["P0_reproduce"] = {"same_as_judge_real": {k: same(now[k], saved[k]) for k in now}, "now": now}
    print("P0", rep["P0_reproduce"]["same_as_judge_real"], round(time.time() - t0), flush=True)
    # 판정 칸 뒤 구간 끝
    rep["P1_back_window_ends"] = {cid: {"back": res["cells"][cid]["info"].get("back"),
                                        "first_D_back": res["cells"][cid]["info"].get("first_D_back"),
                                        "last_D_back": res["cells"][cid]["info"].get("last_D_back"),
                                        "h": J.CELL[cid]["h"]} for cid in CELLS4}
    rep["P1_outcome"] = outcome_check(J, frames, px)
    print("P1 done", round(time.time() - t0), flush=True)
    rep["P2_price_perturb"] = price_perturb_test(J, px, nq, cal, inp["windows"])
    print("P2", {k: v["any_leak"] for k, v in rep["P2_price_perturb"].items()}, round(time.time() - t0), flush=True)
    rep["P3_m0_perturb"] = m0_perturb_test(J, frames)
    print("P3", {k: v["any_leak"] for k, v in rep["P3_m0_perturb"].items()}, round(time.time() - t0), flush=True)
    rep["P4_rz8"] = rz8_test()
    rep["P5_label_timing"] = label_timing()
    rep["P6_nq"] = nq_check(frames)
    rep["seconds"] = round(time.time() - t0)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit_price.json").write_text(json.dumps(J.clean(rep), ensure_ascii=False, indent=1, default=str), "utf-8")
    print(json.dumps(J.clean({k: rep[k] for k in ("P4_rz8", "P5_label_timing", "P6_nq")}), ensure_ascii=False, indent=1, default=str))
    print("wrote", OUT / "audit_price.json", rep["seconds"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
