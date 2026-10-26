"""미리보기(부록 9) 시점 감사 ②~④와 고친 계산. 포럼 교란(①)은 audit_timing_forum.py.

시험
  S  정적: 시험 코드의 통제·M0 목록에 fut_* 열이나 진입 순간 값이 섞였는가, 특징·표본 코드가 가격·결과를 읽는가
  F  결과 교란: panel의 fut_* 값을 모두 무작위로 바꿔도(결측 자리는 그대로) 특징·M0 잔차·적합값·통제 설계가 그대로인가
  P  가격 교란: 시각 T 뒤에 열린 봉(KRX·NYSE 일봉, NQ 60분봉)을 모두 바꾸고 panel을 다시 만든다.
     마감 ≤ T 창의 m0_*가 그대로인가(진입 순간 g만 예외), 그리고 미리보기 M0 잔차(표본 안 OLS)가 그대로인가
  E  진입·결과 기간: 창 마감·진입 세션·P·직전 미국/국내 세션을 달력에서 독립으로 다시 정하고, Rh·MDDh·HL·끝 세션을
     참조 구현으로 다시 계산해 panel과 맞춘다
  L  예측 라벨 표본: 층(창 × 글/댓글)·창별 표본 수·재현(같은 시드), 표본 코드가 가격·날짜 사건 변수를 쓰는가,
     조각 순서와 날짜의 상관, 라벨 파일 필드. 라벨 교란: 마감 > T 창의 라벨을 바꾸고 일부를 지워도 마감 ≤ T 창의
     예측 특징(P̄·P_share·P_strong·P_sd·rz8)과 주 창 값이 그대로인가
  X  고친 계산: 동결 §2.2·§2.3의 '확장 OLS(D−1까지, 월 1회 재적합)' 모양으로 M0 잔차·위약 X̂를 앞 자료만으로 다시 만든다
     (최소 적합 창 30 = 부록 9 단계표 대체와 같은 수, 동결 120은 이 자료로 못 채운다). T2 칸(H46·H31·H66·H68)의
     IC·t·원형 이동 p·강건·위약·모양, T1 칸(H03·H08·H70, H67 설명판) 위약을 다시 계산한다
  D  설명용 시차표: 달력일 글 수와 '그 뒤' 수익이 시간상 겹치지 않는 판(KOSPI +1일은 다음 세션 시가→종가,
     SOXX 0일 대신 +1일부터)
읽기만: preview/results/*.parquet·json, preview/label_key.parquet, preview/labels/*.json, label_shards
쓰기: preview/results/audit_timing.json
실행: py audit_timing.py
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
PREVIEW = HERE.parent
ANT = PREVIEW.parent
RES = PREVIEW / "results"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ANT / "interim" / "code"))
import panel as PN  # noqa: E402
import test_free as TF  # noqa: E402
import test_pred as TP  # noqa: E402
import pred_windows as PW  # noqa: E402
import label_sample as LS  # noqa: E402

KST, ET = ZoneInfo("Asia/Seoul"), ZoneInfo("America/New_York")
SEED = 20261018
MIN_FIT = 30            # 부록 9 단계표 대체와 같은 최소 창 수(동결 120은 못 채움)
CLOSE_KR = time(8, 0)   # W_KR 마감(07:59 포함) = 08:00 배타
CLOSE_US_ET = time(8, 30)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def same(a, b) -> bool:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return bool(a.shape == b.shape and np.all((np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= 1e-12 * np.maximum(1, np.abs(a)))))


def n_diff(a, b) -> int:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return int(np.sum(~((np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= 1e-12 * np.maximum(1, np.abs(a))))))


# ================================================================ 자료
def schedule_from_meta() -> dict:
    s = json.loads((RES / "panel_meta.json").read_text("utf-8"))["schedule"]
    earn = sorted({date.fromisoformat(d) for v in s["earnings"].values() if isinstance(v, list) for d in v})
    return {"kr_opex": [date.fromisoformat(d) for d in s["kr_opex"]],
            "us_opex": [date.fromisoformat(d) for d in s["us_opex"]],
            "fomc": [date.fromisoformat(d) for d in s["fomc"]], "earn_days": earn}


def merge_frames(pk: pd.DataFrame, pu: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """test_free.load()·test_pred.load()와 같은 병합을 주어진 panel로 한다."""
    pk, pu = pk.reset_index(drop=True).copy(), pu.reset_index(drop=True).copy()
    for p in (pk, pu):
        p["pos"] = np.arange(len(p))
        p["Dd"] = pd.to_datetime(p["D"]).dt.date
        for k in (1, 2, 3, 4):
            p[f"wd_{k}"] = (p["m0_weekday"] == k).astype(float)
    fa = pd.read_parquet(RES / "feat_activity.parquet")
    ft = pd.read_parquet(RES / "feat_topic.parquet")
    fx = pd.read_parquet(RES / "feat_text.parquet")
    ka = fa[fa["market"] == "kr"][["win_id", "A_rz8", "A_saved_rz8", "fight", "fight_pc", "ln_n_comments_onposts",
                                   "trunc_share", "trunc_share_pc", "NW90_rz8", "NW90", "NW30_rz8"]]
    kr = pk.merge(ka, on="win_id", how="left", validate="1:1")
    tu = ft[ft["table"] == "US_D"][["win_id", "h46_dtilt", "h46_valid_win"]]
    xu = fx[fx["market"] == "us"][["win_id", "LL_US_d20", "LL_US_noNVDL_d20"]]
    us = pu.merge(tu, on="win_id", how="left", validate="1:1").merge(xu, on="win_id", how="left", validate="1:1")
    F = pd.read_parquet(RES / "pred_windows.parquet")
    kr = kr.merge(F[["win_id"] + TP.PRED_COLS], on="win_id", how="left", validate="1:1")
    kr["P_bar_n5"] = kr["P_bar"].where(kr["n_pred"] >= 5)
    return kr, us


# 칸 정의(test_free·test_pred main()과 같은 인자)
def t2_specs(kr, us) -> dict:
    valid46 = us["h46_valid_win"].fillna(False).astype(bool).to_numpy()
    return {
        "H46": dict(df="us", xcol="h46_dtilt", ycol="fut_soxx_R10", kind="R", h=10, m0c=TF.M0_US_CONT, m0d=[],
                    ctrl=TF.H46_CTRL, sign=-1, mask=valid46, eps=("us", "SOXX")),
        "H31": dict(df="us", xcol="LL_US_d20", ycol="fut_soxx_MDD10", kind="MDD", h=10, m0c=TF.M0_US_CONT, m0d=[],
                    ctrl=TF.H31_CTRL, sign=-1, mask=None, eps=("us", "SOXX")),
        "H66": dict(df="kr", xcol="P_bar", ycol="fut_kospi_R5", kind="R", h=5, m0c=TF.M0_KR_CONT, m0d=TF.M0_KR_DUM,
                    ctrl=TP.H66_CTRL, sign=-1, mask=None, eps=("kr", "^KS11")),
        "H68": dict(df="kr", xcol="P_strong1", ycol="fut_kospi_R10", kind="R", h=10, m0c=TF.M0_KR_CONT,
                    m0d=TF.M0_KR_DUM, ctrl=TP.H66_CTRL, sign=+1, mask=None, eps=("kr", "^KS11")),
    }


T1_SPECS = {"H03": ("A_rz8", []), "H08": ("fight", ["A_rz8", "ln_n_comments_onposts", "trunc_share"]),
            "H70": ("P_share_rz8", []), "H67_lite": ("P_sd_lite", [])}


# ================================================================ S 정적
def static_scan() -> dict:
    lists = {"HAR_CONT": TF.HAR_CONT, "HAR_DUM": TF.HAR_DUM, "M0_KR_CONT": TF.M0_KR_CONT, "M0_KR_DUM": TF.M0_KR_DUM,
             "M0_US_CONT": TF.M0_US_CONT, "H46_CTRL": TF.H46_CTRL, "H31_CTRL": TF.H31_CTRL, "H66_CTRL": TP.H66_CTRL}
    out = {"lists": {}}
    for k, v in lists.items():
        out["lists"][k] = {"n": len(v), "fut_cols": [c for c in v if c.startswith("fut_")],
                           "non_m0_cols": [c for c in v if not (c.startswith("m0_") or c.startswith("wd_"))],
                           "entry_instant_cols": [c for c in v if c.endswith("_g") or c.endswith("absg")]}
    feats = ["A_rz8", "A_saved_rz8", "fight", "fight_pc", "ln_n_comments_onposts", "trunc_share", "trunc_share_pc",
             "NW90_rz8", "NW30_rz8", "h46_dtilt", "LL_US_d20", "LL_US_noNVDL_d20"] + TP.PRED_COLS
    out["feature_cols_fut"] = [c for c in feats if c.startswith("fut_")]
    rx = {"fut_": re.compile(r"fut_"), "prices": re.compile(r"prices(_nq_1h)?\.parquet|load_prices|yfinance|yf\."),
          "panel": re.compile(r"panel_(kr|us)\.parquet"), "judge_ok": re.compile(r"judge_ok")}
    src = {}
    for name in ("label_sample.py", "pred_windows.py", "feature_summary.py", "run_features.py", "snapshot.py"):
        txt = (HERE / name).read_text("utf-8").splitlines()
        src[name] = {k: [i + 1 for i, l in enumerate(txt) if r.search(l) and not l.lstrip().startswith("#")
                         and '"""' not in l] for k, r in rx.items()}
    out["source_hits"] = src
    # label_sample.draw의 인자와 본문에 쓰는 열
    import inspect
    body = inspect.getsource(LS.draw)
    out["label_draw_columns_used"] = sorted(set(re.findall(r'f\["(\w+)"\]', body)))
    return out


# ================================================================ F 결과 교란
def fut_perturb_test(kr: pd.DataFrame, us: pd.DataFrame) -> dict:
    rng = np.random.default_rng(SEED)
    out = {}
    for nm, df in (("kr", kr), ("us", us)):
        d2 = df.copy()
        fc = [c for c in df.columns if c.startswith("fut_") and df[c].dtype.kind == "f"]
        for c in fc:
            v = d2[c].to_numpy(float).copy()
            m = np.isfinite(v)
            v[m] = rng.normal(size=m.sum()) * 0.3
            d2[c] = v
        out[nm] = {"fut_cols_perturbed": len(fc), "d2": d2}
    kr2, us2 = out["kr"].pop("d2"), out["us"].pop("d2")
    res = {"cells": {}}
    for cid, sp in t2_specs(kr, us).items():
        df, df2 = (kr, kr2) if sp["df"] == "kr" else (us, us2)
        base = df["judge_ok"].to_numpy(bool)
        if sp["mask"] is not None:
            base = base & sp["mask"]
        S = base & df[[sp["xcol"], sp["ycol"]] + sp["m0c"] + sp["m0d"] + sp["ctrl"]].notna().all(axis=1).to_numpy()
        x = df[sp["xcol"]].to_numpy(float)
        a = TF.m0_hat(df, S, x, sp["m0c"], sp["m0d"])
        b = TF.m0_hat(df2, S, df2[sp["xcol"]].to_numpy(float), sp["m0c"], sp["m0d"])
        Ca = df.loc[S, sp["ctrl"]].to_numpy(float)
        Cb = df2.loc[S, sp["ctrl"]].to_numpy(float)
        res["cells"][cid] = {"n": int(S.sum()), "xperp_same": same(a["xperp"], b["xperp"]),
                             "xhat_same": same(a["xhat"], b["xhat"]), "ctrl_same": same(Ca, Cb),
                             "feature_same": same(x, df2[sp["xcol"]].to_numpy(float))}
    for cid, (xcol, extra) in T1_SPECS.items():
        base = kr["judge_ok"].to_numpy(bool)
        need = ["fut_kospi_lnHL", xcol] + TF.HAR_CONT + extra + TF.M0_KR_CONT + TF.M0_KR_DUM
        S = base & kr[need].notna().all(axis=1).to_numpy()
        Xa, na, _ = TF.build_design(kr[S], [kr.loc[S, xcol].to_numpy(float)], TF.HAR_CONT, TF.HAR_DUM)
        Xb, nb, _ = TF.build_design(kr2[S], [kr2.loc[S, xcol].to_numpy(float)], TF.HAR_CONT, TF.HAR_DUM)
        a = TF.m0_hat(kr, S, kr[xcol].to_numpy(float), TF.M0_KR_CONT, TF.M0_KR_DUM)
        b = TF.m0_hat(kr2, S, kr2[xcol].to_numpy(float), TF.M0_KR_CONT, TF.M0_KR_DUM)
        res["cells"][cid] = {"n": int(S.sum()), "har_design_same": same(Xa, Xb) and na == nb,
                             "xhat_same": same(a["xhat"], b["xhat"]), "xperp_same": same(a["xperp"], b["xperp"])}
    res["fut_cols_perturbed"] = {k: v["fut_cols_perturbed"] for k, v in out.items()}
    return res


# ================================================================ P 가격 교란
def perturb_prices(px: dict, nq: pd.DataFrame, T: pd.Timestamp, seed: int) -> tuple[dict, pd.DataFrame, dict]:
    """T(KST, 시각 없는 Timestamp) 뒤에 열린 봉만 바꾼다. 일봉: KRX 09:00 KST, NYSE 09:30 ET 시가 시각 ≥ T.
    NQ 60분봉: 바 끝(KST) > T(T를 걸친 바 포함)."""
    rng = np.random.default_rng(seed)
    out, cnt = {}, {}
    for t, df in px.items():
        d = df.copy()
        if t in PN.KR_ASSETS.values():
            op = [datetime.combine(x, time(9, 0)) for x in d.index]
        else:
            op = [datetime.combine(x, time(9, 30), ET).astimezone(KST).replace(tzinfo=None) for x in d.index]
        m = np.array([o >= T.to_pydatetime() for o in op])
        f = np.exp(rng.normal(0, 0.05, m.sum()))
        for c in ("open", "high", "low", "close"):
            v = d[c].to_numpy(float).copy()
            v[m] = v[m] * f
            d[c] = v
        vol = d["volume"].to_numpy(float).copy()
        vol[m] = vol[m] * np.exp(rng.normal(0, 0.5, m.sum()))
        d["volume"] = vol
        out[t] = d
        cnt[t] = int(m.sum())
    q = nq.copy()
    end_kst = pd.DatetimeIndex(q["bar_end_et"]).tz_localize(ET).tz_convert(KST).tz_localize(None)
    mq = np.asarray(end_kst > T)
    fq = np.exp(rng.normal(0, 0.02, mq.sum()))
    for c in ("open", "high", "low", "close"):
        v = q[c].to_numpy(float).copy()
        v[mq] = v[mq] * fq
        q[c] = v
    cnt["NQ=F_1h"] = int(mq.sum())
    return out, q, cnt


def price_perturb_test() -> dict:
    px, _ = PN.load_prices()
    nq = pd.read_parquet(RES / "prices_nq_1h.parquet")
    sch = schedule_from_meta()
    k0, u0 = PN.build_kr(px, sch), PN.build_us(px, sch, nq)
    saved_k, saved_u = pd.read_parquet(RES / "panel_kr.parquet"), pd.read_parquet(RES / "panel_us.parquet")
    rep = {"rebuild_vs_saved": {}}
    for nm, a, b in (("kr", k0, saved_k), ("us", u0, saved_u)):
        cols = [c for c in b.columns if c.startswith(("m0_", "fut_")) and b[c].dtype.kind in "fi"]
        bad = [c for c in cols if not same(a[c], b[c])]
        rep["rebuild_vs_saved"][nm] = {"cols": len(cols), "diff_cols": bad}
    kr0, us0 = merge_frames(k0, u0)
    Ts = {"K2026-09-15 마감": pd.Timestamp("2026-09-15 08:00"), "U2026-09-21 마감": pd.Timestamp("2026-09-21 21:30"),
          "K2026-10-13 마감": pd.Timestamp("2026-10-13 08:00")}
    rep["cuts"] = {}
    for j, (lab, T) in enumerate(Ts.items()):
        pxp, nqp, cnt = perturb_prices(px, nq, T, SEED + j)
        k1, u1 = PN.build_kr(pxp, sch), PN.build_us(pxp, sch, nqp)
        r = {"T": str(T), "bars_changed": cnt}
        for nm, a, b in (("kr", k0, k1), ("us", u0, u1)):
            pre = (pd.to_datetime(a["close_kst"]) <= T).to_numpy()
            post = ~pre
            m0c = [c for c in a.columns if c.startswith("m0_") and a[c].dtype.kind in "fi"]
            g = [c for c in m0c if c.endswith("_g") or c.endswith("absg")]
            changed = {c: n_diff(a.loc[pre, c], b.loc[pre, c]) for c in m0c}
            changed = {c: v for c, v in changed.items() if v}
            # 진입 순간 g: 마감 ≤ T이면서 진입 시가가 T 뒤인 창(마감 = T인 창)만 바뀔 수 있다
            fc = [c for c in a.columns if c.startswith("fut_") and a[c].dtype.kind == "f"]
            r[nm] = {"windows_close_le_T": int(pre.sum()), "m0_cols": len(m0c),
                     "m0_changed_cols_pre": {c: v for c, v in changed.items() if c not in g},
                     "entry_instant_changed_pre": {c: v for c, v in changed.items() if c in g},
                     "m0_changed_cols_post_count": int(sum(n_diff(a.loc[post, c], b.loc[post, c]) > 0 for c in m0c)),
                     "fut_changed_windows_pre": int(sum(n_diff(a.loc[pre, c], b.loc[pre, c]) for c in fc) > 0),
                     "fut_cols_changed_pre": int(sum(n_diff(a.loc[pre, c], b.loc[pre, c]) > 0 for c in fc))}
        kr1, us1 = merge_frames(k1, u1)
        cells = {}
        for cid, sp in t2_specs(kr0, us0).items():
            d0, d1 = (kr0, kr1) if sp["df"] == "kr" else (us0, us1)
            base = d0["judge_ok"].to_numpy(bool)
            if sp["mask"] is not None:
                base = base & sp["mask"]
            need = [sp["xcol"], sp["ycol"]] + sp["m0c"] + sp["m0d"] + sp["ctrl"]
            S0 = base & d0[need].notna().all(axis=1).to_numpy()
            S1 = base & d1[need].notna().all(axis=1).to_numpy()
            pre = (pd.to_datetime(d0["close_kst"]) <= T).to_numpy()
            x = d0[sp["xcol"]].to_numpy(float)
            # 미리보기 방식(표본 안 OLS): 같은 표본에서 m0 열만 바뀐 효과
            S = S0 & S1
            a = TF.m0_hat(d0, S, x, sp["m0c"], sp["m0d"])
            b = TF.m0_hat(d1, S, x, sp["m0c"], sp["m0d"])
            xa, xb = np.full(len(d0), np.nan), np.full(len(d0), np.nan)
            xa[S], xb[S] = a["xperp"], b["xperp"]
            ha, hb = np.full(len(d0), np.nan), np.full(len(d0), np.nan)
            ha[S], hb[S] = a["xhat"], b["xhat"]
            okp = pre & S
            # 고친 방식(앞 자료만)
            pa = m0_past(d0, x, sp["m0c"], sp["m0d"], base)
            pb = m0_past(d1, x, sp["m0c"], sp["m0d"], base)
            cells[cid] = {"pre_T_windows_in_sample": int(okp.sum()),
                          "insample_xperp_changed_pre": n_diff(xa[okp], xb[okp]),
                          "insample_xperp_max_abs_change_pre": float(np.nanmax(np.abs(xa[okp] - xb[okp]))) if okp.any() else None,
                          "insample_xperp_sd": float(np.nanstd(xa[okp])) if okp.any() else None,
                          "insample_xhat_changed_pre": n_diff(ha[okp], hb[okp]),
                          "past_xperp_finite_pre": int(np.isfinite(pa["xperp"][pre]).sum()),
                          "past_xperp_changed_pre": n_diff(pa["xperp"][pre], pb["xperp"][pre]),
                          "past_xhat_changed_pre": n_diff(pa["xhat"][pre], pb["xhat"][pre])}
        r["m0_residual"] = cells
        rep["cuts"][lab] = r
    return rep


# ================================================================ E 진입·결과 기간
def entry_test() -> dict:
    cal = pd.read_parquet(RES / "loader_calendar.parquet")
    krx = sorted(d.date() for d in cal.loc[cal["krx"], "date"])
    nyse = sorted(d.date() for d in cal.loc[cal["nyse"], "date"])
    px, _ = PN.load_prices()
    nq = pd.read_parquet(RES / "prices_nq_1h.parquet")
    pk, pu = pd.read_parquet(RES / "panel_kr.parquet"), pd.read_parquet(RES / "panel_us.parquet")
    out = {}

    def prev(sess, d):
        c = [s for s in sess if s < d]
        return c[-1] if c else None

    def us_close_kst(d):
        return datetime.combine(d, time(16, 0), ET).astimezone(KST).replace(tzinfo=None)

    def ref_fut(df, D):
        o = {}
        idx = list(df.index)
        if D not in df.index:
            return None
        i = idx.index(D)
        op, cl, hi, lo = (df[c].to_numpy(float) for c in ("open", "close", "high", "low"))
        for h in PN.HORIZONS:
            e = i + h - 1
            o[f"R{h}"] = cl[e] / op[i] - 1 if e < len(idx) else np.nan
            o[f"end{h}"] = idx[e] if e < len(idx) else None
        for h in PN.MDD_H:
            e = i + h - 1
            if e >= len(idx) or not (np.isfinite(op[i]) and np.isfinite(cl[e])):
                o[f"MDD{h}"] = np.nan
                continue
            path = [op[i]] + [cl[j] for j in range(i, e + 1) if np.isfinite(cl[j])]
            pk_, worst = -np.inf, 0.0
            for v in path:
                pk_ = max(pk_, v)
                worst = min(worst, v / pk_ - 1)
            o[f"MDD{h}"] = worst
        o["HL"] = math.log(hi[i] / lo[i]) if np.isfinite(hi[i]) and np.isfinite(lo[i]) else np.nan
        return o

    # 국내 창
    bad = []
    fut_bad = {}
    for r in pk.itertuples(index=False):
        D, P, ck = pd.Timestamp(r.D).date(), pd.Timestamp(r.P).date(), pd.Timestamp(r.close_kst)
        if not (ck == pd.Timestamp(datetime.combine(D, CLOSE_KR)) and D in krx and P == prev(krx, D)):
            bad.append({"win": r.win_id, "why": "마감·진입·P"})
        u = pd.Timestamp(r.m0_us_session).date() if pd.notna(r.m0_us_session) else None
        u_ref = max(s for s in nyse if us_close_kst(s) < ck.to_pydatetime())
        if u != u_ref:
            bad.append({"win": r.win_id, "why": f"직전 미국 세션 {u} ≠ {u_ref}"})
        for k, t in PN.KR_ASSETS.items():
            ref = ref_fut(px[t], D)
            for h in PN.HORIZONS:
                v = getattr(r, f"fut_{k}_R{h}")
                rv = np.nan if ref is None else ref[f"R{h}"]
                if not same([v], [rv]):
                    fut_bad.setdefault(f"fut_{k}_R{h}", []).append(r.win_id)
            for h in PN.MDD_H:
                v = getattr(r, f"fut_{k}_MDD{h}")
                rv = np.nan if ref is None else ref[f"MDD{h}"]
                if not same([v], [rv]):
                    fut_bad.setdefault(f"fut_{k}_MDD{h}", []).append(r.win_id)
            v = getattr(r, f"fut_{k}_HL")
            if not same([v], [np.nan if ref is None else ref["HL"]]):
                fut_bad.setdefault(f"fut_{k}_HL", []).append(r.win_id)
        ref = ref_fut(px["^KS11"], D)
        for h in PN.HORIZONS:
            e = getattr(r, f"fut_end_krx{h}")
            re_ = ref[f"end{h}"] if ref else None
            if not ((pd.isna(e) and re_ is None) or (pd.notna(e) and pd.Timestamp(e).date() == re_)):
                fut_bad.setdefault(f"fut_end_krx{h}", []).append(r.win_id)
    out["kr"] = {"windows": int(len(pk)), "window_entry_bad": bad[:10], "n_window_entry_bad": len(bad),
                 "fut_mismatch": {k: v[:5] for k, v in fut_bad.items()},
                 "n_fut_mismatch": int(sum(len(v) for v in fut_bad.values())),
                 "entry_after_close_min_minutes": 60}
    # 미국 창
    bad, fut_bad = [], {}
    nq_end = pd.to_datetime(nq["bar_end_et"])
    for r in pu.itertuples(index=False):
        D, P, ck = pd.Timestamp(r.D).date(), pd.Timestamp(r.P).date(), pd.Timestamp(r.close_kst)
        ck_ref = datetime.combine(D, CLOSE_US_ET, ET).astimezone(KST).replace(tzinfo=None)
        op_kst = datetime.combine(D, time(9, 30), ET).astimezone(KST).replace(tzinfo=None)
        if not (ck.to_pydatetime() == ck_ref and D in nyse and P == prev(nyse, D) and op_kst > ck.to_pydatetime()):
            bad.append({"win": r.win_id, "why": "마감·진입·P"})
        k_ = pd.Timestamp(r.m0_kr_session).date() if pd.notna(r.m0_kr_session) else None
        k_ref = max(s for s in krx if datetime.combine(s, time(15, 30)) < ck.to_pydatetime())
        if k_ != k_ref:
            bad.append({"win": r.win_id, "why": f"직전 국내 세션 {k_} ≠ {k_ref}"})
        if r.m0_nq_to_bar_end_et is not None:
            te = pd.Timestamp(r.m0_nq_to_bar_end_et)
            lim = pd.Timestamp(datetime.combine(D, time(8, 29)))
            ref_te = nq_end[nq_end <= pd.Timestamp(datetime.combine(D, time(8, 30)))].max()
            if not (te <= lim and te == ref_te):
                bad.append({"win": r.win_id, "why": f"NQ 끝 바 {te}"})
        for k, t in PN.US_ASSETS.items():
            ref = ref_fut(px[t], D)
            for h in PN.HORIZONS:
                v = getattr(r, f"fut_{k}_R{h}")
                if not same([v], [np.nan if ref is None else ref[f"R{h}"]]):
                    fut_bad.setdefault(f"fut_{k}_R{h}", []).append(r.win_id)
            for h in PN.MDD_H:
                v = getattr(r, f"fut_{k}_MDD{h}")
                if not same([v], [np.nan if ref is None else ref[f"MDD{h}"]]):
                    fut_bad.setdefault(f"fut_{k}_MDD{h}", []).append(r.win_id)
    out["us"] = {"windows": int(len(pu)), "window_entry_bad": bad[:10], "n_window_entry_bad": len(bad),
                 "fut_mismatch": {k: v[:5] for k, v in fut_bad.items()},
                 "n_fut_mismatch": int(sum(len(v) for v in fut_bad.values()))}
    # 칸별 결과 열이 동결 대상·기간과 같은가
    sp = t2_specs(*merge_frames(pk, pu))
    out["cell_targets"] = {"H03": "fut_kospi_lnHL", "H08": "fut_kospi_lnHL", "H70": "fut_kospi_lnHL",
                           "H67_lite": "fut_kospi_lnHL", "H66": sp["H66"]["ycol"], "H68": sp["H68"]["ycol"],
                           "H46": sp["H46"]["ycol"], "H31": sp["H31"]["ycol"], "H35(설명판)": "fut_kospi_R10"}
    out["frozen_targets"] = {"H03": "KOSPI D일 HL(HAR-K, ln HL)", "H08": "KOSPI D일 HL", "H70": "KOSPI D일 HL",
                             "H67": "KOSPI D일 HL", "H66": "KOSPI R5", "H68": "KOSPI R10", "H46": "SOXX R10",
                             "H31": "SOXX MDD10", "H35": "KOSPI R10"}
    # 하닉·SOX가 쉬는 날 경로: 결측 봉(06-29 지수)이 경로 중간에 있는 창
    out["bad_bar_paths"] = {"kr_windows_with_0629_in_R10_path":
                            [w for w, d, e in zip(pk["win_id"], pd.to_datetime(pk["D"]).dt.date, pk["fut_end_krx10"])
                             if pd.notna(e) and d <= date(2026, 6, 29) <= pd.Timestamp(e).date()]}
    return out


# ================================================================ L 예측 라벨 표본
def label_tests() -> dict:
    key = pd.read_parquet(PREVIEW / "label_key.parquet")
    out = {}
    per = key.groupby(["win_id", "kind"]).size().unstack(fill_value=0)
    out["per_window"] = {"windows": int(len(per)), "post_counts": sorted(set(per["글"].tolist())),
                         "comment_counts": sorted(set(per["댓글"].tolist()))}
    # 재현: 저장 units로 틀·뽑기를 다시 돌려 같은 단위인가
    f, w, st = LS.frame()
    smp, strata = LS.draw(f, w)
    a = set(zip(key["win_id"], key["unit_idx"].astype(int)))
    b = set(zip(smp["win_kr"], smp["unit_idx"].astype(int)))
    out["reproduce"] = {"saved": len(a), "redraw": len(b), "same": a == b}
    # 층의 표본 비율과 틀 크기(가격과 무관한 크기 차이만 있는가: 정보용)
    strata = strata.merge(w[["win_id", "D"]], on="win_id")
    out["frame_size_per_window"] = {k: {"min": int(g["N_frame"].min()), "median": float(g["N_frame"].median()),
                                        "max": int(g["N_frame"].max())} for k, g in strata.groupby("kind")}
    # 조각 순서와 날짜의 상관(라벨러가 시간 순서로 받지 않았는가)
    order = key["shard"].astype(int) * 1000 + key["shard_pos"].astype(int)
    out["shard_order_vs_D_spearman"] = float(stats.spearmanr(order, pd.to_datetime(key["D"]).map(pd.Timestamp.toordinal))[0])
    # 라벨 파일·조각 필드
    lab_keys, shard_keys = set(), set()
    for p in sorted((PREVIEW / "labels").glob("pred_*.json")):
        for r in json.loads(p.read_text("utf-8")):
            lab_keys |= set(r)
    for p in sorted((PREVIEW / "label_shards").glob("shard_*.json")):
        for r in json.loads(p.read_text("utf-8")):
            shard_keys |= set(r)
    out["label_fields"] = sorted(lab_keys)
    out["shard_fields"] = sorted(shard_keys)
    # 날짜 가림 점검(조각 본문): 월·일·연도 숫자 꼴
    rxd = re.compile(r"(?<!\d)(20\d\d|\d{1,2}\s?월\s?\d{1,2}\s?일|\d{1,2}/\d{1,2}(?!\d))")
    hits = 0
    for p in sorted((PREVIEW / "label_shards").glob("shard_*.json")):
        hits += sum(bool(rxd.search(r["text"])) for r in json.loads(p.read_text("utf-8")))
    out["shard_date_pattern_hits"] = hits
    # 지식 마감: 06-30 단위가 특징에서 빠졌는가
    m, info = PW.load_units()
    out["knowledge_cutoff"] = {"dropped_units": info["dropped_knowledge_period_units"],
                               "min_dt_used": str(m["dt_kst"].min())}
    # 라벨 교란
    out["label_perturb"] = {}
    F0, _ = PW.build_windows(m)
    W0 = PW.build_weeks(m)
    wk = pd.read_parquet(RES / "windows_kr.parquet")
    wk = wk[(wk["variant"] == "main") & wk["complete"]]
    close = dict(zip(wk["win_id"], pd.to_datetime(wk["close_kst"])))
    rng = np.random.default_rng(SEED)
    for lab, T in (("K2026-08-14 마감", pd.Timestamp("2026-08-14 08:00")),
                   ("K2026-09-15 마감", pd.Timestamp("2026-09-15 08:00")),
                   ("K2026-10-13 마감", pd.Timestamp("2026-10-13 08:00"))):
        m2 = m.copy()
        fut = m2["win_id"].map(close) > T
        m2.loc[fut, "pred"] = rng.integers(0, 6, int(fut.sum()))
        m2.loc[fut, "a"] = [f"X{rng.integers(0, 50):02d}" for _ in range(int(fut.sum()))]
        drop = fut & (rng.random(len(m2)) < 0.3)
        m2 = m2[~drop]
        F1, _ = PW.build_windows(m2)
        W1 = PW.build_weeks(m2)
        pre0 = F0[pd.to_datetime(F0["close_kst"]) <= T].set_index("win_id")
        pre1 = F1[pd.to_datetime(F1["close_kst"]) <= T].set_index("win_id").loc[pre0.index]
        num = [c for c in pre0.columns if pre0[c].dtype.kind in "fiub"]
        dc = [c for c in num if not same(pre0[c].astype(float), pre1[c].astype(float))]
        wk0 = W0[pd.to_datetime(W0["week_close"]) <= T].set_index("week_close")
        wk1 = W1[pd.to_datetime(W1["week_close"]) <= T].set_index("week_close").loc[wk0.index]
        wnum = [c for c in wk0.columns if wk0[c].dtype.kind in "fiub"]
        wdc = [c for c in wnum if not same(wk0[c].astype(float), wk1[c].astype(float))]
        post0 = F0[pd.to_datetime(F0["close_kst"]) > T].set_index("win_id")
        post1 = F1[pd.to_datetime(F1["close_kst"]) > T].set_index("win_id")
        jj = post0.index.intersection(post1.index)
        out["label_perturb"][lab] = {"units_changed": int(fut.sum()), "units_dropped": int(drop.sum()),
                                     "windows_le_T": int(len(pre0)), "cols": len(num), "diff_cols": dc,
                                     "weeks_le_T": int(len(wk0)), "week_diff_cols": wdc,
                                     "sanity_post_T_P_bar_changed": n_diff(post0.loc[jj, "P_bar"], post1.loc[jj, "P_bar"])}
    return out


# ================================================================ X 앞 자료만 쓰는 M0
def m0_past(df: pd.DataFrame, x: np.ndarray, cont: list[str], dums: list[str], ok: np.ndarray,
            mode: str = "monthly", min_fit: int = MIN_FIT) -> dict:
    """동결 §2.3 모양: X를 M0에 OLS 적합하되, 창 i의 계수는 i보다 앞 창만으로 정한다.
    monthly = 그 달 첫 거래일 전 창(D가 전달까지)으로 적합해 그 달 내내 쓴다(§2.2 '재적합은 적용하는 달의 첫 거래일 전 자료로만').
    X⊥ = (x − x̂) ÷ 적합 잔차 SD(§2.0 'X⊥ = 잔차 ÷ 잔차 SD'). 적합 창 수 < min_fit이면 결측."""
    n = len(df)
    M = df[cont + dums].to_numpy(float)
    avail = ok & np.isfinite(x) & np.isfinite(M).all(axis=1)
    Dd = list(df["Dd"])
    xp, xh, nfit = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, int)
    cache = {}
    for i in range(n):
        if not avail[i]:
            continue
        if mode == "monthly":
            key = (Dd[i].year, Dd[i].month)
            fit = avail & np.array([d < date(key[0], key[1], 1) for d in Dd])
        else:
            key = i
            fit = avail & (np.arange(n) < i)
        if key not in cache:
            if fit.sum() < min_fit:
                cache[key] = None
            else:
                d = df[fit]
                X, names, _ = TF.build_design(d, [], cont, dums)
                beta, e = TF.ols(x[fit], X)
                k = X.shape[1]
                sd = math.sqrt(float(e @ e) / max(int(fit.sum()) - k, 1))
                cache[key] = (names, beta, sd, int(fit.sum()))
        c = cache[key]
        if c is None:
            continue
        names, beta, sd, nf = c
        row = np.concatenate([[1.0], df.iloc[i][names].to_numpy(float)])
        h = float(row @ beta)
        xh[i] = h
        xp[i] = (x[i] - h) / sd if sd > 0 else np.nan
        nfit[i] = nf
    return {"xperp": xp, "xhat": xh, "nfit": nfit}


def ic_past(df, S, xp, y, kind, ctrl):
    yy = np.log1p(y[S]) if kind == "R" else y[S]
    C = df.loc[S, ctrl].to_numpy(float) if ctrl else None
    ic, ex, ey = TF.spearman_partial(xp[S], yy, C)
    return ic, ex, ey, yy, C


def t2_past(df: pd.DataFrame, cid: str, sp: dict, eps: np.ndarray, mode: str) -> dict:
    base = df["judge_ok"].to_numpy(bool)
    if sp["mask"] is not None:
        base = base & sp["mask"]
    x = df[sp["xcol"]].to_numpy(float)
    F = base & np.isfinite(x)
    r = m0_past(df, x, sp["m0c"], sp["m0d"], base, mode)
    y = df[sp["ycol"]].to_numpy(float)
    ctrlok = df[sp["ctrl"]].notna().all(axis=1).to_numpy()
    S = base & np.isfinite(r["xperp"]) & np.isfinite(y) & ctrlok
    out = {"id": cid, "mode": mode, "min_fit": MIN_FIT, "n_obs": int(S.sum()),
           "n_feature_windows": int(F.sum()), "n_xperp_finite": int(np.isfinite(r["xperp"]).sum())}
    if S.sum() < 8:
        out["note"] = "X⊥ 유효 창이 8개 미만이라 계산하지 않음"
        return out
    out["first_D"], out["last_D"] = str(df.loc[S, "Dd"].min()), str(df.loc[S, "Dd"].max())
    ic, ex, ey, yy, C = ic_past(df, S, r["xperp"], y, sp["kind"], sp["ctrl"])
    pos = df.loc[S, "pos"].to_numpy()
    xs = r["xperp"][S]
    Xr = np.column_stack([np.ones(len(yy)), TF.zs(TF.rank(xs))] + ([C] if C is not None else []))
    if sp["kind"] == "R":
        t = TF.tstat(yy, Xr, 1, "hodrick", pos, eps=eps, h=sp["h"])
    else:
        t = TF.tstat(yy, Xr, 1, "hac", pos, L=sp["h"])
    sign = sp["sign"]
    p_an = float(stats.t.sf(sign * t["t"], t["df"]))
    ks = TF.shifts(int(F.sum()))
    null = []
    for k in ks:
        xr = TF.rolled(x, F, k)
        rr = m0_past(df, xr, sp["m0c"], sp["m0d"], base, mode)
        Sk = base & np.isfinite(rr["xperp"]) & np.isfinite(y) & ctrlok
        null.append(ic_past(df, Sk, rr["xperp"], y, sp["kind"], sp["ctrl"])[0] if Sk.sum() >= 8 else np.nan)
    null = np.array([v for v in null if np.isfinite(v)])
    p_sh = TF.shift_p(ic, null, sign)
    p_final = max(p_an, p_sh) if p_sh is not None else p_an
    rho = TF.ac1_pos(xs, pos)
    ne = TF.n_eff(int(S.sum()), rho, sp["h"])
    rob = {}
    for key, m in (("crash4", S & ~TF.overlap_mask(df, TF.CRASH_DAYS)), ("opened8", S & ~TF.overlap_mask(df, TF.OPENED_DAYS))):
        v = ic_past(df, m, r["xperp"], y, sp["kind"], sp["ctrl"])[0]
        rob[key] = {"n": int(m.sum()), "ic": v, "ratio": v / ic if ic else float("nan")}
    contrib = sign * ex * ey
    top2 = np.argsort(contrib)[-2:]
    sidx = np.where(S)[0]
    m = S.copy()
    m[sidx[top2]] = False
    v = ic_past(df, m, r["xperp"], y, sp["kind"], sp["ctrl"])[0]
    rob["top2"] = {"n": int(m.sum()), "ic": v, "ratio": v / ic if ic else float("nan"),
                   "dropped_win": [str(df.loc[i, "win_id"]) for i in sidx[top2]]}
    robust_ok = bool(all(q["ratio"] >= 0.5 for q in rob.values()))
    ic_x, _, _ = TF.spearman_partial(x[S], yy, C)
    ic_ph, _, _ = TF.spearman_partial(r["xhat"][S], yy, C)
    flag = bool(np.sign(ic_ph) == np.sign(ic_x) and abs(ic_ph) >= 0.7 * abs(ic_x))
    out.update({"ic_main_xperp_partial": ic, "t": t["t"], "df": t["df"], "p_analytic_one": p_an, "shift_K": len(null),
                "p_shift_one": p_sh, "p_final_one": p_final, "rho1_xperp": rho, "n_eff": ne,
                "mde_ic": 2.49 / math.sqrt(ne), "robust": rob, "robust_ok": robust_ok,
                "ic_x_ctrl_partial_same_sample": ic_x,
                "placebo": {"ic": ic_ph, "ratio_vs_x_ctrl": ic_ph / ic_x if ic_x else float("nan"),
                            "flag_70pct_same_sign": flag},
                "nfit_range": [int(r["nfit"][S].min()), int(r["nfit"][S].max())]})
    return out


def t1_placebo_past(kr: pd.DataFrame, cid: str, xcol: str, extra: list[str], mode: str) -> dict:
    base = kr["judge_ok"].to_numpy(bool)
    need = ["fut_kospi_lnHL", xcol] + TF.HAR_CONT + extra + TF.M0_KR_CONT + TF.M0_KR_DUM
    S0 = base & kr[need].notna().all(axis=1).to_numpy()
    x = kr[xcol].to_numpy(float)
    r = m0_past(kr, x, TF.M0_KR_CONT, TF.M0_KR_DUM, base, mode)
    S = S0 & np.isfinite(r["xhat"])
    out = {"id": cid, "mode": mode, "n_main_sample": int(S0.sum()), "n_with_past_xhat": int(S.sum())}
    if S.sum() < 15:
        out["note"] = "앞 자료 X̂가 있는 창이 15개 미만이라 계산하지 않음"
        return out
    fit = TF.har_fit(kr, S, x, extra)
    fph = TF.har_fit(kr, S, r["xhat"], extra)
    flag = bool(np.sign(fph["pr"]) == np.sign(fit["pr"]) and abs(fph["pr"]) >= 0.7 * abs(fit["pr"]))
    out.update({"partial_r_x_same_sample": fit["pr"], "t_x_same_sample": fit["t"]["t"],
                "placebo_partial_r": fph["pr"], "placebo_t": fph["t"]["t"],
                "ratio": fph["pr"] / fit["pr"] if fit["pr"] else float("nan"), "flag_70pct_same_sign": flag,
                "first_D": str(kr.loc[S, "Dd"].min()), "last_D": str(kr.loc[S, "Dd"].max())})
    return out


def fixed_m0() -> dict:
    kr, us = merge_frames(pd.read_parquet(RES / "panel_kr.parquet"), pd.read_parquet(RES / "panel_us.parquet"))
    P = TF.load()["px"]
    eps = {("us", "SOXX"): TF.cc_by_window(us, P["SOXX"]), ("kr", "^KS11"): TF.cc_by_window(kr, P["^KS11"])}
    tf = json.loads((RES / "test_free.json").read_text("utf-8"))["cells"]
    tp = json.loads((RES / "test_pred.json").read_text("utf-8"))["cells"]
    orig = {**{k: tf[k] for k in ("H03", "H08", "H46", "H31")}, **{k: tp[k] for k in ("H66", "H68", "H70")},
            "H67_lite": tp["H67"]["explanatory_lite"]}
    out = {"t2": {}, "t1_placebo": {}, "orig": {}}
    for mode in ("monthly", "window"):
        out["t2"][mode] = {}
        for cid, sp in t2_specs(kr, us).items():
            df = kr if sp["df"] == "kr" else us
            out["t2"][mode][cid] = t2_past(df, cid, sp, eps[sp["eps"]], mode)
        out["t1_placebo"][mode] = {cid: t1_placebo_past(kr, cid, xc, ex, mode) for cid, (xc, ex) in T1_SPECS.items()}
    for cid in ("H46", "H31", "H66", "H68"):
        o = orig[cid]
        out["orig"][cid] = {"n_obs": o["n_obs"], "ic": o["ic_main_xperp_partial"], "t": o["t"], "p_final_one": o["p_final_one"],
                            "placebo_ic": o["placebo"]["ic"], "placebo_flag": o["placebo"]["flag_70pct_same_sign"],
                            "verdict": o.get("verdict_shape") or o.get("shape_info_only")}
    for cid in ("H03", "H08", "H70", "H67_lite"):
        o = orig[cid]
        out["orig"][cid] = {"n_obs": o["n_obs"], "partial_r": o["partial_r"], "placebo_partial_r": o["placebo"]["partial_r"],
                            "placebo_flag": o["placebo"]["flag_70pct_same_sign"],
                            "verdict": o.get("verdict_shape") or o.get("shape_info_only"),
                            "p_final_one": o["p_final_one"]}
    # 모양 다시 매기기(주판 monthly). T2 BH = H46·H31·H66(test_pred 합친 판과 같은 칸), T1 Holm은 p가 그대로다
    m = out["t2"]["monthly"]
    ps = {k: m[k]["p_final_one"] for k in ("H46", "H31", "H66") if "p_final_one" in m[k]}
    q = TF.bh(ps) if ps else {}
    shapes = {}
    for k in ("H46", "H31", "H66", "H68"):
        c = m[k]
        if "p_final_one" not in c:
            shapes[k] = "계산 안 함(앞 자료 X⊥ 창 부족)"
            continue
        padj = q.get(k, c["p_final_one"])
        shapes[k] = TF.verdict_shape(c["ic_main_xperp_partial"] * t2_specs(kr, us)[k]["sign"],
                                     1 / math.sqrt(c["n_eff"]), padj, 0.10, c["robust_ok"],
                                     c["placebo"]["flag_70pct_same_sign"], TF.EXPECTED["IC"] / 2)
    t1h = {"H03": orig["H03"]["p_final_one"], "H08": orig["H08"]["p_final_one"], "H70": orig["H70"]["p_final_one"]}
    holm = TF.holm(t1h)
    for k in ("H03", "H08", "H70"):
        o = tf.get(k) or tp.get(k)
        pl = out["t1_placebo"]["monthly"][k]
        flag = pl.get("flag_70pct_same_sign", False)
        shapes[k] = TF.verdict_shape(o["partial_r"], o["se_partial_r"], holm[k], 0.05, o["robust_ok"], flag,
                                     TF.EXPECTED["T1"] / 2)
    out["shapes_monthly"] = shapes
    out["bh_monthly"] = q
    return out


# ================================================================ D 설명용 시차표(겹치지 않는 판)
def lag_clean() -> dict:
    md = pd.read_csv(RES / "loader_missing_by_day.csv", encoding="utf-8-sig")
    md["date"] = pd.to_datetime(md["date"]).dt.date
    posts = md.set_index("date")["d_no"].astype(float)
    P = TF.load()["px"]
    out = {}
    for tk, lab in (("^KS11", "kospi"), ("SOXX", "soxx")):
        px = P[tk]
        c, o = px["close"].astype(float), px["open"].astype(float)
        sess = [d for d in px.index if min(posts.index) <= d <= max(posts.index)]
        rows = []
        for k in range(1, 6):
            xs, ys_cc, ys_clean = [], [], []
            for i, d in enumerate(sess):
                j = i + k
                if j >= len(sess):
                    break
                pv = posts.get(d, np.nan)
                d1, dk = sess[j], sess[j]
                prev_c = c.get(sess[j - 1], np.nan)
                cc = c.get(dk, np.nan) / prev_c - 1
                # 깨끗한 판: 수익 구간이 그날 24:00 뒤에 시작
                if lab == "kospi" and k == 1:
                    clean = c.get(dk, np.nan) / o.get(dk, np.nan) - 1        # 다음 세션 시가(09:00) → 종가
                else:
                    clean = cc                                                  # 시작 = 앞 세션 종가 > 그날 24:00
                xs.append(pv)
                ys_cc.append(cc)
                ys_clean.append(clean)
            xs, ys_cc, ys_clean = map(np.array, (xs, ys_cc, ys_clean))
            for nm, yy in (("cc", ys_cc), ("clean", ys_clean)):
                for kind, v in (("absr", np.abs(yy)), ("r", yy)):
                    ok = np.isfinite(xs) & np.isfinite(v)
                    rows.append({"lag": k, "ret": nm, "kind": kind, "n": int(ok.sum()),
                                 "spearman": float(stats.spearmanr(xs[ok], v[ok])[0])})
        out[lab] = rows
    out["rule"] = ("KST 달력일 글 수(t, 0~24시)와 k거래일 뒤 수익. 종가 대 종가 수익의 구간이 그날 24:00 전에 시작하면 겹친다: "
                   "KOSPI +1일(구간 시작 = t일 15:30), SOXX 0일(구간 시작 = t일 05:00 KST). 깨끗한 판: KOSPI +1일은 "
                   "다음 세션 시가(09:00)→종가, +2일부터와 SOXX +1일부터는 종가 대 종가 그대로(시작이 24:00 뒤)")
    tf = json.loads((RES / "test_free.json").read_text("utf-8"))["descriptive"]["lead_lag"]
    out["orig_kospi_absr_lag1"] = tf["kospi_absr"][11]["spearman"]
    out["orig_kospi_r_lag1"] = tf["kospi_r"][11]["spearman"]
    out["orig_soxx_absr_lag0"] = tf["soxx_absr"][10]["spearman"]
    out["orig_soxx_r_lag0"] = tf["soxx_r"][10]["spearman"]
    ka = {r["lag"]: r["spearman"] for r in out["kospi"] if r["ret"] == "clean" and r["kind"] == "absr"}
    kr_ = {r["lag"]: r["spearman"] for r in out["kospi"] if r["ret"] == "clean" and r["kind"] == "r"}
    out["kospi_absr_lead_mean_1_5_clean"] = float(np.mean([ka[k] for k in range(1, 6)]))
    out["kospi_r_lead_mean_1_5_clean"] = float(np.mean([kr_[k] for k in range(1, 6)]))
    out["orig_kospi_absr_lead_mean_1_5"] = float(np.mean([tf["kospi_absr"][10 + k]["spearman"] for k in range(1, 6)]))
    out["orig_kospi_r_lead_mean_1_5"] = float(np.mean([tf["kospi_r"][10 + k]["spearman"] for k in range(1, 6)]))
    # 상위 10일 표의 '다음 거래일' 수익: 다음 세션 시가 → 종가 판
    kc, ko = P["^KS11"]["close"].astype(float), P["^KS11"]["open"].astype(float)
    ks = sorted(kc.index)
    top = []
    for r in json.loads((RES / "test_free.json").read_text("utf-8"))["descriptive"]["top10"]:
        d = date.fromisoformat(r["date"])
        nx = next((s for s in ks if s > d), None)
        top.append({"date": r["date"], "next": str(nx), "next_cc": r["kospi_r_next_session"],
                    "next_open_to_close": float(kc[nx] / ko[nx] - 1) if nx and np.isfinite(ko[nx]) else None})
    out["top10_next_session"] = top
    return out


# ================================================================ 실행
def clean(o):
    return TF.clean(o)


def main() -> int:
    t0 = datetime.now().replace(microsecond=0)
    kr, us = merge_frames(pd.read_parquet(RES / "panel_kr.parquet"), pd.read_parquet(RES / "panel_us.parquet"))
    rep = {"made": t0.isoformat()}
    rep["S_static"] = static_scan()
    print("S done", flush=True)
    rep["F_fut_perturb"] = fut_perturb_test(kr, us)
    print("F done", flush=True)
    rep["E_entry"] = entry_test()
    print("E done", flush=True)
    rep["L_label"] = label_tests()
    print("L done", flush=True)
    rep["P_price_perturb"] = price_perturb_test()
    print("P done", flush=True)
    rep["X_fixed_m0"] = fixed_m0()
    print("X done", flush=True)
    rep["D_lag_clean"] = lag_clean()
    rep["inputs_sha256"] = {p: sha(RES / p) for p in ("panel_kr.parquet", "panel_us.parquet", "prices.parquet",
                                                     "pred_windows.parquet", "test_free.json", "test_pred.json")}
    rep["code_sha256"] = sha(Path(__file__))
    (RES / "audit_timing.json").write_text(json.dumps(clean(rep), ensure_ascii=False, indent=1, default=str), "utf-8")
    print("wrote audit_timing.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
