"""최종 판정 준비: 가격 전 점검표(HYPOTHESES.md §5.1) 1~8·10·11을 전체 스냅샷(2025-08-20~2026-10-21)으로 다시 계산한다.

9번(가격만 쓰는 셈)은 하지 않는다(판정 때). 앞날 가격도 지난 가격도 읽거나 만들지 않는다.
- feat_text.parquet의 지난 가격 표지 열(px_*, ev_*)은 읽지 않는다(열 선택에서 뺀다).
- #11 외부 자료는 네트워크에 묻지 않고, 저장소에 있는 파일과 중간 검증(10/16) 결과만 본다. 가격 파일은 parquet 꼬리표(행 수)와
  시각 열만 읽는다.

동결 코드 재사용(고치지 않고 import): interim/code/prechecks.py(n_eff·AC1·묶음·상관·SPECS·구조 결측 규칙·M9/M10),
loader.py(창 달력·소금 지문), feat_text.py(rz8·z_prev·pct_prev), preview/code/pred_windows.py(예측 층 창 특징).

입력(읽기만): final/results/{feat_*,windows_*,units,loader_*}.parquet|csv|json, final/snapshot_manifest.json,
              final/label_key.parquet, final/labels/shard_*.json(다 찬 조각만), preview/label_key.parquet, preview/labels/pred_*.json,
              interim/results/{lexicon_v02_precision.json, prechecks_external.json, prechecks_summary.json},
              data/derived/desk/ant/{posts/*.jsonl(스냅샷 바이트만), state.json·백업(소금 지문만), collect.log}
산출: final/results/prechecks_full.json, prechecks_full.md, prechecks_full_features.csv, prechecks_full_pred_windows.parquet
실행: py prechecks_full.py
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
FINAL = HERE.parent
ANT = FINAL.parent
RES = FINAL / "results"
INTERIM = ANT / "interim"
PREVIEW = ANT / "preview"
ROOT = ANT.parents[3]
ANT_DIR = ROOT / "data" / "derived" / "desk" / "ant"
sys.path.insert(0, str(INTERIM / "code"))
import loader as LD  # noqa: E402
import prechecks as P  # noqa: E402  (중간 검증 점검 코드: 함수만 쓴다)
import feat_text as FX  # noqa: E402
sys.path.insert(1, str(PREVIEW / "code"))
import pred_windows as PW  # noqa: E402  (예측 층 창 특징: group_stats·psd_norm 그대로)

SNAP = json.loads((FINAL / "snapshot_manifest.json").read_text("utf-8"))
PRE = (date(2025, 8, 21), date(2025, 12, 31))
FRONT, BACK = P.FRONT, P.BACK                      # 2026-01-02~05-29 / 06-01~10-23
LLM = (date(2026, 7, 1), date(2026, 10, 23))       # 예측 층 주 판정 구간(§2.2·부록 3)
JAN2 = date(2026, 1, 2)
PROJ_START = datetime(2025, 1, 1)                  # 수집 완료 가정(2025-01-01부터)
M0_MIN = 120                                       # §2.3 확장 OLS 최소 창
MISS_MAX, MIN_BLOCKS = P.MISS_MAX, P.MIN_BLOCKS
REALTIME_FROM = date(2026, 10, 13)
DROP_PREFIX = ("px_", "ev_")                       # feat_text의 지난 가격 표지 열: 읽지 않는다
PER_NAME = {"front": "앞", "back": "뒤", "llm": "7/1~10/23"}
FROZEN_SHA = {                                      # docs/desk_system.md §12.5에 남긴 값
    "HYPOTHESES.md": "b23d1eacc95d0ae4ce1361c2449d9efd0f8feecc9792c4515ee2aff80ac308e5",
    "HYPOTHESES_addendum1.md": "430dd4ffe856420fa9779dff91dc6e3cb3d3b7548551e02ed4dfab2596044eb4",
    "HYPOTHESES_addendum2.md": "8530bf79c469d0593c5bdc7420fa23aa78c169d8ee79d73c263dfdfed02bfd7b",
    "HYPOTHESES_addendum3.md": "d66733053f1a6342603b6c9642ce3dd5ddd851336d44663dc6f741094fa18320",
    "HYPOTHESES_addendum4.md": "2960391a00b98fb8c776476352595878dde2fa4a0949b4b3ce73676678d5842b",
    "HYPOTHESES_addendum5.md": "079bd3d09a7296b66648d166cc3d93c6fe9476edef2e96425c1eb0c44ba0c9d8",
    "HYPOTHESES_addendum6.md": "c64c53556789cf09513cb4ffc77c84926f055a8b532bd2af9d7a77fc16a327c5",
    "HYPOTHESES_addendum7.md": "cc9376ccaa320e35a41fc50cfc00f03e29e6f523226410d04ab29d1792806ec4",
    "HYPOTHESES_addendum8.md": "d1723022ce07cce241e3de9a2a15c91d75f5141b9ac43bdc872bde031d9aabad",
    "HYPOTHESES_addendum9.md": "54719cdffb28ea54e9c10cdaf41ab66e96d1b5700c12d2945117ac32fdd66581",
    "HYPOTHESES_addendum10.md": "14533e0eb9aa7aa83119c07166a9f646f4f9415781782974908985ce06404a4e",
    "lexicon_draft.md": "a4e28b08ddc3e6cef8b4f49c746c98845cc0f97bb8f03f534ba181a42795f8e1",
}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def r4(x):
    if x is None:
        return None
    try:
        if isinstance(x, (float, np.floating)) and not np.isfinite(x):
            return None
        return round(float(x), 4)
    except (TypeError, ValueError):
        return x


def in_period(D: pd.Series, lo: date, hi: date) -> pd.Series:
    return (D >= lo) & (D <= hi)


# ---------------------------------------------------------------- 달력·창 격자(가격 없음: 거래일 날짜만)

def calendars() -> dict:
    """loader 달력(yfinance 거래일 날짜, 2024-11-01~2026-10-21) + 10/22·10/23은 exchange_calendars 세션."""
    c = pd.read_parquet(RES / "loader_calendar.parquet")
    c["date"] = pd.to_datetime(c["date"]).dt.date
    kr = sorted(c.loc[c.krx, "date"])
    us = sorted(c.loc[c.nyse, "date"])
    early = set(c.loc[c.nyse_early, "date"])
    last = max(c["date"])
    import exchange_calendars as xc
    for name, lst in (("XKRX", kr), ("XNYS", us)):
        sess = xc.get_calendar(name).sessions_in_range(pd.Timestamp(last + timedelta(days=1)), pd.Timestamp(BACK[1]))
        lst.extend(d.date() for d in sess)
    return {"krx": kr, "nyse": us, "nyse_early": early}


def grids(cal: dict) -> dict:
    out = {}
    for mk in ("kr", "us"):
        w = LD.make_windows(cal, mk)
        w = w[w.variant == "main"].sort_values("close_kst").reset_index(drop=True)
        w["Dd"] = pd.to_datetime(w["D"]).dt.date
        out[mk] = w
    return out


# ---------------------------------------------------------------- 입력

def load_tables() -> dict:
    fa = pd.read_parquet(RES / "feat_activity.parquet")
    cols = [c for c in pq.read_schema(RES / "feat_text.parquet").names if not c.startswith(DROP_PREFIX)]
    ft = pd.read_parquet(RES / "feat_text.parquet", columns=cols)
    fto = pd.read_parquet(RES / "feat_topic.parquet")
    daily = {}
    for mk, tb in (("kr", "KR_D"), ("us", "US_D")):
        a = fa[fa.market == mk].set_index("win_id")
        b = ft[ft.market == mk].set_index("win_id")
        c = fto[fto.table == tb].set_index("win_id")
        df = a.join(b[[x for x in b.columns if x not in a.columns]])
        df = df.join(c[[x for x in c.columns if x not in df.columns]])
        df["D"] = pd.to_datetime(df["D"]).dt.date
        daily[mk] = df.sort_values("D")
    weekly = {}
    for mk, tb in (("kr", "KR_W"), ("us", "US_W")):
        w = fto[fto.table == tb].copy()
        w["D"] = pd.to_datetime(w["D"]).dt.date
        weekly[mk] = w.sort_values("D").set_index("win_id")
    wknd = fto[fto.table == "WKND"].copy()
    wknd["D"] = pd.to_datetime(wknd["D"]).dt.date
    wk = pd.read_parquet(RES / "feat_activity_weekend.parquet")
    wk["D"] = pd.to_datetime(wk["mon"]).dt.date
    h40 = pd.read_parquet(RES / "feat_activity_h40_weekly.parquet")
    stock = pd.read_parquet(RES / "feat_topic_stock.parquet")
    return {"daily": daily, "weekly": weekly, "wknd": wknd.sort_values("D").set_index("win_id"),
            "wk": wk.sort_values("D"), "h40": h40, "stock": stock,
            "dropped_price_cols": [c for c in pq.read_schema(RES / "feat_text.parquet").names if c.startswith(DROP_PREFIX)]}


# ---------------------------------------------------------------- 예측 층 창 특징(부록 4, 미리보기와 같은 함수)

def load_pred_units() -> tuple[pd.DataFrame, dict]:
    cols = ["id", "win_id", "a", "dt_kst", "utype"]
    kp = pd.read_parquet(PREVIEW / "label_key.parquet", columns=cols)
    lp = []
    for f in sorted((PREVIEW / "labels").glob("pred_*.json")):
        lp += json.loads(f.read_text("utf-8"))
    lp = pd.DataFrame(lp)[["id", "pred"]]
    kf = pd.read_parquet(FINAL / "label_key.parquet", columns=cols + ["shard"])
    summ = json.loads((FINAL / "label_sample_summary.json").read_text("utf-8"))
    want = {int(k.split("_")[1].split(".")[0]): v["n"] for k, v in summ["files"].items()}
    lf, shard_info = [], {"complete": [], "missing_or_partial": []}
    for s, n in sorted(want.items()):
        f = FINAL / "labels" / f"shard_{s:03d}.json"
        ids = set(kf.loc[kf.shard == s, "id"])
        try:
            lab = json.loads(f.read_text("utf-8"))
            ok = isinstance(lab, list) and len(lab) == n and {r["id"] for r in lab} == ids \
                and all(r.get("pred") in (0, 1, 2, 3, 4, 5) for r in lab)
        except (OSError, ValueError, KeyError, TypeError):
            ok = False
        if ok:
            lf += [{"id": r["id"], "pred": int(r["pred"])} for r in lab]
            shard_info["complete"].append(s)
        else:
            shard_info["missing_or_partial"].append(s)
    lf = pd.DataFrame(lf, columns=["id", "pred"])
    mp = kp.merge(lp, on="id", how="left", validate="1:1").assign(src="preview")
    mf = kf.drop(columns="shard").merge(lf, on="id", how="inner", validate="1:1").assign(src="final")
    m = pd.concat([mp, mf], ignore_index=True)
    assert m["id"].is_unique
    m = m[m["pred"].notna()].copy()
    m["pred"] = m["pred"].astype(int)
    m["dt_kst"] = pd.to_datetime(m["dt_kst"])
    info = {"preview_units": int(len(mp)), "preview_unlabeled": int(mp["pred"].isna().sum()),
            "final_units_labeled": int(len(mf)), "final_units_total": int(len(kf)),
            "final_shards_complete": len(shard_info["complete"]),
            "final_shards_missing_or_partial": shard_info["missing_or_partial"],
            "label_files_sha256": {**{f"preview/labels/{f.name}": sha(f) for f in sorted((PREVIEW / "labels").glob("pred_*.json"))},
                                   **{f"final/labels/shard_{s:03d}.json": sha(FINAL / "labels" / f"shard_{s:03d}.json")
                                      for s in shard_info["complete"]}}}
    return m, info


def build_pred(m: pd.DataFrame, judge: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """두 판: (a) 예열 허용판 = 7월 전 창 라벨(지식 기간)을 기준(rz8·풀)에만 쓰고, 7월 이후 창에서는 6/30 이전 단위를 뺀다.
    (b) 7월 이후만 판 = 미리보기와 같은 규칙(7/1 이후 창·단위만)."""
    w = pd.read_parquet(RES / "windows_kr.parquet")
    w = w[(w["variant"] == "main") & w["complete"]].sort_values("close_kst").reset_index(drop=True)
    w["Dd"] = pd.to_datetime(w["D"]).dt.date
    win_D = dict(zip(w["win_id"], w["Dd"]))
    m = m.assign(D=m["win_id"].map(win_D))
    d0701 = pd.Timestamp("2026-07-01")
    late_win = m["D"] >= LLM[0]
    out = {}
    meta = {}
    for tag, mm, ww in (("a", m[~(late_win & (m["dt_kst"] < d0701))], w),
                        ("b", m[late_win & (m["dt_kst"] >= d0701)], w[w["Dd"] >= LLM[0]].reset_index(drop=True))):
        idx = pd.Index(ww["win_id"], name="win_id")
        F, pairs = PW.group_stats(mm, "win_id", idx)
        valid = ~ww["outage"].fillna(False).astype(bool).to_numpy()
        holiday = (ww["main_kind"] == "holiday").to_numpy()
        wd = ww["weekday"].to_numpy()
        PW.psd_norm(F, pairs, valid, PW.POOL_MIN_REG, PW.N_MIN_REG, "P_sd")
        PW.psd_norm(F, pairs, valid, PW.POOL_MIN_LITE, PW.N_MIN_LITE, "P_sd_lite")
        for c in ["P_sd", "P_share"]:
            F[f"{c}_rz8"] = FX.rz8(F[c].to_numpy(float), valid, wd, holiday)
        F["P_bar_pct250"] = FX.pct_prev(F["P_bar"].to_numpy(float), valid)
        F["P_bar_stage10"] = FX.stage_of(F["P_bar_pct250"].to_numpy(float), 10)
        F["D"] = ww["Dd"].to_numpy()
        F["weekday"] = wd
        F["judge_ok"] = F.index.map(judge).fillna(False).astype(bool)
        F["labeled"] = F["n_units"] > 0
        out[tag] = F
        meta[tag] = {"windows": int(len(F)), "labeled_windows": int(F["labeled"].sum()),
                     "units": int(len(mm)), "first_labeled": str(F.loc[F.labeled, "D"].min()) if F.labeled.any() else None}
    # 미리보기 재현: (b)판의 미리보기 72창이 preview/results/pred_windows.parquet와 같아야 한다
    pv = pd.read_parquet(PREVIEW / "results" / "pred_windows.parquet").set_index("win_id")
    b = out["b"].reindex(pv.index)
    chk = {}
    for c in ["n_pred", "P_bar", "P_share", "P_strong1", "P_sd_raw", "P_share_rz8"]:
        x, y = b[c].astype(float).to_numpy(), pv[c].astype(float).to_numpy()
        same = np.isclose(x, y, equal_nan=True, atol=1e-12)
        chk[c] = {"windows": int(len(pv)), "equal": int(same.sum())}
    meta["preview_reproduction"] = chk
    for tag in ("a", "b"):
        F = out[tag]
        late = F[(pd.Series(list(F["D"]), index=F.index) >= LLM[0]) & F["judge_ok"]]
        meta[tag]["late_valid"] = {c: int(late[c].notna().sum()) for c in ("P_bar", "P_share_rz8", "P_sd", "P_sd_rz8", "P_strong1", "P_sd_lite")}
        meta[tag]["late_windows"] = int(len(late))
        meta[tag]["P_share_rz8_first"] = str(min(F.loc[F["P_share_rz8"].notna(), "D"])) if F["P_share_rz8"].notna().any() else None
    meta["preview_reproduction_ok"] = all(v["equal"] == v["windows"] for v in chk.values())
    return out["a"], out["b"], meta


# ---------------------------------------------------------------- 특징 사양

def raw_missing(*cols):
    return P.raw_missing(*cols)


# (id, 칸, 표, 열, h, 정규화, 예열 종류, X⊥(M0 120창) 필요, 구조 결측 규칙, 사전 묶음, 메모)
def specs() -> list[tuple]:
    S = []
    tags = {  # interim SPECS 행 → (예열 종류, X⊥ 필요). X⊥: §2.3 '방향 가설의 주 특징은 X⊥'(정의가 따로 정한 H54·H55·사건형 제외)
        ("H03", "A_rz8"): ("rz8", False), ("H03", "A_saved_rz8"): ("rz8", False),
        ("H06", "h06_dsd"): ("pool60", False), ("H10", "h10_dsd_z60"): ("pool60+z60", False),
        ("H09", "h06_dsd"): ("pool60", True), ("H14", "S"): ("M0", True), ("H19", "H19_E"): ("z60", True),
        ("H24", "H24_R"): ("base20+2025계수", False), ("H25", "H25_CR_d9"): ("none", False),
        ("H26", "H26_PH"): ("sum5+z60", False), ("H31", "LL_US_d20"): ("chg20", True),
        ("H35", "NW90_rz8"): ("hist90+rz8", True), ("H46", "h46_dtilt"): ("tilt5+20", True),
        ("H49", "h49_s_e"): ("M0", True), ("H51", "h51_s_nm"): ("M0", True),
        ("H57a", "stage10"): ("pct250", False), ("H57b", "stage10"): ("pct250", False),
        ("H05", "H05_ED"): ("z60", False), ("H04", "Bperp_rz8"): ("rz8+reg20", False), ("H08", "fight"): ("none", False),
        ("H11", "H11_X"): ("pool60+z60", True), ("H15압력", "Pess"): ("z60", True), ("H12", "H12_Gap"): ("att20+z60", True),
        ("H15반전", "Pess"): ("z60", True), ("H16", "Pess_news_event"): ("z60", True), ("H16", "Pess_opinion"): ("z60", True),
        ("H54", "A6_SH"): ("rz8", False), ("H55", "A7_K"): ("att20+rz8", False), ("H21", "H21_FO"): ("z60", True),
        ("H36", "NEW90_rz8"): ("hist90+rz8", True), ("H37", "h37_bull_conv_rate"): ("hist30+5", True),
        ("H50", "h50_m_z20"): ("z20", True), ("H33", "H33_sig"): ("z60|pct250", False), ("H34", "NS_share"): ("mean20", False),
        ("H01", "S"): ("M0(R²)", True),
    }
    for hid, cell, tb, col, h, fam, struct, lex, memo in P.SPECS:
        wk, m0 = tags[(hid, col)]
        S.append((hid, cell, tb, col, h, fam, wk, m0, struct, lex, memo))
    S += [
        ("H66", "T2 대표(예측 층)", "pred", "P_bar", 5, "M0 확장 회귀 120창(X⊥)", "none", True,
         raw_missing("P_bar"), "AI 예측 라벨(2층)", "판정 구간 2026-07-01~10-23"),
        ("H67", "T1(예측 층)", "pred", "P_sd_rz8", 1, "풀 60창(40) + 표 20 + rz8", "pool60+rz8", False,
         P.votes_lt20("n_pred", "P_sd_raw"), "AI 예측 라벨(2층)", "구조 결측 = 예측 쌍 20개 미만"),
        ("H68", "T2(예측 층)", "pred", "P_strong1", 10, "M0 확장 회귀 120창(X⊥)", "none", True,
         raw_missing("P_strong1"), "AI 예측 라벨(2층)", ""),
        ("H70", "T1(예측 층)", "pred", "P_share_rz8", 1, "rz8", "rz8", False,
         raw_missing("P_share"), "AI 예측 라벨(2층)", ""),
    ]
    return S


EXTRA_WARM = [  # 예열 표에만 싣는 재료·보조 열(판정 칸 아님)
    ("H04 재료", "kr", "U_rz8", "rz8"), ("H04 재료", "kr", "N_rz8", "rz8"),
    ("H35 재료", "kr", "NW90", "hist90"), ("H35 비등록", "kr", "NW30_rz8", "hist30+rz8"),
    ("H01·H57", "kr", "S_pct250", "pct250"), ("H57c", "kr", "SG", "pct250+mean20"),
    ("H33", "kr", "MC_A_all_pct250", "pct250"), ("H33", "kr", "MC_A_all_z60", "z60"),
    ("H08 보조", "kr", "fight_z60", "z60"), ("부록1 주 단계", "kr", "week_stage10", "pct52주"),
    ("H43", "kr", "h43_event_buy_SKH", "roll120"), ("H43", "us", "h43_event_buy_NVDA", "roll120"),
    ("H06 잠정", "kr", "h06_dsd_prov40", "pool40"),
    ("예측 단계표", "pred", "P_bar_stage10", "pct250"), ("H67 비등록", "pred", "P_sd_lite", "pool10"),
]


# ---------------------------------------------------------------- 6. 특징 표

def trade_pos(days: list[date]) -> dict:
    return {d: i for i, d in enumerate(days)}


def period_stats(x: pd.Series, D: pd.Series, judge: pd.Series, h: int, lo: date, hi: date,
                 days: list[date], rho: float) -> dict:
    pos = trade_pos(days)
    end_day = max(d for d in days if d <= hi)
    end_pos = pos[end_day]
    hh = max(h, 1)
    inp = in_period(D, lo, hi) & judge
    v = inp & x.notna()
    elig = v & D.map(lambda d: pos.get(d, -10**9) + hh - 1 <= end_pos)
    p_el = sorted(pos[d] for d in D[elig])
    n_days = sum(1 for d in days if lo <= d <= hi)
    n_valid = int(v.sum())
    ne = P.n_eff(n_valid, hh, rho)
    return {"n_judge_windows": int(inp.sum()), "n_valid": n_valid, "n_target_ok": int(elig.sum()), "trading_days": n_days,
            "miss": r4(1 - n_valid / inp.sum()) if inp.sum() else None,
            "neff": r4(ne), "mde": r4(P.MDE_K / math.sqrt(ne)) if ne > 0 else None,
            "blocks_gap_h": P.greedy_blocks(p_el, hh), "blocks_strict": P.greedy_blocks(p_el, hh + 1),
            # 예열이 다 풀린 판(2025-01 수집 완료 가정): 구간 전 거래일이 값을 가진다고 보고 엄한 간격으로 센 묶음 수
            "blocks_strict_full": P.greedy_blocks(range(n_days - hh + 1), hh + 1),
            "neff_full": r4(P.n_eff(n_days, hh, rho))}


def feature_rows(T: dict, pred: pd.DataFrame, cal: dict) -> list[dict]:
    rows = []
    tabs = {"kr": T["daily"]["kr"], "us": T["daily"]["us"], "pred": pred}
    for hid, cell, tb, col, h, fam, wk, m0, struct, lex, memo in specs():
        df = tabs[tb]
        days = cal["nyse"] if tb == "us" else cal["krx"]
        D = pd.Series(list(df["D"]), index=df.index)
        x = df[col].astype(float)
        judge = df["judge_ok"].astype(bool)
        sm = struct(df)
        j26 = judge & in_period(D, FRONT[0], BACK[1])
        jl = judge & in_period(D, *LLM)
        jbase = jl if tb == "pred" else j26
        if sm.isna().all():
            s26 = s_all = None
        else:
            a = sm[jbase].dropna()
            s26 = float(a.astype(float).mean()) if len(a) else None
            b = sm[judge].dropna()
            s_all = float(b.astype(float).mean()) if len(b) else None
        r_all, n_pairs = P.ac1(x.to_numpy())
        r26, n26 = P.ac1(x.where(D >= JAN2).to_numpy())
        rho = r_all if r_all == r_all else r26
        valid = x.notna()
        rec = {"id": hid, "cell": cell, "market": tb, "col": col, "h": h, "family": fam, "warm": wk, "needs_M0": m0,
               "lex": lex, "memo": memo, "n_rows": int(len(df)), "n_valid": int(valid.sum()),
               "first_valid": str(min(D[valid])) if valid.any() else None,
               "last_valid": str(max(D[valid])) if valid.any() else None,
               "miss_judge_all": r4(1 - (valid & judge).sum() / judge.sum()),
               "miss_struct_judgeperiod": r4(s26), "miss_struct_all": r4(s_all),
               "ac1_all": r4(r_all), "ac1_all_pairs": n_pairs, "ac1_2026": r4(r26), "ac1_2026_pairs": n26}
        if tb == "pred":
            rec["front"] = period_stats(x, D, judge, h, *FRONT, days, rho)   # AI 지식 기간: 보조 확인만
            rec["llm"] = period_stats(x, D, judge, h, *LLM, days, rho)
        else:
            rec["pre2025"] = {"miss": r4(1 - (valid & judge & in_period(D, *PRE)).sum() / (judge & in_period(D, *PRE)).sum())}
            rec["front"] = period_stats(x, D, judge, h, *FRONT, days, rho)
            rec["back"] = period_stats(x, D, judge, h, *BACK, days, rho)
        rows.append(rec)
    return rows


# ---------------------------------------------------------------- 예열 표(rz8·z60·pct250·M0 120창)

def m0_first(valid_D: list[date], days: list[date], need: int = M0_MIN) -> str | None:
    """확장 OLS(D−1까지, 최소 need창, 월 1회 재적합): 그 달 첫 거래일 전 값 있는 창이 need개 이상인 첫 달의 첫 거래일."""
    vd = sorted(valid_D)
    months = sorted({(d.year, d.month) for d in days})
    for y, mth in months:
        first = min(d for d in days if (d.year, d.month) == (y, mth))
        if sum(1 for d in vd if d < first) >= need:
            return str(first)
    return None


def warmup_table(T: dict, pred: pd.DataFrame, feats: list[dict], cal: dict, G: dict) -> list[dict]:
    tabs = {"kr": T["daily"]["kr"], "us": T["daily"]["us"], "pred": pred}
    rows = []
    items = [(f["id"], f["market"], f["col"], f["warm"], f["needs_M0"], f["family"]) for f in feats]
    seen = set()
    items += [(a, b, c, d, False, d) for a, b, c, d in EXTRA_WARM]
    for hid, tb, col, wk, m0, fam in items:
        key = (hid, tb, col)
        if key in seen:
            continue
        seen.add(key)
        df = tabs[tb]
        if col not in df.columns:
            continue
        mk = "us" if tb == "us" else "kr"
        days = cal["nyse"] if mk == "us" else cal["krx"]
        g = G[mk]
        gpos = {d: i for i, d in enumerate(g["Dd"])}
        D = pd.Series(list(df["D"]), index=df.index)
        judge = df["judge_ok"].astype(bool)
        x = pd.to_numeric(df[col], errors="coerce").astype(float)
        v = x.notna() & judge
        first = min(D[v]) if v.any() else None
        # 지금 스냅샷의 첫 창(라벨 있는 첫 창) → 예열 창 수(lag)
        base_first = min(D[df["labeled"]]) if tb == "pred" else min(D)
        lag = (gpos[first] - gpos[base_first]) if first is not None else None
        # 2025-01-01 수집 완료 가정: 첫 완전 창 + 같은 예열 창 수
        g0 = g[g["open_kst"] >= PROJ_START]
        proj_first = None
        if lag is not None:
            i0 = int(g0.index[0])
            proj_first = g.loc[min(i0 + lag, len(g) - 1), "Dd"]
        fr = judge & in_period(D, *FRONT)
        warm_front = int((fr & (D < first)).sum()) if first is not None else int(fr.sum())
        rec = {"id": hid, "market": tb, "col": col, "warm": wk, "family": fam, "first_valid_now": str(first) if first else None,
               "lag_windows": lag, "filled_before_2026_01_02": bool(first is not None and first < JAN2),
               "front_windows_missing_by_warmup": warm_front, "front_judge_windows": int(fr.sum()),
               "proj_first_valid_2025_01_start": str(proj_first) if proj_first else None,
               "proj_filled_before_2026_01_02": bool(proj_first is not None and proj_first < JAN2)}
        if m0:
            vd = list(D[v])
            xp = m0_first(vd, days)
            j26m = in_period(D, FRONT[0], BACK[1]) & judge
            share = float(v[j26m].mean()) if j26m.any() else 1.0
            if not share > 0:
                share = 1.0
            # 투영: 투영 첫 값부터 모든 창이 있다고 보고, 2026 실제 값 비율만큼 창이 더 필요하다고 본다
            need = math.ceil(M0_MIN / max(min(share, 1.0), 0.05))
            pd_days = [d for d in g["Dd"] if proj_first is not None and d >= proj_first]
            xp_proj = m0_first(pd_days, days, need=need) if pd_days else None
            rec.update({"m0_first_now": xp, "m0_before_2026_01_02_now": bool(xp is not None and date.fromisoformat(xp) <= JAN2),
                        "m0_front_windows_missing_now": int((fr & (D < date.fromisoformat(xp))).sum()) if xp else int(fr.sum()),
                        "m0_first_proj_2025_01": xp_proj, "valid_share_2026": r4(share)})
            if tb == "pred":
                # 7월 이후 라벨만 쓰는 판: X⊥ 첫 달
                vl = [d for d in vd if d >= LLM[0]]
                rec["m0_first_now_july_only"] = m0_first(vl, days)
        changes = (first is None or first >= JAN2) or wk.startswith("pct") or m0 or "2025계수" in wk or wk == "roll120"
        rec["recompute_after_2025_01"] = bool(changes)
        rows.append(rec)
    return rows


# ---------------------------------------------------------------- 1. 결번율·장애 창·스팸 파도

def check1() -> dict:
    out = {}
    for mk, name in (("kr", "windows_kr.parquet"), ("us", "windows_us.parquet")):
        w = pd.read_parquet(RES / name)
        w["Dd"] = pd.to_datetime(w["D"]).dt.date
        res = {}
        for v in sorted(w.variant.unique()):
            ww = w[(w.variant == v) & w.complete].sort_values("close_kst")
            mr = ww["missing_rate"].astype(float)
            res[v] = {"n": int(len(ww)), "first": ww.win_id.iloc[0], "last": ww.win_id.iloc[-1],
                      "kinds": {str(k): int(c) for k, c in ww["kind"].value_counts().items()},
                      "over96": ww.loc[ww.over96, "win_id"].tolist(),
                      "outage": ww.loc[ww.outage.fillna(False).astype(bool), "win_id"].tolist(),
                      "outage_unknown": ww.loc[ww.outage.isna(), "win_id"].tolist(),
                      "spam_wave": ww.loc[ww.spam_wave.fillna(False).astype(bool), "win_id"].tolist(),
                      "missing_rate": {"median": r4(mr.median()), "p90": r4(mr.quantile(0.9)), "max": r4(mr.max()),
                                       "max_win": ww.loc[mr.idxmax(), "win_id"]}}
        main = w[(w.variant == "main") & w.complete]
        per = {}
        for lab, (lo, hi) in (("2025-08~12", PRE), ("2026 앞", FRONT), ("2026 뒤", BACK), ("실시간 10/13~", (REALTIME_FROM, BACK[1]))):
            s = main[in_period(main.Dd, lo, hi)]["missing_rate"].astype(float)
            per[lab] = {"n": int(len(s)), "median": r4(s.median()), "max": r4(s.max())}
        res["main_by_period"] = per
        out[mk] = res
    mb = pd.read_csv(RES / "loader_missing_by_day.csv")
    mb["date"] = pd.to_datetime(mb["date"]).dt.date
    mb["ym"] = [f"{d.year}-{d.month:02d}" for d in mb.date]
    bym = mb.groupby("ym")["missing_rate"].median()
    top = mb.sort_values("missing_rate", ascending=False).head(8)
    out["by_day"] = {"days": int(len(mb)), "median": r4(mb.missing_rate.median()), "max": r4(mb.missing_rate.max()),
                     "top": [{"date": str(r.date), "rate": r4(r.missing_rate), "d_no": int(r.d_no)} for r in top.itertuples()],
                     "by_month_median": {k: r4(v) for k, v in bym.items()},
                     "realtime_median": r4(mb[mb.date >= REALTIME_FROM].missing_rate.median()),
                     "before_realtime_last30_median": r4(mb[(mb.date < REALTIME_FROM) & (mb.date >= REALTIME_FROM - timedelta(days=30))].missing_rate.median())}
    return out


# ---------------------------------------------------------------- 2~5 원자료 점검

def run_boundaries() -> list[dict]:
    rx = re.compile(r"^(\S+ \S+)\s+번호 ([\d,]+) \((\S+)\)\s+저장 ([\d,]+)")
    prev, out = None, []
    for line in (ANT_DIR / "collect.log").read_text("utf-8").splitlines():
        mm = rx.match(line)
        if not mm:
            continue
        cur = {"t": mm.group(1), "no": int(mm.group(2).replace(",", "")), "d": mm.group(3),
               "saved": int(mm.group(4).replace(",", ""))}
        if prev and cur["saved"] < prev["saved"]:
            out.append({"stop_log": prev["t"], "resume_log": cur["t"], "no_last_before": prev["no"], "no_first_after": cur["no"],
                        "post_date_last_before": prev["d"], "post_date_first_after": cur["d"]})
        prev = cur
    return out


def multi_row_recs(multi_nos: set[int]) -> dict[int, list[dict]]:
    """스냅샷 바이트(manifest)만 읽어 두 줄 이상 받은 글의 줄만 남긴다(loader.read_snapshot과 같은 바이트·SHA 확인)."""
    by_no: dict[int, list[dict]] = defaultdict(list)
    for name, spec in sorted(SNAP["files"].items()):
        p = LD.POSTS_DIR / name
        with open(p, "rb") as fh:
            data = fh.read(spec["bytes"])
        if hashlib.sha256(data).hexdigest() != spec["sha256"]:
            raise ValueError(f"{name}: 스냅샷 SHA-256 불일치")
        for i, line in enumerate(data.split(b"\n")):
            if not line.strip():
                continue
            rec = json.loads(line.decode("utf-8"))
            if rec["no"] in multi_nos:
                by_no[rec["no"]].append({"file": p.stem, "line": i, "rec": rec})
    return by_no


def raw_checks(units: pd.DataFrame, hits_excl: pd.Series) -> dict:
    out: dict = {}
    posts = units[units.utype == "post"]
    multi_nos = set(posts.loc[posts.n_rows.fillna(1) >= 2, "no"].astype(int))
    by_no = multi_row_recs(multi_nos)
    multi = {no: sorted(rs, key=lambda r: (r["file"], r["line"])) for no, rs in by_no.items() if len(rs) > 1}

    # 2. 소금
    salt_now = LD.salt_check()
    bk = ANT_DIR / "state_salt_backup_20261017.json"
    salt_bk = LD.salt_check(bk) if bk.exists() else {"present": False}
    same_a = sum(1 for rs in multi.values() if len({r["rec"]["a"] for r in rs}) == 1)
    c_pairs = c_same = 0
    for rs in multi.values():
        maps = []
        for r in rs[:2]:
            d = defaultdict(set)
            for c in r["rec"]["comments"]:
                d[(c["text"], c["dt"][-8:])].add(c["a"])
            maps.append(d)
        for k, a1 in maps[0].items():
            a2 = maps[1].get(k)
            if a2 and len(a1) == 1 and len(a2) == 1 and k[0]:
                c_pairs += 1
                c_same += int(a1 == a2)
    out["salt"] = {"state_present": salt_now.get("present"),
                   "fingerprint_same_as_snapshot": salt_now.get("fingerprint") == SNAP.get("salt_fingerprint"),
                   "backup_present": salt_bk.get("present"),
                   "backup_same_as_now": salt_bk.get("fingerprint") == salt_now.get("fingerprint"),
                   "multi_row_posts": len(multi), "post_author_same": same_a,
                   "comment_pairs": c_pairs, "comment_author_same": c_same}
    # 날짜별 겹침(로더 표) + 일주일 전 같은 요일 겹침
    sc = pd.read_csv(RES / "loader_salt_continuity.csv")
    sc["date"] = pd.to_datetime(sc["date"]).dt.date
    u = units[(units.dt_conf != "latest") & units.dt_kst.notna() & (units.a.astype(str) != "")]
    sets = u.groupby(u.dt_kst.dt.date)["a"].apply(set)
    ov7, hist = [], []
    for d, s in sets.items():
        s7 = sets.get(d - timedelta(days=7))
        o = len(s & s7) / len(s) if s7 else np.nan
        base = float(np.median(hist[-8:])) if len(hist) >= 3 else np.nan
        ov7.append({"date": d, "ov7": o, "base7": base, "flag7": bool(np.isfinite(base) and o < 0.5 * base)})
        if np.isfinite(o):
            hist.append(o)
    ov7 = pd.DataFrame(ov7)
    flagged = sc[sc.flag_drop.astype(bool)]
    fl = []
    for r in flagged.itertuples():
        o7 = ov7.loc[ov7.date == r.date, "ov7"]
        fl.append({"date": str(r.date), "weekday": "월화수목금토일"[r.date.weekday()], "overlap_prev": r4(r.overlap_prev),
                   "base8": r4(r.base_overlap), "overlap_week_ago": r4(o7.iloc[0]) if len(o7) else None})
    sc_in = sc[(sc.date >= date(2025, 8, 21)) & (sc.date <= date(2026, 10, 21))]
    out["salt_daily"] = {"days": int(len(sc_in)), "median_overlap_prev": r4(sc_in.overlap_prev.median()),
                         "min_overlap_prev": r4(sc_in.overlap_prev.min()),
                         "min_day": str(sc_in.loc[sc_in.overlap_prev.idxmin(), "date"]),
                         "flagged_prev_day": fl,
                         "week_ago": {"median": r4(ov7.ov7.median()), "min": r4(ov7.ov7.min()),
                                      "min_day": str(ov7.loc[ov7.ov7.idxmin(), "date"]) if ov7.ov7.notna().any() else None,
                                      "flagged": [{"date": str(r.date), "weekday": "월화수목금토일"[r.date.weekday()],
                                                   "week_ago_date": str(r.date - timedelta(days=7)), "ov7": r4(r.ov7),
                                                   "base8": r4(r.base7),
                                                   "overlap_prev": r4(sc.loc[sc.date == r.date, "overlap_prev"].iloc[0])
                                                   if (sc.date == r.date).any() else None}
                                                  for r in ov7[ov7.flag7].itertuples()]},
                         "small_days_lt100_authors": [{"date": str(r.date), "n_authors": int(r.n_authors)}
                                                       for r in sc.itertuples() if r.n_authors < 100]}
    # 글 번호 500개 묶음 사이 작성자 겹침: 수집 실행 경계
    allu = pd.concat([posts[["no", "a"]], units[units.utype == "comment"][["no", "a"]]])
    allu = allu[allu.a.astype(str) != ""]
    allu = allu.assign(blk=allu["no"] // 500)
    bsets = allu.groupby("blk")["a"].apply(set)
    blks = sorted(bsets.index)
    ov = []
    for b0, b1 in zip(blks[:-1], blks[1:]):
        if b1 != b0 + 1:
            continue
        s0, s1 = bsets[b0], bsets[b1]
        ov.append({"blk": int(b1) * 500, "overlap": len(s0 & s1) / max(1, min(len(s0), len(s1)))})
    ovd = pd.DataFrame(ov)
    rb = run_boundaries()
    rt_first_no = int(posts.loc[pd.to_datetime(posts.dt_kst).dt.date >= REALTIME_FROM, "no"].min())
    bounds = [{"name": f"실행 경계 {b['stop_log']} → {b['resume_log']}", "no": b["no_first_after"] + 1000, **b} for b in rb]
    bounds.append({"name": "실시간 수집 시작(10/13 등록 첫 글)", "no": rt_first_no})
    near = []
    for b in bounds:
        s = ovd[(ovd.blk >= b["no"] - 1500) & (ovd.blk <= b["no"] + 1500)]
        near.append({"boundary": b["name"], "no": int(b["no"]),
                     "blocks": [{"blk": int(r.blk), "overlap": r4(r.overlap)} for r in s.itertuples()]})
    out["salt_blocks"] = {"pairs": int(len(ovd)), "median": r4(ovd.overlap.median()), "min": r4(ovd.overlap.min()),
                          "min_blk": int(ovd.loc[ovd.overlap.idxmin(), "blk"]),
                          "lt_half_median": int((ovd.overlap < 0.5 * ovd.overlap.median()).sum()),
                          "run_boundaries": rb, "near": near}

    # 3. 첫 쪽 = 가장 이른 댓글인가(두 줄 글)
    t_rows = t_kept = t_prefix = 0
    a_rows = a_kept_num = a_kept_den = 0
    new_root_after = new_root_total = 0
    for rs in multi.values():
        q1, q2 = rs[0]["rec"], rs[-1]["rec"]
        k1 = [(c["a"], c["text"], c["dt"][-8:]) for c in q1["comments"]]
        k2 = [(c["a"], c["text"], c["dt"][-8:]) for c in q2["comments"]]
        if not k1:
            continue
        s2 = set(k2)
        a_rows += 1
        a_kept_num += sum(1 for k in k1 if k in s2)
        a_kept_den += len(k1)
        roots1 = [(c["a"], c["text"], c["dt"][-8:]) for c in q1["comments"] if c["depth"] == 0]
        roots2 = [(c["a"], c["text"], c["dt"][-8:]) for c in q2["comments"] if c["depth"] == 0]
        trunc2 = int(q2["ncomment"]) > len(q2["comments"])
        if trunc2 and len(roots2) >= 10 and roots1:
            t_rows += 1
            sr2 = set(roots2)
            t_kept += int(all(k in sr2 for k in roots1))
            pos_ = [roots2.index(k) for k in roots1 if k in sr2]
            t_prefix += int(len(pos_) == len(roots1) and max(pos_) < len(roots1))
        sr1 = set(roots1)
        newr = [i for i, k in enumerate(roots2) if k not in sr1]
        new_root_total += len(newr)
        idx1 = [roots2.index(k) for k in roots1 if k in set(roots2)]
        if idx1:
            new_root_after += sum(1 for i in newr if i > max(idx1))
    out["first_page"] = {"rows_with_c1": a_rows, "c1_kept_share": r4(a_kept_num / max(1, a_kept_den)),
                         "trunc_rows_10roots": t_rows, "trunc_c1_roots_all_kept": t_kept, "trunc_c1_roots_prefix": t_prefix,
                         "new_roots_total": new_root_total, "new_roots_after_old": new_root_after}
    p = posts.set_index("no")
    c = units[(units.utype == "comment") & units.dt_conf.isin(["full", "matched", "row"]) & ~units.before_post.fillna(False)]
    c = c.assign(delay=(c.dt_kst - c.post_dt_kst).dt.total_seconds() / 60)
    g = c.groupby("no")["delay"]
    roots = units[(units.utype == "comment") & (units.depth == 0)].groupby("no").size()
    p = p.assign(roots=roots.reindex(p.index).fillna(0), fd=g.min().reindex(p.index), ld=g.max().reindex(p.index))
    tr = p[p.trunc_last.fillna(False).astype(bool)]
    nt = p[~p.trunc_last.fillna(False).astype(bool) & (p.roots >= 8)]
    diff = (tr.ncomment_last - tr.nsaved_last).astype(float)
    out["page"] = {"posts": int(len(p)), "trunc_posts": int(len(tr)), "trunc_share": r4(len(tr) / len(p)),
                   "roots_max": int(p.roots.max()), "trunc_roots_eq10": int((tr.roots == 10).sum()),
                   "trunc_roots_ge10": int((tr.roots >= 10).sum()),
                   "trunc_diff_le2": int((diff <= 2).sum()), "trunc_diff_median": r4(diff.median()),
                   "first_delay_min_trunc_median": r4(tr.fd.median()), "first_delay_min_nontrunc8_median": r4(nt.fd.median()),
                   "last_delay_min_trunc_median": r4(tr.ld.median()), "last_delay_min_nontrunc8_median": r4(nt.ld.median())}

    # 4. 본문 수정, 수정 표지, 시각만 있는 댓글
    pp = posts.copy()
    pp["pdate"] = pd.to_datetime(pp.dt_kst).dt.date
    pp["rt"] = pp.pdate >= REALTIME_FROM
    two = pp[pp.n_rows.fillna(1) >= 2]
    mark = re.compile(r"추가|수정|업데이트|(?<![A-Za-z])p\.?s(?![A-Za-z])", re.I)
    pp["edit_mark"] = (pp.title.fillna("") + "\n" + pp.text.fillna("")).str.contains(mark)
    ed = {}
    for lab, sub in (("실시간(10/13~)", two[two.rt]), ("그 전", two[~two.rt])):
        ed[lab] = {"two_row_posts": int(len(sub)), "body_edited": int(sub.body_edited.fillna(False).sum()),
                   "title_edited": int(sub.title_edited.fillna(False).sum()),
                   "body_edited_share": r4(sub.body_edited.fillna(False).mean()) if len(sub) else None}
    ed["edit_marker_share"] = {"실시간(10/13~)": r4(pp[pp.rt].edit_mark.mean()), "그 전": r4(pp[~pp.rt].edit_mark.mean())}
    cd = pd.read_csv(RES / "loader_comment_dates.csv")
    cdp = cd[cd["by"] == "post_date"].copy()
    cdp["date"] = pd.to_datetime(cdp["date"]).dt.date
    t_only = cdp[cdp.time_only > 0]
    ed["time_only_by_post_date"] = [{"date": str(r.date), "time_only": int(r.time_only),
                                     "conf_rate": r4(r.conf_rate_time_only), "latest": int(r.latest)} for r in t_only.itertuples()]
    summ = json.loads((RES / "loader_summary.json").read_text("utf-8"))
    ed["comment_dates"] = summ["comment_dates"]
    # 마감 전(국내) 댓글 중 날짜를 '가장 늦은 날'로 둔 것(latest)
    pre_kr = units[(units.utype == "comment") & units.pre_close_kr.fillna(False).astype(bool)]
    ed["pre_close_kr_latest"] = int((pre_kr.dt_conf == "latest").sum())
    ed["pre_close_kr_total"] = int(len(pre_kr))
    out["edits"] = ed

    # 5. 날짜별 구조(글 등록일): 수집 시점이 다른 구간끼리
    cm = units.loc[units.utype == "comment", ["no"]].assign(deleted=(hits_excl[units.utype == "comment"] == "deleted").to_numpy())
    dele = cm.groupby("no")["deleted"].agg(["sum", "count"])
    pq_ = pp.set_index("no").join(dele)
    by = pq_.groupby("pdate").agg(posts=("a", "size"), stored=("count", "sum"), deleted=("sum", "sum"),
                                  trunc=("trunc_last", "sum"), two_rows=("n_rows", lambda s: int((s >= 2).sum())))
    mb = pd.read_csv(RES / "loader_missing_by_day.csv")
    mb["date"] = pd.to_datetime(mb["date"]).dt.date
    by = by.join(mb.set_index("date")[["missing_rate"]])
    by.to_csv(RES / "prechecks_full_by_postdate.csv", encoding="utf-8-sig")

    def grp(lo, hi, weekday_only=False):
        s = by[(by.index >= lo) & (by.index <= hi)]
        if weekday_only:
            s = s[[d.weekday() < 5 for d in s.index]]
        if not len(s):
            return None
        return {"days": int(len(s)), "posts": int(s.posts.sum()), "stored_per_post": r4(s.stored.sum() / s.posts.sum()),
                "deleted_share": r4(s.deleted.sum() / max(1, s.stored.sum())), "trunc_share": r4(s.trunc.sum() / s.posts.sum()),
                "missing_rate_median": r4(s.missing_rate.median()), "two_row_share": r4(s.two_rows.sum() / s.posts.sum())}
    periods = [("2025-08-20~12-31 (10/21~22 수집, 약 10~14달 뒤)", date(2025, 8, 20), date(2025, 12, 31)),
               ("2026-01-01~04-14 (10/19~21 수집, 약 6~9달 뒤)", date(2026, 1, 1), date(2026, 4, 14)),
               ("2026-04-15~07-17 (10/16~19 수집, 약 3~6달 뒤)", date(2026, 4, 15), date(2026, 7, 17)),
               ("2026-07-18~10-11 (10/14~16 수집, 4일~3달 뒤)", date(2026, 7, 18), date(2026, 10, 11)),
               ("2026-10-12 (경계)", date(2026, 10, 12), date(2026, 10, 12)),
               ("2026-10-13~10-21 (실시간 + 24시간 다시 받기)", REALTIME_FROM, date(2026, 10, 21))]
    out["by_period"] = {lab: grp(lo, hi) for lab, lo, hi in periods}
    out["by_period_weekday"] = {"9/14~10/11 평일": grp(date(2026, 9, 14), date(2026, 10, 11), True),
                                "10/13~10/21 평일(실시간)": grp(REALTIME_FROM, date(2026, 10, 21), True)}
    trs = by.trunc / by.posts
    out["trunc_by_day"] = {"median": r4(trs.median()), "min": r4(trs.min()), "max": r4(trs.max()), "max_day": str(trs.idxmax())}
    return out


def nw_boundary(kr: pd.DataFrame) -> dict:
    """H35 가격 전 점검: 이력 누적 추세, 10/13 앞뒤 수준 점프(NW90 등록판·NW30 비등록판)."""
    out = {}
    for col in ("NW90", "NW30", "NW90_rz8"):
        d = kr[["D", col, "realtime", "judge_ok"]].copy()
        d = d[d.judge_ok.astype(bool) & d[col].notna()]
        pre = d[~d.realtime.astype(bool)]
        rt = d[d.realtime.astype(bool)]
        if len(pre) < 20:
            out[col] = None
            continue
        last20 = pre[col].iloc[-20:]
        p26 = pre[pre.D >= JAN2]
        slope = float(np.polyfit(np.arange(len(p26)), p26[col].to_numpy(float), 1)[0]) if len(p26) > 10 else None
        out[col] = {"pre_n": int(len(pre)), "pre_first": str(pre.D.iloc[0]), "pre_mean_2026": r4(p26[col].mean()),
                    "pre_last20_mean": r4(last20.mean()), "pre_last20_sd": r4(last20.std(ddof=1)),
                    "rt_n": int(len(rt)), "rt_mean": r4(rt[col].mean()),
                    "jump_z_vs_last20": r4((rt[col].mean() - last20.mean()) / last20.std(ddof=1)) if len(rt) else None,
                    "slope_per_window_2026": r4(slope),
                    "monthly_mean": {f"{k[0]}-{k[1]:02d}": r4(v) for k, v in
                                     d.groupby([d.D.map(lambda x: x.year), d.D.map(lambda x: x.month)])[col].mean().items()}}
    return out


# ---------------------------------------------------------------- 7. 분모 분포

def denominators(T: dict, units: pd.DataFrame, hits: pd.DataFrame, pred_a: pd.DataFrame) -> dict:
    kr, us = T["daily"]["kr"], T["daily"]["us"]
    j = kr.judge_ok.astype(bool)
    j26 = j & in_period(pd.Series(list(kr.D), index=kr.index), FRONT[0], BACK[1])
    out: dict = {}
    ok = (hits.excl == "") & (units.utype == "post")
    p = units[ok][["no", "a", "title", "text", "win_kr"]].join(hits[ok][["score"]])
    tag = p[(p.title.fillna("") + "\n" + p.text.fillna("")).str.contains("코스닥")]
    a_mean = tag.groupby(["win_kr", "a"])["score"].mean()
    votes = (a_mean != 0).groupby(level=0).sum().reindex(kr.index).fillna(0)
    out["H09_kosdaq_tag_lower"] = {"tag_posts_median": r4(tag.groupby("win_kr").size().reindex(kr.index).fillna(0)[j].median()),
                                   "votes_median": r4(votes[j].median()), "votes_max": r4(votes[j].max()),
                                   "share_votes_ge20": r4((votes[j] >= 20).mean())}
    u = units[(hits.excl == "")][["utype", "title", "text", "win_kr"]]
    txt = np.where(u.utype == "post", u.title.fillna("") + "\n" + u.text.fillna(""), u.text.fillna(""))
    rec = []
    for wid, t in zip(u.win_kr.to_numpy(), txt):
        if wid is None or wid != wid:
            continue
        cl = P.m9_claims(t)
        if not cl:
            continue
        first = bool(P.M10.search(t))
        bw = any(s > 0 and ((k == "%" and v >= 100) or (k == "배" and v >= 2)) for s, v, k in cl)
        bl = any(s < 0 and k == "%" and v >= 30 for s, v, k in cl)
        rec.append((wid, first, bw, bl))
    cdf = pd.DataFrame(rec, columns=["win", "m10", "bw", "bl"])
    res = {}
    for lab, sub in (("M9", cdf), ("M9+M10", cdf[cdf.m10])):
        n = sub.groupby("win").size().reindex(kr.index).fillna(0)
        s5 = n.rolling(5, min_periods=5).sum()
        ev = s5.notna() & j26
        res[lab] = {"claims_per_window_median": r4(n[j].median()), "claims_total": int(n.sum()),
                    "bw_total": int(sub.bw.sum()), "bl_total": int(sub.bl.sum()),
                    "miss_5win_lt5_2026": r4((s5[ev] < 5).mean())}
    out["H23_claims"] = res
    mc = kr["MC_n_all"]
    out["H33_mc_authors"] = {"median": r4(mc[j].median()), "share_ge3": r4((mc[j] >= 3).mean()),
                             "sig_windows": int(kr["H33_sig"].fillna(False).astype(bool).sum()),
                             "sig_computable": int(kr["H33_sig"].notna().sum()),
                             "sig_windows_realtime": int(kr.loc[kr.realtime.astype(bool), "H33_sig"].fillna(False).astype(bool).sum())}
    ju = us.judge_ok.astype(bool)
    out["H33_mc_authors_us"] = {"median": r4(us["MC_n_all"][ju].median()), "share_ge3": r4((us["MC_n_all"][ju] >= 3).mean())}
    den = kr["h37_den_bear"]
    comp = den.notna() & j26
    ym = pd.Series([f"{d.year}-{d.month:02d}" for d in kr["D"]], index=kr.index)
    out["H37_den"] = {"computed_2026": int(comp.sum()), "ge5_of_computed_2026": r4((den[comp] >= 5).mean()),
                      "lt5_of_computed_2026": r4((den[comp] < 5).mean()),
                      "ge5_of_all_judge_2026": r4((den[j26] >= 5).sum() / j26.sum()), "median_2026": r4(den[comp].median()),
                      "by_month_median": {k: r4(den[(den.notna() & j) & (ym == k)].median()) for k in sorted(ym[den.notna() & j].unique())}}
    ne = kr["n_news_event"]
    out["H16_news"] = {"event_zero_share_2026": r4((ne[j26] == 0).mean()), "event_lt5_share_2026": r4((ne[j26] < 5).mean()),
                       "event_median_2026": r4(ne[j26].median()), "event_zero_share_all": r4((ne[j] == 0).mean()),
                       "price_zero_share_2026": r4((kr["n_news_price"][j26] == 0).mean()),
                       "us_event_zero_share": r4((us["n_news_event"][ju] == 0).mean())}
    # H43 Herd 사건(직전 120창 상위 5%, 같은 종목 10일 안 재점등은 한 사건 — feat_topic 그대로)
    ev = {}
    for mk, df in (("kr", kr), ("us", us)):
        Dm = pd.Series(list(df.D), index=df.index)
        cols = [c for c in df.columns if c.startswith("h43_event_")]
        for c in cols:
            s = df[c].fillna(0).astype(float)
            ev.setdefault(mk, {})[c.replace("h43_event_", "")] = {
                "computable_windows": int(df[c].notna().sum()), "first": str(min(Dm[df[c].notna()])) if df[c].notna().any() else None,
                "events_2026": int(s[in_period(Dm, FRONT[0], BACK[1])].sum()),
                "events_front": int(s[in_period(Dm, *FRONT)].sum()), "events_back": int(s[in_period(Dm, *BACK)].sum())}
    tot = {k: sum(v["events_2026"] for mk in ev.values() for c, v in mk.items() if c.startswith(k)) for k in ("buy", "sell")}
    out["H43_events"] = {"by_stock": ev, "total_2026": tot}
    # 예측 층 분모: 창당 예측 쌍(작성자당 1회)
    pa = pred_a[pred_a.judge_ok & pred_a.labeled]
    late = pa[pd.Series(list(pa.D), index=pa.index) >= LLM[0]]
    out["pred_pairs"] = {"all_labeled_median": r4(pa.n_pred.median()), "late_median": r4(late.n_pred.median()),
                         "late_min": int(late.n_pred.min()) if len(late) else None, "late_max": int(late.n_pred.max()) if len(late) else None,
                         "late_share_ge20": r4((late.n_pred >= 20).mean()), "late_share_ge5": r4((late.n_pred >= 5).mean()),
                         "late_share_zero": r4((late.n_pred == 0).mean()), "late_windows": int(len(late)),
                         "pred_unit_share_late": r4(late.n_pred_unit.sum() / max(1, late.n_pairs_unit.sum())),
                         "strong1_pairs_late_total": int(late.n_pred1.sum()), "strong1_zero_share_late": r4((late.n_pred1 == 0).mean())}
    return out


# ---------------------------------------------------------------- 8. 대표 특징끼리의 상관

def corr2(x: pd.Series, y: pd.Series, D: pd.Series, judge: pd.Series) -> dict:
    j26 = judge & in_period(D, FRONT[0], BACK[1])
    return {"all": P.corr_pair(x[judge], y[judge]), "y2026": P.corr_pair(x[j26], y[j26]),
            "back": P.corr_pair(x[judge & in_period(D, *BACK)], y[judge & in_period(D, *BACK)])}


def exceeds(c: dict, thr: float) -> bool | None:
    vals = []
    for k in ("all", "y2026"):
        cc = c.get(k) or {}
        for m in ("pearson", "spearman"):
            if cc.get(m) is not None:
                vals.append(abs(cc[m]))
    return (max(vals) > thr) if vals else None


def correlations(T: dict, pred_a: pd.DataFrame, pred_b: pd.DataFrame) -> dict:
    kr, us = T["daily"]["kr"], T["daily"]["us"]
    Dk = pd.Series(list(kr.D), index=kr.index)
    Du = pd.Series(list(us.D), index=us.index)
    jk, ju = kr.judge_ok.astype(bool), us.judge_ok.astype(bool)
    out = {
        "H04_corr_U_N_rz8": corr2(kr.U_rz8, kr.N_rz8, Dk, jk),
        "H04_corr_U_N_raw_per_h": corr2(kr.U_per_h, kr.N_per_h, Dk, jk),
        "H04_corr_U_N_rz8_us": corr2(us.U_rz8, us.N_rz8, Du, ju),
        "H54_corr_SH_raw_A_rz8": corr2(kr.SH, kr.A_rz8, Dk, jk),
        "H54_corr_A6_A_rz8": corr2(kr.A6_SH, kr.A_rz8, Dk, jk),
        "H54_corr_SH_A_raw": corr2(kr.SH, kr.A_raw, Dk, jk),
        "H55_corr_K_raw_A_rz8": corr2(kr.K, kr.A_rz8, Dk, jk),
        "H55_corr_A7_A_rz8": corr2(kr.A7_K, kr.A_rz8, Dk, jk),
        "H55_corr_K_A_raw": corr2(kr.K, kr.A_raw, Dk, jk),
        "H36_corr_A4_A_rz8(NEW90)": corr2(kr.NEW90_rz8, kr.A_rz8, Dk, jk),
        "H36_corr_NEW90_A_raw": corr2(kr.NEW90_per_h, kr.A_raw, Dk, jk),
        "H35_corr_NW90rz8_A_rz8": corr2(kr.NW90_rz8, kr.A_rz8, Dk, jk),
        "H11_corr_DSDraw_B(표20+)": corr2(kr.D_SD_raw.where(kr.n_votes >= 20), kr.B, Dk, jk),
        "H11_corr_DSDraw_absB(표20+)": corr2(kr.D_SD_raw.where(kr.n_votes >= 20), kr.B.abs(), Dk, jk),
        "H11_corr_DSD_B(등록)": corr2(kr.D_SD, kr.B, Dk, jk),
        "H03_corr_A_Asaved_rz8": corr2(kr.A_rz8, kr.A_saved_rz8, Dk, jk),
        "H08_corr_fight_lncomments": corr2(kr.fight, kr.ln_n_comments_onposts, Dk, jk),
        "H08_corr_fight_A_rz8": corr2(kr.fight, kr.A_rz8, Dk, jk),
    }
    # H02: corr(TMP, S) — TMP 자리는 예측 층 P̄(부록 4). 7/1 이후 창(판정 구간), 그리고 예열 라벨까지 넣은 참고판
    for tag, pr in (("a", pred_a), ("b", pred_b)):
        x = pr["P_bar"].reindex(kr.index)
        lab = pr["labeled"].reindex(kr.index).fillna(False).astype(bool)
        late = jk & lab & (Dk >= LLM[0])
        allp = jk & lab
        out[f"H02_corr_Pbar_S_{tag}"] = {"late": P.corr_pair(x[late], kr.S[late]), "all_labeled": P.corr_pair(x[allp], kr.S[allp])}
    pa = pred_a.reindex(kr.index)
    late = jk & pa["labeled"].fillna(False).astype(bool) & (Dk >= LLM[0])
    out["H70_corr_Pshare_rz8_A_rz8(late)"] = P.corr_pair(pa.P_share_rz8[late], kr.A_rz8[late])
    out["H67_corr_Psd_lite_Pbar_abs(late)"] = P.corr_pair(pa.P_sd_lite[late], pa.P_bar[late].abs())
    out["H66_H68_corr_Pbar_Pstrong1(late)"] = P.corr_pair(pa.P_bar[late], pa.P_strong1[late])
    # H34: NS 20창 평균 vs LL 신호(같은 W_U). 원값(가격 잔차화 전), 상위 15% 겹침(전 표본 순위: 설명용 점검값)
    u = us[ju]
    ns20 = u.NS_share.rolling(20, min_periods=20).mean()
    ll20 = u.LL_US_A_all.rolling(20, min_periods=20).mean()
    d = pd.DataFrame({"ns20": ns20, "ll_d20": u.LL_US_d20, "ll20": ll20, "D": u.D}).dropna()
    ov = {"n": int(len(d)), "corr_ns20_lld20": P.corr_pair(d.ns20, d.ll_d20), "corr_ns20_ll20": P.corr_pair(d.ns20, d.ll20)}
    for nm in ("ll_d20", "ll20"):
        for lab, sub in (("all", d), ("y2026", d[d.D >= JAN2])):
            if len(sub) < 10:
                continue
            on_ns = sub.ns20 >= sub.ns20.quantile(0.85)
            on_ll = sub[nm] >= sub[nm].quantile(0.85)
            both = int((on_ns & on_ll).sum())
            ov[f"top15_overlap_{nm}_{lab}"] = {"ns_on": int(on_ns.sum()), "ll_on": int(on_ll.sum()), "both": both,
                                                "share_of_ns_on": r4(both / max(1, int(on_ns.sum())))}
    out["H34_NS20_vs_LL"] = ov
    out["H35_NW_boundary"] = nw_boundary(kr)
    return out


# ---------------------------------------------------------------- 10·11

def lexicon_and_frozen() -> dict:
    lexp = json.loads((INTERIM / "results" / "lexicon_v02_precision.json").read_text("utf-8"))
    prm = lexp["params"]
    bundles = {}
    for k, v in lexp["bundles"].items():
        irony = v.get("irony_rate_any") or 0.0
        bundles[k] = {"precision": r4(v["precision"]), "irony_any": r4(irony),
                      "pass": bool(v["precision"] >= prm["BUNDLE_PREC_MIN"] and irony <= prm["IRONY_MAX"])}
    drop = [w for w in lexp["words"] if w.get("drop")]
    fixed = {"lexicon_v02.json": "915a2d24b82eebe6ae3ca12c697be4710675ebdfe82477fbb4fa2e0605cf32d3",
             "lexicon_v02_full.json": "b067c3d023b0bd123f04dd932926f59acf34f763084ee983029996bac0ea1762"}
    lex_fp = {n: {"final_copy": sha(RES / n), "interim": sha(INTERIM / "results" / n), "fixed": fixed[n]} for n in fixed}
    for n in lex_fp:
        lex_fp[n]["ok"] = lex_fp[n]["final_copy"] == lex_fp[n]["interim"] == fixed[n]
    frozen = {n: {"sha256": sha(ANT / n), "ok": sha(ANT / n) == s} for n, s in FROZEN_SHA.items()}
    rt = json.loads((RES / "run_features_timings.json").read_text("utf-8"))
    code_now = {n: sha(INTERIM / "code" / n) for n in rt.get("code_sha256", {})}
    return {"bundles": bundles, "n_pass": sum(b["pass"] for b in bundles.values()), "n_fail": sum(not b["pass"] for b in bundles.values()),
            "fail": sorted(k for k, b in bundles.items() if not b["pass"]), "pass": sorted(k for k, b in bundles.items() if b["pass"]),
            "dropped_words": len(drop), "lexicon_fingerprints": lex_fp, "frozen_docs": frozen,
            "frozen_code_same_as_run": {n: code_now[n] == s for n, s in rt.get("code_sha256", {}).items()},
            "precision_sample_period": "2026-07-13~10-15(가린 표본, 중간 검증). 2025-08~2026-07 글로는 정밀도를 다시 재지 않았다"}


def external() -> dict:
    ext = json.loads((INTERIM / "results" / "prechecks_external.json").read_text("utf-8"))
    out = {"interim_checked_at": ext.get("checked_at"), "network_used_now": False}
    nq = PREVIEW / "results" / "prices_nq_1h.parquet"
    if nq.exists():
        t = pq.read_table(nq, columns=["bar_start_et"]).column(0).to_pandas()
        out["NQ_1h_preview_file"] = {"rows": int(pq.ParquetFile(nq).metadata.num_rows), "first_bar_et": str(t.min()),
                                     "last_bar_et": str(t.max()), "note": "시각 열만 읽음(값 미사용)"}
    pm = PREVIEW / "results" / "prices_meta.json"
    if pm.exists():
        m = json.loads(pm.read_text("utf-8"))
        out["preview_prices_meta"] = {k: m.get(k) for k in ("start", "end", "source")}
        out["preview_prices_meta"]["tickers"] = list(m.get("tickers", {}).keys()) if isinstance(m.get("tickers"), dict) else m.get("tickers")
    # 저장소 검색(문서·분석 결과, ant_index·포럼 원본 제외)
    pats = {"8월 레버리지 규제 시행일": r"단일\s?종목 레버리지.{0,40}(시행|부터)|레버리지.{0,20}(규제|기본예탁금).{0,30}(\d{1,2}월 \d{1,2}일|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2})",
            "KOSPI200 야간선물": r"야간선물|야간 선물|KOSPI200 야간|night futures",
            "예탁원 보관 상위": r"보관 상위|보관금액|seibro|SEIBro",
            "VKOSPI": r"VKOSPI",
            "CPI·고용 일정": r"CPI 발표일|고용보고서 발표|BLS 일정|cpi_dates|nfp_dates",
            "KIND 청약 일정": r"수요예측 일정|청약 일정|공모 일정"}
    hits = {k: [] for k in pats}
    bases = [ROOT / "docs", ROOT / "data" / "derived" / "analysis", ROOT / "data" / "derived" / "desk"]
    for base in bases:
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in (".md", ".csv", ".json", ".py"):
                continue
            sp = str(p)
            if "ant_index" in sp or (ANT_DIR / "posts").as_posix() in p.as_posix() or p.stat().st_size > 3_000_000:
                continue
            try:
                txt = p.read_text("utf-8", errors="ignore")
            except OSError:
                continue
            for k, rx in pats.items():
                if re.search(rx, txt):
                    hits[k].append(str(p.relative_to(ROOT)))
    out["repo_search"] = {k: sorted(set(v))[:12] for k, v in hits.items()}
    out["files"] = {
        "VKOSPI 수동표": [str(p.relative_to(ROOT)) for p in (ROOT / "data").rglob("*vkospi*") if p.is_file()][:5],
        "금투협(KOFIA) 파일": sorted(p.name for p in (ROOT / "data" / "derived" / "analysis" / "leverage_flood" / "results").glob("kofia_*"))
        if (ROOT / "data" / "derived" / "analysis" / "leverage_flood" / "results").exists() else []}
    out["interim"] = {k: ext.get(k) for k in ("NQ=F_1h", "VKOSPI", "KOFIA_freesis", "KIND", "SEIBro", "FOMC_calendar", "BLS_schedule",
                                               "earnings_dates_yf")}
    return out


# ---------------------------------------------------------------- 판정 묶기

def verdicts(feats: list[dict], wrows: list[dict], cor: dict, den: dict, lexf: dict, warm: list[dict]) -> list[dict]:
    fail = set(lexf["fail"])
    by_id: dict[str, list[dict]] = defaultdict(list)
    for f in feats:
        by_id[f["id"]].append(f)
    wmap = {(w["id"], w["col"]): w for w in warm}
    rows = []

    def lexfail(lex: str) -> list[str]:
        return [g for g in re.split(r"[·,\s]+", lex.replace("없음", "")) if g in fail]

    for hid, fs in by_id.items():
        f = fs[0]
        reasons, notes = [], []
        lf = sorted({g for x in fs for g in lexfail(x["lex"])})
        is_pred = f["market"] == "pred"
        ms = f["miss_struct_judgeperiod"]
        if ms is not None and ms > MISS_MAX:
            reasons.append(f"구조 결측 {100 * ms:.1f}% > 30%")
        per_keys = ["llm"] if is_pred else ["front", "back"]
        h = f["h"]
        if hid not in ("H24", "H25", "H26", "H01") and h > 0:
            low = [k for k in per_keys if f[k]["blocks_strict_full"] < MIN_BLOCKS]
            if low:
                reasons.append("최소 표본: " + ", ".join(f"{PER_NAME[k]} 구간 독립 묶음 "
                                                     f"{f[k]['blocks_strict_full']} < 8 (h {h})" for k in low))
        w = wmap.get((hid, f["col"]))
        structural = (ms is not None and ms > MISS_MAX) or (w is not None and w["first_valid_now"] is None)
        if w and not w["filled_before_2026_01_02"] and not structural:
            notes.append(f"예열: 첫 값 {w['first_valid_now']} (2026 앞 구간 {w['front_windows_missing_by_warmup']}/{w['front_judge_windows']}창 비어 있음)")
        if w and w.get("needs_M0") is not False and "m0_first_now" in w and not structural:
            now_s = f"{w['m0_first_now']}부터" if w["m0_first_now"] else "2026 안에 없음"
            notes.append(f"X⊥(M0 120창) 지금 {now_s}, 2025-01 수집 투영 {w['m0_first_proj_2025_01'] or '2026 안에 없음'}")
        rows.append({"id": hid, "cell": f["cell"], "col": f["col"], "lex_fail": lf, "auto_explanatory": reasons, "notes": notes})
    # 상관 문턱
    extra = {
        "H04": ("corr(rz8 U, rz8 N)", exceeds(cor["H04_corr_U_N_rz8"], 0.9), cor["H04_corr_U_N_rz8"], 0.9),
        "H54": ("corr(SH, A)", exceeds(cor["H54_corr_SH_raw_A_rz8"], 0.8), cor["H54_corr_SH_raw_A_rz8"], 0.8),
        "H55": ("corr(K, A)", exceeds(cor["H55_corr_K_raw_A_rz8"], 0.8), cor["H55_corr_K_raw_A_rz8"], 0.8),
    }
    for r in rows:
        if r["id"] in extra:
            nm, ex, c, thr = extra[r["id"]]
            if ex:
                r["auto_explanatory"].append(f"{nm} > {thr}")
        if r["id"] == "H34":
            o = cor["H34_NS20_vs_LL"]
            sh = [v["share_of_ns_on"] for k, v in o.items() if k.startswith("top15_overlap")]
            if sh and max(sh) > 0.7:
                r["auto_explanatory"].append("LL 신호와 겹침 > 70%")
        if r["id"] == "H37":
            d = den["H37_den"]
            if d["ge5_of_computed_2026"] is not None and d["ge5_of_computed_2026"] < 0.6:
                r["auto_explanatory"].append(f"분모 5 이상인 날 {100 * d['ge5_of_computed_2026']:.1f}% < 60%(행 규칙)")
        if r["id"] == "H16":
            z = den["H16_news"]["event_zero_share_2026"]
            if z is not None and z > MISS_MAX and not any("구조 결측" in x for x in r["auto_explanatory"]):
                r["auto_explanatory"].append(f"사건 뉴스형 0개 창 {100 * z:.1f}% > 30%")
    return rows


# ---------------------------------------------------------------- 칸별 상태(전 가설)

EMOTION_IDS = ["H60", "H61", "H62", "H63", "H64", "H65", "H69", "H71", "H72", "H73", "H74", "H75"]
CELLS = [  # (묶음, id, 이름)
    ("T1", "H03", "글량 → KOSPI 다음 장 변동"), ("T1", "H08", "댓글 싸움 → 다음 장 변동(F2 대표, 부록 2)"),
    ("T1", "H67", "예측 분산 → 다음 장 변동"), ("T1", "H70", "예측 몫 → 다음 장 변동"),
    ("T1", "H06", "엇갈림 → 거래대금"), ("T1", "H10", "반도체 분산 → SOXX 변동"),
    ("T1", "H64·H65", "불안·감정 쏠림 → 다음 날 변동"),
    ("T2", "H35", "신참 비율 → KOSPI R10"), ("T2", "H46", "미장 이동 → SOXX R10"),
    ("T2", "H31", "레버리지 롱 → SOXX MDD10"), ("T2", "H66", "군중 예측 → KOSPI R5"),
    ("T2", "H68", "강한 하락 예측 → KOSPI R10"), ("T2", "H09", "분산 → KOSDAQ R5"),
    ("T2", "H14", "남는 순심리 → KOSPI R5"), ("T2", "H19", "환호 → KOSPI R5"), ("T2", "H24", "공포 과부족 → R10(사건)"),
    ("T2", "H39+H48", "종목 관심 → 21일"), ("T2", "H49", "저녁 흥분 → SOXX 장중"), ("T2", "H51", "새벽 심리 → KOSDAQ 장중"),
    ("T2", "H60~H63·H69·H71~H74", "감정 층 칸"),
    ("F11", "H57a", "기분 단계 → R5"), ("F11", "H57b", "끝 단계 → 변동"),
    ("순차", "H05", "감정 밀도"), ("순차", "H04", "참여 폭"), ("순차", "H52", "주말 곱씹기 → 월 갭"),
    ("순차", "H11", "싸움 끝 → 반대"), ("순차", "H15압력", "비관 → 다음 장"), ("순차", "H12", "시끄러운 소수"),
    ("순차", "H15반전", "비관 → 2~5일 되돌림"), ("순차", "H16", "사건 뉴스 vs 의견 비관"),
    ("순차", "H54", "짧은 글(SH)"), ("순차", "H55", "주목 쏠림(K)"), ("순차", "H21", "소외(FOMO)"),
    ("순차", "H25", "항복 완료(사건)"), ("순차", "H26", "분노 → 체념(사건)"), ("순차", "H36", "상승 중 새 사람"),
    ("순차", "H37", "단골의 전향"), ("순차", "H43", "추격 고백 쏠림"), ("순차", "H40", "관심의 폭"),
    ("순차", "H41", "조용한 상승"), ("순차", "H50", "낮 메모리 수다 → 미국 메모리"), ("순차", "H53", "주말 감정 변화"),
    ("실시간", "H33", "반대매매 언급"), ("실시간", "H34", "심야 활동"),
    ("측정", "H01", "메아리 관문"), ("측정", "H02", "측정 선택(TMP vs S)"),
]


def cell_table(out: dict) -> list[dict]:
    fv = {v["id"]: v for v in out["verdicts"]}
    wk = {(w["id"], w["cell"], w["market"]): w for w in out["weekly"]}
    den, cor = out["denominators"], out["correlations"]
    rows = []
    for grp, hid, name in CELLS:
        reasons, notes, status = [], [], None
        if hid.startswith("H60") or hid == "H64·H65":
            status = "설명용(감정 층)"
            reasons.append("부록 8 시험: 주된 감정 일치 51% < 75%, 감정별 κ 미달 → 감정 층 칸은 설명용(desk_system §12.5)")
        elif hid in ("H39+H48", "H40", "H41"):
            status = "자동 설명용"
            b = wk.get(("H39+H48", "강건판(현재 12종목)", "kr"), {}).get("blocks_back")
            reasons.append(f"21일 목표라 구간당 독립 묶음 {b} < 8(부록 2 #2와 같은 결론, 횡단면도 시간 묶음으로 셈)")
            if hid == "H39+H48":
                reasons.append("문서 규칙판(15종목)은 현재 12종목 목록이라 모든 주 결측. 미국 강건판 구조 결측 "
                               f"{100 * wk[('H39+H48', '강건판(현재 12종목)', 'us')]['miss_struct']:.1f}%")
        elif hid == "H52":
            w = wk[("H52", "F1-3", "wknd")]
            status = "통과"
            notes.append(f"주말 {w['weeks']}개, 구조 결측 {100 * w['miss_struct']:.1f}%(월요일 휴장·명절 주말), 뒤 구간 주 {w['blocks_back']}개")
        elif hid == "H53":
            status = "측정 실패(사전 E·F) — 멈춤"
        elif hid == "H43":
            t = den["H43_events"]["total_2026"]
            status = "측정 실패(사전 CHASE_BUY·CHASE_SELL) — 멈춤"
            notes.append(f"2026 Herd 사건(현재 별칭 목록): 매수 {t['buy']}·매도 {t['sell']} → 30개 문턱은 넘음")
        elif hid == "H02":
            c = cor["H02_corr_Pbar_S_a"]["late"]
            status = "결정됨: P̄(예측 층)"
            notes.append(f"corr(P̄, S) 7/1~ {c['pearson']} / {c['spearman']} (n {c['n']}) ≤ 0.9. S는 사전 측정 실패, "
                         "예측 층은 사용자 시험 92% 통과(부록 8) → TMP 자리는 P̄")
        else:
            v = fv.get(hid)
            if v is None:
                status = "—"
            else:
                if v["lex_fail"]:
                    status = f"측정 실패(사전 {'·'.join(v['lex_fail'])}) — 멈춤"
                if v["auto_explanatory"]:
                    reasons += v["auto_explanatory"]
                    status = status + " / 자동 설명용" if status else "자동 설명용"
                notes += v["notes"]
                if hid in ("H24", "H25", "H26"):
                    notes.append("사건 수(구간마다 8)는 #9 가격 셈이라 판정 때(중간 검증 10/14까지: 2025 8·앞 3·뒤 5)")
                if hid == "H31" and not status:
                    status = "잠정 설명용(자료 대기)"
                    reasons.append("8월 단일종목 레버리지 규제 시행일을 아직 확보하지 못함(§5.1 #11). 날짜 하나만 확인하면 풀림")
                if hid == "H33" and not status:
                    status = "잠정 설명용(자료 대기) · 실시간 전용"
                    reasons.append("금투협 값의 실제 공개 시각은 원천에 없어 소급 불가")
                if hid == "H34" and not status:
                    o = cor["H34_NS20_vs_LL"]
                    sh = max(x["share_of_ns_on"] for k, x in o.items() if k.startswith("top15_overlap"))
                    status = "통과 · 실시간 전용"
                    notes.append(f"LL 신호와 상위 15% 겹침 최대 {100 * sh:.0f}% ≤ 70%")
                if hid in ("H57a", "H57b"):
                    notes.append("부록 3은 S 대신 들뜸 지수(감정 층)를 쓰게 했고, 감정 층은 부록 8 시험 실패로 설명용. 군중 예측 단계표는 보기용")
                if hid == "H01" and v["lex_fail"]:
                    notes.append("관문은 사전 S로 잰다. S 묶음(E·F)이 측정 실패라 감정 사전 칸은 멈춤 상태 그대로")
                if hid == "H03":
                    notes.append(f"저장 글 수판과 rz8 상관 {cp(cor['H03_corr_A_Asaved_rz8']['y2026'])}(2026), 부호 같음")
                if hid == "H08":
                    notes.append(f"corr(Fight, ln 댓글 수) {cp(cor['H08_corr_fight_lncomments']['y2026'])}(2026)")
                if hid == "H54":
                    notes.append(f"corr(SH, A) {cp(cor['H54_corr_SH_raw_A_rz8']['y2026'])}(2026) ≤ 0.8")
                if hid == "H55":
                    notes.append(f"corr(K, A) {cp(cor['H55_corr_K_raw_A_rz8']['y2026'])}(2026) ≤ 0.8")
                if hid == "H36":
                    notes.append(f"corr(A4, A) {cp(cor['H36_corr_A4_A_rz8(NEW90)']['y2026'])}(문턱 없음, A에 잔차화)")
                if hid == "H46":
                    f46 = next(f for f in out["features"] if f["id"] == "H46")
                    notes.append(f"AC1 {f46['ac1_all']:.2f} → 뒤 구간 n_eff {f46['back']['neff']:.1f}, MDE {f46['back']['mde']:.2f}(검정력 매우 낮음)")
                if hid == "H70":
                    pb = out["pred_build"]["b"]
                    notes.append(f"rz8 기준에 7월 전 라벨(예열)을 쓰면 7/1~ 74창 모두 값, 7월 이후만 쓰면 {pb['P_share_rz8_first']}부터 "
                                 f"{pb['late_valid']['P_share_rz8']}창")
                if hid == "H66" and not status:
                    status = "통과(조건: 7월 전 라벨로 M0 적합)"
                    notes.insert(0, "7/1~10/23 주 판정 구간은 X⊥가 모두 있음(7월 전 라벨을 M0 적합 예열에 쓸 때). 7월 이후 라벨만 쓰면 2026 안에 X⊥ 없음 → 시험 불가")
                if not status:
                    status = "통과" if not any(n.startswith(("예열", "X⊥")) for n in v["notes"]) else "통과(예열·X⊥는 2025-01 수집 뒤 다시 잼)"
        rows.append({"group": grp, "id": hid, "name": name, "status": status, "reasons": reasons, "notes": notes})
    return rows


# ---------------------------------------------------------------- 보고서(md)

WARM_LABEL = {
    "rz8": "rz8(같은 요일 8창)", "z60": "z60(40창)", "pct250": "pct250(120창)", "M0": "M0 확장 OLS(120창)",
    "M0(R²)": "M0 R²(120창)", "hist90+rz8": "90일 이력 + rz8", "hist90": "90일 이력", "hist30+rz8": "30일 이력 + rz8",
    "pool60": "풀 60창(40) + 표 20", "pool60+z60": "풀 60창 + 표 20 + z60", "pool60+rz8": "풀 60창 + 표 20 + rz8",
    "pool40": "잠정 풀 40창", "pool10": "풀 10창(비등록)", "chg20": "20창 변화", "tilt5+20": "5창 평균 − 20창 전",
    "z20": "z20(20창)", "att20+z60": "관심 기준 20일 + z60", "att20+rz8": "관심 기준 20일 + rz8", "sum5+z60": "5창 합 + z60",
    "hist30+5": "30일 이력 + 5창", "base20+2025계수": "직전 20창 + 2025 계수", "mean20": "20창 평균", "none": "없음(원값)",
    "rz8+reg20": "rz8 + 확장 회귀 20창", "z60|pct250": "z60 또는 pct250", "pct250+mean20": "pct250 + 20창 평균",
    "pct52주": "직전 52주 순위", "roll120": "직전 120창 상위 5%",
}


def pct(x, d=1) -> str:
    return "—" if x is None else f"{100 * x:.{d}f}%"


def num(x, d=2) -> str:
    if x is None:
        return "—"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{x:.{d}f}"


def cp(c: dict | None) -> str:
    if not c or c.get("pearson") is None:
        return "—"
    return f"{c['pearson']:.3f} / {c['spearman']:.3f} (n {c['n']})"


def render_md(out: dict) -> str:
    L = []
    a = L.append
    snap = out["snapshot"]
    c1, raw, den, cor, lexf, ext = (out[k] for k in ("check1", "raw", "denominators", "correlations", "lexicon_frozen", "external"))
    cells = out["cells"]
    pb = out["pred_build"]
    a("# 가격 전 점검표 — 전체 스냅샷 (2025-08-20~2026-10-21)")
    a("")
    a("> 정의: `HYPOTHESES.md` §5.1·§2.7·§3과 부록 1~10(동결판) 그대로. 해석 규칙은 중간 검증(`interim/notes_prechecks.md`)과 같다(구조 결측, 엄한 묶음 간격 h+1, 상관 문턱은 Pearson·Spearman 중 하나라도 넘으면 넘음).")
    a(f"> 자료: 포럼 스냅샷 {snap['first']}~{snap['cutoff']}(파일 {snap['n_files']}개, 줄 {snap['n_lines']:,}개, 기준 시각 {snap['S']}). 국내 W_KR 283창(판정 자격 280), 미국 W_U 294창(판정 자격 292). 예측 층 AI 라벨 {pb['a']['units']:,}단위(미리보기 재사용 4,320 + 새 12,660, 43조각 모두 다 참).")
    a("> 가격: 이번 점검은 가격을 쓰지 않았다(9번은 판정 때). feat_text의 지난 가격 표지 열 " + str(len(out["price_cols_not_read"])) + "개(px_·ev_)는 읽지 않았고, 외부 자료(#11)는 네트워크에 묻지 않았다.")
    a("> 코드: `final/code/prechecks_full.py`(interim `prechecks.py`·`loader.py`·`feat_text.py`, preview `pred_windows.py`를 고치지 않고 import). 결과: `final/results/prechecks_full.json`, `prechecks_full_features.csv`, `prechecks_full_by_postdate.csv`, `prechecks_full_pred_windows.parquet`.")
    fz = lexf["frozen_docs"]
    a(f"> 지문: 동결 문서 12개(HYPOTHESES·부록 1~10·lexicon_draft) SHA-256이 desk_system §12.5 기록과 {'모두 같음' if all(v['ok'] for v in fz.values()) else '다름 — 확인 필요'}(실행 전후). 사전 v0.2 두 파일 {'같음' if all(v['ok'] for v in lexf['lexicon_fingerprints'].values()) else '다름'}. 특징 코드 6개가 특징 실행 때와 {'같음' if all(lexf['frozen_code_same_as_run'].values()) else '다름'}. 예측 층 7월 이후 판은 미리보기 72창 값과 {'완전히 같음' if pb['preview_reproduction_ok'] else '다름'}.")
    a("")
    # 0. 결론
    a("## 0. 결론")
    a("")
    a("### 주 판정 칸 상태")
    a("| 묶음 | 가설 | 칸 | 상태 | 이유·메모 |")
    a("|---|---|---|---|---|")
    for r in cells:
        if r["group"] not in ("T1", "T2", "F11"):
            continue
        txt = "; ".join(r["reasons"] + r["notes"]) or "—"
        a(f"| {r['group']} | {r['id']} | {r['name']} | **{r['status']}** | {txt} |")
    a("")
    t1_ok = [r["id"] for r in cells if r["group"] == "T1" and r["status"].startswith("통과")]
    t2_ok = [r["id"] for r in cells if r["group"] == "T2" and r["status"].startswith("통과")]
    t2_wait = [r["id"] for r in cells if r["group"] == "T2" and r["status"].startswith("잠정")]
    a(f"- 지금 시험할 수 있는 주 칸: T1 {len(t1_ok)}칸({', '.join(t1_ok)}), T2 {len(t2_ok)}칸({', '.join(t2_ok)}). 자료 대기 T2: {', '.join(t2_wait) or '없음'}.")
    a("- 새로 자동 설명용이 된 칸: H67(창당 예측 쌍이 최대 " + str(den["pred_pairs"]["late_max"]) + "개라 '표 20개' 조건을 한 창도 못 채움, 구조 결측 100%), H68(7/1~10/23 거래일이 77일이라 10일 목표의 독립 묶음이 7개 < 8. 간격을 h로 읽어도 7이다. 강한 하락 예측 쌍도 7/1~ 74창 합 " + str(den["pred_pairs"]["strong1_pairs_late_total"]) + "개, 0인 창 " + pct(den["pred_pairs"]["strong1_zero_share_late"], 0) + "로 얇다).")
    a("- 중간 검증 때 자동 설명용이던 H16·H37은 전체 자료에서 결측 문턱 아래로 내려왔다(H16 사건 뉴스 0개 창 " + pct(den["H16_news"]["event_zero_share_2026"]) + ", H37 분모 5 미만 " + pct(den["H37_den"]["lt5_of_computed_2026"]) + "). 다만 둘 다 사전 측정 실패라 멈춤 상태는 그대로다.")
    a("- H04는 전체 자료에서도 corr(rz8 U, rz8 N)이 0.9를 넘어(" + cp(cor["H04_corr_U_N_rz8"]["y2026"]) + ", 2026) 설명용이 확정이다.")
    a("")
    a("### 2025-08 시작이라 생기는 예열 문제 (판정 전에 2025-01부터 다시 계산할 것)")
    warm = {(w["id"], w["col"]): w for w in out["warmup"]}
    a("| 가설 | 특징 | 지금 첫 값 | 2026 앞 구간 빈 창(X⊥ 기준) | X⊥(M0 120창) 지금 | 2025-01 수집 투영 |")
    a("|---|---|---|---|---|---|")
    for hid, col in (("H35", "NW90_rz8"), ("H36", "NEW90_rz8"), ("H46", "h46_dtilt"), ("H31", "LL_US_d20"), ("H50", "h50_m_z20"),
                     ("H66", "P_bar"), ("H68", "P_strong1"), ("H57a", "stage10")):
        w = warm.get((hid, col))
        if not w:
            continue
        xp = w.get("m0_first_now")
        proj = f"첫 값 {w['proj_first_valid_2025_01_start']}" + (f", X⊥ {w.get('m0_first_proj_2025_01') or '—'}" if "m0_first_now" in w else "")
        a(f"| {hid} | `{col}` | {w['first_valid_now']} | {w['front_windows_missing_by_warmup'] if 'm0_first_now' not in w else w['m0_front_windows_missing_now']}/{w['front_judge_windows']} | {(xp + '부터') if xp else ('—' if 'm0_first_now' not in w else '2026 안에 없음')} | {proj} |")
    a("")
    a("- rz8·z60·풀 60창 특징은 모두 2026-01-02 전에 값이 찬다(첫 값 2025-10-23~11-28). 못 채우는 것은 pct250(120창: 첫 값 2026-02-20), 90일 이력 + rz8(H35·H36: 2026-01-19), 그리고 §2.3 확장 OLS 최소 120창이 필요한 X⊥다.")
    a("- 방향(T2) 칸의 주 특징은 X⊥라서, 지금 스냅샷만으로는 앞 구간(1~5월)의 상당 부분이 비어 앞 구간 같은 부호 조건을 채울 수 없다. H35·H36은 X⊥가 8/3부터라 앞 구간 전체가 빈다. 2025-01부터 수집이 끝나면 X⊥가 2025-07~10월(H35·H36은 2026-01-02)부터 생겨 앞·뒤 구간이 모두 찬다.")
    a(f"- 예측 층(H66·H68)의 X⊥는 7월 전 라벨을 M0 적합(예열)에 쓰면 {warm[('H66', 'P_bar')]['m0_first_now']}부터 생기고, 7월 이후 라벨만 쓰면 2026 안에 생기지 않는다(미리보기의 '시험 불가'와 같은 이유). 라벨 표본 요약은 7월 전 단위를 '설명·예열용'으로 적었다. 7월 이후 창의 앞날 정보는 섞이지 않지만(라벨 모델 지식은 6월까지), 계수가 지식 기간 라벨로 정해진다는 점은 판정 때 표시해야 한다.")
    a(f"- H70의 rz8도 같은 갈림이 있다: 예열 허용판은 7/1부터 74창 모두 값이 있고, 7월 이후만 판은 {pb['b']['P_share_rz8_first']}부터 {pb['b']['late_valid']['P_share_rz8']}창만 값이 있다.")
    a("")
    # 1
    a("## 1. 결번율·장애 창·스팸 파도")
    k, u = c1["kr"], c1["us"]
    a("| 항목 | 국내 W_KR 본판 | 미국 W_U 본판 |")
    a("|---|---|---|")
    a(f"| 창 수 | {k['main']['n']} ({k['main']['first']}~{k['main']['last']}) | {u['main']['n']} ({u['main']['first']}~{u['main']['last']}) |")
    kname = {"weekday": "평일", "weekend": "주말 뒤", "holiday": "연휴 뒤"}
    a("| 종류 | " + ", ".join(f"{kname.get(x, x)} {n}" for x, n in k["main"]["kinds"].items()) + " | "
      + ", ".join(f"{kname.get(x, x)} {n}" for x, n in u["main"]["kinds"].items()) + " |")
    a(f"| 96시간 초과(판정 제외) | {', '.join(k['main']['over96']) or '없음'} | {', '.join(u['main']['over96']) or '없음'} |")
    a(f"| 장애 창(같은 요일 중앙값 20% 미만) | {', '.join(k['main']['outage']) or '0'} | {', '.join(u['main']['outage']) or '0'} (추석 휴장으로 조용한 창, 부록 2 #3대로 결측) |")
    a(f"| 장애 판단 불가(기준선 짧음) | {len(k['main']['outage_unknown'])}창(2025-08-21~09-10) | {len(u['main']['outage_unknown'])}창(2025-08-21~09-16) |")
    a(f"| 스팸 파도 | {len(k['main']['spam_wave'])} | {len(u['main']['spam_wave'])} |")
    a(f"| 결번율 중앙값 / 상위 10% / 최대 | {pct(k['main']['missing_rate']['median'])} / {pct(k['main']['missing_rate']['p90'])} / {pct(k['main']['missing_rate']['max'])}({k['main']['missing_rate']['max_win']}) | {pct(u['main']['missing_rate']['median'])} / {pct(u['main']['missing_rate']['p90'])} / {pct(u['main']['missing_rate']['max'])}({u['main']['missing_rate']['max_win']}) |")
    for lab in k["main_by_period"]:
        a(f"| 결번율 중앙값 {lab} | {pct(k['main_by_period'][lab]['median'])} (n {k['main_by_period'][lab]['n']}) | {pct(u['main_by_period'][lab]['median'])} (n {u['main_by_period'][lab]['n']}) |")
    a(f"| 강건판 장애 창 | 고정 24시간판 {len(k['fixed24']['outage'])} | 09:00 ET판 {len(u['close0900']['outage'])}, 고정 24시간판 {len(u['fixed24']['outage'])}({', '.join(u['fixed24']['outage'])}) |")
    a("")
    bd = c1["by_day"]
    a(f"- 날짜별 결번율 {bd['days']}일: 중앙값 {pct(bd['median'])}, 최대 {pct(bd['max'])}({bd['top'][0]['date']}, 번호 증가 {bd['top'][0]['d_no']}). 상위 8일이 모두 2025 주말·휴일이다(번호 증가 23~65로 분모가 작음).")
    a("- 달별 중앙값: " + ", ".join(f"{m} {pct(v)}" for m, v in bd["by_month_median"].items()) + ".")
    a(f"- 오래된 글일수록 결번율이 높다(2025-08~11 10~12% → 2026 하반기 7%대). §2.5 예상대로 지운 글 몫이 수집 시차와 함께 커진다. 실시간 날(10/13~)은 {pct(bd['realtime_median'])}로 직전 30일 {pct(bd['before_realtime_last30_median'])}의 절반이다. 결번율은 모든 텍스트 칸의 통제 변수로 넣는다(§2.2).")
    a("- 장애 창은 미국 9/24·9/25 두 개뿐이다(2025 추석·설 창은 장애 규칙에 걸리지 않음).")
    a("")
    # 2
    s = raw["salt"]
    sd = raw["salt_daily"]
    sb = raw["salt_blocks"]
    a("## 2. 소금 연속성")
    a("| 점검 | 값 | 판단 |")
    a("|---|---|---|")
    a(f"| state.json 소금 | 있음. 지문이 스냅샷 때와 {'같음' if s['fingerprint_same_as_snapshot'] else '다름'}. 값은 출력하지 않음 | 바뀌지 않음 |")
    a(f"| 소금 백업(state_salt_backup_20261017.json) | {'있음' if s['backup_present'] else '없음'}, 지금 소금과 {'같음' if s['backup_same_as_now'] else '다름'} | §2.2 백업 요구 충족 |")
    a(f"| 두 번 받은 글의 작성자 해시 | {s['post_author_same']:,}/{s['multi_row_posts']:,} 같음 | 같음 |")
    a(f"| 두 줄에 같이 나온 댓글(본문·시각)의 해시 | {s['comment_author_same']:,}/{s['comment_pairs']:,} 같음 | 같음 |")
    a(f"| 이웃 날짜 작성자 겹침률 | {sd['days']}일, 중앙값 {num(sd['median_overlap_prev'], 3)}, 최저 {num(sd['min_overlap_prev'], 3)}({sd['min_day']}) | 아래 표지 6일은 모두 월요일 |")
    fl = "; ".join(f"{x['date']}({x['weekday']}) {num(x['overlap_prev'], 3)}, 일주일 전 같은 요일 {num(x['overlap_week_ago'], 3)}" for x in sd["flagged_prev_day"])
    a(f"| 로더 표지(직전 8일 중앙값의 절반 미만) | {fl} | 주말 뒤 월요일 효과. 일주일 전 같은 요일과의 겹침은 평소(중앙값 {num(sd['week_ago']['median'], 3)})와 같음 → 끊김 아님 |")
    wf = "; ".join(f"{x['date']}({x['weekday']}) {num(x['ov7'], 3)} vs 기준 {num(x['base8'], 3)}, 전날 겹침 {num(x['overlap_prev'], 3)}" for x in sd["week_ago"]["flagged"])
    a(f"| 일주일 전 같은 요일 겹침 표지 | {wf or '없음'} | 비교 날이 연휴·주말이라 작성자가 적은 경우. 전날 겹침은 정상 → 끊김 아님 |")
    a(f"| 글 번호 500개 묶음 사이 작성자 겹침 | {sb['pairs']}쌍, 중앙값 {num(sb['median'], 3)}, 최저 {num(sb['min'], 3)}(번호 {sb['min_blk']:,}), 중앙값의 절반 미만 {sb['lt_half_median']}쌍 | 끊김 없음 |")
    for n in sb["near"]:
        a(f"| {n['boundary']} 근처 | " + ", ".join(f"{b['blk']:,}: {num(b['overlap'], 2)}" for b in n["blocks"]) + " | 경계에서 떨어지지 않음 |")
    a("")
    # 3
    fp, pg = raw["first_page"], raw["page"]
    a("## 3. 댓글 첫 쪽이 가장 이른 댓글인가, 잘림 비율")
    a("| 점검 | 값 |")
    a("|---|---|")
    a(f"| 한 쪽 크기 | 뿌리 댓글 10개. 잘린 글 {pg['trunc_posts']:,}개 중 {pg['trunc_roots_eq10']:,}개({pct(pg['trunc_roots_eq10'] / pg['trunc_posts'])})가 뿌리 10개, 최대 {pg['roots_max']} |")
    a(f"| 두 줄 글: 첫 줄 댓글이 둘째 줄에 남은 비율 | {pct(fp['c1_kept_share'], 2)} ({fp['rows_with_c1']:,}글) |")
    a(f"| 둘째 줄에서 잘린(뿌리 10개) 글: 첫 줄 뿌리 댓글이 둘째 줄 맨 앞에 그대로 | {fp['trunc_c1_roots_prefix']}/{fp['trunc_rows_10roots']} ({pct(fp['trunc_c1_roots_prefix'] / max(1, fp['trunc_rows_10roots']))}) |")
    a(f"| 둘째 줄의 새 뿌리 댓글이 옛 댓글 뒤에 붙은 비율 | {fp['new_roots_after_old']}/{fp['new_roots_total']} ({pct(fp['new_roots_after_old'] / max(1, fp['new_roots_total']))}) |")
    a(f"| 첫 저장 댓글 지연(중앙값) | 잘린 글 {num(pg['first_delay_min_trunc_median'], 1)}분, 안 잘린 글(뿌리 8개+) {num(pg['first_delay_min_nontrunc8_median'], 1)}분 |")
    a(f"| 마지막 저장 댓글 지연(중앙값) | 잘린 글 {num(pg['last_delay_min_trunc_median'], 0)}분, 안 잘린 글(뿌리 8개+) {num(pg['last_delay_min_nontrunc8_median'], 0)}분 |")
    td = raw["trunc_by_day"]
    a(f"| 잘린 글 비율 | 전체 {pct(pg['trunc_share'])}. 날짜별 중앙값 {pct(td['median'])}, 최소 {pct(td['min'])}, 최대 {pct(td['max'])}({td['max_day']}). 표는 `prechecks_full_by_postdate.csv` |")
    a(f"| 잘림 하한 표지의 성격 | 잘린 글 중 ncomment − 저장 수 ≤ 2인 글 {pg['trunc_diff_le2']:,}개({pct(pg['trunc_diff_le2'] / pg['trunc_posts'])}), 차이 중앙값 {num(pg['trunc_diff_median'], 0)} |")
    a("")
    a("- 판단: 중간 검증과 같다. 첫 쪽은 등록 순서로 가장 이른 10뿌리다. 잘린 글은 '초반 댓글은 다 있고 늦은 댓글이 빠진' 하한이므로 H08 주판(잘리지 않은 글)과 창별 잘림 비율 통제를 그대로 쓴다. 잘린 글의 첫 쪽은 등록 뒤 약 " + num(pg['last_delay_min_trunc_median'], 0) + "분에서 끝나 H12·H55의 120분 관심은 잘린 글에서 하한이다.")
    a("")
    # 4
    ed = raw["edits"]
    a("## 4. 실시간 첫 줄 vs 24시간 줄")
    a("| 점검 | 값 |")
    a("|---|---|")
    rt = ed["실시간(10/13~)"]
    a(f"| 실시간 글(10/13~) 두 줄 | {rt['two_row_posts']:,}글. 본문이 바뀐 글 {rt['body_edited']}({pct(rt['body_edited_share'], 2)}), 제목 {rt['title_edited']} |")
    a(f"| 수정 표지(추가·수정·ps·업데이트) 든 글 | 실시간 {pct(ed['edit_marker_share']['실시간(10/13~)'])}, 그 전 {pct(ed['edit_marker_share']['그 전'])} |")
    cdx = ed["comment_dates"]
    a(f"| 시각만 있는 댓글 날짜 확정 | {cdx['time_only']:,}개 중 {pct(cdx['conf_rate_time_only'], 2)}(다른 줄 맞춤 {cdx['matched']:,}, 줄 안 좁히기 {cdx['row']:,}, 가장 늦은 날짜 {cdx['latest']}). 전체 댓글 기준 {pct(cdx['conf_rate_all'], 2)} |")
    a("| 글 날짜별 확정률 | " + ", ".join(f"{x['date'][5:]} {pct(x['conf_rate'])}({x['time_only']:,})" for x in ed["time_only_by_post_date"]) + " |")
    a(f"| 국내 마감 전 댓글 중 '가장 늦은 날짜'로 둔 것 | {ed['pre_close_kr_latest']}/{ed['pre_close_kr_total']:,} |")
    a("")
    a("- 판단: 첫 줄 규칙 때문에 생기는 차이는 0.4% 안팎으로 작다. 날짜를 못 정한 댓글은 늦은 쪽으로만 틀리고, 국내 마감 전 집계에는 하나도 들지 않았다(H08·H28·H34 영향 없음).")
    a("")
    # 5
    a("## 5. 수집 시점이 다른 구간끼리의 구조 (경계 10/13 포함)")
    a("| 글 등록일 구간 | 날 | 글 | 글당 저장 댓글 | 지운 댓글 몫 | 잘린 글 | 결번율 중앙값 | 두 줄 글 |")
    a("|---|---|---|---|---|---|---|---|")
    for lab, g in raw["by_period"].items():
        if g:
            a(f"| {lab} | {g['days']} | {g['posts']:,} | {num(g['stored_per_post'])} | {pct(g['deleted_share'], 2)} | {pct(g['trunc_share'])} | {pct(g['missing_rate_median'])} | {pct(g['two_row_share'])} |")
    a("")
    a("| 평일만 | 날 | 글당 저장 댓글 | 잘린 글 | 지운 댓글 몫 |")
    a("|---|---|---|---|---|")
    for lab, g in raw["by_period_weekday"].items():
        a(f"| {lab} | {g['days']} | {num(g['stored_per_post'])} | {pct(g['trunc_share'])} | {pct(g['deleted_share'], 2)} |")
    a("")
    nb = cor["H35_NW_boundary"]
    a("| H35 신참 비율 | 실시간 전 첫 값 | 2026 평균 | 실시간 직전 20창 평균(SD) | 실시간 창 평균(n) | 점프 z | 2026 추세(창당) |")
    a("|---|---|---|---|---|---|---|")
    for col in ("NW90", "NW90_rz8", "NW30"):
        v = nb.get(col)
        if v:
            a(f"| `{col}` | {v['pre_first']} | {num(v['pre_mean_2026'], 3)} | {num(v['pre_last20_mean'], 3)} ({num(v['pre_last20_sd'], 3)}) | {num(v['rt_mean'], 3)} ({v['rt_n']}) | {num(v['jump_z_vs_last20'], 2)} | {num(v['slope_per_window_2026'], 4)} |")
    a("")
    a("- NW90 달 평균: " + ", ".join(f"{m} {num(x, 3)}" for m, x in nb["NW90"]["monthly_mean"].items()) + ".")
    a("- 판단: 지운 댓글 몫이 2025 글(10~14달 뒤 수집)에서 1.5%로 2026 글(0.06~0.2%)보다 열 배 넘게 높다. 2025 구간의 댓글 수·싸움 같은 댓글 특징은 수준이 다르므로 rz8·z60처럼 가까운 구간 기준 정규화를 그대로 쓰고, 2025는 예열·보조로만 쓴다(§2.5). 글당 저장 댓글과 잘린 글 비율은 실시간 쪽이 낮다(평일 4.76 vs 5.77, 7.6% vs 13.7%). 아직 댓글이 덜 붙은 상태에서 받은 몫이다(24시간 다시 받기 뒤에도 남음). lexicon §9-2대로 댓글 양 특징은 경계 앞뒤를 따로 본다.")
    a("- NW90은 2025-11~2026-02 약 0.11에서 2026-08~09 약 0.05로 내려왔다. 이력이 쌓여서 생긴 추세는 아니다(90일 고정 창). 오래된 구간일수록 지운 글이 많아 이력이 빠지는 몫과, 6~7월 급등락기에 새로 들어온 작성자가 단골이 된 몫이 섞여 있다. rz8(같은 요일 직전 8창)이 수준 변화를 대부분 흡수한다. 실시간 6창의 점프는 z 0.8~1.1로 수준 점프라 할 만큼 크지 않지만, 10/13 급락 직후와 겹쳐 활동 변화와 가를 수 없다.")
    a("")
    # 6
    a("## 6. 특징별 결측률·자기상관·n_eff·독립 묶음")
    a("- 결측률: 판정 자격 창 기준. **구조 결측** = 예열(최소 창 수)을 빼고도 남는 결측이고, 판정 구간(2026-01-02~10-23, 예측 층은 2026-07-01~10-23)에서 셌다. 30% 초과면 자동 설명용이다.")
    a("- n_eff = (n − h) ÷ (1 + 2Σρᵏ(1 − k/h)), ρ = 이웃 창 AC1(전체 기간, 0~0.99로 자름). MDE = 2.49/√n_eff. n은 구간 안 값 있는 판정 자격 창(스냅샷은 10/21까지라 10/22·10/23 두 창이 아직 없다).")
    a("- 묶음 = 목표가 구간 끝(앞 5/29, 뒤 10/23) 안에서 끝나는 창을 h+1창 간격(엄함)으로 고른 수. '지금'은 값 있는 창만, '예열 해소'는 구간 전 거래일에 값이 있다고 본 수다. 최소 표본 판단은 예열 해소판으로 한다(예열은 2025-01 자료로 사라지므로).")
    a("")
    a("| 가설 | 칸 | 열 | h | 정규화 | 첫 값 | 구조 결측 | 결측 앞 / 뒤 | AC1 | n_eff 뒤 | MDE 뒤 | 엄한 묶음 앞 / 뒤 (지금 → 예열 해소) |")
    a("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for f in out["features"]:
        bk = f.get("back") or f.get("llm")
        fr = f["front"]
        lab_b = "뒤" if "back" in f else "7/1~"
        a(f"| {f['id']} | {f['cell']} | `{f['col']}` ({f['market']}) | {f['h']} | {f['family']} | {f['first_valid'] or '—'} | {pct(f['miss_struct_judgeperiod'])} | "
          f"{pct(fr['miss'])} / {pct(bk['miss'])}{'' if lab_b == '뒤' else '(7/1~)'} | {num(f['ac1_all'])} | {num(bk['neff'], 1)} | {num(bk['mde'])} | "
          f"{fr['blocks_strict']}→{fr['blocks_strict_full']} / {bk['blocks_strict']}→{bk['blocks_strict_full']} |")
    a("")
    a("- 예측 층(H66~H70)의 '앞'은 AI 지식 기간(1~5월)이라 보조 확인만이고, '뒤' 칸은 주 판정 구간 7/1~10/23이다(부록 3).")
    a("- 사건형(H24·H25·H26)의 묶음은 사건 수로 센다. 사건 날짜는 가격 셈(#9)이라 판정 때 센다.")
    a("- H46 ΔTilt는 AC1 " + num(next(f['ac1_all'] for f in out['features'] if f['id'] == 'H46')) + "로 뒤 구간 n_eff " + num(next(f['back']['neff'] for f in out['features'] if f['id'] == 'H46'), 1) + ", MDE " + num(next(f['back']['mde'] for f in out['features'] if f['id'] == 'H46')) + "다. 통과 조건은 채우지만 검정력이 매우 낮아 판정 불가가 기본 예상이다.")
    a("")
    a("주 단위 칸(값/행은 종목 × 주, 21일 목표는 5주 간격)")
    a("")
    a("| 가설 | 칸 | 열 | 시장 | 값/행 | 주 | 결측 | 구조 결측 | 묶음 지금 | 묶음 뒤 | 메모 |")
    a("|---|---|---|---|---|---|---|---|---|---|---|")
    for w in out["weekly"]:
        a(f"| {w['id']} | {w['cell']} | `{w['col']}` | {w['market']} | {w['n_valid']}/{w['n_rows']} | {w['weeks']} | {pct(w['miss_all'])} | {pct(w['miss_struct'])} | {w.get('blocks_now', '—')} | {w['blocks_back']} | {w['memo']} |")
    a("")
    # 6b 예열 표
    a("## 6b. 최소 창 수(rz8 8창·z60 40창·pct250 120창·M0 120창)를 2026-01-02 전에 채우나")
    a("- '지금 첫 값'은 이 스냅샷(2025-08-20 시작)에서 값이 처음 생긴 창이다. '2025-01 투영'은 수집이 2025-01-01부터 다 찼다고 보고, 지금과 같은 예열 창 수를 더한 날이다(근사, 명절 위치에 따라 며칠 다를 수 있음).")
    a("- X⊥(M0 120창): §2.3 확장 OLS(D−1까지, 최소 120창, 월 1회 재적합). 값 있는 창이 그 달 첫 거래일 전에 120개 이상인 첫 달부터 생긴다. 가격 쪽 결측은 판정 때 다시 센다. 투영은 2026 실제 값 비율만큼 창이 더 필요하다고 봤다.")
    a("- '다시 계산' = 2025-01 자료가 들어오면 2026 값이 새로 생기거나 바뀌는 특징(첫 값이 2026, pct250, M0, 2025 계수, 120창 순위). rz8·z60·90일 이력은 2026 값이 바뀌지 않는다.")
    a("")
    a("| 가설 | 열 | 정규화 | 지금 첫 값 | 01-02 전 | 2026 앞 빈 창(예열) | X⊥ 지금 | X⊥ 앞 빈 창 | 2025-01 투영 첫 값 | X⊥ 투영 | 다시 계산 |")
    a("|---|---|---|---|---|---|---|---|---|---|---|")
    for w in out["warmup"]:
        has_m0 = "m0_first_now" in w
        xnow = (w["m0_first_now"] or "2026 안 없음") if has_m0 else "—"
        if has_m0 and w["market"] == "pred":
            xnow += f" (7월 이후만: {w.get('m0_first_now_july_only') or '2026 안 없음'})"
        a(f"| {w['id']} | `{w['col']}` | {WARM_LABEL.get(w['warm'], w['warm'])} | {w['first_valid_now'] or '없음'} | {'예' if w['filled_before_2026_01_02'] else '**아니오**'} | "
          f"{w['front_windows_missing_by_warmup']}/{w['front_judge_windows']} | {xnow} | {w.get('m0_front_windows_missing_now', '—') if has_m0 else '—'} | "
          f"{w['proj_first_valid_2025_01_start'] or '—'} | {(w.get('m0_first_proj_2025_01') or '2026 안 없음') if has_m0 else '—'} | {'**예**' if w['recompute_after_2025_01'] else '아니오'} |")
    a("")
    a("- H10·H11·H67의 '첫 값'이 늦거나 없는 것은 예열이 아니라 구조 결측(표 20개 미달)이다. 2025-01 자료로도 풀리지 않는다.")
    a("- 2025-01부터 다 차도 2026-01-02 전에 못 채우는 것은 H16 사건 뉴스 X⊥(투영 2026-02-02), 주 단계표(52주 순위, 2026-01-14), 자동 설명용인 H09·H11뿐이다. pct250 계열(단계표·H57·H43 사건)은 2025-07에 찬다. H35·H36 X⊥는 투영 2026-01-02로 앞 구간 첫날에 겨우 맞으므로(명절 위치에 따라 며칠 늦을 수 있음) 판정 때 실제 값으로 확인한다.")
    a("")
    # 7
    a("## 7. 분모 분포")
    a("| 대상 | 분모 | 값 | 판단 |")
    a("|---|---|---|---|")
    h9 = den["H09_kosdaq_tag_lower"]
    a(f"| H09 보조 코스닥 태그 | 태그 글의 표(점수 ≠ 0 작성자) | 하한판('코스닥' 낱말만): 창당 태그 글 중앙값 {num(h9['tag_posts_median'], 0)}, 표 중앙값 {num(h9['votes_median'], 0)}, 최대 {num(h9['votes_max'], 0)}, 표 20 이상 창 {pct(h9['share_votes_ge20'])} | 보조 표는 결측 100%(설명용). H09 자체도 자동 설명용 |")
    h23 = den["H23_claims"]
    a(f"| H23 수익률 주장 | 5창 합 주장 수(5 미만 결측) | M9+M10: 창 중앙값 {num(h23['M9+M10']['claims_per_window_median'], 0)}, 전체 {h23['M9+M10']['claims_total']:,}(큰 이익 {h23['M9+M10']['bw_total']}, 큰 손실 {h23['M9+M10']['bl_total']}), 2026 결측 {pct(h23['M9+M10']['miss_5win_lt5_2026'])}. M9만: 창 중앙값 {num(h23['M9']['claims_per_window_median'], 0)} | 결측 문턱 안. 부호 정확도 80%(100개) 점검은 아직 안 함(이미 설명용) |")
    h33 = den["H33_mc_authors"]
    a(f"| H33 MC 작성자 | 창 안 MC 고유 작성자 | 국내 중앙값 {num(h33['median'], 0)}, 3명 이상 창 {pct(h33['share_ge3'])}; 미국 중앙값 {num(den['H33_mc_authors_us']['median'], 0)}, {pct(den['H33_mc_authors_us']['share_ge3'])}. 신호 창 {h33['sig_windows']}/{h33['sig_computable']}(실시간 창 {h33['sig_windows_realtime']}) | 실시간 전방 트랙(신호 묶음 10개) |")
    h37 = den["H37_den"]
    a(f"| H37 약세 단골 | 5창 묶음의 글 쓴 약세 단골 | 2026 계산된 {h37['computed_2026']}창 중 5 이상 {pct(h37['ge5_of_computed_2026'])}(5 미만 {pct(h37['lt5_of_computed_2026'])}), 중앙값 {num(h37['median_2026'], 0)}. 달 중앙값: " + ", ".join(f"{k2} {num(v2, 0)}" for k2, v2 in h37["by_month_median"].items()) + " | 행 규칙(60%)·결측 30% 모두 통과. 다만 BULL·BEAR 측정 실패라 멈춤 |")
    h16 = den["H16_news"]
    a(f"| H16 뉴스형 | 창 안 사건 뉴스 글 | 2026 0개 창 {pct(h16['event_zero_share_2026'])}(전 기간 {pct(h16['event_zero_share_all'])}), 5개 미만 {pct(h16['event_lt5_share_2026'])}, 중앙값 {num(h16['event_median_2026'], 0)}. 가격 보도 0개 창 {pct(h16['price_zero_share_2026'])}. 미국 창 사건 뉴스 0개 {pct(h16['us_event_zero_share'])} | 30% 문턱 아래(중간 검증 33.3%에서 내려옴). E·F 측정 실패라 멈춤 |")
    t = den["H43_events"]["total_2026"]
    a(f"| H43 Herd 사건 | 직전 120창 상위 5%(현재 별칭 목록) | 2026 매수 쏠림 {t['buy']}·매도 쏠림 {t['sell']}(첫 값 국내 2026-02-20, 미국 2026-02-12: 120창 예열) | 30개 문턱은 넘음. CHASE 측정 실패라 멈춤. 2025-01 자료가 있으면 1~2월 사건도 셈 |")
    pp = den["pred_pairs"]
    a(f"| 예측 층(H66~H70) | 창당 예측 쌍(작성자당 1회, 60단위 표본) | 7/1~ {pp['late_windows']}창: 중앙값 {num(pp['late_median'], 0)}, 최소 {pp['late_min']}, 최대 {pp['late_max']}, 20개 이상 {pct(pp['late_share_ge20'])}, 5개 이상 {pct(pp['late_share_ge5'])}. 예측 있는 단위 몫 {pct(pp['pred_unit_share_late'])}. 강한 하락(1) 쌍 합 {pp['strong1_pairs_late_total']}, 0인 창 {pct(pp['strong1_zero_share_late'])} | H67 등록판 '표 20개'는 한 창도 못 채움 → 구조 결측 100%. H68은 값의 {pct(pp['strong1_zero_share_late'], 0)}가 0이라 정보가 얇다 |")
    a("")
    # 8
    a("## 8. 대표 특징끼리의 상관 (Pearson / Spearman, 판정 자격 창)")
    a("| 쌍 | 판 | 전 기간 | 2026 | 뒤 구간 | 문턱 | 판단 |")
    a("|---|---|---|---|---|---|---|")

    def crow(name, ver, key, thr, judge):
        c = cor[key]
        a(f"| {name} | {ver} | {cp(c['all'])} | {cp(c['y2026'])} | {cp(c['back'])} | {thr} | {judge} |")
    ex4 = exceeds(cor["H04_corr_U_N_rz8"], 0.9)
    crow("corr(U, N) H04", "rz8 등록판", "H04_corr_U_N_rz8", "0.9", "**넘음 → 자동 설명용**" if ex4 else "안 넘음")
    crow("", "시간당 원값", "H04_corr_U_N_raw_per_h", "", "참고")
    crow("", "미국 rz8", "H04_corr_U_N_rz8_us", "", "참고(보조)")
    crow("corr(SH, A) H54", "SH 원값 × A(rz8), 문자 그대로", "H54_corr_SH_raw_A_rz8", "0.8", "넘음" if exceeds(cor["H54_corr_SH_raw_A_rz8"], 0.8) else "안 넘음")
    crow("", "A6(rz8 SH) × A", "H54_corr_A6_A_rz8", "", "회귀에 같이 드는 꼴. 참고")
    crow("", "원값끼리", "H54_corr_SH_A_raw", "", "참고(요일·추세가 섞임)")
    crow("corr(K, A) H55", "K 원값 × A(rz8), 문자 그대로", "H55_corr_K_raw_A_rz8", "0.8", "넘음" if exceeds(cor["H55_corr_K_raw_A_rz8"], 0.8) else "안 넘음")
    crow("", "A7(rz8 K) × A", "H55_corr_A7_A_rz8", "", "참고")
    crow("corr(A4, A) H36", "등록판(NEW90 rz8 × A rz8)", "H36_corr_A4_A_rz8(NEW90)", "문서에 문턱 없음", "보고만(A4⊥가 A에 잔차화하는 설계)")
    crow("", "원값(NEW90 시간당 × A 원값)", "H36_corr_NEW90_A_raw", "", "참고")
    crow("corr(NW, A) H35", "NW90 rz8 × A rz8", "H35_corr_NW90rz8_A_rz8", "", "참고(H35는 A에 잔차화)")
    crow("D_SD와 B H11", "D_SD 원값 × B(표 20 이상 창)", "H11_corr_DSDraw_B(표20+)", "보고만", "")
    crow("", "D_SD 원값 × \\|B\\|", "H11_corr_DSDraw_absB(표20+)", "", "")
    crow("corr(A, 저장판 A) H03", "rz8", "H03_corr_A_Asaved_rz8", "부호 같아야 인정", "같음")
    crow("corr(Fight, ln 댓글 수) H08", "원값", "H08_corr_fight_lncomments", "", "참고")
    c2 = cor["H02_corr_Pbar_S_a"]
    a(f"| corr(TMP, S) H02 | TMP 자리 = 예측 층 P̄(부록 4) | 라벨 있는 전 창 {cp(c2['all_labeled'])} | 7/1~ {cp(c2['late'])} | — | 0.9 | 안 넘음. S는 측정 실패라 P̄를 쓴다 |")
    o = cor["H34_NS20_vs_LL"]
    for nm, lab in (("ll_d20", "LL 20창 변화(H31 특징)"), ("ll20", "LL 20창 평균")):
        x1, x2 = o.get(f"top15_overlap_{nm}_all"), o.get(f"top15_overlap_{nm}_y2026")
        a(f"| LL 신호와 NS20 겹침 H34 | 상위 15%끼리, {lab} | {x1['both']}/{x1['ns_on']} = {pct(x1['share_of_ns_on'])} | {x2['both']}/{x2['ns_on']} = {pct(x2['share_of_ns_on'])} | — | 70% | 안 넘음 |")
    a(f"| | 순위 상관 NS20 × LL 20창 변화 | {cp(o['corr_ns20_lld20'])} | | | | 참고(가격 잔차화 전 원값, 전 표본 순위라 설명용 점검값) |")
    a(f"| | 순위 상관 NS20 × LL 20창 평균 | {cp(o['corr_ns20_ll20'])} | | | | 참고 |")
    a(f"| 예측 층 참고 | corr(P_share rz8, A rz8) 7/1~ | {cp(cor['H70_corr_Pshare_rz8_A_rz8(late)'])} | | | — | H70과 H03이 같은 HAR-K에 들어가도 겹침 작음 |")
    a(f"| | corr(P̄, P_strong1) 7/1~ | {cp(cor['H66_H68_corr_Pbar_Pstrong1(late)'])} | | | — | H66·H68은 다른 칸 |")
    a("")
    # 9
    a("## 9. 가격만 쓰는 셈 — 이번에 안 함(판정 때)")
    a("- 판정 때 셀 것: 2025·2026 KOSPI·하닉 −2σ 사건 수(H24~H26·H28, 구간마다 8), 하락 상태 날 수(×0.90·×0.92, H26), 하락 조건 국면 수(H36 보조), ±3% 날 수(H29), §12.4 급락 시작(H31·H33). 중간 검증(10/14까지) 값은 2025 8·앞 3·뒤 5 사건이다.")
    a("")
    # 10
    a("## 10. 사전 정밀도 점검 → v0.2 지문")
    a("| 항목 | 값 |")
    a("|---|---|")
    a(f"| 묶음 | 36개 중 통과 {lexf['n_pass']}, 측정 실패 {lexf['n_fail']}(묶음 정밀도 60% 미만 또는 반어율 30% 초과) |")
    a(f"| 통과 묶음 | {', '.join(lexf['pass'])} |")
    a("| 실패 묶음 | " + ", ".join(f"{g} {pct(lexf['bundles'][g]['precision'], 0)}" for g in lexf["fail"]) + " |")
    a(f"| 뺀 단어 | {lexf['dropped_words']}개 |")
    lf = lexf["lexicon_fingerprints"]
    a(f"| v0.2 SHA-256 | `lexicon_v02.json` {lf['lexicon_v02.json']['fixed']}, `lexicon_v02_full.json` {lf['lexicon_v02_full.json']['fixed']} — final 사본·interim 원본 모두 같음 |")
    a(f"| 정밀도 표본 기간 | {lexf['precision_sample_period']}. 감정 묶음은 이미 실패라 판정에 영향이 없고, 통과 묶음(별칭·레버리지 상품·반대매매 등)은 고유명사라 시기 영향이 작다고 본다(확인은 안 함) |")
    a("")
    # 11
    a("## 11. 외부 자료 확보 (네트워크 미사용: 저장소 + 중간 검증 10/16 결과)")
    ei = ext["interim"]
    nq = ext.get("NQ_1h_preview_file", {})
    a("| 자료 | 쓰는 곳 | 결과 | 판단 |")
    a("|---|---|---|---|")
    a(f"| NQ=F 시간봉 | H10·H49·H50·H52 | 중간 검증: yfinance 60분봉 2025-01 받기 가능(730일 한도, 2025-01-02 봉은 2027-01-02 무렵 사라짐). 미리보기 파일: {nq.get('rows', '—')}봉, {nq.get('first_bar_et', '—')}~{nq.get('last_bar_et', '—')}(시각 열만 읽음) | 받기 가능. 판정 때 2025-08(또는 2025-01)~10/23을 한 번에 받아 고정 |")
    a("| KOSPI200 야간선물 | H51(있는 구간만) | 저장소에 원천 없음(검색 0건) | H51은 이 항 없이 진행 가능(이미 사전 측정 실패) |")
    a("| KRX·네이버 시가 | H51 | 중간 검증: 네이버 KOSDAQ 일봉 응답 200, yfinance 시가와 6/6일 0.1% 안 일치 | 받기 가능 |")
    kofia = ext["files"].get("금투협(KOFIA) 파일", [])
    a(f"| 금투협 공개 시각 | H30·H33 | 저장소 파일 {len(kofia)}개(2025-01~2026-10-13). 원천에 공개 시각 필드 없음 | 값은 있음. 실제 공개 시각은 소급 불가 → H30은 D−2 규칙, H33은 실시간 기록 필요 |")
    a("| KIND 청약 일정·상장 이름 사전 | H45·H39 국내·H09 보조 | 중간 검증: 화면·목록 내려받기 응답 200. 저장소에 아직 없음 | 경로 있음(H45·H39+H48은 이미 설명용) |")
    a(f"| 예탁원 해외주식 보관 상위 | H48 미국 유니버스 | 저장소 검색 {len(ext['repo_search']['예탁원 보관 상위'])}건. 경로 미확인 | 미확보 → H48 미국 부분 설명용(이미 H39+H48 전체 설명용) |")
    a(f"| 8월 단일종목 레버리지 규제 시행일 | H31 | 저장소에는 '8월 규제(기본예탁금 3,000만 원·신규 상장 중단)'만 있고 날짜가 없다({', '.join(ext['repo_search']['8월 레버리지 규제 시행일'])}) | **미확보 → H31 잠정 설명용**. 공식 공지로 날짜 하나만 확인하면 풀린다 |")
    vk = ext["files"].get("VKOSPI 수동표", [])
    a(f"| (참고) VKOSPI 이력 | M0 한국·HAR-K | yfinance 기호 없음(중간 검증). 저장소에는 수동 표뿐({', '.join(vk) or '없음'}) | M0·HAR-K 재료라 판정 전에 원천 필요. 못 받으면 M0에서 빼고 그 사실을 적는다 |")
    a("| 일정 더미 달력 | 모든 회귀 | FOMC: 저장소 stop_research 표. 실적일: 중간 검증에서 yfinance NVDA·MU·AVGO·하닉·삼전 받기 가능. CPI·고용(BLS 403)·MSCI·삼전 잠정실적 날짜 원천은 저장소에 없음 | CPI·고용 날짜는 다른 원천 필요(미리보기도 빠짐) |")
    a("")
    # 전 가설 표
    a("## 가설별 판정 전 상태(순차·실시간·측정)")
    a("| 묶음 | 가설 | 칸 | 상태 | 이유·메모 |")
    a("|---|---|---|---|---|")
    for r in cells:
        if r["group"] in ("T1", "T2", "F11"):
            continue
        txt = "; ".join(r["reasons"] + r["notes"]) or "—"
        a(f"| {r['group']} | {r['id']} | {r['name']} | **{r['status']}** | {txt} |")
    a("")
    a("- 원래 설명용 14개(H13·H20·H22·H23·H27·H28·H29·H30·H38·H45·H47·H57c·H59·H75)와 기각 7개는 판정 칸이 아니다. 실시간 전용 새 가설 H76~H78(부록 10)은 10/19 이후 자료로만 판정한다.")
    a("")
    a("## 계산하지 못한 것과 판정 때 할 일")
    a("| 대상 | 이유·할 일 |")
    a("|---|---|")
    a("| #9 가격만 쓰는 셈 | 이번 작업 범위 밖(가격 금지). 판정 때 사건·국면 수를 세고 H24~H26·H36·H29 최소 표본을 정한다 |")
    a("| X⊥·M0 R²·위약 X̂ | 가격 단계. 위 예열 표의 날짜부터 값이 생긴다. 2025-01 수집 완료 뒤 다시 계산 |")
    a("| 2025-01~08 포럼 자료 | 수집 중(2025-08-19 아래). 끝나면 같은 코드(`py run_features.py` → `py prechecks_full.py`)로 이 표를 다시 낸다 |")
    a("| 예측 층 X⊥에 7월 전 라벨을 쓸지 | 쓰면 H66·H68 X⊥가 2026-03부터, 안 쓰면 2026 안에 없음. 판정 때 정하고 표시한다(이 점검은 둘 다 냈다) |")
    a("| 8월 규제 시행일·VKOSPI·CPI·고용 날짜 | 웹 검색 금지 작업이라 확보하지 않았다 |")
    a("| 사전 정밀도 재측정(2025-08~2026-07 글) | 라벨 표본이 필요하다. 감정 묶음은 이미 실패라 판정에 영향 없음 |")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- 실행

def main() -> int:
    t0 = datetime.now().replace(microsecond=0)
    frozen_before = {n: sha(ANT / n) for n in FROZEN_SHA}
    cal = calendars()
    G = grids(cal)
    T = load_tables()
    kr = T["daily"]["kr"]
    m, linfo = load_pred_units()
    pred_a, pred_b, pmeta = build_pred(m, kr["judge_ok"].astype(bool))
    pred_a.reset_index().to_parquet(RES / "prechecks_full_pred_windows.parquet", index=False)

    feats = feature_rows(T, pred_a, cal)
    n_back_weeks = sum(1 for k in range((BACK[1] - BACK[0]).days + 1) if (BACK[0] + timedelta(days=k)).weekday() == 0)
    wrows = P.weekly_rows(T, n_back_weeks)
    warm = warmup_table(T, pred_a, feats, cal, G)
    for w in warm:   # 판정 칸 행에 X⊥ 필요 여부를 붙인다(예열 표 쪽 이름)
        f = next((f for f in feats if f["id"] == w["id"] and f["col"] == w["col"]), None)
        w["needs_M0"] = f["needs_M0"] if f else False

    units = pd.read_parquet(RES / "units.parquet")
    hits = pd.read_parquet(RES / "feat_topic_unit_hits.parquet", columns=["no", "cid", "utype", "excl", "score"])
    if not ((hits.no.to_numpy() == units.no.to_numpy()).all() and (hits.utype.to_numpy() == units.utype.to_numpy()).all()):
        raise RuntimeError("feat_topic_unit_hits와 units의 줄 순서가 다르다")
    hits.index = units.index
    c1 = check1()
    raw = raw_checks(units, hits["excl"])
    den = denominators(T, units, hits, pred_a)
    del units, hits
    cor = correlations(T, pred_a, pred_b)
    lexf = lexicon_and_frozen()
    ext = external()
    ver = verdicts(feats, wrows, cor, den, lexf, warm)

    frozen_after = {n: sha(ANT / n) for n in FROZEN_SHA}
    if frozen_after != frozen_before:
        raise RuntimeError("동결 문서 지문이 실행 중에 바뀌었다")
    out = {"generated": t0.isoformat(), "snapshot": {k: SNAP[k] for k in ("cutoff", "first", "S", "n_files", "n_lines")},
           "price_data_read": False, "price_cols_not_read": T["dropped_price_cols"],
           "periods": {"pre2025": [str(x) for x in PRE], "front": [str(x) for x in FRONT], "back": [str(x) for x in BACK],
                       "llm": [str(x) for x in LLM]},
           "pred_labels": linfo, "pred_build": pmeta,
           "check1": c1, "raw": raw, "features": feats, "weekly": wrows, "warmup": warm,
           "denominators": den, "correlations": cor, "lexicon_frozen": lexf, "external": ext, "verdicts": ver}
    out["cells"] = cell_table(out)
    (RES / "prechecks_full.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), "utf-8")
    (RES / "prechecks_full.md").write_text(render_md(json.loads(json.dumps(out, ensure_ascii=False, default=str))), "utf-8")
    flat = []
    for f in feats:
        g = {k: v for k, v in f.items() if not isinstance(v, dict)}
        for per in ("front", "back", "llm"):
            if per in f:
                for k, v in f[per].items():
                    g[f"{per}_{k}"] = v
        flat.append(g)
    pd.DataFrame(flat).to_csv(RES / "prechecks_full_features.csv", index=False, encoding="utf-8-sig")
    print("ok", (datetime.now() - t0).total_seconds(), "s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
