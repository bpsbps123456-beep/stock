"""부록 13 사건 연구, 확정 사건만 모드(부록 14) — 한 조합(자산 × 문턱)만 실행.

event_study.py의 함수를 그대로 부른다(분석 로직·문턱·귀무·위약·누출 규칙 불변). 바꾸는 것은 '대비를 한 조합으로 고르는 것'과
산출 위치뿐이다. 날짜 가드 대신 confirmed_only_events(2026-10-22 종가 판정)로 끝 극값·보류 가짜가 빠졌는지 확인한다.

  py -B run_combo_SKHYNIX_8.py                       # SK하이닉스 8% → results/run_SKHYNIX_8.json/.md, results/run_SKHYNIX_8/
  py -B run_combo_SKHYNIX_8.py --asset KOSPI --threshold 12

추가로 적는 것(설명용, 등록된 판정 규칙 아님)
  - 실시간성: 오프셋 0·+1 창 마감(K창 = 그 거래일 08:00 KST)이 사건 확정(확정일 15:30 장 마감) 전인지 사건마다 확인.
  - 예측 층 지식 기간: D < 2026-07-01 창은 라벨 모델 지식 기간이라, 예측 층 칸은 D ≥ 2026-07-01 사건·가짜만으로 BA를 다시 적는다.
  - 가격 메아리 점검: 앞섬 창(오프셋 −5~−1, 글 시각 D−6 08:00 ~ D−1 08:00 → 장 D−6~D−2)의 가격만으로 같은 '극값 닿음'을
    만들어(일간 수익률이 직전 60거래일 하위·상위 10%, |수익률| 상위 10%) 사건·가짜·위약 비율을 재고, 포럼 앞섬 비율을
    '가격 극값 있음/없음' 층으로 나눠 사건과 위약을 견준다.
  - 지금 위치: 가르기 후보 칸은 그 칸의 오프셋(0 또는 +1) 분포와 견준다(event_study.now_position은 오프셋 0 분포만 쓴다).
    앞섬 후보는 10/15~10/23 창을 가상의 사건일 D로 두고 직전 1~5창 극값 닿음을 적는다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import event_study as E  # noqa: E402

KNOW_END = pd.Timestamp("2026-07-01")
SHOW_OFFSETS = list(range(-5, 6))


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (pd.Timestamp, np.datetime64)):
        return str(o)
    return o


def f2(x, nd=2):
    return E.fmt(x, nd)


def window_slice(panel: E.Panel) -> np.ndarray:
    return np.flatnonzero((panel.D >= np.datetime64(E.NOW_FROM.date(), "D")) & (panel.D <= np.datetime64(E.NOW_TO.date(), "D")))


def realtime_check(c: E.Contrast, panel: E.Panel, F: pd.DataFrame, pivots: list[dict]) -> pd.DataFrame:
    """사건마다 오프셋 0·+1 창의 마감 시각과 확정 시각(확정일 15:30 KST)을 맞대 본다."""
    g = F[F["market"] == panel.market].sort_values("close_kst").reset_index(drop=True)
    close_kst = pd.to_datetime(g["close_kst"]).to_numpy()
    pk = "low" if c.kind == "bottom" else "high"
    conf = {p["date"]: p for p in pivots if p["kind"] == pk}
    rows = []
    for d, w in zip(c.events["date"], c.events["win_id"]):
        i = panel.pos.get(w, -1)
        p = conf.get(d, {})
        cd = p.get("confirm_date")
        conf_ts = pd.Timestamp(cd) + pd.Timedelta(hours=15, minutes=30) if cd else pd.NaT
        c0 = pd.Timestamp(close_kst[i]) if i >= 0 else pd.NaT
        c1 = pd.Timestamp(close_kst[i + 1]) if 0 <= i < len(close_kst) - 1 else pd.NaT
        rows.append({"date": d, "win_id": w, "win_plus1": panel.win_id[i + 1] if 0 <= i < len(close_kst) - 1 else None,
                     "close_off0": str(c0), "close_off1": str(c1), "confirm_date": cd,
                     "confirm_lag_td": p.get("confirm_lag_td"), "confirm_ts": str(conf_ts),
                     "off1_before_confirm": bool(c1 < conf_ts) if (pd.notna(c1) and pd.notna(conf_ts)) else None,
                     "amp_in": p.get("amp_in"), "amp_out": p.get("amp_out"),
                     "n_windows_before": p.get("n_windows_before")})
    return pd.DataFrame(rows)


def knowledge_sensitivity(r: dict, panel: E.Panel) -> list[dict]:
    """예측 층 칸: D ≥ 2026-07-01 사건·가짜 대표만으로 LOEO BA(같은 함수, 섞기 1,000번)."""
    c = r["contrast"]
    ev_pos = E.positions(panel, c.events["win_id"])
    fk_pos = E.positions(panel, c.fake_reps["win_id"])
    late = np.datetime64(KNOW_END.date(), "D")
    ev_late = np.array([p for p in ev_pos if p >= 0 and panel.D[p] >= late], int)
    fk_late = np.array([p for p in fk_pos if p >= 0 and panel.D[p] >= late], int)
    feats = [f for f in panel.feats if panel.meta[f]["bundle"] == "예측 층"]
    if len(ev_late) < E.MIN_TRUE or len(fk_late) < E.MIN_FAKE:
        return [{"note": f"D ≥ {KNOW_END.date()} 사건 {len(ev_late)}개·가짜 대표 {len(fk_late)}개 — 표본 부족(각 {E.MIN_TRUE}개 이상)",
                 "n_events": int(len(ev_late)), "n_fakes": int(len(fk_late))}]
    idx = [panel.feats.index(f) for f in feats]
    sub = E.Panel(panel.market, panel.win_id, panel.D, feats, panel.X[:, idx], panel.Z[:, idx], panel.PCT[:, idx],
                  panel.meta, panel.realtime)
    s = int(hashlib.sha256(f"{E.SEED}:{c.name}:late".encode()).hexdigest()[:8], 16)
    cls = E.classify(sub, ev_late, fk_late, n_null=E.N_NULL, seed=s)
    return cls["table"].to_dict("records")


def price_lead_indicators(close: pd.Series, dates, kind: str) -> np.ndarray:
    """(n, 3): [방향 극값, 크기 극값, 앞섬 창 수익률]. 장 D−6~D−2(앞섬 창 글 시각에 든 장)의 일간 수익률을 직전 60거래일(40개 이상)
    분포에 놓고, 바닥 대비 = 하위 10% 닿음 / 고점 대비 = 상위 10% 닿음, 크기 = |수익률| 상위 10% 닿음. 값 3개 미만이면 결측."""
    d = close.index.to_numpy("datetime64[D]")
    c = close.to_numpy(float)
    pos_of = {x: i for i, x in enumerate(d)}
    r1 = np.full(len(c), np.nan)
    r1[1:] = c[1:] / c[:-1] - 1
    pr, pa = E.pct_prev(r1, E.P_N, E.P_MIN), E.pct_prev(np.abs(r1), E.P_N, E.P_MIN)
    out = np.full((len(dates), 3), np.nan)
    for k, x in enumerate(dates):
        i = pos_of.get(np.datetime64(x, "D")) if isinstance(x, str) else None
        if i is None or i - 7 < 0:
            continue
        a, b = pr[i - 6:i - 1], pa[i - 6:i - 1]
        if np.isfinite(a).sum() < E.LEAD_MIN:
            continue
        out[k, 0] = float((a <= E.P_LO).any()) if kind == "bottom" else float((a >= E.P_HI).any())
        out[k, 1] = float((b >= E.P_HI).any())
        out[k, 2] = c[i - 2] / c[i - 7] - 1
    return out


def price_echo_check(r: dict, panel: E.Panel, close: pd.Series) -> dict:
    """설명용(등록 밖). ① 가격만 앞섬 기준선 ② 포럼 앞섬 비율을 가격 극값 층으로 나눠 사건 대 가짜 대 위약."""
    from scipy.stats import binom
    c = r["contrast"]
    ev_pos = E.positions(panel, c.events["win_id"])
    fk_pos = E.positions(panel, c.fake_reps["win_id"])
    pp = E.pool_positions(r["pools"])

    def dates(pos):
        return [str(panel.D[i]) if i >= 0 else None for i in pos]
    PE, PF, PP = (price_lead_indicators(close, dates(x), c.kind) for x in (ev_pos, fk_pos, pp))
    rng = np.random.default_rng(int(hashlib.sha256(f"{E.SEED}:{c.name}:price".encode()).hexdigest()[:8], 16))
    names = {0: "방향 극값(" + ("하위" if c.kind == "bottom" else "상위") + " 10% 일간 수익률)", 1: "크기 극값(|수익률| 상위 10%)"}
    base = {}
    for k, nm in names.items():
        e, f, pl = PE[:, k], PF[:, k], PP[:, k]
        e, f, pl = e[np.isfinite(e)], f[np.isfinite(f)], pl[np.isfinite(pl)]
        x = np.r_[e, f]
        y = np.r_[np.ones(len(e), bool), np.zeros(len(f), bool)]
        Yp = E.perm_labels(y, E.N_NULL, rng)
        with np.errstate(invalid="ignore", divide="ignore"):
            null = (Yp * x).sum(1) / Yp.sum(1) - ((~Yp) * x).sum(1) / (~Yp).sum(1)
        ph = float(pl.mean()) if len(pl) else np.nan
        base[nm] = {"n_events": int(len(e)), "n_fakes": int(len(f)), "n_placebo_days": int(len(pl)),
                    "rate_events": float(e.mean()) if len(e) else np.nan, "rate_fakes": float(f.mean()) if len(f) else np.nan,
                    "rate_placebo": ph,
                    "p_vs_fakes": E.p_upper(float(e.mean() - f.mean()), null) if (len(e) and len(f)) else np.nan,
                    "p_vs_placebo": float(binom.sf(int(e.sum()) - 1, len(e), ph)) if (len(e) and np.isfinite(ph)) else np.nan}
    lt = r["lead"]
    if lt["candidate"].any():
        cand = lt[lt["candidate"]]
    else:
        cand = lt[(lt["status"] == "시험") & lt["status_feature"].eq("통과")].nsmallest(3, "p")
    hiE, loE = E.lead_hits(panel, ev_pos)
    hiF, loF = E.lead_hits(panel, fk_pos)
    hiP, loP = E.lead_hits(panel, pp)
    strata = []
    for f, side in zip(cand["feature"], cand["side"]):
        j = panel.feats.index(f)
        hE, hF, hP = ((h[:, j] if side == "상위 10%" else l_[:, j]) for h, l_ in ((hiE, loE), (hiF, loF), (hiP, loP)))
        row = {"feature": f, "side": side,
               "candidate": bool(lt.loc[(lt["feature"] == f) & (lt["side"] == side), "candidate"].any())}
        for k, nm in ((0, "dir"), (1, "abs")):
            for sv in (1.0, 0.0):
                mE = np.isfinite(hE) & (PE[:, k] == sv)
                mF = np.isfinite(hF) & (PF[:, k] == sv)
                mP = np.isfinite(hP) & (PP[:, k] == sv)
                row[f"{nm}{int(sv)}"] = {"ev_rate": float(hE[mE].mean()) if mE.any() else np.nan, "ev_n": int(mE.sum()),
                                         "fk_rate": float(hF[mF].mean()) if mF.any() else np.nan, "fk_n": int(mF.sum()),
                                         "pl_rate": float(hP[mP].mean()) if mP.any() else np.nan, "pl_n": int(mP.sum())}
        strata.append(row)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ret = {"events": float(np.nanmean(PE[:, 2])), "fakes": float(np.nanmean(PF[:, 2])), "placebo": float(np.nanmean(PP[:, 2]))}
    return {"price_only_lead": base, "forum_lead_by_price_stratum": strata, "lead_window_return_mean": ret,
            "rule": "장 D−6~D−2(앞섬 창 글 시각에 든 장) 일간 수익률의 직전 60거래일 백분위. 등록 밖 설명용 점검"}


def now_candidates(r: dict, panel: E.Panel) -> list[dict]:
    """가르기 후보 칸(특징@오프셋): 10/15~10/23 창 z를 같은 오프셋의 사건·가짜·위약 z 분포에 놓는다."""
    c, tb = r["contrast"], r["table"]
    cand = tb[tb["candidate"]]
    if not len(cand):
        return []
    sel = window_slice(panel)
    ev_pos = E.positions(panel, c.events["win_id"])
    fk_pos = E.positions(panel, c.fake_reps["win_id"])
    pp = np.unique(E.pool_positions(r["pools"]))
    out = []
    for f, o in zip(cand["feature"], cand["offset"]):
        j, o = panel.feats.index(f), int(o)
        refs = {k: E.gather(panel.Z, x, [o])[:, 0, j] for k, x in (("events", ev_pos), ("fakes", fk_pos), ("placebo", pp))}
        refs = {k: v[np.isfinite(v)] for k, v in refs.items()}
        x = tb[(tb["feature"] == f) & (tb["offset"] == o)].iloc[0]
        rows = []
        for t in sel:
            z = panel.Z[t, j]

            def pr(ref):
                return float(((ref < z).sum() + 0.5 * (ref == z).sum()) / len(ref)) if (len(ref) and np.isfinite(z)) else np.nan
            rows.append({"win_id": panel.win_id[t], "hypothetical_event_day": str(panel.D[t - o]) if t - o >= 0 else None,
                         "z": z, "pct_in_events": pr(refs["events"]), "pct_in_fakes": pr(refs["fakes"]),
                         "pct_in_placebo": pr(refs["placebo"])})
        out.append({"feature": f, "offset": o, "direction": x["direction"], "mean_z_events": x["mean_z_events"],
                    "mean_z_fakes": x["mean_z_fakes"], "n_ref": {k: int(len(v)) for k, v in refs.items()}, "windows": rows})
    return out


def now_lead(r: dict, panel: E.Panel) -> list[dict]:
    """앞섬 후보: 10/15~10/23 창을 가상의 사건일 D로 두고 직전 1~5창(오프셋 −5~−1)의 극값 닿음."""
    lt = r["lead"]
    cand = lt[lt["candidate"]]
    if not len(cand):
        return []
    sel = window_slice(panel)
    hi, lo = E.lead_hits(panel, sel)
    P_all = E.gather(panel.PCT, sel, np.arange(E.LEAD[0], E.LEAD[1] + 1))
    out = []
    for f, side in zip(cand["feature"], cand["side"]):
        j = panel.feats.index(f)
        h = hi[:, j] if side == "상위 10%" else lo[:, j]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ext = np.nanmax(P_all[:, :, j], 1) if side == "상위 10%" else np.nanmin(P_all[:, :, j], 1)
        x = lt[(lt["feature"] == f) & (lt["side"] == side)].iloc[0]
        out.append({"feature": f, "side": side, "rate_events": x["rate_events"], "rate_fakes": x["rate_fakes"],
                    "rate_placebo_mean": x.get("rate_placebo_mean"),
                    "windows": [{"win_id": panel.win_id[t], "lead_windows": f"{panel.win_id[max(t - 5, 0)]}~{panel.win_id[t - 1]}",
                                 "hit": h[k], "extreme_pct": ext[k], "pct_now": panel.PCT[t, j]} for k, t in enumerate(sel)]})
    return out


def loeo_hits(v: np.ndarray, y: np.ndarray) -> tuple[int, int]:
    """E.loeo_ba_naive와 같은 규칙(하나 빼고 문턱·방향, 같은 점수면 위쪽·낮은 문턱)으로 빠진 표본을 맞힌 개수(사건, 가짜)."""
    v, y = np.asarray(v, float), np.asarray(y, bool)
    pred = np.zeros(len(v), bool)
    for i in range(len(v)):
        te = np.zeros(len(v), bool)
        te[i] = True
        vt, yt = v[~te], y[~te]
        s = np.sort(vt)
        cuts = [-np.inf] + [(s[k - 1] + s[k]) / 2 for k in range(1, len(s)) if s[k] > s[k - 1]] + [np.inf]
        best, choice = -np.inf, None
        for d in ("up", "dn"):
            for c in cuts:
                pr = vt > c if d == "up" else ~(vt > c)
                ba = 0.5 * ((pr & yt).sum() / yt.sum() + (~pr & ~yt).sum() / (~yt).sum())
                if ba > best + 1e-12:
                    best, choice = ba, (d, c)
        d, c = choice
        pred[i] = (v[i] > c) if d == "up" else not (v[i] > c)
    return int((pred & y).sum()), int((~pred & ~y).sum())


def contrast_block(r: dict, panel: E.Panel, F: pd.DataFrame, pivots: list[dict], close: pd.Series) -> dict:
    c = r["contrast"]
    tb = r["table"].copy()
    lt = r["lead"].copy()
    ev_pos = E.positions(panel, c.events["win_id"])
    fk_pos = E.positions(panel, c.fake_reps["win_id"])
    rt = realtime_check(c, panel, F, pivots)
    pp = r["price_paths"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        price = {k: {int(o): float(np.nanmean(v[:, j]) * 100) if len(v) else None
                     for j, o in enumerate(E.OFFSETS) if o in SHOW_OFFSETS} for k, v in pp.items()}
    n_z0 = int(np.isfinite(E.gather(panel.Z, ev_pos, [0])[:, 0, :]).any(1).sum())
    split = []
    late = np.datetime64(KNOW_END.date(), "D")
    for f, o in zip(tb.loc[tb["candidate"], "feature"], tb.loc[tb["candidate"], "offset"]):
        j, o = panel.feats.index(f), int(o)
        row = {"feature": f, "offset": o}
        for role, pos in (("events", ev_pos), ("fakes", fk_pos)):
            v = E.gather(panel.Z, pos, [o])[:, 0, j]
            dd = np.array([panel.D[p] if p >= 0 else np.datetime64("NaT") for p in pos], "datetime64[D]")
            for per, m in (("early", dd < late), ("late", dd >= late)):
                vv = v[m & np.isfinite(v)]
                row[f"{role}_{per}"] = {"mean_z": float(vv.mean()) if len(vv) else np.nan, "n": int(len(vv))}
        split.append(row)
    tb["hit_events"], tb["hit_fakes"] = np.nan, np.nan
    for ix in tb.index[tb["candidate"]]:
        j, o = panel.feats.index(tb.at[ix, "feature"]), int(tb.at[ix, "offset"])
        v = np.r_[E.gather(panel.Z, ev_pos, [o])[:, 0, j], E.gather(panel.Z, fk_pos, [o])[:, 0, j]]
        y = np.r_[np.ones(len(ev_pos), bool), np.zeros(len(fk_pos), bool)]
        ok = np.isfinite(v)
        he, hf = loeo_hits(v[ok], y[ok])
        assert abs(0.5 * (he / y[ok].sum() + hf / (~y[ok]).sum()) - tb.at[ix, "ba"]) < 1e-9   # loeo_ba와 같은 값
        tb.at[ix, "hit_events"], tb.at[ix, "hit_fakes"] = he, hf
    return {"name": c.name, "asset": c.asset, "threshold": c.threshold, "kind": c.kind,
            "events": c.events.assign(pos=ev_pos).to_dict("records"),
            "fake_reps": c.fake_reps.assign(pos=fk_pos).to_dict("records"),
            "fakes_all_n": int(len(c.fakes_all)), "info": c.info, "n": r["n"], "events_with_any_z0": n_z0,
            "family": r["family"], "lead_family": r["lead_family"], "leak": r["leak"],
            "outside_band": r["outside_band"], "placebo": r["placebo"],
            "tests": tb.to_dict("records"), "lead": lt.to_dict("records"),
            "candidates": tb[tb["candidate"]].to_dict("records"),
            "lead_candidates": lt[lt["candidate"]].to_dict("records"),
            "realtime_check": rt.to_dict("records"),
            "price_path_mean_pct": price,
            "knowledge_sensitivity_pred_layer": knowledge_sensitivity(r, panel),
            "price_echo_check": price_echo_check(r, panel, close), "candidate_period_split": split,
            "now_candidates": now_candidates(r, panel), "now_lead": now_lead(r, panel)}


def other_threshold_overlap(blocks: list[dict]) -> dict | None:
    """같은 자산의 다른 문턱 결과(results/run_<자산>_<문턱>.json, 다른 실행이 만든 파일)가 있으면 후보가 겹치는 칸을 적는다(읽기만)."""
    a, X = blocks[0]["asset"], blocks[0]["threshold"]
    other = {"8": "12", "12": "8"}.get(X)
    p = E.RESULTS / f"run_{a}_{other}.json"
    if not other or not p.exists():
        return None
    try:
        s = json.loads(p.read_text("utf-8"))
        cs = s["contrasts"]
        by_kind = {b["kind"]: b for b in cs} if isinstance(cs, list) else cs
        out = {"file": p.name, "threshold": other, "sha256": E.sha(p), "both": [], "only_here": [], "only_other": []}
        for b in blocks:
            o = by_kind.get(b["kind"], {})
            mine = {("가르기", x["feature"], f"{int(x['offset']):+d}") for x in b["candidates"]} | \
                   {("앞섬", x["feature"], x["side"]) for x in b["lead_candidates"]}
            theirs = {("가르기", x["feature"], f"{int(x['offset']):+d}") for x in o.get("candidates_cls", o.get("candidates", []))} | \
                     {("앞섬", x["feature"], x["side"]) for x in o.get("lead_candidates", [])}
            for key, sset in (("both", mine & theirs), ("only_here", mine - theirs), ("only_other", theirs - mine)):
                out[key] += [f"{E.TYPE_KO[b['kind']]} {k[0]} {k[1]}({k[2]})" for k in sorted(sset)]
        return out
    except Exception as e:  # noqa: BLE001  (다른 실행의 파일 꼴이 다르면 건너뛴다)
        return {"file": p.name, "error": repr(e)}


def md_report(blocks: list[dict], mode: dict, meta_run: dict, overlap: dict | None = None) -> str:
    L = []
    a, X = blocks[0]["asset"], blocks[0]["threshold"]
    ko = E.ASSET_KO[a]
    tag = f"{a}_{X}"
    L.append(f"# 사건 연구 결과: {ko} {X}% (부록 13, 확정 사건만 모드 — 탐색, 결과는 '후보')\n")
    L.append(f"> {meta_run['made']} 실행. `event_study.py`(sha256 {meta_run['event_study_sha'][:12]}…) 함수를 그대로 불렀다"
             f"(`code/run_combo_{tag}.py`, 분석·문턱·귀무 불변). 사건 `events.json`(sha256 {meta_run['events_sha'][:12]}…, "
             f"{mode['asof']} 종가 판정), 특징 `features_all.parquet`(sha256 {meta_run['features_sha'][:12]}…, 스냅샷 2026-10-23 "
             f"07:59 KST까지, W_KR {meta_run['n_windows']}창). 특징 격자 {meta_run['n_features']}개(통과 {meta_run['n_pass']} · 설명용 "
             f"{meta_run['n_expl']}) × 오프셋 2개(0, +1) = {2 * meta_run['n_features']}칸(통과 {2 * meta_run['n_pass']}칸). 섞기 "
             f"{E.N_NULL}번, 위약 = 사건마다 직전 5·20거래일 수익률이 가장 비슷한 날 {E.PLACEBO_POOL}일. 매매 규칙이 아니며 "
             f"'지금이 바닥/고점'이라는 판단이 아니다.\n")
    t = mode["contrasts"][tag]
    te = t["tail_excluded"] or {}
    if te:
        L.append(f"> 뺀 것(미확정): 끝 극값 {'저점' if te.get('kind') == 'low' else '고점'} {te.get('date')}(창 {te.get('win_id')}, "
                 f"종가 {te.get('close'):,.0f}, 확정에 필요한 종가 {te.get('confirm_level'):,.0f} "
                 f"{'이상' if te.get('kind') == 'low' else '이하'}, 10/22 종가 {te.get('last_close'):,.0f}). 같이 뺀 그 ±3거래일 보류 가짜 바닥: "
                 f"{', '.join(t['held_fakes_excluded'].get('fake_bottoms', [])) or '없음'}. 쓴 사건의 마지막 확정일 "
                 f"{t['last_confirm_date']}. 끝 극값은 10/23 종가 뒤 다시 판정한다.\n")

    L.append("## 0. 결론\n")
    L.append("| 대비 | 사건(z 있음/전체) | 가짜 묶음 대표 | 가르기 칸(통과) | p<0.05 관측 | 우연 기대 | 귀무 95% | 묶음 p | 누출 시험 | 후보(가르기) | 앞섬 p<0.05 관측 | 앞섬 우연 기대 | 앞섬 귀무 95% | 앞섬 묶음 p | 후보(앞섬) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for b in blocks:
        fm, lf = b["family"]["통과"], b["lead_family"]["통과"]
        cs = ", ".join(f"{x['feature']}@{x['offset']:+d}" for x in b["candidates"]) or "없음"
        ls = ", ".join(f"{x['feature']}({x['side']})" for x in b["lead_candidates"]) or "없음"
        L.append(f"| {E.TYPE_KO[b['kind']]} 대 가짜 {E.TYPE_KO[b['kind']]} | {b['events_with_any_z0']}/{b['n']['events_total']} | "
                 f"{b['n']['fake_reps_with_window']}(가짜 날 {b['fakes_all_n']}) | {fm['n_tests']} | {fm['observed']} | "
                 f"{f2(fm['expected_by_chance'])} | {f2(fm['null_q95'], 0)} | {f2(fm['p_family'], 3)} | "
                 f"{'의심' if b['leak']['flag'] else '통과'} | {cs} | {lf['observed']} | {f2(lf['expected_by_chance'])} | "
                 f"{f2(lf['null_q95'], 0)} | {f2(lf['p_family'], 3)} | {ls} |")
    L.append("")
    for b in blocks:
        fm_all, fm_ex = b["family"]["전체"], b["family"]["설명용"]
        la, le = b["lead_family"]["전체"], b["lead_family"]["설명용"]
        L.append(f"- {E.TYPE_KO[b['kind']]} 가르기: 설명용 {fm_ex['n_tests']}칸 중 p<0.05 {fm_ex['observed']}개(우연 기대 "
                 f"{f2(fm_ex['expected_by_chance'])}), 전체 {fm_all['n_tests']}칸 중 {fm_all['observed']}개(우연 기대 "
                 f"{f2(fm_all['expected_by_chance'])}, 귀무 95% {f2(fm_all['null_q95'], 0)}, 묶음 p {f2(fm_all['p_family'], 3)}). "
                 f"앞섬: 설명용 {le['n_tests']}칸 중 {le['observed']}개(기대 {f2(le['expected_by_chance'])}), 전체 {la['n_tests']}칸 중 "
                 f"{la['observed']}개(기대 {f2(la['expected_by_chance'])}, 묶음 p {f2(la['p_family'], 3)}).")
    L.append("- '우연 기대' = 사건·가짜 표지를 섞은 1,000번에서 p<0.05가 나온 칸 수의 평균(모든 특징이 같은 섞기 → 특징 사이 상관 유지). "
             "관측이 귀무 95% 이하이면 그 대비의 p<0.05 칸들은 우연과 구별되지 않는다. 후보 = p<0.05 + 검수 '통과' + 누출 의심 아님"
             "(앞섬은 + 위약 97.5% 분위 위).\n")

    L.append("## 1. 사건과 실시간성\n")
    for b in blocks:
        rt = pd.DataFrame(b["realtime_check"])
        lag = rt["confirm_lag_td"].dropna()
        ok1 = rt["off1_before_confirm"].dropna()
        L.append(f"- {E.TYPE_KO[b['kind']]} {len(rt)}개: " + ", ".join(rt["date"]) + ".")
        L.append(f"  - 확정까지 걸린 거래일: 중앙 {lag.median():.0f}, 범위 {lag.min():.0f}~{lag.max():.0f}. 오프셋 +1 창(사건 다음 거래일 "
                 f"08:00 마감)이 확정(확정일 15:30 장 마감) 전인 사건 {int(ok1.sum())}/{len(ok1)}개 → 오프셋 0·+1 특징은 '사건일 다음 "
                 f"날 아침까지 쓴 글'이고 확정 정보가 섞이지 않는다. 그 시각에는 그 날이 극값인지 아직 모른다(확정은 "
                 f"{lag.min():.0f}~{lag.max():.0f}거래일 뒤).")
        L.append(f"  - 가짜 {E.TYPE_KO[b['kind']]}: 가짜 날 {b['fakes_all_n']}개, 묶음 {b['info']['n_fake_runs']}개 → 묶음 대표 "
                 f"{len(b['fake_reps'])}개(" + ", ".join(x["date"] for x in b["fake_reps"]) + ").")
    L.append("- 첫 바닥 2025-08-21은 자료 첫 창이라 z가 없다(z는 직전 20창 중 15창, 앞섬 백분위는 직전 60창 중 40창이 있어야 계산). "
             "오프셋 +2 이후 그림은 사건 뒤에 쓴 글이라 '알아챔' 근거로 쓰지 않는다(설명용).\n")

    L.append("## 2. 가르기 (하나 빼고 맞히기, 균형 정확도 BA)\n")
    for b in blocks:
        tb = pd.DataFrame(b["tests"])
        tt = tb[tb["status"] == "시험"].sort_values(["p", "ba"], ascending=[True, False])
        L.append(f"### {E.TYPE_KO[b['kind']]} 대 가짜 {E.TYPE_KO[b['kind']]} — p 낮은 순 10칸 (전 칸 `run_{tag}/tests.csv`)\n")
        L.append("| 특징 | 묶음·지위 | 오프셋 | 사건/가짜 n | BA(LOEO) | p(섞기) | 귀무 BA 95% | 방향 | 사건 z 평균 | 가짜 z 평균 | 가짜 날 전부 BA | 누출 의심 | 후보 |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for x in tt.head(10).to_dict("records"):
            L.append(f"| {x['feature']} | {x['bundle']}·{x['status_feature']} | {x['offset']:+d} | {x['n_events']}/{x['n_fakes']} | "
                     f"{f2(x['ba'])} | {f2(x['p'], 3)} | {f2(x.get('null_ba_q95'))} | {x.get('direction', '')} | "
                     f"{f2(x.get('mean_z_events'))} | {f2(x.get('mean_z_fakes'))} | {f2(x.get('ba_all_fake_days_run_groups'))} | "
                     f"{'예' if x['leak_suspect'] else ''} | {'예' if x['candidate'] else ''} |")
        if b["candidates"]:
            L.append("\n- 후보 칸 하나 빼고 맞힌 개수: " + "; ".join(
                f"{x['feature']}@{x['offset']:+d} 사건 {int(x['hit_events'])}/{x['n_events']}, 가짜 {int(x['hit_fakes'])}/{x['n_fakes']}"
                for x in b["candidates"]) + ".")
        short = tb[tb["status"] != "시험"]
        if len(short):
            L.append(f"\n- 표본 부족(사건·가짜 각 {E.MIN_TRUE}개 미만) 칸 {len(short)}개: "
                     + ", ".join(sorted({f"{f}@{o:+d}" for f, o in zip(short['feature'], short['offset'])})) + ".")
        L.append("")
    L.append("- '가짜 날 전부 BA' = 묶음 대표 대신 가짜 날 전부를 넣고 묶음째 빼며 잰 보조 BA. 대표 BA보다 크게 낮으면 그 칸은 "
             "대표 고르기에 민감하다.\n")

    L.append("## 3. 앞섬 (사건 전 1~5창에 직전 60창 상위·하위 10% 극값 닿음)\n")
    for b in blocks:
        lt = pd.DataFrame(b["lead"])
        lt = lt[lt["status"] == "시험"].sort_values("p")
        L.append(f"### {E.TYPE_KO[b['kind']]} — p 낮은 순 8칸 (전 칸 `run_{tag}/lead.csv`)\n")
        L.append("| 특징 | 지위 | 쪽 | 사건/가짜 n | 사건 비율 | 가짜 비율 | 위약 비율 | 위약 97.5% | p(대 가짜) | p(대 위약) | 후보 |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for x in lt.head(8).to_dict("records"):
            L.append(f"| {x['feature']} | {x['status_feature']} | {x['side']} | {x['n_events']}/{x['n_fakes']} | "
                     f"{f2(x['rate_events'])} | {f2(x['rate_fakes'])} | {f2(x.get('rate_placebo_mean'))} | "
                     f"{f2(x.get('rate_placebo_q975'))} | {f2(x['p'], 3)} | {f2(x.get('p_vs_placebo'), 3)} | "
                     f"{'예' if x['candidate'] else ''} |")
        L.append("")

    L.append("### 3-1. 가격 메아리 점검 (설명용, 등록 밖)\n")
    L.append("앞섬 창 글(D−6 08:00 ~ D−1 08:00)은 장 D−6~D−2를 보고 쓴 글이다. 같은 장의 가격만으로 '극값 닿음'을 만들면 사건·가짜·위약이 "
             "어떻게 갈리는지, 그리고 포럼 앞섬 비율이 가격 극값이 있던 날끼리·없던 날끼리도 사건과 위약 사이에 차이가 남는지 본다.\n")
    for b in blocks:
        pe = b["price_echo_check"]
        lr = pe["lead_window_return_mean"]
        L.append(f"**{E.TYPE_KO[b['kind']]}** — 앞섬 창 장(D−6~D−2) 누적 수익률 평균: 사건 {f2(lr['events'] * 100)}%, 가짜 "
                 f"{f2(lr['fakes'] * 100)}%, 위약 {f2(lr['placebo'] * 100)}%.\n")
        L.append("| 가격만 지표 | 사건 비율(n) | 가짜 비율(n) | 위약 비율(날 수) | p(대 가짜) | p(대 위약) |")
        L.append("|---|---|---|---|---|---|")
        for nm, v in pe["price_only_lead"].items():
            L.append(f"| {nm} | {f2(v['rate_events'])}({v['n_events']}) | {f2(v['rate_fakes'])}({v['n_fakes']}) | "
                     f"{f2(v['rate_placebo'])}({v['n_placebo_days']}) | {f2(v['p_vs_fakes'], 3)} | {f2(v['p_vs_placebo'], 3)} |")
        L.append("")
        L.append("| 포럼 앞섬 특징(쪽) | 후보 | 가격 방향 극값 있음: 사건 / 가짜 / 위약 | 없음: 사건 / 가짜 / 위약 | 가격 크기 극값 있음: 사건 / 가짜 / 위약 | 없음: 사건 / 가짜 / 위약 |")
        L.append("|---|---|---|---|---|---|")
        for x in pe["forum_lead_by_price_stratum"]:
            def cell(s):
                return (f"{f2(s['ev_rate'])}({s['ev_n']}) / {f2(s['fk_rate'])}({s['fk_n']}) / {f2(s['pl_rate'])}({s['pl_n']})")
            L.append(f"| {x['feature']}({x['side']}) | {'예' if x['candidate'] else '아님'} | {cell(x['dir1'])} | {cell(x['dir0'])} | "
                     f"{cell(x['abs1'])} | {cell(x['abs0'])} |")
        L.append("")
    L.append("- 위약(직전 5·20거래일 수익률 최근접)은 앞섬 창 가격까지 맞추지 못한다. 층 안에서도 사건이 위약보다 높으면 '가격만으로는 "
             "설명 안 되는 몫'의 후보, 층 안에서 비슷하면 가격 움직임의 메아리로 읽는다. 층마다 사건 수가 작아(괄호) 설명용이다.\n")

    L.append("## 4. 조회·추천 누출 시험 ((b) 조회 ÷ (마감 전 댓글 + 1) 창 중앙값의 z)\n")
    for b in blocks:
        lk = b["leak"]
        parts = []
        for k, v in lk["by_offset"].items():
            if "median_events" in v:
                parts.append(f"오프셋 {int(k):+d}: 사건 중앙 {f2(v['median_events'])}(n {v['n_events']}) vs 가짜 "
                             f"{f2(v['median_fakes'])}(n {v['n_fakes']}), 사건 > 가짜 {'예' if v['events_greater'] else '아니오'}, "
                             f"MW p {f2(v.get('mannwhitney_p_greater'), 3)}")
            else:
                parts.append(f"오프셋 {int(k):+d}: {v.get('result')}(사건 {v['n_events']}, 가짜 {v['n_fakes']})")
        L.append(f"- {E.TYPE_KO[b['kind']]}: **{'누출 의심' if lk['flag'] else '통과'}** — " + "; ".join(parts) + ".")
    L.append("- 규칙(부록 13·event_study): 오프셋 0·+1 중 하나라도 사건 중앙값 > 가짜 중앙값이거나 값이 모자라면 그 대비의 조회·추천 특징 "
             "전부를 후보에서 뺀다. 점추정 규칙이라 신호가 없어도 0.65~0.70 켜지는 보수적 규칙이다(시운전 3절). 조회 원값 특징은 소급 창"
             "(실시간 글 5% 미만)끼리만 비교하므로 10/13 뒤 실시간 창은 시험·후보·지금 위치에 들지 않는다.")
    L.append("- 실시간성: 소급 창의 조회·추천 값은 수집 시점(사건 뒤 몇 주~몇 달)의 누적값이다. 누출 시험을 통과해도 사건 다음 날 아침 "
             "07:59에 같은 값을 잴 수는 없으므로, 조회·추천 후보(`vw_*`)는 '그 시각에 알아챌 수 있었다'는 근거가 아니다. 실시간에 쓰려면 "
             "10/13 뒤 실시간 수집값(c)으로 다시 재야 한다.\n")

    L.append("## 5. 사건 전후 그림 (설명용)\n")
    for b in blocks:
        ob = b["outside_band"]
        pm = b["price_path_mean_pct"]
        L.append(f"- {E.TYPE_KO[b['kind']]}: 위약 띠 밖 특징 수 — 오프셋 0: {ob['0']['outside_band']}/{ob['0']['n_features']}"
                 f"(독립이면 기대 {ob['0']['expected_if_independent']}), +1: {ob['1']['outside_band']}/{ob['1']['n_features']}"
                 f"(기대 {ob['1']['expected_if_independent']}). 띠 밖: 0 = {', '.join(ob['0']['features']) or '없음'}; "
                 f"+1 = {', '.join(ob['1']['features']) or '없음'}.")
        L.append("  - 가격(오프셋 종가 ÷ 사건일 종가 − 1, %, 평균) −5 / −1 / +1 / +5: 사건 " +
                 " / ".join(f2(pm['events'].get(k)) for k in (-5, -1, 1, 5)) + ", 가짜 " +
                 " / ".join(f2(pm['fakes'].get(k)) for k in (-5, -1, 1, 5)) + ", 위약 " +
                 " / ".join(f2(pm['placebo'].get(k)) for k in (-5, -1, 1, 5)) + f". 위약 후보 날 {b['placebo'].get('n_candidates')}개.")
        L.append(f"  - 그림 `run_{tag}/figs/{b['name']}.html`(가격 칸 + 특징 46칸: 사건별 선·사건 평균·가짜 평균·위약 평균과 95% 띠), "
                 f"값 `run_{tag}/profiles.parquet`.")
    L.append("- 위약의 사건 전 가격 낙폭(상승폭)이 사건보다 작아, 띠 밖 특징 수에는 가격 차이의 몫이 섞여 있다(3-1절).\n")

    L.append("## 6. 지금 위치 (설명용, 10/15~10/23 창 — '지금이 바닥/고점'이라는 판단이 아니다)\n")
    if a == "SKHYNIX":
        L.append(f"- 하이닉스 끝 극값 {te.get('date')}(저점 후보)는 미확정이다(10/22 종가 {te.get('last_close'):,.0f}원, 8% 확정에는 "
                 f"{te.get('confirm_level'):,.0f}원 이상 종가 필요). 10/15 저점은 10/20·10/22 더 낮은 종가로 깨져 후보가 아니다. "
                 "10/15~10/22 가짜 바닥 후보 6일은 보류라 위 분석에 없다.")
    L.append("- 10/13 뒤 창은 실시간 수집이다. 조회·추천 원값 특징은 값이 없고(소급 창만 비교), 댓글 양 특징(`act_ln_comments_pre_per_h`)은 "
             "실시간 창에서 글당 댓글이 덜 붙은 채 받혀 낮게 나올 수 있으며, 저장 글은 아직 안 지워진 글이 섞여 소급 창보다 몇 % 많게 "
             "나올 수 있다.\n")
    for b in blocks:
        nc, nl = b["now_candidates"], b["now_lead"]
        if nc:
            L.append(f"### {E.TYPE_KO[b['kind']]} 가르기 후보 칸 — 창 z를 같은 오프셋의 사건·가짜·위약 분포에 놓은 백분위\n")
            L.append("| 특징@오프셋 | 방향(사건 z / 가짜 z 평균) | 창 | 가상 사건일 | z | 사건 분포 안 | 가짜 분포 안 | 위약 분포 안 |")
            L.append("|---|---|---|---|---|---|---|---|")
            for x in nc:
                for w in x["windows"]:
                    if not np.isfinite(w["z"]):
                        continue
                    L.append(f"| {x['feature']}@{x['offset']:+d} | {x['direction']}({f2(x['mean_z_events'])} / "
                             f"{f2(x['mean_z_fakes'])}) | {w['win_id']} | {w['hypothetical_event_day']} | {f2(w['z'])} | "
                             f"{f2(w['pct_in_events'])} | {f2(w['pct_in_fakes'])} | {f2(w['pct_in_placebo'])} |")
                if not any(np.isfinite(w["z"]) for w in x["windows"]):
                    L.append(f"| {x['feature']}@{x['offset']:+d} | {x['direction']} | 10/15~10/23 | — | 값 없음(실시간 창) | | | |")
            L.append("")
        if nl:
            L.append(f"### {E.TYPE_KO[b['kind']]} 앞섬 후보 — 각 창을 가상의 사건일 D로 둘 때 직전 1~5창 극값 닿음\n")
            L.append("| 특징(쪽) | 사건 / 가짜 / 위약 비율 | " + " | ".join(w["win_id"] for w in nl[0]["windows"]) + " |")
            L.append("|---|---|" + "---|" * len(nl[0]["windows"]))
            for x in nl:
                cells = []
                for w in x["windows"]:
                    hit = "닿음" if w["hit"] == 1 else ("안 닿음" if w["hit"] == 0 else "—")
                    cells.append(f"{hit}({f2(w['extreme_pct'], 3)})")
                L.append(f"| {x['feature']}({x['side']}) | {f2(x['rate_events'])} / {f2(x['rate_fakes'])} / "
                         f"{f2(x['rate_placebo_mean'])} | " + " | ".join(cells) + " |")
            re_ = [x["rate_events"] for x in nl]
            rp_ = [x["rate_placebo_mean"] for x in nl if x["rate_placebo_mean"] is not None]
            L.append(f"\n- 괄호 = 직전 1~5창 중 가장 극단인 직전 60창 백분위(상위 쪽은 최댓값, 하위 쪽은 최솟값, 기준 0.9 이상·0.1 이하). "
                     f"확정 {E.TYPE_KO[b['kind']]} 앞에서도 닿지 않은 경우가 {100 * (1 - max(re_)):.0f}~{100 * (1 - min(re_)):.0f}%이고, "
                     f"위약 날도 {100 * min(rp_):.0f}~{100 * max(rp_):.0f}%는 닿는다. 닿음이 그 날이 {E.TYPE_KO[b['kind']]}이라는 뜻이 "
                     "아니다. 가격 메아리 점검(3-1절)에서 글 수·저장 글·반대매매 몫은 가격 급락이 있던 날끼리 견주면 위약과 비슷했다.\n")
        if not nc and not nl:
            L.append(f"### {E.TYPE_KO[b['kind']]}\n\n- 가르기·앞섬 후보가 없어 지금 위치 표를 내지 않는다(전체 값 `run_{tag}/now.csv`).\n")

    L.append("## 7. 기간 나눔 (예측 층 지식 기간 2026-07-01 전후, 설명용)\n")
    for b in blocks:
        ks = b["knowledge_sensitivity_pred_layer"]
        if ks and "note" in ks[0]:
            L.append(f"- {E.TYPE_KO[b['kind']]}: {ks[0]['note']}.")
            continue
        L.append(f"- {E.TYPE_KO[b['kind']]}: " + "; ".join(
            f"{x['feature']}@{x['offset']:+d} BA {f2(x.get('ba'))} p {f2(x.get('p'), 3)} (n {x['n_events']}/{x['n_fakes']})"
            for x in ks if x.get("status") == "시험") + ".")
    L.append("- 예측 층 라벨 모델 지식 기간(2026-06까지) 창은 설명·예열용이다. 주 분석은 등록 정의대로 전 기간을 썼으므로 예측 층 후보"
             "(`pr_*`)는 지식 기간 밖에서 따로 확인되지 않았다.")
    for b in blocks:
        for x in b["candidate_period_split"]:
            def c_(k):
                return f"{f2(x[k]['mean_z'])}(n {x[k]['n']})"
            L.append(f"- {E.TYPE_KO[b['kind']]} 후보 {x['feature']}@{x['offset']:+d} 기간 나눔(z 평균): 2026-07-01 전 사건 "
                     f"{c_('events_early')} vs 가짜 {c_('fakes_early')}; 뒤 사건 {c_('events_late')} vs 가짜 {c_('fakes_late')}.")
    L.append("")

    L.append("## 8. 읽는 법과 한계\n")
    L.append("- 탐색 연구다. '후보'는 2026-10-24 이후 확정되는 새 고점·저점으로만 확인한다(부록 13). 두 문턱 모두 후보여야 '강건 후보'다.")
    if overlap and "error" not in overlap:
        L.append(f"- 다른 문턱 {overlap['threshold']}% 결과(`{overlap['file']}`, 다른 실행이 만든 파일, sha256 {overlap['sha256'][:12]}…)와 "
                 f"겹치는 후보: {', '.join(overlap['both']) or '없음'}. 이 문턱에만: {', '.join(overlap['only_here']) or '없음'}. "
                 f"{overlap['threshold']}%에만: {', '.join(overlap['only_other']) or '없음'}. 큰 문턱의 꺾임점은 작은 문턱 꺾임점에 대부분 "
                 "들어 있어 두 문턱 일치는 독립 확인이 아니다.")
    elif overlap:
        L.append(f"- 다른 문턱 결과 `{overlap['file']}`를 읽지 못했다({overlap['error']}).")
    for b in blocks:
        ltb = pd.DataFrame(b["lead"])
        nfl = int(ltb.loc[ltb["status"] == "시험", "n_fakes"].max()) if (ltb["status"] == "시험").any() else 0
        L.append(f"- {E.TYPE_KO[b['kind']]}: 사건 {b['n']['events_total']}개, 가짜 묶음 대표 {len(b['fake_reps'])}개(앞섬 값이 있는 것 "
                 f"최대 {nfl}개). 가짜가 적으면 '대 가짜' 비교는 힘이 작고 가짜 몇 개에 크게 흔들린다(시운전: 사건 8·가짜 6에서 1 SD "
                 "차이 검출률 0.3~0.4).")
    L.append(f"- 시험 범위: 특징 {meta_run['n_features']}개 × 오프셋 2 = {2 * meta_run['n_features']}칸(통과 {2 * meta_run['n_pass']}칸)을 "
             f"가르기에, 같은 특징 × 상위·하위 = {2 * meta_run['n_features']}칸을 앞섬에 썼다. 대비 2개(바닥·고점). 우연 기대 개수와 "
             "귀무 95%를 같이 본다.\n")
    L.append("## 산출\n")
    L.append(f"- `results/run_{tag}.json`(요약·전 칸 표·점검), `results/run_{tag}.md`(이 문서)")
    L.append(f"- `results/run_{tag}/`: `tests.csv`(가르기 전 칸), `lead.csv`(앞섬 전 칸), `now.csv`(10/15~10/23 창 × 특징, 오프셋 0 분포 "
             f"기준), `profiles.parquet`(오프셋 −10~+10 사건·가짜·위약 평균과 띠), `events_detail.csv`(사건·가짜 대표 날의 오프셋 0·+1 z), "
             f"`figs/*.html`(사건 전후 그림), `event_study.md`(event_study.write_report 표준판)")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="SKHYNIX")
    ap.add_argument("--threshold", default="8")
    a = ap.parse_args(argv)
    t0 = datetime.now(E.KST)
    ev_path, feat_path = E.RESULTS / "events.json", E.RESULTS / "features_all.parquet"
    meta_path, prices_path = E.RESULTS / "features_all_meta.json", E.RESULTS / "prices_close.csv"
    ev = json.loads(ev_path.read_text("utf-8"))
    ev, mode = E.confirmed_only_events(ev, E.CONFIRMED_ASOF)
    pivots = ev["assets"][a.asset]["thresholds"][a.threshold]["pivots"]
    F, meta = E.load_features(feat_path, meta_path)
    prices = pd.read_csv(prices_path)
    contrasts = [c for c in E.load_contrasts(ev) if c.asset == a.asset and c.threshold == a.threshold]
    assert [c.kind for c in contrasts] == ["bottom", "top"], [c.kind for c in contrasts]
    market = contrasts[0].market
    panel = E.build_panel(F, meta, market)
    close = E.price_series(prices, a.asset)
    tag = f"{a.asset}_{a.threshold}"
    out = E.RESULTS / f"run_{tag}"
    figdir = out / "figs"
    figdir.mkdir(parents=True, exist_ok=True)
    results = []
    for c in contrasts:
        r = E.run_contrast(c, panel, close, n_null=E.N_NULL)
        results.append(r)
        (figdir / f"{c.name}.html").write_text(E.figure_html(r, panel, title_note="확정 사건만(2026-10-22 종가) · "), "utf-8")
    tests = pd.concat([r["table"].assign(kind=r["contrast"].kind) for r in results])
    leads = pd.concat([r["lead"].assign(kind=r["contrast"].kind) for r in results])
    nows = pd.concat([r["now"].assign(kind=r["contrast"].kind) for r in results if len(r["now"])])
    tests.to_csv(out / "tests.csv", index=False, encoding="utf-8-sig")
    leads.to_csv(out / "lead.csv", index=False, encoding="utf-8-sig")
    nows.to_csv(out / "now.csv", index=False, encoding="utf-8-sig")
    prof_rows, det_rows = [], []
    for r in results:
        c, pr = r["contrast"], r["profiles"]
        for fi, f in enumerate(panel.feats):
            for j, k in enumerate(E.OFFSETS):
                prof_rows.append({"kind": c.kind, "feature": f, "offset": int(k), "ev_mean": pr["ev_mean"][j, fi],
                                  "ev_n": int(pr["ev_n"][j, fi]),
                                  "fk_mean": pr["fk_mean"][j, fi] if pr["fk_mean"] is not None else np.nan,
                                  "pl_mean": pr["pl_mean"][j, fi] if "pl_mean" in pr else np.nan,
                                  "pl_lo": pr["pl_lo"][j, fi] if "pl_lo" in pr else np.nan,
                                  "pl_hi": pr["pl_hi"][j, fi] if "pl_hi" in pr else np.nan})
        for role, df in (("사건", c.events), ("가짜 대표", c.fake_reps)):
            pos = E.positions(panel, df["win_id"])
            for d, w, p in zip(df["date"], df["win_id"], pos):
                for k in (0, 1):
                    row = {"kind": c.kind, "role": role, "date": d, "win_id": w, "offset": k}
                    for fi, f in enumerate(panel.feats):
                        row[f] = panel.Z[p + k, fi] if (p >= 0 and p + k < len(panel.win_id)) else np.nan
                    det_rows.append(row)
    pd.DataFrame(prof_rows).to_parquet(out / "profiles.parquet", index=False)
    pd.DataFrame(det_rows).to_csv(out / "events_detail.csv", index=False, encoding="utf-8-sig")
    E.write_report(results, {market: panel}, ev_path, feat_path, out, E.N_NULL, mode=mode)   # 표준판 event_study.md
    blocks = [contrast_block(r, panel, F, pivots, close) for r in results]
    feats = panel.feats
    meta_run = {"made": t0.replace(microsecond=0).isoformat(), "event_study_sha": E.sha(HERE / "event_study.py"),
                "runner_sha": E.sha(Path(__file__)), "events_sha": E.sha(ev_path), "features_sha": E.sha(feat_path),
                "prices_sha": E.sha(prices_path), "n_windows": int(len(panel.win_id)), "n_features": len(feats),
                "n_pass": sum(meta[f]["status"] == "통과" for f in feats),
                "n_expl": sum(meta[f]["status"] == "설명용" for f in feats),
                "offsets_classify": list(E.CLS_OFFSETS), "n_null": E.N_NULL, "placebo_pool": E.PLACEBO_POOL}
    overlap = other_threshold_overlap(blocks)
    md = md_report(blocks, mode, meta_run, overlap)
    meta_run["seconds"] = round((datetime.now(E.KST) - t0).total_seconds(), 1)
    summary = {"combo": tag, "run": meta_run, "mode": mode, "contrasts": blocks, "other_threshold_overlap": overlap}
    (E.RESULTS / f"run_{tag}.json").write_text(json.dumps(clean(summary), ensure_ascii=False, indent=1, default=str), "utf-8")
    (E.RESULTS / f"run_{tag}.md").write_text(md, "utf-8")
    print(json.dumps({b["name"]: {"family": b["family"]["통과"], "lead": b["lead_family"]["통과"], "leak": b["leak"]["flag"],
                                  "cand": [(x["feature"], x["offset"]) for x in b["candidates"]],
                                  "lead_cand": [(x["feature"], x["side"]) for x in b["lead_candidates"]]}
                      for b in blocks}, ensure_ascii=False, default=str))
    print("seconds", meta_run["seconds"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
