"""부록 15 사건 연구 3단계: 앞선 실행(events/REPORT.md §9) 후보 8개를 새로 들어온 사건(2025-01-21 ~ 2025-08-20 확정 꺾임점)만으로
확인한다. 결과를 본 뒤 규칙을 바꾸지 않도록, 판정 규칙은 아래에 먼저 적고 코드로 고정한다.

기준(과제 지시 그대로)
  - 가르기 칸: 하나 빼고 맞히기 대신 '앞선 실행에서 정한 문턱·방향'을 그대로 새 사건·새 가짜(묶음 대표)에 적용한다.
    앞선 문턱 = 앞선 실행 전체 표본(확정 사건만, 10/22 종가)에 맞춘 균형 정확도 최대 문턱(같은 점수면 위쪽·낮은 문턱).
    events/results/run_*.json에 적힌 값(KOSPI_12 rule_cut, SKHYNIX_12 full_threshold_z, SOXX_8 rules_in_sample,
    SOXX_12 fit_threshold_z)을 읽는다. 적히지 않은 칸(KOSPI_8·SKHYNIX_8 전부, 몇몇 근접 칸)은 앞선 실행 입력(events/results의
    events.json·features_all.parquet·prices_close.csv)으로 같은 규칙을 다시 맞춰 쓰고, 적힌 칸에서 다시 맞춘 값이 적힌 값과 같은지
    확인한다. 앞선 실행의 BA·p도 같은 코드로 다시 내어 앞선 JSON의 후보 값과 같은지 확인한다.
    새 표본 p = 고정 문턱 2×2 표의 피셔 단측 p(사건 쪽 판정이 사건에 몰렸는가). 방향 = 새 사건·새 가짜 z 평균의 차 부호.
  - 앞섬 칸: 새 사건 전 1~5창이 직전 60창 상위(하위) 10%에 닿은 비율 대 위약(새 사건마다 가격이 비슷한 날 20일,
    event_study.placebo_pools 그대로) 비율. p = 이항 분포 위쪽 꼬리, 97.5% 분위도 적는다. 새 가짜 비율은 참고로 적는다.
    5번(반대매매·저장 글)은 '급락 날끼리'(run_combo_SKHYNIX_8.price_lead_indicators의 방향 극값 = 장 D−6~D−2 일간 수익률이
    직전 60거래일 하위 10%에 닿음) 층 안에서 사건 대 위약을 함께 본다.
판정 규칙(결과 보기 전에 고정)
  가르기: 새 사건·새 가짜가 각각 1개 이상 값이 있어야 판정한다.
    재현 = 고정 문턱 BA > 0.5 그리고 피셔 p < 0.05 / 같은 쪽 = BA > 0.5, p ≥ 0.05(우연과 구별 안 됨) / 재현 안 됨 = BA ≤ 0.5.
    사건이나 가짜가 4개 미만이면(event_study MIN_TRUE·MIN_FAKE) '표본 부족' 표시를 붙인다.
  앞섬: 새 사건 1개 이상 값이 있어야 판정한다.
    재현 = 사건 비율 > 위약 97.5% 분위 그리고 p < 0.05 / 같은 쪽 = 사건 비율 > 위약 비율 / 재현 안 됨 = 사건 비율 ≤ 위약 비율.
  판정 불가: 7번(예측 층) — 기준이 '7/1 뒤 사건만으로'인데 새 사건은 모두 라벨 지식 기간(D < 2026-07-01)이다.
            8번(조회 ÷ 댓글) — 기준이 '실시간 수집값으로 새로 재기'인데 새 사건 창은 모두 소급 창이다.
            두 칸은 고정 문턱 값을 설명용으로만 적는다.
실행: py -B new_event_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import warnings  # noqa: E402
from datetime import datetime  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import binom, hypergeom  # noqa: E402

import event_study_rerun as R  # noqa: E402  (같은 폴더: ES 모듈·사건 기간 거르기·fit_rule)
import snap_paths as SP  # noqa: E402

ES = R.ES
OUT = SP.RUN_EV / "results"
OLD = SP.EV / "results"
NEW_START, NEW_END = R.STUDY_START, R.NEW_END
KO = {"bottom": "바닥", "top": "고점"}
ASSET_KO = R.ASSET_KO
MIN_E, MIN_F = ES.MIN_TRUE, ES.MIN_FAKE


def load_rc8():
    spec = importlib.util.spec_from_file_location("events_code_run_combo_SKHYNIX_8", SP.EV / "code" / "run_combo_SKHYNIX_8.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


RC8 = load_rc8()

# 후보 8개(REPORT.md §9). cls = (자산, 문턱, 종류, 특징, 오프셋), lead = (자산, 문턱, 종류, 특징, 쪽)
CANDS = [
    {"id": 1, "label": "하이닉스 바닥 앞섬 tp_MEM_A 상위 10%", "source": "8%", "kind": "lead",
     "criterion": "새 바닥 전 1~5창 닿은 비율이 위약보다 높은가", "main": [("SKHYNIX", "8", "bottom", "tp_MEM_A", "상위 10%")],
     "ref": [("SKHYNIX", "12", "bottom", "tp_MEM_A", "상위 10%")]},
    {"id": 2, "label": "SOXX 바닥 tp_LL_US_A@0 (낮음)", "source": "8%", "kind": "cls",
     "criterion": "새 바닥 장 전 창 z가 가짜보다 낮은가(앞선 문턱 그대로)", "main": [("SOXX", "8", "bottom", "tp_LL_US_A", 0)],
     "ref": [("SOXX", "12", "bottom", "tp_LL_US_A", 0)]},
    {"id": 3, "label": "하이닉스 바닥 tp_LS_A@0 (낮음)", "source": "12%", "kind": "cls",
     "criterion": "같은 방향인가(앞선 문턱 그대로)", "main": [("SKHYNIX", "12", "bottom", "tp_LS_A", 0)],
     "ref": [("SKHYNIX", "8", "bottom", "tp_LS_A", 0)]},
    {"id": 4, "label": "KOSPI 고점 act_share_night@+1 (낮음)", "source": "12% 후보, 8% 근접", "kind": "cls",
     "criterion": "같은 방향인가(앞선 문턱 그대로)", "main": [("KOSPI", "12", "top", "act_share_night", 1),
                                                    ("KOSPI", "8", "top", "act_share_night", 1)], "ref": []},
    {"id": 5, "label": "하이닉스 바닥 앞섬 tp_MC_A·저장 글 상위 10%", "source": "두 문턱", "kind": "lead",
     "criterion": "급락 날끼리 비교해도 위약보다 높은가",
     "main": [("SKHYNIX", "8", "bottom", "tp_MC_A", "상위 10%"), ("SKHYNIX", "8", "bottom", "act_ln_saved_per_h", "상위 10%"),
              ("SKHYNIX", "12", "bottom", "tp_MC_A", "상위 10%"), ("SKHYNIX", "12", "bottom", "act_ln_saved_per_h", "상위 10%")],
     "ref": [], "price_stratum": True},
    {"id": 6, "label": "고점 act_new30_share@+1 (낮음)", "source": "6개 조합 방향 같음", "kind": "cls",
     "criterion": "같은 방향인가(조합마다 앞선 문턱 그대로)",
     "main": [(a, x, "top", "act_new30_share", 1) for a, x in R.COMBOS], "ref": []},
    {"id": 7, "label": "하이닉스 고점 pr_P_share@0", "source": "8%", "kind": "cls", "criterion": "7/1 뒤 사건만으로",
     "main": [("SKHYNIX", "8", "top", "pr_P_share", 0)], "ref": [], "undecidable": "knowledge"},
    {"id": 8, "label": "고점 vw_ln_b_median@+1", "source": "3개 조합", "kind": "cls", "criterion": "실시간 수집값으로 새로 재기",
     "main": [("KOSPI", "12", "top", "vw_ln_b_median", 1), ("SKHYNIX", "8", "top", "vw_ln_b_median", 1),
              ("SKHYNIX", "12", "top", "vw_ln_b_median", 1)], "ref": [], "undecidable": "realtime"},
]


def sha(p: Path) -> str:
    return R.sha(p)


# ---------------------------------------------------------------- 앞선 실행: JSON에 적힌 문턱

def json_rules() -> dict:
    """(자산, 문턱, 종류, 특징, 오프셋) → {thr_z, is_up, where}. 형식이 조합마다 달라 아는 자리만 읽는다."""
    out = {}

    def put(key, thr, is_up, where):
        out.setdefault(key, []).append({"thr_z": float(thr), "is_up": bool(is_up), "where": where})

    d = json.loads((OLD / "run_KOSPI_12.json").read_text("utf-8"))
    for i, x in enumerate(d.get("now_matched", [])):
        put(("KOSPI", "12", x["kind"], x["feature"], int(x["offset"])), x["rule_cut"], x["rule_direction"] == "up",
            f"run_KOSPI_12.json now_matched[{i}].rule_cut")
    d = json.loads((OLD / "run_SKHYNIX_12.json").read_text("utf-8"))
    for kind, b in d["contrasts"].items():
        for nm in ("candidates_cls", "near_cls_p_lt_0_10"):
            for i, x in enumerate(b.get(nm, [])):
                if x.get("full_threshold_z") is not None:
                    put(("SKHYNIX", "12", kind, x["feature"], int(x["offset"])), x["full_threshold_z"],
                        "크면" in x["full_rule"], f"run_SKHYNIX_12.json contrasts.{kind}.{nm}[{i}].full_threshold_z")
    d = json.loads((OLD / "run_SOXX_8.json").read_text("utf-8"))
    for kind, b in d["contrasts"].items():
        for cell, v in (b.get("rules_in_sample") or {}).items():
            f, k = cell.split("@")
            put(("SOXX", "8", kind, f, int(k)), v["thr_z"], v["is_up"], f"run_SOXX_8.json contrasts.{kind}.rules_in_sample['{cell}']")
    d = json.loads((OLD / "run_SOXX_12.json").read_text("utf-8"))
    for b in d["contrasts"]:
        kind = b["kind"]
        for i, x in enumerate(b.get("top_tests_pass", [])):
            if x.get("fit_threshold_z") is not None:
                put(("SOXX", "12", kind, x["feature"], int(x["offset"])), x["fit_threshold_z"], "크면" in x["fit_direction"],
                    f"run_SOXX_12.json contrasts[{kind}].top_tests_pass[{i}].fit_threshold_z")
    return out


# ---------------------------------------------------------------- 앞선 실행 다시 내기(같은 코드·입력)

class Prior:
    def __init__(self):
        ev = json.loads((OLD / "events.json").read_text("utf-8"))
        self.ev, self.mode = ES.confirmed_only_events(ev, ES.CONFIRMED_ASOF)
        F, meta = ES.load_features(OLD / "features_all.parquet", OLD / "features_all_meta.json")
        self.panels = {m: ES.build_panel(F, meta, m) for m in ("kr", "us")}
        self.prices = pd.read_csv(OLD / "prices_close.csv")
        self.contrasts = {c.name: c for c in ES.load_contrasts(self.ev)}
        self.runs = {}
        self.inputs = {k: sha(OLD / k) for k in ("events.json", "features_all.parquet", "features_all_meta.json", "prices_close.csv")}

    def run(self, name: str) -> dict:
        if name not in self.runs:
            c = self.contrasts[name]
            self.runs[name] = ES.run_contrast(c, self.panels[c.market], ES.price_series(self.prices, c.asset), n_null=ES.N_NULL,
                                              with_now=False)
        return self.runs[name]

    def rule(self, name: str, feature: str, offset: int) -> dict:
        r = self.run(name)
        panel = self.panels[self.contrasts[name].market]
        fi = panel.feats.index(feature)
        v = ES.gather(panel.Z, r["cls"]["base"], [offset])[:, 0, fi]
        ok = np.isfinite(v)
        return R.fit_rule(v[ok], r["cls"]["y"][ok])


# ---------------------------------------------------------------- 새 사건 쪽

class New:
    def __init__(self):
        ev = json.loads(R.EVJ.read_text("utf-8"))
        ES.guard_real_run(ev)
        self.ev, self.rec = R.study_period_events(ev)
        F, meta = ES.load_features(R.FEAT, R.META)
        self.panels = {m: ES.build_panel(F, meta, m) for m in ("kr", "us")}
        self.prices = pd.read_csv(R.PRICES)
        self.contrasts = {c.name: c for c in ES.load_contrasts(self.ev)}
        self._pools = {}
        self.inputs = {"events.json": sha(R.EVJ), "features_all.parquet": sha(R.FEAT), "features_all_meta.json": sha(R.META),
                       "prices_close.csv": sha(R.PRICES)}

    def pools(self, name: str):
        if name not in self._pools:
            c = self.contrasts[name]
            close = ES.price_series(self.prices, c.asset)
            self._pools[name] = ES.placebo_pools(self.panels[c.market], close, c.events["date"], c.pivot_dates)
        return self._pools[name]

    def samples(self, name: str) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        c = self.contrasts[name]
        em = (c.events["date"] >= NEW_START) & (c.events["date"] <= NEW_END)
        fm = (c.fake_reps["date"] >= NEW_START) & (c.fake_reps["date"] <= NEW_END)
        return c.events[em].reset_index(drop=True), c.fake_reps[fm].reset_index(drop=True), np.flatnonzero(em.to_numpy())


def fixed_rule_check(new: New, name: str, feature: str, offset: int, rule: dict, prior_dir: str) -> dict:
    c = new.contrasts[name]
    panel = new.panels[c.market]
    fi = panel.feats.index(feature)
    evs, fks, _ = new.samples(name)
    ev_pos, fk_pos = ES.positions(panel, evs["win_id"]), ES.positions(panel, fks["win_id"])
    ze = ES.gather(panel.Z, ev_pos, [offset])[:, 0, fi]
    zf = ES.gather(panel.Z, fk_pos, [offset])[:, 0, fi]

    def side(z):
        if not np.isfinite(z):
            return None
        return bool((z > rule["thr_z"]) if rule["is_up"] else (z <= rule["thr_z"]))

    ev_rows = [{"date": d, "win_id": w, "z": z, "event_side": side(z)} for d, w, z in zip(evs["date"], evs["win_id"], ze)]
    fk_rows = [{"date": d, "win_id": w, "run": int(rn), "z": z, "event_side": side(z)}
               for d, w, rn, z in zip(fks["date"], fks["win_id"], fks["run"], zf)]
    e_ok = [r for r in ev_rows if r["event_side"] is not None]
    f_ok = [r for r in fk_rows if r["event_side"] is not None]
    P, N = len(e_ok), len(f_ok)
    a = sum(r["event_side"] for r in e_ok)
    d = sum(not r["event_side"] for r in f_ok)
    out = {"n_new_events": len(ev_rows), "n_new_fakes": len(fk_rows), "n_events_with_z": P, "n_fakes_with_z": N,
           "events_hit": a, "fakes_hit": d, "events": ev_rows, "fakes": fk_rows}
    if P and N:
        ba = 0.5 * (a / P + d / N)
        pred_event = a + (N - d)
        p = float(hypergeom.sf(a - 1, P + N, P, pred_event))
        me, mf = float(np.mean([r["z"] for r in e_ok])), float(np.mean([r["z"] for r in f_ok]))
        nd = "사건 쪽이 낮음" if me < mf else "사건 쪽이 높음"
        if ba > 0.5 and p < 0.05:
            verdict = "재현"
        elif ba > 0.5:
            verdict = "같은 쪽(우연과 구별 안 됨)"
        else:
            verdict = "재현 안 됨"
        out.update({"ba_fixed_rule": ba, "fisher_p_one_sided": p, "mean_z_events": me, "mean_z_fakes": mf,
                    "direction_new": nd, "direction_same_as_prior": nd == prior_dir, "verdict": verdict,
                    "small_sample": P < MIN_E or N < MIN_F})
    else:
        out.update({"verdict": "판정 불가(새 사건 또는 새 가짜에 값 없음)", "small_sample": True,
                    "events_side_share": (a / P) if P else None})
    return out


def lead_check(new: New, name: str, feature: str, side_ko: str, price_stratum: bool) -> dict:
    c = new.contrasts[name]
    panel = new.panels[c.market]
    fi = panel.feats.index(feature)
    evs, fks, ev_idx = new.samples(name)
    ev_pos, fk_pos = ES.positions(panel, evs["win_id"]), ES.positions(panel, fks["win_id"])
    pools, pinfo = new.pools(name)
    new_pools = [pools[i] for i in ev_idx]
    pp = ES.pool_positions(new_pools)
    hiE, loE = ES.lead_hits(panel, ev_pos)
    hiF, loF = ES.lead_hits(panel, fk_pos)
    hE = (hiE if side_ko == "상위 10%" else loE)[:, fi]
    hF = (hiF if side_ko == "상위 10%" else loF)[:, fi]
    if len(pp):
        hiP, loP = ES.lead_hits(panel, pp)
        hP = (hiP if side_ko == "상위 10%" else loP)[:, fi]
    else:
        hP = np.array([])
    pct = ES.gather(panel.PCT, ev_pos, np.arange(ES.LEAD[0], ES.LEAD[1] + 1))[:, :, fi]
    ev_rows = [{"date": d, "win_id": w, "hit": (None if not np.isfinite(h) else bool(h)),
                "lead_pct": [None if not np.isfinite(x) else round(float(x), 3) for x in row]}
               for d, w, h, row in zip(evs["date"], evs["win_id"], hE, pct)]
    ok = np.isfinite(hE)
    n, k = int(ok.sum()), int(np.nansum(hE))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ph = float(np.nanmean(hP)) if len(hP) and np.isfinite(hP).any() else np.nan
        rf = float(np.nanmean(hF)) if np.isfinite(hF).any() else np.nan
    out = {"n_new_events": len(ev_rows), "n_events_with_value": n, "events_hit": k, "events": ev_rows,
           "n_new_fakes": int(len(fks)), "n_fakes_with_value": int(np.isfinite(hF).sum()), "rate_fakes": rf,
           "placebo_days": int(np.isfinite(hP).sum()) if len(hP) else 0, "rate_placebo": ph, "placebo_info": pinfo}
    if n and np.isfinite(ph):
        re = k / n
        q975 = float(binom.ppf(0.975, n, ph) / n)
        p = float(binom.sf(k - 1, n, ph))
        if re > q975 and p < 0.05:
            verdict = "재현"
        elif re > ph:
            verdict = "같은 쪽(우연과 구별 안 됨)"
        else:
            verdict = "재현 안 됨"
        out.update({"rate_events": re, "rate_placebo_q975": q975, "p_vs_placebo": p, "verdict": verdict,
                    "small_sample": n < MIN_E})
    else:
        out.update({"verdict": "판정 불가(새 사건 앞섬 창에 값 없음)", "small_sample": True})
    if price_stratum:
        close = ES.price_series(new.prices, c.asset)

        def dates(pos):
            return [str(panel.D[i]) if i >= 0 else None for i in pos]
        PE = RC8.price_lead_indicators(close, dates(ev_pos), c.kind)
        PF = RC8.price_lead_indicators(close, dates(fk_pos), c.kind)
        PPd = RC8.price_lead_indicators(close, dates(pp), c.kind) if len(pp) else np.zeros((0, 3))
        st = {}
        for sv, nm in ((1.0, "급락 있음(방향 극값 닿음)"), (0.0, "급락 없음")):
            mE = np.isfinite(hE) & (PE[:, 0] == sv)
            mF = np.isfinite(hF) & (PF[:, 0] == sv)
            mP = (np.isfinite(hP) & (PPd[:, 0] == sv)) if len(pp) else np.array([], bool)
            ne, ke = int(mE.sum()), int(hE[mE].sum()) if mE.any() else 0
            pr = float(hP[mP].mean()) if len(pp) and mP.any() else np.nan
            row = {"ev_n": ne, "ev_hit": ke, "ev_rate": (ke / ne) if ne else None,
                   "fk_n": int(mF.sum()), "fk_rate": float(hF[mF].mean()) if mF.any() else None,
                   "pl_n": int(mP.sum()) if len(pp) else 0, "pl_rate": pr}
            if ne and np.isfinite(pr):
                row["p_vs_placebo"] = float(binom.sf(ke - 1, ne, pr))
                row["placebo_q975"] = float(binom.ppf(0.975, ne, pr) / ne)
            st[nm] = row
        out["price_echo"] = {"rule": "장 D−6~D−2 일간 수익률이 직전 60거래일 하위 10%에 닿았나(run_combo_SKHYNIX_8.price_lead_indicators)",
                             "events_price_dir": [None if not np.isfinite(x) else bool(x) for x in PE[:, 0]],
                             "strata": st}
        g = st["급락 있음(방향 극값 닿음)"]
        if g["ev_n"] and g["pl_rate"] is not None and np.isfinite(g["pl_rate"]):
            if g["ev_rate"] > g.get("placebo_q975", np.inf) and g.get("p_vs_placebo", 1) < 0.05:
                out["verdict_crash_stratum"] = "재현"
            elif g["ev_rate"] > g["pl_rate"]:
                out["verdict_crash_stratum"] = "같은 쪽(우연과 구별 안 됨)"
            else:
                out["verdict_crash_stratum"] = "재현 안 됨"
        else:
            out["verdict_crash_stratum"] = "판정 불가(급락 층에 새 사건 없음)"
    return out


def z_shift_old_events(prior: Prior, new: New, name: str, feature: str, offset: int) -> dict:
    """설명용: 같은 옛 사건(2025-08-21 뒤)의 z가 앞선 판과 이번 판에서 얼마나 다른가(예열 창이 늘어서)."""
    co, cn = prior.contrasts[name], new.contrasts[name]
    po, pn = prior.panels[co.market], new.panels[cn.market]
    fo, fn = po.feats.index(feature), pn.feats.index(feature)
    old_ev = co.events
    zo = ES.gather(po.Z, ES.positions(po, old_ev["win_id"]), [offset])[:, 0, fo]
    zn = ES.gather(pn.Z, ES.positions(pn, old_ev["win_id"]), [offset])[:, 0, fn]
    both = np.isfinite(zo) & np.isfinite(zn)
    return {"n_old_events": int(len(old_ev)), "both_finite": int(both.sum()),
            "max_abs_diff": float(np.max(np.abs(zo[both] - zn[both]))) if both.any() else None,
            "n_diff_gt_0_01": int((np.abs(zo[both] - zn[both]) > 0.01).sum()) if both.any() else 0,
            "nan_to_value": int((~np.isfinite(zo) & np.isfinite(zn)).sum())}


def f2(x, nd=2):
    return R.f2(x, nd)


def main() -> int:
    t0 = datetime.now()
    jsha = sha(R.JUDGE)
    prior, new = Prior(), New()
    jr = json_rules()
    # 앞선 실행 다시 내기 확인: 앞선 JSON 후보 BA·p와 같은가
    verify = []
    for a, x in R.COMBOS:
        ps = R.prior_summary(a, x)
        for kind, b in ps["contrasts"].items():
            r = prior.run(f"{a}_{x}_{kind}")
            tb, lt = r["table"], r["lead"]
            for cnd in b["candidates"]:
                q = tb[(tb["feature"] == cnd["feature"]) & (tb["offset"] == cnd["offset"])].iloc[0]
                verify.append({"cell": f"{a}_{x}_{kind} {cnd['feature']}@{cnd['offset']:+d}", "prior_ba": cnd["ba"], "redo_ba": q["ba"],
                               "prior_p": cnd["p"], "redo_p": q["p"], "redo_candidate": bool(q["candidate"]),
                               "same": abs(q["ba"] - cnd["ba"]) < 1e-6 and abs(q["p"] - cnd["p"]) < 1e-6 and bool(q["candidate"])})
            for cnd in b["lead_candidates"]:
                q = lt[(lt["feature"] == cnd["feature"]) & (lt["side"] == cnd["side"])].iloc[0]
                verify.append({"cell": f"{a}_{x}_{kind} 앞섬 {cnd['feature']} {cnd['side']}", "prior_rate": cnd["rate_events"],
                               "redo_rate": q["rate_events"], "prior_p": cnd["p"], "redo_p": q["p"], "redo_candidate": bool(q["candidate"]),
                               "same": abs(q["rate_events"] - cnd["rate_events"]) < 1e-6 and abs(q["p"] - cnd["p"]) < 1e-6
                               and bool(q["candidate"])})
    # 적힌 문턱 = 다시 맞춘 문턱인가(적힌 칸 전부)
    rule_match = []
    for key, vals in jr.items():
        a, x, kind, f, k = key
        rr = prior.rule(f"{a}_{x}_{kind}", f, k)
        for v in vals:
            rule_match.append({"cell": f"{a}_{x}_{kind} {f}@{k:+d}", "where": v["where"], "json_thr": v["thr_z"], "json_is_up": v["is_up"],
                               "redo_thr": rr["thr_z"], "redo_is_up": rr["is_up"],
                               "same": abs(rr["thr_z"] - v["thr_z"]) < 1e-9 and rr["is_up"] == v["is_up"]})
    results = []
    for cd in CANDS:
        blk = {k: v for k, v in cd.items() if k not in ("main", "ref")}
        cells = []
        for role, lst in (("main", cd["main"]), ("ref", cd["ref"])):
            for item in lst:
                a, x, kind, f, k = item
                name = f"{a}_{x}_{kind}"
                rp = prior.run(name)
                cell = {"role": "후보 칸" if role == "main" else "참고(다른 문턱, 앞선 실행 후보 아님)", "combo": f"{a}_{x}", "kind": kind,
                        "feature": f, "ko": new.panels[new.contrasts[name].market].meta[f]["ko"]}
                if cd["kind"] == "cls":
                    q = rp["table"][(rp["table"]["feature"] == f) & (rp["table"]["offset"] == k)].iloc[0]
                    redo = prior.rule(name, f, k)
                    js = jr.get((a, x, kind, f, k))
                    if js:
                        rule = {"thr_z": js[0]["thr_z"], "is_up": js[0]["is_up"]}
                        src = f"앞선 JSON({js[0]['where']}); 같은 입력으로 다시 맞춘 값과 " + (
                            "일치" if abs(redo["thr_z"] - js[0]["thr_z"]) < 1e-9 and redo["is_up"] == js[0]["is_up"] else "불일치")
                    else:
                        rule = {"thr_z": redo["thr_z"], "is_up": redo["is_up"]}
                        src = "앞선 JSON에 없음 → 앞선 실행 입력(events/results)으로 같은 규칙 다시 맞춤"
                    rule["event_side"] = "z > 문턱" if rule["is_up"] else "z ≤ 문턱"
                    prior_dir = q.get("direction")
                    cell.update({"offset": k, "prior": {"ba_loeo": q["ba"], "p": q["p"], "direction": prior_dir,
                                                        "candidate": bool(q["candidate"]), "n_events": int(q["n_events"]),
                                                        "n_fakes": int(q["n_fakes"]), "rule": rule, "rule_source": src,
                                                        "rule_ba_in_sample": redo["ba_in_sample"]},
                                 "new": fixed_rule_check(new, name, f, k, rule, prior_dir),
                                 "old_event_z_shift": z_shift_old_events(prior, new, name, f, k)})
                    if cd.get("undecidable"):
                        cell["new"]["verdict_descriptive"] = cell["new"]["verdict"]
                        cell["new"]["verdict"] = ("판정 불가(새 사건 모두 라벨 지식 기간)" if cd["undecidable"] == "knowledge"
                                                  else "판정 불가(새 사건 창은 소급 창 — 조회는 사건 뒤 누적값)")
                else:
                    q = rp["lead"][(rp["lead"]["feature"] == f) & (rp["lead"]["side"] == k)].iloc[0]
                    cell.update({"side": k, "prior": {"rate_events": q["rate_events"], "rate_fakes": q["rate_fakes"],
                                                      "rate_placebo": q.get("rate_placebo_mean"), "p": q["p"],
                                                      "p_vs_placebo": q.get("p_vs_placebo"), "candidate": bool(q["candidate"])},
                                 "new": lead_check(new, name, f, k, bool(cd.get("price_stratum")))})
                cells.append(cell)
        blk["cells"] = cells
        mains = [c for c in cells if c["role"] == "후보 칸"]
        vk = "verdict_crash_stratum" if cd.get("price_stratum") else "verdict"
        blk["summary"] = {c["combo"] + ("" if cd["kind"] == "cls" else f" {c['feature']}"): c["new"][vk] for c in mains}
        results.append(blk)
    out = {"made": datetime.now().replace(microsecond=0).isoformat(), "addendum": "HYPOTHESES_addendum15.md (부록 13 사건 연구)",
           "new_events": {"period": [NEW_START, NEW_END], "rule": "극값 날짜가 이 기간인 확정 꺾임점(10/23 종가까지 확정), 가짜는 같은 기간 묶음 대표"},
           "judge_py": {"sha256": jsha, "matches": jsha == R.JUDGE_SHA_EXPECT},
           "inputs": {"new": new.inputs, "prior": prior.inputs, "event_study_py_sha256": sha(SP.EV / "code" / "event_study.py"),
                      "run_combo_SKHYNIX_8_py_sha256": sha(SP.EV / "code" / "run_combo_SKHYNIX_8.py"),
                      "this_code_sha256": sha(Path(__file__)), "event_study_rerun_py_sha256": sha(HERE / "event_study_rerun.py")},
           "prior_redo_check": {"cells": verify, "all_same": all(v["same"] for v in verify)},
           "prior_rule_json_vs_redo": {"cells": rule_match, "all_same": all(v["same"] for v in rule_match)},
           "new_sample_counts": {nm: {"events_new": int(len(new.samples(nm)[0])), "fake_reps_new": int(len(new.samples(nm)[1])),
                                      "events_new_dates": new.samples(nm)[0]["date"].tolist(),
                                      "fake_reps_new_dates": new.samples(nm)[1]["date"].tolist()}
                                 for nm in sorted(new.contrasts)},
           "fake_runs_crossing_2025_08_20": crossing_runs(new),
           "warmup_diff": warmup_diff(),
           "candidates": results,
           "seconds": (datetime.now() - t0).total_seconds()}
    out = R.clean(out)
    (OUT / "new_event_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf-8")
    (OUT / "new_event_check.md").write_text(write_md(out), "utf-8")
    for b in out["candidates"]:
        print(b["id"], b["label"], b["summary"])
    print("prior redo all same:", out["prior_redo_check"]["all_same"], "| json rule vs redo all same:", out["prior_rule_json_vs_redo"]["all_same"])
    return 0


def warmup_diff() -> list:
    c = json.loads((OUT / "events_rerun_compare.json").read_text("utf-8"))["warmup_extended"]["assets"]
    out = []
    for a, b in c.items():
        for x in ("8", "12"):
            t = b.get(x) or {}
            for kind, d in t.get("only_used", []):
                out.append(f"{ASSET_KO[a]} {x}% {KO['bottom' if kind == 'low' else 'top']} {d}(연장 예열에서는 꺾임점 아님)")
            for kind, d in t.get("only_extended", []):
                out.append(f"{ASSET_KO[a]} {x}% {KO['bottom' if kind == 'low' else 'top']} {d}(연장 예열에서만 꺾임점)")
    return out


def crossing_runs(new: New) -> list:
    out = []
    for nm, c in new.contrasts.items():
        fa = c.fakes_all
        if not len(fa):
            continue
        for rn, g in fa.groupby("run"):
            if (g["date"] <= NEW_END).any() and (g["date"] > NEW_END).any():
                rep = c.fake_reps[c.fake_reps["run"] == rn]["date"].tolist()
                out.append({"contrast": nm, "run": int(rn), "dates": g["date"].tolist(), "rep": rep})
    return out


# ---------------------------------------------------------------- 보고서

def _ev_list(rows, key="event_side"):
    s = []
    for r in rows:
        if r.get("z") is None:
            s.append(f"{r['date']}(값 없음)")
        else:
            s.append(f"{r['date']} z {r['z']:+.2f}{'●' if r[key] else ''}")
    return ", ".join(s) or "없음"


def write_md(o: dict) -> str:
    L = ["# 새 사건으로 앞선 후보 8개 확인 (부록 15, 2025-01-21 ~ 2025-08-20 확정 사건)\n",
         "> 앞선 실행(events/REPORT.md §9)의 후보를 이번에 처음 포럼과 이은 사건으로만 다시 봤습니다. 가르기 칸은 앞선 실행이 정한 "
         "문턱·방향을 그대로 썼고(새로 맞추지 않음), 앞섬 칸은 위약과 비교했습니다. 탐색 결과이고 매매 규칙이 아닙니다. "
         "'지금이 바닥/고점'이라는 판단도 하지 않습니다.\n"]
    # 1. 결론
    L += ["## 1. 결론\n", "| 순서 | 칸 | 기준 | 새 사건(사건·가짜) | 결과 | 판정 |", "|---|---|---|---|---|---|"]
    for b in o["candidates"]:
        for c in b["cells"]:
            if c["role"] != "후보 칸":
                continue
            n = c["new"]
            if b["kind"] == "cls":
                ns = f"{n['n_events_with_z']}/{n['n_new_events']}·{n['n_fakes_with_z']}/{n['n_new_fakes']}"
                res = (f"사건 {n['events_hit']}/{n['n_events_with_z']}·가짜 {n['fakes_hit']}/{n['n_fakes_with_z']}, BA "
                       f"{f2(n.get('ba_fixed_rule'))}, p {f2(n.get('fisher_p_one_sided'), 3)}, 방향 {n.get('direction_new', '—')}")
                v = n["verdict"] + (" · 표본 부족" if n.get("small_sample") and not n["verdict"].startswith("판정 불가") else "")
                if n.get("direction_same_as_prior") is False and n["verdict"].startswith("같은 쪽"):
                    v += " · 평균 z 방향은 앞선 실행과 반대"
                cell = f"{ASSET_KO[c['combo'].split('_')[0]]} {c['combo'].split('_')[1]}% {c['feature']}@{c['offset']:+d}"
            else:
                ns = f"{n['n_events_with_value']}/{n['n_new_events']}·—"
                res = (f"닿음 사건 {n['events_hit']}/{n['n_events_with_value']} 대 위약 {f2(n.get('rate_placebo'))}"
                       f"(97.5% {f2(n.get('rate_placebo_q975'))}), p {f2(n.get('p_vs_placebo'), 3)}; 새 가짜 {f2(n.get('rate_fakes'))}")
                if "price_echo" in n:
                    g = n["price_echo"]["strata"]["급락 있음(방향 극값 닿음)"]
                    res += (f"; 급락 층 사건 {g['ev_hit']}/{g['ev_n']} 대 위약 {f2(g['pl_rate'])}")
                    v = n.get("verdict_crash_stratum") + " (급락 층) / 전체 " + n["verdict"]
                else:
                    v = n["verdict"]
                if n.get("small_sample") and "판정 불가" not in v:
                    v += " · 표본 부족"
                cell = f"{ASSET_KO[c['combo'].split('_')[0]]} {c['combo'].split('_')[1]}% {c['feature']} {c['side']}"
            L.append(f"| {b['id']} | {cell} | {b['criterion']} | {ns} | {res} | {v} |")
    L.append("")
    tally = {}
    for b in o["candidates"]:
        for c in b["cells"]:
            if c["role"] == "후보 칸":
                v = c["new"].get("verdict_crash_stratum") if b.get("price_stratum") else c["new"]["verdict"]
                tally[v.split("(")[0].strip()] = tally.get(v.split("(")[0].strip(), 0) + 1
    L.append("- 판정 칸 수(후보 칸 " + str(sum(tally.values())) + "개): " + ", ".join(f"{k} {v}" for k, v in tally.items()) + ".")
    L.append("- 후보별: " + "; ".join(f"{b['id']}번 " + ", ".join(sorted({v.split('(')[0].strip() for v in b['summary'].values()}))
                                     for b in o["candidates"]) + f". '재현' 판정 칸은 {tally.get('재현', 0)}개입니다.")
    L.append("- 새 사건은 자산·문턱마다 1~5개입니다. 칸 하나에서 '재현'이 나오려면 새 표본이 거의 완벽히 갈려야 합니다. "
             "그래서 '같은 쪽'은 지지 근거로 약하고, '재현 안 됨'도 반증으로 강하지 않습니다.")
    L.append("- KOSPI와 하이닉스는 같은 W_KR 창을 씁니다. 같은 날 사건(예: 2025-02-19 고점)은 두 자산에서 같은 z라 독립 확인이 아닙니다.\n")
    # 2. 기준
    L += ["## 2. 기준과 판정 규칙(결과 보기 전에 코드로 고정)\n",
          "| 항목 | 내용 |", "|---|---|",
          "| 새 사건 | 극값 날짜 2025-01-21 ~ 2025-08-20인 확정 꺾임점(10/23 종가까지). 가짜는 같은 기간 묶음 대표 |",
          "| 가르기 | 앞선 실행 전체 표본 문턱·방향을 그대로 적용. 새 표본 BA와 피셔 단측 p(2×2 표) |",
          "| 앞섬 | 사건 전 1~5창이 직전 60창 상위(하위) 10%에 닿은 비율 대 위약(새 사건마다 가격이 비슷한 날 20일). 이항 p·97.5% 분위 |",
          "| 재현 | 가르기: BA > 0.5이고 p < 0.05. 앞섬: 위약 97.5% 분위 위이고 p < 0.05 |",
          "| 같은 쪽 | 가르기: BA > 0.5, p ≥ 0.05. 앞섬: 위약 비율보다 높음 |",
          "| 재현 안 됨 | 가르기: BA ≤ 0.5. 앞섬: 위약 비율 이하 |",
          "| 판정 불가 | 값이 있는 새 사건이나 새 가짜가 없을 때. 7번은 새 사건이 모두 라벨 지식 기간, 8번은 새 사건 창이 모두 소급 창 |",
          "| 표본 부족 표시 | 값 있는 새 사건이나 새 가짜가 4개 미만(event_study MIN_TRUE·MIN_FAKE) |", ""]
    pr, rm = o["prior_redo_check"], o["prior_rule_json_vs_redo"]
    L.append(f"- 앞선 문턱 출처: 앞선 JSON에 적힌 칸은 그 값을 썼습니다. 같은 입력·같은 코드로 다시 맞춘 값과 {len(rm['cells'])}칸 중 "
             f"{sum(c['same'] for c in rm['cells'])}칸이 같았습니다. JSON에 없는 칸(KOSPI_8·SKHYNIX_8 전부 등)은 다시 맞춘 값을 썼습니다.")
    L.append(f"- 앞선 실행 다시 내기: 앞선 JSON 후보 {len(pr['cells'])}칸의 BA·p(앞섬은 비율·p)가 "
             f"{'모두 같았습니다' if pr['all_same'] else '일부 달랐습니다'}({sum(c['same'] for c in pr['cells'])}/{len(pr['cells'])}).\n")
    # 3. 세부
    L += ["## 3. 칸별 세부\n", "●는 앞선 문턱으로 '사건 쪽' 판정, 앞섬은 ●가 '닿음'입니다.\n"]
    for b in o["candidates"]:
        L.append(f"### {b['id']}. {b['label']} — 출처 {b['source']}\n")
        for c in b["cells"]:
            n, p = c["new"], c["prior"]
            if b["kind"] == "cls":
                r = p["rule"]
                L.append(f"- **{c['combo']} {KO[c['kind']]} {c['feature']}@{c['offset']:+d}** ({c['role']}) — 앞선: BA {f2(p['ba_loeo'])}, "
                         f"p {f2(p['p'], 3)}, {p['direction']}, 문턱 {r['event_side']} {f2(r['thr_z'], 3)}({p['rule_source']}).")
                L.append(f"  - 새 사건: {_ev_list(n['events'])}")
                L.append(f"  - 새 가짜: {_ev_list(n['fakes'])}")
                if "ba_fixed_rule" in n:
                    L.append(f"  - 결과: 사건 {n['events_hit']}/{n['n_events_with_z']}, 가짜 {n['fakes_hit']}/{n['n_fakes_with_z']} 맞힘, BA "
                             f"{f2(n['ba_fixed_rule'])}, 피셔 p {f2(n['fisher_p_one_sided'], 3)}, 평균 z 사건 {f2(n['mean_z_events'])} 대 가짜 "
                             f"{f2(n['mean_z_fakes'])}({n['direction_new']}, 앞선 방향과 {'같음' if n['direction_same_as_prior'] else '반대'}) → "
                             f"**{n['verdict']}**" + (f" (설명용 값: {n['verdict_descriptive']})" if n.get("verdict_descriptive") else ""))
                else:
                    L.append(f"  - 결과: **{n['verdict']}**")
                zs = c["old_event_z_shift"]
                if zs["n_diff_gt_0_01"] or zs["nan_to_value"]:
                    L.append(f"  - 옛 사건 z 변화(설명용): {zs['both_finite']}개 중 {zs['n_diff_gt_0_01']}개가 0.01 넘게 바뀜(최대 "
                             f"{f2(zs['max_abs_diff'])}), 결측→값 {zs['nan_to_value']}개")
            else:
                L.append(f"- **{c['combo']} {KO[c['kind']]} 앞섬 {c['feature']} {c['side']}** ({c['role']}) — 앞선: 사건 {f2(p['rate_events'])} / "
                         f"가짜 {f2(p['rate_fakes'])} / 위약 {f2(p['rate_placebo'])}, p {f2(p['p'], 3)}.")
                ev_s = ", ".join(f"{e['date']}{'(값 없음)' if e['hit'] is None else ('●' if e['hit'] else '')}" for e in n["events"]) or "없음"
                L.append(f"  - 새 사건: {ev_s}")
                if "rate_events" in n:
                    L.append(f"  - 결과: 닿음 {n['events_hit']}/{n['n_events_with_value']}({f2(n['rate_events'])}) 대 위약 {f2(n['rate_placebo'])}"
                             f"(97.5% {f2(n['rate_placebo_q975'])}, 위약 날 {n['placebo_days']}), p {f2(n['p_vs_placebo'], 3)}; 새 가짜 "
                             f"{f2(n['rate_fakes'])}({n['n_fakes_with_value']}개) → **{n['verdict']}**")
                else:
                    L.append(f"  - 결과: **{n['verdict']}**")
                if "price_echo" in n:
                    for nm, g in n["price_echo"]["strata"].items():
                        L.append(f"  - {nm}: 사건 {g['ev_hit']}/{g['ev_n']} 대 위약 {f2(g['pl_rate'])}(위약 날 {g['pl_n']})"
                                 + (f", p {f2(g.get('p_vs_placebo'), 3)}" if "p_vs_placebo" in g else "")
                                 + f"; 가짜 {f2(g['fk_rate'])}({g['fk_n']}개)")
                    L.append(f"  - 급락 층 판정: **{n['verdict_crash_stratum']}**")
        L.append("")
    # 4. 한계
    L += ["## 4. 한계\n",
          "- **표본이 매우 작습니다.** 새 사건(바닥/고점)·새 가짜 묶음 대표(바닥/고점): " + "; ".join(
              f"{ASSET_KO[a]} {x}% {o['new_sample_counts'][f'{a}_{x}_bottom']['events_new']}/{o['new_sample_counts'][f'{a}_{x}_top']['events_new']}"
              f"·{o['new_sample_counts'][f'{a}_{x}_bottom']['fake_reps_new']}/{o['new_sample_counts'][f'{a}_{x}_top']['fake_reps_new']}"
              for a, x in R.COMBOS) + ".",
          "- **z 예열:** 2025-01~02 사건은 직전 20창 중 값 15개, 평일 아닌 창 보정 5개가 모자라 z가 없는 칸이 많습니다. "
          "앞섬은 직전 60창 중 값 40개가 필요해 2025-03 전 사건은 값이 없습니다.",
          "- **지그재그 예열:** 가격 자료가 2025-01-02부터라 처음 꺾임점은 시작점에 민감합니다. 2024-06부터 yfinance 원값을 붙여 다시 돌린 "
          "점검(events_rerun_compare.json warmup_extended)에서 2025-01-21 뒤 꺾임점이 달라진 곳: " + (", ".join(o["warmup_diff"]) or "없음")
          + ". 사건 목록은 정의대로 두었습니다.",
          "- **앞선 문턱은 앞선 판 z로 정했습니다.** 이번 판에서는 2025-08-21 앞 창이 생겨 옛 사건 z도 조금 바뀝니다(칸별 세부의 '옛 사건 z 변화').",
          "- **같은 게시판:** KOSPI와 하이닉스는 같은 W_KR 창입니다. 8%·12%도 사건이 겹칩니다. 6번의 6개 조합은 서로 독립이 아닙니다.",
          "- **7·8번:** 기준 자체가 이번 새 사건으로는 채워지지 않습니다. 값은 설명용으로만 적었습니다.",
          "- **소급 창:** 2025년 창은 모두 나중에 받은 글입니다. 수집 전에 지워진 글이 빠져 있습니다.", ""]
    L += ["## 5. 산출\n", "- `new_event_check.json`(칸별 표본·문턱 출처·다시 내기 확인), 코드 `events/run2026_10_26/code/new_event_check.py`"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
