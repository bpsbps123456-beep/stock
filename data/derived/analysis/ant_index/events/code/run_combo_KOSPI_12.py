"""부록 13 사건 연구 — 확정 사건만 모드(부록 14), KOSPI 12% 조합 하나만 실행.

event_study.py의 함수(확정 사건만 확인, 대비 읽기, 판 만들기, run_contrast, 그림)를 그대로 부른다. 분석 로직·문턱·귀무·위약은
바꾸지 않는다. 이 파일이 더하는 것은 (1) 조합 하나만 고르기, (2) 하나 빼고 맞히기의 사건·가짜별 적중 수(BA를 loeo_ba와 대조),
(3) 끝 극값(미확정) 창의 위치, (4) AI 라벨 지식 기간(동결 §2.2: 2026-07-01 전 창) 표시뿐이다.

실행: py -B run_combo_KOSPI_12.py  → events/results/run_KOSPI_12.json, run_KOSPI_12.md, run_KOSPI_12/(그림·표)
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import event_study as es  # noqa: E402

ASSET, THR = "KOSPI", "12"
COMBO = f"{ASSET}_{THR}"
RES = es.RESULTS
OUT_JSON = RES / f"run_{COMBO}.json"
OUT_MD = RES / f"run_{COMBO}.md"
OUT_DIR = RES / f"run_{COMBO}"
KNOW_END = np.datetime64("2026-07-01", "D")      # 동결 §2.2: LLM 라벨은 2026-07-01부터만 판정에 쓴다
LABEL_BUNDLES = ("예측 층", "감정 층")


def loeo_pred(v: np.ndarray, y: np.ndarray) -> np.ndarray:
    """loeo_ba_naive와 같은 규칙(표본 하나 빼고 문턱·방향을 균형 정확도 최대로, 같은 점수면 위쪽·낮은 문턱)으로 빠진 표본의
    예측(True = 사건)을 원래 순서로 돌려준다."""
    v = np.asarray(v, float)
    y = np.asarray(y, bool)
    n = len(v)
    pred = np.zeros(n, bool)
    for i in range(n):
        te = np.zeros(n, bool)
        te[i] = True
        vt, yt = v[~te], y[~te]
        s = np.sort(vt)
        cuts = [-np.inf] + [(s[k - 1] + s[k]) / 2 for k in range(1, len(s)) if s[k] > s[k - 1]] + [np.inf]
        best, choice = -np.inf, None
        P, N = yt.sum(), (~yt).sum()
        for direction in ("up", "dn"):
            for c in cuts:
                pr = vt > c if direction == "up" else ~(vt > c)
                ba = 0.5 * ((pr & yt).sum() / P + (~pr & ~yt).sum() / N)
                if ba > best + 1e-12:
                    best, choice = ba, (direction, c)
        d, c = choice
        pr = v[te] > c
        pred[te] = pr if d == "up" else ~pr
    return pred


def knowledge_mix(panel: es.Panel, ev_pos: np.ndarray, fk_pos: np.ndarray) -> dict:
    def cnt(pos):
        pos = pos[pos >= 0]
        return {"n": int(len(pos)), "in_knowledge": int((panel.D[pos] < KNOW_END).sum())}
    return {"events": cnt(ev_pos), "fake_reps": cnt(fk_pos)}


def cell_hits(panel: es.Panel, r: dict) -> pd.DataFrame:
    """특징 × 오프셋(0, +1)마다 빠진 표본 예측의 사건·가짜 적중 수와 사건별 예측. BA를 classify 값과 대조한다."""
    cls = r["cls"]
    base, y = cls["base"], cls["y"]
    c = r["contrast"]
    ev_dates = list(c.events["date"])
    ev_pos_all = es.positions(panel, c.events["win_id"])
    lab = {}
    for d, p in zip(ev_dates, ev_pos_all):
        if p >= 0:
            lab[int(p)] = d
    fk_dates = dict(zip(es.positions(panel, c.fake_reps["win_id"]).tolist(), c.fake_reps["date"]))
    V = es.gather(panel.Z, base, es.CLS_OFFSETS)
    rows = []
    tb = r["table"].set_index(["feature", "offset"])
    for oi, k in enumerate(es.CLS_OFFSETS):
        for fi, f in enumerate(panel.feats):
            v = V[:, oi, fi]
            ok = np.isfinite(v)
            row = tb.loc[(f, int(k))]
            if row["status"] != "시험":
                continue
            pr = loeo_pred(v[ok], y[ok])
            yy = y[ok]
            hit_e, hit_f = int((pr & yy).sum()), int((~pr & ~yy).sum())
            ba = 0.5 * (hit_e / yy.sum() + hit_f / (~yy).sum())
            if abs(ba - row["ba"]) > 1e-9:
                raise SystemExit(f"BA 불일치 {f}@{k}: {ba} vs {row['ba']}")
            b_ok = base[ok]
            ev_detail = {lab.get(int(b), str(b)): bool(p_) for b, p_, t in zip(b_ok, pr, yy) if t}
            fk_detail = {fk_dates.get(int(b), str(b)): bool(not p_) for b, p_, t in zip(b_ok, pr, yy) if not t}
            rows.append({"feature": f, "offset": int(k), "hit_events": hit_e, "n_events": int(yy.sum()),
                         "hit_fakes": hit_f, "n_fakes": int((~yy).sum()), "ba_check": ba,
                         "events_pred_correct": json.dumps(ev_detail, ensure_ascii=False),
                         "fakes_pred_correct": json.dumps(fk_detail, ensure_ascii=False)})
    return pd.DataFrame(rows)


def fit_rule(v: np.ndarray, y: np.ndarray) -> tuple[str, float, float]:
    """전 표본으로 문턱·방향(loeo_pred와 같은 규칙). 반환 (방향 up|dn, 문턱, 표본 안 BA). up = 문턱 위가 사건 쪽."""
    v = np.asarray(v, float)
    y = np.asarray(y, bool)
    s = np.sort(v)
    cuts = [-np.inf] + [(s[k - 1] + s[k]) / 2 for k in range(1, len(s)) if s[k] > s[k - 1]] + [np.inf]
    best, choice = -np.inf, None
    P, N = y.sum(), (~y).sum()
    for direction in ("up", "dn"):
        for c in cuts:
            pr = v > c if direction == "up" else ~(v > c)
            ba = 0.5 * ((pr & y).sum() / P + (~pr & ~y).sum() / N)
            if ba > best + 1e-12:
                best, choice = ba, (direction, c)
    return choice[0], float(choice[1]), float(best)


def matched_now(panel: es.Panel, r: dict, cells: list[tuple[str, int]]) -> pd.DataFrame:
    """분석 4 보조(설명용): 지금 창(10/15~10/23)의 z를 그 칸과 같은 오프셋의 사건·가짜 z 분포와 견준다(코드 now_position은 모두
    오프셋 0 분포와 견준다). 전 표본 문턱의 어느 쪽인지도 적는다 — '지금이 바닥/고점'이라는 뜻이 아니다."""
    cls = r["cls"]
    base, y = cls["base"], cls["y"]
    sel = np.flatnonzero((panel.D >= np.datetime64(es.NOW_FROM.date(), "D")) &
                         (panel.D <= np.datetime64(es.NOW_TO.date(), "D")))
    rows = []
    for f, k in cells:
        fi = panel.feats.index(f)
        v = es.gather(panel.Z, base, [k])[:, 0, fi]
        ok = np.isfinite(v)
        ev, fk = v[ok & y], v[ok & ~y]
        d, cut, ba_in = fit_rule(v[ok], y[ok])
        for t in sel:
            z = panel.Z[t, fi]

            def pr(ref):
                return float(((ref < z).sum() + 0.5 * (ref == z).sum()) / len(ref)) if (len(ref) and np.isfinite(z)) else np.nan
            side = (z > cut if d == "up" else not (z > cut)) if np.isfinite(z) else None
            rows.append({"feature": f, "offset": int(k), "win_id": panel.win_id[t], "D": str(panel.D[t]), "z": z,
                         "pct_in_events_same_offset": pr(ev), "pct_in_fakes_same_offset": pr(fk),
                         "rule_direction": d, "rule_cut": cut, "rule_ba_in_sample": ba_in, "event_side": side,
                         "realtime_window": bool(panel.realtime[t]),
                         "realtime_caution": (not panel.meta[f]["realtime_comparable"]) or
                                             ("실시간" in panel.meta[f].get("note", ""))})
    return pd.DataFrame(rows)


def tail_lead(panel: es.Panel, tail_win: str) -> pd.DataFrame:
    """끝 극값(미확정) 창 기준 오프셋 −5~−1 극값 닿음(분석 3과 같은 계산, 설명용)."""
    pos = np.array([panel.pos.get(tail_win, -1)])
    hi, lo = es.lead_hits(panel, pos)
    return pd.DataFrame({"feature": panel.feats, "hit_hi": hi[0], "hit_lo": lo[0]})


def main() -> int:
    t0 = datetime.now(es.KST).replace(microsecond=0)
    ev_path = RES / "events.json"
    ev_raw = json.loads(ev_path.read_text("utf-8"))
    ev, mode = es.confirmed_only_events(ev_raw, es.CONFIRMED_ASOF)
    F, meta = es.load_features()
    prices = pd.read_csv(RES / "prices_close.csv")
    contrasts = [c for c in es.load_contrasts(ev) if c.asset == ASSET and c.threshold == THR]
    assert [c.kind for c in contrasts] == ["bottom", "top"], [c.name for c in contrasts]
    panel = es.build_panel(F, meta, contrasts[0].market)
    close = es.price_series(prices, ASSET)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figdir = OUT_DIR / "figs"
    figdir.mkdir(exist_ok=True)
    results, tests, leads, nows, profs, hits = [], [], [], [], [], []
    for c in contrasts:
        r = es.run_contrast(c, panel, close, n_null=es.N_NULL)
        r["hits"] = cell_hits(panel, r)
        ev_pos = es.positions(panel, c.events["win_id"])
        fk_pos = es.positions(panel, c.fake_reps["win_id"])
        r["knowledge"] = knowledge_mix(panel, ev_pos, fk_pos)
        tb0 = r["table"]
        cells = [(f, int(o)) for f, o in zip(tb0.loc[tb0["p"] < 0.10, "feature"], tb0.loc[tb0["p"] < 0.10, "offset"])]
        r["now_matched"] = matched_now(panel, r, cells).assign(kind=c.kind)
        results.append(r)
        h = r["hits"]
        tb = r["table"].set_index(["feature", "offset"])
        chk = h.set_index(["feature", "offset"])
        if not ((tb.loc[chk.index, "n_events"] == chk["n_events"]).all() and
                (tb.loc[chk.index, "n_fakes"] == chk["n_fakes"]).all()):
            raise SystemExit("표본 수 불일치")
        tests.append(r["table"].merge(h.drop(columns=["n_events", "n_fakes"]), on=["feature", "offset"], how="left")
                     .assign(kind=c.kind))
        leads.append(r["lead"].assign(kind=c.kind))
        nows.append(r["now"].assign(kind=c.kind))
        pr = r["profiles"]
        for fi, f in enumerate(panel.feats):
            for j, k in enumerate(es.OFFSETS):
                profs.append({"kind": c.kind, "feature": f, "offset": int(k), "ev_mean": pr["ev_mean"][j, fi],
                              "ev_n": int(pr["ev_n"][j, fi]), "fk_mean": pr["fk_mean"][j, fi],
                              "pl_mean": pr["pl_mean"][j, fi], "pl_lo": pr["pl_lo"][j, fi], "pl_hi": pr["pl_hi"][j, fi]})
        (figdir / f"{c.name}.html").write_text(es.figure_html(
            r, panel, "확정 사건만 모드(2026-10-22 종가). 오프셋 +2 이후는 사건 뒤 정보라 '알아챔'의 근거가 아니다. "), "utf-8")
    T = pd.concat(tests, ignore_index=True)
    T["label_knowledge_mix"] = T["bundle"].isin(LABEL_BUNDLES)
    T.to_csv(OUT_DIR / "tests.csv", index=False, encoding="utf-8-sig")
    Ld = pd.concat(leads, ignore_index=True)
    Ld.to_csv(OUT_DIR / "lead.csv", index=False, encoding="utf-8-sig")
    Nw = pd.concat(nows, ignore_index=True)
    Nw.to_csv(OUT_DIR / "now.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(profs).to_csv(OUT_DIR / "profiles.csv", index=False, encoding="utf-8-sig")
    # 사건별 원 z(오프셋 −10~+10) — 그림의 사건별 선 자료
    ev_rows = []
    for r in results:
        c = r["contrast"]
        E = r["profiles"]["events"]
        for i, d in enumerate(c.events["date"]):
            for j, k in enumerate(es.OFFSETS):
                for fi, f in enumerate(panel.feats):
                    ev_rows.append({"kind": c.kind, "event_date": d, "offset": int(k), "feature": f, "z": E[i, j, fi]})
    pd.DataFrame(ev_rows).to_csv(OUT_DIR / "profiles_by_event.csv", index=False, encoding="utf-8-sig")
    pp_rows = []
    for r in results:
        for grp, M in r["price_paths"].items():
            with np.errstate(invalid="ignore"):
                m = np.nanmean(M, 0) if len(M) else np.full(len(es.OFFSETS), np.nan)
            for j, k in enumerate(es.OFFSETS):
                pp_rows.append({"kind": r["contrast"].kind, "group": grp, "offset": int(k), "mean_ret": m[j],
                                "n": int(np.isfinite(M[:, j]).sum()) if len(M) else 0})
    pd.DataFrame(pp_rows).to_csv(OUT_DIR / "price_paths.csv", index=False, encoding="utf-8-sig")
    NM = pd.concat([r["now_matched"] for r in results], ignore_index=True)
    NM.to_csv(OUT_DIR / "now_matched.csv", index=False, encoding="utf-8-sig")
    tail = ev_raw["assets"][ASSET]["thresholds"][THR]["tail_unconfirmed"]
    TL = tail_lead(panel, tail["win_id"])
    TL.to_csv(OUT_DIR / "tail_lead.csv", index=False, encoding="utf-8-sig")
    summary = build_summary(results, panel, mode, ev_raw, t0, T, Ld, Nw)
    summary["now_matched"] = json.loads(NM.to_json(orient="records", force_ascii=False))
    summary["tail_lead"] = json.loads(TL.to_json(orient="records", force_ascii=False))
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), "utf-8")
    OUT_MD.write_text(build_md(summary, results, panel, T, Ld, Nw), "utf-8")
    print(json.dumps({k: summary[k] for k in ("combo", "n", "family", "lead_family", "leak")}, ensure_ascii=False,
                     indent=1, default=str))
    return 0


def _rec(df: pd.DataFrame, cols) -> list:
    return json.loads(df.reindex(columns=cols).to_json(orient="records", force_ascii=False))


def build_summary(results, panel, mode, ev_raw, t0, T, Ld, Nw) -> dict:
    a = ev_raw["assets"][ASSET]["thresholds"][THR]
    s = {"combo": COMBO, "made": t0.isoformat(), "mode": mode["contrasts"].get(COMBO), "asof": mode["asof"],
         "inputs": {"events_json_sha256": es.sha(RES / "events.json"),
                    "features_all_sha256": es.sha(RES / "features_all.parquet"),
                    "prices_close_sha256": es.sha(RES / "prices_close.csv")},
         "settings": {"n_null": es.N_NULL, "offsets_cls": list(es.CLS_OFFSETS), "lead": list(es.LEAD),
                      "placebo_pool": es.PLACEBO_POOL, "alpha": es.ALPHA, "seed": es.SEED,
                      "n_features_grid": len(panel.feats),
                      "n_features_by_status": {st: int(sum(panel.meta[f]["status"] == st for f in panel.feats))
                                               for st in ("통과", "설명용")}},
         "tail_unconfirmed": a.get("tail_unconfirmed"), "held_near_tail": a.get("held_near_tail"),
         "n": {}, "family": {}, "lead_family": {}, "leak": {}, "outside_band": {}, "knowledge": {}, "events": {},
         "cells_p_lt_005": {}, "candidates": {}, "lead_p_lt_005": {}, "lead_candidates": {}, "now": {}}
    for r in results:
        c = r["contrast"]
        k = c.kind
        s["n"][k] = r["n"]
        s["family"][k] = r["family"]
        s["lead_family"][k] = r["lead_family"]
        s["leak"][k] = r["leak"]
        s["outside_band"][k] = r["outside_band"]
        s["knowledge"][k] = r["knowledge"]
        s["events"][k] = {"events": c.events.to_dict("records"),
                          "fake_reps": c.fake_reps[["date", "win_id", "run", "close"]].to_dict("records"),
                          "placebo": r["placebo"]}
        t = T[T["kind"] == k]
        cols = ["feature", "offset", "bundle", "status_feature", "ba", "p", "direction", "mean_z_events", "mean_z_fakes",
                "hit_events", "n_events", "hit_fakes", "n_fakes", "leak_suspect", "candidate", "label_knowledge_mix",
                "ba_all_fake_days_run_groups", "null_ba_q95"]
        s["cells_p_lt_005"][k] = _rec(t[t["p"] < es.ALPHA].sort_values("p"), cols)
        s["candidates"][k] = _rec(t[t["candidate"]].sort_values("p"), cols)
        lt = Ld[Ld["kind"] == k]
        lcols = ["feature", "side", "bundle", "status_feature", "rate_events", "rate_fakes", "rate_placebo_mean",
                 "rate_placebo_q975", "p", "p_vs_placebo", "leak_suspect", "candidate", "n_events", "n_fakes"]
        s["lead_p_lt_005"][k] = _rec(lt[lt["p"] < es.ALPHA].sort_values("p"), lcols)
        s["lead_candidates"][k] = _rec(lt[lt["candidate"]].sort_values("p"), lcols)
        nw = Nw[Nw["kind"] == k]
        feats_now = sorted(set(t.loc[t["p"] < es.ALPHA, "feature"]))
        s["now"][k] = _rec(nw[nw["feature"].isin(feats_now)],
                           ["win_id", "feature", "z", "pct60", "pct_in_events", "pct_in_fakes", "pct_in_placebo",
                            "realtime_window", "realtime_caution"])
    return s


def f2(x, nd=2):
    return es.fmt(x, nd)


def _flags(q) -> str:
    fl = []
    if q.get("leak_suspect"):
        fl.append("누출 의심")
    if q.get("bundle") in LABEL_BUNDLES:
        fl.append("지식 기간 섞임")
    if q.get("status_feature") == "설명용":
        fl.append("설명용")
    if q.get("candidate"):
        fl.append("**코드 후보**")
    return ", ".join(fl)


def build_md(s, results, panel, T, Ld, Nw) -> str:
    L = []
    ko = lambda f: panel.meta[f]["ko"]  # noqa: E731
    tail = s["tail_unconfirmed"] or {}
    fb, ft = s["family"]["bottom"]["통과"], s["family"]["top"]["통과"]
    lb, lt_ = s["lead_family"]["bottom"]["통과"], s["lead_family"]["top"]["통과"]
    cb, ct = s["candidates"]["bottom"], s["candidates"]["top"]
    L.append(f"# 개미 지수 사건 연구 — KOSPI 12% (확정 사건만, {s['asof']} 종가)\n")
    L.append("> 탐색 연구(부록 13 정의, 부록 14 순서). 결과는 '후보'이고 매매 규칙이 아니다. 후보는 2026-10-24 뒤 새로 확정되는 "
             "고점·저점으로만 확인한다. '지금이 바닥/고점'이라고 판단하지 않는다.\n")
    inp = s["inputs"]
    L.append(f"> 코드 `events/code/event_study.py`(분석 1~4·누출 시험, 로직 그대로 부름) + `events/code/run_combo_KOSPI_12.py`"
             f"(이 조합만 실행, 사건별 적중 수, 같은 오프셋 지금 위치, 지식 기간 표시). 만든 시각 {s['made']}. 입력 sha256: events.json "
             f"{inp['events_json_sha256'][:12]}…, features_all.parquet {inp['features_all_sha256'][:12]}…, prices_close.csv "
             f"{inp['prices_close_sha256'][:12]}….\n")
    L.append(f"> 시험 범위: 특징 {s['settings']['n_features_grid']}개(검수 통과 {s['settings']['n_features_by_status']['통과']} · "
             f"설명용 {s['settings']['n_features_by_status']['설명용']}) × 오프셋 2개(0 = 사건일 08:00 마감 창, +1 = 다음 날 07:59 마감 창) "
             f"× 바닥/고점 = 가르기 칸 {2 * 2 * s['settings']['n_features_grid']}개 중 시험 {s['family']['bottom']['전체']['n_tests'] + s['family']['top']['전체']['n_tests']}개(나머지 값 부족), 앞섬 칸 시험 {s['lead_family']['bottom']['전체']['n_tests'] + s['lead_family']['top']['전체']['n_tests']}개. 섞기 {s['settings']['n_null']}번(모든 칸 같은 섞기), "
             f"위약 = 사건마다 직전 5·20일 수익률이 가장 비슷한 날 {s['settings']['placebo_pool']}일.\n")
    L.append("## 0. 결론\n")
    L.append(f"**KOSPI 12%에서는 바닥·고점 모두 포럼 특징이 가짜 사건과 '우연 이상'으로 갈리지 않았다.** 칸 하나하나로는 후보가 바닥 "
             f"{len(cb)}개·고점 {len(ct)}개 나왔지만, p<0.05 칸 수가 표지를 섞었을 때 기대되는 수와 구별되지 않는다(묶음 p 바닥 "
             f"{f2(fb['p_family'], 2)}, 고점 {f2(ft['p_family'], 2)}). 앞섬(사건 전 1~5창 극값 도달)은 후보 0개.\n")
    L.append("| 대비 | 사건 / 가짜 대표 | 통과 칸 | p<0.05 관측 | 우연 기대 | 귀무 95% | 묶음 p | 앞섬 관측 / 우연 기대 | 누출 시험 | 코드 후보 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for k, fm, lf, cd in (("bottom", fb, lb, cb), ("top", ft, lt_, ct)):
        L.append(f"| {es.TYPE_KO[k]} | {s['n'][k]['events_with_window']} / {s['n'][k]['fake_reps_with_window']} | {fm['n_tests']} | "
                 f"{fm['observed']} | {f2(fm['expected_by_chance'])} | {f2(fm['null_q95'], 0)} | {f2(fm['p_family'], 3)} | "
                 f"{lf['observed']} / {f2(lf['expected_by_chance'])} | {'의심' if s['leak'][k]['flag'] else '통과'} | "
                 + (", ".join(f"{q['feature']}@{int(q['offset']):+d}" for q in cd) or "없음") + " |")
    L.append("")
    L.append("- 바닥 후보 `vw_ln_b_median`@+1(바닥 다음 날 아침까지 글의 '조회 ÷ 마감 전 댓글'이 낮음): 사건 3/4·가짜 7/7 맞힘. "
             "바닥 다음 날 마감 전 댓글이 크게 늘어(z 평균 사건 +1.8 대 가짜 +1.2) 분모가 커진 몫이 섞여 있다. "
             "실시간 창(10/13 뒤)에는 값이 없어 지금 창에 쓸 수 없다.")
    L.append("- 고점 후보 4개는 모두 '고점 쪽이 낮음'이다: 새 작성자 몫@+1, 새벽 글 몫@+1, 추천 지니@0, 조회÷댓글@+1. +1 칸은 "
             "9/23 고점의 +1 창(K2026-09-28, 추석 120시간 창)이 판정 자격 밖이라 고점 4개(2~6월)로만 정해졌다. 가짜 날 전부로 다시 재면 "
             "BA가 새 작성자 몫 0.86을 빼고 0.63~0.68로 낮아진다.")
    L.append(f"- 미확정: 끝 극값 저점 {tail.get('date')}(창 {tail.get('win_id')}, 종가 {tail.get('close'):,.2f})은 사건에서 뺐다. "
             f"12% 확정에는 종가 {tail.get('confirm_level'):,.2f} 이상이 필요하다. 보류 가짜 바닥 "
             f"{', '.join(h['date'] for h in (s['held_near_tail'] or {}).get('fake_bottoms', [])) or '없음'}도 뺐다.\n")

    L.append("## 1. 사건과 가짜\n")
    pv = {p["date"]: p for p in json.loads((RES / "events.json").read_text("utf-8"))["assets"][ASSET]["thresholds"][THR]["pivots"]}
    for r in results:
        c = r["contrast"]
        k = c.kind
        kn = s["knowledge"][k]
        L.append(f"### {es.TYPE_KO[k]} {len(c.events)}개\n")
        L.append("| 사건일 | 창 | 종가 | 확정일(지연) | AI 라벨 지식 기간 |")
        L.append("|---|---|---|---|---|")
        for d, w in zip(c.events["date"], c.events["win_id"]):
            p = pv.get(d, {})
            L.append(f"| {d} | {w} | {p.get('close', float('nan')):,.2f} | {p.get('confirm_date')}({p.get('confirm_lag_td')}거래일) | "
                     f"{'예' if np.datetime64(d) < KNOW_END else ''} |")
        L.append("")
        L.append(f"- 가짜 묶음 대표 {len(c.fake_reps)}개(가짜 날 {len(c.fakes_all)}개, 묶음 {c.info['n_fake_runs']}개, 묶음마다 "
                 f"{'최저' if k == 'bottom' else '최고'} 종가 날): " + ", ".join(c.fake_reps["date"]) + ".")
        L.append(f"- 위약 후보 날 {r['placebo'].get('n_candidates')}일. AI 라벨 지식 기간(2026-07-01 전, 동결 §2.2) 창: 사건 "
                 f"{kn['events']['in_knowledge']}/{kn['events']['n']}, 가짜 대표 {kn['fake_reps']['in_knowledge']}/{kn['fake_reps']['n']}.\n")

    L.append("## 2. 가르기: 하나 빼고 맞히기 + 표지 섞기 1,000번\n")
    L.append("| 대비 | 묶음 | 칸 수 | p<0.05 관측 | 우연 기대 | 귀무 95% | 묶음 p |")
    L.append("|---|---|---|---|---|---|---|")
    for k in ("bottom", "top"):
        for nm in ("통과", "설명용", "전체"):
            fm = s["family"][k][nm]
            L.append(f"| {es.TYPE_KO[k]} | {nm} | {fm['n_tests']} | {fm['observed']} | {f2(fm['expected_by_chance'])} | "
                     f"{f2(fm['null_q95'], 0)} | {f2(fm['p_family'], 3)} |")
    L.append("\n- 우연 기대 = 섞기마다 p<0.05인 칸 수의 평균. 표본이 작아 BA가 몇 단계 값뿐이라 0.05 × 칸 수보다 작게 나온다(시운전과 같음).\n")
    for k in ("bottom", "top"):
        t = T[(T["kind"] == k) & T["p"].notna()].sort_values(["p", "ba"], ascending=[True, False])
        L.append(f"### {es.TYPE_KO[k]}: p 낮은 12칸\n")
        L.append("| 특징 | 오프셋 | 묶음 | BA | p | 맞힘 사건·가짜 | 방향 | z 평균 사건 / 가짜 | 가짜 날 전부 BA | 표시 |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for _, q in t.head(12).iterrows():
            L.append(f"| `{q['feature']}` {ko(q['feature'])[:20]} | {int(q['offset']):+d} | {q['bundle']} | {f2(q['ba'])} | "
                     f"{f2(q['p'], 3)} | {int(q['hit_events'])}/{int(q['n_events'])} · {int(q['hit_fakes'])}/{int(q['n_fakes'])} | "
                     f"{q['direction']} | {f2(q['mean_z_events'])} / {f2(q['mean_z_fakes'])} | "
                     f"{f2(q.get('ba_all_fake_days_run_groups'))} | {_flags(q)} |")
        L.append("")
    L.append("- 사건별 예측(어느 사건·가짜를 맞혔나)은 `run_KOSPI_12/tests.csv`의 `events_pred_correct`·`fakes_pred_correct`.")
    L.append("- 바닥 쪽 활동량(글 수·저장 글@0, z 평균 +2.1 대 가짜 +0.2~0.3)은 p 0.08로 후보가 아니다. 위약(가격이 비슷한 날) 평균 "
             "+0.6이고 위약 띠(−2.0~+3.2) 안이다 — 급락 날 글이 느는 것은 바닥이 아닌 급락 날에도 같다.\n")

    L.append("## 3. 앞섬: 사건 전 1~5창 중 직전 60창 상위·하위 10%에 닿은 비율\n")
    L.append("| 대비 | 묶음 | 칸 수 | p<0.05 관측 | 우연 기대 | 귀무 95% | 묶음 p | 후보 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for k in ("bottom", "top"):
        for nm in ("통과", "설명용", "전체"):
            fm = s["lead_family"][k][nm]
            L.append(f"| {es.TYPE_KO[k]} | {nm} | {fm['n_tests']} | {fm['observed']} | {f2(fm['expected_by_chance'])} | "
                     f"{f2(fm['null_q95'], 0)} | {f2(fm['p_family'], 3)} | {len(s['lead_candidates'][k]) if nm == '통과' else ''} |")
    L.append("")
    for k in ("bottom", "top"):
        lt = Ld[(Ld["kind"] == k) & Ld["p"].notna()].sort_values("p")
        L.append(f"### {es.TYPE_KO[k]}: 앞섬 p 낮은 8칸\n")
        L.append("| 특징 | 쪽 | 닿은 비율 사건 | 가짜 | 위약 평균 | 위약 97.5% | p(가짜 섞기) | p(위약 이항) | 표시 |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for _, q in lt.head(8).iterrows():
            L.append(f"| `{q['feature']}` | {q['side']} | {f2(q['rate_events'])} (n={int(q['n_events'])}) | "
                     f"{f2(q['rate_fakes'])} (n={int(q['n_fakes'])}) | {f2(q.get('rate_placebo_mean'))} | "
                     f"{f2(q.get('rate_placebo_q975'))} | {f2(q['p'], 3)} | {f2(q.get('p_vs_placebo'), 3)} | {_flags(q)} |")
        L.append("")
    L.append("- 앞섬 후보 조건 = p(가짜 섞기)<0.05 + 사건 비율이 위약 97.5% 분위 위 + 검수 통과 + 누출 아님. 바닥은 4개뿐이라 "
             "사건 비율 1.00도 위약 97.5%(1.00)를 넘지 못한다. 고점 쪽 p<0.05 3칸은 모두 감정 층(설명용, 지식 기간 섞임)이다.\n")

    L.append("## 4. 사건 전후 그림(분석 1, 설명용)\n")
    L.append("| 대비 | 오프셋 0: 위약 띠 밖 / 특징 | 오프셋 +1: 위약 띠 밖 / 특징 | 독립이면 기대 | 그림 |")
    L.append("|---|---|---|---|---|")
    for k in ("bottom", "top"):
        ob = s["outside_band"][k]
        L.append(f"| {es.TYPE_KO[k]} | {ob['0']['outside_band']} / {ob['0']['n_features']} {', '.join(ob['0']['features'])} | "
                 f"{ob['1']['outside_band']} / {ob['1']['n_features']} {', '.join(ob['1']['features'])} | "
                 f"{ob['0']['expected_if_independent']} | `run_KOSPI_12/figs/KOSPI_12_{k}.html` |")
    L.append("\n- 위약 띠 = 위약 날 z 평균 ± t(0.975, n−1) × SD ÷ √n. 사건이 4~5개라 띠 폭이 ±2 z 안팎으로 넓어, 띠 밖이 거의 "
             "없다는 것은 '차이가 없다'보다 '이 표본으로는 위약과 가를 수 없다'에 가깝다.")
    L.append("- 자료: `run_KOSPI_12/profiles.csv`(오프셋 −10~+10 사건·가짜·위약 평균과 띠), `profiles_by_event.csv`(사건별 z), "
             "`price_paths.csv`(사건일 종가 대비 가격 경로). 오프셋 +2 뒤는 사건 뒤 정보라 '알아챔'의 근거가 아니다.\n")

    L.append("## 5. 조회·추천 누출 시험((b) 조회 ÷ 마감 전 댓글, 부록 13)\n")
    L.append("| 대비 | 오프셋 | 사건 중앙 z (n) | 가짜 중앙 z (n) | 사건 > 가짜? | 한쪽 MW p | 판정 |")
    L.append("|---|---|---|---|---|---|---|")
    for k in ("bottom", "top"):
        lk = s["leak"][k]
        for o, v in lk["by_offset"].items():
            if "median_events" in v:
                L.append(f"| {es.TYPE_KO[k]} | {int(o):+d} | {f2(v['median_events'])} ({v['n_events']}) | {f2(v['median_fakes'])} "
                         f"({v['n_fakes']}) | {'예' if v['events_greater'] else '아니오'} | {f2(v['mannwhitney_p_greater'], 3)} | "
                         f"{'의심' if lk['flag'] else '통과'} |")
            else:
                L.append(f"| {es.TYPE_KO[k]} | {int(o):+d} | 값 부족 | | | | 의심 |")
    L.append("\n- 두 대비 모두 사건일 (b)가 가짜보다 낮아 '사건 뒤 재조명된 주목이 섞였다'는 신호가 없다 → 조회·추천 특징을 후보에서 빼지 "
             "않는다. 가짜 바닥 대표 10/15는 실시간 창이라 (b)가 없어 가짜가 7개다.")
    L.append("- 등록 시험 밖 주의: 고점 (b)는 사건 전 −5~−2창에서 사건 평균 z +0.5~+1.5(가짜 −0.4~−0.7)로 높다. 고점 직전 글이 나중에 더 "
             "많이 조회됐을 수 있다(재조명 누출 가능). 등록 규칙은 오프셋 0·+1만 보므로 판정은 바꾸지 않고 표시만 한다.\n")

    L.append("## 6. 지금 위치(설명용, 2026-10-15~10-23 창)\n")
    L.append("- 각 칸의 지금 z를 같은 오프셋의 사건·가짜 z 분포와 견주고, 전 표본으로 정한 문턱의 어느 쪽인지 적는다. 문턱 쪽이 "
             "'사건 쪽'이어도 그 특징은 우연과 구별되지 않은 칸이므로 '지금이 바닥/고점'이라는 뜻이 아니다. 코드 기본 표(모두 오프셋 0 "
             "분포와 견줌)는 `run_KOSPI_12/now.csv`.")
    L.append("- 10/13 뒤 창은 실시간 수집이다. 조회 기반 특징(소급 창만)은 결측이고, 댓글 양 특징(⚠)은 수집 방식이 달라 바로 견주면 안 된다.")
    L.append(f"- 끝 극값 {tail.get('date')}(미확정)의 오프셋 0 창 = {tail.get('win_id')}, +1 창 = K2026-10-23(10/22 08:00~10/23 07:59).\n")
    NMall = pd.DataFrame(s.get("now_matched", []))
    for k in ("bottom", "top"):
        nm = NMall[NMall["kind"] == k] if len(NMall) else NMall
        if not len(nm):
            continue
        cells = list(dict.fromkeys(zip(nm["feature"], nm["offset"])))
        tk = T[T["kind"] == k].set_index(["feature", "offset"])
        L.append(f"### {es.TYPE_KO[k]} 쪽 p<0.10 칸\n")
        L.append("| 칸(p, 사건 쪽) | " + " | ".join(sorted(nm["win_id"].unique())) + " |")
        L.append("|---|" + "---|" * nm["win_id"].nunique())
        for f, o in cells:
            q0 = nm[(nm["feature"] == f) & (nm["offset"] == o)].sort_values("win_id")
            rule = q0.iloc[0]
            side_txt = f"z {'>' if rule['rule_direction'] == 'up' else '≤'} {f2(rule['rule_cut'])}"
            p_ = tk.loc[(f, o), "p"]
            cells_txt = []
            for _, q in q0.iterrows():
                if q["z"] is None or not np.isfinite(q["z"]):
                    cells_txt.append("—")
                    continue
                mark = "●" if q["event_side"] else "○"
                cells_txt.append(f"{f2(q['z'])} {mark} ({f2(q['pct_in_events_same_offset'])}/{f2(q['pct_in_fakes_same_offset'])})")
            caut = " ⚠" if bool(rule["realtime_caution"]) else ""
            L.append(f"| `{f}`@{int(o):+d} (p {f2(p_, 3)}, 사건 쪽 {side_txt}){caut} | " + " | ".join(cells_txt) + " |")
        L.append("\n- 칸 = z ●(전 표본 문턱의 사건 쪽)/○(가짜 쪽) (같은 오프셋 사건 z 분포 안 백분위/가짜 대표 분포 안 백분위).\n")
    tl = pd.DataFrame(s.get("tail_lead", []))
    if len(tl):
        bl = Ld[(Ld["kind"] == "bottom") & (Ld["p"] < 0.10)]
        rows = []
        for _, q in bl.iterrows():
            h = tl[tl["feature"] == q["feature"]].iloc[0]
            hit = h["hit_hi"] if q["side"] == "상위 10%" else h["hit_lo"]
            tag = ", 설명용·지식 기간 섞임" if q["bundle"] in LABEL_BUNDLES else ""
            rows.append(f"`{q['feature']}`({q['side']}{tag}, 사건 {f2(q['rate_events'])}·가짜 {f2(q['rate_fakes'])}, p {f2(q['p'], 3)}): "
                        f"끝 극값 앞 10/15~10/21 창 {'닿음' if hit == 1 else ('안 닿음' if hit == 0 else '값 없음')}")
        if rows:
            L.append("- 끝 극값(10/22) 앞섬(설명용, 바닥 앞섬 p<0.10 칸): " + "; ".join(rows) + ". 10/16 창 글 수 z +4.2지만 직전 60창 "
                     "(7월 급락 포함) 백분위는 0.90 바로 아래다.\n")

    L.append("## 7. 읽는 법과 한계\n")
    L.append("- '알아챔'은 오프셋 0과 +1(사건일 다음 날 07:59 마감)까지만이다. 12% 극값은 1~10거래일 뒤에야 확정되므로, 이 칸들이 갈렸다는 것은 "
             "'그때 극값임을 알았다'가 아니라 '나중에 극값으로 판명된 날의 포럼이 가짜 날과 달랐다'는 뜻이다. 확정 뒤 정보로 극값을 "
             "알아챘다고 하지 않는다.")
    L.append("- 표본이 작다(바닥 4·고점 5, +1 칸 고점 4). 하나 빼고 맞히기 문턱은 다른 사건(뒤 사건 포함)으로 정한 교차 검증이라 "
             "앞 구간→뒤 구간 전방 검증이 아니다. 8% 결과와 다르면 둘 다 적는다(이 문서는 12%만).")
    L.append("- AI 라벨(예측·감정 층): 사건 대부분이 2026-07-01 전(라벨 모델 지식 기간)이라 동결 §2.2에 따라 이 묶음 칸은 후보로 올리지 않는다. "
             "이번에는 예측 층 칸이 p<0.05에 없었고, 감정 층(설명용) 칸만 있었다.")
    L.append("- 소급 창의 저장 글·댓글·새 작성자 몫은 수집 전에 지워진 글(결번율 약 8%)이 빠진 값이다. 지운 시점은 사건 뒤일 수 있으나 "
             "등록 누출 시험은 조회·추천만 다룬다. 글 수(번호 증가)는 지운 글을 포함한다.")
    L.append("- 산출: `run_KOSPI_12.json`, `run_KOSPI_12/tests.csv`(전 칸 + 적중 수·사건별 예측), `lead.csv`, `now.csv`, "
             "`now_matched.csv`, `tail_lead.csv`, `profiles.csv`, `profiles_by_event.csv`, `price_paths.csv`, `figs/*.html`.")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
