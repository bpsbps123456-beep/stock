"""부록 13·14 사건 연구: KOSPI 8% 조합만, 확정 사건만 모드(2026-10-22 종가로 판정한 events.json).

event_study.py의 함수를 그대로 부른다(분석 로직·문턱·귀무·위약·씨앗 불변). run_real과 다른 점은
  - 대비를 KOSPI 8%(바닥·고점)로만 거른다. 씨앗은 대비 이름(KOSPI_8_bottom/top)에서 나오므로 전체 실행과 같은 값이 나온다.
  - 하나 빼고 맞히기의 표본별 예측(사건·가짜 몇 개를 맞혔나)과 지식 기간(라벨 모델이 결과를 알 수 있는 2026-07-01 전 창) 표시,
    사건별 확정 지연(실시간성)을 더 적는다.
산출: events/results/run_KOSPI_8.json / .md, run_KOSPI_8_tests.csv, _lead.csv, _now.csv, _loo.csv, _profiles.csv,
      run_KOSPI_8_figs/*.html
실행: cd events/code; py -B run_KOSPI_8.py
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import event_study as es  # noqa: E402

ASSET, THR = "KOSPI", "8"
TAG = f"run_{ASSET}_{THR}"
RES = es.RESULTS
FIG = RES / f"{TAG}_figs"
KNOW_END = np.datetime64("2026-07-01", "D")     # 라벨 모델 지식 기간: D < 2026-07-01 창(부록 3·11)
LABEL_BUNDLES = ("예측 층", "감정 층")


def loeo_pred(v: np.ndarray, y: np.ndarray) -> np.ndarray:
    """es.loeo_ba와 같은 규칙(표본 하나씩 빼기, 균형 정확도 최대 문턱, 같은 점수 처리 동일)으로 빠진 표본의 예측을 원래 순서로 돌려준다."""
    v = np.asarray(v, float)
    y = np.asarray(y, bool)
    n = len(v)
    order = np.argsort(v, kind="mergesort")
    vs, ys = v[order], y[order]
    pred_s = np.zeros(n, bool)
    for i in range(n):
        test = np.zeros(n, bool)
        test[i] = True
        vt, yt = vs[~test], ys[~test]
        m = len(vt)
        P = float(yt.sum())
        N = m - P
        cp = np.concatenate([[0.0], np.cumsum(yt)])
        cn = np.arange(m + 1) - cp
        valid = np.ones(m + 1, bool)
        valid[1:m] = vt[1:] > vt[:-1]
        with np.errstate(invalid="ignore", divide="ignore"):
            up = 0.5 * ((P - cp) / P + cn / N)
            dn = 0.5 * (cp / P + (N - cn) / N)
        up = np.where(valid & np.isfinite(up), up, -np.inf)
        dn = np.where(valid & np.isfinite(dn), dn, -np.inf)
        best = int(np.round(np.concatenate([up, dn]), 12).argmax())
        is_up = best <= m
        j = best if is_up else best - (m + 1)
        lo = vt[j - 1] if j > 0 else -np.inf
        hi = vt[j] if j < m else np.inf
        thr = -np.inf if j == 0 else (np.inf if j == m else (lo + hi) / 2)
        upper = vs[i] > thr
        pred_s[i] = upper if is_up else not upper
    pred = np.zeros(n, bool)
    pred[order] = pred_s
    return pred


def clean(o):
    """JSON용: NaN·inf → None, numpy 값 → 파이썬 값."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, np.ndarray)):
        return [clean(v) for v in (o.tolist() if isinstance(o, np.ndarray) else o)]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if o is None or isinstance(o, str):
        return o
    if o is pd.NA or o is pd.NaT:
        return None
    return str(o)


def full_fit(v: np.ndarray, y: np.ndarray) -> dict:
    """전체 표본으로 정한 문턱·방향(설명용 — 빠진 표본 평가가 아니다)."""
    s = np.sort(np.unique(v))
    cuts = [-np.inf] + [(s[i - 1] + s[i]) / 2 for i in range(1, len(s))] + [np.inf]
    best, ch = -np.inf, None
    for d in ("up", "dn"):
        for c in cuts:
            pr = v > c if d == "up" else ~(v > c)
            ba = 0.5 * ((pr & y).sum() / y.sum() + (~pr & ~y).sum() / (~y).sum())
            if ba > best + 1e-12:
                best, ch = ba, (d, c)
    return {"direction": "z가 문턱보다 크면 사건" if ch[0] == "up" else "z가 문턱 이하이면 사건",
            "threshold_z": float(ch[1]) if np.isfinite(ch[1]) else None, "ba_in_sample": float(best)}


def _f(x, nd=2):
    return es.fmt(x if x is None else (float(x) if not isinstance(x, (int, np.integer)) else int(x)), nd)


def _pct(x):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{100 * x:.0f}%"


def write_md(out: dict, T: pd.DataFrame) -> None:
    KO = {"bottom": "바닥", "top": "고점"}
    st = out["settings"]
    md = out["mode"]
    L = [f"# 개미 지수 부록 13 사건 연구 — {out['combo']} (확정 사건만, 2026-10-22 종가)\n",
         "> 탐색 연구다. 결과는 '후보'이고 매매 규칙이 아니다. 후보가 나와도 2026-10-24 뒤 새로 확정되는 고점·저점으로만 확인한다. "
         "'지금이 바닥/고점'이라는 판단은 하지 않는다.\n",
         f"> 만든 시각 {out['made']}. 코드 `events/code/run_KOSPI_8.py`(`event_study.py` 함수 그대로, sha256 "
         f"{out['inputs']['event_study_py_sha256'][:12]}…). 입력 `events.json` {out['inputs']['events_json_sha256'][:12]}…, "
         f"`features_all.parquet` {out['inputs']['features_all_sha256'][:12]}…, `prices_close.csv` "
         f"{out['inputs']['prices_close_sha256'][:12]}…. 씨앗은 대비 이름에서 나오므로 `event_study.py run --confirmed-only`의 "
         "같은 대비와 같은 값이다.\n",
         f"> 모드(부록 14): 날짜 가드 대신 끝 극값을 뺐다 — 저점 {md['tail_excluded']['date']}(창 {md['tail_excluded']['win_id']}, "
         f"종가 {md['tail_excluded']['close']:,.2f}, 확정하려면 {md['tail_excluded']['confirm_level']:,.2f} 이상, 상태 "
         f"{md['tail_excluded']['status']}). 그 ±3거래일 보류 가짜 바닥 {', '.join(md['held_fakes_excluded']['fake_bottoms']) or '없음'}"
         f"도 뺐다. 사용한 확정 사건의 마지막 확정일 {md['last_confirm_date']}.\n",
         f"> 격자: 특징 {st['grid_features']}개(통과 {st['grid_pass']} · 설명용 {st['grid_explain']}) × 오프셋 2개(0, +1) = "
         f"{2 * st['grid_features']}칸(통과 {2 * st['grid_pass']}칸). 섞기 {st['n_null']}번, 위약 = 사건마다 가격이 비슷한 날 "
         f"{st['placebo_pool']}일, 앞섬 = 사건 전 1~5창.\n",
         "## 0. 결론\n",
         "| 대비 | 사건 | 가짜 묶음 대표(전체 가짜 날) | 가르기 p<0.05 관측 / 우연 기대(95% 상한) — 통과 칸 | 같은 값 — 전체 칸 | "
         "앞섬 관측 / 우연 기대(95% 상한) — 통과 칸 | 누출 시험 | 후보 |",
         "|---|---|---|---|---|---|---|---|"]
    for c in out["contrasts"]:
        fp, fa, lp = c["family"]["통과"], c["family"]["전체"], c["lead_family"]["통과"]
        cand = ", ".join(f"{d['feature']}@{int(d['offset']):+d}" for d in c["candidates"]) or "없음"
        lc = ", ".join(f"{d['feature']}({d['side']})" for d in c["lead_candidates"])
        L.append(f"| {KO[c['kind']]} | {c['n']['events_with_window']} | {c['n']['fake_reps_with_window']}({c['n_fakes_all_days']}) | "
                 f"{fp['observed']} / {_f(fp['expected_by_chance'])}({_f(fp['null_q95'], 0)}) | "
                 f"{fa['observed']} / {_f(fa['expected_by_chance'])}({_f(fa['null_q95'], 0)}) | "
                 f"{lp['observed']} / {_f(lp['expected_by_chance'])}({_f(lp['null_q95'], 0)}) | "
                 f"{'의심' if c['leak']['flag'] else '통과'} | 가르기 {cand}; 앞섬 {lc or '없음'} |")
    qb = {c: T[(T["contrast"] == c) & T["status"].eq("시험")]["null_ba_q95"] for c in T["contrast"].unique()}
    concl = []
    for c in out["contrasts"]:
        fp, lp = c["family"]["통과"], c["lead_family"]["통과"]
        nc, nl = len(c["candidates"]), len(c["lead_candidates"])
        concl.append(f"{KO[c['kind']]}: 가르기 후보 {nc}개(통과 칸 p<0.05 관측 {fp['observed']} vs 우연 기대 "
                     f"{_f(fp['expected_by_chance'])}, 묶음 p {_f(fp['p_family'], 3)}), 앞섬 후보 {nl}개(관측 {lp['observed']} vs "
                     f"{_f(lp['expected_by_chance'])}, 묶음 p {_f(lp['p_family'], 3)})")
    L += ["",
          "- " + "; ".join(concl) + ".",
          "- 검정력 한계: 사건 8~9개 대 가짜 대표 6~8개라 하나 빼고 맞히기 BA가 귀무 95%("
          + ", ".join(f"{k.split('_')[-1].replace('bottom', '바닥').replace('top', '고점')} {v.min():.2f}~{v.max():.2f}"
                      for k, v in qb.items())
          + ")를 넘어야 p < 0.05다. 시운전(`results/dryrun.md`)에서 이 크기는 1 SD 차이를 절반 넘게 놓쳤다. '후보 없음'은 "
            "'포럼에 바닥·고점 정보가 없다'는 증명이 아니라 '이 표본으로는 우연과 구별되지 않는다'는 뜻이다.",
          "- 실시간성: 오프셋 0 창은 사건일 08:00, +1 창은 다음 거래일 08:00(W_KR(D+1) 마감) 전에 닫힌다. 그런데 극값은 "
          "아래 표처럼 1~14거래일 뒤에야 확정된다. 그래서 이 창들의 값은 '그날이 바닥·고점인 줄 알고 쓴 글'이 아니며, 확정 이후 "
          "정보로 극값을 알아챘다고 하지 않는다.\n",
          "## 1. 사건과 실시간성\n",
          "| 종류 | 극값일 | 종가 | 오프셋 0 창(마감 KST) | +1 창(마감 KST) | 확정일 | 확정 지연(거래일) | 라벨 지식 기간 |",
          "|---|---|---|---|---|---|---|---|"]
    for c in out["contrasts"]:
        for e in c["events"]:
            L.append(f"| {KO[c['kind']]} | {e['date']} | {e['close']:,.2f} | {e['win_id_offset0']}({e['offset0_close_kst'][:16]}) | "
                     f"{e['win_id_offset1']}({(e['offset1_close_kst'] or '')[:16]}) | {e['confirm_date']} | {e['confirm_lag_td']} | "
                     f"{'예' if e['knowledge_period'] else ''} |")
    L += ["", "| 가짜 대표 | 날짜(묶음) | 종가 | 라벨 지식 기간 |", "|---|---|---|---|"]
    for c in out["contrasts"]:
        for f in c["fake_reps"]:
            L.append(f"| 가짜 {KO[c['kind']]} | {f['date']}(묶음 {f['run']}) | {f['close']:,.2f} | {'예' if f['knowledge_period'] else ''} |")
    L += ["", "- '라벨 지식 기간' = D < 2026-07-01 창. 예측 층·감정 층 라벨을 붙인 AI가 그 기간 결과를 알 수 있다(부록 3·11). 바닥 사건 "
          f"{out['contrasts'][0]['knowledge_period_counts']['events']}/{len(out['contrasts'][0]['events'])}개, 고점 사건 "
          f"{out['contrasts'][1]['knowledge_period_counts']['events']}/{len(out['contrasts'][1]['events'])}개가 이 기간이다. 라벨 층 칸 "
          "후보는 7/1 뒤 사건만으로 따로 확인해야 한다(라벨 층 후보: "
          + (", ".join(f"{c['name']} {d['feature']}@{int(d['offset']):+d}" for c in out["contrasts"] for d in c["candidates"]
                       if d.get("label_layer")) or "없음") + ").",
          "- 가짜 대표 중 '지금 위치' 기간(10/15~10/23) 안의 날: "
          + (", ".join(f"가짜 {KO[c['kind']]} {f['date']}" for c in out["contrasts"] for f in c["fake_reps"]
                       if "2026-10-15" <= f["date"] <= "2026-10-23") or "없음")
          + ". 끝 극값보다 종가가 높아(바닥) 가짜로 들어간 날이라, 지금 위치 표의 그 창은 표본 안의 값이다.\n",
          "## 2. 가르기(분석 2): 바닥 대 가짜 바닥, 고점 대 가짜 고점\n",
          "| 대비 | 묶음 | 칸 수 | p<0.05 관측 | 우연 기대 | 귀무 95% | 묶음 p |", "|---|---|---|---|---|---|---|"]
    for c in out["contrasts"]:
        for nm in ("통과", "설명용", "전체"):
            f = c["family"][nm]
            L.append(f"| {KO[c['kind']]} | {nm} | {f['n_tests']} | {f['observed']} | {_f(f['expected_by_chance'])} | "
                     f"{_f(f['null_q95'], 0)} | {_f(f['p_family'], 3)} |")
    L += ["", "p가 가장 작은 통과 칸(후보 아님 — 모두 p ≥ 0.05):\n",
          "| 대비 | 특징@오프셋 | 하나 빼고 맞힘(사건 / 가짜) | BA | 섞기 p | 방향 | 가짜 날 전부(묶음째 빼기) BA |",
          "|---|---|---|---|---|---|---|"]
    for c in out["contrasts"]:
        sub = T[(T["contrast"] == c["name"]) & T["status"].eq("시험") & T["status_feature"].eq("통과")].sort_values("p").head(5)
        for _, r in sub.iterrows():
            L.append(f"| {KO[c['kind']]} | {r['ko']} `{r['feature']}`@{int(r['offset']):+d} | {int(r['loo_hit_events'])}/{int(r['n_events'])} / "
                     f"{int(r['loo_hit_fakes'])}/{int(r['n_fakes'])} | {r['ba']:.3f} | {r['p']:.3f} | {r['direction']} | "
                     f"{_f(r.get('ba_all_fake_days_run_groups'), 3)} |")
    L += ["", "## 3. 사건 전후(분석 1, 설명용)\n",
          "| 대비 | 오프셋 0: 위약 띠 밖 / 특징(독립이면 기대) | +1: 띠 밖 / 특징 | 띠 밖 특징(0 / +1) | 그림 |", "|---|---|---|---|---|"]
    for c in out["contrasts"]:
        ob = c["outside_band"]
        L.append(f"| {KO[c['kind']]} | {ob['0']['outside_band']} / {ob['0']['n_features']}({ob['0']['expected_if_independent']}) | "
                 f"{ob['1']['outside_band']} / {ob['1']['n_features']} | {', '.join(ob['0']['features']) or '—'} / "
                 f"{', '.join(ob['1']['features']) or '—'} | `{c['figure']}` |")
    L += ["", "| 대비 | 평균 가격 경로(사건일 종가 대비 %): −10 / −1 / +1 / +5 / +10 | 가짜 대표 | 위약 |", "|---|---|---|---|"]
    for c in out["contrasts"]:
        pm = c["price_path_mean_pct"]
        s = lambda a: " / ".join(f"{a[i]:+.1f}" if a[i] is not None and np.isfinite(a[i]) else "—" for i in (0, 9, 11, 15, 20))  # noqa: E731
        L.append(f"| {KO[c['kind']]} | {s(pm['events'])} | {s(pm['fakes'])} | {s(pm['placebo'])} |")
    L += ["", "- 띠 = 위약 날 z 평균 ± t × SD ÷ √(사건 수). 띠 밖 특징 수는 독립 가정 기대(0.05 × 특징 수)와 견주는 설명용이고, 판정은 2절 섞기 귀무로 한다.\n",
          "## 4. 앞섬(분석 3): 사건 전 1~5창 중 직전 60창 상위·하위 10%에 닿은 비율\n",
          "| 대비 | 묶음 | 칸 수 | p<0.05 관측 | 우연 기대 | 귀무 95% | 묶음 p |", "|---|---|---|---|---|---|---|"]
    for c in out["contrasts"]:
        for nm in ("통과", "설명용", "전체"):
            f = c["lead_family"][nm]
            L.append(f"| {KO[c['kind']]} | {nm} | {f['n_tests']} | {f['observed']} | {_f(f['expected_by_chance'])} | "
                     f"{_f(f['null_q95'], 0)} | {_f(f['p_family'], 3)} |")
    Ld = pd.read_csv(RES / f"{TAG}_lead.csv")
    L += ["", "p가 가장 작은 칸(통과 3 + p<0.05인 설명용, 모두 후보 아님):\n",
          "| 대비 | 특징(쪽) | 묶음 | 사건 닿음 | 가짜 닿음 | 위약 평균 | 위약 97.5% | 섞기 p | 위약 대비 p |", "|---|---|---|---|---|---|---|---|---|"]
    for c in out["contrasts"]:
        g = Ld[(Ld["contrast"] == c["name"]) & Ld["status"].eq("시험")]
        sub = pd.concat([g[g["status_feature"].eq("통과")].sort_values("p").head(3),
                         g[g["status_feature"].eq("설명용") & (g["p"] < es.ALPHA)]])
        for _, r in sub.iterrows():
            L.append(f"| {KO[c['kind']]} | {r['ko']} `{r['feature']}`({r['side']}) | {r['status_feature']} | "
                     f"{_pct(r['rate_events'])}({int(r['n_events'])}) | {_pct(r['rate_fakes'])}({int(r['n_fakes'])}) | "
                     f"{_pct(r['rate_placebo_mean'])} | {_pct(r['rate_placebo_q975'])} | {r['p']:.3f} | {r['p_vs_placebo']:.3f} |")
    L += ["", "- 앞섬 후보 조건 = 섞기 p < 0.05 + 사건 닿음 비율 > 위약 97.5% + 통과 + 누출 의심 아님."]
    for c in out["contrasts"]:
        nm, f10 = c["near_miss_p_lt_0_10"], c["family_alpha_0_10"]
        items = ([f"가르기 `{d['feature']}`@{int(d['offset']):+d}(p {d['p']:.3f}, 맞힘 {int(d['loo_hit_events'])}/{int(d['n_events'])}·"
                  f"{int(d['loo_hit_fakes'])}/{int(d['n_fakes'])})" for d in nm["cls"]] +
                 [f"앞섬 `{d['feature']}`({d['side']}, 사건 {_pct(d['rate_events'])} · 가짜 {_pct(d['rate_fakes'])} · 위약 "
                  f"{_pct(d['rate_placebo_mean'])}, p {d['p']:.3f})" for d in nm["lead"]])
        fc, fl = f10["cls_pass"], f10["lead_pass"]
        within = fc["observed"] <= fc["null_q95"] and fl["observed"] <= fl["null_q95"]
        L.append(f"- {KO[c['kind']]} 근접 칸(통과, 0.05 ≤ p < 0.10 — 후보 아님): {'; '.join(items) or '없음'}. 같은 섞기로 센 p < 0.10 "
                 f"칸: 가르기 관측 {fc['observed']} · 우연 기대 {_f(fc['expected_by_chance'])}(95% 상한 {_f(fc['null_q95'], 0)}), 앞섬 관측 "
                 f"{fl['observed']} · 우연 기대 {_f(fl['expected_by_chance'])}(95% 상한 {_f(fl['null_q95'], 0)}) → "
                 + ("근접 칸 수도 우연 범위 안이다." if within else "근접 칸 수가 우연 95% 상한을 넘는 쪽이 있다(그래도 후보 아님).")
                 + " 10/24 뒤 새 사건에서 볼 관찰 목록으로만 남긴다.")
    L.append("")
    L += [
          "## 5. 조회·추천 누출 시험((b) 조회 ÷ 마감 전 댓글, `vw_ln_b_median` z)\n",
          "| 대비 | 오프셋 | 사건 중앙 z(수) | 가짜 중앙 z(수) | 사건 > 가짜 | 한쪽 MW p | 판정 |", "|---|---|---|---|---|---|---|"]
    for c in out["contrasts"]:
        for k, v in c["leak"]["by_offset"].items():
            L.append(f"| {KO[c['kind']]} | {int(k):+d} | {_f(v.get('median_events'))}({v['n_events']}) | {_f(v.get('median_fakes'))}"
                     f"({v['n_fakes']}) | {'예' if v.get('events_greater') else '아니오'} | {_f(v.get('mannwhitney_p_greater'), 3)} | "
                     f"{'의심' if c['leak']['flag'] else '통과'} |")
    leak_s = []
    for c in out["contrasts"]:
        vb = [v for v in c["leak"]["by_offset"].values()]
        leak_s.append(f"{KO[c['kind']]} " + ("의심 — 조회·추천 특징 전부 후보에서 뺌" if c["leak"]["flag"] else
                                            "통과(사건일 (b) 중앙값이 두 오프셋 모두 가짜 이하 → '사건 뒤 재조명' 신호 없음)"))
    any_cand = any(c["candidates"] for c in out["contrasts"])
    L += ["", "- 판정: " + "; ".join(leak_s) + ". 규칙은 점추정이라 주입 없이도 65~70% 켜지는 보수적 시험이다(시운전). "
          "조회·추천 원값 특징(실시간 비교 불가)은 소급 창(10/13 전)만 쓴다.\n",
          "## 6. 지금 위치(분석 4, 설명용 — '지금이 바닥/고점'이라고 단정하지 않는다)\n",
          ("후보 칸의 10/15~10/23 창이다." if any_cand else "후보가 없어 대비마다 p가 가장 작은 통과 칸 3개를 보인다(후보 아님).")
          + " 백분위 = 그 창 z가 사건·가짜·위약의 오프셋 0 z 분포에서 차지하는 자리.\n",
          "| 대비 | 특징(칸) | 창 | z | 직전 60창 백분위 | 사건 안 | 가짜 안 | 위약 안 | 실시간 주의 |", "|---|---|---|---|---|---|---|---|---|"]
    allnan = {}
    for r in out["now"]:
        allnan.setdefault((r["contrast"], r["feature"]), []).append(r["z"] is None or not np.isfinite(r["z"]))
    for r in out["now"]:
        if all(allnan[(r["contrast"], r["feature"])]):
            continue
        L.append(f"| {KO['bottom' if r['contrast'].endswith('bottom') else 'top']} | `{r['feature']}`@{r['cell_offset']:+d} | {r['win_id']} | "
                 f"{_f(r['z'])} | {_pct(r['pct60'])} | {_pct(r['pct_in_events'])} | {_pct(r['pct_in_fakes'])} | "
                 f"{_pct(r['pct_in_placebo'])} | {'예' if r['realtime_caution'] else ''} |")
    gone = sorted({f"`{f}`({KO['bottom' if c.endswith('bottom') else 'top']})" for (c, f), v in allnan.items() if all(v)})
    if gone:
        L.append(f"\n- {', '.join(gone)}: 10/15~10/23 창이 모두 결측(실시간 수집 창이라 소급 창끼리만 비교하는 특징).")
    if out["tail_same_offset"]:
        L += ["", "끝 극값(미확정 저점 2026-10-22) 창을 같은 오프셋의 사건·가짜 분포와 견줌(사건으로 쓰지 않음):\n",
              "| 특징@오프셋 | 창 | z | 사건 안 백분위(사건 중앙 z) | 가짜 안 백분위(가짜 중앙 z) |", "|---|---|---|---|---|"]
        for r in out["tail_same_offset"]:
            L.append(f"| `{r['feature']}`@{r['offset']:+d} | {r['win_id']} | {_f(r['z'])} | {_pct(r['pct_in_events_same_offset'])}"
                     f"({_f(r['events_median_z'])}) | {_pct(r['pct_in_fakes_same_offset'])}({_f(r['fakes_median_z'])}) |")
    L += ["", "- 10/14 뒤 창은 실시간 수집 창이다. 조회·추천 원값 특징은 결측(소급 창끼리만 비교), 댓글 양 특징은 수집 방식이 달라 주의한다.",
          "- 위 칸들은 사건과 가짜를 가르지 못한 특징이라, 지금 값이 사건 분포 어디에 있든 바닥·고점 여부를 말해 주지 않는다.\n",
          "## 산출\n",
          f"- `{TAG}.json`, `{TAG}.md`, `{TAG}_tests.csv`(칸별 BA·p·하나 빼고 맞힘), `{TAG}_lead.csv`, `{TAG}_now.csv`, "
          f"`{TAG}_loo.csv`(p<0.05 칸·누출 특징의 표본별 예측), `{TAG}_profiles.csv`(오프셋 −10~+10 평균 z·위약 띠), "
          f"`{TAG}_figs/KOSPI_8_bottom.html`, `{TAG}_figs/KOSPI_8_top.html`(사건 전후 그림)",
          f"- 걸린 시간 {out['seconds']:.0f}초."]
    (RES / f"{TAG}.md").write_text("\n".join(L) + "\n", "utf-8")


def main() -> int:
    t0 = datetime.now(es.KST).replace(microsecond=0)
    ev_path = RES / "events.json"
    feat_path = RES / "features_all.parquet"
    ev_raw = json.loads(ev_path.read_text("utf-8"))
    ev, mode = es.confirmed_only_events(ev_raw, es.CONFIRMED_ASOF)
    F, meta = es.load_features(feat_path, RES / "features_all_meta.json")
    prices = pd.read_csv(RES / "prices_close.csv")
    contrasts = [c for c in es.load_contrasts(ev) if c.asset == ASSET and c.threshold == THR]
    assert [c.kind for c in contrasts] == ["bottom", "top"], [c.name for c in contrasts]
    panel = es.build_panel(F, meta, "kr")
    close = es.price_series(prices, ASSET)
    g = F[F["market"] == "kr"].sort_values("close_kst").reset_index(drop=True)
    know = dict(zip(g["win_id"], g["knowledge_period"].astype(bool)))
    close_kst = dict(zip(g["win_id"], g["close_kst"].astype(str)))
    th = ev["assets"][ASSET]["thresholds"][THR]
    piv = {p["date"]: p for p in th["pivots"]}
    FIG.mkdir(parents=True, exist_ok=True)

    out = {"made": t0.isoformat(), "combo": f"{ASSET} {THR}%", "mode": mode["contrasts"][f"{ASSET}_{THR}"] | {
        "asof": mode["asof"], "date_guard": mode["date_guard"]},
        "inputs": {"events_json_sha256": es.sha(ev_path), "features_all_sha256": es.sha(feat_path),
                   "prices_close_sha256": es.sha(RES / "prices_close.csv"),
                   "event_study_py_sha256": es.sha(HERE / "event_study.py")},
        "settings": {"n_null": es.N_NULL, "cls_offsets": list(es.CLS_OFFSETS), "lead": list(es.LEAD),
                     "placebo_pool": es.PLACEBO_POOL, "alpha": es.ALPHA, "grid_features": len(panel.feats),
                     "grid_pass": sum(meta[f]["status"] == "통과" for f in panel.feats),
                     "grid_explain": sum(meta[f]["status"] == "설명용" for f in panel.feats)},
        "contrasts": []}
    tests_all, lead_all, now_all, loo_all, prof_all = [], [], [], [], []

    for c in contrasts:
        r = es.run_contrast(c, panel, close, n_null=es.N_NULL)
        (FIG / f"{c.name}.html").write_text(es.figure_html(
            r, panel, f"확정 사건만(2026-10-22 종가 판정, 끝 극값 {c.info['tail_unconfirmed']} 미확정은 뺌). "), "utf-8")
        cls, tb = r["cls"], r["table"].copy()
        base, y = cls["base"], cls["y"]
        win_of_base = panel.win_id[base]
        # 표본별 하나 빼고 맞히기
        hit_e, hit_f, ba_chk, fit_rows = [], [], [], []
        for _, row in tb.iterrows():
            if row["status"] != "시험":
                hit_e.append(np.nan); hit_f.append(np.nan); ba_chk.append(np.nan); fit_rows.append({})
                continue
            fi, k = panel.feats.index(row["feature"]), int(row["offset"])
            v = es.gather(panel.Z, base, [k])[:, 0, fi]
            ok = np.isfinite(v)
            pr = loeo_pred(v[ok], y[ok])
            yy = y[ok]
            he, hf = int((pr & yy).sum()), int((~pr & ~yy).sum())
            hit_e.append(he); hit_f.append(hf)
            ba_chk.append(0.5 * (he / yy.sum() + hf / (~yy).sum()))
            fit_rows.append(full_fit(v[ok], yy))
            if row["p"] < es.ALPHA or row["feature"] == es.LEAK_FEATURE:
                for w, vv, yv, pv in zip(win_of_base[ok], v[ok], yy, pr):
                    loo_all.append({"contrast": c.name, "feature": row["feature"], "offset": k, "win_id": w,
                                    "D": w[1:], "label": "사건" if yv else "가짜", "z": vv, "loo_pred_event": bool(pv),
                                    "loo_correct": bool(pv == yv), "knowledge_period": bool(know.get(w, False))})
        tb["loo_hit_events"], tb["loo_hit_fakes"], tb["ba_recheck"] = hit_e, hit_f, ba_chk
        bad = tb["status"].eq("시험") & ~np.isclose(tb["ba"].astype(float), tb["ba_recheck"].astype(float), atol=1e-9)
        assert not bad.any(), tb.loc[bad, ["feature", "offset", "ba", "ba_recheck"]]
        tb["fit_direction"] = [d.get("direction") for d in fit_rows]
        tb["fit_threshold_z"] = [d.get("threshold_z") for d in fit_rows]
        tb["ko"] = tb["feature"].map(lambda f: meta[f]["ko"])
        tb["label_layer"] = tb["bundle"].isin(LABEL_BUNDLES)
        tests_all.append(tb.assign(contrast=c.name))
        lt = r["lead"].copy()
        lt["ko"] = lt["feature"].map(lambda f: meta[f]["ko"])
        lead_all.append(lt.assign(contrast=c.name))
        nw = r["now"].copy()
        if len(nw):
            now_all.append(nw.assign(contrast=c.name))
        pr_ = r["profiles"]
        for fi, f in enumerate(panel.feats):
            for j, k in enumerate(es.OFFSETS):
                prof_all.append({"contrast": c.name, "feature": f, "offset": int(k), "ev_mean": pr_["ev_mean"][j, fi],
                                 "ev_n": int(pr_["ev_n"][j, fi]), "fk_mean": pr_["fk_mean"][j, fi],
                                 "pl_mean": pr_["pl_mean"][j, fi], "pl_lo": pr_["pl_lo"][j, fi],
                                 "pl_hi": pr_["pl_hi"][j, fi]})
        # 가격 경로 평균(그림 자료)
        pp = r["price_paths"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            price_mean = {k: (np.nanmean(v, 0) * 100).round(2).tolist() for k, v in pp.items() if len(v)}
        # 사건 표(실시간성: 창 마감·확정일·지연)
        pk = "low" if c.kind == "bottom" else "high"
        ev_rows = []
        for d, w in zip(c.events["date"], c.events["win_id"]):
            p = piv[d]
            i = panel.pos.get(w, -1)
            nxt = panel.win_id[i + 1] if 0 <= i < len(panel.win_id) - 1 else None
            ev_rows.append({"date": d, "close": p["close"], "win_id_offset0": w, "offset0_close_kst": close_kst.get(w),
                            "win_id_offset1": nxt, "offset1_close_kst": close_kst.get(nxt), "confirm_date": p["confirm_date"],
                            "confirm_lag_td": p["confirm_lag_td"], "amp_in": p["amp_in"],
                            "knowledge_period": bool(know.get(w, False))})
        fk_rows = [{"date": d, "win_id": w, "run": int(rn), "close": cl, "knowledge_period": bool(know.get(w, False))}
                   for d, w, rn, cl in zip(c.fake_reps["date"], c.fake_reps["win_id"], c.fake_reps["run"], c.fake_reps["close"])]
        cand = tb[tb["candidate"]]
        tested_pass = tb[tb["status"].eq("시험") & tb["status_feature"].eq("통과")].sort_values("p")
        # 근접 칸(0.05 ≤ p < 0.10)을 말할 때 쓸 우연 기대(같은 섞기, α = 0.10) — 후보 판정에는 쓰지 않는다
        pass_mask = tb["status_feature"].eq("통과").to_numpy()
        ld_full = es.lead_analysis(panel, cls, r["pools"])
        lpass = ld_full["table"]["feature"].map(lambda f: meta[f]["status"]).eq("통과").to_numpy()
        assert np.allclose(np.nan_to_num(ld_full["p_obs"], nan=-1), np.nan_to_num(lt["p"].to_numpy(float), nan=-1))
        fam10 = {"cls_pass": es.family_count(np.where(pass_mask, cls["p_obs"], np.nan), cls["T_null"], alpha=0.10),
                 "lead_pass": es.family_count(np.where(lpass, ld_full["p_obs"], np.nan), ld_full["T_null"], alpha=0.10)}
        near = {"cls": tested_pass[(tested_pass["p"] >= es.ALPHA) & (tested_pass["p"] < 0.10)].reindex(columns=[
                    "feature", "ko", "offset", "ba", "p", "direction", "loo_hit_events", "loo_hit_fakes", "n_events",
                    "n_fakes"]).to_dict("records"),
                "lead": lt[lt["status"].eq("시험") & lt["status_feature"].eq("통과") & (lt["p"] >= es.ALPHA) &
                           (lt["p"] < 0.10)].reindex(columns=[
                    "feature", "ko", "side", "rate_events", "rate_fakes", "rate_placebo_mean", "rate_placebo_q975", "p",
                    "p_vs_placebo", "n_events", "n_fakes"]).to_dict("records")}
        lc = lt[lt["candidate"]]
        rec = {
            "name": c.name, "kind": c.kind, "n": r["n"], "events": ev_rows, "fake_reps": fk_rows,
            "n_fakes_all_days": int(len(c.fakes_all)), "held_near_tail": c.info["held_near_tail"],
            "tail_unconfirmed": c.info["tail_unconfirmed"], "pivot_dates_excluded_from_placebo": c.pivot_dates,
            "samples_used": cls["n_samples"],
            "knowledge_period_counts": {"events": sum(e["knowledge_period"] for e in ev_rows),
                                        "fake_reps": sum(f["knowledge_period"] for f in fk_rows)},
            "family": r["family"], "lead_family": r["lead_family"], "leak": r["leak"],
            "outside_band": r["outside_band"], "placebo": r["placebo"],
            "candidates": cand.reindex(columns=["feature", "ko", "offset", "ba", "p", "direction", "mean_z_events",
                                                "mean_z_fakes", "loo_hit_events", "loo_hit_fakes", "n_events", "n_fakes",
                                                "ba_all_fake_days_run_groups", "bundle", "label_layer"]).to_dict("records"),
            "p_lt_alpha_all_cells": tb[tb["p"] < es.ALPHA].reindex(columns=[
                "feature", "ko", "offset", "status_feature", "bundle", "ba", "p", "direction", "loo_hit_events",
                "loo_hit_fakes", "n_events", "n_fakes", "leak_suspect", "candidate"]).to_dict("records"),
            "top_pass_cells": tested_pass.head(6).reindex(columns=[
                "feature", "ko", "offset", "ba", "p", "direction", "loo_hit_events", "loo_hit_fakes", "n_events",
                "n_fakes", "leak_suspect"]).to_dict("records"),
            "lead_candidates": lc.reindex(columns=["feature", "ko", "side", "rate_events", "rate_fakes",
                                                   "rate_placebo_mean", "rate_placebo_q975", "p", "p_vs_placebo"]
                                          ).to_dict("records"),
            "lead_p_lt_alpha": lt[lt["p"] < es.ALPHA].reindex(columns=[
                "feature", "ko", "side", "status_feature", "rate_events", "rate_fakes", "rate_placebo_mean",
                "rate_placebo_q975", "p", "p_vs_placebo", "leak_suspect", "candidate"]).to_dict("records"),
            "near_miss_p_lt_0_10": near, "family_alpha_0_10": fam10,
            "price_path_mean_pct": price_mean,
            "figure": f"{TAG}_figs/{c.name}.html"}
        out["contrasts"].append(rec)

    T = pd.concat(tests_all)
    T.to_csv(RES / f"{TAG}_tests.csv", index=False, encoding="utf-8-sig")
    pd.concat(lead_all).to_csv(RES / f"{TAG}_lead.csv", index=False, encoding="utf-8-sig")
    NOW = pd.concat(now_all) if now_all else pd.DataFrame()
    NOW.to_csv(RES / f"{TAG}_now.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(loo_all).to_csv(RES / f"{TAG}_loo.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(prof_all).to_csv(RES / f"{TAG}_profiles.csv", index=False, encoding="utf-8-sig")

    # 지금 위치: 후보 + (후보가 없으면) 통과 묶음 p 상위 3칸, 10/15~10/23 창 전부
    now_rows = []
    for rec in out["contrasts"]:
        picks = [(d["feature"], int(d["offset"]), "후보") for d in rec["candidates"]]
        if not picks:
            picks = [(d["feature"], int(d["offset"]), "후보 아님(p 상위)") for d in rec["top_pass_cells"][:3]]
        for f, k, why in picks:
            q = NOW[(NOW["contrast"] == rec["name"]) & (NOW["feature"] == f)].sort_values("win_id")
            for _, s in q.iterrows():
                now_rows.append({"contrast": rec["name"], "feature": f, "cell_offset": k, "why": why, "win_id": s["win_id"],
                                 "z": s["z"], "pct60": s["pct60"], "pct_in_events": s["pct_in_events"],
                                 "pct_in_fakes": s["pct_in_fakes"], "pct_in_placebo": s["pct_in_placebo"],
                                 "realtime_window": bool(s["realtime_window"]),
                                 "realtime_caution": bool(s["realtime_caution"])})
    out["now"] = now_rows
    # 끝 극값(미확정) 창을 같은 오프셋의 사건·가짜 분포와 견줌(설명용 — 사건으로 쓰지 않는다)
    tail_cmp = []
    for rec in out["contrasts"]:
        tw = rec["tail_unconfirmed"]
        tail_win = mode["contrasts"][f"{ASSET}_{THR}"]["tail_excluded"]["win_id"] if tw else None
        ti = panel.pos.get(tail_win, -1) if tail_win else -1
        if ti < 0 or (rec["kind"] == "bottom") != (mode["contrasts"][f"{ASSET}_{THR}"]["tail_excluded"]["kind"] == "low"):
            continue
        cells = [(d["feature"], int(d["offset"])) for d in (rec["candidates"] or rec["top_pass_cells"][:3])]
        ev_pos = es.positions(panel, [e["win_id_offset0"] for e in rec["events"]])
        fk_pos = es.positions(panel, [f["win_id"] for f in rec["fake_reps"]])
        for f, k in cells:
            fi = panel.feats.index(f)
            z = panel.Z[ti + k, fi] if ti + k < len(panel.win_id) else np.nan
            E = es.gather(panel.Z, ev_pos, [k])[:, 0, fi]
            Fk = es.gather(panel.Z, fk_pos, [k])[:, 0, fi]

            def pr(ref):
                ref = ref[np.isfinite(ref)]
                return float(((ref < z).sum() + 0.5 * (ref == z).sum()) / len(ref)) if (len(ref) and np.isfinite(z)) else None
            tail_cmp.append({"contrast": rec["name"], "tail_date": tw, "feature": f, "offset": k,
                             "win_id": panel.win_id[ti + k] if ti + k < len(panel.win_id) else None,
                             "z": z, "pct_in_events_same_offset": pr(E), "pct_in_fakes_same_offset": pr(Fk),
                             "events_median_z": float(np.nanmedian(E)), "fakes_median_z": float(np.nanmedian(Fk)),
                             "realtime_window": bool(panel.realtime[ti + k]) if ti + k < len(panel.win_id) else None})
    out["tail_same_offset"] = tail_cmp
    out["seconds"] = (datetime.now(es.KST) - t0).total_seconds()
    write_md(out, T)
    (RES / f"{TAG}.json").write_text(json.dumps(clean(out), ensure_ascii=False, indent=1, allow_nan=False), "utf-8")
    print(json.dumps({"seconds": out["seconds"], "contrasts": [
        {"name": x["name"], "n": x["n"], "family_pass": x["family"]["통과"], "family_all": x["family"]["전체"],
         "lead_pass": x["lead_family"]["통과"], "leak": x["leak"]["flag"],
         "cands": [(d["feature"], d["offset"], round(d["p"], 4)) for d in x["candidates"]],
         "lead_cands": [(d["feature"], d["side"], round(d["p"], 4)) for d in x["lead_candidates"]]}
        for x in out["contrasts"]]}, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
