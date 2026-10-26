"""부록 15 등록 판정(부록 12 4칸) 실행 — final/code/judge.py를 실제 모드로 한 번 돌린다(2026-10-26).

judge.py는 한 줄도 고치지 않고 불러 쓴다(실행 전 SHA-256이 등록값과 다르면 멈춤). 이 감싸기가 바꾸는 것:
  경로  RES·FINAL 상수와 load_windows·load_features·load_pred_features의 기본 인자(res=RES, 정의 때 묶임)만
        final/run2026_10_26/results(·pred_all)로 바꾼다.
  예측 층 표  부록 11 #4대로 judge가 정식 표 pred_windows_full.parquet를 읽게, 같은 폴더에 pred_windows.parquet로
        바이트 사본을 둔다(judge.load_pred_features가 찾는 이름).
  칸 등록(부록 12, 가격 판정 전 결정)  H35·H46·H31·H68(목표 10일 이상)의 role을 main → expl로 바꾼다. 판정 칸은
        T1 Holm H03·H08·H70, T2 H66 1칸(BH = Holm)이다. H67은 등록 그대로(main)이고 가격 전 점검에서 자동 설명용이라
        시험 칸이 아니다(시험 칸이 되면 멈춘다). 계산·문턱·순차·원형 이동 로직은 그대로다.
  _late 강건판(부록 11 #1, 판정 칸 수에 넣지 않음)  7/1 이후 라벨만 쓴 판을 설명용 칸으로 더해 같은 CellRun으로 잰다.
        H70_late = P_share_rz8_late(pred_windows_full, 7/1 이후 창·단위만으로 rz8), H66_late = P̄에서 7/1 전 창을 결측으로
        둔 판(M0 확장 OLS 120창을 못 채우면 X⊥ 없음 → 계산 불가로 기록).
산출(final/run2026_10_26/results): judge_real.json(judge main의 real 분기와 같은 내용), judgment.json, judgment.md,
        judge_run_meta_real.json. earnings_dates.json은 judge가 없으면 yfinance 실적일 날짜만 받아 쓴다(동결 동작).
실행: py -B judge_run.py fake --out <임시 폴더>   감싸기 시운전(가짜 주가, 실제 가격을 읽지 않음)
      py -B judge_run.py real                     실제 판정 한 번(judge_real.json이 있으면 멈춤)
      py -B judge_run.py report                   judge_real.json으로 judgment.json·md만 다시 정리(판정 다시 안 함)
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import copy  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from datetime import date, datetime  # noqa: E402
from zoneinfo import ZoneInfo  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import run_paths as RP  # noqa: E402

KST = ZoneInfo("Asia/Seoul")
JUDGE = RP.FINAL / "code" / "judge.py"
A12_DEMOTE = {"H35": "R10", "H46": "R10", "H31": "MDD10", "H68": "R10"}
A12_CELLS = {"T1": ["H03", "H08", "H70"], "T2": ["H66"]}
LATE = {"H70_late": ("H70", "P_share_rz8_late"), "H66_late": ("H66", "P_bar_late")}
KNOW_END = pd.Timestamp("2026-07-01")


def load_judge():
    j = RP.sha(JUDGE)
    if j != RP.JUDGE_SHA256:
        raise SystemExit(f"멈춤: judge.py SHA-256 {j} ≠ 등록값 {RP.JUDGE_SHA256} — 판정하지 않는다")
    spec = importlib.util.spec_from_file_location("judge_frozen", JUDGE)
    J = importlib.util.module_from_spec(spec)
    sys.modules["judge_frozen"] = J
    spec.loader.exec_module(J)
    return J, j


def patch(J) -> dict:
    J.RES = RP.RES
    J.FINAL = RP.RUN / "pred_all"          # 메모리판 예측 층(쓰지 않음: 정식 표가 있음)의 라벨 폴더
    for f in (J.load_windows, J.load_features, J.load_pred_features):
        f.__defaults__ = (RP.RES,)
    src, dst = RP.RES / "pred_windows_full.parquet", RP.RES / "pred_windows.parquet"
    shutil.copyfile(src, dst)
    assert RP.sha(src) == RP.sha(dst)
    # 부록 12: 목표 10일 이상 칸을 설명용으로(판정 칸 수에서 뺌)
    for cid, tgt in A12_DEMOTE.items():
        c = J.CELL[cid]
        assert c["role"] == "main", cid
        c["role"] = "expl"
        c["why"] = f"부록 12: 목표 {tgt}(10일 이상) → 설명용(판정 칸 수에 넣지 않음)"
    # 부록 11 #1: _late 강건판(설명용)
    for lid, (base, x) in LATE.items():
        s = copy.deepcopy(J.CELL[base])
        s.update({"id": lid, "role": "expl", "x": x, "family": lid,
                  "label": s["label"] + " — _late(7/1 이후 라벨만)",
                  "why": "부록 11 #1 _late 강건판: 7/1 이후 라벨만 쓴 판(판정 칸 수에 넣지 않음)"})
        J.CELLS.append(s)
        J.CELL[lid] = s
    return {"pred_windows_copy": {"from": str(src), "to": str(dst), "sha256": RP.sha(dst)},
            "demoted": A12_DEMOTE, "late_cells": {k: {"base": b, "x": x} for k, (b, x) in LATE.items()}}


def add_late_columns(inp: dict) -> dict:
    pw = pd.read_parquet(RP.RES / "pred_windows_full.parquet")
    late = pd.DataFrame({"win_id": pw["win_id"], "P_share_rz8_late": pw["P_share_rz8_late"],
                         "P_bar_late": pw["P_bar"].where(pd.to_datetime(pw["D"]) >= KNOW_END)})
    kr = inp["feat"]["kr"]
    n0 = len(kr)
    kr = kr.merge(late, on="win_id", how="left", validate="1:1")
    assert len(kr) == n0 and (kr["win_id"].to_numpy() == inp["feat"]["kr"]["win_id"].to_numpy()).all()
    inp["feat"]["kr"] = kr
    d = pd.to_datetime(kr["D"])
    return {"P_share_rz8_late_finite": int(kr["P_share_rz8_late"].notna().sum()),
            "P_bar_late_finite": int(kr["P_bar_late"].notna().sum()),
            "P_bar_late_first_D": str(d[kr["P_bar_late"].notna()].min().date()),
            "P_share_rz8_late_first_D": str(d[kr["P_share_rz8_late"].notna()].min().date())}


# ------------------------------------------------------------------ 정리(해석 없이 숫자와 범주만)
def fnum(x, nd=3, sign=True):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def fp(x):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{x:.4f}" if x >= 0.0001 else f"{x:.1e}"


def cell_summary(cid: str, c: dict, mde: dict) -> dict:
    v = c.get("values") or {}
    info = c.get("info") or {}
    m0 = info.get("M0") or {}
    out = {"id": cid, "label": c.get("label"), "fam": c.get("fam"), "family": c.get("family"), "role": c.get("role"),
           "layer": c.get("layer"), "x": c.get("x"), "y": c.get("y"), "h": c.get("h"),
           "expected_sign": c.get("sign"), "status": c.get("status"), "verdict": c.get("verdict"), "why": c.get("why"),
           "reasons": c.get("reasons"),
           "n_back": info.get("n_back"), "back": info.get("back"), "first_D_back": info.get("first_D_back"),
           "last_D_back": info.get("last_D_back"), "blocks_back": info.get("blocks_back"),
           "n_front": info.get("n_front"), "blocks_front": info.get("blocks_front"), "n_2025": info.get("n_2025"),
           "N_rot": info.get("N_rot"), "shift_K": info.get("shift_K"), "shift_min": info.get("shift_min"),
           "est": v.get("est"), "t": v.get("t"), "p_an": v.get("p_an"), "p_shift_or_wy": v.get("p_shift"),
           "p_final": v.get("p_final"), "p_adj": c.get("p_adj"), "alpha": c.get("alpha"), "m_family": c.get("m_family"),
           "thr_local": c.get("thr_local"), "se_ci": v.get("se_ci"),
           "ci90_upper_signed": (c.get("sign") * v["est"] + 1.645 * v["se_ci"]) if v.get("est") is not None
           and v.get("se_ci") is not None and c.get("sign") else None,
           "est_front": v.get("est_front"), "p_an_front": v.get("p_an_front"), "front_ok": v.get("front_ok"),
           "est_2025": v.get("est_2025"), "opposite_2025": v.get("opposite_2025"),
           "robust_ratio": v.get("robust"), "robust_ok": v.get("robust_ok"), "n_rob": info.get("n_rob"),
           "top2_dropped": info.get("top2_dropped_obs"),
           "placebo_est": v.get("placebo_est"), "placebo_x_est": v.get("placebo_x_est"),
           "placebo_flag": v.get("placebo_flag"), "n_placebo": info.get("n_placebo"),
           "xperp_est": v.get("xp_est"), "xperp_p_final": v.get("xp_p_final"), "xperp_front_ok": v.get("xp_front_ok"),
           "xperp_robust_ratio": v.get("xp_robust"), "xperp_robust_ok": v.get("xp_robust_ok"),
           "comp_est": v.get("comp_est"), "comp_ok": v.get("comp_ok"), "stouffer_z": v.get("stouffer_z"),
           "n_eff": v.get("n_eff"), "dR2_back": info.get("dR2_back"),
           "M0_first_month": m0.get("first_month_fitted"), "M0_R2_last": m0.get("R2_last"),
           "M0_R2_mean": m0.get("R2_mean"), "M0_echo_dominant": m0.get("echo_dominant"),
           "controls_used": info.get("controls_used"), "controls_dropped": info.get("controls_dropped"),
           "wy_members": c.get("wy_members"), "notes": c.get("notes")}
    if cid in mde:
        out.update({"mde": mde[cid]["mde"], "expected_effect": mde[cid]["expected"], "detectable": mde[cid]["detectable"]})
    return out


def build_md(J_out: dict) -> str:
    S = J_out
    L = []
    L.append("# 개미 지수 등록 판정 결과 (부록 12 4칸, 2026-10-26 실행)\n")
    L.append("> 지위: 동결 판정 코드 `final/code/judge.py`(SHA-256 `" + S["meta"]["judge_sha256"] + "`)를 실제 가격으로 "
             "한 번 돌린 결과를 숫자와 범주로만 정리한 표다. 해석·매매 제안은 넣지 않는다.\n")
    L.append("## 1. 판정 칸 4개\n")
    L.append(f"- 규칙: {S['rule']}")
    L.append(f"- 가격: {S['meta']['prices_source']}. 판정 구간 끝 2026-10-23 종가, 목표 기간이 그 뒤로 넘는 신호는 결측(§2.5).\n")
    L.append("| id | 칸 | 묶음(보정) | 뒤 구간 n | 통계 | 점추정 | t | 해석 p | 이동 p(WY) | 최종 p | 보정 p | 국소 문턱 | 판정 | 이유 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for cid in S["judgment_cells_order"]:
        c = S["judgment_cells"][cid]
        stat = "부분 r" if c["fam"] == "T1" else "IC"
        corr = f"{c['fam']} {'Holm' if c['fam'] == 'T1' else 'BH'} m={c.get('m_family') or '—'}"
        L.append(f"| {cid} | {(c['label'] or '').replace('|', '/')} | {corr} | {c.get('n_back') or '—'} | {stat} | "
                 f"{fnum(c.get('est'))} | {fnum(c.get('t'), 2)} | {fp(c.get('p_an'))} | {fp(c.get('p_shift_or_wy'))} "
                 f"({c.get('shift_K') or '—'}) | {fp(c.get('p_final'))} | {fp(c.get('p_adj'))} | {fp(c.get('thr_local'))} | "
                 f"**{c.get('verdict')}** | {c.get('why') or '—'} |")
    L.append("\n### 1.1 통과 조건 항목\n")
    L.append("| id | 앞 구간 점추정(같은 부호?) | 2025 보조(반대?) | 강건 비: 급락 4일 / 기여 상위 2 / 열어 본 8일 | 강건 통과 | "
             "메아리 위약 X̂ vs X(표지) | X⊥판 점추정·최종 p·앞 부호·강건 | 짝판(저장판) | 90% 상한(부호 맞춤) | 기대 효과 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for cid in S["judgment_cells_order"]:
        c = S["judgment_cells"][cid]
        rr = c.get("robust_ratio") or {}
        rob = " / ".join(fnum(rr.get(k), 2) for k in ("crash4", "top2", "opened8"))
        if "jul3" in rr:
            rob += f" (7/29~31 {fnum(rr.get('jul3'), 2)})"
        ph = (f"{fnum(c.get('placebo_est'))} vs {fnum(c.get('placebo_x_est'))} ({'예' if c.get('placebo_flag') else '아니오'})"
              if c.get("placebo_est") is not None else "—")
        if c.get("xperp_est") is not None:
            xr = c.get("xperp_robust_ratio") or {}
            xp = (f"{fnum(c['xperp_est'])}, {fp(c.get('xperp_p_final'))}, {'같음' if c.get('xperp_front_ok') else '다름/없음'}, "
                  f"{'/'.join(fnum(xr.get(k), 2) for k in ('crash4', 'top2', 'opened8'))}")
        else:
            xp = "—"
        comp = (f"{fnum(c.get('comp_est'))} ({'같음' if c.get('comp_ok') else '다름'})" if c.get("comp_est") is not None
                else "—")
        fr = (f"{fnum(c.get('est_front'))} ({'예' if c.get('front_ok') else '아니오'})"
              + (" — 라벨 층: 보조만" if c.get("layer") == "label" else ""))
        e25 = f"{fnum(c.get('est_2025'))} ({'예' if c.get('opposite_2025') else '아니오'})"
        L.append(f"| {cid} | {fr} | {e25} | {rob} | {'예' if c.get('robust_ok') else '아니오'} | {ph} | {xp} | {comp} | "
                 f"{fnum(c.get('ci90_upper_signed'))} | {fnum(c.get('expected_effect'), 2, False)} |")
    L.append("\n### 1.2 표본·M0\n")
    L.append("| id | 뒤 구간 | 뒤 n·독립 묶음 | 앞 n·묶음 | 2025 n | 이동 열 N·K | n_eff | MDE | 검출 가능 | M0 첫 적합 달 | M0 R² 마지막·평균(메아리 우세) | 부분 R² 증가 | 기록 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for cid in S["judgment_cells_order"]:
        c = S["judgment_cells"][cid]
        b = c.get("back") or ["—", "—"]
        L.append(f"| {cid} | {b[0]}~{b[1]} ({c.get('first_D_back')}~{c.get('last_D_back')}) | {c.get('n_back')}·{c.get('blocks_back')} | "
                 f"{c.get('n_front')}·{c.get('blocks_front')} | {c.get('n_2025')} | {c.get('N_rot')}·{c.get('shift_K')} | "
                 f"{fnum(c.get('n_eff'), 1, False)} | {fnum(c.get('mde'), 3, False)} | {'예' if c.get('detectable') else '아니오'} | "
                 f"{c.get('M0_first_month') or '—'} | {fnum(c.get('M0_R2_last'), 2, False)}·{fnum(c.get('M0_R2_mean'), 2, False)}"
                 f" ({'예' if c.get('M0_echo_dominant') else '아니오'}) | {fnum(c.get('dR2_back'), 4)} | "
                 f"{'; '.join(c.get('notes') or []) or '—'} |")
    L.append("\n- 이동 p(WY)의 족 구성(대표 + '시험' 상태 순차 칸): "
             + "; ".join(f"{cid} {'·'.join(S['judgment_cells'][cid].get('wy_members') or [])}" for cid in S["judgment_cells_order"])
             + ". 기여 상위 2창: " + "; ".join(f"{cid} {'·'.join(S['judgment_cells'][cid].get('top2_dropped') or [])}"
                                         for cid in S["judgment_cells_order"]) + ".")
    pg = S.get("price_gap_windows") or {}
    if pg:
        L.append("- 가격 결측으로 뒤 구간 표본에서 빠진 창(judge_ok 창 중 가격 열 결측): "
                 + "; ".join(f"{cid} {v['n_judge_ok_back']}창 중 {len(v['windows'])}창 "
                             f"({', '.join(w['win_id'] + '[' + '·'.join(w['cols']) + ']' for w in v['windows'])})"
                             for cid, v in pg.items()) + ".")
    L.append("\n### 1.3 족 안 순차 (§2.6: 대표가 통과한 족만, 대표의 국소 문턱, 고정 순서)\n")
    L.append("| id | 족·순서 | 대표 | 칸 | 뒤 n | 통계 | 점추정 | 해석 p | 이동 p | 최종 p | 문턱 | 앞 구간 | 강건 비(급락/상위2/열어 본) | 범주 | 이유 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for cid, c in S["sequential_cells"].items():
        rr = c.get("robust_ratio") or {}
        stat = "부분 r" if c.get("fam") == "T1" else "IC"
        L.append(f"| {cid} | {c.get('family')}·{c.get('order') or '—'} | {c.get('rep') or '—'} | {(c.get('label') or '—').replace('|', '/')} | "
                 f"{c.get('n_back') or '—'} | {stat} | {fnum(c.get('est'))} | {fp(c.get('p_an'))} | {fp(c.get('p_shift_or_wy'))} | "
                 f"{fp(c.get('p_final'))} | {fp(c.get('thr_local'))} | {fnum(c.get('est_front'))} | "
                 f"{' / '.join(fnum(rr.get(k), 2) for k in ('crash4', 'top2', 'opened8')) if rr else '—'} | "
                 f"**{c.get('verdict')}** | {(c.get('why') or '—').replace('|', '/')} |")
    g = S["global_rotation"]
    L.append("\n## 2. 우연 기대 칸 수 (전체 원형 이동)\n")
    L.append(f"- 이동 {g.get('n')}회(씨앗 {g.get('seed')}, 이동 몫 {fnum(g['u_range'][0], 3, False)}~{fnum(g['u_range'][1], 3, False)}), "
             f"주 칸(role main) {', '.join(S['main_ids'])} 중 시험 칸 T1 {', '.join(S['tested']['T1']) or '없음'}, "
             f"T2 {', '.join(S['tested']['T2']) or '없음'}.")
    L.append(f"- 이동마다 통과 칸 수 평균 **{fnum(g.get('mean_main_passes'), 3, False)}칸**, 1칸 이상 비율 {fnum(100 * (g.get('p_ge1') or 0), 1, False)}%, "
             f"최대 {g.get('max')}칸, 분포 {g.get('hist')}. 칸별 통과율 {g.get('per_cell_pass_rate') or {}}. 순차 칸 통과 합 {g.get('seq_passes_total')}.")
    L.append(f"- 관측 통과 칸 수: **{S['observed_passes']}칸** ({', '.join(S['observed_pass_ids']) or '없음'}).\n")
    L.append("## 3. _late 강건판 (부록 11 #1, 판정 칸 수에 넣지 않음)\n")
    L.append("| id | 특징 | 뒤 n·묶음 | 첫 D | 점추정 | t | 해석 p | 이동 p | 최종 p | 강건 비(급락/상위2/열어 본) | 주판 점추정 | 같은 부호 | 상태·이유 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for lid, c in S["late_cells"].items():
        rr = c.get("robust_ratio") or {}
        main_est = S["judgment_cells"][LATE[lid][0]].get("est")
        same = ("예" if (c.get("est") is not None and main_est is not None and np.sign(c["est"]) == np.sign(main_est))
                else ("아니오" if c.get("est") is not None and main_est is not None else "—"))
        L.append(f"| {lid} | `{c.get('x')}` | {c.get('n_back') or '—'}·{c.get('blocks_back') or '—'} | {c.get('first_D_back') or '—'} | "
                 f"{fnum(c.get('est'))} | {fnum(c.get('t'), 2)} | {fp(c.get('p_an'))} | {fp(c.get('p_shift_or_wy'))} | "
                 f"{fp(c.get('p_final'))} | {' / '.join(fnum(rr.get(k), 2) for k in ('crash4', 'top2', 'opened8')) if rr else '—'} | "
                 f"{fnum(main_est)} | {same} | {c.get('status')}: {c.get('why') or '; '.join(c.get('reasons') or []) or '—'} |")
    L.append("\n## 4. 설명용 칸 (α 없음, 보정 칸 수에 넣지 않음)\n")
    L.append("| id | 칸 | 층 | 뒤 n | 통계 | 점추정 | t | 해석 p | 이동 p | 앞 구간 | 2025 | 강건 비(급락/상위2/열어 본) | 위약 표지 | 범주 | 이유 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for cid, c in S["explanatory_cells"].items():
        rr = c.get("robust_ratio") or {}
        stat = "부분 r" if c.get("fam") == "T1" else "IC"
        L.append(f"| {cid} | {(c.get('label') or '—').replace('|', '/')} | {c.get('layer') or '—'} | {c.get('n_back') or '—'} | {stat} | "
                 f"{fnum(c.get('est'))} | {fnum(c.get('t'), 2)} | {fp(c.get('p_an'))} | {fp(c.get('p_shift_or_wy'))} | "
                 f"{fnum(c.get('est_front'))} | {fnum(c.get('est_2025'))} | "
                 f"{' / '.join(fnum(rr.get(k), 2) for k in ('crash4', 'top2', 'opened8')) if rr else '—'} | "
                 f"{'예' if c.get('placebo_flag') else ('아니오' if c.get('placebo_est') is not None else '—')} | "
                 f"{c.get('verdict') or c.get('status')} | {(c.get('why') or '; '.join(c.get('reasons') or []) or '—').replace('|', '/')} |")
    L.append("\n### 4.1 계산하지 않는 칸(상태만)\n")
    for grp, d in S["status_only"].items():
        L.append(f"- {grp}: " + "; ".join(f"{k} ({v})" for k, v in d.items()))
    L.append("\n## 5. MDE (검출 한계, 2.49/√n_eff)\n")
    L.append("| id | 역할 | 뒤 n | n_eff | MDE | 기대 효과 | 검출 가능 |")
    L.append("|---|---|---|---|---|---|---|")
    for m in S["mde"]:
        L.append(f"| {m['id']} | {m['role']} | {m.get('n_back')} | {fnum(m['n_eff'], 1, False)} | {fnum(m['mde'], 3, False)} | "
                 f"{fnum(m['expected'], 2, False)} | {'예' if m['detectable'] else '아니오'} |")
    L.append("\n## 6. 기록 (동결 문서가 적으라고 한 것, 코드는 그대로)\n")
    for s in S["records"]:
        L.append(f"- {s}")
    L.append("\n## 7. 입력과 지문\n")
    for k, v in S["meta"]["inputs_sha256"].items():
        L.append(f"- {k}: `{v}`")
    L.append(f"- 실행: {S['meta']['started']} ~ {S['meta']['finished']} KST, judge 계산 {S['meta']['judge_seconds']}초.")
    return "\n".join(L) + "\n"


def run_judge(mode: str, out_dir) -> None:
    """judge main의 real 분기와 같은 순서(가드 → prepare_inputs → run_once → judge_real.json). fake는 감싸기 시운전용
    (가짜 주가, 실제 가격 미사용, 결과는 out_dir에만)."""
    t0 = datetime.now(KST)
    J, jsha = load_judge()
    pmeta = patch(J)
    opts = {"h31_reg_date": None}
    if mode == "real":
        J._assert_real_allowed(None, True)
    inp = J.prepare_inputs(opts)
    assert len(inp["windows"]["kr"]) == 438 and len(inp["windows"]["us"]) == 453, \
        {k: len(v) for k, v in inp["windows"].items()}
    assert inp["info"]["pred"]["built_in_memory"] is False
    late_info = add_late_columns(inp)
    t1 = time.time()
    if mode == "real":
        res = J.run_once(inp, "real", "", 0, opts, J.N_GLOBAL, confirm_real=True)
    else:
        res = J.run_once(inp, "fake", "walk", 0, opts, J.N_GLOBAL)
    secs = round(time.time() - t1, 1)
    res["inputs"] = inp["info"]
    res["checklist"] = inp["chk"]
    name = "judge_real.json" if mode == "real" else "judge_fake_walk_s0.json"
    (out_dir / name).write_text(json.dumps(J.clean(res), ensure_ascii=False, indent=1), "utf-8")
    meta = {"started": t0.replace(microsecond=0).isoformat(), "finished": datetime.now(KST).replace(microsecond=0).isoformat(),
            "judge_sha256": jsha, "judge_sha256_registered": RP.JUDGE_SHA256, "judge_seconds": secs, "mode": mode,
            "n_global": J.N_GLOBAL, "h31_reg_date": None, "result_file": str(out_dir / name),
            "wrapper": {"file": "final/run2026_10_26/code/judge_run.py", "sha256": RP.sha(RP.HERE / "judge_run.py"), **pmeta},
            "late_columns": late_info}
    (out_dir / f"judge_run_meta_{mode}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str), "utf-8")
    print(f"{mode} 끝: {name}, judge {secs}초")


def price_gap_windows(J) -> dict:
    """진단(판정 다시 안 함): 판정 칸의 뒤 구간 judge_ok 창 중 가격 열(목표·M0·HAR 통제)이 결측인 창.
    judge와 같은 입력·패널 함수(prepare_inputs·load_prices·build_frames)로 만든다. 통계는 계산하지 않는다."""
    inp = J.prepare_inputs({"h31_reg_date": None})
    pr = J.load_prices("real", windows=inp["windows"], confirm_real=True)
    kr = J.build_frames(inp["feat"], pr)["kr"]
    d = kr["Dd"].to_numpy()
    out = {}
    for cid in ("H03", "H08", "H70", "H66"):
        s = J.CELL[cid]
        rng = J.BACK_LABEL if s["layer"] == "label" else J.BACK
        m = J.in_range(d, rng) & kr["judge_ok"].to_numpy(bool)
        need = [s["y"]] + ((J.HAR_CONT + list(s.get("extra", [])) + J.HAR_DUM) if s["kind"] == "har" else list(s["ctrl"]))
        pcols = [c for c in need if c.startswith(("m0_", "fut_", "eps_", "wd_"))]
        sub = kr.loc[m, ["win_id"] + pcols]
        bad = sub[sub[pcols].isna().any(axis=1)]
        out[cid] = {"n_judge_ok_back": int(m.sum()),
                    "windows": [{"win_id": r["win_id"], "cols": [c for c in pcols if pd.isna(r[c])]} for _, r in bad.iterrows()]}
    return out


def report(mode: str, out_dir) -> int:
    J, jsha = load_judge()
    patch(J)                             # 경로·칸 등록만(판정은 다시 돌리지 않는다)
    name = "judge_real.json" if mode == "real" else "judge_fake_walk_s0.json"
    R = json.loads((out_dir / name).read_text("utf-8"))
    meta0 = json.loads((out_dir / f"judge_run_meta_{mode}.json").read_text("utf-8"))
    pmeta, late_info, secs, t0 = meta0["wrapper"], meta0["late_columns"], meta0["judge_seconds"], meta0["started"]
    tested = R["tested"]
    if not (set(tested["T1"]) <= set(A12_CELLS["T1"]) and set(tested["T2"]) <= set(A12_CELLS["T2"])):
        raise SystemExit(f"멈춤: 시험 칸 {tested}이 부록 12 판정 칸 밖 — 결과를 쓰지 않는다")
    mde = {m["id"]: m for m in R["mde"]}
    cells = R["cells"]
    jc = {cid: cell_summary(cid, cells[cid], mde) for cid in ["H03", "H08", "H70", "H66"]}
    late = {lid: cell_summary(lid, cells[lid], mde) for lid in LATE}
    reps = {c["family"]: k for k, c in cells.items() if c["role"] == "main"}
    seq = {}
    for cid, c in sorted(((k, c) for k, c in cells.items() if c["role"] in ("seq", "stopped")),
                         key=lambda kc: (kc[1].get("family") or "", J.CELL[kc[0]].get("order", 0))):
        seq[cid] = {**cell_summary(cid, c, mde), "order": J.CELL[cid].get("order"), "rep": reps.get(c.get("family")),
                    "label": c.get("label") or J.CELL[cid].get("why")}
    expl = {cid: cell_summary(cid, c, mde) for cid, c in cells.items()
            if cid not in jc and cid not in late and cid not in seq}
    pgap = price_gap_windows(J) if mode == "real" else {}
    main_ids = [k for k, c in cells.items() if c["role"] == "main"]
    obs_pass = [k for k, c in cells.items() if c.get("verdict") == "통과" and c["role"] in ("main", "seq")]
    pw = pd.read_parquet(RP.RES / "pred_windows_full.parquet")
    kp = pw.loc[pw["knowledge_period"] & pw["judge_ok"], "P_bar"]
    jp = pw.loc[~pw["knowledge_period"] & pw["judge_ok"], "P_bar"]
    pmeta_px = json.loads((RP.RES / "prices_meta.json").read_text("utf-8"))
    sch = R["prices_meta"].get("schedule") or {}
    earn = json.loads((RP.RES / "earnings_dates.json").read_text("utf-8")) if (RP.RES / "earnings_dates.json").exists() else {}
    hx = pmeta_px["naver"]["000660.KS"]
    records = [
        "부록 12 적용: 판정 칸은 T1 Holm H03·H08·H70, T2 H66 1칸(BH = Holm). H35·H46·H31·H68은 judge.py 등록에서 main이었으나 "
        "부록 12(가격 판정 전 결정)대로 이 감싸기에서 설명용(expl)으로 바꿔 판정 칸 수에서 뺐다. 그 족의 순차 칸(H36·H50)은 대표가 "
        "판정 칸이 아니어서 '순차 닫힘'으로 나온다. H67은 등록 그대로 main이고 가격 전 점검(P_sd_rz8 결측 100%)에서 자동 설명용이다.",
        "부록 12: H46·H31은 미리보기(부록 9)에서 6~10월 값을 이미 봤다. 미리보기 값(표본 안 M0 설명판): H46 IC +0.011, 최종 p 0.566, "
        "'기각 모양'; H31 IC −0.081, 최종 p 0.167, 메아리 위약 표지로 '기각 모양'(preview/results/test_free.json).",
        "부록 11 #6: 뒤 구간(6~10월)은 미리보기(부록 9)로 미리 봤다. 하네스 작성 에이전트는 미리보기 문서를 읽었으나 설계는 동결 문서만 따른다.",
        f"부록 11 #1: 라벨 층의 M0 계수·rz8 기준·재표집 풀에 7월 전(AI 지식 기간) 라벨을 예열로 썼다(주판). 판정 기간 창 P̄ 평균 "
        f"{jp.mean():+.3f}(judge_ok {len(jp)}창) 대 지식 기간 창 P̄ 평균 {kp.mean():+.3f}(judge_ok {len(kp)}창, 2025-01-03~2026-06-30). "
        "_late 판은 3절.",
        "부록 15: 2025-01-03 ~ 2025-08-20 창 153개는 이번에 새로 라벨(Y)을 매겨 예열·M0 적합·2025 보조에만 들어간다(판정 신호 아님).",
        "일정 더미: CPI·고용·MSCI는 schedule_dates.json(SHA-256 " + RP.sha(RP.RES / "schedule_dates.json")[:12] + "…, 빠짐 없음; "
        "2025년 10월분 CPI·고용은 BLS 발표 취소로 날짜 없음). MSCI는 'as of the close of' 날짜. 날짜 수 " + json.dumps(sch.get("n_dates", {}), ensure_ascii=False)
        + f", 빈 원천 {sch.get('missing')}.",
        "judge.py FOMC 상수는 2025-09-17부터만 있다(동결 코드, 2025 하반기부터 쓰던 스냅샷 기준). 이번 스냅샷의 2025-01~08 창에는 "
        "2025-01-29·03-19·05-07·06-18·07-30 FOMC가 더미 0으로 들어간다(M0 확장 적합 표본·2025 보조에만 영향). 코드는 고치지 않았다.",
        "실적일 더미: judge가 yfinance get_earnings_dates(limit=12)로 받은 날짜(earnings_dates.json): "
        + "; ".join(f"{k} {len(v) if isinstance(v, list) else v}" for k, v in earn.items()) + ".",
        "가격(prices_meta.json): yfinance 일봉 2024-10-01 ~ 2026-10-23, 12종목(judge KR_T·US_T·VIX). KOSPI 종가는 네이버 일별 종가와 "
        "500일 모두 일치(고친 날 0). SK하이닉스는 네이버 다음 거래일 기준가와 대조해 6일 종가를 고치고(" +
        ", ".join(r["date"] for r in hx["repairs_close_changed"]) + "), yfinance 막대가 없는 " +
        ", ".join(r["date"] for r in hx["repairs_bar_missing_close_only"]) + "는 종가만 넣었다(시가·고가·저가 결측).",
        "KOSPI·KOSDAQ 2026-06-29 yfinance 막대는 시가=고가=저가=종가·거래량 0이다(종가는 네이버와 같음, 네이버 시가·고가·저가는 다름). "
        "judge.py 동결 규칙(평평한 막대 + 거래량 0 → 그날 OHLC 결측)대로 결측이 된다. 시가·고가·저가는 고치지 않았다(종가만 대조 규칙). "
        "이 결측으로 뒤 구간에서 빠진 창은 1.2절 끝 줄(가격 결측 창)에 적었다(H66의 K2026-10-20~23은 R5가 10/23 종가를 넘는 정상 결측).",
        "족 안 순차(§2.6): 판정 칸 대표가 '통과'한 족만 열린다. 1.3절에 순서·문턱·범주를 그대로 적었다(멈춤·가격 전 설명용 칸은 건너뜀).",
        "삼성전자 2025-09-19 yfinance 막대 없음(대조 대상 아님, 결측 그대로). ^VIX의 미국 휴장일 줄 2026-05-25·09-07은 달력 밖이라 뺐다. "
        "2024-10 달력은 yfinance ^KS11·^GSPC 날짜(XKRX·XNYS와 차이 0).",
        "NQ=F 60분봉은 yfinance 730일 한계로 2024-10-27 18:00 ET부터다(judge 첫 사용 시각 2025-01-02 16:00 ET 이전).",
        "H31: 8월 단일종목 레버리지 규제 시행일을 받지 않았다(--h31-reg-date 없음). 부록 12로 설명용이라 ±10거래일 제외 없이 계산만 했다.",
        "감정 층 설명용(H60~H65)은 judge 동결 경로 emotion/window_emotion.parquet(국내 창 61개, 묶음 A 분류기)를 그대로 읽었다. "
        "events/run2026_10_26 감정 층 라벨은 judge 입력이 아니다.",
    ]
    pm = R["prices_meta"]
    ep = RP.RES / "earnings_dates.json"
    out = {
        "meta": {"started": t0, "finished": meta0["finished"],
                 "judge_sha256": jsha, "judge_sha256_registered": RP.JUDGE_SHA256, "judge_seconds": secs,
                 "mode": mode, "n_global": meta0["n_global"], "h31_reg_date": None,
                 "prices_source": pm.get("source") or "실제 가격(results/prices.parquet)",
                 "wrapper": pmeta,
                 "wrapper_note": "wrapper.sha256은 판정 실행 때의 judge_run.py. 그 뒤 정리(report) 부분만 고쳤고(순차 표·가격 결측 창 "
                                 "진단·이 칸), 판정 실행 부분(run_judge·patch·add_late_columns)은 그대로다",
                 "report_wrapper_sha256": RP.sha(RP.HERE / "judge_run.py"),
                 "late_columns": late_info,
                 "inputs_sha256": {**{f"results/{k}": v for k, v in R["inputs"]["features"]["sha256"].items()},
                                   "results/pred_windows.parquet (= pred_windows_full.parquet)": R["inputs"]["pred"]["sha256"],
                                   "results/prices.parquet": pm.get("prices_sha256"),
                                   "results/prices_nq_1h.parquet": pm.get("nq_sha256"),
                                   "results/schedule_dates.json": pm.get("schedule_sha256"),
                                   "results/earnings_dates.json": RP.sha(ep) if (mode == "real" and ep.exists()) else None,
                                   "emotion/window_emotion.parquet": R["inputs"]["emotion"].get("sha256"),
                                   "final/code/judge.py": jsha,
                                   f"{name}": RP.sha(out_dir / name)},
                 "schedule_meta": sch},
        "rule": "부록 12: T1 Holm(α 0.05) H03·H08·H70, T2 BH(q 0.10) H66. 통과 = 보정 p ≤ 기준 + 앞 구간 같은 부호(라벨 층은 보조) "
                "+ 강건 3개 점추정 50% 이상·같은 부호(+H03 7/29~31판, 짝판 부호). 기각 = 뒤 부호 반대 또는 90% 상한 < 기대 효과 절반 "
                "또는 메아리 위약. 그 밖은 판정 불가(전방 트랙).",
        "tested": tested, "main_ids": main_ids,
        "judgment_cells_order": ["H03", "H08", "H70", "H66"], "judgment_cells": jc,
        "observed_passes": len(obs_pass), "observed_pass_ids": obs_pass,
        "global_rotation": R["global_rotation"],
        "price_gap_windows": pgap,
        "sequential_cells": seq,
        "late_cells": late,
        "explanatory_cells": expl,
        "status_only": J.STATUS_ONLY,
        "mde": [{**m, "role": cells[m["id"]]["role"]} for m in R["mde"]],
        "records": records,
    }
    sfx = "" if mode == "real" else "_fake"
    (out_dir / f"judgment{sfx}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), "utf-8")
    (out_dir / f"judgment{sfx}.md").write_text(build_md(out), "utf-8")
    print(json.dumps({cid: {k: c.get(k) for k in ("verdict", "why", "est", "p_final", "p_adj", "front_ok", "robust_ok")}
                      for cid, c in jc.items()}, ensure_ascii=False, indent=1, default=str))
    print("global", {k: R["global_rotation"].get(k) for k in ("mean_main_passes", "p_ge1", "max", "hist")})
    print("late", {k: (c.get("status"), c.get("est"), c.get("p_final")) for k, c in late.items()})
    print("tested", tested, "secs", secs)
    return 0


def main(argv=None) -> int:
    import argparse
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fake", "real", "report"])
    ap.add_argument("--mode", default="real", choices=["fake", "real"], help="report가 읽을 결과")
    ap.add_argument("--out", default=None, help="fake 결과 폴더(실제 결과 폴더 밖)")
    a = ap.parse_args(argv)
    if a.cmd == "fake":
        out = Path(a.out)
        assert out.resolve() != RP.RES.resolve()
        out.mkdir(parents=True, exist_ok=True)
        run_judge("fake", out)
        return report("fake", out)
    if a.cmd == "real":
        if (RP.RES / "judge_real.json").exists():
            raise SystemExit("멈춤: judge_real.json이 이미 있다 — 판정은 한 번만 돌린다(정리만 다시 하려면 report)")
        run_judge("real", RP.RES)
        return report("real", RP.RES)
    out = RP.RES if a.mode == "real" else Path(a.out)
    return report(a.mode, out)


if __name__ == "__main__":
    raise SystemExit(main())
