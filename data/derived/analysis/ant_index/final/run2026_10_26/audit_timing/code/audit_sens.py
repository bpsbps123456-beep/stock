"""판정 시점 감사 3: 찾은 문제의 판정 숫자 영향 크기(민감도). 판정이 아니다 — judge_real.json·judgment.md는 그대로다.

final/code/judge.py는 고치지 않는다(SHA-256 확인 뒤 메모리에서 불러 씀, audit_price.load_judge와 같은 경로·칸 등록).
입력만 바꿔 run_all(200회 이동, 같은 씨앗)을 다시 돌리고 판정 4칸 숫자를 기준(S0 = judge_real 재현)과 나란히 낸다.
  S1  FOMC 2025 상반기 5회(01-29·03-19·05-07·06-18·07-30, 연준 성명일) 더함 — judge.FOMC 상수가 2025-09-17부터만 있음
  S2  국내 날짜 행사(국내 옵션 만기, MSCI 'as of the close' 날)를 국내 창에서 그 행사 세션(D = 행사일)에 둠
      — judge._in_window(P ≤ e ≤ D−1)는 국내 세션에 일어나는 행사를 다음 세션 창에 붙인다
  S3  삼성전자 잠정실적일(§2.4가 적은 '삼전 잠정실적')을 실적 더미에 더함. judge는 yfinance 확정 실적일만 쓴다.
      yfinance 한국 종목 날짜와 같은 꼴(미국 동부 날짜 = KST 날짜 − 1)로 넣어 국내 창에서 발표 세션에 붙게 한다.
      확인한 날짜만: 2025-01-08·04-08·07-08·10-14, 2026-01-08·07-07·10-08(2026-04 1분기 잠정일은 확인 못 해 뺌)
  S4  S1 + S2 + S3
  S5  H08 '마감 전 정보만' 잘림 판(fight_pc·trunc_share_pc, ncomment 미사용) — interim audit_leak §3-2 #1 권고(설명용)
  S6  KOSPI 2026-06-29 평평한 yfinance 막대(시가=고가=저가=종가, 거래량 0) 대신 네이버 시가·고가·저가(종가 같음) — 시점 문제가
      아니라 결과 자료 결측 참고
산출: audit_timing/results/audit_sens.json
실행: py -B audit_sens.py
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import time  # noqa: E402
from datetime import date, timedelta  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit_price as AP  # noqa: E402

FOMC_2025H1 = [date(2025, 1, 29), date(2025, 3, 19), date(2025, 5, 7), date(2025, 6, 18), date(2025, 7, 30)]
SEC_PRELIM_KST = [date(2025, 1, 8), date(2025, 4, 8), date(2025, 7, 8), date(2025, 10, 14),
                  date(2026, 1, 8), date(2026, 7, 7), date(2026, 10, 8)]
NAVER_0629 = {"open": 8334.28, "high": 8525.53, "low": 8127.99,
              "close": 8394.650390625}        # 종가는 yfinance = 네이버(prices_meta), judge 읽기 규칙이 막대 전체를 결측으로 둠


def run(J, inp, px, nq, cal, *, kr_override=None, frames_fn=None, opts=None) -> tuple[dict, dict]:
    opts = opts or {"h31_reg_date": None}
    panel = J.build_panel(px, nq, inp["windows"], cal)
    kr = panel["kr"]
    if kr_override:
        kr = kr_override(J, kr)
    frames = J.build_frames(inp["feat"], {"kr": kr, "us": panel["us"]})
    if frames_fn:
        frames = frames_fn(frames)
    res = J.run_all(frames, opts, inp["chk"], n_global=J.N_GLOBAL)
    return json.loads(json.dumps(J.clean(AP.cell_nums(res)))), {"sched_n": panel["schedule_meta"]["n_dates"]}


def align_kr_events(J, kr: pd.DataFrame) -> pd.DataFrame:
    kr = kr.copy()
    ks = sorted(kr["Dd"])
    sch = J.schedule(sorted(d for d in AP_PX["^KS11"].index), sorted(d for d in AP_PX["^SOX"].index), AP_CAL)
    kss = set(d for d in AP_PX["^KS11"].index)
    D = kr["Dd"].to_numpy()
    info = {}
    for nm in ("kr_opex", "msci"):
        ev = set()
        for e in sch[nm]:
            d = e
            while d not in kss and d > date(2024, 10, 1):      # 국내 휴장이면 직전 세션(국내 옵션 만기 규칙과 같음)
                d -= timedelta(days=1)
            ev.add(d)
        new = np.array([float(d in ev) for d in D])
        old = kr[f"m0_cal_{nm}"].to_numpy(float)
        back = (D >= date(2026, 6, 1)) & (D <= date(2026, 10, 23))
        info[nm] = {"changed_windows": int((new != old).sum()), "changed_back": int(((new != old) & back).sum()),
                    "ones_old_back": int(old[back].sum()), "ones_new_back": int(new[back].sum())}
        kr[f"m0_cal_{nm}"] = new
    ALIGN_INFO.update(info)
    return kr


ALIGN_INFO: dict = {}
AP_PX: dict = {}
AP_CAL: dict = {}


def main() -> int:
    t0 = time.time()
    J, jsha = AP.load_judge()
    opts = {"h31_reg_date": None}
    inp = J.prepare_inputs(opts)
    AP.add_late_columns(inp)
    px, nq, cal, src = J._read_real_daily()
    AP_PX.update(px)
    AP_CAL.update(cal)
    saved = json.loads(json.dumps(AP.cell_nums(json.loads((AP.RES / "judge_real.json").read_text("utf-8")))))
    out = {"judge_sha256": jsha, "note": "민감도(판정 아님). judge.py·판정 결과 파일은 그대로", "variants": {}}
    fomc0 = list(J.FOMC)

    def rec(name, nums, extra=None):
        out["variants"][name] = {"cells": nums, **(extra or {})}
        print(name, {k: (v.get("verdict"), round(v["est"], 4) if v.get("est") is not None else None,
                         round(v["p_final"], 4) if v.get("p_final") is not None else None,
                         round(v["p_adj"], 4) if v.get("p_adj") is not None else None)
                     for k, v in nums.items() if not k.startswith("_")}, nums["_global"]["mean_main_passes"],
              round(time.time() - t0), flush=True)

    s0, m = run(J, inp, px, nq, cal)
    rec("S0_baseline", s0, {"same_as_judge_real": {k: AP.same(s0[k], saved[k]) for k in s0}})
    # S1
    J.FOMC = sorted(set(fomc0) | set(FOMC_2025H1))
    s1, m = run(J, inp, px, nq, cal)
    rec("S1_fomc_2025H1", s1, m)
    J.FOMC = fomc0
    # S2
    s2, m = run(J, inp, px, nq, cal, kr_override=align_kr_events)
    rec("S2_kr_event_on_session", s2, {**m, "align": dict(ALIGN_INFO)})
    # S3
    cal3 = {**cal, "earn": {**cal["earn"], "005930.KS_prelim": [d - timedelta(days=1) for d in SEC_PRELIM_KST]}}
    s3, m = run(J, inp, px, nq, cal3)
    rec("S3_sec_prelim", s3, m)
    # S4
    J.FOMC = sorted(set(fomc0) | set(FOMC_2025H1))
    s4, m = run(J, inp, px, nq, cal3, kr_override=align_kr_events)
    rec("S4_S1+S2+S3", s4, m)
    J.FOMC = fomc0
    # S5
    fa = pd.read_parquet(AP.RES / "feat_activity.parquet")
    fk = fa[fa["market"] == "kr"][["win_id", "fight_pc", "trunc_share_pc"]]

    def pc(frames):
        k = frames["kr"].merge(fk, on="win_id", how="left", validate="1:1")
        k["fight"] = k["fight_pc"]
        k["trunc_share"] = k["trunc_share_pc"]
        frames = dict(frames)
        frames["kr"] = k.drop(columns=["fight_pc", "trunc_share_pc"])
        return frames
    s5, m = run(J, inp, px, nq, cal, frames_fn=pc)
    kk = fa[fa["market"] == "kr"]
    corr = float(kk[["fight", "fight_pc"]].dropna().corr().iloc[0, 1])
    rec("S5_H08_preclose_trunc", s5, {"corr_fight_vs_fight_pc_kr": corr,
                                      "mean_abs_diff": float((kk["fight"] - kk["fight_pc"]).abs().mean())})
    # S6
    px6 = {t: d.copy() for t, d in px.items()}
    d0629 = date(2026, 6, 29)
    for k, v in NAVER_0629.items():
        px6["^KS11"].at[d0629, k] = v
    s6, m = run(J, inp, px6, nq, cal)
    rec("S6_kospi_0629_naver_ohl", s6, m)
    # S7: 결번율(수집 시점까지 지워진 글로 잰 값, §2.2가 통제로 정한 예외) 없이 HAR·M0
    har0, m0k0 = list(J.HAR_CONT), list(J.M0_KR_CONT)
    J.HAR_CONT = [c for c in har0 if c != "m0_missing_rate"]
    J.M0_KR_CONT = [c for c in m0k0 if c != "m0_missing_rate"]
    s7, m = run(J, inp, px, nq, cal)
    rec("S7_no_missing_rate_control", s7, m)
    J.HAR_CONT, J.M0_KR_CONT = har0, m0k0
    out["seconds"] = round(time.time() - t0)
    (AP.OUT / "audit_sens.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), "utf-8")
    print("wrote", AP.OUT / "audit_sens.json", out["seconds"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
