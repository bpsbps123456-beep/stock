# -*- coding: utf-8 -*-
"""F1-A 독립 재구현(눈가림) — lib.py·fam_*.py를 import하거나 읽지 않는다.

규칙 근거: results/fam_F1_frozen.json의 F1-A 매개변수
  {N: 1, entry: confirm, tgt: 2, stop: None, rev: False, filt: None}
와 BRIEF2.md 체결 절만 쓴다.

내 해석(주 판):
  - 범위 봉 = SOXL 09:30 1분봉(N=1). 종가 > 시가면 SOXL 방향, 종가 < 시가면 SOXS 방향, 같으면 거래 없음.
  - 확인 진입: 09:31 봉부터 SOXL 봉 종가가 범위 고가 위(SOXL 방향) / 범위 저가 아래(SOXS 방향)로
    처음 마감한 봉 t에서 결정 → 다음 봉 t+1 종가 + 반 호가로 보유 종목 매수.
  - R = SOXL 기준 진입(결정 봉 종가)에서 범위 반대쪽 끝까지 거리 %. 목표 = 매수가 × (1 + 2R) 지정가.
    무손절이라 손절선은 없다.
  - 목표는 체결 다음 봉부터, 보유 종목 고가 ≥ 목표 × 1.0002이면 목표가에 판다(호가 비용 없음).
  - 마감: actual = 그날 24:00 KST의 ET(서머타임 11:00, 표준시 10:00), std = 모든 날 10:00.
    남은 보유는 마감 1분 전 봉 종가 − 반 호가로 판다. 체결 봉이 마감 1분 전 봉 이후가 되는 결정은 하지 않는다.
  - 반 호가: cost_table.json, 시간대는 체결 봉 시각(pre <09:30, open5 09:30~09:34, early 09:35~09:59, late 10:00~).
  - 하루 수익 = 매매 수익의 곱 − 1 (이 후보는 하루 최대 1회).

conf(2026-07-01~10-28) 날만 읽고 results/confirm_daily_F1-A_{actual,std}.csv의 conf 행과 날별로 비교한다.
"""
import json
import sys
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = "C:/Users/geunt/OneDrive/Desktop/주식/data/derived/analysis/sox_open_switch"
CONF_START, CONF_END = "2026-07-01", "2026-10-28"
ET = ZoneInfo("America/New_York")
KST = ZoneInfo("Asia/Seoul")
TOL = 1e-6

COST = json.load(open(f"{ROOT}/cost_table.json", encoding="utf-8"))


def zone(m):
    """m = 분(자정부터). 체결 봉 시각의 호가 시간대."""
    if m < 570:
        return "pre"
    if m < 575:
        return "open5"
    if m < 600:
        return "early"
    return "late"


def hs(sym, m, cm):
    return COST[sym][zone(m)] * cm


def load_conf():
    out = {}
    for s in ("SOXL", "SOXS"):
        df = pd.read_parquet(f"{ROOT}/cache/alp1m/{s}.parquet")
        idx = df.index
        if idx.tz is None:
            raise SystemExit("index tz 없음")
        idx = idx.tz_convert(ET)
        df = df.copy()
        df.index = idx
        d = idx.date
        keep = (d >= date.fromisoformat(CONF_START)) & (d <= date.fromisoformat(CONF_END))
        df = df[keep]
        df["day"] = df.index.date
        df["m"] = df.index.hour * 60 + df.index.minute
        out[s] = df
    return out


def actual_cutoff_min(day):
    """그날 밤 24:00 KST(= 다음 날 00:00 KST)를 ET 분으로."""
    k = datetime(day.year, day.month, day.day, tzinfo=KST) + timedelta(days=1)
    e = k.astimezone(ET)
    assert e.date() == day, (day, e)
    return e.hour * 60 + e.minute


def day_frames(data, day):
    fr = {}
    for s in ("SOXL", "SOXS"):
        g = data[s][data[s]["day"] == day]
        fr[s] = g.set_index("m")[["Open", "High", "Low", "Close"]]
    return fr


def run_day(fr, cutoff, cm=1.0, fill="close", mirror=False, variant=None):
    """하루 실행. 반환 (수익률 fraction, 거래 수, 정보 dict)."""
    variant = variant or {}
    sx = fr["SOXL"]
    if 570 not in sx.index:
        return 0.0, 0, {"why": "no 09:30 bar"}
    o, h, l, c = sx.loc[570, ["Open", "High", "Low", "Close"]]
    if c > o:
        d = "L"
    elif c < o:
        d = "S"
    else:
        return 0.0, 0, {"why": "doji"}
    held = "SOXL" if d == "L" else "SOXS"
    if mirror:
        held = "SOXS" if held == "SOXL" else "SOXL"
    hb = fr[held]
    last = cutoff - 1  # 남은 보유를 파는 봉
    max_fill = last if variant.get("allow_fill_at_last") else last - 1
    dec = None
    for t in range(571, max_fill):  # 결정 봉 t, 체결 봉 t+1 <= max_fill
        if t not in sx.index:
            continue
        ct = sx.loc[t, "Close"]
        if (d == "L" and ct > h) or (d == "S" and ct < l):
            dec = t
            break
    if dec is None:
        return 0.0, 0, {"why": "no confirm"}
    f = dec + 1
    ref = sx.loc[dec, "Close"] if not variant.get("ref_fill") else sx.loc[f, "Close"]
    R = (ref - l) / ref if d == "L" else (h - ref) / ref
    raw = hb.loc[f, "Close"] if fill == "close" else hb.loc[f, "Open"]
    entry = raw * (1 + hs(held, f, cm))
    base = raw if variant.get("tgt_on_raw") else entry
    tgt = base * (1 + 2 * R)
    start = f + 1 if (fill == "close" or not variant.get("open_check_fill_bar")) else f
    exit_px, why, xb = None, None, None
    for b in range(start, last + 1):
        if b not in hb.index:
            continue
        if hb.loc[b, "High"] >= tgt * 1.0002:
            exit_px, why, xb = tgt, "target", b
            break
    if exit_px is None:
        exit_px = hb.loc[last, "Close"] * (1 - hs(held, last, cm))
        why, xb = "cutoff", last
    r = exit_px / entry - 1
    return r, 1, {"why": why, "held": held, "dec": dec, "fill": f, "exit_bar": xb, "R": R,
                  "entry": entry, "tgt": tgt, "exit": exit_px}


def hhmm(m):
    return f"{m // 60:02d}:{m % 60:02d}"


def stats(x):
    x = np.asarray(x, float)
    n = len(x)
    m = x.mean()
    sd = x.std(ddof=1)
    return m, m / (sd / np.sqrt(n)), n


def main(variant=None, verbose=True):
    data = load_conf()
    days = sorted(set(data["SOXL"]["day"]) & set(data["SOXS"]["day"]))
    days = [d for d in days if 570 in set(data["SOXL"].loc[data["SOXL"]["day"] == d, "m"])]
    summary = {}
    for scheme in ("actual", "std"):
        ref = pd.read_csv(f"{ROOT}/results/confirm_daily_F1-A_{scheme}.csv", encoding="utf-8-sig")
        ref = ref[ref["period"] == "conf"].set_index("day")
        rows = []
        for day in days:
            fr = day_frames(data, day)
            cut = actual_cutoff_min(day) if scheme == "actual" else 600
            r, n, info = run_day(fr, cut, variant=variant)
            r2, _, _ = run_day(fr, cut, cm=2.0, variant=variant)
            ro, _, _ = run_day(fr, cut, fill="open", variant=variant)
            rm, _, _ = run_day(fr, cut, mirror=True, variant=variant)
            rows.append({"day": day.isoformat(), "cutoff_et": hhmm(cut), "ret_pct": r * 100, "n": n,
                         "ret_cost2x_pct": r2 * 100, "ret_open_pct": ro * 100, "mirror_ret_pct": rm * 100,
                         "why": info.get("why"), "held": info.get("held"),
                         "dec": hhmm(info["dec"]) if "dec" in info else "",
                         "exit_bar": hhmm(info["exit_bar"]) if "exit_bar" in info else ""})
        mine = pd.DataFrame(rows).set_index("day")
        miss_mine = sorted(set(ref.index) - set(mine.index))
        miss_ref = sorted(set(mine.index) - set(ref.index))
        common = sorted(set(ref.index) & set(mine.index))
        res = {"days_ref": len(ref), "days_mine": len(mine), "only_ref": miss_mine, "only_mine": miss_ref}
        for col in ("ret_pct", "n", "ret_cost2x_pct", "ret_open_pct", "mirror_ret_pct"):
            diff = (mine.loc[common, col].astype(float) - ref.loc[common, col].astype(float)).abs()
            bad = diff[diff > TOL]
            res[col] = {"max_abs_diff": float(diff.max()), "n_bad": int(len(bad)),
                        "bad_days": [(dd, float(mine.loc[dd, col]), float(ref.loc[dd, col])) for dd in bad.index]}
        m, t, n = stats(mine["ret_pct"])
        rm_, rt_, _ = stats(ref.loc[common, "ret_pct"])
        res["mine_mean_t"] = (round(m, 3), round(t, 3), n)
        res["ref_mean_t"] = (round(rm_, 3), round(rt_, 3))
        res["active_mine"] = f"{int(mine['n'].sum())}/{len(mine)}"
        res["exits_mine"] = mine.loc[mine["n"] > 0, "why"].value_counts().to_dict()
        res["by_sym_mine"] = {s: (int(len(g)), round(float(g["ret_pct"].mean()), 3))
                              for s, g in mine[mine["n"] > 0].groupby("held")}
        res["cutoffs"] = sorted(mine["cutoff_et"].unique())
        summary[scheme] = res
        if verbose:
            out = mine.join(ref[["ret_pct", "n"]], rsuffix="_ref")
            out.to_csv(f"{ROOT}/results/verify_F1-A_{scheme}.csv", encoding="utf-8-sig")
    return summary


def explain_day(day_iso, cutoff, variant=None):
    data = load_conf()
    day = date.fromisoformat(day_iso)
    r, n, info = run_day(day_frames(data, day), cutoff, variant=variant)
    info = {k: (hhmm(v) if k in ("dec", "fill", "exit_bar") else v) for k, v in info.items()}
    return {"ret_pct": r * 100, "n": n, **{k: (float(v) if isinstance(v, (np.floating, float)) else v)
                                           for k, v in info.items()}}


if __name__ == "__main__":
    print("== 주 판(체결 봉이 마감 1분 전 봉이면 진입 안 함) ==")
    s = main()
    print(json.dumps(s, ensure_ascii=False, indent=1, default=str))
    print("== 변형: 마감 1분 전 봉 체결 허용(같은 봉 종가에 바로 청산) ==")
    s2 = main(variant={"allow_fill_at_last": True}, verbose=False)
    for k in s2:
        print(k, {c: (s2[k][c]["n_bad"], s2[k][c]["max_abs_diff"])
                  for c in ("ret_pct", "n", "ret_cost2x_pct", "ret_open_pct", "mirror_ret_pct")},
              s2[k]["mine_mean_t"], s2[k]["active_mine"], s2[k]["exits_mine"])
    print("2026-07-17 std 변형:", explain_day("2026-07-17", 600, {"allow_fill_at_last": True}))
