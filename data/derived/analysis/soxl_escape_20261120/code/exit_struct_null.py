"""갈래 exit_struct 보정: 빈도 맞춘 무작위 사건 귀무분포 (2026-11-20).

발견: hold_sim의 x_rand_mean%는 드문 규칙일수록 음(-)으로 치우친다. 하루 1~2회 무작위 사건을 탈출로 넣으면
TRAIN x_rand_mean% 평균 -0.14~-0.17(30회, 표준편차 0.15~0.22). 원인: 무작위 대조가 보유시간을 섞은 뒤 16:00에서 자르므로
대조의 실효 보유시간이 규칙보다 짧고, 이 하락 장세에선 짧게 들고 있을수록 덜 잃는다.
그래서 각 설정마다 '같은 날 같은 개수'의 무작위 정규장 1분봉 사건을 30회 뽑아 hold_sim을 돌린 평균(null)을 빼서 null_adj와
백분위(null_pct, 규칙이 무작위 30회 중 몇 %보다 나은가)를 함께 적는다. 선택 순위는 과제 규정대로 primary(x_rand_mean%)를 유지한다.
실행: py code/exit_struct_null.py → results/exit_struct.json 'null' 항목 추가.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H  # noqa: E402
import exit_struct_scan as S  # noqa: E402
import exit_struct_retest as R  # noqa: E402

D1 = H.load_1m()
REGPOS = np.nonzero(D1["reg"].to_numpy())[0]
REGDATE = D1["date"].values[REGPOS]
NDRAW = 30


def signals():
    """모든 설정 키 → (bars, sig). scan/retest와 같은 정의로 다시 만든다."""
    out = {}
    for tf in (1, 3):
        bars = S.TF[tf]
        atr_ = H.atr(bars, 14)
        lines = S.build_lines(tf)
        vol_ok = bars["Volume"] > 1.5 * bars["Volume"].rolling(20).mean()
        F = S.filters_for(tf)
        for ln, line in lines.items():
            for conf in S.CONF:
                cond = S.cond_below(bars, line, conf, atr_)
                ev = S.event_of(cond)
                for vol in (False, True):
                    k = f"{tf}m|{ln}|{conf}|{'vol' if vol else 'novol'}|event"
                    sig = ev & vol_ok if vol else ev
                    out[k] = (bars, sig)
                    # 2·3단계 파생(키가 결과에 있으면)
                    out[k + "#F"] = (bars, sig, F, line, cond)
            out[f"{tf}m|{ln}|c0|novol|state"] = (bars, S.cond_below(bars, line, "c0", atr_).fillna(False))
        for ln in R.LINES:
            for mode in R.MODES:
                out[f"{tf}m|{ln}|c0|{mode}|event"] = (bars, R.follow_signal(bars, lines[ln], tf, mode))
    return out


def resolve(key, base):
    if key in base:
        return base[key][:2]
    b, rest = key.split("+", 1)
    bars, ev, F, line, cond = base[b + "#F"]
    tf = 1 if b.startswith("1m") else 3
    if rest == "two_bar":
        return bars, S.two_bar(cond)
    if rest == "retest_fail":
        return bars, S.retest_fail(bars, line, ev, tf)
    m = ev.copy()
    for fn in rest.split("&"):
        m = m & F[fn]
    return bars, m


def null_stats(pos, rng, actual):
    sel = H.in_range(D1["date"].values[pos], rng) & D1["reg"].to_numpy()[pos]
    p = pos[sel]
    cnt = pd.Series(D1["date"].values[p]).value_counts()
    g = np.random.default_rng(11)
    vals = []
    for _ in range(NDRAW):
        draw = []
        for dt, c in cnt.items():
            cand = REGPOS[REGDATE == dt]
            draw.append(g.choice(cand, size=min(int(c), len(cand)), replace=False))
        dp = np.sort(np.concatenate(draw)) if draw else np.array([], dtype=int)
        vals.append(H.hold_sim(dp, rng)["x_rand_mean%"])
    vals = np.array(vals)
    return {"null_mean": round(float(vals.mean()), 3), "null_sd": round(float(vals.std()), 3),
            "null_adj": round(float(actual - vals.mean()), 3),
            "null_pct": round(float(100 * (vals < actual).mean()), 1)}


def main():
    with open(S.OUT, encoding="utf-8") as fh:
        res = json.load(fh)
    base = signals()
    allc = {}
    for st in ("stage1", "stage2", "stage3", "stage4"):
        allc.update(res[st])
    res["null"] = {}
    for i, (k, v) in enumerate(allc.items()):
        if not S.ok_freq(v):
            continue
        bars, sig = resolve(k, base)
        pos = H.exec_index(sig, bars)
        pos = pos[pos >= 0]
        res["null"][k] = {nm: null_stats(pos, rng, v[nm]["primary"])
                          for nm, rng in (("train", H.TRAIN), ("test", H.TEST))}
        if i % 50 == 0:
            print(i, k, res["null"][k], flush=True)
    na = pd.DataFrame({k: {"tr": v["train"]["null_adj"], "te": v["test"]["null_adj"],
                           "trp": v["train"]["null_pct"], "tep": v["test"]["null_pct"]} for k, v in res["null"].items()}).T
    res["null_summary"] = {
        "n": int(len(na)), "train_adj_mean": round(float(na.tr.mean()), 3), "test_adj_mean": round(float(na.te.mean()), 3),
        "train_share_pct_ge95": round(float((na.trp >= 95).mean()), 3),
        "test_share_pct_ge95": round(float((na.tep >= 95).mean()), 3),
        "rank_corr_adj": round(float(na.tr.corr(na.te, method="spearman")), 3),
        "top10_train_adj": na.sort_values("tr", ascending=False).head(10).round(3).to_dict("index"),
    }
    with open(S.OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=float)
    print(json.dumps(res["null_summary"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
