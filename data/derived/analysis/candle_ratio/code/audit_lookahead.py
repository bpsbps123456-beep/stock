"""감사(audit) 2: engine.py 자체의 미리보기(look-ahead)·경계 점검. (재계산은 audit_reimpl.py, 여기서는 엔진을 시험 대상으로 부른다)

1. ratio: 봉 j 뒤 자료를 마구 바꿔도 p[:j+1]이 그대로인가 (MU 5분봉 창 전 조합 4,752개 × 자르는 점 3개).
2. ratio 정확도: 엔진 이동 합 vs 창을 직접 더한 값. 모두 도지인 창의 body 값(사양 중립 0.5 vs 엔진).
3. positions: p[j]를 바꿔도 pos[:j+1]이 그대로이고, 체결은 신호 다음 봉 시가인가.
4. tf_bars: 원자료 시각 T 이후를 바꿔도 T 전에 끝난 봉은 그대로인가. 40분 연장 봉 시작 시각 목록, start 표시와 실제 첫 봉 시각.
5. 격자 칸 수: BRIEF 표의 곱 vs 결과 파일 행 수.
사용: py code/audit_lookahead.py → results/audit_lookahead.json
"""
from __future__ import annotations

import json
import random
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import engine as E  # noqa: E402

ROOT = E.HERE
rng = np.random.default_rng(20261023)
out = {}

d5 = E.load("MU", "5m")
d1 = E.load("MU", "1h")


def direct(O, C, V, n, kind, flat_body=0.5):
    m = len(O)
    p = np.full(m, np.nan)
    for i in range(n - 1, m):
        o, c, v = O[i - n + 1:i + 1], C[i - n + 1:i + 1], V[i - n + 1:i + 1]
        if kind == "count":
            u, dn = (c > o).sum(), (c < o).sum()
            p[i] = u / (u + dn) if u + dn else 0.5
        elif kind == "body":
            den = np.abs(c - o).sum()
            p[i] = (1 + (c - o).sum() / den) / 2 if den > 0 else flat_body
        else:
            den = v[c != o].sum()
            p[i] = v[c > o].sum() / den if den > 0 else 0.5
    return p


# 1·2. ratio 미리보기·정확도 -----------------------------------------------------------
la_fail = 0; la_tests = 0
acc = {"count": 0.0, "body": 0.0, "vol": 0.0}
flat_body = 0  # 모두 도지인 body 창 수
flat_vol = 0   # 거래량 0만 있는 vol 창 수(연장 거래량 0)
out_of_range = 0
combos = 0
for base, d, tfs in (("5m", d5, E.TF_5M), ("1h", d1, E.TF_1H)):
    for sess in ("eth", "rth"):
        for tf in tfs:
            tb = E.tf_bars(d, tf, sess)
            tb = tb[tb.day >= E.WIN[base]["warm"]].reset_index(drop=True)
            O, C, V = tb.O.values.astype(float), tb.C.values.astype(float), tb.V.values.astype(float)
            for n in E.NS:
                for kind in E.RATIOS:
                    combos += 1
                    p = E.ratio(O, C, V, n, kind)
                    out_of_range += int(((p < 0) | (p > 1)).sum())
                    for _ in range(3):
                        j = int(rng.integers(n, len(O) - 1))
                        O2, C2, V2 = O.copy(), C.copy(), V.copy()
                        k = len(O) - j - 1
                        O2[j + 1:] = rng.uniform(1, 2000, k); C2[j + 1:] = rng.uniform(1, 2000, k); V2[j + 1:] = rng.uniform(0, 1e7, k)
                        p2 = E.ratio(O2, C2, V2, n, kind)
                        la_tests += 1
                        if not np.allclose(p[:j + 1], p2[:j + 1], equal_nan=True, atol=0, rtol=0):
                            la_fail += 1
                    if (tf in (5, 40, 70, 295) and base == "5m") or base == "1h":
                        q = direct(O, C, V, n, kind)
                        m = ~np.isnan(q)
                        acc[kind] = max(acc[kind], float(np.nanmax(np.abs(p[m] - q[m]))) if m.any() else 0.0)
                        if kind == "body":
                            w = np.lib.stride_tricks.sliding_window_view(np.abs(C - O), n).sum(1) == 0
                            flat_body += int(w.sum())
                        if kind == "vol":
                            vv = V * (C != O)
                            w = np.lib.stride_tricks.sliding_window_view(vv, n).sum(1) == 0
                            flat_vol += int(w.sum())
out["ratio"] = dict(combos=combos, lookahead_tests=la_tests, lookahead_fail=la_fail, out_of_range=out_of_range,
                    max_abs_diff_vs_direct=acc, flat_body_windows_MU=flat_body, zero_vol_windows_MU=flat_vol)

# 모두 도지 body 창: 엔진 값 직접 확인
O = np.array([10.0, 10.0, 10.0, 10.0, 11.0]); C = np.array([10.0, 10.0, 10.0, 10.0, 12.0]); V = np.ones(5)
out["ratio"]["engine_body_all_doji_value"] = float(E.ratio(O, C, V, 3, "body")[2])
out["ratio"]["engine_count_all_doji_value"] = float(E.ratio(O, C, V, 3, "count")[2])
out["ratio"]["engine_vol_all_doji_value"] = float(E.ratio(O, C, V, 3, "vol")[2])

# 다른 종목: 모두 도지 body 창(5분 연장 N3·N4)과 연장 거래량 0 비율
syms = sorted(p.stem[:-3] for p in (ROOT / "cache").glob("*_5m.parquet"))
fb = []
for s in syms:
    d = E.load(s, "5m")
    mins = d.index.hour * 60 + d.index.minute
    ext = (mins < 570) | (mins >= 960)
    tb = E.tf_bars(d, 5, "eth")
    dz = (tb.C.values == tb.O.values)
    w3 = np.lib.stride_tricks.sliding_window_view(dz, 3).all(1).sum()
    tb40 = E.tf_bars(d, 40, "eth")
    w40 = np.lib.stride_tricks.sliding_window_view(tb40.C.values == tb40.O.values, 3).all(1).sum()
    fb.append(dict(sym=s, ext_bars=int(ext.sum()), ext_vol0=float((d.Volume[ext] == 0).mean()) if ext.any() else np.nan,
                   rth_vol0=float((d.Volume[~ext] == 0).mean()), doji3_5m_eth=int(w3), doji3_40m_eth=int(w40), bars5m_eth=len(tb)))
fb = pd.DataFrame(fb)
fb.to_csv(ROOT / "results" / "audit_lookahead_symbols.csv", index=False)
out["symbols"] = dict(n=len(fb), ext_vol0_median=float(fb.ext_vol0.median()), ext_vol0_min=float(fb.ext_vol0.min()),
                      syms_ext_vol0_ge_99pct=int((fb.ext_vol0 >= 0.99).sum()),
                      doji3_5m_eth_total=int(fb.doji3_5m_eth.sum()), doji3_5m_eth_syms=int((fb.doji3_5m_eth > 0).sum()),
                      doji3_5m_eth_share_median=float((fb.doji3_5m_eth / fb.bars5m_eth).median()),
                      doji3_40m_eth_total=int(fb.doji3_40m_eth.sum()), doji3_40m_eth_syms=int((fb.doji3_40m_eth > 0).sum()),
                      top_doji=fb.sort_values("doji3_5m_eth", ascending=False).head(8)[["sym", "doji3_5m_eth", "bars5m_eth"]].to_dict("records"))

# 3. positions 미리보기·다음 봉 시가 체결 ---------------------------------------------------
tb = E.tf_bars(d5, 40, "eth"); tb = tb[tb.day >= E.WIN["5m"]["warm"]].reset_index(drop=True)
O, C, V = tb.O.values.astype(float), tb.C.values.astype(float), tb.V.values.astype(float)
lod = tb.lod.values; awake = np.ones(len(tb), np.bool_)
pos_fail = 0; pos_tests = 0; nextopen_fail = 0
for kind in E.RATIOS:
    for n in (3, 10, 30):
        p = E.ratio(O, C, V, n, kind)
        for follow in (True, False):
            for dayflat in (False, True):
                for b, s in ((0.5, 0.5), (0.7, 0.3)):
                    pos, ev = E.positions(O, C, p, lod, awake, b, s, follow, dayflat)
                    for _ in range(20):
                        j = int(rng.integers(1, len(O) - 1))
                        p2 = p.copy(); p2[j:] = rng.uniform(0, 1, len(O) - j)
                        pos2, _ = E.positions(O, C, p2, lod, awake, b, s, follow, dayflat)
                        pos_tests += 1
                        if not np.array_equal(pos[:j + 1], pos2[:j + 1]):
                            pos_fail += 1
                    # 체결 봉 j의 매수는 p[j−1]로 상태가 바뀌었거나(신규) dayflat 뒤 이어진 상태여야 한다
                    st = 0; states = np.zeros(len(O), int)
                    for j in range(len(O)):
                        pj = p[j]
                        if not np.isnan(pj):
                            if st == 0 and ((follow and pj > b) or (not follow and pj < 1 - b)):
                                st = 1
                            elif st == 1 and ((follow and pj < s) or (not follow and pj > 1 - s)):
                                st = 0
                        states[j] = st
                    for j in np.where(ev == 1)[0]:
                        if j == 0 or states[j - 1] != 1:
                            nextopen_fail += 1
                    for j in np.where(ev == -1)[0]:
                        if states[j - 1] != 0:
                            nextopen_fail += 1
out["positions"] = dict(tests=pos_tests, lookahead_fail=pos_fail, fill_not_after_signal=nextopen_fail)

# 4. tf_bars 미리보기·묶음 시각 ------------------------------------------------------------
tbars_fail = 0; tbars_tests = 0
for tf in (5, 40, 70, 125, 295, 360):
    for sess in ("eth", "rth"):
        a = E.tf_bars(d5, tf, sess)
        for _ in range(5):
            T = d5.index[int(rng.integers(100, len(d5) - 100))]
            d2 = d5.copy()
            fut = d2.index >= T
            d2.loc[fut, ["Open", "High", "Low", "Close"]] = rng.uniform(1, 2000, (fut.sum(), 4))
            d2.loc[fut, "Volume"] = rng.uniform(0, 1e7, fut.sum())
            b2 = E.tf_bars(d2, tf, sess)
            end = a.start + pd.to_timedelta(tf, unit="m")
            done = (end <= T).values
            # 시가 기준 묶음이 같은지 (세션 끝에서 잘린 봉도 end<=T로 충분)
            k = int(done.sum())
            tbars_tests += 1
            if not (np.allclose(a.O.values[:k], b2.O.values[:k]) and np.allclose(a.C.values[:k], b2.C.values[:k])
                    and np.allclose(a.V.values[:k], b2.V.values[:k])):
                tbars_fail += 1
a40 = E.tf_bars(d5, 40, "eth")
starts = sorted(set(a40.start.dt.strftime("%H:%M")))
a40r = E.tf_bars(d5, 40, "rth")
h60e = E.tf_bars(d1, 60, "eth")
lab = h60e.start.dt.strftime("%H:%M"); act = pd.DatetimeIndex(h60e.t).strftime("%H:%M")
mis = pd.Series(lab.values != np.asarray(act))
pairs = pd.DataFrame({"label": lab.values, "first_bar": np.asarray(act)})[mis.values].drop_duplicates().values.tolist()
out["tf_bars"] = dict(tests=tbars_tests, lookahead_fail=tbars_fail, eth40_starts=starts, n_eth40_starts=len(starts),
                      rth40_starts=sorted(set(a40r.start.dt.strftime("%H:%M"))),
                      eth60_from_1h_label_vs_first_bar_mismatch=pairs, eth60_bars_per_day=int(h60e.groupby("day").size().median()))
# 1시간봉 연장 묶음: 09:00 반 봉과 09:30 봉이 한 봉으로 합쳐지는가
k5 = h60e[h60e.start.dt.strftime("%H:%M") == "09:00"].head(1)
grp = d1[(d1.index.date == k5.day.iloc[0])]
g9 = grp[(grp.index.hour * 60 + grp.index.minute >= 540) & (grp.index.hour * 60 + grp.index.minute < 600)]
out["tf_bars"]["eth60_0900_group_members"] = [t.strftime("%H:%M") for t in g9.index]

# 5. 격자 칸 수 -----------------------------------------------------------------------------
g5 = pd.read_parquet(ROOT / "results" / "grid_MU_5m.parquet", columns=["cost"])
g1 = pd.read_parquet(ROOT / "results" / "grid_MU_1h.parquet", columns=["cost"])
out["grid_counts"] = dict(brief_5m=202752, product_5m=72 * 11 * 16 * 3 * 2 * 2 * 2, file_5m_cost0=int((g5.cost == 0).sum()),
                          brief_1h=16896, product_1h=6 * 11 * 16 * 3 * 2 * 2 * 2, file_1h_cost0=int((g1.cost == 0).sum()))

(ROOT / "results" / "audit_lookahead.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
