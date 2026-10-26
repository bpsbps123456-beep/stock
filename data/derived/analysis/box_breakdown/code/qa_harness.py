"""하네스 감사(QA). harness.py와 독립으로 다시 짠 계산과 비교하고, 미래 값 교란·순열 보정 시험을 돌린다.

실행: py qa_harness.py [replicate|trace|lookahead|terciles|perm|calib|earn|all]
결과는 표준 출력. 요약은 ../results/qa_harness.md에 사람이 옮겨 적는다.
특징 칸의 실제 통과 결과는 보지 않는다(가짜 특징·무작위 특징만 쓴다).
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

import harness as H


# ───────────────────────── 1. 독립 재계산(긴 형식 원자료에서 직접) ─────────────────────────
def independent_events(df_long, N, cal):
    """원자료(긴 형식)만으로 사건·청산·분류를 반복문으로 다시 계산한다. 반환 DataFrame."""
    out = []
    for s in H.EVENT_SYMS:
        g = df_long[df_long["sym"] == s].set_index("date").reindex(cal)
        o, h, l, c = (g[f].to_numpy(float) for f in ("Open", "High", "Low", "Close"))
        T = len(c)
        last_brk = -10 ** 9
        for i in range(T):
            if i < N or i < 21:
                continue
            win = l[i - N:i]
            if np.isnan(win).any() or np.isnan(c[i]):
                continue
            trs = []
            okatr = True
            for j in range(i - 20, i):
                if np.isnan(h[j]) or np.isnan(l[j]) or np.isnan(c[j - 1]):
                    okatr = False
                    break
                trs.append(max(h[j] - l[j], abs(h[j] - c[j - 1]), abs(l[j] - c[j - 1])))
            if not okatr:
                continue
            atr = sum(trs) / 20
            if atr <= 0:
                continue
            Lv = win.min()
            if not c[i] < Lv:
                continue
            new = (i - last_brk) > H.DEDUP
            last_brk = i
            if not new:
                continue
            rec = {"sym": s, "i": i, "L": Lv, "atr20": atr, "H": h[i - N:i].max()}
            for hh in (5, 10, 20):
                rec[f"r{hh}"] = _outcome(o, c, i, Lv, hh)
            rec["cls"] = _cls(l, c, i, Lv, atr)
            out.append(rec)
    return pd.DataFrame(out)


def _outcome(o, c, i, Lv, hh):
    T = len(c)
    if i + 1 >= T or np.isnan(o[i + 1]):
        return np.nan
    e = o[i + 1]
    for k in range(i + 1, i + hh):
        if k >= T:
            return np.nan
        if c[k] > Lv:
            return (e - o[k + 1]) / e - H.COST if k + 1 < T else np.nan
    if i + hh >= T:
        return np.nan
    return (e - c[i + hh]) / e - H.COST


def _cls(l, c, i, Lv, atr):
    T = len(c)
    reclaim = None
    first_true = None
    first_fake_touch = None
    for k in range(i + 1, i + 11):
        if k >= T:
            break
        if first_true is None and l[k] <= Lv - 2 * atr:
            first_true = k
        if first_fake_touch is None and l[k] <= Lv - atr:
            first_fake_touch = k
        if reclaim is None and c[k] > Lv:
            reclaim = k
    have10 = i + 10 < T
    # 진짜: 2ATR 닿은 날 ≤ 첫 되돌림 날(같은 날이면 진짜)
    if first_true is not None and (reclaim is None or first_true <= reclaim):
        return "진짜"
    if reclaim is not None:
        if reclaim <= i + 5 and (first_fake_touch is None or first_fake_touch > reclaim):
            return "가짜"
        return "중간"
    return "중간" if have10 else None


def cmd_replicate():
    df = H.build_daily()
    P = H.load_panel()
    for N in H.NS:
        ev = H.load_events(N)
        ind = independent_events(df, N, P.cal)
        a = ev.set_index(["sym", "i"]).sort_index()
        b = ind.set_index(["sym", "i"]).sort_index()
        only_a = a.index.difference(b.index)
        only_b = b.index.difference(a.index)
        common = a.index.intersection(b.index)
        a, b = a.loc[common], b.loc[common]
        mx = {k: float(np.nanmax(np.abs(a[k] - b[k]))) for k in ("L", "H", "atr20")}
        r = {}
        for hh in (5, 10, 20):
            x, y = a[f"short_ret_h{hh}"].to_numpy(float), b[f"r{hh}"].to_numpy(float)
            r[hh] = (int(np.sum(np.isnan(x) != np.isnan(y))), float(np.nanmax(np.abs(x - y))))
        cls_mis = int(((a["cls"].fillna("NA")) != (b["cls"].fillna("NA"))).sum())
        print(f"N={N}: 하네스 {len(ev)} / 독립 {len(ind)} / 하네스만 {len(only_a)} / 독립만 {len(only_b)} | "
              f"최대차 L {mx['L']:.2e} H {mx['H']:.2e} ATR {mx['atr20']:.2e} | "
              + " ".join(f"h{hh}: NaN불일치 {v[0]} 최대차 {v[1]:.2e}" for hh, v in r.items())
              + f" | 분류 불일치 {cls_mis}")


# ───────────────────────── 2. 손 추적 ─────────────────────────
def cmd_trace(picks=None):
    P = H.load_panel()
    ev = H.load_events(20)
    if picks is None:
        b = ev[H.in_period(ev, H.BACK) | H.in_period(ev, H.FRONT)]
        picks = []
        for want in (("reclaim", "가짜"), ("time", "진짜"), ("reclaim", "중간"), ("time", "중간")):
            x = b[(b["exit_reason"] == want[0]) & (b["cls"] == want[1])]
            if len(x):
                picks.append(x.iloc[len(x) // 2][["sym", "i"]].tolist())
        x = b[b["kind"] == "etf"]
        picks.append(x.iloc[0][["sym", "i"]].tolist())
    for s, i in picks:
        i = int(i)
        r = ev[(ev["sym"] == s) & (ev["i"] == i)].iloc[0]
        o, h, l, c = (P.col(f, s) for f in "OHLC")
        Lw = l[i - 20:i]
        argL = int(np.argmin(Lw)) + i - 20
        print(f"\n### {s} t={P.cal[i].date()} (i={i})")
        print(f"L20 = 최저 저가 {P.cal[i-20].date()}~{P.cal[i-1].date()} = {Lw.min():.4f} ({P.cal[argL].date()}) "
              f"| 하네스 L {r['L']:.4f} | ATR20(t−1) {r['atr20']:.4f}")
        prev = [k for k in range(i - 15, i) if c[k] < pd.Series(l).shift(1).rolling(20).min().iloc[k]]
        print(f"직전 15거래일 이탈일: {[str(P.cal[k].date()) for k in prev] or '없음'}")
        for k in range(i - 1, min(i + 12, len(c))):
            tag = "t" if k == i else (f"t+{k-i}" if k > i else f"t{k-i}")
            flag = []
            if k > i and c[k] > r["L"]:
                flag.append("종가>L")
            if k > i and l[k] <= r["L"] - 2 * r["atr20"]:
                flag.append("저가≤L−2ATR")
            elif k > i and l[k] <= r["L"] - r["atr20"]:
                flag.append("저가≤L−1ATR")
            print(f"  {tag:5} {P.cal[k].date()} O {o[k]:9.3f} H {h[k]:9.3f} L {l[k]:9.3f} C {c[k]:9.3f} {' '.join(flag)}")
        print(f"하네스: 진입 {r['entry_open']:.3f}(t+1 시가) 청산 {r['exit_reason']} {r['exit_date'].date()} {r['exit_type']} "
              f"h10 {r['short_ret_h10']*100:+.3f}% 분류 {r['cls']}")
        xi = int(r["exit_i"])
        px = o[xi] if r["exit_type"] == "open" else c[xi]
        print(f"손 계산: ({r['entry_open']:.4f} − {px:.4f}) / {r['entry_open']:.4f} − 0.001 = "
              f"{((r['entry_open'] - px) / r['entry_open'] - 0.001) * 100:+.3f}%")


# ───────────────────────── 3. 미래 값 교란 ─────────────────────────
INFO = ["L", "H", "close_t", "atr20"] + H.FEATURES_BASE + H.FEATURES_EXTRA


def cmd_lookahead(n_pick=6, seed=7):
    P = H.load_panel()
    er = H.earnings_react_dates(P.cal)
    rng = np.random.default_rng(seed)
    bad = 0
    for N in H.NS:
        base = H.load_events(N)
        cand = base[H.in_period(base, ("2025-01-01", "2026-09-29"))]
        dates = rng.choice(np.unique(cand["i"]), n_pick, replace=False)
        for i in sorted(dates):
            Q = H.Panel(P.cal, P.syms, *(a.copy() for a in (P.O, P.H, P.L, P.C, P.V)))
            for a in (Q.O, Q.H, Q.L, Q.C, Q.V):
                a[i + 1:] *= rng.uniform(0.6, 1.4, a[i + 1:].shape)
            Q.H[i + 1:] = np.fmax(Q.H[i + 1:], np.fmax(Q.O[i + 1:], Q.C[i + 1:]))
            Q.L[i + 1:] = np.fmin(Q.L[i + 1:], np.fmin(Q.O[i + 1:], Q.C[i + 1:]))
            alt = H.build_events(Q, N, earnings=er)
            a = base[base["i"] <= i].set_index(["sym", "i"]).sort_index()
            b = alt[alt["i"] <= i].set_index(["sym", "i"]).sort_index()
            same_idx = a.index.equals(b.index)
            diffcols = []
            if same_idx:
                for col in INFO:
                    x, y = a[col].to_numpy(float), b[col].to_numpy(float)
                    if not np.array_equal(np.isnan(x), np.isnan(y)) or np.nanmax(np.abs(x - y), initial=0) > 1e-12:
                        diffcols.append(col)
            ch_out = int((~np.isclose(a.loc[a.index.get_level_values(1) == i, "short_ret_h10"],
                                      b.loc[b.index.get_level_values(1) == i, "short_ret_h10"],
                                      equal_nan=True)).sum()) if same_idx else -1
            n_i = int((a.index.get_level_values(1) == i).sum())
            ok = same_idx and not diffcols
            bad += not ok
            print(f"N={N} t={P.cal[i].date()}: 그날 사건 {n_i}개, t 이전 사건 {len(a)}개 동일={same_idx}, "
                  f"바뀐 특징 {diffcols or '없음'}, (대조) 결과 h10 바뀐 사건 {ch_out}/{n_i}")
    print("미래 값 교란 실패 수:", bad)


# ───────────────────────── 4. 3분위 문턱 ─────────────────────────
def cmd_terciles():
    ev = H.load_events_all()
    feats = H.FEATURES_BASE
    c1 = dict(H.tercile_cells(ev, feats))
    ev2 = ev.copy()
    nonfront = ~H.in_period(ev2, H.FRONT)
    rng = np.random.default_rng(3)
    for f in feats:
        ev2.loc[nonfront, f] = rng.permutation(ev2.loc[nonfront, f].to_numpy())
    c2 = dict(H.tercile_cells(ev2, feats))
    fr = H.in_period(ev, H.FRONT)
    same_names = list(c1) == list(c2)
    same_front = all(np.array_equal(c1[k][fr].fillna(-1).to_numpy(), c2[k][fr].fillna(-1).to_numpy()) for k in c1)
    # 문턱을 앞 구간 값으로 직접 다시 계산해 이름의 문턱과 비교
    mism = 0
    for N in H.NS:
        g = ev[(ev["N"] == N) & fr]
        for f in feats:
            x = g[f].dropna().astype(float)
            if x.nunique() <= 2:
                continue
            q1, q2 = np.quantile(x, [1 / 3, 2 / 3])
            hi = [k for k in c1 if k.startswith(f"N{N}|{f}≥")][0]
            on = c1[hi][ev["N"] == N]
            want = (ev.loc[ev["N"] == N, f] >= q2).astype(float).where(ev.loc[ev["N"] == N, f].notna())
            mism += int((on.fillna(-1) != want.fillna(-1)).sum())
    print(f"칸 {len(c1)}개, 앞 밖 값을 섞어도 칸 이름(문턱) 같음={same_names}, 앞 구간 표지 같음={same_front}, "
          f"앞 구간 분위수 직접 계산과 불일치 {mism}")


# ───────────────────────── 5. 순열 지도 점검 ─────────────────────────
def cmd_perm():
    ev = H.load_events_all()
    ev = ev[ev["short_ret_h10"].notna()]
    b = ev[H.in_period(ev, H.BACK)]
    st = b[["N", "kind"]].astype(str).agg("|".join, axis=1).to_numpy()
    dt = b["t"].to_numpy()
    rng = np.random.default_rng(0)
    maps = H.block_maps(dt, st, 200, rng)
    keep_st = all(np.array_equal(st[m], st) for m in maps)
    one_src = all((pd.Series(dt[m]).groupby([dt, st]).nunique() == 1).all() for m in maps)
    # 층끼리 같은 원천 날짜를 받는 비율(같은 날짜가 여러 층에 있을 때)
    share = []
    for m in maps:
        src = pd.DataFrame({"d": dt, "s": st, "src": dt[m]}).drop_duplicates(["d", "s"])
        g = src.groupby("d")["src"].nunique()
        multi = src.groupby("d")["s"].nunique() > 1
        share.append(float((g[multi] == 1).mean()))
    ident = np.mean([np.mean(m == np.arange(len(m))) for m in maps])
    print(f"뒤 구간 사건 {len(b)} / 날짜 {len(np.unique(dt))} / 층 {len(np.unique(st))}")
    print(f"층 유지={keep_st}, 한 날짜·층은 한 원천 날짜={one_src}, 제자리 행 비율 {ident:.3f}")
    print(f"(행 지도) 여러 층에 걸친 날짜가 층끼리 같은 원천 날짜를 받는 비율(평균) {np.mean(share):.3f}")
    # 실제 날짜 고정 칸(N을 가로지르는 시장 맥락 3분위)을 Permuter.apply로 옮기면 같은 날 표지가 층을 가로질러 같은가
    cells = H.tercile_cells(ev, ["vix", "breadth50", "qqq_above50"], by=("kind",))
    M = H._cell_matrix(b, cells)
    g = H._codes(dt)
    dcst = H.date_constant(M, g)
    Pm = H.Permuter(dt, st, 200, np.random.default_rng(1))
    bad = 0
    for r in range(len(Pm)):
        Mp = Pm.apply(M, r, dcst)
        for j in np.flatnonzero(dcst):
            bad += int((pd.Series(Mp[:, j]).groupby(g).nunique() > 1).any())
    print(f"날짜 고정 칸 {int(dcst.sum())}/{len(cells)}개, Permuter 200회 × 칸에서 같은 날 표지가 갈린 경우 {bad}")


# ───────────────────────── 6. 가짜 특징 보정 시험 ─────────────────────────
def old_block_maps(dates, strata, k, rng):
    """수정 전 block_maps(층마다 날짜 순위 정렬 대응). 비교용."""
    dc = H._codes(dates)
    sc = H._codes(strata)
    n = len(dc)
    nd = dc.max() + 1 if n else 0
    groups = H._groups(dc, sc)
    maps = np.empty((k, n), dtype=int)
    for r in range(k):
        rank = rng.permutation(nd)
        out = np.arange(n)
        for s, (u, ev) in groups.items():
            src_dates = u[np.argsort(rank[u], kind="stable")]
            for d_t, d_s in zip(u, src_dates):
                tgt, src = ev[d_t], ev[d_s]
                p = rng.permutation(src)
                if len(p) < len(tgt):
                    p = np.r_[p, rng.choice(src, len(tgt) - len(p), replace=True)]
                out[tgt] = p[:len(tgt)]
        maps[r] = out
    return maps


def old_perm_p(M, y, bank, obs):
    cnt = np.zeros(M.shape[1])
    for mp in bank:
        d = H.cluster_diff_fast(M[mp], y)
        cnt += np.nan_to_num(d >= obs - 1e-15, nan=0)
    return (1 + cnt) / (1 + len(bank))


def placebo_cells(ev, kind, n, rng, min_on=15, min_dates=8):
    """y와 무관한 가짜 칸 n개(1/0). event_iid: 사건마다 1/3 확률. date_iid: 날짜마다 1/3 확률(그날 사건 공통).
    date_ar: 거래일 AR(1)(phi 0.9~0.99) 계열의 위/아래 3분위(그날 사건 공통, 국면처럼 지속).
    판정 고르기 조건(켜짐 ≥ 15건, 켜짐 날짜 ≥ 8일, 꺼짐 ≥ 15건)을 만족하는 칸만 남긴다."""
    T = len(H.load_panel().cal)
    ti = ev["i"].to_numpy()
    u, inv = np.unique(ti, return_inverse=True)
    out = []
    tries = 0
    while len(out) < n and tries < 50 * n:
        tries += 1
        if kind == "event_iid":
            m = (rng.random(len(ev)) < 1 / 3).astype(float)
        elif kind == "date_iid":
            m = (rng.random(len(u)) < 1 / 3).astype(float)[inv]
        else:
            phi = rng.uniform(0.9, 0.99)
            x = np.zeros(T)
            e = rng.normal(size=T)
            for k in range(1, T):
                x[k] = phi * x[k - 1] + e[k]
            v = x[ti]
            q = np.quantile(x, 2 / 3) if rng.random() < 0.5 else np.quantile(x, 1 / 3)
            m = ((v >= q) if rng.random() < 0.5 else (v <= q)).astype(float)
        on = m == 1
        if on.sum() >= min_on and (~on).sum() >= min_on and len(np.unique(ti[on])) >= min_dates:
            out.append(m)
    return np.array(out).T


def cmd_calib(n=400, nperm=200):
    rows = []
    for label, ev in (("N20", H.load_events(20)), ("N20·40·60 함께", H.load_events_all())):
        ev = ev[ev["short_ret_h10"].notna()]
        for per_name, per in (("뒤", H.BACK), ("앞", H.FRONT), ("2025", H.AUX)):
            b = ev[H.in_period(ev, per)].reset_index(drop=True)
            yb = b["short_ret_h10"].to_numpy(float)
            dtb = b["t"].to_numpy()
            stb = b[["N", "kind"]].astype(str).agg("|".join, axis=1).to_numpy()
            gb = H._codes(dtb)
            for kind in ("event_iid", "date_iid", "date_ar"):
                rng = np.random.default_rng(11)
                M = placebo_cells(b, kind, n, rng)
                _, _, m1, _, d, t = H.cluster_diff(M, yb, gb)
                bank_old = old_block_maps(dtb, stb, nperm, np.random.default_rng(5))
                bank_new = H.Permuter(dtb, stb, nperm, np.random.default_rng(5), "iid")
                bank_sh = H.Permuter(dtb, stb, rng=np.random.default_rng(5), mode="shift")
                dc = H.date_constant(M, gb)
                p_old = old_perm_p(M, yb, bank_old, d)
                p_new = H._perm_p(M, yb, gb, bank_new, d, dc)
                p_sh = H._perm_p(M, yb, gb, bank_sh, d, dc)
                rows.append((label, per_name, kind, len(b), len(np.unique(dtb)), M.shape[1],
                             np.mean(t >= 2), np.mean(p_old < 0.05), np.mean(p_new < 0.05),
                             np.mean((t >= 2) & (p_new < 0.05)), np.mean(p_sh < 0.05),
                             np.mean((t >= 2) & (p_new < 0.05) & (p_sh < 0.05))))
    print("| 사건표 | 구간 | 가짜 특징 | 사건 | 날짜 | 칸 | P(t≥2) | P(p<.05) 수정 전 | P(p<.05) 수정 뒤 | "
          "P(t≥2 & p<.05) | P(원형 이동 p<.05) | P(t≥2 & p<.05 & 이동 p<.05) |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print("| " + " | ".join(str(x) if not isinstance(x, float) else f"{x:.3f}" for x in r) + " |")


# ───────────────────────── 7. 실적일 ─────────────────────────
def cmd_earn():
    P = H.load_panel()
    er = H.earnings_react_dates(P.cal)
    e = pd.read_csv(H.ANALYSIS / "box_trade/results/earnings_cache.csv")
    print("열:", e.columns.tolist(), "| 하네스가 쓰는 열: ticker, earnings_dt(날짜·시각)만")
    loc = pd.to_datetime(e["earnings_dt"].str[:19])
    print("시간대 꼬리:", e["earnings_dt"].str[19:].value_counts().to_dict())
    before = int((loc < P.cal[0]).sum())
    print(f"달력 시작 전 실적 {before}행 → 하네스 수정 전에는 searchsorted가 0번 날(2022-06-01)로 보냈다")
    # 반응일이 맞는지: 실제 가격 반응(가장 큰 |시가 갭|)이 반응일인지 사후 점검(설명용)
    hit, tot = 0, 0
    for s, idx in er.items():
        if s not in P.j:
            continue
        o, c = P.col("O", s), P.col("C", s)
        gap = np.abs(np.log(o[1:] / c[:-1]))
        for ri in idx:
            if ri < 2 or ri + 1 >= len(c) or P.cal[ri] < pd.Timestamp("2024-01-01"):
                continue
            w = gap[ri - 2:ri + 1]   # 갭 인덱스 k = 날 k+1의 시가 갭 → 날 ri−1, ri, ri+1
            if np.isnan(w).any():
                continue
            tot += 1
            hit += int(np.argmax(w) == 1)
    print(f"2024 이후 실적 {tot}건 중 반응일 시가 갭이 앞뒤 날보다 큰 비율 {hit / tot:.3f}")
    for N in H.NS:
        ev = H.load_events(N)
        st = ev[ev["kind"] == "stock"]
        fwd = st["earn_dist"].between(1, 3).sum()
        past = st["earn_dist"].between(-3, 0).sum()
        print(f"N={N}: earn_pm3=1 {int(st['earn_pm3'].sum())}건 (반응일 t−3..t {int(past)}, 예정 t+1..t+3 {int(fwd)})")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    fns = {"replicate": cmd_replicate, "trace": cmd_trace, "lookahead": cmd_lookahead, "terciles": cmd_terciles,
           "perm": cmd_perm, "calib": cmd_calib, "earn": cmd_earn}
    for k, f in fns.items():
        if cmd in (k, "all"):
            print(f"\n==== {k} ====")
            f()
    print(f"\n({time.time() - t0:.1f}초)")
