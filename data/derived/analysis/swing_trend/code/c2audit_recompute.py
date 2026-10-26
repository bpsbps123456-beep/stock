"""c2 audit 1: events.py 독립 재계산 (events.py·swing.py를 import하지 않는다).

종목 5개(MU·BE·NBIS·NVDA·AMD) × 두 판정(ZZA3·ZZP10)의 사건 전부에 대해 단서 18개와 y1·y10·y20을
처음부터 다시 계산해 results/events_*.parquet과 대조한다. 시장 단서(M1·M2·M5)는 주식 136개 전부로 다시 계산한다.
정의는 BRIEF2와 events.py 문서에 맞춘다(백분위·자기상관 포함). 정의상 갈림이 있는 곳은 '대안 정의'로 따로 잰다.

py code/c2audit_recompute.py → results/c2audit_recompute.csv(사건별 차이), results/c2audit_recompute_summary.csv
"""
from __future__ import annotations

import math
import pathlib
from datetime import date

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
ETF = {"SPY", "QQQ", "SOXX", "SMH", "IGV", "IWM", "XBI", "ARKG"}
FOCUS = ["MU", "BE", "NBIS", "NVDA", "AMD"]
PERIODS = {"P1": (date(2026, 1, 2), date(2026, 5, 29)), "P2": (date(2026, 6, 1), date(2026, 10, 22)),
           "Y25": (date(2025, 1, 2), date(2025, 12, 31)), "Y24": (date(2024, 1, 2), date(2024, 12, 31)),
           "Y23": (date(2023, 1, 3), date(2023, 12, 29))}
FEATS = ["M1", "M2", "M3a", "M3b", "M4a", "M4b", "M4c", "M5",
         "S1", "S2", "S3a", "S3b", "S4", "S5", "S6", "S7", "S8", "S9"]
TOL = 1e-9


def which_period(d):
    hit = [k for k, (a, b) in PERIODS.items() if a <= d <= b]
    return hit[0] if hit else None


def wilder(H, L, C, n=20):
    """ATR: 첫 값 = 처음 n개 TR 평균(TR[0] = H−L), 그다음 Wilder 평활."""
    m = len(C)
    tr = np.empty(m)
    tr[0] = H[0] - L[0]
    tr[1:] = np.max(np.vstack([H[1:] - L[1:], np.abs(H[1:] - C[:-1]), np.abs(L[1:] - C[:-1])]), axis=0)
    out = np.full(m, np.nan)
    if m >= n:
        out[n - 1] = math.fsum(tr[:n]) / n
        for i in range(n, m):
            out[i] = out[i - 1] + (tr[i] - out[i - 1]) / n
    return out


def zz(C, thr=None, pct=None):
    """실시간 지그재그(파이썬 순수 구현). thr: 날별 가격 문턱(NaN이면 극점만 갱신), pct: 비율 문턱.
    반환: 상태 배열(1 상승 확정 뒤, 0 하락 확정 뒤 또는 시작), [(극점, 확정, 방향)]."""
    n = len(C)
    st = np.zeros(n, np.int64)
    out = []
    up = False
    ext, ei = C[0], 0
    for i in range(n):
        x = C[i]
        if (not up and x < ext) or (up and x > ext):
            ext, ei = x, i
        if thr is not None:
            if np.isnan(thr[i]):
                st[i] = int(up)
                continue
            hit = x >= ext + thr[i] if not up else x <= ext - thr[i]
        else:
            hit = x >= ext * (1 + pct) if not up else x <= ext * (1 - pct)
        if hit:
            out.append((ei, i, -1 if up else 1))
            up = not up
            ext, ei = x, i
        st[i] = int(up)
    return st, out


def reaction_dates(er: pd.DataFrame, dates: list) -> set:
    """amc는 발표일보다 뒤 첫 거래일, 그 밖(bmo·day)은 발표일 이후(같은 날 포함) 첫 거래일."""
    out = set()
    for d0, ses in zip(er.date_et, er.session):
        cand = [d for d in dates if (d > d0 if ses == "amc" else d >= d0)]
        if cand:
            out.add(cand[0])
    return out


def pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    den = math.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else np.nan


def main():
    raw = pd.read_parquet(ROOT / "cache" / "daily.parquet")
    vix = pd.read_parquet(ROOT / "cache" / "vix.parquet")
    earn = pd.read_parquet(ROOT / "cache" / "earnings.parquet")
    G = {s: g.sort_values("date").reset_index(drop=True) for s, g in raw.groupby("sym")}
    stocks = sorted(set(G) - ETF)
    cal = sorted(set(d for s in stocks for d in G[s].date))
    pos_of = {d: i for i, d in enumerate(cal)}
    Tn = len(cal)

    # SOXX·VIX 달력값: 달력 날의 값, 없으면 앞 달력 날 값(앞값 채움)
    sx_map = dict(zip(G["SOXX"].date, G["SOXX"].C.astype(float)))
    sx = np.full(Tn, np.nan)
    vx = np.full(Tn, np.nan)
    v3 = np.full(Tn, np.nan)
    vmap = {d: (a, b) for d, a, b in zip(vix.index, vix.VIX, vix.VIX3M)}
    for i, d in enumerate(cal):
        if d in sx_map:
            sx[i] = sx_map[d]
        elif i > 0:
            sx[i] = sx[i - 1]
        if d in vmap:
            vx[i], v3[i] = vmap[d]
        elif i > 0:
            vx[i], v3[i] = vx[i - 1], v3[i - 1]
    M3a = np.array([float(sx[i] > np.mean(sx[i - 49:i + 1])) if i >= 49 else 0.0 for i in range(Tn)])
    M3b = np.array([math.log(sx[i] / sx[i - 20]) if i >= 20 else np.nan for i in range(Tn)])
    M4a = vx
    M4b = np.array([math.log(vx[i] / vx[i - 5]) if i >= 5 else np.nan for i in range(Tn)])
    M4c = vx / v3

    out_rows, ycheck = [], []
    for det in ("ZZA3", "ZZP10"):
        # 판정별 시장 단서 M1·M2·M5
        n_have = np.zeros(Tn)
        n_above = np.zeros(Tn)
        cnt_up = np.zeros(Tn)
        cnt_dn = np.zeros(Tn)
        tm_sum = np.zeros(Tn)
        tm_n = np.zeros(Tn)
        per = {}
        for s in stocks:
            g = G[s]
            O, H, L, C, V = (g[k].to_numpy(float) for k in "OHLCV")
            a = wilder(H, L, C, 20)
            st, piv = zz(C, thr=3.0 * a) if det == "ZZA3" else zz(C, pct=0.10)
            rows = [pos_of[d] for d in g.date]
            ups = {rows[c] for _, c, t in piv if t == 1}
            dns = {rows[c] for _, c, t in piv if t == -1}
            for k in range(len(C)):
                r = rows[k]
                if k >= 19:
                    n_have[r] += 1
                    n_above[r] += C[k] > math.fsum(C[k - 19:k + 1]) / 20
                    # 같은 방향 확정: 달력 r−2..r 안
                    cnt_up[r] += any(x in ups for x in (r - 2, r - 1, r))
                    cnt_dn[r] += any(x in dns for x in (r - 2, r - 1, r))
                if k >= 1:
                    tm_sum[r] += (2 * st[k - 1] - 1) * (C[k] / C[k - 1] - 1)
                    tm_n[r] += 1
            per[s] = (g, O, H, L, C, V, a, st, piv, rows)
        M2 = np.where(n_have > 0, n_above / np.maximum(n_have, 1), np.nan)
        M1u = cnt_up / np.maximum(n_have, 1)
        M1d = cnt_dn / np.maximum(n_have, 1)
        tmean = np.where(tm_n > 0, tm_sum / np.maximum(tm_n, 1), np.nan)
        M5 = np.full(Tn, np.nan)
        for i in range(Tn):
            w = tmean[max(0, i - 39):i + 1]
            w = w[~np.isnan(w)]
            if len(w) >= 30:
                M5[i] = math.fsum(w)

        ev = pd.read_parquet(ROOT / "results" / f"events_{det}.parquet")
        for s in FOCUS:
            g, O, H, L, C, V, a, st, piv, rows = per[s]
            dates = list(g.date)
            n = len(C)
            rd = reaction_dates(earn[earn.sym == s], dates)
            alt_ok = all(piv[k][2] == -piv[k - 1][2] for k in range(1, len(piv)))
            mine = []
            for e, (p, c, t) in enumerate(piv):
                d = dates[c]
                pr = which_period(d)
                if pr is None or c < 260 or c + 1 >= n:
                    continue
                r = rows[c]
                nxt = next((piv[k] for k in range(e + 1, len(piv)) if piv[k][2] == -t), None)
                c2 = nxt[1] if nxt else None
                cens = c2 is None or c2 + 1 >= n
                exit_px = C[-1] if cens else O[c2 + 1]
                y1 = t * math.log(exit_px / O[c + 1])
                y10 = t * math.log(O[c + 11] / O[c + 1]) if c + 11 < n else np.nan
                y20 = t * math.log(O[c + 21] / O[c + 1]) if c + 21 < n else np.nan
                pp = piv[e - 1][0] if e > 0 else None
                vr = lambda i: V[i] / (math.fsum(V[i - 20:i]) / 20) if i >= 20 else np.nan  # noqa: E731
                win = np.array([math.log(C[i] / C[i - 1]) for i in range(max(c - 59, 1), c + 1)])
                s7 = sum(pearson(win[k:], win[:-k]) for k in range(1, 6)) if len(win) > 30 else np.nan
                ap = a / C
                w9 = ap[c - 251:c + 1]
                s9 = float(np.mean(w9[:-1] < w9[-1]))                       # events.py 정의(NaN은 '작지 않음', 분모 251)
                w9c = w9[:-1][~np.isnan(w9[:-1])]
                s9_clean = float(np.mean(w9c < w9[-1]))                     # 대안: NaN 제외
                f = dict(
                    M1=(M1u if t == 1 else M1d)[r], M2=M2[r], M3a=M3a[r], M3b=M3b[r], M4a=M4a[r], M4b=M4b[r], M4c=M4c[r], M5=M5[r],
                    S1=c - p,
                    S2=abs(C[p] - C[pp]) / a[p] if pp is not None and a[p] > 0 else np.nan,
                    S3a=vr(c), S3b=np.nanmax([vr(i) for i in range(p, c + 1)]),
                    S4=t * (O[c] - C[c - 1]) / a[c - 1] if a[c - 1] > 0 else np.nan,
                    S5=math.log(C[c] / C[c - 20]) - M3b[r],
                    S6=float(any(dates[i] in rd for i in range(p, c + 1))),
                    S7=s7, S8=C[c] / H[c - 251:c + 1].max() - 1, S9=s9,
                )
                mine.append(dict(sym=s, date=d, type=t, period=pr, pivot_date=dates[p], conf_idx=c,
                                 exit_date=dates[c2] if c2 is not None else None, censored=cens,
                                 y1=y1, y10=y10, y20=y20, S9_clean=s9_clean, alt_ok=alt_ok, **f))
            mine = pd.DataFrame(mine)
            ref = ev[ev.sym == s].copy()
            key = ["sym", "date", "type"]
            m = mine.merge(ref, on=key, how="outer", suffixes=("_me", "_ev"), indicator=True)
            only_me = int((m._merge == "left_only").sum())
            only_ev = int((m._merge == "right_only").sum())
            mm = m[m._merge == "both"]
            for _, row in mm.iterrows():
                diffs = {}
                for col in FEATS + ["y1", "y10", "y20", "conf_idx"]:
                    x, y = float(row[col + "_me"]), float(row[col + "_ev"])
                    diffs[col] = 0.0 if (np.isnan(x) and np.isnan(y)) else (np.inf if np.isnan(x) != np.isnan(y) else abs(x - y))
                same_meta = (row.period_me == row.period_ev) and (row.pivot_date_me == row.pivot_date_ev) and \
                            (row.exit_date_me == row.exit_date_ev) and (bool(row.censored_me) == bool(row.censored_ev))
                out_rows.append(dict(det=det, sym=s, date=row.date, type=row.type, period=row.period_ev,
                                     max_diff=max(diffs.values()), worst=max(diffs, key=diffs.get), meta_same=same_meta,
                                     S9_clean_minus_S9=row.S9_clean - row.S9_ev, **{f"d_{k}": v for k, v in diffs.items()}))
            ycheck.append(dict(det=det, sym=s, events_me=len(mine), events_ev=len(ref), only_me=only_me, only_ev=only_ev,
                               alternating=alt_ok))
    R = pd.DataFrame(out_rows)
    R.to_csv(ROOT / "results" / "c2audit_recompute.csv", index=False)
    summ = []
    for (det, s), g in R.groupby(["det", "sym"]):
        summ.append(dict(det=det, sym=s, n=len(g), n_over_tol=int((g.max_diff > TOL).sum()), max_diff=g.max_diff.max(),
                         meta_mismatch=int((~g.meta_same).sum()), s9_nan_window=int((g.S9_clean_minus_S9.abs() > 0).sum())))
    S = pd.DataFrame(summ).merge(pd.DataFrame(ycheck), on=["det", "sym"])
    S.to_csv(ROOT / "results" / "c2audit_recompute_summary.csv", index=False)
    print(S.to_string())
    print("전체", len(R), "건, 허용 초과", int((R.max_diff > TOL).sum()), "최대 차이", R.max_diff.max())
    print("단서별 최대 차이", {k[2:]: float(R[k].max()) for k in R.columns if k.startswith("d_")})


if __name__ == "__main__":
    main()
