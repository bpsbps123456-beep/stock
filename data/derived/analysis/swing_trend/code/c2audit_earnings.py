"""c2 audit 4: 실적일 자료 품질(cache/earnings.parquet)과 S6 영향.

- 날짜·세션 검증: 시가 갭 log(O/전날 C)에서 그날 주식 136개 갭의 중앙값(시장 공통)을 뺀 |잔차 갭|을 쓴다.
  발표일 d의 '그날 종가까지 반응' 점수 = max(d 시가 |잔차 갭|, d 장중 |log C/O|)(장중 발표 포함)와 '장 마감 뒤 반응'
  점수 = d 다음 거래일 시가 |잔차 갭|을 각각 직전 60거래일 중앙값의 배수(z)로 잰다. 큰 쪽이 4배 이상이고 작은 쪽의
  2배 이상이면 그쪽을 실제로 본다(그 밖은 '약함'). amc 표의 '장 전' 판정은 갭만 쓴다(그날 장중 움직임은 발표 전이므로).
  하루 어긋남(d−1 시가 또는 d+2 시가)이 6배 이상이고 두 후보 최대의 3배 이상이면 '날짜 어긋남 의심'으로 따로 센다(S6 수정에는 안 씀).
- 이른 쪽 오류(표는 장 전·장중인데 실제는 장 마감 뒤): 코드가 반응일을 하루 일찍 잡는다 → 확정일 저녁 발표를
  확정일 종가 전 정보로 넣는 꼴(BRIEF2 기준 누출). 늦은 쪽 오류는 S6 잡음.
- 중복(같은 종목 20일 안 두 줄)·미래 예정일·분기 빠짐(연속 간격 120일 초과).
- S6 영향: 위 판정으로 고친 반응일로 S6를 다시 매겨 사건별로 바뀌는 수를 센다(두 판정, 구간별).

py code/c2audit_earnings.py → results/c2audit_earn_rows.csv, results/c2audit_earn_s6.csv, results/c2audit_earn_focus.csv
"""
from __future__ import annotations

import pathlib
from datetime import date

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
ETF = {"SPY", "QQQ", "SOXX", "SMH", "IGV", "IWM", "XBI", "ARKG"}
FOCUS = ["MU", "BE", "NBIS", "NVDA", "AMD"]
END = date(2026, 10, 22)
START = date(2022, 9, 1)   # 사건(2023-06~)의 극점~확정 창이 닿는 가장 이른 날보다 앞


def main():
    raw = pd.read_parquet(ROOT / "cache" / "daily.parquet")
    earn = pd.read_parquet(ROOT / "cache" / "earnings.parquet").sort_values(["sym", "date_et"]).reset_index(drop=True)
    G = {s: g.sort_values("date").reset_index(drop=True) for s, g in raw.groupby("sym")}
    stocks = sorted(set(G) - ETF)
    sg = pd.concat([pd.Series(np.log(g.O.values[1:] / g.C.values[:-1]), index=g.date.values[1:]) for s, g in G.items() if s not in ETF], axis=1)
    mkt_gap = sg.median(axis=1)

    # 1) 표 전체 기본 점검
    fut = earn[earn.date_et > END]
    print("미래 예정일:", len(fut), "줄, 종목", fut.sym.nunique(), "/ 세션", fut.session.value_counts().to_dict())
    w = earn[(earn.date_et >= START) & (earn.date_et <= END)].copy()
    w["dd"] = w.groupby("sym").date_et.diff().apply(lambda x: x.days if pd.notna(x) else np.nan)
    dup = w[w.dd <= 20]
    print("창 안 20일 이내 중복:", len(dup), dup[["sym", "date_et", "session", "dd"]].to_dict("records"))
    wk = w[pd.to_datetime(w.date_et).dt.dayofweek >= 5]
    print("창 안 주말 날짜:", len(wk), wk[["sym", "date_et", "session"]].to_dict("records"))

    miss = []
    for s in stocks:
        g = G[s]
        d0 = max(g.date.iloc[0], START)
        e = sorted(earn[(earn.sym == s) & (earn.date_et >= d0) & (earn.date_et <= END)].date_et)
        pts = [d0] + e + [END]
        for a, b in zip(pts[:-1], pts[1:]):
            if (b - a).days > 120:
                miss.append(dict(sym=s, from_=a, to=b, days=(b - a).days))
    M = pd.DataFrame(miss)
    print("분기 빠짐(간격 120일 초과):", len(M), "구간,", M.sym.nunique() if len(M) else 0, "종목")
    if len(M):
        print(M.to_string())

    # 2) 갭으로 세션·날짜 검증
    rows = []
    for _, r in w.iterrows():
        if r.sym not in G:
            continue
        g = G[r.sym]
        ds = list(g.date)
        O, C = g.O.to_numpy(float), g.C.to_numpy(float)
        n = len(ds)
        i0 = int(np.searchsorted(np.array(ds), r.date_et, side="left"))
        if i0 >= n - 1 or i0 < 65:
            continue
        trading = ds[i0] == r.date_et
        mg = mkt_gap.reindex(ds).fillna(0).values
        gap = np.abs(np.r_[np.nan, np.log(O[1:] / C[:-1])] - mg)
        intr = np.abs(np.log(C / O))
        base = np.nanmedian(gap[i0 - 62:i0 - 2])
        base_i = np.nanmedian(intr[i0 - 62:i0 - 2])
        z = gap / base if base > 0 else gap * np.nan
        rb = i0
        ra = i0 + 1 if trading else i0
        code = ra if r.session == "amc" else rb
        zi = intr / base_i if base_i > 0 else intr * np.nan
        zb = z[rb] if r.session == "amc" else max(z[rb], zi[rb])
        za = z[ra]
        side = [k for k in (i0 - 1, ra + 1) if 1 <= k < n]
        am = max(side, key=lambda k: z[k]) if side else rb
        if not trading:
            inferred = "non_trading_day"
            actual = rb if z[rb] >= 4 else None
        elif max(zb, za) < 4 or max(zb, za) < 2 * min(zb, za):
            inferred, actual = "weak", None
        else:
            inferred = "pre" if zb > za else "amc"
            actual = rb if zb > za else ra
        misdated = bool(side) and z[am] >= 6 and z[am] >= 3 * max(zb, za)
        label = "amc" if r.session == "amc" else "pre"
        if actual is None:
            verdict = "weak"
        elif actual == code:
            verdict = "ok"
        elif actual > code:
            verdict = "code_early"     # 코드 반응일이 실제보다 이르다(누출형)
        else:
            verdict = "code_late"
        rows.append(dict(sym=r.sym, date_et=r.date_et, hm=r.ts_et.strftime("%H:%M"), session=r.session, label=label,
                         code_day=ds[code], actual_day=ds[actual] if actual is not None else None,
                         z_pre=round(float(zb), 2), z_amc=round(float(za), 2), z_intraday_pre=round(float(intr[rb] / base_i), 2) if base_i > 0 else np.nan,
                         z_max_win=round(float(z[am]), 2), argmax_off=am - i0, inferred=inferred, misdated=misdated, verdict=verdict))
    R = pd.DataFrame(rows)
    R.to_csv(ROOT / "results" / "c2audit_earn_rows.csv", index=False)
    print("검증한 줄", len(R), "| 판정", R.verdict.value_counts().to_dict(), "| 날짜 어긋남 의심", int(R.misdated.sum()))
    print(R[R.misdated].to_string())
    print(pd.crosstab(R.session, R.verdict))
    print("이른 쪽 오류(누출형):")
    print(R[R.verdict == "code_early"].to_string())
    print("늦은 쪽 오류 예:")
    print(R[R.verdict == "code_late"].head(30).to_string())
    print("'day' 세션 줄:")
    print(R[R.session == "day"].to_string())
    F = R[R.sym.isin(FOCUS) & (R.date_et >= date(2024, 6, 1))]
    F.to_csv(ROOT / "results" / "c2audit_earn_focus.csv", index=False)
    print("5종목:")
    print(F.to_string())

    # 3) S6 영향: 코드 반응일 대 고친 반응일
    code_days = {s: set() for s in stocks}
    fix_days = {s: set() for s in stocks}
    early_days = {s: set() for s in stocks}
    for _, r in earn.iterrows():
        if r.sym not in G:
            continue
        ds = np.array(G[r.sym].date)
        i = int(np.searchsorted(ds, r.date_et, side="left"))
        if r.session == "amc" and i < len(ds) and ds[i] == r.date_et:
            i += 1
        if i < len(ds):
            code_days[r.sym].add(ds[i])
    for s in stocks:
        fix_days[s] = set(code_days[s])
    for _, r in R.iterrows():
        if r.verdict in ("code_early", "code_late"):
            fix_days[r.sym].discard(r.code_day)
            fix_days[r.sym].add(r.actual_day)
            if r.verdict == "code_early":
                early_days[r.sym].add(r.code_day)
    out = []
    for det in ("ZZA3", "ZZP10"):
        ev = pd.read_parquet(ROOT / "results" / f"events_{det}.parquet")
        for _, e in ev.iterrows():
            ds = G[e.sym].date
            seg = set(ds[(ds >= e.pivot_date) & (ds <= e.date)])
            s_code = float(bool(seg & code_days[e.sym]))
            s_fix = float(bool(seg & fix_days[e.sym]))
            # 누출형: 코드 반응일이 확정일 당일인데 실제 반응은 다음 날(그날 저녁 발표)
            leak = e.date in early_days[e.sym]
            out.append(dict(det=det, sym=e.sym, date=e.date, type=e.type, period=e.period, S6=e.S6, S6_code=s_code,
                            S6_fix=s_fix, leak_on_c=leak and s_code == 1 and s_fix == 0))
    S = pd.DataFrame(out)
    if len(M):
        ev3 = pd.concat([pd.read_parquet(ROOT / "results" / f"events_{d}.parquet") for d in ("ZZA3", "ZZP10")])
        hit = []
        for _, m in M.iterrows():
            q = ev3[(ev3.sym == m.sym) & (ev3.date > m.from_ + pd.Timedelta(days=95)) & (ev3.date < m.to)]
            hit.append(dict(sym=m.sym, from_=m.from_, to=m.to, n_events=len(q), periods=q.period.value_counts().to_dict()))
        print("분기 빠짐 구간 안(앞 실적 +95일 뒤) 사건 수(두 판정 합):")
        print(pd.DataFrame(hit).to_string())
    S.to_csv(ROOT / "results" / "c2audit_earn_s6.csv", index=False)
    print("S6 재현(코드 규칙) 불일치:", int((S.S6 != S.S6_code).sum()))
    t = S.assign(chg=S.S6 != S.S6_fix, up=(S.S6 == 0) & (S.S6_fix == 1), dn=(S.S6 == 1) & (S.S6_fix == 0)) \
        .groupby(["det", "period"]).agg(n=("S6", "size"), S6_on=("S6", "sum"), changed=("chg", "sum"), to1=("up", "sum"),
                                        to0=("dn", "sum"), leak_on_c=("leak_on_c", "sum"))
    print(t.to_string())


if __name__ == "__main__":
    main()
