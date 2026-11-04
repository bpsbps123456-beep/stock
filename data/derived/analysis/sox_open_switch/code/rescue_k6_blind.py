"""K6 (BRIEF2 F1-A) 눈가림 구현 B — BRIEF7 재확인용.

원래 연구 코드(lib·fam_F1·stage5·redfirst·ev4_build·c3_hyp)와 rescue_k6.py를 import하지 않는다.
정의 출처: BRIEF2.md F1(시가 범위 돌파), report.md §'F1-A'(09:30 봉 방향, 돌파 종가 확인 진입, 목표 2R, 손절 없음),
체결 규칙(BRIEF2 '체결'·과제 문장).

F1-A (N=1, entry=confirm, tgt=2, stop=None, rev=False, filt=None)
- 범위 = SOXL 09:30 1분봉(O,H,L,C). C > O면 SOXL 방향, C < O면 SOXS 방향, 같으면 거래 없음.
- 확인: 09:31 봉부터 SOXL 종가가 H 위(SOXL 방향) 또는 L 아래(SOXS 방향)로 마감한 첫 봉이 결정 봉.
- 체결: 결정 봉 다음 봉의 산 종목 종가 × (1 + 반 호가[체결 봉 시간대] × 비용 배수).
- R = 결정 봉 SOXL 종가에서 범위 반대쪽 끝까지 거리(SOXL 기준 %). 목표 = 체결가 × (1 + 2R).
  체결 다음 봉부터 고가 ≥ 목표 × 1.0002이면 목표가에 매도(지정가, 호가 비용 없음). 손절 없음.
- 못 닿으면 마감 1분 전 봉 종가 × (1 − 반 호가[그 봉 시간대] × 비용 배수). 체결 봉이 마감 1분 전 봉이면 그 봉 종가에 바로 판다.
- 마감(actual) = 그날 24:00 KST의 ET 시각(서머타임 11:00, 표준시 10:00). 하루 한 번.
- 빈 분은 앞 종가로 채운다(O=H=L=C=앞 종가). 체결·청산 봉이 채운 봉이면 표시한다.

실행: py rescue_k6_blind.py [--calib]
  --calib: 2025-01~2026-10 자료(cache/alp1m, cost_table.json)로 results/confirm_daily_F1-A_actual.csv와 날별 대조(정의 해석 점검).
"""
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ET = ZoneInfo("America/New_York")
KST = timezone(timedelta(hours=9))
SEED = 20261104
NBOOT = 2000
TGT_R = 2.0
PIERCE = 1.0002


def cutoff_minute(day):
    """그날 24:00 KST(= 다음날 00:00 KST)의 ET 시각을 09:30부터 분 단위로."""
    d = pd.Timestamp(day).date()
    k = datetime(d.year, d.month, d.day, tzinfo=KST) + timedelta(days=1)
    e = k.astimezone(ET)
    return e.hour * 60 + e.minute  # 660(11:00) 또는 600(10:00)


def bucket(minute_of_day):
    if minute_of_day < 9 * 60 + 30:
        return "pre"
    if minute_of_day < 9 * 60 + 35:
        return "open5"
    if minute_of_day < 10 * 60:
        return "early"
    return "late"


def load_sym(cache_dir, sym):
    df = pd.read_parquet(Path(cache_dir) / f"{sym}.parquet", columns=["Open", "High", "Low", "Close"])
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx = idx.tz_convert(ET)
    df.index = idx
    df["date"] = idx.tz_localize(None).normalize()
    df["mod"] = idx.hour * 60 + idx.minute
    return df


def day_grids(df, last_mod=10 * 60 + 59):
    """날마다 09:30~last_mod 분 격자. 빈 분은 앞 종가(같은 날 프리장 포함, 없으면 전날 마지막 봉)로 채운다.
    반환: {date: (O,H,L,C, filled_mask)}, 정규장 봉이 있는 날 목록."""
    out = {}
    reg_days = set()
    prev_last_close = np.nan
    grid = np.arange(9 * 60 + 30, last_mod + 1)
    for date, g in df.groupby("date", sort=True):
        mods = g["mod"].to_numpy()
        reg = (mods >= 9 * 60 + 30) & (mods < 16 * 60)
        if reg.any():
            reg_days.add(date)
        sub = g[mods <= last_mod]
        o = np.full(len(grid), np.nan)
        h = o.copy(); l = o.copy(); c = o.copy()
        m_sub = sub["mod"].to_numpy()
        pos = m_sub - grid[0]
        ok = pos >= 0
        o[pos[ok]] = sub["Open"].to_numpy()[ok]
        h[pos[ok]] = sub["High"].to_numpy()[ok]
        l[pos[ok]] = sub["Low"].to_numpy()[ok]
        c[pos[ok]] = sub["Close"].to_numpy()[ok]
        filled = np.isnan(c)
        # 앞 종가: 09:30 전 마지막 봉(같은 날), 없으면 전날 마지막 봉
        pre = sub[m_sub < grid[0]]
        last = pre["Close"].iloc[-1] if len(pre) else prev_last_close
        for i in range(len(grid)):
            if filled[i]:
                o[i] = h[i] = l[i] = c[i] = last
            else:
                last = c[i]
        out[date] = (o, h, l, c, filled)
        prev_last_close = g["Close"].iloc[-1]
    return out, sorted(reg_days)


def run_day(day, gl, gs, costs, cost_mult=1.0, cutoff=None):
    """F1-A 하루. gl, gs = SOXL·SOXS 격자. 반환 dict."""
    cm = cutoff_minute(day) if cutoff is None else cutoff
    last = cm - 1 - (9 * 60 + 30)  # 마감 1분 전 봉 색인
    lo, lh, ll, lc, lf = gl
    o0, h0, l0, c0 = lo[0], lh[0], ll[0], lc[0]
    rec = dict(day=str(pd.Timestamp(day).date()), cutoff_et=f"{cm // 60}:{cm % 60:02d}", ret=0.0, n=0,
               why="", held="", dec="", fill_bar="", exit_bar="", fill_filled=False, exit_filled=False,
               soxl0930_filled=bool(lf[0]))
    if not (c0 > o0 or c0 < o0):
        rec["why"] = "doji"
        return rec
    up = c0 > o0
    dec = None
    for i in range(1, last):  # 결정 봉 i → 체결 봉 i+1 ≤ last
        if (up and lc[i] > h0) or ((not up) and lc[i] < l0):
            dec = i
            break
    if dec is None:
        rec["why"] = "no confirm"
        return rec
    sym = "SOXL" if up else "SOXS"
    g = gl if up else gs
    so, sh, sl, sc, sf = g
    f = dec + 1
    mod = lambda k: 9 * 60 + 30 + k
    ref = lc[dec]
    R = (ref - l0) / ref if up else (h0 - ref) / ref
    hs_in = costs[sym][bucket(mod(f))] * cost_mult
    entry = sc[f] * (1 + hs_in)
    tgt = entry * (1 + TGT_R * R)
    exit_px, why, xb = None, None, None
    for k in range(f + 1, last + 1):
        if sh[k] >= tgt * PIERCE:
            exit_px, why, xb = tgt, "target", k
            break
    if exit_px is None:
        xb = last
        exit_px = sc[last] * (1 - costs[sym][bucket(mod(last))] * cost_mult)
        why = "cutoff"
    hm = lambda k: f"{mod(k) // 60:02d}:{mod(k) % 60:02d}"
    rec.update(ret=(exit_px / entry - 1) * 100, n=1, why=why, held=sym, dec=hm(dec), fill_bar=hm(f),
               exit_bar=hm(xb), fill_filled=bool(sf[f]), exit_filled=bool(sf[xb]) if why == "cutoff" else False,
               R_pct=R * 100)
    return rec


def run_all(days, GL, GS, costs, cost_mult=1.0):
    rows = []
    for d in days:
        if d not in GL or d not in GS:
            rows.append(dict(day=str(pd.Timestamp(d).date()), cutoff_et="", ret=0.0, n=0, why="missing grid"))
            continue
        rows.append(run_day(d, GL[d], GS[d], costs, cost_mult))
    return pd.DataFrame(rows)


def boot_p(x, nboot=NBOOT, seed=SEED):
    x = np.asarray(x, float)
    if len(x) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(nboot, len(x)))
    m = x[idx].mean(axis=1)
    return float((m <= 0).mean())


def stats(x, x2=None, years=None):
    x = np.asarray(x, float)
    n = len(x)
    s = dict(n=int(n), mean=float(x.mean()) if n else float("nan"))
    if n == 0:
        return s
    srt = np.sort(x)
    s.update(median=float(np.median(x)), worst=float(srt[0]), best=float(srt[-1]),
             sd=float(x.std(ddof=1)) if n > 1 else float("nan"),
             drop_top2=float(srt[:-2].mean()) if n > 2 else float("nan"),
             p_one_sided=boot_p(x),
             active_days=int((x != 0).sum()),
             hit=float((x[x != 0] > 0).mean()) if (x != 0).any() else float("nan"))
    s["t"] = float(s["mean"] / (s["sd"] / np.sqrt(n))) if n > 1 and s["sd"] > 0 else float("nan")
    if x2 is not None:
        s["cost2x"] = float(np.mean(x2))
    if years is not None:
        by = {}
        for y in sorted(set(years)):
            m = np.asarray(years) == y
            by[str(y)] = dict(n=int(m.sum()), mean=float(x[m].mean()), active=int((x[m] != 0).sum()))
        s["by_year"] = by
        pos = sum(1 for v in by.values() if v["mean"] > 0)
        act = sum(1 for v in by.values() if v["active"] > 0)
        s["years_positive"] = f"{pos}/{act}"
    return s


def calib():
    t0 = time.time()
    costs = json.load(open(ROOT / "cost_table.json", encoding="utf-8"))
    L = load_sym(ROOT / "cache" / "alp1m", "SOXL")
    S = load_sym(ROOT / "cache" / "alp1m", "SOXS")
    GL, dl = day_grids(L)
    GS, ds = day_grids(S)
    ref = pd.read_csv(ROOT / "results" / "confirm_daily_F1-A_actual.csv", encoding="utf-8-sig")
    days = [pd.Timestamp(d) for d in ref["day"]]
    a = run_all(days, GL, GS, costs)
    b = run_all(days, GL, GS, costs, 2.0)
    m = ref.merge(a[["day", "ret", "n", "why", "held", "dec", "exit_bar", "cutoff_et"]], on="day", suffixes=("_ref", ""))
    m["ret2"] = b["ret"].to_numpy()
    m["d"] = m["ret"] - m["ret_pct"]
    m["d2"] = m["ret2"] - m["ret_cost2x_pct"]
    print("calib days", len(m), "ref mean", m.ret_pct.mean(), "mine", m.ret.mean(),
          "max|d|", m.d.abs().max(), "max|d2|", m.d2.abs().max(), "mismatch days", int((m.d.abs() > 1e-9).sum()),
          "n mismatch", int((m["n"] != m["n_ref"]).sum()) if "n_ref" in m else "", f"{time.time() - t0:.0f}s")
    cols = ["day", "cutoff_et_ref", "cutoff_et", "ret_pct", "ret", "n", "why", "held", "dec", "exit_bar"]
    print(m.loc[m.d.abs() > 1e-9, [c for c in cols if c in m]].to_string())
    print("cutoff label mismatch", int((m["cutoff_et_ref"] != m["cutoff_et"]).sum()),
          m["cutoff_et"].value_counts().to_dict())
    return m


def main():
    if "--calib" in sys.argv:
        calib()
        return
    t0 = time.time()
    costs = json.load(open(ROOT / "cost_table_2016.json", encoding="utf-8"))
    L = load_sym(ROOT / "cache" / "alp1m_2016", "SOXL")
    S = load_sym(ROOT / "cache" / "alp1m_2016", "SOXS")
    lo, hi = pd.Timestamp("2016-01-04"), pd.Timestamp("2022-12-30")
    GL, dl = day_grids(L)
    GS, ds = day_grids(S)
    days = [d for d in dl if lo <= d <= hi]
    only_l = sorted(set(d for d in dl if lo <= d <= hi) - set(d for d in ds if lo <= d <= hi))
    print("days", len(days), "SOXS 정규장 봉 없는 날", len(only_l), f"{time.time() - t0:.0f}s")
    a = run_all(days, GL, GS, costs, 1.0)
    b = run_all(days, GL, GS, costs, 2.0)
    a["ret_cost2x"] = b["ret"].to_numpy()
    a["year"] = [int(d[:4]) for d in a["day"]]
    a["dst"] = a["cutoff_et"] == "11:00"
    out_daily = ROOT / "results" / "rescue_k6_blind_daily.csv"
    a.to_csv(out_daily, index=False, encoding="utf-8-sig")

    def block(mask):
        sub = a[mask]
        return stats(sub["ret"].to_numpy(), sub["ret_cost2x"].to_numpy(), sub["year"].tolist())

    allm = np.ones(len(a), bool)
    res = dict(
        candidate="K6 = BRIEF2 F1-A (N=1, entry=confirm, tgt=2R, stop=None, rev=False, filt=None), actual 마감",
        impl="눈가림 구현 B(code/rescue_k6_blind.py), 원래 코드 import 없음",
        data="cache/alp1m_2016 SOXL·SOXS 1분봉, 2016-01-04~2022-12-30 정규장 날 전체",
        costs="cost_table_2016.json (반 호가, 비율)",
        boot=dict(n=NBOOT, seed=SEED, p="부트스트랩 평균 ≤ 0 비율"),
        all=block(allm),
        dst_11=block(a["dst"].to_numpy()),
        std_10=block(~a["dst"].to_numpy()),
    )
    tr = a[a["n"] > 0]
    ff = tr["fill_filled"].astype(bool) | tr["exit_filled"].astype(bool)
    res["trades"] = dict(
        n=int(len(tr)), SOXL=int((tr.held == "SOXL").sum()), SOXS=int((tr.held == "SOXS").sum()),
        target=int((tr.why == "target").sum()), cutoff=int((tr.why == "cutoff").sum()),
        why_counts={k: int(v) for k, v in a["why"].value_counts().items()},
        mean_per_trade=float(tr["ret"].mean()),
        SOXL_mean=float(tr.loc[tr.held == "SOXL", "ret"].mean()),
        SOXS_mean=float(tr.loc[tr.held == "SOXS", "ret"].mean()),
        zero_exposure=int((tr["fill_bar"] == tr["exit_bar"]).sum()),
        filled_fill=int(tr["fill_filled"].astype(bool).sum()),
        filled_exit=int(tr["exit_filled"].astype(bool).sum()),
        filled_any=int(ff.sum()),
        filled_any_by_year={str(k): int(v) for k, v in tr[ff].groupby("year").size().items()},
        soxl0930_filled_days=int(a["soxl0930_filled"].fillna(False).astype(bool).sum()),
    )
    # 채운 봉 매매를 뺀 감도: 그 날을 0으로(거래 안 함) / 그 날을 표본에서 제외
    a_zero = a.copy()
    fm = (a_zero["n"] > 0) & (a_zero["fill_filled"].astype(bool) | a_zero["exit_filled"].astype(bool))
    a_zero.loc[fm, ["ret", "ret_cost2x"]] = 0.0
    res["excl_filled_as_zero"] = stats(a_zero["ret"].to_numpy(), a_zero["ret_cost2x"].to_numpy(), a_zero["year"].tolist())
    a_drop = a[~fm]
    res["excl_filled_drop_days"] = stats(a_drop["ret"].to_numpy(), a_drop["ret_cost2x"].to_numpy(), a_drop["year"].tolist())
    res["judge"] = dict(
        mean_pos=res["all"]["mean"] > 0,
        p_le_0_05_raw=res["all"]["p_one_sided"] <= 0.05,
        drop_top2_pos=res["all"]["drop_top2"] > 0,
        cost2x_pos=res["all"]["cost2x"] > 0,
        half_years_pos=None,
        note="Holm m=6 판정은 전체 후보를 모아서 한다(여기는 원 p만).",
    )
    pos, act = map(int, res["all"]["years_positive"].split("/"))
    res["judge"]["half_years_pos"] = pos * 2 >= act
    res["elapsed_s"] = round(time.time() - t0, 1)
    out = ROOT / "results" / "rescue_k6_blind.json"
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: res[k] for k in ("all", "dst_11", "std_10", "trades", "judge")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
