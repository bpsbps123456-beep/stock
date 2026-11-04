"""K4·K5 눈가림 재구현 B (BRIEF7, 2026-11-04).

원래 연구 코드(lib.py·ev4_build.py 등)와 rescue_k45.py를 import하지 않는다.
BRIEF3·BRIEF4·BRIEF7의 정의, cache/alp1m_2016/*.parquet, cost_table_2016.json만 쓴다.

K5 = 사용자 방향 x = 3% 단독 사슬: SOXL 1분 종가 >= L*1.03(L = 09:30 봉 또는 앞 사건 청산
     다음 봉부터 지금 봉까지 SOXL 저가 최저), 결정 봉 마감 <= 10:45(봉 시작 <= 10:44),
     다음 봉 종가 x (1 + 반 호가)에 SOXS 매수, +1% 지정가(체결 다음 봉부터, 고가 >= 목표 x 1.0002),
     못 닿으면 10:59 봉 종가 x (1 - 반 호가). 손절 없음. 들고 있는 동안 신호 무시.
K4 = 그 사슬 사건 가운데 신호 봉 SOXL 윗꼬리 / (고가 - 저가) >= 0.5 - 1e-9.
빈 분은 앞 실제 봉 종가로 채운다(O=H=L=C). 채운 봉 체결·청산 매매를 센다.
검산: 같은 코드를 alp1m_long(2023-01-03~2026-10-28)·cost_table.json에 돌려 report4 수치와 대조.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
X = 3.0
WICK_MIN = 0.5 - 1e-9
DEC_LAST = 74          # 결정 봉 시작 인덱스 상한(09:30 + 74분 = 10:44, 마감 10:45)
EXIT_IDX = 89          # 10:59 봉
NBAR = 90              # 09:30~10:59
TGT = 0.01
PIERCE = 1.0002
SEED = 20261104
NBOOT = 10000


def tier(minute):
    """minute = 자정부터 분(봉 시작). 반 호가 시간대."""
    if minute < 570:
        return "pre"
    if minute < 575:
        return "open5"
    if minute < 600:
        return "early"
    return "late"


def load_grid(folder, sym, start=None, end=None):
    df = pd.read_parquet(ROOT / "cache" / folder / f"{sym}.parquet",
                         columns=["Open", "High", "Low", "Close"])
    idx = df.index
    mins = idx.hour * 60 + idx.minute
    reg = idx[(mins >= 570) & (mins < 960)]
    days = pd.DatetimeIndex(sorted(set(reg.tz_localize(None).normalize())))
    if start:
        days = days[days >= pd.Timestamp(start)]
    if end:
        days = days[days <= pd.Timestamp(end)]
    offs = pd.to_timedelta(np.arange(570, 570 + NBAR), unit="min")
    need = (days.values[:, None] + offs.values[None, :]).ravel()
    need = pd.DatetimeIndex(need).tz_localize("America/New_York")
    full = df.reindex(df.index.union(need))
    filled = full["Close"].isna()
    cff = full["Close"].ffill()
    for c in ["Open", "High", "Low", "Close"]:
        full[c] = full[c].where(~filled, cff)
    sel = full.loc[need]
    fl = filled.loc[need].values.reshape(len(days), NBAR)
    arr = {c: sel[c].values.reshape(len(days), NBAR) for c in ["Open", "High", "Low", "Close"]}
    arr["filled"] = fl
    return days, arr


def trade(sx, i, f, hs, mult):
    """SOXS 매수: f = 체결 봉 인덱스. 반환 (손익 %, 청산 봉, 도달 여부)."""
    entry = sx["Close"][i, f] * (1 + mult * hs[tier(570 + f)])
    tgt = entry * (1 + TGT)
    hi = sx["High"][i, f + 1:EXIT_IDX + 1]
    hit = np.nonzero(hi >= tgt * PIERCE)[0]
    if len(hit):
        return TGT * 100, f + 1 + int(hit[0]), True
    ex = sx["Close"][i, EXIT_IDX] * (1 - mult * hs[tier(570 + EXIT_IDX)])
    return (ex / entry - 1) * 100, EXIT_IDX, False


def build_events(days, sl, sx, costs):
    hs = costs["SOXS"]
    rows = []
    for i, d in enumerate(days):
        t = 0
        L = np.inf
        while t <= DEC_LAST:
            L = min(L, sl["Low"][i, t])
            if sl["Close"][i, t] >= L * (1 + X / 100):
                f = t + 1
                r1, e, hit = trade(sx, i, f, hs, 1.0)
                r2, _, hit2 = trade(sx, i, f, hs, 2.0)
                h, lo = sl["High"][i, t], sl["Low"][i, t]
                o, c = sl["Open"][i, t], sl["Close"][i, t]
                rng = h - lo
                wick = (h - max(o, c)) / rng if rng > 0 else np.nan
                rows.append(dict(
                    day=d, dec=t, fill=f, exit=e, r=r1, r_c2=r2, hit=hit, hit_c2=hit2,
                    wick=wick,
                    sig_filled=bool(sl["filled"][i, t]),
                    fill_filled=bool(sx["filled"][i, f]),
                    exit_filled=bool(sx["filled"][i, e]),
                    any_hold_filled=bool(sx["filled"][i, f:e + 1].any()),
                ))
                t = e + 1
                L = np.inf
                continue
            t += 1
    ev = pd.DataFrame(rows)
    ev["year"] = pd.DatetimeIndex(ev["day"]).year
    return ev


def day_series(ev, days, col="r"):
    s = pd.Series(0.0, index=days)
    if len(ev):
        g = ev.groupby("day")[col].apply(lambda v: (np.prod(1 + v.values / 100) - 1) * 100)
        s.loc[g.index] = g.values
    return s


def boot_p(s, rng, n=NBOOT):
    v = s.values
    idx = rng.integers(0, len(v), size=(n, len(v)))
    m = v[idx].mean(axis=1)
    return float((m <= 0).mean()), [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def drop_top2(s):
    v = np.sort(s.values)[::-1]
    return float(v[2:].mean())


def summarize(ev, days, rng):
    s = day_series(ev, days, "r")
    s2 = day_series(ev, days, "r_c2")
    yrs = {}
    for y in sorted(set(days.year)):
        dm = days[days.year == y]
        e = ev[ev["year"] == y]
        yrs[str(y)] = dict(
            n=int(len(e)), days=int(len(dm)), active_days=int(e["day"].nunique()),
            day_mean=float(s.loc[dm].mean()),
            trade_mean=float(e["r"].mean()) if len(e) else None,
            hit=float(e["hit"].mean()) if len(e) else None,
            day_mean_c2=float(s2.loc[dm].mean()),
        )
    with_ev = [k for k, v in yrs.items() if v["n"] > 0]
    pos = sum(1 for k in with_ev if yrs[k]["day_mean"] > 0)
    p, ci = boot_p(s, rng)
    return dict(
        n=int(len(ev)), active_days=int(ev["day"].nunique()) if len(ev) else 0,
        n_days=int(len(days)),
        mean=float(s.mean()), trade_mean=float(ev["r"].mean()) if len(ev) else None,
        median=float(ev["r"].median()) if len(ev) else None,
        hit=float(ev["hit"].mean()) if len(ev) else None,
        worst=float(ev["r"].min()) if len(ev) else None,
        worst_day=float(s.min()),
        drop_top2=drop_top2(s), cost2x=float(s2.mean()),
        boot_p_le0=p, boot_ci95=ci,
        years_positive=f"{pos}/{len(with_ev)}", by_year=yrs,
    ), s


def k4_control_p(ev_all, ev_k4, rng, n=NBOOT):
    """같은 해 조건 없는 사슬 사건에서 같은 수를 비복원으로 뽑은 매매당 평균 >= 관측 비율."""
    obs = ev_k4["r"].mean()
    tot = np.zeros(n)
    ntot = 0
    for y, e4 in ev_k4.groupby("year"):
        pool = ev_all.loc[ev_all["year"] == y, "r"].values
        k = len(e4)
        keys = rng.random((n, len(pool)))
        pick = np.argpartition(keys, k - 1, axis=1)[:, :k]
        tot += pool[pick].sum(axis=1)
        ntot += k
    m = tot / ntot
    return float((m >= obs - 1e-12).mean()), float(obs), float(m.mean())


def filled_report(ev, days, rng):
    bad = ev["fill_filled"] | ev["exit_filled"]
    clean = ev[~bad]
    s = day_series(clean, days, "r")
    p, _ = boot_p(s, rng, 2000)
    return dict(
        n_fill_filled=int(ev["fill_filled"].sum()),
        n_exit_filled=int(ev["exit_filled"].sum()),
        n_fill_or_exit_filled=int(bad.sum()),
        n_sig_filled=int(ev["sig_filled"].sum()),
        n_any_hold_filled=int(ev["any_hold_filled"].sum()),
        by_year_fill_or_exit=ev[bad].groupby("year").size().astype(int).to_dict(),
        excl_n=int(len(clean)), excl_mean=float(s.mean()),
        excl_trade_mean=float(clean["r"].mean()) if len(clean) else None,
        excl_drop_top2=drop_top2(s),
        excl_cost2x=float(day_series(clean, days, "r_c2").mean()),
        excl_boot_p_le0=p,
    )


def run(folder, cost_file, start=None, end=None):
    costs = json.load(open(ROOT / cost_file, encoding="utf-8"))
    days, sl = load_grid(folder, "SOXL", start, end)
    days2, sx = load_grid(folder, "SOXS", start, end)
    assert days.equals(days2), "SOXL·SOXS 거래일 불일치"
    ev = build_events(days, sl, sx, costs)
    k4 = ev[ev["wick"] >= WICK_MIN]
    return days, ev, k4


def main():
    out = {"brief": "BRIEF7 K4·K5 눈가림 재구현 B", "x": X, "wick_min": WICK_MIN,
           "decision_last_bar_start": "10:44", "seed": SEED, "nboot": NBOOT}
    # 새 자료 2016~2022
    days, ev, k4 = run("alp1m_2016", "cost_table_2016.json")
    rng = np.random.default_rng(SEED)
    k5s, _ = summarize(ev, days, rng)
    k5s["p_one_sided"] = k5s["boot_p_le0"]
    k5s["filled"] = filled_report(ev, days, np.random.default_rng(SEED))
    rng = np.random.default_rng(SEED)
    k4s, _ = summarize(k4, days, np.random.default_rng(SEED))
    p4, obs4, null4 = k4_control_p(ev, k4, rng)
    k4s["p_one_sided"] = p4
    k4s["control_obs_trade_mean"] = obs4
    k4s["control_null_trade_mean"] = null4
    k4s["filled"] = filled_report(k4, days, np.random.default_rng(SEED))
    bad = k4["fill_filled"] | k4["exit_filled"]
    if (~bad).sum():
        pc, _, _ = k4_control_p(ev[~(ev["fill_filled"] | ev["exit_filled"])], k4[~bad],
                                np.random.default_rng(SEED), 2000)
        k4s["filled"]["excl_control_p"] = pc
    k4s["events"] = [dict(day=str(r.day.date()), dec=f"{9 + (30 + r.dec) // 60:02d}:{(30 + r.dec) % 60:02d}",
                          r=round(r.r, 4), r_c2=round(r.r_c2, 4), wick=round(r.wick, 4),
                          fill_filled=r.fill_filled, exit_filled=r.exit_filled)
                     for r in k4.itertuples()]
    out["K5"] = k5s
    out["K4"] = k4s
    ev.assign(day=ev["day"].dt.strftime("%Y-%m-%d")).to_csv(
        ROOT / "results" / "rescue_k45_blind_events.csv", index=False)

    # 검산: 2023~2026 기존 자료(report4 기준 사용자 x3 131/161/166/258건, H4 2026 9건 +0.875)
    if "--no-check" not in sys.argv:
        d2, e2, h2 = run("alp1m_long", "cost_table.json")
        chk = {}
        for y in sorted(set(d2.year)):
            e = e2[e2["year"] == y]
            h = h2[h2["year"] == y]
            s = day_series(e, d2[d2.year == y])
            chk[str(y)] = dict(n=int(len(e)), trade_mean=float(e["r"].mean()),
                               day_mean=float(s.mean()), days=int(len(s)),
                               h4_n=int(len(h)), h4_trade_mean=float(h["r"].mean()) if len(h) else None,
                               h4_trade_mean_c2=float(h["r_c2"].mean()) if len(h) else None)
        chk["h4_2026_days"] = [str(x.date()) for x in h2[h2["year"] == 2026]["day"]]
        out["check_2023_2026"] = chk

    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "rescue_k45_blind.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
    show = {k: {kk: vv for kk, vv in out[k].items() if kk not in ("events", "by_year")}
            for k in ("K5", "K4")}
    print(json.dumps(show, ensure_ascii=False, indent=1, default=str))
    for k in ("K5", "K4"):
        print(k, json.dumps(out[k]["by_year"], ensure_ascii=False))
    if "check_2023_2026" in out:
        print(json.dumps(out["check_2023_2026"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
