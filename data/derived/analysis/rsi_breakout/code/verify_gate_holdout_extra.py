"""verify_gate_holdout.py 보충 (2026-10-10, 연구용). 읽기만, 쓰기는 results/gate/verify_holdout_extra.json 하나.
  a. 결정 종가가 SMA20에 얼마나 붙어 있나(주간 게이트가 동전 던지기에 가까운 정도)
  b. lag1·strict가 갈린 상위 3주의 실제 보유와 일별 수익
  c. SMCI 하나만 반도체 예산에 다시 넣은 민감도(지금 watchlist 기준)
  d. 낙폭 저점 날짜, qqq_w 결정 요일별 낙폭
  e. 2026 복제 시작일 맞춤(모든 모드 2026-01-05 첫 결정) — 첫 주 경로 의존 제거
실행: PYTHONIOENCODING=utf-8 py data/derived/analysis/rsi_breakout/code/verify_gate_holdout_extra.py
"""
import json
import os
import sys
from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import verify_gate_holdout as G  # noqa: E402
V = G.V
from desk.strategies import UNKNOWN, in_semis_budget  # noqa: E402

OUT = os.path.join(HERE, "../results/gate/verify_holdout_extra.json")


def main():
    D = V.Data(restore=True)
    sig = G.Sig(D)
    out = {}
    k0 = next(i for i, d in enumerate(D.days) if d >= V.START)
    k1 = max(i for i, d in enumerate(D.days) if d <= V.END)
    st, l1 = V.schedule(D, "strict", k0, k1), V.schedule(D, "lag1", k0, k1)

    # a. 임계 근접
    def gap(sym, k):
        idx, c, sma = sig.cache[sym, 20]
        i = G.bisect_right(idx, D.days[k - 1]) - 1
        return float(c[i] / sma[i] - 1)
    sig.below("QQQ", D.days[k0]); sig.below("SOXX", D.days[k0])
    a = {}
    for sym in ("QQQ", "SOXX"):
        for nm, s in (("strict", st), ("lag1", l1)):
            g = np.array([gap(sym, k) for k in s])
            a[f"{sym}·{nm}"] = {"|거리|<0.5%": float(np.mean(np.abs(g) < 0.005)), "|거리|<1%": float(np.mean(np.abs(g) < 0.01)),
                                "아래 비율": float(np.mean(g < 0))}
        # 같은 주 strict·lag1 신호가 다른 비율
        wk_s = {V.iso(D.days[k]): gap(sym, k) < 0 for k in st}
        wk_l = {V.iso(D.days[k]): gap(sym, k) < 0 for k in l1}
        com = [w for w in wk_s if w in wk_l]
        a[f"{sym}·신호 불일치 주"] = float(np.mean([wk_s[w] != wk_l[w] for w in com]))
    print("[a] 결정 종가의 SMA20 거리:", json.dumps(a, ensure_ascii=False))
    out["threshold"] = a

    # b. 상위 3주 보유와 일별 수익
    rule = V.desk_rule(semis=0.50)
    Pl = G.picks(D, rule, l1)
    Ps = G.picks(D, rule, st)
    b = {}
    for mon in ("2025-01-06", "2025-11-10", "2025-03-24"):
        kl = [k for k in Pl if V.iso(D.days[k]) == V.iso(mon)][0]
        ks = [k for k in Ps if V.iso(D.days[k]) == V.iso(mon)][0]
        tgt = Pl[kl][1]
        rows = []
        for kk in range(kl, kl + 5):
            r = 0.0
            for x, w in tgt.items():
                j = D.j[x]
                r += w * (D.O[kk + 1, j] / D.O[kk, j] - 1)
            rows.append((D.days[kk], round(r * 100, 2)))
        names = {D.tk[x]: round(w, 3) for x, w in sorted(tgt.items(), key=lambda t: -t[1])}
        wk_ret = {D.tk[x]: round(float(D.O[kl + 5, D.j[x]] / D.O[kl, D.j[x]] - 1) * 100, 1) for x in tgt}
        b[mon] = {"lag1 결정": D.days[kl], "strict 결정": D.days[ks], "lag1 보유": names, "종목별 5일 시가수익%": wk_ret,
                  "일별 포트 시가수익%(lag1 보유)": rows,
                  "QQQ 결정종가/SMA20 strict": sig.detail("QQQ", D.days[ks - 1]),
                  "QQQ 결정종가/SMA20 lag1": sig.detail("QQQ", D.days[kl - 1]),
                  "같은 종목?": sorted(Pl[kl][0]) == sorted(Ps[ks][0])}
        print(f"[b] {mon}: {b[mon]}")
    out["worst_weeks_detail"] = b

    # c. SMCI만 반도체 예산 안으로
    smci = [k for k, t in D.tk.items() if t == "SMCI"][0]
    print(f"[c] SMCI 지금 섹터 {D.sector[smci]} (예산 {in_semis_budget(D.sector[smci])}), 소분류 {D.sub[smci]}")
    base, _, _, _, _ = G.run_holdout(D, sig, ["live50", "c774", "sub5eq"], ["none", "soxx_w", "qqq_w"], ["lag1", "strict"])
    saved = D.sector[smci]
    D.sector[smci] = "AI 반도체"
    try:
        alt, _, _, _, _ = G.run_holdout(D, sig, ["live50", "c774", "sub5eq"], ["none", "soxx_w", "qqq_w"], ["lag1", "strict"])
    finally:
        D.sector[smci] = saved
    c = {}
    for k in base:
        c[k] = {"지금": (base[k]["전체"], base[k]["낙폭"]), "SMCI 예산 안": (alt[k]["전체"], alt[k]["낙폭"])}
        print(f"  {k:<22} 지금 {base[k]['전체'] * 100:+7.1f} ({base[k]['낙폭'] * 100:5.1f}) | SMCI 예산 안 "
              f"{alt[k]['전체'] * 100:+7.1f} ({alt[k]['낙폭'] * 100:5.1f}, 저점 {alt[k]['낙폭저점']})")
    out["smci_in_budget"] = c

    # d. 낙폭 저점, qqq_w 요일별 낙폭
    with open(os.path.join(HERE, "../results/gate/verify_holdout_timing.json"), encoding="utf-8") as f:
        mine = json.load(f)["all_mine"]
    d = {k: (round(x["낙폭"] * 100, 1), x["낙폭저점"]) for k, x in mine.items()
         if k.split("·")[0] in ("live50", "sub5eq") and k.split("·")[2] in ("lag1", "strict", "same")}
    print("[d] 낙폭(저점):", d)
    wdm = {f"{bb}·{g}": [round(mine[f"{bb}·{g}·wd{i}"]["낙폭"] * 100, 1) for i in range(5)] + [round(mine[f"{bb}·{g}·same"]["낙폭"] * 100, 1)]
           for bb in ("live50", "sub5eq") for g in ("none", "soxx_w", "qqq_w")}
    print("    요일 월~금 + same 낙폭:", wdm)
    out["mdd"] = d
    out["mdd_weekday"] = wdm

    # e. 2026 복제, 첫 결정 2026-01-05로 맞춤
    from bucket_main import BucketEngine, BParams
    from deskx import START
    E = BucketEngine()
    all_pdfs = list(E.book.pdfs)
    weekly, seen = [], set()
    for p in all_pdfs:
        if V.iso(p) not in seen:
            weekly.append(p)
            seen.add(V.iso(p))
    kstart = E.days.index("2026-01-05")
    k0e = E.days.index([x for x in E.days if x >= START][0])
    k1e = len(E.days) - 1
    o = pd.read_parquet(G.PARQ, columns=["date", "sym", "Close"])
    src = {}
    for s in ("SOXX", "QQQ"):
        x = o[o.sym == s]
        ser = pd.Series(x["Close"].astype(float).to_numpy(), index=pd.to_datetime(x["date"]).dt.strftime("%Y-%m-%d"))
        ser = ser[~ser.index.duplicated()].sort_index()
        src[s] = (ser, ser.rolling(20, min_periods=20).mean())

    def below(sym, day):
        s, m = src[sym]
        i = s.index.searchsorted(day, side="right") - 1
        return bool(i >= 0 and np.isfinite(m.iloc[i]) and s.iloc[i] < m.iloc[i])
    semis = lambda x: in_semis_budget(E.sector.get(x, UNKNOWN))
    bases = {"live50": BParams(gran="name", wmode="fund", n=7, semis_cap=0.50),
             "sub5eq": BParams(gran="sub", wmode="eq", n=5)}
    modes = {"desk": (all_pdfs, "strict"), "weekly_strict": (weekly, "strict"), "weekly_same": (weekly, "same"),
             "weekly_lag1": (weekly, "lag1")}
    e, paths = {}, {}
    for mname, (pdfs, mode) in modes.items():
        sched = {k: p for k, p in V.schedule(SimpleNamespace(days=E.days, pdfs=pdfs), mode, k0e, k1e).items() if k >= kstart}
        for bn, p in bases.items():
            sel, pk = set(), {}
            for k in sorted(sched):
                keep, tgt, ranked = E.target(p, k, sched[k] + "~", sel)
                if ranked:
                    sel = set(keep)
                    pk[k] = (sorted(keep), dict(tgt))
            for g in ("none", "soxx_w", "qqq_w"):
                units, cash, vals, lastp = {}, 1.0, [], {}
                for k in range(k0e, k1e + 1):
                    for x in units:
                        ox = E.col("opn", x)[k]
                        if np.isfinite(ox) and ox > 0:
                            lastp[x] = ox
                    value = cash + sum(units[x] * lastp[x] for x in units)
                    if k in pk:
                        on = g != "none" and below(G.GSYM[g], E.days[k - 1])
                        tgt = G.gated(pk[k][1], g, on, semis)
                        units, cash = {}, value
                        for x, wt in tgt.items():
                            ox = E.col("opn", x)[k]
                            lastp[x] = ox
                            units[x] = value * wt / ox
                            cash -= value * wt
                    vals.append(value)
                v = pd.Series(np.array(vals) / vals[0], index=E.days[k0e:k1e + 1])
                r = E.evaluate("x", v)
                paths[bn, g, mname] = (v, {V.iso(E.days[k]): tuple(pk[k][0]) for k in pk})
                e[f"{bn}·{g}·{mname}"] = {"앞": r["앞 1/2~5/29"]["수익"], "뒤": r["뒤 6/1~10/9"]["수익"],
                                          "전체": r["전체"]["수익"], "낙폭": r["전체"]["낙폭"]}
    dist = {}
    for bn in bases:
        for g in ("none", "soxx_w", "qqq_w"):
            ref, rpk = paths[bn, g, "desk"]
            rr = ref.pct_change().dropna()
            for mname in ("weekly_strict", "weekly_same", "weekly_lag1"):
                v, pk = paths[bn, g, mname]
                r = v.pct_change().dropna()
                com = [w for w in rpk if w in pk]
                dist[f"{bn}·{g}·{mname}"] = {"상관": float(np.corrcoef(rr, r)[0, 1]), "추적오차": float((r - rr).std() * np.sqrt(252)),
                                             "전체차%p": (float(v.iloc[-1]) - float(ref.iloc[-1])) * 100,
                                             "종목일치": float(np.mean([rpk[w] == pk[w] for w in com]))}
    for k, x in e.items():
        print(f"[e] {k:<26} 앞 {x['앞'] * 100:+7.1f} 뒤 {x['뒤'] * 100:+6.1f} 전체 {x['전체'] * 100:+7.1f} 낙폭 {x['낙폭'] * 100:6.1f}")
    for k, x in dist.items():
        print(f"    거리 {k:<26} 상관 {x['상관']:.3f} 추적오차 {x['추적오차']:.3f} 전체차 {x['전체차%p']:+6.1f}%p 종목일치 {x['종목일치']:.2f}")
    out["replica_aligned"] = {"runs": e, "dist": dist}

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=str)
    print("저장:", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
