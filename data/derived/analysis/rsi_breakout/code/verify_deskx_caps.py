"""caps 패밀리 감사(2026-10-10, 연구용 스크래치). deskx를 쓰지 않는 독립 재계산으로 대조한다.

1) Params() == desk.strategies.run_model (최대 차)
2) 원자료(동결 가격 CSV + TIME PDF)로 순위·버퍼·상한·시가→시가 경로를 따로 짜서
   semis_cap 0.50과 기본의 가치 경로·재조정 비중을 엔진과 대조, 두 재조정 결정을 표로 출력
   상한 투영은 엔진과 다른 방식(작은 보폭으로 키우는 무차별 시뮬레이션)으로도 한 번 더 확인
3) 견고성: 월별 차, 하루 단위 차의 쏠림, 종목별 기여 차, 같은 투자비중의 현금 섞은 기본과 비교,
   이웃 값(semis_cap 0.45·0.55, 감사용 진단이지 후보 아님), 회전율
4) MU가 TIME PDF에 몇 번 나오는지
실행: py verify_deskx_caps.py (code 폴더, PYTHONIOENCODING=utf-8)
"""
import os, sys
from datetime import date

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from deskx import Engine, Params, ROOT  # noqa: E402  (엔진은 대조 상대로만 쓴다)

sys.path.insert(0, ROOT)
from desk import config  # noqa: E402
from desk.strategies import run_model, us_weights, sector_of  # noqa: E402
from gbrc.etf.trades import load_fund_snapshots  # noqa: E402
from gbrc.etf.market import yahoo_ticker  # noqa: E402

FROZEN = os.path.join(HERE, "../cache/deskx_prices")
START = "2026-01-02"
SEMIS = config.SEMIS_SECTORS
UNK = "미분류"


# ---------------------------------------------------------------- 독립 구현
def load_raw():
    px = {f: pd.read_csv(os.path.join(FROZEN, f"prices_{f}.csv"), index_col=0, parse_dates=True)
          for f in ("open", "close", "adj_close")}
    days = [d.strftime("%Y-%m-%d") for d in px["adj_close"]["QQQ"].dropna().index]
    for f in px:
        px[f].index = px[f].index.strftime("%Y-%m-%d")
        px[f] = px[f].reindex(days)
    snaps = load_fund_snapshots("456600", "9999-12-31")
    tw = {d: us_weights(s) for d, s in snaps.items()}
    keys = {k for w in tw.values() for k in w}
    tick = {k: yahoo_ticker(k) for k in keys}
    sect = {k: sector_of(k, tick[k]) for k in keys}
    return px, days, tw, sorted(tw), tick, sect


def caps_exact(raw, sect, stock, sector, semis):
    """비례 성장: 이름마다 raw에 비례해 키우다 자기·섹터·반도체 예산이 차면 멈춘다. 이벤트 단위 정확 계산."""
    a = {k: 0.0 for k in raw}
    frozen = set()
    budget = {k for k in raw if sect[k] in SEMIS or sect[k] == UNK}
    for _ in range(100):
        act = [k for k in raw if k not in frozen]
        if not act or sum(a.values()) >= 1 - 1e-12:
            break
        gsum = lambda g: sum(a[k] for k in raw if sect[k] == g)
        s_in = sum(a[k] for k in budget)
        cand = [(1 - sum(a.values())) / sum(raw[k] for k in act)]
        cand += [(stock - a[k]) / raw[k] for k in act]
        for g in {sect[k] for k in act}:
            cand.append((sector - gsum(g)) / sum(raw[k] for k in act if sect[k] == g))
        r_s = sum(raw[k] for k in act if k in budget)
        if r_s > 0:
            cand.append((semis - s_in) / r_s)
        dt = max(0.0, min(cand))
        for k in act:
            a[k] += dt * raw[k]
        s_in = sum(a[k] for k in budget)
        for k in act:
            if a[k] >= stock - 1e-12 or gsum(sect[k]) >= sector - 1e-12 or (k in budget and s_in >= semis - 1e-12):
                frozen.add(k)
    return {k: v for k, v in a.items() if v > 1e-12}


def caps_brute(raw, sect, stock, sector, semis, dt=2e-6):
    """무차별 확인: 아주 작은 보폭으로 모두 키우고, 제약을 넘기기 직전에 멈춘 이름은 얼린다."""
    ks = list(raw)
    r = np.array([raw[k] for k in ks])
    a = np.zeros(len(ks))
    act = np.ones(len(ks), bool)
    g = np.array([sect[k] for k in ks])
    bud = np.array([sect[k] in SEMIS or sect[k] == UNK for k in ks])
    while act.any() and a.sum() < 1 - 1e-9:
        nxt = a + dt * r * act
        bad = nxt > stock + 1e-12
        for grp in set(g):
            m = g == grp
            if nxt[m].sum() > sector + 1e-12:
                bad |= m
        if nxt[bud].sum() > semis + 1e-12:
            bad |= bud
        if nxt.sum() > 1 + 1e-12:
            break
        act &= ~bad
        a = a + dt * r * act
    return {k: float(v) for k, v in zip(ks, a) if v > 1e-12}


def independent_run(raw_data, semis_cap=0.60, stock_cap=0.25, sector_cap=0.60, n=7, buf=5, lookback=120,
                    log_days=()):
    px, days, tw, pdfs, tick, sect = raw_data
    O = {k: px["open"][tick[k]].to_numpy(float) for k in tick if tick[k] in px["open"].columns}
    A = {k: px["adj_close"][tick[k]].to_numpy(float) for k in tick if tick[k] in px["adj_close"].columns}
    k0 = days.index(next(d for d in days if d >= START))
    V = [1.0]
    w, sel, started = {}, set(), False
    hold, logs, turnover = [], {}, 0.0
    for k in range(k0, len(days)):
        day = days[k]
        if k > k0:
            ret = {}
            for key in w:
                o = O.get(key)
                ok = o is not None and np.isfinite(o[k]) and np.isfinite(o[k - 1]) and o[k - 1] > 0
                ret[key] = o[k] / o[k - 1] - 1 if ok else 0.0
            g = sum(w[x] * ret[x] for x in w)
            V.append(V[-1] * (1 + g))
            w = {x: w[x] * (1 + ret[x]) / (1 + g) for x in w}
        new_week = k == 0 or date.fromisoformat(day).isocalendar()[:2] != date.fromisoformat(days[k - 1]).isocalendar()[:2]
        if not started or new_week:
            prior = [d for d in pdfs if d < day]          # PDF 날짜가 세션보다 엄격히 앞선 것만
            if prior:
                pw = tw[prior[-1]]
                rk = []
                for key in sorted(pw):
                    if pw[key] <= 0 or key not in A or k - 1 - lookback < 0:
                        continue
                    p1, p0 = A[key][k - 1], A[key][k - 1 - lookback]   # k-1일 종가까지만
                    if np.isfinite(p1) and np.isfinite(p0) and p0 > 0:
                        rk.append((key, p1 / p0 - 1))
                rk.sort(key=lambda t: -t[1])
                if rk:
                    started = True
                    order = [x for x, _ in rk]
                    keep = [x for x in order[:n + buf] if x in sel][:n]
                    keep += [x for x in order if x not in keep][:n - len(keep)]
                    tot = sum(pw[x] for x in keep)
                    raw = {x: pw[x] / tot for x in keep}
                    tgt = caps_exact(raw, sect, stock_cap, sector_cap, semis_cap)
                    turnover += 0.5 * sum(abs(tgt.get(x, 0) - w.get(x, 0)) for x in set(tgt) | set(w))
                    if day in log_days:
                        logs[day] = dict(pdf=prior[-1], ranked=rk[:n + buf], keep=keep, raw=raw, tgt=tgt)
                    w, sel = tgt, set(keep)
        hold.append(dict(w))
    return pd.Series(V, index=days[k0:]), hold, logs, turnover


def periods(V):
    d = list(V.index)

    def at(s):
        return d.index(next(x for x in d if x >= s))

    out = {}
    for lab, a, b in (("앞", "2026-01-02", "2026-06-01"), ("뒤", "2026-06-01", "2026-10-09"),
                      ("9/1~", "2026-09-01", "2026-10-09"), ("전체", "2026-01-02", "2026-10-09")):
        i, j = at(a), d.index(b)
        seg = V.iloc[i:j + 1]
        out[lab] = (float(seg.iloc[-1] / seg.iloc[0] - 1), float((seg / seg.cummax() - 1).min()))
    return out


# ---------------------------------------------------------------- 실행
def main():
    E = Engine()
    print("== 1) 기본 Params() vs desk run_model")
    vb, rb = E.run(Params())
    ref = run_model(E.book, start=START)
    print("  최대 차:", float(np.max(np.abs(vb.to_numpy() - np.asarray(ref.value)))))
    v5, r5 = E.run(Params(semis_cap=0.50))

    print("\n== 2) 독립 재계산 (원자료)")
    raw = load_raw()
    days = raw[1]
    chk_days = ("2026-06-01", "2026-07-06", "2026-07-13")
    for name, sc, ve, re_ in (("기본", 0.60, vb, rb), ("semis0.50", 0.50, v5, r5)):
        Vi, hi, logs, to = independent_run(raw, semis_cap=sc, log_days=chk_days)
        dv = float(np.max(np.abs(Vi.to_numpy() - ve.to_numpy())))
        dw = max(abs(hi[i].get(x, 0) - re_[i]["w"].get(x, 0)) for i in range(len(hi))
                 for x in set(hi[i]) | set(re_[i]["w"]))
        print(f"  {name}: 가치 경로 최대 차 {dv:.2e}, 보유 비중 최대 차 {dw:.2e}, 회전율 합 {to:.2f}")
        for lab, (ret, mdd) in periods(Vi).items():
            print(f"    {lab}: 수익 {ret:+.4f}  낙폭 {mdd:+.4f}")
        if name == "semis0.50":
            sect, tick = raw[5], raw[4]
            for d, L in logs.items():
                print(f"\n  [결정 {d}] PDF {L['pdf']} (세션보다 앞섬: {L['pdf'] < d}), 종가 기준일 {days[days.index(d) - 1]}")
                print("   순위 상위 12:", ", ".join(f"{tick[x]}({m:+.2f})" for x, m in L["ranked"]))
                print("   선택:", [tick[x] for x in L["keep"]])
                brute = caps_brute(L["raw"], sect, 0.25, 0.60, 0.50)
                eng = r5[days.index(d) - days.index(START)]["w"]
                for x in L["keep"]:
                    print(f"    {tick[x]:<6} {sect[x]:<10} raw {L['raw'][x]:.4f}  독립 {L['tgt'].get(x, 0):.4f}"
                          f"  무차별 {brute.get(x, 0):.4f}  엔진 {eng.get(x, 0):.4f}")
                s_bud = sum(v for x, v in L["tgt"].items() if sect[x] in SEMIS or sect[x] == UNK)
                print(f"   반도체 예산 {s_bud:.4f} (상한 0.50), 투자 {sum(L['tgt'].values()):.4f}, 현금 {1 - sum(L['tgt'].values()):.4f}")

    print("\n== 3) 견고성")
    rb_d = (vb.shift(-1) / vb - 1).dropna()
    r5_d = (v5.shift(-1) / v5 - 1).dropna()
    for lab, a, b in (("앞", "2026-01-02", "2026-05-29"), ("뒤", "2026-06-01", "2026-10-09")):
        m = (rb_d.index >= a) & (rb_d.index <= b)
        diff = np.log1p(r5_d[m]) - np.log1p(rb_d[m])
        tot = diff.sum()
        top = diff.reindex(diff.abs().sort_values(ascending=False).index)[:5]
        print(f"  {lab}: 로그 초과 합 {tot:+.4f}, 다른 날 {int((diff.abs() > 1e-9).sum())}/{int(m.sum())}일, "
              f"절댓값 상위 5일 합 {top.sum():+.4f} -> {', '.join(f'{i}:{x:+.4f}' for i, x in top.items())}")
        print(f"     상위 5일 빼면 {tot - top.sum():+.4f}")
    # 월별 차
    def monthly(r):
        return {m: float((1 + r[r.index.str.startswith(m)]).prod() - 1) for m in [f"2026-{i:02d}" for i in range(1, 11)]}
    mb, m5 = monthly(rb_d), monthly(r5_d)
    print("  월별 차(0.50-기본):", {m[-2:]: round(m5[m] - mb[m], 3) for m in mb},
          " 이긴 달", sum(m5[m] > mb[m] + 1e-9 for m in mb), "/10")
    # 같은 투자비중 현금 섞기 (기본 수익 x f, 매일)
    inv_b = np.mean([sum(r["w"].values()) for r in rb])
    inv_5 = np.mean([sum(r["w"].values()) for r in r5])
    f = inv_5 / inv_b
    vbl = pd.Series(np.concatenate([[1.0], np.cumprod(1 + f * rb_d.to_numpy())]), index=vb.index)
    print(f"  투자비중 기본 {inv_b:.3f}, 0.50 {inv_5:.3f} -> 기본 x {f:.3f}(나머지 현금) 대조:")
    for nm, V in (("기본", vb), ("기본x현금", vbl), ("semis0.50", v5)):
        p = periods(V)
        print(f"    {nm:<10} " + "  ".join(f"{k} {x[0]:+.4f}/{x[1]:+.4f}" for k, x in p.items()))
    # 7월 종목별 기여 (기본 vs 0.50)
    k0 = E.days.index(START)

    def contrib(v, rows, a, b):
        out = {}
        vals = v.to_numpy()
        i0 = next(i for i, r in enumerate(rows) if r["day"] >= a)
        for i in range(i0, len(rows) - 1):
            if rows[i]["day"] > b:
                break
            k = k0 + i + 1
            for key, wt in rows[i]["w"].items():
                o = E.col("opn", key)
                ret = o[k] / o[k - 1] - 1 if o is not None and np.isfinite(o[k]) and np.isfinite(o[k - 1]) else 0.0
                out[E.ticker[key]] = out.get(E.ticker[key], 0.0) + vals[i] / vals[i0] * wt * ret
        return out
    for lab, a, b in (("7월", "2026-07-01", "2026-07-31"), ("앞", "2026-01-02", "2026-05-29"), ("뒤", "2026-06-01", "2026-10-08")):
        cb, c5 = contrib(vb, rb, a, b), contrib(v5, r5, a, b)
        names = set(cb) | set(c5)
        d = sorted(((n_, c5.get(n_, 0) - cb.get(n_, 0)) for n_ in names), key=lambda t: -abs(t[1]))[:5]
        print(f"  {lab} 종목 기여 차(0.50-기본, 구간 시작 가치=1): " + ", ".join(f"{n_} {x:+.3f}" for n_, x in d))
        if lab == "7월":
            print("    기본 7월 기여:", ", ".join(f"{n_} {x:+.3f}" for n_, x in sorted(cb.items(), key=lambda t: t[1])[:7]))
    # 이웃 값 (감사용 진단, 후보 아님)
    print("  이웃 값(감사 진단):")
    for sc in (0.45, 0.50, 0.55, 0.60):
        v, rows = E.run(Params(semis_cap=sc))
        p = periods(v)
        print(f"    semis_cap {sc:.2f}: " + "  ".join(f"{k} {x[0]:+.4f}/{x[1]:+.4f}" for k, x in p.items()))

    print("\n== 4) MU")
    snaps = load_fund_snapshots("456600", "9999-12-31")
    mu = [d for d, s in snaps.items() if any(yahoo_ticker(k) == "MU" for k in us_weights(s))]
    print(f"  TIME PDF {len(snaps)}개 중 MU 보유 {len(mu)}개: {mu[:3]}...{mu[-3:]}")
    ks = load_fund_snapshots("2ETFQ1", "9999-12-31")
    kmu = [d for d, s in ks.items() if any(yahoo_ticker(k) == "MU" for k in us_weights(s))]
    print(f"  KoAct PDF {len(ks)}개 중 MU 보유 {len(kmu)}개")
    mu_key = next(k for s in ks.values() for k in us_weights(s) if yahoo_ticker(k) == "MU")
    print(f"  MU 섹터(watchlist): {sector_of(mu_key, 'MU')}")
    # 만약 MU가 TIME 후보에 있었다면 120일 수익 순위 몇 위였나 (진단만)
    mu_adj = E.book.adj["MU"].to_numpy(float) if "MU" in E.book.adj.columns else None
    if mu_adj is not None:
        for d in ("2026-06-01", "2026-07-06", "2026-09-08", "2026-10-05"):
            k = E.days.index(d)
            rk = E.ranking(Params(), k, d)
            m = mu_adj[k - 1] / mu_adj[k - 1 - 120] - 1
            pos = 1 + sum(r["mom"] > m for r in rk)
            print(f"    {d}: MU 120일 {m:+.2f} -> TIME 후보 {len(rk)}개 중 {pos}위 가정 (7위 경계 {rk[6]['mom']:+.2f})")


if __name__ == "__main__":
    main()
