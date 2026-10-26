"""universe 계열 감사(2026-10-10). deskx_universe.py 결과를 원자료로 다시 계산해 반박을 시도한다.

1) 자료 점검: KoAct PDF 날짜·공백, 두 펀드 키가 같은 Yahoo 티커로 겹치는지, 가격이 없는 KoAct 종목,
   TIME·KoAct의 MU·PLTR 보유 이력.
2) 독립 재계산: 얼린 가격 CSV + 두 펀드 PDF만으로 union·avg 경로를 deskx 없이 처음부터 다시 돌린다
   (순위·완충·상한(별도 구현한 비례 채우기)·시가 표류). 엔진 경로와 비교하고 구간 수익을 다시 잰다.
3) 특정 결정: 5/26(DELL 순위 TIME 안 vs 합집합), 6/1(MU #2), 2/17(MU 첫 선택), 4~5월 MU 순위.
4) 강건성(감사용, 고르기 아님): 월별 차, 9/1~ 개선의 종목·날짜 쏠림, 이웃 값(lookback·n), 사후 문턱 1~5%,
   'TIME + MU만' 민감도, 회전율과 스프레드 민감도.
"""
import os, sys
from bisect import bisect_left
from collections import defaultdict
from datetime import date
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, *[".."] * 5))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from gbrc.etf.trades import load_fund_snapshots                     # noqa: E402
from gbrc.etf.market import company_key, is_cash_holding, listing_market, yahoo_ticker   # noqa: E402
from gbrc.etf.research_data import CORPORATE_ALIASES, exclusion_reason                   # noqa: E402
from watchlist import resolve_sector                                                       # noqa: E402

FROZEN = os.path.join(HERE, "..", "cache", "deskx_prices")
START = "2026-01-02"
SEMIS = {"AI 반도체", "메모리", "반도체 장비", "반도체 장비(적자)", "반도체(적자)", "반도체 소재"}
UNK = "미분류"
PER = (("앞", "2026-01-02", "2026-05-29"), ("뒤", "2026-06-01", "2026-10-09"),
       ("9/1~", "2026-09-01", "2026-10-09"), ("전체", "2026-01-02", "2026-10-09"))


# ------------------------------------------------------------------ 원자료
def us_w(snap):
    out = {}
    for row in snap["holdings"]:
        if is_cash_holding(row) or exclusion_reason(row):
            continue
        key = company_key(row.get("code"))
        key = CORPORATE_ALIASES.get(key, key)
        if listing_market(key) == "US":
            out[key] = out.get(key, 0.0) + float(row["weight"]) / 100.0
    return out


def load():
    T = {d: us_w(s) for d, s in load_fund_snapshots("456600", "9999-12-31").items()}
    K = {d: us_w(s) for d, s in load_fund_snapshots("2ETFQ1", "9999-12-31").items()}
    px = {}
    for f, n in (("adj", "prices_adj_close.csv"), ("opn", "prices_open.csv"), ("cls", "prices_close.csv")):
        px[f] = pd.read_csv(os.path.join(FROZEN, n), index_col=0, parse_dates=True)
    days = [d.strftime("%Y-%m-%d") for d in px["adj"]["QQQ"].dropna().index]
    for f in px:
        df = px[f].copy()
        df.index = df.index.strftime("%Y-%m-%d")
        px[f] = df.reindex(days)
    return T, K, px, days


def prev_pdf(dates, day):
    i = bisect_left(dates, day) - 1
    return dates[i] if i >= 0 else None


def wk(d):
    return date.fromisoformat(d).isocalendar()[:2]


def sector(key, tk):
    renamed = [o for o, n in CORPORATE_ALIASES.items() if n == key]
    for s in (tk, key.split()[0], *(o.split()[0] for o in renamed)):
        lab = resolve_sector(s, default=UNK)
        if lab != UNK:
            return lab
    return UNK


def fill(raw, sec, stock=0.25, sect=0.60, semis=0.60):
    """비례 채우기를 사건 단위로 따로 구현: 활성 종목이 raw 비율로 자라다 종목·섹터·반도체 예산이 차면 멈춘다."""
    names = [k for k, v in raw.items() if v > 0]
    a = {k: 0.0 for k in names}
    act = set(names)
    goal = sum(raw[k] for k in names)
    inbud = lambda k: sec[k] in SEMIS or sec[k] == UNK
    while act:
        left = goal - sum(a.values())
        if left <= 1e-12:
            break
        ev = [(left / sum(raw[k] for k in act), ("done", None))]
        for k in act:
            ev.append(((stock - a[k]) / raw[k], ("stock", k)))
        for g in {sec[k] for k in act}:
            used = sum(a[k] for k in names if sec[k] == g)
            ev.append(((sect - used) / sum(raw[k] for k in act if sec[k] == g), ("sec", g)))
        br = sum(raw[k] for k in act if inbud(k))
        if br > 0:
            ev.append(((semis - sum(a[k] for k in names if inbud(k))) / br, ("semis", None)))
        dt = max(0.0, min(e[0] for e in ev))
        for k in act:
            a[k] += dt * raw[k]
        # 이번 걸음에서 꽉 찬 것을 모두 멈춘다
        act = {k for k in act if a[k] < stock - 1e-12}
        act = {k for k in act if sum(a[j] for j in names if sec[j] == sec[k]) < sect - 1e-12}
        if sum(a[j] for j in names if inbud(j)) >= semis - 1e-12:
            act = {k for k in act if not inbud(k)}
    return {k: v for k, v in a.items() if v > 1e-12}


class Sim:
    def __init__(self, T, K, px, days):
        self.T, self.K, self.px, self.days = T, K, px, days
        self.tdates, self.kdates = sorted(T), sorted(K)
        self.tk = {}
        for W in list(T.values()) + list(K.values()):
            for key in W:
                self.tk.setdefault(key, yahoo_ticker(key))
        self.sec = {k: sector(k, t) for k, t in self.tk.items()}
        self.arr = {f: {c: px[f][c].to_numpy(float) for c in px[f].columns} for f in px}

    def a(self, f, key):
        return self.arr[f].get(self.tk[key])

    def cand(self, day, universe="union", weight="avg", extra=None):
        tp, kp = prev_pdf(self.tdates, day), prev_pdf(self.kdates, day)
        tw = self.T.get(tp, {}) if tp else {}
        kw = self.K.get(kp, {}) if kp else {}
        if universe == "time":
            out = {k: v for k, v in tw.items() if v > 0}
            for x in (extra or ()):                    # 'TIME + MU만' 민감도
                if x not in out and kw.get(x, 0) > 0:
                    out[x] = kw[x]
            return out
        out = {}
        for k in set(tw) | set(kw):
            t, c = tw.get(k, 0.0), kw.get(k, 0.0)
            v = {"avg": (t + c) / 2, "max": max(t, c), "time": t if t > 0 else c, "equal": 1.0}[weight]
            if v > 0:
                out[k] = v
        return out

    def rank(self, k, day, lookback=120, **kw):
        c = self.cand(day, **kw)
        rows = []
        for key in sorted(c):
            p = self.a("adj", key)
            if p is None or k - 1 - lookback < 0:
                continue
            p1, p0 = p[k - 1], p[k - 1 - lookback]
            if np.isfinite(p1) and np.isfinite(p0) and p0 > 0:
                rows.append((key, p1 / p0 - 1, c[key]))
        rows.sort(key=lambda r: -r[1])
        return rows

    def decide(self, k, day, cur, n=7, buf=5, **kw):
        r = self.rank(k, day, **kw)
        if not r:
            return None, r
        order = [x[0] for x in r]
        keep = [x for x in order[:n + buf] if x in cur][:n]
        for x in order:
            if len(keep) >= n:
                break
            if x not in keep:
                keep.append(x)
        wm = {x[0]: x[2] for x in r}
        tot = sum(wm[x] for x in keep)
        return (keep, fill({x: wm[x] / tot for x in keep}, self.sec)), r

    def run(self, n=7, buf=5, **kw):
        days = [d for d in self.days if d >= START]
        k0 = self.days.index(days[0])
        val, w, cur, started, hist, turn = [1.0], {}, set(), False, [], 0.0
        for k in range(k0, len(self.days)):
            day = self.days[k]
            if k > k0:
                r = {}
                for key in w:
                    o = self.a("opn", key)
                    r[key] = o[k] / o[k - 1] - 1 if o is not None and np.isfinite(o[k]) and np.isfinite(o[k - 1]) and o[k - 1] > 0 else 0.0
                g = sum(w[x] * r[x] for x in w)
                tot = 1 + g
                w = {x: w[x] * (1 + r[x]) / tot for x in w}
                val.append(val[-1] * (1 + g))
            if not started or wk(day) != wk(self.days[k - 1]):
                d, _ = self.decide(k, day, cur, n=n, buf=buf, **kw)
                if d:
                    started = True
                    keep, tgt = d
                    turn += sum(abs(tgt.get(x, 0) - w.get(x, 0)) for x in set(tgt) | set(w))
                    w, cur = dict(tgt), set(keep)
            hist.append(dict(w))
        return pd.Series(val, index=days), hist, turn


def per(v):
    r = (v.shift(-1) / v - 1).dropna()
    out = {}
    for lab, a, b in PER:
        x = r[(r.index >= a) & (r.index <= b)]
        cv = (1 + x).cumprod()
        out[lab] = (float(cv.iloc[-1] - 1), float((cv / cv.cummax() - 1).min()))
    out["월"] = {m: float((1 + r[r.index.str.startswith(f"2026-{m:02d}")]).prod() - 1) for m in range(1, 11)}
    return out


def f(x):
    return f"{x:+.4f}"


def main():
    T, K, px, days = load()
    S = Sim(T, K, px, days)
    print("=" * 100, "\n[1] 자료 점검")
    kd = sorted(K)
    gaps = [(kd[i - 1], kd[i]) for i in range(1, len(kd)) if (date.fromisoformat(kd[i]) - date.fromisoformat(kd[i - 1])).days > 4]
    print(f"KoAct PDF {len(kd)}개 {kd[0]}~{kd[-1]}, 4일 넘는 공백: {gaps}")
    td = sorted(T)
    print(f"TIME  PDF {len(td)}개 {td[0]}~{td[-1]}")
    by_tk = defaultdict(set)
    for key, t in S.tk.items():
        by_tk[t].add(key)
    print("같은 티커로 겹치는 키:", {t: ks for t, ks in by_tk.items() if len(ks) > 1} or "없음")
    tkeys = {k for W in T.values() for k in W}
    kkeys = {k for W in K.values() for k in W}
    nopx = sorted(S.tk[k] for k in kkeys | tkeys if S.tk[k] not in px["adj"].columns)
    print("가격 열이 없는 키(후보에서 조용히 빠짐):", nopx or "없음")
    # 회사 이름으로 같은 회사가 두 키인지(예: GOOGL/GOOG)
    print("GOOG/GOOGL 키:", sorted(k for k in tkeys | kkeys if k.split()[0] in ("GOOG", "GOOGL")))
    mu = [k for k in tkeys | kkeys if S.tk[k] == "MU"][0]
    pl = [k for k in tkeys | kkeys if S.tk[k] == "PLTR"][0]
    tmu = [d for d in td if T[d].get(mu, 0) > 0]
    print(f"TIME MU 보유 PDF: {tmu[0] if tmu else None}~{tmu[-1] if tmu else None} ({len(tmu)}개)")
    kmu = [K[d].get(mu, 0) for d in kd]
    print(f"KoAct MU 비중: 보유 {sum(x > 0 for x in kmu)}/{len(kd)}, 평균 {np.mean(kmu):.4f}, 최소 {min(kmu):.4f}, 최대 {max(kmu):.4f}")
    kmu_m = defaultdict(list)
    for d in kd:
        kmu_m[d[:7]].append(K[d].get(mu, 0))
    print("  KoAct MU 월평균:", {m: round(float(np.mean(v)), 4) for m, v in sorted(kmu_m.items())})
    tpl = [d for d in td if T[d].get(pl, 0) > 0]
    print(f"TIME PLTR 보유 PDF {len(tpl)}개 ({tpl[0] if tpl else None}~{tpl[-1] if tpl else None}), KoAct PLTR 보유 {sum(K[d].get(pl, 0) > 0 for d in kd)}/{len(kd)}")

    print("=" * 100, "\n[2] 독립 재계산: union·avg + 기준")
    from deskx import Engine, Params, PERIODS
    E = Engine()
    sec_mis = {S.tk[k]: (S.sec[k], E.sector.get(k)) for k in S.tk if k in E.sector and S.sec[k] != E.sector[k]}
    print("섹터 불일치(독립 vs 엔진):", sec_mis or "없음")
    out = {}
    for name, kw, p in (("기준", dict(universe="time"), Params()),
                        ("union·avg", dict(universe="union", weight="avg"), Params(universe="union", weight="avg")),
                        ("union·time", dict(universe="union", weight="time"), Params(universe="union", weight="time")),
                        ("union·equal", dict(universe="union", weight="equal"), Params(universe="union", weight="equal"))):
        v, hist, turn = S.run(**kw)
        ve, rows = E.run(p)
        dv = float(np.max(np.abs(v.to_numpy() - ve.to_numpy())))
        dw = max(abs(hist[i].get(x, 0) - rows[i]["w"].get(x, 0)) for i in range(len(hist)) for x in set(hist[i]) | set(rows[i]["w"]))
        pr = per(v)
        ev = E.evaluate(name, ve, rows)
        print(f"{name:<12} 가치 최대차 {dv:.2e}, 비중 최대차 {dw:.2e} | " +
              " ".join(f"{lab} {f(pr[lab][0])}(MDD {pr[lab][1]:+.3f})" for lab, *_ in PER) + f" | 회전 합 {turn:.2f}")
        for (lab, *_), (plab, *_) in zip(PER, PERIODS):
            assert abs(pr[lab][0] - ev[plab]["수익"]) < 1e-9, (name, lab)
        out[name] = (v, hist, turn, pr)
    print("구간 수익은 엔진 evaluate와 1e-9 안에서 같다.")

    print("=" * 100, "\n[3] 특정 결정 독립 재계산")
    vb, hb, _, _ = out["기준"]
    va, ha, _, _ = out["union·avg"]
    d_all = [d for d in days if d >= START]

    def show(day, kw, hist, top=20):
        k = days.index(day)
        i = d_all.index(day)
        r = S.rank(k, day, **kw)
        tp, kp = prev_pdf(S.tdates, day), prev_pdf(S.kdates, day)
        print(f"-- {day} ({kw}) 쓰인 PDF: TIME {tp}, KoAct {kp}; 후보 {len(r)}개")
        prev = set(hist[i - 1]) if i > 0 else set()
        for j, (key, m, wt) in enumerate(r[:top], 1):
            src = "T" if T[tp].get(key, 0) > 0 else "K"
            print(f"   #{j:<2} {S.tk[key]:<6} {src} mom {m:+.3f} w {wt:.4f} sec {S.sec[key]:<10} "
                  f"직전보유 {'Y' if key in prev else '-'} 결정비중 {hist[i].get(key, 0):.3f}")
        return r

    for day in ("2026-05-26",):
        rb_ = show(day, dict(universe="time"), hb, top=10)
        ra_ = show(day, dict(universe="union", weight="avg"), ha, top=20)
        dell = [x for x in S.tk if S.tk[x] == "DELL"][0]
        print(f"   DELL 순위: TIME 안 #{[x[0] for x in rb_].index(dell) + 1}, 합집합 #{[x[0] for x in ra_].index(dell) + 1}")
    show("2026-06-01", dict(universe="union", weight="avg"), ha, top=10)
    show("2026-02-17", dict(universe="union", weight="avg"), ha, top=12)
    print("-- union·avg 재조정일마다 MU·PLTR 순위와 보유 여부")
    reb = [d for j, d in enumerate(d_all) if j == 0 or wk(d) != wk(d_all[j - 1])]
    line_mu, line_pl = [], []
    for d in reb:
        k = days.index(d)
        r = [x[0] for x in S.rank(k, d, universe="union", weight="avg")]
        i = d_all.index(d)
        line_mu.append(f"{d[5:]}#{r.index(mu) + 1 if mu in r else '-'}{'*' if mu in ha[i] else ''}")
        line_pl.append(r.index(pl) + 1 if pl in r else None)
    print("   MU (*=보유):", " ".join(line_mu))
    print(f"   PLTR 최고 순위 #{min(x for x in line_pl if x)} (재조정 {sum(1 for x in line_pl if x)}회 순위 있음)")
    # MU 그냥 들고 있기(시가→시가)
    o = S.a("opn", mu)
    k0 = days.index(d_all[0])
    for t in ("MU", "DELL", "SNDK"):
        oo = S.arr["opn"][t][k0:]
        oo = oo[np.isfinite(oo)]
        print(f"   {t} 보유 수익 전체 {oo[-1] / oo[0] - 1:+.4f}, 낙폭 {float((oo / np.maximum.accumulate(oo) - 1).min()):+.4f}")

    print("=" * 100, "\n[4] 강건성")
    prb, pra = out["기준"][3], out["union·avg"][3]
    print("월별 union·avg - 기준:", {m: round(pra["월"][m] - prb["월"][m], 4) for m in range(1, 11)})
    for nm in ("union·time", "union·equal"):
        pp = out[nm][3]
        print(f"월별 {nm} - 기준:", {m: round(pp["월"][m] - prb["월"][m], 4) for m in range(6, 11)})
    # 9/1~ 개선 쏠림: 종목 기여 차, 날짜 쏠림
    def contrib(v, hist, a, b):
        k0 = days.index(d_all[0])
        idx = [i for i, d in enumerate(d_all[:-1]) if a <= d <= b]
        base = v.iloc[idx[0]]
        acc, daily = defaultdict(float), []
        for i in idx:
            k = k0 + i
            s = 0.0
            for key, wt in hist[i].items():
                oo = S.a("opn", key)
                r = oo[k + 1] / oo[k] - 1 if np.isfinite(oo[k + 1]) and np.isfinite(oo[k]) and oo[k] > 0 else 0.0
                acc[S.tk[key]] += v.iloc[i] * wt * r / base
                s += v.iloc[i] * wt * r / base
            daily.append((d_all[i], s))
        return acc, daily
    cb, db = contrib(vb, hb, "2026-09-01", "2026-10-09")
    ca, da = contrib(va, ha, "2026-09-01", "2026-10-09")
    diff = {t: ca.get(t, 0) - cb.get(t, 0) for t in set(ca) | set(cb)}
    top = sorted(diff.items(), key=lambda x: -abs(x[1]))[:8]
    tot = sum(diff.values())
    print(f"9/1~ 기여 차 합 {tot:+.4f}, 상위:", {t: round(x, 4) for t, x in top})
    pos = sorted((x for x in diff.values() if x > 0), reverse=True)
    print(f"   양(+) 기여 차 중 상위 2종목 몫 {sum(pos[:2]) / sum(pos):.2f}, 상위 4종목 {sum(pos[:4]) / sum(pos):.2f}")
    dd = sorted(((d, a - b) for (d, a), (_, b) in zip(da, db)), key=lambda x: -x[1])
    print(f"   일별 차 합 {sum(x for _, x in dd):+.4f}, 가장 큰 3일:", [(d, round(x, 4)) for d, x in dd[:3]],
          f"→ 3일 합 {sum(x for _, x in dd[:3]):+.4f}")

    print("-- 이웃 값(감사용 민감도, 고르기 아님): lookback × n, 기준 vs union·avg (앞/뒤/전체)")
    for lb in (100, 120, 140):
        for n in (6, 7, 8):
            rs = []
            for kw in (dict(universe="time"), dict(universe="union", weight="avg")):
                ve, _ = E.run(Params(lookback=lb, n=n, **kw))
                pr = per(ve)
                rs.append(pr)
            b, u = rs
            print(f"   lb {lb} n {n}: 기준 {f(b['앞'][0])}/{f(b['뒤'][0])}/{f(b['전체'][0])}  union·avg {f(u['앞'][0])}/{f(u['뒤'][0])}/{f(u['전체'][0])}"
                  f"  → 앞 {'+' if u['앞'][0] > b['앞'][0] else '-'} 뒤 {'+' if u['뒤'][0] > b['뒤'][0] else '-'}")

    print("-- 사후 문턱(KoAct 전용은 KoAct 비중 ≥ x) 1~5%, union·time")
    from deskx_universe import CoreEngine
    CE = CoreEngine()
    for x in (0.01, 0.02, 0.03, 0.04, 0.05):
        CE.min_koact = x
        ve, _ = CE.run(Params(universe="union", weight="time"))
        pr = per(ve)
        print(f"   ≥{x:.0%}: 앞 {f(pr['앞'][0])} 뒤 {f(pr['뒤'][0])} 9/1~ {f(pr['9/1~'][0])} 전체 {f(pr['전체'][0])} MDD {pr['전체'][1]:+.3f}")

    print("-- 'TIME + MU만'(MU를 KoAct 비중으로 후보에 추가, 나머지는 #774 그대로)")
    v, hist, turn = S.run(universe="time", extra=(mu,))
    pr = per(v)
    print(f"   앞 {f(pr['앞'][0])} 뒤 {f(pr['뒤'][0])} 9/1~ {f(pr['9/1~'][0])} 전체 {f(pr['전체'][0])} MDD {pr['전체'][1]:+.3f}"
          f" | 월 6~10 " + " ".join(f"{pr['월'][m]:+.3f}" for m in range(6, 11)))
    days_mu = sum(1 for h in hist if mu in h)
    print(f"   MU 보유일 {days_mu}")

    print("-- 회전율·스프레드 민감도(편도 0.1%·0.3%를 재조정 회전에 부과, 수수료 0은 그대로)")
    for nm in ("기준", "union·avg"):
        v, hist, turn, pr = out[nm]
        weeks = sum(1 for j, d in enumerate(d_all) if j == 0 or wk(d) != wk(d_all[j - 1]))
        # 근사: 전체 수익 × (1 - c)^(회전 합)
        approx = {c: (1 + pr["전체"][0]) * (1 - c) ** turn - 1 for c in (0.001, 0.003)}
        print(f"   {nm}: 회전 합 {turn:.2f}({weeks}주, 주평균 {turn / weeks:.3f}) 전체 {f(pr['전체'][0])} → 0.1% {f(approx[0.001])}, 0.3% {f(approx[0.003])}")

    print("=" * 100, "\n[5] 남은 주장 확인")
    o = S.arr["opn"]["MU"]
    k0 = days.index(d_all[0])
    mret = {}
    for m in range(1, 11):
        idx = [i for i, d in enumerate(d_all[:-1]) if d.startswith(f"2026-{m:02d}")]
        if idx:
            mret[m] = round(float(o[k0 + idx[-1] + 1] / o[k0 + idx[0]] - 1), 3)
    print("MU 월별(시가→다음 달 첫 시가):", mret)
    best = None
    for d in reb:
        k = days.index(d)
        r = [x[0] for x in S.rank(k, d, universe="union", weight="avg")]
        if pl in r and (best is None or r.index(pl) + 1 < best[1]):
            best = (d, r.index(pl) + 1, len(r))
    print("PLTR union 최고 순위(날짜, 순위, 후보 수):", best)
    small = ("ONDS", "DRUG", "PEPG", "ORKA", "SIMO", "VICR", "AXTI", "LUNR", "AEHR")
    kw_held = defaultdict(list)
    for i, d in enumerate(d_all):
        kp = prev_pdf(S.kdates, d)
        for key in ha[i]:
            if S.tk[key] in small:
                kw_held[S.tk[key]].append(K[kp].get(key, 0))
    print("union·avg 보유일의 KoAct 비중(최소~최대):", {t: (round(min(v), 4), round(max(v), 4), len(v)) for t, v in kw_held.items()})
    for nm in ("union·avg", "union·time", "union·equal"):
        v, hist, _, _ = out[nm]
        c, _ = contrib(v, hist, "2026-09-01", "2026-10-09")
        print(f"{nm} 9/1~ MU 기여 {c.get('MU', 0):+.4f}")
    vm, hm, _ = S.run(universe="time", extra=(mu,))
    for lab, a_, b_ in PER[:2]:
        c1, _ = contrib(vb, hb, a_, b_)
        c2, _ = contrib(vm, hm, a_, b_)
        dd = {t: c2.get(t, 0) - c1.get(t, 0) for t in set(c1) | set(c2)}
        print(f"'TIME + MU만' - 기준 {lab} 기여 차 상위:", {t: round(x, 4) for t, x in sorted(dd.items(), key=lambda x: -abs(x[1]))[:6]})
    mu_reb = [d[5:] for d in reb if mu in hm[d_all.index(d)]]
    print("'TIME + MU만' MU 보유 재조정일:", mu_reb)


if __name__ == "__main__":
    main()
