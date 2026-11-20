"""갈래 adaptive — 장세에 따라 스스로 바뀌는 모멘텀 규칙 (2026-11-20).

계열
  F0 static  : 고정 모멘텀 lb∈{20,60,120,250}, 반전(직전 5세션 최하위) — 부품·참고용, 최종 후보에서 제외(사전 고정)
  F1 icdial  : 모멘텀 순위 ↔ 뒤 h세션 수익 순위 상관(확정된 값의 최근 K개 평균, h일 뒤부터)이 음수면 대체 바구니
  F2 lbrot   : 매주 lb∈{20,60,120,250} 바구니 중 직전 L세션 실현 성적 1위 바구니를 따름
  F3 regime  : QQQ·SOXX 종가가 L일선 아래면 노출 0 또는 0.5 (기본 바구니 = lb120)
  F4 voltgt  : 기본 바구니(lb120)의 직전 20세션 실현 변동성으로 노출 = min(1, 목표/변동성)
시점: 점수 k-1 종가까지(엔진), 다이얼은 ic[m](m-1+h 종가 확정) 을 m ≤ k-h 만, 노출 j는 j-1 종가까지.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "code"))
import engine as E  # noqa: E402

UNIVS = ["TIME", "FUNDS", "AI", "ALL50", "WIDETECH50"]
NS = [7, 10]
LBS = [20, 60, 120, 250]
FUND_U = {"TIME", "KOACT", "FUNDS"}
W26 = (E.Y26, E.LAST)
W25 = ("2025-07-01", "2025-12-31")


def weight_of(u):
    return "time" if u == "TIME" else "equal"


# ---------------------------------------------------------------- picks
def buffer_pick(order, prev, n, buffer):
    top = list(order[: n + buffer])
    chosen = [c for c in top if c in prev][:n]
    for c in order:
        if len(chosen) >= n:
            break
        if c not in chosen:
            chosen.append(c)
    return chosen


def recording_pick(rec, n, buffer=5):
    def pick(k, order, sk, prev):
        ch = buffer_pick(order, prev, n, buffer)
        rec[k] = list(ch)
        return ch
    return pick


# ---------------------------------------------------------------- IC dial
class Dial:
    def __init__(self, e):
        self.e = e
        self.cache = {}

    def ic(self, univ, lb, h):
        key = (univ, lb, h)
        if key in self.cache:
            return self.cache[key]
        e = self.e
        S = e.ret(lb)
        A = e.AC
        N = len(e.days)
        out = np.full(N, np.nan)
        for k in range(lb + 2, N):
            if k - 1 + h >= N:
                break
            m = e.universe(univ, k)
            x = S[k][m]
            y = A[k - 1 + h][m] / A[k - 1][m] - 1
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 10:
                continue
            rx = np.argsort(np.argsort(x[ok])).astype(float)
            ry = np.argsort(np.argsort(y[ok])).astype(float)
            out[k] = np.corrcoef(rx, ry)[0, 1]
        self.cache[key] = out
        return out

    def dial(self, univ, lb, h, K):
        """D[k] = mean(ic[k-h-K+1 .. k-h]) — ic[m] 은 m-1+h 종가에 확정 → 결정 k ≥ m+h 에서 사용."""
        ic = self.ic(univ, lb, h)
        N = len(ic)
        D = np.full(N, np.nan)
        for k in range(N):
            hi = k - h
            lo = hi - K + 1
            if lo < 0:
                continue
            seg = ic[lo:hi + 1]
            f = seg[np.isfinite(seg)]
            if len(f) >= max(3, K // 2):
                D[k] = f.mean()
        return D


def icdial_pick(D, alt_score, n, buffer=5, log=None):
    def pick(k, order, sk, prev):
        if np.isfinite(D[k]) and D[k] < 0:
            cand = np.asarray(order)
            a = alt_score[k][cand]
            cand = cand[np.isfinite(a)]
            o2 = cand[np.argsort(-alt_score[k][cand], kind="stable")]
            if log is not None:
                log.append(1)
            return buffer_pick(o2, prev, n, buffer)
        if log is not None:
            log.append(0)
        return buffer_pick(order, prev, n, buffer)
    return pick


# ---------------------------------------------------------------- lookback rotation
def lbrot_pick(subs, L, default=120, log=None):
    """subs: {lb: (Result, rec)} — 각 바구니 실제 경로. 결정 k 는 k-1 종가까지 v_close 로 L세션 성적 비교."""
    def pick(k, order, sk, prev):
        perf = {}
        for lb, (res, rec, k0s) in subs.items():
            j = k - k0s
            if j - 1 < 0:
                continue
            lo = max(0, j - 1 - L)
            if j - 1 - lo < 5:
                continue
            base = res.v_close[lo] if lo < j - 1 else res.v_open[0]
            perf[lb] = res.v_close[j - 1] / base - 1
        ranked = sorted(perf, key=lambda b: -perf[b]) + [default]
        for lb in ranked:
            res, rec, k0s = subs[lb]
            ks = [x for x in rec if x <= k]
            if ks and rec[max(ks)]:
                if log is not None:
                    log.append(lb)
                return rec[max(ks)]
        return []
    return pick


# ---------------------------------------------------------------- exposures
def expo_regime(e, start, end, sym, L, low):
    k0, k1 = e.pos[start], e.pos[end]
    syms = ["QQQ", "SOXX"] if sym == "ANY" else [sym]
    ex = np.ones(k1 - k0 + 1)
    for s in syms:
        a = e.bench_ac[s].to_numpy(float)
        for j, k in enumerate(range(k0, k1 + 1)):
            if k - L < 0:
                continue
            sma = a[k - L:k].mean()          # closes k-L .. k-1
            if a[k - 1] < sma:
                ex[j] = low
    return ex


def expo_voltgt(base, tgt, win=20):
    rc = np.r_[base.v_close[0] / base.v_open[0] - 1, base.v_close[1:] / base.v_close[:-1] - 1]
    T = len(rc)
    ex = np.ones(T)
    for j in range(T):
        seg = rc[max(0, j - win):j]          # closes through j-1
        if len(seg) >= 10:
            v = seg.std(ddof=1) * np.sqrt(252)
            ex[j] = min(1.0, tgt / v) if v > 0 else 1.0
    return ex


def scaled(res):
    """walk_forward 용: 노출을 반영한 세션 수익(엔진 walk_forward 는 r_oo/r_oc 를 노출 없이 씀)."""
    return E.Result(res.days, res.v_open, res.v_close, res.expo, res.r_oo * res.expo, res.r_oc * res.expo,
                    res.turnover, {})


# ---------------------------------------------------------------- builders (재계산 가능)
class Builder:
    def __init__(self, e):
        self.e = e
        self.dialer = Dial(e)
        self.S = {lb: e.ret(lb) for lb in LBS + [5]}
        self.REV = -self.S[5]
        self.sub_cache = {}

    def kw(self, u, n, **o):
        d = dict(universe=u, n=n, buffer=5, weight=weight_of(u), caps=True, freq="W")
        d.update(o)
        return d

    def static(self, u, n, lb, start=E.Y26, end=E.LAST, **o):
        sc = self.REV if lb == "rev5" else self.S[lb]
        return self.e.run(sc, start=start, end=end, **self.kw(u, n, **o))

    def sub(self, u, n, lb, start, end):
        """lbrot 용 부품 바구니: 펀드 종목군 아니면 70세션 앞에서 시작(성적 기록용)."""
        key = (u, n, lb, start, end)
        if key not in self.sub_cache:
            s0 = start if u in FUND_U else self.e.days[max(0, self.e.pos[start] - 70)]
            rec = {}
            r = self.e.run(self.S[lb], start=s0, end=end, pick=recording_pick(rec, n), **self.kw(u, n))
            self.sub_cache[key] = (r, rec, self.e.pos[s0])
        return self.sub_cache[key]

    def icdial(self, u, n, lb, h, K, alt, du, start=E.Y26, end=E.LAST, log=None, **o):
        D = self.dialer.dial(du, lb, h, K)
        alt_s = self.REV if alt == "rev5" else self.S[20]
        return self.e.run(self.S[lb], start=start, end=end, pick=icdial_pick(D, alt_s, n, log=log), **self.kw(u, n, **o))

    def lbrot(self, u, n, L, start=E.Y26, end=E.LAST, log=None, **o):
        subs = {lb: self.sub(u, n, lb, start, end) for lb in LBS}
        # 정렬 순서용 점수는 lb120(후보 필터에만 쓰임, 선택은 pick 이 부품 바구니에서 가져옴)
        sc = np.where(np.isfinite(self.S[120]), self.S[120], 0.0)
        return self.e.run(sc, start=start, end=end, pick=lbrot_pick(subs, L, log=log), **self.kw(u, n, **o))

    def regime(self, u, n, sym, L, low, start=E.Y26, end=E.LAST, **o):
        ex = expo_regime(self.e, start, end, sym, L, low)
        return self.static(u, n, 120, start=start, end=end, exposure=ex, **o)

    def voltgt(self, u, n, tgt, start=E.Y26, end=E.LAST, **o):
        base = self.static(u, n, 120, start=start, end=end)
        ex = expo_voltgt(base, tgt)
        return self.static(u, n, 120, start=start, end=end, exposure=ex, **o)

    def build(self, fam, p, **o):
        if fam == "F0":
            return self.static(p["u"], p["n"], p["lb"], **o)
        if fam == "F1":
            return self.icdial(p["u"], p["n"], p["lb"], p["h"], p["K"], p["alt"], p["du"], **o)
        if fam == "F2":
            return self.lbrot(p["u"], p["n"], p["L"], **o)
        if fam == "F3":
            return self.regime(p["u"], p["n"], p["sym"], p["L"], p["low"], **o)
        if fam == "F4":
            return self.voltgt(p["u"], p["n"], p["tgt"], **o)
        raise ValueError(fam)

    def base_of(self, fam, p, **o):
        """노출 대조용: 같은 바구니를 노출 1로."""
        return self.static(p["u"], p["n"], 120, **o)


def grid():
    G = []
    for u in UNIVS:
        for n in NS:
            for lb in LBS + ["rev5"]:
                G.append(("F0", dict(u=u, n=n, lb=lb)))
    for u in UNIVS:
        dus = ["ALL50"] if u == "ALL50" else ["ALL50", u]
        for n in NS:
            for du in dus:
                for lb in [60, 120]:
                    for h in [10, 20]:
                        for K in [10, 20]:
                            for alt in ["rev5", "mom20"]:
                                G.append(("F1", dict(u=u, n=n, lb=lb, h=h, K=K, alt=alt, du=du)))
    for u in UNIVS:
        for n in NS:
            for L in [20, 40, 60]:
                G.append(("F2", dict(u=u, n=n, L=L)))
    for u in UNIVS:
        for n in NS:
            for sym in ["QQQ", "SOXX", "ANY"]:
                for L in [20, 50, 100, 200]:
                    for low in [0.0, 0.5]:
                        G.append(("F3", dict(u=u, n=n, sym=sym, L=L, low=low)))
    for u in UNIVS:
        for n in NS:
            for tgt in [0.25, 0.35, 0.45, 0.60]:
                G.append(("F4", dict(u=u, n=n, tgt=tgt)))
    return G


def cid(fam, p):
    return fam + ":" + ",".join(f"{k}={v}" for k, v in p.items())


def main():
    t0 = time.time()
    e = E.Engine()
    B = Builder(e)
    base = e.run(e.ret(120), universe="TIME", n=7, buffer=5, weight="time", caps=True)
    bm = e.metrics(base)
    bm["H1_MDD"] = e.mdd(base, "2026-01-02", "2026-06-30")
    print("baseline", {k: round(v, 4) for k, v in bm.items()}, flush=True)

    G = grid()
    rows, results = [], {}
    for i, (fam, p) in enumerate(G):
        log = [] if fam in ("F1", "F2") else None
        r = B.build(fam, p, **({"log": log} if log is not None else {}))
        m = e.metrics(r)
        m["H1_MDD"] = e.mdd(r, "2026-01-02", "2026-06-30")
        if log:
            if fam == "F1":
                m["alt_share"] = float(np.mean(log))
            else:
                m["lb_share"] = {str(lb): float(np.mean([x == lb for x in log])) for lb in LBS}
        c = cid(fam, p)
        rows.append(dict(id=c, family=fam, params=p, **m))
        results[c] = r
        if i % 50 == 0:
            print(i, len(G), round(time.time() - t0, 1), flush=True)

    # ---------------------------------------------------- 고르기 (상반기만, 사전 고정)
    lim = bm["H1_MDD"] - 0.05
    elig = [r for r in rows if r["family"] != "F0" and r["H1_MDD"] >= lim and np.isfinite(r["H1"])]
    for r in elig:
        r["calmar_H1"] = r["H1"] / abs(r["H1_MDD"]) if r["H1_MDD"] < 0 else float("inf")
    elig.sort(key=lambda r: -r["calmar_H1"])
    fin, fams = [], set()
    for r in elig:
        if r["family"] in fams:
            continue
        fin.append(r)
        fams.add(r["family"])
        if len(fin) == 3:
            break
    fam_best = {}
    for r in elig:
        fam_best.setdefault(r["family"], []).append(r["id"])
    fam_top = {f: v[:5] for f, v in fam_best.items()}

    # ---------------------------------------------------- 후보 상세
    finals = []
    for r in fin:
        fam, p = r["family"], r["params"]
        d = {k: r[k] for k in ["id", "family", "params", "H1", "H1_MDD", "Y2026", "H2", "RECENT", "MDD2026",
                               "MDD_H2", "avg_expo", "turnover", "calmar_H1"]}
        rd = B.build(fam, p, delay=1)
        d["delay1"] = e.metrics(rd)
        rc = B.build(fam, p, cost_bps=10)
        d["cost10"] = e.metrics(rc)
        if p["u"] in FUND_U:
            d["Y2025H2"] = None
        else:
            if fam == "F3":
                r25 = B.regime(p["u"], p["n"], p["sym"], p["L"], p["low"], start=W25[0], end=W25[1])
            elif fam == "F4":
                r25 = B.voltgt(p["u"], p["n"], p["tgt"], start=W25[0], end=W25[1])
            elif fam == "F1":
                r25 = B.icdial(p["u"], p["n"], p["lb"], p["h"], p["K"], p["alt"], p["du"], start=W25[0], end=W25[1])
            else:
                r25 = B.lbrot(p["u"], p["n"], p["L"], start=W25[0], end=W25[1])
            d["Y2025H2"] = e.metrics_2025(r25)
        if r["avg_expo"] < 0.999:
            b0 = B.base_of(fam, p)
            ctl = e.exposure_control(b0, r["avg_expo"])
            d["expo_ctrl"] = e.metrics(ctl)
            d["base_full"] = e.metrics(b0)
        else:
            d["expo_ctrl"] = None
        finals.append(d)
        print("FINAL", d["id"], flush=True)

    # ---------------------------------------------------- 이동 전방 검증
    def wf(ids):
        out = e.walk_forward({c: scaled(results[c]) for c in ids})
        rr = out["result"]
        return {"Mar_Nov": float(rr.v_close[-1] / rr.v_open[0] - 1), "MDD": e.mdd(rr),
                "n_switch": out["n_switch"], "picks_by_month": out["picks_by_month"], "n_configs": len(ids)}
    all_ids = [r["id"] for r in rows]
    wf_all = wf(all_ids)
    wf_adapt = wf([r["id"] for r in rows if r["family"] != "F0"])
    wf_fam = {f: wf([r["id"] for r in rows if r["family"] == f]) for f in ["F0", "F1", "F2", "F3", "F4"]}
    # 엔진 walk_forward 를 노출 미반영 그대로 돌린 값(이상 보고용)
    raw = e.walk_forward({c: results[c] for c in all_ids})
    wf_raw_all = float(raw["result"].v_close[-1] / raw["result"].v_open[0] - 1)
    qqq = e.bench("QQQ")
    bench = {"QQQ_Mar_Nov": e.span(qqq, "2026-03-02", E.LAST),
             "base_Mar_Nov": e.span(base, "2026-03-02", E.LAST)}

    out = dict(baseline=bm, n_configs=len(rows), grid_counts={f: sum(r["family"] == f for r in rows) for f in
                                                                ["F0", "F1", "F2", "F3", "F4"]},
               elig_limit_H1_MDD=lim, n_eligible=len(elig), finalists=finals, family_top5_H1=fam_top,
               walk_forward=dict(all=wf_all, adaptive_only=wf_adapt, by_family=wf_fam,
                                 engine_raw_all_unscaled=wf_raw_all, bench=bench),
               rows=rows, runtime_s=time.time() - t0)
    with open(HERE / "results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print("done", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
