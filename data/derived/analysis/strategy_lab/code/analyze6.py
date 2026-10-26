"""Stage 6: canonical hypotheses, blends, regime switching, leverage reference, bootstrap,
and a walk-forward over a small pre-registered rule set."""

import json

import numpy as np

import common as C
import data as D
import engine as E
import grid as G

d = D.load()
m = E.build_market(d)
names, stack = E.build_scores(m)
ctx = G.Context(m)
mk = G.market_overlays(m, d)
bm = json.loads((D.CACHE / "benchmarks.json").read_text(encoding="utf-8"))
ETF_T, ETF_K = bm["ETF_T_adj"]["total"], bm["ETF_K_adj"]["total"]


def rule(src, score, N=5, wt="FW", buf=None, reb="D", ex="none", pre="all"):
    return [names.index((src, pre, score)), N, G.WEIGHTINGS.index(wt), N if buf is None else buf,
            G.REBS.index(reb), G.EXITS.index(ex), E.SOURCES.index(src)]


# ---- canonical hypotheses: one representative per idea (N=5, fund weights, buffer=N) ----
H = {
    "REPL_T  TIME 미국 전량 복제": rule("T", "W", N=500, buf=0),
    "REPL_K  KoAct 미국 전량 복제": rule("K", "W", N=500, buf=0),
    "H1 TIME 비중 Top10·월1회": rule("T", "W", N=10, reb="M", buf=0),
    "H1b TIME 비중 Top5·매일": rule("T", "W"),
    "H2 공통보유 비중 Top5·매일": rule("I", "W"),
    "H3 TIME 실매수(5일) Top5": rule("T", "FLOW5"),
    "H4 두펀드 공통매수(20일) Top5": rule("U", "CONS20"),
    "H5 TIME 신규편입(10일) Top5": rule("T", "NEW10"),
    "H6 TIME 120일 모멘텀 Top5·매일": rule("T", "MOM120"),
    "H6b TIME 120일 모멘텀 Top5·주1회": rule("T", "MOM120", reb="W"),
    "H6c TIME 120일 모멘텀 Top10·월1회": rule("T", "MOM120", N=10, reb="M"),
    "H7 공통보유 위험조정모멘텀 Top5": rule("I", "RADJ120"),
    "H8 공통보유 비중+60일모멘텀 Top5": rule("I", "WMOM60_50"),
    "H9 KoAct 눌림목 Top5·매일": rule("K", "PULL"),
    "H9b 공통보유 눌림목 Top5·매일": rule("I", "PULL"),
    "H9c 공통보유 눌림목 Top5·주1회": rule("I", "PULL", reb="W"),
    "H10 KoAct 단기반전 Top5·매일": rule("K", "REV5"),
    "H11 TIME 모멘텀 1종목·월1회": rule("T", "RADJ120", N=1, reb="M"),
}
series = {k: C.simulate(m, stack, p)[0] for k, p in H.items()}


def summary(krw):
    lr = np.log1p(krw)
    s = C.stats(krw)
    return {"total": s["total"], "mdd": s["mdd"], "sharpe": s["sharpe"],
            "is": np.expm1(lr[:ctx.is_end + 1].sum()), "oos": np.expm1(lr[ctx.is_end + 1:].sum()),
            "months": np.expm1([lr[ctx.month_idx == i].sum() for i in range(9)]),
            "reg": [lr[ctx.regime == g].sum() / max(1, (ctx.regime == g).sum()) * 252 for g in range(3)]}


def show(title, table):
    print(f"\n== {title} ==")
    print("hypothesis                          | total   xTIME xKoAct | MDD   Sharpe | IS     OOS   | up/box/down ann.")
    for k, krw in table.items():
        s = summary(krw)
        print(f"{k:34s} | {s['total'] * 100:+6.0f}% {s['total'] / ETF_T:4.1f}x {s['total'] / ETF_K:4.1f}x  | "
              f"{s['mdd'] * 100:4.0f}% {s['sharpe']:5.2f}  | {s['is'] * 100:+5.0f}% {s['oos'] * 100:+5.0f}% | "
              + "/".join(f"{x * 100:+.0f}" for x in s["reg"]))


show("Canonical hypotheses (cost 0.10%/side, KRW, US close 1/2 -> 9/22)", series)
print("\nmonthly % Jan..Sep:")
for k in ("REPL_T  TIME 미국 전량 복제", "H6 TIME 120일 모멘텀 Top5·매일", "H9 KoAct 눌림목 Top5·매일",
          "H9b 공통보유 눌림목 Top5·매일", "H10 KoAct 단기반전 Top5·매일"):
    print(f"  {k:32s} " + " ".join(f"{x * 100:+5.1f}" for x in summary(series[k])["months"]))

# ---- blends and regime switching between momentum (A) and pullback (B) ----
A = series["H6 TIME 120일 모멘텀 Top5·매일"]
B = series["H9b 공통보유 눌림목 Top5·매일"]
B2 = series["H9 KoAct 눌림목 Top5·매일"]
print(f"\ncorrelation daily A vs B(공통 눌림목) {np.corrcoef(A[1:], B[1:])[0, 1]:.2f}, "
      f"A vs B2(KoAct 눌림목) {np.corrcoef(A[1:], B2[1:])[0, 1]:.2f}, A vs REPL_T {np.corrcoef(A[1:], series['REPL_T  TIME 미국 전량 복제'][1:])[0, 1]:.2f}")


def switch(sig, on_series, off_series, cost=G.COST):
    """Hold on_series when sig[t-1] (decided at close t-1) else off_series; 2*cost per switch."""
    prev = np.concatenate([[1.0], sig[:-1]])
    out = np.where(prev > 0, on_series, off_series).copy()
    flips = np.abs(np.diff(np.concatenate([[1.0], sig]))) > 0
    out[flips] -= 2 * cost
    return out


blends = {
    "50/50 A+B(공통 눌림목)": 0.5 * A + 0.5 * B,
    "50/50 A+B2(KoAct 눌림목)": 0.5 * A + 0.5 * B2,
    "70/30 A+B": 0.7 * A + 0.3 * B,
    "A if QQQ>MA50 else B": switch(mk["QQQ>MA50"], A, B),
    "A if SMH>MA20 else B": switch(mk["SMH>MA20"], A, B),
    "A if QQQ20>50 else B": switch(mk["QQQ20>50"], A, B),
    "A if BREADTH>50 else B": switch(mk["BREADTH>50"], A, B),
}
# leverage references on following TIME's US book (daily reset, financing 6%/yr on borrowed part)
R_T = series["REPL_T  TIME 미국 전량 복제"]
fin = 0.06 / 252
for L in (1.5, 2.0, 2.5):
    blends[f"REPL_T x{L} 레버리지"] = L * R_T - (L - 1) * fin
blends["A x1.0 (reference)"] = A
show("Blends, regime switches, and leverage references", blends)

# ---- stationary bootstrap of the paired daily returns ----
rng = np.random.default_rng(11)
keys = ["REPL_T  TIME 미국 전량 복제", "H6 TIME 120일 모멘텀 Top5·매일", "H9b 공통보유 눌림목 Top5·매일",
        "H9 KoAct 눌림목 Top5·매일", "H10 KoAct 단기반전 Top5·매일"]
mat = np.stack([series[k][1:] for k in keys] + [blends["50/50 A+B(공통 눌림목)"][1:]], axis=0)
labels = keys + ["50/50 A+B"]
T = mat.shape[1]
n_boot, p_new = 5000, 1 / 10
tot = np.zeros((n_boot, len(labels)))
for b in range(n_boot):
    idx = np.empty(T, int)
    idx[0] = rng.integers(T)
    for t in range(1, T):
        idx[t] = rng.integers(T) if rng.random() < p_new else (idx[t - 1] + 1) % T
    tot[b] = np.prod(1 + mat[:, idx], axis=1) - 1
print("\n== Stationary bootstrap (mean block 10 days, 5000 paths): total return distribution ==")
for i, lab in enumerate(labels):
    q = np.percentile(tot[:, i] * 100, [5, 25, 50, 75, 95])
    beat = np.mean(tot[:, i] > tot[:, 0]) * 100
    two_x = np.mean(tot[:, i] >= 2 * np.maximum(tot[:, 0], 0)) * 100
    print(f"{lab:32s} p5 {q[0]:+5.0f}%  p25 {q[1]:+5.0f}%  p50 {q[2]:+5.0f}%  p75 {q[3]:+5.0f}%  p95 {q[4]:+6.0f}%"
          f" | P(beat REPL_T) {beat:4.0f}%  P(>=2x REPL_T return) {two_x:4.0f}%")

# ---- walk-forward over the small pre-registered set ----
cand = [k for k in H if not k.startswith("REPL_K")]
mon = {k: np.array([np.log1p(series[k])[ctx.month_idx == i].sum() for i in range(9)]) for k in cand}
print("\n== Walk-forward over the 17 pre-registered rules (monthly switch cost 0.2%) ==")
for lb in (1, 2, 3, 99):
    real, picks = [], []
    for mi in range(1, 9):
        lo = max(0, mi - lb)
        best = max(cand, key=lambda k: mon[k][lo:mi].sum())
        picks.append(best.split()[0])
        real.append(np.expm1(mon[best][mi]) - 0.002)
    tot_wf = np.prod(1 + np.array(real)) - 1
    print(f"  lookback {'all' if lb == 99 else str(lb) + 'm':>3}: Feb-Sep {tot_wf * 100:+6.1f}%  picks {' '.join(picks)}")
for k in ("REPL_T  TIME 미국 전량 복제", "H6 TIME 120일 모멘텀 Top5·매일", "H9b 공통보유 눌림목 Top5·매일"):
    print(f"  static {k.split()[0]:6s}: Feb-Sep {np.expm1(mon[k][1:].sum()) * 100:+6.1f}%")
