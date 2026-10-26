"""판정 하네스(BRIEF v2 §7·§8). 실제 SEIBRO 신호든 대리 신호든 같은 함수로 돌린다.

핵심: 구간(앞·뒤) 진입 열 안에서 켜짐 열을 k = 0..n−1 모두 원형 이동한 효과를 FFT로 한 번에 계산한다.
효과는 가설 방향 점수 e_h(클수록 가설 쪽)로 통일한다.
- 높음 칸: 수익·낙폭은 −Δ, 사건은 +RD. 낮음 칸: 수익·낙폭은 +Δ, 사건은 −RD.
- Δ = 평균(켜짐) − 평균(꺼짐), RD = P(적중|켜짐) − P(적중|꺼짐).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DER = ROOT / "derived"

FRONT = ("2026-01-02", "2026-05-29")
BACK = ("2026-06-01", "2026-10-23")
AUX25 = ("2025-01-02", "2025-12-31")
OUTCOMES = ["R1", "R3", "R5", "MDD5", "A5", "T5"]
H = {"R1": 1, "R3": 3, "R5": 5, "MDD5": 5, "A5": 5, "T5": 5}
EVENT = {"A5", "T5"}

PAIRS = (
    [(s, "SOXX") for s in ["S1", "S2", "S7", "L1", "C1", "C2"]]
    + [(s, t) for s in ["S3", "S4", "S5n", "S5b", "S8", "L2", "C3", "C4"] for t in ["SOXX", "QQQ"]]
    + [(s, "EWY") for s in ["S9n", "S9b"]]
    + [(s, "QQQ") for s in ["S10n", "S10b"]]
)
COMBOS = [("P1", ("S1", "C1"), "SOXX"), ("P2", ("S3", "S7"), "SOXX"), ("P2", ("S3", "S7"), "QQQ"),
          ("P4", ("L2", "C3"), "SOXX"), ("P4", ("L2", "C3"), "QQQ")]
PRIMARY = [("S1", "SOXX"), ("S3", "SOXX"), ("S7", "SOXX"), ("S9n", "EWY")]
CENSORED = {"S9n": "cens_KORU", "S9b": "cens_KORU"}


# ---------- 가격·결과 ----------

def load_prices() -> pd.DataFrame:
    px = pd.read_csv(ROOT / "raw" / "prices.csv", parse_dates=["date"])
    return px


def events_asof(close: pd.Series, asof: str) -> dict:
    from events import drawdown_starts, zigzag_tops

    a = [pd.Timestamp(x["A"]) for x in drawdown_starts(close, asof)]
    tops, bottoms, pend = zigzag_tops(close, asof)
    t = [pd.Timestamp(x["T"]) for x in tops]
    last_b = max([pd.Timestamp(x["B"]) for x in bottoms], default=pd.Timestamp("1900-01-01"))
    if pend["mode"] == "down":
        t_ok_until = pd.Timestamp(pend["extreme"])
    else:
        t_ok_until = last_b
    return {"A": a, "T": t, "t_ok_until": t_ok_until}


def outcomes(px: pd.DataFrame, entries: pd.DatetimeIndex, ticker: str, period: tuple[str, str]) -> pd.DataFrame:
    """period 안 진입의 결과. 결과 창이 period 끝을 넘으면 NaN. 사건은 period 끝까지의 가격만으로 정한다."""
    p = px[px["ticker"] == ticker].set_index("date").sort_index()
    p = p[~p.index.duplicated()]
    p = p.reindex(entries)
    lo, hi = pd.Timestamp(period[0]), pd.Timestamp(period[1])
    idx = np.arange(len(entries))
    sel = (entries >= lo) & (entries <= hi)
    end_i = int(np.searchsorted(entries, hi, side="right")) - 1
    o, c = p["Open"].values, p["Close"].values
    out = pd.DataFrame(index=entries[sel])
    ii = idx[sel]
    for name, h in [("R1", 1), ("R3", 3), ("R5", 5)]:
        ok = ii + h - 1 <= end_i
        v = np.full(len(ii), np.nan)
        v[ok] = c[ii[ok] + h - 1] / o[ii[ok]] - 1
        out[name] = v
    ok = ii + 4 <= end_i
    v = np.full(len(ii), np.nan)
    v[ok] = [c[i:i + 5].min() / o[i] - 1 for i in ii[ok]]
    out["MDD5"] = v
    full = px[px["ticker"] == ticker].set_index("date")["Close"].dropna()
    full = full[~full.index.duplicated()]
    ev = events_asof(full, period[1])
    pos = {d: k for k, d in enumerate(entries)}
    for name, dates in [("A5", ev["A"]), ("T5", ev["T"])]:
        hit = np.zeros(len(ii))
        for d in dates:
            if d in pos:
                k = pos[d]
                hit[(ii <= k) & (k <= ii + 4)] = 1
        if name == "A5":
            ok = ii + 8 <= end_i
        else:
            last = int(np.searchsorted(entries, ev["t_ok_until"], side="right")) - 1
            ok = ii + 4 <= min(end_i, last)
        hit[~ok] = np.nan
        out[name] = hit
    return out


# ---------- 백분위·켜짐 ----------

def rolling_pct(x: pd.Series, win=250, minp=120) -> pd.Series:
    v = x.values.astype(float)
    out = np.full(len(v), np.nan)
    for i in range(len(v)):
        lo = max(0, i - win + 1)
        w = v[lo:i + 1]
        w = w[~np.isnan(w)]
        if len(w) < minp or np.isnan(v[i]):
            continue
        out[i] = 100.0 * ((w < v[i]).sum() + 0.5 * ((w == v[i]).sum() - 1) + 1) / len(w)
    return pd.Series(out, index=x.index)


def censor_ok(c: pd.Series, win=250) -> pd.Series:
    return c.rolling(win, min_periods=1).mean() <= 0.5


def on_vectors(pct: pd.DataFrame, raw: pd.DataFrame) -> dict:
    """칸 무늬(신호, 방향, 문턱) → 켜짐(1/0/NaN) 열."""
    ons = {("K1", "high", 1): raw["K1"].astype(float)}
    for s in pct.columns:
        p = pct[s]
        for thr in (80, 90):
            ons[(s, "high", thr)] = (p >= thr).astype(float).where(p.notna())
        for thr in (20, 10):
            ons[(s, "low", thr)] = (p <= thr).astype(float).where(p.notna())
    for name, (a, b), _ in COMBOS:
        for thr in (70, 80):
            pa, pb = pct[a], pct[b]
            ons[(name, "high", thr)] = ((pa >= thr) & (pb >= thr)).astype(float).where(pa.notna() & pb.notna())
    return ons


# ---------- 원형 이동 효과(FFT) ----------

def corr_all(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """S(k) = Σ_i a[(i−k) mod n] b[i], k = 0..n−1. a·b는 (…, n)."""
    A, B = np.fft.rfft(a, axis=-1), np.fft.rfft(b, axis=-1)
    return np.fft.irfft(np.conj(A) * B, n=a.shape[-1], axis=-1)


def shift_effects(on: np.ndarray, y: np.ndarray, event: bool) -> np.ndarray:
    """on: (m, n) 켜짐(NaN = 정의 안 됨), y: (n,) 결과(NaN = 없음). 반환 Δ 또는 RD (m, n) — k별."""
    d = ~np.isnan(on)
    o = np.where(d, on, 0.0)
    m = ~np.isnan(y)
    ym = np.where(m, y, 0.0)
    mm = m.astype(float)
    n_on = corr_all(o, np.broadcast_to(mm, o.shape))
    s_on = corr_all(o, np.broadcast_to(ym, o.shape))
    n_def = corr_all(d.astype(float), np.broadcast_to(mm, o.shape))
    s_def = corr_all(d.astype(float), np.broadcast_to(ym, o.shape))
    n_off, s_off = n_def - n_on, s_def - s_on
    with np.errstate(invalid="ignore", divide="ignore"):
        eff = s_on / np.round(n_on) - s_off / np.round(n_off)
    eff[(np.round(n_on) < 1) | (np.round(n_off) < 1)] = np.nan
    return eff


def clusters(on: np.ndarray, h: int) -> int:
    idx = np.flatnonzero(on == 1)
    if len(idx) == 0:
        return 0
    return int(1 + (np.diff(idx) > h).sum())


@dataclass
class Cell:
    signal: str
    target: str
    direction: str
    thr: int
    outcome: str
    primary: bool = False
    info: dict = field(default_factory=dict)


def all_cells() -> list[Cell]:
    cells = []
    for s, t in PAIRS:
        outs = [o for o in OUTCOMES if not (t == "QQQ" and o in EVENT)]
        for d, thrs in [("high", (80, 90)), ("low", (20, 10))]:
            for thr in thrs:
                for o in outs:
                    cells.append(Cell(s, t, d, thr, o, primary=(s, t) in PRIMARY and d == "high" and thr == 80
                                      and o in ("R5", "T5")))
    for o in OUTCOMES:
        cells.append(Cell("K1", "SOXX", "high", 1, o))
    for name, _, t in COMBOS:
        outs = [o for o in OUTCOMES if not (t == "QQQ" and o in EVENT)]
        for thr in (70, 80):
            for o in outs:
                cells.append(Cell(name, t, "high", thr, o))
    return cells


def sign_of(c: Cell) -> float:
    """Δ(또는 RD)를 가설 방향 점수로 바꾸는 부호."""
    bad = 1.0 if c.outcome in EVENT else -1.0  # 높을수록 나쁨인 쪽의 부호
    return bad if c.direction == "high" else -bad


@dataclass
class Period:
    name: str
    entries: pd.DatetimeIndex
    y: dict  # (target, outcome) → np.ndarray


def make_period(name, rng, px, entries_all) -> Period:
    sel = entries_all[(entries_all >= pd.Timestamp(rng[0])) & (entries_all <= pd.Timestamp(rng[1]))]
    ys = {}
    for t in ["SOXX", "QQQ", "EWY", "^SOX"]:
        o = outcomes(px, entries_all, t, rng)
        for k in OUTCOMES:
            ys[(t, k)] = o[k].reindex(sel).values
    return Period(name, sel, ys)


def evaluate_cells(cells, ons, period: Period, sd_ref: dict | None = None) -> pd.DataFrame:
    """각 칸의 실제 효과, 이동 귀무 p·z, 켜짐 수·덩어리 수."""
    rows = []
    groups: dict = {}
    for i, c in enumerate(cells):
        groups.setdefault((c.target, c.outcome), []).append(i)
    for (t, o), ids in groups.items():
        y = period.y[(t, o)]
        O = np.vstack([ons[(cells[i].signal, cells[i].direction, cells[i].thr)].reindex(period.entries).values
                       for i in ids])
        eff = shift_effects(O, y, o in EVENT)
        for j, i in enumerate(ids):
            c = cells[i]
            e = eff[j] * sign_of(c)
            obs, null = e[0], e[1:]
            valid = ~np.isnan(null)
            K = int(valid.sum())
            p = (1 + np.sum(null[valid] >= obs)) / (1 + K) if not np.isnan(obs) and K > 0 else np.nan
            z = (obs - np.nanmean(null)) / np.nanstd(null) if K > 2 and np.nanstd(null) > 0 else np.nan
            ov = O[j]
            usable = ~np.isnan(ov) & ~np.isnan(y)
            n_on = int(np.nansum(ov[usable] == 1))
            ncl = clusters(np.where(usable, ov, 0), H[o])
            rows.append({"i": i, "signal": c.signal, "target": t, "direction": c.direction, "thr": c.thr,
                         "outcome": o, "primary": c.primary, "e_h": obs, "p": p, "z": z, "K": K,
                         "n_on": n_on, "n_use": int(usable.sum()), "clusters": ncl,
                         "raw": eff[j][0], "sd": float(np.nanstd(y)) if sd_ref is None else sd_ref[(t, o)]})
    return pd.DataFrame(rows).set_index("i").sort_index()


def min_on_ok(df: pd.DataFrame) -> pd.Series:
    return (df["n_on"] >= 15) & (df["clusters"] >= 2)


def practical_ok(df: pd.DataFrame) -> pd.Series:
    ev = df["outcome"].isin(EVENT)
    return np.where(ev, df["e_h"] >= 0.20, df["e_h"] >= 0.25 * df["sd"])


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(np.where(np.isnan(p), 9, p))
    m = int((~np.isnan(p)).sum())
    adj = np.full(len(p), np.nan)
    run = 0.0
    for r, i in enumerate(order):
        if np.isnan(p[i]):
            continue
        run = max(run, min(1.0, (m - r) * p[i]))
        adj[i] = run
    return adj


def judge(cells, ons, front: Period, back: Period, sd_front: dict) -> dict:
    """주 가설(Holm)과 탐색(고르기 → 통과 ①②③)."""
    fr = evaluate_cells(cells, ons, front)
    bk = evaluate_cells(cells, ons, back, sd_ref=sd_front)
    fr_ok, bk_ok = min_on_ok(fr), min_on_ok(bk)
    # 주 가설
    prim = bk[bk["primary"]].copy()
    prim["front_e_h"] = fr.loc[prim.index, "e_h"]
    prim["judgeable"] = bk_ok[prim.index] & fr_ok[prim.index]
    prim["p_holm"] = holm(np.where(prim["judgeable"], prim["p"], np.nan))
    prim["pass"] = (prim["judgeable"] & (prim["p_holm"] < 0.05) & (prim["front_e_h"] > 0)
                    & pd.Series(practical_ok(prim), index=prim.index))
    # 탐색 고르기: 같은 켜짐 무늬에서 z 최대 결과 하나
    cand = fr[fr_ok & ~fr["primary"]].dropna(subset=["z"])
    cand = cand.sort_values("z", ascending=False).groupby(["signal", "target", "direction", "thr"]).head(1)
    top = cand.sort_values("z", ascending=False).head(10)
    sel = bk.loc[top.index].copy()
    sel["front_z"] = top["z"]
    sel["judgeable"] = bk_ok[sel.index]
    sel["pass"] = (sel["judgeable"] & (sel["e_h"] > 0) & pd.Series(practical_ok(sel), index=sel.index)
                   & (sel["p"] < 0.05))
    return {"front": fr, "back": bk, "primary": prim, "explore": sel,
            "n_pass_primary": int(prim["pass"].sum()), "n_pass_explore": int(sel["pass"].sum())}


def family_null(cells, ons, px, entries_all, front_rng, back_rng, sd_front, step=1, stepped=()) -> np.ndarray:
    """2026 전체 진입 열에서 켜짐 열을 k만큼 같이 돌리고 고르기·통과를 다시 한다. 반환: k별 탐색 통과 칸 수."""
    lo, hi = pd.Timestamp(front_rng[0]), pd.Timestamp(back_rng[1])
    yr = entries_all[(entries_all >= lo) & (entries_all <= hi)]
    n = len(yr)
    front = make_period("front", front_rng, px, entries_all)
    back = make_period("back", back_rng, px, entries_all)
    counts = []
    fpos = np.isin(yr, front.entries)
    bpos = np.isin(yr, back.entries)
    for k in range(1, n, step):
        rot = {}
        for key, s in ons.items():
            v = s.reindex(yr).values
            if key[0] in stepped:  # 계단 신호는 앞·뒤 안에서만 돌린다
                w = v.copy()
                for msk in (fpos, bpos):
                    seg = v[msk]
                    w[msk] = np.roll(seg, k % max(1, len(seg)))
                rot[key] = pd.Series(w, index=yr)
            else:
                rot[key] = pd.Series(np.roll(v, k), index=yr)
        r = judge(cells, rot, front, back, sd_front)
        counts.append(r["n_pass_explore"])
    return np.array(counts)


def surrogate(X: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """다변량 위상 무작위화(같은 위상을 모든 열에): 자기상관·열 사이 상관 유지."""
    Z = X.ffill().bfill()
    mu = Z.mean()
    F = np.fft.rfft((Z - mu).values, axis=0)
    ph = np.exp(1j * rng.uniform(0, 2 * np.pi, size=F.shape[0]))
    ph[0] = 1.0
    if Z.shape[0] % 2 == 0:
        ph[-1] = 1.0
    S = np.fft.irfft(F * ph[:, None], n=Z.shape[0], axis=0)
    out = pd.DataFrame(S, index=X.index, columns=X.columns) + mu
    return out.where(X.notna())
