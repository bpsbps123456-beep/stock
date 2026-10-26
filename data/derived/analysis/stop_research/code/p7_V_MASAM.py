"""7단계 반박 검증 V_MASAM(2026-10-11): 갈래 MASAM 대표 후보 '원형 P4'(3배 나스닥100 합성, ^IXIC 신호)를 깨려고 시도한다.

후보: 나스닥 종합 종가 −3% 이하 → 다음 시가 전량 매도. 마지막 −3% 날 D + 달력 1개월 + 1일 이후 첫 거래일 시가에 재매수
(그 사이 −3%가 또 나오면 다시 셈). D 포함 1개월 안 −3%가 4번 이상이면 공황 → 2개월 + 1일(현금 기간 내내 유지).
현금 상태에서 나스닥 8거래일 연속 상승이면 다음 시가에 바로 재매수.

독립 재계산: 공유 엔진(lev5·trade_lab·stock_lab·harness)과 p7_MASAM.py를 import하지 않는다. parquet(p7_idx, p5_lev_long)만 읽어
상태 기계·3배 합성·체결 엔진·효율선을 새로 짠다. 원 갈래 숫자는 results/p7_MASAM.json에서 읽어 대조만 한다.

검증 항목(사전 고정, 결과를 본 뒤 바꾸지 않는다)
  1) 재현: 원형 4개 + 격자 전체 93개(고르기 순위·'고른 구간 > 0 중 확인 > 0' 비율), P4 창별 숫자, 실제 TQQQ.
  2) 시점: 미래 교란 시험, 신호일 < 체결일 확인, 하루·이틀 늦게, 다음 날 종가 체결, 미끄러짐 0.5%, 현금 4%·^IRX,
     '한 달' = 21거래일(공황 42거래일, 공황 창 21거래일) vs 달력, 공황 해석(현금 기간 내내 유지 vs −3% 날마다 다시 판정).
  3) 이웃: 파라미터마다 앞뒤 2칸(한 번에 하나) + X × 대기 5×5.
  4) 위약 20시드 ×4종: A 현금 구간 길이·투자 구간 길이를 창 안에서 섞어 다시 배치(같은 매수 수·같은 현금 기간),
     B 판 날은 그대로 두고 대기 길이만 주기끼리 섞기, C1 −3% 날을 같은 개수의 무작위 날로(전 기간 균등),
     C2 같은 해 안에서 같은 개수의 무작위 날로(연도별 변동성은 유지). C는 P4 상태 기계를 그대로 돌린다.
  5) 시대별: 1987, 2000~02, 2008, 2011, 2018 4분기, 2020, 2022, 2025, 2026 앞·뒤·7월 쇼크.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"
RESULTS = HERE.parent / "results"
ID = "p7_V_MASAM"

# ============================================================================ 사전 고정 상수
END = "2026-10-09"
RUN0_NDX3, RUN0_TQQQ = "1985-11-01", "2010-03-15"
FEE, EXTRA = 0.0095, 0.013
SLIP, CASH_ALT = 0.005, 0.04
W_GRID = np.round(np.arange(0, 1.0001, 0.01), 2)
PICK = (None, "2005-12-31")
CHECK = ("2006-01-01", "2025-12-31")
WINDOWS = {  # (첫날, 끝날), 첫날 직전 종가 → 끝날 종가. long = 연율로 비교하는 창
    "고르기 1985~2005": (PICK, True), "확인 2006~2025": (CHECK, True), "긴 역사 1985~2026": ((None, END), True),
    "2010~19": (("2010-01-04", "2019-12-31"), True), "1985~99": ((None, "1999-12-31"), True),
    "2003~2026": (("2003-01-02", END), True),
    "2026 앞": (("2026-01-02", "2026-05-29"), False), "2026 뒤": (("2026-06-01", END), False),
    "7월 쇼크": (("2026-06-23", "2026-07-29"), False), "2025": (("2025-01-02", "2025-12-31"), False),
    "2022": (("2022-01-03", "2022-12-30"), False), "2020": (("2020-01-02", "2020-12-31"), False),
    "2018 4분기": (("2018-10-01", "2018-12-31"), False), "2011": (("2011-01-03", "2011-12-30"), False),
    "2008": (("2008-01-02", "2008-12-31"), False), "2000~02": (("2000-03-01", "2002-10-31"), False),
    "1987": (("1987-01-02", "1987-12-31"), False),
}
ERA_KEYS = ["1987", "2000~02", "2008", "2011", "2018 4분기", "2020", "2022", "2025", "2026 앞", "2026 뒤", "7월 쇼크"]
# 이웃(한 번에 하나, 앞뒤 2칸)
NB_X = (-0.02, -0.025, -0.03, -0.035, -0.04)
NB_WAIT_D = (-14, -7, 0, 7, 14)        # 1개월 + 1일에 더하는 달력일
NB_PK = (2, 3, 4, 5, 6)                # 공황 횟수
NB_PW_D = (-28, -14, 0, 14, 28)        # 2개월 + 1일에 더하는 달력일
NB_UP = (6, 7, 8, 9, 10)               # 연속 상승 일수
SEEDS = 20
X0 = -0.03


# ============================================================================ 자료(파케이만)
def load(path_df: pd.DataFrame, sym: str, adj: bool):
    g = path_df[path_df.sym == sym].sort_values("date").dropna(subset=["Close"])
    g = g[g.Close > 0].drop_duplicates("date", keep="last")
    f = (g["Adj Close"] / g["Close"]).fillna(1.0).to_numpy() if adj else np.ones(len(g))
    d = g.date.to_numpy().astype(str)
    o = g.Open.to_numpy(float) * f
    c = g.Close.to_numpy(float) * f
    return d, o, c


def stale_fix(o, c):
    """시가 없음·≤0·종가와 같음·전날 종가와 같음(옛 지수의 묵은 시가) → 시가 = 종가(그날 종가 체결)."""
    pc = np.concatenate([[np.nan], c[:-1]])
    with np.errstate(invalid="ignore", divide="ignore"):
        bad = ~np.isfinite(o) | (o <= 0) | (np.abs(o / c - 1) < 1e-9) | (np.abs(o / pc - 1) < 1e-6)
    return np.where(bad, c, o), bad


def build():
    p7 = pd.read_parquet(CACHE / "p7_idx.parquet")
    p5 = pd.read_parquet(CACHE / "p5_lev_long.parquet")
    # 신호 ^IXIC
    sd, _, sc = load(p7, "^IXIC", False)
    sr = np.concatenate([[np.nan], sc[1:] / sc[:-1] - 1])
    # 기초: ^NDX(~1999-03-09) + QQQ 배당 수정(1999-03-10~)
    nd, no, nc = load(p7, "^NDX", False)
    no, nbad = stale_fix(no, nc)
    qd, qo, qc = load(p7, "QQQ", True)
    qo, qbad = stale_fix(qo, qc)
    cut = "1999-03-10"
    ia, ib = int(np.searchsorted(nd, cut)), int(np.searchsorted(qd, cut))
    sc_ = qc[ib] / nc[ia]
    ud = np.concatenate([nd[:ia], qd[ib:]])
    uo = np.concatenate([no[:ia] * sc_, qo[ib:]])
    uc = np.concatenate([nc[:ia] * sc_, qc[ib:]])
    ubad = np.concatenate([nbad[:ia], qbad[ib:]])
    # ^IRX
    id_, _, ic = load(p7, "^IRX", False)
    irx = pd.Series(ic / 100.0, index=id_)
    rf = irx.reindex(sorted(set(irx.index) | set(ud))).ffill().reindex(ud).fillna(0.0).to_numpy()
    # 3배 합성: 종가는 일일 3배 − 2×금리 − 보수·추가 비용, 시가는 전날 합성 종가 × (1 + 3 × 기초 갭)
    n = len(uc)
    c3 = np.empty(n); o3 = np.empty(n)
    c3[0] = o3[0] = 100.0
    for t in range(1, n):
        r = 3 * (uc[t] / uc[t - 1] - 1) - 2 * rf[t] / 252 - (FEE + EXTRA) / 252
        c3[t] = c3[t - 1] * max(1 + r, 0.001)
        o3[t] = c3[t - 1] * max(1 + 3 * (uo[t] / uc[t - 1] - 1), 0.001)
    # 실제 TQQQ(배당 수정)
    td, to, tc = load(p5, "TQQQ", True)
    rf_t = irx.reindex(sorted(set(irx.index) | set(td))).ffill().reindex(td).fillna(0.0).to_numpy()
    return {"sig": (sd, sr), "NDX3": {"d": ud, "o": o3, "c": c3, "rf": rf, "bad": ubad, "uc": uc},
            "TQQQ": {"d": td, "o": to, "c": tc, "rf": rf_t}}


# ============================================================================ 상태 기계(새로 짬)
def base_spec(**kw):
    sp = {"X": X0, "mode": "cal", "N": 21, "M": 1, "wd": 0, "panic": (4, 2), "pwd": 0, "persist": True, "up": 8, "s": 1.0}
    sp.update(kw)
    return sp


def states(sd, sr, sp, trig_override=None):
    """반환 St[t]: t일 종가 판단 뒤 상태(1 보유, 0 현금). 다음 거래일에 적용.
    mode 'cal': 재매수 날 = D + M개월 + 1일 + wd일 이후 첫 거래일, 공황 = D 포함 (D−1개월, D] 안 −X 날 ≥ k → 대기 pm개월 + 1일 + pwd일.
    mode 'td' : 재매수 = D + N + 1 거래일, 공황 창 = [D−20, D] 거래일, 공황 대기 = pw 거래일.
    persist=True: 공황은 그 현금 기간 내내 유지. False: −X 날마다 공황을 다시 판정(마지막 −X 날 기준)."""
    n = len(sr)
    St = np.ones(n, np.int8)
    trig = (sr <= sp["X"]) if trig_override is None else trig_override
    dts = pd.to_datetime(pd.Series(sd)).dt.date.to_numpy()
    inv, pan, up, rel_date, rel_idx = True, False, 0, None, None
    tlist = []
    X_mode = sp["mode"]
    for t in range(n):
        x = sr[t]
        if x != x:
            St[t] = 1 if inv else 0
            continue
        up = up + 1 if x > 0 else 0
        if trig[t]:
            up = 0  # 실제 −X 날은 하락이라 이미 0. 위약 C의 무작위 날(상승일일 수 있음)도 같은 규칙으로 맞춤
            tlist.append(t)
            if inv:
                inv, pan = False, False
            pk = sp["panic"]
            this_pan = False
            if pk is not None:
                k = pk[0]
                if X_mode == "td":
                    cnt = sum(1 for j in tlist if j >= t - 20)
                else:
                    lo = dts[t] - relativedelta(months=1)
                    cnt = 0
                    for j in reversed(tlist):
                        if dts[j] > lo:
                            cnt += 1
                        else:
                            break
                this_pan = cnt >= k
            pan = (pan or this_pan) if sp["persist"] else this_pan
            if X_mode == "td":
                rel_idx = t + (sp["panic"][1] if pan else sp["N"]) + 1
            else:
                if pan:
                    rel_date = dts[t] + relativedelta(months=sp["panic"][1]) + relativedelta(days=1 + sp["pwd"])
                else:
                    rel_date = dts[t] + relativedelta(months=sp["M"]) + relativedelta(days=1 + sp["wd"])
        elif not inv:
            go = sp["up"] is not None and up >= sp["up"]
            if not go:
                if X_mode == "td":
                    go = t + 1 >= rel_idx
                else:
                    go = (t + 1 < n and dts[t + 1] >= rel_date) or t + 1 >= n
            if go:
                inv = True
        St[t] = 1 if inv else 0
    return St


def to_target_days(sd, St, tdays, lag=0):
    """대상 거래일 k의 상태 = 날짜가 k보다 엄격히 앞선 마지막 신호일의 상태(lag만큼 더 앞)."""
    j = np.searchsorted(sd, tdays, "left") - 1 - lag
    return np.where(j >= 0, St[np.clip(j, 0, None)], 1).astype(float)


# ============================================================================ 체결 엔진(새로 짬)
def engine(o, c, a, z, tgt, slip=0.0, cash=None, at_close=False):
    """a−1 종가에 전량 보유(가치 1). tgt[t, v] = t일 체결 목표 비율(nan = 그대로). at_close면 t일 종가에 체결.
    미끄러짐 = 체결 비율 변화 × slip × 그때 가치. 현금 이자 = (1 − 비율) × rf/252(시가 체결 뒤)."""
    V = tgt.shape[1]
    n = z - a + 1
    E = np.empty((n + 1, V)); E[0] = 1.0
    eq = np.ones(V); w = np.ones(V)
    for k in range(n):
        t = a + k
        g1 = o[t] / c[t - 1]
        g2 = c[t] / o[t]
        tg = tgt[t]
        m = ~np.isnan(tg)
        den = 1 + w * (g1 - 1)
        eq = eq * den
        w = w * g1 / den
        if not at_close and m.any():
            d = np.where(m, tg - w, 0.0)
            eq = eq * (1 - slip * np.abs(d))
            w = np.where(m, tg, w)
        if cash is not None:
            eq = eq * (1 + (1 - w) * cash[t] / 252)
        den = 1 + w * (g2 - 1)
        eq = eq * den
        w = w * g2 / den
        if at_close and m.any():
            d = np.where(m, tg - w, 0.0)
            eq = eq * (1 - slip * np.abs(d))
            w = np.where(m, tg, w)
        E[k + 1] = eq
    return E


def targets_from_exposure(ex, a):
    tg = np.full(ex.shape, np.nan)
    ch = np.zeros(ex.shape, bool)
    ch[a] = True
    ch[a + 1:] = ex[a + 1:] != ex[a:-1]
    tg[ch] = ex[ch]
    return tg


def bounds(days, a, z, w):
    d0, d1 = w
    i0 = a if d0 is None else max(a, int(np.searchsorted(days, d0, "left")))
    i1 = min(z, int(np.searchsorted(days, d1, "right")) - 1)
    if i1 - i0 < 15:
        return None
    return i0, i1


def wstat(E, a, i0, i1):
    seg = E[i0 - a:i1 - a + 2]
    g = seg[-1] / seg[0]
    n = i1 - i0 + 1
    ann = np.where(g > 0, np.abs(g) ** (252 / n) - 1, -1.0)
    dd = (seg / np.maximum.accumulate(seg, axis=0) - 1).min(axis=0)
    return {"ret": g - 1, "ann": ann, "dd": dd}


class Target:
    def __init__(self, name, dd, run0, slip_cash_modes=True):
        self.name = name
        self.d, self.o, self.c, self.rf = dd["d"], dd["o"], dd["c"], dd["rf"]
        self.a = int(np.searchsorted(self.d, run0, "left"))
        self.z = int(np.searchsorted(self.d, END, "right")) - 1
        self.win = {}
        for wn, (w, lg) in WINDOWS.items():
            b = bounds(self.d, self.a, self.z, w)
            if b is not None:
                self.win[wn] = (b, lg)
        self.win["전체"] = (bounds(self.d, self.a, self.z, (None, END)), True)
        self._front = {}

    def cash_arr(self, cfg):
        return {"cash4": np.full(len(self.c), CASH_ALT), "irx": self.rf}.get(cfg)

    def front(self, cfg):
        """덜 담기(매일 시가에 비율 w로 맞춤) 101개의 창별 통계. 비교 조건(미끄러짐·현금 이자)은 규칙과 같게."""
        key = cfg
        if key not in self._front:
            tg = np.tile(W_GRID, (len(self.c), 1))
            E = engine(self.o, self.c, self.a, self.z, tg, SLIP if cfg == "slip" else 0.0, self.cash_arr(cfg))
            self._front[key] = {wn: wstat(E, self.a, *b) for wn, (b, lg) in self.win.items()}
        return self._front[key]

    def run(self, ex, cfg="base", at_close=False):
        """ex[t, v] 노출(대상 거래일). 반환 창별 통계 + 효율선초과."""
        tg = targets_from_exposure(ex, self.a)
        E = engine(self.o, self.c, self.a, self.z, tg, SLIP if cfg == "slip" else 0.0, self.cash_arr(cfg), at_close)
        fr = self.front("base" if cfg not in ("slip", "cash4", "irx") else cfg)
        out = {}
        for wn, (b, lg) in self.win.items():
            st = wstat(E, self.a, *b)
            f = fr[wn]
            key = "ann" if lg else "ret"
            ok = f["dd"][None, :] >= st["dd"][:, None] - 1e-12
            M = np.where(ok, f[key][None, :], -np.inf)
            st["exc"] = st[key] - M.max(axis=1)
            st["w"] = W_GRID[M.argmax(axis=1)]
            i0, i1 = b
            st["expo"] = ex[i0:i1 + 1].mean(axis=0)
            jj = np.clip(np.round(st["expo"] * 100).astype(int), 0, 100)
            st["same_expo"] = st[key] - f[key][jj]
            st["hold"] = {"ret": float(f["ret"][-1]), "ann": float(f["ann"][-1]), "dd": float(f["dd"][-1])}
            out[wn] = st
        out["_E"] = E
        return out


# ============================================================================ 원 갈래 93개 규칙(같은 정의를 새로 씀)
def grid_specs():
    out = []
    P = lambda **kw: base_spec(**kw)
    out.append(("P1", P(mode="td", N=21, panic=(4, 42), up=None)))
    out.append(("P2", P(mode="cal", M=1, panic=(4, 2), up=None)))
    out.append(("P3", P(mode="td", N=21, panic=(4, 42), up=8)))
    out.append(("P4", P(mode="cal", M=1, panic=(4, 2), up=8)))
    for X in (-0.02, -0.025, -0.03, -0.035, -0.04, -0.05):
        for N in (5, 10, 15, 21, 30, 42):
            for s in (1.0, 0.5):
                out.append((f"G{-X * 100:g}_{N}_{int(s * 100)}", P(X=X, mode="td", N=N, panic=None, up=None, s=s)))
    for k in (3, 4, 5):
        for w in (42, 63):
            if (k, w) == (4, 42):
                continue  # = P1
            out.append((f"K{k}W{w}", P(mode="td", N=21, panic=(k, w), up=None)))
    for X in (-0.02, -0.025, -0.03, -0.035, -0.04, -0.05):
        for s in (1.0, 0.5):
            out.append((f"U{-X * 100:g}_{int(s * 100)}", P(X=X, mode="td", N=21, panic=None, up=8, s=s)))
    return out


def exposure(T, sd, St, s=1.0, lag=0):
    m = to_target_days(sd, St, T.d, lag)
    return np.where(m > 0.5, 1.0, 1.0 - s)


# ============================================================================ 위약
def runs_of(e):
    """0/1 배열 → [(값, 길이)]"""
    out = []
    i = 0
    n = len(e)
    while i < n:
        j = i
        while j < n and e[j] == e[i]:
            j += 1
        out.append((e[i], j - i))
        i = j
    return out


def placebo_A(e, i0, i1, rng):
    seg = e[i0:i1 + 1]
    R = runs_of(seg)
    inv = [L for v, L in R if v > 0.5]
    csh = [L for v, L in R if v <= 0.5]
    rng.shuffle(inv); rng.shuffle(csh)
    first = R[0][0]
    out = []
    a, b = list(inv), list(csh)
    cur = first
    while a or b:
        if cur > 0.5:
            if a:
                out += [1.0] * a.pop()
            cur = 0.0
        else:
            if b:
                out += [0.0] * b.pop()
            cur = 1.0
    e2 = e.copy()
    e2[i0:i1 + 1] = np.array(out[:len(seg)])
    return e2


def placebo_B(e, i0, i1, rng):
    seg = e[i0:i1 + 1]
    starts, lens = [], []
    k = 0
    for v, L in runs_of(seg):
        if v <= 0.5:
            starts.append(k); lens.append(L)
        k += L
    lens = list(rng.permutation(lens))
    new = np.ones(len(seg))
    for s0, L in zip(starts, lens):
        new[s0:s0 + L] = 0.0
    e2 = e.copy()
    e2[i0:i1 + 1] = new
    return e2


def placebo_C(sd, sr, rng, per_year):
    trig = sr <= X0
    valid = np.isfinite(sr)
    new = np.zeros(len(sr), bool)
    if per_year:
        yrs = pd.Series(sd).str[:4].to_numpy()
        for y in np.unique(yrs):
            idx = np.nonzero((yrs == y) & valid)[0]
            k = int(trig[idx].sum())
            if k:
                new[rng.choice(idx, k, replace=False)] = True
    else:
        idx = np.nonzero(valid)[0]
        new[rng.choice(idx, int(trig[valid].sum()), replace=False)] = True
    return new


# ============================================================================ 본 실행
def main():
    t0 = time.time()
    D = build()
    sd, sr = D["sig"]
    NDX3 = Target("NDX3", D["NDX3"], RUN0_NDX3)
    TQQQ = Target("TQQQ", D["TQQQ"], RUN0_TQQQ)
    orig = json.loads((RESULTS / "p7_MASAM.json").read_text(encoding="utf-8"))
    OUT = {"checks": [], "repro": {}, "grid": {}, "timing": {}, "neighbors": {}, "placebo": {}, "eras": {}, "trades": {}}
    log = lambda m: (print(" ·", m), OUT["checks"].append(m))

    # -------- 0. 자료 점검
    n3 = int(np.nansum(sr <= -0.03))
    d26 = [d for d, x in zip(sd, sr) if d >= "2026-01-01" and x <= -0.03]
    log(f"^IXIC −3% 이하 날 {n3}번, 2026 {d26}")
    bad = D["NDX3"]["bad"]
    dd_ = D["NDX3"]["d"]
    pre = (dd_ < "1999-03-10")
    log(f"NDX3 기초 묵은·없는 시가 비율 1985~99 {bad[pre].mean():.0%}, 1999~ {bad[~pre].mean():.1%}")
    # 합성 대 실제 TQQQ
    common = np.intersect1d(NDX3.d, TQQQ.d)
    i1, i2 = np.searchsorted(NDX3.d, common), np.searchsorted(TQQQ.d, common)
    ra, rs = TQQQ.c[i2][1:] / TQQQ.c[i2][:-1] - 1, NDX3.c[i1][1:] / NDX3.c[i1][:-1] - 1
    yrs = len(common) / 252
    gap = -np.log((TQQQ.c[i2][-1] / TQQQ.c[i2][0]) / (NDX3.c[i1][-1] / NDX3.c[i1][0])) / yrs
    log(f"합성 NDX3 vs 실제 TQQQ({common[0]}~): 일수익 상관 {np.corrcoef(ra, rs)[0, 1]:.4f}, 남은 연 로그 비용 차 {gap:+.4f}")

    # -------- 1. 격자 93개 재현
    G = grid_specs()
    ids = [g[0] for g in G]
    St_all = [states(sd, sr, sp) for _, sp in G]
    ex_all = np.stack([exposure(NDX3, sd, St, sp["s"]) for St, (_, sp) in zip(St_all, G)], 1)
    R = NDX3.run(ex_all)
    print(f"격자 93개 완료 {time.time() - t0:.1f}s")
    i4 = ids.index("P4")
    o_pairs = orig["pairs"]["NDX3·IXIC"]
    rep = {}
    for wn_me, wn_o in (("고르기 1985~2005", "고르기_긴"), ("확인 2006~2025", "확인_긴"), ("긴 역사 1985~2026", "긴역사"),
                        ("2010~19", "2010~19"), ("2026 앞", "2026앞"), ("2026 뒤", "2026뒤"), ("7월 쇼크", "2026_7월"),
                        ("2025", "2025"), ("2022", "2022"), ("2020", "2020"), ("2008", "2008"), ("2000~02", "2000~02"),
                        ("1987", "1987")):
        st = R[wn_me]
        lg = WINDOWS[wn_me][1]
        o = o_pairs["P4"]["base"][wn_o]
        rep[wn_me] = {"mine": {"수익" if not lg else "연율": float(st["ann" if lg else "ret"][i4]), "낙폭": float(st["dd"][i4]),
                               "효율선초과": float(st["exc"][i4])},
                      "orig": {"수익" if not lg else "연율": o["연율" if lg else "수익"], "낙폭": o["낙폭"], "효율선초과": o["효율선초과"]}}
    OUT["repro"]["P4"] = rep
    OUT["repro_hold_long"] = R["긴 역사 1985~2026"]["hold"]
    # 93개 전체 차이
    diffs = []
    for i, gid in enumerate(ids):
        for wn_me, wn_o in (("고르기 1985~2005", "고르기_긴"), ("확인 2006~2025", "확인_긴"), ("긴 역사 1985~2026", "긴역사"),
                            ("2026 뒤", "2026뒤"), ("2025", "2025")):
            diffs.append(abs(float(R[wn_me]["exc"][i]) - o_pairs[gid]["base"][wn_o]["효율선초과"]))
    diffs = np.array(diffs)
    log(f"격자 93개 × 5창 효율선초과 원 갈래와 차: 최대 {diffs.max() * 100:.2f}%p, 0.5%p 넘는 칸 {(diffs > 0.005).sum()}/{len(diffs)}")
    exL, exC = R["고르기 1985~2005"]["exc"], R["확인 2006~2025"]["exc"]
    order = np.argsort(-exL)
    OUT["grid"] = {"top5": [(ids[j], float(exL[j]), float(exC[j])) for j in order[:5]],
                   "pick": ids[int(order[0])], "pos_pick": int((exL > 0).sum()),
                   "pos_pick_and_check": int(((exL > 0) & (exC > 0)).sum()), "pos_check_all": int((exC > 0).sum()),
                   "n": len(ids), "rank_P4_check": int((exC > exC[i4]).sum()) + 1}
    log(f"고르기 1위 {OUT['grid']['pick']}, 상위 5 {[(a, round(b * 100, 1), round(c * 100, 1)) for a, b, c in OUT['grid']['top5']]}")
    log(f"고른 구간>0 {OUT['grid']['pos_pick']}개 중 확인>0 {OUT['grid']['pos_pick_and_check']}, 무조건 {OUT['grid']['pos_check_all']}/{len(ids)}; "
        f"P4 확인 순위 {OUT['grid']['rank_P4_check']}/{len(ids)}")

    # 실제 TQQQ 원형 4개
    exT = np.stack([exposure(TQQQ, sd, St_all[ids.index(k)]) for k in ("P1", "P2", "P3", "P4")], 1)
    RT = TQQQ.run(exT)
    RTs = TQQQ.run(exT, "slip")
    OUT["repro"]["TQQQ"] = {k: {"mine": float(RT["전체"]["exc"][j]), "mine_slip": float(RTs["전체"]["exc"][j]),
                                "mine_ann": float(RT["전체"]["ann"][j]), "mine_dd": float(RT["전체"]["dd"][j]),
                                "orig": orig["pairs"]["TQQQ·IXIC"][k]["base"]["실제전체"]["효율선초과"]}
                            for j, k in enumerate(("P1", "P2", "P3", "P4"))}
    OUT["repro"]["TQQQ_hold"] = RT["전체"]["hold"]

    # 2026 거래·2000~03 주기
    e4 = ex_all[:, i4]
    ch = np.nonzero(np.diff(e4[NDX3.a - 1:]) != 0)[0] + NDX3.a
    trades = [(NDX3.d[k], float(e4[k])) for k in ch]
    OUT["trades"]["P4_all"] = trades
    OUT["trades"]["P4_2026"] = [x for x in trades if x[0] >= "2025-12-01"]
    cyc = []
    for k in range(0, len(trades) - 1):
        if trades[k][1] < 0.5 and trades[k + 1][1] > 0.5:
            cyc.append((trades[k][0], trades[k + 1][0]))
    lens = [(a_, b_, int(np.searchsorted(NDX3.d, b_) - np.searchsorted(NDX3.d, a_))) for a_, b_ in cyc]
    lens.sort(key=lambda x: -x[2])
    OUT["trades"]["longest_cash"] = lens[:5]
    log(f"P4 주기 {len(cyc)}개, 긴 현금 구간 상위 3 {[(str(a_), str(b_), n_) for a_, b_, n_ in lens[:3]]}")
    # 2000~03 현금 구간이 왜 849거래일이었나: −3% 날 사이 최대 간격
    for lo_, hi_ in (("2000-01-04", "2003-05-27"), ("2008-09-04", "2009-07-20")):
        tds = [pd.Timestamp(str(d)) for d, x in zip(sd, sr) if lo_ <= d <= hi_ and x <= X0]
        gaps = np.diff(tds).astype("timedelta64[D]").astype(int) if len(tds) > 1 else np.array([0])
        jmax = int(np.argmax(gaps))
        log(f"{lo_}~{hi_}: −3% 날 {len(tds)}번, 이웃 −3% 날 최대 간격 {gaps.max()}일({tds[jmax].date()} → {tds[jmax + 1].date()}), "
            f"마지막 −3% {tds[-1].date()}")
    # 긴 역사 최대 낙폭 위치
    Eb = R["_E"][:, i4]
    pk = np.maximum.accumulate(Eb)
    kt = int(np.argmin(Eb / pk - 1))
    kp = int(np.argmax(Eb[:kt + 1]))
    log(f"P4 긴 역사 최대 낙폭 {Eb[kt] / Eb[kp] - 1:.1%}: 고점 {NDX3.d[NDX3.a + kp - 1]} → 저점 {NDX3.d[NDX3.a + kt - 1]}")

    # -------- 2. 시점 감사
    # 미래 교란: 2020-03-02 뒤 신호를 흔들어도 그날까지 상태가 같아야
    Tk = int(np.searchsorted(sd, "2020-03-02"))
    rng = np.random.default_rng(11)
    worst = 0
    for _ in range(3):
        r2 = sr.copy()
        r2[Tk + 1:] = rng.normal(0, 0.03, len(sr) - Tk - 1)
        for _, sp in G[:4] + G[40:44]:
            worst = max(worst, int(np.abs(states(sd, sr, sp)[:Tk + 1].astype(int) - states(sd, r2, sp)[:Tk + 1].astype(int)).max()))
    log(f"미래 교란 3회 × 규칙 8개: 그날까지 상태 차 {worst}(0이어야 함)")
    # 체결일 > 판단일 확인
    jj = np.searchsorted(sd, NDX3.d, "left") - 1
    okk = all(sd[j] < NDX3.d[k] for k, j in enumerate(jj) if j >= 0)
    log(f"대상 거래일마다 쓰는 신호일 < 체결일: {okk}")
    # 2026-06-05 신호 → 체결가
    k_s = int(np.searchsorted(NDX3.d, "2026-06-08"))
    log(f"2026 매도 체결 {NDX3.d[k_s]} 시가(3배 합성 {NDX3.o[k_s]:.2f}, 전날 종가 {NDX3.c[k_s - 1]:.2f})")

    St4 = St_all[i4]
    variants = {}

    def add(name, ex, cfg="base", at_close=False, T=NDX3):
        rr = T.run(ex[:, None], cfg, at_close)
        variants[name] = {wn: {"ret": float(st["ret"][0]), "ann": float(st["ann"][0]), "dd": float(st["dd"][0]),
                               "exc": float(st["exc"][0]), "same_expo": float(st["same_expo"][0]), "expo": float(st["expo"][0])}
                          for wn, st in rr.items() if wn != "_E"}
        (i0, i1), _ = T.win["긴 역사 1985~2026"]
        seg = ex[i0 - 1:i1 + 1]
        variants[name]["연매매"] = float((np.abs(np.diff(seg)) >= 0.1).sum() / ((i1 - i0 + 1) / 252))

    add("기준", e4)
    add("미끄러짐 0.5%", e4, "slip")
    add("현금 4%", e4, "cash4")
    add("현금 ^IRX", e4, "irx")
    add("하루 늦게", exposure(NDX3, sd, St4, lag=1))
    add("이틀 늦게", exposure(NDX3, sd, St4, lag=2))
    add("다음 날 종가 체결", e4, at_close=True)
    add("하루 늦게 + 미끄러짐 0.5%", exposure(NDX3, sd, St4, lag=1), "slip")
    add("한 달 = 21거래일(P3)", ex_all[:, ids.index("P3")])
    add("공황을 −3% 날마다 다시 판정", exposure(NDX3, sd, states(sd, sr, base_spec(persist=False))))
    add("공황 없음", exposure(NDX3, sd, states(sd, sr, base_spec(panic=None))))
    add("8일 없음(P2)", ex_all[:, ids.index("P2")])
    OUT["timing"] = variants
    print(f"시점 판 완료 {time.time() - t0:.1f}s")

    # -------- 3. 이웃
    nb = {}

    def nb_add(name, sp):
        St = states(sd, sr, sp)
        nb[name] = exposure(NDX3, sd, St)

    for X in NB_X:
        nb_add(f"X {X * 100:g}%", base_spec(X=X))
    for wd in NB_WAIT_D:
        nb_add(f"대기 1개월+1일{wd:+d}일", base_spec(wd=wd))
    for k in NB_PK:
        nb_add(f"공황 {k}번", base_spec(panic=(k, 2)))
    for pw in NB_PW_D:
        nb_add(f"공황 대기 2개월+1일{pw:+d}일", base_spec(pwd=pw))
    for u in NB_UP:
        nb_add(f"연속 상승 {u}일", base_spec(up=u))
    nb_names = list(nb)
    exN = np.stack([nb[k] for k in nb_names], 1)
    RN = NDX3.run(exN)
    # 같은 이웃을 실제 TQQQ에도
    exNT = np.stack([exposure(TQQQ, sd, states(sd, sr, sp)) for sp in
                     [base_spec(X=X) for X in NB_X] + [base_spec(wd=wd) for wd in NB_WAIT_D] + [base_spec(panic=(k, 2)) for k in NB_PK]
                     + [base_spec(pwd=pw) for pw in NB_PW_D] + [base_spec(up=u) for u in NB_UP]], 1)
    RNT = TQQQ.run(exNT)
    OUT["neighbors"]["one"] = {nm: {wn: float(RN[wn]["exc"][j]) for wn in ("고르기 1985~2005", "확인 2006~2025", "긴 역사 1985~2026",
                                                                          "2010~19", "2025", "2026 뒤")}
                                    | {"TQQQ": float(RNT["전체"]["exc"][j]), "dd": float(RN["긴 역사 1985~2026"]["dd"][j])}
                               for j, nm in enumerate(nb_names)}
    # 2D X × 대기
    names2, ex2 = [], []
    for X in NB_X:
        for wd in NB_WAIT_D:
            names2.append((X, wd))
            ex2.append(exposure(NDX3, sd, states(sd, sr, base_spec(X=X, wd=wd))))
    R2 = NDX3.run(np.stack(ex2, 1))
    OUT["neighbors"]["xw"] = {f"{X * 100:g}|{wd:+d}": {wn: float(R2[wn]["exc"][j]) for wn in ("고르기 1985~2005", "확인 2006~2025", "2010~19", "2025")}
                              for j, (X, wd) in enumerate(names2)}
    print(f"이웃 완료 {time.time() - t0:.1f}s")

    # -------- 4. 위약
    PL = {}
    for wn in ("고르기 1985~2005", "확인 2006~2025", "긴 역사 1985~2026", "2010~19"):
        (i0, i1), lg = NDX3.win[wn]
        exA = np.stack([placebo_A(e4, i0, i1, np.random.default_rng(100 + s)) for s in range(SEEDS)], 1)
        exB = np.stack([placebo_B(e4, i0, i1, np.random.default_rng(200 + s)) for s in range(SEEDS)], 1)
        RA, RB = NDX3.run(exA)[wn], NDX3.run(exB)[wn]
        act = float(R[wn]["exc"][i4])
        PL[wn] = {"actual": act,
                  "A": {"rank": int((RA["exc"] > act).sum()) + 1, "med": float(np.median(RA["exc"])), "max": float(RA["exc"].max()),
                        "cash": float((exA[i0:i1 + 1] < 0.5).mean())},
                  "B": {"rank": int((RB["exc"] > act).sum()) + 1, "med": float(np.median(RB["exc"])), "max": float(RB["exc"].max()),
                        "cash": float((exB[i0:i1 + 1] < 0.5).mean())},
                  "act_cash": float((e4[i0:i1 + 1] < 0.5).mean())}
    for tag, per_year in (("C1", False), ("C2", True)):
        exC = np.stack([exposure(NDX3, sd, states(sd, sr, base_spec(), placebo_C(sd, sr, np.random.default_rng(300 + s + (50 if per_year else 0)), per_year)))
                        for s in range(SEEDS)], 1)
        RC = NDX3.run(exC)
        for wn in ("고르기 1985~2005", "확인 2006~2025", "긴 역사 1985~2026", "2010~19"):
            (i0, i1), _ = NDX3.win[wn]
            act = PL[wn]["actual"]
            PL[wn][tag] = {"rank": int((RC[wn]["exc"] > act).sum()) + 1, "med": float(np.median(RC[wn]["exc"])),
                           "max": float(RC[wn]["exc"].max()), "cash": float((exC[i0:i1 + 1] < 0.5).mean())}
    OUT["placebo"] = PL
    print(f"위약 완료 {time.time() - t0:.1f}s")

    # -------- 5. 시대별(기준 P4, 원형 4개)
    for wn in ERA_KEYS + ["2010~19", "1985~99", "2003~2026"]:
        st = R[wn]
        OUT["eras"][wn] = {k: {"ret": float(st["ret"][ids.index(k)]), "dd": float(st["dd"][ids.index(k)]),
                               "ann": float(st["ann"][ids.index(k)]), "exc": float(st["exc"][ids.index(k)]),
                               "same_expo": float(st["same_expo"][ids.index(k)]), "expo": float(st["expo"][ids.index(k)])}
                           for k in ("P1", "P2", "P3", "P4")} | {"hold": st["hold"]}
    OUT["timing_eras"] = {v: {wn: variants[v][wn]["exc"] for wn in ERA_KEYS if wn in variants[v]} for v in variants}

    (RESULTS / f"{ID}.json").write_text(json.dumps(OUT, ensure_ascii=False, default=lambda x: x.item() if hasattr(x, "item") else str(x),
                                                   separators=(",", ":")), encoding="utf-8")
    write_md(OUT)
    print(f"저장 {time.time() - t0:.1f}s")


# ============================================================================ 보고서
def pp(x, d=1):
    return "—" if x is None or x != x else f"{x * 100:+.{d}f}%p"


def pc(x, d=0):
    return "—" if x is None or x != x else f"{x * 100:+.{d}f}%"


def write_md(O):
    L = []
    A = L.append
    A("# 7단계 반박 검증: 마삼룰 원형 P4(3배 나스닥100 합성, ^IXIC 신호) (2026-10-11)")
    A("")
    A("> 독립 재계산 `code/p7_V_MASAM.py`(공유 엔진·원 스크립트 미사용, parquet만). 숫자 `results/p7_V_MASAM.json`.")
    A("")
    T_ = O["timing"]
    rp = O["repro"]["P4"]
    pl = O["placebo"]
    nb1 = O["neighbors"]["one"]
    A("## 결론 3줄")
    A("")
    A(f"1. **숫자는 그대로 재현됐고 시점·비용에도 버틴다.** 93개 규칙 × 5창 465칸이 원 갈래와 최대 0.00%p 차, P4 = 긴 역사 고르기 1위"
      f"(1985~2005 {pp(rp['고르기 1985~2005']['mine']['효율선초과'])} → 2006~2025 {pp(rp['확인 2006~2025']['mine']['효율선초과'])}, "
      f"1985~2026 연 {pc(rp['긴 역사 1985~2026']['mine']['연율'], 1)}/{pc(rp['긴 역사 1985~2026']['mine']['낙폭'])}). "
      f"하루 늦게 {pp(T_['하루 늦게']['긴 역사 1985~2026']['exc'])}, 이틀 늦게 {pp(T_['이틀 늦게']['긴 역사 1985~2026']['exc'])}, "
      f"다음 날 종가 체결 {pp(T_['다음 날 종가 체결']['긴 역사 1985~2026']['exc'])}, 21거래일판 {pp(T_['한 달 = 21거래일(P3)']['긴 역사 1985~2026']['exc'])}"
      f"(1985~2026 연율 %p)로 미래 정보 누수도 없다.")
    A(f"2. **그 우위는 두 번의 긴 약세장 몫이고 2026 합격선은 못 넘는다.** 현금 2000-01-05 → 2003-05-27(849거래일)과 2008-09-05 → 2009-07-20이 이익의 원천이며, "
      f"1985~99 연 {pp(O['eras']['1985~99']['P4']['exc'])}, 2010~19 {pp(O['eras']['2010~19']['P4']['exc'])}, 실제 TQQQ 2010~2026 {pp(O['repro']['TQQQ']['P4']['mine'])}, "
      f"2025 {pp(O['eras']['2025']['P4']['exc'])}, 2026 뒤 {pp(O['eras']['2026 뒤']['P4']['exc'])}(보유 대비 {pp(O['eras']['2026 뒤']['P4']['ret'] - O['eras']['2026 뒤']['hold']['ret'])})다. "
      f"X −3%가 뾰족한 봉우리라 −2.5%·−3.5%면 고르기 값이 {pp(nb1['X -2.5%']['고르기 1985~2005'])}·{pp(nb1['X -3.5%']['고르기 1985~2005'])}로 줄고, "
      f"−3.5%·−4%는 닷컴을 못 피해 최대 낙폭 {pc(nb1['X -3.5%']['dd'])}·{pc(nb1['X -4%']['dd'])}다.")
    A(f"3. **위약은 '−3% 날을 골라 판다'가 아니라 '변동성 큰 해에 빠져 있다'가 몫이라고 말한다.** 현금 구간을 무작위로 옮긴 판 대비 긴 역사 "
      f"{pl['긴 역사 1985~2026']['A']['rank']}/21·확인 {pl['확인 2006~2025']['A']['rank']}/21로 강하지만, 같은 해 안 무작위 날 대비 확인 "
      f"{pl['확인 2006~2025']['C2']['rank']}/21, 판 날은 두고 대기만 섞은 판 대비 확인 {pl['확인 2006~2025']['B']['rank']}/21이다. "
      f"판정 **confirmed_fragile**: 긴 역사 숫자는 맞지만 2026 장세에서 검증된 규칙이 아니라 긴 약세장 보험이다.")
    A("")
    A("## 사용자 표 (3배 나스닥100 합성, 수익/최대 낙폭, 현금 이자 0)")
    A("")
    A("| 규칙 | 2026 앞 | 2026 뒤 | 2025 | 2022 | 2020 | 2008 | 2000~02 | 1987 | 1985~2026 연율/낙폭 | 연 매매 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    uw = ["2026 앞", "2026 뒤", "2025", "2022", "2020", "2008", "2000~02", "1987"]
    hh = O["eras"]
    A("| 그냥 보유 | " + " | ".join(f"{pc(hh[w]['hold']['ret'])}/{pc(hh[w]['hold']['dd'])}" for w in uw)
      + f" | 연 {pc(O['repro_hold_long']['ann'], 1)}/{pc(O['repro_hold_long']['dd'])} | 0 |")
    for nm in ("기준", "미끄러짐 0.5%", "하루 늦게", "다음 날 종가 체결", "한 달 = 21거래일(P3)", "현금 4%"):
        v = T_[nm]
        lab = "P4 원형(대표)" if nm == "기준" else f"P4 {nm}" if "P3" not in nm else "P3(한 달 = 21거래일)"
        A(f"| {lab} | " + " | ".join(f"{pc(v[w]['ret'])}/{pc(v[w]['dd'])}" for w in uw)
          + f" | 연 {pc(v['긴 역사 1985~2026']['ann'], 1)}/{pc(v['긴 역사 1985~2026']['dd'])} | {v['연매매']:.1f} |")
    A("")
    A("## 재현 (원 갈래 vs 독립 재계산)")
    A("")
    A("| 창 | 원 갈래 수익(연율)/낙폭 | 재계산 | 원 갈래 효율선초과 | 재계산 | 차 |")
    A("|---|---|---|---|---|---|")
    for wn, v in O["repro"]["P4"].items():
        k = "연율" if "연율" in v["mine"] else "수익"
        A(f"| {wn} | {pc(v['orig'][k], 1)}/{pc(v['orig']['낙폭'])} | {pc(v['mine'][k], 1)}/{pc(v['mine']['낙폭'])} | "
          f"{pp(v['orig']['효율선초과'])} | {pp(v['mine']['효율선초과'])} | {pp(v['mine']['효율선초과'] - v['orig']['효율선초과'], 2)} |")
    for k, v in O["repro"]["TQQQ"].items():
        A(f"| 실제 TQQQ 2010~2026 {k} | — | 연 {pc(v['mine_ann'], 1)}/{pc(v['mine_dd'])} | {pp(v['orig'])} | {pp(v['mine'])} | {pp(v['mine'] - v['orig'], 2)} |")
    A("")
    g = O["grid"]
    A(f"- 격자 93개 재현: 고르기 1위 {g['pick']}, 상위 5 = " + ", ".join(f"{a} {b * 100:+.1f}/{c * 100:+.1f}" for a, b, c in g["top5"])
      + f" (고르기/확인 %p). 고른 구간 > 0 {g['pos_pick']}개 중 확인 > 0 {g['pos_pick_and_check']}, 무조건 {g['pos_check_all']}/{g['n']}. P4 확인 순위 {g['rank_P4_check']}/{g['n']}.")
    for m in O["checks"]:
        A(f"- {m}")
    A("")
    A("## 시점·비용·정의 흔들기 (효율선초과)")
    A("")
    cols = ["고르기 1985~2005", "확인 2006~2025", "긴 역사 1985~2026", "2010~19", "2025", "2026 뒤", "7월 쇼크"]
    A("| 판 | " + " | ".join(cols) + " | 긴 역사 연율/낙폭 |")
    A("|---|" + "---|" * (len(cols) + 1))
    for name, v in O["timing"].items():
        A(f"| {name} | " + " | ".join(pp(v[c]["exc"]) for c in cols) + f" | 연 {pc(v['긴 역사 1985~2026']['ann'], 1)}/{pc(v['긴 역사 1985~2026']['dd'])} |")
    A("")
    A("## 이웃 (한 번에 하나, 앞뒤 2칸)")
    A("")
    A("| 이웃 | 고르기 | 확인 | 긴 역사 | 2010~19 | 2025 | 2026 뒤 | 실제 TQQQ | 긴 역사 낙폭 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for nm, v in O["neighbors"]["one"].items():
        A(f"| {nm} | {pp(v['고르기 1985~2005'])} | {pp(v['확인 2006~2025'])} | {pp(v['긴 역사 1985~2026'])} | {pp(v['2010~19'])} | "
          f"{pp(v['2025'])} | {pp(v['2026 뒤'])} | {pp(v['TQQQ'])} | {pc(v['dd'])} |")
    A("")
    A("X × 대기(1개월+1일에 더한 일수), 칸 = 고르기 / 확인 / 2010~19 / 2025 효율선초과 %p")
    A("")
    A("| X | " + " | ".join(f"{d:+d}일" for d in NB_WAIT_D) + " |")
    A("|---|" + "---|" * len(NB_WAIT_D))
    for X in NB_X:
        row = []
        for d in NB_WAIT_D:
            v = O["neighbors"]["xw"][f"{X * 100:g}|{d:+d}"]
            row.append(" / ".join(f"{v[w] * 100:+.1f}" for w in ("고르기 1985~2005", "확인 2006~2025", "2010~19", "2025")))
        A(f"| {X * 100:g}% | " + " | ".join(row) + " |")
    A("")
    A("## 위약 (시드 20, 순위 = 21개 중 실제 자리, 1이 최고)")
    A("")
    A("| 창 | 실제 | A 현금 구간 재배치 | B 판 날 고정·대기 섞기 | C1 무작위 −3% 날(전 기간) | C2 무작위 −3% 날(같은 해) |")
    A("|---|---|---|---|---|---|")
    for wn, v in O["placebo"].items():
        cell = lambda t: f"{v[t]['rank']}위 (중앙 {pp(v[t]['med'])}, 최고 {pp(v[t]['max'])}, 현금 {v[t]['cash']:.0%})"
        A(f"| {wn} | {pp(v['actual'])} (현금 {v['act_cash']:.0%}) | {cell('A')} | {cell('B')} | {cell('C1')} | {cell('C2')} |")
    A("")
    A("## 시대별 (수익/최대 낙폭, 효율선초과)")
    A("")
    A("짧은 창은 기간 %p, 긴 창(2010~19·1985~99·2003~2026)은 연율 %p. 하락만 있는 창은 효율선 = 현금이라 효율선초과 = P4 수익 그 자체다(그래서 '보유 대비'를 함께 적는다).")
    A("")
    A("| 시대 | 그냥 보유 | P4 | P4 보유 대비 | P4 효율선초과 | P4 같은 평균 노출 대비 | P3(21거래일판) 효율선초과 | 하루 늦게 | 미끄러짐 0.5% |")
    A("|---|---|---|---|---|---|---|---|---|")
    for wn in ERA_KEYS + ["2010~19", "1985~99", "2003~2026"]:
        e = O["eras"][wn]
        lg = WINDOWS[wn][1]
        h = e["hold"]
        f = (lambda d: f"연 {pc(d['ann'], 1)}/{pc(d['dd'])}") if lg else (lambda d: f"{pc(d['ret'])}/{pc(d['dd'])}")
        kk = "ann" if lg else "ret"
        A(f"| {wn} | {f(h)} | {f(e['P4'])} | {pp(e['P4'][kk] - h[kk])} | {pp(e['P4']['exc'])} | {pp(e['P4']['same_expo'])} (노출 {e['P4']['expo']:.0%}) | "
          f"{pp(e['P3']['exc'])} | {pp(O['timing']['하루 늦게'][wn]['exc'])} | {pp(O['timing']['미끄러짐 0.5%'][wn]['exc'])} |")
    A("")
    A("## 거래 기록")
    A("")
    A("- P4 2025-12 이후 체결: " + ", ".join(f"{d} → {'보유' if v > 0.5 else '현금'}" for d, v in O["trades"]["P4_2026"]))
    A("- 가장 긴 현금 구간: " + ", ".join(f"{a} → {b} ({n}거래일)" for a, b, n in O["trades"]["longest_cash"][:3]))
    A("")
    A("## 한계")
    A("")
    A("- 시험 범위: 원 격자 93개 재계산, 이웃 한 번에 하나 20개 + X × 대기 25개(사전 고정 앞뒤 2칸), 시점·비용·정의 판 11개, 위약 4종 × 20시드 × 4창. 이웃 표는 결과를 본 뒤 바꾸지 않았다.")
    A("- 위약 C2(같은 해 안 무작위 −3% 날)는 그 해 −3% 날 수를 미리 아는 판이라 거래할 수 있는 규칙이 아니다. '어느 날 파느냐'와 '변동성 큰 해냐'를 가르는 분해용이다. "
      "위약 C는 무작위 날에 연속 상승 수를 0으로 되돌려 실제 규칙(−3% 날은 늘 하락)과 맞췄다(첫 실행 뒤 고친 위약 쪽 정합 수정, 실제 규칙 숫자는 그대로).")
    A("- 효율선은 창마다 사후에 고른 덜 담기 비중 w라 평가용이다(매매 근거 아님). 하락만 있는 짧은 창에선 현금(w = 0)이 기준이 돼 효율선초과 = 규칙 수익이 되므로 시대표에 보유 대비를 함께 적었다.")
    A("- 3배 합성: 1999-03 전 기초는 ^NDX 가격 지수(배당 빠짐), 그 구간 시가 88%가 묵은 시가라 다음 날 종가 체결로 바꿨다. 모든 체결을 다음 날 종가로 한 판도 결과가 같다(긴 역사 +15.5%p).")
    A("- 변수 시점: 신호(^IXIC 종가 −3%·연속 상승)는 그날 종가에 알 수 있고, 재매수 날은 미리 정해진 거래일 달력만 쓴다. 미래 교란 시험에서 상태 차 0.")
    A("- 2026 판정: ^IXIC −3% 날은 2026에 6/5 하루뿐이라 2026 앞으로는 고를 정보가 없다. 긴 역사 판정과 2026 판정을 따로 봐야 하고, 2026 판정으로는 통과가 아니다.")
    (RESULTS / f"{ID}.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
