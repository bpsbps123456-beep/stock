"""8단계 반박 검증 p8_V_FGDIRECT (2026-10-12): 갈래 FGDIRECT 대표 후보 'F&G ≤25인 날마다 사서 63거래일 보유'(E4_25·H63)를 깨려는 독립 재계산.

독립성: p8_FGDIRECT.py·trade_lab·stock_lab·harness·lev5를 import하지 않는다. 가격은 parquet·json을 직접 읽고(배당 수정 계수 = Adj Close/Close),
  F&G는 fg_data.fg()(자료 읽기)만 쓴다. 재구성 F&G는 갈래 문서의 정의대로 이 파일에서 다시 만들고 갈래 캐시(p8_FGDIRECT_recon.parquet)와 대조한다.
정의(재현 대상): 그날 종가 F&G ≤ T → 다음 거래일 시가 매수(그 구간 매일, 묶지 않음) → H거래일 뒤 시가 매도.
  비교 기준 = 같은 창 안 모든 거래일을 결정일로 둔 '아무 날 매수'(같은 H). 창 = (결정일 시작, 결정일 끝, 청산 한도: 매도일이 이 날짜 이하인 거래만).
  추가 −20% = 보유 중 종가 최저 또는 청산 시가가 진입가 × 0.8 이하. 재구성 트랙은 옛 지수라 다음 날 종가 체결·종가 청산.
사전 고정(아래 상수, 확인 구간을 본 뒤 바꾸지 않는다): 이웃 격자, 시대 창, 위약 수, 가격 대조 규칙, 군집 간격.
출력: results/p8_V_FGDIRECT.md·json. 실행: PYTHONIOENCODING=utf-8 py p8_V_FGDIRECT.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

import fg_data  # 자료 읽기만

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"
RESULTS = HERE.parent / "results"
ID = "p8_V_FGDIRECT"

# ============================================================================ 사전 고정 상수
T0, H0 = 25, 63
T_NEI = (15, 20, 25, 30, 35)          # 앞뒤 2칸(5점 간격)
H_NEI = (21, 42, 63, 84, 126)         # 앞뒤 2칸(21거래일 간격)
T_FINE = (23, 24, 25, 26, 27)         # 정수 문턱 민감도
N_PLA, MIN_SHIFT = 20, 63             # 원형 밀기 위약 20판(63일 이상)
N_PLA_BIG = 200                       # 참고용 큰 위약(순위 해상도)
N_RND = 20                            # 같은 수 무작위 날짜 위약
SEED = 20261012
SLIP = 0.005
DD = 0.20
CL_GAP = 21                           # 사건일 사이가 21거래일 넘게 벌어지면 새 군집
N_BOOT = 2000
END = "2026-10-09"
FG_START = "2011-01-03"
FEE3, EXTRA_NDX3 = 0.0095, 0.013
TARGETS = ("QQQ", "SPY", "TQQQ", "SOXL")

WIN = {"sel": ("2011-01-01", "2018-12-31", "2018-12-31"), "chk": ("2019-01-01", "2025-12-31", END),
       "2025": ("2025-01-01", "2025-12-31", END), "2026앞x": ("2026-01-02", "2026-05-29", END),
       "2026뒤": ("2026-06-01", END, END), "AI쇼크": ("2026-06-22", "2026-07-29", END), "2026전체": ("2026-01-02", END, END)}
WIN_HOLD = {"2025": ("2025-01-01", "2025-12-31", "2025-12-31")}
WIN_R = {"rsel": ("1991-04-01", "2010-12-31", "2010-12-31"), "rchk": ("2011-01-01", "2025-12-31", END)}
ERAS = {"2011": ("2011-01-01", "2011-12-31"), "2015~16": ("2015-01-01", "2016-12-31"), "2018 4분기": ("2018-10-01", "2018-12-31"),
        "2020": ("2020-01-01", "2020-12-31"), "2022": ("2022-01-01", "2022-12-31"), "2025": ("2025-01-01", "2025-12-31"),
        "2026 앞": ("2026-01-02", "2026-05-29"), "2026 뒤": ("2026-06-01", END), "7월 쇼크": ("2026-06-22", "2026-07-29")}
ERAS_R = {"1990": ("1990-01-01", "1990-12-31"), "2000~02": ("2000-03-01", "2002-10-31"), "2008": ("2007-10-01", "2009-03-31"),
          "1991~2010": ("1991-04-01", "2010-12-31"), "2011~2025": ("2011-01-01", "2025-12-31")}


# ============================================================================ 자료
P7 = pd.read_parquet(CACHE / "p7_idx.parquet")
P5 = pd.read_parquet(CACHE / "p5_lev_long.parquet")


def load_px(df, sym):
    g = df[df.sym == sym].dropna(subset=["Close"]).copy()
    g["date"] = g["date"].astype(str).str[:10]
    g = g[g.Close > 0].sort_values("date").drop_duplicates("date", keep="last")
    k = (g["Adj Close"] / g["Close"]).fillna(1.0).to_numpy(float)
    c = g["Close"].to_numpy(float) * k
    o = g["Open"].to_numpy(float)
    o = np.where(np.isfinite(o) & (o > 0), o * k, c)
    return g["date"].to_numpy(str), o, c


class Asset:
    def __init__(self, name, days, o, c, start, mode="open"):
        self.name = name
        self.days = np.asarray(days, dtype=str)
        self.c = np.asarray(c, float)
        self.px = np.asarray(o if mode == "open" else c, float)
        self.n = len(self.days)
        self.j0 = int(np.searchsorted(self.days, start))
        self._mn = {}

    def last(self, cap):
        return int(np.searchsorted(self.days, cap, side="right")) - 1

    def ent(self, dec, a, z, cap, H, delay=0):
        dec = np.asarray(dec, dtype=str)
        d = dec[(dec >= a) & (dec <= z)]
        e = np.searchsorted(self.days, d, side="right") + delay
        return e[(e < self.n) & (e + H <= self.last(cap))]

    def base_ent(self, a, z, cap, H):
        j0 = max(int(np.searchsorted(self.days, a)), self.j0)
        j1 = int(np.searchsorted(self.days, z, side="right")) - 1
        e = np.arange(j0 + 1, j1 + 2)
        return e[(e < self.n) & (e + H <= self.last(cap))]

    def r(self, e, H, slip=0.0):
        return self.px[e + H] * (1 - slip) / (self.px[e] * (1 + slip)) - 1

    def hit(self, e, H):
        if H not in self._mn:
            m = np.full(self.n, np.nan)
            m[: self.n - H + 1] = sliding_window_view(self.c, H).min(1)
            self._mn[H] = m
        mn = np.fmin(self._mn[H][e], self.px[e + H])
        return (mn <= (1 - DD) * self.px[e]).astype(float)


def n_ind(e, H):
    k, endp = 0, -1
    for x in np.sort(e):
        if x > endp:
            k += 1
            endp = x + H
    return k


def cell(A: Asset, dec, H, win, delay=0, slip_ev=0.0, slip_base=0.0):
    a, z, cap = win
    e = A.ent(dec, a, z, cap, H, delay)
    b = A.base_ent(a, z, cap, H)
    br = A.r(b, H, slip_base)
    out = {"n": int(len(e)), "base": float(br.mean()) if len(b) else np.nan, "base_n": int(len(b)),
           "base_hit": float(A.hit(b, H).mean()) if len(b) else np.nan, "base_win": float((br > 0).mean()) if len(b) else np.nan}
    if not len(e):
        return out
    r = A.r(e, H, slip_ev)
    out.update({"n_ind": n_ind(e, H), "mean": float(r.mean()), "ex": float(r.mean() - out["base"]), "win": float((r > 0).mean()),
                "hit": float(A.hit(e, H).mean()), "first": str(A.days[e.min() - 1 - delay]), "last": str(A.days[e.max() - 1 - delay])})
    return out


def clusters(A: Asset, dec, H, win):
    """사건 진입일을 CL_GAP 간격으로 묶은 군집별 초과(창 기준값 대비)."""
    a, z, cap = win
    e = np.sort(A.ent(dec, a, z, cap, H))
    base = A.r(A.base_ent(a, z, cap, H), H).mean()
    if not len(e):
        return []
    groups, cur = [], [e[0]]
    for x in e[1:]:
        if x - cur[-1] > CL_GAP:
            groups.append(cur)
            cur = [x]
        else:
            cur.append(x)
    groups.append(cur)
    out = []
    for g in groups:
        g = np.asarray(g)
        rr = A.r(g, H)
        out.append({"start": str(A.days[g[0] - 1]), "end": str(A.days[g[-1] - 1]), "n": int(len(g)), "ex": float(rr.mean() - base),
                    "sum_ex": float((rr - base).sum())})
    return out


def rank_of(x, arr):
    a = np.asarray([y for y in arr if y is not None and np.isfinite(y)])
    if x is None or not np.isfinite(x):
        return None
    return [int((a > x).sum()) + 1, int(len(a)) + 1]


def le(dates, vals, T):
    return dates[vals <= T]


# ============================================================================ 재구성 F&G(갈래 정의대로 독립 구현)
def pct_rank(a, win=504, mn=252):
    out = np.full(len(a), np.nan)
    for i in range(len(a)):
        if not np.isfinite(a[i]):
            continue
        w = a[max(0, i - win + 1): i + 1]
        w = w[np.isfinite(w)]
        if len(w) < mn:
            continue
        out[i] = 100.0 * ((w < a[i]).sum() + 0.5 * ((w == a[i]).sum() - 1)) / (len(w) - 1)
    return out


def build_recon(real: pd.Series) -> pd.Series:
    d, _, c = load_px(P7, "^GSPC")
    spx = pd.Series(c, index=d)
    d, _, c = load_px(P7, "^VIX")
    vix = pd.Series(c, index=d)
    yx = pd.read_parquet(CACHE / "p8_FGDIRECT_yf.parquet")
    yx.index = pd.Index(yx.index.astype(str))
    idx = spx.index[spx.index >= "1986-06-01"]
    spx, vix = spx.reindex(idx), vix.reindex(idx)

    def al(s):
        return s.reindex(sorted(set(s.dropna().index) | set(idx))).ffill().reindex(idx)
    hy, lt, tnx = al(yx["VWEHX"]).shift(1), al(yx["VUSTX"]).shift(1), al(yx["^TNX"]) / 100.0
    comp = pd.DataFrame(index=idx)
    comp["momentum"] = spx / spx.rolling(125, min_periods=125).mean() - 1
    hi, lo = spx.rolling(252, min_periods=252).max(), spx.rolling(252, min_periods=252).min()
    comp["strength"] = (spx - lo) / (hi - lo)
    comp["vix"] = -(vix / vix.rolling(50, min_periods=50).mean() - 1)
    comp["junk"] = (hy / hy.shift(20) - 1) - (-5.0 * (tnx - tnx.shift(20)) + tnx.shift(20) * 20 / 252)
    comp["safe_haven"] = (spx / spx.shift(20) - 1) - (lt / lt.shift(20) - 1)
    rk = pd.DataFrame({k: pct_rank(comp[k].to_numpy(float)) for k in comp.columns}, index=idx)
    raw = rk.mean(axis=1, skipna=False).dropna()
    ov = [x for x in raw.index if x in real.index and "2011-01-01" <= x <= "2018-12-31"]
    rq, fq = np.sort(raw.reindex(ov).to_numpy()), np.sort(real.reindex(ov).to_numpy())
    qq = np.searchsorted(rq, raw.to_numpy(), side="right") / len(rq)
    return pd.Series(np.quantile(fq, np.clip(qq, 0, 1)), index=raw.index, name="recon")


def synth3(days, c, irx_days, irx):
    rf = pd.Series(irx, index=irx_days).reindex(sorted(set(irx_days) | set(days))).ffill().reindex(days).fillna(0).to_numpy() / 100
    v = np.empty(len(c))
    v[0] = 100.0
    for t in range(1, len(c)):
        rr = 3 * (c[t] / c[t - 1] - 1) - 2 * rf[t] / 252 - (FEE3 + EXTRA_NDX3) / 252
        v[t] = v[t - 1] * max(1 + rr, 0.001)
    return v


# ============================================================================ 실행
def main():
    t0 = time.time()
    J = {"id": ID, "const": {"T0": T0, "H0": H0, "T_NEI": T_NEI, "H_NEI": H_NEI, "T_FINE": T_FINE, "N_PLA": N_PLA,
                             "N_PLA_BIG": N_PLA_BIG, "N_RND": N_RND, "SLIP": SLIP, "CL_GAP": CL_GAP}}
    fg = fg_data.fg()
    fd, fv = fg.index.to_numpy(str), fg.to_numpy(float)
    # 원자료 직접 대조(fg_data 경로 확인)
    arc = pd.read_csv(CACHE / "fg_archive.csv")
    arc = pd.Series(arc["Fear Greed"].astype(float).to_numpy(), index=arc["Date"].astype(str))
    J["fg_check"] = {"n": int(len(fg)), "start": fd[0], "end": fd[-1], "le25_total": int((fv <= 25).sum()),
                     "archive_vs_fg_2011_2020_maxdiff": float((arc[arc.index < "2021-02-01"] - fg.reindex(arc[arc.index < "2021-02-01"].index)).abs().max())}

    # ---------------------------------------------------------------- 대상
    AS = {}
    for s in ("QQQ", "SPY"):
        AS[s] = Asset(s, *load_px(P7, s), FG_START)
    for s in ("TQQQ", "SOXL"):
        AS[s] = Asset(s, *load_px(P5, s), FG_START)
    b26 = json.loads((CACHE / "base_2026.json").read_text(encoding="utf-8"))
    bh = json.loads((CACHE / "base_hold.json").read_text(encoding="utf-8"))
    AS["7종목26"] = Asset("7종목26", b26["days"], b26["open"], b26["close"], "2026-01-02")
    AS["7종목hold"] = Asset("7종목hold", bh["days"], bh["open"], bh["close"], "2023-05-22")
    dec0 = le(fd, fv, T0)

    # ---------------------------------------------------------------- 1) 재현
    rep = {}
    for s in TARGETS:
        A = AS[s]
        rep[s] = {w: cell(A, dec0, H0, win) for w, win in WIN.items()}
        rep[s]["H21_2026뒤"] = cell(A, dec0, 21, WIN["2026뒤"])
        rep[s]["H21_AI쇼크"] = cell(A, dec0, 21, WIN["AI쇼크"])
    rep["7종목26"] = {w: cell(AS["7종목26"], dec0, H0, WIN[w]) for w in ("2026전체", "2026앞x", "2026뒤", "AI쇼크")}
    rep["7종목26"]["H21_2026뒤"] = cell(AS["7종목26"], dec0, 21, WIN["2026뒤"])
    rep["7종목hold"] = {"2025": cell(AS["7종목hold"], dec0, H0, WIN_HOLD["2025"])}
    J["reproduce"] = rep
    print("재현", f"{time.time() - t0:.0f}s")

    # ---------------------------------------------------------------- 재구성 트랙
    recon = build_recon(fg)
    their = pd.read_parquet(CACHE / "p8_FGDIRECT_recon.parquet")["recon"]
    their.index = pd.Index(their.index.astype(str))
    com = recon.index.intersection(their.index)
    ovr = sorted(set(recon.index) & set(fg.index))
    J["recon_check"] = {"start": recon.index[0], "n": int(len(recon)), "max_abs_diff_vs_branch": float((recon[com] - their[com]).abs().max()),
                        "corr_real_2011_2018": float(np.corrcoef(recon[[d for d in ovr if d <= "2018-12-31"]], fg[[d for d in ovr if d <= "2018-12-31"]])[0, 1]),
                        "corr_real_2019_2026": float(np.corrcoef(recon[[d for d in ovr if d >= "2019"]], fg[[d for d in ovr if d >= "2019"]])[0, 1])}
    cd, cv = recon.index.to_numpy(str), recon.to_numpy(float)
    nd, _, nc = load_px(P7, "^NDX")
    RN = Asset("NDX", nd, nc, nc, cd[0], mode="close")
    ird, _, irc = load_px(P7, "^IRX")
    RN3 = Asset("NDX3", nd, None, synth3(nd, nc, ird, irc), cd[0], mode="close")
    rdec = le(cd, cv, T0)
    rrep = {}
    for nm, A in (("NDX", RN), ("NDX3", RN3)):
        rrep[nm] = {w: cell(A, rdec, H0, win) for w, win in WIN_R.items()}
        rrep[nm].update({w + "_늦춤": cell(A, rdec, H0, win, delay=1) for w, win in WIN_R.items()})
        rrep[nm].update({k: cell(A, rdec, H0, (a, z, END)) for k, (a, z) in ERAS_R.items()})
    J["recon_rep"] = rrep
    print("재구성", J["recon_check"], f"{time.time() - t0:.0f}s")

    # ---------------------------------------------------------------- 2) 시점 감사
    gd, _, gc = load_px(P7, "^GSPC")
    spr = pd.Series(gc, index=gd).pct_change()
    tim = {}
    for lab, (a, z) in {"보관본 2011~2021-01": ("2011-01-01", "2021-01-31"), "CNN 공식 2021-02~": ("2021-02-01", END)}.items():
        f = fg[(fg.index >= a) & (fg.index <= z)]
        common = [d for d in f.index if d in spr.index]
        f = f.reindex(common)
        dfg = f.diff()
        r = spr.reindex(spr.index)  # 거래일 축
        pos = np.searchsorted(spr.index.to_numpy(str), np.asarray(common, dtype=str))
        sv = spr.to_numpy()
        same = sv[pos]
        prev = sv[pos - 1]
        nxt = np.where(pos + 1 < len(sv), sv[np.clip(pos + 1, 0, len(sv) - 1)], np.nan)
        m = np.isfinite(dfg.to_numpy()) & np.isfinite(same) & np.isfinite(prev) & np.isfinite(nxt)
        x = dfg.to_numpy()[m]
        tim[lab] = {"n": int(m.sum()), "corr_dfg_same": float(np.corrcoef(x, same[m])[0, 1]), "corr_dfg_prev": float(np.corrcoef(x, prev[m])[0, 1]),
                    "corr_dfg_next": float(np.corrcoef(x, nxt[m])[0, 1]),
                    "non_trading_dates": int(sum(1 for d in f.index if d not in spr.index))}
    tim["fg_dates_not_trading_day"] = int(sum(1 for d in fd if d not in set(gd)))
    # 하루 늦춤·미끄러짐
    tim_rule = {}
    for s in TARGETS + ("7종목26",):
        A = AS[s]
        ws = ("sel", "chk", "2025", "2026앞x", "2026뒤") if s != "7종목26" else ("2026전체", "2026앞x", "2026뒤")
        tim_rule[s] = {}
        for w in ws:
            tim_rule[s][w] = {"base": cell(A, dec0, H0, WIN[w]).get("ex"), "delay1": cell(A, dec0, H0, WIN[w], delay=1).get("ex"),
                              "delay2": cell(A, dec0, H0, WIN[w], delay=2).get("ex"),
                              "slip_both": cell(A, dec0, H0, WIN[w], slip_ev=SLIP, slip_base=SLIP).get("ex"),
                              "slip_event_only": cell(A, dec0, H0, WIN[w], slip_ev=SLIP).get("ex")}
    J["timing"] = {"corr": tim, "rule": tim_rule}
    print("시점", tim, f"{time.time() - t0:.0f}s")

    # ---------------------------------------------------------------- 3) 이웃
    nei = {}
    for s in TARGETS:
        A = AS[s]
        for T in T_NEI:
            dec = le(fd, fv, T)
            for H in H_NEI:
                nei[f"{s}|{T}|{H}"] = {w: cell(A, dec, H, WIN[w]).get("ex") for w in ("sel", "chk", "2025", "2026앞x", "2026뒤")}
                nei[f"{s}|{T}|{H}"]["n_sel"] = cell(A, dec, H, WIN["sel"])["n"]
                nei[f"{s}|{T}|{H}"]["n_chk"] = cell(A, dec, H, WIN["chk"])["n"]
        for T in T_FINE:
            dec = le(fd, fv, T)
            nei[f"{s}|{T}|{H0}"] = {w: cell(A, dec, H0, WIN[w]).get("ex") for w in ("sel", "chk", "2025", "2026앞x", "2026뒤")}
            nei[f"{s}|{T}|{H0}"]["n_sel"] = cell(A, dec, H0, WIN["sel"])["n"]
            nei[f"{s}|{T}|{H0}"]["n_chk"] = cell(A, dec, H0, WIN["chk"])["n"]
    for T in T_NEI:
        dec = le(cd, cv, T)
        for H in H_NEI:
            nei[f"NDX_r|{T}|{H}"] = {w: cell(RN, dec, H, win).get("ex") for w, win in WIN_R.items()}
    J["neighbors"] = nei
    print("이웃", f"{time.time() - t0:.0f}s")

    # ---------------------------------------------------------------- 4) 위약
    rng = np.random.default_rng(SEED)
    sh = np.linspace(MIN_SHIFT, len(fv) - MIN_SHIFT, N_PLA).astype(int)
    shb = np.linspace(MIN_SHIFT, len(fv) - MIN_SHIFT, N_PLA_BIG).astype(int)
    pla = {}
    for s in TARGETS:
        A = AS[s]
        ws = ("sel", "chk", "2025", "2026앞x", "2026뒤")
        real = {w: rep[s][w].get("ex") for w in ws}
        roll = {w: [cell(A, le(fd, np.roll(fv, k), T0), H0, WIN[w]).get("ex") for k in sh] for w in ws}
        rollb = {w: [cell(A, le(fd, np.roll(fv, k), T0), H0, WIN[w]).get("ex") for k in shb] for w in ("sel", "chk")}
        rnd = {}
        for w in ws:
            a, z, cap = WIN[w]
            b = A.base_ent(a, z, cap, H0)
            n = rep[s][w]["n"]
            base = A.r(b, H0).mean()
            vals = []
            for _ in range(N_RND):
                if n == 0 or n > len(b):
                    vals.append(np.nan)
                    continue
                pick = rng.choice(b, n, replace=False)
                vals.append(float(A.r(pick, H0).mean() - base))
            rnd[w] = vals
        pla[s] = {w: {"real": real[w], "roll_rank": rank_of(real[w], roll[w]), "roll_mean": float(np.nanmean([x if x is not None else np.nan for x in roll[w]])),
                      "rnd_rank": rank_of(real[w], rnd[w]), "rnd_mean": float(np.nanmean(rnd[w])),
                      "roll_none": int(sum(1 for x in roll[w] if x is None))} for w in ws}
        for w in ("sel", "chk"):
            pla[s][w]["roll200_rank"] = rank_of(real[w], rollb[w])
        # 두 창 동시: 고르기·확인 모두 실제 이상인 위약 판 수
        pla[s]["both_ge"] = int(sum(1 for x, y in zip(rollb["sel"], rollb["chk"]) if x is not None and y is not None
                                     and x >= real["sel"] and y >= real["chk"]))
    # 재구성 위약
    shr = np.linspace(MIN_SHIFT, len(cv) - MIN_SHIFT, N_PLA).astype(int)
    for nm, A in (("NDX_r", RN), ("NDX3_r", RN3)):
        key = nm.split("_")[0]
        pla[nm] = {}
        for w, win in WIN_R.items():
            real = rrep[key][w].get("ex")
            roll = [cell(A, le(cd, np.roll(cv, k), T0), H0, win).get("ex") for k in shr]
            a, z, cap = win
            b = A.base_ent(a, z, cap, H0)
            n = rrep[key][w]["n"]
            base = A.r(b, H0).mean()
            rnd = [float(A.r(rng.choice(b, n, replace=False), H0).mean() - base) for _ in range(N_RND)]
            pla[nm][w] = {"real": real, "roll_rank": rank_of(real, roll), "roll_mean": float(np.nanmean([x if x is not None else np.nan for x in roll])),
                          "rnd_rank": rank_of(real, rnd), "rnd_mean": float(np.mean(rnd))}
    J["placebo"] = pla
    print("위약", f"{time.time() - t0:.0f}s")

    # 격자 필터 위약 다시 보기: 갈래 json에서 '통과 칸 0'인 위약 판 수(그 판은 '1위가 넘지 못함'으로 셈해졌다)
    try:
        G = json.loads((RESULTS / "p8_FGDIRECT.json").read_text(encoding="utf-8"))["grid_filter_placebo"]
        J["grid_filter_placebo_audit"] = {s: {"real_top": d["real"][1], "n_placebo_zero_pass": int(sum(1 for a, b in d["placebo"] if a == 0)),
                                              "max_placebo_top": max([b for a, b in d["placebo"] if b is not None], default=None)}
                                          for s, d in G.items()}
    except Exception as ex:  # 갈래 결과가 없으면 건너뜀
        J["grid_filter_placebo_audit"] = {"error": str(ex)}

    # ---------------------------------------------------------------- 군집(독립 사건) 분해·군집 부트스트랩
    clu = {}
    for s in TARGETS:
        A = AS[s]
        clu[s] = {}
        for w in ("sel", "chk"):
            cl = clusters(A, dec0, H0, WIN[w])
            a, z, cap = WIN[w]
            e = np.sort(A.ent(dec0, a, z, cap, H0))
            base = A.r(A.base_ent(a, z, cap, H0), H0).mean()
            # 군집 단위로 다시 뽑기(일 가중 평균 유지)
            sums = np.array([c["sum_ex"] for c in cl])
            ns = np.array([c["n"] for c in cl])
            bs = []
            for _ in range(N_BOOT):
                ix = rng.integers(0, len(cl), len(cl))
                bs.append(sums[ix].sum() / ns[ix].sum())
            loo = [float((sums.sum() - sums[i]) / (ns.sum() - ns[i])) for i in range(len(cl))]
            clu[s][w] = {"clusters": cl, "n_clusters": len(cl), "pos_share": float(np.mean([c["ex"] > 0 for c in cl])),
                         "eq_weight_ex": float(np.mean([c["ex"] for c in cl])), "boot_p05": float(np.percentile(bs, 5)),
                         "boot_p50": float(np.percentile(bs, 50)), "loo_min": float(min(loo)),
                         "loo_min_drop": cl[int(np.argmin(loo))]["start"], "p_le0": float(np.mean(np.asarray(bs) <= 0))}
    J["clusters"] = clu
    print("군집", f"{time.time() - t0:.0f}s")

    # ---------------------------------------------------------------- 5) 시대별
    era = {}
    for s in TARGETS:
        A = AS[s]
        era[s] = {k: cell(A, dec0, H0, (a, z, END)) for k, (a, z) in ERAS.items()}
        era[s].update({k + "_21": cell(A, dec0, 21, (a, z, END)) for k, (a, z) in ERAS.items() if k.startswith("2026") or k == "7월 쇼크"})
    era["7종목26"] = {k: cell(AS["7종목26"], dec0, H0, (a, z, END)) for k, (a, z) in ERAS.items() if k.startswith("2026") or k == "7월 쇼크"}
    era["7종목hold"] = {"2025": cell(AS["7종목hold"], dec0, H0, WIN_HOLD["2025"])}
    J["eras"] = era

    # ---------------------------------------------------------------- 가격만 쓴 대조(사전 고정, 2011~2018 빈도 맞춤)
    qd, qo, qc = AS["QQQ"].days, AS["QQQ"].px, AS["QQQ"].c
    qdd = pd.Series(qc, index=qd) / pd.Series(qc, index=qd).rolling(252, min_periods=252).max() - 1
    vd, _, vc = load_px(P7, "^VIX")
    vix = pd.Series(vc, index=vd)
    n_sel_days = int(((fd >= "2011-01-01") & (fd <= "2018-12-31") & (fv <= T0)).sum())
    q_sel = qdd[(qdd.index >= "2011-01-01") & (qdd.index <= "2018-12-31")].dropna()
    v_sel = vix[(vix.index >= "2011-01-01") & (vix.index <= "2018-12-31")]
    x_dd = float(np.sort(q_sel.to_numpy())[n_sel_days - 1])
    x_vix = float(np.sort(v_sel.to_numpy())[::-1][n_sel_days - 1])
    pdec = {"QQQ 252일 고점 대비 ≤%.1f%%" % (x_dd * 100): qdd[qdd <= x_dd].index.to_numpy(str),
            "VIX ≥%.1f" % x_vix: vix[vix >= x_vix].index.to_numpy(str)}
    pcmp = {"n_sel_days": n_sel_days}
    for lab, dec in pdec.items():
        pcmp[lab] = {s: {w: cell(AS[s], dec, H0, WIN[w]) for w in ("sel", "chk", "2025", "2026앞x", "2026뒤")} for s in TARGETS}
    # F&G 신호가 가격 신호 위에 더하는 것: 가격 신호 날 중 F&G ≤25 인 날 vs 아닌 날(확인 구간)
    add = {}
    fset = set(dec0)
    for lab, dec in pdec.items():
        both = np.asarray([d for d in dec if d in fset])
        only = np.asarray([d for d in dec if d not in fset])
        add[lab] = {s: {"both": cell(AS[s], both, H0, WIN["chk"]), "price_only": cell(AS[s], only, H0, WIN["chk"]),
                        "fg_only": cell(AS[s], np.asarray([d for d in dec0 if d not in set(dec)]), H0, WIN["chk"])} for s in ("QQQ", "SPY")}
    J["price_compare"] = pcmp
    J["fg_add"] = add
    print("가격 대조", f"{time.time() - t0:.0f}s")

    J["fg_2026_le25"] = [d for d in dec0 if d >= "2026"]
    gset = set(gd)
    J["fg_check"]["missing_trading_days"] = sorted(d for d in gset if d >= FG_START and d <= END and d not in set(fd))
    out = RESULTS / f"{ID}.json"
    J = clean(J)
    out.write_text(json.dumps(J, ensure_ascii=False, indent=1), encoding="utf-8")
    (RESULTS / f"{ID}.md").write_text(write_md(J), encoding="utf-8")
    print("저장", out, f"{time.time() - t0:.0f}s")


# ============================================================================ 보고서
CLAIM = {  # 갈래 주장(p8_FGDIRECT.md·대표 후보 설명), %p
    ("QQQ", "sel"): 1.3, ("QQQ", "chk"): 3.6, ("QQQ", "2025"): 4.7, ("QQQ", "2026앞x"): 11.3, ("QQQ", "2026뒤"): 2.8,
    ("SPY", "sel"): 0.6, ("SPY", "chk"): 3.4, ("SOXL", "sel"): 1.3, ("SOXL", "chk"): 20.7,
    ("7종목26", "2026전체"): 36.7, ("7종목hold", "2025"): 36.9}


def p(x, d=1):
    return "-" if x is None else f"{x * 100:+.{d}f}%p"


def pc(x, d=1):
    return "-" if x is None else f"{x * 100:.{d}f}%"


def rk(r):
    return "-" if not r else f"{r[0]}/{r[1]}"


def write_md(J) -> str:
    R, RR, T, N, PL, CL, ER, PC, AD = (J["reproduce"], J["recon_rep"], J["timing"], J["neighbors"], J["placebo"], J["clusters"],
                                       J["eras"], J["price_compare"], J["fg_add"])
    tc = T["corr"]
    L = []
    a = L.append
    q, s_ = R["QQQ"], R["SPY"]
    a("# 8단계 반박 검증 FGDIRECT: 'F&G ≤25인 날마다 사서 63거래일 들기' (2026-10-12)\n")
    a("> 후보: 그날 CNN F&G 종가 확정값 ≤25 → 다음 거래일 시가 매수(매일, 묶지 않음) → 63거래일 뒤 시가 매도. 비교 = 같은 창 안 아무 날에 사서 63일.")
    a("> 코드 `code/p8_V_FGDIRECT.py`(갈래 스크립트·공유 엔진 import 없음, parquet·json 직접 읽기, F&G는 `fg_data.fg()`), 숫자 `results/p8_V_FGDIRECT.json`.\n")
    a("## 결론\n")
    pq = PL["QQQ"]
    a(f"1. **숫자는 그대로 재현됐고 시점도 맞다.** QQQ 2011~18 {p(q['sel']['ex'])} → 2019~25 {p(q['chk']['ex'])}, 2026 앞 {p(q['2026앞x']['ex'])}, 뒤 {p(q['2026뒤']['ex'])}(갈래와 차이 0.0%p). "
      f"F&G 날짜는 그날 장 마감 값이다(F&G 하루 변화와 S&P 당일 수익 상관 보관본 {tc['보관본 2011~2021-01']['corr_dfg_same']:.2f}·공식 {tc['CNN 공식 2021-02~']['corr_dfg_same']:.2f}, 전날 수익과는 "
      f"{tc['보관본 2011~2021-01']['corr_dfg_prev']:.2f}·{tc['CNN 공식 2021-02~']['corr_dfg_prev']:.2f}). 하루 늦춰도 2019~25 {p(T['rule']['QQQ']['chk']['delay1'])}.")
    a(f"2. **그러나 우연과 잘 구별되지 않는다.** F&G를 원형으로 민 위약 200판 중 순위: 2011~18 QQQ {rk(pq['sel']['roll200_rank'])}·SPY {rk(PL['SPY']['sel']['roll200_rank'])}·SOXL {rk(PL['SOXL']['sel']['roll200_rank'])}, "
      f"2019~25 QQQ {rk(pq['chk']['roll200_rank'])}. 독립 군집(구간마다 {CL['QQQ']['sel']['n_clusters']}·{CL['QQQ']['chk']['n_clusters']}개)을 다시 뽑으면 QQQ 초과가 0 이하일 확률이 "
      f"{CL['QQQ']['sel']['p_le0']:.0%}·{CL['QQQ']['chk']['p_le0']:.0%}다. 2011~18에는 같은 빈도의 단순 가격 규칙(QQQ 1년 고점 대비 하락, VIX 높음)이 F&G보다 나았다. "
      f"2026 뒤 근거는 6/26 하루(F&G 24.7)뿐이고, 문턱 24면 사건이 없고 21·42일 보유면 마이너스다({p(N['QQQ|25|21']['2026뒤'])}·{p(N['QQQ|25|42']['2026뒤'])}).")
    a("3. **판정: confirmed_fragile(맞지만 약함).** 부호는 이웃 격자 대부분에서 유지되지만 크기는 사건 몇 개(2020-03, 2025-04, 2026-03)에 기댄다. "
      "F&G는 가격 급락과 겹친 날에만 의미가 있었다(아래 '가격만 쓴 대조'). 실무: '공포 구간에 새 돈 나눠 넣기, 3개월 기준' 관찰 후보 이상으로 올리지 않는다. 손절·재진입·비중 신호가 아니다.\n")

    # 사용자 표
    a("## 사용자 표\n")
    a("칸 = 그날 산 돈의 '아무 날 산 돈' 대비 초과(사건 일수). 2026 앞은 앞 구간에 산 돈을 10/9까지 잰 값. 7월 쇼크 = 6/22~7/29에 산 돈.\n")
    a("| 규칙 | 대상 | 2026 앞 | 2026 뒤 | 7월 쇼크 | 2025 | 실제 11~18 → 19~25 | 재구성 91~10 → 11~25 | 군집(11~18/19~25) |")
    a("|---|---|---|---|---|---|---|---|---|")
    for s in TARGETS:
        x = R[s]
        rr = f"{p(RR['NDX']['rsel']['ex'])} → {p(RR['NDX']['rchk']['ex'])} (NDX)" if s == "QQQ" else (
            f"{p(RR['NDX3']['rsel']['ex'])} → {p(RR['NDX3']['rchk']['ex'])} (NDX 3배)" if s == "TQQQ" else "-")
        a(f"| F&G ≤25 매일, 63일 | {s} | {p(x['2026앞x']['ex'])}({x['2026앞x']['n']}일) | {p(x['2026뒤']['ex'])}({x['2026뒤']['n']}일) | {p(x['AI쇼크']['ex'])} | "
          f"{p(x['2025']['ex'])}({x['2025']['n']}일) | {p(x['sel']['ex'])} → {p(x['chk']['ex'])} | {rr} | {CL[s]['sel']['n_clusters']}/{CL[s]['chk']['n_clusters']} |")
    x7, xh = R["7종목26"], R["7종목hold"]
    a(f"| 같은 규칙 | 7종목 | {p(x7['2026앞x']['ex'])}({x7['2026앞x']['n']}일) | {p(x7['2026뒤']['ex'])}(1일) | {p(x7['AI쇼크']['ex'])} | {p(xh['2025']['ex'])}({xh['2025']['n']}일) | - | - | - |")
    a(f"| 같은 규칙, 21일 보유 | QQQ | {p(ER['QQQ']['2026 앞_21']['ex'])} | {p(q['H21_2026뒤']['ex'])} | {p(q['H21_AI쇼크']['ex'])} | {p(N['QQQ|25|21']['2025'])} | "
          f"{p(N['QQQ|25|21']['sel'])} → {p(N['QQQ|25|21']['chk'])} | - | - |")
    for lab, d in PC.items():
        if lab == "n_sel_days":
            continue
        y = d["QQQ"]
        a(f"| (대조) {lab} 매일, 63일 | QQQ | {p(y['2026앞x']['ex'])}({y['2026앞x']['n']}일) | {p(y['2026뒤']['ex'])}({y['2026뒤']['n']}일) | - | {p(y['2025']['ex'])} | "
          f"{p(y['sel']['ex'])} → {p(y['chk']['ex'])} | - | - |")
    a("")
    a(f"- 추가 −20% 확률(63일 안, 2019~25): QQQ {pc(q['chk']['hit'])} 대 아무 날 {pc(q['chk']['base_hit'])}, TQQQ {pc(R['TQQQ']['chk']['hit'])} 대 {pc(R['TQQQ']['chk']['base_hit'])}, "
      f"SOXL {pc(R['SOXL']['chk']['hit'])} 대 {pc(R['SOXL']['chk']['base_hit'])}. 3배는 공포에 사도 꼬리가 줄지 않는다.")
    a("- 가격 대조 두 줄은 2011~18 사건일 수를 F&G ≤25(372일)에 맞춰 문턱을 미리 정했다(확인 구간을 보지 않음).\n")

    # 1 재현
    a("## 1) 재현\n")
    a("| 대상·구간 | 갈래 주장 | 독립 재계산 | 사건 일수(독립 사건) | 평균 수익 / 아무 날 |")
    a("|---|---|---|---|---|")
    lab = {"sel": "2011~18", "chk": "2019~25", "2025": "2025", "2026앞x": "2026 앞", "2026뒤": "2026 뒤", "2026전체": "2026 전체"}
    for (s, w), cv in CLAIM.items():
        x = R[s][w]
        a(f"| {s} {lab[w]} | {cv:+.1f}%p | {p(x['ex'], 2)} | {x['n']}({x.get('n_ind')}) | {pc(x.get('mean'))} / {pc(x['base'])} |")
    a(f"| 재구성 NDX 1991~2010 | +0.2%p | {p(RR['NDX']['rsel']['ex'], 2)} | {RR['NDX']['rsel']['n']}({RR['NDX']['rsel']['n_ind']}) | {pc(RR['NDX']['rsel']['mean'])} / {pc(RR['NDX']['rsel']['base'])} |")
    a(f"| 재구성 NDX 2011~25 | +1.4%p | {p(RR['NDX']['rchk']['ex'], 2)} | {RR['NDX']['rchk']['n']}({RR['NDX']['rchk']['n_ind']}) | {pc(RR['NDX']['rchk']['mean'])} / {pc(RR['NDX']['rchk']['base'])} |")
    a(f"| 재구성 NDX 2008 | -1.1%p | {p(RR['NDX']['2008']['ex'], 2)} | {RR['NDX']['2008']['n']}({RR['NDX']['2008']['n_ind']}) | {pc(RR['NDX']['2008']['mean'])} / {pc(RR['NDX']['2008']['base'])} |")
    rc = J["recon_check"]
    a("")
    a(f"- 재구성 F&G를 갈래 문서 정의대로 다시 만들었다: 갈래 캐시와 최대 차이 {rc['max_abs_diff_vs_branch']:.2f}점, 실제 F&G와 상관 {rc['corr_real_2011_2018']:.2f}(2011~18)·{rc['corr_real_2019_2026']:.2f}(2019~26).")
    a(f"- 갈래 설명의 '2026 앞 3/9~3/31 20일'은 실제로 3/9~4/7의 20일이다(3/12는 F&G 자료 빠짐). 숫자에는 영향 없음. F&G가 빠진 거래일은 2011~2026에 {len(J['fg_check']['missing_trading_days'])}일(2020-06~07 22일, 2026-03-12).\n")

    # 2 시점
    a("## 2) 시점 감사\n")
    a("| F&G 출처 | 일수 | 하루 변화 vs S&P 당일 수익 | vs 전날 수익 | vs 다음 날 수익 |")
    a("|---|---|---|---|---|")
    for k in ("보관본 2011~2021-01", "CNN 공식 2021-02~"):
        c = tc[k]
        a(f"| {k} | {c['n']} | {c['corr_dfg_same']:.2f} | {c['corr_dfg_prev']:.2f} | {c['corr_dfg_next']:.2f} |")
    a("")
    a("- 날짜가 하루 밀렸다면 '전날 수익' 상관이 가장 커야 한다. 두 출처 모두 당일이 압도적이라 밀림은 없다(공식값의 전날 0.10은 5일 평균 풋콜 등 느린 요소 탓). CNN 타임스탬프는 그날 00:00 UTC이고 값은 그날 마감값과 같다(10/9 00:00과 23:59 모두 45).\n")
    a("| 대상·구간 | 기본 | 하루 늦춤 | 이틀 늦춤 | 미끄러짐 0.5%(양쪽 다) | 미끄러짐 0.5%(공포일 매매만, 최악) |")
    a("|---|---|---|---|---|---|")
    for s in ("QQQ", "SPY", "SOXL", "7종목26"):
        for w, c in T["rule"][s].items():
            if w in ("sel", "chk", "2026뒤", "2026전체"):
                a(f"| {s} {lab.get(w, w)} | {p(c['base'])} | {p(c['delay1'])} | {p(c['delay2'])} | {p(c['slip_both'])} | {p(c['slip_event_only'])} |")
    a("")
    a("- 늦춤은 긴 구간에 거의 영향이 없다. 2026 뒤(하루짜리)는 늦출수록 0에 가까워진다. 공포일 호가가 넓다고 보고 공포일 매매에만 0.5%씩 물리면 2011~18 QQQ +0.3%p, SPY −0.4%p로 사라진다.\n")

    # 3 이웃
    a("## 3) 이웃 격자 (사전 고정: 문턱 15·20·25·30·35 × 보유 21·42·63·84·126일)\n")
    a("QQQ, 칸 = 2011~18 → 2019~25 초과 [2026 뒤].\n")
    a("| 문턱 \\ 보유 | " + " | ".join(f"{h}일" for h in H_NEI) + " |")
    a("|---|" + "---|" * len(H_NEI))
    for t in T_NEI:
        row = []
        for h in H_NEI:
            c = N[f"QQQ|{t}|{h}"]
            row.append(f"{p(c['sel'])} → {p(c['chk'])} [{p(c['2026뒤'])}]")
        a(f"| ≤{t} | " + " | ".join(row) + " |")
    a("")
    cnt = []
    for s in TARGETS:
        cells = [N[f"{s}|{t}|{h}"] for t in T_NEI for h in H_NEI]
        both = sum(1 for c in cells if c["sel"] is not None and c["chk"] is not None and c["sel"] > 0 and c["chk"] > 0)
        b26 = [c["2026뒤"] for c in cells if c["2026뒤"] is not None]
        cnt.append(f"{s} 두 구간 모두 플러스 {both}/25, 2026 뒤 사건이 있는 칸 {len(b26)}개 중 플러스 {sum(x > 0 for x in b26)}")
    a("- " + "; ".join(cnt) + ".")
    a("- 정수 문턱(63일): " + ", ".join(f"≤{t} {p(N[f'QQQ|{t}|63']['sel'])} → {p(N[f'QQQ|{t}|63']['chk'])} [{p(N[f'QQQ|{t}|63']['2026뒤'])}]" for t in T_FINE) + " (QQQ). ≤24 이하는 2026 뒤 사건이 없다.")
    nr = [N[f"NDX_r|{t}|{h}"] for t in T_NEI for h in H_NEI]
    a(f"- 재구성 NDX: 1991~2010 플러스 {sum(c['rsel'] > 0 for c in nr)}/25, 2011~25 플러스 {sum(c['rchk'] > 0 for c in nr)}/25. 옛 구간에서는 문턱이 낮을수록(더 극단적 공포) 오히려 마이너스였다(≤15·63일 {p(N['NDX_r|15|63']['rsel'])}).\n")

    # 4 위약
    a("## 4) 위약\n")
    a("원형 밀기 = F&G를 63거래일 이상 시간축으로 민 판(자기상관 유지). 무작위 날짜 = 같은 창에서 같은 일수를 아무 날로 뽑은 판(사건 몰림을 무시해 덜 보수적).\n")
    a("| 대상 | 구간 | 실제 | 원형 밀기 20판 순위(평균) | 원형 밀기 200판 순위 | 무작위 날짜 20판 순위(평균) |")
    a("|---|---|---|---|---|---|")
    for s in TARGETS:
        for w in ("sel", "chk", "2025", "2026앞x", "2026뒤"):
            c = PL[s][w]
            a(f"| {s} | {lab[w]} | {p(c['real'])} | {rk(c['roll_rank'])}({p(c['roll_mean'])}) | {rk(c.get('roll200_rank'))} | {rk(c['rnd_rank'])}({p(c['rnd_mean'])}) |")
    for s, key in (("NDX_r", "NDX"), ("NDX3_r", "NDX 3배")):
        for w, nm in (("rsel", "1991~2010"), ("rchk", "2011~25")):
            c = PL[s][w]
            a(f"| 재구성 {key} | {nm} | {p(c['real'])} | {rk(c['roll_rank'])}({p(c['roll_mean'])}) | - | {rk(c['rnd_rank'])}({p(c['rnd_mean'])}) |")
    a("")
    a("- 2026 칸의 순위 분모가 21보다 작은 것은 민 F&G가 그 구간에 ≤25를 만들지 않은 판을 뺐기 때문이다.")
    a(f"- 두 구간 동시: 200판 중 2011~18과 2019~25가 모두 실제 이상인 판 QQQ {PL['QQQ']['both_ge']}개, SPY {PL['SPY']['both_ge']}개, TQQQ {PL['TQQQ']['both_ge']}개, SOXL {PL['SOXL']['both_ge']}개(1.5~4%).")
    ga = J.get("grid_filter_placebo_audit", {})
    if "QQQ" in ga:
        a(f"- 갈래의 '격자 필터 위약 1위가 실제 1위를 넘은 적 0/20'은 맞지만, 위약 20판 중 통과 칸이 하나도 없던 판이 QQQ {ga['QQQ']['n_placebo_zero_pass']}·SPY {ga['SPY']['n_placebo_zero_pass']}·"
          f"TQQQ {ga['TQQQ']['n_placebo_zero_pass']}·SOXL {ga['SOXL']['n_placebo_zero_pass']}판이다('2026 뒤 초과>0' 필터가 6/1~7/7의 좁은 창에 사건을 요구해서). 남은 판의 최고는 QQQ {p(ga['QQQ']['max_placebo_top'])}(실제 {p(ga['QQQ']['real_top'])}), "
          f"TQQQ {p(ga['TQQQ']['max_placebo_top'])}(실제 {p(ga['TQQQ']['real_top'])})로 차이가 작다.\n")

    # 5 군집·시대
    a("## 5) 독립 군집과 시대별\n")
    a("군집 = 사건일 사이가 21거래일 넘게 벌어지면 새 군집. 부트스트랩 = 군집 단위 2,000번 다시 뽑기(일수 가중).\n")
    a("| 대상 | 구간 | 군집 수 | 플러스 군집 | 군집 동일가중 초과 | 5% 하한 | 0 이하 확률 | 한 군집 빼면 최저(뺀 군집) |")
    a("|---|---|---|---|---|---|---|---|")
    for s in TARGETS:
        for w in ("sel", "chk"):
            c = CL[s][w]
            a(f"| {s} | {lab[w]} | {c['n_clusters']} | {c['pos_share']:.0%} | {p(c['eq_weight_ex'])} | {p(c['boot_p05'])} | {c['p_le0']:.0%} | {p(c['loo_min'])}({c['loo_min_drop']}) |")
    a("")
    a("QQQ 2019~25 군집: " + ", ".join(f"{c['start'][:7]}({c['n']}일) {p(c['ex'], 0)}" for c in CL["QQQ"]["chk"]["clusters"]) + ".\n")
    a("| 시대(결정일) | QQQ | SPY | TQQQ | SOXL | 사건 일수(독립) | QQQ 평균 수익 / 아무 날 |")
    a("|---|---|---|---|---|---|---|")
    for k in ERAS:
        c = ER["QQQ"][k]
        a(f"| {k} | {p(c.get('ex'))} | {p(ER['SPY'][k].get('ex'))} | {p(ER['TQQQ'][k].get('ex'))} | {p(ER['SOXL'][k].get('ex'))} | {c['n']}({c.get('n_ind')}) | {pc(c.get('mean'))} / {pc(c['base'])} |")
    for k in ("2000~02", "2008", "1990"):
        c = RR["NDX"][k]
        if not c["n"]:
            a(f"| 재구성 {k} | 자료 없음(재구성 F&G 1991-03-11~) | - | - | - | 0 | - |")
            continue
        a(f"| 재구성 {k} | NDX {p(c.get('ex'))} | - | NDX 3배 {p(RR['NDX3'][k].get('ex'))} | - | {c['n']}({c.get('n_ind')}) | {pc(c.get('mean'))} / {pc(c['base'])} |")
    a("")
    a("- 1990은 재구성 F&G가 1991-03-11부터라(백분위에 252~504일 필요) 잴 수 없다.")
    a("- 2022는 초과 플러스지만 QQQ 평균 수익은 마이너스였다(덜 잃었을 뿐). 2011(유럽 위기)과 2008(재구성)은 마이너스. 공포가 길게 이어지는 장에서는 매일 사면 바닥 전 돈이 먼저 들어간다.\n")

    # 가격 대조
    a("## 가격만 쓴 대조 (F&G가 가격보다 나은가)\n")
    a("| 규칙(63일, 매일) | 대상 | 2011~18 | 2019~25 | 2025 | 2026 앞 | 2026 뒤 | 사건 일수 11~18/19~25 |")
    a("|---|---|---|---|---|---|---|---|")
    for s in ("QQQ", "SPY", "TQQQ", "SOXL"):
        a(f"| F&G ≤25 | {s} | {p(R[s]['sel']['ex'])} | {p(R[s]['chk']['ex'])} | {p(R[s]['2025']['ex'])} | {p(R[s]['2026앞x']['ex'])} | {p(R[s]['2026뒤']['ex'])} | {R[s]['sel']['n']}/{R[s]['chk']['n']} |")
        for labp, d in PC.items():
            if labp == "n_sel_days":
                continue
            y = d[s]
            a(f"| {labp} | {s} | {p(y['sel']['ex'])} | {p(y['chk']['ex'])} | {p(y['2025']['ex'])} | {p(y['2026앞x']['ex'])} | {p(y['2026뒤']['ex'])} | {y['sel']['n']}/{y['chk']['n']} |")
    a("")
    for labp, d in AD.items():
        y = d["QQQ"]
        a(f"- 2019~25 QQQ, {labp} 기준으로 나누면: 가격 신호 + F&G ≤25 {p(y['both']['ex'])}({y['both']['n']}일), 가격 신호만 {p(y['price_only']['ex'])}({y['price_only']['n']}일), "
          f"F&G ≤25만 {p(y['fg_only']['ex'])}({y['fg_only']['n']}일).")
    a("- 2011~18(고르기 구간)에는 두 가격 규칙이 F&G보다 2%p 넘게 나았고, 2019~25에는 F&G가 나았다(QQQ·SPY 1~2%p, 3배 2.5~11%p; 가격 규칙은 2019~25에 2.7~3.3배 자주 켜져 덜 극단적인 날까지 샀다). F&G 단독 정보는 확인되지 않는다.\n")

    a("## 판정\n")
    a("- **confirmed_fragile.** 재현·시점·늦춤·이웃 부호는 통과했다. 깨지지 않은 이유: 이웃 25칸 대부분이 두 구간 모두 플러스이고, 재구성 2011~25도 플러스다.")
    a("- 약한 이유: (1) 2011~18 효과가 위약과 구별되지 않는다(QQQ 200판 중 30위, SOXL 94위). (2) 독립 군집 16~18개, QQQ 0 이하 확률 12~15%. (3) 2026 뒤는 문턱에 0.3 걸친 하루, 21·42일 보유면 마이너스. "
      "(4) 격자를 본 뒤 고른 칸이고, 갈래의 격자 필터 위약은 절반 가까이가 빈 판이었다. (5) 2011~18에는 단순 가격 규칙이 더 나았다. (6) 공포일 호가 비용을 따로 물리면 2011~18 이익이 사라진다.")
    a("- 장세 전환 위험: 2011·2008처럼 공포가 몇 달 이어지면 매일 사기는 아무 날보다 못했다.\n")
    a("## 한계\n")
    a("- F&G 2011~2020은 CNN 개편 때 거꾸로 채운 값일 수 있다(검증 불가). 재구성은 7요소 중 5개 대리.")
    a("- 사건형 비교라 현금 기간 비용·기회비용(평소 현금으로 기다리기)은 따로 재지 않았다. '새 돈을 언제 넣나'에만 답한다.")
    a("- 위약은 F&G 하나만 민다. 가격 쪽 장세(2019~25의 빠른 V자 반등)가 모든 '공포 매수'를 돕는 효과는 가격 대조로만 일부 봤다.")
    a("- 수수료 0, 현금 이자 0, 배당 포함(Adj Close 계수). 매수 추천·주문 수량이 아니다.")
    return "\n".join(L) + "\n"


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(x) for x in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 5)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    return o


if __name__ == "__main__":
    main()
