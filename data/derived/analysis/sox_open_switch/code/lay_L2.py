"""BRIEF3 조건 겹치기 L2 — 반전 확인과 청산(sel 2026-01-02~06-30만, 잠금 켜지 않음).

사용: py lay_L2.py → results/lay_L2_sel.csv(모든 설정), results/lay_L2_frozen.json(얼린 것), results/lay_L2_trades_<id>_<방식>.csv

신호(BRIEF3 H0에서 출발, 격자 실행 전에 고정):
- 두 종목 s(SOXL·SOXS) 각각 L_s = 기준 봉부터 지금 봉 i까지 s 저가의 최저. 기준 봉 = 09:30 봉, 또는 앞 매매 청산 봉(확인 대기를
  포기한 봉)의 다음 봉. s 종가[i] ≥ L_s × (1 + x%)이면 s가 '오름' → 반대 종목을 산다(거울은 오른 종목을 산다).
  두 종목이 같은 봉에 함께 채우면 종가/L 비율이 큰 쪽(같으면 SOXL)을 오른 종목으로 본다.
- 확인(오른 종목 s의 봉으로 판단, 신호 봉 i 다음 봉 j = i+1 … i+W 안에서 처음 채운 봉에서 결정):
    none     : 확인 없음. 신호 봉 i 마감에 바로 주문(H0).
    red      : 첫 음봉. s 종가[j] < s 시가[j].
    low      : 앞 봉 저가 아래 마감. s 종가[j] < s 저가[j−1](j = i+1이면 앞 봉은 신호 봉).
    rsi      : RSI 70 아래 복귀. s 1분 RSI14[j−1] ≥ 70이고 RSI14[j] < 70(아래로 건넘, j−1 ≥ i).
  W(최대 대기) = 5·10·15분. W 안에 확인이 없으면 그 신호를 버리고, 두 종목 L을 포기한 봉 다음 봉부터 다시 잰다(H0의 '청산 뒤
  다시 재기'와 같은 처리). 대기 중에는 새 신호를 보지 않는다.
- 진입 마감: 진입 결정 봉(none은 신호 봉, 나머지는 확인 봉)의 마감이 창 끝 15분 전까지(w90 10:45, std 09:45 = 결정 봉 색인 ≤ cut − 16).
- 체결·비용·손절·목표는 lib.Engine(결정 봉 i → 봉 i+1 종가 + 반 호가, 목표는 체결 다음 봉부터 고가 ≥ 목표 × 1.0002, 손절 우선).
- 청산(먼저 오는 것): 목표 +target(0.5·0.75·1·1.5%), 손절 −stop(없음·1·2·3%), 시간 손절 hold분(15·30·45·없음: 체결 뒤 hold분 되는
  봉 종가 − 반 호가, F4와 같은 식 = 결정 봉 i에서 (i+1) − entry_i ≥ hold), 창 끝(w90 10:59 봉 종가, std 09:59 봉 종가 − 반 호가).
- 한 번에 한 종목. 들고 있는 동안 새 신호는 무시하고, 청산 봉 다음 봉부터 두 종목 L을 다시 잰다. 하루 매매 수 제한 없음.
- RSI14: Wilder, 정규장 봉 중 거래가 있는 봉만(거래량 > 0, events_sel.py와 같은 뜻), 앞 2거래일 정규장 09:30~15:59로 예열 후 오늘
  09:30부터 이어 붙임(연장 끔). 거래 없는 분은 앞 값 유지. 반일장의 13:00 뒤 연장 봉도 거래가 있으면 예열에 들어간다(ctx가 반일장
  여부를 주지 않음, 예열 2일 중 한 날만 해당, RSI14는 수십 봉이면 잊힘).

격자(설정 390개, 실행 전에 고정):
  A 확인×목표: x{2.5,3,4} × 확인{none, red·low·rsi × 대기 5·10·15} × 목표{0.5,0.75,1,1.5%}, 손절·시간 손절 없음 = 120
  B 청산: x 3 × 확인{none, red10, low10, rsi10} × 목표 4 × 손절{없음,3,2,1%} × 시간 손절{15,30,45,없음} = 256, A와 겹침 16 → 240
  C x 축 이웃: x{2.5,4} × none × 목표 1% × 손절 4 × 시간 손절 4 = 32, A와 겹침 2 → 30
고르기(BRIEF3): 점수 = sel 하루 평균 t(모든 sel 날, 활동일 20일 이상만 자격), 최종 = 설정과 격자 이웃(순서 있는 축 하나를 한 칸 옮긴
  격자 안 설정: x·대기·목표·손절·시간 손절, 확인 종류는 범주라 이웃 아님) t의 중앙값. 같으면 자기 t. w90 1위·std 1위(같으면 1개).
진단(고르기에 안 씀): 모든 설정의 거울(같은 신호·확인·청산, 종목만 뒤집기) sel 하루 평균과 방향 성분 (전략 − 거울)/2.
"""
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lib import Ctx, Data, Engine, Order, idx, hm, summarize, RTH0, RTH1, ROOT  # noqa: E402

RSI_N, RSI_HI = 14, 70.0
SIGN = {"SOXL": 1, "SOXS": -1}
SYMS2 = ("SOXL", "SOXS")
DEFAULTS = dict(x=3.0, conf="none", wait=None, target=0.01, stop=None, hold=None)
KEYS = ["x", "conf", "wait", "target", "stop", "hold"]
AX = {"x": [2.5, 3.0, 4.0], "wait": [5, 10, 15], "target": [0.005, 0.0075, 0.01, 0.015],
      "stop": [None, 0.03, 0.02, 0.01], "hold": [15, 30, 45, None]}


# ------------------------------------------------------------------ RSI(봉 i까지 본 값만, 설정끼리 공유하는 캐시)
_RS = {}   # (날짜, 종목) → {"ag", "al", "last", "upto", "val"}


def _rsi_seed(ctx, sym):
    s = []
    for k in (2, 1):
        c, v = ctx.prev_bars(sym, "Close", k), ctx.prev_bars(sym, "Volume", k)
        if c is None:
            continue
        c, v = c[RTH0:RTH1], v[RTH0:RTH1]
        s.append(c[(v > 0) & np.isfinite(c)])
    s = np.concatenate(s) if s else np.array([])
    if len(s) <= RSI_N:
        return {"ag": None, "al": None, "last": s[-1] if len(s) else None, "upto": RTH0 - 1, "val": np.nan,
                "buf": list(s)}
    d = np.diff(s)
    g, l = np.clip(d, 0, None), np.clip(-d, 0, None)
    ag, al = g[:RSI_N].mean(), l[:RSI_N].mean()
    a = 1.0 / RSI_N
    for x, y in zip(g[RSI_N:], l[RSI_N:]):
        ag += a * (x - ag)
        al += a * (y - al)
    return {"ag": ag, "al": al, "last": s[-1], "upto": RTH0 - 1, "val": np.nan, "buf": None}


def rsi_at(ctx, sym, i):
    """오늘 봉 i의 RSI14(봉 i 마감 때 아는 값). 캐시는 봉 순서대로 늘린다(봉 i 넘어서는 계산하지 않음)."""
    key = (ctx.day, sym)
    st = _RS.get(key)
    if st is None:
        st = _RS[key] = _rsi_seed(ctx, sym)
        st["vals"] = {}
    if i in st["vals"]:
        return st["vals"][i]
    c, v = ctx.bars(sym, "Close"), ctx.bars(sym, "Volume")
    a = 1.0 / RSI_N
    for j in range(st["upto"] + 1, i + 1):
        if v[j] > 0 and np.isfinite(c[j]):
            if st["ag"] is None:                      # 예열 부족(자료 첫날 근처)일 때만
                st["buf"].append(c[j])
                if len(st["buf"]) > RSI_N:
                    d = np.diff(np.array(st["buf"]))
                    st["ag"] = np.clip(d, 0, None).mean()
                    st["al"] = np.clip(-d, 0, None).mean()
                    st["last"] = c[j]
            else:
                dl = c[j] - st["last"]
                st["ag"] += a * (max(dl, 0.0) - st["ag"])
                st["al"] += a * (max(-dl, 0.0) - st["al"])
                st["last"] = c[j]
            if st["ag"] is not None:
                st["val"] = 100.0 if st["al"] == 0 else 100.0 - 100.0 / (1.0 + st["ag"] / st["al"])
        st["vals"][j] = st["val"]
    st["upto"] = max(st["upto"], i)
    return st["vals"][i]


# ------------------------------------------------------------------ 전략
class Fade2:
    """H0 신호 → (확인) → 반대 종목 1개. 상태: flat(신호 찾기) / wait(확인 대기) / entering / in / exiting / done."""

    def __init__(self, p, mirror=False):
        self.p, self.m = p, (-1 if mirror else 1)
        self.k = 1.0 + p["x"] / 100.0
        self.L = {s: np.inf for s in SYMS2}
        self.st, self.sig = "flat", None
        self.log = []                                # (신호 봉, 오른 종목, 결정 봉) — 점검용

    def _reset(self):
        self.L = {s: np.inf for s in SYMS2}
        self.st, self.sig = "flat", None

    def _enter(self, s, i):
        self.st = "entering"
        self.log.append((self.sig[1] if self.sig else i, s, i))
        p = self.p
        return Order(-SIGN[s] * self.m, stop=p["stop"], target=p["target"])

    def _confirmed(self, ctx, s, j):
        cf = self.p["conf"]
        if cf == "red":
            return ctx.bars(s, "Close")[j] < ctx.bars(s, "Open")[j]
        if cf == "low":
            return ctx.bars(s, "Close")[j] < ctx.bars(s, "Low")[j - 1]
        if cf == "rsi":
            r0, r1 = rsi_at(ctx, s, j - 1), rsi_at(ctx, s, j)
            return np.isfinite(r0) and np.isfinite(r1) and r0 >= RSI_HI and r1 < RSI_HI
        raise ValueError(cf)

    def on_bar(self, ctx, i):
        p = self.p
        # 1) 보유 상태 정리(청산은 이 봉 i에서 일어남 → L은 i+1부터)
        if self.st == "entering":
            if ctx.pos != 0:
                self.st = "in"
            else:                                    # 시가 체결 감도에서 체결 봉 안에 나간 경우
                self._reset()
                return None
        if self.st in ("in", "exiting"):
            if ctx.pos == 0:
                self._reset()
                return None
            if self.st == "in" and p["hold"] is not None and (i + 1) - ctx.entry_i >= p["hold"]:
                self.st = "exiting"
                return Order(0)
            return None
        if self.st == "done":
            return None
        last_entry = ctx.cut - 16                    # 결정 봉 마감 ≤ 창 끝 15분 전
        # 2) 확인 대기
        if self.st == "wait":
            s, i0 = self.sig
            if i > last_entry:
                self.st = "done"
                return None
            if self._confirmed(ctx, s, i):
                return self._enter(s, i)
            if i >= i0 + p["wait"]:
                self._reset()                        # 신호 포기, L은 i+1부터 다시
            return None
        # 3) 신호 찾기
        for s in SYMS2:
            lo = ctx.bars(s, "Low")[i]
            if lo < self.L[s]:
                self.L[s] = lo
        if i > last_entry:
            self.st = "done"
            return None
        best, br = None, 0.0
        for s in SYMS2:
            r = ctx.bars(s, "Close")[i] / self.L[s]
            if r >= self.k and r > br:
                best, br = s, r
        if best is None:
            return None
        self.sig = (best, i)
        if p["conf"] == "none":
            return self._enter(best, i)
        self.st = "wait"
        return None


def make(params, mirror=False):
    p = dict(DEFAULTS, **params)
    return lambda: Fade2(p, mirror)


# ------------------------------------------------------------------ 격자
def grid():
    out = {}

    def add(**kw):
        p = dict(DEFAULTS, **kw)
        if p["conf"] == "none":
            p["wait"] = None
        out.setdefault(tuple(p[k] for k in KEYS), p)
    confs = [("none", None)] + [(c, w) for c in ("red", "low", "rsi") for w in AX["wait"]]
    # A 확인 × 목표 (손절·시간 손절 없음) = 3 × 10 × 4 = 120
    for x, (cf, w), tg in itertools.product(AX["x"], confs, AX["target"]):
        add(x=x, conf=cf, wait=w, target=tg)
    # B 청산 (x 3, 확인 none·대기 10분 셋) = 4 × 4 × 4 × 4 = 256, A와 겹침 16
    for (cf, w), tg, sp, h in itertools.product([("none", None), ("red", 10), ("low", 10), ("rsi", 10)],
                                                AX["target"], AX["stop"], AX["hold"]):
        add(x=3.0, conf=cf, wait=w, target=tg, stop=sp, hold=h)
    # C x 축 이웃 (none, 목표 1%) = 2 × 4 × 4 = 32, A와 겹침 2
    for x, sp, h in itertools.product([2.5, 4.0], AX["stop"], AX["hold"]):
        add(x=x, conf="none", target=0.01, stop=sp, hold=h)
    return out


def cid(p):
    f = lambda v: "N" if v is None else f"{v * 100:g}"
    w = "" if p["conf"] == "none" else str(p["wait"])
    return f"L2-x{p['x']:g}-{p['conf']}{w}-t{f(p['target'])}-s{f(p['stop'])}-h{'N' if p['hold'] is None else p['hold']}"


def neighbors(key, keys):
    out = []
    for a, vals in AX.items():
        j = KEYS.index(a)
        v = key[j]
        if v not in vals:
            continue
        k = vals.index(v)
        for kk in (k - 1, k + 1):
            if 0 <= kk < len(vals):
                nk = key[:j] + (vals[kk],) + key[j + 1:]
                if nk in keys:
                    out.append(nk)
    return out


def trade_stats(tr):
    o = {"avg_trade_pct": float(tr["ret"].mean() * 100) if len(tr) else np.nan}
    for w in ["target", "stop", "signal", "cutoff"]:
        o[f"x_{w}"] = int((tr["why"] == w).sum()) if len(tr) else 0
    o["hit_target"] = float((tr["why"] == "target").mean()) if len(tr) else np.nan
    o["n_soxl"] = int((tr["sym"] == "SOXL").sum()) if len(tr) else 0
    if len(tr):
        mins = [(idx(b) - idx(a)) for a, b in zip(tr["entry"], tr["exit"])]
        o["hold_min_mean"] = float(np.mean(mins))
        o["miss_avg_pct"] = float(tr.loc[tr["why"] != "target", "ret"].mean() * 100) if (tr["why"] != "target").any() else np.nan
        o["worst_trade_pct"] = float(tr["ret"].min() * 100)
    else:
        o["hold_min_mean"] = o["miss_avg_pct"] = o["worst_trade_pct"] = np.nan
    return o


def flat(s, tr, pre):
    row = {f"{pre}_{k}": v for k, v in s.items() if k != "ci95"}
    row[f"{pre}_ci_lo"], row[f"{pre}_ci_hi"] = s.get("ci95", [np.nan, np.nan])
    row.update({f"{pre}_{k}": v for k, v in trade_stats(tr).items()})
    return row


def check_vs_events(D, E):
    """H0와 같은 설정(x3·none·목표1%·손절·시간 손절 없음)의 그날 첫 매매가 events_sel.py의 x=3 first 사건(두 종목 중 이른 것)의
    반대 매수 +1% 손익(opp_p10)과 같은지 대조."""
    ev = pd.read_csv(os.path.join(ROOT, "results", "events_sel_events.csv"))
    ev = ev[(ev["x"] == 3.0) & (ev["kind"] == "first")]
    out = {}
    for sc, eng in E.items():
        daily, tr = eng.run(make({"x": 3.0, "conf": "none", "target": 0.01}), period="sel")
        first = tr.groupby("day").head(1).copy()
        first["day"] = first["day"].astype(str)
        e = ev[ev["win"] == sc].sort_values(["day", "i"]).groupby("day").head(1)
        m = first.merge(e, on="day", how="outer", indicator=True)
        both = m[m["_merge"] == "both"]
        out[sc] = {"days_strategy": int(len(first)), "days_events": int(len(e)),
                   "only_one_side": int((m["_merge"] != "both").sum()),
                   "entry_time_mismatch": int((both["entry"] != both["sig"]).sum()),
                   "sym_mismatch": int((both["sym"] != both["buy_opp"]).sum()),
                   "max_abs_diff_pct": float((both["ret"] * 100 - both["opp_p10"]).abs().max())}
    return out


def check_confirm(D, E, params_list):
    """확인 진입이 정의대로인지: 결정 봉(체결 봉 − 1)에서 오른 종목의 확인 조건과 진입 마감을 다시 본다(w90·std)."""
    A = D.A
    bad = {}
    for p in params_list:
        for sc, eng in E.items():
            daily, tr = eng.run(make(p), period="sel")
            n_bad = 0
            last = (idx("11:00") if sc == "w90" else idx("10:00")) - 16
            for r in tr.itertuples(index=False):
                d = D.pos[r.day]
                j = idx(r.entry) - 1
                s = "SOXS" if r.sym == "SOXL" else "SOXL"
                if j > last:
                    n_bad += 1
                    continue
                if p["conf"] == "red":
                    ok = A[s]["Close"][d, j] < A[s]["Open"][d, j]
                elif p["conf"] == "low":
                    ok = A[s]["Close"][d, j] < A[s]["Low"][d, j - 1]
                else:
                    ok = True
                n_bad += (not ok)
            bad[f"{cid(p)}_{sc}"] = {"trades": int(len(tr)), "bad": int(n_bad)}
    return bad


def check_rsi(D):
    """rsi_at(앞 2일 예열)을 events_sel.py의 RSI 열(앞 10거래일부터 이은 사슬)과 사건 봉에서 대조."""
    ev = pd.read_csv(os.path.join(ROOT, "results", "events_sel_events.csv"),
                     usecols=["d", "rise", "i", "rsi"]).drop_duplicates(["d", "rise", "i"])
    diffs = []
    for r in ev.itertuples(index=False):
        ctx = Ctx(D, int(r.d))
        ctx.i = int(r.i)
        v = rsi_at(ctx, r.rise, int(r.i))
        if np.isfinite(r.rsi):
            diffs.append(abs(v - r.rsi))
    diffs = np.array(diffs)
    return {"n": int(len(diffs)), "max_abs": float(diffs.max()), "p99_abs": float(np.percentile(diffs, 99))}


def main():
    t0 = time.time()
    D = Data()
    print(f"data {time.time() - t0:.1f}s", flush=True)
    G = grid()
    assert len(G) <= 400, len(G)
    assert len({cid(p) for p in G.values()}) == len(G)
    print("configs", len(G), flush=True)
    E = {"w90": Engine(D, scheme="w90"), "std": Engine(D, scheme="std")}
    chk = check_vs_events(D, E)
    print("check_vs_events", chk, flush=True)
    chk2 = check_confirm(D, E, [dict(DEFAULTS, x=3.0, conf=c, wait=10) for c in ("red", "low", "rsi")])
    print("check_confirm", chk2, flush=True)
    chk3 = check_rsi(D)
    print("check_rsi", chk3, flush=True)

    rows, keys = [], list(G)
    for n, k in enumerate(keys):
        p = G[k]
        row = {"id": cid(p), **{kk: p[kk] for kk in KEYS}}
        for sc, eng in E.items():
            daily, tr = eng.run(make(p), period="sel")
            assert np.isfinite(daily["ret"]).all(), (cid(p), sc)
            row.update(flat(summarize(daily), tr, sc))
            dm, _ = eng.run(make(p, mirror=True), period="sel")            # 진단(고르기에 안 씀)
            row[f"{sc}_mirror_mean_pct"] = float(dm["ret"].mean() * 100)
            dir_ = (daily["ret"].values - dm["ret"].values) * 50
            row[f"{sc}_dir_mean_pct"] = float(dir_.mean())
            row[f"{sc}_dir_t"] = float(dir_.mean() / (dir_.std(ddof=1) / np.sqrt(len(dir_)))) if dir_.std() > 0 else 0.0
        rows.append(row)
        if n % 25 == 0:
            print(n, f"{time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    kset = set(keys)
    pos = {k: n for n, k in enumerate(keys)}
    for sc in E:
        med, nn = [], []
        for k in keys:
            nb = neighbors(k, kset)
            ts = [df.at[pos[k], f"{sc}_t"]] + [df.at[pos[q], f"{sc}_t"] for q in nb]
            med.append(float(np.median(ts)))
            nn.append(len(nb))
        df[f"{sc}_nbr_med_t"] = med
        df[f"{sc}_n_nbr"] = nn
        df[f"{sc}_eligible"] = df[f"{sc}_active_days"] >= 20
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    df.to_csv(os.path.join(ROOT, "results", "lay_L2_sel.csv"), index=False, encoding="utf-8-sig")

    # 고르기
    frozen, landscape = [], {}
    for sc in E:
        el = df[df[f"{sc}_eligible"]]
        ts = df[f"{sc}_t"]
        landscape[sc] = {"n_eligible": int(len(el)), "t_max_all": float(ts.max()), "t_median_all": float(ts.median()),
                         "t_pos_frac_all": float((ts > 0).mean()),
                         "t_max_eligible": float(el[f"{sc}_t"].max()) if len(el) else None,
                         "t_median_eligible": float(el[f"{sc}_t"].median()) if len(el) else None,
                         "t_pos_frac_eligible": float((el[f"{sc}_t"] > 0).mean()) if len(el) else None,
                         "nbr_med_max_eligible": float(el[f"{sc}_nbr_med_t"].max()) if len(el) else None,
                         "mean_pct_median_eligible": float(el[f"{sc}_mean_pct"].median()) if len(el) else None,
                         "dir_mean_median_all": float(df[f"{sc}_dir_mean_pct"].median()),
                         "dir_t_pos_frac_all": float((df[f"{sc}_dir_t"] > 0).mean())}
        if not len(el):
            continue
        best = el.sort_values([f"{sc}_nbr_med_t", f"{sc}_t"], ascending=False).iloc[0]
        if any(f["id"] == best["id"] for f in frozen):
            frozen[[f["id"] for f in frozen].index(best["id"])]["scheme"] += "+" + sc
            continue
        frozen.append({"id": best["id"], "scheme": sc})
    id2p = {cid(p): p for p in G.values()}
    out = []
    for f in frozen:
        p = dict(id2p[f["id"]])
        k = tuple(p[kk] for kk in KEYS)
        r = df.iloc[pos[k]]
        sel, mir = {}, {}
        for sc, eng in E.items():
            daily, tr = eng.run(make(p), period="sel")
            s = summarize(daily)
            s.update(trade_stats(tr))
            s["exits"] = tr["why"].value_counts().to_dict() if len(tr) else {}
            s["entries_by_sym"] = tr["sym"].value_counts().to_dict() if len(tr) else {}
            s["nbr_med_t"] = float(r[f"{sc}_nbr_med_t"])
            dm, trm = eng.run(make(p, mirror=True), period="sel")
            ms = summarize(dm)
            ms.update(trade_stats(trm))
            dir_ = (daily["ret"].values - dm["ret"].values) * 50
            ms["dir_mean_pct"] = float(dir_.mean())
            sel[sc], mir[sc] = s, ms
            tr.to_csv(os.path.join(ROOT, "results", f"lay_L2_trades_{f['id']}_{sc}.csv"), index=False, encoding="utf-8-sig")
        nb = neighbors(k, kset)
        out.append({"id": f["id"], "scheme": f["scheme"], "params": p,
                    "factory": f"lay_L2.make({json.dumps(p)})".replace("null", "None"),
                    "code": os.path.join(HERE, "lay_L2.py"),
                    "sel_w90": sel["w90"], "sel_std": sel["std"], "mirror_sel": mir,
                    "neighbors": [{"id": cid(G[q]), "w90_t": float(df.at[pos[q], "w90_t"]),
                                   "std_t": float(df.at[pos[q], "std_t"])} for q in nb]})
    res = {"family": "L2 반전 확인과 청산", "n_configs": len(G), "period": "sel 2026-01-02~06-30 (123일)",
           "engine": "Engine(D, scheme='w90') / Engine(D, scheme='std'), fill='close', cost_table 1배",
           "score": "sel 하루 평균 t(모든 sel 날), 자격 활동일 ≥ 20, 최종 = 설정 + 격자 이웃(x·대기·목표·손절·시간 손절 한 칸) t 중앙값, 같으면 자기 t",
           "grid": "A 120 = x{2.5,3,4} × 확인{none, red/low/rsi × 대기 5·10·15} × 목표{0.5,0.75,1,1.5%}; "
                   "B 240 = x3 × 확인{none, red10, low10, rsi10} × 목표 4 × 손절{없음,3,2,1%} × 시간 손절{15,30,45,없음} − A 겹침 16; "
                   "C 30 = x{2.5,4} × none × 목표1% × 손절 4 × 시간 손절 4 − A 겹침 2",
           "checks": {"vs_events_sel": chk, "confirm_rule": chk2, "rsi_vs_events_sel": chk3},
           "landscape": landscape, "frozen": out, "runtime_s": round(time.time() - t0, 1)}
    with open(os.path.join(ROOT, "results", "lay_L2_frozen.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({"landscape": landscape, "frozen": [{k: o[k] for k in ["id", "scheme", "params"]} for o in out]},
                     ensure_ascii=False, default=str), flush=True)
    print(f"done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
