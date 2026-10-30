"""L3 맥락 묶음(BRIEF3 '조건 겹치기') — H0 신호에 맥락 조건을 더한다. sel(2026-01-02~06-30)만, 잠금 켜지 않음.
사용: py lay_L3.py → results/lay_L3_sel.csv, results/lay_L3_frozen.json, results/lay_L3_trades_<id>_<scheme>.csv

H0(BRIEF3 그대로, 고르는 값 없음):
- 종목 s∈{SOXL, SOXS}마다 L_s = 시작 봉(09:30 봉, 또는 앞 매매 청산 봉 j의 다음 봉 j+1)부터 지금 봉 i까지 s 저가의 최저.
- 신호: s 1분 종가 ≥ L_s × 1.03 → 반대 종목 시장가 매수(엔진: 봉 i+1 종가 + 반 호가). 목표 +1% 지정가, 손절 없음,
  못 닿으면 창 끝(w90 10:59 봉, std 09:59 봉 종가 − 반 호가).
- 진입 마감: 결정 봉 마감 ≤ 창 끝 15분 전(i ≤ cut − 16: w90 10:44 봉, std 09:44 봉). 들고 있는 동안 신호 무시.
  청산 봉 j가 지나면 두 종목의 L(과 맥락 M의 기준 창)을 j+1 봉부터 다시 잰다.
- 같은 봉에 두 종목 신호가 함께 서면 종가/L이 큰 쪽부터 맥락 문을 본다(드묾).
L3 맥락 조건(격자 실행 전에 정함). 모두 '오른 종목' s 기준, 신호 봉 i 마감 때 아는 값:
- 맥락 조건은 문(gate)이다: H0 신호가 참이고 맥락 조건이 모두 참인 첫 봉에 산다. 맥락이 막은 봉은 넘어가고 L은 H0대로 계속 잰다
  (H0 신호가 다음 봉에도 참이고 그때 맥락이 맞으면 그때 산다. 예: 첫 5분 방향은 09:34 봉 마감 전엔 몰라서 막힌다).
- T 신호 시각대(결정 봉 마감 시각): 1 = 09:31~09:45, 2 = 09:46~10:15, 3 = 10:16~10:45. std는 09:45까지라 1만 있다(T=1 ≡ any).
- G 갭(오른 종목 09:30 시가 ÷ 전날 정규장 종가 − 1, %): down < −1, flat −1~+1, up > +1 (events_sel과 같은 경계).
- F 첫 5분(오른 종목 09:34 봉 종가 ≥ 09:30 시가면 up, 아니면 down).
- M 시장 동행 수 0~2: QQQ·NVDA 각각, s가 SOXL이면 종가 ≥ (L과 같은 창의 저가 최저) × 1.01, s가 SOXS면 종가 ≤ (같은 창 고가 최고) × 0.99.
  (events_sel의 x/300 문턱, x = 3.) 0 = 반도체만, 2 = 시장 전체.
- P 전날 SOXL 정규장 고저폭(고가/저가 − 1) 3분위 lo/mid/hi. 문턱 PR_Q = sel 날들의 전날 값 분포 1/3·2/3 분위(얼림, 아래 상수).
- S 되돌릴 쪽: both, SOXL(SOXL 오름만 → SOXS 매수), SOXS(SOXS 오름만 → SOXL 매수).
격자: T·G·F·M·P 중 any가 아닌 축 0~2개 × S 3 = (1 + 14 + 78) × 3 = 279.
이웃: 한 축만 한 칸 다른 격자 안 설정. 칸 = any↔특정 값(조건 넣기·빼기), 순서 축(T 1-2-3, G down-flat-up, M 0-1-2, P lo-mid-hi)의
  붙은 값. F up↔down, S SOXL↔SOXS는 이웃이 아니다(방향을 뒤집는 것은 한 칸이 아니다).
점수(BRIEF3): sel 하루 평균 t(123일 전부, 매매 없는 날 0), 활동일 ≥ 20만 고를 자격. 최종 = 자기와 이웃 t의 중앙값
  (매매 0건인 설정은 t가 없어서 뺀다). 같으면 t, 조건 수 적은 것, id 순. w90 1위·std 1위를 얼린다(같으면 1개).
거울(mirror=True): 같은 신호에 오른 종목 자체를 산다(SOXL↔SOXS 뒤집기 = 추종). 방향 성분 = (전략 − 거울)/2.
전략은 ctx로만 자료를 본다. 날 단위 맥락 캐시는 (날짜, 종목) 키로 그날 개장 뒤 아는 값만 담는다(설정끼리 공유).
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
from lib import Data, Engine, Order, idx, hm, summarize, RTH0, ROOT  # noqa: E402

X_RISE = 0.03                      # H0 '3% 더'
TARGET = 0.01                      # H0 '+1%'
MKT_THR = X_RISE / 3               # QQQ·NVDA 동행 문턱(events_sel x/300)
GAP_THR = 1.0                      # 갭 ±1%
PR_Q = (0.067834671827, 0.104454534104)   # sel 날들의 전날 SOXL 고저폭 1/3·2/3 분위(main에서 다시 재서 확인)
I0934 = idx("09:34")               # 첫 5분이 확정되는 봉(09:34 봉, 마감 09:35)
SIGN = {"SOXL": 1, "SOXS": -1}
AXES = ["T", "G", "F", "M", "P"]
LEVELS = {"T": ["1", "2", "3"], "G": ["down", "flat", "up"], "F": ["up", "down"],
          "M": ["0", "1", "2"], "P": ["lo", "mid", "hi"]}
SIDES = ["both", "SOXL", "SOXS"]
KEYS = AXES + ["S"]
DEFAULTS = dict(T="any", G="any", F="any", M="any", P="any", S="both")

_GAP, _F5, _PR = {}, {}, {}


def tbin(i):
    c = i + 1                                      # 봉 i 마감 시각 색인
    return "1" if c <= idx("09:45") else ("2" if c <= idx("10:15") else "3")


def gap_bin(ctx, s):
    k = (ctx.day, s)
    g = _GAP.get(k)
    if g is None:
        pc = ctx.prev_day(s)["close"]
        g = (ctx.bars(s, "Open")[RTH0] / pc - 1) * 100
        g = _GAP[k] = "down" if g < -GAP_THR else ("flat" if g <= GAP_THR else "up")
    return g


def first5(ctx, s, i):
    if i < I0934:
        return None
    k = (ctx.day, s)
    f = _F5.get(k)
    if f is None:
        f = _F5[k] = "up" if ctx.bars(s, "Close")[I0934] >= ctx.bars(s, "Open")[RTH0] else "down"
    return f


def prev_rng_bin(ctx):
    b = _PR.get(ctx.day)
    if b is None:
        p = ctx.prev_day("SOXL")
        r = p["high"] / p["low"] - 1
        b = _PR[ctx.day] = "lo" if r <= PR_Q[0] else ("mid" if r <= PR_Q[1] else "hi")
    return b


class Fade3:
    start = "09:30"

    def __init__(self, p, mirror=False):
        self.p, self.mirror = p, mirror
        self.sides = ["SOXL", "SOXS"] if p["S"] == "both" else [p["S"]]
        self.useM = p["M"] != "any"
        self.st = "flat"
        self._reset(RTH0)

    def _reset(self, s0):
        self.s0 = s0
        self.L = {"SOXL": np.inf, "SOXS": np.inf}
        self.mlo = {"QQQ": np.inf, "NVDA": np.inf}
        self.mhi = {"QQQ": -np.inf, "NVDA": -np.inf}

    def gate(self, ctx, i, s):
        p = self.p
        if p["T"] != "any" and tbin(i) != p["T"]:
            return False
        if p["G"] != "any" and gap_bin(ctx, s) != p["G"]:
            return False
        if p["F"] != "any" and first5(ctx, s, i) != p["F"]:
            return False
        if p["P"] != "any" and prev_rng_bin(ctx) != p["P"]:
            return False
        if self.useM:
            n = 0
            for m in ("QQQ", "NVDA"):
                c = ctx.bars(m, "Close")[i]
                n += (c >= self.mlo[m] * (1 + MKT_THR)) if s == "SOXL" else (c <= self.mhi[m] * (1 - MKT_THR))
            if str(int(n)) != p["M"]:
                return False
        return True

    def on_bar(self, ctx, i):
        if self.st == "in":
            if ctx.pos != 0:
                return None
            self.st = "flat"                        # 봉 i에서 청산됨 → 다음 봉부터 L을 다시 잰다
            self._reset(i + 1)
            return None
        if i < self.s0:
            return None
        for s in ("SOXL", "SOXS"):
            lo = ctx.bars(s, "Low")[i]
            if lo < self.L[s]:
                self.L[s] = lo
        if self.useM:
            for m in ("QQQ", "NVDA"):
                lo, hi = ctx.bars(m, "Low")[i], ctx.bars(m, "High")[i]
                if lo < self.mlo[m]:
                    self.mlo[m] = lo
                if hi > self.mhi[m]:
                    self.mhi[m] = hi
        if i > ctx.cut - 16:                        # 진입 마감 = 창 끝 15분 전
            return None
        cand = []
        for s in self.sides:
            r = ctx.bars(s, "Close")[i] / self.L[s]
            if r >= 1 + X_RISE:
                cand.append((r, s))
        for _, s in sorted(cand, reverse=True):
            if self.gate(ctx, i, s):
                self.st = "in"
                buy = SIGN[s] if self.mirror else -SIGN[s]
                return Order(buy, target=TARGET)
        return None


def make(params, mirror=False):
    p = dict(DEFAULTS, **params)
    return lambda: Fade3(p, mirror)


# ---------------------------------------------------------------- 격자
def grid():
    out = {}
    for k in range(0, 3):
        for axes in itertools.combinations(AXES, k):
            for vals in itertools.product(*[LEVELS[a] for a in axes]):
                for s in SIDES:
                    p = dict(DEFAULTS, S=s, **dict(zip(axes, vals)))
                    out.setdefault(tuple(p[kk] for kk in KEYS), p)
    return out


def cid(p):
    return "L3-" + "-".join(f"{k}{p[k]}" for k in KEYS)


def ncond(p):
    return sum(p[a] != "any" for a in AXES) + (p["S"] != "both")


ADJ = {"T": [("any", "1"), ("any", "2"), ("any", "3"), ("1", "2"), ("2", "3")],
       "G": [("any", "down"), ("any", "flat"), ("any", "up"), ("down", "flat"), ("flat", "up")],
       "F": [("any", "up"), ("any", "down")],
       "M": [("any", "0"), ("any", "1"), ("any", "2"), ("0", "1"), ("1", "2")],
       "P": [("any", "lo"), ("any", "mid"), ("any", "hi"), ("lo", "mid"), ("mid", "hi")],
       "S": [("both", "SOXL"), ("both", "SOXS")]}
ADJMAP = {a: {} for a in ADJ}
for _a, _es in ADJ.items():
    for _u, _v in _es:
        ADJMAP[_a].setdefault(_u, []).append(_v)
        ADJMAP[_a].setdefault(_v, []).append(_u)


def neighbors(key, keys):
    out = []
    for j, a in enumerate(KEYS):
        for v in ADJMAP[a].get(key[j], []):
            nk = key[:j] + (v,) + key[j + 1:]
            if nk in keys:
                out.append(nk)
    return out


def flat(s, tr, pre):
    row = {f"{pre}_{k}": v for k, v in s.items() if k != "ci95"}
    row[f"{pre}_ci_lo"], row[f"{pre}_ci_hi"] = s.get("ci95", [np.nan, np.nan])
    n = len(tr)
    row[f"{pre}_avg_trade_pct"] = float(tr["ret"].mean() * 100) if n else np.nan
    row[f"{pre}_target_rate"] = float((tr["why"] == "target").mean()) if n else np.nan
    row[f"{pre}_avg_miss_pct"] = float(tr.loc[tr["why"] != "target", "ret"].mean() * 100) if (tr["why"] != "target").any() else np.nan
    row[f"{pre}_n_buy_soxs"] = int((tr["sym"] == "SOXS").sum()) if n else 0
    row[f"{pre}_n_buy_soxl"] = int((tr["sym"] == "SOXL").sum()) if n else 0
    if s["active_days"] == 0:
        row[f"{pre}_t"] = np.nan                    # 매매 0건: t 없음(이웃 중앙값에서 뺀다)
    return row


def detail(eng, p, mirror=False):
    daily, tr = eng.run(make(p, mirror), period="sel")
    s = summarize(daily)
    n = len(tr)
    s["avg_trade_pct"] = float(tr["ret"].mean() * 100) if n else None
    s["target_rate"] = float((tr["why"] == "target").mean()) if n else None
    miss = tr[tr["why"] != "target"]
    s["avg_miss_pct"] = float(miss["ret"].mean() * 100) if len(miss) else None
    s["worst_trade_pct"] = float(tr["ret"].min() * 100) if n else None
    s["exits"] = tr["why"].value_counts().to_dict() if n else {}
    s["buys_by_sym"] = tr["sym"].value_counts().to_dict() if n else {}
    if n:
        mon = pd.to_datetime(daily["day"]).dt.month
        h = {}
        for name, ms in [("1~3월", [1, 2, 3]), ("4~6월", [4, 5, 6])]:
            dm = daily[mon.isin(ms)]
            h[name] = {"days": int(len(dm)), "active_days": int((dm["n"] > 0).sum()), "mean_pct": float(dm["ret"].mean() * 100)}
        s["sel_halves"] = h
        top = (daily["ret"] * 100).sort_values(ascending=False)
        s["top3_days_share_of_sum"] = float(top.iloc[:3].sum() / top.sum()) if top.sum() > 0 else None
    return s, daily, tr


def main():
    t_start = time.time()
    D = Data()
    sel = D.days_in("sel")
    pr = [(lambda q: q["high"] / q["low"] - 1)(D.daily.loc[("SOXL", D.days[d - 1])]) for d in sel]
    q = np.quantile(pr, [1 / 3, 2 / 3])
    assert abs(q[0] - PR_Q[0]) < 1e-9 and abs(q[1] - PR_Q[1]) < 1e-9, q
    print(f"data {time.time() - t_start:.1f}s, sel {len(sel)}일", flush=True)
    G = grid()
    assert len(G) <= 400, len(G)
    assert len({cid(p) for p in G.values()}) == len(G)
    print("configs", len(G), flush=True)
    E = {"w90": Engine(D, scheme="w90"), "std": Engine(D, scheme="std")}
    rows, keys = [], list(G)
    for n, k in enumerate(keys):
        p = G[k]
        row = {"id": cid(p), **{kk: p[kk] for kk in KEYS}, "n_cond": ncond(p)}
        for sc, eng in E.items():
            daily, tr = eng.run(make(p), period="sel")
            assert np.isfinite(daily["ret"]).all(), (cid(p), sc)
            row.update(flat(summarize(daily), tr, sc))
        rows.append(row)
        if n % 25 == 0:
            print(n, f"{time.time() - t_start:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    kset = set(keys)
    pos = {k: n for n, k in enumerate(keys)}
    for sc in E:
        med, nn = [], []
        for k in keys:
            ts = [df.at[pos[q_], f"{sc}_t"] for q_ in [k] + neighbors(k, kset)]
            ts = [t for t in ts if np.isfinite(t)]
            med.append(float(np.median(ts)) if ts else np.nan)
            nn.append(len(neighbors(k, kset)))
        df[f"{sc}_nbr_med_t"] = med
        df[f"{sc}_n_nbr"] = nn
        df[f"{sc}_eligible"] = df[f"{sc}_active_days"] >= 20
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    df.to_csv(os.path.join(ROOT, "results", "lay_L3_sel.csv"), index=False, encoding="utf-8-sig")

    frozen, landscape = [], {}
    for sc in E:
        el = df[df[f"{sc}_eligible"]]
        ts = df[f"{sc}_t"].dropna()
        base = df[df["id"] == cid(DEFAULTS)].iloc[0]
        landscape[sc] = {"n_configs_with_trades": int(len(ts)), "n_eligible": int(len(el)),
                         "base_H0": {"mean_pct": float(base[f"{sc}_mean_pct"]), "t": float(base[f"{sc}_t"]),
                                     "active_days": int(base[f"{sc}_active_days"]), "trades": int(base[f"{sc}_trades"]),
                                     "nbr_med_t": float(base[f"{sc}_nbr_med_t"])},
                         "t_max_eligible": float(el[f"{sc}_t"].max()) if len(el) else None,
                         "t_median_eligible": float(el[f"{sc}_t"].median()) if len(el) else None,
                         "t_pos_frac_eligible": float((el[f"{sc}_t"] > 0).mean()) if len(el) else None,
                         "n_eligible_t_gt_2": int((el[f"{sc}_t"] > 2).sum()),
                         "nbr_med_max_eligible": float(el[f"{sc}_nbr_med_t"].max()) if len(el) else None,
                         "mean_pct_median_eligible": float(el[f"{sc}_mean_pct"].median()) if len(el) else None}
        if not len(el):
            continue
        best = el.assign(_neg_t=-el[f"{sc}_t"]).sort_values(
            [f"{sc}_nbr_med_t", "_neg_t", "n_cond", "id"], ascending=[False, True, True, True]).iloc[0]
        top5 = el.sort_values([f"{sc}_nbr_med_t", f"{sc}_t"], ascending=False).head(5)
        landscape[sc]["top5_by_score"] = [{"id": r["id"], "nbr_med_t": float(r[f"{sc}_nbr_med_t"]), "t": float(r[f"{sc}_t"]),
                                           "mean_pct": float(r[f"{sc}_mean_pct"]), "active_days": int(r[f"{sc}_active_days"])}
                                          for _, r in top5.iterrows()]
        if any(f["id"] == best["id"] for f in frozen):
            frozen[[f["id"] for f in frozen].index(best["id"])]["scheme"] += "+" + sc
            continue
        frozen.append({"id": best["id"], "scheme": sc})
    out = []
    id2p = {cid(p): p for p in G.values()}
    for f in frozen:
        p = dict(id2p[f["id"]])
        sel_s, mir, rec = {}, {}, {}
        for sc, eng in E.items():
            s, daily, tr = detail(eng, p)
            sm, dm, _ = detail(eng, p, mirror=True)
            s["nbr_med_t"] = float(df.loc[df["id"] == f["id"], f"{sc}_nbr_med_t"].iloc[0])
            dir_comp = (daily["ret"].values - dm["ret"].values) / 2 * 100
            s["direction_component_mean_pct"] = float(dir_comp.mean())
            sel_s[sc], mir[sc] = s, sm
            tr.to_csv(os.path.join(ROOT, "results", f"lay_L3_trades_{f['id']}_{sc}.csv"), index=False, encoding="utf-8-sig")
        k = tuple(p[kk] for kk in KEYS)
        nb = neighbors(k, kset)
        out.append({"id": f["id"], "scheme": f["scheme"], "params": p,
                    "factory": f"lay_L3.make({json.dumps(p, ensure_ascii=False)})",
                    "code": os.path.join(HERE, "lay_L3.py"),
                    "sel_w90": sel_s["w90"], "sel_std": sel_s["std"],
                    "mirror_sel": {"w90": mir["w90"], "std": mir["std"]},
                    "neighbors": [{"id": cid(G[q_]), "w90_t": float(df.at[pos[q_], "w90_t"]),
                                   "std_t": float(df.at[pos[q_], "std_t"]),
                                   "w90_active_days": int(df.at[pos[q_], "w90_active_days"]),
                                   "std_active_days": int(df.at[pos[q_], "std_active_days"])} for q_ in nb]})
    res = {"family": "L3", "n_configs": len(G), "period": "sel 2026-01-02~06-30 (123일)",
           "engine": "Engine(D, scheme='w90') / Engine(D, scheme='std'), fill='close', cost_table 1배",
           "signal": "H0: 오른 종목 1분 종가 ≥ L(09:30 또는 청산 다음 봉부터 저가 최저) × 1.03 → 반대 종목, +1% 지정가, 손절 없음, 창 끝 청산, 진입 마감 창 끝 15분 전",
           "gate": "맥락 조건은 문: H0 신호 참 & 맥락 참인 첫 봉에 진입, 막힌 봉은 넘어가고 L은 H0대로",
           "score": "sel 하루 평균 t(123일, 매매 없는 날 0), 자격 활동일 ≥ 20, 최종 = 자기+이웃 t 중앙값(매매 0건 설정 제외); 같으면 t, 조건 수, id",
           "grid": "T{1,2,3}·G{down,flat,up}·F{up,down}·M{0,1,2}·P{lo,mid,hi} 중 any 아닌 축 0~2개 (1+14+78=93) × S{both,SOXL,SOXS} = 279",
           "neighbors": "한 축 한 칸: any↔특정 값, 순서 축의 붙은 값(T 1-2-3, G down-flat-up, M 0-1-2, P lo-mid-hi); F up↔down, S SOXL↔SOXS 제외",
           "PR_Q": list(PR_Q), "gap_thr_pct": GAP_THR, "mkt_thr": MKT_THR,
           "landscape": landscape, "frozen": out, "runtime_s": round(time.time() - t_start, 1)}
    with open(os.path.join(ROOT, "results", "lay_L3_frozen.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({"landscape": landscape, "frozen": [{k: o[k] for k in ["id", "scheme", "params"]} for o in out]},
                     ensure_ascii=False, default=str), flush=True)
    print(f"done {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
