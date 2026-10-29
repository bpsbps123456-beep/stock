"""F4 되돌림 묶음(사용자 가설 '원웨이는 드물다') — 전략·격자·고르기(sel만).

규칙(BRIEF2 F4, 격자 실행 전에 정함):
- 벗어남: SOXL 지금 봉 종가 C와 기준(오늘 09:30 시가 또는 세션 VWAP)의 차이.
  z = ln(C/기준) / (s1 * sqrt(경과분)). s1 = 정규장 1분 로그수익(첫 값 ln(C0930/O0930))의 최근 30개 표준편차
  (모자라면 있는 만큼, 5개 미만이면 신호 없음), 경과분 = 09:30부터 지금 봉 마감까지 분.
  % 판은 (C/기준 - 1)*100. 세션 VWAP = 09:30~지금 봉의 (봉 VWAP x 거래량) 합 / 거래량 합.
- 위로 문턱 이상 벗어나면 SOXS(-1), 아래로 벗어나면 SOXL(+1). 거울(mirror=True)은 같은 신호에 반대 종목.
- 결정 시작: 지금 시각(봉 마감) >= start. 새 진입은 마감 7분 전 봉까지(5분 이상 들 수 있게).
- 첫 진입은 상태(|벗어남| >= 문턱)로 들어간다. 청산 뒤 다시 들어가려면 |벗어남| < 문턱인 봉을 한 번 거쳐야 한다(재무장).
  같은 날 최대 maxent번.
- 청산(먼저 오는 것): 기준선 복귀(SOXL 종가가 기준선을 다시 넘음, VWAP는 지금 값), 시간 손절 hold분
  (체결 봉부터 hold분 뒤 봉 종가), 손절 stop·목표 target(보유 종목 체결가 대비, 엔진이 봉 안에서 확인), 마감.
- RSI(선택): SOXL 1분 Wilder RSI14, 정규장 봉만 날을 넘어 이어 붙임(연장 끔, 앞 2거래일 정규장으로 시작).
  SOXS 진입은 RSI >= R, SOXL 진입은 RSI <= 100 - R, 벗어남 조건과 함께(그리고).
전략은 ctx로만 자료를 본다. 특징 캐시는 (날짜, 봉) 키로 그 봉까지의 값만 담는다(설정끼리 공유해 시간을 아낀다).
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
from lib import Data, Engine, Order, idx, hm, summarize, RTH0, RTH1, ROOT  # noqa: E402

L_SIG, MIN_N, RSI_N = 30, 5, 14
Z_LEVELS = [1.5, 2.0, 2.5, 3.0]
P_LEVELS = [1.0, 2.0, 3.0, 4.0]
DEFAULTS = dict(ref="open", unit="z", level=1, start="10:00", hold=30,
                stop=None, target=None, rsi=None, maxent=1)

_F, _RS, _R = {}, {}, {}


def feats(ctx, i):
    k = (ctx.day, i)
    f = _F.get(k)
    if f is not None:
        return f
    seg = ctx.c("SOXL")[RTH0:i + 1]
    o0 = ctx.bars("SOXL", "Open")[RTH0]
    n = len(seg)
    lr = np.empty(n)
    lr[0] = np.log(seg[0] / o0)
    lr[1:] = np.diff(np.log(seg))
    w = lr[-L_SIG:]
    sig = float(w.std(ddof=1)) if len(w) >= MIN_N else np.nan
    v = ctx.bars("SOXL", "Volume")[RTH0:i + 1]
    pv = ctx.bars("SOXL", "VWAP")[RTH0:i + 1]
    sv = float(v.sum())
    vw = float((pv * v).sum() / sv) if sv > 0 else float(seg[-1])
    C = float(seg[-1])
    den = sig * np.sqrt(n)
    ok = np.isfinite(den) and den > 0
    f = {"C": C, "open": float(o0), "vwap": vw,
         "z_open": np.log(C / o0) / den if ok else np.nan,
         "z_vwap": np.log(C / vw) / den if ok else np.nan,
         "p_open": (C / o0 - 1) * 100, "p_vwap": (C / vw - 1) * 100}
    _F[k] = f
    return f


def _rsi_seed(ctx):
    p2, p1 = ctx.prev_bars("SOXL", "Close", 2), ctx.prev_bars("SOXL", "Close", 1)
    s = np.concatenate([p2[RTH0:RTH1], p1[RTH0:RTH1]])
    s = s[np.isfinite(s)]
    d = np.diff(s)
    g, l = np.clip(d, 0, None), np.clip(-d, 0, None)
    ag, al = g[:RSI_N].mean(), l[:RSI_N].mean()
    a = 1.0 / RSI_N
    for x, y in zip(g[RSI_N:], l[RSI_N:]):
        ag += a * (x - ag)
        al += a * (y - al)
    return ag, al, s[-1]


def rsi_at(ctx, i):
    k = (ctx.day, i)
    r = _R.get(k)
    if r is not None:
        return r
    seed = _RS.get(ctx.day)
    if seed is None:
        seed = _RS[ctx.day] = _rsi_seed(ctx)
    ag, al, last = seed
    a = 1.0 / RSI_N
    for x in ctx.c("SOXL")[RTH0:i + 1]:
        dl = x - last
        ag += a * (max(dl, 0.0) - ag)
        al += a * (max(-dl, 0.0) - al)
        last = x
    r = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    _R[k] = r
    return r


class Fade:
    def __init__(self, p, mirror=False):
        self.p, self.m = p, (-1 if mirror else 1)
        self.t0 = idx(p["start"]) - 1                    # 첫 결정 봉(마감 시각 = start)
        self.start = hm(self.t0)
        self.thr = (Z_LEVELS if p["unit"] == "z" else P_LEVELS)[p["level"]]
        self.key = ("z_" if p["unit"] == "z" else "p_") + p["ref"]
        self.st, self.dir, self.n, self.armed = "flat", 0, 0, True

    def on_bar(self, ctx, i):
        p = self.p
        if self.st == "entering":
            self.st = "in"
        if self.st == "exiting" or (self.st == "in" and ctx.pos == 0):
            self.st, self.armed = "flat", False
        if i < self.t0:
            return None
        f = feats(ctx, i)
        if self.st == "in":
            ref = f["open"] if p["ref"] == "open" else f["vwap"]
            back = f["C"] >= ref if self.dir > 0 else f["C"] <= ref
            if back or (i + 1) - ctx.entry_i >= p["hold"]:
                self.st = "exiting"
                return Order(0)
            return None
        x = f[self.key]
        big = bool(np.isfinite(x) and abs(x) >= self.thr)
        if not big:
            self.armed = True
            return None
        if not self.armed or self.n >= p["maxent"] or i > ctx.cut - 7:
            return None
        d = -1 if x > 0 else 1
        if p["rsi"]:
            r = rsi_at(ctx, i)
            if not (r >= p["rsi"] if d < 0 else r <= 100 - p["rsi"]):
                return None
        self.dir, self.st = d, "entering"
        self.n += 1
        return Order(d * self.m, stop=p["stop"], target=p["target"])


def make(params, mirror=False):
    p = dict(DEFAULTS, **params)
    return lambda: Fade(p, mirror)


# ---------------------------------------------------------------- 격자
AX = {"level": [0, 1, 2, 3], "start": ["09:35", "09:45", "10:00"], "hold": [10, 20, 30, 60],
      "stop": [None, 0.02, 0.01], "target": [None, 0.015, 0.01, 0.005],
      "rsi": [None, 80, 85, 90], "maxent": [1, 2]}
KEYS = ["ref", "unit", "level", "start", "hold", "stop", "target", "rsi", "maxent"]


def grid():
    out = {}

    def add(**kw):
        p = dict(DEFAULTS, **kw)
        out.setdefault(tuple(p[k] for k in KEYS), p)
    # A 핵심: 기준 2 x 단위 2 x 문턱 4 x 시작 3 x 시간 손절 4 = 192 (기준선 복귀 + 시간 손절, 손절·목표 없음, 하루 1번)
    for ref, unit, lv, st, h in itertools.product(["open", "vwap"], ["z", "pct"], AX["level"], AX["start"], AX["hold"]):
        add(ref=ref, unit=unit, level=lv, start=st, hold=h)
    # B 손절·목표: 기준 2 x z2 x 시작(09:45·10:00) x 시간(30·60) x 손절 3 x 목표 4 = 96, A와 겹침 8 -> 88
    for ref, st, h, sp, tg in itertools.product(["open", "vwap"], ["09:45", "10:00"], [30, 60], AX["stop"], AX["target"]):
        add(ref=ref, unit="z", level=1, start=st, hold=h, stop=sp, target=tg)
    # C RSI 극단: RSI 80·85·90 x 기준 2 x z1.5 x 시작(09:45·10:00) x 시간 30 = 12
    for r, ref, st in itertools.product([80, 85, 90], ["open", "vwap"], ["09:45", "10:00"]):
        add(ref=ref, unit="z", level=0, start=st, hold=30, rsi=r)
    # D 하루 최대 2번: 기준 2 x z2 x 시작(09:45·10:00) x 시간(10·20) = 8
    for ref, st, h in itertools.product(["open", "vwap"], ["09:45", "10:00"], [10, 20]):
        add(ref=ref, unit="z", level=1, start=st, hold=h, maxent=2)
    return out


def cid(p):
    f = lambda v: "N" if v is None else (f"{v * 100:g}" if isinstance(v, float) else str(v))
    return (f"F4-{p['ref'][0]}{p['unit'][0]}{p['level']}-{p['start'].replace(':', '')}-h{p['hold']}"
            f"-s{f(p['stop'])}-t{f(p['target'])}-r{f(p['rsi'])}-m{p['maxent']}")


def neighbors(key, keys):
    """순서 있는 축 하나를 한 칸 옮긴 설정(격자에 있는 것만). 기준·단위는 범주라 이웃이 아니다."""
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


def flat(s, tr, pre):
    row = {f"{pre}_{k}": v for k, v in s.items() if k != "ci95"}
    row[f"{pre}_ci_lo"], row[f"{pre}_ci_hi"] = s.get("ci95", [np.nan, np.nan])
    row[f"{pre}_avg_trade_pct"] = float(tr["ret"].mean() * 100) if len(tr) else np.nan
    for w in ["signal", "stop", "target", "cutoff"]:
        row[f"{pre}_x_{w}"] = int((tr["why"] == w).sum()) if len(tr) else 0
    row[f"{pre}_n_soxl"] = int((tr["sym"] == "SOXL").sum()) if len(tr) else 0
    return row


def main():
    t_start = time.time()
    D = Data()
    print(f"data {time.time() - t_start:.1f}s", flush=True)
    G = grid()
    assert len(G) <= 300, len(G)
    assert len({cid(p) for p in G.values()}) == len(G)
    print("configs", len(G), flush=True)
    E = {"actual": Engine(D), "std": Engine(D, scheme="std")}
    rows, keys = [], list(G)
    for n, k in enumerate(keys):
        p = G[k]
        row = {"id": cid(p), **{kk: p[kk] for kk in KEYS}}
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
            nb = neighbors(k, kset)
            ts = [df.at[pos[k], f"{sc}_t"]] + [df.at[pos[q], f"{sc}_t"] for q in nb]
            med.append(float(np.median(ts)))
            nn.append(len(nb))
        df[f"{sc}_nbr_med_t"] = med
        df[f"{sc}_n_nbr"] = nn
        df[f"{sc}_eligible"] = df[f"{sc}_active_days"] >= 25
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    df.to_csv(os.path.join(ROOT, "results", "fam_F4_sel.csv"), index=False, encoding="utf-8-sig")

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
                         "mean_pct_median_eligible": float(el[f"{sc}_mean_pct"].median()) if len(el) else None}
        if not len(el):
            continue
        best = el.sort_values([f"{sc}_nbr_med_t", f"{sc}_t"], ascending=False).iloc[0]
        if any(f["id"] == best["id"] for f in frozen):
            frozen[[f["id"] for f in frozen].index(best["id"])]["scheme"] += "+" + sc
            continue
        frozen.append({"id": best["id"], "scheme": sc, "row": best})
    out = []
    id2p = {cid(p): p for p in G.values()}
    for f in frozen:
        b = f["row"]
        p = dict(id2p[f["id"]])
        sel, mir, trd = {}, {}, {}
        for sc, eng in E.items():
            daily, tr = eng.run(make(p), period="sel")
            sel[sc] = summarize(daily)
            sel[sc]["avg_trade_pct"] = float(tr["ret"].mean() * 100) if len(tr) else None
            sel[sc]["exits"] = tr["why"].value_counts().to_dict() if len(tr) else {}
            sel[sc]["entries_by_sym"] = tr["sym"].value_counts().to_dict() if len(tr) else {}
            sel[sc]["nbr_med_t"] = float(b[f"{sc}_nbr_med_t"])
            dm, trm = eng.run(make(p, mirror=True), period="sel")
            mir[sc] = summarize(dm)
            trd[sc] = tr
        k = tuple(p[kk] for kk in KEYS)
        nb = neighbors(k, kset)
        out.append({"id": f["id"], "scheme": f["scheme"], "params": p,
                    "factory": f"fam_F4.make({json.dumps(p)})".replace("null", "None"),
                    "code": os.path.join(HERE, "fam_F4.py"),
                    "sel_actual": sel["actual"], "sel_std": sel["std"],
                    "mirror_sel": {"actual": mir["actual"], "std": mir["std"]},
                    "neighbors": [{"id": cid(G[q]), "actual_t": float(df.at[pos[q], "actual_t"]),
                                   "std_t": float(df.at[pos[q], "std_t"])} for q in nb]})
        for sc in E:
            trd[sc].to_csv(os.path.join(ROOT, "results", f"fam_F4_trades_{f['id']}_{sc}.csv"), index=False, encoding="utf-8-sig")
    res = {"family": "F4", "n_configs": len(G), "period": "sel 2026-01-02~06-30",
           "engine": "Engine(D) / Engine(D, scheme='std'), fill='close', cost_table 1x",
           "score": "t of daily mean (all sel days), eligible active_days>=25, final = median t of config + grid neighbors (one ordered axis one step; ref/unit categorical)",
           "grid": "A 192 = ref{open,vwap} x unit{z,pct} x level4 (z 1.5/2/2.5/3, % 1/2/3/4) x start{09:35,09:45,10:00} x hold{10,20,30,60}; "
                   "B 88 = ref2 x z2 x start{09:45,10:00} x hold{30,60} x stop{none,2%,1%} x target{none,1.5%,1%,0.5%} minus 8 overlap; "
                   "C 12 = RSI{80,85,90} x ref2 x z1.5 x start{09:45,10:00} x hold30; D 8 = maxent2 x ref2 x z2 x start{09:45,10:00} x hold{10,20}",
           "landscape": landscape, "frozen": out, "runtime_s": round(time.time() - t_start, 1)}
    with open(os.path.join(ROOT, "results", "fam_F4_frozen.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({"landscape": landscape, "frozen": [{k: o[k] for k in ["id", "scheme", "params"]} for o in out]},
                     ensure_ascii=False, default=str), flush=True)
    print(f"done {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
