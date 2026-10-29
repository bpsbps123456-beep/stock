"""F3 갭·프리장 묶음 — BRIEF2.md 사전 등록 규칙의 sel 탐색(conf·aux는 보지 않는다).

시각 규약(structure.py와 같음): '시각 T에 결정' = T에 마감한 봉(색인 idx(T) − 1)까지 보고 on_bar에서 주문,
체결은 엔진 규칙대로 다음 봉 종가(1분 지연). 'T에 청산' = idx(T) − 2 봉에서 매도 주문 → idx(T) − 1 봉 종가(= T 가격)에 체결.
그래서 10:00 청산은 std판(10:00 마감)의 마감 청산과 같은 봉·같은 비용이다. 청산 시각이 마감 이후면 엔진이 마감에 판다.

네 하위 격자(kind), 설정 296개:
  A 갭 핵심(216): g = 09:30 시가 ÷ 전날 정규장 종가 − 1, |g| ≥ θ{0,2,4}%면 따라가기(g>0 SOXL) 또는 메우기(반대),
     결정 {09:31, 09:35, 09:45}, 청산 {10:00, 10:30, 마감}, 손절 {없음, 1%, 2%, 갭 메움 지점}.
     '갭 메움 지점': 따라가기는 SOXL이 전날 종가에 닿는 거리를 손절로, 메우기는 같은 거리를 목표(지정가)로 건다.
     거리는 결정 때 SOXL 가격 S 기준(SOXL 보유: 1 − P/S 또는 P/S − 1, SOXS 보유: 같은 거리를 SOXS % 로 근사).
     결정 때 이미 전날 종가를 넘어섰으면(갭이 메워졌으면) 이 칸은 매매하지 않는다.
  B 프리장 방향(48): 신호 종목 {SOXX, QQQ, NVDA}의 프리장 수익(창 시작 {04:00, 08:00} → 09:25 가격),
     부호대로 따라가기 또는 반대, 결정 {09:31, 09:35}, 청산 {10:00, 마감}, 손절 없음.
  C 프리장 진입(24): start 08:00, 결정 {09:00, 09:15, 09:25}, 신호 {gap: SOXL 결정 때 가격 ÷ 전날 종가 − 1,
     trend: SOXX 08:00 → 결정 때}, 따라가기 또는 반대, 청산 {09:45, 10:00}, 손절 없음.
  D 갭 × 프리장 일치(8): 갭 부호와 SOXX 프리장(08:00 → 09:25) 부호가 같은 날(same) 또는 다른 날(diff)만,
     갭 따라가기 또는 메우기, 09:31 결정, 청산 {10:00, 마감}, 손절 없음.

사용: py fam_F3.py  (sel만, 결과 results/fam_F3_sel.csv · results/fam_F3_frozen.json)
"""
import itertools
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd

from lib import Data, Engine, Order, INST, idx, summarize

RES = os.path.join(os.path.dirname(HERE), "results")


def _first_trade_open(ctx, sym, a, b):
    """색인 a..b(포함)에서 처음 거래된 봉의 시가. 없으면 nan."""
    v = ctx.bars(sym, "Volume")[a:b + 1]
    hit = np.nonzero(v > 0)[0]
    return float(ctx.bars(sym, "Open")[a + hit[0]]) if len(hit) else np.nan


def pre_ret(ctx, sym, wstart, end_i):
    """프리장 수익: 창 시작 가격 → end_i 봉 종가. 시작 가격 = 시작 직전 봉 종가(같은 날 앞 종가),
    없으면(그때까지 거래 없음) 창 안 첫 거래 봉 시가. 04:00 시작은 그날 첫 거래 봉 시가."""
    c = ctx.bars(sym, "Close")
    s = idx(wstart)
    p0 = c[s - 1] if s >= 1 else np.nan
    if not np.isfinite(p0):
        p0 = _first_trade_open(ctx, sym, s, end_i)
    p1 = c[end_i]
    if not (np.isfinite(p0) and np.isfinite(p1)) or p0 <= 0:
        return np.nan
    return p1 / p0 - 1


class GapPre:
    """한 날 한 번 결정, 시각 청산. ctx로만 자료를 본다."""

    def __init__(self, p):
        self.p = p
        self.kind = p["kind"]
        self.start = "08:00" if self.kind == "C" else "09:30"
        self.dec_i = idx(p["decide"]) - 1
        self.exit_i = None if p["exit"] == "cut" else idx(p["exit"]) - 2
        self.pc = None
        self.entered = False

    def on_day(self, ctx):
        pdv = ctx.prev_day("SOXL")
        self.pc = None if pdv is None else float(pdv["close"])

    def _signal(self, ctx, i):
        """(pos, stop, target) 또는 None."""
        p, k = self.p, self.kind
        mult = 1 if p["mode"] == "follow" else -1
        if k in ("A", "D"):
            if self.pc is None:
                return None
            g = ctx.bars("SOXL", "Open")[idx("09:30")] / self.pc - 1
            if not np.isfinite(g) or g == 0:
                return None
            if k == "A":
                if abs(g) * 100 < p["theta"]:
                    return None
            else:
                s = pre_ret(ctx, "SOXX", "08:00", idx("09:25") - 1)
                if not np.isfinite(s) or s == 0:
                    return None
                same = np.sign(s) == np.sign(g)
                if (p["agree"] == "same") != same:
                    return None
            pos = int(np.sign(g)) * mult
            stop = target = None
            if k == "A":
                st = p["stop"]
                if st in (1, 2):
                    stop = st / 100
                elif st == "gapfill":
                    S = ctx.bars("SOXL", "Close")[i]
                    dist = (1 - self.pc / S) if g > 0 else (self.pc / S - 1)   # SOXL이 전날 종가까지 가는 거리
                    if not np.isfinite(dist) or dist <= 0:                      # 이미 메워짐 → 매매 안 함
                        return None
                    if p["mode"] == "follow":
                        stop = dist
                    else:
                        target = dist
            return pos, stop, target
        if k == "B":
            s = pre_ret(ctx, p["sym"], p["pwin"], idx("09:25") - 1)
            if not np.isfinite(s) or s == 0:
                return None
            return int(np.sign(s)) * mult, None, None
        if k == "C":
            if p["sig"] == "gap":
                if self.pc is None:
                    return None
                S = ctx.bars("SOXL", "Close")[i]
                s = S / self.pc - 1 if np.isfinite(S) else np.nan
            else:
                s = pre_ret(ctx, "SOXX", "08:00", i)
            if not np.isfinite(s) or s == 0:
                return None
            return int(np.sign(s)) * mult, None, None
        raise ValueError(k)

    def on_bar(self, ctx, i):
        if i == self.dec_i and not self.entered:
            self.entered = True
            sig = self._signal(ctx, i)
            if sig is None:
                return None
            pos, stop, target = sig
            last = ctx.bars(INST[pos], "Close")[i]
            if not np.isfinite(last):                  # 아직 거래 없는 종목(SOXS 프리장 nan) → 다음 봉 체결가도 nan
                return None
            return Order(pos, stop=stop, target=target)
        if self.exit_i is not None and i == self.exit_i and ctx.pos != 0:
            return Order(0)
        return None


class Mirror:
    """거울: 같은 규칙에서 SOXL↔SOXS만 뒤집는다(손절·목표 비율은 그대로)."""

    def __init__(self, inner):
        self.inner = inner
        self.start = inner.start

    def on_day(self, ctx):
        ctx_pos = ctx.pos
        self.inner.on_day(ctx)
        ctx.pos = ctx_pos

    def on_bar(self, ctx, i):
        ctx.pos = -ctx.pos                     # 안쪽 전략에는 자기 방향으로 보이게
        try:
            o = self.inner.on_bar(ctx, i)
        finally:
            ctx.pos = -ctx.pos
        if o is None:
            return None
        if o.pos != 0 and not np.isfinite(ctx.bars(INST[-o.pos], "Close")[i]):
            return None
        return Order(-o.pos, stop=o.stop, target=o.target, keep=o.keep)


def make(params):
    """Engine.run(make(params), ...)에 그대로 넣는다(날마다 새 객체)."""
    p = dict(params)
    return lambda: GapPre(p)


def make_mirror(params):
    p = dict(params)
    return lambda: Mirror(GapPre(p))


# ───────── 격자 ─────────
DEC_A = ["09:31", "09:35", "09:45"]
EXIT_A = ["10:00", "10:30", "cut"]
STOP_ORDER = {"follow": ["none", "gapfill", 2, 1],      # 넓음 → 좁음(갭 메움 거리는 보통 2~4%)
              "fade": ["gapfill", "none", 2, 1]}        # 메우기의 gapfill = 손절 없음 + 목표
AXES = {"A": {"theta": [0, 2, 4], "decide": DEC_A, "exit": EXIT_A, "stop": None},
        "B": {"pwin": ["04:00", "08:00"], "decide": ["09:31", "09:35"], "exit": ["10:00", "cut"]},
        "C": {"decide": ["09:00", "09:15", "09:25"], "exit": ["09:45", "10:00"]},
        "D": {"exit": ["10:00", "cut"]}}


def grid():
    G = []
    for th, mode, dec, ex, st in itertools.product([0, 2, 4], ["follow", "fade"], DEC_A, EXIT_A,
                                                   ["none", 1, 2, "gapfill"]):
        G.append({"kind": "A", "theta": th, "mode": mode, "decide": dec, "exit": ex, "stop": st})
    for sym, pw, dec, ex, mode in itertools.product(["SOXX", "QQQ", "NVDA"], ["04:00", "08:00"],
                                                    ["09:31", "09:35"], ["10:00", "cut"], ["follow", "fade"]):
        G.append({"kind": "B", "sym": sym, "pwin": pw, "decide": dec, "exit": ex, "mode": mode})
    for sig, mode, dec, ex in itertools.product(["gap", "trend"], ["follow", "fade"],
                                                ["09:00", "09:15", "09:25"], ["09:45", "10:00"]):
        G.append({"kind": "C", "sig": sig, "mode": mode, "decide": dec, "exit": ex})
    for ag, mode, ex in itertools.product(["same", "diff"], ["follow", "fade"], ["10:00", "cut"]):
        G.append({"kind": "D", "agree": ag, "mode": mode, "decide": "09:31", "exit": ex})
    return G


def cid(p):
    return "F3" + p["kind"] + "|" + "|".join(f"{k}={p[k]}" for k in sorted(p) if k != "kind")


def neighbors(G):
    """바로 옆 격자 이웃: 순서 있는 축 하나만 한 칸 다른 설정(범주 축 mode·sym·sig·agree는 이웃 아님)."""
    key = {cid(p): k for k, p in enumerate(G)}
    out = []
    for p in G:
        nb = [key[cid(p)]]
        for ax, vals in AXES[p["kind"]].items():
            vals = STOP_ORDER[p["mode"]] if ax == "stop" else vals
            j = vals.index(p[ax])
            for jj in (j - 1, j + 1):
                if 0 <= jj < len(vals):
                    q = dict(p)
                    q[ax] = vals[jj]
                    nb.append(key[cid(q)])
        out.append(nb)
    return out


def main():
    t0 = time.time()
    D = Data()
    print(f"자료 {time.time() - t0:.1f}s, 날 {len(D.days)}", flush=True)
    sel = D.days_in("sel")
    engines = {"actual": Engine(D), "std": Engine(D, scheme="std")}
    G = grid()
    NB = neighbors(G)
    print(f"설정 {len(G)}개", flush=True)
    rows, dailies = [], {}
    for k, p in enumerate(G):
        for sch, E in engines.items():
            dl, tr = E.run(make(p), days=sel)
            s = summarize(dl)
            dailies[(k, sch)] = dl["ret"].values
            rows.append({"id": cid(p), "k": k, "scheme": sch, **{f"p_{a}": v for a, v in p.items()},
                         **{a: (json.dumps(v) if isinstance(v, list) else v) for a, v in s.items()},
                         "stops": int((tr["why"] == "stop").sum()) if len(tr) else 0,
                         "targets": int((tr["why"] == "target").sum()) if len(tr) else 0,
                         "soxs_trades": int((tr["sym"] == "SOXS").sum()) if len(tr) else 0})
        if k % 25 == 0:
            print(f"  {k}/{len(G)} {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    # 이웃 t 중앙값
    tmap = {(r.k, r.scheme): r.t for r in df.itertuples()}
    df["nb_n"] = [len(NB[r.k]) for r in df.itertuples()]
    df["nb_median_t"] = [float(np.median([tmap[(j, r.scheme)] for j in NB[r.k]])) for r in df.itertuples()]
    df["eligible"] = df["active_days"] >= 25
    os.makedirs(RES, exist_ok=True)
    df.to_csv(os.path.join(RES, "fam_F3_sel.csv"), index=False, encoding="utf-8-sig")
    print(f"격자 끝 {time.time() - t0:.0f}s", flush=True)

    # 고르기
    picks = {}
    for sch in ["actual", "std"]:
        e = df[(df.scheme == sch) & df.eligible].sort_values(["nb_median_t", "t"], ascending=False)
        picks[sch] = int(e.iloc[0]["k"])
    frozen = []
    for k in sorted(set(picks.values())):
        p = G[k]
        schemes = [s for s, kk in picks.items() if kk == k]
        rec = {"id": cid(p), "params": p, "factory": f"fam_F3.make({json.dumps(p, ensure_ascii=False)})",
               "picked_for": schemes, "neighbors": [cid(G[j]) for j in NB[k] if j != k]}
        for sch, E in engines.items():
            r = df[(df.k == k) & (df.scheme == sch)].iloc[0]
            dl, tr = E.run(make(p), days=sel)
            s = summarize(dl)
            s["nb_median_t"] = float(r.nb_median_t)
            s["nb_t"] = {cid(G[j]): float(tmap[(j, sch)]) for j in NB[k]}
            s["exits"] = tr["why"].value_counts().to_dict() if len(tr) else {}
            s["soxl_trades"] = int((tr["sym"] == "SOXL").sum()) if len(tr) else 0
            s["soxs_trades"] = int((tr["sym"] == "SOXS").sum()) if len(tr) else 0
            dm, _ = E.run(make_mirror(p), days=sel)
            s["mirror"] = summarize(dm)
            # sel 감도(고르기 구간 안에서만): 비용 2배, 시가 체결
            dc, _ = Engine(D, scheme=sch, cost_mult=2.0).run(make(p), days=sel)
            do, _ = Engine(D, scheme=sch, fill="open").run(make(p), days=sel)
            s["cost2x_mean_pct"] = summarize(dc, nboot=1)["mean_pct"]
            s["fill_open_mean_pct"] = summarize(do, nboot=1)["mean_pct"]
            rec[f"sel_{sch}"] = s
        frozen.append(rec)
    land = {}
    for sch in ["actual", "std"]:
        x = df[df.scheme == sch]
        e = x[x.eligible]
        land[sch] = {"n": int(len(x)), "eligible": int(len(e)), "t_max": float(e.t.max()),
                     "t_median": float(e.t.median()), "t_pos_share": float((e.t > 0).mean()),
                     "t_ge2": int((e.t >= 2).sum()), "nbmed_max": float(e.nb_median_t.max()),
                     "corr_t_nbmed": float(np.corrcoef(e.t, e.nb_median_t)[0, 1]),
                     "by_kind": {kk: {"n": int(len(g_)), "t_max": float(g_.t.max()), "t_median": float(g_.t.median()),
                                      "nbmed_max": float(g_.nb_median_t.max())}
                                 for kk, g_ in e.groupby("p_kind")}}
    out = {"family": "F3", "code": os.path.abspath(__file__), "n_configs": len(G),
           "grid": AXES | {"A_stop_order": STOP_ORDER}, "picks": {s: cid(G[k]) for s, k in picks.items()},
           "frozen": frozen, "landscape": land, "sel_days": len(sel),
           "rule": "점수 = sel 하루 평균 수익(비용 1배, fill=close) t, 활동일 ≥ 25, 최종 = 자신+바로 옆 이웃 t 중앙값"}
    json.dump(out, open(os.path.join(RES, "fam_F3_frozen.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=str)
    print(json.dumps({"picks": out["picks"], "landscape": land}, ensure_ascii=False, indent=1))
    print(f"끝 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
