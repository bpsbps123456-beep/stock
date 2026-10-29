"""F1 시가 범위 돌파(ORB) 묶음 — BRIEF2.md '묶음' 표 F1, sel 고르기만.

핵심 규칙(반드시 포함, Zarattini·Aziz 2023 5분 ORB 꼴):
  처음 N분(09:30~09:30+N) 범위의 'N분 봉'(09:30 봉 시가 → N번째 봉 종가)이 양봉이면 SOXL, 음봉이면 SOXS, 같으면 쉼.
  손절 = 결정 가격(SOXL 종가)에서 범위 반대쪽 끝까지의 SOXL 기준 거리 %(= 1R). SOXS 보유에도 같은 %를 SOXS 체결가에 건다.
  마감(lib 엔진: actual 11:00/10:00, std 10:00)까지 보유.

넓힌 축(설정 300개):
  N        1·5·15·30
  entry    candle = N분 봉 마감 즉시 결정 / confirm = 같은 방향 범위 끝 밖 종가 마감을 기다려 결정(돌파 확인)
  tgt      목표 R배 1·2·5·10·없음(지정가)
  stop     손절 거리 배수 0.5·1·없음(무손절, 구조 진단 i의 '넓은 손절·무손절' 비교)
  rev      손절로 나가면 반대 종목으로 1회 바꾸기(같은 손절·목표 %)
  filt     band = 오늘 범위 폭이 지난 20거래일 같은 N분 범위 폭 평균의 0.5배 미만 또는 2배 초과면 쉼(전날까지 아는 값).
           격자를 300개 안에 두려고 band는 entry=candle에만 붙였다.
  (mirror  거울 대조용: 모든 주문의 SOXL↔SOXS를 뒤집는다. 격자에는 없다.)

결정은 봉 마감 때 ctx로 아는 값만 쓴다. 체결·비용·손절 실행은 lib.Engine이 한다(1분 지연 다음 봉 종가).
실행: py fam_F1.py  → results/fam_F1_sel.csv, results/fam_F1_frozen.json
"""
import json, os, sys, time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lib import Data, Engine, Order, summarize, RTH0  # noqa: E402

RES = os.path.normpath(os.path.join(HERE, "..", "results"))
FLT_L, FLT_LO, FLT_HI = 20, 0.5, 2.0

_WCACHE = {}


def _past_width(ctx, k, N):
    """k거래일 전 첫 N분 범위 폭(시가 대비 비율). 지난날 자료라 다 볼 수 있다."""
    key = (id(ctx._D), ctx.d - k, N)
    if key not in _WCACHE:
        h = ctx.prev_bars("SOXL", "High", k)
        if h is None:
            _WCACHE[key] = None
        else:
            l = ctx.prev_bars("SOXL", "Low", k)
            o = ctx.prev_bars("SOXL", "Open", k)[RTH0]
            _WCACHE[key] = float((h[RTH0:RTH0 + N].max() - l[RTH0:RTH0 + N].min()) / o)
    return _WCACHE[key]


class ORB:
    start = "09:30"

    def __init__(self, N=1, entry="candle", tgt=None, stop=1.0, rev=False, filt=None, mirror=False):
        self.N, self.entry, self.tgt, self.stop = N, entry, tgt, stop
        self.rev = bool(rev) and stop is not None
        self.filt, self.mirror = filt, mirror
        self.iN = RTH0 + N - 1          # 범위 마지막 봉
        self.state = "wait"             # wait → armed(confirm) → ordered → held → done
        self.dir = 0
        self.hi = self.lo = None
        self.side = 0                   # 실제 주문한 종목(+1 SOXL, −1 SOXS)
        self.sfrac = self.tfrac = None
        self.entry_px = None
        self.revd = False
        self.avgw = None

    def on_day(self, ctx):
        if self.filt:
            ws = [_past_width(ctx, k, self.N) for k in range(1, FLT_L + 1)]
            ws = [w for w in ws if w is not None and np.isfinite(w)]
            self.avgw = float(np.mean(ws)) if len(ws) >= FLT_L // 2 else None

    def _order(self, d, P):
        R = (P - self.lo) / P if d > 0 else (self.hi - P) / P
        self.sfrac = self.stop * R if self.stop else None
        self.tfrac = self.tgt * R if self.tgt else None
        self.side = -d if self.mirror else d
        self.state = "ordered"
        return Order(self.side, stop=self.sfrac, target=self.tfrac)

    def _exited(self, ctx, i, ref_px):
        """우리가 내지 않은 청산(손절·목표). 손절이고 rev면 반대로 1회 바꾼다."""
        sym = "SOXL" if self.side > 0 else "SOXS"
        stopped = (self.sfrac is not None and ref_px is not None
                   and ctx.bars(sym, "Low")[i] <= ref_px * (1 - self.sfrac))
        if stopped and self.rev and not self.revd:
            self.revd = True
            self.side = -self.side
            self.state = "ordered"
            return Order(self.side, stop=self.sfrac, target=self.tfrac)
        self.state = "done"
        return None

    def on_bar(self, ctx, i):
        st = self.state
        if st == "done" or i < self.iN:
            return None
        if st == "wait":                                   # i == iN: 범위 확정
            o = ctx.bars("SOXL", "Open")[RTH0]
            c = ctx.c("SOXL")[i]
            self.hi = float(ctx.bars("SOXL", "High")[RTH0:i + 1].max())
            self.lo = float(ctx.bars("SOXL", "Low")[RTH0:i + 1].min())
            self.dir = int(np.sign(c - o)) if np.isfinite(c - o) else 0
            if self.dir == 0:
                self.state = "done"
                return None
            if self.filt:
                if self.avgw is None or self.avgw <= 0:
                    self.state = "done"
                    return None
                ratio = (self.hi - self.lo) / o / self.avgw
                if ratio < FLT_LO or ratio > FLT_HI:
                    self.state = "done"
                    return None
            if self.entry == "candle":
                return self._order(self.dir, c)
            self.state = "armed"
            return None
        if st == "armed":
            c = ctx.c("SOXL")[i]
            if (self.dir > 0 and c > self.hi) or (self.dir < 0 and c < self.lo):
                return self._order(self.dir, c)
            return None
        if st == "ordered":
            if ctx.pos == self.side:                       # 체결됨
                self.entry_px = ctx.entry_px
                self.state = "held"
                return None
            if ctx.pos == 0:                               # fill='open'에서 체결 봉 안에 이미 나감(호가 뺀 시가로 근사)
                sym = "SOXL" if self.side > 0 else "SOXS"
                return self._exited(ctx, i, float(ctx.bars(sym, "Open")[i]))
            return None
        if st == "held":
            if ctx.pos == 0:
                return self._exited(ctx, i, self.entry_px)
            return None
        return None


def make(params):
    """Engine.run(make(params), ...)에 그대로 넣는다. 날마다 새 ORB 객체를 만든다."""
    p = dict(params)
    return lambda: ORB(**p)


# ---------------- 격자 ----------------
AX = {"N": [1, 5, 15, 30], "entry": ["candle", "confirm"], "tgt": [1, 2, 5, 10, None],
      "stop": [0.5, 1.0, None], "rev": [False, True], "filt": [None, "band"]}
KEYS = ["N", "entry", "tgt", "stop", "rev", "filt"]


def canon(p):
    p = dict(p)
    if p["stop"] is None:
        p["rev"] = False
    return p


def key(p):
    return tuple(p[k] for k in KEYS)


def grid():
    out = []
    for N in AX["N"]:
        for entry in AX["entry"]:
            for tgt in AX["tgt"]:
                for stop, rev in ((0.5, False), (0.5, True), (1.0, False), (1.0, True), (None, False)):
                    for filt in AX["filt"]:
                        if filt and entry != "candle":
                            continue
                        out.append(dict(N=N, entry=entry, tgt=tgt, stop=stop, rev=rev, filt=filt))
    return out


def neighbors(p, keyset):
    """한 축을 한 칸 움직인 설정(격자에 있는 것만). stop=없음이면 rev는 뜻이 없어 False로 맞춘다."""
    out = []
    for k in KEYS:
        vals = AX[k]
        j = vals.index(p[k])
        for jj in (j - 1, j + 1):
            if 0 <= jj < len(vals):
                q = canon({**p, k: vals[jj]})
                kq = key(q)
                if kq in keyset and kq != key(p) and kq not in [key(x) for x in out]:
                    out.append(q)
    return out


def pid(p):
    t = "none" if p["tgt"] is None else f"{p['tgt']}R"
    s = "none" if p["stop"] is None else f"{p['stop']:g}x"
    return f"N{p['N']}_{p['entry']}_t{t}_s{s}_{'rev' if p['rev'] else 'norev'}_{p['filt'] or 'all'}"


class Always:
    """대조(설명용): 09:31 결정 → 마감까지 한 종목."""
    start = "09:30"

    def __init__(self, pos):
        self.pos = pos

    def on_bar(self, ctx, i):
        return Order(self.pos) if i == RTH0 else None


def exit_mix(tr):
    if not len(tr):
        return {}
    w = tr["why"].value_counts().to_dict()
    return {"trades": int(len(tr)), "why": {k: int(v) for k, v in w.items()},
            "by_sym": {s: int(n) for s, n in tr["sym"].value_counts().items()},
            "trade_mean_pct": float(tr["ret"].mean() * 100),
            "trade_mean_by_why_pct": {k: float(g["ret"].mean() * 100) for k, g in tr.groupby("why")}}


def main():
    t0 = time.time()
    D = Data()
    print(f"Data {time.time() - t0:.1f}s, 날 {len(D.days)}")
    G = grid()
    keyset = {key(p) for p in G}
    assert len(G) == len(keyset) <= 300, len(G)
    sel_days = D.days_in("sel")
    rows, store = [], {}
    for scheme in ("actual", "std"):
        E = Engine(D, scheme=scheme)
        for n, p in enumerate(G):
            daily, tr = E.run(make(p), days=sel_days)
            s = summarize(daily)
            store[(scheme, key(p))] = (daily, tr, s)
            rows.append({"id": pid(p), "scheme": scheme, **{k: p[k] for k in KEYS}, **{
                k: s.get(k) for k in ["days", "active_days", "trades", "mean_pct", "sd_pct", "hit_days", "sum_pct",
                                      "compound_pct", "t", "p_one_sided", "max_dd_pct"]},
                "ci_lo": s["ci95"][0], "ci_hi": s["ci95"][1]})
            if n % 50 == 0:
                print(scheme, n, f"{time.time() - t0:.0f}s")
    df = pd.DataFrame(rows)
    # 이웃 포함 t 중앙값(원래 매개변수 dict로 찾는다; rows 순서 = scheme × G 순서)
    nb_med, nb_n = [], []
    for scheme in ("actual", "std"):
        for p in G:
            nb = neighbors(p, keyset)
            ts = [store[(scheme, key(p))][2]["t"]] + [store[(scheme, key(q))][2]["t"] for q in nb]
            nb_med.append(float(np.median(ts)))
            nb_n.append(len(nb))
    df["nb_median_t"], df["n_neighbors"] = nb_med, nb_n
    df["eligible"] = df["active_days"] >= 25
    os.makedirs(RES, exist_ok=True)
    df.to_csv(os.path.join(RES, "fam_F1_sel.csv"), index=False, encoding="utf-8-sig")

    # 고르기: 활동일 ≥ 25 중 이웃 t 중앙값 1위(같으면 자기 t)
    picks = {}
    for scheme in ("actual", "std"):
        e = df[(df.scheme == scheme) & df.eligible].sort_values(["nb_median_t", "t"], ascending=False)
        picks[scheme] = e.iloc[0]
    # 대조(설명용)
    base = {}
    for scheme in ("actual", "std"):
        E = Engine(D, scheme=scheme)
        for nm, pos in (("always_SOXL", 1), ("always_SOXS", -1)):
            dly, _ = E.run(lambda pos=pos: Always(pos), days=sel_days)
            base[f"{nm}_{scheme}"] = summarize(dly)

    idmap = {pid(p): p for p in G}
    assert len(idmap) == len(G)

    def params_of(r):
        return dict(idmap[r.id])

    frozen, seen = [], {}
    for scheme in ("actual", "std"):
        r = picks[scheme]
        p = params_of(r)
        k = key(p)
        if k in seen:
            seen[k]["scheme"] = "actual+std"
            seen[k]["id"] = "F1-AS"
            continue
        ent = {"id": "F1-A" if scheme == "actual" else "F1-S", "scheme": scheme, "params": p,
               "factory": f"fam_F1.make({json.dumps(p)})".replace("null", "None").replace("false", "False").replace("true", "True"),
               "code_path": os.path.join(HERE, "fam_F1.py").replace("\\", "/")}
        seen[k] = ent
        frozen.append(ent)

    for ent in frozen:
        p = ent["params"]
        ent["sel"], ent["sel_mirror"], ent["neighbor_median_t"], ent["exit_mix"], ent["sel_sensitivity"] = {}, {}, {}, {}, {}
        ent["sel_by_season"], ent["neighbors"] = {}, {}
        nbk = [key(q) for q in neighbors(p, keyset)]
        for scheme in ("actual", "std"):
            daily, tr, s = store[(scheme, key(p))]
            ent["sel"][scheme] = s
            ent["neighbor_median_t"][scheme] = float(df[(df.scheme == scheme) & (df.id == pid(p))].nb_median_t.iloc[0])
            ent["neighbors"][scheme] = {pid(dict(zip(KEYS, q))): round(float(store[(scheme, q)][2]["t"]), 3) for q in nbk}
            ent["exit_mix"][scheme] = exit_mix(tr)
            E = Engine(D, scheme=scheme)
            dm, _ = E.run(make({**p, "mirror": True}), days=sel_days)
            ent["sel_mirror"][scheme] = summarize(dm)
            d2, _ = Engine(D, scheme=scheme, cost_mult=2.0).run(make(p), days=sel_days)
            d3, _ = Engine(D, scheme=scheme, fill="open").run(make(p), days=sel_days)
            ent["sel_sensitivity"][scheme] = {"cost2x_mean_pct": summarize(d2)["mean_pct"], "cost2x_t": summarize(d2)["t"],
                                              "fill_open_mean_pct": summarize(d3)["mean_pct"], "fill_open_t": summarize(d3)["t"]}
            from lib import cutoff_idx, idx as _idx
            summer = np.array([cutoff_idx(d) == _idx("11:00") for d in daily["day"]])
            ent["sel_by_season"][scheme] = {
                "summer_11h": {"days": int(summer.sum()), "mean_pct": float(daily["ret"][summer].mean() * 100)},
                "winter_10h": {"days": int((~summer).sum()), "mean_pct": float(daily["ret"][~summer].mean() * 100)}}

    land = {}
    for scheme in ("actual", "std"):
        x = df[df.scheme == scheme]
        e = x[x.eligible]
        land[scheme] = {"n": int(len(x)), "eligible": int(len(e)), "t_max": float(e.t.max()), "t_median": float(e.t.median()),
                        "t_pos_frac": float((e.t > 0).mean()), "t_ge2_frac": float((e.t >= 2).mean()),
                        "nb_median_t_max": float(e.nb_median_t.max()),
                        "median_t_by_N": {int(k): float(v) for k, v in e.groupby("N").t.median().items()},
                        "median_t_by_entry": e.groupby("entry").t.median().to_dict(),
                        "median_t_by_stop": {str(k): float(v) for k, v in e.groupby(e.stop.fillna(-1)).t.median().items()},
                        "median_t_by_tgt": {str(k): float(v) for k, v in e.groupby(e.tgt.fillna(-1)).t.median().items()},
                        "median_t_by_rev": {str(k): float(v) for k, v in e.groupby("rev").t.median().items()},
                        "median_t_by_filt": {str(k): float(v) for k, v in e.groupby(e.filt.fillna("all")).t.median().items()},
                        "top10": e.sort_values("nb_median_t", ascending=False).head(10)[
                            ["id", "t", "nb_median_t", "mean_pct", "active_days"]].to_dict("records")}
    # 설명용 자르기(고르기에 쓰지 않는다): 무손절 N=1 candle 대 confirm, 손절 1배의 청산 구성
    explain = {}
    for scheme in ("actual", "std"):
        ca = store[(scheme, key(dict(N=1, entry="candle", tgt=None, stop=None, rev=False, filt=None)))][0]
        cf = store[(scheme, key(dict(N=1, entry="confirm", tgt=None, stop=None, rev=False, filt=None)))][0]
        skip = (cf.n.values == 0) & (ca.n.values > 0)
        both = (cf.n.values > 0) & (ca.n.values > 0)
        _, ts, s1 = store[(scheme, key(dict(N=1, entry="candle", tgt=None, stop=1.0, rev=False, filt=None)))]
        explain[scheme] = {
            "N1_candle_nostop_mean_pct": float(ca.ret.mean() * 100), "N1_confirm_nostop_mean_pct": float(cf.ret.mean() * 100),
            "confirm_skipped_days": int(skip.sum()), "candle_mean_on_skipped_pct": float(ca.ret[skip].mean() * 100),
            "candle_mean_on_both_pct": float(ca.ret[both].mean() * 100), "confirm_mean_on_both_pct": float(cf.ret[both].mean() * 100),
            "N1_candle_stop1x_mean_pct": s1["mean_pct"], "N1_candle_stop1x_exits": ts.why.value_counts().to_dict(),
            "N1_candle_stop1x_stopped_trade_mean_pct": float(ts[ts.why == "stop"].ret.mean() * 100) if (ts.why == "stop").any() else None}
    out = {"family": "F1", "brief": "BRIEF2.md", "period": "sel 2026-01-02~06-30", "n_configs": len(G), "explain_sel": explain,
           "n_runs": len(G) * 2, "grid": {k: [str(v) for v in AX[k]] for k in KEYS},
           "grid_note": "N 4 × entry 2 × tgt 5 × (stop 0.5·1 × rev 2 + 무손절 1) = 200, band 거르기는 entry=candle에만 4×5×5 = 100 → 300",
           "filter": {"L": FLT_L, "lo": FLT_LO, "hi": FLT_HI},
           "selection": "활동일 ≥ 25, 점수 = t, 최종 = 자기 + 격자 이웃(한 축 한 칸) t 중앙값, 같으면 자기 t",
           "engine": "lib.Engine fill=close, 비용표 1배(cost_table.json)", "baselines_sel": base,
           "landscape": land, "frozen": frozen}
    json.dump(out, open(os.path.join(RES, "fam_F1_frozen.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({s: {k: v for k, v in land[s].items() if k != "top10"} for s in land}, ensure_ascii=False, indent=1))
    for s in land:
        print(s, "top10")
        for r in land[s]["top10"]:
            print("  ", r)
    for ent in frozen:
        print(ent["id"], ent["params"], {s: (round(ent["sel"][s]["mean_pct"], 3), round(ent["sel"][s]["t"], 2),
                                            ent["sel"][s]["active_days"]) for s in ent["sel"]},
              "nb", ent["neighbor_median_t"], "mirror", {s: round(ent["sel_mirror"][s]["mean_pct"], 3) for s in ent["sel_mirror"]})
    print({k: round(v["mean_pct"], 3) for k, v in base.items()})
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
