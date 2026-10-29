"""F2 소음 경계 추세(Zarattini·Aziz·Barbon 2024 'Beat the Market'을 장초 창에 맞춤) — 전략·격자·고르기(sel만).

규칙(BRIEF2.md F2 + 오케스트레이터 핵심 규칙, 결과 보기 전 고정):
- sigma_t = 지난 L거래일 각 날의 |종가_t / 그날 09:30 시가 - 1| 평균(같은 분 t, ctx.prev_bars로 계산).
- 위 경계 UB = max(오늘 09:30 시가, 전날 정규장 종가) x (1 + k*sigma_t),
  아래 경계 LB = min(오늘 09:30 시가, 전날 정규장 종가) x (1 - k*sigma_t). 모두 SOXL 가격.
- 확인 시각(첫 확인 시각부터 m분마다, 시각 = 봉 마감)에 SOXL 종가 c를 본다.
  c > UB 이고 c가 롱 손절선 위면 SOXL, c < LB 이고 c가 숏 손절선 아래면 SOXS(반대 보유면 바꿔 산다),
  사이면 보유 유지 또는 현금. (진입 순간 이미 손절선 밖이면 사지 않는다: 바로 손절될 매수를 막는다.)
- 손절선(SOXL 가격): 롱 = band: UB / vwap: 오늘 09:30부터 VWAP / near: max(UB, VWAP)(둘 중 가까운 쪽, 원 논문).
  숏 = band: LB / vwap: VWAP / near: min(LB, VWAP).
  확인 시각마다 손절선을 새 값으로 고친다(따라가는 손절, Order(keep=True) -> 엔진이 1분 뒤부터 적용).
  확인 시각 사이에는 걸어 둔 손절이 봉 안에서 체결된다(엔진 규칙). 확인 시각에 종가가 이미 손절선을 넘었으면
  시장가로 판다(Order(0), 다음 봉 종가).
- SOXS 보유의 손절가 = SOXL 손절선까지의 거리(%)를 SOXS 현재가에 거울로 옮긴 값
  (구조 진단: 정규장 1분 상관 0.93~0.97, 기울기 0.99~1.01).
- 거울(mirror=True): 같은 시각에 반대 종목을 산다. 청산 사건(SOXL이 손절선을 넘음)은 원 전략과 같으므로
  보유 종목 기준으로는 손절이 아니라 지정가 목표가 된다.
- 남은 보유는 엔진이 마감 1분 전 봉 종가에 판다(actual: 서머타임 11:00·표준시 10:00, std: 10:00).

격자(결과 보기 전 고정): L(10,14,20) x k(0.5,0.75,1,1.5) x m(1,5,15) x 첫 확인(09:35,09:45,10:00) x 손절(band,near,vwap)
= 324에서 (첫 확인 10:00, m=15) 36칸을 뺀 288개(300 한도). 뺀 칸은 서머타임 날 확인 4번(10:00·10:15·10:30·10:45)뿐이고
std에서는 한 번도 확인하지 않는 가장 얇은 칸이다.
이웃 = 한 축만 한 칸 다른 설정(손절 축 순서 band - near - vwap: near는 두 선 중 가까운 쪽이라 가운데에 둔다).

사용: cd code && py fam_F2.py   (sel만, actual·std, fill='close', 비용표 1배)
"""
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from lib import Data, Engine, Order, NMIN, ROOT, RTH0, cutoff_idx, hm, idx, summarize  # noqa: E402

GRID = {"L": [10, 14, 20], "k": [0.5, 0.75, 1.0, 1.5], "m": [1, 5, 15],
        "first": ["09:35", "09:45", "10:00"], "stop": ["band", "near", "vwap"]}
AXES = list(GRID)
MIN_ACTIVE = 25

_SIG, _PC = {}, {}          # 날짜(·L)별 메모: 지난날 자료로만 만든 값이라 미래 참조가 아니다


def excluded(p):
    return p["first"] == "10:00" and p["m"] == 15


class NoiseBand:
    def __init__(self, L=14, k=1.0, m=1, first="09:35", stop="near", mirror=False):
        self.L, self.k, self.m, self.kind, self.mirror = int(L), float(k), int(m), stop, bool(mirror)
        self.first_i = idx(first) - 1            # 첫 확인 시각에 마감하는 봉
        self.start = hm(self.first_i)
        self.ok = False

    def on_day(self, ctx):
        key = (ctx.day, self.L)
        if key not in _SIG:
            acc, n = np.zeros(NMIN), 0
            for j in range(1, self.L + 1):
                c = ctx.prev_bars("SOXL", "Close", j)
                if c is None:
                    break
                o = ctx.prev_bars("SOXL", "Open", j)[RTH0]
                acc += np.abs(c / o - 1)
                n += 1
            _SIG[key] = acc / n if n == self.L else None
        if ctx.day not in _PC:
            p = ctx.prev_day("SOXL")
            _PC[ctx.day] = float(p["close"]) if p else np.nan
        o, pc = float(ctx.bars("SOXL", "Open")[RTH0]), _PC[ctx.day]
        self.sig = _SIG[key]
        self.hi, self.lo = max(o, pc), min(o, pc)
        self.ok = self.sig is not None and np.isfinite(o) and np.isfinite(pc)

    def levels(self, ctx, i):
        cL = float(ctx.c("SOXL")[i])
        s = self.k * self.sig[i]
        ub, lb = self.hi * (1 + s), self.lo * (1 - s)
        v = ctx.bars("SOXL", "Volume")[RTH0:]
        pv = ctx.bars("SOXL", "VWAP")[RTH0:]
        ok = np.isfinite(pv) & (v > 0)
        vw = float(np.dot(pv[ok], v[ok]) / v[ok].sum()) if ok.any() else cL
        if self.kind == "band":
            sl, ss = ub, lb
        elif self.kind == "vwap":
            sl, ss = vw, vw
        else:
            sl, ss = max(ub, vw), min(lb, vw)
        return cL, ub, lb, vw, sl, ss

    def on_bar(self, ctx, i):
        if not self.ok or i < self.first_i or (i - self.first_i) % self.m:
            return None
        cL, ub, lb, vw, sl, ss = self.levels(ctx, i)
        p = -ctx.pos if self.mirror else ctx.pos          # 원 신호 기준 보유 방향
        if cL > ub and cL > sl and p != 1:
            return self._order(ctx, i, 1, sl, cL, keep=False)
        if cL < lb and cL < ss and p != -1:
            return self._order(ctx, i, -1, ss, cL, keep=False)
        if p == 1:
            return Order(0) if cL <= sl else self._order(ctx, i, 1, sl, cL, keep=True)
        if p == -1:
            return Order(0) if cL >= ss else self._order(ctx, i, -1, ss, cL, keep=True)
        return None

    def _order(self, ctx, i, sgn, S, cL, keep):
        """sgn = 원 신호 방향, S = SOXL 가격 청산선. 보유 종목 가격의 손절(원) 또는 목표(거울) 비율로 바꾼다."""
        hold = -sgn if self.mirror else sgn
        if hold == 1:
            X, cur = S, cL
        else:
            cur = float(ctx.c("SOXS")[i])
            X = cur * (2 - S / cL)                          # SOXL이 S에 닿을 때의 SOXS 가격(거리 % 거울)
        if not (np.isfinite(X) and np.isfinite(cur)) or X <= 0:
            return None
        ref = ctx.entry_px if keep else cur
        f = (1 - X / ref) if not self.mirror else (X / ref - 1)
        f = f if f != 0 else 1e-12                          # 엔진은 0을 '없음'으로 읽는다
        return Order(hold, stop=f, keep=keep) if not self.mirror else Order(hold, target=f, keep=keep)


def make(params, mirror=False):
    """Engine.run(make(params), ...)에 그대로 넣는다. 날마다 새 전략 객체."""
    p = {a: params[a] for a in AXES}
    return lambda: NoiseBand(mirror=mirror, **p)


def cid(p):
    return f"L{p['L']}_k{p['k']}_m{p['m']}_f{p['first'].replace(':', '')}_{p['stop']}"


def configs():
    out = []
    for vals in itertools.product(*(GRID[a] for a in AXES)):
        p = dict(zip(AXES, vals))
        if not excluded(p):
            out.append(p)
    return out


def neighbors(p, have):
    out = []
    for a in AXES:
        vals = GRID[a]
        j = vals.index(p[a])
        for jj in (j - 1, j + 1):
            if 0 <= jj < len(vals):
                q = dict(p)
                q[a] = vals[jj]
                if cid(q) in have:
                    out.append(cid(q))
    return out


def flat(s, tr=None):
    out = {k: v for k, v in s.items() if k != "ci95"}
    if "ci95" in s:
        out["ci_lo"], out["ci_hi"] = s["ci95"]
    if tr is not None:
        why = tr["why"].value_counts() if len(tr) else {}
        for w in ("stop", "target", "signal", "cutoff"):
            out[f"n_{w}"] = int(why.get(w, 0)) if len(tr) else 0
        out["mean_trade_pct"] = float(tr["ret"].mean() * 100) if len(tr) else None
    return out


def diag(E, p, mirror=False):
    """얼린 후보 설명용(sel만): 진입 시각대별 손익, 청산 이유별 손익."""
    daily, tr = E.run(make(p, mirror), period="sel")
    out = {"summary": flat(summarize(daily), tr)}
    if len(tr):
        ent = tr["entry"].map(idx)
        tr = tr.assign(bucket=np.where(ent < idx("09:45"), "09:30-09:44",
                                       np.where(ent < idx("10:00"), "09:45-09:59", "10:00-")))
        out["by_entry_bucket"] = {b: {"n": int(len(g)), "mean_pct": float(g["ret"].mean() * 100),
                                      "sum_pct": float(g["ret"].sum() * 100), "hit": float((g["ret"] > 0).mean())}
                                  for b, g in tr.groupby("bucket")}
        out["by_exit"] = {w: {"n": int(len(g)), "mean_pct": float(g["ret"].mean() * 100),
                              "sum_pct": float(g["ret"].sum() * 100)} for w, g in tr.groupby("why")}
        out["by_sym"] = {s: {"n": int(len(g)), "mean_pct": float(g["ret"].mean() * 100),
                             "sum_pct": float(g["ret"].sum() * 100)} for s, g in tr.groupby("sym")}
        out["trades_per_active_day"] = float(len(tr) / max(1, (daily["n"] > 0).sum()))
    dst = np.array([cutoff_idx(d) == idx("11:00") for d in daily["day"]])
    h1 = np.array([d.month <= 3 for d in daily["day"]])
    r = daily["ret"].values * 100
    out["mean_pct_by_daytype"] = {"DST_11:00(79일)": float(r[dst].mean()), "EST_10:00(44일)": float(r[~dst].mean()),
                                  "1~3월": float(r[h1].mean()), "4~6월": float(r[~h1].mean())}
    return out, daily


def main():
    t0 = time.time()
    ct = os.path.join(ROOT, "cost_table.json")
    assert os.path.exists(ct), "cost_table.json 없음 — 기본값으로 재면 안 된다"
    D = Data()
    print(f"data {time.time() - t0:.1f}s", flush=True)
    E = {"actual": Engine(D), "std": Engine(D, scheme="std")}
    cfgs = configs()
    have = {cid(p) for p in cfgs}
    rows, dailies = [], {}
    for n, p in enumerate(cfgs):
        for sch, eng in E.items():
            daily, tr = eng.run(make(p), period="sel")
            s = flat(summarize(daily), tr)
            rows.append({"id": cid(p), "scheme": sch, **p, **s})
            dailies[(cid(p), sch)] = daily["ret"].values
        if n % 24 == 0:
            print(f"{n + 1}/{len(cfgs)} {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    # 이웃 포함 t 중앙값
    tmap = {(r["id"], r["scheme"]): r["t"] for r in rows}
    nb_med, nb_n = [], []
    for r in rows:
        p = {a: r[a] for a in AXES}
        ids = [cid(p)] + neighbors(p, have)
        ts = [tmap[(x, r["scheme"])] for x in ids]
        nb_med.append(float(np.median(ts)))
        nb_n.append(len(ids) - 1)
    df["nb_median_t"], df["n_neighbors"] = nb_med, nb_n
    df["eligible"] = df["active_days"] >= MIN_ACTIVE
    out_csv = os.path.join(ROOT, "results", "fam_F2_sel.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    picks = {}
    for sch in E:
        g = df[(df.scheme == sch) & df.eligible].sort_values(["nb_median_t", "t"], ascending=False)
        picks[sch] = g.iloc[0]["id"]
    land = {}
    for sch in E:
        g = df[df.scheme == sch]
        ge = g[g.eligible]
        top = ge.sort_values(["nb_median_t", "t"], ascending=False).head(10)
        land[sch] = {
            "n_configs": int(len(g)), "n_eligible": int(len(ge)),
            "t_max": float(g.t.max()), "t_median": float(g.t.median()), "t_min": float(g.t.min()),
            "share_t_pos": float((g.t > 0).mean()), "share_mean_pos": float((g.mean_pct > 0).mean()),
            "eligible_t_max": float(ge.t.max()) if len(ge) else None,
            "eligible_t_median": float(ge.t.median()) if len(ge) else None,
            "nb_median_t_max": float(ge.nb_median_t.max()) if len(ge) else None,
            "mean_pct_median": float(g.mean_pct.median()),
            "top10": top[["id", "t", "nb_median_t", "mean_pct", "active_days", "trades"]].to_dict("records"),
            "axis_median_t": {a: {str(v): float(g[g[a] == v].t.median()) for v in GRID[a]} for a in AXES},
            "axis_median_mean_pct": {a: {str(v): float(g[g[a] == v].mean_pct.median()) for v in GRID[a]}
                                     for a in AXES},
        }
    frozen = []
    for pid in dict.fromkeys(picks.values()):
        r = df[(df.id == pid)].set_index("scheme")
        p = {a: (r.loc["actual", a].item() if hasattr(r.loc["actual", a], "item") else r.loc["actual", a])
             for a in AXES}
        p = {"L": int(p["L"]), "k": float(p["k"]), "m": int(p["m"]), "first": str(p["first"]), "stop": str(p["stop"])}
        item = {"id": pid, "picked_for": [s for s in E if picks[s] == pid], "params": p,
                "factory": f"fam_F2.make({p!r})", "mirror_factory": f"fam_F2.make({p!r}, mirror=True)",
                "code": os.path.join(HERE, "fam_F2.py").replace("\\", "/")}
        nbs = neighbors(p, have)
        for sch in E:
            rr = r.loc[sch].to_dict()
            item[f"sel_{sch}"] = {k: (v.item() if hasattr(v, "item") else v) for k, v in rr.items()
                                   if k not in AXES and k not in ("id",)}
            item[f"neighbors_t_{sch}"] = {x: tmap[(x, sch)] for x in nbs}
            md, mtr = E[sch].run(make(p, mirror=True), period="sel")
            item[f"mirror_sel_{sch}"] = flat(summarize(md), mtr)
            # 같은 날 원+거울 합(비용·체결 비대칭 점검)
            item[f"orig_plus_mirror_mean_{sch}"] = float((dailies[(pid, sch)] + md["ret"].values).mean() * 100)
            dg, _ = diag(E[sch], p)
            item[f"diag_{sch}"] = dg
            for lab, eng in (("cost0", Engine(D, scheme=sch, cost_mult=0.0)),
                             ("cost2", Engine(D, scheme=sch, cost_mult=2.0)),
                             ("fill_open", Engine(D, scheme=sch, fill="open"))):
                dd_, _t = eng.run(make(p), period="sel")
                s_ = summarize(dd_, nboot=2000)
                item.setdefault(f"sens_{sch}", {})[lab] = {"mean_pct": s_["mean_pct"], "t": s_["t"]}
        frozen.append(item)
    res = {"family": "F2", "rule": "Zarattini·Aziz·Barbon 2024 소음 경계 추세, 장초 창", "grid": GRID,
           "excluded": "first=10:00 & m=15 (36칸)", "n_configs": len(cfgs), "period": "sel 2026-01-02~06-30",
           "engine": "lib.Engine fill=close, cost_table.json 1배, actual·std",
           "selection": "점수 = 하루 평균 수익 t, 활동일 >= 25, 최종 = 자기+한 축 한 칸 이웃 t 중앙값(동점은 자기 t)",
           "picks": picks, "landscape": land, "frozen": frozen,
           "files": {"sel_csv": out_csv.replace("\\", "/")},
           "runtime_s": round(time.time() - t0, 1)}
    out_json = os.path.join(ROOT, "results", "fam_F2_frozen.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({"picks": picks, "landscape": {s: {k: v for k, v in land[s].items()
                                                          if k not in ("top10",)} for s in land}},
                     ensure_ascii=False, indent=1, default=str))
    for it in frozen:
        print(it["id"], it["picked_for"])
        for sch in E:
            a = it[f"sel_{sch}"]
            print(f"  {sch}: mean {a['mean_pct']:+.3f} t {a['t']:+.2f} nb {a['nb_median_t']:+.2f} "
                  f"ci [{a['ci_lo']:+.2f},{a['ci_hi']:+.2f}] act {a['active_days']} tr {a['trades']} "
                  f"| mirror {it[f'mirror_sel_{sch}']['mean_pct']:+.3f} | sens {it[f'sens_{sch}']}")
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
