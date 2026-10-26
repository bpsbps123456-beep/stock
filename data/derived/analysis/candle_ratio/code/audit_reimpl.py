"""감사(audit) 1: engine.py를 쓰지 않고 파이썬 반복문으로 다시 계산해 격자 결과와 맞춰 본다.

- 분봉 묶음: 5분·1시간 원자료를 시간순으로 한 줄씩 읽어 (날짜, (분−세션 시작)//tf) 묶음이 바뀔 때 새 봉을 연다.
- 비율: 봉마다 창(지금 봉 포함 최근 N봉)을 잘라 math.fsum으로 직접 더한다(이동 합 누적 없음).
- 모의매매: 현금·주식 장부(곱셈)로 체결하고, 구간 수익은 구간 첫 봉 시가(체결 전) 평가액 비로 잰다.
- 보유 기준은 원자료 5분봉(1시간봉) 시가·종가에서 바로 구한다.
사용: py code/audit_reimpl.py   → results/audit_reimpl_cells.csv, results/audit_reimpl_sample.csv, results/audit_reimpl_summary.json
"""
from __future__ import annotations

import json
import math
import pathlib
import random
from datetime import date

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
ET = "America/New_York"
SESS = {"eth": (240, 1200), "rth": (570, 960)}
WIN = {  # BRIEF.md 판정 구간
    "5m": dict(warm=date(2026, 7, 17), sel=(date(2026, 7, 17), date(2026, 9, 4)), conf=(date(2026, 9, 8), date(2026, 10, 22))),
    "1h": dict(warm=date(2026, 5, 1), sel=(date(2026, 6, 1), date(2026, 8, 14)), conf=(date(2026, 8, 17), date(2026, 10, 22))),
}
LAST_DAY = date(2026, 10, 22)
TOL = 1e-6


# ---------------------------------------------------------------------------
# 자료와 분봉 묶음
# ---------------------------------------------------------------------------

def load_raw(sym, base):
    d = pd.read_parquet(ROOT / "cache" / f"{sym}_{base}.parquet")
    idx = d.index.tz_convert(ET).to_pydatetime()
    rows = list(zip(idx, d.Open.tolist(), d.High.tolist(), d.Low.tolist(), d.Close.tolist(), d.Volume.tolist()))
    return [r for r in rows if r[0].date() <= LAST_DAY]


def make_bars(raw, tf, sess, warm):
    s0, s1 = SESS[sess]
    bars, key = [], None
    for t, o, h, l, c, v in raw:
        m = t.hour * 60 + t.minute
        if m < s0 or m >= s1:
            continue
        k = (t.date(), (m - s0) // tf)
        if k != key:
            bars.append({"day": k[0], "k": k[1], "t0": t, "t1": t, "O": o, "H": h, "L": l, "C": c, "V": v, "n": 1})
            key = k
        else:
            b = bars[-1]
            b["H"] = max(b["H"], h); b["L"] = min(b["L"], l); b["C"] = c; b["V"] += v; b["n"] += 1; b["t1"] = t
    bars = [b for b in bars if b["day"] >= warm]
    for i, b in enumerate(bars):
        b["lod"] = i == len(bars) - 1 or bars[i + 1]["day"] != b["day"]
    return bars


def seg_of(day, win):
    if win["sel"][0] <= day <= win["sel"][1]:
        return 0
    if win["conf"][0] <= day <= win["conf"][1]:
        return 1
    return -1


# ---------------------------------------------------------------------------
# 비율 (창을 매번 잘라 직접 더함)
# ---------------------------------------------------------------------------

def ratio_direct(bars, n, kind, body_flat=0.5):
    """body_flat: 창이 모두 도지(Σ|C−O|=0)일 때 body 값. 사양상 중립 0.5. 엔진은 0.75가 된다(감사 보고)."""
    out = []
    for i in range(len(bars)):
        if i < n - 1:
            out.append(None)
            continue
        w = bars[i - n + 1:i + 1]
        if kind == "count":
            up = sum(1 for b in w if b["C"] > b["O"])
            dn = sum(1 for b in w if b["C"] < b["O"])
            out.append(up / (up + dn) if up + dn else 0.5)
        elif kind == "body":
            num = math.fsum(b["C"] - b["O"] for b in w)
            den = math.fsum(abs(b["C"] - b["O"]) for b in w)
            out.append((1.0 + num / den) / 2.0 if den > 0 else body_flat)
        elif kind == "vol":
            num = math.fsum(b["V"] for b in w if b["C"] > b["O"])
            den = math.fsum(b["V"] for b in w if b["C"] != b["O"])
            out.append(num / den if den > 0 else 0.5)
        else:
            raise ValueError(kind)
    return out


# ---------------------------------------------------------------------------
# 장부식 모의매매
# ---------------------------------------------------------------------------

def ledger(bars, p, seg, b, s, follow, dayflat, cost, reentry="engine"):
    """reentry='engine': dayflat 매도 뒤 상태가 살아 있으면 다음 날 첫 봉 시가에 다시 산다(엔진 동작).
    reentry='spec': BRIEF 문구대로 다음 날 첫 봉 마감에 신호가 살아 있을 때 둘째 봉 시가에 다시 산다."""
    n = len(bars)
    cash, sh = 1.0, 0.0
    state = 0
    flat_block = False
    vopen = [0.0] * n
    ent = [0, 0]; hold = [0, 0]; nb = [0, 0]
    trades = []  # (봉 번호, 'B'|'S'|'SC', 가격, 신호 봉 번호)
    for j in range(n):
        bj = bars[j]
        O, C = bj["O"], bj["C"]
        vopen[j] = sh * O if sh > 0 else cash
        want = state
        if reentry == "spec" and flat_block:
            want = 0
        flat_block = False
        if want == 1 and sh == 0:
            sh = cash * (1.0 - cost) / O; cash = 0.0
            trades.append((j, "B", O, j - 1))
            if seg[j] >= 0:
                ent[seg[j]] += 1
        elif want == 0 and sh > 0:
            cash = sh * O * (1.0 - cost); sh = 0.0
            trades.append((j, "S", O, j - 1))
        if seg[j] >= 0:
            hold[seg[j]] += int(sh > 0); nb[seg[j]] += 1
        if sh > 0 and dayflat and bj["lod"]:
            cash = sh * C * (1.0 - cost); sh = 0.0
            trades.append((j, "SC", C, j))
            flat_block = state == 1
        pj = p[j]
        if pj is not None:
            if state == 0:
                if (follow and pj > b) or ((not follow) and pj < 1.0 - b):
                    state = 1
            elif (follow and pj < s) or ((not follow) and pj > 1.0 - s):
                state = 0
    final = sh * bars[-1]["C"] if sh > 0 else cash
    marks = vopen[1:] + [final]  # 봉 j 처리 뒤 평가액 = 다음 봉 시가(체결 전) 또는 마지막 종가
    i0 = seg.index(0); i1 = seg.index(1)

    def mdd(idx, base):
        pk, worst = 0.0, 0.0
        for j in idx:
            x = math.log(marks[j] / base)
            pk = max(pk, x)
            worst = min(worst, x - pk)
        return worst

    js0 = [j for j in range(n) if seg[j] == 0]
    js1 = [j for j in range(n) if seg[j] == 1]
    return dict(ret_sel=math.log(vopen[i1] / vopen[i0]), ret_conf=math.log(final / vopen[i1]),
                ent_sel=ent[0], ent_conf=ent[1], hold_sel=hold[0], hold_conf=hold[1], bars_sel=nb[0], bars_conf=nb[1],
                mdd_sel=mdd(js0, vopen[i0]), mdd_conf=mdd(js1, vopen[i1]), mdd_all=mdd(js0 + js1, vopen[i0]),
                trades=trades)


def bh_raw(raw, sess, win):
    """원자료에서 바로: 고르기 = 확인 첫 세션 봉 시가 / 고르기 첫 세션 봉 시가, 확인 = 마지막 세션 봉 종가 / 확인 첫 시가."""
    s0, s1 = SESS[sess]
    rr = [r for r in raw if s0 <= r[0].hour * 60 + r[0].minute < s1 and r[0].date() >= win["warm"]]
    first_sel = next(r for r in rr if r[0].date() >= win["sel"][0])
    first_conf = next(r for r in rr if r[0].date() >= win["conf"][0])
    last = rr[-1]
    return math.log(first_conf[1] / first_sel[1]), math.log(last[4] / first_conf[1])


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------

NAMED = [  # (이름, 창, tf, sess, N, ratio, dir, b, s, dayflat, cost)
    ("원안 eth 밤보유", "5m", 40, "eth", 10, "count", "follow", 0.5, 0.5, False, 0.0),
    ("원안 eth dayflat", "5m", 40, "eth", 10, "count", "follow", 0.5, 0.5, True, 0.0),
    ("원안 rth 밤보유", "5m", 40, "rth", 10, "count", "follow", 0.5, 0.5, False, 0.0),
    ("원안 rth dayflat", "5m", 40, "rth", 10, "count", "follow", 0.5, 0.5, True, 0.0),
    ("고르기 1위 70분 eth N15 vol fade", "5m", 70, "eth", 15, "vol", "fade", 0.8, 0.2, False, 0.0),
    ("5분 rth N4 vol follow 밤보유", "5m", 5, "rth", 4, "vol", "follow", 0.5, 0.3, False, 0.0),
    ("5분 rth N4 vol follow dayflat", "5m", 5, "rth", 4, "vol", "follow", 0.5, 0.3, True, 0.0),
    ("295분 rth N3 count fade 밤보유", "5m", 295, "rth", 3, "count", "fade", 0.5, 0.5, False, 0.0),
    ("295분 rth N3 count fade dayflat", "5m", 295, "rth", 3, "count", "fade", 0.5, 0.5, True, 0.0),
    ("1h창 고르기 1위 180분 eth N6 vol fade", "1h", 180, "eth", 6, "vol", "fade", 0.8, 0.3, False, 0.0),
    ("1h창 60분 rth N10 count follow", "1h", 60, "rth", 10, "count", "follow", 0.5, 0.5, False, 0.0),
    ("body 40분 eth N10 follow", "5m", 40, "eth", 10, "body", "follow", 0.5, 0.5, False, 0.0),
    ("vol 40분 eth N10 follow dayflat", "5m", 40, "eth", 10, "vol", "follow", 0.5, 0.5, True, 0.0),
    ("비용 0.1% 원안 eth 밤보유", "5m", 40, "eth", 10, "count", "follow", 0.5, 0.5, False, 0.001),
]
CMP = ["ret_sel", "ret_conf", "ent_sel", "ent_conf", "mdd_all", "hold_sel", "hold_conf", "bars_sel", "bars_conf", "mdd_sel", "mdd_conf"]


class Cache:
    def __init__(self, sym):
        self.sym = sym
        self.raw = {}
        self.bars = {}
        self.p = {}

    def get_raw(self, base):
        if base not in self.raw:
            self.raw[base] = load_raw(self.sym, base)
        return self.raw[base]

    def get_bars(self, base, tf, sess):
        k = (base, tf, sess)
        if k not in self.bars:
            bars = make_bars(self.get_raw(base), tf, sess, WIN[base]["warm"])
            self.bars[k] = (bars, [seg_of(b["day"], WIN[base]) for b in bars])
        return self.bars[k]

    def get_p(self, base, tf, sess, n, kind, body_flat=0.5):
        k = (base, tf, sess, n, kind, body_flat)
        if k not in self.p:
            self.p[k] = ratio_direct(self.get_bars(base, tf, sess)[0], n, kind, body_flat)
        return self.p[k]


def check_cell(cache, grids, base, tf, sess, n, kind, d, b, s, dayflat, cost):
    g = grids[base]
    row = g[(g.tf == tf) & (g.sess == sess) & (g.N == n) & (g.ratio == kind) & (g.dir == d) & (np.isclose(g.b, b))
            & (np.isclose(g.s, s)) & (g.dayflat == dayflat) & (np.isclose(g.cost, cost))]
    assert len(row) == 1, (base, tf, sess, n, kind, d, b, s, dayflat, cost, len(row))
    row = row.iloc[0]
    bars, seg = cache.get_bars(base, tf, sess)
    p = cache.get_p(base, tf, sess, n, kind)
    r = ledger(bars, p, seg, b, s, d == "follow", dayflat, cost)
    out = dict(base=base, tf=tf, sess=sess, N=n, ratio=kind, dir=d, b=b, s=s, dayflat=dayflat, cost=cost)
    worst = 0.0
    for c in CMP:
        out[f"eng_{c}"] = float(row[c]); out[f"re_{c}"] = float(r[c])
        diff = abs(float(row[c]) - float(r[c]))
        out[f"d_{c}"] = diff
        if c in ("ret_sel", "ret_conf", "mdd_all", "mdd_sel", "mdd_conf"):
            worst = max(worst, diff)
        elif diff > 0:
            worst = max(worst, 1.0)
    out["max_diff"] = worst
    out["match"] = worst <= TOL
    # 체결 시점 점검: 매수·시가 매도는 신호 봉(j−1) 다음 봉 시가, 종가 매도는 dayflat 마지막 봉
    tr = r["trades"]
    out["fills_next_open"] = all(sig == j - 1 for j, kind_, px, sig in tr if kind_ in ("B", "S"))
    out["sc_on_lod_only"] = all(bars[j]["lod"] for j, kind_, px, sig in tr if kind_ == "SC")
    if dayflat:
        rs = ledger(bars, p, seg, b, s, d == "follow", dayflat, cost, reentry="spec")
        out["spec_ret_sel"] = rs["ret_sel"]; out["spec_ret_conf"] = rs["ret_conf"]
        out["spec_ent_sel"] = rs["ent_sel"]; out["spec_ent_conf"] = rs["ent_conf"]
    if kind == "body":  # 엔진 방식(모두 도지 → 0.75)으로 다시 재 보고 차이가 그것 때문인지 본다
        pe = cache.get_p(base, tf, sess, n, kind, body_flat=0.75)
        re_ = ledger(bars, pe, seg, b, s, d == "follow", dayflat, cost)
        out["body075_ret_sel"] = re_["ret_sel"]; out["body075_ret_conf"] = re_["ret_conf"]
    return out


def main():
    grids = {b: pd.read_parquet(ROOT / "results" / f"grid_MU_{b}.parquet") for b in ("5m", "1h")}
    cache = Cache("MU")
    rows = []
    for name, base, tf, sess, n, kind, d, b, s, dayflat, cost in NAMED:
        o = check_cell(cache, grids, base, tf, sess, n, kind, d, b, s, dayflat, cost)
        o["name"] = name
        rows.append(o)
        print(f"{name:40s} match={o['match']} maxdiff={o['max_diff']:.2e} sel {o['eng_ret_sel']:+.6f}/{o['re_ret_sel']:+.6f}"
              f" conf {o['eng_ret_conf']:+.6f}/{o['re_ret_conf']:+.6f} ent {o['eng_ent_sel']:.0f}/{o['re_ent_sel']:.0f},"
              f"{o['eng_ent_conf']:.0f}/{o['re_ent_conf']:.0f}", flush=True)
    cells = pd.DataFrame(rows)
    cells.to_csv(ROOT / "results" / "audit_reimpl_cells.csv", index=False, encoding="utf-8-sig")

    # 무작위 표본: 5m 창 400칸 + 1h 창 200칸 (비용 0·민감도 섞음), 시드 20261023
    rng = random.Random(20261023)
    samp = []
    for base, k in (("5m", 400), ("1h", 200)):
        g = grids[base]
        idx = rng.sample(range(len(g)), k)
        sub = g.iloc[sorted(idx, key=lambda i: (g.tf.iat[i], g.sess.iat[i], g.N.iat[i], g.ratio.iat[i]))]
        for _, r in sub.iterrows():
            o = check_cell(cache, grids, base, int(r.tf), r.sess, int(r.N), r.ratio, r.dir, float(r.b), float(r.s), bool(r.dayflat), float(r.cost))
            samp.append(o)
        print(base, "표본", k, "칸 끝", flush=True)
    sp = pd.DataFrame(samp)
    sp.to_csv(ROOT / "results" / "audit_reimpl_sample.csv", index=False, encoding="utf-8-sig")

    # 보유 기준: 원자료 가격비 vs 격자 bh
    bhc = []
    for base in ("5m", "1h"):
        for sess in ("eth", "rth"):
            a, c = bh_raw(cache.get_raw(base), sess, WIN[base])
            g = grids[base]; gg = g[g.sess == sess]
            bhc.append(dict(base=base, sess=sess, raw_sel=a, raw_conf=c, eng_sel_min=float(gg.bh_sel.min()), eng_sel_max=float(gg.bh_sel.max()),
                            eng_conf_min=float(gg.bh_conf.min()), eng_conf_max=float(gg.bh_conf.max())))
    summ = dict(
        named=dict(n=len(cells), match=int(cells.match.sum()), mismatch=cells.loc[~cells.match, "name"].tolist()),
        sample=dict(n=len(sp), match=int(sp.match.sum()),
                    mismatch_by=sp.loc[~sp.match].groupby(["base", "sess", "ratio"]).size().reset_index().to_dict("records")),
        fills_next_open=bool(cells.fills_next_open.all() and sp.fills_next_open.all()),
        sc_on_lod_only=bool(cells.sc_on_lod_only.all() and sp.sc_on_lod_only.all()),
        bh=bhc,
    )
    (ROOT / "results" / "audit_reimpl_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
