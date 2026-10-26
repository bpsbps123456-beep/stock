"""6단계 V_TRAIL15 관점 A(2026-10-11): 단일 종목 2배 '추적 15% 손절 → 판 날 + 20거래일 시가 재매수' 독립 재현·이웃·시점.

사용자(2026-10-11): "오 너가 말한것중에 하나가 좋은데 단일종목2배레버리지는 20일 거래일 뒤에사라고했는데..."
갈래 p4_LEVSINGLE의 35개 사전 고정 규칙 중 하나(대표 후보 아님, 독립 검증 없음)를 다시 잰다.

독립성: lev_lab·stock_lab·trade_lab·p4_LEVSINGLE을 import하지 않는다. cache/p4_lev.parquet·p4_meta.json만 읽는다.
  비교용으로 results/p4_LEVSINGLE.json(갈래 숫자)을 읽기만 한다.

정의(갈래 문서의 문장으로부터 다시 짬)
  자료   : 배당 수정(Adj Close/Close 비율을 시가·고가·저가에도 곱함). Close·Open·Low > 0, 30행 미만 종목 제외.
  대상   : p4_meta.lev에서 L=2이고 기초가 QQQ·SPY·SOXX가 아닌 기초 24개 − 자료 오류(하루 종가 +150% 초과 또는 −70% 미만) = 23개.
           IONQ·RGTI·OKLO는 합병 전 SPAC 구간을 자른다(2021-10-01·2022-03-02·2024-05-10부터).
  합성   : r2[t] = 2·r_u[t] − rf[t]/252 − (보수 1.05% + 추가 비용)/252, rf = ^IRX/100(기초 날짜에 맞춰 앞값 채움, 없으면 0).
           시가·고가·저가 = 전일 합성 종가 × (1 + 2·(기초 시·고·저/기초 전일 종가 − 1)), 0.01배 하한, 고가·저가는 시가·종가로 감쌈.
  추가 비용: 실제 2배 상품과 '기본 합성(추가 비용 0)'이 겹치는 마지막 253개 날짜(252일 수익)의 연 로그 추적차
           −(ln 실제 배율 − ln 합성 배율)/연수. 상품이 여럿이면 중앙값, 0 미만은 0, 어떤 상품도 겹침 60일 미만이면 연 10%.
  규칙   : 창 시작 직전 종가에 상품 전량. 손절선 = (산 뒤 최고 종가, t−1까지; 산 날은 체결가) × (1 − X).
           t일 시가 ≤ 선이면 시가, 저가 ≤ 선이면 선 가격에 전량 매도(장중). 판 날 e로부터 t − e ≥ W인 첫 날 시가에 가격 무관 전량 재매수.
           같은 날 순서: 시가 재매수 → 손절 점검. 판 돈은 현금(이자 0). 수수료 0(desk_system §1).
  시점 변형: delay      = 장중 선 닿음을 감지한 다음 날 시가에 매도(재매수는 그 매도일 + W, 즉 하루 뒤)
             close_next = 종가 ≤ 선이면 다음 날 시가 매도(재매수 = 매도일 + W)
             close_same = 종가 ≤ 선이면 그 종가에 매도(참고)
             slip       = 체결마다 0.5% 불리(매도 ×0.995, 매수 ×1.005)
  평가   : 효율선초과 = 규칙 수익 − (규칙보다 낙폭이 깊지 않은 고정 배수 k 보유의 최고 수익), k = 0~2(0.05 간격).
           k ≤ 1: k·기초 + 현금. k > 1: k·r_u − (k−1)·rf/252 − (k−1)·(보수+추가 비용)/252(일 −99% 하한).
           실제 상품 대조에서는 k > 1 = (k−1)·실제 상품 + (2−k)·기초.
  판 1회 = 판 가격 ÷ 되산 가격 − 1. 창 안에 판 것만, 창 끝까지 못 되샀으면 창 끝 종가로 값을 매긴다.
  상품-해 = 2015~2025 각 해 + 2026(1/2~10/9) 들고 시작한 (기초, 해). 창 시작 전 기초 자료 126일 미만이면 뺀다.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"

# ============================================================================ 사전 고정(결과를 보기 전에 정함)
L = 2
FEE = 0.0105
TRACK_DAYS, TRACK_MIN, EXTRA_FALLBACK = 252, 60, 0.10
SPAC_START = {"IONQ": "2021-10-01", "RGTI": "2022-03-02", "OKLO": "2024-05-10"}
INDEX_UNDER = {"QQQ", "SPY", "SOXX"}
MIN_HIST = 126
BASE = (0.15, 20)
NB_X = (0.10, 0.12, 0.15, 0.18, 0.20, 0.25)          # 손절 %
NB_W = (5, 10, 15, 20, 25, 30, 40)                   # 대기 거래일
K_GRID = np.round(np.arange(0, L + 1e-9, 0.05), 2)   # 41개
SLIP = 0.005
DEATH = -0.80
INSURE_MAX = 0.02                                    # '전멸 보험' 성립: −80% 상품-해 비율 ≤ 2%
# 이웃 고르기(사전 고정): 보험 성립 이웃 가운데 2026 앞 효율선초과 중앙 최대 → 뒤·2025로 확인
TIMING = {"기본(장중 손절)": ("intra", 0.0), "하루 늦춘 판": ("delay", 0.0), "종가 판단 → 다음 시가": ("close_next", 0.0),
          "종가 판단 → 그 종가(참고)": ("close_same", 0.0), "미끄러짐 0.5%": ("intra", SLIP),
          "하루 늦춤 + 미끄러짐": ("delay", SLIP), "종가 판단 + 미끄러짐": ("close_next", SLIP)}

RUNS = {"2026": ("2026-01-02", "2026-10-09"), "2025": ("2025-01-02", "2025-12-31"), "2024": ("2024-01-02", "2024-12-31"),
        "2023": ("2023-01-03", "2023-12-29"), "2022": ("2022-01-03", "2022-12-30"), "2021": ("2021-01-04", "2021-12-31"),
        "2020": ("2020-01-02", "2020-12-31"), "2019": ("2019-01-02", "2019-12-31"), "2018": ("2018-01-02", "2018-12-31"),
        "2017": ("2017-01-01", "2017-12-31"), "2016": ("2016-01-01", "2016-12-31"), "2015": ("2015-01-01", "2015-12-31")}
WIN = {"2026앞": ("2026", "2026-01-02", "2026-05-29"), "2026뒤": ("2026", "2026-06-01", "2026-10-09"),
       "2026": ("2026", "2026-01-02", "2026-10-09")}
for _r, (_a, _b) in RUNS.items():
    if _r != "2026":
        WIN[_r] = (_r, _a, _b)
WIN["2018Q4"] = ("2018", "2018-10-01", "2018-12-31")
WIN["2020급락"] = ("2020", "2020-02-19", "2020-03-31")
YEAR_WINS = [str(y) for y in range(2015, 2027)]
FULL = ("2015-01-02", "2026-10-09")
MAIN_W = ("2026앞", "2026뒤", "2025")


# ============================================================================ 자료
def load():
    df = pd.read_parquet(CACHE / "p4_lev.parquet")
    irx = df[df.sym == "^IRX"].set_index("date")["Close"].astype(float).sort_index()
    irx = irx[~irx.index.duplicated()] / 100.0
    d = {}
    for sym, g in df[df.sym != "^IRX"].sort_values(["sym", "date"]).groupby("sym"):
        g = g.dropna(subset=["Open", "High", "Low", "Close"])
        g = g[(g.Close > 0) & (g.Open > 0) & (g.Low > 0)]
        if len(g) < 30:
            continue
        f = (g["Adj Close"] / g["Close"]).fillna(1.0).to_numpy()
        d[sym] = {"days": g.date.to_numpy().astype("U10"), "o": g.Open.to_numpy() * f, "h": g.High.to_numpy() * f,
                  "l": g.Low.to_numpy() * f, "c": g.Close.to_numpy() * f}
    meta = json.loads((CACHE / "p4_meta.json").read_text(encoding="utf-8"))["lev"]
    return d, irx, meta


def idx(days, day, side="left"):
    return int(np.searchsorted(days, day, side=side))


def synth(u, rf, cost):
    n = len(u["c"])
    c, o, h, l = np.empty(n), np.empty(n), np.empty(n), np.empty(n)
    c[0] = o[0] = h[0] = l[0] = 100.0
    uc, uo, uh, ul = u["c"], u["o"], u["h"], u["l"]
    for t in range(1, n):
        pc = uc[t - 1]
        r = L * (uc[t] / pc - 1) - (L - 1) * rf[t] / 252 - cost / 252
        c[t] = c[t - 1] * max(1 + r, 0.01)
        o[t] = c[t - 1] * max(1 + L * (uo[t] / pc - 1), 0.01)
        hi = c[t - 1] * max(1 + L * (uh[t] / pc - 1), 0.01)
        lo = c[t - 1] * max(1 + L * (ul[t] / pc - 1), 0.01)
        h[t], l[t] = max(hi, o[t], c[t]), min(lo, o[t], c[t])
    return {"days": u["days"], "o": o, "h": h, "l": l, "c": c}


def cut(b, i0):
    return {k: (v[i0:].copy() if isinstance(v, np.ndarray) else v) for k, v in b.items()}


def extra_cost(u, basic, d, meta):
    rows = []
    for p, info in meta.items():
        if info["under"] != u or info["L"] != L or p not in d:
            continue
        a = d[p]
        com = np.intersect1d(a["days"], basic["days"])
        if len(com) < TRACK_MIN:
            rows.append({"product": p, "겹침": int(len(com)), "drag": None})
            continue
        full_n = len(com)
        com = com[-(TRACK_DAYS + 1):]
        ia, is_ = np.searchsorted(a["days"], com), np.searchsorted(basic["days"], com)
        yrs = (len(com) - 1) / 252
        drag = -(np.log(a["c"][ia][-1] / a["c"][ia][0]) - np.log(basic["c"][is_][-1] / basic["c"][is_][0])) / yrs
        rows.append({"product": p, "겹침": int(full_n), "drag": float(drag)})
    vals = [r["drag"] for r in rows if r["drag"] is not None]
    return (max(0.0, float(np.median(vals))) if vals else EXTRA_FALLBACK), rows


def prepare():
    d, irx, meta = load()
    unders = sorted({v["under"] for v in meta.values() if v["L"] == L} - INDEX_UNDER)
    excluded = []
    U = {}
    for u in unders:
        b = d[u]
        r = b["c"][1:] / b["c"][:-1] - 1
        if np.any(r > 1.5) or np.any(r < -0.7):
            excluded.append(u)
            continue
        rf = irx.reindex(b["days"]).ffill().fillna(0.0).to_numpy()
        basic = synth(b, rf, FEE)
        extra, track = extra_cost(u, basic, d, meta)
        cal = synth(b, rf, FEE + extra)
        i0 = idx(b["days"], SPAC_START[u]) if u in SPAC_START else 0
        U[u] = {"und": cut(b, i0), "rf": rf[i0:], "extra": extra, "track": track,
                "syn": {"cal": cut(cal, i0), "basic": cut(basic, i0)}}
    return U, excluded, d, meta


# ============================================================================ 규칙 엔진(직접 작성)
def sim(o, h, l, c, a, z, X, W, mode="intra", slip=0.0):
    """a−1 종가에 전량(가치 1). 반환 (가치[z−a+2], 보유 여부[z−a+1], 주기[(판 t, 되산 t|None, 판 가격, 되산 가격)])."""
    sh, cash, held, peak = 1.0 / c[a - 1], 0.0, True, c[a - 1]
    exit_t, exit_px, pend, pk_det = -10 ** 9, None, False, None
    vals, hold, cyc = [1.0], [], []
    depth, nxt = [], []                                    # 진단: 매도가/최고 종가 − 1(미끄러짐 전), 장중 매도 다음 날 시가/매도가 − 1
    for t in range(a, z + 1):
        if held:
            if pend:                                       # 전날 감지 → 오늘 시가 매도
                depth.append(o[t] / pk_det - 1)
                exit_px = o[t] * (1 - slip)
                cash, sh, held, exit_t, pend = sh * exit_px, 0.0, False, t, False
        elif t - exit_t >= W:                              # 판 날 + W거래일 시가, 가격 무관
            fill = o[t] * (1 + slip)
            sh, cash, held, peak = cash / fill, 0.0, True, fill
            cyc.append((exit_t, t, exit_px, fill))
        if held and not pend:
            line = peak * (1 - X)
            if mode == "intra":
                px = o[t] if o[t] <= line else (line if l[t] <= line else None)
                if px is not None:
                    depth.append(px / peak - 1)
                    if t + 1 < len(o):
                        nxt.append(o[t + 1] / px - 1)
                    exit_px = px * (1 - slip)
                    cash, sh, held, exit_t = sh * exit_px, 0.0, False, t
            elif mode == "delay":
                if o[t] <= line or l[t] <= line:
                    pend, pk_det = True, peak
            elif mode == "close_next":
                if c[t] <= line:
                    pend, pk_det = True, peak
            elif mode == "close_same":
                if c[t] <= line:
                    depth.append(c[t] / peak - 1)
                    exit_px = c[t] * (1 - slip)
                    cash, sh, held, exit_t = sh * exit_px, 0.0, False, t
            else:
                raise ValueError(mode)
        vals.append(cash + sh * c[t])
        hold.append(held)
        if held:
            peak = max(peak, c[t])
    if not held:
        cyc.append((exit_t, None, exit_px, None))
    sim.last_depth, sim.last_next = depth, nxt
    return np.array(vals), np.array(hold), cyc


def mdd(v):
    return float((v / np.maximum.accumulate(v) - 1).min())


def kpaths(ru, rfd, extra, rp=None):
    out = []
    for k in K_GRID:
        if k <= 1:
            r = k * ru
        elif rp is not None:
            r = (k - 1) * rp + (2 - k) * ru
        else:
            r = np.maximum(k * ru - (k - 1) * rfd - (FEE + extra) * (k - 1) / (L - 1) / 252, -0.99)
        cur = np.concatenate([[1.0], np.cumprod(1 + r)])
        out.append((float(cur[-1] - 1), mdd(cur)))
    return np.array(out)


def eff(ret, dd, fr):
    ok = fr[:, 1] >= dd - 1e-12
    j = int(np.argmax(np.where(ok, fr[:, 0], -np.inf)))
    return float(ret - fr[j, 0]), float(K_GRID[j])


# ============================================================================ 평가
_FR = {}


def frontier_of(key, ru, rfd, extra, rp=None):
    if key not in _FR:
        _FR[key] = kpaths(ru, rfd, extra, rp)
    return _FR[key]


def eval_cfg(U, X, W, mode="intra", slip=0.0, var="cal", wins=None, full=True):
    """반환 rows(DataFrame: 기초×창), cyc(DataFrame: 기초×창×판 1회)."""
    wins = wins or WIN
    rows, cycs = [], []
    for u, D in U.items():
        ub, P = D["und"], D["syn"][var]
        days = ub["days"]
        ex = D["extra"] if var == "cal" else 0.0
        o, h, l, c = P["o"].tolist(), P["h"].tolist(), P["l"].tolist(), P["c"].tolist()
        for rn, (d0, d1) in RUNS.items():
            a, z = idx(days, d0), idx(days, d1, "right") - 1
            if a < MIN_HIST or z <= a:
                continue
            v, hd, cy = sim(o, h, l, c, a, z, X, W, mode, slip)
            for w, (wr, w0, w1) in wins.items():
                if wr != rn:
                    continue
                i0, i1 = max(idx(days, w0), a), min(idx(days, w1, "right") - 1, z)
                if i1 <= i0:
                    continue
                vv = v[i0 - a:i1 - a + 2] / v[i0 - a]
                hv = P["c"][i0 - 1:i1 + 1] / P["c"][i0 - 1]
                ru = ub["c"][i0:i1 + 1] / ub["c"][i0 - 1:i1] - 1
                fr = frontier_of((var, u, w), ru, D["rf"][i0:i1 + 1] / 252, ex)
                r, dd = float(vv[-1] - 1), mdd(vv)
                hr, hdd = float(hv[-1] - 1), mdd(hv)
                e, ke = eff(r, dd, fr)
                he, _ = eff(hr, hdd, fr)
                uv = ub["c"][i0 - 1:i1 + 1] / ub["c"][i0 - 1]
                rows.append({"under": u, "win": w, "ret": r, "mdd": dd, "exc": e, "k_eq": ke, "hold_ret": hr,
                             "hold_mdd": hdd, "hold_exc": he, "u_ret": float(uv[-1] - 1), "u_mdd": mdd(uv),
                             "kbar": float(L * hd[i0 - a:i1 - a + 1].mean())})
                for (et, rt, ep, rp_) in cy:
                    if not (w0 <= days[et] <= w1):
                        continue
                    closed = rt is not None and days[rt] <= w1
                    g = ep / rp_ - 1 if closed else ep / P["c"][i1] - 1
                    cycs.append({"under": u, "win": w, "gain": g, "exit": str(days[et]),
                                 "re": str(days[rt]) if rt is not None else None, "out": (rt - et) if rt is not None else None})
        if full:
            a, z = max(idx(days, FULL[0]), MIN_HIST), idx(days, FULL[1], "right") - 1
            v, hd, cy = sim(o, h, l, c, a, z, X, W, mode, slip)
            hv = P["c"][a - 1:z + 1] / P["c"][a - 1]
            uv = ub["c"][a - 1:z + 1] / ub["c"][a - 1]
            yrs = (z - a + 1) / 252
            rows.append({"under": u, "win": "전체", "start": str(days[a]), "ret": float(v[-1] - 1), "mdd": mdd(v),
                         "cagr": float(max(v[-1], 1e-9) ** (1 / yrs) - 1), "hold_ret": float(hv[-1] - 1), "hold_mdd": mdd(hv),
                         "hold_cagr": float(max(hv[-1], 1e-9) ** (1 / yrs) - 1), "u_mdd": mdd(uv),
                         "u_cagr": float(uv[-1] ** (1 / yrs) - 1), "kbar": float(L * hd.mean()), "n_cyc": len(cy)})
    return pd.DataFrame(rows), pd.DataFrame(cycs)


def cyc_stats(x):
    x = np.asarray(x, float)
    if not len(x):
        return {"n": 0}
    w, lo = x[x > 0], x[x <= 0]
    return {"n": int(len(x)), "승률": float(len(w) / len(x)), "평균이익": float(w.mean()) if len(w) else np.nan,
            "평균손실": float(lo.mean()) if len(lo) else np.nan,
            "손익비": float(w.mean() / -lo.mean()) if len(w) and len(lo) and lo.mean() < 0 else np.nan,
            "기대값": float(x.mean()), "하위5%": float(np.quantile(x, 0.05))}


def summarize(rows, cyc):
    s = {}
    yr = rows[rows.win.isin(YEAR_WINS)]
    s["상품해"] = int(len(yr))
    s["전멸"] = int((yr.mdd <= DEATH).sum())
    s["전멸비율"] = float((yr.mdd <= DEATH).mean())
    s["보유전멸"] = int((yr.hold_mdd <= DEATH).sum())
    s["1배전멸"] = int((yr.u_mdd <= DEATH).sum())
    s["전멸목록"] = [f"{r.under} {r.win} {r.mdd:+.1%}" for r in yr[yr.mdd <= DEATH].itertuples()]
    for w in ["2026앞", "2026뒤", "2026", "2025", "2024", "2023", "2022", "2021", "2020", "2019", "2018", "2017", "2016",
              "2015", "2018Q4", "2020급락"]:
        g = rows[rows.win == w]
        if not len(g):
            continue
        c = cyc[cyc.win == w]["gain"] if len(cyc) else []
        s[w] = {"n": int(len(g)), "수익_중앙": float(g.ret.median()), "낙폭_중앙": float(g.mdd.median()),
                "수익_평균": float(g.ret.mean()), "보유수익_중앙": float(g.hold_ret.median()),
                "보유낙폭_중앙": float(g.hold_mdd.median()), "1배수익_중앙": float(g.u_ret.median()),
                "1배낙폭_중앙": float(g.u_mdd.median()),
                "효율_평균": float(g.exc.mean()), "효율_중앙": float(g.exc.median()),
                "보유효율_평균": float(g.hold_exc.mean()), "보유효율_중앙": float(g.hold_exc.median()),
                "보유대비_평균": float((g.exc - g.hold_exc).mean()),
                "보유보다나은": int((g.exc > g.hold_exc + 1e-12).sum()),
                "전멸": int((g.mdd <= DEATH).sum()), "보유전멸": int((g.hold_mdd <= DEATH).sum()),
                "평균배수": float(g.kbar.mean()), "효율선배수_중앙": float(g.k_eq.median()),
                "효율_평균_SNDK제외": float(g[g.under != "SNDK"].exc.mean()), "주기": cyc_stats(c)}
    f = rows[rows.win == "전체"]
    f14 = f[f.start <= "2015-01-02"]
    s["전체"] = {"n14": int(len(f14)), "연율_중앙": float(f14.cagr.median()), "낙폭_중앙": float(f14.mdd.median()),
                "보유연율_중앙": float(f14.hold_cagr.median()), "보유낙폭_중앙": float(f14.hold_mdd.median()),
                "1배연율_중앙": float(f14.u_cagr.median()), "1배낙폭_중앙": float(f14.u_mdd.median()),
                "1배이상_종목": int((f14.cagr >= f14.u_cagr).sum()),
                "장기전멸23": int((f.mdd <= DEATH).sum()), "보유장기전멸23": int((f.hold_mdd <= DEATH).sum()),
                "1배장기전멸23": int((f.u_mdd <= DEATH).sum())}
    return s


# ============================================================================ 실제 상품(2026)
def actual(U, d, meta, X, W, mode="intra", slip=0.0):
    rows, cycs = [], []
    for u, D in U.items():
        cands = sorted((d[p]["days"][0], p) for p, i in meta.items()
                       if i["under"] == u and i["L"] == L and p in d and d[p]["days"][0] < "2025-12-31")
        if not cands:
            continue
        pn = cands[0][1]
        p = d[pn]
        ub = D["und"]
        ui = np.searchsorted(ub["days"], p["days"])
        ok = ui < len(ub["days"])
        ok[ok] = ub["days"][ui[ok]] == p["days"][ok]
        keep = np.flatnonzero(ok)
        pk = {k: p[k][keep] for k in ("days", "o", "h", "l", "c")}
        uc = ub["c"][ui[keep]]
        days = pk["days"]
        a, z = idx(days, RUNS["2026"][0]), idx(days, RUNS["2026"][1], "right") - 1
        if a < 1 or z <= a:
            continue
        v, hd, cy = sim(pk["o"].tolist(), pk["h"].tolist(), pk["l"].tolist(), pk["c"].tolist(), a, z, X, W, mode, slip)
        for w in ("2026앞", "2026뒤", "2026"):
            _, w0, w1 = WIN[w]
            i0, i1 = max(idx(days, w0), a), min(idx(days, w1, "right") - 1, z)
            ru = uc[i0:i1 + 1] / uc[i0 - 1:i1] - 1
            rp = pk["c"][i0:i1 + 1] / pk["c"][i0 - 1:i1] - 1
            fr = frontier_of(("act", u, w), ru, None, 0.0, rp)
            vv = v[i0 - a:i1 - a + 2] / v[i0 - a]
            hv = pk["c"][i0 - 1:i1 + 1] / pk["c"][i0 - 1]
            e, _ = eff(float(vv[-1] - 1), mdd(vv), fr)
            he, _ = eff(float(hv[-1] - 1), mdd(hv), fr)
            rows.append({"product": pn, "under": u, "win": w, "ret": float(vv[-1] - 1), "mdd": mdd(vv),
                         "hold_ret": float(hv[-1] - 1), "hold_mdd": mdd(hv), "exc": e, "hold_exc": he})
            for (et, rt, ep, rp_) in cy:
                if not (w0 <= days[et] <= w1):
                    continue
                closed = rt is not None and days[rt] <= w1
                cycs.append({"under": u, "win": w, "gain": ep / rp_ - 1 if closed else ep / pk["c"][i1] - 1})
    return pd.DataFrame(rows), pd.DataFrame(cycs)


def act_summary(ar, ac):
    out = {}
    for w in ("2026앞", "2026뒤", "2026"):
        g = ar[ar.win == w]
        out[w] = {"n": int(len(g)), "효율_평균": float(g.exc.mean()), "효율_중앙": float(g.exc.median()),
                  "보유효율_평균": float(g.hold_exc.mean()), "보유효율_중앙": float(g.hold_exc.median()),
                  "보유보다나은": int((g.exc > g.hold_exc + 1e-12).sum()),
                  "수익보다나은": int((g.ret > g.hold_ret + 1e-12).sum()),
                  "수익_중앙": float(g.ret.median()), "보유수익_중앙": float(g.hold_ret.median()),
                  "주기": cyc_stats(ac[ac.win == w]["gain"] if len(ac) else [])}
    out["상품"] = sorted(ar["product"].unique().tolist())
    return out


def depth_stats(U, X, W, mode):
    """2015~2026 각 해 실행(보정 합성)에서 1회 매도가의 '산 뒤 최고 종가 대비'와 장중 매도 다음 날 시가."""
    dep, nx = [], []
    for u, D in U.items():
        P, days = D["syn"]["cal"], D["und"]["days"]
        o, h, l, c = P["o"].tolist(), P["h"].tolist(), P["l"].tolist(), P["c"].tolist()
        for rn, (d0, d1) in RUNS.items():
            a, z = idx(days, d0), idx(days, d1, "right") - 1
            if a < MIN_HIST or z <= a:
                continue
            sim(o, h, l, c, a, z, X, W, mode)
            dep += sim.last_depth
            nx += sim.last_next
    dep, nx = np.array(dep), np.array(nx)
    out = {"n": int(len(dep)), "평균": float(dep.mean()), "중앙": float(np.median(dep)), "하위5%": float(np.quantile(dep, 0.05)),
           "−20%보다깊음": float((dep < -0.20).mean()), "−25%보다깊음": float((dep < -0.25).mean())}
    if len(nx):
        out.update({"다음시가_평균": float(nx.mean()), "다음시가_중앙": float(np.median(nx)), "다음시가_높음비율": float((nx > 0).mean())})
    return out


# ============================================================================ 점검
def checks(U, d, meta):
    out = {}
    u = "NVDA"
    D = U[u]
    P = D["syn"]["cal"]
    days = D["und"]["days"]
    a, z = idx(days, "2026-01-02"), idx(days, "2026-10-09", "right") - 1
    v, _, _ = sim(P["o"].tolist(), P["h"].tolist(), P["l"].tolist(), P["c"].tolist(), a, z, 9.0, 20)
    out["손절 없음 = 상품 보유 최대 차"] = float(np.abs(v - P["c"][a - 1:z + 1] / P["c"][a - 1]).max())
    ru = D["und"]["c"][a:z + 1] / D["und"]["c"][a - 1:z] - 1
    rfd = D["rf"][a:z + 1] / 252
    k2 = np.concatenate([[1.0], np.cumprod(1 + np.maximum(2 * ru - rfd - (FEE + D["extra"]) / 252, -0.99))])
    out["k=2 공식 = 보정 합성 최대 차"] = float(np.abs(k2 - v).max())
    # 미래 교란: T 뒤 기초 가격을 흔들고 합성을 다시 만들어도 T까지 가치가 같아야 한다
    rng = np.random.default_rng(7)
    worst = 0.0
    for T_off in (40, 90, 150):
        T = a + T_off
        ub = {k: (vv.copy() if isinstance(vv, np.ndarray) else vv) for k, vv in D["und"].items()}
        for k in ("o", "h", "l", "c"):
            ub[k][T + 1:] *= 1 + rng.normal(0, 0.05, len(ub[k]) - T - 1)
        ub["h"][T + 1:] = np.maximum.reduce([ub["h"][T + 1:], ub["o"][T + 1:], ub["c"][T + 1:]])
        ub["l"][T + 1:] = np.minimum.reduce([ub["l"][T + 1:], ub["o"][T + 1:], ub["c"][T + 1:]])
        P2 = synth(ub, D["rf"], FEE + D["extra"])
        P1 = synth(D["und"], D["rf"], FEE + D["extra"])
        for mode in ("intra", "delay", "close_next"):
            for X, W in ((0.15, 20), (0.10, 5), (0.25, 40)):
                w1, _, _ = sim(P1["o"].tolist(), P1["h"].tolist(), P1["l"].tolist(), P1["c"].tolist(), a, z, X, W, mode)
                w2, _, _ = sim(P2["o"].tolist(), P2["h"].tolist(), P2["l"].tolist(), P2["c"].tolist(), a, z, X, W, mode)
                worst = max(worst, float(np.abs(w1[:T - a + 2] - w2[:T - a + 2]).max()))
    out["미래 교란 뒤 t까지 가치 최대 차(27경우)"] = worst
    # 수동 예: 첫 손절·재매수 날짜와 가격
    v, hd, cy = sim(P["o"].tolist(), P["h"].tolist(), P["l"].tolist(), P["c"].tolist(), a, z, 0.15, 20)
    out["NVDA 2026 주기(판 날, 되산 날, 판 가격/되산 가격−1)"] = [
        (str(days[e]), str(days[r]) if r is not None else None, round(ep / rp - 1, 4) if rp else None) for e, r, ep, rp in cy]
    return out


def compare_branch(rows, ar):
    """갈래 JSON(results/p4_LEVSINGLE.json)의 같은 규칙 행과 기초×창별 최대 차."""
    p = RESULTS / "p4_LEVSINGLE.json"
    if not p.exists():
        return {"갈래 JSON": "없음"}
    J = json.loads(p.read_text(encoding="utf-8"))
    rule = "S 추적15% → 20일 뒤"
    br = pd.DataFrame([r for r in J["rows"] if r["var"] == "cal" and r["rule"] == rule])
    m = rows.merge(br, on=["under", "win"], suffixes=("", "_b"))
    out = {"맞춘 행": int(len(m)), "내 행": int(len(rows)), "갈래 행": int(len(br))}
    for k in ("ret", "mdd", "exc", "kbar"):
        if k + "_b" in m:
            dd = (m[k] - m[k + "_b"]).abs()
            out[f"{k} 최대 차"] = float(dd.max())
    if "cagr" in m and "cagr_b" in m:
        out["cagr 최대 차"] = float((m.cagr - m.cagr_b).abs().max())
    ba = pd.DataFrame([r for r in J.get("actual", []) if r["rule"] == rule])
    if len(ba):
        ma = ar.merge(ba, on=["under", "win"], suffixes=("", "_b"))
        out["실제 상품 맞춘 행"] = int(len(ma))
        out["실제 exc 최대 차"] = float((ma.exc - ma.exc_b).abs().max())
        out["실제 상품 이름 일치"] = bool((ma["product"] == ma["product_b"]).all())
    hold = pd.DataFrame([r for r in J["rows"] if r["var"] == "cal" and r["rule"] == "보유(2배)"])
    mh = rows.merge(hold, on=["under", "win"], suffixes=("", "_b"))
    out["보유 exc 최대 차"] = float((mh.hold_exc - mh.exc_b).abs().max())
    out["보유 ret 최대 차"] = float((mh.hold_ret - mh.ret_b).abs().max())
    return out


# ============================================================================ 실행
def p(x, d=1):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x * 100:+.{d}f}%"


def main():
    t0 = time.time()
    U, excluded, d, meta = prepare()
    print(f"기초 {len(U)}개, 제외 {excluded}")
    for u, D in U.items():
        print(f"  {u:5s} 시작 {D['und']['days'][0]} 추가비용 {D['extra']:.4f} "
              f"{[(r['product'], None if r['drag'] is None else round(r['drag'], 4)) for r in D['track']]}")
    out = {"config": {"BASE": BASE, "NB_X": NB_X, "NB_W": NB_W, "K": len(K_GRID), "SLIP": SLIP, "DEATH": DEATH,
                      "INSURE_MAX": INSURE_MAX, "TIMING": TIMING, "excluded": excluded},
           "extra": {u: {"extra": D["extra"], "track": D["track"]} for u, D in U.items()}}
    out["checks"] = checks(U, d, meta)
    print("점검", json.dumps(out["checks"], ensure_ascii=False, default=str))

    # 1) 재현(기본 규칙, 보정·기본 합성 + 실제 상품)
    rows, cyc = eval_cfg(U, *BASE)
    ar, ac = actual(U, d, meta, *BASE)
    out["base"] = summarize(rows, cyc)
    out["base_actual"] = act_summary(ar, ac)
    rb, cb = eval_cfg(U, *BASE, var="basic")
    out["base_basic"] = summarize(rb, cb)
    out["compare_branch"] = compare_branch(rows, ar)
    out["base_rows"] = rows.to_dict("records")
    out["base_cycles"] = cyc.to_dict("records")
    out["base_actual_rows"] = ar.to_dict("records")
    print("갈래 대조", json.dumps(out["compare_branch"], ensure_ascii=False))
    print(f"재현 끝 {time.time() - t0:.0f}s")

    # 2) 이웃 42개(보정 합성 + 실제 상품 2026)
    nb = []
    for X in NB_X:
        for W in NB_W:
            r_, c_ = eval_cfg(U, X, W)
            s = summarize(r_, c_)
            a_r, a_c = actual(U, d, meta, X, W)
            sa = act_summary(a_r, a_c)
            nb.append({"X": X, "W": W, "s": s, "act": sa})
        print(f"  이웃 X={X} 끝 {time.time() - t0:.0f}s", flush=True)
    out["neighbors"] = nb

    # 3) 시점 변형(기본 규칙) + 이웃 42개에 하루 늦춤·종가 판단
    tm = {}
    for name, (mode, slip) in TIMING.items():
        r_, c_ = eval_cfg(U, *BASE, mode=mode, slip=slip)
        a_r, a_c = actual(U, d, meta, *BASE, mode=mode, slip=slip)
        tm[name] = {"s": summarize(r_, c_), "act": act_summary(a_r, a_c)}
    out["timing"] = tm
    out["depth"] = {m: depth_stats(U, *BASE, m) for m in ("intra", "delay", "close_next")}
    print("매도 깊이", json.dumps(out["depth"], ensure_ascii=False))
    nbt = []
    for mode in ("delay", "close_next"):
        for X in NB_X:
            for W in NB_W:
                r_, c_ = eval_cfg(U, X, W, mode=mode)
                s = summarize(r_, c_)
                nbt.append({"mode": mode, "X": X, "W": W, "s": s})
        print(f"  이웃 {mode} 끝 {time.time() - t0:.0f}s", flush=True)
    out["neighbors_timing"] = nbt
    (RESULTS / "p6_V_TRAIL15_A.json").write_text(json.dumps(out, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"저장 {RESULTS / 'p6_V_TRAIL15_A.json'} · {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
