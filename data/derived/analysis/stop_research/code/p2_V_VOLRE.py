"""반박 검증 VOLRE 대표 후보(2026-10-10): G1 X1·시간20.

후보 = 보유·무장 상태에서 t-1 종가 < t-1 SMA50 → t 시가 전량 매도, 판 날 e + 20거래일 시가 전량 재매수(조건 없음).
되산 순간 무장 해제, 보유 중 종가 ≥ SMA50이 나오면 재무장. 창 시작: 직전 종가에 전량 보유, 직전 종가 ≥ SMA50이면 무장.

stock_lab.py를 import하지 않는다. cache/p2_long.parquet만 읽어 새로 짠 엔진(배당 수정 = Adj Close/Close 비율을 OHLC에 곱함).
ETF 제외는 cache/p2_universe.json의 'ETF' 표시로 정한다.

시점: t일 시가 판단에는 t-1 종가까지 쓴다(신호는 종가 s에서 확정 → s+1+지연 시가 체결). 장중 스탑 없음.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"
RESULTS = HERE.parent / "results"

# ============================================================================ 사전 고정(뒤·2025를 본 뒤 바꾸지 않는다)
SMA_N, WAIT = 50, 20                                  # 후보
SMA_GRID = (30, 40, 50, 60, 70)                       # 이웃: SMA 길이 앞뒤 2개씩(간격 10)
WAIT_GRID = (10, 15, 20, 25, 30)                      # 이웃: 대기 거래일 앞뒤 2개씩(간격 5)
WAIT_FINE = (18, 19, 21, 22)                          # 보조 이웃: SMA50 고정, 하루 단위 앞뒤 2개씩
SEEDS = tuple(range(1001, 1021))                      # 위약 시드 20개
RE_JITTER = (10, 30)                                  # 재진입 위약: 같은 SMA50 매도, 대기 = 균등 정수 10~30(평균 20)
SLIP = 0.005                                          # 체결 미끄러짐 민감도
DELAY = 1                                             # 하루 늦춘 판
EGRID = np.arange(201) * 0.005                        # 효율선 노출 격자 0~1, 0.005 간격
RUNS = {"2026": ("2026-01-02", "2026-10-09"), "2025": ("2025-01-02", "2025-12-31"),
        "2022": ("2022-01-03", "2022-12-30"), "2018": ("2018-01-02", "2018-12-31")}
SLICES = {"2026": {"2026": RUNS["2026"], "2026앞": ("2026-01-02", "2026-05-29"), "2026뒤": ("2026-06-01", "2026-10-09")}}
EXCLUDE_2018_ALT = ("QUBT",)                          # 2018 보조 판: 장외 시절 가격 오류(0.015 등) 종목 빼고
CLAIM = {  # 갈래가 보고한 숫자(소수)
    "2026": {"포트_수익": 0.569, "포트_낙폭": -0.148, "포트BH_수익": 0.586, "포트BH_낙폭": -0.255},
    "2026앞": {"포트_수익": 0.545, "포트_낙폭": -0.083, "효율선초과": 0.132},
    "2026뒤": {"포트_수익": 0.016, "포트_낙폭": -0.148, "효율선초과": 0.016, "포트BH_수익": -0.041, "포트BH_낙폭": -0.255,
              "손절주기": 265, "승률": 0.585, "손익비": 1.69, "기대값": 0.052},
    "2025": {"포트_수익": 0.481, "포트_낙폭": -0.273, "포트BH_수익": 0.743, "포트BH_낙폭": -0.336, "효율선초과": -0.086,
             "승률": 0.454, "손익비": 1.03, "기대값": -0.010},
    "2022": {"포트_수익": -0.110, "포트_낙폭": -0.204, "포트BH_수익": -0.306, "포트BH_낙폭": -0.363},
}
CLAIM_SLIP = {"2026": {"포트_수익": 0.518}, "2026뒤": {"포트_수익": -0.003}, "2025": {"효율선초과": -0.170}}


# ============================================================================ 자료
def load():
    df = pd.read_parquet(CACHE / "p2_long.parquet").sort_values(["sym", "date"])
    uni = json.loads((CACHE / "p2_universe.json").read_text(encoding="utf-8"))
    etf = {s for s, v in uni.items() if "ETF" in v}
    out = {}
    for sym, g in df.groupby("sym"):
        if sym in etf:
            continue
        g = g.dropna(subset=["Open", "High", "Low", "Close"])
        g = g[(g.Close > 0) & (g.Open > 0) & (g.Low > 0)]
        f = (g["Adj Close"] / g["Close"]).fillna(1.0).to_numpy()
        c = g.Close.to_numpy() * f
        out[sym] = {"d": g.date.to_numpy().astype(str), "o": g.Open.to_numpy() * f, "c": c, "sma": {}}
    return out


def sma(s: dict, n: int) -> np.ndarray:
    if n not in s["sma"]:
        c = s["c"]
        cs = np.concatenate([[0.0], np.cumsum(c)])
        m = np.full(len(c), np.nan)
        m[n - 1:] = (cs[n:] - cs[:-n]) / n           # m[t] = c[t-n+1..t] 평균(t까지의 값)
        s["sma"][n] = m
    return s["sma"][n]


def span(s, d0, d1):
    a = int(np.searchsorted(s["d"], d0, "left"))
    z = int(np.searchsorted(s["d"], d1, "right")) - 1
    return a, z


def members(D, d0, exclude=()):
    return sorted(k for k, s in D.items() if k not in exclude and int(np.searchsorted(s["d"], d0, "left")) >= 252)


# ============================================================================ 엔진(종목 하나)
def sim_rule(s, a, z, n=SMA_N, wait=WAIT, sb=0.0, ss=0.0, delay=0, rng=None, jitter=None):
    """규칙 판. 반환 V(길이 z-a+2, 첫 값 1), held(종가 시점 보유 여부), 주기 [(판 인덱스, 손익, 열림, 나가있던 날, 되산 인덱스)]."""
    c, o, m = s["c"], s["o"], sma(s, n)
    cash, sh, held = 0.0, 1.0 / c[a - 1], True
    armed = bool(c[a - 1] >= m[a - 1])
    pending, ex_t, ex_px, w = None, None, None, None
    V, H, cyc = [1.0], [], []
    for t in range(a, z + 1):
        if held and pending is not None and t == pending:          # 시가 매도(신호는 종가 pending-1-delay에서 확정)
            ex_px = o[t] * (1 - ss)
            cash, sh, held, ex_t, pending = sh * ex_px, 0.0, False, t, None
            w = wait if jitter is None else int(rng.integers(jitter[0], jitter[1] + 1))
        elif (not held) and t - ex_t >= w:                           # 판 날 + w 거래일 시가 재매수
            px = o[t] * (1 + sb)
            sh, cash, held, armed = cash / px, 0.0, True, False
            cyc.append((ex_t, ex_px / px - 1, False, t - ex_t, t))
        V.append(cash + sh * c[t])
        H.append(held)
        if held:                                                     # 종가 뒤: 재무장 → 하향 신호
            if c[t] >= m[t]:
                armed = True
            if armed and c[t] < m[t] and pending is None:
                pending = t + 1 + delay
    if not held:
        cyc.append((ex_t, ex_px / c[z] - 1, True, z - ex_t, None))
    return np.array(V), np.array(H), cyc


def sim_blocks(s, a, z, blocks, sb=0.0, ss=0.0):
    """위약 판: 미리 정한 (판 인덱스, 길이) 구간만 현금. 판 날 시가 매도, 판 날+길이 시가 재매수(창 끝을 넘으면 못 되삼)."""
    c, o = s["c"], s["o"]
    start = {e: L for e, L in blocks}
    cash, sh, held, ex_t, ex_px, back = 0.0, 1.0 / c[a - 1], True, None, None, None
    V, H, cyc = [1.0], [], []
    for t in range(a, z + 1):
        if held and t in start:
            ex_px = o[t] * (1 - ss)
            cash, sh, held, ex_t, back = sh * ex_px, 0.0, False, t, t + start[t]
        elif (not held) and t == back:
            px = o[t] * (1 + sb)
            sh, cash, held = cash / px, 0.0, True
            cyc.append((ex_t, ex_px / px - 1, False, t - ex_t, t))
        V.append(cash + sh * c[t])
        H.append(held)
    if not held:
        cyc.append((ex_t, ex_px / c[z] - 1, True, z - ex_t, None))
    return np.array(V), np.array(H), cyc


# ============================================================================ 묶기·지표
def mdd(v):
    return float((v / np.maximum.accumulate(v) - 1).min())


def frontier_excess(pr, bh_daily):
    tgt = mdd(pr)
    best = -1.0
    for e in EGRID:
        cur = np.concatenate([[1.0], np.cumprod(1 + e * bh_daily)])
        if mdd(cur) >= tgt - 1e-12:
            best = max(best, cur[-1] - 1)
    return float(pr[-1] - 1 - best)


def run(D, syms, d0, d1, fn):
    """fn(sym, s, a, z) → (V, H, cyc). 반환 일별 수익 R·B, 보유 E, 주기 목록(날짜 포함)."""
    R, B, E, cyc, meta = {}, {}, {}, [], {}
    for k in syms:
        s = D[k]
        a, z = span(s, d0, d1)
        if a < 252 or z <= a:
            continue
        V, H, cy = fn(k, s, a, z)
        dd = s["d"][a:z + 1]
        R[k] = pd.Series(V[1:] / V[:-1] - 1, index=dd)
        B[k] = pd.Series(s["c"][a:z + 1] / s["c"][a - 1:z] - 1, index=dd)
        E[k] = pd.Series(H.astype(float), index=dd)
        meta[k] = (a, z, cy)
        for e_t, g, op, dout, r_t in cy:
            cyc.append({"sym": k, "exit_day": s["d"][e_t], "gain": g, "open": op, "days_out": dout,
                        "re_day": None if r_t is None else s["d"][r_t]})
    return {"R": pd.DataFrame(R).sort_index(), "B": pd.DataFrame(B).sort_index(), "E": pd.DataFrame(E).sort_index(),
            "cyc": cyc, "meta": meta}


def stats(res, d0, d1):
    R = res["R"].loc[(res["R"].index >= d0) & (res["R"].index <= d1)]
    B = res["B"].loc[(res["B"].index >= d0) & (res["B"].index <= d1)]
    E = res["E"].loc[(res["E"].index >= d0) & (res["E"].index <= d1)]
    rm, bm = R.mean(axis=1).fillna(0).to_numpy(), B.mean(axis=1).fillna(0).to_numpy()
    pr = np.concatenate([[1.0], np.cumprod(1 + rm)])
    pb = np.concatenate([[1.0], np.cumprod(1 + bm)])
    sr, sbh = (1 + R.fillna(0)).prod() - 1, (1 + B.fillna(0)).prod() - 1
    g = np.array([cy["gain"] for cy in res["cyc"] if d0 <= cy["exit_day"] <= d1])
    win, loss = g[g > 0], g[g <= 0]
    return {"종목수": int(R.shape[1]), "포트_수익": float(pr[-1] - 1), "포트_낙폭": mdd(pr),
            "포트BH_수익": float(pb[-1] - 1), "포트BH_낙폭": mdd(pb), "효율선초과": frontier_excess(pr, bm),
            "평균노출": float(E.mean().mean()), "수익_나은종목비율": float((sr > sbh + 1e-12).mean()),
            "최악5%_종목수익": float(sr.quantile(0.05)), "최악5%_BH": float(sbh.quantile(0.05)),
            "손절주기": int(len(g)), "승률": float(len(win) / len(g)) if len(g) else float("nan"),
            "평균이익": float(win.mean()) if len(win) else float("nan"),
            "평균손실": float(loss.mean()) if len(loss) else float("nan"),
            "손익비": float(win.mean() / -loss.mean()) if len(win) and len(loss) and loss.mean() < 0 else float("nan"),
            "기대값": float(g.mean()) if len(g) else float("nan")}


def all_windows(D, fn, runs=RUNS, exclude=()):
    out, raw = {}, {}
    for rn, (d0, d1) in runs.items():
        res = run(D, members(D, d0, exclude), d0, d1, fn)
        raw[rn] = res
        for w, (w0, w1) in SLICES.get(rn, {rn: (d0, d1)}).items():
            out[w] = stats(res, w0, w1)
    return out, raw


def rule_fn(**kw):
    return lambda k, s, a, z: sim_rule(s, a, z, **kw)


# ============================================================================ 위약
def place_blocks(rng, blocks, a, z, strata):
    """blocks = [(길이, 층 (lo, hi), 열림)]. 겹치지 않게(되산 날 포함 구간 [e, e+L]) 무작위 배치. 실패 시 None."""
    occ = np.zeros(z + 3, bool)
    out = []
    for L, (lo, hi), op in sorted(blocks, key=lambda x: -x[0]):
        last = z + 1 - L if op else z - L                         # 닫힌 구간은 창 안에서 되산다
        cand = [e for e in range(lo, min(hi, last) + 1) if not occ[e:min(e + L, z) + 1].any()]
        if not cand:
            return None
        e = int(rng.choice(cand))
        occ[e:min(e + L, z) + 1] = True
        out.append((e, L))
    return out


def placebo_exit_fn(raw_cand, seed, sb=0.0, ss=0.0):
    """같은 종목·같은 손절 횟수·같은 나가있던 날 수(20일, 끝까지 못 되산 것은 남은 날 수)를 무작위 날짜에.
    2026은 판 날이 앞/뒤 어느 쪽인지 층을 지켜 노출을 구간별로도 맞춘다."""
    rng = np.random.default_rng(seed)

    def fn(k, s, a, z):
        _, _, cy = raw_cand["meta"][k]
        bounds = [(a, z)]
        if s["d"][a] <= "2026-01-02" and s["d"][z] >= "2026-10-09":
            mid = int(np.searchsorted(s["d"], "2026-06-01", "left"))
            bounds = [(a, mid - 1), (mid, z)]
        blocks = []
        for e_t, g, op, dout, r_t in cy:
            L = (r_t - e_t) if not op else (z + 1 - e_t)
            st = next(b for b in bounds if b[0] <= e_t <= b[1])
            blocks.append((L, st, op))
        pl = None
        for _ in range(200):
            pl = place_blocks(rng, blocks, a, z, bounds)
            if pl is not None:
                break
        if pl is None:                                              # 층을 풀고 다시
            blocks2 = [(L, (a, z), op) for L, _, op in blocks]
            for _ in range(200):
                pl = place_blocks(rng, blocks2, a, z, [(a, z)])
                if pl is not None:
                    break
        fn.fail += pl is None
        return sim_blocks(s, a, z, pl or [], sb, ss)
    fn.fail = 0
    return fn


# ============================================================================ 시점 자체 점검
def lookahead_check(D, syms):
    """T 이후 가격을 교란해도 T까지 가치가 같아야 한다(지연 0·1 모두)."""
    worst = 0.0
    rng = np.random.default_rng(0)
    for k in syms[:30]:
        s = D[k]
        a, z = span(s, *RUNS["2026"])
        T = a + 80
        for delay in (0, 1):
            V1, _, _ = sim_rule(s, a, z, delay=delay)
            s2 = {"d": s["d"], "o": s["o"].copy(), "c": s["c"].copy(), "sma": {}}
            s2["o"][T + 1:] *= 1 + rng.normal(0, 0.05, len(s2["o"]) - T - 1)
            s2["c"][T + 1:] *= 1 + rng.normal(0, 0.05, len(s2["c"]) - T - 1)
            V2, _, _ = sim_rule(s2, a, z, delay=delay)
            worst = max(worst, float(np.max(np.abs(V1[:T - a + 2] - V2[:T - a + 2]))))
    return worst


# ============================================================================ 실행
WKEYS = ["2026", "2026앞", "2026뒤", "2025", "2022", "2018"]


def main():
    t0 = time.time()
    D = load()
    out = {"생성": "2026-10-10", "스크립트": "code/p2_V_VOLRE.py", "후보": "X1·시간20 (SMA50 하향 돌파 → 20거래일째 시가 재매수)"}
    out["종목수_2026"] = len(members(D, RUNS["2026"][0]))
    out["미래교란_최대차"] = lookahead_check(D, members(D, RUNS["2026"][0]))

    # 1) 독립 재계산
    base, raw = all_windows(D, rule_fn())
    out["재현"] = base
    diffs = []
    for w, kv in CLAIM.items():
        for key, v in kv.items():
            mine = base[w][key]
            tol = 0.005 if key not in ("손절주기", "손익비") else (0 if key == "손절주기" else 0.01)
            diffs.append({"창": w, "항목": key, "주장": v, "재계산": mine, "차": mine - v, "허용초과": abs(mine - v) > tol + 1e-9})
    out["재현_비교"] = diffs
    print(f"재현 {time.time() - t0:.0f}s", flush=True)

    # 2) 시점·체결
    var = {"지연1일": rule_fn(delay=DELAY), "매도만_미끄러짐0.5%": rule_fn(ss=SLIP),
           "양쪽_미끄러짐0.5%": rule_fn(sb=SLIP, ss=SLIP), "지연1일+양쪽0.5%": rule_fn(delay=DELAY, sb=SLIP, ss=SLIP)}
    out["체결변형"] = {k: all_windows(D, f)[0] for k, f in var.items()}
    for w, kv in CLAIM_SLIP.items():
        for key, v in kv.items():
            mine = out["체결변형"]["양쪽_미끄러짐0.5%"][w][key]
            diffs.append({"창": w + "(미끄러짐)", "항목": key, "주장": v, "재계산": mine, "차": mine - v,
                          "허용초과": abs(mine - v) > 0.005})
    print(f"체결 {time.time() - t0:.0f}s", flush=True)

    # 3) 이웃
    grid = {}
    for n in SMA_GRID:
        for wt in WAIT_GRID:
            grid[f"SMA{n}·시간{wt}"] = all_windows(D, rule_fn(n=n, wait=wt))[0]
    fine = {f"SMA50·시간{wt}": all_windows(D, rule_fn(wait=wt))[0] for wt in WAIT_FINE}
    out["이웃"], out["이웃_하루"] = grid, fine

    def ok3(ws):
        return ws["2026앞"]["효율선초과"] > 0 and ws["2026뒤"]["효율선초과"] > 0 and ws["2025"]["효율선초과"] > 0
    nb = {k: v for k, v in grid.items() if k != "SMA50·시간20"}
    out["이웃_요약"] = {
        "격자_개수": len(grid), "후보_제외_이웃": len(nb), "하루_이웃": len(fine),
        "모두>0_비율_이웃24": float(np.mean([ok3(v) for v in nb.values()])),
        "모두>0_비율_하루4": float(np.mean([ok3(v) for v in fine.values()])),
        "뒤>0_비율_이웃24": float(np.mean([v["2026뒤"]["효율선초과"] > 0 for v in nb.values()])),
        "2025>0_비율_이웃24": float(np.mean([v["2025"]["효율선초과"] > 0 for v in nb.values()])),
        "앞>0_비율_이웃24": float(np.mean([v["2026앞"]["효율선초과"] > 0 for v in nb.values()])),
        "후보_모두>0": bool(ok3(base)),
    }
    print(f"이웃 {time.time() - t0:.0f}s", flush=True)

    # 4) 위약
    pe, pr_ = [], []
    fails = 0
    for sd in SEEDS:
        ws = {}
        for rn, (d0, d1) in RUNS.items():
            fn = placebo_exit_fn(raw[rn], sd)
            res = run(D, members(D, d0), d0, d1, fn)
            fails += fn.fail
            for w, (w0, w1) in SLICES.get(rn, {rn: (d0, d1)}).items():
                ws[w] = stats(res, w0, w1)
        pe.append(ws)
        pr_.append(all_windows(D, rule_fn(rng=np.random.default_rng(sd), jitter=RE_JITTER))[0])
    out["위약_배치실패"] = fails

    def rank_tab(pls):
        tab = {}
        for w in WKEYS:
            tab[w] = {}
            for key in ["효율선초과", "포트_수익", "포트_낙폭", "기대값", "승률", "손익비", "평균노출", "손절주기"]:
                vals = np.array([p[w][key] for p in pls], float)
                cv = base[w][key]
                tab[w][key] = {"후보": cv, "위약_중앙": float(np.nanmedian(vals)), "위약_최소": float(np.nanmin(vals)),
                               "위약_최대": float(np.nanmax(vals)), "순위": int(1 + np.sum(vals > cv)), "n": len(vals)}
        return tab
    out["위약_무작위매도"] = rank_tab(pe)
    out["위약_재진입흔듦"] = rank_tab(pr_)
    print(f"위약 {time.time() - t0:.0f}s", flush=True)

    # 5) 2018 보조(가격 오류 종목 제외)
    out["2018_QUBT제외"] = all_windows(D, rule_fn(), runs={"2018": RUNS["2018"]}, exclude=EXCLUDE_2018_ALT)[0]["2018"]
    q = D["QUBT"]
    a, z = span(q, *RUNS["2018"])
    out["QUBT_2018_BH"] = float(q["c"][z] / q["c"][a - 1] - 1)
    out["QUBT_2018_최대일수익"] = float(np.max(q["c"][a:z + 1] / q["c"][a - 1:z] - 1))

    extra(D, out, raw, grid)
    (RESULTS / "p2_V_VOLRE.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"끝 {time.time() - t0:.0f}s", flush=True)


# ============================================================================ 보조(첫 실행 결과를 본 뒤 추가 — 격자·후보·위약 정의는 그대로)
PHASE_STARTS = ("2025-10-01", "2025-11-03", "2025-12-01", "2026-01-02")   # 실행 시작일 흔들기(주기 위상)
CTX_YEARS = ("2019", "2020", "2021", "2023", "2024")


def extra(D, out, raw, grid):
    # (a) 이웃 격자 안 '앞 > 0이면 뒤·2025도 > 0인가'
    nb = {k: v for k, v in grid.items() if k != "SMA50·시간20"}
    pas = [v for v in nb.values() if v["2026앞"]["효율선초과"] > 0]
    out["이웃_조건부"] = {"앞>0_개수": len(pas),
                       "앞>0중_뒤>0": float(np.mean([v["2026뒤"]["효율선초과"] > 0 for v in pas])) if pas else float("nan"),
                       "앞>0중_2025>0": float(np.mean([v["2025"]["효율선초과"] > 0 for v in pas])) if pas else float("nan")}
    # (b) 실행 시작일 흔들기: 같은 규칙을 더 일찍 시작해 2026 앞·뒤를 잘라 잰다
    ph = {}
    for st in PHASE_STARTS:
        res = run(D, members(D, st), st, RUNS["2026"][1], rule_fn())
        ph[st] = {w: stats(res, *SLICES["2026"][w]) for w in ("2026앞", "2026뒤")}
        ph[st]["종목수"] = len(res["meta"])
    out["시작일_흔들기"] = ph
    # (c) 2018 가격 오류 종목(QUBT) 빼고 무작위 매도 위약
    d0, d1 = RUNS["2018"]
    syms = members(D, d0, EXCLUDE_2018_ALT)
    base_res = run(D, syms, d0, d1, rule_fn())
    cand = stats(base_res, d0, d1)
    pls = []
    for sd in SEEDS:
        pls.append(stats(run(D, syms, d0, d1, placebo_exit_fn(base_res, sd)), d0, d1))
    out["2018_QUBT제외_위약"] = {k: {"후보": cand[k], "위약_중앙": float(np.median([p[k] for p in pls])),
                                  "순위": int(1 + sum(p[k] > cand[k] for p in pls))}
                              for k in ("효율선초과", "포트_수익", "기대값", "승률", "손익비")}
    # (d) 달력 묶음: 2026 앞 매도 날짜가 몰린 날과 대기 15·20·25일 재매수 날
    cal = {}
    for wt in (15, 20, 25):
        res = run(D, members(D, RUNS["2026"][0]), *RUNS["2026"], rule_fn(wait=wt))
        cy = [c for c in res["cyc"] if c["exit_day"] <= "2026-05-29"]
        ex = pd.Series([c["exit_day"] for c in cy]).value_counts().head(4)
        rb = pd.Series([c["re_day"] for c in cy if c["re_day"]]).value_counts().head(4)
        cal[f"시간{wt}"] = {"매도_상위": {str(k): int(v) for k, v in ex.items()},
                          "재매수_상위": {str(k): int(v) for k, v in rb.items()},
                          "앞_효율선초과": stats(res, *SLICES["2026"]["2026앞"])["효율선초과"]}
    Bm = raw["2026"]["B"].mean(axis=1)
    cal["동일가중BH_일수익"] = {d: float(Bm.get(d, np.nan)) for d in ("2026-03-30", "2026-03-31", "2026-04-07", "2026-04-08",
                                                              "2026-04-14")}
    out["달력_묶음"] = cal
    # (e) 맥락 해의 매수·보유에 QUBT가 미치는 영향(갈래 맥락 표 점검)
    q = {}
    for y in ("2018",) + CTX_YEARS:
        d0, d1 = {"2019": ("2019-01-02", "2019-12-31"), "2020": ("2020-01-02", "2020-12-31"),
                  "2021": ("2021-01-04", "2021-12-31"), "2023": ("2023-01-03", "2023-12-29"),
                  "2024": ("2024-01-02", "2024-12-31")}.get(y, RUNS["2018"])
        bh = lambda k, s, a, z: (np.concatenate([[1.0], s["c"][a:z + 1] / s["c"][a - 1]]), np.ones(z - a + 1, bool), [])  # noqa: E731
        r1 = stats(run(D, members(D, d0), d0, d1, bh), d0, d1)
        r2 = stats(run(D, members(D, d0, EXCLUDE_2018_ALT), d0, d1, bh), d0, d1)
        q[y] = {"BH_전체": r1["포트_수익"], "BH_QUBT제외": r2["포트_수익"], "QUBT_포함": "QUBT" in members(D, d0)}
    out["QUBT_영향"] = q


if __name__ == "__main__":
    main()
