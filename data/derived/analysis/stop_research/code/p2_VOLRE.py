"""2단계 갈래 VOLRE(2026-10-10): 손절 뒤 '진폭·변동성이 진정되면 다시 사기'(사용자 아이디어 2)의 재진입 신호 질을 잰다.

질문: 판 뒤 '위아래 진폭이 미친 듯하지 않을 때' 되사면, 같은 나가는 규칙에서 무작위 날짜(위약)에 되사는 것보다 나은가.
엔진 stock_lab.py(공유, 수정 안 함). 대상 = AI 종목 전체(그 창 시작 전 252거래일 자료가 있는 주식, ETF 제외).
창 시작 상태: 창 첫날 직전 종가에 그 종목을 다 들고 있다(가치 1). 추적 고점·샹들리에 고가는 그 종가에서 새로 시작.

시점(엔진 규약)
  - 나가기 X1: t-1 종가 < SMA50[t-1] → t 시가 매도. 단 (재)진입 뒤 종가가 한 번이라도 SMA50 이상이었어야 켜진다
    (= 진입 뒤 하향 돌파). SMA50 아래에서 되산 다음 날 바로 다시 파는 기계적 왕복을 막기 위함. 창 시작은 c[a-1] ≥ SMA50이면 켜진 채 시작.
  - 나가기 X2: t일 장중 손절가 = (진입 뒤 최고 종가, t-1까지) × 0.8.  X3: (진입 뒤 최고 고가, t-1까지) − 3×ATR22[t-1] (LeBeau 원형).
  - 재진입: 신호는 s = t-1 종가까지의 값으로 판단 → t 시가에 전량 매수.
    진폭 신호는 판 날 e 뒤 3거래일이 지난 e+3 종가부터 본다(t ≥ e+4). 기준·위약은 판 날 종가부터(t ≥ e+1).
  - 같은 날 순서: 시가 거래 → 장중 손절(되산 날에도 손절가는 살아 있다).
"""
from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stock_lab as L  # noqa: E402

# ============================================================================ 사전 고정 격자(뒤·2025를 본 뒤 바꾸지 않는다)
EXITS = {"X1": "종가<SMA50(진입 뒤 하향 돌파) → 다음 시가", "X2": "추적 −20%(진입 뒤 최고 종가, 장중 스탑)",
         "X3": "샹들리에 진입 뒤 최고 고가 − 3×ATR22(장중 스탑)"}
VOL = [  # (코드, 문턱, 설명) — 모두 s=t-1 종가까지의 값
    ("v1", 0.70, "ATR5/ATR20 < 0.70"), ("v1", 0.85, "ATR5/ATR20 < 0.85"), ("v1", 1.00, "ATR5/ATR20 < 1.00"),
    ("v2", 1.00, "10일 평균 일중진폭 ÷ 직전 252일 중앙 < 1.0"), ("v2", 1.20, "10일 평균 일중진폭 ÷ 직전 252일 중앙 < 1.2"),
    ("v2", 1.50, "10일 평균 일중진폭 ÷ 직전 252일 중앙 < 1.5"),
    ("v3", 0.80, "rvol10/rvol60 < 0.8"), ("v3", 1.00, "rvol10/rvol60 < 1.0"),
    ("v4", 0.20, "볼린저 폭(20,2σ) 직전 252일 백분위 < 20%"), ("v4", 0.40, "볼린저 폭(20,2σ) 직전 252일 백분위 < 40%"),
    ("v5", None, "NR7(오늘 고가−저가가 7일 중 최소)"),
    ("v6", 1.00, "10일 상승일 진폭 합 ÷ 하락일 진폭 합 ≥ 1.0"), ("v6", 1.50, "10일 상승일 진폭 합 ÷ 하락일 진폭 합 ≥ 1.5"),
    ("v7", 1.00, "5일 평균 |일수익| ≤ 직전 252일 중앙 × 1.0"), ("v7", 1.30, "5일 평균 |일수익| ≤ 직전 252일 중앙 × 1.3"),
]
CONFIRM = {"a": "단독", "b": "+종가>SMA10"}
BASES = [("line", None), ("time", 10), ("time", 20), ("time", 40)]   # + 매수·보유 1(나가기와 무관, 한 번만 돌림)
MIN_WAIT = 3                      # 진폭 신호: 판 날 e 뒤 e+3 종가부터
PLACEBO_SEEDS = (11, 22, 33)      # 위약 시드 3개 평균
# 위약 A(과제 명시): 나가기별, 그 나가기의 진폭 재진입 30개가 2026 실행에서 만든 닫힌 주기 '나가 있던 날 수'를 모은 분포에서 뽑는다.
# 위약 B(보조, 사전 선언): 진폭 변형마다 자기 2026 닫힌 주기 분포에서 뽑는다(짝 위약). 닫힌 주기 20개 미만이면 생략.
MATCHED_MIN = 20
SLIP_SENS = 0.005                 # 대표 후보 체결 미끄러짐 민감도(사고팔 때 모두 0.5% 불리)
MAIN_RUNS = ["2026", "2025", "2022"]
CTX_RUNS = ["2024", "2023", "2021", "2020", "2019", "2018"]   # 고른 변형·기준만, 맥락
G2_MIN_CYCLES = 30
N_DECLARED = len(EXITS) * (len(VOL) * len(CONFIRM) + len(BASES) + 1)   # 105(매수·보유 3칸은 같은 하나)

OUT_JSON = L.RESULTS / "p2_VOLRE.json"
OUT_MD = L.RESULTS / "p2_VOLRE.md"

# ============================================================================ 지표(종목별 1회 계산, [t]에 t까지의 값)
_F: dict = {}


def feats(b: L.Bars) -> dict:
    if b.sym in _F:
        return _F[b.sym]
    c, h, l = b.c, b.h, b.l
    n = len(c)
    rp = b.range_pct()
    rp10 = pd.Series(rp).rolling(10, min_periods=10).mean()
    rp10_med = rp10.shift(1).rolling(252, min_periods=252).median()
    bbw = b.bb_width(20, 2.0)
    pct = np.full(n, np.nan)
    if n > 253:
        W = sliding_window_view(bbw[:-1], 252)        # W[i] = bbw[i..i+251] → t = i+252 의 직전 252일
        cur = bbw[252:]
        ok = ~np.isnan(W).any(axis=1) & ~np.isnan(cur)
        pct[252:] = np.where(ok, (W < cur[:, None]).mean(axis=1), np.nan)
    ra = h - l
    nr7 = ra <= pd.Series(ra).rolling(7, min_periods=7).min().to_numpy() + 1e-12
    up = np.r_[False, c[1:] > c[:-1]]
    dn = np.r_[False, c[1:] < c[:-1]]
    rpz = np.nan_to_num(rp)
    ups = pd.Series(np.where(up, rpz, 0.0)).rolling(10, min_periods=10).sum().to_numpy()
    dns = pd.Series(np.where(dn, rpz, 0.0)).rolling(10, min_periods=10).sum().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        v6 = np.where(dns > 0, ups / dns, np.where(ups > 0, np.inf, np.nan))
        v6[np.isnan(dns)] = np.nan
    ar = pd.Series(np.abs(b.ret()))
    m5 = ar.rolling(5, min_periods=5).mean()
    m5_med = m5.shift(1).rolling(252, min_periods=252).median()
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = {"v1": b.atr(5) / b.atr(20), "v2": (rp10 / rp10_med).to_numpy(), "v3": b.rvol(10) / b.rvol(60),
               "v4": pct, "v6": v6, "v7": (m5 / m5_med).to_numpy()}
    sig = {}
    for code, thr, _ in VOL:
        if code == "v5":
            sig[(code, thr)] = nr7
        elif code == "v6":
            sig[(code, thr)] = raw["v6"] >= thr          # NaN → False
        elif code == "v7":
            sig[(code, thr)] = raw["v7"] <= thr
        else:
            sig[(code, thr)] = raw[code] < thr
    f = {"sig": sig, "sma10": b.sma(10), "sma50": b.sma(50), "atr22": b.atr(22), "hh20": b.high_n(20)}
    _F[b.sym] = f
    return f


# ============================================================================ 정책
class VolRe(L.Policy):
    """spec = (exit, kind, param, confirm). kind: vol(param=(code,thr)) | line | time(param=N) | placebo(param=pool) | bh."""

    def __init__(self, spec, rng=None):
        self.exit, self.kind, self.param, self.confirm = spec
        self.rng = rng
        self.armed = None
        self.N = None
        self.N_for = None

    def on_open(self, b, t, st):
        if self.kind == "bh":
            return None
        f = feats(b)
        s = t - 1
        c = b.c
        if st["pos"] > 1e-9:
            if self.exit == "X1":
                if self.armed is None:                       # 창 첫날
                    self.armed = bool(c[s] >= f["sma50"][s])
                if self.armed and c[s] < f["sma50"][s]:
                    return 0.0
            return None
        d = st["days_out"]                                    # = t − 판 날
        if d < 1:
            return None
        buy = False
        if self.kind == "vol":
            if d >= MIN_WAIT + 1 and f["sig"][self.param][s]:
                buy = (self.confirm == "a") or bool(c[s] > f["sma10"][s])
        elif self.kind == "line":
            if self.exit == "X1":
                buy = bool(c[s] > f["sma50"][s])
            else:
                buy = bool(s >= 1 and c[s] > f["hh20"][s - 1])
        elif self.kind == "time":
            buy = d >= self.param
        elif self.kind == "placebo":
            if self.N_for != st["exit_t"]:
                self.N = int(self.rng.choice(self.param))
                self.N_for = st["exit_t"]
            buy = d >= self.N
        if buy:
            self.armed = False
            return 1.0
        return None

    def stop(self, b, t, st):
        if self.kind == "bh" or st["pos"] <= 1e-9:
            return None
        if self.exit == "X2":
            return st["peak"] * 0.8
        if self.exit == "X3":
            a = feats(b)["atr22"][t - 1]
            return st["high_peak"] - 3.0 * a if np.isfinite(a) else None
        return None

    def after_close(self, b, t, st):
        if self.exit == "X1" and self.kind != "bh" and st["pos"] > 1e-9 and self.armed is not None:
            if b.c[t] >= feats(b)["sma50"][t]:
                self.armed = True


# ============================================================================ 실행
def label(spec) -> str:
    ex, kind, p, cf = spec
    if kind == "bh":
        return "매수·보유"
    if kind == "vol":
        return f"{ex}·{p[0]}{'' if p[1] is None else f'@{p[1]:g}'}·{cf}"
    if kind == "line":
        return f"{ex}·선회복"
    if kind == "time":
        return f"{ex}·시간{p}"
    return f"{ex}·위약"


def rule_text(spec) -> str:
    ex, kind, p, cf = spec
    if kind == "bh":
        return "매수·보유"
    if kind == "vol":
        desc = next(d for c_, t_, d in VOL if (c_, t_) == p)
        return f"{EXITS[ex]} / 재진입: 판 날 e+3 종가부터 [{desc}]{' 그리고 종가>SMA10' if cf == 'b' else ''} → 다음 시가"
    if kind == "line":
        return f"{EXITS[ex]} / 재진입: " + ("종가>SMA50" if ex == "X1" else "종가>직전 20일 최고 고가") + " → 다음 시가"
    if kind == "time":
        return f"{EXITS[ex]} / 재진입: 판 뒤 {p}거래일째 시가"
    return f"{EXITS[ex]} / 재진입: 2026 진폭 재진입 '나가 있던 날 수' 분포에서 무작위"


def cycle_extra(cycles, data, d0, d1) -> dict:
    cyc = [cy for cy in cycles if d0 <= cy["exit_day"] <= d1]
    closed = [cy for cy in cyc if not cy["open"]]
    late, drop, dout = [], [], []
    for cy in closed:
        b = data[cy["sym"]]
        i, j = b.idx(cy["exit_day"]), b.idx(cy["re_day"])
        tr = min(cy["exit_px"], float(b.c[i:j].min())) if j > i else cy["exit_px"]
        late.append(cy["re_px"] / tr - 1)
        drop.append(tr / cy["exit_px"] - 1)
        dout.append(cy["days_out"])
    med = lambda x: float(np.median(x)) if len(x) else float("nan")  # noqa: E731
    return {"늦음_중앙": med(late), "늦음_평균": float(np.mean(late)) if late else float("nan"),
            "판뒤추가하락_중앙": med(drop), "나가있던날_중앙": med(dout),
            "나가있던날_평균": float(np.mean(dout)) if dout else float("nan"),
            "열린주기비율": float(1 - len(closed) / len(cyc)) if cyc else float("nan")}


def run_spec(args):
    spec, seed, runs, slip = args
    data = L.load()
    rng = np.random.default_rng(seed) if seed is not None else None
    res = L.run_universe(lambda: VolRe(spec, rng), runs={k: L.RUNS[k] for k in runs}, slip=slip)
    ws = L.window_stats(res)
    for w in ws:
        r = res["2026"] if w.startswith("2026") else res[w]
        d0, d1 = L.WINDOWS[w]
        ws[w].update(cycle_extra(r["cycles"], data, d0, d1))
    pool = None
    if "2026" in res:
        pool = [cy["days_out"] for cy in res["2026"]["cycles"] if not cy["open"]]
    return {"spec": spec, "seed": seed, "slip": slip, "ws": ws, "pool26": pool}


def avg_ws(items):
    """시드 평균(수치 키별 평균)."""
    out = {}
    for w in items[0]["ws"]:
        keys = items[0]["ws"][w].keys()
        out[w] = {k: float(np.nanmean([it["ws"][w][k] for it in items])) for k in keys}
    return out


def main():
    t0 = time.time()
    specs = []
    for ex in EXITS:
        for code, thr, _ in VOL:
            for cf in CONFIRM:
                specs.append((ex, "vol", (code, thr), cf))
        for kind, p in BASES:
            specs.append((ex, kind, p, None))
    bh = ("-", "bh", None, None)
    assert len(specs) + len(EXITS) == N_DECLARED, (len(specs), N_DECLARED)
    with Pool(6) as pool:
        main_out = pool.map(run_spec, [(s, None, MAIN_RUNS, 0.0) for s in specs + [bh]])
        print(f"본 격자 {len(specs)}+1 완료 {time.time() - t0:.0f}s", flush=True)
        R = {label(o["spec"]): o for o in main_out}
        # 위약 A(나가기별 공동 분포)
        poolA = {ex: sum((R[label(s)]["pool26"] for s in specs if s[0] == ex and s[1] == "vol"), []) for ex in EXITS}
        jobs = [((ex, "placebo", tuple(poolA[ex]), None), sd, MAIN_RUNS, 0.0) for ex in EXITS for sd in PLACEBO_SEEDS]
        # 위약 B(짝 위약)
        mjobs = []
        for s in specs:
            if s[1] == "vol" and len(R[label(s)]["pool26"]) >= MATCHED_MIN:
                for sd in PLACEBO_SEEDS:
                    mjobs.append(((s[0], "placebo", tuple(R[label(s)]["pool26"]), None), sd, MAIN_RUNS, 0.0))
        pl_out = pool.map(run_spec, jobs + mjobs)
        print(f"위약 {len(jobs)}+{len(mjobs)} 완료 {time.time() - t0:.0f}s", flush=True)
    placeboA = {ex: avg_ws([o for o in pl_out[:len(jobs)] if o["spec"][0] == ex]) for ex in EXITS}
    # 위약 시드 간 흔들림(시드 3개 표준편차) — 진폭 변형과 위약의 차이가 이것보다 큰지 보려고
    seed_sd = {"A": {}, "B": {}}
    for ex in EXITS:
        its = [o for o in pl_out[:len(jobs)] if o["spec"][0] == ex]
        seed_sd["A"][ex] = {w: {k_: float(np.std([o["ws"][w][k_] for o in its], ddof=1)) for k_ in ["기대값", "효율선초과"]}
                            for w in its[0]["ws"]}
    matched = {}
    k = 0
    for s in specs:
        if s[1] == "vol" and len(R[label(s)]["pool26"]) >= MATCHED_MIN:
            its = pl_out[len(jobs) + k: len(jobs) + k + len(PLACEBO_SEEDS)]
            matched[label(s)] = avg_ws(its)
            seed_sd["B"][label(s)] = {w: {k_: float(np.std([o["ws"][w][k_] for o in its], ddof=1))
                                          for k_ in ["기대값", "효율선초과"]} for w in its[0]["ws"]}
            k += len(PLACEBO_SEEDS)
    seed_sd_med = {w: {k_: float(np.median([v[w][k_] for v in seed_sd["B"].values()])) for k_ in ["기대값", "효율선초과"]}
                   for w in ["2026", "2026앞", "2026뒤", "2025", "2022"]}

    # ---- 고르기(앞 구간만 보고)
    cand = [label(s) for s in specs]
    W = {lb: R[lb]["ws"] for lb in cand}
    front = lambda lb, key: W[lb]["2026앞"][key]  # noqa: E731
    g1 = max(cand, key=lambda lb: front(lb, "효율선초과"))
    g2_pool = [lb for lb in cand if W[lb]["2026앞"]["손절주기"] >= G2_MIN_CYCLES and np.isfinite(front(lb, "기대값"))]
    g2 = max(g2_pool, key=lambda lb: front(lb, "기대값"))
    g3 = max(cand, key=lambda lb: front(lb, "포트_수익") / max(1e-9, -front(lb, "포트_낙폭")))
    picks = {"G1": g1, "G2": g2, "G3": g3}
    top_g1 = sorted(cand, key=lambda lb: -front(lb, "효율선초과"))[:5]
    # 보조: 같은 세 고르기를 진폭 변형 90개 안에서만(진폭 신호 자체의 최선이 뒤·2025에서 어떤지)
    vcand = [lb for lb in cand if "·v" in lb]
    picks_vol = {
        "G1v": max(vcand, key=lambda lb: front(lb, "효율선초과")),
        "G2v": max([lb for lb in vcand if W[lb]["2026앞"]["손절주기"] >= G2_MIN_CYCLES], key=lambda lb: front(lb, "기대값")),
        "G3v": max(vcand, key=lambda lb: front(lb, "포트_수익") / max(1e-9, -front(lb, "포트_낙폭"))),
    }

    # 고르기에 정보가 있나
    pas = [lb for lb in cand if front(lb, "효율선초과") > 0]
    def frac(lst, cond):
        return (sum(1 for lb in lst if cond(lb)) / len(lst)) if lst else float("nan")
    info = {
        "앞통과_개수": len(pas), "전체_개수": len(cand),
        "통과중_뒤>0": frac(pas, lambda lb: W[lb]["2026뒤"]["효율선초과"] > 0),
        "전체중_뒤>0": frac(cand, lambda lb: W[lb]["2026뒤"]["효율선초과"] > 0),
        "통과중_2025>0": frac(pas, lambda lb: W[lb]["2025"]["효율선초과"] > 0),
        "전체중_2025>0": frac(cand, lambda lb: W[lb]["2025"]["효율선초과"] > 0),
        "통과중_뒤·2025>0": frac(pas, lambda lb: W[lb]["2026뒤"]["효율선초과"] > 0 and W[lb]["2025"]["효율선초과"] > 0),
        "전체중_뒤·2025>0": frac(cand, lambda lb: W[lb]["2026뒤"]["효율선초과"] > 0 and W[lb]["2025"]["효율선초과"] > 0),
        "통과_목록": pas,
    }

    # 위약 대비(진폭 변형 90개)
    vol_lbs = [label(s) for s in specs if s[1] == "vol"]
    vs = {}
    for w in ["2026", "2026앞", "2026뒤", "2025", "2022"]:
        rowA = {}
        for key in ["기대값", "승률", "손익비", "효율선초과", "포트_수익"]:
            rowA[key + "_A이김"] = frac(vol_lbs, lambda lb: W[lb][w][key] > placeboA[lb.split("·")[0]][w][key])
            ml = [lb for lb in vol_lbs if lb in matched]
            rowA[key + "_B이김"] = frac(ml, lambda lb: W[lb][w][key] > matched[lb][w][key])
        vs[w] = rowA
    # 차이의 크기(진폭 변형 − 짝 위약) 중앙
    diffB = {w: {key: float(np.nanmedian([W[lb][w][key] - matched[lb][w][key] for lb in vol_lbs if lb in matched]))
                 for key in ["기대값", "효율선초과", "늦음_중앙", "나가있던날_중앙"]} for w in ["2026", "2026뒤", "2025"]}

    # ---- 대표(G1 1위) 미끄러짐·맥락, 고른 변형·기준 맥락
    spec_of = {label(s): s for s in specs}
    ctx_specs = sorted({spec_of[g] for g in list(picks.values()) + list(picks_vol.values())} |
                       {s for s in specs if s[1] == "line"} | {bh}, key=lambda s: label(s))
    slip_lbs = list(dict.fromkeys([g1, g2, g3]))          # G1 = 대표(과제 명시), G2·G3는 보조
    with Pool(6) as pool:
        extra = pool.map(run_spec, [(spec_of[lb], None, MAIN_RUNS, SLIP_SENS) for lb in slip_lbs] +
                         [(s, None, CTX_RUNS, 0.0) for s in ctx_specs])
    slip_all = {lb: extra[i]["ws"] for i, lb in enumerate(slip_lbs)}
    slip_ws = slip_all[g1]
    ctx = {label(o["spec"]): o["ws"] for o in extra[len(slip_lbs):]}
    print(f"맥락 완료 {time.time() - t0:.0f}s", flush=True)

    payload = {
        "생성": "2026-10-10", "스크립트": "code/p2_VOLRE.py", "격자_선언": N_DECLARED,
        "격자_실제": {"서로다른_변형": len(specs), "매수보유": 1, "위약A": len(EXITS), "위약A_시드": list(PLACEBO_SEEDS),
                   "위약B_짝": len(matched)},
        "정의": {"나가기": EXITS, "진폭": [{"code": c_, "thr": t_, "desc": d_} for c_, t_, d_ in VOL], "판": CONFIRM,
               "최소대기": MIN_WAIT},
        "rules": {lb: rule_text(spec_of[lb]) for lb in cand},
        "결과": {lb: W[lb] for lb in cand}, "매수보유": R["매수·보유"]["ws"],
        "위약A": placeboA, "위약A_분포길이": {ex: len(poolA[ex]) for ex in EXITS},
        "위약A_분포_중앙": {ex: float(np.median(poolA[ex])) for ex in EXITS},
        "위약B": matched, "picks": picks, "picks_진폭만": picks_vol, "G1_상위5": top_g1, "고르기_정보": info,
        "위약대비_이긴비율": vs, "짝위약_차이_중앙": diffB, "위약_시드표준편차": seed_sd, "짝위약_시드표준편차_중앙": seed_sd_med,
        "대표_미끄러짐": slip_ws, "미끄러짐_고른3": slip_all, "맥락": ctx,
    }
    L.save("p2_VOLRE", payload)
    write_md(payload)
    print(f"끝 {time.time() - t0:.0f}s", flush=True)


# ============================================================================ 보고서
CONCL = [
    "**진폭이 잦아들면 되사기는 무작위 날짜에 되사기보다 낫지 않았다.** 진폭 변형 90개 중 '같은 대기 일수 분포에서 날짜만 무작위'인 짝 위약보다 "
    "주기 기대값이 높은 비율은 2026 20%, 뒤 29%, 2025 39%다(50%면 무작위와 같음). 승률은 10%만 위약보다 높았다. "
    "중앙 차이는 주기당 −0.6%p, 효율선초과 −3.5%p다(위약 시드 흔들림 0.36%p·3.0%p).",
    "**이유: 진폭이 가라앉을 때는 이미 반등이 진행된 뒤다.** 진폭 단독 재진입은 판 뒤 최저 종가보다 중앙 5~8% 위에서 되샀다. "
    "같은 날 수의 무작위보다 1.0~1.5%p 비쌌다. 가격 확인(종가>SMA10)을 붙이면 손익비는 1.1~1.3으로 오르지만 승률이 내려가 기대값은 그대로다(−0.7~−3.3%). "
    "사용자가 말한 '위아래 진폭 비율'(v6)만 위약보다 주기당 +0.1~0.3%p 나았다. 하지만 뒤 구간 포트 효율선초과는 12개 중 1개만 위약을 이겼다(사후 분해, 근거 아님).",
    "**앞 구간 고르기 3종은 모두 진폭이 아니라 시간 재진입을 골랐다.** G1 = SMA50 하향 돌파 매도 → 20거래일째 시가 재매수다. "
    "뒤 구간 +1.6%/−14.8%(매수·보유 −4.1%/−25.5%, 주기 승률 59%·손익비 1.69·기대값 +5.2%)였다. 그러나 2025 효율선초과 −8.6%이고, 미끄러짐 0.5%면 뒤 −0.3%가 된다. "
    "판정: 진폭 재진입 기각. 시간 40일(G3)만 뒤·2025 모두 효율선초과 >0이었으나 미끄러짐 0.5%에 뒤 −0.9%로 뒤집힌다.",
]
LIMITS = [
    "생존편향: 종목은 2026-10 AI 목록(+TIME 보유 이력)이다. 빠졌다가 회복한 종목만 남아 '되사기' 전체가 좋게 나올 수 있다. "
    "다만 핵심 결론(진폭 신호 ≤ 무작위 날짜)은 같은 표본·같은 나가기 안의 비교라 이 편향에 덜 민감하다.",
    "탐색 범위: 본 격자 102개(+매수·보유 1)에 위약 A 3개×시드 3, 짝 위약 90개×시드 3을 더했다. 고르기는 사전 고정 3종이다. "
    "진폭 계열별 분해(7계열×3창×2지표 = 42칸)는 결과를 본 뒤 한 것이라 v6·v4가 낫다는 말은 가설일 뿐이다.",
    "위약 날 수 분포는 2026 전체 진폭 재진입 결과(사후)에서 뽑은 비교 기준이고 매매 근거가 아니다. 닫힌 주기만으로 만들었다. "
    "끝까지 못 되산 주기 비율은 진폭 쪽과 다를 수 있다(노출 차이). 효율선초과가 이를 일부 보정한다.",
    "뒤 구간(6/1~10/9) 매수·보유가 −4.1%라 고정 노출 최선은 노출 0(현금)이다. 따라서 뒤 구간 효율선초과는 '현금보다 벌었나'와 같다.",
    "X1은 '진입 뒤 하향 돌파'로 구현했다(결과를 보기 전 설계 선택). 상태형(종가<SMA50이면 언제나 매도)이면 SMA50 아래에서 진폭으로 되산 다음 날 바로 다시 팔게 된다. 그러면 하루 왕복이 의미 없이 쌓인다.",
    "주기 손익(판 가격 ÷ 되산 가격 − 1)에는 나가 있는 동안 피한 낙폭·현금 기회비용이 안 잡힌다. 효율선초과와 함께 봐야 한다. "
    "선 회복 재진입은 사용자가 말한 '승률 30%·손익비 높음' 꼴(X1 승률 26%·손익비 2.32)이지만 기대값은 −0.7%다.",
    "체결: 장중 스탑은 손절가 그대로 체결된다(갭이면 시가). 기본 미끄러짐은 0이다. 결과를 본 뒤 보고 항목을 더했다(위약 시드 표준편차, G2·G3 미끄러짐, 진폭만 고르기, 맥락 효율선초과). "
    "격자·정의·고르기는 바꾸지 않았고 첫 실행과 숫자가 같다. 버그 수정은 없다.",
    "장세 전환(맥락): G3(시간 40일)는 2018 효율선초과 −17.5%, 2024 −1.8%였다. 강세장에서는 수익을 크게 덜 받았다(2023 +47% vs 매수·보유 +80%, 2024 +22% vs +62%).",
]


def P(x, d=1):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x * 100:+.{d}f}%"


def Q(x, d=0):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x * 100:.{d}f}%"


def F(x, d=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:.{d}f}"


def write_md(p):
    W, bh, A, B = p["결과"], p["매수보유"], p["위약A"], p["위약B"]
    L_ = []
    a = L_.append
    a("# 갈래 VOLRE: 진폭이 진정되면 다시 사기 (2026-10-10)\n")
    a("스크립트 `code/p2_VOLRE.py` · 전 변형 `results/p2_VOLRE.json` · 엔진 `code/stock_lab.py`\n")
    a("## 결론 3줄\n")
    for i_, t_ in enumerate(CONCL, 1):
        a(f"{i_}. {t_}")
    a("")
    a("## 격자 (사전 선언, 뒤·2025를 본 뒤 고치지 않음)\n")
    a("| 부 | 내용 | 개수 |\n|---|---|---|")
    a("| 나가기 | X1 종가<SMA50(진입 뒤 하향 돌파) · X2 추적 −20%(장중) · X3 샹들리에 3×ATR22(장중) | 3 |")
    a("| 진폭 신호 | v1 ATR5/20<{0.7,0.85,1.0} · v2 10일 일중진폭÷252일 중앙<{1.0,1.2,1.5} · v3 rvol10/60<{0.8,1.0} · "
      "v4 볼린저 폭 백분위<{20,40%} · v5 NR7 · v6 상승일÷하락일 진폭 합≥{1.0,1.5} · v7 5일 평균 절대 일수익÷252일 중앙≤{1.0,1.3} | 15 |")
    a("| 판 | (a) 단독 · (b) +종가>SMA10 | 2 |")
    a("| 기준 | 선 회복(X1 SMA50 위, X2·X3 직전 20일 최고 고가 돌파) · 시간 10·20·40일 · 매수·보유 | 5 |")
    a(f"| 합계 | 3×(15×2+5) = {p['격자_선언']}(매수·보유 3칸은 같은 하나 → 서로 다른 변형 102 + 매수·보유 1) | {p['격자_선언']} |")
    a(f"| 위약 A | 나가기별 1개 × 시드 3, 분포 = 그 나가기의 진폭 재진입 30개가 2026에 만든 닫힌 주기 '나가 있던 날 수'"
      f"(X1 {p['위약A_분포길이']['X1']}개 중앙 {p['위약A_분포_중앙']['X1']:.0f}일, X2 {p['위약A_분포길이']['X2']}개 중앙 "
      f"{p['위약A_분포_중앙']['X2']:.0f}일, X3 {p['위약A_분포길이']['X3']}개 중앙 {p['위약A_분포_중앙']['X3']:.0f}일) | 3 |")
    a(f"| 위약 B(보조, 사전 선언) | 진폭 변형마다 자기 2026 닫힌 주기 분포에서 뽑는 짝 위약 × 시드 3(닫힌 주기 {MATCHED_MIN}개 미만 생략) | "
      f"{p['격자_실제']['위약B_짝']} |")
    a("\n고르기에 쓰는 후보는 102개(진폭 90 + 기준 12). 대표 후보 = G1 1위.\n")
    a("## 변수 시점\n")
    a("| 변수 | 언제 아는가 | 체결 |\n|---|---|---|")
    a("| SMA50·SMA10, 20일 최고 고가(s−20~s−1) | s일 종가 | s+1 시가 |")
    a("| 추적 고점(진입 뒤 최고 종가), 샹들리에 고가(진입 뒤 최고 고가), ATR22 | t−1 종가까지 | t일 장중 스탑(시가가 이미 아래면 시가) |")
    a("| ATR5/ATR20, 10일 일중진폭과 그 직전 252일 중앙, rvol10/60, 볼린저 폭 백분위(직전 252일), NR7, 상승·하락일 진폭 합, "
      "5일 평균 절대 일수익과 직전 252일 중앙 | s일 종가(판 날 e 뒤 s ≥ e+3) | s+1 시가 |")
    a("| 위약 날 수 N | 2026 전체 진폭 재진입 결과에서 뽑음 — **사후 분포**(비교 기준일 뿐 매매 근거 아님) | 판 날 + N 시가 |")
    a("| 늦게 산 정도 = 되산 가격 ÷ 판 뒤 최저 종가 − 1 | 사후 측정(설명용) | — |\n")

    def row(lb, ws, rule=None):
        g = lambda w, k: ws[w][k] if w in ws else float("nan")  # noqa: E731
        return (f"| {lb} | {P(g('2026','포트_수익'))}/{P(g('2026','포트_낙폭'))} | {P(g('2026앞','포트_수익'))} · "
                f"초과 {P(g('2026앞','효율선초과'))} | {P(g('2026뒤','포트_수익'))}/{P(g('2026뒤','포트_낙폭'))} · 초과 "
                f"{P(g('2026뒤','효율선초과'))} | {P(g('2025','포트_수익'))}/{P(g('2025','포트_낙폭'))} · 초과 "
                f"{P(g('2025','효율선초과'))} | {F(g('2026','평균노출'))} |")

    a("## 고르기 결과 (수익/최대 낙폭, 초과 = 효율선초과)\n")
    a(f"매수·보유(동일가중): 2026 {P(bh['2026']['포트_수익'])}/{P(bh['2026']['포트_낙폭'])}, 앞 {P(bh['2026앞']['포트_수익'])}/"
      f"{P(bh['2026앞']['포트_낙폭'])}, 뒤 {P(bh['2026뒤']['포트_수익'])}/{P(bh['2026뒤']['포트_낙폭'])}, 2025 "
      f"{P(bh['2025']['포트_수익'])}/{P(bh['2025']['포트_낙폭'])}, 2022 {P(bh['2022']['포트_수익'])}/{P(bh['2022']['포트_낙폭'])}\n")
    a("| 변형 | 2026 | 앞(고르기) | 뒤(확인) | 2025 | 노출 | 2026 나아진 종목 · 최악 5% 종목(BH) |\n"
      "|---|---|---|---|---|---|---|")

    def tail(ws):
        s = ws["2026"]
        return f" {Q(s['수익_나은종목비율'])} · {P(s['최악5%_종목수익'])}({P(s['최악5%_BH'])}) |"
    for g, lb in p["picks"].items():
        a(row(f"**{g}** {lb}", W[lb]) + tail(W[lb]))
    for g, lb in p.get("picks_진폭만", {}).items():
        a(row(f"{g}(진폭만, 보조) {lb}", W[lb]) + tail(W[lb]))
    for ex in EXITS:
        a(row(f"위약 A {ex}", A[ex]) + tail(A[ex]))
    for lb, ws in p.get("미끄러짐_고른3", {"G1": p["대표_미끄러짐"]}).items():
        a(row(f"{lb} +미끄러짐 0.5%", ws) + tail(ws))
    a("")
    a("(진폭만 = 같은 세 고르기를 진폭 변형 90개 안에서만 적용한 보조 결과. 미끄러짐 0.5%는 사고팔 때 모두 불리하게. "
      "G1이 과제 명시 대표, G2·G3 미끄러짐은 보조.)\n")
    sd = p.get("짝위약_시드표준편차_중앙")
    if sd:
        a("위약 시드 3개 사이 흔들림(짝 위약 표준편차의 중앙): " + "; ".join(
            f"{w} 기대값 {Q(sd[w]['기대값'], 2)} · 효율선초과 {Q(sd[w]['효율선초과'], 1)}" for w in sd) + "\n")
    a("G1 상위 5(앞 효율선초과): " + ", ".join(f"{lb} {P(W[lb]['2026앞']['효율선초과'])}→뒤 {P(W[lb]['2026뒤']['효율선초과'])}"
                                          for lb in p["G1_상위5"]) + "\n")
    i = p["고르기_정보"]
    a("고르기에 정보가 있나(효율선초과 > 0 비율)\n")
    a("| 집합 | 개수 | 뒤 > 0 | 2025 > 0 | 뒤·2025 모두 > 0 |\n|---|---|---|---|---|")
    a(f"| 앞 > 0 통과 | {i['앞통과_개수']} | {Q(i['통과중_뒤>0'])} | {Q(i['통과중_2025>0'])} | {Q(i['통과중_뒤·2025>0'])} |")
    a(f"| 전체(무조건) | {i['전체_개수']} | {Q(i['전체중_뒤>0'])} | {Q(i['전체중_2025>0'])} | {Q(i['전체중_뒤·2025>0'])} |\n")

    a("## 주기 승률·손익비 (손절 1회 = 판 가격 ÷ 되산 가격 − 1, +면 싸게 되산 것)\n")
    for w, nm in [("2026", "2026 전체"), ("2026뒤", "2026 뒤"), ("2025", "2025")]:
        a(f"### {nm}\n")
        a("| 나가기·재진입 | 주기 | 승률 | 평균이익 | 평균손실 | 손익비 | 기대값 | 늦음 중앙 | 나가있던 날 중앙 | 열린 주기 | 효율선초과 |")
        a("|---|---|---|---|---|---|---|---|---|---|---|")
        for ex in EXITS:
            lbs = [f"{ex}·선회복", f"{ex}·시간10", f"{ex}·시간20", f"{ex}·시간40"]
            for lb in lbs:
                s = W[lb][w]
                a(f"| {lb} | {s['손절주기']} | {Q(s['승률'])} | {P(s['평균이익'])} | {P(s['평균손실'])} | {F(s['손익비'])} | "
                  f"{P(s['기대값'])} | {P(s['늦음_중앙'])} | {F(s['나가있던날_중앙'], 0)} | {Q(s['열린주기비율'])} | {P(s['효율선초과'])} |")
            s = A[ex][w]
            a(f"| {ex}·위약 A | {s['손절주기']:.0f} | {Q(s['승률'])} | {P(s['평균이익'])} | {P(s['평균손실'])} | {F(s['손익비'])} | "
              f"{P(s['기대값'])} | {P(s['늦음_중앙'])} | {F(s['나가있던날_중앙'], 0)} | {Q(s['열린주기비율'])} | {P(s['효율선초과'])} |")
            vl = [lb for lb in W if lb.startswith(ex) and "·v" in lb]
            for cf in ("a", "b"):
                sub = [lb for lb in vl if lb.endswith("·" + cf)]
                md = lambda k: float(np.nanmedian([W[lb][w][k] for lb in sub]))  # noqa: E731
                a(f"| {ex}·진폭({cf}) 15개 중앙 | {md('손절주기'):.0f} | {Q(md('승률'))} | {P(md('평균이익'))} | {P(md('평균손실'))} | "
                  f"{F(md('손익비'))} | {P(md('기대값'))} | {P(md('늦음_중앙'))} | {F(md('나가있던날_중앙'), 0)} | "
                  f"{Q(md('열린주기비율'))} | {P(md('효율선초과'))} |")
        a("")
    a("### 진폭 변형 90개가 위약을 이긴 비율\n")
    a("| 창 | 기대값 A | 기대값 B | 승률 B | 손익비 B | 효율선초과 A | 효율선초과 B | 포트 수익 B |\n|---|---|---|---|---|---|---|---|")
    for w, r in p["위약대비_이긴비율"].items():
        a(f"| {w} | {Q(r['기대값_A이김'])} | {Q(r['기대값_B이김'])} | {Q(r['승률_B이김'])} | {Q(r['손익비_B이김'])} | "
          f"{Q(r['효율선초과_A이김'])} | {Q(r['효율선초과_B이김'])} | {Q(r['포트_수익_B이김'])} |")
    a("\n(A = 나가기별 공동 분포 위약, B = 변형별 짝 위약. 50% 근처면 무작위 날짜와 구분되지 않는다.)\n")
    d = p["짝위약_차이_중앙"]
    a("진폭 변형 − 짝 위약, 중앙: " + "; ".join(
        f"{w} 기대값 {P(d[w]['기대값'])} · 효율선초과 {P(d[w]['효율선초과'])} · 늦음 {P(d[w]['늦음_중앙'])}" for w in d) + "\n")

    a("### 진폭 변형 전체 표(2026 전체 · 뒤 · 2025: 기대값 / 효율선초과)\n")
    a("| 변형 | 2026 주기·승률·손익비·기대 | 2026 초과 | 뒤 기대·초과 | 2025 기대·초과 | 짝위약 2026 기대·초과 |\n|---|---|---|---|---|---|")
    for lb in W:
        if "·v" not in lb:
            continue
        s, sb, s5 = W[lb]["2026"], W[lb]["2026뒤"], W[lb]["2025"]
        m = B.get(lb)
        mt = f"{P(m['2026']['기대값'])} · {P(m['2026']['효율선초과'])}" if m else "—"
        a(f"| {lb} | {s['손절주기']}·{Q(s['승률'])}·{F(s['손익비'])}·{P(s['기대값'])} | {P(s['효율선초과'])} | "
          f"{P(sb['기대값'])} · {P(sb['효율선초과'])} | {P(s5['기대값'])} · {P(s5['효율선초과'])} | {mt} |")
    a("")
    a("## 보조·맥락 (2022 및 2018~2024는 장세 전환 위험을 보는 맥락)\n")
    a("수익/최대 낙폭 · 효율선초과\n")
    a("| 변형 | 2022 | 2024 | 2023 | 2021 | 2020 | 2019 | 2018 |\n|---|---|---|---|---|---|---|---|")
    for lb, ws in p["맥락"].items():
        w22 = W[lb]["2022"] if lb in W else p["매수보유"]["2022"]
        cells = [f"{P(w22['포트_수익'])}/{P(w22['포트_낙폭'])} · {P(w22['효율선초과'])}"]
        for y in ["2024", "2023", "2021", "2020", "2019", "2018"]:
            cells.append(f"{P(ws[y]['포트_수익'])}/{P(ws[y]['포트_낙폭'])} · {P(ws[y]['효율선초과'])}" if y in ws else "—")
        a(f"| {lb} | " + " | ".join(cells) + " |")
    a("")
    a("## 선택편향·한계\n")
    for t_ in LIMITS:
        a(f"- {t_}")
    a("")
    OUT_MD.write_text("\n".join(L_), encoding="utf-8")


if __name__ == "__main__":
    if "--md" in sys.argv:          # 결과 JSON으로 보고서만 다시 쓰기
        write_md(json.loads(OUT_JSON.read_text(encoding="utf-8")))
    else:
        main()
