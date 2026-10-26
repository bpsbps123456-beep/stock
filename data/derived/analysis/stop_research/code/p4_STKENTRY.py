"""4단계 STKENTRY(2026-10-11): 일반 주식을 신호로 사서 손절·청산 규칙으로 나오는 '거래 1건' 단위 시험.

질문: "7종목만이 아니라 일반 주식도 한다. 거기 쓸 만한 손절이 있나?" — 어떤 매매(진입 방식)에서
어떤 손절이 기대값을 덜 잃고 꼬리(최악 손실)를 가장 많이 자르나.

엔진: trade_lab(entries·trade·summary·compare·bad_bars). 엔진 파일은 고치지 않는다.
  - 거래 1건마다 trade_lab.trade를 '모든 다리(비중 1)'로 한 번 불러 다리별 청산일·청산가를 얻고,
    규칙 수익 = Σ 비중 × 다리 수익으로 조립한다(엔진의 다리 합산과 같은 식, check()에서 run()과 오차 0 확인).
  - 엔진이 다리별이라 '같은 다리에서 먼저 닿는 쪽' 규칙(O'Neil 묶음, 추적 −25% + 20일 시간)은
    두 다리의 청산일을 비교해 이른 쪽을 쓴다(같은 날: O'Neil은 시가가 익절선 위면 익절, 아니면 손절 먼저 = 보수적;
    추적+시간은 시간 청산이 시가라 먼저). check()에서 하루씩 도는 독립 구현과 대조.
  - 달력 포트폴리오(칼마용): 거래마다 그날 가치 경로(청산 전 종가/진입가, 청산 뒤 청산가/진입가)를 만들고
    같은 날 열린 거래의 일수익을 동일가중 평균(매일 재조정)한다. 청산 뒤 현금은 0%.

평가 규약(사전 고정, 과제 지시 그대로. 뒤·2025를 본 뒤 바꾸지 않는다)
  - 창(진입일 기준): 2026앞 1/2~5/29(고르기, 청산도 5/29 안), 2026뒤 6/1~10/9(확인), 2025(보조),
    맥락 2018~2024 각 해와 폭락 구간 2018Q4(10/1~12/24)·2020급락(2/19~3/23).
  - 보유 H: 2026 {20, 60}, 2025·맥락 {60, 120}. 2025 확인은 H60(두 H 공통)으로 한다.
  - 시점: 신호 s일 종가 → s+1일 시가 진입. 장중 손절선은 t−1까지 정보. 종가형 청산(SMA·시간)은 다음 날 시가.
  - 비용: 수수료 0, 미끄러짐 0.5% 민감도(2026·2025). 자료 오류 의심 종목(trade_lab.bad_bars) 제외.
  - 고르기: G1 앞 짝 평균차(규칙 − 들고만 있기) 최대, G2 앞 기대값(거래 30건 이상) 최대, G3 앞 달력 칼마 최대.

실행: py p4_STKENTRY.py          (전체 계산 → results/p4_STKENTRY.json, .md)
      py p4_STKENTRY.py --check  (엔진 대조 점검만)
      py p4_STKENTRY.py --md     (json에서 md만 다시 쓰기)
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import trade_lab as T

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
OUT_JSON = RESULTS / "p4_STKENTRY.json"
OUT_MD = RESULTS / "p4_STKENTRY.md"

# ============================================================================ 사전 고정 격자
ENTRY_KINDS = ("random", "dip7", "breakout", "pullback", "deep40")
ENTRY_KO = {"random": "아무 날(10일마다)", "dip7": "하루 −7% 급락 다음 날", "breakout": "252일 신고가 첫날 다음 날",
            "pullback": "200일선 위 20일 고점 −10% 첫날 다음 날", "deep40": "252일 고점 −40% 첫날 다음 날"}
GAP, EVERY = 20, 10                       # 같은 종목 같은 신호 최소 간격, random 주기(엔진 기본값)

# 엔진 다리(비중 1로 한 번에 돌림)
BASE = {
    "hold": T.Leg(1.0, "hold"),
    "fixed8": T.Leg(1.0, "fixed", 0.08), "fixed15": T.Leg(1.0, "fixed", 0.15), "fixed25": T.Leg(1.0, "fixed", 0.25),
    "trail10": T.Leg(1.0, "trail", 0.10), "trail15": T.Leg(1.0, "trail", 0.15), "trail20": T.Leg(1.0, "trail", 0.20),
    "trail25": T.Leg(1.0, "trail", 0.25), "trail35": T.Leg(1.0, "trail", 0.35),
    "chand3": T.Leg(1.0, "chand", 3.0),
    "sma20": T.Leg(1.0, "sma", 20), "sma50": T.Leg(1.0, "sma", 50), "sma100": T.Leg(1.0, "sma", 100),
    "time10": T.Leg(1.0, "time", 10), "time20": T.Leg(1.0, "time", 20),
    "target20": T.Leg(1.0, "target", 0.20),
}
BASE_NAMES = list(BASE)
LEGS = [BASE[k] for k in BASE_NAMES]
DERIVED = ("bracket", "trail25_time20")   # 같은 다리 '먼저 닿는 쪽'
PSEUDO = BASE_NAMES + list(DERIVED)

# 규칙 = [(가상 다리, 비중)]. 사전 고정 20개(들고만 있기 포함).
RULES = {
    "hold": [("hold", 1.0)],
    "fixed8": [("fixed8", 1.0)], "fixed15": [("fixed15", 1.0)], "fixed25": [("fixed25", 1.0)],
    "trail10": [("trail10", 1.0)], "trail15": [("trail15", 1.0)], "trail20": [("trail20", 1.0)],
    "trail25": [("trail25", 1.0)], "trail35": [("trail35", 1.0)],
    "chand3": [("chand3", 1.0)],
    "sma20": [("sma20", 1.0)], "sma50": [("sma50", 1.0)], "sma100": [("sma100", 1.0)],
    "time10": [("time10", 1.0)], "time20": [("time20", 1.0)],
    "oneil_split": [("target20", 0.5), ("fixed8", 0.5)],
    "oneil_bracket": [("bracket", 1.0)],
    "trail20_half": [("trail20", 0.5), ("hold", 0.5)],
    "trail15_quarter": [("trail15", 0.25), ("hold", 0.75)],
    "trail25_time20": [("trail25_time20", 1.0)],
}
RULE_KO = {
    "hold": "들고만 있기", "fixed8": "고정 −8%", "fixed15": "고정 −15%", "fixed25": "고정 −25%",
    "trail10": "추적 −10%", "trail15": "추적 −15%", "trail20": "추적 −20%", "trail25": "추적 −25%", "trail35": "추적 −35%",
    "chand3": "샹들리에 3×ATR22", "sma20": "종가 < 20일선", "sma50": "종가 < 50일선", "sma100": "종가 < 100일선",
    "time10": "10일째 진입가 이하면", "time20": "20일째 진입가 이하면",
    "oneil_split": "O'Neil 나눔(+20% 익절 50% · −8% 손절 50%)", "oneil_bracket": "O'Neil 묶음(+20%/−8% 먼저 닿는 쪽)",
    "trail20_half": "추적 −20% 50% + 보유 50%", "trail15_quarter": "추적 −15% 25% + 보유 75%",
    "trail25_time20": "추적 −25% + 20일 시간",
}
RULE_NAMES = list(RULES)
W = np.zeros((len(RULE_NAMES), len(PSEUDO)))
for i, rn in enumerate(RULE_NAMES):
    for leg, fr in RULES[rn]:
        W[i, PSEUDO.index(leg)] += fr
assert np.allclose(W.sum(1), 1.0)

# 창: 이름 → (진입 시작, 진입 끝, 청산 마감 또는 None, H 목록, 미끄러짐 목록)
H26, HOTHER = (20, 60), (60, 120)
SLIPS_MAIN = (0.0, 0.005)
WINDOWS = {
    "2026앞": ("2026-01-02", "2026-05-29", "2026-05-29", H26, SLIPS_MAIN),
    "2026뒤": ("2026-06-01", "2026-10-09", None, H26, SLIPS_MAIN),
    "2025": ("2025-01-02", "2025-12-31", None, HOTHER, SLIPS_MAIN),
    **{str(y): (f"{y}-01-01", f"{y}-12-31", None, HOTHER, (0.0,)) for y in range(2018, 2025)},
    "2018Q4": ("2018-10-01", "2018-12-24", None, HOTHER, (0.0,)),
    "2020급락": ("2020-02-19", "2020-03-23", None, HOTHER, (0.0,)),
}
CTX = [str(y) for y in range(2018, 2025)] + ["2018Q4", "2020급락"]
CRASH = ["2018Q4", "2020급락", "2022"]
H25_CHECK = 60
MIN_N_G2 = 30

N_COMBOS_2026 = len(ENTRY_KINDS) * len(RULE_NAMES) * len(H26)


# ============================================================================ 거래 1건 → 규칙 20개
def eval_trade(b, e: int, H: int, slip: float):
    """반환 (규칙 수익[R], 규칙 보유일[R], 들고만 있기 수익, 규칙 가치 경로[R×H]) 또는 None."""
    r = T.trade(b, e, H, LEGS, slip)
    if r is None:
        return None
    entry = b.o[e] * (1 + slip)
    trig, day, val = {}, {}, {}
    for name, (why, d, rr) in zip(BASE_NAMES, r["legs"]):
        trig[name], day[name], val[name] = why != "기간끝", d, 1.0 + rr
    # O'Neil 묶음: +20% 익절과 −8% 손절 중 먼저. 같은 날이면 시가 ≥ 익절선일 때만 익절, 아니면 손절(보수적).
    dt = day["target20"] if trig["target20"] else 10 ** 9
    df_ = day["fixed8"] if trig["fixed8"] else 10 ** 9
    if dt == df_ == 10 ** 9:
        pick = "hold"
    elif dt < df_ or (dt == df_ and b.o[e + dt] >= entry * 1.2):
        pick = "target20"
    else:
        pick = "fixed8"
    day["bracket"], val["bracket"] = day[pick], val[pick]
    # 추적 −25% + 20일 시간: 먼저 닿는 쪽(같은 날이면 시간 청산이 시가라 먼저).
    da = day["trail25"] if trig["trail25"] else 10 ** 9
    db = day["time20"] if trig["time20"] else 10 ** 9
    pick = "hold" if da == db == 10 ** 9 else ("time20" if db <= da else "trail25")
    day["trail25_time20"], val["trail25_time20"] = day[pick], val[pick]

    days = np.array([day[n] for n in PSEUDO], float)
    vals = np.array([val[n] for n in PSEUDO])
    cp = b.c[e:e + H] / entry
    k = np.arange(H)
    P = np.where(k[None, :] < days[:, None], cp[None, :], vals[:, None])
    V = W @ P
    ret = V[:, -1] - 1.0
    din = W @ (days + 1.0)
    return ret, din, r["bh"], V


# ============================================================================ 계산
def calendar_stats(Ssum: np.ndarray, Cnt: np.ndarray):
    """달력 포트폴리오(동일가중·매일 재조정) → 규칙별 (수익, 최대 낙폭, 연율, 칼마)."""
    act = np.flatnonzero(Cnt > 0)
    if not len(act):
        return None
    a, z = act[0], act[-1]
    r = np.where(Cnt[a:z + 1, None] > 0, Ssum[a:z + 1] / np.maximum(Cnt[a:z + 1, None], 1), 0.0)
    eq = np.vstack([np.ones((1, r.shape[1])), np.cumprod(1 + r, axis=0)])
    mdd = (eq / np.maximum.accumulate(eq, axis=0) - 1).min(axis=0)
    tot = eq[-1] - 1
    n = r.shape[0]
    cagr = np.sign(eq[-1]) * np.abs(eq[-1]) ** (252.0 / n) - 1
    calmar = np.where(mdd < -1e-9, cagr / np.abs(np.minimum(mdd, -1e-9)), np.nan)
    return {"수익": tot, "낙폭": mdd, "연율": cagr, "칼마": calmar, "일수": int(n)}


def group_stats(rows: dict, Ssum, Cnt, H: int) -> dict:
    """한 (창, 진입, H, 미끄러짐) 묶음의 규칙별 요약."""
    n = len(rows["entry"])
    if n == 0:
        return {"n": 0}
    R = np.array(rows["ret"])            # N×R
    D = np.array(rows["din"])
    bh = np.array(rows["bh"])
    entry = pd.Series(rows["entry"])
    cal = calendar_stats(Ssum, Cnt)
    hold_df = pd.DataFrame({"entry": entry, "ret": bh, "bh": bh, "days_in": np.full(n, float(H))})
    hold_sum = T.summary(hold_df)
    q5_bh = float(np.quantile(bh, 0.05))
    out = {"n": n, "종목수": int(len(set(rows["sym"]))), "월수": int(entry.str[:7].nunique()), "rules": {}}
    for j, rn in enumerate(RULE_NAMES):
        df = pd.DataFrame({"entry": entry, "ret": R[:, j], "bh": bh, "days_in": D[:, j]})
        s = T.summary(df)
        c = T.compare(df)
        # 같은 하위5%가 나오는 고정 비중 k(0~1)로 처음부터 덜 샀다면: 기대값 초과 = 규칙 기대값 − k × 보유 기대값
        q5r = s["하위5%"]
        k = float(np.clip(q5r / q5_bh, 0.0, 1.0)) if q5_bh < 0 and q5r < 0 else (0.0 if q5_bh < 0 else 1.0)
        rec = {**s, **c, "보유_기대값": hold_sum["기대값"], "보유_보유일당": hold_sum["보유일당"],
               "보유_승률": hold_sum["승률"], "보유_손익비": hold_sum["손익비"],
               "꼬리맞춤_k": k, "꼬리맞춤초과": float(s["기대값"] - k * hold_sum["기대값"]),
               "평균보유일": float(D[:, j].mean()), "청산비율": float((D[:, j] < H - 0.5).mean())}
        if cal is not None:
            rec.update({"달력_수익": float(cal["수익"][j]), "달력_낙폭": float(cal["낙폭"][j]),
                        "달력_연율": float(cal["연율"][j]), "달력_칼마": float(cal["칼마"][j]), "달력_일수": cal["일수"]})
        out["rules"][rn] = rec
    return out


def compute(data: dict, syms: list[str]) -> dict:
    G = np.array(sorted({d for s in syms for d in data[s].days if d >= "2017-01-01"}))
    gidx = {s: np.searchsorted(G, data[s].days) for s in syms}
    res = {}
    t0 = time.time()
    for w, (d0, d1, exit_by, Hs, slips) in WINDOWS.items():
        res[w] = {}
        for kind in ENTRY_KINDS:
            acc = {(H, sl): {"rows": {"ret": [], "din": [], "bh": [], "entry": [], "sym": []},
                             "S": np.zeros((len(G), len(RULE_NAMES))), "C": np.zeros(len(G))}
                   for H in Hs for sl in slips}
            for s in syms:
                b = data[s]
                es = T.entries(b, kind, d0, d1, GAP, EVERY)
                if not len(es):
                    continue
                es = es[(b.days[es] >= d0) & (b.days[es] <= d1)]
                for e in es:
                    for H in Hs:
                        z = e + H - 1
                        if z >= len(b.c):
                            continue
                        if exit_by is not None and b.days[z] > exit_by:
                            continue
                        gi = gidx[s][e:z + 1]
                        if gi[0] >= len(G) or b.days[e] < G[0]:
                            continue
                        for sl in slips:
                            out = eval_trade(b, e, H, sl)
                            if out is None:
                                continue
                            ret, din, bh, V = out
                            a = acc[(H, sl)]
                            a["rows"]["ret"].append(ret)
                            a["rows"]["din"].append(din)
                            a["rows"]["bh"].append(bh)
                            a["rows"]["entry"].append(b.days[e])
                            a["rows"]["sym"].append(s)
                            prev = np.concatenate([np.ones((V.shape[0], 1)), V[:, :-1]], axis=1)
                            a["S"][gi] += (V / prev - 1).T
                            a["C"][gi] += 1
            res[w][kind] = {}
            for (H, sl), a in acc.items():
                res[w][kind].setdefault(str(H), {})[f"{sl:.3f}"] = group_stats(a["rows"], a["S"], a["C"], H)
            ns = {f"H{H}/s{sl}": len(a["rows"]["entry"]) for (H, sl), a in acc.items()}
            print(f"{w} {kind} {ns} {time.time() - t0:.0f}s", flush=True)
    return res


# ============================================================================ 고르기
def get(res, w, kind, H, rule, key, slip="0.000"):
    try:
        return res[w][kind][str(H)][slip]["rules"][rule][key]
    except KeyError:
        return float("nan")


def selection(res: dict) -> dict:
    cands = [(k, r, H) for k in ENTRY_KINDS for r in RULE_NAMES for H in H26]
    rows = []
    for k, r, H in cands:
        g = lambda w, key, h=H, sl="0.000": get(res, w, k, h, r, key, sl)
        rows.append({"진입": k, "규칙": r, "H": H, "앞_n": g("2026앞", "n"),
                     "앞_평균차": g("2026앞", "평균차"), "앞_t": g("2026앞", "월묶음_t"), "앞_기대값": g("2026앞", "기대값"),
                     "앞_칼마": g("2026앞", "달력_칼마"),
                     "뒤_n": g("2026뒤", "n"), "뒤_평균차": g("2026뒤", "평균차"), "뒤_t": g("2026뒤", "월묶음_t"),
                     "뒤_기대값": g("2026뒤", "기대값"), "뒤_칼마": g("2026뒤", "달력_칼마"),
                     "뒤_평균차_slip": g("2026뒤", "평균차", H, "0.005"), "앞_평균차_slip": g("2026앞", "평균차", H, "0.005"),
                     "y25_n": g("2025", "n", H25_CHECK), "y25_평균차": g("2025", "평균차", H25_CHECK),
                     "y25_t": g("2025", "월묶음_t", H25_CHECK),
                     "y25_기대값": g("2025", "기대값", H25_CHECK), "y25_칼마": g("2025", "달력_칼마", H25_CHECK),
                     "y25_평균차_H120": g("2025", "평균차", 120)})
    df = pd.DataFrame(rows)
    nonhold = df[df["규칙"] != "hold"]
    picks = {
        "G1": nonhold.sort_values("앞_평균차", ascending=False).iloc[0].to_dict(),
        "G2": df[df["앞_n"] >= MIN_N_G2].sort_values("앞_기대값", ascending=False).iloc[0].to_dict(),
        "G3": df.sort_values("앞_칼마", ascending=False).iloc[0].to_dict(),
    }

    def rate(sub, col):
        sub = sub[np.isfinite(sub[col])]
        return float((sub[col] > 0).mean()) if len(sub) else float("nan"), int(len(sub))

    succ = {}
    for gname, base, front, back, y25 in (("G1", nonhold, "앞_평균차", "뒤_평균차", "y25_평균차"),
                                          ("G2", df, "앞_기대값", "뒤_기대값", "y25_기대값"),
                                          ("G3", df, "앞_칼마", "뒤_칼마", "y25_칼마")):
        pos = base[base[front] > 0]
        succ[gname] = {"앞>0_개수": int(len(pos)), "전체": int(len(base)),
                       "뒤>0|앞>0": rate(pos, back), "뒤>0|무조건": rate(base, back),
                       "2025>0|앞>0": rate(pos, y25), "2025>0|무조건": rate(base, y25)}
    top = nonhold.sort_values("앞_평균차", ascending=False).head(10).to_dict("records")
    return {"picks": picks, "success": succ, "top10_G1": top, "table": df.to_dict("records")}


# ============================================================================ 점검
def check(data: dict, syms: list[str]):
    """1) 엔진 run()과 분해 조립 오차 0. 2) '먼저 닿는 쪽' 규칙을 하루씩 도는 독립 구현과 대조. 3) 경로 끝 = 수익."""
    sample = syms[::12]
    d0, d1 = "2025-01-02", "2025-12-31"
    worst = 0.0
    for kind in ("random", "dip7", "deep40"):
        for rn in ("hold", "fixed15", "trail20", "chand3", "sma50", "time10", "oneil_split", "trail20_half", "trail15_quarter"):
            legs = [T.Leg(fr, BASE[l].kind, BASE[l].p) for l, fr in RULES[rn]]
            ref = T.run(kind, legs, d0, d1, H=60, syms=sample, data=data)
            ref = ref[ref.entry <= d1]
            j = RULE_NAMES.index(rn)
            for row in ref.itertuples():
                b = data[row.sym]
                e = b.idx(row.entry)
                ret, din, bh, V = eval_trade(b, e, 60, 0.0)
                worst = max(worst, abs(ret[j] - row.ret), abs(bh - row.bh), abs(din[j] - row.days_in), abs(V[j, -1] - 1 - ret[j]))
    print("엔진 run() 대비 분해 조립 최대 오차(0이어야 함):", worst)

    def loop_bracket(b, e, H):
        entry = b.o[e]
        Tg, L = entry * 1.2, entry * 0.92
        for k in range(H):
            o, h, l = b.o[e + k], b.h[e + k], b.l[e + k]
            if o >= Tg:
                return o / entry - 1
            if o <= L:
                return o / entry - 1
            if l <= L:
                return L / entry - 1
            if h >= Tg:
                return Tg / entry - 1
        return b.c[e + H - 1] / entry - 1

    def loop_trail_time(b, e, H, p=0.25, t0=20):
        entry = b.o[e]
        peak = entry
        for k in range(H):
            if k == t0 + 1 and t0 < H - 1 and b.c[e + t0] <= entry:
                return b.o[e + k] / entry - 1
            L = peak * (1 - p)
            if b.o[e + k] <= L:
                return b.o[e + k] / entry - 1
            if b.l[e + k] <= L:
                return L / entry - 1
            peak = max(peak, b.c[e + k])
        return b.c[e + H - 1] / entry - 1

    w2 = 0.0
    n = 0
    jb, jt = RULE_NAMES.index("oneil_bracket"), RULE_NAMES.index("trail25_time20")
    for s in sample:
        b = data[s]
        for kind in ("random", "dip7"):
            for e in T.entries(b, kind, d0, d1, GAP, EVERY):
                for H in (20, 60):
                    if e + H - 1 >= len(b.c):
                        continue
                    ret, _, _, _ = eval_trade(b, e, H, 0.0)
                    w2 = max(w2, abs(ret[jb] - loop_bracket(b, e, H)), abs(ret[jt] - loop_trail_time(b, e, H)))
                    n += 1
    print(f"'먼저 닿는 쪽' 규칙 독립 구현 대조 최대 오차(0이어야 함): {w2:.2e} (거래 {n})")
    return worst, w2


# ============================================================================ md
def pct(x, d=1, sign=True):
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x * 100:+.{d}f}%" if sign else f"{x * 100:.{d}f}%"


def tval(x):
    return "—" if x is None or not np.isfinite(x) else f"{x:+.1f}"


def write_md(J: dict):
    res, sel, meta = J["res"], J["selection"], J["meta"]
    L = []
    g = lambda w, k, H, r, key, sl="0.000": get(res, w, k, H, r, key, sl)
    p1 = sel["picks"]["G1"]
    L.append("# 4단계 STKENTRY: 일반 주식 '신호로 사서 손절로 나오는 거래' 단위 손절 시험 (2026-10-11)\n")
    L.append("> 사용자: \"7종목만 하는 게 아니라 일반 주식도 한다. 거기 쓸 만한 게 없어?\" — 진입 방식 5종 × 청산 규칙 20개 × 보유 기간 H.")
    L.append(f"> 대상: 데스크 급락 탭 종목표(시총 10억 달러 이상, 현재 생존) {meta['n_all']}종목 중 자료 오류 의심 {meta['n_bad']}종목을 뺀 "
             f"{meta['n_used']}종목. 코드 `code/p4_STKENTRY.py`, 수치 `results/p4_STKENTRY.json`.\n")
    L.append("## 결론 3줄\n")
    L.extend(J.get("conclusion", []))
    L.append("")
    # ---------------------------------------------------------------- 격자
    L.append("## 시험 범위(사전 고정, 뒤·2025를 본 뒤 바꾸지 않음)\n")
    L.append("| 항목 | 값 |\n|---|---|")
    L.append("| 진입 방식 5 | " + " · ".join(f"{k}({ENTRY_KO[k]})" for k in ENTRY_KINDS) + f" — 같은 종목 같은 신호는 {GAP}거래일 이상 띄움 |")
    L.append("| 청산 규칙 20 | " + " · ".join(RULE_KO[r] for r in RULE_NAMES) + " |")
    L.append("| 보유 H | 2026: 20·60거래일, 2025·맥락: 60·120거래일(2025 확인은 H60) |")
    L.append("| 창(진입일 기준) | 2026앞 1/2~5/29(청산도 5/29 안, 고르기) · 2026뒤 6/1~10/9(확인) · 2025(보조) · 맥락 2018~2024, 2018Q4(10/1~12/24), 2020급락(2/19~3/23) |")
    L.append(f"| 고르기 후보 | 2026: 5 × 20 × 2 = {N_COMBOS_2026}개(들고만 있기 10개 포함, 손절 규칙 190개) |")
    L.append(f"| 전체 계산 칸 | 창 12 × 진입 5 × 규칙 20 × H 2 = 2,400칸, 미끄러짐 0.5% 민감도 3창 600칸 추가 |")
    L.append("| 비용 | 수수료 0(사용자 매매 조건), 미끄러짐 0·0.5%(진입·손절 청산에 적용, 익절 지정가·기간끝 종가에는 없음) |\n")
    # ---------------------------------------------------------------- 시점
    L.append("## 변수 시점\n")
    L.append("| 변수 | 언제 아나 | 매매 근거로 쓰나 |\n|---|---|---|")
    L.append("| 진입 신호(−7%·신고가·200일선·20일 고점·252일 고점 대비) | 신호일 종가 → 다음 날 시가 진입 | 예 |")
    L.append("| 고정·추적 손절선 | 진입가·전일까지 최고 종가 → 그날 장중(시가가 선 아래면 시가) | 예 |")
    L.append("| 샹들리에 | 전일까지 최고 고가 − 3 × 전일 ATR22 → 그날 장중 | 예 |")
    L.append("| 이동평균·시간 청산 | 그날 종가 → 다음 날 시가 | 예 |")
    L.append("| +20% 익절 | 진입가로 미리 정한 지정가 → 장중 | 예 |")
    L.append("| 종목표(시총 10억 달러 이상) | **2026-10 현재 기준**(사후) | 아니오 — 생존·성공 편향 원인(한계 참조) |")
    L.append("| 2026앞 고르기 | 청산이 5/29 안에 끝난 거래만 | 미래 정보 차단 |\n")
    # ---------------------------------------------------------------- 사용자 표
    L.append("## 핵심 표: 매매별 손절 효과 (들고만 있기 대비 기대값 차 · −30% 손실 확률)\n")
    L.extend(J.get("user_table_md", []))
    L.append("")
    # ---------------------------------------------------------------- 고르기
    L.append("## 고르기 (2026앞으로 고르고 뒤·2025로 확인)\n")
    L.append("| 방식 | 고른 칸 | 앞 | 뒤 | 2025(H60) | 뒤 미끄러짐 0.5% |\n|---|---|---|---|---|---|")
    for gname, desc, key in (("G1", "앞 짝 평균차 최대", "평균차"), ("G2", "앞 기대값 최대(30건+)", "기대값"), ("G3", "앞 달력 칼마 최대", "칼마")):
        p = sel["picks"][gname]
        if key == "평균차":
            f = lambda pre: f"차 {pct(p[pre + '_평균차'], 2)}p (t {tval(p[pre + '_t'])})"
            row = f"| {gname} {desc} | {p['진입']} · {RULE_KO[p['규칙']]} · H{int(p['H'])} | {f('앞')} | {f('뒤')} | {f('y25')} | 차 {pct(p['뒤_평균차_slip'], 2)}p |"
        elif key == "기대값":
            row = (f"| {gname} {desc} | {p['진입']} · {RULE_KO[p['규칙']]} · H{int(p['H'])} | 기대값 {pct(p['앞_기대값'])} (n {int(p['앞_n'])}) | "
                   f"기대값 {pct(p['뒤_기대값'])} (n {int(p['뒤_n'])}) | 기대값 {pct(p['y25_기대값'])} | 차 {pct(p['뒤_평균차_slip'])} |")
        else:
            row = (f"| {gname} {desc} | {p['진입']} · {RULE_KO[p['규칙']]} · H{int(p['H'])} | 칼마 {p['앞_칼마']:.1f} | "
                   f"칼마 {p['뒤_칼마']:.1f} | 칼마 {p['y25_칼마']:.1f} | 차 {pct(p['뒤_평균차_slip'])} |")
        L.append(row)
    L.append("")
    L.append("| 지표 | '앞 > 0' 개수 / 전체 | 뒤 > 0: 앞>0 집합 / 무조건 | 2025 > 0: 앞>0 집합 / 무조건 |\n|---|---|---|---|")
    for gname, nm in (("G1", "짝 평균차(손절 190개)"), ("G2", "기대값(200개)"), ("G3", "달력 칼마(200개)")):
        s = sel["success"][gname]
        L.append(f"| {nm} | {s['앞>0_개수']} / {s['전체']} | {pct(s['뒤>0|앞>0'][0], 0, False)} / {pct(s['뒤>0|무조건'][0], 0, False)} | "
                 f"{pct(s['2025>0|앞>0'][0], 0, False)} / {pct(s['2025>0|무조건'][0], 0, False)} |")
    L.append("")
    L.extend(J.get("extra_md", []))
    L.append("앞 짝 평균차 상위 10(손절 규칙):\n")
    L.append("| 순위 | 진입 · 규칙 · H | 앞 차(t) | 뒤 차(t) | 2025 H60 차(t) |\n|---|---|---|---|---|")
    for i, p in enumerate(sel["top10_G1"], 1):
        L.append(f"| {i} | {p['진입']} · {RULE_KO[p['규칙']]} · H{int(p['H'])} | {pct(p['앞_평균차'])} ({tval(p['앞_t'])}) | "
                 f"{pct(p['뒤_평균차'])} ({tval(p['뒤_t'])}) | {pct(p['y25_평균차'])} ({tval(p['y25_t'])}) |")
    L.append("")
    # ---------------------------------------------------------------- 진입별 상세
    L.append("## 진입 방식별 상세 (행 = 청산 규칙)\n")
    L.append("- 차 = 같은 거래에서 규칙 수익 − 들고만 있기 수익의 평균(괄호: 진입 월 묶음 t). P(−30%) = 거래 수익 −30% 이하 비율.")
    L.append("- 하위5% = 거래 수익 5백분위. 보유일당 = 수익 합 ÷ 실제 들고 있던 날 합(손절로 생긴 현금을 다른 데 쓴다고 칠 때의 속도).")
    L.append("- 꼬리맞춤 = 같은 하위5%가 나오도록 처음부터 k만큼만 샀을 때(나머지 현금) 대비 기대값 초과. +면 손절이 '덜 사기'보다 낫다.\n")
    for kind in ENTRY_KINDS:
        for H, ws in ((60, ("2026앞", "2026뒤", "2025")), (20, ("2026앞", "2026뒤"))):
            ns = " · ".join(f"{w} {g(w, kind, H, 'hold', 'n'):.0f}건" for w in ws)
            L.append(f"### {kind} — {ENTRY_KO[kind]} · H{H} ({ns})\n")
            hdr = "| 규칙 | " + " | ".join(f"{w} 차(t)" for w in ws) + " | P(−30%) " + "/".join(ws) + " | 하위5% " + "/".join(ws) + \
                  " | 보유일당 " + "/".join(ws) + " | 꼬리맞춤 " + "/".join(ws) + " |"
            L.append(hdr)
            L.append("|" + "---|" * (1 + len(ws) + 4))
            for rn in RULE_NAMES:
                cells = [RULE_KO[rn]]
                for w in ws:
                    if rn == "hold":
                        cells.append(f"기대값 {pct(g(w, kind, H, rn, '기대값'))}")
                    else:
                        cells.append(f"{pct(g(w, kind, H, rn, '평균차'))} ({tval(g(w, kind, H, rn, '월묶음_t'))})")
                cells.append("/".join(pct(g(w, kind, H, rn, 'P(-30%이하)'), 1, False) for w in ws))
                cells.append("/".join(pct(g(w, kind, H, rn, '하위5%'), 0) for w in ws))
                cells.append("/".join(pct(g(w, kind, H, rn, '보유일당'), 2) for w in ws))
                cells.append("/".join("—" if rn == "hold" else pct(g(w, kind, H, rn, '꼬리맞춤초과'), 1) for w in ws))
                L.append("| " + " | ".join(cells) + " |")
            L.append("")
    # ---------------------------------------------------------------- 승률·손익비
    L.append("## 승률·손익비·꼬리 (H60, 주요 규칙)\n")
    main_rules = ("hold", "fixed8", "fixed15", "trail20", "trail25", "chand3", "sma50", "oneil_split", "trail20_half", "trail25_time20")
    L.append("| 진입 | 규칙 | 승률 앞/뒤/2025 | 손익비 앞/뒤/2025 | 기대값 앞/뒤/2025 | 평균 보유일 앞/뒤/2025 | 청산 비율 2025 |\n|---|---|---|---|---|---|---|")
    for kind in ENTRY_KINDS:
        for rn in main_rules:
            ws = ("2026앞", "2026뒤", "2025")
            L.append(f"| {kind} | {RULE_KO[rn]} | " + "/".join(pct(g(w, kind, 60, rn, '승률'), 0, False) for w in ws) + " | "
                     + "/".join("—" if not np.isfinite(g(w, kind, 60, rn, '손익비')) else f"{g(w, kind, 60, rn, '손익비'):.2f}" for w in ws) + " | "
                     + "/".join(pct(g(w, kind, 60, rn, '기대값')) for w in ws) + " | "
                     + "/".join(f"{g(w, kind, 60, rn, '평균보유일'):.0f}" for w in ws) + " | "
                     + pct(g("2025", kind, 60, rn, '청산비율'), 0, False) + " |")
    L.append("")
    # ---------------------------------------------------------------- 맥락
    L.append("## 맥락: 2018~2024와 폭락 구간 (H60, 차 / P(−30%) 들고만 있기→규칙)\n")
    ctx_rules = ("fixed8", "fixed15", "trail15", "trail20", "trail25", "chand3", "sma50", "oneil_split", "trail20_half")
    for kind in ENTRY_KINDS:
        L.append(f"### {kind} — {ENTRY_KO[kind]}\n")
        L.append("| 규칙 | " + " | ".join(CTX) + " |")
        L.append("|" + "---|" * (1 + len(CTX)))
        L.append("| 들고만 있기 기대값 (n) | " + " | ".join(
            f"{pct(g(w, kind, 60, 'hold', '기대값'))} ({g(w, kind, 60, 'hold', 'n'):.0f})" for w in CTX) + " |")
        L.append("| 들고만 있기 P(−30%) | " + " | ".join(pct(g(w, kind, 60, 'hold', 'P(-30%이하)'), 1, False) for w in CTX) + " |")
        for rn in ctx_rules:
            L.append(f"| {RULE_KO[rn]} | " + " | ".join(
                f"{pct(g(w, kind, 60, rn, '평균차'))} / {pct(g(w, kind, 60, rn, 'P(-30%이하)'), 1, False)}" for w in CTX) + " |")
        L.append("")
    # 달력 포트폴리오 폭락 해
    L.append("### 동일가중 달력 포트폴리오 (진입 거래를 모두 같은 돈으로, 청산 뒤 현금) — 수익 / 최대 낙폭, H60\n")
    cal_rules = ("hold", "fixed15", "trail20", "trail25", "chand3", "sma50", "trail20_half")
    cal_ws = ("2026앞", "2026뒤", "2025", "2018Q4", "2020급락", "2022")
    L.append("| 진입 | 규칙 | " + " | ".join(cal_ws) + " |")
    L.append("|" + "---|" * (2 + len(cal_ws)))
    for kind in ENTRY_KINDS:
        for rn in cal_rules:
            L.append(f"| {kind} | {RULE_KO[rn]} | " + " | ".join(
                f"{pct(g(w, kind, 60, rn, '달력_수익'))} / {pct(g(w, kind, 60, rn, '달력_낙폭'), 1)}" for w in cal_ws) + " |")
    L.append("")
    # ---------------------------------------------------------------- 미끄러짐
    L.append("## 미끄러짐 0.5% 민감도 (H60, 차: 0% → 0.5%)\n")
    L.append("| 진입 | 규칙 | 2026앞 | 2026뒤 | 2025 |\n|---|---|---|---|---|")
    for kind in ENTRY_KINDS:
        for rn in ("fixed8", "trail10", "trail20", "trail25", "chand3", "sma20", "sma50"):
            L.append(f"| {kind} | {RULE_KO[rn]} | " + " | ".join(
                f"{pct(g(w, kind, 60, rn, '평균차'))} → {pct(g(w, kind, 60, rn, '평균차', '0.005'))}" for w in ("2026앞", "2026뒤", "2025")) + " |")
    L.append("")
    L.append("## 선택편향·한계\n")
    L.extend(J.get("limits", []))
    L.append("")
    L.append(f"점검: 엔진 run() 대비 분해 조립 최대 오차 {meta['check_engine']:.1e}, '먼저 닿는 쪽' 규칙 독립 구현 대조 최대 오차 {meta['check_loop']:.1e}. 계산 {meta['secs']:.0f}초.")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("md →", OUT_MD)


# ============================================================================ 결론·사용자 표
def _pp(x, d=1):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:+.{d}f}"


# 사용자 표 행: (진입, 규칙, 2026 H, 왜 이 행인가). 2025·폭락 칸은 H60.
USER_ROWS = [
    ("random", "fixed25", 60, "꼬리를 가장 싸게 자르는 손절(앞 구간 기준, 4/5 진입에서 1위)"),
    ("random", "trail20", 60, "흔히 쓰는 추적 −20%"),
    ("dip7", "fixed25", 60, "급락 매수의 가장 싼 보험"),
    ("dip7", "fixed8", 60, "O'Neil식 좁은 손절"),
    ("breakout", "time10", 20, "G1 대표 후보(앞 짝 평균차 1위)"),
    ("pullback", "fixed15", 60, "'같은 꼬리 덜 사기'를 세 창 모두 이긴 유일한 칸(관찰)"),
    ("deep40", "fixed25", 60, "−40% 매수의 넓은 손절"),
    ("deep40", "trail20_half", 60, "−40% 매수의 일부 손절"),
]


def _aux_select(res, key: str, H: int, rel_hold: bool = False):
    """보조 고르기(사전 고정 G1~G3 밖): 앞 > 0 집합의 뒤·2025(H60) > 0 비율 vs 무조건."""
    rows = []
    for k in ENTRY_KINDS:
        for r in RULE_NAMES:
            if r == "hold":
                continue
            v = []
            for w, h in (("2026앞", H), ("2026뒤", H), ("2025", H25_CHECK)):
                x = get(res, w, k, h, r, key)
                if rel_hold:
                    x = x - get(res, w, k, h, "hold", key)
                v.append(x)
            rows.append((k, r, *v))
    A = [x for x in rows if x[2] > 1e-12]
    m = lambda s, i: float(np.mean([x[i] > 1e-12 for x in s])) if s else float("nan")
    best = sorted(A, key=lambda x: -x[2])[:3]
    return {"앞>0": len(A), "전체": len(rows), "뒤|앞": m(A, 3), "뒤|무조건": m(rows, 3), "25|앞": m(A, 4), "25|무조건": m(rows, 4),
            "상위": [(k, r, a, b, c) for k, r, a, b, c in best]}


def build_text(J: dict) -> dict:
    """결론·사용자 표·보조 고르기·한계 문장(모두 json 수치에서 만든다)."""
    res, sel = J["res"], J["selection"]
    g = lambda w, k, H, r, key, sl="0.000": get(res, w, k, H, r, key, sl)
    nonhold = [r for r in RULE_NAMES if r != "hold"]
    pos = lambda w, H: sum(g(w, k, H, r, "평균차") > 1e-12 for k in ENTRY_KINDS for r in nonhold)
    n_front = pos("2026앞", 20) + pos("2026앞", 60)
    n_back = pos("2026뒤", 20) + pos("2026뒤", 60)
    n_back_bo = sum(g("2026뒤", "breakout", H, r, "평균차") > 1e-12 for H in H26 for r in nonhold)
    n25 = {H: pos("2025", H) for H in HOTHER}
    ctx = {w: pos(w, 60) for w in CTX}
    p1 = sel["picks"]["G1"]
    k1, r1, h1 = p1["진입"], p1["규칙"], int(p1["H"])

    def cell(w, k, H, r):
        d = g(w, k, H, r, "평균차")
        a, b = g(w, k, H, "hold", "P(-30%이하)"), g(w, k, H, r, "P(-30%이하)")
        return f"{_pp(d)}%p · {a * 100:.1f}→{b * 100:.1f}%"

    ut = ["각 칸 = 거래 1건 기대값 차(규칙 − 같은 거래 들고만 있기, %p) · 거래가 −30% 이하로 끝날 확률(들고만 있기→규칙). "
          "2025·폭락 칸은 60거래일 보유. 폭락 = 2020-02-19~03-23 진입 / 2022년 진입.\n",
          "| 상황(진입 · 2026 보유일) | 규칙 | 2026 앞(고르기) | 2026 뒤(확인) | 2025 | 폭락 2020.3 / 2022 |",
          "|---|---|---|---|---|---|"]
    for k, r, H, why in USER_ROWS:
        hold = "/".join(_pp(g(w, k, hh, "hold", "기대값")) + "%" for w, hh in (("2026앞", H), ("2026뒤", H), ("2025", 60)))
        ut.append(f"| {ENTRY_KO[k]} · {H}일 (들고만 있기 {hold}) | {RULE_KO[r]} — {why} | {cell('2026앞', k, H, r)} | "
                  f"{cell('2026뒤', k, H, r)} | {cell('2025', k, 60, r)} | {cell('2020급락', k, 60, r)} / {cell('2022', k, 60, r)} |")

    # 보조 고르기
    aux = {"꼬리맞춤 H60": _aux_select(res, "꼬리맞춤초과", 60), "꼬리맞춤 H20": _aux_select(res, "꼬리맞춤초과", 20),
           "보유일당 H60": _aux_select(res, "보유일당", 60, True), "보유일당 H20": _aux_select(res, "보유일당", 20, True)}
    ex = ["### 보조 고르기(사전 고정 G1~G3 밖, 참고): 같은 꼬리 덜 사기·보유일당\n",
          "| 지표 · H | 앞 > 0 / 전체 | 뒤 > 0: 앞>0 집합 / 무조건 | 2025(H60) > 0: 앞>0 집합 / 무조건 | 앞 상위 3 (앞/뒤/2025, %p) |",
          "|---|---|---|---|---|"]
    for name, a in aux.items():
        top = "; ".join(f"{k}·{RULE_KO[r]} {_pp(x, 2)}/{_pp(y, 2)}/{_pp(z, 2)}" for k, r, x, y, z in a["상위"])
        ex.append(f"| {name} | {a['앞>0']} / {a['전체']} | {a['뒤|앞'] * 100:.0f}% / {a['뒤|무조건'] * 100:.0f}% | "
                  f"{a['25|앞'] * 100:.0f}% / {a['25|무조건'] * 100:.0f}% | {top} |")
    pf = {w: g(w, "pullback", 60, "fixed15", "꼬리맞춤초과") for w in ["2026앞", "2026뒤", "2025"] + CTX}
    ex.append("")
    ex.append("- 꼬리맞춤 = 규칙 기대값 − k × 들고만 있기 기대값, k = 규칙 하위5% ÷ 들고만 있기 하위5%(0~1). '손절 대신 처음부터 k만큼만 샀다면'과의 비교.")
    ex.append("- 눌림목 × 고정 −15%(H60) 꼬리맞춤 해마다: " + " · ".join(f"{w} {_pp(v)}%p" for w, v in pf.items())
              + ". 평상시엔 '반만 사기'보다 나았고, 2018·2022·폭락 구간에서는 바닥에서 팔아 졌다.")
    ex.append("- 보유일당이 들고만 있기보다 높은 칸은 대부분 이동평균 청산이 진입 직후(평균 3~8일) 나가는 경우다. 앞 우위가 뒤를 맞히지 못했다.")
    ex.append("- 진입 직후 나가는 칸: 급락·−40% 매수는 산 날 이미 20·50·100일선 아래라 이동평균 청산이 다음 날 시가에 판다"
              f"(deep40 · 50일선 평균 보유 {g('2026앞', 'deep40', 60, 'sma50', '평균보유일'):.0f}일, dip7 · 20일선 "
              f"{g('2026앞', 'dip7', 60, 'sma20', '평균보유일'):.0f}일). 이 칸의 꼬리 0%는 '사지 않은 것'과 같다.\n")

    fx = lambda k, w, r="fixed25", H=60: g(w, k, H, r, "평균차")
    concl = [
        f"1. **일반 주식 거래에서 손절은 기대값을 올리지 못했다.** 2026 앞 구간 손절 칸 190개 중 들고만 있기보다 나은 것은 {n_front}개였고 "
        f"(최대 {_pp(p1['앞_평균차'], 2)}%p, t {p1['앞_t']:+.1f}), 2025는 H60 {n25[60]}/95·H120 {n25[120]}/95, 2023·2024는 {ctx['2023']}·{ctx['2024']}/95, "
        f"2022 약세장도 {ctx['2022']}/95였다. 뒤 구간은 {n_back}/190이 이겼는데 {n_back_bo}개가 신고가 돌파 매수였다"
        f"(뒤 구간 돌파 매수는 들고만 있기 기대값이 {_pp(g('2026뒤', 'breakout', 60, 'hold', '기대값'))}%·60일). "
        f"대표 G1({k1} · {RULE_KO[r1]} · {h1}일)은 뒤 {_pp(p1['뒤_평균차'], 2)}%p(t {p1['뒤_t']:+.1f})였지만 미끄러짐 0.5%면 {_pp(p1['뒤_평균차_slip'], 2)}%p, "
        f"2025(60일) {_pp(p1['y25_평균차'])}%p(t {p1['y25_t']:+.1f})로 기각.",
        f"2. **손절의 값은 꼬리(−30% 이하 손실) 절단이고, 넓을수록 싸다.** 고정 −25%가 진입 5종 중 4종에서 가장 싼 보험이었다. "
        f"아무 날 산 60일 거래의 −30% 손실 확률을 {g('2026앞', 'random', 60, 'hold', 'P(-30%이하)') * 100:.1f}%→{g('2026앞', 'random', 60, 'fixed25', 'P(-30%이하)') * 100:.1f}%로 줄이며 "
        f"기대값은 {_pp(fx('random', '2026앞'))}%p(뒤 {_pp(fx('random', '2026뒤'))}, 2025 {_pp(fx('random', '2025'))})만 잃었다. "
        f"좁은 손절(−8%·추적 −10%·20일선)은 거래당 3~15%p를 잃었다. 급락(−7%) 매수·−40% 매수는 같은 −25%도 "
        f"{_pp(fx('dip7', '2026앞'))}·{_pp(fx('deep40', '2026앞'))}%p로 약 3배 비쌌고, 2020년 3월 같은 V자 폭락에 산 거래는 손절이 바닥에서 팔아 "
        f"{_pp(fx('dip7', '2020급락'))}·{_pp(fx('deep40', '2020급락'))}%p를 잃었다(좁은 손절은 −34%p까지).",
        f"3. **'같은 꼬리만큼 처음부터 덜 사기'와 비교해도 손절은 대개 졌다**(2026 앞 60일 칸 {aux['꼬리맞춤 H60']['앞>0']}/95만 이김). "
        f"예외 관찰: 눌림목(200일선 위 20일 고점 −10%) 매수 × 고정 −15%는 앞 {_pp(pf['2026앞'])}·뒤 {_pp(pf['2026뒤'])}·2025 {_pp(pf['2025'])}%p로 '반만 사기'보다 나았지만 "
        f"2018·2022·폭락 구간에서 졌다. 실무 결론: 일반 주식은 손절로 수익을 노리지 않는다. 손절을 걸려면 −25% 고정(보험료 거래당 약 1~3%p)으로 "
        f"'한 종목 −30~−50%'만 막고, 급락·−40% 매수에는 손절보다 비중으로 위험을 정한다.",
    ]
    limits = [
        f"- 탐색 범위: 2026 고르기 후보 {N_COMBOS_2026}개(손절 190개). G1 1위 앞 차 {_pp(p1['앞_평균차'], 2)}%p(t {p1['앞_t']:+.1f})는 190개 잡음 가운데 최댓값 수준이다. "
        "'앞 > 0' 집합의 뒤·2025 성공률을 무조건 비율과 나란히 보였다(고르기 표).",
        "- 2026 앞 60일 칸은 청산이 5/29 안이어야 해서 진입이 1/2~3/3(3개월)뿐이고, 뒤 60일 칸은 6/1~7/14(2개월)라 월 묶음 t를 낼 수 없다(—). 20일 칸이 월 수가 더 많다.",
        "- 2025 확인은 사전 규약대로 60·120일이다. 20일 후보(G1)는 2025에서 60일로 확인했다. 60일은 손절 뒤 놓치는 반등 기간이 길어 20일 규칙에 불리할 수 있다.",
        f"- 생존·성공 편향: 종목표는 2026-10 현재 시총 10억 달러 이상 생존 종목이다. 상장폐지·파산 종목(손절이 살렸을 거래)이 없어 손절에 불리하고, "
        "과거에 작았다가 커진 종목만 있어 들고만 있기에 유리하다. 실제 손절 효과는 여기 숫자보다 조금 낫다.",
        f"- 자료 오류 의심 {J['meta']['n_bad']}종목(하루 +150% 이상 또는 −70% 이하 종가 변화)을 뺐다. 갭 하락은 손절선이 아니라 시가에 팔린다(고정 −8%도 −30% 손실 0.0~0.2%).",
        "- O'Neil 나눔은 엔진이 다리별이라 '+20% 익절 50% + −8% 손절 50%'로 정의했다(익절 다리는 손절이 없고 손절 다리는 익절이 없다). "
        "같은 다리의 묶음(먼저 닿는 쪽)도 따로 쟀다(같은 날 둘 다 닿으면 손절 먼저, 시가가 익절선 위면 익절).",
        "- 진입 신호마다 같은 종목은 20거래일 이상 띄웠다. 같은 달 거래는 서로 엮여 있어 t는 진입 월 묶음으로 냈다(독립 표본 수 = 월 수).",
        "- 달력 포트폴리오는 그날 열린 거래를 같은 돈으로 매일 재조정한 근사다(칼마용). 청산 뒤 현금 이자 0%.",
        f"- 수수료 0, 미끄러짐 0.5%는 진입·손절 청산에만 적용했다. 다른 해는 장세 전환 위험으로만 본다: 2019는 손절 칸 {ctx['2019']}/95가 이겼고, 2020년 2월 말~3월 신고가 매수(폭락 직전 진입)는 좁은 손절·이동평균 청산이 +1~+10%p였다(넓은 추적 −20·−25%는 −3·−7%p).",
    ]
    return {"conclusion": concl, "user_table_md": ut, "extra_md": ex, "limits": limits,
            "aux_select": {k: {kk: vv for kk, vv in v.items()} for k, v in aux.items()},
            "counts": {"앞_양수": n_front, "뒤_양수": n_back, "2025_양수": n25, "맥락_양수": ctx}}


# ============================================================================ 본문
def load_universe():
    data = T.load_stocks()
    allsyms = sorted(data)
    bad = [s for s in allsyms if T.bad_bars(data[s])]
    syms = [s for s in allsyms if s not in set(bad)]
    return data, allsyms, bad, syms


def main():
    if "--md" in sys.argv:
        J = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        J.update(build_text(J))
        OUT_JSON.write_text(json.dumps(J, ensure_ascii=False, default=float), encoding="utf-8")
        write_md(J)
        return
    data, allsyms, bad, syms = load_universe()
    print("종목", len(allsyms), "· 자료 오류 의심 제외", len(bad), "· 사용", len(syms))
    ce, cl = check(data, syms)
    if "--check" in sys.argv:
        return
    t0 = time.time()
    res = compute(data, syms)
    J = {"meta": {"n_all": len(allsyms), "n_bad": len(bad), "bad": bad, "n_used": len(syms), "check_engine": ce, "check_loop": cl,
                  "secs": time.time() - t0, "rules": RULE_NAMES, "rule_ko": RULE_KO, "entries": list(ENTRY_KINDS),
                  "windows": {k: [v[0], v[1], v[2], list(v[3]), list(v[4])] for k, v in WINDOWS.items()}},
         "res": res}
    J["selection"] = selection(res)
    J.update(build_text(J))
    OUT_JSON.write_text(json.dumps(J, ensure_ascii=False, default=float), encoding="utf-8")
    print("json →", OUT_JSON)
    write_md(J)


if __name__ == "__main__":
    main()
