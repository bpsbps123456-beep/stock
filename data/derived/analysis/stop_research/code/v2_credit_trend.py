"""2차 반박 검증 credit_trend(2026-10-10): 'HYG/IEF(Adj Close) < SMA50 이면 노출 0.5, 최소 5거래일 유지 뒤 신호가 꺼지면 1.0'.

harness.py를 import하지 않고 base_*.json·ohlc.parquet만으로 처음부터 다시 잰다.
시점: 비율·SMA는 t일 종가까지 → t 다음 거래일 시가부터 적용(신호 날짜 < 거래일인 최신 값). 지연판은 t+2 시가.
수익: ret[k] = open[k+1]/open[k]-1, 변형 수익 = e[k]*ret[k](현금 이자 0, 수수료 0).
구간 수익 = 구간에 속한 ret[k]의 곱, 낙폭 = 그 곱 경로(시작 1.0 포함)의 최대 낙폭.
"""
from __future__ import annotations

import json
from bisect import bisect_left
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"
RESULTS = HERE.parent / "results"

P26 = {"전체": ("2026-01-02", "2026-10-09"), "앞": ("2026-01-02", "2026-05-29"), "뒤": ("2026-06-01", "2026-10-09"),
       "AI쇼크": ("2026-06-22", "2026-07-29"), "7~9월": ("2026-07-01", "2026-09-30")}
PHOLD = {"2025": ("2025-01-01", "2025-12-31"), "2024": ("2024-01-01", "2024-12-31"), "관세": ("2025-02-18", "2025-04-04"),
         "24여름": ("2024-06-18", "2024-08-05"), "25가을": ("2025-10-13", "2025-11-21")}

CLAIM = {"2026_전체": (2.264, -0.180), "2026_앞": (2.033, -0.142), "2026_뒤": (0.076, -0.180), "2026_AI쇼크": (-0.164, None),
         "2026_7~9월": (0.048, None), "hold_2025": (0.479, -0.319), "hold_관세": (-0.319, None),
         "hold_24여름": (-0.190, None), "hold_25가을": (-0.234, None)}
CLAIM_E0 = {"2026_전체": (2.218, -0.158), "hold_2025": (0.397, -0.235)}

SMAS = (30, 40, 50, 60, 80, 100)
LOWS = (0.0, 0.25, 0.5, 0.75)
HOLDS = (0, 5, 10)
PAIRS = (("HYG", "IEF"), ("JNK", "IEF"), ("LQD", "IEF"), ("HYG", "TLT"), ("HYG", "SHY"))


# ----------------------------------------------------------------------------- 자료
def load_base(name: str):
    b = json.loads((CACHE / f"{name}.json").read_text(encoding="utf-8"))
    days = list(b["days"])
    op = np.asarray(b["open"], float)
    ret = pd.Series(op[1:] / op[:-1] - 1, index=days[:-1])
    return days, ret


OH = pd.read_parquet(CACHE / "ohlc.parquet")
OH["date"] = pd.to_datetime(OH["date"]).dt.strftime("%Y-%m-%d")


def col(sym: str, field: str) -> pd.Series:
    s = OH[OH.sym == sym].set_index("date")[field].astype(float).sort_index()
    return s[~s.index.duplicated()].dropna()


def flag(num: str, den: str, n: int, field: str = "Adj Close", delay: int = 0) -> pd.Series:
    """index = 그 값을 아는 날(종가일). 1.0 = 비율 < SMA(n). SMA가 정의되지 않은 날은 빠진다.
    delay=1이면 값을 '자기 거래일 하나 뒤에 안다'로 옮겨 t+2 시가 적용판을 만든다."""
    a, b = col(num, field), col(den, field)
    r = (a / b).dropna()
    sma = r.rolling(n).mean()
    f = (r < sma).astype(float)[sma.notna()]
    if delay:
        f = pd.Series(f.to_numpy()[:-delay], index=f.index[delay:])
    return f


def on_open(f: pd.Series, days: list) -> np.ndarray:
    """거래일 k 시가에 쓰는 값 = 날짜가 k보다 엄격히 앞선 최신 신호. 자료 전이면 꺼짐."""
    fd, fv = list(f.index), f.to_numpy()
    out = np.zeros(len(days), bool)
    for k, d in enumerate(days):
        j = bisect_left(fd, d) - 1
        out[k] = j >= 0 and fv[j] > 0.5
    return out


def exposure(on: np.ndarray, low: float, m: int) -> np.ndarray:
    """신호가 켜진 시가에 low로 줄이고, 줄인 첫날 k0부터 max(m,1)거래일이 지난 날(k-k0 ≥ max(m,1)) 신호가 꺼져 있으면 1.0 복귀.
    m=0과 m=1은 같다(같은 시가에 줄였다 돌아올 수 없다)."""
    e = np.ones(len(on))
    red, k0 = False, -1
    for k in range(len(on)):
        if red and k - k0 >= max(m, 1) and not on[k]:
            red = False
        if not red and on[k]:
            red, k0 = True, k
        e[k] = low if red else 1.0
    return e


# ----------------------------------------------------------------------------- 지표
def stats(r: pd.Series, lo: str, hi: str):
    x = r[(r.index >= lo) & (r.index <= hi)].to_numpy()
    c = np.concatenate([[1.0], np.cumprod(1 + x)])
    return float(c[-1] - 1), float((c / np.maximum.accumulate(c) - 1).min())


def avg_e(e: pd.Series, lo: str, hi: str) -> float:
    return float(e[(e.index >= lo) & (e.index <= hi)].mean())


def run(e_full: np.ndarray, days: list, ret: pd.Series, periods: dict) -> dict:
    e = pd.Series(e_full, index=days).reindex(ret.index)
    r = e * ret
    out = {k: stats(r, *v) for k, v in periods.items()}
    out["_e"] = e
    return out


def spells(e: pd.Series) -> list:
    out, s = [], None
    idx = list(e.index)
    for i, d in enumerate(idx):
        red = e.iloc[i] < 0.999
        if red and s is None:
            s = i
        if s is not None and (not red or i == len(idx) - 1):
            end = i if not red else i + 1
            out.append({"줄임시가": idx[s], "복귀시가": idx[end] if end < len(idx) else "끝까지", "거래일": end - s})
            s = None
    return out


B26 = load_base("base_2026")
BH = load_base("base_hold")


def evaluate(f: pd.Series, low: float, m: int) -> dict:
    res = {}
    for tag, (days, ret), per in (("2026", B26, P26), ("hold", BH, PHOLD)):
        e = exposure(on_open(f, days), low, m)
        x = run(e, days, ret, per)
        ee = x.pop("_e")
        for k, v in x.items():
            res[f"{tag}_{k}"] = v
        if tag == "2026":
            res["2026_평균노출"] = float(ee.mean())
            res["2026_빠짐"] = int(((ee.shift(1).fillna(1.0) >= 0.999) & (ee < 0.999)).sum())
            res["_spells26"] = spells(ee)
        else:
            res["2025_평균노출"] = avg_e(ee, *PHOLD["2025"])
            res["2024_평균노출"] = avg_e(ee, *PHOLD["2024"])
            res["hold_빠짐"] = int(((ee.shift(1).fillna(1.0) >= 0.999) & (ee < 0.999)).sum())
    return res


def fixed(c: float) -> dict:
    res = {}
    for tag, (days, ret), per in (("2026", B26, P26), ("hold", BH, PHOLD)):
        x = run(np.full(len(days), c), days, ret, per)
        x.pop("_e")
        for k, v in x.items():
            res[f"{tag}_{k}"] = v
    return res


def fixed_ret(c: float, key: str) -> tuple:
    return fixed(c)[key]


def fmt(v):
    return f"{v[0]:+.1%}/{v[1]:.1%}"


def main():
    base = fixed(1.0)
    bw_front = 1 + base["2026_앞"][0]
    print("기본", {k: fmt(v) for k, v in base.items()})

    # 1) 후보 재계산 + 주장 비교
    f50 = flag("HYG", "IEF", 50)
    cand = evaluate(f50, 0.5, 5)
    cand0 = evaluate(f50, 0.0, 5)
    disc = []
    for k, (rv, dd) in CLAIM.items():
        got = cand[k]
        d1 = got[0] - rv
        line = f"{k}: 주장 {rv:+.3f} 재계산 {got[0]:+.4f} 차 {d1:+.4f}"
        if dd is not None:
            line += f" | 낙폭 주장 {dd:.3f} 재계산 {got[1]:.4f} 차 {got[1] - dd:+.4f}"
        disc.append(line)
    for k, (rv, dd) in CLAIM_E0.items():
        got = cand0[k]
        disc.append(f"[노출0] {k}: 주장 {rv:+.3f}/{dd:.3f} 재계산 {got[0]:+.4f}/{got[1]:.4f}")
    print("\n".join(disc))
    print("평균노출", cand["2026_평균노출"], "빠짐", cand["2026_빠짐"], cand["_spells26"])

    # 2) 시점 감사: 신호 날짜 ↔ 적용 시가, 하루 지연판
    r = (col("HYG", "Adj Close") / col("IEF", "Adj Close")).dropna()
    sma = r.rolling(50).mean()
    audit = {}
    for d in ("2026-02-09", "2026-02-10", "2026-02-11", "2026-04-01", "2026-04-02", "2026-06-23", "2026-06-24",
              "2026-06-25", "2026-07-01", "2026-07-02"):
        audit[d] = {"비율": round(float(r[d]), 6), "SMA50": round(float(sma[d]), 6), "켜짐": bool(r[d] < sma[d])}
    # SMA50 창이 정확히 t 포함 50개인지(하루 밀림 없음) 직접 확인
    i = list(r.index).index("2026-06-24")
    audit["SMA50_6/24_직접"] = round(float(r.iloc[i - 49:i + 1].mean()), 6)
    late = evaluate(flag("HYG", "IEF", 50, delay=1), 0.5, 5)
    late0 = evaluate(flag("HYG", "IEF", 50, delay=1), 0.0, 5)
    # 참고: 같은 날 종가로 같은 날 시가에 적용(미래 참조) — 시점이 맞다면 이 판보다 나빠야 정상인 것은 아니지만 크기 비교용
    print("시점 감사", audit)

    # 4) 고정 노출 대비
    a26 = cand["2026_평균노출"]
    fx26 = fixed(a26)
    fx25 = fixed(cand["2025_평균노출"])
    fx24 = fixed(cand["2024_평균노출"])
    keys = ("2026_전체", "2026_앞", "2026_뒤", "2026_AI쇼크", "2026_7~9월", "hold_2025", "hold_2024", "hold_관세",
            "hold_24여름", "hold_25가을")
    tv = {k: {"후보": cand[k], "고정(2026평균)": fx26[k], "수익차": cand[k][0] - fx26[k][0], "낙폭차": cand[k][1] - fx26[k][1]}
          for k in keys}
    tv["hold_2025"]["고정(2025평균)"] = fx25["hold_2025"]
    tv["hold_2024"]["고정(2024평균)"] = fx24["hold_2024"]
    a26_0 = cand0["2026_평균노출"]
    fx26_0 = fixed(a26_0)
    tv0 = {k: {"후보": cand0[k], "고정": fx26_0[k], "수익차": cand0[k][0] - fx26_0[k][0],
               "낙폭차": cand0[k][1] - fx26_0[k][1]} for k in keys}
    tv0["hold_2025"]["고정(2025평균)"] = fixed(cand0["2025_평균노출"])["hold_2025"]
    tv0["hold_2024"]["고정(2024평균)"] = fixed(cand0["2024_평균노출"])["hold_2024"]

    # 3) 이웃 격자 72
    rows = []
    fcache = {n: flag("HYG", "IEF", n) for n in SMAS}
    fixc: dict = {}

    def fx(c, key):
        c = round(c, 6)
        if c not in fixc:
            fixc[c] = fixed(c)
        return fixc[c][key]

    for n in SMAS:
        for low in LOWS:
            for m in HOLDS:
                x = evaluate(fcache[n], low, m)
                front_ok = 1 + x["2026_앞"][0] >= 0.9 * bw_front
                back_ok = x["2026_뒤"][0] > base["2026_뒤"][0] and x["2026_AI쇼크"][0] > base["2026_AI쇼크"][0]
                f25 = fx(x["2025_평균노출"], "hold_2025")
                f24 = fx(x["2024_평균노출"], "hold_2024")
                f26 = fx(x["2026_평균노출"], "2026_전체")
                rows.append({"sma": n, "low": low, "hold": m, "2026_전체": x["2026_전체"], "2026_앞": x["2026_앞"],
                             "2026_뒤": x["2026_뒤"], "AI쇼크": x["2026_AI쇼크"][0], "7~9월": x["2026_7~9월"][0],
                             "2025": x["hold_2025"], "2024": x["hold_2024"], "관세": x["hold_관세"][0],
                             "24여름": x["hold_24여름"][0], "25가을": x["hold_25가을"][0],
                             "노출26": x["2026_평균노출"], "노출25": x["2025_평균노출"], "노출24": x["2024_평균노출"],
                             "빠짐26": x["2026_빠짐"], "고정26_전체": f26, "고정25": f25, "고정24": f24,
                             "앞통과": bool(front_ok), "뒤통과": bool(back_ok), "2026통과": bool(front_ok and back_ok),
                             "2026>고정": bool(x["2026_전체"][0] > f26[0]),
                             "2025>고정": bool(x["hold_2025"][0] > f25[0]), "2024>고정": bool(x["hold_2024"][0] > f24[0])})
    nb = len(rows)
    nontriv = [x for x in rows if x["low"] < 1]  # 모두 해당(노출 1.0은 격자에 없음)
    cnt = {k: sum(x[k] for x in rows) for k in ("앞통과", "뒤통과", "2026통과", "2026>고정", "2025>고정", "2024>고정")}
    cnt["2025·2024 모두>고정"] = sum(x["2025>고정"] and x["2024>고정"] for x in rows)
    cnt["2026통과 & 2025·2024>고정"] = sum(x["2026통과"] and x["2025>고정"] and x["2024>고정"] for x in rows)
    print("이웃", nb, cnt)
    by_sma = {n: {k: sum(x[k] for x in rows if x["sma"] == n) for k in ("2026통과", "2025>고정", "2024>고정")}
              for n in SMAS}
    by_low = {lw: {k: sum(x[k] for x in rows if x["low"] == lw) for k in ("2026통과", "2025>고정", "2024>고정")}
              for lw in LOWS}
    by_hold = {m: {k: sum(x[k] for x in rows if x["hold"] == m) for k in ("2026통과", "2025>고정", "2024>고정")}
               for m in HOLDS}
    print("SMA별", by_sma, "\n노출별", by_low, "\n유지별", by_hold)

    # 다른 상품 쌍(SMA50·0.5·5일), Close 비율
    pairs = {}
    for a, b in PAIRS:
        for fld in ("Adj Close", "Close"):
            x = evaluate(flag(a, b, 50, fld), 0.5, 5)
            f25 = fx(x["2025_평균노출"], "hold_2025")
            f24 = fx(x["2024_평균노출"], "hold_2024")
            f26 = fx(x["2026_평균노출"], "2026_전체")
            pairs[f"{a}/{b} {fld}"] = {"2026_전체": x["2026_전체"], "2026_앞": x["2026_앞"], "2026_뒤": x["2026_뒤"],
                                       "AI쇼크": x["2026_AI쇼크"][0], "2025": x["hold_2025"], "2024": x["hold_2024"],
                                       "노출26": x["2026_평균노출"], "빠짐26": x["2026_빠짐"], "고정26": f26, "고정25": f25,
                                       "고정24": f24, "spells26": x["_spells26"],
                                       "2026통과": bool(1 + x["2026_앞"][0] >= 0.9 * bw_front
                                                       and x["2026_뒤"][0] > base["2026_뒤"][0]
                                                       and x["2026_AI쇼크"][0] > base["2026_AI쇼크"][0])}
    for k, v in pairs.items():
        print(f"{k:<22} 2026 {fmt(v['2026_전체'])} (고정 {fmt(v['고정26'])}) 앞 {fmt(v['2026_앞'])} 뒤 {fmt(v['2026_뒤'])} "
              f"AI {v['AI쇼크']:+.1%} 2025 {fmt(v['2025'])} (고정 {fmt(v['고정25'])}) 2024 {fmt(v['2024'])} "
              f"(고정 {fmt(v['고정24'])}) 노출 {v['노출26']:.2f} 빠짐 {v['빠짐26']} 통과 {v['2026통과']}")

    # 위약 검사: 후보의 2026 노출 경로를 원형 이동(같은 노출 분포, 시점만 무작위)
    days26, ret26 = B26
    e26 = pd.Series(exposure(on_open(f50, days26), 0.5, 5), index=days26).reindex(ret26.index).to_numpy()
    shifts = []
    for s in range(1, len(e26)):
        es = np.roll(e26, s)
        shifts.append(float(np.prod(1 + es * ret26.to_numpy()) - 1))
    actual = float(np.prod(1 + e26 * ret26.to_numpy()) - 1)
    pct = float(np.mean(np.array(shifts) < actual))
    print(f"원형 이동 위약: 실제 {actual:+.3f}, 이동 {len(shifts)}개 중 실제보다 낮은 비율 {pct:.2f}, 중앙값 {np.median(shifts):+.3f}")
    # 2026 이익을 만든 구간 분해: 6/25~7/2 한 번 빼면?
    e_no_jul = e26.copy()
    idx = list(ret26.index)
    for i, d in enumerate(idx):
        if "2026-06-25" <= d < "2026-07-02":
            e_no_jul[i] = 1.0
    no_jul = float(np.prod(1 + e_no_jul * ret26.to_numpy()) - 1)
    e_no_feb = e26.copy()
    for i, d in enumerate(idx):
        if "2026-02-11" <= d < "2026-04-02":
            e_no_feb[i] = 1.0
    no_feb = float(np.prod(1 + e_no_feb * ret26.to_numpy()) - 1)
    print(f"분해: 6/25~7/1 빠짐만 없앤 2026 {no_jul:+.3f}, 2/11~4/1 빠짐만 없앤 2026 {no_feb:+.3f}")

    # 보유 기간 위약(같은 방식) + 지연판 고정 노출 대비 + 2월 진입 신호의 여유
    daysh, reth = BH
    eh = pd.Series(exposure(on_open(f50, daysh), 0.5, 5), index=daysh).reindex(reth.index).to_numpy()
    rh, ih = reth.to_numpy(), np.array(reth.index)
    placebo_hold = {}
    for y in ("2025", "2024"):
        msk = (ih >= PHOLD[y][0]) & (ih <= PHOLD[y][1])
        act = float(np.prod(1 + (eh * rh)[msk]) - 1)
        sh = np.array([np.prod(1 + (np.roll(eh, s) * rh)[msk]) - 1 for s in range(1, len(eh))])
        placebo_hold[y] = {"actual": act, "share_below_actual": float(np.mean(sh < act)), "median": float(np.median(sh))}
    print("보유 위약", placebo_hold)
    late_fx = {"2026": fixed(late["2026_평균노출"]), "2025": fixed(late["2025_평균노출"])["hold_2025"],
               "2024": fixed(late["2024_평균노출"])["hold_2024"]}
    sig_feb = "".join("1" if v > 0.5 else "0" for v in f50["2026-02-02":"2026-04-10"].to_numpy())
    margin_0210 = float(r["2026-02-10"] / sma["2026-02-10"] - 1)
    print("2/2~4/10 신호", sig_feb, "2/10 비율/SMA-1", margin_0210)

    clean = lambda d: {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items() if not k.startswith("_")}
    payload = {
        "candidate": "HYG/IEF(Adj Close)<SMA50 → 노출 0.5, 최소 5거래일 유지",
        "base": clean(base), "recomputed": clean(cand), "spells26": cand["_spells26"], "recomputed_e0": clean(cand0),
        "claim_compare": disc, "timing_audit": audit,
        "delay_plus1": clean(late), "delay_plus1_e0": clean(late0),
        "timing_value_vs_fixed": {k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in v.items()}
                                  for k, v in tv.items()}, "avg_exposure_2026": a26,
        "timing_value_vs_fixed_e0": {k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in v.items()}
                                     for k, v in tv0.items()}, "avg_exposure_2026_e0": a26_0,
        "neighbors": {"n": nb, "counts": cnt, "by_sma": by_sma, "by_low": {str(k): v for k, v in by_low.items()},
                      "by_hold": by_hold, "rows": rows},
        "pairs": pairs,
        "placebo_circular_shift_2026": {"actual": actual, "n": len(shifts), "share_below_actual": pct,
                                        "median": float(np.median(shifts)), "p90": float(np.quantile(shifts, 0.9))},
        "decomp_2026": {"없앤_6/25~7/1": no_jul, "없앤_2/11~4/1": no_feb},
        "placebo_circular_shift_hold": placebo_hold,
        "delay_plus1_vs_fixed": {"2026_평균노출": late["2026_평균노출"], "spells26": late["_spells26"],
                                 "고정_2026": clean(late_fx["2026"]), "고정_2025평균": list(late_fx["2025"]),
                                 "고정_2024평균": list(late_fx["2024"])},
        "signal_2026_02_02_04_10": sig_feb, "margin_2026_02_10": margin_0210,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "v2_credit_trend.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=float),
                                                  encoding="utf-8")
    # 출력 요약
    print("\n후보", {k: fmt(v) for k, v in cand.items() if isinstance(v, tuple)})
    print("지연+1", {k: fmt(v) for k, v in late.items() if isinstance(v, tuple)})
    print("지연+1 노출0", {k: fmt(v) for k, v in late0.items() if isinstance(v, tuple)})
    print(f"고정 {a26:.4f}", {k: fmt(v) for k, v in fx26.items()})
    print(f"고정 2025평균 {cand['2025_평균노출']:.4f}", fmt(fx25["hold_2025"]), f"2024평균 {cand['2024_평균노출']:.4f}",
          fmt(fx24["hold_2024"]))
    print(f"노출0 후보 평균 {a26_0:.4f}", {k: fmt(v) for k, v in cand0.items() if isinstance(v, tuple)})
    print("고정(노출0판 평균)", {k: fmt(v) for k, v in fx26_0.items()})
    print("노출0 2025/2024 평균", cand0["2025_평균노출"], cand0["2024_평균노출"], tv0["hold_2025"], tv0["hold_2024"])
    print("\n[이웃 72]")
    for x in rows:
        print(f"SMA{x['sma']:>3} 노출{x['low']:.2f} 유지{x['hold']:>2} | 2026 {fmt(x['2026_전체'])} 고정 {fmt(x['고정26_전체'])} "
              f"앞 {fmt(x['2026_앞'])} 뒤 {x['2026_뒤'][0]:+.1%} AI {x['AI쇼크']:+.1%} | 2025 {x['2025'][0]:+.1%} 고정 "
              f"{x['고정25'][0]:+.1%} | 2024 {x['2024'][0]:+.1%} 고정 {x['고정24'][0]:+.1%} | 노출 {x['노출26']:.2f} "
              f"빠짐 {x['빠짐26']} | {'통과' if x['2026통과'] else '-'}")


if __name__ == "__main__":
    main()
