"""40분봉 대피·재매수 — 기존 지표 노하우 적용 (2026-11-22).

사용자(2026-11-22): "자주 뜨는 건 좋은데 형편없다. 기존에 지표 만들던 노하우를 써서, 특정 진입한 게 언제 대피하는지. 지금 40분봉을 본다. 기민하게."
노하우(정본: docs/research_index.md, kalman_zones·trend_zones 보고서)
- 정규장 봉만 쓴다(연장시간 봉은 모든 시간봉에서 나빴다).
- 평활은 거래일 단위로 맞춘다(40분 정규장 하루 10봉). 칼만선은 정상상태에서 EMA와 같다. 폭은 가격−선의 RMS.
- 개장 직후 봉(첫 두 봉)에 반응하면 나빴다 → 그 봉을 빼거나 장 시작 전에 판단.
- 대피(매도 구간) 뒤에는 평균 반등이 잦다. 손절은 급락 방어용이다.

사전 선언(결과 보기 전)
- 종목 9: SOXL, MU, AMD, ARM, DELL, INTC, SNDK, STX, WDC (2026-09 데스크 7종목 + SOXL + 사용자 차트 MU). 종목마다 따로 맞추지 않는다.
- 봉: 정규장 40분봉(09:30 기준). 체결은 정규장 1분봉(종가 신호 = 봉 완성 뒤 첫 1분봉 시가, 닿으면 = 봉 시작 때 아는 선에 닿은 첫 1분봉, min(시가, 선)).
- 선: EMA(n), n = d×10봉, d ∈ {2,3,5,10}거래일. 폭 W = √(n봉 평균((종가−EMA)²)).
  대피선 = EMA − b·W (b ∈ {0, 0.75, 1.75}), 재매수선 = EMA + a·W (a ∈ {0, 0.75}).
- 대피 방식: 종가 확인 / 닿으면(미리 거는 손절 주문).
- 판단 시점: T0 모든 봉 / T1 개장 뒤 첫 두 봉(09:30·10:10 시작)의 종가 신호 무시 / T2 하루 마지막 봉에서만 판단(장 시작 전 판단, 닿으면 선은 전날 마지막 봉 값으로 하루 고정).
- 격자 4×3×2×2×3 = 144개 × 9종목. 편도 0.1% 차감.
- 자격: 고르는 구간 종목 중앙 전환 월 3~12번(느린 선 2.2번보다 기민). 순위: 고르는 구간(2026-01-02~10-14) '큰 3일 뺀 시점 값'의 종목 중앙값.
- 확인(선택에 안 씀): 급락 구간 10-15~11-20, 최근 분기 08-20~11-20, 2025(보조). 비교: 보유, 지금 지표 느린 선·보통 선.
결과: ../results/grid40.json, grid40.csv
"""
import json, os, sys, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../soxl_escape_20261120/lines/code"))
import engine as E  # noqa: E402

OUT = os.path.join(HERE, "..", "results")
os.makedirs(OUT, exist_ok=True)
SYMS = ["SOXL", "MU", "AMD", "ARM", "DELL", "INTC", "SNDK", "STX", "WDC"]
COST = 0.001
WIN = {"SEL": ("2026-01-02", "2026-10-14"), "CONF": ("2026-10-15", "2026-11-20"), "Q": ("2026-08-20", "2026-11-20"),
       "Y26": ("2026-01-02", "2026-11-20"), "Y2025": ("2025-03-03", "2025-12-31")}
SEL_MONTHS = 9.4
L = lambda x: 100 * np.log1p(x / 100)   # noqa: E731


def configs():
    out = []
    for d in (2, 3, 5, 10):
        for b in (0.0, 0.75, 1.75):
            for a in (0.0, 0.75):
                for mode in ("close", "touch"):
                    for tm in ("T0", "T1", "T2"):
                        out.append(dict(d=d, b=b, a=a, mode=mode, tm=tm))
    return out


def name(c):
    m = "종가" if c["mode"] == "close" else "닿으면"
    t = {"T0": "모든 봉", "T1": "첫 두 봉 무시", "T2": "장 전 판단"}[c["tm"]]
    return f"{c['d']}일 EMA · 대피 −{c['b']}W · 재매수 +{c['a']}W · {m} · {t}"


def run(sym, c, cache):
    k = (sym, c["d"])
    if k not in cache:
        b = E.bars(sym, 40, "reg")
        n = c["d"] * 10
        e = E.ema(b["Close"], n)
        w = np.sqrt(((b["Close"] - e) ** 2).rolling(n, min_periods=n).mean())
        hm = b.index.hour * 60 + b.index.minute
        last = pd.Series(b["date"].values, index=b.index)
        is_last = last.ne(last.shift(-1))
        cache[k] = (b, e, w, pd.Series(hm >= 10 * 60 + 50, index=b.index), is_last)
    b, e, w, late, is_last = cache[k]
    C = b["Close"]
    xl, el = e - c["b"] * w, e + c["a"] * w
    if c["tm"] == "T0":
        gate = pd.Series(True, index=b.index)
        xt = xl.shift(1)
    elif c["tm"] == "T1":
        gate = late
        xt = xl.shift(1)
    else:
        gate = is_last
        xt = xl.where(is_last).ffill().shift(1)
    if c["mode"] == "close":
        return E.simulate(b, exit_close=(C < xl) & gate, entry_close=(C > el) & gate, exec_sess="reg")
    return E.simulate(b, exit_touch=xt, entry_close=(C > el) & gate, exec_sess="reg")


def refs(sym):
    bd = E.bars(sym, "D", "reg"); Cd = bd["Close"]
    e10, e20, e3, e15 = (E.ema(Cd, n) for n in (10, 20, 3, 15))
    slow = E.simulate(bd, exit_close=Cd < np.minimum(e10, e20), entry_close=Cd > np.maximum(e10, e20), exec_sess="reg")
    mid = E.simulate(bd, exit_touch=np.minimum(e3, e15).shift(1), entry_close=Cd > np.maximum(e3, e15), exec_sess="reg")
    return {"느린 선(일봉 EMA10·20 종가)": slow, "보통 선(일봉 EMA3·15 닿으면)": mid}


def score(pos, corr, sw):
    ev = {w: E.evaluate(pos, corr, sw, rng, COST) for w, rng in WIN.items()}
    f = ev["SEL"]["expo%"] / 100
    for w in ("CONF", "Q"):
        ev[w]["tvL%p"] = round(L(ev[w]["ret%"]) - f * L(ev[w]["bh%"]), 2)
    return ev


def main():
    t0 = time.time()
    cfgs = configs()
    rows = []
    for sym in SYMS:
        E.use(sym)
        cache = {}
        for i, c in enumerate(cfgs):
            ev = score(*run(sym, c, cache))
            rows.append({"sym": sym, "i": i, **{f"{w}|{k}": v for w in WIN for k, v in ev[w].items()}})
        for nm, p in refs(sym).items():
            ev = score(*p)
            rows.append({"sym": sym, "i": nm, **{f"{w}|{k}": v for w in WIN for k, v in ev[w].items()}})
        print(sym, round(time.time() - t0), "s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "grid40.csv"), index=False, encoding="utf-8-sig")

    def pooled(g):
        return pd.Series({
            "전환/월": g["SEL|switches"].median() / SEL_MONTHS,
            "고른구간 큰3일뺌": g["SEL|tv_ex_top3%p"].median(), "고른구간 시점값": g["SEL|tv%p"].median(),
            "고른구간 양수 종목": int((g["SEL|tv%p"] > 0).sum()),
            "고른구간 수익%": g["SEL|ret%"].median(), "고른구간 낙폭%": g["SEL|mdd%"].median(),
            "급락 tvL": g["CONF|tvL%p"].median(), "급락 수익%": g["CONF|ret%"].median(),
            "최근분기 tvL": g["Q|tvL%p"].median(), "최근분기 양수 종목": int((g["Q|tvL%p"] > 0).sum()),
            "최근분기 수익%": g["Q|ret%"].median(), "최근분기 낙폭%": g["Q|mdd%"].median(),
            "2026 수익%": g["Y26|ret%"].median(), "2026 낙폭%": g["Y26|mdd%"].median(),
            "2025 시점값": g["Y2025|tv%p"].median(), "2025 양수 종목": int((g["Y2025|tv%p"] > 0).sum()),
            "보유 2026%": g["Y26|bh%"].median(), "보유 최근분기%": g["Q|bh%"].median(), "보유 2026 낙폭%": g["Y26|bh_mdd%"].median(),
        })
    P = df.groupby("i").apply(pooled)
    names = {i: name(c) for i, c in enumerate(cfgs)}
    P.index = [names.get(i, i) for i in P.index]
    P = P.round(2)
    grid = P[[isinstance(k, str) and "EMA ·" in k for k in P.index]]
    elig = grid[(grid["전환/월"] >= 3) & (grid["전환/월"] <= 12)].sort_values("고른구간 큰3일뺌", ascending=False)
    ref = P.loc[[k for k in P.index if "선(" in k]]
    by = {}
    for key, fn in (("평활", lambda c: c["d"]), ("대피 방식", lambda c: c["mode"]), ("판단 시점", lambda c: c["tm"]), ("대피 띠 b", lambda c: c["b"])):
        lab = pd.Series({name(c): fn(c) for c in cfgs})
        g = grid.groupby(lab.reindex(grid.index).values)[["전환/월", "고른구간 큰3일뺌", "급락 tvL", "최근분기 tvL", "2025 시점값"]].median().round(2)
        by[key] = g.reset_index().rename(columns={"index": key}).to_dict("records")
    res = {"configs": len(cfgs), "symbols": SYMS, "eligible": int(len(elig)), "ranking": elig.reset_index().rename(columns={"index": "규칙"}).to_dict("records"),
           "reference": ref.reset_index().rename(columns={"index": "규칙"}).to_dict("records"), "by": by,
           "cfgs": {name(c): c for c in cfgs}}
    json.dump(res, open(os.path.join(OUT, "grid40.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0, default=float)
    pd.set_option("display.width", 330); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 60)
    cols = ["전환/월", "고른구간 큰3일뺌", "고른구간 양수 종목", "고른구간 수익%", "고른구간 낙폭%", "급락 tvL", "최근분기 tvL", "최근분기 양수 종목", "최근분기 수익%", "최근분기 낙폭%", "2026 수익%", "2026 낙폭%", "2025 시점값", "2025 양수 종목"]
    print("자격", len(elig), "/", len(grid))
    print(elig[cols].head(15).to_string())
    print("--- 비교"); print(ref[cols + ["보유 2026%", "보유 최근분기%", "보유 2026 낙폭%"]].to_string())
    for k, v in by.items():
        print("---", k); print(pd.DataFrame(v).to_string(index=False))
    print(round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
