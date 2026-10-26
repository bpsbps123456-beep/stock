"""hybrid 계열(2026-10-10): 160분봉 RSI 70 돌파를 사는 신호로 두되, 그때 상대 강도가 최상위일 때만 산다.
상대 강도가 상위권에서 밀리면(또는 종가 고점 대비 15% 하락) 판다.

  산다  k일 시가: sig70[s].iloc[k-1] 이고, k일 대상(H.eligible) 안에서 R.iloc[k-1] 순위가 N위 안
  판다  k일 시가: R.iloc[k-1] 순위가 M위 밖, 또는 k-1일까지 종가 최고 대비 k-1일 종가가 -15% 이하(stop=True일 때)
  우선  빈자리보다 신호가 많으면 R.iloc[k-1] 큰 순
  순위  k일 대상 중 R.iloc[k-1] 값이 있는 종목끼리. 들고 있는데 대상에서 빠진 종목은
        '대상 중 자기보다 R이 큰 종목 수 + 1'로 잰다(대상 이탈만으로 팔지 않는다). R이 비면 순위 매도 안 함.
격자(사전 등록 10개): R ∈ {rs(20), r160} × N ∈ {10, 15} × M ∈ {15, 25}, 손절 15% 포함 8개
                      + R=rs(20), N=10, M ∈ {15, 25} 손절 없음 2개.
고르기: 앞 1/2~5/29 수익 최고(거래 20회 미만·평균 보유 종목 수 3 미만 제외) → 뒤 구간은 확인용.
다시 돌리기: PYTHONIOENCODING=utf-8 py variants/hybrid.py   (code 폴더에서)
"""
import os, sys, json
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, CODE)
from harness import Harness, table, months_table   # noqa: E402
from rsi_breakout import RES                         # noqa: E402

FRONT, BACK = "앞 1/2~5/29", "뒤 6/1~10/9"
STOP = 0.15


def make_ranker(H, R):
    """k → (rank(s) 함수). k일 대상 중 R.iloc[k-1] 기준, 1이 가장 강함. 같은 값은 낮은 순위 번호."""
    cache = {}
    arr = R.to_numpy()
    col = {s: i for i, s in enumerate(R.columns)}

    def at(k):
        if k not in cache:
            elig = [s for s in H.eligible(H.days[k]) if s in col]
            v = arr[k - 1, [col[s] for s in elig]] if elig else np.array([])
            v = np.sort(v[np.isfinite(v)])[::-1]
            row = arr[k - 1]

            def rank(s):
                x = row[col[s]] if s in col else np.nan
                if not np.isfinite(x):
                    return None
                return int(np.sum(v > x)) + 1
            cache[k] = rank
        return cache[k]
    return at


def run_variant(H, R, rname, N, M, stop):
    ranker = make_ranker(H, R)
    sig = H.sig70

    def entry(s, k):
        if s not in sig.columns or not bool(sig[s].iloc[k - 1]):
            return False
        r = ranker(k)(s)
        return r is not None and r <= N

    def exit_rule(s, k, p):
        r = ranker(k)(s)
        if r is not None and r > M:
            return True
        if stop:
            c = H.cls[s].iloc[k - 1]
            if np.isfinite(c) and c <= (1 - STOP) * p["peak"]:
                return True
        return False

    pick = H.entry_exit_picker(entry, exit_rule, priority=lambda k: R.iloc[k - 1])
    df = H.simulate(pick)
    name = f"hybrid R={rname} N{N} M{M}" + (" 손절15%" if stop else " 손절없음")
    res = H.evaluate(name, df)
    res["params"] = dict(R=rname, N=N, M=M, stop=(STOP if stop else None))
    res.update(trade_stats(df))
    return res, df


def trade_stats(df):
    """보유 목록 변화로 센 거래(같은 시가에 팔고 다시 산 것은 거래가 아니다)와 거래당 보유 거래일."""
    held = [set(h.split(",")) - {""} for h in df.held]
    runs, open_ = [], {}
    prev = set()
    for i, h in enumerate(held):
        for s in h - prev:
            open_[s] = i
        for s in prev - h:
            runs.append(i - open_.pop(s))
        prev = h
    runs += [len(held) - i0 for i0 in open_.values()]
    return {"거래수": len(runs), "거래당보유일": float(np.mean(runs)) if runs else 0.0}


def main():
    H = Harness()
    Rs = {"rs20": H.rs(20), "r160": H.r160}
    grid = [(rn, N, M, True) for rn in ("rs20", "r160") for N in (10, 15) for M in (15, 25)]
    grid += [("rs20", 10, 15, False), ("rs20", 10, 25, False)]
    results, dfs = [], {}
    for rn, N, M, st in grid:
        res, df = run_variant(H, Rs[rn], rn, N, M, st)
        results.append(res)
        dfs[res["name"]] = df

    base = json.load(open(os.path.join(RES, "variants", "baselines.json"), encoding="utf-8"))
    print(f"격자 {len(grid)}개 (사전 등록)")
    print(table(results + base))
    print(months_table(results + base))
    print("\n  이름                                  거래수  거래당보유일  평균보유  기여상위5")
    for r in results:
        print(f"  {r['name'][:36].ljust(36)} {r['거래수']:6d} {r['거래당보유일']:12.1f} {r['평균보유']:8.2f}  {r['기여상위5']}")

    ok = [r for r in results if r["거래수"] >= 20 and r["평균보유"] >= 3]
    best = max(ok, key=lambda r: r[FRONT]["수익"]) if ok else None
    if best:
        print(f"\n앞 구간으로 고른 것: {best['name']}  앞 {best[FRONT]['수익']:+.1%} → 뒤 {best[BACK]['수익']:+.1%}, "
              f"9/1~ {best['9/1~10/9']['수익']:+.1%}, 10/1~ {best['10/1~10/9']['수익']:+.1%}")
    bmap = {b["name"]: b for b in base}
    qqq_b, desk_b = bmap["QQQ"][BACK]["수익"], bmap.get("데스크 7종목(#774)", {}).get(BACK, {}).get("수익", np.nan)
    both = [r["name"] for r in results if r[FRONT]["수익"] > 0 and r[BACK]["수익"] > 0]
    beat = [r["name"] for r in results if r[BACK]["수익"] > qqq_b and r[BACK]["수익"] > desk_b]
    print("앞·뒤 모두 +:", both)
    print(f"뒤 구간 QQQ({qqq_b:+.1%})·데스크({desk_b:+.1%}) 둘 다 이김:", beat)

    for r in results:
        r["선택"] = bool(best is not None and r["name"] == best["name"])
    os.makedirs(os.path.join(RES, "variants"), exist_ok=True)
    out = os.path.join(RES, "variants", "hybrid.json")
    json.dump(results, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("저장:", os.path.abspath(out))
    return H, results, dfs


if __name__ == "__main__":
    main()
