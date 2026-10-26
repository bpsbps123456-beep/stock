"""매도 규칙 비교(2026-10-10): 사용자 매수(160분봉 RSI 70 상향 돌파 다음 날 시가, 5일 수익 큰 순, 7자리)는 그대로,
매도만 바꾼다. 미리 정한 13개 + 사용자 K>95 기준 = 14개. 고르기는 앞 1/2~5/29 수익(거래 20건 미만·평균 보유 3 미만 제외),
뒤 6/1~10/9·9/1~10/9·10/1~10/9는 확인용.

시점: k일 시가 매도 판단은 k-1일(또는 k-2일) 종가·K·160분봉 RSI만 쓴다. peak는 harness가 k-1일 종가까지로 갱신한다.
실행: PYTHONIOENCODING=utf-8 py data/derived/analysis/rsi_breakout/code/variants/exits.py
"""
import os, sys, json
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from harness import Harness, table, months_table, PERIODS          # noqa: E402
from rsi_breakout import RES                                        # noqa: E402

FRONT = PERIODS[0]


def arr(df):
    """지표 표 → {종목: numpy 배열}(속도)."""
    return {s: df[s].to_numpy(dtype=float) for s in df.columns}


def build_rules(H):
    C, K, R = arr(H.cls), arr(H.k), arr(H.r160)
    M10, M20 = arr(H.ma(10)), arr(H.ma(20))
    nan = np.full(len(H.days), np.nan)

    def trail(x):
        return lambda s, k, p: C[s][k - 1] <= p["peak"] * (1 - x)

    def below_ma(M):
        return lambda s, k, p: C[s][k - 1] < M[s][k - 1]

    def r160_below(th):
        return lambda s, k, p: R.get(s, nan)[k - 1] < th

    def k_cross_down(th):
        return lambda s, k, p: K[s][k - 2] >= th > K[s][k - 1]

    def k_below(th):
        return lambda s, k, p: K[s][k - 1] < th

    def hold(n):
        return lambda s, k, p: k - p["k0"] >= n

    t12, m20 = trail(0.12), below_ma(M20)
    return [
        ("기준: K>95(사용자)", "daily Stoch RSI K[k-1] > 95 (P.exit_hit)", lambda s, k, p: H.P.exit_hit(s, k - 1)),
        ("고점 대비 -8% 종가", "cls[k-1] <= peak*(1-0.08)", trail(0.08)),
        ("고점 대비 -12% 종가", "cls[k-1] <= peak*(1-0.12)", t12),
        ("고점 대비 -16% 종가", "cls[k-1] <= peak*(1-0.16)", trail(0.16)),
        ("종가 < 10일선", "cls[k-1] < ma10[k-1]", below_ma(M10)),
        ("종가 < 20일선", "cls[k-1] < ma20[k-1]", m20),
        ("160분 RSI < 50", "r160[k-1] < 50", r160_below(50)),
        ("160분 RSI < 40", "r160[k-1] < 40", r160_below(40)),
        ("K 80 하향 돌파", "K[k-2] >= 80 > K[k-1]", k_cross_down(80)),
        ("K < 50", "K[k-1] < 50", k_below(50)),
        ("5거래일 보유", "k - k0 >= 5", hold(5)),
        ("10거래일 보유", "k - k0 >= 10", hold(10)),
        ("20거래일 보유", "k - k0 >= 20", hold(20)),
        ("고점 -12% 또는 20일선 아래", "cls[k-1] <= peak*0.88 OR cls[k-1] < ma20[k-1]", lambda s, k, p: t12(s, k, p) or m20(s, k, p)),
    ]


def run_one(H, name, params, exit_rule):
    """사용자 매수 + 이 매도로 계좌를 돌리고, 거래 단위 기록(같은 날 팔고 다시 산 것도 한 건씩)을 붙인다."""
    sig = {s: H.sig70[s].to_numpy(dtype=bool) for s in H.sig70.columns}
    exits_now = set()

    def ex(s, k, p):
        hit = bool(exit_rule(s, k, p))
        if hit:
            exits_now.add((s, k))
        return hit

    inner = H.entry_exit_picker(lambda s, k: bool(sig[s][k - 1]) if s in sig else False, ex,
                                priority=lambda k: H.ret(5).iloc[k - 1])
    O = arr(H.opn)
    open_tr, trades = {}, []

    def pick(k, held):
        out = inner(k, held)
        for s in held:
            if s not in out or (s, k) in exits_now:
                k_in = open_tr.pop(s)
                trades.append(dict(sym=s, k_in=k_in, k_out=k, ret=O[s][k] / O[s][k_in] - 1))
        for s in out:
            if s not in open_tr:
                open_tr[s] = k
        return out

    df = H.simulate(pick)
    last = len(H.days) - 1
    for s, k_in in open_tr.items():                                    # 끝까지 들고 있는 것: 마지막 시가로 평가
        trades.append(dict(sym=s, k_in=k_in, k_out=last, ret=O[s][last] / O[s][k_in] - 1, open=True))
    tr = pd.DataFrame(trades)
    tr["entry"] = [H.days[i] for i in tr.k_in]
    tr["days"] = tr.k_out - tr.k_in
    res = H.evaluate(name, df)
    res["params"] = params
    res["거래"] = {}
    for lab, a, b in PERIODS:
        x = tr[(tr.entry >= a) & (tr.entry <= b)]
        res["거래"][lab] = dict(건수=int(len(x)), 승률=float((x.ret > 0).mean()) if len(x) else None,
                              평균수익=float(x.ret.mean()) if len(x) else None,
                              보유일=float(x.days.mean()) if len(x) else None,
                              최악=float(x.ret.min()) if len(x) else None)
    worst = tr.sort_values("ret").head(3)
    res["최악3"] = [(r.sym, r.entry, round(float(r.ret), 4)) for r in worst.itertuples()]
    return res


def main():
    H = Harness()
    rules = build_rules(H)
    res = [run_one(H, n, p, f) for n, p, f in rules]
    os.makedirs(os.path.join(RES, "variants"), exist_ok=True)
    json.dump(res, open(os.path.join(RES, "variants", "exits.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print(f"격자 {len(rules) - 1}개 + 사용자 기준 1개 = {len(rules)}개 (매수·우선순위·7자리 고정, 매도만 다름)\n")
    print(table(res))
    print()
    print(months_table(res))
    print("\n  이름".ljust(36) + "앞 건수 승률 평균 보유일 | 뒤 건수 승률 평균 보유일 | 평균보유 | 최악3")
    for r in res:
        f, b = r["거래"][FRONT[0]], r["거래"]["뒤 6/1~10/9"]
        fmt = lambda t: (f"{t['건수']:4d} {t['승률']:.0%} {t['평균수익']:+.1%} {t['보유일']:4.1f}"
                         if t["건수"] else "   0")
        print(f"  {r['name'][:34].ljust(34)}{fmt(f)} | {fmt(b)} | {r['평균보유']:.2f} | {r['최악3']}")
    ok = [r for r in res if r["거래"][FRONT[0]]["건수"] >= 20 and r["평균보유"] >= 3]
    best = max(ok, key=lambda r: r[FRONT[0]]["수익"])
    print(f"\n앞 구간 선택: {best['name']} 앞 {best[FRONT[0]]['수익']:+.1%} → 뒤 {best['뒤 6/1~10/9']['수익']:+.1%}, "
          f"9/1~ {best['9/1~10/9']['수익']:+.1%}, 10/1~ {best['10/1~10/9']['수익']:+.1%}")
    base = {b["name"]: b for b in json.load(open(os.path.join(RES, "variants", "baselines.json"), encoding="utf-8"))}
    desk = next(v for n, v in base.items() if n.startswith("데스크"))
    qqq = base["QQQ"]
    both = [r["name"] for r in res if r[FRONT[0]]["수익"] > 0 and r["뒤 6/1~10/9"]["수익"] > 0]
    beat = [n for n in both if (lambda r: r["뒤 6/1~10/9"]["수익"] > max(desk["뒤 6/1~10/9"]["수익"], qqq["뒤 6/1~10/9"]["수익"]))(
        next(x for x in res if x["name"] == n))]
    print("앞·뒤 모두 플러스:", both)
    print("그중 뒤에서 데스크·QQQ 모두 이김:", beat)


if __name__ == "__main__":
    main()
