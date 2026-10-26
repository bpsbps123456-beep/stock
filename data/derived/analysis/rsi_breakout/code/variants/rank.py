"""rank 계열(2026-10-10): '가장 강한 종목을 그날그날 다시 고르기' — 점수 상위 7을 같은 비중으로 든다(H.rank_picker).

점수(모두 k-1일 행, k일 시가 체결)
  r160      그날 마지막 160분봉 RSI
  r160_5d   160분봉 RSI(그날 마지막 봉) 최근 5거래일(k-5..k-1) 평균, 5일 중 3일 이상 값이 있을 때
  rs20      QQQ 대비 20일 상대수익
  rs60      QQQ 대비 60일 상대수익
  near252   종가 ÷ 최근 252거래일 고가 최고(1에 가까울수록 전고점 근처)
  comp      rs20·rs60·near252 각각의 그날 대상 종목 안 백분위 순위 평균(셋 다 있는 종목만)
방식(3)    매일+완충12(들고 있으면 12위 안이면 유지) / 주 첫 거래일+완충12 / 매일 완충 없음
격자      6 × 3 = 18개(사전 등록). 고르기는 앞 1/2~5/29 수익 최고(매수 20회 미만·평균 보유 3종목 미만 제외),
          뒤 6/1~10/9·9/1~10/9·10/1~10/9는 확인용.
실행      py data/derived/analysis/rsi_breakout/code/variants/rank.py  → results/variants/rank.json
"""
import json, os, sys, time
from collections import Counter, defaultdict
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, CODE)
from harness import Harness, table, months_table          # noqa: E402
from rsi_breakout import RES, SLOTS                        # noqa: E402

FRONT, BACK = "앞 1/2~5/29", "뒤 6/1~10/9"
MODES = (("매일+완충12", dict(buffer=12, weekly=False)),
         ("주간+완충12", dict(buffer=12, weekly=True)),
         ("매일 완충없음", dict(buffer=None, weekly=False)))


def sector_of(sym, rev):
    try:
        from watchlist import resolve_sector
    except Exception:
        return "?"
    for key in (sym, rev.get(sym, "").split(" ")[0]):
        sec = resolve_sector(key, default="")
        if sec:
            return sec
    return "미분류"


def trade_stats(H, df):
    """매수 횟수(새로 들어온 종목 수)와 평균 보유 거래일."""
    prev, entries, spans, start = set(), 0, [], {}
    for i, h in enumerate(df.held):
        cur = set(h.split(",")) if h else set()
        for s in cur - prev:
            entries += 1
            start[s] = i
        for s in prev - cur:
            spans.append(i - start.pop(s))
        prev = cur
    spans += [len(df) - v for v in start.values()]          # 끝까지 든 것
    return entries, (float(np.mean(spans)) if spans else float("nan"))


def window_detail(H, df, a, b, rev):
    """구간 안 종목별 기여(하루 w×수익 단순 합)와 보유 일수, 섹터 쏠림."""
    o = H.opn
    pos = {d: i for i, d in enumerate(H.days)}
    x = df[(df.day >= a) & (df.day <= b)]
    con, days_held = defaultdict(float), Counter()
    for d, h, n in zip(x.day, x.held, x.n):
        k = pos[d]
        for s in (h.split(",") if h else []):
            a0, b0 = o[s].iloc[k], o[s].iloc[k + 1]
            r = b0 / a0 - 1 if np.isfinite(a0) and np.isfinite(b0) and a0 > 0 else 0.0
            con[s] += r / max(SLOTS, n)
            days_held[s] += 1
    sec = Counter()
    for s, c in days_held.items():
        sec[sector_of(s, rev)] += c
    tot = sum(days_held.values()) or 1
    return {"기여": sorted(((s, round(v, 4)) for s, v in con.items()), key=lambda t: -t[1]),
            "보유일": dict(days_held.most_common()),
            "섹터몫": {k: round(v / tot, 3) for k, v in sec.most_common()},
            "마지막날": x.held.iloc[-1] if len(x) else ""}


def main():
    t0 = time.time()
    H = Harness()
    print(f"Harness {time.time() - t0:.0f}초, 거래일 {len(H.days)}, 종목 {len(H.syms)}")
    rev = {v: k for k, v in H.P.tick.items()}

    rs20, rs60 = H.rs(20), H.rs(60)
    near = H.cls / H.hi(252)
    r160_5 = H.r160.rolling(5, min_periods=3).mean()
    bad = int((near > 1.0001).sum().sum())
    print(f"점검: 종가/252일고가 > 1 인 칸 {bad}개, near 최대 {np.nanmax(near.to_numpy()):.4f}")

    def elig_k(k):
        return [s for s in H.eligible(H.days[k]) if s in H.opn.columns and np.isfinite(H.opn[s].iloc[k])]

    def comp(k):
        z = pd.DataFrame({"a": rs20.iloc[k - 1], "b": rs60.iloc[k - 1], "c": near.iloc[k - 1]}).reindex(elig_k(k)).dropna()
        return z.rank(pct=True).mean(axis=1)

    scores = {"r160": lambda k: H.r160.iloc[k - 1],
              "r160_5d": lambda k: r160_5.iloc[k - 1],
              "rs20": lambda k: rs20.iloc[k - 1],
              "rs60": lambda k: rs60.iloc[k - 1],
              "near252": lambda k: near.iloc[k - 1],
              "comp": comp}

    results, raw = [], {}
    for sname, fn in scores.items():
        for mname, kw in MODES:
            name = f"rank {sname} {mname}"
            df = H.simulate(H.rank_picker(fn, slots=SLOTS, **kw))
            r = H.evaluate(name, df)
            n_tr, hold_d = trade_stats(H, df)
            r["params"] = {"score": sname, "mode": mname, "slots": SLOTS, **kw}
            r["매수횟수"], r["평균보유일"] = n_tr, hold_d
            r["10월"] = window_detail(H, df, "2026-10-01", "2026-10-09", rev)
            r["뒤 구간 섹터몫"] = window_detail(H, df, "2026-06-01", "2026-10-09", rev)["섹터몫"]
            results.append(r)
            raw[name] = df
            print(f"  {name}: 앞 {r[FRONT]['수익']:+.1%} 뒤 {r[BACK]['수익']:+.1%} 매수 {n_tr} 평균보유 {r['평균보유']:.1f}")

    # 고르기: 앞 수익 최고(매수 20회 이상·평균 보유 3종목 이상)
    ok = [r for r in results if r["매수횟수"] >= 20 and r["평균보유"] >= 3]
    best = max(ok, key=lambda r: r[FRONT]["수익"])
    base = {b["name"]: b for b in json.load(open(os.path.join(RES, "variants", "baselines.json"), encoding="utf-8"))}
    qqq_b = base["QQQ"][BACK]["수익"]
    desk_b = base.get("데스크 7종목(#774)", {}).get(BACK, {}).get("수익", float("nan"))
    both = [r["name"] for r in results if r[FRONT]["수익"] > 0 and r[BACK]["수익"] > 0]
    beat = [r["name"] for r in results if r[BACK]["수익"] > qqq_b and r[BACK]["수익"] > desk_b]
    summary = {"격자": "6 점수 × 3 방식 = 18", "선택": best["name"], "선택규칙": "앞 수익 최고, 매수≥20·평균보유≥3",
               "앞뒤 모두 플러스": both, "뒤에서 QQQ·데스크 모두 이김": beat,
               "QQQ 뒤": qqq_b, "데스크 뒤": desk_b}
    out = {"family": "rank", "summary": summary, "results": results}
    os.makedirs(os.path.join(RES, "variants"), exist_ok=True)
    with open(os.path.join(RES, "variants", "rank.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)

    show = results + [base[n] for n in ("사용자 규칙(RSI70→K95)", "데스크 7종목(#774)", "QQQ", "보유 전체 같은 비중") if n in base]
    print()
    print(table(show))
    print()
    print(months_table(show))
    print()
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"\n선택 {best['name']}: 10월 기여 {best['10월']['기여'][:7]}")
    print(f"  10월 섹터몫 {best['10월']['섹터몫']}")
    print(f"  뒤 구간 섹터몫 {best['뒤 구간 섹터몫']}")
    print(f"완료 {time.time() - t0:.0f}초")


if __name__ == "__main__":
    main()
