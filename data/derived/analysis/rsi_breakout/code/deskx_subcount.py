"""데스크 7종목 규칙에 '세부 섹터별 종목 수 한도'를 얹은 subcount 계열(2026-10-10, 연구용 스크래치).

질문(사용자): 반도체 50% 합계 상한 대신 세부 섹터로 나눠 포트폴리오를 다양하게 가져가면 더 나은가?
규칙: 매주 재조정 때 120일 순위를 위에서부터 걷는다. 지난주 종목은 상위 12 안이면 그대로 두되(버퍼),
      세부 섹터 그룹마다 최대 m종목만 둔다(남은 종목이 m을 넘으면 순위가 낮은 쪽을 뺀다). 빈 자리는 위에서부터
      채우되 그룹이 찬 종목은 건너뛴다. 그다음은 평소의 상한 투영(종목 25%, 관심 섹터 60%, 반도체 예산 B).
세부 섹터: desk.strategies.subsector_of(key, ticker). '미분류'는 한 사업이 아니므로 종목마다 따로 한 그룹으로 본다.
          라벨은 지금 watchlist.py 값을 전 기간에 쓴다(섹터 상한과 같은 방식, 시점 고정 라벨 아님).
격자(사전 등록 4개): m ∈ {1, 2} × B ∈ {1.0(반도체 예산 없음), 0.60}. 관심 섹터 60%는 모든 변형에 남긴다.
사후 아이디어(1개, 선택 근거 아님): 'B=1.0, 한도 없음(m=7)' — 반도체 예산만 없앤 대조군.
선택: 앞 1/2~5/29 수익만으로 최선 변형을 고르고 뒤·9/1~10/9·월별·낙폭을 보고한다.
기준: 지금 데스크 규칙(반도체 50%)과 #774(60%). m=7은 두 기준을 정확히 재현해야 한다(check).
종목 기여: 각 시가 비중 × 그 종목 시가→다음 시가 수익 × 그날 계좌 가치(시작 1). 합 = 최종 가치 - 1.
실행: cd data/derived/analysis/rsi_breakout/code && PYTHONIOENCODING=utf-8 py deskx_subcount.py
"""
import json, os, sys
from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deskx import Engine, Params, PERIODS, START  # noqa: E402
from desk import config  # noqa: E402
from desk.strategies import project_caps, run_model, subsector_of, in_semis_budget, UNKNOWN  # noqa: E402
from desk.validate import codex_caps  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/deskx/subcount.json")
FRONT, BACK, SEP, FULL = (p[0] for p in PERIODS)
MONTHS = ["2026-06", "2026-07", "2026-08", "2026-09", "2026-10"]
NEXT_SESSION = "2026-10-12"


@dataclass(frozen=True)
class SubParams(Params):
    m: int = 7          # 세부 섹터 그룹당 최대 종목 수(7 = 한도 없음)


GRID = [(f"m={m} B={b:.2f}", SubParams(m=m, semis_cap=b)) for m in (1, 2) for b in (1.0, 0.60)]
POSTHOC = ("[사후] m=7 B=1.00 (예산만 제거)", SubParams(m=7, semis_cap=1.0))
LIVE = ("live 50% (m=7 B=0.50)", SubParams(m=7, semis_cap=0.50))
C774 = ("#774 60% (m=7 B=0.60)", SubParams(m=7, semis_cap=0.60))


class SubEngine(Engine):
    def __init__(self):
        super().__init__()
        self.log = None

    def group(self, key):
        s = subsector_of(key, self.ticker[key])
        return f"미분류:{self.ticker[key]}" if s == UNKNOWN else s

    @staticmethod
    def base_select(order, current, n, buf):
        keep = [x for x in order[:n + buf] if x in current][:n]
        for x in order:
            if len(keep) >= n:
                break
            if x not in keep:
                keep.append(x)
        return keep

    def limited_select(self, order, current, n, buf, m):
        """버퍼 유지 → 그룹당 m 한도 → 위에서부터 채우기(찬 그룹 건너뜀). m>=n이면 base_select와 같은 목록·순서."""
        cnt, keep = Counter(), []
        for x in order[:n + buf]:
            if x in current and len(keep) < n and cnt[self.group(x)] < m:
                keep.append(x)
                cnt[self.group(x)] += 1
        for x in order:
            if len(keep) >= n:
                break
            if x not in keep and cnt[self.group(x)] < m:
                keep.append(x)
                cnt[self.group(x)] += 1
        return keep

    def target(self, p, k, session, current):
        ranked = self.ranking(p, k, session)
        if not ranked:
            return [], {}, ranked
        order = [r["key"] for r in ranked]
        keep = self.limited_select(order, current, p.n, p.buffer, getattr(p, "m", p.n))
        wmap = {r["key"]: r["w"] for r in ranked}
        tot = sum(wmap[x] for x in keep)
        raw = {x: wmap[x] / tot for x in keep}
        saved = dict(config.RULE)
        try:
            config.RULE.update(stock_cap=p.stock_cap, sector_cap=p.sector_cap, semis_cap=p.semis_cap)
            tgt = project_caps(raw, self.sector)
        finally:
            config.RULE.clear()
            config.RULE.update(saved)
        if self.log is not None:
            unlimited = self.base_select(order, current, p.n, p.buffer)
            rank = {x: i + 1 for i, x in enumerate(order)}
            skipped = [{"name": self.ticker[x], "rank": rank[x], "group": self.group(x),
                        "held_last_week": x in current,
                        "behind": [self.ticker[y] for y in keep if self.group(y) == self.group(x)]}
                       for x in unlimited if x not in keep]
            self.log.append({"session": session, "keep": list(keep), "skipped": skipped,
                             "added": [self.ticker[x] for x in keep if x not in unlimited]})
        return keep, tgt, ranked

    def run_logged(self, p):
        self.log = []
        try:
            v, rows = self.run(p)
            log = self.log
        finally:
            self.log = None
        return v, rows, log


def contributions(E, v, rows):
    k0 = E.days.index(rows[0]["day"])
    vals = v.to_numpy()
    contrib = {}
    for i in range(len(rows) - 1):
        k = k0 + i + 1
        for key, wt in rows[i]["w"].items():
            o = E.col("opn", key)
            ret = (o[k] / o[k - 1] - 1) if o is not None and np.isfinite(o[k]) and np.isfinite(o[k - 1]) \
                and o[k - 1] > 0 else 0.0
            tk = E.ticker[key]
            contrib[tk] = contrib.get(tk, 0.0) + vals[i] * wt * ret
    gain = float(vals[-1] - 1)
    assert abs(sum(contrib.values()) - gain) < 1e-9 * max(1.0, abs(gain))
    top = sorted(contrib.items(), key=lambda x: -x[1])
    return {"상위3": [t for t, _ in top[:3]], "상위3비중": sum(c for _, c in top[:3]) / gain,
            "상위5기여": {t: round(c, 4) for t, c in top[:5]}, "하위3기여": {t: round(c, 4) for t, c in top[-3:]}}


def diversification(E, rows):
    nsub, maxsub, inv = [], [], []
    for row in rows:
        g = {}
        for key, wt in row["w"].items():
            if wt > 1e-12:
                g[E.group(key)] = g.get(E.group(key), 0.0) + wt
        nsub.append(len(g))
        maxsub.append(max(g.values()) if g else 0.0)
        inv.append(sum(row["w"].values()))
    return {"세부섹터수평균": float(np.mean(nsub)), "최대세부섹터비중평균": float(np.mean(maxsub)),
            "평균투자비중": float(np.mean(inv))}


def one(E, name, p, live_log=None):
    v, rows, log = E.run_logged(p)
    res = E.evaluate(name, v, rows)
    res["params"] = asdict(p)
    res.update(diversification(E, rows))
    res.update(contributions(E, v, rows))
    lim = [e for e in log if e["skipped"]]
    skip_info = {}
    for e in log:
        for s in e["skipped"]:
            d = skip_info.setdefault(s["name"], {"weeks": 0, "group": s["group"], "behind": Counter(), "first": e["session"]})
            d["weeks"] += 1
            d["behind"].update(s["behind"])
            d["last"] = e["session"]
    res["재조정수"] = len(log)
    res["한도작동주"] = len(lim)
    res["건너뛴종목"] = {k: {"weeks": d["weeks"], "group": d["group"], "behind": dict(d["behind"]),
                         "first": d["first"], "last": d["last"]}
                    for k, d in sorted(skip_info.items(), key=lambda x: -x[1]["weeks"])}
    res["대신들어온종목"] = dict(Counter(a for e in log for a in e["added"]).most_common())
    if live_log is not None:
        live = {e["session"]: set(e["keep"]) for e in live_log}
        diff = [e["session"] for e in log if set(e["keep"]) != live.get(e["session"])]
        res["live와다른주"] = len(diff)
        res["live와다른주_목록"] = diff
    res["_log"] = log
    res["_v"] = v
    return res


def preview(E, p, log):
    current = set(log[-1]["keep"])
    k = len(E.days)
    keep, tgt, ranked = E.target(p, k, NEXT_SESSION, current)
    rank = {r["key"]: (i + 1, r["mom"]) for i, r in enumerate(ranked)}
    hold = [{"name": E.ticker[x], "weight": round(tgt.get(x, 0.0), 4), "subsector": E.group(x),
             "sector": E.sector.get(x, UNKNOWN), "rank": rank[x][0], "mom120": round(rank[x][1], 4)} for x in keep]
    hold.sort(key=lambda h: -h["weight"])
    semis = sum(tgt.get(x, 0.0) for x in keep if in_semis_budget(E.sector.get(x, UNKNOWN)))
    unlimited = E.base_select([r["key"] for r in ranked], current, p.n, p.buffer)
    skipped = [f"{E.ticker[x]}(순위 {rank[x][0]}, {E.group(x)})" for x in unlimited if x not in keep]
    return {"session": NEXT_SESSION, "data_through": E.days[-1], "holdings": hold, "skipped_by_limit": skipped,
            "held_last_week": sorted(E.ticker[x] for x in current),
            "cash": round(1 - sum(tgt.values()), 4), "semis_budget_weight": round(semis, 4),
            "top12": [E.ticker[r["key"]] + f"({E.group(r['key'])})" for r in ranked[:12]]}


def main():
    E = SubEngine()
    plain = Engine()          # 고치지 않은 deskx 엔진(재현 확인용)
    # 1) 재현 확인: 데스크 run_model(라이브 50%, #774 60%) = deskx Engine = subcount m=7
    ref_live = np.asarray(run_model(E.book, start=START).value)
    with codex_caps():
        ref_774 = np.asarray(run_model(E.book, start=START).value)
    d_eng_live = float(np.max(np.abs(plain.run(Params(semis_cap=0.50))[0].to_numpy() - ref_live)))
    d_eng_774 = float(np.max(np.abs(plain.run(Params())[0].to_numpy() - ref_774)))
    live = one(E, *LIVE)
    c774 = one(E, *C774, live_log=live["_log"])
    d_sub_live = float(np.max(np.abs(live["_v"].to_numpy() - ref_live)))
    d_sub_774 = float(np.max(np.abs(c774["_v"].to_numpy() - ref_774)))
    checks = {"deskx_live50_vs_run_model": d_eng_live, "deskx_774_vs_run_model_codex_caps": d_eng_774,
              "subcount_m7_B050_vs_run_model": d_sub_live, "subcount_m7_B060_vs_run_model_codex_caps": d_sub_774}
    print("재현 확인(최대 절대차):", checks)
    assert max(checks.values()) < 1e-12, checks

    out = [one(E, name, p, live_log=live["_log"]) for name, p in GRID]
    post = one(E, *POSTHOC, live_log=live["_log"])

    best = max(out, key=lambda r: r[FRONT]["수익"])
    both = [r["name"] for r in out if r[FRONT]["수익"] > live[FRONT]["수익"] and r[BACK]["수익"] > live[BACK]["수익"]]
    mdd_ok = [r["name"] for r in out if r[FULL]["낙폭"] > live[FULL]["낙폭"] + 1e-3
              and r[FULL]["수익"] >= 0.9 * live[FULL]["수익"]]

    head = f"{'변형':<30}{'앞':>8}{'뒤':>8}{'9/1~':>8}{'전체':>8}{'MDD':>8}" \
           + "".join(f"{m[-2:] + '월':>7}" for m in MONTHS) \
           + f"{'반도체':>7}{'세부수':>6}{'최대세부':>8}{'상위3':>6}{'한도주':>6}{'live≠':>6}  상위3종목"
    print(head)
    for r in [live, c774] + out + [post]:
        print(f"{r['name']:<30}{r[FRONT]['수익']:>8.3f}{r[BACK]['수익']:>8.3f}{r[SEP]['수익']:>8.3f}"
              f"{r[FULL]['수익']:>8.3f}{r[FULL]['낙폭']:>8.3f}"
              + "".join(f"{r['월별'][m]:>7.3f}" for m in MONTHS)
              + f"{r['반도체평균']:>7.3f}{r['세부섹터수평균']:>6.2f}{r['최대세부섹터비중평균']:>8.3f}"
              + f"{r['상위3비중']:>6.2f}{r['한도작동주']:>6}{r.get('live와다른주', 0):>6}  {','.join(r['상위3'])}")
    print(f"\n앞 기준 최선: {best['name']} | 앞·뒤 모두 live 초과: {both or '없음'}"
          f" | 전체 MDD 개선 & 전체 수익 ≥ live의 90%: {mdd_ok or '없음'}")
    for r in out[::2]:
        print(f"{r['name'][:3]} 건너뛴 종목(주 수, 그룹, 앞에 있던 종목):",
              {k: (d['weeks'], d['group'], d['behind']) for k, d in r["건너뛴종목"].items()})
        print(f"{r['name'][:3]} 대신 들어온 종목(주 수):", r["대신들어온종목"])

    best_p = dict(GRID)[best["name"]]
    nxt = preview(E, best_p, best["_log"])
    nxt_live = preview(E, LIVE[1], live["_log"])
    print(f"\n{NEXT_SESSION} 재조정 미리보기({best['name']}, 자료 {nxt['data_through']}까지):")
    for h in nxt["holdings"]:
        print(f"  {h['name']:<6}{h['weight']:>7.3f}  {h['subsector']}  (순위 {h['rank']})")
    print(f"  현금 {nxt['cash']:.3f}, 반도체 예산 비중 {nxt['semis_budget_weight']:.3f}, 한도로 빠진 종목 {nxt['skipped_by_limit']}")
    print("  live 규칙이면:", [(h["name"], h["weight"]) for h in nxt_live["holdings"]], "현금", nxt_live["cash"])

    def clean(r):
        return {k: v for k, v in r.items() if not k.startswith("_")}

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"family": "subcount", "grid": len(GRID), "grid_spec": "m in {1,2} x semis budget B in {1.0, 0.60}",
                   "group_rule": "subsector_of(key,ticker); '미분류' = 종목마다 별도 그룹",
                   "checks": checks, "live": clean(live), "c774": clean(c774), "variants": [clean(r) for r in out],
                   "posthoc": clean(post), "best_by_front": best["name"], "beats_live_both_halves": both,
                   "lower_mdd_small_cost": mdd_ok, "next_rebalance_preview": nxt, "next_rebalance_live": nxt_live,
                   "weekly_log": {r["name"]: r["_log"] for r in [live] + out}},
                  f, ensure_ascii=False, indent=1, default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print("saved", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
