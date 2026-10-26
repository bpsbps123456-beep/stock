"""세부 섹터 비중 상한(subcap) 6개 변형 비교(2026-10-10, 연구용 스크래치). desk/·deskx.py는 고치지 않는다.

질문(사용자): 반도체 전체 50% 상한 대신 세부 섹터별로 나눠 분산하면 더 나은가?
규칙: 선택(TIME 미국 보유, 120일 모멘텀 상위 7, 버퍼 5, TIME NAV 비중)과 종목 25%·관심 섹터 60% 상한은 그대로.
      상한 투영만 바꾼다: desk.strategies.project_caps와 같은 비례 점진 채우기에
      '세부 섹터 그룹마다 장부의 X 이하' 제약 하나를 더한다(못 들어간 비중은 현금).
세부 섹터 그룹: desk.strategies.subsector_of(key, ticker). '미분류' 종목은 한 사업이 아니므로 종목마다 따로 한 그룹.
      AI_UNIVERSE에 없는 종목은 subsector_of가 관심 섹터 이름을 돌려주므로 그 섹터 이름이 그룹이 된다.
격자(사전 등록 6개): X ∈ {0.20, 0.25, 0.30} × 반도체+미분류 예산 ∈ {1.0(없앰), 0.60}.
사후 아이디어(1개, 선택에 쓰지 않음): X 0.25 + 현행 반도체 예산 0.50 유지.
선택: 앞 1/2~5/29 수익만으로 최선 변형을 고른다. 비교 기준은 현행 50%(사용자가 지금 쓰는 규칙)와 #774(60%).
분산 지표(매 거래일 시가 뒤 보유 기준 평균): 보유 세부 섹터 수, 가장 큰 세부 섹터 비중, 반도체 예산 비중, 투자 비중.
종목 기여: 각 시가 비중 × 그 종목 시가→다음 시가 수익 × 그날 계좌 가치. 상위 3 비중 = 상위 3 기여 합 / 전체 이익.
검증: X=1.0이면 새 투영 = project_caps(무작위 원비중 + 2026 모든 재조정 원비중), 경로도 현행·#774와 같아야 한다.
실행: code 폴더에서  PYTHONIOENCODING=utf-8 py deskx_subcap.py
"""
import json, os, random, sys
from dataclasses import asdict, dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deskx import Engine, Params, PERIODS, START  # noqa: E402
from desk import config  # noqa: E402
from desk.strategies import (UNKNOWN, in_semis_budget, project_caps, run_model, subsector_of,  # noqa: E402
                             iso_week)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/deskx/subcap.json")
FRONT, BACK, SEP, FULL = (p[0] for p in PERIODS)
NEXT_SESSION = "2026-10-12"
MONTHS = ["2026-06", "2026-07", "2026-08", "2026-09", "2026-10"]


@dataclass(frozen=True)
class SubParams(Params):
    sub_cap: float = 1.0          # 세부 섹터 그룹 상한(1.0 = 제약 없음과 같은 경로)


GRID = [(f"sub {x:.2f} / 반도체 {'없음' if s >= 1 else f'{s:.2f}'}", SubParams(sub_cap=x, semis_cap=s))
        for s in (1.0, 0.60) for x in (0.20, 0.25, 0.30)]
POSTHOC = ("[사후] sub 0.25 / 반도체 0.50", SubParams(sub_cap=0.25, semis_cap=0.50))


# ------------------------------------------------------------------ 투영
def project_caps_sub(raw, sector, sub, stock, detail, semis, subcap):
    """project_caps와 같은 비례 점진 채우기 + 세부 섹터 그룹 상한. 줄 순서까지 원본을 따른다."""
    names = [k for k, v in raw.items() if v > 0]
    group = {k: sector.get(k, UNKNOWN) for k in names}
    sgroup = {k: sub[k] for k in names}
    budgeted = {k for k in names if in_semis_budget(group[k])}
    alloc = dict.fromkeys(names, 0.0)
    target = sum(raw[k] for k in names)
    active = set(names)
    for _ in range(4 * len(names) + 8):
        used = sum(alloc.values())
        if target - used <= 1e-12:
            break
        in_group: dict[str, float] = {}
        in_sub: dict[str, float] = {}
        for k in names:
            in_group[group[k]] = in_group.get(group[k], 0.0) + alloc[k]
            in_sub[sgroup[k]] = in_sub.get(sgroup[k], 0.0) + alloc[k]
        in_semis = sum(alloc[k] for k in budgeted)
        active = {k for k in active if stock - alloc[k] > 1e-12 and detail - in_group[group[k]] > 1e-12
                  and not (k in budgeted and semis - in_semis <= 1e-12) and subcap - in_sub[sgroup[k]] > 1e-12}
        if not active:
            break
        steps = [(target - used) / sum(raw[k] for k in active)]
        steps += [(stock - alloc[k]) / raw[k] for k in active]
        for g in {group[k] for k in active}:
            steps.append((detail - in_group[g]) / sum(raw[k] for k in active if group[k] == g))
        semis_rate = sum(raw[k] for k in active if k in budgeted)
        if semis_rate > 0:
            steps.append((semis - in_semis) / semis_rate)
        for g in {sgroup[k] for k in active}:
            steps.append((subcap - in_sub[g]) / sum(raw[k] for k in active if sgroup[k] == g))
        step = max(0.0, min(steps))
        for k in active:
            alloc[k] += step * raw[k]
    return {k: v for k, v in alloc.items() if v > 1e-12}


def desk_project(raw, sector, stock, detail, semis):
    saved = dict(config.RULE)
    try:
        config.RULE.update(stock_cap=stock, sector_cap=detail, semis_cap=semis)
        return project_caps(raw, sector)
    finally:
        config.RULE.clear()
        config.RULE.update(saved)


def feasible_and_maximal(alloc, raw, sector, sub, stock, detail, semis, subcap, tol=1e-9):
    """모든 상한을 지키고, 더 키울 수 있는 종목이 남지 않았는지(투자 = 원비중 합이거나 각 종목이 어떤 상한에 막힘)."""
    names = [k for k, v in raw.items() if v > 0]
    a = {k: alloc.get(k, 0.0) for k in names}
    g, s = {}, {}
    for k in names:
        g[sector.get(k, UNKNOWN)] = g.get(sector.get(k, UNKNOWN), 0.0) + a[k]
        s[sub[k]] = s.get(sub[k], 0.0) + a[k]
    sem = sum(a[k] for k in names if in_semis_budget(sector.get(k, UNKNOWN)))
    ok = all(v <= stock + tol for v in a.values()) and all(v <= detail + tol for v in g.values()) \
        and sem <= semis + tol and all(v <= subcap + tol for v in s.values()) \
        and sum(a.values()) <= sum(raw[k] for k in names) + tol
    if sum(a.values()) < sum(raw[k] for k in names) - 1e-7:
        for k in names:
            blocked = a[k] >= stock - 1e-7 or g[sector.get(k, UNKNOWN)] >= detail - 1e-7 \
                or s[sub[k]] >= subcap - 1e-7 or (in_semis_budget(sector.get(k, UNKNOWN)) and sem >= semis - 1e-7)
            ok = ok and blocked
    return ok


# ------------------------------------------------------------------ 엔진
class SubEngine(Engine):
    """Engine과 같고 상한 투영 단계만 project_caps_sub로 바꾼다(선택·버퍼·비중 재료는 Engine.target 그대로)."""

    def __init__(self):
        super().__init__()
        self.sub = {}
        for k, tk in self.ticker.items():
            lab = subsector_of(k, tk)
            self.sub[k] = f"{UNKNOWN}:{tk}" if lab == UNKNOWN else lab
        self.raws = []            # (raw, params) 재조정마다 — 검증용
        self.last_keep = set()

    def target(self, p, k, session, current):
        keep, _, ranked = super().target(p, k, session, current)
        if not ranked:
            return keep, {}, ranked
        wmap = {r["key"]: r["w"] for r in ranked}
        tot = sum(wmap[x] for x in keep)
        raw = {x: wmap[x] / tot for x in keep}
        self.raws.append(raw)
        self.last_keep = set(keep)
        tgt = project_caps_sub(raw, self.sector, self.sub, p.stock_cap, p.sector_cap, p.semis_cap,
                               getattr(p, "sub_cap", 1.0))
        return keep, tgt, ranked


def label(E, key):
    lab = E.sub[key]
    return UNKNOWN if lab.startswith(UNKNOWN + ":") else lab


def contributions(E, v, rows):
    k0 = E.days.index(rows[0]["day"])
    vals = v.to_numpy()
    contrib: dict[str, float] = {}
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
    return {"상위3": [t for t, _ in top[:3]], "상위3비중": sum(c for _, c in top[:3]) / gain if gain else float("nan"),
            "상위5기여": {t: round(c, 4) for t, c in top[:5]}, "하위3기여": {t: round(c, 4) for t, c in top[-3:]}}


def diversification(E, rows):
    nsub, maxsub, semis, inv = [], [], [], []
    for r in rows:
        w = r["w"]
        by = {}
        for key, wt in w.items():
            by[E.sub[key]] = by.get(E.sub[key], 0.0) + wt
        nsub.append(len(by))
        maxsub.append(max(by.values()) if by else 0.0)
        semis.append(sum(wt for key, wt in w.items() if in_semis_budget(E.sector.get(key, UNKNOWN))))
        inv.append(sum(w.values()))
    return {"세부섹터수평균": float(np.mean(nsub)), "최대세부섹터비중평균": float(np.mean(maxsub)),
            "반도체예산비중평균": float(np.mean(semis)), "투자비중평균": float(np.mean(inv)),
            "투자비중최저": float(np.min(inv))}


def one(E, name, p):
    v, rows = E.run(p)
    res = E.evaluate(name, v, rows)
    res["params"] = asdict(p)
    res.update(diversification(E, rows))
    res.update(contributions(E, v, rows))
    res["_v"] = v
    return res


def preview(E, p, last_keep):
    k = len(E.days)
    assert iso_week(NEXT_SESSION) != iso_week(E.days[-1])
    keep, tgt, ranked = E.target(p, k, NEXT_SESSION, set(last_keep))
    rank = {r["key"]: i + 1 for i, r in enumerate(ranked)}
    rows = [{"ticker": E.ticker[x], "weight": round(tgt.get(x, 0.0), 4), "sub": label(E, x), "sector": E.sector[x],
             "rank": rank[x], "mom120": round(next(r["mom"] for r in ranked if r["key"] == x), 4)}
            for x in sorted(keep, key=lambda x: -tgt.get(x, 0.0))]
    return {"session": NEXT_SESSION, "holdings": rows, "cash": round(1 - sum(tgt.values()), 4),
            "data_through": E.days[-1]}


# ------------------------------------------------------------------ 검증
def verify(E):
    out = {}
    # 1) deskx 엔진 = 데스크 run_model(현행 50%, #774 60%)
    from desk.validate import codex_caps
    E0 = Engine()                                   # 원래 엔진(project_caps)
    v50, _ = E0.run(Params(semis_cap=0.50))
    out["deskx_live50_vs_run_model"] = float(np.max(np.abs(v50.to_numpy() - run_model(E.book, start=START).value)))
    v60, _ = E0.run(Params())
    with codex_caps():
        ref = run_model(E.book, start=START).value
    out["deskx_774_vs_run_model"] = float(np.max(np.abs(v60.to_numpy() - ref)))
    # 2) X=1.0이면 SubEngine 경로 = 원 경로
    s50, _ = E.run(SubParams(sub_cap=1.0, semis_cap=0.50))
    s60, _ = E.run(SubParams(sub_cap=1.0, semis_cap=0.60))
    out["sub1.0_live50_path_maxdiff"] = float(np.max(np.abs(s50.to_numpy() - v50.to_numpy())))
    out["sub1.0_774_path_maxdiff"] = float(np.max(np.abs(s60.to_numpy() - v60.to_numpy())))
    # 3) 투영 성질: 무작위 원비중 + 2026 재조정 원비중
    rng = random.Random(20261010)
    keys = sorted(E.ticker)
    sets = []
    for _ in range(3000):
        n = rng.randint(1, 12)
        ks = rng.sample(keys, n)
        a = rng.choice([0.3, 1.0, 3.0])
        w = [rng.gammavariate(a, 1.0) for _ in ks]
        tot = sum(w)
        sets.append({x: y / tot for x, y in zip(ks, w)})
    rebal = list(E.raws)
    exact = worst = 0
    count = 0
    for raw in sets + rebal:
        for stock, detail, semis in ((0.25, 0.60, 0.50), (0.25, 0.60, 0.60), (0.25, 0.60, 1.0),
                                     (rng.uniform(0.1, 0.5), rng.uniform(0.2, 1.0), rng.uniform(0.2, 1.0))):
            a = desk_project(raw, E.sector, stock, detail, semis)
            b = project_caps_sub(raw, E.sector, E.sub, stock, detail, semis, 1.0)
            d = max(abs(a.get(x, 0.0) - b.get(x, 0.0)) for x in set(a) | set(b)) if (a or b) else 0.0
            exact += (a == b)
            worst = max(worst, d)
            count += 1
    out["x1_property_cases"] = count
    out["x1_random_sets"] = len(sets)
    out["x1_rebalance_sets"] = len(rebal)
    out["x1_exact_equal"] = exact
    out["x1_maxdiff"] = worst
    # 4) X<1 투영이 모든 상한을 지키고 막히지 않은 종목을 남기지 않는지
    bad = 0
    n = 0
    for raw in sets + rebal:
        for x in (0.15, 0.20, 0.25, 0.30, 0.40):
            for semis in (1.0, 0.60, 0.50):
                t = project_caps_sub(raw, E.sector, E.sub, 0.25, 0.60, semis, x)
                bad += not feasible_and_maximal(t, raw, E.sector, E.sub, 0.25, 0.60, semis, x)
                n += 1
    out["subcap_feasible_maximal_cases"] = n
    out["subcap_feasible_maximal_failures"] = bad
    return out


def main():
    E = SubEngine()
    live = one(E, "현행 50%", SubParams(semis_cap=0.50))
    base = one(E, "#774 60%", SubParams())
    out = [one(E, name, p) for name, p in GRID]
    post = one(E, *POSTHOC)
    ver = verify(E)

    best = max(out, key=lambda r: r[FRONT]["수익"])
    both = [r["name"] for r in out if r[FRONT]["수익"] > live[FRONT]["수익"] and r[BACK]["수익"] > live[BACK]["수익"]]
    both774 = [r["name"] for r in out if r[FRONT]["수익"] > base[FRONT]["수익"] and r[BACK]["수익"] > base[BACK]["수익"]]
    mdd_ok = [r["name"] for r in out if r[FULL]["낙폭"] > live[FULL]["낙폭"] + 1e-3
              and r[FULL]["수익"] >= 0.9 * live[FULL]["수익"]]

    # 앞으로 고른 변형의 2026-10-12 재조정 미리보기(+ 비교용 현행 50%)
    E.run(SubParams(**best["params"]))
    prev_best = preview(E, SubParams(**best["params"]), E.last_keep)
    E.run(SubParams(semis_cap=0.50))
    prev_live = preview(E, SubParams(semis_cap=0.50), E.last_keep)

    head = f"{'변형':<22}{'앞':>8}{'뒤':>8}{'9/1~':>8}{'전체':>8}{'MDD':>8}" \
           + "".join(f"{m[-2:] + '월':>7}" for m in MONTHS) + f"{'세부수':>6}{'최대세부':>8}{'반도체':>7}{'투자':>6}{'상위3':>7}  상위3"
    print(head)
    for r in [live, base] + out + [post]:
        print(f"{r['name']:<22}{r[FRONT]['수익']:>8.3f}{r[BACK]['수익']:>8.3f}{r[SEP]['수익']:>8.3f}"
              f"{r[FULL]['수익']:>8.3f}{r[FULL]['낙폭']:>8.3f}"
              + "".join(f"{r['월별'][m]:>7.3f}" for m in MONTHS)
              + f"{r['세부섹터수평균']:>6.2f}{r['최대세부섹터비중평균']:>8.3f}{r['반도체예산비중평균']:>7.3f}"
                f"{r['투자비중평균']:>6.2f}{r['상위3비중']:>7.2f}  {','.join(r['상위3'])}")
    print("\n앞으로 고른 변형:", best["name"])
    print("현행보다 앞·뒤 모두 높음:", both or "없음", "| #774보다:", both774 or "없음")
    print("현행보다 MDD 낮고 전체 손실 10% 이내:", mdd_ok or "없음")
    print("검증:", json.dumps(ver, ensure_ascii=False))
    print(f"\n{NEXT_SESSION} 재조정 미리보기 ({best['name']}), 현금 {prev_best['cash']}")
    for h in prev_best["holdings"]:
        print(f"  {h['ticker']:<6}{h['weight']:>7.3f}  {h['sub']}  (순위 {h['rank']}, 120일 {h['mom120']:+.2f})")
    print(f"비교: 현행 50%, 현금 {prev_live['cash']}")
    for h in prev_live["holdings"]:
        print(f"  {h['ticker']:<6}{h['weight']:>7.3f}  {h['sub']}")

    for r in [live, base] + out + [post]:
        r.pop("_v")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"family": "subcap", "grid": [n for n, _ in GRID], "posthoc": POSTHOC[0],
                   "subsector_groups": {E.ticker[k]: E.sub[k] for k in sorted(E.sub, key=lambda k: E.ticker[k])},
                   "live50": live, "codex774": base, "variants": out, "posthoc_result": post,
                   "best_by_front": best["name"], "beats_live_both": both, "beats_774_both": both774,
                   "lower_mdd_small_cost_vs_live": mdd_ok, "verify": ver,
                   "next_rebalance_preview": {"chosen": prev_best, "live50": prev_live}},
                  f, ensure_ascii=False, indent=1)
    print("저장:", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
