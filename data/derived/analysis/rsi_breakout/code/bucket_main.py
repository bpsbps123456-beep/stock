"""버킷 분산 모멘텀(2026-10-10, 연구용 스크래치). desk/·gbrc/·docs/·watchlist.py와 기존 연구 파일은 고치지 않는다.

질문(사용자): 반도체에만 몰리지 말고 AI 경제의 여러 갈래(코인·냉각·AI 서비스·광섬유·전력·칩·메모리 …)에
             3~10종목으로 나눠 담으면 수익이 오르는가? 손절 연구(1,371개 변형, 모두 취약)를 참고해 손절은 넣지 않는다.

규칙(사전 등록, 결과를 본 뒤 바꾸지 않음)
  매주 ISO 주 첫 미국 거래일 시가에 결정. 그 거래일보다 엄격히 앞선 날짜의 PDF, 전 거래일 종가까지만 쓴다(deskx와 같음).
  후보 U: 'T' TIME 미국 보유 | 'TK' TIME ∪ KoAct 미국 보유(KoAct는 2026년만).
  버킷 G: 'sector' watchlist 섹터(sector_of), {AI 반도체, 반도체(적자)}→'반도체 칩',
                  {반도체 장비, 반도체 장비(적자), 반도체 소재}→'반도체 장비·소재'로 합침.
          'sub'    subsector_of(AI_UNIVERSE 소분류, AI 밖 종목은 watchlist 섹터).
          둘 다 '미분류'는 종목마다 따로 한 버킷.
  모멘텀 = 120거래일 수정종가 수익(전 거래일 종가까지). 버킷 대장 = 버킷 안 모멘텀 1위. 대장을 모멘텀 순으로 줄 세운다.
  선택(N종목, 버킷당 1개): 지난주 종목은 그 버킷 대장 순위가 N+5 안이고 버킷 안 모멘텀 2위 안이면 유지(한 버킷에 둘이면 높은 쪽),
        나머지는 대장 줄 위에서부터 채우되 이미 든 버킷은 건너뛴다.
  비중 W: 'eq' 1/N씩 | 'fund' 펀드 NAV 비중 비례(TIME 비중, KoAct 전용 종목은 KoAct 비중).
  상한: 종목 ≤ max(0.25, 1/N), watchlist 섹터(합치지 않은 원래 섹터, 데스크와 같이 '미분류'는 한 섹터) ≤ 60%,
        반도체 예산 없음(버킷 분산이 위험 관리). 어디에도 안 들어가는 비중은 현금. 손절 없음.
  N ∈ {3,5,7,10}. 격자 = U 2 × G 2 × N 4 × W 2 = 32개.
선택: 앞 1/2~5/29 수익 최고 1개를 고르고 뒤 6/1~10/9로 확인한다. 2025 보조 확인은 이 자료(TIME PDF 2026-01-02부터,
      가격 2025-06-02부터)로는 불가.
사후 점검(앞 기준 최선 1개에만, 선택 근거 아님): (a) 반도체+미분류 50% 예산 추가, (b) 결정 종가에서 SOXX 종가 < SMA20이면
      그 주 반도체류(반도체 예산 섹터+미분류) 비중을 현금으로(선택·버퍼 상태는 그대로), (c) 모멘텀 기간 90·150.
      150은 가격이 2025-06-02부터라 2026-01-07에 시작(앞 3거래일 현금).
SOXX: stop_research/cache/ohlc.parquet(SOXX 종가, 읽기만). 기준: 지금 데스크 규칙(반도체 50%), #774(60%), QQQ 시가 경로.
재현 확인: deskx 기본 = run_model, 그리고 이 엔진에서 '종목마다 한 버킷'(gran='name') + fund + N7 + 예산 50%/60% = 데스크와 같아야 한다.
실행: cd data/derived/analysis/rsi_breakout/code && PYTHONIOENCODING=utf-8 py bucket_main.py
"""
import json, os, sys
from collections import Counter
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from deskx import Engine, Params, PERIODS, START, ROOT  # noqa: E402
from desk import config  # noqa: E402
from desk.strategies import project_caps, run_model, subsector_of, in_semis_budget, UNKNOWN  # noqa: E402
from desk.validate import codex_caps  # noqa: E402

OUT = os.path.join(HERE, "../results/bucket/main.json")
SOXX_SRC = os.path.join(ROOT, "data/derived/analysis/stop_research/cache/ohlc.parquet")
FRONT, BACK, SEP, FULL = (p[0] for p in PERIODS)
MONTHS = ["2026-06", "2026-07", "2026-08", "2026-09", "2026-10"]
NEXT_SESSION = "2026-10-12"
MERGE = {"AI 반도체": "반도체 칩", "반도체(적자)": "반도체 칩",
         "반도체 장비": "반도체 장비·소재", "반도체 장비(적자)": "반도체 장비·소재", "반도체 소재": "반도체 장비·소재"}


@dataclass(frozen=True)
class BParams(Params):
    gran: str = "sector"        # sector | sub | name(재현 확인 전용: 종목마다 한 버킷)
    wmode: str = "fund"         # eq | fund
    semis_cap: float = 1.0      # 1.0 = 반도체 예산 없음
    soxx_gate: bool = False


def vname(u, g, n, w):
    return f"{u}-{g}-N{n}-{w}"


GRID = [(vname(u, g, n, w), BParams(universe=uni, gran=g, n=n, wmode=w))
        for u, uni in (("T", "time"), ("TK", "union")) for g in ("sector", "sub") for n in (3, 5, 7, 10)
        for w in ("eq", "fund")]


class BucketEngine(Engine):
    def __init__(self):
        super().__init__()
        o = pd.read_parquet(SOXX_SRC, columns=["date", "sym", "Close"])
        o = o[o.sym == "SOXX"].copy()
        o["date"] = pd.to_datetime(o["date"]).dt.strftime("%Y-%m-%d")
        s = o.set_index("date")["Close"].astype(float).sort_index()
        s = s[~s.index.duplicated()]
        self.soxx = s
        self.soxx_sma20 = s.rolling(20, min_periods=20).mean()
        self._sub = {}
        self.log = None

    def bucket(self, key, gran):
        if gran == "name":
            return key
        if gran == "sector":
            s = self.sector.get(key, UNKNOWN)
            s = MERGE.get(s, s)
        else:
            if key not in self._sub:
                self._sub[key] = subsector_of(key, self.ticker[key])
            s = self._sub[key]
        return f"미분류:{self.ticker[key]}" if s == UNKNOWN else s

    def soxx_below(self, k):
        """결정 종가(k-1일)에서 SOXX 종가 < SMA20인가. 그날 SOXX 값이 없으면 그 전 마지막 값."""
        day = self.days[k - 1]
        s = self.soxx[self.soxx.index <= day]
        m = self.soxx_sma20[self.soxx_sma20.index <= day]
        if not len(s) or not np.isfinite(m.iloc[-1]):
            return False, None
        return bool(s.iloc[-1] < m.iloc[-1]), (s.index[-1], float(s.iloc[-1]), float(m.iloc[-1]))

    def select(self, ranked, current, p):
        order = [r["key"] for r in ranked]                    # 모멘텀 순(동률은 키 순)
        members = {}
        for x in order:
            members.setdefault(self.bucket(x, p.gran), []).append(x)
        leaders = list(members)                               # 처음 나온 순 = 대장 모멘텀 순
        lrank = {b: i + 1 for i, b in enumerate(leaders)}
        keep, held = [], set()
        for x in order:                                       # 모멘텀 순으로 돌아 한 버킷에 둘이면 높은 쪽이 남음
            if x not in current:
                continue
            b = self.bucket(x, p.gran)
            if lrank[b] <= p.n + p.buffer and members[b].index(x) < 2 and b not in held and len(keep) < p.n:
                keep.append(x)
                held.add(b)
        kept = list(keep)
        for b in leaders:
            if len(keep) >= p.n:
                break
            if b not in held:
                keep.append(members[b][0])
                held.add(b)
        return keep, kept, lrank, members

    def target(self, p, k, session, current):
        ranked = self.ranking(replace(p, weight="time"), k, session)
        if not ranked:
            return [], {}, ranked
        keep, kept, lrank, members = self.select(ranked, current, p)
        if p.wmode == "eq":
            raw = {x: 1.0 / p.n for x in keep}
        else:
            wmap = {r["key"]: r["w"] for r in ranked}
            tot = sum(wmap[x] for x in keep)
            raw = {x: wmap[x] / tot for x in keep}
        saved = dict(config.RULE)
        try:
            config.RULE.update(stock_cap=max(0.25, 1.0 / p.n), sector_cap=p.sector_cap, semis_cap=p.semis_cap)
            tgt = project_caps(raw, self.sector)
        finally:
            config.RULE.clear()
            config.RULE.update(saved)
        gate, gate_info = (False, None)
        if p.soxx_gate:
            gate, gate_info = self.soxx_below(k)
            if gate:
                tgt = {x: w for x, w in tgt.items() if not in_semis_budget(self.sector.get(x, UNKNOWN))}
        if self.log is not None:
            self.log.append({"session": session, "keep": list(keep), "kept": list(kept), "gate": gate,
                             "gate_info": gate_info, "n_buckets_avail": len(lrank)})
        return keep, tgt, ranked

    def run_logged(self, p):
        self.log = []
        try:
            v, rows = self.run(p)
            log = self.log
        finally:
            self.log = None
        return v, rows, log


# ---------------------------------------------------------------- 지표
def open_ret(E, key, k):
    o = E.col("opn", key)
    return (o[k] / o[k - 1] - 1) if o is not None and np.isfinite(o[k]) and np.isfinite(o[k - 1]) and o[k - 1] > 0 \
        else 0.0


def contributions(E, v, rows, label):
    """각 시가 비중 × 그 종목 시가→다음 시가 수익 × 그날 계좌 가치(시작 1). 합 = 최종 가치 - 1."""
    k0 = E.days.index(rows[0]["day"])
    vals = v.to_numpy()
    by_name, by_bucket = {}, {}
    for i in range(len(rows) - 1):
        k = k0 + i + 1
        for key, wt in rows[i]["w"].items():
            c = vals[i] * wt * open_ret(E, key, k)
            by_name[E.ticker[key]] = by_name.get(E.ticker[key], 0.0) + c
            b = label(key)
            by_bucket[b] = by_bucket.get(b, 0.0) + c
    gain = float(vals[-1] - 1)
    assert abs(sum(by_name.values()) - gain) < 1e-9 * max(1.0, abs(gain))
    top = sorted(by_name.items(), key=lambda x: -x[1])
    topb = sorted(by_bucket.items(), key=lambda x: -x[1])
    return {"top3": [t for t, _ in top[:3]], "top3_share": sum(c for _, c in top[:3]) / gain if gain else float("nan"),
            "top5_contrib": {t: round(c, 4) for t, c in top[:5]},
            "bottom3_contrib": {t: round(c, 4) for t, c in top[-3:]},
            "bucket_contrib": {b: round(c, 4) for b, c in topb}}


def structure(E, rows, log, gran):
    nb, ns, inv, unk = [], [], [], []
    for row in rows:
        w = {x: wt for x, wt in row["w"].items() if wt > 1e-12}
        nb.append(len({E.bucket(x, gran) for x in w}))
        ns.append(len({E.sector.get(x, UNKNOWN) if E.sector.get(x, UNKNOWN) != UNKNOWN else "미분류:" + x for x in w}))
        inv.append(sum(w.values()))
        unk.append(sum(wt for x, wt in w.items() if E.sector.get(x, UNKNOWN) == UNKNOWN))
    # 회전: 주간 재조정마다 새로 들어온 종목 수(선택 기준), 비중 회전 = 0.5 Σ|새 비중 - 재조정 직전 흘러간 비중|
    k0 = E.days.index(rows[0]["day"])
    day_i = {r["day"]: i for i, r in enumerate(rows)}
    names_in, wturn = [], []
    for a, b in zip(log, log[1:]):
        names_in.append(len(set(b["keep"]) - set(a["keep"])))
        i = day_i.get(b["session"])
        if i is None or i == 0:
            continue
        k = k0 + i
        prev = rows[i - 1]["w"]
        rets = {x: open_ret(E, x, k) for x in prev}
        total = sum(wt * (1 + rets[x]) for x, wt in prev.items()) + (1 - sum(prev.values()))
        pre = {x: wt * (1 + rets[x]) / total for x, wt in prev.items()}
        new = rows[i]["w"]
        wturn.append(0.5 * (sum(abs(new.get(x, 0.0) - pre.get(x, 0.0)) for x in set(new) | set(pre))
                            + abs((1 - sum(new.values())) - (1 - sum(pre.values())))))
    return {"buckets_avg": float(np.mean(nb)), "sectors_avg": float(np.mean(ns)), "invested_avg": float(np.mean(inv)),
            "unclassified_avg": float(np.mean(unk)), "turnover_names": float(np.mean(names_in)) if names_in else 0.0,
            "turnover_weight": float(np.mean(wturn)) if wturn else 0.0, "rebalances": len(log),
            "gate_weeks": sum(1 for e in log if e["gate"])}


def one(E, name, p):
    v, rows, log = E.run_logged(p)
    res = E.evaluate(name, v, rows)
    res["params"] = asdict(p)
    res.update(structure(E, rows, log, p.gran))
    res["short_weeks"] = sum(1 for e in log if len(e["keep"]) < p.n)
    res.update(contributions(E, v, rows, lambda x: E.bucket(x, p.gran)))
    res["started"] = next((r["day"] for r in rows if r["w"]), None)
    res["bucket_weeks"] = dict(Counter(E.bucket(x, "sub") for e in log for x in e["keep"]).most_common())
    res["_v"], res["_log"], res["_rows"] = v, log, rows
    return res


def flat(r):
    return {"name": r["name"], "front": r[FRONT]["수익"], "back": r[BACK]["수익"], "sepoct": r[SEP]["수익"],
            "full": r[FULL]["수익"], "mdd": r[FULL]["낙폭"], "months": {m: r["월별"][m] for m in MONTHS},
            "buckets_avg": r.get("buckets_avg"), "semis_avg": r.get("반도체평균"), "top3_share": r.get("top3_share"),
            "turnover": r.get("turnover_names"), "top3": r.get("top3")}


def preview(E, p, log):
    current = set(log[-1]["keep"])
    k = len(E.days)
    keep, tgt, ranked = E.target(p, k, NEXT_SESSION, current)
    rank = {r["key"]: (i + 1, r["mom"], r["w"]) for i, r in enumerate(ranked)}
    _, kept, lrank, members = E.select(ranked, current, p)
    hold = [{"name": E.ticker[x], "weight": round(tgt.get(x, 0.0), 4), "bucket": E.bucket(x, p.gran),
             "sector": E.sector.get(x, UNKNOWN), "bucket_rank": lrank[E.bucket(x, p.gran)],
             "rank_in_bucket": members[E.bucket(x, p.gran)].index(x) + 1, "overall_rank": rank[x][0],
             "mom": round(rank[x][1], 4), "fund_weight": round(rank[x][2], 4), "held_last_week": x in current,
             "semis_like": in_semis_budget(E.sector.get(x, UNKNOWN))} for x in keep]
    hold.sort(key=lambda h: -h["weight"])
    gate, info = E.soxx_below(k)
    return {"session": NEXT_SESSION, "data_through": E.days[-1], "holdings": hold,
            "held_last_week": sorted(E.ticker[x] for x in current),
            "dropped": sorted(E.ticker[x] for x in current - set(keep)),
            "cash": round(1 - sum(tgt.values()), 4),
            "semis_like_weight": round(sum(h["weight"] for h in hold if h["semis_like"]), 4),
            "soxx_below_sma20": gate, "soxx_info": info,
            "leaders_top12": [f"{E.ticker[members[b][0]]}({b})" for b in list(members)[:12]]}


def main():
    E = BucketEngine()
    # 1) 재현 확인
    ref_live = np.asarray(run_model(E.book, start=START).value)
    with codex_caps():
        ref_774 = np.asarray(run_model(E.book, start=START).value)
    plain = Engine()                      # 고치지 않은 deskx 엔진(재현 확인용)
    v_live, _ = plain.run(Params(semis_cap=0.50))
    v_774, _ = plain.run(Params(semis_cap=0.60))
    live_b = one(E, "live 50% (gran=name 재현)", BParams(gran="name", wmode="fund", n=7, semis_cap=0.50))
    c774_b = one(E, "#774 60% (gran=name 재현)", BParams(gran="name", wmode="fund", n=7, semis_cap=0.60))
    checks = {"deskx_live50_vs_run_model": float(np.max(np.abs(v_live.to_numpy() - ref_live))),
              "deskx_774_vs_run_model": float(np.max(np.abs(v_774.to_numpy() - ref_774))),
              "bucket_name_live50_vs_run_model": float(np.max(np.abs(live_b["_v"].to_numpy() - ref_live))),
              "bucket_name_774_vs_run_model": float(np.max(np.abs(c774_b["_v"].to_numpy() - ref_774)))}
    print("재현 확인(최대 절대차):", checks)
    assert max(checks.values()) < 1e-12, checks
    live, c774 = live_b, c774_b
    live["name"], c774["name"] = "live 데스크(반도체 50%)", "#774(반도체 60%)"
    # 기준 지표를 버킷 엔진 기준(섹터 버킷)으로도 셈: 라이브가 섹터 버킷 몇 개를 들었나
    live_sector_buckets = structure(E, live["_rows"], live["_log"], "sector")["buckets_avg"]
    live_sub_buckets = structure(E, live["_rows"], live["_log"], "sub")["buckets_avg"]

    q = E.book.opn["QQQ"].to_numpy(float)
    days = [d for d in E.days if d >= START]
    k0 = E.days.index(days[0])
    qqq = E.evaluate("QQQ", pd.Series(q[k0:] / q[k0], index=days))

    # 2) 격자 32개
    out = [one(E, name, p) for name, p in GRID]
    best = max(out, key=lambda r: r[FRONT]["수익"])
    both = [r["name"] for r in out if r[FRONT]["수익"] > live[FRONT]["수익"] and r[BACK]["수익"] > live[BACK]["수익"]]
    tot_mdd = [r["name"] for r in out if r[FULL]["수익"] > live[FULL]["수익"] and r[FULL]["낙폭"] > live[FULL]["낙폭"]]
    rank_front = sorted(out, key=lambda r: -r[FRONT]["수익"])
    back_rank_of_best = 1 + sorted(out, key=lambda r: -r[BACK]["수익"]).index(best)

    # 3) 사후 점검(앞 기준 최선에만)
    bp = dict(GRID)[best["name"]]
    post = [one(E, f"[사후a] {best['name']} +반도체·미분류 50% 예산", replace(bp, semis_cap=0.50)),
            one(E, f"[사후b] {best['name']} +SOXX<SMA20 주 반도체류 현금", replace(bp, soxx_gate=True)),
            one(E, f"[사후c] {best['name']} 모멘텀 90일", replace(bp, lookback=90)),
            one(E, f"[사후c] {best['name']} 모멘텀 150일", replace(bp, lookback=150)),
            # 대조(선택 근거 아님): (b)의 개선이 버킷 때문인지 게이트 때문인지 가르기 위해 같은 게이트를 live 규칙에 얹음
            one(E, "[대조] live 50% +SOXX<SMA20 주 반도체류 현금",
                BParams(gran="name", wmode="fund", n=7, semis_cap=0.50, soxx_gate=True))]

    # 4) 표
    head = f"{'변형':<44}{'앞':>8}{'뒤':>8}{'9/1~':>8}{'전체':>8}{'MDD':>8}" \
           + "".join(f"{m[-2:] + '월':>7}" for m in MONTHS) \
           + f"{'버킷':>6}{'반도체류':>8}{'상위3':>6}{'회전':>6}{'비중회전':>8}  상위3종목"

    def line(r):
        return (f"{r['name']:<44}{r[FRONT]['수익']:>8.3f}{r[BACK]['수익']:>8.3f}{r[SEP]['수익']:>8.3f}"
                f"{r[FULL]['수익']:>8.3f}{r[FULL]['낙폭']:>8.3f}"
                + "".join(f"{r['월별'][m]:>7.3f}" for m in MONTHS)
                + (f"{r['buckets_avg']:>6.2f}{r['반도체평균']:>8.3f}{r['top3_share']:>6.2f}{r['turnover_names']:>6.2f}"
                   f"{r['turnover_weight']:>8.3f}  {','.join(r['top3'])}" if "buckets_avg" in r else ""))

    print(head)
    for r in [live, c774, qqq]:
        print(line(r))
    print("-" * 40)
    for r in out:
        print(line(r) + ("   ◀ 앞 최선" if r is best else ""))
    print("-" * 40)
    for r in post:
        print(line(r) + (f"  (시작 {r['started']}, 게이트 주 {r['gate_weeks']}/{r['rebalances']})"))
    print(f"\nlive가 든 버킷 수 평균: 섹터 버킷 {live_sector_buckets:.2f}, 세부 버킷 {live_sub_buckets:.2f}")
    print(f"앞 기준 최선: {best['name']} (앞 {best[FRONT]['수익']:.3f}, 뒤 {best[BACK]['수익']:.3f}, 뒤 순위 {back_rank_of_best}/32)")
    print("앞 상위 5:", [(r["name"], round(r[FRONT]["수익"], 3), round(r[BACK]["수익"], 3)) for r in rank_front[:5]])
    print(f"앞·뒤 모두 live 초과: {both or '없음'}")
    print(f"전체 수익 > live & MDD 얕음: {tot_mdd or '없음'}")
    print("앞 최선 버킷 기여:", best["bucket_contrib"])
    for nm in ("T-sector-N7-eq", "TK-sub-N10-eq", best["name"]):
        r = next(x for x in out if x["name"] == nm)
        print(f"{nm} 세부 버킷별 보유 주 수(주 {r['rebalances']}):", r["bucket_weeks"])
    print("자리 못 채운 주가 있는 변형:", {r["name"]: r["short_weeks"] for r in out + post if r["short_weeks"]} or "없음")

    nxt = preview(E, bp, best["_log"])
    nxt_live = preview(E, BParams(gran="name", wmode="fund", n=7, semis_cap=0.50), live["_log"])
    print(f"\n{NEXT_SESSION} 재조정 미리보기({best['name']}, 자료 {nxt['data_through']}까지):")
    for h in nxt["holdings"]:
        print(f"  {h['name']:<6}{h['weight']:>7.3f}  {h['bucket']}  (버킷 순위 {h['bucket_rank']}, 버킷 안 {h['rank_in_bucket']}위,"
              f" 120일 {h['mom']:+.3f}, 지난주 보유 {h['held_last_week']})")
    print(f"  현금 {nxt['cash']:.3f}, 반도체류 {nxt['semis_like_weight']:.3f}, 빠지는 종목 {nxt['dropped']},"
          f" SOXX<SMA20 {nxt['soxx_below_sma20']} {nxt['soxx_info']}")
    print("  live 규칙이면:", [(h["name"], h["weight"]) for h in nxt_live["holdings"]], "현금", nxt_live["cash"])

    def clean(r):
        return {k: v for k, v in r.items() if not k.startswith("_")}

    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"family": "bucket_diversified_momentum", "grid": len(GRID),
                   "grid_spec": "U{T,TK} x G{sector,sub} x N{3,5,7,10} x W{eq,fund} = 32; posthoc 4 on front-pick",
                   "merge": MERGE, "checks": checks, "benchmark": {"live": flat(live), "c774": flat(c774),
                                                                   "qqq": flat(qqq)},
                   "live_buckets_avg": {"sector": live_sector_buckets, "sub": live_sub_buckets},
                   "variants": [flat(r) for r in out], "best_by_front": best["name"],
                   "best_back_rank": back_rank_of_best,
                   "beats_live_both_halves": both, "higher_total_lower_mdd": tot_mdd,
                   "posthoc": [flat(r) | {"started": r["started"], "gate_weeks": r["gate_weeks"]} for r in post],
                   "detail": {r["name"]: clean(r) for r in [live, c774] + out + post},
                   "next_rebalance_preview": nxt, "next_rebalance_live": nxt_live,
                   "weekly_log": {r["name"]: r["_log"] for r in [best] + post}},
                  f, ensure_ascii=False, indent=1, default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print("saved", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
