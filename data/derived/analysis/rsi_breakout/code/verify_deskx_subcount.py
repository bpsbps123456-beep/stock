"""subcount 계열 감사(2026-10-10, 연구용 스크래치). 빌더 코드(deskx_subcount.SubEngine)는 대조 상대로만 쓴다.

1) 재현: desk run_model(라이브 50%, codex_caps #774) = deskx Engine. 독립 구현 = desk.strategies.run_model에
   따로 짠 '그룹 한도 select'를 끼워 돌린 경로(다른 경로 엔진). m=1·2 × B=1.0·0.60과 m=7을 빌더 값과 대조.
2) 40개 재조정 전부에서 규칙 확인(버퍼 유지·그룹 한도·채우기 건너뜀·상한·비례·현금), 3개 날짜는 표로 출력.
3) 견고성: 월별(1~10월), 앞 구간 초과의 하루 쏠림, 종목별 기여 차, DELL 5/26~5/29 주,
   이웃 값(감사용, 후보 아님) m∈{1,2,3} × B∈{0.50,0.55,0.60,0.65,0.70,1.0}, 시작일 이동.
실행: py verify_deskx_subcount.py (code 폴더, PYTHONIOENCODING=utf-8)
"""
import os, sys
from collections import Counter, defaultdict
from contextlib import contextmanager

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from deskx import Engine, Params, START  # noqa: E402
from deskx_subcount import SubEngine, SubParams  # noqa: E402  (대조 상대)
from desk import config  # noqa: E402
from desk import strategies as S  # noqa: E402
from desk.validate import codex_caps  # noqa: E402

FRONT = ("2026-01-02", "2026-05-29")
BACK = ("2026-06-01", "2026-10-09")
SEP = ("2026-09-01", "2026-10-09")
FULL = (START, "2026-10-09")
MONTHS = [f"2026-{m:02d}" for m in range(1, 11)]

E = SubEngine()
book = E.book


def grp(key):
    s = S.subsector_of(key, book.ticker[key])
    return ("미분류", key) if s == S.UNKNOWN else s


def gname(g):
    return f"미분류:{book.ticker[g[1]]}" if isinstance(g, tuple) else g


LOG = []


def make_select(m):
    """독립 구현: 버퍼 안 보유 종목을 그룹별로 묶어 순위 상위 m개만 남기고, 위에서부터 채우며 찬 그룹은 건너뛴다."""
    def sel(ranked, current):
        n, buf = config.RULE["n"], config.RULE["buffer"]
        order = [r["key"] for r in ranked]
        rank = {k: i for i, k in enumerate(order)}
        held = [k for k in order[:n + buf] if k in current]
        by_g = defaultdict(list)
        for k in held:
            by_g[grp(k)].append(k)
        kept = sorted((k for ks in by_g.values() for k in sorted(ks, key=rank.get)[:m]), key=rank.get)[:n]
        used = Counter(grp(k) for k in kept)
        for k in order:
            if len(kept) >= n:
                break
            if k in kept or used[grp(k)] >= m:
                continue
            kept.append(k)
            used[grp(k)] += 1
        # 한도 없는 원래 선택(같은 current)
        base = [k for k in order[:n + buf] if k in current][:n]
        for k in order:
            if len(base) >= n:
                break
            if k not in base:
                base.append(k)
        LOG.append({"ranked": ranked, "current": set(current), "kept": list(kept), "base": base})
        return kept
    return sel


@contextmanager
def rule(m, B):
    saved_rule, saved_sel = dict(config.RULE), S.select
    config.RULE.update(semis_cap=B)
    S.select = make_select(m)
    LOG.clear()
    try:
        yield
    finally:
        config.RULE.clear()
        config.RULE.update(saved_rule)
        S.select = saved_sel


def indep(m, B, start=START):
    with rule(m, B):
        path = S.run_model(book, start=start)
        log = list(LOG)
    return path, log


def rets(v, days):
    v = np.asarray(v)
    return {days[i]: v[i + 1] / v[i] - 1 for i in range(len(v) - 1)}


def period(r, a, b):
    x = np.array([r[d] for d in sorted(r) if a <= d <= b])
    c = np.cumprod(1 + x)
    return float(c[-1] - 1), float((c / np.maximum.accumulate(c) - 1).min())


def monthly(r):
    return {m: float(np.prod([1 + x for d, x in r.items() if d.startswith(m)]) - 1) for m in MONTHS}


def summary(path):
    r = rets(path.value, path.days)
    out = {lab: period(r, a, b) for lab, (a, b) in (("앞", FRONT), ("뒤", BACK), ("9/1~", SEP), ("전체", FULL))}
    out["월"] = monthly(r)
    out["r"] = r
    return out


def contrib(path, a=None, b=None):
    k0 = book.days.index(path.days[0])
    c = defaultdict(float)
    for i in range(len(path.days) - 1):
        d = path.days[i]
        if (a and d < a) or (b and d > b):
            continue
        rr = S._open_returns(book, path.weights[i], k0 + i + 1)
        for key, wt in path.weights[i].items():
            c[book.ticker[key]] += path.value[i] * wt * rr[key]
    return dict(c)


def divers(path):
    ns, mx, sm = [], [], []
    for w in path.weights:
        g = defaultdict(float)
        for k, x in w.items():
            if x > 1e-12:
                g[grp(k)] += x
        ns.append(len(g))
        mx.append(max(g.values()) if g else 0.0)
        sm.append(sum(x for k, x in w.items() if S.in_semis_budget(book.sector.get(k, S.UNKNOWN))))
    return float(np.mean(ns)), float(np.mean(mx)), float(np.mean(sm))


# ====================================================================== 1) 재현
print("=" * 30, "1) 재현")
ref_live = S.run_model(book, start=START)
with codex_caps():
    ref_774 = S.run_model(book, start=START)
plain = Engine()
d1 = np.max(np.abs(plain.run(Params(semis_cap=0.50))[0].to_numpy() - ref_live.value))
d2 = np.max(np.abs(plain.run(Params())[0].to_numpy() - ref_774.value))
print(f"deskx 50% vs run_model {d1:.2e} | deskx 기본 vs run_model(codex_caps) {d2:.2e}")
assert config.RULE["semis_cap"] == 0.50, config.RULE
assert d1 < 1e-12 and d2 < 1e-12

runs = {}
for m, B in [(7, 0.50), (7, 0.60), (1, 1.0), (1, 0.60), (2, 1.0), (2, 0.60), (7, 1.0)]:
    path, log = indep(m, B)
    vb = E.run(SubParams(m=m, semis_cap=B))[0].to_numpy()
    d = float(np.max(np.abs(vb - path.value)))
    runs[(m, B)] = (path, log)
    print(f"독립 m={m} B={B:.2f} vs 빌더 SubEngine: 최대 차 {d:.2e}")
    assert d < 1e-12, (m, B, d)
assert np.max(np.abs(runs[(7, 0.50)][0].value - ref_live.value)) < 1e-12
assert np.max(np.abs(runs[(7, 0.60)][0].value - ref_774.value)) < 1e-12
print("m=7 = 라이브 50% / #774: 일치")

# ====================================================================== 2) 규칙 확인
print("=" * 30, "2) 40개 재조정 전부 규칙 확인")


def check_decision(dec, tgt, m, B, verbose=False, day=""):
    ranked, current, kept, base = dec["ranked"], dec["current"], dec["kept"], dec["base"]
    order = [r["key"] for r in ranked]
    rank = {k: i + 1 for i, k in enumerate(order)}
    n, buf = 7, 5
    # (a) 그룹 한도
    cnt = Counter(grp(k) for k in kept)
    assert max(cnt.values()) <= m and len(kept) == min(n, len(order))
    # (b) 버퍼 안 보유 종목은 남거나, 같은 그룹의 더 높은 순위 보유 종목 때문에 빠진다
    for k in order[:n + buf]:
        if k in current and k not in kept:
            ahead = [y for y in kept if grp(y) == grp(k) and rank[y] < rank[k] and y in current]
            assert len(ahead) >= m, (day, book.ticker[k])
    # (c) 버퍼 밖 보유 종목은 위에서부터 채우기로만 다시 들어온다
    fill = [k for k in kept if not (k in current and rank[k] <= n + buf)]
    if fill:
        last = max(rank[k] for k in fill)
        for k in order[:last]:
            if k not in kept:      # 채우기 범위 안에서 빠진 종목은 그룹이 찼어야 한다
                assert cnt[grp(k)] >= m, (day, book.ticker[k])
    # (d) 상한
    tw = {r["key"]: r["time_weight"] for r in ranked}
    tot = sum(tw[k] for k in kept)
    raw = {k: tw[k] / tot for k in kept}
    sec = defaultdict(float)
    for k, x in tgt.items():
        sec[book.sector.get(k, S.UNKNOWN)] += x
    semis = sum(x for k, x in tgt.items() if S.in_semis_budget(book.sector.get(k, S.UNKNOWN)))
    cash = 1 - sum(tgt.values())
    assert max(tgt.values()) <= 0.25 + 1e-9 and max(sec.values()) <= 0.60 + 1e-9 and semis <= B + 1e-9
    assert cash >= -1e-9 and set(tgt) <= set(kept)
    ratio = {k: tgt.get(k, 0.0) / raw[k] for k in kept}
    free = [k for k in kept if tgt.get(k, 0) < 0.25 - 1e-9
            and sec[book.sector.get(k, S.UNKNOWN)] < 0.60 - 1e-9
            and not (S.in_semis_budget(book.sector.get(k, S.UNKNOWN)) and semis >= B - 1e-9)]
    if cash > 1e-9:
        assert not free, (day, "현금이 남았는데 안 묶인 종목", [book.ticker[k] for k in free])
    if free:
        rmax = max(ratio.values())
        assert all(abs(ratio[k] - rmax) < 1e-9 for k in free), (day, "비례 아님")
    if verbose:
        print(f"\n[{day}] m={m} B={B}  지난주: {sorted(book.ticker[k] for k in current)}")
        for k in order[:14]:
            flag = ("유지" if k in kept and k in current else "신규" if k in kept else
                    "한도로 빠짐" if k in base else "-")
            print(f"  {rank[k]:>2} {book.ticker[k]:<6} {gname(grp(k)):<16} {book.sector.get(k):<12} "
                  f"원비중 {raw.get(k, 0):.3f} 목표 {tgt.get(k, 0):.3f} {'보유' if k in current else '':<3} {flag}")
        print(f"  반도체 예산 {semis:.3f} / 현금 {cash:.4f} / 섹터합 {dict((s, round(x, 3)) for s, x in sec.items())}")
    return [k for k in base if k not in kept]


bind = {}
for (m, B), (path, log) in runs.items():
    if m == 7:
        continue
    days = list(path.rebalances)
    assert len(days) == len(log) == 40, (len(days), len(log))
    nb, skipped = 0, Counter()
    for dec, day in zip(log, days):
        sk = check_decision(dec, path.rebalances[day], m, B, day=day)
        nb += bool(sk)
        skipped.update(book.ticker[k] for k in sk)
    live_sel = {}
    with rule(7, 0.50):
        S.run_model(book, start=START)
        lv = list(LOG)
    differ = sum(set(a["kept"]) != set(b["kept"]) for a, b in zip(log, lv))
    bind[(m, B)] = (nb, differ, skipped)
    print(f"m={m} B={B:.2f}: 40개 재조정 규칙 통과 | 한도 작동 {nb}주 | live와 다른 7종목 {differ}주 | 빠진 종목 {dict(skipped)}")

# 상세 3일: m=1 B=0.60에서 (i) 보유 종목이 그룹 때문에 빠진 날 (ii) 채우기에서 건너뛴 날 (iii) 미분류 둘 이상 보유한 날
path, log = runs[(1, 0.60)]
days = list(path.rebalances)
pick = {}
for dec, day in zip(log, days):
    order = [r["key"] for r in dec["ranked"]]
    held_drop = [k for k in order[:12] if k in dec["current"] and k not in dec["kept"]]
    fill_skip = [k for k in dec["base"] if k not in dec["kept"] and k not in dec["current"]]
    unk = [k for k in dec["kept"] if isinstance(grp(k), tuple)]
    if held_drop and "i" not in pick:
        pick["i"] = (day, dec)
    if fill_skip and "ii" not in pick:
        pick["ii"] = (day, dec)
    if len(unk) >= 2 and "iii" not in pick:
        pick["iii"] = (day, dec)
for key in ("i", "ii", "iii"):
    if key in pick:
        day, dec = pick[key]
        print(f"\n--- 사례 {key}")
        check_decision(dec, path.rebalances[day], 1, 0.60, verbose=True, day=day)
    else:
        print(f"\n--- 사례 {key}: 해당 날짜 없음")
# 동점 확인
ties = 0
for (m, B), (p_, lg) in runs.items():
    for dec in lg:
        moms = [r["mom120"] for r in dec["ranked"][:14]]
        ties += len(moms) - len(set(moms))
print("상위 14 안 모멘텀 동점 수(전 변형·전 재조정 합):", ties)
# STX/WDC 사례
for dec, day in zip(log, days):
    for k in dec["base"]:
        if k not in dec["kept"] and book.ticker[k] in ("STX", "WDC"):
            rk = {r["key"]: i + 1 for i, r in enumerate(dec["ranked"])}
            mate = [book.ticker[y] + f"#{rk[y]}" + ("(보유)" if y in dec["current"] else "") for y in dec["kept"] if grp(y) == grp(k)]
            print(f"  {day}: {book.ticker[k]}#{rk[k]}{'(보유)' if k in dec['current'] else ''} 빠짐, 같은 그룹 {mate}")

# ====================================================================== 3) 견고성
print("\n" + "=" * 30, "3) 견고성")
SL = {"live50": summary(ref_live), "774": summary(ref_774)}
for mb in [(1, 1.0), (1, 0.60), (2, 1.0), (2, 0.60), (7, 1.0)]:
    SL[f"m{mb[0]}B{mb[1]:.2f}"] = summary(runs[mb][0])
print(f"{'변형':<10}" + "".join(f"{x:>8}" for x in ("앞", "뒤", "9/1~", "전체", "MDD")) + "".join(f"{m[-2:]:>7}" for m in MONTHS))
for name, s in SL.items():
    print(f"{name:<10}" + "".join(f"{s[x][0]:>8.3f}" for x in ("앞", "뒤", "9/1~", "전체")) + f"{s['전체'][1]:>8.3f}"
          + "".join(f"{s['월'][m]:>7.3f}" for m in MONTHS))
for name, path in (("live50", ref_live), ("774", ref_774), ("m1B0.60", runs[(1, 0.60)][0]), ("m2B0.60", runs[(2, 0.60)][0])):
    c = contrib(path)
    g = path.value[-1] - 1
    top = sorted(c.items(), key=lambda x: -x[1])[:3]
    ns, mx, sm = divers(path)
    print(f"{name}: 상위3 {[t for t, _ in top]} 비중 {sum(x for _, x in top) / g:.4f} | 세부수 {ns:.3f} 최대세부 {mx:.4f} 반도체 {sm:.4f}")

# 앞 초과의 하루 쏠림
for base_name in ("live50", "774"):
    rv, rb = SL["m1B0.60"]["r"], SL[base_name]["r"]
    fd = [d for d in sorted(rv) if FRONT[0] <= d <= FRONT[1]]
    ex = period(rv, *FRONT)[0] - period(rb, *FRONT)[0]
    diff = sorted(fd, key=lambda d: -(rv[d] - rb[d]))
    print(f"\nm1B0.60 vs {base_name}: 앞 초과 {ex:+.3f}; 하루 차 상위 5일:",
          [(d, round(rv[d] - rb[d], 4)) for d in diff[:5]])
    for kdrop in (1, 3, 5, 10):
        rr = dict(rv)
        for d in diff[:kdrop]:
            rr[d] = rb[d]
        print(f"  상위 {kdrop}일을 기준 수익으로 바꾸면 앞 초과 {period(rr, *FRONT)[0] - period(rb, *FRONT)[0]:+.3f}")
    # 5/26~5/29 주
    for nm in ("m1B0.60", base_name):
        x = np.prod([1 + SL[nm]["r"][d] for d in SL[nm]["r"] if "2026-05-26" <= d <= "2026-05-29"]) - 1
        print(f"  {nm} 5/26~5/29 시가 경로 {x:+.4f}")
    # 3개월 구간
    for a, b in (("2026-01-02", "2026-02-27"), ("2026-03-02", "2026-04-30"), ("2026-05-01", "2026-05-29")):
        print(f"  {a}~{b}: m1B0.60 {period(rv, a, b)[0]:+.3f} vs {base_name} {period(rb, a, b)[0]:+.3f}")

cv = contrib(runs[(1, 0.60)][0], *FRONT)
cl = contrib(ref_live, *FRONT)
dd = sorted(((t, cv.get(t, 0) - cl.get(t, 0)) for t in set(cv) | set(cl)), key=lambda x: -abs(x[1]))
print("\n앞 구간 종목 기여 차(m1B0.60 - live50, 가치 가중):", [(t, round(x, 3)) for t, x in dd[:10]])
cvb = contrib(runs[(1, 0.60)][0], *BACK)
clb = contrib(ref_live, *BACK)
ddb = sorted(((t, cvb.get(t, 0) - clb.get(t, 0)) for t in set(cvb) | set(clb)), key=lambda x: -abs(x[1]))
print("뒤 구간 종목 기여 차:", [(t, round(x, 3)) for t, x in ddb[:10]])

# 이웃 값(감사용)
print("\n이웃 값(감사용, 후보 아님): 앞 / 뒤 / 전체 / MDD")
nb_rows = []
for m in (1, 2, 3):
    for B in (0.50, 0.55, 0.60, 0.65, 0.70, 1.0):
        s = summary(indep(m, B)[0])
        nb_rows.append((m, B, s))
        win = "앞·뒤 모두 live 초과" if s["앞"][0] > SL["live50"]["앞"][0] and s["뒤"][0] > SL["live50"]["뒤"][0] else ""
        print(f"  m={m} B={B:.2f}: {s['앞'][0]:+.3f} {s['뒤'][0]:+.3f} {s['전체'][0]:+.3f} {s['전체'][1]:+.3f} {win}")

# 시작일 이동(경로 의존성): m=1 B=0.60과 live의 앞 초과
print("\n시작일 이동: (시작~5/29) m1B0.60 - live50, 뒤 초과")
for st in ("2026-01-02", "2026-01-12", "2026-01-20", "2026-02-02", "2026-02-17", "2026-03-02"):
    a = summary(indep(1, 0.60, start=st)[0])
    b = summary(indep(7, 0.50, start=st)[0])
    fa = period(a["r"], st, FRONT[1])[0]
    fb = period(b["r"], st, FRONT[1])[0]
    print(f"  {st}: 앞 {fa:+.3f} vs {fb:+.3f} (차 {fa - fb:+.3f}) | 뒤 {a['뒤'][0]:+.3f} vs {b['뒤'][0]:+.3f}")
# ====================================================================== 4) 세부
print("\n" + "=" * 30, "4) 앞 초과 상위 날의 종목 분해, 낙폭 시점, 10/12 미리보기")
pv, pl = runs[(1, 0.60)][0], ref_live
for d in ("2026-04-23", "2026-04-28", "2026-05-20", "2026-03-16", "2026-05-08"):
    i = pv.days.index(d)
    k = book.days.index(d) + 1
    out = []
    for nm, p in (("m1B0.60", pv), ("live50", pl)):
        rr = S._open_returns(book, p.weights[i], k)
        out.append(f"{nm}: " + ", ".join(f"{book.ticker[x]} {w:.2f}×{rr[x]:+.3f}" for x, w in
                                          sorted(p.weights[i].items(), key=lambda t: -t[1])))
    print(f"  {d}→{book.days[k]}  " + " | ".join(out))


def mdd_where(path, a=START, b="2026-10-09"):
    v = np.array([1.0] + [x for d, x in zip(path.days[1:], path.value[1:]) if a <= d <= b])
    ds = ["start"] + [d for d in path.days[1:] if a <= d <= b]
    dd = v / np.maximum.accumulate(v) - 1
    j = int(dd.argmin())
    pk = int(np.argmax(v[:j + 1]))
    return round(float(dd[j]), 4), ds[pk], ds[j]


m150 = indep(1, 0.50)[0]
for nm, p in (("live50", pl), ("774", ref_774), ("m1B0.60", pv), ("m1B0.50(감사용)", m150)):
    print(f"  {nm}: 전체 MDD(시가 경로, 시작 1.0 포함) {mdd_where(p)} | 앞 {mdd_where(p, *FRONT)} | 뒤 {mdd_where(p, *BACK)}")
s150 = summary(m150)
print("  m1B0.50(감사용) 월별:", {m[-2:]: round(x, 3) for m, x in s150["월"].items()})

# 10/12 미리보기: live = desk next_target, m=1 B=0.60 = 독립 select로 weekly_target
nt = S.next_target(book, pl, "2026-10-12")
print("  live next_target:", {book.ticker[x]: round(w, 4) for x, w in sorted(nt["target"].items(), key=lambda t: -t[1])})
with rule(1, 0.60):
    d = S.weekly_target(book, len(book.days), "2026-10-12", pv.selected)
print("  m1B0.60 독립 weekly_target:", {book.ticker[x]: round(w, 4) for x, w in sorted(d["target"].items(), key=lambda t: -t[1])},
      "pdf", d["pdf"], "건너뜀", [book.ticker[x] for x in LOG[-1]["base"] if x not in LOG[-1]["kept"]])
print("\n완료")
