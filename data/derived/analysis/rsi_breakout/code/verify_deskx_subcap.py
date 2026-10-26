"""subcap 패밀리 감사(2026-10-10, 연구용 스크래치). deskx_subcap.py를 반박하려는 독립 점검.

1) 재현: deskx 현행 50%·#774 vs desk.strategies.run_model(가치 + 매일 비중), sub_cap=1.0 경로 = 기본 경로
   (반도체 0.50·0.60·1.0 모두), 미리 보기 결정 = desk next_target과 같은 데이터 흐름인지는 생략(앞의 경로가 같음)
2) 미래 참조: 재조정일 k마다 k일 이후 가격 행을 NaN으로 지우고 k일 이후 PDF를 빼서 결정을 다시 계산 → 같아야 한다
3) 투영: 다른 방식(수준 λ 물 채우기, 닫힌 식)으로 다시 짜서 project_caps_sub와 대조 + 가중 max-min 병목 조건
   (모든 종목은 자기가 속한 묶인 상한 하나에서 비율 alloc/raw가 최대) + 실제 재조정 3일 상세 출력
4) 견고성: 월별, 종목별 기여 차, 하루 쏠림, 앞 구간 마지막 주(DELL 실적 주) 제외, 이웃 값(감사용),
   분할 시점을 옮긴 전방 검증, 투자 비중·QQQ 베타
실행: code 폴더에서  PYTHONIOENCODING=utf-8 py verify_deskx_subcap.py
"""
import json, os, random, sys
from dataclasses import asdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from deskx import Engine, Params, START, END  # noqa: E402
from deskx_subcap import SubEngine, SubParams, GRID, POSTHOC, project_caps_sub  # noqa: E402  (감사 대상)
from desk import config  # noqa: E402
from desk.strategies import run_model, subsector_of, in_semis_budget, UNKNOWN, latest_before  # noqa: E402
from desk.validate import codex_caps  # noqa: E402

OUT = os.path.join(HERE, "../results/deskx/verify_subcap.json")
FRONT = ("2026-01-02", "2026-05-29")
BACK = ("2026-06-01", "2026-10-09")
SEPO = ("2026-09-01", "2026-10-09")
REPORT = {}


def per(v, a, b):
    r = (v.shift(-1) / v - 1).dropna()
    x = r[(r.index >= a) & (r.index <= b)]
    c = (1 + x).cumprod()
    return float(c.iloc[-1] - 1), float((c / np.maximum.accumulate(c) - 1).min())


def daily(v):
    return (v.shift(-1) / v - 1).dropna()


# ------------------------------------------------------------------ 1) 재현
def check_reproduction():
    assert abs(config.RULE["semis_cap"] - 0.50) < 1e-12, config.RULE
    E = Engine()
    out = {}
    v50, rows50 = E.run(Params(semis_cap=0.50))
    ref50 = run_model(E.book, start=START)
    out["live50_value_maxdiff"] = float(np.max(np.abs(v50.to_numpy() - ref50.value)))
    out["live50_weights_maxdiff"] = max(
        max([abs(r["w"].get(k, 0) - rw.get(k, 0)) for k in set(r["w"]) | set(rw)] or [0])
        for r, rw in zip(rows50, ref50.weights))
    v60, rows60 = E.run(Params())
    with codex_caps():
        ref60 = run_model(E.book, start=START)
    out["774_value_maxdiff"] = float(np.max(np.abs(v60.to_numpy() - ref60.value)))
    out["774_weights_maxdiff"] = max(
        max([abs(r["w"].get(k, 0) - rw.get(k, 0)) for k in set(r["w"]) | set(rw)] or [0])
        for r, rw in zip(rows60, ref60.weights))
    S = SubEngine()
    for s in (0.50, 0.60, 1.0):
        a, ra = E.run(Params(semis_cap=s))
        b, rb = S.run(SubParams(semis_cap=s, sub_cap=1.0))
        out[f"sub1.0_vs_base_semis{s}_value"] = float(np.max(np.abs(a.to_numpy() - b.to_numpy())))
        out[f"sub1.0_vs_base_semis{s}_weights"] = max(
            max([abs(x["w"].get(k, 0) - y["w"].get(k, 0)) for k in set(x["w"]) | set(y["w"])] or [0])
            for x, y in zip(ra, rb))
    out["last_day"] = E.days[-1]
    out["n_days_2026"] = len(v50)
    REPORT["reproduction"] = out
    print("[1] 재현:", json.dumps(out, ensure_ascii=False))
    return S


# ------------------------------------------------------------------ 2) 미래 참조
class RecEngine(SubEngine):
    """재조정마다 (k, 날짜, 이전 선택, 결과)를 기록한다."""

    def __init__(self):
        super().__init__()
        self.log = []

    def target(self, p, k, session, current):
        keep, tgt, ranked = super().target(p, k, session, current)
        self.log.append({"k": k, "day": session, "current": set(current), "keep": list(keep), "tgt": dict(tgt),
                         "raw": dict(self.raws[-1]) if ranked else {}})
        return keep, tgt, ranked


def check_lookahead(R, p):
    R.log = []
    R.run(p)
    log = list(R.log)
    book = R.book
    orig = (book.adj, book.opn, book.cls, book.pdfs)
    bad, used_pdfs = 0, []
    for e in log:
        k, day = e["k"], e["day"]
        try:
            for name in ("adj", "opn", "cls"):
                df = getattr(book, name).copy()
                df.iloc[k:] = np.nan
                setattr(book, name, df)
            book.pdfs = [d for d in orig[3] if d < day]
            keep, tgt, _ = SubEngine.target(R, p, k, day, e["current"])
            R.raws.pop()
        finally:
            book.adj, book.opn, book.cls, book.pdfs = orig
        same = keep == e["keep"] and set(tgt) == set(e["tgt"]) and \
            all(abs(tgt[x] - e["tgt"][x]) < 1e-15 for x in tgt)
        bad += not same
        used_pdfs.append((day, latest_before(orig[3], day)))
    late = [(d, pdf) for d, pdf in used_pdfs if pdf is not None and pdf >= d]
    none = [d for d, pdf in used_pdfs if pdf is None]
    return {"rebalances": len(log), "mismatch_after_masking": bad, "pdf_not_strictly_before": len(late),
            "no_pdf_yet_cash_days": none,
            "first": log[0]["day"], "last": log[-1]["day"]}, log


# ------------------------------------------------------------------ 3) 독립 투영
def level_fill(raw, sector, sub, stock, detail, semis, subcap, tol=1e-12):
    """수준 λ 물 채우기: 아직 안 묶인 종목은 모두 λ·raw. λ를 닫힌 식으로 가장 먼저 묶이는 상한까지 올리고
    그 상한에 든 종목을 그 자리에 고정한다. 반복."""
    names = [k for k, v in raw.items() if v > 0]
    cons = [({k}, stock) for k in names]
    for lab, cap in ((lambda k: ("sec", sector.get(k, UNKNOWN)), detail), (lambda k: ("sub", sub[k]), subcap)):
        groups = {}
        for k in names:
            groups.setdefault(lab(k), set()).add(k)
        cons += [(g, cap) for g in groups.values()]
    bud = {k for k in names if in_semis_budget(sector.get(k, UNKNOWN))}
    if bud:
        cons.append((bud, semis))
    cons.append((set(names), sum(raw[k] for k in names)))
    fixed, free = {}, set(names)
    while free:
        lam = np.inf
        for mem, cap in cons:
            rate = sum(raw[k] for k in mem & free)
            if rate > 0:
                lam = min(lam, (cap - sum(fixed.get(k, 0.0) for k in mem)) / rate)
        lam = max(lam, 0.0)
        tight = set()
        for mem, cap in cons:
            rate = sum(raw[k] for k in mem & free)
            if rate > 0 and cap - (sum(fixed.get(k, 0.0) for k in mem) + lam * rate) <= tol:
                tight |= mem & free
        for k in tight:
            fixed[k] = lam * raw[k]
        free -= tight
    return {k: v for k, v in fixed.items() if v > 1e-12}, cons


def bottleneck_ok(alloc, raw, cons, tol=1e-9):
    """가중 max-min 공정: 각 종목에 대해 묶인(꽉 찬) 상한 c∋k가 있고 그 안에서 alloc/raw가 최대."""
    a = {k: alloc.get(k, 0.0) for k in raw if raw[k] > 0}
    ratio = {k: a[k] / raw[k] for k in a}
    top = max(ratio.values()) if ratio else 0.0
    for k in a:
        if k not in alloc and raw[k] * top <= 1e-11:
            continue          # 원비중이 1e-12 아래라 출력에서 0으로 지운 종목(두 투영 모두 같은 처리)
        ok = False
        for mem, cap in cons:
            if k in mem and sum(a[j] for j in mem) >= cap - 1e-9 and ratio[k] >= max(ratio[j] for j in mem) - tol:
                ok = True
                break
        if not ok:
            return False
    # 상한 위반 없음
    return all(sum(a[j] for j in mem) <= cap + 1e-12 for mem, cap in cons)


def check_projector(R, logs_by_variant):
    rng = random.Random(77)
    keys = sorted(R.ticker)
    # 세부 섹터가 겹치는 종목 위주의 스트레스 집합(무작위 전체 추출은 세부 상한이 거의 안 묶인다)
    by_sub = {}
    for k in keys:
        by_sub.setdefault(R.sub[k], []).append(k)
    dense = [k for g, ks in by_sub.items() if len(ks) >= 3 for k in ks]
    sets = []
    for i in range(3000):
        pool = dense if i % 2 else keys
        n = rng.randint(1, min(12, len(pool)))
        ks = rng.sample(pool, n)
        a = rng.choice([0.3, 1.0, 3.0])
        w = [rng.gammavariate(a, 1.0) for _ in ks]
        t = sum(w)
        sets.append({x: y / t for x, y in zip(ks, w)})
    uniq_rebal = {}
    for lg in logs_by_variant.values():
        for e in lg:
            uniq_rebal[(e["day"], tuple(sorted(e["raw"].items())))] = e["raw"]
    rebal = list(uniq_rebal.values())
    caps = [(0.25, 0.60, s, x) for s in (1.0, 0.70, 0.60, 0.50) for x in (0.15, 0.20, 0.25, 0.30, 0.40, 1.0)]
    n = worst = bad_bn = bad_mine = 0
    for raw in sets + rebal:
        for stock, detail, semis, x in caps:
            a = project_caps_sub(raw, R.sector, R.sub, stock, detail, semis, x)
            b, cons = level_fill(raw, R.sector, R.sub, stock, detail, semis, x)
            d = max([abs(a.get(z, 0.0) - b.get(z, 0.0)) for z in set(a) | set(b)] or [0.0])
            worst = max(worst, d)
            bad_bn += not bottleneck_ok(a, raw, cons)
            bad_mine += not bottleneck_ok(b, raw, cons)
            n += 1
    sub_binds = sum(1 for raw in sets for (_, _, s, x) in caps[:5]
                    if level_fill(raw, R.sector, R.sub, 0.25, 0.60, s, x)[0] !=
                    level_fill(raw, R.sector, R.sub, 0.25, 0.60, s, 1.0)[0])
    out = {"cases": n, "random_sets": len(sets), "unique_rebalance_raw_sets": len(rebal),
           "maxdiff_vs_level_fill": worst, "builder_bottleneck_failures": bad_bn,
           "level_fill_bottleneck_failures": bad_mine, "random_cases_where_subcap_changed_result": sub_binds}
    REPORT["projector"] = out
    print("[3] 투영 독립 대조:", json.dumps(out, ensure_ascii=False))


def show_rebalance(R, e, p, title):
    raw, tgt = e["raw"], e["tgt"]
    if not raw:
        print(f"  {title} {e['day']}: 순위 없음(현금)")
        return {"day": e["day"], "ok": True, "empty": True}
    sub = {}
    sec = {}
    for k, w in tgt.items():
        sub[R.sub[k]] = sub.get(R.sub[k], 0.0) + w
        sec[R.sector[k]] = sec.get(R.sector[k], 0.0) + w
    semis = sum(w for k, w in tgt.items() if in_semis_budget(R.sector[k]))
    cash = 1 - sum(tgt.values())
    _, cons = level_fill(raw, R.sector, R.sub, p.stock_cap, p.sector_cap, p.semis_cap, p.sub_cap)
    print(f"\n  {title} {e['day']}  현금 {cash:.4f}  반도체예산 {semis:.4f} (상한 {p.semis_cap})")
    ratios = {k: tgt.get(k, 0) / raw[k] for k in raw}
    top = max(ratios.values())
    for k in sorted(raw, key=lambda z: -tgt.get(z, 0)):
        binds = []
        if tgt.get(k, 0) >= p.stock_cap - 1e-9:
            binds.append("종목")
        if sec[R.sector[k]] >= p.sector_cap - 1e-9:
            binds.append("섹터")
        if sub[R.sub[k]] >= p.sub_cap - 1e-9:
            binds.append("세부")
        if in_semis_budget(R.sector[k]) and semis >= p.semis_cap - 1e-9:
            binds.append("반도체")
        print(f"    {R.ticker[k]:<6} raw {raw[k]:.4f} → {tgt.get(k, 0):.4f}  비율 {ratios[k]:.4f}"
              f"{'(최대)' if abs(ratios[k] - top) < 1e-9 else ''}  {R.sub[k]} / {R.sector[k]}  묶임:{','.join(binds) or '-'}")
    viol = [g for g, w in sub.items() if w > p.sub_cap + 1e-12] + [g for g, w in sec.items() if w > p.sector_cap + 1e-12]
    free_ratios = {round(ratios[k], 12) for k in raw
                   if not (tgt.get(k, 0) >= p.stock_cap - 1e-9 or sec[R.sector[k]] >= p.sector_cap - 1e-9
                           or sub[R.sub[k]] >= p.sub_cap - 1e-9
                           or (in_semis_budget(R.sector[k]) and semis >= p.semis_cap - 1e-9))}
    ok = not viol and max(tgt.values()) <= p.stock_cap + 1e-12 and semis <= p.semis_cap + 1e-12 \
        and len(free_ratios) <= 1 and (cash < 1e-9 or not free_ratios) and bottleneck_ok(tgt, raw, cons)
    print(f"    상한 위반 {viol or '없음'}, 안 묶인 종목 비율 {sorted(free_ratios)} → {'OK' if ok else 'FAIL'}")
    return {"day": e["day"], "cash": cash, "semis": semis, "max_sub": max(sub.values()), "ok": bool(ok),
            "weights": {R.ticker[k]: round(w, 4) for k, w in tgt.items()}}


# ------------------------------------------------------------------ 4) 견고성
def contrib_window(R, v, rows, a, b):
    days = list(v.index)
    out = {}
    base = None
    for i in range(len(rows) - 1):
        d = days[i]
        if not (a <= d <= b):
            continue
        if base is None:
            base = v.iloc[i]
        k = R.days.index(d) + 1
        for key, wt in rows[i]["w"].items():
            o = R.col("opn", key)
            ret = (o[k] / o[k - 1] - 1) if o is not None and np.isfinite(o[k]) and np.isfinite(o[k - 1]) \
                and o[k - 1] > 0 else 0.0
            out[R.ticker[key]] = out.get(R.ticker[key], 0.0) + v.iloc[i] / base * wt * ret
    return out


def divers(R, rows, v):
    nsub, mx, sem, inv = [], [], [], []
    for r in rows:
        by = {}
        for k, w in r["w"].items():
            lab = subsector_of(k, R.ticker[k])
            lab = f"미분류:{R.ticker[k]}" if lab == UNKNOWN else lab
            by[lab] = by.get(lab, 0) + w
        nsub.append(len(by))
        mx.append(max(by.values()) if by else 0)
        sem.append(sum(w for k, w in r["w"].items() if in_semis_budget(R.sector[k])))
        inv.append(sum(r["w"].values()))
    c = contrib_window(R, v, rows, "2026-01-02", "2026-10-09")
    gain = sum(c.values())
    top = sorted(c.items(), key=lambda x: -x[1])[:3]
    return {"subsectors_avg": float(np.mean(nsub)), "max_sub_avg": float(np.mean(mx)), "semis_avg": float(np.mean(sem)),
            "invested_avg": float(np.mean(inv)), "top3": [t for t, _ in top],
            "top3_share": sum(x for _, x in top) / gain, "gain_check": gain - (v.iloc[-1] - 1)}


def main():
    S = check_reproduction()
    R = RecEngine()
    best_p = SubParams(sub_cap=0.25, semis_cap=1.0)
    live_p = SubParams(semis_cap=0.50)
    variants = {"live50": live_p, "774": SubParams(semis_cap=0.60)}
    variants.update({n: p for n, p in GRID})
    variants[POSTHOC[0]] = POSTHOC[1]
    logs, paths = {}, {}
    la = {}
    for name, p in variants.items():
        res, lg = check_lookahead(R, p)
        logs[name] = lg
        la[name] = res
        paths[name] = R.run(p)
    REPORT["lookahead"] = la
    print("[2] 미래 참조(가려도 결정 같음):", json.dumps(la, ensure_ascii=False))

    check_projector(R, logs)

    # 실제 재조정 상세(3일 + 마지막) — 앞 최선 변형과 sub 0.20 / 반도체 0.60
    print("\n[3b] 실제 재조정 상세")
    det = []
    lg = {e["day"]: e for e in logs["sub 0.25 / 반도체 없음"]}
    for d in ("2026-01-05", "2026-05-26", "2026-07-06", "2026-10-05"):
        if d in lg:
            det.append(show_rebalance(R, lg[d], best_p, "sub 0.25/반도체 없음"))
    lg2 = {e["day"]: e for e in logs["sub 0.20 / 반도체 0.60"]}
    for d in ("2026-03-02", "2026-08-03"):
        if d in lg2:
            det.append(show_rebalance(R, lg2[d], SubParams(sub_cap=0.20, semis_cap=0.60), "sub 0.20/반도체 0.60"))
    REPORT["rebalance_details"] = det

    # 그룹 라벨 점검: 2026에 실제로 담긴 종목
    held = set()
    for name, (v, rows) in paths.items():
        for r in rows:
            held |= set(r["w"])
    from watchlist import resolve_subsector
    lab = []
    for k in sorted(held, key=lambda z: R.ticker[z]):
        raw_sub = resolve_subsector(R.ticker[k])
        lab.append((R.ticker[k], R.sub[k], R.sector[k], "AI소분류" if raw_sub else "섹터대체"))
    REPORT["held_groups"] = lab
    print("\n[3c] 2026 보유 종목 그룹:", "; ".join(f"{t}={s}({src})" for t, s, _, src in lab))

    # 4) 견고성 ---------------------------------------------------------------
    v_live, rows_live = paths["live50"]
    v_774, rows_774 = paths["774"]
    v_best, rows_best = paths["sub 0.25 / 반도체 없음"]
    print("\n[4a] 분산 지표 독립 재계산")
    dv = {n: divers(R, paths[n][1], paths[n][0]) for n in paths}
    for n, d in dv.items():
        print(f"  {n:<26} 세부수 {d['subsectors_avg']:.3f} 최대세부 {d['max_sub_avg']:.4f} 반도체 {d['semis_avg']:.4f}"
              f" 투자 {d['invested_avg']:.4f} 상위3 {d['top3_share']:.4f} {d['top3']}  (이익합 차 {d['gain_check']:.1e})")
    REPORT["diversification"] = dv

    print("\n[4b] 월별(6~10월)과 구간")
    tab = {}
    for n, (v, _) in paths.items():
        r = daily(v)
        m = {mo: float((1 + r[r.index.str.startswith(mo)]).prod() - 1) for mo in
             ("2026-06", "2026-07", "2026-08", "2026-09", "2026-10")}
        f, fm = per(v, *FRONT)
        b, bm = per(v, *BACK)
        s, _ = per(v, *SEPO)
        a, am = per(v, "2026-01-02", "2026-10-09")
        tab[n] = {"front": f, "back": b, "sepoct": s, "full": a, "mdd": am, "front_mdd": fm, "back_mdd": bm, **m}
        print(f"  {n:<26} 앞 {f:.4f} 뒤 {b:+.4f} 9/1~ {s:+.4f} 전체 {a:.4f} MDD {am:.4f} 뒤MDD {bm:.4f} | "
              + " ".join(f"{x[-2:]}월 {y:+.4f}" for x, y in m.items()))
    REPORT["table"] = tab

    # 종목별 기여 차(앞, 뒤): 앞 최선 - 현행
    print("\n[4c] 종목별 기여 차 (sub 0.25/반도체 없음 − 현행 50%)")
    for lab_, (a, b) in (("앞", FRONT), ("뒤", BACK)):
        cb = contrib_window(R, v_best, rows_best, a, b)
        cl = contrib_window(R, v_live, rows_live, a, b)
        diff = {t: cb.get(t, 0) - cl.get(t, 0) for t in set(cb) | set(cl)}
        tot = sum(diff.values())
        top = sorted(diff.items(), key=lambda x: -abs(x[1]))[:6]
        print(f"  {lab_}: 합 {tot:+.4f} | " + ", ".join(f"{t} {x:+.4f}" for t, x in top))
        REPORT[f"contrib_diff_{lab_}"] = {"total": tot, "top": top}

    # 하루 쏠림: 앞 구간 일별 log 차
    print("\n[4d] 하루 쏠림(앞): log(1+r_best) − log(1+r_live)")
    rb, rl = daily(v_best), daily(v_live)
    m = (rb.index >= FRONT[0]) & (rb.index <= FRONT[1])
    dlog = (np.log1p(rb[m]) - np.log1p(rl[m]))
    tot = float(dlog.sum())
    top = dlog.sort_values(ascending=False)
    print(f"  앞 log 차 합 {tot:+.4f}; 상위 5일 " + ", ".join(f"{d} {x:+.4f}" for d, x in top.head(5).items())
          + f"; 상위 2일 합 {top.head(2).sum():+.4f} ({top.head(2).sum() / tot:.0%}), 상위 5일 합 {top.head(5).sum():+.4f}")
    wk = dlog[(dlog.index >= "2026-05-26")]
    print(f"  5/26~5/29 시가 4일 log 차 합 {wk.sum():+.4f}")
    REPORT["front_day_concentration"] = {"total_log_diff": tot, "top5": top.head(5).to_dict(),
                                         "top2_sum": float(top.head(2).sum()), "dell_week": float(wk.sum())}

    # 앞 구간에서 DELL 실적 주 빼고 다시 고르기
    print("\n[4e] 앞 구간을 5/22 시가(→5/26 시가)까지로 줄여 다시 고르기")
    ex = {}
    for n, (v, _) in paths.items():
        ex[n] = per(v, "2026-01-02", "2026-05-22")[0]
    grid_names = [n for n, _ in GRID]
    pick_ex = max(grid_names, key=lambda n: ex[n])
    for n in paths:
        print(f"  {n:<26} 앞(5/22까지) {ex[n]:.4f}")
    print("  → 다시 고른 변형:", pick_ex)
    REPORT["front_ex_dell_week"] = {"values": ex, "pick": pick_ex}

    # 분할 시점을 옮긴 전방 검증(6개 사전 등록 격자에서 고르고 그 뒤 구간을 현행과 비교)
    print("\n[4f] 분할 시점 이동 전방 검증")
    roll = []
    for cut, nxt in (("2026-03-31", "2026-04-01"), ("2026-04-30", "2026-05-01"), ("2026-05-29", "2026-06-01"),
                     ("2026-06-30", "2026-07-01"), ("2026-07-31", "2026-08-01")):
        pick = max(grid_names, key=lambda n: per(paths[n][0], "2026-01-02", cut)[0])
        after = per(paths[pick][0], nxt, "2026-10-09")
        after_live = per(v_live, nxt, "2026-10-09")
        roll.append({"cut": cut, "pick": pick, "after": after[0], "after_mdd": after[1], "live_after": after_live[0],
                     "live_after_mdd": after_live[1]})
        print(f"  {cut}까지로 고름 → {pick}: 이후 {after[0]:+.4f} (MDD {after[1]:.4f}) vs 현행 {after_live[0]:+.4f}"
              f" (MDD {after_live[1]:.4f})")
    REPORT["rolling_origin"] = roll

    # 이웃 값(감사용, 후보 아님)
    print("\n[4g] 이웃 값 (감사용 진단)")
    neigh = {}
    for s in (1.0, 0.70, 0.60, 0.50):
        for x in (0.15, 0.175, 0.20, 0.225, 0.25, 0.275, 0.30, 0.35, 0.40, 0.50, 1.0):
            v, rows = S.run(SubParams(sub_cap=x, semis_cap=s))
            f = per(v, *FRONT)[0]
            b = per(v, *BACK)[0]
            a, am = per(v, "2026-01-02", "2026-10-09")
            neigh[f"{x}/{s}"] = {"front": f, "back": b, "full": a, "mdd": am,
                                "inv": float(np.mean([sum(r["w"].values()) for r in rows]))}
        print(f"  반도체 {s}: " + " | ".join(f"X{x}: 앞 {neigh[f'{x}/{s}']['front']:.3f} 뒤 {neigh[f'{x}/{s}']['back']:+.3f}"
                                           f" MDD {neigh[f'{x}/{s}']['mdd']:.3f}"
                                           for x in (0.15, 0.175, 0.20, 0.225, 0.25, 0.275, 0.30, 0.35, 0.40, 0.50, 1.0)))
    REPORT["neighbours"] = neigh

    # 노출 효과: QQQ 베타와 투자 비중
    print("\n[4h] 노출: QQQ 시가→시가 베타(2026 전체)")
    q = R.book.opn["QQQ"]
    qr = (q.shift(-1) / q - 1)
    beta = {}
    for n, (v, rows) in paths.items():
        r = daily(v)
        x = qr.reindex(r.index)
        ok = x.notna()
        beta[n] = float(np.cov(r[ok], x[ok])[0, 1] / np.var(x[ok], ddof=1))
        print(f"  {n:<26} 베타 {beta[n]:.3f}  투자 평균 {dv[n]['invested_avg']:.3f}  변동성(연) {r.std() * np.sqrt(252):.3f}")
    REPORT["beta_qqq"] = beta

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(REPORT, f, ensure_ascii=False, indent=1, default=str)
    print("\n저장:", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
