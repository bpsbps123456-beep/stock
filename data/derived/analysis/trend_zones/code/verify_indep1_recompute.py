"""검증자 #1 — 독립 재계산(2026-10-11). zones.py를 import하지 않는다(verify_indep1_core.py 사용).
1) 432개 전부 + 옛 칼만을 다시 계산해 zones.json과 대조(허용 0.05%p), 자격 107개·1위 확인.
2) 1위·옛 칼만·여러 계열 10개는 groupby 긴 표 방식으로 한 번 더 계산(numpy 방식과 대조).
3) K1·K2·K3 숫자 재계산(SOXX 경고는 달력 월요일 기준으로 따로 구현해 zones_regime.json과 대조).
실행: $env:PYTHONIOENCODING='utf-8'; py data/derived/analysis/trend_zones/code/verify_indep1_recompute.py
결과: results/verify_indep1_recompute.json, .txt
"""
import json, os, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import verify_indep1_core as V  # noqa: E402

H = 10
TOL = 0.0005   # 0.05%p


def pct(v):
    return f"{v * 100:+.2f}"


def main():
    t0 = time.time()
    dates, syms, P = V.load()
    Pn = V.Panel(dates, syms, P)
    EX = V.excess(Pn.fwd(H))
    rows = {p: V.rows_of(dates, p, H) for p in V.PER}
    G = V.grid()
    assert len(G) == 432 and len(set(map(V.key, G))) == 432
    mine = {}
    for c in G + [V.BASE]:
        Z = V.zones(Pn, c)
        mine[V.key(c)] = {"cfg": c, **{p: V.score(Z, EX, rows[p]) for p in V.PER}}
    out, L = {}, []
    L.append(f"종목 {len(syms)}개, {dates[0]}~{dates[-1]}, 계산 {time.time() - t0:.0f}초")
    L.append("구간 행 수: " + ", ".join(f"{p} {len(r)}일({dates[r[0]]}~{dates[r[-1]]})" for p, r in rows.items()))
    L.append(f"앞 마지막 신호일 {dates[rows['앞'][-1]]} → 청산 시가일 {dates[rows['앞'][-1] + 1 + H]} (앞 끝 2026-05-29 이내)")

    # ---------- 1. zones.json 대조
    ref = json.load(open(os.path.join(V.RES, "zones.json"), encoding="utf-8"))
    tmap = {"T1": "T1 상방-하방", "T2": "T2 극하방-하방", "T3": "T3 상방-극상방"}
    worst = {"S": 0, "T": 0, "share": 0}
    worst_at = {}
    elig_ref, elig_mine = set(), set()
    for g in ref["grid"]:
        k = V.key(g["cfg"])
        m = mine[k]
        if g["eligible"]:
            elig_ref.add(k)
        if V.eligible(m["앞"]["share"]):
            elig_mine.add(k)
        for p in ("앞", "뒤", "2025"):
            worst_at["n"] = worst_at.get("n", 0) + 1
            d = abs(m[p]["S"] - g[p]["S"])
            if d > worst["S"]:
                worst["S"], worst_at["S"] = d, (g["name"], p)
            for t in V.TK:
                a, b = m[p]["T"][t], g[p]["terms"][tmap[t]]
                if np.isfinite(a) or np.isfinite(b):
                    d = abs(a - b) if (np.isfinite(a) and np.isfinite(b)) else 9
                    if d > worst["T"]:
                        worst["T"], worst_at["T"] = d, (g["name"], p, t)
            for z in V.ZN:
                d = abs(m[p]["share"][z] - g[p]["share"][str(z)])
                if d > worst["share"]:
                    worst["share"], worst_at["share"] = d, (g["name"], p, z)
    L += ["", "== 1. 432개 전체 대조(zones.json)",
          f"최대 차이: S {worst['S'] * 100:.5f}%p {worst_at.get('S')}, 항 {worst['T'] * 100:.5f}%p {worst_at.get('T')}, "
          f"비중 {worst['share'] * 100:.5f}%p {worst_at.get('share')}",
          f"허용 0.05%p 통과: {max(worst['S'], worst['T']) <= TOL} (설정×구간 비교 {worst_at.pop('n')}건, 차이 0이면 위치 None)",
          f"자격: 원본 {len(elig_ref)}개, 내 계산 {len(elig_mine)}개, 일치 {elig_ref == elig_mine}"]
    ranked = sorted(elig_mine, key=lambda k: -mine[k]["앞"]["S"])
    L.append("내 계산 자격 설정 앞 S 상위 5: " + "; ".join(f"{k} {pct(mine[k]['앞']['S'])}" for k in ranked[:5]))
    L.append(f"1위 = mz(60)≥0.5 + xrsi≥70 ? {ranked[0] == V.key(V.WIN)}  (2위와 차 {pct(mine[ranked[0]]['앞']['S'] - mine[ranked[1]]['앞']['S'])}%p)")
    # 자격 탈락이지만 앞 S가 더 높은 설정(참고)
    allr = sorted(mine, key=lambda k: -mine[k]["앞"]["S"])
    above = [k for k in allr if mine[k]["앞"]["S"] > mine[V.key(V.WIN)]["앞"]["S"] and k != V.key(V.BASE)]
    L.append(f"자격 무시 시 1위보다 앞 S가 높은 설정 {len(above)}개: " + "; ".join(
        f"{k} {pct(mine[k]['앞']['S'])}(극상 {mine[k]['앞']['share'][2] * 100:.1f}%·극하 {mine[k]['앞']['share'][-2] * 100:.1f}%)" for k in above[:6]))
    out["compare"] = {"worst": worst, "worst_at": {k: list(map(str, v)) for k, v in worst_at.items()},
                      "eligible_ref": len(elig_ref), "eligible_mine": len(elig_mine), "eligible_same": elig_ref == elig_mine,
                      "winner_mine": ranked[0], "top5": [(k, mine[k]["앞"]["S"]) for k in ranked[:5]]}

    # ---------- 2. 개별 설정, groupby 방식으로 한 번 더 + 원본 대조
    pick = [V.WIN, V.BASE, ("dev", 10, "rms20", 1.0, 2.0), ("dev", 20, "atr14", 0.75, 2.0), ("dev", 35, "rms60", 0.5, 1.5),
            ("tx", "slope", (50, 10), 0.05, "xrsi", 70), ("tx", "cross", (20, 60), 1.0, "xrsi", 70),
            ("tx", "cross", (50, 150), 1.0, "xrsi", 70), ("tx", "don", (120), 0.4, "xrsi", 70),
            ("tx", "mz", 120, 0.5, "xrsi", 70), ("tx", "mz", 60, 0.5, "xret", 1.5), ("tx", "don", 20, 0.2, "xdev", 2.0),
            ("tx", "slope", (10, 3), 0.1, "xret", 2.0), ("tx", "mz", 20, 1.0, "xdev", 1.5)]
    refd = {V.key(g["cfg"]): g for g in ref["grid"]}
    L += ["", "== 2. 개별 설정(numpy 방식 / groupby 방식 / 원본) h10 초과, %p"]
    out["picked"] = {}
    for c in pick:
        k = V.key(c)
        Z = V.zones(Pn, c)
        rec = {}
        for p in V.PER:
            a = mine[k][p]
            b = V.score_pandas(Z, EX, rows[p], dates)
            dd = max(abs(a["S"] - b["S"]), *(abs(a["T"][t] - b["T"][t]) for t in V.TK))
            r0 = refd.get(k, {}).get(p) if p != "2024" else None
            if k == V.key(V.BASE) or k == V.key(V.WIN):
                lab = "옛칼만" if k == V.key(V.BASE) else "선택"
                r0 = r0 or None
            rec[p] = {"S": a["S"], "T": a["T"], "share": a["share"], "groupby_diff": dd,
                      "ref_S": (r0 or {}).get("S")}
            L.append(f"{k:<38} [{p}] S {pct(a['S'])} T1 {pct(a['T']['T1'])} T2 {pct(a['T']['T2'])} T3 {pct(a['T']['T3'])} "
                     f"| 비중 극상 {a['share'][2] * 100:4.1f} 상 {a['share'][1] * 100:4.1f} 애 {a['share'][0] * 100:4.1f} "
                     f"하 {a['share'][-1] * 100:4.1f} 극하 {a['share'][-2] * 100:4.1f} | groupby차 {dd * 100:.6f}"
                     + (f" | 원본 S {pct(r0['S'])}" if r0 else ""))
        out["picked"][k] = rec
    # 선택·옛칼만은 detail(2024 포함)과 대조, t값 포함
    L += ["", "== 2b. 선택·옛칼만 4구간 detail 대조(h10)"]
    for lab, c in (("선택", V.WIN), ("옛칼만", V.BASE)):
        D = ref["detail"][lab]["d"]
        Z = V.zones(Pn, c)
        for p in V.PER:
            a = V.score(Z, EX, rows[p])
            e = D[f"{p}·h10"]
            ts = {t: V.nw_t(a["ser"][t], H) for t in V.TK}
            diff = max(abs(a["S"] - e["S"]), *(abs(a["T"][t] - e["terms"][tmap[t]]) for t in V.TK))
            tdiff = max(abs(ts[t] - e["t"][tmap[t]]) for t in V.TK)
            L.append(f"{lab} [{p}] S {pct(a['S'])} (원본 {pct(e['S'])}) T1 {pct(a['T']['T1'])} t {ts['T1']:+.2f} "
                     f"T2 {pct(a['T']['T2'])} t {ts['T2']:+.2f} T3 {pct(a['T']['T3'])} t {ts['T3']:+.2f} | 최대차 {diff * 100:.5f}%p, t차 {tdiff:.4f}")
            out.setdefault("detail", {}).setdefault(lab, {})[p] = {"S": a["S"], "T": a["T"], "t": ts, "diff": diff}

    # ---------- 3. K1·K2 분포
    L += ["", "== 3. 자격 설정 분포(내 계산 자격 기준)"]
    dist = {}
    for p in V.PER:
        for t in V.TK:
            v = np.array([mine[k][p]["T"][t] for k in ranked])
            v = v[np.isfinite(v)]
            dist[f"{p}|{t}"] = {"pos": float(np.mean(v > 0)), "med": float(np.median(v)), "n": int(len(v))}
            L.append(f"[{p}] {t}: 양수 {np.mean(v > 0) * 100:5.1f}%  중앙 {pct(np.median(v))}  (n {len(v)})")
        v = np.array([mine[k][p]["S"] for k in ranked])
        L.append(f"[{p}] S 양수 {np.mean(v > 0) * 100:5.1f}%  중앙 {pct(np.median(v))}")
    out["dist"] = dist

    # ---------- 4. SOXX 경고 분할(K2 일부·K3)
    warn, src = V.soxx_warn(dates)
    # 원본 방식(주 첫 거래일 직전 거래일)과 같은지: zones_regime.json 값으로 확인
    reg = json.load(open(os.path.join(V.RES, "zones_regime.json"), encoding="utf-8"))
    name_of = {V.key(g["cfg"]): g["name"] for g in ref["grid"]}
    name_of[V.key(V.BASE)] = "옛 Kalman 5단계(일봉 환산)"
    L += ["", "== 4. SOXX 주간 경고 분할"]
    for p in V.PER:
        r = rows[p]
        L.append(f"  {p}: 경고 {warn[r].mean() * 100:.0f}% ({warn[r].sum()}일 / {len(r)}일)")
    split_res = {}
    rmax, rmax_at, ncmp = 0, None, 0
    for k in ranked + [V.key(V.BASE)]:
        c = mine[k]["cfg"]
        Z = V.zones(Pn, c)
        split_res[k] = {}
        for p in V.PER:
            for s, msk in (("평상", ~warn), ("경고", warn)):
                r = rows[p][msk[rows[p]]]
                sc = V.score(Z, EX, r)
                split_res[k][f"{p}|{s}"] = sc["T"]
                rr = reg.get(name_of[k], {}).get(p, {}).get(s)
                if rr:
                    for t in V.TK:
                        a, b = sc["T"][t], rr[tmap[t]]
                        if a is not None and b is not None and np.isfinite(a) and np.isfinite(b):
                            ncmp += 1
                            if abs(a - b) > rmax:
                                rmax, rmax_at = abs(a - b), (name_of[k], p, s, t)
    L.append(f"zones_regime.json 대조 최대 차이 {rmax * 100:.5f}%p {rmax_at} (비교 {ncmp}건)")
    L.append("자격 설정에서 항이 양수인 비율 / 중앙(%p)")
    rdist = {}
    for p in V.PER:
        for s in ("평상", "경고"):
            parts = []
            for t in V.TK:
                v = np.array([split_res[k][f"{p}|{s}"][t] for k in ranked])
                v = v[np.isfinite(v)]
                rdist[f"{p}|{s}|{t}"] = {"pos": float(np.mean(v > 0)), "med": float(np.median(v)), "n": int(len(v))}
                parts.append(f"{t} {np.mean(v > 0) * 100:5.1f}% {pct(np.median(v))} (n{len(v)})")
            L.append(f"[{p}·{s}] " + " | ".join(parts))
    for lab, c in (("선택", V.WIN), ("옛칼만", V.BASE)):
        L.append(f"{lab}: " + "  ".join(f"[{p}·{s}] T3 {pct(split_res[V.key(c)][f'{p}|{s}']['T3'])}"
                                        for p in V.PER for s in ("평상", "경고")))
    out["regime_dist"] = rdist
    out["regime_winner"] = split_res[V.key(V.WIN)]
    out["regime_base"] = split_res[V.key(V.BASE)]
    out["regime_ref_maxdiff"] = rmax
    # 경고 주의 실제 날짜 수·독립 구간 수(겹침 고려)
    for p in V.PER:
        r = rows[p][warn[rows[p]]]
        if len(r):
            runs = 1 + int(np.sum(np.diff(r) > 1))
            L.append(f"  {p} 경고 신호일 {len(r)}일, 연속 묶음 {runs}개, 첫 {dates[r[0]]} 끝 {dates[r[-1]]}")
    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(V.RES, "verify_indep1_recompute.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    with open(os.path.join(V.RES, "verify_indep1_recompute.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))


if __name__ == "__main__":
    main()
