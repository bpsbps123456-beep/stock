"""검증자 #1 보조 (2026-10-10, 연구용).
(a) zones.py 자체의 단계 배열을 칸 단위로 내 독자 구현과 비교하고, zones.py Ind에 '3/2 이후 가격 삭제' 자르기 검사를 직접 건다
    (읽기 전용 import, 파일은 고치지 않음).
(b) 극단 수익 영향: 날짜마다 10일 수익을 횡단 1·99% 분위로 자르기(winsorize), 극단 종목 5개(DRUG·LUNR·SERV·NVTS·PEPG) 제외
    → 자격 107개의 국면별 T1·T2·T3 양수 비율과 1위 설정 T3를 다시 낸다.
실행: $env:PYTHONIOENCODING='utf-8'; py data/derived/analysis/trend_zones/code/verify_1_robust.py
결과: results/verify_1_robust.json, verify_1_robust.txt
"""
import json, os, sys, warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import verify_1_recompute as V  # noqa: E402  (내 구현)
import zones as Zo  # noqa: E402  (원 연구 코드, 읽기 전용)

warnings.filterwarnings("ignore", category=RuntimeWarning)
K = V.K
P = lambda v: f"{v * 100:+.2f}"


def main():
    lines, out = [], {}
    dates, syms, O, H, L, C = V.load()
    I = V.Ind(O, H, L, C)
    G = V.my_grid()
    F10 = V.fwd(I, 10)
    R = {p: V.rows_of(dates, p, 10) for p in V.PER}
    res = {V.ckey(c): V.agg(V.daily(V.zone(I, c), F10[1], F10[0]), R["앞"]) for c in G}
    E = [c for c in G if V.elig(res[V.ckey(c)]["share"])]
    E.sort(key=lambda c: -res[V.ckey(c)]["S"])
    win = E[0]

    # (a) 칸 단위 비교 + zones.py 자르기 검사
    Ot, Ht, Lt, Ct = Zo.load()
    assert list(Ct.index) == dates and list(Ct.columns) == syms
    IZ = Zo.Ind(Ot, Ht, Lt, Ct)
    cell = 0
    for c in G + [V.BASE]:
        cell += int((Zo.zones(IZ, c) != V.zone(I, c)).sum())
    fz = Zo.Fwd(IZ, 10)
    fdiff = float(np.nanmax(np.abs(fz.raw - F10[0])))
    nanmis = int((np.isfinite(fz.raw) != np.isfinite(F10[0])).sum())
    cut = dates.index("2026-03-02")
    trunc = [X.copy() for X in (Ot, Ht, Lt, Ct)]
    for X in trunc:
        X.iloc[cut + 1:] = np.nan
    IZt = Zo.Ind(*trunc)
    tmis = 0
    for c in G + [Zo.BASELINE]:
        tmis += int((Zo.zones(IZ, c)[: cut + 1] != Zo.zones(IZt, c)[: cut + 1]).sum())
    lines += ["== (a) zones.py 직접 점검",
              f"432개+옛칼만 단계 배열 칸 불일치(zones.py vs 독자 구현): {cell}",
              f"10일 성과 배열 최대 차 {fdiff:.2e}, NaN 위치 불일치 {nanmis}",
              f"zones.py Ind에 2026-03-02 이후 가격 삭제 후 3/2까지 단계 불일치(433개 설정): {tmis}"]
    out["칸비교"] = {"단계_불일치": cell, "성과_최대차": fdiff, "성과_NaN불일치": nanmis, "zones_py_자르기_불일치": tmis}

    # 경고 표시
    warn = V.soxx_warn(dates)[0]

    def regime_table(raw, ex, cfgs, tag):
        rows = []
        tab = {}
        Ds = {V.ckey(c): V.daily(V.zone(I, c), ex, raw) for c in cfgs}
        for p in V.PER:
            r = R[p]
            for s, m in (("평상", ~warn[r]), ("경고", warn[r])):
                rr = r[m]
                parts = []
                for t in K:
                    v = np.array([V.nanmean(Ds[V.ckey(c)]["term"][t][rr]) for c in E])
                    v = v[np.isfinite(v)]
                    tab[f"{p}·{s}·{t}"] = float((v > 0).mean())
                    parts.append(f"{t[:2]} {np.mean(v > 0) * 100:3.0f}%")
                wv = {t: V.nanmean(Ds[V.ckey(win)]["term"][t][rr]) for t in K}
                tab[f"{p}·{s}·1위"] = wv
                rows.append(f"[{tag} {p}·{s}] " + " ".join(parts) + " | 1위 " + " ".join(f"{t[:2]} {P(wv[t])}" for t in K))
            # 전체 구간(국면 무시)
            v = {t: np.array([V.nanmean(Ds[V.ckey(c)]["term"][t][r]) for c in E]) for t in K}
            rows.append(f"[{tag} {p}·전체] " + " ".join(f"{t[:2]} {np.mean(v[t][np.isfinite(v[t])] > 0) * 100:3.0f}%" for t in K)
                        + f" | 1위 S {P(V.agg(Ds[V.ckey(win)], r)['S'])}")
        return rows, tab

    # (b1) 날짜별 1·99% 분위 자르기
    raw = F10[0].copy()
    lo = np.nanquantile(raw, 0.01, axis=1, keepdims=True)
    hi = np.nanquantile(raw, 0.99, axis=1, keepdims=True)
    rw = np.clip(raw, lo, hi)
    rw = np.where(np.isfinite(raw), rw, np.nan)
    exw = rw - np.nanmean(rw, axis=1, keepdims=True)
    r1, t1 = regime_table(rw, exw, E + [win], "윈저1%")
    # (b2) 극단 종목 제외
    drop = [s for s in ("DRUG", "LUNR", "SERV", "NVTS", "PEPG") if s in syms]
    mask = np.array([s not in drop for s in syms])
    rd = np.where(mask[None, :], raw, np.nan)
    exd = rd - np.nanmean(rd, axis=1, keepdims=True)
    r2, t2 = regime_table(rd, exd, E + [win], "5종목제외")
    # 원본(대조)
    r0, t0 = regime_table(raw, F10[1], E + [win], "원본")
    lines += ["", "== (b) 극단 수익 영향(h10, 자격 107개 중 항 양수 비율 | 1위 설정 항)"] + r0 + [""] + r1 + [""] + r2
    out["원본"], out["윈저1%"], out["5종목제외"] = t0, t1, t2

    # (c) 가장 큰 단일 날짜 기여: 1위 설정 T3 일별 값 상위
    D = V.daily(V.zone(I, win), F10[1], F10[0])
    t3 = D["term"][K[2]]
    lines += ["", "== (c) 1위 설정 T3 일별 극값(전 구간 신호일)"]
    allr = np.concatenate([R[p] for p in V.PER])
    order = allr[np.argsort(np.nan_to_num(t3[allr], nan=0))]
    for i in list(order[:5]) + list(order[-5:]):
        lines.append(f"  {dates[i]} 경고={bool(warn[i])} T3 {P(t3[i])} (극상방 {D['cnt'][2][i]}종목, 상방 {D['cnt'][1][i]}종목)")

    # (d) 날짜 평균 대신 (i) 극상방·상방이 각각 3종목 이상인 날만, (ii) 칸 가중(모든 종목-날짜를 한데 모은 평균)
    lines += ["", "== (d) 표본 수 보정: T3 양수 비율(자격 107개) | 1위 T3"]
    ex = F10[1]
    Zs = {V.ckey(c): V.zone(I, c) for c in E}
    Ds = {k: V.daily(Z, ex, F10[0]) for k, Z in Zs.items()}
    dtab = {}

    def t3_min(k, rr, nmin):
        D_ = Ds[k]
        ok = (D_["cnt"][2][rr] >= nmin) & (D_["cnt"][1][rr] >= nmin)
        return V.nanmean(D_["term"][K[2]][rr][ok]), int(ok.sum())

    def t3_pool(k, rr):
        Z = Zs[k][rr]
        e = ex[rr]
        f = np.isfinite(e)
        a = e[(Z == 1) & f]
        b = e[(Z == 2) & f]
        return float(a.mean() - b.mean()) if len(a) and len(b) else float("nan")

    kw = V.ckey(win)
    for p in V.PER:
        r = R[p]
        for s, m in (("평상", ~warn[r]), ("경고", warn[r])):
            rr = r[m]
            v3 = np.array([t3_min(V.ckey(c), rr, 3)[0] for c in E]); v3 = v3[np.isfinite(v3)]
            vp = np.array([t3_pool(V.ckey(c), rr) for c in E]); vp = vp[np.isfinite(vp)]
            w3, n3 = t3_min(kw, rr, 3)
            wp = t3_pool(kw, rr)
            dtab[f"{p}·{s}"] = {"3종목이상_양수비율": float((v3 > 0).mean()), "칸가중_양수비율": float((vp > 0).mean()),
                                "1위_3종목이상": w3, "1위_3종목이상_날짜": n3, "1위_칸가중": wp}
            lines.append(f"[{p}·{s}] 3종목 이상 날만 {np.mean(v3 > 0) * 100:3.0f}% | 칸 가중 {np.mean(vp > 0) * 100:3.0f}% "
                         f"| 1위 3종목↑ {P(w3)} ({n3}일) 칸가중 {P(wp)}")
    out["표본보정"] = dtab

    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(V.RES, "verify_1_robust.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    with open(os.path.join(V.RES, "verify_1_robust.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))


if __name__ == "__main__":
    main()
