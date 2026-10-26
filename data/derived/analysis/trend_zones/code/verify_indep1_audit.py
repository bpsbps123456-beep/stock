"""검증자 #1 — 미래정보·자료 점검(2026-10-11, 연구용). 원본 파일은 고치지 않는다.
A. 자르기 시험: 자료를 컷 날짜까지만 주고 다시 계산해도 컷 이전 단계가 같은가(내 구현 + zones.py 원본 코드 둘 다).
B. 미래 수익 정의·앞 구간 정리·SOXX 경고 시점(원천 날짜 ≤ t) 직접 확인.
C. 분할 흔적: 1일 |수익| > 40% 와 시가 갭이 단순 분할비(2·3·4·5·10·1/2…)와 맞는지.
D. 극단값 영향: DRUG(2024-10-15 +1446%) 제외, 날짜별 순위(분위)로 바꾼 초과, ±100% 자르기 → K1·K2·K3 분포 다시.
E. 지표 정의 민감도: RSI·ATR Wilder SMA 시작, 표준편차 ddof=0 → 1위·분포.
F. 사후 종목군: 2026 보유 이력에서 종목별 첫 편입일 → '그날 이미 편입된 종목만'으로 다시 계산.
G. 표본 구조: 자격 107개 중 서로 다른 극상방 정의 개수, 경고 주 T3가 몇 날짜·몇 종목일로 계산됐나.
실행: $env:PYTHONIOENCODING='utf-8'; py data/derived/analysis/trend_zones/code/verify_indep1_audit.py
결과: results/verify_indep1_audit.json, .txt
"""
import glob, importlib.util, json, os, sys, time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import verify_indep1_core as V  # noqa: E402

H = 10
L = []
OUT = {}


def pct(v):
    return f"{v * 100:+.2f}" if np.isfinite(v) else "  nan"


def elig_and_rank(mine):
    el = [k for k in mine if k != V.key(V.BASE) and V.eligible(mine[k]["앞"]["share"])]
    return sorted(el, key=lambda k: -mine[k]["앞"]["S"])


def run_all(Pn, EX, dates, cfgs, periods=("앞", "뒤", "2025", "2024"), warn=None):
    rows = {p: V.rows_of(dates, p, H) for p in periods}
    res = {}
    for c in cfgs:
        Z = V.zones(Pn, c)
        r = {p: V.score(Z, EX, rows[p]) for p in periods}
        if warn is not None:
            for p in periods:
                for s, m in (("평상", ~warn), ("경고", warn)):
                    rr = rows[p][m[rows[p]]]
                    r[f"{p}|{s}"] = V.score(Z, EX, rr)
        res[V.key(c)] = {"cfg": c, **r}
    return res


def summarize(res, label, with_regime=True, elig=None):
    el = sorted(elig, key=lambda k: -res[k]["앞"]["S"]) if elig is not None else elig_and_rank(res)
    d = {"n_elig": len(el), "winner": el[0] if el else None}
    w = V.key(V.WIN)
    line = [f"[{label}] 자격 {len(el)}개, 1위 {el[0] if el else None}"]
    if w in res:
        line.append(f"사전 1위(mz60·xrsi70) 앞 S {pct(res[w]['앞']['S'])} T1 {pct(res[w]['앞']['T']['T1'])} | 뒤 S {pct(res[w]['뒤']['S'])} "
                    f"T1 {pct(res[w]['뒤']['T']['T1'])}")
        d["win"] = {p: res[w][p]["T"] | {"S": res[w][p]["S"]} for p in ("앞", "뒤", "2025", "2024")}
    for p in ("앞", "뒤", "2025", "2024"):
        parts = []
        for t in V.TK:
            v = np.array([res[k][p]["T"][t] for k in el])
            v = v[np.isfinite(v)]
            d[f"{p}|{t}"] = float(np.mean(v > 0))
            parts.append(f"{t} {np.mean(v > 0) * 100:3.0f}%(중앙 {pct(np.median(v))})")
        line.append(f"  {p}: " + " ".join(parts))
    if with_regime:
        for p in ("앞", "뒤", "2025", "2024"):
            for s in ("평상", "경고"):
                parts = []
                for t in V.TK:
                    v = np.array([res[k][f"{p}|{s}"]["T"][t] for k in el])
                    v = v[np.isfinite(v)]
                    d[f"{p}|{s}|{t}"] = float(np.mean(v > 0))
                    parts.append(f"{t} {np.mean(v > 0) * 100:3.0f}%")
                wt3 = res[w][f"{p}|{s}"]["T"]["T3"] if w in res else np.nan
                line.append(f"  {p}·{s}: " + " ".join(parts) + f" | 사전1위 T3 {pct(wt3)}")
    L.extend(line)
    return d


def main():
    global L
    t0 = time.time()
    dates, syms, P = V.load()
    Pn = V.Panel(dates, syms, P)
    G = V.grid()
    CF = G + [V.BASE]
    F = Pn.fwd(H)
    EX = V.excess(F)
    warn, src = V.soxx_warn(dates)

    # ------------------------------------------------ A. 자르기 시험
    L.append("== A. 자르기 시험(컷 이후 자료를 지우고 다시 계산 → 컷 이전 단계 비교)")
    Zfull = {V.key(c): V.zones(Pn, c) for c in CF}
    spec = importlib.util.spec_from_file_location("zones_ro", os.path.join(HERE, "zones.py"))
    Zm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(Zm)                       # main()은 부르지 않음(읽기 전용)
    Of, Hf, Lf, Cf = Zm.load()
    Ifull = Zm.Ind(Of, Hf, Lf, Cf)
    zG = Zm.grid() + [Zm.BASELINE]
    # 원본 코드 전체 자료 단계 == 내 단계?
    col = [Ifull.syms.index(s) for s in syms]
    same_full = all(np.array_equal(Zm.zones(Ifull, c)[:, col], Zfull[V.key(c)]) for c in zG)
    L.append(f"원본 zones.py 단계 배열 == 내 단계 배열(433개 전부, 전 날짜): {same_full}")
    cuts = ["2024-03-28", "2024-10-15", "2025-06-30", "2026-02-27", "2026-05-13", "2026-08-14"]
    A = {}
    for cut in cuts:
        d2, s2, P2 = V.load(upto=cut)
        Pc = V.Panel(d2, s2, P2)
        n = len(d2)
        bad_m = 0
        for c in CF:
            Zc = V.zones(Pc, c)
            Zf = Zfull[V.key(c)][:n][:, [syms.index(s) for s in s2]]
            bad_m += int((Zc != Zf).sum())
        mask = [d <= cut for d in Ifull.dates]
        Ic = Zm.Ind(Of[mask], Hf[mask], Lf[mask], Cf[mask])
        bad_o = 0
        for c in zG:
            a = Zm.zones(Ic, c)
            b = Zm.zones(Ifull, c)[: a.shape[0]]
            bad_o += int((a != b).sum())
        A[cut] = {"mine_diff_cells": bad_m, "orig_diff_cells": bad_o, "n_dates": n, "n_syms": len(s2)}
        L.append(f"컷 {cut}: 날짜 {n}, 종목 {len(s2)} | 내 구현 다른 칸 {bad_m} | 원본 zones.py 다른 칸 {bad_o} (433개 설정 합계)")
    OUT["A_truncation"] = A
    # 미래 수익: 원본 Fwd vs 직접 O[t+1+h]/O[t+1]-1
    Fo = Zm.Fwd(Ifull, H)
    O = Pn.O
    direct = np.full(O.shape, np.nan)
    direct[: -(1 + H)] = O[1 + H:] / O[1:-H] - 1
    direct = np.where(Pn.valid, direct, np.nan)
    raw_o = Fo.raw[:, col]
    same_f = np.allclose(np.nan_to_num(raw_o, nan=-999), np.nan_to_num(direct, nan=-999), atol=1e-12)
    L.append(f"원본 Fwd.raw == O[t+1+h]/O[t+1]-1 (유효 칸만, NaN 위치 포함): {same_f}")
    # 유효성: 130봉 전에는 단계가 없는가
    first_valid = [int(np.argmax(Pn.valid[:, j])) - Pn.start[j][0] + 1 for j in range(len(syms)) if Pn.valid[:, j].any()]
    never = [syms[j] for j in range(len(syms)) if not Pn.valid[:, j].any()]
    L.append(f"종목별 첫 단계 봉 번호 최소 {min(first_valid)}, 최대 {max(first_valid)} (130이어야 함), 끝까지 130봉 못 채운 종목 {never}")
    # 같은 날 평균: 원본 Fwd.ex 와 내 EX
    same_ex = np.allclose(np.nan_to_num(Fo.ex[:, col], nan=-999), np.nan_to_num(EX, nan=-999), atol=1e-12)
    L.append(f"초과 수익(같은 날 유효 종목 평균 차감) 원본 == 내 계산: {same_ex}")
    nb = np.isfinite(EX).sum(1)
    L.append(f"날짜별 평균에 들어간 종목 수: 2024 {int(np.median(nb[V.rows_of(dates, '2024', H)]))}, 2025 "
             f"{int(np.median(nb[V.rows_of(dates, '2025', H)]))}, 앞 {int(np.median(nb[V.rows_of(dates, '앞', H)]))}, "
             f"뒤 {int(np.median(nb[V.rows_of(dates, '뒤', H)]))} (중앙)")

    # ------------------------------------------------ B. 앞 정리·경고 시점
    ra = V.rows_of(dates, "앞", H)
    last_idx = max(i for i, d in enumerate(dates) if d <= "2026-05-29")
    L += ["", "== B. 시점",
          f"앞 신호 {len(ra)}일, 마지막 신호 {dates[ra[-1]]}, 그 청산 시가일 {dates[ra[-1] + 1 + H]} ≤ {dates[last_idx]}: {ra[-1] + 1 + H <= last_idx}",
          f"뒤 첫 신호 {dates[V.rows_of(dates, '뒤', H)[0]]} (앞 청산 뒤에 시작: 겹침 없음)"]
    rb = V.rows_of(dates, "뒤", H)
    L.append(f"뒤 마지막 유효 신호 {dates[rb[np.isfinite(EX[rb]).any(1)][-1]]} (10/9 시가까지 청산 가능한 신호만 남음)")
    viol = [(dates[i], src[i]) for i in range(len(dates) - 1) if src[i] is not None and src[i] > dates[i]]
    L.append(f"SOXX 경고 원천 날짜 > 신호일 t 인 경우 {len(viol)}건 (0이어야 미래정보 없음)")
    lagd = [np.busday_count(src[i], dates[i]) for i in range(len(dates) - 1) if src[i] is not None and dates[i] >= "2024-01-01"]
    L.append(f"경고 원천 날짜가 t보다 앞선 영업일 수: 0일 {np.mean(np.array(lagd) == 0) * 100:.0f}%, 최대 {max(lagd)}일 (주 단위라 평균 {np.mean(lagd):.1f}일 묵은 정보)")
    OUT["B"] = {"purge_ok": bool(ra[-1] + 1 + H <= last_idx), "soxx_violations": len(viol)}

    # ------------------------------------------------ C. 분할 흔적
    L += ["", "== C. 분할 흔적(1일 |종가 수익| > 40% 또는 시가 갭 > 40%)"]
    C_, Op = Pn.C, Pn.O
    ratios = [2, 3, 4, 5, 8, 10, 15, 20, 1.5]
    sus = []
    cnt = 0
    for j, s in enumerate(syms):
        for i in range(1, len(dates)):
            pc, o, c = C_[i - 1, j], Op[i, j], C_[i, j]
            if not (np.isfinite(pc) and np.isfinite(o)):
                continue
            if abs(c / pc - 1) > 0.4 or abs(o / pc - 1) > 0.4:
                cnt += 1
                g = o / pc
                for r in ratios:
                    for rr in (r, 1 / r):
                        if abs(g / rr - 1) < 0.03:
                            # 분할이면 이후 가격이 새 수준에 머문다: 다음 5일 종가/전일 종가
                            nxt = C_[i:i + 5, j] / pc
                            sus.append((s, dates[i], round(pc, 3), round(o, 3), round(c, 3), round(rr, 3),
                                        [round(v, 2) for v in nxt]))
    L.append(f"후보 {cnt}건 중 시가 갭이 단순 분할비 ±3%와 맞는 것 {len(sus)}건:")
    for x in sus:
        L.append(f"  {x}")
    # 알려진 분할(2024)이 반영됐는지: 직전·당일 종가 비
    known = {"NVDA": "2024-06-10", "AVGO": "2024-07-15", "SMCI": "2024-10-01", "LRCX": "2024-10-03", "ANET": "2024-12-04",
             "CMG": "2024-06-26", "WMT": "2024-02-26", "TSM": None, "ORCL": None, "IREN": None}
    for s, d in known.items():
        if d and s in syms and d in dates:
            i, j = dates.index(d), syms.index(s)
            L.append(f"  알려진 분할 {s} {d}: 전일 종가 {C_[i - 1, j]:.2f} → 당일 시가 {Op[i, j]:.2f} (비 {Op[i, j] / C_[i - 1, j]:.3f}, 1에 가까우면 조정됨)")
    # 가장 큰 10일 미래 수익
    L.append("10일 미래 수익 상위(신호일 기준, 유효 칸):")
    flat = [(F[i, j], syms[j], dates[i]) for i in range(len(dates)) for j in range(len(syms)) if np.isfinite(F[i, j])]
    flat.sort(reverse=True)
    top = flat[:12]
    for v, s, d in top:
        L.append(f"  {s} {d} {v * 100:+.0f}%")
    L.append("하위: " + ", ".join(f"{s} {d} {v * 100:+.0f}%" for v, s, d in flat[-5:]))

    # ------------------------------------------------ D. 극단값 영향
    L += ["", "== D. 극단값 영향(K1·K2·K3 분포를 다시)"]
    base = run_all(Pn, EX, dates, CF, warn=warn)
    el0 = elig_and_rank(base)
    OUT["D_base"] = summarize(base, "기준(원본과 같음)")
    # D1 DRUG 제외
    d2, s2, P2 = V.load(drop=("DRUG",))
    P_ = V.Panel(d2, s2, P2)
    w2, _ = V.soxx_warn(d2)
    r1 = run_all(P_, V.excess(P_.fwd(H)), d2, CF, warn=w2)
    OUT["D_noDRUG"] = summarize(r1, "DRUG 제외(자격은 다시 판정)")
    OUT["D_noDRUG_sameelig"] = summarize(r1, "DRUG 제외(원래 자격 107개 고정)", elig=[k for k in el0])
    # D2 날짜별 순위(0~1 분위 - 0.5)
    rk = pd.DataFrame(F).rank(axis=1, pct=True).to_numpy()
    EXr = np.where(np.isfinite(F), rk - 0.5, np.nan)
    r2 = run_all(Pn, EXr, dates, CF, warn=warn)
    OUT["D_rank"] = summarize(r2, "초과 대신 날짜별 순위(분위, 자격 107개 고정)", elig=el0)
    # D3 ±100% 자르기(로그 아님)
    Fc = np.clip(F, -0.5, 1.0)
    r3 = run_all(Pn, V.excess(Fc), dates, CF, warn=warn)
    OUT["D_clip"] = summarize(r3, "미래 수익 -50%~+100%로 자르기(자격 107개 고정)", elig=el0)
    # D4 날짜별 5%·95% 분위로 눌러 담기(winsorize)
    lo = np.nanpercentile(np.where(np.isfinite(F), F, np.nan), 5, axis=1, keepdims=True)
    hi = np.nanpercentile(np.where(np.isfinite(F), F, np.nan), 95, axis=1, keepdims=True)
    Fw = np.where(np.isfinite(F), np.clip(F, lo, hi), np.nan)
    r4 = run_all(Pn, V.excess(Fw), dates, CF, warn=warn)
    OUT["D_winsor"] = summarize(r4, "날짜별 5/95% winsorize(자격 107개 고정)", elig=el0)
    # 2024 원래 값의 쏠림: 사전 1위·옛 칼만의 날짜별 T1 중 상위 절댓값 기여
    for lab, c in (("옛칼만", V.BASE), ("이격 EMA35·rms60·a0.5·b1.5", ("dev", 35, "rms60", 0.5, 1.5))):
        Z = V.zones(Pn, c)
        r = V.rows_of(dates, "2024", H)
        sc = V.score(Z, EX, r)
        t1 = sc["ser"]["T1"]
        order = np.argsort(-np.abs(np.nan_to_num(t1)))
        top5 = order[:5]
        L.append(f"{lab} 2024 T1 {pct(np.nanmean(t1))}: 절댓값 상위 5일 " + ", ".join(f"{dates[r[i]]} {t1[i] * 100:+.0f}" for i in top5)
                 + f" → 이 5일 빼면 {pct(np.nanmean(np.delete(t1, top5)))}")
        jd = syms.index("DRUG")
        zz = Z[r, jd]
        sel = [(dates[r[i]], int(zz[i]), F[r[i], jd]) for i in range(len(r)) if np.isfinite(F[r[i], jd]) and F[r[i], jd] > 1]
        L.append(f"   DRUG 10일 수익 >100% 신호일 {len(sel)}일, 그날 단계 분포 {pd.Series([s[1] for s in sel]).value_counts().to_dict()}")

    # ------------------------------------------------ E. 지표 정의 민감도
    L += ["", "== E. 지표 정의 민감도(자격은 다시 판정)"]
    for lab, kw in (("RSI·ATR Wilder SMA 시작", {"seed": "rsi_sma"}), ("표준편차 ddof=0", {"ddof": 0})):
        Pe = V.Panel(dates, syms, P, **kw)
        re_ = run_all(Pe, V.excess(Pe.fwd(H)), dates, CF, warn=warn)
        OUT[f"E_{lab}"] = summarize(re_, lab)

    # ------------------------------------------------ F. 사후 종목군
    L += ["", "== F. 종목군(2026 보유 이력 전체로 고름) — 첫 편입일 기준 다시 계산"]
    tick = json.load(open(os.path.join(V.ROOT, "data/derived/analysis/rsi_breakout/cache/universe.json"), encoding="utf-8"))["tickers"]
    first = {}
    files = sorted(glob.glob(os.path.join(V.ROOT, "data/normalized/etf_holdings/2026-*_*.json")))
    for fp in files:
        d = os.path.basename(fp)[:10]
        js = json.load(open(fp, encoding="utf-8"))
        for row in js.get("holdings", []):
            y = tick.get(str(row.get("code", "")).strip())
            if y and y not in first:
                first[y] = d
    miss = [s for s in syms if s not in first]
    fs = pd.Series({s: first.get(s, "없음") for s in syms})
    L.append(f"스냅샷 {len(files)}개({os.path.basename(files[0])[:10]}~{os.path.basename(files[-1])[:10]}), 첫 편입일 못 찾은 종목 {len(miss)}개 {miss[:10]}")
    L.append("첫 편입 월 분포: " + str(fs[fs != '없음'].str[:7].value_counts().sort_index().to_dict()))
    late = sorted([s for s in syms if first.get(s, "9") > "2026-05-29"])
    L.append(f"6월 이후 처음 편입 {len(late)}개: {late}")
    # 앞 기간 날짜별 '그날 이미 편입' 마스크
    pit = np.zeros(Pn.C.shape, bool)
    for j, s in enumerate(syms):
        fd = first.get(s)
        if fd:
            pit[:, j] = np.array([d >= fd for d in dates])
    ex_ante = np.array([first.get(s, "9") <= "2026-01-02" for s in syms])
    for lab, M in (("2026-01-02 이미 편입 종목만", np.broadcast_to(ex_ante[None, :], Pn.C.shape)),
                   ("신호일에 이미 편입된 종목만(시점 일치)", pit)):
        Fm = np.where(M, F, np.nan)
        EXm = V.excess(Fm)
        rm = {}
        for c in CF:
            Z = V.zones(Pn, c)
            Z = np.where(M, Z, -9).astype(np.int8)
            rr = {p: V.score(Z, EXm, V.rows_of(dates, p, H)) for p in ("앞", "뒤")}
            rm[V.key(c)] = rr
        v1 = np.array([rm[k]["앞"]["T"]["T1"] for k in el0])
        v1b = np.array([rm[k]["뒤"]["T"]["T1"] for k in el0])
        w = rm[V.key(V.WIN)]
        el_m = sorted([k for k in rm if k != V.key(V.BASE) and V.eligible(rm[k]["앞"]["share"])], key=lambda k: -rm[k]["앞"]["S"])
        L.append(f"[{lab}] 종목 수(앞 중앙) {int(np.median(np.isfinite(EXm[V.rows_of(dates, '앞', H)]).sum(1)))} | 사전 1위 앞 S {pct(w['앞']['S'])} "
                 f"T1 {pct(w['앞']['T']['T1'])} / 뒤 S {pct(w['뒤']['S'])} T1 {pct(w['뒤']['T']['T1'])} | 107개 T1 양수 앞 {np.mean(v1 > 0) * 100:.0f}% "
                 f"뒤 {np.mean(v1b > 0) * 100:.0f}% | 이 종목군으로 다시 고른 1위 {el_m[0] if el_m else None} (자격 {len(el_m)}개)")
        OUT[f"F_{lab}"] = {"win": {p: w[p]["T"] | {"S": w[p]["S"]} for p in w}, "T1pos_앞": float(np.mean(v1 > 0)),
                          "T1pos_뒤": float(np.mean(v1b > 0)), "winner": el_m[0] if el_m else None}
    # 6월 이후 편입 종목의 앞 구간 단계·수익
    jl = [syms.index(s) for s in late]
    if jl:
        Zw = V.zones(Pn, V.WIN)
        r = V.rows_of(dates, "앞", H)
        zz = Zw[np.ix_(r, jl)]
        ee = EX[np.ix_(r, jl)]
        ok = (zz != -9) & np.isfinite(ee)
        dist = {V_: float(((zz == V_) & ok).sum() / ok.sum()) for V_ in V.ZN}
        L.append(f"6월 이후 편입 {len(jl)}개의 앞 구간(사전 1위 단계): 비중 " + " ".join(f"{z}:{dist[z] * 100:.0f}%" for z in V.ZN)
                 + f", 평균 초과 {np.nanmean(np.where(ok, ee, np.nan)) * 100:+.2f}%")

    # ------------------------------------------------ G. 표본 구조
    L += ["", "== G. 표본 구조"]
    masks = {}
    for k in el0:
        Z = V.zones(Pn, base[k]["cfg"])
        h = hash((Z == 2).tobytes())
        masks.setdefault(h, []).append(k)
    L.append(f"자격 107개 중 서로 다른 극상방(+2) 정의 {len(masks)}개: " + "; ".join(
        f"{len(v)}개({v[0].split('|')[-2] if v[0].startswith('tx') else '이격'}…)" for v in sorted(masks.values(), key=len, reverse=True)[:6]))
    fam = pd.Series([("이격" if k.startswith("dev") else k.split("|")[4] + k.split("|")[5]) for k in el0]).value_counts()
    L.append(f"자격 107개의 과열 축 구성: {fam.to_dict()}")
    # 경고 주 T3: 사전 1위가 몇 날짜·몇 극상방 종목일로 계산됐나
    Zw = V.zones(Pn, V.WIN)
    for p in ("앞", "뒤", "2025", "2024"):
        rp = V.rows_of(dates, p, H)
        for s, m in (("평상", ~warn), ("경고", warn)):
            r = rp[m[rp]]
            sc = V.score(Zw, EX, r)
            nd = int(np.isfinite(sc["ser"]["T3"]).sum())
            n2 = int(((Zw[r] == 2) & np.isfinite(EX[r])).sum())
            blocks = 1 + int(np.sum(np.diff(r) > 1)) if len(r) else 0
            L.append(f"  사전 1위 {p}·{s}: 신호일 {len(r)}일(연속 묶음 {blocks}), T3 계산 날짜 {nd}, 극상방 종목일 {n2}, "
                     f"T3 {pct(sc['T']['T3'])} t {V.nw_t(sc['ser']['T3'], H):+.2f}")
    # 107개 T3의 경고·평상 차이를 설정 간 상관으로: 극상방 정의별 평균
    OUT["G_masks"] = len(masks)
    txt = "\n".join(L) + f"\n\n(계산 {time.time() - t0:.0f}초)"
    print(txt)
    with open(os.path.join(V.RES, "verify_indep1_audit.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    with open(os.path.join(V.RES, "verify_indep1_audit.json"), "w", encoding="utf-8") as f:
        json.dump(OUT, f, ensure_ascii=False, indent=1, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))


if __name__ == "__main__":
    main()
