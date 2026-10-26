"""검증 #2 보강 (2026-10-11, 연구용): K3 '극상방 = 팔 자리는 SOXX 경고 주에만' 반박 시도 — verify_k3_attack.py에 없는 시험.
zones.py·zones_regime.py·verify_k3_attack.py는 고치지 않는다(지표·격자만 불러 씀).

0) 독립 재구현: 1위(mz60≥0.5 + RSI14≥70)를 원자료에서 다시 계산(RSI는 트레이딩뷰식 SMA 시드), 경고도 따로 계산 → zones_regime.json과 비교.
1) 새 보류 구간 2023 하반기(2023-07-13~12-29): 누구도 본 적 없는 구간. 격자 107개·1위의 T3 평상/경고, h 5·10·20.
2) 섹터 중립: 같은 날 같은 섹터 평균을 뺀 10일 수익으로 T3. 경고 주 극상방이 특정 섹터(비반도체·원전 등) 몰림인지.
3) SOXX 베타 잔차 + 섹터 중립(가장 엄격).
4) 3주·5주 달력 블록 부트스트랩(10일 수익 겹침 반영)으로 경고-평상 차 CI. 묶음 = 4구간 전체, 검증만 = 앞 제외.
5) 종목·섹터 몰림: 1위 극상방(경고 주) 서로 다른 종목 수, 상위 5종목 비중, 한 종목·한 섹터씩 빼기.
6) 연속값 단조성: 결정 종가의 SOXX/SMA20 - 1 다섯 구간별 T3. 문턱 하나에만 걸린 착시인지.
7) 사후 분해(설명용, 매매 근거 아님): 다음 10일 SOXX가 오른 경우/내린 경우로 나눈 T3. 반등장 모멘텀 붕괴인지.
8) 극상방 정의별(서로 다른 극상방 정의 단위) 경고-평상 차 양수 비율 — 107개가 독립 표본이 아님을 반영.
실행: $env:PYTHONIOENCODING='utf-8'; py data/derived/analysis/trend_zones/code/verify_2_k3_extra.py
결과: results/verify_2_k3_extra.txt, verify_2_k3_extra.json
"""
import json, os, sys, warnings
from datetime import date

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import zones as Zm  # noqa: E402
import zones_regime as ZR  # noqa: E402

sys.path.insert(0, Zm.ROOT)
warnings.filterwarnings("ignore", category=RuntimeWarning)
SEED = 20261011
NB = 2000
PER = dict(Zm.PERIODS)
PER["2023하"] = ("2023-07-13", "2023-12-29", False)
pct = lambda v: "   nan" if v is None or not np.isfinite(v) else f"{v * 100:+6.2f}"


# ---------------------------------------------------------------- 0. 독립 재구현
def rsi_tv(c: np.ndarray, n=14):
    """트레이딩뷰 ta.rsi: 첫 n개 변화의 단순평균으로 시드한 RMA."""
    out = np.full(len(c), np.nan)
    d = np.diff(c)
    ok = np.where(np.isfinite(c))[0]
    if len(ok) < n + 2:
        return out
    s = ok[0]
    d = np.diff(c[s:])
    if not np.isfinite(d).all():          # 중간 결측이면 그 뒤 구간만
        bad = np.where(~np.isfinite(d))[0][-1]
        s = s + bad + 1
        d = np.diff(c[s:])
        if len(d) < n + 1:
            return out
    up, dn = np.clip(d, 0, None), np.clip(-d, 0, None)
    au, ad = up[:n].mean(), dn[:n].mean()
    for k in range(n, len(d) + 1):
        if k > n:
            au = (au * (n - 1) + up[k - 1]) / n
            ad = (ad * (n - 1) + dn[k - 1]) / n
        out[s + k] = 100.0 if ad == 0 else 100 - 100 / (1 + au / ad)
    return out


def winner_indep(O, C):
    Cn = C.to_numpy()
    rsi = np.column_stack([rsi_tv(Cn[:, j]) for j in range(Cn.shape[1])])
    lr = np.log(C).diff()
    mz = (np.log(C / C.shift(60)) / (lr.rolling(60, min_periods=60).std() * np.sqrt(60))).to_numpy()
    valid = ((C.notna().cumsum() >= Zm.MIN_BARS) & C.notna() & O.notna()).to_numpy()
    z = np.select([rsi >= 70, rsi <= 30, mz >= 0.5, mz <= -0.5], [2, -2, 1, -1], 0)
    ok = valid & np.isfinite(rsi) & np.isfinite(mz)
    return np.where(ok, z, -9).astype(np.int8)


def warn_indep(dates, soxx_close: pd.Series):
    sma = soxx_close.rolling(20, min_periods=20).mean()
    below = (soxx_close < sma)
    sd = list(soxx_close.index)
    pos = {d: i for i, d in enumerate(sd)}
    first = {}
    for d in sd:
        wk = date.fromisoformat(d).isocalendar()[:2]
        first.setdefault(wk, d)
    out = np.zeros(len(dates), dtype=bool)
    dist = np.full(len(dates), np.nan)
    for i in range(len(dates) - 1):
        wk = date.fromisoformat(dates[i + 1]).isocalendar()[:2]
        d0 = first.get(wk)
        if d0 is None or pos[d0] == 0:
            continue
        p = sd[pos[d0] - 1]
        out[i] = bool(below[p])
        dist[i] = soxx_close[p] / sma[p] - 1 if np.isfinite(sma[p]) else np.nan
    return out, dist


# ---------------------------------------------------------------- 공통
def rows_of(dates, p, h):
    a, b, purge = PER[p]
    idx = [i for i, d in enumerate(dates) if a <= d <= b]
    if purge:
        end = max(idx)
        idx = [i for i in idx if i + 1 + h <= end]
    return np.array(idx)


def t3_daily(Z, V):
    """날짜별 상방 평균 - 극상방 평균(둘 다 있어야). V: 날짜×종목."""
    have = np.isfinite(V) & (Z != -9)
    out = {}
    for z in (1, 2):
        M = have & (Z == z)
        n = M.sum(axis=1)
        out[z] = np.where(n > 0, np.where(M, V, 0).sum(axis=1) / np.maximum(n, 1), np.nan)
    return out[1] - out[2], out


def nm(v, rows):
    if len(rows) == 0:
        return np.nan
    x = v[rows]
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else np.nan


def grid_stats(D, wi, rw, rn):
    """D: (설정, 날짜) T3. 1위 평상/경고/차, 격자 양수%/중앙."""
    tw = np.array([nm(D[k], rw) for k in range(D.shape[0])])
    tn = np.array([nm(D[k], rn) for k in range(D.shape[0])])
    g = slice(0, D.shape[0] - 0)
    f = lambda v: v[np.isfinite(v)]
    df = tw - tn
    return {"1위평상": tn[wi], "1위경고": tw[wi], "1위차": df[wi],
            "격자평상양수": float(np.mean(f(tn) > 0)), "격자경고양수": float(np.mean(f(tw) > 0)),
            "격자평상중앙": float(np.median(f(tn))), "격자경고중앙": float(np.median(f(tw))),
            "격자차중앙": float(np.median(f(df))), "격자차양수": float(np.mean(f(df) > 0)),
            "경고일": int(len(rw)), "평상일": int(len(rn))}


def fmt_gs(s):
    return (f"1위 {pct(s['1위평상'])}/{pct(s['1위경고'])} (차 {pct(s['1위차'])}) | 격자 평상양수 {s['격자평상양수'] * 100:3.0f}% "
            f"경고양수 {s['격자경고양수'] * 100:3.0f}% 중앙 {pct(s['격자평상중앙'])}/{pct(s['격자경고중앙'])} 차중앙 {pct(s['격자차중앙'])} "
            f"차양수 {s['격자차양수'] * 100:3.0f}% (경고 {s['경고일']}일/평상 {s['평상일']}일)")


def block_boot(D, wi, rows, warn, blk, rng, nb=NB):
    """달력 블록(blk: 날짜별 블록 id) 복원 추출. 차 = 경고 평균 - 평상 평균. (1위, 격자 중앙)."""
    ub = {}
    for r in rows:
        ub.setdefault(blk[r], []).append(r)
    B = [np.array(v) for v in ub.values()]
    o1, o2 = np.empty(nb), np.empty(nb)
    Dw = np.where(np.isfinite(D), D, 0.0)
    Dc = np.isfinite(D).astype(float)
    for k in range(nb):
        rr = np.concatenate([B[j] for j in rng.integers(0, len(B), len(B))])
        w, n = rr[warn[rr]], rr[~warn[rr]]
        mw = Dw[:, w].sum(1) / np.maximum(Dc[:, w].sum(1), 1)
        mn = Dw[:, n].sum(1) / np.maximum(Dc[:, n].sum(1), 1)
        mw[Dc[:, w].sum(1) == 0] = np.nan
        mn[Dc[:, n].sum(1) == 0] = np.nan
        d = mw - mn
        o1[k] = d[wi]
        o2[k] = np.nanmedian(d)
    q = lambda v: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)), float(np.nanmean(v <= 0)))
    return {"블록수": len(B), "1위": q(o1), "격자중앙": q(o2)}


# ---------------------------------------------------------------- 실행
def main():
    rng = np.random.default_rng(SEED)
    O, Hh, L_, C = Zm.load()
    I = Zm.Ind(O, Hh, L_, C)
    dates, syms = I.dates, I.syms
    nd = len(dates)
    res = json.load(open(os.path.join(Zm.OUT, "zones.json"), encoding="utf-8"))
    cfgs = [g["cfg"] for g in res["grid"] if g["eligible"]]
    names = [Zm.cname(c) for c in cfgs]
    wi = names.index(res["best"])
    F = {h: Zm.Fwd(I, h) for h in (5, 10, 20)}
    lines, J = [], {}

    # SOXX
    o = pd.read_parquet(ZR.SOXX_SRC, columns=["date", "sym", "Open", "Close"])
    s = o[o.sym == "SOXX"].copy()
    s["d"] = pd.to_datetime(s["date"]).dt.strftime("%Y-%m-%d")
    s = s.drop_duplicates("d").set_index("d").sort_index()
    sc, so = s["Close"].astype(float), s["Open"].astype(float).reindex(dates)
    warn, dist = warn_indep(dates, sc)
    warn0 = ZR.soxx_week_warning(dates)
    soxx_f = {h: (so.shift(-(1 + h)) / so.shift(-1) - 1).to_numpy() for h in (5, 10, 20)}

    # ---- 0. 독립 재구현
    Zi = winner_indep(O, C)
    Zg = Zm.zones(I, cfgs[wi])
    both = (Zi != -9) & (Zg != -9)
    agree = float((Zi[both] == Zg[both]).mean())
    zr = json.load(open(os.path.join(Zm.OUT, "zones_regime.json"), encoding="utf-8"))[res["best"]]
    lines.append(f"== 0. 독립 재구현: 경고 일치 {np.mean(warn == warn0) * 100:.2f}% · 1위 단계 일치 {agree * 100:.2f}% "
                 f"(RSI 트레이딩뷰식 시드, 유효칸 {both.sum()})")
    d3i, _ = t3_daily(Zi, F[10].ex)
    J["0"] = {"경고일치": float(np.mean(warn == warn0)), "단계일치": agree, "T3": {}}
    for p in ("앞", "뒤", "2025", "2024"):
        rp = rows_of(dates, p, 10)
        mine = (nm(d3i, rp[~warn[rp]]), nm(d3i, rp[warn[rp]]))
        ref = (zr[p]["평상"]["T3 상방-극상방"], zr[p]["경고"]["T3 상방-극상방"])
        J["0"]["T3"][p] = {"내 평상/경고": mine, "원본 평상/경고": ref}
        lines.append(f"  [{p}] 내 계산 T3 평상/경고 {pct(mine[0])}/{pct(mine[1])} | 원본 {pct(ref[0])}/{pct(ref[1])}")

    # ---- 섹터·베타 준비
    from desk import config
    from desk.strategies import sector_of
    sec = [sector_of(x, x) for x in syms]
    grp = []
    cnt = pd.Series(sec).value_counts()
    for x in sec:
        if x in config.SEMIS_SECTORS:
            grp.append("반도체")
        elif x.startswith("헬스케어"):
            grp.append("헬스케어")
        elif cnt[x] >= 5:
            grp.append(x)
        else:
            grp.append("기타")
    grp = np.array(grp)
    G = sorted(set(grp))
    gcount = {g: int((grp == g).sum()) for g in G}
    semis = grp == "반도체"

    def sector_neutral(V):
        out = np.full_like(V, np.nan)
        for g in G:
            m = grp == g
            sub = V[:, m]
            mu = np.nanmean(sub, axis=1, keepdims=True)
            n = np.isfinite(sub).sum(axis=1, keepdims=True)
            out[:, m] = np.where(n >= 3, sub - mu, np.nan)
        return out

    r = C.pct_change(fill_method=None)
    sr = sc.pct_change(fill_method=None).reindex(dates)
    cov = r.mul(sr, axis=0).rolling(60, min_periods=50).mean() - r.rolling(60, min_periods=50).mean().mul(
        sr.rolling(60, min_periods=50).mean(), axis=0)
    var = sr.rolling(60, min_periods=50).var(ddof=0)
    beta = cov.div(var, axis=0).to_numpy()
    V = {}
    for h in (5, 10, 20):
        raw = F[h].raw
        V[("초과", h)] = F[h].ex
        V[("섹터중립", h)] = sector_neutral(raw)
        resid = np.where(np.isfinite(beta), raw - beta * soxx_f[h][:, None], np.nan)
        V[("SOXX베타", h)] = resid
        V[("베타+섹터", h)] = sector_neutral(resid)

    # 전 설정 단계
    Zs = [Zm.zones(I, c) for c in cfgs]
    D = {}
    for key, Vv in V.items():
        D[key] = np.vstack([t3_daily(Z, Vv)[0] for Z in Zs])

    # ---- 1. 2023 하반기 보류 + 기존 4구간, 변형별
    lines += ["", "== 1·2·3. 구간 × 수익 정의 × h (T3 = 상방-극상방, 평상/경고). 2023하 = 처음 보는 구간"]
    J["구간"] = {}
    for h in (5, 10, 20):
        for kind in ("초과", "섹터중립", "SOXX베타", "베타+섹터"):
            if h != 10 and kind != "초과":
                continue
            for p in ("2023하", "앞", "뒤", "2025", "2024"):
                rp = rows_of(dates, p, h)
                st = grid_stats(D[(kind, h)], wi, rp[warn[rp]], rp[~warn[rp]])
                J["구간"][f"{kind}·h{h}·{p}"] = st
                lines.append(f"[{kind}·h{h}·{p:>5}] " + fmt_gs(st))

    # ---- 4. 블록 부트스트랩
    wk = np.array([date.fromisoformat(dates[min(t + 1, nd - 1)]).isocalendar()[:2] for t in range(nd)])
    wid = {w: i for i, w in enumerate(sorted({tuple(x) for x in wk}))}
    widx = np.array([wid[tuple(x)] for x in wk])
    pools = {"묶음(4구간)": np.unique(np.concatenate([rows_of(dates, p, 10) for p in ("앞", "뒤", "2025", "2024")])),
             "검증만(앞 제외)": np.unique(np.concatenate([rows_of(dates, p, 10) for p in ("뒤", "2025", "2024")])),
             "2023하": rows_of(dates, "2023하", 10),
             "뒤": rows_of(dates, "뒤", 10)}
    lines += ["", "== 4. 경고-평상 T3 차, 달력 블록 부트스트랩 2000회 95% CI·P(차≤0)"]
    J["부트"] = {}
    for pn, rp in pools.items():
        for kind in ("초과", "섹터중립", "SOXX베타", "베타+섹터"):
            st = grid_stats(D[(kind, 10)], wi, rp[warn[rp]], rp[~warn[rp]])
            out = [f"[{pn}·{kind}] 1위 차 {pct(st['1위차'])} 격자 차중앙 {pct(st['격자차중앙'])}"]
            J["부트"][f"{pn}·{kind}"] = {"점": st}
            for bw in (3, 5):
                b = block_boot(D[(kind, 10)], wi, rp, warn, widx // bw, rng)
                J["부트"][f"{pn}·{kind}"][f"{bw}주"] = b
                out.append(f"{bw}주블록({b['블록수']}) 1위 [{pct(b['1위'][0])},{pct(b['1위'][1])}] P {b['1위'][2]:.3f} "
                           f"격자 [{pct(b['격자중앙'][0])},{pct(b['격자중앙'][1])}] P {b['격자중앙'][2]:.3f}")
            lines.append(" | ".join(out))

    # ---- 5. 종목·섹터 몰림(1위, 묶음 4구간, 초과)
    rp = pools["묶음(4구간)"]
    rw, rn = rp[warn[rp]], rp[~warn[rp]]
    Zw = Zs[wi]
    ex = F[10].ex
    base = nm(t3_daily(Zw, ex)[0], rw) - nm(t3_daily(Zw, ex)[0], rn)
    top_w = (Zw[rw] == 2) & np.isfinite(ex[rw])
    per_sym = top_w.sum(axis=0)
    order = np.argsort(-per_sym)
    tot = per_sym.sum()
    top5 = [(syms[j], int(per_sym[j]), grp[j]) for j in order[:8]]
    nd_sym = int((per_sym > 0).sum())
    grp_share = {g: float(per_sym[grp == g].sum() / tot) for g in G}
    top_n = (Zw[rn] == 2) & np.isfinite(ex[rn])
    per_sym_n = top_n.sum(axis=0)
    grp_share_n = {g: float(per_sym_n[grp == g].sum() / per_sym_n.sum()) for g in G}
    loo = []
    for j in range(len(syms)):
        if per_sym[j] == 0 and per_sym_n[j] == 0:
            continue
        Zx = Zw.copy()
        Zx[:, j] = -9
        d = t3_daily(Zx, ex)[0]
        loo.append((nm(d, rw) - nm(d, rn), syms[j]))
    loo.sort()
    lgo = {}
    for g in G:
        Zx = np.where((grp == g)[None, :], -9, Zw).astype(np.int8)
        d = t3_daily(Zx, ex)[0]
        dd = nm(d, rw) - nm(d, rn)
        # 격자 중앙도
        diffs = []
        for Z in Zs:
            Zy = np.where((grp == g)[None, :], -9, Z).astype(np.int8)
            dy = t3_daily(Zy, ex)[0]
            diffs.append(nm(dy, rw) - nm(dy, rn))
        lgo[g] = (dd, float(np.nanmedian(diffs)), float(np.nanmean(np.array(diffs) > 0)))
    lines += ["", f"== 5. 1위 극상방 몰림(묶음 4구간, 경고 주). 기준 차 {pct(base)}",
              f"  경고 주 극상방 종목-일 {tot}개, 서로 다른 종목 {nd_sym}개, 상위 8: "
              + ", ".join(f"{a}({b},{c})" for a, b, c in top5)
              + f" · 상위 5 비중 {sum(b for _, b, _ in top5[:5]) / tot * 100:.0f}%",
              "  섹터 비중 경고/평상: " + ", ".join(f"{g}({gcount[g]}) {grp_share[g] * 100:.0f}/{grp_share_n[g] * 100:.0f}%" for g in G),
              f"  한 종목 빼기 범위: {pct(loo[0][0])}({loo[0][1]}) ~ {pct(loo[-1][0])}({loo[-1][1]})",
              "  한 섹터 빼기 → 1위 차 / 격자 차중앙 / 격자 차양수%: "
              + ", ".join(f"{g} {pct(v[0])}/{pct(v[1])}/{v[2] * 100:.0f}%" for g, v in lgo.items())]
    J["몰림"] = {"기준차": base, "종목일": int(tot), "종목수": nd_sym, "상위": top5, "섹터비중경고": grp_share,
                 "섹터비중평상": grp_share_n, "한종목빼기": [loo[0], loo[-1]], "한섹터빼기": lgo}

    # ---- 6. 연속값 단조성 (묶음 4구간 + 2023하)
    rp_all = np.unique(np.concatenate([pools["묶음(4구간)"], pools["2023하"]]))
    dv = dist[rp_all]
    okr = rp_all[np.isfinite(dv)]
    qs = np.nanpercentile(dist[okr], [20, 40, 60, 80])
    bins = np.digitize(dist[okr], qs)
    lines += ["", "== 6. 결정 종가 SOXX/SMA20-1 다섯 구간별 T3(h10 초과, 2023하+4구간). 경고 = 음수 구간",
              "  구간(범위): 1위 T3 | 격자 중앙 T3 · 양수% | 날짜수 | 다음 10일 SOXX"]
    J["단조"] = []
    D10 = D[("초과", 10)]
    for b in range(5):
        rr = okr[bins == b]
        t = np.array([nm(D10[k], rr) for k in range(len(cfgs))])
        lo, hi = dist[rr].min(), dist[rr].max()
        sf = nm(soxx_f[10], rr)
        J["단조"].append({"범위": [float(lo), float(hi)], "1위": t[wi], "격자중앙": float(np.nanmedian(t)),
                         "양수": float(np.nanmean(t > 0)), "날짜": int(len(rr)), "SOXX10일": sf})
        lines.append(f"  Q{b + 1} ({lo * 100:+5.1f}%~{hi * 100:+5.1f}%): {pct(t[wi])} | {pct(np.nanmedian(t))} · "
                     f"{np.nanmean(t > 0) * 100:3.0f}% | {len(rr)} | {pct(sf)}")

    # ---- 7. 사후 분해: 다음 10일 SOXX 방향
    lines += ["", "== 7. (사후·설명용) 다음 10일 SOXX 상승/하락으로 나눈 T3 (h10 초과, 묶음 4구간+2023하)"]
    J["사후"] = {}
    for lab, wm in (("경고", warn), ("평상", ~warn)):
        for dirn, cond in (("SOXX↑", soxx_f[10] > 0), ("SOXX↓", soxx_f[10] <= 0)):
            rr = rp_all[wm[rp_all] & cond[rp_all]]
            t = np.array([nm(D10[k], rr) for k in range(len(cfgs))])
            J["사후"][f"{lab}·{dirn}"] = {"1위": t[wi], "격자중앙": float(np.nanmedian(t)), "양수": float(np.nanmean(t > 0)),
                                         "날짜": int(len(rr)), "SOXX": nm(soxx_f[10], rr)}
            lines.append(f"  [{lab}·{dirn}] {len(rr)}일, SOXX {pct(nm(soxx_f[10], rr))}: 1위 {pct(t[wi])} | 격자 중앙 "
                         f"{pct(np.nanmedian(t))} 양수 {np.nanmean(t > 0) * 100:3.0f}%")

    # ---- 8. 극상방 정의 단위
    keyof = lambda c: (c["x"], c["xv"]) if c["fam"] == "tx" else ("dev", c["L"], c["w"], c["b"])
    groups = {}
    for k, c in enumerate(cfgs):
        groups.setdefault(keyof(c), []).append(k)
    lines += ["", f"== 8. 서로 다른 극상방 정의 {len(groups)}개 단위로(정의 안 중앙값): 경고-평상 T3 차 양수 비율"]
    J["정의단위"] = {}
    for pn in ("2023하", "앞", "뒤", "2025", "2024"):
        rp = rows_of(dates, pn, 10)
        rw_, rn_ = rp[warn[rp]], rp[~warn[rp]]
        per = []
        for key, ks in groups.items():
            dd = [nm(D10[k], rw_) - nm(D10[k], rn_) for k in ks]
            ww = [nm(D10[k], rw_) for k in ks]
            per.append((np.nanmedian(dd), np.nanmedian(ww)))
        per = np.array(per)
        J["정의단위"][pn] = {"차양수": float(np.nanmean(per[:, 0] > 0)), "경고T3양수": float(np.nanmean(per[:, 1] > 0))}
        lines.append(f"  [{pn:>5}] 차 양수 {np.nanmean(per[:, 0] > 0) * 100:3.0f}% · 경고 T3 양수 {np.nanmean(per[:, 1] > 0) * 100:3.0f}% "
                     f"(정의 {len(per)}개, 정의별 설정 수 {min(map(len, groups.values()))}~{max(map(len, groups.values()))})")

    # ---- 9. e 재확인(독립 계산): 1위 극상방·상방 원수익/초과, 묶음 4구간 + 2023하
    lines += ["", "== 9. 1위 극상방·상방 10일 수익(둘 다 있는 날, 날짜 평균): 원 / 초과"]
    J["e"] = {}
    for pn, rp in (("묶음(4구간)", pools["묶음(4구간)"]), ("2023하", pools["2023하"]), ("뒤", pools["뒤"])):
        for lab, wm in (("평상", ~warn), ("경고", warn)):
            rr = rp[wm[rp]]
            _, mex = t3_daily(Zw, F[10].ex)
            _, mraw = t3_daily(Zw, F[10].raw)
            ok = rr[np.isfinite(mex[1][rr]) & np.isfinite(mex[2][rr])]
            uni = np.nanmean(F[10].raw, axis=1)
            v = {"극상방원": nm(mraw[2], ok), "극상방초과": nm(mex[2], ok), "상방원": nm(mraw[1], ok),
                 "상방초과": nm(mex[1], ok), "유니버스": nm(uni, ok), "날짜": int(len(ok)),
                 "극상방상승확률": None}
            J["e"][f"{pn}·{lab}"] = v
            lines.append(f"  [{pn}·{lab}] {v['날짜']}일: 극상방 {pct(v['극상방원'])}/{pct(v['극상방초과'])} | 상방 "
                         f"{pct(v['상방원'])}/{pct(v['상방초과'])} | 유니버스 원 {pct(v['유니버스'])}")

    # ---- 현재 상태
    last_d = sc.index[-1]
    sma_last = sc.rolling(20).mean().iloc[-1]
    cur = [syms[j] for j in range(len(syms)) if Zi[-1, j] == 2]
    lines += ["", f"== 현재: {last_d} SOXX {sc.iloc[-1]:.2f} vs SMA20 {sma_last:.2f} ({(sc.iloc[-1] / sma_last - 1) * 100:+.2f}%) → "
              f"다음 주(10/12~) 경고 {'예' if sc.iloc[-1] < sma_last else '아니오'}; 1위 극상방(독립 계산) {len(cur)}종목: {', '.join(cur)}",
              f"  섹터 묶음: " + ", ".join(f"{g} {gcount[g]}" for g in G)]
    J["현재"] = {"날짜": last_d, "SOXX": float(sc.iloc[-1]), "SMA20": float(sma_last), "극상방": cur}

    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(Zm.OUT, "verify_2_k3_extra.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    with open(os.path.join(Zm.OUT, "verify_2_k3_extra.json"), "w", encoding="utf-8") as f:
        json.dump(J, f, ensure_ascii=False, indent=1,
                  default=lambda o: o.item() if isinstance(o, np.generic) else (list(o) if isinstance(o, tuple) else str(o)))


if __name__ == "__main__":
    main()
