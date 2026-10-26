"""B4 반박 시험 (turn_v2, 연구용). turn_A/turn_compare.py의 함수를 import해 같은 단계값을 만들고
(1) 정답 기준 교체 (2) 선정 문턱 조정 (3) 2026H1 선정 → H2·2025 확인 (4) EWY·SOXX 제외 를 잰다.
추가: 종목 묶음 부트스트랩(B4−B0 점수 차, direct 배수).
실행: PYTHONIOENCODING=utf-8 py refute.py   결과: 같은 폴더 results.json, results.txt
"""
import sys
sys.dont_write_bytecode = True  # 저장소 code/ 에 .pyc를 만들지 않는다
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "turn_A"))
import turn_compare as TC  # noqa: E402

CANDS = TC.CANDS
PER = dict(TC.PER)
H1, H2 = PER["2026H1"], PER["2026H2"]
inper = TC.inper
OUT = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    OUT.append(s)


# ------------------------------------------------------------------ 자료·단계 (turn_compare.main과 같은 식)
def build():
    raw = pd.read_parquet(TC.PANEL)
    raw["ts"] = pd.to_datetime(raw["ts"])
    parts = []
    for sym, g in raw.groupby("sym", sort=True):
        o, _ = TC.per_symbol(g)
        parts.append(o)
    df = pd.concat(parts, ignore_index=True)
    R = df["rsi"].to_numpy()
    T = {c: df[c].to_numpy() for c in ("T3_10", "T3_15", "T3_20", "T0", "TE", "T9")}
    dd, ru, u = df["dd10"].to_numpy(), df["ru10"].to_numpy(), df["u10"].to_numpy()
    b0 = TC.stage5(R, T["T3_20"], 0.5)
    b1 = TC.stage5(R, T["T3_15"], 0.5)
    stg = {"B0": b0, "B1": b1, "B2": TC.stage5(R, T["T3_10"], 0.5),
           "B3": TC.overwrite(b0, R, dd, ru, u, 1.0), "B4": TC.overwrite(b0, R, dd, ru, u, 1.5),
           "B5": TC.overwrite(b1, R, dd, ru, u, 1.5), "B7": TC.stage5(R, T["T0"], 0.5),
           "B8": TC.stage5(R, T["TE"], 0.655), "B9": TC.stage5(R, T["T9"], 0.5)}
    valid = df["valid"].to_numpy()
    for k, v in stg.items():
        df["z_" + k] = np.where(valid, v, -9).astype(int)
    df["z_B6"] = np.where(valid, df["z_B6"], -9).astype(int)
    return df


def add_truths(df):
    """대체 정답: 지그재그 θ 3·7·15%, ZV 0.5·1.5·3.0, 그리고 사후 전방 방향 4종."""
    zz_specs = {"Z3": ("fix", 0.03), "Z7": ("fix", 0.07), "Z15": ("fix", 0.15),
                "ZV05": ("vol", 0.5), "ZV15": ("vol", 1.5), "ZV30": ("vol", 3.0)}
    cols = {k: np.zeros(len(df), np.int8) for k in list(zz_specs) + ["FA10", "TBV1", "TB5", "FR10"]}
    for sym, g in df.groupby("sym", sort=False):
        ix = g.index.to_numpy()
        C = g["C"].to_numpy()
        lc = np.log(C)
        sdv = g["sd20"].to_numpy()
        for name, (kind, v) in zz_specs.items():
            th = np.where(np.isfinite(sdv), v, np.nan) if kind == "fix" else v * sdv * np.sqrt(10)
            lab, _ = TC.zigzag(lc, th)
            cols[name][ix] = lab
        n = len(C)
        H = 10
        fa = np.zeros(n, np.int8)
        tbv = np.zeros(n, np.int8)
        tb5 = np.zeros(n, np.int8)
        fr = np.zeros(n, np.int8)
        for t in range(n - H):
            w = lc[t + 1:t + H + 1] - lc[t]
            imax, imin = int(np.argmax(w)), int(np.argmin(w))
            # 10일 최고·최저 중 먼저 온 쪽의 방향(먼저 온 극값이 지금보다 위면 +1, 아래면 −1)
            if imax < imin:
                fa[t] = 1 if w[imax] > 0 else -1
            elif imin < imax:
                fa[t] = -1 if w[imin] < 0 else 1
            # 삼중 장벽: ±1.0·sd20·√10 / ±5% 중 먼저 닿는 쪽(못 닿으면 라벨 없음)
            for arr, b in ((tbv, sdv[t] * np.sqrt(10) if np.isfinite(sdv[t]) else np.nan), (tb5, 0.05)):
                if not np.isfinite(b):
                    continue
                up = np.flatnonzero(w >= b)
                dn = np.flatnonzero(w <= -b)
                fu = up[0] if len(up) else 99
                fd = dn[0] if len(dn) else 99
                if fu < fd:
                    arr[t] = 1
                elif fd < fu:
                    arr[t] = -1
            fr[t] = 1 if w[-1] > 0 else (-1 if w[-1] < 0 else 0)
        cols["FA10"][ix], cols["TBV1"][ix], cols["TB5"][ix], cols["FR10"][ix] = fa, tbv, tb5, fr
    for k, v in cols.items():
        df["lab_" + k] = v
    return df


# ------------------------------------------------------------------ 지표
def cell(df, st, lab, mask):
    z = np.sign(df[st].to_numpy())
    L = df["lab_" + lab].to_numpy()
    M = mask & (L != 0)
    n = int(M.sum())
    if not n:
        return dict(n=0, agree=np.nan, opp=np.nan, score=np.nan)
    a = float(np.mean(z[M] == L[M]))
    o = float(np.mean(z[M] == -L[M]))
    return dict(n=n, agree=a, opp=o, score=a - 2 * o)


def score(df, st, labs, periods, base):
    return float(np.mean([cell(df, st, lab, base & inper(df["date"], PER[p]).to_numpy())["score"]
                          for lab in labs for p in periods]))


def eligible(m, b0, fm=1.3, dm=1.5, st=0.05):
    return (m["flips"] <= fm * b0["flips"]) and (m["direct"] <= dm * b0["direct"]) and (m["steady_up"] >= b0["steady_up"] - st)


def pick(scores, metrics, fm=1.3, dm=1.5, st=0.05):
    el = [k for k in CANDS if eligible(metrics[k], metrics["B0"], fm, dm, st)]
    return max(el, key=lambda k: scores[k]), el


def side_metrics(df, st, mask_fl, mask_st, stU):
    fd = TC.flips_direct(df, st, mask_fl)
    zz = df[st]
    return {"flips": fd["flips_py"], "direct": fd["direct_py"],
            "steady_up": float(np.mean(np.sign(zz[mask_st & stU]) > 0))}


# ------------------------------------------------------------------ 부트스트랩(종목 묶음)
def per_sym_counts(df, st, labs, periods, base):
    """종목별 (lab, period) 칸의 n, agree수, opp수 — 배열 [sym, cell, 3]."""
    syms = sorted(df.loc[base, "sym"].unique())
    sidx = {s: i for i, s in enumerate(syms)}
    cells = [(lab, p) for lab in labs for p in periods]
    A = np.zeros((len(syms), len(cells), 3))
    z = np.sign(df[st].to_numpy())
    sy = df["sym"].map(sidx).to_numpy()
    for j, (lab, p) in enumerate(cells):
        L = df["lab_" + lab].to_numpy()
        M = (base & inper(df["date"], PER[p]).to_numpy()) & (L != 0)
        s = sy[M].astype(int)
        A[:, j, 0] = np.bincount(s, minlength=len(syms))
        A[:, j, 1] = np.bincount(s, weights=(z[M] == L[M]).astype(float), minlength=len(syms))
        A[:, j, 2] = np.bincount(s, weights=(z[M] == -L[M]).astype(float), minlength=len(syms))
    return A, syms


def sc_from(A):
    tot = A.sum(0)
    return float(np.mean(tot[:, 1] / tot[:, 0] - 2 * tot[:, 2] / tot[:, 0]))


def per_sym_direct(df, st, mask):
    out = {}
    for sym, g in df[mask].groupby("sym", sort=True):
        z = g[st].to_numpy()
        idx = g["idx"].to_numpy()
        sg = np.sign(z)
        ls, li, whip = 0, -99, 0
        for t in range(len(sg)):
            if sg[t] != 0:
                if ls != 0 and sg[t] != ls and idx[t] - li <= 3:
                    whip += 1
                ls, li = sg[t], idx[t]
        ch = int(((np.diff(idx) == 1) & (z[1:] != z[:-1])).sum())
        out[sym] = (whip, ch, len(z))
    return out


def main():
    df = build()
    df = add_truths(df)
    stock = (~df["is_etf"]).to_numpy()
    V = df["valid"].to_numpy()
    S_all = stock & V
    dates = df["date"]
    S_ly = S_all & inper(dates, TC.P_LASTY).to_numpy()
    S_inall = S_all & inper(dates, PER["ALL"]).to_numpy()
    stU = ((df["slope60"] * 60 >= np.log(1.15)) & (df["r2_60"] >= 0.8) & (df["lab_Z10"] == 1)).to_numpy()
    res = {}

    # ---- 기준 재현
    base_labs = ("Z5", "Z10", "ZV1", "ZV2")
    sc0 = {k: score(df, "z_" + k, base_labs, ("2026H1", "2026H2"), S_all) for k in CANDS}
    met = {k: side_metrics(df, "z_" + k, S_ly, S_inall, stU) for k in CANDS}
    w0, el0 = pick(sc0, met)
    say("재현 score:", {k: round(v, 4) for k, v in sc0.items()})
    say("재현 eligible:", el0, "1위:", w0)
    say("B4가 B0와 다른 행 비율(주식 전체):", round(float(np.mean(df.loc[S_inall, "z_B4"] != df.loc[S_inall, "z_B0"])), 5))
    res["repro"] = {"score": sc0, "metrics": met, "winner": w0, "eligible": el0}

    # ---- (1) 정답 기준 교체
    truth_sets = {"base4": base_labs, "Z3": ("Z3",), "Z5": ("Z5",), "Z7": ("Z7",), "Z10": ("Z10",), "Z15": ("Z15",),
                  "ZV05": ("ZV05",), "ZV1": ("ZV1",), "ZV15": ("ZV15",), "ZV2": ("ZV2",), "ZV30": ("ZV30",),
                  "FA10": ("FA10",), "TBV1": ("TBV1",), "TB5": ("TB5",), "FR10": ("FR10",),
                  "alt6_zz": ("Z3", "Z7", "Z15", "ZV05", "ZV15", "ZV30"), "fwd4": ("FA10", "TBV1", "TB5", "FR10")}
    t1 = {}
    say("\n(1) 정답 기준별 2026 score(H1·H2 평균), 1위(규칙), 무제약 1위, B4−B0")
    for name, labs in truth_sets.items():
        sc = {k: score(df, "z_" + k, labs, ("2026H1", "2026H2"), S_all) for k in CANDS}
        sc25 = {k: score(df, "z_" + k, labs, ("2025",), S_all) for k in CANDS}
        w, _ = pick(sc, met)
        free = max(CANDS, key=lambda k: sc[k])
        rank = sorted(CANDS, key=lambda k: -sc[k])
        t1[name] = {"score": sc, "score_2025": sc25, "winner": w, "free_winner": free, "rank": rank,
                    "B4_minus_B0": sc["B4"] - sc["B0"], "B4_minus_B0_2025": sc25["B4"] - sc25["B0"]}
        say(f"  {name:8s} 1위 {w}  무제약 {free}  B4−B0 {sc['B4'] - sc['B0']:+.4f} (2025 {sc25['B4'] - sc25['B0']:+.4f})  "
            f"B0 {sc['B0']:.4f} B4 {sc['B4']:.4f} B7 {sc['B7']:.4f} | 순위 {' '.join(rank)}")
    res["truths"] = t1
    n_b4 = sum(1 for v in t1.values() if v["winner"] == "B4")
    say(f"  → 정답 {len(t1)}개 중 규칙 1위 B4 {n_b4}회, B0 {sum(1 for v in t1.values() if v['winner'] == 'B0')}회")

    # ---- (2) 선정 문턱 조정
    say("\n(2) 선정 문턱 격자 (flips 배수 × direct 배수 × steady 허용)")
    FM = [1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0]
    DM = [1.0, 1.25, 1.5, 1.75, 2.0, 2.1, 2.5, 3.0, 3.5]
    STL = [0.02, 0.05, 0.07, 0.08, 0.10, 0.15]
    grid = {}
    for fm, dm, stl in itertools.product(FM, DM, STL):
        w, el = pick(sc0, met, fm, dm, stl)
        grid[f"{fm}|{dm}|{stl}"] = w
    cnt = pd.Series(list(grid.values())).value_counts().to_dict()
    say("  격자", len(grid), "칸 1위 분포:", cnt)
    ratios = {k: {"flips_x": met[k]["flips"] / met["B0"]["flips"], "direct_x": met[k]["direct"] / met["B0"]["direct"],
                  "steady_gap": met["B0"]["steady_up"] - met[k]["steady_up"]} for k in CANDS}
    for k in CANDS:
        say(f"  {k}: flips×{ratios[k]['flips_x']:.3f} direct×{ratios[k]['direct_x']:.3f} steady 부족 {ratios[k]['steady_gap']:+.4f}  score {sc0[k]:.4f}")
    # 하나씩만 바꿨을 때
    one = {}
    for lab, vals, pos in (("flips", FM, 0), ("direct", DM, 1), ("steady", STL, 2)):
        for v in vals:
            p = [1.3, 1.5, 0.05]
            p[pos] = v
            one[f"{lab}={v}"] = pick(sc0, met, *p)[0]
    say("  한 문턱만 바꿈:", one)
    # 1위가 바뀌는 최소 조정
    res["thresholds"] = {"grid_counts": cnt, "ratios": ratios, "one_at_a_time": one, "grid": grid}

    # ---- (3) 2026H1로만 선정 → H2·2025 확인
    say("\n(3) 연도 내 전방 검증: 2026H1 자료만으로 선정")
    mH1 = S_all & inper(dates, H1).to_numpy()
    mPre = S_all & inper(dates, ("2024-01-02", H1[1])).to_numpy()
    fwd = {}
    for tag, fl_mask, st_mask in (("H1만", mH1, mH1), ("H1까지 전체", mH1, mPre)):
        scH1 = {k: score(df, "z_" + k, base_labs, ("2026H1",), S_all) for k in CANDS}
        mH = {k: side_metrics(df, "z_" + k, fl_mask, st_mask, stU) for k in CANDS}
        w, el = pick(scH1, mH)
        scH2 = {k: score(df, "z_" + k, base_labs, ("2026H2",), S_all) for k in CANDS}
        sc25 = {k: score(df, "z_" + k, base_labs, ("2025",), S_all) for k in CANDS}
        # H2에서 다시 잰 부가지표(문턱 유지 여부)
        mH2 = S_all & inper(dates, H2).to_numpy()
        mHh2 = {k: side_metrics(df, "z_" + k, mH2, mH2, stU) for k in CANDS}
        fwd[tag] = {"winner": w, "eligible": el, "scH1": scH1, "scH2": scH2, "sc25": sc25, "metrics_H1": mH,
                    "metrics_H2": mHh2, "winner_H2_minus_B0": scH2[w] - scH2["B0"], "winner_2025_minus_B0": sc25[w] - sc25["B0"],
                    "winner_still_eligible_H2": eligible(mHh2[w], mHh2["B0"])}
        say(f"  [{tag}] 적격 {el} → 1위 {w}; H2 {w}−B0 {scH2[w] - scH2['B0']:+.4f}, 2025 {sc25[w] - sc25['B0']:+.4f}, "
            f"H2에서도 적격 {fwd[tag]['winner_still_eligible_H2']}")
        for k in CANDS:
            say(f"     {k}: H1 {scH1[k]:.4f} H2 {scH2[k]:.4f} 2025 {sc25[k]:.4f} | H1 flips×{mH[k]['flips'] / mH['B0']['flips']:.2f} "
                f"direct×{mH[k]['direct'] / mH['B0']['direct']:.2f} steady {mH[k]['steady_up']:.3f}")
    res["forward"] = fwd

    # ---- (4) EWY·SOXX 제외
    say("\n(4) EWY·SOXX 제외")
    in_stock = df.loc[stock, "sym"].isin(["EWY", "SOXX"]).any()
    say("  주식 150종목 안에 EWY/SOXX 있음?", bool(in_stock), "(둘 다 is_etf=True → 주 점수·문턱 지표에 원래 안 들어감)")
    E = df["is_etf"].to_numpy() & V
    E_x = E & ~df["sym"].isin(["EWY", "SOXX"]).to_numpy()
    e_sc = {k: score(df, "z_" + k, base_labs, ("2026H1", "2026H2"), E) for k in CANDS}
    e_scx = {k: score(df, "z_" + k, base_labs, ("2026H1", "2026H2"), E_x) for k in CANDS}
    say("  ETF 12개 score:", {k: round(v, 4) for k, v in e_sc.items()})
    say("  ETF 10개(EWY·SOXX 제외):", {k: round(v, 4) for k, v in e_scx.items()})
    # 불만 구간(2026-09-01~10-12) 제외한 H2
    mH2x = S_all & inper(dates, ("2026-06-01", "2026-08-31")).to_numpy()
    PER["2026H2x"] = ("2026-06-01", "2026-08-31")
    scx = {k: score(df, "z_" + k, base_labs, ("2026H1", "2026H2x"), S_all) for k in CANDS}
    wx, _ = pick(scx, met)
    say("  불만 구간(9/1~10/12) 뺀 2026 score:", {k: round(v, 4) for k, v in scx.items()}, "→ 1위", wx)
    # 반도체·한국 관련 대형주 제외 (SOXX 상위 구성과 겹치는 종목)
    semi = ["NVDA", "AVGO", "AMD", "MU", "TSM", "QCOM", "INTC", "TXN", "AMAT", "LRCX", "KLAC", "ADI", "MRVL", "ASML",
            "NXPI", "MCHP", "ON", "MPWR", "ARM", "SMCI"]
    present = sorted(set(semi) & set(df.loc[stock, "sym"]))
    S_ns = S_all & ~df["sym"].isin(present).to_numpy()
    sc_ns = {k: score(df, "z_" + k, base_labs, ("2026H1", "2026H2"), S_ns) for k in CANDS}
    met_ns = {k: side_metrics(df, "z_" + k, S_ly & ~df["sym"].isin(present).to_numpy(),
                              S_inall & ~df["sym"].isin(present).to_numpy(), stU) for k in CANDS}
    wns, elns = pick(sc_ns, met_ns)
    say(f"  반도체 {len(present)}종목 제외 1위 {wns} (적격 {elns}); B4−B0 {sc_ns['B4'] - sc_ns['B0']:+.4f}")
    res["exclude"] = {"in_stock": bool(in_stock), "etf12": e_sc, "etf10": e_scx, "no_complaint_window": scx,
                      "no_complaint_window_winner": wx, "no_semi_present": present, "no_semi_score": sc_ns,
                      "no_semi_winner": wns, "no_semi_eligible": elns}

    # ---- 부트스트랩: B4−B0, B1/B3 문턱 배수
    say("\n(5) 종목 묶음 부트스트랩 2000회")
    rng = np.random.default_rng(20261013)
    cnts = {k: per_sym_counts(df, "z_" + k, base_labs, ("2026H1", "2026H2"), S_all)[0] for k in CANDS}
    nS = cnts["B0"].shape[0]
    draws = rng.integers(0, nS, size=(2000, nS))
    boot = {}
    for k in CANDS:
        if k == "B0":
            continue
        d = np.array([sc_from(cnts[k][ix]) - sc_from(cnts["B0"][ix]) for ix in draws])
        boot[k] = {"diff": sc0[k] - sc0["B0"], "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)),
                   "p_gt0": float(np.mean(d > 0))}
        say(f"  {k}−B0 {boot[k]['diff']:+.4f} [95% {boot[k]['lo']:+.4f}, {boot[k]['hi']:+.4f}] P(>0) {boot[k]['p_gt0']:.3f}")
    # 점수 1위가 B4로 남는 비율(문턱 지표 고정)
    wins = []
    for ix in draws[:1000]:
        s_b = {k: sc_from(cnts[k][ix]) for k in CANDS}
        wins.append(pick(s_b, met)[0])
    wc = pd.Series(wins).value_counts().to_dict()
    say("  부트스트랩 1000회 규칙 1위 분포(문턱 지표 고정):", wc)
    # direct 배수 부트스트랩
    dsy = {k: per_sym_direct(df, "z_" + k, S_ly) for k in CANDS}
    syms = sorted(dsy["B0"])
    dr = {}
    for k in ("B1", "B3", "B4", "B9"):
        w0a = np.array([dsy["B0"][s][0] for s in syms])
        wka = np.array([dsy[k][s][0] for s in syms])
        f0a = np.array([dsy["B0"][s][1] for s in syms])
        fka = np.array([dsy[k][s][1] for s in syms])
        dd_ = rng.integers(0, len(syms), size=(2000, len(syms)))
        rr = wka[dd_].sum(1) / w0a[dd_].sum(1)
        ff = fka[dd_].sum(1) / f0a[dd_].sum(1)
        dr[k] = {"direct_x": float(wka.sum() / w0a.sum()), "lo": float(np.percentile(rr, 2.5)), "hi": float(np.percentile(rr, 97.5)),
                 "flips_x": float(fka.sum() / f0a.sum()), "flo": float(np.percentile(ff, 2.5)), "fhi": float(np.percentile(ff, 97.5))}
        say(f"  {k} direct×{dr[k]['direct_x']:.2f} [95% {dr[k]['lo']:.2f}, {dr[k]['hi']:.2f}]  flips×{dr[k]['flips_x']:.3f} "
            f"[{dr[k]['flo']:.3f}, {dr[k]['fhi']:.3f}]")
    res["bootstrap"] = {"score_diff_vs_B0": boot, "rule_winner_counts": wc, "ratio_ci": dr}

    # ---- B4가 B0와 달라지는 날의 내용
    diff = S_inall & (df["z_B4"] != df["z_B0"]).to_numpy()
    sub = df.loc[diff, ["sym", "date", "z_B0", "z_B4", "lab_Z10", "lab_Z5", "lab_ZV1"]]
    say("\nB4≠B0 행:", int(diff.sum()), "/", int(S_inall.sum()), "; 2026:", int((diff & inper(dates, TC.P_2026).to_numpy()).sum()))
    say("  B0→B4 전환 표:", sub.groupby(["z_B0", "z_B4"]).size().to_dict())
    for lab in ("lab_Z10", "lab_Z5", "lab_ZV1"):
        m = sub[lab] != 0
        say(f"  {lab}: B4 일치 {np.mean(np.sign(sub.loc[m, 'z_B4']) == sub.loc[m, lab]):.3f}, "
            f"B0 일치 {np.mean(np.sign(sub.loc[m, 'z_B0']) == sub.loc[m, lab]):.3f} (n={int(m.sum())})")
    res["b4_diff_rows"] = {"n": int(diff.sum()), "n_total": int(S_inall.sum()),
                           "transitions": {f"{a}->{b}": int(v) for (a, b), v in sub.groupby(["z_B0", "z_B4"]).size().items()}}
    # 사례(10/12) 확인
    for sym in ("SOXX", "EWY"):
        r = df[(df.sym == sym) & (df.date == "2026-10-12")].iloc[0]
        say(f"  {sym} 10/12: B0 {r['z_B0']} B4 {r['z_B4']} B3 {r['z_B3']} dd10/u {r['dd10'] / r['u10']:.3f} (B4 발동 −1.5, B3 −1.0)")

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return o
    with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as f:
        json.dump(clean(res), f, ensure_ascii=False, indent=1)
    with open(os.path.join(HERE, "results.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(OUT))


if __name__ == "__main__":
    main()
