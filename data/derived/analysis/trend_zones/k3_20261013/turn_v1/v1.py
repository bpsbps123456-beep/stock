"""B4 반박 검증 (turn_v1, 연구용). 저장소 파일은 읽기·import만.

독립 구현: RSI(파인 RMA)·T3(20)·10일 다리 덮어쓰기·로그 지그재그를 turn_A 코드를 보지 않고 다시 짠다(정의는 과제 문안).
실행: PYTHONIOENCODING=utf-8 py v1.py  → results.json, results.txt
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PANEL = os.path.join(HERE, "..", "panel_1d_20261012.parquet")
FIRST = 129
PER = {"2024": ("2024-01-02", "2024-12-31"), "2025": ("2025-01-02", "2025-12-31"),
       "2026H1": ("2026-01-02", "2026-05-29"), "2026H2": ("2026-06-01", "2026-10-12")}
ALL = ("2024-01-02", "2026-10-12")
LASTY = ("2025-10-13", "2026-10-12")
THS = ("Z5", "Z10", "ZV1", "ZV2")
K20 = np.sqrt(19 * 39 / 120)
LOWVOL = ("SPY", "QQQ", "EWJ", "EEM")


def rsi14(c):
    n = 14
    r = np.full(len(c), np.nan)
    if len(c) <= n:
        return r
    ch = np.r_[np.nan, np.diff(c)]
    g = np.clip(ch, 0, None)
    l = np.clip(-ch, 0, None)
    ag, al = np.mean(g[1:n + 1]), np.mean(l[1:n + 1])
    for i in range(n, len(c)):
        if i > n:
            ag = ag + (g[i] - ag) / n
            al = al + (l[i] - al) / n
        r[i] = 100.0 if al == 0 and ag > 0 else (50.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al))
    return r


def zz(lc, th):
    """지그재그: 진행 중 극값 e에서 반대로 th[e] 이상 → e 확정. 확정 꺾임점 사이만 라벨."""
    n = len(lc)
    lab = np.zeros(n, np.int8)
    s = next((i for i in range(n) if np.isfinite(th[i])), n)
    if s >= n - 1:
        return lab, []
    piv = []
    hi = lo = s
    d = 0
    e = s
    for k in range(s + 1, n):
        if d == 0:
            if lc[k] >= lc[hi]:
                hi = k
            if lc[k] <= lc[lo]:
                lo = k
            if lc[k] - lc[lo] >= th[lo]:
                piv.append(lo); d = 1; e = k
            elif lc[hi] - lc[k] >= th[hi]:
                piv.append(hi); d = -1; e = k
            continue
        if d == 1:
            if lc[k] > lc[e]:
                e = k
            elif lc[e] - lc[k] >= th[e]:
                piv.append(e); d = -1; e = k
        else:
            if lc[k] < lc[e]:
                e = k
            elif lc[k] - lc[e] >= th[e]:
                piv.append(e); d = 1; e = k
    for a, b in zip(piv[:-1], piv[1:]):
        lab[a + 1:b + 1] = np.sign(lc[b] - lc[a])
    return lab, piv


def build():
    raw = pd.read_parquet(PANEL)
    raw["ts"] = pd.to_datetime(raw["ts"])
    out, pivs = [], {}
    base = 0
    for sym, g in raw.groupby("sym", sort=True):
        g = g.sort_values("ts")
        C = g["Close"].to_numpy(float)
        n = len(C)
        lc = np.log(C)
        lr = np.r_[np.nan, np.diff(lc)]
        sd20 = pd.Series(lr).rolling(20).std().to_numpy()
        m20 = pd.Series(lc).rolling(20).mean().to_numpy()
        T = (lc - m20) / (sd20 * K20)
        R = rsi14(C)
        hi10 = pd.Series(C).rolling(10).max().to_numpy()
        lo10 = pd.Series(C).rolling(10).min().to_numpy()
        hi20 = pd.Series(C).rolling(20).max().to_numpy()
        u = sd20 * np.sqrt(10)
        dd = np.log(C / hi10)
        ru = np.log(C / lo10)
        # 60일 로그 회귀
        sl = np.full(n, np.nan); r2 = np.full(n, np.nan)
        x = np.arange(60) - 29.5
        for i in range(59, n):
            y = lc[i - 59:i + 1]
            yc = y - y.mean()
            b = (x @ yc) / (x @ x)
            ss = yc @ yc
            sl[i] = b
            r2[i] = 1 - ((yc - b * x) @ (yc - b * x)) / ss if ss > 0 else np.nan
        # 앞으로 20일 최고 종가(얕은 눌림 회복 확인, 사후 정답용)
        fmax20 = pd.Series(C[::-1]).rolling(20, min_periods=20).max().to_numpy()[::-1]
        fmax20 = np.r_[fmax20[1:], np.nan]          # t+1..t+20
        f10 = np.r_[lc[10:] - lc[:-10], [np.nan] * 10] if n > 10 else np.full(n, np.nan)
        valid = (np.arange(n) >= FIRST) & np.isfinite(R) & np.isfinite(T) & np.isfinite(dd)
        b0 = np.select([R >= 70, R <= 30, T >= 0.5, T <= -0.5], [2, -2, 1, -1], 0)
        ext = (R >= 70) | (R <= 30)
        b4 = b0.copy()
        for m, name in ((1.5, "b4"), (1.0, "b3")):
            cd = dd <= -m * u
            cu = ru >= m * u
            ov = b0.copy()
            ov = np.where(cu & ~cd, 1, ov)
            ov = np.where(cd & ~cu, -1, ov)
            both = cd & cu
            ov = np.where(both & (-dd > ru), -1, ov)
            ov = np.where(both & (ru > -dd), 1, ov)
            ov = np.where(ext, b0, ov)
            if name == "b4":
                b4 = ov
            else:
                b3 = ov
        th = {"Z5": np.where(np.isfinite(sd20), 0.05, np.nan), "Z10": np.where(np.isfinite(sd20), 0.10, np.nan),
              "ZV1": u * 1.0, "ZV2": u * 2.0}
        d = pd.DataFrame({"sym": sym, "date": g["ts"].dt.strftime("%Y-%m-%d").to_numpy(), "etf": bool(g["is_etf"].iloc[0]),
                          "i": np.arange(n), "C": C, "lr": lr, "sd20": sd20, "T": T, "rsi": R, "dd": dd, "ru": ru, "u": u,
                          "hi20": hi20, "sl60": sl, "r2": r2, "fmax20": fmax20, "f10": f10, "valid": valid,
                          "b0": np.where(valid, b0, -9), "b4": np.where(valid, b4, -9), "b3": np.where(valid, b3, -9)})
        pivs[sym] = {}
        for k, t in th.items():
            lab, p = zz(lc, t)
            d["L" + k] = lab
            pivs[sym][k] = [base + q for q in p]
        out.append(d)
        base += n
    return pd.concat(out, ignore_index=True), pivs


def inp(df, p):
    return (df["date"] >= p[0]) & (df["date"] <= p[1])


def cell(df, col, mask, th):
    M = mask & (df["L" + th] != 0)
    z = np.sign(df.loc[M, col].to_numpy()); lab = df.loc[M, "L" + th].to_numpy()
    n = len(z)
    return n, int((z == lab).sum()), int((z == -lab).sum())


def score(df, col, mask, periods, ths=THS):
    v = []
    for th in ths:
        for p in periods:
            n, a, o = cell(df, col, mask & inp(df, PER[p]), th)
            v.append((a - 2 * o) / n)
    return float(np.mean(v))


def flips(df, col, mask):
    ch = wh = days = 0
    for _, g in df[mask].groupby("sym", sort=False):
        z = g[col].to_numpy(); ii = g["i"].to_numpy()
        days += len(z)
        ch += int(((np.diff(ii) == 1) & (np.diff(z) != 0)).sum())
        last, li = 0, -99
        for t in range(len(z)):
            s = np.sign(z[t])
            if s != 0:
                if last != 0 and s != last and ii[t] - li <= 3:
                    wh += 1
                last, li = s, ii[t]
    sy = days / 252
    return ch / sy, wh / sy, sy


def lags(df, col, pivs, syms, per):
    z = np.sign(df[col].to_numpy()); lc = np.log(df["C"].to_numpy()); v = df["valid"].to_numpy(); dt = df["date"].to_numpy()
    out = []
    for s in syms:
        p = pivs[s]["Z10"]
        for a, b in zip(p[:-1], p[1:]):
            if not (per[0] <= dt[a + 1] <= per[1]) or not v[a + 1:b + 1].all():
                continue
            dr = 1 if lc[b] > lc[a] else -1
            h = np.flatnonzero(z[a + 1:b + 1] == dr)
            out.append(int(h[0]) + 1 if len(h) else (b - a) + 1 + 1000)   # +1000 표식 = 미도달
    arr = np.array(out)
    nev = arr >= 1000
    arr2 = np.where(nev, arr - 1000, arr)
    return {"n": len(arr), "median": float(np.median(arr2)) if len(arr) else None,
            "never": float(nev.mean()) if len(arr) else None}


def main():
    df, pivs = build()
    V = df["valid"]
    S = V & ~df["etf"]
    E = V & df["etf"]
    res = {}
    # ---- 0. 재현: turn_A의 B0·B4 score/flips/direct
    rep = {}
    for c in ("b0", "b4", "b3"):
        f, w, _ = flips(df, c, S & inp(df, LASTY))
        rep[c] = {"score": score(df, c, S, ("2026H1", "2026H2")), "flips": f, "direct": w}
    res["repro"] = rep
    print("재현", json.dumps(rep))
    # ---- 1. 기간별 B0 vs B4
    stU = (df["sl60"] * 60 >= np.log(1.15)) & (df["r2"] >= 0.8) & (df["LZ10"] == 1)
    stD = (df["sl60"] * 60 <= -np.log(1.15)) & (df["r2"] >= 0.8) & (df["LZ10"] == -1)
    ssyms = sorted(df.loc[~df["etf"], "sym"].unique())
    per = {}
    for p, rng in PER.items():
        m = S & inp(df, rng)
        row = {}
        for c in ("b0", "b4", "b3"):
            f, w, sy = flips(df, c, m)
            lg = lags(df, c, pivs, ssyms, rng)
            row[c] = {"score": score(df, c, S, (p,)), "flips": f, "direct": w,
                      "steady_up": float((np.sign(df.loc[m & stU, c]) > 0).mean()),
                      "steady_dn": float((np.sign(df.loc[m & stD, c]) < 0).mean()),
                      "lag_med": lg["median"], "lag_never": lg["never"], "n_swings": lg["n"],
                      "gray": float((df.loc[m, c] == 0).mean())}
        row["changed_rows_b4"] = int((df.loc[m, "b4"] != df.loc[m, "b0"]).sum())
        row["rows"] = int(m.sum())
        per[p] = row
    res["by_period"] = per
    # ---- 차이의 잡음 크기: 종목 부트스트랩(2026 8칸 score B4-B0)
    rng_ = np.random.default_rng(7)
    cnt = {}
    for c in ("b0", "b4"):
        for th in THS:
            for p in ("2026H1", "2026H2"):
                M = S & inp(df, PER[p]) & (df["L" + th] != 0)
                g = df[M].assign(a=(np.sign(df.loc[M, c]) == df.loc[M, "L" + th]).astype(int),
                                 o=(np.sign(df.loc[M, c]) == -df.loc[M, "L" + th]).astype(int))
                cnt[(c, th, p)] = g.groupby("sym")[["a", "o"]].sum().assign(n=g.groupby("sym").size()).reindex(ssyms).fillna(0)
    diffs = []
    for _ in range(1000):
        pick = rng_.integers(0, len(ssyms), len(ssyms))
        sc = {}
        for c in ("b0", "b4"):
            v = []
            for th in THS:
                for p in ("2026H1", "2026H2"):
                    t = cnt[(c, th, p)].to_numpy()[pick].sum(0)
                    v.append((t[0] - 2 * t[1]) / t[2])
            sc[c] = np.mean(v)
        diffs.append(sc["b4"] - sc["b0"])
    diffs = np.array(diffs)
    res["boot_b4_minus_b0"] = {"mean": float(diffs.mean()), "p2.5": float(np.percentile(diffs, 2.5)),
                               "p97.5": float(np.percentile(diffs, 97.5)), "share_le0": float((diffs <= 0).mean())}
    # ---- 덮어쓰기가 어디서 켜지나(중복성)
    m = S & inp(df, ALL)
    ext = (df["rsi"] >= 70) | (df["rsi"] <= 30)
    for mm, nm in ((1.5, "m15"), (1.0, "m10")):
        cd = m & (df["dd"] <= -mm * df["u"]) & ~ext
        cu = m & (df["ru"] >= mm * df["u"]) & ~ext
        res[f"fire_{nm}"] = {"dn_fire_rows": int(cd.sum()), "dn_fire_share": float(cd.sum() / m.sum()),
                            "dn_already_b0_neg": float((df.loc[cd, "b0"] < 0).mean()),
                            "up_fire_rows": int(cu.sum()), "up_already_b0_pos": float((df.loc[cu, "b0"] > 0).mean())}
    # 덮어쓰기 켜질 때 T3(20) 분포 = 왜 거의 안 바뀌나
    cd = m & (df["dd"] <= -1.5 * df["u"]) & ~ext
    res["fire_m15"]["T_when_dn_fire_q"] = [float(q) for q in df.loc[cd, "T"].quantile([.5, .9, .99, 1.0])]
    # ---- 4. 순환 이득: 바뀐 날만 θ별 일치·반대 + 사후와 무관한 10일 앞 수익 방향
    circ = {}
    for c in ("b4", "b3"):
        for scope, mask in (("stock_all", S & inp(df, ALL)), ("stock_2026", S & inp(df, ("2026-01-02", "2026-10-12"))),
                            ("etf_all", E & inp(df, ALL))):
            ch = mask & (df[c] != df["b0"])
            r = {"rows": int(ch.sum())}
            for th in THS:
                lab = df.loc[ch, "L" + th]
                new = np.sign(df.loc[ch, c]); old = np.sign(df.loc[ch, "b0"])
                L = lab != 0
                r[th] = {"labeled": int(L.sum()), "new_agree": float((new[L] == lab[L]).mean()) if L.any() else None,
                         "new_opp": float((new[L] == -lab[L]).mean()) if L.any() else None,
                         "old_agree": float((old[L] == lab[L]).mean()) if L.any() else None,
                         "old_opp": float((old[L] == -lab[L]).mean()) if L.any() else None}
            f = df.loc[ch, "f10"]; new = np.sign(df.loc[ch, c])
            ok = f.notna() & (new != 0)
            r["fwd10_same_dir"] = float((np.sign(f[ok]) == new[ok]).mean()) if ok.any() else None
            r["fwd10_mean_signed"] = float((f[ok] * new[ok]).mean()) if ok.any() else None
            r["fwd10_n"] = int(ok.sum())
            # 같은 시기 모든 하방 계열 날의 10일 앞 방향 적중(비교 기준)
            base_m = mask & (np.sign(df["b0"]) != 0) & df["f10"].notna()
            r["fwd10_base_b0_same_dir"] = float((np.sign(df.loc[base_m, "f10"]) == np.sign(df.loc[base_m, "b0"])).mean())
            circ[f"{c}_{scope}"] = r
    # θ별 score 차이(Z만 vs ZV만)
    for c in ("b4", "b3"):
        circ[f"{c}_score_diff_fixedpct_only"] = score(df, c, S, ("2026H1", "2026H2"), ("Z5", "Z10")) - \
            score(df, "b0", S, ("2026H1", "2026H2"), ("Z5", "Z10"))
        circ[f"{c}_score_diff_vol_only"] = score(df, c, S, ("2026H1", "2026H2"), ("ZV1", "ZV2")) - \
            score(df, "b0", S, ("2026H1", "2026H2"), ("ZV1", "ZV2"))
        circ[f"{c}_score_diff_ZV1"] = score(df, c, S, ("2026H1", "2026H2"), ("ZV1",)) - \
            score(df, "b0", S, ("2026H1", "2026H2"), ("ZV1",))
    res["circularity"] = circ
    # ---- 2a. 저변동 지수 얕은 눌림(20일 최고 대비 −1.5~−5%, 다음 20일 안에 그 최고 종가 회복)
    lv = {}
    for s in LOWVOL:
        g = df[(df.sym == s) & df.valid & inp(df, ALL)]
        dd20 = np.log(g["C"] / g["hi20"])
        shallow = (dd20 <= np.log(0.985)) & (dd20 >= np.log(0.95)) & (g["fmax20"] > g["hi20"])
        r = {"shallow_days": int(shallow.sum())}
        for c in ("b0", "b4"):
            r[c + "_down_share"] = float((g.loc[shallow, c] < 0).mean()) if shallow.any() else None
            r[c + "_down_days"] = int((g.loc[shallow, c] < 0).sum())
            sh26 = shallow & (g["date"] >= "2026-01-02")
            r[c + "_down_days_2026"] = int((g.loc[sh26, c] < 0).sum())
        r["shallow_days_2026"] = int((shallow & (g["date"] >= "2026-01-02")).sum())
        r["b4_ne_b0_days"] = int((g["b4"] != g["b0"]).sum())
        r["b4_ne_b0_list"] = g.loc[g["b4"] != g["b0"], ["date", "C", "T", "rsi", "dd", "ru", "u", "b0", "b4", "LZ5", "LZ10", "LZV1", "LZV2"]] \
            .assign(dd_u=lambda x: x.dd / x.u, ru_u=lambda x: x.ru / x.u).round(4).to_dict("records")
        lv[s] = r
    res["lowvol_shallow"] = lv
    # ---- 2b. 강한 상승 추세 중 2~5% 눌림(10일 최고 대비) — 주식
    m = S & inp(df, ALL)
    strong = (df["sl60"] * 60 >= np.log(1.15)) & (df["r2"] >= 0.8)
    dip = (df["dd"] <= np.log(0.98)) & (df["dd"] >= np.log(0.95))
    rec = df["fmax20"] > df["C"] * np.exp(-df["dd"])          # 다음 20일 안에 10일 최고 회복
    sd = m & strong & dip & rec
    res["strong_dip"] = {"rows": int(sd.sum())}
    for c in ("b0", "b4", "b3"):
        res["strong_dip"][c + "_down_share"] = float((df.loc[sd, c] < 0).mean())
        for p in PER:
            res["strong_dip"].setdefault("by_period", {}).setdefault(p, {})[c] = float(
                (df.loc[sd & inp(df, PER[p]), c] < 0).mean())
    # ---- 2c. 횡보장 깜빡임: |60일 수익| < 5%, R² < 0.2
    side = S & inp(df, ALL) & ((df["sl60"] * 59).abs() < 0.05) & (df["r2"] < 0.2)
    res["sideways"] = {"rows": int(side.sum())}
    for c in ("b0", "b4", "b3"):
        f, w, sy = flips(df, c, side)
        res["sideways"][c] = {"flips_py": f, "direct_py": w}
    # ---- 3. 구체 사례
    cases = {}
    # (a) B4가 B0를 바꾼 날 중 Z10·ZV2 정답과 반대가 된 날(덮어쓰기 오판)
    ch = S & inp(df, ALL) & (df["b4"] != df["b0"])
    bad = df[ch & ((np.sign(df["b4"]) == -df["LZ10"]) | (np.sign(df["b4"]) == -df["LZV2"]) | (np.sign(df["b4"]) == -df["LZ5"]))]
    cases["b4_changed_wrong"] = bad[["sym", "date", "C", "T", "rsi", "dd", "ru", "u", "b0", "b4", "LZ5", "LZ10", "LZV1", "LZV2", "f10"]] \
        .assign(dd_u=lambda x: x.dd / x.u, ru_u=lambda x: x.ru / x.u).round(4).to_dict("records")
    cases["b4_changed_all_n"] = int(ch.sum())
    cases["b4_changed_right_n"] = int((ch & ((np.sign(df["b4"]) == df["LZ10"]))).sum())
    # (b) 2026, 10일 최고 대비 −7% 이하인데 B4 애매·상방이고 Z10 하락 스윙 안
    late = V & inp(df, ("2026-01-02", "2026-10-12")) & (df["dd"] <= np.log(0.93)) & (df["b4"] >= 0) & (df["rsi"] > 30)
    lt = df[late][["sym", "etf", "date", "C", "T", "rsi", "dd", "u", "b0", "b4", "LZ10", "LZV2", "f10"]].assign(dd_u=lambda x: x.dd / x.u)
    cases["late_2026_rows"] = int(late.sum())
    cases["late_2026_rows_LZ10_down"] = int((late & (df["LZ10"] == -1)).sum())
    cases["late_2026_share_of_dd7"] = float(late.sum() / (V & inp(df, ("2026-01-02", "2026-10-12")) & (df["dd"] <= np.log(0.93)) & (df["rsi"] > 30)).sum())
    cases["late_2026_sample"] = lt.sort_values("dd").groupby("sym").head(1).head(25).round(4).to_dict("records")
    # (c) 강한 상승 중 얕은 눌림을 하방으로(2026)
    sd26 = sd & inp(df, ("2026-01-02", "2026-10-12")) & (df["b4"] < 0)
    cases["strong_dip_down_2026_rows"] = int(sd26.sum())
    cases["strong_dip_down_2026_sample"] = df[sd26][["sym", "date", "C", "T", "rsi", "dd", "u", "b0", "b4", "LZ10", "LZ5", "f10"]] \
        .sort_values("f10", ascending=False).groupby("sym").head(1).head(15).round(4).to_dict("records")
    # 사례 표
    for s, p in (("SOXX", ("2026-09-28", "2026-10-12")), ("EWY", ("2026-09-22", "2026-10-12")),
                 ("SPY", ("2026-07-15", "2026-08-07")), ("MU", ("2026-06-22", "2026-08-05"))):
        cases[s] = df[(df.sym == s) & inp(df, p)][["date", "C", "T", "rsi", "dd", "ru", "u", "b0", "b4", "b3", "LZ5", "LZ10", "LZV1", "LZV2"]] \
            .assign(dd_u=lambda x: x.dd / x.u).round(4).to_dict("records")
    res["cases"] = cases

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else round(float(o), 6)
        if isinstance(o, np.bool_):
            return bool(o)
        return o
    with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as f:
        json.dump(clean(res), f, ensure_ascii=False, indent=1)
    df.to_parquet(os.path.join(HERE, "frame.parquet"))
    print("done")


if __name__ == "__main__":
    main()
