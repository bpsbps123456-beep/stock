"""K3n 반박 시도: K0 대비 K3n이 '눈으로 보는 주 추세'를 더 못 보여 주는 경우 사냥.

실행: PYTHONIOENCODING=utf-8 py fail_hunt.py
결과: 같은 폴더 fail_results.json, fail_results.txt, cases.csv
지표 계산은 impl_A/compare_mid.py의 per_symbol·stage5·rsi_pine을 그대로 쓴다(두 구현이 이미 같은 숫자로 확인).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "impl_A"))
import compare_mid as CM  # noqa: E402

PANEL = os.path.join(HERE, "..", "panel_1d_20261012.parquet")
PERIODS = {
    "2024": ("2024-01-02", "2024-12-31"),
    "2025": ("2025-01-02", "2025-12-31"),
    "26H1": ("2026-01-02", "2026-05-29"),
    "26H2": ("2026-06-01", "2026-10-12"),
    "ALL": ("2024-01-02", "2026-10-12"),
}
KS = ["K0", "K3n"]


def inper(d, p):
    return (d >= p[0]) & (d <= p[1])


# ---------------------------------------------------------------- 눈 기준: 지그재그 스윙(사후 라벨)
def zigzag_labels(C, th):
    """th: 되돌림 비율(스칼라). 반환: 각 날의 스윙 방향(+1 상승 스윙, -1 하락 스윙), 피벗 인덱스 목록.
    마지막 미확정 스윙은 현재 진행 방향으로 라벨한다."""
    n = len(C)
    lab = np.zeros(n, int)
    if n < 2:
        return lab, []
    # 시작 방향: 처음으로 th 이상 움직인 쪽
    hi_i = lo_i = 0
    mode = 0
    i = 1
    while i < n and mode == 0:
        if C[i] > C[hi_i]:
            hi_i = i
        if C[i] < C[lo_i]:
            lo_i = i
        if C[i] >= C[lo_i] * (1 + th):
            mode = 1
            piv = [lo_i]
            ext = i
        elif C[i] <= C[hi_i] * (1 - th):
            mode = -1
            piv = [hi_i]
            ext = i
        i += 1
    if mode == 0:
        return lab, []
    for j in range(i, n):
        if mode == 1:
            if C[j] > C[ext]:
                ext = j
            elif C[j] <= C[ext] * (1 - th):
                piv.append(ext)
                mode, ext = -1, j
        else:
            if C[j] < C[ext]:
                ext = j
            elif C[j] >= C[ext] * (1 + th):
                piv.append(ext)
                mode, ext = 1, j
    piv.append(ext)  # 미확정 끝 극값
    # 라벨: 피벗 a(저점)→b(고점) 사이 (a, b] 상승
    for a, b in zip(piv[:-1], piv[1:]):
        d = 1 if C[b] > C[a] else -1
        lab[a + 1:b + 1] = d
    # 마지막 극값 이후(진행 중): 현재 모드
    lab[piv[-1] + 1:] = mode
    lab[:piv[0] + 1] = 0
    return lab, piv


def add_features(df):
    out = []
    for sym, g in df.groupby("sym", sort=False):
        g = g.copy()
        C = g["C"].to_numpy()
        s = pd.Series(C)
        g["ret60"] = np.log(C / s.shift(60).to_numpy())
        g["hi60"] = s.rolling(60, min_periods=60).max().to_numpy()
        g["sma50"] = s.rolling(50, min_periods=50).mean().to_numpy()
        mx20 = s.rolling(20, min_periods=20).max()
        mn20 = s.rolling(20, min_periods=20).min()
        g["rng20"] = ((mx20 - mn20) / s.rolling(20, min_periods=20).mean()).to_numpy()
        # 지그재그: 고정 10%, 15%, 종목 변동성 비례(1×중앙 sd20·√20)
        sdmed = float(np.nanmedian(g["sd20"])) * np.sqrt(20)
        for name, th in (("zz10", 0.10), ("zz15", 0.15), ("zzv", max(sdmed, 0.04))):
            lab, piv = zigzag_labels(C, th)
            g[name] = lab
            # 피벗 이후 경과일·다음 피벗까지 남은 일
            since = np.full(len(C), 999)
            until = np.full(len(C), 999)
            swing = np.full(len(C), np.nan)   # 해당 스윙 전체 크기(로그)
            for a, b in zip(piv[:-1], piv[1:]):
                idx = np.arange(a + 1, b + 1)
                since[idx] = idx - a
                until[idx] = b - idx
                swing[idx] = abs(np.log(C[b] / C[a]))
            if piv:
                idx = np.arange(piv[-1] + 1, len(C))
                since[idx] = idx - piv[-1]
            g[name + "_since"] = since
            g[name + "_until"] = until
            g[name + "_swing"] = swing
        g["zzv_th"] = max(sdmed, 0.04)
        out.append(g)
    return pd.concat(out).sort_index()


# ---------------------------------------------------------------- 측정
def trans_stats(df, st, mask):
    prev = df.groupby("sym")[st].shift(1)
    pidx = df.groupby("sym")["idx"].shift(1)
    cur = df[st]
    ok = mask & prev.notna() & (prev != -9) & (pidx == df["idx"] - 1)
    ch = ok & (cur != prev)
    rev = ch & (np.sign(cur - prev) * np.sign(df["lr"]) < 0)
    sy = mask.sum() / 252.0
    return {"n_ch": int(ch.sum()), "rev_share": float(rev.sum() / ch.sum()) if ch.sum() else np.nan,
            "flips_py": float(ch.sum() / sy) if sy else np.nan, "rev_py": float(rev.sum() / sy) if sy else np.nan}


def flicker(df, st, mask):
    """깜빡임: 단계가 바뀌었다가 1~2일 안에 원래 단계로 돌아온 횟수/종목·연, 1일짜리 구간 비율."""
    nflick, days, runs1, nruns = 0, 0, 0, 0
    for sym, g in df[mask].groupby("sym", sort=False):
        z = g[st].to_numpy()
        idx = g["idx"].to_numpy()
        days += len(z)
        # 연속 구간
        brk = np.r_[True, (np.diff(idx) != 1) | (z[1:] != z[:-1])]
        starts = np.flatnonzero(brk)
        lens = np.diff(np.r_[starts, len(z)])
        nruns += len(lens)
        runs1 += int((lens == 1).sum())
        for t in range(1, len(z) - 1):
            if idx[t] - idx[t - 1] != 1 or z[t] == z[t - 1]:
                continue
            a = z[t - 1]
            for h in (1, 2):
                if t + h < len(z) and idx[t + h] - idx[t] == h and z[t + h] == a and np.all(z[t:t + h] != a):
                    nflick += 1
                    break
    sy = days / 252.0
    return {"flicker_py": nflick / sy if sy else np.nan, "run1_share": runs1 / nruns if nruns else np.nan}


def share(z):
    n = len(z)
    if n == 0:
        return {"n": 0}
    return {"n": int(n), "down": float(np.mean(z < 0)), "gray": float(np.mean(z == 0)), "up": float(np.mean(z > 0)),
            "ext_dn": float(np.mean(z == -2)), "ext_up": float(np.mean(z == 2))}


def lag_stats(df, st, evs, up):
    z = df[st].to_numpy()
    lags = []
    for sym, p, q, qp in evs:
        seg = z[p:q + 21]
        hit = np.flatnonzero(seg > 0) if up else np.flatnonzero(seg < 0)
        lags.append(int(hit[0]) if len(hit) else 36)
    return {"n": len(lags), "median": float(np.median(lags)) if lags else np.nan,
            "mean": float(np.mean(lags)) if lags else np.nan}


def zz_agree(df, st, mask, col, min_since=1, min_until=0):
    lab = df[col].to_numpy()
    z = np.sign(df[st].to_numpy())
    M = mask.to_numpy() & (lab != 0) & (df[col + "_since"].to_numpy() >= min_since) & (df[col + "_until"].to_numpy() >= min_until)
    L, Z = lab[M], z[M]
    return {"n": int(M.sum()), "agree": float(np.mean(Z == L)), "opp": float(np.mean(Z == -L)), "gray": float(np.mean(Z == 0)),
            "opp_in_up": float(np.mean(Z[L > 0] < 0)), "opp_in_dn": float(np.mean(Z[L < 0] > 0))}


def disagree(df, mask, col):
    """K0·K3n 부호가 다른 날: 어느 쪽이 지그재그와 맞는가."""
    lab = df[col].to_numpy()
    a = np.sign(df["z_K0"].to_numpy())
    b = np.sign(df["z_K3n"].to_numpy())
    M = mask.to_numpy() & (lab != 0) & (a != b)
    L, A, B = lab[M], a[M], b[M]
    return {"n_disagree": int(M.sum()), "share_of_days": float(M.sum() / max(1, (mask.to_numpy() & (lab != 0)).sum())),
            "K0_right": int(np.sum(A == L)), "K3n_right": int(np.sum(B == L)),
            "K0_opp": int(np.sum(A == -L)), "K3n_opp": int(np.sum(B == -L))}


def main():
    raw = pd.read_parquet(PANEL)
    raw["ts"] = pd.to_datetime(raw["ts"])
    df = pd.concat([CM.per_symbol(g) for _, g in raw.groupby("sym", sort=True)], ignore_index=True)
    R = df["rsi"].to_numpy()
    valid = (df["idx"] >= CM.MIN_IDX).to_numpy() & np.isfinite(R) & np.isfinite(df["T0"].to_numpy()) & np.isfinite(df["T3"].to_numpy())
    df["z_K0"] = np.where(valid, CM.stage5(R, df["T0"].to_numpy(), 0.5), -9)
    df["z_K3n"] = np.where(valid, CM.stage5(R, df["T3"].to_numpy(), 0.5), -9)
    df = add_features(df)
    valid = pd.Series(valid, index=df.index) & np.isfinite(df["hi20"])
    stock = ~df["is_etf"]

    # 확인
    ew = df[(df.sym == "EWY") & df.date.isin(["2026-10-08", "2026-10-09", "2026-10-12"])][["date", "C", "T0", "T3", "z_K0", "z_K3n"]]
    print(ew.to_string())

    # 단면 순위(종목군 안에서 날짜별)
    for col in ("ret60", "sd20"):
        df[col + "_rk"] = np.nan
        m = stock & valid & np.isfinite(df[col])
        df.loc[m, col + "_rk"] = df[m].groupby("date")[col].rank(pct=True)
    C = df["C"]
    ev = {
        "dn10": (C / df["hi10"] - 1 <= -0.10),
        "up10": (C / df["lo10"] - 1 >= 0.10),
    }
    # 천천히 오르는 강한 추세(60일 수익 상위 20% & 20일 변동성 하위 50%), 거울상 하락
    slow_up = (df["ret60_rk"] >= 0.8) & (df["sd20_rk"] <= 0.5)
    slow_dn = (df["ret60_rk"] <= 0.2) & (df["sd20_rk"] <= 0.5)
    # 상승 추세 속 얕은 눌림: 60일 상위 20%, 10일 고점 -3~-8%, 50일선 위
    dip_up = (df["ret60_rk"] >= 0.8) & (C / df["hi10"] - 1 <= -0.03) & (C / df["hi10"] - 1 > -0.08) & (C > df["sma50"])
    pop_dn = (df["ret60_rk"] <= 0.2) & (C / df["lo10"] - 1 >= 0.03) & (C / df["lo10"] - 1 < 0.08) & (C < df["sma50"])

    dn_ev_all = CM.m4_events(df, valid, stock, up=False)
    up_ev_all = CM.m4_events(df, valid, stock, up=True)
    dn_ev_all = [e for e in dn_ev_all if not df.at[e[2], "is_etf"]]
    up_ev_all = [e for e in up_ev_all if not df.at[e[2], "is_etf"]]

    res = {"by_period": {}, "etf_by_period": {}}
    for pname, p in PERIODS.items():
        inP = inper(df["date"], p)
        for grp, gmask, key in (("stock", stock, "by_period"), ("etf", df["is_etf"], "etf_by_period")):
            M = gmask & valid & inP
            # 횡보: 그 기간·그 종목군 rng20 하위 30%
            thr = np.nanquantile(df.loc[M, "rng20"], 0.30)
            side = M & (df["rng20"] <= thr)
            r = {"rows": int(M.sum()), "side_thr": float(thr)}
            for k in KS:
                st = "z_" + k
                z = df[st]
                d = {}
                d.update(trans_stats(df, st, M))
                d.update(flicker(df, st, M))
                d["dn10"] = share(z[M & ev["dn10"]].to_numpy())
                d["up10"] = share(z[M & ev["up10"]].to_numpy())
                ts = trans_stats(df, st, side)
                d["side_flips_py"] = ts["flips_py"]
                d["side_rev_share"] = ts["rev_share"]
                d["side_flicker"] = flicker(df, st, side)
                d["side_share"] = share(z[side].to_numpy())
                if grp == "stock":
                    d["slow_up"] = share(z[M & slow_up].to_numpy())
                    d["slow_dn"] = share(z[M & slow_dn].to_numpy())
                    d["dip_up"] = share(z[M & dip_up].to_numpy())
                    d["pop_dn"] = share(z[M & pop_dn].to_numpy())
                    dn_ev = [e for e in dn_ev_all if inper(pd.Series([df.at[e[2], "date"]]), p).iloc[0]]
                    up_ev = [e for e in up_ev_all if inper(pd.Series([df.at[e[2], "date"]]), p).iloc[0]]
                    d["crash_lag"] = lag_stats(df, st, dn_ev, up=False)
                    d["rally_lag"] = lag_stats(df, st, up_ev, up=True)
                for zc in ("zz10", "zz15", "zzv"):
                    d[zc] = zz_agree(df, st, M, zc)
                    d[zc + "_mid"] = zz_agree(df, st, M, zc, min_since=5, min_until=3)
                r[k] = d
            r["disagree"] = {zc: disagree(df, M, zc) for zc in ("zz10", "zz15", "zzv")}
            res[key][pname] = r
            print(grp, pname, "done")

    # ---------------------------------------------------------------- 사례 사냥
    # K3n 부호가 지그재그(10%)와 반대이고 K0는 맞는 날, 스윙 20% 이상·피벗에서 5일 이상 떨어진 한가운데.
    M = valid & inper(df["date"], PERIODS["ALL"])
    lab = df["zz10"]
    a = np.sign(df["z_K0"])
    b = np.sign(df["z_K3n"])
    bad = M & (lab != 0) & (b == -lab) & (a == lab) & (df["zz10_since"] >= 5) & (df["zz10_until"] >= 3) & (df["zz10_swing"] >= np.log(1.2))
    # 연속 구간으로 묶는다
    cases = []
    sub = df[bad].copy()
    for sym, g in sub.groupby("sym"):
        ids = g["idx"].to_numpy()
        starts = np.flatnonzero(np.r_[True, np.diff(ids) != 1])
        ends = np.r_[starts[1:], len(ids)]
        for s0, e0 in zip(starts, ends):
            rows = g.iloc[s0:e0]
            r0 = rows.iloc[0]
            cases.append({"sym": sym, "etf": bool(r0["is_etf"]), "start": r0["date"], "end": rows.iloc[-1]["date"], "days": int(e0 - s0),
                          "swing_dir": int(r0["zz10"]), "swing_pct": float(np.expm1(r0["zz10_swing"]) * 100),
                          "C0": float(r0["C"]), "T0": float(r0["T0"]), "T3": float(r0["T3"]), "K0": int(r0["z_K0"]), "K3n": int(r0["z_K3n"])})
    cdf = pd.DataFrame(cases).sort_values(["days", "swing_pct"], ascending=False)
    cdf.to_csv(os.path.join(HERE, "cases.csv"), index=False, encoding="utf-8-sig")
    res["case_counts"] = {"n_episodes": int(len(cdf)), "n_days": int(bad.sum()),
                          "etf_episodes": int(cdf["etf"].sum()) if len(cdf) else 0,
                          "len_ge3": int((cdf["days"] >= 3).sum()) if len(cdf) else 0}
    # 반대: K0가 틀리고 K3n이 맞는 같은 조건
    bad0 = M & (lab != 0) & (a == -lab) & (b == lab) & (df["zz10_since"] >= 5) & (df["zz10_until"] >= 3) & (df["zz10_swing"] >= np.log(1.2))
    res["case_counts"]["mirror_K0_wrong_days"] = int(bad0.sum())
    res["case_counts"]["K3n_wrong_days"] = int(bad.sum())

    df.to_parquet(os.path.join(HERE, "panel_feat.parquet"))

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else round(float(o), 5)
        return o
    with open(os.path.join(HERE, "fail_results.json"), "w", encoding="utf-8") as f:
        json.dump(clean(res), f, ensure_ascii=False, indent=1)

    # 요약 표
    L = []
    for key, grp in (("by_period", "주식150"), ("etf_by_period", "ETF12")):
        L.append(f"== {grp}")
        hdr = "지표".ljust(30) + "".join(f"{p}:K0".rjust(11) + f"{p}:K3n".rjust(11) for p in PERIODS)
        L.append(hdr)
        def row(name, fn):
            vals = []
            for p in PERIODS:
                for k in KS:
                    try:
                        v = fn(res[key][p][k])
                    except (KeyError, TypeError):
                        v = np.nan
                    vals.append(f"{v:.3f}".rjust(11) if isinstance(v, float) else str(v).rjust(11))
            L.append(name.ljust(30) + "".join(vals))
        row("역행 전환 비율", lambda d: d["rev_share"])
        row("바뀜/종목·연", lambda d: d["flips_py"])
        row("깜빡임(2일내 복귀)/종목·연", lambda d: d["flicker_py"])
        row("1일 구간 비율", lambda d: d["run1_share"])
        row("10일고점-10%: 하방", lambda d: d["dn10"]["down"])
        row("10일고점-10%: 애매", lambda d: d["dn10"]["gray"])
        row("10일고점-10%: 상방(반대)", lambda d: d["dn10"]["up"])
        row("10일저점+10%: 상방", lambda d: d["up10"]["up"])
        row("10일저점+10%: 애매", lambda d: d["up10"]["gray"])
        row("10일저점+10%: 하방(반대)", lambda d: d["up10"]["down"])
        row("횡보 바뀜/종목·연", lambda d: d["side_flips_py"])
        row("횡보 깜빡임/종목·연", lambda d: d["side_flicker"]["flicker_py"])
        row("횡보 역행 비율", lambda d: d["side_rev_share"])
        if key == "by_period":
            row("급락 지연 중앙", lambda d: d["crash_lag"]["median"])
            row("급락 지연 평균", lambda d: d["crash_lag"]["mean"])
            row("급등 지연 중앙", lambda d: d["rally_lag"]["median"])
            row("급등 지연 평균", lambda d: d["rally_lag"]["mean"])
            row("급락 사건 수", lambda d: d["crash_lag"]["n"])
            row("완만강세: 상방+극상방", lambda d: d["slow_up"]["up"])
            row("완만강세: 애매", lambda d: d["slow_up"]["gray"])
            row("완만강세: 하방", lambda d: d["slow_up"]["down"])
            row("완만약세: 하방", lambda d: d["slow_dn"]["down"])
            row("완만약세: 애매", lambda d: d["slow_dn"]["gray"])
            row("완만약세: 상방", lambda d: d["slow_dn"]["up"])
            row("상승중 눌림(-3~-8%): 하방", lambda d: d["dip_up"]["down"])
            row("상승중 눌림: 상방", lambda d: d["dip_up"]["up"])
            row("하락중 반등(+3~8%): 상방", lambda d: d["pop_dn"]["up"])
            row("하락중 반등: 하방", lambda d: d["pop_dn"]["down"])
            row("완만강세 n", lambda d: d["slow_up"]["n"])
            row("눌림 n", lambda d: d["dip_up"]["n"])
        for zc in ("zz10", "zz15", "zzv"):
            row(f"{zc} 일치", lambda d, zc=zc: d[zc]["agree"])
            row(f"{zc} 반대", lambda d, zc=zc: d[zc]["opp"])
            row(f"{zc} 상승스윙중 하방", lambda d, zc=zc: d[zc]["opp_in_up"])
            row(f"{zc} 하락스윙중 상방", lambda d, zc=zc: d[zc]["opp_in_dn"])
            row(f"{zc} 한가운데 반대", lambda d, zc=zc: d[zc + "_mid"]["opp"])
            row(f"{zc} 한가운데 일치", lambda d, zc=zc: d[zc + "_mid"]["agree"])
        L.append("불일치일(부호 다름) 누가 맞나: " + json.dumps({p: res[key][p]["disagree"] for p in PERIODS}, ensure_ascii=False))
    L.append("사례: " + json.dumps(res["case_counts"], ensure_ascii=False))
    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(HERE, "fail_results.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    print(cdf.head(40).to_string())


if __name__ == "__main__":
    main()
