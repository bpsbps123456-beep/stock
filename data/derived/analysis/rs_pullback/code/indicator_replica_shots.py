"""indicator_replica A: 2026-07 스크린샷 118행 사후 점검 (예시용, 표본 작음).

진입 = 스크린샷 뒤 첫 정규장 시가 (프리장 촬영: 그날 시가, 장중 촬영: 다음 날 시가).
보유 h = 1·5·10·20거래일, 매도 = 진입일 + h거래일 시가. 초과 = 같은 구간 QQQ 시가→시가 대비.
그룹 비교는 촬영(shot) 안에서 라벨을 섞는 순열 시험(10,000회)으로 p를 낸다.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "trend_grade", "code"))
import harness  # noqa: E402

RES = os.path.join(HERE, "..", "results")
SHOTS = os.path.join(RES, "indicator_shots_2026-07.csv")
HS = (1, 5, 10, 20)
# 촬영 시각(ET)과 진입일 / 마지막 완성 일봉
SHOT_INFO = {
    "S5": {"et": "2026-07-07 08:59 프리장", "bar": "2026-07-06", "entry": "2026-07-07"},
    "S6": {"et": "2026-07-08 11:09 장중", "bar": "2026-07-08(장중)", "entry": "2026-07-09"},
    "S1": {"et": "2026-07-09 09:08 프리장", "bar": "2026-07-08", "entry": "2026-07-09"},
    "S2": {"et": "2026-07-10 08:38 프리장", "bar": "2026-07-09", "entry": "2026-07-10"},
    "S3": {"et": "2026-07-13 10:21 장중", "bar": "2026-07-13(장중)", "entry": "2026-07-14"},
    "S4": {"et": "2026-07-14 09:19 프리장", "bar": "2026-07-13", "entry": "2026-07-14"},
}
NPERM = 10000
RNG = np.random.default_rng(20261013)


def load_opens(tickers: list[str]) -> tuple[pd.DataFrame, pd.Series, dict]:
    """패널 시가 우선, 패널 밖 종목은 yfinance(수정 안 한 Open, 이 구간 분할 없음 확인)."""
    P = harness.load_panel()
    O = P["O"]
    qqq = P["bench"]["O"]
    inside = [t for t in tickers if t in O.columns]
    outside = [t for t in tickers if t not in O.columns]
    out = O[inside].copy()
    meta = {"panel": inside, "yfinance": outside}
    import yfinance as yf
    d = yf.download(outside + inside[:3] + ["QQQ"], start="2026-06-01", end="2026-10-10", auto_adjust=False,
                    progress=False, group_by="ticker", actions=True)
    chk = {}
    for t in inside[:3] + ["QQQ"]:
        ref = qqq if t == "QQQ" else O[t]
        a = d[t]["Open"].reindex(ref.loc["2026-07-01":"2026-09-30"].index)
        chk[t] = round(float((a / ref.loc["2026-07-01":"2026-09-30"] - 1).abs().max() * 100), 4)
    meta["panel_vs_yf_max_abs_diff_pct"] = chk
    splits = {}
    for t in outside:
        s = d[t]["Open"].reindex(O.index)
        out[t] = s
        if "Stock Splits" in d[t]:
            ss = d[t]["Stock Splits"]
            splits[t] = int((ss.fillna(0) != 0).sum())
    meta["yf_split_events_jun_oct"] = splits
    return out, qqq, meta


def forward_table(df: pd.DataFrame, O: pd.DataFrame, qqq: pd.Series) -> pd.DataFrame:
    cal = O.index
    rows = []
    for _, r in df.iterrows():
        e = pd.Timestamp(SHOT_INFO[r.shot]["entry"])
        i = cal.get_loc(e)
        rec = {}
        for h in HS:
            if i + h >= len(cal):
                rec[f"r{h}"] = rec[f"x{h}"] = np.nan
                continue
            x = cal[i + h]
            rr = O.at[x, r.ticker] / O.at[e, r.ticker] - 1
            rq = qqq.at[x] / qqq.at[e] - 1
            rec[f"r{h}"] = rr * 100
            rec[f"x{h}"] = (rr - rq) * 100
        rows.append(rec)
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def perm_diff(df: pd.DataFrame, flag: str, y: str) -> dict:
    """flag(bool) 그룹 평균 − 나머지 평균. 촬영 안에서 flag를 섞어 양측 p."""
    d = df.dropna(subset=[y, flag]).copy()
    d[flag] = d[flag].astype(bool)
    if d[flag].sum() == 0 or (~d[flag]).sum() == 0:
        return {}
    obs = d.loc[d[flag], y].mean() - d.loc[~d[flag], y].mean()
    # 촬영 평균을 뺀 값(같은 날 시장 효과 제거)
    dm = d[y] - d.groupby("shot")[y].transform("mean")
    obs_dm = dm[d[flag]].mean() - dm[~d[flag]].mean()
    groups = [np.where(d["shot"].to_numpy() == s)[0] for s in d["shot"].unique()]
    f = d[flag].to_numpy()
    v = dm.to_numpy()
    cnt = 0
    for _ in range(NPERM):
        g = f.copy()
        for idx in groups:
            g[idx] = RNG.permutation(g[idx])
        stat = v[g].mean() - v[~g].mean()
        cnt += abs(stat) >= abs(obs_dm) - 1e-12
    return {"n_in": int(d[flag].sum()), "n_out": int((~d[flag]).sum()),
            "mean_in": round(float(d.loc[d[flag], y].mean()), 2),
            "mean_out": round(float(d.loc[~d[flag], y].mean()), 2),
            "diff": round(float(obs), 2), "diff_within_shot": round(float(obs_dm), 2),
            "perm_p": round((cnt + 1) / (NPERM + 1), 3)}


def within_shot_spearman(df: pd.DataFrame, col: str, y: str) -> dict:
    d = df.dropna(subset=[col, y])
    rhos = []
    for s, g in d.groupby("shot"):
        if len(g) >= 5:
            rhos.append(g[col].rank().corr(g[y].rank()))
    obs = float(np.mean(rhos))
    cnt = 0
    for _ in range(2000):
        rr = []
        for s, g in d.groupby("shot"):
            if len(g) >= 5:
                rr.append(pd.Series(RNG.permutation(g[col].to_numpy())).rank().corr(
                    pd.Series(g[y].to_numpy()).rank()))
        cnt += abs(np.mean(rr)) >= abs(obs) - 1e-12
    return {"mean_rho": round(obs, 3), "per_shot": [round(float(r), 3) for r in rhos],
            "perm_p": round((cnt + 1) / 2001, 3)}


def main() -> None:
    raw = pd.read_csv(SHOTS)
    tick = sorted(raw.ticker.unique())
    O, qqq, meta = load_opens(tick)
    df = forward_table(raw, O, qqq)

    # 촬영 안 기준 라벨
    df["state_ge1"] = df.state >= 1
    df["state2"] = df.state == 2
    df["score_top5"] = df.groupby("shot")["score"].rank(ascending=False, method="first") <= 5
    med = lambda c: df.groupby("shot")[c].transform("median")
    df["fade_hi"] = (df.fade_gap > med("fade_gap")).where(df.fade_gap.notna())
    df["rsmain_hi"] = df.rs_main > med("rs_main")
    df["latent_hi"] = df.latent > med("latent")

    out = {"meta": meta, "shots": SHOT_INFO, "n_rows": int(len(df)),
           "n_unique_tickers": int(df.ticker.nunique()),
           "n_unique_ticker_entry": int(df.assign(e=df.shot.map(lambda s: SHOT_INFO[s]["entry"]))
                                        .drop_duplicates(["ticker", "e"]).shape[0]),
           "entry_dates": sorted({v["entry"] for v in SHOT_INFO.values()})}
    # 맥락: QQQ·SOXX 경로
    ctx = {}
    for t, s in (("QQQ", qqq), ("SOXX", O.get("SOXX"))):
        if s is None:
            continue
        def ret(a, b):
            return round(float(s.loc[b] / s.loc[a] - 1) * 100, 2)
        ctx[t] = {"open 06-22→07-29": ret("2026-06-22", "2026-07-29"),
                  "open 07-07→08-04": ret("2026-07-07", "2026-08-04"),
                  "open 07-14→08-11": ret("2026-07-14", "2026-08-11")}
    out["context"] = ctx

    # 상태별 평균
    st = {}
    for h in HS:
        y = f"x{h}"
        st[f"h{h}"] = {int(k): {"n": int(len(g)), "mean_x": round(float(g[y].mean()), 2),
                                "median_x": round(float(g[y].median()), 2),
                                "hit": round(float((g[y] > 0).mean()) * 100, 1)}
                       for k, g in df.groupby("state")}
        st[f"h{h}"]["all"] = {"n": int(df[y].notna().sum()), "mean_x": round(float(df[y].mean()), 2),
                              "median_x": round(float(df[y].median()), 2)}
    out["by_state"] = st

    comps = {}
    for flag in ("state_ge1", "state2", "score_top5", "fade_hi", "rsmain_hi", "latent_hi"):
        comps[flag] = {f"h{h}": perm_diff(df, flag, f"x{h}") for h in HS}
    out["group_diffs"] = comps
    out["spearman_within_shot"] = {c: {f"h{h}": within_shot_spearman(df, c, f"x{h}") for h in (5, 10, 20)}
                                   for c in ("score", "rs_main", "latent", "fade_gap", "rs_short")}

    # 쏠림: 한 종목 빼기(촬영 안 차이, h10·h20), SNDK·SOXX 지정
    loo = {}
    for flag in ("state_ge1", "score_top5", "fade_hi", "rsmain_hi"):
        for h in (10, 20):
            y = f"x{h}"
            base = comps[flag][f"h{h}"]["diff_within_shot"]
            vals = {}
            for t in tick:
                d = df[df.ticker != t]
                r = d.dropna(subset=[y, flag])
                f_ = r[flag].astype(bool)
                dm = r[y] - r.groupby("shot")[y].transform("mean")
                if f_.sum() and (~f_).sum():
                    vals[t] = float(dm[f_].mean() - dm[~f_].mean())
            s = pd.Series(vals)
            loo[f"{flag}_h{h}"] = {"base": base, "min": round(float(s.min()), 2), "min_drop": s.idxmin(),
                                   "max": round(float(s.max()), 2), "max_drop": s.idxmax(),
                                   "drop_SNDK": round(vals.get("SNDK", np.nan), 2),
                                   "drop_SOXX": round(vals.get("SOXX", np.nan), 2),
                                   "sign_flips": int((np.sign(s) != np.sign(base)).sum())}
    out["leave_one_ticker_out"] = loo

    # 중복 제거판: 같은 진입일 촬영(S6·S1, S3·S4)은 앞 촬영 하나만 남긴다
    dd = df[df.shot.isin(["S5", "S6", "S2", "S3"])]
    out["dedup_entry"] = {"shots": ["S5", "S6", "S2", "S3"], "n": int(len(dd))}
    for flag in ("state_ge1", "score_top5", "fade_hi", "rsmain_hi"):
        for h in (5, 10, 20):
            out["dedup_entry"][f"{flag}_h{h}"] = perm_diff(dd, flag, f"x{h}")

    # 종목 묶음 부트스트랩: 같은 종목이 여러 촬영에 반복되므로 종목 단위로 다시 뽑아 촬영 안 차이의 95% 구간
    boot = {}
    for flag in ("state_ge1", "score_top5", "fade_hi", "rsmain_hi"):
        for h in (10, 20):
            y = f"x{h}"
            r = df.dropna(subset=[y, flag]).copy()
            r["dm"] = r[y] - r.groupby("shot")[y].transform("mean")
            r["f"] = r[flag].astype(bool)
            by = {t: g for t, g in r.groupby("ticker")}
            names = list(by)
            vals = []
            for _ in range(5000):
                pick = RNG.choice(names, size=len(names), replace=True)
                g = pd.concat([by[t] for t in pick])
                if g.f.sum() and (~g.f).sum():
                    vals.append(g.dm[g.f].mean() - g.dm[~g.f].mean())
            v = np.array(vals)
            boot[f"{flag}_h{h}"] = {"ci95": [round(float(np.percentile(v, 2.5)), 2),
                                             round(float(np.percentile(v, 97.5)), 2)],
                                    "share_gt0": round(float((v > 0).mean()), 3)}
    out["ticker_cluster_bootstrap"] = boot

    # 행 자료(숫자만)
    keep = ["shot", "ticker", "score", "state", "rs_main", "latent", "fade_gap"] + [f"x{h}" for h in HS]
    out["rows"] = df[keep].round(3).astype(object).where(df[keep].notna(), None).to_dict("records")
    path = os.path.join(RES, "indicator_replica_shots.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
    print(path)
    print(json.dumps({k: out[k] for k in ("meta", "context", "by_state")}, ensure_ascii=False, indent=0)[:4000])
    for flag, v in comps.items():
        print(flag, {h: (x.get("diff_within_shot"), x.get("perm_p")) for h, x in v.items()})
    for c, v in out["spearman_within_shot"].items():
        print("rho", c, {h: (x["mean_rho"], x["perm_p"]) for h, x in v.items()})
    print(json.dumps(loo, ensure_ascii=False))
    print(json.dumps(out["dedup_entry"], ensure_ascii=False))
    print(json.dumps(boot, ensure_ascii=False))


if __name__ == "__main__":
    main()
