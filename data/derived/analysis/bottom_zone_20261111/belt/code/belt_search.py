"""바닥 벨트 2차 탐색 (BRIEF_belt.md 고정 정의).

1차 과제 A의 신호판·가격·CNN·과제 B 뉴스 계열을 읽어 조건 49개를 만들고,
1·2·3개 AND 조합 19,649개의 벨트10 적중을 2024~25(고르기)·2026(확인)으로 잰다.
벨트 표시를 연도 안에서 원형 이동한 우연 대조를 함께 낸다.

실행: py data/derived/analysis/bottom_zone_20261111/belt/code/belt_search.py
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]          # bottom_zone_20261111
OUT = ROOT / "belt" / "results"
SEED = 20261112
N_SHIFT = 500
POST = 20                                            # 저점 뒤 벨트로 보는 세션 수
SELECT_END = "2025-12-31"


# ---------------------------------------------------------------- 입력
def load():
    panel = pd.read_csv(ROOT / "data" / "signal_panel.csv", parse_dates=["date"]).set_index("date")
    px = pd.read_csv(ROOT / "data" / "prices.csv", parse_dates=["date"])
    wide = {t: g.set_index("date").sort_index() for t, g in px.groupby("ticker")}
    cnn = json.loads((ROOT / "data" / "cnn.json").read_text(encoding="utf-8"))
    news = pd.read_csv(ROOT / "news" / "v2" / "news_series_B.csv", parse_dates=["date"], encoding="utf-8-sig")
    return panel, wide, cnn, news


def pct_rank(s: pd.Series, win: int = 250, min_n: int = 120) -> pd.Series:
    """당일 포함 최근 win개 안에서의 백분위(0~100, 중간 동률)."""
    v = s.to_numpy(float)
    out = np.full(len(v), np.nan)
    for i in range(len(v)):
        if np.isnan(v[i]):
            continue
        w = v[max(0, i - win + 1): i + 1]
        w = w[~np.isnan(w)]
        if len(w) >= min_n:
            out[i] = 100.0 * ((w < v[i]).sum() + 0.5 * (w == v[i]).sum()) / len(w)
    return pd.Series(out, index=s.index)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn)


def cnn_series(cnn: dict, key: str) -> pd.Series:
    rows = cnn[key]["data"]
    s = pd.Series({pd.Timestamp(r["x"], unit="ms").normalize(): r["y"] for r in rows}).sort_index()
    return s[~s.index.duplicated(keep="last")]


# ---------------------------------------------------------------- 조건 49개
def build_atoms(panel, wide, cnn, news) -> pd.DataFrame:
    idx = panel.index                                  # 관찰일 격자(마지막 날은 시가 없음)
    sx = wide["SOXX"].reindex(idx)
    c = sx["close"]

    def asof(s: pd.Series) -> pd.Series:               # 관찰일 직전 종가까지만
        return s.reindex(idx).shift(1)

    a = pd.DataFrame(index=idx)
    # 가격 13
    a["낙폭15"] = panel["dd60"] <= -0.15
    a["낙폭20"] = panel["dd60"] <= -0.20
    a["낙폭25"] = panel["dd60"] <= -0.25
    a["5일-8"] = panel["ret5"] <= -0.08
    a["5일-12"] = panel["ret5"] <= -0.12
    a["10일-15"] = asof(c / c.shift(10) - 1) <= -0.15
    r = asof(rsi(c))
    a["RSI30"] = r <= 30
    a["RSI25"] = r <= 25
    a["200일선아래"] = panel["M200_value"] < 0
    a["200일선-10"] = panel["M200_value"] <= -10
    newlow = (c == c.rolling(20).min()).astype(float)
    a["신저가3일내"] = asof(newlow.rolling(3).max()) >= 1
    a["전일+3"] = panel["ret1"] >= 0.03
    low10 = c.rolling(10).min()
    a["저점+5반등"] = asof(((low10 == c.rolling(60).min()) & (c >= 1.05 * low10)).astype(float)) >= 1
    # 거래량 3
    for t in ("SOXX", "SOXL", "SOXS"):
        a[f"{t}거래량"] = asof(pct_rank(wide[t]["volume"].reindex(idx))) >= 90
    # 공포 11
    g = panel["G"]
    a["공포25"] = g <= 25
    a["공포20"] = g <= 20
    a["공포15"] = g <= 15
    a["공포완화"] = (g > g.shift(1)) & (g.rolling(5).min() <= 25)
    vix = asof(cnn_series(cnn, "market_volatility_vix"))
    vix50 = asof(cnn_series(cnn, "market_volatility_vix_50"))
    a["VIX30"] = vix >= 30
    a["VIX급등"] = vix / vix50 >= 1.25
    a["풋콜상위"] = asof(pct_rank(cnn_series(cnn, "put_call_options").reindex(idx))) >= 90
    a["안전자산"] = asof(pct_rank(cnn_series(cnn, "safe_haven_demand").reindex(idx))) <= 10
    a["시장폭하위"] = asof(pct_rank(cnn_series(cnn, "stock_price_breadth").reindex(idx))) <= 10
    a["신고가하위"] = asof(pct_rank(cnn_series(cnn, "stock_price_strength").reindex(idx))) <= 10
    a["정크스프레드"] = asof(pct_rank(cnn_series(cnn, "junk_bond_demand").reindex(idx))) >= 90
    # 한국인 수급 13 (신호판이 이미 공개 시점에 맞춰져 있다)
    a["SOXS몫하위20(1)"] = panel["S1_P"] <= 20
    a["SOXS몫하위10(1)"] = panel["S1_P"] <= 10
    a["SOXS몫하위20(5)"] = panel["S5_P"] <= 20
    a["SOXS몫하위10(5)"] = panel["S5_P"] <= 10
    a["SOXS몫상위20(1)"] = panel["S1_P"] >= 80
    a["SOXS몫상위20(5)"] = panel["S5_P"] >= 80
    a["SOXL비중상위20"] = panel["T5_P"] >= 80
    a["SOXL비중상위10"] = panel["T5_P"] >= 90
    a["SOXL보관상위20"] = panel["C_P"] >= 80
    a["전체순매도"] = panel["X2"] == 1
    a["매수반감"] = panel["X1_70"] == 1
    a["보관좌수감소"] = panel["X3"] == 1
    a["SOXL매수상위10"] = pct_rank(panel["buy5"]) >= 90
    # 뉴스 9
    for k, name in (("ND", "PB매도"), ("NR", "PB매수복귀"), ("NS", "PB숏쏠림"), ("NT", "PB롱쏠림")):
        a[name] = panel[k] == 1
    pos = {d: i for i, d in enumerate(idx)}
    dates = idx.to_numpy()
    eff = np.searchsorted(dates, news["date"].to_numpy(), side="right")   # 공개일 다음 세션
    fam = {"U": "전략가상향", "DN": "전략가하향", "TU": "목표상향", "TC": "목표하향"}
    for k, name in fam.items():
        flag = np.zeros(len(idx), bool)
        for e in eff[(news["family"] == k).to_numpy()]:
            flag[e: e + 5] = True
        a[name] = flag
    houses = [set() for _ in idx]
    neg = news[news["family"].isin(["DN", "TC"])]
    for e, h in zip(np.searchsorted(dates, neg["date"].to_numpy(), side="right"), neg["house"]):
        for i in range(e, min(e + 5, len(idx))):
            houses[i].add(h)
    a["하향몰림"] = [len(h) >= 2 for h in houses]
    assert a.shape[1] == 49, a.shape
    return a.fillna(False).astype(bool)


# ---------------------------------------------------------------- 결과 표시
def zigzag(v: np.ndarray, thr: float = 0.10):
    """종가 10% 반전 지그재그. (종류, 극값 위치, 확정 위치) 목록과 끝의 방향을 준다."""
    piv, mode = [], None
    hi = lo = v[0]
    hi_i = lo_i = 0
    for i in range(1, len(v)):
        x = v[i]
        if mode is None:
            if x > hi:
                hi, hi_i = x, i
            if x < lo:
                lo, lo_i = x, i
            if x <= hi * (1 - thr):
                piv.append(("T", hi_i, i)); mode = "down"; lo, lo_i = x, i
            elif x >= lo * (1 + thr):
                piv.append(("B", lo_i, i)); mode = "up"; hi, hi_i = x, i
        elif mode == "up":
            if x > hi:
                hi, hi_i = x, i
            elif x <= hi * (1 - thr):
                piv.append(("T", hi_i, i)); mode = "down"; lo, lo_i = x, i
        else:
            if x < lo:
                lo, lo_i = x, i
            elif x >= lo * (1 + thr):
                piv.append(("B", lo_i, i)); mode = "up"; hi, hi_i = x, i
    return piv, mode


def swing_lows(close: np.ndarray, merge: int = 15):
    """지그재그 저점을 15세션 안 군집은 가장 낮은 하나로 합친다(§12.7 B10과 같은 규칙)."""
    piv, mode = zigzag(close)
    raw = [i for k, i, _ in piv if k == "B"]
    lows: list[int] = []
    for i in raw:
        if lows and i - lows[-1] <= merge:
            if close[i] < close[lows[-1]]:
                lows[-1] = i
        else:
            lows.append(i)
    return lows, mode


def labels(panel, wide) -> pd.DataFrame:
    """벨트 = 그 하락의 지그재그 저점 종가 +10%(+5%) 안의 관찰일 시가.

    저점 앞: 저점으로 내려가는 구간(직전 고점 뒤)의 날. 저점 뒤: POST세션까지.
    확정 저점이 아직 없는 진행 중 하락 구간은 미정(NaN)이다.
    """
    idx = panel.index
    close_s = wide["SOXX"]["close"].reindex(idx)
    last = int(np.where(close_s.notna().to_numpy())[0][-1])
    close = close_s.to_numpy(float)
    opn = panel["open"].to_numpy(float)
    lows, mode = swing_lows(close[: last + 1])
    tops = []                                           # 각 저점으로 내려가기 시작한 고점
    for k, b in enumerate(lows):
        s = lows[k - 1] if k else 0
        tops.append(s + int(np.argmax(close[s: b + 1])))
    out = pd.DataFrame(index=idx, columns=["low", "belt10", "belt5", "r20"], dtype=float)
    anchor = np.full(len(idx), -1)
    for i in range(len(idx)):
        if i + 19 <= last and not np.isnan(opn[i]):
            out.iat[i, 3] = close[i + 19] / opn[i] - 1
        if np.isnan(opn[i]):
            continue
        prev = max((b for b in lows if b < i), default=None)
        nxt_k = next((k for k, b in enumerate(lows) if b >= i), None)
        cands = []                                      # (저점 위치, 세션 거리)
        if prev is not None and i - prev <= POST:
            cands.append((prev, i - prev))
        if nxt_k is not None and i >= tops[nxt_k]:
            cands.append((lows[nxt_k], lows[nxt_k] - i))
        if not cands:
            if nxt_k is None and mode == "down":        # 저점이 아직 확정되지 않은 하락
                continue
            anchor[i] = prev if prev is not None else -1
            out.iloc[i, 0:3] = [np.nan if prev is None else close[prev], 0.0, 0.0]
            continue
        inside = [(b, d) for b, d in cands if opn[i] <= 1.10 * close[b]]
        b = min(inside or cands, key=lambda t: t[1])[0]
        anchor[i] = b
        out.iat[i, 0] = close[b]
        out.iat[i, 1] = float(opn[i] <= 1.10 * close[b])
        out.iat[i, 2] = float(opn[i] <= 1.05 * close[b])
    out["anchor"] = [str(idx[a].date()) if a >= 0 else "" for a in anchor]
    out["episode"] = anchor
    return out


# ---------------------------------------------------------------- 탐색
def combos(n: int):
    for k in (1, 2, 3):
        yield from itertools.combinations(range(n), k)


def on_matrix(A: np.ndarray, cs: list[tuple]) -> np.ndarray:
    M = np.empty((len(cs), A.shape[0]), bool)
    for r, c in enumerate(cs):
        M[r] = A[:, list(c)].all(axis=1)
    return M


def period_stats(M, lab10, lab5, r20, ep, mask):
    """mask 구간에서 조합별 켜진 날·벨트 비율·국면 수·첫 켜짐 적중."""
    Mp = M[:, mask]
    l10, l5, rr, e = lab10[mask], lab5[mask], r20[mask], ep[mask]
    n = Mp.sum(1)
    hit10 = Mp @ l10
    ok5 = ~np.isnan(l5)
    hit5 = Mp[:, ok5] @ l5[ok5]
    n5 = Mp[:, ok5].sum(1)
    okr = ~np.isnan(rr)
    rsum = Mp[:, okr] @ rr[okr]
    nr = Mp[:, okr].sum(1)
    n_epi = np.zeros(len(M), int)
    first_hit = np.zeros(len(M), int)
    for k in np.unique(e):
        cols = np.where(e == k)[0]
        sub = Mp[:, cols]
        anyon = sub.any(1)
        first = sub.argmax(1)
        n_epi += anyon
        first_hit += anyon & (l10[cols][first] == 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return dict(n=n, belt10=hit10 / n, belt5=hit5 / n5, n_epi=n_epi, first_hit=first_hit,
                    first_rate=first_hit / n_epi, r20=rsum / nr)


def both_pass(n1, p1, e1, n2, p2, e2):
    return (n1 >= 4) & (n2 >= 4) & (e1 >= 2) & (e2 >= 2) & (p1 >= 0.8) & (p2 >= 0.8)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel, wide, cnn, news = load()
    atoms = build_atoms(panel, wide, cnn, news)
    lab = labels(panel, wide)
    ep = lab["episode"]

    keep = (panel["down"] == True) & lab["belt10"].notna() & (panel.index >= "2024-01-02")  # noqa: E712
    idx = panel.index[keep]
    A = atoms.loc[idx].to_numpy()
    l10 = lab.loc[idx, "belt10"].to_numpy(float)
    l5 = lab.loc[idx, "belt5"].to_numpy(float)
    r20 = lab.loc[idx, "r20"].to_numpy(float)
    e = ep.loc[idx].to_numpy()
    sel = np.asarray(idx <= SELECT_END)
    con = ~sel
    names = list(atoms.columns)
    cs = list(combos(len(names)))
    assert len(cs) == 19649
    M = on_matrix(A, cs)

    s1 = period_stats(M, l10, l5, r20, e, sel)
    s2 = period_stats(M, l10, l5, r20, e, con)
    label = [" + ".join(names[i] for i in c) for c in cs]
    df = pd.DataFrame({"조합": label, "조건수": [len(c) for c in cs]})
    for tag, s in (("sel", s1), ("con", s2)):
        for k, v in s.items():
            df[f"{tag}_{k}"] = v
    df["both"] = both_pass(s1["n"], s1["belt10"], s1["n_epi"], s2["n"], s2["belt10"], s2["n_epi"])
    df.to_csv(OUT / "all_combos.csv", index=False, encoding="utf-8-sig")

    base = dict(
        sel=dict(days=int(sel.sum()), belt10=float(l10[sel].mean()), belt5=float(np.nanmean(l5[sel])),
                 episodes=int(len(np.unique(e[sel])))),
        con=dict(days=int(con.sum()), belt10=float(l10[con].mean()), belt5=float(np.nanmean(l5[con])),
                 episodes=int(len(np.unique(e[con])))),
    )

    # 우연 대조: 벨트 표시를 연도 안에서 원형 이동
    rng = np.random.default_rng(SEED)
    years = pd.DatetimeIndex(idx).year.to_numpy()
    null_count, null_best = [], []
    elig = (s1["n"] >= 4) & (s2["n"] >= 4) & (s1["n_epi"] >= 2) & (s2["n_epi"] >= 2)
    Me = M[elig]
    n1, n2 = Me[:, sel].sum(1), Me[:, con].sum(1)
    for _ in range(N_SHIFT):
        sh = l10.copy()
        for y in np.unique(years):
            cols = np.where(years == y)[0]
            sh[cols] = np.roll(l10[cols], int(rng.integers(1, len(cols))))
        p1 = (Me[:, sel] @ sh[sel]) / n1
        p2 = (Me[:, con] @ sh[con]) / n2
        null_count.append(int(((p1 >= 0.8) & (p2 >= 0.8)).sum()))
        null_best.append(float(np.minimum(p1, p2).max()))
    obs_count = int(df["both"].sum())
    obs_best = float(np.minimum(s1["belt10"], s2["belt10"])[elig].max())
    null = dict(
        shifts=N_SHIFT, seed=SEED, eligible=int(elig.sum()),
        observed_both_pass=obs_count,
        null_both_pass_mean=float(np.mean(null_count)), null_both_pass_median=float(np.median(null_count)),
        null_both_pass_p95=float(np.percentile(null_count, 95)),
        p_count=float((1 + sum(c >= obs_count for c in null_count)) / (N_SHIFT + 1)),
        observed_best_min_rate=obs_best,
        p_best=float((1 + sum(b >= obs_best for b in null_best)) / (N_SHIFT + 1)),
    )

    # 순위표
    cand = df[(df.sel_n >= 6) & (df.sel_n_epi >= 3)].sort_values(
        ["sel_first_rate", "sel_belt10", "sel_n"], ascending=False)
    cand.head(40).to_csv(OUT / "top_select.csv", index=False, encoding="utf-8-sig")
    rev = df[(df.con_n >= 6) & (df.con_n_epi >= 3)].sort_values(
        ["con_first_rate", "con_belt10", "con_n"], ascending=False)
    rev.head(40).to_csv(OUT / "top_2026.csv", index=False, encoding="utf-8-sig")
    both = df[df.both].assign(minrate=lambda d: d[["sel_belt10", "con_belt10"]].min(axis=1),
                              tot=lambda d: d.sel_n + d.con_n).sort_values(
        ["minrate", "tot"], ascending=False)
    both.to_csv(OUT / "both_pass.csv", index=False, encoding="utf-8-sig")

    # 날짜별 원장(검산·지금 상태용)
    led = pd.concat([panel[["open", "close", "dd60", "down"]], lab, atoms], axis=1)
    led.loc[led.index >= "2024-01-02"].to_csv(OUT / "daily_belt.csv", encoding="utf-8-sig")

    json.dump(dict(base=base, null=null, n_combos=len(cs), n_atoms=len(names), atoms=names,
                   rows=int(len(idx)), select_end=SELECT_END),
              open(OUT / "summary_belt.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(dict(base=base, null=null), ensure_ascii=False, indent=1))
    cols = ["조합", "sel_n", "sel_n_epi", "sel_first_hit", "sel_belt10", "con_n", "con_n_epi", "con_first_hit", "con_belt10"]
    pd.set_option("display.width", 250, "display.max_colwidth", 60)
    print("--- 2024~25로 고른 상위 15"); print(cand[cols].head(15).to_string(index=False))
    print("--- 양쪽 통과 상위 25 (전체", len(both), "개)"); print(both[cols].head(25).to_string(index=False))


if __name__ == "__main__":
    main()
