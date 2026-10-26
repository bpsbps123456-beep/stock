"""N1 공통 계산: Codex analyze.py의 거시 조건 M 정의를 그대로 옮긴다.

- 조건(신호일 k 종가까지 앎): 지수 5거래일 종가 수익 < 0 그리고 ^TNX 5거래일 변화 < thr(기본 0)
- 첫 발생: condition & ~condition.shift(1)
- 진입 i = k+1(다음 장 시가), 직전 채택 진입과 최소 5거래일 간격(i - prev >= 5)
- 결과 r_h = 종가[i+h-1] / 시가[i] - 1 (진입 포함 h번째 장 종가)
- 대조 (a) 같은 해(진입일 연도) 모든 날, (b) 같은 해 + 직전 20일 고점 대비 낙폭 구간(0~5·5~10·10%+) 같은 날
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (1, 3, 5, 10, 20)


def daily_table(op: pd.Series, cl: pd.Series, tnx: pd.Series) -> pd.DataFrame:
    """op, cl, tnx: 같은 세션 색인. 지수 값이 없는 날(상장 전)은 미리 잘라서 넘긴다."""
    d = pd.DataFrame(index=cl.index)
    d["i"] = np.arange(len(d))
    d["year"] = d.index.year
    d["month"] = d.index.year * 12 + d.index.month - 1
    d["quarter"] = d.index.year * 4 + (d.index.month - 1) // 3
    d["dd20"] = (cl / cl.rolling(20, min_periods=20).max() - 1).shift(1)
    d["bucket"] = pd.cut(-d.dd20, [-np.inf, .05, .10, np.inf], right=False, labels=[0, 1, 2]).astype("Int64")
    ma200 = cl.rolling(200, min_periods=200).mean()
    d["above200"] = (cl > ma200).where(ma200.notna()).shift(1)  # 신호일 종가 기준, 진입 행에 둔다
    d["ret5_sig"] = cl.pct_change(5, fill_method=None).shift(1)
    d["dy5_sig"] = tnx.diff(5).shift(1)
    for h in HORIZONS:
        d[f"r{h}"] = cl.shift(-(h - 1)) / op - 1
    return d


def conditions(cl: pd.Series, tnx: pd.Series, thr: float = 0.0) -> dict[str, pd.Series]:
    fall = cl.pct_change(5, fill_method=None) < 0
    dy = tnx.diff(5)
    return {"M": fall & (dy < thr), "C": fall, "D": fall & (dy > 0)}


def onsets(cond: pd.Series, sig_year_min: int | None = None, gap: int = 5) -> list[int]:
    """Codex 반복문 그대로: k=1..n-2, i=k+1, 첫 발생, i-prev>=gap. 반환은 진입 위치 i."""
    c = cond.to_numpy(dtype=bool)
    on = c & ~np.r_[False, c[:-1]]
    years = cond.index.year
    n = len(c)
    prev = -100
    out = []
    for k in range(1, n - 1):
        i = k + 1
        if sig_year_min is not None and years[k] < sig_year_min:
            continue
        if not on[k] or i - prev < gap:
            continue
        prev = i
        out.append(i)
    return out


def attach_controls(daily: pd.DataFrame, entries: list[int], base_mask: pd.Series | None = None,
                    regime_match: bool = False) -> pd.DataFrame:
    """신호별 (a)·(b) 대조값을 붙인다. base_mask: 대조 표본으로 쓸 날(기본 전부)."""
    ev = daily.iloc[entries].copy()
    base = daily if base_mask is None else daily[base_mask]
    keys = ["year", "bucket"] + (["above200"] if regime_match else [])
    kb = base[keys].astype("float64")
    ke = ev[keys].astype("float64")
    for h in HORIZONS:
        col = f"r{h}"
        ok = base[col].notna()
        y = base.loc[ok, col]
        t = pd.DataFrame({"y": y, "hit": (y > 0).astype(float)}).join(kb[ok])
        a = t.groupby("year")[["y", "hit"]].mean()
        bb = t.dropna(subset=keys).groupby(keys)[["y", "hit"]].mean()
        ev[f"a_hit{h}"] = ke.year.map(a.hit).to_numpy()
        ev[f"a_mean{h}"] = ke.year.map(a.y).to_numpy()
        m = ke.merge(bb, left_on=keys, right_index=True, how="left")
        ev[f"b_hit{h}"] = m.hit.to_numpy()
        ev[f"b_mean{h}"] = m.y.to_numpy()
    return ev


def summarize(ev: pd.DataFrame, h: int) -> dict:
    v = ev[ev[f"r{h}"].notna()]
    r = v[f"r{h}"]
    out = dict(n=len(v), wins=int((r > 0).sum()), hit=float((r > 0).mean()) if len(v) else np.nan,
               mean=float(r.mean()) if len(v) else np.nan)
    for c in ["a_hit", "a_mean", "b_hit", "b_mean"]:
        col = f"{c}{h}"
        out[c] = float(v[col].mean()) if col in v and len(v) else np.nan
    return out


def block_boot(groups: dict[str, pd.DataFrame], h: int, months: np.ndarray, B: int = 5000,
               seed: int = 0, col: str = "month") -> dict:
    """달력 월 묶음 부트스트랩(같은 날·같은 달 신호를 함께 뽑는다).
    groups: 'M','C','D' 신호표(진입 행 + b_hit/b_mean). 반환: 차이의 점추정과 95% 구간."""
    rng = np.random.default_rng(seed)
    mi = {m: j for j, m in enumerate(months)}
    W = rng.multinomial(len(months), np.full(len(months), 1 / len(months)), size=B)  # 월을 복원 추출한 횟수
    res = {}
    stats = {}
    for name, ev in groups.items():
        v = ev[ev[f"r{h}"].notna() & ev[col].isin(list(mi.keys()))]
        idx = v[col].map(mi).to_numpy()
        w = W[:, idx].astype(float)  # B x n
        r = v[f"r{h}"].to_numpy()
        hit = (r > 0).astype(float)
        den = w.sum(1)
        den[den == 0] = np.nan
        st = dict(mean=(w @ r) / den, hit=(w @ hit) / den)
        if name == "M":
            ok = v[f"b_mean{h}"].notna().to_numpy()  # 상장 직후 20일은 낙폭 구간이 없어 (b) 비교에서 뺀다
            wb = w[:, ok]
            denb = wb.sum(1)
            denb = np.where(denb == 0, np.nan, denb)
            em = r[ok] - v[f"b_mean{h}"].to_numpy()[ok]
            eh = hit[ok] - v[f"b_hit{h}"].to_numpy()[ok]
            st["dmean_b"] = (wb @ em) / denb
            st["dhit_b"] = (wb @ eh) / denb
            res["dmean_b"] = float(np.mean(em)) if ok.any() else np.nan
            res["dhit_b"] = float(np.mean(eh)) if ok.any() else np.nan
        stats[name] = st
        res[f"mean_{name}"] = float(r.mean()) if len(r) else np.nan
        res[f"hit_{name}"] = float(hit.mean()) if len(r) else np.nan
    def ci(x):
        x = x[np.isfinite(x)]
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] if len(x) else [np.nan, np.nan]
    M = stats["M"]
    res["dmean_b_ci"] = ci(M["dmean_b"])
    res["dhit_b_ci"] = ci(M["dhit_b"])
    for other in [k for k in groups if k != "M"]:
        o = stats[other]
        res[f"dmean_{other}"] = res["mean_M"] - res[f"mean_{other}"]
        res[f"dhit_{other}"] = res["hit_M"] - res[f"hit_{other}"]
        res[f"dmean_{other}_ci"] = ci(M["mean"] - o["mean"])
        res[f"dhit_{other}_ci"] = ci(M["hit"] - o["hit"])
    res["mean_M_ci"] = ci(M["mean"])
    return res
