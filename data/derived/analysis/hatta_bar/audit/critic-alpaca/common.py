"""critic-alpaca 공통: code/ 모듈을 읽기 전용으로 불러 쓴다(수정 안 함)."""
import os, sys, json
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, "../../code"))
sys.path.insert(0, CODE)
import engine as E   # noqa
import grid as G     # noqa
import evaluate as V  # noqa
RES = os.path.abspath(os.path.join(HERE, "../../results"))
OUT = os.path.join(HERE, "out")
SPLIT = 20260831
CELLS = ["orig_A", "orig_B", "top_A", "top_B"]


def specs():
    picks = json.load(open(os.path.join(RES, "pick.json"), encoding="utf-8"))
    d = {"orig_A": ("A", V.ORIG["A"]), "orig_B": ("B", V.ORIG["B"])}
    for pat in ("A", "B"):
        d[f"top_{pat}"] = (pat, V.spec_of(picks[pat], pat))
    return d


def sig_ctx(name, tradable=True):
    pat, spec = specs()[name]
    return V.cell_rows("cand_alp1m", pat, spec, tradable=tradable)


def day_boot(dfs, fn, B=2000, seed=1):
    """거래일 묶음 부트스트랩. dfs: day 열이 있는 DataFrame 목록, fn(list_of_weighted)->스칼라.
    일별 합·개수를 만들어 가중치로 다시 뽑는다."""
    rng = np.random.default_rng(seed)
    days = np.unique(np.concatenate([d.day.to_numpy() for d in dfs]))
    point = fn(dfs, None, days)
    draws = []
    for _ in range(B):
        w = np.bincount(rng.integers(0, len(days), len(days)), minlength=len(days)).astype(float)
        draws.append(fn(dfs, w, days))
    draws = np.array(draws)
    p_gt0 = (1 + np.sum(~(draws > 0))) / (B + 1)
    return point, float(p_gt0), float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))


def wmean(df, col, w, days):
    s = df[col].fillna(0).groupby(df.day).sum().reindex(days, fill_value=0).to_numpy()
    c = df[col].notna().groupby(df.day).sum().reindex(days, fill_value=0).to_numpy()
    if w is None:
        w = np.ones(len(days))
    den = (c * w).sum()
    return (s * w).sum() / den if den > 0 else np.nan


def cluster_t(df, col):
    """거래일 묶음 표준오차 t값(평균의)."""
    x = df[[col, "day"]].dropna()
    n = len(x)
    m = x[col].mean()
    e = (x[col] - m).groupby(x.day).sum()
    G_ = len(e)
    se = np.sqrt((e ** 2).sum() * G_ / (G_ - 1)) / n
    return float(m), float(se), float(m / se), int(n), int(G_)
