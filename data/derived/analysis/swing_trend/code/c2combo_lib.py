"""c2combo(BRIEF2 시험 2) 공통 계산: 사건표, P1 3분위 칸, 날짜 묶음 귀무, 날짜 묶음 부트스트랩.

3분위 칸: 같은 type의 P1 사건으로 1/3·2/3 분위 q1·q2를 정하고 모든 구간에 그대로 쓴다.
  q1 < q2면 칸 0 = x ≤ q1, 칸 1 = q1 < x ≤ q2, 칸 2 = x > q2.
  q1 = q2(같은 값이 1/3 넘게 몰린 이진 단서 M3a·S6)면 칸 0 = x < q1, 칸 1 = x = q1, 칸 2 = x > q1.

귀무 1(주, 날짜 원형 이동):
  같은 구간·같은 type 사건의 고유 날짜 d_0..d_{D-1}(시간순)을 k칸 원형으로 민다.
  날짜 d_i의 사건은 날짜 d_{(i+k) mod D}의 단서를 받는다. 그 날짜에 같은 종목 사건이 있으면 그 사건의 단서를,
  없으면 그 날짜 사건 하나를 무작위로 받는다. 결과 y와 사건의 날짜 배치는 그대로라
  같은 날 결과의 공통 성분, 단서의 날짜 내 군집과 시계열 자기상관은 남고 단서↔결과 연결만 끊긴다.
  k는 15..D−15 전부를 돌고(칸마다 무작위 추출 seed를 바꿔) 200회를 채운다. 시장 단서(M*)는 날짜마다 같아서
  시장 단서만 쓰는 규칙의 귀무 값은 k 개수(P1 약 36~42개)만큼만 서로 다르다.
귀무 1b(민감도, 전 범위 이동 k = 1..D−1): 15칸 미만 이동을 빼면 실제 배열 근처가 빠져서, 한 덩어리로 이어지는
  단서(M3a 등)와 자기상관 있는 결과에서는 귀무 평균이 0보다 낮아진다(실측: 매도 1위 규칙 귀무 평균 −4.5%p).
  그래서 전 범위 이동도 같이 낸다(독립·정상 가정 아래 모든 이동은 교환 가능).
귀무 2(보조, 날짜 블록 순열): 같은 방식이되 날짜 대응을 무작위 순열로 바꾼다(시계열 자기상관은 끊긴다).
부트스트랩: 같은 구간·type 안에서 날짜를 복원 추출하고 그 날짜의 사건을 통째로 넣는다.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent.parent
RES = HERE / "results"
FEATURES = ["M1", "M2", "M3a", "M3b", "M4a", "M4b", "M4c", "M5",
            "S1", "S2", "S3a", "S3b", "S4", "S5", "S6", "S7", "S8", "S9"]
NAME = {
    "M1": "같은 방향 동시 확정 비율(3일)", "M2": "20일선 위 종목 비율", "M3a": "SOXX 50일선 위",
    "M3b": "SOXX 20일 수익", "M4a": "VIX", "M4b": "VIX 5일 변화", "M4c": "VIX/VIX3M",
    "M5": "최근 40일 추세−뒤집기 손익", "S1": "극점→확정 거래일", "S2": "직전 반대 구간 크기(ATR배)",
    "S3a": "확정일 거래량비", "S3b": "극점~확정 최대 거래량비", "S4": "확정일 갭(ATR배, 새 방향+)",
    "S5": "SOXX 대비 20일 상대수익", "S6": "극점~확정 사이 실적", "S7": "60일 자기상관 합",
    "S8": "52주 고점 대비", "S9": "ATR% 1년 백분위",
}
PERIOD_ORDER = ["P1", "P2", "Y25", "Y24", "Y23"]
KMIN = 15
NREP = 200


def load(det: str) -> pd.DataFrame:
    ev = pd.read_parquet(RES / f"events_{det}.parquet")
    ev = ev.sort_values(["date", "sym", "type"]).reset_index(drop=True)
    ev["date"] = pd.to_datetime(ev.date)
    # 단순 수익 표시용: 매수 exp(y)−1, 매도(피한 손실) 1−exp(−y)
    ev["y1s"] = np.where(ev.type == 1, np.expm1(ev.y1), -np.expm1(-ev.y1))
    du = np.sort(ev.date.unique())
    ev["di"] = np.searchsorted(du, ev.date.values)  # 전역 날짜 번호
    return ev


def edges(ev: pd.DataFrame, t: int) -> np.ndarray:
    p1 = ev[(ev.period == "P1") & (ev.type == t)]
    return np.array([np.quantile(p1[f].values, [1 / 3, 2 / 3]) for f in FEATURES])


def binize(X: np.ndarray, e: np.ndarray) -> np.ndarray:
    B = np.empty(X.shape, np.int64)
    for j in range(X.shape[1]):
        q1, q2 = e[j]
        x = X[:, j]
        if q1 < q2:
            B[:, j] = (x > q1).astype(np.int64) + (x > q2)
        else:
            B[:, j] = np.where(x < q1, 0, np.where(x == q1, 1, 2))
    return B


def bin_label(f: str, b: int, e: np.ndarray) -> str:
    q1, q2 = e[FEATURES.index(f)]
    fmt = (lambda v: f"{v:.3g}")
    if q1 < q2:
        txt = [f"≤{fmt(q1)}", f"{fmt(q1)}~{fmt(q2)}", f">{fmt(q2)}"][b]
        pos = ["아래", "가운데", "위"][b]
    else:  # 이진 단서(M3a·S6): 칸이 곧 값
        v = {0: q1 - 1, 1: q1, 2: q1 + 1}[b]
        return f"{f}({NAME[f]}) ={v:g}"
    return f"{f}({NAME[f]}) {pos} {txt}"


def null_src(ev: pd.DataFrame, rows: np.ndarray, mode: str, nrep: int = NREP, seed: int = 0, kmin: int = KMIN):
    """rows(한 구간·한 type 사건의 전역 행 번호)에 대해 귀무 단서 출처 행 번호 (nrep, len(rows))."""
    rng = np.random.default_rng(seed)
    sub = ev.loc[rows]
    du = np.sort(sub.date.unique())
    D = len(du)
    di = np.searchsorted(du, sub.date.values)
    syms = sub.sym.values
    by_date = [rows[di == i] for i in range(D)]
    at = {(i, s): r for i, s, r in zip(di, syms, rows)}
    if mode in ("shift", "shiftall"):
        ks = np.arange(kmin, D - kmin + 1) if mode == "shift" else np.arange(1, D)
        tg = [(np.arange(D) + ks[r % len(ks)]) % D for r in range(nrep)]
    else:
        tg = [rng.permutation(D) for _ in range(nrep)]
    out = np.empty((nrep, len(rows)), np.int64)
    for r in range(nrep):
        t = tg[r]
        for e in range(len(rows)):
            ti = t[di[e]]
            hit = at.get((ti, syms[e]))
            out[r, e] = hit if hit is not None else by_date[ti][rng.integers(len(by_date[ti]))]
    return out


def group_rows(ev: pd.DataFrame, period: str, t: int) -> np.ndarray:
    return np.flatnonzero(((ev.period == period) & (ev.type == t)).values)


def boot_counts(di: np.ndarray, nboot: int = 2000, seed: int = 1):
    """날짜 묶음 부트스트랩 가중치: 날짜별 복원 추출 횟수 (nboot, D)와 사건→날짜 번호."""
    rng = np.random.default_rng(seed)
    du, inv = np.unique(di, return_inverse=True)
    D = len(du)
    W = np.stack([np.bincount(rng.integers(D, size=D), minlength=D) for _ in range(nboot)])
    return W, inv


def boot_diff_ci(y: np.ndarray, sel: np.ndarray, di: np.ndarray, nboot: int = 2000, seed: int = 1):
    """고른 평균 − 안 고른 평균의 날짜 묶음 부트스트랩 95% 구간."""
    W, inv = boot_counts(di, nboot, seed)
    D = W.shape[1]
    ss = np.bincount(inv, weights=y * sel, minlength=D)
    ns = np.bincount(inv, weights=sel.astype(float), minlength=D)
    su = np.bincount(inv, weights=y * ~sel, minlength=D)
    nu = np.bincount(inv, weights=(~sel).astype(float), minlength=D)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = W @ ss / (W @ ns) - W @ su / (W @ nu)
    d = d[np.isfinite(d)]
    if len(d) < 0.5 * nboot:  # 고른 사건이 거의 없으면 구간을 내지 않는다
        return np.nan, np.nan
    return float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def boot_rows(di: np.ndarray, nboot: int = 1000, seed: int = 2):
    """사건 단위 부트스트랩 표본(날짜 통째로) 행 번호 목록."""
    rng = np.random.default_rng(seed)
    du, inv = np.unique(di, return_inverse=True)
    D = len(du)
    by = [np.flatnonzero(inv == i) for i in range(D)]
    return [np.concatenate([by[i] for i in rng.integers(D, size=D)]) for _ in range(nboot)]


def pq(real: float, null: np.ndarray):
    """귀무 95분위와 한쪽 p(귀무 ≥ 실제 비율, +1 보정)."""
    null = np.asarray(null, float)
    null = null[np.isfinite(null)]
    if len(null) == 0 or not np.isfinite(real):  # 확인 구간에서 고른 사건이 0개
        return np.nan, np.nan
    return float(np.quantile(null, 0.95)), float((1 + (null >= real).sum()) / (1 + len(null)))
