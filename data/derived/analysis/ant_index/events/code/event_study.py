"""부록 13 사건 연구: 분석 1~4와 조회·추천 누출 시험.

순서(부록 13 '지위와 순서'): 부록 12 등록 판정(10/24)을 먼저 하고, 그 뒤에만 포럼–가격을 잇는다.
그래서 실제 실행(`run`)은 (1) 2026-10-24 06:00 KST 전, (2) events.json에 10/23 종가가 들어와 끝 극값을 다시 판정하기 전
(`py events.py all`)에는 멈춘다. 지금 돌리는 것은 가짜 사건 시운전(`dryrun`)뿐이다(실제 events.json·가격을 읽지 않는다).

입력
  events/results/features_all.parquet (+ features_all_meta.json)   포럼 특징(가격 없음)
  events/results/events.json                                       사건 목록(가격으로만 정함)
  events/results/prices_close.csv                                  종가(위약 날짜 고르기·사건 전후 가격 그림)

창과 사건일 맞추기
  사건(극값) 거래일 e의 창 = D가 e인 창(events.json win_id). K창은 전 거래일 08:00 ~ e 08:00 KST, U창은 e 21:30 KST에
  닫혀 '사건일 장 전 정보'다. 오프셋 k = 그 창에서 k창 뒤(창 하나 = 그 시장 거래일 하나). k = +1 창에 사건일 장중·장 뒤 글이 든다.

특징 전처리(포럼만)
  - 판정 자격 밖 창(96시간 초과·장애)은 결측. realtime_comparable = False인 조회·추천 특징은 소급 창(실시간 글 5% 미만)만 쓴다.
  - 창 종류 보정: 주말·연휴 뒤 창(72~96시간)은 시간당 양·몫이 평일 창과 수준이 달라(시간당 글 수 중앙값 약 절반) 직전 20창 z가
    월요일마다 음수로 쏠린다. 그래서 평일 아닌 창 값에서 '직전 60창 안 평일 아닌 창 중앙값 − 평일 창 중앙값'(각 5창 이상)을
    뺀다. 평일 창은 그대로다. 모든 특징에 똑같이 적용한다(정의에 없어서 정한 것, 직전 창만 쓴다).
  - z = (x − 직전 20창 평균) ÷ 직전 20창 SD(ddof 1). 직전 20창 중 값 있는 창 15개 이상일 때만(자기 제외).
  - 극값 백분위(분석 3) = 직전 60창(값 40개 이상) 안 순위(중간 순위). 상위 10% = 0.9 이상, 하위 10% = 0.1 이하.

분석
  1. 사건 전후 그림: 오프셋 −10~+10의 z. 사건별 선·사건 평균·가짜 사건(묶음 대표) 평균, 위약 = 사건마다 가격 움직임이
     비슷한 날(직전 5·20거래일 수익률 표준화 거리 최근접 20일, 같은 종류 확정 꺾임점·끝 극값 ±3거래일 밖 — 하이닉스 8%처럼
     꺾임점이 촘촘하면 ±10으로는 후보가 남지 않는다). 위약 띠 = 위약 날들 z의 평균
     ± t(0.975, n−1) × SD ÷ √n(n = 그 오프셋에 값 있는 사건 수): '사건이 위약 날과 같다면 사건 n개 평균이 들어갈 95% 범위'.
     (시운전에서 '사건마다 후보 20일 중 하나씩 뽑은 평균'의 분위 띠는 후보끼리 서로 비슷해 너무 좁았다 — 무작위 날짜 사건의
     10~29%가 띠 밖. 그래서 표본 평균의 퍼짐으로 띠를 잡는다.) 오프셋 0·+1에서 사건 평균이 띠 밖인 특징 수를 센다(설명용).
  2. 가르기: 바닥 대 가짜 바닥, 고점 대 가짜 고점. 가짜는 같은 묶음(run)에서 가장 극단인 날(가짜 바닥 = 묶음 최저 종가,
     가짜 고점 = 묶음 최고 종가, 같으면 앞 날) 하나로 줄여 서로 독립인 표본으로 쓴다. 특징 × 오프셋(0, +1)마다 표본 하나를 빼고
     나머지로 문턱·방향(균형 정확도 최대)을 정해 빠진 표본을 맞힌다(leave-one-event-out). 점수 = 빠진 표본 예측의 균형 정확도(BA).
     귀무 = 사건·가짜 표지를 섞기 1,000번(모든 특징이 같은 섞기를 쓴다 → 특징 사이 상관 유지). 특징별 p = 귀무 BA가 관측 이상인
     비율(+1 보정). 우연 기대 개수 = 섞기마다 p < 0.05인 칸 수의 평균(95% 상한, 관측 이상일 확률도). 묶음 대표 대신 가짜 날 전부를
     쓴 BA(묶음째 빼기)도 보조로 적는다.
  3. 앞섬: 오프셋 −5~−1(값 3개 이상) 중 한 번이라도 극값(직전 60창 상위·하위 10%)에 닿은 비율. 사건 대 가짜(섞기 귀무, 같은
     섞기) 대 위약(위약 날들의 닿는 비율 p̂로 이항 분포: 사건 n개 중 k개 이상일 확률, 97.5% 분위).
  4. 지금 위치(설명용): 2026-10-15~10-23 창의 z가 사건·가짜·위약의 오프셋 0 z 분포에서 어디인지(백분위). '지금이 바닥'이라고
     단정하지 않는다.
  누출 시험(조회·추천): 오프셋 0·+1에서 (b) `vw_ln_b_median` z의 사건 중앙값이 가짜 중앙값보다 크면(어느 쪽이든), 또는 값이
     모자라 시험할 수 없으면 그 대비(자산 × 문턱 × 고점/바닥)의 조회·추천 특징 전부를 '누출 의심'으로 두고 후보에서 뺀다.
  후보 = 분석 2 p < 0.05 + 검수 지위 '통과' + 누출 의심 아님. 앞섬 후보 = 분석 3 p < 0.05 + 위약 97.5% 분위 위 + 같은 조건.
  8%·12% 결과를 모두 적는다.

실행
  py event_study.py dryrun   가짜 사건 시운전(무작위 날짜·주입 신호·가짜 가격) → results/dryrun.md, dryrun.json
  py event_study.py run      실제 실행(10/24 판정 뒤에만) → results/event_study.md 등
  py event_study.py run --confirmed-only   확정 사건만(부록 14, 2026-10-22 종가로 판정한 events.json). 날짜 가드 대신
                             끝 극값(미확정)과 그 ±3거래일 보류 가짜를 뺀다(분석·문턱·귀무 그대로) → results/confirmed_1022/
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
import math
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EV = HERE.parent
RESULTS = EV / "results"
KST = timezone(timedelta(hours=9))

OFFSETS = np.arange(-10, 11)
Z_N, Z_MIN = 20, 15
P_N, P_MIN, P_HI, P_LO = 60, 40, 0.9, 0.1
KIND_N, KIND_MIN = 60, 5
LEAD = (-5, -1)
LEAD_MIN = 3
CLS_OFFSETS = (0, 1)
N_NULL = 1000
PLACEBO_POOL = 20
PLACEBO_EXCL_TD = 3              # 가짜 사건 정의의 '확정 꺾임점 ±3거래일 밖'과 같게
ALPHA = 0.05
MIN_TRUE, MIN_FAKE = 4, 4
SEED = 20261024
LEAK_FEATURE = "vw_ln_b_median"
LEAK_OFFSETS = (0, 1)
LEAK_BUNDLE = "조회·추천"
NOW_FROM, NOW_TO = pd.Timestamp("2026-10-15"), pd.Timestamp("2026-10-23")
RUN_NOT_BEFORE = datetime(2026, 10, 24, 6, 0, tzinfo=KST)
CONFIRMED_ASOF = "2026-10-22"    # 확정 사건만 모드: 이 날 종가까지로 판정한 events.json(부록 14)
CONFIRMED_OUT = RESULTS / "confirmed_1022"
ASSET_MARKET = {"W_KR": "kr", "W_U": "us"}
ASSET_KO = {"KOSPI": "KOSPI", "SKHYNIX": "SK하이닉스", "SOXX": "SOXX"}
TYPE_KO = {"bottom": "바닥", "top": "고점"}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ================================================================ 1. 특징 전처리(포럼만)

def kind_adjust(x: np.ndarray, nonweekday: np.ndarray, n: int = KIND_N, min_n: int = KIND_MIN) -> np.ndarray:
    """평일 아닌 창 값 − (직전 n창 안 평일 아닌 창 중앙값 − 평일 창 중앙값). 각 min_n개 이상, 자기 제외. 평일 창은 그대로."""
    x = np.asarray(x, float)
    out = x.copy()
    for t in np.flatnonzero(nonweekday):
        lo = max(0, t - n)
        h, c = x[lo:t], nonweekday[lo:t]
        a, b = h[c & np.isfinite(h)], h[~c & np.isfinite(h)]
        out[t] = x[t] - (np.median(a) - np.median(b)) if (len(a) >= min_n and len(b) >= min_n) else np.nan
    return out


def z_prev(x: np.ndarray, n: int = Z_N, min_n: int = Z_MIN) -> np.ndarray:
    """직전 n개 창(행 위치) 중 값 있는 창이 min_n개 이상이면 (x − 평균) ÷ SD(ddof 1). 자기 제외."""
    x = np.asarray(x, float)
    z = np.full(len(x), np.nan)
    for t in range(len(x)):
        if not np.isfinite(x[t]):
            continue
        h = x[max(0, t - n):t]
        h = h[np.isfinite(h)]
        if len(h) < min_n:
            continue
        sd = h.std(ddof=1)
        if sd > 0:
            z[t] = (x[t] - h.mean()) / sd
    return z


def pct_prev(x: np.ndarray, n: int = P_N, min_n: int = P_MIN) -> np.ndarray:
    """직전 n개 창(값 min_n개 이상) 안에서 x의 중간 순위 백분위(0~1). 자기 제외."""
    x = np.asarray(x, float)
    p = np.full(len(x), np.nan)
    for t in range(len(x)):
        if not np.isfinite(x[t]):
            continue
        h = x[max(0, t - n):t]
        h = h[np.isfinite(h)]
        if len(h) < min_n:
            continue
        p[t] = ((h < x[t]).sum() + 0.5 * (h == x[t]).sum()) / len(h)
    return p


@dataclass
class Panel:
    market: str
    win_id: np.ndarray
    D: np.ndarray                      # datetime64[D]
    feats: list[str]
    X: np.ndarray                      # 원값(결측 처리 뒤)
    Z: np.ndarray                      # 창 종류 보정 + 직전 20창 z
    PCT: np.ndarray                    # 창 종류 보정 + 직전 60창 백분위
    meta: dict
    realtime: np.ndarray
    pos: dict = field(default_factory=dict)

    def __post_init__(self):
        self.pos = {w: i for i, w in enumerate(self.win_id)}


def load_features(path: Path = RESULTS / "features_all.parquet",
                  meta_path: Path = RESULTS / "features_all_meta.json") -> tuple[pd.DataFrame, dict]:
    F = pd.read_parquet(path)
    meta = json.loads(Path(meta_path).read_text("utf-8"))["features"]
    return F, meta


def grid_features(meta: dict) -> list[str]:
    return [f for f, m in meta.items() if m["event_study"]]


def build_panel(F: pd.DataFrame, meta: dict, market: str, feats: list[str] | None = None,
                X_override: np.ndarray | None = None) -> Panel:
    g = F[F["market"] == market].sort_values("close_kst").reset_index(drop=True)
    feats = feats or grid_features(meta)
    X = g[feats].to_numpy(float).copy() if X_override is None else np.array(X_override, float)
    ok = g["judge_ok"].to_numpy(bool)
    retro = g["retro_window"].to_numpy(bool)
    for j, f in enumerate(feats):
        if not meta[f]["realtime_comparable"]:
            X[~retro, j] = np.nan
    X[~ok, :] = np.nan
    nonwk = (g["kind_class"] == "nonweekday").to_numpy()
    Z = np.full_like(X, np.nan)
    PCT = np.full_like(X, np.nan)
    for j in range(X.shape[1]):
        xa = kind_adjust(X[:, j], nonwk)
        Z[:, j] = z_prev(xa)
        PCT[:, j] = pct_prev(xa)
    return Panel(market, g["win_id"].to_numpy(), pd.to_datetime(g["D"]).to_numpy("datetime64[D]"), feats, X, Z, PCT,
                 meta, (~retro) | g["realtime"].to_numpy(bool))


def with_column(panel: Panel, F: pd.DataFrame, feature: str, x_raw: np.ndarray) -> Panel:
    """한 특징의 원값만 바꾼 판(시운전 주입용). build_panel과 같은 전처리를 그 열에만 다시 한다."""
    g = F[F["market"] == panel.market].sort_values("close_kst").reset_index(drop=True)
    j = panel.feats.index(feature)
    x = np.array(x_raw, float)
    if not panel.meta[feature]["realtime_comparable"]:
        x[~g["retro_window"].to_numpy(bool)] = np.nan
    x[~g["judge_ok"].to_numpy(bool)] = np.nan
    xa = kind_adjust(x, (g["kind_class"] == "nonweekday").to_numpy())
    X, Z, PCT = panel.X.copy(), panel.Z.copy(), panel.PCT.copy()
    X[:, j], Z[:, j], PCT[:, j] = x, z_prev(xa), pct_prev(xa)
    return Panel(panel.market, panel.win_id, panel.D, panel.feats, X, Z, PCT, panel.meta, panel.realtime)


def gather(A: np.ndarray, base: np.ndarray, offsets) -> np.ndarray:
    """A (n_w × n_f), base (n,) 창 위치(−1 = 창 없음) → (n × len(offsets) × n_f), 범위 밖·창 없음은 결측."""
    offsets = np.asarray(offsets)
    base = np.asarray(base, int)
    idx = base[:, None] + offsets[None, :]
    ok = (base[:, None] >= 0) & (idx >= 0) & (idx < A.shape[0])
    out = np.full((len(base), len(offsets), A.shape[1]), np.nan)
    out[ok] = A[idx[ok]]
    return out


# ================================================================ 2. 사건 목록

@dataclass
class Contrast:
    asset: str
    market: str
    threshold: str
    kind: str                           # bottom | top
    events: pd.DataFrame                # date, win_id
    fakes_all: pd.DataFrame             # date, win_id, run, close
    fake_reps: pd.DataFrame             # 묶음 대표
    pivot_dates: list                   # 위약에서 뺄 같은 종류 확정 꺾임점(기간 앞 포함)·같은 종류 끝 극값
    info: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"{self.asset}_{self.threshold}_{self.kind}"


def fake_representatives(fakes: pd.DataFrame, kind: str) -> pd.DataFrame:
    """묶음(run)마다 가장 극단인 날 하나: 가짜 바닥 = 최저 종가, 가짜 고점 = 최고 종가(같으면 앞 날)."""
    if fakes.empty:
        return fakes.copy()
    f = fakes.sort_values("date").copy()
    f["_k"] = f["close"] if kind == "bottom" else -f["close"]
    rep = f.sort_values(["run", "_k", "date"], kind="mergesort").groupby("run", sort=False).head(1)
    return rep.drop(columns="_k").sort_values("date").reset_index(drop=True)


def load_contrasts(ev: dict) -> list[Contrast]:
    out = []
    for asset, a in ev["assets"].items():
        market = ASSET_MARKET[a["window"]]
        for X, t in a["thresholds"].items():
            piv = pd.DataFrame(t["pivots"]) if t["pivots"] else pd.DataFrame(columns=["kind", "date", "win_id"])
            warm = t.get("warmup_pivots", [])
            tail = t.get("tail_unconfirmed") or {}
            for kind, pk, fk in (("bottom", "low", "fake_bottoms"), ("top", "high", "fake_tops")):
                pdates = sorted(set(piv.loc[piv["kind"] == pk, "date"].tolist() +
                                    [w["date"] for w in warm if w.get("kind", pk) == pk] +
                                    ([tail["date"]] if tail.get("date") and tail.get("kind", pk) == pk else [])))
                evs = piv[piv["kind"] == pk][["date", "win_id"]].reset_index(drop=True)
                fk_ = pd.DataFrame(t.get(fk, []), columns=["date", "win_id", "run", "close"])
                fk_ = fk_[["date", "win_id", "run", "close"]].reset_index(drop=True)
                c = Contrast(asset, market, str(X), kind, evs, fk_, fake_representatives(fk_, kind), pdates)
                c.info = {"n_events": int(len(evs)), "n_fakes_all": int(len(fk_)), "n_fake_runs": int(fk_["run"].nunique())
                          if len(fk_) else 0, "tail_unconfirmed": tail.get("date"),
                          "held_near_tail": [h["date"] for h in t.get("held_near_tail", {}).get(fk, [])]}
                out.append(c)
    return out


def positions(panel: Panel, win_ids) -> np.ndarray:
    return np.array([panel.pos.get(w, -1) if isinstance(w, str) else -1 for w in win_ids], int)


# ================================================================ 3. 가르기(LOEO 문턱 분류기)

def loeo_ba(v: np.ndarray, Y: np.ndarray, groups: np.ndarray | None = None) -> np.ndarray:
    """v (n,) 값, Y (S, n) 표지(True = 사건). 그룹 하나씩 빼고 나머지로 문턱·방향을 정해(균형 정확도 최대) 빠진 그룹을 맞힌다.
    문턱 = 이웃한 서로 다른 값의 가운데. 같은 점수면 '위쪽이 사건' 방향, 더 낮은 문턱을 고른다(모두 사건으로 보는 자르기 포함).
    반환: 빠진 표본 예측의 균형 정확도 (S,)."""
    v = np.asarray(v, float)
    Y = np.atleast_2d(np.asarray(Y, bool))
    n = len(v)
    S = Y.shape[0]
    order = np.argsort(v, kind="mergesort")
    v = v[order]
    Y = Y[:, order]
    groups = np.arange(n) if groups is None else np.asarray(groups)[order]
    pred = np.zeros((S, n), bool)
    for gval in np.unique(groups):
        test = groups == gval
        vt, Yt = v[~test], Y[:, ~test]
        m = len(vt)
        P = Yt.sum(1).astype(float)[:, None]
        N = m - P
        cp = np.concatenate([np.zeros((S, 1)), np.cumsum(Yt, 1)], 1)
        cn = np.arange(m + 1)[None, :] - cp
        valid = np.ones(m + 1, bool)
        valid[1:m] = vt[1:] > vt[:-1]
        with np.errstate(invalid="ignore", divide="ignore"):
            up = 0.5 * ((P - cp) / P + cn / N)
            dn = 0.5 * (cp / P + (N - cn) / N)
        up = np.where(valid[None, :] & np.isfinite(up), up, -np.inf)
        dn = np.where(valid[None, :] & np.isfinite(dn), dn, -np.inf)
        best = np.round(np.concatenate([up, dn], 1), 12).argmax(1)   # 끝자리 차이로 같은 점수 순서가 바뀌지 않게
        is_up = best <= m
        j = np.where(is_up, best, best - (m + 1))
        lo = np.where(j > 0, vt[np.clip(j - 1, 0, max(m - 1, 0))], -np.inf)
        hi = np.where(j < m, vt[np.clip(j, 0, max(m - 1, 0))], np.inf)
        thr = np.where(j == 0, -np.inf, np.where(j == m, np.inf, (lo + hi) / 2))
        upper = v[test][None, :] > thr[:, None]
        pred[:, test] = np.where(is_up[:, None], upper, ~upper)
    P = Y.sum(1)
    N = n - P
    tp = (pred & Y).sum(1)
    tn = (~pred & ~Y).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((P > 0) & (N > 0), 0.5 * (tp / P + tn / N), np.nan)


def loeo_ba_naive(v: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None) -> float:
    """loeo_ba의 느린 판(시험용): 같은 규칙을 반복문으로."""
    v = np.asarray(v, float)
    y = np.asarray(y, bool)
    n = len(v)
    groups = np.arange(n) if groups is None else np.asarray(groups)
    pred = np.zeros(n, bool)
    for gval in np.unique(groups):
        te = groups == gval
        vt, yt = v[~te], y[~te]
        s = np.sort(vt)
        cuts = [-np.inf] + [(s[i - 1] + s[i]) / 2 for i in range(1, len(s)) if s[i] > s[i - 1]] + [np.inf]
        best, choice = -np.inf, None
        for direction in ("up", "dn"):
            for c in cuts:
                pr = vt > c if direction == "up" else ~(vt > c)
                P, N = yt.sum(), (~yt).sum()
                ba = 0.5 * ((pr & yt).sum() / P + (~pr & ~yt).sum() / N)
                if ba > best + 1e-12:
                    best, choice = ba, (direction, c)
        d, c = choice
        pr = v[te] > c
        pred[te] = pr if d == "up" else ~pr
    P, N = y.sum(), (~y).sum()
    return float(0.5 * ((pred & y).sum() / P + (~pred & ~y).sum() / N))


def perm_labels(y: np.ndarray, n_perm: int, rng: np.random.Generator) -> np.ndarray:
    """(n_perm, n) 표지 섞기. 모든 특징이 같은 섞기를 쓴다."""
    return np.stack([rng.permutation(y) for _ in range(n_perm)])


def p_upper(obs: float, null: np.ndarray) -> float:
    null = null[np.isfinite(null)]
    if not np.isfinite(obs) or len(null) == 0:
        return np.nan
    return float((1 + (null >= obs - 1e-12).sum()) / (1 + len(null)))


def null_p_matrix(T: np.ndarray) -> np.ndarray:
    """T (S, k) 귀무 통계 → 귀무 표본 각각의 p(나머지 귀무와 견준 위쪽 비율, 자기 포함 = count_ge / S)."""
    S, k = T.shape
    out = np.full((S, k), np.nan)
    for j in range(k):
        t = T[:, j]
        f = np.isfinite(t)
        if f.sum() == 0:
            continue
        s = np.sort(t[f])
        ge = len(s) - np.searchsorted(s, t[f] - 1e-12, side="left")
        out[f, j] = ge / len(s)
    return out


def family_count(p_obs: np.ndarray, T_null: np.ndarray, alpha: float = ALPHA) -> dict:
    """칸 수, 관측 p < α 개수, 섞기마다 p < α 개수의 평균(우연 기대)·95% 상한·관측 이상일 확률."""
    keep = np.isfinite(p_obs)
    if keep.sum() == 0:
        return {"n_tests": 0, "observed": 0, "expected_by_chance": None, "null_q95": None, "p_family": None}
    pn = null_p_matrix(T_null[:, keep])
    cnt = (pn < alpha).sum(1)
    obs = int((p_obs[keep] < alpha).sum())
    return {"n_tests": int(keep.sum()), "observed": obs, "expected_by_chance": round(float(cnt.mean()), 3),
            "null_q95": float(np.quantile(cnt, 0.95)), "p_family": float((1 + (cnt >= obs).sum()) / (1 + len(cnt)))}


def classify(panel: Panel, ev_pos: np.ndarray, fk_pos: np.ndarray, offsets=CLS_OFFSETS, n_null: int = N_NULL,
             seed: int = SEED, fk_all_pos: np.ndarray | None = None, fk_all_groups: np.ndarray | None = None) -> dict:
    """분석 2. 반환: 칸(특징 × 오프셋) 표, 관측 BA·p, 귀무 BA 행렬."""
    base = np.concatenate([ev_pos, fk_pos])
    y = np.r_[np.ones(len(ev_pos), bool), np.zeros(len(fk_pos), bool)]
    have = base >= 0
    base, y = base[have], y[have]
    rng = np.random.default_rng(seed)
    Yp = perm_labels(y, n_null, rng)
    V = gather(panel.Z, base, offsets)                      # (n, k, f)
    rows, null_cols, p_obs = [], [], []
    for oi, k in enumerate(offsets):
        for fi, f in enumerate(panel.feats):
            v = V[:, oi, fi]
            ok = np.isfinite(v)
            nt, nf = int((ok & y).sum()), int((ok & ~y).sum())
            r = {"feature": f, "offset": int(k), "n_events": nt, "n_fakes": nf}
            if nt < MIN_TRUE or nf < MIN_FAKE:
                r.update({"status": "표본 부족", "ba": np.nan, "p": np.nan})
                rows.append(r)
                null_cols.append(np.full(n_null, np.nan))
                p_obs.append(np.nan)
                continue
            ba = float(loeo_ba(v[ok], y[ok][None, :])[0])
            nb = loeo_ba(v[ok], Yp[:, ok])
            p = p_upper(ba, nb)
            mt, mf = float(np.mean(v[ok & y])), float(np.mean(v[ok & ~y]))
            ranks = pd.Series(v[ok]).rank().to_numpy()
            auc = (ranks[y[ok]].sum() - nt * (nt + 1) / 2) / (nt * nf)
            r.update({"status": "시험", "ba": ba, "p": p, "mean_z_events": mt, "mean_z_fakes": mf,
                      "direction": "사건 쪽이 높음" if mt > mf else "사건 쪽이 낮음", "auc_in_sample": float(auc),
                      "null_ba_q95": float(np.nanquantile(nb, 0.95))})
            if fk_all_pos is not None and len(fk_all_pos):
                b2 = np.concatenate([ev_pos, fk_all_pos])
                g2 = np.concatenate([np.arange(len(ev_pos)), 10_000 + np.asarray(fk_all_groups)])
                y2 = np.r_[np.ones(len(ev_pos), bool), np.zeros(len(fk_all_pos), bool)]
                h2 = b2 >= 0
                v2 = gather(panel.Z, b2[h2], [k])[:, 0, fi]
                ok2 = np.isfinite(v2)
                y2, g2 = y2[h2][ok2], g2[h2][ok2]
                if y2.sum() >= MIN_TRUE and (~y2).sum() >= MIN_FAKE and len(np.unique(g2[~y2])) >= 2:
                    r["ba_all_fake_days_run_groups"] = float(loeo_ba(v2[ok2], y2[None, :], g2)[0])
            rows.append(r)
            null_cols.append(nb)
            p_obs.append(p)
    return {"table": pd.DataFrame(rows), "p_obs": np.array(p_obs, float), "T_null": np.stack(null_cols, 1),
            "n_samples": {"events": int(y.sum()), "fakes": int((~y).sum())}, "perm": Yp, "y": y, "base": base}


# ================================================================ 4. 위약(가격 움직임이 비슷한 날)

def price_series(prices: pd.DataFrame, asset: str) -> pd.Series:
    p = prices[prices["asset"] == asset].dropna(subset=["close"])
    s = pd.Series(p["close"].to_numpy(float), index=pd.to_datetime(p["date"]).to_numpy("datetime64[D]"))
    return s.sort_index()


def placebo_pools(panel: Panel, close: pd.Series, ev_dates, pivot_dates, k_pool: int = PLACEBO_POOL,
                  excl_td: int = PLACEBO_EXCL_TD) -> tuple[list[np.ndarray], dict]:
    """사건마다 위약 후보(창 위치) k_pool개: 직전 5·20거래일 수익률(후보 전체 SD로 표준화)의 유클리드 거리 최근접.
    후보 = 창이 있고 가격 20거래일 이력이 있는 날, pivot_dates(같은 종류 확정 꺾임점·끝 극값) ±excl_td 거래일 밖, 사건일이 아닌 날."""
    d = close.index.to_numpy("datetime64[D]")
    c = close.to_numpy(float)
    pos_of = {x: i for i, x in enumerate(d)}
    r5 = np.full(len(c), np.nan)
    r20 = np.full(len(c), np.nan)
    r5[5:] = c[5:] / c[:-5] - 1
    r20[20:] = c[20:] / c[:-20] - 1
    bad = np.zeros(len(c), bool)
    for pdt in pivot_dates:
        i = pos_of.get(np.datetime64(pdt, "D"))
        if i is not None:
            bad[max(0, i - excl_td):i + excl_td + 1] = True
    evd = {np.datetime64(x, "D") for x in ev_dates}
    cand_px, cand_win = [], []
    for wi, D in enumerate(panel.D):
        i = pos_of.get(D)
        if i is None or not np.isfinite(r20[i]) or bad[i] or D in evd:
            continue
        cand_px.append(i)
        cand_win.append(wi)
    cand_px, cand_win = np.array(cand_px, int), np.array(cand_win, int)
    s5, s20 = np.nanstd(r5[cand_px]), np.nanstd(r20[cand_px])
    pools = []
    for e in ev_dates:
        i = pos_of.get(np.datetime64(e, "D"))
        if i is None or not np.isfinite(r20[i]) or len(cand_px) == 0:
            pools.append(np.array([], int))
            continue
        dist = np.hypot((r5[cand_px] - r5[i]) / s5, (r20[cand_px] - r20[i]) / s20)
        o = np.argsort(dist, kind="mergesort")[:k_pool]
        pools.append(cand_win[o])
    return pools, {"n_candidates": int(len(cand_px)), "pool_size": k_pool, "excl_td": excl_td}


# ================================================================ 5. 분석 1·3·4, 누출 시험

def pool_positions(pools: list[np.ndarray] | None) -> np.ndarray:
    """사건마다 후보 20일을 이어 붙인 창 위치(사건마다 같은 무게, 겹치는 날은 겹친 만큼)."""
    if not pools or not any(len(x) for x in pools):
        return np.array([], int)
    return np.concatenate([np.asarray(x, int) for x in pools if len(x)])


def profiles(panel: Panel, ev_pos, fk_pos, pools: list[np.ndarray] | None) -> dict:
    """분석 1: 사건·가짜 평균 z(오프셋 × 특징). 위약 띠 = 위약 날 z 평균 ± t × SD ÷ √(사건 수)."""
    E = gather(panel.Z, ev_pos, OFFSETS)
    Fk = gather(panel.Z, fk_pos, OFFSETS)
    pp = pool_positions(pools)
    PL = gather(panel.Z, pp, OFFSETS) if len(pp) else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = {"events": E, "ev_mean": np.nanmean(E, 0) if len(E) else None,
               "ev_n": np.isfinite(E).sum(0), "fk_mean": np.nanmean(Fk, 0) if len(Fk) else None}
        if PL is not None:
            from scipy.stats import t as t_dist
            mu, sd = np.nanmean(PL, 0), np.nanstd(PL, 0, ddof=1)
            n = out["ev_n"].astype(float)
            tq = np.where(n >= 2, t_dist.ppf(0.975, np.maximum(n - 1, 1)), np.nan)
            half = tq * sd / np.sqrt(np.where(n > 0, n, np.nan))
            out.update({"pl_mean": mu, "pl_lo": mu - half, "pl_hi": mu + half,
                        "pl_n_dates": np.isfinite(PL).sum(0)})
    return out


def lead_hits(panel: Panel, base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """분석 3: (n, f) 상위·하위 극값 닿음(오프셋 −5~−1, 값 3개 이상, 아니면 결측)."""
    P = gather(panel.PCT, base, np.arange(LEAD[0], LEAD[1] + 1))
    nv = np.isfinite(P).sum(1)
    hi = np.where(nv >= LEAD_MIN, (P >= P_HI).any(1), np.nan)
    lo = np.where(nv >= LEAD_MIN, (P <= P_LO).any(1), np.nan)
    return hi.astype(float), lo.astype(float)


def lead_analysis(panel: Panel, cls: dict, pools: list[np.ndarray] | None) -> dict:
    """사건 대 가짜(같은 섞기 귀무), 위약(위약 날 닿는 비율 p̂의 이항 분포). 칸 = 특징 × {상위, 하위}."""
    from scipy.stats import binom
    base, y, Yp = cls["base"], cls["y"], cls["perm"]
    H = dict(zip(("hi", "lo"), lead_hits(panel, base)))
    PLH = None
    pp = pool_positions(pools)
    if len(pp):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            PLH = {k: np.nanmean(v, 0) for k, v in zip(("hi", "lo"), lead_hits(panel, pp))}
    rows, null_cols, p_obs = [], [], []
    for side in ("hi", "lo"):
        h = H[side]
        for fi, f in enumerate(panel.feats):
            x = h[:, fi]
            ok = np.isfinite(x)
            nt, nf = int((ok & y).sum()), int((ok & ~y).sum())
            r = {"feature": f, "side": "상위 10%" if side == "hi" else "하위 10%", "n_events": nt, "n_fakes": nf}
            if nt < MIN_TRUE or nf < MIN_FAKE:
                r.update({"status": "표본 부족", "p": np.nan})
                rows.append(r)
                null_cols.append(np.full(Yp.shape[0], np.nan))
                p_obs.append(np.nan)
                continue
            rt, rf = float(x[ok & y].mean()), float(x[ok & ~y].mean())
            Yk = Yp[:, ok]
            xx = x[ok]
            with np.errstate(invalid="ignore", divide="ignore"):
                nt_ = (Yk * xx).sum(1) / Yk.sum(1) - ((~Yk) * xx).sum(1) / (~Yk).sum(1)
            p = p_upper(rt - rf, nt_)
            r.update({"status": "시험", "rate_events": rt, "rate_fakes": rf, "diff": rt - rf, "p": p})
            if PLH is not None and np.isfinite(PLH[side][fi]):
                ph = float(PLH[side][fi])
                k_ev = int(round(x[ok & y].sum()))
                r.update({"rate_placebo_mean": ph, "rate_placebo_q975": float(binom.ppf(0.975, nt, ph) / nt),
                          "p_vs_placebo": float(binom.sf(k_ev - 1, nt, ph))})
            rows.append(r)
            null_cols.append(nt_)
            p_obs.append(p)
    return {"table": pd.DataFrame(rows), "p_obs": np.array(p_obs, float), "T_null": np.stack(null_cols, 1)}


def leak_test(panel: Panel, ev_pos, fk_pos) -> dict:
    """사건일 (b)가 가짜 사건일보다 크면(오프셋 0 또는 +1, z 중앙값) 누출 의심. 시험할 수 없으면 의심으로 둔다."""
    if LEAK_FEATURE not in panel.feats:
        return {"flag": True, "reason": f"{LEAK_FEATURE} 없음 → 의심으로 둠", "by_offset": {}}
    fi = panel.feats.index(LEAK_FEATURE)
    E = gather(panel.Z, ev_pos, LEAK_OFFSETS)[:, :, fi]
    Fk = gather(panel.Z, fk_pos, LEAK_OFFSETS)[:, :, fi]
    by, flag, testable = {}, False, True
    for oi, k in enumerate(LEAK_OFFSETS):
        a, b = E[:, oi], Fk[:, oi]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) < 3 or len(b) < 3:
            by[int(k)] = {"n_events": int(len(a)), "n_fakes": int(len(b)), "result": "값 부족"}
            testable = False
            continue
        ma, mb = float(np.median(a)), float(np.median(b))
        try:
            from scipy.stats import mannwhitneyu
            pmw = float(mannwhitneyu(a, b, alternative="greater").pvalue)
        except Exception:      # noqa: BLE001  (scipy가 없으면 p만 비운다)
            pmw = np.nan
        big = ma > mb
        flag |= big
        by[int(k)] = {"n_events": int(len(a)), "n_fakes": int(len(b)), "median_events": ma, "median_fakes": mb,
                      "events_greater": bool(big), "mannwhitney_p_greater": pmw}
    if not testable:
        flag = True
    return {"flag": bool(flag), "testable": testable, "by_offset": by,
            "rule": "오프셋 0·+1 중 하나라도 사건 중앙값 > 가짜 중앙값이면(또는 값 부족이면) 조회·추천 묶음 전체를 누출 의심"}


def now_position(panel: Panel, ev_pos, fk_pos, pools: list[np.ndarray] | None) -> pd.DataFrame:
    """분석 4(설명용): 10/15~10/23 창의 z가 사건·가짜·위약 후보의 오프셋 0 z 분포에서 차지하는 백분위."""
    sel = np.flatnonzero((panel.D >= np.datetime64(NOW_FROM.date(), "D")) & (panel.D <= np.datetime64(NOW_TO.date(), "D")))
    E = gather(panel.Z, ev_pos, [0])[:, 0, :]
    Fk = gather(panel.Z, fk_pos, [0])[:, 0, :]
    PL = None
    if pools is not None:
        allp = np.unique(np.concatenate([p for p in pools if len(p)])) if any(len(p) for p in pools) else np.array([], int)
        PL = gather(panel.Z, allp, [0])[:, 0, :] if len(allp) else None
    rows = []
    for t in sel:
        for fi, f in enumerate(panel.feats):
            z = panel.Z[t, fi]

            def pr(ref):
                ref = ref[np.isfinite(ref)]
                return float(((ref < z).sum() + 0.5 * (ref == z).sum()) / len(ref)) if (len(ref) and np.isfinite(z)) else np.nan

            rows.append({"win_id": panel.win_id[t], "D": str(panel.D[t]), "feature": f, "z": z, "pct60": panel.PCT[t, fi],
                         "pct_in_events": pr(E[:, fi]), "pct_in_fakes": pr(Fk[:, fi]),
                         "pct_in_placebo": pr(PL[:, fi]) if PL is not None else np.nan,
                         "n_events": int(np.isfinite(E[:, fi]).sum()), "n_fakes": int(np.isfinite(Fk[:, fi]).sum()),
                         "realtime_window": bool(panel.realtime[t]),
                         "realtime_caution": (not panel.meta[f]["realtime_comparable"]) or
                                             ("실시간" in panel.meta[f].get("note", ""))})
    return pd.DataFrame(rows)


def price_paths(close: pd.Series, dates) -> np.ndarray:
    """(n, 21) 사건일 종가 대비 오프셋 종가 − 1(가격 달력 기준)."""
    d = close.index.to_numpy("datetime64[D]")
    c = close.to_numpy(float)
    pos_of = {x: i for i, x in enumerate(d)}
    out = np.full((len(dates), len(OFFSETS)), np.nan)
    for r, e in enumerate(dates):
        i = pos_of.get(np.datetime64(e, "D"))
        if i is None:
            continue
        for j, k in enumerate(OFFSETS):
            if 0 <= i + k < len(c):
                out[r, j] = c[i + k] / c[i] - 1
    return out


# ================================================================ 6. 한 대비 실행

def run_contrast(c: Contrast, panel: Panel, close: pd.Series | None, n_null: int = N_NULL,
                 seed: int = SEED, with_now: bool = True) -> dict:
    ev_pos = positions(panel, c.events["win_id"])
    fk_pos = positions(panel, c.fake_reps["win_id"])
    fa_pos = positions(panel, c.fakes_all["win_id"])
    s = int(hashlib.sha256(f"{seed}:{c.name}".encode()).hexdigest()[:8], 16)
    cls = classify(panel, ev_pos, fk_pos, n_null=n_null, seed=s, fk_all_pos=fa_pos,
                   fk_all_groups=c.fakes_all["run"].to_numpy() if len(c.fakes_all) else None)
    pools, pinfo, ppaths = None, {}, {}
    if close is not None:
        pools, pinfo = placebo_pools(panel, close, c.events["date"], c.pivot_dates)
        ppaths = {"events": price_paths(close, c.events["date"]), "fakes": price_paths(close, c.fake_reps["date"])}
        pdates = [str(panel.D[i]) for p in pools for i in p]
        ppaths["placebo"] = price_paths(close, pdates)
    prof = profiles(panel, ev_pos, fk_pos, pools)
    lead = lead_analysis(panel, cls, pools)
    leak = leak_test(panel, ev_pos, fk_pos)
    tb = cls["table"]
    tb["status_feature"] = tb["feature"].map(lambda f: panel.meta[f]["status"])
    tb["bundle"] = tb["feature"].map(lambda f: panel.meta[f]["bundle"])
    tb["leak_suspect"] = tb["bundle"].eq(LEAK_BUNDLE) & leak["flag"]
    tb["candidate"] = (tb["p"] < ALPHA) & tb["status_feature"].eq("통과") & ~tb["leak_suspect"]
    fam = {}
    for nm, mask in (("통과", tb["status_feature"].eq("통과").to_numpy()),
                     ("설명용", tb["status_feature"].eq("설명용").to_numpy()), ("전체", np.ones(len(tb), bool))):
        fam[nm] = family_count(np.where(mask, cls["p_obs"], np.nan), cls["T_null"])
    lt = lead["table"]
    lt["status_feature"] = lt["feature"].map(lambda f: panel.meta[f]["status"])
    lt["bundle"] = lt["feature"].map(lambda f: panel.meta[f]["bundle"])
    lt["leak_suspect"] = lt["bundle"].eq(LEAK_BUNDLE) & leak["flag"]
    above_pl = lt["rate_events"] > lt["rate_placebo_q975"] if "rate_placebo_q975" in lt else True
    lt["candidate"] = (lt["p"] < ALPHA) & lt["status_feature"].eq("통과") & ~lt["leak_suspect"] & above_pl
    lfam = {}
    for nm, mask in (("통과", lt["status_feature"].eq("통과").to_numpy()),
                     ("설명용", lt["status_feature"].eq("설명용").to_numpy()), ("전체", np.ones(len(lt), bool))):
        lfam[nm] = family_count(np.where(mask, lead["p_obs"], np.nan), lead["T_null"])
    outside = {}
    if "pl_lo" in prof and prof["ev_mean"] is not None:
        for k in (0, 1):
            j = int(np.flatnonzero(OFFSETS == k)[0])
            m = prof["ev_mean"][j]
            o = (m < prof["pl_lo"][j]) | (m > prof["pl_hi"][j])
            outside[str(k)] = {"n_features": int(np.isfinite(m).sum()), "outside_band": int(np.nansum(o)),
                               "features": [panel.feats[i] for i in np.flatnonzero(o)],
                               "expected_if_independent": round(0.05 * int(np.isfinite(m).sum()), 2)}
    now = now_position(panel, ev_pos, fk_pos, pools) if with_now else pd.DataFrame()
    return {"contrast": c, "cls": cls, "table": tb, "family": fam, "lead": lt, "lead_family": lfam, "leak": leak,
            "profiles": prof, "outside_band": outside, "placebo": pinfo, "pools": pools, "price_paths": ppaths,
            "now": now, "n": {"events_total": int(len(c.events)), "events_with_window": int((ev_pos >= 0).sum()),
                              "fake_reps": int(len(c.fake_reps)), "fake_reps_with_window": int((fk_pos >= 0).sum()),
                              "fakes_all": int(len(c.fakes_all))}}


# ================================================================ 7. 그림(SVG)

def _poly(xs, ys, x0, y0, sx, sy) -> str:
    pts, segs = [], []
    for x, y in zip(xs, ys):
        if np.isfinite(y):
            pts.append(f"{x0 + sx * x:.1f},{y0 - sy * y:.1f}")
        elif pts:
            segs.append(pts)
            pts = []
    if pts:
        segs.append(pts)
    return "".join(f'<polyline points="{" ".join(s)}"/>' for s in segs if len(s) > 1)


def svg_panel(title: str, lines: np.ndarray, mean, fk_mean, lo, hi, pl_mean, ylim=(-3.5, 3.5), w=250, h=150,
              unit="z") -> str:
    pad_l, pad_r, pad_t, pad_b = 30, 8, 18, 18
    xs = OFFSETS - OFFSETS[0]
    sx = (w - pad_l - pad_r) / (len(OFFSETS) - 1)
    y_min, y_max = ylim
    sy = (h - pad_t - pad_b) / (y_max - y_min)
    x0, y0 = pad_l, pad_t + (h - pad_t - pad_b) * (y_max / (y_max - y_min))

    def clip(a):
        return np.clip(np.asarray(a, float), y_min, y_max) if a is not None else None
    parts = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="{html.escape(title)}">',
             f'<text x="{pad_l}" y="12" class="t">{html.escape(title)}</text>',
             f'<line x1="{x0}" x2="{w - pad_r}" y1="{y0:.1f}" y2="{y0:.1f}" class="ax"/>',
             f'<line x1="{x0 + sx * 10:.1f}" x2="{x0 + sx * 10:.1f}" y1="{pad_t}" y2="{h - pad_b}" class="ev"/>']
    for v in (y_min, y_max):
        parts.append(f'<text x="{pad_l - 3}" y="{y0 - sy * v + 3:.1f}" class="lab" text-anchor="end">{v:g}</text>')
    parts.append(f'<text x="{x0:.1f}" y="{h - 4}" class="lab">−10</text>'
                 f'<text x="{x0 + sx * 10:.1f}" y="{h - 4}" class="lab" text-anchor="middle">0</text>'
                 f'<text x="{w - pad_r}" y="{h - 4}" class="lab" text-anchor="end">+10 {unit}</text>')
    if lo is not None and hi is not None:
        lo_, hi_ = clip(lo), clip(hi)
        ok = np.isfinite(lo_) & np.isfinite(hi_)
        if ok.sum() > 1:
            top = [f"{x0 + sx * x:.1f},{y0 - sy * v:.1f}" for x, v in zip(xs[ok], hi_[ok])]
            bot = [f"{x0 + sx * x:.1f},{y0 - sy * v:.1f}" for x, v in zip(xs[ok][::-1], lo_[ok][::-1])]
            parts.append(f'<polygon class="band" points="{" ".join(top + bot)}"/>')
    if pl_mean is not None:
        parts.append(f'<g class="pl">{_poly(xs, clip(pl_mean), x0, y0, sx, sy)}</g>')
    if lines is not None:
        parts.append('<g class="ln">' + "".join(_poly(xs, clip(l_), x0, y0, sx, sy) for l_ in lines) + "</g>")
    if fk_mean is not None:
        parts.append(f'<g class="fk">{_poly(xs, clip(fk_mean), x0, y0, sx, sy)}</g>')
    if mean is not None:
        parts.append(f'<g class="mn">{_poly(xs, clip(mean), x0, y0, sx, sy)}</g>')
    parts.append("</svg>")
    return "".join(parts)


FIG_CSS = """
:root{--bg:#fbfbf9;--fg:#1d1d1b;--mut:#6b6b66;--ax:#c9c9c2;--ln:rgba(40,90,170,.22);--mn:#1f5fbf;--fk:#c2571a;
--pl:#7a7a72;--band:rgba(122,122,114,.18);--ev:#d8d8d0}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecece6;--mut:#a3a39b;--ax:#45453f;--ln:rgba(120,170,255,.25);
--mn:#7fb0ff;--fk:#ff9a5c;--pl:#a3a39b;--band:rgba(163,163,155,.18);--ev:#3a3a35}}
body{background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif;margin:0;padding:16px}
h1{font-size:18px;margin:0 0 4px}p{margin:4px 0;color:var(--mut)}
.grid{display:flex;flex-wrap:wrap;gap:10px;margin-top:12px}
.grid svg{background:transparent;max-width:100%;height:auto}
svg .t{font-size:11px;fill:var(--fg)}svg .lab{font-size:9px;fill:var(--mut)}
svg .ax{stroke:var(--ax)}svg .ev{stroke:var(--ev);stroke-dasharray:2 2}
svg polyline{fill:none}svg .ln polyline{stroke:var(--ln);stroke-width:1}svg .mn polyline{stroke:var(--mn);stroke-width:2.2}
svg .fk polyline{stroke:var(--fk);stroke-width:1.6;stroke-dasharray:4 3}svg .pl polyline{stroke:var(--pl);stroke-width:1}
svg .band{fill:var(--band)}.leg span{margin-right:14px}.leg i{display:inline-block;width:18px;height:3px;vertical-align:middle;margin-right:4px}
"""


def figure_html(res: dict, panel: Panel, title_note: str = "") -> str:
    c, prof = res["contrast"], res["profiles"]
    head = (f"<h1>{html.escape(ASSET_KO.get(c.asset, c.asset))} {c.threshold}% {TYPE_KO[c.kind]}: 사건 전후 z</h1>"
            f"<p>{html.escape(title_note)}사건 {res['n']['events_with_window']}개(창 있음) · 가짜 묶음 대표 "
            f"{res['n']['fake_reps_with_window']}개 · 위약 = 사건마다 가격이 비슷한 날 {PLACEBO_POOL}일. "
            "오프셋 0 = 사건일 장 전 창, +1 = 사건일 장중·장 뒤가 든 창."
            "</p><p class='leg'><span><i style='background:var(--mn)'></i>사건 평균</span><span><i style='background:var(--ln)'></i>"
            "사건별</span><span><i style='background:var(--fk)'></i>가짜(묶음 대표) 평균</span><span><i style='background:var(--pl)'>"
            "</i>위약 평균(띠 = 사건 수만큼 평균했을 때 95% 범위)</span></p>")
    cells = []
    pp = res.get("price_paths") or {}
    if pp:
        with np.errstate(invalid="ignore"):
            em = np.nanmean(pp["events"], 0) * 100 if len(pp["events"]) else None
            fm = np.nanmean(pp["fakes"], 0) * 100 if len(pp["fakes"]) else None
            pm = np.nanmean(pp["placebo"], 0) * 100 if len(pp["placebo"]) else None
        lim = np.nanmax(np.abs(np.concatenate([x for x in (pp["events"] * 100, pp["fakes"] * 100) if x.size]))) \
            if (pp["events"].size or pp["fakes"].size) else 10
        lim = float(min(max(5.0, math.ceil(lim / 5) * 5), 40)) if np.isfinite(lim) else 10.0
        cells.append(svg_panel("가격(사건일 종가 대비 %)", pp["events"] * 100, em, fm, None, None, pm, ylim=(-lim, lim),
                               unit="%"))
    for fi, f in enumerate(panel.feats):
        m = panel.meta[f]
        tt = f"{m['ko'][:26]}" + (" [설명용]" if m["status"] == "설명용" else "")
        cells.append(svg_panel(tt, prof["events"][:, :, fi], prof["ev_mean"][:, fi] if prof["ev_mean"] is not None
                               else None, prof["fk_mean"][:, fi] if prof["fk_mean"] is not None else None,
                               prof.get("pl_lo", None)[:, fi] if "pl_lo" in prof else None,
                               prof.get("pl_hi", None)[:, fi] if "pl_hi" in prof else None,
                               prof.get("pl_mean", None)[:, fi] if "pl_mean" in prof else None))
    return (f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><meta name='viewport' "
            f"content='width=device-width,initial-scale=1'><title>사건 전후 {html.escape(c.name)}</title><style>{FIG_CSS}"
            f"</style></head><body>{head}<div class='grid'>{''.join(cells)}</div></body></html>")


# ================================================================ 8. 실제 실행(10/24 판정 뒤)

def guard_real_run(ev: dict, now: datetime | None = None) -> None:
    now = now or datetime.now(KST)
    if now < RUN_NOT_BEFORE:
        raise SystemExit(f"멈춤: {RUN_NOT_BEFORE.isoformat()} 전에는 포럼–가격을 잇지 않는다(부록 12 판정 먼저).")
    waiting = []
    for asset, a in ev["assets"].items():
        for X, t in a["thresholds"].items():
            if not (t.get("rejudge") or {}).get("deadline_close_in"):
                waiting.append(f"{asset} {X}%")
    if waiting:
        raise SystemExit("멈춤: 10/23 종가로 끝 극값을 다시 판정하지 않았다(`py events.py all` 먼저): " + ", ".join(waiting))


def confirmed_only_events(ev: dict, asof: str = CONFIRMED_ASOF) -> tuple[dict, dict]:
    """확정 사건만 모드(부록 14): guard_real_run(10/24 06:00 전·10/23 종가 재판정 전 멈춤) 대신 끝 극값을 뺀다.

    - 모든 자산의 자료 끝(data_through)이 asof여야 한다(그 날 종가로 판정한 events.json만 받는다).
    - 사건 = 확정 꺾임점(pivots)뿐이고, 확정일(confirm_date)이 asof 이하인지 확인한다. 끝 극값(tail_unconfirmed)은
      사건에 넣지 않는다(같은 날·같은 종류 꺾임점이 있으면 멈춤).
    - 끝 극값 ±3거래일 안 가짜 후보(held_near_tail)는 가짜에도 넣지 않는다(가짜 목록과 겹치면 멈춤).
    - 위약 후보에서 끝 극값 ±3거래일을 빼는 규칙(load_contrasts의 pivot_dates)은 그대로 둔다(분석 로직 불변).
    분석·문턱·귀무는 바꾸지 않는다. 반환: (ev 사본, 뺀 항목 기록)."""
    out = copy.deepcopy(ev)
    rec = {"mode": "confirmed_only", "asof": asof, "date_guard": "대신함(끝 극값·보류 가짜 제외)", "contrasts": {}}
    for asset, a in out["assets"].items():
        if a.get("data_through") != asof:
            raise SystemExit(f"멈춤(확정 사건만): {asset} 자료 끝 {a.get('data_through')} ≠ {asof} — `py events.py all` 먼저")
        for X, t in a["thresholds"].items():
            piv = t.get("pivots") or []
            late = [p["date"] for p in piv if not p.get("confirm_date") or p["confirm_date"] > asof]
            if late:
                raise SystemExit(f"멈춤(확정 사건만): {asset} {X}% 확정일이 {asof} 뒤이거나 없는 꺾임점 {late}")
            tail = t.get("tail_unconfirmed") or {}
            if tail.get("date") and any(p["date"] == tail["date"] and p["kind"] == tail.get("kind", p["kind"]) for p in piv):
                raise SystemExit(f"멈춤(확정 사건만): {asset} {X}% 끝 극값 {tail['date']}이 사건 목록에 있다")
            held = {fk: sorted(h["date"] for h in hs) for fk, hs in (t.get("held_near_tail") or {}).items()}
            for fk in ("fake_bottoms", "fake_tops"):
                clash = sorted({f["date"] for f in t.get(fk, [])} & set(held.get(fk, [])))
                if clash:
                    raise SystemExit(f"멈춤(확정 사건만): {asset} {X}% 보류 가짜 {clash}가 {fk}에 있다")
            rec["contrasts"][f"{asset}_{X}"] = {
                "events_used": {k: sum(p["kind"] == k for p in piv) for k in ("low", "high")},
                "last_confirm_date": max((p["confirm_date"] for p in piv), default=None),
                "tail_excluded": ({k: tail.get(k) for k in ("kind", "date", "win_id", "close", "status", "confirm_level",
                                                            "last_close", "move_from_tail")} if tail else None),
                "held_fakes_excluded": held}
    return out, rec


def fmt(x, nd=2) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{x:.{nd}f}"


def run_real(n_null: int = N_NULL, out_dir: Path = RESULTS,
             ev_path: Path = RESULTS / "events.json", features_path: Path = RESULTS / "features_all.parquet",
             meta_path: Path = RESULTS / "features_all_meta.json", prices_path: Path = RESULTS / "prices_close.csv",
             now: datetime | None = None, confirmed_asof: str | None = None) -> int:
    ev = json.loads(Path(ev_path).read_text("utf-8"))
    mode = None
    if confirmed_asof is None:
        guard_real_run(ev, now)
    else:                                   # 확정 사건만: 날짜 가드 대신 끝 극값을 뺀다(부록 14)
        ev, mode = confirmed_only_events(ev, confirmed_asof)
    F, meta = load_features(features_path, meta_path)
    prices = pd.read_csv(prices_path)
    RES_ = Path(out_dir)
    contrasts = load_contrasts(ev)
    panels = {m: build_panel(F, meta, m) for m in sorted({c.market for c in contrasts})}
    figdir = RES_ / "event_study_figs"
    figdir.mkdir(parents=True, exist_ok=True)
    results, tests, leads, nows, prof_rows = [], [], [], [], []
    for c in contrasts:
        p = panels[c.market]
        r = run_contrast(c, p, price_series(prices, c.asset), n_null=n_null)
        results.append(r)
        for df, acc in ((r["table"], tests), (r["lead"], leads), (r["now"], nows)):
            if len(df):
                acc.append(df.assign(asset=c.asset, threshold=c.threshold, kind=c.kind))
        pr = r["profiles"]
        for fi, f in enumerate(p.feats):
            for j, k in enumerate(OFFSETS):
                prof_rows.append({"asset": c.asset, "threshold": c.threshold, "kind": c.kind, "feature": f,
                                  "offset": int(k), "ev_mean": pr["ev_mean"][j, fi], "ev_n": int(pr["ev_n"][j, fi]),
                                  "fk_mean": pr["fk_mean"][j, fi] if pr["fk_mean"] is not None else np.nan,
                                  "pl_mean": pr.get("pl_mean", np.full((21, 1), np.nan))[j, fi] if "pl_mean" in pr else np.nan,
                                  "pl_lo": pr["pl_lo"][j, fi] if "pl_lo" in pr else np.nan,
                                  "pl_hi": pr["pl_hi"][j, fi] if "pl_hi" in pr else np.nan})
        (figdir / f"{c.name}.html").write_text(figure_html(r, p), "utf-8")
    pd.concat(tests).to_csv(RES_ / "event_study_tests.csv", index=False, encoding="utf-8-sig")
    pd.concat(leads).to_csv(RES_ / "event_study_lead.csv", index=False, encoding="utf-8-sig")
    if nows:
        pd.concat(nows).to_csv(RES_ / "event_study_now.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(prof_rows).to_parquet(RES_ / "event_study_profiles.parquet", index=False)
    summary = write_report(results, panels, Path(ev_path), Path(features_path), RES_, n_null, mode=mode)
    (RES_ / "event_study.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), "utf-8")
    return 0


def write_report(results: list[dict], panels: dict, ev_path: Path, features_path: Path, out_dir: Path,
                 n_null: int = N_NULL, mode: dict | None = None) -> dict:
    L, summ = [], {"made": datetime.now(KST).replace(microsecond=0).isoformat(), "contrasts": []}
    if mode:
        summ["mode"] = mode
    n_grid = {m: len(p.feats) for m, p in panels.items()}
    L.append("# 개미 지수 부록 13 사건 연구 결과 (탐색 — 후보는 10/24 뒤 새 사건으로만 확인)\n")
    L.append(f"> 사건 `events.json`(sha256 {sha(ev_path)[:12]}…), 특징 `features_all.parquet`(sha256 "
             f"{sha(features_path)[:12]}…). 특징 격자 W_KR {n_grid.get('kr', 0)}개 · W_U {n_grid.get('us', 0)}개 × 오프셋 "
             f"{len(CLS_OFFSETS)}개(0, +1). 섞기 {n_null}번, 위약 = 사건마다 가격이 비슷한 날 {PLACEBO_POOL}일. 매매 규칙이 아니다.\n")
    if mode:
        tails = "; ".join(f"{k} {v['tail_excluded']['date']}({v['tail_excluded']['kind']})"
                          for k, v in mode["contrasts"].items() if v.get("tail_excluded"))
        L.append(f"> 확정 사건만 모드(부록 14): {mode['asof']} 종가까지 확정된 꺾임점만 사건으로 썼다. 날짜 가드 대신 끝 극값(미확정)과 "
                 f"그 ±3거래일 보류 가짜를 뺐다 — {tails or '끝 극값 없음'}. 끝 극값은 10/23 종가 뒤 다시 판정한다.\n")
    L.append("## 0. 결론표 (분석 2 가르기)\n")
    L.append("| 자산 | 문턱 | 종류 | 사건(창 있음) | 가짜 묶음 대표 | 칸 수(통과) | p<0.05 관측 | 우연 기대 | 귀무 95% | 묶음 p | 누출 시험 | 후보(가르기) | 후보(앞섬) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        c, fam = r["contrast"], r["family"]["통과"]
        cand = r["table"][r["table"]["candidate"]]
        cand_s = ", ".join(f"{a}@{o:+d}" for a, o in zip(cand["feature"], cand["offset"])) or "없음"
        lc = r["lead"][r["lead"]["candidate"]]
        lead_s = ", ".join(f"{a}({sd})" for a, sd in zip(lc["feature"], lc["side"])) or "없음"
        L.append(f"| {ASSET_KO[c.asset]} | {c.threshold}% | {TYPE_KO[c.kind]} | {r['n']['events_with_window']}/"
                 f"{r['n']['events_total']} | {r['n']['fake_reps_with_window']} | {fam['n_tests']} | {fam['observed']} | "
                 f"{fmt(fam['expected_by_chance'])} | {fmt(fam['null_q95'], 0)} | {fmt(fam['p_family'], 3)} | "
                 f"{'의심' if r['leak']['flag'] else '통과'} | {cand_s} | {lead_s} |")
        summ["contrasts"].append({"name": c.name, "n": r["n"], "family": r["family"], "lead_family": r["lead_family"],
                                  "leak": r["leak"], "outside_band": r["outside_band"], "placebo": r["placebo"],
                                  "candidates": cand.reindex(columns=["feature", "offset", "ba", "p", "direction"]).to_dict("records"),
                                  "lead_candidates": lc.reindex(columns=["feature", "side", "rate_events", "rate_fakes",
                                                                         "rate_placebo_mean", "p"]).to_dict("records")})
    L.append("\n- '우연 기대'는 사건·가짜 표지를 섞은 1,000번에서 p < 0.05가 나온 칸 수의 평균이다. 관측이 귀무 95%를 넘지 않으면 "
             "그 대비의 후보는 우연과 구별되지 않는다.")
    L.append("- 8%(주판)와 12%(강건판)가 다르면 둘 다 위 표에 있다. 두 문턱 모두에서 후보인 특징만 '강건 후보'로 본다.\n")
    both = {}
    for r in results:
        c = r["contrast"]
        for f, o in zip(r["table"].loc[r["table"]["candidate"], "feature"], r["table"].loc[r["table"]["candidate"], "offset"]):
            both.setdefault((c.asset, c.kind, f, int(o)), set()).add(c.threshold)
    rb = [k for k, v in both.items() if len(v) == 2]
    L.append("### 강건 후보(8%·12% 모두)\n")
    L.append("\n".join(f"- {ASSET_KO[a]} {TYPE_KO[k]}: {f} @ {o:+d}" for a, k, f, o in rb) or "- 없음")
    summ["robust_candidates"] = [{"asset": a, "kind": k, "feature": f, "offset": o} for a, k, f, o in rb]
    L.append("\n## 1. 사건 전후(위약 띠 밖 특징 수, 설명용)\n")
    L.append("| 대비 | 오프셋 0: 띠 밖 / 특징 | 오프셋 +1: 띠 밖 / 특징 | 그림 |")
    L.append("|---|---|---|---|")
    for r in results:
        c, ob = r["contrast"], r["outside_band"]
        L.append(f"| {c.name} | {ob.get('0', {}).get('outside_band', '—')} / {ob.get('0', {}).get('n_features', '—')} | "
                 f"{ob.get('1', {}).get('outside_band', '—')} / {ob.get('1', {}).get('n_features', '—')} | "
                 f"`event_study_figs/{c.name}.html` |")
    L.append("\n## 3. 앞섬(사건 전 1~5창 극값 닿음)\n")
    L.append("| 대비 | 칸 수(통과) | p<0.05 관측 | 우연 기대 | 귀무 95% | 묶음 p |")
    L.append("|---|---|---|---|---|---|")
    for r in results:
        f = r["lead_family"]["통과"]
        L.append(f"| {r['contrast'].name} | {f['n_tests']} | {f['observed']} | {fmt(f['expected_by_chance'])} | "
                 f"{fmt(f['null_q95'], 0)} | {fmt(f['p_family'], 3)} |")
    L.append("\n## 누출 시험(조회·추천 (b))\n")
    for r in results:
        lk = r["leak"]
        L.append(f"- {r['contrast'].name}: {'의심' if lk['flag'] else '통과'} — " + "; ".join(
            f"오프셋 {k:+d}: " + (f"사건 중앙 {fmt(v.get('median_events'))} vs 가짜 {fmt(v.get('median_fakes'))}"
                                 if "median_events" in v else v.get("result", "")) for k, v in lk["by_offset"].items()))
    L.append("\n## 4. 지금 위치(설명용, '지금이 바닥'이라고 단정하지 않는다)\n")
    L.append("표: `event_study_now.csv`(창 × 특징의 z, 사건·가짜·위약 분포 안 백분위). 실시간 수집 창이라 조회·추천과 댓글 양 특징은 "
             "소급 창과 수집 방식이 달라 비교하지 않거나(결측) 주의 표시를 붙였다.\n")
    L.append("| 대비 | 후보 특징@오프셋 | 마지막 창 | z | 사건 분포 안 백분위 | 가짜 분포 안 백분위 | 위약 분포 안 백분위 | 실시간 주의 |")
    L.append("|---|---|---|---|---|---|---|---|")
    any_now = False
    for r in results:
        nw = r["now"]
        cand = r["table"][r["table"]["candidate"]]
        if not len(nw) or not len(cand):
            continue
        last = nw["win_id"].max()
        for f, o in zip(cand["feature"], cand["offset"]):
            q = nw[(nw["win_id"] == last) & (nw["feature"] == f)]
            if len(q):
                q = q.iloc[0]
                any_now = True
                L.append(f"| {r['contrast'].name} | {f}@{o:+d} | {last} | {fmt(q['z'])} | {fmt(q['pct_in_events'])} | "
                         f"{fmt(q['pct_in_fakes'])} | {fmt(q['pct_in_placebo'])} | {'예' if q['realtime_caution'] else ''} |")
    if not any_now:
        L.append("| — | 가르기 후보 없음 | | | | | | |")
    L.append("")
    L.append("## 산출\n- `event_study_tests.csv`, `event_study_lead.csv`, `event_study_now.csv`, `event_study_profiles.parquet`, "
             "`event_study_figs/*.html`, `event_study.json`")
    (out_dir / "event_study.md").write_text("\n".join(L) + "\n", "utf-8")
    return summ


# ================================================================ 9. 시운전(가짜 사건만)

def random_event_sets(n_w: int, n_true: int, n_fake: int, rng: np.random.Generator, lo: int = 30, hi_pad: int = 2,
                      gap_true: int = 10, gap_fake: int = 4, tries: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """무작위 창 위치: 사건끼리 gap_true 이상, 가짜는 사건·서로에서 gap_fake 이상(실제 꺾임점·가짜 묶음 간격을 흉내)."""
    for _ in range(tries):
        cand = np.arange(lo, n_w - hi_pad)
        ev = []
        for i in rng.permutation(cand):
            if all(abs(i - e) >= gap_true for e in ev):
                ev.append(int(i))
            if len(ev) == n_true:
                break
        fk = []
        for i in rng.permutation(cand):
            if all(abs(i - e) >= gap_fake for e in ev) and all(abs(i - f) >= gap_fake for f in fk):
                fk.append(int(i))
            if len(fk) == n_fake:
                break
        if len(ev) == n_true and len(fk) == n_fake:
            return np.sort(ev), np.sort(fk)
    raise RuntimeError("무작위 사건 배치 실패")


def synthetic_close(D: np.ndarray, rng: np.random.Generator, vol: float = 0.018) -> pd.Series:
    """창 달력 위 가짜 가격(로그 정규 무작위 걸음, 앞 25일 예열). 실제 가격을 쓰지 않는다."""
    first = D[0] - np.timedelta64(40, "D")
    pre = np.arange(first, D[0], np.timedelta64(1, "D"))
    pre = pre[np.isin((pre.astype("datetime64[D]").view("int64") - 4) % 7, [0, 1, 2, 3, 4])][-25:]
    days = np.concatenate([pre, D])
    r = rng.normal(0, vol, len(days))
    return pd.Series(100 * np.exp(np.cumsum(r)), index=days.astype("datetime64[D]"))


def contrast_from_positions(panel: Panel, ev: np.ndarray, fk: np.ndarray, name: str, kind: str = "bottom") -> Contrast:
    e = pd.DataFrame({"date": [str(panel.D[i]) for i in ev], "win_id": panel.win_id[ev]})
    f = pd.DataFrame({"date": [str(panel.D[i]) for i in fk], "win_id": panel.win_id[fk], "run": np.arange(len(fk)),
                      "close": np.nan})
    return Contrast(name, panel.market, "dry", kind, e, f, f.copy(), list(e["date"]))


def dryrun(reps_null: int = 60, reps_inject: int = 40, n_null: int = 500,
           out_md: Path = RESULTS / "dryrun.md", out_json: Path = RESULTS / "dryrun.json") -> dict:
    """(i) 무작위 날짜를 사건으로: '가르는 특징' 수가 우연 수준인지(섞기 귀무의 보정). (ii) 특정 특징을 사건 전에 일부러 올린
    합성 판: 그 특징을 잡는지(검출률)와 다른 특징 오검출이 우연 수준인지. (iii) (b)를 사건일에 올린 판: 누출 시험이 잡는지.
    실제 events.json·가격을 읽지 않는다. 가격 그림·위약은 가짜 가격(무작위 걸음)으로만 돌린다."""
    t0 = datetime.now(KST).replace(microsecond=0)
    F, meta = load_features()
    feats = grid_features(meta)
    panels = {m: build_panel(F, meta, m, feats) for m in ("kr", "us")}
    out = {"made": t0.isoformat(), "inputs": {"features_all_sha256": sha(RESULTS / "features_all.parquet")},
           "events_json_read": False, "prices_read": False, "settings": {
               "reps_null": reps_null, "reps_inject": reps_inject, "n_null": n_null, "placebo_pool": PLACEBO_POOL,
               "alpha": ALPHA, "cls_offsets": list(CLS_OFFSETS), "lead": list(LEAD)}}
    # 표본 크기 시나리오: events.json 사건 수와 비슷한 크기(날짜는 쓰지 않음)
    scen = [("작은 판(사건 4·가짜 대표 8)", 4, 8), ("중간 판(사건 8·가짜 대표 6)", 8, 6), ("큰 판(사건 18·가짜 대표 12)", 18, 12)]

    # (i) 귀무 보정
    cal = []
    for m in ("kr", "us"):
        p = panels[m]
        for nm, nt, nf in scen:
            rng = np.random.default_rng(int(hashlib.sha256(f"null:{m}:{nm}".encode()).hexdigest()[:8], 16))
            for rep in range(reps_null):
                ev, fk = random_event_sets(len(p.win_id), nt, nf, rng)
                cls = classify(p, ev, fk, n_null=n_null, seed=int(rng.integers(1 << 31)))
                lead = lead_analysis(p, cls, None)
                pe = cls["p_obs"]
                fam_all = family_count(pe, cls["T_null"])
                fam_pass = family_count(np.where(cls["table"]["feature"].map(lambda f: meta[f]["status"]).eq("통과")
                                                 .to_numpy(), pe, np.nan), cls["T_null"])
                lf = family_count(lead["p_obs"], lead["T_null"])
                cal.append({"market": m, "scenario": nm, "rep": rep, "n_tests": fam_all["n_tests"],
                            "observed": fam_all["observed"], "expected": fam_all["expected_by_chance"],
                            "q95": fam_all["null_q95"], "p_family": fam_all["p_family"],
                            "pass_observed": fam_pass["observed"], "pass_expected": fam_pass["expected_by_chance"],
                            "lead_observed": lf["observed"], "lead_expected": lf["expected_by_chance"],
                            "lead_q95": lf["null_q95"], "lead_n_tests": lf["n_tests"],
                            "lead_p_family": lf["p_family"],
                            "share_p_lt_alpha": float(np.nanmean(pe < ALPHA)) if np.isfinite(pe).any() else np.nan,
                            "share_lead_p_lt_alpha": float(np.nanmean(lead["p_obs"] < ALPHA))
                            if np.isfinite(lead["p_obs"]).any() else np.nan})
    C = pd.DataFrame(cal)
    C["over"] = (C["observed"] > C["q95"]).astype(float)
    g = C.groupby(["market", "scenario"], sort=False)
    calib = g.agg(reps=("rep", "size"), n_tests=("n_tests", "median"), obs_mean=("observed", "mean"),
                  exp_mean=("expected", "mean"), share_p=("share_p_lt_alpha", "mean"),
                  fam_reject=("p_family", lambda s: float((s < ALPHA).mean())),
                  over_q95=("over", "mean"), lead_obs_mean=("lead_observed", "mean"),
                  lead_exp_mean=("lead_expected", "mean"), lead_share_p=("share_lead_p_lt_alpha", "mean"),
                  lead_fam_reject=("lead_p_family", lambda s: float((s < ALPHA).mean())),
                  pass_obs_mean=("pass_observed", "mean"), pass_exp_mean=("pass_expected", "mean")).reset_index()
    out["null_calibration"] = calib.to_dict("records")
    out["null_calibration_overall"] = {
        "replicates": int(len(C)), "share_feature_tests_p_lt_alpha": float(C["share_p_lt_alpha"].mean()),
        "family_reject_rate": float((C["p_family"] < ALPHA).mean()),
        "observed_mean": float(C["observed"].mean()), "expected_mean": float(C["expected"].mean()),
        "lead_share_p_lt_alpha": float(C["share_lead_p_lt_alpha"].mean()),
        "lead_family_reject_rate": float((C["lead_p_family"] < ALPHA).mean())}

    # (ii) 주입 신호: 원값에 사건 전 −3~0 창 +δ × (그 특징 원값 SD)
    targets = [("kr", "act_ln_posts_per_h"), ("kr", "pr_P_bar"), ("us", "tp_LS_A")]
    deltas = [0.0, 0.5, 1.0, 2.0, 3.0]
    inj = []
    for m, tf in targets:
        p0 = panels[m]
        g0 = F[F["market"] == m].sort_values("close_kst").reset_index(drop=True)
        X0 = g0[feats].to_numpy(float)
        j = feats.index(tf)
        sd = np.nanstd(X0[:, j])
        for nm, nt, nf in scen[1:]:
            for d in deltas:
                rng = np.random.default_rng(int(hashlib.sha256(f"inj:{m}:{tf}:{nm}:{d}".encode()).hexdigest()[:8], 16))
                for rep in range(reps_inject):
                    ev, fk = random_event_sets(len(p0.win_id), nt, nf, rng)
                    x = X0[:, j].copy()
                    for e in ev:
                        for k in range(-3, 1):
                            if 0 <= e + k < len(x) and np.isfinite(x[e + k]):
                                x[e + k] += d * sd
                    p = with_column(p0, F, tf, x)
                    cls = classify(p, ev, fk, n_null=n_null, seed=int(rng.integers(1 << 31)))
                    tb = cls["table"]
                    lead = lead_analysis(p, cls, None)
                    lt = lead["table"]
                    hit0 = tb[(tb["feature"] == tf) & (tb["offset"] == 0)]
                    others = tb[(tb["feature"] != tf) & tb["p"].notna()]
                    lh = lt[(lt["feature"] == tf) & (lt["side"] == "상위 10%")]
                    inj.append({"market": m, "target": tf, "scenario": nm, "delta_sd": d, "rep": rep,
                                "target_p0": float(hit0["p"].iloc[0]) if len(hit0) else np.nan,
                                "target_ba0": float(hit0["ba"].iloc[0]) if len(hit0) else np.nan,
                                "target_lead_p": float(lh["p"].iloc[0]) if len(lh) else np.nan,
                                "others_share_p_lt_alpha": float((others["p"] < ALPHA).mean()) if len(others) else np.nan})
    Ij = pd.DataFrame(inj)
    gi = Ij.groupby(["market", "target", "scenario", "delta_sd"], sort=False)
    power = gi.agg(reps=("rep", "size"), detect_offset0=("target_p0", lambda s: float((s < ALPHA).mean())),
                   ba0_median=("target_ba0", "median"), detect_lead=("target_lead_p", lambda s: float((s < ALPHA).mean())),
                   others_false=("others_share_p_lt_alpha", "mean")).reset_index()
    out["injection"] = power.to_dict("records")

    # (iii) 누출 시험: (b)를 사건 오프셋 0·+1에 +δ SD
    leak_rows = []
    for m in ("kr", "us"):
        g0 = F[F["market"] == m].sort_values("close_kst").reset_index(drop=True)
        X0 = g0[feats].to_numpy(float)
        j = feats.index(LEAK_FEATURE)
        sd = np.nanstd(X0[:, j])
        for d in (0.0, 1.0, 2.0):
            rng = np.random.default_rng(int(hashlib.sha256(f"leak:{m}:{d}".encode()).hexdigest()[:8], 16))
            flags = []
            for rep in range(reps_inject):
                ev, fk = random_event_sets(len(g0), 8, 6, rng)
                x = X0[:, j].copy()
                for e in ev:
                    for k in LEAK_OFFSETS:
                        if 0 <= e + k < len(x) and np.isfinite(x[e + k]):
                            x[e + k] += d * sd
                p = with_column(panels[m], F, LEAK_FEATURE, x)
                flags.append(leak_test(p, ev, fk)["flag"])
            leak_rows.append({"market": m, "delta_sd": d, "reps": reps_inject, "flag_rate": float(np.mean(flags))})
    out["leak_injection"] = leak_rows

    # (v) 분석 1 위약 띠 보정: 무작위 날짜 사건의 평균 z가 위약 띠(2.5~97.5%) 밖으로 나가는 몫(가짜 가격으로 위약을 고름)
    band = []
    for m in ("kr", "us"):
        p = panels[m]
        for nm, nt, nf in scen[1:]:
            rng = np.random.default_rng(int(hashlib.sha256(f"band:{m}:{nm}".encode()).hexdigest()[:8], 16))
            for rep in range(reps_inject):
                ev, fk = random_event_sets(len(p.win_id), nt, nf, rng)
                close = synthetic_close(p.D, rng)
                dates = [str(p.D[i]) for i in ev]
                pools, _ = placebo_pools(p, close, dates, dates)
                pr = profiles(p, ev, fk, pools)
                ld = lead_analysis(p, classify(p, ev, fk, n_null=20, seed=int(rng.integers(1 << 31))), pools)["table"]
                ldt = ld[ld["status"] == "시험"].reindex(columns=["p_vs_placebo"])
                for k in (0, 1):
                    j = int(np.flatnonzero(OFFSETS == k)[0])
                    mm = pr["ev_mean"][j]
                    ok = np.isfinite(mm) & np.isfinite(pr["pl_lo"][j])
                    o = (mm < pr["pl_lo"][j]) | (mm > pr["pl_hi"][j])
                    band.append({"market": m, "scenario": nm, "offset": k, "rep": rep,
                                 "share_outside": float(o[ok].mean()) if ok.any() else np.nan,
                                 "lead_share_p_vs_placebo_lt_alpha": float((ldt["p_vs_placebo"] < ALPHA).mean())
                                 if len(ldt) else np.nan})
    B = pd.DataFrame(band)
    out["placebo_band_calibration"] = B.groupby(["market", "scenario", "offset"], sort=False).agg(
        reps=("rep", "size"), share_outside_mean=("share_outside", "mean"),
        lead_share_p_vs_placebo_lt_alpha=("lead_share_p_vs_placebo_lt_alpha", "mean")).reset_index().to_dict("records")

    # (iv) 전체 경로 한 번(무작위 날짜 + 가짜 가격): 분석 1·3·4·그림이 돌아가는지
    p = panels["kr"]
    rng = np.random.default_rng(SEED)
    ev, fk = random_event_sets(len(p.win_id), 8, 6, rng)
    c = contrast_from_positions(p, ev, fk, "DRY")
    close = synthetic_close(p.D, rng)
    r = run_contrast(c, p, close, n_null=n_null)
    (RESULTS / "dryrun_fig_sample.html").write_text(
        figure_html(r, p, "시운전 견본 — 사건은 무작위 날짜, 가격은 가짜 무작위 걸음이다(실제 사건·가격 아님). "), "utf-8")
    out["full_path_sample"] = {"n": r["n"], "family_pass": r["family"]["통과"], "lead_family_pass": r["lead_family"]["통과"],
                               "leak": {"flag": r["leak"]["flag"], "testable": r["leak"]["testable"]},
                               "outside_band": {k: {kk: vv for kk, vv in v.items() if kk != "features"}
                                                for k, v in r["outside_band"].items()},
                               "placebo": r["placebo"], "now_rows": int(len(r["now"])),
                               "now_windows": sorted(r["now"]["win_id"].unique().tolist()) if len(r["now"]) else []}
    # 창 종류 보정 효과(월요일 쏠림): 평일 아닌 창의 z 평균, 보정 전후
    eff = {}
    for m in ("kr", "us"):
        g0 = F[F["market"] == m].sort_values("close_kst").reset_index(drop=True)
        nonwk = (g0["kind_class"] == "nonweekday").to_numpy()
        ok = g0["judge_ok"].to_numpy(bool)
        for f in ("act_ln_posts_per_h", "act_ln_comments_pre_per_h", "act_share_night", "tp_SEMI_A"):
            x = np.where(ok, g0[f].to_numpy(float), np.nan)
            z_raw, z_adj = z_prev(x), z_prev(kind_adjust(x, nonwk))
            eff[f"{m}:{f}"] = {"nonweekday_mean_z_raw": float(np.nanmean(z_raw[nonwk])),
                               "nonweekday_mean_z_adj": float(np.nanmean(z_adj[nonwk])),
                               "weekday_mean_z_raw": float(np.nanmean(z_raw[~nonwk])),
                               "weekday_mean_z_adj": float(np.nanmean(z_adj[~nonwk]))}
    out["kind_adjust_effect"] = eff
    out["grid"] = {"features": feats, "n_features": len(feats),
                   "by_status": {s: [f for f in feats if meta[f]["status"] == s] for s in ("통과", "설명용")}}
    out["seconds"] = (datetime.now(KST) - t0).total_seconds()
    Path(out_json).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), "utf-8")
    Path(out_md).write_text(dryrun_md(out, calib, power, leak_rows, eff), "utf-8")
    return out


def _rng(vals) -> str:
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return "—" if not v else (f"{min(v):.2f}" if abs(max(v) - min(v)) < 0.005 else f"{min(v):.2f}~{max(v):.2f}")


def dryrun_md(out: dict, calib: pd.DataFrame, power: pd.DataFrame, leak_rows: list, eff: dict) -> str:
    o = out["null_calibration_overall"]
    s = out["settings"]
    inj0, inj1, inj2 = (_rng(power.loc[power["delta_sd"] == d, "detect_offset0"]) for d in (0.0, 1.0, 2.0))
    oth = _rng(power["others_false"])
    lk0 = _rng([r["flag_rate"] for r in leak_rows if r["delta_sd"] == 0])
    lk1 = _rng([r["flag_rate"] for r in leak_rows if r["delta_sd"] >= 1])
    band = _rng([r["share_outside_mean"] for r in out["placebo_band_calibration"]])
    lband = _rng([r["lead_share_p_vs_placebo_lt_alpha"] for r in out["placebo_band_calibration"]])
    L = ["# 부록 13 사건 연구 시운전 (가짜 사건만)\n",
         f"> {out['made']}. 실제 `events.json`과 가격(`prices_close.csv`)은 읽지 않았다. 사건은 무작위 날짜, 가격은 가짜 무작위 걸음이다. "
         f"특징은 `features_all.parquet` 실제 값(가격 없음). 코드 `events/code/event_study.py`, 테스트 `test_event_study.py`.\n",
         f"> 설정: 격자 특징 {out['grid']['n_features']}개(통과 {len(out['grid']['by_status']['통과'])} · 설명용 "
         f"{len(out['grid']['by_status']['설명용'])}) × 오프셋 {s['cls_offsets']}. 귀무 보정 반복 {s['reps_null']}번 × 3 크기 × 2 시장, "
         f"섞기 {s['n_null']}번(실제 실행은 {N_NULL}번), 주입 반복 {s['reps_inject']}번.\n",
         "## 0. 결론\n",
         f"- (i) 무작위 날짜: 특징 칸 가운데 p < 0.05인 몫 {o['share_feature_tests_p_lt_alpha']:.3f}(기대 0.05 이하), "
         f"'관측 개수가 귀무에서 드물다'(묶음 p < 0.05) 판정 비율 {o['family_reject_rate']:.3f}(기대 0.05 이하). "
         f"관측 개수 평균 {o['observed_mean']:.2f} vs 우연 기대 평균 {o['expected_mean']:.2f}. 앞섬 분석 p < 0.05 몫 "
         f"{o['lead_share_p_lt_alpha']:.3f}, 묶음 판정 비율 {o['lead_family_reject_rate']:.3f}.",
         f"- (ii) 주입 신호(사건 전 −3~0창에 + δ SD): 오프셋 0 검출률이 δ = 0에서 {inj0}, δ = 1에서 {inj1}, δ = 2에서 {inj2}"
         f"(대상·크기별 범위). 다른 특징 오검출 몫 {oth}. 사건이 8개 안팎이면 1 SD짜리 차이는 절반 넘게 놓친다.",
         f"- (iii) 누출 시험: (b)를 사건일에 1 SD 이상 올리면 '누출 의심'이 {lk1}로 켜진다. 규칙이 '사건 중앙값이 가짜보다 크면 "
         f"의심'(점추정)이라 주입이 없어도 {lk0} 켜진다 — 보수적이다(의심이면 조회·추천 묶음 전체를 후보에서 뺀다).",
         f"- 위약: 무작위 날짜 사건 평균이 분석 1 위약 띠 밖인 특징 몫 {band}(기대 약 0.05), 분석 3 위약 대비 p < 0.05 몫 {lband}"
         "(기대 0.05 이하). 처음 판 띠(뽑기 분위)는 0.10~0.29로 너무 좁아 고쳤다(3-2절).\n",
         "## 1. 무작위 날짜 귀무 보정(분석 2 가르기·분석 3 앞섬)\n",
         "| 시장 | 크기 | 반복 | 칸 수 | p<0.05 몫 | 관측 평균 | 우연 기대 평균 | 통과 묶음 관측/기대 | 관측 > 귀무 95% 비율 | 묶음 p<0.05 비율 | 앞섬 p<0.05 몫 | 앞섬 관측/기대 | 앞섬 묶음 p<0.05 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in calib.to_dict("records"):
        L.append(f"| {r['market']} | {r['scenario']} | {r['reps']} | {fmt(r['n_tests'], 0)} | {fmt(r['share_p'], 3)} | "
                 f"{fmt(r['obs_mean'])} | {fmt(r['exp_mean'])} | {fmt(r['pass_obs_mean'])}/{fmt(r['pass_exp_mean'])} | "
                 f"{fmt(r['over_q95'], 3)} | {fmt(r['fam_reject'], 3)} | "
                 f"{fmt(r['lead_share_p'], 3)} | {fmt(r['lead_obs_mean'])}/{fmt(r['lead_exp_mean'])} | "
                 f"{fmt(r['lead_fam_reject'], 3)} |")
    L += ["\n- 섞기 귀무는 표본이 작아 BA 값이 몇 단계뿐이라 p가 0.05에 딱 닿지 않는다. 그래서 '우연 기대'가 0.05 × 칸 수보다 작게 "
          "나오는 것이 정상이고, 실제 실행도 이 귀무로 센다.\n",
          "## 2. 주입 신호(사건 전 −3~0창 원값 + δ × 그 특징 SD)\n",
          "| 시장 | 대상 특징 | 크기 | δ(SD) | 반복 | 오프셋 0 검출률(p<0.05) | 오프셋 0 BA 중앙 | 앞섬(상위 10%) 검출률 | 다른 특징 오검출 몫 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in power.to_dict("records"):
        L.append(f"| {r['market']} | {r['target']} | {r['scenario']} | {r['delta_sd']:g} | {r['reps']} | "
                 f"{fmt(r['detect_offset0'], 3)} | {fmt(r['ba0_median'])} | {fmt(r['detect_lead'], 3)} | "
                 f"{fmt(r['others_false'], 3)} |")
    L += ["\n## 3. 누출 시험 주입((b)를 사건 오프셋 0·+1에 + δ SD, 사건 8·가짜 대표 6)\n",
          "| 시장 | δ(SD) | 반복 | '누출 의심' 켜진 비율 |", "|---|---|---|---|"]
    for r in leak_rows:
        L.append(f"| {r['market']} | {r['delta_sd']:g} | {r['reps']} | {fmt(r['flag_rate'], 3)} |")
    L += ["\n## 3-2. 위약 보정(무작위 날짜 사건 + 가짜 가격으로 고른 위약)\n",
          "| 시장 | 크기 | 오프셋 | 반복 | 분석 1: 띠 밖 특징 몫(기대 약 0.05) | 분석 3: 위약 대비 p<0.05 몫(기대 0.05 이하) |",
          "|---|---|---|---|---|---|"]
    for r in out["placebo_band_calibration"]:
        L.append(f"| {r['market']} | {r['scenario']} | {r['offset']:+d} | {r['reps']} | {fmt(r['share_outside_mean'], 3)} | "
                 f"{fmt(r['lead_share_p_vs_placebo_lt_alpha'], 3)} |")
    L.append("\n- 띠 = 위약 날 z 평균 ± t × SD ÷ √(사건 수). 처음 판(사건마다 후보 20일 중 하나씩 1,000번 뽑은 평균의 2.5~97.5% "
             "분위)은 같은 크기 시운전에서 띠 밖 몫이 0.10~0.29로 너무 좁아 버렸다(후보 20일이 서로 비슷한 날이라 뽑기 평균이 덜 "
             "흔들림). 분석 3의 위약 비교도 같은 이유로 이항 분포로 바꿨다. 분석 1은 설명용이고 판정은 분석 2의 섞기 귀무로 한다.")
    fp = out["full_path_sample"]
    L += ["\n## 4. 전체 경로 견본(무작위 날짜 + 가짜 가격, W_KR)\n",
          f"- 사건 {fp['n']['events_with_window']} · 가짜 대표 {fp['n']['fake_reps_with_window']}. 분석 2 통과 묶음: 칸 "
          f"{fp['family_pass']['n_tests']}, 관측 {fp['family_pass']['observed']}, 우연 기대 {fmt(fp['family_pass']['expected_by_chance'])}. "
          f"앞섬: 칸 {fp['lead_family_pass']['n_tests']}, 관측 {fp['lead_family_pass']['observed']}, 우연 기대 "
          f"{fmt(fp['lead_family_pass']['expected_by_chance'])}.",
          f"- 위약 후보 {fp['placebo'].get('n_candidates')}일(가짜 가격 기준). 위약 띠 밖 특징: "
          + ", ".join(f"오프셋 {k}: {v['outside_band']}/{v['n_features']}(독립이면 기대 {v['expected_if_independent']})"
                      for k, v in fp["outside_band"].items()) + ".",
          f"- 지금 위치 표 {fp['now_rows']}줄(창 {', '.join(fp['now_windows'])}). 그림 견본 `results/dryrun_fig_sample.html`"
          "(가짜 가격·무작위 날짜라는 표시를 붙임).\n",
          "## 5. 창 종류 보정(주말·연휴 뒤 창 쏠림)\n",
          "| 시장:특징 | 평일 아닌 창 z 평균(보정 전 → 후) | 평일 창 z 평균(보정 전 → 후) |", "|---|---|---|"]
    for k, v in eff.items():
        L.append(f"| {k} | {v['nonweekday_mean_z_raw']:+.2f} → {v['nonweekday_mean_z_adj']:+.2f} | "
                 f"{v['weekday_mean_z_raw']:+.2f} → {v['weekday_mean_z_adj']:+.2f} |")
    L += ["\n## 6. 실제 실행 전 남은 일\n",
          "- 10/24 06:00 KST 뒤 부록 12 등록 판정 → `py events.py all`(10/22·10/23 종가, 끝 극값 다시 판정) → 스냅샷에 10/22·10/23 창을 "
          "더하면 `py features_all.py` → `py event_study.py run`. `run`은 날짜와 events.json 재판정 표지를 확인하고 아니면 멈춘다.",
          f"- 걸린 시간 {out['seconds']:.0f}초."]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dryrun", "run"])
    ap.add_argument("--reps-null", type=int, default=60)
    ap.add_argument("--reps-inject", type=int, default=40)
    ap.add_argument("--n-null", type=int, default=500)
    ap.add_argument("--confirmed-only", action="store_true",
                    help="run: 확정 사건만(CONFIRMED_ASOF 종가로 판정한 events.json). 날짜 가드 대신 끝 극값을 뺀다")
    ap.add_argument("--out", default=None, help="run 산출 폴더(기본: results, --confirmed-only면 results/confirmed_1022)")
    a = ap.parse_args(argv)
    if a.mode == "run":
        out = Path(a.out) if a.out else (CONFIRMED_OUT if a.confirmed_only else RESULTS)
        return run_real(out_dir=out, confirmed_asof=CONFIRMED_ASOF if a.confirmed_only else None)
    o = dryrun(reps_null=a.reps_null, reps_inject=a.reps_inject, n_null=a.n_null)
    print(json.dumps(o["null_calibration_overall"], ensure_ascii=False, indent=1))
    print("seconds", o["seconds"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
