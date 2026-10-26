"""verify_indicator_replica: indicator_replica 갈래의 독립 반박 검증 (2026-10-13).

연구 갈래 모듈(indicator_replica_*.py)은 불러오지 않는다. harness는 패널·보유 이력(load_panel, membership,
universe_mask, eligible, WINDOWS)만 쓰고, 미래 수익·구간 날짜·Spearman IC·Newey-West t·집합 초과는 여기서 다시 짠다.

다시 재는 머리 숫자
  H1 ex4(=RS Main) 순위 IC: front h20 +0.054, back h5 −0.113, back h20 −0.115
  H2 ex4 상위 30% 집합 초과: back h5 −1.39
  H3 Strong(ex4 상위30% & FadeGap<=1) 집합 초과: front h5 −1.00, back h5 −1.68(t −3.6), 2025 −0.25, 2024 −0.23
  H4 Strong 변동성 5분위 중립: front h5 −0.40, back h5 −1.77(t −4.8)
  H5 Watch − 안 꺾인 강세: back h5 +2.20
  H6 스크린샷 State>=1 대 0: h20 +7.7 (패널 안 종목만)

반박 시도
  T1 시점: 자체 잘라내기 미래 정보 시험, t+1 시가 진입 확인, 진입 하루 늦추기(t+2 시가), 1·2일 보유
  T2 분해: ex4 상위 30% 안에서 FadeGap<=1 대 >1 (같은 날 짝 차이, 원값·변동성 중립)
  T3 우연: 같은 날 같은 개수 무작위 집합(전체/변동성 5분위 층화) 2,000회, 5일 겹치지 않는 표본 t
  T4 쏠림: 한 종목 빼기 전수, 종목 기여 상위, 월별, 6/22~7/31 빼기, 극단 5일 빼기
  T5 문턱 민감도: FadeGap<=0.5/1/2 × ex4 상위 20/30/40% (9칸, 설명용)
  T6 비용: 편도 0.1%(왕복 0.2%p)
  T7 스크린샷 안에서 연구의 Strong 근사(rs_main 촬영 안 상위30% & fade_gap<=1)와 지표 실제 State 비교
  T8 데스크 7종목 규칙(deskx) 위: 재조정일 신규 매수 후보가 Strong이면 건너뛰기/현금 두기, ex4 상위30%면 건너뛰기
     (3변형 × 첫 매수일 포함/제외 = 6칸)
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
RNG = np.random.default_rng(777)
NDRAW = 2000

P = harness.load_panel()
C, O, H = P["C"], P["O"], P["H"]
Q, QH = P["bench"]["C"], P["bench"]["H"]
CAL = C.index
COLS = list(C.columns)


# ───────────────────────── 자체 채점 도구 ─────────────────────────
def fwd(h: int, skip: int = 0) -> pd.DataFrame:
    """t 신호 → t+1+skip 시가 매수 → t+1+skip+h 시가 매도."""
    return O.shift(-(1 + skip + h)) / O.shift(-(1 + skip)) - 1.0


def wdates(name: str, h: int, skip: int = 0) -> pd.DatetimeIndex:
    a, b = harness.WINDOWS[name]
    idx = np.arange(len(CAL))
    ex = idx + 1 + skip + h
    ok = (CAL >= pd.Timestamp(a)) & (CAL <= pd.Timestamp(b)) & (ex < len(CAL))
    exd = np.array([CAL[min(e, len(CAL) - 1)] for e in ex])
    ok &= exd <= pd.Timestamp(b)
    return CAL[ok]


def nw_t(x, lag: int) -> float:
    x = np.asarray(pd.Series(x).dropna(), float)
    n = len(x)
    if n < 5:
        return float("nan")
    e = x - x.mean()
    s = e @ e / n
    for k in range(1, min(lag, n - 1) + 1):
        s += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    return float(x.mean() / np.sqrt(s / n)) if s > 0 else float("nan")


def nonoverlap_t(d: pd.Series, h: int) -> dict:
    """h일마다 하나씩 뽑은 겹치지 않는 표본 h개(시작점별)의 평균과 단순 t."""
    d = d.dropna()
    out = []
    for off in range(h):
        x = d.iloc[off::h].to_numpy(float)
        if len(x) >= 5:
            out.append((x.mean(), x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))))
    if not out:
        return {"note": f"h={h}이면 시작점별 표본이 5개 미만이라 생략"}
    m = np.array(out)
    return {"mean_range_pp": [round(float(m[:, 0].min()) * 100, 2), round(float(m[:, 0].max()) * 100, 2)],
            "t_range": [round(float(m[:, 1].min()), 2), round(float(m[:, 1].max()), 2)],
            "t_median": round(float(np.median(m[:, 1])), 2), "n_each": int(len(d) // h)}


def block_boot_t(d: pd.Series, block: int = 10, B: int = 4000, seed: int = 5) -> dict:
    """원형 블록 부트스트랩(블록 10거래일) 표준오차로 낸 t. 겹치는 h일 수익과 여러 날 이어지는 집합 소속을 반영."""
    x = d.dropna().to_numpy(float)
    n = len(x)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(B, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(B, -1)[:, :n] % n
    means = x[idx].mean(axis=1)
    return {"t_block10": round(float(x.mean() / means.std()), 2),
            "ci95_pp": [round(float(np.percentile(means, 2.5)) * 100, 2), round(float(np.percentile(means, 97.5)) * 100, 2)]}


def row_spearman(a: pd.DataFrame, b: pd.DataFrame, min_n: int = 10) -> pd.Series:
    ok = a.notna() & b.notna()
    ra = a.where(ok).rank(axis=1).to_numpy(float)
    rb = b.where(ok).rank(axis=1).to_numpy(float)
    out = []
    for i in range(len(a)):
        m = ok.iloc[i].to_numpy()
        if m.sum() < min_n:
            out.append(np.nan)
            continue
        out.append(np.corrcoef(ra[i, m], rb[i, m])[0, 1])
    return pd.Series(out, index=a.index)


def pr_um(x: pd.DataFrame) -> pd.DataFrame:
    """t일까지 아는 종목군(보유 + 봉 130개) 안 백분위."""
    return x.where(UM).rank(axis=1, pct=True, method="average")


UM = harness.universe_mask(P)
MEM = harness.membership()
# 자체 채점 가능 종목: 보유 & 봉 130 & t 종가 & t+1 시가 (harness.eligible과 같은 정의를 다시 계산)
EL = MEM.reindex_like(C).fillna(False).astype(bool) & (C.notna().cumsum() >= harness.MIN_BARS) & C.notna() \
    & O.shift(-1).notna()

# ───────────────────────── 신호 (t 종가까지) ─────────────────────────
def signals(Pp: dict) -> dict:
    Cc, Hh = Pp["C"], Pp["H"]
    q, qh = Pp["bench"]["C"], Pp["bench"]["H"]
    ex4 = 100 * (Cc / Cc.shift(4)).sub(q / q.shift(4), axis=0)
    lat = (100 * (Hh / Cc.shift(4)).sub(qh / q.shift(4), axis=0)).rolling(5, min_periods=5).max()
    fg = (lat - ex4).clip(lower=0)
    return {"ex4": ex4, "lat": lat, "fg": fg}


S = signals(P)
EX4, LAT, FG = S["ex4"], S["lat"], S["fg"]
PR4, PRL, PRF = pr_um(EX4), pr_um(LAT), pr_um(FG)
STRONG = (PR4 >= 0.70) & (FG <= 1)
TOP30 = PR4 >= 0.70
L30, F33 = PRL >= 0.70, PRF >= 2 / 3
VOL = C.pct_change().rolling(20, min_periods=20).std()
VQ = np.ceil(pr_um(VOL) * 5).clip(1, 5)


def excl_cols(w):
    return [c for c in COLS if c not in harness.EXCLUDE.get(w, set())]


def set_series(flag: pd.DataFrame, h: int, w: str, cols=None, vol_neutral=False, skip=0, base_flag=None) -> pd.Series:
    """날짜별 (집합 평균 − 비교 평균). 비교는 eligible 전체(base_flag=None) 또는 다른 집합."""
    cols = cols or excl_cols(w)
    d = wdates(w, h, skip)
    e = EL.loc[d, cols]
    r = fwd(h, skip).loc[d, cols]
    if vol_neutral:
        v = VQ.loc[d, cols]
        e = e & v.notna()
        r = r.where(e)
        adj = r.copy()
        for k in range(1, 6):
            m = v == k
            adj = adj.where(~m, r.sub(r.where(m).mean(axis=1), axis=0))
        r = adj.where(e)
        base = pd.Series(0.0, index=d)
    else:
        r = r.where(e)
        base = r.mean(axis=1)
    f = flag.reindex(index=CAL, columns=COLS).fillna(False).astype(bool).loc[d, cols] & e
    a = r.where(f).mean(axis=1)
    if base_flag is not None:
        g = base_flag.reindex(index=CAL, columns=COLS).fillna(False).astype(bool).loc[d, cols] & e
        base = r.where(g).mean(axis=1)
    return (a - base).dropna()


def summ(d: pd.Series, h: int) -> dict:
    return {"x_pp": round(float(d.mean()) * 100, 3), "t_nw": round(nw_t(d, h), 2), "days": int(len(d)),
            "hit": round(float((d > 0).mean()) * 100, 1), "median_pp": round(float(d.median()) * 100, 3)}


def ic_cell(score: pd.DataFrame, h: int, w: str, cols=None) -> dict:
    cols = cols or excl_cols(w)
    d = wdates(w, h)
    e = EL.loc[d, cols] & score.loc[d, cols].notna()
    ic = row_spearman(score.loc[d, cols].where(e), fwd(h).loc[d, cols].where(e)).dropna()
    return {"ic": round(float(ic.mean()), 4), "t_nw": round(nw_t(ic, h), 2), "days": int(len(ic))}, ic


# ───────────────────────── 무작위 집합 null ─────────────────────────
def random_null(flag: pd.DataFrame, h: int, w: str, stratify: bool) -> dict:
    cols = excl_cols(w)
    d = wdates(w, h)
    e = EL.loc[d, cols].to_numpy() & (VQ.loc[d, cols].notna().to_numpy() if stratify else True)
    r = fwd(h).loc[d, cols].to_numpy(float)
    e &= np.isfinite(r)
    f = flag.reindex(index=CAL, columns=COLS).fillna(False).astype(bool).loc[d, cols].to_numpy() & e
    b = VQ.loc[d, cols].fillna(0).to_numpy(int) if stratify else np.ones_like(r, dtype=int)
    b = np.where(e, b, 0)
    D, N = r.shape
    rr = np.where(e, r, 0.0)
    if stratify:   # 관측값도 버킷 평균을 뺀 값
        adj = np.zeros_like(rr)
        for k in range(1, 6):
            m = b == k
            mu = np.where(m.sum(1) > 0, (rr * m).sum(1) / np.maximum(m.sum(1), 1), 0.0)
            adj += np.where(m, rr - mu[:, None], 0.0)
        val = adj
        base = np.zeros(D)
    else:
        val = rr
        base = (rr * e).sum(1) / np.maximum(e.sum(1), 1)
    nset = f.sum(1)
    use = nset > 0
    obs_series = (val * f).sum(1)[use] / nset[use] - base[use]
    obs = float(obs_series.mean())
    K = np.zeros((D, 6), int)
    cnt = np.zeros((D, 6), int)
    for k in range(1, 6):
        K[:, k] = (f & (b == k)).sum(1)
        cnt[:, k] = (b == k).sum(1)
    lower = np.cumsum(cnt, axis=1) - cnt
    bi = np.where(b > 0, b, 0)
    start = np.take_along_axis(lower, bi, axis=1)
    kk = np.take_along_axis(K, bi, axis=1)
    sims = np.empty(NDRAW)
    rows_ = np.arange(D)[:, None]
    for i in range(NDRAW):
        u = RNG.random((D, N))
        key = np.where(b > 0, b + u, np.inf)
        order = np.argsort(key, axis=1)
        pos = np.empty_like(order)
        pos[rows_, order] = np.arange(N)[None, :]
        sel = (b > 0) & (pos - start < kk)
        sims[i] = float(((val * sel).sum(1)[use] / nset[use] - base[use]).mean())
    p = float((np.abs(sims - sims.mean()) >= abs(obs - sims.mean())).mean())
    return {"obs_pp": round(obs * 100, 3), "null_mean_pp": round(float(sims.mean()) * 100, 3),
            "null_sd_pp": round(float(sims.std()) * 100, 3), "z": round((obs - sims.mean()) / sims.std(), 2),
            "p_two_sided": round(max(p, 1 / NDRAW), 4), "draws": NDRAW, "avg_set_n": round(float(nset[use].mean()), 1)}


# ───────────────────────── 스크린샷 ─────────────────────────
SHOT_ENTRY = {"S5": "2026-07-07", "S6": "2026-07-09", "S1": "2026-07-09", "S2": "2026-07-10",
              "S3": "2026-07-14", "S4": "2026-07-14"}


def screenshots() -> dict:
    t = pd.read_csv(os.path.join(RES, "indicator_shots_2026-07.csv"))
    t = t[t.ticker.isin(COLS)].copy()
    rows = []
    for _, r in t.iterrows():
        e = pd.Timestamp(SHOT_ENTRY[r.shot])
        i = CAL.get_loc(e)
        rec = {}
        for h in (5, 10, 20):
            x = CAL[i + h]
            rec[f"x{h}"] = 100 * ((O.at[x, r.ticker] / O.at[e, r.ticker] - 1) - (P["bench"]["O"].at[x] / P["bench"]["O"].at[e] - 1))
        rows.append(rec)
    t = pd.concat([t.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    t["stateGE1"] = t.state >= 1
    t["rs_pct_in_shot"] = t.groupby("shot").rs_main.rank(pct=True)
    t["proxy_strong"] = (t.rs_pct_in_shot >= 0.70) & (t.fade_gap <= 1)
    rng = np.random.default_rng(13)

    def perm(flag, y):
        d = t.dropna(subset=[y])
        dm = d[y] - d.groupby("shot")[y].transform("mean")
        f = d[flag].to_numpy()
        obs = dm[f].mean() - dm[~f].mean()
        groups = [np.where(d.shot.to_numpy() == s)[0] for s in d.shot.unique()]
        v = dm.to_numpy()
        cnt = 0
        for _ in range(10000):
            g = f.copy()
            for idx in groups:
                g[idx] = rng.permutation(g[idx])
            cnt += abs(v[g].mean() - v[~g].mean()) >= abs(obs) - 1e-12
        raw = d.loc[d[flag], y].mean() - d.loc[~d[flag], y].mean()
        return {"n_in": int(f.sum()), "diff_raw": round(float(raw), 2), "diff_within_shot": round(float(obs), 2),
                "perm_p": round((cnt + 1) / 10001, 4)}

    out = {"rows_in_panel": int(len(t)), "tickers_dropped_outside_panel": ["ENTG", "ORCL", "SOXX", "SWKS", "UMC"],
           "mean_x20_vs_QQQ": round(float(t.x20.mean()), 2)}
    for flag in ("stateGE1", "proxy_strong"):
        out[flag] = {f"h{h}": perm(flag, f"x{h}") for h in (5, 10, 20)}
    out["crosstab_state_vs_proxy_strong"] = pd.crosstab(t.state, t.proxy_strong).to_dict()
    out["state2_fade_gap_min"] = float(t.loc[t.state == 2, "fade_gap"].min())
    out["state_rule"] = ("State는 Score 문턱과 일치: State>=1 행 Score 최소 %.2f, State 0 행 Score 최대 %.2f, "
                         "State 2 최소 %.2f, State 1 최대 %.2f" % (t.loc[t.state >= 1, "score"].min(),
                                                               t.loc[t.state == 0, "score"].max(),
                                                               t.loc[t.state == 2, "score"].min(),
                                                               t.loc[t.state == 1, "score"].max()))
    return out


# ───────────────────────── 데스크 7종목 규칙 ─────────────────────────
def desk_test() -> dict:
    sys.path.insert(0, os.path.join(HERE, "..", "..", "rsi_breakout", "code"))
    import deskx  # noqa: E402

    class DeferEngine(deskx.Engine):
        flag: pd.DataFrame | None = None
        mode: str | None = None
        active_from: str = "0000"

        def blocked(self, key, k):
            if self.flag is None or self.days[k] < self.active_from:
                return False
            tk = self.ticker.get(key, key)
            col = tk if tk in self.flag.columns else harness.ALIASES.get(tk)
            if col is None or col not in self.flag.columns:
                self.unmapped.add(tk)
                return False
            ts = pd.Timestamp(self.days[k - 1])
            if ts not in self.flag.index:
                return False
            return bool(self.flag.at[ts, col])

        def target(self, p, k, session, current):
            ranked = self.ranking(p, k, session)
            if not ranked:
                return [], {}, ranked
            order = [r["key"] for r in ranked]
            keep = [x for x in order[:p.n + p.buffer] if x in current][:p.n]
            held = list(keep)
            cash_names = []
            for x in order:
                if len(keep) + len(cash_names) >= p.n:
                    break
                if x in keep:
                    continue
                if self.mode and self.blocked(x, k):
                    self.events.append((self.days[k], self.ticker.get(x, x)))
                    if self.mode == "cash":
                        cash_names.append(x)
                    continue
                keep.append(x)
                if x not in held:
                    self.new_buys += 1
            full = keep + cash_names
            wmap = {r["key"]: r["w"] for r in ranked}
            tot = sum(wmap[x] for x in full)
            raw = {x: wmap[x] / tot for x in full}
            saved = dict(deskx.config.RULE)
            try:
                deskx.config.RULE.update(stock_cap=p.stock_cap, sector_cap=p.sector_cap, semis_cap=p.semis_cap)
                tgt = deskx.project_caps(raw, self.sector)
            finally:
                deskx.config.RULE.clear()
                deskx.config.RULE.update(saved)
            for x in cash_names:
                tgt.pop(x, None)
            return keep, tgt, ranked

    E = DeferEngine()
    out = {}
    # 첫 재조정(1/5)은 7종목을 처음 사는 날이라 막기가 몰린다 → 1/12부터만 막는 변형을 함께 본다.
    variants = (("base", None, None, "0000"), ("skip_strong", STRONG, "skip", "0000"),
                ("cash_strong", STRONG, "cash", "0000"), ("skip_ex4top30", TOP30, "skip", "0000"),
                ("skip_strong_from_0112", STRONG, "skip", "2026-01-12"),
                ("cash_strong_from_0112", STRONG, "cash", "2026-01-12"),
                ("skip_ex4top30_from_0112", TOP30, "skip", "2026-01-12"))
    for name, flag, mode, act in variants:
        E.flag, E.mode, E.active_from = flag, mode, act
        E.events, E.new_buys, E.unmapped = [], 0, set()
        v, rows = E.run()
        ev = E.evaluate(name, v, rows)
        out[name] = {"front": {k: round(x * 100, 2) for k, x in ev["앞 1/2~5/29"].items()},
                     "back": {k: round(x * 100, 2) for k, x in ev["뒤 6/1~10/9"].items()},
                     "sep_oct": {k: round(x * 100, 2) for k, x in ev["9/1~10/9"].items()},
                     "all": {k: round(x * 100, 2) for k, x in ev["전체"].items()},
                     "monthly": {m: round(x * 100, 2) for m, x in ev["월별"].items()},
                     "blocked_events": len(E.events), "blocked_sample": E.events[:40],
                     "new_buys_done": E.new_buys, "unmapped": sorted(E.unmapped)}
        if name == "base":
            from desk.strategies import run_model
            from desk.validate import codex_caps
            with codex_caps():
                ref = run_model(E.book, start=deskx.START)
            out[name]["max_diff_vs_desk_run_model"] = float(np.max(np.abs(v.to_numpy() - np.asarray(ref.value))))
    return out


SUMMARY_KO = [
    "머리 숫자는 자체 코드로 소수 셋째 자리까지 같다(ex4 IC back h5 −0.113·h20 −0.115, Strong back h5 −1.68 t −3.6, 변동성 중립 −1.77 t −4.8).",
    "미래 정보 없음: t 종가 신호 → t+1 시가 진입, 잘라내기 시험 4개 시점 통과, 보유 PDF는 t일 이전.",
    "ex4·Latent를 매수 순위로 쓰지 않는다는 기각은 버그가 아니다(부호·백분위 집단·구간 날짜 모두 일치).",
    "Strong 회피는 back에서 견고(한 종목 빼기 −1.51~−1.81, 7월 빼기 −2.02, 8월 이후 −1.96, 블록 부트스트랩 t −3.5)하지만 "
    "front에서 고른 것이 아니다(front 변동성 중립 −0.40, 층화 무작위 p 0.40). back 효과는 대부분 4일 상위30% 반전(−1.39)이고 "
    "FadeGap<=1이 더하는 몫은 −0.33(t −0.5). 월별로는 3월부터 음수(3·4·6·7·8·9월), 2024·2025는 0.",
    "이름 주의: 연구의 'Strong'은 지표의 State 2가 아니다. 지표 State는 Score 문턱(>=60, >=72)이고 State 2 행은 전부 FadeGap>1.38이라 "
    "근사 Strong에 하나도 안 들어간다. 7월 촬영에서 지표 State>=1은 h20 +8.1%p(p 0.002)로 오히려 좋았다.",
    "데스크 7종목 규칙 위: 첫 매수일을 빼면 9개월 동안 2번(4/20 TSEM, 5/26 DELL)만 걸리고 back에서는 0번. "
    "front 수익 +205%→+182%, back +4.9%→+5.3%(이월 효과). 데스크 규칙에 넣을 것 없음.",
    "효과는 진입을 하루 늦추면 절반(back −0.94, t −1.7), 5일 미룬 뒤 사면 오히려 +0.91 → '안 산다'가 아니라 '다음 날 시가 추격만 피한다' 수준.",
]


# ───────────────────────── 실행 ─────────────────────────
def main() -> None:
    res = {"family": "verify_indicator_replica", "notes": __doc__}
    chk = {}
    # 채점 도구 일치
    for h in (5, 20):
        a, b = fwd(h), harness.forward(h)
        chk[f"fwd{h}_equals_harness"] = bool(((a - b).abs().max().max() < 1e-12))
        chk[f"window_back_h{h}_equals_harness"] = bool(wdates("back", h).equals(harness.window_dates("back", h)))
    chk["eligible_equals_harness"] = bool((EL == harness.eligible().reindex_like(EL)).all().all())

    # T1 미래 정보: 자료를 잘라 Strong 다시 계산
    nl = {}
    for c in ("2026-03-13", "2026-06-26", "2026-07-22", "2026-09-30"):
        t = pd.Timestamp(c)
        cut = {k: P[k].loc[:t] for k in ("O", "H", "L", "C", "V")}
        cut["bench"] = {k: v.loc[:t] for k, v in P["bench"].items()}
        s2 = signals(cut)
        um2 = harness.universe_mask(cut)
        pr2 = s2["ex4"].where(um2).rank(axis=1, pct=True)
        st2 = (pr2 >= 0.7) & (s2["fg"] <= 1)
        nl[c] = bool((st2.loc[t] == STRONG.loc[t]).all()) and bool(
            np.allclose(s2["fg"].loc[t].fillna(-9), FG.loc[t].fillna(-9)))
    chk["no_lookahead_strong_truncation"] = nl
    res["checks"] = chk
    print("checks", chk)

    # H1 ex4 IC
    h1 = {}
    for w in ("front", "back", "y2025", "y2024"):
        for h in (5, 20):
            h1[f"{w}_h{h}"], _ = ic_cell(EX4, h, w)
    res["H1_ex4_ic"] = h1
    print("H1", h1)

    # H2·H3·H4·H5 집합 초과
    sets = {"top30": TOP30, "bot30": PR4 <= 0.30, "strong": STRONG, "fg_le1": FG <= 1}
    h2 = {}
    for nm, fl in sets.items():
        h2[nm] = {}
        for w in ("front", "back", "y2025", "y2024", "late"):
            for h in (5, 20):
                if w == "late" and h == 20:
                    continue
                h2[nm][f"{w}_h{h}"] = summ(set_series(fl, h, w), h)
                h2[nm][f"{w}_h{h}_volq5"] = summ(set_series(fl, h, w, vol_neutral=True), h)
    res["H2_H4_set_excess"] = h2
    for nm in h2:
        print("SET", nm, {k: (v["x_pp"], v["t_nw"]) for k, v in h2[nm].items()})
    wd = {}
    for w in ("front", "back", "y2025"):
        d = set_series(L30 & F33, 5, w, base_flag=L30 & ~F33)
        wd[w] = summ(d, 5)
        if w == "back":
            wd["back_no_jun22_jul31"] = summ(d[(d.index < "2026-06-22") | (d.index > "2026-07-31")], 5)
            wd["back_nonoverlap"] = nonoverlap_t(d, 5)
    res["H5_watch_minus_unfaded_h5"] = wd
    print("H5", wd)

    # T1 진입 시점 민감도 (Strong)
    tm = {}
    for w in ("front", "back"):
        tm[w] = {"h1": summ(set_series(STRONG, 1, w), 1), "h2": summ(set_series(STRONG, 2, w), 2),
                 "h5_entry_t+1": summ(set_series(STRONG, 5, w), 5),
                 "h5_entry_t+2(하루 늦게)": summ(set_series(STRONG, 5, w, skip=1), 5),
                 "h5_entry_t+6(5일 미룬 뒤 5일)": summ(set_series(STRONG, 5, w, skip=5), 5)}
    # 신호일 종가→다음 날 시가(이 갭은 채점에 안 들어감, 참고)
    gap = (O.shift(-1) / C - 1)
    for w in ("front", "back"):
        d = wdates(w, 1)
        e = EL.loc[d]
        g = gap.loc[d].where(e)
        f = STRONG.loc[d] & e
        tm[w]["overnight_gap_t_close_to_t+1_open"] = summ((g.where(f).mean(axis=1) - g.mean(axis=1)).dropna(), 1)
    res["T1_timing_strong"] = tm
    print("T1", json.dumps(tm, ensure_ascii=False)[:1500])

    # T2 분해: ex4 상위30% 안에서 FG<=1 대 FG>1
    dec = {}
    for w in ("front", "back", "y2025", "y2024"):
        for h in (5, 20):
            a = set_series(TOP30 & (FG <= 1), h, w, base_flag=TOP30 & (FG > 1))
            b = set_series(TOP30 & (FG <= 1), h, w, base_flag=TOP30 & (FG > 1), vol_neutral=True)
            dec[f"{w}_h{h}"] = {"raw": summ(a, h), "volq5": summ(b, h)}
    # 변동성 중립에서 base_flag 비교: set_series는 vol_neutral일 때 base를 base_flag 평균으로 바꾼다(위 정의)
    res["T2_within_top30_fg_le1_minus_gt1"] = dec
    print("T2", {k: (v["raw"]["x_pp"], v["raw"]["t_nw"], v["volq5"]["x_pp"], v["volq5"]["t_nw"]) for k, v in dec.items()})

    # T3 우연: 무작위 집합
    rn = {}
    for w in ("front", "back", "y2025"):
        rn[f"{w}_h5_plain"] = random_null(STRONG, 5, w, stratify=False)
        rn[f"{w}_h5_vol_stratified"] = random_null(STRONG, 5, w, stratify=True)
    d_back = set_series(STRONG, 5, "back")
    d_front = set_series(STRONG, 5, "front")
    rn["nonoverlap_back_h5"] = nonoverlap_t(d_back, 5)
    rn["nonoverlap_front_h5"] = nonoverlap_t(d_front, 5)
    rn["nonoverlap_back_h5_volq5"] = nonoverlap_t(set_series(STRONG, 5, "back", vol_neutral=True), 5)
    bb = {"strong_back_h5": block_boot_t(d_back), "strong_front_h5": block_boot_t(d_front),
          "strong_back_h5_volq5": block_boot_t(set_series(STRONG, 5, "back", vol_neutral=True)),
          "strong_front_h5_volq5": block_boot_t(set_series(STRONG, 5, "front", vol_neutral=True)),
          "top30_back_h5": block_boot_t(set_series(TOP30, 5, "back")),
          "fg_le1_front_h5": block_boot_t(set_series(FG <= 1, 5, "front")),
          "fg_le1_back_h5": block_boot_t(set_series(FG <= 1, 5, "back"))}
    for w, h in (("back", 5), ("back", 20), ("front", 20)):
        _, ic = ic_cell(EX4, h, w)
        bb[f"ex4_ic_{w}_h{h}"] = block_boot_t(ic, block=max(10, h))
        bb[f"ex4_ic_{w}_h{h}"]["nonoverlap"] = nonoverlap_t(ic, h)
    rn["block_bootstrap"] = bb
    from math import erf, sqrt
    phi = lambda z: 0.5 * (1 + erf(z / sqrt(2)))
    rn["family_wise_note"] = {
        "cells_reported_by_study": 480,
        "P(max|z|>=3.6 among 480 independent null cells)": round(min(1.0, 480 * 2 * (1 - phi(3.6))), 3),
        "P(max|z|>=4.8 among 480)": round(480 * 2 * (1 - phi(4.8)), 5),
        "bonferroni_z_480_two_sided_0.05": 3.88}
    res["T3_chance"] = rn
    print("T3", json.dumps(rn, ensure_ascii=False))

    # T4 쏠림 (Strong back h5)
    cc = {}
    base = set_series(STRONG, 5, "back")
    loo = {}
    members = [c for c in COLS if STRONG.loc[wdates("back", 5), c].any()]
    for c in members:
        loo[c] = float(set_series(STRONG, 5, "back", cols=[x for x in COLS if x != c]).mean()) * 100
    s = pd.Series(loo).sort_values()
    cc["leave_one_out_min"] = [s.index[0], round(s.iloc[0], 3)]
    cc["leave_one_out_max"] = [s.index[-1], round(s.iloc[-1], 3)]
    cc["n_stocks_ever_in_set_back"] = len(members)
    # 종목 기여: 날짜별 (r_i − 전체평균)/집합 크기 의 합 / 날짜 수
    d = wdates("back", 5)
    e = EL.loc[d]
    r = fwd(5).loc[d].where(e)
    f = STRONG.loc[d] & e
    contrib = r.sub(r.mean(axis=1), axis=0).where(f).div(f.sum(axis=1), axis=0).sum() / len(base) * 100
    cc["top_negative_contributors_pp"] = {k: round(v, 3) for k, v in contrib.sort_values().head(6).items()}
    cc["top_positive_contributors_pp"] = {k: round(v, 3) for k, v in contrib.sort_values().tail(4).items()}
    cc["sum_contrib_check_pp"] = round(float(contrib.sum()), 3)
    cc["by_month_back_h5"] = {str(k): [round(float(v.mean()) * 100, 2), int(len(v))]
                              for k, v in base.groupby(base.index.to_period("M"))}
    fr_ = set_series(STRONG, 5, "front")
    cc["by_month_front_h5"] = {str(k): [round(float(v.mean()) * 100, 2), int(len(v))]
                               for k, v in fr_.groupby(fr_.index.to_period("M"))}
    nj = base[(base.index < "2026-06-22") | (base.index > "2026-07-31")]
    cc["no_jun22_jul31"] = summ(nj, 5)
    cc["only_aug1_onward"] = summ(base[base.index >= "2026-08-01"], 5)
    cc["drop_5_most_negative_days"] = round(float(base.drop(base.sort_values().index[:5]).mean()) * 100, 3)
    cc["drop_10_most_negative_days"] = round(float(base.drop(base.sort_values().index[:10]).mean()) * 100, 3)
    res["T4_concentration_strong_back_h5"] = cc
    print("T4", json.dumps(cc, ensure_ascii=False))

    # T5 문턱 민감도
    sens = {}
    for fgc in (0.5, 1.0, 2.0):
        for top in (0.6, 0.7, 0.8):
            fl = (PR4 >= top) & (FG <= fgc)
            sens[f"fg<={fgc}_ex4top{int(round((1 - top) * 100))}"] = {
                w: [summ(set_series(fl, 5, w), 5)["x_pp"], summ(set_series(fl, 5, w), 5)["t_nw"],
                    summ(set_series(fl, 5, w, vol_neutral=True), 5)["x_pp"]] for w in ("front", "back", "y2025")}
    res["T5_threshold_grid_h5[x,t,volq5_x]"] = sens
    print("T5", json.dumps(sens))

    # T6 비용
    res["T6_cost"] = {
        "avoid_rule": "다른 보유 종목을 대신 사므로 거래 횟수 불변 → 추가 비용 0",
        "long_short_back_h5_after_0.2pp": round(h2["strong"]["back_h5"]["x_pp"] + 0.2, 3),
        "long_short_front_h5_after_0.2pp": round(h2["strong"]["front_h5"]["x_pp"] + 0.2, 3)}

    # T7 스크린샷
    res["T7_screenshots"] = screenshots()
    print("T7", json.dumps(res["T7_screenshots"], ensure_ascii=False, default=str))

    # T8 데스크
    res["T8_desk"] = desk_test()
    for k, v in res["T8_desk"].items():
        print("DESK", k, v["front"], v["back"], v["sep_oct"], v["all"], "blocked", v["blocked_events"], v.get("unmapped"))

    res["verdict"] = "partially_confirmed"
    res["summary_ko"] = SUMMARY_KO
    path = os.path.join(RES, "verify_indicator_replica.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=str)
    print(path)


if __name__ == "__main__":
    main()
