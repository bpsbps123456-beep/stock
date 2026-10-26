"""갈래 c2trade: BRIEF2 시험 3(매매)과 채택 기준 2~4.

기본 매매는 ZZA k3 상태(swing.zigzag, 3×ATR20). 체결은 확정 다음 날 시가(swing.sim_period와 같은 정의), 비용 0·0.05%.
- (c) 메타 규칙(고르기 없음): 날 d 종가까지의 M5_w(최근 w거래일 주식 평균 '추세 − 뒤집기' 일 손익 합, 종가→종가)가 > 0이면
  ZZA 상태, ≤ 0이면 1 − 상태. w 40(사전 고정), 20·60(민감도). M5_40은 events.py와 같은 식으로 다시 계산해 사건표와 맞춘다.
- (a) 거름: 매수 확정 중 '이어짐'(y1 > 0 예측)인 것만 산다. 아니면 다음 매수 확정까지 현금.
- (b) 장세 전환: 확정마다 이어짐이면 추세(매수→보유, 매도→현금), V자면 뒤집기(매수→현금, 매도→보유).
  - 로지스틱(L2, C=1, 18개를 P1 평균·표준편차로 표준화, 결측 0) 예측확률이 P1 같은 type 사건 중앙 이상이면 이어짐.
  - AND 규칙: P1 3분위 경계(같은 type, x ≤ q1 / ≤ q2 / 그 위), 두 단서 × 3² = 1,377개 중 P1 사건 50개·날짜 10일 이상에서
    y1 평균 1위가 성립하면 이어짐. 매수·매도 type을 따로 학습한다. 사건표에 없는 확정(c < 260 등)은 추세로 둔다.
- 귀무
  - 메타: M5 부호 열을 공통 k만큼 원형 이동. ① 구간 안(15 ≤ k ≤ D−15, BRIEF와 같은 틀) ② 2023-01-03 전날~10-22 전체 열(15 ≤ k ≤ L−15).
  - 거름·전환: ① 날짜 블록 순열 200회 — 구간·type마다 사건 날짜를 섞어 각 사건이 받은 날짜의 같은 종목 사건 단서(없으면 그 날짜
    사건에서 무작위)를 가져온 뒤 P1 학습부터 같은 절차를 반복. ② 원형 이동 — P1·P2 거래일 공통 k(15~86)로 날짜를 밀고
    (그날 같은 type 사건이 없으면 다음 사건 날짜) 같은 절차 반복.
→ results/c2trade_actual.csv, c2trade_null.npz, c2trade_models.json, c2trade_select.csv
"""
from __future__ import annotations

import json
import time
from datetime import date

import numpy as np
import pandas as pd
from numba import njit
from sklearn.linear_model import LogisticRegression

import swing
from events import ETF, FEATURES

OUT = swing.HERE / "results"
COSTS = (0.0, 0.0005)
PERS = ("P1", "P2", "Y25", "Y24", "Y23")
NULL_PERS_EV = ("P1", "P2", "Y25")
WINS = (20, 40, 60)
WARM = 60
NPERM = 200
SEED = 20261025
MODEL_STRATS = ("lg_a", "lg_b", "and_a", "and_b")


# ---------------------------------------------------------------- 매매 계산
@njit(cache=True)
def sim_stats(o, c, st, j0, j1, cost):
    """swing.sim_period + swing.static_bh(같은 비중 x)와 같은 값. 반환: 수익, 낙폭, x, 진입, 같은 비중 수익, 같은 비중 낙폭."""
    n = j1 - j0 + 1
    v = 1.0; pk = 1.0; mdd = 0.0; hold = 0.0; ent = 0
    prev = st[j0 - 1] if j0 > 0 else 0
    for t in range(n):
        j = j0 + t
        pos = st[j - 1] if j > 0 else 0
        r = (o[j + 1] / o[j] - 1.0) if (t < n - 1) else (c[j] / o[j] - 1.0)
        if pos != prev and t > 0:
            v *= 1.0 - cost
            if pos == 1:
                ent += 1
        v *= 1.0 + pos * r
        prev = pos
        hold += pos
        if v > pk:
            pk = v
        if v / pk - 1.0 < mdd:
            mdd = v / pk - 1.0
    x = hold / n
    p0 = o[j0]; pk2 = 1.0; m2 = 0.0; e = 1.0
    for t in range(n + 1):
        px = o[j0 + t] if t < n else c[j1]
        e = 1.0 + x * (px / p0 - 1.0)
        if e > pk2:
            pk2 = e
        if e / pk2 - 1.0 < m2:
            m2 = e / pk2 - 1.0
    return v - 1.0, mdd, x, float(ent), e - 1.0, m2


@njit(cache=True)
def meta_null(o, c, st, sgn, mp, base, L, ks, j0, j1, cost):
    """M5 부호 열을 k만큼 원형 이동한 메타 상태의 매매. 날 j의 부호 = sgn[base + ((mp[j] − base − k) mod L)] (np.roll과 같은 방향)."""
    nk = len(ks)
    res = np.empty((nk, 6))
    a = max(j0 - 1, 0)
    tmp = st.copy()
    for q in range(nk):
        k = ks[q]
        for j in range(a, j1 + 1):
            idx = base + ((mp[j] - base - k) % L)
            tmp[j] = st[j] if sgn[idx] > 0 else 1 - st[j]
        r = sim_stats(o, c, tmp, j0, j1, cost)
        for z in range(6):
            res[q, z] = r[z]
    return res


@njit(cache=True)
def build_state(n, pc, pt, flag, mode):
    """확정 사건으로 상태 만들기. flag 1 이어짐, 0 V자, −1 예측 없음(추세). mode 0 거름(a), 1 장세 전환(b)."""
    st = np.zeros(n, np.int64)
    cur = 0
    e = 0
    m = len(pc)
    for i in range(n):
        while e < m and pc[e] == i:
            cont = flag[e] != 0
            if pt[e] == 1:
                cur = 1 if cont else 0
            else:
                cur = 0 if (mode == 0 or cont) else 1
            e += 1
        st[i] = cur
    return st


# ---------------------------------------------------------------- 사건 모형
def fit_logit(X1, y1):
    mu = np.nanmean(X1, axis=0)
    sd = np.nanstd(X1, axis=0)
    sd[~(sd > 0)] = 1.0
    Z = np.nan_to_num((X1 - mu) / sd)
    m = LogisticRegression(C=1.0, max_iter=2000).fit(Z, (y1 > 0).astype(int))
    p1 = m.predict_proba(Z)[:, 1]
    return dict(mu=mu, sd=sd, coef=m.coef_[0], icpt=float(m.intercept_[0]), thr=float(np.median(p1)))


def pred_logit(md, X):
    Z = np.nan_to_num((X - md["mu"]) / md["sd"])
    p = 1 / (1 + np.exp(-(Z @ md["coef"] + md["icpt"])))
    return p >= md["thr"]


def tert_bins(X, cuts):
    B = np.full(X.shape, -1, np.int64)
    for f in range(X.shape[1]):
        x = X[:, f]
        ok = ~np.isnan(x)
        B[ok, f] = (x[ok] > cuts[f, 0]).astype(int) + (x[ok] > cuts[f, 1]).astype(int)
    return B


def fit_and(X1, y1, d1, min_n=50, min_d=10):
    cuts = np.array([np.nanquantile(X1[:, f], [1 / 3, 2 / 3]) for f in range(X1.shape[1])])
    B = tert_bins(X1, cuts)
    E = np.stack([B[:, f] == b for f in range(X1.shape[1]) for b in range(3)], axis=1)
    _, inv = np.unique(d1, return_inverse=True)
    best = None
    nf = X1.shape[1]
    for i in range(3 * nf):
        for j in range(i + 1, 3 * nf):
            if i // 3 == j // 3:
                continue
            m = E[:, i] & E[:, j]
            cnt = int(m.sum())
            if cnt < min_n:
                continue
            nd = len(np.unique(inv[m]))
            if nd < min_d:
                continue
            mu = float(y1[m].mean())
            if best is None or mu > best[0]:
                best = (mu, i, j, cnt, nd)
    if best is None:
        return None
    mu, i, j, cnt, nd = best
    return dict(cuts=cuts, f1=i // 3, b1=i % 3, f2=j // 3, b2=j % 3, p1_mean=mu, p1_n=cnt, p1_dates=nd,
                p1_rest=float(y1[~(E[:, i] & E[:, j])].mean()))


def pred_and(md, X):
    if md is None:
        return np.zeros(len(X), bool)
    B = tert_bins(X, md["cuts"])
    return (B[:, md["f1"]] == md["b1"]) & (B[:, md["f2"]] == md["b2"])


def fit_predict(ev, X):
    """P1(같은 type)로 학습해 모든 사건의 '이어짐' 예측. 반환: {lg, and}: bool 배열, 모형."""
    out = {"lg": np.zeros(len(ev), bool), "and": np.zeros(len(ev), bool)}
    models = {}
    y = ev.y1.values
    for t in (1, -1):
        sel_t = ev.type.values == t
        tr = sel_t & (ev.period.values == "P1")
        lg = fit_logit(X[tr], y[tr])
        an = fit_and(X[tr], y[tr], ev.date.values[tr])
        out["lg"][sel_t] = pred_logit(lg, X[sel_t])
        out["and"][sel_t] = pred_and(an, X[sel_t])
        models[t] = (lg, an)
    return out, models


# ---------------------------------------------------------------- 귀무 기증 사건
class Donor:
    def __init__(self, ev, cal_per):
        self.ev = ev
        self.sym = ev.sym.values
        self.groups = {}
        for (p, t), g in ev.groupby(["period", "type"]):
            by_date = {d: gg.index.values for d, gg in g.groupby("date")}
            by_ds = {(d, s): i for d, s, i in zip(g.date.values, g.sym.values, g.index.values)}
            self.groups[(p, t)] = (sorted(by_date), by_date, by_ds)
        self.cal_per = cal_per  # 구간 거래일(SPY)

    def _fill(self, donor, d, dd, by_date, by_ds, rng):
        cands = by_date[dd]
        for e in by_date[d]:
            i = by_ds.get((dd, self.sym[e]))
            donor[e] = i if i is not None else cands[rng.integers(len(cands))]

    def perm(self, rng):
        donor = np.arange(len(self.ev))
        for (p, t), (dates, by_date, by_ds) in self.groups.items():
            pm = rng.permutation(len(dates))
            for i, d in enumerate(dates):
                self._fill(donor, d, dates[pm[i]], by_date, by_ds, rng)
        return donor

    def circ(self, k, rng, pers=("P1", "P2")):
        donor = np.arange(len(self.ev))
        for (p, t), (dates, by_date, by_ds) in self.groups.items():
            if p not in pers:
                continue
            cp = self.cal_per[p]
            D = len(cp)
            pos = {d: i for i, d in enumerate(cp)}
            has = np.array([d in by_date for d in cp])
            for d in dates:
                q = (pos[d] + k) % D
                while not has[q]:
                    q = (q + 1) % D
                self._fill(donor, d, cp[q], by_date, by_ds, rng)
        return donor


# ---------------------------------------------------------------- 주 계산
def m5_series(stocks, D, info, cal, w, minp):
    ix = {d: i for i, d in enumerate(cal)}
    tmr = np.full((len(cal), len(stocks)), np.nan)
    for j, s in enumerate(stocks):
        g = D[s]
        st = info[s]["st"]
        C = g.C.values.astype(float)
        rows = np.array([ix[d] for d in g.date])
        r = np.r_[np.nan, C[1:] / C[:-1] - 1]
        pos = np.r_[np.nan, st[:-1]].astype(float)
        tmr[rows, j] = (2 * pos - 1) * r
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean = np.nanmean(tmr, axis=1)
    return pd.Series(mean).rolling(w, min_periods=minp).sum().values


def main():
    t0 = time.time()
    D = swing.load()
    syms = sorted(D)
    stocks = sorted(set(D) - ETF)
    ev = pd.read_parquet(OUT / "events_ZZA3.parquet").reset_index(drop=True)
    X = ev[FEATURES].values.astype(float)
    key = {(s, int(c)): i for i, (s, c) in enumerate(zip(ev.sym.values, ev.conf_idx.values))}
    info = {}
    for s in syms:
        g = D[s]
        O, H, L, C = (g[k].values.astype(np.float64) for k in ("O", "H", "L", "C"))
        st, pi, pc, pt = swing.zigzag(C, 0.0, 3.0 * swing.atr(H, L, C, 20))
        assert np.array_equal(st, swing.states(g)["ZZA_k3.0"]) if s in ("MU", "SPY", "BE") else True
        evrow = np.array([key.get((s, int(c)), -1) for c in pc], np.int64)
        info[s] = dict(o=O, c=C, st=st, pc=pc.astype(np.int64), pt=pt.astype(np.int64), evrow=evrow, n=len(C))
    # 사건표와 확정 대조
    matched = sum(int((info[s]["evrow"] >= 0).sum()) for s in stocks)
    assert matched == len(ev), (matched, len(ev))
    for s in stocks:
        er = info[s]["evrow"]; ok = er >= 0
        assert np.array_equal(ev.type.values[er[ok]], info[s]["pt"][ok])
    # 달력·M5
    cal = sorted(set().union(*[set(D[s].date) for s in stocks]))
    ix = {d: i for i, d in enumerate(cal)}
    M5 = {w: m5_series(stocks, D, info, cal, w, int(round(w * 0.75))) for w in WINS}
    for s in syms:
        rows = np.array([ix.get(d, -1) for d in D[s].date])
        assert (rows >= 0).all(), s
        info[s]["rows"] = rows.astype(np.int64)
    ev_ri = np.array([info[s]["rows"][c] for s, c in zip(ev.sym.values, ev.conf_idx.values)])
    m5_err = float(np.nanmax(np.abs(M5[40][ev_ri] - ev.M5.values)))
    assert m5_err < 1e-9, m5_err
    sgn = {w: (np.nan_to_num(M5[w], nan=1.0) > 0).astype(np.int64) for w in WINS}   # 결측은 추세(구간에는 없음)
    # 구간 포함
    ref = D["SPY"]
    Dper = {p: len(swing.period_index(ref, p)) for p in PERS}
    cal_per = {p: list(ref.date.values[swing.period_index(ref, p)]) for p in PERS}
    incl = {p: [s for s in syms if (lambda idx: len(idx) == Dper[p] and idx[0] >= WARM and D[s].date.iloc[-1] >= swing.PERIODS[p][1])(swing.period_index(D[s], p))] for p in PERS}
    jj = {(s, p): (int(swing.period_index(D[s], p)[0]), int(swing.period_index(D[s], p)[-1])) for p in PERS for s in incl[p]}
    # 자체 확인: sim_stats = sim_period + static_bh, build_state(모두 이어짐) = ZZA
    worst = 0.0
    rng = np.random.default_rng(0)
    for _ in range(60):
        p = PERS[rng.integers(5)]; s = incl[p][rng.integers(len(incl[p]))]
        I = info[s]; j0, j1 = jj[(s, p)]
        st = I["st"] if rng.random() < 0.5 else 1 - I["st"]
        cost = COSTS[rng.integers(2)]
        a = swing.sim_period(I["o"], I["c"], st, j0, j1, cost); b = swing.static_bh(I["o"], I["c"], j0, j1, a[2])
        r = sim_stats(I["o"], I["c"], st, j0, j1, cost)
        worst = max(worst, max(abs(r[0] - a[0]), abs(r[1] - a[1]), abs(r[2] - a[2]), abs(r[3] - a[3]), abs(r[4] - b[0]), abs(r[5] - b[1])))
    assert worst < 1e-10, worst
    for s in stocks:
        I = info[s]
        ones = np.ones(len(I["pc"]), np.int64)
        assert np.array_equal(build_state(I["n"], I["pc"], I["pt"], ones, 0), I["st"])
        assert np.array_equal(build_state(I["n"], I["pc"], I["pt"], ones, 1), I["st"])
    # meta_null k=0 = 실제 메타
    s = "MU"; I = info[s]; j0, j1 = jj[(s, "P2")]
    msig = sgn[40][I["rows"]]
    meta_st = np.where(msig > 0, I["st"], 1 - I["st"])
    r0 = meta_null(I["o"], I["c"], I["st"], msig, np.arange(I["n"], dtype=np.int64), j0 - 1, j1 - j0 + 2, np.array([0], np.int64), j0, j1, 0.0)[0]
    r1 = sim_stats(I["o"], I["c"], meta_st, j0, j1, 0.0)
    assert max(abs(r0[z] - r1[z]) for z in range(6)) < 1e-12
    print("자체 확인 통과: sim 오차", worst, "M5 재계산 오차", m5_err, f"{time.time() - t0:.0f}초")

    # ---------------- 실제 예측
    pred, models = fit_predict(ev, X)
    model_json = {}
    for t, (lg, an) in models.items():
        model_json["buy" if t == 1 else "sell"] = dict(
            logit_coef=dict(zip(FEATURES, np.round(lg["coef"], 4).tolist())), logit_thr=lg["thr"],
            and_rule=None if an is None else dict(f1=FEATURES[an["f1"]], b1=an["b1"], f2=FEATURES[an["f2"]], b2=an["b2"],
                                                   p1_mean=an["p1_mean"], p1_rest=an["p1_rest"], p1_n=an["p1_n"], p1_dates=an["p1_dates"],
                                                   cut1=an["cuts"][an["f1"]].tolist(), cut2=an["cuts"][an["f2"]].tolist()))

    def flags_for(s, pr):
        er = info[s]["evrow"]
        return np.where(er >= 0, pr[np.maximum(er, 0)].astype(np.int64), -1)

    def model_states(s, pr):
        I = info[s]
        out = {}
        for nm, src, mode in (("lg_a", "lg", 0), ("lg_b", "lg", 1), ("and_a", "and", 0), ("and_b", "and", 1)):
            out[nm] = build_state(I["n"], I["pc"], I["pt"], flags_for(s, pr[src]), mode)
        return out

    # 선택 통계(기준 1 참고): 구간·type별 이어짐 대 아님 평균 y1
    def select_stats(pr):
        rows = []
        for (p, t), g in ev.groupby(["period", "type"]):
            ii = g.index.values
            for nm in ("lg", "and"):
                m = pr[nm][ii]
                y = g.y1.values
                rows.append(dict(per=p, type=t, model=nm, n=len(ii), n_sel=int(m.sum()), dates=g.date.nunique(),
                                 dates_sel=g.date.values[m].size and len(set(g.date.values[m])),
                                 y_sel=float(y[m].mean()) if m.any() else np.nan, y_rest=float(y[~m].mean()) if (~m).any() else np.nan))
        r = pd.DataFrame(rows)
        r["diff"] = r.y_sel - r.y_rest
        return r

    sel_real = select_stats(pred)

    # ---------------- 실제 매매
    rows = []
    for p in PERS:
        for s in incl[p]:
            I = info[s]; j0, j1 = jj[(s, p)]
            sts = {"zz": I["st"], "flip": 1 - I["st"]}
            msig_all = {w: sgn[w][I["rows"]] for w in WINS}
            for w in WINS:
                sts[f"meta{w}"] = np.where(msig_all[w] > 0, I["st"], 1 - I["st"])
            if s in stocks:
                sts.update(model_states(s, pred))
            flip_days = float(1 - msig_all[40][j0 - 1:j1].mean())
            for nm, st in sts.items():
                for cost in COSTS:
                    r = sim_stats(I["o"], I["c"], st, j0, j1, cost)
                    _, _, bh, bhd = swing.static_bh(I["o"], I["c"], j0, j1, 1.0)
                    rows.append((p, s, s in stocks, nm, cost, *r, bh, bhd, flip_days))
    act = pd.DataFrame(rows, columns=["per", "sym", "stock", "strat", "cost", "ret", "mdd", "x", "ent", "sw_ret", "sw_mdd", "bh_ret", "bh_mdd", "m5_flip_days"])
    zz = act[act.strat == "zz"].set_index(["per", "sym", "cost"])
    act["zz_ret"] = zz.ret.reindex(pd.MultiIndex.from_frame(act[["per", "sym", "cost"]])).values
    act["zz_mdd"] = zz.mdd.reindex(pd.MultiIndex.from_frame(act[["per", "sym", "cost"]])).values
    act["ex_sw"] = act.ret - act.sw_ret
    act["ex_zz"] = act.ret - act.zz_ret
    act["ex_bh"] = act.ret - act.bh_ret
    act["dd_sw"] = act.mdd - act.sw_mdd
    act.to_csv(OUT / "c2trade_actual.csv", index=False)
    print("실제 매매", len(act), "행", f"{time.time() - t0:.0f}초")

    # ---------------- 메타 귀무(종목 중앙만 저장: 주식·전체 두 묶음)
    null = {}
    zzr = {(p, s, cost): (r.ret, r.mdd) for p, s, cost, r in zip(zz.index.get_level_values(0), zz.index.get_level_values(1), zz.index.get_level_values(2), zz.itertuples())}

    def summ(arr, ss, p):
        """arr (종목, 비용, ..., 6) → 묶음별 종목 중앙 {ex_sw, ex_zz, dd_sw, ret, mdd, x}: (묶음, 비용, ...)."""
        zr = np.array([[zzr[(p, s, cost)][0] for cost in COSTS] for s in ss])
        sh = (len(ss), len(COSTS)) + (1,) * (arr.ndim - 3)
        mets = dict(ex_sw=arr[..., 0] - arr[..., 4], ex_zz=arr[..., 0] - zr.reshape(sh), dd_sw=arr[..., 1] - arr[..., 5],
                    ret=arr[..., 0], mdd=arr[..., 1], x=arr[..., 2])
        isst = np.array([s in stocks for s in ss])
        return {m: np.stack([np.nanmedian(v[isst], axis=0), np.nanmedian(v, axis=0)]) for m, v in mets.items()}

    span0 = ix[cal_per["Y23"][0]] - 1
    Lspan = len(cal) - span0
    ks_full = np.arange(15, Lspan - 15 + 1, dtype=np.int64)
    for p in PERS:
        ks_in = np.arange(15, Dper[p] - 15 + 1, dtype=np.int64)
        arr_in = np.full((len(incl[p]), len(COSTS), len(WINS), len(ks_in), 6), np.nan)
        arr_fu = np.full((len(incl[p]), len(COSTS), len(WINS), len(ks_full), 6), np.nan)
        for si, s in enumerate(incl[p]):
            I = info[s]; j0, j1 = jj[(s, p)]
            for wi, w in enumerate(WINS):
                ss = sgn[w][I["rows"]]
                for ci, cost in enumerate(COSTS):
                    arr_in[si, ci, wi] = meta_null(I["o"], I["c"], I["st"], ss, np.arange(I["n"], dtype=np.int64), j0 - 1, j1 - j0 + 2, ks_in, j0, j1, cost)
                    arr_fu[si, ci, wi] = meta_null(I["o"], I["c"], I["st"], sgn[w], I["rows"], span0, Lspan, ks_full, j0, j1, cost)
        for m, v in summ(arr_in, incl[p], p).items():
            null[f"meta_in_{p}_{m}"] = v          # (묶음 주식/전체, 비용, 창, k)
        for m, v in summ(arr_fu, incl[p], p).items():
            null[f"meta_full_{p}_{m}"] = v
        null[f"ks_in_{p}"] = ks_in
    null["ks_full"] = ks_full
    print("메타 귀무", f"{time.time() - t0:.0f}초")

    # ---------------- 거름·전환 귀무
    donor = Donor(ev, cal_per)

    def run_null(pr, pers):
        res = {p: np.full((len([s for s in incl[p] if s in stocks]), len(COSTS), len(MODEL_STRATS), 6), np.nan) for p in pers}
        pos = {p: {s: i for i, s in enumerate([s for s in incl[p] if s in stocks])} for p in pers}
        for s in stocks:
            if not any(s in pos[p] for p in pers):
                continue
            I = info[s]
            ms = model_states(s, pr)
            for p in pers:
                if s not in pos[p]:
                    continue
                j0, j1 = jj[(s, p)]
                for mi, nm in enumerate(MODEL_STRATS):
                    for ci, cost in enumerate(COSTS):
                        res[p][pos[p][s], ci, mi] = sim_stats(I["o"], I["c"], ms[nm], j0, j1, cost)
        return res

    rng = np.random.default_rng(SEED)
    perm_res = {p: [] for p in NULL_PERS_EV}
    perm_sel = []
    for rep in range(NPERM):
        dn = donor.perm(rng)
        pr, _ = fit_predict(ev, X[dn])
        r = run_null(pr, NULL_PERS_EV)
        for p in NULL_PERS_EV:
            perm_res[p].append(r[p])
        st_ = select_stats(pr); st_["rep"] = rep
        perm_sel.append(st_)
        if rep % 50 == 49:
            print("순열", rep + 1, f"{time.time() - t0:.0f}초")
    for p in NULL_PERS_EV:
        ss = [s for s in incl[p] if s in stocks]
        for m, v in summ(np.stack(perm_res[p], axis=-2), ss, p).items():   # (종목, 비용, 전략, rep, 6)
            null[f"perm_{p}_{m}"] = v[0]                                   # 주식 묶음만: (비용, 전략, rep)
    ks_c = np.arange(15, min(Dper["P1"], Dper["P2"]) - 15 + 1, dtype=np.int64)
    circ_res = {p: [] for p in ("P1", "P2")}
    circ_sel = []
    for k in ks_c:
        dn = donor.circ(int(k), np.random.default_rng(SEED + int(k)))
        pr, _ = fit_predict(ev, X[dn])
        r = run_null(pr, ("P1", "P2"))
        for p in ("P1", "P2"):
            circ_res[p].append(r[p])
        st_ = select_stats(pr); st_["k"] = int(k)
        circ_sel.append(st_)
    for p in ("P1", "P2"):
        ss = [s for s in incl[p] if s in stocks]
        for m, v in summ(np.stack(circ_res[p], axis=-2), ss, p).items():
            null[f"circ_{p}_{m}"] = v[0]
    null["ks_circ"] = ks_c
    print("원형 귀무", f"{time.time() - t0:.0f}초")

    np.savez_compressed(OUT / "c2trade_null.npz", costs=np.array(COSTS), wins=np.array(WINS), model_strats=np.array(MODEL_STRATS), **null)
    sel_all = pd.concat([sel_real.assign(src="real"), pd.concat(perm_sel).assign(src="perm"), pd.concat(circ_sel).assign(src="circ")])
    sel_all.to_csv(OUT / "c2trade_select.csv", index=False)
    missing = {p: int(sum(((info[s]["evrow"] < 0) & np.isin(info[s]["pc"], np.arange(*jj[(s, p)]))).sum() for s in incl[p] if s in stocks)) for p in PERS}
    meta_days = {p: {w: float(np.mean(sgn[w][ix[cal_per[p][0]] - 1: ix[cal_per[p][-1]]] > 0)) for w in WINS} for p in PERS}
    p1_leak = int(((ev.period == "P1") & (pd.to_datetime(ev.exit_date) > pd.Timestamp(date(2026, 5, 29)))).sum())
    (OUT / "c2trade_models.json").write_text(json.dumps(dict(
        models=model_json, n_incl={p: (len(incl[p]), len([s for s in incl[p] if s in stocks])) for p in PERS}, D=Dper,
        trend_mode_share=meta_days, missing_pred_confirms_in_period=missing, p1_events_exit_after_p1=p1_leak,
        n_perm=NPERM, ks_circ=[int(ks_c[0]), int(ks_c[-1])], ks_full=[int(ks_full[0]), int(ks_full[-1])], span_len=int(Lspan),
        self_check=dict(sim=worst, m5=m5_err)), ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("끝", f"{time.time() - t0:.0f}초")


if __name__ == "__main__":
    main()
