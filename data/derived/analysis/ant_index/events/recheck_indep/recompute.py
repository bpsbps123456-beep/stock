"""부록 13 사건 연구 독립 재계산 (event_study.py 를 읽지 않고 작성).

입력: events/results/features_all.parquet, features_all_meta.json, events.json(10/22 종가 판정), prices_close.csv
정의(부록 13): 직전 20창 z, 하나 빼고 맞히기(LOO) 문턱, 표지 섞기 1,000번, 앞섬(사건 전 1~5창이 직전 60창 상위·하위 10%).

이 재계산에서 스스로 정한 세부(원 코드와 다를 수 있음):
- 창 마스크: judge_ok False 창은 결측. realtime_comparable False 특징은 실시간 창(realtime_window) 결측.
- 창 종류 보정(주판 K): 시장·특징마다 kind_class(평일/평일 아님) 평균을 빼고 전체 평균을 더함(전 표본 수준 보정).
  민감도 R: 보정 없음. 민감도 V: 보정 K + 결측 창을 건너뛴 직전 유효값 20개(z)·60개(백분위)가 모두 있어야 함.
  실행: py -B recompute.py K R  /  py -B recompute.py V  → recheck_loo_<tag>.csv, recheck_lead_<tag>.csv
- z: 직전 20개 창 위치(현재 제외)의 평균·표준편차(ddof=1), 유효 15개 이상.
- 단위: 사건 = 확정 꺾임점, 가짜 = 묶음(run)마다 대표 1개(바닥은 최저 종가, 고점은 최고 종가, 동률은 앞 날).
- LOO 규칙 P: 학습 단위로 방향(사건 평균 z − 가짜 평균 z 부호)을 정하고, 그 방향에서 학습 BA가 가장 큰 문턱(인접 값 중점);
  동률 문턱은 가운데(아래 중앙값) 것을 쓴다. 규칙 J: 방향도 BA 최대로 함께 고름(동률이면 평균 차 부호).
- p = (1 + #{섞기 BA >= 관측 BA}) / 1001, 섞기는 그 칸에 값이 있는 단위끼리 표지를 섞음.
- 가짜 날 전부 BA: 사건은 하나씩, 가짜 날은 그 묶음째 빼고 학습(관측값만).
- 앞섬: 보정값 x_t가 직전 60개 창 위치(유효 30개 이상) 대비 백분위 >= 0.9(상위) / <= 0.1(하위). 사건 전 1~5창 중 하나라도 닿으면 1.
  p(가짜 섞기) 한쪽(사건 비율 − 가짜 비율 >= 관측). 위약 = 사건마다 (5일, 20일 수익률) 표준화 거리 최근접 20일
  (후보: 기간 안 거래일, 모든 꺾임점·끝 극값 ±3거래일 밖), 이항 한쪽 p.
"""
import json, sys, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = Path(__file__).resolve().parent
NPERM = 1000
ZMIN = 15
PCTMIN = 30
VALID_ONLY = False

df_all = pd.read_parquet(RES / "features_all.parquet")
meta = json.load(open(RES / "features_all_meta.json", encoding="utf-8"))["features"]
ev = json.load(open(RES / "events.json", encoding="utf-8"))
prices = pd.read_csv(RES / "prices_close.csv", parse_dates=["date"])

FEATS = [k for k, v in meta.items() if v.get("event_study")]
assert len(FEATS) == 46, len(FEATS)
MARKET = {"KOSPI": "kr", "SKHYNIX": "kr", "SOXX": "us"}


def prep_market(mk, kind_adj=True):
    d = df_all[df_all.market == mk].sort_values("D").reset_index(drop=True).copy()
    X = pd.DataFrame(index=d.index)
    for f in FEATS:
        x = d[f].astype(float).copy()
        x[~d.judge_ok] = np.nan
        if not meta[f]["realtime_comparable"]:
            x[d.realtime_window] = np.nan
        if kind_adj:
            allm = x.mean()
            km = x.groupby(d.kind_class).transform("mean")
            x = x - km + allm
        X[f] = x
    Z = pd.DataFrame(index=d.index)
    PCT = {}
    for f in FEATS:
        x = X[f]
        if VALID_ONLY:  # 엄격판 V: 결측 창을 건너뛴 직전 유효값 20개(전부 필요)·60개(전부 필요)
            xv = x.dropna()
            prev = xv.shift(1)
            mu = prev.rolling(20, min_periods=20).mean(); sd = prev.rolling(20, min_periods=20).std(ddof=1)
            z = ((xv - mu) / sd).reindex(x.index)
            sd = sd.reindex(x.index)
        else:
            prev = x.shift(1)
            mu = prev.rolling(20, min_periods=ZMIN).mean()
            sd = prev.rolling(20, min_periods=ZMIN).std(ddof=1)
            z = (x - mu) / sd
        z[sd <= 0] = np.nan
        Z[f] = z
        # 직전 60창 백분위
        arr = x.values
        pct = np.full(len(arr), np.nan)
        vidx = np.where(~np.isnan(arr))[0]
        for t in range(len(arr)):
            if np.isnan(arr[t]):
                continue
            if VALID_ONLY:
                prior = vidx[vidx < t][-60:]
                w = arr[prior]
                if len(w) < 60:
                    continue
            else:
                w = arr[max(0, t - 60):t]
                w = w[~np.isnan(w)]
                if len(w) < PCTMIN:
                    continue
            pct[t] = (np.sum(w < arr[t]) + 0.5 * np.sum(w == arr[t])) / len(w)
        PCT[f] = pct
    pos = {pd.Timestamp(D).strftime("%Y-%m-%d"): i for i, D in enumerate(d.D)}
    return d, Z, pd.DataFrame(PCT, index=d.index), pos


def pick_mid(B, valid):
    """B: (P,K) 점수, valid: (K,) 또는 (P,K) bool. 최대값 동률 중 가운데(아래 중앙값) 인덱스."""
    Bm = np.where(valid, B, -np.inf)
    m = Bm.max(axis=1, keepdims=True)
    T = np.isclose(Bm, m, atol=1e-12) & np.isfinite(Bm)
    cnt = T.sum(1)
    target = (cnt + 1) // 2
    cum = np.cumsum(T, axis=1)
    sel = (cum >= target[:, None]) & T
    k = sel.argmax(axis=1)
    return k, m[:, 0], cnt


def loo_ba(v, Y, rule="P"):
    """v: (n,) 값, Y: (P,n) 0/1 표지(1=사건). 반환: BA(P,), hits(P,n) bool."""
    v = np.asarray(v, float)
    n = len(v)
    order = np.argsort(v, kind="mergesort")
    vs = v[order]
    Ys = Y[:, order].astype(float)
    P = Ys.shape[0]
    hits = np.zeros((P, n), bool)
    for ii in range(n):
        keep = np.ones(n, bool); keep[ii] = False
        vt = vs[keep]; Yt = Ys[:, keep]
        Ft = 1 - Yt
        E = Yt.sum(1); F = Ft.sum(1)
        Ec = np.cumsum(Yt, 1)[:, :-1]; Fc = np.cumsum(Ft, 1)[:, :-1]
        valid = vt[:-1] < vt[1:]
        thr = (vt[:-1] + vt[1:]) / 2
        with np.errstate(invalid="ignore", divide="ignore"):
            ba_low = 0.5 * (Ec / E[:, None] + (F[:, None] - Fc) / F[:, None])
        ba_high = 1 - ba_low
        me = (Yt @ vt) / E; mf = (Ft @ vt) / F
        low_dir = me < mf
        if rule == "P":
            B = np.where(low_dir[:, None], ba_low, ba_high)
            k, _, _ = pick_mid(B, valid[None, :])
            use_low = low_dir
        else:  # J
            kl, ml, _ = pick_mid(ba_low, valid[None, :])
            kh, mh, _ = pick_mid(ba_high, valid[None, :])
            use_low = np.where(np.isclose(ml, mh), low_dir, ml > mh)
            k = np.where(use_low, kl, kh)
        if not valid.any():
            pred_event = np.zeros(P, bool)
        else:
            t = thr[k]
            x = vs[ii]
            pred_event = np.where(use_low, x <= t, x > t)
        hits[:, order[ii]] = pred_event == (Y[:, order[ii]] == 1)
    E_all = Y.sum(1); F_all = n - E_all
    ba = 0.5 * ((hits & (Y == 1)).sum(1) / E_all + (hits & (Y == 0)).sum(1) / F_all)
    return ba, hits


def full_rule(v, y):
    v = np.asarray(v, float); y = np.asarray(y, int)
    o = np.argsort(v, kind="mergesort"); vs = v[o]; ys = y[o].astype(float)
    E = ys.sum(); F = len(ys) - E
    Ec = np.cumsum(ys)[:-1]; Fc = np.cumsum(1 - ys)[:-1]
    valid = vs[:-1] < vs[1:]
    ba_low = 0.5 * (Ec / E + (F - Fc) / F)
    me = vs[ys == 1].mean(); mf = vs[ys == 0].mean()
    low = me < mf
    B = ba_low if low else 1 - ba_low
    k, m, _ = pick_mid(B[None, :], valid[None, :])
    thr = (vs[k[0]] + vs[k[0] + 1]) / 2
    return ("z <= %.2f" if low else "z > %.2f") % thr, float(m[0])


def all_fake_days_ba(ve, vf, groups):
    """사건 하나씩, 가짜 날은 묶음째 빼고 학습. 규칙 P."""
    ve = np.asarray(ve, float); vf = np.asarray(vf, float); groups = np.asarray(groups)
    def fit(vals, labs):
        o = np.argsort(vals, kind="mergesort"); vs = vals[o]; ys = labs[o].astype(float)
        E = ys.sum(); F = len(ys) - E
        if E == 0 or F == 0:
            return None
        Ec = np.cumsum(ys)[:-1]; Fc = np.cumsum(1 - ys)[:-1]
        valid = vs[:-1] < vs[1:]
        if not valid.any():
            return None
        ba_low = 0.5 * (Ec / E + (F - Fc) / F)
        low = vs[ys == 1].mean() < vs[ys == 0].mean()
        B = ba_low if low else 1 - ba_low
        k, _, _ = pick_mid(B[None, :], valid[None, :])
        return low, (vs[k[0]] + vs[k[0] + 1]) / 2
    he = []
    for i in range(len(ve)):
        r = fit(np.r_[np.delete(ve, i), vf], np.r_[np.ones(len(ve) - 1), np.zeros(len(vf))])
        he.append(False if r is None else ((ve[i] <= r[1]) if r[0] else (ve[i] > r[1])))
    hf = []
    for j in range(len(vf)):
        m = groups != groups[j]
        r = fit(np.r_[ve, vf[m]], np.r_[np.ones(len(ve)), np.zeros(m.sum())])
        pe = False if r is None else ((vf[j] <= r[1]) if r[0] else (vf[j] > r[1]))
        hf.append(not pe)
    return 0.5 * (np.mean(he) + np.mean(hf)), int(np.sum(he)), int(np.sum(hf)), len(vf)


def seed_of(name):
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


def perm_matrix(n_e, n, name):
    rng = np.random.default_rng(seed_of(name))
    y0 = np.r_[np.ones(n_e, int), np.zeros(n - n_e, int)]
    Y = np.empty((NPERM + 1, n), int); Y[0] = y0
    for p in range(1, NPERM + 1):
        Y[p] = rng.permutation(y0)
    return Y


def returns_table(asset):
    p = prices[prices.asset == asset].sort_values("date").reset_index(drop=True)
    p["r5"] = p.close / p.close.shift(5) - 1
    p["r20"] = p.close / p.close.shift(20) - 1
    p["ds"] = p.date.dt.strftime("%Y-%m-%d")
    return p


def run_combo(asset, X, kind_adj=True, tag="K"):
    mk = MARKET[asset]
    d, Z, PCT, pos = prep_market(mk, kind_adj)
    th = ev["assets"][asset]["thresholds"][X]
    period = ev["meta"]["period"]
    pt = returns_table(asset)
    tdays = list(pt.ds)
    tix = {s: i for i, s in enumerate(tdays)}
    piv_all = [p["date"] for p in th["pivots"]] + [th["tail_unconfirmed"]["date"]] + [p["date"] for p in th.get("warmup_pivots", [])]
    piv_idx = [tix[s] for s in piv_all if s in tix]
    rows, lead_rows, units_out = [], [], []
    for side in ["bottom", "top"]:
        evd = th["bottoms"] if side == "bottom" else th["tops"]
        fk = th["fake_bottoms"] if side == "bottom" else th["fake_tops"]
        # 묶음 대표
        reps = {}
        for f in fk:
            r = f["run"]
            if r not in reps:
                reps[r] = f
            else:
                cur = reps[r]
                better = f["close"] < cur["close"] if side == "bottom" else f["close"] > cur["close"]
                if better:
                    reps[r] = f
        repd = [reps[r]["date"] for r in sorted(reps)]
        fk_days = [f["date"] for f in fk]; fk_grp = [f["run"] for f in fk]
        units_out.append({"side": side, "events": evd, "fake_reps": repd, "n_fake_days": len(fk_days)})
        # 위약: 사건마다 최근접 20일
        lo, hi = period
        cand = pt[(pt.ds >= lo) & (pt.ds <= hi) & pt.r20.notna()].copy()
        cand = cand[cand.ds.isin(pos.keys())]
        ci = cand.ds.map(tix).values
        far = np.array([min(abs(c - q) for q in piv_idx) >= 4 for c in ci])
        cand = cand[far]
        s5, s20 = pt.r5.std(), pt.r20.std()
        plc = {}
        for e in evd:
            r = pt[pt.ds == e].iloc[0]
            dist = np.hypot((cand.r5 - r.r5) / s5, (cand.r20 - r.r20) / s20)
            plc[e] = list(cand.ds.values[np.argsort(dist.values, kind="mergesort")[:20]])
        for f in FEATS:
            for off in [0, 1]:
                def val(ds):
                    if ds not in pos:
                        return np.nan
                    i = pos[ds] + off
                    return Z[f].iloc[i] if i < len(Z) else np.nan
                ve = np.array([val(s) for s in evd]); vf = np.array([val(s) for s in repd])
                me, mf = ~np.isnan(ve), ~np.isnan(vf)
                rec = dict(asset=asset, X=X, tag=tag, side=side, feature=f, offset=off,
                           n_events=int(me.sum()), n_fakes=int(mf.sum()), status_feature=meta[f]["status"], bundle=meta[f]["bundle"])
                if me.sum() >= 3 and mf.sum() >= 3:
                    v = np.r_[ve[me], vf[mf]]
                    name = f"{asset}_{X}_{side}_{f}_{off}"
                    Y = perm_matrix(int(me.sum()), len(v), name)
                    for rule in ["P", "J"]:
                        ba, hits = loo_ba(v, Y, rule)
                        p = (1 + np.sum(ba[1:] >= ba[0] - 1e-9)) / (NPERM + 1)
                        rec[f"ba_{rule}"] = float(ba[0]); rec[f"p_{rule}"] = float(p)
                        rec[f"hitE_{rule}"] = int(hits[0, :me.sum()].sum()); rec[f"hitF_{rule}"] = int(hits[0, me.sum():].sum())
                        rec[f"null_q95_{rule}"] = float(np.quantile(ba[1:], 0.95))
                        if rule == "P":
                            evs = [s for s, m in zip(evd, me) if m]; fks = [s for s, m in zip(repd, mf) if m]
                            rec["miss_events"] = ",".join(s for s, h in zip(evs, hits[0, :me.sum()]) if not h)
                            rec["miss_fakes"] = ",".join(s for s, h in zip(fks, hits[0, me.sum():]) if not h)
                    rec["mean_z_events"] = float(np.nanmean(ve)); rec["mean_z_fakes"] = float(np.nanmean(vf))
                    rec["full_rule"], rec["full_ba_in"] = full_rule(v, np.r_[np.ones(me.sum()), np.zeros(mf.sum())])
                    vfd = np.array([val(s) for s in fk_days]); mfd = ~np.isnan(vfd)
                    if mfd.sum() >= 3:
                        b, he, hf, nf = all_fake_days_ba(ve[me], vfd[mfd], np.array(fk_grp)[mfd])
                        rec["ba_all_fake_days"] = b
                    # 누출 시험 재료: 사건/가짜 대표 중앙 z
                    rec["med_z_events"] = float(np.nanmedian(ve)); rec["med_z_fakes"] = float(np.nanmedian(vf))
                rows.append(rec)
            # 앞섬
            for lside in ["top10", "bot10"]:
                def lead(ds):
                    if ds not in pos:
                        return np.nan
                    i = pos[ds]
                    w = PCT[f].values[max(0, i - 5):i]
                    w = w[~np.isnan(w)]
                    if len(w) == 0:
                        return np.nan
                    return float((w >= 0.9).any() if lside == "top10" else (w <= 0.1).any())
                le = np.array([lead(s) for s in evd]); lf = np.array([lead(s) for s in repd])
                me, mf = ~np.isnan(le), ~np.isnan(lf)
                rec = dict(asset=asset, X=X, tag=tag, side=side, feature=f, lead_side=lside,
                           n_events=int(me.sum()), n_fakes=int(mf.sum()), status_feature=meta[f]["status"])
                if me.sum() >= 3 and mf.sum() >= 3:
                    v = np.r_[le[me], lf[mf]]; ne = int(me.sum())
                    Y = perm_matrix(ne, len(v), f"lead_{asset}_{X}_{side}_{f}_{lside}")
                    re_ = (Y * v).sum(1) / ne; rf = ((1 - Y) * v).sum(1) / (len(v) - ne)
                    diff = re_ - rf
                    rec.update(rate_events=float(re_[0]), rate_fakes=float(rf[0]), diff=float(diff[0]),
                               p_fake_perm=float((1 + np.sum(diff[1:] >= diff[0] - 1e-9)) / (NPERM + 1)))
                    # 위약
                    pr_rates = []
                    per_event = []
                    for e in [s for s, m in zip(evd, me) if m]:
                        xs = [lead(s) for s in plc[e]]
                        xs = [x for x in xs if not np.isnan(x)]
                        if xs:
                            per_event.append(np.mean(xs)); pr_rates += xs
                    if pr_rates:
                        q = float(np.mean(pr_rates))
                        k = int(round(re_[0] * ne))
                        rec["rate_placebo"] = q
                        rec["p_vs_placebo"] = float(stats.binom.sf(k - 1, ne, q))
                        # 위약 97.5% 분위: 사건마다 위약 20일 중 하나씩 뽑은 비율의 분포(이항 근사)
                        rec["placebo_q975"] = float(stats.binom.ppf(0.975, ne, q) / ne)
                lead_rows.append(rec)
    return pd.DataFrame(rows), pd.DataFrame(lead_rows), units_out


if __name__ == "__main__":
    combos = [("KOSPI", "8"), ("KOSPI", "12"), ("SKHYNIX", "8"), ("SKHYNIX", "12"), ("SOXX", "8"), ("SOXX", "12")]
    tags = sys.argv[1:] or ["K"]
    allr, alll, units = [], [], {}
    for tag in tags:
        globals().update(VALID_ONLY=(tag == "V"))  # V = 보정 K + 직전 유효값 20개·60개 전부 필요
        for a, x in combos:
            r, l, u = run_combo(a, x, kind_adj=(tag in ("K", "V")), tag=tag)
            allr.append(r); alll.append(l); units[f"{a}_{x}"] = u
            print(tag, a, x, "done", flush=True)
    pd.concat(allr).to_csv(OUT / f"recheck_loo_{'_'.join(tags)}.csv", index=False, encoding="utf-8-sig")
    pd.concat(alll).to_csv(OUT / f"recheck_lead_{'_'.join(tags)}.csv", index=False, encoding="utf-8-sig")
    json.dump(units, open(OUT / "units.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
