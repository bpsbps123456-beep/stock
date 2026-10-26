"""칼손절 연구 — 결합 모델(시간순 학습) 계열 (2026-10-07).

매수 뒤 k분(15·30·60) 체크포인트 봉 cp = ei + k//mins 의 종가 시점까지 알 수 있는 특징으로
"이 매수가 결국 -3%(또는 -5%) 넘게 잃고 끝날 확률"을 early 기준 거래로만 학습한 numpy 로지스틱(L2)으로 추정하고,
확률이 p를 넘으면 그다음 봉(i = cp + 1)의 시가에 판다. 체크는 그 한 봉에서만 한다(여러 체크포인트 묶음은 각 모델).

학습·선택 순서(late는 고르는 데 쓰지 않는다):
  1) early 기준 거래(24시간, 137종목)를 날짜순으로 앞 2/3(학습)·뒤 1/3(검증)으로 나눈다.
  2) 앞 2/3로 학습한 모델로 early 전체를 다시 돌려 뒤 1/3 날짜의 사용자78 차이(24시간 주, 정규장만 보조)로 k·p·변형을 고른다.
  3) 고른 설정을 early 전체로 다시 학습해 compare()로 late에 그대로 적용(전방 확인). 시험한 모든 설정의 네 칸도 센다.
과적합 정도: 앞 모델의 앞 2/3(학습 날짜) 하루 평균 차이 vs 뒤 1/3(검증 날짜) 하루 평균 차이, 학습/검증 AUC.

특징(전부 cp 봉 종가까지·매수 직전 값만; 다른 종목·QQQ도 같은 시각 이하 값):
  rk 매수 뒤 수익(c[cp]/ent-1) · lowk 그때까지 최저(min l[ei..cp]/ent-1) · rsi_cp 기준 RSI(cp) · rsi_e 기준 RSI(ei-1)
  slope60 기준 EMA 60분 기울기(ema[cp]/ema[cp-60분]-1) · dev_cp 이격(c[cp]/ema[cp]-1) · gap_e 매수 때 이격(ent/ema[ei-1]-1)
  pre·aft 매수 세션 · tod 매수 ET 시각(시) · prev_ret 전날 수익률(일봉 D-1/D-2) · today_ret 매수가/전날 종가-1
  vol 정규장 매수면 log(매수~cp 평균 거래량 / 직전 하루치 정규장 봉 평균 거래량), 아니면 0
  qqq QQQ 같은 구간 변화(c asof tk[cp] / c asof tk[ei-1] - 1) · mkt 137종목 평균 이격(같은 시각, 최근값)
변형: 라벨 ret(최종 -3%·-5%), fwd(체크포인트 뒤 계속 들면 2% 넘게 더 잃음, 로지스틱), reg(계속 들 때 수익 회귀, 예측 < -θ면 매도).
사용: py model.py            (전체 격자, 결과 ../../results/stops/model.txt)
"""
import os, sys, time, io
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from stop_lab import load, run, compare, base_trades, metrics, stop_audit, UNIV, USER78, RAW, GAP  # noqa: E402
from events import bars, prep  # noqa: E402

OUT = os.path.normpath(os.path.join(HERE, "..", "..", "results", "stops", "model.txt"))
PER = {5: "early", 3: "late"}
FULL = ["rk", "lowk", "rsi_cp", "rsi_e", "slope60", "dev_cp", "gap_e", "pre", "aft", "tod",
        "prev_ret", "today_ret", "vol", "qqq", "mkt"]
MINI = ["rk", "lowk", "rsi_cp"]
LAM = 1.0

_LOG = io.StringIO()


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _LOG.write(s + "\n")


# ---------------------------------------------------------------- 공통 맥락(QQQ·시장 이격)
_CTX = {}


def ctx(period):
    if period in _CTX:
        return _CTX[period]
    S = load(period)
    mins = S[next(iter(S))]["mins"]
    q = bars(RAW, "QQQ", mins)
    q_tk = q.index.as_unit("ns").asi8
    q_c = q.Close.values.astype(float)
    cols = {s: pd.Series(np.clip(A["c"] / A["ema"] - 1, -0.1, 0.1), index=A["tk"]) for s, A in S.items()}
    D = pd.DataFrame(cols).sort_index().ffill(limit=max(2, 30 // mins))
    mkt = D.mean(axis=1)
    _CTX[period] = dict(q_tk=q_tk, q_c=q_c, m_tk=mkt.index.values.astype(np.int64), m_v=mkt.values, mins=mins)
    return _CTX[period]


def _asof(tk_arr, val, t):
    j = np.searchsorted(tk_arr, t, side="right") - 1
    out = np.where(j >= 0, val[np.clip(j, 0, None)], np.nan)
    return out


_DAILY = {}


def daily(sym):
    if sym not in _DAILY:
        c = pd.read_parquet(f"{RAW}/1d_{sym}.parquet").Close
        if isinstance(c, pd.DataFrame):
            c = c.iloc[:, 0]
        _DAILY[sym] = (np.array([d.year * 10000 + d.month * 100 + d.day for d in pd.to_datetime(c.index).date]),
                       c.values.astype(float))
    return _DAILY[sym]


# ---------------------------------------------------------------- 특징(매수 봉 e 마다 미리 계산)
_FE = {}


def feats(A, kb):
    """매수 봉 e마다 체크포인트 cp=e+kb 시점 특징 행렬(n × len(FULL)). 쓸 수 없는 행은 NaN."""
    key = (A["mins"], A["sym"], kb)
    if key in _FE:
        return _FE[key]
    period = PER[A["mins"]]
    C = ctx(period)
    o, l, c, v, r, ema, tk, mod, reg = A["o"], A["l"], A["c"], A["v"], A["r"], A["ema"], A["tk"], A["mod"], A["reg"]
    n, mins = len(o), A["mins"]
    e = np.arange(n)
    cp = e + kb
    ok = (e >= 1) & (cp < n)
    e1 = np.clip(e - 1, 0, n - 1)
    cpc = np.clip(cp, 0, n - 1)
    ent = np.minimum(o, ema[e1] * (1 - GAP))
    lmin = pd.Series(l).rolling(kb + 1, min_periods=1).min().values  # lmin[j] = min(l[j-kb..j])
    F = {}
    F["rk"] = c[cpc] / ent - 1
    F["lowk"] = lmin[cpc] / ent - 1
    F["rsi_cp"] = r[cpc]
    F["rsi_e"] = r[e1]
    s60 = 60 // mins
    F["slope60"] = ema[cpc] / ema[np.clip(cpc - s60, 0, None)] - 1
    F["dev_cp"] = c[cpc] / ema[cpc] - 1
    F["gap_e"] = ent / ema[e1] - 1
    ses = A["ses"]
    F["pre"] = (ses == "프리").astype(float)
    F["aft"] = (ses == "애프터").astype(float)
    F["tod"] = mod / 60.0
    # 일봉: 매수 날짜 D 이전의 마지막 두 종가
    dk, dc = daily(A["sym"])
    dayi = np.array([x.year * 10000 + x.month * 100 + x.day for x in A["day"]])
    j = np.searchsorted(dk, dayi, side="left") - 1  # D 미만 마지막 날
    c1 = np.where(j >= 0, dc[np.clip(j, 0, None)], np.nan)
    c2 = np.where(j >= 1, dc[np.clip(j - 1, 0, None)], np.nan)
    F["prev_ret"] = np.clip(c1 / c2 - 1, -0.3, 0.3)
    F["today_ret"] = np.clip(ent / c1 - 1, -0.3, 0.3)
    # 정규장 거래량 배수
    B = 390 // mins
    cs = np.concatenate([[0.0], np.cumsum(v)])
    win = (cs[cpc + 1] - cs[e]) / (kb + 1)
    ri = np.flatnonzero(reg)
    rcs = np.concatenate([[0.0], np.cumsum(v[ri])])
    pos = np.searchsorted(ri, e, side="left")  # e 이전 정규장 봉 수
    lo = np.clip(pos - B, 0, None)
    base = np.where(pos - lo > 0, (rcs[pos] - rcs[lo]) / np.maximum(pos - lo, 1), np.nan)
    vr = np.where(reg & (base > 0) & (win > 0), np.log(np.maximum(win, 1e-9) / np.where(base > 0, base, 1)), 0.0)
    F["vol"] = np.clip(vr, -3, 3)
    q0 = _asof(C["q_tk"], C["q_c"], tk[e1])
    q1 = _asof(C["q_tk"], C["q_c"], tk[cpc])
    F["qqq"] = np.clip(q1 / q0 - 1, -0.1, 0.1)
    F["mkt"] = _asof(C["m_tk"], C["m_v"], tk[cpc])
    X = np.column_stack([F[k] for k in FULL]).astype(float)
    X[~ok] = np.nan
    X[~np.isfinite(X)] = np.nan
    X[~ok, 0] = np.nan
    _FE[key] = X
    return X


# ---------------------------------------------------------------- 로지스틱(numpy, L2)
def fit_logit(X, y, lam=LAM):
    mu = np.nanmean(X, 0)
    sd = np.nanstd(X, 0)
    sd[~(sd > 0)] = 1.0
    Z = np.clip((np.where(np.isnan(X), mu, X) - mu) / sd, -5, 5)
    Z1 = np.c_[np.ones(len(Z)), Z]
    w = np.zeros(Z1.shape[1])
    pm = min(max(y.mean(), 1e-3), 1 - 1e-3)
    w[0] = np.log(pm / (1 - pm))
    R = lam * np.eye(len(w))
    R[0, 0] = 0
    for _ in range(100):
        p = 1 / (1 + np.exp(-np.clip(Z1 @ w, -30, 30)))
        g = Z1.T @ (p - y) + R @ w
        H = (Z1 * (p * (1 - p))[:, None]).T @ Z1 + R
        dw = np.linalg.solve(H, g)
        w -= dw
        if np.abs(dw).max() < 1e-9:
            break
    return dict(w=w, mu=mu, sd=sd)


def fit_ridge(X, y, lam=LAM):
    """보유 지속 수익(fwd) 선형 회귀(표준화 특징, L2). 점수 = -예측 fwd."""
    mu = np.nanmean(X, 0)
    sd = np.nanstd(X, 0)
    sd[~(sd > 0)] = 1.0
    Z = np.clip((np.where(np.isnan(X), mu, X) - mu) / sd, -5, 5)
    Z1 = np.c_[np.ones(len(Z)), Z]
    R = lam * np.eye(Z1.shape[1])
    R[0, 0] = 0
    w = np.linalg.solve(Z1.T @ Z1 + R, Z1.T @ y)
    return dict(w=w, mu=mu, sd=sd, kind="reg")


def fit(X, y, kind):
    return fit_ridge(X, y) if kind == "reg" else fit_logit(X, y)


def predict(m, X):
    """logit: 확률, reg: -예측 보유 지속 수익(클수록 팔 이유). 체크포인트 없는 행은 0(팔지 않음 쪽)."""
    Z = np.clip((np.where(np.isnan(X), m["mu"], X) - m["mu"]) / m["sd"], -5, 5)
    z = np.c_[np.ones(len(Z)), Z] @ m["w"]
    p = -z if m.get("kind") == "reg" else 1 / (1 + np.exp(-np.clip(z, -30, 30)))
    p[np.isnan(X[:, 0])] = -1.0 if m.get("kind") == "reg" else 0.0
    return p


def auc(p, y):
    y = y.astype(bool)
    if y.all() or (~y).all():
        return np.nan
    rk = pd.Series(p).rank().values
    return (rk[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())


# ---------------------------------------------------------------- 학습 표
def train_table(k, cols, days=None, loss=0.03, kind="ret"):
    """early 기준 거래(24시간, 137종목) 중 체크포인트까지 들고 있던 것. 반환 X, y.
    kind: ret → y = 최종 수익 <= -loss / fwd → y = (최종 수익 - 체크포인트 다음 봉 시가 매도 수익) <= -loss
          reg → y = clip(최종 수익 - 체크포인트 다음 봉 시가 매도 수익, ±10%)  (라벨은 결과값이라 미래를 써도 된다)"""
    S = load("early")
    b = base_trades("early", False)
    if days is not None:
        b = b[b.day.isin(days)]
    ci = [FULL.index(cn) for cn in cols]
    Xs, ys = [], []
    for x in b.itertuples():
        A = S[x.sym]
        kb = k // A["mins"]
        if x.xi <= x.ei + kb:  # 체크포인트 전에 이미 팔림
            continue
        Xs.append(feats(A, kb)[x.ei, ci])
        fwd = x.ret - (A["o"][x.ei + kb + 1] / x.ent - 1)
        ys.append(x.ret <= -loss if kind == "ret" else (fwd <= -loss if kind == "fwd" else np.clip(fwd, -0.1, 0.1)))
    X = np.array(Xs)
    y = np.array(ys, dtype=float)
    keep = ~np.isnan(X[:, 0])
    return X[keep], y[keep]


def factory(models, p, cols):
    """models: {k분: 모델}. 규칙: i - ei == k//mins + 1 이면 그 체크포인트 확률 > p 일 때 봉 i 시가에 판다."""
    ci = [FULL.index(cn) for cn in cols]

    def make_rule(A):
        sells = {}
        for k, m in models.items():
            kb = k // A["mins"]
            pr = predict(m, feats(A, kb)[:, ci])
            sells[kb + 1] = (pr > p).tolist()

        def rule(i, ei, ent):
            s = sells.get(i - ei)
            return s is not None and s[ei]
        return rule
    return make_rule


# ---------------------------------------------------------------- 실행
def early_split():
    b = base_trades("early", False)
    days = sorted(set(b.day))
    cut = int(round(len(days) * 2 / 3))
    return days[:cut], days[cut:]


def eval_front(make_rule, tr_days, va_days):
    """앞 2/3 모델로 early 전체를 돌려 학습 날짜·검증 날짜 차이(사용자78/전체137, 24시간/정규장만)."""
    S = load("early")
    res = {}
    for reg in (False, True):
        t = run(S, make_rule, "early", reg)
        b = base_trades("early", reg)
        for nm, dd in (("tr", tr_days), ("va", va_days)):
            for un, us in UNIV.items():
                m = metrics(t[t.day.isin(dd)], b[b.day.isin(dd)], us)
                res[(nm, "reg" if reg else "24h", un)] = m
    return res


def main():
    t0 = time.time()
    for p_ in ("early", "late"):
        load(p_)
        base_trades(p_, False)
        base_trades(p_, True)
        ctx(p_)
    tr_days, va_days = early_split()
    say(f"early 날짜 {len(tr_days) + len(va_days)}일: 학습 {tr_days[0]}~{tr_days[-1]} ({len(tr_days)}일), "
        f"검증 {va_days[0]}~{va_days[-1]} ({len(va_days)}일)")
    say(f"특징(전체) {FULL}  / 최소 {MINI}  / L2 λ={LAM}")
    lookahead_check()

    # 변형: (이름, 특징, 라벨 종류, 손실, 체크포인트 묶음, 문턱 격자)
    PG3 = (0.15, 0.20, 0.30, 0.40, 0.50)
    PG5 = (0.10, 0.15, 0.20, 0.30, 0.40)
    PGF = (0.20, 0.23, 0.26, 0.29, 0.32)
    PGR = (0.003, 0.005, 0.007)
    variants = []
    for k in (15, 30, 60):
        variants.append((f"전체-3%@{k}", FULL, "ret", 0.03, (k,), PG3))
    variants.append(("전체-3%@15+30+60", FULL, "ret", 0.03, (15, 30, 60), PG3))
    for k in (15, 30, 60):
        variants.append((f"최소-3%@{k}", MINI, "ret", 0.03, (k,), PG3))
    for k in (15, 30, 60):
        variants.append((f"전체-5%@{k}", FULL, "ret", 0.05, (k,), PG5))
    for k in (15, 30, 60):  # 결정용 라벨: 체크포인트에서 계속 들면 2% 넘게 더 잃음
        variants.append((f"전체-더잃음2%@{k}", FULL, "fwd", 0.02, (k,), PGF))
    for k in (15, 30, 60):  # 회귀: 예측 보유 지속 수익 < -θ 이면 판다
        variants.append((f"회귀-지속수익@{k}", FULL, "reg", 0.0, (k,), PGR))

    # 라벨 기초 통계·AUC(앞 2/3 학습, 뒤 1/3 검증)
    say("\n=== 분류 성능(체크포인트까지 들고 있던 early 기준 거래, 137종목) ===")
    fronts, fulls = {}, {}
    for name, cols, kind, loss, ks, pg in variants:
        fm, gm = {}, {}
        for k in ks:
            Xtr, ytr = train_table(k, cols, tr_days, loss, kind)
            Xva, yva = train_table(k, cols, va_days, loss, kind)
            Xal, yal = train_table(k, cols, None, loss, kind)
            fm[k] = fit(Xtr, ytr, kind)
            gm[k] = fit(Xal, yal, kind)
            ptr, pva = predict(fm[k], Xtr), predict(fm[k], Xva)
            if kind == "reg":
                say(f"{name:>18} k={k:>2}분: 학습 {len(ytr)}건 평균 {ytr.mean():+.4f} 상관 {np.corrcoef(-ptr, ytr)[0, 1]:+.3f} | "
                    f"검증 {len(yva)}건 평균 {yva.mean():+.4f} 상관 {np.corrcoef(-pva, yva)[0, 1]:+.3f} | "
                    f"검증 예측<-θ 비율 " + " ".join(f"{p:.3f}:{(pva > p).mean():.1%}(실제 평균 {yva[pva > p].mean() if (pva > p).any() else np.nan:+.4f})" for p in pg))
            else:
                say(f"{name:>18} k={k:>2}분: 학습 {len(ytr)}건 양성 {ytr.mean():.1%} AUC {auc(ptr, ytr):.3f} | "
                    f"검증 {len(yva)}건 양성 {yva.mean():.1%} AUC {auc(pva, yva):.3f} | "
                    f"검증 확률>p 비율 " + " ".join(f"{p:.2f}:{(pva > p).mean():.1%}" for p in pg))
        fronts[name], fulls[name] = fm, gm
    # 전체 학습 계수 기록(-3% 전체 변형만)
    for name in ("전체-3%@15", "전체-3%@30", "전체-3%@60"):
        m = fulls[name][int(name.split("@")[1])]
        say(f" 계수(early 전체 학습, 표준화) {name}: " +
            " ".join(f"{c}={w:+.2f}" for c, w in zip(["절편"] + FULL, m["w"])))

    rows = []
    nrule = 0
    for name, cols, kind, loss, ks, pg in variants:
        for p in pg:
            nrule += 1
            lab = f"{name} {'θ' if kind == 'reg' else 'p'}>{p:.3f}" if kind == "reg" else f"{name} p>{p:.2f}"
            fr = eval_front(factory(fronts[name], p, cols), tr_days, va_days)
            cm = compare(factory(fulls[name], p, cols), lab, show=False)

            def cell(per, rng, un):
                x = cm[(cm.기간 == per) & (cm.범위 == rng) & (cm.종목 == un)].iloc[0]
                return x
            row = dict(규칙=lab, 이름=name, p=p,
                       va24=fr[("va", "24h", "사용자78")]["차이"], vareg=fr[("va", "reg", "사용자78")]["차이"],
                       va24_t=fr[("va", "24h", "사용자78")]["t"], va_stop=fr[("va", "24h", "사용자78")]["손절"],
                       tr24=fr[("tr", "24h", "사용자78")]["차이"], trreg=fr[("tr", "reg", "사용자78")]["차이"],
                       va24_all=fr[("va", "24h", "전체137")]["차이"])
            for per in ("early", "late"):
                for rng in ("24시간", "정규장만"):
                    for un in ("사용자78", "전체137"):
                        x = cell(per, rng, un)
                        tag = f"{per}_{'24' if rng == '24시간' else 'reg'}_{'u' if un == '사용자78' else 'a'}"
                        row[tag] = x["차이"]
                        row[tag + "_t"] = x["t"]
                        row[tag + "_stop"] = x["손절"]
                        row[tag + "_worst"] = x["최악"]
                        row[tag + "_oct6"] = x["차이_10월6일"]
            rows.append(row)
            say(f"[{nrule:>2}] {lab:<26} 검증(뒤1/3) 24h {row['va24']:+.2f} 정규 {row['vareg']:+.2f} 손절 {row['va_stop']} | "
                f"학습(앞2/3) 24h {row['tr24']:+.2f} 정규 {row['trreg']:+.2f} | 전체모델 early 24h {row['early_24_u']:+.2f} "
                f"정규 {row['early_reg_u']:+.2f} · late 24h {row['late_24_u']:+.2f} 정규 {row['late_reg_u']:+.2f}  ({time.time() - t0:.0f}s)")
            with open(OUT, "w", encoding="utf-8") as f:
                f.write(_LOG.getvalue() + "\n(진행 중)\n")
    R = pd.DataFrame(rows)
    ntr, nva = len(tr_days), len(va_days)
    R["tr_day"] = R.tr24 / ntr
    R["va_day"] = R.va24 / nva
    say(f"\n시험한 규칙 수: {nrule}")
    four = ((R.early_24_u > 0) & (R.early_reg_u > 0) & (R.late_24_u > 0) & (R.late_reg_u > 0))
    four_v = ((R.va24 > 0) & (R.vareg > 0) & (R.late_24_u > 0) & (R.late_reg_u > 0))
    say(f"사용자78 네 칸(전체모델 early 24h·정규, late 24h·정규) 모두 플러스: {int(four.sum())}개  "
        f"/ (early를 뒤1/3 검증으로 바꾼 네 칸): {int(four_v.sum())}개 {list(R[four_v].규칙)}")
    say(f"과적합 지표(사용자78 24h 하루 평균 차이): 학습 날짜 평균 {R.tr_day.mean():+.4f} vs 검증 날짜 평균 {R.va_day.mean():+.4f}; "
        f"규칙별 학습-검증 상관 {R.tr_day.corr(R.va_day):+.2f}")

    # 선택: 검증 24h(사용자78) 최대, 검증 정규장만 >= 0 우선
    cand = R[R.vareg >= 0]
    pick = (cand if len(cand) else R).sort_values("va24", ascending=False)
    best = pick.iloc[0]
    say(f"\n=== early(뒤 1/3 검증)로 고른 최선안: {best.규칙} (검증 24h {best.va24:+.2f}, 정규 {best.vareg:+.2f}) ===")
    tab = R.sort_values("va24", ascending=False)[["규칙", "va24", "vareg", "va24_t", "va_stop", "tr24", "trreg",
                                                    "early_24_u", "early_reg_u", "late_24_u", "late_reg_u",
                                                    "early_24_a", "late_24_a"]]
    say(tab.to_string(index=False, float_format=lambda v: f"{v:+.2f}"))

    name = best.이름
    cols = next(v[1] for v in variants if v[0] == name)
    mk = factory(fulls[name], best.p, cols)
    say(f"\n=== 최선안 전체 표(early 전체 학습 모델; early 칸은 학습 안쪽, late 칸은 전방 확인) ===")
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    cm = compare(mk, best.규칙, show=True)
    sys.stdout = old
    say(buf.getvalue())
    for per in ("early", "late"):
        for reg in (False, True):
            a = stop_audit(mk, per, reg_only=reg)
            say(f"stop_audit {per} {'정규장만' if reg else '24시간'} 사용자78: " +
                ", ".join(f"{k} {v:+.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in a.items()))
    # 보충: '결국 -3% 넘게 잃을 확률' 모델이 왜 돈이 안 되나(전체 학습 모델, 사용자78 24시간)
    say("\n=== 보충 stop_audit: -3% 라벨 모델(전체 학습) — 손절 수익 vs 같은 매수의 기준 수익 ===")
    for nm, p in (("전체-3%@15", 0.20), ("전체-3%@30", 0.30), ("전체-3%@60", 0.20), ("전체-3%@60", 0.50)):
        mk2 = factory(fulls[nm], p, FULL)
        for per in ("early", "late"):
            a = stop_audit(mk2, per)
            say(f"{nm} p>{p:.2f} {per}: " + ", ".join(f"{k} {v:+.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in a.items()))
    say("\n=== 전체 격자(열: va=앞 모델 뒤1/3 검증, tr=앞 모델 앞2/3 학습, early_/late_=전체 모델 compare; _u 사용자78, _a 전체137) ===")
    say(R.drop(columns=[c for c in R.columns if c.endswith(("_worst", "_oct6"))]).to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(_LOG.getvalue())
    say(f"\n저장: {OUT}  ({time.time() - t0:.0f}s)")


def lookahead_check(nsym=12, seed=0):
    """미래 참조 검사: 봉 j 이후 o·h·l·c·v·ema·r 를 무작위로 바꿔도 cp < j 인 매수 봉의 특징이 그대로인지(py model.py check)."""
    rng = np.random.default_rng(seed)
    bad = tot = 0
    for per in ("early", "late"):
        S = load(per)
        for s in list(S)[:nsym]:
            A = S[s]
            n = len(A["o"])
            for k in (15, 30, 60):
                kb = k // A["mins"]
                X = feats(A, kb)
                j = int(rng.integers(n // 3, n - 100))
                B = dict(A)
                B["sym"] = s + "_P"
                for key in ("o", "h", "l", "c", "v", "ema", "r"):
                    arr = A[key].copy()
                    arr[j:] = arr[j:] * rng.uniform(0.5, 1.5, n - j)
                    B[key] = arr
                _DAILY[s + "_P"] = daily(s)
                Y = feats(B, kb)
                e = np.arange(n)
                m = (e + kb < j) & (e >= 1)
                tot += 1
                bad += np.nanmax(np.abs(np.nan_to_num(X[m]) - np.nan_to_num(Y[m]))) > 1e-12
    say(f"미래 참조 검사(봉 j 이후 o·h·l·c·v·ema·r 무작위 교란 → cp<j 매수 봉 특징 불변?): {tot}회, 위반 {bad}회")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        lookahead_check()
    else:
        main()
