"""반박 검증 — g_model 최선안 'T60-fwd2 FR top12%(p>0.269)'를 정의 문장만 보고 다시 만든다(2026-10-08).

정의(연구자 보고): 매수 봉 ei, cp = ei + 60//mins. i - ei == 60//mins + 1 인 봉 i 한 번만 판단, P > 0.269면 봉 i 시가에 판다.
모델 = L2 로지스틱(λ=1, 절편 벌점 없음, 학습 표본 평균·표준편차 표준화 후 ±5, NaN은 평균), early 기준 거래(24시간, 137종목) 중 xi > cp,
라벨 = (최종 수익 - (o[cp+1]/ent - 1)) <= -2%. 특징 21개(cp 종가까지 + dev/conf[cp+1], dev/conf[ei]).

연구자 코드(g_model.py·model.py)는 불러오지 않는다. 특징은 규칙이 불릴 때 배열을 [:cp+1]로 잘라 계산한다(구조상 미래 없음).
40분봉 RSI는 stop_lab.htf_rsi를 쓰되, 5분봉에서 직접 만든 40분봉 RSI와 같은지 따로 대조한다.
로지스틱은 scipy L-BFGS로 푼다(연구자는 Newton).
사용: py stops/verify_g_model.py → ../results/stops/verify_g_model.txt
"""
import os, sys, time
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from stop_lab import load, compare, stop_audit, base_trades, run, htf_rsi, RAW, UNIV, GAP  # noqa: E402
from events import bars  # noqa: E402

OUT = os.path.normpath(os.path.join(HERE, "..", "..", "results", "stops", "verify_g_model.txt"))
U78 = UNIV["사용자78"]
NAMES = ["rk", "lowk", "rsi_cp", "rsi_e", "slope60", "dev_cp", "gap_e", "pre", "aft", "tod", "prev_ret", "today_ret",
         "vol", "qqq", "mkt", "el", "d40_cp", "c40_cp", "d40_e", "dd40", "dc40"]
NF = 16                      # 앞 16개 = F(40분봉 특징 없음)
LOG = []
T0 = time.time()


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def flush():
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(LOG) + "\n")


# ------------------------------------------------------------------ 맥락
_C = {}


def context(mins):
    if mins in _C:
        return _C[mins]
    per = "early" if mins == 5 else "late"
    S = load(per)
    q = bars(RAW, "QQQ", mins)
    qt, qc = q.index.as_unit("ns").asi8, q.Close.to_numpy(float)
    grid = np.unique(np.concatenate([A["tk"] for A in S.values()]))
    tot, cnt = np.zeros(len(grid)), np.zeros(len(grid))
    lim = 30 * 60 * 10 ** 9
    for A in S.values():
        val = np.clip(A["c"] / A["ema"] - 1, -0.1, 0.1)
        k = np.searchsorted(A["tk"], grid, side="right") - 1
        okk = (k >= 0)
        kk = np.maximum(k, 0)
        okk &= (grid - A["tk"][kk]) <= lim
        okk &= np.isfinite(val[kk])
        tot += np.where(okk, val[kk], 0.0)
        cnt += okk
    mkt = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    _C[mins] = dict(qt=qt, qc=qc, gt=grid, gm=mkt)
    return _C[mins]


_D = {}


def daily_closes(sym):
    if sym not in _D:
        d = pd.read_parquet(f"{RAW}/1d_{sym}.parquet")
        c = d["Close"]
        if isinstance(c, pd.DataFrame):
            c = c.iloc[:, 0]
        days = np.array([x.date() for x in pd.to_datetime(c.index)], dtype=object)
        _D[sym] = (days, c.to_numpy(float))
    return _D[sym]


def last_le(t_arr, v_arr, t):
    k = np.searchsorted(t_arr, t, side="right") - 1
    return v_arr[k] if k >= 0 else np.nan


_REG = {}


def feature(A, ei, cp, ent, dev, conf):
    """판단 봉 cp+1 시가에 알 수 있는 21개. 가격 배열은 [:cp+1]만 본다."""
    mins = A["mins"]
    o, l, c, v, r, ema = (A[k][:cp + 1] for k in ("o", "l", "c", "v", "r", "ema"))
    tk = A["tk"][:cp + 1]
    reg = A["reg"][:cp + 1]
    C = context(mins)
    f = np.full(21, np.nan)
    f[0] = c[cp] / ent - 1
    f[1] = l[ei:].min() / ent - 1
    f[2] = r[cp]
    f[3] = r[ei - 1]
    f[4] = ema[cp] / ema[max(cp - 60 // mins, 0)] - 1
    f[5] = c[cp] / ema[cp] - 1
    f[6] = ent / ema[ei - 1] - 1
    ses = A["ses"][ei]
    f[7] = 1.0 if ses == "프리" else 0.0
    f[8] = 1.0 if ses == "애프터" else 0.0
    f[9] = A["mod"][ei] / 60.0
    days, dc = daily_closes(A["sym"])
    D = A["day"][ei]
    prior = [k for k in range(len(days)) if days[k] < D]
    c1 = dc[prior[-1]] if len(prior) >= 1 else np.nan
    c2 = dc[prior[-2]] if len(prior) >= 2 else np.nan
    f[10] = np.clip(c1 / c2 - 1, -0.3, 0.3)
    f[11] = np.clip(ent / c1 - 1, -0.3, 0.3)
    f[12] = 0.0
    if reg[ei]:
        prev_reg = np.flatnonzero(reg[:ei])[-(390 // mins):]
        base = v[prev_reg].mean() if len(prev_reg) else np.nan
        win = v[ei:cp + 1].mean()
        if base > 0 and win > 0:
            f[12] = np.clip(np.log(win / base), -3, 3)
    f[13] = np.clip(last_le(C["qt"], C["qc"], tk[cp]) / last_le(C["qt"], C["qc"], tk[ei - 1]) - 1, -0.1, 0.1)
    f[14] = last_le(C["gt"], C["gm"], tk[cp])
    f[15] = (cp - ei) * mins
    f[16] = dev[cp + 1]
    f[17] = conf[cp + 1]
    f[18] = dev[ei]
    f[19] = dev[cp + 1] - dev[ei]
    f[20] = conf[cp + 1] - conf[ei]
    f[~np.isfinite(f)] = np.nan
    return f


# ------------------------------------------------------------------ 로지스틱(L-BFGS)
def fit(X, y, lam=1.0):
    mu = np.nanmean(X, 0)
    sd = np.nanstd(X, 0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = np.clip((np.where(np.isnan(X), mu, X) - mu) / sd, -5, 5)
    Z1 = np.c_[np.ones(len(Z)), Z]

    def obj(w):
        z = Z1 @ w
        nll = np.sum(np.logaddexp(0, z) - y * z)
        g = Z1.T @ (expit(z) - y)
        g[1:] += lam * w[1:]
        return nll + 0.5 * lam * np.sum(w[1:] ** 2), g
    res = minimize(obj, np.zeros(Z1.shape[1]), jac=True, method="L-BFGS-B", options=dict(maxiter=5000, gtol=1e-10))
    return dict(w=res.x, mu=mu, sd=sd)


def prob(m, X):
    X = np.atleast_2d(X)
    Z = np.clip((np.where(np.isnan(X), m["mu"], X) - m["mu"]) / m["sd"], -5, 5)
    return expit(np.c_[np.ones(len(Z)), Z] @ m["w"])


# ------------------------------------------------------------------ 학습 표
_HR = {}


def hr(A):
    k = (A["sym"], A["mins"])
    if k not in _HR:
        _HR[k] = htf_rsi(A, 40, 14)
    return _HR[k]


def train_table(kmin=60, loss=0.02):
    S = load("early")
    b = base_trades("early", False)
    X, y, d = [], [], []
    for x in b.itertuples():
        A = S[x.sym]
        cp = x.ei + kmin // A["mins"]
        if not x.xi > cp:
            continue
        dev, conf = hr(A)
        f = feature(A, x.ei, cp, x.ent, dev, conf)
        if np.isnan(f[0]):
            continue
        X.append(f)
        y.append(float((x.ret - (A["o"][cp + 1] / x.ent - 1)) <= -loss))
        d.append(x.day)
    return np.array(X), np.array(y), np.array(d, dtype=object)


def make(model, p, cols, kmin=60, gate=None):
    def make_rule(A):
        kb = kmin // A["mins"]
        dev, conf = hr(A)

        def rule(i, ei, ent):
            if i - ei != kb + 1:
                return False
            if gate is not None:
                g_op, g = gate
                gv = dev[i]
                if not np.isfinite(gv):
                    return False
                if g_op == ">" and not gv > g:
                    return False
                if g_op == "<=" and not gv <= g:
                    return False
            f = feature(A, ei, i - 1, ent, dev, conf)
            return bool(prob(model, f[cols])[0] > p)
        return rule
    return make_rule


def cell(cm, per, rng, un="사용자78"):
    return cm[(cm.기간 == per) & (cm.범위 == rng) & (cm.종목 == un)].iloc[0]


def summary(cm):
    return " | ".join(f"{per} {rng} {cell(cm, per, rng)['차이']:+.3f}(t {cell(cm, per, rng)['t']:+.2f}, 손절 {cell(cm, per, rng)['손절']})"
                      for per in ("early", "late") for rng in ("24시간", "정규장만"))


def day_diffs(mk, per, reg=False):
    t = run(load(per), mk, per, reg, syms=U78)
    b = base_trades(per, reg)
    b = b[b.sym.isin(U78)]
    days = sorted(set(b.day) | set(t.day))
    d = t.groupby("day").ret.sum().reindex(days, fill_value=0) - b.groupby("day").ret.sum().reindex(days, fill_value=0)
    return d.sort_values()


def htf_independent_check(nsym=8, seed=3):
    """5분봉에서 직접 40분봉(04:00 ET 기준)을 만들고 Wilder RSI14를 계산해 stop_lab.htf_rsi의 conf와 대조.
    conf[i] = '봉 i 시작 시각 이전에 끝난 마지막 40분봉'의 RSI. 미래 쪽 40분봉을 쓰면 불일치가 난다."""
    rng = np.random.default_rng(seed)
    bad = tot = 0
    dbad, dtot = [0], [0]
    for per in ("early", "late"):
        S = load(per)
        syms = list(S)
        for s in [syms[int(k)] for k in rng.choice(len(syms), nsym, replace=False)]:
            A = S[s]
            m5 = pd.read_parquet(f"{RAW}/5m_{s}.parquet")
            t5 = m5.index
            day0 = t5.normalize() + pd.Timedelta(hours=4)
            binstart = day0 + ((t5 - day0) // pd.Timedelta(minutes=40)) * pd.Timedelta(minutes=40)
            g = pd.Series(m5.Close.to_numpy(float), index=binstart).groupby(level=0).last()
            ends = (g.index + pd.Timedelta(minutes=40)).as_unit("ns").asi8
            cl = g.to_numpy(float)
            ch = np.diff(cl, prepend=np.nan)
            up, dn = np.maximum(ch, 0), np.maximum(-ch, 0)
            n = 14
            rs, AG, AL = (np.full(len(cl), np.nan) for _ in range(3))
            ag, al = up[1:n + 1].mean(), dn[1:n + 1].mean()
            AG[n], AL[n] = ag, al
            rs[n] = 100 - 100 / (1 + ag / al) if al > 0 else 100.0
            for k in range(n + 1, len(cl)):
                ag = (ag * (n - 1) + up[k]) / n
                al = (al * (n - 1) + dn[k]) / n
                AG[k], AL[k] = ag, al
                rs[k] = 100 - 100 / (1 + ag / al) if al > 0 else 100.0
            starts = g.index.as_unit("ns").asi8
            dev, conf = hr(A)
            pick = rng.integers(100, len(A["tk"]), 300)
            for i in pick:
                k = np.searchsorted(ends, A["tk"][i], side="right") - 1
                mine = rs[k] if k >= 0 else np.nan
                tot += 1
                if np.isfinite(mine) or np.isfinite(conf[i]):
                    bad += not np.isclose(mine, conf[i], atol=1e-6, equal_nan=True)
                # dev: 봉 i-1이 속한 40분봉 시작 이전에 끝난 마지막 40분봉 상태 + c[i-1]
                tp = A["idx"][i - 1]
                a0 = tp.normalize() + pd.Timedelta(hours=4)
                cur = (a0 + ((tp - a0) // pd.Timedelta(minutes=40)) * pd.Timedelta(minutes=40)).value
                k2 = np.searchsorted(ends, cur, side="right") - 1
                if k2 >= 0 and np.isfinite(AG[k2]):
                    ch2 = A["c"][i - 1] - cl[k2]
                    g2 = (AG[k2] * (n - 1) + max(ch2, 0)) / n
                    l2 = (AL[k2] * (n - 1) + max(-ch2, 0)) / n
                    md = 100 - 100 / (1 + g2 / l2) if l2 > 0 else 100.0
                else:
                    md = np.nan
                dtot[0] += 1
                if np.isfinite(md) or np.isfinite(dev[i]):
                    dbad[0] += not np.isclose(md, dev[i], atol=1e-6, equal_nan=True)
    return bad, tot, dbad[0], dtot[0]


def main():
    for per in ("early", "late"):
        load(per)
        base_trades(per, False)
        base_trades(per, True)
        context(5 if per == "early" else 3)
    say("[verify_g_model] 정의 문장만으로 재구현(연구자 코드 미사용). 기준 사용자78 24h early +27.12 late +17.04")
    bad, tot, db, dt = htf_independent_check()
    say(f"40분봉 RSI 독립 대조(5분봉→40분봉 직접 계산): conf[i](봉 i 시작 전에 끝난 40분봉 RSI) {tot}건 중 불일치 {bad}; "
        f"dev[i](봉 i-1이 속한 40분봉 전 완성 상태 + c[i-1]) {dt}건 중 불일치 {db}")

    # 학습 표·모델
    X, y, d = train_table(60, 0.02)
    days = sorted(set(base_trades("early", False).day))
    cut = int(round(len(days) * 2 / 3))
    trd = np.isin(d, days[:cut])
    say(f"학습 표 T60-fwd2: {len(y)}건 양성 {y.mean():.1%} (연구자: 학습 2969 + 검증 1096 = 4065건, 학습 양성 13.0% 검증 11.4%) "
        f"앞 {cut}일 {trd.sum()}건 / 뒤 {(~trd).sum()}건")
    allc = list(range(21))
    fcols = list(range(NF))
    full = fit(X[:, allc], y)
    front = fit(X[trd][:, allc], y[trd])
    pq = float(np.quantile(prob(front, X[trd][:, allc]), 0.88))
    say(f"앞 2/3 모델 학습 예측 상위 12% 분위 = {pq:.4f} (연구자 p 0.269)")
    say("전체 학습 계수(표준화): " + " ".join(f"{n}={w:+.2f}" for n, w in zip(["절편"] + NAMES, full["w"])))
    flush()

    # 최선안 재현
    mk = make(full, 0.269, allc)
    cm = compare(mk, "검증 T60-fwd2 FR p>0.269", show=True)
    say(cm.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    say("재현 사용자78: " + summary(cm))
    for per in ("early", "late"):
        for reg in (False, True):
            a = stop_audit(mk, per, reg_only=reg)
            say(f"stop_audit {per} {'정규장만' if reg else '24시간'} 사용자78: " +
                ", ".join(f"{k} {v:+.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in a.items()))
    # 손절 세션 구성
    for per in ("early", "late"):
        t = run(load(per), mk, per, False, syms=U78)
        st = t[t.why == "손절"]
        say(f"{per} 24h 손절 {len(st)}건 매수 세션 {st.ses.value_counts().to_dict()}")
    # 하루 의존
    for per in ("early", "late"):
        dd = day_diffs(mk, per)
        say(f"하루 의존 {per} 24h 사용자78: 합 {dd.sum():+.3f}, 나은 날 {int((dd > 1e-12).sum())} 나쁜 날 {int((dd < -1e-12).sum())}, "
            f"최고 {dd.index[-1]} {dd.iloc[-1]:+.3f} → 빼면 {dd.sum() - dd.iloc[-1]:+.3f}; 상위 2일 빼면 {dd.sum() - dd.iloc[-2:].sum():+.3f}; "
            f"최저 {dd.index[0]} {dd.iloc[0]:+.3f}")
    flush()

    # 이웃 매개변수
    say("\n=== 이웃(사용자78 차이: early 24h / early 정규 / late 24h / late 정규) ===")
    rows = []

    def row(lab, m_):
        c_ = compare(m_, lab, show=False)
        vals = [cell(c_, "early", "24시간")["차이"], cell(c_, "early", "정규장만")["차이"],
                cell(c_, "late", "24시간")["차이"], cell(c_, "late", "정규장만")["차이"]]
        st = [cell(c_, "early", "24시간")["손절"], cell(c_, "late", "24시간")["손절"]]
        rows.append((lab, *vals, *st))
        say(f"{lab:<44} e24 {vals[0]:+.3f} e정규 {vals[1]:+.3f} l24 {vals[2]:+.3f} l정규 {vals[3]:+.3f} (손절 e{st[0]}/l{st[1]}) ({time.time() - T0:.0f}s)")
        flush()

    # (a) 문턱 p 한 단계 위·아래(분위 상위 9%·15%, 그리고 격자 이웃 25%·5%)
    for qq in (0.05, 0.09, 0.15, 0.25):
        pp = float(np.quantile(prob(front, X[trd][:, allc]), 1 - qq))
        row(f"p 이웃: 상위 {qq:.0%} (p>{pp:.3f})", make(full, pp, allc))
    # (b) 판단 시점 45·75분
    for km in (45, 75):
        Xk, yk, dk = train_table(km, 0.02)
        tk_ = np.isin(dk, days[:cut])
        fk, frk = fit(Xk, yk), fit(Xk[tk_], yk[tk_])
        pk = float(np.quantile(prob(frk, Xk[tk_]), 0.88))
        row(f"시점 이웃: T{km} fwd2 상위12% (p>{pk:.3f})", make(fk, pk, allc, kmin=km))
    # (c) 라벨 손실 폭 1.5%·2.5%
    for ls in (0.015, 0.025):
        Xl, yl, dl = train_table(60, ls)
        tl_ = np.isin(dl, days[:cut])
        fl, frl = fit(Xl, yl), fit(Xl[tl_], yl[tl_])
        pl = float(np.quantile(prob(frl, Xl[tl_]), 0.88))
        row(f"라벨 이웃: fwd{ls:.1%} 상위12% (p>{pl:.3f})", make(fl, pl, allc))
    # (d) 40분봉 RSI 조건 g 이웃 + 반대 조건 + 40분봉 특징 뺀 F
    for g in (25, 30, 35):
        row(f"조건: 최선안 & dev40>{g}", make(full, 0.269, allc, gate=(">", g)))
    row("반대 조건: 최선안 & dev40<=30", make(full, 0.269, allc, gate=("<=", 30)))
    fF, frF = fit(X[:, fcols], y), fit(X[trd][:, fcols], y[trd])
    pF = float(np.quantile(prob(frF, X[trd][:, fcols]), 0.88))
    row(f"조건 없음 대조: F(40분봉 특징 뺌) 상위12% (p>{pF:.3f})", make(fF, pF, fcols))
    say(f"\n시험(검증) 규칙 수: 1 + {len(rows)}")
    say(f"총 {time.time() - T0:.0f}s")
    flush()


if __name__ == "__main__":
    main()
