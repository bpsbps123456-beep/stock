"""반박 검증 — 결합 모델(시간순 학습) 계열 최선안 '전체-더잃음2%@60 p>0.23'을 정의 문장만 보고 다시 만든다(2026-10-07).

정의(연구자 보고 요약): 매수 봉 ei, 체크포인트 cp = ei + 60/mins 봉. cp 종가까지 알 수 있는 15개 특징으로
early 기준 거래(24시간, 137종목, cp까지 들고 있던 것)에서 P[(최종 수익 - cp+1 시가 매도 수익) <= -2%]를
L2 로지스틱(λ=1, 절편 제외, 특징은 학습 표본 평균·표준편차로 표준화 후 ±5 자름)으로 학습하고,
P > 0.23이면 봉 i = cp+1 시가에 판다(그 봉에서 한 번만).

이 파일은 연구자 코드를 쓰지 않는다. 특징은 규칙이 불릴 때(i = cp+1) 그 자리에서 A[...][:cp+1]만 잘라 계산한다
(미리 계산한 행렬 없음 → 구조상 i 이후 값을 볼 수 없다). 로지스틱은 scipy L-BFGS로 푼다.
사용: py stops/verify_model.py → ../results/stops/verify_model.txt
"""
import os, sys, time
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from stop_lab import load, compare, stop_audit, base_trades, run, RAW, UNIV, GAP  # noqa: E402
from events import bars  # noqa: E402

OUT = os.path.normpath(os.path.join(HERE, "..", "..", "results", "stops", "verify_model.txt"))
U78 = UNIV["사용자78"]
NAMES = ["rk", "lowk", "rsi_cp", "rsi_e", "slope60", "dev_cp", "gap_e", "pre", "aft", "tod",
         "prev_ret", "today_ret", "vol", "qqq", "mkt"]
REPORTED_W = dict(절편=-2.22, rk=-0.51, lowk=-0.12, rsi_cp=+0.15, rsi_e=+0.01, slope60=+0.44, dev_cp=-0.10, gap_e=-0.13,
                  pre=+0.27, aft=-0.02, tod=-0.67, prev_ret=+0.03, today_ret=-0.33, vol=-0.10, qqq=+0.29, mkt=-0.16)
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def flush():
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(LOG) + "\n")


# ------------------------------------------------------------------ 기간별 맥락(QQQ, 137종목 평균 이격, 종목별 보조)
_CTX = {}


def period_of(mins):
    return "early" if mins == 5 else "late"


def ctx(period):
    if period in _CTX:
        return _CTX[period]
    S = load(period)
    mins = next(iter(S.values()))["mins"]
    q = bars(RAW, "QQQ", mins)
    q_t = q.index.as_unit("ns").asi8
    q_c = q.Close.to_numpy(dtype=float)
    # 137종목 이격 평균: 격자 시각 t마다 각 종목의 '시작 시각 <= t'인 마지막 봉 값(30분 이내만), 같은 시각 평균
    grid = np.unique(np.concatenate([A["tk"] for A in S.values()]))
    tot = np.zeros(len(grid))
    cnt = np.zeros(len(grid))
    lim = 30 * 60 * 10 ** 9
    for A in S.values():
        dev = np.clip(A["c"] / A["ema"] - 1, -0.1, 0.1)
        j = np.searchsorted(A["tk"], grid, side="right") - 1
        jj = np.clip(j, 0, None)
        ok = (j >= 0) & (grid - A["tk"][jj] <= lim) & np.isfinite(dev[jj])
        tot[ok] += dev[jj][ok]
        cnt[ok] += 1
    mk = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    sym = {}
    for s, A in S.items():
        d = pd.read_parquet(f"{RAW}/1d_{s}.parquet").Close
        if isinstance(d, pd.DataFrame):
            d = d.iloc[:, 0]
        dd = pd.to_datetime(d.index).date
        ri = np.flatnonzero(A["reg"])
        sym[s] = dict(dd=np.array(dd, dtype=object), dc=d.to_numpy(dtype=float), ri=ri)
    _CTX[period] = dict(q_t=q_t, q_c=q_c, grid=grid, mk=mk, sym=sym, mins=mins)
    return _CTX[period]


def asof(t_arr, v_arr, t):
    j = np.searchsorted(t_arr, t, side="right") - 1
    return v_arr[j] if j >= 0 else np.nan


def feat(A, ei, cp, ent):
    """cp 종가까지(인덱스 <= cp)만 쓰는 15개 특징."""
    C = ctx(period_of(A["mins"]))
    Y = C["sym"][A["sym"]]
    mins = A["mins"]
    o, l, c, r, ema, v, tk = (A[k][:cp + 1] for k in ("o", "l", "c", "r", "ema", "v", "tk"))
    f = np.full(15, np.nan)
    f[0] = c[cp] / ent - 1
    f[1] = l[ei:cp + 1].min() / ent - 1
    f[2] = r[cp]
    f[3] = r[ei - 1]
    f[4] = ema[cp] / ema[max(cp - 60 // mins, 0)] - 1
    f[5] = c[cp] / ema[cp] - 1
    f[6] = ent / ema[ei - 1] - 1
    ses = A["ses"][ei]
    f[7] = 1.0 if ses == "프리" else 0.0
    f[8] = 1.0 if ses == "애프터" else 0.0
    f[9] = A["mod"][ei] / 60.0
    D = A["day"][ei]
    dd, dc = Y["dd"], Y["dc"]
    j = int(np.searchsorted(dd, D, side="left")) - 1  # D 미만 마지막 거래일
    if j >= 1:
        f[10] = np.clip(dc[j] / dc[j - 1] - 1, -0.3, 0.3)
    if j >= 0:
        f[11] = ent / dc[j] - 1
    # 거래량: 정규장 매수만
    f[12] = 0.0
    if A["reg"][ei]:
        ri = Y["ri"]
        k = int(np.searchsorted(ri, ei, side="left"))
        prev = ri[max(0, k - 390 // mins):k]
        if len(prev):
            base = v[prev].mean()
            win = v[ei:cp + 1].mean()
            if base > 0:
                f[12] = np.clip(np.log(win / base), -3, 3) if win > 0 else -3.0
    q1 = asof(C["q_t"], C["q_c"], tk[cp])
    q0 = asof(C["q_t"], C["q_c"], tk[ei - 1])
    f[13] = q1 / q0 - 1
    f[14] = asof(C["grid"], C["mk"], tk[cp])
    return f


# ------------------------------------------------------------------ 학습
def train(k=60, loss=0.02, lam=1.0, clip_today=False):
    S = load("early")
    b = base_trades("early", False)
    X, y = [], []
    for x in b.itertuples():
        A = S[x.sym]
        kb = k // A["mins"]
        cp = x.ei + kb
        if x.xi <= cp:
            continue
        X.append(feat(A, x.ei, cp, x.ent))
        y.append(1.0 if (x.ret - (A["o"][cp + 1] / x.ent - 1)) <= -loss else 0.0)
    X, y = np.array(X), np.array(y)
    if clip_today:
        X[:, 11] = np.clip(X[:, 11], -0.3, 0.3)
    mu = np.nanmean(X, 0)
    sd = np.nanstd(X, 0)
    sd[~(sd > 0)] = 1.0
    Z = np.clip((np.where(np.isnan(X), mu, X) - mu) / sd, -5, 5)
    Z1 = np.c_[np.ones(len(Z)), Z]

    def obj(w):
        z = Z1 @ w
        f = np.sum(np.logaddexp(0, z) - y * z) + 0.5 * lam * np.sum(w[1:] ** 2)
        g = Z1.T @ (expit(z) - y)
        g[1:] += lam * w[1:]
        return f, g
    res = minimize(obj, np.zeros(Z1.shape[1]), jac=True, method="L-BFGS-B", options=dict(maxiter=5000, gtol=1e-9))
    return dict(w=res.x, mu=mu, sd=sd, n=len(y), pos=y.mean(), k=k, clip_today=clip_today)


def prob(m, x):
    x = x.copy()
    if m["clip_today"]:
        x[11] = np.clip(x[11], -0.3, 0.3)
    z = np.clip((np.where(np.isnan(x), m["mu"], x) - m["mu"]) / m["sd"], -5, 5)
    return float(expit(m["w"][0] + z @ m["w"][1:]))


def factory(m, p):
    k = m["k"]

    def make_rule(A):
        kb = k // A["mins"]

        def rule(i, ei, ent):
            if i - ei != kb + 1:
                return False
            return prob(m, feat(A, ei, i - 1, ent)) > p
        return rule
    return make_rule


def cell(cm, per, rng, un="사용자78"):
    return cm[(cm.기간 == per) & (cm.범위 == rng) & (cm.종목 == un)].iloc[0]


def day_dep(mk, period, reg_only=False):
    t = run(load(period), mk, period, reg_only)
    b = base_trades(period, reg_only)
    t, b = t[t.sym.isin(U78)], b[b.sym.isin(U78)]
    days = sorted(set(t.day) | set(b.day))
    d = t.groupby("day").ret.sum().reindex(days, fill_value=0) - b.groupby("day").ret.sum().reindex(days, fill_value=0)
    top = d.idxmax()
    st = t[t.why == "손절"]
    return d, top, st


def main():
    t0 = time.time()
    for per in ("early", "late"):
        ctx(per)
        base_trades(per, False)
        base_trades(per, True)
    say(f"맥락 준비 {time.time() - t0:.0f}s")

    m = train(60, 0.02, 1.0)
    say(f"\n[학습] early 24시간 137종목, cp(60분)까지 들고 있던 기준 거래 {m['n']}건(보고 4065), 양성 {m['pos']:.1%}")
    say("계수(내 구현 / 보고): " + " ".join(
        f"{nm} {w:+.2f}/{REPORTED_W[nm]:+.2f}" for nm, w in zip(["절편"] + NAMES, m["w"])))
    flush()

    say("\n=== 최선안 재현: 전체-더잃음2%@60 p>0.23 (early 칸 = 학습 안쪽, late 칸 = 전방) ===")
    mk = factory(m, 0.23)
    cm = compare(mk, "재현 p>0.23", show=False)
    show = cm[["기간", "범위", "종목", "건수", "합", "차이", "t", "손절", "최악", "차이_10월6일"]].copy()
    say(show.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    flush()

    say("\n=== stop_audit 사용자78 ===")
    for per in ("early", "late"):
        for reg in (False, True):
            a = stop_audit(mk, per, reg_only=reg)
            say(f"{per} {'정규장만' if reg else '24시간'}: " + ", ".join(
                f"{k} {v:+.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in a.items()))
    flush()

    say("\n=== 하루 의존(사용자78, 날짜별 합 차이) ===")
    for per in ("early", "late"):
        for reg in (False, True):
            d, top, st = day_dep(mk, per, reg)
            ses = st.ses.value_counts().to_dict()
            say(f"{per} {'정규장만' if reg else '24시간'}: 차이 {d.sum():+.3f}, 최대 이득일 {top} {d[top]:+.3f}, "
                f"그날 빼면 {d.sum() - d[top]:+.3f}; 최대 손해일 {d.idxmin()} {d.min():+.3f}; "
                f"나은날 {(d > 1e-12).sum()} 나쁜날 {(d < -1e-12).sum()}; 손절 세션 {ses}")
            if per == "early" and not reg:
                srt = d.sort_values(ascending=False)
                say("  early 24h 상위 5일: " + ", ".join(f"{k} {v:+.3f}" for k, v in srt.head(5).items())
                    + f" | 상위 3일 빼면 {d.sum() - srt.head(3).sum():+.3f}")
    flush()

    say("\n=== 이웃 매개변수(사용자78 차이; early는 학습 안쪽) ===")
    rows = []
    variants = [("p 0.20", dict(k=60, loss=0.02, lam=1.0), 0.20), ("p 0.26", dict(k=60, loss=0.02, lam=1.0), 0.26),
                ("k 30분", dict(k=30, loss=0.02, lam=1.0), 0.23), ("k 90분", dict(k=90, loss=0.02, lam=1.0), 0.23),
                ("손실 1.5%", dict(k=60, loss=0.015, lam=1.0), 0.23), ("손실 2.5%", dict(k=60, loss=0.025, lam=1.0), 0.23),
                ("λ 0.3", dict(k=60, loss=0.02, lam=0.3), 0.23), ("λ 3", dict(k=60, loss=0.02, lam=3.0), 0.23),
                ("today_ret ±30% 자름", dict(k=60, loss=0.02, lam=1.0, clip_today=True), 0.23)]
    cache = {}
    for lab, kw, p in variants:
        key = tuple(sorted(kw.items()))
        if key not in cache:
            cache[key] = train(**kw) if key != tuple(sorted(dict(k=60, loss=0.02, lam=1.0).items())) else m
        mm = cache[key]
        c2 = compare(factory(mm, p), lab, show=False)
        r = dict(이웃=lab)
        for per in ("early", "late"):
            for rng in ("24시간", "정규장만"):
                x = cell(c2, per, rng)
                r[f"{per}_{rng}"] = x["차이"]
                r[f"{per}_{rng}_손절"] = x["손절"]
        rows.append(r)
        say(f"{lab:<16} early 24h {r['early_24시간']:+.2f}({r['early_24시간_손절']}) 정규 {r['early_정규장만']:+.2f}"
            f"({r['early_정규장만_손절']}) | late 24h {r['late_24시간']:+.2f}({r['late_24시간_손절']}) "
            f"정규 {r['late_정규장만']:+.2f}({r['late_정규장만_손절']})  ({time.time() - t0:.0f}s)")
        flush()
    say(f"\n시험한 규칙 수(이 검증): 최선안 1 + 이웃 {len(variants)} = {1 + len(variants)}")
    say(f"끝 ({time.time() - t0:.0f}s)")
    flush()


if __name__ == "__main__":
    main()
