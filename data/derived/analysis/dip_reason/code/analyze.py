"""'이유 있는 하락' 신호별 이격 매수 성적. 같은 날 거래끼리 묶어(날짜 군집) t값을 낸다.
창: 3분봉 9/9~10/6(정확한 규칙), 5분봉 앞 구간 7/14~9/8(겹치지 않는 근사 검증).
주의: 전체를 섞으면 세션 구성(애프터 꼬리 체결 +2~3%) 때문에 가짜 효과가 생긴다. 판단은 세션 안 비교로 한다."""
import sys
import numpy as np, pandas as pd

FEATS = {
    "grp_dev": "같은 그룹 동반 하락(그룹 보라선 대비 중앙값)",
    "qqq_dev": "QQQ 보라선 대비",
    "rvol": "최근 30분 거래량 배수(정규장)",
    "d_prev": "전일 종가 대비 하락",
    "d_high": "오늘 고점 대비 하락",
    "speed": "최근 30분 하락 속도",
    "run20": "직전 20일 상승률",
}


def cl_t(x, y, g):
    """y = a + b*x 의 b와 날짜 군집 강건 t값."""
    m = ~(np.isnan(x) | np.isnan(y))
    x, y, g = x[m], y[m], g[m]
    X = np.column_stack([np.ones_like(x), (x - x.mean()) / x.std()])
    XtX = np.linalg.inv(X.T @ X)
    b = XtX @ X.T @ y
    u = y - X @ b
    meat = np.zeros((2, 2))
    for k in np.unique(g):
        s = X[g == k].T @ u[g == k]
        meat += np.outer(s, s)
    G, N = len(np.unique(g)), len(y)
    V = XtX @ meat @ XtX * G / (G - 1) * (N - 1) / (N - 2)
    return b[1], b[1] / np.sqrt(V[1, 1]), G


def diff_t(y, flag, g):
    """flag 거래와 나머지의 평균 차이, 날짜 군집 t값."""
    return cl_t(flag.astype(float), y, g)


def table(e, f, q=5):
    x = e[f]
    ok = x.notna()
    b = pd.qcut(x[ok], q, labels=False, duplicates="drop")
    rows = []
    for k, v in e[ok].groupby(b):
        rows.append(dict(구간=f"Q{k + 1}", 범위=f"{v[f].min():+.3f}~{v[f].max():+.3f}", 건수=len(v),
                         평균=f"{v.ret.mean():+.2%}", 중앙=f"{v.ret.median():+.2%}", 승률=f"{(v.ret > 0).mean():.0%}",
                         최악=f"{v.ret.min():+.1%}", 평균MAE=f"{v.mae.mean():+.1%}"))
    slope, t, G = cl_t(x.values.astype(float), e.ret.values, e.date.astype(str).values)
    return pd.DataFrame(rows), slope, t, G


def load(path, lo=None, hi=None):
    e = pd.read_parquet(path)
    e["rvol"] = e.rvol.replace([np.inf, -np.inf], np.nan)
    d = pd.to_datetime(e.date)
    if lo:
        e = e[d >= lo]
    if hi:
        e = e[pd.to_datetime(e.date) <= hi]
    return e.reset_index(drop=True)


if __name__ == "__main__":
    R = sys.argv[1]
    W = {"3분봉 9/9~10/6": load(f"{R}/ev3.parquet"),
         "5분봉 7/14~9/8": load(f"{R}/ev5.parquet", hi="2026-09-08")}
    pd.set_option("display.width", 220)
    for name, e in W.items():
        print(f"\n===== {name}: {len(e)}건 · 평균 {e.ret.mean():+.2%} · 승률 {(e.ret > 0).mean():.0%} · 최악 {e.ret.min():+.1%}")
        for f, lab in FEATS.items():
            tb, s, t, G = table(e, f)
            print(f"\n[{f}] {lab} — 1표준편차당 거래 수익 {s:+.2%} (날짜 군집 t {t:+.2f}, {G}일)")
            print(tb.to_string(index=False))
        for ses in ("정규", "프리"):
            v = e[e.ses == ses]
            print(f"\n-- {ses} 안에서만 ({len(v)}건, 평균 {v.ret.mean():+.2%}): 1표준편차 기울기 t · 5분위 평균 Q1 … Q5")
            for f in FEATS:
                x = v[f].astype(float)
                if x.notna().sum() < 100:
                    continue
                _, t, _ = cl_t(x.values, v.ret.values, v.date.astype(str).values)
                m = v.groupby(pd.qcut(x, 5, labels=False, duplicates="drop")).ret.mean()
                print(f"   {f:8s} t {t:+5.2f} | {m.iloc[0]:+.2%} … {m.iloc[-1]:+.2%}")
