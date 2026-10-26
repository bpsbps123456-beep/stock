"""큰 봉 RSI 과열이면 3분봉 이격 매수를 조심한다 — 조합 연구(2026-10-07 사용자 가설: 160분·일봉 RSI 70 이상).
기준 신호는 그대로다(3분봉 EMA60 -2% 매수, RSI14>70 또는 다음 아침 8시 매도). 앞 기간 7/14~9/8은 5분봉(EMA36·RSI9) 근사.
조합(시험 범위):
  큰 봉   60·120·160·240분(프리·애프터 포함, 04:00 ET 기준), 일봉(정규장)
  RSI     길이 7·14·21, 기준 60·65·70·75·80
  값      dev(지금 가격을 넣은 진행 중 값, 매수선 가격 기준) / conf(마지막 완성 봉)
  조심    안 삼 / 매수선을 -3·-4·-5%로 내림
  범위    24시간 매매(필터 전 세션) / 정규장에서만 매수
  비교    지금 지표, 전일 +5% 다음 날 정규장 금지, 둘의 AND·OR(160분·일봉, 70, 길이 14, 안 삼)
지표: 거래 수, 평균, 승률, 최악, 합, 날짜별 합의 짝 비교(t), 10/6 유전체 5종목, TEM 10/6.
사용: py htf_study.py <raw 폴더> <출력 폴더>"""
import os, sys, time
from datetime import date
import numpy as np, pandas as pd
from numba import njit
from events import bars, prep
from fetch import universe, user_universe
from htf import rma_state, _rsi_from, intraday_bars
from rules import hot_days

PERIODS = {"앞 7/14~9/8 (5분봉 근사)": (5, 36, 9, date(2026, 7, 14), date(2026, 9, 8)),
           "뒤 9/10~10/6 (3분봉)": (3, 60, 14, date(2026, 9, 10), date(2026, 10, 6))}
TFS = [60, 120, 160, 240, "D"]
NS = [7, 14, 21]
THS = [60, 65, 70, 75, 80]
ACTS = {"안 삼": 9.0, "-3%": 0.03, "-4%": 0.04, "-5%": 0.05}
BIO = {"TEM", "TWST", "TXG", "GRAL", "SDGR"}
NOBUY = 9.0


@njit(cache=True)
def sim_nb(o, l, r, ema, tk, dead_of, gap, x):
    n = len(o)
    oe = np.empty(n, np.int64); ox = np.empty(n, np.int64); oen = np.empty(n); opx = np.empty(n)
    k = 0; hold = False; ent = 0.0; ei = 0; dead = 0
    for i in range(61, n):
        if hold:
            if i - 1 >= ei and r[i - 1] > x:
                oe[k] = ei; ox[k] = i; oen[k] = ent; opx[k] = o[i]; k += 1; hold = False
            elif tk[i] >= dead:
                oe[k] = ei; ox[k] = i; oen[k] = ent; opx[k] = o[i]; k += 1; hold = False
        else:
            g = gap[i]
            if g < 1.0:
                lvl = ema[i - 1] * (1.0 - g)
                if l[i] <= lvl:
                    ent = min(o[i], lvl); ei = i; hold = True; dead = dead_of[i]
    return oe[:k], ox[:k], oen[:k], opx[:k]


def base_arrays(raw, s, mins, el, rl):
    d = prep(bars(raw, s, mins), el, rl)
    idx = d.index
    kst = idx.tz_convert("Asia/Seoul")
    t8 = kst.normalize() + pd.Timedelta(hours=8)
    dead = np.where(kst < t8, t8.as_unit("ns").asi8, (t8 + pd.Timedelta(days=1)).as_unit("ns").asi8)
    ema = d.ema.values
    p2 = np.minimum(d.o.values, np.r_[np.nan, ema[:-1]] * 0.98)
    return dict(d=d, o=d.o.values.astype(float), l=d.l.values.astype(float), r=d.r.values.astype(float), ema=ema,
                tk=idx.as_unit("ns").asi8, dead=dead.astype(np.int64), reg=(d.ses.values == "정규"), p2=p2,
                hot=hot_days(raw, s, d), day=np.asarray(d.date))


def htf_rsi_all(raw, s, A):
    """{(tf, n, kind): RSI 배열} — 기준봉마다, dev는 매수선 가격 p2 기준."""
    out = {}
    d = A["d"]
    t, p2 = d.index, A["p2"]
    m5 = pd.read_parquet(f"{raw}/5m_{s}.parquet")
    for tf in TFS:
        if tf == "D":
            d1 = pd.read_parquet(f"{raw}/1d_{s}.parquet")
            days = pd.to_datetime(d1.index).date
            c = d1.Close.values.astype(float)
            today = A["day"]
            jt = np.searchsorted(days, today, side="left")
            has_today = (jt < len(days)) & (days[np.minimum(jt, len(days) - 1)] == today)
            jy = jt - 1
            ses = d.ses.values
            jc = np.where((ses == "애프터") & has_today, jt, jy)
            for n in NS:
                g, l, r = rma_state(c, n)
                conf = np.where(jc >= 0, r[np.maximum(jc, 0)], np.nan)
                devr = np.where(jy >= 0, _rsi_from(g[np.maximum(jy, 0)], l[np.maximum(jy, 0)], c[np.maximum(jy, 0)], p2, n), np.nan)
                out[(tf, n, "dev")] = np.where(ses == "정규", devr, conf)
                out[(tf, n, "conf")] = conf
        else:
            hs, hc = intraday_bars(m5, tf)
            anchor = t.normalize() + pd.Timedelta(hours=4)
            off = ((t - anchor) // pd.Timedelta(minutes=tf)).astype(np.int64)
            cur = anchor + pd.to_timedelta(off * tf, unit="min")
            j = np.searchsorted(hs.as_unit("ns").asi8, cur.as_unit("ns").asi8, side="left") - 1
            ok, jj = j >= 0, np.maximum(j, 0)
            for n in NS:
                g, l, r = rma_state(hc.astype(float), n)
                out[(tf, n, "dev")] = np.where(ok, _rsi_from(g[jj], l[jj], hc[jj], p2, n), np.nan)
                out[(tf, n, "conf")] = np.where(ok, r[jj], np.nan)
    return out


def run_cfg(S, gap_fn, lo, hi):
    """모든 종목에 같은 규칙을 돌려 거래표를 만든다. gap_fn(A, s) → 봉마다 요구 이격(9.0 = 안 삼)."""
    rows = []
    for s, A in S.items():
        gap = gap_fn(A, s)
        e, x, en, px = sim_nb(A["o"], A["l"], A["r"], A["ema"], A["tk"], A["dead"], gap, 70.0)
        if len(e):
            dd = A["day"][e]
            m = (dd >= lo) & (dd <= hi)
            for ei, xi, a, b, dy in zip(e[m], x[m], en[m], px[m], dd[m]):
                rows.append((s, dy, A["reg"][ei], b / a - 1, A["d"].index[ei]))
    return pd.DataFrame(rows, columns=["sym", "day", "reg", "ret", "t"])


def metrics(t, base, syms, days):
    t, base = t[t.sym.isin(syms)], base[base.sym.isin(syms)]
    a = base.groupby("day").ret.sum().reindex(days, fill_value=0)
    b = t.groupby("day").ret.sum().reindex(days, fill_value=0)
    dlt = b - a
    tt = dlt.mean() / (dlt.std(ddof=1) / np.sqrt(len(dlt))) if dlt.std(ddof=1) > 0 else 0.0
    bio = t[(t.day == date(2026, 10, 6)) & t.sym.isin(BIO)].ret.sum()
    tem = t[(t.day == date(2026, 10, 6)) & (t.sym == "TEM")].ret
    return dict(건수=len(t), 평균=t.ret.mean(), 승률=(t.ret > 0).mean(), 최악=t.ret.min(), 합=t.ret.sum(),
                차이=dlt.sum(), t=tt, 나은날=int((dlt > 1e-12).sum()), 나쁜날=int((dlt < -1e-12).sum()),
                유전체5=bio, TEM=tem.sum() if len(tem) else np.nan)


if __name__ == "__main__":
    raw, outdir = sys.argv[1], sys.argv[2]
    U = user_universe()
    UNIV = {"사용자 78": U["합집합"], "참고 127": list(universe())}
    syms = sorted(set(UNIV["사용자 78"]) | set(UNIV["참고 127"]))
    rows, ev_rows = [], []
    for pname, (mins, el, rl, lo, hi) in PERIODS.items():
        t0 = time.time()
        S = {}
        for s in syms:
            try:
                S[s] = base_arrays(raw, s, mins, el, rl)
            except FileNotFoundError:
                pass
        days = sorted({d for A in S.values() for d in set(A["day"]) if lo <= d <= hi})
        R = {s: htf_rsi_all(raw, s, A) for s, A in S.items()}
        keys = list(R[next(iter(R))])
        for scope in ("24시간", "정규장만"):
            reg_only = scope == "정규장만"
            basegap = lambda A, s: np.where(A["reg"] | (not reg_only), 0.02, NOBUY)
            base = run_cfg(S, basegap, lo, hi)
            hot = run_cfg(S, lambda A, s: np.where(A["hot"], NOBUY, basegap(A, s)), lo, hi)
            for uname, us in UNIV.items():
                rows.append(dict(기간=pname, 범위=scope, 종목=uname, 규칙="지금 지표", **metrics(base, base, us, days)))
                rows.append(dict(기간=pname, 범위=scope, 종목=uname, 규칙="전일 +5% 금지", **metrics(hot, base, us, days)))
            if scope == "24시간":                                # 지금 지표의 거래마다 큰 봉 RSI를 붙인다(사건 분석용)
                pos = {s: pd.Series(np.arange(len(A["d"])), index=A["d"].index) for s, A in S.items()}
                ii = [pos[s].loc[tt] for s, tt in zip(base.sym, base.t)]
                ev = base.copy()
                for key in keys:
                    ev["|".join(map(str, key))] = [R[s][key][i] for s, i in zip(base.sym, ii)]
                ev["기간"] = pname
                ev_rows.append(ev)
            for (tf, n, kind) in keys:
                for th in THS:
                    for aname, av in ACTS.items():
                        if n != 14 and not (tf in (160, "D") and th == 70 and aname == "안 삼"):
                            continue
                        fn2 = lambda A, s: np.where(basegap(A, s) >= 1, NOBUY, np.where(np.nan_to_num(R[s][(tf, n, kind)], nan=-1) >= th, av, 0.02))
                        tr = run_cfg(S, fn2, lo, hi)
                        for uname, us in UNIV.items():
                            rows.append(dict(기간=pname, 범위=scope, 종목=uname, 규칙=f"{tf}{'분' if tf != 'D' else '(일봉)'} RSI{n} {kind} ≥{th} → {aname}",
                                             큰봉=str(tf), 길이=n, 값=kind, 기준=th, 조심=aname, **metrics(tr, base, us, days)))
                if n == 14 and tf in (160, "D"):
                    for mode in ("AND", "OR"):
                        def fn3(A, s, mode=mode):
                            f = np.nan_to_num(R[s][(tf, 14, kind)], nan=-1) >= 70
                            h = A["hot"]
                            flag = (f & h) if mode == "AND" else (f | h)
                            return np.where(basegap(A, s) >= 1, NOBUY, np.where(flag, NOBUY, 0.02))
                        tr = run_cfg(S, fn3, lo, hi)
                        for uname, us in UNIV.items():
                            rows.append(dict(기간=pname, 범위=scope, 종목=uname, 규칙=f"전일 +5% {mode} {tf}{'분' if tf != 'D' else '(일봉)'} RSI14 {kind} ≥70 → 안 삼",
                                             큰봉=str(tf), 길이=14, 값=kind, 기준=70, 조심=f"전일+5% {mode}", **metrics(tr, base, us, days)))
        print(f"{pname} 끝 {time.time() - t0:.0f}초", file=sys.stderr)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(outdir, "htf_rules.csv"), index=False, encoding="utf-8-sig")
    pd.concat(ev_rows).to_parquet(os.path.join(outdir, "htf_events.parquet"))
    print("저장:", len(res), "행")
