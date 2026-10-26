"""테마 급등 뒤 되돌림 사건 연구(2026-10-10 사용자: 10/9 LITE·COHR 광 테마 급등 뒤 반납, LITE는 회복·COHR는 못 함.
3분봉 모양으로 미리 알 수 있었나). 이격매수 거래와 별개의 사건 연구다.

사건 정의(시험 전에 고정, 변형은 이 3개뿐):
  정규장(09:30~16:00 ET) 봉만 본다. P = 전날 정규장 종가(일봉 Close). H_t = 그날 정규장 봉 t까지 최고가(누적 고가). R_t = H_t - P.
  T = H_t/P - 1 >= 급등 문턱 이고, 봉 t 종가 c_t가 (H_t - c_t)/R_t >= 반납 문턱 인 첫 봉. 종목·날짜당 1건.
  V0(주): 급등 +6%, 반납 70%   V1: 급등 +8%, 반납 70%   V2: 급등 +6%, 반납 50%
결과(시험 전에 고정):
  Y1 = 그날 공식 종가(일봉 Close) / c_T - 1   (T 봉 종가에 사서 그날 종가에 판 수익)
  Y2 = 종가 >= P + 0.5 R_T 이면 1   (종가에 상승분의 50% 이상을 되찾았나. T에서는 30% 이하(V0) 또는 50% 이하(V2)였다)
특성(모두 T 봉 종가 시점에 알 수 있는 값. T 이후 봉은 쓰지 않는다):
  spike    H_T/P - 1 (급등 크기)
  t_peak   고점 봉 시작이 개장 몇 분 뒤인가
  pre      (정규장 시가 - P)/R_T  (상승분 중 프리장에 이미 오른 비중)
  gb_min   고점 봉 → T 봉 분 수(반납 속도, 작을수록 빠름)
  big15    고점부터 T까지 가장 큰 15분 하락 / R_T  (반납 중 가장 큰 한 봉. 3분·5분 단위를 맞추려고 15분 창으로 잰다)
  low_p    고점~T 최저가의 전날 종가 대비 위치 (L - P)/R_T  (음수 = 전날 종가 아래로 뚫음)
  low_pm   L / 프리장 최저 종가 - 1  (프리장 한 분짜리 꼬리를 피하려고 저가 대신 봉 종가 최솟값)
  low_o    L / 정규장 시가 - 1
  th_n     같은 소분류 다른 종목 중 T 시점까지 같은 급등 문턱을 넘은 수(테마 동반)
  th_top   테마 안(th_n>=1) 급등 크기 1등이었나
  th_gb    테마 안 내 반납 비율 - 동반 종목 반납 비율 중앙값(T 시점, 음수 = 덜 반납)
  rvol     개장~T 누적 거래량 / 지난 20일 하루 평균 거래량
  v_ratio  반납 구간(고점 다음 봉~T) 분당 거래량 / 상승 구간(개장~고점 봉) 분당 거래량
  tr5      전날까지 5거래일 수익률
  prev     전날 수익률
  hi120    H_T / 지난 120거래일 최고 종가 - 1 (양수 = 장중 신고가 돌파)
통제: atr20(지난 20일 (고가-저가)/종가 평균), vol20(지난 20일 일간 수익률 표준편차), early30(T가 10:00 ET 전).
표본: 앞 = 7/14~9/8 5분봉(raw/5m_), 뒤 = 9/10~10/8 3분봉(raw_1009/1m_ → events.bars 3분). 고르기는 앞, 확인은 뒤.
종목: stop_lab.ALL 137개에서 GOOG(GOOGL 중복)·SOXX(ETF)를 뺀 135개.
소분류: watchlist.AI_UNIVERSE 소분류. 바이오 12개+TEM은 '바이오·유전체', COIN·CRCL·HOOD·MSTR은 '코인·핀테크'(이 연구에서 추가), 나머지는 단독.
저장: ../results/pat_spike.txt(요약), pat_spike_events.csv(사건·특성), pat_spike_tests.csv(모든 시험), pat_spike_cands.csv(후보 판정),
      pat_spike_common.csv(4c 같은 시각 비교 표본)
사용: py pat_spike.py"""
import os, re, sys
from datetime import date
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "../../dip_reason/code"))
sys.path.insert(0, os.path.join(HERE, "../../../../.."))
from events import bars, prep
from fetch import BIO
from stop_lab import ALL
import watchlist as w

SP = r"C:/Users/geunt/AppData/Local/Temp/claude/C--Users-geunt-OneDrive-Desktop---/4e686c73-74be-4337-8ccc-2f6136beaf7d/scratchpad/dip"
RAW_E, RAW_L = os.path.join(SP, "raw"), os.path.join(SP, "raw_1009")
OUT = os.path.join(HERE, "../results")
DEFS = {"V0": (0.06, 0.70), "V1": (0.08, 0.70), "V2": (0.06, 0.50)}
PER = {"앞": (RAW_E, 5, 36, 9, date(2026, 7, 14), date(2026, 9, 8)),
       "뒤": (RAW_L, 3, 60, 14, date(2026, 9, 10), date(2026, 10, 8)),
       "뒤5분": (RAW_L, 5, 36, 9, date(2026, 9, 10), date(2026, 10, 6)),     # 단위 확인용(raw_1009 5분봉은 10/9 10시까지, 10/7~8도 있으나 앞과 같은 5분 기준으로 10/6까지)
       "10/9": (RAW_L, 3, 60, 14, date(2026, 10, 9), date(2026, 10, 9))}     # 10:18 ET까지만 있음, 결과 없음(사례 설명용)
SYMS = [s for s in ALL if s not in ("GOOG", "SOXX")]
FEATS = ["spike", "t_peak", "pre", "gb_min", "big15", "low_p", "low_pm", "low_o",
         "th_n", "th_top", "th_gb", "rvol", "v_ratio", "tr5", "prev", "hi120"]
BINARY = {"th_n": 0.5, "th_top": 0.5}            # 문턱이 자연히 정해진 특성(th_n>=1, 1등 여부)
THEME_ONLY = {"th_top", "th_gb"}                 # 테마 동반(th_n>=1) 사건에서만 정의


def subcats():
    m = {}
    for subs in w.AI_UNIVERSE.values():
        for k, ts in subs.items():
            for t in ts:
                if re.fullmatch(r"[A-Z]{1,5}", t):
                    m[t] = k
    for t in BIO + ["TEM"]:
        m[t] = "바이오·유전체"
    for t in ["COIN", "CRCL", "HOOD", "MSTR"]:
        m[t] = "코인·핀테크"
    return m


SUB = subcats()


def daily(s):
    parts = []
    for raw in (RAW_E, RAW_L):
        p = f"{raw}/1d_{s}.parquet"
        if os.path.exists(p):
            parts.append(pd.read_parquet(p))
    dd = pd.concat(parts)
    dd.index = pd.to_datetime(dd.index).date
    dd = dd[~pd.Index(dd.index).duplicated(keep="last")].sort_index()
    c = dd.Close
    f = pd.DataFrame({"close": c, "P": c.shift(1), "prev": c.shift(1) / c.shift(2) - 1, "tr5": c.shift(1) / c.shift(6) - 1,
                      "hi120": c.shift(1).rolling(120, min_periods=60).max(),
                      "atr20": ((dd.High - dd.Low) / dd.Close).shift(1).rolling(20).mean(),
                      "vol20": c.pct_change().shift(1).rolling(20).std(),
                      "adv20": dd.Volume.shift(1).rolling(20).mean()})
    return f


def day_paths(per):
    """{날짜: {종목: 정규장 경로 dict}} + 프리장 최저 종가."""
    raw, mins, el, rl, lo, hi = PER[per]
    out = {}
    for s in SYMS:
        try:
            d = prep(bars(raw, s, mins), el, rl)
        except FileNotFoundError:
            continue
        f = daily(s)
        d = d[(d.date >= lo) & (d.date <= hi)]
        for day, x in d.groupby("date"):
            if day not in f.index or not np.isfinite(f.at[day, "P"]):
                continue
            reg = x[x.ses == "정규"]
            if len(reg) < 3:
                continue
            pre = x[x.ses == "프리"]
            fr = f.loc[day]
            out.setdefault(day, {})[s] = dict(
                t=reg.index, o=reg.o.values, h=reg.h.values, l=reg.l.values, c=reg.c.values, v=reg.v.values,
                runH=np.maximum.accumulate(reg.h.values), P=fr.P, close=fr.close if per != "10/9" else np.nan,
                pml=pre.c.min() if len(pre) else np.nan, mins=mins, **{k: fr[k] for k in ("prev", "tr5", "hi120", "atr20", "vol20", "adv20")})
    return out


def at_time(p, T):
    """종목 경로 p에서 시작 시각 <= T인 마지막 봉 번호(없으면 -1)."""
    return int(np.searchsorted(p["t"].values, T.to_datetime64(), side="right")) - 1


def find_events(paths, per, dname):
    thr, gbt = DEFS[dname]
    rows = []
    for day, S in paths.items():
        for s, p in S.items():
            P, runH, c = p["P"], p["runH"], p["c"]
            R = runH - P
            with np.errstate(divide="ignore", invalid="ignore"):
                gb = np.where(R > 0, (runH - c) / R, np.nan)
            ok = np.where((runH / P - 1 >= thr) & (gb >= gbt))[0]
            if not len(ok):
                continue
            i = int(ok[0])
            mins = p["mins"]
            H, RT, cT = runH[i], R[i], c[i]
            pk = int(np.argmax(p["h"][: i + 1]))                     # 첫 최고가 봉
            path = np.concatenate([[H], c[pk: i + 1]])
            wdw = 15 // mins
            big = max((path[max(0, j - wdw): j].max() - path[j]) for j in range(1, len(path))) / RT
            L = p["l"][pk: i + 1].min()
            T = p["t"][i]
            # 테마 동반: 같은 소분류 다른 종목의 T 시점 값
            sub = SUB.get(s)
            co = []
            for q, pq in S.items():
                if q == s or sub is None or SUB.get(q) != sub:
                    continue
                k = at_time(pq, T)
                if k < 0:
                    continue
                Hq = pq["runH"][k]
                spq = Hq / pq["P"] - 1
                if spq >= thr:
                    co.append((q, spq, (Hq - pq["c"][k]) / (Hq - pq["P"])))
            gbT = (H - cT) / RT
            v = p["v"]
            vr = (v[pk + 1: i + 1].sum() / ((i - pk) * mins)) / (v[: pk + 1].sum() / ((pk + 1) * mins)) \
                if i > pk and v[: pk + 1].sum() > 0 else np.nan
            close = p["close"]
            rows.append(dict(
                per=per, dfn=dname, sym=s, sub=sub or "단독", day=day, T=T, tmin=(T.hour * 60 + T.minute) - 570,
                P=P, H=H, cT=cT, close=close,
                Y1=close / cT - 1 if np.isfinite(close) else np.nan,
                Y2=float(close >= P + 0.5 * RT) if np.isfinite(close) else np.nan,
                post_max=(p["h"][i + 1:].max() - cT) / RT if i + 1 < len(c) else np.nan,     # 설명용(T 이후)
                gbT=gbT, spike=H / P - 1, t_peak=(p["t"][pk].hour * 60 + p["t"][pk].minute) - 570,
                pre=(p["o"][0] - P) / RT, gb_min=(i - pk) * mins, big15=big,
                low_p=(L - P) / RT, low_pm=L / p["pml"] - 1 if np.isfinite(p["pml"]) else np.nan, low_o=L / p["o"][0] - 1,
                th_n=len(co), th_top=(float(all(H / P - 1 >= x[1] for x in co)) if co else np.nan),
                th_gb=(gbT - np.median([x[2] for x in co])) if co else np.nan, th_peers=",".join(x[0] for x in co),
                rvol=v[: i + 1].sum() / p["adv20"] if p["adv20"] > 0 else np.nan, v_ratio=vr,
                tr5=p["tr5"], prev=p["prev"], hi120=H / p["hi120"] - 1 if np.isfinite(p["hi120"]) else np.nan,
                atr20=p["atr20"], vol20=p["vol20"], early30=float(T.hour * 60 + T.minute < 600)))
    return pd.DataFrame(rows)


# ---------- 통계 ----------
def ols_cl(y, X, g):
    """y = a + X b, 날짜 군집 표준오차(CR1). 반환 (계수, t)."""
    X = np.column_stack([np.ones(len(y))] + ([X] if X.ndim == 1 else [X]))
    n, k = X.shape
    A = np.linalg.pinv(X.T @ X)
    b = A @ X.T @ y
    e = y - X @ b
    gi = pd.factorize(np.asarray(g))[0]
    G = gi.max() + 1
    S = np.zeros((G, k))
    np.add.at(S, gi, X * e[:, None])
    V = (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1)) * A @ (S.T @ S) @ A
    se = np.sqrt(np.maximum(np.diag(V), 1e-300))
    return b, b / se


def sub_of(df, f):
    x = df[df.th_n >= 1] if f in THEME_ONLY else df
    return x[np.isfinite(x[f].astype(float)) & np.isfinite(x.Y1)]


def flag_of(x, f, thr):
    return (x[f].astype(float) > thr).astype(float).values


def diff(x, fl, y="Y1"):
    yy = x[y].values.astype(float)
    if fl.sum() < 3 or (1 - fl).sum() < 3:
        return dict(b=np.nan, t=np.nan, n1=int(fl.sum()), n0=int((1 - fl).sum()), m1=np.nan, m0=np.nan)
    b, t = ols_cl(yy, fl, x.day.values)
    return dict(b=b[1], t=t[1], n1=int(fl.sum()), n0=int((1 - fl).sum()), m1=yy[fl == 1].mean(), m0=yy[fl == 0].mean())


def controlled(x, fl, y="Y1"):
    ok = np.isfinite(x.atr20.values) & np.isfinite(x.vol20.values)
    x, fl = x[ok], fl[ok]
    if fl.sum() < 3 or (1 - fl).sum() < 3:
        return np.nan, np.nan
    X = np.column_stack([fl, x.atr20.values, x.vol20.values, x.early30.values])
    b, t = ols_cl(x[y].values.astype(float), X, x.day.values)
    return b[1], t[1]


def drop_top(x, fl, sign, key, y="Y1"):
    """방향(sign)에 가장 크게 기여한 key(종목 또는 날짜) 하나를 빼고 다시 잰다."""
    best, kept = None, None
    for k in x[key].unique():
        m = (x[key] != k).values
        r = diff(x[m], fl[m], y)
        if np.isfinite(r["b"]) and (best is None or sign * r["b"] < sign * best["b"]):
            best, kept = r, k
    return kept, best


def fe_slope(x, f, y="Y1"):
    """날짜×소분류 고정효과: 같은 날 같은 소분류 사건이 2개 이상인 묶음 안에서 특성·결과를 빼고 기울기(날짜 군집 t)."""
    x = x[x["sub"] != "단독"].copy()
    x = x[np.isfinite(x[f].astype(float)) & np.isfinite(x[y].astype(float))]
    x["grp"] = x.day.astype(str) + "|" + x["sub"]
    cnt = x.grp.map(x.grp.value_counts())
    x = x[cnt >= 2]
    if len(x) < 6:
        return dict(n=len(x), g=x.grp.nunique(), b=np.nan, t=np.nan, agree=np.nan)
    fx = x[f].astype(float) - x.groupby("grp")[f].transform("mean")
    yx = x[y].astype(float) - x.groupby("grp")[y].transform("mean")
    keep = fx.abs() > 1e-12
    if keep.sum() < 4:
        return dict(n=len(x), g=x.grp.nunique(), b=np.nan, t=np.nan, agree=np.nan)
    sd = fx[keep].std()
    b, t = ols_cl(yx[keep].values, (fx[keep] / sd).values, x.day[keep].values)
    # 묶음 안 '특성 높은 쪽이 결과도 높았나' 비율(쌍 비교)
    ag = []
    for _, gx in x.groupby("grp"):
        a = gx[f].astype(float).values
        r = gx[y].astype(float).values
        for i in range(len(a)):
            for j in range(i + 1, len(a)):
                if a[i] != a[j] and r[i] != r[j]:
                    ag.append(np.sign(a[i] - a[j]) == np.sign(r[i] - r[j]))
    return dict(n=len(x), g=x.grp.nunique(), b=b[1], t=t[1], agree=np.mean(ag) if ag else np.nan, pairs=len(ag))


def fe_perm(x, f, y="Y1", n=2000, seed=11):
    """날짜×소분류 묶음 안 쌍 일치율과 묶음 안 결과 섞기 무작위 대조(양측 p). 묶음 = 같은 날 같은 소분류 사건 2개 이상."""
    x = x[x["sub"] != "단독"].copy()
    x = x[np.isfinite(x[f].astype(float)) & np.isfinite(x[y].astype(float))]
    x["grp"] = x.day.astype(str) + "|" + x["sub"]
    x = x[x.grp.map(x.grp.value_counts()) >= 2]
    G = [(g[f].astype(float).values, g[y].astype(float).values) for _, g in x.groupby("grp")]
    if not G:
        return dict(agree=np.nan, pairs=0, p=np.nan, groups=0, dates=0)

    def stat(G2):
        a_, n_ = 0, 0
        for a, r in G2:
            da = np.sign(a[:, None] - a[None, :])
            dr = np.sign(r[:, None] - r[None, :])
            iu = np.triu_indices(len(a), 1)
            m = (da[iu] != 0) & (dr[iu] != 0)
            a_ += (da[iu][m] == dr[iu][m]).sum()
            n_ += m.sum()
        return a_ / n_ if n_ else np.nan, n_

    obs, npair = stat(G)
    rng = np.random.default_rng(seed)
    null = np.array([stat([(a, rng.permutation(r)) for a, r in G])[0] for _ in range(n)])
    pv = np.mean(np.abs(null - 0.5) >= abs(obs - 0.5) - 1e-12)
    return dict(agree=obs, pairs=int(npair), p=pv, groups=len(G), dates=x.day.nunique())


def common_time(paths, per, thr, gbt):
    """같은 시각 비교(4c): 같은 날 같은 소분류에서 급등 문턱을 넘은 종목이 2개 이상인 첫 반납 시각 T*
    (그 묶음 누군가가 처음 반납 문턱을 넘은 봉)에, 묶음 모든 급등 종목의 반납 비율과 그 뒤 종가까지 수익을 잰다."""
    rows = []
    for day, S in paths.items():
        bysub = {}
        for s, p in S.items():
            if SUB.get(s):
                bysub.setdefault(SUB[s], []).append(s)
        for sub, mem in bysub.items():
            if len(mem) < 2:
                continue
            times = sorted(set().union(*[set(S[s]["t"]) for s in mem]))
            for T in times:
                st = []
                for s in mem:
                    p = S[s]
                    k = at_time(p, T)
                    if k < 0:
                        continue
                    H = p["runH"][k]
                    if H / p["P"] - 1 >= thr:
                        st.append((s, k, H, (H - p["c"][k]) / (H - p["P"])))
                if len(st) >= 2 and max(x[3] for x in st) >= gbt:
                    for s, k, H, g in st:
                        p = S[s]
                        rows.append(dict(per=per, day=day, sub=sub, T=T, sym=s, gb=g, spike=H / p["P"] - 1,
                                         first=float(g >= gbt), n=len(st),
                                         Y1=p["close"] / p["c"][k] - 1, Y2=float(p["close"] >= p["P"] + 0.5 * (H - p["P"]))))
                    break
    return pd.DataFrame(rows)


def screen(df, f, thr=None, y="Y1"):
    x = sub_of(df, f)
    if thr is None:
        thr = BINARY.get(f, x[f].astype(float).median())
    fl = flag_of(x, f, thr)
    return thr, x, fl, diff(x, fl, y)


def spearman(a, b):
    return pd.Series(a).rank().corr(pd.Series(b).rank())


def main():
    ev, PA = {}, {}
    for per in PER:
        paths = day_paths(per)
        PA[per] = paths
        for dn in DEFS:
            ev[(per, dn)] = find_events(paths, per, dn)
        print(per, {dn: len(ev[(per, dn)]) for dn in DEFS}, flush=True)
    allev = pd.concat(ev.values(), ignore_index=True)
    allev.to_csv(os.path.join(OUT, "pat_spike_events.csv"), index=False, encoding="utf-8-sig")

    L, tests = [], []
    say = L.append
    say("테마 급등 뒤 되돌림 사건 연구 (pat_spike.py, 2026-10-10)")
    say("정의(시험 전 고정): 정규장 그날 고점 H >= 전날 종가 P +6%, 그 뒤 봉 종가가 상승분(H-P)의 70% 이상 반납한 첫 봉 = T. 종목·날짜당 1건.")
    say("  변형: V1 급등 +8%/반납 70%, V2 급등 +6%/반납 50%. 이 셋만 썼다.")
    say("결과(시험 전 고정): Y1 = 그날 종가 / T 봉 종가 - 1, Y2 = 종가 >= P + 0.5(H-P)(상승분 절반 이상 되찾음).")
    say("고르기 = 앞 7/14~9/8 5분봉, 확인 = 뒤 9/10~10/8 3분봉. 특성은 T 봉 종가까지 값만. t = 날짜 군집.")
    say("")
    say("1. 사건 수와 기본 성적(사건 = 그 자리에서 사서 종가에 판다)")
    say(f"{'기간·정의':<14}{'건수':>6}{'종목':>6}{'날짜':>6}{'Y1 평균':>9}{'Y1 중앙':>9}{'t':>7}{'Y2 비율':>9}{'T<10시':>8}")
    for (per, dn), x in ev.items():
        if per == "10/9" or not len(x):
            continue
        b, t = ols_cl(x.Y1.values, np.zeros(len(x)), x.day.values) if len(x) > 3 else ([np.nan], [np.nan])
        say(f"{per + ' ' + dn:<14}{len(x):>6}{x.sym.nunique():>6}{x.day.nunique():>6}{x.Y1.mean():>+9.2%}{x.Y1.median():>+9.2%}"
            f"{t[0]:>7.2f}{x.Y2.mean():>9.0%}{x.early30.mean():>8.0%}")
    say("")

    # 2. 앞 기간 V0 고르기: 특성 16개 x 결과 2개
    E, Lt = ev[("앞", "V0")], ev[("뒤", "V0")]
    say("2. 앞 기간(5분봉) V0 고르기: 특성 > 앞 중앙값(th_n은 >=1, th_top은 1등) 대 나머지. 차 = 높은 쪽 - 낮은 쪽 평균")
    say(f"{'특성':<9}{'문턱':>9}{'n높':>5}{'n낮':>5}{'Y1 높':>8}{'Y1 낮':>8}{'차':>8}{'t':>7}{'Y2 차':>8}{'t':>7}{'순위상관':>9}")
    scr = []
    for f in FEATS:
        thr, x, fl, r1 = screen(E, f)
        r2 = diff(x, fl, "Y2")
        rho = spearman(x[f].astype(float).values, x.Y1.values)
        scr.append(dict(f=f, thr=thr, **{k + "1": v for k, v in r1.items()}, b2=r2["b"], t2=r2["t"], rho=rho))
        tests.append(dict(stage="앞 고르기", per="앞", dfn="V0", feat=f, thr=thr, y="Y1", **r1))
        tests.append(dict(stage="앞 고르기", per="앞", dfn="V0", feat=f, thr=thr, y="Y2", **r2))
        say(f"{f:<9}{thr:>9.3f}{r1['n1']:>5}{r1['n0']:>5}{r1['m1']:>+8.2%}{r1['m0']:>+8.2%}{r1['b']:>+8.2%}{r1['t']:>7.2f}"
            f"{r2['b']:>+8.0%}{r2['t']:>7.2f}{rho:>9.2f}")
    S = pd.DataFrame(scr)
    S["abs"] = S.t1.abs()
    cands = S.sort_values("abs", ascending=False).head(3)
    say("")

    # 선택편향: 결과를 날짜 묶음째 섞어 16개 중 최대 |t|의 우연 분포
    rng = np.random.default_rng(7)
    mx = []
    for _ in range(300):                               # 사건 단위로 결과를 섞고 t는 그대로 날짜 군집으로 잰다
        Ep = E.copy()
        Ep["Y1"] = rng.permutation(E.Y1.values)
        m = 0
        for f in FEATS:
            _, x, fl, r = screen(Ep, f)
            if np.isfinite(r["t"]):
                m = max(m, abs(r["t"]))
        mx.append(m)
    q95 = np.quantile(mx, 0.95)
    say(f"선택편향: 결과를 무작위로 섞어 같은 16개 시험을 300번 돌리면 최대 |t|의 95% 분위 = {q95:.2f}, 중앙값 = {np.median(mx):.2f}.")
    say(f"앞에서 가장 강한 후보 3개(|t| 순): {', '.join(f'{r.f}(t {r.t1:.2f})' for r in cands.itertuples())}")
    say("")

    # 3. 뒤 기간 확인
    say("3. 뒤 기간(3분봉) 확인: 앞의 문턱·방향 그대로")
    verdicts = []
    for r in cands.itertuples():
        f, thr, sign = r.f, r.thr, np.sign(r.b1)
        say(f"[{f}] 앞 문턱 {thr:.3f}, 방향: {'높을수록' if sign > 0 else '낮을수록'} 좋음. 앞 차 {r.b1:+.2%} (t {r.t1:.2f}), Y2 {r.b2:+.0%} (t {r.t2:.2f})")
        rows = {}
        for dn in DEFS:
            thr_d = screen(ev[("앞", dn)], f)[0]
            for per in ("앞", "뒤", "뒤5분"):
                _, x, fl, r1 = screen(ev[(per, dn)], f, thr_d)
                r2 = diff(x, fl, "Y2")
                rows[(per, dn)] = (x, fl, r1, r2)
                for yy, rr in (("Y1", r1), ("Y2", r2)):
                    tests.append(dict(stage="확인", per=per, dfn=dn, feat=f, thr=thr_d, y=yy, **rr))
                say(f"   {per:<4}{dn}: n {r1['n1']}/{r1['n0']}  Y1 {r1['m1']:+.2%} vs {r1['m0']:+.2%} 차 {r1['b']:+.2%} (t {r1['t']:.2f})"
                    f"  Y2 차 {r2['b']:+.0%} (t {r2['t']:.2f})")
        x, fl, r1, r2 = rows[("뒤", "V0")]
        # 뒤 기간 절반
        for nm, m in (("뒤 앞절반 9/10~9/23", x.day.values < date(2026, 9, 24)), ("뒤 뒤절반 9/24~10/8", x.day.values >= date(2026, 9, 24))):
            rr = diff(x[m], fl[m])
            tests.append(dict(stage="확인 절반", per=nm, dfn="V0", feat=f, thr=thr, y="Y1", **rr))
            say(f"   {nm}: 차 {rr['b']:+.2%} (t {rr['t']:.2f}, n {rr['n1']}/{rr['n0']})")
        ks, rs = drop_top(x, fl, sign, "sym")
        kd, rd = drop_top(x, fl, sign, "day")
        bc, tc = controlled(x, fl)
        # 시간대 나눠 보기
        tz = {}
        for nm, m in (("T<10시", x.early30.values == 1), ("T>=10시", x.early30.values == 0)):
            tz[nm] = diff(x[m], fl[m])
        fe_e, fe_l = fe_slope(ev[("앞", "V0")], f), fe_slope(x, f)
        for nm, rr in (("상위1종목 제외 " + str(ks), rs), ("상위1일 제외 " + str(kd), rd), ("T<10시만", tz["T<10시"]), ("T>=10시만", tz["T>=10시"])):
            tests.append(dict(stage="확인 견고성", per="뒤", dfn="V0", feat=f, thr=thr, y="Y1", note=nm, **(rr or {})))
        tests.append(dict(stage="확인 통제", per="뒤", dfn="V0", feat=f, thr=thr, y="Y1", note="atr20+vol20+early30", b=bc, t=tc))
        tests.append(dict(stage="날짜×소분류", per="앞", dfn="V0", feat=f, y="Y1", note="표준화 기울기", **fe_e))
        tests.append(dict(stage="날짜×소분류", per="뒤", dfn="V0", feat=f, y="Y1", note="표준화 기울기", **fe_l))
        say(f"   상위 1종목({ks}) 제외: 차 {rs['b']:+.2%} (t {rs['t']:.2f}) / 상위 1일({kd}) 제외: 차 {rd['b']:+.2%} (t {rd['t']:.2f})")
        say(f"   통제(atr20·vol20·T<10시): 차 {bc:+.2%} (t {tc:.2f}) / T<10시만 {tz['T<10시']['b']:+.2%} (t {tz['T<10시']['t']:.2f}, n {tz['T<10시']['n1']}/{tz['T<10시']['n0']})"
            f" / T>=10시만 {tz['T>=10시']['b']:+.2%} (t {tz['T>=10시']['t']:.2f}, n {tz['T>=10시']['n1']}/{tz['T>=10시']['n0']})")
        say(f"   날짜×소분류 안 비교(1표준편차당 Y1): 앞 {fe_e['b']:+.2%} (t {fe_e['t']:.2f}, 묶음 {fe_e['g']}, 쌍 일치 {fe_e['agree']:.0%})"
            f" / 뒤 {fe_l['b']:+.2%} (t {fe_l['t']:.2f}, 묶음 {fe_l['g']}, 쌍 일치 {fe_l['agree']:.0%})")
        b_ok = np.sign(r1["b"]) == sign
        rob = all(np.isfinite(v) and np.sign(v) == sign for v in (rs["b"], rd["b"], bc)) and abs(tc) >= 1.5
        if b_ok and abs(r1["t"]) >= 2 and rob:
            vd = "확인됨"
        elif b_ok and abs(r1["t"]) >= 1:
            vd = "약함"
        else:
            vd = "뒤에서 실패"
        verdicts.append(dict(f=f, thr=thr, sign=sign, b1=r.b1, t1=r.t1, lb=r1["b"], lt=r1["t"], ln=(r1["n1"], r1["n0"]),
                             rs=rs, rd=rd, ks=ks, kd=kd, bc=bc, tc=tc, fe_e=fe_e, fe_l=fe_l, vd=vd,
                             t2=r2["t"], b2=r2["b"]))
        say(f"   판정: {vd}")
        say("")

    # 4. 날짜×소분류 안 비교 전 특성(설명용, 고르기에 쓰지 않음)
    say("4. 같은 날 같은 소분류 안 비교(날짜×소분류 고정효과, 특성 1표준편차당 Y1, 고르기에 안 씀)")
    say("   쌍일치 = 같은 묶음 두 사건 중 특성이 높은 쪽이 Y1도 높았던 비율(50% = 무관, 쌍은 서로 독립이 아님)")
    say(f"{'특성':<9}{'앞 기울기':>10}{'t':>7}{'쌍일치':>7}{'쌍':>5}{'뒤 기울기':>10}{'t':>7}{'쌍일치':>7}{'쌍':>5}")
    for f in FEATS:
        a, b = fe_slope(ev[("앞", "V0")], f), fe_slope(ev[("뒤", "V0")], f)
        tests.append(dict(stage="날짜×소분류 전체", per="앞", dfn="V0", feat=f, y="Y1", **a))
        tests.append(dict(stage="날짜×소분류 전체", per="뒤", dfn="V0", feat=f, y="Y1", **b))
        say(f"{f:<9}{a['b']:>+10.2%}{a['t']:>7.2f}{a['agree']:>7.0%}{a.get('pairs', 0):>5}{b['b']:>+10.2%}{b['t']:>7.2f}{b['agree']:>7.0%}{b.get('pairs', 0):>5}")
    fa, fb = fe_slope(ev[("앞", "V0")], "spike"), fe_slope(ev[("뒤", "V0")], "spike")
    say(f"   (묶음 수: 앞 {fa['g']}개 {fa['n']}건, 뒤 {fb['g']}개 {fb['n']}건. 뒤 기울기 t는 날짜 군집이 8개뿐이라 믿기 어렵다)")
    say("")
    say("4b. 묶음 안 쌍 일치율의 무작위 대조(같은 묶음 안에서 결과만 2,000번 섞음, 양측 p). 앞에서 p<0.05인 특성만 뒤에서 같은 방향인지 본다")
    say(f"{'특성':<9}{'앞 일치':>8}{'p':>7}{'7/31 빼면':>10}{'p':>7}{'뒤 일치':>8}{'p':>7}{'뒤 V2':>8}{'p':>7}")
    E0 = ev[("앞", "V0")]
    E0x = E0[E0.day.astype(str) != "2026-07-31"]
    fe2 = []
    for f in FEATS:
        a, a2, b, b2 = fe_perm(E0, f), fe_perm(E0x, f), fe_perm(ev[("뒤", "V0")], f), fe_perm(ev[("뒤", "V2")], f)
        for nm, rr in (("앞", a), ("앞 7/31 제외", a2), ("뒤", b), ("뒤 V2", b2)):
            tests.append(dict(stage="묶음 안 쌍 일치", per=nm, dfn="V2" if "V2" in nm else "V0", feat=f, y="Y1",
                              b=rr["agree"], n1=rr["pairs"], p=rr["p"], note=f"묶음 {rr['groups']} 날짜 {rr['dates']}"))
        fe2.append((f, a, a2, b, b2))
        say(f"{f:<9}{a['agree']:>8.0%}{a['p']:>7.3f}{a2['agree']:>10.0%}{a2['p']:>7.3f}{b['agree']:>8.0%}{b['p']:>7.3f}{b2['agree']:>8.0%}{b2['p']:>7.3f}")
    g0, g1, g2 = fe_perm(E0, "spike"), fe_perm(ev[("뒤", "V0")], "spike"), fe_perm(ev[("뒤", "V2")], "spike")
    say(f"   (묶음·날짜: 앞 {g0['groups']}·{g0['dates']}, 뒤 {g1['groups']}·{g1['dates']}, 뒤 V2 {g2['groups']}·{g2['dates']})")
    pick = [r for r in fe2 if np.isfinite(r[1]["p"]) and r[1]["p"] < 0.05]
    for f, a, a2, b, b2 in pick:
        same = np.sign(a["agree"] - 0.5) == np.sign(b["agree"] - 0.5)
        say(f"   앞 p<0.05 → {f}: 앞 {a['agree']:.0%} → 뒤 {b['agree']:.0%} (p {b['p']:.3f}) {'같은 방향' if same else '반대'}"
            f"{', 뒤 p<0.05' if same and b['p'] < 0.05 else ''}")
    tm = [(nm, fe_perm(ev[k], "tmin")) for nm, k in (("앞 V0", ("앞", "V0")), ("뒤 V0", ("뒤", "V0")), ("뒤 V2", ("뒤", "V2")))]
    for nm, rr in tm:
        tests.append(dict(stage="묶음 안 쌍 일치(설명)", per=nm, dfn=nm[-2:], feat="tmin", y="Y1", b=rr["agree"], n1=rr["pairs"], p=rr["p"]))
    say("   설명 확인 - 자기 T 시각(tmin): 같은 묶음에서 T가 늦은 쪽이 Y1이 높았던 비율 "
        + ", ".join(f"{nm} {rr['agree']:.0%}(p {rr['p']:.3f})" for nm, rr in tm)
        + ". 먼저 70%를 반납한 종목은 묶음이 더 빠지는 구간을 떠안는다. 같은 시각으로 맞춘 4c에서 이 차이가 사라진다.")
    say("")

    # 4c. 같은 시각 비교(LITE·COHR 질문 그대로)
    say("4c. 같은 시각 비교: 같은 날 같은 소분류 급등(+6%) 종목이 2개 이상일 때, 그중 하나가 처음 상승분 70%(V2: 50%)를 반납한 봉 T*에")
    say("    묶음 모든 급등 종목의 '그때까지 반납 비율'과 'T* 종가 → 그날 종가 수익'을 비교한다(시험 전 고정, 특성 2개: 반납 비율 gb, 급등 크기 spike).")
    say(f"{'표본':<10}{'묶음':>5}{'종목':>5}{'특성':>7}{'쌍일치':>7}{'쌍':>5}{'p':>7}{'덜 반납 쪽 Y1':>13}{'더 반납 쪽 Y1':>13}{'Y2 덜/더':>11}")
    CC = []
    for per in ("앞", "뒤", "뒤5분"):
        for dn in ("V0", "V2"):
            thr, gbt = DEFS[dn]
            C = common_time(PA[per], per, thr, gbt)
            CC.append(C.assign(dfn=dn))
            if not len(C):
                continue
            C["rk"] = C.groupby(["day", "sub"]).gb.rank(method="first")
            nmem = C.groupby(["day", "sub"]).sym.transform("count")
            lo = C[C.rk == 1]                                  # 묶음에서 가장 덜 반납
            hi = C[C.rk == nmem]                               # 가장 많이 반납
            for f in ("gb", "spike"):
                r = fe_perm(C.assign(**{"sub": C["sub"]}), f)
                tests.append(dict(stage="같은 시각 비교", per=per, dfn=dn, feat=f, y="Y1", b=r["agree"], n1=r["pairs"], p=r["p"],
                                  note=f"묶음 {r['groups']} 날짜 {r['dates']}"))
                extra = f"{lo.Y1.mean():>+13.2%}{hi.Y1.mean():>+13.2%}{lo.Y2.mean():>6.0%}/{hi.Y2.mean():.0%}" if f == "gb" else ""
                say(f"{per + ' ' + dn:<10}{r['groups']:>5}{len(C):>5}{f:>7}{r['agree']:>7.0%}{r['pairs']:>5}{r['p']:>7.3f}{extra}")
            if f"{per}{dn}" in ("뒤V0", "앞V0"):
                d = (lo.set_index(["day", "sub"]).Y1 - hi.set_index(["day", "sub"]).Y1)
                b, t = ols_cl(d.values, np.zeros(len(d)), [k[0] for k in d.index]) if len(d) > 2 else ([np.nan], [np.nan])
                tests.append(dict(stage="같은 시각 비교 덜-더", per=per, dfn=dn, feat="gb", y="Y1", b=d.mean(), t=t[0], n1=len(d)))
                say(f"   {per} {dn} 덜 반납 1등 - 더 반납 1등 Y1 차 평균 {d.mean():+.2%} (날짜 군집 t {t[0]:.2f}, 묶음 {len(d)}, 덜 반납 쪽이 나은 묶음 {np.mean(d > 0):.0%})")
    say("   쌍일치(gb) < 50%면 '덜 반납한 쪽이 그 뒤 더 올랐다'. p = 묶음 안 결과 섞기 2,000번 양측.")
    for dn in ("V0", "V2"):
        C = common_time(PA["10/9"], "10/9", *DEFS[dn])
        CC.append(C.assign(dfn=dn))
        C = C[C["sub"] == "광 전송·광모듈"]
        if len(C):
            say(f"   10/9 광모듈 {dn}: T* {C['T'].iloc[0].strftime('%H:%M')} ET, " + ", ".join(f"{r.sym} 반납 {r.gb:.0%}(급등 {r.spike:+.1%})" for r in C.itertuples()))
    pd.concat(CC, ignore_index=True).to_csv(os.path.join(OUT, "pat_spike_common.csv"), index=False, encoding="utf-8-sig")
    say("")

    # 5. 10/9 사례(10:18 ET까지 데이터, 결과 없음)
    say("5. 10/9 광 전송·광모듈 사례(데이터 10:18 ET까지, 종가 결과 없음 — 특성만)")
    for s_, p in sorted(PA["10/9"].get(date(2026, 10, 9), {}).items()):
        if SUB.get(s_) != "광 전송·광모듈":
            continue
        R = p["runH"] - p["P"]
        gb = np.where(R > 0, (p["runH"] - p["c"]) / R, np.nan)
        say(f"   {s_:<5} 고점 {p['runH'][-1] / p['P'] - 1:+.1%}, 10:18까지 봉 종가 기준 최대 반납 {np.nanmax(gb):.0%}, 10:18 반납 {gb[-1]:.0%},"
            f" 전날 {p['prev']:+.1%}, 5일 {p['tr5']:+.1%}")
    for dn in ("V0", "V2"):
        x = ev[("10/9", dn)]
        x = x[x["sub"] == "광 전송·광모듈"] if len(x) else x
        say(f"  {dn}: " + ("사건 없음" if not len(x) else ""))
        for r in x.itertuples():
            say(f"   {r.sym:<5} T {r.T.strftime('%H:%M')} 급등 {r.spike:+.1%} 고점 {r.t_peak}분 pre {r.pre:.2f} gb_min {r.gb_min} big15 {r.big15:.2f}"
                f" low_p {r.low_p:+.2f} low_o {r.low_o:+.2%} th_n {r.th_n} th_top {r.th_top} th_gb {r.th_gb:+.2f} rvol {r.rvol:.2f}"
                f" v_ratio {r.v_ratio:.2f} tr5 {r.tr5:+.1%} hi120 {r.hi120:+.1%} T뒤최고(10:18까지) {r.post_max:+.2f}R")
    say("")
    T_ = pd.DataFrame(tests)
    say(f"시험 수: 전체 {len(T_)}개 = 앞 고르기 16특성 x 2결과 = 32, 확인 3후보 x 3정의 x 3표본(앞·뒤3분·뒤5분) x 2결과 = 54,"
        f" 절반·견고성·통제·날짜×소분류 3후보 x 9 = 27, 날짜×소분류 기울기 16 x 2기간 = 32, 묶음 안 쌍 일치 16 x 4 = 64, 같은 시각 비교 2특성 x 3표본 x 2정의 + 덜-더 2 = 14, 설명 확인 3.")
    say("판정 기준(시험 전 고정): 확인됨 = 뒤 같은 방향 |t|>=2, 상위1종목·1일 제외와 통제 뒤에도 같은 방향, 통제 |t|>=1.5. 약함 = 같은 방향 |t|>=1. 그 밖 = 뒤에서 실패.")
    say("판정: " + ", ".join(f"{v['f']} {v['vd']}(앞 {v['b1']:+.2%} t {v['t1']:.2f} → 뒤 {v['lb']:+.2%} t {v['lt']:.2f})" for v in verdicts))
    T_.to_csv(os.path.join(OUT, "pat_spike_tests.csv"), index=False, encoding="utf-8-sig")
    txt = "\n".join(L)
    open(os.path.join(OUT, "pat_spike.txt"), "w", encoding="utf-8").write(txt)
    print(txt)
    pd.DataFrame([{k: v for k, v in d.items() if k not in ("rs", "rd", "fe_e", "fe_l")} for d in verdicts]).to_csv(
        os.path.join(OUT, "pat_spike_cands.csv"), index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
