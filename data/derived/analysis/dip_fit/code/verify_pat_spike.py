"""반박 검증: pat_spike.py(테마 급등 뒤 되돌림 사건 연구) 후보를 처음부터 다시 계산해 깬다(2026-10-10). 기본값은 '살아남지 못함'.

사건 정의(분석가가 시험 전에 고정한 것, 그대로 씀):
  정규장 봉만. P = 전날 정규장 종가(일봉). H_t = 그날 봉 t까지 누적 고가. T = H_t/P-1 >= +6% 이고 봉 종가가 (H_t-P)의 70% 이상을
  반납한 첫 봉(V0). V1 +8%/70%, V2 +6%/50%. Y1 = 그날 종가/T 봉 종가 - 1, Y2 = 종가 >= P + 0.5(H-P).
후보: prev(전날 > 앞 중앙값이면 약함), big15(반납 중 최대 15분 하락/상승분 > 앞 중앙값이면 약함), spike(급등 > 앞 중앙값이면 강함),
      th_gb·gb_min(자기 T 기준 묶음 안 비교), 같은 시각 반납 비율(4c).
확인 항목:
  [1] 재현: 내 코드로 사건을 다시 찾고(분석가 find_events 안 씀) 건수·Y1·후보 특성 값을 분석가 events.csv와 대조.
  [2] 미래 참조: 일봉 날짜 맞춤(P가 전날 1분봉 마지막 정규장 종가와 같은지), 특성을 T 봉까지 자른 배열로 다시 계산해 같은지,
      T 봉 종가 대신 다음 봉 시가에 샀을 때(실행 가능한 진입).
  [3] 후보 숫자 재계산(내 날짜 군집 t), 통제(atr20·vol20·T<10시 → +T분·gb_min·gbT·spike), 같은 날 고정효과, 모든 하루 빼기,
      모든 1종목 빼기, 상하 5% 자르기·중앙값, 고점이 1분 꼬리(그 분 시가·종가·직전 분 종가보다 1% 넘게 위)인 사건 제외,
      T 봉 종가가 1분 꼬리인 사건 제외, 이웃 정의 4개(+5%/70, +7%/70, +6%/60, +6%/80, 검증용 추가 시험).
  [4] 우연 대조: (a) 사건 단위로 결과를 섞어 '16개 중 앞 |t| 상위 3개 → 뒤 확인' 절차 전체를 500번,
      (b) 같은 날 안에서만 결과를 섞어 같은 절차 500번(날짜 구성으로 생기는 효과를 남긴 대조),
      (c) 뒤 기간 prev·big15 고정 문턱에서 같은 날 안 섞기 2,000번(한쪽 p).
  [5] 묶음 안 비교: 자기 T 기준 th_gb·gb_min 쌍 일치를 T 시각 차이로 나눠 보고, 같은 시각 비교(4c)를 내 코드로 다시 계산.
  [6] 10/9 광모듈 사례(10:18 ET까지): 내 코드로 prev·big15·spike·반납 비율과 고점 1분 꼬리 여부, 규칙 적용.
입력(다시 받지 않음): scratchpad dip/raw(5분봉 앞), raw_1009(1분·5분·일봉), ../results/pat_spike_events.csv(16특성, 우연 대조용)
출력: ../results/verify_pat_spike.txt
사용: py verify_pat_spike.py"""
import os, re, sys, time, warnings
from datetime import date
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "../results")
sys.path.append(os.path.join(HERE, "../../dip_reason/code"))
sys.path.insert(0, os.path.join(HERE, "../../../../.."))
from events import bars
from fetch import BIO
from stop_lab import ALL
import watchlist as w

warnings.filterwarnings("ignore", category=RuntimeWarning)
SP = r"C:/Users/geunt/AppData/Local/Temp/claude/C--Users-geunt-OneDrive-Desktop---/4e686c73-74be-4337-8ccc-2f6136beaf7d/scratchpad/dip"
RAW_E, RAW_L = SP + "/raw", SP + "/raw_1009"
SYMS = [s for s in ALL if s not in ("GOOG", "SOXX")]
PER = {"앞": (RAW_E, 5, date(2026, 7, 14), date(2026, 9, 8)),
       "뒤": (RAW_L, 3, date(2026, 9, 10), date(2026, 10, 8)),
       "뒤5분": (RAW_L, 5, date(2026, 9, 10), date(2026, 10, 6)),
       "10/9": (RAW_L, 3, date(2026, 10, 9), date(2026, 10, 9))}
DEFS = {"V0": (0.06, 0.70), "V1": (0.08, 0.70), "V2": (0.06, 0.50)}
NEIGH = {"N5_70": (0.05, 0.70), "N7_70": (0.07, 0.70), "N6_60": (0.06, 0.60), "N6_80": (0.06, 0.80)}
FEATS = ["spike", "t_peak", "pre", "gb_min", "big15", "low_p", "low_pm", "low_o",
         "th_n", "th_top", "th_gb", "rvol", "v_ratio", "tr5", "prev", "hi120"]
BINARY = {"th_n": 0.5, "th_top": 0.5}
THEME_ONLY = {"th_top", "th_gb"}
RNG = np.random.default_rng(20261010)
OUT, NT = [], {"n": 0}


def say(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s, flush=True)


def pc(x, d=2):
    return "nan" if x is None or not np.isfinite(x) else f"{100 * x:+.{d}f}%"


def tick(k=1):
    NT["n"] += k


# ------------------------------------------------------------------ 소분류(분석가와 같은 정의, 따로 구현)
def subcats(extra_coin=True):
    m = {}
    for subs in w.AI_UNIVERSE.values():
        for k, ts in subs.items():
            for t in ts:
                if re.fullmatch(r"[A-Z]{1,5}", t):
                    m[t] = k
    for t in list(BIO) + ["TEM"]:
        m[t] = "바이오·유전체"
    if extra_coin:
        for t in ("COIN", "CRCL", "HOOD", "MSTR"):
            m[t] = "코인·핀테크"
    return m


SUB = subcats()


# ------------------------------------------------------------------ 데이터
def daily(s):
    parts = [pd.read_parquet(f"{r}/1d_{s}.parquet") for r in (RAW_E, RAW_L) if os.path.exists(f"{r}/1d_{s}.parquet")]
    dd = pd.concat(parts)
    dd.index = pd.to_datetime(dd.index).date
    dd = dd[~pd.Index(dd.index).duplicated(keep="last")].sort_index()
    return dd


def load(per):
    """{날짜: {종목: dict}} 정규장 배열. 내 구현(prep 안 씀, 시각으로 정규장 판정)."""
    raw, mins, lo, hi = PER[per]
    out = {}
    for s in SYMS:
        try:
            b = bars(raw, s, mins)
        except FileNotFoundError:
            continue
        dd = daily(s)
        C = dd.Close
        days = list(C.index)
        pos = {d: k for k, d in enumerate(days)}
        mod = b.index.hour * 60 + b.index.minute
        dts = np.asarray(b.index.date)
        regm = (mod >= 570) & (mod < 960)
        for day in sorted(set(dts[(dts >= lo) & (dts <= hi)])):
            k = pos.get(day)
            if k is None or k < 2:
                continue
            m = regm & (dts == day)
            if m.sum() < 3:
                continue
            x = b[m]
            pre = b[(dts == day) & (mod < 570)]
            past = dd.iloc[max(0, k - 20):k]
            P = C.iloc[k - 1]
            rets = C.iloc[max(0, k - 21):k].pct_change().dropna()
            out.setdefault(day, {})[s] = dict(
                t=x.index, o=x.Open.values.astype(float), h=x.High.values.astype(float), l=x.Low.values.astype(float),
                c=x.Close.values.astype(float), v=x.Volume.values.astype(float), P=float(P),
                close=float(C.iloc[k]) if per != "10/9" else np.nan,
                prev=float(C.iloc[k - 1] / C.iloc[k - 2] - 1), tr5=float(C.iloc[k - 1] / C.iloc[k - 6] - 1) if k >= 6 else np.nan,
                atr20=float(((past.High - past.Low) / past.Close).mean()) if len(past) == 20 else np.nan,
                vol20=float(rets.std()) if len(rets) == 20 else np.nan,
                pml=float(pre.Close.min()) if len(pre) else np.nan, mins=mins)
    return out


def big15_of(H, cs, mins):
    """[H, 고점 봉~T 봉 종가]에서 15분 창 최대 하락(직전 창 최고 - 지금). pandas rolling 구현(분석가 반복문과 다른 경로)."""
    s = pd.Series(np.concatenate([[H], cs]))
    pm = s.shift(1).rolling(15 // mins, min_periods=1).max()
    return float((pm - s).max())


def events(paths, per, thr, gbt, keep_idx=False):
    rows = []
    for day, S in paths.items():
        for s, p in S.items():
            h, c = p["h"], p["c"]
            for i in range(len(c)):                      # T 봉까지만 보는 순차 탐색(미래 참조 방지 확인용)
                Hh = h[: i + 1].max()
                R = Hh - p["P"]
                if R <= 0 or Hh / p["P"] - 1 < thr:
                    continue
                if (Hh - c[i]) / R >= gbt:
                    break
            else:
                continue
            hh, cc, ll, vv = h[: i + 1], c[: i + 1], p["l"][: i + 1], p["v"][: i + 1]   # 잘린 배열만 씀
            pk = int(np.argmax(hh))
            H = hh[pk]
            R = H - p["P"]
            T = p["t"][i]
            tmin = T.hour * 60 + T.minute - 570
            nxt = p["o"][i + 1] if i + 1 < len(c) else np.nan
            rows.append(dict(per=per, sym=s, day=day, T=T, i=i, tmin=tmin, P=p["P"], H=H, cT=cc[-1], nxt=nxt,
                             close=p["close"], Y1=p["close"] / cc[-1] - 1, Y2=float(p["close"] >= p["P"] + 0.5 * R),
                             Y1n=p["close"] / nxt - 1 if np.isfinite(nxt) else np.nan,
                             spike=H / p["P"] - 1, prev=p["prev"], tr5=p["tr5"],
                             big15=big15_of(H, cc[pk:], p["mins"]) / R, gb_min=(i - pk) * p["mins"],
                             t_peak=p["t"][pk].hour * 60 + p["t"][pk].minute - 570, gbT=(H - cc[-1]) / R,
                             low_p=(ll[pk:].min() - p["P"]) / R, atr20=p["atr20"], vol20=p["vol20"],
                             early30=float(tmin < 30), sub=SUB.get(s, "단독"), mins=p["mins"]))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 통계(내 구현)
def cl(y, X, g):
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(y)), X])
    n, k = X.shape
    A = np.linalg.pinv(X.T @ X)
    b = A @ X.T @ y
    e = y - X @ b
    gi = pd.factorize(np.asarray(g))[0]
    G = gi.max() + 1
    S = np.zeros((G, k))
    np.add.at(S, gi, X * e[:, None])
    V = G / max(G - 1, 1) * (n - 1) / max(n - k, 1) * A @ S.T @ S @ A
    return b, b / np.sqrt(np.maximum(np.diag(V), 1e-300))


def dif(x, fl, y="Y1"):
    yy = x[y].values.astype(float)
    ok = np.isfinite(yy)
    yy, fl2, g = yy[ok], fl[ok], x.day.values[ok]
    if fl2.sum() < 3 or (1 - fl2).sum() < 3:
        return np.nan, np.nan, int(fl2.sum()), int((1 - fl2).sum())
    b, t = cl(yy, fl2, g)
    return b[1], t[1], int(fl2.sum()), int((1 - fl2).sum())


def dif_ctrl(x, fl, cols, y="Y1"):
    Z = x[cols].astype(float).values
    yy = x[y].values.astype(float)
    ok = np.isfinite(Z).all(1) & np.isfinite(yy)
    if fl[ok].sum() < 3 or (1 - fl[ok]).sum() < 3:
        return np.nan, np.nan
    b, t = cl(yy[ok], np.column_stack([fl[ok], Z[ok]]), x.day.values[ok])
    return b[1], t[1]


def dif_dayfe(x, fl, y="Y1"):
    """같은 날 고정효과: 날짜별로 빼고 기울기, 날짜 군집 t."""
    d = pd.DataFrame({"y": x[y].values.astype(float), "f": fl, "g": x.day.values}).dropna()
    d["yx"] = d.y - d.groupby("g").y.transform("mean")
    d["fx"] = d.f - d.groupby("g").f.transform("mean")
    sxx = (d.fx ** 2).sum()
    if sxx < 1e-9:
        return np.nan, np.nan, 0
    b = (d.fx * d.yx).sum() / sxx
    e = d.yx - b * d.fx
    sc = (d.fx * e).groupby(d.g).sum()
    G = (d.groupby("g").fx.apply(lambda v: (v.abs() > 1e-12).any())).sum()
    se = np.sqrt(G / max(G - 1, 1) * (sc ** 2).sum()) / sxx
    return b, b / se, int(G)


def flag(x, f, thr):
    return (x[f].astype(float).values > thr).astype(float)


def sub_of(x, f):
    x = x[x.th_n >= 1] if f in THEME_ONLY else x
    return x[np.isfinite(x[f].astype(float)) & np.isfinite(x.Y1.astype(float))]


# ------------------------------------------------------------------ 본문
def main():
    t0 = time.time()
    say("반박 검증: pat_spike.py (verify_pat_spike.py, 2026-10-10)")
    say("사건 정의(분석가가 시험 전 고정, 그대로 씀): 정규장 봉, P = 전날 정규장 종가, H = 그날 봉 t까지 누적 고가.")
    say("  T = H/P-1 >= +6% 이고 봉 종가가 상승분(H-P)의 70% 이상 반납한 첫 봉(V0). V1 +8%/70%, V2 +6%/50%. 종목·날짜당 1건.")
    say("  Y1 = 그날 종가 / T 봉 종가 - 1. Y2 = 종가 >= P + 0.5(H-P). 고르기 = 앞 7/14~9/8(5분봉), 확인 = 뒤 9/10~10/8(3분봉).")
    say("  이 검증이 추가한 결과 변형: Y1n = 그날 종가 / T 다음 봉 시가 - 1(실행 가능한 진입). 이웃 정의 4개는 검증용 추가 시험이다.")
    say("")
    PA = {per: load(per) for per in PER}
    say(f"[데이터] 불러오기 {time.time() - t0:.0f}초. 날짜 수: " + ", ".join(f"{k} {len(v)}" for k, v in PA.items()))
    EV = {(per, dn): events(PA[per], per, *DEFS[dn]) for per in PER for dn in DEFS}
    AN = pd.read_csv(os.path.join(RES, "pat_spike_events.csv"))
    AN["day"] = pd.to_datetime(AN.day).dt.date

    # [1] 재현
    say("")
    say("[1] 재현: 내 사건 찾기 vs 분석가 events.csv")
    say(f"{'표본':<10}{'내 건수':>8}{'분석가':>8}{'같은 종목·날짜':>14}{'T 같음':>8}{'Y1 최대차':>10}{'prev 최대차':>11}{'big15 최대차':>12}{'spike 최대차':>12}")
    for (per, dn), x in EV.items():
        a = AN[(AN.per == per) & (AN.dfn == dn)]
        m = x.merge(a, on=["sym", "day"], suffixes=("", "_a"))
        sameT = (pd.to_datetime(m["T"].astype(str), utc=True) == pd.to_datetime(m["T_a"], utc=True)).mean() if len(m) else np.nan
        mx = lambda c: np.nanmax(np.abs(m[c].values - m[c + "_a"].values)) if len(m) else np.nan
        say(f"{per + ' ' + dn:<10}{len(x):>8}{len(a):>8}{len(m):>14}{sameT:>8.0%}{mx('Y1') if per != '10/9' else 0:>10.1e}"
            f"{mx('prev'):>11.1e}{mx('big15'):>12.1e}{mx('spike'):>12.1e}")

    # [2] 미래 참조
    say("")
    say("[2] 미래 참조 점검")
    # P와 전날 1분봉 마지막 정규장 종가
    errs = []
    for s in SYMS:
        p1 = f"{RAW_L}/1m_{s}.parquet"
        if not os.path.exists(p1):
            continue
        m1 = pd.read_parquet(p1)
        mod = m1.index.hour * 60 + m1.index.minute
        reg = m1[(mod >= 570) & (mod < 960)]
        last = reg.groupby(reg.index.date).Close.last()
        dd = daily(s).Close
        j = pd.concat([last.rename("m"), dd.rename("d")], axis=1).dropna()
        errs.append(np.abs(j.m / j.d - 1).max())
    say(f"  일봉 날짜 맞춤: 1분봉 마지막 정규장 종가 vs 같은 날 일봉 종가 최대 차 {np.nanmax(errs):.2%}(종목별 최대의 중앙 {np.nanmedian(errs):.3%}). P는 전날 행, 하루 밀림 없음.")
    say("  특성: 내 코드는 T 봉까지 자른 배열(h[:i+1], c[:i+1], l[:i+1])과 전날까지 일봉만으로 계산했고 [1]에서 분석가 값과 같다.")
    say("  분석가 코드의 T 이후 값은 post_max(설명용)뿐이고 고르기·확인에 쓰지 않았다(코드 확인).")
    x = EV[("뒤", "V0")]
    gap = (x.nxt / x.cT - 1)
    say(f"  T 봉 종가 → 다음 봉 시가 차: 뒤 V0 평균 {pc(gap.mean())}, |차|>1% {np.mean(np.abs(gap) > 0.01):.0%}. 다음 봉 시가 진입 Y1n 평균 {pc(x.Y1n.mean())} (Y1 {pc(x.Y1.mean())})")

    # 1분 꼬리: 뒤 기간 사건의 고점과 T 봉 종가
    def tails(x):
        hw, cw = [], []
        cache = {}
        for r in x.itertuples():
            if r.sym not in cache:
                cache[r.sym] = pd.read_parquet(f"{RAW_L}/1m_{r.sym}.parquet")
            m1 = cache[r.sym]
            d = m1[m1.index.date == r.day]
            mod = d.index.hour * 60 + d.index.minute
            end = r.T + pd.Timedelta(minutes=r.mins)
            reg = d[(mod >= 570) & (d.index < end)]
            k = int(np.argmax(reg.High.values))
            pos = d.index.get_loc(reg.index[k])
            prevc = d.Close.iloc[pos - 1] if pos > 0 else np.nan
            ref = np.nanmax([reg.Open.iloc[k], reg.Close.iloc[k], prevc])
            hw.append(reg.High.iloc[k] / ref - 1)
            last = d[d.index < end].iloc[-1]
            after = d[d.index >= end]
            nx = after.Open.iloc[0] if len(after) else np.nan
            cw.append(np.nanmin([last.Open, nx]) / last.Close - 1)
        return np.array(hw), np.array(cw)

    for key in (("뒤", "V0"), ("뒤", "V2")):
        hw, cw = tails(EV[key])
        EV[key]["hwick"], EV[key]["cwick"] = hw, cw
        say(f"  {key[0]} {key[1]}: 고점 1분이 꼬리(그 분 시가·종가·직전 분 종가보다 1% 넘게 위) {np.sum(hw > 0.01)}건/{len(hw)},"
            f" T 봉 종가가 1분 꼬리(그 분 시가·다음 분 시가보다 1% 넘게 아래) {np.sum(cw > 0.01)}건")

    # [3] 후보 재계산
    say("")
    say("[3] 후보 재계산(문턱 = 앞 같은 정의 중앙값, 방향 = 앞 부호). 차 = 높은 쪽 - 낮은 쪽 Y1, t = 날짜 군집")
    E0 = EV[("앞", "V0")]
    a_thr = {f: float(E0[f].median()) for f in ("prev", "big15", "spike")}
    sign = {}
    for f in ("prev", "big15", "spike"):
        b, t, n1, n0 = dif(E0, flag(E0, f, a_thr[f]))
        sign[f] = np.sign(b)
        tick()
        say(f"  {f}: 앞 문턱 {a_thr[f]:.4f}, 앞 차 {pc(b)} (t {t:.2f}, n {n1}/{n0}) → 방향 {'높을수록 좋음' if b > 0 else '높을수록 나쁨'}")
    say("")
    res = {}
    for f in ("prev", "big15", "spike"):
        say(f"  [{f}]")
        for dn in DEFS:
            thr = float(EV[("앞", dn)][f].median())
            for per in ("앞", "뒤", "뒤5분"):
                x = EV[(per, dn)]
                fl = flag(x, f, thr)
                b, t, n1, n0 = dif(x, fl)
                b2, t2, _, _ = dif(x, fl, "Y2")
                tick(2)
                say(f"    {per:<4}{dn} 문턱 {thr:.3f}: n {n1}/{n0} 차 {pc(b)} (t {t:.2f}), Y2 차 {pc(b2, 0)} (t {t2:.2f})")
        x = EV[("뒤", "V0")]
        thr = a_thr[f]
        fl = flag(x, f, thr)
        b, t, n1, n0 = dif(x, fl)
        res[f] = dict(b=b, t=t)
        rows = []
        rows.append(("Y1n(다음 봉 시가 진입)", *dif(x, fl, "Y1n")[:2]))
        rows.append(("통제 atr20·vol20·T<10시", *dif_ctrl(x, fl, ["atr20", "vol20", "early30"])))
        rows.append(("통제 +T분·gb_min·gbT·spike", *dif_ctrl(x, fl, ["atr20", "vol20", "early30", "tmin", "gb_min", "gbT", "spike"])))
        bfe, tfe, gfe = dif_dayfe(x, fl)
        rows.append((f"같은 날 고정효과(날짜 {gfe})", bfe, tfe))
        for nm, m in (("T<10시만", x.early30.values == 1), ("T>=10시만", x.early30.values == 0),
                      ("고점 1분 꼬리 제외", x.hwick.values <= 0.01), ("T 종가 꼬리 제외", x.cwick.values <= 0.01),
                      ("뒤 앞절반 9/10~9/23", x.day.values < date(2026, 9, 24)), ("뒤 뒤절반 9/24~10/8", x.day.values >= date(2026, 9, 24))):
            rr = dif(x[m], fl[m])
            rows.append((nm + f" n {rr[2]}/{rr[3]}", rr[0], rr[1]))
        lo, hi = np.nanpercentile(x.Y1, [5, 95])
        xw = x.assign(Y1=x.Y1.clip(lo, hi))
        rows.append(("Y1 상하 5% 자름", *dif(xw, fl)[:2]))
        med = np.median(x.Y1[fl == 1]) - np.median(x.Y1[fl == 0])
        rows.append(("중앙값 차(t 없음)", med, np.nan))
        lod = [(d_, *dif(x[x.day.values != d_], fl[x.day.values != d_])[:2]) for d_ in sorted(x.day.unique())]
        los = [(s_, *dif(x[x.sym.values != s_], fl[x.sym.values != s_])[:2]) for s_ in sorted(x.sym.unique())]
        tick(len(rows) + len(lod) + len(los))
        say(f"    뒤 V0 기준: 차 {pc(b)} (t {t:.2f}, n {n1}/{n0})")
        for nm, bb, tt in rows:
            say(f"      {nm:<28} {pc(bb)} (t {tt:.2f})")
        wd_b = max(lod, key=lambda r: sign[f] * r[1] * -1)
        say(f"      하루 빼기 16가지: 차 범위 {pc(min(r[1] for r in lod))} ~ {pc(max(r[1] for r in lod))},"
            f" 가장 유리한 날({wd_b[0]}) 빼면 {pc(wd_b[1])} (t {wd_b[2]:.2f})")
        ws = max(los, key=lambda r: sign[f] * r[1] * -1)
        say(f"      1종목 빼기 {len(los)}가지: 가장 유리한 종목({ws[0]}) 빼면 {pc(ws[1])} (t {ws[2]:.2f})")
        # 10/6 + TWST 함께
        m = (x.day.values != date(2026, 10, 6))
        rr = dif(x[m], fl[m])
        say(f"      10/6 뺀 뒤 V0: {pc(rr[0])} (t {rr[1]:.2f}, n {rr[2]}/{rr[3]})")
        tick()
        # 이웃 정의
        nb = []
        for nn, (th, gb) in NEIGH.items():
            ea = events(PA["앞"], "앞", th, gb)
            el = events(PA["뒤"], "뒤", th, gb)
            tq = float(ea[f].median())
            ba, ta, _, _ = dif(ea, flag(ea, f, tq))
            bl, tl, n1_, n0_ = dif(el, flag(el, f, tq))
            nb.append(f"{nn} 앞 {pc(ba)}(t {ta:.2f}) 뒤 {pc(bl)}(t {tl:.2f}, n {n1_}/{n0_})")
            tick(2)
        say("      이웃 정의: " + " / ".join(nb))
        # 이웃 문턱(앞 분위 40~60%)과 문턱 없는 연속 기울기(1표준편차당 Y1, 같은 날 고정효과 포함)
        qs = []
        for q in (0.4, 0.45, 0.5, 0.55, 0.6):
            tq = float(E0[f].quantile(q))
            bl, tl, n1_, n0_ = dif(x, flag(x, f, tq))
            qs.append(f"{int(q * 100)}%({tq:.3f}) {pc(bl)}(t {tl:.2f}, {n1_}/{n0_})")
            tick()
        say("      이웃 문턱(앞 분위): " + " / ".join(qs))
        for per in ("앞", "뒤"):
            xx = EV[(per, "V0")]
            z = (xx[f] - xx[f].mean()) / xx[f].std()
            bb, tt = cl(xx.Y1.values, z.values, xx.day.values)
            bf, tf_, gf = dif_dayfe(xx.assign(_z=z), z.values)
            tick(2)
            say(f"      연속 기울기 {per}: {pc(bb[1])}/1표준편차 (t {tt[1]:.2f}), 같은 날 고정효과 {pc(bf)} (t {tf_:.2f}, 날짜 {gf})")
        say("")

    # big15과 시간·반납 속도
    for per in ("앞", "뒤"):
        x = EV[(per, "V0")]
        hi_ = flag(x, "big15", a_thr["big15"])
        say(f"  big15 높음 비율: {per} T<10시 {hi_[x.early30.values == 1].mean():.0%} (n {int((x.early30 == 1).sum())}),"
            f" T>=10시 {hi_[x.early30.values == 0].mean():.0%}; 순위상관 big15~gb_min {x.big15.rank().corr(x.gb_min.rank()):+.2f},"
            f" big15~gbT {x.big15.rank().corr(x.gbT.rank()):+.2f}, big15~T분 {x.big15.rank().corr(x.tmin.rank()):+.2f}")
    xa = EV[("앞", "V0")]
    fa = flag(xa, "big15", a_thr["big15"])
    for nm, m in (("앞 T<10시만", xa.early30.values == 1), ("앞 T>=10시만", xa.early30.values == 0)):
        rr = dif(xa[m], fa[m])
        tick()
        say(f"  big15 {nm}: {pc(rr[0])} (t {rr[1]:.2f}, n {rr[2]}/{rr[3]})")
    b_, t_ = dif_ctrl(xa, fa, ["atr20", "vol20", "early30", "tmin", "gb_min", "gbT", "spike"])
    bfe, tfe, gfe = dif_dayfe(xa, fa)
    tick(2)
    say(f"  big15 앞 통제(+T분·gb_min·gbT·spike): {pc(b_)} (t {t_:.2f}); 앞 같은 날 고정효과 {pc(bfe)} (t {tfe:.2f}, 날짜 {gfe})")
    xp = EV[("앞", "V0")]
    fp = flag(xp, "prev", a_thr["prev"])
    bfe, tfe, gfe = dif_dayfe(xp, fp)
    b_, t_ = dif_ctrl(xp, fp, ["atr20", "vol20", "early30", "tmin", "gb_min", "gbT", "spike"])
    tick(2)
    say(f"  prev 앞 같은 날 고정효과: {pc(bfe)} (t {tfe:.2f}, 날짜 {gfe}); 앞 통제(+T분 등): {pc(b_)} (t {t_:.2f})")

    # [4] 우연 대조
    say("")
    say("[4] 우연 대조(분석가 16특성 값 사용, [1]에서 후보 3개는 내 값과 같음)")
    A0 = AN[(AN.per == "앞") & (AN.dfn == "V0")].reset_index(drop=True)
    L0 = AN[(AN.per == "뒤") & (AN.dfn == "V0")].reset_index(drop=True)

    thr16 = {f: BINARY.get(f, float(sub_of(A0, f)[f].astype(float).median())) for f in FEATS}
    FA = {f: (sub_of(A0, f).index.values, flag(sub_of(A0, f), f, thr16[f])) for f in FEATS}
    FL = {f: (sub_of(L0, f).index.values, flag(sub_of(L0, f), f, thr16[f])) for f in FEATS}
    ga, gl = A0.day.values, L0.day.values

    def tval(y, g, idx, fl):
        yy = y[idx]
        if fl.sum() < 3 or (1 - fl).sum() < 3:
            return np.nan, np.nan
        b, t = cl(yy, fl, g[idx])
        return b[1], t[1]

    def pipeline(ya, yl):
        st = []
        for f in FEATS:
            b, t = tval(ya, ga, *FA[f])
            st.append((abs(t) if np.isfinite(t) else -1, f, np.sign(b), t))
        st.sort(reverse=True)
        top = st[:3]
        weak, conf, sgn = 0, 0, []
        for _, f, s_, ta in top:
            b, t = tval(yl, gl, *FL[f])
            v = s_ * t if np.isfinite(t) else 0.0
            sgn.append(v)
            weak += v >= 1
            conf += v >= 2
        return dict(top=top[0][0], weak=weak, conf=conf, mean_t=np.mean(sgn), first_t=sgn[0])

    obs = pipeline(A0.Y1.values.astype(float), L0.Y1.values.astype(float))
    say(f"  관측: 앞 최대 |t| {obs['top']:.2f}, 뒤에서 같은 방향 |t|>=1인 후보 {obs['weak']}/3, |t|>=2 {obs['conf']}/3,"
        f" 뒤 방향 맞춘 t 평균 {obs['mean_t']:.2f}, 1등 후보 뒤 t {obs['first_t']:.2f}")

    def shuffle_within(y, g):
        y2 = y.copy()
        for d_ in np.unique(g):
            ix = np.where(g == d_)[0]
            y2[ix] = y[RNG.permutation(ix)]
        return y2

    NP = 500
    for nm, fn in (("(a) 사건 단위 섞기", lambda y, g: RNG.permutation(y)), ("(b) 같은 날 안 섞기", shuffle_within)):
        sims = [pipeline(fn(A0.Y1.values.astype(float), ga), fn(L0.Y1.values.astype(float), gl)) for _ in range(NP)]
        S = pd.DataFrame(sims)
        say(f"  {nm} {NP}회: 앞 최대 |t| 95% 분위 {S.top.quantile(.95):.2f}(관측 {obs['top']:.2f}, 이상 비율 {np.mean(S.top >= obs['top']):.1%});"
            f" 뒤 '약함 이상' 2개 이상 {np.mean(S.weak >= obs['weak']):.1%}; 뒤 방향 맞춘 t 평균 >= 관측 {np.mean(S.mean_t >= obs['mean_t']):.1%};"
            f" 1등 후보 뒤 t >= 관측 {np.mean(S.first_t >= obs['first_t']):.1%}")
    tick(32)

    # (c) 뒤 기간 고정 문턱, 같은 날 안 섞기
    say("  (c) 뒤 V0, 앞 문턱 고정, 같은 날 안에서 결과 섞기 2,000회(한쪽 p = 섞은 차가 관측만큼 규칙 방향으로 클 확률):")
    for per in ("앞", "뒤"):
        x = EV[(per, "V0")]
        g = x.day.values
        y = x.Y1.values.astype(float)
        for f in ("prev", "big15"):
            fl = flag(x, f, a_thr[f])
            ob = y[fl == 1].mean() - y[fl == 0].mean()
            nul = []
            for _ in range(2000):
                yy = shuffle_within(y, g)
                nul.append(yy[fl == 1].mean() - yy[fl == 0].mean())
            nul = np.array(nul)
            pv = np.mean(sign[f] * nul >= sign[f] * ob)
            say(f"    {per} {f}: 관측 차 {pc(ob)}, 같은 날 안 섞기 평균 {pc(nul.mean())}(날짜 구성만으로 생기는 차), p {pv:.3f}")
            tick()

    # [5] 묶음 안 비교
    say("")
    say("[5] 묶음 안 비교")
    for per in ("앞", "뒤"):
        x = AN[(AN.per == per) & (AN.dfn == "V0")].copy()
        x = x[x["sub"] != "단독"]
        x["grp"] = x.day.astype(str) + "|" + x["sub"]
        x = x[x.grp.map(x.grp.value_counts()) >= 2]
        x["Tn"] = pd.to_datetime(x["T"], utc=True)
        acc = {"near": {"th_gb": [], "gb_min": [], "tmin": []}, "far": {"th_gb": [], "gb_min": [], "tmin": []}}
        for _, gx in x.groupby("grp"):
            gx = gx.reset_index(drop=True)
            for i in range(len(gx)):
                for j in range(i + 1, len(gx)):
                    dt = abs((gx.Tn[i] - gx.Tn[j]).total_seconds()) / 60
                    k = "near" if dt <= 2 * (5 if per == "앞" else 3) else "far"
                    dy = np.sign(gx.Y1[i] - gx.Y1[j])
                    for f, s_ in (("th_gb", -1), ("gb_min", 1), ("tmin", 1)):
                        df_ = np.sign(gx[f][i] - gx[f][j])
                        if df_ != 0 and dy != 0 and np.isfinite(df_):
                            acc[k][f].append(s_ * df_ == dy)
        say(f"  {per} V0 자기 T 기준 쌍 일치(규칙 방향: 덜 반납·천천히·늦은 T가 나음): "
            + "; ".join(f"T 차 {'<=2봉' if k == 'near' else '>2봉'} " + ", ".join(f"{f} {np.mean(v):.0%}({len(v)}쌍)" if v else f"{f} -" for f, v in acc[k].items())
                        for k in ("near", "far")))
        # 묶음 안 gb_min을 T 시각과 같이 넣기
        xx = x.copy()
        for c_ in ("Y1", "gb_min", "tmin", "th_gb"):
            xx[c_ + "_x"] = xx[c_].astype(float) - xx.groupby("grp")[c_].transform("mean")
        for f in ("gb_min", "th_gb"):
            ok = np.isfinite(xx[f + "_x"]) & np.isfinite(xx.tmin_x)
            zf = xx[f + "_x"][ok] / xx[f + "_x"][ok].std()
            zt = xx.tmin_x[ok] / xx.tmin_x[ok].std()
            b1, t1 = cl(xx.Y1_x[ok].values, zf.values, xx.day[ok].values)
            b2, t2 = cl(xx.Y1_x[ok].values, np.column_stack([zf.values, zt.values]), xx.day[ok].values)
            tick(2)
            say(f"    {f} 묶음 안 기울기(1표준편차당 Y1): 단독 {pc(b1[1])} (t {t1[1]:.2f}) → T 시각 함께 {pc(b2[1])} (t {t2[1]:.2f}), T 시각 {pc(b2[2])} (t {t2[2]:.2f})")

    # 4c 같은 시각 비교(내 코드)
    say("  4c 같은 시각 비교(내 코드): 묶음(같은 날·소분류, 급등 문턱 넘은 종목 2개 이상) 중 누군가 처음 반납 문턱을 넘은 봉 T*")
    for per in ("앞", "뒤", "뒤5분"):
        for dn in ("V0", "V2"):
            thr, gbt = DEFS[dn]
            rows = []
            for day, S in PA[per].items():
                grp = {}
                for s in S:
                    if s in SUB:
                        grp.setdefault(SUB[s], []).append(s)
                for sub, mem in grp.items():
                    if len(mem) < 2:
                        continue
                    allt = sorted(set().union(*[set(S[s]["t"]) for s in mem]))
                    for T in allt:
                        st = []
                        for s in mem:
                            p = S[s]
                            k = int(np.searchsorted(p["t"].values, T.to_datetime64(), side="right")) - 1
                            if k < 0:
                                continue
                            Hh = p["h"][: k + 1].max()
                            if Hh / p["P"] - 1 >= thr:
                                st.append((s, (Hh - p["c"][k]) / (Hh - p["P"]), p["close"] / p["c"][k] - 1))
                        if len(st) >= 2 and max(z[1] for z in st) >= gbt:
                            for s, g_, y_ in st:
                                rows.append(dict(day=day, sub=sub, sym=s, gb=g_, Y1=y_))
                            break
            C = pd.DataFrame(rows)
            if not len(C):
                continue
            ag, dl = [], []
            for (d_, sb), gx in C.groupby(["day", "sub"]):
                a, r = gx.gb.values, gx.Y1.values
                for i in range(len(a)):
                    for j in range(i + 1, len(a)):
                        if a[i] != a[j] and r[i] != r[j]:
                            ag.append(np.sign(a[i] - a[j]) == np.sign(r[i] - r[j]))
                dl.append((d_, gx.Y1.values[np.argmin(a)] - gx.Y1.values[np.argmax(a)]))
            dd = np.array([z[1] for z in dl])
            b, t = cl(dd, np.zeros((len(dd), 0)), [z[0] for z in dl]) if len(dd) > 2 else ([np.nan], [np.nan])
            tick(2)
            say(f"    {per} {dn}: 묶음 {len(dl)}, 쌍 일치(더 반납 쪽이 더 오름) {np.mean(ag):.0%} ({len(ag)}쌍),"
                f" 덜 반납 1등 - 더 반납 1등 Y1 {pc(dd.mean())} (t {t[0]:.2f}), 덜 반납 쪽이 나은 묶음 {np.mean(dd > 0):.0%}")

    # [6] 10/9 사례
    say("")
    say("[6] 10/9 광 전송·광모듈(데이터 10:18 ET까지, 종가 없음)")
    S9 = PA["10/9"].get(date(2026, 10, 9), {})
    T42 = pd.Timestamp("2026-10-09 09:42", tz="America/New_York")
    for s in ("LITE", "COHR", "AAOI", "CIEN", "FN", "VIAV"):
        if s not in S9:
            continue
        p = S9[s]
        runH = np.maximum.accumulate(p["h"])
        R = runH - p["P"]
        gb = np.where(R > 0, (runH - p["c"]) / R, np.nan)
        k = int(np.searchsorted(p["t"].values, T42.to_datetime64(), side="right")) - 1
        m1 = pd.read_parquet(f"{RAW_L}/1m_{s}.parquet")
        d = m1[m1.index.date == date(2026, 10, 9)]
        mod = d.index.hour * 60 + d.index.minute
        reg = d[mod >= 570]
        kk = int(np.argmax(reg.High.values))
        pos = d.index.get_loc(reg.index[kk])
        ref = np.nanmax([reg.Open.iloc[kk], reg.Close.iloc[kk], d.Close.iloc[pos - 1]])
        say(f"  {s:<5} P {p['P']:.2f} prev {pc(p['prev'], 1)} (> {a_thr['prev']:.2%}? {'예' if p['prev'] > a_thr['prev'] else '아니오'}),"
            f" 고점 {pc(runH[-1] / p['P'] - 1, 1)} {reg.index[kk].strftime('%H:%M')} (1분 꼬리 {reg.High.iloc[kk] / ref - 1:+.2%}),"
            f" 09:42 봉 반납 {gb[k]:.0%}, 10:18까지 최대 반납 {np.nanmax(gb):.0%}")
    for dn in ("V0", "V2"):
        x = EV[("10/9", dn)]
        x = x[x["sub"] == "광 전송·광모듈"]
        thr_b = float(EV[("앞", dn)]["big15"].median())
        for r in x.itertuples():
            say(f"  {dn} {r.sym:<5} T {r.T.strftime('%H:%M')} 급등 {pc(r.spike, 1)} gbT {r.gbT:.2f} big15 {r.big15:.2f}"
                f" (앞 {dn} 문턱 {thr_b:.3f} 넘음? {'예' if r.big15 > thr_b else '아니오'}) gb_min {r.gb_min} prev {pc(r.prev, 1)}")
        if not len(x):
            say(f"  {dn}: 광모듈 사건 없음")
    say("")
    say(f"이 검증의 시험 수: 약 {NT['n']}개(우연 대조 반복 제외). 걸린 시간 {time.time() - t0:.0f}초.")
    open(os.path.join(RES, "verify_pat_spike.txt"), "w", encoding="utf-8").write("\n".join(OUT))


if __name__ == "__main__":
    main()
