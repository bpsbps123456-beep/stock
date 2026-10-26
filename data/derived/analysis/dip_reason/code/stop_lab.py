"""칼손절 연구 공용 시험대(2026-10-07). 지금 지표(3분봉 EMA60 -2% 매수, RSI14>70 또는 다음 아침 8시 매도,
전날 +7% 다음 날 정규장 매수 안 함)를 기준으로, 매수 뒤 '나가기' 규칙만 바꿔 같은 잣대로 비교한다.

사용(다른 스크립트에서):
    from stop_lab import load, run, compare, base_trades
    S = load("late")                  # {"early": 7/14~9/8 5분봉 근사(EMA36·RSI9), "late": 9/10~10/6 3분봉(EMA60·RSI14)}
    def make_rule(A):                 # 종목 하나의 배열 묶음 A를 받아 규칙 함수를 돌려준다
        c = A["c"]
        def rule(i, ei, ent):         # 들고 있는 봉 i의 시가에 팔지 정한다. ei = 매수 봉, ent = 매수가
            return c[i - 1] <= ent * 0.95     # i-1 봉까지의 값만 써야 한다(미래 참조 금지)
        return rule
    compare(make_rule, "고정 -5%")    # 두 기간 × 두 범위 × 두 종목 묶음 표를 출력하고 DataFrame을 돌려준다

규칙 함수는 봉 i의 '시가에 팔지'만 정한다. i 봉의 고가·저가·종가·거래량을 쓰면 미래 참조다(A[...][i-1]까지만).
매수 봉 ei의 종가·저가는 i >= ei+1일 때 쓸 수 있다(i-1 >= ei).
손절로 판 뒤에는 원래 정리 시각(다음 아침 8시)까지 그 종목을 다시 사지 않는다(cool=True, 10/7 연구: 곧바로 재매수는 급락일을 키움).
매도 순서는 Pine과 같다: RSI 매도 → 손절 → 8시 정리.

A의 키: o h l c v(거래량, 프리·애프터는 0) r(기준 RSI) ema(기준 EMA) tk(봉 시각 ns) dead(그 봉에 샀다면 정리 시각 ns)
        reg(정규장 bool) ses('정규'·'프리'·'애프터') day(ET 날짜, datetime.date) mod(ET 0시부터 분) hot7(전날 +7% 다음 날 정규장)
        idx(DatetimeIndex, ET) sym mins(봉 길이 분)
다른 종목·ETF 값이 필요하면 load()의 다른 종목을 쓰거나 RAW의 1m_/5m_/1d_<티커>.parquet(QQQ SMH IGV ARKG XBI IWM SPY 포함)를 읽는다.
"""
import os, sys
from datetime import date
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from events import bars, prep
from fetch import universe, user_universe
from rules import hot_days

RAW = r"C:/Users/geunt/AppData/Local/Temp/claude/C--Users-geunt-OneDrive-Desktop---/4e686c73-74be-4337-8ccc-2f6136beaf7d/scratchpad/dip/raw"
CACHE = os.path.join(os.path.dirname(RAW), "stop_lab_{}")
PERIODS = {"early": (5, 36, 9, date(2026, 7, 14), date(2026, 9, 8)),
           "late": (3, 60, 14, date(2026, 9, 10), date(2026, 10, 6))}
PNAME = {"early": "앞 7/14~9/8(5분봉 근사)", "late": "뒤 9/10~10/6(3분봉)"}
GAP, X = 0.02, 70.0
USER78 = set(user_universe()["합집합"])
REF127 = set(universe())
ALL = sorted(USER78 | REF127)
UNIV = {"사용자78": USER78, "전체137": set(ALL)}
NUM = ("o", "h", "l", "c", "v", "r", "ema", "tk", "dead", "reg", "mod", "hot7")


def _arrays(s, mins, el, rl):
    d = prep(bars(RAW, s, mins), el, rl)
    idx = d.index
    kst = idx.tz_convert("Asia/Seoul")
    t8 = kst.normalize() + pd.Timedelta(hours=8)
    dead = np.where(kst < t8, t8.as_unit("ns").asi8, (t8 + pd.Timedelta(days=1)).as_unit("ns").asi8).astype(np.int64)
    return dict(sym=s, mins=mins, idx=idx, o=d.o.values.astype(float), h=d.h.values.astype(float),
                l=d.l.values.astype(float), c=d.c.values.astype(float), v=d.v.values.astype(float),
                r=d.r.values.astype(float), ema=d.ema.values.astype(float), tk=idx.as_unit("ns").asi8, dead=dead,
                reg=(d.ses.to_numpy(dtype="U8") == "정규"), ses=d.ses.to_numpy(dtype="U8"), day=np.asarray(d.date),
                mod=d["mod"].values.astype(np.int64), hot7=hot_days(RAW, s, d, 0.07))


def _save(A, path):
    np.savez(path, ses=A["ses"], dayi=np.array([x.year * 10000 + x.month * 100 + x.day for x in A["day"]]),
             mins=np.array([A["mins"]]), **{k: A[k] for k in NUM})


def _read(path, s):
    z = np.load(path, allow_pickle=False)
    A = {k: z[k] for k in NUM}
    A["sym"], A["mins"], A["ses"] = s, int(z["mins"][0]), z["ses"]
    A["day"] = np.array([date(x // 10000, x // 100 % 100, x % 100) for x in z["dayi"]], dtype=object)
    A["idx"] = pd.DatetimeIndex(A["tk"], tz="UTC").tz_convert("America/New_York")
    return A


def load(period):
    """{티커: 배열 묶음} — 처음 한 번 만들어 scratchpad에 npz로 저장한다(pickle 안 씀)."""
    folder = CACHE.format(period)
    os.makedirs(folder, exist_ok=True)
    mins, el, rl, lo, hi = PERIODS[period]
    S = {}
    for s in ALL:
        p = os.path.join(folder, f"{s}.npz")
        if os.path.exists(p):
            S[s] = _read(p, s)
            continue
        try:
            A = _arrays(s, mins, el, rl)
        except FileNotFoundError:
            continue
        _save(A, p)
        S[s] = A
    return S


def sim(A, rule=None, cool=True, reg_only=False):
    """Pine과 같은 상태 기계. 반환: [(매수 봉, 매도 봉, 매수가, 매도가, 이유)]."""
    o, l, r, ema, tk, dead_of = A["o"], A["l"], A["r"], A["ema"], A["tk"], A["dead"]
    reg, hot = A["reg"], A["hot7"]
    out, hold, ent, ei, dead, until = [], False, 0.0, 0, 0, 0
    for i in range(61, len(o)):
        if hold:
            why = None
            if i - 1 >= ei and r[i - 1] > X:
                why = "RSI"
            elif rule is not None and i - 1 >= ei and rule(i, ei, ent):
                why = "손절"
            elif tk[i] >= dead:
                why = "8시"
            if why:
                out.append((ei, i, ent, o[i], why))
                hold = False
                if why == "손절" and cool:
                    until = dead
        else:
            if tk[i] < until or hot[i] or (reg_only and not reg[i]):
                continue
            lvl = ema[i - 1] * (1 - GAP)
            if l[i] <= lvl:
                ent, ei, hold, dead = min(o[i], lvl), i, True, dead_of[i]
    return out


def run(S, make_rule=None, period="late", reg_only=False, cool=True, syms=None):
    """모든 종목 거래표(기간 안 매수만). 열: sym day t ses ret why bars mae ei xi ent px."""
    lo, hi = PERIODS[period][3], PERIODS[period][4]
    rows = []
    for s, A in S.items():
        if syms is not None and s not in syms:
            continue
        rule = make_rule(A) if make_rule else None
        for ei, xi, ent, px, why in sim(A, rule, cool, reg_only):
            dy = A["day"][ei]
            if lo <= dy <= hi:
                rows.append((s, dy, A["idx"][ei], A["ses"][ei], px / ent - 1, why, xi - ei,
                             A["l"][ei:xi + 1].min() / ent - 1, ei, xi, ent, px))
    return pd.DataFrame(rows, columns=["sym", "day", "t", "ses", "ret", "why", "bars", "mae", "ei", "xi", "ent", "px"])


_BASE = {}


def base_trades(period, reg_only=False):
    k = (period, reg_only)
    if k not in _BASE:
        _BASE[k] = run(load(period), None, period, reg_only)
    return _BASE[k]


def metrics(t, b, syms):
    """규칙 거래 t와 기준 거래 b 비교(같은 종목 묶음). 차이 = 합 차이, t = 날짜별 합 차이의 t값."""
    t, b = t[t.sym.isin(syms)], b[b.sym.isin(syms)]
    days = sorted(set(b.day) | set(t.day))
    d = (t.groupby("day").ret.sum().reindex(days, fill_value=0) - b.groupby("day").ret.sum().reindex(days, fill_value=0))
    sd = d.std(ddof=1)
    tt = d.mean() / (sd / np.sqrt(len(d))) if sd > 0 else 0.0
    stops = t[t.why == "손절"]
    oct6 = t[t.day == date(2026, 10, 6)].ret.sum() - b[b.day == date(2026, 10, 6)].ret.sum()
    return dict(건수=len(t), 평균=t.ret.mean(), 승률=(t.ret > 0).mean(), 최악=t.ret.min(), 하위5=t.ret.quantile(.05),
                합=t.ret.sum(), 차이=t.ret.sum() - b.ret.sum(), t=tt, 나은날=int((d > 1e-12).sum()), 나쁜날=int((d < -1e-12).sum()),
                손절=len(stops), 손절평균=stops.ret.mean() if len(stops) else np.nan, 차이_10월6일=oct6)


_HTF = {}


def htf_rsi(A, tf=40, n=14):
    """큰 봉(tf분, 04:00 ET 기준, 프리·애프터 포함) RSI를 기준봉에 붙인다. 반환 (dev, conf) 배열, 둘 다 '봉 i 시가에 판단할 때'
    쓸 수 있는 값이 인덱스 i에 들어 있다(미래 참조 없음).
      dev[i]  = 봉 i-1이 속한 큰 봉을 진행 중으로 보고, 그 전까지 완성된 큰 봉의 RMA 상태에 c[i-1]을 넣은 RSI(트레이딩뷰 실시간 값)
      conf[i] = 봉 i가 시작할 때까지 완성된 마지막 큰 봉의 RSI
    큰 봉은 RAW의 5m_<티커>.parquet로 만든다(1분봉 30일 한도보다 길어 RSI가 일찍 자리 잡음)."""
    key = (A["sym"], A["mins"], len(A["tk"]), tf, n)
    if key in _HTF:
        return _HTF[key]
    from htf import intraday_bars, rma_state, _rsi_from
    m5 = pd.read_parquet(os.path.join(RAW, f"5m_{A['sym']}.parquet"))
    hs, hc = intraday_bars(m5, tf)
    hc = hc.astype(float)
    g, l, r = rma_state(hc, n)
    hs_ns = hs.as_unit("ns").asi8
    step = tf * 60 * 10**9
    tk = A["tk"]
    prev_t = np.r_[tk[0], tk[:-1]]                       # 봉 i-1의 시작 시각
    idx = A["idx"]
    anchor = (idx.normalize() + pd.Timedelta(hours=4)).as_unit("ns").asi8
    anchor_prev = np.r_[anchor[0], anchor[:-1]]
    cur_prev = anchor_prev + ((prev_t - anchor_prev) // step) * step   # 봉 i-1이 속한 큰 봉 시작
    j = np.searchsorted(hs_ns, cur_prev, side="left") - 1             # 그 전 완성 큰 봉
    ok, jj = j >= 0, np.maximum(j, 0)
    c_prev = np.r_[np.nan, A["c"][:-1]]
    dev = np.where(ok, _rsi_from(g[jj], l[jj], hc[jj], c_prev, n), np.nan)
    cur_i = anchor + ((tk - anchor) // step) * step                    # 봉 i가 속한 큰 봉 시작
    jc = np.searchsorted(hs_ns, cur_i, side="left") - 1                # 봉 i 시작 때 완성된 마지막 큰 봉
    conf = np.where(jc >= 0, r[np.maximum(jc, 0)], np.nan)
    dev[0] = conf[0] = np.nan
    _HTF[key] = (dev, conf)
    return dev, conf


def stop_audit(make_rule, period, reg_only=False, syms=None, cool=True):
    """손절된 거래를 같은 매수 봉의 기준 거래와 짝지어 본다.
    반환: 손절 수, 기준이면 수익으로 끝났을 비율(헛손절), 손절 수익 평균, 기준 수익 평균, 아낀 손실 합(손절 - 기준)."""
    syms = syms if syms is not None else UNIV["사용자78"]
    t = run(load(period), make_rule, period, reg_only, cool, syms=syms)
    b = base_trades(period, reg_only)
    b = b[b.sym.isin(syms)].set_index(["sym", "ei"]).ret
    s = t[t.why == "손절"].set_index(["sym", "ei"]).ret
    m = s.to_frame("손절").join(b.rename("기준"), how="inner")
    if not len(m):
        return dict(손절=0)
    return dict(손절=len(s), 짝=len(m), 헛손절비율=(m.기준 > 0).mean(), 손절평균=m.손절.mean(), 기준평균=m.기준.mean(),
                아낀손실합=(m.손절 - m.기준).sum(), 크게잃을거래_막음=int((m.기준 <= -0.05).sum()))


def compare(make_rule, label="", cool=True, show=True):
    """두 기간 × 두 범위(24시간·정규장만) × 두 종목 묶음 결과 표."""
    rows = []
    for period in ("early", "late"):
        S = load(period)
        for reg_only in (False, True):
            b = base_trades(period, reg_only)
            t = run(S, make_rule, period, reg_only, cool)
            for un, us in UNIV.items():
                rows.append(dict(규칙=label, 기간=period, 범위="정규장만" if reg_only else "24시간", 종목=un, **metrics(t, b, us)))
    df = pd.DataFrame(rows)
    if show:
        f = df.copy()
        for k, fmt in (("평균", "{:+.2%}"), ("승률", "{:.0%}"), ("최악", "{:+.1%}"), ("하위5", "{:+.1%}"), ("합", "{:+.2f}"),
                       ("차이", "{:+.2f}"), ("t", "{:+.2f}"), ("손절평균", "{:+.2%}"), ("차이_10월6일", "{:+.2f}")):
            f[k] = f[k].map(lambda v: "" if pd.isna(v) else fmt.format(v))
        print(f.to_string(index=False))
    return df


if __name__ == "__main__":
    for p in ("early", "late"):
        S = load(p)
        for reg in (False, True):
            b = base_trades(p, reg)
            for un, us in UNIV.items():
                x = b[b.sym.isin(us)]
                print(f"{PNAME[p]} {'정규장만' if reg else '24시간'} {un}: 기준 {len(x)}건 합 {x.ret.sum():+.2f} 최악 {x.ret.min():+.1%}")
