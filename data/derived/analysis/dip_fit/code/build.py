"""이격매수매도 종목 궁합 연구(2026-10-09 사용자: 잘 통하는 종목과 안 통하는 종목의 차이, 최근 1개월만) — 거래표와 특성표를 만든다.

규칙은 지금 지표(dip_scalper.pine)와 같다: 3분봉 EMA60[1] x (1-2%)에 저가가 닿으면 min(시가, 선)에 산다(3배 레버리지 목록은 3%),
RSI14 > 70인 봉 다음 봉 시가 또는 매수 뒤 처음 오는 한국 아침 8시 이후 첫 봉 시가에 판다, 전날 정규장 +7% 다음 날 정규장은 사지 않는다.
분석 구간: 매수일(ET) 2026-09-10 ~ 10-08(약 21거래일). 앞 절반 9/10~9/23, 뒤 절반 9/24~10/8.

특성은 모두 구간 시작 전(9/9 종가까지)에 알 수 있는 값이다. 거래 단위 특성은 매수 봉 직전 봉(i-1)까지의 값만 쓴다.
  종목: vol20 atr14 beta60 corr60(QQQ 일봉) ret20 ret60 dist52(52주 고점 대비) ma50 adv20(log 달러 거래대금) price gap60(시가 갭 2% 넘는 날 비율)
        trend60(일봉 |종가-시가|/(고가-저가) 평균: 높을수록 한 방향으로 가는 날) ovn60(밤사이 갭 분산 / 하루 수익률 분산)
        ac1·vr12(7/14~9/9 정규장 5분봉 수익률 1차 자기상관, 1시간 분산비) early_n·early_avg·early_sum(7/14~9/8 같은 규칙 5분봉 근사 성적)
        group(유니버스 묶음) user78 tv_ai lev etf
  거래: ses hour qdev(같은 시각 QQQ 3분봉의 EMA60 대비 위치, i-1 봉) sdev_q(종목 - QQQ, 혼자 빠짐 정도) day_ret(i-1 봉 종가의 전날 정규장 종가 대비)
        prev_ret(전날 정규장 수익률) earn(매수일 -1 ~ 매도일에 실적 발표가 잡혀 있었는가, 실적 일정은 미리 알려짐)
저장: ../results/trades.csv, ../results/features.csv
사용: py build.py                         (raw_1009, 끝 10/8 → trades.csv·features.csv, 1·2부 연구 입력)
      py build.py raw_1010 2026-10-09 _1009  (끝 10/9 → trades_1009.csv, 3부 빠른 반등 연구 입력. 특성표는 다시 만들지 않는다)"""
import os, sys
from datetime import date
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "../../dip_reason/code"))
from events import bars, prep
from rules import hot_days
from fetch import universe, user_universe, TV_AI, ETF
import stop_lab

SP = r"C:/Users/geunt/AppData/Local/Temp/claude/C--Users-geunt-OneDrive-Desktop---/4e686c73-74be-4337-8ccc-2f6136beaf7d/scratchpad/dip"
RAW = os.path.join(SP, "raw_1009")
OUT = os.path.join(HERE, "../results")
LO, MID, HI = date(2026, 9, 10), date(2026, 9, 24), date(2026, 10, 8)
F0 = pd.Timestamp("2026-09-09")                      # 특성 기준일(이날 종가까지)
LEV = set("SOXL,SOXS,TQQQ,SQQQ,TECL,TECS,LABU,LABD,NAIL,DPST,KORU,SPXL,SPXS,UPRO,SPXU,TNA,TZA,FNGU,FNGD,YINN,YANG".split(","))
GRP = universe()
U78 = set(user_universe()["합집합"])
SYMS = sorted((set(GRP) | U78) - set(ETF))


def arrays(s):
    d = prep(bars(RAW, s, 3), 60, 14)
    kst = d.index.tz_convert("Asia/Seoul")
    t8 = kst.normalize() + pd.Timedelta(hours=8)
    dead = np.where(kst < t8, t8.as_unit("ns").asi8, (t8 + pd.Timedelta(days=1)).as_unit("ns").asi8)
    return d, dead, hot_days(RAW, s, d, 0.07)


def sim(d, dead_of, hot, gap):
    o, l, r, ema, tk = d.o.values, d.l.values, d.r.values, d.ema.values, d.index.as_unit("ns").asi8
    out, hold, ent, ei, dead = [], False, 0.0, 0, 0
    for i in range(61, len(d)):
        if hold:
            why = "RSI" if (i - 1 >= ei and r[i - 1] > 70) else ("8시" if tk[i] >= dead else None)
            if why:
                out.append((ei, i, ent, o[i], why))
                hold = False
        elif not hot[i]:
            lvl = ema[i - 1] * (1 - gap)
            if l[i] <= lvl:
                ent, ei, hold, dead = min(o[i], lvl), i, True, dead_of[i]
    return out


def qqq_dev():
    d = prep(bars(RAW, "QQQ", 3), 60, 14)
    return pd.Series((d.c / d.ema - 1).values, index=d.index)


def earn_days():
    e = pd.read_csv(os.path.join(HERE, "../../box_trade/results/earnings_cache.csv"))
    t = pd.to_datetime(e.earnings_dt, utc=True).dt.tz_convert("America/New_York")
    # 장 마감 뒤(16시 이후) 발표는 다음 거래일 반응이지만, 여기서는 발표 날짜와 다음 날을 모두 '실적 구간'으로 본다
    e["d"] = t.dt.date
    return e.groupby("ticker").d.apply(set).to_dict()


def trades():
    qd = qqq_dev()
    ed = earn_days()
    rows = []
    for s in SYMS:
        try:
            d, dead_of, hot = arrays(s)
        except FileNotFoundError:
            continue
        c1d = pd.read_parquet(os.path.join(RAW, f"1d_{s}.parquet")).Close
        c1d.index = pd.to_datetime(c1d.index).date
        prevc = c1d.shift(1)
        pret = (c1d / c1d.shift(1) - 1).shift(1)
        dev = (d.c / d.ema - 1).values
        q = qd.reindex(d.index).ffill().values
        eds = ed.get(s, set())
        for ei, xi, ent, px, why in sim(d, dead_of, hot, 0.03 if s in LEV else 0.02):
            dy = d.date.iloc[ei]
            if not (LO <= dy <= HI):
                continue
            xd = d.date.iloc[xi]
            ern = any(dy - pd.Timedelta(days=1).to_pytimedelta() <= x <= xd for x in eds)
            pc = prevc.get(dy, np.nan)
            rows.append(dict(sym=s, day=dy, t=d.index[ei], ses=d.ses.iloc[ei], hour=d.index[ei].hour, ret=px / ent - 1, why=why,
                             bars=xi - ei, mae=d.l.values[ei:xi + 1].min() / ent - 1, ent=ent, px=px,
                             qdev=q[ei - 1], sdev_q=dev[ei - 1] - q[ei - 1], day_ret=d.c.values[ei - 1] / pc - 1,
                             prev_ret=pret.get(dy, np.nan), earn=ern, half="앞" if dy < MID else "뒤"))
    return pd.DataFrame(rows)


def features():
    q = pd.read_parquet(os.path.join(RAW, "1d_QQQ.parquet"))
    q.index = pd.to_datetime(q.index)
    qr = q.Close.pct_change()
    early = stop_lab.run(stop_lab.load("early"), None, "early")
    eg = early.groupby("sym").ret.agg(["count", "mean", "sum"])
    rows = []
    for s in SYMS:
        p = os.path.join(RAW, f"1d_{s}.parquet")
        if not os.path.exists(p):
            continue
        x = pd.read_parquet(p)
        x.index = pd.to_datetime(x.index)
        x = x[x.index <= F0]
        if len(x) < 70:
            continue
        r = x.Close.pct_change()
        tr = pd.concat([x.High - x.Low, (x.High - x.Close.shift()).abs(), (x.Low - x.Close.shift()).abs()], axis=1).max(axis=1)
        jr = pd.concat([r, qr], axis=1, join="inner").iloc[-60:]
        rng = (x.High - x.Low).replace(0, np.nan)
        gp = x.Open / x.Close.shift() - 1
        f = dict(sym=s, vol20=r.iloc[-20:].std(), atr14=(tr / x.Close).iloc[-14:].mean(),
                 beta60=jr.iloc[:, 0].cov(jr.iloc[:, 1]) / jr.iloc[:, 1].var(), corr60=jr.iloc[:, 0].corr(jr.iloc[:, 1]),
                 ret20=x.Close.iloc[-1] / x.Close.iloc[-21] - 1, ret60=x.Close.iloc[-1] / x.Close.iloc[-61] - 1,
                 dist52=x.Close.iloc[-1] / x.High.iloc[-252:].max() - 1, ma50=x.Close.iloc[-1] / x.Close.iloc[-50:].mean() - 1,
                 adv20=np.log10((x.Close * x.Volume).iloc[-20:].mean()), price=x.Close.iloc[-1],
                 gap60=(gp.iloc[-60:].abs() > 0.02).mean(), trend60=((x.Close - x.Open).abs() / rng).iloc[-60:].mean(),
                 ovn60=gp.iloc[-60:].var() / r.iloc[-60:].var())
        p5 = os.path.join(RAW, f"5m_{s}.parquet")
        if os.path.exists(p5):
            m = pd.read_parquet(p5)
            mod = m.index.hour * 60 + m.index.minute
            m = m[(mod >= 570) & (mod < 960) & (m.index.tz_localize(None) <= F0 + pd.Timedelta(days=1))]
            rr = np.log(m.Close).groupby(m.index.date).diff().dropna()
            f["ac1"] = rr.autocorr(1)
            h = np.log(m.Close).groupby(m.index.date).diff(12).dropna()
            f["vr12"] = h.var() / (12 * rr.var())
        f.update(early_n=eg["count"].get(s, 0), early_avg=eg["mean"].get(s, np.nan), early_sum=eg["sum"].get(s, 0.0),
                 group=GRP.get(s, "사용자 종목"), user78=s in U78, tv_ai=s in TV_AI, lev=s in LEV)
        rows.append(f)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if len(sys.argv) > 1:
        RAW = os.path.join(SP, sys.argv[1])
        HI = date.fromisoformat(sys.argv[2])
    suffix = sys.argv[3] if len(sys.argv) > 3 else ""
    t = trades()
    t.to_csv(os.path.join(OUT, f"trades{suffix}.csv"), index=False, encoding="utf-8-sig")
    if not suffix:
        f = features()
        f.to_csv(os.path.join(OUT, "features.csv"), index=False, encoding="utf-8-sig")
    else:
        f = pd.DataFrame()
    print(f"거래 {len(t)}건, 종목 {t.sym.nunique()}개, 날짜 {t.day.nunique()}일 ({t.day.min()}~{t.day.max()}) | 특성 {len(f)}종목")
    print(t.groupby("half").ret.agg(["count", "mean", "sum"]))
