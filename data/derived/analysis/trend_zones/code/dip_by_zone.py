"""큰 틀(추세 5단계·SOXX 경고) 속에서 이격매수매도를 어떻게 쓸까 (2026-10-11, 연구용).

사용자(10/11): "이 큰 틀 속에서 이격매수매도를 하고 싶다. 조심하려는 거다. 예를 들어 매수(상방) 구간일 때는 이격매수매도가
              더 믿을 만하다든가, 하방 추세일 때는 이격 매수선 밑으로 뚫고 자주 간다든가."

이격매수매도(dip_scalper.pine, 2026-10-08판): 3분봉 저가가 보라선(EMA60)[1] × 0.98에 닿으면 min(시가, 매수선)에 사고,
  RSI14 > 70 다음 봉 시가 또는 매수 뒤 첫 아침 8시(한국) 이후 첫 봉 시가에 판다. 24시간(프리·정규·애프터). 전날 정규장 +7% 이상이면
  그날 정규장 매수 쉼(useHot). 3배 ETF 3%는 ETF를 빼서 해당 없음. 코드는 dip_reason/code(events.prep, rules.hot_days)를 그대로 쓴다.
자료: cache/dip_raw_1010(다른 세션이 10/10에 받은 yfinance 1분·5분·일봉, 144종목 중 ETF 8개 제외 136종목).
  창 1 '5분 근사' 7/20~9/8: 5분봉, EMA36·RSI9(같은 180분 EMA, 약 45분 RSI — dip_reason과 같은 근사)
  창 2 '3분 정확' 9/9~10/9: 1분봉 → 3분봉, EMA60·RSI14
큰 틀(매수 순간에 이미 아는 값):
  추세 5단계 = trend_5zone.pine과 같은 계산(일봉 RSI14 70/30, 60일 흐름 z ±0.5), 그 시각 마지막 완성 정규장 일봉 기준
  SOXX 경고 = 그 주 첫 거래일 직전 종가 SOXX < 20일선
  (참고) 자기 종목 일봉 종가 vs 20일선
재는 것(거래마다): 수익(매수가→매도가), 이긴 비율, RSI 매도 비율, 매수 뒤 매도 전까지 최저가(MAE = 매수선 밑으로 더 뚫린 깊이),
  MAE ≤ -3%·-5% 비율. 차이의 t는 날짜 묶음 부트스트랩(2,000회).
사전 등록 변형(같은 금액 매매의 수익 합·평균·최악으로 비교, 두 창 모두 나아야 '후보'):
  V0 지금 그대로 | V1 하방·극하방이면 매수 안 함 | V2 하방·극하방이면 이격 3% | V3 하방·극하방이면 이격 4%
  V4 SOXX 경고 주면 매수 안 함 | V5 SOXX 경고 주면 이격 3% | V6 상방·극상방에서만 매수
실행: PYTHONIOENCODING=utf-8 py data/derived/analysis/trend_zones/code/dip_by_zone.py
결과: results/dip_by_zone.txt, dip_trades.csv
"""
import os, sys
from datetime import date

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import zones as Zm  # noqa: E402
sys.path.insert(0, os.path.join(Zm.ROOT, "data/derived/analysis/dip_reason/code"))
from events import bars, prep, X, HR  # noqa: E402
from rules import hot_days  # noqa: E402

RAW = os.path.join(HERE, "../cache/dip_raw_1010")
ETF = {"ARKG", "IGV", "IWM", "QQQ", "SMH", "SOXX", "SPY", "XBI"}
WIN = {"5분 근사 7/20~9/8": (5, date(2026, 7, 20), date(2026, 9, 8)),
       "3분 정확 9/9~10/9": (3, date(2026, 9, 9), date(2026, 10, 9))}
WINNER = {"fam": "tx", "t": "mz", "tp": 60, "tau": 0.5, "x": "xrsi", "xv": 70}
NAMES = {2: "극상방", 1: "상방", 0: "애매", -1: "하방", -2: "극하방"}
NY = "America/New_York"


def syms():
    s = sorted(f[3:-8] for f in os.listdir(RAW) if f.startswith("1m_"))
    return [x for x in s if x not in ETF]


# ---------------------------------------------------------------- 큰 틀(일봉)
def daily_context(S):
    fr = {}
    for s in S:
        d = pd.read_parquet(os.path.join(RAW, f"1d_{s}.parquet"))
        d.index = pd.to_datetime(d.index).strftime("%Y-%m-%d")
        fr[s] = d
    pv = {k: pd.DataFrame({s: fr[s][k] for s in S}).sort_index() for k in ("Open", "High", "Low", "Close")}
    Zm.MIN_BARS = 130
    I = Zm.Ind(pv["Open"], pv["High"], pv["Low"], pv["Close"])
    Z = Zm.zones(I, WINNER)
    sma = pv["Close"].rolling(20, min_periods=20).mean()
    above = (pv["Close"] > sma).to_numpy()
    close_t = pd.to_datetime(I.dates).tz_localize(NY) + pd.Timedelta(hours=16)
    return I.dates, close_t.as_unit("ns").asi8, Z, above


def soxx_week_state():
    o = pd.read_parquet(os.path.join(Zm.ROOT, "data/derived/analysis/stop_research/cache/ohlc.parquet"),
                        columns=["date", "sym", "Close"])
    s = o[o.sym == "SOXX"].copy()
    s["d"] = pd.to_datetime(s["date"]).dt.strftime("%Y-%m-%d")
    s = s.set_index("d")["Close"].astype(float).sort_index()
    s = s[~s.index.duplicated()]
    below = s < s.rolling(20, min_periods=20).mean()
    days = list(s.index)
    out, cur, st = {}, None, False
    for i, d in enumerate(days):
        wk = date.fromisoformat(d).isocalendar()[:2]
        if wk != cur:
            cur, st = wk, bool(below.iloc[i - 1]) if i > 0 else False
        out[wk] = st
    return out


# ---------------------------------------------------------------- 매매(가격선 이격을 봉마다 바꿀 수 있게)
def sim(d, gap, hot):
    """dip_reason rules.sim과 같은 상태 기계. gap: 봉마다 이격(np.inf = 그 봉에는 안 삼). 반환에 MAE를 더한다."""
    o, l, r, ema = d.o.values, d.l.values, d.r.values, d.ema.values
    kst = d.index.tz_convert("Asia/Seoul")
    tk = kst.as_unit("ns").asi8
    out, hold, ent, ei, dead = [], False, 0.0, 0, 0
    for i in range(61, len(d)):
        if hold:
            why = "RSI" if (i - 1 >= ei and r[i - 1] > X) else ("8시" if tk[i] >= dead else None)
            if why:
                mae = l[ei:i].min() / ent - 1
                out.append((ei, i, ent, o[i], why, mae))
                hold = False
        else:
            if hot[i] or not np.isfinite(gap[i]):
                continue
            lvl = ema[i - 1] * (1 - gap[i])
            if l[i] <= lvl:
                ent, ei, hold = min(o[i], lvl), i, True
                t8 = kst[i].normalize() + pd.Timedelta(hours=HR)
                dead = (t8 if kst[i] < t8 else t8 + pd.Timedelta(days=1)).as_unit("ns").value
    return out


def variants(zone, warn):
    """봉마다 이격 배열을 만든다(사전 등록 V0~V6)."""
    dn = np.isin(zone, (-1, -2))
    up = np.isin(zone, (1, 2))
    g = np.full(len(zone), 0.02)
    return {
        "V0 지금 그대로": g,
        "V1 하방·극하방 매수 안 함": np.where(dn, np.inf, g),
        "V2 하방·극하방 이격 3%": np.where(dn, 0.03, g),
        "V3 하방·극하방 이격 4%": np.where(dn, 0.04, g),
        "V4 SOXX 경고 주 매수 안 함": np.where(warn, np.inf, g),
        "V5 SOXX 경고 주 이격 3%": np.where(warn, 0.03, g),
        "V6 상방·극상방에서만": np.where(up, g, np.inf),
        # 대조(검증 뒤 추가, 사전 등록 아님): 이격을 넓히면 단계와 상관없이 건당 평균이 오르는지
        "C1 상방·극상방 이격 3%(대조)": np.where(up, 0.03, g),
        "C2 전부 이격 3%(대조)": np.full(len(zone), 0.03),
    }


# ---------------------------------------------------------------- 표
def clustered_diff(a: pd.DataFrame, b: pd.DataFrame, col, n=2000, seed=0):
    """두 묶음 평균 차와 날짜 묶음 부트스트랩 t(근사)."""
    da, db = a.groupby("day")[col].apply(list), b.groupby("day")[col].apply(list)
    days = sorted(set(da.index) | set(db.index))
    rng = np.random.default_rng(seed)
    diff = a[col].mean() - b[col].mean()
    bs = []
    for _ in range(n):
        pick = rng.choice(days, len(days))
        xa = [v for dd in pick if dd in da.index for v in da[dd]]
        xb = [v for dd in pick if dd in db.index for v in db[dd]]
        if xa and xb:
            bs.append(np.mean(xa) - np.mean(xb))
    sd = np.std(bs)
    return diff, diff / sd if sd > 0 else np.nan


def table(T, key, order):
    rows = []
    for k in order:
        x = T[T[key] == k]
        if len(x) == 0:
            continue
        rows.append(f"  {str(k):<6} {len(x):5d}건  평균 {x.ret.mean() * 100:+6.2f}%  중앙 {x.ret.median() * 100:+6.2f}%  "
                    f"이김 {np.mean(x.ret > 0) * 100:4.0f}%  RSI매도 {np.mean(x.why == 'RSI') * 100:4.0f}%  "
                    f"MAE 평균 {x.mae.mean() * 100:+6.2f}%  -3%↓ {np.mean(x.mae <= -0.03) * 100:4.0f}%  "
                    f"-5%↓ {np.mean(x.mae <= -0.05) * 100:4.0f}%  최악 {x.ret.min() * 100:+6.1f}%")
    return rows


def main():
    S = syms()
    ddates, dclose, Z, above = daily_context(S)
    wk = soxx_week_state()
    lines, all_rows, var_rows = [], [], []
    for wname, (mins, lo, hi) in WIN.items():
        ema_len, rsi_len = (60, 14) if mins == 3 else (36, 9)
        for j, s in enumerate(S):
            try:
                d = prep(bars(RAW, s, mins), ema_len, rsi_len)
            except FileNotFoundError:
                continue
            t = d.index.tz_convert(NY)
            tn = t.as_unit("ns").asi8
            k = np.searchsorted(dclose, tn, side="right") - 1           # 그 봉 시작 시각에 끝나 있는 마지막 일봉
            zone = np.where(k >= 0, Z[np.maximum(k, 0), j], -9)
            abv = np.where(k >= 0, above[np.maximum(k, 0), j], False)
            warn = np.array([wk.get(x.isocalendar()[:2], False) for x in t.date])
            hot = hot_days(RAW, s, d, 0.07)
            for vname, gap in variants(zone, warn).items():
                for ei, xi, ent, px, why, mae in sim(d, gap, hot):
                    if not (lo <= t[ei].date() <= hi):
                        continue
                    row = {"win": wname, "var": vname, "sym": s, "t": t[ei], "day": t[ei].date(), "ses": d.ses.iloc[ei],
                           "ret": px / ent - 1, "why": why, "mae": mae, "zone": int(zone[ei]), "warn": bool(warn[ei]),
                           "above20": bool(abv[ei])}
                    (all_rows if vname.startswith("V0") else var_rows).append(row)
    T = pd.DataFrame(all_rows)
    V = pd.DataFrame(var_rows + all_rows)
    T.to_csv(os.path.join(Zm.OUT, "dip_trades.csv"), index=False, encoding="utf-8-sig")
    zord = [2, 1, 0, -1, -2]
    for scope, f in (("24시간", lambda x: x), ("정규장만", lambda x: x[x.ses == "정규"])):
        for wname in WIN:
            x = f(T[T.win == wname]).copy()
            x["단계"] = x.zone.map(NAMES).fillna("계산 전")
            lines += ["", f"== {wname} · {scope} · 매수 순간의 추세 5단계 (전체 {len(x)}건, 평균 {x.ret.mean() * 100:+.2f}%)"]
            lines += table(x, "단계", [NAMES[z] for z in zord] + ["계산 전"])
            up, dn = x[x.zone.isin([1, 2])], x[x.zone.isin([-1, -2])]
            if len(up) and len(dn):
                d1, t1 = clustered_diff(up, dn, "ret")
                d2, t2 = clustered_diff(up.assign(b=(up.mae <= -0.03) * 1.0), dn.assign(b=(dn.mae <= -0.03) * 1.0), "b")
                lines.append(f"  상방계 − 하방계: 평균 수익 {d1 * 100:+.2f}%p (t {t1:+.1f}), -3% 더 뚫림 비율 {d2 * 100:+.1f}%p (t {t2:+.1f})")
            x["SOXX"] = np.where(x.warn, "경고 주", "평상 주")
            lines += table(x, "SOXX", ["평상 주", "경고 주"])
            x["20일선"] = np.where(x.above20, "위", "아래")
            lines += table(x, "20일선", ["위", "아래"])
    lines += ["", "== 사전 등록 변형 (24시간, 같은 금액 매매) — 수익 합 / 건수 / 평균 / 최악 / -5% 더 뚫림 건수"]
    for wname in WIN:
        lines.append(f"-- {wname}")
        for vname in variants(np.zeros(1), np.zeros(1, bool)):
            x = V[(V.win == wname) & (V["var"] == vname)]
            lines.append(f"  {vname:<22} 합 {x.ret.sum() * 100:+8.1f}%  {len(x):5d}건  평균 {x.ret.mean() * 100:+6.2f}%  "
                         f"최악 {x.ret.min() * 100:+6.1f}%  -5%↓ {int((x.mae <= -0.05).sum()):4d}")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(Zm.OUT, "dip_by_zone.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt)


if __name__ == "__main__":
    main()
