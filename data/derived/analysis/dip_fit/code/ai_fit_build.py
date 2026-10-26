"""이격매수매도 AI 종목 궁합 4부(2026-10-23 사용자: AI 종목 중 이격매수매도에 더 잘 맞는 종목은? 예: IREN보다 NBIS) — 거래표.

규칙은 지금 지표(dip_scalper.pine)와 같다(build.sim과 같은 상태 기계):
  3분봉 EMA60[1] x (1-2%)에 저가가 닿으면 min(시가, 선)에 산다. RSI14 > 70인 봉 다음 봉 시가 또는
  매수 뒤 처음 오는 한국 아침 8시 이후 첫 봉 시가에 판다. 전날 정규장 +7% 다음 날 정규장은 사지 않는다.
  AI 종목에는 3배 레버리지 목록 종목이 없어 이격은 모두 2%다.
기간(매수일 ET):
  P0 8/3~9/9  5분봉 (5분봉은 7/31부터, 첫 거래일은 EMA 길들이기) 근사(EMA36·RSI9 = 같은 180분 EMA·약 45분 RSI, stop_lab 'early'와 같은 근사). 1분봉 없음 → 꼬리 표시 없음.
  P1 9/10~10/8 3분봉. 지난 연구(trades_1009.csv)를 그대로 쓴다(9/9~9/23 1분봉은 다시 받을 수 없다).
  P2 10/9~10/22 3분봉. 새로 받은 1분봉(raw_1023)으로 계산. 10/9 이후는 지난 연구가 '종목 궁합'에 쓰지 않은 새 구간이다.
  겹침 확인: 새 1분봉으로 9/28~10/8도 다시 계산(1분봉은 9/25부터, 첫 거래일은 EMA 길들이기)해 trades_1009.csv와 같은지 본다(재현 점검).
꼬리 체결(프리·애프터): 체결 분(매수 봉 안에서 저가가 매수가 이하가 된 첫 1분)이 직전 1분 종가보다 1% 넘게 아래(fr_hold.tail_buy와 같은 정의).
미완료 거래(데이터 끝까지 매도가 안 나온 것)는 뺀다.
유니버스: watchlist.AI_UNIVERSE 미국 상장 115종목(TEM 포함). 비교용 QQQ·SMH·BTC-USD·IBIT도 같은 폴더에 있다.
저장: ../results/ai_fit_trades.csv (P0·P1·P2 합본, 열 period·src), ../results/ai_fit_overlap.txt
사용: py ai_fit_build.py <raw 폴더>"""
import os, re, sys
from datetime import date
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "../../dip_reason/code"))
sys.path.insert(0, os.path.join(HERE, "../../../../.."))
from events import bars, prep
from rules import hot_days
import watchlist as w

OUT = os.path.join(HERE, "../results")
NS_MIN = 60 * 10**9
P0, P1, P2 = (date(2026, 8, 3), date(2026, 9, 9)), (date(2026, 9, 10), date(2026, 10, 8)), (date(2026, 10, 9), date(2026, 10, 22))
AI = {}
for big, subs in w.AI_UNIVERSE.items():
    for sub, ts in subs.items():
        for t in ts:
            if re.fullmatch(r"[A-Z]{1,5}", t):
                AI.setdefault(t, (big, sub))


def dead_times(d):
    kst = d.index.tz_convert("Asia/Seoul")
    t8 = kst.normalize() + pd.Timedelta(hours=8)
    return np.where(kst < t8, t8.as_unit("ns").asi8, (t8 + pd.Timedelta(days=1)).as_unit("ns").asi8)


def sim(d, dead_of, hot, gap=0.02, x=70.0):
    """build.sim과 같다. 반환: (매수 봉, 매도 봉, 매수가, 매도가, 이유). 끝까지 안 팔린 거래는 돌려주지 않는다."""
    o, l, r, ema, tk = d.o.values, d.l.values, d.r.values, d.ema.values, d.index.as_unit("ns").asi8
    out, hold, ent, ei, dead = [], False, 0.0, 0, 0
    for i in range(61, len(d)):
        if hold:
            why = "RSI" if (i - 1 >= ei and r[i - 1] > x) else ("8시" if tk[i] >= dead else None)
            if why:
                out.append((ei, i, ent, o[i], why))
                hold = False
        elif not hot[i]:
            lvl = ema[i - 1] * (1 - gap)
            if l[i] <= lvl:
                ent, ei, hold, dead = min(o[i], lvl), i, True, dead_of[i]
    return out


def tail_flag(m1, t_ns, ent, mins):
    """프리·애프터 매수 꼬리: 체결 분의 직전 1분 종가보다 1% 넘게 아래면 1."""
    t1 = m1.index.as_unit("ns").asi8
    l1, c1 = m1.Low.values, m1.Close.values
    a, b = np.searchsorted(t1, t_ns, "left"), np.searchsorted(t1, t_ns + mins * NS_MIN, "left")
    for k in range(a, b):
        if l1[k] <= ent * (1 + 1e-9):
            return np.nan if k == 0 else float(ent < c1[k - 1] * 0.99)
    return np.nan


def run(raw, s, mins, lo, hi, period):
    el, rl = (60, 14) if mins == 3 else (36, 9)
    d = prep(bars(raw, s, mins), el, rl)
    hot = hot_days(raw, s, d, 0.07)
    m1 = pd.read_parquet(f"{raw}/1m_{s}.parquet") if mins == 3 else None
    c1d = pd.read_parquet(f"{raw}/1d_{s}.parquet").Close
    c1d.index = pd.to_datetime(c1d.index).date
    prevc = c1d.shift(1)
    rows = []
    for ei, xi, ent, px, why in sim(d, dead_times(d), hot):
        dy = d.date.iloc[ei]
        if not (lo <= dy <= hi):
            continue
        ses = d.ses.iloc[ei]
        tail = tail_flag(m1, d.index[ei].as_unit("ns").value, ent, mins) if (m1 is not None and ses != "정규") else np.nan
        rows.append(dict(sym=s, day=dy, t=d.index[ei], ses=ses, hour=d.index[ei].hour, ret=px / ent - 1, why=why,
                         bars=xi - ei, hold_min=(xi - ei) * mins, mae=d.l.values[ei:xi + 1].min() / ent - 1, ent=ent, px=px,
                         t_exit=d.index[xi], day_ret=d.c.values[ei - 1] / prevc.get(dy, np.nan) - 1,
                         tail=tail, period=period, src=f"{mins}m"))
    return rows


def main(raw):
    os.makedirs(OUT, exist_ok=True)
    new, ovl = [], []
    for s in sorted(AI):
        if not os.path.exists(f"{raw}/1m_{s}.parquet"):
            print("1분봉 없음", s)
            continue
        new += run(raw, s, 5, *P0, "P0")
        new += run(raw, s, 3, *P2, "P2")
        ovl += run(raw, s, 3, date(2026, 9, 28), P1[1], "ovl")
    old = pd.read_csv(os.path.join(OUT, "trades_1009.csv"), parse_dates=["t"])
    old["day"] = pd.to_datetime(old.day).dt.date
    old = old[old.sym.isin(AI) & (old.day >= P1[0]) & (old.day <= P1[1])].copy()
    old["period"], old["src"], old["hold_min"] = "P1", "3m_1009", old.bars * 3
    # 지난 표에는 꼬리 표시가 없다 → 새 1분봉이 있는 9/28~10/8만 겹침 재계산 값에서 옮긴다(아래 재현 점검에서 같은 거래일 때만)
    T = pd.concat([pd.DataFrame(new), old], ignore_index=True)
    T["big"] = T.sym.map(lambda s: AI[s][0])
    T["sub"] = T.sym.map(lambda s: AI[s][1])
    O = pd.DataFrame(ovl)
    # 재현 점검: 같은 종목·같은 매수 시각 거래가 양쪽에 있는가, 수익이 같은가
    o2 = old[old.day >= date(2026, 9, 28)]
    k_old = set(zip(o2.sym, o2.t.dt.tz_convert("America/New_York").astype(str)))
    k_new = set(zip(O.sym, O.t.astype(str)))
    both = k_old & k_new
    mo = o2.assign(k=list(zip(o2.sym, o2.t.dt.tz_convert("America/New_York").astype(str)))).set_index("k").ret
    mn = O.assign(k=list(zip(O.sym, O.t.astype(str)))).set_index("k").ret
    dif = np.array([abs(mo[k] - mn[k]) for k in both]) if both else np.array([np.nan])
    tl = O.assign(k=list(zip(O.sym, O.t.astype(str)))).set_index("k")["tail"]
    T.loc[T.period == "P1", "tail"] = [tl.get((s, str(t.tz_convert("America/New_York"))), np.nan) if d >= date(2026, 9, 28) else np.nan
                                       for s, t, d in zip(T.loc[T.period == "P1", "sym"], T.loc[T.period == "P1", "t"], T.loc[T.period == "P1", "day"])]
    msg = [f"재현 점검 9/28~10/8 (AI {len(AI)}종목): 지난 표 {len(k_old)}건, 새 1분봉 {len(k_new)}건, 같은 매수 시각 {len(both)}건",
           f"  같은 거래의 수익 차이: 중앙 {np.nanmedian(dif):.5f}, 0.1%p 넘는 건 {(dif > 0.001).sum()}",
           f"  지난 표에만: {len(k_old - k_new)}건, 새 표에만: {len(k_new - k_old)}건",
           f"  지난 표 합 {o2.ret.sum():+.3f} / 새 표 합 {O.ret.sum():+.3f}"]
    for s in ["NBIS", "IREN"]:
        msg.append(f"  {s}: 지난 {o2[o2.sym == s].ret.sum():+.3f}({(o2.sym == s).sum()}건) / 새 {O[O.sym == s].ret.sum():+.3f}({(O.sym == s).sum()}건)")
    open(os.path.join(OUT, "ai_fit_overlap.txt"), "w", encoding="utf-8").write("\n".join(msg) + "\n")
    T.to_csv(os.path.join(OUT, "ai_fit_trades.csv"), index=False, encoding="utf-8-sig")
    print("\n".join(msg))
    print(T.groupby("period").agg(n=("ret", "size"), syms=("sym", "nunique"), days=("day", "nunique"), mean=("ret", "mean"),
                                  d0=("day", "min"), d1=("day", "max")))


if __name__ == "__main__":
    main(sys.argv[1])
