"""공용 실험 틀(2026-10-10): 두 펀드 미국 보유에서 7자리 계좌를 매일 운용하는 규칙들을 같은 방식으로 잰다.

시점 규칙(모든 변형이 지킨다)
  k일(미국 거래일) 시가에 내리는 결정은 k-1일 종가까지의 일봉, k-1일까지 끝난 160분봉, k일보다 앞선 날짜의 PDF만 쓴다.
  체결은 k일 시가, 수익은 시가→다음 시가. 수수료 0, 호가 차이는 편도 0.1%(COST)로 따로 본다.
쓰는 법
  H = Harness()                                   # 자료·지표를 한 번 만든다(약 1분)
  pick = H.rank_picker(lambda k: H.ret(20).iloc[k - 1], buffer=12)
  res = H.evaluate("이름", H.simulate(pick))       # 월별·구간별 수익, 낙폭, 회전, 기여 쏠림
지표 표(행 = 미국 거래일 H.days, 열 = 종목 기호). 값은 그날 종가(또는 그날 마지막 160분봉)까지 안다 → k일 결정엔 iloc[k - 1].
  H.cls H.opn H.high          일봉 종가·시가·고가(분할 반영)
  H.k                         일봉 Stoch RSI K(사용자 Pine: RSI 14·스토캐스틱 14·평활 3)
  H.rsi_d                     일봉 RSI 14
  H.r160                      그날 마지막 160분봉(16:00 ET로 끝나는 봉)의 RSI 14 — 1시간봉 근사
  H.r160_max                  그날 160분봉 RSI 중 최대
  H.sig70                     그날 160분봉 RSI가 70을 아래에서 위로 넘었나(True/False)
  H.ret(n) H.ma(n) H.rs(n)    n일 수익, n일 종가 평균, QQQ 대비 n일 상대수익((1+r)/(1+r_qqq)-1)
  H.hi(n)                     최근 n거래일(그날 포함) 고가 최고. 2023-01부터의 일봉으로 계산
  H.eligible(day)             그날 시가에 살 수 있는 종목(그날보다 앞선 PDF의 미국 보유 ∪, 가격 있음)
"""
import os, sys
from datetime import date
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rsi_breakout import (Panel, rsi, bars160, frames, desk_curve, START, END, SLOTS, COST, CACHE, RES)

MONTHS = [f"2026-{m:02d}" for m in range(1, 11)]
PERIODS = (("앞 1/2~5/29", "2026-01-02", "2026-05-29"), ("뒤 6/1~10/9", "2026-06-01", "2026-10-09"),
           ("9/1~10/9", "2026-09-01", "2026-10-09"), ("10/1~10/9", "2026-10-01", "2026-10-09"), ("전체", START, END))


def fetch_long():
    """52주·전고점 계산용 긴 일봉(2023-01부터). cache/1d_long.parquet."""
    import json, yfinance as yf
    tick = json.load(open(os.path.join(CACHE, "universe.json"), encoding="utf-8"))["tickers"]
    syms = sorted(set(tick.values()) | {"QQQ"})
    rows = []
    for i in range(0, len(syms), 40):
        d = yf.download(syms[i:i + 40], start="2023-01-01", end="2026-10-10", interval="1d", auto_adjust=False,
                        progress=False, group_by="ticker", threads=True)
        for s in syms[i:i + 40]:
            try:
                x = d[s] if isinstance(d.columns, pd.MultiIndex) else d
                x = x[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Open", "Close"]).copy()
            except KeyError:
                continue
            x["sym"] = s
            rows.append(x)
    df = pd.concat(rows)
    df.index.name = "ts"
    df.reset_index().to_parquet(os.path.join(CACHE, "1d_long.parquet"))
    print("1d_long", len(df), "행", df.sym.nunique(), "종목")


class Harness:
    def __init__(self):
        P = Panel()                                   # 사용자 규칙 값(RSI 70, K 95)으로 만든 판
        self.P = P
        self.days, self.syms = P.days, P.syms
        self.cls, self.opn, self.k = P.cls[P.syms], P.opn[P.syms], P.k
        self.qqq_c, self.qqq_o = P.cls["QQQ"], P.opn["QQQ"]
        idx = pd.Index(self.days)
        hi = {s: P.f["1d"][s].High.set_axis(P.f["1d"][s].index.strftime("%Y-%m-%d")).reindex(idx) for s in self.syms}
        self.high = pd.DataFrame(hi)
        self.rsi_d = pd.DataFrame({s: rsi(self.cls[s].ffill().to_numpy()) for s in self.syms}, index=idx)
        self.rsi_d[self.cls.isna()] = np.nan
        last, mx, s70 = {}, {}, {}
        for s in self.syms:
            if s not in P.f["1h"]:
                continue
            b = bars160(P.f["1h"][s], exact=False)
            r = pd.Series(rsi(b.c.to_numpy()), index=b.index.strftime("%Y-%m-%d"))
            last[s] = r.groupby(level=0).last().reindex(idx)
            mx[s] = r.groupby(level=0).max().reindex(idx)
            s70[s] = pd.Series([d in P.sig.get(s, {}) for d in self.days], index=idx)
        self.r160, self.r160_max, self.sig70 = pd.DataFrame(last), pd.DataFrame(mx), pd.DataFrame(s70)
        lp = os.path.join(CACHE, "1d_long.parquet")
        if not os.path.exists(lp):
            fetch_long()
        lg = pd.read_parquet(lp)
        self._long_high = {s: g.set_index("ts").High.set_axis(g.ts.dt.strftime("%Y-%m-%d").to_numpy())
                           for s, g in lg.groupby("sym")}
        self._cache = {}

    # ------------------------------------------------------------ 지표
    def ret(self, n):
        key = ("ret", n)
        if key not in self._cache:
            self._cache[key] = self.cls / self.cls.shift(n) - 1
        return self._cache[key]

    def ma(self, n):
        key = ("ma", n)
        if key not in self._cache:
            self._cache[key] = self.cls.rolling(n, min_periods=n).mean()
        return self._cache[key]

    def rs(self, n):
        key = ("rs", n)
        if key not in self._cache:
            q = self.qqq_c / self.qqq_c.shift(n) - 1
            self._cache[key] = (1 + self.ret(n)).div(1 + q, axis=0) - 1
        return self._cache[key]

    def hi(self, n):
        """최근 n거래일(그날 포함) 고가 최고. 2023-01부터의 긴 일봉으로 계산해 앞쪽이 잘리지 않는다."""
        key = ("hi", n)
        if key not in self._cache:
            out = {}
            for s in self.syms:
                h = self._long_high.get(s)
                out[s] = (h.rolling(n, min_periods=1).max().reindex(self.days) if h is not None
                          else pd.Series(np.nan, index=self.days))
            self._cache[key] = pd.DataFrame(out, index=pd.Index(self.days))
        return self._cache[key]

    def eligible(self, day):
        return self.P.eligible(day)

    # ------------------------------------------------------------ 고르는 틀
    def rank_picker(self, score, slots=SLOTS, buffer=None, weekly=False, min_score=None):
        """score(k) → 종목별 점수 Series(k-1일까지 정보, 클수록 강함). 매일(또는 주 첫 거래일) 상위 slots.
        buffer=12면 들고 있는 종목은 12위 안이면 유지하고 빈자리를 위에서 채운다."""
        def pick(k, held):
            day = self.days[k]
            if weekly and held and date.fromisoformat(day).isocalendar()[:2] == \
                    date.fromisoformat(self.days[k - 1]).isocalendar()[:2]:
                return [s for s in held if np.isfinite(self.opn[s].iloc[k])]
            elig = [s for s in self.eligible(day) if s in self.opn.columns and np.isfinite(self.opn[s].iloc[k])]
            sc = score(k).reindex(elig).dropna()
            if min_score is not None:
                sc = sc[sc >= min_score]
            ranked = list(sc.sort_values(ascending=False).index)
            keep = [s for s in held if s in ranked[:buffer]] if buffer else []
            return keep + [s for s in ranked if s not in keep][:slots - len(keep)]
        return pick

    def entry_exit_picker(self, entry, exit_rule, priority=None, slots=SLOTS):
        """entry(s, k) → k일 시가에 새로 사나(k-1일까지 정보). exit_rule(s, k, pos) → k일 시가에 파나.
        pos = {'k0': 산 날 위치, 'px': 산 시가, 'peak': k-1일까지 종가 최고(산 시가 포함)}.
        priority(k) → 종목별 점수 Series, 빈자리보다 신호가 많으면 큰 순. 매번 새로 만들어 쓴다(상태를 가진다)."""
        pos = {}

        def pick(k, held):
            for s in [s for s in pos if s not in held]:
                pos.pop(s)
            keep = []
            for s in held:
                p = pos[s]
                c = self.cls[s].iloc[k - 1]
                if k - 1 >= p["k0"] and np.isfinite(c):
                    p["peak"] = max(p["peak"], c)
                if not exit_rule(s, k, p):
                    keep.append(s)
            for s in [s for s in pos if s not in keep]:
                pos.pop(s)
            elig = self.eligible(self.days[k])
            new = [s for s in self.syms if s not in keep and s in elig and np.isfinite(self.opn[s].iloc[k])
                   and entry(s, k)]
            if priority is not None and new:
                pr = priority(k)
                new.sort(key=lambda s: -(pr.get(s) if np.isfinite(pr.get(s, np.nan)) else -1e9))
            add = new[:max(0, slots - len(keep))]
            for s in add:
                px = self.opn[s].iloc[k]
                pos[s] = {"k0": k, "px": px, "peak": px}
            return keep + add
        return pick

    # ------------------------------------------------------------ 계좌와 평가
    def simulate(self, pick, slots=SLOTS):
        """한 종목 1/slots(종목이 더 많으면 1/종목 수), 빈자리는 현금, 매일 다시 맞춘다. 시가→다음 시가."""
        k0 = next(i for i, d in enumerate(self.days) if d >= START)
        held, w_prev, val, rows, contrib = [], {}, 1.0, [], {}
        o = self.opn
        for k in range(k0, len(self.days) - 1):
            held = pick(k, held)
            w = {s: 1 / max(slots, len(held)) for s in held}
            rets = {}
            for s in w:
                a, b = o[s].iloc[k], o[s].iloc[k + 1]
                rets[s] = b / a - 1 if np.isfinite(a) and np.isfinite(b) and a > 0 else 0.0
            r = sum(w[s] * rets[s] for s in w)
            for s in w:
                contrib[s] = contrib.get(s, 0.0) + val * w[s] * rets[s]
            turn = sum(abs(w.get(s, 0) - w_prev.get(s, 0)) for s in set(w) | set(w_prev))
            rows.append(dict(day=self.days[k], ret=r, turn=turn, n=len(held), held=",".join(sorted(held))))
            val *= 1 + r
            w_prev = {s: w[s] * (1 + rets[s]) / (1 + r) for s in w}
        df = pd.DataFrame(rows)
        df.attrs["contrib"] = contrib
        return df

    def baselines(self):
        """사용자 규칙·데스크 7종목·QQQ·보유 전체 같은 비중 — 같은 틀로."""
        P = self.P
        out = {"사용자 규칙(RSI70→K95)": self.simulate(self.entry_exit_picker(
            lambda s, k: bool(self.sig70[s].iloc[k - 1]) if s in self.sig70 else False,
            lambda s, k, p: P.exit_hit(s, k - 1), priority=lambda k: self.ret(5).iloc[k - 1]))}
        q = self.qqq_o
        qq = pd.DataFrame(dict(day=self.days[:-1], ret=(q.shift(-1) / q - 1).to_numpy()[:-1], turn=0.0, n=1, held="QQQ"))
        out["QQQ"] = qq[qq.day >= START].reset_index(drop=True)
        out["보유 전체 같은 비중"] = self.simulate(self.rank_picker(lambda k: self.cls.iloc[k - 1] * 0 + 1, slots=999))
        try:
            d = desk_curve(P)
            d["held"] = ""
            out["데스크 7종목(#774)"] = d
        except Exception as exc:
            print("데스크 비교 실패", exc)
        return out

    def evaluate(self, name, df, cost=COST):
        def tot(x, c=0.0):
            r = x.ret - c * x.turn
            return float((1 + r).prod() - 1) if len(x) else float("nan")

        def mdd(x):
            v = (1 + x.ret).cumprod()
            return float((v / np.maximum.accumulate(v) - 1).min()) if len(x) else float("nan")
        res = {"name": name}
        for lab, a, b in PERIODS:
            x = df[(df.day >= a) & (df.day <= b)]
            res[lab] = {"수익": tot(x), "호가0.1%": tot(x, cost), "낙폭": mdd(x)}
        res["월별"] = {m: tot(df[df.day.str.startswith(m)]) for m in MONTHS}
        res["하루회전"] = float(df.turn.mean())
        res["평균보유"] = float(df.n.mean())
        c = df.attrs.get("contrib") or {}
        gain = sum(v for v in c.values())
        top = sorted(c.items(), key=lambda kv: -kv[1])[:5]
        res["기여상위5"] = [(s, round(v, 4)) for s, v in top]
        res["상위3기여몫"] = (sum(v for _, v in top[:3]) / gain) if gain > 0 else None
        return res


def table(results, periods=("앞 1/2~5/29", "뒤 6/1~10/9", "9/1~10/9", "10/1~10/9", "전체")):
    lines = ["  " + "이름".ljust(34) + " | ".join(p for p in periods) + " | 전체 낙폭 | 회전 | 상위3 기여"]
    for r in results:
        cells = [f"{r[p]['수익']:+.1%}({r[p]['호가0.1%']:+.1%})" for p in periods]
        share = f"{r['상위3기여몫']:.0%}" if r.get("상위3기여몫") is not None else "—"
        lines.append(f"  {r['name'][:34].ljust(34)}" + " | ".join(cells) +
                     f" | {r['전체']['낙폭']:.1%} | {r['하루회전']:.0%} | {share}")
    return "\n".join(lines)


def months_table(results):
    lines = ["  " + "이름".ljust(34) + " ".join(m[5:].rjust(7) for m in MONTHS)]
    for r in results:
        lines.append(f"  {r['name'][:34].ljust(34)}" + " ".join(f"{r['월별'][m]:+7.1%}" for m in MONTHS))
    return "\n".join(lines)


if __name__ == "__main__" and sys.argv[1:] == ["baselines"]:
    import json
    H = Harness()
    res = [H.evaluate(n, df) for n, df in H.baselines().items()]
    os.makedirs(os.path.join(RES, "variants"), exist_ok=True)
    json.dump(res, open(os.path.join(RES, "variants", "baselines.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(table(res))
    print(months_table(res))
