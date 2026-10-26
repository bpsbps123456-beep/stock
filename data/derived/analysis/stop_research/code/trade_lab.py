"""4단계 거래 단위 엔진(2026-10-11): 일반 주식을 어떤 신호로 사서, 어떤 손절·청산으로 끝내면 거래 1건이 어떻게 되나.

사용자: "7종목만 하는 게 아니라 일반 주식도 한다(레버리지 주식도)." 7종목처럼 팔 규칙이 따로 있는 전략이 아니라,
사서 들고 있다가 손절로 나오는 매매에 손절이 쓸모 있는지를 거래 1건 단위(기대값·꼬리)로 잰다.

시점
  - 진입 신호는 s일 종가까지로 판단 → e = s+1일 시가에 산다(진입가 = o[e]).
  - 손절 다리(leg)는 t일 판단에 t-1일까지 정보를 쓴다. 장중 손절은 시가 ≤ 선이면 시가, 저가 ≤ 선이면 선 가격.
    종가형 청산(SMA·시간)은 t일 종가로 판단 → t+1일 시가.
  - 보유 기간 H거래일: e..e+H-1. 끝까지 안 걸리면 e+H-1일 종가에 판 것으로 친다. 나온 뒤 현금은 0%(다시 사지 않음).
다리(Leg): 거래를 비중 frac로 나눠 다리마다 따로 청산(일부 손절). 거래 수익 = Σ frac × (청산가/진입가 − 1).
  kind = hold | fixed(p: 진입가 대비) | trail(p: 산 뒤 최고 종가 대비) | chand(p: 산 뒤 최고 고가 − p×ATR22)
         | sma(p: 종가 < SMA_p) | time(p: p일째 종가가 진입가 이하면) | target(p: 진입가 × (1+p) 지정가 익절)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import stock_lab as S

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"
RESULTS = HERE.parent / "results"


@dataclass(frozen=True)
class Leg:
    frac: float
    kind: str = "hold"
    p: float = 0.0


_ST = None


def _bars_from(df: pd.DataFrame) -> dict:
    out = {}
    for sym, g in df.sort_values(["sym", "date"]).groupby("sym"):
        g = g.dropna(subset=["Open", "High", "Low", "Close"])
        g = g[(g.Close > 0) & (g.Open > 0) & (g.Low > 0) & (g.High > 0)]
        if len(g) < 260:
            continue
        f = (g["Adj Close"] / g["Close"]).fillna(1.0).to_numpy()
        out[sym] = S.Bars(sym, g.date.to_numpy(), g.Open.to_numpy() * f, g.High.to_numpy() * f, g.Low.to_numpy() * f,
                          g.Close.to_numpy() * f, g.Volume.to_numpy(float))
    return out


def load_stocks() -> dict:
    global _ST
    if _ST is None:
        _ST = _bars_from(pd.read_parquet(CACHE / "p4_stocks.parquet"))
    return _ST


def meta() -> dict:
    return json.loads((CACHE / "p4_meta.json").read_text(encoding="utf-8"))


def bad_bars(b: S.Bars, jump: float = 1.5) -> bool:
    """자료 오류 의심(하루 +150% 이상 또는 −70% 이하 종가 변화가 있는 종목). 분석에서 빼거나 따로 보인다."""
    r = b.c[1:] / b.c[:-1] - 1
    return bool(np.any(r > jump) or np.any(r < -0.7))


# ----------------------------------------------------------------------------- 진입 신호
def entries(b: S.Bars, kind: str, d0: str, d1: str, gap: int = 20, every: int = 10) -> np.ndarray:
    """신호일 s의 종가로 판단해 다음 날 e = s+1에 사는 진입일 배열(같은 종목 같은 신호는 gap거래일 이상 띄움).
    kind: random(every거래일마다 한 번, 위상은 종목 이름으로 고정) | dip7(그날 −7% 이하) | breakout(252일 신고가 첫날)
          | pullback(종가 > SMA200이고 20일 최고 종가 대비 −10% 첫날) | deep40(252일 최고 종가 대비 −40% 첫날)"""
    c = b.c
    n = len(c)
    s_lo, s_hi = max(b.idx(d0) - 1, 252), min(b.idx(d1, "right") - 1, n - 2)
    if s_hi <= s_lo:
        return np.array([], int)
    s = np.arange(s_lo, s_hi + 1)
    if kind == "random":
        phase = sum(map(ord, b.sym)) % every
        sig = (s % every) == phase
    elif kind == "dip7":
        sig = c[s] / c[s - 1] - 1 <= -0.07
    elif kind == "breakout":
        hi = b.close_high(252)
        sig = (c[s] >= hi[s] - 1e-12) & (c[s - 1] < hi[s - 1] - 1e-12)
    elif kind == "pullback":
        sma, h20 = b.sma(200), b.close_high(20)
        cond = (c > sma) & (c <= 0.90 * h20)
        sig = cond[s] & ~cond[s - 1]
    elif kind == "deep40":
        dd = b.dd252()
        cond = dd <= -0.40
        sig = cond[s] & ~cond[s - 1]
    else:
        raise ValueError(kind)
    picked, last = [], -10 ** 9
    for si in s[sig]:
        if si - last >= gap:
            picked.append(si + 1)
            last = si
    return np.array(picked, int)


# ----------------------------------------------------------------------------- 거래
def trade(b: S.Bars, e: int, H: int, legs, slip: float = 0.0):
    """e일 시가에 사서 최대 H거래일. 반환 dict(ret, legs별 청산일·사유, mae, mfe) 또는 None(자료 부족)."""
    z = e + H - 1
    if z >= len(b.c) or e < 1:
        return None
    o, h, l, c = b.o[e:z + 1], b.h[e:z + 1], b.l[e:z + 1], b.c[e:z + 1]
    entry = o[0] * (1 + slip)
    peak_c = np.maximum.accumulate(np.concatenate([[entry], c]))[:-1]      # t-1까지 산 뒤 최고 종가(첫날은 진입가)
    peak_h = np.maximum.accumulate(np.concatenate([[entry], h]))[:-1]
    out_ret, rows = 0.0, []
    for leg in legs:
        px, day, why = c[-1], H - 1, "기간끝"
        if leg.kind in ("fixed", "trail", "chand"):
            if leg.kind == "fixed":
                L = np.full(len(o), entry * (1 - leg.p))
            elif leg.kind == "trail":
                L = peak_c * (1 - leg.p)
            else:
                atr = b.atr(22)[e - 1:z]                                    # t-1일 ATR
                L = peak_h - leg.p * atr
            hit = np.flatnonzero((o <= L) | (l <= L))
            if len(hit):
                t = hit[0]
                px, day, why = (o[t] if o[t] <= L[t] else L[t]) * (1 - slip), t, leg.kind
        elif leg.kind == "sma":
            sma = b.sma(int(leg.p))[e:z + 1]
            hit = np.flatnonzero(c[:-1] < sma[:-1])
            if len(hit):
                t = hit[0] + 1
                px, day, why = o[t] * (1 - slip), t, "sma"
        elif leg.kind == "time":
            t0 = int(leg.p)
            if t0 < H - 1 and c[t0] <= entry:
                px, day, why = o[t0 + 1] * (1 - slip), t0 + 1, "time"
        elif leg.kind == "target":
            T = entry * (1 + leg.p)
            hit = np.flatnonzero(h >= T)
            if len(hit):
                t = hit[0]
                px, day, why = max(o[t], T) if o[t] >= T else T, t, "target"
        elif leg.kind != "hold":
            raise ValueError(leg.kind)
        r = px / entry - 1
        out_ret += leg.frac * r
        rows.append((why, int(day), float(r)))
    return {"ret": float(out_ret), "legs": rows, "mae": float(l.min() / entry - 1), "mfe": float(h.max() / entry - 1),
            "bh": float(c[-1] / entry - 1), "days_in": float(sum(leg.frac * (d + 1) for leg, (_, d, _) in zip(legs, rows)))}


def run(entry_kind: str, legs, d0: str, d1: str, H: int = 60, syms=None, slip: float = 0.0, gap: int = 20,
        every: int = 10, data=None) -> pd.DataFrame:
    data = data or load_stocks()
    rows = []
    for s in (syms or sorted(data)):
        b = data[s]
        for e in entries(b, entry_kind, d0, d1, gap, every):
            r = trade(b, e, H, legs, slip)
            if r is None:
                continue
            rows.append({"sym": s, "entry": b.days[e], "exit_end": b.days[e + H - 1], "ret": r["ret"], "bh": r["bh"],
                         "mae": r["mae"], "mfe": r["mfe"], "days_in": r["days_in"],
                         "why": "|".join(w for w, _, _ in r["legs"])})
    return pd.DataFrame(rows)


def summary(df: pd.DataFrame, col: str = "ret") -> dict:
    """거래 1건 기대값·승률·손익비·꼬리. 같은 달 진입 거래는 서로 엮여 있어 표준오차는 진입 월 묶음으로 낸다."""
    if df is None or not len(df):
        return {"n": 0}
    x = df[col].to_numpy()
    win, loss = x[x > 0], x[x <= 0]
    month = df["entry"].str[:7]
    g = df.assign(m=month).groupby("m")[col].mean()
    se = float(g.std(ddof=1) / np.sqrt(len(g))) if len(g) > 1 else float("nan")
    return {"n": int(len(x)), "기대값": float(x.mean()), "중앙": float(np.median(x)), "승률": float(len(win) / len(x)),
            "평균이익": float(win.mean()) if len(win) else float("nan"),
            "평균손실": float(loss.mean()) if len(loss) else float("nan"),
            "손익비": float(win.mean() / -loss.mean()) if len(win) and len(loss) and loss.mean() < 0 else float("nan"),
            "P(-30%이하)": float((x <= -0.30).mean()), "P(-50%이하)": float((x <= -0.50).mean()),
            "하위5%": float(np.quantile(x, 0.05)), "상위5%": float(np.quantile(x, 0.95)),
            "월묶음_SE": se, "보유일당": float(x.sum() / df["days_in"].sum()) if "days_in" in df else float("nan")}


def compare(df: pd.DataFrame) -> dict:
    """같은 거래의 규칙 수익(ret) vs 들고만 있기(bh) 짝 비교: 차이의 평균과 월 묶음 표준오차, 꼬리 비교."""
    d = df["ret"] - df["bh"]
    g = df.assign(m=df["entry"].str[:7], d=d).groupby("m")["d"].mean()
    return {"평균차": float(d.mean()), "월묶음_t": float(g.mean() / (g.std(ddof=1) / np.sqrt(len(g)))) if len(g) > 2 else float("nan"),
            "규칙_P(-30%)": float((df["ret"] <= -0.3).mean()), "보유_P(-30%)": float((df["bh"] <= -0.3).mean()),
            "규칙_하위5%": float(df["ret"].quantile(0.05)), "보유_하위5%": float(df["bh"].quantile(0.05))}


def check():
    data = load_stocks()
    syms = sorted(data)
    print("종목", len(syms), "· 자료 오류 의심", sum(bad_bars(data[s]) for s in syms))
    d0, d1 = "2025-01-02", "2025-12-31"
    sample = syms[::10]
    # hold = bh 이어야 한다
    df = run("random", [Leg(1.0)], d0, d1, H=60, syms=sample)
    print("hold 다리 vs 들고만 있기 최대 차(0이어야 함)", float((df.ret - df.bh).abs().max()), "거래", len(df))
    # 미래 교란: 진입 뒤 가격을 바꿔도 진입 판단은 같아야 한다(신호는 s일 종가까지)
    b = data[sample[0]]
    e1 = entries(b, "dip7", d0, d1)
    b2 = S.Bars(b.sym, b.days, b.o.copy(), b.h.copy(), b.l.copy(), b.c.copy(), b.v)
    if len(e1):
        k = e1[0]
        b2.c[k:] *= 1.3
        b2.o[k:] *= 1.3; b2.h[k:] *= 1.3; b2.l[k:] *= 1.3
        print("진입일 이후 교란 뒤 첫 신호 같음:", entries(b2, "dip7", d0, d1)[0] == k)
    for kind in ("random", "dip7", "breakout", "pullback", "deep40"):
        base = run(kind, [Leg(1.0)], d0, d1, H=60, syms=sample)
        st = run(kind, [Leg(1.0, "trail", 0.20)], d0, d1, H=60, syms=sample)
        print(kind, "보유", {k: round(v, 3) for k, v in summary(base).items() if k in ("n", "기대값", "승률", "P(-30%이하)", "하위5%")},
              "| 추적20%", {k: round(v, 3) for k, v in summary(st).items() if k in ("기대값", "승률", "손익비", "P(-30%이하)", "하위5%")})


if __name__ == "__main__":
    check()
