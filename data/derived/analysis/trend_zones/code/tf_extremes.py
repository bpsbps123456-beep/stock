"""극상방·극하방을 시간봉마다 다시 재기(2026-10-11, 연구용, extremes.py 다음 단계).

사용자(10/11): "40분봉·160분봉·320분봉 등 10분 단위로 다양하게 조합해도 그래요? 일봉도 살펴보고요."
자료 제약: yfinance 5분봉은 최근 60거래일(7/17~10/9)뿐이고 1시간봉은 2023-11부터 있다.

사전 등록
  A. 10분 단위 39개(10, 20, …, 390분): rsi_breakout/cache/5m.parquet(150종목 정규장 5분봉, 7/17~10/9)을 매일 09:30부터 묶는다
     (트레이딩뷰 정규장 분봉과 같은 시작; 마지막 봉은 16:00에 잘린다). 지표는 그 시간봉의 봉 수로 잰다.
     봉 40개가 쌓인 뒤부터 쓰므로 봉 40개 안에서 계산되는 정의만 쓴다(약 100개): RSI 2~14, Stoch RSI, 볼린저 10·20,
     칼만식 EMA 5·10·20 × (RMS20·ATR14), 연속 일수. 기간은 7/17~10/9 하나(2026 '뒤' 안).
  B. 60분 배수 6개(60, 120, 180, 240, 300, 360분)와 하루(390분): premarket_shock/bars/1h_*.parquet 정규장 봉(2023-11~10/7)
     + rsi_breakout 1시간봉(없는 7종목·10/8~9). extremes.py의 212개 전부, 봉 130개 뒤부터.
     구간: 앞 2026-01-02~05-29(끝 정리), 뒤 06-01~10-09, 2025, 2024.
  성과: 신호 봉이 끝난 다음 5분(또는 1시간) 봉 시가에 사서, h 거래일 뒤 같은 시각 시가까지. h ∈ {1, 5}일(B는 10일도).
        그 시각 종목끼리 순위(가운데 맞춘 백분위). 시각마다 평균 → 하루 평균 → 날짜 평균. 위는 −평균(덜 오르면 +), 아래는 +평균.
  자격: 그 시간봉·구간에서 그 쪽 비중 1~20%.
  조합(A만): 같은 정의가 두 시간봉에서 동시에 극단일 때(짧은 봉이 끝나는 시각에 긴 봉의 마지막 완성봉 상태)
        40+160, 40+320, 80+160, 160+320, 그리고 40·160·320 각각 + 일봉(전날 종가까지의 일봉, 1d_long).
실행: PYTHONIOENCODING=utf-8 py data/derived/analysis/trend_zones/code/tf_extremes.py
결과: results/tf_extremes.txt, tf_extremes.json
"""
import glob, json, os, re, sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import zones as Zm  # noqa: E402
import extremes as X  # noqa: E402

ROOT = Zm.ROOT
FIVE = os.path.join(ROOT, "data/derived/analysis/rsi_breakout/cache/5m.parquet")
HOUR_RB = os.path.join(ROOT, "data/derived/analysis/rsi_breakout/cache/1h.parquet")
HOUR_PS = os.path.join(ROOT, "data/derived/analysis/premarket_shock/bars")
REG_1H = {"09:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:30"}
SHORT_OK = re.compile(r"^(RSI(2|3|5|7|9|14) |StochRSI|볼린저(10|20) |칼만식 EMA(5|10|20)·(rms20|atr14) |연속 )")
PERIODS_B = {"앞": ("2026-01-02", "2026-05-29", True), "뒤": ("2026-06-01", "2026-10-09", False),
             "2025": ("2025-01-02", "2025-12-31", False), "2024": ("2024-01-02", "2024-12-31", False)}


# ---------------------------------------------------------------- 기본 격자(5분·1시간 정규장)
class Base:
    def __init__(self, O, H, L, C):
        self.O, self.H, self.L, self.C = O, H, L, C       # index: tz-aware NY timestamps
        ts = O.index
        self.day = np.array(ts.strftime("%Y-%m-%d"))
        days, self.dayi = np.unique(self.day, return_inverse=True)
        self.days = list(days)
        self.slot = np.zeros(len(ts), dtype=int)
        first = np.r_[0, np.flatnonzero(self.dayi[1:] != self.dayi[:-1]) + 1]
        for k, s in enumerate(first):
            e = first[k + 1] if k + 1 < len(first) else len(ts)
            self.slot[s:e] = np.arange(e - s)
        self.first = first
        self.last = np.r_[first[1:] - 1, len(ts) - 1]

    def exit_index(self, entry, h):
        """entry(기본 봉 번호)에서 h 거래일 뒤 같은 칸. 그날 칸이 모자라면(반일장) 그날 마지막 봉."""
        d = self.dayi[entry] + h
        ok = d < len(self.days)
        d = np.minimum(d, len(self.days) - 1)
        idx = np.minimum(self.first[d] + self.slot[entry], self.last[d])
        return np.where(ok, idx, -1)


def load_five():
    x = pd.read_parquet(FIVE)
    x["ts"] = pd.to_datetime(x["ts"])
    x = x[x.sym != "QQQ"]
    f = {k: x.pivot_table(index="ts", columns="sym", values=k).sort_index() for k in ("Open", "High", "Low", "Close")}
    return Base(f["Open"], f["High"], f["Low"], f["Close"])


def load_hour():
    syms = [s for s in pd.read_parquet(Zm.SRC).sym.unique() if s != "QQQ"]
    rb = pd.read_parquet(HOUR_RB)
    rb["ts"] = pd.to_datetime(rb["ts"])
    frames = []
    for s in syms:
        p = os.path.join(HOUR_PS, f"1h_{s}.parquet")
        if os.path.exists(p):
            a = pd.read_parquet(p)
            a.index = a.index.tz_convert("America/New_York")
            a = a[a.index.strftime("%H:%M").isin(REG_1H)]
            b = rb[(rb.sym == s) & (rb.ts > a.index.max())].set_index("ts")[["Open", "High", "Low", "Close"]]
            a = pd.concat([a[["Open", "High", "Low", "Close"]], b])
        else:
            a = rb[rb.sym == s].set_index("ts")[["Open", "High", "Low", "Close"]]
        a = a[~a.index.duplicated()]
        a["sym"] = s
        frames.append(a.reset_index().rename(columns={"index": "ts", "Datetime": "ts"}))
    x = pd.concat(frames)
    x["ts"] = pd.to_datetime(x["ts"], utc=True).dt.tz_convert("America/New_York")
    f = {k: x.pivot_table(index="ts", columns="sym", values=k).sort_index() for k in ("Open", "High", "Low", "Close")}
    return Base(f["Open"], f["High"], f["Low"], f["Close"])


# ---------------------------------------------------------------- 시간봉 만들기
def resample(B: Base, per_bar: int, whole_day=False):
    """기본 봉 per_bar개씩 그날 첫 봉부터 묶는다. 반환: (O,H,L,C DataFrame, 각 봉의 마지막 기본 봉 번호, 날짜)."""
    grp = B.dayi * 10000 + (np.zeros_like(B.slot) if whole_day else B.slot // per_bar)
    _, start = np.unique(grp, return_index=True)
    start = np.sort(start)
    end = np.r_[start[1:] - 1, len(grp) - 1]
    O = B.O.to_numpy()[start]
    C = B.C.to_numpy()[end]
    Hh = np.maximum.reduceat(np.nan_to_num(B.H.to_numpy(), nan=-np.inf), start, axis=0)
    Ll = np.minimum.reduceat(np.nan_to_num(B.L.to_numpy(), nan=np.inf), start, axis=0)
    Hh = np.where(np.isfinite(Hh), Hh, np.nan)
    Ll = np.where(np.isfinite(Ll), Ll, np.nan)
    idx = [str(t) for t in B.O.index[end]]
    mk = lambda a: pd.DataFrame(a, index=idx, columns=B.O.columns)
    return mk(O), mk(Hh), mk(Ll), mk(C), end, B.day[end]


def fwd_rank(B: Base, end, h):
    entry = end + 1
    ok = entry < len(B.day)
    entry = np.minimum(entry, len(B.day) - 1)
    ex = B.exit_index(entry, h)
    Ob = B.O.to_numpy()
    f = np.where((ok & (ex >= 0))[:, None], Ob[np.maximum(ex, 0)] / Ob[entry] - 1, np.nan)
    df = pd.DataFrame(f)
    n = df.notna().sum(axis=1)
    rk = ((df.rank(axis=1) - 0.5).div(n, axis=0) - 0.5).to_numpy()
    return rk


def score(M, rk, days, sel, sign):
    """sel: 쓸 봉 위치(bool). 봉마다 평균 → 날마다 평균 → 날짜 평균."""
    Mr = M[sel] & np.isfinite(rk[sel])
    tot = np.isfinite(rk[sel]).sum()
    n = Mr.sum(axis=1)
    if tot == 0 or n.sum() == 0:
        return {"비중": 0.0, "타당도": np.nan, "날": 0}
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(n > 0, np.where(Mr, rk[sel], 0).sum(axis=1) / n, np.nan)
    s = pd.Series(m).groupby(days[sel]).mean().dropna()
    return {"비중": float(Mr.sum() / tot), "타당도": float(sign * s.mean()), "날": int(len(s))}


def defs_for(O, Hh, L, C, min_bars, short_only):
    Zm.MIN_BARS = min_bars
    I = Zm.Ind(O, Hh, L, C)
    D = X.definitions(I)
    if short_only:
        D = [d for d in D if SHORT_OK.match(d[0])]
    V = I.valid.to_numpy()
    return [(nm, None if t is None else (t & V), None if b is None else (b & V)) for nm, t, b in D], V


def summarize(rows, side):
    v = np.array([r[side] for r in rows if r[side] is not None and np.isfinite(r[side])])
    return (np.mean(v > 0), np.median(v), len(v)) if len(v) else (np.nan, np.nan, 0)


# ---------------------------------------------------------------- A. 10분 단위(7/17~10/9)
def part_a(lines, out):
    B = load_five()
    lines += [f"== A. 5분봉 → 10~390분 39개, {B.days[0]}~{B.days[-1]} ({len(B.days)}일), 봉 40개 뒤부터, 짧은 정의만",
              f"{'봉':>5} | {'극상방 매도 타당도 양수 비율·중앙 (1일 / 5일)':<40} | {'극하방 매수 (1일 / 5일)':<34} | RSI14 70 / 30 (5일)"]
    keep = {}
    for tf in range(10, 400, 10):
        O, Hh, L, C, end, days = resample(B, tf // 5)
        D, V = defs_for(O, Hh, L, C, 40, True)
        rk = {h: fwd_rank(B, end, h) for h in (1, 5)}
        sel = np.ones(len(days), dtype=bool)
        res = []
        for nm, t, b in D:
            r = {"name": nm}
            for h in (1, 5):
                for side, M, sg in (("위", t, -1), ("아래", b, 1)):
                    if M is None:
                        r[f"{side}{h}"] = None
                        continue
                    s = score(M, rk[h], days, sel, sg)
                    r[f"{side}{h}"] = s["타당도"] if 0.01 <= s["비중"] <= 0.20 else None
            res.append(r)
        rsi = next(r for r in res if r["name"] == "RSI14 70/30")
        part = []
        for side in ("위", "아래"):
            a1, m1, n1 = summarize(res, f"{side}1")
            a5, m5, n5 = summarize(res, f"{side}5")
            part.append(f"{a1 * 100:3.0f}% {m1 * 100:+5.2f} / {a5 * 100:3.0f}% {m5 * 100:+5.2f} (n{n5:3d})")
        f = lambda v: "  —  " if v is None else f"{v * 100:+5.2f}"
        lines.append(f"{tf:>4}분 | {part[0]:<40} | {part[1]:<34} | {f(rsi['위5'])} / {f(rsi['아래5'])}")
        out["A"][tf] = res
        if tf in (40, 80, 160, 320):
            keep[tf] = (D, end, days, rk)
    return B, keep


def part_combo(B, keep, lines, out):
    """같은 정의가 두 시간봉에서 동시에 극단. 짧은 봉 끝 시각에 긴 봉의 마지막 완성봉 상태(또는 전날 일봉)."""
    Zm.MIN_BARS = 130
    Idaily = Zm.Ind(*Zm.load())
    Dd = {nm: (t, b) for nm, t, b in X.definitions(Idaily)}
    dpos = {d: i for i, d in enumerate(Idaily.dates)}
    syms = list(B.O.columns)
    jcol = [Idaily.syms.index(s) if s in Idaily.syms else -1 for s in syms]
    lines += ["", "== A-조합: 같은 정의가 두 봉에서 동시에 극단(5일 성과, 7/17~10/9) — 자격 정의 중 양수 비율·중앙, RSI14 70/30"]

    def state_on_base(tf, M):
        """긴 봉 M(봉×종목)을 기본 봉 격자로: 기본 봉 i 시점에 끝나 있는 가장 최근 긴 봉의 상태."""
        D, end, days, rk = keep[tf]
        S = np.zeros((len(B.day), M.shape[1]), dtype=bool)
        k = np.searchsorted(end, np.arange(len(B.day)), side="right") - 1   # end <= i 인 마지막 긴 봉
        ok = k >= 0
        S[ok] = M[k[ok]]
        return S

    def daily_on_base(M):
        S = np.zeros((len(B.day), len(syms)), dtype=bool)
        prev = []
        for d in B.day:
            i = dpos.get(d)
            prev.append(i - 1 if i is not None else -1)
        prev = np.array(prev)
        for j, c in enumerate(jcol):
            if c >= 0:
                S[:, j] = np.where(prev >= 0, M[np.maximum(prev, 0), c], False)
        return S

    pairs = [(40, 160), (40, 320), (80, 160), (160, 320), (40, "일봉"), (160, "일봉"), (320, "일봉")]
    for a, b in pairs:
        Da, enda, daysa, rka = keep[a]
        res = []
        for nm, ta, ba in Da:
            if b == "일봉":
                if nm not in Dd:
                    continue
                tb, bb = Dd[nm]
                tb_s = None if tb is None else daily_on_base(tb)
                bb_s = None if bb is None else daily_on_base(bb)
            else:
                Db = {n2: (t2, b2) for n2, t2, b2 in keep[b][0]}
                tb, bb = Db[nm]
                tb_s = None if tb is None else state_on_base(b, tb)
                bb_s = None if bb is None else state_on_base(b, bb)
            r = {"name": nm}
            for side, Ma, Sb, sg in (("위", ta, tb_s, -1), ("아래", ba, bb_s, 1)):
                if Ma is None or Sb is None:
                    r[side] = None
                    continue
                M = Ma & Sb[enda]
                s = score(M, rka[5], daysa, np.ones(len(daysa), dtype=bool), sg)
                r[side] = s["타당도"] if s["비중"] >= 0.002 and s["날"] >= 5 else None
                r[side + "비중"] = s["비중"]
            res.append(r)
        rsi = next((r for r in res if r["name"] == "RSI14 70/30"), None)
        part = []
        for side in ("위", "아래"):
            a5, m5, n5 = summarize(res, side)
            part.append(f"{side} {a5 * 100:3.0f}% {m5 * 100:+5.2f} (n{n5:3d})")
        f = lambda v: "  —  " if v is None else f"{v * 100:+5.2f}"
        lines.append(f"{a}분+{b}{'분' if b != '일봉' else ''}: " + " | ".join(part)
                     + f" | RSI14 70 {f(rsi and rsi['위'])} / 30 {f(rsi and rsi['아래'])}")
        out["combo"][f"{a}+{b}"] = res


# ---------------------------------------------------------------- B. 60분 배수 + 하루(2024~2026)
def part_b(lines, out):
    B = load_hour()
    lines += ["", f"== B. 1시간봉 → 60~360분·하루, {B.days[0]}~{B.days[-1]}, 정의 212개, 봉 130개 뒤부터",
              "   칸: 극상방 / 극하방 타당도 양수 비율(자격 정의 중), h = 5일 [1일·10일은 json]. 끝: 네 구간 모두 양수인 정의 수, RSI14 70/30(앞/뒤/2025/2024)"]
    for tf, per in ((60, 1), (120, 2), (180, 3), (240, 4), (300, 5), (360, 6), ("하루", None)):
        O, Hh, L, C, end, days = resample(B, per or 1, whole_day=per is None)
        D, V = defs_for(O, Hh, L, C, 130, False)
        rk = {h: fwd_rank(B, end, h) for h in (1, 5, 10)}
        sels = {}
        for p, (a, b, purge) in PERIODS_B.items():
            s = (days >= a) & (days <= b)
            if purge:
                last = max(i for i, d in enumerate(B.days) if d <= b)
                s &= np.array([B.dayi[e] + 1 + 5 <= last for e in end])
            sels[p] = s
        res = []
        for nm, t, b in D:
            r = {"name": nm}
            for h in (1, 5, 10):
                for side, M, sg in (("위", t, -1), ("아래", b, 1)):
                    for p, s in sels.items():
                        if M is None:
                            r[f"{side}{h}{p}"] = None
                            continue
                        sc = score(M, rk[h], days, s, sg)
                        r[f"{side}{h}{p}"] = sc["타당도"] if 0.01 <= sc["비중"] <= 0.20 else None
            res.append(r)
        out["B"][str(tf)] = res
        cells = []
        for side in ("위", "아래"):
            sh = []
            for p in PERIODS_B:
                a5, m5, n5 = summarize(res, f"{side}5{p}")
                sh.append(f"{a5 * 100:3.0f}%")
            allp = sum(1 for r in res if all(r[f"{side}5{p}"] is not None and r[f"{side}5{p}"] > 0 for p in PERIODS_B))
            cells.append(f"{side} " + "/".join(sh) + f" 네 구간 모두 {allp}")
        rsi = next(r for r in res if r["name"] == "RSI14 70/30")
        f = lambda v: "—" if v is None else f"{v * 100:+.1f}"
        rr = " ".join(f(rsi[f"위5{p}"]) for p in PERIODS_B) + " / " + " ".join(f(rsi[f"아래5{p}"]) for p in PERIODS_B)
        lines.append(f"{str(tf):>4}{'분' if tf != '하루' else ''} | " + " | ".join(cells) + f" | RSI14 {rr}")


def main():
    out = {"A": {}, "combo": {}, "B": {}}
    lines = []
    B, keep = part_a(lines, out)
    part_combo(B, keep, lines, out)
    part_b(lines, out)
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(Zm.OUT, "tf_extremes.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    with open(os.path.join(Zm.OUT, "tf_extremes.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, default=lambda o: None if o is None else float(o))


if __name__ == "__main__":
    main()
