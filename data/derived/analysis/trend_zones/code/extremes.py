"""극상방·극하방 정의 넓게 다시 찾기 + 경고를 SOXX 대신 개별주로(2026-10-11, 연구용, zones.py 다음 단계).

사용자(10/11): "극상방 RSI 70, 극하방 RSI 30이 맞아? 칼만 이평선 같은 더 좋은 방법은? 충분히 여러 조합을 다 해 봤어?
              SOXX만 보지 말고 여러 개별주도 다 봐 봐. SOXX는 ETF잖아."
지난 432개의 빈틈: 극단은 RSI14(70·75·80), (C-EMA20)/RMS60, 10일 수익 z 세 종류뿐이었다. 경고는 SOXX 하나였다.

사전 등록(결과를 본 뒤 바꾸지 않음)
  극단 정의(각각 위·아래 한 쌍; 아래는 위의 대칭):
    RSI(n) ≥ hi / ≤ 100-hi           n ∈ {2,3,5,7,9,14,21,28}, hi ∈ {60,65,70,75,80,85,90,95}            64개
    Stoch RSI K(사용자 Pine: RSI n, 스토캐스틱 14, K 평활 3) ≥ hi   n ∈ {7,14}, hi ∈ {80,90,95,98}       8개
    볼린저 z = (C-SMA_n)/표준편차_n ≥ k                       n ∈ {10,20,50}, k ∈ {1,1.5,2,2.5,3}           15개
    칼만식 띠 (C-EMA_L)/폭 ≥ b  (칼만선은 정상상태에서 EMA와 같다 — kalman_zones 보고서)
                               L ∈ {5,10,20,35,50,100}, 폭 ∈ {RMS20, RMS60, ATR14}, b ∈ {1,1.5,2,2.5,3}    90개
    N일 수익 z = ln(C/C[N])/(60일 표준편차·√N) ≥ k             N ∈ {3,5,10,20,60}, k ∈ {1,1.5,2,2.5,3}        25개
    52주 고점 대비(위만): C/최고가252 ≥ {0.95, 0.98, 1.0} / (아래만): ≤ {0.5, 0.6, 0.7}                      3+3개
    연속 상승·하락 일수 ≥ n                                    n ∈ {4,5,6,7}                                  4개
  채점(위·아래 따로, h = 10거래일, 같은 날 종목 순위로 — DRUG 같은 극단값에 휘둘리지 않게):
    rk = 그날 10일 수익 순위 백분위 - 0.5.  매도 타당도(위) = -평균 rk,  매수 타당도(아래) = +평균 rk (날짜마다 먼저 평균).
    자격: 앞 구간에서 그 쪽 비중 1~20%.
  선택: 앞 2026-01-02~05-29(정리) 1위를 위·아래 따로 고른다. 확인: 뒤 2026-06-01~10-09. 보조: 2025, 2024, 2023-07~12(아무도 안 본 구간).
  경고(맥락) 11개 — 극단이 맥락에 따라 뜻이 바뀌는지:
    시장: SOXX<SMA20, QQQ<SMA20 (각각 주간 = 그 주 직전 종가 / 일간 = t 종가)
    개별주 폭: 150종목 중 20일선 위 비율 < 50% (주간·일간), 반도체 종목 중 비율 < 50% (주간·일간)
    그 종목 섹터(watchlist) 안 20일선 위 비율 < 50% (일간, 자기 제외)
    그 종목 자신: C < SMA20 (일간), SMA20 5일 기울기 < 0 (일간)
    맥락 켜짐/꺼짐으로 나눠 위·아래 타당도를 재고, 켜짐-꺼짐 차이가 다섯 구간 모두 같은 부호인지 본다.
  개별주: 고른 정의와 RSI14 70/30에서 종목마다 '극단 뒤 10일 수익 - 그 종목의 같은 구간 평균'이 맞는 방향인 비율.
실행: PYTHONIOENCODING=utf-8 py data/derived/analysis/trend_zones/code/extremes.py
결과: results/extremes.txt, extremes.json
"""
import json, os, sys
from datetime import date

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import zones as Zm  # noqa: E402

H = 10
PERIODS = {"앞": ("2026-01-02", "2026-05-29", True), "뒤": ("2026-06-01", "2026-10-09", False),
           "2025": ("2025-01-02", "2025-12-31", False), "2024": ("2024-01-02", "2024-12-31", False),
           "2023하": ("2023-07-01", "2023-12-29", False)}
SOXX_SRC = os.path.join(Zm.ROOT, "data/derived/analysis/stop_research/cache/ohlc.parquet")


# ---------------------------------------------------------------- 지표
def rsi(C, n):
    d = C.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d).clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        r = 100 - 100 / (1 + up / dn)
    r = r.where(up > 0, 0.0).where(dn > 0, 100.0)          # 파인 ta.rsi와 같은 처리: 하락 0 → 100, 상승 0 → 0
    return r.where(up.notna() & dn.notna())


def stoch_rsi_k(C, n):
    r = rsi(C, n)
    lo, hi = r.rolling(14, min_periods=14).min(), r.rolling(14, min_periods=14).max()
    st = 100 * (r - lo) / (hi - lo)
    return st.rolling(3, min_periods=3).mean()


def streak(C):
    d = np.sign(C.diff().to_numpy())
    up = np.zeros_like(d)
    dn = np.zeros_like(d)
    for i in range(1, len(d)):
        up[i] = np.where(d[i] > 0, up[i - 1] + 1, 0)
        dn[i] = np.where(d[i] < 0, dn[i - 1] + 1, 0)
    return pd.DataFrame(up, index=C.index, columns=C.columns), pd.DataFrame(dn, index=C.index, columns=C.columns)


def definitions(I: Zm.Ind):
    """[(이름, 위 배열 또는 None, 아래 배열 또는 None)] — 배열은 dates × syms bool."""
    C = I.C
    out = []
    for n in (2, 3, 5, 7, 9, 14, 21, 28):
        r = rsi(C, n).to_numpy()
        for hi in (60, 65, 70, 75, 80, 85, 90, 95):
            out.append((f"RSI{n} {hi}/{100 - hi}", r >= hi, r <= 100 - hi))
    for n in (7, 14):
        k = stoch_rsi_k(C, n).to_numpy()
        for hi in (80, 90, 95, 98):
            out.append((f"StochRSI({n}) {hi}/{100 - hi}", k >= hi, k <= 100 - hi))
    for n in (10, 20, 50):
        m, s = C.rolling(n, min_periods=n).mean(), C.rolling(n, min_periods=n).std(ddof=0)
        z = ((C - m) / s).to_numpy()
        for k in (1.0, 1.5, 2.0, 2.5, 3.0):
            out.append((f"볼린저{n} ±{k}", z >= k, z <= -k))
    for L in (5, 10, 20, 35, 50, 100):
        for w in ("rms20", "rms60", "atr14"):
            z = I.zdev(L, w).to_numpy()
            for b in (1.0, 1.5, 2.0, 2.5, 3.0):
                out.append((f"칼만식 EMA{L}·{w} ±{b}", z >= b, z <= -b))
    sd60 = I.lr.rolling(60, min_periods=60).std()
    for N in (3, 5, 10, 20, 60):
        z = (np.log(C / C.shift(N)) / (sd60 * np.sqrt(N))).to_numpy()
        for k in (1.0, 1.5, 2.0, 2.5, 3.0):
            out.append((f"{N}일 수익 z ±{k}", z >= k, z <= -k))
    hh = I.H.rolling(252, min_periods=130).max()
    pos = (C / hh).to_numpy()
    for x in (0.95, 0.98, 1.0):
        out.append((f"52주 고점의 {x:.0%} 이상", pos >= x - 1e-12, None))
    for x in (0.5, 0.6, 0.7):
        out.append((f"52주 고점의 {x:.0%} 이하", None, pos <= x))
    up, dn = streak(C)
    for n in (4, 5, 6, 7):
        out.append((f"연속 {n}일 상승/하락", up.to_numpy() >= n, dn.to_numpy() >= n))
    return out


# ---------------------------------------------------------------- 수익(순위)·맥락
def rank_fwd(I: Zm.Ind, h):
    O = I.O
    f = (O.shift(-(1 + h)) / O.shift(-1) - 1).where(I.valid)
    n = f.notna().sum(axis=1)
    rk = (f.rank(axis=1) - 0.5).div(n, axis=0) - 0.5   # 가운데 맞춘 순위(평균 0). pct 순위는 +1/(2n) 치우침이 있다 — 검증 지적
    lo, hi = f.quantile(0.01, axis=1), f.quantile(0.99, axis=1)
    fw = f.clip(lower=lo, upper=hi, axis=0)
    ex = fw.sub(fw.mean(axis=1), axis=0)
    return rk.to_numpy(), ex.to_numpy(), f.to_numpy()


def week_state(dates, daily_state: np.ndarray):
    """일간 상태(t 종가) → 신호일 t의 '다음 날이 속한 주' 상태(그 주 직전 거래일 종가)."""
    iso = [date.fromisoformat(d).isocalendar()[:2] for d in dates]
    wk = np.zeros(len(dates), dtype=bool)
    cur, st = None, False
    for i in range(len(dates)):
        if iso[i] != cur:
            cur, st = iso[i], bool(daily_state[i - 1]) if i > 0 else False
        wk[i] = st
    out = np.zeros(len(dates), dtype=bool)
    out[:-1] = wk[1:]
    return out


def index_below(sym, dates):
    o = pd.read_parquet(SOXX_SRC, columns=["date", "sym", "Close"])
    s = o[o.sym == sym].copy()
    s["d"] = pd.to_datetime(s["date"]).dt.strftime("%Y-%m-%d")
    s = s.set_index("d")["Close"].astype(float).sort_index()
    s = s[~s.index.duplicated()]
    b = (s < s.rolling(20, min_periods=20).mean()).reindex(dates).ffill().fillna(False)
    return b.to_numpy(dtype=bool)


def contexts(I: Zm.Ind):
    from desk import config
    from desk.strategies import sector_of
    C = I.C
    sma = C.rolling(20, min_periods=20).mean()
    above = (C > sma).where(I.valid)
    sect = {s: sector_of(s, s) for s in I.syms}
    semis = np.array([sect[s] in config.SEMIS_SECTORS for s in I.syms])
    A = above.to_numpy(dtype=float)
    V = I.valid.to_numpy()
    with np.errstate(invalid="ignore"):
        br_all = np.nansum(A, axis=1) / V.sum(axis=1) < 0.5
        br_semi = np.nansum(A[:, semis], axis=1) / V[:, semis].sum(axis=1) < 0.5
    # 섹터 안(자기 제외) 20일선 위 비율
    sec_ctx = np.zeros_like(A, dtype=bool)
    for sec in set(sect.values()):
        cols = np.array([sect[s] == sec for s in I.syms])
        if cols.sum() < 3:
            continue
        a = np.nan_to_num(A[:, cols]); v = V[:, cols].astype(float)
        tot_a, tot_v = a.sum(axis=1, keepdims=True), v.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            share = (tot_a - a) / (tot_v - v)
        sec_ctx[:, cols] = share < 0.5
    own_below = (C < sma).to_numpy()
    own_slope = (sma < sma.shift(5)).to_numpy()
    soxx, qqq = index_below("SOXX", I.dates), index_below("QQQ", I.dates)
    n = len(I.dates)
    full = lambda v: np.repeat(v[:, None], len(I.syms), axis=1)
    return {
        "SOXX<20일선 주간": full(week_state(I.dates, soxx)), "SOXX<20일선 일간": full(soxx),
        "QQQ<20일선 주간": full(week_state(I.dates, qqq)), "QQQ<20일선 일간": full(qqq),
        "개별주 폭<50% 주간": full(week_state(I.dates, br_all)), "개별주 폭<50% 일간": full(br_all),
        "반도체주 폭<50% 주간": full(week_state(I.dates, br_semi)), "반도체주 폭<50% 일간": full(br_semi),
        "같은 섹터 폭<50%": sec_ctx, "자기 종목<20일선": own_below, "자기 20일선 하락": own_slope,
    }, semis


# ---------------------------------------------------------------- 평가
def rows_of(I, p, h):
    a, b, purge = PERIODS[p]
    idx = [i for i, d in enumerate(I.dates) if a <= d <= b and i + 1 + h < len(I.dates)]
    if purge:
        end = max(i for i, d in enumerate(I.dates) if a <= d <= b)
        idx = [i for i in idx if i + 1 + h <= end]
    return np.array(idx)


def score(M, rk, ex, raw, rows, sign, ctx=None):
    """M: bool dates×syms. 날짜마다 평균 → 날짜 평균. sign +1 = 아래(매수), -1 = 위(매도)."""
    if M is None:
        return None
    Mr = M[rows] & np.isfinite(rk[rows])
    if ctx is not None:
        Mr = Mr & ctx[rows]
    tot = np.isfinite(rk[rows]).sum()
    n = Mr.sum(axis=1)
    if n.sum() == 0:
        return {"비중": 0.0, "타당도": np.nan, "초과": np.nan, "상승": np.nan, "날짜": 0, "t": np.nan}
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(n > 0, np.where(Mr, rk[rows], 0).sum(axis=1) / n, np.nan)
        e = np.where(n > 0, np.where(Mr, ex[rows], 0).sum(axis=1) / n, np.nan)
        u = np.where(n > 0, (Mr & (raw[rows] > 0)).sum(axis=1) / n, np.nan)
    return {"비중": float(Mr.sum() / tot), "타당도": float(sign * np.nanmean(m)), "초과": float(np.nanmean(e)),
            "상승": float(np.nanmean(u)), "날짜": int(np.isfinite(m).sum()), "t": sign * Zm.nw_t(m, H)}


def main():
    I = Zm.Ind(*Zm.load())
    rk, ex, raw = rank_fwd(I, H)
    R = {p: rows_of(I, p, H) for p in PERIODS}
    D = definitions(I)
    V = I.valid.to_numpy()
    D = [(nm, None if t is None else (t & V), None if b is None else (b & V)) for nm, t, b in D]
    lines = [f"극단 정의 {len(D)}개(위 {sum(t is not None for _, t, _ in D)}·아래 {sum(b is not None for _, _, b in D)}), "
             f"h={H}, 같은 날 순위 기준(타당도 = 위는 덜 오름, 아래는 더 오름이 +)"]
    res = []
    for nm, t, b in D:
        r = {"name": nm}
        for side, M, sg in (("위", t, -1), ("아래", b, +1)):
            r[side] = {p: score(M, rk, ex, raw, R[p], sg) for p in PERIODS} if M is not None else None
        res.append(r)
    pct = lambda v: "   —  " if v is None or not np.isfinite(v) else f"{v * 100:+6.2f}"
    best = {}
    for side in ("위", "아래"):
        ok = [r for r in res if r[side] is not None and 0.01 <= r[side]["앞"]["비중"] <= 0.20]
        ok.sort(key=lambda r: -r[side]["앞"]["타당도"])
        best[side] = ok[0]
        lab = "극상방(팔 자리인가: 이후 덜 오르면 +)" if side == "위" else "극하방(살 자리인가: 이후 더 오르면 +)"
        lines += ["", f"== {lab} — 자격 {len(ok)}개, 순위 타당도 %p (앞 기준 상위 12)",
                  f"{'정의':<30} {'앞':>7} {'뒤':>7} {'2025':>7} {'2024':>7} {'2023하':>7} | 앞 비중"]
        for r in ok[:12]:
            s = r[side]
            lines.append(f"{r['name']:<30} " + " ".join(pct(s[p]['타당도']) for p in PERIODS) + f" | {s['앞']['비중'] * 100:4.1f}%")
        for nm in ("RSI14 70/30", "볼린저20 ±2.0", "칼만식 EMA20·atr14 ±2.0", "StochRSI(14) 95/5"):
            r = next(x for x in res if x["name"] == nm)
            s = r[side]
            lines.append(f"(참고) {nm:<24} " + " ".join(pct(s[p]['타당도']) for p in PERIODS) + f" | {s['앞']['비중'] * 100:4.1f}%")
        for p in PERIODS:
            v = np.array([r[side][p]["타당도"] for r in ok])
            v = v[np.isfinite(v)]
            lines.append(f"[{p}] 타당도 양수 {np.mean(v > 0) * 100:3.0f}%  중앙 {np.median(v) * 100:+.2f}%p")
        both = [r for r in ok if all(r[side][p]["타당도"] > 0 for p in ("앞", "뒤"))]
        allp = [r for r in ok if all(np.isfinite(r[side][p]["타당도"]) and r[side][p]["타당도"] > 0 for p in PERIODS)]
        lines.append(f"앞·뒤 모두 양수 {len(both)}개, 다섯 구간 모두 양수 {len(allp)}개: "
                     + ", ".join(r["name"] for r in allp[:10]))
        b0 = best[side]
        lines.append(f"→ 1~5월 1위 {b0['name']}: " + "  ".join(
            f"{p} {pct(b0[side][p]['타당도'])}(t {b0[side][p]['t']:+.1f}, 초과 {pct(b0[side][p]['초과'])}, 상승 {b0[side][p]['상승'] * 100:3.0f}%)"
            for p in PERIODS))
    # 맥락
    CTX, semis = contexts(I)
    lines += ["", "== 맥락(경고) 켜짐 / 꺼짐에서 타당도 %p — 자격 정의 전체의 중앙값, 괄호는 양수 비율"]
    ctx_out = {}
    for side in ("위", "아래"):
        ok = [r for r in res if r[side] is not None and 0.01 <= r[side]["앞"]["비중"] <= 0.20]
        lines.append(f"-- {'극상방' if side == '위' else '극하방'}")
        for cn, cm in CTX.items():
            parts, diffs = [], []
            for p in PERIODS:
                on, off = [], []
                for r in ok:
                    M = next(t if side == "위" else b for nm, t, b in D if nm == r["name"])
                    a = score(M, rk, ex, raw, R[p], -1 if side == "위" else 1, cm)
                    c = score(M, rk, ex, raw, R[p], -1 if side == "위" else 1, ~cm)
                    if a["날짜"] >= 5 and c["날짜"] >= 5:
                        on.append(a["타당도"]); off.append(c["타당도"])
                on, off = np.array(on), np.array(off)
                d = on - off
                diffs.append(np.median(d) if len(d) else np.nan)
                parts.append(f"{p} 켜짐 {np.median(on) * 100:+5.2f} 꺼짐 {np.median(off) * 100:+5.2f} 차 {np.median(d) * 100:+5.2f}({np.mean(d > 0) * 100:3.0f}%)"
                             if len(d) else f"{p} —")
            same = np.all(np.array(diffs) > 0) or np.all(np.array(diffs) < 0)
            ctx_out[f"{side}·{cn}"] = diffs
            lines.append(f"{cn:<16} {'★다섯 구간 같은 방향' if same else ''}\n    " + " | ".join(parts))
    # 개별주: 종목마다
    lines += ["", "== 개별주: 극단 뒤 10일 순위가 그 종목 평소보다 맞는 방향인 종목 비율(극단 5회 이상 종목만)"]
    for nm in (best["위"]["name"], best["아래"]["name"], "RSI14 70/30"):
        t, b = next((t, b) for n2, t, b in D if n2 == nm)
        for p in ("앞", "뒤", "2025"):
            rows = R[p]
            out = []
            for side, M, sg in (("위", t, -1), ("아래", b, 1)):
                if M is None:
                    continue
                good, cnt = 0, 0
                for j in range(len(I.syms)):
                    x = rk[rows, j]
                    m = M[rows, j] & np.isfinite(x)
                    if m.sum() >= 5 and np.isfinite(x).sum() > m.sum():
                        cnt += 1
                        good += sg * (x[m].mean() - x[np.isfinite(x) & ~m].mean()) > 0
                out.append(f"{side} {good}/{cnt}")
            lines.append(f"{nm:<28} [{p}] " + "  ".join(out))
    txt = "\n".join(lines)
    print(txt)
    os.makedirs(Zm.OUT, exist_ok=True)
    with open(os.path.join(Zm.OUT, "extremes.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    with open(os.path.join(Zm.OUT, "extremes.json"), "w", encoding="utf-8") as f:
        json.dump({"defs": res, "best": {k: v["name"] for k, v in best.items()}, "ctx": ctx_out}, f,
                  ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
