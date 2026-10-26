"""F2 선 회복·높은 저점·투매 — BRIEF.md(2026-10-26, 결과 전 고정) 갈래 F2, 규칙 7개.

신호만 만든다(평가·대조·판정은 bc_lib.run_rules). 신호일 t = 그날 종가까지 아는 정보만 쓴다.

  ma10x / ma20x / ma50x  C[t] > SMA_n[t] 이고 C[t-1] <= SMA_n[t-1]         (SMA = 종가 n일 평균, t에서 끝나는 창)
  ma20x2                 C[t] > SMA20[t], C[t-1] > SMA20[t-1], C[t-2] <= SMA20[t-2]
  ema1020x               EMA10[t] > EMA20[t] 이고 EMA10[t-1] <= EMA20[t-1]  (ewm(span, adjust=False))
  hl (높은 저점)          pk = t에서 끝나는 60일 종가 최고 위치(같으면 늦은 날), l = C[pk..t] 최저의 마지막 위치.
                         l < s <= t-2, C[s] == min(C[s-2..s+2])(s+2 <= t 라 t에 확정), C[s] >= 1.01·C[l] 인
                         가장 최근 s. H = max(C[l..s]). C[t] > H 이고 C[t-1] <= H 이면 켜짐.
  capit (SOXX만)          d ∈ [t-9, t], SOXX 종가 수익 <= -3% 이고 달러 거래량[d] >= 2·mean(달러 거래량[d-20..d-1]).
                         가장 최근 d 에 대해 C[t] > C[d] 이면 켜짐. 달러 거래량은 그날 종가·거래량으로 장 마감에 알려진다
                         (발행 주식 수·자금 흐름이 아니므로 하루 늦춤 대상 아님). 2026-10-13까지만 있다.

실행: data/derived/analysis/bottom_confirm 에서 `python3 -W ignore code/f2_reclaim.py`
결과: results/f2_SOXX_{summary,signals,verdict}.csv, results/f2_STK_{...}.csv
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import bc_lib  # noqa: E402

FAMILY = "f2"
KNOWN_LOWS = ("2024-08-07", "2025-04-08", "2026-03-30", "2026-07-29")


# ---------------------------------------------------------------- 하락 저점 위치
def decline_low_pos(C: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """pk[t, k] = t에서 끝나는 60일 종가 최고의 위치(같은 값이면 늦은 날, bc_lib.outcome_frame 과 같음),
    l[t, k] = C[pk..t] 최저 종가의 마지막 위치. 60일 창에 결측이 있으면 -1."""
    Cv = C.to_numpy(float)
    N, K = Cv.shape
    PK = np.full((N, K), -1, int)
    LL = np.full((N, K), -1, int)
    for k in range(K):
        c = Cv[:, k]
        for t in range(59, N):
            w = c[t - 59:t + 1]
            if np.isnan(w).any():
                continue
            pk = t - 59 + (59 - int(np.argmax(w[::-1])))
            seg = c[pk:t + 1]
            l = pk + (len(seg) - 1 - int(np.argmin(seg[::-1])))     # 최저의 마지막 위치
            PK[t, k], LL[t, k] = pk, l
    return PK, LL


# ---------------------------------------------------------------- 규칙
def sma_cross(C: pd.DataFrame, n: int) -> pd.DataFrame:
    s = C.rolling(n, min_periods=n).mean()
    return (C > s) & (C.shift(1) <= s.shift(1))


def ma20x2(C: pd.DataFrame) -> pd.DataFrame:
    s = C.rolling(20, min_periods=20).mean()
    return (C > s) & (C.shift(1) > s.shift(1)) & (C.shift(2) <= s.shift(2))


def ema_cross(C: pd.DataFrame, a: int = 10, b: int = 20) -> pd.DataFrame:
    ea = C.ewm(span=a, adjust=False).mean()
    eb = C.ewm(span=b, adjust=False).mean()
    return (ea > eb) & (ea.shift(1) <= eb.shift(1))


def higher_low(C: pd.DataFrame, PK: np.ndarray, LL: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """반환: (신호 bool, 진단용 s 위치(-1 = 없음))."""
    Cv = C.to_numpy(float)
    N, K = Cv.shape
    # 5일 국소 저점: C[s] == min(C[s-2..s+2]). s+2 가 필요하므로 t 에서는 s <= t-2 만 쓴다.
    sw = np.zeros((N, K), bool)
    for s in range(2, N - 2):
        w = Cv[s - 2:s + 3]
        ok = ~np.isnan(w).any(0)
        sw[s] = ok & (Cv[s] == np.nanmin(np.where(np.isnan(w), np.inf, w), 0))
    out = np.zeros((N, K), bool)
    S = np.full((N, K), -1, int)
    for k in range(K):
        c = Cv[:, k]
        for t in range(2, N):
            l = LL[t, k]
            if l < 0 or np.isnan(c[t]) or np.isnan(c[t - 1]):
                continue
            thr = 1.01 * c[l]
            s_hit = -1
            for s in range(t - 2, l, -1):                         # l < s <= t-2, 가장 최근부터
                if sw[s, k] and c[s] >= thr:
                    s_hit = s
                    break
            if s_hit < 0:
                continue
            S[t, k] = s_hit
            H = c[l:s_hit + 1].max()
            out[t, k] = (c[t] > H) and (c[t - 1] <= H)
    return (pd.DataFrame(out, index=C.index, columns=C.columns),
            pd.DataFrame(S, index=C.index, columns=C.columns))


def capitulation(C: pd.Series, dvol: pd.Series) -> tuple[pd.Series, pd.Series]:
    """SOXX 투매 뒤 회복. 반환: (신호, 투매일 표시)."""
    ret = C.pct_change()
    dv = dvol.reindex(C.index)
    base = dv.shift(1).rolling(20, min_periods=20).mean()        # d-20..d-1
    cap = (ret <= -0.03) & (dv >= 2.0 * base)
    cap = cap.fillna(False).astype(bool)
    cv, capv = C.to_numpy(float), cap.to_numpy()
    out = np.zeros(len(C), bool)
    for t in range(len(C)):
        for d in range(t, max(t - 9, 0) - 1, -1):                 # d ∈ [t-9, t], 가장 최근부터
            if capv[d]:
                out[t] = cv[t] > cv[d]
                break
    return pd.Series(out, index=C.index), cap


def build_rules(C: pd.DataFrame, kind: str, dvol: pd.Series | None = None) -> tuple[dict, dict]:
    PK, LL = decline_low_pos(C)
    hl, hl_s = higher_low(C, PK, LL)
    rules = {
        "ma10x": sma_cross(C, 10),
        "ma20x": sma_cross(C, 20),
        "ma50x": sma_cross(C, 50),
        "ma20x2": ma20x2(C),
        "ema1020x": ema_cross(C, 10, 20),
        "hl": hl,
    }
    diag = dict(PK=PK, LL=LL, hl_s=hl_s)
    if kind == "SOXX":
        cs, capd = capitulation(C["SOXX"], dvol)
        rules["capit"] = cs.to_frame("SOXX")
        diag["cap_days"] = capd
    return {k: v.fillna(False).astype(bool) for k, v in rules.items()}, diag


# ---------------------------------------------------------------- 점검
def sanity(rules: dict, fr: dict, kind: str):
    print(f"\n[{kind}] 규칙별 원 신호일 수(전체 / in_dd·평가 창 안) · 첫 신호 수")
    for name, sig in rules.items():
        s = sig.reindex(index=fr["in_dd"].index, columns=fr["in_dd"].columns).fillna(False)
        raw = int(s.to_numpy().sum())
        win = s & fr["in_dd"] & fr["L60"].notna()
        win = win.loc[(win.index >= bc_lib.EVAL_START[kind]) & (win.index <= bc_lib.LAST)]
        fs = bc_lib.first_signals(sig, fr, kind)
        print(f"  {name:9s} raw={raw:6d}  in_dd·window={int(win.to_numpy().sum()):6d}  first={len(fs):5d}"
              + (f"  first_sig={fs.date.min().date()}" if len(fs) else ""))


def soxx_lows_check(rules: dict, fr: dict, diag: dict):
    C = bc_lib.load_soxx()["C"]
    idx = C.index
    print("\n[SOXX] 알려진 저점 주변 신호(저점 -3 ~ +40 세션). 표시: 원 신호(*)·첫 신호(F)")
    firsts = {n: set(bc_lib.first_signals(s, fr, "SOXX").date) for n, s in rules.items()}
    for low in KNOWN_LOWS:
        p = idx.get_loc(pd.Timestamp(low))
        lo_pos = diag["LL"][p + 5, 0] if p + 5 < len(idx) else -1
        print(f"  저점 {low}: C={C.iloc[p]:.2f}, Ldd(t+5)={fr['L60'].iat[p + 5, 0]:.2f} at {idx[lo_pos].date()}"
              f", in_dd={bool(fr['in_dd'].iat[p, 0])}")
        a, b = max(p - 3, 0), min(p + 40, len(idx) - 1)
        for name, s in rules.items():
            hits = []
            for q in range(a, b + 1):
                if s.iat[q, 0]:
                    d = idx[q]
                    tagc = "F" if d in firsts[name] else "*"
                    hits.append(f"{d.strftime('%m-%d')}{tagc}(+{q - p},up={fr['up'].iat[q, 0] * 100:.1f}%)")
            print(f"    {name:9s} " + (" ".join(hits[:6]) if hits else "-"))
    cap = diag["cap_days"]
    print("  투매일(-3% & 달러 거래량 2배) 목록 2024~:",
          ", ".join(d.strftime("%Y-%m-%d") for d in cap[cap].index if d >= pd.Timestamp("2024-01-01")))
    # 높은 저점: 각 저점 뒤 첫 hl 신호의 s·l·H 를 보여 준다
    Cv = C.to_numpy(float)
    for low in KNOWN_LOWS:
        p = idx.get_loc(pd.Timestamp(low))
        for q in range(p, min(p + 60, len(idx))):
            if rules["hl"].iat[q, 0]:
                s = diag["hl_s"].iat[q, 0]
                l = diag["LL"][q, 0]
                H = Cv[l:s + 1].max()
                print(f"  hl 저점 {low} 뒤 첫 신호 {idx[q].date()}: l={idx[l].date()}({Cv[l]:.2f}) "
                      f"s={idx[s].date()}({Cv[s]:.2f}, +{(Cv[s] / Cv[l] - 1) * 100:.1f}%) H={H:.2f} "
                      f"C[t-1]={Cv[q - 1]:.2f} C[t]={Cv[q]:.2f}")
                break


def main():
    pd.set_option("display.width", 220)
    mk = bc_lib.load_market()
    out = {}

    # SOXX
    fr_s = bc_lib.soxx_frames()
    C_s = bc_lib.load_soxx()[["C"]].rename(columns={"C": "SOXX"})
    assert C_s.index.equals(fr_s["in_dd"].index)
    rules_s, diag_s = build_rules(C_s, "SOXX", mk["SOXX_dollar_vol"])
    # 하락 저점 위치 자기 점검: C[l] == bc_lib Ldd
    LL = diag_s["LL"][:, 0]
    okp = LL >= 0
    assert np.allclose(C_s.SOXX.to_numpy()[LL[okp]], fr_s["L60"].SOXX.to_numpy()[okp])
    sanity(rules_s, fr_s, "SOXX")
    soxx_lows_check(rules_s, fr_s, diag_s)
    out["SOXX"] = bc_lib.run_rules(rules_s, "SOXX", FAMILY, fr=fr_s, tag=f"{FAMILY}_SOXX")

    # 종목 82개
    fr_k = bc_lib.stock_frames()
    C_k = bc_lib.load_stocks()[1]
    assert C_k.index.equals(fr_k["in_dd"].index) and list(C_k.columns) == list(fr_k["in_dd"].columns)
    rules_k, diag_k = build_rules(C_k, "STK")
    LLk = diag_k["LL"]
    okk = LLk >= 0
    Ck = C_k.to_numpy(float)
    cols = np.broadcast_to(np.arange(Ck.shape[1]), Ck.shape)
    assert np.allclose(Ck[LLk[okk], cols[okk]], fr_k["L60"].to_numpy(float)[okk])
    sanity(rules_k, fr_k, "STK")
    fv = fr_k["L60"].apply(lambda s: s.first_valid_index())
    print("  자료 공백: 종목 첫 평가 가능일(Ldd 60일 창 완성) →",
          ", ".join(f"{d.date()}×{n}" for d, n in fv.value_counts().sort_index().items()),
          f"| SOXX 달러 거래량 끝 {mk['SOXX_dollar_vol'].last_valid_index().date()}")
    out["STK"] = bc_lib.run_rules(rules_k, "STK", FAMILY, fr=fr_k, tag=f"{FAMILY}_STK")

    for kind, (summ, ver) in out.items():
        print(f"\n[{kind}] 판정")
        print(ver[["rule", "L1", "L2", "main_period", "hold20_n", "hold20", "hold20_base", "near5", "win20", "r20",
                   "up_median"]].round(3).to_string(index=False))
        print(f"[{kind}] 기간별 hold20 (n)")
        pv = summ.pivot(index="rule", columns="period", values="hold20").round(3)
        pn = summ.pivot(index="rule", columns="period", values="n")
        print(pv.to_string())
        print(pn.to_string())


if __name__ == "__main__":
    main()
