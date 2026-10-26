"""F3 저점 유지 기간 — BRIEF.md(2026-10-26, 결과 전 고정) 갈래 F3, 규칙 6개.

신호만 만든다(평가·대조·판정은 bc_lib.run_rules). 신호일 t = 그날 종가까지 아는 정보만 쓴다.

  하락 저점 위치  pk[t] = t에서 끝나는 60일 종가 최고의 위치(같은 값이면 늦은 날, bc_lib.outcome_frame 과 같음)
                  l[t]  = C[pk..t] 최저 종가의 마지막 위치(C[l] = Ldd[t]).  60일 창에 결측이 있으면 정의 안 함(신호 없음).
  since5/10/15/20 (t - l[t]) >= n 세션                        (Ldd를 찍은 뒤 n거래일 동안 새 저점 없음)
  since5_up3      (t - l[t]) >= 5  이고 up[t] = C[t]/Ldd[t] - 1 >= 3%
  since10_up3     (t - l[t]) >= 10 이고 up[t] >= 3%

  모든 창은 t에서 끝난다. 시장 지표·발행 주식 수 자료를 쓰지 않으므로 하루 늦춤 대상이 없다.
  상태형 규칙(조건이 이어지는 동안 매일 참)이라 실제 채택은 bc_lib 의 in_dd 필터·20거래일 쿨다운이 고른 첫 날이다.

실행: data/derived/analysis/bottom_confirm 에서 `python3 -W ignore code/f3_time.py`
결과: results/f3_SOXX_{summary,signals,verdict}.csv, results/f3_STK_{...}.csv
사후(post-hoc, 결과를 본 뒤 추가, 판정 아님): results/f3_SOXX_posthoc_*.csv — L1을 통과한 SOXX since5_up3 의
  이웃 값(n = 4·5·6 × up = 2·3·4%)과 하루 늦춤.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import bc_lib  # noqa: E402

FAMILY = "f3"
KNOWN_LOWS = ("2024-08-07", "2025-04-08", "2026-03-30", "2026-07-29")
SINCE_N = (5, 10, 15, 20)
UP_MIN = 0.03


# ---------------------------------------------------------------- 하락 저점 위치
def decline_low_pos(C: pd.DataFrame) -> np.ndarray:
    """l[t, k] = C[pk..t] 최저 종가의 마지막 위치. pk = t에서 끝나는 60일 종가 최고(같으면 늦은 날).
    60일 창에 결측이 있으면 -1."""
    Cv = C.to_numpy(float)
    N, K = Cv.shape
    LL = np.full((N, K), -1, int)
    for k in range(K):
        c = Cv[:, k]
        for t in range(59, N):
            w = c[t - 59:t + 1]
            if np.isnan(w).any():
                continue
            pk = t - 59 + (59 - int(np.argmax(w[::-1])))            # 최고값의 가장 늦은 위치
            seg = c[pk:t + 1]
            LL[t, k] = pk + (len(seg) - 1 - int(np.argmin(seg[::-1])))  # 최저값의 가장 늦은 위치
    return LL


def since_low(C: pd.DataFrame) -> pd.DataFrame:
    """t - l[t] (정의 안 되면 NaN)."""
    LL = decline_low_pos(C)
    N = LL.shape[0]
    t = np.arange(N)[:, None]
    s = np.where(LL >= 0, t - LL, np.nan).astype(float)
    return pd.DataFrame(s, index=C.index, columns=C.columns)


def since_low_check(C: pd.DataFrame, Ldd: pd.DataFrame) -> pd.DataFrame:
    """독립 재계산: bc_lib 의 Ldd 값을 받아, t 이전(포함) 종가 중 Ldd 와 같은 가장 늦은 날까지의 거리.
    l 은 [pk, t] 안의 마지막 최저이고 pk 이후 다른 날은 Ldd 보다 크거나(같으면 l 이 더 늦다) 하므로 같은 값이 나와야 한다."""
    Cv, Lv = C.to_numpy(float), Ldd.to_numpy(float)
    N, K = Cv.shape
    out = np.full((N, K), np.nan)
    for k in range(K):
        for t in range(N):
            if np.isnan(Lv[t, k]):
                continue
            a = max(t - 59, 0)
            hit = np.flatnonzero(Cv[a:t + 1, k] == Lv[t, k])
            out[t, k] = t - (a + hit[-1])
    return pd.DataFrame(out, index=C.index, columns=C.columns)


# ---------------------------------------------------------------- 규칙
def build_rules(C: pd.DataFrame, up: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    s = since_low(C)
    up = up.reindex(index=C.index, columns=C.columns)
    rules = {f"since{n}": s >= n for n in SINCE_N}
    rules["since5_up3"] = (s >= 5) & (up >= UP_MIN)
    rules["since10_up3"] = (s >= 10) & (up >= UP_MIN)
    return {k: v.fillna(False).astype(bool) for k, v in rules.items()}, s


# ---------------------------------------------------------------- 점검
def sanity(rules: dict, fr: dict, kind: str):
    print(f"\n[{kind}] 규칙별 원 신호일 수(전체 / in_dd·평가 창 안) · 첫 신호 수")
    for name, sig in rules.items():
        s = sig.reindex(index=fr["in_dd"].index, columns=fr["in_dd"].columns).fillna(False)
        raw = int(s.to_numpy().sum())
        win = s & fr["in_dd"] & fr["L60"].notna()
        win = win.loc[(win.index >= bc_lib.EVAL_START[kind]) & (win.index <= bc_lib.LAST)]
        fs = bc_lib.first_signals(sig, fr, kind)
        print(f"  {name:12s} raw={raw:6d}  in_dd·window={int(win.to_numpy().sum()):6d}  first={len(fs):5d}"
              + (f"  first_sig={fs.date.min().date()} last_sig={fs.date.max().date()}" if len(fs) else ""))


def soxx_lows_check(rules: dict, fr: dict, since: pd.DataFrame):
    C = bc_lib.load_soxx()["C"]
    idx = C.index
    firsts = {n: bc_lib.first_signals(s, fr, "SOXX") for n, s in rules.items()}
    print("\n[SOXX] 알려진 저점 주변: 저점 -3 ~ +40 세션 안 첫 원 신호와 그 구간 첫 채택 신호(F)")
    for low in KNOWN_LOWS:
        p = idx.get_loc(pd.Timestamp(low))
        q5 = min(p + 5, len(idx) - 1)
        l5 = q5 - int(since.iat[q5, 0])
        print(f"  저점 {low}: C={C.iloc[p]:.2f}, Ldd(t+5)={fr['L60'].iat[q5, 0]:.2f} at {idx[l5].date()}, "
              f"in_dd={bool(fr['in_dd'].iat[p, 0])}, since(t)={since.iat[p, 0]:.0f}")
        a, b = max(p - 3, 0), min(p + 40, len(idx) - 1)
        for name, s in rules.items():
            raw = [q for q in range(a, b + 1) if s.iat[q, 0]]
            fsd = firsts[name]
            fsd = fsd[(fsd.pos >= a) & (fsd.pos <= b)]
            txt = "-"
            if raw:
                q = raw[0]
                txt = (f"raw {idx[q].strftime('%Y-%m-%d')}(+{q - p}, since={since.iat[q, 0]:.0f}, "
                       f"up={fr['up'].iat[q, 0] * 100:.1f}%)")
            if len(fsd):
                q = int(fsd.pos.iloc[0])
                h = fr["hold20"].iat[q, 0]
                txt += (f"  F {idx[q].strftime('%Y-%m-%d')}(+{q - p}, up={fr['up'].iat[q, 0] * 100:.1f}%, "
                        f"hold20={'NA' if np.isnan(h) else int(h)})")
            print(f"    {name:12s} {txt}")


def posthoc_soxx(C: pd.DataFrame, fr: dict):
    """사후(post-hoc, 결과를 본 뒤 추가): SOXX에서 L1을 통과한 since5_up3 의 이웃 값·하루 늦춤 민감도.
    등록 판정에 쓰지 않는다. 결과: results/f3_SOXX_posthoc_*.csv"""
    s = since_low(C)
    up = fr["up"].reindex(index=C.index, columns=C.columns)
    rules = {}
    for n in (4, 5, 6):
        for u in (0.02, 0.03, 0.04):
            rules[f"since{n}_up{int(u * 100)}"] = ((s >= n) & (up >= u)).fillna(False).astype(bool)
    rules["since5_up3_lag1"] = rules["since5_up3"].shift(1).fillna(False).astype(bool)
    summ, ver = bc_lib.run_rules(rules, "SOXX", FAMILY, fr=fr, tag=f"{FAMILY}_SOXX_posthoc")
    print("\n[SOXX 사후] since5_up3 이웃 값·하루 늦춤(post-hoc, 판정 아님)")
    p = summ.pivot(index="rule", columns="period", values="hold20").round(3)
    n = summ.pivot(index="rule", columns="period", values="hold20_n")
    print(ver[["rule", "L1", "L2", "hold20_n", "hold20", "hold20_base", "near5", "win20", "r20", "up_median"]]
          .round(3).to_string(index=False))
    print(p[["2024~2026", "2026", "2025"]].astype(str).add(" n=").add(n[["2024~2026", "2026", "2025"]].astype(str))
          .to_string())


def main():
    pd.set_option("display.width", 220)
    out = {}

    # SOXX
    fr_s = bc_lib.soxx_frames()
    C_s = bc_lib.load_soxx()[["C"]].rename(columns={"C": "SOXX"})
    assert C_s.index.equals(fr_s["in_dd"].index)
    rules_s, since_s = build_rules(C_s, fr_s["up"])
    # 자기 점검 1: C[l] == bc_lib Ldd, 2: 독립 재계산과 같음
    ok = since_s.notna().to_numpy()
    lpos = (np.arange(len(C_s))[:, None] - np.nan_to_num(since_s.to_numpy()).astype(int))
    assert np.allclose(C_s.to_numpy()[lpos[ok], 0], fr_s["L60"].to_numpy()[ok])
    chk = since_low_check(C_s, fr_s["L60"])
    assert (since_s.isna() == chk.isna()).all().all() and np.allclose(since_s.to_numpy()[ok], chk.to_numpy()[ok])
    print("[SOXX] since 자기 점검 통과(C[l]=Ldd, 독립 재계산 일치), 정의 시작일",
          since_s.SOXX.first_valid_index().date())
    sanity(rules_s, fr_s, "SOXX")
    soxx_lows_check(rules_s, fr_s, since_s)
    out["SOXX"] = bc_lib.run_rules(rules_s, "SOXX", FAMILY, fr=fr_s, tag=f"{FAMILY}_SOXX")
    posthoc_soxx(C_s, fr_s)

    # 종목 82개
    fr_k = bc_lib.stock_frames()
    C_k = bc_lib.load_stocks()[1]
    assert C_k.index.equals(fr_k["in_dd"].index) and list(C_k.columns) == list(fr_k["in_dd"].columns)
    rules_k, since_k = build_rules(C_k, fr_k["up"])
    okk = since_k.notna().to_numpy()
    lk = np.arange(len(C_k))[:, None] - np.nan_to_num(since_k.to_numpy()).astype(int)
    cols = np.broadcast_to(np.arange(C_k.shape[1]), C_k.shape)
    assert np.allclose(C_k.to_numpy(float)[lk[okk], cols[okk]], fr_k["L60"].to_numpy(float)[okk])
    chk_k = since_low_check(C_k, fr_k["L60"])
    assert (since_k.isna() == chk_k.isna()).all().all() and np.allclose(since_k.to_numpy()[okk], chk_k.to_numpy()[okk])
    print("\n[STK] since 자기 점검 통과(C[l]=Ldd, 독립 재계산 일치)")
    sanity(rules_k, fr_k, "STK")
    fv = since_k.apply(lambda s: s.first_valid_index())
    print("  자료 공백: 종목 첫 since 정의일(60일 창 완성) →",
          ", ".join(f"{d.date()}×{n}" for d, n in fv.value_counts().sort_index().items()),
          "| 평가 시작 2025-09-01")
    out["STK"] = bc_lib.run_rules(rules_k, "STK", FAMILY, fr=fr_k, tag=f"{FAMILY}_STK")

    for kind, (summ, ver) in out.items():
        print(f"\n[{kind}] 판정")
        print(ver[["rule", "L1", "L2", "main_period", "hold20_n", "hold20", "hold20_base", "near5", "win20", "r20",
                   "up_median"]].round(3).to_string(index=False))
        print(f"[{kind}] 기간별 hold20 / n")
        print(summ.pivot(index="rule", columns="period", values="hold20").round(3).to_string())
        print(summ.pivot(index="rule", columns="period", values="n").to_string())
        print(f"[{kind}] 주 기간 대조 차이 hold20_diff [dlo, dhi]")
        main_p = "2024~2026" if kind == "SOXX" else "2026"
        m = summ[summ.period == main_p].set_index("rule")
        print(m[["hold20_diff", "hold20_dlo", "hold20_dhi", "clusters"]].round(3).to_string())


if __name__ == "__main__":
    main()
