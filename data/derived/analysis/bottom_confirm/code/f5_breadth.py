"""bottom_confirm F5 시장 폭 — BRIEF.md(2026-10-26, 결과 전 고정) 갈래 F5, 규칙 7개, SOXX·종목 둘 다.

신호만 만든다(평가·대조·판정은 bc_lib.run_rules). 신호일 t = 그날 종가까지 아는 정보만 쓴다(모든 창은 t 에서 끝남).
시장 폭은 82종목 종가(bc_lib.load_stocks()[1], 2025-06-02~2026-10-23)로 날짜마다 한 번 계산해
SOXX 에는 그 날짜 신호로, 종목(STK)에는 82개 열 모두에 같은 값으로 쓴다. 결측 종가는 채우지 않는다.

  pct20[t]  = (C[t] > SMA20[t]) 종목 수 / SMA20[t] 가 유효한 종목 수     SMA20 = 종가 20일 평균(min_periods=20)
  thrust{l}_{h}  min(pct20[t-9..t]) <= l 이고 pct20[t] >= h            l = 0.15·0.25, h = 0.50·0.60   (4개)
  low20[t]  = (C[t] == min(C[t-19..t])) 종목 수 / 20일 종가 창이 유효한 종목 수
  lows{a}_10     max(low20[t-9..t]) >= a 이고 low20[t] <= 0.10          a = 0.40·0.30                  (2개)
  adv[t]    = (C[t] > C[t-1]) 종목 수 / 두 종가가 모두 있는 종목 수(보합은 상승 아님)
  zweig          E = adv.ewm(span=10, adjust=False).mean(); min(E[t-9..t]) < 0.40 이고 E[t] > 0.615      (1개)
  10일 창(t-9..t)은 min_periods=10(창 값이 모두 있어야 함).
  준비 기간: SMA20·20일 최저는 20번째 세션(2025-06-30)부터, 10일 창까지 더하면 첫 신호 가능일은 2025-07-14.
  zweig 의 EWM 은 2025-06-03 첫 adv 에서 시작하므로 처음 20개 값(초기값 비중 (9/11)^20 ≈ 1.8% 이상)은 NaN 으로 두어
  (pandas min_periods=20, 그 뒤 값은 등록식과 똑같다) 첫 신호 가능일을 2025-07-15 로 맞췄다. 이 처리가 채택 신호를
  바꾸는지 실행 때 점검한다.
  늦게 상장한 종목(CRCL 2025-06-05, SPCX 2026-06-12, SKHY 2026-07-10)은 창이 채워진 날부터 분모에 들어간다(날마다 79~82개).

자료 공백: 종목 자료가 2025-06-02 에 시작하므로 SOXX 신호는 2025-07 중순부터만 있다. SOXX 평가 창(2024-01-02~)의
2024 년과 2025 년 상반기(2024-08-07·2025-04-08 저점 포함)는 시장 폭 신호가 없어 SOXX 판정의 n 조건이 그 때문에 깨질 수 있다.
시장 지표(VIX 등, ~2026-10-13)는 쓰지 않는다. 종목 종가는 2026-10-23 까지라 신호도 그날까지지만 hold20 은 20세션 뒤 종가가 필요하다.

결과 태그: results/f5_SOXX_*.csv, results/f5_STK_*.csv(등록 7개 규칙, 문턱 조정 없음).
사후(post-hoc, 결과 본 뒤 추가, 판정에 쓰지 않음): results/f5_SOXX_posthoc_*.csv, results/f5_STK_posthoc_*.csv
  <rule>_lag1 : 등록 판정에서 L1 을 통과한 규칙의 하루 늦춤(BRIEF 반박 검증 항목). L1 이 없으면 만들지 않는다.
실행: cd data/derived/analysis/bottom_confirm && python3 -W ignore code/f5_breadth.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import bc_lib  # noqa: E402

FAMILY = "f5"
WIN = 10
SMA = 20
EWM_SPAN = 10
EWM_WARM = 20
THRUST = {"thrust15_50": (0.15, 0.50), "thrust15_60": (0.15, 0.60), "thrust25_50": (0.25, 0.50), "thrust25_60": (0.25, 0.60)}
LOWS = {"lows40_10": (0.40, 0.10), "lows30_10": (0.30, 0.10)}
ZWEIG = (0.40, 0.615)
KNOWN_LOWS = ("2024-08-07", "2025-04-08", "2026-03-30", "2026-07-29")


# ---------------------------------------------------------------- 시장 폭 지표
def breadth(C: pd.DataFrame, ewm_warm: int = EWM_WARM) -> pd.DataFrame:
    """C: 세션 × 종목 종가. 반환: 세션 × [pct20, n_sma, low20, n_low, adv, n_adv, E]."""
    sma = C.rolling(SMA, min_periods=SMA).mean()
    v_sma = sma.notna() & C.notna()
    n_sma = v_sma.sum(1)
    pct20 = ((C > sma) & v_sma).sum(1) / n_sma.where(n_sma > 0)

    rmin = C.rolling(SMA, min_periods=SMA).min()
    v_low = rmin.notna() & C.notna()
    n_low = v_low.sum(1)
    low20 = ((C <= rmin) & v_low).sum(1) / n_low.where(n_low > 0)

    prev = C.shift(1)
    v_adv = C.notna() & prev.notna()
    n_adv = v_adv.sum(1)
    adv = ((C > prev) & v_adv).sum(1) / n_adv.where(n_adv > 0)
    E = adv.ewm(span=EWM_SPAN, adjust=False, min_periods=ewm_warm).mean()
    return pd.DataFrame(dict(pct20=pct20, n_sma=n_sma, low20=low20, n_low=n_low, adv=adv, n_adv=n_adv, E=E))


def day_rules(b: pd.DataFrame) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    """반환: (규칙명 → 세션 bool, 진단값 표)."""
    d = b.copy()
    d["pct_min10"] = b.pct20.rolling(WIN, min_periods=WIN).min()
    d["low_max10"] = b.low20.rolling(WIN, min_periods=WIN).max()
    d["E_min10"] = b.E.rolling(WIN, min_periods=WIN).min()
    r = {}
    for n, (lo, hi) in THRUST.items():
        r[n] = (d.pct_min10 <= lo) & (d.pct20 >= hi) & d.pct_min10.notna() & d.pct20.notna()
    for n, (a, z) in LOWS.items():
        r[n] = (d.low_max10 >= a) & (d.low20 <= z) & d.low_max10.notna() & d.low20.notna()
    r["zweig"] = (d.E_min10 < ZWEIG[0]) & (d.E > ZWEIG[1]) & d.E_min10.notna() & d.E.notna()
    return {k: v.fillna(False).astype(bool) for k, v in r.items()}, d


def broadcast(sig: pd.Series, fr: dict) -> pd.DataFrame:
    """날짜 신호를 frames 색인·열 전체로 펼친다(SOXX 는 열 1개, 종목은 82개). 종목 자료 이전 날은 신호 없음."""
    idx, cols = fr["in_dd"].index, fr["in_dd"].columns
    v = sig.reindex(idx).fillna(False).astype(bool).to_numpy()
    return pd.DataFrame(np.repeat(v[:, None], len(cols), axis=1), index=idx, columns=cols)


# ---------------------------------------------------------------- 독립 재계산(반복문)
def independent_check(C: pd.DataFrame, rules: dict[str, pd.Series], b: pd.DataFrame) -> None:
    c = C.to_numpy(float)
    N, K = c.shape
    pct = np.full(N, np.nan)
    low = np.full(N, np.nan)
    adv = np.full(N, np.nan)
    for t in range(N):
        a_n = l_n = a_c = l_c = v_n = v_c = 0
        for k in range(K):
            if t >= SMA - 1:
                w = c[t - SMA + 1:t + 1, k]
                if not np.isnan(w).any():
                    a_n += 1
                    a_c += c[t, k] > w.mean()
                    l_n += 1
                    l_c += c[t, k] <= w.min()
            if t >= 1 and not (np.isnan(c[t, k]) or np.isnan(c[t - 1, k])):
                v_n += 1
                v_c += c[t, k] > c[t - 1, k]
        if a_n:
            pct[t] = a_c / a_n
        if l_n:
            low[t] = l_c / l_n
        if v_n:
            adv[t] = v_c / v_n
    assert np.allclose(pct, b.pct20.to_numpy(), equal_nan=True, atol=1e-12), "pct20"
    assert np.allclose(low, b.low20.to_numpy(), equal_nan=True, atol=1e-12), "low20"
    assert np.allclose(adv, b.adv.to_numpy(), equal_nan=True, atol=1e-12), "adv"
    alpha = 2 / (EWM_SPAN + 1)
    E = np.full(N, np.nan)
    e, cnt = np.nan, 0
    for t in range(N):
        if np.isnan(adv[t]):
            continue
        e = adv[t] if np.isnan(e) else (1 - alpha) * e + alpha * adv[t]
        cnt += 1
        if cnt >= EWM_WARM:
            E[t] = e
    assert np.allclose(E, b.E.to_numpy(), equal_nan=True, atol=1e-12), "E"

    def win_ok(x, t):
        w = x[t - WIN + 1:t + 1]
        return (t >= WIN - 1) and not np.isnan(w).any(), w

    out = {n: np.zeros(N, bool) for n in rules}
    for t in range(N):
        ok, w = win_ok(pct, t)
        if ok:
            for n, (lo, hi) in THRUST.items():
                out[n][t] = w.min() <= lo and pct[t] >= hi
        ok, w = win_ok(low, t)
        if ok:
            for n, (a, z) in LOWS.items():
                out[n][t] = w.max() >= a and low[t] <= z
        ok, w = win_ok(E, t)
        if ok:
            out["zweig"][t] = w.min() < ZWEIG[0] and E[t] > ZWEIG[1]
    for n, s in rules.items():
        assert (s.to_numpy() == out[n]).all(), n
    print(f"독립 재계산(반복문): pct20·low20·adv·E 와 7개 규칙 일치")


# ---------------------------------------------------------------- 점검 출력
def data_report(C: pd.DataFrame, b: pd.DataFrame, d: pd.DataFrame) -> None:
    print("== 종목 자료·시장 폭 범위 ==")
    print(f"  종가 {C.index[0].date()} ~ {C.index[-1].date()}, {C.shape[1]}종목, 날마다 유효 종가 "
          f"{int(C.notna().sum(1).min())}~{int(C.notna().sum(1).max())}개")
    fv = C.apply(lambda s: s.first_valid_index())
    print("  늦게 시작한 종목: " + ", ".join(f"{k} {v.date()}" for k, v in fv[fv > C.index[0]].sort_values().items()))
    for c, n in (("pct20", "n_sma"), ("low20", "n_low"), ("adv", "n_adv"), ("E", None), ("pct_min10", None),
                 ("low_max10", None), ("E_min10", None)):
        s = d[c]
        a = s.first_valid_index()
        extra = f", 분모 {int(d.loc[a:, n].min())}~{int(d.loc[a:, n].max())}" if n else ""
        print(f"  {c:9s} 첫 값 {a.date()}, 마지막 {s.last_valid_index().date()}, 범위 안 결측 {int(s.loc[a:].isna().sum())}{extra}")
    q = d[["pct20", "low20", "adv", "E"]].describe(percentiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]).T
    print(q.round(3).to_string())


def sanity(kind: str, rules: dict[str, pd.DataFrame], fr: dict) -> None:
    idx = fr["in_dd"].index
    ev = (idx >= bc_lib.EVAL_START[kind]) & (idx <= bc_lib.LAST)
    ev_dd = fr["in_dd"] & fr["L60"].notna()
    print(f"\n== {kind}: 규칙별 신호 날 수 (raw = 날짜 수, 종목은 날짜×종목 칸 수도 함께) ==")
    print(f"{'rule':12s} {'days_all':>8s} {'days_eval':>9s} {'cells_eval_in_dd':>16s} {'first':>6s} {'first_dates(SOXX)/n_series(STK)'}")
    for name, sig in rules.items():
        day = sig.iloc[:, 0]
        fs = bc_lib.first_signals(sig, fr, kind)
        cells = int((sig & ev_dd).loc[ev].to_numpy().sum())
        extra = (" ".join(str(x.date()) for x in fs.date) if kind == "SOXX"
                 else f"{fs.series.nunique() if len(fs) else 0} series, dates {fs.date.nunique() if len(fs) else 0}: "
                      + " ".join(f"{x.date()}({k})" for x, k in fs.date.value_counts().sort_index().items()))
        print(f"{name:12s} {int(day.sum()):8d} {int(day.loc[ev].sum()):9d} {cells:16d} {len(fs):6d}  {extra}")
    if kind == "STK":
        first_l = fr["L60"].apply(lambda s: s.first_valid_index())
        print(f"종목 Ldd 첫 유효일: 최소 {first_l.min().date()} / 최대 {first_l.max().date()} "
              f"(종목 자료 2025-06-02 시작, 60거래일 준비)")
        late = first_l[first_l > bc_lib.EVAL_START[kind]]
        if len(late):
            print(f"평가 시작 뒤에야 Ldd 가 생기는 종목 {len(late)}개: " + ", ".join(f"{k}({v.date()})" for k, v in late.sort_values().items()))
    else:
        dd = fr["in_dd"].iloc[:, 0]
        cov = dd.loc[(idx >= bc_lib.EVAL_START[kind]) & (idx <= bc_lib.LAST)]
        before = cov.loc[:"2025-07-14"]
        print(f"SOXX 평가 창 in_dd 날 {int(cov.sum())}개 중 시장 폭 신호가 가능하기 전(~2025-07-14) {int(before.sum())}개 "
              f"(2024: {int(cov.loc['2024'].sum())}, 2025-01~07-14: {int(cov.loc['2025-01-01':'2025-07-14'].sum())}) — 자료 공백")


def decline_low_pos(c: np.ndarray, p: int) -> tuple[int, int]:
    """pk = t 에서 끝나는 60세션 종가 최고의 위치(같으면 늦은 날), l = [pk, t] 최저 종가의 위치(같으면 늦은 날)."""
    w = c[p - 59:p + 1]
    pk = p - 59 + (59 - int(np.argmax(w[::-1])))
    seg = c[pk:p + 1]
    return pk, pk + (len(seg) - 1 - int(np.argmin(seg[::-1])))


def soxx_spotcheck(rules: dict[str, pd.DataFrame], fr: dict, d: pd.DataFrame, C: pd.Series) -> None:
    idx = fr["in_dd"].index
    c = C.to_numpy(float)
    print("\n== SOXX 알려진 저점 주변 점검 (저점 −5 ~ +20 세션) ==")
    for day in KNOWN_LOWS:
        p = idx.get_loc(pd.Timestamp(day))
        pk, l = decline_low_pos(c, p)
        print(f"\n저점 {day}: C={c[p]:.2f}, 60일 고점 {idx[pk].date()} {c[pk]:.2f}, 하락 저점 {idx[l].date()} "
              f"Ldd={fr['L60'].iat[p, 0]:.2f}, dd60={fr['dd60'].iat[p, 0]:+.3f}, in_dd={bool(fr['in_dd'].iat[p, 0])}")
        if pd.Timestamp(day) < d.index[0]:
            print("  종목 자료(2025-06-02~) 이전이라 시장 폭 값 없음 → F5 신호 불가(자료 공백)")
            continue
        for q in range(max(p - 5, 0), min(p + 21, len(idx))):
            t = idx[q]
            g = lambda k: d[k].get(t, np.nan)
            fired = [n for n, s in rules.items() if bool(s.iat[q, 0])]
            print(f"  {t.date()} C={c[q]:7.2f} up={fr['up'].iat[q, 0]:+.3f} in_dd={int(fr['in_dd'].iat[q, 0])} "
                  f"pct20={g('pct20'):.3f}/min10 {g('pct_min10'):.3f}  low20={g('low20'):.3f}/max10 {g('low_max10'):.3f}  "
                  f"adv={g('adv'):.3f} E={g('E'):.3f}/min10 {g('E_min10'):.3f}  fires={','.join(fired) or '-'}")
        firsts = []
        for n, s in rules.items():
            ok = s.iloc[p:, 0] & fr["in_dd"].iloc[p:, 0]
            fd = ok[ok].index[:1]
            if len(fd):
                q = idx.get_loc(fd[0])
                firsts.append(f"{n}:{fd[0].date()}(+{q - p}d, up={fr['up'].iat[q, 0]:+.3f})")
            else:
                firsts.append(f"{n}:-")
        print("  저점 뒤 in_dd 첫 발생: " + "  ".join(firsts))


def print_results(out: dict) -> None:
    cols = ["period", "n", "clusters", "up_median", "hold20", "hold20_base", "hold20_lo", "hold20_dlo",
            "near5", "win20", "r20", "r60"]
    for key, (summ, ver) in out.items():
        print(f"\n== {key} 요약 ==")
        for r in summ.rule.unique():
            t = summ[summ.rule == r]
            print(f"-- {r}")
            print(t[[c for c in cols if c in t.columns]].round(3).to_string(index=False))
        print(f"\n== {key} 판정 ==")
        print(ver.round(3).to_string(index=False))


# ---------------------------------------------------------------- 실행
def main() -> None:
    pd.set_option("display.width", 280)
    _, C = bc_lib.load_stocks()
    assert C.shape[1] == 82, C.shape
    b = breadth(C)
    rules_day, diag = day_rules(b)
    data_report(C, b, diag)
    independent_check(C, rules_day, b)

    # EWM 준비 기간(min_periods=20) 처리가 신호를 바꾸는지: 등록식 그대로(준비 기간 없음)와 비교
    raw, _ = day_rules(breadth(C, ewm_warm=0))
    diff = raw["zweig"] != rules_day["zweig"]
    print(f"zweig EWM 준비 기간 없음과 비교: 다른 날 {int(diff.sum())}개 "
          + (", ".join(str(x.date()) for x in diff.index[diff]) if diff.any() else ""))

    out = {}
    fr = bc_lib.soxx_frames()
    assert C.index.isin(fr["in_dd"].index).all()
    srules = {n: broadcast(s, fr) for n, s in rules_day.items()}
    sanity("SOXX", srules, fr)
    soxx_spotcheck(srules, fr, diag, bc_lib.load_soxx()["C"])
    if diff.any():
        r0 = bc_lib.first_signals(broadcast(raw["zweig"], fr), fr, "SOXX")
        r1 = bc_lib.first_signals(srules["zweig"], fr, "SOXX")
        print(f"zweig SOXX 채택 신호 준비 기간 없음 {list(r0.date.dt.date)} / 있음 {list(r1.date.dt.date)}")
    out["SOXX"] = bc_lib.run_rules(srules, "SOXX", FAMILY, fr=fr, tag=f"{FAMILY}_SOXX")

    sf = bc_lib.stock_frames()
    assert sf["in_dd"].index.equals(C.index) and list(sf["in_dd"].columns) == list(C.columns)
    krules = {n: broadcast(s, sf) for n, s in rules_day.items()}
    for n, s in krules.items():                                   # 82개 열이 모두 같은 날짜 신호
        assert (s.to_numpy() == s.iloc[:, [0]].to_numpy()).all(), n
    sanity("STK", krules, sf)
    if diff.any():
        r0 = bc_lib.first_signals(broadcast(raw["zweig"], sf), sf, "STK")
        r1 = bc_lib.first_signals(krules["zweig"], sf, "STK")
        print(f"zweig STK 채택 신호 수 준비 기간 없음 {len(r0)} / 있음 {len(r1)}, 같음={r0[['date', 'series']].equals(r1[['date', 'series']])}")
    out["STK"] = bc_lib.run_rules(krules, "STK", FAMILY, fr=sf, tag=f"{FAMILY}_STK")

    # ------------------------------------------------------------ 사후(post-hoc) — 등록 판정 아님
    ph_day = {}
    for kind in ("SOXX", "STK"):
        ver = out[kind][1]
        for r in ver.loc[ver.L1, "rule"]:
            ph_day.setdefault(f"{r}_lag1", rules_day[r].shift(1, fill_value=False))
    if ph_day:
        for kind, f in (("SOXX", fr), ("STK", sf)):
            ph = {n: broadcast(s, f) for n, s in ph_day.items()}
            print(f"\n[사후] {kind}: " + ", ".join(ph))
            sanity(kind, ph, f)
            out[f"{kind}_posthoc"] = bc_lib.run_rules(ph, kind, FAMILY + "_posthoc", fr=f, tag=f"{FAMILY}_{kind}_posthoc")
    else:
        print("\n[사후] 등록 판정 L1 통과 규칙 없음 → 하루 늦춤 사후 계산 없음")

    print_results(out)


if __name__ == "__main__":
    main()
