"""bottom_confirm F4 변동성 — BRIEF.md(2026-10-26, 결과 전 고정) 갈래 F4, 규칙 9개, SOXX·종목 둘 다.

신호만 만든다(평가·대조·판정은 bc_lib.run_rules). 신호일 t = 그날 종가까지 아는 정보만 쓴다.
시장 전체 신호라 날짜마다 한 번 계산하고 종목(STK)에는 82개 열 모두에 같은 값을 쓴다.
자료: bc_lib.load_market() 의 '^VIX'·'^VIX3M'·'^VXN'(us_prices_close.csv, 2023-01-03~)과 'VVIX'(us_daily_panel.csv, 2023-06-01~).
네 지표 모두 2026-10-13 이 마지막이고 그 뒤는 NaN = 신호 없음. 창(10거래일, t-9..t)에 결측이 하나라도 있으면 신호 없음
(min_periods=10; 자료 범위 안에는 결측이 없어 범위 끝에서만 차이가 난다).
VIX 계열 지수 종가는 16:15 ET 에 확정되지만 진입은 t+1 시가라 '그날 장 마감 뒤 알 수 있는 값'에 해당하고 하루 늦춤 대상이 아니다
(발행 주식 수·자금 흐름 자료가 아님).

  vix{x}_{y}   Mx = max(^VIX[t-9..t]); Mx >= x 이고 ^VIX[t] <= (1-y)·Mx.   x = 25·30·35, y = 15·25(%)  (6개)
  ts_norm      R = ^VIX/^VIX3M; max(R[t-9..t]) >= 1.0 이고 R[t] < 0.95
  vxn35_20     max(^VXN[t-9..t]) >= 35 이고 ^VXN[t] <= 0.8·max
  vvix120_15   max(VVIX[t-9..t]) >= 120 이고 VVIX[t] <= 0.85·max

결과 태그: results/f4_SOXX_*.csv, results/f4_STK_*.csv(등록 9개 규칙, 문턱 조정 없음).
사후(post-hoc, 결과 본 뒤 추가, 판정에 쓰지 않음): results/f4_SOXX_posthoc_*.csv, results/f4_STK_posthoc_*.csv
  ts_norm_panelvix : ^VIX 2026-02-06 값(20.37)이 같은 날 ^VIX3M(20.37)과 똑같아 R = 1.000 이 된다. 같은 날 패널 'VIX'는 17.76
                     (나머지 844일은 ^VIX 와 오차 < 1e-5). 이 하루를 패널 VIX 로 바꾼 ts_norm(자료 오류 민감도).
  <rule>_lag1      : 등록 판정에서 L1 을 통과한 규칙의 하루 늦춤(BRIEF 반박 검증 항목). L1 이 없으면 만들지 않는다.
실행: cd data/derived/analysis/bottom_confirm && python3 -W ignore code/f4_vol.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import bc_lib  # noqa: E402

FAMILY = "f4"
WIN = 10
VIX_LEVELS = (25, 30, 35)
VIX_DROPS = (0.15, 0.25)
KNOWN_LOWS = ("2024-08-07", "2025-04-08", "2026-03-30", "2026-07-29")
COLS = ("^VIX", "^VIX3M", "^VXN", "VVIX")


# ---------------------------------------------------------------- 신호(날짜 하나짜리 시장 신호)
def peak_fade(x: pd.Series, level: float, drop: float) -> tuple[pd.Series, pd.Series]:
    """max(x[t-9..t]) >= level 이고 x[t] <= (1-drop)·max. 반환 (신호, 10일 최고)."""
    mx = x.rolling(WIN, min_periods=WIN).max()
    sig = (mx >= level) & (x <= (1 - drop) * mx) & mx.notna() & x.notna()
    return sig, mx


def term_norm(vix: pd.Series, vix3m: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """R = VIX/VIX3M; max(R[t-9..t]) >= 1.0 이고 R[t] < 0.95. 반환 (신호, R, 10일 최고 R)."""
    r = vix / vix3m
    mx = r.rolling(WIN, min_periods=WIN).max()
    sig = (mx >= 1.0) & (r < 0.95) & mx.notna() & r.notna()
    return sig, r, mx


def market_rules(m: pd.DataFrame) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    """반환: (규칙명 → 세션 bool 시리즈, 진단값)."""
    rules, diag = {}, {}
    vix = m["^VIX"]
    for x in VIX_LEVELS:
        for y in VIX_DROPS:
            s, mx = peak_fade(vix, x, y)
            rules[f"vix{x}_{int(round(y * 100))}"] = s
    diag["VIX"], diag["VIXmax10"] = vix, vix.rolling(WIN, min_periods=WIN).max()
    s, r, rmx = term_norm(vix, m["^VIX3M"])
    rules["ts_norm"] = s
    diag["R"], diag["Rmax10"] = r, rmx
    s, mx = peak_fade(m["^VXN"], 35, 0.20)
    rules["vxn35_20"] = s
    diag["VXN"], diag["VXNmax10"] = m["^VXN"], mx
    s, mx = peak_fade(m["VVIX"], 120, 0.15)
    rules["vvix120_15"] = s
    diag["VVIX"], diag["VVIXmax10"] = m["VVIX"], mx
    return {k: v.fillna(False).astype(bool) for k, v in rules.items()}, diag


def broadcast(sig: pd.Series, fr: dict) -> pd.DataFrame:
    """날짜 신호를 frames 색인·열 전체로 펼친다(SOXX 는 열 1개, 종목은 82개)."""
    idx, cols = fr["in_dd"].index, fr["in_dd"].columns
    v = sig.reindex(idx).fillna(False).astype(bool).to_numpy()
    return pd.DataFrame(np.repeat(v[:, None], len(cols), axis=1), index=idx, columns=cols)


# ---------------------------------------------------------------- 독립 재계산(반복문)
def independent_check(m: pd.DataFrame, rules: dict[str, pd.Series]) -> None:
    idx = m.index
    N = len(idx)

    def loop_peak(a: np.ndarray, level: float, drop: float) -> np.ndarray:
        out = np.zeros(N, bool)
        for t in range(WIN - 1, N):
            w = a[t - WIN + 1:t + 1]
            if np.isnan(w).any():
                continue
            mx = w.max()
            out[t] = mx >= level and a[t] <= (1 - drop) * mx
        return out

    vix, v3, vxn, vv = (m[c].to_numpy(float) for c in COLS)
    for x in VIX_LEVELS:
        for y in VIX_DROPS:
            n = f"vix{x}_{int(round(y * 100))}"
            assert (rules[n].to_numpy() == loop_peak(vix, x, y)).all(), n
    r = vix / v3
    ts = np.zeros(N, bool)
    for t in range(WIN - 1, N):
        w = r[t - WIN + 1:t + 1]
        if np.isnan(w).any():
            continue
        ts[t] = w.max() >= 1.0 and r[t] < 0.95
    assert (rules["ts_norm"].to_numpy() == ts).all(), "ts_norm"
    assert (rules["vxn35_20"].to_numpy() == loop_peak(vxn, 35, 0.20)).all(), "vxn35_20"
    assert (rules["vvix120_15"].to_numpy() == loop_peak(vv, 120, 0.15)).all(), "vvix120_15"
    # 마지막 자료일 뒤에는 신호가 없어야 한다
    last = m["^VIX"].last_valid_index()
    for n, s in rules.items():
        assert not s.loc[last + pd.Timedelta(days=1):].any(), f"{n}: {last.date()} 뒤 신호"
    print(f"독립 재계산(반복문) 9개 규칙 일치, {last.date()} 뒤 신호 없음")


# ---------------------------------------------------------------- 점검 출력
def data_report(m: pd.DataFrame) -> None:
    print("== 시장 지표 범위 ==")
    for c in COLS:
        s = m[c]
        a, b = s.first_valid_index(), s.last_valid_index()
        gaps = int(s.loc[a:b].isna().sum())
        print(f"  {c:7s} {a.date()} ~ {b.date()}  범위 안 결측 {gaps}  세션 끝(2026-10-23)까지 결측 {int(s.loc[b:].isna().sum())}")
    pv = m["VIX"]
    d = (pv - m["^VIX"]).abs()
    bad = d[d > 0.05]
    print(f"  ^VIX vs 패널 VIX(겹침 {int(d.notna().sum())}일): 차이 > 0.05 인 날 {len(bad)}개 "
          + ", ".join(f"{k.date()} ^VIX={m.at[k, '^VIX']:.2f} 패널={pv[k]:.2f} ^VIX3M={m.at[k, '^VIX3M']:.2f}" for k in bad.index))


def sanity(kind: str, rules: dict[str, pd.DataFrame], fr: dict) -> None:
    idx = fr["in_dd"].index
    ev = (idx >= bc_lib.EVAL_START[kind]) & (idx <= bc_lib.LAST)
    ev_dd = fr["in_dd"] & fr["L60"].notna()
    print(f"\n== {kind}: 규칙별 신호 날 수 (raw = 날짜 수, 종목은 날짜×종목 칸 수도 함께) ==")
    print(f"{'rule':11s} {'days_all':>8s} {'days_eval':>9s} {'cells_eval_in_dd':>16s} {'first':>6s} {'first_dates(SOXX)/n_series(STK)'}")
    for name, sig in rules.items():
        day = sig.iloc[:, 0]
        fs = bc_lib.first_signals(sig, fr, kind)
        cells = int((sig & ev_dd).loc[ev].to_numpy().sum())
        extra = (" ".join(str(d.date()) for d in fs.date) if kind == "SOXX"
                 else f"{fs.series.nunique() if len(fs) else 0} series, dates {fs.date.nunique() if len(fs) else 0}")
        print(f"{name:11s} {int(day.sum()):8d} {int(day.loc[ev].sum()):9d} {cells:16d} {len(fs):6d}  {extra}")
    if kind == "STK":
        first_l = fr["L60"].apply(lambda s: s.first_valid_index())
        print(f"종목 Ldd 첫 유효일: 최소 {first_l.min().date()} / 최대 {first_l.max().date()} "
              f"(종목 자료 2025-06-02 시작, 60거래일 준비; 시장 신호 자체는 2023 년부터 계산해 준비 기간 없음)")
        late = first_l[first_l > bc_lib.EVAL_START[kind]]
        if len(late):
            print(f"평가 시작 뒤에야 Ldd 가 생기는 종목 {len(late)}개: " + ", ".join(f"{k}({v.date()})" for k, v in late.sort_values().items()))


def soxx_spotcheck(rules: dict[str, pd.DataFrame], fr: dict, diag: dict, C: pd.Series) -> None:
    idx = fr["in_dd"].index
    c = C.to_numpy(float)
    print("\n== SOXX 알려진 저점 주변 점검 (저점 −3 ~ +20 세션) ==")
    for d in KNOWN_LOWS:
        p = idx.get_loc(pd.Timestamp(d))
        w = c[p - 59:p + 1]
        pk = p - 59 + (59 - int(np.argmax(w[::-1])))
        seg = c[pk:p + 1]
        l = pk + (len(seg) - 1 - int(np.argmin(seg[::-1])))
        print(f"\n저점 {d}: C={c[p]:.2f}, 60일 고점 {idx[pk].date()} {c[pk]:.2f}, 하락 저점 {idx[l].date()} "
              f"Ldd={fr['L60'].iat[p, 0]:.2f}, dd60={fr['dd60'].iat[p, 0]:+.3f}, in_dd={bool(fr['in_dd'].iat[p, 0])}")
        for q in range(max(p - 3, 0), min(p + 21, len(idx))):
            t = idx[q]
            g = lambda k: diag[k].get(t, np.nan)
            fired = [n for n, s in rules.items() if bool(s.iat[q, 0])]
            print(f"  {t.date()} C={c[q]:7.2f} up={fr['up'].iat[q, 0]:+.3f} in_dd={int(fr['in_dd'].iat[q, 0])} "
                  f"VIX={g('VIX'):6.2f}/mx{g('VIXmax10'):6.2f} R={g('R'):.3f}/mx{g('Rmax10'):.3f} "
                  f"VXN={g('VXN'):6.2f}/mx{g('VXNmax10'):6.2f} VVIX={g('VVIX'):6.1f}/mx{g('VVIXmax10'):6.1f} "
                  f"fires={','.join(fired) or '-'}")
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
    pd.set_option("display.width", 260)
    m = bc_lib.load_market()
    data_report(m)
    day_rules, diag = market_rules(m)
    independent_check(m, day_rules)

    out = {}
    fr = bc_lib.soxx_frames()
    assert fr["in_dd"].index.equals(m.index)
    srules = {n: broadcast(s, fr) for n, s in day_rules.items()}
    sanity("SOXX", srules, fr)
    soxx_spotcheck(srules, fr, diag, bc_lib.load_soxx()["C"])
    out["SOXX"] = bc_lib.run_rules(srules, "SOXX", FAMILY, fr=fr, tag=f"{FAMILY}_SOXX")

    sf = bc_lib.stock_frames()
    assert len(sf["in_dd"].columns) == 82, len(sf["in_dd"].columns)
    assert sf["in_dd"].index.isin(m.index).all()
    krules = {n: broadcast(s, sf) for n, s in day_rules.items()}
    for n, s in krules.items():                                   # 82개 열이 모두 같은 날짜 신호
        assert (s.to_numpy() == s.iloc[:, [0]].to_numpy()).all(), n
    sanity("STK", krules, sf)
    out["STK"] = bc_lib.run_rules(krules, "STK", FAMILY, fr=sf, tag=f"{FAMILY}_STK")

    # ------------------------------------------------------------ 사후(post-hoc) — 등록 판정 아님
    vix_fix = m["^VIX"].copy()
    bad = (m["VIX"] - m["^VIX"]).abs() > 0.05
    vix_fix[bad] = m.loc[bad, "VIX"]
    ts_fix, _, _ = term_norm(vix_fix, m["^VIX3M"])
    ts_fix = ts_fix.fillna(False).astype(bool)
    print(f"\n[사후] ^VIX 를 패널 VIX 로 바꾼 날 {int(bad.sum())}개; ts_norm 이 달라지는 날: "
          + (", ".join(str(d.date()) for d in ts_fix.index[ts_fix != day_rules["ts_norm"]]) or "없음"))
    ph_day = {"ts_norm_panelvix": ts_fix}
    for kind in ("SOXX", "STK"):
        ver = out[kind][1]
        for r in ver.loc[ver.L1, "rule"]:
            ph_day.setdefault(f"{r}_lag1", day_rules[r].shift(1, fill_value=False))
    for kind, f in (("SOXX", fr), ("STK", sf)):
        ph = {n: broadcast(s, f) for n, s in ph_day.items()}
        print(f"\n[사후] {kind}")
        sanity(kind, ph, f)
        out[f"{kind}_posthoc"] = bc_lib.run_rules(ph, kind, FAMILY + "_posthoc", fr=f, tag=f"{FAMILY}_{kind}_posthoc")

    print_results(out)


if __name__ == "__main__":
    main()
