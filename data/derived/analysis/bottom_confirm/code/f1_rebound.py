"""bottom_confirm F1 반등 폭 — BRIEF.md 사전 등록 갈래 F1(10개 규칙), SOXX·종목 둘 다.

규칙(신호일 t 종가까지 아는 값만 씀. 진입·결과·대조·판정은 bc_lib.run_rules)
  up3·up5·up8·up10·up15·up20 : up[t] = C[t]/Ldd[t] − 1 ≥ 3·5·8·10·15·20%
  sig2·sig3·sig4·sig6        : up[t] ≥ m·σ20[t], m = 2·3·4·6
                               σ20[t] = t에서 끝나는 20거래일 일간 종가 수익률(C[s]/C[s−1] − 1) 표준편차(표본, ddof=1)
  Ldd·up 은 bc_lib.outcome_frame 의 fr['up'](60일 고점(같은 값이면 늦은 날) 이후 최저 종가 대비)을 그대로 쓴다.
  σ20 은 각 계열 자기 종가로 계산한다: SOXX = bc_lib.load_soxx()['C'], 종목 = bc_lib.load_stocks()[1].
  결측 종가는 채우지 않는다(pct_change 의 앞값 채우기 대신 C/C.shift(1) − 1, 창 안에 결측이 있으면 σ20 = NaN → 신호 없음).

결과 태그: results/f1_SOXX_*.csv, results/f1_STK_*.csv(등록 10개 규칙, 문턱 조정 없음).
사후(post-hoc, 결과 본 뒤 추가, 판정에 쓰지 않음): results/f1_SOXX_posthoc_*.csv, results/f1_STK_posthoc_*.csv
  = 등록 L1 후보의 하루 늦춤(<rule>_lag1), 종목 sig6 의 up 수준 대조(up267_ctrl: up ≥ 0.267, 결과에서 고른 문턱).
실행: cd data/derived/analysis/bottom_confirm && python3 -W ignore code/f1_rebound.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import bc_lib  # noqa: E402

FAMILY = "f1"
UP_LEVELS = {"up3": 0.03, "up5": 0.05, "up8": 0.08, "up10": 0.10, "up15": 0.15, "up20": 0.20}
SIG_MULTS = {"sig2": 2, "sig3": 3, "sig4": 4, "sig6": 6}
KNOWN_LOWS = ["2024-08-07", "2025-04-08", "2026-03-30", "2026-07-29"]


def sigma20(C: pd.DataFrame) -> pd.DataFrame:
    """t에서 끝나는 20거래일 일간 종가 수익률 표준편차. 결측은 채우지 않는다."""
    ret = C / C.shift(1) - 1
    return ret.rolling(20, min_periods=20).std()


def build_rules(fr: dict, C: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """반환: (규칙명 → 세션 × 계열 bool, σ20)."""
    up = fr["up"]
    C = C.reindex(index=up.index, columns=up.columns)
    s20 = sigma20(C)
    rules = {}
    for name, x in UP_LEVELS.items():
        rules[name] = (up >= x) & up.notna()
    for name, m in SIG_MULTS.items():
        rules[name] = (up >= m * s20) & up.notna() & s20.notna()
    return {k: v.fillna(False).astype(bool) for k, v in rules.items()}, s20


def decline_low_pos(c: np.ndarray, t: int) -> tuple[int, int]:
    """(pk, l): 60일 종가 고점 위치(같으면 늦은 날), [pk, t] 최저 종가 위치(같으면 늦은 날)."""
    w = c[t - 59:t + 1]
    pk = t - 59 + (59 - int(np.argmax(w[::-1])))
    seg = c[pk:t + 1]
    l = pk + (len(seg) - 1 - int(np.argmin(seg[::-1])))
    return pk, l


def sanity(kind: str, rules: dict, fr: dict, C: pd.DataFrame, s20: pd.DataFrame) -> None:
    ev = (fr["in_dd"].index >= bc_lib.EVAL_START[kind]) & (fr["in_dd"].index <= bc_lib.LAST)
    print(f"\n== {kind}: 규칙별 신호 날 수 ==")
    print(f"{'rule':6s} {'raw_all':>8s} {'raw_eval':>8s} {'raw_in_dd':>9s} {'first':>6s}")
    for name, sig in rules.items():
        fs = bc_lib.first_signals(sig, fr, kind)
        raw_all = int(sig.to_numpy().sum())
        raw_eval = int(sig.loc[ev].to_numpy().sum())
        raw_dd = int((sig & fr["in_dd"]).loc[ev].to_numpy().sum())
        print(f"{name:6s} {raw_all:8d} {raw_eval:8d} {raw_dd:9d} {len(fs):6d}")
    # 결측·준비 기간
    first_up = fr["up"].apply(lambda s: s.first_valid_index())
    first_s20 = s20.apply(lambda s: s.first_valid_index())
    print(f"첫 up 유효일: 최소 {first_up.min().date()} / 최대 {first_up.max().date()}; "
          f"첫 σ20 유효일: 최소 {first_s20.min().date()} / 최대 {first_s20.max().date()}")
    late = first_up[first_up > bc_lib.EVAL_START[kind]]
    if len(late):
        print(f"평가 시작({bc_lib.EVAL_START[kind].date()}) 뒤에야 up 이 생기는 계열 {len(late)}개: "
              + ", ".join(f"{k}({v.date()})" for k, v in late.sort_values().items()))
    nan_c = C.loc[ev].isna().sum()
    nan_c = nan_c[nan_c > 0]
    if len(nan_c):
        print(f"평가 창 안 종가 결측 계열 {len(nan_c)}개: " + ", ".join(f"{k}={v}" for k, v in nan_c.items()))


def soxx_spotcheck(rules: dict, fr: dict, C: pd.Series, s20: pd.Series) -> None:
    idx = fr["up"].index
    c = C.to_numpy(float)
    print("\n== SOXX 알려진 저점 주변 점검 ==")
    for d in KNOWN_LOWS:
        p = idx.get_loc(pd.Timestamp(d))
        pk, l = decline_low_pos(c, p)
        print(f"\n저점 {d}: C={c[p]:.2f}, 60일 고점 {idx[pk].date()} {c[pk]:.2f}, 하락 저점 {idx[l].date()} "
              f"Ldd={fr['L60'].iat[p, 0]:.2f}, in_dd={bool(fr['in_dd'].iat[p, 0])}, dd60={fr['dd60'].iat[p, 0]:+.3f}")
        rows = []
        for q in range(p, min(p + 25, len(idx))):
            pk2, l2 = decline_low_pos(c, q)
            fired = [n for n, s in rules.items() if bool(s.iat[q, 0])]
            rows.append(f"  {idx[q].date()} C={c[q]:7.2f} Ldd={fr['L60'].iat[q, 0]:7.2f}(@{idx[l2].date()}) "
                        f"up={fr['up'].iat[q, 0]:+.3f} σ20={s20.iat[q]:.4f} in_dd={int(fr['in_dd'].iat[q, 0])} "
                        f"fires={','.join(fired) or '-'}")
        print("\n".join(rows))
        # 각 규칙의 저점 뒤 첫 신호일(in_dd 안)
        firsts = []
        for n, s in rules.items():
            ok = (s.iloc[p:, 0] & fr["in_dd"].iloc[p:, 0])
            fd = ok[ok].index[:1]
            if len(fd):
                q = idx.get_loc(fd[0])
                firsts.append(f"{n}:{fd[0].date()}(+{q - p}d, up={fr['up'].iat[q, 0]:+.3f})")
            else:
                firsts.append(f"{n}:-")
        print("  저점 뒤 in_dd 첫 발생: " + "  ".join(firsts))


def independent_check(fr: dict, C: pd.DataFrame, rules: dict, kind: str) -> None:
    """up 을 decline_low_pos 로 다시 계산해 bc_lib 과 일치하는지, σ20 을 numpy 로 다시 계산해 일치하는지."""
    Cv = C.to_numpy(float)
    N, K = Cv.shape
    up2 = np.full((N, K), np.nan)
    s2 = np.full((N, K), np.nan)
    for k in range(K):
        c = Cv[:, k]
        for t in range(59, N):
            w = c[t - 59:t + 1]
            if np.isnan(w).any():
                continue
            pk, l = decline_low_pos(c, t)
            up2[t, k] = c[t] / c[l] - 1
        for t in range(20, N):
            w = c[t - 20:t + 1]
            if np.isnan(w).any():
                continue
            s2[t, k] = np.std(w[1:] / w[:-1] - 1, ddof=1)
    up1 = fr["up"].to_numpy(float)
    both = np.isfinite(up1) & np.isfinite(up2)
    assert (np.isfinite(up1) == np.isfinite(up2)).all(), "up 유효 위치 불일치"
    assert np.allclose(up1[both], up2[both], atol=1e-12), "up 재계산 불일치"
    s1 = sigma20(C).to_numpy(float)
    bs = np.isfinite(s1) & np.isfinite(s2)
    assert (np.isfinite(s1) == np.isfinite(s2)).all(), "σ20 유효 위치 불일치"
    assert np.allclose(s1[bs], s2[bs], atol=1e-12), "σ20 재계산 불일치"
    # 신호 재계산
    for n, x in UP_LEVELS.items():
        r2 = np.where(np.isfinite(up2), up2 >= x, False)
        assert (rules[n].to_numpy() == r2).all(), f"{n} 재계산 불일치"
    for n, m in SIG_MULTS.items():
        r2 = np.where(np.isfinite(up2) & np.isfinite(s2), up2 >= m * s2, False)
        assert (rules[n].to_numpy() == r2).all(), f"{n} 재계산 불일치"
    print(f"[{kind}] 독립 재계산(up·σ20·신호 10개) 일치")


def main() -> None:
    pd.set_option("display.width", 240)
    out = {}
    # SOXX
    fr = bc_lib.soxx_frames()
    C = bc_lib.load_soxx()[["C"]].rename(columns={"C": "SOXX"})
    assert C.index.equals(fr["up"].index)
    rules, s20 = build_rules(fr, C)
    independent_check(fr, C, rules, "SOXX")
    sanity("SOXX", rules, fr, C, s20)
    soxx_spotcheck(rules, fr, C["SOXX"], s20["SOXX"])
    summ, ver = bc_lib.run_rules(rules, "SOXX", FAMILY, fr=fr, tag="f1_SOXX")
    out["SOXX"] = (summ, ver)
    # 종목
    sf = bc_lib.stock_frames()
    Cs = bc_lib.load_stocks()[1]
    assert Cs.index.equals(sf["up"].index) and list(Cs.columns) == list(sf["in_dd"].columns)
    assert len(Cs.columns) == 82, len(Cs.columns)
    srules, ss20 = build_rules(sf, Cs)
    independent_check(sf, Cs, srules, "STK")
    sanity("STK", srules, sf, Cs, ss20)
    summ2, ver2 = bc_lib.run_rules(srules, "STK", FAMILY, fr=sf, tag="f1_STK")
    out["STK"] = (summ2, ver2)

    # ------------------------------------------------------------ 사후(post-hoc) 점검 — 등록 판정 아님
    # 결과를 본 뒤 추가. 등록 L1 후보의 하루 늦춤(BRIEF 반박 검증 항목)과, 종목 sig6 의 up 수준 대조
    # (up ≥ 0.267 = 종목 2026 sig6 첫 신호 up 중앙값, 결과에서 고른 문턱).
    ph_soxx = {f"{k}_lag1": rules[k].shift(1, fill_value=False) for k in ("up8", "up10", "sig2", "sig3", "sig4")}
    ph_stk = {f"{k}_lag1": srules[k].shift(1, fill_value=False) for k in ("up20", "sig4", "sig6")}
    ph_stk["up267_ctrl"] = (sf["up"] >= 0.267) & sf["up"].notna()
    _, phv1 = bc_lib.run_rules(ph_soxx, "SOXX", FAMILY + "_posthoc", fr=fr, tag="f1_SOXX_posthoc")
    _, phv2 = bc_lib.run_rules(ph_stk, "STK", FAMILY + "_posthoc", fr=sf, tag="f1_STK_posthoc")
    out["SOXX_posthoc"] = (pd.read_csv(bc_lib.RES / "f1_SOXX_posthoc_summary.csv"), phv1)
    out["STK_posthoc"] = (pd.read_csv(bc_lib.RES / "f1_STK_posthoc_summary.csv"), phv2)

    cols = ["period", "n", "clusters", "up_median", "hold20", "hold20_base", "hold20_dlo", "hold20_lo",
            "near5", "win20", "r20", "r60"]
    for kind, (summ, ver) in out.items():
        print(f"\n== {kind} 요약 ==")
        for r in summ.rule.unique():
            t = summ[summ.rule == r]
            print(f"-- {r}")
            print(t[[c for c in cols if c in t.columns]].round(3).to_string(index=False))
        print(f"\n== {kind} 판정 ==")
        print(ver.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
