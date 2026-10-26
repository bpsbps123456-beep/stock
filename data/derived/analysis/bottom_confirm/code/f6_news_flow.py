"""bottom_confirm F6 뉴스·자금·옵션 — BRIEF.md(2026-10-26, 결과 전 고정) 갈래 F6, 규칙 10개, SOXX·종목 둘 다.

신호만 만든다(평가·대조·판정은 bc_lib.run_rules). 신호일 t = 그날 종가까지 아는 정보만 쓴다. 진입은 t+1 시가(엔진).
모두 시장 전체 신호라 날짜마다 한 번 계산하고 종목(STK)에는 82개 열 모두에 같은 값을 쓴다.

뉴스(bc_lib.load_notes(), 노트 245개, entry = 그 노트를 처음 쓸 수 있던 시가의 세션, 2023-02-07~2026-10-23)
  신호일 t = entry 세션 자체(하루 보수적: 노트는 t 시가 전에 알 수 있었지만 진입은 t+1 시가).
  news_S            family == 'S'
  news_D60          family == 'D' 이고 lookback_months >= 60 (lookback 결측은 제외)
  news_D            family == 'D'
  news_R            family == 'R'
  news_RafterD      family == 'R' 이고 D 노트 entry 가 세션 [t-20, t-1] 안에 하나 이상
  news_GSR_posthoc  family == 'R' 이고 bank == 'GS'. 앞선 연구(news_followup)의 1차 사후 후보라 '사후'로 표시한다.
                    BRIEF 의 F6 10칸 안에 들어 있어 등록 태그(f6_*)에서 함께 평가하되 이름에 _posthoc 을 붙였다.
자금·옵션(bc_lib.load_market() = leverage_flood us_daily_panel.csv, 2026-10-13 이 마지막)
  분위 기준선은 모두 직전 250세션 [t-250, t-1](오늘 값 제외), 유효 관측 120개 이상일 때만, 분위는 선형 보간(pandas 기본).
  pc_eq90    pc_equity_5d[t] >= 90분위(pc_equity_5d[t-250..t-1])      CBOE 풋콜은 당일 장 마감 뒤 공개 → 늦춤 없음
  pc_tot90   pc_total_5d 같은 기준
  soxs10     SOXL_SOXS_dollar_vol_ratio_5d[t] <= 10분위(직전 250)     거래량은 당일 종가 뒤 확정 → 늦춤 없음
  tqqq_in90  L[t] = TQQQ_flow_5d[t-1](발행 주식 수가 장 마감 뒤 공개라 하루 늦춤); L[t] >= 90분위(L[t-250..t-1])
             (L 이 하루 늦어 자료 마지막 2026-10-13 의 값은 2026-10-14 신호에 쓰인다)
  '유효 관측 120개'는 과제가 pc_* 에 명시한 조건이며 soxs10·tqqq_in90 에도 같은 값을 적용했다(문턱 조정 아님, 준비 기간 규칙 통일).

결과 태그: results/f6_SOXX_*.csv, results/f6_STK_*.csv(등록 10칸, 문턱 조정 없음).
사후(post-hoc, 결과 본 뒤 추가, 판정에 쓰지 않음): results/f6_SOXX_posthoc_*.csv, results/f6_STK_posthoc_*.csv
  soxs10_nogap : 패널의 SOXL_SOXS 5일 비율은 휴장일 행(2026-05-25·09-07, 거래량 NaN)이 rolling 에 끼어 그 뒤 4세션씩
                 NaN 이다(2026-05-26~29, 09-08~11). 세션 색인에서 SOXL·SOXS 달러 거래량 5일 합으로 다시 만든 비율의 soxs10
                 (자료 결측 민감도).
  <rule>_lag1  : 등록 판정에서 L1 을 통과한 규칙의 하루 늦춤(BRIEF 반박 검증 항목). L1 이 없으면 만들지 않는다.
  무작위 날짜 귀무(화면 출력만, 파일 없음): L1 규칙의 채택 신호마다 같은 조각·같은 up 구간 in_dd 날을 뽑은 평균 hold20 분포.
실행: cd data/derived/analysis/bottom_confirm && python3 -W ignore code/f6_news_flow.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import bc_lib  # noqa: E402

FAMILY = "f6"
QWIN = 250          # 직전 250세션
QMIN = 120          # 유효 관측 하한
D_WIN = 20          # news_RafterD: D 노트가 [t-20, t-1] 안
KNOWN_LOWS = ("2024-08-07", "2025-04-08", "2026-03-30", "2026-07-29")
FLOW_COLS = {"pc_eq90": "pc_equity_5d", "pc_tot90": "pc_total_5d", "soxs10": "SOXL_SOXS_dollar_vol_ratio_5d",
             "tqqq_in90": "TQQQ_flow_5d"}
NEWS_RULES = ("news_S", "news_D60", "news_D", "news_R", "news_RafterD", "news_GSR_posthoc")


# ---------------------------------------------------------------- 신호(날짜 하나짜리 시장 신호)
def note_days(notes: pd.DataFrame, mask: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    """mask 에 맞는 노트의 entry 세션 = True."""
    d = pd.DatetimeIndex(notes.loc[mask, "entry"].unique())
    assert d.isin(idx).all(), "세션 밖 entry"
    return pd.Series(idx.isin(d), index=idx)


def trailing_q(x: pd.Series, q: float) -> pd.Series:
    """x[t-250..t-1] 의 q 분위(오늘 제외), 유효 관측 120 이상."""
    return x.shift(1).rolling(QWIN, min_periods=QMIN).quantile(q)


def news_rules(notes: pd.DataFrame, idx: pd.DatetimeIndex) -> dict[str, pd.Series]:
    fam, lb, bank = notes["family"], notes["lookback_months"], notes["bank"]
    r = {
        "news_S": note_days(notes, fam == "S", idx),
        "news_D60": note_days(notes, (fam == "D") & (lb >= 60), idx),
        "news_D": note_days(notes, fam == "D", idx),
        "news_R": note_days(notes, fam == "R", idx),
    }
    d_recent = r["news_D"].astype(int).shift(1, fill_value=0).rolling(D_WIN, min_periods=1).sum() > 0   # D in [t-20, t-1]
    r["news_RafterD"] = r["news_R"] & d_recent
    r["news_GSR_posthoc"] = note_days(notes, (fam == "R") & (bank == "GS"), idx)
    return r


def flow_rules(m: pd.DataFrame) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    """반환: (규칙명 → 세션 bool, 진단값(값·문턱))."""
    rules, diag = {}, {}
    for name, col, q, hi in (("pc_eq90", "pc_equity_5d", 0.90, True), ("pc_tot90", "pc_total_5d", 0.90, True),
                             ("soxs10", "SOXL_SOXS_dollar_vol_ratio_5d", 0.10, False)):
        x = m[col]
        th = trailing_q(x, q)
        rules[name] = ((x >= th) if hi else (x <= th)) & x.notna() & th.notna()
        diag[name] = (x, th)
    lag = m["TQQQ_flow_5d"].shift(1)                     # 발행 주식 수 자료 하루 늦춤
    th = trailing_q(lag, 0.90)
    rules["tqqq_in90"] = (lag >= th) & lag.notna() & th.notna()
    diag["tqqq_in90"] = (lag, th)
    return {k: v.fillna(False).astype(bool) for k, v in rules.items()}, diag


def soxs_nogap_ratio(m: pd.DataFrame) -> pd.Series:
    """사후: 세션 색인에서 다시 만든 SOXL/SOXS 달러 거래량 5일 비율(휴장일 NaN 행 없음)."""
    a = m["SOXL_dollar_vol"].rolling(5, min_periods=5).sum()
    b = m["SOXS_dollar_vol"].rolling(5, min_periods=5).sum()
    return a / b


def broadcast(sig: pd.Series, fr: dict) -> pd.DataFrame:
    """날짜 신호를 frames 색인·열 전체로 펼친다(SOXX 는 열 1개, 종목은 82개)."""
    idx, cols = fr["in_dd"].index, fr["in_dd"].columns
    v = sig.reindex(idx).fillna(False).astype(bool).to_numpy()
    return pd.DataFrame(np.repeat(v[:, None], len(cols), axis=1), index=idx, columns=cols)


# ---------------------------------------------------------------- 도우미
def decline_low_pos(c: np.ndarray, t: int) -> tuple[int, int]:
    """pk = t 에서 끝나는 60세션 종가 최고점 위치(같은 값이면 늦은 날, bc_lib.outcome_frame 과 같음),
    l = [pk, t] 최저 종가의 위치(같은 값이면 마지막)."""
    w = c[t - 59:t + 1]
    pk = t - 59 + (59 - int(np.argmax(w[::-1])))
    seg = c[pk:t + 1]
    l = pk + (len(seg) - 1 - int(np.argmin(seg[::-1])))
    return pk, l


# ---------------------------------------------------------------- 독립 재계산(반복문)
def independent_check(m: pd.DataFrame, notes: pd.DataFrame, rules: dict[str, pd.Series]) -> None:
    idx = m.index
    N = len(idx)
    pos = {d: i for i, d in enumerate(idx)}

    # 뉴스: 노트 하나씩 세션 위치로
    exp = {k: np.zeros(N, bool) for k in NEWS_RULES}
    dpos = sorted(pos[e] for e, f in zip(notes.entry, notes.family) if f == "D")
    for e, f, lb, bk in zip(notes.entry, notes.family, notes.lookback_months, notes.bank):
        p = pos[e]
        if f == "S":
            exp["news_S"][p] = True
        if f == "D":
            exp["news_D"][p] = True
            if pd.notna(lb) and lb >= 60:
                exp["news_D60"][p] = True
        if f == "R":
            exp["news_R"][p] = True
            if any(p - D_WIN <= q <= p - 1 for q in dpos):
                exp["news_RafterD"][p] = True
            if bk == "GS":
                exp["news_GSR_posthoc"][p] = True
    for k in NEWS_RULES:
        assert (rules[k].to_numpy() == exp[k]).all(), k

    # 자금·옵션: 창을 직접 잘라 np.quantile
    def loop_q(a: np.ndarray, q: float, hi: bool) -> np.ndarray:
        out = np.zeros(N, bool)
        for t in range(N):
            w = a[max(0, t - QWIN):t]
            w = w[~np.isnan(w)]
            if len(w) < QMIN or np.isnan(a[t]):
                continue
            th = np.quantile(w, q)
            out[t] = a[t] >= th if hi else a[t] <= th
        return out

    assert (rules["pc_eq90"].to_numpy() == loop_q(m["pc_equity_5d"].to_numpy(float), 0.9, True)).all(), "pc_eq90"
    assert (rules["pc_tot90"].to_numpy() == loop_q(m["pc_total_5d"].to_numpy(float), 0.9, True)).all(), "pc_tot90"
    assert (rules["soxs10"].to_numpy() == loop_q(m["SOXL_SOXS_dollar_vol_ratio_5d"].to_numpy(float), 0.1, False)).all(), "soxs10"
    f = m["TQQQ_flow_5d"].to_numpy(float)
    lag = np.full(N, np.nan)
    lag[1:] = f[:-1]
    assert (rules["tqqq_in90"].to_numpy() == loop_q(lag, 0.9, True)).all(), "tqqq_in90"
    # 마지막 자료일 뒤 신호: 늦춤 없는 셋은 없어야 하고, tqqq_in90 은 다음 한 세션까지만 가능
    last = m["pc_equity_5d"].last_valid_index()
    for n in ("pc_eq90", "pc_tot90", "soxs10"):
        assert not rules[n].loc[last + pd.Timedelta(days=1):].any(), f"{n}: {last.date()} 뒤 신호"
    nxt = idx[idx.get_loc(last) + 1]
    assert not rules["tqqq_in90"].loc[nxt + pd.Timedelta(days=1):].any(), "tqqq_in90 늦춤 뒤 신호"
    print(f"독립 재계산(반복문) 10개 규칙 일치; 자금·옵션 자료 끝 {last.date()} 뒤 신호 없음(tqqq_in90 은 {nxt.date()}까지 가능)")


# ---------------------------------------------------------------- 점검 출력
def data_report(m: pd.DataFrame, notes: pd.DataFrame, rules: dict[str, pd.Series], diag: dict) -> None:
    print("== 자료 범위·결측(XNYS 세션 2023-01-03~2026-10-23 기준) ==")
    for c in ("pc_equity", "pc_equity_5d", "pc_total_5d", "SOXL_SOXS_dollar_vol_ratio_5d", "SOXL_dollar_vol",
              "SOXS_dollar_vol", "TQQQ_shares", "TQQQ_flow_5d"):
        s = m[c]
        a, b = s.first_valid_index(), s.last_valid_index()
        g = s.loc[a:b]
        gaps = list(g.index[g.isna()].date)
        print(f"  {c:30s} {a.date()} ~ {b.date()}  유효 {int(s.notna().sum()):4d}  범위 안 결측 {len(gaps)} "
              f"{[str(d) for d in gaps] if gaps else ''}  2026-10-23 까지 끝 결측 {int(s.loc[b:].isna().sum()) - 0}")
    print("  (SOXL_SOXS 5일 비율 결측 8세션 = 패널 원본에 휴장일 행 2026-05-25·09-07 이 거래량 NaN 으로 끼어 rolling(5) 이 4세션씩 NaN.)")
    print("  (CBOE 풋콜 원본에는 휴장일 2025-01-09 행이 하나 더 있어 그 주의 pc_*_5d 는 그 행을 5개 중 하나로 썼다; 세션 결측은 없음.)")
    print("== 분위 기준선이 처음 생기는 날(유효 관측 120) ==")
    for n, (x, th) in diag.items():
        print(f"  {n:10s} 문턱 첫 유효 {th.first_valid_index().date()}  마지막 유효 값 {x.last_valid_index().date()}  "
              f"신호 가능 세션 {int((x.notna() & th.notna()).sum())}")
    print("== 노트 ==")
    vc = notes.family.value_counts().to_dict()
    print(f"  노트 {len(notes)}개, entry {notes.entry.min().date()} ~ {notes.entry.max().date()}, family {vc}")
    for n in NEWS_RULES:
        print(f"  {n:17s} 신호 세션 {int(rules[n].sum()):3d}  (2024~: {int(rules[n].loc['2024':].sum())}, "
              f"2025-09~: {int(rules[n].loc['2025-09':].sum())}, 2026: {int(rules[n].loc['2026':].sum())})")
    dl = notes[(notes.family == "D")].lookback_months
    print(f"  D 노트 {len(dl)}개 중 lookback 결측 {int(dl.isna().sum())}개(news_D60 에서 제외), >=60 {int((dl >= 60).sum())}개")


def sanity(kind: str, rules: dict[str, pd.DataFrame], fr: dict) -> None:
    idx = fr["in_dd"].index
    ev = (idx >= bc_lib.EVAL_START[kind]) & (idx <= bc_lib.LAST)
    ev_dd = fr["in_dd"] & fr["L60"].notna()
    print(f"\n== {kind}: 규칙별 신호 날 수 (raw = 날짜 수, 종목은 날짜×종목 칸 수도 함께) ==")
    print(f"{'rule':17s} {'days_all':>8s} {'days_eval':>9s} {'cells_eval_in_dd':>16s} {'first':>6s} {'first_dates(SOXX)/n_series(STK)'}")
    for name, sig in rules.items():
        day = sig.iloc[:, 0]
        fs = bc_lib.first_signals(sig, fr, kind)
        cells = int((sig & ev_dd).loc[ev].to_numpy().sum())
        extra = (" ".join(str(d.date()) for d in fs.date) if kind == "SOXX"
                 else f"{fs.series.nunique() if len(fs) else 0} series, dates {fs.date.nunique() if len(fs) else 0}")
        print(f"{name:17s} {int(day.sum()):8d} {int(day.loc[ev].sum()):9d} {cells:16d} {len(fs):6d}  {extra}")
    if kind == "STK":
        first_l = fr["L60"].apply(lambda s: s.first_valid_index())
        print(f"종목 Ldd 첫 유효일: 최소 {first_l.min().date()} / 최대 {first_l.max().date()} "
              f"(종목 자료 2025-06-02 시작, 60거래일 준비; 시장 신호 자체는 2023 년부터 계산해 종목 쪽 준비 기간과 무관)")
        late = first_l[first_l > bc_lib.EVAL_START[kind]]
        if len(late):
            print(f"평가 시작 뒤에야 Ldd 가 생기는 종목 {len(late)}개: " + ", ".join(f"{k}({v.date()})" for k, v in late.sort_values().items()))
        n_dd = fr["in_dd"].loc[ev].sum(1)
        print(f"평가 창 날짜별 in_dd 종목 수: 중앙 {int(n_dd.median())}, 최대 {int(n_dd.max())}, 0인 날 {int((n_dd == 0).sum())}/{int(ev.sum())}")


def soxx_spotcheck(rules: dict[str, pd.DataFrame], fr: dict, diag: dict, notes: pd.DataFrame, C: pd.Series) -> None:
    idx = fr["in_dd"].index
    c = C.to_numpy(float)
    by_day = notes.groupby("entry").apply(lambda g: ",".join(f"{f}/{b}" for f, b in zip(g.family, g.bank)))
    print("\n== SOXX 알려진 저점 주변 점검 (저점 −3 ~ +20 세션) ==")
    for d in KNOWN_LOWS:
        p = idx.get_loc(pd.Timestamp(d))
        pk, l = decline_low_pos(c, p)
        print(f"\n저점 {d}: C={c[p]:.2f}, 60일 고점 {idx[pk].date()} {c[pk]:.2f}, 하락 저점 {idx[l].date()} "
              f"Ldd={fr['L60'].iat[p, 0]:.2f}, dd60={fr['dd60'].iat[p, 0]:+.3f}, in_dd={bool(fr['in_dd'].iat[p, 0])}")
        assert abs(c[l] - fr["L60"].iat[p, 0]) < 1e-9
        for q in range(max(p - 3, 0), min(p + 21, len(idx))):
            t = idx[q]

            def g(n):
                x, th = diag[n]
                return f"{x.get(t, np.nan):.3g}/{th.get(t, np.nan):.3g}"
            fired = [n for n, s in rules.items() if bool(s.iat[q, 0])]
            print(f"  {t.date()} C={c[q]:7.2f} up={fr['up'].iat[q, 0]:+.3f} in_dd={int(fr['in_dd'].iat[q, 0])} "
                  f"pcEq={g('pc_eq90')} pcTot={g('pc_tot90')} soxs={g('soxs10')} tqqqL1={g('tqqq_in90')} "
                  f"notes={by_day.get(t, '-')} fires={','.join(fired) or '-'}")
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


def random_date_null(kind: str, rule: str, fr: dict, n_draw: int = 5000) -> None:
    """사후 진단(파일 없음): L1 규칙의 채택 신호마다 같은 기간 조각·같은 up 구간의 in_dd 날(hold20 있는 날)을
    무작위로 하나씩 뽑아 평균 hold20 을 만든다. 관측 평균 이상이 나올 확률(주 기간)을 출력한다."""
    sig = pd.read_csv(bc_lib.RES / f"{FAMILY}_{kind}_signals.csv", encoding="utf-8-sig", parse_dates=["date"])
    g = sig[(sig.rule == rule) & sig.hold20.notna()]
    main = {"SOXX": ("2024-01-02", "2026-10-23"), "STK": ("2026-01-02", "2026-10-23")}[kind]
    g = g[(g.date >= main[0]) & (g.date <= main[1])]
    if not len(g):
        return
    up, h = fr["up"], fr["hold20"]
    idx = up.index
    ok = fr["in_dd"] & up.notna() & h.notna() & pd.DataFrame(
        np.broadcast_to((idx >= bc_lib.EVAL_START[kind])[:, None], up.shape), index=idx, columns=up.columns)
    sk = bc_lib._slice_key(idx, kind)
    ub = np.digitize(up.to_numpy(float), bc_lib.UP_BINS[1:-1])
    pools = {}
    hv, okv = h.to_numpy(float), ok.to_numpy()
    for p, s in zip(g.pos, g.series):
        k = up.columns.get_loc(s)
        key = (sk[p], ub[p, k])
        if key not in pools:
            m = okv & (sk[:, None] == key[0]) & (ub == key[1])
            pools[key] = hv[m]
    rng = np.random.default_rng(bc_lib.SEED)
    draws = np.zeros(n_draw)
    for p, s in zip(g.pos, g.series):
        k = up.columns.get_loc(s)
        pool = pools[(sk[p], ub[p, k])]
        draws += pool[rng.integers(0, len(pool), n_draw)]
    draws /= len(g)
    obs = float(g.hold20.mean())
    print(f"  [사후 무작위 날짜 귀무] {kind} {rule}: 관측 hold20 {obs:.3f} (n={len(g)}), 같은 조각·up 구간 무작위 평균 "
          f"{draws.mean():.3f}, P(귀무 >= 관측) = {(draws >= obs - 1e-12).mean():.4f}")


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
    m = bc_lib.load_market()
    notes = bc_lib.load_notes()
    day_rules = news_rules(notes, m.index)
    fl, diag = flow_rules(m)
    day_rules.update(fl)
    order = ["news_S", "news_D60", "news_D", "news_R", "news_RafterD", "news_GSR_posthoc",
             "pc_eq90", "pc_tot90", "soxs10", "tqqq_in90"]
    day_rules = {k: day_rules[k] for k in order}
    data_report(m, notes, day_rules, diag)
    independent_check(m, notes, day_rules)

    out = {}
    fr = bc_lib.soxx_frames()
    assert fr["in_dd"].index.equals(m.index)
    srules = {n: broadcast(s, fr) for n, s in day_rules.items()}
    sanity("SOXX", srules, fr)
    soxx_spotcheck(srules, fr, diag, notes, bc_lib.load_soxx()["C"])
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
    x = soxs_nogap_ratio(m)
    th = trailing_q(x, 0.10)
    nogap = ((x <= th) & x.notna() & th.notna()).fillna(False).astype(bool)
    diff = nogap.index[nogap != day_rules["soxs10"]]
    print(f"\n[사후] soxs10_nogap: 패널 비율과 다시 만든 비율의 최대 차이(둘 다 유효) "
          f"{float((x - m['SOXL_SOXS_dollar_vol_ratio_5d']).abs().max()):.2e}; 신호가 달라지는 날: "
          + (", ".join(str(d.date()) for d in diff) or "없음"))
    ph_day = {"soxs10_nogap": nogap}
    for kind in ("SOXX", "STK"):
        ver = out[kind][1]
        for r in ver.loc[ver.L1, "rule"]:
            ph_day.setdefault(f"{r}_lag1", day_rules[r].shift(1, fill_value=False))
    for kind, f in (("SOXX", fr), ("STK", sf)):
        ph = {n: broadcast(s, f) for n, s in ph_day.items()}
        print(f"\n[사후] {kind}")
        sanity(kind, ph, f)
        out[f"{kind}_posthoc"] = bc_lib.run_rules(ph, kind, FAMILY + "_posthoc", fr=f, tag=f"{FAMILY}_{kind}_posthoc")

    print("\n[사후] 등록 판정 L1 규칙의 같은 상태 무작위 날짜 귀무(주 기간, 쿨다운·묶음 무시)")
    for kind, f in (("SOXX", fr), ("STK", sf)):
        ver = out[kind][1]
        for r in ver.loc[ver.L1, "rule"]:
            random_date_null(kind, r, f)

    print_results(out)


if __name__ == "__main__":
    main()
