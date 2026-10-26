"""bottom_confirm F7 결합 — BRIEF.md(2026-10-26, 결과 전 고정) 갈래 F7, 규칙 6개, SOXX·종목 둘 다.

신호만 만든다(평가·대조·판정은 bc_lib.run_rules). 신호일 t = 그날 종가까지 아는 정보만 쓴다(모든 창은 t 에서 끝남).
BRIEF: 사건형 구성요소(S 노트·VIX 하락·시장 폭·VIX/VIX3M 정상화)는 오늘을 포함한 최근 10거래일 [t-9, t] 안에 한 번이라도
있었으면 참, 가격 상태 구성요소(up·SMA20·저점 뒤 일수·높은 저점)는 오늘 값. 구성요소는 이 파일에서 직접 다시 계산하고
F2·F4·F5·F6 코드와 같은 값인지 실행 때 대조한다(그 파일들은 읽기만 한다).

공통 위치(계열마다 그 계열 자신의 종가 C 로)
  pk[t] = t 에서 끝나는 60세션 종가 최고의 위치(같은 값이면 늦은 날, bc_lib.outcome_frame 과 같음)
  l[t]  = C[pk..t] 최저 종가의 마지막 위치  → C[l] = Ldd = bc_lib L60
  up[t] = C[t]/Ldd[t] − 1 (bc_lib frames 'up')
사건(날짜 d 하나의 시장 값, 10세션 창은 d 에서 끝나고 값이 모두 있어야 함 = min_periods 10)
  E_vix[d]    Mx = max(^VIX[d-9..d]); Mx >= 30 이고 ^VIX[d] <= 0.85·Mx                        (F4 vix30_15 와 같은 식)
  E_thr[d]    pct20 = SMA20 위 종목 비율(82종목, SMA20 유효 종목이 분모); min(pct20[d-9..d]) <= 0.25 이고 pct20[d] >= 0.50
                                                                                              (F5 thrust25_50 과 같은 식)
  E_S[d]      family == 'S' 노트의 entry 세션 == d                                             (F6 news_S 와 같은 날)
  E_ts[d]     R = ^VIX/^VIX3M; max(R[d-9..d]) >= 1.0 이고 R[d] < 0.95                           (F4 ts_norm 과 같은 식)
  recent(E)[t] = E[t-9..t] 중 하나라도 참(세션 색인 2023-01-03~ 위에서 계산한 뒤 frames 색인으로 옮김)
규칙
  c_up5_since5       up >= 5% 이고 (t − l) >= 5
  c_up5_ma20         up >= 5% 이고 C[t] > SMA20[t]                      (SMA20 = 종가 20일 평균, min_periods 20)
  c_vix30crush_up5   recent(E_vix) 이고 up >= 5%
  c_thrust25_50_up5  recent(E_thr) 이고 up >= 5%
  c_newsS_up3        recent(E_S) 이고 up >= 3%
  c_hl_tsnorm        F2 hl 이 t 에 켜짐 이고 recent(E_ts)
                     hl: l < s <= t-2, C[s] == min(C[s-2..s+2]) (s+2 <= t 라 t 에 확정), C[s] >= 1.01·C[l] 인 가장 최근 s
                     (F2 와 같음), H = max(C[l..s]); C[t] > H 이고 C[t-1] <= H.
  시장 사건은 날짜마다 한 번 계산해 종목(STK)에는 82개 열 모두에 같은 값으로, 가격 상태는 계열(종목)마다 자기 종가로.
  하루 늦춤: 쓰는 자료가 모두 그날 종가로 확정되는 값(지수 종가·종목 종가)이거나 시가 전 공개 노트라 늦춤 대상이 없다
  (발행 주식 수·자금 흐름 자료를 쓰지 않음).

자료 공백
  ^VIX·^VIX3M 은 2026-10-13 까지(그 뒤 8세션 NaN → 사건 없음, 10-13 이전 사건은 [t-9, t] 로 10-23 까지 이어질 수 있음).
  시장 폭은 종목 자료 2025-06-02 시작이라 SMA20(20세션)+10일 창 → E_thr 첫 가능일 2025-07-14. SOXX 의 2024·2025 상반기에는
  c_thrust25_50_up5 가 켜질 수 없다(2024-08-07·2025-04-08 저점 포함).
  종목 가격 상태는 60세션 창이 필요해 종목 Ldd 첫 유효일 2025-08-25 경(늦게 상장한 종목은 더 늦음), 평가 창 2025-09-01~.
  노트 S 는 11개(2023-11-13~2026-10-23).

결과 태그: results/f7_SOXX_*.csv, results/f7_STK_*.csv(등록 6개 규칙, 문턱 조정 없음).
사후(post-hoc, 결과 본 뒤 추가, 판정에 쓰지 않음): results/f7_SOXX_posthoc_*.csv, results/f7_STK_posthoc_*.csv
  <rule>_lag1          : 등록 판정에서 L1 을 통과한 규칙의 하루 늦춤(BRIEF 반박 검증 항목). 신호표를 한 세션 뒤로 민다.
  <rule>_<이웃>        : L1 을 통과한 규칙의 이웃 값(BRIEF 반박 검증 항목). 문턱 하나씩 한 칸: up 4·6%, 저점 뒤 4·6일,
                         SMA10·50(F2 격자 이웃). 결과를 본 뒤 고른 칸이라 판정에 쓰지 않는다.
  c_hl_tsnorm_panelvix : ^VIX 2026-02-06 값(20.37)이 같은 날 ^VIX3M 과 똑같아 R = 1.000(F4 가 찾은 자료 오류 의심).
                         그 하루를 패널 'VIX' 로 바꾼 E_ts 로 다시 만든 c_hl_tsnorm(c_hl_tsnorm 신호가 달라질 때만 만든다).
실행: cd data/derived/analysis/bottom_confirm && python3 -W ignore code/f7_combo.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import bc_lib  # noqa: E402

FAMILY = "f7"
EV_WIN = 10            # 사건 [t-9, t]
WIN = 10               # 사건 정의 안의 10세션 창
UP5, UP3 = 0.05, 0.03
SINCE = 5
VIX_LV, VIX_KEEP = 30.0, 0.85
THR_LO, THR_HI = 0.25, 0.50
TS_HI, TS_LO = 1.0, 0.95
HL_MIN = 1.01
RULES = ("c_up5_since5", "c_up5_ma20", "c_vix30crush_up5", "c_thrust25_50_up5", "c_newsS_up3", "c_hl_tsnorm")
KNOWN_LOWS = ("2024-08-07", "2025-04-08", "2026-03-30", "2026-07-29")


# ---------------------------------------------------------------- 가격 상태(계열마다)
def decline_low_pos(C: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """PK[t,k] = 60세션 종가 최고 위치(같으면 늦은 날), LL[t,k] = C[pk..t] 최저의 마지막 위치. 창에 결측이면 -1."""
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
            pk = t - int(np.argmax(w[::-1]))                       # 뒤집어 첫 최고 = 원래의 마지막 최고
            seg = c[pk:t + 1]
            PK[t, k] = pk
            LL[t, k] = t - int(np.argmin(seg[::-1]))               # 마지막 최저
    return PK, LL


def swing_lows(C: pd.DataFrame) -> np.ndarray:
    """sw[s,k] = C[s] == min(C[s-2..s+2]) 이고 그 5개 값이 모두 있음."""
    Cv = C.to_numpy(float)
    N, K = Cv.shape
    sw = np.zeros((N, K), bool)
    for s in range(2, N - 2):
        w = Cv[s - 2:s + 3]
        sw[s] = ~np.isnan(w).any(0) & (Cv[s] == np.min(np.where(np.isnan(w), np.inf, w), 0))
    return sw


def higher_low(C: pd.DataFrame, LL: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """F2 hl. 반환 (신호, 쓴 s 위치(-1 = 없음))."""
    Cv = C.to_numpy(float)
    N, K = Cv.shape
    sw = swing_lows(C)
    out = np.zeros((N, K), bool)
    S = np.full((N, K), -1, int)
    for k in range(K):
        c = Cv[:, k]
        for t in range(2, N):
            l = LL[t, k]
            if l < 0 or np.isnan(c[t]) or np.isnan(c[t - 1]):
                continue
            cand = np.flatnonzero(sw[l + 1:t - 1, k] & (c[l + 1:t - 1] >= HL_MIN * c[l]))   # s ∈ (l, t-2]
            if not len(cand):
                continue
            s = l + 1 + int(cand[-1])                              # 가장 최근 s
            S[t, k] = s
            H = c[l:s + 1].max()
            out[t, k] = (c[t] > H) and (c[t - 1] <= H)
    return pd.DataFrame(out, index=C.index, columns=C.columns), S


# ---------------------------------------------------------------- 시장 사건(날짜 하나짜리)
def pct_above_sma20(C: pd.DataFrame) -> pd.Series:
    sma = C.rolling(20, min_periods=20).mean()
    v = sma.notna() & C.notna()
    n = v.sum(1)
    return ((C > sma) & v).sum(1) / n.where(n > 0)


def market_events(m: pd.DataFrame, Cstk: pd.DataFrame, notes: pd.DataFrame, vix: pd.Series | None = None
                  ) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    """m.index(XNYS 2023-01-03~2026-10-23) 위의 사건 bool. 반환 (사건, 진단표)."""
    idx = m.index
    vix = m["^VIX"] if vix is None else vix
    mx = vix.rolling(WIN, min_periods=WIN).max()
    e_vix = (mx >= VIX_LV) & (vix <= VIX_KEEP * mx) & mx.notna() & vix.notna()

    R = vix / m["^VIX3M"]
    rmx = R.rolling(WIN, min_periods=WIN).max()
    e_ts = (rmx >= TS_HI) & (R < TS_LO) & rmx.notna() & R.notna()

    pct = pct_above_sma20(Cstk).reindex(idx)                      # 종목 자료 이전은 NaN
    pmn = pct.rolling(WIN, min_periods=WIN).min()
    e_thr = (pmn <= THR_LO) & (pct >= THR_HI) & pmn.notna() & pct.notna()

    sd = pd.DatetimeIndex(notes.loc[notes.family == "S", "entry"].unique())
    assert sd.isin(idx).all(), "세션 밖 S 노트 entry"
    e_s = pd.Series(idx.isin(sd), index=idx)

    ev = {k: v.fillna(False).astype(bool) for k, v in dict(vix=e_vix, ts=e_ts, thr=e_thr, S=e_s).items()}
    diag = pd.DataFrame(dict(VIX=vix, VIXmx10=mx, R=R, Rmx10=rmx, pct20=pct, pctmn10=pmn))
    return ev, diag


def recent(e: pd.Series) -> pd.Series:
    """e[t-9..t] 중 하나라도 참(색인 앞쪽은 있는 만큼만)."""
    return e.astype(int).rolling(EV_WIN, min_periods=1).max().astype(bool)


def broadcast(sig: pd.Series, fr: dict) -> pd.DataFrame:
    idx, cols = fr["in_dd"].index, fr["in_dd"].columns
    v = sig.reindex(idx).fillna(False).astype(bool).to_numpy()
    return pd.DataFrame(np.repeat(v[:, None], len(cols), axis=1), index=idx, columns=cols)


# ---------------------------------------------------------------- 규칙
def build_rules(C: pd.DataFrame, fr: dict, ev: dict[str, pd.Series]) -> tuple[dict[str, pd.DataFrame], dict]:
    assert C.index.equals(fr["in_dd"].index) and list(C.columns) == list(fr["in_dd"].columns)
    N, K = C.shape
    PK, LL = decline_low_pos(C)
    # 하락 저점 위치 자기 점검: C[l] == bc_lib Ldd
    Cv = C.to_numpy(float)
    ok = LL >= 0
    cols = np.broadcast_to(np.arange(K), (N, K))
    assert np.allclose(Cv[LL[ok], cols[ok]], fr["L60"].to_numpy(float)[ok]), "C[l] != Ldd"
    assert (np.isnan(fr["L60"].to_numpy(float)) == ~ok).all(), "Ldd 유효 범위 불일치"

    up = fr["up"]
    pos = np.broadcast_to(np.arange(N)[:, None], (N, K))
    since = pd.DataFrame(np.where(ok, pos - LL, -1), index=C.index, columns=C.columns)
    sma20 = C.rolling(20, min_periods=20).mean()
    hl, hl_s = higher_low(C, LL)
    rc = {k: broadcast(recent(v), fr) for k, v in ev.items()}
    rules = {
        "c_up5_since5": (up >= UP5) & (since >= SINCE),
        "c_up5_ma20": (up >= UP5) & (C > sma20),
        "c_vix30crush_up5": rc["vix"] & (up >= UP5),
        "c_thrust25_50_up5": rc["thr"] & (up >= UP5),
        "c_newsS_up3": rc["S"] & (up >= UP3),
        "c_hl_tsnorm": hl & rc["ts"],
    }
    rules = {k: v.fillna(False).astype(bool) for k, v in rules.items()}
    return rules, dict(PK=PK, LL=LL, since=since, sma20=sma20, hl=hl, hl_s=hl_s, rc=rc)


def neighbors(name: str, C: pd.DataFrame, fr: dict, d: dict) -> dict[str, pd.DataFrame]:
    """사후 이웃 값(L1 통과 규칙만): 문턱 하나씩 한 칸 옮김. up ±1%p, 저점 뒤 일수 ±1, SMA 는 F2 격자의 이웃(10·50)."""
    up, since = fr["up"], d["since"]
    sma = lambda n: C.rolling(n, min_periods=n).mean()
    if name == "c_up5_since5":
        v = {"up4": (up >= 0.04) & (since >= SINCE), "up6": (up >= 0.06) & (since >= SINCE),
             "since4": (up >= UP5) & (since >= 4), "since6": (up >= UP5) & (since >= 6)}
    elif name == "c_up5_ma20":
        v = {"up4": (up >= 0.04) & (C > sma(20)), "up6": (up >= 0.06) & (C > sma(20)),
             "ma10": (up >= UP5) & (C > sma(10)), "ma50": (up >= UP5) & (C > sma(50))}
    else:
        v = {}
    return {f"{name}_{k}": s.fillna(False).astype(bool) for k, s in v.items()}


# ---------------------------------------------------------------- 독립 재계산(반복문, 정의에서 바로)
def loop_rules(C: pd.DataFrame, m: pd.DataFrame, Cstk: pd.DataFrame, notes: pd.DataFrame) -> dict[str, np.ndarray]:
    """벡터화·도우미를 쓰지 않고 정의 문장대로 t 마다 다시 계산한다. 반환 규칙 → (N × K) bool."""
    c_all = C.to_numpy(float)
    N, K = c_all.shape
    midx = m.index
    mpos = {d: i for i, d in enumerate(midx)}
    vix, v3 = m["^VIX"].to_numpy(float), m["^VIX3M"].to_numpy(float)
    cs = Cstk.reindex(midx).to_numpy(float)
    M = len(midx)

    def ev_vix(d):
        if d < WIN - 1:
            return False
        w = vix[d - WIN + 1:d + 1]
        return (not np.isnan(w).any()) and w.max() >= VIX_LV and vix[d] <= VIX_KEEP * w.max()

    def ev_ts(d):
        if d < WIN - 1:
            return False
        w = vix[d - WIN + 1:d + 1] / v3[d - WIN + 1:d + 1]
        return (not np.isnan(w).any()) and w.max() >= TS_HI and w[-1] < TS_LO

    pct = np.full(M, np.nan)
    for d in range(19, M):
        a = n = 0
        for j in range(cs.shape[1]):
            w = cs[d - 19:d + 1, j]
            if not np.isnan(w).any():
                n += 1
                a += cs[d, j] > w.mean()
        if n:
            pct[d] = a / n

    def ev_thr(d):
        if d < WIN - 1:
            return False
        w = pct[d - WIN + 1:d + 1]
        return (not np.isnan(w).any()) and w.min() <= THR_LO and pct[d] >= THR_HI

    sdays = {mpos[e] for e, f in zip(notes.entry, notes.family) if f == "S"}
    E = {"vix": np.array([ev_vix(d) for d in range(M)]), "ts": np.array([ev_ts(d) for d in range(M)]),
         "thr": np.array([ev_thr(d) for d in range(M)]), "S": np.array([d in sdays for d in range(M)])}

    def rec(name, t_date):
        d = mpos[t_date]
        return bool(E[name][max(0, d - EV_WIN + 1):d + 1].any())

    out = {r: np.zeros((N, K), bool) for r in RULES}
    for k in range(K):
        c = c_all[:, k]
        for t in range(59, N):
            w = c[t - 59:t + 1]
            if np.isnan(w).any():
                continue
            hi = w.max()
            pk = max(i for i in range(t - 59, t + 1) if c[i] == hi)
            lo = c[pk:t + 1].min()
            l = max(i for i in range(pk, t + 1) if c[i] == lo)
            u = c[t] / lo - 1
            td = C.index[t]
            sma = c[t - 19:t + 1].mean()
            out["c_up5_since5"][t, k] = u >= UP5 and t - l >= SINCE
            out["c_up5_ma20"][t, k] = u >= UP5 and c[t] > sma
            out["c_vix30crush_up5"][t, k] = u >= UP5 and rec("vix", td)
            out["c_thrust25_50_up5"][t, k] = u >= UP5 and rec("thr", td)
            out["c_newsS_up3"][t, k] = u >= UP3 and rec("S", td)
            # hl: 가장 최근 확정 국소 저점 s
            s_hit = None
            for s in range(t - 2, max(l, 1), -1):                  # s ∈ (l, t-2], 5일 창이 색인 안(s >= 2)
                ww = c[s - 2:s + 3]
                if not np.isnan(ww).any() and c[s] == ww.min() and c[s] >= HL_MIN * c[l]:
                    s_hit = s
                    break
            if s_hit is not None and not np.isnan(c[t - 1]):
                H = c[l:s_hit + 1].max()
                out["c_hl_tsnorm"][t, k] = c[t] > H and c[t - 1] <= H and rec("ts", td)
    return out


def cross_family_check(C_s: pd.DataFrame, fr_s: dict, C_k: pd.DataFrame, fr_k: dict, m: pd.DataFrame,
                       notes: pd.DataFrame, ev: dict[str, pd.Series], diag_s: dict, diag_k: dict) -> None:
    """구성요소가 F2·F4·F5·F6 코드의 같은 규칙과 같은 값인지(그 파일은 읽기만)."""
    import f2_reclaim
    import f4_vol
    import f5_breadth
    import f6_news_flow
    f4r, _ = f4_vol.market_rules(m)
    f5r, _ = f5_breadth.day_rules(f5_breadth.breadth(C_k))
    f6r = f6_news_flow.news_rules(notes, m.index)
    msg = []
    for name, ours, theirs in (("E_vix vs f4 vix30_15", ev["vix"], f4r["vix30_15"]),
                               ("E_ts vs f4 ts_norm", ev["ts"], f4r["ts_norm"]),
                               ("E_thr vs f5 thrust25_50", ev["thr"], f5r["thrust25_50"].reindex(m.index, fill_value=False)),
                               ("E_S vs f6 news_S", ev["S"], f6r["news_S"])):
        same = bool((ours.to_numpy() == theirs.reindex(m.index).fillna(False).to_numpy()).all())
        assert same, name
        msg.append(f"{name}={same}")
    for C, d in ((C_s, diag_s), (C_k, diag_k)):
        PK2, LL2 = f2_reclaim.decline_low_pos(C)
        hl2, _ = f2_reclaim.higher_low(C, PK2, LL2)
        assert (LL2 == d["LL"]).all() and (PK2 == d["PK"]).all(), "pk/l vs f2"
        assert (hl2.to_numpy() == d["hl"].to_numpy()).all(), "hl vs f2"
    msg.append("pk·l·hl vs f2 (SOXX·STK)=True")
    print("구성요소 대조: " + ", ".join(msg))


# ---------------------------------------------------------------- 점검 출력
def data_report(m: pd.DataFrame, Cstk: pd.DataFrame, notes: pd.DataFrame, ev: dict, diag: pd.DataFrame,
                fr_k: dict) -> None:
    print("== 자료 범위 ==")
    for c in ("^VIX", "^VIX3M"):
        s = m[c]
        a, b = s.first_valid_index(), s.last_valid_index()
        print(f"  {c:7s} {a.date()} ~ {b.date()}  범위 안 결측 {int(s.loc[a:b].isna().sum())}  끝 뒤 결측 {int(s.loc[b:].isna().sum())}"
              f"  마지막 5값 {', '.join(f'{x:.2f}' for x in s.dropna().iloc[-5:])}")
    print(f"  종목 종가 {Cstk.index[0].date()} ~ {Cstk.index[-1].date()} {Cstk.shape[1]}종목; pct20 첫 값 "
          f"{diag.pct20.first_valid_index().date()}, 10일 최저 첫 값 {diag.pctmn10.first_valid_index().date()}")
    fl = fr_k["L60"].apply(lambda s: s.first_valid_index())
    print(f"  종목 Ldd 첫 유효일 최소 {fl.min().date()} / 최대 {fl.max().date()}; 평가 시작(2025-09-01) 뒤에야 생기는 종목: "
          + (", ".join(f"{k}({v.date()})" for k, v in fl[fl > bc_lib.EVAL_START['STK']].sort_values().items()) or "없음"))
    sn = notes[notes.family == "S"].sort_values("entry")
    print(f"  S 노트 {len(sn)}개 entry: " + ", ".join(str(d.date()) for d in sn.entry))
    print("== 사건 날(전체 / 2024~ / 2025-09~) ==")
    for k, e in ev.items():
        print(f"  E_{k:4s} {int(e.sum()):4d} / {int(e.loc['2024':].sum()):4d} / {int(e.loc['2025-09':].sum()):4d}  "
              f"마지막 {e[e].index.max().date() if e.any() else '-'}")
    for k in ("vix", "ts", "thr"):
        e = ev[k]
        days = e[e].index
        runs, start, prev = [], None, None
        pos = {d: i for i, d in enumerate(m.index)}
        for d in days:
            if prev is None or pos[d] - pos[prev] > 1:
                if start is not None:
                    runs.append((start, prev))
                start = d
            prev = d
        if start is not None:
            runs.append((start, prev))
        print(f"  E_{k} 구간(2024~): " + ", ".join(f"{a.date()}" + ("" if a == b else f"~{b.date()}")
                                                  for a, b in runs if a >= pd.Timestamp("2024-01-01")))


def sanity(kind: str, rules: dict[str, pd.DataFrame], fr: dict) -> dict[str, pd.DataFrame]:
    idx = fr["in_dd"].index
    ev = (idx >= bc_lib.EVAL_START[kind]) & (idx <= bc_lib.LAST)
    ev_dd = fr["in_dd"] & fr["L60"].notna()
    firsts = {}
    print(f"\n== {kind}: 규칙별 원 신호 (raw = 날×계열 칸 전체, eval = 평가 창, eval·in_dd = 엔진이 보는 칸) · 첫 신호 ==")
    for name, sig in rules.items():
        raw = int(sig.to_numpy().sum())
        days = int(sig.any(axis=1).sum())
        e1 = int(sig.loc[ev].to_numpy().sum())
        e2 = int((sig & ev_dd).loc[ev].to_numpy().sum())
        fs = bc_lib.first_signals(sig, fr, kind)
        firsts[name] = fs
        if kind == "SOXX":
            extra = " ".join(str(d.date()) for d in fs.date)
        else:
            extra = (f"{fs.series.nunique() if len(fs) else 0} series, {fs.date.nunique() if len(fs) else 0} dates, "
                     f"2025={int((fs.date < '2026-01-01').sum()) if len(fs) else 0} "
                     f"2026H1={int(((fs.date >= '2026-01-01') & (fs.date <= '2026-05-29')).sum()) if len(fs) else 0} "
                     f"2026H2={int((fs.date >= '2026-06-01').sum()) if len(fs) else 0}")
        print(f"  {name:18s} raw={raw:6d} days={days:4d} eval={e1:6d} eval·in_dd={e2:6d} first={len(fs):4d}  {extra}")
    return firsts


def soxx_spotcheck(rules: dict[str, pd.DataFrame], fr: dict, d: dict, mdiag: pd.DataFrame, ev: dict,
                   C: pd.Series, firsts: dict) -> None:
    idx = fr["in_dd"].index
    c = C.to_numpy(float)
    fset = {n: set(f.date) for n, f in firsts.items()}
    ab = {"c_up5_since5": "s5", "c_up5_ma20": "ma", "c_vix30crush_up5": "vx", "c_thrust25_50_up5": "th",
          "c_newsS_up3": "nS", "c_hl_tsnorm": "hl"}
    print("\n== SOXX 알려진 저점 주변 (저점 −3 ~ +30 세션). 규칙 약어 s5 ma vx th nS hl, 대문자 = 채택 첫 신호 ==")
    for day in KNOWN_LOWS:
        p = idx.get_loc(pd.Timestamp(day))
        pk, l = d["PK"][p, 0], d["LL"][p, 0]
        print(f"\n저점 {day}: C={c[p]:.2f}, 60일 고점 {idx[pk].date()} {c[pk]:.2f}, 하락 저점 l={idx[l].date()} "
              f"Ldd={fr['L60'].iat[p, 0]:.2f}, dd60={fr['dd60'].iat[p, 0]:+.3f}, in_dd={bool(fr['in_dd'].iat[p, 0])}")
        for q in range(max(p - 3, 0), min(p + 31, len(idx))):
            t = idx[q]
            g = lambda k: mdiag[k].get(t, np.nan)
            fired = [(ab[n].upper() if t in fset[n] else ab[n]) for n, s in rules.items() if bool(s.iat[q, 0])]
            hs = d["hl_s"][q, 0]
            print(f"  {t.date()} C={c[q]:7.2f} up={fr['up'].iat[q, 0]:+.3f} t-l={d['since'].iat[q, 0]:3d} "
                  f">ma20={int(c[q] > d['sma20'].iat[q, 0]) if pd.notna(d['sma20'].iat[q, 0]) else '-'} "
                  f"in_dd={int(fr['in_dd'].iat[q, 0])} "
                  f"VIX={g('VIX'):6.2f}/mx{g('VIXmx10'):6.2f} R={g('R'):.3f}/mx{g('Rmx10'):.3f} "
                  f"pct20={g('pct20'):.2f}/mn{g('pctmn10'):.2f} "
                  f"ev[vx{int(ev['vix'].get(t, False))}ts{int(ev['ts'].get(t, False))}th{int(ev['thr'].get(t, False))}S{int(ev['S'].get(t, False))}] "
                  f"rec[vx{int(d['rc']['vix'].iat[q, 0])}ts{int(d['rc']['ts'].iat[q, 0])}th{int(d['rc']['thr'].iat[q, 0])}S{int(d['rc']['S'].iat[q, 0])}] "
                  f"hl_s={idx[hs].date() if hs >= 0 else '-'} hl={int(d['hl'].iat[q, 0])} "
                  f"fires={','.join(fired) or '-'}")
        line = []
        for n, s in rules.items():
            okq = s.iloc[p:, 0] & fr["in_dd"].iloc[p:, 0]
            fd = okq[okq].index[:1]
            if len(fd):
                q = idx.get_loc(fd[0])
                line.append(f"{n}:{fd[0].date()}(+{q - p}d, up={fr['up'].iat[q, 0]:+.3f})")
            else:
                line.append(f"{n}:-")
        print("  저점 뒤 in_dd 첫 발생: " + "  ".join(line))


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
    pd.set_option("display.width", 320)
    m = bc_lib.load_market()
    notes = bc_lib.load_notes()
    _, C_k = bc_lib.load_stocks()
    assert C_k.shape[1] == 82, C_k.shape
    ev, mdiag = market_events(m, C_k, notes)

    fr_s = bc_lib.soxx_frames()
    C_s = bc_lib.load_soxx()[["C"]].rename(columns={"C": "SOXX"})
    assert fr_s["in_dd"].index.equals(m.index)
    fr_k = bc_lib.stock_frames()
    assert fr_k["in_dd"].index.isin(m.index).all()
    data_report(m, C_k, notes, ev, mdiag, fr_k)

    rules_s, d_s = build_rules(C_s, fr_s, ev)
    rules_k, d_k = build_rules(C_k, fr_k, ev)
    cross_family_check(C_s, fr_s, C_k, fr_k, m, notes, ev, d_s, d_k)
    for kind, C, rules in (("SOXX", C_s, rules_s), ("STK", C_k, rules_k)):
        lp = loop_rules(C, m, C_k, notes)
        for n in RULES:
            assert (rules[n].to_numpy() == lp[n]).all(), f"{kind} {n} 독립 재계산 불일치"
        print(f"독립 재계산(반복문) {kind} 6개 규칙 일치")
    for kind, rules in (("SOXX", rules_s), ("STK", rules_k)):          # 시장 자료 끝 뒤 사건형 신호 점검
        last = m["^VIX"].last_valid_index()
        for n in ("c_vix30crush_up5", "c_hl_tsnorm"):
            late = rules[n].loc[last + pd.Timedelta(days=1):]
            if late.to_numpy().any():
                print(f"  {kind} {n}: {last.date()} 뒤 신호 {int(late.to_numpy().sum())}칸(그 전 사건의 [t-9, t] 이월)")

    out = {}
    fs_s = sanity("SOXX", rules_s, fr_s)
    soxx_spotcheck(rules_s, fr_s, d_s, mdiag, ev, C_s["SOXX"], fs_s)
    out["SOXX"] = bc_lib.run_rules(rules_s, "SOXX", FAMILY, fr=fr_s, tag=f"{FAMILY}_SOXX")
    sanity("STK", rules_k, fr_k)
    for n in ("c_vix30crush_up5", "c_thrust25_50_up5", "c_newsS_up3", "c_hl_tsnorm"):
        rc_name = {"c_vix30crush_up5": "vix", "c_thrust25_50_up5": "thr", "c_newsS_up3": "S", "c_hl_tsnorm": "ts"}[n]
        rc = d_k["rc"][rc_name]
        assert (rc.to_numpy() == rc.iloc[:, [0]].to_numpy()).all(), n          # 시장 사건은 82개 열이 같은 값
    out["STK"] = bc_lib.run_rules(rules_k, "STK", FAMILY, fr=fr_k, tag=f"{FAMILY}_STK")

    # ------------------------------------------------------------ 사후(post-hoc) — 등록 판정 아님
    ph = {"SOXX": {}, "STK": {}}
    for kind, C, fr, rules, d in (("SOXX", C_s, fr_s, rules_s, d_s), ("STK", C_k, fr_k, rules_k, d_k)):
        ver = out[kind][1]
        for r in ver.loc[ver.L1, "rule"]:
            ph[kind][f"{r}_lag1"] = rules[r].shift(1, fill_value=False)
            nb = neighbors(r, C, fr, d)
            if not nb:
                print(f"[사후] {kind} {r}: 이웃 값 변형 정의 없음")
            ph[kind].update(nb)
    bad = (m["VIX"] - m["^VIX"]).abs() > 0.05
    vix_fix = m["^VIX"].copy()
    vix_fix[bad] = m.loc[bad, "VIX"]
    ev_fix, _ = market_events(m, C_k, notes, vix=vix_fix)
    dts = ev_fix["ts"] != ev["ts"]
    print(f"\n[사후] ^VIX→패널 VIX 로 바꾼 날 {', '.join(str(x.date()) for x in bad[bad].index)}; E_ts 가 달라지는 날: "
          + (", ".join(str(x.date()) for x in dts[dts].index) or "없음"))
    for kind, C, fr, rules in (("SOXX", C_s, fr_s, rules_s), ("STK", C_k, fr_k, rules_k)):
        r_fix, _ = build_rules(C, fr, ev_fix)
        diff = r_fix["c_hl_tsnorm"] != rules["c_hl_tsnorm"]
        print(f"[사후] {kind} c_hl_tsnorm 패널 VIX 판과 다른 칸 {int(diff.to_numpy().sum())}")
        if diff.to_numpy().any():
            ph[kind]["c_hl_tsnorm_panelvix"] = r_fix["c_hl_tsnorm"]
    for kind, fr in (("SOXX", fr_s), ("STK", fr_k)):
        if ph[kind]:
            print(f"\n[사후] {kind}: " + ", ".join(ph[kind]))
            sanity(kind, ph[kind], fr)
            out[f"{kind}_posthoc"] = bc_lib.run_rules(ph[kind], kind, FAMILY + "_posthoc", fr=fr,
                                                      tag=f"{FAMILY}_{kind}_posthoc")
        else:
            print(f"\n[사후] {kind}: 사후 계산 없음(L1 통과 규칙 없음, 패널 VIX 판 차이 없음)")

    print_results(out)


if __name__ == "__main__":
    main()
