"""갈래 7 — 헤지 실행·계좌 (BRIEF 갈래 7, 2026-10-14).

사용
    py b7_hedge.py            # 1시간봉 캐시(없으면 yfinance) → (c) 판정 칸 → (a) 인버스 실행 → (b) 계좌 → results/b7_hedge*
    py b7_hedge.py fetch      # 1시간봉만 받기

판정 로직은 새로 만들지 않는다: 사건표·3분위 칸·evaluate_cells는 harness.py 그대로, 계좌 겹치기·지표는
../../stop_research/code/harness.py(overlay·stats·select_front) 그대로 쓴다.

정의 해석(결과를 보기 전에 정함. BRIEF가 열어 둔 부분)

(a) ETF 이탈 → 실제 인버스 매수
  - 사건: harness 사건표의 ETF 사건 중 SOXX·SMH·QQQ·SPY, N = 20·40·60, 이탈일 2025-01-01~2026-09-29.
  - 짝(기초 → 인버스, 배수): SOXX→SOXS(3), SMH→SOXS(3), QQQ→SQQQ(3)·QID(2)·PSQ(1), SPY→SH(1). 6짝.
  - 진입 3가지
      E1 t+1 정규장 시가(일봉 Open).
      E2 t+1 프리장 첫 1시간봉 시가: yfinance 1h prepost(04:00 ET 봉, 서머타임 KST 17:00). 그날 09:30 전 봉 중
         시가=고가=저가=종가가 아닌 첫 봉의 시가(yfinance 프리장 봉은 거래량이 모두 0으로 와서 거래량으로 거를 수 없다
         — 받은 뒤 확인, 결과 보기 전 수정). 없으면 '못 구함'(사건에서 뺀다). 730일 한도(2024-10 이후)라 2025~2026은 모두 안.
      E3 t+2 시가(확인 뒤): 기초 종가(t+1) ≤ L일 때만 진입. t+1에 이미 L 위로 돌아오면 거래 없음.
         '사건당'(거래 없음 = 0)과 '거래당'(진입한 것만) 둘 다 적는다. 고르기는 사건당으로 한다.
  - 청산 5가지(신호는 모두 기초 ETF 일봉, 체결은 인버스 가격)
      X1 되돌림(BRIEF 기본): 신호 시작일..t+9 기초 종가 > L → 다음 날 시가, 아니면 t+10 종가.
      X2·X3·X4 시간: t+5·t+10·t+20 종가(손절 없음).
      X5 추적: 진입일부터 기초 최저 저가 m_k, 기초 종가(k) > m_k + 1.5·ATR20(t−1) → 다음 날 시가, 아니면 t+20 종가.
      신호 시작일 = E1·E2는 t+1, E3는 t+2(진입일).
  - 손익 = 인버스 청산가 / 진입가 − 1 − 0.10%(왕복 스프레드). 대차 비용 없음(인버스는 대차가 필요 없다).
  - 대조: 같은 창(E1)의 기초 공매도 손익(−1배, 비용 뒤), 추적 차 = 인버스 총손익 − 배수 × 기초 공매도 총손익.
    E1·X1의 기초 공매도는 harness short_ret_h10과 같아야 한다(자체 점검).
  - 무작위 날 기준선: 구간의 모든 거래일 d를 t로 보고 E1 진입 → X2~X4 시간 청산의 인버스 손익 평균(이탈과 무관한 보유값).
  - 고르기(사전): 3배 짝(SOXX→SOXS, SMH→SOXS, QQQ→SQQQ)을 N과 함께 모은 앞 구간 사건당 평균이 가장 높은
    (진입, 청산) 1개 → 뒤 구간·2025 확인. 앞 구간 사건이 짝·N마다 0~2건이라 BRIEF 칸 판정(켜짐 ≥ 15건,
    날짜 ≥ 8일)을 걸 수 없다 → 기술 통계(날짜 묶음 t 포함)만.

(b) 데스크 7종목 계좌 + 이탈 시 인버스 x%
  - 기본 경로: stop_research/cache/base_2026.json(2026-01-02~10-09, 데스크 현행 규칙)과 base_hold.json(2025 보조, 주간 PDF).
    일별 수익 = 시가 k → 시가 k+1(stop_research harness와 같음).
  - 헤지 창: 계기 사건마다 t+1 시가에 주식 x를 팔고 인버스 x를 산다(합계 ≤ 1, 레버리지 없음). 청산은 X1 되돌림
    (기초 종가 > L → 다음 시가). 시간 청산은 계좌 경로가 시가 → 시가라 t+10 종가 대신 t+11 시가로 한다.
    창이 겹치면 합집합(비중은 x 그대로). 자료 끝까지 안 끝난 창은 끝까지 켠다.
  - 계기: SQ = SOXX 또는 QQQ 사건(과제의 'SOXX 또는 QQQ 이탈'), ALL7 = ETF 7개(QQQ·SPY·SOXX·SMH·IWM·XLK·IGV)
    사건 전부('모든 이탈 헤지'), SEL = SQ 사건 중 (c)의 앞 구간 1위 칸이 켜진 것(선별, 그 칸의 N만).
  - 상품: SOXS, SQQQ, 짝(SOXX·SMH 계기 → SOXS, QQQ·XLK·IGV → SQQQ, SPY·IWM → SH; 여러 상품이 동시에 켜지면 x를 나눔).
  - x ∈ {10%, 20%, 33%}, N ∈ {20, 40, 60}. 비용: 편도 회전율 1단위당 0.05%(= 주식·인버스 각 왕복 0.10%).
  - 지표: 수익·최대 낙폭(2026 전체·앞·뒤, 7월 쇼크 6/22~7/29, 2025). 앞 고르기는 stop_research select_front
    (앞 자산 ≥ 기본의 90%, 앞 낙폭이 1%p 이상 얕은 것 중 앞 낙폭 최소)를 그대로 쓴다.

(c) 판정 칸(BRIEF 칸 = 특징 × 3분위 위/아래 × N): 사건표(종목+ETF, N = 20·40·60)에 ETF 상태 특징 8개를 붙여
    harness.tercile_cells(by=N) → harness.evaluate_cells(기본 인자: short_ret_h10, 층 N·kind, 순열 200, 귀무 200).
    특징은 모두 이탈일 t 종가(미 동부 16:00)에 알 수 있는 일봉 값이고, 사건의 N과 같은 N으로 ETF 자신의 박스를 잰다.
      soxx_brk   SOXX 종가(t) < SOXX L_N(t)                    (날짜 고정, 이진)
      qqq_brk    QQQ 종가(t) < QQQ L_N(t)
      spy_brk    SPY 종가(t) < SPY L_N(t)
      sq_brk     soxx_brk 또는 qqq_brk                          (계좌 헤지 계기와 같은 상태)
      n_etf_brk  ETF 7개 중 종가 < 자기 L_N인 개수(0~7)
      soxx_gap_L (SOXX 종가 − L_N) / ATR20(t−1)                  (음수 = 박스 아래)
      qqq_gap_L  (QQQ 종가 − L_N) / ATR20(t−1)
      inv_dvol   (SOXS+SQQQ 거래대금(t)) / 직전 20일 평균           (인버스 수요 급증)
    모두 날짜 고정 특징이라 국면형 거짓 양성 점검(pass_shift·null_shift)을 함께 적는다.
"""
from __future__ import annotations

import runpy
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
import harness as H  # noqa: E402

# stop_research 하네스(같은 모듈 이름 'harness'라 경로로 읽는다). overlay·stats·select_front·base를 그대로 쓴다.
SR = SimpleNamespace(**runpy.run_path(str(H.ANALYSIS / "stop_research/code/harness.py"), run_name="sr_harness"))

RES = H.RESULTS
ONE_H = RES / "b7_hedge_1h.parquet"
INV = ["SOXS", "SQQQ", "QID", "PSQ", "SH"]
LEV = {"SOXS": 3, "SQQQ": 3, "QID": 2, "PSQ": 1, "SH": 1}
PAIRS = [("SOXX", "SOXS"), ("SMH", "SOXS"), ("QQQ", "SQQQ"), ("QQQ", "QID"), ("QQQ", "PSQ"), ("SPY", "SH")]
PAIRS_3X = [("SOXX", "SOXS"), ("SMH", "SOXS"), ("QQQ", "SQQQ")]
ENTRIES = ["E1_t+1시가", "E2_t+1프리장첫봉", "E3_t+2시가(확인)"]
EXITS = ["X1_되돌림", "X2_시간t+5", "X3_시간t+10", "X4_시간t+20", "X5_추적1.5ATR"]
TRAIL_K = 1.5
EXEC_START, EXEC_END = "2025-01-01", H.BACK[1]
PERIODS = {"앞": H.FRONT, "뒤": H.BACK, "2025": H.AUX}

FEATS = ["soxx_brk", "qqq_brk", "spy_brk", "sq_brk", "n_etf_brk", "soxx_gap_L", "qqq_gap_L", "inv_dvol"]
FEAT_DESC = {
    "soxx_brk": "SOXX 종가(t) < SOXX 박스 하단 L_N(t)",
    "qqq_brk": "QQQ 종가(t) < QQQ 박스 하단 L_N(t)",
    "spy_brk": "SPY 종가(t) < SPY 박스 하단 L_N(t)",
    "sq_brk": "SOXX 또는 QQQ 종가(t) < 자기 L_N(t)",
    "n_etf_brk": "ETF 7개 중 종가(t) < 자기 L_N(t)인 개수",
    "soxx_gap_L": "(SOXX 종가(t) − SOXX L_N) ÷ SOXX ATR20(t−1)",
    "qqq_gap_L": "(QQQ 종가(t) − QQQ L_N) ÷ QQQ ATR20(t−1)",
    "inv_dvol": "(SOXS+SQQQ 거래대금(t)) ÷ 직전 20거래일 평균",
}
KNOWN = "이탈일 t 정규장 종가(미 동부 16:00 = KST 05:00 서머타임) 일봉 — t+1 시가 진입 전에 앎"

ACC_P = {"2026전체": ("2026-01-02", "2026-10-09"), "2026앞": ("2026-01-02", "2026-05-29"),
         "2026뒤": ("2026-06-01", "2026-10-09"), "7월쇼크": ("2026-06-22", "2026-07-29"),
         "2025": ("2025-01-01", "2025-12-31")}
ACC_X = (0.10, 0.20, 0.33)
ACC_PROD = ("SOXS", "SQQQ", "짝")
MATCH = {"SOXX": "SOXS", "SMH": "SOXS", "QQQ": "SQQQ", "XLK": "SQQQ", "IGV": "SQQQ", "SPY": "SH", "IWM": "SH"}


# ───────────────────────────── 1시간봉 ─────────────────────────────
def fetch_1h(force=False):
    if ONE_H.exists() and not force:
        return pd.read_parquet(ONE_H)
    import yfinance as yf
    start = (pd.Timestamp.today().normalize() - pd.Timedelta(days=729)).strftime("%Y-%m-%d")
    end = (pd.Timestamp(H.END) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    rows = []
    for s in INV:
        x = yf.Ticker(s).history(interval="1h", start=start, end=end, prepost=True, auto_adjust=False)
        if x.empty:
            continue
        x.index = x.index.tz_convert("America/New_York").tz_localize(None)
        x.index.name = "ts"
        rows.append(x[["Open", "High", "Low", "Close", "Volume"]].reset_index().assign(sym=s))
    df = pd.concat(rows, ignore_index=True)
    df.to_parquet(ONE_H)
    return df


def premarket_table(h1):
    """(sym, 날짜) → 09:30 전 첫 정상 봉(O=H=L=C 한 값 봉 제외)의 시가·시각. 그리고 정규장 09:30 봉 시가(단위 점검용).
    yfinance 프리장 봉의 거래량은 모두 0이라 거래량으로는 거르지 않는다."""
    h1 = h1.copy()
    h1["day"] = h1["ts"].dt.normalize()
    tm = h1["ts"].dt.hour * 60 + h1["ts"].dt.minute
    flat = (h1["Open"] == h1["High"]) & (h1["High"] == h1["Low"]) & (h1["Low"] == h1["Close"])
    pm = h1[(tm < 570) & ~flat & h1["Open"].notna()].sort_values("ts")
    first = pm.groupby(["sym", "day"]).first()
    reg = h1[tm == 570].set_index(["sym", "day"])["Open"]
    return first, reg


# ───────────────────────────── (c) 특징 ─────────────────────────────
def etf_state(P, N):
    out = {}
    for s in H.ETFS:
        o, h, l, c = (P.col(f, s) for f in "OHLC")
        Ls, _, atr = H.box_levels(o, h, l, c, N)
        ok = ~np.isnan(Ls) & ~np.isnan(c)
        brk = np.where(ok, (c < np.where(ok, Ls, 0)).astype(float), np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            gap = np.where(ok & (atr > 0), (c - Ls) / np.where(atr > 0, atr, 1), np.nan)
        out[s] = (brk, gap, Ls, atr)
    return out


def add_features(ev, P):
    ev = ev.copy()
    dv = P.col("C", "SOXS") * P.col("V", "SOXS") + P.col("C", "SQQQ") * P.col("V", "SQQQ")
    base = pd.Series(dv).shift(1).rolling(20, min_periods=20).mean().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        inv_dvol = np.where(base > 0, dv / base, np.nan)
    for f in FEATS:
        ev[f] = np.nan
    for N in H.NS:
        st = etf_state(P, N)
        m = (ev["N"] == N).to_numpy()
        i = ev.loc[m, "i"].to_numpy(int)
        sb, qb, pb = st["SOXX"][0], st["QQQ"][0], st["SPY"][0]
        sq = np.where(np.isnan(sb) | np.isnan(qb), np.nan, np.fmax(sb, qb))
        allb = np.vstack([st[s][0] for s in H.ETFS])
        n_etf = np.where((~np.isnan(allb)).all(0), np.nansum(allb, 0), np.nan)
        vals = {"soxx_brk": sb, "qqq_brk": qb, "spy_brk": pb, "sq_brk": sq, "n_etf_brk": n_etf,
                "soxx_gap_L": st["SOXX"][1], "qqq_gap_L": st["QQQ"][1], "inv_dvol": inv_dvol}
        for f, v in vals.items():
            ev.loc[m, f] = v[i]
    return ev


def cell_feature(name):
    rest = name.split("|", 1)[1]
    return next(x for x in sorted(FEATS, key=len, reverse=True) if rest.startswith(x))


def cell_definition(name):
    """칸 이름 'N20|soxx_gap_L≥0.12(상위3분위)' → 사람이 읽는 정의."""
    grp, rest = name.split("|", 1)
    f = cell_feature(name)
    cond = rest[len(f):]
    if cond.startswith("=1"):
        txt = f"[{FEAT_DESC[f]}]가 참"
    elif cond.startswith("=0"):
        txt = f"[{FEAT_DESC[f]}]가 거짓"
    else:
        txt = f"[{FEAT_DESC[f]}] {cond.split('(')[0]}({cond.split('(')[1].rstrip(')')}, 문턱은 앞 구간 {grp} 사건 분포)"
    return f"{grp} 박스 이탈 사건(종목+ETF, 이탈일 t) 중 {txt}인 사건 = 켜짐"


# ───────────────────────────── (a) 인버스 실행 ─────────────────────────────
def simulate(P, ev_row, inv, entry, ex, pm_first):
    """사건 1건 → dict(인버스 손익, 기초 공매도 손익, 진입·청산 정보). 거래 없음은 traded=False, 자료 부족은 None."""
    u = ev_row.sym
    i, Lv, atr = int(ev_row.i), float(ev_row.L), float(ev_row.atr20)
    uo, ul, uc = P.col("O", u), P.col("L", u), P.col("C", u)
    io, ic = P.col("O", inv), P.col("C", inv)
    n = len(uc)
    if i + 2 >= n:
        return None
    pm_ts = pd.NaT
    if entry.startswith("E3"):
        if np.isnan(uc[i + 1]):
            return None
        if uc[i + 1] > Lv:
            return {"traded": False, "inv_ret": 0.0, "proxy_ret": 0.0, "track": np.nan, "entry_px": np.nan,
                    "exit_px": np.nan, "entry_day": pd.NaT, "exit_day": pd.NaT, "exit_why": "확인실패(t+1 되돌림)",
                    "hold_days": 0, "pm_ts": pm_ts}
        e_idx, px_in, u_in = i + 2, io[i + 2], uo[i + 2]
    elif entry.startswith("E2"):
        e_idx = i + 1
        key = (inv, P.cal[i + 1])
        if key not in pm_first.index:
            return None
        rec = pm_first.loc[key]
        px_in, u_in, pm_ts = float(rec["Open"]), np.nan, rec["ts"]
    else:
        e_idx, px_in, u_in = i + 1, io[i + 1], uo[i + 1]
    if np.isnan(px_in):
        return None
    chk0 = e_idx
    x_idx, x_type, why = None, None, None
    if ex.startswith("X1"):
        for k in range(chk0, i + 10):
            if k >= n or np.isnan(uc[k]):
                return None
            if uc[k] > Lv:
                x_idx, x_type, why = k + 1, "open", "되돌림"
                break
        if x_idx is None:
            x_idx, x_type, why = i + 10, "close", "시간t+10"
    elif ex.startswith("X5"):
        m = np.inf
        for k in range(chk0, i + 20):
            if k >= n or np.isnan(uc[k]):
                return None
            m = min(m, ul[k])
            if uc[k] > m + TRAIL_K * atr:
                x_idx, x_type, why = k + 1, "open", "추적"
                break
        if x_idx is None:
            x_idx, x_type, why = i + 20, "close", "시간t+20"
    else:
        h = {"X2": 5, "X3": 10, "X4": 20}[ex[:2]]
        x_idx, x_type, why = i + h, "close", f"시간t+{h}"
    if x_idx >= n:
        return None
    px_out = io[x_idx] if x_type == "open" else ic[x_idx]
    u_out = uo[x_idx] if x_type == "open" else uc[x_idx]
    if np.isnan(px_out):
        return None
    g_inv = px_out / px_in - 1
    g_px = (u_in - u_out) / u_in if not np.isnan(u_in) else np.nan
    return {"traded": True, "inv_ret": g_inv - H.COST, "proxy_ret": g_px - H.COST if not np.isnan(g_px) else np.nan,
            "track": g_inv - LEV[inv] * g_px if not np.isnan(g_px) else np.nan,
            "entry_px": px_in, "exit_px": px_out, "entry_day": P.cal[e_idx], "exit_day": P.cal[x_idx],
            "exit_why": why, "hold_days": x_idx - e_idx + (1 if x_type == "close" else 0), "pm_ts": pm_ts}


def run_exec(P, ev_all, pm_first):
    e = ev_all[(ev_all["kind"] == "etf") & ev_all["sym"].isin({u for u, _ in PAIRS})
               & (ev_all["t"] >= EXEC_START) & (ev_all["t"] <= EXEC_END)]
    rows = []
    for r in e.itertuples():
        for u, inv in PAIRS:
            if u != r.sym:
                continue
            for en in ENTRIES:
                for ex in EXITS:
                    s = simulate(P, r, inv, en, ex, pm_first)
                    base = {"sym": r.sym, "inv": inv, "lev": LEV[inv], "N": r.N, "t": r.t, "i": r.i, "L": r.L,
                            "cls": r.cls, "episode": r.episode, "entry": en, "exit": ex,
                            "harness_h10": r.short_ret_h10}
                    rows.append({**base, "ok": False} if s is None else {**base, "ok": True, **s})
    df = pd.DataFrame(rows)
    df["period"] = None
    for p, (a, b) in PERIODS.items():
        df.loc[(df["t"] >= a) & (df["t"] <= b), "period"] = p
    return df


def summarize_exec(df, pairs=None):
    d = df[df["ok"]]
    if pairs is not None:
        keep = set(pairs)
        d = d[[(a, b) in keep for a, b in zip(d["sym"], d["inv"])]]
    out = []
    for (en, ex, p), g in d.groupby(["entry", "exit", "period"]):
        m, t, n, G = H.cluster_mean(g["inv_ret"], g["t"])
        tr = g[g["traded"].astype(bool)]
        out.append({"entry": en, "exit": ex, "period": p, "n_events": n, "n_dates": G, "n_traded": len(tr),
                    "mean_per_event": m, "t_date": t, "mean_per_trade": tr["inv_ret"].mean() if len(tr) else np.nan,
                    "win_rate_trade": (tr["inv_ret"] > 0).mean() if len(tr) else np.nan,
                    "proxy_mean": tr["proxy_ret"].mean() if len(tr) else np.nan,
                    "track_mean": tr["track"].mean() if len(tr) else np.nan,
                    "hold_days_mean": tr["hold_days"].mean() if len(tr) else np.nan})
    return pd.DataFrame(out)


def random_day_baseline(P):
    rows = []
    cal = P.cal
    for u, inv in PAIRS:
        io, ic = P.col("O", inv), P.col("C", inv)
        for p, (a, b) in PERIODS.items():
            idx = np.flatnonzero((cal >= a) & (cal <= b))
            for h in (5, 10, 20):
                ok = idx[idx + h < len(cal)]
                r = ic[ok + h] / io[ok + 1] - 1 - H.COST
                r = r[~np.isnan(r)]
                rows.append({"sym": u, "inv": inv, "period": p, "h": h, "n_days": len(r),
                             "mean": float(r.mean()) if len(r) else np.nan,
                             "win": float((r > 0).mean()) if len(r) else np.nan})
    return pd.DataFrame(rows)


# ───────────────────────────── (b) 계좌 ─────────────────────────────
def hedge_windows(P, rows):
    """사건 행들 → 거래일별 켜짐(0/1) Series(index 'YYYY-MM-DD'). t+1 시가 ~ 되돌림 다음 시가 전날(시간 청산은 t+11 시가)."""
    on = np.zeros(len(P.cal), bool)
    for r in rows.itertuples():
        c = P.col("C", r.sym)
        i, n = int(r.i), len(c)
        x = i + 11
        for k in range(i + 1, i + 10):
            if k >= n:
                break
            if c[k] > r.L:
                x = k + 1
                break
        on[i + 1:min(x, n)] = True
    return pd.Series(on.astype(float), index=P.cal.strftime("%Y-%m-%d"))


def account_variant(P, rows, prod, x, cost=H.COST / 2):
    """rows: 계기 사건. prod: 'SOXS'|'SQQQ'|'짝'. 반환 {기본 경로 이름: (수익 Series, 켜짐 Series)}."""
    if prod == "짝":
        per = {}
        for s, g in rows.groupby("sym"):
            per.setdefault(MATCH[s], []).append(g)
        ons = {p: hedge_windows(P, pd.concat(gs)) for p, gs in per.items()}
    else:
        ons = {prod: hedge_windows(P, rows)}
    out = {}
    for bn in ("2026", "hold"):
        b = SR.base(bn)
        o = {p: s.reindex(b.days).fillna(0.0) for p, s in ons.items()}
        k = sum(o.values())
        anyon = (k > 0).astype(float)
        hedge = {p: x * s / k.where(k > 0, 1.0) for p, s in o.items()}
        e = 1.0 - x * anyon
        out[bn] = (SR.overlay(b, e, hedge, cost=cost), anyon)
    return out


def acc_metrics(res):
    m = {}
    for pn, (lo, hi) in ACC_P.items():
        bn = "hold" if pn == "2025" else "2026"
        r, on = res[bn]
        st = SR.stats(r, lo, hi)
        oo = on[(on.index >= lo) & (on.index <= hi)]
        m[pn] = {"수익": st["수익"], "낙폭": st["낙폭"], "헤지일": int(oo.sum())}
    return m


def run_account(P, ev_all, sel_name, sel_series):
    etf = ev_all[ev_all["kind"] == "etf"]
    variants = []
    base = {}
    for pn, (lo, hi) in ACC_P.items():
        st = SR.stats(SR.base("hold" if pn == "2025" else "2026").ret, lo, hi)
        base[pn] = {"수익": st["수익"], "낙폭": st["낙폭"], "헤지일": 0}
    variants.append({"name": "헤지 없음", "trigger": "-", "N": None, "prod": "-", "x": 0.0, "m": base, "n_trig": 0})
    for trig in ("SQ", "ALL7"):
        for N in H.NS:
            g = etf[etf["N"] == N]
            rows = g[g["sym"].isin(["SOXX", "QQQ"])] if trig == "SQ" else g
            for prod in ACC_PROD:
                for x in ACC_X:
                    res = account_variant(P, rows, prod, x)
                    variants.append({"name": f"{trig}|N{N}|{prod}|x={x:.2f}", "trigger": trig, "N": N, "prod": prod,
                                     "x": x, "m": acc_metrics(res), "n_trig": len(rows)})
    if sel_name is not None:
        Nsel = int(sel_name.split("|")[0][1:])
        g = etf[(etf["N"] == Nsel) & etf["sym"].isin(["SOXX", "QQQ"])]
        on = (pd.Series(sel_series).reindex(g.index) == 1).to_numpy()
        rows = g[on]
        for prod in ACC_PROD:
            for x in ACC_X:
                mm = acc_metrics(account_variant(P, rows, prod, x)) if len(rows) else base
                variants.append({"name": f"SEL|N{Nsel}|{prod}|x={x:.2f}", "trigger": "SEL", "N": Nsel, "prod": prod,
                                 "x": x, "m": mm, "n_trig": len(rows), "sel_cell": sel_name})
    return variants


def acc_front_pick(variants, keep=0.90, min_gain=0.01):
    """stop_research select_front를 그대로 쓴다(행 모양만 맞춘다)."""
    def row(v):
        return {"name": v["name"], "2026": {"앞": v["m"]["2026앞"]}, "v": v}
    pick = SR.select_front([row(v) for v in variants[1:]], row(variants[0]), keep, min_gain)
    return pick["v"] if pick else None


# ───────────────────────────── 실행 ─────────────────────────────
def main():
    t0 = time.time()
    P = H.load_panel()
    h1 = fetch_1h()
    pm_first, reg_open = premarket_table(h1)

    # 1시간봉 단위 점검: 정규장 09:30 봉 시가 / 일봉 시가
    chk = []
    for s in INV:
        o = pd.Series(P.col("O", s), index=P.cal)
        r = reg_open.xs(s, level="sym")
        m = pd.concat([r, o], axis=1, keys=["h", "d"], sort=True).dropna()
        q = (m["h"] / m["d"] - 1).abs()
        chk.append({"sym": s, "days": len(m), "median_absdiff": float(q.median()), "p99": float(q.quantile(0.99)),
                    "n_gt2pct": int((q > 0.02).sum())})
    unit_check = pd.DataFrame(chk)

    # (c) 판정 칸
    ev_all = add_features(H.load_events_all(), P)
    cells = H.tercile_cells(ev_all, FEATS, by=("N",))
    tab, summ = H.evaluate_cells(ev_all, cells)
    tab["feature"] = [cell_feature(c) for c in tab["cell"]]
    tab["definition"] = [cell_definition(c) for c in tab["cell"]]
    tab["known_when"] = KNOWN
    tab.to_csv(RES / "b7_hedge_cells.csv", index=False, encoding="utf-8-sig")
    sel = tab[tab["selected"]].sort_values("rank")
    sel_name = sel.iloc[0]["cell"] if len(sel) else None
    sel_series = dict(cells)[sel_name] if sel_name else None

    # (a) 인버스 실행
    ex = run_exec(P, ev_all, pm_first)
    k = ex[(ex["entry"] == ENTRIES[0]) & (ex["exit"] == EXITS[0]) & ex["ok"]]
    selfcheck = float(np.nanmax(np.abs(k["proxy_ret"] - k["harness_h10"]))) if len(k) else np.nan
    ex.to_csv(RES / "b7_hedge_exec_events.csv", index=False, encoding="utf-8-sig")
    s_all = summarize_exec(ex)
    s_3x = summarize_exec(ex, PAIRS_3X)
    s_pair = []
    for u, inv in PAIRS:
        x = summarize_exec(ex, [(u, inv)])
        x.insert(0, "pair", f"{u}->{inv}")
        s_pair.append(x)
    s_pair = pd.concat(s_pair, ignore_index=True)
    pd.concat([s_3x.assign(pool="3배 짝(SOXX·SMH→SOXS, QQQ→SQQQ)"), s_all.assign(pool="6짝 전체"),
               s_pair.assign(pool="짝별")], ignore_index=True).to_csv(RES / "b7_hedge_exec_summary.csv", index=False,
                                                                     encoding="utf-8-sig")
    fr = s_3x[s_3x["period"] == "앞"].sort_values("mean_per_event", ascending=False)
    pick_exec = (fr.iloc[0]["entry"], fr.iloc[0]["exit"]) if len(fr) else None
    rnd = random_day_baseline(P)
    rnd.to_csv(RES / "b7_hedge_random_baseline.csv", index=False, encoding="utf-8-sig")

    # (b) 계좌
    variants = run_account(P, ev_all, sel_name, sel_series)
    acc_rows = []
    for v in variants:
        r = {"name": v["name"], "trigger": v["trigger"], "N": v["N"], "prod": v["prod"], "x": v["x"],
             "n_trig_events": v.get("n_trig")}
        for pn, d in v["m"].items():
            r[f"{pn}_수익"], r[f"{pn}_낙폭"], r[f"{pn}_헤지일"] = d["수익"], d["낙폭"], d["헤지일"]
        acc_rows.append(r)
    acc = pd.DataFrame(acc_rows)
    acc.to_csv(RES / "b7_hedge_account.csv", index=False, encoding="utf-8-sig")
    pick_acc = acc_front_pick(variants)

    return {"tab": tab, "summ": summ, "cells": cells, "ex": ex, "s_3x": s_3x, "s_all": s_all, "s_pair": s_pair,
            "pick_exec": pick_exec, "rnd": rnd, "acc": acc, "variants": variants, "pick_acc": pick_acc,
            "selfcheck": selfcheck, "unit_check": unit_check, "sel_name": sel_name, "ev_all": ev_all, "P": P,
            "elapsed": time.time() - t0}


# ───────────────────────────── 보고서 ─────────────────────────────
def pct(x, nd=2, sign=True):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "못 구함"
    if not np.isfinite(x):
        return "못 구함"
    return f"{x * 100:+.{nd}f}%" if sign else f"{x * 100:.{nd}f}%"


def num(x, nd=2):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "못 구함"
    return f"{x:.{nd}f}" if np.isfinite(x) else "못 구함"


def windows_text(on, lo, hi):
    """켜짐 Series → '3/9, 3/31, 7/17~7/21' 꼴(구간 안)."""
    o = on[(on.index >= lo) & (on.index <= hi)]
    runs, start, prev = [], None, None
    for d, v in o.items():
        if v > 0.5 and start is None:
            start = d
        if v <= 0.5 and start is not None:
            runs.append((start, prev))
            start = None
        prev = d
    if start is not None:
        runs.append((start, prev))
    f = lambda d: f"{int(d[5:7])}/{int(d[8:10])}"
    return ", ".join(f(a) if a == b else f"{f(a)}~{f(b)}" for a, b in runs) or "없음"


def older_years(P, ev_all):
    """2022-06~2024 E1·X1 3배 짝 사건당 평균(다른 해 한 줄용)."""
    e = ev_all[(ev_all["kind"] == "etf") & ev_all["sym"].isin(["SOXX", "SMH", "QQQ"]) & (ev_all["t"] <= "2024-12-31")]
    vals, dts = [], []
    for r in e.itertuples():
        for u, inv in PAIRS_3X:
            if u == r.sym:
                s = simulate(P, r, inv, ENTRIES[0], EXITS[0], None)
                if s is not None:
                    vals.append(s["inv_ret"])
                    dts.append(r.t)
    return H.cluster_mean(vals, dts)


def write_report(o):
    P, tab, summ, ex = o["P"], o["tab"], o["summ"], o["ex"]
    s3, rnd, acc = o["s_3x"], o["rnd"], o["acc"]
    L = []
    g = lambda en, exn, p: s3[(s3["entry"] == en) & (s3["exit"] == exn) & (s3["period"] == p)].iloc[0]
    base_f, base_b, base_a = (g(ENTRIES[0], EXITS[0], p) for p in ("앞", "뒤", "2025"))
    n_var_pos = {p: int((s3[s3["period"] == p]["mean_per_event"] > 0).sum()) for p in ("앞", "뒤", "2025")}
    pe = o["pick_exec"]
    pf, pb, pa = (g(pe[0], pe[1], p) for p in ("앞", "뒤", "2025"))
    sel = tab[tab["selected"]].sort_values("rank")
    nul, nuls = summ["null"], summ.get("null_shift", {})
    # 7월 쇼크에서 이탈 신호가 하락의 몇 %를 지난 뒤였나(6/22 종가 고점 → 7/29 종가 저점, 과제가 정한 창)
    cal_s = P.cal.strftime("%Y-%m-%d")
    cpx = {s: pd.Series(P.col("C", s), index=cal_s) for s in ("SOXX", "QQQ")}
    shock = []
    for s in ("SOXX", "QQQ"):
        hi_, lo_ = cpx[s]["2026-06-22"], cpx[s]["2026-07-29"]
        evs = o["ev_all"][(o["ev_all"]["sym"] == s) & (o["ev_all"]["t"] >= "2026-06-22") & (o["ev_all"]["t"] <= "2026-07-29")]
        for r in evs.sort_values(["t", "N"]).itertuples():
            done = (hi_ - r.close_t) / (hi_ - lo_)
            shock.append(f"{s} N{r.N} {r.t.month}/{r.t.day}(하락의 {done * 100:.0f}% 지난 뒤)")
    om, ot, on_, oG = older_years(P, o["ev_all"])
    opp = tab[tab["cell"].str.contains(r"_brk=1\(hi\)|n_etf_brk≥", regex=True)]
    opp_neg_m = (opp["f_diff"] < 0) & (opp["b_diff"] < 0) & (opp["a_diff"] < 0)
    opp_tot, opp_neg = len(opp), int(opp_neg_m.sum())
    opp_list = opp.loc[opp_neg_m, "cell"].tolist()
    sp_ = o["s_pair"]
    worse = []
    for u, inv in [("SOXX", "SOXS"), ("SMH", "SOXS"), ("QQQ", "SQQQ"), ("SPY", "SH")]:
        x = sp_[(sp_["pair"] == f"{u}->{inv}") & (sp_["entry"] == ENTRIES[0]) & (sp_["exit"] == EXITS[2])].set_index("period")
        for p in ("앞", "뒤", "2025"):
            if p in x.index:
                rr = rnd[(rnd["sym"] == u) & (rnd["inv"] == inv) & (rnd["period"] == p) & (rnd["h"] == 10)].iloc[0]
                worse.append((p, bool(x.loc[p, "mean_per_event"] < rr["mean"])))
    worse_n, worse_tot = sum(w for _, w in worse), len(worse)
    worse26_n, worse26_tot = sum(w for p, w in worse if p != "2025"), sum(1 for p, _ in worse if p != "2025")
    e2w = {p: int(sum(g(ENTRIES[1], exn, p)["mean_per_event"] < g(ENTRIES[0], exn, p)["mean_per_event"] for exn in EXITS))
           for p in ("앞", "뒤", "2025")}
    # 계좌
    accb = acc.iloc[0]
    rowf = lambda nm: acc[acc["name"] == nm].iloc[0]
    etf = o["ev_all"][o["ev_all"]["kind"] == "etf"]
    win_sq = {N: windows_text(hedge_windows(P, etf[(etf["N"] == N) & etf["sym"].isin(["SOXX", "QQQ"])]),
                              "2026-01-01", "2026-10-09") for N in H.NS}

    L += ["# 갈래 7 — 헤지 실행·계좌: 박스 이탈 뒤 인버스로 벌 수 있었나 (2026-10-14)", "",
          "사전 등록 `../BRIEF.md` 갈래 7. 코드 `../code/b7_hedge.py`(정의 해석은 파일 머리에 결과 보기 전 고정), "
          "판정은 `harness.evaluate_cells`, 계좌는 `stop_research/code/harness.py`의 overlay·stats·select_front 그대로.", "",
          "## 결론", "",
          f"- **2026년 ETF 박스 이탈 뒤 실제 인버스 매수는 앞·뒤 구간 모두 진입 3 × 청산 5 = 15가지 전부 평균 손실이었다** "
          f"(사건당 평균이 플러스인 변형: 앞 {n_var_pos['앞']}/15, 뒤 {n_var_pos['뒤']}/15, 2025 {n_var_pos['2025']}/15). "
          f"기본(t+1 시가 진입·되돌림 손절)은 3배 짝(SOXX·SMH→SOXS, QQQ→SQQQ) 기준 앞 {pct(base_f['mean_per_event'])}"
          f"({base_f['n_events']}건/{base_f['n_dates']}일), 뒤 {pct(base_b['mean_per_event'])}({base_b['n_events']}건/"
          f"{base_b['n_dates']}일, 날짜 묶음 t {num(base_b['t_date'])}), 2025 {pct(base_a['mean_per_event'])}"
          f"({base_a['n_events']}건/{base_a['n_dates']}일).",
          f"- 앞 구간 1위 변형(사전 규칙) {pe[0]}·{pe[1]}: 앞 {pct(pf['mean_per_event'])}(10건 중 {pf['n_traded']}건만 진입), "
          f"뒤 {pct(pb['mean_per_event'])}(t {num(pb['t_date'])}), 2025 {pct(pa['mean_per_event'])} → 확인 실패.",
          f"- 이탈 뒤 인버스(t+1 시가 → t+10 종가)가 같은 구간 아무 날에나 산 인버스보다 나빴던 칸은 {worse_n}/{worse_tot}"
          f"(짝 4 × 구간 3 중 사건 있는 칸; 2026만 {worse26_n}/{worse26_tot}, (a)-3 표). 2026년 ETF 박스 이탈은 평균적으로 "
          "'진짜 하락의 시작'보다 되돌림 직전 자리에 가까웠다(기준선의 가짜 이탈 우세와 같은 방향).",
          f"- 7월 쇼크(SOXX 6/22 종가 고점 → 7/29 종가 저점 {pct(cpx['SOXX']['2026-07-29'] / cpx['SOXX']['2026-06-22'] - 1, 1)})에서 "
          f"ETF 이탈 신호는 늦었다(각 ETF의 6/22 종가 → 7/29 종가 하락폭 중 신호일 종가까지 지난 비율): {'; '.join(shock)}.",
          f"- 판정 칸 {summ['n_cells']}개(ETF 상태 특징 8개) 중 통과 **{summ['n_pass']}개**(선택편향 귀무 평균 "
          f"{num(nul['mean'], 3)}, 95% 분위 {nul['q95']:g}, p {num(nul['p_value'], 3)}; 원형 이동 귀무 평균 "
          f"{num(nuls.get('mean', np.nan), 3)}). 앞 상위 10칸은 모두 'ETF가 자기 박스를 **안** 깼거나 바닥에서 멀리 있는 날"
          "(=0, 0개, 거리 상위 3분위)의 이탈'이 덜 잃는다는 칸이었다. 뒤에서도 같은 방향(1위 N40 SOXX 안 깸: 차이 "
          f"{pct(sel.iloc[0]['b_diff'])}, t {num(sel.iloc[0]['b_t'])})이었지만 켜진 사건 손익이 여전히 "
          f"{pct(sel.iloc[0]['b_mean_on'])}라 통과 조건(켜진 사건 평균 > 0)을 못 넘었다. 반대로 말하면 "
          f"**ETF까지 함께 박스를 깬 날의 이탈이 더 크게 되돌았다**(ETF 이탈 상태 칸 {opp_tot}개 중 앞·뒤·2025 차이가 모두 음수 "
          f"{opp_neg}개) — 헤지를 켜고 싶은 날이 공매도·인버스에는 더 나쁜 날이었다.",
          f"- 계좌(데스크 7종목): 앞 고르기 규칙(앞 자산 ≥ 기본의 90%, 앞 낙폭 1%p 이상 개선)을 만족한 변형 "
          f"{'0개 → 앞 구간 탈락' if o['pick_acc'] is None else o['pick_acc']['name']}. 2026년 헤지가 켜진 날은 SOXX·QQQ "
          f"N20 {int(rowf('SQ|N20|SOXS|x=0.20')['2026전체_헤지일'])}일·N40 {int(rowf('SQ|N40|SOXS|x=0.20')['2026전체_헤지일'])}일"
          f"·N60 {int(rowf('SQ|N60|SOXS|x=0.20')['2026전체_헤지일'])}일뿐이다(되돌림이 하루이틀 만에 와서). 사후로는 N40 SOXS가 "
          f"뒤 구간·7월 쇼크를 줄였지만(x 33%: 뒤 {pct(rowf('SQ|N40|SOXS|x=0.33')['2026뒤_수익'], 1)} vs 헤지 없음 "
          f"{pct(accb['2026뒤_수익'], 1)}) 앞 구간에서 {pct(rowf('SQ|N40|SOXS|x=0.33')['2026앞_수익'] - accb['2026앞_수익'], 1)}p를 잃었고, "
          "뒤의 이득은 7월 말 한 창에서 나왔다.",
          f"- 다른 해(2022-06~2024) 같은 기본 변형: {pct(om)}({on_}건/{oG}일, t {num(ot)}). 장세가 바뀌면 숫자가 달라질 수 있다는 "
          "참고로만 둔다.", ""]

    L += ["## 시험 범위(선택편향 공개)", "",
          f"- 판정 칸: ETF 상태 특징 8개 × 3분위 위/아래(이진은 =1/=0) × N 3개 = **{summ['n_cells']}칸**(N60 soxx_brk는 앞 구간 값이 "
          "0 하나뿐이라 1칸). 고르기 조건(켜짐 ≥ 15건, 날짜 ≥ 8일, 앞 차이 > 0)을 만족한 칸 "
          f"{summ['n_eligible']}개 중 상위 {summ['n_selected']}개 확인.",
          f"- 실행 변형(기술 통계, BRIEF 판정 아님): 짝 6 × N 3 × 진입 3 × 청산 5 = 270칸, 고르기는 3배 짝·N을 모은 15가지 중 1개.",
          f"- 계좌 변형: 계기 2(SQ, ALL7) × N 3 × 상품 3 × x 3 = 54 + 선별 SEL(상품 3 × x 3) 9 = **63개** + 헤지 없음.",
          "- 앞선 박스·손절·레버리지 연구(7월 쇼크와 SOXS 헤지 결과)를 본 뒤의 탐색이라 선택편향이 쌓여 있다.",
          f"- 사건 수(판정 칸, 종목+ETF, N 20·40·60 합): 앞 {summ['events']['f']}건/{summ['dates']['f']}일, "
          f"뒤 {summ['events']['b']}건/{summ['dates']['b']}일, 2025 {summ['events']['a']}건/{summ['dates']['a']}일.", ""]

    # (a)
    L += ["## (a) ETF 이탈 → 실제 인버스", "",
          "진입: E1 t+1 정규장 시가, E2 t+1 프리장 첫 1시간봉(04:00 ET) 시가, E3 t+2 시가(t+1 종가가 L 아래일 때만). "
          "청산: X1 되돌림(종가 > L → 다음 시가, 아니면 t+10 종가), X2·X3·X4 t+5·10·20 종가, X5 진입 뒤 최저가 + 1.5·ATR20 위 종가 → 다음 시가(최대 t+20). "
          "손익은 인버스 가격, 비용 왕복 0.10%. '사건당'은 E3에서 진입 안 한 사건을 0으로 센다.", "",
          "### (a)-1 3배 짝(SOXX·SMH→SOXS, QQQ→SQQQ), N 20·40·60 합, 사건당 평균(날짜 묶음 t)", "",
          "| 진입 | 청산 | 앞 2026-01~05 | 뒤 2026-06~09 | 2025 | 앞 진입 건 | 뒤 진입 건 |", "|---|---|---:|---:|---:|---:|---:|"]
    for en in ENTRIES:
        for exn in EXITS:
            a, b, c = (g(en, exn, p) for p in ("앞", "뒤", "2025"))
            mark = " **(앞 1위)**" if (en, exn) == pe else ""
            L.append(f"| {en}{mark} | {exn} | {pct(a['mean_per_event'])} (t {num(a['t_date'])}) | "
                     f"{pct(b['mean_per_event'])} (t {num(b['t_date'])}) | {pct(c['mean_per_event'])} (t {num(c['t_date'])}) | "
                     f"{a['n_traded']}/{a['n_events']} | {b['n_traded']}/{b['n_events']} |")
    L += ["", f"사건 수: 앞 {base_f['n_events']}건/{base_f['n_dates']}일, 뒤 {base_b['n_events']}건/{base_b['n_dates']}일, "
          f"2025 {base_a['n_events']}건/{base_a['n_dates']}일(같은 날 N20·40·60이 함께 이탈하면 따로 센다. t는 날짜 묶음). "
          f"자체 점검: E1·X1의 기초 공매도 손익과 harness short_ret_h10의 최대 차 {o['selfcheck']:.1e}.", ""]

    sp = o["s_pair"]
    L += ["### (a)-2 짝별 기본 변형(E1·X1)", "",
          "| 짝 | 앞 | 뒤 | 2025 | 기초 공매도(같은 창, 뒤) | 추적 차(뒤) |", "|---|---:|---:|---:|---:|---:|"]
    for u, inv in PAIRS:
        x = sp[(sp["pair"] == f"{u}->{inv}") & (sp["entry"] == ENTRIES[0]) & (sp["exit"] == EXITS[0])].set_index("period")
        cell = lambda p: (f"{pct(x.loc[p, 'mean_per_event'])} ({int(x.loc[p, 'n_events'])}건)" if p in x.index else "사건 없음")
        L.append(f"| {u}→{inv}({LEV[inv]}배) | {cell('앞')} | {cell('뒤')} | {cell('2025')} | "
                 f"{pct(x.loc['뒤', 'proxy_mean']) if '뒤' in x.index else '사건 없음'} | "
                 f"{pct(x.loc['뒤', 'track_mean']) if '뒤' in x.index else '사건 없음'} |")
    L += ["", "추적 차 = 인버스 총손익 − 배수 × 기초 공매도 총손익(일일 재설정 손실·운용 보수 포함).", ""]

    L += ["### (a)-3 이탈 뒤 vs 아무 날(같은 구간 모든 거래일을 t로 본 E1 진입, t+10 종가 청산)", "",
          "| 짝 | 구간 | 이탈 뒤 사건당 | 아무 날 평균 | 아무 날 수 |", "|---|---|---:|---:|---:|"]
    for u, inv in [("SOXX", "SOXS"), ("SMH", "SOXS"), ("QQQ", "SQQQ"), ("SPY", "SH")]:
        x = sp[(sp["pair"] == f"{u}->{inv}") & (sp["entry"] == ENTRIES[0]) & (sp["exit"] == EXITS[2])].set_index("period")
        for p in ("앞", "뒤", "2025"):
            rr = rnd[(rnd["sym"] == u) & (rnd["inv"] == inv) & (rnd["period"] == p) & (rnd["h"] == 10)].iloc[0]
            ev_txt = f"{pct(x.loc[p, 'mean_per_event'])} ({int(x.loc[p, 'n_events'])}건)" if p in x.index else "사건 없음"
            L.append(f"| {u}→{inv} | {p} | {ev_txt} | {pct(rr['mean'])} | {rr['n_days']} |")
    L.append("")

    # 2026 사건별
    e = ex[(ex["t"] >= "2026-01-01") & ex["inv"].isin(["SOXS", "SQQQ", "SH"]) & ex["ok"]]
    w = e.pivot_table(index=["sym", "inv", "N", "t", "cls"], columns=["entry", "exit"], values="inv_ret")
    w.columns = [a[:2] + b[:2] for a, b in w.columns]
    L += ["### (a)-4 2026 사건별 인버스 손익(비용 뒤)", "",
          "| 기초 | 인버스 | N | 이탈일 t | 분류(사후) | E1·X1 | E1·X3(t+10) | E1·X5(추적) | E2·X1(프리장) | E3·X1 | E3·X3 |",
          "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    for (sym, inv, N, t, cls), r in w.sort_index(level=["t", "sym", "N"]).iterrows():
        L.append(f"| {sym} | {inv} | {N} | {t.date()} | {cls} | {pct(r['E1X1'], 1)} | {pct(r['E1X3'], 1)} | "
                 f"{pct(r['E1X5'], 1)} | {pct(r['E2X1'], 1)} | {pct(r['E3X1'], 1)} | {pct(r['E3X3'], 1)} |")
    L += ["", "E3의 +0.0%는 t+1 종가가 이미 L 위로 돌아와 진입하지 않은 사건이다. 분류는 t 뒤 자료로 정한 설명용 표지다.", ""]

    # E2 점검
    e1 = ex[(ex["entry"] == ENTRIES[0]) & (ex["exit"] == EXITS[0]) & ex["ok"]].drop_duplicates(["inv", "t"]).set_index(["inv", "t"])
    e2 = ex[(ex["entry"] == ENTRIES[1]) & (ex["exit"] == EXITS[0]) & ex["ok"]].drop_duplicates(["inv", "t"]).set_index(["inv", "t"])
    jj = e1[["entry_px"]].join(e2[["entry_px", "pm_ts"]], rsuffix="_pm", how="inner")
    jj["d"] = jj["entry_px_pm"] / jj["entry_px"] - 1
    hr = pd.to_datetime(jj["pm_ts"]).dt.hour
    uc = o["unit_check"]
    L += ["### (a)-5 프리장 진입(E2) 점검", "",
          f"- 프리장 첫 봉은 {len(jj)}개 (인버스, 날짜) 중 04:00 ET 봉 {int((hr == 4).sum())}개, 그 뒤 봉 {int((hr != 4).sum())}개. "
          f"정규장 시가 대비 프리장 첫 봉 시가: 평균 {pct(jj['d'].mean())}, 중앙 {pct(jj['d'].median())}, 최소 {pct(jj['d'].min())}, 최대 {pct(jj['d'].max())}.",
          "- yfinance 프리장 봉의 거래량은 모두 0으로 온다. 그래서 '시가=고가=저가=종가' 한 값 봉만 빼고 썼다(받은 뒤·결과 전 정함). "
          "1시간봉 정규장 09:30 시가와 일봉 시가의 차(분할 단위 점검): "
          + ", ".join(f"{r.sym} 중앙 {r.median_absdiff * 100:.2f}%·99% {r.p99 * 100:.2f}%" for r in uc.itertuples()) + ".",
          f"- 3배 짝에서 프리장 진입(E2)이 정규장 시가 진입(E1)보다 사건당 평균이 낮았던 청산 규칙: 앞 {e2w['앞']}/5, 뒤 {e2w['뒤']}/5, "
          f"2025 {e2w['2025']}/5. 표본이 작아(앞 {base_f['n_events']}건/{base_f['n_dates']}일, 뒤 {base_b['n_events']}건/{base_b['n_dates']}일) 일반화는 못 한다.", ""]

    # (b)
    show = (["헤지 없음"]
            + [f"SQ|N{N}|SOXS|x={x:.2f}" for N in H.NS for x in ACC_X]
            + [f"SQ|N{N}|{p}|x=0.20" for N in H.NS for p in ("SQQQ", "짝")]
            + [f"ALL7|N{N}|SOXS|x=0.20" for N in H.NS]
            + [n for n in acc["name"] if n.startswith("SEL|") and "SOXS" in n])
    L += ["## (b) 데스크 7종목 계좌 + 이탈 시 인버스 x%", "",
          "기본 경로 `stop_research/cache/base_2026.json`(2026-01-02~10-09), 2025는 `base_hold.json`(주간 PDF). 계기 사건의 t+1 시가에 "
          "주식 x를 팔고 인버스 x를 산다(합계 ≤ 1). 청산은 되돌림(기초 종가 > L → 다음 시가), 아니면 t+11 시가. 비용 편도 회전율당 0.05%. "
          "SQ = SOXX 또는 QQQ 이탈, ALL7 = ETF 7개 이탈 전부('모든 이탈 헤지'), SEL = SQ 중 판정 칸 앞 1위 "
          f"`{o['sel_name']}`이 켜진 것. 수익/최대 낙폭, 헤지일 = 구간 안 헤지 보유 거래일.", "",
          "| 변형 | 2026 전체 | 2026 앞 | 2026 뒤 | 7월 쇼크 6/22~7/29 | 2025 | 헤지일(26앞/26뒤/25) |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for nm in show:
        r = rowf(nm)
        L.append(f"| {nm} | {pct(r['2026전체_수익'], 1)} / {pct(r['2026전체_낙폭'], 1)} | {pct(r['2026앞_수익'], 1)} / "
                 f"{pct(r['2026앞_낙폭'], 1)} | {pct(r['2026뒤_수익'], 1)} / {pct(r['2026뒤_낙폭'], 1)} | "
                 f"{pct(r['7월쇼크_수익'], 1)} | {pct(r['2025_수익'], 1)} / {pct(r['2025_낙폭'], 1)} | "
                 f"{int(r['2026앞_헤지일'])}/{int(r['2026뒤_헤지일'])}/{int(r['2025_헤지일'])} |")
    L += ["", f"전체 {len(acc) - 1}개 변형은 `b7_hedge_account.csv`. 2026년 SQ 헤지가 켜진 날: "
          + "; ".join(f"N{N} {win_sq[N]}" for N in H.NS) + ".",
          f"- 앞 고르기(stop_research select_front 그대로): {'통과 변형 없음 → 앞 구간 탈락' if o['pick_acc'] is None else o['pick_acc']['name']}. "
          f"앞 구간 최대 낙폭이 헤지 없음({pct(accb['2026앞_낙폭'], 1)})보다 얕아진 변형: "
          f"{int((acc['2026앞_낙폭'].iloc[1:] > accb['2026앞_낙폭'] + 1e-9).sum())}개.",
          "- 7월 쇼크 창은 사후에 정한 고점·저점이라 그 안의 이득은 저점 뒤 되돌림 손실(7/30~)을 빼고 잰 값이다. 2026 뒤 열이 그 손실까지 넣은 값이다.",
          f"- ALL7(모든 ETF 이탈 헤지)은 SQ보다 헤지일이 많다(N20 2026 {int(rowf('ALL7|N20|SOXS|x=0.20')['2026전체_헤지일'])}일 vs "
          f"{int(rowf('SQ|N20|SOXS|x=0.20')['2026전체_헤지일'])}일, 2025 {int(rowf('ALL7|N20|SOXS|x=0.20')['2025_헤지일'])}일 vs "
          f"{int(rowf('SQ|N20|SOXS|x=0.20')['2025_헤지일'])}일). SEL은 계기 사건이 역사 전체 "
          f"{int(rowf([n for n in acc['name'] if n.startswith('SEL|')][0])['n_trig_events'])}건뿐이다.", ""]

    # (c)
    L += ["## (c) 판정 칸: ETF 상태 특징(이탈일 t 종가에 앎)", "",
          "y = 사건표 short_ret_h10(종목·ETF 공매도 대리 손익, 비용 뒤). 통과 = 뒤 차이 > 0, 날짜 묶음 t ≥ 2, 순열 p < 0.05, "
          "켜진 사건 평균 > 0. 특징이 모두 날짜 고정(국면형)이라 원형 이동 p(b_p_shift)도 적는다.", "",
          "| 앞 순위 | 칸 | 앞 차이(켜짐 건/날) | 뒤 차이 | 뒤 t | 순열 p | 원형 p | 뒤 켜짐 평균 | 2025 차이 | 통과 |",
          "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for _, r in sel.iterrows():
        L.append(f"| {int(r['rank'])} | `{r['cell']}` | {pct(r['f_diff'])} ({int(r['f_n_on'])}/{int(r['f_dates_on'])}) | "
                 f"{pct(r['b_diff'])} | {num(r['b_t'])} | {num(r['b_p'], 3)} | {num(r['b_p_shift'], 3)} | "
                 f"{pct(r['b_mean_on'])} ({int(r['b_n_on'])}건) | {pct(r['a_diff'])} | {'통과' if r['pass'] else '아님'} |")
    L += ["", f"- 선택편향 귀무(날짜 묶음째 섞어 고르기·확인 200번): 통과 칸 수 평균 {num(nul['mean'], 3)}, 90% {nul['q90']:g}, "
          f"95% {nul['q95']:g}, 최대 {nul['max']}, 관측 {summ['n_pass']} → p {num(nul['p_value'], 3)}. 원형 이동 귀무: 평균 "
          f"{num(nuls.get('mean', np.nan), 3)}, 최대 {nuls.get('max', '못 구함')}.",
          "- 한 국면 의존 점검: 통과 칸이 없어 해당 없음(상위 칸은 켜진 사건 합계 손익이 마이너스라 비중을 낼 수 없다).",
          f"- 반대쪽 칸(ETF가 함께 박스를 깸: `=1(hi)`, n_etf_brk 상위) {opp_tot}개 중 앞·뒤·2025 차이가 모두 음수인 칸 {opp_neg}개"
          f"({', '.join(opp_list)}). 예: N40 qqq_brk=1 뒤 켜짐 "
          f"{pct(tab.set_index('cell').loc['N40|qqq_brk=1(hi)', 'b_mean_on'])}, 꺼짐 {pct(tab.set_index('cell').loc['N40|qqq_brk=1(hi)', 'b_mean_off'])}. "
          f"반대쪽 칸 중 고르기 조건을 만족한 칸 {int(opp['eligible'].sum())}개(앞 차이가 음수이거나 켜진 날짜가 8일 미만).",
          "- 특징 정의와 시점: " + "; ".join(f"`{f}` = {FEAT_DESC[f]}" for f in FEATS) + ". 모두 " + KNOWN + ".", ""]

    L += ["## 한계", "",
          "- 표본이 작다: 2026 앞 3배 짝 사건 10건/5일, 뒤 7건/5일. ETF 이탈 사건은 BRIEF 칸 판정(켜짐 ≥ 15건, 날짜 ≥ 8일)을 걸 수 없어 (a)는 기술 통계다.",
          "- 같은 날 N20·40·60, SOXX·SMH가 함께 이탈하면 같은 가격 경로가 여러 번 들어간다. t는 날짜로 묶었지만 평균은 중복 가중이다.",
          "- 계좌 겹치기에서 시간 청산은 t+10 종가 대신 t+11 시가다(계좌 경로가 시가 → 시가). 미국 정규장 종가(KST 05:00/06:00)는 사용자 주문 시간"
          "(08:00~24:00 KST) 밖이라 X2~X4의 종가 청산은 장 마감 주문(MOC)을 미리 걸어야 한다(체결 가능 여부는 확인 안 함).",
          "- 프리장 1시간봉은 yfinance(거래량 미제공, 오류 체결 가능)다. 2024-10-15 이후만 있다.",
          "- 인버스 가격에는 운용 보수·일일 재설정이 들어 있다. 세금·환전 스프레드는 넣지 않았다.",
          "- 2025 계좌는 주간 PDF 경로(base_hold)라 2026 일별 경로와 다르고, 보유 종목 절반이 watchlist 미분류라 반도체 한도가 과하게 걸린다(stop_research 기록).",
          "- 7월 쇼크 창(6/22~7/29)은 사후에 정한 고점·저점이다.",
          "- 판정 칸 특징은 모두 날짜 고정이라 국면형 거짓 양성이 명목보다 많을 수 있다(qa_harness.md). 이번에는 통과가 0이라 영향 없음.", ""]
    L += ["## 파일", "",
          "- `b7_hedge_cells.csv`: 판정 칸 전체(정의·시점 열 포함)",
          "- `b7_hedge_exec_events.csv`: (a) 사건 × 짝 × 진입 × 청산 전체 행",
          "- `b7_hedge_exec_summary.csv`: (a) 요약(3배 짝·6짝·짝별)",
          "- `b7_hedge_random_baseline.csv`: (a) 아무 날 기준선",
          "- `b7_hedge_account.csv`: (b) 계좌 변형 64행(헤지 없음 포함)",
          "- `b7_hedge_1h.parquet`: 인버스 5개 1시간봉(프리장 포함) 캐시", ""]
    (RES / "b7_hedge.md").write_text("\n".join(L), encoding="utf-8")
    return RES / "b7_hedge.md"


if __name__ == "__main__":
    if sys.argv[1:] == ["fetch"]:
        d = fetch_1h(force=True)
        print(d.groupby("sym")["ts"].agg(["min", "max", "count"]))
    else:
        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        o = main()
        print(write_report(o))
        print("elapsed", round(o["elapsed"], 1))
