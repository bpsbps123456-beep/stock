"""N5 — '이때 샀으면 한 달 뒤?'(다섯 가설의 매수 조건)와 10/25 결합 해석 C1~C3. BRIEF2.md(2026-10-26 결과 전 고정).

py code/n5_run.py   → results/n5_signals.csv · n5_cells.csv · n5_g5.csv · n5_audit.csv · n5_verdict.json · n5_tables.txt

입력(읽기만)
  research/pb_flow_census_20261026/census.csv        노트(n4h14 진입 규칙: 공개일만, exact 09:30 ET 전이면 당일 시가)
  research/news_turning_points_20261025/prices.csv   Codex 10/25 가격(yfinance 분할 반영·배당 미포함, 2023-12-01~2026-10-23)
  results/n4h14_signals.csv                          재현 점검(진입일·r5·r10·r20·대조 A), 2023 노트(앞선 D·T 조건, 보조 줄)
  results/n4h14_cells.csv                            2023 보조 줄의 같은 해 모든 날 평균(대조 Y)
  results/n4h5_soxx_events.csv                       G5 SOXX 실시간 반전 확인일(ZZA3)·확인일 종가까지 R 공개 표지

정의(등록)
  진입 i = 매수일 시가 O[i]. r20 = C[i+19]/O[i]−1, o21 = O[i+20]/O[i]−1, mae20 = min(0, min(C[i..i+19])/O[i]−1), r5 = C[i+4]/O[i]−1
  대조 Y = 같은 해 모든 날, A = 같은 해 + 직전 20일 고점 대비 낙폭 구간, AM = A + 신호일 M(SOXX 5일<0, ^TNX 5일 변화<0)
  통계 = 날짜 묶음 부트스트랩 95% 구간(진입일 차 < 묶음 길이면 같은 묶음) + 공통 원형 이동 귀무(15 ≤ k ≤ D−15)
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import time

import exchange_calendars as xc
import numpy as np
import pandas as pd

T0 = time.time()
HERE = pathlib.Path(__file__).resolve().parent.parent
ROOT = HERE.parents[3]
CODEX = ROOT / "research" / "news_turning_points_20261025"
RES = HERE / "results"
sys.dont_write_bytecode = True
_spec = importlib.util.spec_from_file_location("n4h14_run", HERE / "code" / "n4h14_run.py")
h14 = importlib.util.module_from_spec(_spec)
sys.modules["n4h14_run"] = h14
_spec.loader.exec_module(h14)

SEED = 20261026
B = 10000
W0, W1 = pd.Timestamp("2024-01-02"), pd.Timestamp("2026-10-23")
PRIOR_D, PRIOR_T = 20, 30
EP = 5                       # C1 new_episode: 10/25 겹침 4건 진입일 ±5거래일 제외
OVERLAP4 = ("2024-07-18", "2025-02-25", "2025-04-01", "2026-07-21")
PERIODS = [("2024~2026", "2024-01-02", "2026-10-23"), ("2026", "2026-01-02", "2026-10-23"),
           ("2026 1~5월", "2026-01-02", "2026-05-29"), ("2026 6~10월", "2026-06-01", "2026-10-23"),
           ("2025", "2025-01-02", "2025-12-31"), ("2024", "2024-01-02", "2024-12-31")]
MIN_N, MIN_CL = 5, 3


def xnys(a, b) -> pd.DatetimeIndex:
    s = pd.DatetimeIndex(xc.get_calendar("XNYS").sessions_in_range(a, b))
    return s.tz_localize(None) if s.tz is not None else s


# ---------------------------------------------------------------- 1. 가격·일별 표
def load_prices() -> pd.DataFrame:
    p = pd.read_csv(CODEX / "prices.csv", header=[0, 1], index_col=0)
    p.index = pd.to_datetime(p.index, errors="coerce")
    p = p[p.index.notna()]
    empty = p[p.isna().any(axis=1)].index.strftime("%Y-%m-%d").tolist()
    assert empty == ["2026-05-25", "2026-09-07"], empty      # 미국 휴장일 빈 행(Memorial·Labor Day)
    p = p.dropna().sort_index()
    assert p.index.equals(xnys("2023-12-01", "2026-10-23")), "XNYS 세션과 다름"
    assert (p > 0).all().all()
    return p


def build_daily(px: pd.DataFrame):
    sess = px.index
    N = len(sess)
    O = px["open"]["SOXX"].to_numpy(float)
    C = px["close"]["SOXX"].to_numpy(float)
    tnx = px["close"]["^TNX"]

    def fwd_close(k):
        out = np.full(N, np.nan)
        out[:N - k] = C[k:] / O[:N - k] - 1
        return out

    out = {"r5": fwd_close(4), "r10": fwd_close(9), "r20": fwd_close(19)}
    o21 = np.full(N, np.nan)
    o21[:N - 20] = O[20:] / O[:N - 20] - 1
    mae = np.full(N, np.nan)
    for i in range(N - 19):
        mae[i] = min(0.0, C[i:i + 20].min() / O[i] - 1)
    out["o21"], out["mae20"] = o21, mae
    cs = pd.Series(C, index=sess)
    dd20 = (cs / cs.rolling(20, min_periods=20).max() - 1).shift(1)
    bucket = pd.cut(-dd20, [-np.inf, .05, .10, np.inf], right=False, labels=[0, 1, 2]).astype("Int64")
    fall = cs.pct_change(5, fill_method=None) < 0
    yfall = tnx.diff(5) < 0
    M = (fall & yfall).shift(1, fill_value=False).to_numpy(bool)     # 신호일(i−1) 종가 기준, 진입 행에 둔다
    w = (sess >= W0) & (sess <= W1)
    i0 = int(np.flatnonzero(w)[0])
    D = int(w.sum())
    day = pd.DataFrame(dict(year=sess[w].year, b=bucket[w].to_numpy(), M=M[w]), index=sess[w])
    assert day.b.notna().all()
    day["b"] = day.b.astype(int)
    arr = {k: v[w] for k, v in out.items()}
    return sess, O, C, day, arr, i0, D


def control(day: pd.DataFrame, y: np.ndarray, keys: list[str]):
    df = day[keys].copy()
    df["y"] = y
    df["u"] = np.where(np.isnan(y), np.nan, (y > 0).astype(float))
    g = df[df.y.notna()].groupby(keys).agg(m=("y", "mean"), u=("u", "mean"))
    j = df[keys].merge(g, left_on=keys, right_index=True, how="left")
    return j.m.to_numpy(float), j.u.to_numpy(float)


# ---------------------------------------------------------------- 2. 신호
def load_signals(sess_px, i0):
    cal = xnys("2023-01-01", "2026-10-23")
    cen = h14.load_signals(pd.DataFrame(index=cal))
    old = pd.read_csv(RES / "n4h14_signals.csv", encoding="utf-8-sig", parse_dates=["entry"])
    keep = ["group_id", "entry"] + [c for c in old.columns if c.startswith("SOXX_")]
    sig = cen.merge(old[keep].rename(columns={"entry": "entry_n4"}), on="group_id", how="inner")
    assert len(sig) == len(old) == 245
    assert (sig.entry == sig.entry_n4).all(), "진입일이 n4h14 와 다름"
    sig["pos_all"] = cal.get_indexer(sig.entry)
    assert (sig.pos_all >= 0).all()
    pa = sig.pos_all.to_numpy(int)
    fam = sig.family.to_numpy()

    def has_prior(src_mask, lag):
        src = pa[src_mask]
        return np.array([bool(np.any((src >= x - lag) & (src <= x - 1))) for x in pa])

    sig["prior_D20"] = has_prior(fam == "D", PRIOR_D)
    sig["prior_T30"] = has_prior(fam == "T", PRIOR_T)
    win = sig[(sig.entry >= W0) & (sig.entry <= W1)].copy()
    win["i"] = sess_px.get_indexer(win.entry)
    assert (win.i >= 0).all()
    win["w"] = win.i - i0
    return sig, win


# ---------------------------------------------------------------- 3. 통계
def stats(pos, y, cm, cu, pb, rng, hclu):
    ok = ~np.isnan(y[pos]) & ~np.isnan(cm[pos])
    pos = np.sort(pos[ok])
    n = len(pos)
    row = dict(n=n)
    if n == 0:
        return row
    yv, cv = y[pos], cm[pos]
    ex = yv - cv
    row.update(mean=yv.mean(), median=float(np.median(yv)), worst=yv.min(), best=yv.max(), win=(yv > 0).mean(),
               deep10=(yv <= -0.10).mean(), ctl_mean=cv.mean(), ctl_win=np.nanmean(cu[pos]), ex_mean=ex.mean())
    lab = h14.clusters(pos, hclu)
    row["clusters"] = len(np.unique(lab))
    lo, hi = h14.boot_mean(ex, lab, rng, B)
    p0, p1 = pb
    S, K = h14.shift_matrix(pos, p0, p1)
    nullm = np.nanmean((y - cm)[S], axis=1)
    obs = ex.mean()
    row.update(ex_lo=lo, ex_hi=hi, null_K=K, p_hi=(np.sum(nullm >= obs) + 1) / (K + 1),
               p_lo=(np.sum(nullm <= obs) + 1) / (K + 1))
    row["p_two"] = min(1.0, 2 * min(row["p_hi"], row["p_lo"]))
    return row


def verdict(row, direction):
    if row.get("n", 0) < MIN_N or row.get("clusters", 0) < MIN_CL:
        return "표본 부족"
    if direction > 0:
        return "통과" if (row["ex_lo"] > 0 and row["p_hi"] < .05) else "불통과"
    return "통과" if (row["ex_hi"] < 0 and row["p_lo"] < .05) else "불통과"


def variant_pos(g: pd.DataFrame, vname: str, ep_pos=None) -> np.ndarray:
    if vname == "new":
        g = g[g.legacy_ids.isna()]
    if vname == "new_episode":
        g = g[[not np.any(np.abs(ep_pos - x) <= EP) for x in g.w]]
    pos = g.w.to_numpy(int)
    if vname == "first":
        pos = pos[h14.first_only(pos, g.group_id.to_numpy())]
    return np.sort(pos)


# ---------------------------------------------------------------- 4. 실행
def main():
    px = load_prices()
    sess, O, C, day, arr, i0, D = build_daily(px)
    ctl = {}
    for k in ("r5", "r10", "r20", "o21", "mae20"):
        for name, keys in (("Y", ["year"]), ("A", ["year", "b"]), ("AM", ["year", "b", "M"])):
            ctl[(k, name)] = control(day, arr[k], keys)
    sig, win = load_signals(sess, i0)
    wpos = win.w.to_numpy(int)
    for k in ("r5", "r20", "o21", "mae20"):
        win[k] = arr[k][wpos]
    win["M"] = day.M.to_numpy()[wpos]
    win["bucket"] = day.b.to_numpy()[wpos]
    for k in ("r5", "r20", "mae20"):
        win[f"A_{k}"] = ctl[(k, "A")][0][wpos]
        win[f"Y_{k}"] = ctl[(k, "Y")][0][wpos]
    win["AM_r5"] = ctl[("r5", "AM")][0][wpos]
    win["AM_r20"] = ctl[("r20", "AM")][0][wpos]

    # ------------------------------------------- 점검: n4h14 재현(진입일은 load_signals 에서 확인)
    aud = []
    for h in (5, 10, 20):
        r_new = arr[f"r{h}"][wpos]
        a_new = ctl[(f"r{h}", "A")][0][wpos]
        r_old = win[f"SOXX_r{h}"].to_numpy(float)
        a_old = win[f"SOXX_A{h}"].to_numpy(float)
        both = ~np.isnan(r_new) & ~np.isnan(r_old)
        aud.append(dict(check=f"r{h} 재현(2024~2026 신호)", n=int(both.sum()), nan_mismatch=int((np.isnan(r_new) != np.isnan(r_old)).sum()),
                        max_abs_diff=float(np.max(np.abs(r_new[both] - r_old[both])))))
        both = ~np.isnan(a_new) & ~np.isnan(a_old)
        aud.append(dict(check=f"대조 A{h} 재현", n=int(both.sum()), nan_mismatch=int((np.isnan(a_new) != np.isnan(a_old)).sum()),
                        max_abs_diff=float(np.max(np.abs(a_new[both] - a_old[both])))))
    # 손 계산 5건(가격표에서 직접)
    hand = []
    rs = np.random.default_rng(20261027)
    for gid in rs.choice(win[win.r20.notna()].group_id.to_numpy(), 5, replace=False):
        x = win[win.group_id == gid].iloc[0]
        i = int(x.i)
        o, c19, cmin, o20 = O[i], C[i + 19], C[i:i + 20].min(), O[i + 20] if i + 20 < len(O) else np.nan
        hand.append(dict(group_id=gid, entry=x.entry.date().isoformat(), exit_close_date=sess[i + 19].date().isoformat(),
                         open_entry=o, close_20th=c19, min_close=cmin, open_21st=o20,
                         r20_hand=c19 / o - 1, r20_code=x.r20, mae20_hand=min(0, cmin / o - 1), mae20_code=x.mae20,
                         o21_hand=o20 / o - 1, o21_code=x.o21))
    hand = pd.DataFrame(hand)
    for a, b in (("r20_hand", "r20_code"), ("mae20_hand", "mae20_code"), ("o21_hand", "o21_code")):
        aud.append(dict(check=f"손 계산 {a[:-5]} 5건", n=5, nan_mismatch=0,
                        max_abs_diff=float(np.nanmax(np.abs(hand[a] - hand[b])))))
    aud = pd.DataFrame(aud)
    aud.to_csv(RES / "n5_audit.csv", index=False, encoding="utf-8-sig")
    hand.to_csv(RES / "n5_audit_hand.csv", index=False, encoding="utf-8-sig")
    assert (aud.max_abs_diff < 1e-9).all() and (aud.nan_mismatch == 0).all(), aud

    # ------------------------------------------- 묶음
    f, lb, sell = win.family, win.lookback_months, win.sell_note.astype(bool)
    A_GROUPS = {
        "G1 기록적 매도(D, 1년+)": (f == "D") & (lb >= 12),
        "G2 매수 복귀·숏 청산(R)": f == "R",
        "G3 극단 매도(5년+)": sell & (lb >= 60),
        "G3b 매도 1~5년": sell & (lb >= 12) & (lb < 60),
        "G3c 매도 1년 미만": sell & (lb < 12),
        "G4 레버리지 쏠림(T)": f == "T",
        "참고 S 숏 쏠림": f == "S",
    }
    C_GROUPS = {
        "C1 (D·S) & 주가·금리 동반 하락": (f.isin(["D", "S"]) & win.M, -1, ("r5", "r20"), "AM"),
        "C1o (D·S) & 동반 하락 아님": (f.isin(["D", "S"]) & ~win.M, 0, ("r5", "r20"), "AM"),
        "C2 R & 앞 20일 D": ((f == "R") & win.prior_D20, +1, ("r20",), "A"),
        "C2o R & 앞 20일 D 없음": ((f == "R") & ~win.prior_D20, 0, ("r20",), "A"),
        "C3 D & 앞 30일 T": ((f == "D") & win.prior_T30, -1, ("r20",), "A"),
        "C3o D & 앞 30일 T 없음": ((f == "D") & ~win.prior_T30, 0, ("r20",), "A"),
    }
    for name, m in A_GROUPS.items():
        win[name.split()[0]] = m.to_numpy()
    for name, (m, *_r) in C_GROUPS.items():
        win[name.split()[0]] = m.to_numpy()
    ep_pos = np.array([int(np.flatnonzero(day.index == pd.Timestamp(d))[0]) for d in OVERLAP4])
    win.sort_values(["entry", "group_id"]).drop(columns=[c for c in win.columns if c.startswith("SOXX_")] + ["entry_n4"]).to_csv(
        RES / "n5_signals.csv", index=False, encoding="utf-8-sig")

    def pbound(a, b):
        m = (day.index >= a) & (day.index <= b)
        p = np.flatnonzero(m)
        return int(p[0]), int(p[-1])

    pbs = {p: pbound(a, b) for p, a, b in PERIODS}
    rng = np.random.default_rng(SEED)
    rows = []
    for gname, m in A_GROUPS.items():
        g = win[m.to_numpy()]
        for vname in ("all", "first"):
            pos_all = variant_pos(g, vname)
            for pname, *_ in PERIODS:
                p0, p1 = pbs[pname]
                pos = pos_all[(pos_all >= p0) & (pos_all <= p1)]
                for k, cname in (("r20", "A"), ("r20", "Y"), ("o21", "A"), ("mae20", "A"), ("mae20", "Y")):
                    cm, cu = ctl[(k, cname)]
                    r = stats(pos, arr[k], cm, cu, (p0, p1), rng, 20)
                    r.update(part="A", group=gname, variant=vname, period=pname, outcome=k, control=cname, direction=0, verdict="")
                    rows.append(r)
    for gname, (m, d, outs, cname) in C_GROUPS.items():
        g = win[m.to_numpy()]
        vlist = ("all", "first", "new", "new_episode") if gname.startswith("C1") else ("all", "first", "new")
        for vname in vlist:
            pos_all = variant_pos(g, vname, ep_pos)
            for pname, *_ in PERIODS:
                p0, p1 = pbs[pname]
                pos = pos_all[(pos_all >= p0) & (pos_all <= p1)]
                for k in outs:
                    for cn in ((cname, "A") if cname == "AM" else (cname,)):
                        cm, cu = ctl[(k, cn)]
                        r = stats(pos, arr[k], cm, cu, (p0, p1), rng, 5 if k == "r5" else 20)
                        reg = d != 0 and cn == cname and pname == "2024~2026"
                        r.update(part="C", group=gname, variant=vname, period=pname, outcome=k, control=cn, direction=d,
                                 verdict=verdict(r, d) if reg else "")
                        rows.append(r)
    cells = pd.DataFrame(rows)
    front = ["part", "group", "variant", "period", "outcome", "control", "direction", "verdict", "n", "clusters"]
    cells = cells[front + [c for c in cells.columns if c not in front]]
    cells.to_csv(RES / "n5_cells.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------- G5: 상승 반전 확인 다음 날 시가 매수
    ev = pd.read_csv(RES / "n4h5_soxx_events.csv", encoding="utf-8-sig")
    ev = ev[(ev.det == "ZZA3") & (ev.type == 1) & (ev.conf_date >= "2024-01-01")].copy()
    ev["c"] = sess.get_indexer(pd.to_datetime(ev.conf_date))
    assert (ev.c >= 0).all()
    ev["entry"] = [sess[c + 1].date().isoformat() if c + 1 < len(sess) else None for c in ev.c]
    ev = ev[ev.entry.notna()].copy()
    ev["w"] = ev.c + 1 - i0
    for k in ("r20", "o21", "mae20"):
        ev[f"{k}_buy"] = arr[k][ev.w.to_numpy(int)]
        ev[f"A_{k}"] = ctl[(k, "A")][0][ev.w.to_numpy(int)]
    g5 = ev[["conf_date", "entry", "pivot_date", "R_all_pre", "R_pre_ids", "r20_buy", "A_r20", "o21_buy", "mae20_buy", "A_mae20", "y1"]]
    g5.to_csv(RES / "n5_g5.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------- 2023 보조 줄(n4h14 계산, 시가 자료가 클라우드에 없어 MAE 없음)
    s23 = sig[sig.entry.dt.year == 2023]
    yc = pd.read_csv(RES / "n4h14_cells.csv", encoding="utf-8-sig")
    y23 = yc[(yc.asset == "SOXX") & (yc.control == "Y") & (yc.h == 20) & (yc.period == "2023") & (yc.n > 0)][["ctl_mean", "ctl_up"]].iloc[0]
    f23, lb23, sell23 = s23.family, s23.lookback_months, s23.sell_note.astype(bool)
    rows23 = []
    for gname, m in (("G1 기록적 매도(D, 1년+)", (f23 == "D") & (lb23 >= 12)), ("G2 매수 복귀·숏 청산(R)", f23 == "R"),
                     ("G3 극단 매도(5년+)", sell23 & (lb23 >= 60)), ("G4 레버리지 쏠림(T)", f23 == "T")):
        g = s23[m]
        rv = g.SOXX_r20.dropna()
        rows23.append(dict(group=gname, n=len(rv), win=(rv > 0).mean() if len(rv) else np.nan, mean=rv.mean() if len(rv) else np.nan,
                           worst=rv.min() if len(rv) else np.nan, ctl_A=g.SOXX_A20[rv.index].mean() if len(rv) else np.nan,
                           all_days_mean=y23.ctl_mean, all_days_win=y23.ctl_up))
    t23 = pd.DataFrame(rows23)

    # ------------------------------------------- 판정·출력
    reg = cells[cells.verdict != ""]
    ver = {f"{r.group}|{r.variant}|{r.outcome}": r.verdict for r in reg.itertuples()}
    tests = dict(registered_cells=len(ver), verdict_counts=pd.Series(ver).value_counts().to_dict(),
                 expected_false_pass=round(0.05 * len(ver), 2), cells_total=len(cells), bootstrap=dict(B=B, seed=SEED),
                 null=dict(kind="공통 원형 이동", kmin=h14.KMIN), window=[str(W0.date()), str(W1.date()), D],
                 signals_in_window=len(win), runtime_s=round(time.time() - T0, 1))
    (RES / "n5_verdict.json").write_text(json.dumps(dict(verdicts=ver, tests=tests), ensure_ascii=False, indent=1), encoding="utf-8")

    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_colwidth", 40)
    L = []
    P = L.append
    P(f"N5 실행 {time.strftime('%Y-%m-%d %H:%M')}  창 {W0.date()}~{W1.date()} D={D}  창 안 신호 {len(win)}")
    P("== 점검 ==")
    P(aud.to_string(index=False))
    P(hand.round(5).to_string(index=False))
    P("")
    P("== A. 다섯 가설의 때에 샀다면: r20(한 달), 대조 A, all ==")
    cols = ["group", "period", "n", "clusters", "win", "mean", "median", "worst", "best", "ctl_win", "ctl_mean", "ex_mean", "ex_lo", "ex_hi", "p_two"]
    a = cells[(cells.part == "A") & (cells.outcome == "r20") & (cells.control == "A") & (cells.variant == "all")]
    P(a[cols].round(4).to_string(index=False))
    P("")
    P("== A. 같은 해 모든 날(Y) 대비 r20, all ==")
    a = cells[(cells.part == "A") & (cells.outcome == "r20") & (cells.control == "Y") & (cells.variant == "all")]
    P(a[["group", "period", "n", "win", "mean", "ctl_win", "ctl_mean", "ex_mean", "ex_lo", "ex_hi", "p_two"]].round(4).to_string(index=False))
    P("")
    P("== A. first(5거래일 안 같은 묶음 연속 제외) r20, 대조 A ==")
    a = cells[(cells.part == "A") & (cells.outcome == "r20") & (cells.control == "A") & (cells.variant == "first")]
    P(a[cols].round(4).to_string(index=False))
    P("")
    P("== A. 한 달 동안 최대 하락 mae20(대조 A·Y), all ==")
    a = cells[(cells.part == "A") & (cells.outcome == "mae20") & (cells.variant == "all")]
    P(a[["group", "period", "control", "n", "mean", "median", "worst", "deep10", "ctl_mean", "ex_mean", "ex_lo", "ex_hi", "p_two"]].round(4).to_string(index=False))
    P("")
    P("== A. 21번째 장 시가 청산 o21(주문 가능 시간 안), 대조 A, all ==")
    a = cells[(cells.part == "A") & (cells.outcome == "o21") & (cells.variant == "all")]
    P(a[["group", "period", "n", "win", "mean", "median", "worst", "ctl_mean", "ex_mean", "ex_lo", "ex_hi", "p_two"]].round(4).to_string(index=False))
    P("")
    P("== G5 상승 반전 확인(ZZA3) 다음 날 시가 매수 ==")
    P(g5.round(4).to_string(index=False))
    for flag, gg in g5.groupby("R_all_pre"):
        v = gg.r20_buy.dropna()
        P(f"R 사전 {flag}: n={len(v)} 승 {int((v > 0).sum())} 평균 {v.mean():.4f} 중앙 {v.median():.4f} 최악 {v.min():.4f} "
          f"대조A {gg.A_r20[v.index].mean():.4f} mae20 평균 {gg.mae20_buy[v.index].mean():.4f}")
    P("")
    P("== C. 10/25 결합 해석 재검증 ==")
    cc = ["group", "variant", "period", "outcome", "control", "verdict", "n", "clusters", "win", "mean", "ctl_mean", "ex_mean", "ex_lo", "ex_hi", "p_hi", "p_lo"]
    P(cells[cells.part == "C"][cc].round(4).to_string(index=False))
    P("")
    P("== 2023 보조(n4h14 r20, 클라우드에 2023 시가 없음) ==")
    P(t23.round(4).to_string(index=False))
    P("")
    P("== 묶음별 한 달 최악 5건(all, 2024~2026) ==")
    for gname in list(A_GROUPS) + [k for k in C_GROUPS if not k.split()[0].endswith("o")]:
        col = gname.split()[0]
        g = win[win[col] & win.r20.notna()].sort_values("r20").head(5)
        P(f"{gname}: " + "; ".join(f"{e.date()} {b} r20 {r:+.3f} mae {mm:+.3f}" for e, b, r, mm in zip(g.entry, g.bank, g.r20, g.mae20)))
    P("")
    P(json.dumps(tests, ensure_ascii=False))
    txt = "\n".join(L)
    (RES / "n5_tables.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
