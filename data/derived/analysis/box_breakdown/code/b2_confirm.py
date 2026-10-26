"""갈래 2 — 확인 대기 진입 변형(BRIEF 갈래 2). 하네스 사건표와 evaluate_cells를 그대로 쓴다(판정 로직을 새로 만들지 않는다).

실행: py b2_confirm.py features   # 변형 진입·청산만 계산하고 점검(즉시 진입 재현, 시점 점검). 결과 집계는 안 봄
      py b2_confirm.py run        # 칸 판정 → ../results/b2_confirm.md, b2_confirm_cells.csv

정의(결과를 보기 전에 정함, 2026-10-14). 사건 = 하네스 사건표(N=20·40·60, 이탈일 t = 인덱스 i, L = L_N(t),
ATR = 하네스 atr20 = ATR20(t−1)). 즉시 진입 = 하네스 short_ret_h10(t+1 시가 공매도).
변형 5개(모두 '확인일 d 종가' 뒤에 알 수 있고 진입은 d+1 정규장 시가):
- hold1   t+1 종가 < L → d = t+1.
- hold2   t+1·t+2 종가 모두 < L → d = t+2.
- hold3   t+1..t+3 종가 모두 < L → d = t+3.
          (어느 날이든 종가 ≥ L이면 진입 안 함. 종가 > L은 즉시 진입이라면 되돌림 손절이 났을 날이다.)
- retest  t+1..t+5에서 처음으로 L − 0.3·ATR ≤ 고가 ≤ L + 0.3·ATR(아래에서 L 재시험)이고 종가 < L인 날 = d.
          그 전에 종가 > L이 나오면 취소. t+5까지 없으면 진입 안 함.
- bounce  첫 반등 고점 돌파 실패. t+1..t+5를 차례로 본다. 종가 > L이면 취소. 처음으로 종가 > 전일 종가인 날을
          반등 시작으로 하고 반등 고점 Hb = 그날 고가. 그 뒤 날마다 종가 < 전일 종가 이고 고가 ≤ Hb 이고 종가 < L
          이면 d(돌파 실패 뒤 다시 내림 = 첫 반등 고점 아래 종가). 아니면 Hb = max(Hb, 고가)로 반등을 이어 본다.
          t+5까지 d가 없으면 진입 안 함.
청산(모든 변형, 즉시 진입과 같은 규칙·같은 끝): 진입일 e = d+1부터 t+9까지 종가 > L이면 다음 날 시가 청산(되돌림
손절), 아니면 t+10 종가. 즉시 진입과 끝 날짜가 같아 '같은 사건에서 기다린 값'만 다르다(d 이전 종가는 모두 ≤ L이라
되돌림 날도 즉시 진입과 같다). 비용 왕복 0.10%. 보조로 t+20 끝(h20)도 계산한다(설명용).
비교 집합: 즉시 진입 손익이 있는(short_ret_h10 유한) 사건 = '같은 사건 집합'.
- 놓친 하락 = (시가(t+1) − 시가(e)) / 시가(t+1), 진입한 사건 평균(양수 = 기다리는 동안 내려 공매도가 놓친 몫).
- 가짜 걸러냄 = 하네스 분류 '가짜' 사건 중 변형이 진입하지 않은 비율. 진짜 유지 = '진짜' 중 진입한 비율.
  분류는 t+1..t+10 결과를 쓰는 사후 설명용이다.
판정(evaluate_cells 그대로, 칸 = 변형 5 × N 3 × 유형 2 = 30칸): 변형마다 사건표를 겹쳐 쌓는다(stack).
  꺼짐 행 = 그 N·유형의 모든 사건의 즉시 진입 손익, 켜짐 행 = 같은 사건 중 변형이 진입한 사건의 변형 손익(거래당).
  따라서 '켜짐 − 꺼짐' = 변형 거래당 손익 − 같은 사건 집합의 즉시 진입 거래당 손익, 켜짐 평균 = 변형 거래당 손익.
  날짜 묶음 t·순열은 이탈일 t로 묶는다(켜짐·꺼짐 행이 같은 날 묶음에 함께 들어가 짝 구조가 반영된다).
  순열 층 = N·kind·변형(각 칸의 행은 한 층 안에만 있다). 켜짐 ≥ 15거래·8날짜, 상위 10칸, 순열 200, 귀무 200 = 기본값.
  사건당(거래 안 하면 0) 비교와 그 분해(기다린 비용 + 거른 이득)는 설명용으로 따로 적는다(판정 아님).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import harness as H  # noqa: E402

RESULTS = H.RESULTS
VARIANTS = ["hold1", "hold2", "hold3", "retest", "bounce"]
W = 5            # retest·bounce 확인 창(t+1..t+5)
BAND = 0.3       # 재시험 띠(ATR 배)
KINDS = ("stock", "etf")
KN = {"stock": "종목", "etf": "ETF"}
PERIODS = (("f", H.FRONT, "앞(2026-01~05)"), ("b", H.BACK, "뒤(2026-06~09)"), ("a", H.AUX, "보조 2025"))

DEF = {
    "hold1": ("t+1 종가 < L_N이면 t+2 시가 공매도(아니면 진입 안 함)", "t+1 종가 뒤, 진입 t+2 시가"),
    "hold2": ("t+1·t+2 종가 모두 < L_N이면 t+3 시가 공매도", "t+2 종가 뒤, 진입 t+3 시가"),
    "hold3": ("t+1~t+3 종가 모두 < L_N이면 t+4 시가 공매도", "t+3 종가 뒤, 진입 t+4 시가"),
    "retest": ("t+1~t+5 중 처음으로 고가가 [L_N−0.3·ATR20, L_N+0.3·ATR20]에 닿고 종가 < L_N인 날 d(그 전 종가 > L_N이면 취소) → d+1 시가 공매도",
               "재시험일 d 종가 뒤(d ≤ t+5), 진입 d+1 시가"),
    "bounce": ("t+1~t+5에서 첫 상승 종가일을 반등 시작(반등 고점 Hb=고가, 이후 최고 고가로 갱신), 그 뒤 처음으로 종가<전일 종가·고가≤Hb·종가<L_N인 날 d(그 전 종가 > L_N이면 취소) → d+1 시가 공매도",
               "반등 실패일 d 종가 뒤(d ≤ t+5), 진입 d+1 시가"),
}
EXIT_TXT = "청산은 즉시 진입과 같음: 진입일~t+9 종가 > L_N이면 다음 시가, 아니면 t+10 종가. 비용 왕복 0.10%."


# ───────────────────────────── 변형 진입·청산 ─────────────────────────────
def trigger(h, c, i, Lv, atr, v, w=W, band=BAND):
    """확인일 d(인덱스)와 상태. 상태: 'trig' | 'cancel'(확인 전 종가 > L 또는 hold의 종가 ≥ L) | 'none'(창 안 미확인)
    | 'nodata'. t+1..확인일 자료만 본다."""
    n = len(c)
    if v.startswith("hold"):
        k = int(v[-1])
        for j in range(i + 1, i + k + 1):
            if j >= n or np.isnan(c[j]):
                return -1, "nodata"
            if not c[j] < Lv:
                return -1, "cancel"
        return i + k, "trig"
    started, hb = False, np.nan
    for j in range(i + 1, i + w + 1):
        if j >= n or np.isnan(c[j]) or np.isnan(h[j]) or np.isnan(c[j - 1]):
            return -1, "nodata"
        if c[j] > Lv:
            return -1, "cancel"
        if v == "retest":
            if Lv - band * atr <= h[j] <= Lv + band * atr and c[j] < Lv:
                return j, "trig"
        elif v == "bounce":
            if not started:
                if c[j] > c[j - 1]:
                    started, hb = True, h[j]
            else:
                if c[j] < c[j - 1] and h[j] <= hb and c[j] < Lv:
                    return j, "trig"
                hb = max(hb, h[j])
        else:
            raise ValueError(v)
    return -1, "none"


def exit_from(o, c, i, e, Lv, hz=10, cost=H.COST):
    """e 시가 공매도. e..i+hz−1 종가 > L → 다음 시가, 아니면 i+hz 종가. 반환 (손익, 사유, 청산 인덱스).
    e = i+1이면 하네스 short_outcome과 같다."""
    n = len(c)
    if e >= n or np.isnan(o[e]) or e > i + hz:
        return np.nan, None, -1
    entry = o[e]
    for k in range(e, i + hz):
        if k >= n or np.isnan(c[k]):
            return np.nan, None, -1
        if c[k] > Lv:
            if k + 1 >= n or np.isnan(o[k + 1]):
                return np.nan, None, -1
            return (entry - o[k + 1]) / entry - cost, "reclaim", k + 1
    k = i + hz
    if k >= n or np.isnan(c[k]):
        return np.nan, None, -1
    return (entry - c[k]) / entry - cost, "time", k


def compute(ev, P):
    """사건표에 변형 열을 붙인다: {v}_status, {v}_d(확인일 − t, 거래일), {v}_y(h10 손익), {v}_y20, {v}_why, {v}_missed."""
    ev = ev.copy()
    cols = {f"{v}_{k}": [] for v in VARIANTS for k in ("status", "d", "y", "y20", "why", "missed")}
    chk = []
    for r in ev.itertuples(index=False):
        j = P.j[r.sym]
        o, h, c = P.O[:, j], P.H[:, j], P.C[:, j]
        i, Lv, a = int(r.i), float(r.L), float(r.atr20)
        chk.append(exit_from(o, c, i, i + 1, Lv, 10)[0])
        o1 = o[i + 1] if i + 1 < len(o) else np.nan
        for v in VARIANTS:
            d, st = trigger(h, c, i, Lv, a, v)
            y = y20 = mis = np.nan
            why = None
            if st == "trig":
                y, why, _ = exit_from(o, c, i, d + 1, Lv, 10)
                y20 = exit_from(o, c, i, d + 1, Lv, 20)[0]
                if d + 1 < len(o) and not np.isnan(o[d + 1]):
                    mis = (o1 - o[d + 1]) / o1
            cols[f"{v}_status"].append(st)
            cols[f"{v}_d"].append(d - i if st == "trig" else np.nan)
            cols[f"{v}_y"].append(y)
            cols[f"{v}_y20"].append(y20)
            cols[f"{v}_why"].append(why)
            cols[f"{v}_missed"].append(mis)
    for k, x in cols.items():
        ev[k] = x
    ev["imm_chk"] = chk
    return ev


def load_all():
    P = H.load_panel()
    ev = H.load_events_all()
    ev = compute(ev, P)
    return ev, P


# ───────────────────────────── 쌓기·칸 ─────────────────────────────
def stack(ev):
    """변형마다 [즉시 진입 행(role=imm, 모든 사건) + 변형 행(role=var, 진입한 사건)]을 쌓는다. y_h10 = 그 행의 손익."""
    base = ev[ev["short_ret_h10"].notna()]
    keep = ["sym", "kind", "N", "t", "i", "episode", "cls"]
    parts = []
    for v in VARIANTS:
        a = base[keep].assign(vb=v, role="imm", y_h10=base["short_ret_h10"].to_numpy())
        tr = base[base[f"{v}_status"] == "trig"]
        b = tr[keep].assign(vb=v, role="var", y_h10=tr[f"{v}_y"].to_numpy())
        parts += [a, b]
    s = pd.concat(parts, ignore_index=True)
    per = np.zeros(len(s), bool)
    for _, p, _ in PERIODS:
        per |= H.in_period(s, p).to_numpy()
    return s[per].reset_index(drop=True)


def cell_name(v, N, k):
    return f"b2|{v}|N{N}|{k}"


def make_cells(s):
    cells = []
    for v in VARIANTS:
        for N in H.NS:
            for k in KINDS:
                m = (s["vb"] == v) & (s["N"] == N) & (s["kind"] == k)
                x = pd.Series(np.nan, index=s.index)
                x[m & (s["role"] == "var")] = 1.0
                x[m & (s["role"] == "imm")] = 0.0
                cells.append((cell_name(v, N, k), x))
    return cells


# ───────────────────────────── 설명용 지표 ─────────────────────────────
def describe(ev):
    """변형 × N × 유형 × 구간: 같은 사건 집합 비교(거래당·사건당), 놓친 하락, 가짜 걸러냄."""
    base = ev[ev["short_ret_h10"].notna()]
    rows = []
    for v in VARIANTS:
        for N in H.NS:
            for k in KINDS:
                for pk, p, pn in PERIODS:
                    g = base[(base["N"] == N) & (base["kind"] == k) & H.in_period(base, p)]
                    st = g[f"{v}_status"]
                    trig = (st == "trig") & g[f"{v}_y"].notna()
                    imm = g["short_ret_h10"].to_numpy(float)
                    vy = g[f"{v}_y"].to_numpy(float)
                    opp = np.where(trig, vy, 0.0)
                    diff = opp - imm
                    m_imm, t_imm, n, G = H.cluster_mean(imm, g["t"])
                    m_tr, t_tr, ntr, Gtr = H.cluster_mean(vy[trig.to_numpy()], g["t"][trig]) if trig.any() else (np.nan, np.nan, 0, 0)
                    m_opp, t_opp, _, _ = H.cluster_mean(opp, g["t"]) if n else (np.nan, np.nan, 0, 0)
                    m_d, t_d, _, _ = H.cluster_mean(diff, g["t"]) if n else (np.nan, np.nan, 0, 0)
                    tm = trig.to_numpy()
                    wait_cost = float(np.sum(np.where(tm, vy - imm, 0.0)) / n) if n else np.nan
                    skip_gain = float(np.sum(np.where(~tm, -imm, 0.0)) / n) if n else np.nan
                    cls = g["cls"]
                    fake, true = (cls == "가짜"), (cls == "진짜")
                    mis = g.loc[trig, f"{v}_missed"]
                    imm_tr = imm[tm]
                    rows.append({
                        "cell": cell_name(v, N, k), "variant": v, "N": N, "kind": k, "period": pk,
                        "events": n, "event_dates": G, "imm_mean": m_imm, "imm_t": t_imm,
                        "trades": int(tm.sum()), "trade_dates": Gtr, "trade_rate": tm.mean() if n else np.nan,
                        "cancel_rate": float((st == "cancel").mean()) if n else np.nan,
                        "none_rate": float((st == "none").mean()) if n else np.nan,
                        "var_mean_trade": m_tr, "var_t_trade": t_tr,
                        "var_med_trade": float(np.median(vy[tm])) if tm.any() else np.nan,
                        "var_win_trade": float(np.mean(vy[tm] > 0)) if tm.any() else np.nan,
                        "var_reclaim_trade": float((g.loc[trig, f"{v}_why"] == "reclaim").mean()) if tm.any() else np.nan,
                        "var_h20_trade": float(np.nanmean(g.loc[trig, f"{v}_y20"])) if g.loc[trig, f"{v}_y20"].notna().any() else np.nan,
                        "imm_mean_on_trig": float(imm_tr.mean()) if tm.any() else np.nan,
                        "imm_mean_on_skip": float(imm[~tm].mean()) if (~tm).any() else np.nan,
                        "opp_mean": m_opp, "opp_diff": m_d, "opp_diff_t": t_d,
                        "wait_cost": wait_cost, "skip_gain": skip_gain,
                        "missed_mean": float(mis.mean()) if len(mis) else np.nan,
                        "missed_median": float(mis.median()) if len(mis) else np.nan,
                        "wait_days_mean": float(g.loc[trig, f"{v}_d"].mean()) if tm.any() else np.nan,
                        "fake_n": int(fake.sum()), "true_n": int(true.sum()),
                        "fake_filtered": float((~tm[fake.to_numpy()]).mean()) if fake.any() else np.nan,
                        "true_kept": float(tm[true.to_numpy()].mean()) if true.any() else np.nan,
                        "skipped_fake_share": float((cls[~tm] == "가짜").mean()) if (~tm).any() and cls[~tm].notna().any() else np.nan,
                        "trades_true_share": float((cls[tm] == "진짜").mean()) if tm.any() else np.nan,
                        "trades_fake_share": float((cls[tm] == "가짜").mean()) if tm.any() else np.nan,
                        "events_true_share": float(true.mean()) if n else np.nan,
                        "events_fake_share": float(fake.mean()) if n else np.nan,
                    })
    return pd.DataFrame(rows)


# ───────────────────────────── 실행 ─────────────────────────────
def cmd_features():
    ev, P = load_all()
    ok = ev["short_ret_h10"].notna()
    d = (ev.loc[ok, "imm_chk"] - ev.loc[ok, "short_ret_h10"]).abs()
    print("즉시 진입 재현: 최대 차이", float(d.max()), "NaN 불일치", int((ev["imm_chk"].isna() != ev["short_ret_h10"].isna()).sum()))
    for v in VARIANTS:
        st = ev.loc[ok, f"{v}_status"].value_counts().to_dict()
        dd = ev.loc[ok & (ev[f"{v}_status"] == "trig"), f"{v}_d"]
        print(v, st, "확인일 t+", dd.value_counts().sort_index().to_dict())
        # 진입가가 확인일 다음 시가인지, 진입일 ≤ t+10인지
        assert int(((ev[f"{v}_status"] == "trig") & ~(ev[f"{v}_d"].between(1, W))).sum()) == 0
    # 미래 교란: 확인일 d 뒤 가격을 바꿔도 status·d가 그대로인지(시점 점검)
    rng = np.random.default_rng(7)
    smp = ev[ok].sample(300, random_state=1)
    bad = 0
    for r in smp.itertuples(index=False):
        j = P.j[r.sym]
        h, c = P.H[:, j].copy(), P.C[:, j].copy()
        for v in VARIANTS:
            d0, s0 = trigger(h, c, int(r.i), r.L, r.atr20, v)
            cut = (d0 if s0 == "trig" else int(r.i) + (int(v[-1]) if v.startswith("hold") else W)) + 1
            h2, c2 = h.copy(), c.copy()
            u = rng.uniform(0.6, 1.4, len(c) - cut) if cut < len(c) else np.array([])
            h2[cut:] *= u
            c2[cut:] *= u
            d1, s1 = trigger(h2, c2, int(r.i), r.L, r.atr20, v)
            bad += int((d0, s0) != (d1, s1))
    print("미래 교란(300사건 × 5변형) 상태 변화:", bad)
    return ev


def cmd_check(ev=None):
    """독립 재계산: 긴 형식 일봉(cache/daily.parquet)에서 날짜로 행을 찾아 변형 상태·확인일·h10 손익을 다시 짠다
    (trigger·exit_from을 쓰지 않는다). 반환 (상태 불일치, 손익 불일치, 사건 수)."""
    if ev is None:
        ev, _ = load_all()
    d = pd.read_parquet(H.CACHE / "daily.parquet")
    cal = sorted(d.loc[d["sym"] == "QQQ", "date"].unique())
    pos = {t: k for k, t in enumerate(cal)}
    g = {s_: x.set_index("date") for s_, x in d.groupby("sym")}
    ev = ev[ev["short_ret_h10"].notna()]
    bad = {v: 0 for v in VARIANTS}
    bady = {v: 0 for v in VARIANTS}
    for r in ev.itertuples(index=False):
        x = g[r.sym]
        k0 = pos[pd.Timestamp(r.t)]
        rows = [x.loc[cal[k0 + j]] if k0 + j < len(cal) and cal[k0 + j] in x.index else None for j in range(0, 11)]
        for v in VARIANTS:
            st, dd = "none", None
            if v.startswith("hold"):
                kk = int(v[-1])
                st, dd = "trig", kk
                for j in range(1, kk + 1):
                    if not rows[j].Close < r.L:
                        st, dd = "cancel", None
                        break
            else:
                hb = None
                for j in range(1, W + 1):
                    b, pb = rows[j], rows[j - 1]
                    if b.Close > r.L:
                        st = "cancel"
                        break
                    if v == "retest":
                        if r.L - BAND * r.atr20 <= b.High <= r.L + BAND * r.atr20 and b.Close < r.L:
                            st, dd = "trig", j
                            break
                    elif hb is None:
                        if b.Close > pb.Close:
                            hb = b.High
                    elif b.Close < pb.Close and b.High <= hb and b.Close < r.L:
                        st, dd = "trig", j
                        break
                    else:
                        hb = max(hb, b.High)
            if getattr(r, f"{v}_status") != st or (st == "trig" and int(getattr(r, f"{v}_d")) != dd):
                bad[v] += 1
                continue
            if st == "trig":
                e = dd + 1
                ent, y = rows[e].Open, None
                for j in range(e, 10):
                    if rows[j].Close > r.L:
                        y = (ent - rows[j + 1].Open) / ent - H.COST
                        break
                if y is None:
                    y = (ent - rows[10].Close) / ent - H.COST
                if abs(y - getattr(r, f"{v}_y")) > 1e-12:
                    bady[v] += 1
    return bad, bady, len(ev)


def trace_rows(ev, P, n=4):
    """손 추적용 예시(뒤 구간 N20 종목, 변형별 1건)."""
    out = []
    g = ev[(ev["N"] == 20) & (ev["kind"] == "stock") & H.in_period(ev, H.BACK) & ev["short_ret_h10"].notna()]
    for v in ("hold2", "retest", "bounce"):
        x = g[g[f"{v}_status"] == "trig"].head(1)
        for r in x.itertuples(index=False):
            j = P.j[r.sym]
            i, d = int(r.i), int(r.i + getattr(r, f"{v}_d"))
            o, c = P.O[:, j], P.C[:, j]
            out.append({"variant": v, "sym": r.sym, "t": str(pd.Timestamp(r.t).date()), "L": r.L, "atr": r.atr20,
                        "closes_t1_td": [round(float(c[k]), 3) for k in range(i + 1, d + 1)],
                        "highs_t1_td": [round(float(P.H[k, j]), 3) for k in range(i + 1, d + 1)],
                        "d": f"t+{d - i} {P.cal[d].date()}", "o_t1": float(o[i + 1]), "entry": float(o[d + 1]),
                        "imm": r.short_ret_h10, "var": getattr(r, f"{v}_y"), "why": getattr(r, f"{v}_why"),
                        "missed": getattr(r, f"{v}_missed")})
    return out


def _f(x, pct=True, nd=2, sign=True):
    if x is None or (isinstance(x, (float, np.floating)) and not np.isfinite(x)):
        return "못 구함"
    if pct:
        return f"{x * 100:+.{nd}f}%" if sign else f"{x * 100:.{nd}f}%"
    return f"{x:.{nd}f}"


def cmd_run():
    ev, P = load_all()
    s = stack(ev)
    cells = make_cells(s)
    tab, summ = H.evaluate_cells(s, cells, y="y_h10", y_ex="__none__", strata=("N", "kind", "vb"))
    de = describe(ev)
    wide = de.pivot_table(index="cell", columns="period", values=[c for c in de.columns if c not in
                                                                   ("cell", "variant", "N", "kind", "period")],
                          aggfunc="first")
    wide.columns = [f"{p}_{c}" for c, p in wide.columns]
    out = tab.merge(wide, left_on="cell", right_index=True, how="left")
    out.insert(1, "variant", [c.split("|")[1] for c in out["cell"]])
    out.insert(2, "N", [int(c.split("|")[2][1:]) for c in out["cell"]])
    out.insert(3, "kind", [c.split("|")[3] for c in out["cell"]])
    out["definition"] = [DEF[v][0] + " " + EXIT_TXT for v in out["variant"]]
    out["known_when"] = [DEF[v][1] for v in out["variant"]]
    out.to_csv(RESULTS / "b2_confirm_cells.csv", index=False, encoding="utf-8-sig")
    # 같은 사건 집합(즉시 진입 손익 유한) 수
    base = ev[ev["short_ret_h10"].notna()]
    cnt = {pk: {"events": int(H.in_period(base, p).sum()),
                "dates": int(base.loc[H.in_period(base, p), "t"].nunique())} for pk, p, _ in PERIODS}
    sj = {k: v for k, v in summ.items()}
    for nn in ("null", "null_shift"):
        if nn in sj:
            sj[nn] = {a: b for a, b in sj[nn].items() if a != "dist"}
    sj["same_event_set"] = cnt
    sj["trace"] = trace_rows(ev, P)
    sj["imm_reproduce_maxdiff"] = float((base["imm_chk"] - base["short_ret_h10"]).abs().max())
    bad, bady, nchk = cmd_check(ev)
    sj["independent_check"] = {"events": nchk, "status_mismatch": bad, "pnl_mismatch": bady}
    (RESULTS / "b2_confirm_summary.json").write_text(json.dumps(sj, ensure_ascii=False, indent=1, default=str),
                                                     encoding="utf-8")
    write_md(out, de, sj, cnt)
    cols = ["rank", "cell", "f_n_on", "f_dates_on", "f_diff", "f_mean_on", "b_n_on", "b_dates_on", "b_diff", "b_t",
            "b_mean_on", "b_p", "b_p_shift", "a_diff", "regime_top_share", "pass"]
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(out.sort_values("f_diff", ascending=False)[cols].to_string())
    print(json.dumps({k: sj[k] for k in ("n_cells", "n_eligible", "n_selected", "n_pass", "passed", "null",
                                         "null_shift", "same_event_set")}, ensure_ascii=False, indent=1, default=str))
    return out, de, sj


def _q(b, v, N, col):
    x = b[(b["variant"] == v) & (b["N"] == N)][col]
    return float(x.iloc[0]) if len(x) else np.nan


def _rng(x, **kw):
    lo, hi = _f(x.min(), **kw), _f(x.max(), **kw)
    return lo if lo == hi else f"{lo}~{hi}"


def summary_section(out, de):
    """결론과 뒤 구간 종목 핵심 표(모두 계산값에서 만든다)."""
    sel = out[out["selected"]]
    b = de[(de["period"] == "b") & (de["kind"] == "stock")]
    f = de[(de["period"] == "f") & (de["kind"] == "stock")]
    a = de[(de["period"] == "a") & (de["kind"] == "stock")]
    t2 = sel[sel["b_t"] >= 2]
    pc = dict(nd=0, sign=False)
    h3, bo = b[b["variant"] == "hold3"], b[b["variant"] == "bounce"]
    imm = b.drop_duplicates("N")["imm_mean"]
    L = ["## 결론", "",
         f"- **확인을 기다려도 뒤 구간(6~9월)에 공매도로 돈을 번 변형은 없다.** 선택 {len(sel)}칸 모두 뒤 구간 변형 거래당 손익이 음수"
         f"({_f(sel['b_mean_on'].min())} ~ {_f(sel['b_mean_on'].max())}). 뒤 차이의 날짜 묶음 t ≥ 2는 {len(t2)}칸"
         + (f"({', '.join(t2['cell'])}, 순열 p {', '.join(_f(x, False, 3) for x in t2['b_p'])})" if len(t2) else "")
         + ". 통과 0칸.",
         f"- **확인은 가짜를 잘 거른다(사후 분류 기준).** 뒤 구간 종목 hold3: 가짜 걸러냄 {_rng(h3['fake_filtered'], **pc)}, 진짜 유지 {_rng(h3['true_kept'], **pc)}. "
         f"bounce: 가짜 걸러냄 {_rng(bo['fake_filtered'], **pc)}, 진짜 유지 {_rng(bo['true_kept'], **pc)}. 진입한 사건 중 진짜 비율은 "
         f"hold3 {_rng(h3['trades_true_share'], **pc)}, bounce {_rng(bo['trades_true_share'], **pc)}로 전체 사건({_rng(b['events_true_share'], **pc)})보다 높지만 절반 아래다.",
         f"- **늦게 알아서 못 번다.** 확인된 사건에 사후로 즉시 진입했다면 뒤 구간 N20 종목 hold3 {_f(_q(b, 'hold3', 20, 'imm_mean_on_trig'))}, "
         f"bounce {_f(_q(b, 'bounce', 20, 'imm_mean_on_trig'))}였다. 확인 뒤 진입하면 {_f(_q(b, 'hold3', 20, 'var_mean_trade'))}, "
         f"{_f(_q(b, 'bounce', 20, 'var_mean_trade'))}다. 놓친 하락 평균은 {_f(_q(b, 'hold3', 20, 'missed_mean'))}, {_f(_q(b, 'bounce', 20, 'missed_mean'))}다.",
         f"- **즉시 진입보다 덜 잃는 것은 대부분 '안 들어가서'다(설명용).** 뒤 구간 종목 사건당 차이는 {_f(b['opp_diff'].min())} ~ {_f(b['opp_diff'].max())}다. "
         f"그중 거른 이득이 {_f(b['skip_gain'].min())} ~ {_f(b['skip_gain'].max())}, 기다린 값(진입한 사건의 변형 − 즉시, 음수 = 비용)은 {_f(b['wait_cost'].min())} ~ {_f(b['wait_cost'].max())}다. "
         f"즉시 진입 자체가 뒤 구간에서 거래당 {_f(imm.max())} ~ {_f(imm.min())}로 지는 장이라, 덜 거래할수록 덜 잃는다.",
         f"- **앞 구간 차이 1~3위(bounce 종목)도 뒤에서는 거래당 손실이다.** 거래당 앞 {', '.join(_f(x) for x in f[f.variant == 'bounce'].sort_values('N')['var_mean_trade'])}"
         f"(N20·40·60) → 뒤 {', '.join(_f(x) for x in bo.sort_values('N')['var_mean_trade'])} → 2025 "
         f"{', '.join(_f(x) for x in a[a.variant == 'bounce'].sort_values('N')['var_mean_trade'])}.",
         "- ETF 칸은 앞 구간 변형 거래가 칸마다 0~6건이라 고르기 조건(켜짐 ≥ 15건·8일)을 넘지 못했다. 뒤 구간 ETF 변형 거래도 칸마다 0~5건이다.",
         "", "## 뒤 구간(6~9월) 종목 핵심 표", "",
         "| 변형 | N | 사건/날짜 | 진입률 | 가짜 걸러냄 | 진짜 유지 | 진입 중 진짜(전체) | 즉시 진입: 전체 / 진입한 사건만 | 변형 거래당 (t) | 놓친 하락 | 기다린 날 |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in b.itertuples():
        L.append(f"| {r.variant} | {r.N} | {r.events}/{r.event_dates} | {_f(r.trade_rate, **pc)} | {_f(r.fake_filtered, **pc)} | "
                 f"{_f(r.true_kept, **pc)} | {_f(r.trades_true_share, **pc)} ({_f(r.events_true_share, **pc)}) | "
                 f"{_f(r.imm_mean)} / {_f(r.imm_mean_on_trig)} | {_f(r.var_mean_trade)} ({_f(r.var_t_trade, False)}) | {_f(r.missed_mean)} | "
                 f"{_f(r.wait_days_mean, False, 1)} |")
    L += ["", "- 한계: 확인 창(t+5), 재시험 띠(±0.3 ATR), 반등 정의는 한 가지씩만 정해 시험했다(조율 없음). 청산 끝을 t+10으로 고정해서 "
          "늦게 들어갈수록 보유 기간이 짧다(t+20 끝 거래당 손익은 cells.csv의 `*_var_h20_trade`). 대차 비용과 배당 지급은 빠졌다. "
          "가짜/진짜 분류는 t+10까지의 결과를 쓰는 사후 설명용이다. 뒤 구간 사건은 episode 34·35 두 국면에 몰려 있다. 선택 칸은 뒤 구간 손익 합이 음수라 "
          "한 국면 비중을 정의할 수 없다(하네스는 합 > 0일 때만 계산한다).",
          "- 탐색 범위: 변형 5개 × N 3개 × 유형 2개 = 30칸을 한 번 돌렸다. 앞선 박스·손절·레버리지 연구를 본 뒤의 탐색이라 선택편향이 쌓여 있다.", ""]
    return L


def write_md(out, de, sj, cnt):
    L = []
    nul, nsh = sj.get("null", {}), sj.get("null_shift", {})
    sel = out[out["selected"]].sort_values("rank")
    L += ["# 갈래 2 — 확인 대기 진입 변형 (b2_confirm)", "",
          f"- 칸 {sj['n_cells']}개 = 변형 5 × N 3 × 유형 2(종목/ETF). 고르기 조건 충족 {sj['n_eligible']}칸, 선택 {sj['n_selected']}칸, "
          f"**통과 {sj['n_pass']}칸**{(' — ' + ', '.join(sj['passed'])) if sj['passed'] else ''}.",
          f"- 선택편향 귀무(날짜 묶음 i.i.d. 200회): 통과 칸 수 평균 {nul.get('mean', float('nan')):.3f}, 95% 분위 {nul.get('q95', float('nan')):g}, "
          f"관측 {sj['n_pass']}칸의 p = {nul.get('p_value', float('nan')):.3f}. 보조 원형 이동 귀무: 평균 {nsh.get('mean', float('nan')):.3f}, p = {nsh.get('p_value', float('nan')):.3f}.",
          f"- 같은 사건 집합(즉시 진입 손익이 있는 사건, N 3개·종목+ETF 합): 앞 {cnt['f']['events']}건/{cnt['f']['dates']}일, "
          f"뒤 {cnt['b']['events']}건/{cnt['b']['dates']}일, 2025 {cnt['a']['events']}건/{cnt['a']['dates']}일.",
          f"- 즉시 진입 재현 점검: 변형 계산 코드로 t+1 시가 진입을 다시 계산한 값과 하네스 short_ret_h10의 최대 차이 {sj['imm_reproduce_maxdiff']:.2e}. "
          f"독립 재계산(`py b2_confirm.py check`, 긴 형식 일봉·별도 코드) {sj['independent_check']['events']}사건 × 5변형: 상태·확인일 불일치 "
          f"{sum(sj['independent_check']['status_mismatch'].values())}, 손익 불일치 {sum(sj['independent_check']['pnl_mismatch'].values())}. "
          "확인일 뒤 가격을 ×U(0.6, 1.4)로 바꿔도 상태·확인일 변화 0(300사건 × 5변형, `features`).",
          ""]
    L += summary_section(out, de)
    L += ["## 정의(결과 전에 고정)", ""]
    for v in VARIANTS:
        L.append(f"- `{v}`: {DEF[v][0]}. 알 수 있는 때: {DEF[v][1]}.")
    L += [f"- 공통: {EXIT_TXT} 즉시 진입과 끝 날짜·되돌림 날이 같아서 '기다린 값'만 다르다.",
          "- 판정 표는 evaluate_cells 그대로다. 꺼짐 = 같은 N·유형 모든 사건의 즉시 진입(거래당), 켜짐 = 변형이 진입한 사건의 변형 손익(거래당). "
          "차이 = 변형 거래당 − 즉시 진입 거래당. 날짜 묶음 t·순열은 이탈일 t로 묶고 순열 층은 N·유형·변형.",
          "- 초과 손익(같은 날 시장 효과 뺀 값)은 못 구함: 변형은 보유 창이 사건마다 달라 하네스의 같은 날 대조를 그대로 쓸 수 없다. "
          "이 갈래의 대조는 같은 사건의 즉시 진입이다.", ""]
    L += ["## 판정 표 (앞 구간 차이 순, 30칸 전체)", "",
          "| 순위 | 칸 | 앞 켜짐(거래/날짜) | 앞 차이 | 앞 변형 거래당 | 뒤 켜짐(거래/날짜) | 뒤 차이 | 뒤 t | 뒤 p | 뒤 변형 거래당 | 2025 차이 | 국면 최대 | 통과 |",
          "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in out.sort_values("f_diff", ascending=False).itertuples():
        L.append(f"| {'' if pd.isna(r.rank) else int(r.rank)} | {r.cell} | {int(r.f_n_on)}/{int(r.f_dates_on)} | {_f(r.f_diff)} | {_f(r.f_mean_on)} | "
                 f"{int(r.b_n_on)}/{int(r.b_dates_on)} | {_f(r.b_diff)} | {_f(r.b_t, False)} | {_f(r.b_p, False, 3)} | {_f(r.b_mean_on)} | "
                 f"{_f(r.a_diff)} | {_f(r.regime_top_share, nd=0, sign=False)} | {'통과' if r._asdict().get('pass') else ('선택·탈락' if r.selected else '-')} |")
    L.append("")
    L += ["## 같은 사건 집합 비교 (설명용, 판정 아님)", "",
          "사건당 = 진입 안 하면 0. 차이 분해: 사건당 차이 = 기다린 값(진입한 사건의 변형 − 즉시, 음수 = 비용) + 거른 이득(진입 안 한 사건의 −즉시). 둘 다 사건 수로 나눈 값. "
          "놓친 하락 = (시가 t+1 − 진입 시가)/시가 t+1, 진입한 사건 평균. 가짜 걸러냄 = '가짜' 중 진입 안 한 비율, 진짜 유지 = '진짜' 중 진입한 비율.", ""]
    for pk, _, pn in PERIODS:
        L += [f"### {pn}", "",
              "| 칸 | 사건/날짜 | 즉시 거래당 (t) | 진입률 | 변형 거래당 (t) | 사건당 차이 (t) | 기다린 값 | 거른 이득 | 놓친 하락 평균/중앙 | 가짜 걸러냄 | 진짜 유지 | 진입 중 진짜/가짜 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in de[de["period"] == pk].itertuples():
            L.append(f"| {r.cell} | {r.events}/{r.event_dates} | {_f(r.imm_mean)} ({_f(r.imm_t, False)}) | {_f(r.trade_rate, nd=0, sign=False)} | "
                     f"{_f(r.var_mean_trade)} ({_f(r.var_t_trade, False)}) | {_f(r.opp_diff)} ({_f(r.opp_diff_t, False)}) | {_f(r.wait_cost)} | {_f(r.skip_gain)} | "
                     f"{_f(r.missed_mean)} / {_f(r.missed_median)} | {_f(r.fake_filtered, nd=0, sign=False)} | {_f(r.true_kept, nd=0, sign=False)} | "
                     f"{_f(r.trades_true_share, nd=0, sign=False)}/{_f(r.trades_fake_share, nd=0, sign=False)} |")
        L.append("")
    L += ["## 손 추적 예시 (뒤 구간 N20 종목)", ""]
    for x in sj["trace"]:
        L.append(f"- `{x['variant']}` {x['sym']} t={x['t']}, L={x['L']:.3f}, ATR={x['atr']:.3f}; t+1..d 종가 {x['closes_t1_td']}, 고가 {x['highs_t1_td']}; "
                 f"확인 {x['d']} → 진입 시가 {x['entry']:.3f}(t+1 시가 {x['o_t1']:.3f}, 놓친 하락 {_f(x['missed'])}); "
                 f"즉시 {_f(x['imm'])}, 변형 {_f(x['var'])}({x['why']}).")
    L.append("")
    Path(RESULTS / "b2_confirm.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "features":
        cmd_features()
    elif cmd == "check":
        print(cmd_check())
    elif cmd == "run":
        cmd_run()
