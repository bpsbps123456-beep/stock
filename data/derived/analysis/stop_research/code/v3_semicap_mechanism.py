"""v3 관점 2: 갈래 I 후보 'S2 0% 현금 주간'의 작동 원리 분해·사건별 점검(2026-10-10).

후보: 재조정일(ISO 주 첫 거래일) 시가에 직전 거래일 종가 기준 SOXX 종가 < SMA20(종가 20일, t 포함)이면
      그 주 반도체+미분류 합계 한도 0(넘친 몫 현금, 비반도체 비중은 현행 목표 그대로), 아니면 현행 50%.

질문: 이득이 '반도체만 깎기'에서 나오는가, '그 주에 현금을 늘린 시점 운'인가.
  1) 같은 계기·같은 재조정일·같은 현금 증가(= 그 주 목표의 반도체+미분류 합계)로 깎는 대상만 바꾼 4판
       i    반도체만 깎기(후보)
       ii   7종목 전체를 같은 비율로(현금 증가 같음)
       iii  비반도체 먼저(위약). 비반도체가 모자라면 남은 몫을 반도체에서 비례로 깎아 현금 증가를 맞춘다
       iiib 비반도체만(위약, 순수). 비반도체가 모자라면 덜 깎는다(현금 증가가 작을 수 있음)
     on 주에서 i − ii ≈ N·S/(S+N) × (비반도체 묶음 수익 − 반도체 묶음 수익) → '반도체만'의 값은 묶음 간 수익 차로 정해진다.
  2) 위약 시점: 주 단위 켜짐 배열을 원형으로 전부 이동(2026 41가지, 보유 기간 137가지) + 무작위 순열 1,000개(같은 켜진 주 수)
  3) 사건별 표(시가→시가): lo 시가 → hi 다음 거래일 시가. 기본·i·ii·iii 수익, 그 사건을 다스린 주 신호, 평균 구성
  4) 같은 최대 낙폭 고정 노출 효율선(c = 0~1, 0.005 간격, 매번 실제 재계산), USD / +현금 이자 연 4% / 원화(USD 현금 유지) / 원화+이자
  5) 지금 상태: 2026-10-09 종가 SOXX·SMA20 → 2026-10-12 시가 한도, 현재 7종목 반도체+미분류(base_2026.json 마지막 날)

엔진: v2_I_semicap.Engine(데스크 weekly_target·_drift, desk는 import만)의 재조정 결정을 그대로 쓰고 깎는 방식만 바꾼다.
      계기를 끄면 desk run_model과 같아야 하고(I.check_engine), i판은 I.Engine.run(S2, 0, 현금, 주간)과 같아야 한다.
공유 파일(harness·build_base·v2_*·fam_*·cache·기존 results)은 고치지 않는다. config.RULE은 건드리지 않는다(실행 끝에 확인).
실행: cd code && PYTHONIOENCODING=utf-8 py v3_semicap_mechanism.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness as H  # noqa: E402
import v2_I_semicap as I  # noqa: E402  (Engine·계기, import만)
from desk import config, strategies as S  # noqa: E402  (import만)

ID = "v3_semicap_mechanism"
RATE = 0.04
SEED = 20261010
N_PERM = 1000
MODES = {"i": "반도체만 깎기(후보)", "ii": "7종목 같은 비율", "iii": "비반도체 먼저(모자라면 반도체)",
         "iiib": "비반도체만(모자라면 덜 깎음)"}
FLAVORS = ("USD", "USD+이자", "원화", "원화+이자")
CLAIM = {"2026 전체": (2.172, -0.142), "2026 앞": (1.903, -0.142), "2026 뒤": (0.093, -0.109), "AI쇼크": (-0.082, None),
         "7~9월": (0.186, None), "2025": (0.484, -0.351), "2024": (0.269, -0.265), "평균노출": 0.78}
CLAIM_BASE = {"2026 전체": (2.237, -0.203), "2026 뒤": (0.046, -0.203), "AI쇼크": (-0.187, None), "2025": (0.422, -0.471),
              "2024": (0.334, -0.382)}
WIN = [("2026", "전체"), ("2026", "앞"), ("2026", "뒤"), ("2026", "AI쇼크"), ("2026", "7~9월"), ("2026", "9/1~"),
       ("hold", "2025"), ("hold", "2024"), ("hold", "관세"), ("hold", "24여름"), ("hold", "25가을"), ("hold", "23가을")]
FRONT_WIN = [("2026", "전체"), ("2026", "앞"), ("2026", "뒤"), ("hold", "2025"), ("hold", "2024")]
EVENTS = [("2026", "26 2/3~2/5", "2026-02-03", "2026-02-05"), ("2026", "26 2/25~3/6", "2026-02-25", "2026-03-06"),
          ("2026", "26 3/19~3/30", "2026-03-19", "2026-03-30"), ("2026", "26 5/13~5/19", "2026-05-13", "2026-05-19"),
          ("2026", "26 5/13~5/18(바닥까지)", "2026-05-13", "2026-05-18"),
          ("2026", "26 AI쇼크 6/22~7/29", "2026-06-22", "2026-07-29"), ("2026", "26 7/30~9/30", "2026-07-30", "2026-09-30"),
          ("2026", "26 6/22~9/30(반등 포함)", "2026-06-22", "2026-09-30"),
          ("2026", "26 8월 하순 8/17~9/11", "2026-08-17", "2026-09-11"), ("2026", "26 9/14~10/9", "2026-09-14", "2026-10-09"),
          ("hold", "25 딥시크 1/23~1/27", "2025-01-23", "2025-01-27"), ("hold", "25 관세 2/18~4/4", "2025-02-18", "2025-04-04"),
          ("hold", "25 반등 4/7~6/30", "2025-04-07", "2025-06-30"), ("hold", "25가을 10/13~11/21", "2025-10-13", "2025-11-21"),
          ("hold", "24 여름 6/18~8/5", "2024-06-18", "2024-08-05"), ("hold", "24 10~11월 10/1~11/29", "2024-10-01", "2024-11-29"),
          ("hold", "23 가을 8/1~10/26", "2023-08-01", "2023-10-26")]
NEXT_REB = "2026-10-12"


def rnd(x, n=4):
    if isinstance(x, dict):
        return {str(k): rnd(v, n) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rnd(v, n) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else round(float(x), n)
    if isinstance(x, np.integer):
        return int(x)
    return x


def pct(x, d=1):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:+.{d}f}"


def md(x, d=1):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:.{d}f}"


# ============================================================================ 깎기 4판
def cut(eng, t: dict, mode: str) -> dict:
    sem = eng.semis
    s = sum(v for k, v in t.items() if k in sem)
    inv = sum(t.values())
    n = inv - s
    if s <= 1e-12:
        return dict(t)
    if mode == "i":
        out = {k: v for k, v in t.items() if k not in sem}
    elif mode == "ii":
        f = n / inv
        out = {k: v * f for k, v in t.items()}
    elif mode == "iii":
        if n >= s:
            f = (n - s) / n
            out = {k: (v if k in sem else v * f) for k, v in t.items()}
        else:
            g = n / s
            out = {k: (v * g if k in sem else 0.0) for k, v in t.items()}
    elif mode == "iiib":
        c = min(s, n)
        f = (n - c) / n if n > 1e-12 else 0.0
        out = {k: (v if k in sem else v * f) for k, v in t.items()}
    else:
        raise ValueError(mode)
    return {k: v for k, v in out.items() if v > 1e-12}


class Path:
    def __init__(self, r, W, eng):
        self.r = r
        self.W = W
        idx = r.index
        n = len(idx)
        self.inv = pd.Series([sum(W[i].values()) for i in range(n)], index=idx)
        self.cash = 1.0 - self.inv
        self.semis = pd.Series([sum(v for k, v in W[i].items() if k in eng.semis) for i in range(n)], index=idx)
        self.unk = pd.Series([sum(v for k, v in W[i].items() if eng.sector[k] == S.UNKNOWN) for i in range(n)], index=idx)
        self.non = self.inv - self.semis


def run(eng, sig, mode="i") -> Path:
    days = eng.b.days
    w: dict = {}
    val, W = [1.0], []
    for i, day in enumerate(days):
        k = eng.k0 + i
        if i:
            w, g = S._drift(w, eng.rets(w, k))
            val.append(val[-1] * (1 + g))
        if i in eng.dec:
            t = eng.dec[i]["target"]
            w = cut(eng, t, mode) if (sig is not None and bool(sig[i])) else dict(t)
        W.append(dict(w))
    v = pd.Series(val, index=days)
    return Path((v.shift(-1) / v - 1).dropna(), W, eng)


# ============================================================================ 수익 변환·지표
class Ctx:
    def __init__(self, bn, eng):
        self.bn, self.eng, self.b = bn, eng, eng.b
        fx = H.px("KRW=X", "Close")
        f = fx.reindex(sorted(set(fx.index) | set(self.b.days))).ffill().reindex(self.b.days)
        self.fx_missing = int(f.isna().sum())
        self.fr = (f.shift(-1) / f).reindex(self.b.ret.index).fillna(1.0)
        self.base = run(eng, None)
        self.soxx = H.open_ret("SOXX", self.b)
        # 기본 경로 반도체 묶음·비반도체 묶음 일별 수익(기본 비중 기준)
        rs, rn = [], []
        for i, d in enumerate(self.b.ret.index):
            w = self.base.W[i]
            rr = eng.rets(w, eng.k0 + i + 1)
            ws = sum(v for k, v in w.items() if k in eng.semis)
            wn = sum(v for k, v in w.items() if k not in eng.semis)
            rs.append(sum(v * rr[k] for k, v in w.items() if k in eng.semis) / ws if ws > 1e-12 else np.nan)
            rn.append(sum(v * rr[k] for k, v in w.items() if k not in eng.semis) / wn if wn > 1e-12 else np.nan)
        self.g_semi = pd.Series(rs, index=self.b.ret.index)
        self.g_non = pd.Series(rn, index=self.b.ret.index)
        self.reb = sorted(eng.dec)
        self._grid = {}

    def flav(self, r, cash, fl):
        if fl == "USD":
            return r
        ri = r + cash * RATE / 252
        if fl == "USD+이자":
            return ri
        if fl == "원화":
            return (1 + r) * self.fr - 1
        return (1 + ri) * self.fr - 1

    def pflav(self, p: Path, fl):
        return self.flav(p.r, p.cash, fl)

    def grid(self, fl, key):
        """고정 노출 c × 기본 경로(c = 0~1, 0.005 간격). 현금 = 1 − c × 기본 투자비중."""
        if (fl, key) not in self._grid:
            lo, hi = self.b.periods[key]
            out = []
            for c in np.round(np.arange(0.0, 1.0001, 0.005), 3):
                r = c * self.b.ret
                cash = 1.0 - c * self.base.inv
                out.append((float(c), H.stats(self.flav(r, cash, fl), lo, hi)))
            self._grid[(fl, key)] = out
        return self._grid[(fl, key)]

    def frontier(self, fl, key, v: dict) -> dict:
        ok = [(c, g) for c, g in self.grid(fl, key) if g["낙폭"] >= v["낙폭"] - 1e-9]
        c, g = max(ok, key=lambda x: (x[1]["수익"], -x[0]))
        return {"c": c, "고정수익": g["수익"], "고정낙폭": g["낙폭"], "초과": v["수익"] - g["수익"]}


def wstats(ctx: Ctx, p: Path, fl="USD") -> dict:
    r = ctx.pflav(p, fl)
    return {k: H.stats(r, *v) for k, v in ctx.b.periods.items()}


def rel_e(ctx: Ctx, p: Path) -> pd.Series:
    bi = ctx.base.inv
    return pd.Series(np.where(bi > 1e-9, p.inv / bi.where(bi > 1e-9, 1.0), 1.0), index=bi.index)


# ============================================================================ 주 단위
def weeks(ctx: Ctx):
    """재조정 주 j: (시작일, 끝 ret 날짜). 끝 = 다음 재조정일 전날(시가 시작 → 다음 재조정일 시가)."""
    days, ri = ctx.b.days, ctx.b.ret.index
    out = []
    for j, i in enumerate(ctx.reb):
        nxt = ctx.reb[j + 1] if j + 1 < len(ctx.reb) else len(days) - 1
        out.append((days[i], days[nxt - 1] if nxt - 1 < len(ri) else ri[-1]))
    return out


def on_weeks(ctx: Ctx, sig) -> np.ndarray:
    return np.array([bool(sig[i]) for i in ctx.reb])


def sig_from(ctx: Ctx, on: np.ndarray) -> np.ndarray:
    s = np.zeros(len(ctx.b.days), bool)
    s[ctx.reb] = on
    return s


def comp_ret(x: pd.Series, lo, hi):
    y = x[(x.index >= lo) & (x.index <= hi)].dropna()
    return float(np.prod(1 + y.to_numpy()) - 1) if len(y) else float("nan")


def week_table(ctx: Ctx, on: np.ndarray, lo: str, hi: str) -> dict:
    """켜진 주 vs 꺼진 주: 기본 경로·반도체 묶음·비반도체 묶음·SOXX 주간 수익(그 주 재조정 시가 → 다음 재조정 시가)."""
    rows = []
    for (a, b_), o in zip(weeks(ctx), on):
        if not (lo <= a <= hi):
            continue
        rows.append({"주": a, "켬": bool(o), "기본": comp_ret(ctx.b.ret, a, b_), "반도체": comp_ret(ctx.g_semi, a, b_),
                     "비반도체": comp_ret(ctx.g_non, a, b_), "SOXX": comp_ret(ctx.soxx, a, b_),
                     "반도체비중": float(ctx.base.semis[(ctx.base.semis.index >= a)].iloc[0])})
    df = pd.DataFrame(rows)
    df["차(비반도체−반도체)"] = df["비반도체"] - df["반도체"]
    out = {}
    for lab, m in (("켬", df["켬"]), ("끔", ~df["켬"])):
        x = df[m]
        out[lab] = {"주수": int(len(x)), "기본 평균": float(x["기본"].mean()), "반도체 평균": float(x["반도체"].mean()),
                    "비반도체 평균": float(x["비반도체"].mean()), "차 평균": float(x["차(비반도체−반도체)"].mean()),
                    "차 중앙": float(x["차(비반도체−반도체)"].median()),
                    "비반도체가 이긴 주 비율": float((x["차(비반도체−반도체)"] > 0).mean()) if len(x) else float("nan"),
                    "SOXX 평균": float(x["SOXX"].mean())}
    return {"요약": out, "주별": df.to_dict("records")}


def spells(ctx: Ctx, on: np.ndarray, paths: dict) -> list:
    """켜진 주가 이어진 묶음(spell)별 수익. 묶음 밖에서는 i·ii·iii가 기본과 같은 비중이라 수익이 같다 →
    log(1+i 전체) − log(1+기본 전체) = Σ 묶음별 log 차(정확한 분해)."""
    wk = weeks(ctx)
    out, j = [], 0
    while j < len(on):
        if not on[j]:
            j += 1
            continue
        j0 = j
        while j < len(on) and on[j]:
            j += 1
        a, b_ = wk[j0][0], wk[j - 1][1]
        row = {"시작": a, "끝(ret)": b_, "주수": j - j0}
        for m, p in paths.items():
            row[m] = comp_ret(p.r, a, b_)
        row["반도체묶음"] = comp_ret(ctx.g_semi, a, b_)
        row["비반도체묶음"] = comp_ret(ctx.g_non, a, b_)
        row["SOXX"] = comp_ret(ctx.soxx, a, b_)
        m_ = (ctx.base.semis.index >= a) & (ctx.base.semis.index <= b_)
        row["기본 반도체·비반도체·현금"] = [float(ctx.base.semis[m_].mean()), float(ctx.base.non[m_].mean()),
                                    float(ctx.base.cash[m_].mean())]
        w0 = ctx.base.W[ctx.b.days.index(a)]
        tk = ctx.eng.book.ticker
        row["비반도체 종목(시작일 기본)"] = {tk[k]: round(v, 3) for k, v in sorted(w0.items(), key=lambda x: -x[1])
                                     if k not in ctx.eng.semis}
        row["반도체예산 종목(시작일 기본)"] = {tk[k]: round(v, 3) for k, v in sorted(w0.items(), key=lambda x: -x[1])
                                       if k in ctx.eng.semis}
        row["log(i/기본)"] = float(np.log1p(row["i"]) - np.log1p(row["기본"]))
        row["log(i/ii)"] = float(np.log1p(row["i"]) - np.log1p(row["ii"]))
        out.append(row)
    return out


# ============================================================================ 사건
def event_row(ctx: Ctx, paths: dict, on: np.ndarray, lab, lo, hi) -> dict:
    wk = weeks(ctx)
    gov = [(a, bool(o)) for (a, b_), o in zip(wk, on) if b_ >= lo and a <= hi]
    row = {"사건": lab, "구간": f"{lo} 시가 → {hi} 다음 거래일 시가", "주 신호": gov}
    for m, p in paths.items():
        row[m] = comp_ret(p.r, lo, hi)
    row["반도체묶음(기본)"] = comp_ret(ctx.g_semi, lo, hi)
    row["비반도체묶음(기본)"] = comp_ret(ctx.g_non, lo, hi)
    row["SOXX"] = comp_ret(ctx.soxx, lo, hi)
    m_ = (ctx.base.inv.index >= lo) & (ctx.base.inv.index <= hi)
    row["구성"] = {m: {"반도체": float(paths[m].semis[m_].mean()), "미분류": float(paths[m].unk[m_].mean()),
                     "비반도체": float(paths[m].non[m_].mean()), "현금": float(paths[m].cash[m_].mean())}
                 for m in ("기본", "i", "ii")}
    return row


# ============================================================================ 위약 시점
def placebo(ctx: Ctx, on: np.ndarray, keys, front_keys, rng) -> dict:
    n = len(on)
    shifts = []
    for s in range(n):
        o = np.roll(on, s)
        sg = sig_from(ctx, o)
        pi, pii = run(ctx.eng, sg, "i"), run(ctx.eng, sg, "ii")
        rec = {"이동": s}
        for tag, p in (("i", pi), ("ii", pii)):
            st = wstats(ctx, p)
            for k in keys:
                rec[f"{tag} {k} 수익"] = st[k]["수익"]
                rec[f"{tag} {k} 낙폭"] = st[k]["낙폭"]
            for k in front_keys:
                rec[f"{tag} {k} 효율선초과"] = ctx.frontier("USD", k, st[k])["초과"]
        shifts.append(rec)
    perms = []
    for _ in range(N_PERM):
        o = rng.permutation(on)
        p = run(ctx.eng, sig_from(ctx, o), "i")
        st = wstats(ctx, p)
        rec = {}
        for k in keys:
            rec[f"i {k} 수익"] = st[k]["수익"]
            rec[f"i {k} 낙폭"] = st[k]["낙폭"]
        for k in front_keys:
            rec[f"i {k} 효율선초과"] = ctx.frontier("USD", k, st[k])["초과"]
        perms.append(rec)
    df, dp = pd.DataFrame(shifts), pd.DataFrame(perms)
    act = df.iloc[0]
    rank = {}
    for col in df.columns:
        if col == "이동":
            continue
        a = act[col]
        better = int((df[col] > a + 1e-12).sum())
        rank[col] = {"실제": float(a), "원형이동 순위": better + 1, "원형이동 수": int(n),
                     "원형이동 중앙": float(df[col].median()), "원형이동 최소": float(df[col].min()),
                     "원형이동 최대": float(df[col].max())}
        if col in dp.columns:
            rank[col]["순열 중 실제 이상 비율"] = float((dp[col] >= a - 1e-12).mean())
            rank[col]["순열 중앙"] = float(dp[col].median())
    # 같은 이동(같은 시점)에서 i가 ii를 이긴 비율 → '반도체만'이 시점과 무관하게 좋은가
    vs = {}
    for k in keys:
        d = df[f"i {k} 수익"] - df[f"ii {k} 수익"]
        vs[k] = {"i−ii 실제": float(d.iloc[0]), "i−ii 원형이동 중앙": float(d.median()),
                 "i>ii 비율(원형이동)": float((d > 0).mean()), "i−ii 순위": int((d > d.iloc[0] + 1e-12).sum()) + 1}
    return {"순위": rank, "i_vs_ii": vs, "원형이동": df.to_dict("records")}


# ============================================================================ 지금 상태
def current_state(ctx26: Ctx, sig_raw: pd.Series) -> dict:
    c = H.px("SOXX", "Close")
    c = c[c.index <= "2026-10-09"]
    sma = c.rolling(20).mean()
    t = c.index[-1]
    prior19 = float(c.iloc[-20:-1].mean())
    eng, b = ctx26.eng, ctx26.b
    last = b.days[-1]
    w = b.weights[-1]
    bk = eng.book
    k = bk.days.index(last)
    cl = {key: S._col(bk, bk.cls, key) for key in w}
    op = {key: S._col(bk, bk.opn, key) for key in w}
    wc = {key: v * cl[key][k] / op[key][k] for key, v in w.items()}
    tot = sum(wc.values()) + (1 - sum(w.values()))
    wc = {key: v / tot for key, v in wc.items()}
    rows = [{"종목": b.ticker[key], "섹터": b.sector[key], "반도체예산": bool(S.in_semis_budget(b.sector[key])),
             "10/9 시가 거래 뒤": w[key], "10/9 종가 평가": wc[key]} for key in sorted(w, key=lambda x: -w[x])]
    on = bool(c.iloc[-1] < sma.iloc[-1])
    # 2026 최근 재조정일 신호
    sig26 = I.signals_at_open("2026")["S2"]
    hist = [(b.days[i], bool(sig26[i])) for i in ctx26.reb[-10:]]
    # 참고: 고정본 가격(10/9까지)·PDF로 데스크 next_target을 그대로 부른 10/12 선택(데스크가 월요일에 다시 계산)
    path = S.run_model(bk, start=b.days[0])
    nt = S.next_target(bk, path, NEXT_REB)
    ref = {"pdf": nt.get("pdf"), "재조정일": bool(nt["rebalance_day"]),
           "선택": [bk.ticker[x] for x in nt["selected"]],
           "반도체예산 종목": [bk.ticker[x] for x in nt["selected"] if S.in_semis_budget(bk.sector[x])],
           "목표 반도체예산 합": float(sum(v for x, v in nt["target"].items() if S.in_semis_budget(bk.sector[x]))),
           "목표 비반도체 합": float(sum(v for x, v in nt["target"].items() if not S.in_semis_budget(bk.sector[x])))}
    return {"기준일": t, "SOXX 종가": float(c.iloc[-1]), "SMA20(t 포함)": float(sma.iloc[-1]),
            "직전 19일 종가 평균(= 켜짐 문턱)": prior19, "종가/SMA20 − 1": float(c.iloc[-1] / sma.iloc[-1] - 1),
            "계기 켜짐": on, "10/12 시가 반도체+미분류 한도": 0.0 if on else 0.5,
            "raw 신호 확인(I.raw_signals)": bool(sig_raw.loc[t] > 0.5),
            "최근 재조정일 신호(시가 적용)": hist, "보유(base_2026 마지막 날)": rows,
            "반도체예산 합(10/9 시가 뒤)": float(sum(r["10/9 시가 거래 뒤"] for r in rows if r["반도체예산"])),
            "반도체예산 합(10/9 종가)": float(sum(r["10/9 종가 평가"] for r in rows if r["반도체예산"])),
            "참고_10/12 데스크 선택(고정본)": ref}


# ============================================================================ 본체
def main():
    saved_rule = dict(config.RULE)
    rng = np.random.default_rng(SEED)
    engs = {bn: I.Engine(bn) for bn in ("2026", "hold")}
    chk = {bn: I.check_engine(e) for bn, e in engs.items()}
    print("엔진 확인(계기 끔 vs desk run_model):", chk)
    for bn, c in chk.items():
        if max(c.values()) > 1e-9:
            raise SystemExit(f"{bn}: 엔진이 run_model과 다르다 {c}")
    ctx = {bn: Ctx(bn, e) for bn, e in engs.items()}
    sig = {bn: I.signals_at_open(bn)["S2"] for bn in engs}
    sig_raw = I.raw_signals()["S2"]

    paths, on, repro = {}, {}, {}
    for bn, c in ctx.items():
        on[bn] = on_weeks(c, sig[bn])
        paths[bn] = {"기본": c.base}
        for m in MODES:
            paths[bn][m] = run(c.eng, sig[bn], m)
        r_ref, w_ref, _ = c.eng.run(sig[bn], 0.0, "현금", "주간")
        repro[bn] = {"i vs I.Engine 최대 차": float(np.max(np.abs(r_ref.to_numpy() - paths[bn]["i"].r.to_numpy()))),
                     "기본 vs base json 최대 차": float(np.max(np.abs(c.base.r.to_numpy() - c.b.ret.to_numpy()))),
                     "현금 증가 같음(i vs ii, 재조정일 최대 차)": float(max(
                         abs(paths[bn]["i"].inv.get(c.b.days[i], np.nan) - paths[bn]["ii"].inv.get(c.b.days[i], np.nan))
                         for i in c.reb if c.b.days[i] in paths[bn]["i"].inv.index)),
                     "iii 현금 증가 같음(재조정일 최대 차)": float(max(
                         abs(paths[bn]["i"].inv.get(c.b.days[i], np.nan) - paths[bn]["iii"].inv.get(c.b.days[i], np.nan))
                         for i in c.reb if c.b.days[i] in paths[bn]["i"].inv.index))}
    print("재현 확인:", repro)
    for bn, x in repro.items():
        if x["i vs I.Engine 최대 차"] > 1e-12 or x["기본 vs base json 최대 차"] > 1e-9:
            raise SystemExit(f"{bn}: 재현 실패 {x}")

    # ---- 1) 주장 대조 + 4판 비교(USD)
    st = {bn: {m: wstats(ctx[bn], p) for m, p in paths[bn].items()} for bn in ctx}
    e_i = {bn: float(rel_e(ctx[bn], paths[bn]["i"]).mean()) for bn in ctx}
    claim_chk = {
        "2026 전체": st["2026"]["i"]["전체"], "2026 앞": st["2026"]["i"]["앞"], "2026 뒤": st["2026"]["i"]["뒤"],
        "AI쇼크": st["2026"]["i"]["AI쇼크"], "7~9월": st["2026"]["i"]["7~9월"], "2025": st["hold"]["i"]["2025"],
        "2024": st["hold"]["i"]["2024"], "평균노출": e_i["2026"],
        "기본": {"2026 전체": st["2026"]["기본"]["전체"], "2026 뒤": st["2026"]["기본"]["뒤"],
               "AI쇼크": st["2026"]["기본"]["AI쇼크"], "2025": st["hold"]["기본"]["2025"], "2024": st["hold"]["기본"]["2024"]}}
    print("\n[주장 대조]")
    for k, v in claim_chk.items():
        print(" ", k, v if not isinstance(v, dict) or "수익" not in v else f"{v['수익']:+.1%}/{v['낙폭']:.1%}")

    four = []
    for bn, k in WIN:
        row = {"구간": f"{'' if bn == '2026' else '보유 '}{k}"}
        for m in ["기본", *MODES]:
            row[m] = st[bn][m][k]
        four.append(row)
    print("\n[4판 비교 USD 수익/낙폭]")
    for r in four:
        print(f"  {r['구간']:<10}", " | ".join(f"{m} {r[m]['수익']:+.1%}/{r[m]['낙폭']:.1%}" for m in ["기본", *MODES]))

    # 판별 집계: 평균 노출·평균 구성
    expo = {}
    for bn in ctx:
        expo[bn] = {m: {"평균 상대노출": float(rel_e(ctx[bn], p).mean()), "평균 투자": float(p.inv.mean()),
                        "평균 반도체": float(p.semis.mean()), "평균 비반도체": float(p.non.mean())}
                    for m, p in paths[bn].items()}

    # 시점 분해: 기본 → 같은 평균 노출 고정 → ii(같은 현금 시점) → i(반도체만)
    decomp = []
    for bn, k in WIN:
        c = ctx[bn]
        lo, hi = c.b.periods[k]
        rowd = {"구간": k if bn == "2026" else f"보유 {k}"}
        for m in ("ii", "i"):
            e = rel_e(c, paths[bn][m])
            mm = (e.index >= lo) & (e.index <= hi)
            cc = float(e[mm].mean())
            rowd[f"{m} 구간평균노출"] = cc
            rowd[f"고정({m}노출)"] = H.stats(cc * c.b.ret, lo, hi)["수익"]
        rowd["기본"] = st[bn]["기본"][k]["수익"]
        rowd["ii"] = st[bn]["ii"][k]["수익"]
        rowd["i"] = st[bn]["i"][k]["수익"]
        rowd["시점값(ii−고정)"] = rowd["ii"] - rowd["고정(ii노출)"]
        rowd["반도체만값(i−ii)"] = rowd["i"] - rowd["ii"]
        decomp.append(rowd)
    print("\n[분해] 기본 / 고정(ii 노출) / ii / i")
    for r in decomp:
        print(f"  {r['구간']:<10} 기본 {r['기본']:+.1%} 고정 {r['고정(ii노출)']:+.1%}(e={r['ii 구간평균노출']:.2f}) "
              f"ii {r['ii']:+.1%} i {r['i']:+.1%} | 시점값 {r['시점값(ii−고정)']:+.1%}p 반도체만값 {r['반도체만값(i−ii)']:+.1%}p")

    # 켜진 주 vs 꺼진 주(묶음 수익 차)
    wt = {"2026": week_table(ctx["2026"], on["2026"], "2026-01-01", "2026-12-31"),
          "2025": week_table(ctx["hold"], on["hold"], "2025-01-01", "2025-12-31"),
          "2024": week_table(ctx["hold"], on["hold"], "2024-01-01", "2024-12-31"),
          "2023": week_table(ctx["hold"], on["hold"], "2023-01-01", "2023-12-31")}
    print("\n[켜진 주 vs 꺼진 주] 주간 평균(%)")
    for y, x in wt.items():
        for lab, s in x["요약"].items():
            print(f"  {y} {lab} {s['주수']}주 기본 {s['기본 평균']:+.2%} 반도체 {s['반도체 평균']:+.2%} 비반도체 {s['비반도체 평균']:+.2%} "
                  f"차 {s['차 평균']:+.2%}(중앙 {s['차 중앙']:+.2%}) 비반도체승 {s['비반도체가 이긴 주 비율']:.0%} SOXX {s['SOXX 평균']:+.2%}")

    # 묶음(spell)별 기여
    sp = {bn: spells(ctx[bn], on[bn], paths[bn]) for bn in ctx}
    tot_log = {bn: float(np.log1p(H.stats(paths[bn]["i"].r, ctx[bn].b.days[0], ctx[bn].b.days[-1])["수익"])
                         - np.log1p(H.stats(paths[bn]["기본"].r, ctx[bn].b.days[0], ctx[bn].b.days[-1])["수익"]))
               for bn in ctx}
    sp_chk = {bn: abs(sum(x["log(i/기본)"] for x in sp[bn]) - tot_log[bn]) for bn in ctx}
    print("\n[묶음 분해 확인] Σ묶음 log차 − 전체 log차:", sp_chk, "| 첫 재조정일", {bn: c.b.days[c.reb[0]] for bn, c in ctx.items()},
          "주 수", {bn: len(c.reb) for bn, c in ctx.items()}, "켜진 주", {bn: int(on[bn].sum()) for bn in ctx})
    for bn in ctx:
        for y in ("2023", "2024", "2025", "2026"):
            xs = [x for x in sp[bn] if x["시작"][:4] == y]
            if xs:
                print(f"  {bn} {y} 묶음 {len(xs)}개 Σlog(i/기본) {sum(x['log(i/기본)'] for x in xs):+.3f} "
                      f"Σlog(i/ii) {sum(x['log(i/ii)'] for x in xs):+.3f}")
    for bn in ctx:
        print(f" {bn} 묶음 {len(sp[bn])}개")
        for x in sp[bn]:
            print(f"  {x['시작']}~{x['끝(ret)']} {x['주수']}주 기본 {x['기본']:+.1%} i {x['i']:+.1%} ii {x['ii']:+.1%} "
                  f"iii {x['iii']:+.1%} 반도체 {x['반도체묶음']:+.1%} 비반도체 {x['비반도체묶음']:+.1%} SOXX {x['SOXX']:+.1%} "
                  f"log(i/기본) {x['log(i/기본)']:+.3f} log(i/ii) {x['log(i/ii)']:+.3f} "
                  f"기본구성 {[round(v, 2) for v in x['기본 반도체·비반도체·현금']]}")
            if bn == "2026":
                print("     비반도체", x["비반도체 종목(시작일 기본)"], "반도체예산", x["반도체예산 종목(시작일 기본)"])

    # ---- 2) 위약 시점
    pl = {"2026": placebo(ctx["2026"], on["2026"], ["전체", "앞", "뒤", "AI쇼크", "7~9월"], ["전체", "앞", "뒤"], rng),
          "hold": placebo(ctx["hold"], on["hold"], ["2025", "2024", "전체"], ["2025", "2024"], rng)}
    print("\n[위약 시점] 실제 순위(원형이동) / 순열 중 실제 이상 비율")
    for bn, x in pl.items():
        for col, v in x["순위"].items():
            if col.startswith("i "):
                print(f"  {bn} {col:<22} 실제 {v['실제']:+.3f} 순위 {v['원형이동 순위']}/{v['원형이동 수']} "
                      f"중앙 {v['원형이동 중앙']:+.3f} 범위 {v['원형이동 최소']:+.3f}~{v['원형이동 최대']:+.3f} "
                      f"순열≥실제 {v.get('순열 중 실제 이상 비율', float('nan')):.3f}")
        for k, v in x["i_vs_ii"].items():
            print(f"  {bn} i−ii {k}: 실제 {v['i−ii 실제']:+.3f} 이동 중앙 {v['i−ii 원형이동 중앙']:+.3f} "
                  f"i>ii 비율 {v['i>ii 비율(원형이동)']:.2f} 순위 {v['i−ii 순위']}")

    # ---- 3) 사건
    ev = []
    for bn, lab, lo, hi in EVENTS:
        ev.append(event_row(ctx[bn], paths[bn], on[bn], lab, lo, hi))
    print("\n[사건] 기본 / i / ii / iii / iiib | 반도체묶음 / 비반도체묶음 / SOXX")
    for x in ev:
        print(f"  {x['사건']:<22} {x['기본']:+.1%} {x['i']:+.1%} {x['ii']:+.1%} {x['iii']:+.1%} {x['iiib']:+.1%} | "
              f"{x['반도체묶음(기본)']:+.1%} {x['비반도체묶음(기본)']:+.1%} {x['SOXX']:+.1%} 신호 "
              f"{''.join('1' if o else '0' for _, o in x['주 신호'])}")
        print("      주:", " ".join(f"{d[5:]}{'켬' if o else '끔'}" for d, o in x["주 신호"]))
        print("      구성(반도체/미분류/비반도체/현금):", " | ".join(
            f"{m} {v['반도체']:.2f}/{v['미분류']:.2f}/{v['비반도체']:.2f}/{v['현금']:.2f}" for m, v in x["구성"].items()))

    # ---- 4) 효율선
    fr = []
    for bn, k in FRONT_WIN:
        c = ctx[bn]
        for fl in FLAVORS:
            row = {"구간": k if bn == "2026" else f"보유 {k}", "단위": fl}
            for m, p in paths[bn].items():
                v = H.stats(c.pflav(p, fl), *c.b.periods[k])
                f = c.frontier(fl, k, v)
                row[m] = {**v, **f}
            fr.append(row)
    print("\n[효율선] 수익/낙폭 | 같은 낙폭 고정 c·수익 | 초과")
    for r in fr:
        print(f"  {r['구간']:<8} {r['단위']:<7}", " | ".join(
            f"{m} {r[m]['수익']:+.1%}/{r[m]['낙폭']:.1%} c{r[m]['c']:.3f} {r[m]['고정수익']:+.1%} 초과 {r[m]['초과']:+.1%}p"
            for m in ("기본", "i", "ii", "iii")))
    fx_missing = {bn: c.fx_missing for bn, c in ctx.items()}

    # ---- 5) 지금 상태
    now = current_state(ctx["2026"], sig_raw)
    print("\n[지금 상태]", json.dumps(rnd(now), ensure_ascii=False))

    if config.RULE != saved_rule:
        raise SystemExit("config.RULE이 바뀌었다")
    payload = {"id": ID, "후보": "S2 SOXX<SMA20 한도0% 현금 주간", "판": MODES, "엔진확인": chk, "재현확인": repro,
               "주장대조": claim_chk, "주장": CLAIM, "주장_기본": CLAIM_BASE, "4판": four, "노출·구성": expo,
               "분해": decomp, "켜진주_꺼진주": {y: x["요약"] for y, x in wt.items()},
               "주별_2026": wt["2026"]["주별"], "묶음": sp, "묶음분해확인": sp_chk, "위약시점": pl, "사건": ev,
               "효율선": fr, "환율빈날": fx_missing, "지금": now,
               "시험범위": {"깎기판": 4, "원형이동": {"2026": len(on["2026"]), "hold": len(on["hold"])},
                        "순열": N_PERM, "순열시드": SEED, "고정노출격자": "0~1, 0.005 간격 201값", "사건": len(EVENTS),
                        "켜진주": {"2026": int(on["2026"].sum()), "hold": int(on["hold"].sum())}},
               "config.RULE": dict(config.RULE)}
    H.RESULTS.mkdir(exist_ok=True)
    p = H.RESULTS / f"{ID}.json"
    p.write_text(json.dumps(rnd(payload, 5), ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("\n저장:", p, "| config.RULE:", config.RULE)
    return payload


if __name__ == "__main__":
    main()
