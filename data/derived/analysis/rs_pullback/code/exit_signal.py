"""rs_pullback 'exit_signal' 갈래 (2026-10-13). 사전 등록은 ../BRIEF.md + 아래 정의. desk/·deskx.py는 고치지 않는다.

질문: 들고 있는 강한 종목의 QQQ 대비 상대강도가 꺾이면(눌림 P가 큼) 더 못 가나(팔 신호) 아니면 되돌아오나(들고 갈 신호)?

Q1 예측(점수만, 매매 없음)
  보유 집합(t일 종가 시점에 들고 있는 종목)
    desk50   데스크 7종목 현행 규칙 = deskx.Engine Params(semis_cap=0.5)의 매일 보유
    desk774  #774 기본 Params()의 매일 보유(참고)
    s120top  그날 universe_mask 안 S_120 = ex(120) 백분위 > 0.90 (상위 10%)
    all      참고: eligible 전체(보유와 무관한 일반 효과인지 보기)
  '꺾임' = 그날 universe_mask 안 P 백분위 > 0.80(상위 20%). P는 BRIEF의 12개.
  보충(선택에 쓰지 않음): Q2 신호 중 백분위가 아닌 3개(P2(5) 상위10%, ex(4) <= -10, Fade(5) >= 15).
  성과: t+1 시가 → t+1+h 시가(h=5·10), 그날 eligible 평균 대비 초과(%p). 날짜마다 (꺾인 보유 평균 − 안 꺾인 보유 평균),
        날짜 평균과 Newey-West t(lag=h). 음수 = 팔 신호, 양수 = 들고 갈 신호.
  구간: front / back / y2025(s120top·all만, 데스크 보유는 2026만 있다) / late(9/1~, 사후 진단).

Q2 덧씌우기(사전 등록 12개 = 신호 6 × 행동 2), 기준 = Params(semis_cap=0.5)
  신호(k-1일 종가로 판정): P1(4)>p80, P2(10)>p80, P3(10)>p80, P2(5)>p90, ex(4)<=-10, Fade(5)=P2(5)>=15
  A: 재조정일이 아닌 날, 들고 있는 종목이 꺾이면 k일 시가에 팔고 다음 주 재조정까지 현금.
     재조정 때는 데스크 규칙 그대로(판 종목도 데스크 모델에서는 아직 '보유' → 완충 5칸 적용, 순위 안이면 다시 산다).
  B: 주간 재조정 때 꺾인 종목을 순위에서 빼고 다음 순위로 채운다(새로 살 종목·이미 든 종목 모두).
  고르기: front(1/2~5/29) 수익 최고 → back·9/1~10/9·낙폭 확인. 민감도(선택에 안 씀): 편도 0.1% 호가 비용,
          A에서 판 종목을 완충에서도 빼기, B를 새 종목에만 적용, #774 기준 위에서 같은 12개.

실행: PYTHONIOENCODING=utf-8 py exit_signal.py   (결과 ../results/exit_signal.json)
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ANA = os.path.abspath(os.path.join(HERE, "..", ".."))
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
_spec = importlib.util.spec_from_file_location("tg_harness", os.path.join(ANA, "trend_grade", "code", "harness.py"))
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)
sys.path.insert(0, os.path.join(ANA, "rsi_breakout", "code"))
from deskx import Engine, Params, START, PERIODS, MONTHS  # noqa: E402
from desk.strategies import iso_week, run_model  # noqa: E402
from desk.validate import codex_caps  # noqa: E402

P_FRONT, P_BACK, P_SEP, P_ALL = (p[0] for p in PERIODS)
COST = 0.001  # 편도 호가 비용 민감도
DESK = Params(semis_cap=0.5)


# ------------------------------------------------------------------ 측정값
def ex(P, n):
    C, Q = P["C"], P["bench"]["C"]
    return 100 * (C / C.shift(n)).sub(Q / Q.shift(n), axis=0)


def p_measures() -> dict:
    """이름 → fn(P) -> DataFrame(클수록 더 눌림). t 종가까지만 쓴다."""
    m = {}
    for n in (3, 4, 5):
        m[f"P1({n})"] = (lambda n: lambda P: -ex(P, n))(n)
    for K in (5, 10, 20):
        m[f"P2({K})"] = (lambda K: lambda P: ex(P, 4).rolling(K, min_periods=K).max() - ex(P, 4))(K)
    for K in (5, 10, 20):
        def p3(P, K=K):
            R = P["C"].div(P["bench"]["C"], axis=0)
            return 1 - R / R.rolling(K, min_periods=K).max()
        m[f"P3({K})"] = p3
    for K in (5, 10, 20):
        m[f"P4({K})"] = (lambda K: lambda P: 1 - P["C"] / P["C"].rolling(K, min_periods=K).max())(K)
    return m


def pct_flag(fn, lo):
    return lambda P: (H.pct_rank(fn(P), H.universe_mask(P)) > lo).astype(float)


def signal_fns() -> dict:
    """Q2 신호(사전 등록 6개). 값 1 = 꺾임."""
    pm = p_measures()
    return {
        "P1(4)>p80": pct_flag(pm["P1(4)"], 0.8),
        "P2(10)>p80": pct_flag(pm["P2(10)"], 0.8),
        "P3(10)>p80": pct_flag(pm["P3(10)"], 0.8),
        "P2(5)>p90": pct_flag(pm["P2(5)"], 0.9),
        "ex4<=-10": lambda P: (ex(P, 4) <= -10).astype(float),
        "Fade5>=15": lambda P: (pm["P2(5)"](P) >= 15).astype(float),
    }


# ------------------------------------------------------------------ 데스크 덧씌우기 엔진
class Overlay(Engine):
    """deskx.Engine을 고치지 않고 덧씌운다. 신호 판정 = k-1일 종가(패널 날짜) 기준."""

    def __init__(self, flags: dict[str, pd.DataFrame]):
        super().__init__()
        self.sets = {}
        for name, F in flags.items():
            F = F.fillna(0).astype(bool)
            self.sets[name] = {d.strftime("%Y-%m-%d"): set(F.columns[row]) for d, row in zip(F.index, F.to_numpy())}
        self._skip = None          # (신호, 범위, 현재 보유) — 재조정 때만 켠다
        self.missing = set()

    def faded(self, sig: str, key: str, k: int) -> bool:
        s = self.sets[sig].get(self.days[k - 1])
        if s is None:
            self.missing.add(("date", self.days[k - 1]))
            return False
        return self.ticker[key] in s

    def ranking(self, p, k, session):
        rows = super().ranking(p, k, session)
        if self._skip is not None:
            sig, scope, current = self._skip
            rows = [r for r in rows if not (self.faded(sig, r["key"], k) and (scope == "all" or r["key"] not in current))]
        return rows

    def run_overlay(self, p: Params, sig=None, action=None, a_keep_selected=True, b_scope="all"):
        days = [d for d in self.days if d >= START]
        k0 = self.days.index(days[0])
        value, rows, turnover = [1.0], [], []
        w: dict[str, float] = {}
        selected: set[str] = set()
        started = False
        sells, skips = [], []
        for k in range(k0, len(self.days)):
            day = self.days[k]
            if k > k0:
                rets = {}
                for key in w:
                    o = self.col("opn", key)
                    rets[key] = (o[k] / o[k - 1] - 1) if o is not None and np.isfinite(o[k]) and np.isfinite(o[k - 1]) \
                        and o[k - 1] > 0 else 0.0
                gross = sum(wt * rets[key] for key, wt in w.items())
                total = sum(wt * (1 + rets[key]) for key, wt in w.items()) + (1 - sum(w.values()))
                w = {key: wt * (1 + rets[key]) / total for key, wt in w.items()}
                value.append(value[-1] * (1 + gross))
            reb = (not started) or iso_week(day) != iso_week(self.days[k - 1])
            to = 0.0
            if action == "A" and not reb:
                for key in sorted(w):
                    if self.faded(sig, key, k):
                        sells.append({"day": day, "k": k, "key": key, "w": float(w[key])})
                        to += w[key]
                        w.pop(key)
                        if not a_keep_selected:
                            selected.discard(key)
            if reb:
                if action == "B":
                    self._skip = (sig, b_scope, set(selected))
                keep, tgt, ranked = self.target(p, k, day, selected)
                self._skip = None
                if action == "B" and ranked:
                    keep0, _, _ = self.target(p, k, day, selected)   # 같은 상태에서 거르지 않았을 때
                    out_ = [x for x in keep0 if x not in keep]
                    in_ = [x for x in keep if x not in keep0]
                    if out_ or in_:
                        skips.append({"day": day, "k": k, "skipped": out_, "replaced_by": in_,
                                      "skipped_held": [x for x in out_ if x in selected]})
                if ranked:
                    started = True
                    to += sum(abs(tgt.get(x, 0.0) - w.get(x, 0.0)) for x in set(tgt) | set(w))
                    w, selected = dict(tgt), set(keep)
            turnover.append(to)
            rows.append(dict(day=day, w=dict(w)))
        v = pd.Series(value, index=days)
        to = pd.Series(turnover, index=days)
        net = v * (1 - COST * to).cumprod().shift(1).fillna(1.0)
        return v, net, rows, sells, skips, to

    # -------------------------------------------------------------- 사건 분석
    def next_reb(self, rows, i):
        days = [r["day"] for r in rows]
        for j in range(i + 1, len(days)):
            if iso_week(days[j]) != iso_week(days[j - 1]):
                return j
        return None

    def open_ret(self, key, k_from, k_to):
        o = self.col("opn", key)
        if o is None or k_to is None or not (np.isfinite(o[k_from]) and np.isfinite(o[k_to]) and o[k_from] > 0):
            return None
        return float(o[k_to] / o[k_from] - 1)

    def describe_sells(self, rows, sells):
        idx = {r["day"]: i for i, r in enumerate(rows)}
        out = []
        for s in sells:
            i = idx[s["day"]]
            j = self.next_reb(rows, i)
            kn = self.days.index(rows[j]["day"]) if j is not None else len(self.days) - 1
            r = self.open_ret(s["key"], s["k"], kn)
            out.append({"day": s["day"], "ticker": self.ticker[s["key"]], "w": round(s["w"], 4),
                        "until": self.days[kn], "ret_until_reb": None if r is None else round(r, 4),
                        "contrib": None if r is None else round(s["w"] * r, 5),
                        "rebought": bool(j is not None and s["key"] in rows[j]["w"])})
        return out

    def describe_skips(self, rows, skips):
        idx = {r["day"]: i for i, r in enumerate(rows)}
        out = []
        for s in skips:
            i = idx[s["day"]]
            j = self.next_reb(rows, i)
            kn = self.days.index(rows[j]["day"]) if j is not None else len(self.days) - 1
            rs = [self.open_ret(x, s["k"], kn) for x in s["skipped"]]
            rr = [self.open_ret(x, s["k"], kn) for x in s["replaced_by"]]
            rs = [x for x in rs if x is not None]
            rr = [x for x in rr if x is not None]
            out.append({"day": s["day"], "until": self.days[kn],
                        "skipped": [self.ticker[x] for x in s["skipped"]],
                        "skipped_held": [self.ticker[x] for x in s["skipped_held"]],
                        "replaced_by": [self.ticker[x] for x in s["replaced_by"]],
                        "skipped_ret": round(float(np.mean(rs)), 4) if rs else None,
                        "repl_ret": round(float(np.mean(rr)), 4) if rr else None})
        return out


def held_frame(E, rows) -> pd.DataFrame:
    P = H.load_panel()
    out = pd.DataFrame(False, index=P["C"].index, columns=P["C"].columns)
    for r in rows:
        d = pd.Timestamp(r["day"])
        if d in out.index:
            cols = [E.ticker[k] for k in r["w"] if E.ticker[k] in out.columns]
            out.loc[d, cols] = True
    return out


# ------------------------------------------------------------------ Q1
def q1_cell(held, faded, h, wname, drop=()):
    dates = H.window_dates(wname, h)
    if len(dates) < 10:
        return None
    el = H.eligible().loc[dates]
    fr = H.forward(h).loc[dates].where(el)
    exc = fr.sub(fr.mean(axis=1), axis=0) * 100
    hd = held.reindex(index=dates, columns=fr.columns).fillna(False).astype(bool) & el
    if drop:
        hd.loc[:, [c for c in drop if c in hd.columns]] = False
    fd = faded.reindex(index=dates, columns=fr.columns).fillna(0).astype(bool)
    f_ex = exc.where(hd & fd).mean(axis=1)
    n_ex = exc.where(hd & ~fd).mean(axis=1)
    diff = (f_ex - n_ex).dropna()
    nf = (hd & fd).sum(axis=1)
    nh = hd.sum(axis=1)
    res = {"diff": round(float(diff.mean()), 3) if len(diff) else None,
           "diff_t": round(H.newey_west_t(diff, h), 2) if len(diff) >= 5 else None,
           "diff_pos": round(float((diff > 0).mean()) * 100, 1) if len(diff) else None,
           "faded_ex": round(float(f_ex.dropna().mean()), 3) if f_ex.notna().any() else None,
           "faded_t": round(H.newey_west_t(f_ex.dropna(), h), 2) if f_ex.notna().sum() >= 5 else None,
           "nonfaded_ex": round(float(n_ex.dropna().mean()), 3) if n_ex.notna().any() else None,
           "days_both": int(len(diff)), "days": int(len(dates)),
           "faded_share": round(float(nf.sum() / max(nh.sum(), 1)) * 100, 1),
           "faded_per_day": round(float(nf.mean()), 2), "held_per_day": round(float(nh.mean()), 2)}
    if len(diff) >= 5:
        top = diff.abs().idxmax()
        res["drop_top_day"] = round(float(diff.drop(top).mean()), 3)
        res["top_day"] = top.strftime("%Y-%m-%d")
        res["top5_share"] = round(float(diff.loc[diff.abs().nlargest(5).index].sum() / diff.sum()) * 100, 1) \
            if diff.sum() != 0 else None
        if wname in ("back",):
            nj = diff[~((diff.index >= "2026-06-22") & (diff.index <= "2026-07-31"))]
            res["ex_jun22_jul"] = round(float(nj.mean()), 3)
    # 꺾인 쪽 종목별 기여(날짜 평균에 대한 몫)
    w = (hd & fd).astype(float)
    w = w.div(w.sum(axis=1).replace(0, np.nan), axis=0)
    contrib = (exc * w).sum(axis=0) / max(int(f_ex.notna().sum()), 1)
    res["faded_top_contrib"] = {k: round(float(v), 3) for k, v in contrib.reindex(contrib.abs().sort_values(ascending=False).index[:3]).items()}
    return res


def run_q1(held_sets, faded_flags, supp_flags):
    out = {}
    for sname, (held, windows) in held_sets.items():
        out[sname] = {}
        for mname, F in list(faded_flags.items()) + list(supp_flags.items()):
            out[sname][mname] = {}
            for h in (5, 10):
                cell = {}
                for wn in windows:
                    c = q1_cell(held, F, h, wn)
                    if c is not None:
                        cell[wn] = c
                if sname != "all":
                    for wn in ("front", "back"):
                        c = q1_cell(held, F, h, wn, drop=("SNDK",))
                        if c is not None:
                            cell[wn + "_exSNDK"] = {k: c[k] for k in ("diff", "diff_t", "faded_ex")}
                out[sname][mname][f"h{h}"] = cell
    return out


# ------------------------------------------------------------------ Q2 요약
def summarize(E, name, v, net, rows, sells, skips, base_v):
    ev = E.evaluate(name, v, rows)
    evn = E.evaluate(name, net)
    res = {"name": name}
    for lab in (P_ALL, P_FRONT, P_BACK, P_SEP):
        res[lab] = {"ret": round(ev[lab]["수익"] * 100, 2), "mdd": round(ev[lab]["낙폭"] * 100, 2),
                    "ret_net0.1%": round(evn[lab]["수익"] * 100, 2)}
    res["monthly"] = {m[5:]: round(ev["월별"][m] * 100, 2) for m in MONTHS}
    res["semis_avg"] = round(ev["반도체평균"], 3)
    # 기준 대비 하루 수익 차이 쏠림
    r = (v.shift(-1) / v - 1).dropna()
    rb = (base_v.shift(-1) / base_v - 1).dropna()
    d = (r - rb).loc[(r.index >= START) & (r.index <= "2026-10-09")]
    if d.abs().sum() > 0:
        top = d.abs().nlargest(5)
        res["diff_vs_base_top_days"] = {k: round(float(d[k]) * 100, 2) for k in top.index}
        res["diff_vs_base_sum_pp"] = round(float(d.sum()) * 100, 2)
    if sells:
        ds = E.describe_sells(rows, sells)
        ok = [s for s in ds if s["ret_until_reb"] is not None]
        res["sells"] = {"count": len(ds), "rebought_next_reb": sum(s["rebought"] for s in ds),
                        "by_month": pd.Series([s["day"][:7] for s in ds]).value_counts().sort_index().to_dict(),
                        "by_ticker": pd.Series([s["ticker"] for s in ds]).value_counts().head(8).to_dict(),
                        "mean_ret_until_reb": round(float(np.mean([s["ret_until_reb"] for s in ok])) * 100, 2) if ok else None,
                        "hit_fell": round(float(np.mean([s["ret_until_reb"] < 0 for s in ok])) * 100, 1) if ok else None,
                        "avoided_loss_pp": round(-sum(s["contrib"] for s in ok if s["contrib"] < 0) * 100, 2),
                        "missed_gain_pp": round(sum(s["contrib"] for s in ok if s["contrib"] > 0) * 100, 2),
                        "net_pp(+=도움)": round(-sum(s["contrib"] for s in ok) * 100, 2),
                        "net_pp_by_half": {lab: round(-sum(s["contrib"] for s in ok if a <= s["day"] <= b) * 100, 2)
                                           for lab, a, b in PERIODS},
                        "net_pp_ex_SNDK": round(-sum(s["contrib"] for s in ok if s["ticker"] != "SNDK") * 100, 2),
                        "largest": sorted(ok, key=lambda s: -abs(s["contrib"]))[:5],
                        "events": ds}
    if skips:
        dk = E.describe_skips(rows, skips)
        ok = [s for s in dk if s["skipped_ret"] is not None and s["repl_ret"] is not None]
        res["skips"] = {"rebalances_changed": len(dk), "names_skipped": sum(len(s["skipped"]) for s in dk),
                        "skipped_were_held": sum(len(s["skipped_held"]) for s in dk),
                        "mean_skipped_ret": round(float(np.mean([s["skipped_ret"] for s in ok])) * 100, 2) if ok else None,
                        "mean_repl_ret": round(float(np.mean([s["repl_ret"] for s in ok])) * 100, 2) if ok else None,
                        "repl_beat_share": round(float(np.mean([s["repl_ret"] > s["skipped_ret"] for s in ok])) * 100, 1) if ok else None,
                        "by_ticker": pd.Series([t for s in dk for t in s["skipped"]]).value_counts().head(8).to_dict(),
                        "events": dk}
    return res


def run_q2(E, base_params, label):
    bv, bnet, brows, _, _, _ = E.run_overlay(base_params)
    base = summarize(E, f"기준 {label}", bv, bnet, brows, [], [], bv)
    var = []
    for sig in E.sets:
        for act in ("A", "B"):
            v, net, rows, sells, skips, _ = E.run_overlay(base_params, sig, act)
            var.append(summarize(E, f"{sig}|{act}", v, net, rows, sells, skips, bv))
    best = max(var, key=lambda e: e[P_FRONT]["ret"])
    both = [e["name"] for e in var if e[P_FRONT]["ret"] > base[P_FRONT]["ret"] and e[P_BACK]["ret"] > base[P_BACK]["ret"]]
    mdd_better = [e["name"] for e in var if e[P_ALL]["mdd"] > base[P_ALL]["mdd"]]
    return {"base": base, "variants": var, "best_by_front": best["name"], "beat_base_both_halves": both,
            "shallower_full_mdd": mdd_better,
            "beat_base_count": {lab: sum(e[lab]["ret"] > base[lab]["ret"] for e in var) for lab in (P_ALL, P_FRONT, P_BACK, P_SEP)}}, bv


def main():
    P = H.load_panel()
    pm = p_measures()
    sigs = signal_fns()
    # 미래 정보 시험
    nla = {k: H.no_lookahead(fn) for k, fn in pm.items()}
    nla.update({f"flag {k}": H.no_lookahead(fn) for k, fn in sigs.items()})
    nla["S120 top10"] = H.no_lookahead(pct_flag(lambda P: ex(P, 120), 0.9))
    print("no_lookahead", nla)

    flags = {k: fn(P) for k, fn in sigs.items()}
    E = Overlay(flags)
    # 동등성: 덧씌우기 없음 = deskx 기본 = 데스크 run_model
    par = {}
    v0, _, rows50, _, _, _ = E.run_overlay(DESK)
    vx, _ = E.run(DESK)
    ref = run_model(E.book, start=START)
    par["desk50_overlay_vs_deskx"] = float(np.max(np.abs(v0.to_numpy() - vx.to_numpy())))
    par["desk50_vs_run_model_live"] = float(np.max(np.abs(vx.to_numpy() - np.asarray(ref.value))))
    v774, _, rows774, _, _, _ = E.run_overlay(Params())
    with codex_caps():
        ref774 = run_model(E.book, start=START)
    par["774_overlay_vs_run_model_codex_caps"] = float(np.max(np.abs(v774.to_numpy() - np.asarray(ref774.value))))
    print("parity", par)

    # Q1
    umask = H.universe_mask(P)
    s120 = H.pct_rank(ex(P, 120), umask) > 0.9
    held_sets = {"desk50": (held_frame(E, rows50), ("front", "back", "late")),
                 "desk774": (held_frame(E, rows774), ("front", "back", "late")),
                 "s120top": (s120, ("front", "back", "y2025", "late")),
                 "all": (H.eligible(), ("front", "back", "y2025", "late"))}
    faded = {k: (H.pct_rank(fn(P), umask) > 0.8) for k, fn in pm.items()}
    supp = {"P2(5)>p90": flags["P2(5)>p90"], "ex4<=-10": flags["ex4<=-10"], "Fade5>=15": flags["Fade5>=15"]}
    q1 = run_q1(held_sets, faded, supp)

    # Q1 요약: 집합 × h마다 12개 중 음수 개수, front 최선(가장 음수 t·가장 양수 t)과 그 back
    q1sum = {}
    for sname in ("desk50", "desk774", "s120top", "all"):
        for h in ("h5", "h10"):
            cells = {m: q1[sname][m][h] for m in pm}
            neg = {wn: sum(1 for c in cells.values() if wn in c and c[wn]["diff"] is not None and c[wn]["diff"] < 0)
                   for wn in ("front", "back", "y2025", "late")}
            ft = {m: c["front"]["diff_t"] for m, c in cells.items() if c.get("front", {}).get("diff_t") is not None}
            sell = min(ft, key=ft.get)
            hold = max(ft, key=ft.get)
            g = lambda m, wn, k: cells[m].get(wn, {}).get(k)
            q1sum[f"{sname}|{h}"] = {
                "neg_count_of_12": neg,
                "mean_diff_12": {wn: round(float(np.mean([c[wn]["diff"] for c in cells.values() if wn in c and c[wn]["diff"] is not None])), 3)
                                 for wn in ("front", "back", "y2025", "late") if any(wn in c for c in cells.values())},
                "front_best_sell": {"measure": sell, **{wn: (g(sell, wn, "diff"), g(sell, wn, "diff_t")) for wn in ("front", "back", "y2025", "late")}},
                "front_best_hold": {"measure": hold, **{wn: (g(hold, wn, "diff"), g(hold, wn, "diff_t")) for wn in ("front", "back", "y2025", "late")}}}
            print(sname, h, q1sum[f"{sname}|{h}"])

    # Q2
    q2, bv = run_q2(E, DESK, "desk50")
    q2_774, _ = run_q2(E, Params(), "#774")
    # 민감도(선택에 쓰지 않음): front 최선 A·B 각각
    sens = {}
    best_a = max((e for e in q2["variants"] if e["name"].endswith("|A")), key=lambda e: e[P_FRONT]["ret"])["name"]
    best_b = max((e for e in q2["variants"] if e["name"].endswith("|B")), key=lambda e: e[P_FRONT]["ret"])["name"]
    v, net, rows, sells, skips, _ = E.run_overlay(DESK, best_a.split("|")[0], "A", a_keep_selected=False)
    sens[best_a + " (판 종목 완충 제외)"] = {k: x for k, x in summarize(E, best_a, v, net, rows, sells, skips, bv).items()
                                         if k not in ("sells",)}
    v, net, rows, sells, skips, _ = E.run_overlay(DESK, best_b.split("|")[0], "B", b_scope="new")
    sens[best_b + " (새 종목에만)"] = {k: x for k, x in summarize(E, best_b, v, net, rows, sells, skips, bv).items()
                                    if k not in ("skips",)}
    print("missing signal dates", sorted(E.missing)[:5])

    hdr = f"{'변형':<18}{'전체':>8}{'MDD':>7}{'앞':>8}{'뒤':>8}{'뒤MDD':>7}{'9/1~':>7}{'net전체':>8}{'매도/스킵':>9}"
    print(hdr)
    for e in [q2["base"]] + q2["variants"]:
        n = e.get("sells", {}).get("count") or e.get("skips", {}).get("names_skipped") or 0
        print(f"{e['name']:<18}{e[P_ALL]['ret']:>8.1f}{e[P_ALL]['mdd']:>7.1f}{e[P_FRONT]['ret']:>8.1f}{e[P_BACK]['ret']:>8.1f}"
              f"{e[P_BACK]['mdd']:>7.1f}{e[P_SEP]['ret']:>7.1f}{e[P_ALL]['ret_net0.1%']:>8.1f}{n:>9}")
    print("best_by_front", q2["best_by_front"], "both", q2["beat_base_both_halves"], "count", q2["beat_base_count"])
    print("#774 best", q2_774["best_by_front"], "both", q2_774["beat_base_both_halves"], "count", q2_774["beat_base_count"])

    os.makedirs(RES, exist_ok=True)
    out = {"family": "exit_signal", "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
           "spec": {"P_measures": list(pm), "faded": "그날 universe_mask 안 P 백분위 > 0.80",
                    "held_sets": {"desk50": "deskx Params(semis_cap=0.5) 매일 보유(현행 데스크)", "desk774": "deskx Params() #774",
                                  "s120top": "universe_mask 안 ex(120) 백분위 > 0.90", "all": "eligible 전체(참고)"},
                    "q1_metric": "날짜마다 꺾인 보유 평균 초과 − 안 꺾인 보유 평균 초과(%p, eligible 평균 대비), NW t lag=h",
                    "q2_signals": list(sigs), "q2_actions": {"A": "주중 꺾이면 다음 시가 매도, 다음 재조정까지 현금(재조정은 데스크 규칙대로)",
                                                             "B": "재조정 때 꺾인 종목을 순위에서 빼고 다음 순위로"},
                    "search_size": "Q1: 12 측정 × 2 h × 보유 3집합(+전체 참고) × 구간 3~4 (+보충 3신호). Q2: 사전 등록 12개(+#774 위 12개, 민감도 2개는 선택에 안 씀)",
                    "cost_sensitivity": "편도 0.1% × 회전율"},
           "parity": par, "no_lookahead": nla,
           "q1_summary": q1sum, "q1": q1,
           "q2_desk50": q2, "q2_774": q2_774, "q2_sensitivity": sens}
    path = os.path.join(RES, "exit_signal.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
    print("저장", path)


if __name__ == "__main__":
    main()
