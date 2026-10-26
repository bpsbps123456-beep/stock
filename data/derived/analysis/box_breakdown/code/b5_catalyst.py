"""갈래 5 사건·재료(BRIEF 갈래 5). 하네스의 사건표와 evaluate_cells를 그대로 쓴다(판정 로직을 새로 만들지 않는다).

실행: py b5_catalyst.py features   # 특징만 계산하고 점검(결과값 안 봄)
      py b5_catalyst.py run        # 칸 판정 → ../results/b5_catalyst.md, b5_catalyst_cells.csv

특징 정의(결과를 보기 전에 정함, 2026-10-14). 모두 사건표의 이탈일 t(인덱스 i), 사건의 L = L_N(t) 기준.
본 칸(고르기 상위 10칸 후보 풀) — t 종가까지 알 수 있는 값만:
- earn_pm3   하네스 열 그대로. 실적 반응일 r(16시 이후 발표면 다음 거래일)이 t−3..t+3 거래일 안이면 1. ETF는 0.
             t+1..t+3 예정분은 공시된 일정으로 t 종가에 알 수 있다고 보지만, 캐시는 실제 발표일이라 일정 변경은 확인 못 함.
- earn_post3 실적 '뒤' 이탈만: r이 t−3..t 안(발표·반응이 이탈일 전에 이미 끝남)이면 1. ETF는 0.
- earn_day0  실적 반응일 당일 이탈: r = t이면 1. ETF는 0.
- gap_size   갭 이탈 크기 = 시가(t) / L − 1. 음수면 시가가 이미 박스 하단 아래(갭 이탈). 종목·ETF 모두.
- gap_pct    프리장 갭(일봉 시가 근사) = 시가(t) / 종가(t−1) − 1. 하네스 보조 열 그대로. 종목·ETF 모두.
- sub_cobreak_n      같은 세부 섹터(watchlist.AI_UNIVERSE 소분류) 다른 종목 중 같은 N에서 종가(t) < L_N(t)인
                     종목 수(BRIEF 이탈일 정의 = 이탈 상태, 새 사건 여부 무관). 소분류에 L_N이 있는 다른 종목이 없거나
                     ETF면 NaN(모집단 밖).
- sub_cobreak_share  sub_cobreak_n ÷ 같은 소분류에서 그날 L_N과 종가가 있는 다른 종목 수. NaN 규칙 동일.
진입 변형(따로 판정·표시, 상위 10칸 풀에 넣지 않음):
- gap_next   t+1 프리장 갭(일봉 시가 근사) = 시가(t+1) / 종가(t) − 1. t+1 정규장 직전 프리장 가격으로 알 수 있지만
             여기서는 진입가인 t+1 시가 자체로 근사하므로 '프리장 확인 뒤 진입' 변형으로만 표시한다.
칸 = 하네스 tercile_cells(by=N, 문턱은 앞 구간 사건 분포). 값이 두 개 이하인 특징은 =1/=0 두 칸.
판정 = 하네스 evaluate_cells 기본값(상위 10칸, 켜짐 ≥ 15건·8일, 순열 200회, 귀무 200회, 층 N·kind).
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
MAIN = ["earn_pm3", "earn_post3", "earn_day0", "gap_size", "gap_pct", "sub_cobreak_n", "sub_cobreak_share"]
VARIANT = ["gap_next"]

DESC = {
    "earn_pm3": ("실적 반응일(16시 이후 발표면 다음 거래일)이 이탈일 t−3..t+3 거래일 안", "t 종가(예정 실적은 공시 일정, 일정 변경 미확인)"),
    "earn_post3": ("실적 반응일이 이탈일 t−3..t 안(실적 뒤 이탈)", "t 종가(발표·반응이 이미 끝남)"),
    "earn_day0": ("실적 반응일 당일 이탈(r = t)", "t 종가"),
    "gap_size": ("갭 이탈 크기 = 시가(t)/L_N − 1", "t 정규장 시가(이탈 확정은 t 종가)"),
    "gap_pct": ("프리장 갭(일봉 근사) = 시가(t)/종가(t−1) − 1", "t 정규장 시가(프리장 갭의 일봉 근사)"),
    "sub_cobreak_n": ("같은 watchlist 소분류 다른 종목 중 종가(t) < L_N(t)인 종목 수", "t 종가"),
    "sub_cobreak_share": ("같은 watchlist 소분류 다른 종목 중 종가(t) < L_N(t) 비율", "t 종가"),
    "gap_next": ("t+1 프리장 갭(일봉 근사) = 시가(t+1)/종가(t) − 1 [진입 변형]", "t+1 정규장 직전(프리장, 시가로 근사 — 진입 변형)"),
}


def subgroup_peers():
    peers = {}
    for g in H.watchlist.AI_UNIVERSE.values():
        for sub, ss in g.items():
            ss = [s for s in ss if "." not in s]
            for s in ss:
                peers.setdefault(s, (sub, []))[1].extend(x for x in ss if x != s)
    return peers


def add_features(ev_all, P):
    er = H.earnings_react_dates(P.cal)
    peers = subgroup_peers()
    T = len(P.cal)
    ev_all = ev_all.copy()
    cols = {c: np.full(len(ev_all), np.nan) for c in ["earn_post3", "earn_day0", "earn_pm3_chk", "gap_size",
                                                       "gap_next", "sub_cobreak_n", "sub_cobreak_share", "sub_peers"]}
    subname = np.full(len(ev_all), None, dtype=object)
    for N in sorted(ev_all["N"].unique()):
        state = {}
        for s in H.STOCKS:
            if s not in P.j:
                continue
            o, h, l, c = (P.col(f, s) for f in "OHLC")
            _, brk, Ls, _, _ = H.detect_events(o, h, l, c, N)
            state[s] = (brk, ~np.isnan(Ls) & ~np.isnan(c))
        idx = np.flatnonzero(ev_all["N"].to_numpy() == N)
        for k in idx:
            r = ev_all.iloc[k]
            s, i, kind, Lv = r["sym"], int(r["i"]), r["kind"], float(r["L"])
            o, c = P.col("O", s), P.col("C", s)
            cols["gap_size"][k] = o[i] / Lv - 1 if not np.isnan(o[i]) else np.nan
            cols["gap_next"][k] = o[i + 1] / c[i] - 1 if i + 1 < T and not np.isnan(o[i + 1]) else np.nan
            if kind == "stock":
                ed = er.get(s, np.array([], int)) - i
                cols["earn_post3"][k] = float(np.any((ed >= -3) & (ed <= 0)))
                cols["earn_day0"][k] = float(np.any(ed == 0))
                cols["earn_pm3_chk"][k] = float(np.any(np.abs(ed) <= 3))
                sub, pe = peers.get(s, (None, []))
                subname[k] = sub
                pe = [p for p in pe if p in state]
                nv = sum(bool(state[p][1][i]) for p in pe)
                cols["sub_peers"][k] = nv
                if nv > 0:
                    nb = sum(bool(state[p][0][i]) for p in pe if state[p][1][i])
                    cols["sub_cobreak_n"][k] = nb
                    cols["sub_cobreak_share"][k] = nb / nv
            else:
                cols["earn_post3"][k] = 0.0
                cols["earn_day0"][k] = 0.0
                cols["earn_pm3_chk"][k] = 0.0
    for c, v in cols.items():
        ev_all[c] = v
    ev_all["subgroup"] = subname
    return ev_all


def load_with_features():
    P = H.load_panel()
    ev = H.load_events_all()
    return add_features(ev, P), P


def cmd_features():
    ev, P = load_with_features()
    bad = (ev["earn_pm3"] != ev["earn_pm3_chk"]).sum()
    print("earn_pm3 재계산 불일치:", int(bad))
    for N in H.NS:
        g = ev[ev["N"] == N]
        for pn, p in (("front", H.FRONT), ("back", H.BACK), ("aux", H.AUX)):
            gg = g[H.in_period(g, p) & g["short_ret_h10"].notna()]
            st = gg[gg["kind"] == "stock"]
            print(N, pn, "events", len(gg), "stock", len(st),
                  "pm3", int(st["earn_pm3"].sum()), "post3", int(st["earn_post3"].sum()),
                  "day0", int(st["earn_day0"].sum()),
                  "sub NaN", int(st["sub_cobreak_n"].isna().sum()),
                  "sub_n uniq(front)", sorted(st["sub_cobreak_n"].dropna().unique().astype(int).tolist())[:12])
    print(ev[["gap_size", "gap_pct", "gap_next", "sub_cobreak_n", "sub_cobreak_share"]].describe().T)
    # 시점 점검: gap_size<0 이면 gap_break=1이어야 한다
    gb = ev.dropna(subset=["gap_size", "gap_break"])
    print("gap_size<0 vs gap_break 불일치:", int(((gb["gap_size"] < 0).astype(float) != gb["gap_break"]).sum()))


# ───────────────────────────── 판정 ─────────────────────────────
def _feat_of(cell):
    body = cell.split("|", 1)[1]
    for f in sorted(DESC, key=len, reverse=True):
        if body.startswith(f):
            return f
    return body


def cell_definition(cell):
    n, body = cell.split("|", 1)
    f = _feat_of(cell)
    rest = body[len(f):]
    d, _ = DESC[f]
    if rest.startswith("="):
        val = rest[1:].split("(")[0]
        cond = f"{f} = {val}"
    else:
        op = "≥" if rest.startswith("≥") else "≤"
        thr = rest[1:].split("(")[0]
        side = "상위" if op == "≥" else "하위"
        cond = f"{f} {op} {thr}(앞 구간 {n} 사건 {side} 3분위 문턱)"
    return f"{n}, {d}; 켜짐: {cond}. 진입 t+1 시가 공매도, 되돌림 손절 또는 t+10 종가, 비용 0.10%"


def cls_shares(ev, cells, period):
    """칸별 켜진 사건의 진짜/가짜/되돌림 청산 비율(설명용)."""
    out = {}
    m = H.in_period(ev, period) & ev["short_ret_h10"].notna()
    for nm, s in cells:
        on = m & (s == 1)
        c = ev.loc[on, "cls"].dropna()
        out[nm] = ((c == "진짜").mean() if len(c) else np.nan, (c == "가짜").mean() if len(c) else np.nan,
                   (ev.loc[on, "exit_reason"] == "reclaim").mean() if on.any() else np.nan)
    return out


def earnings_table(ev):
    """실적 이탈은 가짜가 많은가(설명용): 종목 사건을 실적 뒤 이탈 / 실적 앞 이탈(예정) / 실적 무관으로 나눔."""
    rows = []
    for N in H.NS:
        for pn, p in (("앞(1~5월)", H.FRONT), ("뒤(6~9월)", H.BACK), ("2025", H.AUX)):
            g = ev[(ev["N"] == N) & (ev["kind"] == "stock") & H.in_period(ev, p) & ev["short_ret_h10"].notna()]
            grp = np.where(g["earn_post3"] == 1, "실적 뒤(t−3..t)",
                           np.where(g["earn_pm3"] == 1, "실적 앞(t+1..t+3)", "실적 무관"))
            for gn in ("실적 뒤(t−3..t)", "실적 앞(t+1..t+3)", "실적 무관"):
                x = g[grp == gn]
                m, t, n, G = H.cluster_mean(x["short_ret_h10"], x["t"]) if len(x) else (np.nan, np.nan, 0, 0)
                c = x["cls"].dropna()
                rows.append({"N": N, "구간": pn, "묶음": gn, "사건": n, "날짜": G, "h10": m, "t": t,
                             "초과h10": x["excess_h10"].mean() if len(x) else np.nan,
                             "진짜": (c == "진짜").mean() if len(c) else np.nan,
                             "가짜": (c == "가짜").mean() if len(c) else np.nan,
                             "되돌림청산": (x["exit_reason"] == "reclaim").mean() if len(x) else np.nan})
    return pd.DataFrame(rows)


def _f(x, pct=True, nd=2):
    if x is None or (isinstance(x, (float, np.floating)) and not np.isfinite(x)):
        return "못 구함"
    return f"{x * 100:+.{nd}f}%" if pct else f"{x:.{nd}f}"


def cmd_run():
    ev, P = load_with_features()
    assert int((ev["earn_pm3"] != ev["earn_pm3_chk"]).sum()) == 0
    cells = H.tercile_cells(ev, MAIN, by=("N",))
    tab, summ = H.evaluate_cells(ev, cells)
    cells_v = H.tercile_cells(ev, VARIANT, by=("N",))
    tabv, summv = H.evaluate_cells(ev, cells_v)
    shares = {}
    for nm, p in (("f", H.FRONT), ("b", H.BACK), ("a", H.AUX)):
        sh = cls_shares(ev, cells + cells_v, p)
        shares[nm] = sh
    for t_, pool in ((tab, "main"), (tabv, "variant_entry")):
        t_.insert(0, "pool", pool)
        t_["feature"] = [_feat_of(c) for c in t_["cell"]]
        t_["definition"] = [cell_definition(c) for c in t_["cell"]]
        t_["known_when"] = [DESC[_feat_of(c)][1] for c in t_["cell"]]
        for p in ("f", "b", "a"):
            t_[f"{p}_on_true"] = [shares[p][c][0] for c in t_["cell"]]
            t_[f"{p}_on_fake"] = [shares[p][c][1] for c in t_["cell"]]
            t_[f"{p}_on_reclaim"] = [shares[p][c][2] for c in t_["cell"]]
    allt = pd.concat([tab, tabv], ignore_index=True)
    allt.to_csv(RESULTS / "b5_catalyst_cells.csv", index=False, encoding="utf-8-sig")
    et = earnings_table(ev)
    et.to_csv(RESULTS / "b5_catalyst_earnings_groups.csv", index=False, encoding="utf-8-sig")
    sj = {"main": {k: v for k, v in summ.items()}, "variant": {k: v for k, v in summv.items()}}
    for k in ("main", "variant"):
        for nn in ("null", "null_shift"):
            if nn in sj[k]:
                sj[k][nn] = {a: b for a, b in sj[k][nn].items() if a != "dist"}
    (RESULTS / "b5_catalyst_summary.json").write_text(json.dumps(sj, ensure_ascii=False, indent=1, default=str),
                                                      encoding="utf-8")
    # 국면 비중 상세(통과·상위 칸)
    print(json.dumps(sj, ensure_ascii=False, indent=1, default=str)[:4000])
    sel = tab[tab["selected"]].sort_values("rank")
    cols = ["rank", "cell", "f_n_on", "f_dates_on", "f_diff", "f_t", "b_n_on", "b_dates_on", "b_diff", "b_t",
            "b_mean_on", "b_p", "b_p_shift", "a_diff", "regime_top_ep", "regime_top_share", "pass", "pass_shift",
            "f_ex_diff", "b_ex_diff", "b_on_true", "b_on_fake"]
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(sel[cols].to_string())
        print(tabv[["cell", "f_n_on", "f_diff", "f_t", "b_n_on", "b_diff", "b_t", "b_mean_on", "b_p", "a_diff",
                    "selected", "pass"]].to_string())
        print(et.to_string())
    return ev, tab, summ, tabv, summv, et


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    {"features": cmd_features, "run": cmd_run}[cmd]()
