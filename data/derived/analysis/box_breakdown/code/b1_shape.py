"""갈래 1 b1_shape — 박스·이탈 봉 모양(BRIEF 갈래 1). 판정은 harness.evaluate_cells 그대로.

사용
    py b1_shape.py            # 칸 만들기 → 판정 → results/b1_shape.md, b1_shape_cells.csv, b1_shape_summary.json

사전 등록(결과를 보기 전에 정함, 2026-10-14)
- 사건표: harness.load_events_all() = N20·40·60 사건표를 이어 붙인 것(같은 종목·날짜가 N마다 따로 한 행).
  판정 표본은 short_ret_h10이 있는 행(evaluate_cells 기본). 층(strata) = (N, kind) 기본값.
- 칸 A(3분위): 종목(kind=stock)과 ETF(kind=etf)를 따로, N마다 따로 harness.tercile_cells로 만든다.
  문턱은 앞 구간(2026-01-02~05-29) 같은 kind·같은 N 사건 분포의 1/3·2/3 분위. 위(≥ 2/3 분위)와 아래(≤ 1/3 분위)
  두 칸. 값이 두 개 이하인 특징(gap_break, brk60)은 =1(hi)/=0(lo) 칸.
  특징 = 하네스 기본 9개(box_w_atr, slope50, touches, depth_atr, gap_break, close_loc, vol_ratio, ret5_pre,
  ret20_pre) + 박스 길이를 사건 단위로 잰 brk60(아래, N20·N40 사건만; N60 사건은 정의상 늘 1이라 뺀다).
- 박스 길이 N: N=20·40·60 차원 자체(칸마다 N을 따로)와 brk60 = 종가(t) < 최저 저가(t−60..t−1)
  ('더 긴 60일 박스까지 깼는가')로 본다. N을 그대로 특징으로 쓰는 칸은 층에 N이 들어 있어 순열이 안 움직이므로
  만들지 않는다(qa_harness.md 한계 3).
- 칸 H(가설 '박스다움'): 폭 box_w_atr ≤ 앞 구간 같은 kind·N 1/3 분위(칸 A의 아래 문턱과 같은 값)
  & 접촉 touches ≥ 3 & 50일선 평평 |slope50| ≤ 2.0(%/20거래일). 셋 중 하나라도 NaN이면 NaN.
  N20·N40·N60 각각과 N 합침(각 행은 자기 N 문턱) 4칸, 종목·ETF 각각.
  민감도(종목만): 평평을 |SMA50(t) − SMA50(t−20)| ≤ 1.0 × ATR20(t−1)로 바꾼 4칸.
  가설 방향: 박스다움 켜짐이 진짜 하락(cls='진짜')이 더 많고 공매도 손익이 더 크다.
- 판정: 칸 A + 칸 H 전체를 evaluate_cells 한 번(top 10, 켜짐 ≥ 15건, 켜짐 날짜 ≥ 8일, 순열 200, 귀무 200)에
  넣어 갈래 상위 10칸·통과·선택편향 귀무를 낸다. 가설 칸은 순위와 무관하게 보려고 evaluate_cells를 가설 칸만으로
  한 번 더 부른다(top = 가설 칸 수, 같은 시드·같은 조건, 귀무 없음). 앞 구간 차이 ≤ 0이면 BRIEF 고르기에서
  빠지므로 '앞 구간 반대 방향 → 탈락'으로 적는다.
- 설명용(판정 아님): 진짜·가짜 비율 켜짐−꺼짐(날짜 묶음 t), excess_h10 차이, h5·h20 차이.
- 특징 시점: 모두 t 봉(종가)까지. gap_break만 t 시가에 알 수 있고 나머지는 t 종가 뒤. 진입은 t+1 시가.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
import harness as H  # noqa: E402

RES = H.RESULTS
FEATS = ["box_w_atr", "slope50", "touches", "depth_atr", "gap_break", "close_loc", "vol_ratio",
         "ret5_pre", "ret20_pre"]
NEW = ["brk60"]
FLAT_PCT = 2.0
FLAT_ATR = 1.0
TOUCH_MIN = 3

DESC = {
    "box_w_atr": "박스 폭 = (최고 고가 − 최저 저가)(t−N..t−1) ÷ ATR20(t−1)",
    "slope50": "50일선 기울기 = SMA50(t)/SMA50(t−20) − 1 (%)",
    "touches": "바닥 접촉 수 = t−N..t−1 중 |저가 − L_N| ≤ 0.5·ATR20인 날 수",
    "depth_atr": "이탈 깊이 = (L_N − 종가(t)) ÷ ATR20",
    "gap_break": "갭 이탈 = 시가(t) < L_N(1) / 장중 이탈(0)",
    "close_loc": "종가 위치 = (종가 − 저가) ÷ (고가 − 저가), t 봉",
    "vol_ratio": "거래량 배수 = 거래량(t) ÷ 평균 거래량(t−20..t−1)",
    "ret5_pre": "이탈 전 5일 수익 = 종가(t−1)/종가(t−6) − 1",
    "ret20_pre": "이탈 전 20일 수익 = 종가(t−1)/종가(t−21) − 1",
    "brk60": "60일 저점도 이탈 = 종가(t) < 최저 저가(t−60..t−1)",
}
KNOWN = {f: "t 종가(장 마감) 뒤 → t+1 시가 진입" for f in DESC}
KNOWN["gap_break"] = "t 시가(장 시작)에 확정, 진입은 t+1 시가"
KNOWN["box_w_atr"] = KNOWN["touches"] = "t−1 종가 뒤(박스·ATR은 t−1까지) → t+1 시가 진입"
KNOWN["ret5_pre"] = KNOWN["ret20_pre"] = "t−1 종가 뒤 → t+1 시가 진입"
KIND_KO = {"stock": "종목", "etf": "ETF"}


# ───────────────────────────── 특징 ─────────────────────────────
def add_features(ev, P):
    """brk60(N60 행은 NaN), flat_atr = |SMA50(t) − SMA50(t−20)| ÷ ATR20, 진짜·가짜 표지."""
    ev = ev.copy()
    ev["brk60"] = np.nan
    ev["flat_atr"] = np.nan
    for s, g in ev.groupby("sym"):
        o, h, l, c = (P.col(f, s) for f in "OHLC")
        L60 = H.box_levels(o, h, l, c, 60)[0]
        s50 = H._sma(c, 50)
        i = g["i"].to_numpy()
        b = np.where(np.isnan(L60[i]), np.nan, (c[i] < L60[i]).astype(float))
        ev.loc[g.index, "brk60"] = np.where(g["N"].to_numpy() == 60, np.nan, b)
        prev = np.where(i >= 20, s50[np.maximum(i - 20, 0)], np.nan)
        ev.loc[g.index, "flat_atr"] = np.abs(s50[i] - prev) / g["atr20"].to_numpy()
    ev["is_true"] = np.where(ev["cls"].isna(), np.nan, (ev["cls"] == "진짜").astype(float))
    ev["is_fake"] = np.where(ev["cls"].isna(), np.nan, (ev["cls"] == "가짜").astype(float))
    return ev


# ───────────────────────────── 칸 ─────────────────────────────
def parse_main(name):
    kind, nkey, rest = name.split("|", 2)
    f = max((x for x in FEATS + NEW if rest.startswith(x)), key=len)
    cond = rest[len(f):]
    return dict(group="A", kind=kind, N=int(nkey[1:]), feature=f, cond=cond)


def main_cells(ev):
    cells, meta = [], []
    for kind in ("stock", "etf"):
        for nm, s in H.tercile_cells(ev, FEATS + NEW, kind=kind, tag=f"{kind}|"):
            cells.append((nm, s))
            meta.append(parse_main(nm))
    return cells, meta


def hyp_cells(ev):
    """박스다움 칸. 폭 문턱은 tercile_cells와 같은 방법(앞 구간 같은 kind·N 1/3 분위)."""
    fr = H.in_period(ev, H.FRONT)
    cells, meta, q1s = [], [], {}
    for kind in ("stock", "etf"):
        variants = [("pct", f"|slope50|≤{FLAT_PCT:g}")]
        if kind == "stock":
            variants.append(("atr", f"|ΔSMA50|≤{FLAT_ATR:g}ATR"))
        for vkey, vlab in variants:
            on_all = pd.Series(np.nan, index=ev.index)
            for N in H.NS:
                g = ev[(ev["kind"] == kind) & (ev["N"] == N)]
                x = g["box_w_atr"].astype(float)
                xf = x[fr.loc[g.index] & x.notna()]
                q1 = float(np.quantile(xf, 1 / 3))
                q1s[(kind, N)] = q1
                flat = (g["slope50"].abs() <= FLAT_PCT) if vkey == "pct" else (g["flat_atr"] <= FLAT_ATR)
                fl_nan = g["slope50"].isna() if vkey == "pct" else g["flat_atr"].isna()
                nan = x.isna() | g["touches"].isna() | fl_nan
                on = (x <= q1) & (g["touches"] >= TOUCH_MIN) & flat
                v = pd.Series(np.where(nan, np.nan, on.astype(float)), index=g.index)
                s = pd.Series(np.nan, index=ev.index)
                s.loc[g.index] = v
                on_all.loc[g.index] = v
                nm = f"hyp|{kind}|N{N}|박스다움(폭≤{q1:.4g}·접촉≥{TOUCH_MIN}·{vlab})"
                cells.append((nm, s))
                meta.append(dict(group="H" if vkey == "pct" else "H_sens", kind=kind, N=N, feature="box_like",
                                 cond=f"폭≤{q1:.4g}·접촉≥{TOUCH_MIN}·{vlab}"))
            nm = f"hyp|{kind}|N20·40·60|박스다움(폭≤N별 하위3분위·접촉≥{TOUCH_MIN}·{vlab})"
            cells.append((nm, on_all))
            meta.append(dict(group="H" if vkey == "pct" else "H_sens", kind=kind, N=0, feature="box_like",
                             cond=f"폭≤N별 하위3분위·접촉≥{TOUCH_MIN}·{vlab}"))
    return cells, meta, q1s


def definition(m, q1s):
    kind = KIND_KO[m["kind"]]
    nlab = "N=20·40·60 합침" if m["N"] == 0 else f"N={m['N']}"
    if m["feature"] == "box_like":
        if m["N"] == 0:
            qs = ", ".join(f"N{N} {q1s[(m['kind'], N)]:.3f}" for N in H.NS)
            wq = f"폭 ≤ 같은 N 앞 구간 1/3 분위({qs})"
        else:
            wq = f"폭 ≤ {q1s[(m['kind'], m['N'])]:.3f}(앞 구간 1/3 분위)"
        flat = (f"|slope50| ≤ {FLAT_PCT:g}%(50일선 20거래일 변화)" if m["group"] == "H"
                else f"|SMA50(t) − SMA50(t−20)| ≤ {FLAT_ATR:g}·ATR20")
        return (f"{kind} {nlab} 사건 중 박스다움: {wq} & 바닥 접촉 ≥ {TOUCH_MIN} & {flat}. "
                f"꺼짐 = 같은 {kind}·N의 나머지 사건")
    c = m["cond"]
    if c.startswith("="):
        cond = f"{m['feature']} = {c[1:].split('(')[0]}"
    elif c.startswith("≥"):
        cond = f"{m['feature']} ≥ {c[1:].split('(')[0]}(앞 구간 {kind} N{m['N']} 사건 2/3 분위)"
    else:
        cond = f"{m['feature']} ≤ {c[1:].split('(')[0]}(앞 구간 {kind} N{m['N']} 사건 1/3 분위)"
    return f"{kind} {nlab} 사건 중 {cond}. {DESC[m['feature']]}. 꺼짐 = 같은 {kind}·N의 나머지 사건"


def known_when(m):
    if m["feature"] == "box_like":
        return "t 종가 뒤(폭·접촉은 t−1까지, 50일선은 t 종가) → t+1 시가 진입"
    return KNOWN[m["feature"]]


# ───────────────────────────── 설명용 통계 ─────────────────────────────
def desc_stats(df, cells, ycol, period):
    """evaluate_cells 내부 stats와 같은 계산(켜짐−꺼짐 차이, 날짜 묶음 t)을 다른 결과 열에 쓴다."""
    d = df[df["short_ret_h10"].notna()]
    M = H._cell_matrix(d, cells)
    mask = H.in_period(d, period).to_numpy()
    y = d[ycol].to_numpy(float)[mask]
    Mm = np.where(~np.isnan(y)[:, None], M[mask], np.nan)
    g = H._codes(d["t"].to_numpy()[mask])
    n1, n0, m1, m0, diff, t = H.cluster_diff(Mm, np.nan_to_num(y), g)
    return m1, m0, diff, t


# ───────────────────────────── 실행 ─────────────────────────────
def run():
    P = H.load_panel()
    ev = add_features(H.load_events_all(), P)
    cA, mA = main_cells(ev)
    cH, mH, q1s = hyp_cells(ev)
    cells, meta = cA + cH, mA + mH
    names = [n for n, _ in cells]
    assert len(set(names)) == len(names), "칸 이름 중복"

    tab, summ = H.evaluate_cells(ev, cells)
    tabH, summH = H.evaluate_cells(ev, cH, top=len(cH), null=False)

    md = pd.DataFrame(meta)
    tab = pd.concat([md, tab], axis=1)
    tab["definition"] = [definition(m, q1s) for m in meta]
    tab["known_when"] = [known_when(m) for m in meta]
    for p, per in (("f", H.FRONT), ("b", H.BACK), ("a", H.AUX)):
        for ycol, lab in (("is_true", "true"), ("is_fake", "fake")):
            m1, m0, d, t = desc_stats(ev, cells, ycol, per)
            tab[f"{p}_{lab}_on"], tab[f"{p}_{lab}_off"], tab[f"{p}_{lab}_diff"], tab[f"{p}_{lab}_t"] = m1, m0, d, t
        for ycol in ("short_ret_h5", "short_ret_h20"):
            _, _, d, t = desc_stats(ev, cells, ycol, per)
            hn = ycol.split("_")[-1]
            tab[f"{p}_{hn}_diff"], tab[f"{p}_{hn}_t"] = d, t
    # 가설 칸 별도 판정 붙이기
    hcols = ["eligible", "selected", "b_p", "b_p_shift", "pass", "pass_shift", "regime_top_ep", "regime_top_share",
             "one_regime"]
    hx = tabH.set_index("cell")[hcols].add_prefix("hyp_")
    tab = tab.join(hx, on="cell")
    tab.to_csv(RES / "b1_shape_cells.csv", index=False, encoding="utf-8-sig")

    d = ev[ev["short_ret_h10"].notna()]
    counts = {}
    for pn, per in (("f", H.FRONT), ("b", H.BACK), ("a", H.AUX)):
        x = d[H.in_period(d, per)]
        counts[pn] = {
            "rows": int(len(x)), "dates": int(x["t"].nunique()),
            "uniq_sym_date": int(x[["sym", "t"]].drop_duplicates().shape[0]),
            **{f"{k}_N{N}": int(((x["kind"] == k) & (x["N"] == N)).sum()) for k in ("stock", "etf") for N in H.NS},
            **{f"{k}_N{N}_dates": int(x.loc[(x["kind"] == k) & (x["N"] == N), "t"].nunique())
               for k in ("stock", "etf") for N in H.NS},
        }
    out = {"summ": summ, "summ_hyp": summH, "counts": counts, "q1s": {f"{k}|N{N}": v for (k, N), v in q1s.items()},
           "n_cells": len(cells),
           "n_cells_by_group": {f"{g}|{k}": int(v) for (g, k), v in md.groupby(["group", "kind"]).size().items()}}
    (RES / "b1_shape_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str),
                                               encoding="utf-8")
    return tab, summ, summH, counts, q1s, ev


# ───────────────────────────── 보고서 ─────────────────────────────
def _ok(x):
    return x is not None and not (isinstance(x, float) and np.isnan(x)) and np.isfinite(float(x))


def pc(x, nd=2):
    return f"{float(x) * 100:+.{nd}f}%" if _ok(x) else "못 구함"


def pp(x, nd=2):
    return f"{float(x) * 100:+.{nd}f}%p" if _ok(x) else "못 구함"


def nm(x, nd=2):
    return f"{float(x):.{nd}f}" if _ok(x) else "못 구함"


def sh(x):
    return f"{float(x) * 100:.0f}%" if _ok(x) else "못 구함"


def _cell(tab, kind, N, feature, prefix):
    x = tab[(tab["kind"] == kind) & (tab["N"] == N) & (tab["feature"] == feature) & tab["cond"].str.startswith(prefix)]
    return x.iloc[0] if len(x) else None


def _base(ev, kind, N, per):
    g = ev[(ev["kind"] == kind) & (ev["N"] == N) & H.in_period(ev, per) & ev["short_ret_h10"].notna()]
    m, t, n, G = H.cluster_mean(g["short_ret_h10"], g["t"])
    c = g["cls"].dropna()
    return m, t, n, G, ((c == "진짜").mean() if len(c) else np.nan), ((c == "가짜").mean() if len(c) else np.nan)


def report(tab, summ, counts, ev, path=None):
    path = path or RES / "b1_shape.md"
    sel = tab[tab["selected"]].sort_values("rank")
    stA = tab[(tab["group"] == "A") & (tab["kind"] == "stock")]
    etA = tab[(tab["group"] == "A") & (tab["kind"] == "etf")]
    hyp = tab[tab["group"] != "A"]
    nul, nus = summ["null"], summ.get("null_shift", {})
    f, b, a = counts["f"], counts["b"], counts["a"]
    pos = {p: int((stA.loc[stA[f"{p}_n_on"] >= 15, f"{p}_mean_on"] > 0).sum()) for p in "fba"}
    nst = {p: int((stA[f"{p}_n_on"] >= 15).sum()) for p in "fba"}
    hp = hyp[(hyp["kind"] == "stock") & (hyp["group"] == "H") & (hyp["N"] == 0)].iloc[0]
    hs = hyp[(hyp["kind"] == "stock") & (hyp["group"] == "H_sens") & (hyp["N"] == 0)].iloc[0]
    n_cells = len(tab)
    be_rows, be_b = [], {}
    for N in H.NS:
        for pn, per in (("앞", H.FRONT), ("뒤", H.BACK), ("2025", H.AUX)):
            g = ev[(ev["kind"] == "stock") & (ev["N"] == N) & H.in_period(ev, per)
                   & ev["short_ret_h10"].notna() & ev["cls"].notna()]
            tr, nt = g[g["cls"] == "진짜"]["short_ret_h10"], g[g["cls"] != "진짜"]["short_ret_h10"]
            # 진짜 평균이 0 이하이면 어떤 진짜 비율로도 플러스가 안 된다 → NaN('없음')
            be = -nt.mean() / (tr.mean() - nt.mean()) if len(tr) and len(nt) and tr.mean() > 0 > nt.mean() else np.nan
            if pn == "뒤":
                be_b[N] = be
            be_rows.append([str(N), pn, f"{pc(tr.mean())} ({len(tr)})", f"{pc(nt.mean())} ({len(nt)})",
                            sh(be) if _ok(be) else "없음(진짜도 손실)", sh(len(tr) / len(g)) if len(g) else "못 구함"])
    be_txt = "뒤 구간 N20·40·60: " + ", ".join(sh(be_b[N]) if _ok(be_b[N]) else "없음" for N in H.NS)
    max_true = {p: stA.loc[stA[f"{p}_n_on"] >= 15, f"{p}_true_on"].max() for p in "fb"}
    L = [
        "# 갈래 1 b1_shape — 박스·이탈 봉 모양으로 진짜 하락을 고를 수 있나 (2026-10-14)", "",
        f"**결론: 못 고른다.** {n_cells}칸 중 통과 {summ['n_pass']}칸(선택편향 귀무 평균 {nul['mean']:.2f}칸, "
        f"p = {nul['p_value']:.2f}). 2026년 앞·뒤 구간 모두, 켜진 사건이 15건 이상인 종목 칸 가운데 평균 공매도 손익이 "
        f"플러스인 칸이 없다(앞 {pos['f']}/{nst['f']}, 뒤 {pos['b']}/{nst['b']}). '박스다움' 가설도 2026년에는 반대 방향이었다.",
        "",
        "## 핵심",
        "",
        f"- 앞 구간(1~5월) 상위 10칸은 모두 종목 칸이고 뒤 구간(6~9월) 통과 0. 뒤 구간 날짜 묶음 t가 2를 넘은 상위 칸도 없다"
        f"(최대 {nm(sel['b_t'].max())}). 상위 10칸의 뒤 구간 켜짐 평균은 {pc(sel['b_mean_on'].min())} ~ "
        f"{pc(sel['b_mean_on'].max())}로 모두 손실이다.",
        f"- 박스다움(폭 하위 3분위·접촉 ≥ 3·|slope50| ≤ 2%, N 합침): 앞 {pp(hp['f_diff'])}(t {nm(hp['f_t'])}), "
        f"뒤 {pp(hp['b_diff'])}(t {nm(hp['b_t'])}) — 2026 두 구간 모두 반대 방향. 2025에는 {pp(hp['a_diff'])}"
        f"(t {nm(hp['a_t'])}, 켜짐 {int(hp['a_n_on'])}건)로 통했다. 장세가 바뀌었다는 뜻이고 2026 근거로는 쓸 수 없다.",
        f"- 진짜 하락 비율은 박스다움 켜짐이 조금 높았지만(뒤 {sh(hp['b_true_on'])} vs {sh(hp['b_true_off'])}, "
        f"t {nm(hp['b_true_t'])}) 공매도 손익은 켜짐 {pc(hp['b_mean_on'])} vs 꺼짐 {pc(hp['b_mean_off'])}로 나아지지 않았다. "
        f"진짜 비율이 손익분기({be_txt})에 한참 못 미치기 때문이다(아래 설명용 관찰).",
        "- ETF는 칸마다 앞 구간 켜짐이 최대 "
        f"{int(etA['f_n_on'].max())}건·{int(etA['f_dates_on'].max())}일이라 고르기 조건(15건·8일)에 하나도 못 들었다. 판정 불가.",
        "",
        "## 정의(사전 등록, `code/b1_shape.py` 머리말)",
        "",
        "- 사건표: 하네스 N20·40·60 사건을 이어 붙임(같은 종목·날짜가 N마다 한 행). 진입 t+1 시가 공매도, "
        "되돌림 손절 또는 t+10 종가, 비용 왕복 0.10%(대차 비용 없음). 판정은 `harness.evaluate_cells` 그대로"
        "(층 = N·kind, 순열 200, 귀무 200, 켜짐 ≥ 15건·날짜 ≥ 8일, 상위 10칸).",
        "- 칸 A(3분위): 종목·ETF 따로, N마다 따로. 문턱은 앞 구간 같은 kind·N 사건의 1/3·2/3 분위. "
        "특징 = box_w_atr, slope50, touches, depth_atr, gap_break, close_loc, vol_ratio, ret5_pre, ret20_pre + "
        "brk60(종가(t) < 60일 최저 저가, N20·N40 사건만; 박스 길이를 사건 단위로 잰 것).",
        f"- 칸 H(박스다움): 폭 ≤ 같은 kind·N 앞 구간 1/3 분위 & 접촉 ≥ {TOUCH_MIN} & |slope50| ≤ {FLAT_PCT:g}%"
        f"(50일선 20거래일 변화). N별 3칸 + N 합침 1칸, 종목·ETF. 민감도(종목): 평평 = |ΔSMA50(20일)| ≤ {FLAT_ATR:g}·ATR20 4칸.",
        "- 박스 길이 N 자체를 특징으로 쓰는 칸은 만들지 않았다(층에 N이 있어 순열이 움직이지 않음, 감사 한계 3). "
        "N은 칸 차원과 brk60, 그리고 N별 기준선으로 본다.",
        "- 특징 시점: 박스 폭·접촉·이탈 전 수익은 t−1까지, 나머지는 t 봉(종가)까지. gap_break만 t 시가에 확정. 모두 t+1 시가 전에 안다.",
        "",
        "## 탐색 범위",
        "",
        f"- 시험한 칸 **{n_cells}개** = 종목 3분위 {len(stA)}(9특징×2방향×3N = 54 + brk60 2방향×N20·N40 = 4) + "
        f"ETF 3분위 {len(etA)}(N60 gap_break는 앞 구간 값이 한 가지라 1칸) + 박스다움 {len(hyp)}"
        f"(종목 4 + 종목 민감도 4 + ETF 4).",
        f"- 고르기 적격 {summ['n_eligible']}칸(모두 종목 3분위), 고름 {summ['n_selected']}칸, 통과 {summ['n_pass']}칸"
        f"(보조 원형 이동 기준 {summ['n_pass_shift']}칸).",
        f"- 표본(판정에 쓴 행, N 합침): 앞 {f['rows']}행(서로 다른 종목·날짜 {f['uniq_sym_date']}, 이탈 날짜 {f['dates']}일), "
        f"뒤 {b['rows']}행({b['uniq_sym_date']}, {b['dates']}일), 2025 {a['rows']}행({a['uniq_sym_date']}, {a['dates']}일).",
        "",
        "| 구간 | 종목 N20 | N40 | N60 | ETF N20 | N40 | N60 |",
        "|---|---|---|---|---|---|---|",
    ]
    for pn, c in (("앞(1~5월)", f), ("뒤(6~9월)", b), ("2025", a)):
        L.append(f"| {pn} | " + " | ".join(f"{c[f'{k}_N{N}']}건/{c[f'{k}_N{N}_dates']}일"
                                           for k in ("stock", "etf") for N in H.NS) + " |")
    L += ["", "앞선 박스·손절·레버리지 연구와 이 연구의 기준선을 본 뒤의 탐색이라 선택편향이 쌓여 있다.", "",
          "## 앞 구간 상위 10칸 (모두 종목)", "",
          "차이 = 켜짐 − 꺼짐 h10 공매도 손익(비용 뒤). t는 이탈일 묶음. p = 날짜 묶음 순열(한쪽), 원형 p = 보조.",
          "",
          "| 순위 | 칸 | 앞 켜짐 | 앞 차이(t) | 뒤 켜짐 | 뒤 켜짐 평균 | 뒤 차이(t) | p / 원형 p | 2025 차이(t) | 뒤 초과 차이 | 뒤 진짜 비율 켜짐/꺼짐 | 통과 |",
          "|---:|---|---|---|---|---:|---|---|---|---:|---|---|"]
    for r in sel.to_dict("records"):
        L.append(f"| {int(r['rank'])} | `{r['cell']}` | {int(r['f_n_on'])}건/{int(r['f_dates_on'])}일 | "
                 f"{pp(r['f_diff'])} ({nm(r['f_t'])}) | {int(r['b_n_on'])}건/{int(r['b_dates_on'])}일 | "
                 f"{pc(r['b_mean_on'])} | {pp(r['b_diff'])} ({nm(r['b_t'])}) | {nm(r['b_p'], 3)} / {nm(r['b_p_shift'], 3)} | "
                 f"{pp(r['a_diff'])} ({nm(r['a_t'])}) | {pp(r['b_ex_diff'])} | {sh(r['b_true_on'])}/{sh(r['b_true_off'])} | "
                 f"{'통과' if r['pass'] else '탈락'} |")
    L += ["",
          f"- 한 국면 의존 점검: 상위 10칸 모두 뒤 구간 켜짐 손익 합이 음수라 비중을 계산하지 않았다(맞힌 손익이 없음).",
          f"- 선택편향 귀무(특징 표지를 날짜 묶음째 섞어 고르기·확인 반복 200번): 통과 칸 수 평균 {nul['mean']:.3f}, "
          f"95% 분위 {nul['q95']:.0f}, 최대 {nul['max']}. 실제 {summ['n_pass']}칸 → p = {nul['p_value']:.3f}. "
          + (f"원형 이동 귀무 평균 {nus['mean']:.3f}, 최대 {nus['max']}." if nus else ""),
          "",
          "## 가설: 박스다움이 진짜 하락을 더 낳나 (종목)", "",
          "| 칸 | 앞 켜짐 | 앞 켜짐/꺼짐 평균 | 앞 차이(t) | 뒤 켜짐 | 뒤 켜짐/꺼짐 평균 | 뒤 차이(t) | 2025 켜짐 | 2025 차이(t) | 진짜 비율 켜짐/꺼짐 앞·뒤·2025 | 판정 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in hyp[hyp["kind"] == "stock"].itertuples():
        if r.f_n_on < 15:
            v = "앞 켜짐 < 15건 → 판정 불가"
        elif not (r.f_diff > 0):
            v = "앞 구간 반대 방향 → 탈락"
        else:
            v = "통과" if r.hyp_pass else "탈락"
        lab = ("민감도 " if r.group == "H_sens" else "") + ("N 합침" if r.N == 0 else f"N{r.N}")
        L.append(f"| {lab} | {int(r.f_n_on)}건/{int(r.f_dates_on)}일 | {pc(r.f_mean_on)} / {pc(r.f_mean_off)} | "
                 f"{pp(r.f_diff)} ({nm(r.f_t)}) | {int(r.b_n_on)}건/{int(r.b_dates_on)}일 | {pc(r.b_mean_on)} / "
                 f"{pc(r.b_mean_off)} | {pp(r.b_diff)} ({nm(r.b_t)}) | {int(r.a_n_on)}건 | {pp(r.a_diff)} ({nm(r.a_t)}) | "
                 f"{sh(r.f_true_on)}/{sh(r.f_true_off)} · {sh(r.b_true_on)}/{sh(r.b_true_off)} · "
                 f"{sh(r.a_true_on)}/{sh(r.a_true_off)} | {v} |")
    w60 = _cell(tab, "stock", 60, "box_w_atr", "≤")
    L += ["",
          f"- 사전 등록 주 칸(N 합침, |slope50| ≤ 2%)은 앞 {int(hp['f_n_on'])}건에서 {pp(hp['f_diff'])}, 뒤 "
          f"{int(hp['b_n_on'])}건에서 {pp(hp['b_diff'])}. 민감도(ATR 기준 평평)는 앞 {pp(hs['f_diff'])}"
          f"(t {nm(hs['f_t'])}), 뒤 {pp(hs['b_diff'])}(t {nm(hs['b_t'])})로 더 나쁘다.",
          f"- 박스다움의 한 요소인 좁은 박스만 봐도 같다: N60 폭 하위 3분위는 뒤 {pp(w60['b_diff'])}(t {nm(w60['b_t'])}), "
          f"2025 {pp(w60['a_diff'])}(t {nm(w60['a_t'])}). 해마다 부호가 뒤집힌다." if w60 is not None else "",
          "- 장세 전환 위험: 2025에는 박스다움 공매도가 통했고(N 합침 켜짐 평균 "
          f"{pc(hp['a_mean_on'])}), 2026에는 켜짐 평균이 앞 {pc(hp['f_mean_on'])}, 뒤 {pc(hp['b_mean_on'])}로 손실이었다.",
          "",
          "## 박스 길이 N (종목 기준선과 brk60)", "",
          "| N | 앞 h10 평균(t) | 뒤 h10 평균(t) | 2025 h10 평균(t) | 진짜 비율 앞·뒤·2025 |",
          "|---|---|---|---|---|"]
    for N in H.NS:
        rr = [_base(ev, "stock", N, per) for per in (H.FRONT, H.BACK, H.AUX)]
        L.append(f"| {N} | " + " | ".join(f"{pc(x[0])} ({nm(x[1])}, {x[2]}건/{x[3]}일)" for x in rr)
                 + " | " + " · ".join(sh(x[4]) for x in rr) + " |")
    L += ["", "| brk60 칸 | 앞 켜짐 | 앞 차이(t) | 뒤 켜짐 | 뒤 차이(t) | 2025 차이(t) |", "|---|---|---|---|---|---|"]
    for r in stA[stA["feature"] == "brk60"].itertuples():
        L.append(f"| `{r.cell}` | {int(r.f_n_on)}건 | {pp(r.f_diff)} ({nm(r.f_t)}) | {int(r.b_n_on)}건 | "
                 f"{pp(r.b_diff)} ({nm(r.b_t)}) | {pp(r.a_diff)} ({nm(r.a_t)}) |")
    L += ["", "- 뒤 구간에서는 박스가 길수록 공매도 손실이 컸고(N20 → N60), 앞 구간과 2025는 N 사이 차이가 작았다. "
          "60일 저점까지 깬 이탈(brk60=1)도 공매도에 더 낫지 않았다(2026 뒤·2025 모두 반대 방향, t < 2).",
          "",
          "## ETF (판정 불가)", "",
          f"- 3분위 {len(etA)}칸 + 박스다움 4칸, 앞 구간 켜짐 최대 {int(etA['f_n_on'].max())}건·"
          f"{int(etA['f_dates_on'].max())}일, 뒤 구간 켜짐 최대 {int(etA['b_n_on'].max())}건. 고르기 조건(15건·8일)에 든 칸 0.",
          "- 기준선(조건 없음, h10 공매도):", "",
          "| N | 앞 | 뒤 | 2025 |", "|---|---|---|---|"]
    for N in H.NS:
        rr = [_base(ev, "etf", N, per) for per in (H.FRONT, H.BACK, H.AUX)]
        L.append(f"| {N} | " + " | ".join(f"{pc(x[0])} ({x[2]}건/{x[3]}일, t {nm(x[1])})" for x in rr) + " |")
    d6 = _cell(tab, "stock", 20, "depth_atr", "≥")
    d6l = _cell(tab, "stock", 20, "depth_atr", "≤")
    L += ["", "## 설명용 관찰(판정 아님)", "",
          "이 절의 진짜·가짜 비율 비교와 손익분기 표는 판정 결과를 본 뒤 설명하려고 덧붙였다. 칸 정의·판정은 바꾸지 않았다.", ""]
    if d6 is not None and d6l is not None:
        L += [f"- 깊게 깬 이탈(N20 depth_atr 상위 3분위)은 '진짜' 분류가 많다(뒤 {sh(d6['b_true_on'])} vs "
              f"{sh(d6['b_true_off'])}, t {nm(d6['b_true_t'])}). 하지만 공매도 손익 차이는 {pp(d6['b_diff'])}"
              f"(t {nm(d6['b_t'])}). 진짜 기준(L − 2·ATR)까지 이미 일부 내려와 있어 분류가 기계적으로 쉬워질 뿐이고, "
              "t+1 시가 진입 뒤 수익은 늘지 않는다.",
              f"- 반대로 얕게 깬 이탈(N20 depth_atr 하위 3분위)은 뒤 구간 공매도가 덜 나빴다({pp(d6l['b_diff'])}, "
              f"t {nm(d6l['b_t'])}). 하지만 앞 구간은 {pp(d6l['f_diff'])}여서 고르기 상위 10에 못 들었고, "
              f"켜짐 평균도 {pc(d6l['b_mean_on'])}로 손실이다."]
    L += ["- 왜 진짜 비율이 조금 올라도 소용없나: 진짜로 분류된 사건도 이익이 작거나 손실이고, 가짜·중간은 평균 4~7%를 잃는다. "
          "손익분기 진짜 비율 = −(가짜·중간 평균) ÷ (진짜 평균 − 가짜·중간 평균).", "",
          "| N | 구간 | 진짜 평균(건) | 가짜·중간 평균(건) | 손익분기 진짜 비율 | 실제 진짜 비율 |", "|---|---|---|---|---|---|"]
    for row in be_rows:
        L.append("| " + " | ".join(row) + " |")
    L += ["", f"- 종목 칸(켜짐 ≥ 15건) 중 진짜 비율이 가장 높은 칸도 앞 {sh(max_true['f'])}, 뒤 {sh(max_true['b'])}로 "
          "손익분기에 못 미친다. 진짜 분류는 사후 표지(t+1..t+10 저가)라 매매 근거가 아니고 설명용이다.",
          "",
          "## 한계", "",
          "- 박스다움 N별 칸은 앞 구간 켜짐 11~14건이라 판정 조건(15건)에 못 미친다. N 합침 칸은 같은 종목·날짜가 N마다 "
          "중복되며, 날짜 묶음 t와 순열로 처리했다.",
          "- '평평' 문턱(|slope50| ≤ 2%)과 '접촉 ≥ 3'은 결과를 보기 전에 정한 값이다. ATR 기준 민감도도 같은 결론이다.",
          "- 종목 수준 특징에서 날짜 묶음 순열은 보수적이다(감사 한계 2). 다만 상위 10칸은 t ≥ 2와 켜짐 평균 > 0을 "
          "먼저 못 넘어서, 순열의 보수성이 결론을 바꾸지 않는다.",
          "- 개별 종목은 공매도 대리 수익(−주가 수익)이다. 대차 비용을 넣으면 더 나빠진다. ETF는 표본이 모자라 실제 인버스 ETF 손익은 보지 않았다(갈래 7).",
          "",
          "## 파일", "",
          "- `code/b1_shape.py` — 특징·칸·판정·이 보고서(`py b1_shape.py`, 기계 부하에 따라 20초~1분 남짓)",
          "- `results/b1_shape_cells.csv` — 127칸 전체(앞·뒤·2025 차이, t, p, 진짜·가짜 비율, h5·h20, 정의·시점)",
          "- `results/b1_shape_summary.json` — 요약·귀무 분포·표본 수", ""]
    Path(path).write_text("\n".join(x for x in L if x is not None), encoding="utf-8")
    return path


if __name__ == "__main__":
    import time
    t0 = time.time()
    tab, summ, summH, counts, q1s, ev = run()
    report(tab, summ, counts, ev)
    print(f"칸 {summ['n_cells']}개, 적격 {summ['n_eligible']}, 고름 {summ['n_selected']}, 통과 {summ['n_pass']} "
          f"({time.time() - t0:.1f}초)")
