"""갈래 6 — 조합(BRIEF 갈래 6). 판정은 harness.evaluate_cells 그대로(새 판정 로직 없음).

실행
    py b6_combo.py plan   # 특징·문턱만 계산(결과값 안 봄) → ../results/b6_combo_plan.md, b6_combo_plan_cells.json
    py b6_combo.py run    # 계획과 같은 칸인지 대조 → 로지스틱 학습(앞 구간) → evaluate_cells
                          #   → ../results/b6_combo.md, b6_combo_cells.csv, b6_combo_summary.json

정의(결과를 보기 전에 정함, 2026-10-14). 자세한 목록은 b6_combo_plan.md.
- 재료 = 다른 갈래 앞 구간 상위 칸(과제가 준 정의). 한 칸 = 원자(atom): (특징, 방향, 문턱, N, 모집단).
  문턱은 그 갈래와 같은 규칙(harness.tercile_cells: 앞 구간 같은 N(갈래 1은 같은 kind도) 사건의 1/3·2/3 분위,
  두 값 이하면 =1/=0)으로 다시 계산하고, 과제가 준 문턱 문자열과 같은지 확인한다.
- 조합은 같은 N 안에서만, 서로 다른 갈래 원자 2개 또는 3개의 AND. 모집단 = 모든 원자가 정의된(NaN 아님) 사건,
  갈래 1 원자가 끼면 종목만. 켜짐 = 모든 원자 켜짐, 꺼짐 = 모집단 나머지.
- 가족 D(일봉 사건, 결과 short_ret_h10): N20 원자 11개·N40 4개·N60 4개 → 2개 조합 59칸 + 3개 조합 105칸.
- 가족 b2(확인 대기 진입, 종목): bounce N60·N40·N20, hold2 N20 × 같은 N 일봉 원자 1개 = 30칸. 갈래 2와 같은 쌓기:
  켜짐 행 = 변형이 진입했고 원자가 켜진 사건의 변형 손익, 꺼짐 행 = 같은 N·종목 모든 사건의 즉시 진입 손익.
- 가족 b8(60분봉 사건): b8 원자 3개(N28 depth_atr 아래, N14·N7 dbox_pos 위) × 시장 원자(전 거래일까지 알려진 값:
  ^VIX 전일 종가, b4 NVDL·풋/콜 백분위(이탈일 시가 전 공개분), SOXX·QQQ 일봉 20일 박스 상태(전일 종가))
  2개 조합 18칸 + 3개 조합 33칸. 시장 원자 문턱은 b8 앞 구간 같은 N 사건의 3분위(같은 규칙).
  b8 N14 tod_close=0은 앞 구간 꺼짐이 1건뿐인 퇴화 칸이라 쓰지 않는다. b5 sub_cobreak_share ≤ 0은 sub_cobreak_n ≤ 0과
  같은 사건 집합이라 하나만 쓴다.
- 로지스틱 1개: 일봉 사건(N20·40·60, 종목+ETF) 앞 구간 전체로 학습, 목표 = 하네스 분류 '진짜'(1) vs 나머지(0),
  특징 8개(ret5_pre, depth_atr, gap_size, cobreak_share, vix, nvdl_flow20_p, pc_equity5_p, soxx_brk) 앞 구간 평균·표준편차로
  표준화, L2 규제 C = 0.1 고정(sklearn lbfgs). 칸 = N마다 앞 구간 예측확률 2/3 분위 이상 = 켜짐(3칸).
- 판정: 위 248칸을 한 번의 evaluate_cells(상위 10칸, 켜짐 ≥ 15건·8날짜, 순열 200, 귀무 200, 층 = 가족·N·kind).
"""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
import harness as H  # noqa: E402

RES = H.RESULTS
PLAN_MD = RES / "b6_combo_plan.md"
PLAN_JSON = RES / "b6_combo_plan_cells.json"
OUT_MD = RES / "b6_combo.md"
OUT_CSV = RES / "b6_combo_cells.csv"
OUT_JSON = RES / "b6_combo_summary.json"
LOGIT_FEATS = ["ret5_pre", "depth_atr", "gap_size", "cobreak_share", "vix", "nvdl_flow20_p", "pc_equity5_p",
               "soxx_brk"]
LOGIT_C = 0.1
PERIODS = (("f", H.FRONT), ("b", H.BACK), ("a", H.AUX))


@dataclass(frozen=True)
class Atom:
    key: str        # 짧은 이름
    br: str         # 갈래
    fam: str        # 'D'(일봉 사건) | 'b8'(60분봉 사건)
    N: int
    feat: str
    dir: str        # 'hi'(≥ 2/3 분위) | 'lo'(≤ 1/3 분위) | '=0' | '=1'
    kind: str | None
    given: str | None   # 과제가 준 문턱 문자열(대조용). 없으면 None


# 과제가 준 다른 갈래 앞 구간 상위 칸(같은 N에서 쓸 원자)
ATOMS_D = [
    Atom("ret5", "b1", "D", 20, "ret5_pre", "hi", "stock", "-0.02387"),
    Atom("depth", "b1", "D", 20, "depth_atr", "hi", "stock", "0.4892"),
    Atom("gap0", "b1", "D", 40, "gap_break", "=0", "stock", None),
    Atom("gap0", "b1", "D", 60, "gap_break", "=0", "stock", None),
    Atom("cob", "b3", "D", 20, "cobreak_share", "lo", None, "0.0354"),
    Atom("cob", "b3", "D", 40, "cobreak_share", "lo", None, "0.02655"),
    Atom("vix", "b3", "D", 20, "vix", "lo", None, "18.38"),
    Atom("rs20", "b3", "D", 60, "rs20_qqq", "hi", None, "-0.08175"),
    Atom("nvdl", "b4", "D", 20, "nvdl_flow20_p", "hi", None, "0.6426"),
    Atom("pc", "b4", "D", 20, "pc_equity5_p", "hi", None, "0.8447"),
    Atom("dtc", "b4", "D", 20, "si_dtc", "hi", None, "2.83"),
    Atom("shvr", "b4", "D", 60, "shvr_z20", "hi", None, "-0.1192"),
    Atom("gapsz", "b5", "D", 20, "gap_size", "lo", None, "0.01087"),
    Atom("sub0", "b5", "D", 40, "sub_cobreak_n", "lo", None, "0"),
    Atom("noearn", "b5", "D", 60, "earn_pm3", "=0", None, None),
    Atom("soxx0", "b7", "D", 20, "soxx_brk", "=0", None, None),
    Atom("sq0", "b7", "D", 20, "sq_brk", "=0", None, None),
    Atom("qqq0", "b7", "D", 20, "qqq_brk", "=0", None, None),
    Atom("soxx0", "b7", "D", 40, "soxx_brk", "=0", None, None),
]
ATOMS_B8 = [
    Atom("b8depth", "b8", "b8", 28, "depth_atr", "lo", None, "0.3454"),
    Atom("b8dbox", "b8", "b8", 14, "dbox_pos", "hi", None, "0.1937"),
    Atom("b8dbox", "b8", "b8", 7, "dbox_pos", "hi", None, "0.4709"),
]
# b8 사건에 붙이는 시장 원자(전 거래일까지 알려진 값). 문턱은 b8 앞 구간 같은 N 사건 3분위.
MKT_B8 = [("vix_prev", "b3", "lo"), ("nvdl_D", "b4", "hi"), ("pc_D", "b4", "hi"),
          ("sq_brk20_prev", "b7", "=0"), ("soxx_brk20_prev", "b7", "=0"), ("qqq_brk20_prev", "b7", "=0")]
B2_SETS = [("bounce", 60), ("bounce", 40), ("bounce", 20), ("hold2", 20)]

DESC = {
    "ret5_pre": "이탈 전 5일 수익 = 종가(t−1)/종가(t−6) − 1",
    "depth_atr": "이탈 깊이 = (L_N − 종가(t)) ÷ ATR20(t−1)",
    "gap_break": "갭 이탈 표지(시가(t) < L_N이면 1, 장중 이탈이면 0)",
    "cobreak_share": "같은 날 같은 유형(종목/ETF)에서 새 사건이 난 종목 수 ÷ 그날 L_N이 있는 같은 유형 종목 수",
    "vix": "^VIX 종가(t)",
    "rs20_qqq": "QQQ 대비 20일 상대강도 = (C_t/C_{t−20})/(QQQ_t/QQQ_{t−20}) − 1",
    "nvdl_flow20_p": "NVDL 20일 순유입(좌수 변화×NAV 합)의 250일 백분위(진입일 t+1 시가 전 공개분)",
    "pc_equity5_p": "CBOE 주식 풋/콜 5일 평균의 250일 백분위(진입일 t+1 시가 전 공개분)",
    "si_dtc": "가장 최근 공개 FINRA 공매도 잔고 보고의 상환 일수",
    "shvr_z20": "그 종목 FINRA 공매도 거래량 비율의 직전 20개(최소 15개) 대비 z값(D 저녁 공개분)",
    "gap_size": "갭 이탈 크기 = 시가(t)/L_N − 1",
    "sub_cobreak_n": "같은 watchlist 소분류 다른 종목 중 종가(t) < L_N(t)인 종목 수",
    "earn_pm3": "실적 반응일(16시 이후 발표면 다음 거래일)이 t−3..t+3 거래일 안",
    "soxx_brk": "SOXX 종가(t) < SOXX L_N(t)(사건과 같은 N)",
    "qqq_brk": "QQQ 종가(t) < QQQ L_N(t)(사건과 같은 N)",
    "sq_brk": "SOXX 또는 QQQ 종가(t) < 자기 L_N(t)",
    # b8
    "dbox_pos": "일봉 20일 박스(전 거래일까지 최저 저가 Ld20·최고 고가 Hd20) 안 이탈 봉 종가 위치 (C − Ld20)/(Hd20 − Ld20)",
    "b8_depth_atr": "(L − 이탈 봉 종가) ÷ 20봉 ATR(봉 i−1까지)",
    "vix_prev": "^VIX 전 거래일 종가",
    "nvdl_D": "NVDL 20일 순유입 250일 백분위 중 이탈 거래일 시가 전 공개분(b4 배치 규칙)",
    "pc_D": "CBOE 주식 풋/콜 5일 평균 250일 백분위 중 이탈 거래일 시가 전 공개분(b4 배치 규칙)",
    "sq_brk20_prev": "전 거래일 종가 기준 SOXX 또는 QQQ 종가 < 자기 일봉 20일 박스 하단",
    "soxx_brk20_prev": "전 거래일 종가 기준 SOXX 종가 < SOXX 일봉 20일 박스 하단",
    "qqq_brk20_prev": "전 거래일 종가 기준 QQQ 종가 < QQQ 일봉 20일 박스 하단",
}
KNOWN = {
    "ret5_pre": "t−1 종가", "depth_atr": "t 종가", "gap_break": "t 시가", "cobreak_share": "t 종가",
    "vix": "t ^VIX 종가(16:15 ET)", "rs20_qqq": "t 종가", "nvdl_flow20_p": "t 저녁 공개(GraniteShares, 가정)",
    "pc_equity5_p": "t 저녁 공개(CBOE, 가정)", "si_dtc": "가장 최근 공개일 다음 거래일", "shvr_z20": "t 저녁 공개(FINRA)",
    "gap_size": "t 시가", "sub_cobreak_n": "t 종가", "earn_pm3": "t 종가(공시 일정)", "soxx_brk": "t 종가",
    "qqq_brk": "t 종가", "sq_brk": "t 종가",
}


# ───────────────────────────── 자료 ─────────────────────────────
def build_daily():
    """일봉 사건표(N20·40·60 이어 붙임) + 갈래 4·5·7 특징."""
    import b5_catalyst as B5
    import b7_hedge as B7
    P = H.load_panel()
    ev = H.load_events_all()
    b4 = pd.read_parquet(RES / "b4_flow_events.parquet")[
        ["sym", "N", "t", "i", "nvdl_flow20_p", "pc_equity5_p", "shvr_z20", "si_dtc"]]
    b4["t"] = pd.to_datetime(b4["t"])
    ev["t"] = pd.to_datetime(ev["t"])
    ev = ev.merge(b4, on=["sym", "N", "t", "i"], how="left", validate="1:1")
    ev = B5.add_features(ev, P)
    ev = B7.add_features(ev, P)
    return ev.reset_index(drop=True), P


def check_b4_dates(ev, P):
    """b4 날짜 고정 특징이 b4 배치 규칙(진입일 t+1에 놓인 값)과 같은지 다시 계산해 대조."""
    import b4_flow as B4
    dv, dw = B4.date_features(B4.load_sources(), P.cal)
    e = ev["i"].to_numpy() + 1
    out = {}
    # 배치 시점 점검: 달력 k일에 놓인 값의 원자료 공개 시점(when)은 k일보다 앞서야 한다(b8 사건은 k = 이탈 거래일)
    for f in ("nvdl_flow20_p", "pc_equity5_p"):
        w = pd.to_datetime(dw[f])
        has = dv[f].notna().to_numpy()
        out[f"{f}_when_not_before_day"] = int((w[has].to_numpy() >= P.cal.to_numpy()[has]).sum())
    for f in ("nvdl_flow20_p", "pc_equity5_p"):
        a = np.where(e < len(P.cal), dv[f].to_numpy()[np.minimum(e, len(P.cal) - 1)], np.nan)
        b = ev[f].to_numpy()
        out[f] = {"max_abs": float(np.nanmax(np.abs(a - b))) if np.isfinite(a - b).any() else 0.0,
                  "nan_mismatch": int((np.isnan(a) != np.isnan(b)).sum())}
    return out, dv


def build_b8(P, dv):
    """60분봉 사건표(갈래 8 저장본) + 전 거래일까지 알려진 시장 원자."""
    ev8 = pd.concat([pd.read_parquet(RES / f"b8_events_N{N}.parquet") for N in (7, 14, 28)], ignore_index=True)
    ev8["t"] = pd.to_datetime(ev8["t"]).astype("datetime64[ns]")
    j = P.cal.get_indexer(ev8["t"])
    if (j < 1).any():
        raise RuntimeError(f"b8 사건 날짜가 달력에 없음: {int((j < 0).sum())}")
    vix = P.col("C", "^VIX")
    ev8["vix_prev"] = vix[j - 1]
    ev8["nvdl_D"] = dv["nvdl_flow20_p"].to_numpy()[j]
    ev8["pc_D"] = dv["pc_equity5_p"].to_numpy()[j]
    st = {}
    for s in ("SOXX", "QQQ"):
        o, h, l, c = (P.col(f, s) for f in "OHLC")
        Ls, _, _ = H.box_levels(o, h, l, c, 20)
        ok = ~np.isnan(Ls) & ~np.isnan(c)
        st[s] = np.where(ok, (c < np.where(ok, Ls, 0)).astype(float), np.nan)
    sb, qb = st["SOXX"][j - 1], st["QQQ"][j - 1]
    ev8["soxx_brk20_prev"] = sb
    ev8["qqq_brk20_prev"] = qb
    ev8["sq_brk20_prev"] = np.where(np.isnan(sb) | np.isnan(qb), np.nan, np.fmax(sb, qb))
    return ev8.reset_index(drop=True)


# ───────────────────────────── 원자·조합 ─────────────────────────────
def atom_rule(df, feat, N, direction, kind=None, front=H.FRONT):
    """tercile_cells와 같은 규칙으로 (켜짐 Series 1/0/NaN, 문턱 값, 문턱 문자열)."""
    pop = (df["N"] == N) if kind is None else ((df["N"] == N) & (df["kind"] == kind))
    x = df[feat].astype(float)
    fr = H.in_period(df, front) & pop & x.notna()
    xf = x[fr].to_numpy()
    u = np.unique(xf)
    s = pd.Series(np.nan, index=df.index)
    if direction in ("=0", "=1"):
        val = float(direction[1])
        if len(u) > 2:
            raise ValueError(f"{feat} 이진 아님")
        s[pop] = np.where(x[pop].isna(), np.nan, (x[pop] == val).astype(float))
        return s, val, f"={val:g}"
    if len(u) <= 2:
        raise ValueError(f"{feat} N{N} 값이 두 개 이하")
    q1, q2 = np.quantile(xf, [1 / 3, 2 / 3])
    if direction == "hi":
        on, thr, txt = x >= q2, q2, f"≥{q2:.4g}"
    else:
        on, thr, txt = x <= q1, q1, f"≤{q1:.4g}"
    s[pop] = np.where(x[pop].isna(), np.nan, on[pop].astype(float))
    return s, float(thr), txt


def atom_label(a, txt):
    return f"{a.feat}{txt}"


def and_series(series):
    m = np.vstack([s.to_numpy(float) for s in series])
    nan = np.isnan(m).any(0)
    v = np.where(nan, np.nan, np.all(m == 1, axis=0).astype(float))
    return v


def enumerate_cells(evD, ev8):
    """원자 계산 → 조합 칸 목록(정의만, 결과 없음). 반환 (atoms 정보, combos 목록)."""
    ainfo = {}
    for a in ATOMS_D:
        s, thr, txt = atom_rule(evD, a.feat, a.N, a.dir, a.kind)
        ainfo[a] = dict(series=s, thr=thr, txt=txt, label=atom_label(a, txt))
    for a in ATOMS_B8:
        s, thr, txt = atom_rule(ev8, a.feat, a.N, a.dir, a.kind)
        ainfo[a] = dict(series=s, thr=thr, txt=txt, label=atom_label(a, txt))
    mk = {}
    for b8a in ATOMS_B8:
        for f, br, d in MKT_B8:
            m = Atom(f, br, "b8", b8a.N, f, d, None, None)
            s, thr, txt = atom_rule(ev8, f, b8a.N, d)
            ainfo[m] = dict(series=s, thr=thr, txt=txt, label=atom_label(m, txt))
            mk.setdefault(b8a.N, []).append(m)
    combos = []
    # 가족 D
    for N in H.NS:
        A = [a for a in ATOMS_D if a.N == N]
        for r in (2, 3):
            for comb in itertools.combinations(A, r):
                if len({a.br for a in comb}) == r:
                    combos.append(("D", None, N, comb))
    # 가족 b2
    for vb, N in B2_SETS:
        for a in [a for a in ATOMS_D if a.N == N]:
            combos.append((f"b2{vb}", vb, N, (a,)))
    # 가족 b8
    for b8a in ATOMS_B8:
        M = mk[b8a.N]
        for m in M:
            combos.append(("b8", None, b8a.N, (b8a, m)))
        for m1, m2 in itertools.combinations(M, 2):
            if m1.br != m2.br:
                combos.append(("b8", None, b8a.N, (b8a, m1, m2)))
    return ainfo, combos


def combo_name(fam, vb, N, comb, ainfo):
    lab = " & ".join(ainfo[a]["label"] for a in comb)
    if fam == "D":
        k = "stock|" if any(a.kind == "stock" for a in comb) else ""
        return f"D|N{N}|{k}{lab}"
    if fam.startswith("b2"):
        return f"{fam}|N{N}|stock|{lab}"
    return f"b8|N{N}|{lab}"


def combo_definition(fam, vb, N, comb, ainfo):
    parts = []
    for a in comb:
        d = DESC["b8_depth_atr"] if (a.fam == "b8" and a.feat == "depth_atr") else DESC.get(a.feat, a.feat)
        thr = ainfo[a]["thr"]
        cond = ainfo[a]["txt"] if a.dir in ("=0", "=1") else ("≥ " if a.dir == "hi" else "≤ ") + f"{thr:.6g}"
        parts.append(f"[{a.br}] {d} {cond.strip()}")
    cond = " 그리고 ".join(parts)
    if fam == "D":
        pop = "종목" if any(a.kind == "stock" for a in comb) else "종목+ETF"
        return (f"일봉 N={N} 박스 하단 종가 이탈 사건({pop}, 10거래일 연쇄 묶기 첫날) 중 {cond}인 사건이 켜짐(꺼짐 = 같은 모집단 나머지). "
                "t+1 시가 공매도, t+1..t+9 종가 > L_N이면 다음 시가 청산, 아니면 t+10 종가. 비용 0.10%")
    if fam.startswith("b2"):
        from b2_confirm import DEF, EXIT_TXT
        return (f"N={N} 종목 이탈 중 {cond}인 사건에 확인 대기 진입 '{vb}'({DEF[vb][0]}). {EXIT_TXT} "
                "켜짐 = 그 거래의 손익, 꺼짐 = 같은 N 종목 모든 사건의 t+1 시가 즉시 진입 손익(갈래 2와 같은 쌓기)")
    return (f"60분봉 N={N}거래일({N * 7}봉) 박스 하단 이탈(봉 종가 < L, 70봉 연쇄 묶기, 종목+ETF) 중 {cond}인 사건이 켜짐. "
            "다음 봉 시가 공매도, 봉 종가 > L이면 다음 봉 시가 되돌림 손절, 아니면 70봉 뒤 종가. 비용 0.10%")


def combo_known(fam, vb, comb):
    if fam == "D":
        ks = sorted({KNOWN.get(a.feat, "t 종가") for a in comb})
        return "이탈일 t 종가 뒤(" + ", ".join(ks) + ") → t+1 정규장 시가 진입 전"
    if fam.startswith("b2"):
        return f"확인일 d(t+1..t+5) 종가 뒤 진입 d+1 시가. 일봉 특징은 t 종가(또는 t 저녁 공개)라 진입 전"
    return "이탈 봉 종가 뒤 다음 봉 시가 진입. 시장 특징은 전 거래일 종가 또는 이탈일 시가 전 공개분, b8 특징은 이탈 봉 종가"


def cell_list_hash(names):
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


# ───────────────────────────── 계획 ─────────────────────────────
def cmd_plan():
    evD, P = build_daily()
    chk, dv = check_b4_dates(evD, P)
    ev8 = build_b8(P, dv)
    ainfo, combos = enumerate_cells(evD, ev8)
    # 하네스 tercile_cells와 같은지(일봉 원자) 대조
    tc_bad = []
    for a in ATOMS_D:
        if a.kind == "stock":
            tc = dict(H.tercile_cells(evD, [a.feat], kind="stock", tag="stock|"))
            pre = f"stock|N{a.N}|{a.feat}"
        else:
            tc = dict(H.tercile_cells(evD, [a.feat]))
            pre = f"N{a.N}|{a.feat}"
        want = {"hi": "≥", "lo": "≤", "=0": "=0", "=1": "=1"}[a.dir]
        nm = [k for k in tc if k.startswith(pre + want)]
        if len(nm) != 1:
            tc_bad.append((a.key, a.N, "칸 못 찾음"))
            continue
        s = tc[nm[0]]
        x, y = s.to_numpy(float), ainfo[a]["series"].to_numpy(float)
        if not np.array_equal(np.isnan(x), np.isnan(y)) or not np.array_equal(x[~np.isnan(x)], y[~np.isnan(y)]):
            tc_bad.append((a.key, a.N, "값 다름"))
    given_bad = [(a.key, a.N, ainfo[a]["txt"], a.given) for a in ATOMS_D + ATOMS_B8
                 if a.given is not None and ainfo[a]["txt"][1:] != a.given]
    names = [combo_name(*c, ainfo) for c in combos]
    if len(set(names)) != len(names):
        raise RuntimeError("칸 이름 중복")
    lg = [f"D|N{N}|logit_top3분위" for N in H.NS]
    all_names = names + lg
    fam_count = pd.Series([c[0] for c in combos]).value_counts().to_dict()
    size_count = pd.Series([f"{c[0][:2]}:{len(c[3])}" for c in combos]).value_counts().to_dict()
    plan = {"cells": all_names, "hash": cell_list_hash(all_names), "n_combo": len(names), "n_logit": len(lg),
            "family_count": fam_count, "size_count": size_count,
            "atoms": [{"fam": a.fam, "br": a.br, "N": a.N, "feat": a.feat, "dir": a.dir, "kind": a.kind,
                       "thr": ainfo[a]["thr"], "txt": ainfo[a]["txt"], "given": a.given} for a in ainfo],
            "tercile_check_bad": tc_bad, "given_threshold_bad": given_bad, "b4_date_check": chk}
    PLAN_JSON.write_text(json.dumps(plan, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    write_plan_md(plan, combos, ainfo, names, lg)
    print(json.dumps({k: plan[k] for k in ("n_combo", "n_logit", "family_count", "size_count", "tercile_check_bad",
                                           "given_threshold_bad", "b4_date_check", "hash")}, ensure_ascii=False,
                     indent=1, default=str))


def write_plan_md(plan, combos, ainfo, names, lg):
    L = ["# 갈래 6 조합 — 사전 계획 (2026-10-14, 계산 전 작성)", "",
         "이 파일은 `code/b6_combo.py plan`이 특징·문턱만 계산해서 썼다. 결과(손익·분류)는 아직 보지 않았다. "
         "`run`은 이 목록과 칸 이름·순서가 같은지(sha256) 확인한 뒤에만 판정한다.", "",
         "## 계획 전에 이미 본 것(공개)", "",
         "- 과제가 준 다른 갈래 앞 구간 상위 칸의 정의와 앞 구간 차이.",
         "- `code/b2_confirm.py`를 읽다가 보고서 문구 틀(결과 뒤 작성)에서 '확인 대기 변형이 뒤 구간에서 거래당 손실'이라는 결론 문장을 보았다.",
         "- `results/verify_b4_flow_timing.json` 구조를 확인하다가 갈래 4의 `N20|gvip_rel20_p≥0.5382` 칸이 뒤 구간에서 통과했다는 값을 보았다. "
         "이 특징은 과제가 준 상위 칸 목록에 없고, 이 갈래에서도 쓰지 않는다.",
         "- 다른 갈래의 결과 보고서(.md)와 칸 표(.csv)는 열지 않았다.", "",
         "## 원자(다른 갈래 앞 구간 상위 칸)", "",
         "문턱은 같은 규칙(앞 구간 같은 N 사건의 1/3·2/3 분위, 갈래 1은 같은 kind도)으로 다시 계산했다. "
         f"과제가 준 문턱 문자열과 다른 원자: {plan['given_threshold_bad'] or '없음'}. "
         f"하네스 `tercile_cells`와 켜짐 표지가 다른 일봉 원자: {plan['tercile_check_bad'] or '없음'}. "
         f"b4 날짜 특징 재계산 대조(배치 규칙): {plan['b4_date_check']}.", "",
         "| 가족 | 갈래 | N | 특징 | 방향 | 문턱(정밀) | 모집단 | 언제 알 수 있나 |", "|---|---|---:|---|---|---:|---|---|"]
    for a, inf in ainfo.items():
        if a.fam == "b8":
            known = "이탈 봉 종가(봉 i−1까지 ATR, 일봉 박스는 전 거래일까지)" if a.br == "b8" else \
                "이탈 거래일 시가 전(전 거래일 종가 또는 그 전 저녁 공개분)"
        else:
            known = KNOWN.get(a.feat, "t 종가")
        L.append(f"| {a.fam} | {a.br} | {a.N} | {a.feat} | {a.dir} | {inf['thr']:.6g} | {a.kind or '종목+ETF'} | {known} |")
    L += ["", "## 조합 규칙", "",
          "- 같은 N 안에서만, 서로 다른 갈래 원자 2개 또는 3개의 AND(같은 갈래 원자끼리는 묶지 않는다). "
          "모집단 = 모든 원자가 정의된 사건(갈래 1 원자가 끼면 종목만). 켜짐 = 모든 원자 켜짐, 꺼짐 = 모집단 나머지.",
          "- 가족 D(일봉 사건, 결과 = 하네스 short_ret_h10, 초과 = excess_h10): N20 원자 11개(b1 2, b3 2, b4 3, b5 1, b7 3), "
          "N40 4개(b1·b3·b5·b7), N60 4개(b1·b3·b4·b5).",
          "- 가족 b2(확인 대기 진입, 종목): bounce N60·N40·N20, hold2 N20 × 같은 N 일봉 원자 1개. 갈래 2와 같은 쌓기 — "
          "켜짐 행 = 변형이 진입했고 원자가 켜진 사건의 변형 손익(거래당), 꺼짐 행 = 같은 N 종목 모든 사건의 t+1 시가 즉시 진입 손익. "
          "확인 대기 변형은 진입 규칙이라 2개 조합(변형 + 원자 1개)만 만든다(3개면 106칸으로 상한 300을 넘김).",
          "- 가족 b8(60분봉 사건, 갈래 8 저장 사건표, 결과 = 70봉 short_ret_h10): b8 원자 3개 × 시장 원자 6개(갈래 3 VIX, 갈래 4 NVDL·풋/콜, "
          "갈래 7 SOXX·QQQ·둘 중 하나 일봉 20일 박스 이탈 상태). 시장 원자는 모두 이탈 거래일 시가 전에 알려진 값"
          "(VIX·박스 상태는 전 거래일 종가, b4 값은 그날 시가 전 공개분)이고 문턱은 b8 앞 구간 같은 N 사건 3분위다. "
          "갈래 7 원자는 일봉 사건과 같은 '사건과 같은 N' 대신 갈래 7 상위 칸의 N20 일봉 박스를 쓴다(60분봉 N과 일봉 N이 다르다).",
          "- 쓰지 않는 것: b8 N14 tod_close=0(앞 구간 꺼짐 1건뿐인 퇴화 칸), b5 sub_cobreak_share ≤ 0(sub_cobreak_n ≤ 0과 같은 사건 집합), "
          "과제 목록 밖 특징.", "",
          "## 로지스틱(1개)", "",
          f"- 학습: 가족 D 앞 구간(2026-01-02~05-29) 사건 전체(N20·40·60, 종목+ETF; 특징 8개가 모두 있는 사건). "
          "목표 = 하네스 분류 '진짜'(t+1..t+10 안에 L 위 종가 전에 저가 ≤ L − 2·ATR20)면 1, 가짜·중간이면 0. "
          "앞 구간 끝 사건의 분류는 6월 중순까지 정해진다(하네스 앞 구간 고르기와 같은 처리).",
          f"- 특징 8개: {', '.join(LOGIT_FEATS)}(갈래 1 두 개, 갈래 3 두 개, 갈래 4 두 개, 갈래 5 하나, 갈래 7 하나). "
          "모두 t 종가(또는 t 저녁 공개) 값. 학습 사건 평균·표준편차로 표준화, sklearn LogisticRegression(L2, C = 0.1 고정, lbfgs).",
          "- 칸: N마다 그 N 앞 구간 사건(종목+ETF)의 예측확률 2/3 분위 이상 = 켜짐, 나머지 = 꺼짐(특징 결측은 모집단 밖). 3칸.", "",
          "## 판정(하네스 그대로)", "",
          f"- 칸 {len(names) + len(lg)}개(조합 {len(names)} + 로지스틱 {len(lg)})를 쌓은 표 하나에 넣고 `evaluate_cells` 한 번: "
          "앞 구간 차이(켜짐 − 꺼짐 short_ret_h10) 상위 10칸(켜짐 ≥ 15건, 서로 다른 날짜 ≥ 8일), 뒤 구간 같은 방향·날짜 묶음 t ≥ 2·"
          "날짜 묶음 순열 200번 p < 0.05·켜짐 평균 > 0이면 통과. 선택편향 귀무 200번, 층 = 가족·N·kind, 한 국면 비중 > 50% 표시.",
          f"- 가족별 칸 수: {plan['family_count']}, 가족:원자 수별: {plan['size_count']}.",
          f"- 칸 목록 sha256: `{plan['hash']}`", "",
          "## 칸 목록(순서 = 판정 입력 순서)", ""]
    for k, nm in enumerate(names + lg, 1):
        L.append(f"{k}. `{nm}`")
    L.append("")
    PLAN_MD.write_text("\n".join(L), encoding="utf-8")


# ───────────────────────────── 실행 ─────────────────────────────
def logistic(evD):
    from sklearn.linear_model import LogisticRegression
    X = evD[LOGIT_FEATS].astype(float)
    okx = X.notna().all(axis=1)
    tr = okx & H.in_period(evD, H.FRONT) & evD["cls"].notna()
    yb = (evD.loc[tr, "cls"] == "진짜").astype(int).to_numpy()
    mu, sd = X[tr].mean(), X[tr].std(ddof=0)
    Z = (X - mu) / sd
    m = LogisticRegression(C=LOGIT_C, solver="lbfgs", max_iter=5000)   # 기본 규제 = L2
    m.fit(Z[tr].to_numpy(), yb)
    p = pd.Series(np.nan, index=evD.index)
    p[okx] = m.predict_proba(Z[okx].to_numpy())[:, 1]
    info = {"n_train": int(tr.sum()), "pos_rate": float(yb.mean()), "intercept": float(m.intercept_[0]),
            "coef": dict(zip(LOGIT_FEATS, map(float, m.coef_[0]))), "mean": mu.to_dict(), "sd": sd.to_dict(),
            "train_dates": int(evD.loc[tr, "t"].nunique())}
    cells, thr = {}, {}
    for N in H.NS:
        pop = (evD["N"] == N) & p.notna()
        q2 = float(np.quantile(p[pop & H.in_period(evD, H.FRONT)], 2 / 3))
        s = pd.Series(np.nan, index=evD.index)
        s[pop] = (p[pop] >= q2).astype(float)
        cells[N], thr[N] = s, q2
    info["thr"] = thr
    # 설명용(사후): 구간별 예측확률과 실제 '진짜' 비율의 순위 상관
    rk = {}
    for pk, per in PERIODS:
        mm = H.in_period(evD, per) & p.notna() & evD["cls"].notna()
        if mm.sum() > 10:
            rk[pk] = float(pd.Series(p[mm]).corr((evD.loc[mm, "cls"] == "진짜").astype(float), method="spearman"))
    info["spearman_p_vs_true"] = rk
    return p, cells, info


def build_stack(evD, ev8, P):
    import b2_confirm as B2
    evB = B2.compute(evD, P)
    keep = ["sym", "kind", "N", "t", "i", "episode", "cls"]
    parts = [evD[keep].assign(fam="D", role="ev", key=evD.index, y=evD["short_ret_h10"], y_ex=evD["excess_h10"])]
    for vb, N in B2_SETS:
        base = evB[(evB["kind"] == "stock") & (evB["N"] == N) & evB["short_ret_h10"].notna()]
        imm = base[keep].assign(fam=f"b2{vb}", role="imm", key=base.index, y=base["short_ret_h10"], y_ex=np.nan)
        tr = base[base[f"{vb}_status"] == "trig"]
        var = tr[keep].assign(fam=f"b2{vb}", role="var", key=tr.index, y=tr[f"{vb}_y"], y_ex=np.nan)
        parts += [imm, var]
    parts.append(ev8[keep].assign(fam="b8", role="ev", key=ev8.index, y=ev8["short_ret_h10"], y_ex=ev8["excess_h10"]))
    S = pd.concat(parts, ignore_index=True)
    S["t"] = pd.to_datetime(S["t"]).astype("datetime64[ns]")
    per = np.zeros(len(S), bool)
    for _, p in PERIODS:
        per |= H.in_period(S, p).to_numpy()
    S = S[per].reset_index(drop=True)
    S["episode"] = S["episode"].astype(int)
    return S, evB


def stack_cells(S, combos, ainfo, logit_cells):
    cells, meta = [], []
    famv, rolev, keyv, Nv = S["fam"].to_numpy(), S["role"].to_numpy(), S["key"].to_numpy(), S["N"].to_numpy()
    for fam, vb, N, comb in combos:
        v = and_series([ainfo[a]["series"] for a in comb])     # 사건표 색인(D: evD, b8: ev8)
        x = np.full(len(S), np.nan)
        if fam == "D":
            m = famv == "D"
            x[m] = v[keyv[m]]
        elif fam == "b8":
            m = famv == "b8"
            x[m] = v[keyv[m]]
        else:
            m = (famv == fam) & (Nv == N)
            mv = m & (rolev == "var")
            vv = v[keyv[mv]]
            x[mv] = np.where(vv == 1, 1.0, np.nan)
            x[m & (rolev == "imm")] = 0.0
        nm = combo_name(fam, vb, N, comb, ainfo)
        cells.append((nm, pd.Series(x, index=S.index)))
        meta.append({"cell": nm, "family": fam, "N": N, "n_atoms": len(comb) + (1 if fam.startswith("b2") else 0),
                     "branches": "+".join(([f"b2:{vb}"] if vb else []) + [a.br for a in comb]),
                     "definition": combo_definition(fam, vb, N, comb, ainfo),
                     "known_when": combo_known(fam, vb, comb)})
    for N in H.NS:
        x = np.full(len(S), np.nan)
        m = famv == "D"
        x[m] = logit_cells[N].to_numpy()[keyv[m]]
        nm = f"D|N{N}|logit_top3분위"
        cells.append((nm, pd.Series(x, index=S.index)))
        meta.append({"cell": nm, "family": "D", "N": N, "n_atoms": len(LOGIT_FEATS), "branches": "logit(b1,b3,b4,b5,b7)",
                     "definition": "", "known_when": "이탈일 t 종가 뒤(특징 8개 모두 t 종가 또는 t 저녁 공개) → t+1 시가 진입 전. "
                                                     "모형 계수는 앞 구간(1~5월) 사건으로 고정"})
    return cells, pd.DataFrame(meta)


def describe_on(S, cells, tab, sel_idx):
    """설명용: 선택 칸의 구간별 켜진 사건 중 '진짜'·'가짜' 비율, 되돌림 없이 이어진 비율(판정 아님)."""
    rows = []
    for j in sel_idx:
        nm, s = cells[j]
        x = s.to_numpy(float)
        for pk, p in PERIODS:
            m = H.in_period(S, p).to_numpy() & S["y"].notna().to_numpy()
            on = m & (x == 1)
            off = m & (x == 0)
            c_on, c_off = S.loc[on, "cls"].dropna(), S.loc[off, "cls"].dropna()
            rows.append({"cell": nm, "period": pk, "n_on": int(on.sum()), "dates_on": int(S.loc[on, "t"].nunique()),
                         "true_on": float((c_on == "진짜").mean()) if len(c_on) else np.nan,
                         "fake_on": float((c_on == "가짜").mean()) if len(c_on) else np.nan,
                         "true_off": float((c_off == "진짜").mean()) if len(c_off) else np.nan,
                         "fake_off": float((c_off == "가짜").mean()) if len(c_off) else np.nan,
                         "win_on": float((S.loc[on, "y"] > 0).mean()) if on.any() else np.nan})
    return pd.DataFrame(rows)


def cmd_run():
    plan = json.loads(PLAN_JSON.read_text(encoding="utf-8"))
    evD, P = build_daily()
    chk, dv = check_b4_dates(evD, P)
    ev8 = build_b8(P, dv)
    ainfo, combos = enumerate_cells(evD, ev8)
    names = [combo_name(*c, ainfo) for c in combos] + [f"D|N{N}|logit_top3분위" for N in H.NS]
    if cell_list_hash(names) != plan["hash"]:
        raise RuntimeError("계획과 칸 목록이 다르다 — 중단")
    p, lcells, linfo = logistic(evD)
    S, evB = build_stack(evD, ev8, P)
    cells, meta = stack_cells(S, combos, ainfo, lcells)
    assert [c[0] for c in cells] == names
    lg_def = ("로지스틱 1개(앞 구간 일봉 사건으로 학습, 목표 '진짜', 특징 " + ", ".join(LOGIT_FEATS)
              + ", 표준화, L2 C=0.1) 예측확률 ≥ {thr:.6g}(그 N 앞 구간 2/3 분위)인 N={N} 사건(종목+ETF)이 켜짐. "
              "t+1 시가 공매도, 되돌림 손절 또는 t+10 종가, 비용 0.10%")
    for N in H.NS:
        meta.loc[meta["cell"] == f"D|N{N}|logit_top3분위", "definition"] = lg_def.format(thr=linfo["thr"][N], N=N)
    tab, summ = H.evaluate_cells(S, cells, y="y", y_ex="y_ex", strata=("fam", "N", "kind"))
    out = meta.merge(tab, on="cell", how="left")
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    sel = out[out["selected"]].sort_values("rank")
    sel_idx = [names.index(c) for c in sel["cell"]]
    desc = describe_on(S, cells, tab, sel_idx)
    # 사건·날짜 수(가족 D = 고유 일봉 사건, 판정 표본 = y 있음)
    cnt = {}
    for fam in ("D", "b2bounce", "b2hold2", "b8"):
        for pk, per in PERIODS:
            m = (S["fam"] == fam) & H.in_period(S, per) & S["y"].notna()
            if fam.startswith("b2"):
                m &= S["role"] == "imm"
            cnt[f"{fam}_{pk}"] = {"events": int(m.sum()), "dates": int(S.loc[m, "t"].nunique())}
    sj = {k: v for k, v in summ.items()}
    for nn in ("null", "null_shift"):
        if nn in sj:
            sj[nn] = {a: b for a, b in sj[nn].items() if a != "dist"}
    sj.update({"counts": cnt, "logit": linfo, "plan_hash": plan["hash"], "b4_date_check": chk,
               "family_cells": meta["family"].value_counts().to_dict()})
    OUT_JSON.write_text(json.dumps(sj, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    ex = extras(S, cells, out, names, p)
    sj["extras"] = {k: v for k, v in ex.items() if k != "fam_table"}
    sj["extras"]["fam_table"] = ex["fam_table"]
    OUT_JSON.write_text(json.dumps(sj, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    write_md(out, sj, desc, linfo, S, ex)
    cols = ["rank", "cell", "f_n_on", "f_dates_on", "f_diff", "b_n_on", "b_dates_on", "b_diff", "b_t", "b_mean_on",
            "b_p", "b_p_shift", "a_diff", "regime_top_share", "pass"]
    with pd.option_context("display.width", 260, "display.max_columns", 30, "display.max_colwidth", 70):
        print(sel[cols].to_string())
        lgc = out[out["cell"].str.contains("logit")]
        print(lgc[cols].to_string())
    print(json.dumps({k: sj[k] for k in ("n_cells", "n_eligible", "n_selected", "n_pass", "passed", "null",
                                         "null_shift", "counts")}, ensure_ascii=False, indent=1, default=str))
    print(json.dumps(linfo, ensure_ascii=False, indent=1, default=str))


def extras(S, cells, out, names, p):
    """설명용 집계(판정 아님): 선택 칸 안 같은 사건 집합 묶음, 가족별 최고 순위, 로지스틱 켜짐의 '진짜' 비율."""
    sel = out[out["selected"]].sort_values("rank")
    yok = S["y"].notna().to_numpy()
    sig = {}
    for pk, per in PERIODS[:2]:
        m = H.in_period(S, per).to_numpy() & yok
        groups = {}
        for r in sel.itertuples():
            x = cells[names.index(r.cell)][1].to_numpy(float)
            key = (tuple(np.flatnonzero(m & (x == 1))), tuple(np.flatnonzero(m & (x == 0))))
            groups.setdefault(key, []).append(int(r.rank))
        sig[pk] = sorted(groups.values())
    el = out[out["eligible"]].copy()
    el["front_rank"] = el["f_diff"].rank(ascending=False, method="min")
    fam_rows = []
    for fam, nm in (("D", "일봉 조합"), ("b2", "확인 대기 × 원자"), ("b8", "60분봉 조합"), ("logit", "로지스틱")):
        if fam == "logit":
            g, e = out[out["cell"].str.contains("logit")], el[el["cell"].str.contains("logit")]
        else:
            g = out[out["family"].str.startswith(fam) & ~out["cell"].str.contains("logit")]
            e = el[el["family"].str.startswith(fam) & ~el["cell"].str.contains("logit")]
        b = e.sort_values("f_diff", ascending=False).head(1)
        fam_rows.append({"fam": nm, "cells": len(g), "eligible": len(e),
                         "best_rank": int(b["front_rank"].iloc[0]) if len(b) else None,
                         "best_cell": b["cell"].iloc[0] if len(b) else None,
                         "best_f_diff": float(b["f_diff"].iloc[0]) if len(b) else np.nan})
    lg = {}
    for N in H.NS:
        x = cells[names.index(f"D|N{N}|logit_top3분위")][1].to_numpy(float)
        for pk, per in PERIODS:
            m = H.in_period(S, per).to_numpy() & yok & S["cls"].notna().to_numpy()
            on, off = m & (x == 1), m & (x == 0)
            lg[f"N{N}_{pk}"] = {"true_on": float((S.loc[on, "cls"] == "진짜").mean()) if on.any() else np.nan,
                                "true_off": float((S.loc[off, "cls"] == "진짜").mean()) if off.any() else np.nan,
                                "n_on": int(on.sum()), "n_off": int(off.sum()),
                                # 켜진 사건의 분류별 평균 공매도 손익과 건수(사후 분류, 설명용)
                                "on_by_cls": {k: (float(S.loc[on & (S["cls"] == k).to_numpy(), "y"].mean())
                                                  if (on & (S["cls"] == k).to_numpy()).any() else np.nan,
                                                  int((on & (S["cls"] == k).to_numpy()).sum()))
                                              for k in ("진짜", "가짜", "중간")}}
    return {"same_sets": sig, "fam_table": fam_rows, "logit_true": lg,
            "sel_back_pos": int((sel["b_mean_on"] > 0).sum()), "sel_back_diff_pos": int((sel["b_diff"] > 0).sum()),
            "sel_back_t2": int((sel["b_t"] >= 2).sum()),
            "sel_back_off_range": (float(sel["b_mean_off"].min()), float(sel["b_mean_off"].max()))}


# ───────────────────────────── 보고서 ─────────────────────────────
def _f(x, pct=True, nd=2, sign=True):
    if x is None or (isinstance(x, (float, np.floating)) and not np.isfinite(x)):
        return "못 구함"
    if pct:
        return f"{x * 100:+.{nd}f}%" if sign else f"{x * 100:.{nd}f}%"
    return f"{x:.{nd}f}"


def write_md(out, sj, desc, linfo, S, ex):
    nul, nsh = sj.get("null", {}), sj.get("null_shift", {})
    sel = out[out["selected"]].sort_values("rank")
    c = sj["counts"]
    fc = sj["family_cells"]
    top = sel.iloc[0]
    lgt = ex["logit_true"]
    ss = ex["same_sets"]
    sp = linfo["spearman_p_vs_true"]
    L = ["# 갈래 6 — 조합 (b6_combo)", "",
         "## 결론", "",
         f"- **다른 갈래 상위 특징을 2~3개 묶어도, 로지스틱으로 합쳐도 뒤 구간(6~9월)에서 '진짜 하락'을 골라 공매도로 번 칸은 없다"
         f"(통과 0/{sj['n_cells']}).** 선택 10칸 중 뒤 구간 켜진 사건 평균이 플러스인 칸은 {ex['sel_back_pos']}개, "
         f"날짜 묶음 t ≥ 2는 {ex['sel_back_t2']}개다.",
         f"- 뒤 구간에서 크게 이긴 1·2위(`{top['cell']}` 외 1칸, 같은 사건 집합)는 뒤 켜짐이 {int(top['b_n_on'])}건/"
         f"{int(top['b_dates_on'])}일뿐이고 한 국면 비중 {_f(top['regime_top_share'], nd=0, sign=False)}, 순열 p "
         f"{_f(top['b_p'], False, 3)}라 우연과 구별되지 않는다(뒤 켜짐 평균 {_f(top['b_mean_on'])}, 2025 차이 {_f(top['a_diff'])}).",
         f"- 상위 10칸은 사실상 몇 개 규칙의 반복이다. 앞 구간 켜짐·꺼짐 사건이 완전히 같은 묶음(순위): {ss['f']}. "
         "앞 구간에서는 갈래 7 원자(SOXX·QQQ·둘 중 하나가 박스 안)가 다른 두 원자가 이미 고른 사건을 하나도 빼지 않아 칸 이름만 다르고 사건은 같았다.",
         f"- 로지스틱은 '진짜'(L − 2·ATR 도달) 자체는 뒤 구간에서도 더 자주 골랐다(켜짐/꺼짐 '진짜' 비율 N20 "
         f"{_f(lgt['N20_b']['true_on'], nd=0, sign=False)}/{_f(lgt['N20_b']['true_off'], nd=0, sign=False)}, N40 "
         f"{_f(lgt['N40_b']['true_on'], nd=0, sign=False)}/{_f(lgt['N40_b']['true_off'], nd=0, sign=False)}, N60 "
         f"{_f(lgt['N60_b']['true_on'], nd=0, sign=False)}/{_f(lgt['N60_b']['true_off'], nd=0, sign=False)}; 예측확률 순위 상관 앞 "
         f"{sp.get('f', float('nan')):+.2f} → 뒤 {sp.get('b', float('nan')):+.2f} → 2025 {sp.get('a', float('nan')):+.2f}). "
         "그래도 켜진 사건 공매도는 세 N 모두 뒤 구간 평균 손실이다. 켜진 사건의 3분의 2 이상이 '가짜·중간'이고 그쪽 손실이 '진짜'의 이익보다 컸다"
         f"(N20 뒤 켜짐: 진짜 {lgt['N20_b']['on_by_cls']['진짜'][1]}건 {_f(lgt['N20_b']['on_by_cls']['진짜'][0])}, 가짜 "
         f"{lgt['N20_b']['on_by_cls']['가짜'][1]}건 {_f(lgt['N20_b']['on_by_cls']['가짜'][0])}, 중간 {lgt['N20_b']['on_by_cls']['중간'][1]}건 "
         f"{_f(lgt['N20_b']['on_by_cls']['중간'][0])}). 앞 구간 차이로도 10위 안에 못 들어 판정 대상이 아니었다(아래 표).",
         f"- 뒤 구간은 이탈 뒤 되돌림이 우세한 장이다(선택 칸들의 꺼진 사건 평균 {_f(ex['sel_back_off_range'][0])} ~ "
         f"{_f(ex['sel_back_off_range'][1])}). 조합으로 '덜 지는' 사건은 골라도(뒤 차이 > 0: {ex['sel_back_diff_pos']}/10칸) "
         "공매도 자체가 돈이 되는 사건 묶음은 찾지 못했다.",
         "", "## 범위", "",
         f"- 칸 {sj['n_cells']}개 = 조합 {sj['n_cells'] - 3}(일봉 D {fc.get('D', 0) - 3}, 확인 대기 b2 "
         f"{fc.get('b2bounce', 0) + fc.get('b2hold2', 0)}, 60분봉 b8 {fc.get('b8', 0)}) + 로지스틱 3. 사전 계획 `b6_combo_plan.md`"
         f"(sha256 `{sj['plan_hash'][:12]}…`)과 같은 목록을 그대로 판정했다.",
         f"- 고르기 조건(앞 켜짐 ≥ 15건·8날짜, 앞 차이 > 0) 충족 {sj['n_eligible']}칸, 선택 {sj['n_selected']}칸, "
         f"**통과 {sj['n_pass']}칸**{(' — ' + ', '.join(sj['passed'])) if sj['passed'] else ''}.",
         f"- 선택편향 귀무(날짜 묶음 i.i.d. 200회): 통과 칸 수 평균 {nul.get('mean', float('nan')):.3f}, 95% 분위 "
         f"{nul.get('q95', float('nan')):g}, 관측 {sj['n_pass']}칸의 p = {nul.get('p_value', float('nan')):.3f}. "
         f"보조 원형 이동 귀무: 평균 {nsh.get('mean', float('nan')):.3f}, p = {nsh.get('p_value', float('nan')):.3f}.",
         f"- 일봉 사건(가족 D, 종목+ETF, N 세 개 합): 앞 {c['D_f']['events']}건/{c['D_f']['dates']}일, 뒤 {c['D_b']['events']}건/"
         f"{c['D_b']['dates']}일, 2025 {c['D_a']['events']}건/{c['D_a']['dates']}일. 60분봉(b8): 앞 {c['b8_f']['events']}건/"
         f"{c['b8_f']['dates']}일, 뒤 {c['b8_b']['events']}건/{c['b8_b']['dates']}일.",
         "", "| 가족 | 칸 | 고르기 조건 충족 | 그 가족 최고 앞 순위(충족 칸 중) | 최고 칸 | 앞 차이 |",
         "|---|---:|---:|---:|---|---:|"]
    for r in ex["fam_table"]:
        L.append(f"| {r['fam']} | {r['cells']} | {r['eligible']} | {r['best_rank'] if r['best_rank'] is not None else '못 구함'} | "
                 f"`{r['best_cell']}` | {_f(r['best_f_diff'])} |")
    L += ["", "## 상위 10칸 판정(앞 구간 차이 순)", "",
         "| 순위 | 칸 | 앞 켜짐(건/날짜) | 앞 차이 | 앞 켜짐 평균 | 뒤 켜짐(건/날짜) | 뒤 차이 | 뒤 t | 뒤 p | 원형 p | 뒤 켜짐 평균 | 2025 차이 | 국면 최대 | 통과 |",
         "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in sel.assign(passed=sel["pass"]).itertuples():
        L.append(f"| {int(r.rank)} | `{r.cell}` | {int(r.f_n_on)}/{int(r.f_dates_on)} | {_f(r.f_diff)} | {_f(r.f_mean_on)} | "
                 f"{int(r.b_n_on)}/{int(r.b_dates_on)} | {_f(r.b_diff)} | {_f(r.b_t, False)} | {_f(r.b_p, False, 3)} | "
                 f"{_f(r.b_p_shift, False, 3)} | {_f(r.b_mean_on)} | {_f(r.a_diff)} | {_f(r.regime_top_share, nd=0, sign=False)} | "
                 f"{'통과' if r.passed else '탈락'} |")
    L += ["", "## 상위 10칸 정의", ""]
    for r in sel.itertuples():
        L.append(f"{int(r.rank)}. `{r.cell}` — {r.definition} 알 수 있는 때: {r.known_when}.")
    L += ["", "## 로지스틱(1개, 앞 구간 학습)", "",
          f"- 학습 {linfo['n_train']}건/{linfo['train_dates']}일, '진짜' 비율 {_f(linfo['pos_rate'], nd=1, sign=False)}. "
          f"절편 {linfo['intercept']:.3f}. 표준화 계수: " + ", ".join(f"{k} {v:+.3f}" for k, v in linfo["coef"].items()) + ".",
          "- 예측확률과 실제 '진짜'의 순위 상관(설명용): " + ", ".join(f"{k} {v:+.3f}" for k, v in linfo["spearman_p_vs_true"].items())
          + " (f = 앞·학습 구간, b = 뒤, a = 2025).", "",
          "| 칸 | 문턱 | 앞 켜짐 | 앞 차이 | 선택 | 뒤 켜짐 | 뒤 차이 | 뒤 t | 뒤 켜짐 평균 | 2025 차이 |", "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|"]
    for r in out[out["cell"].str.contains("logit")].itertuples():
        N = int(r.N)
        L.append(f"| `{r.cell}` | {linfo['thr'][N]:.4f} | {int(r.f_n_on)}/{int(r.f_dates_on)} | {_f(r.f_diff)} | "
                 f"{'선택' if r.selected else '-'} | {int(r.b_n_on)}/{int(r.b_dates_on)} | {_f(r.b_diff)} | {_f(r.b_t, False)} | "
                 f"{_f(r.b_mean_on)} | {_f(r.a_diff)} |")
    L += ["", "## 선택 칸의 켜진 사건 분류(설명용, 사후 분류)", "",
          "| 칸 | 구간 | 켜짐 건/날짜 | 진짜 켜짐/꺼짐 | 가짜 켜짐/꺼짐 | 켜짐 승률 |", "|---|---|---:|---:|---:|---:|"]
    pn = {"f": "앞", "b": "뒤", "a": "2025"}
    for r in desc.itertuples():
        L.append(f"| `{r.cell}` | {pn[r.period]} | {r.n_on}/{r.dates_on} | {_f(r.true_on, nd=0, sign=False)}/{_f(r.true_off, nd=0, sign=False)} | "
                 f"{_f(r.fake_on, nd=0, sign=False)}/{_f(r.fake_off, nd=0, sign=False)} | {_f(r.win_on, nd=0, sign=False)} |")
    pn2 = {"f": "앞", "b": "뒤", "a": "2025"}
    lt = "; ".join(f"N{N} " + ", ".join(f"{pn2[pk]} {_f(lgt[f'N{N}_{pk}']['true_on'], nd=0, sign=False)}/"
                                         f"{_f(lgt[f'N{N}_{pk}']['true_off'], nd=0, sign=False)}" for pk in ("f", "b", "a"))
                   for N in H.NS)
    L += ["", f"- 뒤 구간 켜짐·꺼짐 사건이 완전히 같은 묶음(순위): {ss['b']}.",
          f"- 로지스틱 켜짐의 '진짜' 비율(켜짐/꺼짐, 설명용): {lt}.",
          "", "## 한계", "",
          "- 탐색 범위: 원자 22개(일봉 19 = 과제가 준 칸의 N별 원자, 60분봉 3) + 60분봉용 시장 원자 6종 × N 3, 조합 245칸(일봉 2개 59·3개 105, "
          "확인 대기 × 원자 30, 60분봉 2개 18·3개 33) + 로지스틱 3칸 = 248칸을 한 번 판정했다. 원자 자체가 이미 갈래 1~5·7·8에서 각자 수십~백여 칸 중 "
          "앞 구간 상위로 뽑힌 것이라 선택편향이 겹겹이 쌓여 있다(귀무는 이 갈래 248칸 안의 선택만 반영한다).",
          f"- 조합은 같은 N 안에서만 만들었다. 3개 조합은 앞 구간 켜짐이 15건·8날짜에 못 미친 칸이 많다(충족 {sj['n_eligible']}/{sj['n_cells']}).",
          "- 확인 대기 가족의 꺼짐은 갈래 2와 같이 '같은 N 종목 모든 사건의 즉시 진입'이다. 그래서 차이에 원자 효과와 진입 변형 효과가 함께 들어 있다.",
          "- 60분봉 가족의 갈래 7 원자는 일봉 20일 박스 상태(전 거래일 종가)로 바꿔 쟀다(이탈 봉 시각의 ETF 60분봉 박스가 아님).",
          "- 종목 사건 손익은 공매도 대리 수익(−주가 수익, 비용 0.10%, 대차 비용·배당 없음)이다. 실제 실행은 단일종목 인버스가 있는 일부 종목만 된다.",
          "- 로지스틱 앞 구간 끝(5월 중순~말) 사건의 분류는 6월 초중순 값으로 정해진다(하네스 앞 구간 고르기와 같은 처리). 학습 1회, 조율 없음.",
          "- 2025 등 다른 해는 보조로만 본다. 선택 10칸의 2025 차이는 작고 부호가 칸마다 다르다(표).",
          "", "## 점검", "",
          f"- 원자 22개(일봉 19, 60분봉 3) 중 문턱이 있는 15개를 과제가 준 문턱 문자열과 대조해 모두 같았고(나머지 7개는 =0 이진 원자), 일봉 원자 켜짐 표지는 하네스 `tercile_cells`와 "
          f"같았다. b4 날짜 특징을 배치 규칙으로 다시 계산한 값 차이 0, 원자료 공개 시점이 놓인 날보다 늦은 경우 0({sj['b4_date_check']}).",
          (lambda ck: "- 독립 재계산(`code/b6_combo_check.py`, 긴 형식 일봉·저장 사건표에서 하네스 쌓기·판정 없이): 가족마다 1칸의 앞·뒤 켜짐 수·평균·차이 최대 오차 "
           + ", ".join(f"`{k}` {v['max_abs_diff']:.1e}" for k, v in ck.items() if isinstance(v, dict) and "max_abs_diff" in v) + "."
           )(json.loads((RES / "b6_combo_check.json").read_text(encoding="utf-8")))
          if (RES / "b6_combo_check.json").exists() else "- 독립 재계산: 못 구함(`b6_combo_check.py` 미실행).",
          "", "## 결과 뒤 변경", "",
          "- 없음. 결과를 본 뒤에는 보고서 문구(결론·가족 표·같은 사건 집합 묶음·한계·점검)와 독립 대조 스크립트 b6_combo_check.py만 덧붙였고 정의·칸·판정은 계획 그대로다.", ""]
    OUT_MD.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    {"plan": cmd_plan, "run": cmd_run}[cmd]()
