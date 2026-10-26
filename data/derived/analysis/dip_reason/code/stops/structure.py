"""칼손절 연구 — 구조 붕괴(바닥 이탈) 계열(2026-10-07).
매수 뒤 가격이 '의미 있는 바닥'을 깨면 봉 i 시가에 판다. 바닥은 매수 봉 ei 시점에 고정한다(그때 이미 아는 값만).

바닥 종류(kind):
  DAYLOW_IN   오늘(ET 날짜) 4시부터 매수 봉까지(매수 봉 포함) 최저가 = 매수 뒤 '새 저점'
  DAYLOW_EX   오늘(ET 날짜) 매수 봉 직전까지 최저가(매수 봉 제외; 매수가가 이미 아래면 종가 확인 시 곧바로 판다)
  PDL_below   전날(직전 정규장이 있던 ET 날짜) 정규장 저가, 매수가보다 아래일 때만(아니면 손절 없음)
  PDL_any     전날 정규장 저가, 항상(매수가가 이미 아래면 확인되는 즉시 판다)
  LCR_*       직전 '끝난' 정규장 저가(애프터 매수는 그날 정규장, 프리·정규 매수는 전날)
  N{m}_IN     매수 직전 m분 + 매수 봉 최저가          N{m}_EX  매수 직전 m분 최저가(매수 봉 제외)
  SW5_{d}d    기준봉 좌우 5봉 피벗 저점 중 매수가 아래 가장 가까운 것(피벗 확인 = 피벗+5봉 ≤ 매수 봉, 최근 d거래일)
  SW40_{d}d   40분봉(ET 날짜·0시부터 분//40 묶음) 좌우 5봉 피벗 저점, 같은 방식(확인 = 오른쪽 5번째 40분봉의 마지막 기준봉 ≤ 매수 봉)
확인(conf): c = 봉 i-1 종가 < 바닥×(1-여유), l = 봉 i-1 저가 < 바닥×(1-여유). 여유 buf 0 / 0.5% / 1%.
범위(scope): all = 모든 세션, reg = 봉 i-1과 i가 모두 정규장일 때만, ent_reg = 정규장 매수만, ent_ext = 프리·애프터 매수만,
             w120 = 매수 뒤 120분 안(봉 i 시작이 매수 봉 시작 + 120분 이내)만.
고르는 법(early만): 사용자78 24시간 차이가 플러스이고 정규장만 차이가 0 이상인 규칙 중 24시간 차이 최대. late는 확인용.
사용: py stops/structure.py   (code 폴더에서)  → ../results/stops/structure.txt
"""
import os, sys, time
import numpy as np, pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from stop_lab import load, run, compare, stop_audit, base_trades, UNIV, PNAME

OUT = os.path.normpath(os.path.join(HERE, "..", "..", "results", "stops", "structure.txt"))
_fh = None


def say(*a):
    global _fh
    if _fh is None:
        _fh = open(OUT, "w", encoding="utf-8")
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _fh.write(s + "\n"); _fh.flush()


# ---------------------------------------------------------------- 종목별 미리 계산(같은 종목·봉 길이는 한 번만)
_F = {}


def _pivots(low, conf_idx, rank):
    """좌우 5봉 피벗 저점: low[j] <= low[j-5..j+5]. 반환 (확인 기준봉 인덱스, 값, 거래일 순번) — 확인 순으로 정렬."""
    n = len(low)
    if n < 11:
        return dict(conf=np.zeros(0, np.int64), val=np.zeros(0), rank=np.zeros(0, np.int64))
    w = sliding_window_view(low, 11).min(axis=1)
    p = np.zeros(n, bool)
    p[5:n - 5] = low[5:n - 5] <= w
    j = np.nonzero(p)[0]
    return dict(conf=conf_idx(j), val=low[j], rank=rank(j))


def feats(A):
    key = (A["sym"], A["mins"], len(A["o"]))
    if key in _F:
        return _F[key]
    l, mod, reg = A["l"], A["mod"], A["reg"]
    n = len(l)
    dayi = np.array([d.toordinal() for d in A["day"]], dtype=np.int64)
    ud, inv = np.unique(dayi, return_inverse=True)
    s = pd.Series(l)
    cm = s.groupby(inv).cummin()
    dl_in = cm.values
    dl_ex = cm.groupby(inv).shift(1).values                     # 그날 첫 봉은 NaN
    regL = np.full(len(ud), np.nan)
    if reg.any():
        g = pd.Series(l[reg]).groupby(inv[reg]).min()
        regL[g.index.values] = g.values
    prev = np.full(len(ud), np.nan)
    last = np.nan
    for k in range(len(ud)):
        prev[k] = last
        if not np.isnan(regL[k]):
            last = regL[k]
    pdl = prev[inv]
    lcr = np.where((mod >= 960) & ~np.isnan(regL[inv]), regL[inv], pdl)
    p5 = _pivots(l, lambda j: j + 5, lambda j: inv[j])
    bk = dayi * 10000 + mod // 40
    starts = np.nonzero(np.r_[True, bk[1:] != bk[:-1]])[0]
    ends = np.r_[starts[1:] - 1, n - 1]
    blow = np.minimum.reduceat(l, starts)
    p40 = _pivots(blow, lambda j: ends[j + 5], lambda j: inv[starts[j]])
    F = dict(inv=inv, dl_in=dl_in, dl_ex=dl_ex, pdl=pdl, lcr=lcr, p5=p5, p40=p40)
    _F[key] = F
    return F


def _nearest_below(P, ei, ent, r_e, days):
    k = np.searchsorted(P["conf"], ei, side="right")             # 매수 봉까지 확인된 피벗만
    lo = np.searchsorted(P["rank"][:k], r_e - days + 1, side="left")
    v = P["val"][lo:k]
    v = v[v < ent]
    return v.max() if len(v) else np.nan


def level(F, A, kind, ei, ent):
    l, mins = A["l"], A["mins"]
    if kind == "DAYLOW_IN":
        return F["dl_in"][ei]
    if kind == "DAYLOW_EX":
        return F["dl_ex"][ei]
    if kind.startswith("PDL") or kind.startswith("LCR"):
        v = (F["pdl"] if kind.startswith("PDL") else F["lcr"])[ei]
        if kind.endswith("_below") and not (v < ent):
            return np.nan
        return v
    if kind[0] == "N":
        m, side = kind[1:].split("_")
        k = int(m) // mins
        a = max(0, ei - k)
        seg = l[a:ei + 1] if side == "IN" else l[a:ei]
        return seg.min() if len(seg) else np.nan
    if kind.startswith("SW"):
        name, d = kind.split("_")
        P = F["p5"] if name == "SW5" else F["p40"]
        return _nearest_below(P, ei, ent, F["inv"][ei], int(d[:-1]))
    raise ValueError(kind)


def make(kind, conf="c", buf=0.0, scope="all"):
    def make_rule(A):
        F = feats(A)
        src = A["c"] if conf == "c" else A["l"]
        reg, ses, mins = A["reg"], A["ses"], A["mins"]
        memo = [-1, np.nan, True]

        def rule(i, ei, ent):
            if memo[0] != ei:
                lv = level(F, A, kind, ei, ent)
                on = True
                if scope == "ent_reg":
                    on = ses[ei] == "정규"
                elif scope == "ent_ext":
                    on = ses[ei] != "정규"
                memo[0], memo[1], memo[2] = ei, lv * (1 - buf), on
            L = memo[1]
            if not memo[2] or L != L:
                return False
            if scope == "reg" and not (reg[i - 1] and reg[i]):
                return False
            if scope == "w120" and (i - ei) * mins > 120:
                return False
            return src[i - 1] < L
        return rule
    return make_rule


def name(kind, conf, buf, scope):
    return f"{kind}|{conf}|{buf * 100:.1f}%|{scope}"


# ---------------------------------------------------------------- 실행·기록
RES = {}      # 이름 → compare DataFrame
ORDER = []
# 중간에 끊겨도 다시 계산하지 않게 scratchpad에 규칙별 compare 표를 쌓아 둔다(결과에는 영향 없음; 지우면 처음부터 계산)
CACHE_CSV = r"C:/Users/geunt/AppData/Local/Temp/claude/C--Users-geunt-OneDrive-Desktop---/4e686c73-74be-4337-8ccc-2f6136beaf7d/scratchpad/structure_cache.csv"
_CACHE = {}
if os.path.exists(CACHE_CSV):
    for k, g in pd.read_csv(CACHE_CSV, encoding="utf-8").groupby("규칙", sort=False):
        _CACHE[k] = g.reset_index(drop=True)


def cell(df, period, scope, un="사용자78"):
    r = df[(df.기간 == period) & (df.범위 == scope) & (df.종목 == un)].iloc[0]
    return r


def test(kind, conf, buf, scope="all"):
    nm = name(kind, conf, buf, scope)
    if nm in RES:
        return nm
    t0 = time.time()
    if nm in _CACHE:
        df = _CACHE[nm]
    else:
        df = compare(make(kind, conf, buf, scope), nm, show=False)
        df.to_csv(CACHE_CSV, mode="a", header=not os.path.exists(CACHE_CSV), index=False, encoding="utf-8")
    RES[nm] = df
    ORDER.append(nm)
    e24, erg = cell(df, "early", "24시간"), cell(df, "early", "정규장만")
    l24, lrg = cell(df, "late", "24시간"), cell(df, "late", "정규장만")
    a24, al24 = cell(df, "early", "24시간", "전체137"), cell(df, "late", "24시간", "전체137")
    say(f"{len(ORDER):>2} {nm:<28} E24 {e24.차이:+6.2f}(t{e24.t:+.2f},손절{e24.손절:>4}) ERg {erg.차이:+6.2f}(t{erg.t:+.2f},{erg.손절:>3})"
        f" | L24 {l24.차이:+6.2f}(t{l24.t:+.2f},{l24.손절:>3}) LRg {lrg.차이:+6.2f}(t{lrg.t:+.2f},{lrg.손절:>3})"
        f" | 137: E24 {a24.차이:+6.2f} L24 {al24.차이:+6.2f}  [{time.time() - t0:.0f}s]")
    return nm


def ekey(nm):
    df = RES[nm]
    e24, erg = cell(df, "early", "24시간").차이, cell(df, "early", "정규장만").차이
    ok = e24 > 0 and erg >= 0
    return (ok, e24)


def parse(nm):
    k, c, b, s = nm.split("|")
    return k, c, float(b[:-1]) / 100, s


if __name__ == "__main__":
    T0 = time.time()
    say("칼손절 — 구조 붕괴(바닥 이탈) 계열. 차이 = 기준 대비 수익 합 차이(사용자78), t = 날짜별 짝 t, 손절 = 손절 횟수")
    say("기준: 사용자78 24시간 early +27.12 late +17.04, 정규장만 early +1.47 late +2.28 / 전체137 24시간 early +41.35 late +24.79")
    say("고르는 법(early만): 사용자78 early 24시간 차이>0 이고 early 정규장만 차이>=0 인 규칙 중 early 24시간 차이 최대. late는 전방 확인.\n")

    say("== 1단계: 바닥 8종 × 확인(종가 0/0.5/1%, 저가 0.5/1%) × 모든 세션")
    BASE = ["DAYLOW_IN", "DAYLOW_EX", "PDL_below", "PDL_any", "N60_IN", "N60_EX", "SW5_2d", "SW40_5d"]
    GRID = [("c", 0.0), ("c", 0.005), ("c", 0.01), ("l", 0.005), ("l", 0.01)]
    for kd in BASE:
        for cf, bf in GRID:
            test(kd, cf, bf)

    say("\n== 1.5단계: 바닥 매개변수 변형(각 바탕 바닥의 early 최선 확인·여유 그대로)")
    VAR = {"PDL_below": ["LCR_below"], "PDL_any": ["LCR_any"], "N60_EX": ["N30_EX", "N240_EX"],
           "SW5_2d": ["SW5_1d", "SW5_5d"], "SW40_5d": ["SW40_3d", "SW40_10d"]}
    for kd, vs in VAR.items():
        best = max([nm for nm in ORDER if parse(nm)[0] == kd], key=ekey)
        _, cf, bf, _ = parse(best)
        for v in vs:
            test(v, cf, bf)

    say("\n== 2단계: early 상위 3개(바닥 종류가 다른 것) × 범위 변형(reg, ent_reg, ent_ext, w120)")
    ranked = sorted(ORDER, key=ekey, reverse=True)
    top, seen = [], set()
    for nm in ranked:
        k = parse(nm)[0]
        if k not in seen:
            seen.add(k); top.append(nm)
        if len(top) == 3:
            break
    say("  상위 3:", ", ".join(top))
    for nm in top:
        kd, cf, bf, _ = parse(nm)
        for sc in ("reg", "ent_reg", "ent_ext", "w120"):
            test(kd, cf, bf, sc)

    # ------------------------------------------------------------ 요약
    say(f"\n== 시험한 규칙 수: {len(ORDER)}  (걸린 시간 {time.time() - T0:.0f}s)")
    pos4 = [nm for nm in ORDER if all(cell(RES[nm], p, s).차이 > 0 for p in ("early", "late") for s in ("24시간", "정규장만"))]
    say(f"사용자78 네 칸(early/late × 24시간/정규장만) 모두 플러스: {len(pos4)}개", "; ".join(pos4))
    ok = [nm for nm in ORDER if ekey(nm)[0]]
    say(f"early 선택 조건(24시간>0, 정규장만>=0) 통과: {len(ok)}개")
    ranked = sorted(ORDER, key=ekey, reverse=True)
    best = ranked[0]
    say("\n== early 순위 상위 10(선택은 early만; late는 확인용)")
    for nm in ranked[:10]:
        df = RES[nm]
        say(f"  {nm:<28} E24 {cell(df,'early','24시간').차이:+6.2f} ERg {cell(df,'early','정규장만').차이:+6.2f}"
            f" | L24 {cell(df,'late','24시간').차이:+6.2f} LRg {cell(df,'late','정규장만').차이:+6.2f}")
    say(f"\n== 최선안(early로 고름): {best}  {'(선택 조건 통과)' if ekey(best)[0] else '(선택 조건 통과 규칙 없음 — early 24시간 최대)'}")
    f = RES[best].copy()
    for k, fmt in (("평균", "{:+.2%}"), ("승률", "{:.0%}"), ("최악", "{:+.1%}"), ("하위5", "{:+.1%}"), ("합", "{:+.2f}"),
                   ("차이", "{:+.2f}"), ("t", "{:+.2f}"), ("손절평균", "{:+.2%}"), ("차이_10월6일", "{:+.2f}")):
        f[k] = f[k].map(lambda v: "" if pd.isna(v) else fmt.format(v))
    say(f.drop(columns=["규칙"]).to_string(index=False))
    kd, cf, bf, sc = parse(best)
    mk = make(kd, cf, bf, sc)
    for p in ("early", "late"):
        for ro in (False, True):
            a = stop_audit(mk, p, reg_only=ro)
            say(f"  stop_audit {p} {'정규장만' if ro else '24시간'} 사용자78: " +
                ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in a.items()))
    say("\n== 최선안이 자른 거래의 갈래(stop_paths.py 정의: 반등/작은 손실/폭포/흘러내림), 사용자78 24시간")
    for p in ("early", "late"):
        S = load(p)
        b = base_trades(p)
        b = b[b.sym.isin(UNIV["사용자78"])]
        cls = {}
        for x in b.itertuples():
            A = S[x.sym]
            dd = A["l"][x.ei:x.xi + 1] / x.ent - 1
            mae = dd.min()
            fast = mae < 0 and dd[:max(1, 60 // A["mins"]) + 1].min() <= mae * 0.5
            cls[(x.sym, x.ei)] = ("반등" if x.ret > 0 else "폭포" if (mae <= -0.05 and fast)
                                  else "흘러내림" if x.ret <= -0.03 else "작은 손실")
        t = run(S, mk, p, False, syms=UNIV["사용자78"])
        st_ = t[t.why == "손절"].copy()
        st_["cls"] = [cls.get((s, e), "?") for s, e in zip(st_.sym, st_.ei)]
        st_["base"] = [b.set_index(["sym", "ei"]).ret.get((s, e), np.nan) for s, e in zip(st_.sym, st_.ei)]
        st_["min"] = st_.bars * S[next(iter(S))]["mins"]
        g = st_.groupby("cls").agg(건수=("ret", "size"), 손절평균=("ret", "mean"), 기준평균=("base", "mean"),
                                    차이합=("ret", lambda v: 0.0), 보유분_중앙=("min", "median"))
        g["차이합"] = st_.assign(d=st_.ret - st_.base).groupby("cls").d.sum()
        allc = pd.Series(cls).value_counts()
        g["갈래전체"] = allc.reindex(g.index)
        g = g.rename(index={"?": "기준에 없는 매수"})
        for k in ("손절평균", "기준평균"):
            g[k] = g[k].map(lambda v: "" if pd.isna(v) else f"{v:+.2%}")
        g["차이합"] = g["차이합"].map("{:+.2f}".format)
        g["보유분_중앙"] = g["보유분_중앙"].map("{:.0f}".format)
        g["갈래전체"] = g["갈래전체"].map(lambda v: "" if pd.isna(v) else f"{v:.0f}")
        say(f"  {PNAME[p]} (차이합 = 손절 수익 - 같은 매수의 기준 수익, 보유분 = 매수에서 손절까지 분):\n" + g.to_string())

    say("\n== 차순위(바닥 종류가 다른 early 상위)")
    seen = {kd}
    for nm in ranked[1:]:
        k = parse(nm)[0]
        if k in seen:
            continue
        seen.add(k)
        df = RES[nm]
        say(f"  {nm:<28} E24 {cell(df,'early','24시간').차이:+6.2f} ERg {cell(df,'early','정규장만').차이:+6.2f}"
            f" | L24 {cell(df,'late','24시간').차이:+6.2f} LRg {cell(df,'late','정규장만').차이:+6.2f}")
        if len(seen) == 5:
            break
    _fh.close()
