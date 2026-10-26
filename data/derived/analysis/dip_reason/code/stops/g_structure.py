"""칼손절 연구 — 구조 붕괴 + 40분봉 RSI 조건(2026-10-07).
사용자 가설: 40분봉 RSI가 30 이하(이미 과매도)일 때 손절하면 손해다 → 구조 붕괴 손절 신호가 나와도 40분봉 RSI가 낮으면 보류.

구조 바닥(kind)·확인(conf c=종가/l=저가)·여유(buf)·범위(scope)의 정의는 stops/structure.py와 같다(그 파일의 feats·level을 그대로 가져옴).
  LCR_below = 직전 '끝난' 정규장 저가(애프터 매수는 그날 정규장, 프리·정규 매수는 전날), 매수가보다 아래일 때만
  PDL_below = 전날 정규장 저가(매수가보다 아래일 때만), SW5_{d}d / SW40_{d}d = 매수가 아래 가장 가까운 확인된 피벗 저점,
  DAYLOW_IN = 오늘 4시부터 매수 봉까지 최저가, N60_IN = 매수 직전 60분+매수 봉 최저가.
  scope all = 모든 세션, reg = 봉 i-1과 i가 모두 정규장일 때만, w120 = 매수 뒤 120분 안만.
  구조 신호: src[i-1] < 바닥 × (1 - buf)  (src = 종가 c 또는 저가 l, 바닥은 매수 봉 ei 시점에 고정)

40분봉 RSI 조건(gate) — stop_lab.htf_rsi(A, tf=40, n=14)의 인덱스 i 값(봉 i 시가에 판단할 때 쓸 수 있는 값):
  dev[i]  = 진행 중 40분봉 RSI(그 전 완성 40분봉 RMA 상태에 c[i-1]을 넣은 값, 트레이딩뷰 실시간 값)
  conf[i] = 봉 i 시작까지 완성된 마지막 40분봉 RSI
  'dev>g'  : 구조 신호 and dev[i] > g 일 때만 판다(가설 방향, 과매도면 보류 → 나중에 RSI가 g 위로 오르고 여전히 바닥 아래면 판다)
  'conf>g' : 같은 방식, conf[i] 사용
  'dev<=g' : 반대 조건(40분봉 RSI가 낮을 때만 자름) — 조건 효과를 분리하는 대조군
  mode cont  = 매 봉 다시 판단(위 설명)
  mode first = 구조 신호가 처음 나온 봉에서만 판단: 그때 조건 통과면 판다, 아니면 그 거래는 끝까지 손절 안 함
  NaN(RSI 미형성)이면 조건 불통과(손절 안 함).

고르는 법(early만): 사용자78 early 24시간 차이>0 이고 early 정규장만 차이>=0 인 '조건 붙은' 규칙 중 early 24시간 차이 최대.
late는 전방 확인으로만 계산한다(late로 다시 고르지 않음).
사용: py stops/g_structure.py   (code 폴더에서)  → ../results/stops/g_structure.txt
"""
import os, sys, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from stop_lab import load, run, compare, stop_audit, base_trades, htf_rsi, UNIV, PNAME
from structure import feats, level          # 원본 수정 없이 같은 바닥 정의를 재사용

import gc
import stop_lab
_S = {}
_load0 = stop_lab.load


def load(period):
    """stop_lab.load를 한 번만 읽어 재사용(같은 값; 병렬 작업 중 메모리 부족을 피하려고 반복 읽기만 없앰)."""
    if period not in _S:
        _S[period] = _load0(period)
    return _S[period]


stop_lab.load = load            # compare·stop_audit·base_trades가 이 함수를 쓰게 함(파일은 그대로)

OUT = os.path.normpath(os.path.join(HERE, "..", "..", "results", "stops", "g_structure.txt"))
CACHE_CSV = r"C:/Users/geunt/AppData/Local/Temp/claude/C--Users-geunt-OneDrive-Desktop---/4e686c73-74be-4337-8ccc-2f6136beaf7d/scratchpad/g_structure_cache.csv"
_fh = None


def say(*a):
    global _fh
    if _fh is None:
        _fh = open(OUT, "w", encoding="utf-8")
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _fh.write(s + "\n"); _fh.flush()


def make(kind, conf="c", buf=0.0, scope="all", gate="none", mode="cont", tf=40, n=14):
    """gate: 'none' | 'dev>30' | 'conf>35' | 'dev<=30' ..."""
    if gate == "none":
        gk, op, g = None, None, None
    else:
        op = "<=" if "<=" in gate else ">"
        gk, g = gate.split(op)
        g = float(g)

    def make_rule(A):
        F = feats(A)
        src = A["c"] if conf == "c" else A["l"]
        reg, mins = A["reg"], A["mins"]
        if gk is not None:
            dev, cf = htf_rsi(A, tf, n)
            H = dev if gk == "dev" else cf
        memo = [-1, np.nan, False]            # [매수 봉, 손절선, first 모드에서 이 거래 손절 끔]

        def rule(i, ei, ent):
            if memo[0] != ei:
                lv = level(F, A, kind, ei, ent)
                memo[0], memo[1], memo[2] = ei, lv * (1 - buf), False
            L = memo[1]
            if memo[2] or L != L:
                return False
            if scope == "reg" and not (reg[i - 1] and reg[i]):
                return False
            if scope == "w120" and (i - ei) * mins > 120:
                return False
            if not (src[i - 1] < L):
                return False
            if gk is None:
                return True
            h = H[i]
            ok = (h > g) if op == ">" else (h <= g)     # NaN이면 False
            if mode == "first" and not ok:
                memo[2] = True
            return bool(ok)
        return rule
    return make_rule


def name(kind, conf, buf, scope, gate="none", mode="cont", tf=40, n=14):
    s = f"{kind}|{conf}|{buf * 100:.1f}%|{scope}|{gate}"
    if gate != "none" and mode != "cont":
        s += f"|{mode}"
    if gate != "none" and (tf, n) != (40, 14):
        s += f"|tf{tf}n{n}"
    return s


def parse(nm):
    p = nm.split("|")
    kind, conf, buf, scope, gate = p[0], p[1], float(p[2][:-1]) / 100, p[3], p[4]
    mode, tf, n = "cont", 40, 14
    for x in p[5:]:
        if x in ("first", "cont"):
            mode = x
        elif x.startswith("tf"):
            tf, n = int(x[2:].split("n")[0]), int(x.split("n")[1])
    return dict(kind=kind, conf=conf, buf=buf, scope=scope, gate=gate, mode=mode, tf=tf, n=n)


RES, ORDER = {}, []
_CACHE = {}
if os.path.exists(CACHE_CSV):
    for k, g in pd.read_csv(CACHE_CSV, encoding="utf-8").groupby("규칙", sort=False):
        _CACHE[k] = g.reset_index(drop=True)


def cell(df, period, scope, un="사용자78"):
    return df[(df.기간 == period) & (df.범위 == scope) & (df.종목 == un)].iloc[0]


def test(**kw):
    nm = name(**kw)
    if nm in RES:
        return nm
    t0 = time.time()
    if nm in _CACHE:
        df = _CACHE[nm]
    else:
        for k in range(6):
            try:
                df = compare(make(**kw), nm, show=False)
                break
            except MemoryError:
                gc.collect(); time.sleep(30)
        else:
            raise MemoryError(nm)
        df.to_csv(CACHE_CSV, mode="a", header=not os.path.exists(CACHE_CSV), index=False, encoding="utf-8")
    RES[nm] = df
    ORDER.append(nm)
    e24, erg = cell(df, "early", "24시간"), cell(df, "early", "정규장만")
    l24, lrg = cell(df, "late", "24시간"), cell(df, "late", "정규장만")
    a24, al24 = cell(df, "early", "24시간", "전체137"), cell(df, "late", "24시간", "전체137")
    say(f"{len(ORDER):>2} {nm:<40} E24 {e24.차이:+6.2f}(t{e24.t:+.2f},손절{e24.손절:>4}) ERg {erg.차이:+6.2f}(t{erg.t:+.2f},{erg.손절:>3})"
        f" | L24 {l24.차이:+6.2f}(t{l24.t:+.2f},{l24.손절:>3}) LRg {lrg.차이:+6.2f}(t{lrg.t:+.2f},{lrg.손절:>3})"
        f" | 137: E24 {a24.차이:+6.2f} L24 {al24.차이:+6.2f}  [{time.time() - t0:.0f}s]")
    return nm


def gated(nm):
    g = parse(nm)["gate"]
    return g != "none" and "<=" not in g


def ekey(nm):
    df = RES[nm]
    e24, erg = cell(df, "early", "24시간").차이, cell(df, "early", "정규장만").차이
    return (e24 > 0 and erg >= 0, e24)


def line(nm):
    df = RES[nm]
    return (f"{nm:<40} E24 {cell(df,'early','24시간').차이:+6.2f} ERg {cell(df,'early','정규장만').차이:+6.2f}"
            f" | L24 {cell(df,'late','24시간').차이:+6.2f} LRg {cell(df,'late','정규장만').차이:+6.2f}"
            f" | 137 E24 {cell(df,'early','24시간','전체137').차이:+6.2f} L24 {cell(df,'late','24시간','전체137').차이:+6.2f}")


def lookahead_check(nsamp=300, seed=7):
    """dev[i]·conf[i]를 '봉 i 시작 전에 끝난 5분봉만'으로 다시 계산해 같은지 본다(미래 참조 점검)."""
    from htf import intraday_bars, rma_state, _rsi_from
    from stop_lab import RAW
    rng = np.random.default_rng(seed)
    bad, tot = 0, 0
    for p in ("early", "late"):
        S = load(p)
        syms = sorted(S)
        for _ in range(nsamp):
            s = syms[rng.integers(len(syms))]
            A = S[s]
            i = int(rng.integers(200, len(A["tk"])))
            dev, cf = htf_rsi(A)
            m5 = pd.read_parquet(os.path.join(RAW, f"5m_{s}.parquet"))
            ts = m5.index.tz_convert("UTC").as_unit("ns").asi8 if m5.index.tz is not None else m5.index.as_unit("ns").asi8
            m5c = m5[ts + 5 * 60 * 10**9 <= A["tk"][i]]               # 봉 i 시작 전에 끝난 5분봉만
            hs, hc = intraday_bars(m5c, 40)
            hc = hc.astype(float)
            g, l, r = rma_state(hc, 14)
            hs_ns = hs.as_unit("ns").asi8
            step = 40 * 60 * 10**9
            idx = A["idx"]
            anc_p = (idx[i - 1].normalize() + pd.Timedelta(hours=4)).value
            cur_p = anc_p + ((A["tk"][i - 1] - anc_p) // step) * step
            j = np.searchsorted(hs_ns, cur_p, side="left") - 1
            d2 = _rsi_from(g[[j]], l[[j]], hc[[j]], np.array([A["c"][i - 1]]), 14)[0] if j >= 0 else np.nan
            anc = (idx[i].normalize() + pd.Timedelta(hours=4)).value
            cur = anc + ((A["tk"][i] - anc) // step) * step
            jc = np.searchsorted(hs_ns, cur, side="left") - 1
            c2 = r[jc] if jc >= 0 else np.nan
            tot += 1
            same = (np.isclose(d2, dev[i], equal_nan=True) and np.isclose(c2, cf[i], equal_nan=True))
            bad += 0 if same else 1
    return bad, tot


if __name__ == "__main__":
    T0 = time.time()
    say("칼손절 — 구조 붕괴 + 40분봉 RSI 조건. 차이 = 기준 대비 수익 합 차이(사용자78), t = 날짜별 짝 t, 손절 = 손절 횟수")
    say("기준: 사용자78 24시간 early +27.12 late +17.04, 정규장만 early +1.47 late +2.28")
    say("고르는 법(early만): 조건 붙은 규칙 중 사용자78 early 24시간 차이>0, early 정규장만 차이>=0 이고 early 24시간 차이 최대. late는 전방 확인.")
    say("이름 = 바닥|확인(c 종가/l 저가)|여유|범위|조건[|mode][|tf..n..]. 조건 none = 조건 없는 같은 손절, dev<=30 = 반대 조건.\n")

    BASES = [("LCR_below", "c", 0.0, "reg"), ("LCR_below", "c", 0.0, "all"), ("PDL_below", "c", 0.0, "all"),
             ("SW5_1d", "c", 0.01, "reg"), ("SW5_2d", "c", 0.01, "all"), ("SW40_5d", "c", 0.01, "all"),
             ("DAYLOW_IN", "c", 0.01, "all"), ("N60_IN", "c", 0.01, "all")]
    say("== 1단계: 구조 바닥 8개(구조 계열 early 상위·종류별) × 조건 {none, dev>30, dev>35, dev>40, dev<=30(반대)}")
    for kd, cf, bf, sc in BASES:
        for gt in ("none", "dev>30", "dev>35", "dev>40", "dev<=30"):
            test(kind=kd, conf=cf, buf=bf, scope=sc, gate=gt)
    say("\n== 1b단계: 같은 바닥 8개 × first 모드(첫 구조 신호에서만 판단) {dev>30, dev<=30(반대)}")
    for kd, cf, bf, sc in BASES:
        for gt in ("dev>30", "dev<=30"):
            test(kind=kd, conf=cf, buf=bf, scope=sc, gate=gt, mode="first")

    say("\n== 2단계: early 상위 3개(조건 붙은 규칙, 바닥이 다른 것) × 변형(conf 사용, 다른 mode, 문턱 45, 범위 바꿈)")
    ranked = sorted([nm for nm in ORDER if gated(nm)], key=ekey, reverse=True)
    top, seen = [], set()
    for nm in ranked:
        P = parse(nm)
        k = (P["kind"], P["scope"])
        if P["kind"] not in seen:
            seen.add(P["kind"]); top.append(nm)
        if len(top) == 3:
            break
    say("  상위 3:", ", ".join(top))
    for nm in top:
        P = parse(nm)
        base = dict(kind=P["kind"], conf=P["conf"], buf=P["buf"])
        g, md = P["gate"], P["mode"]
        thr = g.split(">")[1]
        test(**base, scope=P["scope"], gate=f"conf>{thr}", mode=md)
        test(**base, scope=P["scope"], gate=g, mode="cont" if md == "first" else "first")
        test(**base, scope=P["scope"], gate="dev>45", mode=md)
        for sc in ("all", "reg", "w120"):
            if sc != P["scope"]:
                test(**base, scope=sc, gate=g, mode=md)
                test(**base, scope=sc, gate="none")

    say("\n== 3단계: 2단계까지 early 최선안의 이웃(여유·확인 방식) + 같은 손절의 조건 없음·반대 조건 보장")
    ranked = sorted([nm for nm in ORDER if gated(nm)], key=ekey, reverse=True)
    b0 = parse(ranked[0])
    for cf, bf in (("c", 0.005), ("l", 0.005)):
        test(kind=b0["kind"], conf=cf, buf=bf, scope=b0["scope"], gate=b0["gate"], mode=b0["mode"])
    ranked = sorted([nm for nm in ORDER if gated(nm)], key=ekey, reverse=True)
    best = ranked[0]
    B = parse(best)
    thr = B["gate"].split(">")[1]
    ung = test(kind=B["kind"], conf=B["conf"], buf=B["buf"], scope=B["scope"], gate="none")
    rev = test(kind=B["kind"], conf=B["conf"], buf=B["buf"], scope=B["scope"], gate=f"{B['gate'].split('>')[0]}<={thr}",
               mode=B["mode"])

    # ------------------------------------------------------------ 요약
    say(f"\n== 시험한 규칙 수: {len(ORDER)} (조건 붙은 {sum(gated(x) for x in ORDER)}, 조건 없음 {sum(parse(x)['gate'] == 'none' for x in ORDER)},"
        f" 반대 조건 {sum('<=' in parse(x)['gate'] for x in ORDER)})  (걸린 시간 {time.time() - T0:.0f}s)")
    pos4 = [nm for nm in ORDER if all(cell(RES[nm], p, s).차이 > 0 for p in ("early", "late") for s in ("24시간", "정규장만"))]
    say(f"사용자78 네 칸(early/late × 24시간/정규장만) 모두 플러스: {len(pos4)}개 " + "; ".join(pos4))
    ok = [nm for nm in ORDER if gated(nm) and ekey(nm)[0]]
    say(f"early 선택 조건 통과(조건 붙은 규칙): {len(ok)}개")

    say("\n== 조건 효과(같은 바닥에서 조건만 바꿈, 사용자78 차이 E24/ERg/L24/LRg)")
    groups = {}
    for nm in ORDER:
        P = parse(nm)
        groups.setdefault((P["kind"], P["conf"], P["buf"], P["scope"]), []).append(nm)
    for k, v in groups.items():
        if len(v) < 2:
            continue
        say(f"  [{k[0]} {k[1]} {k[2]*100:.1f}% {k[3]}]")
        for nm in v:
            say("    " + line(nm))

    ranked = sorted([nm for nm in ORDER if gated(nm)], key=ekey, reverse=True)
    say("\n== early 순위 상위 10(조건 붙은 규칙, 선택은 early만; late는 확인용)")
    for nm in ranked[:10]:
        say("  " + line(nm))
    best = ranked[0]
    B = parse(best)
    say(f"\n== 최선안(early로 고름): {best}  {'(선택 조건 통과)' if ekey(best)[0] else '(선택 조건 통과 규칙 없음 — early 24시간 최대)'}")
    f = RES[best].copy()
    for k, fmt in (("평균", "{:+.2%}"), ("승률", "{:.0%}"), ("최악", "{:+.1%}"), ("하위5", "{:+.1%}"), ("합", "{:+.2f}"),
                   ("차이", "{:+.2f}"), ("t", "{:+.2f}"), ("손절평균", "{:+.2%}"), ("차이_10월6일", "{:+.2f}")):
        f[k] = f[k].map(lambda v: "" if pd.isna(v) else fmt.format(v))
    say(f.drop(columns=["규칙"]).to_string(index=False))
    thr = B["gate"].split(">")[1]
    ung = name(B["kind"], B["conf"], B["buf"], B["scope"], "none")
    rev = name(B["kind"], B["conf"], B["buf"], B["scope"], f"{B['gate'].split('>')[0]}<={thr}", B["mode"])
    for lab, nm in (("조건 없는 같은 손절", ung), ("반대 조건", rev)):
        if nm not in RES:
            test(**parse(nm))
        say(f"  {lab}: " + line(nm))
    for lab, nm in (("최선안", best), ("조건 없음", ung)):
        mk = make(**parse(nm))
        for p in ("early", "late"):
            for ro in (False, True):
                a = stop_audit(mk, p, reg_only=ro)
                say(f"  stop_audit[{lab}] {p} {'정규장만' if ro else '24시간'} 사용자78: " +
                    ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in a.items()))

    say("\n== 하루 의존(최선안, 사용자78 날짜별 합 차이)")
    mk = make(**B)
    for p in ("early", "late"):
        for ro in (False, True):
            b = base_trades(p, ro)
            b = b[b.sym.isin(UNIV["사용자78"])]
            t = run(load(p), mk, p, ro, syms=UNIV["사용자78"])
            d = (t.groupby("day").ret.sum().reindex(sorted(set(b.day) | set(t.day)), fill_value=0)
                 - b.groupby("day").ret.sum().reindex(sorted(set(b.day) | set(t.day)), fill_value=0))
            top3 = d.sort_values(ascending=False).head(3)
            say(f"  {p} {'정규장만' if ro else '24시간'}: 합 {d.sum():+.3f}, 나은날 {(d > 1e-12).sum()} 나쁜날 {(d < -1e-12).sum()},"
                f" 최대이득일 {top3.index[0]} {top3.iloc[0]:+.3f} → 빼면 {d.sum() - top3.iloc[0]:+.3f}; 최대손해일 {d.idxmin()} {d.min():+.3f}")

    say("\n== 차순위(바닥이 다른 early 상위, 조건 붙은 규칙)")
    seen = {B["kind"]}
    for nm in ranked[1:]:
        k = parse(nm)["kind"]
        if k in seen:
            continue
        seen.add(k)
        say("  " + line(nm))
        if len(seen) == 4:
            break

    bad, tot = lookahead_check()
    say(f"\n== 미래 참조 점검: dev[i]·conf[i]를 봉 i 시작 전에 끝난 5분봉만으로 다시 계산 → {tot}개 표본 중 불일치 {bad}개")
    say(f"걸린 시간 {time.time() - T0:.0f}s")
    _fh.close()
