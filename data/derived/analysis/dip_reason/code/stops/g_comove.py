"""칼손절 + 40분봉 RSI 조건 — 묶음 동반 하락 / 혼자 빠짐 계열(g_comove). 2026-10-07
사용자 가설(10/7): "40분봉 RSI가 30 이하(이미 과매도)일 때는 손절하면 안 좋다."
→ 손절 신호가 나와도 40분봉 RSI가 낮으면 보류하고, 40분봉 RSI가 아직 높은데 빠질 때만 자른다.

기본 손절(stops/comove.py의 make를 그대로 불러 쓴다, 원본 수정 없음. 모두 봉 i 시가에 팔지 i-1 봉까지 값으로 정한다):
  G  묶음중앙 30분: 같은 대분류 그룹 다른 종목(3개 미만이면 QQQ) '지금까지 30분' 수익률 중앙값 <= -X.
     값 = [시작 <= tk[i-1]인 마지막 봉 종가] / [시작 <= tk[i-1]-30분인 마지막 봉 종가] - 1. [정규장만 적용] = i-1 봉이 정규장일 때만.
  I  혼자 빠짐: 자기 c[i-1] <= 매수가 x (1-Y) 이고 묶음중앙 '매수 뒤' 수익률(기준점 = 매수 봉 열리기 전 끝난 마지막 봉 종가) > -X.
조건(40분봉 RSI14, stop_lab.htf_rsi, 04:00 ET 기준, 5분봉으로 만든 큰 봉; 인덱스 i 값 = 봉 i 시가에 쓸 수 있는 값):
  dev>T   기본 손절 신호가 참이고 dev[i] > T 일 때만 판다(dev[i] = 그 전 완성 40분봉 RSI 상태에 c[i-1]을 넣은 진행 중 값).
  conf>T  같은데 conf[i](봉 i 시작까지 완성된 마지막 40분봉 RSI)를 쓴다.
  첫신호 dev>T  기본 손절 신호가 그 거래에서 처음 참이 된 봉에서만 판정: dev > T면 판다, 아니면 그 거래는 끝까지 손절 안 함.
  반대 dev<=T  기본 손절 신호가 참이고 dev[i] <= T 일 때만 판다(조건 효과 분리용).
  RSI 값이 NaN(early 보유 봉의 0.8%, 큰 봉 RSI가 아직 안 잡힌 종목 초기)이면 '낮지 않음'으로 본다 → 조건 쪽은 자르고 반대 쪽은 안 자름.
선택: 조건 붙은 규칙(dev>·conf>·첫신호) 중 early 사용자78 24시간 차이 최대(단 early 정규장만 차이 >= 0). late는 고른 뒤 확인만.
고른 뒤(선택 후 추가, late 숫자로 다시 고르지 않음): 이웃(40분 → 30·60분, RSI 길이 9), 같은 T의 반대 조건, 하루 의존, 손절 분해.
사용: py stops/g_comove.py   (code 폴더에서)   → ../results/stops/g_comove.txt
"""
import io, os, sys, time, warnings
from contextlib import redirect_stdout
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, CODE)
sys.path.insert(0, HERE)
from stop_lab import load, compare, stop_audit, run, base_trades, htf_rsi, UNIV, RAW
import comove as CM

warnings.filterwarnings("ignore", category=RuntimeWarning)
OUT = os.path.normpath(os.path.join(CODE, "..", "results", "stops", "g_comove.txt"))
U78 = UNIV["사용자78"]
CELLS = [(p, sc) for p in ("early", "late") for sc in ("24시간", "정규장만")]

BASES = {  # 키: (comove.make 인자 kind, X, Y, regonly), 설명
    "G3r": (("gwin", 0.03, None, True), "묶음중앙30분<=-3% [i-1 정규장만]"),
    "G25r": (("gwin", 0.025, None, True), "묶음중앙30분<=-2.5% [i-1 정규장만]"),
    "G2r": (("gwin", 0.02, None, True), "묶음중앙30분<=-2% [i-1 정규장만]"),
    "G3": (("gwin", 0.03, None, False), "묶음중앙30분<=-3% [24시간]"),
    "G25": (("gwin", 0.025, None, False), "묶음중앙30분<=-2.5% [24시간]"),
    "G2": (("gwin", 0.02, None, False), "묶음중앙30분<=-2% [24시간]"),
    "I5_1": (("idio", 0.01, 0.05, False), "혼자빠짐 자기종가<=-5% & 묶음매수뒤>-1%"),
    "I5_2": (("idio", 0.02, 0.05, False), "혼자빠짐 자기종가<=-5% & 묶음매수뒤>-2%"),
    "I3_1": (("idio", 0.01, 0.03, False), "혼자빠짐 자기종가<=-3% & 묶음매수뒤>-1%"),
}


def gated(bkey, gate=None, T=30, tf=40, n=14, mode="cont"):
    """gate: None(조건 없음) | 'dev' | 'conf' (RSI > T일 때만 자름) | 'rev' (dev <= T일 때만 자름). mode 'veto' = 첫 신호에서만 판정."""
    base_mk = CM.make(*BASES[bkey][0])

    def make_rule(A):
        base = base_mk(A)
        if gate is None:
            return base
        dev, conf = htf_rsi(A, tf, n)
        low = (conf if gate == "conf" else dev) <= T              # NaN → False(낮지 않음)
        state = {}

        def rule(i, ei, ent):
            if not base(i, ei, ent):
                return False
            if gate == "rev":
                return bool(low[i])
            if mode == "veto":
                st = state.get(ei)
                if st is None:
                    st = state[ei] = not bool(low[i])
                return st
            return not bool(low[i])
        return rule
    return make_rule


def glabel(gate, T, tf=40, n=14, mode="cont"):
    if gate is None:
        return "조건없음"
    tag = "" if (tf, n) == (40, 14) else f"[{tf}분 RSI{n}]"
    if gate == "rev":
        return f"반대 dev<={T}{tag}"
    return ("첫신호 " if mode == "veto" else "") + f"{gate}>{T}{tag}"


def htf_trunc_test():
    """40분 RSI 미래 참조 점검: 컷오프 뒤에 끝나는 5분봉을 지우고 자기 종가도 컷오프 뒤 봉은 무작위로 바꿔 다시 계산 →
    시가 <= 컷오프인 결정 봉 i의 dev[i]·conf[i]가 같아야 한다(htf_rsi 본문을 그대로 옮겨 잘린 자료로 계산)."""
    from htf import intraday_bars, rma_state, _rsi_from
    msgs = []
    rng = np.random.default_rng(11)
    for p in ("early", "late"):
        S = load(p)
        tot = bad = 0
        for s in ("NVDA", "MU", "PLTR", "TEM", "COIN", "IONQ"):
            if s not in S:
                continue
            A = S[s]
            dev0, conf0 = htf_rsi(A)
            m5full = pd.read_parquet(os.path.join(RAW, f"5m_{s}.parquet"))
            tk, ms = A["tk"], A["mins"] * 60 * 10**9
            for cut in rng.choice(tk[len(tk) // 5: -len(tk) // 5], 3, replace=False):
                m5 = m5full[m5full.index.as_unit("ns").asi8 + 5 * 60 * 10**9 <= cut]
                c = A["c"].copy()
                late_m = tk + ms > cut
                c[late_m] *= rng.uniform(0.5, 1.5, late_m.sum())
                hs, hc = intraday_bars(m5, 40)
                hc = hc.astype(float)
                g, l, r = rma_state(hc, 14)
                hs_ns = hs.as_unit("ns").asi8
                step = 40 * 60 * 10**9
                prev_t = np.r_[tk[0], tk[:-1]]
                anchor = (A["idx"].normalize() + pd.Timedelta(hours=4)).as_unit("ns").asi8
                anchor_prev = np.r_[anchor[0], anchor[:-1]]
                cur_prev = anchor_prev + ((prev_t - anchor_prev) // step) * step
                j = np.searchsorted(hs_ns, cur_prev, side="left") - 1
                ok, jj = j >= 0, np.maximum(j, 0)
                c_prev = np.r_[np.nan, c[:-1]]
                dev = np.where(ok, _rsi_from(g[jj], l[jj], hc[jj], c_prev, 14), np.nan)
                cur_i = anchor + ((tk - anchor) // step) * step
                jc = np.searchsorted(hs_ns, cur_i, side="left") - 1
                conf = np.where(jc >= 0, r[np.maximum(jc, 0)], np.nan)
                k = np.nonzero(tk <= cut)[0]
                k = k[k >= 1]
                tot += 2 * len(k)
                bad += int((~np.isclose(dev0[k], dev[k], equal_nan=True)).sum())
                bad += int((~np.isclose(conf0[k], conf[k], equal_nan=True)).sum())
        msgs.append(f"{p} 40분 RSI 잘라내기 시험(6종목×컷오프 3개): 결정값 {tot:,}개 중 다른 값 {bad}개")
    return "; ".join(msgs)


# ---- 메모리 절약 실행: compare와 같은 계산을 기간별로 나눠 한다(early 전부 → 선택 → late 전부). ----
# compare(make_rule)는 매번 두 기간 자료를 새로 읽고, comove 특징 행렬도 두 기간치를 함께 들고 있어
# 병렬 작업 중 메모리가 모자랐다(10/7 첫 실행 MemoryError). 아래 cmp_period는 stop_lab.compare 본문과 같은 순서·같은
# 함수(base_trades, run, metrics)로 한 기간 행만 만든다. 이 구조에서는 early 선택이 끝난 뒤에야 late를 계산하므로
# late가 선택에 끼어들 수 없다.
from stop_lab import metrics
import stop_lab as SL
import gc


def cmp_period(S, make_rule, label, period):
    rows = []
    for reg_only in (False, True):
        b = base_trades(period, reg_only)
        t = run(S, make_rule, period, reg_only, True)
        for un, us in UNIV.items():
            rows.append(dict(규칙=label, 기간=period, 범위="정규장만" if reg_only else "24시간", 종목=un, **metrics(t, b, us)))
    return rows


def show_df(df):
    f = df.copy()
    for k, fm in (("평균", "{:+.2%}"), ("승률", "{:.0%}"), ("최악", "{:+.1%}"), ("하위5", "{:+.1%}"), ("합", "{:+.2f}"),
                  ("차이", "{:+.2f}"), ("t", "{:+.2f}"), ("손절평균", "{:+.2%}"), ("차이_10월6일", "{:+.2f}")):
        f[k] = f[k].map(lambda v, fm=fm: "" if pd.isna(v) else fm.format(v))
    return f.to_string(index=False)


def audit_S(S, make_rule, period, reg_only=False, syms=U78):
    """stop_lab.stop_audit와 같은 계산(이미 읽은 S를 씀)."""
    t = run(S, make_rule, period, reg_only, True, syms=syms)
    b = base_trades(period, reg_only)
    b = b[b.sym.isin(syms)].set_index(["sym", "ei"]).ret
    s = t[t.why == "손절"].set_index(["sym", "ei"]).ret
    m = s.to_frame("손절").join(b.rename("기준"), how="inner")
    if not len(m):
        return dict(손절=0)
    return dict(손절=len(s), 짝=len(m), 헛손절비율=(m.기준 > 0).mean(), 손절평균=m.손절.mean(), 기준평균=m.기준.mean(),
                아낀손실합=(m.손절 - m.기준).sum(), 크게잃을거래_막음=int((m.기준 <= -0.05).sum()))


def free_period(mins):
    CM._CTX.pop(mins, None)
    for k in [k for k in SL._HTF if k[1] == mins]:
        SL._HTF.pop(k)
    gc.collect()


def spec_label(sp):
    bkey, gate, T, tf, n, mode = sp
    return f"{BASES[bkey][1]} | {glabel(gate, T, tf, n, mode)}"


def fmt_audit(a):
    return ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in a.items())


def decompose(S, period, bk, T):
    """조건 없는 같은 손절의 손절 거래를 결정 봉 dev로 나눈다(사용자78, 손절 수익 - 같은 매수 봉 기준 거래 수익)."""
    umk = gated(bk)
    out = []
    for ro in (False, True):
        t = run(S, umk, period, ro, syms=U78)
        b = base_trades(period, ro)
        b = b[b.sym.isin(U78)].set_index(["sym", "ei"]).ret
        s = t[t.why == "손절"].copy()
        s["dev"] = [htf_rsi(S[x.sym])[0][x.xi] for x in s.itertuples()]
        s = s.set_index(["sym", "ei"]).join(b.rename("기준"), how="inner")
        s["득실"] = s.ret - s.기준
        parts = []
        for nm, m in ((f"dev<={T}", s.dev <= T), (f"dev>{T}", ~(s.dev <= T))):
            x = s[m]
            parts.append(f"{nm} {len(x)}건 득실합 {x.득실.sum():+.3f} 평균 {(x.득실.mean() if len(x) else 0):+.2%} "
                         f"헛손절 {((x.기준 > 0).mean() if len(x) else 0):.0%}")
        out.append(f"  {period} {'정규장만' if ro else '24시간'}: " + " | ".join(parts))
    return out


def daydep(S, mk, period):
    out = []
    for ro in (False, True):
        t = run(S, mk, period, ro, syms=U78)
        b = base_trades(period, ro)
        b = b[b.sym.isin(U78)]
        days = sorted(set(t.day) | set(b.day))
        d = t.groupby("day").ret.sum().reindex(days, fill_value=0) - b.groupby("day").ret.sum().reindex(days, fill_value=0)
        d = d[d.abs() > 1e-12].sort_values(ascending=False)
        tot = d.sum()
        top = d.head(3)
        out.append(f"  {period} {'정규장만' if ro else '24시간'}: 차이 {tot:+.3f}, 다른 날 {len(d)}(+{int((d > 0).sum())}/-{int((d < 0).sum())}), "
                   + (f"최대 이득일 {top.index[0]} {top.iloc[0]:+.3f} → 빼면 {tot - top.iloc[0]:+.3f}, 상위 2일 빼면 {tot - top.head(2).sum():+.3f}, "
                      f"최대 손해일 {d.index[-1]} {d.iloc[-1]:+.3f}" if len(d) else "차이 없음"))
    return out


STAGE = {}


def summarize(res, specs):
    R = []
    for sp in specs:
        df = pd.DataFrame(res[sp])
        per = set(df.기간)
        u = {(p, sc): df[(df.기간 == p) & (df.범위 == sc) & (df.종목 == "사용자78")].iloc[0] for p, sc in CELLS if p in per}
        a = {(p, sc): df[(df.기간 == p) & (df.범위 == sc) & (df.종목 == "전체137")].iloc[0] for p, sc in CELLS if p in per}

        def g(D, p, sc, k):
            return D[p, sc][k] if (p, sc) in D else np.nan
        bkey, gate, T, tf, n, mode = sp
        R.append(dict(규칙=spec_label(sp), base=bkey, gate=gate or "none", T=T, tf=tf, n=n, mode=mode, stage=STAGE[sp],
                      e24=g(u, "early", "24시간", "차이"), eR=g(u, "early", "정규장만", "차이"),
                      l24=g(u, "late", "24시간", "차이"), lR=g(u, "late", "정규장만", "차이"),
                      e24t=g(u, "early", "24시간", "t"), eRt=g(u, "early", "정규장만", "t"),
                      l24t=g(u, "late", "24시간", "t"), lRt=g(u, "late", "정규장만", "t"),
                      e손절=g(u, "early", "24시간", "손절"), l손절=g(u, "late", "24시간", "손절"),
                      A_e24=g(a, "early", "24시간", "차이"), A_eR=g(a, "early", "정규장만", "차이"),
                      A_l24=g(a, "late", "24시간", "차이"), A_lR=g(a, "late", "정규장만", "차이")))
    return pd.DataFrame(R)


def main():
    t0 = time.time()
    log = io.StringIO()

    def P(*a):
        print(*a, flush=True)
        print(*a, file=log)

    P("칼손절 + 40분봉 RSI 조건 — 묶음 동반/혼자 빠짐(g_comove) 시험 결과 (2026-10-07, stop_lab 시험대)")
    P("기준(손절 없음): 사용자78 24시간 early +27.12 · late +17.04, 정규장만 early +1.47 · late +2.28")
    P("미래 참조 점검(묶음 신호, comove.lookahead_check):", CM.lookahead_check())
    P("미래 참조 점검(40분 RSI):", htf_trunc_test())
    free_period(3)
    free_period(5)

    specs = []
    for bk in BASES:  # 1단계 격자: 기본 손절 9개 × (조건없음, dev>25·30·35·40, conf>30·35, 반대 dev<=30) = 72
        specs.append((bk, None, 30, 40, 14, "cont"))
        for T in (25, 30, 35, 40):
            specs.append((bk, "dev", T, 40, 14, "cont"))
        for T in (30, 35):
            specs.append((bk, "conf", T, 40, 14, "cont"))
        specs.append((bk, "rev", 30, 40, 14, "cont"))
    for bk in ("G3r", "I5_1", "I5_2", "I3_1"):  # 첫 신호 판정형 8
        for T in (30, 35):
            specs.append((bk, "dev", T, 40, 14, "veto"))
    for sp in specs:
        STAGE[sp] = 1
    n1 = len(specs)
    res = {sp: [] for sp in specs}

    def run_spec(S, sp, period):
        rows = cmp_period(S, gated(sp[0], sp[1], sp[2], sp[3], sp[4], sp[5]), spec_label(sp), period)
        res.setdefault(sp, []).extend(rows)
        u = {r["범위"]: r for r in rows if r["종목"] == "사용자78"}
        print(f"[{period} {specs.index(sp) + 1:2d}] {spec_label(sp):60s} 24h {u['24시간']['차이']:+.2f} 정규 {u['정규장만']['차이']:+.2f} "
              f"손절 {u['24시간']['손절']}  ({time.time() - t0:.0f}s)", flush=True)

    # ---------------- early: 전부 계산 → 선택 ----------------
    S = CM._ctx(5)["S"]
    for sp in specs:
        run_spec(S, sp, "early")
    R = summarize(res, specs)
    G = R[R.gate.isin(["dev", "conf"])]
    cand = G[G.eR >= 0].sort_values("e24", ascending=False)
    best = cand.iloc[0] if len(cand) else G.sort_values("e24", ascending=False).iloc[0]
    bsp = next(sp for sp in specs if spec_label(sp) == best.규칙)
    bk, bgate, bT, _, _, bmode = bsp
    usp, rsp = (bk, None, 30, 40, 14, "cont"), (bk, "rev", bT, 40, 14, "cont")
    ru = cand[cand.규칙 != best.규칙]
    rsps = [next(sp for sp in specs if spec_label(sp) == x) for x in ru.규칙.iloc[:2]]
    # 고른 뒤 추가(이웃·대조): 40분 → 30·60분, RSI 길이 9, 같은 T 반대 조건(T가 30이 아니면)
    extra = [(bk, bgate, bT, 30, 14, bmode), (bk, bgate, bT, 60, 14, bmode), (bk, bgate, bT, 40, 9, bmode)]
    if bT != 30:
        extra.append(rsp)
    for sp in extra:
        specs.append(sp)
        STAGE[sp] = 2
        run_spec(S, sp, "early")
    AUD = {}
    aud_specs = (("best", bsp), ("ungated", usp), ("reverse", rsp), ("ru1", rsps[0]), ("ru2", rsps[1]))
    for nm, sp in aud_specs:
        mk = gated(*sp[:5], mode=sp[5])
        for ro in (False, True):
            AUD[nm, "early", ro] = audit_S(S, mk, "early", ro)
    DEC = decompose(S, "early", bk, bT)
    DD = daydep(S, gated(*bsp[:5], mode=bsp[5]), "early")
    del S
    free_period(5)
    print(f"early 끝, 선택: {best.규칙} ({time.time() - t0:.0f}s)", flush=True)

    # ---------------- late: 확인만 ----------------
    S = CM._ctx(3)["S"]
    for sp in specs:
        run_spec(S, sp, "late")
    for nm, sp in aud_specs:
        mk = gated(*sp[:5], mode=sp[5])
        for ro in (False, True):
            AUD[nm, "late", ro] = audit_S(S, mk, "late", ro)
    DEC += decompose(S, "late", bk, bT)
    DD += daydep(S, gated(*bsp[:5], mode=bsp[5]), "late")
    del S
    free_period(3)

    # ---------------- 보고 ----------------
    R = summarize(res, specs)
    n = len(R)
    allpos = R[(R.e24 > 0) & (R.eR > 0) & (R.l24 > 0) & (R.lR > 0)]
    allpos_g = allpos[allpos.gate.isin(["dev", "conf"])]
    P(f"\n시험한 규칙 {n}개 (1단계 격자 {n1}개: 기본 손절 9개 × 8 + 첫신호형 8, early로 고른 뒤 이웃·대조 {n - n1}개). 걸린 시간 {time.time() - t0:.0f}초")
    P(f"사용자78 네 칸(early/late × 24시간/정규장만) 모두 플러스: {len(allpos)}개 (그중 40분 RSI 조건(dev>·conf>) 붙은 것 {len(allpos_g)}개)")
    for _, r in allpos.iterrows():
        P(f"  {r.규칙}: e24 {r.e24:+.2f} eR {r.eR:+.2f} l24 {r.l24:+.2f} lR {r.lR:+.2f} (손절 e {r.e손절} / l {r.l손절})")
    P(f"early 24시간 플러스: {(R.e24 > 0).sum()}개, early 두 칸 플러스: {((R.e24 > 0) & (R.eR > 0)).sum()}개")

    P("\n== 전체 규칙 요약(사용자78 차이; l*는 확인용, 선택에 쓰지 않음; d* = 같은 기본 손절 '조건없음' 대비 변화)")
    ung = R[R.gate == "none"].set_index("base")
    for c in ("e24", "eR", "l24", "lR"):
        R["d" + c] = R[c] - R.base.map(ung[c])
    show = R[["규칙", "e24", "e24t", "eR", "eRt", "l24", "l24t", "lR", "lRt", "e손절", "l손절", "de24", "deR", "dl24", "dlR", "A_e24", "A_l24"]]
    for bkey in BASES:
        P(f"\n-- {BASES[bkey][1]}")
        P(show[R.base == bkey].to_string(index=False, float_format=lambda v: f"{v:+.2f}"))

    P("\n== 조건 효과 평균(1단계, 기본 손절별 '조건 - 조건없음'의 평균, 사용자78)")
    R1 = R[R.stage == 1].copy()
    R1["조건"] = R1.규칙.str.split(" | ", regex=False).str[1]
    P("  기본 손절 9개 전체:")
    P(R1.groupby("조건")[["de24", "deR", "dl24", "dlR"]].mean().to_string(float_format=lambda v: f"{v:+.3f}"))
    for fam, m in (("G(묶음 30분) 6개", R1.base.str.startswith("G")), ("I(혼자 빠짐) 3개", R1.base.str.startswith("I"))):
        P(f"  {fam}:")
        P(R1[m].groupby("조건")[["de24", "deR", "dl24", "dlR"]].mean().to_string(float_format=lambda v: f"{v:+.3f}"))

    P(f"\n== 선택(40분 RSI 조건 붙은 규칙 중 early 사용자78 24시간 최대, early 정규장만 >= 0): {best.규칙}")
    P(show_df(pd.DataFrame(res[bsp])))
    for p in ("early", "late"):
        for ro in (False, True):
            P(f"stop_audit {p} {'정규장만' if ro else '24시간'} 사용자78: {fmt_audit(AUD['best', p, ro])}")
    for nm, key, sp in (("같은 손절 조건없음", "ungated", usp), (f"반대 조건(dev<={bT})", "reverse", rsp)):
        P(f"\n-- {nm}: {spec_label(sp)}")
        P(show_df(pd.DataFrame(res[sp])))
        for p in ("early", "late"):
            P(f"stop_audit {p} 24시간 사용자78: {fmt_audit(AUD[key, p, False])}")

    P("\n== 차순위(40분 RSI 조건 붙은 규칙, early 정규장만 >= 0, early 24시간 순, 선택안 제외 상위 6개)")
    Rk = R.set_index("규칙")
    for lb in ru.규칙.iloc[:6]:
        r = Rk.loc[lb]
        P(f"  {lb}: early 24h {r.e24:+.2f} (t {r.e24t:+.2f}) 정규 {r.eR:+.2f} | late 24h {r.l24:+.2f} 정규 {r.lR:+.2f} | 손절 {r.e손절}/{r.l손절}")
    for nm, sp in (("ru1", rsps[0]), ("ru2", rsps[1])):
        for p in ("early", "late"):
            P(f"  stop_audit {p} 24시간 [{spec_label(sp)}]: {fmt_audit(AUD[nm, p, False])}")

    P("\n== 이웃(early로 고른 뒤 추가, 사용자78 차이)")
    for _, r in R[R.stage == 2].iterrows():
        P(f"  {r.규칙}: e24 {r.e24:+.2f} eR {r.eR:+.2f} | l24 {r.l24:+.2f} lR {r.lR:+.2f} | 손절 {r.e손절}/{r.l손절}")

    P(f"\n== 분해: '{spec_label(usp)}' 손절 거래를 결정 봉 dev(40분 RSI14)로 나눔"
      "(사용자78, 손절 수익 - 같은 매수 봉 기준 거래 수익; 재매수 금지 효과는 빠짐)")
    for x in DEC:
        P(x)
    P("\n== 하루 의존(선택안, 사용자78 날짜별 합 차이)")
    for x in DD:
        P(x)

    P("\n== 각 규칙 전체 표")
    for sp in specs:
        P(f"\n--- {spec_label(sp)}\n{show_df(pd.DataFrame(res[sp]))}\n")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(log.getvalue())
    R.drop(columns=["de24", "deR", "dl24", "dlR"]).to_csv(OUT.replace(".txt", ".csv"), index=False, encoding="utf-8-sig")
    print("저장:", OUT)


if __name__ == "__main__":
    main()
