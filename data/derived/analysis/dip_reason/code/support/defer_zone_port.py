"""이격매수 '지지 구역 아래에서 사기' — 자금 한도 계좌 시험(2026-10-10, 키 defer_zone_port).

질문: 사용자는 모든 신호를 다 살 수 없다(뒤 기간 사용자78만 하루 약 50건). 자금이 K칸으로 한정되면 건당 평균이 높은
  Z2(신호 가격 아래 두 번째 지지 구역 하단 지정가)가 계좌 수익에서 지금 지표(신호 때 삼)를 이기는가. 어느 K부터 이기고 지는가.

입력: defer_zone.py가 만든 신호표(scratchpad defer_zone_trades.parquet). 종목마다 상태 기계로 만든 신호·체결·매도 봉과
  그 시각(ts·tfill·tx, UTC ns). 비교 규칙 7개: 지금 지표, Z1·Z2 하단 x 수명 L1(원래 매도까지)·L2(아침 8시까지),
  같은 깊이 대조 고정 -1%·-3%(L2). 종목 묶음 사용자78·전체137·데스크7, 기간 early·late·oos, 범위 24시간·정규장만.
규칙
  계좌 = K칸(K = 3·5·10·20, 경계 참고로 2·40). 한 칸 = 포지션 하나 또는 대기 주문 하나.
  신호가 오면 빈 칸이 있을 때만 받는다. 같은 시각(같은 봉 시작) 신호끼리는 무작위 우선순위, 50번 돌려 평균·분포를 본다.
  묶음(기본): 지금 지표는 신호 시각에 칸을 잡고 매도 시각에 놓는다. 구역·고정 깊이 변형은 주문을 건 신호 시각부터 칸을 잡고
    (지정가 매수 주문은 주문 가능 금액을 묶는다), 미체결이면 취소 시각(L1 = 지금 지표가 팔았을 시각, L2 = 신호 뒤 첫 한국
    아침 8시)에, 체결되면 매도 시각에 놓는다. 구역이 없어 주문을 안 건 신호(stat 2)는 칸을 안 쓴다.
  안묶음(비교): 대기 주문은 칸을 안 쓰고, 체결 시각에 빈 칸이 있어야 체결로 친다(없으면 그 거래는 없음). 지정가 대신 가격
    알림을 걸고 닿을 때 손으로 사는 것에 가깝다(밤에 깨어 있어야 함). 지금 지표는 두 방식이 같다.
  같은 시각에는 매도·취소로 칸이 먼저 비고 그다음 신호를 받는다(매도는 봉 시가, 신호는 봉 안에서 매수선에 닿음).
  기간 끝 미완료(stat 3)는 버린다. 기간 시작 때 계좌는 비어 있다.
가정
  단순 합: 칸 크기 = 시작 자금/K 고정, 계좌 수익 = 받은 거래 수익 합 / K. 이유: (1) 선형이라 규칙 간 차이가 날짜별 손익
    차이로 그대로 쪼개져 날짜 군집 t를 바로 쓸 수 있고 (2) 모든 신호를 받을 때(K 무한) defer_zone.txt의 '합'/K와 같은 잣대이며
    (3) 4~7주에 이익 재투자 여부는 규칙 순위를 바꾸지 않는다. 참고로 복리(매도 때 계좌 x (1 + 수익/K))도 한 칸 적었다.
  평가 낙폭: 기준봉(early 5분·late/oos 3분) 끝마다 들고 있는 포지션을 마지막 종가로 평가한 계좌의 최대 낙폭. 수수료·슬리피지 없음.
  칸 사용률: 그 묶음 종목 봉이 있는 시간(04:00~20:00 ET, 기간 날짜) 동안 잡힌 칸의 평균 비율.
  이길 확률: 같은 번호 무작위끼리 짝지어(서로 독립) 변형 계좌 수익 > 지금 지표 계좌 수익인 비율.
  t: 무작위 50번 평균 날짜별 손익(신호 ET 날짜에 귀속)의 '변형 - 지금' 차이로 계산한 날짜 군집 t.
한계
  - 신호표는 종목마다 '들고 있다고 친' 상태 기계다. 칸이 없어 건너뛴 신호 뒤 그 종목에 더 일찍 나올 새 신호는 표에 없다.
    Pine 차트도 내 자금 사정을 모르고 같은 상태 기계로 표시하므로 '차트에 뜬 신호만 산다'면 이 표가 맞다.
    점검(표 3): 매수선에 닿은 모든 봉을 후보로 다시 만들어(defer_zone.machine을 그 봉에서 시작한 결과), 칸이 없어 버린 종목은
    다음 봉부터 다시 신호를 받게 한 '재신호' 계좌(묶음 방식만)도 돌렸다. K 무한이면 신호표와 똑같이 재현됨을 확인한다.
  - 같은 시각이 아닌 신호는 먼저 온 순서로 받는다(신호를 고르는 다른 기준은 시험하지 않음).
  - 칸 크기 고정·주문 1건 = 1칸: 주문 금액과 체결 금액 차이, 미수·증거금, 환전, 매도 대금 재사용 지연은 무시.
사용: py defer_zone_port.py → results/support/defer_zone_port.txt,
      scratchpad defer_zone_port_runs.csv(설정·무작위별 결과), defer_zone_port_summary.csv
"""
import os, sys
import numpy as np, pandas as pd
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, CODE)
sys.path.insert(0, HERE)
import stop_lab as SL
import defer_zone as DZ                          # 기간(WIN)·종목 묶음(UNIV)·배열·구역·주문가(_target)를 그대로 쓴다

OUT = os.path.normpath(os.path.join(CODE, "../results/support/defer_zone_port.txt"))
TMP = DZ.TMP
TRADES = os.path.join(TMP, "defer_zone_trades.parquet")
BASE = "지금 지표(신호 때 삼)"
RULES = {BASE: "지금", "Z1 하단, 수명 L2(아침 8시까지)": "Z1 L2", "Z2 하단, 수명 L2(아침 8시까지)": "Z2 L2",
         "Z1 하단, 수명 L1(원래 매도까지)": "Z1 L1", "Z2 하단, 수명 L1(원래 매도까지)": "Z2 L1",
         "고정 -1%, L2": "고정-1% L2", "고정 -3%, L2": "고정-3% L2"}
SPEC = {BASE: dict(mode=0, k=0, life=0, fixd=0.0)}                    # 재신호 후보 만들기용(defer_zone.MAIN·FIXED와 같은 뜻)
SPEC.update({nm: dict(mode=1, k=int(nm[1]), life=int(nm[nm.index("수명 L") + 4]), fixd=0.0) for nm in RULES if nm.startswith("Z")})
SPEC.update({"고정 -1%, L2": dict(mode=2, k=0, life=2, fixd=0.01), "고정 -3%, L2": dict(mode=2, k=0, life=2, fixd=0.03)})
KS = (2, 3, 5, 10, 20, 40)                       # 3·5·10·20이 본 시험, 2·40은 경계 참고
KMAIN = (3, 5, 10, 20)
NSEED = 50
DAY = 86_400 * 10**9
PERIODS = ("early", "late", "oos")
RANGES = ("24시간", "정규장만")
MODES = ("묶음", "안묶음")
MARK = {"이김": "+", "짐": "-", "엇갈림": "~"}


# ---------------------------------------------------------------- 자료 준비
def bar_ns(A):
    return int(next(iter(A.values()))["mins"]) * 60 * 10**9          # 평가 격자 = 기준봉 길이


def prep(T, A, p, rg):
    """기간·범위 하나: 규칙별 후보 표(칸 잡는 시각 acq·놓는 시각 rel, 체결 여부, 수익, 평가 경로)."""
    lo, hi = DZ.WIN[p]
    G = bar_ns(A)
    t = T[(T.기간 == p) & (T.범위 == rg) & T.규칙.isin(list(RULES)) & (T.stat != 3)].copy()
    assert t.day.between(lo, hi).all()
    b = t[t.규칙 == BASE]
    assert (b.stat == 0).all()
    btx = dict(zip(zip(b.sym, b.s), b.tx))
    T0 = int(pd.Timestamp(lo).tz_localize("UTC").value) - DAY
    out = {}
    for nm in RULES:
        x = t[t.규칙 == nm]
        x = x[x.stat.isin([0, 1])].reset_index(drop=True)              # stat 2(구역 없음) = 주문 없음 → 칸 안 씀
        fill = (x.stat == 0).values
        rel = np.where(fill, x.tx.values, -1).astype(np.int64)
        if nm != BASE:
            c1 = np.flatnonzero(~fill)
            if "L1" in nm:                                            # L1 취소 = 지금 지표가 팔았을 봉(같은 신호 봉)
                rel[c1] = [btx[(x.sym[i], x.s[i])] for i in c1]
            else:                                                     # L2 취소 = 신호 뒤 첫 한국 아침 8시(상태 기계의 dead)
                rel[c1] = [A[x.sym[i]]["dead"][x.s[i]] for i in c1]
        acq_b = x.ts.values.astype(np.int64)
        acq_u = np.where(fill, x.tfill.values, -1).astype(np.int64)
        assert (rel > acq_b).all() and (x.tx[fill] > x.tfill[fill]).all() and (x.tfill[fill] >= x.ts[fill]).all()
        # 평가 경로: 체결 거래마다 체결 칸 ~ 매도 칸-1의 칸 끝 평가 수익(마지막 종가/매수가 - 1)
        lens = np.zeros(len(x), np.int64)
        pcell, pval = [], []
        cx = np.full(len(x), -1, np.int64)
        for i in np.flatnonzero(fill):
            a = A[x.sym[i]]
            f, xi, ent = int(x.f[i]), int(x.x[i]), float(x.ent[i])
            cells = (a["tk"][f:xi] - T0) // G
            cx[i] = (a["tk"][xi] - T0) // G
            grid = np.arange(cells[0], cx[i])
            pos = np.searchsorted(cells, grid, side="right") - 1      # 봉이 없는 칸(밤)은 앞 종가를 이어 씀
            pcell.append(grid); pval.append(a["c"][f:xi][pos] / ent - 1)
            lens[i] = len(grid)
        off = np.r_[0, np.cumsum(lens)].astype(np.int64)
        out[nm] = dict(sym=x.sym.values, day=x.day.values, fill=fill, ret=np.where(fill, x.ret.values, 0.0),
                       acq_b=acq_b, rel=rel, acq_u=acq_u, cx=cx, off=off,
                       pcell=np.concatenate(pcell).astype(np.int64) if pcell else np.zeros(0, np.int64),
                       pval=np.concatenate(pval) if pval else np.zeros(0))
    days = sorted(set(t.day))
    return T0, G, days, out


def active_cells(A, syms, p, T0, G, ncell):
    """그 묶음 종목 봉이 하나라도 있는 격자 칸(기간 날짜만)."""
    lo, hi = DZ.WIN[p]
    act = np.zeros(ncell, np.bool_)
    for s in syms:
        if s not in A:
            continue
        a = A[s]
        m = (a["day"] >= lo) & (a["day"] <= hi)
        c = (a["tk"][m] - T0) // G
        act[c[(c >= 0) & (c < ncell)]] = True
    return act


def sub(C, m):
    """종목 묶음으로 행을 고르고 평가 경로 오프셋을 다시 만든다."""
    idx = np.flatnonzero(m)
    lens = C["off"][idx + 1] - C["off"][idx]
    off = np.r_[0, np.cumsum(lens)].astype(np.int64)
    take = (np.repeat(C["off"][idx] - off[:-1], lens) + np.arange(off[-1])).astype(np.int64)
    out = {k: C[k][idx] for k in ("sym", "day", "fill", "ret", "acq_b", "rel", "acq_u", "cx")}
    out.update(off=off, pcell=C["pcell"][take], pval=C["pval"][take])
    return out


# ---------------------------------------------------------------- 계좌 시뮬레이션
@njit(cache=True)
def alloc(order, acq, rel, K):
    """우선순위 순서대로 빈 칸(놓는 시각 <= 잡는 시각)이 있으면 받는다."""
    slots = np.full(K, np.iinfo(np.int64).min)
    took = np.zeros(len(acq), np.bool_)
    for e in order:
        t = acq[e]
        for j in range(K):
            if slots[j] <= t:
                slots[j] = rel[e]
                took[e] = True
                break
    return took


@njit(cache=True)
def account(took, fill, ret, cx, off, pcell, pval, acq, rel, dayi, K, ncell, nday, T0, G, actcum, nact):
    """받은 거래로 계좌 지표: 단순 합, 복리, 평가 낙폭, 칸 사용률, 날짜별 손익."""
    R = np.zeros(ncell); U = np.zeros(ncell); D = np.zeros(nday)
    tot, comp, occ = 0.0, 1.0, 0.0
    for e in range(len(took)):
        if not took[e]:
            continue
        # 칸 사용 시간: [acq, rel)과 봉이 있는 칸의 겹침(양 끝 칸은 부분)
        ca, cr = (acq[e] - T0) // G, (rel[e] - T0) // G
        ca = min(max(ca, 0), ncell - 1); cr = min(max(cr, 0), ncell - 1)
        if ca == cr:
            if actcum[ca + 1] - actcum[ca] > 0:
                occ += (rel[e] - acq[e]) / G
        else:
            if actcum[ca + 1] - actcum[ca] > 0:
                occ += (T0 + (ca + 1) * G - acq[e]) / G
            occ += actcum[cr] - actcum[ca + 1]
            if actcum[cr + 1] - actcum[cr] > 0:
                occ += (rel[e] - (T0 + cr * G)) / G
        if not fill[e]:
            continue
        r = ret[e] / K
        tot += r
        comp *= 1.0 + r
        R[cx[e]] += r
        D[dayi[e]] += r
        for q in range(off[e], off[e + 1]):
            U[pcell[q]] += pval[q] / K
    eq = 1.0 + np.cumsum(R) + U
    peak, mdd = 1.0, 0.0
    for c in range(ncell):
        if eq[c] > peak:
            peak = eq[c]
        dd = eq[c] / peak - 1.0
        if dd < mdd:
            mdd = dd
    return tot, comp - 1.0, mdd, occ / (K * nact) if nact > 0 else np.nan, D


def simulate(C, mode, K, seed, T0, G, ncell, ndays, actcum, nact):
    acq = C["acq_b"] if mode == "묶음" else C["acq_u"]
    use = np.ones(len(acq), bool) if mode == "묶음" else C["fill"]
    idx = np.flatnonzero(use)
    rng = np.random.default_rng(seed)
    order = idx[np.lexsort((rng.random(len(idx)), acq[idx]))]      # 시각 순, 같은 시각은 무작위
    took = alloc(order, acq, C["rel"], K)
    took &= use
    tot, comp, mdd, util, D = account(took, C["fill"], C["ret"], C["cx"], C["off"], C["pcell"], C["pval"], acq, C["rel"],
                                      C["dayi"], K, ncell, ndays, T0, G, actcum, nact)
    return dict(계좌=tot, 복리=comp, 낙폭=mdd, 사용률=util, 받음=int(took.sum()), 체결=int((took & C["fill"]).sum()),
                후보=int(use.sum()), 놓침=int(use.sum() - took.sum())), D


# ---------------------------------------------------------------- 한계 점검: 재신호(묶음 방식)
DZ_target = DZ._target                           # P 아래 k번째 구역 주문가(defer_zone과 같은 함수)


@njit(cache=True)
def outcomes(o, l, r, ema, tk, dead_of, reg, hot, sid, ZLO, ZHI, gap, mode, k, life, reg_only, fixd, X):
    """매수선에 닿은 모든 봉 i를 신호로 쳤을 때의 결과 = defer_zone.machine을 봉 i에서 시작한 것.
    행: i, rel(칸 놓는 시각), unb(이 시각 뒤 봉부터 그 종목 새 신호 가능), need(주문 있음), fill, ret, stat."""
    n = len(o)
    I = np.empty(n, np.int64); REL = np.empty(n, np.int64); UNB = np.empty(n, np.int64)
    NEED = np.zeros(n, np.bool_); FILL = np.zeros(n, np.bool_); RET = np.zeros(n); ST = np.empty(n, np.int64)
    m = 0
    for i in range(61, n):
        if hot[i] or (reg_only and not reg[i]):
            continue
        lvl = ema[i - 1] * (1 - gap)
        if l[i] > lvl:
            continue
        P = min(o[i], lvl)
        dead_s = dead_of[i]
        I[m] = i
        f, j = i, i + 1
        L = P
        if mode != 0:
            if mode == 2:
                L = P * (1 - fixd)
            else:
                L = DZ_target(P, ZLO[sid[i]], ZHI[sid[i]], k, 0, False)
            NEED[m] = L == L
            f = -1
            while j < n:                                            # 대기: 취소 확인 → 체결 확인(machine과 같은 순서)
                if life == 1:
                    cancel = r[j - 1] > X or tk[j] >= dead_s
                else:
                    cancel = tk[j] >= dead_s
                if cancel:
                    break
                if L == L and (not reg_only or reg[j]) and l[j] <= L:
                    f = j
                    break
                j += 1
            if f < 0:
                if j >= n:                                          # 자료 끝: 구역 없음은 machine처럼 stat 2(칸 안 씀)로 둔다
                    ST[m] = 2 if L != L else 3
                    REL[m] = UNB[m] = np.iinfo(np.int64).max
                else:
                    ST[m] = 1 if L == L else 2
                    REL[m] = tk[j] if life == 1 else dead_s
                    UNB[m] = tk[j]
                m += 1
                continue
        else:
            NEED[m] = True
        ent = P if mode == 0 else min(o[f], L)
        dead = dead_of[f]
        j = f + 1
        while j < n:                                                # 보유: RSI>70 다음 봉 또는 8시
            if r[j - 1] > X or tk[j] >= dead:
                break
            j += 1
        if j >= n:
            ST[m] = 3
        else:
            ST[m], FILL[m], RET[m], REL[m], UNB[m] = 0, True, o[j] / ent - 1, tk[j], tk[j]
        m += 1
    return I[:m], REL[:m], UNB[:m], NEED[:m], FILL[:m], RET[:m], ST[:m]


@njit(cache=True)
def alloc_rs(order, symi, acq, rel, unb, need, K, free0):
    """재신호 배정. 반환 st: 0 막힘(그 종목이 주문·보유·구역 없음 대기 중) / 1 칸 받음 / 2 구역 없음(칸 없이 대기) / 3 칸 없어 버림."""
    free = free0.copy()
    slots = np.full(K, np.iinfo(np.int64).min)
    st = np.zeros(len(acq), np.int64)
    for e in order:
        s = symi[e]
        if acq[e] <= free[s]:
            continue
        if not need[e]:
            free[s] = unb[e]
            st[e] = 2
            continue
        st[e] = 3
        for j in range(K):
            if slots[j] <= acq[e]:
                slots[j] = rel[e]
                free[s] = unb[e]
                st[e] = 1
                break
    return st


def resignal_cands(p, rg, nm):
    """전체137 후보(기간 안) + 기간 시작 때 종목별 막힘 시각(기간 앞 신호를 상태 기계대로 처리한 결과)."""
    lo, hi = DZ.WIN[p]
    sp, reg_only = SPEC[nm], rg == "정규장만"
    rows, free0, syms = [], [], []
    for si, (s, (A, Z, sid)) in enumerate(DZ.ctx(p).items()):
        gap = 0.03 if s in DZ.LEV else 0.02
        I, REL, UNB, NEED, FILL, RET, ST = outcomes(A["o"], A["l"], A["r"], A["ema"], A["tk"], A["dead"], A["reg"], A["hot7"],
                                                    sid, Z["ZLO"], Z["ZHI"], gap, sp["mode"], sp["k"], sp["life"], reg_only,
                                                    sp["fixd"], SL.X)
        day = A["day"][I]
        pre = day < lo
        fr = np.iinfo(np.int64).min
        for e in np.flatnonzero(pre):                               # 기간 앞: 칸 제약 없는 상태 기계
            if A["tk"][I[e]] > fr and ST[e] != 3:
                fr = UNB[e]
        free0.append(fr); syms.append(s)
        m = (day >= lo) & (day <= hi) & (ST != 3)
        rows.append(pd.DataFrame(dict(symi=si, sym=s, s=I[m], acq=A["tk"][I[m]], rel=REL[m], unb=UNB[m], need=NEED[m],
                                      fill=FILL[m], ret=RET[m], stat=ST[m], day=day[m])))
    return pd.concat(rows, ignore_index=True), np.array(free0, np.int64), syms


def run_resignal(T):
    """재신호 계좌(묶음). K 무한 재현 점검 뒤 KMAIN x 무작위 50번."""
    out = []
    for p in PERIODS:
        for rg in RANGES:
            for nm in RULES:
                X, free0, syms = resignal_cands(p, rg, nm)
                # K 무한 = 신호표 그대로(같은 신호 봉·같은 stat·같은 수익)
                st = alloc_rs(np.lexsort((X.s.values, X.acq.values)), X.symi.values, X.acq.values, X.rel.values, X.unb.values,
                              X.need.values, 10**6, free0)
                got = X[st > 0].set_index(["sym", "s"]).sort_index()
                ref = T[(T.기간 == p) & (T.범위 == rg) & (T.규칙 == nm) & (T.stat != 3)].set_index(["sym", "s"]).sort_index()
                assert got.index.equals(ref.index) and (got.stat.values == ref.stat.values).all(), (p, rg, nm)
                assert np.allclose(got.ret.values[got.fill.values], ref.ret.values[ref.stat.values == 0]), (p, rg, nm)
                for un, us in DZ.UNIV.items():
                    m = X.sym.isin(us).values
                    Y = X[m]
                    acq, rel, unb, need, fill = (Y[c].values for c in ("acq", "rel", "unb", "need", "fill"))
                    ret = Y.ret.values
                    for K in KMAIN:
                        for sd in range(NSEED):
                            rng = np.random.default_rng(1000 * sd + K)
                            order = np.lexsort((rng.random(len(Y)), acq))
                            st = alloc_rs(order, Y.symi.values, acq, rel, unb, need, K, free0)
                            took = st == 1
                            out.append(dict(기간=p, 범위=rg, 종목=un, 규칙=RULES[nm], K=K, 무작위=sd,
                                            계좌=ret[took & fill].sum() / K, 받음=int(took.sum()), 체결=int((took & fill).sum()),
                                            버림=int((st == 3).sum())))
            print("재신호", p, rg, "완료", flush=True)
    return pd.DataFrame(out)


# ---------------------------------------------------------------- 실행·보고
def run_all(T):
    runs, daily, mech = [], {}, []
    for p in PERIODS:
        A = DZ.load_period(p)
        for rg in RANGES:
            T0, G, days, CC = prep(T, A, p, rg)
            dpos = {d: i for i, d in enumerate(days)}
            ncell = int(max(C["rel"].max() for C in CC.values()) - T0) // G + 2
            for un, us in DZ.UNIV.items():
                act = active_cells(A, us, p, T0, G, ncell)
                actcum = np.r_[0, np.cumsum(act)].astype(np.int64)
                nact = int(act.sum())
                for nm, C0 in CC.items():
                    C = sub(C0, np.isin(C0["sym"], list(us)))
                    C["dayi"] = np.array([dpos[d] for d in C["day"]], np.int64)
                    mech.append(mechanism(C, p, rg, un, nm))
                    for mode in (("묶음",) if nm == BASE else MODES):
                        for K in KS:
                            Ds = []
                            for sd in range(NSEED):
                                r, D = simulate(C, mode, K, 1000 * sd + K, T0, G, ncell, len(days), actcum, nact)
                                runs.append(dict(기간=p, 범위=rg, 종목=un, 규칙=RULES[nm], 방식=mode, K=K, 무작위=sd, **r))
                                Ds.append(D)
                            daily[(p, rg, un, RULES[nm], mode, K)] = np.mean(Ds, axis=0)
            print(p, rg, "완료", flush=True)
    return pd.DataFrame(runs), daily, pd.DataFrame(mech)


def mechanism(C, p, rg, un, nm):
    """칸 제약이 없을 때 주문 1건이 칸을 잡는 시간과 칸-시간당 수익(포화 계좌의 수익 = 쓸 수 있는 칸-시간 x 이 값)."""
    f = C["fill"]
    hb = (C["rel"] - C["acq_b"]) / 3.6e12
    hu = (C["rel"][f] - C["acq_u"][f]) / 3.6e12
    return dict(기간=p, 범위=rg, 종목=un, 규칙=RULES[nm], 주문=len(f), 체결=int(f.sum()),
                체결평균=C["ret"][f].mean() if f.any() else np.nan,
                주문당시간_묶음=hb.mean(), 칸시간당_묶음=C["ret"][f].sum() / hb.sum() * 100 if len(hb) else np.nan,
                체결당시간_안묶음=hu.mean() if f.any() else np.nan,
                칸시간당_안묶음=C["ret"][f].sum() / hu.sum() * 100 if f.any() else np.nan)


def summarize(R, daily):
    key = ["기간", "범위", "종목", "규칙", "방식", "K"]
    S = R.groupby(key).agg(계좌=("계좌", "mean"), 계좌p5=("계좌", lambda v: v.quantile(.05)), 계좌p95=("계좌", lambda v: v.quantile(.95)),
                           복리=("복리", "mean"), 낙폭=("낙폭", "mean"), 낙폭p5=("낙폭", lambda v: v.quantile(.05)),
                           사용률=("사용률", "mean"), 받음=("받음", "mean"), 체결=("체결", "mean"), 놓침=("놓침", "mean")).reset_index()
    # 지금 지표와 비교(같은 기간·범위·종목·K, 무작위 번호끼리 짝)
    b = R[R.규칙 == "지금"][["기간", "범위", "종목", "K", "무작위", "계좌"]].rename(columns={"계좌": "b"})
    J = R.merge(b, on=["기간", "범위", "종목", "K", "무작위"])
    W = J.assign(w=J.계좌 > J.b, d=J.계좌 - J.b).groupby(key).agg(이길확률=("w", "mean"), 차이=("d", "mean")).reset_index()
    S = S.merge(W, on=key)
    tt = []
    for r in S.itertuples():
        dd = daily[(r.기간, r.범위, r.종목, r.규칙, r.방식, r.K)] - daily[(r.기간, r.범위, r.종목, "지금", "묶음", r.K)]
        sd = dd.std(ddof=1)
        tt.append(dd.mean() / (sd / np.sqrt(len(dd))) if sd > 0 and len(dd) > 1 else 0.0)
    S["t"] = tt
    return S


def verdict(e, l):
    return "이김" if e > 0 and l > 0 else ("짐" if e < 0 and l < 0 else "엇갈림")


def fmt(df, cols):
    f = df[cols].copy()
    for k, fm in (("계좌", "{:+.1%}"), ("복리", "{:+.1%}"), ("낙폭", "{:+.1%}"), ("낙폭p5", "{:+.1%}"),
                  ("사용률", "{:.0%}"), ("받음", "{:.0f}"), ("체결", "{:.0f}"), ("놓침", "{:.0f}"), ("차이", "{:+.1%}"),
                  ("이길확률", "{:.0%}"), ("t", "{:+.2f}"), ("체결평균", "{:+.2%}"), ("주문당시간_묶음", "{:.1f}"),
                  ("칸시간당_묶음", "{:+.3f}"), ("체결당시간_안묶음", "{:.1f}"), ("칸시간당_안묶음", "{:+.3f}")):
        if k in f:
            f[k] = f[k].map(lambda v: "" if pd.isna(v) else fm.format(v))
    return f.to_string(index=False)


def main():
    pd.set_option("display.width", 250)
    T = pd.read_parquet(TRADES)
    R, daily, M = run_all(T)
    R.to_csv(os.path.join(TMP, "defer_zone_port_runs.csv"), index=False, encoding="utf-8-sig")
    S = summarize(R, daily)
    S.to_csv(os.path.join(TMP, "defer_zone_port_summary.csv"), index=False, encoding="utf-8-sig")
    S["5~95%"] = [f"{a:+.1%}~{b:+.1%}" for a, b in zip(S.계좌p5, S.계좌p95)]
    RS = run_resignal(T)
    RS.to_csv(os.path.join(TMP, "defer_zone_port_resignal.csv"), index=False, encoding="utf-8-sig")
    get = lambda **q: S.loc[np.logical_and.reduce([S[k] == v for k, v in q.items()])].iloc[0]
    others = [v for v in RULES.values() if v != "지금"]

    L = []
    w = L.append
    w("이격매수 '지지 구역 아래에서 사기' — 자금 한도 계좌 시험 (2026-10-10, defer_zone_port.py)")
    w(f"입력: defer_zone.py 신호표. 계좌 K칸, 같은 시각 신호는 무작위 우선순위 {NSEED}번. 수익 = 받은 거래 수익 합/K(칸 크기 고정, 재투자 안 함).")
    w("묶음 = 지정가 대기 주문도 칸을 잡음(신호~취소/매도), 안묶음 = 체결 때 빈 칸이 있어야 체결(대기는 칸 안 씀). 지금 지표는 둘이 같다.")
    w("칸: 계좌 = 무작위 평균 계좌 수익, 5~95% = 무작위 분포, 복리 = 매도마다 계좌 x(1+수익/K), 낙폭 = 기준봉 평가 최대 낙폭 평균,")
    w("    낙폭p5 = 무작위 중 나쁜 5% 낙폭, 받음 = 받은 신호(묶음: 건 주문, 안묶음: 받은 체결), 체결 = 그중 체결, 놓침 = 칸이 없어 버린 후보,")
    w("    사용률 = 봉 있는 시간 동안 잡힌 칸 비율, 차이 = 계좌 - 같은 K 지금 지표 계좌, 이길확률 = 무작위 짝 중 변형이 이긴 비율,")
    w("    t = 날짜별 손익 차이(무작위 평균)의 날짜 군집 t. oos는 3일이라 판단 근거로 쓰지 않는다.")
    w("검증: K 무한이면 계좌 x K = defer_zone '합'과 같음, 종목별 주문·보유 겹침 없음, 미체결 취소 전 주문가 닿은 봉 없음,")
    w("      재신호 후보로 K 무한을 돌리면 신호표와 같은 신호·stat·수익(scratchpad verify_port.py, 이 스크립트 안 assert).")

    # 요약: 판정 기호 표
    w("")
    w("=" * 20 + " 요약: 같은 K 지금 지표 대비 판정(early·late 둘 다 이기면 +, 둘 다 지면 -, 엇갈리면 ~). 글자 순서 K = " + "·".join(map(str, KS)))
    rows = []
    for nm in others:
        row = dict(규칙=nm)
        for un in DZ.UNIV:
            for rg in RANGES:
                for mode in MODES:
                    row[f"{un}·{rg[:2]}·{mode}"] = "".join(
                        MARK[verdict(get(기간="early", 범위=rg, 종목=un, 규칙=nm, 방식=mode, K=K).차이,
                                     get(기간="late", 범위=rg, 종목=un, 규칙=nm, 방식=mode, K=K).차이)] for K in KS)
        rows.append(row)
    Z = pd.DataFrame(rows).set_index("규칙").T
    w(Z.to_string())

    # 0) 구조
    w("")
    w("=" * 20 + " 0) 구조: 칸 제약 없을 때 주문 1건의 칸 점유 시간(시간)과 '칸-시간당 수익'(%/시간). 포화 계좌 수익 ≈ 쓸 수 있는 칸-시간 x 이 값")
    w("   묶음 = 신호~취소/매도 시간(미체결 주문 포함), 안묶음 = 체결~매도 시간. 시계 시간(밤 포함).")
    mc = ["규칙", "주문", "체결", "체결평균", "주문당시간_묶음", "칸시간당_묶음", "체결당시간_안묶음", "칸시간당_안묶음"]
    for un in DZ.UNIV:
        for p in PERIODS:
            for rg in RANGES:
                w(f"\n[{un} · {DZ.PN[p]} · {rg}]")
                w(fmt(M[(M.종목 == un) & (M.기간 == p) & (M.범위 == rg)], mc))

    # 1) 판정표
    w("")
    w("=" * 20 + " 1) 판정: 같은 K 지금 지표 대비 계좌 수익 차이(%p). 판정 = early·late 둘 다 같은 방향일 때만 이김/짐, 아니면 엇갈림")
    w("   (K 2·40은 경계 참고. t는 late 기준)")
    for un in DZ.UNIV:
        for rg in RANGES:
            for mode in MODES:
                rows = []
                for nm in others:
                    for K in KS:
                        e, l, o = (get(기간=p, 범위=rg, 종목=un, 규칙=nm, 방식=mode, K=K) for p in PERIODS)
                        be, bl = (get(기간=p, 범위=rg, 종목=un, 규칙="지금", K=K) for p in ("early", "late"))
                        rows.append(dict(규칙=nm, K=K, early지금=f"{be.계좌:+.1%}", early변형=f"{e.계좌:+.1%}", early차이=f"{e.차이:+.1%}",
                                         early이길=f"{e.이길확률:.0%}", late지금=f"{bl.계좌:+.1%}", late변형=f"{l.계좌:+.1%}",
                                         late차이=f"{l.차이:+.1%}", late이길=f"{l.이길확률:.0%}", late_t=f"{l.t:+.2f}",
                                         oos차이=f"{o.차이:+.1%}", 판정=verdict(e.차이, l.차이)))
                w(f"\n[{un} · {rg} · {mode}]")
                w(pd.DataFrame(rows).to_string(index=False))

    # 2) 자세히
    w("")
    w("=" * 20 + " 2) 자세히: 기간·범위·방식별 K x 규칙")
    cols = ["규칙", "K", "계좌", "5~95%", "복리", "낙폭", "낙폭p5", "받음", "체결", "놓침", "사용률", "차이", "이길확률", "t"]
    order = {v: i for i, v in enumerate(RULES.values())}
    for un in DZ.UNIV:
        for p in PERIODS:
            for rg in RANGES:
                for mode in MODES:
                    x = S[(S.종목 == un) & (S.기간 == p) & (S.범위 == rg) & ((S.방식 == mode) | (S.규칙 == "지금")) & S.K.isin(KMAIN)]
                    x = x.assign(_o=x.규칙.map(order)).sort_values(["K", "_o"])
                    w(f"\n[{un} · {DZ.PN[p]} · {rg} · {mode}]")
                    w(fmt(x, cols))

    # 3) 재신호 점검
    w("")
    w("=" * 20 + " 3) 한계 점검: 재신호 계좌(묶음). 칸이 없어 버린 종목은 다음 봉부터 매수선 닿으면 다시 신호(구역·주문가도 그 봉에서 새로)")
    w("   표 = 위 신호표 계좌, 재 = 재신호 계좌(무작위 50번 평균). 차이 = 변형 - 같은 모형·같은 K 지금 지표. 판정은 재신호 early·late 기준.")
    G2 = RS.groupby(["기간", "범위", "종목", "규칙", "K"]).agg(계좌=("계좌", "mean"), 받음=("받음", "mean"), 버림=("버림", "mean")).reset_index()
    g2 = lambda **q: G2.loc[np.logical_and.reduce([G2[k] == v for k, v in q.items()])].iloc[0]
    for un in DZ.UNIV:
        for rg in RANGES:
            rows = []
            for nm in RULES.values():
                for K in KMAIN:
                    row = dict(규칙=nm, K=K)
                    dif = {}
                    for p in ("early", "late"):
                        tb = get(기간=p, 범위=rg, 종목=un, 규칙=nm, 방식="묶음", K=K)
                        r2, b2 = g2(기간=p, 범위=rg, 종목=un, 규칙=nm, K=K), g2(기간=p, 범위=rg, 종목=un, 규칙="지금", K=K)
                        dif[p] = r2.계좌 - b2.계좌
                        row.update({f"{p}표": f"{tb.계좌:+.1%}", f"{p}재": f"{r2.계좌:+.1%}", f"{p}재받음": f"{r2.받음:.0f}",
                                    f"{p}재차이": "" if nm == "지금" else f"{dif[p]:+.1%}"})
                    row["재판정"] = "" if nm == "지금" else verdict(dif["early"], dif["late"])
                    rows.append(row)
            w(f"\n[{un} · {rg}]")
            w(pd.DataFrame(rows).to_string(index=False))
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("저장:", OUT)


if __name__ == "__main__":
    main()
