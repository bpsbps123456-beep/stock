"""이격매수 혼합(본장 신호만 지지 구역 하단에서 사기) — 자금 한도 계좌 시험(2026-10-10, 키 defer_zone_hybrid_port).

질문(사용자 10/10): "이 매수 방법은 본장에서만 하면 어때? 큰 하락은 주로 본장에서 나오거든."
  defer_zone_hybrid.py는 모든 신호를 다 산다고 치고(칸 무한) 혼합 규칙이 지금 지표보다 낫지 않다고 봤다. 실제 계좌는 칸이
  한정돼 있어 본장 신호를 미루면 (1) 대기 지정가가 칸을 묶어 손해일 수도, (2) 미체결 주문이 본장 마감에 풀려 건당 수익이 큰
  애프터 신호에 칸을 돌려줘 이득일 수도 있다. 자금 K칸 계좌에서 혼합 규칙이 지금 지표를 이기는가.

입력: defer_zone_hybrid.py 신호표(scratchpad defer_zone_hybrid_trades.parquet, 24시간 매매). 비교 규칙 6개(+ 아래 대조군):
  지금 지표, 혼합 Z1·Z2 하단 Lr(본장 마감까지), 혼합 Z2 하단 L1(원래 매도까지), 혼합 고정 -3% Lr(같은 깊이 대조),
  프리·본장 Z2 하단 L2(아침 8시까지, 프리장 신호도 미룸). 종목 사용자78·전체137·데스크7, 기간 early·late(oos는 참고).
계좌 모형: defer_zone_port.py와 같다(그 파일의 bar_ns·sub·active_cells·alloc·account·summarize·alloc_rs·verdict·fmt를 import).
  K칸(2·3·5·10·20), 한 칸 = 포지션 또는 대기 주문 하나, 빈 칸이 있을 때만 신호를 받고 같은 시각 신호는 무작위 우선순위 50번.
  묶음(기본): 대기 지정가도 주문~취소/매도까지 칸을 잡는다. 안묶음(비교): 대기는 칸을 안 쓰고 체결 때 빈 칸이 있어야 체결.
  P에 바로 사는 신호(혼합의 프리·애프터 신호, 지금 지표 전부)는 신호~매도. 같은 시각에는 매도·취소가 먼저 칸을 비운다.
  구역이 없어 주문을 안 건 신호(stat 2)는 칸을 안 쓴다. 기간 끝 미완료(stat 3)는 버린다.
  수익 = 받은 거래 수익 합/K(칸 크기 고정, 단순 합), 복리 = 매도마다 계좌 x(1+수익/K), 낙폭 = 기준봉 평가 최대 낙폭,
  t = 신호 ET 날짜별 손익(무작위 50번 평균) 차이의 날짜 군집 t, 이길확률 = 같은 번호 무작위 짝 중 변형 계좌 > 지금 지표 계좌 비율.
미체결 주문이 칸을 놓는 시각
  Lr(새로 정함): 신호 뒤 첫 정규장 밖 봉 시각과 그날 16:00 ET 중 이른 쪽(정규장 지정가는 장 마감에 만료. 애프터 봉이 빠진 날도 마감에 풀림).
  L1: 지금 지표가 같은 신호를 팔았을 봉 시각(신호표의 지금 지표 매도 시각과 같음을 assert). L2: 신호 뒤 첫 한국 아침 8시(defer_zone_port와 같음).
검증(이 스크립트 안 assert): 미체결 주문의 취소 봉을 상태 기계대로 다시 찾아 그 전 봉에 주문가 닿음 없음, 종목별 칸 겹침 없음,
  K 무한이면 계좌 x K = defer_zone_hybrid '합'(두 방식 모두), 재신호 후보로 K 무한을 돌리면 신호표와 같은 신호·stat·수익.
표
  요약 판정 기호(계좌·낙폭), 0) 구조(칸-시간당 수익, 신호 세션별), 1) 판정(계좌·이길확률·t), 2) 낙폭, 3) 신호 세션별 계좌 손익,
  4) 자세히, 5) 한계 점검: 재신호 계좌, 6) 참고: 사용자가 깨어 있는 시간(한국 08~24시)에 나온 신호 몫.
대조군: 정규장 신호를 아예 안 사는 규칙(정규안삼, 칸도 안 씀)을 더해 혼합의 계좌 이득이 구역 매수 때문인지 칸 재배치 때문인지 가른다(표 1b).
한계: defer_zone_port.py와 같다(같은 시각이 아닌 신호는 먼저 온 순서, 주문 1건 = 1칸, 수수료·슬리피지 없음). oos는 3일이고
  10/7·10/8 애프터 봉 결측이 많아 판단 근거로 쓰지 않는다.
사용: py defer_zone_hybrid_port.py → results/support/defer_zone_hybrid_port.txt,
      scratchpad defer_zone_hybrid_port_runs.csv(설정·무작위별), defer_zone_hybrid_port_summary.csv, defer_zone_hybrid_port_resignal.csv
"""
import os, sys
import numpy as np, pandas as pd
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, CODE)
sys.path.insert(0, HERE)
import stop_lab as SL
import defer_zone as DZ
import defer_zone_hybrid as DH                   # 혼합 규칙 설정(HYB·FIX)
import defer_zone_port as DP                     # 계좌 모형(칸 배정·평가·요약)을 그대로 쓴다

OUT = os.path.normpath(os.path.join(CODE, "../results/support/defer_zone_hybrid_port.txt"))
TMP = DZ.TMP
TRADES = os.path.join(TMP, "defer_zone_hybrid_trades.parquet")
METRICS = os.path.join(TMP, "defer_zone_hybrid_metrics.csv")
BASE = "지금 지표(신호 때 삼)"
RULES = {BASE: "지금",
         "혼합 Z1 하단, Lr(본장 마감까지)": "혼합Z1 Lr",
         "혼합 Z2 하단, Lr(본장 마감까지)": "혼합Z2 Lr",
         "혼합 Z2 하단, L1(원래 매도까지)": "혼합Z2 L1",
         "혼합 고정 -3%, Lr": "혼합-3% Lr",
         "프리·본장 Z2 하단, L2(아침 8시까지)": "프리본장Z2 L2",
         "대조: 정규장 신호 안 삼(마감까지 그 종목 쉼)": "정규안삼"}
# 대조군: 정규장 신호는 사지 않고 칸도 안 쓴다(혼합 Lr의 '못 닿는 주문'과 같은 상태 기계: 주문가 0, 마감에 취소 → stat 2로 바꿈).
#   혼합의 계좌 이득이 '구역에서 싸게 산 것'인지 '정규장 신호를 덜 사서 칸이 애프터 신호로 간 것'인지 가른다.
SKIP = "대조: 정규장 신호 안 삼(마감까지 그 종목 쉼)"
SKIP_RUN = dict(mode=2, fixd=1.0, life=4, hyb=True)
_ALLSPEC = {**DH.HYB, **DH.FIX, SKIP: SKIP_RUN}
SPEC = {BASE: dict(mode=0, k=0, life=0, fixd=0.0, dfr=("정규",), skip=False)}
SPEC.update({nm: dict(mode=_ALLSPEC[nm]["mode"], k=_ALLSPEC[nm].get("k", 0), life=_ALLSPEC[nm]["life"],
                      fixd=_ALLSPEC[nm].get("fixd", 0.0), dfr=tuple(_ALLSPEC[nm].get("dfr_ses", ("정규",))), skip=nm == SKIP)
             for nm in RULES if nm != BASE})
KS = (2, 3, 5, 10, 20)
NSEED = 50
DAY = 86_400 * 10**9
MIN = 60 * 10**9
PERIODS = ("early", "late", "oos")
RG = "24시간"
MODES = ("묶음", "안묶음")
SESS = ("프리", "정규", "애프터")
MARK = {"이김": "+", "짐": "-", "엇갈림": "~"}


# ---------------------------------------------------------------- 자료 준비
@njit(cache=True)
def cancel_bar(r, tk, reg, dead_s, s, life, X):
    """미체결 주문이 취소되는 봉(defer_zone.machine state 1의 취소 조건 그대로). 없으면 -1."""
    for j in range(s + 1, len(tk)):
        if life == 1:
            if r[j - 1] > X or tk[j] >= dead_s:
                return j
        elif life == 4:
            if not reg[j]:
                return j
        elif tk[j] >= dead_s:
            return j
    return -1


def lr_close(a, s):
    """신호 봉 s가 속한 날 16:00 ET(정규장 지정가 만료 시각). mod = ET 0시부터 분."""
    return int(a["tk"][s] + (960 - int(a["mod"][s])) * MIN)


def prep_h(T, A, p):
    """기간 하나: 규칙별 후보 표(칸 잡는 시각 acq·놓는 시각 rel, 체결 여부, 수익, 신호 세션, 평가 경로)."""
    lo, hi = DZ.WIN[p]
    G = DP.bar_ns(A)
    t = T[(T.기간 == p) & (T.범위 == RG) & T.규칙.isin(list(RULES)) & (T.stat != 3)].copy()
    assert t.day.between(lo, hi).all()
    b = t[t.규칙 == BASE]
    assert (b.stat == 0).all()
    btx = dict(zip(zip(b.sym, b.s), b.tx))
    T0 = int(pd.Timestamp(lo).tz_localize("UTC").value) - DAY
    out, chk = {}, []
    for nm in RULES:
        sp = SPEC[nm]
        x = t[t.규칙 == nm]
        x = x[x.stat.isin([0, 1])].reset_index(drop=True)              # stat 2(구역 없음) = 주문 없음 → 칸 안 씀
        fill = (x.stat == 0).values
        rel = np.where(fill, x.tx.values, -1).astype(np.int64)
        ncap = 0
        for i in np.flatnonzero(~fill):
            a, s = A[x.sym[i]], int(x.s[i])
            j = cancel_bar(a["r"], a["tk"], a["reg"], a["dead"][s], s, sp["life"], SL.X)
            assert j > s, (p, nm, x.sym[i], s)
            assert not (a["l"][s + 1:j] <= x.L[i]).any(), (p, nm, x.sym[i], s)   # 취소 전에 주문가 닿은 봉 없음
            if sp["life"] == 1:                                         # L1 = 지금 지표가 팔았을 봉
                rel[i] = a["tk"][j]
                assert btx[(x.sym[i], s)] == rel[i], (p, nm, x.sym[i], s)
            elif sp["life"] == 4:                                       # Lr = 첫 정규장 밖 봉과 16:00 ET 중 이른 쪽
                assert a["reg"][s]
                c = lr_close(a, s)
                ncap += c < a["tk"][j]
                rel[i] = min(int(a["tk"][j]), c)
            else:                                                       # L2 = 신호 뒤 첫 한국 아침 8시
                rel[i] = a["dead"][s]
        acq_b = x.ts.values.astype(np.int64)
        acq_u = np.where(fill, x.tfill.values, -1).astype(np.int64)
        assert (rel > acq_b).all() and (x.tx[fill] > x.tfill[fill]).all() and (x.tfill[fill] >= x.ts[fill]).all()
        # 종목별 칸 겹침 없음(다음 주문 시각 >= 앞 주문 놓는 시각)
        sy = x.sym.to_numpy(dtype=str)
        o = np.lexsort((acq_b, sy))
        same = sy[o][1:] == sy[o][:-1]
        assert (acq_b[o][1:][same] >= rel[o][:-1][same]).all(), (p, nm)
        chk.append(dict(기간=p, 규칙=RULES[nm], 미체결=int((~fill).sum()), 마감캡=ncap))
        # 평가 경로: 체결 거래마다 체결 칸 ~ 매도 칸-1의 칸 끝 평가 수익(마지막 종가/매수가 - 1) — defer_zone_port.prep와 같은 계산
        lens = np.zeros(len(x), np.int64)
        pcell, pval = [], []
        cx = np.full(len(x), -1, np.int64)
        for i in np.flatnonzero(fill):
            a = A[x.sym[i]]
            f, xi, ent = int(x.f[i]), int(x.x[i]), float(x.ent[i])
            cells = (a["tk"][f:xi] - T0) // G
            cx[i] = (a["tk"][xi] - T0) // G
            grid = np.arange(cells[0], cx[i])
            pos = np.searchsorted(cells, grid, side="right") - 1
            pcell.append(grid); pval.append(a["c"][f:xi][pos] / ent - 1)
            lens[i] = len(grid)
        off = np.r_[0, np.cumsum(lens)].astype(np.int64)
        out[nm] = dict(sym=x.sym.to_numpy(dtype=str), day=x.day.values, fill=fill, ret=np.where(fill, x.ret.values, 0.0),
                       acq_b=acq_b, rel=rel, acq_u=acq_u, cx=cx, off=off, ses=x.ses.to_numpy(dtype=str), slp=sleep_of(acq_b),
                       pcell=np.concatenate(pcell).astype(np.int64) if pcell else np.zeros(0, np.int64),
                       pval=np.concatenate(pval) if pval else np.zeros(0))
    days = sorted(set(t.day))
    return T0, G, days, out, chk


def skip_table():
    """대조군 신호표(defer_zone.run, 주문가 0 = 절대 체결 안 됨). 미체결 취소(stat 1)를 '주문 없음'(stat 2)으로 바꾼다."""
    out = []
    for p in PERIODS:
        t = DZ.run(p, SKIP_RUN, False)
        assert not ((t.ses == "정규") & (t.stat == 0)).any() and (t[t.ses != "정규"].stat != 1).all()
        t.loc[t.stat == 1, "stat"] = 2
        t["기간"], t["범위"], t["규칙"] = p, RG, SKIP
        out.append(t)
    return pd.concat(out, ignore_index=True)


def sleep_of(ts):
    """신호 시각이 사용자가 자는 한국 00~08시인가(docs/desk_system.md §1 주문 가능 시간 08~24시). UTC+9 고정."""
    return (ts // 3_600_000_000_000 + 9) % 24 < 8


def sub_h(C, m):
    out = DP.sub(C, m)
    out["ses"], out["slp"] = C["ses"][m], C["slp"][m]
    return out


# ---------------------------------------------------------------- 계좌 시뮬레이션(defer_zone_port.simulate + 신호 세션별 손익)
def simulate(C, mode, K, seed, T0, G, ncell, ndays, actcum, nact):
    acq = C["acq_b"] if mode == "묶음" else C["acq_u"]
    use = np.ones(len(acq), bool) if mode == "묶음" else C["fill"]
    idx = np.flatnonzero(use)
    rng = np.random.default_rng(seed)
    order = idx[np.lexsort((rng.random(len(idx)), acq[idx]))]      # 시각 순, 같은 시각은 무작위
    took = DP.alloc(order, acq, C["rel"], K) & use
    tot, comp, mdd, util, D = DP.account(took, C["fill"], C["ret"], C["cx"], C["off"], C["pcell"], C["pval"], acq, C["rel"],
                                         C["dayi"], K, ncell, ndays, T0, G, actcum, nact)
    r = dict(계좌=tot, 복리=comp, 낙폭=mdd, 사용률=util, 받음=int(took.sum()), 체결=int((took & C["fill"]).sum()),
             후보=int(use.sum()), 놓침=int(use.sum() - took.sum()))
    tf = took & C["fill"]
    for s in SESS:
        m = tf & (C["ses"] == s)
        r[f"{s}손익"] = C["ret"][m].sum() / K
        r[f"{s}체결"] = int(m.sum())
    r["자는중손익"] = C["ret"][tf & C["slp"]].sum() / K
    return r, D


def mechanism(C, p, un, nm):
    """칸 제약이 없을 때 주문 1건이 칸을 잡는 시간과 칸-시간당 수익(전체와 신호 세션별)."""
    rows = []
    for s in ("전체",) + SESS:
        m = np.ones(len(C["fill"]), bool) if s == "전체" else C["ses"] == s
        f = C["fill"] & m
        hb = (C["rel"][m] - C["acq_b"][m]) / 3.6e12
        hu = (C["rel"][f] - C["acq_u"][f]) / 3.6e12
        rows.append(dict(기간=p, 종목=un, 규칙=RULES[nm], 세션=s, 주문=int(m.sum()), 체결=int(f.sum()),
                         합=C["ret"][f].sum(), 체결평균=C["ret"][f].mean() if f.any() else np.nan,
                         주문당시간_묶음=hb.mean() if len(hb) else np.nan,
                         칸시간당_묶음=C["ret"][f].sum() / hb.sum() * 100 if hb.sum() > 0 else np.nan,
                         체결당시간_안묶음=hu.mean() if f.any() else np.nan,
                         칸시간당_안묶음=C["ret"][f].sum() / hu.sum() * 100 if f.any() and hu.sum() > 0 else np.nan))
    return rows


def run_all(T, M):
    runs, daily, mech, chks = [], {}, [], []
    for p in PERIODS:
        A = DZ.load_period(p)
        T0, G, days, CC, chk = prep_h(T, A, p)
        chks += chk
        dpos = {d: i for i, d in enumerate(days)}
        ncell = int(max(C["rel"].max() for C in CC.values()) - T0) // G + 2
        for un, us in DZ.UNIV.items():
            act = DP.active_cells(A, us, p, T0, G, ncell)
            actcum = np.r_[0, np.cumsum(act)].astype(np.int64)
            nact = int(act.sum())
            for nm, C0 in CC.items():
                C = sub_h(C0, np.isin(C0["sym"], list(us)))
                C["dayi"] = np.array([dpos[d] for d in C["day"]], np.int64)
                mech += mechanism(C, p, un, nm)
                tt = T[(T.기간 == p) & (T.규칙 == nm) & (T.stat == 0) & T.sym.isin(us)]
                ref = tt.ret.sum()
                mm = M[(M.기간 == p) & (M.종목 == un) & (M.규칙 == nm)]
                if len(mm):
                    assert abs(mm.합.iloc[0] - ref) < 1e-9, (p, un, nm)
                else:
                    assert nm == SKIP
                for mode in (("묶음",) if nm == BASE else MODES):
                    r, _ = simulate(C, mode, 100_000, 1, T0, G, ncell, len(days), actcum, nact)   # K 무한 = 신호표 합
                    assert abs(r["계좌"] * 100_000 - ref) < 1e-6, (p, un, nm, mode, r["계좌"] * 100_000, ref)
                    for K in KS:
                        Ds = []
                        for sd in range(NSEED):
                            r, D = simulate(C, mode, K, 1000 * sd + K, T0, G, ncell, len(days), actcum, nact)
                            runs.append(dict(기간=p, 범위=RG, 종목=un, 규칙=RULES[nm], 방식=mode, K=K, 무작위=sd, **r))
                            Ds.append(D)
                        daily[(p, RG, un, RULES[nm], mode, K)] = np.mean(Ds, axis=0)
        print(p, "완료", flush=True)
    return pd.DataFrame(runs), daily, pd.DataFrame(mech), pd.DataFrame(chks)


# ---------------------------------------------------------------- 한계 점검: 재신호(묶음 방식)
DZ_target = DZ._target


@njit(cache=True)
def outcomes_h(o, l, r, ema, tk, dead_of, reg, hot, mod, sid, ZLO, ZHI, gap, mode, k, life, fixd, X, dfr):
    """매수선에 닿은 모든 봉 i를 신호로 쳤을 때의 결과 = defer_zone.machine(hyb)을 봉 i에서 시작한 것(24시간).
    dfr[i]가 참인 봉의 신호만 구역·깊이 지정가, 나머지는 P에 삼. 행: i, rel, unb, need, fill, ret, stat(defer_zone_port.outcomes와 같은 뜻)."""
    n = len(o)
    I = np.empty(n, np.int64); REL = np.empty(n, np.int64); UNB = np.empty(n, np.int64)
    NEED = np.zeros(n, np.bool_); FILL = np.zeros(n, np.bool_); RET = np.zeros(n); ST = np.empty(n, np.int64)
    BIG = np.iinfo(np.int64).max
    m = 0
    for i in range(61, n):
        if hot[i]:
            continue
        lvl = ema[i - 1] * (1 - gap)
        if l[i] > lvl:
            continue
        P = min(o[i], lvl)
        dead_s = dead_of[i]
        I[m] = i
        f, j = i, i + 1
        L = P
        defer = mode != 0 and dfr[i]
        if defer:
            if mode == 2:
                L = P * (1 - fixd)
            else:
                L = DZ_target(P, ZLO[sid[i]], ZHI[sid[i]], k, 0, False)
            NEED[m] = L == L
            f = -1
            while j < n:                                            # 대기: 취소 확인 → 체결 확인(machine과 같은 순서)
                if life == 1:
                    cancel = r[j - 1] > X or tk[j] >= dead_s
                elif life == 4:
                    cancel = not reg[j]
                else:
                    cancel = tk[j] >= dead_s
                if cancel:
                    break
                if L == L and l[j] <= L:
                    f = j
                    break
                j += 1
            if f < 0:
                if j >= n:
                    ST[m] = 2 if L != L else 3
                    REL[m] = UNB[m] = BIG
                else:
                    ST[m] = 1 if L == L else 2
                    if life == 1:
                        REL[m] = tk[j]
                    elif life == 4:
                        REL[m] = min(tk[j], tk[i] + (960 - mod[i]) * 60_000_000_000)
                    else:
                        REL[m] = dead_s
                    UNB[m] = tk[j]
                m += 1
                continue
        else:
            NEED[m] = True
        ent = P if not defer else min(o[f], L)
        dead = dead_of[f]
        j = f + 1
        while j < n:                                                # 보유: RSI>70 다음 봉 또는 8시
            if r[j - 1] > X or tk[j] >= dead:
                break
            j += 1
        if j >= n:
            ST[m] = 3
            REL[m] = UNB[m] = BIG
        else:
            ST[m], FILL[m], RET[m], REL[m], UNB[m] = 0, True, o[j] / ent - 1, tk[j], tk[j]
        m += 1
    return I[:m], REL[:m], UNB[:m], NEED[:m], FILL[:m], RET[:m], ST[:m]


def resignal_cands(p, nm):
    """전체137 후보(기간 안) + 기간 시작 때 종목별 막힘 시각(기간 앞 신호를 상태 기계대로 처리한 결과)."""
    lo, hi = DZ.WIN[p]
    sp = SPEC[nm]
    rows, free0 = [], []
    for si, (s, (A, Z, sid)) in enumerate(DZ.ctx(p).items()):
        gap = 0.03 if s in DZ.LEV else 0.02
        dfr = np.isin(A["ses"], list(sp["dfr"]))
        I, REL, UNB, NEED, FILL, RET, ST = outcomes_h(A["o"], A["l"], A["r"], A["ema"], A["tk"], A["dead"], A["reg"], A["hot7"],
                                                      A["mod"], sid, Z["ZLO"], Z["ZHI"], gap, sp["mode"], sp["k"], sp["life"],
                                                      sp["fixd"], SL.X, dfr)
        day = A["day"][I]
        fr = np.iinfo(np.int64).min
        for e in np.flatnonzero(day < lo):                          # 기간 앞: 칸 제약 없는 상태 기계
            if A["tk"][I[e]] > fr and ST[e] != 3:
                fr = UNB[e]
        free0.append(fr)
        # 자료 끝 미완료(stat 3)는 칸을 안 쓰지만(본 계좌에서 버리는 것과 같게) 그 종목을 끝까지 막는다(상태 기계와 같게)
        m = (day >= lo) & (day <= hi)
        NEED[ST == 3] = False
        if sp["skip"]:                                              # 대조군: 정규장 신호 = 주문 없음(stat 2)
            NEED[ST == 1] = False
            ST[ST == 1] = 2
        rows.append(pd.DataFrame(dict(symi=si, sym=s, s=I[m], acq=A["tk"][I[m]], rel=REL[m], unb=UNB[m], need=NEED[m],
                                      fill=FILL[m], ret=RET[m], stat=ST[m], day=day[m])))
    return pd.concat(rows, ignore_index=True), np.array(free0, np.int64)


def run_resignal(T):
    out = []
    for p in PERIODS:
        for nm in RULES:
            X, free0 = resignal_cands(p, nm)
            st = DP.alloc_rs(np.lexsort((X.s.values, X.acq.values)), X.symi.values, X.acq.values, X.rel.values, X.unb.values,
                             X.need.values, 10**6, free0)
            got = X[(st > 0) & (X.stat.values != 3)].set_index(["sym", "s"]).sort_index()
            ref = T[(T.기간 == p) & (T.범위 == RG) & (T.규칙 == nm) & (T.stat != 3)].set_index(["sym", "s"]).sort_index()
            assert got.index.equals(ref.index) and (got.stat.values == ref.stat.values).all(), (p, nm)
            assert np.allclose(got.ret.values[got.fill.values], ref.ret.values[ref.stat.values == 0]), (p, nm)
            for un, us in DZ.UNIV.items():
                Y = X[X.sym.isin(us).values]
                acq, rel, unb, need, fill = (Y[c].values for c in ("acq", "rel", "unb", "need", "fill"))
                ret = Y.ret.values
                for K in KS:
                    for sd in range(NSEED):
                        rng = np.random.default_rng(1000 * sd + K)
                        order = np.lexsort((rng.random(len(Y)), acq))
                        st = DP.alloc_rs(order, Y.symi.values, acq, rel, unb, need, K, free0)
                        took = st == 1
                        out.append(dict(기간=p, 종목=un, 규칙=RULES[nm], K=K, 무작위=sd, 계좌=ret[took & fill].sum() / K,
                                        받음=int(took.sum()), 버림=int((st == 3).sum())))
        print("재신호", p, "완료", flush=True)
    return pd.DataFrame(out)


# ---------------------------------------------------------------- 보고
def main():
    pd.set_option("display.width", 250)
    T = pd.concat([pd.read_parquet(TRADES), skip_table()], ignore_index=True)
    M = pd.read_csv(METRICS)
    M["기간"] = M.기간.astype(str)
    R, daily, MC, CHK = run_all(T, M)
    R.to_csv(os.path.join(TMP, "defer_zone_hybrid_port_runs.csv"), index=False, encoding="utf-8-sig")
    S = DP.summarize(R, daily)
    S.to_csv(os.path.join(TMP, "defer_zone_hybrid_port_summary.csv"), index=False, encoding="utf-8-sig")
    S["5~95%"] = [f"{a:+.1%}~{b:+.1%}" for a, b in zip(S.계좌p5, S.계좌p95)]
    RS = run_resignal(T)
    RS.to_csv(os.path.join(TMP, "defer_zone_hybrid_port_resignal.csv"), index=False, encoding="utf-8-sig")
    get = lambda **q: S.loc[np.logical_and.reduce([S[k] == v for k, v in q.items()])].iloc[0]
    others = [v for v in RULES.values() if v != "지금"]
    sess_cols = [f"{s}손익" for s in SESS] + [f"{s}체결" for s in SESS] + ["자는중손익"]
    SS = R.groupby(["기간", "종목", "규칙", "방식", "K"])[sess_cols].mean().reset_index()
    gs = lambda **q: SS.loc[np.logical_and.reduce([SS[k] == v for k, v in q.items()])].iloc[0]

    L = []
    w = L.append
    w("이격매수 혼합(본장 신호만 지지 구역 하단 지정가) — 자금 한도 계좌 시험 (2026-10-10, defer_zone_hybrid_port.py)")
    w("질문(사용자 10/10): 구역 매수를 본장에서만 하면 어떤가(큰 하락은 주로 본장에서 나옴). 칸이 한정된 계좌에서도 지금 지표보다 못한가.")
    w(f"입력: defer_zone_hybrid.py 신호표(24시간). 계좌 모형 = defer_zone_port.py(K칸, 같은 시각 신호 무작위 우선순위 {NSEED}번, 단순 합).")
    w("규칙 약칭: 혼합Z1 Lr·혼합Z2 Lr = 정규장 신호만 P 아래 1·2번째 지지 구역 하단 지정가, 정규장 마감에 취소 / 혼합Z2 L1 = 지금 지표가")
    w("  팔았을 시각에 취소 / 혼합-3% Lr = 같은 깊이 대조(P x 0.97) / 프리본장Z2 L2 = 프리장 신호도 미룸, 한국 아침 8시 취소. 나머지 신호는 P에 삼.")
    w("  정규안삼 = 대조군: 정규장 신호는 아예 안 사고 칸도 안 씀(프리·애프터 신호만 P에 삼). 혼합의 이득이 구역 때문인지 칸 재배치 때문인지 가른다.")
    w("묶음 = 대기 지정가도 칸을 잡음(신호~취소/매도), 안묶음 = 체결 때 빈 칸이 있어야 체결(가격 알림 보고 손으로 사기에 가까움). 지금 지표는 둘이 같다.")
    w("칸 놓는 시각: Lr 미체결 = 첫 정규장 밖 봉과 16:00 ET 중 이른 쪽, L1 = 지금 지표 매도 시각, L2 = 첫 한국 아침 8시.")
    w("칸: 계좌 = 무작위 평균 계좌 수익(받은 거래 수익 합/K), 차이 = 계좌 - 같은 K 지금 지표, 이길 = 무작위 짝 중 변형이 이긴 비율,")
    w("    t = 신호 ET 날짜별 손익 차이의 날짜 군집 t, 낙폭 = 기준봉(early 5분·late 3분) 끝 평가 계좌 최대 낙폭(무작위 평균).")
    w("판정: early·late 둘 다 같은 방향일 때만 이김/짐(얕음/깊음), 아니면 엇갈림. oos(10/7~10/9, 3일·애프터 봉 결측)는 참고.")
    w("검증(assert 통과): 미체결 주문 취소 봉 재계산·그 전 주문가 닿음 없음, L1 취소 = 지금 지표 매도 시각, 종목별 칸 겹침 없음,")
    w("      K 무한 계좌 x K = defer_zone_hybrid '합'(두 방식), 재신호 후보로 K 무한 = 신호표(신호·stat·수익).")
    cap = CHK[CHK.규칙.str.contains("Lr")]
    w("Lr 미체결 중 16:00 ET로 깎은 건(애프터 첫 봉이 늦거나 빠짐): " +
      ", ".join(f"{r.기간} {r.규칙} {r.마감캡}/{r.미체결}" for r in cap.itertuples()))

    # 요약 기호
    w("")
    w("=" * 20 + " 요약 A: 계좌 수익, 같은 K 지금 지표 대비(early·late 둘 다 이기면 +, 둘 다 지면 -, 엇갈리면 ~). 글자 순서 K = "
      + "·".join(map(str, KS)))
    w("   대문자 표시 없음. 두 기간 모두 |t|>=2인 칸은 표 1의 '두기간t' 열에 따로 적었다.")
    rows = []
    for nm in others:
        row = dict(규칙=nm)
        for un in DZ.UNIV:
            for mode in MODES:
                row[f"{un}·{mode}"] = "".join(
                    MARK[DP.verdict(get(기간="early", 범위=RG, 종목=un, 규칙=nm, 방식=mode, K=K).차이,
                                    get(기간="late", 범위=RG, 종목=un, 규칙=nm, 방식=mode, K=K).차이)] for K in KS)
        rows.append(row)
    w(pd.DataFrame(rows).set_index("규칙").T.to_string())
    w("")
    w("=" * 20 + " 요약 B: 평가 낙폭, 같은 K 지금 지표 대비(둘 다 얕으면 +, 둘 다 깊으면 -, 엇갈리면 ~). 글자 순서 K = "
      + "·".join(map(str, KS)))
    rows = []
    for nm in others:
        row = dict(규칙=nm)
        for un in DZ.UNIV:
            for mode in MODES:
                s = ""
                for K in KS:
                    de = get(기간="early", 범위=RG, 종목=un, 규칙=nm, 방식=mode, K=K).낙폭 - get(기간="early", 범위=RG, 종목=un, 규칙="지금", K=K).낙폭
                    dl = get(기간="late", 범위=RG, 종목=un, 규칙=nm, 방식=mode, K=K).낙폭 - get(기간="late", 범위=RG, 종목=un, 규칙="지금", K=K).낙폭
                    s += MARK[DP.verdict(de, dl)]
                row[f"{un}·{mode}"] = s
        rows.append(row)
    w(pd.DataFrame(rows).set_index("규칙").T.to_string())

    # 0) 구조
    w("")
    w("=" * 20 + " 0) 구조: 칸 제약 없을 때(K 무한) 주문 1건의 칸 점유 시간(시간)과 칸-시간당 수익(%/시간), 신호 세션별")
    w("   합 = 체결 수익 합(K 무한 계좌 x K). 묶음 = 신호~취소/매도 시간, 안묶음 = 체결~매도 시간(시계 시간, 밤 포함).")
    w("   포화 계좌(작은 K)의 수익 ≈ 쓸 수 있는 칸-시간 x 칸-시간당 수익이라, 칸-시간당이 낮은 세션의 신호를 덜 받을수록 유리하다.")
    mc = ["규칙", "세션", "주문", "체결", "합", "체결평균", "주문당시간_묶음", "칸시간당_묶음", "체결당시간_안묶음", "칸시간당_안묶음"]
    order = {v: i for i, v in enumerate(RULES.values())}
    for un in DZ.UNIV:
        for p in PERIODS:
            x = MC[(MC.종목 == un) & (MC.기간 == p)].copy()
            x = x.assign(_o=x.규칙.map(order), _s=x.세션.map({"전체": 0, "프리": 1, "정규": 2, "애프터": 3})).sort_values(["_o", "_s"])
            x["합"] = x.합.map("{:+.2f}".format)
            w(f"\n[{un} · {DZ.PN[p]} · 24시간]")
            w(DP.fmt(x, mc))

    # 1) 판정
    w("")
    w("=" * 20 + " 1) 판정: 같은 K 지금 지표 대비 계좌 수익. 판정 = early·late 둘 다 같은 방향일 때만 이김/짐")
    w("   두기간t = early·late t가 둘 다 +2 이상이면 '유의+', 둘 다 -2 이하면 '유의-'.")
    for un in DZ.UNIV:
        for mode in MODES:
            rows = []
            for nm in others:
                for K in KS:
                    e, l, o = (get(기간=p, 범위=RG, 종목=un, 규칙=nm, 방식=mode, K=K) for p in PERIODS)
                    be, bl = (get(기간=p, 범위=RG, 종목=un, 규칙="지금", K=K) for p in ("early", "late"))
                    rows.append(dict(규칙=nm, K=K, e지금=f"{be.계좌:+.1%}", e변형=f"{e.계좌:+.1%}", e차이=f"{e.차이:+.1%}",
                                     e이길=f"{e.이길확률:.0%}", e_t=f"{e.t:+.2f}", l지금=f"{bl.계좌:+.1%}", l변형=f"{l.계좌:+.1%}",
                                     l차이=f"{l.차이:+.1%}", l이길=f"{l.이길확률:.0%}", l_t=f"{l.t:+.2f}", oos차이=f"{o.차이:+.1%}",
                                     판정=DP.verdict(e.차이, l.차이),
                                     두기간t="유의+" if e.t >= 2 and l.t >= 2 else ("유의-" if e.t <= -2 and l.t <= -2 else "")))
            w(f"\n[{un} · {mode}]")
            w(pd.DataFrame(rows).to_string(index=False))

    # 1b) 혼합 vs 대조군
    w("")
    w("=" * 20 + " 1b) 혼합 vs 대조군(정규안삼): 같은 K에서 혼합 계좌 - 정규안삼 계좌. +면 정규장 신호를 구역·깊이 지정가로라도 사는 편이 아예 안 사는 것보다 낫다")
    w("   t = 날짜별 손익 차이(무작위 평균)의 날짜 군집 t, 이길 = 같은 번호 무작위 짝 중 혼합이 이긴 비율.")
    Rk = R.set_index(["기간", "종목", "규칙", "방식", "K", "무작위"]).계좌.sort_index()
    for un in DZ.UNIV:
        for mode in MODES:
            rows = []
            for nm in [v for v in others if v.startswith("혼합")]:
                for K in KS:
                    row, dif = dict(규칙=nm, K=K), {}
                    for p in PERIODS:
                        a = Rk.loc[(p, un, nm, mode, K)].values
                        c = Rk.loc[(p, un, "정규안삼", mode, K)].values
                        dd = daily[(p, RG, un, nm, mode, K)] - daily[(p, RG, un, "정규안삼", mode, K)]
                        sd = dd.std(ddof=1)
                        tv = dd.mean() / (sd / np.sqrt(len(dd))) if sd > 0 and len(dd) > 1 else 0.0
                        dif[p] = a.mean() - c.mean()
                        if p == "oos":
                            row["oos차이"] = f"{dif[p]:+.1%}"
                        else:
                            row.update({f"{p[0]}혼합": f"{a.mean():+.1%}", f"{p[0]}안삼": f"{c.mean():+.1%}", f"{p[0]}차이": f"{dif[p]:+.1%}",
                                        f"{p[0]}이길": f"{(a > c).mean():.0%}", f"{p[0]}_t": f"{tv:+.2f}"})
                    row["판정"] = DP.verdict(dif["early"], dif["late"])
                    rows.append(row)
            w(f"\n[{un} · {mode}]")
            w(pd.DataFrame(rows).to_string(index=False))

    # 2) 낙폭
    w("")
    w("=" * 20 + " 2) 평가 낙폭(무작위 평균, 괄호 = 나쁜 5%). 판정 = 둘 다 얕으면 얕음, 둘 다 깊으면 깊음")
    for un in DZ.UNIV:
        for mode in MODES:
            rows = []
            for nm in others:
                for K in KS:
                    e, l = (get(기간=p, 범위=RG, 종목=un, 규칙=nm, 방식=mode, K=K) for p in ("early", "late"))
                    be, bl = (get(기간=p, 범위=RG, 종목=un, 규칙="지금", K=K) for p in ("early", "late"))
                    v = DP.verdict(e.낙폭 - be.낙폭, l.낙폭 - bl.낙폭)
                    rows.append(dict(규칙=nm, K=K, e지금=f"{be.낙폭:+.1%}({be.낙폭p5:+.1%})", e변형=f"{e.낙폭:+.1%}({e.낙폭p5:+.1%})",
                                     l지금=f"{bl.낙폭:+.1%}({bl.낙폭p5:+.1%})", l변형=f"{l.낙폭:+.1%}({l.낙폭p5:+.1%})",
                                     판정={"이김": "얕음", "짐": "깊음", "엇갈림": "엇갈림"}[v]))
            w(f"\n[{un} · {mode}]")
            w(pd.DataFrame(rows).to_string(index=False))

    # 3) 신호 세션별 손익
    w("")
    w("=" * 20 + " 3) 신호 세션별 계좌 손익(무작위 평균, 계좌 % = 그 세션 신호 거래 수익 합/K)과 체결 건수. 변형-지금 차이를 세션별로 쪼갬")
    w("   정규 칸이 줄면서 프리·애프터 칸이 늘어나는지(묶음에서는 대기 주문이 칸을 잡아 줄 수도) 본다.")
    w("   자는중 = 신호가 한국 00~08시(사용자가 자는 시간, 뉴욕 11:00~19:00)에 나온 거래의 계좌 손익(세션 무관). 애프터 신호는 대부분 여기 든다.")
    for un in DZ.UNIV:
        for mode in MODES:
            rows = []
            for p in ("early", "late"):
                for K in KS:
                    b = gs(기간=p, 종목=un, 규칙="지금", 방식="묶음", K=K)
                    for nm in ["지금"] + others:
                        x = gs(기간=p, 종목=un, 규칙=nm, 방식="묶음" if nm == "지금" else mode, K=K)
                        d = dict(기간=p, K=K, 규칙=nm)
                        for s in SESS:
                            d[f"{s}"] = f"{x[f'{s}손익']:+.1%}" + ("" if nm == "지금" else f"({x[f'{s}손익'] - b[f'{s}손익']:+.1%})")
                            d[f"{s}건"] = f"{x[f'{s}체결']:.0f}"
                        d["자는중"] = f"{x['자는중손익']:+.1%}" + ("" if nm == "지금" else f"({x['자는중손익'] - b['자는중손익']:+.1%})")
                        rows.append(d)
            w(f"\n[{un} · {mode}] 괄호 = 같은 K 지금 지표 대비")
            w(pd.DataFrame(rows).to_string(index=False))

    # 4) 자세히
    w("")
    w("=" * 20 + " 4) 자세히: 기간·방식별 K x 규칙")
    cols = ["규칙", "K", "계좌", "5~95%", "복리", "낙폭", "낙폭p5", "받음", "체결", "놓침", "사용률", "차이", "이길확률", "t"]
    for un in DZ.UNIV:
        for p in PERIODS:
            for mode in MODES:
                x = S[(S.종목 == un) & (S.기간 == p) & ((S.방식 == mode) | (S.규칙 == "지금"))]
                x = x.assign(_o=x.규칙.map(order)).sort_values(["K", "_o"])
                w(f"\n[{un} · {DZ.PN[p]} · {mode}]")
                w(DP.fmt(x, cols))

    # 5) 재신호 점검
    w("")
    w("=" * 20 + " 5) 한계 점검: 재신호 계좌(묶음). 칸이 없어 버린 종목은 다음 봉부터 매수선 닿으면 다시 신호(구역·주문가도 그 봉에서 새로)")
    w("   표 = 위 신호표 계좌, 재 = 재신호 계좌(무작위 50번 평균), 재차이 = 변형 - 같은 K 재신호 지금 지표, 재이길 = 무작위 짝 중 변형이 이긴 비율.")
    G2 = RS.groupby(["기간", "종목", "규칙", "K"]).agg(계좌=("계좌", "mean"), 받음=("받음", "mean"), 버림=("버림", "mean")).reset_index()
    g2 = lambda **q: G2.loc[np.logical_and.reduce([G2[k] == v for k, v in q.items()])].iloc[0]
    rb = RS[RS.규칙 == "지금"][["기간", "종목", "K", "무작위", "계좌"]].rename(columns={"계좌": "b"})
    RJ = RS.merge(rb, on=["기간", "종목", "K", "무작위"])
    WIN2 = RJ.assign(w=RJ.계좌 > RJ.b).groupby(["기간", "종목", "규칙", "K"]).w.mean()
    for un in DZ.UNIV:
        rows = []
        for nm in RULES.values():
            for K in KS:
                row = dict(규칙=nm, K=K)
                dif = {}
                for p in ("early", "late"):
                    tb = get(기간=p, 범위=RG, 종목=un, 규칙=nm, 방식="묶음", K=K)
                    r2, b2 = g2(기간=p, 종목=un, 규칙=nm, K=K), g2(기간=p, 종목=un, 규칙="지금", K=K)
                    dif[p] = r2.계좌 - b2.계좌
                    row.update({f"{p[0]}표": f"{tb.계좌:+.1%}", f"{p[0]}재": f"{r2.계좌:+.1%}", f"{p[0]}재받음": f"{r2.받음:.0f}",
                                f"{p[0]}재차이": "" if nm == "지금" else f"{dif[p]:+.1%}",
                                f"{p[0]}재이길": "" if nm == "지금" else f"{WIN2[(p, un, nm, K)]:.0%}"})
                row["재판정"] = "" if nm == "지금" else DP.verdict(dif["early"], dif["late"])
                rows.append(row)
        w(f"\n[{un} · 24시간 · 묶음]")
        w(pd.DataFrame(rows).to_string(index=False))
    # 6) 참고: 사용자가 깨어 있는 시간에 나온 신호
    w("")
    w("=" * 20 + " 6) 참고: 지금 지표 거래(칸 무한)를 신호 시각이 사용자가 깨어 있는 한국 08~24시인지로 나눔(docs/desk_system.md §1)")
    w("   애프터 신호의 대부분은 사용자가 자는 한국 00~08시에 나온다. 그 시간 매수는 미리 건 지정가로만 가능하고 애프터 체결은 확인 전이다.")
    rows = []
    b = T[(T.규칙 == BASE) & (T.stat == 0)]
    for p in ("early", "late"):
        for un, us in DZ.UNIV.items():
            x = b[(b.기간 == p) & b.sym.isin(us)]
            sl = sleep_of(x.ts.values.astype(np.int64))
            for ss in SESS:
                m = (x.ses == ss).values
                rows.append(dict(기간=p, 종목=un, 세션=ss, 건수=int(m.sum()), 합=f"{x.ret.values[m].sum():+.2f}",
                                 깨어있음비율=f"{(~sl[m]).mean():.0%}" if m.any() else "",
                                 깨어있음합=f"{x.ret.values[m & ~sl].sum():+.2f}", 자는중합=f"{x.ret.values[m & sl].sum():+.2f}"))
    w(pd.DataFrame(rows).to_string(index=False))
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("저장:", OUT)


if __name__ == "__main__":
    main()
