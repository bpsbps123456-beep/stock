"""급등 후 눌림목 매수 — 자금 한도 계좌 시험(2026-10-10, 키 surge_pullback).

질문: 급등 사건은 하루 10~20건이라 다 살 수 없다. 계좌를 칸으로 나눠 빈 칸이 있을 때만 받으면, 눌림목 매수(F50 등) 계좌는
  QQQ·SPY 보유나 '급등 다음 날 시가에 바로 사기(M1)'·'추격(M0)'·'사건 뒤 아무 날 사기'보다 나은가.

규칙
  거래: study.py 거래표(scratchpad sp_trades.parquet)의 매수·매도 날짜·가격·미청산 표시를 그대로 쓴다.
    비교 8개: F50 H10, F50 TGT, F618 H10, EMA10 H10, M0 H10, M1 H10, D1 H10(= M1 H10과 같은 거래, 확인용),
    아무날 H10 = 사건마다 D1~D10 중 하루를 무작위로 골라 그날 시가 매수(씨앗마다 다시 고름, study.py '아무날대비'의 계좌판).
  계좌: 시작 자금 1을 K칸(3·5·10·20)으로 똑같이 나눈다. 거래를 매수 시각 순서로 보고 빈 칸이 있을 때만 받는다.
    같은 시각은 무작위 우선순위(씨앗 30개로 30번). 빈 칸이 여럿이면 가장 먼저 빈 칸. 이미 들고 있는 종목은 받지 않는다.
    칸은 매수~매도 동안 묶이고, 같은 시각의 매도로 빈 칸은 그 시각 매수에 바로 쓴다(매도 먼저).
  시각(ET, 하루 안 순서): 09:30 시가 매도·시가 매수 → 장중 지정가 매도(TGT 목표) → 장중 지정가 매수 → 16:00 종가 매도·종가 매수.
    지정가 매수(F50·F618·EMA10)는 장중 자리(같은 날 장중 매도 뒤)에 두되, 시가가 이미 지정가 아래라 시가에 체결된 거래
    (매수가 = 시가)는 09:30에 둔다. M0 = 사건 날 16:00, M1·D = 그날 09:30. H10 매도 = 16:00.
    TGT 매도: 시가가 목표 위(갭 체결)·손절(종가 < B 다음 날 시가) = 09:30, 목표 장중 체결 = 장중 매도 자리, 20일 만기 = 16:00.
    자료 끝까지 안 팔린 거래(미청산)는 칸을 계속 묶고 종가로 평가한다.
  단순 합: 칸 크기 1/K 고정, 계좌 = 1 + Σ(받은 거래 수익)/K. 복리: 칸마다 자기 돈을 전부 다시 넣는다(계좌 = 칸 값의 합).
  평가: 매 거래일 정규장 종가(sp_daily c, 빠진 날은 앞 값)로 들고 있는 종목을 평가해 하루 계좌 값 → 최대 낙폭.
    기간마다 빈 계좌로 새로 시작하고 기간 안에 매수한 거래만 받는다. 기간 끝에 안 팔린 칸은 기간 마지막 날 종가로 평가.
  기간(매수 시각 기준): 2025(보조) / 2026 상 1~6월(고르기) / 2026 하 7/1~10/6(확인, 자료 끝).
  지표: 계좌 수익(30번 평균, 5~95% 분위), 낙폭(하루 종가 계좌, 30번 평균),
    상위5뺀 = 받은 거래 중 가장 좋은 5건의 기여를 0으로 친 단순 합(소수 대박 의존 점검),
    t일간 = (하루 복리 계좌 수익 - 하루 QQQ 수익) t, 베타·알파t = 하루 복리 계좌 수익을 하루 QQQ 수익에 회귀(시장 반등 의존 점검),
    t거래 = 받은 거래의 (수익 - 같은 보유 구간 QQQ, 매수 전날 종가→매도 날 종가)를 매수 날짜로 묶은 날짜 군집 t(모두 30번 중앙값),
    P>QQQ = 30번 중 QQQ를 이긴 비율, P>M1·P>아무날 = 같은 칸 비교 규칙과 30×30 = 900쌍 중 이긴 비율,
    붓QQQ·붓아무날 = 30번 평균 하루 로그 초과수익을 5거래일 묶음 부트스트랩(2,000번)한 합이 0보다 클 확률,
    매수일 = 받은 거래의 서로 다른 매수 날 수(칸이 한꺼번에 비고 차서 몇 날짜에 몰리는지).
  시작 위상 점검: 빈 계좌로 시작하면 칸이 첫날 한꺼번에 차고 10일 뒤 한꺼번에 비어, 30번 씨앗이 매수 날짜를 거의 못 바꾼다.
    그래서 씨앗 sd마다 기간 시작 (sd % 11)거래일 전부터 계좌를 돌려(예열, 그 사이 산 거래는 기간 시작 종가부터 평가)
    칸이 비는 날짜를 바꾼 복리 계좌도 본다(같은 기간 창, 위상 0 = 본 표).
  거래표 전체 평균: 계좌 제한 없이 기간 안에 매수한 모든 거래의 평균을 거래 가중·매수 날 균등 가중으로 나눠,
    하루 거래가 몰린 날(상위 10% 날)의 몫을 본다(study.py 평균과 계좌 결과가 왜 다른지).
가정·한계
  수수료·세금·호가 차이 0, 지정가는 저가가 닿으면 다 체결(대기 줄·부분 체결 없음). 같은 날 장중 매도가 장중 지정가 매수보다
  먼저라고 본다(조금 낙관). 매도 대금은 바로 다시 쓴다(결제일 무시). 사용자 주문 가능 시간(docs/desk_system.md §1)은 반영하지 않는다.
  현재 상장·시총 3억 달러 이상 종목뿐이라 생존 편향이 있다(상폐·폭락 종목이 빠져 급등 후 매수 계좌가 부풀 수 있다).
  탐색 범위: 규칙 8(D1 H10 = M1 H10이라 실제 7) × 칸 4 = 28조합 × 기간 3 × 시작 방식 2(빈 계좌, 위상 0~10일 예열).
    본 표의 씨앗 30번은 같은 시각 순서의 운만 바꾼다(표본 불확실성은 부트스트랩, 날짜 운은 시작 위상 점검).
    2026 상에서 28조합 복리 순위를 매겨 2026 하에서 확인한다(순위 상관 포함). 2025는 보조.
사용: py account.py → results/account.txt
메모리: 거래표는 필요한 열·규칙만(pyarrow 필터), 일봉은 2024-11 이후 시가·종가만 넓은 표(날짜 482 × 종목 약 3,560, float).
"""
import os
from datetime import date
import numpy as np, pandas as pd
import pyarrow as pa, pyarrow.parquet as pq
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
SP = r"C:/Users/geunt/AppData/Local/Temp/claude/C--Users-geunt-OneDrive-Desktop---/6a57b230-3ac1-4ae6-9d34-b1d8a1c2ce1b/scratchpad"
DS = [f"D{d}" for d in range(1, 11)]
H10E = ["F50", "F618", "EMA10", "M0", "M1"] + DS
LIMIT = ("F50", "F618", "EMA10")
RAND = "아무날 H10"
NAMES = ["F50 H10", "F50 TGT", "F618 H10", "EMA10 H10", "M0 H10", "M1 H10", "D1 H10", RAND]
SLOTS = (3, 5, 10, 20)
SEEDS = 30
T_OPEN, T_LSELL, T_LBUY, T_CLOSE = 570, 700, 701, 960      # 하루 안 순서(분): 시가 → 장중 매도 → 장중 지정가 매수 → 종가
DAYMIN = 1440
INF = np.iinfo(np.int64).max
T0 = date(2024, 11, 1)
EPOCH = date(1970, 1, 1)
PERIODS = [("2025(보조)", date(2025, 1, 1), date(2026, 1, 1)),
           ("2026 상(고르기)", date(2026, 1, 1), date(2026, 7, 1)),
           ("2026 하(확인)", date(2026, 7, 1), date(2026, 12, 31))]


def dnum(d):
    return (d - EPOCH).days


# ---------------------------------------------------------------- 자료
def arrow(path, cols, filters=None):
    """parquet → numpy 열. 날짜 = 1970-01-01부터 날 수(int32), 사전형 = (번호, 이름표). 파이썬 객체를 만들지 않는다."""
    t = pq.read_table(path, columns=cols, filters=filters).unify_dictionaries().combine_chunks()
    out = {}
    for c in cols:
        a = t.column(c).combine_chunks()
        if pa.types.is_dictionary(a.type):
            out[c] = (a.indices.to_numpy(zero_copy_only=False).astype(np.int64), a.dictionary.to_pylist())
        elif pa.types.is_date32(a.type):
            out[c] = a.cast(pa.int32()).to_numpy().astype(np.int64)
        else:
            out[c] = a.to_numpy(zero_copy_only=False)
    return out


def load_daily():
    """2024-11 이후 정규장 시가·종가 → 넓은 표(SPY 거래일 × 종목)."""
    d = arrow(os.path.join(SP, "sp_daily.parquet"), ["day", "sym", "o", "c"], [("day", ">=", T0)])
    si, names = d["sym"]
    days = np.unique(d["day"])
    di = np.searchsorted(days, d["day"])
    O = np.full((len(days), len(names)), np.nan, np.float32)
    C = np.full((len(days), len(names)), np.nan, np.float32)
    O[di, si] = d["o"]
    C[di, si] = d["c"]
    del d
    keep = ~np.isnan(C[:, names.index("SPY")])
    assert keep.all(), "SPY 없는 날이 있다"
    M = pd.DataFrame(C).ffill().values.astype(np.float64)      # 평가 종가(빠진 날은 앞 값)
    return days, O, C, M, names


def load_trades(days, O, C, names):
    cols = ["sym", "eday", "entry", "exit", "tday", "ent", "xday", "px", "open", "tgt"]
    t1 = ("tday", ">=", date(2024, 12, 1))                       # 기간(2025~)에 매수한 거래만 쓰면 되므로 앞은 버린다
    f = [[("exit", "==", "H10"), ("entry", "in", H10E), t1], [("exit", "==", "TGT"), ("entry", "==", "F50"), t1]]
    t = arrow(os.path.join(SP, "sp_trades.parquet"), cols, f)
    si, sn = t["sym"]
    smap = {n: i for i, n in enumerate(names)}
    s = np.array([smap[n] for n in sn], np.int64)[si]           # 거래표 종목 번호 → 일봉 표 열 번호
    ei, en = t["entry"]
    xi_, xn = t["exit"]
    ent_c = np.array(en)                                         # 이름표(19개)만 문자열
    ent_name = ent_c[ei]
    is_tgt = np.array(xn)[xi_] == "TGT"
    ti, xi, eix = (np.searchsorted(days, t[k]) for k in ("tday", "xday", "eday"))
    assert (days[ti] == t["tday"]).all() and (days[xi] == t["xday"]).all()
    ent, px, op, tgt = t["ent"].astype(float), t["px"].astype(float), t["open"].astype(bool), t["tgt"].astype(float)
    # 매수 시각
    o_t = O[ti, s].astype(float)
    gap = np.isclose(ent, o_t, rtol=1e-6, atol=0)
    lim = np.isin(ent_name, LIMIT)
    bcode = np.where(ent_name == "M0", T_CLOSE, np.where(lim & ~gap, T_LBUY, T_OPEN))
    te = t["tday"] * DAYMIN + bcode
    # 매도 시각
    o_x, c_x = O[xi, s].astype(float), C[xi, s].astype(float)
    tg = np.maximum(tgt, ent * 1.0001)
    at_open = np.isclose(px, o_x, rtol=1e-6, atol=0)
    at_tg = np.isclose(px, tg, rtol=1e-6, atol=0)
    xcode = np.where(is_tgt, np.where(at_open, T_OPEN, np.where(at_tg, T_LSELL, T_CLOSE)), T_CLOSE)
    tx = t["xday"] * DAYMIN + xcode
    tx[op] = INF
    chk = (~op) & (xcode == T_CLOSE)
    bad = int((~np.isclose(px[chk], c_x[chk], rtol=1e-6, atol=0)).sum())
    qi = names.index("QQQ")
    q = C[:, qi].astype(float)
    k0 = ti - np.where(ent_name == "M0", 0, 1)                   # 같은 보유 구간 QQQ(매수 전날 종가 → 매도 날 종가; M0는 사건 날 종가부터)
    qret = q[xi] / q[np.maximum(k0, 0)] - 1
    T = dict(s=s, te=te.astype(np.int64), tx=tx.astype(np.int64), ent=ent, px=px, ret=px / ent - 1, qret=qret,
             bday=t["tday"], eid=s * 100000 + t["eday"])
    rules = {}
    for nm in NAMES:
        if nm == RAND:
            continue
        e_, x_ = nm.split()
        rules[nm] = np.flatnonzero((ent_name == e_) & (is_tgt == (x_ == "TGT")))
    dr = np.flatnonzero(np.isin(ent_name, DS) & ~is_tgt)
    diag = dict(n=len(ent), bad_close=bad, gap_lim=int((lim & gap & ~is_tgt).sum()), lim=int((lim & ~is_tgt).sum()),
                tgt_open=int((is_tgt & ~op & (xcode == T_OPEN)).sum()), tgt_lim=int((is_tgt & ~op & (xcode == T_LSELL)).sum()),
                tgt_close=int((is_tgt & ~op & (xcode == T_CLOSE)).sum()), tgt_n=int(is_tgt.sum()))
    return T, rules, dr, diag


def rand_days(T, dr):
    """씨앗마다 사건 하나에 D1~D10 중 하루(H10)를 무작위로 고른 거래 번호."""
    out = []
    eid = T["eid"][dr]
    for sd in range(SEEDS):
        rnd = np.random.default_rng(5000 + sd).random(len(dr))
        o = np.lexsort((rnd, eid))
        es = eid[o]
        last = np.r_[es[1:] != es[:-1], True]
        out.append(np.sort(dr[o[last]]))
    return out


# ---------------------------------------------------------------- 계좌
@njit(cache=True)
def accept(te, tx, symi, order, K):
    """매수 시각 순서대로 빈 칸(가장 먼저 빈 칸)에 받는다. 같은 시각 매도는 먼저 끝난 것으로 본다. 반환: 칸 번호(-1 = 못 받음)."""
    n = len(te)
    slot = np.full(n, -1, np.int64)
    free = np.full(K, -INF, np.int64)
    held = np.full(K, -1, np.int64)
    for q in range(len(order)):
        j = order[q]
        t = te[j]
        best, bf, dup = -1, INF, False
        for k in range(K):
            if free[k] > t:
                if held[k] == symi[j]:
                    dup = True
            elif free[k] < bf:
                bf, best = free[k], k
        if dup or best < 0:
            continue
        slot[j] = best
        free[best] = tx[j]
        held[best] = symi[j]
    return slot


@njit(cache=True)
def equity(te, tx, ent, px, col, slot, acc, snaps, M, K, compound):
    """하루 16:00 계좌 값(시작 1)과 칸 사용률. acc = 받은 거래(매수 시각 순)."""
    nd = len(snaps)
    eq = np.zeros(nd)
    occ = np.zeros(nd)
    for k in range(K):
        cnt = 0
        for q in range(len(acc)):
            if slot[acc[q]] == k:
                cnt += 1
        ids = np.empty(cnt, np.int64)
        c = 0
        for q in range(len(acc)):
            if slot[acc[q]] == k:
                ids[c] = acc[q]
                c += 1
        cap, bank, p = 1.0 / K, 1.0 / K, 0
        for d in range(nd):
            s = snaps[d]
            while p < cnt and tx[ids[p]] < s:
                j = ids[p]
                if compound:
                    cap *= px[j] / ent[j]
                else:
                    bank += (px[j] / ent[j] - 1.0) / K
                p += 1
            if p < cnt and te[ids[p]] < s:
                j = ids[p]
                m = M[d, col[j]]
                if m != m:
                    m = ent[j]
                v = cap * m / ent[j] if compound else bank + (m / ent[j] - 1.0) / K
                occ[d] += 1.0
            else:
                v = cap if compound else bank
            eq[d] += v
    return eq, occ / K


def mdd(eq):
    return float(np.max(1 - eq / np.maximum.accumulate(eq)))


def tstat(x):
    x = x[np.isfinite(x)]
    if len(x) < 3 or x.std(ddof=1) == 0:
        return np.nan
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))


def alpha_beta(y, x):
    """하루 계좌 수익 y = a + b·QQQ 수익 x → (b, a의 t)."""
    n = len(x)
    xm = x - x.mean()
    sxx = (xm ** 2).sum()
    if n < 5 or sxx == 0:
        return np.nan, np.nan
    b = (xm * (y - y.mean())).sum() / sxx
    a = y.mean() - b * x.mean()
    r = y - a - b * x
    se = np.sqrt((r ** 2).sum() / (n - 2) * (1 / n + x.mean() ** 2 / sxx))
    return b, (a / se if se > 0 else np.nan)


def boot_p(e, B=2000, L=5, seed=0):
    """하루 로그 초과수익 e의 5일 묶음(원형) 부트스트랩 합 > 0 확률."""
    n = len(e)
    if n < L:
        return np.nan
    rng = np.random.default_rng(seed)
    nb = -(-n // L)
    st = rng.integers(0, n, (B, nb))
    idx = (st[:, :, None] + np.arange(L)[None, None, :]).reshape(B, -1)[:, :n] % n
    return float((e[idx].sum(1) > 0).mean())


def run_period(T, rules, rsel, days, M, d0, d1):
    lo, hi = dnum(d0), dnum(d1)
    idx = np.flatnonzero((days >= lo) & (days < hi))
    rows = np.r_[idx[0] - 1, idx]                                 # 기간 직전 거래일 종가에서 시작
    snaps = days[rows] * DAYMIN + T_CLOSE
    Mv = np.ascontiguousarray(M[rows])
    qi, si_ = NAMES_IDX["QQQ"], NAMES_IDX["SPY"]
    q, sp = Mv[:, qi] / Mv[0, qi], Mv[:, si_] / Mv[0, si_]
    lq, rq = np.diff(np.log(q)), np.diff(q) / q[:-1]
    res = {"QQQ": dict(fin=q[-1] - 1, mdd=mdd(q)), "SPY": dict(fin=sp[-1] - 1, mdd=mdd(sp)), "_n": len(idx)}
    inper = (T["te"] >= lo * DAYMIN) & (T["te"] < hi * DAYMIN)
    res["_ev"] = int(inper[rules["M0 H10"]].sum())               # 기간 안 급등 사건 수(M0 매수 = 사건 날)
    end = snaps[-1]
    for nm in NAMES:
        for K in SLOTS:
            R = {k: [] for k in ("fs", "fc", "f5", "ms", "mc", "n", "nd", "occ", "td", "beta", "alpha", "tt", "avg", "sig", "lc")}
            for sd in range(SEEDS):
                sel = rsel[sd] if nm == RAND else rules[nm]
                sel = sel[inper[sel]]
                te, tx, ent, px, s = T["te"][sel], T["tx"][sel], T["ent"][sel], T["px"][sel], T["s"][sel]
                rnd = np.random.default_rng(1000 + sd).random(len(te))
                order = np.lexsort((rnd, te)).astype(np.int64)
                slot = accept(te, tx, s, order, K)
                acc = order[slot[order] >= 0]
                es, _ = equity(te, tx, ent, px, s, slot, acc, snaps, Mv, K, False)
                ec, oc = equity(te, tx, ent, px, s, slot, acc, snaps, Mv, K, True)
                # 기간 끝 기여(안 팔린 칸은 마지막 종가 평가) → 상위 5건 뺀 단순 합
                ca = np.where(tx[acc] <= end, px[acc], Mv[-1, s[acc]]) / ent[acc] - 1
                f5 = es[-1] - 1 - np.sort(ca)[::-1][:5].clip(min=0).sum() / K
                rc = np.diff(ec) / ec[:-1]
                b, a = alpha_beta(rc, rq)
                R["fs"].append(es[-1] - 1); R["fc"].append(ec[-1] - 1); R["f5"].append(f5)
                R["ms"].append(mdd(es)); R["mc"].append(mdd(ec))
                R["n"].append(len(acc)); R["occ"].append(oc[1:].mean()); R["sig"].append(len(te))
                R["nd"].append(len(np.unique(T["bday"][sel][acc])))      # 받은 거래의 서로 다른 매수 날 수(날짜 몰림 점검)
                R["td"].append(tstat(rc - rq)); R["beta"].append(b); R["alpha"].append(a)
                R["lc"].append(np.diff(np.log(ec)))
                if len(acc):
                    xs = T["ret"][sel][acc] - T["qret"][sel][acc]
                    u, inv = np.unique(T["bday"][sel][acc], return_inverse=True)
                    dm = np.bincount(inv, xs) / np.bincount(inv)
                    R["tt"].append(tstat(dm)); R["avg"].append(T["ret"][sel][acc].mean())
                    if abs(1 + ca.sum() / K - es[-1]) > 1e-6:          # 단순 합 계좌와 거래 기여 합이 맞는지 확인
                        raise RuntimeError(f"단순 합 불일치 {nm} {K}")
            for k_ in R:
                if k_ != "lc":
                    R[k_] = np.array(R[k_], float)
            R["lc"] = np.mean(R["lc"], axis=0)
            res[(nm, K)] = R
    for nm in NAMES:
        for K in SLOTS:
            R, M1, RD = res[(nm, K)], res[("M1 H10", K)], res[(RAND, K)]
            R["pq"] = float((R["fc"] > res["QQQ"]["fin"]).mean())
            R["ps"] = float((R["fc"] > res["SPY"]["fin"]).mean())
            R["pm1"] = float((R["fc"][:, None] > M1["fc"][None, :]).mean())
            R["prd"] = float((R["fc"][:, None] > RD["fc"][None, :]).mean())
            R["bq"] = boot_p(R["lc"] - lq)
            R["brd"] = boot_p(R["lc"] - RD["lc"]) if nm != RAND else np.nan
    return res


def run_phase(T, rules, rsel, days, M, d0, d1):
    """시작 위상 점검: 씨앗 sd마다 기간 시작 (sd % 11)거래일 전부터 계좌를 돌려(예열) 칸이 비는 날짜를 바꾼다.
    평가는 같은 기간 창(기간 직전 종가 → 기간 끝 종가)의 복리 계좌 값 비율. 위상 0은 본 표와 같다."""
    lo, hi = dnum(d0), dnum(d1)
    idx = np.flatnonzero((days >= lo) & (days < hi))
    p0 = idx[0] - 1
    qi = NAMES_IDX["QQQ"]
    qfin = M[idx[-1], qi] / M[p0, qi] - 1
    res = {"QQQ": qfin}
    cache = {}
    for nm in NAMES:
        for K in SLOTS:
            R = dict(fc=[], mc=[], nd=[])
            for sd in range(SEEDS):
                off = sd % 11
                if off not in cache:
                    rows = np.arange(p0 - off, idx[-1] + 1)
                    cache[off] = (days[rows] * DAYMIN + T_CLOSE, np.ascontiguousarray(M[rows]), days[p0 - off + 1] * DAYMIN)
                snaps, Mv, t0 = cache[off]
                sel = rsel[sd] if nm == RAND else rules[nm]
                sel = sel[(T["te"][sel] >= t0) & (T["te"][sel] < hi * DAYMIN)]
                te, tx, ent, px, s = T["te"][sel], T["tx"][sel], T["ent"][sel], T["px"][sel], T["s"][sel]
                rnd = np.random.default_rng(1000 + sd).random(len(te))
                order = np.lexsort((rnd, te)).astype(np.int64)
                slot = accept(te, tx, s, order, K)
                acc = order[slot[order] >= 0]
                ec, _ = equity(te, tx, ent, px, s, slot, acc, snaps, Mv, K, True)
                ep = ec[off:] / ec[off]                                  # 기간 창만
                R["fc"].append(ep[-1] - 1); R["mc"].append(mdd(ep))
                b = T["bday"][sel][acc]
                R["nd"].append(len(np.unique(b[b >= lo])))
            res[(nm, K)] = {k: np.array(v, float) for k, v in R.items()}
    return res


def pop_stats(T, rules, d0, d1):
    """거래표 전체(계좌 제한 없음)의 평균: 거래(사건) 가중 vs 매수 날 균등 가중, 거래가 몰린 날(하루 거래 수 상위 10% 날)의 몫."""
    lo, hi = dnum(d0) * DAYMIN, dnum(d1) * DAYMIN
    out = []
    for nm in NAMES[:-2]:
        sel = rules[nm]
        sel = sel[(T["te"][sel] >= lo) & (T["te"][sel] < hi)]
        r = T["ret"][sel]
        u, inv, cnt = np.unique(T["bday"][sel], return_inverse=True, return_counts=True)
        dm = np.bincount(inv, r) / cnt
        big = cnt[inv] >= np.quantile(cnt, .9)
        out.append({"규칙": nm, "거래": len(r), "날": len(u), "하루중앙": f"{np.median(cnt):.0f}", "하루최대": int(cnt.max()),
                    "거래가중": P(r.mean(), "{:+.2%}"), "날짜균등": P(dm.mean(), "{:+.2%}"),
                    "몰린날비중": P(big.mean(), "{:.0%}"), "몰린날평균": P(r[big].mean(), "{:+.2%}"), "나머지평균": P(r[~big].mean(), "{:+.2%}")})
    return pd.DataFrame(out).to_string(index=False)


# ---------------------------------------------------------------- 출력
def P(x, f="{:+.1%}"):
    return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f.format(x)


def med(a):
    a = np.asarray(a, float)
    return np.nanmedian(a) if np.isfinite(a).any() else np.nan


def table(res, name):
    n = res["_n"]
    L = [f"--- {name}  ({n}거래일, 급등 사건 {res['_ev']}건 = 하루 {res['_ev'] / max(n, 1):.1f}건)"
         f"   QQQ {P(res['QQQ']['fin'])} (낙폭 {P(res['QQQ']['mdd'], '{:.1%}')})"
         f"   SPY {P(res['SPY']['fin'])} (낙폭 {P(res['SPY']['mdd'], '{:.1%}')})"]
    rows = []
    for K in SLOTS:
        for nm in NAMES:
            R = res[(nm, K)]
            rows.append({"칸": K, "규칙": nm, "거래": f"{R['sig'].mean():.0f}", "받음": f"{R['n'].mean():.0f}",
                         "매수일": f"{med(R['nd']):.0f}",
                         "사용률": P(R["occ"].mean(), "{:.0%}"), "받은평균": P(med(R["avg"]), "{:+.2%}"),
                         "단순합": P(R["fs"].mean()), "상위5뺀": P(R["f5"].mean()), "복리": P(R["fc"].mean()),
                         "복리5~95%": f"{P(np.quantile(R['fc'], .05))}~{P(np.quantile(R['fc'], .95))}",
                         "낙폭단순": P(R["ms"].mean(), "{:.1%}"), "낙폭복리": P(R["mc"].mean(), "{:.1%}"),
                         "t일간": P(med(R["td"]), "{:+.2f}"), "베타": P(med(R["beta"]), "{:.2f}"), "알파t": P(med(R["alpha"]), "{:+.2f}"),
                         "t거래": P(med(R["tt"]), "{:+.2f}"), "P>QQQ": P(R["pq"], "{:.0%}"), "P>M1": P(R["pm1"], "{:.0%}"),
                         "P>아무날": "" if nm == RAND else P(R["prd"], "{:.0%}"),
                         "붓QQQ": P(R["bq"], "{:.0%}"), "붓아무날": "" if nm == RAND else P(R["brd"], "{:.0%}")})
    L.append(pd.DataFrame(rows).to_string(index=False))
    return L


def sgn(xs):
    return "같음+" if all(x > 0 for x in xs) else ("같음-" if all(x < 0 for x in xs) else "엇갈림")


def direction(allres):
    """규칙×칸마다 기간별 (복리 - QQQ), (복리 - 같은 칸 M1), (복리 - 같은 칸 아무날)과 2026 상·하 같은 방향 여부."""
    pa_, pb_, aux = PERIODS[1][0], PERIODS[2][0], PERIODS[0][0]
    rows = []
    for K in SLOTS:
        for nm in [n for n in NAMES if n != "D1 H10"]:
            row = {"칸": K, "규칙": nm}
            sq, sm, sr = [], [], []
            for p in (aux, pa_, pb_):
                rs = allres[p]
                f = rs[(nm, K)]["fc"].mean()
                dq, dm, dr = f - rs["QQQ"]["fin"], f - rs[("M1 H10", K)]["fc"].mean(), f - rs[(RAND, K)]["fc"].mean()
                tag = p.split("(")[0]
                row[f"{tag} -QQQ"] = P(dq)
                row[f"{tag} -M1"] = P(dm)
                row[f"{tag} -아무날"] = P(dr)
                if p != aux:
                    sq.append(dq); sm.append(dm); sr.append(dr)
            row["QQQ 두기간"], row["M1 두기간"], row["아무날 두기간"] = sgn(sq), sgn(sm), sgn(sr)
            rows.append(row)
    return [pd.DataFrame(rows).to_string(index=False)]


def phase_table(ph):
    """시작 위상을 바꾼 복리 계좌(30번 = 위상 0~10 × 씨앗): 기간별 평균·분위·이길 확률과 2026 상·하 방향."""
    pa_, pb_, aux = PERIODS[1][0], PERIODS[2][0], PERIODS[0][0]
    rows = []
    for K in SLOTS:
        for nm in NAMES:
            if nm == "D1 H10":
                continue
            row = {"칸": K, "규칙": nm}
            sq, sm, sr = [], [], []
            for p in (aux, pa_, pb_):
                r = ph[p]
                f = r[(nm, K)]["fc"]
                tag = p.split("(")[0]
                row[f"{tag} 복리"] = P(f.mean())
                row[f"{tag} 5~95%"] = f"{P(np.quantile(f, .05))}~{P(np.quantile(f, .95))}"
                row[f"{tag} 매수일"] = f"{med(r[(nm, K)]['nd']):.0f}"
                row[f"{tag} P>QQQ"] = P(float((f > r['QQQ']).mean()), "{:.0%}")
                row[f"{tag} P>M1"] = P(float((f[:, None] > r[('M1 H10', K)]['fc'][None, :]).mean()), "{:.0%}")
                row[f"{tag} P>아무날"] = "" if nm == RAND else P(float((f[:, None] > r[(RAND, K)]['fc'][None, :]).mean()), "{:.0%}")
                if p != aux:
                    sq.append(f.mean() - r["QQQ"]); sm.append(f.mean() - r[("M1 H10", K)]["fc"].mean())
                    sr.append(f.mean() - r[(RAND, K)]["fc"].mean())
            row["QQQ 두기간"], row["M1 두기간"], row["아무날 두기간"] = sgn(sq), sgn(sm), sgn(sr)
            rows.append(row)
    df = pd.DataFrame(rows)
    L = []
    for p in (aux, pa_, pb_):
        tag = p.split("(")[0]
        cols = ["칸", "규칙"] + [c for c in df.columns if c.startswith(tag + " ")]
        L += [f"[{p}] QQQ {P(ph[p]['QQQ'])}", df[cols].to_string(index=False), ""]
    L += ["[2026 상·하 방향]", df[["칸", "규칙", "QQQ 두기간", "M1 두기간", "아무날 두기간"]].to_string(index=False)]
    return L


def pick(allres):
    a, b = PERIODS[1][0], PERIODS[2][0]
    cand = [(nm, K) for nm in NAMES if nm != "D1 H10" for K in SLOTS]     # D1 H10 = M1 H10(같은 거래)이라 뺀다
    va = np.array([allres[a][c]["fc"].mean() for c in cand])
    vb = np.array([allres[b][c]["fc"].mean() for c in cand])
    ra = (-va).argsort().argsort() + 1
    rb = (-vb).argsort().argsort() + 1
    rho = np.corrcoef(ra, rb)[0, 1]
    L = [f"{a}에서 복리 수익 순위({len(cand)}조합) → {b}에서의 성적·순위. 두 기간 순위 상관(스피어만) {rho:+.2f}"]
    for i in np.argsort(ra)[:6]:
        nm, K = cand[i]
        L.append(f"  {nm} {K}칸: {a} {P(va[i])} (QQQ {P(allres[a]['QQQ']['fin'])}) → {b} {P(vb[i])}"
                 f" (QQQ {P(allres[b]['QQQ']['fin'])}), 순위 {rb[i]}/{len(cand)}")
    L.append(f"  {b} 1~3등: " + ", ".join(f"{cand[i][0]} {cand[i][1]}칸 {P(vb[i])}(상 순위 {ra[i]})" for i in np.argsort(rb)[:3]))
    return L


NAMES_IDX = {}


def main():
    pd.set_option("display.width", 400)
    pd.set_option("display.max_columns", 40)
    days, O, C, M, names = load_daily()
    NAMES_IDX.update({"QQQ": names.index("QQQ"), "SPY": names.index("SPY")})
    T, rules, dr, diag = load_trades(days, O, C, names)
    del O, C
    rsel = rand_days(T, dr)
    allres, ph, pops = {}, {}, {}
    for name, d0, d1 in PERIODS:
        print(name, flush=True)
        allres[name] = run_period(T, rules, rsel, days, M, d0, d1)
        ph[name] = run_phase(T, rules, rsel, days, M, d0, d1)
        pops[name] = pop_stats(T, rules, d0, d1)
    L = ["급등 후 눌림목 매수 — 자금 한도 계좌 (account.py, 2026-10-10)",
         "칸: 칸 = 계좌를 똑같이 나눈 칸 수, 거래 = 기간 안에 매수한 그 규칙의 거래 수, 받음 = 받은 거래 수(30번 평균),",
         "    사용률 = 하루 종가 시점 찬 칸 비율, 받은평균 = 받은 거래의 실제 매도까지 수익 평균(30번 중앙값),",
         "    단순합·복리 = 기간 계좌 수익(30번 평균), 상위5뺀 = 받은 거래 중 최고 5건 기여를 0으로 친 단순 합,",
         "    복리5~95% = 30번 분위, 낙폭 = 하루 종가 계좌 최대 낙폭(30번 평균), t일간 = (하루 계좌 - QQQ) t,",
         "    베타·알파t = 하루 계좌 수익의 QQQ 회귀, t거래 = 받은 거래의 QQQ 초과를 매수 날짜로 묶은 군집 t (모두 30번 중앙값),",
         "    P>QQQ = 30번 중 QQQ를 이긴 비율, P>M1·P>아무날 = 같은 칸 비교 규칙과 900쌍 승률, 붓 = 5일 묶음 부트스트랩 이길 확률.",
         "규칙: F50·F618 = 급등 뒤 1~10일 50%·61.8% 되돌림 지정가, EMA10 = 전날 EMA10 지정가, M0 = 급등 날 종가, M1 = 다음 날 시가,",
         "    D1 = 사건 뒤 1번째 날 시가(= M1, 확인용), 아무날 = 사건마다 D1~D10 중 무작위 하루 시가. H10 = 매수 뒤 10거래일 종가 매도,",
         "    TGT = 전 고점 지정가·B 아래 마감 다음 날 시가 손절·최대 20일. 거래 정의는 code/study.py, 계좌 규칙은 code/account.py 맨 위.",
         "가정: 비용 0, 지정가는 닿으면 체결, 같은 날 장중 매도 뒤 장중 매수, 기간마다 빈 계좌, 기간 끝 미청산은 그날 종가 평가, 생존 편향 있음.",
         f"점검: 거래 {diag['n']}건(H10 15규칙 + F50 TGT), 지정가 매수 {diag['lim']}건 중 시가 체결(09:30) {diag['gap_lim']}건,"
         f" F50 TGT {diag['tgt_n']}건 매도 = 시가 {diag['tgt_open']} · 장중 목표 {diag['tgt_lim']} · 종가 {diag['tgt_close']} (나머지 미청산),"
         f" 종가 매도인데 종가와 다른 가격 {diag['bad_close']}건.",
         f"탐색 범위: 규칙 {len(NAMES)}(D1 = M1 확인용이라 실제 7) × 칸 {len(SLOTS)} = 28조합 × 기간 {len(PERIODS)} × 시작 방식 2(빈 계좌·위상 0~10일 예열),"
         f" 씨앗 {SEEDS}번. 고르기→확인 순위는 28조합.", ""]
    for name, _, _ in PERIODS:
        L += table(allres[name], name) + [""]
    L += ["=" * 24 + " 기간별 방향(복리 계좌 - QQQ, - 같은 칸 M1, - 같은 칸 아무날; '두기간' = 2026 상·하가 같은 방향일 때만 의미 있게 본다)"]
    L += direction(allres) + [""]
    L += ["=" * 24 + " 고르기 → 확인"] + pick(allres) + [""]
    L += ["=" * 24 + " 시작 위상 점검(칸이 한꺼번에 비고 차서 계좌가 몇 안 되는 날짜에 몰린다 → 씨앗마다 기간 시작 0~10거래일 전부터 예열해",
          "    칸이 비는 날짜를 바꾼다. 같은 기간 창 복리 수익, 30번 = 위상 11가지 × 순서 운)"] + phase_table(ph) + [""]
    L += ["=" * 24 + " 거래표 전체 평균(계좌 제한 없음, 매수 시각 기준 기간): 거래 가중 vs 매수 날 균등 가중",
          "    몰린날 = 하루 거래 수 상위 10% 날. 계좌는 하루에 몇 칸만 받으므로 날짜 균등 쪽에 가깝다."]
    for name, _, _ in PERIODS:
        L += [f"[{name}]", pops[name]]
    path = os.path.join(ROOT, "results", "account.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
