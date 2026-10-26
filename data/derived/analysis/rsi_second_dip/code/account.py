"""RSI 두 번째 30 이탈 매수 · RSI 70 매도 — 자금 한도 계좌 시험(2026-10-10, 키 rsi_second_dip).

질문: 신호를 다 살 수 없다(1시간봉 급락주 C K2만 2026 하반기 하루 약 45건). 계좌를 칸으로 나눠 빈 칸이 있을 때만 받으면
  실제 계좌는 QQQ·SPY 보유나 '급락 다음 날 사서 5일 들고 파는' 단순 급락 매수보다 나은가.

규칙
  계좌: 시작 자금 1을 K칸(K = 3·5·10·20)으로 똑같이 나눈다. 신호를 매수 시각(te) 순서로 보고 빈 칸이 있을 때만 받는다.
    같은 시각 신호는 무작위 우선순위(씨앗 30개로 30번 반복해 분포를 본다). 칸은 매수~매도 동안 묶인다.
    빈 칸이 여럿이면 가장 먼저 빈 칸을 쓴다. 이미 들고 있는 종목의 신호는 받지 않는다.
    C(봉 마감 확정, 다음 봉 시가 체결)와 단순 급락 매수(종가 체결)는 같은 시각 매도 자리에 바로 살 수 있고(매도 먼저),
    I(봉 안 체결)는 같은 봉 안의 매도·매수 순서를 모르므로 다음 봉부터 다시 쓴다.
  단순 합: 칸 크기 = 시작 자금/K로 고정, 계좌 = 1 + 받은 거래 수익 × 1/K 의 합.
  복리: 칸마다 자기 돈을 전부 다시 넣는다(칸별 복리, 계좌 = 칸 값의 합).
  평가: 매 거래일 16:00 ET 정규장 종가(그날 16:00 전 마지막 봉 종가)로 들고 있는 종목을 평가해 하루 계좌 값을 만든다.
    기간 끝에 안 팔린 칸은 기간 마지막 날 종가로 평가한다(기간마다 빈 계좌로 새로 시작).
  비교 규칙(study.py 거래표, 급락주 신호만): C K1·C K2·C K3·I K2 (ARM30 R70 W0).
  현실 제약 점검(C K2·I K2, 매수 시점에 아는 조건으로만 거름; 매도는 원래 규칙 그대로):
    정규장매수 = 매수 봉이 정규장(09:30~16:00 ET)일 때만 받음(장 밖 얇은 체결가 매수 배제).
    깨어있을때 = 매수 시각이 한국 08~24시(사용자 주문 가능 시간, desk_system.md §1)일 때만 받음.
    체결확인(40m I K2만) = 매수 40분봉 안 5분봉 종가 하나라도 매수가 이하인 신호만 받음(꼬리 한 번 닿은 얇은 체결 배제).
      실제로는 그 지정가가 안 채워지면 포지션이 없으므로 '빼는' 것이 맞다. 매도 쪽 확인 비율은 따로 적는다(고칠 수 없어 보고만).
    깨어+확인(40m I K2만) = 깨어있을때와 체결확인을 둘 다 만족하는 신호만.
  기준: QQQ·SPY 보유(기간 직전 거래일 종가 → 기간 끝 종가).
    단순 급락 매수: 정규장 종가 기준 하루 -7% 이하인 날(급락주와 같은 정의)의 다음 거래일 종가에 사서 5거래일 뒤 종가에 판다.
    같은 K칸 계좌·같은 무작위 우선순위 방식으로 돌린다.
  기간(매수 시각 기준): 1h = 2025(보조) / 2026 상 1~6월(고르기) / 2026 하 7/1~10/7(확인).
    40m = 급락987 앞 8/12~8/31 / 뒤 9/1~10/9. 급락987은 종목을 8/11 이후 급락으로 골랐으므로 study.py 거래표가 이미 신호 시각에 아는
    표시(8/11 이후 급락일만)로 거른다(10/10 재검증 뒤; 예전의 아래 '엄격' 거르기는 신호 시각을 보지 않아 쓰지 않는다).
    단순 급락 매수도 8/11 이후 급락일만.
  지표: 계좌 수익(30번 평균, [5%·95%]), 최대 낙폭(하루 종가 계좌 기준, 30번 평균),
    t일간 = 하루 계좌 수익률 - 하루 QQQ 수익률의 t(날짜 하나 = 관측 하나, 복리 계좌, 30번 중앙값),
    t거래 = 받은 거래의 수익을 매수 날짜로 묶은 날짜 군집 t(30번 중앙값, 거래 수익은 실제 매도까지 = study.py와 같음),
    P>QQQ = 30번 중 QQQ보다 높은 비율, P>급락 = 같은 K 단순 급락 매수 30번과 짝지은 900쌍 중 이긴 비율,
    붓QQQ·붓급락 = 30번 평균 하루 로그 초과수익을 5거래일 묶음 부트스트랩(2,000번)한 합이 0보다 클 확률.
가정·한계
  수수료·세금·호가 차이 0(사용자 매매 조건은 docs/desk_system.md §1). 신호가 뜬 그 가격에 바로 다 체결된다고 본다.
  프리·애프터 얇은 체결가(I의 장 밖 체결)는 그대로 둔다. 현재 상장·시총 3억 달러 이상 종목이라 생존 편향이 있다.
  1h는 40분봉 RSI14를 1시간봉 RSI9로 근사한 값이다. 40m 앞 기간은 14거래일뿐이라 t가 약하다.
  탐색 범위: 규칙 4 × 칸 4 = 16조합(+ 기준 급락 매수 4, 현실 제약 점검 1h 4·40m 6 변형 × 칸 4)을 기간 5개에서 본다.
    2026 상(40m은 앞)에서 16조합 순위를 매겨 2026 하(뒤)에서 확인한다. 30번 반복은 같은 시각 순서의 운만 바꾼다(표본 불확실성은 부트스트랩).
사용: py account.py          → results/account.txt (1h·40m 둘 다)
     py account.py 1h | 40m → 한쪽만(결과 파일은 그쪽만 담아 덮어쓴다)
  종가 캐시: scratchpad acct_marks_<원천>.parquet·acct_crash_<원천>.parquet(지우면 봉 파일에서 다시 만든다).
메모리: 거래표는 필요한 열·규칙만, 봉 파일은 종목 하나씩 읽고 버린다.
"""
import glob, os, sys
from datetime import date
import numpy as np, pandas as pd
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
AN = os.path.normpath(os.path.join(ROOT, ".."))
SP = r"C:/Users/geunt/AppData/Local/Temp/claude/C--Users-geunt-OneDrive-Desktop---/6a57b230-3ac1-4ae6-9d34-b1d8a1c2ce1b/scratchpad"
B1H = os.path.join(AN, "premarket_shock", "bars")
B5M = os.path.join(SP, "raw5m_rsi2")
RULES = ["C K1 ARM30 R70 W0", "C K2 ARM30 R70 W0", "C K3 ARM30 R70 W0", "I K2 ARM30 R70 W0"]
SHORT = {r: r.split(" ARM")[0] for r in RULES}
BENCH = "급락 단순매수"
VARS = ["C K2 정규장매수", "C K2 깨어있을때", "I K2 정규장매수", "I K2 깨어있을때", "I K2 체결확인", "I K2 깨어+확인"]   # 현실 제약 점검(체결확인은 40m만)
ORDER = [BENCH] + [SHORT[r] for r in RULES] + VARS
SLOTS = (3, 5, 10, 20)
SEEDS = 30
CRASH = -0.07
HOLD = 5
INF = np.iinfo(np.int64).max
NS_DAY, NS_16 = 86_400_000_000_000, 16 * 3_600_000_000_000
PERIODS = {"1h": [("2025(보조)", date(2025, 1, 1), date(2026, 1, 1)),
                  ("2026 상(고르기)", date(2026, 1, 1), date(2026, 7, 1)),
                  ("2026 하(확인)", date(2026, 7, 1), date(2026, 12, 31))],
           "40m": [("앞 8/12~8/31", date(2026, 8, 12), date(2026, 9, 1)),
                   ("뒤 9/1~10/9", date(2026, 9, 1), date(2026, 12, 31))]}


# ---------------------------------------------------------------- 하루 종가·급락 이벤트(종목 하나씩 읽고 버림)
def daily(d):
    """봉 → (정규장 종가: 09:30~16:00 봉 마지막 종가, 평가 종가: 16:00 전 마지막 봉 종가). 날짜는 tz 없는 자정."""
    idx = d.index.tz_localize(None)
    mod = idx.hour * 60 + idx.minute
    c = d.Close.values.astype(float)
    dd = idx.normalize()
    reg, pre = (mod >= 570) & (mod < 960), mod < 960
    rc = pd.Series(c[reg], index=dd[reg]).groupby(level=0).last()
    mk = pd.Series(c[pre], index=dd[pre]).groupby(level=0).last()
    return rc, mk


def crash_events(sym, rc, since):
    """하루 -7% 이하(정규장 종가) → 다음 거래일 종가 매수, 그 5거래일 뒤 종가 매도. 자료 끝이면 마지막 종가 평가(미청산)."""
    r = rc.pct_change().values
    days = rc.index
    out = []
    for k in np.flatnonzero(r <= CRASH):
        if days[k] < since or k + 1 >= len(days):
            continue
        b, s = k + 1, k + 1 + HOLD
        op = s >= len(days)
        out.append((sym, days[k], days[b], pd.NaT if op else days[s], rc.iloc[b], rc.iloc[-1] if op else rc.iloc[s], op))
    return out


def load_marks(src):
    fm, fc = os.path.join(SP, f"acct_marks_{src}.parquet"), os.path.join(SP, f"acct_crash_{src}.parquet")
    if os.path.exists(fm) and os.path.exists(fc):
        return pd.read_parquet(fm).astype(float), pd.read_parquet(fc)
    if src == "1h":
        files, cut, since = sorted(glob.glob(os.path.join(B1H, "1h_*.parquet"))), pd.Timestamp("2024-12-01"), pd.Timestamp("2024-12-01")
    else:
        files, cut, since = sorted(glob.glob(os.path.join(B5M, "5m_*.parquet"))), pd.Timestamp("2026-07-01"), pd.Timestamp("2026-08-11")
    marks, ev = {}, []
    for i, f in enumerate(files):
        s = os.path.basename(f).split("_", 1)[1][:-8]
        d = pd.read_parquet(f, columns=["Close", "High"])
        d = d[(d.High > 0) & (d.index.tz_localize(None) >= cut)]
        if len(d) < 10:
            continue
        rc, mk = daily(d)
        marks[s] = mk.astype("float32")
        if s not in ("SPY", "QQQ"):
            ev += crash_events(s, rc, since)
        if i % 500 == 0:
            print(src, "종가", i, "/", len(files), flush=True)
    cal = marks["SPY"].index
    M = pd.DataFrame(marks).reindex(cal).ffill()
    M.columns = [str(c) for c in M.columns]
    C = pd.DataFrame(ev, columns=["sym", "cday", "bday", "sday", "ent", "px", "open"])
    M.astype("float32").to_parquet(fm)
    C.to_parquet(fc)
    return M.astype(float), C


# ---------------------------------------------------------------- 신호표
def ns(s):
    """tz 붙은 시각 열 → ET 벽시계 ns(int64), NaT → INF."""
    v = s.dt.tz_localize(None).values.astype("datetime64[ns]").astype(np.int64)
    v[pd.isna(s).values] = INF
    return v


def _pack(x, le):
    return dict(sym=x.sym.values.astype(str), te=ns(x.te), tx=ns(x.tx), ent=x.ent.values.astype(float),
                px=x.px.values.astype(float), le=le)


def load_signals(src, crash_ev):
    cols = ["규칙", "sym", "day", "te", "tx", "ent", "px", "e_reg"]
    if src == "1h":
        t = pd.read_parquet(os.path.join(SP, "rsi2_trades_1h.parquet"), columns=cols,
                            filters=[("규칙", "in", RULES), ("crash", "==", True)])
    else:
        t = pd.read_parquet(os.path.join(SP, "rsi2_trades_40m.parquet"), columns=cols + ["src"],
                            filters=[("규칙", "in", RULES), ("src", "==", "급락987")])
        # study.py 거래표가 이미 신호 시각에 아는 급락 표시(8/11 이후 급락일만)로 거른다. 예전 strict_flags는
        # 신호 시각을 보지 않아 '그날 -7%로 끝나서만 통과한' 신호 57건을 남겼으므로 쓰지 않는다(10/10 재검증).
    t = t[(t.ent > 0) & (t.px > 0)]
    out = {}
    for r in RULES:
        x = t[t.규칙 == r]
        out[SHORT[r]] = _pack(x, not r.startswith("I"))
    for r in ("C K2 ARM30 R70 W0", "I K2 ARM30 R70 W0"):              # 현실 제약 점검(매수 시점에 아는 조건만)
        x = t[t.규칙 == r]
        out[SHORT[r] + " 정규장매수"] = _pack(x[x.e_reg.values], not r.startswith("I"))
        awake = x.te.dt.tz_convert("Asia/Seoul").dt.hour.values >= 8          # 한국 08~24시 = 사용자 주문 가능 시간
        out[SHORT[r] + " 깨어있을때"] = _pack(x[awake], not r.startswith("I"))
    if src == "40m":                                                     # 5분봉으로 I 체결가가 실제로 거래됐는지 확인
        x = t[t.규칙 == "I K2 ARM30 R70 W0"]
        ce, cx = confirm_5m(x)
        out["I K2 체결확인"] = _pack(x[ce], False)
        aw = x.te.dt.tz_convert("Asia/Seoul").dt.hour.values >= 8
        out["I K2 깨어+확인"] = _pack(x[ce & aw], False)
        r_ = (x.px / x.ent - 1).values
        lab = np.where(x.e_reg.values, "정규장", "장밖")
        for b_, m_ in (("전체", np.ones(len(x), bool)), ("장밖 매수", lab == "장밖"), ("정규장 매수", lab == "정규장")):
            DIAG.append(f"  I K2 40m 신호 {b_} {m_.sum()}건: 매수 체결 확인 {ce[m_].mean():.0%}"
                        f" (확인 평균 {np.nanmean(r_[m_ & ce]):+.2%}, 미확인 평균 {np.nanmean(r_[m_ & ~ce]):+.2%}),"
                        f" 매도 체결 확인 {np.nanmean(cx[m_ & ~np.isnan(cx)]):.0%}")
    b = crash_ev
    te = (pd.to_datetime(b.bday).values.astype("datetime64[ns]").astype(np.int64) + NS_16)
    tx = pd.to_datetime(b.sday).values.astype("datetime64[ns]").astype(np.int64) + NS_16
    tx[b.sday.isna().values] = INF
    out[BENCH] = dict(sym=b.sym.values.astype(str), te=te, tx=tx, ent=b.ent.values.astype(float), px=b.px.values.astype(float), le=True)
    return out


DIAG = []


def confirm_5m(x):
    """I 체결 확인(40m 전용): 매수 40분봉 안 5분봉 종가 중 하나라도 매수가 이하면 '매수 확인',
    매도 40분봉 안 5분봉 종가 중 하나라도 매도가 이상이면 '매도 확인'(미청산은 NaN). 꼬리 한 번(얇은 체결)만 닿은 체결을 가려낸다."""
    ce = np.zeros(len(x), bool)
    cx = np.full(len(x), np.nan)
    pos = {k: i for i, k in enumerate(x.index)}
    for s, g in x.groupby("sym"):
        m5 = pd.read_parquet(os.path.join(B5M, f"5m_{s}.parquet"), columns=["Close", "High"])
        m5 = m5[m5.High > 0]
        tk = m5.index.tz_localize(None).values.astype("datetime64[ns]").astype(np.int64)
        c = m5.Close.values
        W = 40 * 60_000_000_000
        for k, te, tx, ent, px in zip(g.index, ns(g.te), ns(g.tx), g.ent.values, g.px.values):
            a, b = np.searchsorted(tk, te), np.searchsorted(tk, te + W)
            ce[pos[k]] = b > a and c[a:b].min() <= ent * (1 + 1e-9)
            if tx < INF:
                a, b = np.searchsorted(tk, tx), np.searchsorted(tk, tx + W)
                cx[pos[k]] = float(b > a and c[a:b].max() >= px * (1 - 1e-9))
    return ce, cx


# ---------------------------------------------------------------- 계좌
@njit(cache=True)
def accept(te, tx, symi, order, K, le):
    """매수 시각 순서대로 빈 칸(가장 먼저 빈 칸)에 받는다. 반환: 칸 번호(-1 = 못 받음)."""
    n = len(te)
    slot = np.full(n, -1, np.int64)
    free = np.full(K, -INF, np.int64)
    held = np.full(K, -1, np.int64)
    for q in range(len(order)):
        j = order[q]
        t = te[j]
        best, bf, dup = -1, INF, False
        for k in range(K):
            busy = free[k] > t if le else free[k] >= t
            if busy:
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


def run_period(sig, M, name, d0, d1):
    cal = M.index
    days = cal[(cal >= pd.Timestamp(d0)) & (cal < pd.Timestamp(d1))]
    base = cal[cal < pd.Timestamp(d0)][-1]
    snapd = days.insert(0, base)
    snaps = snapd.values.astype("datetime64[ns]").astype(np.int64) + NS_16
    lo, hi = pd.Timestamp(d0).value, snaps[-1]
    Mp = M.loc[snapd]
    colmap = {s: i for i, s in enumerate(Mp.columns)}
    Mv = Mp.values
    q = Mv[:, colmap["QQQ"]] / Mv[0, colmap["QQQ"]]
    sp = Mv[:, colmap["SPY"]] / Mv[0, colmap["SPY"]]
    lq = np.diff(np.log(q))
    rq = np.diff(q) / q[:-1]
    res = {"QQQ": dict(fin=q[-1] - 1, mdd=mdd(q)), "SPY": dict(fin=sp[-1] - 1, mdd=mdd(sp)), "_n": len(days)}
    for rn, g in sig.items():
        m = (g["te"] >= lo) & (g["te"] < hi) & np.isin(g["sym"], Mp.columns)
        te, tx, ent, px = g["te"][m], g["tx"][m], g["ent"][m], g["px"][m]
        sym = g["sym"][m]
        col = np.array([colmap[s] for s in sym], np.int64)
        u, symi = np.unique(sym, return_inverse=True)
        eday = (te // NS_DAY)
        ret = px / ent - 1
        res[(rn, "신호")] = len(te)
        for K in SLOTS:
            R = dict(fs=[], fc=[], ms=[], mc=[], n=[], occ=[], td=[], tt=[], avg=[], lc=[])
            for s in range(SEEDS):
                rnd = np.random.default_rng(1000 + s).random(len(te))
                order = np.lexsort((rnd, te)).astype(np.int64)
                slot = accept(te, tx, symi.astype(np.int64), order, K, g["le"])
                acc = order[slot[order] >= 0]
                es, _ = equity(te, tx, ent, px, col, slot, acc, snaps, Mv, K, False)
                ec, oc = equity(te, tx, ent, px, col, slot, acc, snaps, Mv, K, True)
                R["fs"].append(es[-1] - 1); R["fc"].append(ec[-1] - 1)
                R["ms"].append(mdd(es)); R["mc"].append(mdd(ec))
                R["n"].append(len(acc)); R["occ"].append(oc[1:].mean())
                rc = np.diff(ec) / ec[:-1]
                R["td"].append(tstat(rc - rq))
                R["lc"].append(np.diff(np.log(ec)))
                if len(acc):
                    dm = pd.Series(ret[acc]).groupby(eday[acc]).mean().values
                    R["tt"].append(tstat(dm)); R["avg"].append(ret[acc].mean())
            for k_ in ("fs", "fc", "ms", "mc", "n", "occ", "td", "tt", "avg"):
                R[k_] = np.array(R[k_], float)
            R["lc"] = np.mean(R["lc"], axis=0)
            res[(rn, K)] = R
    # 비교 확률
    for rn in sig:
        for K in SLOTS:
            R, Bn = res[(rn, K)], res[(BENCH, K)]
            R["pq"] = float((R["fc"] > res["QQQ"]["fin"]).mean())
            R["ps"] = float((R["fc"] > res["SPY"]["fin"]).mean())
            R["pb"] = float((R["fc"][:, None] > Bn["fc"][None, :]).mean())
            R["bq"] = boot_p(R["lc"] - lq)
            R["bb"] = boot_p(R["lc"] - Bn["lc"]) if rn != BENCH else np.nan
    return res


# ---------------------------------------------------------------- 출력
def P(x, f="{:+.1%}"):
    return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f.format(x)


def table(res, name):
    n = res["_n"]
    L = [f"--- {name}  ({n}거래일)   QQQ {P(res['QQQ']['fin'])} (낙폭 {P(res['QQQ']['mdd'], '{:.1%}')})"
         f"   SPY {P(res['SPY']['fin'])} (낙폭 {P(res['SPY']['mdd'], '{:.1%}')})"]
    rows = []
    for K in SLOTS:
        for rn in [r for r in ORDER if (r, K) in res]:
            R = res[(rn, K)]
            rows.append({"칸": K, "규칙": rn, "신호/일": f"{res[(rn, '신호')] / max(n, 1):.1f}", "받음": f"{R['n'].mean():.0f}",
                         "사용률": P(R["occ"].mean(), "{:.0%}"), "받은평균": P(np.nanmedian(R["avg"]) if len(R["avg"]) else np.nan, "{:+.2%}"),
                         "단순합": P(R["fs"].mean()), "복리": P(R["fc"].mean()),
                         "복리5~95%": f"{P(np.quantile(R['fc'], .05))}~{P(np.quantile(R['fc'], .95))}",
                         "낙폭단순": P(R["ms"].mean(), "{:.1%}"), "낙폭복리": P(R["mc"].mean(), "{:.1%}"),
                         "t일간QQQ초과": P(np.nanmedian(R["td"]), "{:+.2f}"), "t거래": P(np.nanmedian(R["tt"]) if len(R["tt"]) else np.nan, "{:+.2f}"),
                         "P>QQQ": P(R["pq"], "{:.0%}"), "P>급락": "" if rn == BENCH else P(R["pb"], "{:.0%}"),
                         "붓QQQ": P(R["bq"], "{:.0%}"), "붓급락": "" if rn == BENCH else P(R["bb"], "{:.0%}")})
    L.append(pd.DataFrame(rows).to_string(index=False))
    return L


def direction(allres):
    """규칙×칸마다 기간별 (복리 - QQQ), (복리 - 같은 칸 급락 매수) 부호와 두 기간 같은 방향 여부."""
    L = []
    for src, pairs in (("1h", ("2026 상(고르기)", "2026 하(확인)", "2025(보조)")), ("40m", ("앞 8/12~8/31", "뒤 9/1~10/9"))):
        if src not in allres:
            continue
        rs = allres[src]
        rows = []
        for K in SLOTS:
            for rn in [r for r in ORDER[1:] + [BENCH] if (r, K) in rs[pairs[0]]]:
                row = {"원천": src, "칸": K, "규칙": rn}
                sq, sb = [], []
                for p in pairs:
                    R = rs[p][(rn, K)]
                    dq = R["fc"].mean() - rs[p]["QQQ"]["fin"]
                    db = R["fc"].mean() - rs[p][(BENCH, K)]["fc"].mean()
                    row[f"{p} -QQQ"] = P(dq)
                    if rn != BENCH:
                        row[f"{p} -급락"] = P(db)
                    sq.append(np.sign(dq)); sb.append(np.sign(db))
                main = slice(0, 2)
                row["QQQ 두기간"] = "같음+" if all(x > 0 for x in sq[main]) else ("같음-" if all(x < 0 for x in sq[main]) else "엇갈림")
                if rn != BENCH:
                    row["급락 두기간"] = "같음+" if all(x > 0 for x in sb[main]) else ("같음-" if all(x < 0 for x in sb[main]) else "엇갈림")
                rows.append(row)
        L += [f"[{src}] 두 기간 = {pairs[0]}·{pairs[1]}" + (f" (보조 {pairs[2]}는 칸만 보임)" if len(pairs) > 2 else ""),
              pd.DataFrame(rows).fillna("").to_string(index=False), ""]
    return L


def pick(allres):
    L = []
    for src, a, b in (("1h", "2026 상(고르기)", "2026 하(확인)"), ("40m", "앞 8/12~8/31", "뒤 9/1~10/9")):
        if src not in allres:
            continue
        rs = allres[src]
        cand = [(rn, K) for rn in [SHORT[r] for r in RULES] for K in SLOTS]
        sa = sorted(cand, key=lambda c: -rs[a][c]["fc"].mean())
        sb_ = sorted(cand, key=lambda c: -rs[b][c]["fc"].mean())
        rk = {c: i + 1 for i, c in enumerate(sb_)}
        L.append(f"[{src}] {a}에서 복리 수익 순위(16조합) → {b}에서의 성적·순위")
        for c in sa[:5]:
            ra, rb = rs[a][c], rs[b][c]
            L.append(f"  {c[0]} {c[1]}칸: {a} {P(ra['fc'].mean())} (QQQ {P(rs[a]['QQQ']['fin'])}, 급락 {P(rs[a][(BENCH, c[1])]['fc'].mean())})"
                     f" → {b} {P(rb['fc'].mean())} (QQQ {P(rs[b]['QQQ']['fin'])}, 급락 {P(rs[b][(BENCH, c[1])]['fc'].mean())}), 순위 {rk[c]}/16")
        L.append("")
    return L


def main():
    srcs = [sys.argv[1]] if len(sys.argv) > 1 else ["1h", "40m"]
    pd.set_option("display.width", 300)
    allres = {}
    for src in srcs:
        M, C = load_marks(src)
        sig = load_signals(src, C)
        allres[src] = {}
        for name, d0, d1 in PERIODS[src]:
            print(src, name, flush=True)
            allres[src][name] = run_period(sig, M, name, d0, d1)
        del M, C, sig
    L = ["RSI 두 번째 30 이탈 매수 · RSI 70 매도 — 자금 한도 계좌 (account.py, 2026-10-10)",
         "칸: 칸 = 계좌를 똑같이 나눈 칸 수, 신호/일 = 기간 안 급락주 신호(매수 시각 기준)/거래일, 받음 = 받은 거래 수(30번 평균),",
         "    사용률 = 하루 종가 시점 찬 칸 비율, 받은평균 = 받은 거래의 실제 매도까지 수익 평균(30번 중앙값),",
         "    단순합·복리 = 기간 계좌 수익(30번 평균), 복리5~95% = 30번의 5%·95% 분위, 낙폭 = 하루 종가 계좌 최대 낙폭(30번 평균),",
         "    t일간QQQ초과 = (하루 계좌 수익 - QQQ) 날짜 t, t거래 = 받은 거래의 매수 날짜 군집 t, P>QQQ = 30번 중 QQQ를 이긴 비율,",
         "    P>급락 = 같은 칸 단순 급락 매수와 900쌍 비교 승률, 붓QQQ·붓급락 = 5일 묶음 부트스트랩으로 본 이길 확률.",
         "규칙: C = 봉 마감 확정 다음 시가, I = 봉 안 RSI 30·70 가격, K = 몇 번째 30 이탈. 급락 단순매수 = -7% 다음 날 종가 매수, 5거래일 뒤 종가 매도.",
         "    정규장매수 = 매수 봉이 정규장일 때만 받음, 깨어있을때 = 매수 시각이 한국 08~24시일 때만 받음(둘 다 매도는 원래 규칙).",
         "가정: 비용 0, 신호가 그대로 체결, 기간마다 빈 계좌로 시작, 기간 끝 미청산은 그날 종가 평가. 자세한 규칙은 code/account.py 맨 위.", ""]
    for src in srcs:
        L.append("=" * 24 + f" {src} ({'1시간봉 RSI9 근사, 3,570종목' if src == '1h' else '진짜 40분봉 RSI14, 급락987(신호 시각 급락 표시)'})")
        for name, _, _ in PERIODS[src]:
            L += table(allres[src][name], name) + [""]
        if src == "40m" and DIAG:
            L += ["5분봉 체결 확인(40m 급락987 I K2 전체 신호, 8/12~10/9; 확인 = 그 40분봉 안 5분봉 종가가 체결가를 넘어섬):"] + DIAG + [""]
    L += ["=" * 24 + " 두 기간 방향(복리 계좌 수익 - QQQ, - 같은 칸 급락 매수; '같음'일 때만 의미 있게 본다)"] + direction(allres)
    L += ["=" * 24 + " 고르기 → 확인(앞 기간 1~5등이 뒤 기간에 어떻게 됐나)"] + pick(allres)
    path = os.path.join(ROOT, "results", "account.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
