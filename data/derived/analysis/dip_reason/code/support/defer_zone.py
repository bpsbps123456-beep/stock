"""이격매수 '한 단계 아래 지지 구역에서 사기' 연구(2026-10-10 사용자 아이디어, 키 defer_zone).

질문(사용자): 이격 매수 신호 뒤 바로 반등하는 경우가 적고 대부분 횡보 뒤 오른다면, 신호 때 바로 사지 말고 신호 가격 아래
  첫 번째(Z1) 지지 구역 하단에 지정가를 걸어 사면 어떤가. 두 번째(Z2) 하단이 최선일 것 같지만 매수가 너무 적을 것 같다.

신호: 지금 지표(dip_scalper.pine, stop_lab.sim)와 같다. 저가가 EMA[i-1] x 0.98에 닿음, 전날 +7% 다음 날 정규장 안 삼.
  신호 가격 P = min(시가, 매수선). 지금 지표는 여기서 산다.
지지 구역: 사용자 트레이딩뷰 지표의 구역 계산(sr_zones_ref.py·zone40.py와 같음)을 차트 봉(기본 40분봉, 04:00 ET 기준)으로,
  신호 봉 시작 때 확정된 스윙만 써서 만든다(스윙 봉 i는 i+5봉이 끝난 시각부터).
  Zk = 상단 < P인 구역을 가격이 높은 쪽부터 셀 때 k번째. 주문가 L = 그 구역 하단(구역 안 가장 낮은 스윙 가격).
  k번째 구역이 없으면 주문하지 않고, 그 주문 수명 동안 새 신호도 받지 않는다(지금 지표가 들고 있는 동안 안 사는 것과 같게).
주문 수명(신호 1개에 주문 1개, 수명 동안 새 신호 무시)
  L1 원래 신호가 팔렸을 시각까지(RSI>70 다음 봉 또는 신호 뒤 첫 한국 아침 8시). 체결되면 지금 지표와 같은 봉·같은 가격에 판다.
  L2 신호 뒤 첫 한국 아침 8시까지(자기 전에 걸어 두는 하루 주문).
  L3 신호 뒤 세 번째 한국 아침 8시까지(거래가 있는 주기로 셈, 금요일 신호는 화요일 아침 8시까지).
  Lr 정규장이 끝날 때까지(혼합 연구용, defer_zone_hybrid.py).
체결: 신호 다음 봉부터 저가 <= L이면 min(시가, L). 정규장만 범위면 정규장 봉에서만 체결.
매도: 지금 지표와 같다. RSI(기준봉) > 70인 봉 다음 봉 시가, 아니면 체결 뒤 첫 한국 아침 8시 이후 첫 봉 시가.
대조군(지지 구역 고유의 힘이 있나)
  고정 깊이: L = P x (1-d), d = 1~8%.
  무작위 거리: 같은 신호·같은 구역 유무, 거리만 같은 종목의 실제 Zk 거리 모음에서 무작위로 뽑음(200번).
  구역 이동: L = 하단 x (1±0.6%), 구역 상단.
기간: 고르기 early 7/21~9/8(5분봉 근사, EMA36·RSI9, 7/14 자료 시작 뒤 1주는 구역 기억이 짧아 뺌),
      확인 late 9/10~10/6(3분봉), 보류 oos 10/7~10/9(3분봉, 새로 받은 자료). 신호 날짜(ET) 기준으로 기간에 넣는다.
사용: py defer_zone.py [raw_새자료] → results/support/defer_zone.txt, defer_zone_trades.parquet(scratchpad)
"""
import os, sys
from datetime import date
import numpy as np, pandas as pd
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, CODE)
sys.path.insert(0, HERE)
import stop_lab as SL
from stop_lab import USER78, ALL
from zone40 import cluster, N, KEEP

OUT = os.path.normpath(os.path.join(CODE, "../results/support/defer_zone.txt"))
RAW_OLD = SL.RAW
RAW_NEW = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    r"C:/Users/geunt/AppData/Local/Temp/claude/C--Users-geunt-OneDrive-Desktop---/6a57b230-3ac1-4ae6-9d34-b1d8a1c2ce1b/scratchpad", "raw_1010")
TMP = os.path.dirname(RAW_NEW)
WIN = {"early": (date(2026, 7, 21), date(2026, 9, 8)), "late": (date(2026, 9, 10), date(2026, 10, 6)),
       "oos": (date(2026, 10, 7), date(2026, 10, 9))}
PN = {"early": "고르기 7/21~9/8(5분봉)", "late": "확인 9/10~10/6(3분봉)", "oos": "보류 10/7~10/9(3분봉)"}
DESK7 = {"AMD", "ARM", "DELL", "INTC", "SNDK", "STX", "WDC"}           # kalman_zones 연구 때 야간 데스크 7종목(참고용 작은 묶음)
UNIV = {"사용자78": USER78, "전체137": set(ALL), "데스크7": DESK7}
LEV = {"SOXL", "SOXS", "TQQQ", "SQQQ", "TECL", "TECS", "LABU", "LABD", "NAIL", "DPST", "KORU", "SPXL", "SPXS", "UPRO",
       "SPXU", "TNA", "TZA", "FNGU", "FNGD", "YINN", "YANG"}           # Pine levList: 이격 3%
DAY = 86_400 * 10**9
NSEED = 200


# ---------------------------------------------------------------- 자료
def load_period(p):
    if p in ("early", "late"):
        return SL.load(p)
    folder = os.path.join(TMP, "defer_oos")
    os.makedirs(folder, exist_ok=True)
    S, old = {}, SL.RAW
    SL.RAW = RAW_NEW
    try:
        for s in ALL:
            f = os.path.join(folder, f"{s}.npz")
            if os.path.exists(f):
                S[s] = SL._read(f, s)
                continue
            try:
                A = SL._arrays(s, 3, 60, 14)
            except FileNotFoundError:
                continue
            SL._save(A, f)
            S[s] = A
    finally:
        SL.RAW = old
    return S


_Z = {}


def zone_sets(raw, s, tf=40):
    """확정 스윙 수 k마다의 구역 묶음(zone40.zone_sets와 같은 계산, 원자료·차트 봉 길이만 인자)."""
    key = (raw, s, tf)
    if key in _Z:
        return _Z[key]
    m5 = pd.read_parquet(f"{raw}/5m_{s}.parquet")
    m5 = m5[m5.High > 0]
    b = m5.resample(f"{tf}min", origin="start_day", offset="4h", label="left", closed="left").agg(
        {"High": "max", "Low": "min", "Close": "last"}).dropna(subset=["Close"])
    hs = b.index.as_unit("ns").asi8
    h, l = b.High.values.astype(float), b.Low.values.astype(float)
    px, bi = [], []
    for i in range(N, len(h) - N):
        if l[i] == l[i - N:i + N + 1].min() and (l[i - N:i] > l[i]).all():
            px.append(l[i]); bi.append(i)
        if h[i] == h[i - N:i + N + 1].max() and (h[i - N:i] < h[i]).all():
            px.append(h[i]); bi.append(i)
    px, bi = np.array(px, float), np.array(bi, np.int64)
    conf = hs[bi + N] + tf * 60 * 10**9
    sets = [cluster(px[max(0, k - KEEP):k], bi[max(0, k - KEEP):k]) for k in range(len(px) + 1)]
    J = max(1, max(len(z) for z in sets))
    ZLO, ZHI = np.full((len(sets), J), np.inf), np.full((len(sets), J), np.inf)
    for k, zs in enumerate(sets):
        for j, z in enumerate(zs):
            ZLO[k, j], ZHI[k, j] = z[0], z[1]
    _Z[key] = dict(conf=conf, ZLO=ZLO, ZHI=ZHI)
    return _Z[key]


def zone_index(A, Z):
    sid = np.searchsorted(Z["conf"], A["tk"], side="right")       # 봉 시작 때까지 확정된 스윙 수
    used = sid > 0
    assert (Z["conf"][np.maximum(sid - 1, 0)][used] <= A["tk"][used]).all()   # 미래 참조 없음
    return sid


# ---------------------------------------------------------------- 상태 기계
@njit(cache=True)
def _target(P, row_lo, row_hi, k, lev, incl):
    """P 아래 k번째 구역의 주문가. 없으면 nan. lev 0 하단 / 1 상단 / 2 하단-0.6% / 3 하단+0.6%."""
    cnt = 0
    for j in range(row_lo.shape[0] - 1, -1, -1):
        zl, zh = row_lo[j], row_hi[j]
        if zl == np.inf:
            continue
        below = zl < P if incl else zh < P
        if below:
            cnt += 1
            if cnt == k:
                if lev == 0:
                    return zl
                if lev == 1:
                    return zh
                if lev == 2:
                    return zl * 0.994
                return zl * 1.006
    return np.nan


@njit(cache=True)
def machine(o, l, r, ema, tk, dead_of, reg, hot, sid, ZLO, ZHI, gap, mode, k, lev, incl, life, reg_only,
            from_sig, fixd, rd, X, cyc, hyb, dfr):
    """mode 0 지금 지표 / 1 구역 / 2 고정 깊이 fixd / 3 구역 유무는 같고 거리만 rd[주문 순번].
    신호마다 한 행: s, P, L, f(체결 봉, -1), x(매도 봉, -1), ent, px, why(0 RSI·1 8시), stat
    stat 0 체결·매도 / 1 미체결 취소 / 2 구역 없음(주문 안 함) / 3 자료 끝(미완)
    cyc = 봉이 속한 '한국 아침 8시 주기'의 거래 주기 순번(봉이 있는 주기만 셈, 주말은 건너뜀)
    life 4 = 정규장이 끝날 때까지(신호 뒤 첫 정규장 밖 봉에서 취소). hyb = dfr[i]가 참인 봉의 신호만 구역·깊이 주문, 나머지 신호는 지금처럼 P에 삼(dfr 기본 = 정규장)"""
    n = len(o)
    S = np.full(n, -1, np.int64); F = np.full(n, -1, np.int64); XI = np.full(n, -1, np.int64)
    PP = np.full(n, np.nan); LL = np.full(n, np.nan); EN = np.full(n, np.nan); PX = np.full(n, np.nan)
    WH = np.full(n, -1, np.int64); ST = np.full(n, 3, np.int64)
    m, state, nord = 0, 0, 0
    s = f = 0
    L = ent = 0.0
    dead_s = dead = 0
    for i in range(61, n):
        if state == 2:
            why = -1
            if i - 1 >= f and r[i - 1] > X:
                why = 0
            elif tk[i] >= dead:
                why = 1
            if why >= 0:
                XI[m], PX[m], WH[m], ST[m] = i, o[i], why, 0
                m += 1
                state = 0
            continue
        if state == 1:
            if life == 1:
                cancel = (i - 1 >= s and r[i - 1] > X) or tk[i] >= dead_s
            elif life == 2:
                cancel = tk[i] >= dead_s
            elif life == 4:
                cancel = not reg[i]
            else:
                cancel = cyc[i] >= cyc[s] + 3                      # 거래가 있는 8시 주기로 세 번째 8시
            if cancel:
                if ST[m] != 2:
                    ST[m] = 1
                m += 1
                state = 0
                continue
            if L == L and (not reg_only or reg[i]) and l[i] <= L:
                ent = min(o[i], L)
                f, dead, state = i, dead_of[i], 2
                F[m], EN[m] = f, ent
            continue
        if hot[i] or (reg_only and not reg[i]):
            continue
        lvl = ema[i - 1] * (1 - gap)
        if l[i] > lvl:
            continue
        P = min(o[i], lvl)
        s, dead_s = i, dead_of[i]
        S[m], PP[m] = s, P
        if mode == 0 or (hyb and not dfr[i]):
            LL[m], F[m], EN[m] = P, i, P
            ent, f, dead, state = P, i, dead_s, 2
            continue
        if mode == 2:
            L = P * (1 - fixd)
        else:
            L = _target(P, ZLO[sid[i]], ZHI[sid[i]], k, lev, incl)
            if L == L and mode == 3:
                L = P * (1 - rd[nord])
        LL[m] = L
        if L != L:
            ST[m] = 2
        else:
            nord += 1
        state = 1
        if from_sig and L == L and l[i] <= L:
            ent = min(o[i], L)
            f, dead, state = i, dead_of[i], 2
            F[m], EN[m] = f, ent
    if state != 0:
        m += 1
    return S[:m], PP[:m], LL[:m], F[:m], XI[:m], EN[:m], PX[:m], WH[:m], ST[:m]


# ---------------------------------------------------------------- 실행
CTX = {}


def ctx(p, tf=40):
    key = (p, tf)
    if key in CTX:
        return CTX[key]
    S = load_period(p)
    raw = RAW_NEW if p == "oos" else RAW_OLD
    out = {}
    for s, A in S.items():
        try:
            Z = zone_sets(raw, s, tf)
        except FileNotFoundError:
            continue
        _, cyc = np.unique((A["tk"] + 3_600 * 10**9) // DAY, return_inverse=True)   # KST 08시 = UTC 23시 경계
        A["cyc"] = cyc.astype(np.int64)
        out[s] = (A, Z, zone_index(A, Z))
    CTX[key] = out
    return out


def run(p, spec, reg_only=False, tf=40, rd_of=None):
    """spec: dict(mode, k, lev, incl, life, from_sig, fixd). 반환: 신호표(기간 안 신호만)."""
    lo, hi = WIN[p]
    rows = []
    for s, (A, Z, sid) in ctx(p, tf).items():
        gap = 0.03 if s in LEV else 0.02
        rd = rd_of(s) if rd_of is not None else np.zeros(1)
        S_, P_, L_, F_, X_, E_, PX_, W_, ST_ = machine(
            A["o"], A["l"], A["r"], A["ema"], A["tk"], A["dead"], A["reg"], A["hot7"], sid, Z["ZLO"], Z["ZHI"],
            gap, spec.get("mode", 1), spec.get("k", 1), spec.get("lev", 0), spec.get("incl", False), spec.get("life", 2),
            reg_only, spec.get("from_sig", False), spec.get("fixd", 0.0), rd, SL.X, A["cyc"], spec.get("hyb", False),
            np.isin(A["ses"], list(spec["dfr_ses"])) if "dfr_ses" in spec else A["reg"])
        day = A["day"]
        for j in range(len(S_)):
            dy = day[S_[j]]
            if not (lo <= dy <= hi):
                continue
            rows.append((s, dy, S_[j], P_[j], L_[j], F_[j], X_[j], E_[j], PX_[j], W_[j], ST_[j],
                         A["ses"][S_[j]], A["ses"][F_[j]] if F_[j] >= 0 else "",
                         (A["tk"][F_[j]] - A["tk"][S_[j]]) / 6e10 if F_[j] >= 0 else np.nan,
                         A["tk"][S_[j]], A["tk"][F_[j]] if F_[j] >= 0 else -1, A["tk"][X_[j]] if X_[j] >= 0 else -1))
    t = pd.DataFrame(rows, columns=["sym", "day", "s", "P", "L", "f", "x", "ent", "px", "why", "stat", "ses", "fses", "wait_min",
                                    "ts", "tfill", "tx"])
    t["ret"] = np.where(t.stat == 0, t.px / t.ent - 1, np.nan)
    t["d"] = 1 - t.L / t.P
    return t


_DAYO = {}


def rand_once(p, k, lf, reg_only, pool, rng):
    """무작위 거리 한 번: 종목 묶음별 (체결 수익 합, 체결 수). DataFrame 없이 빠르게."""
    lo, hi = WIN[p][0].toordinal(), WIN[p][1].toordinal()
    acc = {un: [0.0, 0] for un in UNIV}
    for s, (A, Z, sid) in ctx(p).items():
        if (p, s) not in _DAYO:
            _DAYO[(p, s)] = np.array([d.toordinal() for d in A["day"]], np.int64)
        dayo = _DAYO[(p, s)]
        v = pool.get(s)
        rd = rng.choice(v, 4096) if v is not None and len(v) else np.zeros(4096)
        S_, P_, L_, F_, X_, E_, PX_, W_, ST_ = machine(
            A["o"], A["l"], A["r"], A["ema"], A["tk"], A["dead"], A["reg"], A["hot7"], sid, Z["ZLO"], Z["ZHI"],
            0.03 if s in LEV else 0.02, 3, k, 0, False, lf, reg_only, False, 0.0, rd, SL.X, A["cyc"], False, A["reg"])
        m = (ST_ == 0) & (dayo[S_] >= lo) & (dayo[S_] <= hi)
        r = PX_[m] / E_[m] - 1
        for un, us in UNIV.items():
            if s in us:
                acc[un][0] += r.sum()
                acc[un][1] += len(r)
    return acc


def metrics(t, b, syms):
    """t 변형 신호표, b 지금 지표 신호표(같은 기간·범위). 종목 묶음 syms."""
    t, b = t[t.sym.isin(syms)], b[b.sym.isin(syms)]
    tr, br = t[t.stat == 0], b[b.stat == 0]
    days = sorted(set(b.day) | set(t.day))
    dd = tr.groupby("day").ret.sum().reindex(days, fill_value=0) - br.groupby("day").ret.sum().reindex(days, fill_value=0)
    sd = dd.std(ddof=1)
    tt = dd.mean() / (sd / np.sqrt(len(dd))) if sd > 0 and len(dd) > 1 else 0.0
    comp, mdd = [], []
    for s in syms:
        x = tr[tr.sym == s].sort_values("f").ret.values
        if len(x) == 0:
            comp.append(0.0); mdd.append(0.0)
            continue
        eq = np.cumprod(1 + x)
        comp.append(eq[-1] - 1)
        mdd.append((eq / np.maximum.accumulate(np.r_[1.0, eq])[1:] - 1).min())
    nsig = int((t.stat != 3).sum())
    top = dd.sort_values(ascending=False)
    return dict(신호=nsig, 주문=int(t.stat.isin([0, 1]).sum()), 체결=len(tr), 체결률=len(tr) / nsig if nsig else np.nan,
                평균=tr.ret.mean(), 승률=(tr.ret > 0).mean(), 최악=tr.ret.min(), 하위5=tr.ret.quantile(.05),
                합=tr.ret.sum(), 차이=tr.ret.sum() - br.ret.sum(), t=tt, 나은날=int((dd > 1e-12).sum()), 나쁜날=int((dd < -1e-12).sum()),
                최대이득일=top.iloc[0] if len(top) else np.nan,
                계좌복리=np.mean(comp), 최악낙폭=min(mdd), 깊이중앙=t[t.stat.isin([0, 1])].d.median(),
                대기중앙분=tr.wait_min.median())


BASE = dict(mode=0)
MAIN = {
    "지금 지표(신호 때 삼)": BASE,
    "Z1 하단, 수명 L1(원래 매도까지)": dict(mode=1, k=1, life=1),
    "Z1 하단, 수명 L2(아침 8시까지)": dict(mode=1, k=1, life=2),
    "Z1 하단, 수명 L3(3번째 8시까지)": dict(mode=1, k=1, life=3),
    "Z2 하단, 수명 L1(원래 매도까지)": dict(mode=1, k=2, life=1),
    "Z2 하단, 수명 L2(아침 8시까지)": dict(mode=1, k=2, life=2),
    "Z2 하단, 수명 L3(3번째 8시까지)": dict(mode=1, k=2, life=3),
}
SENS = {
    "Z1 상단, L2": dict(mode=1, k=1, lev=1, life=2),
    "Z2 상단, L2": dict(mode=1, k=2, lev=1, life=2),
    "Z1 하단-0.6%, L2": dict(mode=1, k=1, lev=2, life=2),
    "Z1 하단+0.6%, L2": dict(mode=1, k=1, lev=3, life=2),
    "Z2 하단-0.6%, L2": dict(mode=1, k=2, lev=2, life=2),
    "Z2 하단+0.6%, L2": dict(mode=1, k=2, lev=3, life=2),
    "Z1 하단, L2, 신호 봉 체결 허용": dict(mode=1, k=1, life=2, from_sig=True),
    "Z2 하단, L2, 신호 봉 체결 허용": dict(mode=1, k=2, life=2, from_sig=True),
    "Z1' 하단(P가 든 구역도 셈), L2": dict(mode=1, k=1, life=2, incl=True),
    "Z2' 하단(P가 든 구역도 셈), L2": dict(mode=1, k=2, life=2, incl=True),
}
FIXED = {f"고정 -{d:g}%, L{lf}": dict(mode=2, fixd=d / 100, life=lf) for lf in (1, 2) for d in (1, 2, 3, 4, 5, 6, 8)}
TFS = {f"Z{k} 하단, L2, {tf}분봉 구역": (dict(mode=1, k=k, life=2), tf) for tf in (20, 80) for k in (1, 2)}


def fmt_table(df):
    f = df.copy()
    for k, fm in (("체결률", "{:.0%}"), ("평균", "{:+.2%}"), ("승률", "{:.0%}"), ("최악", "{:+.1%}"), ("하위5", "{:+.1%}"),
                  ("합", "{:+.2f}"), ("차이", "{:+.2f}"), ("t", "{:+.2f}"), ("최대이득일", "{:+.2f}"), ("계좌복리", "{:+.1%}"),
                  ("최악낙폭", "{:+.1%}"), ("깊이중앙", "{:.1%}"), ("대기중앙분", "{:.0f}")):
        if k in f:
            f[k] = f[k].map(lambda v: "" if pd.isna(v) else fm.format(v))
    return f.to_string(index=False)


def main():
    pd.set_option("display.width", 250)
    L = []
    w = L.append
    w("이격매수 '한 단계 아래 지지 구역에서 사기' 연구 (2026-10-10, defer_zone.py)")
    w("칸: 신호 = 지금 지표 매수 신호 수(구역 변형은 자기 상태 기계의 신호), 주문 = 구역이 있어 지정가를 건 수, 체결률 = 체결/신호")
    w("     평균·승률·최악·하위5 = 체결 거래, 합 = 거래 수익 합(거래마다 같은 금액), 차이 = 합 - 지금 지표 합, t = 날짜별 합 차이의 t")
    w("     계좌복리 = 종목마다 자기 몫으로 거래를 이어 산 복리의 평균, 최악낙폭 = 가장 나쁜 종목의 낙폭, 깊이중앙 = 주문가가 P보다 낮은 정도 중앙값")
    w("     대기중앙분 = 신호에서 체결까지 분(중앙값)")
    tables = {}
    allspecs = [(nm, sp, 40) for nm, sp in {**MAIN, **SENS, **FIXED}.items()] + [(nm, sp, tf) for nm, (sp, tf) in TFS.items()]
    for p in ("early", "late", "oos"):
        for reg_only in (False, True):
            b = run(p, BASE, reg_only)
            # 지금 지표 재현 점검: stop_lab.sim과 같아야 한다(레버리지 목록 종목이 없으면 완전히 같음)
            if p in ("early", "late"):
                ref = SL.base_trades(p, reg_only)
                ref = ref[(ref.day >= WIN[p][0]) & (ref.day <= WIN[p][1])]
                bb = b[b.stat == 0]
                assert len(ref) == len(bb) and abs(ref.ret.sum() - bb.ret.sum()) < 1e-9, (p, reg_only, len(ref), len(bb))
            for nm, sp, tf in allspecs:
                t = b if sp is BASE else run(p, sp, reg_only, tf)
                tables[(p, reg_only, nm)] = t
    # 표 1: 주요 변형
    rows = []
    for p in ("early", "late", "oos"):
        for reg_only in (False, True):
            b = tables[(p, reg_only, "지금 지표(신호 때 삼)")]
            for nm in list(MAIN) + list(SENS) + list(FIXED) + list(TFS):
                for un in UNIV:
                    rows.append(dict(기간=p, 범위="정규장만" if reg_only else "24시간", 종목=un, 규칙=nm,
                                     **metrics(tables[(p, reg_only, nm)], b, UNIV[un])))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(TMP, "defer_zone_metrics.csv"), index=False, encoding="utf-8-sig")
    cols = ["규칙", "신호", "주문", "체결", "체결률", "평균", "승률", "최악", "하위5", "합", "차이", "t", "나은날", "나쁜날",
            "최대이득일", "계좌복리", "최악낙폭", "깊이중앙", "대기중앙분"]
    for title, names in (("1) 사용자안: Z1·Z2 하단 지정가, 주문 수명 3가지", list(MAIN)),
                         ("2) 같은 신호에서 고정 깊이 지정가(대조군: 지지 구역 없이 그냥 더 깊게)", ["지금 지표(신호 때 삼)"] + list(FIXED)),
                         ("3) 민감도: 주문가(상단·±0.6%), 신호 봉 체결, P가 든 구역도 세기, 차트 봉 길이", ["지금 지표(신호 때 삼)"] + list(SENS) + list(TFS))):
        w("")
        w("=" * 20 + " " + title)
        for un in UNIV:
            for p in ("early", "late", "oos"):
                for rg in ("24시간", "정규장만"):
                    x = R[(R.기간 == p) & (R.범위 == rg) & (R.종목 == un) & (R.규칙.isin(names))][cols]
                    w(f"\n[{un} · {PN[p]} · {rg}]")
                    w(fmt_table(x))
    # 표 4: L1 짝 비교(같은 신호, 같은 매도) — 체결된 거래와 놓친 거래
    w("")
    w("=" * 20 + " 4) 수명 L1 짝 비교: 지금 지표 거래를 '구역까지 내려옴(체결)'과 '안 내려오고 팔림(놓침)'으로 나눔")
    for un in UNIV:
        for p in ("early", "late", "oos"):
            for reg_only in (False, True):
                b = tables[(p, reg_only, "지금 지표(신호 때 삼)")]
                b = b[(b.stat == 0) & b.sym.isin(UNIV[un])].set_index(["sym", "s"])
                line = []
                for k in (1, 2):
                    t = tables[(p, reg_only, f"Z{k} 하단, 수명 L1(원래 매도까지)")]
                    t = t[t.sym.isin(UNIV[un]) & (t.stat != 3)].set_index(["sym", "s"])
                    j = b[["ret"]].join(t[["stat", "ret"]].rename(columns={"ret": "dret"}), how="inner")
                    fill, miss = j[j.stat == 0], j[j.stat != 0]
                    line.append(f"Z{k}: 짝 {len(j)} | 체결 {len(fill)}건 지금 {fill.ret.mean():+.2%} → 구역 {fill.dret.mean():+.2%} "
                                f"| 놓침 {len(miss)}건 지금 평균 {miss.ret.mean():+.2%}·승률 {(miss.ret > 0).mean():.0%}·합 {miss.ret.sum():+.2f}")
                w(f"[{un} · {PN[p]} · {'정규장만' if reg_only else '24시간'}] " + " || ".join(line))
    # 표 5: 무작위 거리 대조(같은 신호, 같은 구역 유무, 거리만 같은 종목의 실제 거리에서 뽑음)
    w("")
    w("=" * 20 + f" 5) 무작위 거리 대조 {NSEED}번: 실제 구역 하단이 '같은 종목의 같은 거리 분포에서 뽑은 아무 선'보다 나은가")
    w(f"   백분위 = 무작위 {NSEED}번 중 실제보다 낮은 비율(95% 이상이면 구역이 특별, 50% 근처면 거리 효과뿐)")
    rand_rows = []
    for p in ("early", "late", "oos"):
        for reg_only in (False, True):
            for k in (1, 2):
                for lf in (1, 2):
                    nm = f"Z{k} 하단, 수명 L{lf}(" + ("원래 매도까지)" if lf == 1 else "아침 8시까지)")
                    act = tables[(p, reg_only, nm)]
                    # 같은 종목·같은 기간에 실제로 건 주문의 거리 모음
                    pool = act[act.stat.isin([0, 1])].groupby("sym").d.apply(lambda v: v.values).to_dict()
                    sims = {un: [] for un in UNIV}
                    for seed in range(NSEED):
                        rng = np.random.default_rng(10_000 * seed + 100 * k + lf)
                        acc = rand_once(p, k, lf, reg_only, pool, rng)
                        for un in UNIV:
                            sm, n = acc[un]
                            sims[un].append((sm, sm / n if n else np.nan, n))
                    for un, us in UNIV.items():
                        a = act[(act.stat == 0) & act.sym.isin(us)]
                        arr = np.array(sims[un])
                        rand_rows.append(dict(기간=p, 범위="정규장만" if reg_only else "24시간", 종목=un, 규칙=f"Z{k} L{lf}",
                                              실제체결=len(a), 무작위체결=np.mean(arr[:, 2]), 실제합=a.ret.sum(), 무작위합=np.mean(arr[:, 0]),
                                              합백분위=(arr[:, 0] < a.ret.sum()).mean(), 실제평균=a.ret.mean(), 무작위평균=np.nanmean(arr[:, 1]),
                                              평균백분위=(arr[:, 1] < a.ret.mean()).mean()))
    RR = pd.DataFrame(rand_rows)
    RR.to_csv(os.path.join(TMP, "defer_zone_random.csv"), index=False, encoding="utf-8-sig")
    f = RR.copy()
    for k, fm in (("무작위체결", "{:.0f}"), ("실제합", "{:+.2f}"), ("무작위합", "{:+.2f}"), ("합백분위", "{:.0%}"),
                  ("실제평균", "{:+.2%}"), ("무작위평균", "{:+.2%}"), ("평균백분위", "{:.0%}")):
        f[k] = f[k].map(lambda v: "" if pd.isna(v) else fm.format(v))
    w(f.to_string(index=False))
    # 저장: 모든 변형 신호표(검증용, ts·tfill·tx = 신호·체결·매도 봉 시작 시각 ns UTC)
    keep = []
    for (p, reg_only, nm), t in tables.items():
        if True:
            x = t.copy()
            x["기간"], x["범위"], x["규칙"] = p, "정규장만" if reg_only else "24시간", nm
            keep.append(x)
    pd.concat(keep).to_parquet(os.path.join(TMP, "defer_zone_trades.parquet"))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L[:5]))
    print("저장:", OUT)


if __name__ == "__main__":
    main()
