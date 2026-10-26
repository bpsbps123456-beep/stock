"""지지 피로(반대 가설) 연구(2026-10-09). 지지 구역을 오래 버틸수록 위로 돌아서나, 아니면 결국 깨지나.

구역(사용자 트레이딩뷰 지표 = sr_zones_ref.py와 같은 계산):
  차트 봉의 스윙 저점·고점(좌우 5봉 피벗, 왼쪽은 엄격히 큼/작음) → 최근 스윙 60개 → 가격순으로 0.5% 안은 한 구역(lo~hi, 닿은 횟수 cnt)
  → 닿은 횟수 많은 순(같으면 최근)으로 남기며 이미 남긴 구역과 가운데값이 1.2%보다 가까우면 뺀다.
  피벗 j는 j+5 봉이 끝난 뒤(40분봉: 시작+40분, 일봉: 다음 날 시가)부터만 쓴다(미래 참조 없음).
  40분봉 = 5분봉(프리·애프터 포함)을 미 동부 04:00 기준 40분으로 묶음. 일봉 = 정규장 일봉 1년.

에피소드(한 종목에 동시에 하나):
  닿음  봉 p에서 직전 봉 종가 c[p-1]보다 아래인 구역 중 가장 위(hi < c[p-1])의 hi를 저가가 건드림(l[p] <= hi). 구역은 그때 값으로 고정.
  깨짐  이후 저가 <= lo × 0.98(구역 하단 -2%)  /  돌아섬  고가 >= hi × 1.03(구역 상단 +3%). 같은 봉에 둘 다면 깨짐(보수적).
        닿는 봉에서는 깨짐만 본다(그 봉 고가는 닿기 전일 수 있음).
  버틴 시간 T = 닿은 봉 시작부터 지금 봉 끝까지 거래 시계(24시간 범위: 04~20시 ET 960분/일, 정규장 범위: 390분/일, 일봉: 일).
  다시 두드림 k  닿은 뒤 종가가 hi×1.01 위로 한 번 올라갔다가 다시 저가 <= hi면 k+1.
  이전 시험  최근 5거래일 안 같은 종목에서 hi가 0.5% 안인 구역에 닿았던 에피소드 수.
  미결  3거래일(일봉 20일) 안에 결판 안 남 → 그만 봄.

범위: 24시간 = 모든 봉으로 추적·진입, 정규장만 = 정규장 봉만으로 추적·진입(나가기는 24시간 봉, stop_lab과 같음).
기간: 분봉 early 7/14~9/8(5분봉)로 고르고 late 9/10~10/6(3분봉)으로 확인. 일봉 앞 절반(닿은 날 < 2026-04-07)으로 고르고 뒤 절반으로 확인.

진입 규칙(시험 88개):
  버팀 T(Tm, band)  닿은 뒤 처음으로 '살아 있고(결판 안 남) 버틴 시간 >= Tm, 종가 <= hi×(1+band)'인 봉이 끝나면 다음 봉 시가에 산다.
                   Tm 분봉 30·60·120·240·480·960분, 일봉 1·2·3·5일. band 0.01(구역 위 1% 안) 또는 없음(살아 있기만).
  k번째 두드림   k=2·3번째로 다시 닿는 봉에서 hi 지정가(체결 min(시가, hi)). 그 봉에서 깨져도 체결로 센다.
  매도  RSI/8시(stop_lab과 같음: 기준 RSI>70 다음 봉 시가 또는 매수 뒤 첫 한국 아침 8시 뒤 첫 봉 시가), 고정 1시간·4시간(시계),
        1일·3일(다음·3번째 거래일 같은 시각 첫 봉 시가). 일봉은 1일·3일(1·3일 뒤 시가).
비교 기준: (a) 첫 닿음 = 모든 에피소드의 닿는 봉 hi 지정가, 같은 매도  (b) 무작위 = 같은 종목·같은 범위·같은 ET 시각(1시간 칸)
  모든 봉 시가 진입 평균(일봉은 같은 종목 모든 날)  (c) 지금 이격매수 base_trades(RSI/8시).
고르는 법: early·정규장만·사용자78에서 건수 >= 30인 규칙 중 '무작위 대비 초과 평균' 최대. late는 확인만.
사용: py support/fatigue.py   (code 폴더에서) → ../results/support/fatigue.txt
"""
import bisect, io, os, sys, time
from datetime import date
import numpy as np, pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as sw

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, CODE)
from stop_lab import load, base_trades, RAW, PERIODS, USER78, ALL

OUT = os.path.normpath(os.path.join(CODE, "../results/support/fatigue.txt"))
NP, KEEP, TOL, ZGAP = 5, 60, 0.005, 0.012
DN, UP, BOUNCE, KMAX = 0.02, 0.03, 0.01, 6
NS_MIN = 60 * 10**9
DAYLEN = {"24시간": 960, "정규장만": 390}
BINS = {"24시간": [0, 15, 30, 60, 120, 240, 480, 960, 1920], "정규장만": [0, 15, 30, 60, 120, 240, 390, 780]}
TMS = [30, 60, 120, 240, 480, 960]
BANDS = [0.01, None]
SPECS = [(tm, b) for tm in TMS for b in BANDS]
KSPECS = [2, 3]
EXITS = ["RSI/8시", "1시간", "4시간", "1일", "3일"]
D_TMS = [1, 2, 3, 5]
D_SPECS = [(tm, b) for tm in D_TMS for b in BANDS]
D_BINS = [1, 2, 3, 5, 10]
D_EXITS = ["1일", "3일"]
HALF = date(2026, 4, 7)


def zones_from(px, bi):
    order = np.argsort(px, kind="stable")
    Z = []
    for k in order:
        p, q = px[k], bi[k]
        if Z and (p - Z[-1][0]) / Z[-1][0] <= TOL:
            Z[-1][1] = p; Z[-1][2] += 1; Z[-1][3] = max(Z[-1][3], q)
        else:
            Z.append([p, p, 1, q])
    keep = []
    for z in sorted(Z, key=lambda z: z[2] * 1e7 + z[3], reverse=True):
        mid = (z[0] + z[1]) / 2
        if all(abs(mid - (k[0] + k[1]) / 2) / ((k[0] + k[1]) / 2) >= ZGAP for k in keep):
            keep.append(z)
    keep.sort(key=lambda z: (z[0] + z[1]) / 2)
    return [z[0] for z in keep], [z[1] for z in keep], [z[2] for z in keep]


def zone_sets(h, l, conf_of):
    """확정 시각(conf_of(j))이 오름차순인 구역 묶음 목록. conf[m] 이후 봉은 sets[m]을 본다."""
    n = len(h)
    if n < 2 * NP + 1:
        return np.array([], dtype=np.int64), []
    Lw, Hw = sw(l, 2 * NP + 1), sw(h, 2 * NP + 1)
    lc, hc = l[NP:n - NP], h[NP:n - NP]
    plo = (lc <= Lw.min(1)) & (Lw[:, :NP] > lc[:, None]).all(1)
    phi = (hc >= Hw.max(1)) & (Hw[:, :NP] < hc[:, None]).all(1)
    px, bi, conf, sets = [], [], [], []
    for jj in np.flatnonzero(plo | phi):
        j = jj + NP
        if plo[jj]:
            px.append(l[j]); bi.append(j)
        if phi[jj]:
            px.append(h[j]); bi.append(j)
        conf.append(conf_of(j))
        sets.append(zones_from(np.array(px[-KEEP:]), np.array(bi[-KEEP:])))
    return np.array(conf, dtype=np.int64), sets


def track(o, h, l, c, clock, mins, ztime, conf, sets, start_ok, horizon, bins, specs, dayclock, atr=None, ad=1.0, au=1.5):
    """에피소드 추적. 반환 eps[[eid,p0,lo,hi,cnt,prior,out,res_el,kmax,amb,imm]], snaps[(eid,bin,el,rel,k)],
    ksn[(eid,k,p,rel)], ent[(spec,eid,p)]. pr = 결판(또는 그만 본) 봉. out: 1 돌아섬, -1 깨짐, 0 미결(시간 끝), 2 자료 끝."""
    n = len(o)
    si = np.searchsorted(conf, ztime, side="right") - 1 if len(conf) else np.full(n, -1)
    eps, snaps, ksn, ent, hist = [], [], [], [], []
    nb, ns = len(bins), len(specs)
    act = False
    eid = p0 = c0 = k = nxt = 0
    zlo = zhi = zc = dlv = ulv = 0.0
    prior = 0
    bounced = False
    h0 = 0
    fired = None
    for p in range(1, n):
        if not act:
            if not start_ok[p]:
                continue
            s = si[p]
            if s < 0:
                continue
            los, his, cnts = sets[s]
            q = bisect.bisect_left(his, c[p - 1]) - 1
            if q < 0 or l[p] > his[q]:
                continue
            zlo, zhi, zc = los[q], his[q], cnts[q]
            if atr is None:
                dlv, ulv = zlo * (1 - DN), zhi * (1 + UP)
            else:
                a = atr[p - 1]
                if not a > 0:
                    continue
                dlv, ulv = zlo - ad * a, zhi + au * a
            while h0 < len(hist) and clock[p] - hist[h0][0] > 5 * dayclock:
                h0 += 1
            prior = sum(1 for (cc, hh) in hist[h0:] if abs(hh / zhi - 1) <= TOL)
            hist.append((clock[p], zhi))
            eid, p0, c0, k = len(eps), p, clock[p], 1
            el = clock[p] + mins - c0
            rel = c[p] / zhi - 1
            ksn.append((eid, 1, p, rel))
            if l[p] <= dlv:
                eps.append([eid, p, zlo, zhi, zc, prior, -1, el, 1, 0, 1, p, dlv / zhi, ulv / zhi])
                continue
            bounced = c[p] > zhi * (1 + BOUNCE)
            act, nxt, fired = True, 0, [False] * ns
            while nxt < nb and bins[nxt] <= el:
                snaps.append((eid, nxt, el, rel, k)); nxt += 1
            for z, (tm, band) in enumerate(specs):
                if el >= tm and (band is None or rel <= band):
                    ent.append((z, eid, p)); fired[z] = True
            continue
        if clock[p] - c0 >= horizon:
            eps.append([eid, p0, zlo, zhi, zc, prior, 0, clock[p] - c0, k, 0, 0, p, dlv / zhi, ulv / zhi]); act = False
            continue
        el = clock[p] + mins - c0
        if bounced and l[p] <= zhi:
            k += 1; bounced = False
            if k <= KMAX:
                ksn.append((eid, k, p, c[p] / zhi - 1))
        dn, up = l[p] <= dlv, h[p] >= ulv
        if dn or up:
            eps.append([eid, p0, zlo, zhi, zc, prior, -1 if dn else 1, el, k, int(dn and up), 0, p, dlv / zhi, ulv / zhi]); act = False
            continue
        if c[p] > zhi * (1 + BOUNCE):
            bounced = True
        rel = c[p] / zhi - 1
        while nxt < nb and bins[nxt] <= el:
            snaps.append((eid, nxt, el, rel, k)); nxt += 1
        for z in range(ns):
            if not fired[z]:
                tm, band = specs[z]
                if el >= tm and (band is None or rel <= band):
                    ent.append((z, eid, p)); fired[z] = True
    if act:
        eps.append([eid, p0, zlo, zhi, zc, prior, 2, clock[n - 1] + mins - c0, k, 0, 0, n - 1, dlv / zhi, ulv / zhi])
    return eps, snaps, ksn, ent


def rw(x, dl, ul):
    """무작위 걸음(표류 0)이면 x에서 위(ul)에 먼저 닿을 확률 = (x-dl)/(ul-dl). 모두 구역 상단 대비 비율."""
    return float(min(max((x - dl) / (ul - dl), 0.0), 1.0))


def rand_eps(o, h, l, c, clock, mins, start_ok, horizon, bins, step, atr=None, ad=1.0, au=1.5):
    """구역과 무관한 가짜 에피소드(비교 기준): 봉 i 시가 x를 폭 0 구역으로 보고 같은 규칙으로 결판(x×0.98 깨짐, x×1.03 돌아섬,
    첫 봉은 깨짐만). 반환 [(bin, rel, out, rw)], bin=-1은 시작 순간."""
    n = len(o)
    S = np.flatnonzero(start_ok)[::step]
    S = S[S >= 1]
    if atr is not None:
        S = S[atr[S - 1] > 0]
    if not len(S):
        return []
    end = np.searchsorted(clock, clock[S] + horizon, side="left")
    W = int((end - S).max())
    I = S[:, None] + np.arange(W)[None, :]
    valid = I < end[:, None]
    Ic = np.minimum(I, n - 1)
    x = o[S]
    if atr is None:
        dl, ul = np.full(len(S), 1 - DN), np.full(len(S), 1 + UP)
    else:
        dl, ul = 1 - ad * atr[S - 1] / x, 1 + au * atr[S - 1] / x
    L = (l[Ic] <= (x * dl)[:, None]) & valid
    H = (h[Ic] >= (x * ul)[:, None]) & valid
    H[:, 0] = False
    fd = np.where(L.any(1), L.argmax(1), W)
    fu = np.where(H.any(1), H.argmax(1), W)
    out = np.where((fd <= fu) & (fd < W), -1, np.where(fu < W, 1, 0))
    res = np.minimum(fd, fu)
    el = np.where(valid, clock[Ic] + mins - clock[S][:, None], -1)
    recs = [(-1, 0.0, int(out[r]), rw(1.0, dl[r], ul[r])) for r in range(len(S))]
    for b, T in enumerate(bins):
        hit = el >= T
        q = np.where(hit.any(1), hit.argmax(1), -1)
        ok = (q >= 0) & (res > q)
        for r in np.flatnonzero(ok):
            rel = c[Ic[r, q[r]]] / x[r] - 1
            recs.append((b, rel, int(out[r]), rw(1 + rel, dl[r], ul[r])))
    return recs


# ---------------------------------------------------------------- 분봉(40분봉 구역)
def exit_index(A):
    """모든 봉 e에 대해 '봉 e에 샀을 때' 파는 봉 번호(없으면 n)."""
    n = len(A["o"])
    e = np.arange(n)
    pos = np.flatnonzero(A["r"] > 70)
    q = np.searchsorted(pos, e)
    j_rsi = np.where(q < len(pos), pos[np.minimum(q, len(pos) - 1)] + 1, n)
    j_dead = np.searchsorted(A["tk"], A["dead"], side="left")
    days = np.array(sorted(set(A["day"])))
    di = np.searchsorted(days, A["day"])
    key = di * 1440 + A["mod"]
    J = {"RSI/8시": np.minimum(j_rsi, j_dead),
         "1시간": np.searchsorted(A["tk"], A["tk"] + 60 * NS_MIN), "4시간": np.searchsorted(A["tk"], A["tk"] + 240 * NS_MIN),
         "1일": np.searchsorted(key, (di + 1) * 1440 + A["mod"]), "3일": np.searchsorted(key, (di + 3) * 1440 + A["mod"])}
    return J, di


def intraday(period):
    S = load(period)
    lo_d, hi_d = PERIODS[period][3], PERIODS[period][4]
    EP, SN, KS, TR, RN = [], [], [], [], []
    for s, A in S.items():
        m5 = pd.read_parquet(os.path.join(RAW, f"5m_{s}.parquet"))
        b = m5.resample("40min", origin="start_day", offset="4h", label="left", closed="left").agg(
            {"High": "max", "Low": "min", "Close": "last"}).dropna(subset=["Close"])
        b = b[b.High > 0]
        t40 = b.index.as_unit("ns").asi8
        conf, sets = zone_sets(b.High.values.astype(float), b.Low.values.astype(float), lambda j: t40[j + NP] + 40 * NS_MIN)
        J, di = exit_index(A)
        n = len(A["o"])
        o = A["o"]
        inper = np.array([lo_d <= x <= hi_d for x in A["day"]])
        hour = A["mod"] // 60
        for scope in ("24시간", "정규장만"):
            idx = np.arange(n) if scope == "24시간" else np.flatnonzero(A["reg"])
            if len(idx) < 50:
                continue
            base = 240 if scope == "24시간" else 570
            clock = di[idx] * DAYLEN[scope] + (A["mod"][idx] - base)
            eps, snaps, ksn, ent = track(o[idx], A["h"][idx], A["l"][idx], A["c"][idx], clock, A["mins"], A["tk"][idx], conf, sets,
                                         inper[idx], 3 * DAYLEN[scope], BINS[scope], SPECS, DAYLEN[scope])
            for (bn, rel, ot, rwv) in rand_eps(o[idx], A["h"][idx], A["l"][idx], A["c"][idx], clock, A["mins"], inper[idx],
                                               3 * DAYLEN[scope], BINS[scope], 5):
                RN.append((period, scope, s, bn, rel, ot, rwv))
            # 무작위 기준: 같은 종목·범위·기간·ET 시각(1시간) 모든 봉 시가 진입
            pool = np.zeros(n, bool); pool[idx] = True; pool &= inper
            rnd = {}
            for x in EXITS:
                ok = pool & (J[x] < n)
                rr = np.full(n, np.nan); rr[ok] = o[np.minimum(J[x], n - 1)][ok] / o[ok] - 1
                rnd[x] = pd.Series(rr[pool]).groupby(hour[pool]).mean()
            epd = {r[0]: r for r in eps}

            def add(rule, e, fill, eid):
                if not inper[e]:                                               # 매수일이 기간 밖이면 뺀다(고르기·확인 기간 섞임 방지)
                    return
                for x in EXITS:
                    j = J[x][e]
                    if j >= n:
                        continue
                    TR.append((period, scope, s, rule, x, A["day"][e], e, fill, o[j] / fill - 1,
                               rnd[x].get(hour[e], np.nan), eid, A["ses"][e]))
            for r in eps:                                                      # (a) 첫 닿음: hi 지정가
                e = idx[r[1]]
                add("첫닿음", e, min(o[e], r[3]), r[0])
                if A["l"][e] <= r[3] * 0.999:                                  # 0.1% 넘게 뚫고 내려온 봉만 체결로(보수적)
                    add("첫닿음관통", e, min(o[e], r[3]), r[0])
            for (eid, kk, p, rel) in ksn:
                if kk in KSPECS:
                    e = idx[p]
                    add(f"{kk}번째두드림", e, min(o[e], epd[eid][3]), eid)
            for (z, eid, p) in ent:
                if p + 1 >= len(idx):
                    continue
                e = idx[p + 1]
                tm, band = SPECS[z]
                add(f"버팀{tm}분·{'1%안' if band else '생존'}", e, o[e], eid)
            for r in eps:
                EP.append((period, scope, s, A["day"][idx[r[1]]], *r[1:]))
            for (eid, bn, el, rel, kk) in snaps:
                r = epd[eid]
                SN.append((period, scope, s, eid, bn, el, rel, kk, r[6], r[7], rw(1 + rel, r[12], r[13])))
            for (eid, kk, p, rel) in ksn:
                r = epd[eid]
                KS.append((period, scope, s, eid, kk, rel, r[6], r[7], int(r[6] == -1 and r[11] == p), rw(1.0, r[12], r[13])))
    EP = pd.DataFrame(EP, columns=["period", "scope", "sym", "day", "p0", "lo", "hi", "cnt", "prior", "out", "res_el", "kmax", "amb", "imm", "pr", "dl", "ul"])
    SN = pd.DataFrame(SN, columns=["period", "scope", "sym", "eid", "bin", "el", "rel", "k", "out", "res_el", "rw"])
    KS = pd.DataFrame(KS, columns=["period", "scope", "sym", "eid", "k", "rel", "out", "res_el", "imm", "rw"])
    TR = pd.DataFrame(TR, columns=["period", "scope", "sym", "rule", "exit", "day", "e", "fill", "ret", "rnd", "eid", "ses"])
    RN = pd.DataFrame(RN, columns=["period", "scope", "sym", "bin", "rel", "out", "rw"])
    return EP, SN, KS, TR, RN


# ---------------------------------------------------------------- 일봉 구역
def daily(atr_mode=False):
    EP, SN, KS, TR, RN = [], [], [], [], []
    for s in ALL:
        f = os.path.join(RAW, f"1d_{s}.parquet")
        if not os.path.exists(f):
            continue
        d = pd.read_parquet(f).dropna(subset=["Open", "High", "Low", "Close"])
        d = d[d.High > 0]
        if len(d) < 60:
            continue
        o, h, l, c = (d[k].values.astype(float) for k in ("Open", "High", "Low", "Close"))
        days = np.array([x.date() for x in pd.to_datetime(d.index)])
        n = len(o)
        conf, sets = zone_sets(h, l, lambda j: j + NP + 1)
        pc = np.r_[np.nan, c[:-1]]
        tr = np.nanmax(np.c_[h - l, np.abs(h - pc), np.abs(l - pc)], axis=1)
        atr = pd.Series(tr).rolling(14).mean().values
        ar = np.arange(n)
        eps, snaps, ksn, ent = track(o, h, l, c, ar, 1, ar, conf, sets, np.ones(n, bool), 20, D_BINS, D_SPECS, 1,
                                     atr=atr if atr_mode else None)
        epd = {r[0]: r for r in eps}
        half = np.where(days < HALF, "early", "late")
        for hf in ("early", "late"):
            for (bn, rel, ot, rwv) in rand_eps(o, h, l, c, ar, 1, half == hf, 20, D_BINS, 1, atr=atr if atr_mode else None):
                RN.append((hf, "일봉", s, bn, rel, ot, rwv))
        rnd = {}
        for x, H in (("1일", 1), ("3일", 3)):
            rr = np.full(n, np.nan); rr[:n - H] = o[H:] / o[:n - H] - 1
            rnd[x] = {hf: np.nanmean(rr[half == hf]) for hf in ("early", "late")}

        def add(rule, e, fill, eid):
            for x, H in (("1일", 1), ("3일", 3)):
                if e + H < n:
                    TR.append((half[e], "일봉", s, rule, x, days[e], e, fill, o[e + H] / fill - 1, rnd[x][half[e]], eid, "정규"))
        for r in eps:
            add("첫닿음", r[1], min(o[r[1]], r[3]), r[0])
        for (eid, kk, p, rel) in ksn:
            if kk == 2:
                add("2번째두드림", p, min(o[p], epd[eid][3]), eid)
        for (z, eid, p) in ent:
            if p + 1 < n:
                tm, band = D_SPECS[z]
                add(f"버팀{tm}일·{'1%안' if band else '생존'}", p + 1, o[p + 1], eid)
        for r in eps:
            EP.append((half[r[1]], "일봉", s, days[r[1]], *r[1:]))
        for (eid, bn, el, rel, kk) in snaps:
            r = epd[eid]
            SN.append((half[r[1]], "일봉", s, eid, bn, el, rel, kk, r[6], r[7], rw(1 + rel, r[12], r[13])))
        for (eid, kk, p, rel) in ksn:
            r = epd[eid]
            KS.append((half[r[1]], "일봉", s, eid, kk, rel, r[6], r[7], int(r[6] == -1 and r[11] == p), rw(1.0, r[12], r[13])))
    EP = pd.DataFrame(EP, columns=["period", "scope", "sym", "day", "p0", "lo", "hi", "cnt", "prior", "out", "res_el", "kmax", "amb", "imm", "pr", "dl", "ul"])
    SN = pd.DataFrame(SN, columns=["period", "scope", "sym", "eid", "bin", "el", "rel", "k", "out", "res_el", "rw"])
    KS = pd.DataFrame(KS, columns=["period", "scope", "sym", "eid", "k", "rel", "out", "res_el", "imm", "rw"])
    TR = pd.DataFrame(TR, columns=["period", "scope", "sym", "rule", "exit", "day", "e", "fill", "ret", "rnd", "eid", "ses"])
    RN = pd.DataFrame(RN, columns=["period", "scope", "sym", "bin", "rel", "out", "rw"])
    return EP, SN, KS, TR, RN


# ---------------------------------------------------------------- 표
def pct(x):
    return "" if pd.isna(x) else f"{x:.0%}"


def upshare(df):
    r = df[df.out.isin([1, -1])]
    return (r.out == 1).mean() if len(r) else np.nan, len(r)


def surv_table(SN, EP, RN, bins, unit, out):
    """버틴 시간 T까지 살아남은 에피소드의 다음 구간·최종 결판 + 같은 위치의 무작위 걸음 기대(RW)·구역 무관 가짜 에피소드(무작위)."""
    rows = []
    res = EP[EP.out.isin([1, -1])]
    rw0 = np.mean([rw(1.0, a, b) for a, b in zip(res.dl, res.ul)]) if len(res) else np.nan
    u0, nu0 = upshare(EP[EP.sym.isin(USER78)])
    rows.append({"T": "닿는 순간", "n": len(EP), "다음위": "", "다음아래": pct((EP.imm == 1).mean()), "최종위": pct((EP.out == 1).mean()),
                 "최종아래": pct((EP.out == -1).mean()), "미결": pct(EP.out.isin([0, 2]).mean()), "위비중": pct(upshare(EP)[0]),
                 "RW": pct(rw0), "무작위": pct(upshare(RN[RN.bin == -1])[0]), "위비중78": f"{u0:.0%}({nu0})",
                 "위치<0 구역/무작위": "", "0~1%": "", ">1%": ""})
    for b in range(len(bins)):
        x = SN[SN.bin == b]
        if not len(x):
            continue
        t2 = bins[b + 1] if b + 1 < len(bins) else np.inf
        rs = x[x.out.isin([1, -1])]
        y = RN[RN.bin == b]
        nx = (x.res_el <= t2)
        loc = {}
        for nm, f in (("위치<0 구역/무작위", lambda v: v < 0), ("0~1%", lambda v: (v >= 0) & (v <= 0.01)), (">1%", lambda v: v > 0.01)):
            a, na = upshare(x[f(x.rel)])
            r_, nr = upshare(y[f(y.rel)])
            loc[nm] = f"{pct(a)}/{pct(r_)}({na})" if na else ""
        u, nu = upshare(x[x.sym.isin(USER78)])
        rows.append({"T": f"{bins[b]}~{'' if t2 == np.inf else t2}{unit}", "n": len(x),
                     "다음위": pct(((x.out == 1) & nx).mean()), "다음아래": pct(((x.out == -1) & nx).mean()),
                     "최종위": pct((x.out == 1).mean()), "최종아래": pct((x.out == -1).mean()), "미결": pct(x.out.isin([0, 2]).mean()),
                     "위비중": pct(upshare(x)[0]), "RW": pct(rs.rw.mean()), "무작위": pct(upshare(y)[0]),
                     "위비중78": f"{u:.0%}({nu})" if nu else "", **loc})
    out.write(pd.DataFrame(rows).to_string(index=False) + "\n")
    return rows


def k_table(KS, out):
    rows = []
    for kk in range(1, KMAX + 1):
        x = KS[KS.k == kk] if kk < KMAX else KS[KS.k >= kk]
        if not len(x):
            continue
        rs = x[x.out.isin([1, -1])]
        u, nu = upshare(x[x.sym.isin(USER78)])
        rows.append({"두드림": f"{kk}{'+' if kk == KMAX else ''}번째", "n": len(x), "그봉에서깨짐": pct(x.imm.mean()),
                     "최종위": pct((x.out == 1).mean()), "최종아래": pct((x.out == -1).mean()), "미결": pct(x.out.isin([0, 2]).mean()),
                     "위비중": pct(upshare(x)[0]), "RW(닿는 순간)": pct(rs.rw.mean()), "위비중78": f"{u:.0%}({nu})" if nu else ""})
    out.write(pd.DataFrame(rows).to_string(index=False) + "\n")


def attr_table(EP, col, cuts, out):
    rows = []
    for nm, m in cuts:
        x = EP[m(EP[col])]
        if not len(x):
            continue
        rs = x[x.out.isin([1, -1])]
        rows.append(dict(구분=nm, n=len(x), 닿자마자깨짐=pct((x.imm == 1).mean()), 최종위=pct((x.out == 1).mean()),
                         최종아래=pct((x.out == -1).mean()), 위비중=pct((rs.out == 1).mean())))
    out.write(f"[{col}] " + pd.DataFrame(rows).to_string(index=False).replace("\n", "\n  ") + "\n")


def tstat(v, days):
    s = pd.Series(v).groupby(np.asarray(days)).mean()
    if len(s) < 3 or s.std(ddof=1) == 0:
        return np.nan
    return s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))


def cell(t):
    if not len(t):
        return dict(n=0)
    ex = t.ret - t.rnd
    return dict(n=len(t), avg=t.ret.mean(), win=(t.ret > 0).mean(), worst=t.ret.min(), sum=t.ret.sum(),
                t=tstat(t.ret.values, t.day.values), rnd=t.rnd.mean(), ex=ex.mean(), tx=tstat(ex.values, t.day.values))


def fmt(c):
    if c.get("n", 0) == 0:
        return "n=0"
    return (f"n={c['n']:4d} 평균 {c['avg']:+.2%} 승률 {c['win']:.0%} 최악 {c['worst']:+.1%} 합 {c['sum']:+.2f} t {c['t']:+.2f} | "
            f"무작위 {c['rnd']:+.2%} 초과 {c['ex']:+.2%}(t {c['tx']:+.2f})")


def grid(TR, scopes, exits, periods, sel_scope, sel_univ, out, label):
    """규칙 × 매도 격자. 반환 (규칙표 DataFrame)."""
    rows = []
    rules = [r for r in TR.rule.unique() if not r.startswith("첫닿음")]
    for scope in scopes:
        for un, us in (("사용자78", USER78), ("전체", set(ALL))):
            for x in exits:
                for per in periods:
                    T = TR[(TR.scope == scope) & (TR.exit == x) & (TR.period == per) & TR.sym.isin(us)]
                    ft = cell(T[T.rule == "첫닿음"])
                    for r in rules:
                        c = cell(T[T.rule == r])
                        rows.append(dict(scope=scope, univ=un, exit=x, period=per, rule=r, first=ft.get("avg", np.nan), **c))
    G = pd.DataFrame(rows)
    return G


def summary_line(SN, EP, RN, bins, unit):
    """T 구간별 위비중 한 줄: 전체, 구역 안·아래(<0), 같은 위치 무작위."""
    a = [f"닿음 {pct(upshare(EP)[0])}"] + [f"{bins[b]}{unit} {pct(upshare(SN[SN.bin == b])[0])}" for b in range(len(bins)) if (SN.bin == b).any()]
    z = [f"{bins[b]}{unit} {pct(upshare(SN[(SN.bin == b) & (SN.rel < 0)])[0])}/{pct(upshare(RN[(RN.bin == b) & (RN.rel < 0)])[0])}"
         for b in range(len(bins)) if (SN.bin == b).any()]
    return "위비중 " + " → ".join(a) + "\n      구역 안·아래(종가<상단) 구역/무작위: " + " → ".join(z)


def main():
    t0 = time.time()
    out = io.StringIO()
    W = out.write
    W(__doc__.split("사용:")[0] + "\n")
    W("표 읽는 법: 위비중 = 돌아섬/(돌아섬+깨짐). RW = 그 시점 위치에서 표류 0 무작위 걸음이면 위 장벽에 먼저 닿을 확률 (x-하단장벽)/(상단장벽-하단장벽).\n"
      "  무작위 = 구역과 무관하게 같은 종목·범위·기간의 봉(분봉 5봉마다, 일봉 매일) 시가를 폭 0 구역으로 보고 같은 장벽으로 결판낸 가짜 에피소드.\n"
      "  위치 칸 = T 시점 종가/구역 상단-1 구간별 위비중 '구역/무작위(구역 건수)'.\n")
    # ---------- 분봉
    parts = [intraday(p) for p in ("early", "late")]
    EP, SN, KS, TR, RN = (pd.concat([p[i] for p in parts]) for i in range(5))
    print("분봉 계산", round(time.time() - t0), "초", flush=True)
    summ = []
    for scope in ("24시간", "정규장만"):
        for per in ("early", "late"):
            m = (EP.scope == scope) & (EP.period == per)
            sn, rn = SN[(SN.scope == scope) & (SN.period == per)], RN[(RN.scope == scope) & (RN.period == per)]
            W(f"\n=== 40분봉 구역 생존표 · {scope} · {per}({PERIODS[per][3]}~{PERIODS[per][4]}) · 전체 종목 에피소드 {m.sum()}개 "
              f"(같은 봉 위·아래 동시 {EP[m].amb.sum()}개는 깨짐으로), 무작위 가짜 {(rn.bin == -1).sum()}개 ===\n")
            surv_table(sn, EP[m], rn, BINS[scope], "분", out)
            summ.append(f"  40분봉 {scope} {per}: " + summary_line(sn, EP[m], rn, BINS[scope], "분"))
            W(f"\n  -- 다시 두드린 횟수(k번째 닿음 뒤 결판) · {scope} · {per}\n")
            k_table(KS[(KS.scope == scope) & (KS.period == per)], out)
            W(f"\n  -- 구역 세기(지표의 닿은 횟수 cnt)·최근 5거래일 이전 시험 횟수 · {scope} · {per}\n")
            attr_table(EP[m], "cnt", [("1", lambda v: v == 1), ("2", lambda v: v == 2), ("3", lambda v: v == 3), ("4+", lambda v: v >= 4)], out)
            attr_table(EP[m], "prior", [("0", lambda v: v == 0), ("1", lambda v: v == 1), ("2", lambda v: v == 2), ("3+", lambda v: v >= 3)], out)
    # ---------- 일봉
    DEP, DSN, DKS, DTR, DRN = daily(False)
    AEP, ASN, AKS, _, ARN = daily(True)
    print("일봉 계산", round(time.time() - t0), "초", flush=True)
    for nm, (ep, sn, ks, rn) in (("고정 -2%/+3%", (DEP, DSN, DKS, DRN)), ("ATR(14) 하단-1ATR/상단+1.5ATR(민감도)", (AEP, ASN, AKS, ARN))):
        for per in ("early", "late"):
            W(f"\n=== 일봉 구역 생존표 · {nm} · {'앞 절반(~2026-04-06)' if per == 'early' else '뒤 절반(2026-04-07~)'} · 에피소드 {(ep.period == per).sum()}개 "
              f"(T = 닿은 날 포함 버틴 일수) ===\n")
            surv_table(sn[sn.period == per], ep[ep.period == per], rn[rn.period == per], D_BINS, "일", out)
            summ.append(f"  일봉 {nm[:6]} {per}: " + summary_line(sn[sn.period == per], ep[ep.period == per], rn[rn.period == per], D_BINS, "일"))
            W("  -- 다시 두드린 횟수\n")
            k_table(ks[ks.period == per], out)
            attr_table(ep[ep.period == per], "cnt", [("1", lambda v: v == 1), ("2", lambda v: v == 2), ("3+", lambda v: v >= 3)], out)
            attr_table(ep[ep.period == per], "prior", [("0", lambda v: v == 0), ("1", lambda v: v == 1), ("2+", lambda v: v >= 2)], out)
    W("\n=== 단조성 요약(버틴 시간이 길수록 위비중) ===\n" + "\n".join(summ) + "\n")

    # ---------- 진입 규칙
    G = grid(TR, ("24시간", "정규장만"), EXITS, ("early", "late"), "정규장만", "사용자78", out, "분봉")
    DG = grid(DTR, ("일봉",), D_EXITS, ("early", "late"), "일봉", "사용자78", out, "일봉")
    n_int, n_day = G[["rule", "exit"]].drop_duplicates().shape[0], DG[["rule", "exit"]].drop_duplicates().shape[0]

    def beats(Gx, scope, un):
        g = Gx[(Gx.scope == scope) & (Gx.univ == un)].copy()
        g["ok"] = (g.avg > g["first"]) & (g.avg > g.rnd) & (g.n >= 10)
        pv = g.pivot_table(index=["rule", "exit"], columns="period", values="ok", aggfunc="first").fillna(False).astype(bool)
        both = pv[pv["early"] & pv["late"]]
        return both.index.tolist(), len(pv)

    W(f"\n=== 진입 규칙 시험 {n_int + n_day}개(분봉 {n_int} + 일봉 {n_day}) ===\n")
    W("  기준을 이김 = 평균 > 첫닿음 평균(같은 매도·범위·종목) 그리고 평균 > 무작위 평균, 건수 >= 10, 두 기간 모두\n")
    for (Gx, scope) in ((G, "정규장만"), (G, "24시간"), (DG, "일봉")):
        for un in ("사용자78", "전체"):
            lst, tot = beats(Gx, scope, un)
            W(f"  {scope} · {un}: 두 기간 모두 이긴 규칙 {len(lst)}/{tot}  " + ", ".join(f"{r}·{x}" for r, x in lst) + "\n")

    for scope in ("정규장만", "24시간"):
        for x in EXITS:
            W(f"\n  -- 버팀 시간별 진입 성과 · {scope} · 매도 {x} · 사용자78 (평균/건수; Tm·1% = 구역 위 1% 안일 때만)\n")
            rows = []
            for per in ("early", "late"):
                T = TR[(TR.scope == scope) & (TR.exit == x) & (TR.period == per) & TR.sym.isin(USER78)]
                g = G[(G.scope == scope) & (G.univ == "사용자78") & (G.exit == x) & (G.period == per)]
                f, fv = T[T.rule == "첫닿음"], T[T.rule == "첫닿음관통"]
                row = {"기간": per, "첫닿음": f"{f.ret.mean():+.2%}/{len(f)}", "관통": f"{fv.ret.mean():+.2%}/{len(fv)}"}
                for tm in TMS:
                    for band in BANDS:
                        r = g[g.rule == f"버팀{tm}분·{'1%안' if band else '생존'}"]
                        if len(r) and r.n.iloc[0] > 0:
                            row[f"{tm}{'·1%' if band else ''}"] = f"{r.avg.iloc[0]:+.2%}/{int(r.n.iloc[0])}"
                for kk in KSPECS:
                    r = g[g.rule == f"{kk}번째두드림"]
                    if len(r) and r.n.iloc[0] > 0:
                        row[f"k{kk}"] = f"{r.avg.iloc[0]:+.2%}/{int(r.n.iloc[0])}"
                row["무작위(첫닿음짝)"] = f"{f.rnd.mean():+.2%}"
                rows.append(row)
            W(pd.DataFrame(rows).to_string(index=False) + "\n")
    W("\n  -- 일봉 버팀 일수별 진입 성과 · 사용자78 (평균/건수)\n")
    for x in D_EXITS:
        rows = []
        for per in ("early", "late"):
            g = DG[(DG.univ == "사용자78") & (DG.exit == x) & (DG.period == per)]
            row = {"매도": x, "기간": per, "첫닿음": f"{g['first'].iloc[0]:+.2%}"}
            for tm in D_TMS:
                for band in BANDS:
                    r = g[g.rule == f"버팀{tm}일·{'1%안' if band else '생존'}"]
                    if len(r) and r.n.iloc[0] > 0:
                        row[f"{tm}일{'·1%' if band else ''}"] = f"{r.avg.iloc[0]:+.2%}/{int(r.n.iloc[0])}"
            r = g[g.rule == "2번째두드림"]
            if len(r) and r.n.iloc[0] > 0:
                row["k2"] = f"{r.avg.iloc[0]:+.2%}/{int(r.n.iloc[0])}"
            row["무작위"] = f"{g.rnd.mean():+.2%}"
            rows.append(row)
        W(pd.DataFrame(rows).to_string(index=False) + "\n")

    # ---------- 비교 기준 칸
    W("\n=== 비교 기준 (a) 첫 닿음 hi 지정가 / 관통(저가 <= hi×0.999만) / (c) 지금 이격매수 ===\n")
    for scope in ("정규장만", "24시간"):
        for per in ("early", "late"):
            for un, us in (("사용자78", USER78), ("전체", set(ALL))):
                T = TR[(TR.scope == scope) & (TR.period == per) & TR.sym.isin(us)]
                for x in EXITS:
                    W(f"  {scope} {per} {un} 매도 {x:<6} 첫닿음 {fmt(cell(T[(T.rule == '첫닿음') & (T.exit == x)]))}\n"
                      f"  {'':>28}관통   {fmt(cell(T[(T.rule == '첫닿음관통') & (T.exit == x)]))}\n")
                b = base_trades(per, scope == "정규장만")
                b = b[b.sym.isin(us)]
                W(f"  {scope} {per} {un} (c) 이격매수 RSI/8시: n={len(b)} 평균 {b.ret.mean():+.2%} 승률 {(b.ret > 0).mean():.0%} "
                  f"최악 {b.ret.min():+.1%} 합 {b.ret.sum():+.2f} t {tstat(b.ret.values, b.day.values):+.2f}\n")

    # ---------- 최선안(미리 정한 기준)
    sel = G[(G.scope == "정규장만") & (G.univ == "사용자78") & (G.period == "early") & (G.n >= 30)].sort_values("ex", ascending=False)
    W("\n=== 고르기(early·정규장만·사용자78, 건수>=30) 상위 8 — 무작위 대비 초과 평균 순 ===\n")
    for _, r in sel.head(8).iterrows():
        lt = G[(G.scope == "정규장만") & (G.univ == "사용자78") & (G.period == "late") & (G.rule == r.rule) & (G.exit == r.exit)].iloc[0]
        W(f"  {r.rule:<14} 매도 {r.exit:<6} early 평균 {r.avg:+.2%}(n {int(r.n)}, 초과 {r.ex:+.2%}, 첫닿음 {r['first']:+.2%}) "
          f"→ late 평균 {lt.avg:+.2%}(n {int(lt.n)}, 초과 {lt.ex:+.2%}, 첫닿음 {lt['first']:+.2%})\n")
    best = sel.iloc[0]
    W(f"\n=== 최선안: {best.rule} · 매도 {best.exit} — 모든 칸 ===\n")
    for scope in ("정규장만", "24시간"):
        for per in ("early", "late"):
            for un, us in (("사용자78", USER78), ("전체", set(ALL))):
                T = TR[(TR.scope == scope) & (TR.exit == best.exit) & (TR.period == per) & TR.sym.isin(us)]
                c = cell(T[T.rule == best.rule]); f = cell(T[T.rule == "첫닿음"])
                R = T[T.rule == best.rule][["sym", "eid", "ret"]]
                F = T[T.rule == "첫닿음"][["sym", "eid", "ret"]].rename(columns={"ret": "fret"})
                pr = R.merge(F, on=["sym", "eid"])
                W(f"  {scope} {per} {un}: {fmt(c)}\n      (a) 첫닿음 전체: {fmt(f)}\n"
                  f"      같은 에피소드의 첫닿음 평균 {pr.fret.mean():+.2%} vs 이 규칙 {pr.ret.mean():+.2%} (n {len(pr)})\n")
    dsel = DG[(DG.univ == "사용자78") & (DG.period == "early") & (DG.n >= 30)].sort_values("ex", ascending=False)
    W("\n=== 일봉 고르기(앞 절반·사용자78, 건수>=30) 상위 5 ===\n")
    for _, r in dsel.head(5).iterrows():
        lt = DG[(DG.univ == "사용자78") & (DG.period == "late") & (DG.rule == r.rule) & (DG.exit == r.exit)].iloc[0]
        W(f"  {r.rule:<12} 매도 {r.exit} 앞 평균 {r.avg:+.2%}(n {int(r.n)}, 초과 {r.ex:+.2%}, 첫닿음 {r['first']:+.2%}) "
          f"→ 뒤 평균 {lt.avg:+.2%}(n {int(lt.n)}, 초과 {lt.ex:+.2%}, 첫닿음 {lt['first']:+.2%})\n")
    dbest = dsel.iloc[0]
    for per in ("early", "late"):
        for un, us in (("사용자78", USER78), ("전체", set(ALL))):
            T = DTR[(DTR.exit == dbest.exit) & (DTR.period == per) & DTR.sym.isin(us)]
            W(f"  일봉 최선 {dbest.rule}·{dbest.exit} {per} {un}: {fmt(cell(T[T.rule == dbest.rule]))}\n"
              f"      (a) 첫닿음: {fmt(cell(T[T.rule == '첫닿음']))}\n")
    G.to_csv(OUT.replace(".txt", "_grid.csv"), index=False, encoding="utf-8-sig")
    DG.to_csv(OUT.replace(".txt", "_daily_grid.csv"), index=False, encoding="utf-8-sig")
    W(f"\n계산 {time.time() - t0:.0f}초\n")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w", encoding="utf-8").write(out.getvalue())
    print(out.getvalue())


if __name__ == "__main__":
    main()
