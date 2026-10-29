"""하따봉 탐지·결과 엔진 v2 — BRIEF.md v2 정의를 그대로 옮긴다(봉 마감 때 아는 값만 쓴다).

지지선 S_t = min(저가[t−N … t−2]). 처음 이탈 봉 t: 저가[t−1] ≥ S(1−tol), 저가[t] < S(1−tol), 저가[t] < 저가[t−1].
'여러 지지선' K: 확정 스윙 저점(왼쪽 p봉보다 작고 오른쪽 p봉보다 작거나 같음, j+p ≤ t−1, j ∈ [t−N, t−2]) + (S가 스윙이 아니면 S 한 점)
  touch = 가격 ≤ S×1.003 점 수, level = 가격 ≤ S×1.01 점을 오름차순으로 보며 묶음 첫 점보다 0.15% 넘게 높으면 새 묶음.
hatta_bar.pine과 같은 순서·같은 비교 부호를 쓴다.
"""
import os
import numpy as np
import pandas as pd
import numba as nb

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "../cache")
ET = "America/New_York"
SESS_NAMES = ["장전", "개장", "장중", "장후"]       # 04:00~09:30 / 09:30~11:00 / 11:00~16:00 / 16:00~20:00 ET
COST_FLOOR = np.array([0.003, 0.001, 0.0005, 0.003])  # E2 편도 바닥 비용(시간대별)
MIN_NOTIONAL = 25_000.0
MIN_TRADES = 5
MIN_RANGE = 0.002
MIN_DENS = 15
TOUCH_BAND, LEVEL_BAND, LEVEL_MERGE = 0.003, 0.01, 0.0015
NS, TOLS, PS = (60, 120, 240, 390), (0.0, 0.0005, 0.001), (2, 3, 5)


def load(sym, src="alp1m"):
    path = os.path.join(CACHE, src, f"{sym}.parquet")
    if not os.path.exists(path):
        return None
    d = pd.read_parquet(path)
    cols = ["Open", "High", "Low", "Close", "Volume"] + (["Trades"] if "Trades" in d.columns else [])
    d = d[cols].astype(float)
    d = d[(d.High >= d.Low) & (d.Low > 0)]
    return d[~d.index.duplicated()].sort_index()


def ema_pine(x, n):
    """Pine ta.ema 정의(첫 값으로 시작, alpha = 2/(n+1)) — 지표도 같은 식으로 직접 계산한다."""
    return pd.Series(x).ewm(alpha=2.0 / (n + 1), adjust=False).mean().to_numpy()


@nb.njit(cache=True)
def _rma_atr(h, l, c, n):
    N = len(h)
    tr = np.empty(N)
    out = np.full(N, np.nan)
    for i in range(N):
        if i == 0:
            tr[i] = h[i] - l[i]
        else:
            tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    if N < n:
        return out
    s = 0.0
    for i in range(n):
        s += tr[i]
    out[n - 1] = s / n
    for i in range(n, N):
        out[i] = (out[i - 1] * (n - 1) + tr[i]) / n
    return out


def prep(d, tv=False):
    """원자료 → 계산 배열 묶음. tv=True면 트레이딩뷰처럼 체결 건수 없이 한 점 봉만 뺀다(Pine 일치 점검용)."""
    idx = d.index
    o, h, l, c, v = (d[k].to_numpy() for k in ("Open", "High", "Low", "Close", "Volume"))
    mins = (idx.hour * 60 + idx.minute).to_numpy().astype(np.int64)
    day = (idx.year * 10000 + idx.month * 100 + idx.day).to_numpy().astype(np.int64)
    ts = ((idx.tz_convert("UTC") - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(minutes=1)).to_numpy().astype(np.int64)
    rng = h - l
    with np.errstate(invalid="ignore", divide="ignore"):
        wick = np.where(rng > 0, (np.minimum(o, c) - l) / rng, -1.0)
        cpos = np.where(rng > 0, (c - l) / rng, -1.0)
    vs = pd.Series(v)
    vavg = vs.shift(1).rolling(20, min_periods=20).mean()
    volr = (vs / vavg.where(vavg > 0)).to_numpy()                # 직전 20봉 평균이 0이면 없음(조건 실패)
    st = pd.Series(h, index=idx)
    hh120 = st.rolling("120min").max().to_numpy()              # (t−120분, t] 시각 기준
    dens = st.rolling("30min").count().to_numpy()               # 마감 전 30분 안 봉 수(자기 포함)
    e10, e25, e60 = ema_pine(c, 10), ema_pine(c, 25), ema_pine(c, 60)
    stack = (e10 < e25) & (e25 < e60)
    s30 = pd.Series(stack.astype(float)).rolling(30, min_periods=30).mean().to_numpy()
    e60_30 = pd.Series(e60).shift(30).to_numpy()
    persist = (s30 >= 0.8) & (e60 < e60_30)
    sess = np.select([mins < 570, mins < 660, mins < 960], [0, 1, 2], 3).astype(np.int8)
    tradable = ((mins >= 240) & (mins < 660)) | ((mins >= 1140) & (mins <= 1169))
    if "Trades" in d.columns and not tv:
        prints_ok = d["Trades"].to_numpy() >= MIN_TRADES
    else:                                                        # 야후: 체결 건수가 없어 한 점 봉만 뺀다
        prints_ok = ~((o == h) & (h == l) & (l == c))
    sane = prints_ok & (v * c >= MIN_NOTIONAL) & (rng >= MIN_RANGE * c)
    return dict(ts=ts, day=day, mins=mins, o=o, h=h, l=l, c=c, v=v, wick=wick, cpos=cpos, volr=volr, hh120=hh120,
                dens=dens, stack=stack, persist=persist, atr=_rma_atr(h, l, c, 14), sess=sess, tradable=tradable,
                sane=sane, e10=e10, e25=e25, e60=e60)


def support_line(low, N):
    """S_t = min(저가[t−N … t−2])."""
    return pd.Series(low).rolling(N - 1, min_periods=N - 1).min().shift(2).to_numpy()


def first_break(low, S, tol):
    prev = np.roll(low, 1)
    prev[0] = np.inf
    thr = S * (1 - tol)
    with np.errstate(invalid="ignore"):
        return (prev >= thr) & (low < thr) & (low < prev)


def swing_flags(low, p):
    """봉 j가 왼쪽 p봉보다 작고 오른쪽 p봉보다 작거나 같은 저가인가(확정은 j+p 봉 마감)."""
    s = pd.Series(low)
    left = s.shift(1).rolling(p, min_periods=p).min()
    right = s[::-1].shift(1).rolling(p, min_periods=p).min()[::-1]
    return ((s < left) & (s <= right)).to_numpy()


@nb.njit(cache=True)
def count_levels(ks, low, S, sw, N, p):
    """이탈 봉 ks마다 touch·level 수."""
    M = len(ks)
    touch = np.zeros(M, np.int32)
    level = np.zeros(M, np.int32)
    buf = np.empty(N + 2)
    for q in range(M):
        t = ks[q]
        s = S[q]
        n = 0
        s_in = False
        lo_j = max(t - N, 0)
        hi_j = t - 1 - p                     # j + p ≤ t − 1
        for j in range(lo_j, hi_j + 1):
            if sw[j]:
                buf[n] = low[j]
                if low[j] == s:
                    s_in = True
                n += 1
        if not s_in:
            buf[n] = s
            n += 1
        pts = np.sort(buf[:n])
        tc = 0
        for i in range(n):
            if pts[i] <= s * (1 + 0.003):
                tc += 1
        lc = 0
        anchor = -1.0
        for i in range(n):
            x = pts[i]
            if x > s * (1 + 0.01):
                break
            if lc == 0 or x > anchor * (1 + 0.0015):     # 묶음 첫(가장 낮은) 점보다 0.15% 넘게 높으면 새 지지선
                lc += 1
                anchor = x
        touch[q] = tc
        level[q] = lc
    return touch, level


@nb.njit(cache=True)
def outcomes(ts, day, mins, sess, o, h, l, c, cost):
    """봉 i 마감 뒤 E0(다음 봉 시가·다음 봉부터 판정)과 E2(다음 봉 종가·그다음 봉부터 판정).
    30분 창 = i 시작 + 1 ~ + 30분 시작 봉. 반환: valid[2,N], f[7,2,N](r1, r1net, hit, hits, race, mfe, mae), zz."""
    N = len(o)
    out_v = np.zeros((2, N), np.bool_)
    out_f = np.full((7, 2, N), np.nan)
    zz = np.full(N, -1, np.int64)
    for i in range(N - 1):
        if mins[i] > 1169:
            continue
        e = i + 1
        tlim = ts[i] + 30
        if ts[e] > ts[i] + 3 or day[e] != day[i]:      # 다음 봉이 신호 마감 뒤 2분 안에 시작
            continue
        z = e
        while z + 1 < N and ts[z + 1] <= tlim and day[z + 1] == day[i]:
            z += 1
        zz[i] = z
        cs_in = cost[sess[e]]                          # 진입 봉 시간대
        cs_out = cost[sess[z]]                         # 시간 청산 봉 시간대
        for m in range(2):
            if m == 0:
                en = o[e]
                j0 = e
            else:
                en = c[e]
                j0 = e + 1
                if j0 > z:
                    continue
            up, ups, dn = en * 1.01, en * 1.011, en * 0.99
            fu = -1
            fd = -1
            hs = False
            hi = -np.inf
            lo = np.inf
            for j in range(j0, z + 1):
                if fu < 0 and h[j] >= up:
                    fu = j
                if h[j] >= ups:
                    hs = True
                if fd < 0 and l[j] <= dn:
                    fd = j
                hi = max(hi, h[j])
                lo = min(lo, l[j])
            out_v[m, i] = True
            hit = fu >= 0
            out_f[0, m, i] = 0.01 if hit else c[z] / en - 1.0
            out_f[1, m, i] = (0.01 - cs_in) if hit else (c[z] / en - 1.0 - cs_in - cs_out)
            out_f[2, m, i] = 1.0 if hit else 0.0
            out_f[3, m, i] = 1.0 if hs else 0.0
            if fu >= 0 and (fd < 0 or fu < fd):
                out_f[4, m, i] = 1.0
            elif fd >= 0 and (fu < 0 or fd < fu):
                out_f[4, m, i] = 0.0
            elif fu >= 0 and fu == fd:
                out_f[4, m, i] = 0.5
            out_f[5, m, i] = hi / en - 1.0
            out_f[6, m, i] = lo / en - 1.0
    return out_v, out_f, zz


@nb.njit(cache=True)
def stop_exit(sig, stop, o, h, l, c, zz):
    """R2(E2): 다음 봉 종가 진입, 그다음 봉부터 목표 +1%와 손절(저가 < stop) 중 먼저, 한 봉에 둘 다면 손절."""
    M = len(sig)
    r2 = np.full(M, np.nan)
    amb = np.zeros(M, np.bool_)
    for q in range(M):
        i = sig[q]
        z = zz[i]
        if z < 0 or i + 2 > z:
            continue
        en = c[i + 1]
        st = stop[q]
        up = en * 1.01
        res = np.nan
        if en < st:                            # 진입가가 이미 손절선 아래면 바로 정리(0)
            res = 0.0
        else:
            for j in range(i + 2, z + 1):
                if o[j] < st:
                    res = o[j] / en - 1.0
                    break
                dn = l[j] < st
                upx = h[j] >= up
                if dn and upx:
                    amb[q] = True
                    res = st / en - 1.0
                    break
                if dn:
                    res = st / en - 1.0
                    break
                if upx:
                    res = 0.01
                    break
        if np.isnan(res):
            res = c[z] / en - 1.0
        r2[q] = res
    return r2, amb


@nb.njit(cache=True)
def dedup(sym_id, ts, gap):
    """같은 종목에서 앞에 남긴 신호 시작 gap분 안의 신호를 뺀다(종목·시각 정렬 입력)."""
    M = len(ts)
    keep = np.zeros(M, np.bool_)
    last_s = -1
    last_t = -10 ** 12
    for q in range(M):
        if sym_id[q] != last_s or ts[q] - last_t >= gap:
            keep[q] = True
            last_s = sym_id[q]
            last_t = ts[q]
    return keep


OUT_FIELDS = ["r1", "r1net", "hit", "hits", "race", "mfe", "mae"]


def bar_outcomes(a):
    v, f, zz = outcomes(a["ts"], a["day"], a["mins"], a["sess"], a["o"], a["h"], a["l"], a["c"], COST_FLOOR)
    res = {"zz": zz}
    for m, tag in ((0, "e0"), (1, "e2")):
        res[f"valid_{tag}"] = v[m]
        for k, name in enumerate(OUT_FIELDS):
            res[f"{name}_{tag}"] = f[k, m]
    # 변동성 맞춤: 같은 날·시간대 묶음 안 ATR% 3분위의 모든 유효 봉 평균
    df = pd.DataFrame({"day": a["day"], "sess": a["sess"], "atrp": a["atr"] / a["c"]})
    pct = df.groupby(["day", "sess"])["atrp"].rank(pct=True)
    ter = np.minimum(np.floor(pct * 3), 2)                       # ATR 없는 봉(예열 13봉)은 NaN → 묶음에서 빠짐
    keys = [df.day, df.sess, ter]
    for tag in ("e0", "e2"):
        for name in ("r1", "r1net", "hit", "race"):
            x = pd.Series(np.where(res[f"valid_{tag}"], res[f"{name}_{tag}"], np.nan))
            res[f"bv_{name}_{tag}"] = x.groupby(keys).transform("mean").to_numpy()
    return res


BAR_KEEP = ["r1_e0", "r1net_e0", "hit_e0", "race_e0", "r1_e2", "r1net_e2", "hit_e2", "hits_e2", "race_e2",
            "mfe_e2", "mae_e2", "bv_r1_e0", "bv_hit_e0", "bv_r1_e2", "bv_r1net_e2", "bv_hit_e2", "bv_race_e2"]


def _attach(X, a, out, tt, stop):
    tt = tt.astype(np.int64)
    X["t"] = tt
    X["ts"] = a["ts"][tt]
    X["day"] = a["day"][tt]
    X["sess"] = a["sess"][tt]
    X["tradable"] = a["tradable"][tt]
    X["valid"] = out["valid_e2"][tt] & out["valid_e0"][tt]
    for f in BAR_KEEP:
        X[f] = out[f][tt]
    r2, amb = stop_exit(tt, stop, a["o"], a["h"], a["l"], a["c"], out["zz"])
    X["r2_e2"] = r2
    X["amb_e2"] = amb
    return X


def candidates(sym, a, out, N, tol, plain=False):
    """(N, tol) 하나의 ①·② 후보 행(봉 모양 조건 전). plain=True면 '단순 저점'(N봉 신저가, '처음'·tol 없음) 비교군."""
    low = a["l"]
    if plain:
        S = pd.Series(low).rolling(N, min_periods=N).min().shift(1).to_numpy()   # min(저가[t−N … t−1])
        with np.errstate(invalid="ignore"):
            brk = low < S
    else:
        S = support_line(low, N)
        brk = first_break(low, S, tol)
    k = np.nonzero(brk)[0]
    k = k[(k >= 31) & (k < len(low) - 3)]
    if len(k) == 0:
        return None, None
    Sk = S[k]
    b = k - 1
    with np.errstate(invalid="ignore", divide="ignore"):
        dd = np.nan_to_num((a["hh120"][k] - low[k]) / a["hh120"][k], nan=-1.0)
        rngatr = np.nan_to_num((a["h"][k] - low[k]) / a["atr"][b], nan=0.0)
    base = dict(sym=sym, N=N, tol=tol, k=k, S=Sk, stack=a["stack"][b], persist=a["persist"][b], dd=dd,
                volr=np.nan_to_num(a["volr"][k], nan=0.0))
    for p in PS:
        sw = swing_flags(low, p)
        tc, lc = count_levels(k.astype(np.int64), low, Sk, sw, N, p)
        base[f"touch{p}"] = tc
        base[f"level{p}"] = lc
    thr = Sk * (1 - tol)
    # ① 신호 봉 = 이탈 봉 k
    A = pd.DataFrame(base)
    A["openok"] = a["o"][k] >= thr
    A["wick"] = a["wick"][k]
    A["cpos"] = a["cpos"][k]
    A["rngatr"] = rngatr
    A["above"] = a["c"][k] >= Sk
    A["sane"] = a["sane"][k] & (a["dens"][k] >= MIN_DENS)
    A = _attach(A, a, out, k, low[k].astype(float))
    # ② 이탈 봉 k(음봉) → 신호 봉 t = k+1
    t = k + 1
    B = pd.DataFrame(base)
    B["red"] = (a["c"][k] < a["o"][k]) & (a["day"][t] == a["day"][k])
    B["below"] = a["c"][k] < thr
    green = a["c"][t] > a["o"][t]
    rk = a["h"][k] - low[k]
    B["eng_basic"] = green & (a["c"][t] >= a["o"][k] - 0.1 * rk)
    B["eng_body"] = green & (a["c"][t] >= a["o"][k])
    B["eng_full"] = green & (a["c"][t] >= a["h"][k])
    B["eng_reclaim"] = green & (a["c"][t] >= Sk)
    B["sane"] = a["sane"][k] & a["sane"][t] & (a["dens"][t] >= MIN_DENS)
    B = _attach(B, a, out, t, np.minimum(low[k], low[t]).astype(float))
    B = B[B.red].drop(columns="red")
    return A.reset_index(drop=True), B.reset_index(drop=True)     # 결과 창이 유효한지는 valid 열(쓰는 쪽에서 거름)
