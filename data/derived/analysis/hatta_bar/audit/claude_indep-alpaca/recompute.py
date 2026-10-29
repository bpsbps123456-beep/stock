"""하따봉 v3 원안 ①·② 독립 재계산 (BRIEF.md만 보고 작성, engine/grid/evaluate 미참조).

출력: out/mine_A.csv, out/mine_B.csv, out/ctx_A.csv, out/ctx_B.csv (겹침 제거 전 전체 후보 포함 플래그)
      out/summary.json
"""
import glob
import json
import os
import sys

import numba as nb
import numpy as np
import pandas as pd

BASE = r'C:/Users/geunt/OneDrive/Desktop/주식/data/derived/analysis/hatta_bar'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
os.makedirs(OUT, exist_ok=True)

N = 120
TOL = 0.0005
P = 2
D_MIN = 0.02
DD_WIN = 120          # 분(시각 기준), 이탈 봉 포함
DENS_WIN = 30         # 분, 신호 봉 포함
DENS_MIN = 15
GAP_MAX = 2           # 신호 마감(ts+1) 뒤 다음 봉 시작까지 분


def session_of(mod):
    # 0 pre 04:00-09:30, 1 open 09:30-11:00, 2 mid 11:00-16:00, 3 post 16:00-20:00
    return np.where(mod < 570, 0, np.where(mod < 660, 1, np.where(mod < 960, 2, 3)))


COST = np.array([0.003, 0.001, 0.0005, 0.003])


@nb.njit(cache=True)
def ema(x, n):
    a = 2.0 / (n + 1.0)
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


@nb.njit(cache=True)
def swing_lows(L, p, right_strict=False):
    n = len(L)
    sw = np.zeros(n, np.bool_)
    for j in range(p, n - p):
        ok = True
        for i in range(1, p + 1):
            if not (L[j] < L[j - i]):
                ok = False
                break
            if (L[j] > L[j + i]) or (right_strict and L[j] == L[j + i]):
                ok = False
                break
        sw[j] = ok
    return sw


@nb.njit(cache=True)
def level_count(L, sw, t, S, Nw, p, chain=False, add_s=True):
    """창 [t-N, t-1-p]의 확정 스윙 저점 + (정확히 S인 스윙이 없으면) S.
    S*1.01 이하 점을 오름차순, 묶음 첫 점 * 1.0015 초과면 새 묶음. (level, touch)"""
    lo = t - Nw
    hi = t - 1 - p
    buf = np.empty(Nw + 1)
    m = 0
    hasS = False
    for j in range(lo, hi + 1):
        if j < 0:
            continue
        if sw[j] and L[j] <= S * 1.01:
            buf[m] = L[j]
            m += 1
            if L[j] == S:
                hasS = True
    if (not hasS) and add_s:
        buf[m] = S
        m += 1
    if m == 0:
        return 0, 0
    v = np.sort(buf[:m])
    cnt = 1
    first = v[0]
    for i in range(1, m):
        if v[i] > first * 1.0015:
            cnt += 1
            first = v[i]
        elif chain:
            first = v[i]
    touch = 0
    for i in range(m):
        if v[i] <= S * 1.003:
            touch += 1
    return cnt, touch


@nb.njit(cache=True)
def time_window_max(H, ts, win):
    # 각 i에 대해 ts in [ts_i - (win-1), ts_i]인 봉의 고가 최대
    n = len(H)
    out = np.empty(n)
    lo = 0
    for i in range(n):
        while ts[lo] < ts[i] - (win - 1):
            lo += 1
        mx = H[lo]
        for j in range(lo + 1, i + 1):
            if H[j] > mx:
                mx = H[j]
        out[i] = mx
    return out


@nb.njit(cache=True)
def time_window_max_excl(H, ts, win):
    # 이탈 봉 자신을 뺀 변형: ts in [ts_i - (win-1), ts_i) 의 고가 최대(없으면 nan)
    n = len(H)
    out = np.full(n, np.nan)
    lo = 0
    for i in range(n):
        while ts[lo] < ts[i] - (win - 1):
            lo += 1
        if lo < i:
            mx = H[lo]
            for j in range(lo + 1, i):
                if H[j] > mx:
                    mx = H[j]
            out[i] = mx
    return out


@nb.njit(cache=True)
def time_window_count(ts, win):
    n = len(ts)
    out = np.empty(n, np.int64)
    lo = 0
    for i in range(n):
        while ts[lo] < ts[i] - (win - 1):
            lo += 1
        out[i] = i - lo + 1
    return out


@nb.njit(cache=True)
def outcome_e2(t, O, H, L, C, ts, day, sess, cost, win=30, exit_cost_entry=False, same_bar=0.5):
    """E2: 진입 = t+1 종가, 창 = ts_t+1..ts_t+30 같은 날, 판정 봉 t+2..w.
    반환 (valid, hit, r1, r1net, race)"""
    n = len(C)
    e = t + 1
    if e >= n or day[e] != day[t] or ts[e] > ts[t] + win:
        return False, np.nan, np.nan, np.nan, np.nan
    w = e
    while w + 1 < n and day[w + 1] == day[t] and ts[w + 1] <= ts[t] + win:
        w += 1
    if w < e + 1:
        return False, np.nan, np.nan, np.nan, np.nan
    ent = C[e]
    tgt = ent * 1.01
    dn = ent * 0.99
    hit = False
    race = np.nan
    for j in range(e + 1, w + 1):
        up = H[j] >= tgt
        lw = L[j] <= dn
        if np.isnan(race):
            if up and lw:
                race = same_bar
            elif up:
                race = 1.0
            elif lw:
                race = 0.0
        if up:
            hit = True
    if hit:
        r1 = 0.01
        r1net = r1 - cost[sess[e]]
    else:
        r1 = C[w] / ent - 1.0
        r1net = r1 - cost[sess[e]] - (cost[sess[e]] if exit_cost_entry else cost[sess[w]])
    return True, 1.0 if hit else 0.0, r1, r1net, race


DEFAULT = dict(dd_win=DD_WIN, dd_excl_break=False, dens_win=DENS_WIN, gap_max=GAP_MAX, gap_strict=False,
               s_sameday=False, b_sane_k=True, chain=False, add_s=True, right_strict=False, persist_min=24,
               win=30, exit_cost_entry=False, same_bar=0.5, ctx_b_sameday=False, nwin=N, dollar_vwap=False, wick_strict=False, wick_eps=0.0, trades_min=5, eng_strict=False)


def process(sym, sid, cfg=None, df=None):
    c = dict(DEFAULT, **(cfg or {}))
    if df is None:
        df = pd.read_parquet(os.path.join(BASE, 'cache', 'alp1m', sym + '.parquet'))
    df = df.sort_index()
    idx = df.index
    O = df['Open'].to_numpy(float)
    H = df['High'].to_numpy(float)
    L = df['Low'].to_numpy(float)
    C = df['Close'].to_numpy(float)
    V = df['Volume'].to_numpy(float)
    TR = df['Trades'].to_numpy(float)
    utc = idx.tz_convert('UTC')
    ts = ((utc - pd.Timestamp('1970-01-01', tz='UTC')) // pd.Timedelta('1min')).to_numpy(np.int64)
    mod = (idx.hour * 60 + idx.minute).to_numpy(np.int64)
    day = (idx.year * 10000 + idx.month * 100 + idx.day).to_numpy(np.int64)
    sess = session_of(mod).astype(np.int64)
    n = len(C)

    e10, e25, e60 = ema(C, 10), ema(C, 25), ema(C, 60)
    al = ((e10 < e25) & (e25 < e60)).astype(float)
    cnt30 = pd.Series(al).rolling(30, min_periods=30).sum().to_numpy()
    e60_30 = np.full(n, np.nan)
    e60_30[30:] = e60[:-30]
    persist = (cnt30 >= c['persist_min']) & (e60 < e60_30)   # 기준봉 b에서
    stack = al.astype(bool)

    NW = c['nwin']
    Sp = pd.Series(L).rolling(NW, min_periods=NW).min().shift(1).to_numpy()  # min L[t-N..t-1]
    if c['s_sameday']:
        g = pd.Series(L).groupby(day)
        Sp = g.transform(lambda s: s.rolling(NW, min_periods=1).min().shift(1)).to_numpy()
    brk = L < Sp * (1 - TOL)
    brk = np.where(np.isnan(Sp), False, brk)

    sw = swing_lows(L, P, c['right_strict'])
    mxH = time_window_max(H, ts, c['dd_win'])
    if c['dd_excl_break']:
        mxH = time_window_max_excl(H, ts, c['dd_win'])
    dd = (mxH - L) / mxH
    dens = time_window_count(ts, c['dens_win'])

    rng = H - L
    px = df['VWAP'].to_numpy(float) if c['dollar_vwap'] else C
    sane_bar = (TR >= c['trades_min']) & (V * px >= 25000) & (rng >= 0.002 * C)
    nxt_ok = np.zeros(n, bool)
    gap = ts[1:] - (ts[:-1] + 1)
    nxt_ok[:-1] = ((gap < c['gap_max']) if c['gap_strict'] else (gap <= c['gap_max'])) & (day[1:] == day[:-1])
    late_ok = mod < 19 * 60 + 30
    trad = (mod >= 240) & (mod < 660) | (mod >= 1140) & (mod < 1170)

    # 이탈 봉 공통 특징
    kb = np.nonzero(brk)[0]
    level = np.zeros(n, np.int64)
    touch = np.zeros(n, np.int64)
    for k in kb:
        level[k], touch[k] = level_count(L, sw, k, Sp[k], NW, P, c['chain'], c['add_s'])
    pers_b = np.zeros(n, bool)
    pers_b[1:] = persist[:-1]   # 이탈 봉 k의 기준봉 k-1

    base_k = brk & pers_b & (dd >= D_MIN)

    rows = []
    # ① 신호 봉 t = 이탈 봉 k
    for t in np.nonzero(base_k & (level >= 2))[0]:
        S = Sp[t]
        openok = O[t] >= S * (1 - TOL)
        wick = (min(O[t], C[t]) - L[t]) / rng[t] if rng[t] > 0 else np.nan
        shape = (wick > 0.5) if c['wick_strict'] else (wick >= 0.5 - c['wick_eps'])
        fixed = openok and sane_bar[t] and dens[t] >= DENS_MIN and nxt_ok[t] and late_ok[t]
        v, hit, r1, r1n, race = outcome_e2(t, O, H, L, C, ts, day, sess, COST, c['win'], c['exit_cost_entry'], c['same_bar'])
        rows.append(('A', sid, sym, t, t, ts[t], day[t], sess[t], trad[t], S, level[t], touch[t], dd[t],
                     openok, wick, bool(shape), bool(sane_bar[t]), int(dens[t]), bool(nxt_ok[t]), bool(late_ok[t]),
                     bool(fixed), v, hit, r1, r1n, race, False, False))
    # ② 이탈 봉 k, 신호 봉 t = k+1
    for k in np.nonzero(base_k)[0]:
        t = k + 1
        if t >= n:
            continue
        S = Sp[k]
        red = C[k] < O[k]
        below = C[k] < S * (1 - TOL)
        sameday = day[t] == day[k]
        green = C[t] > O[t]
        eng = (C[t] > O[k] - 0.1 * rng[k]) if c['eng_strict'] else (C[t] >= O[k] - 0.1 * rng[k])
        shape = sameday and green and eng
        if (not shape) and c['ctx_b_sameday'] and not sameday:
            continue
        fixed = red and below and sane_bar[t] and (sane_bar[k] or not c['b_sane_k']) and dens[t] >= DENS_MIN and nxt_ok[t] and late_ok[t]
        v, hit, r1, r1n, race = outcome_e2(t, O, H, L, C, ts, day, sess, COST, c['win'], c['exit_cost_entry'], c['same_bar'])
        rows.append(('B', sid, sym, k, t, ts[t], day[t], sess[t], trad[t], S, level[k], touch[k], dd[k],
                     True, np.nan, bool(shape), bool(sane_bar[t]), int(dens[t]), bool(nxt_ok[t]), bool(late_ok[t]),
                     bool(fixed), v, hit, r1, r1n, race, bool(red and below), bool(sane_bar[k])))
    return rows


COLS = ['arm', 'sid', 'sym', 'k', 't', 'ts', 'day', 'sess', 'tradable', 'S', 'level', 'touch', 'dd',
        'openok', 'wick', 'shape', 'sane_t', 'dens', 'nxt_ok', 'late_ok', 'fixed', 'valid', 'hit', 'r1', 'r1net',
        'race', 'redbelow', 'sane_k']


def dedup(df):
    keep = np.zeros(len(df), bool)
    df = df.sort_values(['sid', 'ts'])
    last = {}
    for i, (s, tsv) in enumerate(zip(df['sid'].to_numpy(), df['ts'].to_numpy())):
        if s not in last or tsv - last[s] >= 30:
            keep[i] = True
            last[s] = tsv
    return df[keep]


def stats(df):
    r = df['race'].dropna()
    return dict(n=int(len(df)), days=int(df['day'].nunique()), syms=int(df['sid'].nunique()),
                hit_e2=float(df['hit'].mean()), r1_e2=float(df['r1'].mean()), r1net_e2=float(df['r1net'].mean()),
                race_e2=float(r.mean()), race_resolved=float(len(r) / len(df)) if len(df) else np.nan)


_DF = {}


def universe():
    files = sorted(glob.glob(os.path.join(BASE, 'cache', 'alp1m', '*.parquet')))
    syms = sorted(os.path.basename(f)[:-8] for f in files)
    syms = [s for s in syms if s not in ('SPY', 'QQQ')]
    sys.path.insert(0, r'C:/Users/geunt/OneDrive/Desktop/주식')
    import watchlist
    wl = sorted(t for t in watchlist.AI_SUBSECTOR if '.' not in t)
    assert wl == syms, (set(wl) ^ set(syms))
    return syms


def candidates(cfg=None):
    rows = []
    for sid, s in enumerate(universe()):
        if s not in _DF:
            _DF[s] = pd.read_parquet(os.path.join(BASE, 'cache', 'alp1m', s + '.parquet'))
        rows += process(s, sid, cfg, _DF[s])
    return pd.DataFrame(rows, columns=COLS)


def select(allc, arm, dedup_stage='final'):
    """dedup_stage: final(모든 거르기 뒤 겹침 제거, 기본) / pre_tradable(시간대 거르기 전) / pre_fixed(위생·유효 거르기 전)"""
    c = allc[(allc.arm == arm) & (allc.day >= 20260701) & (allc.day <= 20261028)]
    out = {}
    for name, part in (('sig', c[c['shape']]), ('ctx', c[~c['shape']])):
        if dedup_stage == 'final':
            d = dedup(part[part.fixed & part.valid & part.tradable])
        elif dedup_stage == 'pre_tradable':
            d = dedup(part[part.fixed & part.valid])
            d = d[d.tradable]
        else:
            d = dedup(part)
            d = d[d.fixed & d.valid & d.tradable]
        out[name] = d
    return out['sig'], out['ctx']


def run(cfg=None, dedup_stage='final', save=True, verbose=True):
    allc = candidates(cfg)
    if save:
        allc.to_parquet(os.path.join(OUT, 'cands.parquet'))
    res = {}
    for arm in ('A', 'B'):
        sig, ctx = select(allc, arm, dedup_stage)
        if save:
            sig.to_csv(os.path.join(OUT, f'mine_{arm}.csv'), index=False)
            ctx.to_csv(os.path.join(OUT, f'ctx_{arm}.csv'), index=False)
        res[arm] = dict(back=stats(sig[sig.day >= 20260901]), front=stats(sig[sig.day < 20260901]), all=stats(sig),
                        ctx_back=stats(ctx[ctx.day >= 20260901]))
    if save:
        json.dump(res, open(os.path.join(OUT, 'summary.json'), 'w'), indent=1)
    if verbose:
        print(json.dumps(res, indent=1))
    return res


if __name__ == '__main__':
    run()
