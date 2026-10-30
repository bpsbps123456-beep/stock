"""하따봉 2차 조건 변수 독립 감사 (blind: 성과 열은 읽지 않는다).

cond_2026.parquet에서 무작위 300행(seed 20261030)을 뽑아 BRIEF2 조건 변수를
원자료(1분봉·일봉·universe2·ev_meta)로 다시 계산해 엔진 값과 비교한다.
cond_build.py는 읽지 않았다. dd·kcnt는 BRIEF.md v3와 hatta_bar.pine 정의로 구현했다.

usage: py feat_audit.py [all|sig|full]
  all  = 전체 행에서 300행(seed 20261030)  — 주 표본
  sig  = arm=='sig' 행에서 300행(같은 seed) — 보조 표본
  full = 전체 56,930행(보조, 드문 경우 확인)
"""
import json
import sys

import numpy as np
import pandas as pd

BASE = 'C:/Users/geunt/OneDrive/Desktop/주식/data/derived/analysis/hatta_bar'
EARN = 'C:/Users/geunt/OneDrive/Desktop/주식/data/derived/analysis/earnings_support/cache/ev_meta.parquet'
OUT = BASE + '/audit/feat-audit'
SEED = 20261030
N_SAMPLE = 300
# 조건 변수와 키만 읽는다(성과 열 r1_*, hit_*, race_*, mfe/mae, r2, bt2_r1 제외)
COLS = ['arm', 'pat', 'sym', 'sid', 'ts', 'day', 't', 'k', 'sess', 'mins', 'ai',
        'prevclose', 'cap_b', 'adv20', 'ret5d', 'atrd', 'dayret', 'qqq_day', 'qqq_30',
        'rel', 'vol_tod', 'vwap_gap', 'dd', 'chain', 'kcnt', 'recsize', 'earn']
N_WIN, P_SW, TOL = 120, 2, 0.0005


def load_cond():
    return pd.read_parquet(f'{BASE}/cache/cond_2026.parquet', columns=COLS)


def load_1m(sym):
    m = pd.read_parquet(f'{BASE}/cache/alp1m/{sym}.parquet')
    m = m[~((m['High'] < m['Low']) | (m['Low'] <= 0))]
    m = m[~m.index.duplicated(keep='first')]
    idx = m.index
    out = {
        'ts': idx.as_unit('ns').asi8 // 60_000_000_000,
        'day': (idx.year * 10000 + idx.month * 100 + idx.day).to_numpy('int64'),
        'mins': (idx.hour * 60 + idx.minute).to_numpy('int64'),
    }
    for c in ['Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']:
        out[c] = m[c].to_numpy('float64')
    return out


class Daily:
    """한 종목 일봉 배열 + Wilder ATR(첫 TR = h-l)."""

    def __init__(self, g):
        self.dates = g['date'].to_numpy()
        self.c = g['c'].to_numpy('float64')
        h, lo = g['h'].to_numpy('float64'), g['l'].to_numpy('float64')
        self.dv = g['v'].to_numpy('float64') * g['vw'].to_numpy('float64')
        pc = np.r_[np.nan, self.c[:-1]]
        tr = np.maximum.reduce([h - lo, np.abs(h - pc), np.abs(lo - pc)])
        tr[0] = h[0] - lo[0]
        self.atr = pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean().to_numpy()

    def feats(self, D):
        """전일(마지막 날짜 < D)까지 값과, 당일 D 포함(누설) 값."""
        dates, c, dv, atr = self.dates, self.c, self.dv, self.atr
        p = int(np.searchsorted(dates, np.datetime64(D), side='left')) - 1
        r = {'prev_date': pd.Timestamp(dates[p]) if p >= 0 else pd.NaT}
        if p < 0:
            return r
        r['prevclose_mine'] = c[p]
        w = dv[max(0, p - 19):p + 1]
        r['adv20_n'] = len(w)
        r['adv20_mine'] = float(np.median(w)) if len(w) >= 15 else np.nan
        r['ret5d_mine'] = c[p] / c[p - 5] - 1 if p >= 5 else np.nan
        r['atrd_mine'] = atr[p] / c[p]
        if p + 1 < len(dates) and dates[p + 1] == np.datetime64(D):
            q = p + 1
            r['prevclose_leak'] = c[q]
            w2 = dv[max(0, q - 19):q + 1]
            r['adv20_leak'] = float(np.median(w2)) if len(w2) >= 15 else np.nan
            r['ret5d_leak'] = c[q] / c[q - 5] - 1 if q >= 5 else np.nan
            r['atrd_leak'] = atr[q] / c[q]
        return r


def kcount(L, k):
    """BRIEF v3 신저가 방식·Pine: 이탈 봉 k에서 S'·touch·level·brk."""
    S = L[k - N_WIN:k].min()                         # min(low[k-N .. k-1])
    lo_j, hi_j = max(k - N_WIN, P_SW), k - 1 - P_SW   # 확정 스윙 j in [k-N, k-1-p]
    pts = [L[j] for j in range(lo_j, hi_j + 1)
           if L[j] < min(L[j - 1], L[j - 2]) and L[j] <= min(L[j + 1], L[j + 2])]
    if not any(x == S for x in pts):
        pts.append(S)
    pts.sort()
    touch = sum(1 for x in pts if x <= S * (1 + 0.003))
    level, anchor = 0, -1.0
    for x in pts:
        if x <= S * (1 + 0.01):
            if level == 0 or x > anchor * (1 + 0.0015):
                level += 1
                anchor = x
    return S, touch, level, bool(L[k] < S * (1 - TOL))


def recompute(rows):
    daily = pd.read_parquet(f'{BASE}/cache/daily_2026.parquet').sort_values(['sym', 'date'])
    dgrp = {s: Daily(g.reset_index(drop=True)) for s, g in daily.groupby('sym', sort=False)}
    uni = json.load(open(f'{BASE}/results/universe2.json', encoding='utf-8'))
    ev = pd.read_parquet(EARN, columns=['ticker', 'reaction_date'])
    ev_set = set(zip(ev['ticker'], ev['reaction_date']))
    ev_tick = set(ev['ticker'])

    q = load_1m('QQQ')
    q_days = np.unique(q['day'])
    qprev = {}
    for d_ in q_days[1:]:
        pd_ = q_days[q_days < d_][-1]
        sel = np.flatnonzero((q['day'] == pd_) & (q['mins'] <= 959))
        s16 = np.flatnonzero((q['day'] == pd_) & (q['mins'] == 960))
        qprev[d_] = (q['Close'][sel[-1]], int(q['mins'][sel[-1]]), q['Open'][s16[0]] if len(s16) else np.nan)

    out = []
    for sym, g in rows.groupby('sym', sort=False):
        m = load_1m(sym)
        dd_ = dgrp[sym]
        days_u = np.unique(m['day'])
        # (날짜, 분) -> 거래량
        vol_map = dict(zip(zip(m['day'].tolist(), m['mins'].tolist()), m['Volume'].tolist()))
        day_start = {d_: int(np.searchsorted(m['day'], d_, side='left')) for d_ in days_u}
        for ridx, r in zip(g.index, g.itertuples(index=False)):
            o = {'ridx': ridx}
            pos = int(np.searchsorted(m['ts'], r.ts, side='left'))
            o['found'] = pos < len(m['ts']) and m['ts'][pos] == r.ts
            if not o['found']:
                out.append(o)
                continue
            i = pos
            D = pd.Timestamp(str(r.day))
            o['t_mine'] = i
            k = i if r.pat == 'A' else i - 1
            o['k_mine'] = k
            o['day_mine'] = int(m['day'][i])
            mn = int(m['mins'][i])
            o['mins_mine'] = mn
            o['sess_mine'] = 0 if mn < 570 else (1 if mn < 660 else (2 if mn < 960 else 3))
            o['ai_mine'] = bool(uni[sym]['ai'])
            o['B_same_day'] = bool(m['day'][k] == m['day'][i])
            # ---- 일봉 ----
            o.update(dd_.feats(D))
            pc = o.get('prevclose_mine', np.nan)
            o['cap_b_mine'] = uni[sym]['shares'] * pc / 1e9
            # ---- 1분봉 ----
            O_, H_, L_, C_, V_ = m['Open'][i], m['High'][i], m['Low'][i], m['Close'][i], m['Volume'][i]
            o['dayret_mine'] = L_ / pc - 1
            o['dayret_engpc'] = L_ / r.prevclose - 1
            d0 = day_start[r.day]
            vw = m['VWAP'][d0:i + 1]
            tp = (m['High'][d0:i + 1] + m['Low'][d0:i + 1] + m['Close'][d0:i + 1]) / 3
            vw = np.where(np.isnan(vw), tp, vw)
            vv = m['Volume'][d0:i + 1]
            o['vwap_gap_mine'] = C_ / ((vw * vv).sum() / vv.sum()) - 1
            if i + 1 < len(m['ts']) and m['day'][i + 1] == r.day:
                vw2, vv2 = m['VWAP'][d0:i + 2], m['Volume'][d0:i + 2]
                o['vwap_gap_leak'] = m['Close'][i + 1] / ((vw2 * vv2).sum() / vv2.sum()) - 1
            o['vwap_first_mins'] = int(m['mins'][d0])
            prior = days_u[days_u < r.day][-10:]
            vols = [vol_map.get((int(d_), mn), 0.0) for d_ in prior]
            o['vt_ndays'] = len(prior)
            med = float(np.median(vols)) if vols else np.nan
            o['vt_med'] = med
            o['vol_tod_mine'] = (V_ / med) if med > 0 else (np.inf if V_ > 0 else np.nan)
            prior9 = [int(x) for x in days_u[days_u < r.day][-9:]] + [r.day]
            med2 = float(np.median([vol_map.get((d_, mn), 0.0) for d_ in prior9]))
            o['vol_tod_leak'] = (V_ / med2) if med2 > 0 else np.inf
            o['recsize_mine'] = (min(O_, C_) - L_) / C_ if r.pat == 'A' else (C_ - O_) / C_
            tsk = m['ts'][k]
            j0 = int(np.searchsorted(m['ts'], tsk - 120, side='right'))
            hh = m['High'][j0:k + 1].max()
            o['dd_mine'] = (hh - m['Low'][k]) / hh
            j0i = int(np.searchsorted(m['ts'], tsk - 120, side='left'))
            hhi = m['High'][j0i:k + 1].max()
            o['dd_incl'] = (hhi - m['Low'][k]) / hhi
            hb = m['High'][max(0, k - 119):k + 1].max()
            o['dd_bars120'] = (hb - m['Low'][k]) / hb
            S, touch, level, brk = kcount(m['Low'], k)
            o['S_mine'], o['touch_mine'], o['level_mine'], o['brk_mine'] = S, touch, level, brk
            o['kcnt_mine'] = level if r.pat == 'A' else touch
            # ---- QQQ ----
            qp = int(np.searchsorted(q['ts'], r.ts, side='right')) - 1
            q30 = int(np.searchsorted(q['ts'], r.ts - 30, side='right')) - 1
            qn = int(np.searchsorted(q['ts'], r.ts + 1, side='right')) - 1
            qn30 = int(np.searchsorted(q['ts'], r.ts + 1 - 30, side='right')) - 1
            o['q_now'], o['q_now_day'], o['q_now_lag'] = q['Close'][qp], int(q['day'][qp]), int(r.ts - q['ts'][qp])
            o['q_30'], o['q_30_day'], o['q_30_lag'] = q['Close'][q30], int(q['day'][q30]), int(r.ts - q['ts'][q30])
            o['qqq_30_mine'] = o['q_now'] / o['q_30'] - 1
            o['qqq_30_leak'] = q['Close'][qn] / q['Close'][qn30] - 1
            qc, qcm, q16 = qprev[r.day]
            o['q_prev_1559'], o['q_prev_1559_mins'], o['q_prev_1600open'] = qc, qcm, q16
            o['qqq_day_mine'] = o['q_now'] / qc - 1
            o['qqq_day_mine_1600'] = o['q_now'] / q16 - 1
            o['qqq_day_leak'] = q['Close'][qn] / qc - 1
            o['q_prev_implied'] = o['q_now'] / (1 + r.qqq_day)
            o['rel_mine'] = (C_ / pc - 1) - o['qqq_day_mine']
            o['rel_enginputs'] = (C_ / r.prevclose - 1) - r.qqq_day
            # ---- 실적 ----
            ds = f'{str(r.day)[:4]}-{str(r.day)[4:6]}-{str(r.day)[6:]}'
            o['earn_mine'] = 1 if (sym, ds) in ev_set else (0 if sym in ev_tick else -1)
            out.append(o)
    res = pd.DataFrame(out).set_index('ridx')
    return rows.join(res, how='left')


def main(mode='all'):
    df = load_cond()
    if mode == 'full':
        rows = df
    else:
        pool = df if mode == 'all' else df[df['arm'] == 'sig']
        rows = pool.sample(n=N_SAMPLE, random_state=SEED)
    res = recompute(rows.copy())
    res.to_csv(f'{OUT}/sample_{mode}_recomputed.csv') if mode != 'full' else res.to_parquet(f'{OUT}/full_recomputed.parquet')
    return res


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'
    main(mode)
    print('done', mode)
