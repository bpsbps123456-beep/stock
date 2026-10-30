"""feat_audit.py 결과(sample_<mode>_recomputed.csv)를 엔진 값과 비교한다. 성과 열 없음."""
import sys

import numpy as np
import pandas as pd

OUT = 'C:/Users/geunt/OneDrive/Desktop/주식/data/derived/analysis/hatta_bar/audit/feat-audit'
RTOL, ATOL = 1e-6, 1e-12


def close(a, b, rtol=RTOL, atol=ATOL):
    a = np.asarray(a, dtype='float64')
    b = np.asarray(b, dtype='float64')
    both_nan = np.isnan(a) & np.isnan(b)
    both_inf = np.isinf(a) & np.isinf(b) & (np.sign(a) == np.sign(b))
    with np.errstate(invalid='ignore'):
        ok = np.isclose(a, b, rtol=rtol, atol=atol)
    return ok | both_nan | both_inf


def main(mode):
    d = (pd.read_parquet(f'{OUT}/full_recomputed.parquet') if mode == 'full'
         else pd.read_csv(f'{OUT}/sample_{mode}_recomputed.csv', index_col=0))
    d['prev_date'] = pd.to_datetime(d['prev_date'])
    print(f'== {mode}: rows {len(d)}, found by ts {int(d.found.sum())}, arm {d.arm.value_counts().to_dict()}, pat {d.pat.value_counts().to_dict()}')
    print('t == t_mine', int((d.t == d.t_mine).sum()), '| k == k_mine', int((d.k == d.k_mine).sum()),
          '| day', int((d.day == d.day_mine).sum()), '| mins', int((d.mins == d.mins_mine).sum()),
          '| sess', int((d.sess == d.sess_mine).sum()), '| ai', int((d.ai == d.ai_mine).sum()))
    exact = {
        'prevclose': 'prevclose_mine', 'cap_b': 'cap_b_mine', 'adv20': 'adv20_mine', 'ret5d': 'ret5d_mine',
        'atrd': 'atrd_mine', 'dayret': 'dayret_mine', 'qqq_30': 'qqq_30_mine', 'vol_tod': 'vol_tod_mine',
        'vwap_gap': 'vwap_gap_mine', 'dd': 'dd_mine', 'recsize': 'recsize_mine',
        'kcnt': 'kcnt_mine', 'earn': 'earn_mine',
    }
    res = []
    for ev, mi in exact.items():
        ok = close(d[ev], d[mi])
        res.append((ev, int(ok.sum()), len(d)))
    # 근사 입력 변수: qqq_day(전일 QQQ 15:59 봉 종가 근사), rel(그 qqq_day 사용)
    for ev, mi, tol in [('qqq_day', 'qqq_day_mine', 1e-6), ('qqq_day', 'qqq_day_mine', 2e-4),
                        ('qqq_day', 'qqq_day_mine_1600', 2e-4),
                        ('rel', 'rel_mine', 1e-6), ('rel', 'rel_mine', 2e-4)]:
        ok = close(d[ev], d[mi], rtol=0, atol=tol)
        res.append((f'{ev} vs {mi} |abs|<={tol:g}', int(ok.sum()), len(d)))
    for ev, mi in [('rel', 'rel_enginputs'), ('dayret', 'dayret_engpc')]:
        ok = close(d[ev], d[mi])
        res.append((f'{ev} vs {mi} (엔진 입력, 공식만)', int(ok.sum()), len(d)))
    print(pd.DataFrame(res, columns=['var', 'match', 'n']).assign(rate=lambda x: (x.match / x.n).round(4)).to_string(index=False))
    # 시점 누설 변형과의 일치 수(엔진이 누설판과 맞으면 위반)
    leak = [('prevclose', 'prevclose_leak'), ('adv20', 'adv20_leak'), ('ret5d', 'ret5d_leak'),
            ('atrd', 'atrd_leak'), ('vwap_gap', 'vwap_gap_leak'), ('vol_tod', 'vol_tod_leak'),
            ('qqq_30', 'qqq_30_leak'), ('qqq_day', 'qqq_day_leak')]
    print('-- leak variant (same-day / next-bar data) — on discriminating rows where PIT and leak values differ:')
    pit = {'prevclose': 'prevclose_mine', 'adv20': 'adv20_mine', 'ret5d': 'ret5d_mine', 'atrd': 'atrd_mine',
           'vwap_gap': 'vwap_gap_mine', 'vol_tod': 'vol_tod_mine', 'qqq_30': 'qqq_30_mine', 'qqq_day': 'qqq_day_mine'}
    for ev, lk in leak:
        if lk in d:
            has = d[lk].notna()
            disc = has & ~close(d[pit[ev]], d[lk], rtol=1e-6 if ev != 'qqq_day' else 0, atol=ATOL if ev != 'qqq_day' else 2e-4)
            if ev == 'vol_tod':
                disc &= d.vt_ndays >= 10
            m_leak = close(d.loc[disc, ev], d.loc[disc, lk], rtol=RTOL if ev != 'qqq_day' else 0, atol=ATOL if ev != 'qqq_day' else 2e-4)
            m_pit = close(d.loc[disc, ev], d.loc[disc, pit[ev]], rtol=RTOL if ev != 'qqq_day' else 0, atol=ATOL if ev != 'qqq_day' else 2e-4)
            print(f'   {ev:10s}: discriminating rows {int(disc.sum()):6d} | engine==PIT {int(m_pit.sum()):6d} | engine==leak {int(m_leak.sum()):6d}')
    # dd 변형
    for v in ['dd_incl', 'dd_bars120']:
        print(f'   dd vs {v}: {int(close(d.dd, d[v]).sum())}/{len(d)}')
    # QQQ 전일 종가 역산 vs 근사
    rel_err = d.q_prev_implied / d.q_prev_1559 - 1
    rel_err2 = d.q_prev_implied / d.q_prev_1600open - 1
    print('-- implied engine QQQ prevclose vs 15:59 close: |rel| median %.2e p95 %.2e max %.2e' % (
        rel_err.abs().median(), rel_err.abs().quantile(.95), rel_err.abs().max()))
    print('-- implied engine QQQ prevclose vs 16:00 open : |rel| median %.2e p95 %.2e max %.2e' % (
        rel_err2.abs().median(), rel_err2.abs().quantile(.95), rel_err2.abs().max()))
    diff = (d.qqq_day - d.qqq_day_mine).abs()
    print('-- |qqq_day eng - mine(15:59)|: median %.2e p95 %.2e max %.2e' % (diff.median(), diff.quantile(.95), diff.max()))
    print('-- QQQ lags (min): now lag dist', d.q_now_lag.value_counts().sort_index().head(8).to_dict(),
          '| q_now on other day', int((d.q_now_day != d.day).sum()), '| q_30 on other day', int((d.q_30_day != d.day).sum()))
    print('-- vol_tod: n prior days dist', d.vt_ndays.value_counts().sort_index().to_dict(),
          '| inf in eng', int(np.isinf(d.vol_tod).sum()), 'inf in mine', int(np.isinf(d.vol_tod_mine).sum()))
    print('-- vwap first bar mins dist', d.vwap_first_mins.value_counts().sort_index().head(6).to_dict())
    print('-- brk at k true:', int(d.brk_mine.sum()), '/', len(d))
    print('-- prev_date < day always:', bool((pd.to_datetime(d.prev_date) < pd.to_datetime(d.day.astype(str))).all()))
    # 불일치 예시
    show = ['arm', 'pat', 'sym', 'ts', 'day', 'mins']
    for ev, mi in exact.items():
        bad = ~close(d[ev], d[mi])
        if bad.any():
            print(f'\n### mismatch {ev}: {int(bad.sum())}')
            extra = {'vol_tod': ['vt_ndays', 'vt_med'], 'adv20': ['adv20_n'], 'kcnt': ['touch_mine', 'level_mine', 'S_mine'],
                     'dd': ['dd_incl', 'dd_bars120']}.get(ev, [])
            print(d.loc[bad, show + [ev, mi] + extra].head(12).to_string())
    return d


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'all')
