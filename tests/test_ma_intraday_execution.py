"""Independent execution/scheduling fixtures for the intraday MA experiment."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'data/derived/analysis/ma_entry_filter_20261108/intraday_review.py'
NY = 'America/New_York'


def ny(value):
    return pd.Timestamp(value, tz=NY)


class IntradayExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('ma_intraday_execution_subject', SOURCE)
        cls.I = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.I)

    def fixture(self, *, two_stocks=False, missing_raw=None, missing_weekly=None):
        days = ['2025-12-31', '2026-01-02', '2026-01-05', '2026-01-06',
                '2026-01-07', '2026-01-08', '2026-01-09', '2026-01-12', '2026-01-13']
        tickers = ['A','B'] if two_stocks else ['A']
        frame = pd.DataFrame(100., index=days, columns=tickers)
        daily = SimpleNamespace(days=days, adj=frame.copy(), opn=frame.copy(),
                                cls=frame.copy(), fx=np.ones(len(days)),
                                ticker={t:t for t in tickers}, keys=tickers,
                                names={t:t for t in tickers}, sector={t:'test' for t in tickers},
                                snaps={}, pdfs=[], _w={}, etf=pd.Series(dtype=float))
        index = pd.DatetimeIndex([t for day in days for t in
            pd.date_range(ny(day+' 04:00'), ny(day+' 19:30'), freq='30min')])
        raw = {t:pd.DataFrame({'Open':100., 'Close':100.}, index=index) for t in tickers}
        if missing_raw is not None:
            ticker, stamp = missing_raw
            raw[ticker] = raw[ticker].drop(ny(stamp))
        if missing_weekly is not None:
            ticker, day = missing_weekly
            daily.opn.loc[day,ticker] = np.nan
        with patch.object(self.I.S, 'momentum_ranking', return_value=('fixture', [])):
            book, meta, okay = self.I.event_book(daily,raw,'extended')
        return daily,book,meta,okay

    def signals(self, meta, tickers):
        prices = np.full((len(meta),len(tickers)),100.)
        means = prices.copy()
        stamps = np.repeat(pd.DatetimeIndex(meta.time).as_unit('ns').asi8[:,None],
                           len(tickers),axis=1)-60_000_000_000
        return prices,means,stamps

    def run_case(self, fixture, signals, *, policy='awake', chooser=None):
        daily,book,meta,okay = fixture
        calls = []
        def target(b,k,day,current):
            self.assertIs(b,daily)
            calls.append((day,set(current)))
            allocations = chooser(day,current) if chooser else {t:.4 for t in daily.ticker}
            return {'pdf':'fixture', 'ranked':[{'key':t} for t in allocations],
                    'selected':list(allocations), 'raw':dict(allocations),
                    'target':dict(allocations)}
        with patch.object(self.I.S,'weekly_target',target):
            path,logs = self.I.run_variant(daily,book,meta,okay,None,'SMA',2,policy,signals=signals)
        return path,logs,calls

    def test_first_weekday_premarket_does_not_consume_nine_thirty_rebalance(self):
        fixture = self.fixture()
        _,book,meta,_ = fixture
        path,logs,calls = self.run_case(fixture,self.signals(meta,book.opn.columns))
        self.assertEqual([day for day,_ in calls], ['2026-01-02','2026-01-05','2026-01-12'])
        self.assertEqual(list(path.rebalances), [ny(d+' 09:30').isoformat() for d,_ in calls])
        self.assertTrue(all(pd.Timestamp(e['date']).strftime('%H:%M') == '09:30'
                            for e in path.events if e['side'] == 'buy'))
        monday = meta[(meta.session == '2026-01-05')&(meta.time < ny('2026-01-05 09:30'))]
        self.assertTrue(all(w == (2026,1) for w in monday.week))
        self.assertEqual(meta.loc[meta.time.eq(ny('2026-01-05 09:30')),'week'].iloc[0], (2026,2))
        self.assertEqual(logs['exits'],[])

    def test_asleep_break_and_recovery_does_not_create_a_sticky_sell(self):
        fixture = self.fixture()
        _,book,meta,_ = fixture
        signals = self.signals(meta,book.opn.columns)
        asleep = (meta.time >= ny('2026-01-06 12:00'))&(meta.time < ny('2026-01-06 18:00'))
        signals[0][asleep,0] = 80.
        awake_path,awake_logs,_ = self.run_case(fixture,signals)
        automatic,auto_logs,_ = self.run_case(fixture,signals,policy='unrestricted')
        self.assertEqual(awake_logs['exits'],[])
        self.assertEqual(len(auto_logs['exits']),1)
        self.assertEqual(pd.Timestamp(auto_logs['exits'][0]['event']),ny('2026-01-06 12:00'))
        self.assertIn('A',awake_path.weights[-1])
        auto_buys = [pd.Timestamp(e['date']) for e in automatic.events if e['side']=='buy']
        self.assertEqual(auto_buys,[ny('2026-01-02 09:30'),ny('2026-01-12 09:30')])

    def test_pending_break_sells_once_when_awake_without_redistributing_other_holding(self):
        fixture = self.fixture(two_stocks=True)
        _,book,meta,_ = fixture
        signals = self.signals(meta,book.opn.columns)
        below = (meta.time >= ny('2026-01-06 12:00'))&(meta.time < ny('2026-01-07 04:00'))
        signals[0][below,0] = 80.
        path,logs,_ = self.run_case(fixture,signals)
        self.assertEqual(len(logs['exits']),1)
        sold = ny('2026-01-06 18:00').isoformat()
        self.assertEqual(logs['exits'][0]['event'],sold)
        i = path.days.index(sold)
        self.assertEqual(path.weights[i],{'B':.4})
        self.assertAlmostEqual(path.weights[i]['B'],path.weights[i-1]['B'])
        self.assertAlmostEqual(1.-sum(path.weights[i].values()),.6)
        a_buys = [e['date'] for e in path.events if e['side']=='buy' and e['key']=='A']
        self.assertEqual(a_buys,[ny('2026-01-02 09:30').isoformat(),ny('2026-01-12 09:30').isoformat()])

    def test_absent_open_is_a_nav_mark_only_and_exit_waits_for_next_real_quote(self):
        fixture = self.fixture(missing_raw=('A','2026-01-06 18:00'))
        _,book,meta,okay = fixture
        signals = self.signals(meta,book.opn.columns)
        below = (meta.time >= ny('2026-01-06 12:00'))&(meta.time < ny('2026-01-07 04:00'))
        signals[0][below,0] = 80.
        k = int(np.flatnonzero(meta.time.eq(ny('2026-01-06 18:00')))[0])
        self.assertEqual(float(book.opn.iloc[k,0]),100.)
        self.assertFalse(okay[k,0])
        path,logs,_ = self.run_case(fixture,signals)
        self.assertEqual(len(logs['exits']),1)
        self.assertEqual(pd.Timestamp(logs['exits'][0]['event']),ny('2026-01-06 18:30'))
        self.assertEqual(logs['exits'][0]['price'],100.)
        self.assertIn('A',path.weights[path.days.index(ny('2026-01-06 18:00').isoformat())])

    def test_weekly_exit_clears_buffer_before_entry_gate_and_waits_until_next_week(self):
        fixture = self.fixture()
        _,book,meta,_ = fixture
        signals = self.signals(meta,book.opn.columns)
        below = (meta.time >= ny('2026-01-05 09:30'))&(meta.time < ny('2026-01-05 18:00'))
        signals[0][below,0] = 80.
        path,logs,calls = self.run_case(fixture,signals)
        self.assertEqual(dict(calls)['2026-01-05'],set())
        self.assertEqual(path.rebalances[ny('2026-01-05 09:30').isoformat()],{})
        denied = [e for e in logs['entries'] if e['session']=='2026-01-05']
        self.assertEqual(len(denied),1)
        self.assertFalse(denied[0]['allowed'])
        self.assertEqual([e['date'] for e in path.events if e['side']=='buy'],
                         [ny('2026-01-02 09:30').isoformat(),ny('2026-01-12 09:30').isoformat()])

    def test_mutating_a_forming_bar_cannot_change_earlier_weekly_entry(self):
        fixture = self.fixture(two_stocks=True)
        daily,book,meta,okay = fixture
        closes_index = pd.DatetimeIndex([ny('2025-12-31 20:00'),ny('2026-01-02 08:00'),
            ny('2026-01-02 12:00'),ny('2026-01-02 16:00'),ny('2026-01-02 20:00'),
            ny('2026-01-05 08:00'),ny('2026-01-05 12:00'),ny('2026-01-05 16:00')])
        closes = pd.DataFrame(100.,index=closes_index,columns=book.opn.columns)
        changed = closes.copy()
        changed.loc[ny('2026-01-05 12:00'),'B'] = 1.
        chooser = lambda d,c: {'A':.4} if d=='2026-01-02' else {'A':.4,'B':.4}
        original,_,_ = self.run_case(fixture,self.I.signal_matrix(closes,'SMA',2,meta.time),chooser=chooser)
        mutated,_,_ = self.run_case(fixture,self.I.signal_matrix(changed,'SMA',2,meta.time),chooser=chooser)
        cutoff = ny('2026-01-05 09:30').isoformat()
        self.assertEqual(original.rebalances[cutoff],mutated.rebalances[cutoff])
        self.assertIn('B',original.rebalances[cutoff])
        before = lambda p:[e for e in p.events if e['date']<=cutoff]
        self.assertEqual(before(original),before(mutated))

    def test_missing_weekly_open_cannot_be_filled_from_a_past_nav_mark(self):
        fixture = self.fixture(two_stocks=True,missing_weekly=('B','2026-01-05'))
        _,book,meta,_ = fixture
        signals = self.signals(meta,book.opn.columns)
        chooser = lambda d,c: {'A':.4} if d=='2026-01-02' else {'A':.4,'B':.4}
        with self.assertRaises(ValueError):
            self.run_case(fixture,signals,chooser=chooser)


if __name__ == '__main__':
    unittest.main()
