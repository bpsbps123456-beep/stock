"""Independent boundaries and information-time checks for the intraday MA study."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT/'data/derived/analysis/ma_entry_filter_20261108/intraday_review.py'
NY = 'America/New_York'


def ny(value):
    return pd.Timestamp(value, tz=NY)


class IntradayBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('ma_intraday_test_subject', MODULE)
        cls.I = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.I)

    def test_session_times_use_new_york_dst_and_early_close_calendar(self):
        dates = ['2025-07-03', '2026-03-06', '2026-03-09', '2026-11-02']
        regular = self.I.session_windows(dates, 'regular')
        extended = self.I.session_windows(dates, 'extended')
        for day in dates:
            self.assertEqual(regular.loc[day, 'open'], ny(day+' 09:30'))
            self.assertEqual(extended.loc[day, 'open'], ny(day+' 04:00'))
        self.assertEqual(regular.loc['2025-07-03', 'close'], ny('2025-07-03 13:00'))
        self.assertEqual(extended.loc['2025-07-03', 'close'], ny('2025-07-03 17:00'))
        self.assertEqual(regular.loc['2026-03-06', 'open'].tz_convert('Asia/Seoul').hour, 23)
        self.assertEqual(regular.loc['2026-03-09', 'open'].tz_convert('Asia/Seoul').hour, 22)
        self.assertEqual(regular.loc['2026-11-02', 'open'].tz_convert('Asia/Seoul').hour, 23)

    def test_awake_includes_eight_and_excludes_midnight(self):
        for date in ['2026-01-07', '2026-07-07']:
            for clock, expected in [('00:00', False), ('07:59', False),
                                    ('08:00', True), ('23:59', True)]:
                stamp = pd.Timestamp(date+' '+clock, tz='Asia/Seoul')
                self.assertEqual(bool(self.I.awake(stamp.tz_convert(NY))), expected)

    def test_regular_bars_reset_daily_and_seven_to_sixteen_hours_are_aliases(self):
        windows = self.I.session_windows(['2026-01-05', '2026-01-06'], 'regular')
        bars = self.I.bar_windows(windows, 4)
        expected_start = pd.DatetimeIndex([ny('2026-01-05 09:30'), ny('2026-01-05 13:30'),
                                          ny('2026-01-06 09:30'), ny('2026-01-06 13:30')])
        expected_end = pd.DatetimeIndex([ny('2026-01-05 13:30'), ny('2026-01-05 16:00'),
                                        ny('2026-01-06 13:30'), ny('2026-01-06 16:00')])
        np.testing.assert_array_equal(pd.DatetimeIndex(bars.start).as_unit('ns').asi8, expected_start.as_unit('ns').asi8)
        np.testing.assert_array_equal(pd.DatetimeIndex(bars.end).as_unit('ns').asi8, expected_end.as_unit('ns').asi8)
        np.testing.assert_allclose(np.asarray(bars.u_end)-np.asarray(bars.u_start), [4., 2.5, 4., 2.5])
        # No overnight trading time is inserted into the session clock.
        self.assertAlmostEqual(float(bars.u_start.iloc[2]-bars.u_end.iloc[1]), 0.)
        daily = self.I.bar_windows(windows, 7)
        for hours in range(8, 17):
            pd.testing.assert_frame_equal(self.I.bar_windows(windows, hours), daily)

    def test_extended_remainder_bar_and_half_day_are_not_full_nominal_hours(self):
        windows = self.I.session_windows(['2026-01-05'], 'extended')
        for hours, expected_clocks, expected_lengths in [
            (5, ['09:00', '14:00', '19:00', '20:00'], [5., 5., 5., 1.]),
            (7, ['11:00', '18:00', '20:00'], [7., 7., 2.]),
            (16, ['20:00'], [16.]),
        ]:
            bars = self.I.bar_windows(windows, hours)
            self.assertEqual(pd.DatetimeIndex(bars.end).strftime('%H:%M').tolist(), expected_clocks)
            np.testing.assert_allclose(np.asarray(bars.u_end)-np.asarray(bars.u_start), expected_lengths)
        half_day = self.I.bar_windows(self.I.session_windows(['2025-07-03'], 'regular'), 4)
        self.assertEqual(len(half_day), 1)
        self.assertAlmostEqual(float(half_day.u_end.iloc[0]-half_day.u_start.iloc[0]), 3.5)

    def test_aggregation_uses_only_half_open_window_and_never_fills_an_empty_bar(self):
        idx = pd.date_range(ny('2026-01-05 09:00'), ny('2026-01-05 16:00'), freq='30min')
        values = np.r_[999., np.arange(1., 14.), 999.]
        raw = pd.DataFrame({name: values.copy() for name in ['Open', 'High', 'Low', 'Close']}, index=idx)
        raw['Volume'] = 1.
        windows = self.I.session_windows(['2026-01-05'], 'regular')
        close, _ = self.I.aggregate_closes({'TEST': raw}, windows, 4)
        ends = pd.DatetimeIndex([ny('2026-01-05 13:30'), ny('2026-01-05 16:00')])
        np.testing.assert_array_equal(close.reindex(ends)['TEST'].to_numpy(), [8., 13.])
        empty_first = raw[(raw.index < ny('2026-01-05 09:30')) | (raw.index >= ny('2026-01-05 13:30'))]
        close, _ = self.I.aggregate_closes({'TEST': empty_first}, windows, 4)
        self.assertTrue(pd.isna(close.reindex(ends).iloc[0]['TEST']))
        self.assertEqual(float(close.reindex(ends).iloc[1]['TEST']), 13.)

    def test_asof_uses_completed_bars_with_inclusive_end_and_rejects_future_information(self):
        idx = pd.DatetimeIndex([ny('2026-01-05 13:30'), ny('2026-01-05 16:00'),
                                ny('2026-01-06 13:30'), ny('2026-01-06 16:00')])
        close = pd.DataFrame({'TEST': [10., 8., 20., 999.]}, index=idx)
        events = pd.DatetimeIndex([ny('2026-01-05 13:29'), ny('2026-01-05 13:30'),
                                   ny('2026-01-05 15:59'), ny('2026-01-05 16:00'),
                                   ny('2026-01-06 09:30')]).as_unit('ns')
        c, a, ends = self.I.signal_matrix(close, 'SMA', 2, events)
        np.testing.assert_allclose(np.asarray(c)[:, 0], [np.nan, 10., 10., 8., 8.], equal_nan=True)
        np.testing.assert_allclose(np.asarray(a)[:, 0], [np.nan, np.nan, np.nan, 9., 9.], equal_nan=True)
        self.assertEqual(ends[0, 0], np.iinfo(np.int64).min)
        self.assertEqual(ends[1, 0], idx[0].value)
        self.assertEqual(ends[-1, 0], idx[1].value)
        future = close.copy()
        future.iloc[2:, 0] = [-1e8, 1e9]
        fc, fa, fe = self.I.signal_matrix(future, 'SMA', 2, events)
        np.testing.assert_allclose(c, fc, equal_nan=True)
        np.testing.assert_allclose(a, fa, equal_nan=True)
        np.testing.assert_array_equal(ends, fe)

    def test_ema_seed_is_first_valid_close_and_one_bar_average_is_identity(self):
        idx = pd.date_range(ny('2026-01-05 13:00'), periods=4, freq='h')
        close = pd.DataFrame({'TEST': [10., 8., 12., 14.]}, index=idx)
        c, a, _ = self.I.signal_matrix(close, 'EMA', 3, idx)
        np.testing.assert_allclose(np.asarray(a)[:, 0], [np.nan, np.nan, 10.5, 12.25], equal_nan=True)
        for kind in ['SMA', 'EMA']:
            c, a, _ = self.I.signal_matrix(close, kind, 1, idx)
            np.testing.assert_array_equal(c, a)

    def test_empty_bar_retains_latest_real_bar_and_does_not_count_as_an_observation(self):
        idx = pd.date_range(ny('2026-01-05 13:00'), periods=4, freq='h')
        close = pd.DataFrame({'TEST': [10., np.nan, 8., 12.]}, index=idx)
        c, a, stamps = self.I.signal_matrix(close, 'EMA', 3, idx)
        np.testing.assert_allclose(np.asarray(c)[:, 0], [10., 10., 8., 12.], equal_nan=True)
        np.testing.assert_allclose(np.asarray(a)[:, 0], [np.nan, np.nan, np.nan, 10.5], equal_nan=True)
        self.assertEqual(stamps[1, 0], idx[0].value)
        _, sma, _ = self.I.signal_matrix(close, 'SMA', 3, idx)
        np.testing.assert_allclose(np.asarray(sma)[:, 0], [np.nan, np.nan, np.nan, 10.], equal_nan=True)

    def test_sma_timing_uses_close_samples_and_session_hours_not_wall_clock(self):
        meta = self.I.bar_windows(self.I.session_windows(['2026-01-05', '2026-01-06'], 'regular'), 4)
        stats = self.I.timing_stats(meta, 'SMA', 2)
        self.assertAlmostEqual(float(stats['mean_lag_hours_min']), 1.25)
        self.assertAlmostEqual(float(stats['mean_lag_hours_max']), 2.)
        self.assertAlmostEqual(float(stats['mean_lag_hours_median']), 1.25)
        self.assertAlmostEqual(float(stats['window_hours_median']), 6.5)
        # A one-bar average has zero sample lag despite spanning a positive-length bar.
        one = self.I.timing_stats(meta, 'SMA', 1)
        self.assertEqual(float(one['mean_lag_hours_max']), 0.)
        self.assertAlmostEqual(float(one['window_hours_median']), 3.25)

    def test_ema_timing_has_the_same_seed_and_weights_as_ema_price(self):
        meta = self.I.bar_windows(self.I.session_windows(['2026-01-05', '2026-01-06'], 'regular'), 4)
        stats = self.I.timing_stats(meta, 'EMA', 3)
        # u_close = [4, 6.5, 10.5, 13]; alpha = 1/2, first-value seed.
        # EMA(u) = [4, 5.25, 7.875, 10.4375], valid from observation 3.
        self.assertAlmostEqual(float(stats['mean_lag_hours_min']), 2.5625)
        self.assertAlmostEqual(float(stats['mean_lag_hours_max']), 2.625)
        self.assertAlmostEqual(float(stats['mean_lag_hours_median']), 2.59375)
        self.assertAlmostEqual(float(stats['window_hours_median']), 9.75)
        for kind in ['SMA', 'EMA']:
            one = self.I.timing_stats(meta, kind, 1)
            self.assertAlmostEqual(float(one['mean_lag_hours_median']), 0.)

    def test_actual_bar_timing_skips_empty_candle_without_compressing_market_clock(self):
        meta = self.I.bar_windows(self.I.session_windows(['2026-01-05', '2026-01-06'], 'regular'), 4)
        closes = pd.DataFrame({'TEST': [100., np.nan, 100., 100.]}, index=meta.index)
        stats = self.I.actual_timing_stats(closes, meta, 'SMA', 2)
        self.assertAlmostEqual(float(stats['mean_lag_hours_min']), 1.25)
        self.assertAlmostEqual(float(stats['mean_lag_hours_max']), 3.25)
        self.assertAlmostEqual(float(stats['mean_lag_hours_median']), 2.25)
        self.assertAlmostEqual(float(stats['window_hours_median']), 8.5)
        ema = self.I.actual_timing_stats(closes, meta, 'EMA', 3)
        self.assertAlmostEqual(float(ema['mean_lag_hours_median']), 2.875)
        self.assertAlmostEqual(float(ema['window_hours_median']), 13.)

    def test_event_quotes_keep_missing_fills_false_and_use_frozen_daily_anchors(self):
        days = ['2025-12-31', '2026-01-02', '2026-01-05']
        close = pd.DataFrame({'TEST': [90., 110., 130.]}, index=days)
        opening = pd.DataFrame({'TEST': [80., 100., 120.]}, index=days)
        daily = SimpleNamespace(days=days, cls=close, opn=opening, adj=close.copy(),
                                fx=np.ones(3), ticker={'test': 'TEST'}, pdfs=[])
        indices = [pd.date_range(ny(day+' 04:00'), ny(day+' 19:30'), freq='30min') for day in days]
        idx = indices[0].append(indices[1:])
        raw = pd.DataFrame({'Open': np.arange(len(idx), dtype=float)+1000.,
                            'Close': np.arange(len(idx), dtype=float)+1000.}, index=idx)
        missing = ny('2026-01-05 08:00')
        raw = raw.drop(missing)
        book, meta, trade_ok = self.I.event_book(daily, {'TEST': raw}, 'extended')
        at = {stamp: k for k, stamp in enumerate(pd.DatetimeIndex(meta.time))}
        self.assertFalse(trade_ok[at[missing], 0])
        self.assertTrue(trade_ok[at[missing+pd.Timedelta(minutes=30)], 0])
        self.assertEqual(float(book.opn.iloc[at[missing], 0]),
                         float(raw.loc[missing-pd.Timedelta(minutes=30), 'Open']))
        for day, o, c in [('2026-01-02', 100., 110.), ('2026-01-05', 120., 130.)]:
            self.assertEqual(float(book.opn.iloc[at[ny(day+' 09:30')], 0]), o)
            self.assertEqual(float(book.opn.iloc[at[ny(day+' 16:00')], 0]), c)
            self.assertFalse(trade_ok[at[ny(day+' 16:00')], 0])

    def test_event_clock_preserves_weekly_open_and_entire_session_duration(self):
        days = ['2025-12-31', '2026-01-02', '2026-01-05']
        prices = pd.DataFrame({'TEST': [100., 100., 100.]}, index=days)
        daily = SimpleNamespace(days=days, cls=prices, opn=prices, adj=prices,
                                fx=np.ones(3), ticker={'test': 'TEST'}, pdfs=[])
        idx = pd.date_range(ny('2025-12-31 04:00'), ny('2026-01-05 20:00'), freq='30min')
        raw = pd.DataFrame({'Open': np.full(len(idx), 100.), 'Close': np.full(len(idx), 100.)}, index=idx)
        for mode, duration in [('regular', 6.5), ('extended', 16.)]:
            _, meta, _ = self.I.event_book(daily, {'TEST': raw}, mode)
            evaluated = meta[meta.session >= '2026-01-02']
            self.assertEqual(evaluated.loc[evaluated.weekly_open, 'time'].tolist(),
                             [ny('2026-01-02 09:30'), ny('2026-01-05 09:30')])
            for actual in evaluated.groupby('session').duration_hours.sum():
                self.assertAlmostEqual(float(actual), duration)
            if mode == 'extended':
                before = meta[(meta.session == '2026-01-05') & (meta.time < ny('2026-01-05 09:30'))]
                self.assertTrue((before.week == self.I.S.iso_week('2026-01-02')).all())


if __name__ == '__main__':
    unittest.main()
