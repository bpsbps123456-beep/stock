"""Protect the frozen 30-minute input clock, provider contract and missing bars."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'ma_intraday_inputs',
    ROOT / 'data/derived/analysis/ma_entry_filter_20261108/intraday_inputs.py')
I = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(I)


def bar(local_timestamp, **changes):
    row = {'t': pd.Timestamp(local_timestamp, tz=I.ET).tz_convert('UTC').isoformat(),
           'o': 100., 'h': 102., 'l': 98., 'c': 100.,
           'v': 10, 'n': 2, 'vw': 100.}
    row.update(changes)
    return row


def response(payload):
    result = Mock(status_code=200)
    result.json.return_value = payload
    return result


def book_for(days, *, open_price=100., close_price=100.):
    return SimpleNamespace(
        days=days,
        opn={'TEST': pd.Series(open_price, index=days)},
        cls={'TEST': pd.Series(close_price, index=days)})


class NormaliseIntradayBarsTests(unittest.TestCase):
    def test_new_york_dst_and_start_label_boundaries(self):
        rows = [bar(f'{day} {clock}')
                for day in ['2025-03-07', '2025-03-10']
                for clock in ['03:30', '04:00', '09:00', '09:30', '19:30', '20:00']]
        rows.extend([bar('2025-03-06 12:00'), bar('2025-03-11 12:00')])
        result = I.normalise_bars(list(reversed(rows)), '2025-03-07', '2025-03-10')

        self.assertEqual(str(result.index.tz), I.ET)
        self.assertEqual(result.index.name, 'timestamp')
        self.assertEqual(list(result.columns), I.COLUMNS)
        self.assertEqual(result.index.strftime('%Y-%m-%d %H:%M').tolist(),
                         [f'{day} {clock}'
                          for day in ['2025-03-07', '2025-03-10']
                          for clock in ['04:00', '09:00', '09:30', '19:30']])
        self.assertEqual(result.index[0].tz_convert('UTC').isoformat(),
                         '2025-03-07T09:00:00+00:00')
        self.assertEqual(result.index[4].tz_convert('UTC').isoformat(),
                         '2025-03-10T08:00:00+00:00')
        self.assertEqual(result.iloc[0].to_dict(),
                         {'Open': 100., 'High': 102., 'Low': 98., 'Close': 100.,
                          'Volume': 10., 'Trades': 2., 'VWAP': 100.})

    def test_empty_rows_preserve_timezone_and_schema(self):
        result = I.normalise_bars([])
        self.assertTrue(result.empty)
        self.assertEqual(list(result.columns), I.COLUMNS)
        self.assertEqual(str(result.index.tz), I.ET)
        self.assertEqual(result.index.name, 'timestamp')

    def test_duplicate_timestamps_are_rejected(self):
        row = bar('2025-11-28 09:30')
        with self.assertRaisesRegex(ValueError, 'Duplicate provider timestamps'):
            I.normalise_bars([row, dict(row)], '2025-11-28', '2025-11-28')

    def test_invalid_prices_envelopes_and_volume_are_rejected(self):
        cases = [({'o': np.nan}, 'Non-finite/non-positive OHLC'),
                 ({'h': np.inf}, 'Non-finite/non-positive OHLC'),
                 ({'l': 0.}, 'Non-finite/non-positive OHLC'),
                 ({'c': -1.}, 'Non-finite/non-positive OHLC'),
                 ({'h': 99.}, 'High below another OHLC field'),
                 ({'l': 101.}, 'Low above another OHLC field'),
                 ({'v': -1}, 'Negative volume')]
        for changes, error in cases:
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(ValueError, error):
                    I.normalise_bars([bar('2025-11-28 09:30', **changes)],
                                     '2025-11-28', '2025-11-28')

    def test_unaligned_bar_starts_are_rejected(self):
        for clock in ['09:31', '09:30:01']:
            with self.subTest(clock=clock):
                with self.assertRaisesRegex(ValueError, 'not aligned to :00/:30'):
                    I.normalise_bars([bar(f'2025-11-28 {clock}')],
                                     '2025-11-28', '2025-11-28')


class FetchIntradayBarsTests(unittest.TestCase):
    def test_pagination_requests_split_sip_30min_and_exclusive_end(self):
        pages = iter([
            response({'bars': {'TEST': [bar('2025-03-07 09:30')]},
                      'next_page_token': 'page-two'}),
            response({'bars': {'TEST': [bar('2025-03-10 09:30'),
                                       bar('2025-03-11 04:00')]},
                      'next_page_token': None})])
        requests_seen = []

        def get(url, *, headers, params, timeout):
            # The implementation reuses its params dict; retain each request's values.
            requests_seen.append((url, headers, dict(params), timeout))
            return next(pages)

        pace = Mock()
        headers = {'test-header': 'test-value'}
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(I.requests, 'get', side_effect=get) as request:
                result = I.fetch_one('TEST', headers, pace, begin='2025-03-07',
                                     end='2025-03-10', cache=directory)
            stored = I.load_bars('TEST', directory)
            self.assertEqual(request.call_count, 2)
            self.assertFalse((Path(directory) / 'TEST.tmp.parquet').exists())

        self.assertEqual(result, {'ticker': 'TEST', 'fetch_status': 'downloaded', 'error': ''})
        self.assertEqual(pace.wait.call_count, 2)
        first = requests_seen[0]
        self.assertEqual(first[0], I.A.URL)
        self.assertIs(first[1], headers)
        self.assertEqual(first[3], 45)
        self.assertEqual(first[2],
                         {'symbols': 'TEST', 'timeframe': '30Min',
                          'start': '2025-03-07T05:00:00+00:00',
                          'end': '2025-03-11T04:00:00+00:00',
                          'limit': 10000, 'adjustment': 'split',
                          'feed': 'sip', 'sort': 'asc'})
        self.assertEqual(requests_seen[1][2], {**first[2], 'page_token': 'page-two'})
        self.assertEqual(stored.index.strftime('%Y-%m-%d %H:%M').tolist(),
                         ['2025-03-07 09:30', '2025-03-10 09:30'])

    def test_end_on_dst_transition_is_following_new_york_midnight(self):
        for day, exclusive_end in [('2025-03-09', '2025-03-10T04:00:00+00:00'),
                                   ('2025-11-02', '2025-11-03T05:00:00+00:00')]:
            with self.subTest(day=day), tempfile.TemporaryDirectory() as directory:
                with patch.object(I.requests, 'get',
                                  return_value=response({'bars': {}, 'next_page_token': None})) as request:
                    I.fetch_one('TEST', {}, Mock(), begin=day, end=day, cache=directory)
                params = request.call_args.kwargs['params']
                self.assertEqual(params['end'], exclusive_end)
                self.assertEqual(pd.Timestamp(params['end']).tz_convert(I.ET).hour, 0)

    def test_cached_file_skips_network_and_pacing(self):
        pace = Mock()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'TEST.parquet'
            path.write_bytes(b'existing frozen cache')
            with patch.object(I.requests, 'get') as request:
                result = I.fetch_one('TEST', {}, pace, cache=directory)
            self.assertEqual(path.read_bytes(), b'existing frozen cache')
        self.assertEqual(result['fetch_status'], 'cached')
        request.assert_not_called()
        pace.wait.assert_not_called()

    def test_repeated_pagination_token_fails_without_publishing_partial_cache(self):
        payload = {'bars': {'TEST': [bar('2025-11-28 09:30')]},
                   'next_page_token': 'same-token'}
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(I.requests, 'get',
                              side_effect=[response(payload), response(payload)]) as request:
                with self.assertRaisesRegex(RuntimeError, 'pagination token repeated'):
                    I.fetch_one('TEST', {}, Mock(), begin='2025-11-28',
                                end='2025-11-28', cache=directory)
            self.assertEqual(list(Path(directory).iterdir()), [])
        self.assertEqual(request.call_count, 2)


class IntradayCoverageTests(unittest.TestCase):
    def test_xnys_early_close_has_seven_slots_and_0930_open_boundary(self):
        day = '2025-11-28'  # XNYS closes at 13:00 after Thanksgiving.
        rows = [bar(f'{day} 09:00', o=50., l=49.),
                *[bar(f'{day} {clock}') for clock in
                  ['09:30', '10:00', '10:30', '11:00', '11:30', '12:00', '12:30']],
                bar(f'{day} 13:00', c=150., h=151.)]
        bars = I.normalise_bars(rows, day, day)
        with tempfile.TemporaryDirectory() as directory:
            bars.to_parquet(Path(directory) / 'TEST.parquet')
            result = I.coverage_row('TEST', book_for([day]),
                                    {'TEST': {'weekly_top12_n': 3}},
                                    cache=directory, begin=day, end=day)

        self.assertEqual(result['data_status'], 'available')
        self.assertEqual(result['rows'], 9)
        self.assertEqual(result['weekly_top12_n'], 3)
        self.assertEqual(result['expected_anchor_sessions'], 1)
        self.assertEqual(result['observed_rth_anchor_sessions'], 1)
        self.assertEqual(result['missing_rth_slots'], 0)
        self.assertEqual(result['missing_rth_days_n'], 0)
        self.assertEqual(result['daily_open_abs_relerr_max'], 0.)
        self.assertEqual(result['daily_close_abs_relerr_max'], 0.)
        self.assertEqual(result['close_over_1pct_days_n'], 0)

    def test_absent_no_trade_slots_and_entire_session_stay_missing(self):
        days = ['2025-11-28', '2025-12-01']
        bars = I.normalise_bars([bar('2025-11-28 09:30'),
                                 bar('2025-11-28 12:30')], days[0], days[-1])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'TEST.parquet'
            bars.to_parquet(path)
            original_bytes = path.read_bytes()
            result = I.coverage_row('TEST', book_for(days), {}, cache=directory,
                                    begin=days[0], end=days[-1])
            self.assertEqual(path.read_bytes(), original_bytes)
            pd.testing.assert_frame_equal(I.load_bars('TEST', directory), bars)

        self.assertEqual(result['rows'], 2)
        self.assertEqual(result['expected_anchor_sessions'], 2)
        self.assertEqual(result['observed_rth_anchor_sessions'], 1)
        self.assertEqual(result['missing_rth_slots'], 5 + 13)
        self.assertEqual(result['missing_rth_days_n'], 1)
        self.assertEqual(result['missing_rth_days'], '2025-12-01')

    def test_missing_and_empty_files_have_distinct_statuses(self):
        day = '2025-11-28'
        book = book_for([day])
        with tempfile.TemporaryDirectory() as directory:
            missing = I.coverage_row('TEST', book, {}, cache=directory, begin=day, end=day)
            I.normalise_bars([]).to_parquet(Path(directory) / 'TEST.parquet')
            empty = I.coverage_row('TEST', book, {}, cache=directory, begin=day, end=day)
        self.assertEqual(missing['data_status'], 'missing')
        self.assertEqual(empty['data_status'], 'empty')
        self.assertEqual(empty['rows'], 0)
        self.assertEqual(empty['first_timestamp'], '')


if __name__ == '__main__':
    unittest.main()
