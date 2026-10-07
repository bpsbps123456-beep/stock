"""Network-free checks of research input causality, identity, FX and flow units."""

from __future__ import annotations

import unittest

import numpy as np

from gbrc.etf.research_data import (
    _ffill, _flow_view, _fx_arrays, _snapshot_arrays, _universe, exclusion_reason,
)


def snapshot(day, quantities, prices=None):
    prices = prices or {key: 10 for key in quantities}
    values = {key: qty * prices[key] for key, qty in quantities.items()}
    nav = sum(values.values()) + 100
    rows = [{'code': key, 'name': key, 'qty': qty, 'value_krw': values[key],
             'weight': values[key] / nav * 100, 'asset_type': 'security'}
            for key, qty in quantities.items()]
    rows.append({'code': '', 'name': 'Cash', 'asset_type': 'cash', 'weight': 100 / nav * 100,
                 'qty': 100, 'value_krw': 100})
    return {'as_of_date': day, 'holdings': rows}


BASE = {f'S{i} US EQUITY': 100 for i in range(6)}


class ResearchIdentityTests(unittest.TestCase):
    def test_exited_names_remain_and_aliases_collapse(self):
        first = {**BASE, 'GOOGL US EQUITY': 50, 'SKHYV US EQUITY': 10,
                 'SATS US EQUITY': 20}
        second = {**BASE, 'GOOG US EQUITY': 50, 'SKHY US EQUITY': 10,
                  'ECHO US EQUITY': 20}
        del second['S0 US EQUITY']
        funds = [{'2026-01-02': snapshot('2026-01-02', first),
                  '2026-01-05': snapshot('2026-01-05', second)}] * 2
        universe, excluded = _universe(funds)
        rows = {row['symbol']: row for row in universe}
        self.assertIn('S0', rows)
        self.assertNotIn('GOOGL', rows)
        self.assertEqual(rows['GOOG']['aliases'], ['GOOG US EQUITY', 'GOOGL US EQUITY'])
        self.assertEqual(rows['ECHO']['aliases'], ['ECHO US EQUITY', 'SATS US EQUITY'])
        self.assertEqual(rows['SKHY']['aliases'], ['SKHY US EQUITY', 'SKHYV US EQUITY'])
        self.assertEqual(len(excluded), 1)

    def test_product_name_filter_does_not_remove_ordinary_holdings(self):
        for code, name in [('0043Y0', 'TIME 차이나AI테크액티브'),
                           ('478150', 'TIME 글로벌우주테크&방산액티브'),
                           ('TQQQ US EQUITY', 'ProShares UltraPro QQQ'),
                           ('NQZ6 INDEX', 'NASDAQ 100 E-MINI INDEX')]:
            self.assertIsNotNone(exclusion_reason({'code': code, 'name': name}))
        self.assertIsNone(exclusion_reason({'code': 'SPCX US EQUITY',
                                           'name': 'Space Exploration Technologies Corp'}))

    def test_same_security_rename_precedes_quantity_flow_and_does_not_mutate(self):
        before = snapshot('2026-06-23', {**BASE, 'SATS US EQUITY': 100})
        view = _flow_view(before)
        self.assertEqual(before['holdings'][-2]['code'], 'SATS US EQUITY')
        self.assertEqual(view['holdings'][-2]['code'], 'ECHO US EQUITY')
        after = snapshot('2026-06-24', {**BASE, 'ECHO US EQUITY': 100})
        fund = {'2026-06-23': before, '2026-06-24': after}
        universe, _ = _universe([fund, fund])
        arrays, _ = _snapshot_arrays([fund, fund], sorted(fund), universe, sorted(fund))
        n = [item['symbol'] for item in universe].index('ECHO')
        self.assertEqual(arrays['flow_kind'][1, 0, 0, n], 0)
        self.assertEqual(arrays['flow_pp'][1, 0, 0, n], 0)


class ResearchFlowTests(unittest.TestCase):
    def test_short_windows_nan_creation_adjustment_and_pp_units(self):
        days = ['2026-01-02', '2026-01-05', '2026-01-06', '2026-01-07']
        after = {key: qty * 1.1 for key, qty in BASE.items()}
        after['S0 US EQUITY'] *= 1.3
        fund = {days[0]: snapshot(days[0], BASE)}
        fund.update({day: snapshot(day, after) for day in days[1:]})
        universe, _ = _universe([fund, fund])
        arrays, failures = _snapshot_arrays([fund, fund], days, universe, days)
        self.assertEqual(failures, [])
        self.assertTrue(np.isnan(arrays['flow_pp'][0]).all())
        self.assertTrue(np.isnan(arrays['flow_pp'][1, 1]).all())  # D-3 unavailable
        n = [item['symbol'] for item in universe].index('S0')
        self.assertEqual(arrays['flow_kind'][1, 0, 0, n], 1)
        weight_pp = arrays['weights'][1, 0, n] * 100
        self.assertAlmostEqual(arrays['flow_pp'][1, 0, 0, n], weight_pp * 0.3 / 1.3)
        self.assertEqual(arrays['flow_pp'][1, 0, 0, 1], 0)
        self.assertTrue(np.isfinite(arrays['flow_pp'][3, 1]).all())
        np.testing.assert_allclose(arrays['weights'].sum(axis=2) + arrays['cash'], 1)

    def test_missing_common_factor_invalidates_whole_window(self):
        quantities = {'S0 US EQUITY': 100, 'S1 US EQUITY': 100}
        days = ['2026-01-02', '2026-01-05']
        fund = {day: snapshot(day, quantities) for day in days}
        universe, _ = _universe([fund, fund])
        arrays, failures = _snapshot_arrays([fund, fund], days, universe, days)
        self.assertTrue(np.isnan(arrays['flow_kind'][1, 0]).all())
        self.assertEqual(len(failures), 2)

    def test_krx_session_missing_snapshot_does_not_create_short_window(self):
        sessions = ['2026-01-02', '2026-01-05', '2026-01-06', '2026-01-07']
        days = sessions[2:]
        fund = {day: snapshot(day, BASE) for day in days}
        universe, _ = _universe([fund, fund])
        arrays, _ = _snapshot_arrays([fund, fund], days, universe, sessions)
        self.assertTrue(np.isnan(arrays['flow_kind'][1, 1]).all())
        self.assertTrue(np.isfinite(arrays['flow_kind'][1, 0]).all())


class ResearchPriceTests(unittest.TestCase):
    def test_no_backfill_and_real_observation_mask(self):
        days = ['2026-01-02', '2026-01-05', '2026-01-06', '2026-01-12']
        values, observed, ages = _ffill({'2026-01-05': 10, '2026-01-06': 11}, days)
        self.assertTrue(np.isnan(values[0]))
        np.testing.assert_allclose(values[1:], [10, 11, 11])
        self.assertEqual(observed.tolist(), [False, True, True, False])
        self.assertEqual(ages[-1], 6)

    def test_observed_cross_currency_and_missing_fx_not_approximated(self):
        days = ['2026-01-02', '2026-01-05']
        fx = _fx_arrays({'KRW=X': {'2026-01-02': 1400, '2026-01-05': 1350},
                         'CNY=X': {'2026-01-05': 7.5},
                         'JPYKRW=X': {'2026-01-02': 9}}, days)
        self.assertTrue(np.isnan(fx['CNY'][0][0]))
        self.assertEqual(fx['CNY'][0][1], 180)
        self.assertEqual(fx['JPY'][1][1], 3)
        self.assertTrue(np.isnan(fx['HKD'][0]).all())


if __name__ == '__main__':
    unittest.main()
