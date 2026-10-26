"""harness.py 단위 테스트: 이탈·묶기·청산·분류·특징 시점·판정 함수. 실행: py -m unittest test_harness (code 폴더)."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402


def box(n=60, lo=102.0, hi=108.0, mid=105.0):
    """TR = 6, L = lo, H = hi 인 평평한 박스."""
    return [np.full(n, mid), np.full(n, hi), np.full(n, lo), np.full(n, mid)]


def append(arrs, bars):
    o, h, l, c = arrs
    for bo, bh, bl, bc in bars:
        o, h, l, c = np.r_[o, bo], np.r_[h, bh], np.r_[l, bl], np.r_[c, bc]
    return [o, h, l, c]


def pad(arrs, k=25):
    return append(arrs, [(105, 108, 102, 105)] * k)


BREAK = (104, 105, 99, 100)  # t=60: 종가 100 < L=102, TR=6


class EventRules(unittest.TestCase):
    def test_box_levels_and_atr(self):
        o, h, l, c = pad(append(box(), [BREAK]))
        Ls, Hs, atr = H.box_levels(o, h, l, c, 20)
        self.assertEqual(Ls[60], 102)
        self.assertEqual(Hs[60], 108)
        self.assertAlmostEqual(atr[60], 6.0)
        ev, *_ = H.detect_events(o, h, l, c, 20)
        self.assertEqual(ev.tolist(), [60])

    def test_fake_reclaim_exit(self):
        o, h, l, c = pad(append(box(), [BREAK, (100, 103.5, 98, 103), (103, 106, 102.5, 105)]))
        r, why, xi, xt = H.short_outcome(o, c, 60, 102.0, 10)
        self.assertEqual((why, xi, xt), ("reclaim", 62, "open"))
        self.assertAlmostEqual(r, (100 - 103) / 100 - H.COST)
        self.assertEqual(H.classify(l, c, 60, 102.0, 6.0), "가짜")

    def test_true_time_exit(self):
        down = [(99, 100, 95, 96), (96, 96, 89, 90)] + [(90 - k, 90 - k, 88 - k, 89 - k) for k in range(8)]
        o, h, l, c = pad(append(box(), [BREAK] + down))
        r, why, xi, xt = H.short_outcome(o, c, 60, 102.0, 10)
        self.assertEqual((why, xi, xt), ("time", 70, "close"))
        self.assertAlmostEqual(r, (99 - c[70]) / 99 - H.COST)
        self.assertEqual(H.classify(l, c, 60, 102.0, 6.0), "진짜")
        r5 = H.short_outcome(o, c, 60, 102.0, 5)
        self.assertEqual(r5[2], 65)

    def test_middle_late_reclaim(self):
        side = [(99, 100, 97, 99)] * 6 + [(99, 103.5, 98.5, 103), (104, 106, 103, 105)]
        o, h, l, c = pad(append(box(), [BREAK] + side))
        r, why, xi, xt = H.short_outcome(o, c, 60, 102.0, 10)
        self.assertEqual((why, xi), ("reclaim", 68))
        self.assertAlmostEqual(r, (99 - 104) / 99 - H.COST)
        self.assertEqual(H.classify(l, c, 60, 102.0, 6.0), "중간")

    def test_middle_reclaim_after_deep_low(self):
        o, h, l, c = pad(append(box(), [BREAK, (99, 100, 95, 99), (99, 103.5, 98.5, 103)]))
        self.assertEqual(H.classify(l, c, 60, 102.0, 6.0), "중간")

    def test_true_touch_same_day_as_reclaim_counts_true(self):
        o, h, l, c = pad(append(box(), [BREAK, (99, 104, 89.5, 103)]))
        self.assertEqual(H.classify(l, c, 60, 102.0, 6.0), "진짜")

    def test_undetermined_without_data(self):
        o, h, l, c = append(box(), [BREAK, (99, 100, 97, 99), (99, 100, 97, 99)])
        self.assertIsNone(H.classify(l, c, 60, 102.0, 6.0))
        self.assertTrue(np.isnan(H.short_outcome(o, c, 60, 102.0, 10)[0]))

    def _chain(self, second_gap):
        # t=60 이탈(저가 99) → 61..64 L(=99) 위 → 65 재이탈(저가 97.5) → 그 뒤 L(=97.5) 위 → 65+gap 재이탈
        bars = [BREAK] + [(101, 103, 100.5, 101)] * 4 + [(100, 100.5, 97.5, 98)]
        bars += [(100, 101, 99.5, 100)] * (second_gap - 1) + [(99, 99.5, 95, 96)]
        o, h, l, c = pad(append(box(), bars))
        ev, brk, *_ = H.detect_events(o, h, l, c, 20)
        return ev, np.flatnonzero(brk)

    def test_dedup_chain_merges_within_10(self):
        ev, brk = self._chain(10)
        self.assertEqual(brk[:3].tolist(), [60, 65, 75])
        self.assertEqual(ev[:2].tolist(), [60])  # 65는 60 뒤 5일, 75는 65 뒤 10일 → 같은 사건

    def test_dedup_chain_new_after_11(self):
        ev, brk = self._chain(11)
        self.assertEqual(brk[:3].tolist(), [60, 65, 76])
        self.assertEqual(ev[:2].tolist(), [60, 76])

    def test_detection_matches_brute_force(self):
        rng = np.random.default_rng(1)
        for N in (20, 40):
            c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 600)))
            o = c * np.exp(rng.normal(0, 0.005, 600))
            h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.01, 600)))
            l = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.01, 600)))
            ev, brk, Ls, Hs, atr = H.detect_events(o, h, l, c, N)
            bf, last = [], -10 ** 9
            for i in range(len(c)):
                if i < N or i < 21:
                    continue
                Lb = l[i - N:i].min()
                tr = [max(h[j] - l[j], abs(h[j] - c[j - 1]), abs(l[j] - c[j - 1])) for j in range(i - 20, i)]
                self.assertAlmostEqual(atr[i], np.mean(tr))
                self.assertAlmostEqual(Ls[i], Lb)
                self.assertAlmostEqual(Hs[i], h[i - N:i].max())
                if c[i] < Lb:
                    if i - last > 10:
                        bf.append(i)
                    last = i
            self.assertEqual(ev.tolist(), bf)
            self.assertGreater(len(bf), 3)


def synth_panel(seed=0, T=420):
    rng = np.random.default_rng(seed)
    syms = ["NVDA", "AMD", "MU", "MSFT", "AAPL", "ORCL", "QQQ", "SOXX", "^VIX", "^VIX3M"]
    cal = pd.bdate_range("2024-01-01", periods=T)
    k = len(syms)
    mkt = np.cumsum(rng.normal(0, 0.012, T))
    C = 100 * np.exp(mkt[:, None] + np.cumsum(rng.normal(0, 0.02, (T, k)), 0))
    C[:, syms.index("^VIX")] = 18 + 4 * np.sin(np.arange(T) / 15) + rng.normal(0, 1, T)
    C[:, syms.index("^VIX3M")] = 20 + rng.normal(0, 0.5, T)
    O = C * np.exp(rng.normal(0, 0.006, (T, k)))
    Hh = np.maximum(O, C) * np.exp(np.abs(rng.normal(0, 0.01, (T, k))))
    Ll = np.minimum(O, C) * np.exp(-np.abs(rng.normal(0, 0.01, (T, k))))
    V = rng.uniform(1e6, 3e6, (T, k))
    return H.Panel(cal, syms, O, Hh, Ll, C, V)


EV_SYMS = ["NVDA", "AMD", "MU", "MSFT", "AAPL", "ORCL", "QQQ", "SOXX"]
ETF_SET = ["QQQ", "SOXX"]
INFO_COLS = ["L", "H", "close_t", "atr20"] + H.FEATURES_BASE + H.FEATURES_EXTRA


class NoLookahead(unittest.TestCase):
    def test_features_ignore_future(self):
        P = synth_panel()
        er = {"NVDA": np.array([100, 230, 300]), "MU": np.array([150, 280])}
        base = H.build_events(P, 20, earnings=er, event_syms=EV_SYMS, etfs=ETF_SET)
        self.assertGreater(len(base), 20)
        picks = base[(base["i"] > 250) & (base["i"] < 380)].drop_duplicates("i").head(6)
        self.assertGreater(len(picks), 2)
        rng = np.random.default_rng(9)
        for _, ev in picks.iterrows():
            i = int(ev["i"])
            Q = H.Panel(P.cal, P.syms, *(a.copy() for a in (P.O, P.H, P.L, P.C, P.V)))
            for a in (Q.O, Q.H, Q.L, Q.C, Q.V):
                a[i + 1:] *= rng.uniform(0.5, 1.5, a[i + 1:].shape)
            Q.H[i + 1:] = np.maximum(Q.H[i + 1:], np.maximum(Q.O[i + 1:], Q.C[i + 1:]))
            Q.L[i + 1:] = np.minimum(Q.L[i + 1:], np.minimum(Q.O[i + 1:], Q.C[i + 1:]))
            alt = H.build_events(Q, 20, earnings=er, event_syms=EV_SYMS, etfs=ETF_SET)
            a = base[base["i"] == i].set_index("sym")[INFO_COLS]
            b = alt[alt["i"] == i].set_index("sym")[INFO_COLS]
            self.assertEqual(sorted(a.index), sorted(b.index))
            pd.testing.assert_frame_equal(a.sort_index(), b.sort_index())
            # 결과는 미래를 쓰므로 바뀌어야 한다(진입이 t+1 시가라는 확인)
            r0 = base[base["i"] == i]["entry_open"].to_numpy()
            r1 = alt[alt["i"] == i]["entry_open"].to_numpy()
            self.assertFalse(np.allclose(r0, r1))

    def test_excess_group_mean_and_kinds(self):
        P = synth_panel(3)
        ev = H.build_events(P, 20, earnings={}, event_syms=EV_SYMS, etfs=ETF_SET)
        g = ev[ev["excess_ref"] == "group"]
        s = g.groupby(["t", "kind"])["excess_h10"].sum()
        self.assertTrue(np.allclose(s.to_numpy(), 0, atol=1e-12))
        rnd = ev[ev["excess_ref"].fillna("").str.startswith("random:")]
        for _, r in rnd.iterrows():
            ctrl = r["excess_ref"].split(":")[1]
            self.assertEqual(ctrl in ETF_SET, r["kind"] == "etf")
            self.assertNotEqual(ctrl, r["sym"])

    def test_earn_pm3_window(self):
        P = synth_panel(4)
        ev = H.build_events(P, 20, earnings={}, event_syms=EV_SYMS, etfs=ETF_SET)
        r = ev[ev["kind"] == "stock"].iloc[5]
        i, s = int(r["i"]), r["sym"]
        for d, want in ((3, 1.0), (-3, 1.0), (4, 0.0), (-4, 0.0)):
            e2 = H.build_events(P, 20, earnings={s: np.array([i + d])}, event_syms=EV_SYMS, etfs=ETF_SET)
            got = e2[(e2["sym"] == s) & (e2["i"] == i)]["earn_pm3"].iloc[0]
            self.assertEqual(got, want, d)


def synth_events(seed=0, plant=0.03):
    rng = np.random.default_rng(seed)
    dates = list(pd.bdate_range("2026-01-05", "2026-05-29")) + list(pd.bdate_range("2026-06-01", "2026-09-29"))
    rows = []
    for d in dates:
        if rng.random() < 0.5:
            continue
        mkt = rng.normal(0, 0.03)
        for k in range(rng.integers(1, 6)):
            x = rng.normal()
            mk = rng.random() < 0.4
            rows.append({"t": d, "N": 20, "kind": "stock", "sym": f"S{k}", "feat": x, "mfeat": float(mk),
                         "noise": rng.normal(), "episode": int(d.month),
                         "short_ret_h10": mkt + rng.normal(0, 0.02) + plant * (x > 0.43)})
    df = pd.DataFrame(rows)
    df["excess_h10"] = df["short_ret_h10"] - df.groupby("t")["short_ret_h10"].transform("mean")
    return df


class Judging(unittest.TestCase):
    def test_cluster_diff_matches_sandwich(self):
        rng = np.random.default_rng(2)
        n = 120
        g = np.sort(rng.integers(0, 30, n))
        g = pd.factorize(g)[0]
        x = (rng.random(n) < 0.4).astype(float)
        y = rng.normal(size=n) + 0.3 * x
        _, _, _, _, d, t = H.cluster_diff(x, y, g)
        X = np.c_[np.ones(n), x]
        b = np.linalg.solve(X.T @ X, X.T @ y)
        u = y - X @ b
        meat = sum(np.outer(X[g == k].T @ u[g == k], X[g == k].T @ u[g == k]) for k in np.unique(g))
        bread = np.linalg.inv(X.T @ X)
        G = len(np.unique(g))
        V = bread @ meat @ bread * G / (G - 1) * (n - 1) / (n - 2)
        self.assertAlmostEqual(d[0], b[1])
        self.assertAlmostEqual(t[0], b[1] / np.sqrt(V[1, 1]))

    def test_block_maps_keep_dates_and_strata(self):
        df = synth_events(1)
        df.loc[df.index[::3], "N"] = 40
        rng = np.random.default_rng(0)
        maps = H.block_maps(df["t"].to_numpy(), df["N"].to_numpy(), 20, rng)
        Nv, dv = df["N"].to_numpy(), df["t"].to_numpy()
        dconst = (pd.factorize(dv)[0] % 2).astype(float)
        for m in maps:
            self.assertTrue(np.all(Nv[m] == Nv))  # 층 유지
            lab = pd.Series(dconst[m]).groupby([dv, Nv]).nunique()
            self.assertTrue((lab == 1).all())  # 날짜 고정 표지는 날짜째 이동
            src_date = pd.Series(dv[m]).groupby([dv, Nv]).nunique()
            self.assertTrue((src_date == 1).all())  # 한 날짜(층)는 한 원천 날짜에서만 받는다
        self.assertFalse(all(np.array_equal(m, np.arange(len(df))) for m in maps))

    def test_tercile_thresholds_from_front_only(self):
        df = synth_events(2)
        c1 = dict(H.tercile_cells(df, ["feat"]))
        df2 = df.copy()
        back = H.in_period(df2, H.BACK)
        df2.loc[back, "feat"] += 5
        c2 = dict(H.tercile_cells(df2, ["feat"]))
        self.assertEqual(list(c1), list(c2))
        fr = H.in_period(df, H.FRONT)
        for k in c1:
            pd.testing.assert_series_equal(c1[k][fr], c2[k][fr])
        q2 = np.quantile(df.loc[fr, "feat"], 2 / 3)
        hi = [k for k in c1 if "상위" in k][0]
        self.assertTrue(((df["feat"] >= q2).astype(float) == c1[hi]).all())

    def test_planted_signal_passes_noise_does_not(self):
        df = synth_events(3, plant=0.04)
        cells = H.tercile_cells(df, ["feat", "noise"])
        tab, s = H.evaluate_cells(df, cells, top=2, nperm=100, nnull=20, seed=1)
        hi = tab[tab["cell"].str.contains(r"\|feat≥")].iloc[0]
        self.assertTrue(hi["selected"])
        self.assertTrue(hi["pass"])
        self.assertGreater(hi["b_t"], 2)
        self.assertLess(hi["b_p"], 0.05)
        self.assertLessEqual(s["null"]["mean"], 0.5)
        self.assertLess(s["null"]["p_value"], 0.2)
        self.assertIn("one_regime", tab)

    def test_no_signal_rarely_passes(self):
        df = synth_events(4, plant=0.0)
        cells = H.tercile_cells(df, ["feat", "noise", "mfeat"])
        tab, s = H.evaluate_cells(df, cells, top=3, nperm=100, nnull=0, null=False, seed=2)
        self.assertLessEqual(int(tab["pass"].sum()), 1)
        self.assertEqual(s["n_selected"], min(3, s["n_eligible"]))


class PermutationFixes(unittest.TestCase):
    """2026-10-14 감사(results/qa_harness.md)에서 고친 순열·p·실적일 처리."""

    def test_first_return_uniform_bijection(self):
        rng = np.random.default_rng(0)
        D = np.zeros(12, bool)
        D[[1, 4, 5, 8, 11]] = True
        hits = np.zeros(12)
        for _ in range(20000):
            f = H._first_return(rng.permutation(12), D)
            self.assertEqual(sorted(f[D]), [1, 4, 5, 8, 11])  # D 위 전단사
            self.assertTrue(np.all(f[~D] == -1))
            hits[f[1]] += 1
        frac = hits[D] / hits.sum()
        self.assertTrue(np.all(np.abs(frac - 0.2) < 0.02), frac)  # 균일

    def _two_strata(self, full=False):
        """30일. 20|stock 2건은 매일, 40|stock은 (k%3≠1)인 날, 40|etf는 짝수 날(full=True면 매일 4건)."""
        days = pd.bdate_range("2026-06-01", periods=30).to_numpy()
        d, s, lab = [], [], []
        for k, day in enumerate(days):
            put = ["20|stock", "20|stock"]
            if full or k % 3 != 1:
                put.append("40|stock")
            if full or k % 2 == 0:
                put.append("40|etf")
            d += [day] * len(put)
            s += put
            lab += [float(k % 3 == 0)] * len(put)
        return np.array(d), np.array(s), np.array(lab)[:, None]

    def test_date_constant_cell_moves_whole_date_across_strata(self):
        dates, strata, M = self._two_strata()
        dc = H._codes(dates)
        self.assertTrue(H.date_constant(M, dc)[0])
        P = H.Permuter(dates, strata, 50, np.random.default_rng(1))
        for r in range(len(P)):
            Mp = P.apply(M, r, np.array([True]))
            per_date = pd.Series(Mp[:, 0]).groupby(dc).nunique()
            self.assertTrue((per_date == 1).all())  # 층(N·kind)을 가로질러 같은 날 같은 표지
            self.assertEqual(sorted(pd.Series(Mp[:, 0]).groupby(dc).first()),
                             sorted(pd.Series(M[:, 0]).groupby(dc).first()))  # 날짜 표지 재배열
        # 층마다 날짜 집합이 다르면 행 지도만으로는 같은 날 표지가 갈릴 수 있다(그래서 날짜 고정 칸은 apply로 옮김)
        split = 0
        for r in range(len(P)):
            Mp = M[P.rowmaps[r]]
            split += int((pd.Series(Mp[:, 0]).groupby(dc).nunique() > 1).any())
        self.assertGreater(split, 0)

    def test_rowmaps_share_source_when_strata_share_dates(self):
        dates, strata, _ = self._two_strata(full=True)
        maps = H.block_maps(dates, strata, 30, np.random.default_rng(2))
        for m in maps:
            src = pd.DataFrame({"d": dates, "src": dates[m]}).groupby("d")["src"].nunique()
            self.assertTrue((src == 1).all())

    def test_perm_p_nan_observed_and_shift_count(self):
        df = synth_events(5)
        b = df[H.in_period(df, H.BACK)]
        y = b["short_ret_h10"].to_numpy()
        g = H._codes(b["t"].to_numpy())
        M = np.c_[np.ones(len(b)), (b["feat"] > 0).astype(float)]
        P = H.Permuter(b["t"].to_numpy(), b["N"].to_numpy(), 50, np.random.default_rng(3))
        _, _, _, _, d, _ = H.cluster_diff(M, y, g)
        p = H._perm_p(M, y, g, P, d, H.date_constant(M, g))
        self.assertTrue(np.isnan(d[0]) and np.isnan(p[0]))  # 꺼짐 0건 → p NaN(예전 1/51)
        self.assertTrue(0 < p[1] <= 1)
        S = H.Permuter(b["t"].to_numpy(), b["N"].to_numpy(), rng=np.random.default_rng(3), mode="shift")
        self.assertEqual(len(S), b["t"].nunique() - 1)

    def test_shift_rotates_date_sequence(self):
        dates = pd.bdate_range("2026-06-01", periods=20).to_numpy()
        lab = np.r_[np.ones(6), np.zeros(14)][:, None]
        S = H.Permuter(dates, np.zeros(20), rng=np.random.default_rng(0), mode="shift")
        for r in range(len(S)):
            Mp = S.apply(lab, r, np.array([True]))
            self.assertTrue(np.array_equal(Mp[:, 0], np.roll(lab[:, 0], -(r + 1))))

    def test_evaluate_cells_supplementary_columns(self):
        df = synth_events(6, plant=0.04)
        cells = H.tercile_cells(df, ["feat", "noise", "mfeat"])
        tab, s = H.evaluate_cells(df, cells, top=3, nperm=50, nnull=10, seed=3)
        self.assertTrue(set(["b_p_shift", "pass_shift", "date_const"]) <= set(tab.columns))
        self.assertTrue((~tab["pass_shift"] | tab["pass"]).all())
        self.assertIn("null_shift", s)
        self.assertEqual(len(s["null_shift"]["dist"]), 10)

    def test_earnings_before_calendar_dropped(self):
        import tempfile
        cal = pd.bdate_range("2024-01-02", periods=40)
        rows = "ticker,earnings_dt\nAAA,2023-11-01T16:00:00-04:00\nAAA,2024-01-10T16:00:00-05:00\n" \
               "AAA,2024-01-12T07:00:00-05:00\n"
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "e.csv"
            p.write_text(rows, encoding="utf-8")
            er = H.earnings_react_dates(cal, p)
        self.assertEqual(er["AAA"].tolist(), [cal.get_loc(pd.Timestamp("2024-01-11")),
                                              cal.get_loc(pd.Timestamp("2024-01-12"))])


if __name__ == "__main__":
    unittest.main()
