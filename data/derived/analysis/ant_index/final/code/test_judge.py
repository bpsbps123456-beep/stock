"""judge.py 단위 테스트. 실제 가격을 읽지 않는다(load_prices('real')은 호출하지 않고 가드 함수와 소스 순서만 본다).

실행: py -m unittest test_judge   (이 폴더에서)
"""
from __future__ import annotations

import importlib.util
import inspect
import sys
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import judge as J  # noqa: E402


def _load_preview_test_free():
    """미리보기의 기준 구현(hac_cov·hodrick_cov). 모듈 최상단은 함수·상수 정의뿐이라 자료를 읽지 않는다."""
    sp = importlib.util.spec_from_file_location("preview_test_free", J.PREVIEW / "code" / "test_free.py")
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


class TestCorrections(unittest.TestCase):
    def test_holm(self):
        adj, thr = J.holm({"a": 0.01, "b": 0.04, "c": 0.03}, 0.05)
        self.assertAlmostEqual(adj["a"], 0.03)
        self.assertAlmostEqual(adj["c"], 0.06)
        self.assertAlmostEqual(adj["b"], 0.06)
        self.assertEqual(set(thr), {"a"})
        self.assertAlmostEqual(thr["a"], 0.05 / 3)

    def test_bh(self):
        adj, thr = J.bh({"a": 0.01, "b": 0.04, "c": 0.03}, 0.10)
        self.assertAlmostEqual(adj["a"], 0.03)
        self.assertAlmostEqual(adj["c"], 0.04)
        self.assertAlmostEqual(adj["b"], 0.04)
        self.assertEqual(set(thr), {"a", "b", "c"})
        adj1, thr1 = J.bh({"x": 0.2}, 0.10)
        self.assertAlmostEqual(adj1["x"], 0.2)
        self.assertEqual(thr1, {})


class TestSE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.TF = _load_preview_test_free()

    def test_fwl_hac_equals_full_sandwich(self):
        rng = np.random.default_rng(1)
        n = 120
        pos = np.sort(rng.choice(160, n, replace=False))
        C = np.column_stack([np.ones(n), rng.standard_normal((n, 3))])
        x = rng.standard_normal(n) + 0.5 * C[:, 1]
        y = 0.3 * x + C @ np.array([0.1, 0.2, -0.1, 0.0]) + rng.standard_normal(n)
        X = np.column_stack([C[:, :1], x, C[:, 1:]])
        beta, e = self.TF.ols(y, X)
        V = self.TF.hac_cov(X, e, pos, 5)
        Q = J._qr(C)
        r = J.fwl_hac(J._resid(Q, x[:, None]), J._resid(Q, y), pos, 5, X.shape[1])
        self.assertAlmostEqual(float(r["beta"][0]), float(beta[1]), places=10)
        self.assertAlmostEqual(float(r["se"][0]), float(np.sqrt(V[1, 1])), places=10)

    def test_fwl_hodrick_equals_reference(self):
        rng = np.random.default_rng(2)
        n, h = 90, 10
        pos = np.arange(5, 5 + n)
        eps_all = rng.standard_normal(5 + n + 20) * 0.01
        C = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
        z = rng.standard_normal(n)
        y = rng.standard_normal(n)
        X = np.column_stack([C[:, :1], z, C[:, 1:]])
        V = self.TF.hodrick_cov(X, pos, eps_all, h)
        Q = J._qr(C)
        r = J.fwl_hodrick(J._resid(Q, z[:, None]), J._resid(Q, y), pos, eps_all, h, X.shape[1])
        self.assertAlmostEqual(float(r["se"][0]), float(np.sqrt(V[1, 1])), places=10)

    def test_hac_vectorized_columns(self):
        rng = np.random.default_rng(3)
        n = 80
        pos = np.arange(n)
        C = np.column_stack([np.ones(n), rng.standard_normal(n)])
        M = rng.standard_normal((n, 4))
        y = rng.standard_normal(n)
        Q = J._qr(C)
        allc = J.fwl_hac(J._resid(Q, M), J._resid(Q, y), pos, 5, 3)
        for j in range(4):
            one = J.fwl_hac(J._resid(Q, M[:, [j]]), J._resid(Q, y), pos, 5, 3)
            self.assertAlmostEqual(float(allc["t"][j]), float(one["t"][0]), places=12)


class TestM0(unittest.TestCase):
    def _df(self, n=320, seed=4):
        rng = np.random.default_rng(seed)
        D = pd.bdate_range("2025-06-02", periods=n)
        df = pd.DataFrame({"D": D, "a": rng.standard_normal(n), "b": rng.standard_normal(n),
                           "d": (rng.random(n) < 0.2).astype(float), "e1": rng.standard_normal(n),
                           "e2": rng.standard_normal(n), "e3": rng.standard_normal(n), "e4": rng.standard_normal(n)})
        x = 0.8 * df["a"].to_numpy() + rng.standard_normal(n)
        return df, x

    def test_no_lookahead_and_min_fit(self):
        df, x = self._df()
        ok = np.ones(len(df), bool)
        r1 = J.expanding_m0(df, x[:, None], ["a", "b"], ["d"], [], ok, min_n=120)
        x2 = x.copy()
        last = pd.to_datetime(df["D"]).dt.to_period("M") == pd.to_datetime(df["D"]).dt.to_period("M").iloc[-1]
        x2[last.to_numpy()] += 100.0                      # 마지막 달 특징만 바꿈
        r2 = J.expanding_m0(df, x2[:, None], ["a", "b"], ["d"], [], ok, min_n=120)
        early = ~last.to_numpy()
        np.testing.assert_allclose(r1["xhat"][early], r2["xhat"][early], equal_nan=True)
        np.testing.assert_allclose(r1["xperp"][early], r2["xperp"][early], equal_nan=True)
        # 적합 창 120개 전 달은 결측
        Dt = pd.to_datetime(df["D"])
        for rec in r1["months"]:
            m = Dt.dt.to_period("M").astype(str) == rec["month"]
            if rec["n_fit"] < 120:
                self.assertTrue(np.isnan(r1["xperp"][m.to_numpy(), 0]).all())
            else:
                self.assertTrue(np.isfinite(r1["xperp"][m.to_numpy(), 0]).all())
                self.assertGreater(rec["R2_obs"], 0.2)

    def test_fit_uses_only_rows_before_month(self):
        df, x = self._df()
        ok = np.ones(len(df), bool)
        r = J.expanding_m0(df, x[:, None], ["a", "b"], [], [], ok, min_n=120)
        Dt = pd.to_datetime(df["D"])
        for rec in r["months"]:
            start = pd.Timestamp(rec["month"] + "-01")
            self.assertEqual(rec["n_fit"], int((Dt < start).sum()))

    def test_ridge_when_more_than_3_extras(self):
        df, x = self._df()
        ok = np.ones(len(df), bool)
        ex = [("e1", "price"), ("e2", "price"), ("e3", "price"), ("e4", "forum")]
        r = J.expanding_m0(df, x[:, None], ["a"], [], ex, ok, min_n=120)
        fitted = [m for m in r["months"] if "R2_obs" in m]
        self.assertTrue(fitted and all(m.get("ridge") for m in fitted))
        r3 = J.expanding_m0(df, x[:, None], ["a"], [], ex[:3], ok, min_n=120)
        self.assertFalse(any(m.get("ridge") for m in r3["months"]))


class TestRotation(unittest.TestCase):
    def test_rotate(self):
        x0 = np.array([1.0, np.nan, 2.0, 3.0, 4.0, np.nan, 5.0])
        F = np.array([0, 2, 3, 4, 6])
        X = J._rotate(x0, F, len(F))
        np.testing.assert_array_equal(X[:, 0], x0)
        self.assertTrue(np.isnan(X[[1, 5], :]).all())
        np.testing.assert_array_equal(X[F, 1], np.roll(x0[F], 1))
        self.assertEqual(sorted(X[F, 3]), sorted(x0[F]))

    def test_shift_p(self):
        N, m = 60, 5
        stat = np.linspace(-1, 1, N)[::-1].copy()     # stat[0]이 가장 큼
        p = J.shift_p_all(stat, +1, m)
        K = N - 2 * m + 1
        self.assertAlmostEqual(p[0], 1 / (1 + K))
        p_neg = J.shift_p_all(stat, -1, m)
        self.assertAlmostEqual(p_neg[0], 1.0)

    def test_n_blocks(self):
        self.assertEqual(J.n_blocks(np.arange(21), 10), 2)
        self.assertEqual(J.n_blocks(np.arange(0, 50, 5), 1), 10)
        self.assertEqual(J.n_blocks(np.arange(22), 10), 2)
        self.assertEqual(J.n_blocks(np.arange(23), 10), 3)


class _Fake:
    def __init__(self, kind="har", sign=+1, front_required=True):
        self.s = {"kind": kind, "sign": sign}
        self.front_required = front_required


class TestVerdict(unittest.TestCase):
    def v(self, **kw):
        base = {"est": 0.3, "se_ci": 0.1, "placebo_flag": False, "front_ok": True, "robust_ok": True, "comp_ok": True,
                "xp_p_final": 0.001, "xp_est": 0.3, "xp_front_ok": True, "xp_robust_ok": True}
        base.update(kw)
        return base

    def test_t1_placebo_needs_full_xperp_pass(self):
        """§2.3 위약: 'X⊥가 통과하지 못하면' 기각 — T1 X⊥판도 §2.7 통과(p·부호·앞 부호·강건)를 다 채워야 한다."""
        c = _Fake()
        self.assertEqual(J.verdict(c, self.v(placebo_flag=True, xp_front_ok=False), True, 0.05, 0.15)[0], "기각")
        self.assertEqual(J.verdict(c, self.v(placebo_flag=True, xp_robust_ok=False), True, 0.05, 0.15)[0], "기각")
        self.assertEqual(J.verdict(c, self.v(placebo_flag=True, xp_est=-0.1), True, 0.05, 0.15)[0], "기각")
        lab = _Fake(front_required=False)            # 라벨 층: 앞 구간은 보조
        self.assertEqual(J.verdict(lab, self.v(placebo_flag=True, xp_front_ok=False), True, 0.05, 0.15)[0], "통과")

    def test_categories(self):
        c = _Fake()
        self.assertEqual(J.verdict(c, self.v(), True, 0.05, 0.15)[0], "통과")
        self.assertEqual(J.verdict(c, self.v(robust_ok=False), True, 0.05, 0.15)[0], "취약")
        self.assertEqual(J.verdict(c, self.v(), False, 0.05, 0.15)[0], "판정 불가")
        self.assertEqual(J.verdict(c, self.v(front_ok=False), True, 0.05, 0.15)[0], "판정 불가")
        self.assertEqual(J.verdict(c, self.v(est=-0.01), True, 0.05, 0.15)[0], "기각")
        self.assertEqual(J.verdict(c, self.v(est=0.01, se_ci=0.01), False, 0.05, 0.15)[0], "기각")
        self.assertEqual(J.verdict(c, self.v(placebo_flag=True, xp_p_final=0.5), True, 0.05, 0.15)[0], "기각")
        self.assertEqual(J.verdict(c, self.v(placebo_flag=True), True, 0.05, 0.15)[0], "통과")
        self.assertEqual(J.verdict(c, self.v(comp_ok=False), True, 0.05, 0.15)[0], "판정 불가")
        lab = _Fake(front_required=False)
        self.assertEqual(J.verdict(lab, self.v(front_ok=False), True, 0.05, 0.15)[0], "통과")
        ic = _Fake(kind="ic", sign=-1)
        self.assertEqual(J.verdict(ic, self.v(est=-0.3), True, 0.10, 0.10)[0], "통과")
        self.assertEqual(J.verdict(ic, self.v(est=-0.3, placebo_flag=True), False, 0.10, 0.10)[0], "기각")


class TestSpec(unittest.TestCase):
    """동결 문서·부록과의 칸별 대조(2026-10-22)에서 고친 곳."""

    def test_us_m0_has_missing_rate_and_calendar(self):
        # §2.2 결번율은 모든 텍스트 가설(H31·H46·H50 포함)의 통제, §2.4 달력 더미
        self.assertIn("m0_missing_rate", J.M0_US_CONT)
        self.assertEqual(J.M0_US_DUM, J.M0_KR_DUM)
        for k in ("m0_cal_cpi", "m0_cal_jobs", "m0_cal_msci"):
            self.assertIn(k, J.M0_KR_DUM)
            self.assertIn(k, J.HAR_DUM)

    def test_direction_controls(self):
        # §2.3: 방향 판정 회귀 = 대상 r(D−1)·r5·r20·RV20·g(D) (+ 한국이면 전야 SOX) + 칸별 통제
        for c in J.CELLS:
            if c.get("kind") != "ic":
                continue
            ctrl = set(c["ctrl"])
            if c["market"] == "kr":
                self.assertTrue(set(J.KR_IC_CTRL) <= ctrl, c["id"])
            else:
                pre = c["y"].split("_")[1]          # fut_soxx_* → soxx, fut_mem3x_* → mem3x
                for k in ("r", "r5", "r20", "lnRV20", "g"):
                    self.assertIn(f"m0_{pre}_{k}", ctrl, c["id"])
        self.assertTrue({"m0_kospi_absr20", "m0_kospi_r20xRV20"} <= set(J.CELL["H36"]["ctrl"]))

    def test_cell_registry(self):
        self.assertEqual((J.CELL["H68"]["role"], J.CELL["H68"]["fam"]), ("main", "T2"))     # 부록 4
        self.assertIn(("m0_cal_earn_mem", "price"), J.CELL["H50"]["m0_extra"])              # §3 H50 메모리 일정 더미
        self.assertIn(("jul3", J.JUL3_DAYS), J.CELL["H03"]["rob_extra"])                    # §3 H03 7/29~31
        self.assertEqual(J.CELL["H52"]["ctrl"], [])
        self.assertEqual(J.CELL["H52"]["gap_pred"], J.H52_CTRL)
        stopped = J.STATUS_ONLY["멈춤(사전 측정 실패, §2.6·부록 2 #4)"]
        for k in ("H14", "H19", "H49", "H51"):
            self.assertIn(k, stopped)

    def test_every_registered_id_has_a_status(self):
        ids = {f"H{i:02d}" for i in range(1, 57)} | {"H57a", "H57b", "H57c", "H59"} | {f"H{i}" for i in range(60, 79)}
        ids -= {"H39", "H48", "H54", "H55"}
        ids |= {"H39+H48", "H54·H55"}
        covered = {c["id"] for c in J.CELLS} | set(J.CHECKLIST_ROWS)
        for d in J.STATUS_ONLY.values():
            covered |= set(d)
        self.assertEqual(sorted(ids - covered), [])

    def test_gap_resid_uses_only_earlier_mondays(self):
        rng = np.random.default_rng(9)
        n = 60
        df = pd.DataFrame({"Dd": [date(2025, 9, 1) + timedelta(weeks=i) for i in range(n)],
                           "a": rng.standard_normal(n), "b": rng.standard_normal(n)})
        spec = {"id": "T", "gap_pred": ["a", "b"]}
        y = 0.5 * df["a"].to_numpy() + rng.standard_normal(n)
        cr = J.CellRun(spec, df, {})
        cr.base = np.ones(n, bool)
        r1 = cr._gap_resid(y)
        self.assertTrue(np.isnan(r1[:J.H52_FIT_MIN]).all())
        self.assertTrue(np.isfinite(r1[J.H52_FIT_MIN:]).all())
        y2 = y.copy()
        y2[40:] += 50.0                                   # 뒤 월요일만 바꿈
        r2 = cr._gap_resid(y2)
        np.testing.assert_allclose(r1[:41], r2[:41] - np.r_[np.zeros(40), 50.0])
        # 손 계산: 21번째 값 = y − 앞 20개 OLS 예측
        A = np.column_stack([np.ones(20), df[["a", "b"]].to_numpy()[:20]])
        b = np.linalg.lstsq(A, y[:20], rcond=None)[0]
        self.assertAlmostEqual(r1[20], y[20] - np.r_[1.0, df.loc[20, ["a", "b"]].to_numpy(float)] @ b, places=10)


class TestRealGuard(unittest.TestCase):
    def test_time_gate(self):
        self.assertFalse(J.real_allowed(datetime(2026, 10, 22, 12, 0, tzinfo=J.KST))[0])
        self.assertFalse(J.real_allowed(datetime(2026, 10, 24, 5, 59, tzinfo=J.KST))[0])
        self.assertTrue(J.real_allowed(datetime(2026, 10, 24, 6, 0, tzinfo=J.KST))[0])
        with self.assertRaises(J.RealPriceGuard):
            J._assert_real_allowed(datetime(2026, 10, 25, tzinfo=J.KST), confirm=False)
        with self.assertRaises(J.RealPriceGuard):
            J._assert_real_allowed(datetime(2026, 10, 23, 20, tzinfo=J.KST), confirm=True)

    def test_guard_runs_before_any_read(self):
        src = inspect.getsource(J.load_prices)
        self.assertLess(src.index("_assert_real_allowed(now, confirm_real)"), src.index("_read_real_daily()"))
        self.assertLess(src.index("_read_real_daily()"), src.index("_assert_real_data(px)"))
        msrc = inspect.getsource(J.main)
        self.assertLess(msrc.index("_assert_real_allowed(None, a.confirm_real)"), msrc.index("prepare_inputs(opts)"))

    def test_data_gate(self):
        px = {"^KS11": pd.DataFrame({"close": [1.0]}, index=[date(2026, 10, 22)]),
              "SOXX": pd.DataFrame({"close": [1.0]}, index=[date(2026, 10, 23)])}
        with self.assertRaises(J.RealPriceGuard):
            J._assert_real_data(px)
        px["^KS11"] = pd.DataFrame({"close": [1.0]}, index=[date(2026, 10, 23)])
        J._assert_real_data(px)

    def test_fomc_matches_frozen_code(self):
        sp = importlib.util.spec_from_file_location("h01p", J.INTERIM / "code" / "h01_preview.py")
        m = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(m)
        self.assertEqual([d for d in J.FOMC if d.year == 2026], list(m.FOMC_2026))


class TestFakePrices(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.windows = J.load_windows()
        feat, _ = J.load_features(cls.windows)
        cls.feat = feat

    def test_fake_reads_no_price_file(self):
        seen = []
        orig = pd.read_parquet

        def spy(p, *a, **k):
            seen.append(str(p))
            return orig(p, *a, **k)
        with mock.patch.object(J.pd, "read_parquet", side_effect=spy), \
                mock.patch.object(J, "_read_real_daily", side_effect=AssertionError("real read")):
            pr = J.load_prices("fake", windows=self.windows, seed=7)
        self.assertTrue(all("prices" not in Path(s).name for s in seen), seen)
        self.assertTrue(all("preview" not in s for s in seen), seen)
        self.assertEqual(pr["meta"]["mode"], "fake")
        kr = pr["kr"].set_index("win_id")
        # §2.5: 목표 끝 세션이 10/23을 넘으면 결측
        self.assertTrue(np.isfinite(kr.at["K2026-10-19", "fut_kospi_R5"]))
        self.assertTrue(np.isnan(kr.at["K2026-10-20", "fut_kospi_R5"]))
        self.assertTrue(np.isnan(kr.at["K2026-10-13", "fut_kospi_R10"]))
        # M0 열은 창 마감 전 값: P 종가 기준 r는 D 시가와 무관
        self.assertTrue(np.isfinite(kr["m0_kospi_r20"].iloc[-1]))

    def test_schedule_dummies(self):
        """§2.4·§3 H50: 티커별 실적일 → 전체 실적 더미와 메모리(MU·삼전) 더미, CPI·고용·MSCI 파일 더미. 미국 창에도 같은 더미."""
        px, nq, cal0, _ = J._fake_daily(5, "walk", None, None)
        self.assertEqual(cal0, {})
        cal = {"earn": {"MU": [date(2026, 6, 24)], "NVDA": [date(2026, 8, 26)]}, "cpi": [date(2026, 7, 14)]}
        pn = J.build_panel(px, nq, self.windows, cal)
        self.assertEqual(sorted(pn["schedule_meta"]["missing"]), ["jobs", "msci"])
        for mk in ("kr", "us"):
            d = pn[mk]
            for c in J.M0_KR_DUM:
                self.assertIn(c, d.columns)

            def hit(col, e):
                return d.loc[d[col] == 1.0, "win_id"].tolist(), [w for w, P, D in zip(d["win_id"], d["Pd"], d["Dd"])
                                                                  if P <= e <= D - timedelta(days=1)]
            got, want = hit("m0_cal_cpi", date(2026, 7, 14))
            self.assertEqual(got, want)
            self.assertEqual(len(got), 1)
            got, want = hit("m0_cal_earn_mem", date(2026, 6, 24))
            self.assertEqual(got, want)
            self.assertEqual(int(d["m0_cal_earn"].sum()), 2)
            self.assertEqual(int(d["m0_cal_earn_mem"].sum()), 1)
        pn0 = J.build_panel(px, nq, self.windows, {})
        self.assertIn("earn", pn0["schedule_meta"]["missing"])
        self.assertEqual(float(pn0["us"]["m0_cal_msci"].sum()), 0.0)

    def test_plant_ic(self):
        x = self.feat["kr"].set_index("win_id")["A_rz8"]
        plant = {"name": "A_rz8", "x": x, "windows": self.windows["kr"], "ic": 0.30}
        pr = J.load_prices("fake", windows=self.windows, seed=3, scenario="plant", plant=plant)
        m = pr["meta"]["plant"]
        self.assertEqual(m["ic_target_population"], 0.30)
        self.assertGreater(m["beta"], 0)
        self.assertGreater(m["ic_realized_2026"], 0.10)          # 모집단 0.30 둘레(씨앗마다 흔들림)
        self.assertLess(m["ic_realized_2026"], 0.50)
        pr2 = J.load_prices("fake", windows=self.windows, seed=4, scenario="plant", plant=plant)
        self.assertEqual(pr2["meta"]["plant"]["beta"], m["beta"])       # β는 씨앗과 무관
        self.assertNotEqual(pr2["meta"]["plant"]["ic_realized_2026"], m["ic_realized_2026"])
        walk = J.load_prices("fake", windows=self.windows, seed=3)
        # 심은 효과는 KOSPI HL만 바꾼다(종가·M0는 같다)
        np.testing.assert_allclose(pr["kr"]["m0_kospi_r20"], walk["kr"]["m0_kospi_r20"], equal_nan=True)
        np.testing.assert_allclose(pr["kr"]["fut_kospi_R5"], walk["kr"]["fut_kospi_R5"], equal_nan=True)


class TestEndToEnd(unittest.TestCase):
    def test_small_run(self):
        opts = {"h31_reg_date": None}
        inp = J.prepare_inputs(opts)
        res = J.run_once(inp, "fake", "walk", 11, opts, 5)
        self.assertIn("H03", res["tested"]["T1"])
        self.assertEqual(res["cells"]["H31"]["verdict"], "설명용")
        self.assertEqual(res["cells"]["H05"]["verdict"], "멈춤")
        self.assertIn("jul3", res["cells"]["H03"]["values"]["robust"])
        self.assertIn("xp_robust_ok", res["cells"]["H03"]["values"])
        self.assertIn("est_2025", res["cells"]["H03"]["values"])
        self.assertEqual(res["cells"]["H68"]["role"], "main")
        self.assertNotIn("H68", res["tested"]["T2"])                   # 7/1~10/23 독립 묶음 7 < 8
        self.assertIsNotNone(res["cells"]["H52"]["info"]["gap_model"]["first_D"])
        self.assertIn("LL_US_noNVDL_d20", res["cells"]["H31"]["info"]["aux"])
        fitted = [m for m in res["cells"]["H46"]["info"]["M0"]["months"] if "kept" in m]
        self.assertTrue(fitted and all("m0_missing_rate" in m["kept"] for m in fitted))     # 미국 M0에 결번율
        for k in ("H33", "H34", "H76", "H77", "H78"):
            self.assertIn(k, J.STATUS_ONLY["전방(실시간 전용)"])
        self.assertEqual(res["global_rotation"]["n"], 5)


if __name__ == "__main__":
    unittest.main()
