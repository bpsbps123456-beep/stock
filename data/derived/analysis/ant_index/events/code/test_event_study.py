"""event_study.py·features_all.py 점검. 실제 events.json·가격은 읽지 않는다(합성 파일로 실제 실행 경로를 돌린다).

실행: py -m unittest test_event_study   (이 폴더에서)
"""
from __future__ import annotations

import ast
import inspect
import json
import textwrap
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import event_study as E
import snap_paths as SP

RES = Path(__file__).resolve().parent.parent / "results"


def synth_frame(n: int = 220, seed: int = 0, feats=("f_sig", "f_noise", "vw_ln_b_median"), start="2025-08-21") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    D = pd.bdate_range(start, periods=n)
    F = pd.DataFrame({"win_id": [f"K{d.date()}" for d in D], "market": "kr", "D": D,
                      "close_kst": D + pd.Timedelta(hours=8), "judge_ok": True, "retro_window": True,
                      "realtime": False, "kind_class": np.where(D.weekday == 0, "nonweekday", "weekday")})
    for f in feats:
        F[f] = rng.normal(size=n)
    return F


def synth_meta(feats=("f_sig", "f_noise", "vw_ln_b_median")) -> dict:
    m = {}
    for f in feats:
        m[f] = {"bundle": "조회·추천" if f.startswith("vw_") else "활동", "ko": f, "status": "통과",
                "leak": "의심" if f.startswith("vw_") else "없음", "realtime_comparable": not f.startswith("vw_"),
                "event_study": True, "note": ""}
    return m


class TestTransforms(unittest.TestCase):
    def test_z_prev_matches_naive(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=80)
        x[[5, 30, 31, 50]] = np.nan
        z = E.z_prev(x)
        for t in range(80):
            h = x[max(0, t - 20):t]
            h = h[np.isfinite(h)]
            if not np.isfinite(x[t]) or len(h) < 15:
                self.assertTrue(np.isnan(z[t]))
            else:
                self.assertAlmostEqual(z[t], (x[t] - h.mean()) / h.std(ddof=1))

    def test_pct_prev_midrank(self):
        x = np.r_[np.arange(60, dtype=float), 29.0, 100.0, -1.0]
        p = E.pct_prev(x)
        self.assertAlmostEqual(p[60], (29 + 0.5) / 60)        # 같은 값 하나는 반으로
        self.assertAlmostEqual(p[61], 1.0)
        self.assertAlmostEqual(p[62], 0.0)
        self.assertTrue(np.isnan(p[39]))                        # 직전 값 40개 미만

    def test_kind_adjust_removes_level_shift(self):
        rng = np.random.default_rng(2)
        n = 150
        nonwk = (np.arange(n) % 5) == 0
        x = np.where(nonwk, 5.0, 10.0) + rng.normal(0, 0.1, n)
        xa = E.kind_adjust(x, nonwk)
        self.assertTrue(np.allclose(xa[~nonwk], x[~nonwk]))   # 평일 창은 그대로
        late = np.flatnonzero(nonwk)[np.flatnonzero(nonwk) > 60]
        self.assertTrue(np.all(np.abs(xa[late] - 10.0) < 0.5))
        self.assertTrue(np.isnan(xa[0]))                        # 이력 부족

    def test_gather_edges(self):
        A = np.arange(20, dtype=float).reshape(10, 2)
        G = E.gather(A, np.array([0, 9, -1]), [-1, 0, 1])
        self.assertTrue(np.isnan(G[0, 0]).all() and (G[0, 1] == A[0]).all() and (G[0, 2] == A[1]).all())
        self.assertTrue(np.isnan(G[1, 2]).all())
        self.assertTrue(np.isnan(G[2]).all())                   # 창 없음


class TestClassifier(unittest.TestCase):
    def test_vectorized_equals_naive(self):
        rng = np.random.default_rng(3)
        for trial in range(40):
            n = int(rng.integers(8, 25))
            v = np.round(rng.normal(size=n), 1)                 # 같은 값(동점) 섞기
            y = np.zeros(n, bool)
            y[rng.choice(n, int(rng.integers(3, n - 3)), replace=False)] = True
            Y = np.stack([rng.permutation(y) for _ in range(5)] + [y])
            groups = None if trial % 2 == 0 else rng.integers(0, max(3, n // 3), n)
            if groups is not None:          # 그룹마다 양쪽 표지가 남도록 사건은 따로 둔다
                groups = np.where(y, 1000 + np.arange(n), groups)
                Y = y[None, :]
            vec = E.loeo_ba(v, Y, groups)
            for s in range(Y.shape[0]):
                self.assertAlmostEqual(vec[s], E.loeo_ba_naive(v, Y[s], groups), msg=f"trial {trial} s {s}")

    def test_separable_and_constant(self):
        v = np.r_[np.arange(6) + 10.0, np.arange(8, dtype=float)]
        y = np.r_[np.ones(6, bool), np.zeros(8, bool)]
        self.assertEqual(E.loeo_ba(v, y[None])[0], 1.0)
        self.assertEqual(E.loeo_ba(-v, y[None])[0], 1.0)        # 방향도 배운다
        self.assertLessEqual(E.loeo_ba(np.zeros(14), y[None])[0], 0.5)

    def test_family_count_continuous_null(self):
        rng = np.random.default_rng(4)
        T = rng.normal(size=(2000, 30))
        p_obs = np.array([E.p_upper(o, T[:, j]) for j, o in enumerate(rng.normal(size=30))])
        fam = E.family_count(p_obs, T)
        self.assertEqual(fam["n_tests"], 30)
        self.assertAlmostEqual(fam["expected_by_chance"], 0.05 * 30, delta=0.25)
        self.assertTrue(np.all((p_obs > 0) & (p_obs <= 1)))

    def test_classify_detects_signal_and_not_noise(self):
        F = synth_frame(240, seed=5)
        meta = synth_meta()
        rng = np.random.default_rng(6)
        ev, fk = E.random_event_sets(len(F), 8, 8, rng)
        F.loc[ev, "f_sig"] += 4.0                                # 사건 창에서만 높다
        p = E.build_panel(F, meta, "kr")
        cls = E.classify(p, ev, fk, n_null=300, seed=7)
        tb = cls["table"].set_index(["feature", "offset"])
        self.assertLess(tb.loc[("f_sig", 0), "p"], 0.01)
        self.assertGreater(tb.loc[("f_sig", 0), "ba"], 0.9)
        self.assertGreater(tb.loc[("f_noise", 0), "p"], 0.001)
        self.assertEqual(cls["T_null"].shape, (300, len(tb)))

    def test_same_permutations_for_all_features(self):
        F = synth_frame(200, seed=8)
        meta = synth_meta()
        p = E.build_panel(F, meta, "kr")
        ev, fk = E.random_event_sets(len(F), 6, 6, np.random.default_rng(9))
        a = E.classify(p, ev, fk, n_null=50, seed=11)
        b = E.classify(p, ev, fk, n_null=50, seed=11)
        self.assertTrue(np.array_equal(a["perm"], b["perm"]))
        self.assertTrue(np.allclose(a["T_null"], b["T_null"], equal_nan=True))


class TestEventsAndPlacebo(unittest.TestCase):
    def test_fake_representatives(self):
        f = pd.DataFrame({"date": ["2026-01-02", "2026-01-05", "2026-01-06", "2026-02-02", "2026-02-03"],
                          "win_id": list("abcde"), "run": [0, 0, 0, 1, 1], "close": [10, 8, 8, 5, 6]})
        b = E.fake_representatives(f, "bottom")
        self.assertEqual(b["win_id"].tolist(), ["b", "d"])        # 묶음 최저, 같으면 앞 날
        t = E.fake_representatives(f, "top")
        self.assertEqual(t["win_id"].tolist(), ["a", "e"])

    def test_load_contrasts_same_kind_pivots(self):
        """위약에서 빼는 날짜는 같은 종류 꺾임점(기간 앞 포함)과 같은 종류 끝 극값뿐. 가짜는 묶음 대표로 줄인다."""
        t = {"pivots": [{"kind": "low", "date": "2026-01-05", "win_id": "K1"},
                        {"kind": "high", "date": "2026-02-02", "win_id": "K2"}],
             "warmup_pivots": [{"kind": "high", "date": "2025-03-03"}, {"kind": "low", "date": "2025-04-04"}],
             "tail_unconfirmed": {"kind": "low", "date": "2026-10-22"},
             "fake_bottoms": [{"date": "2026-03-03", "win_id": "K3", "run": 0, "close": 5.0},
                              {"date": "2026-03-04", "win_id": "K4", "run": 0, "close": 4.0}],
             "fake_tops": [], "held_near_tail": {"fake_bottoms": [{"date": "2026-10-19"}]}}
        cs = {c.kind: c for c in E.load_contrasts({"assets": {"KOSPI": {"window": "W_KR", "thresholds": {"8": t}}}})}
        self.assertEqual(cs["bottom"].pivot_dates, ["2025-04-04", "2026-01-05", "2026-10-22"])
        self.assertEqual(cs["top"].pivot_dates, ["2025-03-03", "2026-02-02"])
        self.assertEqual(cs["bottom"].events["win_id"].tolist(), ["K1"])
        self.assertEqual(cs["bottom"].fake_reps["win_id"].tolist(), ["K4"])
        self.assertEqual(cs["bottom"].info["held_near_tail"], ["2026-10-19"])
        self.assertEqual(len(cs["top"].fake_reps), 0)

    def test_placebo_pools_exclude_pivots_and_match(self):
        F = synth_frame(200, seed=12)
        p = E.build_panel(F, synth_meta(), "kr")
        rng = np.random.default_rng(13)
        close = E.synthetic_close(p.D, rng)
        ev_dates = [str(p.D[100]), str(p.D[150])]
        pivots = ev_dates + [str(p.D[60])]
        pools, info = E.placebo_pools(p, close, ev_dates, pivots, k_pool=10, excl_td=10)
        for pl in pools:
            self.assertEqual(len(pl), 10)
            for w in pl:
                for pv in (60, 100, 150):
                    self.assertGreater(abs(int(w) - pv), 10)
        d = close.index.to_numpy("datetime64[D]")
        c = close.to_numpy()
        pos = {x: i for i, x in enumerate(d)}

        def desc(D):
            i = pos[D]
            return np.array([c[i] / c[i - 5] - 1, c[i] / c[i - 20] - 1])
        e = desc(p.D[100])
        dist_in = [np.linalg.norm(desc(p.D[w]) - e) for w in pools[0]]
        self.assertLessEqual(np.median(dist_in), 0.1)               # 가까운 날을 골랐다
        self.assertEqual(len(E.pool_positions(pools)), 20)

    def test_profiles_band_is_mean_interval(self):
        """위약 띠 = 위약 날 z 평균 ± t(0.975, n−1) × SD ÷ √n (n = 그 오프셋에 값 있는 사건 수)."""
        from scipy.stats import t as t_dist
        F = synth_frame(200, seed=30, feats=("f_sig",))
        p = E.build_panel(F, synth_meta(("f_sig",)), "kr")
        ev = np.array([80, 120, 160])
        pools = [np.array([60, 61, 62]), np.array([90, 100]), np.array([140])]
        pr = E.profiles(p, ev, np.array([70, 110]), pools)
        j = int(np.flatnonzero(E.OFFSETS == 0)[0])
        pl = p.Z[[60, 61, 62, 90, 100, 140], 0]
        half = t_dist.ppf(0.975, 2) * pl.std(ddof=1) / np.sqrt(3)
        self.assertAlmostEqual(pr["pl_mean"][j, 0], pl.mean())
        self.assertAlmostEqual(pr["pl_hi"][j, 0] - pr["pl_mean"][j, 0], half)
        self.assertAlmostEqual(pr["ev_mean"][j, 0], p.Z[ev, 0].mean())

    def test_lead_hits(self):
        F = synth_frame(150, seed=14, feats=("f_sig",))
        p = E.build_panel(F, synth_meta(("f_sig",)), "kr")
        PCT = np.full_like(p.PCT, 0.5)
        PCT[97, 0] = 0.95
        PCT[120:125, 0] = np.nan
        q = E.Panel(p.market, p.win_id, p.D, p.feats, p.X, p.Z, PCT, p.meta, p.realtime)
        hi, lo = E.lead_hits(q, np.array([100, 110, 126]))
        self.assertEqual(hi[0, 0], 1.0)                             # 오프셋 −3에서 닿음
        self.assertEqual(hi[1, 0], 0.0)
        self.assertEqual(lo[1, 0], 0.0)
        self.assertTrue(np.isnan(hi[2, 0]))                         # −5~−1 값 0개

    def test_leak_test(self):
        F = synth_frame(220, seed=15)
        meta = synth_meta()
        ev, fk = E.random_event_sets(len(F), 8, 6, np.random.default_rng(16))
        G = F.copy()
        G.loc[ev + 1, "vw_ln_b_median"] += 5.0
        self.assertTrue(E.leak_test(E.build_panel(G, meta, "kr"), ev, fk)["flag"])
        G = F.copy()
        G.loc[np.r_[ev, ev + 1], "vw_ln_b_median"] -= 5.0
        lk = E.leak_test(E.build_panel(G, meta, "kr"), ev, fk)
        self.assertFalse(lk["flag"])
        self.assertTrue(E.leak_test(E.build_panel(F, meta, "kr"), ev[:2], fk)["flag"])   # 값 부족 → 의심

    def test_guard(self):
        ev = {"assets": {"KOSPI": {"thresholds": {"8": {"rejudge": {"deadline_close_in": False}}}}}}
        with self.assertRaises(SystemExit):
            E.guard_real_run(ev, datetime(2026, 10, 23, 12, 0, tzinfo=E.KST))
        with self.assertRaises(SystemExit):
            E.guard_real_run(ev, datetime(2026, 10, 25, 12, 0, tzinfo=E.KST))
        ev["assets"]["KOSPI"]["thresholds"]["8"]["rejudge"]["deadline_close_in"] = True
        E.guard_real_run(ev, datetime(2026, 10, 25, 12, 0, tzinfo=E.KST))


class TestConfirmedOnly(unittest.TestCase):
    """확정 사건만 모드(부록 14): 날짜 가드 대신 끝 극값·보류 가짜를 뺀다. 분석 경로는 같다."""

    @staticmethod
    def ev(data_through="2026-10-22", confirm="2026-10-12"):
        t = {"pivots": [{"kind": "low", "date": "2026-07-30", "win_id": "K2026-07-30", "confirm_date": "2026-07-31"},
                        {"kind": "high", "date": "2026-09-23", "win_id": "K2026-09-23", "confirm_date": confirm}],
             "warmup_pivots": [], "tail_unconfirmed": {"kind": "low", "date": "2026-10-22", "win_id": "K2026-10-22",
                                                       "status": "미확정"},
             "fake_bottoms": [{"date": "2026-10-05", "win_id": "K2026-10-05", "run": 0, "close": 1.0}],
             "fake_tops": [], "held_near_tail": {"fake_bottoms": [{"date": "2026-10-19"}], "fake_tops": []},
             "rejudge": {"deadline_close_in": False}}
        return {"assets": {"KOSPI": {"window": "W_KR", "data_through": data_through, "thresholds": {"8": t}}}}

    def test_checks_and_record(self):
        ev = self.ev()
        out, rec = E.confirmed_only_events(ev, "2026-10-22")
        self.assertEqual(out, ev)                                   # 사건·가짜 목록은 그대로(끝 극값은 원래 사건이 아니다)
        c = rec["contrasts"]["KOSPI_8"]
        self.assertEqual(c["tail_excluded"]["date"], "2026-10-22")
        self.assertEqual(c["held_fakes_excluded"]["fake_bottoms"], ["2026-10-19"])
        self.assertEqual(c["events_used"], {"low": 1, "high": 1})
        cs = {x.kind: x for x in E.load_contrasts(out)}
        self.assertNotIn("2026-10-22", cs["bottom"].events["date"].tolist())
        self.assertIn("2026-10-22", cs["bottom"].pivot_dates)        # 위약 제외 규칙은 그대로
        with self.assertRaises(SystemExit):
            E.confirmed_only_events(self.ev(data_through="2026-10-21"), "2026-10-22")   # 10/22 종가가 안 들어옴
        with self.assertRaises(SystemExit):
            E.confirmed_only_events(self.ev(confirm="2026-10-23"), "2026-10-22")         # 확정일이 기준일 뒤
        bad = self.ev()
        bad["assets"]["KOSPI"]["thresholds"]["8"]["fake_bottoms"].append(
            {"date": "2026-10-19", "win_id": "K2026-10-19", "run": 1, "close": 0.5})
        with self.assertRaises(SystemExit):
            E.confirmed_only_events(bad, "2026-10-22")                                  # 보류 가짜가 가짜 목록에
        bad = self.ev()
        bad["assets"]["KOSPI"]["thresholds"]["8"]["pivots"].append(
            {"kind": "low", "date": "2026-10-22", "win_id": "K2026-10-22", "confirm_date": "2026-10-22"})
        with self.assertRaises(SystemExit):
            E.confirmed_only_events(bad, "2026-10-22")                                  # 끝 극값이 사건 목록에

    def test_run_real_confirmed_only_end_to_end(self):
        """10/24 전·10/23 재판정 전이면 기본 run은 멈추고, 확정 사건만 모드는 같은 분석 경로로 끝까지 돈다."""
        feats = ("f_sig", "f_noise", "vw_ln_b_median")
        F = synth_frame(230, seed=40, feats=feats, start="2025-09-01")
        meta = synth_meta(feats)
        rng = np.random.default_rng(41)
        ev, fk = E.random_event_sets(len(F), 12, 10, rng)
        F.loc[ev, "f_sig"] += 3.0
        D = pd.to_datetime(F["D"])
        last = str(D.iloc[-1].date())
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            F.to_parquet(td / "features_all.parquet", index=False)
            (td / "features_all_meta.json").write_text(json.dumps({"features": meta}, ensure_ascii=False), "utf-8")
            close = E.synthetic_close(D.to_numpy("datetime64[D]"), rng)
            pd.DataFrame({"asset": "KOSPI", "date": close.index.astype(str), "close": close.to_numpy()}).to_csv(
                td / "prices_close.csv", index=False)
            piv = [{"kind": "low" if k < 6 else "high", "date": str(D[i].date()), "win_id": F.loc[i, "win_id"],
                    "confirm_date": str(D[min(i + 2, len(D) - 1)].date())} for k, i in enumerate(ev)]
            fb = [{"date": str(D[i].date()), "close": 1.0 + k, "win_id": F.loc[i, "win_id"], "run": k // 2}
                  for k, i in enumerate(fk)]
            evj = {"assets": {"KOSPI": {"window": "W_KR", "data_through": last, "thresholds": {"8": {
                "pivots": piv, "warmup_pivots": [], "tail_unconfirmed": {"kind": "low", "date": last},
                "fake_bottoms": fb, "fake_tops": fb, "held_near_tail": {"fake_bottoms": []},
                "rejudge": {"deadline_close_in": False}}}}}}
            (td / "events.json").write_text(json.dumps(evj), "utf-8")
            kw = dict(n_null=60, out_dir=td, ev_path=td / "events.json", features_path=td / "features_all.parquet",
                      meta_path=td / "features_all_meta.json", prices_path=td / "prices_close.csv",
                      now=datetime(2026, 10, 23, 10, 0, tzinfo=E.KST))
            with self.assertRaises(SystemExit):
                E.run_real(**kw)                                     # 날짜 가드
            self.assertEqual(E.run_real(**kw, confirmed_asof=last), 0)
            md = (td / "event_study.md").read_text("utf-8")
            self.assertIn("확정 사건만 모드", md)
            self.assertIn(f"KOSPI_8 {last}(low)", md)
            summ = json.loads((td / "event_study.json").read_text("utf-8"))
            self.assertEqual(summ["mode"]["asof"], last)
            tests = pd.read_csv(td / "event_study_tests.csv")
            self.assertTrue(set(tests["offset"]) == {0, 1})
            n = {c["name"]: c["n"]["events_total"] for c in summ["contrasts"]}
            self.assertEqual(n, {"KOSPI_8_bottom": 6, "KOSPI_8_top": 6})


class TestRealPathOnSynthetic(unittest.TestCase):
    """실제 실행 경로(run_real)를 합성 특징·합성 events.json·합성 가격으로 끝까지 돌린다."""

    def test_run_real_end_to_end(self):
        feats = ("f_sig", "f_noise", "vw_ln_b_median")
        F = synth_frame(230, seed=20, feats=feats, start="2025-09-01")
        meta = synth_meta(feats)
        rng = np.random.default_rng(21)
        ev, fk = E.random_event_sets(len(F), 13, 10, rng)       # 바닥 3 · 고점 10, 가짜 10(묶음 5)
        F.loc[ev, "f_sig"] += 3.0
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            F.to_parquet(td / "features_all.parquet", index=False)
            (td / "features_all_meta.json").write_text(json.dumps({"features": meta}, ensure_ascii=False), "utf-8")
            D = pd.to_datetime(F["D"])
            close = E.synthetic_close(D.to_numpy("datetime64[D]"), rng)
            pd.DataFrame({"asset": "KOSPI", "date": close.index.astype(str), "close": close.to_numpy()}).to_csv(
                td / "prices_close.csv", index=False)
            piv = [{"kind": "low", "date": str(D[i].date()), "win_id": F.loc[i, "win_id"]} for i in ev[:3]] + \
                  [{"kind": "high", "date": str(D[i].date()), "win_id": F.loc[i, "win_id"]} for i in ev[3:]]
            fb = [{"date": str(D[i].date()), "close": 1.0 + k, "win_id": F.loc[i, "win_id"], "run": k // 2}
                  for k, i in enumerate(fk)]
            evj = {"assets": {"KOSPI": {"window": "W_KR", "thresholds": {"8": {
                "pivots": piv, "warmup_pivots": [], "tail_unconfirmed": {"date": str(D[len(D) - 1].date())},
                "fake_bottoms": fb, "fake_tops": fb, "held_near_tail": {"fake_bottoms": []},
                "rejudge": {"deadline_close_in": True}}}}}}
            (td / "events.json").write_text(json.dumps(evj), "utf-8")
            rc = E.run_real(n_null=80, out_dir=td, ev_path=td / "events.json",
                            features_path=td / "features_all.parquet", meta_path=td / "features_all_meta.json",
                            prices_path=td / "prices_close.csv", now=datetime(2026, 10, 25, tzinfo=E.KST))
            self.assertEqual(rc, 0)
            for fn in ("event_study.md", "event_study.json", "event_study_tests.csv", "event_study_lead.csv",
                       "event_study_profiles.parquet", "event_study_figs/KOSPI_8_bottom.html"):
                self.assertTrue((td / fn).exists(), fn)
            md = (td / "event_study.md").read_text("utf-8")
            self.assertIn("우연 기대", md)
            tests = pd.read_csv(td / "event_study_tests.csv")
            self.assertTrue(set(tests["offset"]) == {0, 1})
            # 바닥 대비는 사건 3개 < 4라 표본 부족이어야 한다
            self.assertTrue((tests.loc[tests["kind"] == "bottom", "status"] == "표본 부족").all())
            top = tests[tests["kind"] == "top"].set_index(["feature", "offset"])
            self.assertTrue((top["status"] == "시험").all())
            self.assertTrue(bool(top.loc[("f_sig", 0), "candidate"]))       # 주입한 특징을 후보로 잡는다
            self.assertFalse(bool(top.loc[("vw_ln_b_median", 0), "candidate"]) and bool(top["leak_suspect"].any()))
            fig = (td / "event_study_figs/KOSPI_8_top.html").read_text("utf-8")
            self.assertIn("가격(사건일 종가 대비 %)", fig)
            self.assertEqual(fig.count("<svg"), 1 + len(feats))
            summ = json.loads((td / "event_study.json").read_text("utf-8"))
            self.assertEqual({c["name"] for c in summ["contrasts"]}, {"KOSPI_8_bottom", "KOSPI_8_top"})


class TestFeatureTable(unittest.TestCase):
    """features_all.parquet: 가격 열 없음, 메타 완비, 창 수."""

    @classmethod
    def setUpClass(cls):
        cls.path = RES / "features_all.parquet"
        if not cls.path.exists():
            raise unittest.SkipTest("features_all.parquet 없음(먼저 py features_all.py)")
        cls.F = pd.read_parquet(cls.path)
        cls.meta = json.loads((RES / "features_all_meta.json").read_text("utf-8"))

    def test_rows_and_markets(self):
        # 부록 14 갱신 스냅샷(2026-10-23 07:59): W_KR K2025-08-21~K2026-10-23 285창, W_U U2025-08-21~U2026-10-22 295창
        self.assertEqual(int((self.F["market"] == "kr").sum()), SP.EXPECT_WINDOWS["kr"])
        self.assertEqual(int((self.F["market"] == "us").sum()), SP.EXPECT_WINDOWS["us"])
        self.assertEqual(self.F.loc[self.F["market"] == "kr", "win_id"].max(), "K2026-10-23")
        self.assertEqual(self.F.loc[self.F["market"] == "us", "win_id"].max(), "U2026-10-22")
        self.assertTrue(self.F["win_id"].is_unique)

    def test_no_price_columns(self):
        bad = [c for c in self.F.columns if any(t in c.lower() for t in ("px_", "ev_", "kospi", "hynix", "soxx", "price",
                                                                          "ret_", "close_px"))]
        self.assertEqual(bad, [])

    def test_meta_complete(self):
        fm = self.meta["features"]
        md = json.loads(pq.read_schema(self.path).metadata[b"features_meta"].decode("utf-8"))
        self.assertEqual(fm, md)
        for f, m in fm.items():
            self.assertIn(f, self.F.columns)
            self.assertIn(m["status"], ("통과", "설명용"))
            self.assertIn(m["leak"], ("없음", "낮음", "의심", "있음"))
            self.assertTrue(m["bundle"])
        self.assertTrue(all(fm[f]["status"] == "설명용" for f in fm if f.startswith("em_")))
        self.assertEqual(fm["vw_ln_b_median"]["leak"], "의심")
        for f in ("tp_MC_A", "tp_LL_US_A", "tp_LS_A", "tp_KX_A", "tp_ANG_A", "tp_SEMI_A", "tp_MEM_A", "tp_TILT",
                  "pr_P_bar", "pr_P_share", "pr_P_strong1", "pr_P_strong5", "act_ln_posts_per_h"):
            self.assertEqual(fm[f]["status"], "통과", f)

    def test_build_checks_passed(self):
        ch = self.meta["checks"]
        for k in ("kx_pass_excl_equals_feat_text_units", "kx_pass_MC_equals_feat_text_units",
                  "new30_kr_calc_equals_feat_activity", "MC_recalc_equals_feat_text", "SEMI_post_recalc_equals_feat_text",
                  "auth_valid_equals_feat_text_n_auth_all", "TILT_us_equals_feat_topic_h46_tilt",
                  "frame_sizes_equal_label_strata", "emotion_kr_weights_proportional_to_day_cells",
                  "emotion_us_assign_matches_loader"):
            self.assertTrue(ch[k], k)
        self.assertEqual(ch["emotion_labels"]["missing"], 0)
        self.assertEqual(ch["pred_labels"]["missing"], 0)

    def test_outage_windows_empty(self):
        out = self.F[self.F["outage"]]
        feats = list(self.meta["features"])
        self.assertTrue(out[feats].isna().all().all())


class TestDryrunIsolation(unittest.TestCase):
    def test_dryrun_reads_no_events_or_prices(self):
        """시운전 함수 본문(설명문·주석 제외)에 실제 사건·가격 파일 이름이나 그 읽기 함수가 없다."""
        tree = ast.parse(textwrap.dedent(inspect.getsource(E.dryrun)))
        fn = tree.body[0]
        body = fn.body[1:] if isinstance(fn.body[0], ast.Expr) and isinstance(fn.body[0].value, ast.Constant) else fn.body
        consts = [n.value for b in body for n in ast.walk(b) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        names = {n.id for b in body for n in ast.walk(b) if isinstance(n, ast.Name)} |                 {n.attr for b in body for n in ast.walk(b) if isinstance(n, ast.Attribute)}
        self.assertFalse([c for c in consts if "events.json" in c or "prices_close" in c])
        self.assertFalse(names & {"load_contrasts", "price_series", "run_real", "read_csv"})


if __name__ == "__main__":
    unittest.main()
