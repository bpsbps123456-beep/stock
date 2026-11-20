"""시즌 계기판(docs/desk_system.md §11.2): 확정값만 쓰는 모멘텀 성적, 다이얼 선택, 경로 이어 붙이기, 종이 기록 잠금."""

import unittest

import numpy as np

from desk import season as SE


class KnownIcTest(unittest.TestCase):
    def test_only_matured_values(self):
        ic = np.arange(30, dtype=float)
        kn = SE.known_ic(ic, horizon=5, window=4)
        # at t the newest usable ic index is t-5: mean of ic[t-8..t-5]
        self.assertTrue(np.isnan(kn[:5]).all())
        self.assertAlmostEqual(kn[12], np.mean(ic[4:8]))
        # changing values that are not matured yet must not move the known value
        later = ic.copy()
        later[8:] = -100
        self.assertAlmostEqual(SE.known_ic(later, 5, 4)[12], kn[12])


class ChooseTest(unittest.TestCase):
    def test_character_dial(self):
        self.assertEqual(SE.choose(0.1, None, gate=False), "MOM")
        self.assertEqual(SE.choose(-0.01, True, gate=False), "REV")
        self.assertEqual(SE.choose(float("nan"), False, gate=False), "MOM")  # unmatured counts as working; no gate

    def test_two_dials(self):
        self.assertEqual(SE.choose(0.3, False, gate=True), "CASH")
        self.assertEqual(SE.choose(-0.3, True, gate=True), "REV")
        self.assertEqual(SE.choose(0.3, None, gate=True), "MOM")  # unknown SOXX state is not "below"


class ChainTest(unittest.TestCase):
    def test_switch_uses_each_path(self):
        opn = {"MOM": np.array([1.0, 1.1, 1.21]), "REV": np.array([1.0, 0.9, 0.99])}
        cls = {"MOM": np.array([1.05, 1.15, 1.2]), "REV": np.array([0.95, 0.95, 1.0])}
        v = SE.chain(["MOM", "REV", "CASH"], opn, cls)
        self.assertAlmostEqual(v[0], 1.05)               # MOM open → close
        self.assertAlmostEqual(v[1], 1.1 * 0.95 / 0.9)   # MOM open→open, then REV open → close
        self.assertAlmostEqual(v[2], 1.1 * 0.99 / 0.9)   # REV open→open, then cash holds
        same = SE.chain(["MOM"] * 3, opn, cls)
        np.testing.assert_allclose(same, cls["MOM"])


class LedgerTest(unittest.TestCase):
    def test_lock_once(self):
        led = {"records": []}
        self.assertTrue(SE.append_record(led, {"session": "2026-11-20", "dial1": "REV"}))
        self.assertFalse(SE.append_record(led, {"session": "2026-11-20", "dial1": "MOM"}))
        self.assertEqual(led["records"][0]["dial1"], "REV")
        self.assertTrue(SE.append_record(led, {"session": "2026-11-19", "dial1": "MOM"}))
        self.assertEqual([r["session"] for r in led["records"]], ["2026-11-19", "2026-11-20"])
        self.assertEqual(led["start"], "2026-11-19")

    def test_session_after_skips_weekend_and_holiday(self):
        self.assertEqual(SE.session_after("2026-11-20"), "2026-11-23")
        self.assertEqual(SE.session_after("2026-11-25"), "2026-11-27")  # 11/26 추수감사절

    def test_paper_score_missing_day_keeps_previous_pick(self):
        days = ["2026-11-20", "2026-11-23", "2026-11-24"]
        opn = {"MOM": np.array([1.0, 1.1, 1.2]), "REV": np.array([1.0, 1.0, 0.9]), "QQQ": np.array([1.0, 1.0, 1.0])}
        cls = {k: v * 1.0 for k, v in opn.items()}
        recs = [{"session": "2026-11-20", "dial1": "REV", "dial2": "MOM"}]
        out = SE.paper_score(recs, days, opn, cls)
        self.assertEqual(out["missing"], ["2026-11-23", "2026-11-24"])
        self.assertAlmostEqual(out["dial1"], 0.9 - 1)  # REV kept on the missing days
        self.assertAlmostEqual(out["dial2"], 1.2 - 1)
        self.assertAlmostEqual(out["MOM"], 0.2)


class LabelTest(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(SE.season_label(0.2, 0.1, True)[0], "모멘텀 시즌")
        self.assertEqual(SE.season_label(-0.2, -0.1, True)[0], "되돌림 시즌")
        self.assertEqual(SE.season_label(-0.2, 0.1, True)[0], "전환 의심")
        self.assertEqual(SE.season_label(0.2, 0.1, False)[0], "급락 경계")


if __name__ == "__main__":
    unittest.main()
