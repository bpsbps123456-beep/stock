"""특징 3단계(feat_topic) 테스트: 작은 가짜 단위로 창 경계·S·D_SD 풀·z(자기 제외)·Tilt·Herd·AS 상한을 본다.

실행: py -m unittest test_feat_topic   (이 폴더에서)
가격·네트워크를 쓰지 않는다. 사전 v0.2 매처만 실제 것을 쓴다.
"""

from __future__ import annotations

import sys
import unittest
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import feat_topic as ft  # noqa: E402
from lexicon_v02_final import load_v02  # noqa: E402


def make_frame(rows):
    """rows: (dt, utype, a, no, title, text, extra dict)."""
    units = pd.DataFrame([{"dt_kst": pd.Timestamp(r[0]), "utype": r[1], "a": r[2], "no": r[3],
                           "cid": (-1 if r[1] == "post" else i), "title": r[4], "text": r[5]}
                          for i, r in enumerate(rows)])
    units = units.sort_values("dt_kst", kind="stable").reset_index(drop=True)
    base = {"excl": "", "score": np.nan, "alias": "", "coin": False, "nE": 0, "nF": 0, "cb": 0, "cs": 0, "mem": 0,
            "semi": 0, "tus": 0, "tkr": 0}
    ex = {(r[3], pd.Timestamp(r[0]), r[2]): r[6] for r in rows}   # (no, dt, a) → 적중 값
    hits = pd.DataFrame([{**base, **ex[(u["no"], u["dt_kst"], u["a"])]} for _, u in units.iterrows()])
    hits.insert(0, "no", units["no"].to_numpy())
    hits.insert(1, "cid", units["cid"].to_numpy())
    return ft.Frame(units, hits)


class WindowTest(unittest.TestCase):
    def test_get_half_open(self):
        F = make_frame([("2026-10-14 06:00:00", "post", "a0", 1, "t", "b", {}),
                        ("2026-10-15 07:59:59", "comment", "a1", 1, None, "x", {}),
                        ("2026-10-15 08:00:00", "comment", "a2", 1, None, "y", {})])
        sub = F.get(datetime(2026, 10, 14, 8), datetime(2026, 10, 15, 8))
        self.assertEqual(list(sub["a"]), ["a1"])                       # 07:59:59 포함, 08:00 제외
        self.assertTrue(F.complete(datetime(2026, 10, 14, 8), datetime(2026, 10, 15, 8), datetime(2026, 10, 16)))
        self.assertFalse(F.complete(datetime(2026, 10, 14, 5), datetime(2026, 10, 15, 8), datetime(2026, 10, 16)))

    def test_nm_window(self):
        nyse = [date(2026, 9, 3), date(2026, 9, 4), date(2026, 9, 8), date(2026, 9, 9), date(2026, 9, 10)]
        o, c, k = ft.nm_window(date(2026, 9, 10), nyse)
        self.assertEqual((o, c, k), (datetime(2026, 9, 9, 22, 31), datetime(2026, 9, 10, 8), "night"))
        o, c, k = ft.nm_window(date(2026, 9, 7), nyse)          # 월요일 → 금요일 개장부터
        self.assertEqual((o, k), (datetime(2026, 9, 4, 22, 31), "weekend"))
        o, c, k = ft.nm_window(date(2026, 9, 8), nyse)          # 미국 노동절 다음 날 → 금요일 개장부터
        self.assertEqual((o, k), (datetime(2026, 9, 4, 22, 31), "extended"))
        o, _, _ = ft.nm_window(date(2026, 11, 10), [date(2026, 11, 9)])   # 서머타임 해제 → 23:31
        self.assertEqual(o, datetime(2026, 11, 9, 23, 31))

    def test_weekend_windows_dst(self):
        a, b = ft.weekend_windows(date(2026, 10, 10))
        self.assertEqual(a, (datetime(2026, 10, 9, 15, 31), datetime(2026, 10, 10, 6)))
        self.assertEqual(b[1], datetime(2026, 10, 12, 7))          # 월 06:59까지(서머타임)
        _, b = ft.weekend_windows(date(2026, 11, 7))
        self.assertEqual(b[1], datetime(2026, 11, 9, 8))           # 월 07:59까지(해제)


class StatTest(unittest.TestCase):
    def test_s_stats(self):
        sub = pd.DataFrame({"utype": ["post"] * 4 + ["comment"] * 2, "a": ["a", "b", "c", "c", "x", "y"],
                            "nE": [1, 1, 0, 0, 0, 0], "nF": [0, 0, 1, 0, 1, 1]})
        s = ft.s_stats(sub)
        self.assertAlmostEqual(s["E_A_p"], 2 / 3)
        self.assertAlmostEqual(s["F_A_p"], 1 / 3)
        self.assertAlmostEqual(s["S"], 0.5 * (1 / 3) + 0.5 * (0 - 1))

    def test_author_scores(self):
        p = pd.DataFrame({"utype": "post", "a": ["a", "a", "b", "c", "d"], "score": [2, -2, 1, np.nan, -3]})
        sc = ft.author_scores(p, {"d"})
        self.assertEqual(sorted(sc.tolist()), [1.0])               # a 평균 0 → 뺌, c 뉴스(점수 없음), d 연재

    def test_dsd_pool(self):
        rng = np.random.default_rng(0)
        scores = [rng.choice([-2.0, -1.0, 1.0, 2.0], 25) for _ in range(62)]
        scores[61] = scores[61][:10]                                # 표 10개 → 결측
        d = ft.dsd_series(scores, [f"w{i}" for i in range(62)], "t")
        self.assertTrue(d.loc[:39, "dsd_prov40"].isna().all())
        self.assertTrue(np.isfinite(d.loc[40, "dsd_prov40"]))
        self.assertTrue(d.loc[:59, "dsd"].isna().all())
        self.assertTrue(np.isfinite(d.loc[60, "dsd"]))
        self.assertTrue(np.isnan(d.loc[61, "dsd"]))
        self.assertEqual(d.loc[60, "pool_windows"], 60)
        d2 = ft.dsd_series(scores, [f"w{i}" for i in range(62)], "t")
        self.assertEqual(d.loc[60, "dsd"], d2.loc[60, "dsd"])       # 시드 고정 재현

    def test_z_prior_excludes_self(self):
        x = np.array([1.0, 2.0, 3.0, 100.0])
        z = ft.z_prior(x, np.ones(4, bool), 3, 3)
        self.assertTrue(np.isnan(z[:3]).all())
        self.assertAlmostEqual(z[3], (100 - 2.0) / 1.0)

    def test_tilt_delta(self):
        x = np.arange(30, dtype=float)
        valid = np.ones(30, bool)
        valid[10] = False                                           # 한쪽 휴장 창
        t5, dm, da = ft.tilt_delta(x, valid)
        v = [i for i in range(30) if i != 10]
        k = len(v)                                                  # 마지막 줄의 유효 개수 29
        self.assertAlmostEqual(t5[29], np.mean(v[k - 5:k]))
        self.assertAlmostEqual(dm[29], np.mean(v[k - 5:k]) - np.mean(v[k - 25:k - 20]))
        self.assertAlmostEqual(da[29], np.mean(v[k - 5:k]) - v[k - 21])
        self.assertTrue(np.isnan(dm[23]))                           # 유효 24개 → 아직 없음

    def test_herd_events(self):
        x = np.zeros(140)
        x[::7] = 0.01
        x[125], x[128], x[139] = 0.05, 0.06, 0.07
        e = ft.herd_events(x, np.ones(140, bool))
        self.assertTrue(np.isnan(e[:120]).all())
        self.assertEqual(e[125], 1.0)
        self.assertEqual(e[128], 0.0)                               # 10창 안 재점화 = 같은 사건
        self.assertEqual(e[139], 1.0)


class LexTest(unittest.TestCase):
    def test_unit_hits_alias_chase(self):
        an = load_v02()
        units = pd.DataFrame({"utype": ["post", "post", "post", "comment"], "no": [1, 2, 3, 3], "cid": [-1, -1, -1, 0],
                              "title": ["삼전닉스 풀매수", "엔비디아 몰빵", "[속보] 하닉", None],
                              "text": ["오늘 들어갔습니다", "그냥 그래요", "기사", "비트코인 얘기"]})
        h = ft.unit_hits(units, an)
        self.assertEqual(h.loc[0, "alias"], "SEC,SKH")
        self.assertGreater(h.loc[0, "cb"], 0)
        self.assertEqual(h.loc[1, "alias"], "NVDA")
        self.assertEqual(h.loc[1, "cb"], 0)                         # v0.2에서 뺀 '몰빵'
        self.assertEqual(h.loc[2, "excl"], "news")
        self.assertTrue(h.loc[3, "coin"])


class WeeklyTest(unittest.TestCase):
    def test_mention_cap(self):
        rows = [("2026-10-12 07:00:00", "post", "z", 1, "아무 글", "", {})]
        for i in range(5):                                          # 한 작성자 5글 → 3글로 자른다
            rows.append((f"2026-10-13 1{i}:00:00", "post", "a", 10 + i, "삼전", "", {"alias": "SEC"}))
        rows.append(("2026-10-14 10:00:00", "post", "b", 20, "삼전", "", {"alias": "SEC"}))
        F = make_frame(rows)
        old = ft.F_T_END
        ft.F_T_END = datetime(2026, 11, 1)
        try:
            cal = {"krx": [date(2026, 10, 12), date(2026, 10, 19)], "nyse": []}
            wd = pd.DataFrame({"D": pd.to_datetime(["2026-10-12", "2026-10-19"]),
                               "close_kst": pd.to_datetime(["2026-10-12 08:00", "2026-10-19 08:00"])})
            w, _ = ft.build_weekly(F, wd, "kr", cal)
        finally:
            ft.F_T_END = old
        self.assertEqual(int(w.loc[0, "h39_m_SEC"]), 4)
        self.assertEqual(int(w.loc[0, "h39_m_uncapped_SEC"]), 6)
        self.assertAlmostEqual(w.loc[0, "h39_s_SEC"], 4 / 6)


if __name__ == "__main__":
    unittest.main()
