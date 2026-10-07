"""AI 탐방 contract: the hand-made route must stay checkable (stops in product order, every link
dated and sourced, every company on one main stop) and its join with the fund PDFs must follow
the funds' own codes (Korean listing → US ADR, share classes, futures)."""

from __future__ import annotations

import copy
import unittest
from datetime import date
from unittest import mock

from desk import tour


class TourMapTest(unittest.TestCase):
    def setUp(self):
        self.m = tour.load_map()

    def test_shipped_map_has_no_problems(self):
        self.assertEqual(tour.check_map(self.m), [])

    def test_route_runs_from_basic_inputs_to_the_finished_product(self):
        ids = [s["id"] for s in self.m["stops"]]
        self.assertEqual(ids[0], "power")
        self.assertLess(ids.index("fab"), ids.index("packaging"))
        self.assertLess(ids.index("packaging"), ids.index("chips"))
        self.assertLess(ids.index("assembly"), ids.index("datacenter"))
        self.assertLess(ids.index("datacenter"), ids.index("labs"))
        self.assertEqual(ids[-1], "apps")
        for s in self.m["stops"]:  # what a stop receives comes from earlier stops
            for src in s["from"]:
                self.assertLess(ids.index(src), ids.index(s["id"]), (src, s["id"]))

    def test_links_are_sourced_and_not_after_the_map_date(self):
        for ln in self.m["links"]:
            self.assertTrue(ln["src"].startswith("http"), ln)
            self.assertLessEqual(ln["date"], self.m["as_of"], ln)

    def test_checker_catches_what_would_make_the_tour_lie(self):
        bad = copy.deepcopy(self.m)
        bad["links"].append({"axis": "openai", "t": "NOPE", "kind": "칩", "what": "x",
                             "date": "2099-01-01", "status": "확정", "src": "ftp://x"})
        first = next(z for _, z in tour.zones(bad) if not z.get("special"))
        first["members"].append({"t": "ASML", "role": "second placement without a main"})
        bad["stops"][0]["to"].append("nowhere")
        bad["stops"][1]["no"] = 9
        first["lead"] = "NOPE"
        second = [z for _, z in tour.zones(bad) if not z.get("special")][1]
        second.pop("lead_src", None)
        second["lead_why"] = "점유율 99%"                       # a number no shown source carries
        bad["companies"]["ZZRANK"] = {"name": "Z", "ko": "Z", "market": "US"}
        second["members"].append({"t": "ZZRANK", "role": "세계 1위"})  # a rank with no src and no deal link
        problems = "\n".join(tour.check_map(bad))
        for needle in ("지도에 없는 회사", "날짜", "출처 URL", "main 하나를", "없는 방문지", "순번", "대장 lead",
                       "출처 있는 문장", "순위·최대"):
            self.assertIn(needle, problems)

    def test_every_ai_universe_company_is_on_the_route(self):
        # watchlist.AI_UNIVERSE (the ATH-gap study's universe) is canonical for "which stocks are AI";
        # a name added there must get a stop here before the tour ships again.
        placed = {mem["t"] for _, z in tour.zones(self.m) for mem in z["members"]}
        self.assertEqual(sorted(t for t in tour.universe(self.m) if t not in placed), [])

    def test_universe_symbols_match_map_keys(self):
        self.assertEqual(tour.universe_symbol("005930.KS"), "005930")
        self.assertEqual(tour.universe_symbol("8299.TWO"), "8299")
        self.assertEqual(tour.universe_symbol("285A.T"), "285A")
        self.assertEqual(tour.universe_symbol("0100.HK"), "100")
        self.assertEqual(tour.universe_symbol("ATS.VI"), "ATS")
        self.assertEqual(tour.universe_symbol("ENR.DE"), "ENR")   # not Energizer (US ENR)
        self.assertEqual(tour.universe_symbol("SU.PA"), "SU")     # not Suncor (US SU)
        self.assertIn("SKHY", tour.universe(self.m))   # 000660.KS reaches the ADR key through aliases
        self.assertIn("GOOG", tour.universe(self.m))   # GOOGL share class

    def test_symbols_follow_the_fund_codes(self):
        self.assertEqual(tour.symbol("NVDA US EQUITY"), "NVDA")
        self.assertEqual(tour.symbol("GOOGL US EQUITY"), "GOOG")   # share class joins GOOG
        self.assertEqual(tour.symbol("SKHYV US EQUITY"), "SKHY")   # when-issued ADR ticker
        self.assertEqual(tour.symbol("NQZ6 INDEX"), "NQ")          # futures roll, one exposure
        self.assertEqual(tour.symbol("285A JP EQUITY"), "285A")
        self.assertEqual(tour.symbol("009150"), "009150")


class TourPayloadTest(unittest.TestCase):
    FAKE = {
        "456600": {"date": "2026-10-02", "weights": {"000660": 0.04, "NVDA": 0.03, "MSTR": 0.02, "ZZZZ": 0.01},
                   "names": {"000660": "SK하이닉스", "NVDA": "NVIDIA", "MSTR": "Strategy", "ZZZZ": "New Co"}},
        "2ETFQ1": {"date": "2026-10-02", "weights": {"GOOG": 0.01}, "names": {"GOOG": "Alphabet"}},
    }

    def _payload(self, today):
        with mock.patch.object(tour, "latest_holdings", side_effect=lambda fund_id, root=None: self.FAKE[fund_id]):
            return tour.payload(today=today)

    def test_join_uses_aliases_and_flags_new_holdings(self):
        d = self._payload(date(2026, 10, 5))
        self.assertAlmostEqual(d["funds"]["TIME"]["weights"]["SKHY"], 0.04)  # 000660 is SK hynix's KRX line
        self.assertEqual([r["t"] for r in d["unplaced"]], ["ZZZZ"])        # not on the route, not reviewed
        self.assertIn("MSTR", [r["t"] for r in d["outside_held"]])           # reviewed and kept outside
        self.assertFalse(d["stale"])

    def test_map_goes_stale_after_the_review_window(self):
        m = tour.load_map()
        late = date.fromisoformat(m["as_of"]).toordinal() + m["review_days"] + 1
        self.assertTrue(self._payload(date.fromordinal(late))["stale"])


if __name__ == "__main__":
    unittest.main()
