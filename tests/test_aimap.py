"""AI 산업 지도 explainer (aimap): every stop and zone has plain-language text, every pictured zone
gets a spotlight crop, crops stay inside the picture, and nothing personal reaches the public site.
Build: py -m aimap · rules: docs/desk_system.md §8.3."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import unittest

from aimap import build
from desk import scenes, tour

CUSTOM = {"lab"}          # stop 13's picture is drawn in the page, not rendered


class AimapTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m, cls.sc, cls.g = tour.load_map(), scenes.load(), build.load_guide()

    def test_guide_explains_the_whole_map(self):
        # a new stop, zone, deal kind or status needs its plain-language text in aimap/guide.json
        self.assertEqual(build.check(self.m, self.sc, self.g), [])

    def test_every_pictured_zone_gets_a_crop(self):
        for st in self.m["stops"]:
            if not st.get("scene") or st["scene"] in CUSTOM:
                continue
            self.assertTrue(build.hero_panels(st, self.sc), f"{st['id']}: 전체 그림 없음")
            for z in st["zones"]:
                self.assertTrue(build.zone_panels(z["id"], st, self.sc), f"{st['id']}/{z['id']}: 확대 그림 없음")

    def test_own_scene_comes_first(self):
        cooling = next(z for st in self.m["stops"] for z in st["zones"] if z["id"] == "cooling")
        dc_stop = next(st for st in self.m["stops"] if any(z is cooling for z in st["zones"]))
        self.assertEqual(build.zone_panels("cooling", dc_stop, self.sc)[0], "dc")

    def test_crop_box_stays_inside_at_the_aspect(self):
        W, H = 1200, 716
        for bbox in ([10, 10, 30, 40], [1150, 690, 1199, 715], [0, 0, 1200, 716], [400, 100, 1180, 300]):
            x, y, cw, ch = build.crop_box(bbox, W, H, 1.5, 300, 0.16)
            self.assertGreaterEqual(x, 0)
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(x + cw, W + 1e-6)
            self.assertLessEqual(y + ch, H + 1e-6)
            self.assertAlmostEqual(cw / ch, 1.5, places=6)
            if bbox[2] - bbox[0] < cw:   # the zone sits inside its crop
                self.assertLessEqual(x, bbox[0])
                self.assertGreaterEqual(x + cw, bbox[2])

    def test_mask_paths_parse(self):
        for p in self.sc["panels"].values():
            for sp in p["spots"]:
                self.assertTrue(build.RECT.findall(sp["path"]), sp["zone"])

    def test_payload_carries_no_personal_data(self):
        data = build.payload(self.m, self.sc, self.g, {"zones": {}, "heroes": {}, "thumbs": {}})
        self.assertFalse(build.PRIVATE_KEYS & set(data))
        self.assertFalse(any(c.get("market") in build.HIDE_MARKETS for c in data["companies"].values()))
        raw = json.dumps(data, ensure_ascii=False)
        for word in ("weights", "journal", "내 보유", "7종목", "trades"):
            self.assertNotIn(word, raw)

    def test_payload_numbers_zones_and_keeps_members_known(self):
        data = build.payload(self.m, self.sc, self.g, {"zones": {}, "heroes": {}, "thumbs": {}})
        self.assertEqual([st["no"] for st in data["stops"]], list(range(1, len(data["stops"]) + 1)))
        for st in data["stops"]:
            self.assertIn("from", st)
            self.assertIn("to", st)
            for i, z in enumerate(st["zones"]):
                self.assertEqual(z["n"], f"{st['no']}-{i + 1}")
                self.assertTrue(z["easy"])
                for mem in z["members"]:
                    self.assertIn(mem["t"], data["companies"])
        self.assertEqual(data["counts"]["zones"], sum(len(st["zones"]) for st in self.m["stops"]))

    def test_page_loads_no_third_party_script(self):
        html = (build.WEB / "index.html").read_text(encoding="utf-8")
        self.assertIn("app.js?v=__V__", html)
        self.assertIn("__SITE__/og.jpg", html)
        hosts = set(re.findall(r'(?:src|href)="https?://([^/"]+)', html))
        self.assertLessEqual(hosts, {"fonts.googleapis.com", "fonts.gstatic.com"})
        self.assertNotIn("<script src=\"http", html)

    @unittest.skipUnless(shutil.which("node"), "node 없음")
    def test_scripts_parse(self):
        for name in ("app.js", "sw.js"):
            r = subprocess.run(["node", "--check", str(build.WEB / name)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()
