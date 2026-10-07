"""AI 탐방 cutaways (단면도): every zone of a stop that has a picture must be drawn and labelled in
it, labels must not collide or hide an anchor, and the shipped layout must match the current map
and the last Blender export. Rebuild: desk/scenes/blender/build.py, then py -m desk.scenes.layout."""

from __future__ import annotations

import json
import unittest

from desk import scenes, tour
from desk.scenes import layout

CUSTOM = {"lab"}          # drawn by hand in tour.js (the two spires), not rendered


def webp_size(path) -> tuple[int, int]:
    head = path.read_bytes()[:30]
    assert head[:4] == b"RIFF" and head[8:12] == b"WEBP" and head[12:16] == b"VP8X", path.name
    return int.from_bytes(head[24:27], "little") + 1, int.from_bytes(head[27:30], "little") + 1


class ScenesTest(unittest.TestCase):
    def setUp(self):
        self.m = tour.load_map()
        self.s = scenes.load()
        self.panels = self.s.get("panels", {})

    def test_every_stop_picture_exists(self):
        for st in self.m["stops"]:
            sid = st.get("scene")
            if not sid or sid in CUSTOM:
                continue
            self.assertIn(sid, self.s["scenes"], f"{st['id']}: 단면도 {sid} 없음")
            self.assertTrue(self.s["scenes"][sid]["panels"], sid)
            for pid in self.s["scenes"][sid]["panels"]:
                self.assertTrue((scenes.STATIC / self.panels[pid]["img"]).exists(), pid)
        self.assertEqual(self.s.get("problems"), [], "py -m desk.scenes.layout 의 문제 목록을 먼저 해결하세요")

    def test_every_zone_of_a_pictured_stop_is_drawn(self):
        # a zone added to such a stop needs a model in desk/scenes/blender/ before the tour ships
        for st in self.m["stops"]:
            sid = st.get("scene")
            if not sid or sid in CUSTOM:
                continue
            drawn = {sp["zone"] for pid in self.s["scenes"][sid]["panels"] for sp in self.panels[pid]["spots"]}
            missing = [z["id"] for z in st["zones"] if z["id"] not in drawn]
            self.assertEqual(missing, [], f"{st['no']}. {st['name']}: 그림에 없는 분야")

    def test_labels_print_what_the_map_says_now(self):
        # names, ★ leads and 병목 come from tour_map.json; the layout measured those strings
        for pid, p in self.panels.items():
            for sp in p["spots"]:
                self.assertEqual(sp["text"], list(layout.label_text(self.m, sp["zone"])),
                                 f"{pid}/{sp['zone']}: 지도가 바뀌었습니다 — py -m desk.scenes.layout 을 다시 실행하세요")

    def test_labels_do_not_collide_or_cover_anchors(self):
        for pid, p in self.panels.items():
            boxes = [sp["box"] for sp in p["spots"]]
            for i, (x, y, w, h) in enumerate(boxes):
                self.assertGreaterEqual(x, 0, pid)
                self.assertGreaterEqual(y, 0, pid)
                self.assertLessEqual(x + w, p["w"], pid)
                self.assertLessEqual(y + h, p["h"], pid)
                for j, (x2, y2, w2, h2) in enumerate(boxes[i + 1:], i + 1):
                    apart = x + w <= x2 or x2 + w2 <= x or y + h <= y2 or y2 + h2 <= y
                    self.assertTrue(apart, f"{pid}: {p['spots'][i]['zone']} / {p['spots'][j]['zone']} 이름표 겹침")
                for sp in p["spots"]:
                    ax, ay = sp["anchor"]
                    self.assertFalse(x - 2 <= ax <= x + w + 2 and y - 2 <= ay <= y + h + 2,
                                     f"{pid}: {p['spots'][i]['zone']} 이름표가 {sp['zone']} 위치를 가림")

    def test_layout_matches_the_last_render(self):
        for pid, p in self.panels.items():
            raw = json.loads((scenes.ANCHORS / f"{pid}.json").read_text(encoding="utf-8"))
            self.assertEqual((raw["w"], raw["h"]), (p["w"], p["h"]), pid)
            self.assertEqual({s["zone"]: s["anchor"] for s in raw["spots"] if not s.get("hidden")},
                             {s["zone"]: s["anchor"] for s in p["spots"]},
                             f"{pid}: 다시 렌더한 뒤 py -m desk.scenes.layout 을 실행하세요")
            self.assertEqual(webp_size(scenes.STATIC / p["img"]), (p["w"] * 2, p["h"] * 2), pid)

    def test_payload_carries_the_scenes(self):
        self.assertEqual(tour.payload()["scenes"]["panels"].keys(), self.panels.keys())


if __name__ == "__main__":
    unittest.main()
