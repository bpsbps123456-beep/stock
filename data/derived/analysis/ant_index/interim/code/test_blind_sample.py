"""사전 v0.2 후보 매처·가림 표본 테스트.

실행: py -m unittest test_blind_sample   (이 폴더에서)
앞 묶음은 작은 가짜 문장으로 셈 규칙(2음절 경계·가장 긴 표현·부정·질문형·옮기기·제외 문맥·가격 서술 문장·점수)과
가림을 본다. 마지막 묶음은 실제 산출물(blind_samples.json·blind_key.parquet)이 표본 틀 규칙을 지키는지 본다.
"""

from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blind_sample as B  # noqa: E402
import lexicon_v02 as L  # noqa: E402


def strata(title, body, utype="post"):
    r = L.analyze(title, body, utype)
    return [(h.stratum, h.word, h.origin) for h in r.hits], r


class MatcherRules(unittest.TestCase):
    def test_two_syllable_left_boundary(self):
        hs, _ = strata("", "부분익절 함")
        self.assertNotIn("E", [h[0] for h in hs])
        hs, _ = strata("", "오늘 익절했다")
        self.assertIn(("E", "익절", ""), hs)

    def test_longest_expression_wins_within_count(self):
        hs, _ = strata("", "결국 깡통계좌 됐다", "comment")
        self.assertIn(("MC", "깡통계좌", ""), hs)
        self.assertNotIn("F", [h[0] for h in hs])

    def test_negation_drops_cheer_and_fear(self):
        hs, r = strata("", "풀매수 안 함", "comment")
        self.assertNotIn("E", [h[0] for h in hs])
        self.assertEqual(r.neg_dropped, 1)
        hs, _ = strata("", "폭락 아님", "comment")
        self.assertNotIn("F", [h[0] for h in hs])

    def test_question_form_is_not_main(self):
        hs, r = strata("", "졸업할까요?", "comment")
        self.assertNotIn("E", [h[0] for h in hs])
        self.assertEqual(r.eq, 1)

    def test_capitulation_moves(self):
        hs, _ = strata("", "다 팔았던 건 아니고 들고 있어요", "comment")
        self.assertIn(("RES", "다 팔았", "CAP 부정→RES"), hs)
        hs, _ = strata("", "손절했나요?", "comment")
        self.assertIn(("HES", "손절했", "CAP 질문형→HES"), hs)
        hs, _ = strata("", "다 팔았어요 익절이라 좋네", "comment")
        self.assertIn(("E", "다 팔았", "CAP 이익 실현 문장→E"), hs)

    def test_context_exclusions(self):
        hs, _ = strata("", "신용카드로 신용 대출", "comment")
        self.assertNotIn("LL_KR", [h[0] for h in hs])
        hs, _ = strata("", "매출 2배 늘었다", "comment")
        self.assertNotIn("LL_KR", [h[0] for h in hs])
        hs, _ = strata("", "ETF 2배 샀다", "comment")
        self.assertIn(("LL_KR", "2배", ""), hs)
        hs, _ = strata("", "외인이 던졌다", "comment")
        self.assertNotIn("CAP", [h[0] for h in hs])

    def test_price_sentence_removed_from_sentiment(self):
        hs, r = strata("", "코스피 3% 폭락했다 무섭다", "comment")
        self.assertNotIn("F", [h[0] for h in hs])
        self.assertGreaterEqual(r.price_dropped, 1)
        hs, _ = strata("", "폭락 무섭다", "comment")
        self.assertIn("F", [h[0] for h in hs])

    def test_quotes_are_blanked(self):
        hs, _ = strata("", '누가 "떡상"이라고 하더라', "comment")
        self.assertNotIn("E", [h[0] for h in hs])

    def test_post_score_title_double_and_negation_flip(self):
        _, r = strata("가즈아", "떡락")
        self.assertEqual(r.score, 1)            # 제목 강세 2 − 본문 약세 1
        _, r = strata("", "떡상 안 함")
        self.assertEqual(r.score, -1)           # 부정이면 뒤집는다

    def test_tilt_drops_posts_with_both_sides(self):
        hs, _ = strata("미장 국장", "")
        self.assertFalse([h for h in hs if h[0].startswith("TILT")])
        hs, _ = strata("미장 가자", "")
        self.assertIn("TILT_US", [h[0] for h in hs])

    def test_noise_filters(self):
        self.assertEqual(L.analyze("[속보] 삼전", "본문", "post").excluded, "news")
        self.assertEqual(L.analyze(None, "코멘트가 삭제되었습니다", "comment").excluded, "deleted")
        self.assertEqual(L.analyze(None, "대통령 탄핵", "comment").excluded, "politics")

    def test_mock_second_person_rule(self):
        hs, _ = strata("", "ㅋㅋㅋㅋ 너 인버스 샀냐", "comment")
        self.assertIn("MOCK_BEAR", [h[0] for h in hs])

    def test_case_sensitive_alias(self):
        hs, _ = strata("", "mu 소리", "comment")
        self.assertNotIn("ALIAS_US", [h[0] for h in hs])
        hs, _ = strata("", "MU 실적", "comment")
        self.assertIn(("ALIAS_US", "MU", ""), hs)


class Blinding(unittest.TestCase):
    def test_mask_dates_times_numbers_events(self):
        s = B.mask_text("10/13 오후 3시 코스피 5,600 +3.2% 7월 쇼크 ADR 상장 2026년 하닉 180, 삼전 9만원")
        self.assertFalse(re.search(r"\d{3,}|%|\d\s?월|\d:\d\d", s), s)
        for tag in ("[날짜]", "[시각]", "[숫자]", "[사건]"):
            self.assertIn(tag, s)

    def test_hit_span_is_protected_and_marked(self):
        self.assertEqual(B.blind_segment("현금 100 했다 2026년", [(0, 6, 1)]), "⟦1:현금 100⟧ 했다 [날짜]")

    def test_excerpt_keeps_marks(self):
        s = "가" * 2000 + "⟦1:떡상⟧" + "나" * 2000
        out, cut = B.excerpt(s)
        self.assertTrue(cut)
        self.assertIn("⟦1:떡상⟧", out)
        self.assertLess(len(out), 1000)


class Outputs(unittest.TestCase):
    """실제 산출물 점검(파일이 없으면 건너뛴다)."""

    @classmethod
    def setUpClass(cls):
        js, kp = B.RESULTS / "blind_samples.json", B.RESULTS / "blind_key.parquet"
        if not (js.exists() and kp.exists()):
            raise unittest.SkipTest("산출물 없음")
        cls.d = json.loads(js.read_text(encoding="utf-8"))
        cls.k = pd.read_parquet(kp)

    def test_frame_days_excluded(self):
        days = set(self.k["dt_kst"].dt.date) | set(self.k["post_dt_kst"].dt.date)
        self.assertFalse(days & B.EXCLUDED_DAYS)
        self.assertLessEqual(max(days), B.CUTOFF)
        self.assertFalse(self.k["dt_conf"].eq("latest").any())

    def test_ids_match_key_and_no_identifiers_in_samples(self):
        ids = [x["id"] for x in self.d["samples"] + self.d["score_samples"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), set(self.k["id"]))
        banned = {"no", "cid", "a", "dt", "dt_kst", "post_dt_kst", "src_line", "unit_idx", "views", "up", "down"}
        for x in self.d["samples"] + self.d["score_samples"]:
            self.assertFalse(banned & set(x), x["id"])

    def test_counts(self):
        n = pd.Series([x["bundle"] for x in self.d["samples"]]).value_counts()
        self.assertTrue((n <= B.N_PER_STRATUM).all())
        self.assertEqual(len(self.d["score_samples"]), B.N_SCORE)
        for x in self.d["samples"]:
            self.assertTrue(x["hits"])
            s = " ".join(str(x.get(f, "")) for f in ("title", "body", "text"))
            self.assertEqual(len(re.findall("⟦", s)), len(x["hits"]), x["id"])


if __name__ == "__main__":
    unittest.main()
