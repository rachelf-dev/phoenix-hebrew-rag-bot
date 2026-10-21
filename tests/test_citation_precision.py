# Unit tests for topic-scoped citation filtering (apartment vs business liability URLs).
"""Run with: python tests/test_citation_precision.py"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phoenix_rag.answering.context import claim_path_markers  # noqa: E402
from phoenix_rag.answering.rerank import (  # noqa: E402
    demote_mismatched_paths,
    filter_citation_urls,
    filter_hits_for_question,
    path_matches_question,
    preferred_action_urls,
)
from phoenix_rag.indexing.retrieve import Hit  # noqa: E402

LEAK_Q = "יש לי נזילה בבית, איך מגישים תביעה?"
COMBINED_Q = (
    "איך מגישים תביעה על נזק לתכולה בדירה, "
    "ואיך מעדכנים מספר טלפון בשירות עצמי?"
)

APT_DAMAGE = "https://www.fnx.co.il/claims-customer-service/claims-appartment-insurance/damage-to-the-apartment/"
APT_PIPELINE = "https://www.fnx.co.il/claims-customer-service/claims-appartment-insurance/pipeline-damage/"
APT_SELF = "https://www.fnx.co.il/customer-service/apartment-insurance/selfservice_personal_info_update/"
BURGLARY = "https://www.fnx.co.il/claims-customer-service/claims-liability-insurance/business-burglary-old/"
FIRE = "https://www.fnx.co.il/claims-customer-service/claims-liability-insurance/business-fire-old/"
EMPLOYERS = "https://www.fnx.co.il/claims-customer-service/claims-liability-insurance/employers-liability/"


def _hit(url: str, title: str = "t") -> Hit:
    return Hit(
        chunk_id=url,
        url=url,
        display_url=url,
        page_title=title,
        heading_path="",
        text=title + " תביעה דירה נזק",
        score=0.5,
        section_type="claim",
        form_links="",
        rerank_score=0.5,
    )


class ClaimPathMarkerTests(unittest.TestCase):
    def test_leak_includes_pipeline_damage(self):
        markers = claim_path_markers(LEAK_Q)
        self.assertIn("/claims-customer-service/claims-appartment-insurance/pipeline-damage/", markers)


class PathMatchTests(unittest.TestCase):
    def test_apartment_claim_ok(self):
        self.assertTrue(path_matches_question(LEAK_Q, APT_DAMAGE))

    def test_business_liability_rejected_for_leak(self):
        self.assertFalse(path_matches_question(LEAK_Q, BURGLARY))
        self.assertFalse(path_matches_question(LEAK_Q, EMPLOYERS))

    def test_business_question_allows_liability(self):
        q = "איך מגישים תביעת חבות מעבידים בעסק?"
        self.assertTrue(path_matches_question(q, EMPLOYERS))

    def test_combined_keeps_apartment_service(self):
        self.assertTrue(path_matches_question(COMBINED_Q, APT_DAMAGE))
        self.assertTrue(path_matches_question(COMBINED_Q, APT_SELF))


class FilterHitsTests(unittest.TestCase):
    def test_drops_liability_hits(self):
        hits = [
            _hit(APT_DAMAGE, "נזק לדירה"),
            _hit(BURGLARY, "פריצה"),
            _hit(FIRE, "אש"),
            _hit(EMPLOYERS, "חבות מעבידים"),
        ]
        kept = filter_hits_for_question(LEAK_Q, hits)
        urls = {h.url for h in kept}
        self.assertIn(APT_DAMAGE, urls)
        self.assertNotIn(BURGLARY, urls)
        self.assertNotIn(FIRE, urls)
        self.assertNotIn(EMPLOYERS, urls)


class PreferredActionTests(unittest.TestCase):
    def test_prefers_marker_matched_apartment_urls_only(self):
        hits = [
            _hit(BURGLARY, "תביעה פריצה בעסק"),
            _hit(
                APT_PIPELINE,
                "נזילה בבית — איך מגישים תביעה על נזק צנרת ודוד",
            ),
            _hit(
                APT_DAMAGE,
                "נזילה בבית — איך מגישים תביעה על נזק לדירה ותכולה",
            ),
        ]
        urls = preferred_action_urls(LEAK_Q, hits)
        self.assertTrue(urls)
        for u in urls:
            self.assertIn("claims-appartment-insurance", u)
        self.assertLessEqual(len(urls), 2)


class DemotePathsTests(unittest.TestCase):
    def test_car_tow_question_does_not_crash_demote(self):
        hits = [_hit(APT_DAMAGE, "גרירה")]
        out = demote_mismatched_paths("איך מזמינים גרר ושירותי דרך ברכב?", hits)
        self.assertEqual(len(out), 1)


class FilterCitationTests(unittest.TestCase):
    def test_strips_liability_and_caps(self):
        mixed = [APT_DAMAGE, BURGLARY, FIRE, EMPLOYERS]
        out = filter_citation_urls(LEAK_Q, mixed, max_n=2)
        self.assertLessEqual(len(out), 2)
        self.assertTrue(all("claims-appartment-insurance" in u for u in out))
        self.assertFalse(any("liability" in u for u in out))


if __name__ == "__main__":
    unittest.main()
