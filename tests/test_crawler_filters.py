# Unit tests for crawl-time filter-page detection and canonical/hash dedup.
"""Run with: python tests/test_crawler_filters.py"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phoenix_rag.crawler.filters import (  # noqa: E402
    canonical_target,
    filter_reason,
    pick_canonical_keeper,
    plan_duplicates,
)


class FilterReasonTests(unittest.TestCase):
    def test_netfree_body(self):
        html = "<html><body>blockByNetFree: true</body></html>"
        self.assertEqual(filter_reason(html, "Home", "hello world " * 20), "filter_fingerprint")

    def test_blocked_title(self):
        self.assertEqual(
            filter_reason("<html></html>", "האתר נחסם | NetFree", ""),
            "filter_fingerprint",
        )

    def test_empty_main_foreign_title(self):
        self.assertEqual(
            filter_reason("<html><title>Welcome</title></html>", "Welcome", ""),
            "empty_main_foreign_title",
        )

    def test_phoenix_title_empty_main_is_not_a_filter(self):
        self.assertIsNone(filter_reason("<html></html>", "ביטוח דירה | הפניקס", ""))

    def test_real_phoenix_main_is_kept(self):
        main = "כך מגישים תביעת נזקי מים בדירה דרך האתר. " * 5
        self.assertIsNone(filter_reason("<html><main id='main'>" + main + "</main></html>", "תביעה | הפניקס", main))


class CanonicalTests(unittest.TestCase):
    def test_same_url_is_not_a_target(self):
        url = "https://www.fnx.co.il/car-insurance/"
        self.assertIsNone(canonical_target(url, url))
        self.assertIsNone(canonical_target(url, url.rstrip("/")))

    def test_alias_points_at_canonical(self):
        self.assertEqual(
            canonical_target(
                "https://www.fnx.co.il/car-insurance/tesla/",
                "https://www.fnx.co.il/car-insurance/comprehensive/insure-my-tesla/",
            ),
            "https://www.fnx.co.il/car-insurance/comprehensive/insure-my-tesla/",
        )

    def test_keeper_prefers_canonical_url(self):
        canon = "https://www.fnx.co.il/a/"
        urls = ["https://www.fnx.co.il/a/?from=tags", "https://www.fnx.co.il/a/"]
        self.assertEqual(pick_canonical_keeper(canon, urls), canon)


class PlanDuplicatesTests(unittest.TestCase):
    def test_canonical_alias_dropped(self):
        owner = "https://www.fnx.co.il/car-insurance/"
        alias = "https://www.fnx.co.il/tags/car/"
        marks = plan_duplicates([
            (owner, owner, "aaa"),
            (alias, owner, "bbb"),
        ])
        self.assertEqual(marks, [(alias, "duplicate_canonical", f"canonical={owner.rstrip('/')} keeper={owner}")])

    def test_identical_hash_second_url_dropped(self):
        a = "https://www.fnx.co.il/a/"
        b = "https://www.fnx.co.il/a/copy/"
        marks = plan_duplicates([
            (a, a, "samehash"),
            (b, b, "samehash"),
        ])
        self.assertEqual(len(marks), 1)
        self.assertEqual(marks[0][0], b)
        self.assertEqual(marks[0][1], "duplicate_hash")

    def test_canonical_wins_over_hash(self):
        owner = "https://www.fnx.co.il/a/"
        alias = "https://www.fnx.co.il/b/"
        marks = plan_duplicates([
            (owner, owner, "same"),
            (alias, owner, "same"),
        ])
        self.assertEqual([m[0] for m in marks], [alias])
        self.assertEqual(marks[0][1], "duplicate_canonical")

    def test_undefined_canonical_not_merged(self):
        a = "https://www.fnx.co.il/car-insurance/car_information/tips/"
        b = "https://www.fnx.co.il/savings/savings-information/investing-real-estate/"
        marks = plan_duplicates([
            (a, "undefined", "hash-a"),
            (b, "undefined", "hash-b"),
        ])
        self.assertEqual(marks, [])

    def test_empty_hash_not_merged(self):
        from phoenix_rag.crawler.filters import EMPTY_HASH

        marks = plan_duplicates([
            ("https://www.fnx.co.il/a/", None, EMPTY_HASH),
            ("https://www.fnx.co.il/b/", None, EMPTY_HASH),
        ])
        self.assertEqual(marks, [])


if __name__ == "__main__":
    unittest.main()
