# Guards against keyboard mash inheriting insurance thread context (e.g. גשד → water-damage answer).
"""Run with: python tests/test_unintelligible.py"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phoenix_rag.answering.boundaries import detect_with_history  # noqa: E402
from phoenix_rag.answering.context import heuristic_rewrite, is_followup  # noqa: E402
from phoenix_rag.answering.sanitize import is_unintelligible_input  # noqa: E402

_LEAK_HISTORY = [
    {"role": "user", "content": "יש לי נזילה בבית, איך מגישים תביעה?"},
    {"role": "assistant", "content": "ניתן לפנות למוקד התביעות או למלא טופס נזק רכוש."},
]


class UnintelligibleTests(unittest.TestCase):
    def test_gibberish_after_thread(self):
        self.assertTrue(is_unintelligible_input("גשד"))

    def test_real_follow_up_not_gibberish(self):
        self.assertFalse(is_unintelligible_input("תוך כמה זמן מקבלים תשובה?"))
        self.assertFalse(is_unintelligible_input("הנזק הוא ממזג אוויר"))

    def test_not_follow_up_mash(self):
        self.assertFalse(is_followup("גשד", _LEAK_HISTORY))

    def test_rewrite_does_not_inject_slots(self):
        out = heuristic_rewrite("גשד", "גשד", _LEAK_HISTORY)
        self.assertEqual(out, "גשד")
        self.assertNotIn("נזקי מים", out)

    def test_boundary_stays_off_topic_with_history(self):
        bound = detect_with_history("גשד", "גשד", _LEAK_HISTORY)
        self.assertEqual(bound.kind, "out_of_scope")

    def test_quote_wrapper_user_mash(self):
        q = "בהתייחס לציטוט הבא מהתשובה: «כיסוי צנרת»\n\nגשד"
        self.assertTrue(is_unintelligible_input(q))


if __name__ == "__main__":
    unittest.main()
