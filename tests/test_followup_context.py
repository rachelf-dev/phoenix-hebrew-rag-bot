# Follow-up rewrite, slot inheritance, and component-scoped topic gate tests.
"""Run with: python tests/test_followup_context.py"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phoenix_rag.answering.context import (  # noqa: E402
    claim_path_markers,
    extract_thread_slots,
    heuristic_rewrite,
    retrieval_query,
)
from phoenix_rag.answering.rerank import path_matches_question  # noqa: E402
from phoenix_rag.indexing.retrieve import Hit  # noqa: E402

SOLAR_HISTORY = [
    {"role": "user", "content": "נשבר לי דוד השמש, מה עושים?"},
    {
        "role": "assistant",
        "content": "במידה והנזק לדוד השמש אינו כתוצאה מנזקי מזג אויר ונרכשה הרחבה לדוודים, יש לפנות לנותן השירות.",
    },
]
WEATHER_FU = "הנזק הוא ממזג אויר"

APT_WH = "https://www.fnx.co.il/claims-customer-service/claims-appartment-insurance/water-heater/"
APT_PIPE = "https://www.fnx.co.il/claims-customer-service/claims-appartment-insurance/pipeline-damage/"
NATAV = "https://natav.net/natav_online"


class SlotInheritanceTests(unittest.TestCase):
    def test_weather_followup_keeps_solar_heater_slots(self):
        slots = extract_thread_slots(SOLAR_HISTORY, WEATHER_FU)
        self.assertIn("דוד שמש", slots.damages)
        self.assertIn("נזקי מזג אוויר", slots.damages)
        self.assertEqual(slots.product, "ביטוח דירה")

    def test_assistant_plumber_text_does_not_add_plumber_coverage(self):
        hist = SOLAR_HISTORY + [
            {
                "role": "assistant",
                "content": "במידה ורכשת כיסוי שרברב בהסדר ונגרם נזק מהצנרת של הדוד, פתחו קריאה.",
            }
        ]
        slots = extract_thread_slots(hist, WEATHER_FU)
        self.assertNotIn("שרברב הסדר", (slots.coverage,))


class RewriteTests(unittest.TestCase):
    def test_heuristic_rewrite_weather_after_solar(self):
        out = heuristic_rewrite(WEATHER_FU, WEATHER_FU, SOLAR_HISTORY)
        self.assertIn("דוד", out)
        self.assertIn("מזג", out)


class RetrievalQueryTests(unittest.TestCase):
    def test_quote_followup_uses_prior_user_not_bot_quote(self):
        quoted = (
            "בהתייחס לציטוט הבא מהתשובה: «במידה והנזק לדוד השמש אינו כתוצאה מנזקי מזג אויר»\n\n"
            "הנזק הוא ממזג אויר"
        )
        q = retrieval_query(quoted, SOLAR_HISTORY)
        self.assertIn("נשבר", q)
        self.assertIn("ממזג", q)
        self.assertNotIn("אינו כתוצאה", q)


class MarkerAndGateTests(unittest.TestCase):
    def test_markers_water_heater_only_for_weather_thread(self):
        markers = claim_path_markers(WEATHER_FU, SOLAR_HISTORY)
        self.assertTrue(any("water-heater" in m for m in markers))
        self.assertFalse(any("pipeline-damage" in m for m in markers))

    def test_pipeline_blocked_for_weather_followup(self):
        self.assertFalse(path_matches_question(WEATHER_FU, APT_PIPE, SOLAR_HISTORY))

    def test_water_heater_allowed(self):
        self.assertTrue(path_matches_question(WEATHER_FU, APT_WH, SOLAR_HISTORY))

    def test_external_plumber_link_blocked(self):
        self.assertFalse(path_matches_question(WEATHER_FU, NATAV, SOLAR_HISTORY))


if __name__ == "__main__":
    unittest.main()
