"""Regression tests for unambiguous checkpoint-evaluation artifacts."""

from __future__ import annotations

import unittest

from utils.naming import (
    algorithm_display_name,
    checkpoint_role,
    precise_timestamp,
)


class EvaluationArtifactNamingTests(unittest.TestCase):
    def test_ra_sac_v3_is_not_labeled_as_sac_attention(self) -> None:
        self.assertEqual(
            algorithm_display_name("ra_sac_v3_medium_seed_18207"),
            "RA-SAC-v3",
        )
        self.assertEqual(
            algorithm_display_name("sac_attention_v3_medium_seed_18207"),
            "SAC-Attention",
        )

    def test_checkpoint_role_is_preserved(self) -> None:
        self.assertEqual(checkpoint_role("run_best.pt"), "best")
        self.assertEqual(checkpoint_role("run_final.pt"), "final")
        self.assertEqual(checkpoint_role("run_step_500000.pt"), "step_500000")

    def test_precise_timestamp_contains_microseconds(self) -> None:
        value = precise_timestamp()
        self.assertRegex(value, r"^\d{8}_\d{6}_\d{6}$")


if __name__ == "__main__":
    unittest.main()
