"""Tests for formal periodic-evaluation curve aggregation."""

from __future__ import annotations

import unittest

import numpy as np

from utils.evaluation_curves import (
    aggregate_seed_curves,
    aggregate_seed_series,
    moving_average,
)


class EvaluationCurveTests(unittest.TestCase):
    def test_centered_moving_average_preserves_curve_length(self) -> None:
        values = np.array([0.0, 3.0, 6.0, 9.0, 12.0])

        smoothed = moving_average(values, 3)

        np.testing.assert_allclose(smoothed, [1.5, 3.0, 6.0, 9.0, 10.5])

    def test_aggregate_computes_standard_error_after_per_seed_smoothing(self) -> None:
        rows_by_seed = []
        for offset in (0.0, 1.0, 2.0, 3.0, 4.0):
            rows_by_seed.append(
                [
                    {"step": 100.0, "eval_success_rate": 0.1 + offset * 0.01},
                    {"step": 200.0, "eval_success_rate": 0.4 + offset * 0.01},
                    {"step": 300.0, "eval_success_rate": 0.7 + offset * 0.01},
                ]
            )

        curve = aggregate_seed_curves(
            rows_by_seed,
            "eval_success_rate",
            smooth_window=3,
        )

        self.assertEqual(curve["seed_count"][0], 5)
        self.assertEqual(curve["steps"].tolist(), [100.0, 200.0, 300.0])
        expected_seed_std = np.std([0.0, 0.01, 0.02, 0.03, 0.04], ddof=1)
        np.testing.assert_allclose(
            curve["standard_error"],
            np.full(3, expected_seed_std / np.sqrt(5)),
        )

    def test_invalid_smoothing_window_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            moving_average(np.array([1.0, 2.0]), 7)

    def test_episode_reward_supports_long_smoothing_windows(self) -> None:
        rows_by_seed = [
            [
                {"episode": float(index), "episode_reward": float(index + seed)}
                for index in range(120)
            ]
            for seed in range(5)
        ]

        curve = aggregate_seed_series(
            rows_by_seed,
            x_key="episode",
            metric="episode_reward",
            smooth_window=50,
        )

        self.assertEqual(curve["steps"].shape, (120,))
        self.assertEqual(curve["mean"].shape, (120,))
        self.assertTrue(np.isfinite(curve["standard_error"]).all())

    def test_even_window_uses_exactly_fifty_interior_points(self) -> None:
        values = np.zeros(100)
        values[25:75] = 1.0

        smoothed = moving_average(values, 50)

        self.assertAlmostEqual(smoothed[50], 0.98)


if __name__ == "__main__":
    unittest.main()
