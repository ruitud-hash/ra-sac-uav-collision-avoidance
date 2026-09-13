"""Contract and leakage tests for the formal OOD evaluation suite."""

from __future__ import annotations

import copy
import unittest
from pathlib import Path

import yaml

from envs.uav_2d_env import UAV2DEnv
from utils.experiment_protocol import (
    ensure_training_environment_allowed,
    evaluation_protocol_metadata,
)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_NAME = "env_medium_v3_unified_id.yaml"
OOD_NAMES = {
    "disturbance": "env_medium_v3_disturbance_ood.yaml",
    "behavior": "env_medium_v3_behavior_ood.yaml",
    "composition": "env_medium_v3_composition_ood.yaml",
    "combined": "env_medium_v3_combined_ood.yaml",
}


def load_config(name: str) -> dict:
    with (ROOT / "configs" / name).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def behavior_without_probabilities(config: dict) -> dict:
    profiles = copy.deepcopy(config["obstacles"]["heterogeneous_dynamic"]["types"])
    for profile in profiles.values():
        profile.pop("probability")
    return profiles


def scenarios_without_counts(config: dict) -> dict:
    scenarios = copy.deepcopy(config["obstacles"]["nonlinear_scenarios"]["scenarios"])
    for scenario in scenarios.values():
        scenario.pop("count")
    return scenarios


class OODEnvironmentSuiteTests(unittest.TestCase):
    def test_all_ood_configs_keep_the_formal_observation_contract(self) -> None:
        for axis, name in OOD_NAMES.items():
            with self.subTest(axis=axis):
                config = load_config(name)
                env = UAV2DEnv(config)
                observation = env.reset(seed=config["seed"])
                protocol = config["protocol"]

                self.assertEqual(observation["tokens"].shape, (48, 15))
                self.assertEqual(tuple(env.dynamic_type_names), env.DYNAMIC_TYPE_ORDER)
                self.assertEqual(protocol["name"], "unified_15d_ood")
                self.assertEqual(protocol["split"], "ood")
                self.assertEqual(protocol["ood_axis"], axis)
                self.assertTrue(protocol["evaluation_only"])
                self.assertFalse(protocol["checkpoint_selection_eligible"])

    def test_disturbance_ood_only_changes_disturbance_distribution(self) -> None:
        reference = load_config(REFERENCE_NAME)
        config = load_config(OOD_NAMES["disturbance"])

        self.assertEqual(config["obstacles"], reference["obstacles"])
        self.assertEqual(config["risk"], reference["risk"])
        self.assertEqual(config["reward"], reference["reward"])
        self.assertNotEqual(config["domain_randomization"], reference["domain_randomization"])
        self.assertGreaterEqual(
            config["domain_randomization"]["wind"]["base_speed_mps"][0],
            reference["domain_randomization"]["wind"]["base_speed_mps"][1],
        )
        self.assertGreater(
            config["domain_randomization"]["control_delay_steps"][0],
            reference["domain_randomization"]["control_delay_steps"][1],
        )

    def test_behavior_ood_keeps_id_disturbances_and_composition(self) -> None:
        reference = load_config(REFERENCE_NAME)
        config = load_config(OOD_NAMES["behavior"])

        self.assertEqual(config["domain_randomization"], reference["domain_randomization"])
        self.assertNotEqual(
            behavior_without_probabilities(config),
            behavior_without_probabilities(reference),
        )
        self.assertEqual(
            {
                name: values["count"]
                for name, values in config["obstacles"]["nonlinear_scenarios"]["scenarios"].items()
            },
            {
                name: values["count"]
                for name, values in reference["obstacles"]["nonlinear_scenarios"]["scenarios"].items()
            },
        )

    def test_composition_ood_keeps_id_disturbances_and_behavior(self) -> None:
        reference = load_config(REFERENCE_NAME)
        config = load_config(OOD_NAMES["composition"])

        self.assertEqual(config["domain_randomization"], reference["domain_randomization"])
        self.assertEqual(
            behavior_without_probabilities(config),
            behavior_without_probabilities(reference),
        )
        self.assertEqual(
            scenarios_without_counts(config),
            scenarios_without_counts(reference),
        )
        self.assertNotEqual(
            config["obstacles"]["heterogeneous_dynamic"]["types"],
            reference["obstacles"]["heterogeneous_dynamic"]["types"],
        )

    def test_training_and_checkpoint_selection_guard_rejects_ood(self) -> None:
        ensure_training_environment_allowed(load_config(REFERENCE_NAME))
        for axis, name in OOD_NAMES.items():
            with self.subTest(axis=axis):
                with self.assertRaisesRegex(ValueError, "checkpoint selection"):
                    ensure_training_environment_allowed(load_config(name))

    def test_ood_protocol_rejects_selection_eligible_configuration(self) -> None:
        config = load_config(OOD_NAMES["combined"])
        config["protocol"]["checkpoint_selection_eligible"] = True

        with self.assertRaisesRegex(ValueError, "checkpoint_selection_eligible=false"):
            UAV2DEnv(config)

    def test_evaluation_artifact_metadata_marks_ood_as_report_only(self) -> None:
        metadata = evaluation_protocol_metadata(load_config(OOD_NAMES["combined"]))

        self.assertEqual(metadata["split"], "ood")
        self.assertEqual(metadata["ood_axis"], "combined")
        self.assertTrue(metadata["evaluation_only"])
        self.assertFalse(metadata["checkpoint_selection_eligible"])


if __name__ == "__main__":
    unittest.main()
