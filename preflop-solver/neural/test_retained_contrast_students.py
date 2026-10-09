import json
from pathlib import Path
import unittest
from unittest.mock import patch

import run_retained_contrast_students as module


class RetainedContrastContractTests(unittest.TestCase):
    def test_counterfactual_weight_pilot_requires_only_the_matched_aligned_control(self):
        settings = dict(fixedFinalStep=600)
        corpus, bundles = str(Path("corpus").resolve()), str(Path("bundles").resolve())
        control = dict(status="complete", schema="retained-initialization-contrast-pilot-v1",
            armDifference="serving_contrast_forward_vjp_only", servingControlSha256="protected",
            matmulPrecision="full-float32", fitSettings=settings,
            protectionSettingsSha256="retention", pinnedInputs={corpus: "corpus-sha", bundles: "bundles-sha"})
        args = ("protected", settings, (corpus, "corpus-sha"), (bundles, "bundles-sha"), "retention")
        module.check_counterfactual_control(control, *args)
        for field, value in (("nativeCounterfactualFraction", .5),
                             ("protectionSettingsSha256", "changed"),
                             ("initializationTransform", "wide-to-wide-pooled-zero-columns-v1")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                module.check_counterfactual_control(dict(control, **{field: value}), *args)

    def test_loader_changes_only_the_explicit_calibration_weight_argument(self):
        from types import SimpleNamespace
        import run_action_contrast_students as base
        path = Path("source")
        for fraction, args in ((.1, SimpleNamespace()), (.5, SimpleNamespace(native_counterfactual_fraction=.5))):
            with patch.object(base.training, "load_dataset", return_value="dataset") as loader:
                self.assertEqual(base.load_training_dataset(args, path), "dataset")
                loader.assert_called_once_with(path, 1, "payoff-exposure", native_counterfactual_fraction=fraction)

    def test_pooling_requires_matched_serving_control_not_a_scratch_or_protection_arm(self):
        settings = dict(fixedFinalStep=600)
        corpus, bundles = str(Path("corpus").resolve()), str(Path("bundles").resolve())
        control = dict(status="complete", schema="retained-initialization-contrast-pilot-v1",
            armDifference="serving_contrast_forward_vjp_only", servingControlSha256="protected",
            matmulPrecision="full-float32", fitSettings=settings,
            pinnedInputs={corpus: "corpus-sha", bundles: "bundles-sha"})
        args = ("protected", settings, (corpus, "corpus-sha"), (bundles, "bundles-sha"))
        module.check_pooling_control(control, *args)
        for field, value in (("status", "running"), ("armDifference", "retained_value_protection_only"),
                             ("servingControlSha256", "wrong"), ("fitSettings", {}),
                             ("matmulPrecision", "default"), ("pinnedInputs", {})):
            with self.subTest(field=field), self.assertRaises(ValueError):
                module.check_pooling_control(dict(control, **{field: value}), *args)

    def test_frozen_settings_reject_hash_or_schedule_changes(self):
        settings = dict(status="complete", fixedFinalStep=600, cadence=4, pairedSeeds=[10601, 10602])
        with patch.object(module, "sha256", return_value=module.SETTINGS_SHA), \
             patch.object(Path, "read_text", return_value=json.dumps(settings)):
            self.assertEqual(module.checked_settings(Path("settings")), settings)
        with patch.object(module, "sha256", return_value="changed"):
            with self.assertRaisesRegex(ValueError, "unchanged"):
                module.checked_settings(Path("settings"))
        for field, value in (("fixedFinalStep", 400), ("cadence", 8), ("pairedSeeds", [10601])):
            wrong = {**settings, field: value}
            with patch.object(module, "sha256", return_value=module.SETTINGS_SHA), \
                 patch.object(Path, "read_text", return_value=json.dumps(wrong)), \
                 self.assertRaisesRegex(ValueError, "schedule"):
                module.checked_settings(Path("settings"))


if __name__ == "__main__": unittest.main()
