from types import SimpleNamespace
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import run_action_contrast_students as module


class ContrastStudentContractTests(unittest.TestCase):
    def test_prepare_uses_existing_training_refresh_but_protects_frozen_eval_split(self):
        source, reference = {"source": True}, {"reference": True}
        args = SimpleNamespace(corpus="corpus", split_reference="reference")
        # Stop at the split boundary, before allocating features. Native split
        # tests separately verify byte-identical tuning/holdout and same family.
        with patch.object(module, "read_capture", side_effect=[source, reference]), \
             patch.object(module.native, "family_split", side_effect=ValueError("audit stop")) as split:
            with self.assertRaisesRegex(ValueError, "audit stop"): module.prepare(args)
        split.assert_called_once_with(source, 10601, .25, .25, reference=reference, refresh_training=True)

    def test_inconclusive_data_rejected_before_output_or_feature_allocation(self):
        arguments = ["run_action_contrast_students.py"]
        for name in ("binary", "corpus", "split-reference", "bundles"):
            arguments += ["--" + name, name, "--" + name + "-sha256", "a" * 64]
        arguments += ["--feature-cache", "cache", "--output", "output",
                      "--quality-decision", "decision", "--quality-decision-sha256", "a" * 64]
        decision = dict(schema="action-contrast-data-decision-v1", status="inconclusive",
                        primaryTrainingManifestSha256="a" * 64, releaseAccepted=False)
        with patch.object(module.sys, "argv", arguments), \
             patch.object(module, "sha256", return_value="a" * 64), \
             patch.object(Path, "read_text", return_value=json.dumps(decision)), \
             patch.object(Path, "mkdir") as mkdir, patch.object(module, "prepare") as prepare:
            with self.assertRaisesRegex(ValueError, "inconclusive or mismatched"):
                module.main()
        mkdir.assert_not_called()
        prepare.assert_not_called()


if __name__ == "__main__": unittest.main()
