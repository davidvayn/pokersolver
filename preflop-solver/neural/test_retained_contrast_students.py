import json
from pathlib import Path
import unittest
from unittest.mock import patch

import run_retained_contrast_students as module


class RetainedContrastContractTests(unittest.TestCase):
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
