import json
from pathlib import Path
import tempfile
import unittest

from flop_patch_pilot import matching_reports


class FrozenPanelTests(unittest.TestCase):
    def test_reports_cannot_cross_checkpoint_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f'{i}.json' for i in range(3)]
            for path, digest in zip(paths, ['a', 'b', 'a']):
                path.write_text(json.dumps({'policy_sha256': digest}))
            self.assertEqual(matching_reports(paths, 'a'), [paths[0], paths[2]])
            self.assertEqual(matching_reports(paths, 'c'), [])

    def test_missing_identity_is_not_a_fallback_opponent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'invalid.json'
            path.write_text('{}')
            with self.assertRaises(KeyError):
                matching_reports([path], 'a')


if __name__ == '__main__':
    unittest.main()
