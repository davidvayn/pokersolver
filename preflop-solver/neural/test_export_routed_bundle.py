import unittest

from export_routed_bundle import parse_args


class ExportRoutedBundleTests(unittest.TestCase):
    def test_entrypoint_is_importable(self):
        self.assertTrue(callable(parse_args))


if __name__ == "__main__":
    unittest.main()
