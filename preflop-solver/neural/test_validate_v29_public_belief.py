import unittest

import validate_v29_public_belief as module


class V29GateTests(unittest.TestCase):
    def test_module_is_import_safe(self) -> None:
        self.assertTrue(callable(module.main))


if __name__ == "__main__":
    unittest.main()
