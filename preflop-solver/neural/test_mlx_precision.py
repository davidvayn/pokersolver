import unittest

from mlx_precision import matmul_precision


class MLXPrecisionTests(unittest.TestCase):
    def test_only_explicit_full_precision_changes_legacy_math_identity(self):
        self.assertIsNone(matmul_precision({}))
        self.assertIsNone(matmul_precision({'MLX_ENABLE_TF32':'1'}))
        self.assertEqual(matmul_precision({'MLX_ENABLE_TF32':'0'}),'full-fp32-v1')
        for value in ('', 'false', 'true', '2', False):
            with self.assertRaises(ValueError):
                matmul_precision({'MLX_ENABLE_TF32':value})
