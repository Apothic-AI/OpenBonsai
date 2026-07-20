import unittest

import numpy as np

from openbonsai.matrix_lab import evaluate_matrix


class MatrixLabTests(unittest.TestCase):
    def test_smoke(self):
        rng = np.random.default_rng(4)
        weight = rng.normal(size=(2, 128)).astype(np.float32)
        activations = rng.normal(size=(64, 128)).astype(np.float32)
        result = evaluate_matrix(weight, activations)
        self.assertIn("binary_rtn", result)
        self.assertIn("first_row_curvature_ternary", result)
        self.assertTrue(np.isfinite(result["binary_rtn"]["output_relative_error"]))


if __name__ == "__main__":
    unittest.main()

