import unittest

import numpy as np

from openbonsai.quantize import (
    binary_rtn,
    closed_form_scale,
    curvature_coordinate_descent,
    ternary_rtn,
)


class QuantizeTests(unittest.TestCase):
    def test_binary_scale_is_mean_absolute(self):
        values = np.linspace(-2, 3, 128, dtype=np.float32)
        quantized, scales, codes = binary_rtn(values)
        self.assertAlmostEqual(float(scales[0]), float(np.mean(np.abs(values))), places=6)
        self.assertTrue(np.all(np.isin(codes, [-1, 1])))
        np.testing.assert_allclose(quantized, codes.reshape(-1) * scales[0])

    def test_ternary_alphabet(self):
        values = np.concatenate([np.zeros(64), np.ones(32), -np.ones(32)]).astype(np.float32)
        _, _, codes = ternary_rtn(values)
        self.assertEqual(set(np.unique(codes)), {-1, 0, 1})

    def test_closed_form_scale(self):
        w = np.asarray([2.0, -1.0])
        z = np.asarray([1.0, -1.0])
        self.assertAlmostEqual(closed_form_scale(w, z), 1.5)

    def test_coordinate_descent_does_not_worsen_curvature_loss(self):
        rng = np.random.default_rng(7)
        w = rng.normal(size=8)
        a = rng.normal(size=(8, 8))
        h = a.T @ a + np.eye(8) * 0.1
        initial = np.where(w >= 0, 1, -1)
        initial_scale = closed_form_scale(w, initial, h)
        delta = w - initial_scale * initial
        initial_loss = float(delta @ h @ delta)
        _, _, final_loss = curvature_coordinate_descent(w, h, (-1, 1))
        self.assertLessEqual(final_loss, initial_loss + 1e-10)


if __name__ == "__main__":
    unittest.main()

