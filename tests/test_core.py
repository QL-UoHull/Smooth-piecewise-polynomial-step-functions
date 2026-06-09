import unittest

from smooth_steps.core import (
    cubic_smoothstep,
    generalized_smoothstep,
    quintic_smootherstep,
)


class GeneralizedSmoothstepTests(unittest.TestCase):
    def test_order_one_matches_cubic_smoothstep(self) -> None:
        for sample in (0.0, 0.2, 0.5, 0.8, 1.0):
            self.assertAlmostEqual(generalized_smoothstep(sample, 1), cubic_smoothstep(sample))

    def test_order_two_matches_quintic_smootherstep(self) -> None:
        for sample in (0.0, 0.2, 0.5, 0.8, 1.0):
            self.assertAlmostEqual(generalized_smoothstep(sample, 2), quintic_smootherstep(sample))

    def test_family_is_symmetric(self) -> None:
        for sample in (0.1, 0.25, 0.4):
            self.assertAlmostEqual(
                generalized_smoothstep(sample, 4),
                1.0 - generalized_smoothstep(1.0 - sample, 4),
            )

    def test_values_are_clamped(self) -> None:
        self.assertEqual(generalized_smoothstep(-1.0, 3), 0.0)
        self.assertEqual(generalized_smoothstep(2.0, 3), 1.0)


if __name__ == "__main__":
    unittest.main()
