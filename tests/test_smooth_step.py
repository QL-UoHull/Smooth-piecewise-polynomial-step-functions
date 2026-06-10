"""Tests for smooth step function implementations."""

import unittest

import numpy as np
from numpy.testing import assert_allclose

from smooth_step.classical import smooth_step as classical_step
from smooth_step.recursive import smooth_step as recursive_step


class TestRecursiveSmoothStep(unittest.TestCase):
    """Tests for the recursive piecewise polynomial smooth step function."""

    def test_boundary_values(self):
        """f(0) = 0 and f(1) = 1 for all tested orders."""
        for n in range(5):
            with self.subTest(order=n):
                assert_allclose(recursive_step(0.0, n), 0.0, atol=1e-12,
                                err_msg=f"f(0) != 0 for order {n}")
                assert_allclose(recursive_step(1.0, n), 1.0, atol=1e-12,
                                err_msg=f"f(1) != 1 for order {n}")

    def test_midpoint_symmetry(self):
        """f(0.5) = 0.5 for all tested orders (anti-symmetry about (0.5, 0.5))."""
        for n in range(6):
            with self.subTest(order=n):
                assert_allclose(recursive_step(0.5, n), 0.5, atol=1e-12,
                                err_msg=f"f(0.5) != 0.5 for order {n}")

    def test_antisymmetry(self):
        """f(x) + f(1-x) = 1 (point symmetry about (0.5, 0.5))."""
        x = np.linspace(0.0, 1.0, 51)
        for n in range(5):
            with self.subTest(order=n):
                assert_allclose(
                    recursive_step(x, n) + recursive_step(1.0 - x, n),
                    np.ones_like(x),
                    atol=1e-12,
                    err_msg=f"Antisymmetry failed for order {n}",
                )

    def test_monotonicity(self):
        """f is non-decreasing on [0, 1]."""
        x = np.linspace(0.0, 1.0, 1001)
        for n in range(5):
            with self.subTest(order=n):
                values = recursive_step(x, n)
                diffs = np.diff(values)
                self.assertTrue(
                    np.all(diffs >= -1e-12),
                    msg=f"Non-monotone for order {n}: min diff = {diffs.min()}",
                )

    def test_range(self):
        """f values lie in [0, 1]."""
        x = np.linspace(0.0, 1.0, 100)
        for n in range(5):
            with self.subTest(order=n):
                values = recursive_step(x, n)
                self.assertTrue(np.all(values >= 0.0),
                                msg=f"Values below 0 for order {n}")
                self.assertTrue(np.all(values <= 1.0),
                                msg=f"Values above 1 for order {n}")

    def test_order_1_exact_values(self):
        """Order-1 function matches its explicit piecewise-quadratic formula."""
        # Piece 1, x in [0, 0.5]: 2x^2
        x1 = np.array([0.0, 0.25, 0.5])
        expected1 = 2.0 * x1 ** 2
        assert_allclose(recursive_step(x1, 1), expected1, atol=1e-12)

        # Piece 2, x in [0.5, 1]: -2x^2 + 4x - 1
        x2 = np.array([0.5, 0.75, 1.0])
        expected2 = -2.0 * x2 ** 2 + 4.0 * x2 - 1.0
        assert_allclose(recursive_step(x2, 1), expected2, atol=1e-12)

    def test_order_2_exact_boundary_piece(self):
        """Order-2, first piece: 9x^3/2 on [0, 1/3]."""
        x = np.linspace(0.0, 1.0 / 3.0, 10)
        expected = 4.5 * x ** 3
        assert_allclose(recursive_step(x, 2), expected, atol=1e-12)

    def test_vector_input(self):
        """Function preserves array shape."""
        x = np.linspace(0.0, 1.0, 20)
        result = recursive_step(x, 2)
        self.assertEqual(result.shape, x.shape)

    def test_scalar_input(self):
        """Function accepts scalar input."""
        result = recursive_step(0.5, 2)
        self.assertAlmostEqual(float(result), 0.5, places=12)

    def test_invalid_order(self):
        """Negative order raises ValueError."""
        with self.assertRaises(ValueError):
            recursive_step(0.5, -1)

    def test_zero_derivative_at_boundaries(self):
        """Numerical first derivative ≈ 0 at both endpoints for order >= 1."""
        h = 1e-5
        for n in range(1, 5):
            with self.subTest(order=n):
                deriv_left = (recursive_step(h, n) - recursive_step(0.0, n)) / h
                deriv_right = (recursive_step(1.0, n) - recursive_step(1.0 - h, n)) / h
                self.assertAlmostEqual(
                    float(deriv_left), 0.0, places=4,
                    msg=f"Left derivative not ~0 for order {n}",
                )
                self.assertAlmostEqual(
                    float(deriv_right), 0.0, places=4,
                    msg=f"Right derivative not ~0 for order {n}",
                )


class TestClassicalSmoothStep(unittest.TestCase):
    """Tests for the classical polynomial smooth step function."""

    def test_boundary_values(self):
        """f(0) = 0 and f(1) = 1 for all tested orders."""
        for n in range(1, 6):
            with self.subTest(order=n):
                assert_allclose(classical_step(0.0, n), 0.0, atol=1e-12,
                                err_msg=f"f(0) != 0 for order {n}")
                assert_allclose(classical_step(1.0, n), 1.0, atol=1e-12,
                                err_msg=f"f(1) != 1 for order {n}")

    def test_midpoint_symmetry(self):
        """f(0.5) = 0.5 for all tested orders."""
        for n in range(1, 6):
            with self.subTest(order=n):
                assert_allclose(classical_step(0.5, n), 0.5, atol=1e-12)

    def test_cubic_smoothstep(self):
        """Order 1: matches 3x^2 - 2x^3 (cubic smoothstep)."""
        x = np.linspace(0.0, 1.0, 21)
        expected = 3.0 * x ** 2 - 2.0 * x ** 3
        assert_allclose(classical_step(x, 1), expected, atol=1e-12)

    def test_quintic_smootherstep(self):
        """Order 2: matches 10x^3 - 15x^4 + 6x^5 (quintic smootherstep)."""
        x = np.linspace(0.0, 1.0, 21)
        expected = 10.0 * x ** 3 - 15.0 * x ** 4 + 6.0 * x ** 5
        assert_allclose(classical_step(x, 2), expected, atol=1e-12)

    def test_degree_7_smoothstep(self):
        """Order 3: matches 35x^4 - 84x^5 + 70x^6 - 20x^7."""
        x = np.linspace(0.0, 1.0, 21)
        expected = 35.0 * x ** 4 - 84.0 * x ** 5 + 70.0 * x ** 6 - 20.0 * x ** 7
        assert_allclose(classical_step(x, 3), expected, atol=1e-12)

    def test_monotonicity(self):
        """f is non-decreasing on [0, 1]."""
        x = np.linspace(0.0, 1.0, 1001)
        for n in range(1, 6):
            with self.subTest(order=n):
                values = classical_step(x, n)
                diffs = np.diff(values)
                self.assertTrue(
                    np.all(diffs >= -1e-12),
                    msg=f"Non-monotone for order {n}: min diff = {diffs.min()}",
                )

    def test_range(self):
        """f values lie in [0, 1]."""
        x = np.linspace(0.0, 1.0, 100)
        for n in range(1, 6):
            with self.subTest(order=n):
                values = classical_step(x, n)
                self.assertTrue(np.all(values >= 0.0))
                self.assertTrue(np.all(values <= 1.0))

    def test_invalid_order(self):
        """Order 0 raises ValueError."""
        with self.assertRaises(ValueError):
            classical_step(0.5, 0)

    def test_zero_derivative_at_boundaries(self):
        """Numerical first derivative ≈ 0 at both endpoints."""
        h = 1e-6
        for n in range(1, 5):
            with self.subTest(order=n):
                deriv_left = (classical_step(h, n) - classical_step(0.0, n)) / h
                deriv_right = (classical_step(1.0, n) - classical_step(1.0 - h, n)) / h
                self.assertAlmostEqual(float(deriv_left), 0.0, places=4)
                self.assertAlmostEqual(float(deriv_right), 0.0, places=4)


class TestComparison(unittest.TestCase):
    """Cross-checks between recursive and classical constructions."""

    def test_same_endpoints(self):
        """Both constructions agree at x = 0, 0.5, and 1 for all orders."""
        for n in range(1, 5):
            with self.subTest(order=n):
                for xi in [0.0, 0.5, 1.0]:
                    r = float(recursive_step(xi, n))
                    c = float(classical_step(xi, n))
                    self.assertAlmostEqual(r, c, places=12,
                                          msg=f"Mismatch at x={xi}, order={n}")

    def test_recursive_uses_lower_degree(self):
        """Recursive and classical produce distinct shapes in the interior.

        Both constructions agree at x=0, 0.5, and 1, but differ elsewhere
        because they use polynomials of different degrees. This verifies that
        the implementations are genuinely distinct rather than numerically
        identical.
        """
        x = np.linspace(0.1, 0.4, 20)  # avoid endpoints and midpoint
        r = recursive_step(x, 2)
        c = classical_step(x, 2)
        diff = np.abs(r - c)
        # All interior points should differ by more than floating-point noise
        self.assertTrue(
            np.all(diff > 1e-6),
            msg="Recursive and classical should differ at all interior points",
        )
        # Maximum difference should be substantial (order 0.01+)
        self.assertGreater(
            float(np.max(diff)), 0.01,
            msg="Maximum difference between constructions is unexpectedly small",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
