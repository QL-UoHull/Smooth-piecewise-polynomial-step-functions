import unittest

from smooth_steps.compare import render_report


class ComparisonReportTests(unittest.TestCase):
    def test_report_mentions_repository_strengths(self) -> None:
        report = render_report((2, 4))
        self.assertIn("smooth_piecewise_polynomial(order=4)", report)
        self.assertIn("quintic_smootherstep", report)
        self.assertIn("logistic_sigmoid", report)
        self.assertIn("Key takeaways", report)
        self.assertIn("special cases", report)


if __name__ == "__main__":
    unittest.main()
