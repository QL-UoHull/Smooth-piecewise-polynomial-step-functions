import unittest

from smooth_steps.compare import comparison_rows, render_report


class ComparisonReportTests(unittest.TestCase):
    def test_comparison_rows_capture_expected_strengths(self) -> None:
        rows = comparison_rows((2,))
        by_name = {row["name"]: row for row in rows}

        self.assertEqual(by_name["smooth_piecewise_polynomial(order=2)"]["compact_support"], "yes")
        self.assertEqual(by_name["smooth_piecewise_polynomial(order=2)"]["endpoint_flatness"], "2")
        self.assertEqual(by_name["logistic_sigmoid"]["compact_support"], "no")
        self.assertEqual(by_name["cubic_smoothstep"]["endpoint_flatness"], "1")

    def test_report_mentions_repository_strengths(self) -> None:
        report = render_report((2, 4))
        self.assertIn("smooth_piecewise_polynomial(order=4)", report)
        self.assertIn("quintic_smootherstep", report)
        self.assertIn("logistic_sigmoid", report)
        self.assertIn("Key takeaways", report)
        self.assertIn("special cases", report)


if __name__ == "__main__":
    unittest.main()
