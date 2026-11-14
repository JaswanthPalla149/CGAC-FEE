import unittest

from analyze_benchmark import compare_runs


def run(configuration, ate, failures, latency, memory):
    return {
        "sequence": "sequence_a",
        "degradation": "blur",
        "configuration": configuration,
        "ate_rmse_similarity_aligned": str(ate),
        "tracking_failure_rate": str(failures),
        "mean_latency_ms": str(latency),
        "peak_rss_bytes": str(memory),
    }


class CompareRunsTests(unittest.TestCase):
    def test_adaptive_dominates_when_all_compared_metrics_improve(self):
        runs = {
            ("sequence_a", "blur", "adaptive"): run("adaptive", 0.8, 0.1, 5, 100),
            ("sequence_a", "blur", "fixed_low"): run("fixed_low", 1.0, 0.2, 6, 110),
            ("sequence_a", "blur", "fixed_high"): run("fixed_high", 0.9, 0.15, 8, 130),
        }

        results = compare_runs(runs)

        self.assertEqual([row["result"] for row in results],
                         ["adaptive_dominates", "adaptive_dominates"])

    def test_reports_tradeoff_and_ignores_missing_ground_truth_metrics(self):
        adaptive = run("adaptive", float("nan"), 0.1, 7, 100)
        baseline = run("fixed_low", float("nan"), 0.2, 5, 110)
        runs = {
            ("sequence_a", "blur", "adaptive"): adaptive,
            ("sequence_a", "blur", "fixed_low"): baseline,
        }

        result = compare_runs(runs)[0]

        self.assertEqual(result["result"], "tradeoff")
        self.assertNotIn("ate_rmse_similarity_aligned", result["metrics_compared"])
        self.assertEqual(result["delta_mean_latency_ms"], 2.0)

    def test_omits_unavailable_negative_memory_sentinel(self):
        runs = {
            ("sequence_a", "blur", "adaptive"): run(
                "adaptive", float("nan"), 0.1, 5, -1
            ),
            ("sequence_a", "blur", "fixed_low"): run(
                "fixed_low", float("nan"), 0.2, 5, -1
            ),
        }

        result = compare_runs(runs)[0]

        self.assertEqual(result["result"], "adaptive_dominates")
        self.assertNotIn("peak_rss_bytes", result["metrics_compared"])


if __name__ == "__main__":
    unittest.main()
