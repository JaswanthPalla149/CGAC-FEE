import sys
import tempfile
import unittest
from pathlib import Path
from typing import Tuple
from unittest import mock

import generate_synthetic_euroc as synthetic


class SyntheticTrajectoryTests(unittest.TestCase):
    def generate_sequence(self, output: Path, seed: int) -> Tuple[str, str]:
        arguments = [
            "generate_synthetic_euroc.py",
            "--output",
            str(output),
            "--frames",
            "4",
            "--width",
            "64",
            "--height",
            "48",
            "--seed",
            str(seed),
        ]
        with mock.patch.object(sys, "argv", arguments):
            synthetic.main()

        ground_truth = (
            output / "mav0" / "state_groundtruth_estimate0" / "data.csv"
        ).read_text(encoding="ascii")
        manifest = (output / "EcoLIVO_Retail_Street.bag").read_text(
            encoding="ascii"
        )
        return ground_truth, manifest

    def test_seed_reproduces_path_and_different_seed_changes_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path_a, manifest_a = self.generate_sequence(root / "a", 42)
            path_a_repeat, _ = self.generate_sequence(root / "a_repeat", 42)
            path_b, _ = self.generate_sequence(root / "b", 99)

        self.assertEqual(path_a, path_a_repeat)
        self.assertNotEqual(path_a, path_b)
        self.assertIn("seed=42\n", manifest_a)

    def test_missing_seed_uses_and_records_random_seed(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "random"
            arguments = [
                "generate_synthetic_euroc.py",
                "--output",
                str(output),
                "--frames",
                "2",
                "--width",
                "64",
                "--height",
                "48",
            ]
            with mock.patch.object(sys, "argv", arguments), mock.patch.object(
                synthetic.secrets, "randbits", return_value=8675309
            ):
                synthetic.main()

            manifest = (output / "EcoLIVO_Retail_Street.bag").read_text(
                encoding="ascii"
            )

        self.assertIn("seed=8675309\n", manifest)

    def test_negative_seed_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "seed must be non-negative"):
            synthetic.create_trajectory_parameters(-1)

    def test_closed_trajectory_returns_to_start(self):
        for frame_count in (1, 2, 60, 300):
            _, parameters = synthetic.create_trajectory_parameters(
                seed=42, closed_loop=True, total_frames=frame_count
            )
            start = synthetic.trajectory_position(0, parameters)
            end = synthetic.trajectory_position(frame_count - 1, parameters)
            for start_coordinate, end_coordinate in zip(start, end):
                self.assertAlmostEqual(start_coordinate, end_coordinate, places=12)


if __name__ == "__main__":
    unittest.main()
