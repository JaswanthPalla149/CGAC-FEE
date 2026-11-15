from pathlib import Path
import tempfile
import unittest

from download_euroc import validate_euroc_sequence
from run_euroc_batch import find_sequences


def create_short_sequence(root):
    sequence = root / "MH_01_easy"
    camera = sequence / "mav0" / "cam0"
    imu = sequence / "mav0" / "imu0"
    ground_truth = sequence / "mav0" / "state_groundtruth_estimate0"
    camera.mkdir(parents=True)
    imu.mkdir(parents=True)
    ground_truth.mkdir(parents=True)
    (camera / "sensor.yaml").write_text("sensor_type: camera\n", encoding="utf-8")
    (camera / "data.csv").write_text(
        "#timestamp,filename\n1,1.png\n2,2.png\n", encoding="utf-8"
    )
    (imu / "data.csv").write_text("#timestamp,gyro\n1,0\n", encoding="utf-8")
    (ground_truth / "data.csv").write_text(
        "#timestamp,position\n1,0\n2,0\n", encoding="utf-8"
    )
    return sequence


class FullEuRoCValidationTests(unittest.TestCase):
    def test_downloader_rejects_short_sequence(self):
        with tempfile.TemporaryDirectory() as temporary:
            sequence = create_short_sequence(Path(temporary))
            with self.assertRaisesRegex(ValueError, "expected at least 1000"):
                validate_euroc_sequence(sequence)

    def test_batch_runner_skips_short_sequence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_short_sequence(root)
            self.assertEqual(find_sequences(root), [])


if __name__ == "__main__":
    unittest.main()
