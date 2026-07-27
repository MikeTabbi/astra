"""Focused regression tests for ASTRA's synthetic-data generator."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import generator as astra_generator  # noqa: E402


def _sample_parser_output(*, zero_variance: bool = False) -> pd.DataFrame:
    variance = [0.0, 0.0, 0.0, 0.0] if zero_variance else [0.1, 0.2, 0.3, 0.15]
    return pd.DataFrame(
        {
            "target_name": ["SB1", "SB1", "SB1", "SB1"],
            "time_point": [168, 168, 168, 168],
            "salinity": ["Low", "Low", "High", "Control"],
            "fold_change_rq": [10.0, 20.0, 2.0, 1.0],
            "variance_sd": variance,
        }
    )


class GeneratorTests(unittest.TestCase):
    def test_cli_without_input_uses_batch_directories(self) -> None:
        arguments = astra_generator._build_cli_parser().parse_args([])

        self.assertIsNone(arguments.input_file)
        self.assertEqual(
            Path(arguments.parsed_dir),
            astra_generator.DEFAULT_PARSED_DIRECTORY,
        )
        self.assertEqual(
            Path(arguments.synthetic_dir),
            astra_generator.DEFAULT_SYNTHETIC_DIRECTORY,
        )

    def test_generation_preserves_observed_categories_and_time_points(self) -> None:
        source = _sample_parser_output()

        synthetic = astra_generator.generate_synthetic_dataframe(
            source,
            rows=60,
            seed=7,
            source_dataset="sb1.csv",
        )

        self.assertEqual(len(synthetic), 60)
        self.assertEqual(set(synthetic["target_name"]), {"SB1"})
        self.assertEqual(set(synthetic["time_point"]), {168})
        self.assertEqual(
            set(synthetic["salinity"]),
            {"Control", "High", "Low"},
        )
        self.assertNotIn("C5", set(synthetic["target_name"]))

    def test_every_synthetic_row_has_reconstructable_provenance(self) -> None:
        synthetic = astra_generator.generate_synthetic_dataframe(
            _sample_parser_output(),
            rows=30,
            seed=11,
            source_dataset="sb1.csv",
        )

        reconstructed = (
            synthetic["source_fold_change_rq"]
            * np.exp2(-synthetic["jitter_delta_ct"])
        )
        np.testing.assert_allclose(
            synthetic["fold_change_rq"],
            reconstructed,
        )
        self.assertTrue(synthetic["synthetic_id"].is_unique)
        self.assertTrue(synthetic["source_row"].between(2, 5).all())
        self.assertEqual(
            set(synthetic["generation_method"]),
            {astra_generator.GENERATION_METHOD},
        )
        self.assertEqual(set(synthetic["random_seed"]), {11})

    def test_same_seed_produces_identical_output(self) -> None:
        source = _sample_parser_output()

        first = astra_generator.generate_synthetic_dataframe(
            source,
            rows=50,
            seed=2026,
        )
        second = astra_generator.generate_synthetic_dataframe(
            source,
            rows=50,
            seed=2026,
        )

        pd.testing.assert_frame_equal(first, second)

    def test_zero_variance_only_bootstraps_observed_values(self) -> None:
        source = _sample_parser_output(zero_variance=True)

        synthetic = astra_generator.generate_synthetic_dataframe(
            source,
            rows=50,
            seed=5,
        )

        self.assertTrue(
            set(synthetic["fold_change_rq"]).issubset(
                set(source["fold_change_rq"])
            )
        )
        self.assertTrue((synthetic["jitter_delta_ct"] == 0).all())

    def test_requested_rows_must_retain_every_observed_group(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least"):
            astra_generator.generate_synthetic_dataframe(
                _sample_parser_output(),
                rows=2,
            )

    def test_run_generation_writes_data_and_traceability_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_file = root / "parsed.csv"
            output_file = root / "synthetic.csv"
            report_file = root / "report.json"
            _sample_parser_output().to_csv(input_file, index=False)

            artifact = astra_generator.run_generation(
                input_file=input_file,
                output_file=output_file,
                report_file=report_file,
                rows=40,
                seed=9,
            )

            self.assertEqual(artifact.synthetic_rows, 40)
            self.assertTrue(output_file.is_file())
            self.assertTrue(report_file.is_file())
            payload = json.loads(report_file.read_text(encoding="utf-8"))
            self.assertEqual(payload["synthetic_rows_written"], 40)
            self.assertEqual(payload["time_points_preserved"], [168])
            self.assertFalse(payload["new_time_points_generated"])
            self.assertEqual(len(payload["input_sha256"]), 64)
            self.assertTrue(
                any(
                    "cannot train or validate a time-series forecast"
                    in limitation
                    for limitation in payload["limitations"]
                )
            )

    def test_batch_ignores_rejection_reports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parsed_directory = root / "parsed"
            synthetic_directory = root / "synthetic"
            parsed_directory.mkdir()
            _sample_parser_output().to_csv(
                parsed_directory / "accepted.csv",
                index=False,
            )
            _sample_parser_output().to_csv(
                parsed_directory / "accepted_rejected.csv",
                index=False,
            )

            artifacts = astra_generator.run_generation_batch(
                parsed_directory=parsed_directory,
                synthetic_directory=synthetic_directory,
                rows=30,
                seed=100,
            )

            self.assertEqual(len(artifacts), 1)
            self.assertTrue(
                (synthetic_directory / "accepted_synthetic.csv").is_file()
            )
            self.assertTrue(
                (
                    synthetic_directory / "accepted_generation_report.json"
                ).is_file()
            )


if __name__ == "__main__":
    unittest.main()
