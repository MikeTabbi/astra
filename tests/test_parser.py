"""Focused regression tests for ASTRA's qPCR parser."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import parser as astra_parser  # noqa: E402


class ParserTests(unittest.TestCase):
    def test_cli_without_input_uses_manifest_batch(self) -> None:
        arguments = astra_parser._build_cli_parser().parse_args([])

        self.assertIsNone(arguments.input_file)
        self.assertEqual(
            Path(arguments.manifest),
            astra_parser.DEFAULT_MANIFEST_FILE,
        )

    def test_messy_csv_keeps_three_valid_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "accepted.csv"
            rejected = Path(directory) / "rejected.csv"
            result = astra_parser.run_pipeline(
                PROJECT_ROOT / "datasets" / "examples" / "messy_lab_export.csv",
                output,
                target_gene="C5",
                rejected_output_file=rejected,
            )

            self.assertEqual(len(result), 3)
            self.assertEqual(result["target_name"].tolist(), ["C5", "C5", "C5"])
            self.assertEqual(result["time_point"].tolist(), [0, 6, 72])
            self.assertTrue(output.is_file())
            self.assertTrue(rejected.is_file())

            rejected_rows = pd.read_csv(rejected)
            reasons = " ".join(rejected_rows["_rejection_reason"].astype(str))
            self.assertIn("NOAMP", reasons.upper())
            self.assertIn("EXPFAIL", reasons.upper())
            self.assertIn("MISSING_OR_INVALID:VARIANCE_SD", reasons.upper())

    def test_explicit_qc_columns_treat_n_as_pass_and_y_as_failure(self) -> None:
        dataframe = pd.DataFrame(
            {
                "Target Name": ["C5", "C5", "C5"],
                "Time Point": [1, 2, 3],
                "Salinity": ["Low", "Low", "High"],
                "RQ": [1.0, 2.0, 3.0],
                "Ct SD": [0.1, 0.2, 0.3],
                "HIGHSD": ["N", "Y", ""],
                "NOAMP": ["N", "N", "N"],
            }
        )

        result = astra_parser.process_qpcr_data(
            dataframe,
            target_gene="C5",
            aggregate_replicates=False,
        )

        self.assertEqual(len(result.data), 2)
        self.assertEqual(result.data["time_point"].tolist(), [1, 3])
        self.assertEqual(len(result.rejected), 1)
        self.assertIn("highsd=Y", result.rejected.iloc[0]["_rejection_reason"])

    def test_new_quantstudio_qc_columns_are_rejected(self) -> None:
        dataframe = pd.DataFrame(
            {
                "Target Name": ["SB1", "SB1", "SB1"],
                "Time Point": [168, 168, 168],
                "Salinity": ["Low", "Low", "Low"],
                "RQ": [1.0, 2.0, 3.0],
                "Ct SD": [0.1, 0.2, 0.3],
                "AMPNC": ["Y", "N", "N"],
                "MTP": ["N", "Y", "N"],
            }
        )

        result = astra_parser.process_qpcr_data(
            dataframe,
            target_gene="SB1",
            aggregate_replicates=False,
        )

        self.assertEqual(result.data["fold_change_rq"].tolist(), [3.0])
        reasons = " ".join(result.rejected["_rejection_reason"].astype(str))
        self.assertIn("qc:ampnc=Y", reasons)
        self.assertIn("qc:mtp=Y", reasons)

    def test_sample_setup_metadata_is_joined_by_well(self) -> None:
        results = pd.DataFrame(
            {
                "Well": [1, 2],
                "Sample Name": [101, 102],
                "Target Name": ["C5", "C5"],
                "RQ": [1.5, 2.5],
                "Cт SD": [0.1, 0.2],
            }
        )
        setup = pd.DataFrame(
            {
                "Well": [1, 2],
                "Sample Name": [101, 102],
                "Biogroup Name": ["Low", "High"],
                "Target Name": ["C5", "C5"],
            }
        )

        merged = astra_parser._merge_setup_metadata(results, setup)

        self.assertEqual(merged["biogroup_name"].tolist(), ["Low", "High"])
        self.assertIn("ct_sd", merged.columns)

    def test_categorical_salinity_is_preserved(self) -> None:
        dataframe = pd.DataFrame(
            {
                "target_name": ["C5"],
                "time_point": [1],
                "salinity": ["Low"],
                "fold_change_rq": [1.25],
                "variance_sd": [0.05],
            }
        )

        result = astra_parser.normalize_to_schema(dataframe)

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["salinity"], "Low")

    def test_missing_time_point_is_rejected(self) -> None:
        dataframe = pd.DataFrame(
            {
                "sample_name": [101],
                "target_name": ["C5"],
                "biogroup_name": ["Low"],
                "rq": [1.25],
                "ct_sd": [0.05],
            }
        )

        result = astra_parser.process_qpcr_data(dataframe, target_gene="C5")

        self.assertTrue(result.data.empty)
        self.assertIn(
            "missing_or_invalid:time_point",
            result.rejected.iloc[0]["_rejection_reason"],
        )

    def test_run_level_time_point_fills_workbook_rows(self) -> None:
        dataframe = pd.DataFrame(
            {
                "sample_name": [101, 101],
                "target_name": ["C5", "C5"],
                "biogroup_name": ["Low", "Low"],
                "rq": [2.0, 2.0],
                "ct_sd": [0.1, 0.2],
                "highsd": ["N", "N"],
            }
        )

        result = astra_parser.process_qpcr_data(
            dataframe,
            target_gene="C5",
            time_point=6,
        )

        self.assertEqual(len(result.data), 1)
        self.assertEqual(result.data.iloc[0]["time_point"], 6)
        self.assertAlmostEqual(result.data.iloc[0]["variance_sd"], 0.15)

    def test_manifest_processes_configured_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_directory = root / "raw"
            raw_directory.mkdir()
            pd.DataFrame(
                {
                    "target_name": ["SB1"],
                    "salinity": ["Low"],
                    "fold_change_rq": [2.5],
                    "variance_sd": [0.1],
                }
            ).to_csv(raw_directory / "sample.csv", index=False)
            manifest = {
                "datasets": [
                    {
                        "name": "sample",
                        "input": "raw/sample.csv",
                        "target": "SB1",
                        "time_point": 168,
                        "output": "parsed/sample.csv",
                    }
                ]
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            results = astra_parser.run_dataset_manifest(manifest_path)

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].data.iloc[0]["time_point"], 168)
            self.assertTrue((root / "parsed" / "sample.csv").is_file())
            self.assertTrue(
                (root / "parsed" / "sample_rejected.csv").is_file()
            )


if __name__ == "__main__":
    unittest.main()
