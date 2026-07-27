"""Focused regression tests for ASTRA's expression-atlas grid."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import grid as astra_grid  # noqa: E402


def _sample_parser_output() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_name": ["SB1", "SB1", "SB1", "SB1"],
            "time_point": [1, 1, 1, 2],
            "salinity": ["Low", "Low", "High", "Low"],
            "fold_change_rq": [2.0, 4.0, 0.5, 1.0],
            "variance_sd": [0.1, 0.3, 0.2, 0.1],
        }
    )


class GridTests(unittest.TestCase):
    def test_cli_without_input_uses_batch_directories(self) -> None:
        arguments = astra_grid._build_cli_parser().parse_args([])

        self.assertIsNone(arguments.input_file)
        self.assertEqual(
            Path(arguments.parsed_dir),
            astra_grid.DEFAULT_PARSED_DIRECTORY,
        )
        self.assertEqual(
            Path(arguments.grid_dir),
            astra_grid.DEFAULT_GRID_DIRECTORY,
        )

    def test_grid_averages_nested_samples_at_each_intersection(self) -> None:
        atlas = astra_grid.build_expression_atlas(_sample_parser_output())

        self.assertEqual(atlas.target_name, "SB1")
        self.assertAlmostEqual(atlas.matrix.loc["Low", 1], 3.0)
        self.assertAlmostEqual(atlas.matrix.loc["Low", 2], 1.0)
        self.assertAlmostEqual(atlas.matrix.loc["High", 1], 0.5)
        self.assertTrue(pd.isna(atlas.matrix.loc["High", 2]))

    def test_grid_assigns_expression_classes_and_css_colors(self) -> None:
        atlas = astra_grid.build_expression_atlas(_sample_parser_output())
        cells = {
            (cell["row"], cell["column"]): cell for cell in atlas.cells
        }

        self.assertEqual(cells[("Low", 1)]["classification"], "above_reference")
        self.assertEqual(cells[("Low", 2)]["classification"], "reference")
        self.assertEqual(cells[("High", 1)]["classification"], "below_reference")
        self.assertEqual(cells[("High", 2)]["classification"], "missing")
        for cell in cells.values():
            self.assertRegex(cell["color"], r"^#[0-9A-F]{6}$")

    def test_grid_requires_the_parser_contract(self) -> None:
        incomplete = _sample_parser_output().drop(columns="variance_sd")

        with self.assertRaisesRegex(ValueError, "Missing column"):
            astra_grid.build_expression_atlas(incomplete)

    def test_automatic_target_detection_rejects_mixed_targets(self) -> None:
        mixed = _sample_parser_output()
        mixed.loc[0, "target_name"] = "P2"

        with self.assertRaisesRegex(ValueError, "exactly one target"):
            astra_grid.build_expression_atlas(mixed)

        atlas = astra_grid.build_expression_atlas(mixed, target_gene="SB1")
        self.assertEqual(atlas.target_name, "SB1")

    def test_run_grid_writes_matrix_and_json_atlas(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_file = root / "parsed.csv"
            matrix_file = root / "grid.csv"
            atlas_file = root / "atlas.json"
            _sample_parser_output().to_csv(input_file, index=False)

            artifact = astra_grid.run_grid(
                input_file=input_file,
                matrix_file=matrix_file,
                atlas_file=atlas_file,
            )

            self.assertTrue(matrix_file.is_file())
            self.assertTrue(atlas_file.is_file())
            payload = json.loads(atlas_file.read_text(encoding="utf-8"))
            self.assertEqual(payload["target_name"], "SB1")
            self.assertEqual(payload["row_axis"], "salinity")
            self.assertEqual(payload["column_axis"], "time_point")
            self.assertEqual(payload["matrix"]["rows"], ["High", "Low"])
            self.assertEqual(artifact.atlas.matrix.shape, (2, 2))

    def test_batch_ignores_rejection_reports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parsed_directory = root / "parsed"
            grid_directory = root / "grids"
            parsed_directory.mkdir()
            _sample_parser_output().to_csv(
                parsed_directory / "accepted.csv",
                index=False,
            )
            _sample_parser_output().to_csv(
                parsed_directory / "accepted_rejected.csv",
                index=False,
            )

            artifacts = astra_grid.run_grid_batch(
                parsed_directory=parsed_directory,
                grid_directory=grid_directory,
            )

            self.assertEqual(len(artifacts), 1)
            self.assertTrue((grid_directory / "accepted_grid.csv").is_file())
            self.assertTrue((grid_directory / "accepted_atlas.json").is_file())


if __name__ == "__main__":
    unittest.main()
