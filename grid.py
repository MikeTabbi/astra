"""Build ASTRA expression-atlas matrices from clean parser output.

The grid is Stage 2 of the local ASTRA workflow:

    parser.py -> grid.py -> narrator.py

``parser.py`` remains the owner of raw-file cleanup and the five-column data
contract. This module validates that contract, averages measurements at each
selected row/column intersection, and exports:

* a conventional pivot-table CSV for analysis; and
* a JSON atlas containing the matrix plus a CSS color and expression class for
  every cell.

With no arguments, ``python grid.py`` processes every accepted parser CSV in
``datasets/parsed``. Rejection reports are ignored automatically.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd


PathLike = Union[str, Path]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PARSED_DIRECTORY = PROJECT_ROOT / "datasets" / "parsed"
DEFAULT_GRID_DIRECTORY = PROJECT_ROOT / "datasets" / "grids"
DEFAULT_ROW_AXIS = "salinity"
DEFAULT_COLUMN_AXIS = "time_point"
DEFAULT_VALUE_COLUMN = "fold_change_rq"
DEFAULT_AGGREGATION = "mean"
DEFAULT_TARGET_GENE = "auto"

PARSER_SCHEMA_COLUMNS = [
    "target_name",
    "time_point",
    "salinity",
    "fold_change_rq",
    "variance_sd",
]
DIMENSION_COLUMNS = ["target_name", "time_point", "salinity"]
VALUE_COLUMNS = ["fold_change_rq", "variance_sd"]
AGGREGATIONS = {
    "mean": "mean",
    "median": "median",
    "min": "min",
    "max": "max",
}

MISSING_COLOR = "#E5E7EB"
REFERENCE_COLOR = "#F7F7F7"
BELOW_REFERENCE_COLOR = "#2166AC"
ABOVE_REFERENCE_COLOR = "#B2182B"
REFERENCE_VALUE = 1.0
REFERENCE_TOLERANCE = 1e-9


@dataclass
class ExpressionAtlas:
    """One validated and aggregated ASTRA expression matrix."""

    target_name: str
    row_axis: str
    column_axis: str
    value_column: str
    aggregation: str
    matrix: pd.DataFrame
    cells: list[dict[str, Any]]


@dataclass(frozen=True)
class GridArtifact:
    """Files produced from one accepted parser CSV."""

    input_file: Path
    matrix_file: Path
    atlas_file: Path
    atlas: ExpressionAtlas


def _invalid_row_numbers(mask: pd.Series) -> list[int]:
    """Convert a validation mask to human-readable CSV row numbers."""
    return [int(index) + 2 for index in mask[mask].index]


def _resolve_target_name(dataframe: pd.DataFrame, target_gene: str) -> str:
    """Resolve ``auto`` to the single target present in a parser CSV."""
    available = sorted(
        {
            str(value).strip()
            for value in dataframe["target_name"].dropna().astype(str)
            if str(value).strip()
        },
        key=str.casefold,
    )
    requested = target_gene.strip()
    if requested.casefold() != DEFAULT_TARGET_GENE:
        return requested
    if len(available) != 1:
        raise ValueError(
            "Automatic target detection requires exactly one target; "
            f"available targets: {', '.join(available) or '<none>'}"
        )
    return available[0]


def validate_parser_dataframe(
    dataframe: pd.DataFrame,
    *,
    target_gene: str = DEFAULT_TARGET_GENE,
) -> pd.DataFrame:
    """Validate parser.py's boundary and isolate one requested target."""
    missing = [
        column for column in PARSER_SCHEMA_COLUMNS if column not in dataframe.columns
    ]
    if missing:
        raise ValueError(
            "Grid input is not valid parser output. Missing column(s): "
            + ", ".join(missing)
        )

    data = dataframe[PARSER_SCHEMA_COLUMNS].copy()
    if data.empty:
        raise ValueError("Grid input contains no accepted parser rows")

    for column in ("target_name", "salinity"):
        data[column] = data[column].astype("string").str.strip()
        invalid = data[column].isna() | data[column].eq("")
        if invalid.any():
            raise ValueError(
                f"Column '{column}' is blank at CSV row(s): "
                f"{_invalid_row_numbers(invalid)}"
            )

    for column in ("time_point", "fold_change_rq", "variance_sd"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
        invalid = data[column].isna() | ~data[column].map(
            lambda value: math.isfinite(float(value)) if pd.notna(value) else False
        )
        if invalid.any():
            raise ValueError(
                f"Column '{column}' is not numeric at CSV row(s): "
                f"{_invalid_row_numbers(invalid)}"
            )

        negative = data[column] < 0
        if negative.any():
            raise ValueError(
                f"Column '{column}' cannot be negative at CSV row(s): "
                f"{_invalid_row_numbers(negative)}"
            )

    requested_target = _resolve_target_name(data, target_gene)
    target_mask = data["target_name"].str.casefold().eq(
        requested_target.casefold()
    )
    selected = data.loc[target_mask].copy()
    if selected.empty:
        available = sorted(
            {str(value) for value in data["target_name"]},
            key=str.casefold,
        )
        raise ValueError(
            f"No rows found for target '{requested_target}'. "
            f"Available targets: {', '.join(available)}"
        )

    selected["target_name"] = requested_target
    return selected.reset_index(drop=True)


def _validate_grid_options(
    *,
    row_axis: str,
    column_axis: str,
    value_column: str,
    aggregation: str,
) -> None:
    """Reject axes or calculations that cannot produce a meaningful pivot."""
    for label, value in (("row axis", row_axis), ("column axis", column_axis)):
        if value not in DIMENSION_COLUMNS:
            raise ValueError(
                f"Invalid {label} '{value}'. Choose from: "
                f"{', '.join(DIMENSION_COLUMNS)}"
            )
    if row_axis == column_axis:
        raise ValueError("The row and column axes must be different")
    if value_column not in VALUE_COLUMNS:
        raise ValueError(
            f"Invalid value column '{value_column}'. Choose from: "
            f"{', '.join(VALUE_COLUMNS)}"
        )
    if aggregation not in AGGREGATIONS:
        raise ValueError(
            f"Invalid aggregation '{aggregation}'. Choose from: "
            f"{', '.join(AGGREGATIONS)}"
        )


def _interpolate_hex(start: str, end: str, fraction: float) -> str:
    """Interpolate two CSS hex colors using a fraction from zero to one."""
    fraction = min(max(float(fraction), 0.0), 1.0)
    start_rgb = tuple(int(start[index : index + 2], 16) for index in (1, 3, 5))
    end_rgb = tuple(int(end[index : index + 2], 16) for index in (1, 3, 5))
    rgb = tuple(
        round(first + (second - first) * fraction)
        for first, second in zip(start_rgb, end_rgb)
    )
    return "#" + "".join(f"{channel:02X}" for channel in rgb)


def classify_expression(value: object) -> str:
    """Classify RQ relative to its 1.0 reference value."""
    if pd.isna(value):
        return "missing"
    numeric = float(value)
    if math.isclose(
        numeric,
        REFERENCE_VALUE,
        rel_tol=0,
        abs_tol=REFERENCE_TOLERANCE,
    ):
        return "reference"
    return "below_reference" if numeric < REFERENCE_VALUE else "above_reference"


def expression_color(value: object, *, maximum: float) -> str:
    """Map one RQ value to a dynamic blue-neutral-red CSS color."""
    classification = classify_expression(value)
    if classification == "missing":
        return MISSING_COLOR
    if classification == "reference":
        return REFERENCE_COLOR

    numeric = float(value)
    if classification == "below_reference":
        # Four fold-change halvings (1 -> 0.0625) reach full blue.
        strength = (
            1.0
            if numeric <= 0
            else min(abs(math.log2(numeric)) / 4.0, 1.0)
        )
        return _interpolate_hex(REFERENCE_COLOR, BELOW_REFERENCE_COLOR, strength)

    if maximum <= REFERENCE_VALUE:
        strength = 1.0
    else:
        denominator = math.log2(maximum)
        strength = (
            math.log2(numeric) / denominator if denominator > 0 else 1.0
        )
    return _interpolate_hex(REFERENCE_COLOR, ABOVE_REFERENCE_COLOR, strength)


def build_expression_atlas(
    dataframe: pd.DataFrame,
    *,
    target_gene: str = DEFAULT_TARGET_GENE,
    row_axis: str = DEFAULT_ROW_AXIS,
    column_axis: str = DEFAULT_COLUMN_AXIS,
    value_column: str = DEFAULT_VALUE_COLUMN,
    aggregation: str = DEFAULT_AGGREGATION,
) -> ExpressionAtlas:
    """Create a dynamic pivot matrix and heatmap cells from parser rows."""
    _validate_grid_options(
        row_axis=row_axis,
        column_axis=column_axis,
        value_column=value_column,
        aggregation=aggregation,
    )
    data = validate_parser_dataframe(dataframe, target_gene=target_gene)
    target_name = str(data.iloc[0]["target_name"])

    matrix = pd.pivot_table(
        data,
        index=row_axis,
        columns=column_axis,
        values=value_column,
        aggfunc=AGGREGATIONS[aggregation],
        dropna=False,
        observed=False,
        sort=True,
    )
    matrix.index.name = row_axis
    matrix.columns.name = column_axis

    populated = matrix.to_numpy(dtype=float)
    finite = populated[np.isfinite(populated)]
    maximum = (
        float(finite.max())
        if finite.size
        else REFERENCE_VALUE
    )

    cells: list[dict[str, Any]] = []
    for row_value in matrix.index:
        for column_value in matrix.columns:
            cell_value = matrix.loc[row_value, column_value]
            rendered_value = None if pd.isna(cell_value) else float(cell_value)
            if value_column == "fold_change_rq":
                classification = classify_expression(cell_value)
                color = expression_color(cell_value, maximum=maximum)
            else:
                classification = "missing" if pd.isna(cell_value) else "measured"
                color = MISSING_COLOR if pd.isna(cell_value) else REFERENCE_COLOR
            cells.append(
                {
                    "row": _json_scalar(row_value),
                    "column": _json_scalar(column_value),
                    "value": rendered_value,
                    "classification": classification,
                    "color": color,
                }
            )

    return ExpressionAtlas(
        target_name=target_name,
        row_axis=row_axis,
        column_axis=column_axis,
        value_column=value_column,
        aggregation=aggregation,
        matrix=matrix,
        cells=cells,
    )


def _json_scalar(value: object) -> object:
    """Convert NumPy/Pandas scalar values into JSON-compatible Python values."""
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def atlas_to_dict(
    atlas: ExpressionAtlas,
    *,
    source_file: Optional[PathLike] = None,
) -> dict[str, Any]:
    """Serialize an expression atlas without Pandas-specific objects."""
    return {
        "schema_version": 1,
        "source_file": str(Path(source_file)) if source_file is not None else None,
        "target_name": atlas.target_name,
        "row_axis": atlas.row_axis,
        "column_axis": atlas.column_axis,
        "value_column": atlas.value_column,
        "aggregation": atlas.aggregation,
        "reference_value": (
            REFERENCE_VALUE
            if atlas.value_column == "fold_change_rq"
            else None
        ),
        "legend": {
            "missing": MISSING_COLOR,
            "reference": REFERENCE_COLOR,
            "below_reference": BELOW_REFERENCE_COLOR,
            "above_reference": ABOVE_REFERENCE_COLOR,
        },
        "matrix": {
            "rows": [_json_scalar(value) for value in atlas.matrix.index],
            "columns": [_json_scalar(value) for value in atlas.matrix.columns],
            "values": [
                [
                    None if pd.isna(value) else float(value)
                    for value in row
                ]
                for row in atlas.matrix.to_numpy()
            ],
        },
        "cells": atlas.cells,
    }


def load_parser_output(
    input_file: PathLike,
    *,
    target_gene: str = DEFAULT_TARGET_GENE,
) -> pd.DataFrame:
    """Load and validate one accepted CSV created by parser.py."""
    path = Path(input_file)
    if not path.is_file():
        raise FileNotFoundError(
            f"Clean parser output not found: {path}. Run parser.py first."
        )
    return validate_parser_dataframe(
        pd.read_csv(path),
        target_gene=target_gene,
    )


def discover_parser_outputs(
    parsed_directory: PathLike = DEFAULT_PARSED_DIRECTORY,
) -> list[Path]:
    """Find accepted parser CSVs while excluding rejection reports."""
    directory = Path(parsed_directory)
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Parsed-data directory not found: {directory}. Run parser.py first."
        )
    outputs = sorted(
        (
            path
            for path in directory.glob("*.csv")
            if not path.stem.endswith("_rejected")
        ),
        key=lambda path: path.name.casefold(),
    )
    if not outputs:
        raise FileNotFoundError(
            f"No accepted parser CSVs found in {directory}. Run parser.py first."
        )
    return outputs


def run_grid(
    input_file: PathLike,
    matrix_file: PathLike,
    atlas_file: PathLike,
    *,
    target_gene: str = DEFAULT_TARGET_GENE,
    row_axis: str = DEFAULT_ROW_AXIS,
    column_axis: str = DEFAULT_COLUMN_AXIS,
    value_column: str = DEFAULT_VALUE_COLUMN,
    aggregation: str = DEFAULT_AGGREGATION,
) -> GridArtifact:
    """Build and write one CSV matrix and JSON heatmap atlas."""
    input_path = Path(input_file)
    data = load_parser_output(input_path, target_gene=target_gene)
    atlas = build_expression_atlas(
        data,
        target_gene=target_gene,
        row_axis=row_axis,
        column_axis=column_axis,
        value_column=value_column,
        aggregation=aggregation,
    )

    matrix_path = Path(matrix_file)
    atlas_path = Path(atlas_file)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    atlas_path.parent.mkdir(parents=True, exist_ok=True)
    atlas.matrix.to_csv(matrix_path)
    with atlas_path.open("w", encoding="utf-8") as file_handle:
        json.dump(
            atlas_to_dict(atlas, source_file=input_path),
            file_handle,
            indent=2,
        )

    logger.info("Wrote expression matrix to %s", matrix_path)
    logger.info("Wrote color-mapped atlas to %s", atlas_path)
    return GridArtifact(
        input_file=input_path,
        matrix_file=matrix_path,
        atlas_file=atlas_path,
        atlas=atlas,
    )


def run_grid_batch(
    parsed_directory: PathLike = DEFAULT_PARSED_DIRECTORY,
    grid_directory: PathLike = DEFAULT_GRID_DIRECTORY,
    *,
    row_axis: str = DEFAULT_ROW_AXIS,
    column_axis: str = DEFAULT_COLUMN_AXIS,
    value_column: str = DEFAULT_VALUE_COLUMN,
    aggregation: str = DEFAULT_AGGREGATION,
) -> list[GridArtifact]:
    """Build an expression atlas for every accepted parser CSV."""
    output_directory = Path(grid_directory)
    artifacts: list[GridArtifact] = []
    for input_path in discover_parser_outputs(parsed_directory):
        artifacts.append(
            run_grid(
                input_file=input_path,
                matrix_file=output_directory / f"{input_path.stem}_grid.csv",
                atlas_file=output_directory / f"{input_path.stem}_atlas.json",
                row_axis=row_axis,
                column_axis=column_axis,
                value_column=value_column,
                aggregation=aggregation,
            )
        )
    return artifacts


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a pivot-table expression atlas from ASTRA parser output."
        )
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        help=(
            "Accepted parser CSV. If omitted, process every accepted CSV in "
            "datasets/parsed."
        ),
    )
    parser.add_argument(
        "--rows",
        default=DEFAULT_ROW_AXIS,
        choices=DIMENSION_COLUMNS,
        help=f"Pivot row axis (default: {DEFAULT_ROW_AXIS})",
    )
    parser.add_argument(
        "--columns",
        default=DEFAULT_COLUMN_AXIS,
        choices=DIMENSION_COLUMNS,
        help=f"Pivot column axis (default: {DEFAULT_COLUMN_AXIS})",
    )
    parser.add_argument(
        "--value",
        default=DEFAULT_VALUE_COLUMN,
        choices=VALUE_COLUMNS,
        help=f"Numeric value to aggregate (default: {DEFAULT_VALUE_COLUMN})",
    )
    parser.add_argument(
        "--aggregation",
        default=DEFAULT_AGGREGATION,
        choices=sorted(AGGREGATIONS),
        help=f"Cell aggregation (default: {DEFAULT_AGGREGATION})",
    )
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET_GENE,
        help="Target gene for a single CSV (default: detect its only target)",
    )
    parser.add_argument(
        "--matrix-output",
        help="Pivot CSV path for a single input",
    )
    parser.add_argument(
        "--atlas-output",
        help="Color-mapped JSON path for a single input",
    )
    parser.add_argument(
        "--parsed-dir",
        default=str(DEFAULT_PARSED_DIRECTORY),
        help=(
            "Parsed CSV directory used when no input is supplied "
            f"(default: {DEFAULT_PARSED_DIRECTORY})"
        ),
    )
    parser.add_argument(
        "--grid-dir",
        default=str(DEFAULT_GRID_DIRECTORY),
        help=(
            "Batch output directory "
            f"(default: {DEFAULT_GRID_DIRECTORY})"
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point."""
    arguments = _build_cli_parser().parse_args(argv)
    if arguments.rows == arguments.columns:
        logger.error("The row and column axes must be different")
        return 1

    if arguments.input_file is None:
        if (
            arguments.matrix_output
            or arguments.atlas_output
            or arguments.target.casefold() != DEFAULT_TARGET_GENE
        ):
            logger.error(
                "--matrix-output, --atlas-output, and --target require an "
                "explicit input CSV"
            )
            return 1
        try:
            artifacts = run_grid_batch(
                parsed_directory=arguments.parsed_dir,
                grid_directory=arguments.grid_dir,
                row_axis=arguments.rows,
                column_axis=arguments.columns,
                value_column=arguments.value,
                aggregation=arguments.aggregation,
            )
        except Exception as exc:
            logger.error("ASTRA grid failed: %s", exc)
            return 1

        print("\nASTRA grid batch complete")
        for artifact in artifacts:
            rows, columns = artifact.atlas.matrix.shape
            print(
                f"- {artifact.input_file.name}: {rows}x{columns} matrix -> "
                f"{artifact.matrix_file}; atlas -> {artifact.atlas_file}"
            )
        return 0

    input_path = Path(arguments.input_file)
    output_directory = Path(arguments.grid_dir)
    matrix_file = Path(
        arguments.matrix_output
        or output_directory / f"{input_path.stem}_grid.csv"
    )
    atlas_file = Path(
        arguments.atlas_output
        or output_directory / f"{input_path.stem}_atlas.json"
    )
    try:
        artifact = run_grid(
            input_file=input_path,
            matrix_file=matrix_file,
            atlas_file=atlas_file,
            target_gene=arguments.target,
            row_axis=arguments.rows,
            column_axis=arguments.columns,
            value_column=arguments.value,
            aggregation=arguments.aggregation,
        )
    except Exception as exc:
        logger.error("ASTRA grid failed: %s", exc)
        return 1

    print("\nASTRA grid complete")
    print(f"Target: {artifact.atlas.target_name}")
    print(
        f"Matrix shape: {artifact.atlas.matrix.shape[0]} row(s) x "
        f"{artifact.atlas.matrix.shape[1]} column(s)"
    )
    print(f"Matrix: {artifact.matrix_file}")
    print(f"Atlas: {artifact.atlas_file}")
    print(artifact.atlas.matrix.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
