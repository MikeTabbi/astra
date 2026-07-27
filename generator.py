"""Generate traceable synthetic rows from ASTRA parser output.

This module does not invent targets, conditions, or time points. It performs a
stratified bootstrap within each observed target/time/condition group, samples
real parser rows with replacement, and applies controlled multiplicative RQ
jitter derived from the sampled row's ``variance_sd``:

    synthetic_rq = source_rq * 2 ** (-Normal(0, variance_sd))

The formula treats the parser's Ct SD field as uncertainty on a log2 scale. It
is a transparent local oversampling method, not a substitute for new biological
replicates and not a time-series forecasting model.

With no arguments, ``python generator.py`` processes every accepted CSV in
``datasets/parsed`` and writes data plus generation reports to
``datasets/synthetic``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd

import grid as astra_grid


PathLike = Union[str, Path]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PARSED_DIRECTORY = PROJECT_ROOT / "datasets" / "parsed"
DEFAULT_SYNTHETIC_DIRECTORY = PROJECT_ROOT / "datasets" / "synthetic"
DEFAULT_ROWS = 1000
DEFAULT_SEED = 2026
DEFAULT_TARGET_GENE = "auto"
GENERATION_METHOD = "stratified_bootstrap_ct_sd_log2_jitter"
GROUP_COLUMNS = ["target_name", "time_point", "salinity"]
PROVENANCE_COLUMNS = [
    "synthetic_id",
    "source_dataset",
    "source_row",
    "source_fold_change_rq",
    "source_variance_sd",
    "jitter_delta_ct",
    "generation_method",
    "random_seed",
]
SYNTHETIC_OUTPUT_COLUMNS = [
    *astra_grid.PARSER_SCHEMA_COLUMNS,
    *PROVENANCE_COLUMNS,
]


@dataclass(frozen=True)
class GenerationArtifact:
    """Files and counts produced from one accepted parser CSV."""

    input_file: Path
    output_file: Path
    report_file: Path
    source_rows: int
    synthetic_rows: int
    seed: int


def _json_scalar(value: object) -> object:
    """Convert NumPy/Pandas scalar values to JSON-compatible values."""
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _safe_identifier(value: str) -> str:
    """Create a stable identifier segment without changing source data."""
    normalized = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-")
    return normalized or "synthetic"


def _sha256(path: Path) -> str:
    """Return a content hash for source-file traceability."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_parser_output(
    input_file: PathLike,
    *,
    target_gene: str = DEFAULT_TARGET_GENE,
) -> pd.DataFrame:
    """Load parser output while preserving its original CSV row numbers."""
    path = Path(input_file)
    if not path.is_file():
        raise FileNotFoundError(
            f"Clean parser output not found: {path}. Run parser.py first."
        )

    raw = pd.read_csv(path)
    validated = astra_grid.validate_parser_dataframe(
        raw,
        target_gene=target_gene,
    )
    resolved_target = str(validated.iloc[0]["target_name"])
    target_values = raw["target_name"].astype("string").str.strip()
    source_mask = target_values.str.casefold().eq(resolved_target.casefold())
    source_rows = (raw.index[source_mask] + 2).tolist()
    if len(source_rows) != len(validated):
        raise ValueError(
            "Could not preserve source-row provenance after target filtering"
        )

    validated["_source_row"] = source_rows
    return validated


def _allocate_group_rows(dataframe: pd.DataFrame, rows: int) -> pd.Series:
    """Allocate output rows proportionally while retaining every source group."""
    group_sizes = dataframe.groupby(
        GROUP_COLUMNS,
        sort=True,
        dropna=False,
    ).size()
    group_count = len(group_sizes)
    if rows < group_count:
        raise ValueError(
            f"Requested rows ({rows}) must be at least the number of observed "
            f"target/time/condition groups ({group_count})"
        )

    allocation = pd.Series(1, index=group_sizes.index, dtype="int64")
    remaining = rows - group_count
    if remaining == 0:
        return allocation

    weights = group_sizes / group_sizes.sum()
    expected = weights * remaining
    floor_counts = np.floor(expected).astype("int64")
    allocation += floor_counts

    leftover = rows - int(allocation.sum())
    if leftover:
        fractional = (expected - floor_counts).sort_values(
            ascending=False,
            kind="stable",
        )
        for group_key in fractional.index[:leftover]:
            allocation.loc[group_key] += 1
    return allocation


def generate_synthetic_dataframe(
    dataframe: pd.DataFrame,
    *,
    rows: int = DEFAULT_ROWS,
    seed: int = DEFAULT_SEED,
    source_dataset: str = "parser_output.csv",
    target_gene: str = DEFAULT_TARGET_GENE,
) -> pd.DataFrame:
    """Bootstrap traceable rows from one validated parser dataframe."""
    if rows <= 0:
        raise ValueError("Synthetic row count must be greater than zero")
    if not isinstance(seed, int):
        raise ValueError("Random seed must be an integer")

    data = astra_grid.validate_parser_dataframe(
        dataframe,
        target_gene=target_gene,
    )
    if "_source_row" in dataframe.columns and len(data) == len(dataframe):
        data["_source_row"] = pd.to_numeric(
            dataframe["_source_row"],
            errors="raise",
        ).astype("int64").to_numpy()
    else:
        data["_source_row"] = np.arange(2, len(data) + 2)

    allocation = _allocate_group_rows(data, rows)
    rng = np.random.default_rng(seed)
    generated_frames: list[pd.DataFrame] = []

    grouped = data.groupby(GROUP_COLUMNS, sort=True, dropna=False)
    for group_key, group in grouped:
        output_count = int(allocation.loc[group_key])
        picked_positions = rng.integers(0, len(group), size=output_count)
        sampled = group.iloc[picked_positions].reset_index(drop=True)
        source_sd = sampled["variance_sd"].to_numpy(dtype=float)
        jitter_delta_ct = rng.normal(loc=0.0, scale=source_sd)
        source_rq = sampled["fold_change_rq"].to_numpy(dtype=float)
        synthetic_rq = source_rq * np.exp2(-jitter_delta_ct)
        if not np.isfinite(synthetic_rq).all() or (synthetic_rq < 0).any():
            raise ValueError(
                "Synthetic generation produced a non-finite or negative RQ value"
            )

        generated = sampled[astra_grid.PARSER_SCHEMA_COLUMNS].copy()
        generated["fold_change_rq"] = synthetic_rq
        generated["source_dataset"] = source_dataset
        generated["source_row"] = sampled["_source_row"].astype("int64")
        generated["source_fold_change_rq"] = source_rq
        generated["source_variance_sd"] = source_sd
        generated["jitter_delta_ct"] = jitter_delta_ct
        generated["generation_method"] = GENERATION_METHOD
        generated["random_seed"] = seed
        generated_frames.append(generated)

    synthetic = pd.concat(generated_frames, ignore_index=True)
    identifier_prefix = _safe_identifier(Path(source_dataset).stem)
    synthetic.insert(
        len(astra_grid.PARSER_SCHEMA_COLUMNS),
        "synthetic_id",
        [
            f"{identifier_prefix}-S{position:06d}"
            for position in range(1, len(synthetic) + 1)
        ],
    )
    return synthetic[SYNTHETIC_OUTPUT_COLUMNS]


def _group_report(
    source: pd.DataFrame,
    synthetic: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Summarize source and generated distributions for manual review."""
    records: list[dict[str, Any]] = []
    source_groups = source.groupby(GROUP_COLUMNS, sort=True, dropna=False)
    synthetic_groups = synthetic.groupby(GROUP_COLUMNS, sort=True, dropna=False)

    for group_key, source_group in source_groups:
        key = group_key if isinstance(group_key, tuple) else (group_key,)
        synthetic_group = synthetic_groups.get_group(group_key)
        records.append(
            {
                **{
                    column: _json_scalar(value)
                    for column, value in zip(GROUP_COLUMNS, key)
                },
                "source_rows": len(source_group),
                "synthetic_rows": len(synthetic_group),
                "source_fold_change_mean": float(
                    source_group["fold_change_rq"].mean()
                ),
                "synthetic_fold_change_mean": float(
                    synthetic_group["fold_change_rq"].mean()
                ),
                "source_fold_change_min": float(
                    source_group["fold_change_rq"].min()
                ),
                "source_fold_change_max": float(
                    source_group["fold_change_rq"].max()
                ),
                "synthetic_fold_change_min": float(
                    synthetic_group["fold_change_rq"].min()
                ),
                "synthetic_fold_change_max": float(
                    synthetic_group["fold_change_rq"].max()
                ),
                "single_source_row_warning": len(source_group) == 1,
            }
        )
    return records


def build_generation_report(
    *,
    input_file: Path,
    output_file: Path,
    source: pd.DataFrame,
    synthetic: pd.DataFrame,
    rows: int,
    seed: int,
) -> dict[str, Any]:
    """Build an auditable description of one synthetic generation run."""
    time_points = sorted(
        {_json_scalar(value) for value in source["time_point"]},
        key=float,
    )
    single_time_point = len(time_points) == 1
    limitations = [
        (
            "Synthetic rows are derived from existing measurements and are not "
            "new independent biological evidence."
        ),
        (
            "No target, treatment condition, or time point absent from the "
            "source data was generated."
        ),
        (
            "Ct SD is used as log2-scale jitter; Dr. Todd should confirm that "
            "this interpretation is appropriate for the instrument export."
        ),
    ]
    if single_time_point:
        limitations.append(
            "Only one real time point is present, so this dataset cannot train "
            "or validate a time-series forecast."
        )

    return {
        "schema_version": 1,
        "generation_method": GENERATION_METHOD,
        "formula": (
            "synthetic_rq = source_rq * "
            "2 ** (-Normal(mean=0, sd=source_variance_sd))"
        ),
        "input_file": str(input_file),
        "input_sha256": _sha256(input_file),
        "output_file": str(output_file),
        "random_seed": seed,
        "source_rows": len(source),
        "synthetic_rows_requested": rows,
        "synthetic_rows_written": len(synthetic),
        "targets": sorted(
            {str(value) for value in source["target_name"]},
            key=str.casefold,
        ),
        "time_points_preserved": time_points,
        "conditions_preserved": sorted(
            {str(value) for value in source["salinity"]},
            key=str.casefold,
        ),
        "new_time_points_generated": False,
        "group_comparison": _group_report(source, synthetic),
        "limitations": limitations,
    }


def run_generation(
    input_file: PathLike,
    output_file: PathLike,
    report_file: PathLike,
    *,
    rows: int = DEFAULT_ROWS,
    seed: int = DEFAULT_SEED,
    target_gene: str = DEFAULT_TARGET_GENE,
) -> GenerationArtifact:
    """Generate one synthetic CSV and its JSON traceability report."""
    input_path = Path(input_file)
    output_path = Path(output_file)
    report_path = Path(report_file)
    source = load_parser_output(input_path, target_gene=target_gene)
    synthetic = generate_synthetic_dataframe(
        source,
        rows=rows,
        seed=seed,
        source_dataset=input_path.name,
        target_gene=target_gene,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    synthetic.to_csv(output_path, index=False)
    report = build_generation_report(
        input_file=input_path,
        output_file=output_path,
        source=source,
        synthetic=synthetic,
        rows=rows,
        seed=seed,
    )
    with report_path.open("w", encoding="utf-8") as file_handle:
        json.dump(report, file_handle, indent=2)

    logger.info("Wrote %d traceable synthetic rows to %s", len(synthetic), output_path)
    logger.info("Wrote generation report to %s", report_path)
    return GenerationArtifact(
        input_file=input_path,
        output_file=output_path,
        report_file=report_path,
        source_rows=len(source),
        synthetic_rows=len(synthetic),
        seed=seed,
    )


def discover_parser_outputs(
    parsed_directory: PathLike = DEFAULT_PARSED_DIRECTORY,
) -> list[Path]:
    """Find accepted parser CSVs while excluding rejection reports."""
    return astra_grid.discover_parser_outputs(parsed_directory)


def run_generation_batch(
    parsed_directory: PathLike = DEFAULT_PARSED_DIRECTORY,
    synthetic_directory: PathLike = DEFAULT_SYNTHETIC_DIRECTORY,
    *,
    rows: int = DEFAULT_ROWS,
    seed: int = DEFAULT_SEED,
) -> list[GenerationArtifact]:
    """Generate synthetic rows for every accepted parser CSV."""
    output_directory = Path(synthetic_directory)
    artifacts: list[GenerationArtifact] = []
    for position, input_path in enumerate(
        discover_parser_outputs(parsed_directory)
    ):
        dataset_seed = seed + position
        artifacts.append(
            run_generation(
                input_file=input_path,
                output_file=(
                    output_directory / f"{input_path.stem}_synthetic.csv"
                ),
                report_file=(
                    output_directory / f"{input_path.stem}_generation_report.json"
                ),
                rows=rows,
                seed=dataset_seed,
            )
        )
    return artifacts


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate traceable synthetic rows from ASTRA parser output."
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
        type=int,
        default=DEFAULT_ROWS,
        help=f"Synthetic rows per dataset (default: {DEFAULT_ROWS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Reproducible random seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET_GENE,
        help="Target gene for a single CSV (default: detect its only target)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Synthetic CSV path for a single input",
    )
    parser.add_argument(
        "--report-output",
        help="Generation-report JSON path for a single input",
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
        "--synthetic-dir",
        default=str(DEFAULT_SYNTHETIC_DIRECTORY),
        help=(
            "Batch output directory "
            f"(default: {DEFAULT_SYNTHETIC_DIRECTORY})"
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point."""
    arguments = _build_cli_parser().parse_args(argv)
    if arguments.rows <= 0:
        logger.error("--rows must be greater than zero")
        return 1

    if arguments.input_file is None:
        if (
            arguments.output
            or arguments.report_output
            or arguments.target.casefold() != DEFAULT_TARGET_GENE
        ):
            logger.error(
                "--output, --report-output, and --target require an explicit "
                "input CSV"
            )
            return 1
        try:
            artifacts = run_generation_batch(
                parsed_directory=arguments.parsed_dir,
                synthetic_directory=arguments.synthetic_dir,
                rows=arguments.rows,
                seed=arguments.seed,
            )
        except Exception as exc:
            logger.error("ASTRA generator failed: %s", exc)
            return 1

        print("\nASTRA generator batch complete")
        for artifact in artifacts:
            print(
                f"- {artifact.input_file.name}: {artifact.source_rows} source "
                f"row(s) -> {artifact.synthetic_rows} traceable synthetic "
                f"row(s), seed={artifact.seed} -> {artifact.output_file}"
            )
        print(
            "Synthetic rows preserve observed time points and are not new "
            "biological evidence."
        )
        return 0

    input_path = Path(arguments.input_file)
    output_directory = Path(arguments.synthetic_dir)
    output_file = Path(
        arguments.output
        or output_directory / f"{input_path.stem}_synthetic.csv"
    )
    report_file = Path(
        arguments.report_output
        or output_directory / f"{input_path.stem}_generation_report.json"
    )
    try:
        artifact = run_generation(
            input_file=input_path,
            output_file=output_file,
            report_file=report_file,
            rows=arguments.rows,
            seed=arguments.seed,
            target_gene=arguments.target,
        )
    except Exception as exc:
        logger.error("ASTRA generator failed: %s", exc)
        return 1

    print("\nASTRA generator complete")
    print(f"Source rows: {artifact.source_rows}")
    print(f"Synthetic rows: {artifact.synthetic_rows}")
    print(f"Random seed: {artifact.seed}")
    print(f"Synthetic data: {artifact.output_file}")
    print(f"Generation report: {artifact.report_file}")
    print(
        "Synthetic rows preserve observed time points and are not new "
        "biological evidence."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
