"""Parse qPCR exports into ASTRA's normalized five-column data contract.

CSV inputs are read as a single table. Excel inputs prefer the ``Results``
worksheet and merge condition metadata from ``Sample Setup`` by well (or sample
name when a well column is unavailable).

Pipeline:
    load workbook/CSV
    -> normalize headers
    -> merge sample metadata
    -> select the requested target
    -> apply instrument QC flags
    -> validate the ASTRA schema
    -> aggregate technical replicates by sample
    -> write accepted and rejected rows

For the current QuantStudio workbook, ``Ct SD`` is the closest available
technical-replicate dispersion field and is mapped to ``variance_sd``. The
workbook does not contain an experiment time point, so callers must supply one
with ``time_point=...`` or ``--time-point``.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd


PathLike = Union[str, Path]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_INPUT_FILE = "lab_data.xls"
DEFAULT_RESULTS_SHEET = "Results"
DEFAULT_SETUP_SHEET = "Sample Setup"

# Values that mean a measurement is absent, not that a QC flag has failed.
_NULL_TOKENS = {
    "",
    "na",
    "n/a",
    "nan",
    "none",
    "null",
    "undetermined",
}

_QC_COLUMNS = {
    "highsd",
    "noamp",
    "outlierrg",
    "expfail",
    "omit",
}
_QC_FAILURE_TOKENS = {
    "highsd",
    "noamp",
    "outlierrg",
    "expfail",
    "omit",
    "failed",
    "failure",
    "fail",
    "flagged",
    "true",
    "1",
    "yes",
    "y",
}
_QC_PASS_TOKENS = _NULL_TOKENS | {
    "false",
    "0",
    "no",
    "n",
    "pass",
    "passed",
    "ok",
}

# Column-name synonyms, keyed by canonical normalized name.
_COLUMN_SYNONYMS: dict[str, list[str]] = {
    "well": ["well", "well_position", "well_id"],
    "sample_name": ["sample_name", "sample", "sample_id"],
    "biogroup_name": ["biogroup_name", "biogroup", "condition", "treatment_group"],
    "target_name": [
        "target_name",
        "target",
        "gene",
        "gene_name",
        "detector",
        "detector_name",
    ],
    "time_point": ["time_point", "time", "timepoint", "time_pt", "elapsed_hours"],
    "salinity": [
        "salinity",
        "salinity_level",
        "salt_ppt",
        "salt",
        "ppt",
        "biogroup_name",
        "condition",
    ],
    "fold_change_rq": [
        "fold_change_rq",
        "fold_change",
        "rq",
        "rel_quant",
        "relative_quantity",
    ],
    "variance_sd": [
        "variance_sd",
        "sd",
        "std_dev",
        "stdev",
        "std",
        "variance",
        "ct_sd",
        "cq_sd",
    ],
    "ct": ["ct", "cq", "ct_mean", "cq_mean"],
}

TARGET_SCHEMA_COLUMNS = [
    "target_name",
    "time_point",
    "salinity",
    "fold_change_rq",
    "variance_sd",
]


@dataclass
class ParserResult:
    """Accepted ASTRA rows plus every auditable rejected source row."""

    data: pd.DataFrame
    rejected: pd.DataFrame


def _normalize_column_name(name: object) -> str:
    """Convert instrument labels into stable lowercase ASCII identifiers."""
    normalized = unicodedata.normalize("NFKC", str(name)).strip().lower()
    # QuantStudio exports can use Cyrillic "т" in labels that visually read Ct.
    normalized = normalized.replace("т", "t").replace("δ", "delta")
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    return normalized.strip("_")


def _normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize headers, remove empty columns, and make duplicates explicit."""
    normalized = df.copy()
    raw_names = [_normalize_column_name(column) for column in normalized.columns]

    counts: dict[str, int] = {}
    unique_names: list[str] = []
    for name in raw_names:
        base = name or "unnamed"
        counts[base] = counts.get(base, 0) + 1
        unique_names.append(base if counts[base] == 1 else f"{base}_{counts[base]}")

    normalized.columns = unique_names
    empty_columns = [
        column
        for column in normalized.columns
        if (column == "unnamed" or column.startswith("unnamed_") or column == "nan")
        and normalized[column].isna().all()
    ]
    if empty_columns:
        normalized = normalized.drop(columns=empty_columns)
    return normalized


def _resolve_column(df: pd.DataFrame, canonical: str) -> Optional[str]:
    """Find the actual column matching a canonical field name."""
    for candidate in _COLUMN_SYNONYMS.get(canonical, [canonical]):
        if candidate in df.columns:
            return candidate
    return None


def _get_series(df: pd.DataFrame, canonical: str, default: object) -> pd.Series:
    """Return the resolved column or a same-length default Series."""
    column = _resolve_column(df, canonical)
    if column is not None:
        return df[column]
    return pd.Series(default, index=df.index)


def _detect_header_row(
    raw_rows: Sequence[Sequence[object]],
    expected_tokens: Optional[Sequence[str]] = None,
) -> int:
    """Detect the table header beneath an instrument metadata banner."""
    if expected_tokens is None:
        expected_tokens = [
            token for synonyms in _COLUMN_SYNONYMS.values() for token in synonyms
        ]

    best_index = 0
    best_score = -1
    for index, row in enumerate(raw_rows):
        cells = [
            _normalize_column_name(cell)
            for cell in row
            if str(cell).strip() and str(cell).strip().lower() != "nan"
        ]
        if len(cells) < 2:
            continue
        score = sum(
            1
            for cell in cells
            if any(
                cell == token
                or cell.startswith(f"{token}_")
                or cell.endswith(f"_{token}")
                for token in expected_tokens
            )
        )
        if score > best_score:
            best_score = score
            best_index = index

    return best_index


def _open_excel_file(path: Path) -> pd.ExcelFile:
    """Open Excel with a clear explanation when legacy .xls support is absent."""
    try:
        return pd.ExcelFile(path)
    except ImportError as exc:
        if path.suffix.lower() == ".xls":
            raise RuntimeError(
                "Reading legacy .xls files requires the 'xlrd' package. "
                "Install the project requirements or convert the workbook to .xlsx."
            ) from exc
        raise


def _select_sheet(
    available_sheets: Sequence[str],
    preferred: Optional[str],
    fallbacks: Sequence[str] = (),
    *,
    required: bool = True,
) -> Optional[str]:
    """Resolve a sheet case-insensitively, using known instrument fallbacks."""
    normalized_lookup = {
        _normalize_column_name(sheet): sheet for sheet in available_sheets
    }

    for candidate in [preferred, *fallbacks]:
        if not candidate:
            continue
        match = normalized_lookup.get(_normalize_column_name(candidate))
        if match is not None:
            return match

    if required:
        requested = preferred or ", ".join(fallbacks) or "<first worksheet>"
        available = ", ".join(available_sheets)
        raise ValueError(
            f"Could not find worksheet '{requested}'. Available worksheets: {available}"
        )
    return None


def _read_excel_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    """Read one instrument worksheet and preserve its source row numbers."""
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None, dtype=str)
    header_index = _detect_header_row(raw.values.tolist())
    dataframe = pd.read_excel(path, sheet_name=sheet_name, header=header_index)
    dataframe = _normalize_dataframe_columns(dataframe)
    dataframe["_source_sheet"] = sheet_name
    dataframe["_source_row"] = np.arange(
        header_index + 2,
        header_index + 2 + len(dataframe),
    )
    logger.info(
        "Loaded worksheet '%s' from %s (header row %d)",
        sheet_name,
        path,
        header_index + 1,
    )
    return dataframe


def load_raw_data(
    file_path: PathLike,
    sheet_name: Optional[str] = None,
) -> pd.DataFrame:
    """Load one CSV table or one selected Excel worksheet."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"qPCR export file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        workbook = _open_excel_file(path)
        try:
            selected_sheet = _select_sheet(
                workbook.sheet_names,
                sheet_name or DEFAULT_RESULTS_SHEET,
                ("Technical Analysis Result", workbook.sheet_names[0]),
            )
        finally:
            workbook.close()
        return _read_excel_sheet(path, selected_sheet)

    if suffix == ".csv":
        last_error: Optional[Exception] = None
        for encoding in ("utf-8", "latin-1", "cp1252"):
            try:
                with path.open("r", encoding=encoding, newline="") as file_handle:
                    raw_rows = list(csv.reader(file_handle))
                if not raw_rows:
                    raise ValueError(f"CSV file is empty: {path}")
                header_index = _detect_header_row(raw_rows)
                dataframe = pd.read_csv(
                    path,
                    header=header_index,
                    encoding=encoding,
                    skip_blank_lines=False,
                )
                dataframe = _normalize_dataframe_columns(dataframe)
                dataframe["_source_sheet"] = "CSV"
                dataframe["_source_row"] = np.arange(
                    header_index + 2,
                    header_index + 2 + len(dataframe),
                )
                logger.info(
                    "Loaded CSV %s using %s (header row %d)",
                    path,
                    encoding,
                    header_index + 1,
                )
                return dataframe
            except (UnicodeDecodeError, pd.errors.ParserError) as exc:
                last_error = exc
                logger.warning(
                    "Failed to parse %s with encoding=%s: %s",
                    path,
                    encoding,
                    exc,
                )
        raise ValueError(
            f"Could not parse CSV {path} with any supported encoding"
        ) from last_error

    raise ValueError(
        f"Unsupported file extension for {path}: expected .csv, .xlsx, or .xls"
    )


def _normalize_identifier(value: object) -> object:
    """Normalize Excel identifiers without changing their biological meaning."""
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if re.fullmatch(r"-?\d+\.0", text):
        text = text[:-2]
    return text.upper()


def _merge_setup_metadata(
    results: pd.DataFrame,
    setup: pd.DataFrame,
) -> pd.DataFrame:
    """Join Sample Setup condition metadata onto Results rows."""
    results = _normalize_dataframe_columns(results)
    setup = _normalize_dataframe_columns(setup)

    join_column = next(
        (
            column
            for column in ("well", "sample_name")
            if column in results.columns and column in setup.columns
        ),
        None,
    )
    if join_column is None:
        raise ValueError(
            "Results and Sample Setup cannot be joined: neither a shared "
            "'Well' nor 'Sample Name' column was found."
        )

    join_key = "_metadata_join_key"
    results[join_key] = results[join_column].map(_normalize_identifier)
    setup[join_key] = setup[join_column].map(_normalize_identifier)

    metadata_columns = [
        column
        for column in (
            "sample_name",
            "target_name",
            "biogroup_name",
            "salinity",
            "time_point",
        )
        if column in setup.columns and column != join_column
    ]
    setup_subset = setup[[join_key, *metadata_columns]].dropna(subset=[join_key])

    duplicate_keys = setup_subset[join_key].duplicated(keep=False)
    if duplicate_keys.any():
        logger.warning(
            "Sample Setup contains duplicate %s values; keeping the first mapping",
            join_column,
        )
        setup_subset = setup_subset.drop_duplicates(join_key, keep="first")

    setup_subset = setup_subset.rename(
        columns={column: f"{column}_setup" for column in metadata_columns}
    )
    merged = results.merge(setup_subset, on=join_key, how="left", validate="many_to_one")

    for column in metadata_columns:
        setup_column = f"{column}_setup"
        if column in merged.columns:
            merged[column] = merged[column].combine_first(merged[setup_column])
        else:
            merged[column] = merged[setup_column]
        merged = merged.drop(columns=setup_column)

    matched = merged[join_key].notna().sum()
    logger.info(
        "Merged Sample Setup metadata onto %d/%d Results rows by %s",
        matched,
        len(merged),
        join_column,
    )
    return merged.drop(columns=join_key)


def _fill_override(
    dataframe: pd.DataFrame,
    canonical: str,
    value: Optional[object],
) -> pd.DataFrame:
    """Fill a missing canonical field from a run-level argument."""
    if value is None:
        return dataframe

    column = _resolve_column(dataframe, canonical)
    if column is None:
        dataframe[canonical] = value
    else:
        dataframe[column] = dataframe[column].where(dataframe[column].notna(), value)
    return dataframe


def load_workbook_data(
    file_path: PathLike,
    *,
    results_sheet: str = DEFAULT_RESULTS_SHEET,
    setup_sheet: Optional[str] = DEFAULT_SETUP_SHEET,
    time_point: Optional[float] = None,
    salinity: Optional[object] = None,
) -> pd.DataFrame:
    """Load Results and merge Sample Setup metadata for an Excel workbook."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"qPCR export file not found: {path}")

    workbook = _open_excel_file(path)
    try:
        selected_results = _select_sheet(
            workbook.sheet_names,
            results_sheet,
            ("Technical Analysis Result",),
        )
        selected_setup = _select_sheet(
            workbook.sheet_names,
            setup_sheet,
            required=False,
        )
    finally:
        workbook.close()

    results = _read_excel_sheet(path, selected_results)
    if selected_setup is not None and selected_setup != selected_results:
        setup = _read_excel_sheet(path, selected_setup)
        results = _merge_setup_metadata(results, setup)
    elif setup_sheet:
        logger.warning(
            "Worksheet '%s' was not found; continuing without setup metadata",
            setup_sheet,
        )

    results = _fill_override(results, "time_point", time_point)
    results = _fill_override(results, "salinity", salinity)
    return results


def _replace_null_tokens(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Replace textual missing-value markers without altering QC pass values."""
    cleaned = dataframe.copy()
    for column in cleaned.columns:
        if column.startswith("_source_"):
            continue
        as_text = cleaned[column].astype("string").str.strip().str.lower()
        cleaned.loc[as_text.isin(_NULL_TOKENS), column] = pd.NA
    return cleaned


def _rejected_copy(dataframe: pd.DataFrame, reasons: Iterable[str]) -> pd.DataFrame:
    """Copy rows and attach one human-auditable reason to each."""
    rejected = dataframe.copy()
    rejected["_rejection_reason"] = list(reasons)
    return rejected


def _filter_target(
    dataframe: pd.DataFrame,
    target_gene: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep one requested gene and report other target rows separately."""
    gene_column = _resolve_column(dataframe, "target_name")
    if gene_column is None:
        available = ", ".join(dataframe.columns) or "<no columns>"
        raise ValueError(
            "No target gene column was found. Expected one of "
            f"{_COLUMN_SYNONYMS['target_name']}; available columns: {available}"
        )

    gene_values = dataframe[gene_column].astype("string").str.strip().str.upper()
    data_rows = gene_values.notna() & ~gene_values.isin({"", "NAN", "NONE"})
    requested = target_gene.strip().upper()
    selected_mask = data_rows & gene_values.eq(requested)
    mismatch_mask = data_rows & ~selected_mask

    observed = sorted(gene_values[data_rows].dropna().unique())
    if not selected_mask.any():
        raise ValueError(
            f"Target gene '{requested}' was not found. Observed targets: {observed}"
        )

    selected = dataframe.loc[selected_mask].copy()
    selected[gene_column] = requested
    mismatched = dataframe.loc[mismatch_mask].copy()
    reasons = [
        f"target_mismatch:{value}"
        for value in gene_values.loc[mismatch_mask].astype(str)
    ]
    rejected = _rejected_copy(mismatched, reasons)
    return selected, rejected


def _is_qc_failure(column: str, value: object) -> bool:
    """Interpret explicit QuantStudio QC columns and generic flag columns."""
    token = "" if pd.isna(value) else str(value).strip().lower()
    if token in _QC_PASS_TOKENS:
        return False

    if column in _QC_COLUMNS:
        # Explicit HIGHSD/NOAMP/etc. columns are expected to be Y/N or True/False.
        # An unfamiliar non-empty value is treated conservatively as a failure.
        return True

    if token in _QC_FAILURE_TOKENS:
        return True
    return any(failure in token for failure in _QC_COLUMNS)


def _apply_quality_control(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reject instrument failures while retaining blank/N/PASS flag values."""
    qc_columns = [
        column
        for column in dataframe.columns
        if column in _QC_COLUMNS
        or "flag" in column
        or "status" in column
        or "omit" in column
    ]
    if not qc_columns:
        return dataframe.copy(), pd.DataFrame()

    reasons_by_index: dict[object, list[str]] = {}
    for index, row in dataframe[qc_columns].iterrows():
        reasons: list[str] = []
        for column in qc_columns:
            value = row[column]
            if _is_qc_failure(column, value):
                rendered_value = str(value).strip()
                reasons.append(f"qc:{column}={rendered_value}")
        if reasons:
            reasons_by_index[index] = reasons

    rejected_indices = list(reasons_by_index)
    accepted = dataframe.drop(index=rejected_indices).copy()
    rejected = dataframe.loc[rejected_indices].copy()
    if not rejected.empty:
        rejected["_rejection_reason"] = [
            ";".join(reasons_by_index[index]) for index in rejected.index
        ]
    return accepted, rejected


def clean_qpcr_data(
    dataframe: pd.DataFrame,
    target_gene: str = "C5",
) -> pd.DataFrame:
    """Normalize headers, select a gene, and remove instrument QC failures."""
    cleaned = _normalize_dataframe_columns(dataframe)
    cleaned = _replace_null_tokens(cleaned)
    selected, _ = _filter_target(cleaned, target_gene)
    accepted, rejected = _apply_quality_control(selected)
    if not rejected.empty:
        logger.warning("Rejected %d row(s) for instrument QC flags", len(rejected))
    return accepted.reset_index(drop=True)


def _normalize_salinity(series: pd.Series) -> pd.Series:
    """Preserve categorical conditions while keeping fully numeric ppt numeric."""
    text = series.astype("string").str.strip()
    text = text.replace(
        {
            "low": "Low",
            "med": "Med",
            "medium": "Med",
            "high": "High",
            "control": "Control",
        }
    )
    numeric = pd.to_numeric(text, errors="coerce")
    populated = text.notna()
    if populated.any() and numeric.loc[populated].notna().all():
        return numeric
    return text


def _normalize_to_schema_with_rejections(
    dataframe: pd.DataFrame,
    *,
    target_gene: str,
    time_point: Optional[float] = None,
    salinity: Optional[object] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Project rows onto the ASTRA contract and report invalid values."""
    working = _fill_override(dataframe.copy(), "time_point", time_point)
    working = _fill_override(working, "salinity", salinity)

    mapped = pd.DataFrame(index=working.index)
    mapped["target_name"] = (
        _get_series(working, "target_name", target_gene)
        .astype("string")
        .str.strip()
        .str.upper()
    )
    mapped["time_point"] = pd.to_numeric(
        _get_series(working, "time_point", pd.NA),
        errors="coerce",
    )
    mapped["salinity"] = _normalize_salinity(
        _get_series(working, "salinity", pd.NA)
    )
    mapped["fold_change_rq"] = pd.to_numeric(
        _get_series(working, "fold_change_rq", pd.NA),
        errors="coerce",
    )
    mapped["variance_sd"] = pd.to_numeric(
        _get_series(working, "variance_sd", pd.NA),
        errors="coerce",
    )

    reasons_by_index: dict[object, list[str]] = {}
    required_fields = TARGET_SCHEMA_COLUMNS
    for index, row in mapped.iterrows():
        reasons = [
            f"missing_or_invalid:{field}"
            for field in required_fields
            if pd.isna(row[field]) or str(row[field]).strip() == ""
        ]
        if pd.notna(row["fold_change_rq"]) and row["fold_change_rq"] < 0:
            reasons.append("invalid:fold_change_rq_negative")
        if pd.notna(row["variance_sd"]) and row["variance_sd"] < 0:
            reasons.append("invalid:variance_sd_negative")
        if reasons:
            reasons_by_index[index] = reasons

    rejected_indices = list(reasons_by_index)
    accepted = mapped.drop(index=rejected_indices).copy()
    rejected = working.loc[rejected_indices].copy()
    if not rejected.empty:
        rejected["_rejection_reason"] = [
            ";".join(reasons_by_index[index]) for index in rejected.index
        ]

    if not accepted.empty:
        whole_time_points = np.isclose(
            accepted["time_point"],
            accepted["time_point"].round(),
        ).all()
        if whole_time_points:
            accepted["time_point"] = accepted["time_point"].round().astype("int64")

    return accepted, rejected


def normalize_to_schema(
    dataframe: pd.DataFrame,
    target_gene: str = "C5",
    *,
    time_point: Optional[float] = None,
    salinity: Optional[object] = None,
) -> pd.DataFrame:
    """Map cleaned data onto the five canonical ASTRA columns."""
    normalized = _normalize_dataframe_columns(dataframe)
    normalized = _replace_null_tokens(normalized)
    accepted, rejected = _normalize_to_schema_with_rejections(
        normalized,
        target_gene=target_gene,
        time_point=time_point,
        salinity=salinity,
    )
    if not rejected.empty:
        logger.warning(
            "Dropped %d row(s) with missing or invalid schema values",
            len(rejected),
        )
    return accepted.reset_index(drop=True)[TARGET_SCHEMA_COLUMNS]


def _aggregate_technical_replicates(
    canonical: pd.DataFrame,
    source_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Collapse surviving wells to one row per sample/condition/time point."""
    if canonical.empty:
        return canonical.reset_index(drop=True)

    sample_column = _resolve_column(source_rows, "sample_name")
    if sample_column is None:
        return canonical.reset_index(drop=True)

    samples = source_rows.loc[canonical.index, sample_column].map(_normalize_identifier)
    if samples.isna().all():
        return canonical.reset_index(drop=True)

    aggregated = canonical.copy()
    aggregated["_sample_name"] = samples
    # Missing sample IDs must not cause unrelated rows to collapse together.
    missing_samples = aggregated["_sample_name"].isna()
    aggregated.loc[missing_samples, "_sample_name"] = [
        f"__ROW_{index}" for index in aggregated.index[missing_samples]
    ]

    group_columns = [
        "target_name",
        "time_point",
        "salinity",
        "_sample_name",
    ]
    before = len(aggregated)
    aggregated = (
        aggregated.groupby(group_columns, dropna=False, sort=False, as_index=False)
        .agg(
            fold_change_rq=("fold_change_rq", "mean"),
            variance_sd=("variance_sd", "mean"),
        )
        .drop(columns="_sample_name")
    )
    if len(aggregated) != before:
        logger.info(
            "Aggregated %d accepted wells into %d sample-level rows",
            before,
            len(aggregated),
        )
    return aggregated[TARGET_SCHEMA_COLUMNS].reset_index(drop=True)


def process_qpcr_data(
    dataframe: pd.DataFrame,
    *,
    target_gene: str = "C5",
    time_point: Optional[float] = None,
    salinity: Optional[object] = None,
    aggregate_replicates: bool = True,
) -> ParserResult:
    """Process a loaded table without reading or writing files."""
    working = _normalize_dataframe_columns(dataframe)
    working = _replace_null_tokens(working)

    selected, target_rejected = _filter_target(working, target_gene)
    qc_accepted, qc_rejected = _apply_quality_control(selected)
    canonical, schema_rejected = _normalize_to_schema_with_rejections(
        qc_accepted,
        target_gene=target_gene,
        time_point=time_point,
        salinity=salinity,
    )

    data = (
        _aggregate_technical_replicates(canonical, qc_accepted)
        if aggregate_replicates
        else canonical.reset_index(drop=True)[TARGET_SCHEMA_COLUMNS]
    )
    rejected_frames = [
        frame
        for frame in (target_rejected, qc_rejected, schema_rejected)
        if not frame.empty
    ]
    rejected = (
        pd.concat(rejected_frames, ignore_index=True, sort=False)
        if rejected_frames
        else pd.DataFrame(columns=["_rejection_reason"])
    )
    return ParserResult(data=data, rejected=rejected)


def run_pipeline(
    input_file: PathLike,
    output_file: PathLike = "cleaned_qpcr_data.csv",
    *,
    target_gene: str = "C5",
    time_point: Optional[float] = None,
    salinity: Optional[object] = None,
    results_sheet: str = DEFAULT_RESULTS_SHEET,
    setup_sheet: Optional[str] = DEFAULT_SETUP_SHEET,
    rejected_output_file: Optional[PathLike] = None,
    aggregate_replicates: bool = True,
    allow_empty: bool = False,
) -> pd.DataFrame:
    """Run the complete ASTRA parser and write accepted/rejected CSV files."""
    input_path = Path(input_file)
    logger.info("Loading raw data from %s", input_path)

    if input_path.suffix.lower() in (".xlsx", ".xls"):
        raw_dataframe = load_workbook_data(
            input_path,
            results_sheet=results_sheet,
            setup_sheet=setup_sheet,
            time_point=time_point,
            salinity=salinity,
        )
    else:
        raw_dataframe = load_raw_data(input_path)

    raw_dataframe["_source_file"] = str(input_path)
    result = process_qpcr_data(
        raw_dataframe,
        target_gene=target_gene,
        time_point=time_point,
        salinity=salinity,
        aggregate_replicates=aggregate_replicates,
    )

    if rejected_output_file is not None:
        rejected_path = Path(rejected_output_file)
        rejected_path.parent.mkdir(parents=True, exist_ok=True)
        result.rejected.to_csv(rejected_path, index=False)
        logger.info(
            "Wrote %d rejected row(s) to %s",
            len(result.rejected),
            rejected_path,
        )

    if result.data.empty and not allow_empty:
        raise ValueError(
            "The parser produced zero accepted rows. Review the rejection report "
            "and provide any missing --time-point or --salinity metadata."
        )

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.data.to_csv(output_path, index=False)
    logger.info(
        "Exported %d accepted row(s) for target %s to %s",
        len(result.data),
        target_gene.upper(),
        output_path,
    )
    return result.data


def _default_rejection_path(output_file: PathLike) -> Path:
    output_path = Path(output_file)
    return output_path.with_name(f"{output_path.stem}_rejected.csv")


def _prompt_for_default_workbook_time_point(
    input_file: PathLike,
    time_point: Optional[float],
) -> Optional[float]:
    """Prompt for metadata absent from the default lab workbook in a terminal."""
    if (
        time_point is not None
        or Path(input_file).name != DEFAULT_INPUT_FILE
        or not sys.stdin.isatty()
    ):
        return time_point

    entered = input(
        "lab_data.xls does not contain an experiment time point. "
        "Enter the numeric time point: "
    ).strip()
    if not entered:
        return None

    try:
        return float(entered)
    except ValueError as exc:
        raise ValueError(
            f"Time point must be numeric; received {entered!r}"
        ) from exc


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize a qPCR CSV/Excel export for the ASTRA grid and narrator."
        )
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default=DEFAULT_INPUT_FILE,
        help=f"Raw .csv, .xlsx, or .xls export (default: {DEFAULT_INPUT_FILE})",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="parsed_output.csv",
        help="Accepted-row CSV (default: parsed_output.csv)",
    )
    parser.add_argument(
        "-t",
        "--target",
        default="C5",
        help="Target gene to retain, such as C5, C3, or Actin",
    )
    parser.add_argument(
        "--time-point",
        type=float,
        help="Run-level time point when the workbook does not contain one",
    )
    parser.add_argument(
        "--salinity",
        help="Run-level salinity/condition when setup metadata does not contain one",
    )
    parser.add_argument(
        "--results-sheet",
        default=DEFAULT_RESULTS_SHEET,
        help="Measurement worksheet (default: Results)",
    )
    parser.add_argument(
        "--setup-sheet",
        default=DEFAULT_SETUP_SHEET,
        help="Metadata worksheet to merge (default: Sample Setup)",
    )
    parser.add_argument(
        "--rejected-output",
        help="Rejected-row CSV (default: <output>_rejected.csv)",
    )
    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Keep one accepted row per well instead of aggregating by sample",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Permit a header-only accepted CSV",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point."""
    arguments = _build_cli_parser().parse_args(argv)
    try:
        arguments.time_point = _prompt_for_default_workbook_time_point(
            arguments.input_file,
            arguments.time_point,
        )
    except ValueError as exc:
        logger.error("ASTRA parser failed: %s", exc)
        return 1

    rejected_output = (
        Path(arguments.rejected_output)
        if arguments.rejected_output
        else _default_rejection_path(arguments.output)
    )

    try:
        result = run_pipeline(
            input_file=arguments.input_file,
            output_file=arguments.output,
            target_gene=arguments.target,
            time_point=arguments.time_point,
            salinity=arguments.salinity,
            results_sheet=arguments.results_sheet,
            setup_sheet=arguments.setup_sheet,
            rejected_output_file=rejected_output,
            aggregate_replicates=not arguments.no_aggregate,
            allow_empty=arguments.allow_empty,
        )
    except Exception as exc:
        logger.error("ASTRA parser failed: %s", exc)
        return 1

    print("\nASTRA parser complete")
    print(f"Accepted rows: {len(result)}")
    print(f"Accepted output: {arguments.output}")
    print(f"Rejected output: {rejected_output}")
    print(result.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
