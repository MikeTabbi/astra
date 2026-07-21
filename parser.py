"""Parser for qPCR export files (QuantStudio, Bio-Rad, etc.).

Pipeline: load_raw_data() -> clean_qpcr_data() -> normalize_to_schema().
"""

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Values emitted by qPCR instruments/software that mean "no usable numeric result".
_NULL_TOKENS = {
    "", "na", "n/a", "nan", "none", "null",
    "undetermined", "noamp", "expfail", "omit",
}

# Column-name synonyms, keyed by canonical (normalized) name.
_COLUMN_SYNONYMS: Dict[str, List[str]] = {
    "target_name": ["target_name", "target", "gene", "gene_name", "detector", "detector_name"],
    "time_point": ["time_point", "time", "timepoint", "time_pt"],
    "salinity": ["salinity", "salt_ppt", "salt", "ppt"],
    "fold_change_rq": ["fold_change_rq", "fold_change", "rq", "rel_quant", "relative_quantity"],
    "variance_sd": ["variance_sd", "sd", "std_dev", "stdev", "std", "variance"],
    "ct": ["ct", "cq", "ct_mean", "cq_mean"],
}

TARGET_SCHEMA_COLUMNS: List[str] = [
    "target_name", "time_point", "salinity", "fold_change_rq", "variance_sd",
]


def _normalize_column_name(name: object) -> str:
    """Lowercase/strip/underscore-ify a raw column label.

    Args:
        name: Raw column label as read from the file (may be any type pandas
            produces for a header cell, e.g. str, int, float('nan')).

    Returns:
        str: Normalized column name (lowercase, stripped, spaces -> underscores).
    """
    return str(name).strip().lower().replace(" ", "_")


def _resolve_column(df: pd.DataFrame, canonical: str) -> Optional[str]:
    """Find the actual column in `df` matching a canonical field name.

    Args:
        df: DataFrame whose columns have already been normalized via
            `_normalize_column_name`.
        canonical: Key into `_COLUMN_SYNONYMS` (e.g. "target_name").

    Returns:
        Optional[str]: The matching column name in `df`, or None if no
            synonym is present.
    """
    for candidate in _COLUMN_SYNONYMS.get(canonical, [canonical]):
        if candidate in df.columns:
            return candidate
    return None


def _get_series(df: pd.DataFrame, canonical: str, default: object) -> pd.Series:
    """Return the resolved column as a Series, or a full-length default column.

    Args:
        df: Source DataFrame with normalized column names.
        canonical: Canonical field name to resolve via `_COLUMN_SYNONYMS`.
        default: Scalar value to fill every row with if no matching column exists.

    Returns:
        pd.Series: The matched column, or a constant Series of `default`.
    """
    col = _resolve_column(df, canonical)
    if col is not None:
        return df[col]
    return pd.Series(default, index=df.index)


def _detect_header_row(raw_rows: List[List[str]], expected_tokens: Optional[List[str]] = None) -> int:
    """Auto-detect the index of the real header row amid leading metadata lines.

    Many qPCR instruments (QuantStudio, Bio-Rad CFX) prepend 10-20 lines of
    run metadata (block ID, software version, timestamps, etc.) before the
    actual data table header. This scans rows for the first one that looks
    like a header: several non-empty cells, and ideally a recognizable
    qPCR column name.

    Args:
        raw_rows: Rows of raw string cells, as read without a header, in
            file order.
        expected_tokens: Optional list of lowercase tokens (e.g. "target",
            "ct", "cq") used to positively identify the header row. Falls
            back to the union of all known column synonyms when omitted.

    Returns:
        int: Row index (0-based) of the detected header row. Defaults to 0
            if no better candidate is found (i.e. assume no metadata banner).
    """
    if expected_tokens is None:
        expected_tokens = [
            token for synonyms in _COLUMN_SYNONYMS.values() for token in synonyms
        ]

    best_idx = 0
    best_score = -1
    for idx, row in enumerate(raw_rows):
        cells = [_normalize_column_name(c) for c in row if str(c).strip() != ""]
        if len(cells) < 2:
            continue
        score = sum(1 for cell in cells if any(tok in cell for tok in expected_tokens))
        # A row with recognizable qPCR column names always wins outright.
        if score > 0 and score >= best_score:
            best_score = score
            best_idx = idx

    return best_idx


def load_raw_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load a qPCR export file, auto-skipping any leading metadata banner.

    Supports .csv (utf-8 with latin-1/cp1252 fallback) and Excel (.xlsx/.xls).

    Args:
        file_path: Union[str, Path] path to the export file.

    Returns:
        pd.DataFrame: Raw table with the detected header row applied as columns.

    Raises:
        FileNotFoundError: If `file_path` does not point to an existing file.
        ValueError: If the file extension is not one of .csv/.xlsx/.xls, or
            the file could not be decoded/parsed with any supported strategy.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"qPCR export file not found: {path}")

    suffix = path.suffix.lower()

    if suffix in (".xlsx", ".xls"):
        # Read without a header first so we can locate the real table start.
        raw = pd.read_excel(path, header=None, dtype=str)
        header_idx = _detect_header_row(raw.values.tolist())
        df = pd.read_excel(path, header=header_idx)
        logger.info("Loaded Excel file %s (header row detected at index %d)", path, header_idx)
        return df

    if suffix == ".csv":
        last_error: Optional[Exception] = None
        for encoding in ("utf-8", "latin-1", "cp1252"):
            try:
                # Ragged metadata banners break pandas' delimiter sniffing, so
                # scan raw rows with csv.reader (tolerant of inconsistent
                # column counts) purely to locate the header, then let
                # pandas parse the real table with a plain comma delimiter.
                with path.open("r", encoding=encoding, newline="") as fh:
                    raw_rows = list(csv.reader(fh))
                header_idx = _detect_header_row(raw_rows)
                # skip_blank_lines=False keeps pandas' row numbering aligned
                # with the csv.reader scan above (which counts blank lines).
                df = pd.read_csv(path, header=header_idx, encoding=encoding, skip_blank_lines=False)
                logger.info(
                    "Loaded CSV file %s using encoding=%s (header row detected at index %d)",
                    path, encoding, header_idx,
                )
                return df
            except (UnicodeDecodeError, pd.errors.ParserError) as exc:
                last_error = exc
                logger.warning("Failed to parse %s with encoding=%s: %s", path, encoding, exc)
                continue
        raise ValueError(f"Could not parse CSV {path} with any supported encoding") from last_error

    raise ValueError(f"Unsupported file extension for {path}: expected .csv, .xlsx, or .xls")


def clean_qpcr_data(df: pd.DataFrame, target_gene: str = "C5") -> pd.DataFrame:
    """Normalize column names, isolate the target gene, and drop bad/omitted rows.

    Args:
        df: pd.DataFrame as returned by `load_raw_data`.
        target_gene: str name of the gene/target to keep (case-insensitive,
            whitespace-tolerant), e.g. "C5".

    Returns:
        pd.DataFrame: Filtered copy containing only rows for `target_gene`
            with instrument-flagged failures (NOAMP/EXPFAIL/OMIT/etc.) removed.
    """
    df = df.copy()
    df.columns = [_normalize_column_name(col) for col in df.columns]

    gene_col = _resolve_column(df, "target_name")
    if gene_col is None:
        available = ", ".join(df.columns) or "<no columns>"
        logger.warning(
            "No target gene column found (expected one of %s). Available columns: %s",
            _COLUMN_SYNONYMS["target_name"], available,
        )
    else:
        gene_values = df[gene_col].astype(str).str.strip().str.upper()
        mask = gene_values == target_gene.upper()
        if not mask.any():
            observed = sorted(v for v in gene_values.unique() if v not in ("", "NAN"))
            logger.warning(
                "Target gene '%s' not found in column '%s'. Observed values: %s",
                target_gene, gene_col, observed,
            )
        df = df[mask].copy()

    # Drop rows flagged bad by any column whose name suggests a QC/status flag.
    flag_cols = [c for c in df.columns if "flag" in c or "status" in c or "omit" in c]
    for col in flag_cols:
        flag_values = df[col].astype(str).str.strip().str.upper()
        df = df[~flag_values.isin({t.upper() for t in _NULL_TOKENS} | {"TRUE", "1"})]

    # Blank out any remaining null-token cells across all columns so downstream
    # numeric coercion treats them as missing rather than literal strings.
    for col in df.columns:
        as_str = df[col].astype(str).str.strip().str.lower()
        df.loc[as_str.isin(_NULL_TOKENS), col] = np.nan

    return df.reset_index(drop=True)


def normalize_to_schema(df: pd.DataFrame, target_gene: str = "C5") -> pd.DataFrame:
    """Map cleaned qPCR data onto the fixed output schema with numeric coercion.

    Args:
        df: pd.DataFrame as returned by `clean_qpcr_data`.
        target_gene: str fallback value for `target_name` when no gene column
            is present in `df`.

    Returns:
        pd.DataFrame: Columns exactly `['target_name', 'time_point',
            'salinity', 'fold_change_rq', 'variance_sd']`, numeric columns
            coerced with `pd.to_numeric`, rows with any unparseable/missing
            numeric value dropped, and the index reset.
    """
    mapped_df = pd.DataFrame(index=df.index)

    mapped_df["target_name"] = _get_series(df, "target_name", default=target_gene)
    mapped_df["time_point"] = pd.to_numeric(_get_series(df, "time_point", default=np.nan), errors="coerce")
    mapped_df["salinity"] = pd.to_numeric(_get_series(df, "salinity", default=np.nan), errors="coerce")
    mapped_df["fold_change_rq"] = pd.to_numeric(
        _get_series(df, "fold_change_rq", default=np.nan), errors="coerce"
    )
    mapped_df["variance_sd"] = pd.to_numeric(_get_series(df, "variance_sd", default=np.nan), errors="coerce")

    before = len(mapped_df)
    mapped_df = mapped_df.dropna(subset=["time_point", "salinity", "fold_change_rq", "variance_sd"])
    dropped = before - len(mapped_df)
    if dropped:
        logger.warning("Dropped %d row(s) with unparseable/missing numeric values", dropped)

    mapped_df = mapped_df.reset_index(drop=True)
    return mapped_df[TARGET_SCHEMA_COLUMNS]


def run_pipeline(input_file: Union[str, Path], output_file: Union[str, Path] = "cleaned_qpcr_data.csv") -> pd.DataFrame:
    """Run load -> clean -> normalize and write the result to CSV.

    Args:
        input_file: Union[str, Path] path to the raw qPCR export file.
        output_file: Union[str, Path] destination CSV path for the cleaned table.

    Returns:
        pd.DataFrame: The final normalized DataFrame that was written to disk.
    """
    logger.info("Loading raw data from %s", input_file)
    raw_df = load_raw_data(input_file)

    logger.info("Cleaning quality flags and isolating target gene C5")
    cleaned_df = clean_qpcr_data(raw_df, target_gene="C5")

    logger.info("Normalizing schema")
    final_df = normalize_to_schema(cleaned_df)

    final_df.to_csv(output_file, index=False)
    logger.info("Exported %d clean rows to %s", len(final_df), output_file)
    return final_df


if __name__ == "__main__":
    import tempfile

    # --- Lightweight self-test: mock a malformed qPCR export in memory. ---
    # Simulates: a leading metadata banner, mixed-case/aliased headers,
    # 'Undetermined'/'NOAMP' hardware flags, and blank/garbage numeric cells.
    mock_csv_lines = [
        "Block Type,QuantStudio 7 Flex",
        "Experiment File Name,run_2024_03_01.eds",
        "Instrument Type,QuantStudio",
        "Software Version,1.7.2",
        "",
        "Target Name,Time Point,Salinity,Fold Change,Std Dev,Flag",
        "C5,0,10,1.02,0.05,",
        "c5, 6 ,10,2.31,0.11,",
        "C5,12,10,Undetermined,0.20,NOAMP",
        "C5,24,10,4.87,,",
        "C5,48,10,garbage,0.30,EXPFAIL",
        "GAPDH,0,10,1.00,0.01,",
        "C5,72, 10 , 6.02 ,0.15,",
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as tmp:
        tmp.write("\n".join(mock_csv_lines))
        mock_path = tmp.name

    try:
        print(f"Testing malformed mock qPCR file: {mock_path}")
        raw = load_raw_data(mock_path)
        print(f"Raw shape after header-detection load: {raw.shape}")

        cleaned = clean_qpcr_data(raw, target_gene="C5")
        print(f"Cleaned shape (C5 only, flags removed): {cleaned.shape}")

        final = normalize_to_schema(cleaned, target_gene="C5")
        print(f"Final schema columns: {list(final.columns)}")
        print(final)

        assert list(final.columns) == TARGET_SCHEMA_COLUMNS
        assert not final.empty
        print("Self-test passed: parser handled malformed input without crashing.")
    finally:
        Path(mock_path).unlink(missing_ok=True)
