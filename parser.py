"""Parser for qPCR export files (QuantStudio, Bio-Rad, etc.).

Pipeline: load_raw_data() -> clean_qpcr_data() -> normalize_to_schema().

Emits the Project ASTRA unified table schema (spec Stage 1):
    target_name    string (e.g. C5) - uppercased
    time_point     integer 1 through 7
    salinity       string (Low, Med, High)
    fold_change_rq float
    variance_sd    float
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

SALINITY_LEVELS: List[str] = ["Low", "Med", "High"]

# Instrument sheets record salinity in ppt; the unified schema wants categories.
# TODO: confirm these cut points with Dr. Todd before the Weeks 3-4 milestone.
# Read as: ppt < 20 -> Low, 20 <= ppt < 35 -> Med, ppt >= 35 -> High.
SALINITY_CUT_POINTS: List[float] = [20.0, 35.0]

# Text spellings that may already appear in a salinity column.
_SALINITY_ALIASES: Dict[str, str] = {
    "low": "Low", "l": "Low", "control": "Low", "ctrl": "Low",
    "med": "Med", "medium": "Med", "mid": "Med", "m": "Med", "moderate": "Med",
    "high": "High", "hi": "High", "h": "High",
}

# Raw time values (e.g. hours) mapped onto the schema's ordinal 1-7 scale.
# Set to None to pass through values that are already 1-7.
# TODO: confirm the lab's 7 sampling times with Dr. Todd.
TIME_POINT_MAP: Optional[Dict[float, int]] = None


def _normalize_column_name(name: object) -> str:
    """Lowercase, strip, and underscore-ify a raw column label."""
    return str(name).strip().lower().replace(" ", "_")


def _resolve_column(df: pd.DataFrame, canonical: str) -> Optional[str]:
    """Find the column in `df` matching a canonical field name, or None."""
    for candidate in _COLUMN_SYNONYMS.get(canonical, [canonical]):
        if candidate in df.columns:
            return candidate
    return None


def _get_series(df: pd.DataFrame, canonical: str, default: object) -> pd.Series:
    """Return the resolved column, or a full-length Series of `default`."""
    col = _resolve_column(df, canonical)
    if col is not None:
        return df[col]
    return pd.Series(default, index=df.index)


def _to_salinity_level(value: object) -> object:
    """Map one raw salinity value (ppt number or text) to Low/Med/High.

    Numeric values are bucketed by SALINITY_CUT_POINTS; text values are matched
    against known spellings. Unrecognized values become NaN so the row is dropped.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan

    text = str(value).strip().lower()
    if text in _SALINITY_ALIASES:
        return _SALINITY_ALIASES[text]

    numeric = pd.to_numeric(text, errors="coerce")
    if pd.isna(numeric):
        return np.nan

    low_cut, high_cut = SALINITY_CUT_POINTS
    if numeric < low_cut:
        return "Low"
    if numeric < high_cut:
        return "Med"
    return "High"


def _to_time_point(value: object) -> object:
    """Map one raw time value onto the schema's ordinal 1-7 scale."""
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return np.nan

    if TIME_POINT_MAP is not None:
        return TIME_POINT_MAP.get(float(numeric), np.nan)

    ordinal = int(round(float(numeric)))
    if 1 <= ordinal <= 7:
        return ordinal
    return np.nan


def _detect_header_row(raw_rows: List[List[str]], expected_tokens: Optional[List[str]] = None) -> int:
    """Auto-detect the index of the real header row amid leading metadata lines.

    Many qPCR instruments (QuantStudio, Bio-Rad CFX) prepend 10-20 lines of run
    metadata before the actual data table header. This scans for the first row
    that looks like a header: several non-empty cells, ideally with a
    recognizable qPCR column name. Defaults to row 0 if nothing better is found.
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
    """Load a qPCR export (.csv/.xlsx/.xls), auto-skipping any metadata banner.

    CSV decoding falls back through utf-8 -> latin-1 -> cp1252.

    Raises:
        FileNotFoundError: if the path does not exist.
        ValueError: on an unsupported extension, or if no encoding parses the file.
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
                # scan raw rows with csv.reader (tolerant of inconsistent column
                # counts) purely to locate the header, then let pandas parse the
                # real table with a plain comma delimiter.
                with path.open("r", encoding=encoding, newline="") as fh:
                    raw_rows = list(csv.reader(fh))
                header_idx = _detect_header_row(raw_rows)
                # skip_blank_lines=False keeps pandas' row numbering aligned with
                # the csv.reader scan above (which counts blank lines).
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
    """Normalize column names, isolate the target gene, and drop flagged rows.

    Gene matching is case-insensitive and whitespace-tolerant. Rows carrying
    instrument QC failures (NOAMP/EXPFAIL/OMIT/etc.) are removed, and remaining
    null-token cells are blanked so downstream coercion treats them as missing.
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

    for col in df.columns:
        as_str = df[col].astype(str).str.strip().str.lower()
        df.loc[as_str.isin(_NULL_TOKENS), col] = np.nan

    return df.reset_index(drop=True)


def normalize_to_schema(df: pd.DataFrame, target_gene: str = "C5") -> pd.DataFrame:
    """Map cleaned qPCR data onto the unified ASTRA schema.

    target_name is uppercased, time_point is coerced to the ordinal 1-7 scale,
    and salinity is bucketed into Low/Med/High. Rows with any unusable value in
    those fields are dropped and the index is reset.
    """
    mapped_df = pd.DataFrame(index=df.index)

    mapped_df["target_name"] = (
        _get_series(df, "target_name", default=target_gene).astype(str).str.strip().str.upper()
    )
    mapped_df["time_point"] = _get_series(df, "time_point", default=np.nan).map(_to_time_point)
    mapped_df["salinity"] = _get_series(df, "salinity", default=np.nan).map(_to_salinity_level)
    mapped_df["fold_change_rq"] = pd.to_numeric(
        _get_series(df, "fold_change_rq", default=np.nan), errors="coerce"
    )
    mapped_df["variance_sd"] = pd.to_numeric(
        _get_series(df, "variance_sd", default=np.nan), errors="coerce"
    )

    before = len(mapped_df)
    mapped_df = mapped_df.dropna(subset=TARGET_SCHEMA_COLUMNS)
    dropped = before - len(mapped_df)
    if dropped:
        logger.warning("Dropped %d row(s) with unparseable/missing schema values", dropped)

    if not mapped_df.empty:
        mapped_df["time_point"] = mapped_df["time_point"].astype(int)
        unexpected = sorted(set(mapped_df["salinity"]) - set(SALINITY_LEVELS))
        if unexpected:
            logger.warning("Unexpected salinity levels after mapping: %s", unexpected)

    mapped_df = mapped_df.reset_index(drop=True)
    return mapped_df[TARGET_SCHEMA_COLUMNS]


def run_pipeline(
    input_file: Union[str, Path],
    output_file: Union[str, Path] = "cleaned_qpcr_data.csv",
    target_gene: str = "C5",
) -> pd.DataFrame:
    """Run load -> clean -> normalize and write the result to CSV.

    Gene-agnostic: pass target_gene to process C5, C3, Actin, or any other
    target with no code changes (spec Stage 1).
    """
    logger.info("Loading raw data from %s", input_file)
    raw_df = load_raw_data(input_file)

    logger.info("Cleaning quality flags and isolating target gene %s", target_gene)
    cleaned_df = clean_qpcr_data(raw_df, target_gene=target_gene)

    logger.info("Normalizing schema")
    final_df = normalize_to_schema(cleaned_df, target_gene=target_gene)

    final_df.to_csv(output_file, index=False)
    logger.info("Exported %d clean rows to %s", len(final_df), output_file)
    return final_df


if __name__ == "__main__":
    import argparse
    import tempfile

    ap = argparse.ArgumentParser(description="ASTRA parser: ingest raw qPCR exports.")
    ap.add_argument("input_file", nargs="?", help="Raw export file. Omit to run the self-test.")
    ap.add_argument("-g", "--gene", default="C5", help="Target gene (C5, C3, Actin, ...)")
    ap.add_argument("-o", "--output", default="parsed_output.csv")
    args = ap.parse_args()

    if args.input_file:
        run_pipeline(args.input_file, args.output, args.gene)
        raise SystemExit(0)

    # --- Self-test: mock a malformed qPCR export in memory. ---
    # Simulates a leading metadata banner, mixed-case/aliased headers,
    # 'Undetermined'/'NOAMP' hardware flags, blank/garbage numeric cells,
    # ppt salinity values, and a pre-categorized salinity spelling.
    mock_csv_lines = [
        "Block Type,QuantStudio 7 Flex",
        "Experiment File Name,run_2024_03_01.eds",
        "Instrument Type,QuantStudio",
        "Software Version,1.7.2",
        "",
        "Target Name,Time Point,Salinity,Fold Change,Std Dev,Flag",
        "C5,1,10,1.02,0.05,",
        "c5, 2 ,25,2.31,0.11,",
        "C5,3,45,Undetermined,0.20,NOAMP",
        "C5,4,45,4.87,,",
        "C5,5,10,garbage,0.30,EXPFAIL",
        "GAPDH,1,10,1.00,0.01,",
        "C5,6, High , 6.02 ,0.15,",
        "C5,99,25,3.10,0.09,",
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
        assert (final["target_name"] == "C5").all(), "target_name not uppercased"
        assert set(final["salinity"]) <= set(SALINITY_LEVELS), "salinity not categorical"
        assert final["time_point"].between(1, 7).all(), "time_point outside 1-7"
        assert str(final["time_point"].dtype).startswith("int"), "time_point not integer"
        print("Self-test passed: output conforms to the unified ASTRA schema.")
    finally:
        Path(mock_path).unlink(missing_ok=True)