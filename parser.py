import pandas as pd
import numpy as np
import logging

def load_raw_data(file_path: str) -> pd.DataFrame:
    normalized_path = file_path.lower()
    if normalized_path.endswith(('.xlsx', '.xls')):
        return pd.read_excel(file_path)
    if normalized_path.endswith('.csv'):
        return pd.read_csv(file_path)
    raise ValueError(f"Unsupported file extension for {file_path}")


def clean_qpcr_data(df: pd.DataFrame, target_gene: str = "C5") -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip().lower().replace(" ", "_") for col in df.columns]

    gene_col = "target_name" if "target_name" in df.columns else "gene"
    if gene_col in df.columns:
        df = df[df[gene_col].astype(str).str.strip().str.upper() == target_gene.upper()].copy()

    flag_cols = [c for c in df.columns if "flag" in c or "status" in c or "omit" in c]
    for col in flag_cols:
        df = df[~df[col].astype(str).str.strip().str.upper().isin(["NOAMP", "EXPFAIL", "TRUE", "1"])]

    return df


def _get_series(df: pd.DataFrame, *names, default) -> pd.Series:
    """Return the first matching column as a Series; else a full-length default column."""
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series(default, index=df.index)


def normalize_to_schema(df: pd.DataFrame, target_gene: str = "C5") -> pd.DataFrame:
    mapped_df = pd.DataFrame(index=df.index)

    if "target_name" in df.columns:
        gene_col = "target_name"
    elif "gene" in df.columns:
        gene_col = "gene"
    else:
        raise ValueError("Input data must contain either 'target_name' or 'gene' column")

    df = df[df[gene_col].astype(str).str.strip().str.upper() == target_gene.upper()].copy()

    mapped_df['target_name'] = _get_series(df, 'target_name', 'gene', default='C5')
    mapped_df['time_point'] = pd.to_numeric(_get_series(df, 'time_point', 'time', default=np.nan), errors='coerce')
    mapped_df['salinity'] = pd.to_numeric(_get_series(df, 'salinity', 'salt_ppt', default=np.nan), errors='coerce')
    mapped_df['fold_change_rq'] = pd.to_numeric(
        _get_series(df, 'fold_change_rq', 'rq', default=np.nan), errors='coerce'
    )
    mapped_df['variance_sd'] = pd.to_numeric(_get_series(df, 'variance_sd', 'sd', default=np.nan), errors='coerce')

    mapped_df = mapped_df.dropna().reset_index(drop=True)
    return mapped_df


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def run_pipeline(input_file: str, output_file: str = "cleaned_qpcr_data.csv"):
    logger.info("Loading raw data from %s", input_file)
    raw_df = load_raw_data(input_file)

    logger.info("Cleaning quality flags and isolating target gene C5")
    cleaned_df = clean_qpcr_data(raw_df, target_gene="C5")

    logger.info("Normalizing schema")
    final_df = normalize_to_schema(cleaned_df)

    final_df.to_csv(output_file, index=False)
    logger.info("Exported %d clean rows to %s", len(final_df), output_file)

if __name__ == "__main__":
    run_pipeline("mock_data.csv", "parsed_output.csv")