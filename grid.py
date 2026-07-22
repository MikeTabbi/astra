
import pandas as pd
import matplotlib.colors as mcolors

def create_pivot_matrix(df: pd.DataFrame, index_col: str, columns_col: str, value_col: str = 'fold_change_rq') -> pd.DataFrame:
    """
    Generates a programmatic pivot table mapping multi-variable parameters[cite: 1].
    Averages nested replicate samples per intersection[cite: 1].
    """
    return df.pivot_table(
        index=index_col,
        columns=columns_col,
        values=value_col,
        aggfunc='mean'
    )

def rq_to_hex_color(val: float, min_val: float = 0.0, mid_val: float = 1.0, max_val: float = 19.0) -> str:
    """
    Maps RQ values to CSS hex colors[cite: 1]:
    - RQ < 1.0 (Downregulation / Transcriptional collapse): Blue scale[cite: 1]
    - RQ == 1.0 (Baseline): White (#FFFFFF)[cite: 1]
    - RQ > 1.0 (Upregulation): Red scale up to max value (e.g., 19.0)[cite: 1]
    """
    if pd.isna(val):
        return "#E0E0E0"

    down_cmap = mcolors.LinearSegmentedColormap.from_list("down", ["#2B83BA", "#FFFFFF"])
    up_cmap = mcolors.LinearSegmentedColormap.from_list("up", ["#FFFFFF", "#D7191C"])

    if val <= mid_val:
        norm = max(0.0, (val - min_val) / (mid_val - min_val)) if mid_val > min_val else 0.0
        return mcolors.to_hex(down_cmap(norm))
    else:
        norm = min(1.0, (val - mid_val) / (max_val - mid_val)) if max_val > mid_val else 1.0
        return mcolors.to_hex(up_cmap(norm))

def apply_grid_color_mapping(pivot_df: pd.DataFrame, max_rq: float = 19.0) -> pd.DataFrame:
    """
    Applies mathematical threshold scaling to assign CSS hex color codes across the matrix[cite: 1].
    """
    return pivot_df.applymap(lambda val: rq_to_hex_color(val, max_val=max_rq))