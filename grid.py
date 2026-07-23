import matplotlib.colors as mcolors
import pandas as pd


def create_expression_matrix(df, row_axis="salinity", col_axis="time_point"):
    """Pivots normalized dataframe and averages nested RQ values[cite: 1]."""
    return df.pivot_table(
        index=row_axis,
        columns=col_axis,
        values="fold_change_rq",
        aggfunc="mean",
    )


def get_css_color(value, vmin=0.0, vmax=19.0):
    """Maps numerical RQ value to CSS hex color code across threshold scale[cite: 1]."""
    if pd.isna(value):
        return "#FFFFFF"

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "rq_scale", ["#2b83ba", "#ffffff", "#d7191c"]
    )
    return mcolors.to_hex(cmap(norm(value))[:3])


def generate_color_mapped_grid(matrix):
    """Transforms analytical matrix into corresponding CSS hex color grid[cite: 1]."""
    return matrix.applymap(get_css_color)