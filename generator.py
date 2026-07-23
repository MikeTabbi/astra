import pandas as pd
import numpy as np

def generate_synthetic_data(filename="synthetic_data.csv", rows=1000):
    data = {
        "target_name": ["C5"] * rows,
        "time_point": np.random.choice([0, 6, 12, 24, 48, 72], size=rows),
        "salinity": np.random.choice([10, 20, 30, 40], size=rows),
        "fold_change_rq": np.random.uniform(0.5, 10.0, size=rows),
        "variance_sd": np.random.uniform(0.01, 0.2, size=rows)
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Generated {rows} rows of synthetic data in {filename}")

if __name__ == "__main__":
    generate_synthetic_data()