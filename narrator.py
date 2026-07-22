import logging
import pandas as pd
import json
from typing import Literal
from ollama import Client
from pydantic import BaseModel, Field, ValidationError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HOST_URL = "http://127.0.0.1:11434" 
MODEL_NAME = "qwen2.5-coder:7b"
client = Client(host=HOST_URL)

# --- 1. Schemas ---

# my schema (Mike)for the synthetic data
class SyntheticDataPoint(BaseModel):
    target_name: Literal["C5"] = Field(default="C5")
    time_point: float = Field(ge=0)
    salinity: float = Field(ge=0)
    fold_change_rq: float = Field(ge=0)
    variance_sd: float = Field(ge=0)

class SyntheticTrajectory(BaseModel):
    samples: list[SyntheticDataPoint]

# my schema (Zion) for the analytical summary
class Summ(BaseModel):
    target_name: str
    upregulated: bool
    peak_time: float
    confidence: Literal["Low", "Medium", "High"]


# function to generate synthetic data using the Ollama API
def generate_synthetic_data(num_samples: int = 10, output_file: str = "synthetic_data.csv") -> pd.DataFrame:
    """Generates raw mock qPCR data matching the parser schema."""
    prompt = (
        f"Generate {num_samples} realistic synthetic data rows for a qPCR biological experiment tracking the C5 gene. "
        "Vary time_point (e.g. 0, 6, 12, 24, 48 hrs), salinity (e.g. 15 to 45 ppt), fold_change_rq, and variance_sd."
    )

    logger.info("Generating %d synthetic rows using %s", num_samples, MODEL_NAME)

    response = client.chat(
        model=MODEL_NAME,
        messages=[{'role': 'user', 'content': prompt}],
        # Upgrade: Force Ollama to strictly follow the Pydantic schema
        format=SyntheticTrajectory.model_json_schema(), 
    )

    try:
        parsed_data = SyntheticTrajectory.model_validate_json(response['message']['content'])
        if len(parsed_data.samples) != num_samples:
            raise ValueError(
                f"Expected {num_samples} samples, got {len(parsed_data.samples)} from model response"
            )
        df = pd.DataFrame([sample.model_dump() for sample in parsed_data.samples])
        df.to_csv(output_file, index=False)
        logger.info("Exported %d synthetic rows to %s", len(df), output_file)
        return df
    except ValidationError as e:
        logger.error("Schema Validation Failed: %s", e)
        raise
# function to analyze the generated synthetic data using the Ollama API
def analyze_trajectory(df: pd.DataFrame) -> dict:
    """Reads a generated dataframe and returns an analytical summary."""
    matrix_data = df.to_dict(orient="records")
    prompt = (
        "Analyze the following qPCR time-series matrix. "
        f"Data: {json.dumps(matrix_data)}\n"
        "Return a summary containing the target_name, a boolean indicating if it was upregulated, "
        "the time_point where the fold change peaked, and a confidence string ('Low', 'Medium', 'High')."
    )

    logger.info("Analyzing trajectory to generate summary...")

    response = client.chat(
        model=MODEL_NAME,
        messages=[{'role': 'user', 'content': prompt}],
        format=Summ.model_json_schema()
    )

    try:
        summary = Summ.model_validate_json(response['message']['content'])
        return summary.model_dump()
    except ValidationError as e:
        logger.error("Summary Validation Failed: %s", e)
        raise


if __name__ == "__main__":
    # 1. Generate the data (Mike's Logic)
    synthetic_df = generate_synthetic_data(num_samples=5)
    
    # 2. Analyze the data (Zion's Logic)
    summary_output = analyze_trajectory(synthetic_df)
    
    print("\n--- Final Analytical Summary ---")
    print(json.dumps(summary_output, indent=2))