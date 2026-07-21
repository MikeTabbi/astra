import logging
import pandas as pd
from typing import Literal
from ollama import Client
from pydantic import BaseModel, Field, ValidationError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "qwen2.5-coder:7b"


class SyntheticDataPoint(BaseModel):
    target_name: Literal["C5"] = Field(default="C5")
    time_point: float
    salinity: float
    fold_change_rq: float
    variance_sd: float


class SyntheticTrajectory(BaseModel):
    samples: list[SyntheticDataPoint]


def generate_synthetic_data(num_samples: int = 10, output_file: str = "synthetic_data.csv") -> pd.DataFrame:
    if num_samples <= 0:
        raise ValueError("num_samples must be a positive integer")

    client = Client()

    prompt = f"""
    Generate {num_samples} realistic synthetic data rows for a qPCR biological experiment tracking the C5 gene.
    Vary time_point (e.g. 0, 6, 12, 24, 48 hrs), salinity (e.g. 15 to 45 ppt), fold_change_rq, and variance_sd.
    Return ONLY valid JSON matching this schema:
    {{
      "samples": [
        {{"target_name": "C5", "time_point": 12.0, "salinity": 30.0, "fold_change_rq": 1.85, "variance_sd": 0.12}}
      ]
    }}
    """

    logger.info("Generating %d synthetic rows using %s", num_samples, MODEL_NAME)

    response = client.chat(
        model=MODEL_NAME,
        messages=[{'role': 'user', 'content': prompt}],
        format="json",
    )

    content = response['message']['content']

    try:
        parsed_data = SyntheticTrajectory.model_validate_json(content)
    except ValidationError as e:
        logger.error("Model returned data that failed schema validation: %s", e)
        logger.debug("Raw content was: %s", content)
        raise

    if len(parsed_data.samples) != num_samples:
        raise ValueError(f"Expected {num_samples} samples, got {len(parsed_data.samples)}")

    df = pd.DataFrame([sample.model_dump() for sample in parsed_data.samples])

    df.to_csv(output_file, index=False)
    logger.info("Exported %d synthetic rows to %s", len(df), output_file)
    return df


if __name__ == "__main__":
    generate_synthetic_data(num_samples=10)