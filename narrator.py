import logging
import pandas as pd
import json
from typing import Literal
from ollama import Client
from pydantic import BaseModel, Field, ValidationError


HOST_URL = "http://127.0.0.1:11434"
NGROK_URL = ""
client = Client(host=HOST_URL) #local ollama host for now

class Summ(BaseModel):
    target_name: str
    upregulated: bool
    peak_time: int
    confidence: str

def generate_narration_with_self_consistency(matrix_data: dict) -> dict:
    """
    Prompts the local Ollama model to analyze the qPCR matrix and forces 
    the output to match the strict Pydantic Summ schema.
    """
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