from ollama import Client
import json
from typing import List, Optional
from pydantic import BaseModel, ValidationError

MODEL_NAME = "qwen2.5-coder:7b"
HOST_URL = "http://127.0.0.1:11434"
NGROK_URL = ""
N_CONSISTENCY_RUNS = 6

client = Client(host=HOST_URL) #local ollama host for now

#No functions inside class
class Summ(BaseModel):
    target_name: str
    upregulated: bool
    peak_time: int
    confidence: str

def prompt_builder(matrix_data: dict) -> str:
    "Matrix output is the data form grid.py"

    schema_hint = {
        "target_name" : "string",
        "upregulated" : "true/false",
        "peak_time" : "# (1-7)",
        "confidence" : "string",
    }

    return f"""You are a molecular biology data analyst reviewing qPCR fold-change
(RQ) results. Analyze the following experimental data and describe the trend
factually and conservatively. Do not speculate beyond what the numbers show.
DATA:
{json.dumps(matrix_data, indent=2)}

Respond with ONLY a JSON object matching this exact schema, no other text:
{json.dumps(schema_hint, indent=2)}
"""

def generate_narration_once(matrix_data: dict) -> Optional[Summ]:
    prompt = prompt_builder(matrix_data)
    try:
        response = client.chat(
            model= MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            format="json",  # Ollama's native JSON-schema-mapping mode
        )
        raw_text = response["message"]["content"]
        parsed = json.loads(raw_text)
        return Summ(**parsed)
    except (json.JSONDecodeError, ValidationError, KeyError) as e:
        print(f"  [run failed to parse/validate]: {e}")
        return None
    
def generate_narration_with_self_consistency(matrix_data: dict) -> dict:
    #runs multiple times for confidence.

    successful_runs: List[Summ] = []

    for i in range(N_CONSISTENCY_RUNS):
        print(f"Run {i + 1}/{N_CONSISTENCY_RUNS}...")
        result = generate_narration_once(matrix_data)
        if result is not None:
            successful_runs.append(result)

    if not successful_runs:
        return {
            "status": "error",
            "message": "All runs failed to produce valid, schema-conforming JSON.",
        }

    directions = [r.upregulated for r in successful_runs]
    peaks = [r.peak_time_point for r in successful_runs]

    direction_agrees = len(set(directions)) == 1
    peak_spread = max(peaks) - min(peaks)  # 0 = perfect agreement

    if direction_agrees and peak_spread <= 1:
        confidence = "HIGH"
    elif direction_agrees:
        confidence = "MEDIUM — peak timing varies, spot-check recommended"
    else:
        confidence = "LOW — FLAGGED FOR MANUAL REVIEW (contradictory runs)"

    return {
        "target_name": successful_runs[0].target_name,
        "n_successful_runs": len(successful_runs),
        "n_requested_runs": N_CONSISTENCY_RUNS,
        "consistency_confidence": confidence,
        "representative_summary": successful_runs[0].trend_summary,
        "all_run_outputs": [r.model_dump() for r in successful_runs],
    }

#Smoke Test
if __name__ == "__main__":
    # Test call to your MacBook's Ollama instance
    response = client.chat(
        model='qwen2.5-coder:7b',
        messages=[{'role': 'user', 'content': 'Checking connection to host MacBook...'}]
    )
    print(response['message']['content'])
    print("-" * 60)
    sample_matrix = {
        "target_name": "C5",
        "salinity": "High",
        "time_points": [1, 2, 3, 4, 5, 6, 7],
        "fold_change_rq": [1.0, 1.4, 2.1, 3.8, 5.2, 4.9, 4.5],
    }
    output = generate_narration_with_self_consistency(sample_matrix)
    print(json.dumps(output, indent=2))



