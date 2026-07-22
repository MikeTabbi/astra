from ollama import Client
import json
from typing import List, Optional
from pydantic import BaseModel, ValidationError


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
        "Return a JSON object containing the target_name, a boolean indicating if it was upregulated, "
        "the time_point (int) where the fold change peaked, and a confidence string ('Low', 'Medium', 'High')."
    )

    try:
        # The format parameter forces Ollama to output valid JSON matching the Pydantic schema
        response = client.chat(
            model='qwen2.5-coder:7b',
            messages=[{'role': 'user', 'content': prompt}],
            format=Summ.model_json_schema()
        )
        
        # Validate the AI's string response back through Pydantic to ensure no hallucinations
        raw_output = response['message']['content']
        validated_summary = Summ.model_validate_json(raw_output)
        
        return validated_summary.model_dump()
        
    except ValidationError as e:
        print("Schema Validation Failed. The model hallucinated the format.")
        print(e)
        return {}


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



