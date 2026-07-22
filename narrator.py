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



