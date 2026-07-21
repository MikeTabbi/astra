from ollama import Client

# Zion pastes your ngrok URL here:
client = Client()

# Test call to your MacBook's Ollama instance
response = client.chat(
    model='llama3.2:latest',
    messages=[{'role': 'user', 'content': 'Checking connection to host MacBook...'}]
)

print(response['message']['content'])