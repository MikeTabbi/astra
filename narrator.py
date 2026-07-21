from ollama import Client

# Zion pastes your ngrok URL here:
client = Client(host='https://a1b2-c3d4-x5y6.ngrok-free.app')

# Test call to your MacBook's Ollama instance
response = client.chat(
    model='llama3.2:latest',
    messages=[{'role': 'user', 'content': 'Checking connection to host MacBook...'}]
)

print(response['message']['content'])