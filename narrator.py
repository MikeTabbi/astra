from ollama import Client

client = Client()

response = client.chat(
    model='llama3.2:latest',
    messages=[{'role': 'user', 'content': 'Checking connection to local Ollama...'}]
)

# Standard dict access or attribute access
print(response['message']['content'])