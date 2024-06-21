import requests

prompt = "why is the sky blue?"
model = "gemma:2b" # or "gemma:2b"
payload = {
  "model": model,
  "stream": False,
  "messages": [
    {
      "role": "user",
      "content": prompt
    }
  ]
}
response = requests.post("http://localhost:11434/api/chat", json=payload)
llm_response = response.json()['message']['content']
print(f"{prompt} \n LLM Response: {llm_response}")