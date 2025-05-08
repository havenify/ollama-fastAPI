import requests

OLLAMA_HOST = "http://host.docker.internal:11434"

def get_embedding(text, model="mistral"):
    resp = requests.post(f"{OLLAMA_HOST}/api/embeddings", json={
        "model": model,
        "prompt": text
    })
    resp.raise_for_status()
    return resp.json()["embedding"]

def ask_ollama(prompt, model="mistral"):
    resp = requests.post(f"{OLLAMA_HOST}/api/generate", json={
        "model": model,
        "prompt": prompt,
        "stream": False
    })
    resp.raise_for_status()
    return resp.json()["response"]