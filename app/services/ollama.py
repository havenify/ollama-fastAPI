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

def get_models():
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags")
        response.raise_for_status()
        return response.json().get("models", [])
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to fetch models: {e}")