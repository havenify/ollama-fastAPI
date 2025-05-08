import requests
import json

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

def stream_ollama(full_prompt, model, ollama_host=OLLAMA_HOST):
    def generate():
        collected = ""
        try:
            with requests.post(f"{ollama_host}/api/generate", json={
                "model": model,
                "prompt": full_prompt,
                "stream": True
            }, stream=True) as r:
                for line in r.iter_lines():
                    if not line:
                        continue
                    try:
                        token_json = line.decode("utf-8").replace("data: ", "")
                        if token_json.strip() == "[DONE]":
                            yield f"data: {json.dumps({'status': 'done', 'response': collected})}\n\n"
                            break
                        token_data = json.loads(token_json)
                        word = token_data.get("response", "")
                        if word:
                            collected += word
                            yield f"data: {json.dumps({'status': 'streaming', 'response': word})}\n\n"
                    except Exception as e:
                        yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"
                        break
        except Exception as e:
            yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"
    return generate

def get_models():
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags")
        response.raise_for_status()
        return response.json().get("models", [])
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to fetch models: {e}")

def forward_to_ollama(endpoint, data, method="POST", ollama_host=OLLAMA_HOST):
    url = f"{ollama_host}/{endpoint}"
    try:
        if method == "POST":
            response = requests.post(url, json=data)
        elif method == "GET":
            response = requests.get(url, params=data)
        else:
            raise ValueError("Unsupported HTTP method")
        return response.json(), response.status_code
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}, 500