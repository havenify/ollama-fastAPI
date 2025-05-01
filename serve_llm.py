from flask import Flask, request, Response, jsonify
from flask_cors import CORS
import requests
import uuid, json

app = Flask(__name__)
CORS(app)

sessions = {}  # session_id: [{"user":..., "bot":...}, ...]

OLLAMA_HOST = "http://host.docker.internal:11434"  # Host Ollama
# OLLAMA_HOST = "http://localhost:11434"  # Local Ollama
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    prompt = data.get("prompt", "")
    session_id = data.get("session_id") or str(uuid.uuid4())
    model = data.get("model", "mistral")

    if session_id not in sessions:
        sessions[session_id] = []

    history = sessions[session_id]
    chat_prompt = "\n".join([f"User: {h['user']}\nAI: {h['bot']}" for h in history])
    full_prompt = f"{chat_prompt}\nUser: {prompt}\nAI:"

    resp = requests.post(f"{OLLAMA_HOST}/api/generate", json={
        "model": model,
        "prompt": full_prompt,
        "stream": False
    })

    response = resp.json()["response"]
    sessions[session_id].append({"user": prompt, "bot": response})

    return jsonify({"response": response, "session_id": session_id})
def clean_markdown(raw_text):
    # Optional: remove backticks around tables
    if '|' in raw_text and '```' in raw_text:
        lines = raw_text.splitlines()
        new_lines = []
        inside_code = False
        for line in lines:
            if line.strip().startswith('```') and '|' in line:
                inside_code = not inside_code
                continue  # skip the backtick line
            if inside_code:
                new_lines.append(line)
            else:
                new_lines.append(line)
        return '\n'.join(new_lines)
    return raw_text

@app.route("/stream", methods=["POST"])
def stream():
    data = request.json
    prompt = data.get("prompt", "")
    session_id = data.get("session_id") or str(uuid.uuid4())
    model = data.get("model", "mistral")

    if session_id not in sessions:
        sessions[session_id] = []

    history = sessions[session_id]
    chat_prompt = "\n".join([f"User: {h['user']}\nAI: {h['bot']}" for h in history])

    system_instruction = """You are a helpful AI assistant. 
Always respond using clean GitHub-flavored **Markdown**.

- For tables, use:
  - A blank line before and after
  - `|` to separate columns
  - `|---|---|` separator row after header
  - Never wrap tables inside triple backticks
- Use **bold**, `inline code`, bullet lists, and `###` headings
- Do NOT repeat your answer or prefix each line with the same phrase
"""

    full_prompt = f"{system_instruction}\n{chat_prompt}\nUser: {prompt}\nAI:"

    def generate():
        collected = ""
        try:
            with requests.post(f"{OLLAMA_HOST}/api/generate", json={
                "model": model,
                "prompt": full_prompt,
                "stream": True
            }, stream=True) as r:
                buffer = ""
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
        sessions[session_id].append({"user": prompt, "bot": collected})

    return Response(generate(), content_type="text/event-stream")
@app.route("/models", methods=["GET"])
def get_models():
    """API to fetch all available models dynamically from Ollama."""
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags")
        response.raise_for_status()
        models = response.json().get("models", [])
        return jsonify({"models": models})
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error fetching models: {e}")
        return jsonify({"error": "Failed to fetch models."}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8288)
