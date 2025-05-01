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
Format all your responses in GitHub-flavored **Markdown**.
Use:
- Headings (##, ###)
- Bold text
- Bullet points and numbered lists
- Tables using pipes (`|`) and separator rows
- Inline code with backticks, code blocks with triple backticks
- Links, images, emojis, and blockquotes
Keep it readable, elegant, and informative.
"""

    full_prompt = f"{system_instruction}\n{chat_prompt}\nUser: {prompt}\nAI:"
    app.logger.info(f"Session ID: {session_id}")
    app.logger.info(f"Model: {model}")
    app.logger.info(f"Prompt: {full_prompt}")

    def generate():
        collected = ""
        try:
            with requests.post(f"{OLLAMA_HOST}/api/generate", json={
                "model": model,
                "prompt": full_prompt,
                "stream": True
            }, stream=True) as r:
                app.logger.info("Stream request sent successfully.")
                for line in r.iter_lines():
                    if line:
                        line_decoded = line.decode("utf-8").replace("data: ", "")
                        if line_decoded.strip() == "[DONE]":
                            yield f"data: {json.dumps({'status': 'done', 'response': collected})}\n\n"
                            break
                        try:
                            # Avoid using eval; use json.loads after sanitizing
                            token_data = json.loads(line_decoded)
                            if "response" in token_data:
                                word = token_data["response"]
                                collected += word
                                yield f"data: {json.dumps({'status': 'streaming', 'response': word})}\n\n"
                            else:
                                error = "'response' key missing"
                                yield f"data: {json.dumps({'status': 'error', 'message': error})}\n\n"
                                break
                        except Exception as e:
                            error = f"Error parsing chunk: {e}"
                            yield f"data: {json.dumps({'status': 'error', 'message': error})}\n\n"
                            break
        except Exception as e:
            error = f"Stream connection error: {e}"
            yield f"data: {json.dumps({'status': 'error', 'message': error})}\n\n"
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
