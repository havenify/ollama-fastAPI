from flask import Flask, request, Response, jsonify
from flask_cors import CORS
import requests
import uuid

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
Format all your responses in GitHub-flavoured **Markdown**.
Use:
- Use headings like ##, ###, etc. for sections
- Bold text for important points
- Bullet points with `-` or numbered lists with `1.` for lists
- Tables where comparisons or structures are needed
  Use a header row, seperator row (`---`) and separater columns with `|`.
- Single backticks for inline code
- Use emojis to enhance the message (e.g., 😊, 🚀)
- Use links for references (e.g., [OpenAI](https://openai.com))
- Use images where necessary (e.g., ![alt text](image_url))
- Use blockquotes for quotes or important notes
- Triple backticks for code blocks
Keep responses clear and readable.
"""
    full_prompt = f"{system_instruction}{chat_prompt}\nUser: {prompt}\nAI:"
    print(f"Full Prompt: {full_prompt}")
    app.logger.info(f"Session ID: {session_id}")
    app.logger.info(f"Model: {model}")
    app.logger.info(f"Full Prompt: {full_prompt}")

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
                        token = line.decode("utf-8").replace("data: ", "")
                        app.logger.debug(f"Received line: {token}")
                        if token.strip() == "[DONE]":
                            app.logger.info("Stream completed.")
                            yield f"data: {{\"status\": \"done\", \"response\": \"{collected}\"}}\n\n"
                            break
                        try:
                            # Replace 'false' with 'False' and 'true' with 'True' for Python compatibility
                            token = token.replace("false", "False").replace("true", "True")
                            token_data = eval(token)
                            if "response" in token_data:
                                word = token_data["response"]
                                collected += word
                                yield f"data: {{\"status\": \"streaming\", \"response\": \"{word}\", \"markdown\": \"{word}\"}}\n\n"
                            else:
                                app.logger.error(f"'response' key not found in token data: {token_data}")
                                yield f"data: {{\"status\": \"error\", \"message\": \"'response' key not found in token data.\"}}\n\n"
                                break
                        except Exception as e:
                            error_message = f"Error processing token: {e}"
                            app.logger.error(error_message)
                            yield f"data: {{\"status\": \"error\", \"message\": \"{error_message}\"}}\n\n"
                            break
        except Exception as e:
            error_message = f"Error in stream request: {e}"
            app.logger.error(error_message)
            yield f"data: {{\"status\": \"error\", \"message\": \"{error_message}\"}}\n\n"
        sessions[session_id].append({"user": prompt, "bot": collected})

    return Response(generate(), content_type="application/json")

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
