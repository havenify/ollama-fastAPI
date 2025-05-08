import requests
from flask import request, Response, jsonify
import uuid, json
from app.services.grn import fetch_and_rank_grns, build_prompt
from app.services.ollama import get_embedding, ask_ollama, get_models, stream_ollama, forward_to_ollama

sessions = {}

def register_routes(app):
    @app.route("/chat", methods=["POST"])
    def chat():
        prompt = request.json.get("prompt", "")
        session_id = request.json.get("session_id") or str(uuid.uuid4())
        model = request.json.get("model", "mistral")
        if session_id not in sessions:
            sessions[session_id] = []
        history = sessions[session_id]
        chat_prompt = "\n".join([f"User: {h['user']}\nAI: {h['bot']}" for h in history])
        full_prompt = f"{chat_prompt}\nUser: {prompt}\nAI:"
        response = ask_ollama(full_prompt, model)
        sessions[session_id].append({"user": prompt, "bot": response})
        return jsonify({"response": response, "session_id": session_id})

    @app.route("/embed", methods=["POST"])
    def embed():
        text = request.json.get("text", "")
        model = request.json.get("model", "mistral")
        if not text:
            return jsonify({"error": "Text is required"}), 400
        embedding = get_embedding(text, model)
        return jsonify({
            "embedding": embedding,
            "model": model,
            "dimension": len(embedding)
        })

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
        generate = stream_ollama(full_prompt, model)
        return Response(generate(), content_type="text/event-stream")

    @app.route("/rag_query", methods=["POST"])
    def rag_query():
        user_question = request.json.get("question", "")
        model = request.json.get("model", "mistral")
        top_k = int(request.json.get("top_k", 3))
        if not user_question:
            return jsonify({"error": "Question is required"}), 400
        query_embedding = get_embedding(user_question, model)
        top_docs = fetch_and_rank_grns(query_embedding, top_k=top_k)
        prompt = build_prompt(top_docs, user_question)
        answer = ask_ollama(prompt, model)
        return jsonify({
            "answer": answer,
            "top_docs": [
                {"summary": doc["summary"], "score": doc["score"]} for doc in top_docs
            ]
        })
    
    @app.route("/models", methods=["GET"])
    def models_endpoint():
        try:
            models = get_models()
            return jsonify({"models": models})
        except Exception as e:
            return jsonify({
                "error": "Failed to fetch models.",
                "details": str(e)
            }), 500

    @app.route('/api/<endpoint>', methods=['POST'])
    def proxy(endpoint):
        data = request.json
        resp, status = forward_to_ollama(endpoint, data, method="POST")
        return jsonify(resp), status

    @app.route('/api/status', methods=['GET'])
    def status():
        resp, status = forward_to_ollama("status", {}, method="GET")
        return jsonify(resp), status