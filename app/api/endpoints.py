import requests
from flask import request, Response, jsonify
import uuid, json
from werkzeug.utils import secure_filename
import os
from app.services.grn import fetch_and_rank_grns, build_prompt
from app.services.ollama import get_embedding, ask_ollama, get_models, stream_ollama, forward_to_ollama
from app.services.whisper import whisper_service
from app.api.vision_to_json_utils import vision_to_json_endpoint

sessions = {}

# Allowed audio file extensions
ALLOWED_AUDIO_EXTENSIONS = {'mp3', 'wav', 'flac', 'm4a', 'mp4', 'avi', 'mov', 'wmv', 'ogg', 'webm'}

def allowed_audio_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_AUDIO_EXTENSIONS

def register_routes(app):
    @app.route("/transcribe", methods=["POST"])
    def transcribe_audio():
        """
        Transcribe audio file using Whisper or other supported models
        Accepts: multipart/form-data with audio file
        Optional parameters: language, task, word_timestamps, vad_filter, model
        """
        SUPPORTED_MODELS = [
            "whisper",
            "canary",
            "parakeet"
        ]
        try:
            # Check if audio file is provided
            if 'audio' not in request.files:
                return jsonify({"error": "No audio file provided"}), 400
            audio_file = request.files['audio']
            if audio_file.filename == '':
                return jsonify({"error": "No audio file selected"}), 400
            if not allowed_audio_file(audio_file.filename):
                return jsonify({
                    "error": "Unsupported audio format",
                    "supported_formats": list(ALLOWED_AUDIO_EXTENSIONS)
                }), 400
            # Get optional parameters
            language = request.form.get('language')  # e.g., 'en', 'es', 'fr'
            task = request.form.get('task', 'transcribe')  # 'transcribe' or 'translate'
            word_timestamps = request.form.get('word_timestamps', 'false').lower() == 'true'
            vad_filter = request.form.get('vad_filter', 'true').lower() == 'true'
            model = request.form.get('model', 'whisper')
            # Validate task parameter
            if task not in ['transcribe', 'translate']:
                return jsonify({"error": "Task must be 'transcribe' or 'translate'"}), 400
            # Validate model parameter (simple names: 'canary' and 'parakeet')
            if model not in SUPPORTED_MODELS:
                return jsonify({
                    "error": "Unsupported model",
                    "supported_models": SUPPORTED_MODELS,
                    "note": "Default is 'whisper'. Use 'canary' or 'parakeet' to select NVIDIA models."
                }), 400
            # Perform transcription
            result = whisper_service.transcribe_audio(
                audio_file,
                language=language,
                task=task,
                word_timestamps=word_timestamps,
                vad_filter=vad_filter,
                model=model
            )
            return jsonify({
                "success": True,
                "transcription": result,
                "model": model
            })
        except Exception as e:
            return jsonify({
                "error": "Transcription failed",
                "details": str(e)
            }), 500
    
    @app.route("/transcribe/languages", methods=["GET"])
    def get_supported_languages():
        """Get list of supported languages for transcription"""
        try:
            languages = whisper_service.get_supported_languages()
            return jsonify({
                "success": True,
                "supported_languages": languages,
                "note": "Use language codes (e.g., 'en' for English, 'es' for Spanish)"
            })
        except Exception as e:
            return jsonify({
                "error": "Failed to get supported languages",
                "details": str(e)
            }), 500
    
    @app.route("/transcribe/health", methods=["GET"])
    def whisper_health_check():
        """Check if Whisper service is working"""
        try:
            # Get detailed model information
            model_info = whisper_service.get_model_info()
            
            if whisper_service.model is None:
                return jsonify({
                    "status": "unhealthy",
                    "error": "Whisper model not initialized",
                    "model_info": model_info
                }), 503
            
            # Determine overall health status
            if model_info.get("status") == "initialized":
                status = "healthy"
                http_code = 200
            else:
                status = "degraded"
                http_code = 200  # Still functional but not optimal
            
            return jsonify({
                "status": status,
                "model_info": model_info,
                "supported_languages_count": len(whisper_service.get_supported_languages())
            }), http_code
            
        except Exception as e:
            return jsonify({
                "status": "unhealthy",
                "error": str(e),
                "model_loaded": whisper_service.model is not None
            }), 503
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
    
    # Vision to JSON endpoint
    @app.route('/vision-to-json', methods=['POST'])
    def vision_to_json():
        return vision_to_json_endpoint()

    @app.route("/rag_query", methods=["POST"])
    def rag_query():
        try:
            user_question = request.json.get("question", "")
            if not user_question:
                return jsonify({"error": "Question is required"}), 400

            model = request.json.get("model", "mistral")
            top_k = request.json.get("top_k", 3)
            if not isinstance(top_k, int) or top_k <= 0:
                return jsonify({"error": "top_k must be a positive integer"}), 400

            try:
                query_embedding = get_embedding(user_question, model)
            except Exception as e:
                return jsonify({"error": "Failed to generate embedding", "details": str(e)}), 500

            try:
                top_docs = fetch_and_rank_grns(query_embedding, top_k=top_k)
            except Exception as e:
                return jsonify({"error": "Failed to fetch and rank GRNs", "details": str(e)}), 500

            try:
                prompt = build_prompt(top_docs, user_question)
            except Exception as e:
                return jsonify({"error": "Failed to build prompt", "details": str(e)}), 500

            try:
                answer = ask_ollama(prompt, model)
            except Exception as e:
                return jsonify({"error": "Failed to generate answer", "details": str(e)}), 500

            return jsonify({
                "answer": answer,
                "top_docs": [
                    {"summary": doc["summary"], "score": doc["score"]} for doc in top_docs
                ]
            })
        except Exception as e:
            return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500

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