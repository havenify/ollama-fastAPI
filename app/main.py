from flask import Flask
from flask_cors import CORS
from app.api.endpoints import register_routes

app = Flask(__name__)

# Configure maximum file upload size (500MB for large audio files)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

CORS(app, origins=[
    "*",
    "http://localhost",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "https://chat.techpranee.com"
], supports_credentials=True)

register_routes(app)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8288)