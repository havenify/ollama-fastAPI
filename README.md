# Ollama FastAPI with Audio Transcription

A comprehensive API service that provides:
- **LLM Chat & Streaming**: Interact with Ollama models
- **Embeddings**: Generate text embeddings
- **RAG Queries**: Retrieval-augmented generation
- **Audio Transcription**: High-quality speech-to-text using Whisper large-v3

## Features

### 🤖 LLM Integration
- Chat with various Ollama models (Mistral, Llama, etc.)
- Real-time streaming responses
- Session management
- Model switching

### 🎵 Audio Transcription (NEW!)
- **Whisper large-v3** model with GPU acceleration
- Support for 99+ languages
- Word-level timestamps
- Voice activity detection
- Translation to English
- Multiple audio formats (MP3, WAV, FLAC, M4A, MP4, etc.)

### 📊 Advanced Features
- Text embeddings generation
- RAG (Retrieval-Augmented Generation) queries
- MongoDB integration for document storage
- CORS enabled for web applications

## Quick Start

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd ollama-fastAPI

# Install dependencies
pip install -r requirements.txt
```

### Docker Deployment
```bash
# Build the image
docker build -t llm-api .

# Delete old container (if exists)
docker rm -f llm-api

# Deploy with new image
docker run -d -p 8288:8288 --name llm-api llm-api
```

### Local Development
```bash
# Run the Flask application
python -m app.main

# Or with flask command
flask --app app.main run --host=0.0.0.0 --port=8288
```

## API Endpoints

### Chat Endpoints
- `POST /chat` - Single response chat
- `POST /stream` - Streaming chat responses
- `POST /embed` - Generate text embeddings
- `POST /rag_query` - RAG-based queries
- `GET /models` - List available models

### Audio Transcription Endpoints
- `POST /transcribe` - Transcribe audio files
- `GET /transcribe/languages` - Get supported languages
- `GET /transcribe/health` - Health check for Whisper service

### Utility Endpoints
- `POST /api/<endpoint>` - Proxy to Ollama API
- `GET /api/status` - Service status

## Audio Transcription Usage

### Basic Transcription
```bash
curl -X POST http://localhost:8288/transcribe \
  -F "audio=@audio.mp3"
```

### With Language Detection
```bash
curl -X POST http://localhost:8288/transcribe \
  -F "audio=@audio.mp3" \
  -F "language=en" \
  -F "word_timestamps=true"
```

### Translation to English
```bash
curl -X POST http://localhost:8288/transcribe \
  -F "audio=@spanish_audio.mp3" \
  -F "task=translate"
```

## Testing

### Web Interface
Open `transcription_client.html` in your browser for interactive testing.

### Command Line Testing
```bash
python test_transcription.py path/to/audio.mp3
```

## Documentation

- **[Complete Transcription API Documentation](TRANSCRIPTION_API.md)** - Detailed API reference
- **[Thunder Collection](thunder-collection_ollama.json)** - API testing collection

## Requirements

### System Requirements
- Python 3.8+
- CUDA-compatible GPU (recommended for transcription)
- FFmpeg (for audio processing)

### Dependencies
See `requirements.txt` for complete list including:
- Flask & Flask-CORS
- faster-whisper
- torch & torchaudio
- requests, pymongo, numpy

## Configuration

### Environment Variables
- `OLLAMA_HOST`: Ollama server URL (default: http://host.docker.internal:11434)
- `MAX_CONTENT_LENGTH`: Max upload size (default: 500MB)

### Audio Transcription Settings
- **Model**: Whisper large-v3
- **Device**: Auto-detects CUDA, falls back to CPU
- **Precision**: float16 (GPU) / int8 (CPU)
- **Supported Formats**: MP3, WAV, FLAC, M4A, MP4, AVI, MOV, WMV, OGG, WebM

## Architecture

```
app/
├── main.py              # Flask application entry point
├── api/
│   └── endpoints.py     # API route definitions
├── services/
│   ├── ollama.py       # Ollama integration
│   ├── whisper.py      # Audio transcription service
│   ├── grn.py          # RAG functionality
│   └── similarity.py   # Similarity calculations
└── utils/
    └── similarity.py    # Utility functions
```

## Performance

### Audio Transcription Performance
- **GPU (CUDA)**: ~0.1x real-time processing
- **CPU**: ~0.5-1x real-time processing
- **Memory**: 2-4GB VRAM (GPU) / 4-8GB RAM (CPU)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open-source. Please check the license file for details.