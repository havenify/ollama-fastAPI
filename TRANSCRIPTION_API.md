# Audio Transcription API Documentation

This API provides audio transcription capabilities using OpenAI's Whisper large-v3 model with GPU acceleration (CUDA) and automatic CPU fallback.

## Prerequisites

### System Requirements
- CUDA-compatible GPU (recommended) or CPU
- Python 3.8+
- FFmpeg (for audio processing)

### Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure CUDA is available (optional, will fallback to CPU):
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## API Endpoints

### 1. Transcribe Audio
**POST** `/transcribe`

Transcribes audio files using the Whisper large-v3 model.

#### Request
- **Content-Type**: `multipart/form-data`
- **Body Parameters**:
  - `audio` (file, required): Audio file to transcribe
  - `language` (string, optional): Language code (e.g., 'en', 'es', 'fr'). Auto-detects if not provided.
  - `task` (string, optional): Either 'transcribe' (default) or 'translate'
  - `word_timestamps` (boolean, optional): Include word-level timestamps (default: false)
  - `vad_filter` (boolean, optional): Use voice activity detection (default: true)

#### Supported Audio Formats
- MP3, WAV, FLAC, M4A, MP4, AVI, MOV, WMV, OGG, WebM

#### Response
```json
{
  "success": true,
  "transcription": {
    "text": "Complete transcribed text...",
    "segments": [
      {
        "id": 0,
        "start": 0.0,
        "end": 5.2,
        "text": "First segment text",
        "avg_logprob": -0.3,
        "no_speech_prob": 0.1,
        "words": [  // Only if word_timestamps=true
          {
            "word": "First",
            "start": 0.0,
            "end": 0.5,
            "probability": 0.99
          }
        ]
      }
    ],
    "language": "en",
    "language_probability": 0.95,
    "duration": 120.5,
    "duration_after_vad": 118.2
  }
}
```

#### Example Usage

**cURL:**
```bash
curl -X POST http://localhost:8288/transcribe \
  -F "audio=@audio.mp3" \
  -F "language=en" \
  -F "task=transcribe" \
  -F "word_timestamps=true"
```

**Python:**
```python
import requests

with open('audio.mp3', 'rb') as f:
    files = {'audio': f}
    data = {
        'language': 'en',
        'task': 'transcribe',
        'word_timestamps': 'true'
    }
    response = requests.post('http://localhost:8288/transcribe', files=files, data=data)
    result = response.json()
    print(result['transcription']['text'])
```

**JavaScript (Browser):**
```javascript
const formData = new FormData();
formData.append('audio', audioFile);
formData.append('language', 'en');
formData.append('task', 'transcribe');

fetch('http://localhost:8288/transcribe', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => console.log(data.transcription.text));
```

### 2. Get Supported Languages
**GET** `/transcribe/languages`

Returns a list of all supported language codes.

#### Response
```json
{
  "success": true,
  "supported_languages": ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh", ...],
  "note": "Use language codes (e.g., 'en' for English, 'es' for Spanish)"
}
```

### 3. Health Check
**GET** `/transcribe/health`

Checks if the Whisper service is initialized and ready.

#### Response
```json
{
  "status": "healthy",
  "model": "large-v3",
  "device": "cuda"
}
```

## Language Codes

The API supports 99 languages. Common language codes include:

| Language | Code | Language | Code |
|----------|------|----------|------|
| English | en | Spanish | es |
| French | fr | German | de |
| Italian | it | Portuguese | pt |
| Russian | ru | Japanese | ja |
| Korean | ko | Chinese | zh |
| Arabic | ar | Hindi | hi |
| Dutch | nl | Swedish | sv |

For a complete list, call the `/transcribe/languages` endpoint.

## Features

### GPU Acceleration
- Automatically uses CUDA if available
- Falls back to CPU with optimized settings
- Uses float16 precision on GPU for faster processing

### Voice Activity Detection (VAD)
- Filters out non-speech segments
- Reduces processing time and improves accuracy
- Can be disabled with `vad_filter=false`

### Word-Level Timestamps
- Provides precise timing for each word
- Useful for subtitle generation and audio alignment
- Enabled with `word_timestamps=true`

### Translation
- Can translate any language to English
- Set `task=translate` instead of `transcribe`

## Error Handling

The API returns appropriate HTTP status codes and error messages:

- **400 Bad Request**: Invalid parameters or missing audio file
- **500 Internal Server Error**: Transcription failed or service unavailable
- **503 Service Unavailable**: Whisper model not initialized

Example error response:
```json
{
  "error": "Transcription failed",
  "details": "CUDA out of memory. Consider using CPU mode."
}
```

## Performance Notes

### File Size Limits
- Maximum upload size: 500MB
- Larger files may require more processing time

### Processing Time
- GPU: ~0.1x real-time (10-minute audio in ~1 minute)
- CPU: ~0.5-1x real-time (slower but still efficient)

### Memory Usage
- GPU: ~2-4GB VRAM for large-v3 model
- CPU: ~4-8GB RAM depending on audio length

## Testing

### Web Interface
Open `transcription_client.html` in your browser for a user-friendly testing interface.

### Command Line Testing
```bash
python test_transcription.py path/to/audio.mp3
```

### Integration Testing
```python
# Test basic transcription
response = requests.post('http://localhost:8288/transcribe', 
                        files={'audio': open('test.mp3', 'rb')})

# Test with specific language
response = requests.post('http://localhost:8288/transcribe', 
                        files={'audio': open('spanish.mp3', 'rb')},
                        data={'language': 'es'})

# Test translation
response = requests.post('http://localhost:8288/transcribe', 
                        files={'audio': open('french.mp3', 'rb')},
                        data={'task': 'translate'})
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce audio file size or length
   - The service will automatically fall back to CPU

2. **Slow Performance**
   - Ensure CUDA is available and working
   - Check GPU memory usage
   - Consider using shorter audio clips for testing

3. **Unsupported Audio Format**
   - Convert audio to supported format (MP3, WAV, etc.)
   - Use FFmpeg: `ffmpeg -i input.xyz -f mp3 output.mp3`

4. **Service Unavailable**
   - Check if the model downloaded successfully
   - Verify system has sufficient memory/VRAM
   - Check server logs for detailed error messages

### Logs and Debugging
The service logs initialization and processing status. Check console output for:
- Model loading status
- Device selection (CUDA/CPU)
- Processing errors and warnings
