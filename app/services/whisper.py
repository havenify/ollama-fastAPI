import os
import tempfile
from faster_whisper import WhisperModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WhisperService:
    def __init__(self):
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the Whisper model with error handling"""
        try:
            logger.info("Initializing Whisper model (large-v3)...")
            self.model = WhisperModel("large-v3", device="cuda", compute_type="float16")
            logger.info("Whisper model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Whisper model with CUDA: {e}")
            logger.info("Falling back to CPU...")
            try:
                self.model = WhisperModel("large-v3", device="cpu", compute_type="int8")
                logger.info("Whisper model initialized successfully on CPU")
            except Exception as cpu_error:
                logger.error(f"Failed to initialize Whisper model on CPU: {cpu_error}")
                raise RuntimeError(f"Could not initialize Whisper model: {cpu_error}")
    
    def transcribe_audio(self, audio_file, language=None, task="transcribe", word_timestamps=False, vad_filter=True):
        """
        Transcribe audio file using Faster Whisper
        
        Args:
            audio_file: Audio file path or file-like object
            language: Language code (e.g., 'en', 'es', 'fr') - auto-detect if None
            task: 'transcribe' or 'translate'
            word_timestamps: Whether to include word-level timestamps
            vad_filter: Whether to use voice activity detection
            
        Returns:
            dict: Transcription results with text, segments, and metadata
        """
        if not self.model:
            raise RuntimeError("Whisper model not initialized")
        
        try:
            # Create temporary file if needed
            temp_file_path = None
            if hasattr(audio_file, 'read'):
                # It's a file-like object, save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as temp_file:
                    temp_file.write(audio_file.read())
                    temp_file_path = temp_file.name
                audio_path = temp_file_path
            else:
                # It's a file path
                audio_path = audio_file
            
            logger.info(f"Starting transcription of audio file: {audio_path}")
            
            # Perform transcription
            segments, info = self.model.transcribe(
                audio_path,
                language=language,
                task=task,
                word_timestamps=word_timestamps,
                vad_filter=vad_filter,
                beam_size=5,
                best_of=5,
                temperature=0.0
            )
            
            # Process segments
            transcription_segments = []
            full_text = ""
            
            for segment in segments:
                segment_data = {
                    "id": segment.id,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "avg_logprob": segment.avg_logprob,
                    "no_speech_prob": segment.no_speech_prob
                }
                
                if word_timestamps and hasattr(segment, 'words'):
                    segment_data["words"] = [
                        {
                            "word": word.word,
                            "start": word.start,
                            "end": word.end,
                            "probability": word.probability
                        }
                        for word in segment.words
                    ]
                
                transcription_segments.append(segment_data)
                full_text += segment.text.strip() + " "
            
            result = {
                "text": full_text.strip(),
                "segments": transcription_segments,
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration,
                "duration_after_vad": info.duration_after_vad if hasattr(info, 'duration_after_vad') else None,
                "all_language_probs": info.all_language_probs if hasattr(info, 'all_language_probs') else None
            }
            
            logger.info(f"Transcription completed. Language: {info.language}, Duration: {info.duration:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            raise RuntimeError(f"Transcription failed: {str(e)}")
        
        finally:
            # Clean up temporary file if created
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary file {temp_file_path}: {e}")
    
    def get_supported_languages(self):
        """Get list of supported languages"""
        return [
            "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su"
        ]

# Global instance
whisper_service = WhisperService()
