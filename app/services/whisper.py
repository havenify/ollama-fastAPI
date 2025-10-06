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
        self.model_type = None
        self.current_model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the Whisper model with error handling"""
        # First, check CUDA availability and driver compatibility
        cuda_available = self._check_cuda_compatibility()
        
        if cuda_available:
            try:
                logger.info("Initializing Whisper model (large-v3) with CUDA...")
                self.model = WhisperModel("large-v3", device="cuda", compute_type="float16")
                logger.info("Whisper model initialized successfully with CUDA")
                return
            except Exception as e:
                logger.error(f"Failed to initialize Whisper model with CUDA: {e}")
                if "CUDA driver version is insufficient" in str(e):
                    logger.error("CUDA driver version is too old for the required CUDA runtime")
                    logger.info("Please update your NVIDIA drivers or use CPU mode")
        
        # Fallback to CPU
        logger.info("Falling back to CPU mode...")
        try:
            # Try with large-v3 first
            logger.info("Attempting large-v3 model on CPU...")
            self.model = WhisperModel("large-v3", device="cpu", compute_type="int8")
            logger.info("Whisper large-v3 model initialized successfully on CPU")
        except Exception as large_error:
            logger.warning(f"Large-v3 model failed on CPU: {large_error}")
            logger.info("Trying with base model as fallback...")
            try:
                # Fallback to smaller model if large-v3 fails on CPU
                self.model = WhisperModel("base", device="cpu", compute_type="int8")
                logger.info("Whisper base model initialized successfully on CPU")
            except Exception as base_error:
                logger.error(f"All model initialization attempts failed: {base_error}")
                raise RuntimeError(f"Could not initialize any Whisper model: {base_error}")
    
    def _check_cuda_compatibility(self):
        """Check if CUDA is available and compatible"""
        try:
            import torch
            if not torch.cuda.is_available():
                logger.info("CUDA not available on this system")
                return False
            
            # Check CUDA device count
            device_count = torch.cuda.device_count()
            if device_count == 0:
                logger.info("No CUDA devices found")
                return False
            
            # Get CUDA version info
            cuda_version = torch.version.cuda
            driver_version = torch.cuda.get_device_properties(0).major
            
            logger.info(f"CUDA runtime version: {cuda_version}")
            logger.info(f"GPU compute capability: {driver_version}")
            logger.info(f"Available CUDA devices: {device_count}")
            
            # Try a simple CUDA operation to verify compatibility
            test_tensor = torch.tensor([1.0]).cuda()
            _ = test_tensor * 2  # Simple operation
            
            logger.info("CUDA compatibility check passed")
            return True
            
        except Exception as e:
            logger.warning(f"CUDA compatibility check failed: {e}")
            if "CUDA driver version is insufficient" in str(e):
                logger.error("CUDA driver version is insufficient for CUDA runtime version")
                logger.error("Please update your NVIDIA drivers to a version compatible with CUDA runtime")
            return False
    
    def transcribe_audio(self, audio_file, language=None, task="transcribe", word_timestamps=False, vad_filter=True, model="whisper"):
        """
        Transcribe audio file using Faster Whisper
        
        Args:
            audio_file: Audio file path or file-like object
            language: Language code (e.g., 'en', 'es', 'fr') - auto-detect if None
            task: 'transcribe' or 'translate'
            word_timestamps: Whether to include word-level timestamps
            vad_filter: Whether to use voice activity detection
            model: Model name ("whisper", "nvidia/canary-qwen-2.5b", "nvidia/parakeet-tdt-0.6b-v2")
            
        Returns:
            dict: Transcription results with text, segments, and metadata
        """
        # Model selection logic
        # Map simple model names to actual model identifiers
        model_map = {
            "whisper": "large-v3",
            "canary": "nvidia/canary-qwen-2.5b",
            "parakeet": "nvidia/parakeet-tdt-0.6b-v2"
        }
        model_name = model_map.get(model, "large-v3")
        # If requested model is not loaded or it's a different model, load it
        if not self.model or self.current_model != model_name:
            try:
                logger.info(f"Loading model: {model_name}")
                cuda_available = self._check_cuda_compatibility()
                device = "cuda" if cuda_available else "cpu"
                
                if "parakeet" in model_name.lower():
                    try:
                        # First, try to verify NeMo installation
                        import pkg_resources
                        nemo_version = pkg_resources.get_distribution('nemo_toolkit').version
                        logger.info(f"Found NeMo toolkit version: {nemo_version}")
                        
                        # Import NeMo ASR components with error handling
                        try:
                            import nemo
                            import nemo.collections.tts as nemo_tts
                            from nemo.collections.tts.models import FastPitchModel
                            logger.info("Successfully imported NeMo TTS components")
                        except ImportError as e:
                            logger.error(f"Failed to import NeMo components: {e}")
                            raise ImportError("Please ensure NeMo toolkit is properly installed")
                        
                        # Ensure CUDA is available
                        import torch
                        if not torch.cuda.is_available():
                            logger.warning("CUDA is not available. Model may run slower on CPU.")
                        
                        # Load model from local path or download
                        try:
                            self.model = FastPitchModel.restore_from("/app/models/parakeet-tdt-0.6b-v2.nemo")
                        except FileNotFoundError:
                            logger.info("Model not found locally, downloading from HuggingFace")
                            from huggingface_hub import hf_hub_download
                            
                            # Download the model file
                            model_path = hf_hub_download(
                                repo_id="nvidia/parakeet-tdt-0.6b-v2",
                                filename="parakeet-tdt-0.6b-v2.nemo",
                                cache_dir="/app/models"
                            )
                            self.model = FastPitchModel.restore_from(model_path)
                        
                        if cuda_available:
                            self.model = self.model.cuda()
                        self.model_type = "parakeet"
                        logger.info("Parakeet model loaded successfully")
                    except Exception as e:
                        logger.error(f"Failed to load model {model_name}: {str(e)}")
                        raise
                    except ImportError as e:
                        logger.error(f"NeMo import error: {e}")
                        logger.error("Please ensure NeMo toolkit is installed: pip install 'nemo_toolkit[asr]>=1.20.0'")
                        raise
                    except Exception as e:
                        logger.error(f"Error loading Parakeet model: {e}")
                        logger.error("This could be due to CUDA issues or insufficient memory")
                        raise
                        
                else:
                    # Default to Whisper model
                    compute_type = "float16" if cuda_available else "int8"
                    self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
                    self.model_type = "whisper"
                
                self.current_model = model_name
                logger.info(f"Model {model_name} loaded successfully on {device}")
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                raise RuntimeError(f"Could not load model {model_name}: {e}")
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
            logger.info(f"Starting transcription of audio file: {audio_path} using model: {model_name}")
            # Process audio based on model type
            if self.model_type == "whisper":
                segments, info = self.model.transcribe(
                    audio_path,
                    language=language,
                    task=task,
                    word_timestamps=word_timestamps,
                    vad_filter=vad_filter,
                    beam_size=1,
                    best_of=1,
                    temperature=0.0,
                    repetition_penalty=1.1,
                    condition_on_previous_text=False
                )
                
                # Process Whisper segments
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
                    "all_language_probs": info.all_language_probs if hasattr(info, 'all_language_probs') else None,
                    "model_type": "whisper"
                }
                
            elif self.model_type == "parakeet":
                import torch
                import torchaudio
                import librosa
                
                # Load and process audio
                waveform, sample_rate = torchaudio.load(audio_path)
                
                # Resample if needed (Parakeet expects 16kHz)
                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                    waveform = resampler(waveform)
                
                # Convert to mono if stereo
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                # Get audio duration
                duration = waveform.shape[1] / 16000
                
                # Process with Parakeet model
                with torch.no_grad():
                    logprobs = self.model.forward(input_signal=waveform.cuda() if torch.cuda.is_available() else waveform,
                                                input_signal_length=torch.tensor([waveform.shape[1]]))
                    
                    # Convert logprobs to text using CTC decoding
                    transcription = self.model.decoding.ctc_decoder_predictions_tensor(logprobs)[0]
                
                result = {
                    "text": transcription,
                    "segments": [{"text": transcription, "start": 0, "end": duration}],
                    "duration": duration,
                    "model_type": "parakeet"
                }
            
            logger.info(f"Transcription completed. Duration: {result['duration']:.2f}s, Model: {self.model_type}")
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
    
    def get_model_info(self):
        """Get information about the currently loaded model"""
        if not self.model:
            return {"status": "not_initialized", "error": "Model not initialized"}
        
        try:
            import torch
            
            model_info = {
                "status": "initialized",
                "model_size": getattr(self.model, 'model_size_or_path', 'unknown'),
                "device": getattr(self.model, 'device', 'unknown'),
                "compute_type": getattr(self.model, 'compute_type', 'unknown')
            }
            
            # Add CUDA info if available
            if torch.cuda.is_available():
                model_info["cuda_available"] = True
                model_info["cuda_device_count"] = torch.cuda.device_count()
                model_info["cuda_version"] = torch.version.cuda
                
                if torch.cuda.device_count() > 0:
                    props = torch.cuda.get_device_properties(0)
                    model_info["gpu_name"] = props.name
                    model_info["gpu_memory"] = f"{props.total_memory / 1024**3:.1f} GB"
            else:
                model_info["cuda_available"] = False
                model_info["reason"] = "CUDA not available or incompatible drivers"
            
            return model_info
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "model_loaded": self.model is not None
            }

# Global instance
whisper_service = WhisperService()
