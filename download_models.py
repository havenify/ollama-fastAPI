import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_models():
    """Download all required models for transcription service"""
    try:
        # Create models directory
        models_dir = Path("/app/models")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Set environment variables for model caching
        os.environ["TRANSFORMERS_CACHE"] = str(models_dir)
        os.environ["HF_HOME"] = str(models_dir)
        
        # Import required libraries
        import torch
        import json
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        from huggingface_hub import snapshot_download
        from faster_whisper import download_model
        
        # Download Whisper models
        whisper_models = ['large-v3', 'base']
        for model in whisper_models:
            try:
                logger.info(f"Downloading Whisper model: {model}")
                download_model(model)
                logger.info(f"Successfully downloaded Whisper model: {model}")
            except Exception as e:
                logger.error(f"Failed to download Whisper model {model}: {e}")
        
        # Download NVIDIA models
        nvidia_models = [
            'nvidia/canary-qwen-2.5b',
            'nvidia/parakeet-tdt-0.6b-v2'
        ]
        for model in nvidia_models:
            try:
                logger.info(f"Downloading model: {model}")
                # First, download the model files
                local_model_path = snapshot_download(
                    repo_id=model,
                    cache_dir=str(models_dir),
                    revision="main",
                    token=os.getenv("HF_TOKEN")  # In case the model requires authentication
                )
                
                if 'canary-qwen' in model:
                    # For Canary model, just verify the download and create a config
                    model_file = next(Path(local_model_path).glob("*.safetensors"), None)
                    if not model_file:
                        raise FileNotFoundError("No .safetensors file found in the downloaded model")
                    
                    # Create a simple config for the Canary model
                    config = {
                        "model_type": "canary_qwen",
                        "model_path": str(model_file),
                        "model_format": "safetensors",
                        "architecture": "qwen",
                        "torch_dtype": "float16"
                    }
                    
                    config_path = Path(local_model_path) / "config.json"
                    with open(config_path, "w") as f:
                        json.dump(config, f, indent=2)
                
                elif 'parakeet' in model:
                    # For Parakeet, just download the NEMO file
                    # The model will be loaded using NeMo library at runtime
                    nemo_file = next(Path(local_model_path).glob("*.nemo"))
                    if not nemo_file:
                        raise FileNotFoundError("No .nemo file found in the downloaded model")
                    
                    # Create a simple config.json for reference
                    config = {
                        "model_type": "nemo_tts",
                        "model_path": str(nemo_file),
                        "model_format": "nemo"
                    }
                    
                    with open(Path(local_model_path) / "config.json", "w") as f:
                        json.dump(config, f, indent=2)
                
                logger.info(f"Successfully downloaded model: {model}")
            except Exception as e:
                logger.error(f"Failed to download model {model}: {e}")
                
    except Exception as e:
        logger.error(f"Error in download_models: {e}")
        raise

if __name__ == "__main__":
    download_models()