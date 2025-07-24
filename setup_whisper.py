#!/usr/bin/env python3
"""
Setup script for Whisper Audio Transcription API

This script helps install the correct dependencies based on your system's
CUDA compatibility.
"""

import subprocess
import sys
import importlib.util

def run_command(command, description):
    """Run a shell command and return success status"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Command: {command}")
        print(f"Error: {e.stderr}")
        return False

def check_cuda_available():
    """Check if CUDA is available and working"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def install_pytorch_cuda():
    """Install PyTorch with CUDA support"""
    commands = [
        "pip uninstall -y torch torchaudio",
        "pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121"
    ]
    
    for command in commands:
        if not run_command(command, f"Running: {command}"):
            return False
    return True

def install_pytorch_cpu():
    """Install PyTorch CPU-only version"""
    commands = [
        "pip uninstall -y torch torchaudio", 
        "pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu"
    ]
    
    for command in commands:
        if not run_command(command, f"Running: {command}"):
            return False
    return True

def install_other_dependencies():
    """Install other required dependencies"""
    return run_command("pip install -r requirements.txt", "Installing other dependencies")

def test_installation():
    """Test the installation"""
    print(f"\n🧪 Testing installation...")
    
    try:
        # Test PyTorch
        import torch
        print(f"✅ PyTorch {torch.__version__} imported successfully")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        # Test faster-whisper
        from faster_whisper import WhisperModel
        print(f"✅ faster-whisper imported successfully")
        
        # Test model initialization (tiny model for quick test)
        if torch.cuda.is_available():
            try:
                model = WhisperModel("tiny", device="cuda", compute_type="float16")
                print(f"✅ Whisper CUDA initialization successful")
                del model
            except Exception as e:
                print(f"⚠️  Whisper CUDA failed, but CPU should work: {e}")
                try:
                    model = WhisperModel("tiny", device="cpu", compute_type="int8")
                    print(f"✅ Whisper CPU initialization successful")
                    del model
                except Exception as cpu_e:
                    print(f"❌ Whisper CPU also failed: {cpu_e}")
                    return False
        else:
            try:
                model = WhisperModel("tiny", device="cpu", compute_type="int8")
                print(f"✅ Whisper CPU initialization successful")
                del model
            except Exception as e:
                print(f"❌ Whisper CPU initialization failed: {e}")
                return False
        
        # Test Flask
        import flask
        print(f"✅ Flask {flask.__version__} imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    print("🚀 Whisper Audio Transcription API Setup")
    print("="*50)
    
    print(f"Python version: {sys.version}")
    
    # Check if CUDA is currently available
    cuda_available = False
    try:
        # Run CUDA diagnostic first
        print(f"\n🔍 Running CUDA diagnostic...")
        result = subprocess.run([sys.executable, "cuda_diagnostic.py"], 
                              capture_output=True, text=True, timeout=30)
        print("Diagnostic output:")
        print(result.stdout)
        
        # Try to import torch to check CUDA
        import torch
        cuda_available = torch.cuda.is_available()
    except Exception as e:
        print(f"Initial CUDA check failed: {e}")
        cuda_available = False
    
    print(f"\n🎯 Installation Plan:")
    if cuda_available:
        print("📦 Installing PyTorch with CUDA support")
        install_success = install_pytorch_cuda()
    else:
        print("📦 Installing PyTorch CPU-only version")
        print("💡 This is safer and will definitely work, but slower for transcription")
        
        # Ask user preference
        try:
            choice = input("\nDo you want to try CUDA version anyway? (y/N): ").lower().strip()
            if choice == 'y' or choice == 'yes':
                print("🎲 Attempting CUDA installation...")
                install_success = install_pytorch_cuda()
            else:
                install_success = install_pytorch_cpu()
        except KeyboardInterrupt:
            print("\n❌ Installation cancelled")
            return
    
    if not install_success:
        print("❌ PyTorch installation failed")
        return
    
    # Install other dependencies
    if not install_other_dependencies():
        print("❌ Failed to install other dependencies")
        return
    
    # Test the installation
    if test_installation():
        print(f"\n🎉 Installation completed successfully!")
        print(f"\nNext steps:")
        print(f"1. Start the API server: python -m app.main")
        print(f"2. Test with: python test_transcription.py <audio_file>")
        print(f"3. Open transcription_client.html in your browser")
        print(f"4. Check health: curl http://localhost:8288/transcribe/health")
    else:
        print(f"\n❌ Installation completed but tests failed")
        print(f"You may need to run the CUDA diagnostic script for troubleshooting:")
        print(f"python cuda_diagnostic.py")

if __name__ == "__main__":
    main()
