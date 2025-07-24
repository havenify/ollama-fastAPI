#!/usr/bin/env python3
"""
CUDA Diagnostic Script for Whisper Audio Transcription API

This script helps diagnose CUDA compatibility issues and provides
recommendations for resolving them.
"""

import sys
import subprocess
import importlib.util

def check_nvidia_driver():
    """Check NVIDIA driver version"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Driver Version:' in line:
                    driver_version = line.split('Driver Version:')[1].split()[0]
                    print(f"✅ NVIDIA Driver Version: {driver_version}")
                    return driver_version
        else:
            print("❌ nvidia-smi command failed or NVIDIA drivers not installed")
            return None
    except subprocess.TimeoutExpired:
        print("❌ nvidia-smi command timed out")
        return None
    except FileNotFoundError:
        print("❌ nvidia-smi not found - NVIDIA drivers may not be installed")
        return None
    except Exception as e:
        print(f"❌ Error checking NVIDIA driver: {e}")
        return None

def check_cuda_toolkit():
    """Check CUDA toolkit installation"""
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'release' in line.lower():
                    cuda_version = line.split('release')[1].split(',')[0].strip()
                    print(f"✅ CUDA Toolkit Version: {cuda_version}")
                    return cuda_version
        else:
            print("❌ nvcc command failed - CUDA toolkit may not be installed")
            return None
    except subprocess.TimeoutExpired:
        print("❌ nvcc command timed out")
        return None
    except FileNotFoundError:
        print("❌ nvcc not found - CUDA toolkit not installed")
        return None
    except Exception as e:
        print(f"❌ Error checking CUDA toolkit: {e}")
        return None

def check_python_packages():
    """Check if required Python packages are installed"""
    packages = {
        'torch': 'PyTorch',
        'faster_whisper': 'Faster Whisper',
        'flask': 'Flask'
    }
    
    print("\n📦 Checking Python packages:")
    installed_packages = {}
    
    for package, display_name in packages.items():
        try:
            spec = importlib.util.find_spec(package)
            if spec is not None:
                # Try to import and get version
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'unknown')
                print(f"✅ {display_name}: {version}")
                installed_packages[package] = version
            else:
                print(f"❌ {display_name}: Not installed")
                installed_packages[package] = None
        except Exception as e:
            print(f"❌ {display_name}: Error checking - {e}")
            installed_packages[package] = None
    
    return installed_packages

def check_pytorch_cuda():
    """Check PyTorch CUDA compatibility"""
    try:
        import torch
        print(f"\n🔥 PyTorch CUDA Information:")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version (PyTorch): {torch.version.cuda}")
            print(f"CUDA device count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
                print(f"  Compute Capability: {props.major}.{props.minor}")
            
            # Test CUDA functionality
            try:
                test_tensor = torch.tensor([1.0, 2.0, 3.0]).cuda()
                result = test_tensor * 2
                print("✅ CUDA tensor operations working")
                return True
            except Exception as cuda_error:
                print(f"❌ CUDA tensor operations failed: {cuda_error}")
                if "CUDA driver version is insufficient" in str(cuda_error):
                    print("🚨 CUDA driver version is insufficient for CUDA runtime version")
                return False
        else:
            print("❌ CUDA not available in PyTorch")
            return False
            
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    except Exception as e:
        print(f"❌ Error checking PyTorch CUDA: {e}")
        return False

def test_whisper_initialization():
    """Test Whisper model initialization"""
    print(f"\n🎵 Testing Whisper Model Initialization:")
    
    try:
        from faster_whisper import WhisperModel
        
        # Test CUDA initialization
        print("Testing CUDA initialization...")
        try:
            model = WhisperModel("tiny", device="cuda", compute_type="float16")
            print("✅ Whisper CUDA initialization successful (tiny model)")
            del model  # Free memory
        except Exception as cuda_error:
            print(f"❌ Whisper CUDA initialization failed: {cuda_error}")
            
            # Test CPU fallback
            print("Testing CPU fallback...")
            try:
                model = WhisperModel("tiny", device="cpu", compute_type="int8")
                print("✅ Whisper CPU initialization successful (tiny model)")
                del model
            except Exception as cpu_error:
                print(f"❌ Whisper CPU initialization failed: {cpu_error}")
                return False
        
        return True
        
    except ImportError:
        print("❌ faster-whisper not installed")
        return False
    except Exception as e:
        print(f"❌ Error testing Whisper: {e}")
        return False

def provide_recommendations(driver_version, cuda_toolkit, pytorch_cuda_works):
    """Provide recommendations based on diagnostic results"""
    print(f"\n💡 Recommendations for Remote Server:")
    
    if not driver_version:
        print("1. NVIDIA drivers not found:")
        print("   - Contact your system administrator to install NVIDIA drivers")
        print("   - Or use CPU-only mode if GPU acceleration is not available")
    
    if not cuda_toolkit:
        print("2. CUDA toolkit not found:")
        print("   - This is normal - PyTorch includes its own CUDA runtime")
        print("   - No action needed if PyTorch works correctly")
    
    if not pytorch_cuda_works:
        print("3. CUDA not working with PyTorch - FOR REMOTE SERVER:")
        print("   Since you have CUDA 12.8, try these PyTorch versions in order:")
        print("   ")
        print("   Option A - Latest PyTorch with CUDA 12.1 (most compatible):")
        print("     pip uninstall torch torchaudio")
        print("     pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("   ")
        print("   Option B - PyTorch with CUDA 11.8 (fallback):")
        print("     pip uninstall torch torchaudio") 
        print("     pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("   ")
        print("   Option C - CPU-only version (safest for remote servers):")
        print("     pip uninstall torch torchaudio")
        print("     pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu")
        print("   ")
        print("   💡 For remote servers, CPU-only is often more reliable!")
    
    print("\n4. For Remote Server Whisper API:")
    print("   - CPU performance: ~0.5-1x real-time (still very usable)")
    print("   - GPU performance: ~0.1x real-time (if CUDA works)")
    print("   - Consider using 'base' model instead of 'large-v3' for CPU")
    print("   - The service will automatically fallback to CPU if CUDA fails")
    
    print("\n5. Remote Server CUDA Compatibility:")
    print("   - Your CUDA 12.8 should work with PyTorch CUDA 12.1 builds")
    print("   - NVIDIA Driver 450.80.02+ required for CUDA 11.0+")
    print("   - Driver 520.61.05+ required for CUDA 11.8+") 
    print("   - Driver 525.60.13+ required for CUDA 12.0+")
    print("   - Driver 530.30.02+ required for CUDA 12.1+")
    
    print("\n6. Quick Test Commands:")
    print("   Test current PyTorch: python -c \"import torch; print(torch.cuda.is_available())\"")
    print("   Check CUDA version: nvidia-smi")
    print("   Test after reinstall: python cuda_diagnostic.py")

def main():
    print("🔍 CUDA Diagnostic Script for Whisper Audio Transcription API")
    print("=" * 70)
    
    print("\n🖥️  System Information:")
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    
    # Check NVIDIA driver
    print(f"\n🚗 NVIDIA Driver Check:")
    driver_version = check_nvidia_driver()
    
    # Check CUDA toolkit
    print(f"\n🛠️  CUDA Toolkit Check:")
    cuda_toolkit = check_cuda_toolkit()
    
    # Check Python packages
    packages = check_python_packages()
    
    # Check PyTorch CUDA
    pytorch_cuda_works = check_pytorch_cuda()
    
    # Test Whisper
    whisper_works = test_whisper_initialization()
    
    # Provide recommendations
    provide_recommendations(driver_version, cuda_toolkit, pytorch_cuda_works)
    
    print(f"\n📊 Summary:")
    print(f"NVIDIA Driver: {'✅' if driver_version else '❌'}")
    print(f"CUDA Toolkit: {'✅' if cuda_toolkit else '⚠️'}")
    print(f"PyTorch CUDA: {'✅' if pytorch_cuda_works else '❌'}")
    print(f"Whisper Model: {'✅' if whisper_works else '❌'}")
    
    if pytorch_cuda_works and whisper_works:
        print(f"\n🎉 Your system should work with CUDA acceleration!")
    elif whisper_works:
        print(f"\n⚠️  Your system will work with CPU fallback")
    else:
        print(f"\n❌ Issues detected - please follow the recommendations above")

if __name__ == "__main__":
    main()
