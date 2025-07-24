#!/usr/bin/env python3
"""
Quick CUDA Fix for Remote Servers
Specifically designed for servers with CUDA 12.8
"""

import subprocess
import sys

def run_command(command):
    """Run command and return success status"""
    print(f"🔄 Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ Success")
        if result.stdout.strip():
            print(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e}")
        if e.stderr.strip():
            print(f"Error: {e.stderr.strip()}")
        return False

def test_cuda():
    """Test if CUDA is working"""
    test_code = """
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    try:
        x = torch.tensor([1.0]).cuda()
        print('✅ CUDA tensor test passed')
    except Exception as e:
        print(f'❌ CUDA tensor test failed: {e}')
else:
    print('❌ CUDA not available')
"""
    
    try:
        result = subprocess.run([sys.executable, '-c', test_code], 
                              capture_output=True, text=True, timeout=30)
        print("🧪 CUDA Test Results:")
        print(result.stdout)
        if result.stderr:
            print("Errors:")
            print(result.stderr)
        return "CUDA available: True" in result.stdout and "CUDA tensor test passed" in result.stdout
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    print("🚀 Quick CUDA Fix for Remote Server (CUDA 12.8)")
    print("="*50)
    
    # First, test current state
    print("\n1. Testing current CUDA state...")
    if test_cuda():
        print("🎉 CUDA is already working! No fix needed.")
        return
    
    print("\n2. Your CUDA 12.8 is not working with current PyTorch.")
    print("   Let's try different PyTorch versions...")
    
    # Option 1: PyTorch with CUDA 12.1 (most compatible with CUDA 12.8)
    print("\n🔧 Attempting Fix 1: PyTorch with CUDA 12.1")
    if run_command("pip uninstall -y torch torchaudio"):
        if run_command("pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121"):
            print("\n🧪 Testing PyTorch CUDA 12.1...")
            if test_cuda():
                print("🎉 SUCCESS! CUDA is now working with PyTorch CUDA 12.1")
                print("\nNext steps:")
                print("1. Install other requirements: pip install -r requirements.txt")
                print("2. Start the API: python -m app.main")
                print("3. Test health: curl http://localhost:8288/transcribe/health")
                return
    
    # Option 2: PyTorch with CUDA 11.8 (fallback)
    print("\n🔧 Attempting Fix 2: PyTorch with CUDA 11.8")
    if run_command("pip uninstall -y torch torchaudio"):
        if run_command("pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118"):
            print("\n🧪 Testing PyTorch CUDA 11.8...")
            if test_cuda():
                print("🎉 SUCCESS! CUDA is now working with PyTorch CUDA 11.8")
                print("\nNext steps:")
                print("1. Install other requirements: pip install -r requirements.txt")
                print("2. Start the API: python -m app.main")
                print("3. Test health: curl http://localhost:8288/transcribe/health")
                return
    
    # Option 3: CPU-only (most reliable for remote servers)
    print("\n🔧 Attempting Fix 3: CPU-only PyTorch (most reliable)")
    if run_command("pip uninstall -y torch torchaudio"):
        if run_command("pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu"):
            print("\n🧪 Testing CPU-only PyTorch...")
            try:
                result = subprocess.run([sys.executable, '-c', 
                    "import torch; print('✅ CPU PyTorch working')"],
                    capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    print("🎉 SUCCESS! CPU-only PyTorch is working")
                    print("\n⚠️  Note: This will use CPU for transcription (slower but reliable)")
                    print("CPU performance: ~0.5-1x real-time (still very usable)")
                    print("\nNext steps:")
                    print("1. Install other requirements: pip install -r requirements.txt")
                    print("2. Start the API: python -m app.main")
                    print("3. Test health: curl http://localhost:8288/transcribe/health")
                    return
            except Exception as e:
                print(f"❌ CPU test failed: {e}")
    
    print("\n❌ All fixes failed. Manual debugging needed:")
    print("1. Check if you have sufficient permissions")
    print("2. Verify internet connectivity for pip installs")
    print("3. Try running: python cuda_diagnostic.py")
    print("4. Contact your system administrator")

if __name__ == "__main__":
    main()
