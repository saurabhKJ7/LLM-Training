#!/usr/bin/env python3
"""
Quick setup verification script
Run this to check if your environment is ready for RLHF training
"""

import sys
import importlib

def test_imports():
    """Test if all required packages can be imported"""
    required_packages = [
        'torch',
        'transformers', 
        'datasets',
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn'
    ]
    
    print("🔍 Testing package imports...")
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - Not found")
            failed_imports.append(package)
    
    return failed_imports

def test_torch():
    """Test PyTorch functionality"""
    try:
        import torch
        print(f"\n🔥 PyTorch version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA devices: {torch.cuda.device_count()}")
            print(f"   Current device: {torch.cuda.current_device()}")
        
        # Test basic tensor operations
        x = torch.randn(2, 3)
        y = torch.randn(3, 2)
        z = torch.mm(x, y)
        print(f"   Basic operations: ✅")
        return True
    except Exception as e:
        print(f"   PyTorch test failed: {e}")
        return False

def test_transformers():
    """Test Transformers library"""
    try:
        from transformers import GPT2Tokenizer
        print(f"\n🤖 Testing transformers...")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        test_text = "Hello world"
        tokens = tokenizer.encode(test_text)
        print(f"   Tokenizer test: ✅")
        return True
    except Exception as e:
        print(f"   Transformers test failed: {e}")
        return False

def main():
    print("="*50)
    print("    RLHF Setup Verification")
    print("="*50)
    
    # Test imports
    failed = test_imports()
    
    if failed:
        print(f"\n❌ Missing packages: {', '.join(failed)}")
        print("   Please run: pip install -r requirements.txt")
        return False
    
    # Test PyTorch
    if not test_torch():
        return False
    
    # Test Transformers
    if not test_transformers():
        return False
    
    print("\n" + "="*50)
    print("✅ All tests passed! Ready for RLHF training.")
    print("   Run: python run_rlhf.py")
    print("="*50)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)