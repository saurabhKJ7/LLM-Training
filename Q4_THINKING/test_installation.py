#!/usr/bin/env python3
"""
Test script to validate that the Thinking Mode application is properly installed
and all dependencies are available.
"""

import sys
import importlib

def test_dependencies():
    """Test that all required dependencies are available"""
    required_modules = [
        'json',
        'random',
        'collections',
        'typing',
        're',
        'os'
    ]
    
    optional_modules = [
        'openai',
        'dotenv'
    ]
    
    print("Testing Core Dependencies:")
    print("-" * 30)
    
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"✓ {module}")
        except ImportError:
            print(f"✗ {module} - MISSING")
            return False
    
    print("\nTesting Optional Dependencies:")
    print("-" * 30)
    
    for module in optional_modules:
        try:
            importlib.import_module(module)
            print(f"✓ {module}")
        except ImportError:
            print(f"✗ {module} - MISSING (install with: pip install {module})")
    
    return True

def test_demo_functionality():
    """Test basic demo functionality"""
    print("\nTesting Demo Functionality:")
    print("-" * 30)
    
    try:
        from demo_thinking_mode import MockThinkingModeApp
        
        app = MockThinkingModeApp()
        problems = app.get_demo_problems()
        
        if len(problems) > 0:
            print("✓ Demo problems loaded")
            
            # Test answer extraction
            test_completion = "Let's think step-by-step. The total cost is $19."
            answer = app.extract_numerical_answer(test_completion)
            
            if answer == 19.0:
                print("✓ Answer extraction working")
                return True
            else:
                print("✗ Answer extraction failed")
                return False
        else:
            print("✗ No demo problems found")
            return False
            
    except Exception as e:
        print(f"✗ Demo functionality test failed: {e}")
        return False

def main():
    """Main test function"""
    print("Thinking Mode Application - Installation Test")
    print("=" * 50)
    
    # Test Python version
    python_version = sys.version_info
    print(f"Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 6):
        print("✗ Python 3.6+ required")
        return False
    else:
        print("✓ Python version compatible")
    
    print()
    
    # Test dependencies
    deps_ok = test_dependencies()
    
    # Test demo functionality
    demo_ok = test_demo_functionality()
    
    print("\n" + "=" * 50)
    
    if deps_ok and demo_ok:
        print("✅ All tests passed! The application is ready to use.")
        print("\nNext steps:")
        print("1. Run demo: python demo_thinking_mode.py")
        print("2. For full version: Set up .env file and run thinking_mode.py")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)