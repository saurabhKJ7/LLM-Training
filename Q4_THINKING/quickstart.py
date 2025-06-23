#!/usr/bin/env python3
"""
Quickstart script for the Thinking Mode Application
This script provides an easy menu-driven interface to run the application.
"""

import os
import sys
import subprocess

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import openai
        import dotenv
        return True
    except ImportError:
        return False

def install_dependencies():
    """Install required dependencies"""
    print("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("✗ Failed to install dependencies")
        return False

def setup_env_file():
    """Help user set up environment file"""
    if os.path.exists('.env'):
        print("✓ .env file already exists")
        return True
    
    print("\nSetting up environment file...")
    api_key = input("Enter your OpenAI API key (or press Enter to skip): ").strip()
    
    if api_key:
        with open('.env', 'w') as f:
            f.write(f"OPENAI_API_KEY={api_key}\n")
        print("✓ .env file created successfully!")
        return True
    else:
        print("⚠ Skipping .env setup - you can run demo mode without API key")
        return False

def run_demo():
    """Run the demo version"""
    print("\n" + "="*50)
    print("Running Demo Mode...")
    print("="*50)
    try:
        subprocess.run([sys.executable, "demo_thinking_mode.py"])
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")

def run_full_version():
    """Run the full version with OpenAI API"""
    if not os.path.exists('.env'):
        print("✗ .env file not found. Please set up your API key first.")
        return
    
    print("\n" + "="*50)
    print("Running Full Version...")
    print("="*50)
    try:
        subprocess.run([sys.executable, "thinking_mode.py"])
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")

def run_tests():
    """Run installation tests"""
    print("\n" + "="*50)
    print("Running Tests...")
    print("="*50)
    try:
        subprocess.run([sys.executable, "test_installation.py"])
    except KeyboardInterrupt:
        print("\nTests interrupted by user")

def show_menu():
    """Display the main menu"""
    print("\n" + "="*50)
    print("🧠 THINKING MODE APPLICATION")
    print("="*50)
    print("1. Run Demo (no API key required)")
    print("2. Run Full Version (requires OpenAI API key)")
    print("3. Install Dependencies")
    print("4. Setup Environment (.env file)")
    print("5. Run Tests")
    print("6. View README")
    print("7. Exit")
    print("-"*50)

def view_readme():
    """Display README content"""
    if os.path.exists('README.md'):
        print("\n" + "="*50)
        print("README")
        print("="*50)
        with open('README.md', 'r') as f:
            content = f.read()
            # Show first 50 lines
            lines = content.split('\n')[:50]
            print('\n'.join(lines))
            if len(content.split('\n')) > 50:
                print("\n... (truncated, see README.md for full content)")
    else:
        print("README.md not found")

def main():
    """Main quickstart function"""
    print("Welcome to the Thinking Mode Application Quickstart!")
    
    while True:
        show_menu()
        choice = input("Select an option (1-7): ").strip()
        
        if choice == '1':
            run_demo()
        elif choice == '2':
            run_full_version()
        elif choice == '3':
            if install_dependencies():
                print("Dependencies installed successfully!")
            else:
                print("Failed to install dependencies. Please check the error messages above.")
        elif choice == '4':
            setup_env_file()
        elif choice == '5':
            run_tests()
        elif choice == '6':
            view_readme()
        elif choice == '7':
            print("Thank you for using Thinking Mode Application!")
            break
        else:
            print("Invalid choice. Please select 1-7.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        sys.exit(0)