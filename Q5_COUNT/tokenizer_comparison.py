#!/usr/bin/env python3
"""
Tokenizer Blindness Comparison
Demonstrates why LLMs struggle with character counting and how Python execution solves it.
"""

import subprocess
import sys
from typing import Dict, Any

def simulate_tokenizer_blindness():
    """Simulate how LLMs might incorrectly count characters due to tokenization"""
    
    # Common tokenizer failures
    test_cases = [
        {
            "text": "strawberry",
            "char": "r",
            "correct_count": 3,
            "llm_guess": 2,  # Common mistake due to tokenization
            "reasoning": "LLM sees 'strawberry' as tokens like ['straw', 'berry'] and miscounts"
        },
        {
            "text": "banana", 
            "char": "a",
            "correct_count": 3,
            "llm_guess": 2,
            "reasoning": "Tokenizer might split as ['ban', 'ana'] causing miscount"
        },
        {
            "text": "mississippi",
            "char": "s", 
            "correct_count": 4,
            "llm_guess": 3,
            "reasoning": "Complex tokenization leads to missing characters"
        }
    ]
    
    return test_cases

def python_solution(text: str, char: str) -> int:
    """Accurate character counting using Python"""
    return len([c for c in text if c == char])

def execute_python_code(code: str) -> str:
    """Execute Python code and return result"""
    try:
        script = f"print({code})"
        result = subprocess.run([sys.executable, "-c", script], 
                              capture_output=True, text=True, timeout=5)
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    print("🔍 TOKENIZER BLINDNESS vs PYTHON SOLUTION")
    print("=" * 60)
    print()
    
    test_cases = simulate_tokenizer_blindness()
    
    for i, case in enumerate(test_cases, 1):
        print(f"Test Case {i}: Count '{case['char']}' in '{case['text']}'")
        print("-" * 50)
        
        # Show the problem
        print(f"❌ Typical LLM (Tokenizer-based): {case['llm_guess']}")
        print(f"   Reason: {case['reasoning']}")
        
        # Show our solution
        python_code = f"len([c for c in '{case['text']}' if c == '{case['char']}'])"
        result = execute_python_code(python_code)
        
        print(f"✅ Our Python Tool: {result}")
        print(f"   Code: {python_code}")
        print(f"   Correct Answer: {case['correct_count']}")
        
        # Verification
        is_correct = int(result) == case['correct_count']
        print(f"   Status: {'CORRECT' if is_correct else 'ERROR'}")
        print()
    
    print("=" * 60)
    print("SUMMARY:")
    print("• LLMs struggle with character counting due to tokenization")
    print("• Tokens don't align with individual characters")
    print("• Python execution provides byte-level accuracy")
    print("• Tool-calling bridges the gap between NLP and computation")
    print()
    print("This is why tool-calling agents are essential for reliable")
    print("counting and mathematical operations!")

if __name__ == "__main__":
    main()