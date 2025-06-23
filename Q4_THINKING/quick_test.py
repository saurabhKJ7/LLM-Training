#!/usr/bin/env python3
"""
Quick test script to verify the full Thinking Mode application works correctly
"""

import os
from dotenv import load_dotenv
from thinking_mode import ThinkingModeApp

# Load environment variables
load_dotenv()

def quick_test():
    """Run a quick test with just one problem"""
    print("Quick Test - Thinking Mode Application")
    print("=" * 40)
    
    # Check API key
    if not os.getenv('OPENAI_API_KEY'):
        print("✗ No OpenAI API key found in .env file")
        return False
    
    try:
        app = ThinkingModeApp()
        
        # Get one problem
        problems = app.get_gre_problems()
        test_problem = problems[0]  # Use first problem
        
        print(f"Testing problem: {test_problem['question'][:50]}...")
        print(f"Expected answer: {test_problem['answer']}")
        
        # Test deterministic approach
        print("\nTesting deterministic approach...")
        det_answer, det_completion = app.solve_deterministic(test_problem['question'])
        print(f"Deterministic answer: {det_answer}")
        
        # Test majority vote with fewer completions for speed
        print("\nTesting majority vote (3 completions)...")
        maj_answer, all_answers, all_completions = app.solve_with_majority_vote(
            test_problem['question'], num_completions=3
        )
        print(f"Majority vote answer: {maj_answer}")
        print(f"All answers: {all_answers}")
        
        # Check accuracy
        correct_answer = test_problem['answer']
        det_correct = abs(det_answer - correct_answer) < 0.01 if det_answer is not None else False
        maj_correct = abs(maj_answer - correct_answer) < 0.01 if maj_answer is not None else False
        
        print(f"\nResults:")
        print(f"Deterministic correct: {det_correct}")
        print(f"Majority vote correct: {maj_correct}")
        
        if det_correct or maj_correct:
            print("✓ Quick test PASSED!")
            return True
        else:
            print("✗ Quick test failed - neither method got correct answer")
            return False
            
    except Exception as e:
        print(f"✗ Quick test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = quick_test()
    if success:
        print("\n🎉 The full application is working correctly!")
        print("You can now run: python thinking_mode.py")
    else:
        print("\n❌ There was an issue with the application")