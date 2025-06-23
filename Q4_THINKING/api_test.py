#!/usr/bin/env python3
"""
Mini test script to verify OpenAI API connection works
"""

import os
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

def test_api_connection():
    """Test basic API connection with a simple prompt"""
    try:
        client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "What is 2 + 3? Just answer with the number."}
            ],
            temperature=0,
            max_tokens=10
        )
        
        result = response.choices[0].message.content.strip()
        print(f"API Response: {result}")
        
        if "5" in result:
            print("✓ API connection successful!")
            return True
        else:
            print("✗ API returned unexpected result")
            return False
            
    except Exception as e:
        print(f"✗ API connection failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing OpenAI API connection...")
    test_api_connection()