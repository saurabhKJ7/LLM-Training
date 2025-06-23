#!/usr/bin/env python3
"""
Tool-Calling Agent: Count Characters Correctly
Goal: Overcome tokenizer blindness by wiring the model to Python.

This demo shows a minimal agent with two tools:
1. python_exec - runs Python snippets and returns stdout
2. noop - returns nothing (used for normal, non-tool responses)
"""

import subprocess
import sys
import re
from typing import Dict, Any

class ToolCallingAgent:
    """Simple agent that can execute Python code or return normal responses"""
    
    def __init__(self):
        self.tools = ["python_exec", "noop"]
    
    def _should_use_python(self, query: str) -> bool:
        """Determine if query requires Python execution"""
        counting_keywords = ["count", "how many", "number of", "occurrences"]
        arithmetic_keywords = ["calculate", "compute", "evaluate", "add", "subtract", 
                             "multiply", "divide", "+", "-", "*", "/", "sum", "average"]
        
        query_lower = query.lower()
        return (any(keyword in query_lower for keyword in counting_keywords + arithmetic_keywords) or
                re.search(r'\d+\s*[+\-*/]\s*\d+', query))
    
    def _generate_python_code(self, query: str) -> str:
        """Generate Python code based on the query"""
        query_lower = query.lower()
        
        # Pattern: How many 'X' in 'Y'
        match = re.search(r"how many ['\"]?([^'\"]*)['\"]? in ['\"]([^'\"]*)['\"]", query_lower)
        if match:
            char_to_count = match.group(1)
            target_string = match.group(2)
            return f"len([c for c in '{target_string}' if c == '{char_to_count}'])"
        
        # Pattern: Count the letter 'X' in 'Y'
        match = re.search(r"count.*letter ['\"]?([^'\"]*)['\"]? in ['\"]([^'\"]*)['\"]", query_lower)
        if match:
            char_to_count = match.group(1)
            target_string = match.group(2)
            return f"len([c for c in '{target_string}' if c == '{char_to_count}'])"
        
        # Pattern: How many times does 'X' appear in 'Y'
        match = re.search(r"how many times does ['\"]?([^'\"]*)['\"]? appear in ['\"]([^'\"]*)['\"]", query_lower)
        if match:
            substring = match.group(1)
            target_string = match.group(2)
            return f"'{target_string}'.count('{substring}')"
        
        # Pattern: Calculate X + Y
        match = re.search(r'calculate\s+(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)', query_lower)
        if match:
            num1, op, num2 = match.groups()
            return f"{num1} {op} {num2}"
        
        # Pattern: Sum of numbers
        if "sum" in query_lower:
            numbers = re.findall(r'\d+(?:\.\d+)?', query)
            if len(numbers) > 1:
                return f"sum([{', '.join(numbers)}])"
        
        return f"# Unable to parse: {query}"
    
    def _execute_python(self, code: str) -> str:
        """Execute Python code and return stdout"""
        try:
            script = f"print({code})"
            result = subprocess.run([sys.executable, "-c", script], 
                                  capture_output=True, text=True, timeout=10)
            return result.stdout.strip() if result.returncode == 0 else f"Error: {result.stderr.strip()}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _generate_response(self, query: str, result: str) -> str:
        """Generate natural language response"""
        query_lower = query.lower()
        
        if "how many" in query_lower:
            return f"There are {result}."
        elif "count" in query_lower:
            return f"The count is {result}."
        elif "sum" in query_lower:
            return f"The sum is {result}."
        else:
            return f"The result is {result}."
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process query and return complete transcript"""
        
        if self._should_use_python(query):
            code = self._generate_python_code(query)
            output = self._execute_python(code)
            response = self._generate_response(query, output) if not output.startswith("Error") else output
            
            return {
                "user_input": query,
                "tool_call": "python_exec",
                "code": code,
                "output": output,
                "final_response": response
            }
        else:
            return {
                "user_input": query,
                "tool_call": "noop",
                "code": None,
                "output": "",
                "final_response": "I can help with counting and arithmetic. Please ask me to count characters or calculate numbers."
            }

def main():
    """Run demonstration with 5 example queries"""
    
    agent = ToolCallingAgent()
    
    # 3 counting queries + 2 arithmetic queries
    queries = [
        "How many 'r' in 'strawberry'?",
        "Count the letter 'a' in 'banana'",
        "How many times does 'th' appear in 'the quick brown fox jumps over the lazy dog'?",
        "Calculate 15 + 27",
        "What is the sum of 10, 20, 30, and 40?"
    ]
    
    print("=== Tool-Calling Agent: Count Characters Correctly ===")
    print("Goal: Overcome tokenizer blindness by wiring the model to Python.\n")
    
    for i, query in enumerate(queries, 1):
        print(f"--- Example {i} ---")
        
        result = agent.process_query(query)
        
        print(f"User: \"{result['user_input']}\"")
        print(f"\nLLM: decides to call the {result['tool_call']} tool")
        
        if result['tool_call'] == 'python_exec':
            print(f"\nTool Call:")
            print(f"  python_exec({result['code']})")
            print(f"\nFramework: executes the code and returns {result['output']}")
        else:
            print(f"\nTool Call: noop (no operation)")
        
        print(f"\nLLM: responds, \"{result['final_response']}\"")
        print()

if __name__ == "__main__":
    main()