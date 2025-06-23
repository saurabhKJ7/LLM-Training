import subprocess
import sys
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json
import re

@dataclass
class ToolCall:
    name: str
    args: Dict[str, Any]

@dataclass
class ToolResult:
    output: str
    success: bool

class PythonExecutor:
    """Tool for executing Python code snippets safely"""
    
    def __init__(self):
        self.name = "python_exec"
        self.description = "Execute Python code and return stdout"
    
    def execute(self, code: str) -> ToolResult:
        """Execute Python code in a subprocess and return the result"""
        try:
            # Create a safe Python script
            script = f"""
import sys
import math
import re
from collections import Counter

# Execute the user code
{code}
"""
            
            # Run the code in a subprocess for safety
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                output = result.stdout.strip()
                return ToolResult(output=output, success=True)
            else:
                error_msg = result.stderr.strip()
                return ToolResult(output=f"Error: {error_msg}", success=False)
                
        except subprocess.TimeoutExpired:
            return ToolResult(output="Error: Code execution timed out", success=False)
        except Exception as e:
            return ToolResult(output=f"Error: {str(e)}", success=False)

class NoopTool:
    """Tool that does nothing - used for normal responses"""
    
    def __init__(self):
        self.name = "noop"
        self.description = "No operation - used for regular responses"
    
    def execute(self) -> ToolResult:
        """Return empty result"""
        return ToolResult(output="", success=True)

class ToolCallingAgent:
    """Simple tool-calling agent that can execute Python code or return normal responses"""
    
    def __init__(self):
        self.python_tool = PythonExecutor()
        self.noop_tool = NoopTool()
        self.tools = {
            "python_exec": self.python_tool,
            "noop": self.noop_tool
        }
    
    def _should_use_python(self, query: str) -> bool:
        """Determine if query requires Python execution"""
        counting_keywords = ["count", "how many", "number of", "occurrences"]
        arithmetic_keywords = ["calculate", "compute", "evaluate", "add", "subtract", "multiply", "divide", "+", "-", "*", "/", "sum", "average"]
        
        query_lower = query.lower()
        
        # Check for counting or arithmetic operations
        if any(keyword in query_lower for keyword in counting_keywords + arithmetic_keywords):
            return True
        
        # Check for mathematical expressions
        if re.search(r'\d+\s*[+\-*/]\s*\d+', query):
            return True
            
        return False
    
    def _generate_python_code(self, query: str) -> str:
        """Generate Python code based on the query"""
        query_lower = query.lower()
        
        # Character counting patterns
        if "how many" in query_lower and "in" in query_lower:
            # Extract character and string using regex
            match = re.search(r"how many ['\"]?([^'\"]*)['\"]? in ['\"]([^'\"]*)['\"]", query_lower)
            if match:
                char_to_count = match.group(1)
                target_string = match.group(2)
                return f"print(len([c for c in '{target_string}' if c == '{char_to_count}']))"
        
        # Count letter patterns  
        if "count" in query_lower and "letter" in query_lower:
            match = re.search(r"count.*letter ['\"]?([^'\"]*)['\"]? in ['\"]([^'\"]*)['\"]", query_lower)
            if match:
                char_to_count = match.group(1)
                target_string = match.group(2)
                return f"print(len([c for c in '{target_string}' if c == '{char_to_count}']))"
        
        # How many times does X appear in Y
        if "how many times" in query_lower and "appear" in query_lower:
            match = re.search(r"how many times does ['\"]?([^'\"]*)['\"]? appear in ['\"]([^'\"]*)['\"]", query_lower)
            if match:
                substring = match.group(1)
                target_string = match.group(2)
                return f"print('{target_string}'.count('{substring}'))"
        
        # Arithmetic expressions
        math_match = re.search(r'(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)', query)
        if math_match:
            num1, op, num2 = math_match.groups()
            return f"print({num1} {op} {num2})"
        
        # Sum calculation
        if "sum" in query_lower:
            numbers = re.findall(r'\d+(?:\.\d+)?', query)
            if len(numbers) > 1:
                return f"print(sum([{', '.join(numbers)}]))"
        
        # Average calculation
        if "average" in query_lower:
            numbers = re.findall(r'\d+(?:\.\d+)?', query)
            if len(numbers) > 1:
                return f"print(sum([{', '.join(numbers)}]) / {len(numbers)})"
        
        # Default fallback for general counting
        return f"# Query: {query}\nprint('Unable to automatically generate code for this query')"
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a user query and return the result"""
        
        # Determine if we need to use Python
        if self._should_use_python(query):
            # Generate Python code
            code = self._generate_python_code(query)
            
            # Execute the code
            tool_result = self.python_tool.execute(code)
            
            if tool_result.success and tool_result.output:
                # Generate natural language response
                response = self._generate_response(query, tool_result.output)
            else:
                response = f"I encountered an error: {tool_result.output}"
            
            return {
                "user_input": query,
                "tool_call": {
                    "name": "python_exec",
                    "code": code
                },
                "tool_output": tool_result.output,
                "final_response": response
            }
        else:
            # Use noop tool for regular responses
            response = self._generate_direct_response(query)
            
            return {
                "user_input": query,
                "tool_call": {
                    "name": "noop"
                },
                "tool_output": "",
                "final_response": response
            }
    
    def _generate_response(self, query: str, result: str) -> str:
        """Generate natural language response based on query and result"""
        query_lower = query.lower()
        
        if "how many" in query_lower:
            return f"There are {result}."
        elif "count" in query_lower:
            return f"The count is {result}."
        elif any(op in query for op in ["+", "-", "*", "/"]):
            return f"The result is {result}."
        elif "sum" in query_lower:
            return f"The sum is {result}."
        elif "average" in query_lower:
            return f"The average is {result}."
        else:
            return f"The answer is {result}."
    
    def _generate_direct_response(self, query: str) -> str:
        """Generate direct response for non-computational queries"""
        return "I can help you with counting characters, substrings, and arithmetic calculations. Please ask me to count something or calculate a mathematical expression."

def run_examples():
    """Run example queries to demonstrate the agent"""
    
    agent = ToolCallingAgent()
    
    # Example queries
    queries = [
        # Counting queries
        "How many 'r' in 'strawberry'?",
        "Count the letter 'a' in 'banana'?",
        "How many times does 'th' appear in 'the quick brown fox jumps over the lazy dog'?",
        
        # Arithmetic queries  
        "Calculate 15 + 27",
        "What is the sum of 10, 20, 30, and 40?"
    ]
    
    print("=== Tool-Calling Agent Demonstration ===\n")
    
    for i, query in enumerate(queries, 1):
        print(f"Example {i}:")
        print("-" * 50)
        
        result = agent.process_query(query)
        
        print(f"User Input: {result['user_input']}")
        print(f"Tool Call: {result['tool_call']['name']}")
        
        if result['tool_call']['name'] == 'python_exec':
            print(f"Python Code: {result['tool_call']['code']}")
            print(f"Tool Output: {result['tool_output']}")
        
        print(f"Final Response: {result['final_response']}")
        print()

if __name__ == "__main__":
    run_examples()