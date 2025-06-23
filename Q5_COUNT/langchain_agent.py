import os
import sys
import subprocess
from typing import Dict, Any, List, Optional, Type
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import re

class PythonExecutorInput(BaseModel):
    code: str = Field(description="Python code to execute")

class PythonExecutorTool(BaseTool):
    name = "python_exec"
    description = "Execute Python code and return the output. Use this for counting characters, arithmetic, or any computational tasks."
    args_schema: Type[BaseModel] = PythonExecutorInput
    
    def _run(self, code: str) -> str:
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
                return output if output else "Code executed successfully (no output)"
            else:
                error_msg = result.stderr.strip()
                return f"Error: {error_msg}"
                
        except subprocess.TimeoutExpired:
            return "Error: Code execution timed out"
        except Exception as e:
            return f"Error: {str(e)}"

class NoopTool(BaseTool):
    name = "noop"
    description = "Use this for general conversation when no computation is needed."
    
    def _run(self) -> str:
        """Return empty result for normal responses"""
        return ""

class MockLLM:
    """Mock LLM that simulates tool calling behavior"""
    
    def __init__(self):
        self.python_tool = PythonExecutorTool()
        self.noop_tool = NoopTool()
    
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
        
        # Substring counting
        if "count" in query_lower and "in" in query_lower:
            match = re.search(r"count ['\"]?([^'\"]*)['\"]? in ['\"]([^'\"]*)['\"]", query_lower)
            if match:
                substring = match.group(1)
                target_string = match.group(2)
                return f"print('{target_string}'.count('{substring}'))"
        
        # How many times does X appear
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
        
        # Calculate pattern
        if "calculate" in query_lower:
            match = re.search(r'calculate\s+(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)', query_lower)
            if match:
                num1, op, num2 = match.groups()
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
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a user query and return the interaction transcript"""
        
        if self._should_use_python(query):
            # Generate Python code
            code = self._generate_python_code(query)
            
            # Execute the code using the tool
            tool_output = self.python_tool._run(code)
            
            # Generate final response
            if tool_output and not tool_output.startswith("Error"):
                final_response = self._generate_response(query, tool_output)
            else:
                final_response = f"I encountered an error: {tool_output}"
            
            return {
                "user_input": query,
                "tool_call": {
                    "name": "python_exec",
                    "code": code
                },
                "tool_output": tool_output,
                "final_response": final_response
            }
        else:
            # Use noop tool for regular responses
            tool_output = self.noop_tool._run()
            final_response = "I can help you with counting characters, substrings, and arithmetic calculations. Please ask me to count something or calculate a mathematical expression."
            
            return {
                "user_input": query,
                "tool_call": {
                    "name": "noop"
                },
                "tool_output": tool_output,
                "final_response": final_response
            }

def run_examples():
    """Run example queries to demonstrate the agent"""
    
    # Create the mock LLM agent
    agent = MockLLM()
    
    # Example queries
    queries = [
        # Counting queries
        "How many 'r' in 'strawberry'?",
        "Count the letter 'a' in 'banana'",
        "How many times does 'th' appear in 'the quick brown fox jumps over the lazy dog'?",
        
        # Arithmetic queries  
        "Calculate 15 + 27",
        "What is the sum of 10, 20, 30, and 40?"
    ]
    
    print("=== Tool-Calling Agent: Count Characters Correctly ===\n")
    print("Goal: Overcome tokenizer blindness by wiring the model to Python.\n")
    
    for i, query in enumerate(queries, 1):
        print(f"Example {i}:")
        print("-" * 60)
        
        result = agent.process_query(query)
        
        print(f"User Input: {result['user_input']}")
        print(f"\nLLM Decision: Use {result['tool_call']['name']} tool")
        
        if result['tool_call']['name'] == 'python_exec':
            print(f"\nTool Call:")
            print(f"  Function: python_exec")
            print(f"  Code: {result['tool_call']['code']}")
            print(f"\nTool Output: {result['tool_output']}")
        else:
            print(f"\nTool Call: {result['tool_call']['name']} (no operation)")
        
        print(f"\nFinal LLM Response: {result['final_response']}")
        print("\n" + "="*60 + "\n")

def main():
    """Main function to run the agent demonstration"""
    run_examples()

if __name__ == "__main__":
    main()