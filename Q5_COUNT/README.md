# Tool-Calling Agent: Count Characters Correctly

## Goal
Overcome tokenizer blindness by wiring the model to Python execution.

## Overview
This project implements a minimal tool-calling agent that can accurately count characters and perform arithmetic by executing Python code, bypassing the limitations of tokenizer-based character counting.

## Architecture

### Tools
1. **python_exec** - Executes Python code snippets and returns stdout
2. **noop** - No-operation tool for regular responses

### Pipeline
1. User submits a query
2. Agent determines if computation is needed
3. If yes: generates Python code → executes → returns result
4. If no: uses noop tool for regular response

## Files

- `demo.py` - Main demonstration script with 5 example queries
- `showcase.py` - Complete formatted demonstration with full transcripts
- `tool_calling_agent.py` - Pure Python implementation with detailed logging  
- `langchain_agent.py` - LangChain-based implementation (more robust)
- `tokenizer_comparison.py` - Shows tokenizer blindness vs Python solution
- `requirements.txt` - Dependencies for LangChain version

## Usage

Run the basic demo:
```bash
python demo.py
```

Run the complete showcase:
```bash
python showcase.py
```

Run the full implementation:
```bash
python tool_calling_agent.py
```

See tokenizer blindness comparison:
```bash
python tokenizer_comparison.py
```

## Complete Example Transcripts

### Counting Queries (3)

**Example 1:** "How many 'r' in 'strawberry'?"
- **LLM Decision:** Use python_exec tool
- **Tool Call:** `python_exec(len([c for c in 'strawberry' if c == 'r']))`
- **Framework Output:** 3
- **Final Response:** "There are 3."

**Example 2:** "Count the letter 'a' in 'banana'"
- **LLM Decision:** Use python_exec tool
- **Tool Call:** `python_exec(len([c for c in 'banana' if c == 'a']))`
- **Framework Output:** 3
- **Final Response:** "The count is 3."

**Example 3:** "How many times does 'th' appear in 'the quick brown fox jumps over the lazy dog'?"
- **LLM Decision:** Use python_exec tool
- **Tool Call:** `python_exec('the quick brown fox jumps over the lazy dog'.count('th'))`
- **Framework Output:** 2
- **Final Response:** "There are 2."

### Arithmetic Queries (2)

**Example 4:** "Calculate 15 + 27"
- **LLM Decision:** Use python_exec tool
- **Tool Call:** `python_exec(15 + 27)`
- **Framework Output:** 42
- **Final Response:** "The result is 42."

**Example 5:** "What is the sum of 10, 20, 30, and 40?"
- **LLM Decision:** Use python_exec tool
- **Tool Call:** `python_exec(sum([10, 20, 30, 40]))`
- **Framework Output:** 100
- **Final Response:** "The sum is 100."

## Key Features

- **Tokenizer Blindness Solution**: Uses Python execution instead of relying on token-based counting
- **Safe Execution**: Runs Python code in subprocess with timeout protection
- **Pattern Recognition**: Automatically detects counting/arithmetic queries
- **Natural Language Response**: Converts numerical results back to conversational responses
- **Error Handling**: Graceful handling of code execution errors

## Implementation Details

### Query Classification
The agent uses keyword detection and regex patterns to identify:
- Counting operations: "how many", "count", "occurrences"
- Arithmetic operations: "calculate", "sum", mathematical operators
- String patterns: Character/substring extraction from quotes

### Code Generation
Automatically generates appropriate Python code:
- Character counting: `len([c for c in string if c == char])`
- Substring counting: `string.count(substring)`  
- Arithmetic: Direct mathematical expressions
- Aggregation: `sum()`, `average` calculations

### Safety Measures
- Subprocess execution with timeout (10 seconds)
- Limited imports (math, re, collections)
- Error capture and user-friendly messages
- No file system or network access

## Dependencies

Basic version (demo.py):
- Python 3.7+ standard library only

LangChain version:
- langchain>=0.1.0
- langchain-core>=0.1.0  
- langchain-community>=0.0.20

## Extension Points

The agent can be extended to handle:
- More complex mathematical expressions
- String manipulation operations
- Data analysis queries
- File processing tasks
- Integration with external APIs