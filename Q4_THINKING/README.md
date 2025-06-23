# Thinking Mode Application

A Python application that explores chain-of-thought reasoning with self-consistency by implementing majority voting over multiple AI completions.

## Overview

This application demonstrates two approaches to solving GRE-style arithmetic word problems:

1. **Deterministic Approach**: Single completion with temperature=0
2. **Majority Vote Approach**: 10 completions with temperature=1.1, then majority voting

## Features

- 🧠 Chain-of-thought reasoning with "Let's think step-by-step" prompts
- 🗳️ Self-consistency through majority voting
- 📊 Automatic answer extraction from text completions
- 📈 Accuracy comparison between methods
- 🎯 Built-in GRE-style arithmetic problems
- 🔧 Demo mode with mock responses (no API key required)

## Installation

1. Clone or download this project
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Option 1: Use the Interactive Menu
```bash
python quickstart.py
```

This provides a menu-driven interface to:
- Run the demo
- Set up your API key 
- Install dependencies
- Run tests
- Launch the full application

### Option 2: Demo Mode (No API Key Required)

Run the demo version to see how the application works:

```bash
python demo_thinking_mode.py
```

This uses pre-written mock responses to demonstrate the concept without requiring an OpenAI API key.

### Option 3: Full Version (Requires OpenAI API Key)

1. Get an OpenAI API key from [OpenAI Platform](https://platform.openai.com/api-keys)

2. Create a `.env` file (copy from `.env.example`):
```bash
cp .env.example .env
```

3. Edit `.env` and add your API key:
```
OPENAI_API_KEY=your_actual_api_key_here
```

4. Test the connection:
```bash
python api_test.py
```

5. Run the full application:
```bash
python thinking_mode.py
```

## How It Works

### 1. Problem Setup
The application uses GRE-style arithmetic word problems like:
- "A store sells apples for $2.50 per pound and oranges for $3.00 per pound. If John buys 4 pounds of apples and 3 pounds of oranges, how much does he spend in total?"

### 2. Chain-of-Thought Prompting
Each problem is prompted with "Let's think step-by-step" to encourage reasoning.

### 3. Multiple Completions
- **Deterministic**: 1 completion with temperature=0
- **Majority Vote**: 10 completions with temperature=1.1

### 4. Answer Extraction
Uses regex patterns to extract numerical answers from text:
- "The answer is X"
- "Total cost = $X"
- "Final answer: X"
- And more patterns...

### 5. Majority Voting
Counts all extracted answers and selects the most frequent one.

### 6. Evaluation
Compares both methods' accuracy against known correct answers.

## Sample Output

```
EVALUATING PROBLEM 1/2
============================================================
Problem: A store sells apples for $2.50 per pound and oranges for $3.00 per pound...
Correct Answer: 19.0
============================================================

Generating deterministic solution...
Deterministic Answer: 19.0
Deterministic Correct: True

Generating 10 completions for majority vote...
  Completion 1/10
  Completion 2/10
  ...
Majority Vote Answer: 19.0
All Answers: [19.0, 19.0, 19.0, 18.0, 19.0, 19.0, 19.0, 19.0, 19.0, 19.0]
Answer Distribution: {19.0: 9, 18.0: 1}
Majority Vote Correct: True

============================================================
FINAL RESULTS
============================================================
Total Problems: 2
Deterministic Accuracy: 2/2 (100.0%)
Majority Vote Accuracy: 2/2 (100.0%)
```

## Files Structure

```
Q4_THINKING/
├── thinking_mode.py          # Main application (requires OpenAI API)
├── demo_thinking_mode.py     # Demo version (mock responses)
├── quickstart.py            # Interactive menu interface
├── api_test.py              # Test OpenAI API connection
├── quick_test.py            # Quick functionality test
├── test_installation.py     # Comprehensive installation test
├── requirements.txt          # Python dependencies
├── .env.example             # Environment file template
├── README.md               # This file
├── results.json            # Results from full version
└── demo_results.json       # Results from demo version
```

## Key Classes and Methods

### ThinkingModeApp (Full Version)
- `generate_completion()`: Creates AI completions via OpenAI API
- `extract_numerical_answer()`: Extracts numbers from text
- `solve_with_majority_vote()`: Implements majority voting
- `solve_deterministic()`: Single deterministic solution
- `run_evaluation()`: Evaluates multiple problems

### MockThinkingModeApp (Demo Version)
- Same interface but uses pre-written mock responses
- Demonstrates the concept without API costs

## Configuration

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key
- `MODEL`: AI model to use (default: gpt-3.5-turbo)

### Customizable Parameters
- Number of problems to evaluate
- Number of completions for majority vote (default: 10)
- Temperature settings
- Regex patterns for answer extraction

## Expected Results

The majority vote approach typically shows:
- **Higher accuracy** on problems where the model sometimes makes calculation errors
- **Better consistency** across multiple runs
- **More robust** performance on complex word problems

## Testing

### Run All Tests
```bash
python test_installation.py
```

### Test API Connection Only
```bash
python api_test.py
```

### Quick Functionality Test
```bash
python quick_test.py
```

## Troubleshooting

### Common Issues

1. **API Key Error**: Make sure your `.env` file contains a valid OpenAI API key
2. **Rate Limits**: If you hit rate limits, reduce the number of problems or add delays
3. **Answer Extraction**: If answers aren't being extracted, check the regex patterns
4. **Dependencies**: Run `pip install -r requirements.txt` to install all required packages

### Debug Mode

To see more detailed output, you can modify the print statements in the code or add logging.

## Cost Considerations

- Each problem requires 11 API calls (1 deterministic + 10 majority vote)
- With 10 problems, that's 110 API calls
- Estimated cost with gpt-3.5-turbo: ~$0.50-1.00 for full evaluation

## Extending the Application

### Adding New Problems
Add problems to the `get_gre_problems()` method:

```python
{
    "question": "Your new problem here...",
    "answer": 42.0
}
```

### Improving Answer Extraction
Add new regex patterns to `extract_numerical_answer()`:

```python
r'your_new_pattern_here'
```

### Different Models
Change the model in the constructor:

```python
self.model = "gpt-4"  # or other models
```

## License

This project is for educational purposes. Please respect OpenAI's usage policies when using their API.