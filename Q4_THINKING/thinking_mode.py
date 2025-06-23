import openai
import re
import json
import random
from collections import Counter
from typing import List, Dict, Tuple
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ThinkingModeApp:
    def __init__(self):
        """Initialize the Thinking Mode application"""
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model = "gpt-3.5-turbo"
        
    def get_gre_problems(self) -> List[Dict]:
        """Sample GRE-style arithmetic word problems"""
        problems = [
            {
                "question": "A store sells apples for $2.50 per pound and oranges for $3.00 per pound. If John buys 4 pounds of apples and 3 pounds of oranges, how much does he spend in total?",
                "answer": 19.0
            },
            {
                "question": "A train travels 240 miles in 4 hours. At this rate, how many miles will it travel in 7 hours?",
                "answer": 420.0
            },
            {
                "question": "Sarah has 45 marbles. She gives away 1/3 of them to her brother and 1/5 of the remaining marbles to her sister. How many marbles does she have left?",
                "answer": 24.0
            },
            {
                "question": "A rectangular garden has a length of 12 feet and a width of 8 feet. If the owner wants to put a fence around the entire perimeter, how many feet of fencing will be needed?",
                "answer": 40.0
            },
            {
                "question": "In a class of 30 students, 18 students like pizza, 12 students like burgers, and 6 students like both. How many students like neither pizza nor burgers?",
                "answer": 6.0
            },
            {
                "question": "A car's gas tank holds 15 gallons. If the car uses 3 gallons every 90 miles, how many miles can it travel on a full tank?",
                "answer": 450.0
            },
            {
                "question": "Tom works 8 hours a day and earns $15 per hour. If he works 5 days a week, how much does he earn in 3 weeks?",
                "answer": 1800.0
            },
            {
                "question": "A recipe calls for 2.5 cups of flour to make 12 cookies. How many cups of flour are needed to make 36 cookies?",
                "answer": 7.5
            },
            {
                "question": "The temperature dropped from 75°F to 45°F over 6 hours. What was the average rate of temperature change per hour?",
                "answer": 5.0
            },
            {
                "question": "A book costs $24. If there's a 25% discount, and then an 8% sales tax is added to the discounted price, what is the final price?",
                "answer": 19.44
            }
        ]
        return problems
    
    def generate_completion(self, problem: str, temperature: float = 1.1) -> str:
        """Generate a single completion for a problem"""
        prompt = f"""Solve this arithmetic problem step by step.

Problem: {problem}

Let's think step-by-step."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that solves math problems step by step."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating completion: {e}")
            return ""
    
    def extract_numerical_answer(self, completion: str) -> float:
        """Extract numerical answer from completion text"""
        # Look for patterns like "The answer is X", "Final answer: X", etc.
        patterns = [
            r'(?:final answer|answer|result|total)(?:\s*is|\s*:)\s*\$?(\d+(?:\.\d+)?)',
            r'(?:therefore|thus|so),?\s*(?:the answer is)?\s*\$?(\d+(?:\.\d+)?)',
            r'\$?(\d+(?:\.\d+)?)\s*(?:dollars?|is the (?:final )?answer)',
            r'(?:equals?|=)\s*\$?(\d+(?:\.\d+)?)',
            r'\b(\d+(?:\.\d+)?)\s*(?:dollars?|feet|miles|marbles|students|hours?|cups?|degrees?|gallons?)\b'
        ]
        
        completion_lower = completion.lower()
        
        for pattern in patterns:
            matches = re.findall(pattern, completion_lower, re.IGNORECASE)
            if matches:
                try:
                    return float(matches[-1])  # Take the last match
                except ValueError:
                    continue
        
        # Fallback: extract all numbers and take the last one
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', completion)
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                pass
        
        return None
    
    def solve_with_majority_vote(self, problem: str, num_completions: int = 10) -> Tuple[float, List[float], List[str]]:
        """Solve problem using majority vote over multiple completions"""
        print(f"Generating {num_completions} completions for majority vote...")
        
        completions = []
        answers = []
        
        for i in range(num_completions):
            print(f"  Completion {i+1}/{num_completions}")
            completion = self.generate_completion(problem, temperature=1.1)
            answer = self.extract_numerical_answer(completion)
            
            completions.append(completion)
            if answer is not None:
                answers.append(answer)
        
        if not answers:
            return None, [], completions
        
        # Perform majority vote
        answer_counts = Counter(answers)
        majority_answer = answer_counts.most_common(1)[0][0]
        
        return majority_answer, answers, completions
    
    def solve_deterministic(self, problem: str) -> Tuple[float, str]:
        """Solve problem with single deterministic run"""
        print("Generating deterministic solution...")
        completion = self.generate_completion(problem, temperature=0.0)
        answer = self.extract_numerical_answer(completion)
        return answer, completion
    
    def evaluate_problem(self, problem_data: Dict) -> Dict:
        """Evaluate a single problem with both methods"""
        question = problem_data["question"]
        correct_answer = problem_data["answer"]
        
        print(f"\n{'='*60}")
        print(f"Problem: {question}")
        print(f"Correct Answer: {correct_answer}")
        print(f"{'='*60}")
        
        # Method A: Deterministic
        det_answer, det_completion = self.solve_deterministic(question)
        det_correct = abs(det_answer - correct_answer) < 0.01 if det_answer is not None else False
        
        print(f"\nDeterministic Answer: {det_answer}")
        print(f"Deterministic Correct: {det_correct}")
        
        # Method B: Majority Vote
        maj_answer, all_answers, all_completions = self.solve_with_majority_vote(question)
        maj_correct = abs(maj_answer - correct_answer) < 0.01 if maj_answer is not None else False
        
        print(f"\nMajority Vote Answer: {maj_answer}")
        print(f"All Answers: {all_answers}")
        print(f"Answer Distribution: {dict(Counter(all_answers))}")
        print(f"Majority Vote Correct: {maj_correct}")
        
        return {
            "question": question,
            "correct_answer": correct_answer,
            "deterministic": {
                "answer": det_answer,
                "correct": det_correct,
                "completion": det_completion
            },
            "majority_vote": {
                "answer": maj_answer,
                "correct": maj_correct,
                "all_answers": all_answers,
                "completions": all_completions
            }
        }
    
    def run_evaluation(self, num_problems: int = 10):
        """Run evaluation on multiple problems"""
        problems = self.get_gre_problems()[:num_problems]
        results = []
        
        det_correct = 0
        maj_correct = 0
        
        for i, problem in enumerate(problems):
            print(f"\n{'#'*60}")
            print(f"EVALUATING PROBLEM {i+1}/{len(problems)}")
            print(f"{'#'*60}")
            
            result = self.evaluate_problem(problem)
            results.append(result)
            
            if result["deterministic"]["correct"]:
                det_correct += 1
            if result["majority_vote"]["correct"]:
                maj_correct += 1
        
        # Final Summary
        print(f"\n{'='*60}")
        print("FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Total Problems: {len(problems)}")
        print(f"Deterministic Accuracy: {det_correct}/{len(problems)} ({det_correct/len(problems)*100:.1f}%)")
        print(f"Majority Vote Accuracy: {maj_correct}/{len(problems)} ({maj_correct/len(problems)*100:.1f}%)")
        
        # Save results
        with open('results.json', 'w') as f:
            json.dump({
                "summary": {
                    "total_problems": len(problems),
                    "deterministic_correct": det_correct,
                    "majority_vote_correct": maj_correct,
                    "deterministic_accuracy": det_correct/len(problems),
                    "majority_vote_accuracy": maj_correct/len(problems)
                },
                "detailed_results": results
            }, f, indent=2)
        
        print(f"\nDetailed results saved to results.json")
        
        return results

def main():
    """Main function to run the thinking mode application"""
    print("Thinking Mode Application")
    print("========================")
    
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: Please set your OPENAI_API_KEY environment variable")
        print("You can create a .env file with: OPENAI_API_KEY=your_key_here")
        return
    
    app = ThinkingModeApp()
    
    # Run evaluation
    try:
        results = app.run_evaluation(num_problems=5)  # Start with 5 problems for testing
        print("\nEvaluation completed successfully!")
    except Exception as e:
        print(f"Error during evaluation: {e}")

if __name__ == "__main__":
    main()