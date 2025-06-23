import random
import re
import json
from collections import Counter
from typing import List, Dict, Tuple

class MockThinkingModeApp:
    def __init__(self):
        """Initialize the demo Thinking Mode application with mock responses"""
        self.mock_responses = {
            "apples_oranges": [
                "Let's think step-by-step. John buys 4 pounds of apples at $2.50 per pound = 4 × $2.50 = $10. He buys 3 pounds of oranges at $3.00 per pound = 3 × $3.00 = $9. Total cost = $10 + $9 = $19.",
                "Let's think step-by-step. Apples cost: 4 × $2.50 = $10.00. Oranges cost: 3 × $3.00 = $9.00. The total is $10 + $9 = $19.",
                "Let's think step-by-step. 4 pounds of apples = 4 × 2.5 = $10. 3 pounds of oranges = 3 × 3 = $9. Total = $19.",
                "Let's think step-by-step. Cost of apples = 4 × $2.50 = $10. Cost of oranges = 3 × $3.00 = $9. Total cost = $19.",
                "Let's think step-by-step. John spends 4 × $2.50 = $10 on apples and 3 × $3.00 = $9 on oranges. So $10 + $9 = $19 total.",
                "Let's think step-by-step. Apples: 4 × $2.50 = $10. Oranges: 3 × $3.00 = $9. Total: $19.",
                "Let's think step-by-step. 4 pounds × $2.50 = $10 for apples. 3 pounds × $3.00 = $9 for oranges. $10 + $9 = $19.",
                "Let's think step-by-step. The apples cost $10 (4 × $2.50). The oranges cost $9 (3 × $3.00). Total is $19.",
                "Let's think step-by-step. For apples: 4 × 2.5 = 10. For oranges: 3 × 3 = 9. Total: 19 dollars.",
                "Let's think step-by-step. Apples = $2.50 × 4 = $10. Oranges = $3.00 × 3 = $9. Total = $19."
            ],
            "train_distance": [
                "Let's think step-by-step. The train travels 240 miles in 4 hours, so its speed is 240 ÷ 4 = 60 mph. In 7 hours, it will travel 60 × 7 = 420 miles.",
                "Let's think step-by-step. Speed = distance ÷ time = 240 ÷ 4 = 60 mph. Distance in 7 hours = 60 × 7 = 420 miles.",
                "Let's think step-by-step. Rate is 240 miles ÷ 4 hours = 60 miles per hour. So in 7 hours: 60 × 7 = 420 miles.",
                "Let's think step-by-step. The train's speed is 240/4 = 60 mph. At this rate, in 7 hours it travels 60 × 7 = 420 miles.",
                "Let's think step-by-step. First find the rate: 240 miles ÷ 4 hours = 60 mph. Then 7 hours × 60 mph = 420 miles.",
                "Let's think step-by-step. Speed = 240 miles / 4 hours = 60 miles per hour. Distance = 60 mph × 7 hours = 420 miles.",
                "Let's think step-by-step. The train goes 60 miles per hour (240/4). In 7 hours: 60 × 7 = 420 miles.",
                "Let's think step-by-step. Rate = 240/4 = 60 mph. Distance in 7 hours = 60 × 7 = 420 miles.",
                "Let's think step-by-step. Speed is 240 ÷ 4 = 60 miles per hour. So 7 × 60 = 420 miles in 7 hours.",
                "Let's think step-by-step. 240 miles in 4 hours means 60 mph. Therefore, 7 hours × 60 mph = 420 miles."
            ]
        }
        
    def get_demo_problems(self) -> List[Dict]:
        """Demo GRE-style arithmetic word problems"""
        problems = [
            {
                "question": "A store sells apples for $2.50 per pound and oranges for $3.00 per pound. If John buys 4 pounds of apples and 3 pounds of oranges, how much does he spend in total?",
                "answer": 19.0,
                "mock_key": "apples_oranges"
            },
            {
                "question": "A train travels 240 miles in 4 hours. At this rate, how many miles will it travel in 7 hours?",
                "answer": 420.0,
                "mock_key": "train_distance"
            }
        ]
        return problems
    
    def generate_mock_completion(self, mock_key: str, temperature: float = 1.1) -> str:
        """Generate a mock completion based on the problem type"""
        if mock_key in self.mock_responses:
            responses = self.mock_responses[mock_key]
            if temperature > 0.5:
                # Add some variation for high temperature
                response = random.choice(responses)
                # Occasionally introduce a calculation error for realism
                if random.random() < 0.1:  # 10% chance of error
                    response = response.replace("$19", "$18" if "19" in response else response)
                    response = response.replace("420", "410" if "420" in response else response)
                return response
            else:
                # Deterministic response for temperature 0
                return responses[0]
        return "Let's think step-by-step. I need to solve this problem."
    
    def extract_numerical_answer(self, completion: str) -> float:
        """Extract numerical answer from completion text"""
        patterns = [
            r'(?:final answer|answer|result|total)(?:\s*is|\s*:|\s*=)\s*\$?(\d+(?:\.\d+)?)',
            r'(?:therefore|thus|so),?\s*(?:the answer is)?\s*\$?(\d+(?:\.\d+)?)',
            r'\$?(\d+(?:\.\d+)?)\s*(?:dollars?|is the (?:final )?answer)',
            r'(?:equals?|=)\s*\$?(\d+(?:\.\d+)?)',
            r'Total(?:\s*cost)?\s*(?:is|=)\s*\$?(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*(?:miles|dollars?)\b'
        ]
        
        completion_lower = completion.lower()
        
        for pattern in patterns:
            matches = re.findall(pattern, completion_lower, re.IGNORECASE)
            if matches:
                try:
                    return float(matches[-1])
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
    
    def solve_with_majority_vote(self, problem_data: Dict, num_completions: int = 10) -> Tuple[float, List[float], List[str]]:
        """Solve problem using majority vote over multiple completions"""
        mock_key = problem_data["mock_key"]
        print(f"Generating {num_completions} mock completions for majority vote...")
        
        completions = []
        answers = []
        
        for i in range(num_completions):
            print(f"  Completion {i+1}/{num_completions}")
            completion = self.generate_mock_completion(mock_key, temperature=1.1)
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
    
    def solve_deterministic(self, problem_data: Dict) -> Tuple[float, str]:
        """Solve problem with single deterministic run"""
        mock_key = problem_data["mock_key"]
        print("Generating deterministic solution...")
        completion = self.generate_mock_completion(mock_key, temperature=0.0)
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
        det_answer, det_completion = self.solve_deterministic(problem_data)
        det_correct = abs(det_answer - correct_answer) < 0.01 if det_answer is not None else False
        
        print(f"\nDeterministic Answer: {det_answer}")
        print(f"Deterministic Correct: {det_correct}")
        print(f"Deterministic Completion: {det_completion[:100]}...")
        
        # Method B: Majority Vote
        maj_answer, all_answers, all_completions = self.solve_with_majority_vote(problem_data)
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
                "answer_distribution": dict(Counter(all_answers))
            }
        }
    
    def run_demo_evaluation(self):
        """Run demo evaluation on sample problems"""
        problems = self.get_demo_problems()
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
        print("DEMO RESULTS")
        print(f"{'='*60}")
        print(f"Total Problems: {len(problems)}")
        print(f"Deterministic Accuracy: {det_correct}/{len(problems)} ({det_correct/len(problems)*100:.1f}%)")
        print(f"Majority Vote Accuracy: {maj_correct}/{len(problems)} ({maj_correct/len(problems)*100:.1f}%)")
        
        # Save results
        with open('demo_results.json', 'w') as f:
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
        
        print(f"\nDetailed results saved to demo_results.json")
        
        return results

def main():
    """Main function to run the demo thinking mode application"""
    print("Demo Thinking Mode Application")
    print("==============================")
    print("This demo uses mock AI responses to demonstrate the concept")
    print("without requiring an OpenAI API key.\n")
    
    app = MockThinkingModeApp()
    
    # Run demo evaluation
    try:
        results = app.run_demo_evaluation()
        print("\nDemo evaluation completed successfully!")
        print("\nKey Concepts Demonstrated:")
        print("1. Chain-of-thought reasoning with 'Let's think step-by-step'")
        print("2. Multiple completions with temperature=1.1 for variation")
        print("3. Answer extraction using regex patterns")
        print("4. Majority voting for self-consistency")
        print("5. Comparison between deterministic vs majority vote approaches")
        
    except Exception as e:
        print(f"Error during demo evaluation: {e}")

if __name__ == "__main__":
    main()