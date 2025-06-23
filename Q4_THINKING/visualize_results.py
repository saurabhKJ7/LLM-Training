import matplotlib.pyplot as plt
import json
import numpy as np
import seaborn as sns
from pathlib import Path

class ResultsVisualizer:
    def __init__(self, results_file='results.json'):
        self.results_file = results_file
        self.output_dir = Path('visualizations')
        self.output_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        
    def load_results(self):
        """Load results from JSON file"""
        with open(self.results_file, 'r') as f:
            return json.load(f)
        
    def create_accuracy_comparison(self, results):
        """Create bar chart comparing deterministic vs majority vote accuracy"""
        plt.figure(figsize=(10, 6))
        
        # Data
        methods = ['Deterministic', 'Majority Vote']
        accuracies = [
            results['summary']['deterministic_accuracy'] * 100,
            results['summary']['majority_vote_accuracy'] * 100
        ]
        
        # Create bars
        bars = plt.bar(methods, accuracies, color=['#66B2FF', '#99FF99'])
        
        # Customize plot
        plt.title('Accuracy Comparison: Deterministic vs Majority Vote', pad=20)
        plt.ylabel('Accuracy (%)')
        plt.ylim(0, 100)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom')
        
        # Save plot
        plt.savefig(self.output_dir / 'accuracy_comparison.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()
        
    def create_answer_distributions(self, results):
        """Create pie charts showing answer distribution for each problem"""
        detailed_results = results['detailed_results']
        num_problems = len(detailed_results)
        
        # Create subplot grid
        fig = plt.figure(figsize=(15, 3 * ((num_problems + 2) // 3)))
        
        for i, problem in enumerate(detailed_results, 1):
            ax = plt.subplot(((num_problems + 2) // 3), 3, i)
            
            # Get answer distribution
            distribution = problem['majority_vote'].get('answer_distribution', {})
            answers = list(distribution.keys())
            counts = list(distribution.values())
            
            # Create pie chart
            colors = plt.cm.Set3(np.linspace(0, 1, len(answers)))
            wedges, texts, autotexts = plt.pie(counts, 
                   labels=[f'{float(a):.1f}' for a in answers],
                   autopct='%1.1f%%', colors=colors,
                   explode=[0.1 if c != max(counts) else 0 for c in counts])
            
            # Add title (truncated if too long)
            question = problem['question']
            if len(question) > 50:
                question = question[:47] + "..."
            plt.title(f"Problem {i}\n{question}", pad=20, wrap=True)
            
            # Enhance text visibility
            plt.setp(autotexts, size=8, weight="bold")
            plt.setp(texts, size=8)
            
        plt.tight_layout(pad=3.0)
        plt.savefig(self.output_dir / 'answer_distributions.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()
        
    def create_problem_details(self, results):
        """Create detailed visualization for each problem"""
        detailed_results = results['detailed_results']
        
        for i, problem in enumerate(detailed_results, 1):
            plt.figure(figsize=(12, 6))
            
            # Get answer frequencies
            distribution = problem['majority_vote'].get('answer_distribution', {})
            answers = list(distribution.keys())
            counts = list(distribution.values())
            
            # Create bar chart with enhanced styling
            bars = plt.bar([f'{float(a):.1f}' for a in answers], counts,
                          color=plt.cm.Set3(np.linspace(0, 1, len(answers))),
                          edgecolor='black', linewidth=1,
                          alpha=0.8)
            
            # Add value labels with improved visibility
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)} answers\n({height/sum(counts)*100:.1f}%)',
                        ha='center', va='bottom',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            
            # Customize plot with enhanced styling
            plt.title(f'Answer Distribution - Problem {i}\n{problem["question"][:50]}...', 
                     pad=20, wrap=True)
            plt.ylabel('Number of Responses')
            plt.xlabel('Generated Answers')
            
            # Add correct answer marker with improved visibility
            correct_answer = problem['correct_answer']
            correct_x = [i for i, a in enumerate([float(x) for x in answers]) 
                        if abs(a - correct_answer) < 0.01]
            if correct_x:
                plt.axvline(x=correct_x[0], color='green', linestyle='-', linewidth=2,
                          label=f'Correct Answer: {correct_answer}')
                
            # Add majority vote marker
            majority_answer = problem['majority_vote']['answer']
            majority_x = [i for i, a in enumerate([float(x) for x in answers]) 
                         if abs(a - majority_answer) < 0.01]
            if majority_x:
                plt.axvline(x=majority_x[0], color='blue', linestyle='--', linewidth=2,
                          label=f'Majority Vote: {majority_answer}')
            
            plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
            plt.grid(True, alpha=0.3)
            
            # Save plot
            plt.savefig(self.output_dir / f'problem_{i}_details.png', 
                       bbox_inches='tight', dpi=300)
            plt.close()
    
    def create_summary_stats(self, results):
        """Create summary statistics visualization"""
        plt.figure(figsize=(12, 6))
        
        # Prepare data
        data = {
            'Total Problems': results['summary']['total_problems'],
            'Deterministic Correct': results['summary']['deterministic_correct'],
            'Majority Vote Correct': results['summary']['majority_vote_correct']
        }
        
        # Create horizontal bars
        y_pos = np.arange(len(data))
        bars = plt.barh(y_pos, list(data.values()), 
                       color=['#FF9999', '#66B2FF', '#99FF99'])
        
        # Customize plot
        plt.yticks(y_pos, list(data.keys()))
        plt.xlabel('Count')
        plt.title('Summary Statistics', pad=20)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{int(width)}',
                    ha='left', va='center', color='black')
        
        # Save plot
        plt.savefig(self.output_dir / 'summary_stats.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()
    
    def generate_all_visualizations(self):
        """Generate all visualizations"""
        print("Loading results...")
        results = self.load_results()
        
        print("Creating visualizations...")
        self.create_accuracy_comparison(results)
        self.create_answer_distributions(results)
        self.create_problem_details(results)
        self.create_summary_stats(results)
        
        print(f"\nVisualizations saved in: {self.output_dir}/")
        print("Generated files:")
        print("- accuracy_comparison.png")
        print("- answer_distributions.png")
        print("- problem_*_details.png")
        print("- summary_stats.png")

def main():
    visualizer = ResultsVisualizer()
    visualizer.generate_all_visualizations()

if __name__ == "__main__":
    main()