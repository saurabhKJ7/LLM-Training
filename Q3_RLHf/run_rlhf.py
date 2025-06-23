#!/usr/bin/env python3
"""
Simplified RLHF Runner Script
Run this script to execute the complete RLHF pipeline
"""

import sys
import os
sys.path.append('src')

from rlhf_trainer import PPOTrainer
import pandas as pd
from datetime import datetime

def print_banner():
    print("="*60)
    print("    RLHF (PPO-style) Training Pipeline")
    print("    Goal: Make model refuse unsafe instructions")
    print("="*60)

def create_results_table(results_df):
    """Create a clean 6x2 results table"""
    
    # Create the comparison table
    table_data = []
    
    for _, row in results_df.iterrows():
        prompt_desc = f"{row['prompt_type'].title()} {row['prompt_id']}"
        
        # Truncate responses for table display
        base_resp = row['base_model_response'][:50] + "..." if len(row['base_model_response']) > 50 else row['base_model_response']
        trained_resp = row['trained_model_response'][:50] + "..." if len(row['trained_model_response']) > 50 else row['trained_model_response']
        
        table_data.append({
            'Prompt': prompt_desc,
            'Base Model Output': base_resp,
            'Trained Model Output': trained_resp,
            'Base Reward': row['base_model_reward'],
            'Trained Reward': row['trained_model_reward']
        })
    
    return pd.DataFrame(table_data)

def main():
    print_banner()
    
    try:
        # Create results directory
        os.makedirs("results", exist_ok=True)
        
        print("\nInitializing RLHF trainer...")
        trainer = PPOTrainer()
        
        print("\n" + "="*40)
        print("PHASE 1: PPO TRAINING (200 updates)")
        print("="*40)
        
        # Train the model
        trainer.train(num_updates=200)
        
        print("\n" + "="*40)
        print("PHASE 2: EVALUATION")
        print("="*40)
        
        # Evaluate both models
        results_df = trainer.evaluate_models()
        
        # Create and display the results table
        table_df = create_results_table(results_df)
        
        print("\n📊 RESULTS TABLE (6x2 Comparison):")
        print("-" * 100)
        for i, row in table_df.iterrows():
            print(f"{row['Prompt']:<12} | Base: {row['Base Reward']:>2} | Trained: {row['Trained Reward']:>2}")
        
        print("-" * 100)
        
        # Calculate summary statistics
        base_correct = len(results_df[results_df['base_model_reward'] == 1.0])
        trained_correct = len(results_df[results_df['trained_model_reward'] == 1.0])
        
        print(f"\n📈 SUMMARY:")
        print(f"   Base Model:    {base_correct}/6 correct responses")
        print(f"   Trained Model: {trained_correct}/6 correct responses")
        print(f"   Improvement:   {trained_correct - base_correct} additional correct responses")
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"results/detailed_results_{timestamp}.csv"
        table_file = f"results/summary_table_{timestamp}.csv"
        
        results_df.to_csv(results_file, index=False)
        table_df.to_csv(table_file, index=False)
        
        print(f"\n💾 Results saved:")
        print(f"   Detailed: {results_file}")
        print(f"   Summary:  {table_file}")
        
        # Show some example outputs
        print(f"\n🔍 EXAMPLE OUTPUTS:")
        print("-" * 80)
        
        # Show one benign example
        benign_example = results_df[results_df['prompt_type'] == 'benign'].iloc[0]
        print(f"BENIGN PROMPT: {benign_example['prompt']}")
        print(f"Base Response: {benign_example['base_model_response'][:100]}...")
        print(f"Trained Response: {benign_example['trained_model_response'][:100]}...")
        print()
        
        # Show one disallowed example
        disallowed_example = results_df[results_df['prompt_type'] == 'disallowed'].iloc[0]
        print(f"UNSAFE PROMPT: {disallowed_example['prompt']}")
        print(f"Base Response: {disallowed_example['base_model_response'][:100]}...")
        print(f"Trained Response: {disallowed_example['trained_model_response'][:100]}...")
        
        print("\n✅ RLHF Pipeline completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        print("Please check your environment and dependencies.")
        sys.exit(1)

if __name__ == "__main__":
    main()