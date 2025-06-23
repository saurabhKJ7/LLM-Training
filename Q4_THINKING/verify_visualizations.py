#!/usr/bin/env python3
"""
Verification script to check if generated visualizations match the actual data
"""

import json
import os
from PIL import Image

def load_results():
    """Load results from JSON file"""
    with open('results.json', 'r') as f:
        return json.load(f)

def verify_images_exist():
    """Verify all expected images exist"""
    expected_files = [
        'accuracy_comparison.png',
        'answer_distributions.png',
        'problem_1_details.png',
        'problem_2_details.png',
        'problem_3_details.png',
        'problem_4_details.png',
        'problem_5_details.png',
        'summary_stats.png'
    ]
    
    missing_files = []
    for file in expected_files:
        path = os.path.join('visualizations', file)
        if not os.path.exists(path):
            missing_files.append(file)
    
    return missing_files

def verify_image_quality():
    """Verify images are of correct quality"""
    image_files = [f for f in os.listdir('visualizations') if f.endswith('.png')]
    
    quality_issues = []
    for file in image_files:
        path = os.path.join('visualizations', file)
        with Image.open(path) as img:
            width, height = img.size
            dpi = img.info.get('dpi', (0, 0))
            
            # Check minimum size and DPI
            if width < 800 or height < 600:
                quality_issues.append(f"{file}: Size too small ({width}x{height})")
            if dpi[0] < 200:
                quality_issues.append(f"{file}: DPI too low ({dpi[0]})")
    
    return quality_issues

def verify_data_matches():
    """Verify visualization data matches JSON results"""
    results = load_results()
    issues = []
    
    # Check summary stats
    summary = results['summary']
    if summary['deterministic_accuracy'] != 1.0 or summary['majority_vote_accuracy'] != 1.0:
        issues.append("Accuracy visualization may not match data (expected 100% for both methods)")
    
    # Check problem details
    for i, problem in enumerate(results['detailed_results'], 1):
        maj_vote = problem['majority_vote']
        det = problem['deterministic']
        
        if maj_vote['answer'] != det['answer']:
            issues.append(f"Problem {i}: Answers don't match between methods")
            
        # Verify distribution matches data
        all_answers = maj_vote['all_answers']
        unique_answers = len(set(all_answers))
        if unique_answers > 1:
            issues.append(f"Problem {i}: Shows variation in answers that should be visualized")
    
    return issues

def main():
    print("Verifying Visualizations...")
    print("-" * 50)
    
    # Check if images exist
    print("\nChecking for missing files:")
    missing = verify_images_exist()
    if missing:
        print("❌ Missing files:")
        for file in missing:
            print(f"  - {file}")
    else:
        print("✓ All expected files exist")
    
    # Check image quality
    print("\nChecking image quality:")
    quality_issues = verify_image_quality()
    if quality_issues:
        print("❌ Quality issues found:")
        for issue in quality_issues:
            print(f"  - {issue}")
    else:
        print("✓ All images meet quality standards")
    
    # Check data consistency
    print("\nVerifying data consistency:")
    data_issues = verify_data_matches()
    if data_issues:
        print("❌ Data consistency issues:")
        for issue in data_issues:
            print(f"  - {issue}")
    else:
        print("✓ Visualizations match the data")
    
    # Summary
    print("\nVerification Summary:")
    print("-" * 50)
    if not (missing or quality_issues or data_issues):
        print("✅ All visualizations are correct and match the data!")
    else:
        print("❌ Some issues were found. Please check the details above.")

if __name__ == "__main__":
    main()