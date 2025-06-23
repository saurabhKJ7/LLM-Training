# Thinking Mode Application - Visualizations

## Overview of Generated Visualizations

### 1. Accuracy Comparison (`accuracy_comparison.png`)
- Bar chart comparing deterministic vs majority vote accuracy
- Shows percentage accuracy for both methods
- Higher bars indicate better performance

### 2. Answer Distribution Overview (`answer_distributions.png`)
- Pie charts showing answer distribution for each problem
- Each slice represents a different answer
- Size of slice indicates frequency of that answer
- Percentage labels show proportion of each answer

### 3. Problem-Specific Details (`problem_*_details.png`)
Individual analysis for each problem:
- `problem_1_details.png`: Apples and Oranges Cost
- `problem_2_details.png`: Train Distance
- `problem_3_details.png`: Marbles Distribution
- `problem_4_details.png`: Garden Perimeter
- `problem_5_details.png`: Pizza and Burger Preferences

Each visualization includes:
- Bar chart of answer frequencies
- Red dotted line showing correct answer
- Value labels showing exact counts

### 4. Summary Statistics (`summary_stats.png`)
Horizontal bar chart showing:
- Total number of problems evaluated
- Number of correct answers from deterministic method
- Number of correct answers from majority vote method

## Reading the Visualizations

### Understanding Color Coding
- Blue bars: Deterministic results
- Green bars: Majority vote results
- Red lines: Correct answers
- Multiple colors in pie charts: Different answer options

### Interpreting Results
- Look for consensus in pie charts
- Compare bar heights in accuracy comparison
- Check alignment with red lines in problem details
- Note discrepancies between methods

## Using These Visualizations

### For Analysis
- Identify problems where majority voting helped
- Spot patterns in answer distributions
- Compare method effectiveness
- Find problematic questions

### For Reporting
- Use accuracy comparison for overall performance
- Reference problem details for specific examples
- Include summary stats for quick overview
- Show answer distributions for reliability analysis

## Generated Files Summary

```
visualizations/
├── accuracy_comparison.png
├── answer_distributions.png
├── problem_1_details.png
├── problem_2_details.png
├── problem_3_details.png
├── problem_4_details.png
├── problem_5_details.png
└── summary_stats.png
```

## Updating Visualizations

To regenerate these visualizations:
1. Run the main application: `python thinking_mode.py`
2. Run visualization script: `python visualize_results.py`
3. New images will replace existing ones in this directory