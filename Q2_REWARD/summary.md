# Reward Modelling Pipeline

This directory contains everything required to build, train and evaluate a simple
reward model that captures **your** preferences.

```
Q2_REWARD/
├── answers.csv          # prompt / answer / rank triples (generated or edited)
├── generate_answers.py  # create answers.csv with 5×4 samples
├── train_reward.py      # fine-tune a small DistilBERT reward model (50–100 steps)
├── reward_model/        # output directory produced by training
├── evaluate_reward.py   # run new generations and plot reward scores
└── summary.md           # this file
```

## 1. Generate candidate answers
```bash
python generate_answers.py  # creates answers.csv
```
Inspect **answers.csv** and adjust the *rank* column so that `1 = best` and `4 = worst`
according to your personal taste.

## 2. Train the reward model
```bash
python train_reward.py --steps 80  # 50-100 steps recommended
```
The trained model is saved to `reward_model/`.

## 3. Evaluate
```bash
python evaluate_reward.py
```
A scatter plot `reward_vs_rank.png` will be produced.  Properly trained models
should assign **higher scores to lower rank numbers (better answers)**, showing a
clear downward trend.
