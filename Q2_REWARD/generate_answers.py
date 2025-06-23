"""
Generate candidate answers for a small set of prompts and store them in a CSV file
with human-defined (or heuristic) rankings.

Usage
-----
$ python generate_answers.py --csv_path answers.csv

The script will create `answers.csv` in the same folder with the following schema:
    prompt,answer,rank

You may later edit the CSV manually to adjust the ranks so that they reflect your
personal preferences.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# Five diverse prompts we want the model to answer
PROMPTS = [
    "Tell me a funny programming joke about Python. Keep it short and clean.",
    "In 2-3 sentences, summarize the key idea behind the theory of evolution.",
    "Write a one-line motivational quote for students preparing for exams.",
    "Explain quantum entanglement in one simple sentence.",
    "Compose a 2-3 sentence friendly email asking for feedback on a recent project."
]

# Using T5 for more controlled text generation
MODEL_NAME = "t5-small"  # small and fast model that works well for short text
NUM_ANSWERS = 4
MAX_LENGTH = 64  # tokens, not characters


def generate_candidates(prompt: str, n: int, generator) -> list[str]:
    """Generate *n* answers for *prompt* using *generator*."""
    # Format for T5
    inputs = [f"answer: {prompt}"] * n
    
    outputs = generator(
        inputs,
        max_length=MAX_LENGTH,
        num_return_sequences=1,
        do_sample=True,
        top_k=20,
        top_p=0.95,
        temperature=0.7,
        no_repeat_ngram_size=1,
        truncation=True,
        num_beams=n,
        early_stopping=True
    )
    
    # Extract just the generated text
    answers = [out["generated_text"].strip() for out in outputs]
    return answers


def heuristic_rank(answers: list[str]) -> list[int]:
    """A very naive ranking function: shorter answers get slightly lower rank
    as they may be less informative.
    Rank 1 = best, 4 = worst.
    """
    # Sort by length descending (longer = presumably more detailed = better)
    sorted_idx = sorted(range(len(answers)), key=lambda i: len(answers[i]), reverse=True)
    ranks = [0] * len(answers)
    for rank, idx in enumerate(sorted_idx, start=1):
        ranks[idx] = rank
    return ranks


def main(csv_path: Path):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    generator = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
    )

    all_rows = []
    for prompt in PROMPTS:
        answers = generate_candidates(prompt, NUM_ANSWERS, generator)
        ranks = heuristic_rank(answers)
        for answer, rank in zip(answers, ranks):
            all_rows.append({"prompt": prompt, "answer": answer, "rank": rank})

    df = pd.DataFrame(all_rows)
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} rows to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path",
        type=Path,
        default=Path(__file__).parent / "answers.csv",
        help="Where to write the answers CSV",
    )
    args = parser.parse_args()
    main(args.csv_path)
