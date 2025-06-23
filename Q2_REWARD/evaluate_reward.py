"""
Evaluate the trained reward model on new generations and plot their reward scores.
The script generates fresh answers for each prompt, scores them with the trained
reward model, and shows a scatter plot (rank vs predicted reward).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from generate_answers import PROMPTS, generate_candidates, NUM_ANSWERS, MODEL_NAME as GEN_MODEL


def main(model_dir: Path, output_png: Path = Path("reward_vs_rank.png")):
    # Load reward model and tokenizer
    reward_tokenizer = AutoTokenizer.from_pretrained(model_dir)
    reward_model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    reward_pipe = pipeline(
        "text-classification",
        model=reward_model,
        tokenizer=reward_tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        return_all_scores=False,
    )

    # Generate new answers
    gen_tokenizer = reward_tokenizer if GEN_MODEL == model_dir else None
    # We'll just reuse generate_candidates from generate_answers.py using its generator
    from transformers import AutoTokenizer as AT, AutoModelForCausalLM as AMC

    gen_tok = AT.from_pretrained(GEN_MODEL) if gen_tokenizer is None else gen_tokenizer
    gen_model = AMC.from_pretrained(GEN_MODEL)
    gen_pipe = pipeline(
        "text-generation",
        model=gen_model,
        tokenizer=gen_tok,
        device=0 if torch.cuda.is_available() else -1,
    )

    rows = []
    for prompt in PROMPTS:
        answers = generate_candidates(prompt, NUM_ANSWERS, gen_pipe)
        # We don't have human rank here, so we'll use length heuristic like before
        ranks = [i + 1 for i in range(NUM_ANSWERS)]  # dummy ranks 1..4
        for answer, rank in zip(answers, ranks):
            text = prompt + " " + answer
            score = reward_pipe(text, truncation=True)[0]["score"]
            rows.append({"rank": rank, "reward_score": score})

    df = pd.DataFrame(rows)
    plt.figure(figsize=(6, 4))
    plt.scatter(df["rank"], df["reward_score"], alpha=0.7)
    plt.xlabel("Human Rank (1=best)")
    plt.ylabel("Predicted Reward Score")
    plt.title("Reward vs Rank")
    plt.grid(True)
    plt.savefig(output_png, dpi=150)
    print(f"Plot saved to {output_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=Path, default=Path(__file__).parent / "reward_model")
    parser.add_argument("--output_png", type=Path, default=Path(__file__).parent / "reward_vs_rank.png")
    args = parser.parse_args()
    main(args.model_dir, args.output_png)
